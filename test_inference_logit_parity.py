import importlib
from types import SimpleNamespace

import torch
import torch.nn as nn

from hierarchos import HierarchosCore
from hierarchos.evaluation.arc_agi import generate_text as generate_arc_text
from hierarchos.evaluation.lm_eval_wrapper import HierarchosLM
from hierarchos.inference.chat import (
    advance_chat_model_state,
    boundary_drift_seed,
    resolve_inference_prefill_chunk_size,
    uses_full_sample_inference_recurrence,
)
from hierarchos.models.core import WorkerLoop
from hierarchos.models.quantized import _uses_exact_refinement_policy
from hierarchos.utils.checkpoint import (
    load_full_model_with_config,
    sanitize_model_state_dict,
)
from hierarchos.utils.rosa import precompute_rosa_ids_for_chunks
from test_rwkv_v8_integrity import _tiny_config


def _full_sample_config():
    config = _tiny_config()
    config.full_sample_bptt = True
    config.training_chunk_size = 3
    config.compile_static_worker_loop = True
    config.memory_gate_warmup_steps = 0
    config.detach_every_n_steps = None
    config.max_h_steps = 5
    config.max_l_steps = 4
    config.h_halt_thresh = 0.4
    return config


def test_full_sample_checkpoint_defaults_to_one_chat_prefill_graph():
    exact = SimpleNamespace(full_sample_bptt=True, training_chunk_size=256)
    parity = SimpleNamespace(inference_logit_parity=True, training_chunk_size=256)
    legacy = SimpleNamespace(full_sample_bptt=False, training_chunk_size=256)
    exported_tbptt = {
        "full_sample_bptt": False,
        "inference_logit_parity": True,
        "training_chunk_size": 256,
    }
    resumed_tbptt = SimpleNamespace(
        full_sample_bptt=False,
        inference_logit_parity=True,
        inference_recurrence_mode="tbptt",
        training_chunk_size=256,
    )
    explicit_full_sample = {
        "full_sample_bptt": False,
        "inference_logit_parity": False,
        "inference_recurrence_mode": "full-sample",
        "training_chunk_size": 256,
    }

    assert resolve_inference_prefill_chunk_size(exact) == 0
    # Backward compatibility: old parity checkpoints without explicit geometry
    # retain their historical full-sample inference behavior.
    assert resolve_inference_prefill_chunk_size(parity) == 0
    assert resolve_inference_prefill_chunk_size(legacy) == 256
    assert resolve_inference_prefill_chunk_size(exported_tbptt) == 256
    assert resolve_inference_prefill_chunk_size(resumed_tbptt) == 256
    assert resolve_inference_prefill_chunk_size(explicit_full_sample) == 0
    assert uses_full_sample_inference_recurrence(parity)
    assert not uses_full_sample_inference_recurrence(exported_tbptt)
    assert not uses_full_sample_inference_recurrence(resumed_tbptt)
    assert uses_full_sample_inference_recurrence(explicit_full_sample)
    assert resolve_inference_prefill_chunk_size(exact, requested=128) == 128
    drift = torch.ones(1, 4)
    assert boundary_drift_seed(drift, 128, 128, exact_full_sample=True) is None
    assert boundary_drift_seed(drift, 128, 128) is drift


def test_corrected_rosa_sentinel_is_zero_masked_before_memory_routing():
    torch.manual_seed(15)
    config = _tiny_config()
    config.use_deepembed = False
    config.memory_token_routers = False
    config.rosa_zero_no_prediction = False
    model = HierarchosCore(config).eval()
    with torch.no_grad():
        model.rosa_emb.weight[config.vocab_size].fill_(3.0)
        model.rosa_gate_logit.fill_(20.0)

    captured = []

    def capture_query_input(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.qproj.register_forward_pre_hook(capture_query_input)
    try:
        with torch.no_grad():
            model(torch.tensor([[11]], dtype=torch.long), suppress_hebbian=True)
        legacy_token_input = captured[-1][..., :config.context_dim]
        captured.clear()

        config.rosa_zero_no_prediction = True
        with torch.no_grad():
            model(torch.tensor([[11]], dtype=torch.long), suppress_hebbian=True)
        corrected_token_input = captured[-1][..., :config.context_dim]
    finally:
        handle.remove()

    expected_token_input = model.tok_emb(torch.tensor([11]))
    assert not torch.allclose(legacy_token_input, expected_token_input)
    torch.testing.assert_close(corrected_token_input, expected_token_input)


def test_bounded_cached_and_live_rosa_keep_logits_and_continuation_state_aligned():
    torch.manual_seed(16)
    config = _tiny_config()
    config.use_deepembed = False
    config.memory_gate_warmup_steps = 0
    config.enforce_rosa_max_context = True
    config.rosa_max_context = 4
    model = HierarchosCore(config).eval()
    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4, 1, 2]], dtype=torch.long)
    cached_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=config.vocab_size,
                chunk_size=3,
                rosa_max_ctx=config.rosa_max_context,
                enforce_max_context=True,
            )
        ],
        dtype=torch.long,
    )

    def run(use_cache):
        state = None
        logits = []
        for start in range(0, input_ids.shape[1], 3):
            end = min(start + 3, input_ids.shape[1])
            kwargs = {}
            if use_cache:
                kwargs["rosa_ids"] = cached_ids[:, start:end]
                kwargs["rosa_ids_context_mode"] = "bounded-segment-v1"
            outputs = model(
                input_ids[:, start:end],
                ltm_memory_state=state,
                global_pos_offset=start,
                suppress_hebbian=True,
                **kwargs,
            )
            state = outputs["ltm_memory_state"]
            logits.append(outputs["logits"])
            if use_cache:
                assert state[2].shape[1] <= config.rosa_max_context
                assert state[3] is None
            else:
                assert state[2] is None
                assert len(state[3][0].tokens) <= config.rosa_max_context
        return torch.cat(logits, dim=1), state

    with torch.no_grad():
        live_logits, live_state = run(False)
        cached_logits, cached_state = run(True)

    torch.testing.assert_close(cached_logits, live_logits, rtol=1e-6, atol=5e-7)
    assert cached_state[2][0].tolist() == live_state[3][0].tokens


def test_rosa_live_cached_live_transition_preserves_complete_history():
    torch.manual_seed(160)
    config = _tiny_config()
    config.use_deepembed = False
    config.memory_gate_warmup_steps = 0
    config.enforce_rosa_max_context = True
    config.rosa_max_context = 5
    model = HierarchosCore(config).eval()
    input_ids = torch.tensor(
        [[1, 2, 1, 2, 3, 1, 2, 4, 1]],
        dtype=torch.long,
    )
    cached_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=config.vocab_size,
                chunk_size=3,
                rosa_max_ctx=config.rosa_max_context,
                enforce_max_context=True,
            )
        ],
        dtype=torch.long,
    )

    def run(use_middle_cache):
        state = None
        logits = []
        for chunk_index, start in enumerate(range(0, input_ids.shape[1], 3)):
            end = min(start + 3, input_ids.shape[1])
            kwargs = {}
            if use_middle_cache and chunk_index == 1:
                kwargs = {
                    "rosa_ids": cached_ids[:, start:end],
                    "rosa_ids_context_mode": "bounded-segment-v1",
                }
            outputs = model(
                input_ids[:, start:end],
                ltm_memory_state=state,
                global_pos_offset=start,
                suppress_hebbian=True,
                **kwargs,
            )
            state = outputs["ltm_memory_state"]
            logits.append(outputs["logits"])
        return torch.cat(logits, dim=1), state

    with torch.no_grad():
        live_logits, live_state = run(False)
        mixed_logits, mixed_state = run(True)

    torch.testing.assert_close(mixed_logits, live_logits, rtol=1e-6, atol=5e-7)
    assert mixed_state[2] is None
    assert mixed_state[3][0].tokens == live_state[3][0].tokens


def test_explicit_chat_prefill_segments_preserve_full_sample_drift_recurrence():
    torch.manual_seed(17)
    config = _full_sample_config()
    model = HierarchosCore(config).eval()
    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        monolithic = model(input_ids, suppress_hebbian=True)["logits"]

    model.reset_memory()
    state = (None, None, None, None, None, None)
    segmented_logits = []
    segment_size = 3
    with torch.no_grad():
        for start in range(0, input_ids.shape[1], segment_size):
            end = min(start + segment_size, input_ids.shape[1])
            h_state, l_state, prev_context, target_context, drift_state, ltm_state = state
            outputs, state = advance_chat_model_state(
                model,
                input_ids[:, start:end],
                device=torch.device("cpu"),
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
                drift_seed=boundary_drift_seed(
                    drift_state,
                    start,
                    segment_size,
                    exact_full_sample=True,
                ),
                ltm_state=ltm_state,
                global_pos_offset=start,
            )
            segmented_logits.append(outputs["logits"])

    torch.testing.assert_close(
        torch.cat(segmented_logits, dim=1),
        monolithic,
        rtol=1e-6,
        atol=5e-7,
    )


def test_lm_eval_explicit_segments_preserve_exact_drift_recurrence():
    torch.manual_seed(19)
    model = HierarchosCore(_full_sample_config()).eval()
    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        expected = model(input_ids, suppress_hebbian=True)["logits"]

    # Bypass the optional lm-eval package constructor and force explicit
    # segmentation so this directly exercises the wrapper's boundary policy.
    wrapper = object.__new__(HierarchosLM)
    wrapper.model = model
    wrapper.device = torch.device("cpu")
    wrapper._prefill_chunk_size = 3
    model.reset_memory()
    actual = wrapper._model_call(input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=5e-7)


class _SurfaceTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, _text, add_special_tokens=False, return_tensors=None):
        ids = [1, 2, 1, 2, 3, 1, 2, 4]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in tokens)


class _RecordingExactInferenceModel:
    def __init__(self):
        self.config = SimpleNamespace(
            full_sample_bptt=True,
            inference_logit_parity=True,
            training_chunk_size=3,
            context_dim=4,
        )
        self.calls = []
        self.suppress_hebbian = True

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.detach().clone(), kwargs.get("drift_state")))
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device)
        next_id = 1 if len(self.calls) == 1 else 0
        logits[..., next_id] = 10.0
        return {
            "logits": logits,
            "h_state": kwargs.get("h_state"),
            "l_state": kwargs.get("l_state"),
            "prev_context": kwargs.get("prev_context"),
            "target_context": kwargs.get("target_context"),
            "drift_state": torch.ones(batch, 4),
            "ltm_memory_state": kwargs.get("ltm_memory_state"),
        }


class _RecordingTBPTTParityModel(_RecordingExactInferenceModel):
    def __init__(self):
        super().__init__()
        self.config.full_sample_bptt = False
        self.config.inference_logit_parity = True
        self.config.inference_recurrence_mode = "tbptt"


def test_lm_eval_tbptt_geometry_survives_refinement_logit_parity():
    model = _RecordingTBPTTParityModel()
    wrapper = object.__new__(HierarchosLM)
    wrapper.model = model
    wrapper.device = torch.device("cpu")
    wrapper._prefill_chunk_size = resolve_inference_prefill_chunk_size(model.config)

    wrapper._model_call(torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long))

    assert [call_ids.shape[1] for call_ids, _drift in model.calls] == [3, 3, 2]
    assert model.calls[0][1] is None
    assert all(drift is not None for _ids, drift in model.calls[1:])


def test_arc_tbptt_geometry_survives_refinement_logit_parity():
    model = _RecordingTBPTTParityModel()
    generate_arc_text(
        model,
        _SurfaceTokenizer(),
        torch.device("cpu"),
        "grid prompt",
        max_new_tokens=1,
        temperature=0.0,
    )

    assert [call_ids.shape[1] for call_ids, _drift in model.calls] == [3, 3, 2]
    assert model.calls[0][1] is None
    assert all(drift is not None for _ids, drift in model.calls[1:])


def test_arc_generation_ignores_cache_chunk_geometry_and_never_reseeds_exact_drift():
    model = _RecordingExactInferenceModel()
    generate_arc_text(
        model,
        _SurfaceTokenizer(),
        torch.device("cpu"),
        "grid prompt",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert len(model.calls) == 2
    assert model.calls[0][0].shape[1] == 8
    assert all(drift_seed is None for _ids, drift_seed in model.calls)


def test_gui_bridge_uses_exact_prefill_geometry_and_no_external_drift_seed(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    model = _RecordingExactInferenceModel()
    emitted = []

    class _ImmediateThread:
        def __init__(self, *, target, daemon=None):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_tokenizer", _SurfaceTokenizer())
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(vars(model.config)))
    monkeypatch.setattr(bridge, "_h_state", None)
    monkeypatch.setattr(bridge, "_l_state", None)
    monkeypatch.setattr(bridge, "_prev_context", None)
    monkeypatch.setattr(bridge, "_target_context", None)
    monkeypatch.setattr(bridge, "_drift_state", torch.full((1, 4), 7.0))
    monkeypatch.setattr(bridge, "_ltm_state", None)
    monkeypatch.setattr(bridge, "_total_tokens_generated", 3)
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)

    bridge.handle_generate(
        {
            "message": "hello",
            "sampling": {"max_new_tokens": 0, "temperature": 0.0},
        }
    )

    assert not [event for event, _data in emitted if event == "error"]
    assert len(model.calls) == 1
    assert model.calls[0][0].shape[1] == 8
    assert model.calls[0][1] is None


def test_gui_bridge_uses_tbptt_geometry_with_refinement_parity_enabled(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    model = _RecordingTBPTTParityModel()
    emitted = []

    class _ImmediateThread:
        def __init__(self, *, target, daemon=None):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_tokenizer", _SurfaceTokenizer())
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(vars(model.config)))
    monkeypatch.setattr(bridge, "_h_state", None)
    monkeypatch.setattr(bridge, "_l_state", None)
    monkeypatch.setattr(bridge, "_prev_context", None)
    monkeypatch.setattr(bridge, "_target_context", None)
    monkeypatch.setattr(bridge, "_drift_state", None)
    monkeypatch.setattr(bridge, "_ltm_state", None)
    monkeypatch.setattr(bridge, "_total_tokens_generated", 0)
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)

    bridge.handle_generate(
        {
            "message": "hello",
            "sampling": {"max_new_tokens": 0, "temperature": 0.0},
        }
    )

    assert not [event for event, _data in emitted if event == "error"]
    assert [call_ids.shape[1] for call_ids, _drift in model.calls] == [3, 3, 2]
    assert model.calls[0][1] is None
    assert all(drift is not None for _ids, drift in model.calls[1:])


def test_quantized_exact_checkpoint_uses_training_refinement_policy_flag():
    assert _uses_exact_refinement_policy(
        SimpleNamespace(full_sample_bptt=True, inference_logit_parity=False)
    )
    assert _uses_exact_refinement_policy(
        {"full_sample_bptt": False, "inference_logit_parity": True}
    )
    assert not _uses_exact_refinement_policy(
        SimpleNamespace(full_sample_bptt=False, inference_logit_parity=False)
    )


class _StateStableRNN(nn.Module):
    def forward(self, x, state, timestep=None, deepemb_vec=None):
        return x, state


def test_static_worker_convergence_is_independent_per_batch_row():
    config = SimpleNamespace(
        max_l_steps=3,
        l_conv_atol=0.2,
        commitment_threshold=0.0,
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        drift_delta_scale=1.0,
        activation_clamp=100.0,
        compile_static_worker_loop=True,
    )
    rnn = _StateStableRNN()
    input_projection = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        input_projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        )
    loop = WorkerLoop(
        config,
        rnn,
        input_projection,
        nn.Identity(),
        nn.Identity(),
    )
    rnn.train()

    row = torch.tensor([[0.05, 0.05]])
    peer = torch.tensor([[1.0, 1.0]])
    single = loop(
        row,
        torch.zeros_like(row),
        torch.zeros(1, 2, 3),
        torch.zeros_like(row),
    )
    paired = loop(
        torch.cat([row, peer], dim=0),
        torch.zeros(2, 2),
        torch.zeros(2, 2, 3),
        torch.zeros(2, 2),
    )

    for single_value, paired_value in zip(single, paired):
        torch.testing.assert_close(single_value[0], paired_value[0], rtol=0, atol=0)


def test_full_sample_worker_eval_uses_the_training_refinement_policy():
    config = SimpleNamespace(
        max_l_steps=3,
        l_conv_atol=0.01,
        commitment_threshold=0.1,
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        drift_delta_scale=1.0,
        activation_clamp=100.0,
        compile_static_worker_loop=True,
        full_sample_bptt=True,
    )
    rnn = _StateStableRNN()
    input_projection = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        input_projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        )
    loop = WorkerLoop(
        config,
        rnn,
        input_projection,
        nn.Identity(),
        nn.Identity(),
    )
    inputs = (
        torch.tensor([[0.5, 0.25]]),
        torch.tensor([[0.1, 0.2]]),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2),
    )

    rnn.train()
    training_outputs = loop(*inputs)
    rnn.eval()
    inference_outputs = loop(*inputs)

    for inference_value, training_value in zip(inference_outputs, training_outputs):
        torch.testing.assert_close(inference_value, training_value, rtol=0, atol=0)


def test_cached_training_and_live_chat_rosa_have_exact_full_sample_logits():
    torch.manual_seed(3)
    config = _full_sample_config()
    model = HierarchosCore(config)
    with torch.no_grad():
        # This would activate the legacy inference-only manager early exit.
        model.h_halt_proj.weight.zero_()
        model.h_halt_proj.bias.zero_()

    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)
    cached_rosa_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=config.vocab_size,
                chunk_size=config.training_chunk_size,
                rosa_max_ctx=config.rosa_max_context,
            )
        ],
        dtype=torch.long,
    )

    model.train()
    model.reset_memory()
    training_logits = model(
        input_ids,
        labels=input_ids,
        rosa_ids=cached_rosa_ids,
        suppress_hebbian=True,
    )["logits"].detach()

    model.eval()
    model.reset_memory()
    with torch.no_grad():
        chat_logits = model(input_ids, suppress_hebbian=True)["logits"]

    torch.testing.assert_close(chat_logits, training_logits, rtol=0, atol=0)


def test_tokenwise_chat_matches_cached_tbptt_training_logits():
    torch.manual_seed(23)
    config = _full_sample_config()
    config.full_sample_bptt = False
    config.inference_recurrence_mode = "tbptt"
    config.inference_logit_parity = True
    config.training_chunk_size = 3
    model = HierarchosCore(config)
    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)
    cached_rosa_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=config.vocab_size,
                chunk_size=config.training_chunk_size,
                rosa_max_ctx=config.rosa_max_context,
            )
        ],
        dtype=torch.long,
    )

    model.train()
    model.reset_memory()
    training_state = (None, None, None, None, None, None)
    training_logits = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[1], config.training_chunk_size):
            end = min(start + config.training_chunk_size, input_ids.shape[1])
            h_state, l_state, prev_context, target_context, drift_state, ltm_state = training_state
            outputs = model(
                input_ids[:, start:end],
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
                ltm_memory_state=ltm_state,
                global_pos_offset=start,
                rosa_ids=cached_rosa_ids[:, start:end],
                suppress_hebbian=True,
            )
            training_logits.append(outputs["logits"])
            training_state = (
                outputs["h_state"],
                outputs["l_state"],
                outputs["prev_context"],
                outputs["target_context"],
                outputs["drift_state"],
                outputs["ltm_memory_state"],
            )
    training_logits = torch.cat(training_logits, dim=1)

    model.eval()
    model.reset_memory()
    chat_state = (None, None, None, None, None, None)
    chat_logits = []
    with torch.no_grad():
        for position in range(input_ids.shape[1]):
            h_state, l_state, prev_context, target_context, drift_state, ltm_state = chat_state
            outputs, chat_state = advance_chat_model_state(
                model,
                input_ids[:, position:position + 1],
                device=torch.device("cpu"),
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
                drift_seed=boundary_drift_seed(
                    drift_state,
                    position,
                    config.training_chunk_size,
                ),
                ltm_state=ltm_state,
                global_pos_offset=position,
            )
            chat_logits.append(outputs["logits"])
    chat_logits = torch.cat(chat_logits, dim=1)

    torch.testing.assert_close(
        chat_logits,
        training_logits,
        rtol=1e-6,
        atol=5e-7,
    )


def test_autoregressive_chat_state_matches_the_full_sequence_next_logit():
    torch.manual_seed(4)
    config = _full_sample_config()
    model = HierarchosCore(config).eval()
    prompt_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)
    next_id = torch.tensor([[5]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        prefill = model(prompt_ids, suppress_hebbian=True)
        incremental, _ = advance_chat_model_state(
            model,
            next_id,
            device=torch.device("cpu"),
            h_state=prefill["h_state"],
            l_state=prefill["l_state"],
            prev_context=prefill["prev_context"],
            target_context=prefill["target_context"],
            drift_state=prefill["drift_state"],
            drift_seed=None,
            ltm_state=prefill["ltm_memory_state"],
            global_pos_offset=prompt_ids.shape[1],
        )

        model.reset_memory()
        monolithic = model(
            torch.cat([prompt_ids, next_id], dim=1),
            suppress_hebbian=True,
        )

    # Different GEMM shapes may round the last bit differently; the recurrence,
    # ROSA history, positions, and resulting next-token distribution must agree.
    torch.testing.assert_close(
        incremental["logits"][:, -1],
        monolithic["logits"][:, -1],
        rtol=1e-6,
        atol=5e-7,
    )


def test_inference_export_resets_transient_ltm_without_logit_drift(tmp_path):
    torch.manual_seed(9)
    config = _full_sample_config()
    model = HierarchosCore(config).eval()
    input_ids = torch.tensor([[7, 3, 7, 3, 9]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        expected_logits = model(input_ids, suppress_hebbian=True)["logits"].clone()
        model.ltm.fast_vals.fill_(4.0)
        model.ltm._mom_vals.fill_(3.0)
        model.ltm.timestamps.fill_(2.0)
        model.ltm.sources.fill_(1)

    torch.save(
        {
            "model_state_dict": sanitize_model_state_dict(model),
            "config": dict(config),
            "training_complete": True,
        },
        tmp_path / "hierarchos.pt",
    )
    loaded, loaded_config = load_full_model_with_config(str(tmp_path), "cpu")

    assert loaded_config.full_sample_bptt is True
    assert loaded_config.inference_recurrence_mode == "full-sample"
    assert torch.count_nonzero(loaded.ltm.fast_vals).item() == 0
    assert torch.count_nonzero(loaded.ltm._mom_vals).item() == 0
    with torch.no_grad():
        actual_logits = loaded(input_ids, suppress_hebbian=True)["logits"]
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)


def test_legacy_export_without_gate_step_loads_warmup_complete(tmp_path):
    torch.manual_seed(91)
    config = _tiny_config()
    config.memory_gate_warmup_steps = 100
    config.memory_gate_warmup_floor = 0.5
    model = HierarchosCore(config).eval()
    model.set_training_step(100)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        expected = model(ids, suppress_hebbian=True)["logits"]

    legacy_state = dict(model.state_dict())
    legacy_state.pop("memory_gate_warmup_step")
    torch.save(
        {
            "model_state_dict": legacy_state,
            "config": dict(config),
            "training_complete": True,
        },
        tmp_path / "hierarchos.pt",
    )

    loaded, _ = load_full_model_with_config(str(tmp_path), "cpu")
    assert loaded.memory_gate_warmup_step.item() == 100
    with torch.no_grad():
        actual = loaded(ids, suppress_hebbian=True)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_loader_backfills_explicit_tbptt_recurrence(tmp_path):
    config = _full_sample_config()
    config.full_sample_bptt = False
    config.inference_logit_parity = True
    config.pop("inference_recurrence_mode", None)
    model = HierarchosCore(config).eval()

    torch.save(
        {
            "model_state_dict": sanitize_model_state_dict(model),
            "config": dict(config),
            "training_complete": True,
        },
        tmp_path / "hierarchos.pt",
    )
    _loaded, loaded_config = load_full_model_with_config(str(tmp_path), "cpu")

    assert loaded_config.full_sample_bptt is False
    assert loaded_config.inference_logit_parity is True
    assert loaded_config.inference_recurrence_mode == "tbptt"
    assert resolve_inference_prefill_chunk_size(loaded_config) == config.training_chunk_size
