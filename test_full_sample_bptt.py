from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hierarchos import AttrDict, HierarchosCore
from hierarchos.training.trainer import (
    _checkpointed_training_model_call,
    compute_chunk_training_weights,
    configure_full_sample_bptt,
    train_step,
)
from hierarchos.utils.checkpoint import load_model_state_dict_compatible
from hierarchos.utils.rosa import precompute_rosa_ids_for_chunks


def _tiny_config(*, use_rosa=True, detach_every_n_steps=None):
    return AttrDict(
        vocab_size=64,
        context_dim=16,
        persistent_dim=8,
        ltm_slots=16,
        ltm_key_dim=8,
        ltm_val_dim=8,
        ltm_topk=2,
        h_hidden=16,
        l_hidden=16,
        max_h_steps=2,
        max_l_steps=2,
        h_stride=2,
        l_conv_atol=1e-4,
        commitment_threshold=0.05,
        use_deepembed=True,
        use_rosa=use_rosa,
        rosa_max_context=32,
        compile=False,
        gradient_checkpointing=False,
        detach_every_n_steps=detach_every_n_steps,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
    )


def _forward_kwargs(input_ids, labels, *, rosa_ids=None, raw_ltm=True):
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        ltm_memory_state=None,
        global_pos_offset=0,
        return_logits=True,
        return_topk_values=False,
        return_raw_topk_values=raw_ltm,
        return_topk_indices=raw_ltm,
        compute_ltm_value_alignment=False,
        rosa_ids=rosa_ids,
        loss_weights=None,
    )


def _backward_loss(outputs):
    loss = outputs["loss"]
    if outputs.get("ponder_cost") is not None:
        loss = loss + 0.01 * outputs["ponder_cost"]
    if outputs.get("commitment_cost") is not None:
        loss = loss + 0.5 * outputs["commitment_cost"]
    return loss


class _RecordingModel(nn.Module):
    def __init__(self, *, oom=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(
            vocab_size=64,
            cpu_chunked_lm_loss=False,
            cuda_chunked_lm_loss=False,
        )
        self.oom = oom
        self.seen = []
        self.reset_called = False

    def reset_memory(self):
        self.reset_called = True

    def forward(self, **kwargs):
        if self.oom:
            raise RuntimeError("CUDA out of memory in dummy forward")
        self.seen.append(
            (
                int(kwargs["global_pos_offset"]),
                kwargs["input_ids"].detach().clone(),
                kwargs["labels"].detach().clone(),
                bool(kwargs["return_raw_topk_values"]),
            )
        )
        return {
            "loss": self.weight,
            "ponder_cost": self.weight.new_zeros(()),
            "commitment_cost": self.weight.new_zeros(()),
            "ltm_value_alignment_cost": None,
            "raw_topk_vals": None,
            "topk_idx": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


def _step_args(**overrides):
    values = dict(
        amp=False,
        training_chunk_size=2,
        full_sample_bptt=True,
        full_sample_activation_checkpointing=False,
        persist_state=False,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode="read-only",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_without_optimizer_step(model, batch, args):
    """Run backward while retaining gradients for direct comparison."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=2,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )


def _assert_nested_close(actual, expected, *, rtol=2e-5, atol=2e-6):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_close(actual_item, expected_item, rtol=rtol, atol=atol)
        return
    assert actual == expected


def _assert_parameter_gradients_close(actual_model, expected_model):
    actual_parameters = dict(actual_model.named_parameters())
    expected_parameters = dict(expected_model.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, expected_parameter in expected_parameters.items():
        actual_grad = actual_parameters[name].grad
        expected_grad = expected_parameter.grad
        assert (actual_grad is None) == (expected_grad is None), name
        if expected_grad is not None:
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=2e-5,
                atol=2e-6,
                msg=lambda message: f"gradient mismatch for {name}: {message}",
            )


def test_full_sample_configuration_is_exact_and_cache_compatible():
    args = SimpleNamespace(
        full_sample_bptt=True,
        full_sample_activation_checkpointing=None,
        training_chunk_size=256,
        detach_every_n_steps=32,
        persist_state=True,
        gradient_checkpointing=True,
        ltm_training_mode="inner-update",
    )
    config = AttrDict(vars(args))

    assert configure_full_sample_bptt(args, config) is True

    assert args.training_chunk_size == 256
    assert args.detach_every_n_steps == 0
    assert args.persist_state is False
    assert args.full_sample_activation_checkpointing is True
    assert args.gradient_checkpointing is False
    assert args.ltm_training_mode == "read-only"
    assert config.training_chunk_size == 256
    assert config.detach_every_n_steps is None
    assert config.persist_state is False
    assert config.full_sample_bptt is True
    assert args.inference_recurrence_mode == "full-sample"
    assert config.inference_recurrence_mode == "full-sample"
    assert config.full_sample_activation_checkpointing is True
    assert config.ltm_training_mode == "read-only"


def test_tbptt_configuration_decouples_recurrence_from_refinement_parity():
    args = SimpleNamespace(
        full_sample_bptt=False,
        full_sample_activation_checkpointing=False,
        full_sample_checkpoint_segment_size=224,
        training_chunk_size=256,
        detach_every_n_steps=0,
        persist_state=False,
        ltm_training_mode="read-only",
    )
    config = AttrDict(vars(args))

    assert configure_full_sample_bptt(args, config) is False

    assert args.inference_recurrence_mode == "tbptt"
    assert config.inference_recurrence_mode == "tbptt"
    # Training-equivalent Manager/Worker refinement remains enabled even though
    # inference now follows the resumed 256-token TBPTT boundary geometry.
    assert args.inference_logit_parity is True
    assert config.inference_logit_parity is True
    assert config.full_sample_bptt is False


def test_full_sample_train_step_uses_one_trimmed_graph_despite_small_chunk_metadata():
    model = _RecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14, 0, 0]], dtype=torch.long),
        "labels": torch.tensor([[10, 11, 12, 13, 14, -100, -100]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.long),
    }

    outputs, _states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=_step_args(training_chunk_size=2),
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert len(model.seen) == 1
    offset, input_ids, labels, raw_ltm = model.seen[0]
    assert offset == 0
    assert input_ids.tolist() == [[10, 11, 12, 13, 14]]
    assert labels.tolist() == [[10, 11, 12, 13, 14]]
    assert raw_ltm is False


def test_full_sample_oom_fails_closed_instead_of_silently_truncating():
    model = _RecordingModel(oom=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="Refusing to silently truncate"):
        train_step(
            model,
            batch,
            optimizer,
            scaler=None,
            accumulation_steps=1,
            step=0,
            args=_step_args(full_sample_activation_checkpointing=True),
            running_states=(None, None, None, None, None, None),
        )

    assert model.reset_called is True
    assert all(parameter.grad is None for parameter in model.parameters())


def test_runtime_zero_detach_is_normalized_and_synchronized_to_both_rwkv_cells():
    config = _tiny_config(detach_every_n_steps=32)
    model = HierarchosCore(config)
    model.config.detach_every_n_steps = 0

    model.refresh_runtime_config()

    assert model.config.detach_every_n_steps is None
    assert model.h_rnn.detach_every_n_steps is None
    assert model.l_rnn.detach_every_n_steps is None
    model.train()
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        labels=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        return_logits=True,
    )
    assert torch.isfinite(outputs["loss"])


def test_old_checkpoint_weights_load_unchanged_into_full_sample_runtime():
    torch.manual_seed(11)
    old_config = _tiny_config(detach_every_n_steps=32)
    old_model = HierarchosCore(old_config)
    old_state = {name: tensor.detach().clone() for name, tensor in old_model.state_dict().items()}

    new_config = _tiny_config(detach_every_n_steps=0)
    new_config.full_sample_bptt = True
    new_config.full_sample_activation_checkpointing = True
    new_model = HierarchosCore(new_config)
    load_model_state_dict_compatible(new_model, old_state, "pre-chat checkpoint")

    assert set(new_model.state_dict()) == set(old_state)
    for name, expected in old_state.items():
        torch.testing.assert_close(new_model.state_dict()[name], expected, rtol=0, atol=0)


def test_whole_sample_checkpoint_matches_real_model_outputs_and_all_gradients():
    torch.manual_seed(23)
    direct_model = HierarchosCore(_tiny_config(use_rosa=True))
    checkpoint_model = HierarchosCore(AttrDict(dict(direct_model.config)))
    checkpoint_model.load_state_dict(direct_model.state_dict())
    direct_model.train()
    checkpoint_model.train()

    input_ids = torch.tensor([[3, 5, 3, 7, 3, 9]], dtype=torch.long)
    labels = input_ids.clone()
    rosa_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=64,
                chunk_size=2,
                rosa_max_ctx=32,
            )
        ],
        dtype=torch.long,
    )
    kwargs = _forward_kwargs(input_ids, labels, rosa_ids=rosa_ids, raw_ltm=True)

    direct_outputs = direct_model(**kwargs)
    checkpoint_outputs = _checkpointed_training_model_call(checkpoint_model, kwargs)
    for values in (direct_outputs["raw_topk_vals"], checkpoint_outputs["raw_topk_vals"]):
        for value in values:
            value.retain_grad()

    _backward_loss(direct_outputs).backward()
    _backward_loss(checkpoint_outputs).backward()

    for key in (
        "loss",
        "logits",
        "ponder_cost",
        "commitment_cost",
        "h_state",
        "l_state",
        "prev_context",
        "target_context",
        "drift_state",
    ):
        torch.testing.assert_close(
            checkpoint_outputs[key],
            direct_outputs[key],
            rtol=0,
            atol=0,
        )

    direct_grads = dict(direct_model.named_parameters())
    checkpoint_grads = dict(checkpoint_model.named_parameters())
    assert direct_grads.keys() == checkpoint_grads.keys()
    for name in direct_grads:
        direct_grad = direct_grads[name].grad
        checkpoint_grad = checkpoint_grads[name].grad
        assert (direct_grad is None) == (checkpoint_grad is None), name
        if direct_grad is not None:
            torch.testing.assert_close(checkpoint_grad, direct_grad, rtol=0, atol=0)

    direct_raw = direct_outputs["raw_topk_vals"]
    checkpoint_raw = checkpoint_outputs["raw_topk_vals"]
    assert len(direct_raw) == len(checkpoint_raw) == input_ids.shape[1]
    for direct_value, checkpoint_value in zip(direct_raw, checkpoint_raw):
        assert direct_value.grad is not None
        assert checkpoint_value.grad is not None
        torch.testing.assert_close(checkpoint_value.grad, direct_value.grad, rtol=0, atol=0)


def test_final_target_reaches_first_token_embedding_through_checkpointed_full_bptt():
    torch.manual_seed(31)
    model = HierarchosCore(_tiny_config(use_rosa=False, detach_every_n_steps=None))
    model.train()
    input_ids = torch.tensor([[2, 4, 6, 8, 10, 12]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    outputs = _checkpointed_training_model_call(
        model,
        _forward_kwargs(input_ids, labels, raw_ltm=False),
    )

    outputs["loss"].backward()

    first_token_grad = model.tok_emb.weight.grad[input_ids[0, 0]]
    assert torch.isfinite(first_token_grad).all()
    assert float(first_token_grad.norm().item()) > 1e-9
    # Read-only disables only the discarded post-backward fast-state write;
    # token-level addressing, retrieval values, and their trainable routes stay live.
    assert float(model.qproj.weight.grad.norm().item()) > 0.0
    assert float(model.ltm.keys.grad.norm().item()) > 0.0
    assert float(model.ltm.vals.grad.norm().item()) > 0.0


def test_segment_weights_track_manager_stride_across_short_final_segment():
    labels = torch.tensor(
        [
            [1, 2, -100, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, -100, 7, 8, -100, -100],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    loss_weights = torch.tensor(
        [
            [0.0, 0.2, 1.0, 2.0, 0.5, 3.0, 0.7, 1.2, 0.1, 4.0],
            [0.0, 1.0, 0.5, 2.0, 1.0, 0.3, 2.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    plan = compute_chunk_training_weights(
        labels,
        attention_mask,
        chunk_size=3,
        loss_weights=loss_weights,
        h_stride=4,
    )

    assert [(chunk["start"], chunk["end"]) for chunk in plan] == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
    ]
    assert [chunk["real_tokens"] for chunk in plan] == [6, 6, 5, 1]
    # Manager steps occur at absolute positions 0, 4, and 8. Position 8 is
    # padding in the second row, and the short final segment has no manager step.
    assert [chunk["ponder_tokens"] for chunk in plan] == [2, 2, 1, 0]
    torch.testing.assert_close(
        torch.tensor([chunk["ponder_ratio"] for chunk in plan]),
        torch.tensor([0.4, 0.4, 0.2, 0.0]),
        rtol=0,
        atol=0,
    )


def test_segmented_checkpointed_train_step_matches_direct_full_sample_math():
    torch.manual_seed(47)
    direct_config = _tiny_config(use_rosa=True, detach_every_n_steps=None)
    direct_config.h_stride = 4
    direct_config.commitment_threshold = 0.0
    direct_model = HierarchosCore(direct_config)
    segmented_model = HierarchosCore(AttrDict(dict(direct_model.config)))
    segmented_model.load_state_dict(direct_model.state_dict())

    input_ids = torch.tensor(
        [[3, 5, 7, 9, 11, 13, 15, 17, 19, 21]],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[:, 2] = -100
    labels[:, 7] = -100
    attention_mask = torch.ones_like(input_ids)
    loss_weights = torch.tensor(
        [[0.0, 0.25, 2.0, 0.5, 3.0, 0.75, 1.5, 0.0, 4.0, 1.0]],
        dtype=torch.float32,
    )
    rosa_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=64,
                chunk_size=3,
                rosa_max_ctx=32,
            )
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loss_weights": loss_weights,
        "rosa_ids": rosa_ids,
    }
    common_args = dict(
        training_chunk_size=3,
        full_sample_checkpoint_segment_size=3,
        full_sample_bptt=True,
        persist_state=False,
        adaptive_ponder=True,
        ponder_target_scale=0.65,
        max_h_steps=2,
        ponder_loss_weight=0.003,
        commitment_loss_weight=0.5,
        max_ce_loss_for_backward=0.0,
        max_ponder_cost_for_backward=0.0,
        max_commitment_cost_for_backward=4.0,
        ltm_value_alignment_weight=0.0,
    )

    direct_outputs, direct_states = _run_without_optimizer_step(
        direct_model,
        batch,
        _step_args(full_sample_activation_checkpointing=False, **common_args),
    )
    segmented_outputs, segmented_states = _run_without_optimizer_step(
        segmented_model,
        batch,
        _step_args(full_sample_activation_checkpointing=True, **common_args),
    )

    assert direct_outputs is not None
    assert segmented_outputs is not None
    for key in ("loss", "ponder_cost", "commitment_cost"):
        torch.testing.assert_close(
            segmented_outputs[key],
            direct_outputs[key],
            rtol=2e-5,
            atol=2e-6,
        )
    _assert_nested_close(segmented_states, direct_states)
    _assert_parameter_gradients_close(segmented_model, direct_model)


def test_segmented_checkpointed_full_bptt_does_not_detach_boundary_states():
    torch.manual_seed(73)
    direct_config = _tiny_config(use_rosa=False, detach_every_n_steps=None)
    direct_config.h_stride = 4
    direct_model = HierarchosCore(direct_config)
    segmented_model = HierarchosCore(AttrDict(dict(direct_model.config)))
    segmented_model.load_state_dict(direct_model.state_dict())

    input_ids = torch.tensor(
        [[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]],
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_weights[:, -1] = 2.0
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
        "loss_weights": loss_weights,
    }
    common_args = dict(
        training_chunk_size=3,
        full_sample_checkpoint_segment_size=3,
        full_sample_bptt=True,
        persist_state=False,
        adaptive_ponder=False,
        ponder_loss_weight=0.0,
        commitment_loss_weight=0.0,
        max_ce_loss_for_backward=0.0,
        ltm_value_alignment_weight=0.0,
    )

    _run_without_optimizer_step(
        direct_model,
        batch,
        _step_args(full_sample_activation_checkpointing=False, **common_args),
    )
    _run_without_optimizer_step(
        segmented_model,
        batch,
        _step_args(full_sample_activation_checkpointing=True, **common_args),
    )

    # The only supervised target is predicted in the third segment. A detach at
    # either earlier boundary makes the unique first-token embedding gradient zero.
    direct_first_grad = direct_model.tok_emb.weight.grad[input_ids[0, 0]]
    segmented_first_grad = segmented_model.tok_emb.weight.grad[input_ids[0, 0]]
    assert float(segmented_first_grad.norm().item()) > 1e-9
    torch.testing.assert_close(
        segmented_first_grad,
        direct_first_grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_segmented_full_bptt_preserves_all_recurrent_clamps_and_chunked_loss_math():
    torch.manual_seed(89)
    direct_config = _tiny_config(use_rosa=False, detach_every_n_steps=None)
    direct_config.cpu_chunked_lm_loss = True
    direct_config.cpu_loss_chunk_rows = 2
    direct_config.recurrent_state_clamp = 0.02
    direct_config.context_state_clamp = 0.015
    direct_config.drift_state_clamp = 0.01
    direct_config.drift_norm_clamp = 0.025
    direct_config.activation_clamp = 0.04
    direct_config.halt_logit_clamp = 0.03
    direct_config.rwkv_channel_mix_key_clamp = 0.02
    direct_config.rwkv_channel_mix_deepembed_clamp = 0.02
    direct_model = HierarchosCore(direct_config)
    segmented_model = HierarchosCore(AttrDict(dict(direct_model.config)))
    segmented_model.load_state_dict(direct_model.state_dict())

    input_ids = torch.tensor([[3, 6, 9, 12, 15, 18, 21]], dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
    }
    common_args = dict(
        full_sample_bptt=True,
        full_sample_checkpoint_segment_size=3,
        persist_state=False,
        cpu_chunked_lm_loss=True,
        adaptive_ponder=True,
        ponder_target_scale=0.65,
        max_h_steps=2,
        ponder_loss_weight=0.003,
        commitment_loss_weight=0.5,
        max_ce_loss_for_backward=0.0,
        max_ponder_cost_for_backward=0.0,
        max_commitment_cost_for_backward=4.0,
        ltm_value_alignment_weight=0.0,
        recurrent_state_clamp=direct_config.recurrent_state_clamp,
        context_state_clamp=direct_config.context_state_clamp,
        drift_state_clamp=direct_config.drift_state_clamp,
        drift_norm_clamp=direct_config.drift_norm_clamp,
    )

    direct_outputs, direct_states = _run_without_optimizer_step(
        direct_model,
        batch,
        _step_args(full_sample_activation_checkpointing=False, **common_args),
    )
    segmented_outputs, segmented_states = _run_without_optimizer_step(
        segmented_model,
        batch,
        _step_args(full_sample_activation_checkpointing=True, **common_args),
    )

    assert torch.isfinite(segmented_outputs["loss"])
    _assert_nested_close(segmented_states, direct_states)
    _assert_parameter_gradients_close(segmented_model, direct_model)

    h_state, l_state, prev_context, target_context, drift_state, _ltm_state = segmented_states
    for state in (h_state, l_state, prev_context, target_context, drift_state):
        assert torch.isfinite(state).all()
    assert float(h_state.abs().max()) <= direct_config.recurrent_state_clamp
    assert float(l_state.abs().max()) <= direct_config.recurrent_state_clamp
    assert float(prev_context.abs().max()) <= direct_config.context_state_clamp
    assert float(target_context.abs().max()) <= direct_config.context_state_clamp
    assert float(drift_state.abs().max()) <= direct_config.drift_state_clamp
    assert float(torch.linalg.vector_norm(drift_state.float(), dim=-1).max()) <= (
        direct_config.drift_norm_clamp + 1e-6
    )
