from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hierarchos import AttrDict, HierarchosCore
from hierarchos.models.core import WorkerLoop
from hierarchos.models.rwkv_cell import RWKVCell


def _tiny_config(*, version=2, use_rosa=False, detach_every_n_steps=0):
    return AttrDict(
        vocab_size=41,
        context_dim=8,
        h_hidden=8,
        l_hidden=8,
        persistent_dim=4,
        ltm_slots=8,
        ltm_key_dim=4,
        ltm_val_dim=4,
        ltm_topk=2,
        max_h_steps=3,
        max_l_steps=3,
        h_stride=2,
        l_conv_atol=1e-4,
        commitment_threshold=0.01,
        use_deepembed=False,
        use_rosa=use_rosa,
        memory_token_routers=False,
        compile=False,
        gradient_checkpointing=False,
        inference_logit_parity=True,
        detach_every_n_steps=detach_every_n_steps,
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        activation_clamp=100.0,
        z_loss_weight=0.0,
        core_recurrence_version=version,
    )


class _StableRNN(nn.Module):
    def forward(self, x, state, timestep=None, deepemb_vec=None):
        return x, state


def _worker(width, *, static, commitment_mode="mean-square"):
    config = SimpleNamespace(
        max_l_steps=3,
        l_conv_atol=0.2,
        commitment_threshold=0.01,
        commitment_cost_mode=commitment_mode,
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        drift_delta_scale=1.0,
        activation_clamp=100.0,
        compile_static_worker_loop=static,
        inference_logit_parity=True,
    )
    projection = nn.Linear(width * 2, width, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[:, :width].copy_(torch.eye(width))
    rnn = _StableRNN()
    rnn.train()
    return WorkerLoop(
        config,
        rnn,
        projection,
        nn.Identity(),
        nn.Identity(),
    )


def test_worker_math_is_static_flag_independent_and_batch_row_local():
    loop = _worker(2, static=False)
    row = torch.tensor([[0.05, 0.05]])
    peer = torch.tensor([[1.0, 1.0]])
    single_args = (
        row,
        torch.zeros_like(row),
        torch.zeros(1, 2, 1),
        torch.zeros_like(row),
    )
    paired_args = (
        torch.cat([row, peer], dim=0),
        torch.zeros(2, 2),
        torch.zeros(2, 2, 1),
        torch.zeros(2, 2),
    )

    eager_single = loop(*single_args)
    eager_paired = loop(*paired_args)
    loop.compile_static_worker_loop = True
    static_single = loop(*single_args)
    static_paired = loop(*paired_args)

    for eager, static in zip(eager_single, static_single):
        torch.testing.assert_close(eager, static, rtol=0, atol=0)
    for single, paired in zip(eager_single, eager_paired):
        torch.testing.assert_close(single[0], paired[0], rtol=0, atol=0)
    for eager, static in zip(eager_paired, static_paired):
        torch.testing.assert_close(eager, static, rtol=0, atol=0)


def test_corrected_commitment_cost_is_width_invariant():
    def cost(width, mode):
        loop = _worker(width, static=False, commitment_mode=mode)
        enc = torch.full((1, width), 0.5)
        output = loop(
            enc,
            torch.zeros_like(enc),
            torch.zeros(1, width, 1),
            torch.zeros_like(enc),
        )
        return output[2]

    corrected_2 = cost(2, "mean-square")
    corrected_8 = cost(8, "mean-square")
    torch.testing.assert_close(corrected_2, corrected_8, rtol=1e-6, atol=1e-7)

    legacy_2 = cost(2, "sum-square")
    legacy_8 = cost(8, "sum-square")
    assert not torch.allclose(legacy_2, legacy_8)


def test_mask_contract_rejects_holes_and_masked_supervision():
    model = HierarchosCore(_tiny_config())
    model.train()
    ids = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="right padding only"):
        model(ids, attention_mask=torch.tensor([[1, 0, 1]]))
    with pytest.raises(ValueError, match="right padding only"):
        model(ids, attention_mask=torch.tensor([[0, 1, 1]]))
    with pytest.raises(ValueError, match="ignore_index=-100"):
        model(
            ids,
            attention_mask=torch.tensor([[1, 0, 0]]),
            labels=torch.tensor([[1, 2, -100]]),
        )
    with pytest.raises(ValueError, match="must be zero"):
        model(
            ids,
            attention_mask=torch.tensor([[1, 0, 0]]),
            labels=torch.tensor([[1, -100, -100]]),
            loss_weights=torch.tensor([[1.0, 1.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="lookahead labels"):
        model(
            ids,
            attention_mask=torch.tensor([[1, 0, 0]]),
            labels=torch.tensor([[1, -100, -100, 2]]),
        )


def test_right_padded_rows_freeze_all_recurrent_carriers():
    torch.manual_seed(11)
    model = HierarchosCore(_tiny_config())
    model.train()
    ids = torch.tensor([[1, 9, 10], [1, 2, 3]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])

    padded = model(ids, attention_mask=mask)
    single = model(ids[:1, :1], attention_mask=torch.ones(1, 1, dtype=torch.long))

    for name in (
        "h_state",
        "l_state",
        "prev_context",
        "target_context",
        "drift_state",
    ):
        torch.testing.assert_close(
            padded[name][0],
            single[name][0],
            rtol=1e-5,
            atol=1e-6,
        )
    assert padded["step_telemetry"]["h_effective_steps"][0].tolist()[1:] == [0.0, 0.0]
    assert padded["step_telemetry"]["l_effective_steps"][0].tolist()[1:] == [0.0, 0.0]


def test_complete_graph_cut_occurs_before_boundary_computation():
    torch.manual_seed(12)
    model = HierarchosCore(
        _tiny_config(detach_every_n_steps=2)
    )
    model.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, -100, 5]])

    initial_tensors = [
        model.h_rnn.initial_state(1).requires_grad_(),
        model.l_rnn.initial_state(1).requires_grad_(),
        torch.zeros(1, 8, requires_grad=True),
        torch.zeros(1, 8, requires_grad=True),
    ]
    output = model(
        ids,
        labels=labels,
        h_state=initial_tensors[0],
        l_state=initial_tensors[1],
        prev_context=initial_tensors[2],
        target_context=initial_tensors[3],
    )
    gradients = torch.autograd.grad(
        output["loss"],
        initial_tensors,
        allow_unused=True,
    )
    for gradient in gradients:
        assert gradient is None or torch.count_nonzero(gradient) == 0


def test_corrected_drift_is_invariant_to_forward_chunking():
    torch.manual_seed(13)
    model = HierarchosCore(_tiny_config())
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with torch.no_grad():
        full = model(ids)
        first = model(ids[:, :3])
        second = model(
            ids[:, 3:],
            h_state=first["h_state"],
            l_state=first["l_state"],
            prev_context=first["prev_context"],
            target_context=first["target_context"],
            drift_state=first["drift_state"],
            global_pos_offset=3,
        )

    chunked_logits = torch.cat([first["logits"], second["logits"]], dim=1)
    torch.testing.assert_close(full["logits"], chunked_logits, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(full["h_state"], second["h_state"], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(full["l_state"], second["l_state"], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(full["drift_state"], second["drift_state"], rtol=1e-5, atol=2e-6)


def test_manager_commits_the_same_act_weighted_state_it_ponders():
    torch.manual_seed(14)
    model = HierarchosCore(_tiny_config())
    model.eval()
    records = []

    def record_step(_module, _inputs, output):
        records.append((output[0].detach(), output[1].detach()))

    hook = model.h_rnn.register_forward_hook(record_step)
    try:
        with torch.no_grad():
            output = model(torch.tensor([[1]]))
    finally:
        hook.remove()

    assert len(records) == model.config.max_h_steps
    step_outputs = torch.stack([item[0] for item in records]).float()
    step_states = torch.stack([item[1] for item in records]).float()
    halt = torch.stack(
        [
            torch.sigmoid(model.h_halt_proj(item).squeeze(-1)).clamp(
                1e-6,
                1.0 - 1e-6,
            )
            for item in step_outputs
        ]
    ).float()
    remain = 1.0 - halt
    shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
    cumulative = torch.cumprod(shifted, dim=0)
    weights = halt * cumulative
    remainder = cumulative[-1] * (1.0 - halt[-1])
    total = weights.sum(dim=0) + remainder + 1e-8
    weights = weights / total.unsqueeze(0)
    remainder = remainder / total
    expected = (
        (weights[..., None, None] * step_states).sum(dim=0)
        + remainder[..., None, None] * step_states[-1]
    )
    torch.testing.assert_close(output["h_state"], expected, rtol=1e-5, atol=1e-6)


def test_explicit_rwkv_readout_tracks_output_with_legacy_fallback():
    torch.manual_seed(15)
    explicit = RWKVCell(
        4,
        head_size=2,
        state_readout_mode="explicit-output",
    )
    x = torch.randn(2, 4)
    output, state = explicit(x, None)
    assert state.shape[-1] == 6
    torch.testing.assert_close(explicit.state_hidden(state), output.float())

    legacy = RWKVCell(
        4,
        head_size=2,
        state_readout_mode="legacy-input-cache",
    )
    _, legacy_state = legacy(x, None)
    assert legacy_state.shape[-1] == 5
    torch.testing.assert_close(
        legacy.state_hidden(legacy_state),
        legacy_state[:, :, 0],
    )


def test_core_rejects_malformed_coherent_state_before_context_readout():
    model = HierarchosCore(_tiny_config(version=2)).eval()
    malformed_h = torch.zeros(
        1,
        model.config.h_hidden - 1,
        model.h_rnn.state_size,
    )
    with pytest.raises(ValueError, match="strict 'explicit-output' contract"):
        model(torch.tensor([[1, 2]]), h_state=malformed_h)


def test_core_rejects_context_without_manager_state():
    model = HierarchosCore(_tiny_config(version=2)).eval()
    with pytest.raises(ValueError, match="cannot be restored without"):
        model(
            torch.tensor([[1, 2]]),
            prev_context=torch.zeros(1, model.config.context_dim),
        )


def test_core_rejects_malformed_drift_instead_of_silently_resetting_it():
    model = HierarchosCore(_tiny_config(version=2)).eval()
    with pytest.raises(ValueError, match="drift_state must have shape"):
        model(
            torch.tensor([[1, 2]]),
            drift_state=torch.zeros(1, model.config.context_dim + 1),
        )


def test_core_rejects_ambiguous_or_malformed_ltm_state_carriers():
    model = HierarchosCore(_tiny_config(version=2)).eval()
    fast = model.ltm.fast_vals.clone()
    momentum = model.ltm._mom_vals.clone()
    timestamps = model.ltm.timestamps.clone()
    sources = model.ltm.sources.clone()

    with pytest.raises(ValueError, match="exactly 2, 3, 4, 6, or 7"):
        model(
            torch.tensor([[1, 2]]),
            ltm_memory_state=(fast, momentum, None, None, timestamps),
        )

    with pytest.raises(ValueError, match="slot count"):
        model(
            torch.tensor([[1, 2]]),
            ltm_memory_state=(
                fast,
                momentum,
                None,
                None,
                timestamps[:-1],
                sources[:-1],
            ),
        )


def test_zero_drift_delta_scale_is_not_silently_replaced_by_one():
    config = _tiny_config(version=2)
    config.drift_delta_scale = 0.0
    model = HierarchosCore(config)
    assert model.worker_loop_module.drift_delta_scale == 0.0


def test_saturated_training_logits_keep_gradients_and_report_telemetry():
    model = HierarchosCore(_tiny_config())
    model.train()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[1].fill_(100.0)

    hidden = torch.ones(1, 2, model.config.context_dim, requires_grad=True)
    labels = torch.tensor([[0, 0, 0]])
    loss, numerics = model._compute_cuda_chunked_lm_loss(
        hidden,
        labels,
        z_loss_weight=0.0,
        return_telemetry=True,
    )
    loss.backward()

    assert numerics["raw_logit_max_abs"] > 30.0
    assert numerics["raw_logit_saturation_fraction"] > 0.0
    assert numerics["raw_logit_nonfinite_count"] == 0
    assert torch.count_nonzero(model.lm_head.weight.grad[1]) > 0
