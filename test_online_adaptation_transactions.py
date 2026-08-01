import copy
import os
from types import SimpleNamespace

import pytest
import torch

import hierarchos.inference.chat as chat_module
from hierarchos import AttrDict, HierarchosCore
from hierarchos.inference.chat import (
    LTM_DELTA_OVERLAY_VERSION,
    _default_ltm_delta_overlay_path,
    apply_online_feedback_transaction,
    load_ltm_delta_overlay,
    resolve_online_adaptation_policy,
    save_ltm_delta_overlay_atomic,
)


def _tiny_online_config():
    return AttrDict(
        architecture_revision="coherent-v9",
        vocab_size=32,
        context_dim=8,
        h_hidden=8,
        l_hidden=8,
        persistent_dim=4,
        ltm_slots=8,
        ltm_key_dim=4,
        ltm_val_dim=4,
        ltm_topk=2,
        max_h_steps=2,
        min_h_steps=1,
        max_l_steps=2,
        h_stride=2,
        h_halt_thresh=0.9,
        use_deepembed=False,
        use_rosa=False,
        memory_token_routers=False,
        compile=False,
        gradient_checkpointing=False,
        detach_every_n_steps=0,
        training_chunk_size=4,
        reference_chunk_len=4,
        max_length=16,
        inference_logit_parity=True,
        z_loss_weight=0.0,
    )


def _tiny_online_model(seed=42):
    torch.manual_seed(seed)
    config = _tiny_online_config()
    model = HierarchosCore(config)
    with torch.no_grad():
        # Make the memory path strong enough that the real objective has a stable,
        # measurable gradient even at tiny unit-test width.
        model.ltm_gate_logit.fill_(10.0)
    model.ltm.accumulate_deltas = True
    model._hierarchos_checkpoint_metadata = {
        "checkpoint_kind": "inference",
        "run_identity": {"sha256": "a" * 64},
    }
    return model, config


def _clone_named_parameters(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _clone_runtime_tensors(model):
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    for name in (
        "fast_vals",
        "_mom_vals",
        "ltm_deltas",
        "timestamps",
        "sources",
        "wallclock_timestamps",
    ):
        value = getattr(model.ltm, name, None)
        if torch.is_tensor(value):
            state[f"runtime::{name}"] = value.detach().clone()
    return state


def _assert_tensor_snapshot_unchanged(model, before):
    after = _clone_runtime_tensors(model)
    assert set(after) == set(before)
    for name in before:
        assert torch.equal(after[name], before[name]), name


class _IdentityTokenizer:
    name_or_path = "online-adaptation-test"
    special_tokens_map = {"pad_token": "token-0", "eos_token": "token-0"}

    def __init__(self, *, swap_ids=False):
        self._vocab = {f"token-{index}": index for index in range(32)}
        if swap_ids:
            self._vocab["token-1"], self._vocab["token-2"] = (
                self._vocab["token-2"],
                self._vocab["token-1"],
            )

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)


@pytest.mark.parametrize(
    ("settings", "expected"),
    (
        ({}, ("validated", False, False)),
        (
            {
                "online_adaptation_policy": "off",
                "passive_learning": True,
                "passive_response_learning": True,
            },
            ("off", False, False),
        ),
        (
            {"online_adaptation_policy": "validated", "passive_learning": True},
            ("validated", True, False),
        ),
        (
            {
                "online_adaptation_policy": "validated",
                "passive_response_learning": True,
            },
            ("validated", True, True),
        ),
        ({"online_adaptation_policy": "prompt"}, ("prompt", True, False)),
        (
            {"online_adaptation_policy": "prompt+response"},
            ("prompt+response", True, True),
        ),
    ),
)
def test_online_adaptation_policy_defaults_and_legacy_flag_compatibility(
    settings,
    expected,
):
    assert resolve_online_adaptation_policy(SimpleNamespace(**settings)) == expected


def test_online_adaptation_policy_rejects_unknown_values():
    with pytest.raises(ValueError, match="unknown online adaptation policy"):
        resolve_online_adaptation_policy(
            SimpleNamespace(online_adaptation_policy="validateed")
        )


def test_default_ltm_overlay_path_is_deterministic_and_model_local(tmp_path):
    model_directory = tmp_path / "model-directory"
    model_directory.mkdir()
    checkpoint = tmp_path / "hierarchos_epoch_1.pt"
    extensionless = tmp_path / "hierarchos_checkpoint"

    assert _default_ltm_delta_overlay_path(model_directory) == os.path.join(
        os.path.abspath(model_directory),
        "hierarchos_ltm_updates.pt",
    )
    assert _default_ltm_delta_overlay_path(checkpoint) == os.path.abspath(
        tmp_path / "hierarchos_epoch_1_ltm_updates.pt"
    )
    assert _default_ltm_delta_overlay_path(extensionless) == os.path.abspath(
        tmp_path / "hierarchos_checkpoint_ltm_updates.pt"
    )


def test_online_transaction_accepts_bounded_objective_improvement_without_weight_updates():
    model, config = _tiny_online_model()
    parameters_before = _clone_named_parameters(model)
    runtime_before = _clone_runtime_tensors(model)
    fast_before = model.ltm.fast_vals.detach().clone()
    delta_before = model.ltm.ltm_deltas.detach().clone()

    result = apply_online_feedback_transaction(
        model,
        torch.tensor([1, 2]),
        torch.tensor([5, 6]),
        config=config,
        learning_rate=0.05,
        max_delta_norm=1.0,
        max_fast_norm=64.0,
        max_slot_norm=4.0,
        max_backoff_steps=4,
    )

    assert result["committed"] is True
    assert result["reason"] == "accepted"
    assert result["loss_after"] <= result["loss_before"] + max(
        1e-7,
        abs(result["loss_before"]) * 1e-6,
    )
    assert 0.0 < result["delta_norm"] <= 1.0
    assert result["fast_norm"] <= 64.0
    assert not torch.equal(model.ltm.fast_vals, fast_before)
    assert not torch.equal(model.ltm.ltm_deltas, delta_before)
    torch.testing.assert_close(
        model.ltm.ltm_deltas - delta_before,
        model.ltm.fast_vals - fast_before,
    )
    runtime_after = _clone_runtime_tensors(model)
    changed_runtime_tensors = {
        name
        for name in runtime_before
        if not torch.equal(runtime_before[name], runtime_after[name])
    }
    assert {"runtime::fast_vals", "runtime::ltm_deltas"}.issubset(
        changed_runtime_tensors
    )
    assert changed_runtime_tensors.issubset(
        {
            "ltm.fast_vals",
            "ltm.timestamps",
            "ltm.sources",
            "runtime::fast_vals",
            "runtime::ltm_deltas",
            "runtime::timestamps",
            "runtime::sources",
            "runtime::wallclock_timestamps",
        }
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), parameters_before[name]), name
        assert parameter.grad is None


def test_online_transaction_compute_only_leaves_all_memory_and_weights_untouched():
    model, config = _tiny_online_model()
    before = _clone_runtime_tensors(model)
    training_before = model.training
    suppress_before = getattr(model, "suppress_hebbian", True)

    result = apply_online_feedback_transaction(
        model,
        torch.tensor([1, 2]),
        torch.tensor([5, 6]),
        config=config,
        learning_rate=0.05,
        compute_only=True,
    )

    assert result["committed"] is False
    assert result["reason"] == "compute-only"
    assert torch.isfinite(torch.tensor(result["loss_before"]))
    _assert_tensor_snapshot_unchanged(model, before)
    assert model.training is training_before
    assert getattr(model, "suppress_hebbian", True) is suppress_before


def test_online_transaction_rejects_over_budget_candidate_without_partial_mutation():
    model, config = _tiny_online_model()
    before = _clone_runtime_tensors(model)

    result = apply_online_feedback_transaction(
        model,
        torch.tensor([1, 2]),
        torch.tensor([5, 6]),
        config=config,
        learning_rate=0.05,
        max_delta_norm=1e-12,
        max_fast_norm=64.0,
        max_slot_norm=4.0,
        max_backoff_steps=1,
    )

    assert result["committed"] is False
    assert result["reason"] == "objective-or-budget-rejected"
    _assert_tensor_snapshot_unchanged(model, before)


def test_online_transaction_rolls_back_objective_worsening_candidate(
    monkeypatch,
):
    model, config = _tiny_online_model()
    before = _clone_runtime_tensors(model)
    replay = chat_module.replay_online_feedback_with_training_recurrence
    verification_calls = 0

    def force_worse_verification(*args, **kwargs):
        nonlocal verification_calls
        outputs = replay(*args, **kwargs)
        if not kwargs.get("return_memory_trace", True):
            verification_calls += 1
            logits = torch.full_like(outputs["logits"], -100.0)
            logits[..., 0] = 100.0
            outputs = dict(outputs)
            outputs["logits"] = logits
        return outputs

    monkeypatch.setattr(
        chat_module,
        "replay_online_feedback_with_training_recurrence",
        force_worse_verification,
    )
    result = apply_online_feedback_transaction(
        model,
        torch.tensor([1, 2]),
        torch.tensor([5, 6]),
        config=config,
        learning_rate=0.05,
        max_delta_norm=1.0,
        max_fast_norm=64.0,
        max_slot_norm=4.0,
        max_backoff_steps=2,
    )

    assert verification_calls == 2
    assert result["committed"] is False
    assert result["reason"] == "objective-or-budget-rejected"
    _assert_tensor_snapshot_unchanged(model, before)


def test_atomic_v3_overlay_round_trip_is_identity_bound_and_cumulative(tmp_path):
    base_model, _config = _tiny_online_model()
    source = copy.deepcopy(base_model)
    matching = copy.deepcopy(base_model)
    tokenizer = _IdentityTokenizer()
    first_delta = torch.linspace(
        -0.02,
        0.02,
        source.ltm.ltm_deltas.numel(),
        dtype=source.ltm.ltm_deltas.dtype,
    ).reshape_as(source.ltm.ltm_deltas)
    with torch.no_grad():
        source.ltm.ltm_deltas.copy_(first_delta)
        source.ltm.timestamps.copy_(
            torch.arange(1, source.ltm.timestamps.numel() + 1, dtype=torch.float32)
        )
        source.ltm.sources.copy_(
            torch.arange(source.ltm.sources.numel(), dtype=torch.long) % 4
        )
        source.ltm.wallclock_timestamps.copy_(
            torch.arange(
                100,
                100 + source.ltm.wallclock_timestamps.numel(),
                dtype=source.ltm.wallclock_timestamps.dtype,
            )
        )

    overlay_path = tmp_path / "online-memory.pt"
    assert save_ltm_delta_overlay_atomic(
        source,
        overlay_path,
        tokenizer=tokenizer,
    ) == os.path.abspath(overlay_path)
    assert overlay_path.is_file()
    assert not (tmp_path / "online-memory.pt.tmp").exists()

    base_vals = matching.ltm.vals.detach().clone()
    assert load_ltm_delta_overlay(
        matching,
        overlay_path,
        tokenizer=tokenizer,
    ) == LTM_DELTA_OVERLAY_VERSION == 3
    torch.testing.assert_close(matching.ltm.vals, base_vals + first_delta)
    torch.testing.assert_close(matching.ltm.ltm_deltas, first_delta)
    torch.testing.assert_close(matching.ltm.timestamps, source.ltm.timestamps)
    assert torch.equal(matching.ltm.sources, source.ltm.sources)
    torch.testing.assert_close(
        matching.ltm.wallclock_timestamps,
        source.ltm.wallclock_timestamps,
    )

    second_delta = torch.full_like(first_delta, 0.005)
    with torch.no_grad():
        matching.ltm.fast_vals.copy_(second_delta)
        matching.ltm.ltm_deltas.add_(second_delta)
    cumulative_path = tmp_path / "online-memory-cumulative.pt"
    save_ltm_delta_overlay_atomic(
        matching,
        cumulative_path,
        tokenizer=tokenizer,
    )
    fresh = copy.deepcopy(base_model)
    fresh_base_vals = fresh.ltm.vals.detach().clone()
    assert load_ltm_delta_overlay(
        fresh,
        cumulative_path,
        tokenizer=tokenizer,
    ) == 3
    torch.testing.assert_close(
        fresh.ltm.vals,
        fresh_base_vals + first_delta + second_delta,
    )
    torch.testing.assert_close(
        fresh.ltm.ltm_deltas,
        first_delta + second_delta,
    )

    for mismatch in ("tokenizer", "model"):
        rejected = copy.deepcopy(base_model)
        if mismatch == "model":
            rejected._hierarchos_checkpoint_metadata = {
                "checkpoint_kind": "inference",
                "run_identity": {"sha256": "b" * 64},
            }
            rejected_tokenizer = tokenizer
        else:
            rejected_tokenizer = _IdentityTokenizer(swap_ids=True)
        before = _clone_runtime_tensors(rejected)
        with pytest.raises(RuntimeError, match="different model weights or tokenizer"):
            load_ltm_delta_overlay(
                rejected,
                overlay_path,
                tokenizer=rejected_tokenizer,
            )
        _assert_tensor_snapshot_unchanged(rejected, before)
