from types import SimpleNamespace
import argparse
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from hierarchos.training.datasets import EpochShuffleSampler, LengthGroupedBatchSampler
from hierarchos.training.trainer import (
    build_hierarchos_optimizer,
    build_training_checkpoint,
    build_exact_resume_identity,
    build_lr_scheduler,
    capture_effective_training_config,
    capture_ltm_lr_scheduler_state,
    capture_main_lr_scheduler_state,
    capture_dataloader_state,
    capture_rng_state,
    accumulation_divisor_for_step,
    configure_ltm_lr_schedule,
    compute_update_steps,
    compute_remaining_update_steps,
    get_current_ltm_lr,
    get_model_training_step,
    mark_val_proj_trained,
    advance_ltm_lr_schedule,
    restore_dataloader_state,
    restore_rng_state,
    restore_model_grad_state,
    restore_checkpoint_gradient_accumulation,
    restore_scheduler_state_and_live_lrs,
    resolve_training_step_offset,
    save_training_checkpoint_if_finite,
    set_dataloader_start_batch,
    host_batches_from_resume,
    should_step_accumulation,
    supervised_weight_mass,
    train_step,
    training_state_is_finite,
    _sanitize_model_nonfinite_,
    _sanitize_model_transient_state_,
    _sanitize_gradient_nonfinite_,
    _clamp_model_finite_magnitude_,
    _clip_gradients_and_check,
    configure_finetune_ltm_mode,
    ensure_finetune_training_mode,
    ltm_inner_updates_enabled,
    normalize_ltm_training_mode,
    validate_exact_resume_identity,
    validate_exact_running_states,
)
from hierarchos.models.revisions import architecture_contract_hash
from hierarchos.utils.checkpoint import (
    load_checkpoint_payload_compatible,
    sanitize_model_state_dict,
)
from hierarchos.utils.rosa import ROSAState
from hierarchos.utils.tokenizer import tokenizer_identity
import hierarchos_cli


class _UnlistedCheckpointMetadata:
    pass


class _FakeLTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fast_vals", torch.zeros(3, 2))
        self.register_buffer("_mom_vals", torch.zeros(3, 2))
        self.register_buffer("timestamps", torch.zeros(3))
        self.register_buffer("sources", torch.zeros(3, dtype=torch.long))
        self.register_buffer("neg_inf", torch.tensor(-float("inf")), persistent=False)


class _FakeTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = _FakeLTM()
        self.proj = nn.Linear(2, 2)
        self.config = {"context_dim": 2}


class _FakeDeepEmbedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_deepemb = nn.Embedding(4, 8)
        self.l_deepemb = nn.Embedding(4, 8)
        self.tok_emb = nn.Embedding(4, 2)
        self.proj = nn.Linear(2, 2)


class _RawRWKVMatrixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(4, 4))
        self.a1 = nn.Parameter(torch.ones(4, 2))
        self.r_k = nn.Parameter(torch.ones(4, 4))
        self.norm_scale = nn.Parameter(torch.ones(4))


class _CountingLTM:
    def __init__(self):
        self.inner_update_calls = 0

    def inner_update(self, *args, **kwargs):
        self.inner_update_calls += 1
        fast_vals = kwargs.get("fast_vals")
        mom_vals = kwargs.get("mom_vals")
        return fast_vals, mom_vals


class _LTMModeRecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(vocab_size=8, cpu_chunked_lm_loss=False, cuda_chunked_lm_loss=False)
        self.ltm = _CountingLTM()
        self.forward_flags = []
        self.reset_called = False

    def reset_memory(self):
        self.reset_called = True

    def forward(self, **kwargs):
        self.forward_flags.append({
            "return_raw_topk_values": kwargs.get("return_raw_topk_values"),
            "return_topk_indices": kwargs.get("return_topk_indices"),
        })
        device = self.weight.device
        raw_topk_vals = None
        topk_idx = None
        if kwargs.get("return_raw_topk_values", True):
            raw_topk_vals = [(self.weight * torch.ones(1, 1, 2, device=device))]
            topk_idx = torch.zeros(1, 1, 1, dtype=torch.long, device=device)
        return {
            "loss": self.weight * 1.0,
            "ponder_cost": torch.zeros((), device=device),
            "commitment_cost": torch.zeros((), device=device),
            "raw_topk_vals": raw_topk_vals,
            "topk_idx": topk_idx,
            "ltm_memory_state": (
                torch.zeros(1, 3, 2, device=device),
                torch.zeros(1, 3, 2, device=device),
                torch.arange(2, device=device).reshape(1, 2),
                [{"rosa": "state"}],
                torch.zeros(1, 3, device=device),
                torch.zeros(1, 3, dtype=torch.long, device=device),
            ),
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _NaNLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(vocab_size=8, cpu_chunked_lm_loss=False, cuda_chunked_lm_loss=False)
        self.reset_called = False

    def reset_memory(self):
        self.reset_called = True

    def forward(self, **kwargs):
        nan_loss = self.weight * torch.tensor(float("nan"), device=self.weight.device)
        return {
            "loss": nan_loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _InfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return weight.detach() * 0.0 + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("inf"))


class _InfGradModel(_NaNLossModel):
    def forward(self, **kwargs):
        finite_loss = _InfGradient.apply(self.weight)
        return {
            "loss": finite_loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _OOMModel(_NaNLossModel):
    def forward(self, **kwargs):
        raise RuntimeError("CUDA out of memory")


class _HighFiniteLossModel(_NaNLossModel):
    def forward(self, **kwargs):
        high_loss = self.weight * 20.0
        high_commitment = self.weight * 20.0
        return {
            "loss": high_loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": high_commitment,
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _FiniteLinearLossModel(_NaNLossModel):
    def forward(self, **kwargs):
        return {
            "loss": self.weight,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _NonfiniteCarrierModel(_FiniteLinearLossModel):
    def __init__(self, carrier_name):
        super().__init__()
        self.carrier_name = carrier_name

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        outputs[self.carrier_name] = torch.tensor(
            [float("nan")],
            device=self.weight.device,
        )
        return outputs


class _InputScaledLossModel(_NaNLossModel):
    def forward(self, **kwargs):
        scale = kwargs["input_ids"][:, 0].to(dtype=self.weight.dtype).mean()
        return {
            "loss": self.weight * scale,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _BoundaryRecordingModel(_NaNLossModel):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        labels = kwargs["labels"]
        self.seen.append((
            kwargs.get("global_pos_offset"),
            input_ids.detach().cpu().clone(),
            labels.detach().cpu().clone(),
        ))
        shifted_labels = labels[:, 1:]
        has_supervision = (shifted_labels != -100).any()
        loss = self.weight * (1.0 if bool(has_supervision.item()) else 0.0)
        return {
            "loss": loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


def test_training_checkpoint_preserves_resume_only_state():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)
    model.proj.weight.grad = torch.ones_like(model.proj.weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace()
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=2,
        mid_epoch_step=7,
        running_states=(torch.ones(1), None, None, None, None, None),
    )

    assert checkpoint["checkpoint_kind"] == "training"
    assert checkpoint["completed_epoch"] == 2
    assert checkpoint["mid_epoch_step"] == 7
    assert torch.equal(checkpoint["model_state_dict"]["ltm.fast_vals"], torch.full((3, 2), 3.0))
    assert torch.equal(checkpoint["model_state_dict"]["ltm._mom_vals"], torch.full((3, 2), 4.0))
    assert torch.equal(checkpoint["model_state_dict"]["ltm.timestamps"], torch.full((3,), 5.0))
    assert torch.equal(checkpoint["model_state_dict"]["ltm.sources"], torch.full((3,), 2, dtype=torch.long))
    assert checkpoint["grad_accumulation_active"] is True
    assert "proj.weight" in checkpoint["grad_state_dict"]
    assert checkpoint["grad_state_keys"] == ("proj.weight",)
    assert checkpoint["running_states"][0].device.type == "cpu"


def test_training_checkpoint_preserves_writer_alignment_readiness_progress():
    model = _FakeTrainModel()
    model.val_proj = nn.Linear(2, 2, bias=False)
    writer_norm = float(model.val_proj.weight.detach().float().norm().item())
    model.config.update(
        {
            "architecture_revision": "coherent-v9",
            "ltm_value_alignment_weight": 0.01,
            "ltm_value_alignment_stride": 8,
            "ltm_value_alignment_min_updates": 3,
            "ltm_value_alignment_ready_threshold": 0.2,
            "ltm_value_alignment_ema_decay": 0.0,
            "ltm_value_writer_max_norm": 64.0,
            "val_proj_alignment_updates": 2,
            "val_proj_alignment_last": 0.1,
            "val_proj_alignment_ema": 0.1,
            "val_proj_alignment_best": 0.08,
            "val_proj_writer_norm": writer_norm,
            "val_proj_trained": False,
        }
    )
    args = SimpleNamespace(
        ltm_value_alignment_weight=0.01,
        ltm_value_alignment_stride=8,
        ltm_value_alignment_min_updates=3,
        ltm_value_alignment_ready_threshold=0.2,
        ltm_value_alignment_ema_decay=0.0,
        ltm_value_writer_max_norm=64.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=1,
    )

    assert checkpoint["architecture_contract"][
        "architecture_contract_schema_version"
    ] == 3
    for key, expected in (
        ("val_proj_alignment_updates", 2),
        ("val_proj_alignment_last", 0.1),
        ("val_proj_alignment_ema", 0.1),
        ("val_proj_alignment_best", 0.08),
        ("val_proj_writer_norm", writer_norm),
        ("val_proj_trained", False),
    ):
        assert checkpoint["config"][key] == expected
        assert key not in checkpoint["architecture_contract"]
    for key in (
        "ltm_value_alignment_weight",
        "ltm_value_alignment_stride",
        "ltm_value_alignment_min_updates",
        "ltm_value_alignment_ready_threshold",
        "ltm_value_alignment_ema_decay",
        "ltm_value_writer_max_norm",
    ):
        assert checkpoint["effective_training_config"][key] == getattr(args, key)

    resumed = _FakeTrainModel()
    resumed.val_proj = nn.Linear(2, 2, bias=False)
    resumed.val_proj.load_state_dict(model.val_proj.state_dict())
    resumed.config = dict(checkpoint["config"])
    mark_val_proj_trained(resumed, alignment_cost=0.1)

    assert resumed.config["val_proj_alignment_updates"] == 3
    assert resumed.config["val_proj_trained"] is True
    # Readiness progress is mutable checkpoint capability metadata; continuing
    # it must not change the immutable learned-function contract identity.
    assert (
        architecture_contract_hash(resumed.config)
        == checkpoint["architecture_contract_sha256"]
    )


def test_training_checkpoint_omits_terminal_state_for_independent_samples():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=SimpleNamespace(persist_state=False),
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=3,
        running_states=(torch.ones(128), None, None, None, None, None),
    )

    assert "running_states" not in checkpoint


def test_training_checkpoint_keeps_terminal_state_for_contiguous_streams():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=SimpleNamespace(persist_state=True),
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=3,
        running_states=(torch.ones(128), None, None, None, None, None),
    )

    assert checkpoint["running_states"][0].device.type == "cpu"


def test_training_checkpoint_builder_does_not_silently_repair_nonfinite_model():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.data.view(-1)[0] = float("inf")
    args = SimpleNamespace(startup_weight_max_abs=0.0)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=1,
    )

    saved_weight = checkpoint["model_state_dict"]["proj.weight"]
    assert torch.isinf(saved_weight.view(-1)[0])
    assert torch.isinf(model.proj.weight.data.view(-1)[0])


def test_training_checkpoint_does_not_apply_startup_weight_clamp_mid_run():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.data.view(-1)[0] = 123.0
    args = SimpleNamespace(startup_weight_max_abs=100.0)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=1,
    )

    assert checkpoint["model_state_dict"]["proj.weight"].view(-1)[0].item() == 123.0
    assert model.proj.weight.data.view(-1)[0].item() == 123.0


def test_mid_epoch_checkpoint_preserves_v8_running_ltm_state():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ltm_state = (
        torch.full((2, 3, 2), 1.0),
        torch.full((2, 3, 2), 2.0),
        torch.arange(4).reshape(1, 4),
        [{"rosa": "state"}],
        torch.full((2, 3), 3.0),
        torch.full((2, 3), 2, dtype=torch.long),
    )
    running_states = (
        torch.ones(2, 4),
        torch.ones(2, 4) * 2,
        torch.ones(2, 2),
        torch.ones(2, 2) * 3,
        torch.ones(2, 2) * 4,
        ltm_state,
    )

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=SimpleNamespace(),
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=5,
        running_states=running_states,
    )

    saved_ltm_state = checkpoint["running_states"][5]
    assert len(saved_ltm_state) == 6
    assert torch.equal(saved_ltm_state[0], ltm_state[0])
    assert torch.equal(saved_ltm_state[1], ltm_state[1])
    assert torch.equal(saved_ltm_state[2], ltm_state[2])
    assert saved_ltm_state[3] == [{"rosa": "state"}]
    assert torch.equal(saved_ltm_state[4], ltm_state[4])
    assert torch.equal(saved_ltm_state[5], ltm_state[5])


def test_remaining_update_steps_counts_mid_accumulation_boundaries():
    assert compute_update_steps(dataloader_len=100, accumulation_steps=4) == 25
    assert compute_update_steps(dataloader_len=101, accumulation_steps=4) == 26
    assert compute_remaining_update_steps(
        dataloader_len=100,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=1,
        start_step=5,
    ) == 24
    assert compute_remaining_update_steps(
        dataloader_len=100,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=2,
        start_step=5,
    ) == 49
    assert compute_remaining_update_steps(
        dataloader_len=101,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=1,
        start_step=100,
    ) == 1
    assert compute_remaining_update_steps(
        dataloader_len=101,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=1,
        start_step=101,
    ) == 1
    assert compute_remaining_update_steps(
        dataloader_len=101,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=2,
        start_step=101,
    ) == 26


def test_accumulation_helpers_flush_tail_window():
    assert accumulation_divisor_for_step(0, dataloader_len=5, accumulation_steps=4) == 4
    assert accumulation_divisor_for_step(3, dataloader_len=5, accumulation_steps=4) == 4
    assert accumulation_divisor_for_step(4, dataloader_len=5, accumulation_steps=4) == 1
    assert should_step_accumulation(2, dataloader_len=5, accumulation_steps=4) is False
    assert should_step_accumulation(3, dataloader_len=5, accumulation_steps=4) is True
    assert should_step_accumulation(4, dataloader_len=5, accumulation_steps=4) is True


def test_lr_scheduler_warms_up_then_cosine_decays():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
    )

    scheduler = build_lr_scheduler(optimizer, args, num_update_steps=10)

    initial_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    warmed_lr = optimizer.param_groups[0]["lr"]
    for _ in range(9):
        optimizer.step()
        scheduler.step()
    final_lr = optimizer.param_groups[0]["lr"]

    assert 1e-5 < initial_lr < 1e-3
    assert abs(warmed_lr - 1e-3) < 1e-12
    assert abs(final_lr - 1e-5) < 1e-12


def test_scheduler_resume_restores_live_optimizer_lr_exactly():
    model = _FakeTrainModel()
    args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, args, num_update_steps=20)
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()
    expected_live_lr = optimizer.param_groups[0]["lr"]

    resumed_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler = build_lr_scheduler(
        resumed_optimizer,
        args,
        num_update_steps=20,
    )
    assert resumed_optimizer.param_groups[0]["lr"] != expected_live_lr

    restored = restore_scheduler_state_and_live_lrs(
        resumed_scheduler,
        resumed_optimizer,
        saved_scheduler,
        "test.pt",
    )
    assert restored == [expected_live_lr]
    assert resumed_optimizer.param_groups[0]["lr"] == expected_live_lr
    assert resumed_scheduler.get_last_lr() == [expected_live_lr]

    optimizer.step()
    scheduler.step()
    expected_next_lr = optimizer.param_groups[0]["lr"]
    resumed_optimizer.step()
    resumed_scheduler.step()
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_next_lr,
        rel=0,
        abs=1e-15,
    )


def test_scheduler_rebuild_uses_requested_peak_not_saved_initial_lr():
    model = _FakeTrainModel()
    old_args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
        rebuild_lr_schedule=False,
        override_scheduling=False,
    )
    old_optimizer = torch.optim.AdamW(model.parameters(), lr=old_args.starting_lr)
    old_scheduler = build_lr_scheduler(
        old_optimizer,
        old_args,
        num_update_steps=20,
    )
    for _ in range(7):
        old_optimizer.step()
        old_scheduler.step()

    rebuilt_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    rebuilt_optimizer.load_state_dict(old_optimizer.state_dict())
    rebuilt_args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=2e-4,
        min_lr=1e-6,
        warmup_steps=2,
        warmup_ratio=0.0,
        rebuild_lr_schedule=True,
        override_scheduling=False,
    )
    rebuilt_scheduler = build_lr_scheduler(
        rebuilt_optimizer,
        rebuilt_args,
        num_update_steps=10,
    )

    assert rebuilt_scheduler.base_lrs == [2e-4]
    for _ in range(2):
        rebuilt_optimizer.step()
        rebuilt_scheduler.step()
    assert rebuilt_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)


def test_scheduler_resume_reconstructs_saved_lambda_not_changed_cli_curve():
    model = _FakeTrainModel()
    saved_args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
        rebuild_lr_schedule=False,
        override_scheduling=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, saved_args, num_update_steps=20)
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    curve_state = capture_main_lr_scheduler_state(
        saved_args,
        scheduler,
        num_update_steps=20,
    )
    optimizer.step()
    scheduler.step()
    expected_next_lr = optimizer.param_groups[0]["lr"]

    changed_args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=1e-3,
        min_lr=2e-4,
        warmup_steps=0,
        warmup_ratio=0.25,
        rebuild_lr_schedule=False,
        override_scheduling=False,
    )
    resumed_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler = build_lr_scheduler(
        resumed_optimizer,
        changed_args,
        num_update_steps=20,
        resume_schedule_state=curve_state,
    )
    restore_scheduler_state_and_live_lrs(
        resumed_scheduler,
        resumed_optimizer,
        scheduler_state,
        "test.pt",
    )
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_next_lr,
        rel=0,
        abs=1e-15,
    )


def test_memory_gate_continuation_uses_persisted_step_not_new_loader_geometry():
    class _GateModel(nn.Module):
        def __init__(self, step):
            super().__init__()
            self.register_buffer(
                "memory_gate_warmup_step",
                torch.tensor(float(step)),
            )

    model = _GateModel(step=1234)
    wrapped = SimpleNamespace(
        base_model=SimpleNamespace(model=model),
    )

    # A weights-only continuation starts a fresh local epoch/session. Its new
    # dataloader length or inherited completed-epoch count must not rewind or
    # jump the persisted gate curriculum.
    offset = resolve_training_step_offset(wrapped, next_local_step=0)
    assert get_model_training_step(wrapped) == 1234
    assert offset == 1235
    assert offset + (3 * 17 + 4) == 1290

    # Existing exact-resume checkpoints already use the absolute local batch
    # formula; their derived correction remains zero.
    model.memory_gate_warmup_step.fill_(61799.0)
    assert resolve_training_step_offset(
        wrapped,
        next_local_step=61800,
    ) == 0


def test_ltm_lr_cosine_schedule_decays_and_advances():
    args = SimpleNamespace(
        ltm_lr=1e-3,
        min_ltm_lr=1e-5,
        min_lr=1e-7,
        disable_ltm_lr_schedule=False,
    )

    configure_ltm_lr_schedule(args, num_update_steps=10)

    assert abs(get_current_ltm_lr(args) - 1e-3) < 1e-12
    for _ in range(5):
        advance_ltm_lr_schedule(args)
    midpoint_lr = get_current_ltm_lr(args)
    assert 1e-5 < midpoint_lr < 1e-3
    for _ in range(5):
        advance_ltm_lr_schedule(args)
    assert abs(get_current_ltm_lr(args) - 1e-5) < 1e-12


def test_ltm_lr_scheduler_state_round_trips_in_training_checkpoint():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        ltm_lr=1e-4,
        min_ltm_lr=1e-8,
        min_lr=1e-8,
        disable_ltm_lr_schedule=False,
    )
    configure_ltm_lr_schedule(args, num_update_steps=20)
    for _ in range(7):
        advance_ltm_lr_schedule(args)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=7,
    )

    expected = capture_ltm_lr_scheduler_state(args)
    assert checkpoint["ltm_scheduler_state"] == expected
    assert checkpoint["ltm_scheduler_state"]["step"] == 7
    assert checkpoint["ltm_scheduler_state"]["total_steps"] == 20


def test_ltm_lr_resume_uses_saved_bounds_and_enabled_state():
    saved_args = SimpleNamespace(
        ltm_lr=1e-3,
        min_ltm_lr=1e-5,
        min_lr=1e-6,
        disable_ltm_lr_schedule=False,
    )
    configure_ltm_lr_schedule(saved_args, num_update_steps=20)
    for _ in range(7):
        advance_ltm_lr_schedule(saved_args)
    saved_state = capture_ltm_lr_scheduler_state(saved_args)

    changed_args = SimpleNamespace(
        ltm_lr=7e-4,
        min_ltm_lr=4e-4,
        min_lr=4e-4,
        disable_ltm_lr_schedule=True,
    )
    restored_lr = configure_ltm_lr_schedule(
        changed_args,
        num_update_steps=3,
        checkpoint={"ltm_scheduler_state": saved_state},
        override_schedule=False,
    )

    assert changed_args._ltm_lr_schedule_enabled is True
    assert changed_args._ltm_lr_schedule_total_steps == 20
    assert changed_args._ltm_lr_schedule_step == 7
    assert changed_args._ltm_lr_max == pytest.approx(1e-3)
    assert changed_args._ltm_lr_min == pytest.approx(1e-5)
    assert restored_lr == pytest.approx(
        get_current_ltm_lr(saved_args),
        rel=0,
        abs=1e-15,
    )


def test_epoch_boundary_checkpoint_has_clean_resume_position():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=SimpleNamespace(),
        dataloader=None,
        completed_epoch=3,
        mid_epoch_step=0,
    )

    assert checkpoint["completed_epoch"] == 3
    assert checkpoint["mid_epoch_step"] == 0
    assert "running_states" not in checkpoint
    assert checkpoint["grad_accumulation_active"] is False
    assert checkpoint["grad_state_dict"] is None


def _continuation_parser_and_args(model_path=None, resume_from_ckpt=None, epochs=3):
    defaults = {
        "mode": "train",
        "model_path": model_path,
        "resume_from_ckpt": resume_from_ckpt,
        "out_dir": "./hierarchos_model",
        "epochs": epochs,
        "train": None,
        "hf_dataset": None,
        "hf_dataset_config": None,
        "hf_dataset_split": "train",
        "text_column": None,
        "prompt_column": None,
        "completion_column": None,
        "max_length": 1024,
        "h_stride": 4,
        "training_chunk_size": 256,
        "batch_size": 64,
        "starting_lr": 1e-4,
        "min_lr": 1e-6,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "ltm_lr": 1e-3,
        "min_ltm_lr": None,
        "ltm_training_mode": "inner-update",
        "alpaca": False,
        "kayla": False,
        "compile": False,
        "force_compile": False,
        "amp": False,
        "train_prompt_tokens": True,
        "prompt_loss_weight": 1.0,
        "response_loss_weight": 1.0,
        "response_boundary_loss_weight": 1.0,
        "response_boundary_tokens": 0,
        "min_response_tokens": 1,
        "drop_empty_completions": True,
        "ponder_loss_weight": 0.01,
        "memory_gate_warmup_steps": 2000,
        "assistant_recovery": False,
        "refresh_hf_token_cache": False,
        "refresh_hf_shards": False,
        "max_ce_loss_for_backward": 0.0,
        "rwkv_channel_mix_key_clamp": 12.0,
        "rwkv_channel_mix_deepembed_clamp": 4.0,
    }
    parser = argparse.ArgumentParser()
    for key, value in defaults.items():
        parser.add_argument(f"--{key}", dest=key, default=value)
    return parser, SimpleNamespace(**defaults)


def test_finetune_forces_read_only_ltm_to_prevent_cross_batch_leakage(capsys):
    args = SimpleNamespace(ltm_training_mode="inner-update")

    assert configure_finetune_ltm_mode(args) == "read-only"
    assert args.ltm_training_mode == "read-only"
    assert "leak into unrelated batches" in capsys.readouterr().out


def test_finetune_training_mode_recursively_reactivates_loaded_base_and_adapter():
    class _LoadedBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

    class _PeftLikeWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = _LoadedBase()
            self.adapter_dropout = nn.Dropout(0.5)
            self.adapter = nn.Linear(2, 2, bias=False)

    model = _PeftLikeWrapper().eval()
    assert not model.training
    assert not model.base_model.model.training
    assert not model.adapter.training

    assert ensure_finetune_training_mode(model) is model
    assert model.training
    assert model.base_model.model.training
    assert model.adapter_dropout.training
    assert model.adapter.training


def test_cli_finetune_hydrates_shape_sensitive_runtime_defaults():
    saved_config = {
        "max_length": 8880,
        "h_stride": 7,
        "ltm_training_mode": "read-only",
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "hierarchos_config.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(saved_config, f)

        parser, args = _continuation_parser_and_args(model_path=tmp)
        args.mode = "finetune"
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"out_dir"},
        )

    assert args.max_length == 8880
    assert args.h_stride == 7
    assert args.ltm_training_mode == "read-only"


def test_cli_model_path_continuation_hydrates_saved_training_config():
    saved_config = {
        "hf_dataset": "netcat420/Experiment_0.1",
        "hf_dataset_split": "train",
        "alpaca": True,
        "max_length": 8880,
        "starting_lr": 7.5e-5,
        "min_lr": 9e-9,
        "ltm_lr": 5e-7,
        "min_ltm_lr": 1e-10,
        "compile": True,
        "force_compile": True,
        "amp": True,
        "completed_epoch": 11,
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "hierarchos_config.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(saved_config, f)

        parser, args = _continuation_parser_and_args(model_path=tmp, epochs=3)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.alpaca is True
    assert args.max_length == 8880
    assert args.starting_lr == 7.5e-5
    assert args.min_lr == 9e-9
    assert args.ltm_lr == 5e-7
    assert args.min_ltm_lr == 1e-10
    assert args.compile is True
    assert args.force_compile is True
    assert args.amp is True
    assert args.train_prompt_tokens is True
    assert args.epochs == 3
    assert args.base_completed_epoch == 11


def test_cli_model_path_continuation_prefers_checkpoint_config_over_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "hierarchos_config.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(
                {
                    "hf_dataset": "stale/dataset",
                    "max_length": 1024,
                    "completed_epoch": 3,
                },
                f,
            )
        torch.save(
            {
                "model_state_dict": {},
                "config": {
                    "hf_dataset": "fresh/dataset",
                    "max_length": 8880,
                    "completed_epoch": 9,
                },
                "completed_epoch": 9,
            },
            os.path.join(tmp, "hierarchos.pt"),
        )

        parser, args = _continuation_parser_and_args(model_path=tmp, epochs=3)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.hf_dataset == "fresh/dataset"
    assert args.max_length == 8880
    assert args.base_completed_epoch == 9


def test_cli_checkpoint_preflight_fails_before_dataset_work_on_unsafe_payload(tmp_path):
    checkpoint_path = tmp_path / "hierarchos_epoch_13.pt"
    torch.save(
        {
            "model_state_dict": {},
            "config": {"completed_epoch": 13},
            "unlisted_metadata": _UnlistedCheckpointMetadata(),
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="before dataset preparation"):
        hierarchos_cli._read_model_config_defaults(str(checkpoint_path))


def test_safe_checkpoint_loader_accepts_project_owned_rosa_resume_state(tmp_path):
    checkpoint_path = tmp_path / "hierarchos_epoch_1_step_1.pt"
    rosa_state = ROSAState.new()
    rosa_state.tokens.extend([1, 2, 3])
    torch.save(
        {
            "model_state_dict": {},
            "config": {"completed_epoch": 0},
            "running_states": (
                None,
                None,
                None,
                None,
                None,
                (None, None, None, [rosa_state]),
            ),
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_payload_compatible(
        str(checkpoint_path),
        map_location="cpu",
    )

    restored = loaded["running_states"][5][3][0]
    assert isinstance(restored, ROSAState)
    assert restored.tokens == [1, 2, 3]


def test_effective_training_config_canonicalizes_detach_zero():
    effective = capture_effective_training_config(
        SimpleNamespace(detach_every_n_steps=0)
    )

    assert effective["detach_every_n_steps"] is None


def test_assistant_recovery_defaults_target_large_assistant_sft():
    parser, args = _continuation_parser_and_args(epochs=3)
    args.assistant_recovery = True

    hierarchos_cli._apply_assistant_recovery_defaults(args, explicit_dests=set())

    assert args.alpaca is True
    assert args.epochs == 4
    assert args.starting_lr == 6e-5
    assert args.min_lr == 1e-6
    assert args.warmup_ratio == 0.03
    assert args.prompt_loss_weight == 0.10
    assert args.response_loss_weight == 1.0
    assert args.response_boundary_loss_weight == 2.0
    assert args.response_boundary_tokens == 32
    assert args.min_response_tokens == 16
    assert args.ponder_loss_weight == 0.003
    assert args.memory_gate_warmup_steps == 5000
    assert args.ltm_training_mode == "read-only"


def test_assistant_recovery_respects_explicit_overrides():
    parser, args = _continuation_parser_and_args(epochs=7)
    args.assistant_recovery = True

    hierarchos_cli._apply_assistant_recovery_defaults(
        args,
        explicit_dests={"epochs", "prompt_loss_weight", "warmup_ratio", "ltm_training_mode"},
    )

    assert args.epochs == 7
    assert args.prompt_loss_weight == 1.0
    assert args.warmup_ratio == 0.0
    assert args.response_boundary_tokens == 32
    assert args.ltm_training_mode == "inner-update"


def test_resume_hydrates_saved_ltm_training_mode():
    checkpoint = {
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "ltm_training_mode": "read-only",
        },
        "completed_epoch": 5,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "hierarchos_epoch_5.pt")
        torch.save(checkpoint, ckpt_path)

        parser, args = _continuation_parser_and_args(resume_from_ckpt=ckpt_path, epochs=7)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.ltm_training_mode == "read-only"
    assert args.resume_completed_epoch == 5


def test_ltm_training_mode_normalization():
    assert normalize_ltm_training_mode("inner") == "inner-update"
    assert normalize_ltm_training_mode("inference-like") == "read-only"
    assert ltm_inner_updates_enabled(SimpleNamespace(ltm_training_mode="inner-update")) is True
    assert ltm_inner_updates_enabled(SimpleNamespace(ltm_training_mode="read-only")) is False
    with pytest.raises(ValueError, match="ltm_training_mode"):
        normalize_ltm_training_mode("inner-updtae")
    with pytest.raises(ValueError, match="ltm_training_mode"):
        ltm_inner_updates_enabled(SimpleNamespace(ltm_training_mode="inner-updtae"))


def test_deepembed_weights_are_excluded_from_weight_decay():
    model = _FakeDeepEmbedModel()
    args = SimpleNamespace(starting_lr=1e-3, rwkv_weight_decay=0.1)

    optimizer = build_hierarchos_optimizer(model, args, torch.device("cpu"))

    decay_params = set(map(id, optimizer.param_groups[0]["params"]))
    no_decay_params = set(map(id, optimizer.param_groups[1]["params"]))
    assert id(model.h_deepemb.weight) in no_decay_params
    assert id(model.l_deepemb.weight) in no_decay_params
    assert id(model.h_deepemb.weight) not in decay_params
    assert id(model.l_deepemb.weight) not in decay_params
    assert id(model.tok_emb.weight) in decay_params


def test_raw_rwkv_matrices_receive_v2_weight_decay():
    model = _RawRWKVMatrixModel()
    args = SimpleNamespace(starting_lr=1e-3, rwkv_weight_decay=0.1)

    optimizer = build_hierarchos_optimizer(model, args, torch.device("cpu"))

    decay_params = set(map(id, optimizer.param_groups[0]["params"]))
    no_decay_params = set(map(id, optimizer.param_groups[1]["params"]))
    assert {id(model.w1), id(model.a1), id(model.r_k)} <= decay_params
    assert id(model.norm_scale) in no_decay_params
    assert args._optimizer_grouping_version == 2


def test_read_only_ltm_training_skips_inner_update_but_carries_state():
    model = _LTMModeRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=2,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode="read-only",
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert model.ltm.inner_update_calls == 0
    assert model.forward_flags
    assert all(flag["return_raw_topk_values"] is False for flag in model.forward_flags)
    assert all(flag["return_topk_indices"] is False for flag in model.forward_flags)
    assert states[5] is not None
    assert states[5][2] is not None


def test_inner_update_ltm_training_retains_legacy_fast_memory_path():
    model = _LTMModeRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=2,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode="inner-update",
        ltm_lr=1e-4,
        min_ltm_lr=1e-8,
        min_lr=1e-8,
        disable_ltm_lr_schedule=True,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    outputs, _states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert model.ltm.inner_update_calls > 0
    assert all(flag["return_raw_topk_values"] is True for flag in model.forward_flags)


def test_resume_hydration_does_not_persist_refresh_cache_flags(tmp_path):
    ckpt_path = tmp_path / "hierarchos_epoch_1_step_600.pt"
    torch.save({
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "refresh_hf_token_cache": True,
            "refresh_hf_shards": True,
            "max_ce_loss_for_backward": 10.0,
            "completed_epoch": 1,
        },
        "model_state_dict": {},
    }, ckpt_path)
    parser, args = _continuation_parser_and_args(resume_from_ckpt=str(ckpt_path))

    hierarchos_cli._hydrate_training_args_from_model_config(args, parser, explicit_dests=set())

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.refresh_hf_token_cache is False
    assert args.refresh_hf_shards is False
    assert args.max_ce_loss_for_backward == 0.0


def test_cli_resume_checkpoint_hydrates_config_without_base_epoch_offset():
    checkpoint = {
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "alpaca": True,
            "max_length": 8880,
            "starting_lr": 7.5e-5,
            "ltm_lr": 5e-7,
        },
        "completed_epoch": 11,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "epoch11.pt")
        torch.save(checkpoint, ckpt_path)

        parser, args = _continuation_parser_and_args(resume_from_ckpt=ckpt_path, epochs=14)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.alpaca is True
    assert args.max_length == 8880
    assert args.starting_lr == 7.5e-5
    assert args.ltm_lr == 5e-7
    assert args.train_prompt_tokens is True
    assert args.epochs == 14
    assert args.resume_completed_epoch == 11
    assert not hasattr(args, "base_completed_epoch")


def test_cli_resume_checkpoint_hydrates_channel_mix_clamp_defaults():
    checkpoint = {
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "rwkv_channel_mix_key_clamp": 9.0,
            "rwkv_channel_mix_deepembed_clamp": 2.5,
        },
        "completed_epoch": 5,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "hierarchos_epoch_5.pt")
        torch.save(checkpoint, ckpt_path)

        parser, args = _continuation_parser_and_args(resume_from_ckpt=ckpt_path, epochs=9)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.rwkv_channel_mix_key_clamp == 9.0
    assert args.rwkv_channel_mix_deepembed_clamp == 2.5
    assert args.resume_completed_epoch == 5


def test_cli_resume_checkpoint_rejects_non_advancing_epoch_target():
    args = SimpleNamespace(
        mode="train",
        resume_from_ckpt="epoch11.pt",
        epochs=3,
        resume_completed_epoch=11,
    )

    try:
        hierarchos_cli._validate_resume_epoch_target(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected non-advancing resume target to exit")


def test_cli_resume_checkpoint_preserves_explicit_colab_overrides():
    checkpoint = {
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "alpaca": True,
            "max_length": 1024,
            "starting_lr": 7.5e-5,
            "min_lr": 9e-9,
            "ltm_lr": 5e-7,
            "min_ltm_lr": 1e-10,
            "train_prompt_tokens": False,
            "rwkv_channel_mix_key_clamp": 12.0,
            "rwkv_channel_mix_deepembed_clamp": 4.0,
        },
        "completed_epoch": 11,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "hierarchos_epoch_11.pt")
        torch.save(checkpoint, ckpt_path)

        parser, args = _continuation_parser_and_args(resume_from_ckpt=ckpt_path, epochs=14)
        args.max_length = 8880
        args.starting_lr = 1e-5
        args.min_lr = 1e-8
        args.ltm_lr = 1e-5
        args.min_ltm_lr = 1e-9
        args.train_prompt_tokens = True
        args.rwkv_channel_mix_key_clamp = 8.0
        args.rwkv_channel_mix_deepembed_clamp = 2.0
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={
                "epochs",
                "out_dir",
                "max_length",
                "starting_lr",
                "min_lr",
                "ltm_lr",
                "min_ltm_lr",
                "train_prompt_tokens",
                "rwkv_channel_mix_key_clamp",
                "rwkv_channel_mix_deepembed_clamp",
            },
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.alpaca is True
    assert args.max_length == 8880
    assert args.starting_lr == 1e-5
    assert args.min_lr == 1e-8
    assert args.ltm_lr == 1e-5
    assert args.min_ltm_lr == 1e-9
    assert args.rwkv_channel_mix_key_clamp == 8.0
    assert args.rwkv_channel_mix_deepembed_clamp == 2.0
    assert args.train_prompt_tokens is True
    assert args.resume_completed_epoch == 11


def test_cli_resume_preserves_saved_masked_prompt_objective_by_default():
    checkpoint = {
        "config": {"train_prompt_tokens": False},
        "completed_epoch": 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "masked.pt")
        torch.save(checkpoint, ckpt_path)
        parser, args = _continuation_parser_and_args(
            resume_from_ckpt=ckpt_path,
            epochs=2,
        )
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )
    assert args.train_prompt_tokens is False


def test_inference_sanitization_still_clears_transient_ltm():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)

    state = sanitize_model_state_dict(model)

    assert torch.count_nonzero(state["ltm.fast_vals"]) == 0
    assert torch.count_nonzero(state["ltm._mom_vals"]) == 0
    assert torch.count_nonzero(state["ltm.timestamps"]) == 0
    assert torch.count_nonzero(state["ltm.sources"]) == 0


def test_safe_inference_checkpoint_save_clears_transient_ltm_payload(tmp_path):
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "hierarchos.pt"

    ok = save_training_checkpoint_if_finite(
        {
            "model_state_dict": sanitize_model_state_dict(model),
            "config": dict(model.config),
            "training_complete": True,
        },
        str(path),
        model,
        optimizer=None,
    )

    assert ok is True
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["model_state_dict"]
    assert torch.count_nonzero(state["ltm.fast_vals"]) == 0
    assert torch.count_nonzero(state["ltm._mom_vals"]) == 0
    assert torch.count_nonzero(state["ltm.timestamps"]) == 0
    assert torch.count_nonzero(state["ltm.sources"]) == 0


def test_restore_model_grad_state_round_trips_pending_accumulation():
    model = _FakeTrainModel()
    grad_state = {"proj.weight": torch.full_like(model.proj.weight, 9.0)}

    restored = restore_model_grad_state(model, grad_state, torch.device("cpu"))

    assert restored is True
    assert torch.equal(model.proj.weight.grad, torch.full_like(model.proj.weight, 9.0))


def test_restore_model_grad_state_rejects_nonfinite_pending_accumulation():
    model = _FakeTrainModel()
    grad_state = {"proj.weight": torch.full_like(model.proj.weight, float("nan"))}

    with pytest.raises(RuntimeError, match="cannot be resumed safely"):
        restore_model_grad_state(model, grad_state, torch.device("cpu"))


def test_persisted_running_state_is_required_for_exact_mid_epoch_resume():
    args = SimpleNamespace(persist_state=True)
    ltm_state = (
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.long),
        [],
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1),
    )
    valid = {
        "running_states": (
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
            ltm_state,
        ),
    }
    assert validate_exact_running_states(valid, args, 2, "checkpoint.pt") is True

    for invalid in (
        {},
        {"running_states": None},
        {"running_states": (None,) * 5},
        {
            "running_states": (
                torch.tensor([float("nan")]),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                ltm_state,
            ),
        },
        {
            "running_states": (
                torch.ones(1),
                None,
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                ltm_state,
            ),
        },
        {
            "running_states": (
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                ltm_state[:6],
            ),
        },
    ):
        with pytest.raises(RuntimeError, match="running state|running_states"):
            validate_exact_running_states(invalid, args, 2, "checkpoint.pt")

    assert validate_exact_running_states(
        {},
        SimpleNamespace(persist_state=False),
        2,
        "checkpoint.pt",
    ) is False
    assert validate_exact_running_states({}, args, 0, "checkpoint.pt") is False


def test_declared_gradient_accumulation_restores_strictly_or_fails_closed():
    model = _FakeTrainModel()
    grad = torch.full_like(model.proj.weight, 3.0)
    args = SimpleNamespace(
        reset_optimizer_state=False,
        override_scheduling=False,
        accumulation_normalization="microbatch",
    )
    valid = {
        "grad_accumulation_active": True,
        "grad_state_dict": {"proj.weight": grad},
        "grad_state_keys": ("proj.weight",),
        "accumulation_state": {
            "normalization": "microbatch",
            "weighted_token_mass": 0.0,
        },
    }
    assert restore_checkpoint_gradient_accumulation(
        model,
        valid,
        args,
        torch.device("cpu"),
    ) is True
    assert torch.equal(model.proj.weight.grad, grad)

    invalid_checkpoints = [
        {**valid, "grad_state_dict": None},
        {**valid, "grad_state_dict": "malformed"},
        {**valid, "grad_accumulation_active": False},
        {**valid, "grad_accumulation_active": 1},
        {**valid, "grad_state_keys": None},
        {**valid, "grad_state_keys": ("proj.weight", "proj.bias")},
        {
            **valid,
            "grad_state_dict": {"unknown.weight": grad},
            "grad_state_keys": ("unknown.weight",),
        },
        {**valid, "accumulation_state": None},
        {**valid, "accumulation_state": {"weighted_token_mass": 0.0}},
        {**valid, "accumulation_state": {"normalization": "microbatch"}},
        {
            **valid,
            "accumulation_state": {
                "normalization": "microbatch",
                "weighted_token_mass": float("nan"),
            },
        },
        {
            **valid,
            "accumulation_state": {
                "normalization": "microbatch",
                "weighted_token_mass": float("inf"),
            },
        },
    ]
    for invalid in invalid_checkpoints:
        with pytest.raises(RuntimeError):
            restore_checkpoint_gradient_accumulation(
                model,
                invalid,
                args,
                torch.device("cpu"),
            )

    reset_args = SimpleNamespace(
        reset_optimizer_state=True,
        override_scheduling=False,
        accumulation_normalization="microbatch",
    )
    assert restore_checkpoint_gradient_accumulation(
        model,
        invalid_checkpoints[0],
        reset_args,
        torch.device("cpu"),
    ) is False

    weighted_args = SimpleNamespace(
        reset_optimizer_state=False,
        override_scheduling=False,
        accumulation_normalization="weighted-token",
    )
    nonpositive_weighted = {
        **valid,
        "accumulation_state": {
            "normalization": "weighted-token",
            "weighted_token_mass": 0.0,
        },
    }
    with pytest.raises(RuntimeError, match="positive accumulated token mass"):
        restore_checkpoint_gradient_accumulation(
            model,
            nonpositive_weighted,
            weighted_args,
            torch.device("cpu"),
        )


def test_training_state_finite_rejects_poisoned_optimizer_state():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.ones(1, 2)
    loss = model.proj(x).sum()
    loss.backward()
    optimizer.step()

    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"].view(-1)[0] = float("nan")

    assert training_state_is_finite(model, optimizer) is False


def test_training_state_finite_rejects_poisoned_pending_grad():
    model = _FakeTrainModel()
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    model.proj.weight.grad.view(-1)[0] = float("nan")

    assert training_state_is_finite(model, include_grads=True) is False


def test_clip_gradients_and_check_rejects_nonfinite_gradients():
    model = _FakeTrainModel()
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    model.proj.bias.grad = torch.ones_like(model.proj.bias)
    model.proj.weight.grad.view(-1)[0] = float("inf")
    model.proj.weight.grad.view(-1)[1] = float("-inf")
    model.proj.bias.grad.view(-1)[0] = float("nan")

    ok, issue = _clip_gradients_and_check(model, max_norm=1.0)

    assert ok is False
    assert issue is not None
    assert "Top non-finite gradient tensors" in issue
    assert "proj.weight" in issue
    assert "proj.bias" in issue


def test_clip_gradients_and_check_saturates_huge_finite_gradients():
    model = _FakeTrainModel()
    model.proj.weight.grad = torch.full_like(model.proj.weight, 1e30)
    model.proj.bias.grad = torch.full_like(model.proj.bias, -1e30)

    ok, issue = _clip_gradients_and_check(model, max_norm=1.0)

    assert ok is True
    assert issue is not None or torch.isfinite(model.proj.weight.grad).all()
    assert torch.isfinite(model.proj.weight.grad).all()
    assert torch.isfinite(model.proj.bias.grad).all()
    assert model.proj.weight.grad.abs().max().item() <= 1.0
    assert model.proj.bias.grad.abs().max().item() <= 1.0


def test_gradient_sanitizer_preserves_finite_gradients_for_global_norm_clip():
    model = _FakeTrainModel()
    finite_grad = torch.tensor([[2.0, -3.0], [0.5, 1.5]])
    model.proj.weight.grad = finite_grad.clone()

    cleaned = _sanitize_gradient_nonfinite_(model, max_abs=1.0)

    assert cleaned == 0
    torch.testing.assert_close(model.proj.weight.grad, finite_grad)

    ok, total_norm = _clip_gradients_and_check(model, max_norm=1.0)

    assert ok is True
    assert total_norm.item() > 1.0
    expected = finite_grad * (1.0 / (finite_grad.norm().item() + 1e-6))
    torch.testing.assert_close(model.proj.weight.grad, expected)


def test_model_nonfinite_sanitizer_repairs_parameters_and_buffers():
    model = _FakeTrainModel()
    model.proj.weight.data.view(-1)[0] = float("inf")
    model.proj.weight.data.view(-1)[1] = float("-inf")
    model.proj.bias.data.view(-1)[0] = float("nan")
    model.ltm.fast_vals[0, 0] = float("inf")

    cleaned = _sanitize_model_nonfinite_(model)

    assert cleaned == 3
    assert model.proj.weight.data.view(-1)[0].item() == 1.0
    assert model.proj.weight.data.view(-1)[1].item() == -1.0
    assert model.proj.bias.data.view(-1)[0].item() == 0.0
    assert torch.isinf(model.ltm.fast_vals[0, 0])


def test_model_sanitizer_preserves_intentional_ltm_neg_inf_buffer():
    model = _FakeTrainModel()

    cleaned = _sanitize_model_nonfinite_(model)

    assert cleaned == 0
    assert torch.isneginf(model.ltm.neg_inf)
    assert training_state_is_finite(model) is True


def test_model_startup_magnitude_clamp_repairs_all_weights_and_buffers():
    model = _FakeTrainModel()
    model.proj.weight.data.fill_(123.0)
    model.proj.bias.data.fill_(-123.0)
    model.register_buffer("finite_buffer", torch.tensor([77.0]))

    clamped = _clamp_model_finite_magnitude_(model, 0.75)

    assert clamped == model.proj.weight.numel() + model.proj.bias.numel() + 1
    assert model.proj.weight.data.abs().max().item() == 0.75
    assert model.proj.bias.data.abs().max().item() == 0.75
    assert model.finite_buffer.abs().max().item() == 0.75


def test_train_step_skips_nonfinite_loss_before_backward():
    model = _NaNLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is None
    assert states == (None, None, None, None, None, None)
    assert args._train_step_had_nonfinite is True
    assert model.weight.grad is None
    assert model.reset_called is True


@pytest.mark.parametrize(
    "carrier_name",
    ("h_state", "l_state", "prev_context", "target_context", "drift_state"),
)
def test_train_step_rejects_nonfinite_recurrent_carrier_before_rewrite(carrier_name):
    model = _NonfiniteCarrierModel(carrier_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }
    before = model.weight.detach().clone()

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is None
    assert states == (None, None, None, None, None, None)
    assert args._train_step_had_nonfinite is True
    assert model.weight.grad is None
    assert torch.equal(model.weight.detach(), before)
    assert model.reset_called is True


def test_full_sample_train_step_rejects_nonfinite_terminal_carrier_before_backward():
    model = _NonfiniteCarrierModel("h_state")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=2,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        full_sample_bptt=True,
        full_sample_activation_checkpointing=False,
        persist_state=False,
        ltm_training_mode="read-only",
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }
    before = model.weight.detach().clone()

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is None
    assert states == (None, None, None, None, None, None)
    assert args._train_step_had_nonfinite is True
    assert args._train_step_had_backward is False
    assert model.weight.grad is None
    assert torch.equal(model.weight.detach(), before)
    assert model.reset_called is True


def test_train_step_rejects_nonfinite_gradient_before_optimizer_step():
    model = _InfGradModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    before = model.weight.detach().clone()
    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is None
    assert states == (None, None, None, None, None, None)
    assert args._train_step_had_nonfinite is True
    assert model.weight.grad is None
    assert torch.equal(model.weight.detach(), before)
    assert model.reset_called is True


def test_train_step_preserves_prior_accumulation_on_malformed_labels():
    model = _FiniteLinearLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prior_grad = torch.full_like(model.weight, 0.75)
    model.weight.grad = prior_grad.clone()
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[1.0, 1.0, float("nan"), 1.0]]),
    }

    with pytest.raises(RuntimeError, match="earlier microbatch"):
        train_step(
            model,
            batch,
            optimizer,
            scaler=None,
            accumulation_steps=2,
            step=1,
            args=args,
            running_states=(None, None, None, None, None, None),
        )

    torch.testing.assert_close(model.weight.grad, prior_grad)
    assert args._train_step_had_nonfinite is True


def test_train_step_preserves_prior_accumulation_on_oom():
    model = _OOMModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prior_grad = torch.full_like(model.weight, 0.75)
    model.weight.grad = prior_grad.clone()
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="out-of-memory failure"):
        train_step(
            model,
            batch,
            optimizer,
            scaler=None,
            accumulation_steps=2,
            step=1,
            args=args,
            running_states=(None, None, None, None, None, None),
        )

    torch.testing.assert_close(model.weight.grad, prior_grad)


def test_train_step_caps_finite_loss_explosion_but_preserves_commitment_gradient():
    model = _HighFiniteLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        max_ce_loss_for_backward=10.0,
        max_commitment_cost_for_backward=2.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    before = model.weight.detach().clone()
    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert outputs["loss"].item() == 20.0
    assert outputs["commitment_cost"].item() == 20.0
    assert args._train_step_had_nonfinite is False
    assert not torch.equal(model.weight.detach(), before)
    assert model.weight.detach().item() < before.item()
    assert states == (None, None, None, None, None, None)


def test_train_step_default_does_not_cap_random_vocab_ce():
    model = _HighFiniteLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=10.0,
        max_commitment_cost_for_backward=2.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    before = model.weight.detach().clone()
    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert outputs["loss"].item() == 20.0
    assert args._train_step_had_nonfinite is False
    assert not torch.equal(model.weight.detach(), before)
    assert states == (None, None, None, None, None, None)


def test_train_step_flushes_tail_accumulation_with_real_divisor():
    model = _FiniteLinearLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=10.0,
        max_ce_loss_for_backward=0.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=4,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
        force_optimizer_step=False,
        accumulation_divisor=2,
    )

    assert outputs is not None
    assert states == (None, None, None, None, None, None)
    assert args._optimizer_step_was_taken is False
    assert model.weight.grad is not None
    assert abs(model.weight.grad.item() - 0.5) < 1e-6

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=4,
        step=1,
        args=args,
        running_states=states,
        force_optimizer_step=True,
        accumulation_divisor=2,
    )

    assert outputs is not None
    assert args._optimizer_step_was_taken is True
    assert model.weight.grad is None
    assert abs(model.weight.item() - 0.9) < 1e-6


def test_weighted_token_accumulation_matches_combined_loss_mass():
    model = _InputScaledLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=100.0,
        accumulation_normalization="weighted-token",
    )
    first = {
        "input_ids": torch.tensor([[2, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[2, 1, 1]], dtype=torch.long),
        "loss_weights": torch.tensor([[0.0, 0.5, 0.5]]),
    }
    second = {
        "input_ids": torch.tensor([[4, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[4, 1, 1]], dtype=torch.long),
        "loss_weights": torch.tensor([[0.0, 1.5, 1.5]]),
    }
    assert supervised_weight_mass(first) == 1.0
    assert supervised_weight_mass(second) == 3.0

    _outputs, states = train_step(
        model,
        first,
        optimizer,
        scaler=None,
        accumulation_steps=2,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )
    assert args._accumulation_weighted_token_mass == 1.0
    _outputs, _states = train_step(
        model,
        second,
        optimizer,
        scaler=None,
        accumulation_steps=2,
        step=1,
        args=args,
        running_states=states,
    )

    # Combined objective gradient = (1*2 + 3*4) / (1+3) = 3.5.
    assert abs(model.weight.item() - (1.0 - 3.5)) < 1e-6
    assert args._accumulation_weighted_token_mass == 0.0


def test_train_step_passes_one_token_label_lookahead_across_chunks():
    model = _BoundaryRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=3,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long),
        "labels": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
    }

    outputs, _states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert [entry[0] for entry in model.seen] == [0, 3, 6]
    assert [entry[1].shape[1] for entry in model.seen] == [3, 3, 1]
    assert [entry[2].shape[1] for entry in model.seen] == [4, 4, 1]
    assert model.seen[0][2].tolist() == [[10, 11, 12, 13]]
    assert model.seen[1][2].tolist() == [[13, 14, 15, 16]]
    assert model.seen[2][2].tolist() == [[16]]


def test_train_step_rejects_masked_active_labels_for_alpaca_all_token_recovery():
    model = _BoundaryRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        alpaca=True,
        train_prompt_tokens=True,
        strict_all_token_loss=True,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        "labels": torch.tensor([[10, -100, 12, 13]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    try:
        train_step(
            model,
            batch,
            optimizer,
            scaler=None,
            accumulation_steps=1,
            step=0,
            args=args,
            running_states=(None, None, None, None, None, None),
        )
    except RuntimeError as exc:
        assert "All-token loss audit failed" in str(exc)
    else:
        raise AssertionError("masked active Alpaca labels should fail the recovery audit")

    assert model.seen == []


def test_ltm_transient_recovery_resets_fast_and_saturates_momentum():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm._mom_vals[0, 0] = float("inf")
    model.ltm._mom_vals[0, 1] = float("-inf")
    model.ltm._mom_vals[1, 0] = float("nan")
    model.ltm.timestamps[0] = float("inf")
    model.ltm.sources[0] = 2

    cleaned = _sanitize_model_transient_state_(model, max_abs=0.75)

    assert cleaned > 0
    assert torch.count_nonzero(model.ltm.fast_vals) == 0
    assert model.ltm._mom_vals[0, 0].item() == 0.75
    assert model.ltm._mom_vals[0, 1].item() == -0.75
    assert model.ltm._mom_vals[1, 0].item() == 0.0
    assert model.ltm._mom_vals[1, 1].item() == 4.0
    assert model.ltm.timestamps[0].item() == 0.0


def test_checkpoint_save_allows_clean_state_and_writes_file():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clean.pt")

        saved = save_training_checkpoint_if_finite(
            {"model_state_dict": model.state_dict(), "training_complete": False},
            path,
            model,
            optimizer,
        )
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        assert saved is True
        assert os.path.exists(path)
        assert os.path.exists(path + ".sha256")
        assert torch.all(model.ltm.fast_vals == 3.0)
        assert torch.all(loaded["model_state_dict"]["ltm.fast_vals"] == 3.0)
        verified = load_checkpoint_payload_compatible(path, map_location="cpu")
        assert verified["training_complete"] is False


def test_checkpoint_checksum_rejects_post_save_corruption(tmp_path):
    path = str(tmp_path / "corrupt.pt")
    from hierarchos.utils.checkpoint import save_checkpoint_safely

    save_checkpoint_safely({"value": torch.arange(4)}, path)
    with open(path, "r+b") as checkpoint_file:
        checkpoint_file.seek(-1, os.SEEK_END)
        byte = checkpoint_file.read(1)
        checkpoint_file.seek(-1, os.SEEK_END)
        checkpoint_file.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        load_checkpoint_payload_compatible(path, map_location="cpu")


def test_checkpoint_save_rejects_poisoned_gradient_without_mutating_it():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    model.proj.weight.grad.view(-1)[0] = float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_grad.pt")

        with pytest.raises(RuntimeError, match="non-finite learned/gradient state"):
            save_training_checkpoint_if_finite({"bad": torch.tensor(1)}, path, model, optimizer)

        assert not os.path.exists(path)
        assert torch.isnan(model.proj.weight.grad.view(-1)[0])


def test_checkpoint_save_rejects_poisoned_optimizer_state_without_mutating_it():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.ones(1, 2)
    loss = model.proj(x).sum()
    loss.backward()
    optimizer.step()
    next(iter(optimizer.state.values()))["exp_avg"].view(-1)[0] = float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_optimizer.pt")

        with pytest.raises(RuntimeError, match="non-finite optimizer state"):
            save_training_checkpoint_if_finite({"bad": torch.tensor(1)}, path, model, optimizer)

        assert not os.path.exists(path)
        assert torch.isnan(next(iter(optimizer.state.values()))["exp_avg"].view(-1)[0])


def test_checkpoint_save_drops_poisoned_transient_running_state():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "running_states": (torch.tensor([float("nan")]), None, None, None, None, None),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_running_state.pt")

        saved = save_training_checkpoint_if_finite(checkpoint, path, model, optimizer)
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        assert saved is True
        assert os.path.exists(path)
        assert loaded["running_states"] is None


def test_checkpoint_save_rejects_nonfinite_live_learned_weight():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.data.view(-1)[0] = float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_weight.pt")
        with pytest.raises(RuntimeError, match="non-finite learned/gradient state"):
            save_training_checkpoint_if_finite({"model_state_dict": model.state_dict()}, path, model, optimizer)

        assert not os.path.exists(path)
        assert torch.isnan(model.proj.weight.data.view(-1)[0])


def test_checkpoint_save_rejects_nonfinite_python_scheduler_scalar():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = {"scheduler_state_dict": {"_last_lr": [float("nan")]}}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_scheduler.pt")
        with pytest.raises(RuntimeError, match="non-finite payload state"):
            save_training_checkpoint_if_finite(checkpoint, path, model, optimizer)

        assert not os.path.exists(path)


def test_checkpoint_save_refreshes_stale_poisoned_model_snapshot():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stale_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stale_state["proj.weight"].view(-1)[0] = float("inf")
    checkpoint = {"model_state_dict": stale_state}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stale_model_snapshot.pt")

        saved = save_training_checkpoint_if_finite(checkpoint, path, model, optimizer)
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        assert saved is True
        assert os.path.exists(path)
        assert torch.isfinite(loaded["model_state_dict"]["proj.weight"]).all()
        assert torch.equal(loaded["model_state_dict"]["proj.weight"], model.state_dict()["proj.weight"])


def test_length_grouped_sampler_state_restores_epoch_order():
    dataset = TensorDataset(torch.arange(12))
    sampler = LengthGroupedBatchSampler(
        lengths=list(range(1, 13)),
        batch_size=3,
        shuffle=True,
        seed=123,
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    sampler.set_epoch(4)
    expected_order = list(iter(sampler))
    saved_state = capture_dataloader_state(dataloader)

    sampler.seed = 999
    sampler.set_epoch(0)
    restore_dataloader_state(dataloader, saved_state)

    assert sampler.seed == 123
    assert sampler.epoch == 4
    assert list(iter(sampler)) == expected_order


def test_epoch_shuffle_sampler_state_restores_epoch_order():
    dataset = TensorDataset(torch.arange(10))
    sampler = EpochShuffleSampler(dataset, shuffle=True, seed=321)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    sampler.set_epoch(3)
    expected_order = list(iter(sampler))
    saved_state = capture_dataloader_state(dataloader)

    sampler.seed = 111
    sampler.set_epoch(0)
    restore_dataloader_state(dataloader, saved_state)

    assert sampler.seed == 321
    assert sampler.epoch == 3
    assert list(iter(sampler)) == expected_order


@pytest.mark.parametrize("preserve_order", [False, True])
def test_length_grouped_resume_cursor_preserves_exact_remaining_batches(
    preserve_order,
):
    dataset = TensorDataset(torch.arange(23))
    sampler = LengthGroupedBatchSampler(
        lengths=list(range(1, 24)),
        batch_size=4,
        shuffle=True,
        drop_last=False,
        bucket_size=8,
        preserve_order=preserve_order,
        seed=987,
    )
    sampler.set_epoch(5)
    expected = list(iter(sampler))
    loader = DataLoader(dataset, batch_sampler=sampler)

    assert set_dataloader_start_batch(loader, 3) is True
    assert list(iter(sampler)) == expected[3:]
    actual_values = [
        batch[0].reshape(-1).tolist()
        for batch in loader
    ]
    expected_values = [
        torch.tensor(indices).reshape(-1).tolist()
        for indices in expected[3:]
    ]
    assert actual_values == expected_values

    assert set_dataloader_start_batch(loader, 0) is True
    assert list(iter(sampler)) == expected


def test_epoch_shuffle_resume_cursor_avoids_fetching_skipped_records():
    class RecordingDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
            self.fetched = []

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            self.fetched.append(int(idx))
            return int(idx)

    dataset = RecordingDataset(17)
    sampler = EpochShuffleSampler(dataset, shuffle=True, seed=321)
    sampler.set_epoch(7)
    expected_indices = list(iter(sampler))
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, num_workers=0)

    assert set_dataloader_start_batch(loader, 2) is True
    actual = [
        int(value)
        for batch in loader
        for value in batch.tolist()
    ]
    assert actual == expected_indices[6:]
    assert dataset.fetched == expected_indices[6:]
    assert set(dataset.fetched).isdisjoint(expected_indices[:6])


def test_iterable_resume_fallback_skips_before_device_prefetch_boundary():
    host_materialized = []

    def host_batches():
        for value in range(6):
            host_materialized.append(value)
            yield value

    # The returned iterable is what CUDABatchPrefetcher receives. It may need to
    # consume host records for a third-party/iterable loader, but discarded
    # records never cross this boundary into H2D.
    device_prefetch_input = host_batches_from_resume(
        host_batches(),
        start_batch=4,
        sampler_cursor_applied=False,
    )
    assert list(device_prefetch_input) == [4, 5]
    assert host_materialized == [0, 1, 2, 3, 4, 5]


def test_host_epoch_bound_prevents_scheduler_geometry_overrun():
    downstream = list(host_batches_from_resume(
        range(20),
        start_batch=3,
        sampler_cursor_applied=False,
        total_batches=8,
    ))
    assert downstream == [3, 4, 5, 6, 7]


def test_exact_resume_identity_rejects_cursor_geometry_change():
    class _Tokenizer:
        def __len__(self):
            return 8

    dataset = TensorDataset(torch.arange(8))
    loader = DataLoader(dataset, batch_size=2)
    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        train_prompt_tokens=True,
        prompt_loss_weight=1.0,
        response_loss_weight=1.0,
        response_boundary_loss_weight=1.0,
        response_boundary_tokens=0,
        min_response_tokens=1,
        drop_empty_completions=True,
        _optimizer_grouping_version=2,
        _token_cache_identity={
            "cache_key": "abc",
            "ordered_record_sha256": "1" * 64,
            "samples": 8,
        },
    )
    architecture = {"architecture_revision": "legacy-v8"}
    saved_identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        architecture,
    )
    validate_exact_resume_identity(
        {
            "run_identity": saved_identity,
            "mid_epoch_step": 2,
            "data_state": {"sampler": {"class": "fixture"}},
            "rng_state": capture_rng_state(),
        },
        saved_identity,
        "checkpoint.pt",
    )

    args.batch_size = 4
    changed_identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        architecture,
    )
    with pytest.raises(RuntimeError, match="batch_size"):
        validate_exact_resume_identity(
            {"run_identity": saved_identity},
            changed_identity,
            "checkpoint.pt",
        )


def test_exact_resume_tokenizer_behavior_v2_and_legacy_compatibility():
    class _Backend:
        def __init__(self, normalizer):
            self.normalizer = normalizer

        def to_str(self):
            return '{"normalizer":{"type":"' + self.normalizer + '"}}'

    class _Tokenizer:
        special_tokens_map = {"eos_token": "<eos>"}

        def __init__(self, normalizer):
            self.backend_tokenizer = _Backend(normalizer)

        def __len__(self):
            return 3

        def get_vocab(self):
            return {"<eos>": 0, "alpha": 1, "beta": 2}

    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2)
    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        _optimizer_grouping_version=2,
        _token_cache_identity={
            "cache_key": "abc",
            "ordered_record_sha256": "1" * 64,
            "samples": 8,
        },
    )
    trained = _Tokenizer("NFC")
    changed = _Tokenizer("NFKC")
    args._tokenizer_identity = tokenizer_identity(trained)
    saved_v2 = build_exact_resume_identity(
        args,
        trained,
        loader,
        len(loader),
        {"architecture_revision": "legacy-v8"},
    )
    args._tokenizer_identity = tokenizer_identity(changed)
    changed_v2 = build_exact_resume_identity(
        args,
        changed,
        loader,
        len(loader),
        {"architecture_revision": "legacy-v8"},
    )
    with pytest.raises(RuntimeError, match="behavior_sha256_v2"):
        validate_exact_resume_identity(
            {"run_identity": saved_v2, "mid_epoch_step": 2},
            changed_v2,
            "checkpoint.pt",
        )

    args._tokenizer_identity = tokenizer_identity(trained)
    args._tokenizer_identity.pop("behavior_sha256_v2")
    saved_legacy = build_exact_resume_identity(
        args,
        trained,
        loader,
        len(loader),
        {"architecture_revision": "legacy-v8"},
    )
    assert validate_exact_resume_identity(
        {
            "run_identity": saved_legacy,
            "mid_epoch_step": 2,
            "data_state": {"sampler": {"class": "fixture"}},
            "rng_state": capture_rng_state(),
        },
        changed_v2,
        "checkpoint.pt",
    ) is True


def test_mid_epoch_exact_resume_requires_and_strictly_restores_replay_state():
    class _Tokenizer:
        def __len__(self):
            return 8

    dataset = TensorDataset(torch.arange(8))
    sampler = EpochShuffleSampler(dataset, shuffle=True, seed=321)
    sampler.set_epoch(3)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)
    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        _optimizer_grouping_version=2,
        _token_cache_identity={
            "cache_key": "abc",
            "ordered_record_sha256": "1" * 64,
            "samples": 8,
        },
    )
    identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        {"architecture_revision": "legacy-v8"},
    )
    data_state = capture_dataloader_state(loader)
    rng_state = capture_rng_state()
    checkpoint = {
        "run_identity": identity,
        "mid_epoch_step": 2,
        "data_state": data_state,
        "rng_state": rng_state,
    }
    assert validate_exact_resume_identity(
        checkpoint,
        identity,
        "checkpoint.pt",
    ) is True

    for invalid_data_state in (None, "not-a-mapping", {}):
        invalid = dict(checkpoint, data_state=invalid_data_state)
        with pytest.raises(RuntimeError, match="dataloader state"):
            validate_exact_resume_identity(invalid, identity, "checkpoint.pt")
    for invalid_rng_state in (None, "not-a-mapping", {"torch": rng_state["torch"]}):
        invalid = dict(checkpoint, rng_state=invalid_rng_state)
        with pytest.raises(RuntimeError, match="RNG state"):
            validate_exact_resume_identity(invalid, identity, "checkpoint.pt")

    sampler.seed = 999
    sampler.epoch = 9
    torch.rand(4)
    restore_dataloader_state(loader, data_state, strict=True)
    assert sampler.seed == 321
    assert sampler.epoch == 3
    assert restore_rng_state(rng_state, strict=True) is True
    assert torch.equal(torch.random.get_rng_state(), rng_state["torch"])

    partial_data_state = dict(data_state)
    partial_data_state.pop("sampler")
    partial_data_state["unrelated"] = {}
    with pytest.raises(RuntimeError, match="missing saved sampler state"):
        restore_dataloader_state(loader, partial_data_state, strict=True)

    corrupt_rng_state = dict(rng_state, torch="not-a-tensor")
    with pytest.raises(RuntimeError, match="Could not restore RNG state"):
        restore_rng_state(corrupt_rng_state, strict=True)

    # At an epoch boundary no cursor is replayed, so legacy/restart paths retain
    # their established compatibility with absent transient replay metadata.
    assert validate_exact_resume_identity(
        {"run_identity": identity, "mid_epoch_step": 0},
        identity,
        "checkpoint.pt",
    ) is True


def test_strict_rng_restore_requires_cuda_state_on_cuda_runtime(monkeypatch):
    rng_state = capture_rng_state()
    cuda_restores = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda state: cuda_restores.append(state),
    )

    with pytest.raises(RuntimeError, match="every CUDA device"):
        restore_rng_state(rng_state, strict=True, require_cuda=True)

    cuda_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    complete = dict(rng_state, cuda_all=cuda_state)
    assert restore_rng_state(
        complete,
        strict=True,
        require_cuda=True,
    ) is True
    assert len(cuda_restores) == 1
    assert torch.equal(cuda_restores[0][0], cuda_state[0])


def test_strict_dataloader_restore_tracks_loader_generator_and_targets():
    generator = torch.Generator().manual_seed(77)
    loader = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    state = capture_dataloader_state(loader)
    assert "dataloader" in state
    generator.manual_seed(999)
    restore_dataloader_state(loader, state, strict=True)
    assert torch.equal(generator.get_state(), state["dataloader"]["generator_state"])

    missing_loader_state = dict(state)
    missing_loader_state.pop("dataloader")
    with pytest.raises(RuntimeError, match="missing saved dataloader state"):
        restore_dataloader_state(loader, missing_loader_state, strict=True)

    targetless = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
    with pytest.raises(RuntimeError, match="no runtime generator target"):
        restore_dataloader_state(
            targetless,
            {
                "dataloader": {
                    "class": "DataLoader",
                    "generator_state": generator.get_state(),
                },
            },
            strict=True,
        )


def test_exact_resume_identity_binds_scheduler_closure_unless_rebuilt():
    class _Tokenizer:
        def __len__(self):
            return 8

    dataset = TensorDataset(torch.arange(8))
    loader = DataLoader(dataset, batch_size=2)
    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
        disable_lr_schedule=False,
        ltm_lr=1e-4,
        min_ltm_lr=1e-6,
        disable_ltm_lr_schedule=False,
        _optimizer_grouping_version=2,
        _token_cache_identity={
            "cache_key": "abc",
            "ordered_record_sha256": "1" * 64,
            "samples": 8,
        },
    )
    architecture = {"architecture_revision": "legacy-v8"}
    saved_identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        architecture,
    )

    args.min_lr = 2e-5
    changed_identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        architecture,
    )
    with pytest.raises(RuntimeError, match="min_lr.*rebuild-lr-schedule"):
        validate_exact_resume_identity(
            {"run_identity": saved_identity},
            changed_identity,
            "checkpoint.pt",
        )

    assert validate_exact_resume_identity(
        {"run_identity": saved_identity},
        changed_identity,
        "checkpoint.pt",
        allow_schedule_rebuild=True,
    ) is True


def test_old_identity_promotes_saved_effective_schedule_for_comparison():
    class _Tokenizer:
        def __len__(self):
            return 8

    dataset = TensorDataset(torch.arange(8))
    loader = DataLoader(dataset, batch_size=2)
    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
        disable_lr_schedule=False,
        ltm_lr=1e-4,
        min_ltm_lr=1e-6,
        disable_ltm_lr_schedule=False,
        _optimizer_grouping_version=2,
        _token_cache_identity={
            "cache_key": "abc",
            "ordered_record_sha256": "1" * 64,
            "samples": 8,
        },
    )
    architecture = {"architecture_revision": "legacy-v8"}
    current_identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        architecture,
    )
    old_identity = dict(current_identity)
    old_identity["objective"] = {
        key: value
        for key, value in current_identity["objective"].items()
        if key not in {
            "starting_lr",
            "min_lr",
            "warmup_steps",
            "warmup_ratio",
            "disable_lr_schedule",
            "ltm_lr",
            "min_ltm_lr",
            "disable_ltm_lr_schedule",
        }
    }

    assert validate_exact_resume_identity(
        {
            "run_identity": old_identity,
            "effective_training_config": {
                "starting_lr": 1e-3,
                "min_lr": 1e-5,
                "warmup_steps": 2,
                "warmup_ratio": 0.0,
                "disable_lr_schedule": False,
                "ltm_lr": 1e-4,
                "min_ltm_lr": 1e-6,
                "disable_ltm_lr_schedule": False,
            },
        },
        current_identity,
        "checkpoint.pt",
    ) is True


def test_iterable_mid_epoch_resume_fails_closed_and_binds_worker_topology(capsys):
    class _Tokenizer:
        def __len__(self):
            return 8

    class _Stream(IterableDataset):
        def __iter__(self):
            yield from range(8)

    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        streaming_datasets=True,
        hf_streaming_shuffle_buffer=100,
        hf_auto_shard=False,
        train_prompt_tokens=True,
        prompt_loss_weight=1.0,
        response_loss_weight=1.0,
        response_boundary_loss_weight=1.0,
        response_boundary_tokens=0,
        min_response_tokens=1,
        drop_empty_completions=True,
        _optimizer_grouping_version=2,
    )
    architecture = {"architecture_revision": "legacy-v8"}
    loader_zero_workers = DataLoader(_Stream(), batch_size=2, num_workers=0)
    loader_one_worker = DataLoader(_Stream(), batch_size=2, num_workers=1)
    identity_zero = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader_zero_workers,
        4,
        architecture,
    )
    identity_one = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader_one_worker,
        4,
        architecture,
    )
    assert identity_zero["loader"]["iterable_dataset"] is True
    assert identity_zero["loader"]["iterable_num_workers"] == 0
    assert identity_one["loader"]["iterable_num_workers"] == 1
    with pytest.raises(RuntimeError, match="iterable_num_workers"):
        validate_exact_resume_identity(
            {"run_identity": identity_zero, "mid_epoch_step": 0},
            identity_one,
            "checkpoint.pt",
        )
    with pytest.raises(RuntimeError, match="cannot be proven for an IterableDataset"):
        validate_exact_resume_identity(
            {"run_identity": identity_zero, "mid_epoch_step": 2},
            identity_zero,
            "checkpoint.pt",
        )

    assert validate_exact_resume_identity(
        {"run_identity": identity_zero, "mid_epoch_step": 0},
        identity_zero,
        "checkpoint.pt",
    ) is False
    assert "cannot be proven" in capsys.readouterr().out


def test_mutable_map_dataset_mid_epoch_resume_fails_closed_but_boundary_warns(
    capsys,
):
    class _Tokenizer:
        def __len__(self):
            return 8

    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        train="mutable.jsonl",
        train_prompt_tokens=True,
        prompt_loss_weight=1.0,
        response_loss_weight=1.0,
        response_boundary_loss_weight=1.0,
        response_boundary_tokens=0,
        min_response_tokens=1,
        drop_empty_completions=True,
        _optimizer_grouping_version=2,
    )
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2)
    identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        {"architecture_revision": "legacy-v8"},
    )
    assert (
        identity["dataset"]["replay_guarantee"]
        == "unproven-mutable-source"
    )

    with pytest.raises(RuntimeError, match="content-unverified"):
        validate_exact_resume_identity(
            {"run_identity": identity, "mid_epoch_step": 2},
            identity,
            "checkpoint.pt",
        )

    assert validate_exact_resume_identity(
        {"run_identity": identity, "mid_epoch_step": 0},
        identity,
        "checkpoint.pt",
    ) is False
    assert "Exact data replay cannot be proven" in capsys.readouterr().out


def test_legacy_checkpoint_without_identity_cannot_resume_mid_epoch():
    class _Tokenizer:
        def __len__(self):
            return 8

    args = SimpleNamespace(
        seed=123,
        batch_size=2,
        accumulation_steps=1,
        accumulation_normalization="weighted-token",
        max_length=16,
        training_chunk_size=4,
        hf_dataset="owner/pinned",
        _resolved_hf_dataset_revision="a" * 40,
        train_prompt_tokens=True,
        prompt_loss_weight=1.0,
        response_loss_weight=1.0,
        response_boundary_loss_weight=1.0,
        response_boundary_tokens=0,
        min_response_tokens=1,
        drop_empty_completions=True,
        _optimizer_grouping_version=2,
    )
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2)
    identity = build_exact_resume_identity(
        args,
        _Tokenizer(),
        loader,
        len(loader),
        {"architecture_revision": "legacy-v8"},
    )
    assert identity["dataset"]["replay_guarantee"] == "immutable-hf-revision"

    with pytest.raises(RuntimeError, match="no saved run/data identity"):
        validate_exact_resume_identity(
            {"mid_epoch_step": 2},
            identity,
            "legacy.pt",
        )
