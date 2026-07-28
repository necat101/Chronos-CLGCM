from types import SimpleNamespace
import argparse
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hierarchos.training.datasets import EpochShuffleSampler, LengthGroupedBatchSampler
from hierarchos.training.trainer import (
    build_hierarchos_optimizer,
    build_training_checkpoint,
    build_exact_resume_identity,
    build_lr_scheduler,
    capture_effective_training_config,
    capture_ltm_lr_scheduler_state,
    capture_dataloader_state,
    accumulation_divisor_for_step,
    configure_ltm_lr_schedule,
    compute_update_steps,
    compute_remaining_update_steps,
    get_current_ltm_lr,
    advance_ltm_lr_schedule,
    restore_dataloader_state,
    restore_model_grad_state,
    restore_scheduler_state_and_live_lrs,
    save_training_checkpoint_if_finite,
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
    ltm_inner_updates_enabled,
    normalize_ltm_training_mode,
    validate_exact_resume_identity,
)
from hierarchos.utils.checkpoint import (
    load_checkpoint_payload_compatible,
    sanitize_model_state_dict,
)
from hierarchos.utils.rosa import ROSAState
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
        {"run_identity": saved_identity},
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
