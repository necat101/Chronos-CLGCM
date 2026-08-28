#!/usr/bin/env python3
"""Compare a masked multi-batch historical-TBPTT AdamW trajectory: PyTorch vs Vulkan."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore, load_full_model_with_config
from hierarchos.training.trainer import (
    _cap_loss_component_for_backward,
    _clip_gradients_and_check,
    _detach_finite_clamp,
    _detach_finite_l2_clamp,
    compute_chunk_training_weights,
    detach_ltm_state_from_outputs,
    mark_val_proj_trained,
)
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture
from tools.verify_vulkan_worker_refinement_loss_parity import (
    DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV,
    NATIVE_FP16_LOW_RANK_BACKWARD_ENV,
    NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_ENV,
    NATIVE_FP16_OUT_NORM_BACKWARD_ENV,
    NATIVE_FP16_PROJECTION_BACKWARD_ENV,
    NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD_ENV,
    TRAINING_PRECISION_ENV,
    Fp16LmBackwardLinear,
    env_flag_enabled,
    fp16_storage_fp32_compute_weight,
    install_fp16_execution_storage,
    install_native_fp16_backward_oracle,
    restore_fp32_masters,
)
import tools.verify_vulkan_worker_refinement_loss_parity as worker_parity


TBPTT_CHUNK_SIZE = 3
DEFAULT_UPDATE_COUNT = 3
INFERENCE_TOKENS = [1, 2, 1, 2, 3, 1]
TAPE_PROFILE_DB_ENV = "HIERARCHOS_VULKAN_TAPE_PROFILE_DB"
DEFAULT_TAPE_PROFILE_DB = (
    ROOT / "benchmark_results" / "vulkan_training_submission_profiles.v1.jsonl"
)
PLAN_FACTOR_FIELDS = (
    "sequence_microbatch_size",
    "state_checkpoint_stride",
    "h_backward_segment_schedule",
    "l_backward_segment_schedule",
    "h_backward_kernel_geometry",
    "l_backward_kernel_geometry",
    "rwkv_numerics_policy",
)


def _checkpoint_torch_dtype(label: str) -> torch.dtype | None:
    return {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[label]


def _round_model_state_to_checkpoint_dtype(model: HierarchosCore, label: str) -> None:
    dtype = _checkpoint_torch_dtype(label)
    if dtype is None:
        return
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.detach().to(dtype=dtype).float())


def _rewrite_exported_checkpoint_dtype(model_dir: Path, label: str) -> None:
    dtype = _checkpoint_torch_dtype(label)
    if dtype is None:
        return
    model_path = model_dir / "model.safetensors"
    state = load_file(str(model_path))
    converted = {
        name: (
            tensor.to(dtype=dtype).clone()
            if tensor.is_floating_point()
            else tensor.clone()
        )
        for name, tensor in state.items()
    }
    del state
    gc.collect()
    with safe_open(str(model_path), framework="pt", device="cpu") as tensors:
        metadata = tensors.metadata()
    rewritten = model_path.with_name("model.checkpoint-dtype.safetensors")
    save_file(converted, str(rewritten), metadata=metadata)
    model_path.unlink()
    rewritten.replace(model_path)


def _run(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    if lhs.numel() == 0:
        return 0.0
    return float((lhs.float() - rhs.float()).abs().max().item())


def _profile_value(profile_key: dict, field: str):
    if field == "rwkv_numerics_policy":
        return profile_key.get(field, "strict-parity")
    return profile_key.get(field)


def _load_profile_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid Vulkan tape-profile JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if record.get("status") != "ok" or not isinstance(record.get("profile_key"), dict):
            continue
        records.append(record)
    return records


def _plan_synthesis_evidence(
    plans: list[dict],
    rust_result: dict,
    *,
    profile_db: Path,
    training_precision: str,
) -> dict:
    """Show whether an executed composite existed in the persisted arm table.

    The Rust plan's ``profile_records`` field is authoritative for whether its
    active factor had evidence in the runtime's strict geometry population.
    This independent JSONL audit asks a different question: had the complete
    seven-coordinate execution arm ever been observed for the same device and
    coarse training window? A zero exact-arm count paired with nonzero runtime
    factor evidence is direct evidence that the selector composed an unmeasured
    configuration instead of replaying a Cartesian arm.
    """

    records = _load_profile_records(profile_db)
    plan_evidence: list[dict] = []
    for plan in plans:
        sequence_count = int(plan["update_end"]) - int(plan["update_start"])
        population: list[dict] = []
        for record in records:
            key = record["profile_key"]
            if key.get("device") != rust_result.get("device_name"):
                continue
            if key.get("batch") != rust_result.get("batch"):
                continue
            if key.get("tokens_per_sequence") != rust_result.get("tokens"):
                continue
            if key.get("sequences") != sequence_count:
                continue
            if key.get("training_precision_policy", "fp32") != training_precision:
                continue
            population.append(record)

        exact_records = 0
        marginal_records = {field: 0 for field in PLAN_FACTOR_FIELDS}
        for record in population:
            key = record["profile_key"]
            exact = True
            for field in PLAN_FACTOR_FIELDS:
                if _profile_value(key, field) == plan.get(field):
                    marginal_records[field] += 1
                else:
                    exact = False
            if exact:
                exact_records += 1

        factor_records = int(plan.get("profile_records", 0))
        plan_evidence.append(
            {
                "update_start": int(plan["update_start"]),
                "update_end": int(plan["update_end"]),
                "coarse_population_records": len(population),
                "exact_full_arm_records": exact_records,
                "marginal_full_coordinate_records": marginal_records,
                "runtime_matched_active_factor_records": factor_records,
                "synthesized_from_profiled_factor_evidence": (
                    exact_records == 0 and factor_records > 0
                ),
            }
        )
    return {
        "profile_database": str(profile_db.resolve()),
        "profile_database_records": len(records),
        "plans": plan_evidence,
    }


def _load_joint_runtime_profile(path: Path) -> tuple[int, int, dict]:
    """Extract the synthesized tape coordinate from a persisted runtime profile."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported Vulkan joint-runtime profile schema: {payload.get('schema_version')!r}"
        )
    winning_arm = payload.get("winning_arm")
    if not isinstance(winning_arm, dict):
        raise ValueError("joint-runtime profile is missing winning_arm")
    tape_geometry = winning_arm.get("tape_geometry")
    if not isinstance(tape_geometry, dict):
        raise ValueError("joint-runtime profile winning_arm is missing tape_geometry")
    microbatch = int(tape_geometry["sequence_microbatch_size"])
    stride = int(tape_geometry["state_checkpoint_stride"])
    if microbatch <= 0 or stride <= 0:
        raise ValueError(
            "joint-runtime profile contains a non-positive synthesized tape geometry"
        )
    return microbatch, stride, payload


def _vulkan_devices() -> list[dict]:
    completed = _run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-devices",
        ]
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise ValueError("Vulkan device catalog did not return a JSON list")
    return payload


def _runtime_profile_selected_device_member_match(
    profile: dict, device: dict
) -> tuple[bool, int]:
    profile_key = profile.get("profile_key")
    if not isinstance(profile_key, dict):
        raise ValueError("joint-runtime profile is missing profile_key")
    device_uuids = profile_key.get("device_uuids")
    driver_uuids = profile_key.get("driver_uuids")
    if not isinstance(device_uuids, list) or not isinstance(driver_uuids, list):
        raise ValueError("joint-runtime profile key is missing Vulkan UUID arrays")
    if len(device_uuids) != len(driver_uuids):
        raise ValueError("joint-runtime profile Vulkan UUID arrays have different lengths")
    fingerprint = (device.get("device_uuid"), device.get("driver_uuid"))
    profile_fingerprints = set(zip(device_uuids, driver_uuids, strict=True))
    return fingerprint in profile_fingerprints, len(profile_fingerprints)


def _align_g2_uses_monotonic(
    torch_uses: list[torch.Tensor], rust_uses: list[list[float]]
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Match PyTorch's dynamic uses to Vulkan's static-use superset in order."""

    if len(torch_uses) > len(rust_uses):
        raise AssertionError(
            "PyTorch emitted more g2 uses than Vulkan: "
            f"{len(torch_uses)} > {len(rust_uses)}"
        )
    rust_tensors = [torch.tensor(values, dtype=torch.float32) for values in rust_uses]
    rows = len(torch_uses)
    cols = len(rust_tensors)
    inf = float("inf")
    cost = [[inf] * (cols + 1) for _ in range(rows + 1)]
    take = [[False] * (cols + 1) for _ in range(rows + 1)]
    for col in range(cols + 1):
        cost[0][col] = 0.0
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            skip_cost = cost[row][col - 1]
            match_cost = cost[row - 1][col - 1] + _max_abs(
                torch_uses[row - 1].flatten(), rust_tensors[col - 1]
            )
            if match_cost <= skip_cost:
                cost[row][col] = match_cost
                take[row][col] = True
            else:
                cost[row][col] = skip_cost
    if not torch.isfinite(torch.tensor(cost[rows][cols])):
        raise AssertionError("unable to align PyTorch and Vulkan g2 use traces")

    aligned: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    row = rows
    col = cols
    while row > 0:
        if col == 0:
            raise AssertionError("g2 use alignment exhausted Vulkan uses")
        if take[row][col]:
            aligned.append((col - 1, torch_uses[row - 1], rust_tensors[col - 1]))
            row -= 1
            col -= 1
        else:
            col -= 1
    aligned.reverse()
    return aligned


def _labeled_g2_use_labels(
    tower: str,
    *,
    tokens: int,
    max_h_steps: int,
    max_l_steps: int,
    h_stride: int,
) -> list[str]:
    """Describe the exact static recurrent-backward order recorded by Vulkan."""

    labels: list[str] = []
    for token_index in reversed(range(tokens)):
        if tower == "l":
            labels.append(f"token={token_index} committed")
            for shadow_timestep in reversed(range(max_l_steps)):
                labels.append(
                    f"token={token_index} shadow_timestep={shadow_timestep}"
                )
            continue

        if tower != "h":
            raise AssertionError(f"unknown recurrent tower {tower!r}")
        h_steps = max_h_steps if token_index % h_stride == 0 else 1
        for shadow_step in reversed(range(max(0, h_steps - 1))):
            labels.append(
                f"token={token_index} shadow_step={shadow_step} "
                f"candidate={shadow_step + 1}"
            )
        labels.append(f"token={token_index} committed")
    return labels


def _install_lm_execution_oracle(
    model: HierarchosCore,
    precision: str,
    *,
    native_input_grad: bool,
) -> None:
    if precision == "fp32":
        return

    if precision in ("fp16-storage-parity", "fp16-storage-fp16-lm-backward"):
        def lm_forward(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
            return Fp16LmBackwardLinear.apply(input, module.weight, native_input_grad)
    else:
        def lm_forward(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
            return F.linear(
                input,
                fp16_storage_fp32_compute_weight(module.weight),
            )

    model.lm_head.forward = types.MethodType(lm_forward, model.lm_head)


@dataclass
class UpdateFixture:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    loss_weights: torch.Tensor
    previous_context: torch.Tensor
    target_context: torch.Tensor
    h_state: torch.Tensor
    l_state: torch.Tensor


def _right_padding_mask(lengths: tuple[int, ...], tokens: int) -> torch.Tensor:
    columns = torch.arange(tokens).unsqueeze(0)
    return (columns < torch.tensor(lengths).unsqueeze(1)).to(dtype=torch.float32)


def _build_updates(model: HierarchosCore, config, update_count: int) -> list[UpdateFixture]:
    token_rows = [
        ([2, 7, 5, 8, 3, 12], [11, 4, 10, 6, 9, 13]),
        ([14, 3, 9, 6, 15, 4], [5, 12, 7, 2, 16, 8]),
        ([17, 6, 4, 13, 9, 2], [10, 3, 18, 5, 11, 7]),
    ]
    lengths = [(6, 4), (5, 3), (6, 2)]
    weights = [
        [[1.0, 0.5, 1.75, 1.0, 0.8, 1.2], [1.0, 1.25, 0.4, 1.6, 0.0, 0.0]],
        [[0.8, 1.3, 0.6, 1.4, 1.1, 0.0], [1.2, 0.7, 1.5, 0.0, 0.0, 0.0]],
        [[1.1, 0.9, 1.6, 0.75, 1.25, 0.55], [0.65, 1.35, 0.0, 0.0, 0.0, 0.0]],
    ]
    updates: list[UpdateFixture] = []
    batch = len(token_rows[0])
    tokens = len(token_rows[0][0])
    for update_index in range(update_count):
        fixture_index = update_index % len(token_rows)
        cycle_index = update_index // len(token_rows)
        rows = token_rows[fixture_index]
        active_lengths = lengths[fixture_index]
        weight_rows = weights[fixture_index]
        input_ids = torch.tensor(rows, dtype=torch.long)
        if cycle_index:
            # Extend the old three-update oracle into an arbitrarily long but
            # bounded deterministic trajectory. Cycling only the original
            # fixtures would repeatedly present identical token rows; this
            # offset keeps later optimizer windows distinct while staying in
            # the tiny model vocabulary and preserving the same masking shape.
            vocab_span = max(int(config.vocab_size) - 1, 1)
            input_ids = ((input_ids + 5 * cycle_index - 1) % vocab_span) + 1
        attention_mask = _right_padding_mask(active_lengths, tokens)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if fixture_index == 1:
            labels[0, 4] = -100
        loss_weights = torch.tensor(weight_rows, dtype=torch.float32)
        if cycle_index:
            loss_weights = loss_weights * (1.0 + 0.02 * (cycle_index % 5))
        loss_weights[attention_mask == 0] = 0.0

        bounded_phase = update_index % 17 + 1
        context_base = 0.003 * bounded_phase
        previous_context = torch.arange(
            batch * config.context_dim, dtype=torch.float32
        ).reshape(batch, config.context_dim)
        previous_context = (previous_context + 1.0) * context_base
        target_context = previous_context * -0.35

        h_state = torch.arange(
            batch * config.h_hidden * model.h_rnn.state_size, dtype=torch.float32
        ).reshape(batch, config.h_hidden, model.h_rnn.state_size)
        h_state = (h_state + 1.0) * (0.0002 * bounded_phase)
        l_state = torch.arange(
            batch * config.l_hidden * model.l_rnn.state_size, dtype=torch.float32
        ).reshape(batch, config.l_hidden, model.l_rnn.state_size)
        l_state = (l_state + 1.0) * (-0.00015 * bounded_phase)

        updates.append(
            UpdateFixture(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                loss_weights=loss_weights,
                previous_context=previous_context,
                target_context=target_context,
                h_state=h_state,
                l_state=l_state,
            )
        )
    return updates


def _optimizer(model: HierarchosCore, optimizer_case: dict[str, float]) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        clean_name = name.replace("_orig_mod.", "")
        if clean_name.startswith("val_proj.") or ".val_proj." in clean_name:
            no_decay.append(parameter)
        elif parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": optimizer_case["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=optimizer_case["lr"],
        betas=(optimizer_case["beta1"], optimizer_case["beta2"]),
        eps=optimizer_case["eps"],
        foreach=False,
    )


def _train_pytorch_update(
    model: HierarchosCore,
    optimizer: torch.optim.AdamW,
    fixture: UpdateFixture,
    objective: dict[str, float],
    config,
    *,
    zero_grad: bool = True,
    step_optimizer: bool = True,
    loss_scale: float = 1.0,
    gradient_divisor_before_step: float | None = None,
    grad_clip: float = 1.0,
    update_ltm_controller: bool = False,
    fp16_execution_masters: list[tuple[torch.nn.Parameter, torch.Tensor]] | None = None,
    gradient_snapshots: list[dict[str, torch.Tensor]] | None = None,
    host_diagnostics: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor, float | None]:
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    if not (loss_scale > 0.0):
        raise AssertionError(f"PyTorch accumulation loss_scale must be positive, got {loss_scale}")
    input_ids = fixture.input_ids
    labels = fixture.labels
    attention_mask = fixture.attention_mask
    loss_weights = fixture.loss_weights
    tokens = input_ids.shape[1]

    chunk_plan = compute_chunk_training_weights(
        labels,
        attention_mask,
        TBPTT_CHUNK_SIZE,
        loss_weights=loss_weights,
    )

    h_state = fixture.h_state
    l_state = fixture.l_state
    previous_context = fixture.previous_context
    target_context = fixture.target_context
    drift_state = None
    ltm_state = None
    weighted_objective = 0.0
    writer_alignment_score = None

    for chunk in chunk_plan:
        start_t = int(chunk["start"])
        end_t = int(chunk["end"])
        label_ratio = float(chunk["label_ratio"])
        token_ratio = float(chunk["token_ratio"])
        if label_ratio == 0.0 and token_ratio == 0.0:
            continue

        loss_end_t = min(end_t + 1, tokens)
        outputs = model(
            input_ids[:, start_t:end_t],
            attention_mask=attention_mask[:, start_t:end_t],
            labels=labels[:, start_t:loss_end_t],
            h_state=h_state,
            l_state=l_state,
            prev_context=previous_context,
            target_context=target_context,
            drift_state=drift_state,
            ltm_memory_state=ltm_state,
            global_pos_offset=start_t,
            loss_weights=loss_weights[:, start_t:loss_end_t],
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
            return_step_telemetry=False,
            return_numerics=False,
            compute_ltm_value_alignment=True,
        )

        ce_loss = _cap_loss_component_for_backward(
            outputs["loss"],
            objective["max_ce_loss_for_backward"],
        )
        aux_loss = torch.zeros_like(ce_loss)
        ponder_cost = outputs.get("ponder_cost")
        if ponder_cost is not None:
            aux_loss = aux_loss + objective["ponder_loss_weight"] * _cap_loss_component_for_backward(
                ponder_cost,
                objective["max_ponder_cost_for_backward"],
            )
        commitment_cost = outputs.get("commitment_cost")
        if commitment_cost is not None:
            aux_loss = aux_loss + objective["commitment_loss_weight"] * _cap_loss_component_for_backward(
                commitment_cost,
                objective["max_commitment_cost_for_backward"],
                preserve_gradient=True,
            )
        alignment_cost = outputs.get("ltm_value_alignment_cost")
        if alignment_cost is not None:
            weighted_writer_score = alignment_cost.detach().float() * token_ratio
            writer_alignment_score = (
                weighted_writer_score
                if writer_alignment_score is None
                else writer_alignment_score + weighted_writer_score
            )
            aux_loss = aux_loss + config.ltm_value_alignment_weight * alignment_cost

        chunk_loss = (ce_loss * label_ratio + aux_loss * token_ratio) * loss_scale
        if not bool(torch.isfinite(chunk_loss).item()):
            raise AssertionError(
                f"PyTorch TBPTT objective became non-finite in chunk {start_t}:{end_t}"
            )
        chunk_loss.backward()
        if host_diagnostics:
            weighted_objective += float(chunk_loss.detach().cpu())

        # Match the production trainer's historical-TBPTT carrier boundary
        # exactly. The model clamps its public recurrent outputs internally,
        # but the trainer deliberately re-applies the finite-preserving clamp
        # after detach so a future model-side refactor cannot silently weaken
        # the safety contract at a chunk boundary. NaN/Inf are preserved here
        # (rather than repaired) and are rejected by the finite checks below or
        # by the optimizer safety boundary.
        recurrent_state_clamp = float(getattr(config, "recurrent_state_clamp", 50.0))
        context_state_clamp = float(getattr(config, "context_state_clamp", 50.0))
        drift_state_clamp = float(getattr(config, "drift_state_clamp", 5.0))
        drift_norm_clamp = float(getattr(config, "drift_norm_clamp", 0.0))
        h_state = _detach_finite_clamp(outputs["h_state"], recurrent_state_clamp)
        l_state = _detach_finite_clamp(outputs["l_state"], recurrent_state_clamp)
        previous_context = _detach_finite_clamp(
            outputs["prev_context"], context_state_clamp
        )
        target_context = _detach_finite_clamp(
            outputs["target_context"], context_state_clamp
        )
        drift_value = outputs.get("drift_state")
        drift_state = (
            _detach_finite_l2_clamp(
                drift_value,
                drift_state_clamp,
                drift_norm_clamp,
            )
            if torch.is_tensor(drift_value)
            else drift_value
        )
        ltm_state = detach_ltm_state_from_outputs(outputs)

    if step_optimizer and gradient_snapshots is not None:
        gradient_snapshots.append(
            {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
    if step_optimizer and gradient_divisor_before_step is not None:
        if not (gradient_divisor_before_step > 0.0):
            raise AssertionError(
                "PyTorch accumulation gradient divisor must be positive, got "
                f"{gradient_divisor_before_step}"
            )
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(gradient_divisor_before_step)
    optimizer_gradient_norm = None
    if step_optimizer:
        grads_ok, grad_issue = _clip_gradients_and_check(model, grad_clip)
        if not grads_ok:
            raise AssertionError(
                "PyTorch parity window was rejected by the production gradient safety boundary: "
                f"{grad_issue}"
            )
        if torch.is_tensor(grad_issue):
            optimizer_gradient_norm = float(grad_issue.detach().cpu().item())
        elif grad_issue is not None:
            optimizer_gradient_norm = float(grad_issue)
        if fp16_execution_masters:
            restore_fp32_masters(fp16_execution_masters)
        optimizer.step()
        if update_ltm_controller and config.ltm_value_alignment_weight > 0.0:
            mark_val_proj_trained(model, alignment_cost=writer_alignment_score)
    return (
        weighted_objective,
        h_state.detach().cpu() if host_diagnostics else h_state.detach(),
        l_state.detach().cpu() if host_diagnostics else l_state.detach(),
        optimizer_gradient_norm,
    )


def _rust_update_payload(fixture: UpdateFixture) -> dict:
    return {
        "input_ids": fixture.input_ids.flatten().tolist(),
        "labels": fixture.labels.flatten().tolist(),
        "attention_mask": fixture.attention_mask.flatten().tolist(),
        "loss_weights": fixture.loss_weights.flatten().tolist(),
        "initial_previous_context": fixture.previous_context.flatten().tolist(),
        "initial_target_context": fixture.target_context.flatten().tolist(),
        "h_initial_packed_state": fixture.h_state.flatten().tolist(),
        "l_initial_packed_state": fixture.l_state.flatten().tolist(),
        "global_pos_offset": 0,
        "reset_rosa_at_start": True,
        "pytorch_tbptt_chunk_size": TBPTT_CHUNK_SIZE,
    }


def _shifted_supervision_mass(fixture: UpdateFixture) -> float:
    shifted_labels = fixture.labels[:, 1:]
    shifted_weights = fixture.loss_weights[:, 1:]
    usable = shifted_labels != -100
    mass = float(shifted_weights[usable].sum().item())
    if not (mass > 0.0):
        raise AssertionError(f"update has no positive shifted supervision mass: {mass}")
    return mass


def _pytorch_inference_logits(model: HierarchosCore, device: torch.device) -> torch.Tensor:
    model = model.to(device).eval()
    input_ids = torch.tensor([INFERENCE_TOKENS], dtype=torch.long, device=device)
    with torch.no_grad():
        return model(
            input_ids,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
            return_step_telemetry=False,
            return_numerics=False,
        )["logits"][0].float().cpu()


def _native_inference(package: Path) -> tuple[torch.Tensor, dict]:
    completed = _run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-inference" / "Cargo.toml"),
            "--bin",
            "hierarchos-infer",
            "--",
            "--model",
            str(package),
            "--tokens",
            ",".join(map(str, INFERENCE_TOKENS)),
        ]
    )
    payload = json.loads(completed.stdout)
    return torch.tensor(payload["logits"], dtype=torch.float32), payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless the final Vulkan-trained package also passes PyTorch/CUDA inference parity",
    )
    parser.add_argument(
        "--precision",
        choices=(
            "fp32",
            "fp16-storage-fp32-compute",
            "fp16-storage-parity",
            "fp16-storage-fp16-lm-backward",
        ),
        default="fp32",
        help="Vulkan/PyTorch trainable execution precision contract",
    )
    parser.add_argument(
        "--checkpoint-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="SafeTensors storage dtype presented to both Vulkan and native Rust loaders",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="run the Vulkan parity leg on one explicit physical-device index",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="labeled microbatches per AdamW step (default: 1)",
    )
    parser.add_argument(
        "--update-count",
        type=int,
        default=DEFAULT_UPDATE_COUNT,
        help=(
            "number of deterministic labeled updates in the PyTorch/Vulkan optimizer trajectory; "
            f"default {DEFAULT_UPDATE_COUNT}, larger values enable sustained-window drift tests"
        ),
    )
    parser.add_argument(
        "--budgeted-windows",
        action="store_true",
        help="execute each complete accumulation window through the Vulkan VRAM planner/sparse-replay scheduler",
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=None,
        help=(
            "replay winning_arm.tape_geometry from a persisted "
            "vulkan_joint_runtime_profile.v1.json and A/B it against the automatic budgeted path"
        ),
    )
    parser.add_argument(
        "--require-runtime-profile-device-match",
        action="store_true",
        help=(
            "fail unless the selected Vulkan device UUID+driver UUID appears in the "
            "persisted runtime profile topology"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="write a machine-readable cross-backend parity qualification certificate",
    )
    parser.add_argument(
        "--sequence-microbatch-size",
        type=int,
        default=None,
        help="force an exact Vulkan tape sequence microbatch for parity qualification",
    )
    parser.add_argument(
        "--state-checkpoint-stride",
        type=int,
        default=None,
        help="force an exact Vulkan recurrent checkpoint stride for parity qualification",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        type=float,
        default=None,
        help="scale native/PyTorch backward sources, then unscale before AdamW",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3.0e-4,
        help="AdamW learning rate; set 0 for a no-parameter-update recurrent parity control",
    )
    parser.add_argument(
        "--optimizer-eps",
        type=float,
        default=1.0e-8,
        help="AdamW epsilon shared by the PyTorch and Vulkan optimizer contracts",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help=(
            "global L2 gradient clipping threshold shared by the PyTorch and Vulkan "
            "optimizer safety boundaries (0 disables clipping but still rejects non-finite gradients)"
        ),
    )
    parser.add_argument(
        "--safety-clamp-stress",
        action="store_true",
        help=(
            "lower the model's finite/state/RWKV clamp ceilings so the labeled PyTorch/Vulkan "
            "trajectory is forced through saturated safety boundaries instead of merely carrying "
            "the production-default clamp configuration"
        ),
    )
    for option, destination, description in (
        ("--recurrent-state-clamp", "recurrent_state_clamp", "packed/public RWKV recurrent state"),
        ("--context-state-clamp", "context_state_clamp", "manager/sliding context state"),
        ("--drift-state-clamp", "drift_state_clamp", "worker context-drift state"),
        ("--drift-norm-clamp", "drift_norm_clamp", "worker context-drift L2 norm"),
        ("--activation-clamp", "activation_clamp", "manager/worker activation"),
        ("--halt-logit-clamp", "halt_logit_clamp", "manager ACT halt logit"),
        ("--rwkv-channel-mix-key-clamp", "rwkv_channel_mix_key_clamp", "RWKV channel-mix key"),
        (
            "--rwkv-channel-mix-deepembed-clamp",
            "rwkv_channel_mix_deepembed_clamp",
            "RWKV channel-mix DeepEmbed modulation",
        ),
    ):
        parser.add_argument(
            option,
            dest=destination,
            type=float,
            default=None,
            help=f"override the {description} safety ceiling for cross-backend qualification",
        )
    parser.add_argument(
        "--diagnose-gradients",
        action="store_true",
        help="capture and compare source-scaled canonical gradients before AdamW",
    )
    parser.add_argument(
        "--diagnose-g2-uses",
        action="store_true",
        help="capture and compare every recurrent h/l g2 scratch dW contribution",
    )
    parser.add_argument(
        "--low-rank-dw-semantics",
        choices=(
            "fp16-product",
            "widened-fp16-inputs",
            "compensated-widened-fp16-inputs",
        ),
        default="fp16-product",
        help=(
            "PyTorch oracle for the experimental native-FP16 low-rank dW product; "
            "the isolated Vulkan diagnostic determines which device behavior applies"
        ),
    )
    parser.add_argument(
        "--max-h-steps",
        type=int,
        default=None,
        help="override manager recurrence depth for diagnostic geometry sweeps",
    )
    parser.add_argument(
        "--max-l-steps",
        type=int,
        default=None,
        help="override worker recurrence depth for diagnostic geometry sweeps",
    )
    parser.add_argument(
        "--diagnostic-tokens",
        type=int,
        default=None,
        help="truncate each fixed parity update to the first N tokens (minimum 2)",
    )
    args = parser.parse_args()
    if (args.sequence_microbatch_size is None) != (args.state_checkpoint_stride is None):
        parser.error(
            "--sequence-microbatch-size and --state-checkpoint-stride must be supplied together"
        )
    if args.runtime_profile is not None and args.sequence_microbatch_size is not None:
        parser.error(
            "--runtime-profile cannot be combined with an explicit tape geometry"
        )
    if args.require_runtime_profile_device_match and args.runtime_profile is None:
        parser.error("--require-runtime-profile-device-match requires --runtime-profile")
    if args.device_index is not None and args.device_index < 0:
        parser.error("--device-index must be non-negative")
    runtime_profile_payload: dict | None = None
    if args.runtime_profile is not None:
        try:
            (
                args.sequence_microbatch_size,
                args.state_checkpoint_stride,
                runtime_profile_payload,
            ) = _load_joint_runtime_profile(args.runtime_profile)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --runtime-profile: {exc}")
    if args.sequence_microbatch_size is not None:
        if args.sequence_microbatch_size <= 0 or args.state_checkpoint_stride <= 0:
            parser.error("explicit tape geometry values must be positive")
        # Exact geometry is an override of the budgeted scheduler, not the
        # legacy one-sequence tape path. Make the relationship explicit in the
        # command line that reaches Rust even when the caller supplied only a
        # persisted runtime profile.
        args.budgeted_windows = True
    if args.accumulation_steps <= 0:
        parser.error("--accumulation-steps must be positive")
    if args.update_count <= 0:
        parser.error("--update-count must be positive")
    if args.dynamic_loss_scale is not None and not args.dynamic_loss_scale > 0.0:
        parser.error("--dynamic-loss-scale must be positive")
    if args.learning_rate < 0.0:
        parser.error("--learning-rate must be non-negative")
    if not args.optimizer_eps > 0.0:
        parser.error("--optimizer-eps must be positive")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0.0:
        parser.error("--grad-clip must be finite and non-negative")
    for option, value, allow_zero in (
        ("--recurrent-state-clamp", args.recurrent_state_clamp, False),
        ("--context-state-clamp", args.context_state_clamp, False),
        ("--drift-state-clamp", args.drift_state_clamp, False),
        ("--drift-norm-clamp", args.drift_norm_clamp, True),
        ("--activation-clamp", args.activation_clamp, False),
        ("--halt-logit-clamp", args.halt_logit_clamp, False),
        ("--rwkv-channel-mix-key-clamp", args.rwkv_channel_mix_key_clamp, True),
        (
            "--rwkv-channel-mix-deepembed-clamp",
            args.rwkv_channel_mix_deepembed_clamp,
            True,
        ),
    ):
        if value is None:
            continue
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            relation = "non-negative" if allow_zero else "positive"
            parser.error(f"{option} must be finite and {relation}")
    if args.max_h_steps is not None and args.max_h_steps <= 0:
        parser.error("--max-h-steps must be positive")
    if args.max_l_steps is not None and args.max_l_steps <= 0:
        parser.error("--max-l-steps must be positive")
    if args.diagnostic_tokens is not None and args.diagnostic_tokens < 2:
        parser.error("--diagnostic-tokens must be at least 2")
    os.environ[TRAINING_PRECISION_ENV] = args.precision
    if args.precision == "fp16-storage-parity":
        os.environ[DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV] = "1"
    if args.diagnose_g2_uses:
        os.environ["HIERARCHOS_VULKAN_DIAGNOSTIC_G2_PER_USE"] = "1"
    torch.manual_seed(20260819)
    config = tiny_coherent_config(32)
    if args.safety_clamp_stress:
        # Deliberately small but non-degenerate ceilings. The fixture's public
        # recurrent state exceeds 0.25, so this mode proves the packed RWKV
        # clamp/backward mask on the full labeled optimizer trajectory rather
        # than relying only on isolated kernel tests. The context/drift and
        # channel-mix ceilings similarly force the control/memory safety paths
        # into the numerically interesting regime.
        config.recurrent_state_clamp = 0.25
        config.context_state_clamp = 0.05
        config.drift_state_clamp = 0.04
        config.drift_norm_clamp = 0.05
        config.activation_clamp = 0.35
        config.halt_logit_clamp = 0.75
        config.rwkv_channel_mix_key_clamp = 0.50
        config.rwkv_channel_mix_deepembed_clamp = 0.20
    for field in (
        "recurrent_state_clamp",
        "context_state_clamp",
        "drift_state_clamp",
        "drift_norm_clamp",
        "activation_clamp",
        "halt_logit_clamp",
        "rwkv_channel_mix_key_clamp",
        "rwkv_channel_mix_deepembed_clamp",
    ):
        override = getattr(args, field)
        if override is not None:
            setattr(config, field, override)
    if args.max_h_steps is not None:
        config.max_h_steps = args.max_h_steps
    if args.max_l_steps is not None:
        config.max_l_steps = args.max_l_steps
    config.z_loss_weight = 0.003
    config.ltm_value_alignment_weight = 0.021
    config.ltm_value_alignment_stride = 2
    config.ltm_value_alignment_min_updates = 1
    config.memory_gate_warmup_steps = 0.0
    config.memory_gate_warmup_floor = 0.0
    config.detach_every_n_steps = 32
    model = HierarchosCore(config).train()
    _make_nontrivial_memory_fixture(model, config)
    _round_model_state_to_checkpoint_dtype(model, args.checkpoint_dtype)

    updates = _build_updates(model, config, args.update_count)
    if args.safety_clamp_stress:
        recurrent_ceiling = float(config.recurrent_state_clamp)
        fixture_recurrent_peak = max(
            max(float(fixture.h_state.abs().max().item()), float(fixture.l_state.abs().max().item()))
            for fixture in updates
        )
        if fixture_recurrent_peak <= recurrent_ceiling:
            raise AssertionError(
                "safety clamp stress fixture did not cross recurrent_state_clamp: "
                f"peak={fixture_recurrent_peak:.9g}, ceiling={recurrent_ceiling:.9g}"
            )
    if args.diagnostic_tokens is not None:
        available_tokens = updates[0].input_ids.shape[1]
        if args.diagnostic_tokens > available_tokens:
            parser.error(
                f"--diagnostic-tokens cannot exceed fixture width {available_tokens}"
            )
        for fixture in updates:
            fixture.input_ids = fixture.input_ids[:, : args.diagnostic_tokens].contiguous()
            fixture.labels = fixture.labels[:, : args.diagnostic_tokens].contiguous()
            fixture.attention_mask = fixture.attention_mask[
                :, : args.diagnostic_tokens
            ].contiguous()
            fixture.loss_weights = fixture.loss_weights[
                :, : args.diagnostic_tokens
            ].contiguous()
    batch, tokens = updates[0].input_ids.shape
    objective = {
        "z_loss_weight": config.z_loss_weight,
        "ponder_loss_weight": 0.013,
        "commitment_loss_weight": 0.37,
        # Match the production PyTorch trainer safety policy. CE/ponder caps
        # are disabled by default; commitment uses a straight-through cap so
        # the scalar is bounded while its optimizer adjoint remains unchanged.
        "max_ce_loss_for_backward": 0.0,
        "max_ponder_cost_for_backward": 0.0,
        "max_commitment_cost_for_backward": 2.0,
    }
    optimizer_case = {
        "lr": args.learning_rate,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": args.optimizer_eps,
        "weight_decay": 0.07,
    }

    initial_model = copy.deepcopy(model).eval()
    native_fp16_policy = args.precision == "fp16-storage-fp16-lm-backward"
    source_scaled_fp32_source_adjoint_guard = (
        native_fp16_policy and args.dynamic_loss_scale is not None
    )
    native_fp16_lm_input_grad = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and not env_flag_enabled(DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV, default=False)
    )
    _install_lm_execution_oracle(
        model,
        args.precision,
        native_input_grad=native_fp16_lm_input_grad,
    )
    native_fp16_low_rank_backward = native_fp16_policy and env_flag_enabled(
        NATIVE_FP16_LOW_RANK_BACKWARD_ENV, default=True
    )
    native_fp16_low_rank_parameter_grad = (
        native_fp16_low_rank_backward
        and env_flag_enabled(NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_ENV, default=False)
    )
    native_fp16_out_norm_backward = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and env_flag_enabled(NATIVE_FP16_OUT_NORM_BACKWARD_ENV, default=True)
    )
    native_fp16_projection_backward = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and env_flag_enabled(NATIVE_FP16_PROJECTION_BACKWARD_ENV, default=True)
    )
    native_fp16_recurrent_projection_backward = (
        args.dynamic_loss_scale is not None
        and env_flag_enabled(NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD_ENV, default=False)
    )
    install_native_fp16_backward_oracle(
        model,
        include_out_norm=native_fp16_out_norm_backward,
        include_projections=native_fp16_projection_backward,
        include_low_rank=native_fp16_low_rank_backward,
        include_low_rank_inter_stage=False,
        include_low_rank_parameter_grad=(
            native_fp16_low_rank_parameter_grad and args.dynamic_loss_scale is not None
        ),
        low_rank_parameter_grad_product_semantics=args.low_rank_dw_semantics,
        include_recurrent_projection=native_fp16_recurrent_projection_backward,
    )
    optimizer = _optimizer(model, optimizer_case)
    pytorch_objectives: list[float] = []
    pytorch_final_h: list[torch.Tensor] = []
    pytorch_final_l: list[torch.Tensor] = []
    pytorch_optimizer_window_gradient_norms: list[float] = []
    pytorch_gradient_snapshots: list[dict[str, torch.Tensor]] = []
    pytorch_g2_per_use: list[dict[str, list[torch.Tensor]]] = []
    fp16_execution_masters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    supervision_masses = [_shifted_supervision_mass(fixture) for fixture in updates]
    for update_index, fixture in enumerate(updates):
        if args.diagnose_g2_uses:
            worker_parity.LOW_RANK_G2_BACKWARD_TRACE = []
        window_start = (update_index // args.accumulation_steps) * args.accumulation_steps
        window_end = min(window_start + args.accumulation_steps, len(updates))
        window_mass = sum(supervision_masses[window_start:window_end])
        group_offset = update_index - window_start
        zero_grad = group_offset == 0
        step_optimizer = update_index + 1 == window_end
        if args.precision != "fp32" and zero_grad:
            if fp16_execution_masters:
                raise AssertionError("FP16 execution masters leaked across accumulation windows")
            fp16_execution_masters = install_fp16_execution_storage(model)
        if args.accumulation_steps > 1:
            # Historical weighted-token accumulation scales each microbatch back
            # to an unnormalized supervision sum, accumulates those FP32 grads,
            # then divides exactly once at the optimizer boundary.
            loss_scale = supervision_masses[update_index]
            gradient_divisor = window_mass if step_optimizer else None
        else:
            loss_scale = 1.0
            gradient_divisor = None
        if args.dynamic_loss_scale is not None:
            loss_scale *= args.dynamic_loss_scale
            if step_optimizer:
                gradient_divisor = (gradient_divisor or 1.0) * args.dynamic_loss_scale
        value, final_h, final_l, optimizer_gradient_norm = _train_pytorch_update(
            model,
            optimizer,
            fixture,
            objective,
            config,
            zero_grad=zero_grad,
            step_optimizer=step_optimizer,
            loss_scale=loss_scale,
            gradient_divisor_before_step=gradient_divisor,
            grad_clip=args.grad_clip,
            fp16_execution_masters=fp16_execution_masters,
            gradient_snapshots=pytorch_gradient_snapshots if args.diagnose_gradients else None,
        )
        if args.diagnose_g2_uses:
            local_trace = worker_parity.LOW_RANK_G2_BACKWARD_TRACE or []
            pytorch_g2_per_use.append(
                {
                    tower: [
                        values
                        for name, values in local_trace
                        if name == f"{tower}_rnn.g2"
                    ]
                    for tower in ("h", "l")
                }
            )
            worker_parity.LOW_RANK_G2_BACKWARD_TRACE = None
        if step_optimizer:
            if optimizer_gradient_norm is None:
                raise AssertionError("optimizer step did not report its pre-clip global gradient norm")
            pytorch_optimizer_window_gradient_norms.append(optimizer_gradient_norm)
            fp16_execution_masters = []
        pytorch_objectives.append(value)
        pytorch_final_h.append(final_h)
        pytorch_final_l.append(final_l)

    if fp16_execution_masters:
        raise AssertionError("FP16 execution masters remained live after the final optimizer window")

    case = {
        "batch": batch,
        "tokens": tokens,
        "max_h_steps": config.max_h_steps,
        "max_l_steps": config.max_l_steps,
        "gradient_accumulation_steps": args.accumulation_steps,
        "dynamic_loss_scale": args.dynamic_loss_scale,
        "grad_clip": args.grad_clip,
        "capture_pending_gradients": args.diagnose_gradients,
        **_rust_update_payload(updates[0]),
        "additional_updates": [_rust_update_payload(fixture) for fixture in updates[1:]],
        "objective": objective,
        "optimizer": optimizer_case,
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-labeled-parity-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        case_path = temp / "case.json"
        rust_package = temp / "rust-trained"
        export_model(initial_model, config, model_dir)
        _rewrite_exported_checkpoint_dtype(model_dir, args.checkpoint_dtype)
        case_path.write_text(json.dumps(case), encoding="utf-8")
        rust_command = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-labeled-sequence-parity",
            "--",
            "--model",
            str(model_dir),
            "--case",
            str(case_path),
            "--output-package",
            str(rust_package),
        ]
        if args.device_index is not None:
            rust_command.extend(["--device-index", str(args.device_index)])
        if args.budgeted_windows:
            rust_command.append("--budgeted-windows")
        if args.sequence_microbatch_size is not None:
            rust_command.extend(
                [
                    "--sequence-microbatch-size",
                    str(args.sequence_microbatch_size),
                    "--state-checkpoint-stride",
                    str(args.state_checkpoint_stride),
                ]
            )
        rust_result = json.loads(_run(rust_command).stdout)
        qualified_budgeted_plans = rust_result.get("budgeted_plans", [])
        if args.budgeted_windows and not qualified_budgeted_plans:
            raise AssertionError(
                "budgeted Vulkan parity run did not report the scheduler plans that executed"
            )
        if args.sequence_microbatch_size is not None:
            for plan in qualified_budgeted_plans:
                update_count = int(plan["update_end"]) - int(plan["update_start"])
                expected_microbatch = min(args.sequence_microbatch_size, update_count)
                expected_stride = min(args.state_checkpoint_stride, tokens)
                if int(plan["sequence_microbatch_size"]) != expected_microbatch:
                    raise AssertionError(
                        "runtime-profile tape microbatch was not replayed after local-window clamping: "
                        f"expected {expected_microbatch}, got {plan['sequence_microbatch_size']}"
                    )
                if int(plan["state_checkpoint_stride"]) != expected_stride:
                    raise AssertionError(
                        "runtime-profile checkpoint stride was not replayed after token clamping: "
                        f"expected {expected_stride}, got {plan['state_checkpoint_stride']}"
                    )
        profile_control_result = None
        profile_control_model = None
        automatic_budgeted_plans: list[dict] = []
        if args.sequence_microbatch_size is not None:
            # Run the ordinary production planner from the identical model/case
            # as a direct Vulkan control. The explicit profile leg is still
            # independently checked against PyTorch below, so this A/B proves
            # that schedule synthesis itself did not create a new numerical ABI.
            profile_control_package = temp / "rust-budgeted-control"
            profile_control_command = [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-labeled-sequence-parity",
                "--",
                "--model",
                str(model_dir),
                "--case",
                str(case_path),
                "--output-package",
                str(profile_control_package),
                "--budgeted-windows",
            ]
            if args.device_index is not None:
                profile_control_command.extend(["--device-index", str(args.device_index)])
            profile_control_result = json.loads(_run(profile_control_command).stdout)
            automatic_budgeted_plans = profile_control_result.get("budgeted_plans", [])
            if not automatic_budgeted_plans:
                raise AssertionError(
                    "automatic budgeted control did not report the scheduler plans that executed"
                )
            profile_control_model, _ = load_full_model_with_config(
                str(profile_control_package), torch.device("cpu")
            )
        portable_dw_result = None
        if args.diagnose_g2_uses:
            portable_dw_package = temp / "rust-portable-dw"
            portable_dw_env = os.environ.copy()
            portable_dw_env[NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_ENV] = "0"
            portable_dw_command = list(rust_command)
            output_package_index = portable_dw_command.index("--output-package") + 1
            portable_dw_command[output_package_index] = str(portable_dw_package)
            portable_dw_result = json.loads(
                _run(portable_dw_command, env=portable_dw_env).stdout
            )
        rust_model, _ = load_full_model_with_config(str(rust_package), torch.device("cpu"))
        profile_control_parameter_diff = None
        profile_control_loss_diff = None
        if profile_control_model is not None and profile_control_result is not None:
            profile_state = profile_control_model.state_dict()
            qualified_state = rust_model.state_dict()
            shared_profile_names = sorted(set(profile_state) & set(qualified_state))
            profile_control_parameter_diff = max(
                (
                    _max_abs(
                        profile_state[name].detach().cpu(),
                        qualified_state[name].detach().cpu(),
                    )
                    for name in shared_profile_names
                    if profile_state[name].is_floating_point()
                    and qualified_state[name].is_floating_point()
                ),
                default=0.0,
            )
            profile_control_loss_diff = _max_abs(
                torch.tensor(profile_control_result["losses"], dtype=torch.float32),
                torch.tensor(rust_result["losses"], dtype=torch.float32),
            )
            if profile_control_parameter_diff > 8.0e-6:
                raise AssertionError(
                    "synthesized runtime-profile tape geometry changed the Vulkan optimizer "
                    f"trajectory by {profile_control_parameter_diff:.9g}"
                )
            if profile_control_loss_diff > 8.0e-6:
                raise AssertionError(
                    "synthesized runtime-profile tape geometry changed the Vulkan loss "
                    f"trajectory by {profile_control_loss_diff:.9g}"
                )
        cpu_inference = _pytorch_inference_logits(rust_model, torch.device("cpu"))
        native_inference, native_metadata = _native_inference(rust_package)
        native_inference_diff = _max_abs(cpu_inference, native_inference)
        torch.testing.assert_close(native_inference, cpu_inference, rtol=2.0e-4, atol=2.0e-5)
        rust_package_config = json.loads(
            (rust_package / "hierarchos_rust_config.json").read_text(encoding="utf-8")
        )
        expected_contract_hash = rust_package_config.get("architecture_contract_sha256")
        if native_metadata.get("architecture_revision") != config.architecture_revision:
            raise AssertionError(
                "native Rust inference reported a different architecture revision: "
                f"{native_metadata.get('architecture_revision')!r}"
            )
        if native_metadata.get("architecture_contract_sha256") != expected_contract_hash:
            raise AssertionError(
                "native Rust inference did not preserve the exported architecture contract identity"
            )

        cuda_inference_diff: float | None = None
        if torch.cuda.is_available():
            cuda_model, _ = load_full_model_with_config(
                str(rust_package), torch.device("cuda")
            )
            cuda_inference = _pytorch_inference_logits(cuda_model, torch.device("cuda"))
            cuda_inference_diff = _max_abs(cpu_inference, cuda_inference)
            torch.testing.assert_close(
                cuda_inference,
                cpu_inference,
                rtol=3.0e-4,
                atol=4.0e-5,
            )
        elif args.require_cuda:
            raise RuntimeError(
                "CUDA parity was required, but torch.cuda.is_available() is false on this host"
            )

        device_catalog = _vulkan_devices()
        selected_device = next(
            (
                device
                for device in device_catalog
                if device.get("index") == rust_result.get("device_index")
            ),
            None,
        )
        if selected_device is None:
            raise AssertionError(
                "Vulkan parity device was not present in the post-run device catalog: "
                f"index={rust_result.get('device_index')!r}"
            )
        runtime_profile_device_member_match: bool | None = None
        runtime_profile_topology_device_count: int | None = None
        if runtime_profile_payload is not None:
            try:
                (
                    runtime_profile_device_member_match,
                    runtime_profile_topology_device_count,
                ) = _runtime_profile_selected_device_member_match(
                    runtime_profile_payload, selected_device
                )
            except (TypeError, ValueError) as exc:
                raise AssertionError(f"invalid runtime-profile Vulkan fingerprint: {exc}") from exc
            if args.require_runtime_profile_device_match and not runtime_profile_device_member_match:
                raise AssertionError(
                    "selected Vulkan device UUID+driver UUID is not a member of the runtime-profile topology"
                )

        profile_db = Path(
            os.environ.get(TAPE_PROFILE_DB_ENV, str(DEFAULT_TAPE_PROFILE_DB))
        )
        qualified_synthesis_evidence = _plan_synthesis_evidence(
            qualified_budgeted_plans,
            rust_result,
            profile_db=profile_db,
            training_precision=args.precision,
        )
        automatic_synthesis_evidence = _plan_synthesis_evidence(
            automatic_budgeted_plans,
            profile_control_result if profile_control_result is not None else rust_result,
            profile_db=profile_db,
            training_precision=args.precision,
        )

    pytorch_state = model.state_dict()
    rust_state = rust_model.state_dict()
    common = sorted(set(pytorch_state) & set(rust_state))
    if not common:
        raise AssertionError("PyTorch/Vulkan trained models have no common state tensors")

    diffs: list[tuple[float, str]] = []
    changed: list[str] = []
    initial_state = initial_model.state_dict()
    for name in common:
        lhs = pytorch_state[name].detach().cpu()
        rhs = rust_state[name].detach().cpu()
        if lhs.is_floating_point() and rhs.is_floating_point():
            diff = _max_abs(lhs, rhs)
            diffs.append((diff, name))
            if name in initial_state and _max_abs(lhs, initial_state[name].detach().cpu()) > 0.0:
                changed.append(name)

    diffs.sort(reverse=True)
    max_diff, max_name = diffs[0]
    max_parameter_delta = (
        pytorch_state[max_name].detach().cpu().float()
        - rust_state[max_name].detach().cpu().float()
    ).abs()
    max_flat_index = int(max_parameter_delta.reshape(-1).argmax().item())
    rust_final_h = [torch.tensor(values) for values in rust_result["final_h_packed_states"]]
    rust_final_l = [torch.tensor(values) for values in rust_result["final_l_packed_states"]]
    state_diffs: list[tuple[float, str, float]] = []
    for update_index in range(len(updates)):
        torch_h_flat = pytorch_final_h[update_index].flatten()
        torch_l_flat = pytorch_final_l[update_index].flatten()
        state_diffs.append(
            (
                _max_abs(torch_h_flat, rust_final_h[update_index]),
                f"update{update_index}.h_state",
                float(torch_h_flat.float().abs().max().item()),
            )
        )
        state_diffs.append(
            (
                _max_abs(torch_l_flat, rust_final_l[update_index]),
                f"update{update_index}.l_state",
                float(torch_l_flat.float().abs().max().item()),
            )
        )
    state_diffs.sort(reverse=True)
    max_state_diff, max_state_name, max_state_reference_abs = state_diffs[0]
    max_state_relative_to_reference = (
        max_state_diff / max_state_reference_abs if max_state_reference_abs > 0.0 else float("inf")
    )

    print("pytorch_tbptt_objectives=" + ",".join(f"{value:.9g}" for value in pytorch_objectives))
    print(
        "vulkan="
        f"device_index:{rust_result['device_index']} device:{rust_result['device_name']} "
        f"updates:{rust_result['updates']} tokens_per_update:{rust_result['tokens']} "
        f"accumulation_steps:{rust_result['gradient_accumulation_steps']} "
        f"dynamic_loss_scale:{rust_result['dynamic_loss_scale']} "
        f"grad_clip:{rust_result['grad_clip']} "
        f"low_rank_fp16_dW:{rust_result['h_low_rank_native_fp16_parameter_grad_compute_active']} "
        f"submissions:{rust_result['queue_submissions']} "
        f"optimizer_step:{rust_result['optimizer_step']} params:{rust_result['parameter_count']}"
    )
    window_ms = [float(value) for value in rust_result.get("optimizer_window_ms", [])]
    if window_ms:
        mean_window_ms = sum(window_ms) / len(window_ms)
        variance = sum((value - mean_window_ms) ** 2 for value in window_ms) / len(window_ms)
        print(
            "optimizer_window_timing="
            f"count:{len(window_ms)} mean_ms:{mean_window_ms:.6f} "
            f"min_ms:{min(window_ms):.6f} max_ms:{max(window_ms):.6f} "
            f"stddev_ms:{variance ** 0.5:.6f} "
            f"training_elapsed_ms:{float(rust_result.get('training_elapsed_ms', 0.0)):.6f}"
        )
    rust_optimizer_window_gradient_norms = [
        float(value) for value in rust_result.get("optimizer_window_gradient_norms", [])
    ]
    rust_optimizer_window_clip_coefficients = [
        float(value) for value in rust_result.get("optimizer_window_clip_coefficients", [])
    ]
    if rust_optimizer_window_gradient_norms:
        print(
            "optimizer_window_gradient_safety="
            f"grad_clip:{args.grad_clip:g} "
            "pytorch_norms:"
            + ",".join(f"{value:.9g}" for value in pytorch_optimizer_window_gradient_norms)
            + " vulkan_norms:"
            + ",".join(f"{value:.9g}" for value in rust_optimizer_window_gradient_norms)
            + " vulkan_clip_coefficients:"
            + ",".join(f"{value:.9g}" for value in rust_optimizer_window_clip_coefficients)
        )
    print(f"changed_pytorch_state_tensors={len(changed)}")
    print(f"parameter_max_abs_diff={max_diff:.9g} tensor={max_name}")
    for label, plans in (
        ("qualified", qualified_budgeted_plans),
        ("automatic-control", automatic_budgeted_plans),
    ):
        for plan_index, plan in enumerate(plans):
            print(
                f"vulkan_plan={label}[{plan_index}] "
                f"updates={plan['update_start']}..{plan['update_end']} "
                f"microbatch={plan['sequence_microbatch_size']} "
                f"stride={plan['state_checkpoint_stride']} "
                f"h_schedule={plan['h_backward_segment_schedule']} "
                f"l_schedule={plan['l_backward_segment_schedule']} "
                f"h_kernel={plan['h_backward_kernel_geometry']} "
                f"l_kernel={plan['l_backward_kernel_geometry']} "
                f"numerics={plan['rwkv_numerics_policy']} "
                f"profile_records={plan['profile_records']} "
                f"measured_iterations={plan['profile_measured_iterations']} "
                f"online_exploration={plan['online_exploration']}"
            )
    for label, evidence in (
        ("qualified", qualified_synthesis_evidence),
        ("automatic-control", automatic_synthesis_evidence),
    ):
        for plan_index, plan in enumerate(evidence["plans"]):
            print(
                f"vulkan_plan_synthesis={label}[{plan_index}] "
                f"exact_full_arm_records={plan['exact_full_arm_records']} "
                f"runtime_factor_records={plan['runtime_matched_active_factor_records']} "
                f"synthesized={plan['synthesized_from_profiled_factor_evidence']}"
            )
    if args.sequence_microbatch_size is not None:
        print(
            "runtime_profile_tape_geometry="
            f"microbatch={args.sequence_microbatch_size},stride={args.state_checkpoint_stride}"
        )
        print(
            "runtime_profile_vs_budgeted_control_parameter_max_abs_diff="
            f"{profile_control_parameter_diff:.9g}"
        )
        print(
            "runtime_profile_vs_budgeted_control_loss_max_abs_diff="
            f"{profile_control_loss_diff:.9g}"
        )
        if runtime_profile_payload is not None:
            winning_arm = runtime_profile_payload["winning_arm"]
            print(
                "runtime_profile_schedule_only_factors="
                f"gradient_stream_chunk_values={winning_arm.get('gradient_stream_chunk_values')},"
                f"optimizer_broadcast_overlap={winning_arm.get('optimizer_broadcast_overlap')}"
            )
            print(
                "runtime_profile_selected_device_member_match="
                f"{runtime_profile_device_member_match} "
                f"topology_device_count={runtime_profile_topology_device_count}"
            )
    if args.diagnose_g2_uses:
        rust_g2_per_use = rust_result.get("low_rank_g2_per_use", [])
        if len(rust_g2_per_use) != len(pytorch_g2_per_use):
            raise AssertionError(
                "g2 per-use diagnostic update count mismatch: "
                f"PyTorch={len(pytorch_g2_per_use)} Vulkan={len(rust_g2_per_use)}"
            )
        first_pytorch_local_divergence = None
        for update_index, (torch_update, rust_update) in enumerate(
            zip(pytorch_g2_per_use, rust_g2_per_use, strict=True)
        ):
            for tower in ("h", "l"):
                torch_uses = torch_update[tower]
                rust_uses = rust_update[f"{tower}_uses"]
                if torch_uses and rust_uses:
                    torch_norms = [float(values.abs().max().item()) for values in torch_uses]
                    rust_norms = [
                        max((abs(value) for value in values), default=0.0) for values in rust_uses
                    ]
                    print(
                        f"g2_use_norms update={update_index} tower={tower} "
                        f"pytorch={torch_norms} vulkan={rust_norms}"
                    )
                    rust_tensors = [
                        torch.tensor(values, dtype=torch.float32) for values in rust_uses
                    ]
                    for torch_use_index, torch_values in enumerate(torch_uses):
                        closest = min(
                            (
                                _max_abs(torch_values.flatten(), rust_tensor),
                                rust_use_index,
                            )
                            for rust_use_index, rust_tensor in enumerate(rust_tensors)
                        )
                        print(
                            f"g2_use_closest update={update_index} tower={tower} "
                            f"torch_use={torch_use_index} vulkan_use={closest[1]} "
                            f"max_abs_diff={closest[0]:.9g}"
                        )
                if len(torch_uses) != len(rust_uses):
                    print(
                        f"g2_use_count update={update_index} tower={tower} "
                        f"pytorch={len(torch_uses)} vulkan={len(rust_uses)}"
                    )
                aligned_uses = _align_g2_uses_monotonic(torch_uses, rust_uses)
                for torch_use_index, (rust_use_index, torch_values, rust_tensor) in enumerate(
                    aligned_uses
                ):
                    diff = _max_abs(torch_values.flatten(), rust_tensor)
                    print(
                        f"g2_pytorch_vs_vulkan update={update_index} tower={tower} "
                        f"torch_use={torch_use_index} vulkan_use={rust_use_index} "
                        f"max_abs_diff={diff:.9g}"
                    )
                    if diff > 0.0 and first_pytorch_local_divergence is None:
                        first_pytorch_local_divergence = (
                            update_index,
                            tower,
                            rust_use_index,
                            diff,
                        )
        if first_pytorch_local_divergence is None:
            print("g2_first_pytorch_local_divergence=NONE")
        else:
            update_index, tower, use_index, diff = first_pytorch_local_divergence
            print(
                "g2_first_pytorch_local_divergence="
                f"update:{update_index} tower:{tower} use:{use_index} "
                f"max_abs_diff:{diff:.9g}"
            )

        if portable_dw_result is None:
            raise AssertionError("g2 quantization diagnostic did not run its portable dW control")
        portable_g2_per_use = portable_dw_result.get("low_rank_g2_per_use", [])
        if len(portable_g2_per_use) != len(rust_g2_per_use):
            raise AssertionError(
                "native/portable Vulkan g2 trace update count mismatch: "
                f"native={len(rust_g2_per_use)} portable={len(portable_g2_per_use)}"
            )
        for tower in ("h", "l"):
            labels = _labeled_g2_use_labels(
                tower,
                tokens=tokens,
                max_h_steps=config.max_h_steps,
                max_l_steps=config.max_l_steps,
                h_stride=config.h_stride,
            )
            first_quantization_crossing = None
            for update_index, (native_update, portable_update) in enumerate(
                zip(rust_g2_per_use, portable_g2_per_use, strict=True)
            ):
                native_uses = native_update[f"{tower}_uses"]
                portable_uses = portable_update[f"{tower}_uses"]
                if len(native_uses) != len(portable_uses):
                    raise AssertionError(
                        f"update {update_index} {tower} native/portable g2 use-count mismatch: "
                        f"native={len(native_uses)} portable={len(portable_uses)}"
                    )
                if len(native_uses) != len(labels):
                    raise AssertionError(
                        f"update {update_index} {tower} g2 trace has {len(native_uses)} uses, "
                        f"but the static schedule describes {len(labels)}"
                    )
                for use_index, (native_values, portable_values) in enumerate(
                    zip(native_uses, portable_uses, strict=True)
                ):
                    native_tensor = torch.tensor(native_values, dtype=torch.float32)
                    portable_tensor = torch.tensor(portable_values, dtype=torch.float32)
                    diff = _max_abs(native_tensor, portable_tensor)
                    print(
                        f"g2_native_vs_portable update={update_index} tower={tower} "
                        f"use={use_index} {labels[use_index]} max_abs_diff={diff:.9g}"
                    )
                    if diff > 0.0 and first_quantization_crossing is None:
                        first_quantization_crossing = (
                            update_index,
                            use_index,
                            labels[use_index],
                            diff,
                        )
            if first_quantization_crossing is None:
                print(f"g2_first_quantization_crossing tower={tower} NONE")
            else:
                update_index, use_index, label, diff = first_quantization_crossing
                print(
                    f"g2_first_quantization_crossing tower={tower} "
                    f"update={update_index} use={use_index} {label} "
                    f"max_abs_diff={diff:.9g}"
                )
    print(f"recurrent_state_max_abs_diff={max_state_diff:.9g} tensor={max_state_name}")
    print(
        "recurrent_state_max_relative_to_reference="
        f"{max_state_relative_to_reference:.9g} reference_abs_max={max_state_reference_abs:.9g}"
    )
    for state_diff, state_name, reference_abs_max in sorted(
        state_diffs, key=lambda item: item[1]
    ):
        relative = state_diff / reference_abs_max if reference_abs_max > 0.0 else float("inf")
        print(
            f"  state {state_name}: diff={state_diff:.9g} "
            f"reference_abs_max={reference_abs_max:.9g} relative={relative:.9g}"
        )
    print(f"native_inference_max_abs_diff={native_inference_diff:.9g}")
    print(
        "native_architecture_contract_sha256="
        f"{native_metadata['architecture_contract_sha256']}"
    )
    if cuda_inference_diff is None:
        print("cuda_inference=SKIPPED(no CUDA device)")
    else:
        print(f"cuda_inference_max_abs_diff={cuda_inference_diff:.9g}")
    for diff, name in diffs[:12]:
        print(f"  {name}: {diff:.9g}")

    if args.diagnose_gradients:
        rust_gradient_snapshots = rust_result.get("pending_gradients_before_step", [])
        if len(rust_gradient_snapshots) != len(pytorch_gradient_snapshots):
            raise AssertionError(
                "gradient diagnostic snapshot count mismatch: "
                f"rust={len(rust_gradient_snapshots)} torch={len(pytorch_gradient_snapshots)}"
            )
        for update_index, (torch_snapshot, rust_snapshot_list) in enumerate(
            zip(pytorch_gradient_snapshots, rust_gradient_snapshots)
        ):
            # The open Vulkan sequence window still holds the unnormalized
            # supervision-weight sum. Its finish path applies 1 / mass before
            # AdamW. PyTorch's chunk objective is already a weighted mean, so
            # lift it back into the same pre-finish units for this diagnostic.
            pending_unit_scale = supervision_masses[update_index]
            rust_snapshot = {
                item["name"]: torch.tensor(item["values"], dtype=torch.float32)
                for item in rust_snapshot_list
            }
            gradient_diffs: list[tuple[float, str, float, float]] = []
            for name, torch_gradient in torch_snapshot.items():
                if name == "val_proj.weight":
                    # Native LTM alignment applies the controller's deferred
                    # named-gradient scale at finish; the open-window raw
                    # snapshot is therefore intentionally in different units.
                    continue
                rust_gradient = rust_snapshot.get(name)
                if rust_gradient is None or rust_gradient.numel() != torch_gradient.numel():
                    continue
                torch_flat = torch_gradient.flatten() * pending_unit_scale
                rust_flat = rust_gradient.flatten()
                gradient_diffs.append(
                    (
                        _max_abs(torch_flat, rust_flat),
                        name,
                        float(torch_flat.abs().max().item()),
                        float(rust_flat.abs().max().item()),
                    )
                )
            gradient_diffs.sort(reverse=True)
            if not gradient_diffs:
                raise AssertionError(
                    f"gradient diagnostic update {update_index} had no common gradient tensors"
                )
            worst_gradient_diff, worst_gradient_name, _, _ = gradient_diffs[0]
            print(
                f"gradient_update{update_index}_max_abs_diff={worst_gradient_diff:.9g} "
                f"tensor={worst_gradient_name} common={len(gradient_diffs)}"
            )
            for gradient_diff, name, torch_max, rust_max in gradient_diffs[:8]:
                print(
                    f"  grad {name}: diff={gradient_diff:.9g} "
                    f"torch_max={torch_max:.9g} rust_max={rust_max:.9g}"
                )
            tracked_g2_names = {"h_rnn.g2", "l_rnn.g2"}
            tracked_g2_diffs = [
                item for item in gradient_diffs if item[1] in tracked_g2_names
            ]
            for gradient_diff, name, torch_max, rust_max in tracked_g2_diffs:
                print(
                    f"  tracked_grad {name}: diff={gradient_diff:.9g} "
                    f"torch_max={torch_max:.9g} rust_max={rust_max:.9g}"
                )
            if native_fp16_low_rank_parameter_grad and args.dynamic_loss_scale is not None:
                promoted_names = {
                    f"{tower}_rnn.{name}"
                    for tower in ("h", "l")
                    for name in ("w1", "w2", "a1", "a2", "g1", "g2")
                }
                promoted_diffs = [
                    item for item in gradient_diffs if item[1] in promoted_names
                ]
                for gradient_diff, name, torch_max, rust_max in promoted_diffs:
                    print(
                        f"  promoted_grad {name}: diff={gradient_diff:.9g} "
                        f"torch_max={torch_max:.9g} rust_max={rust_max:.9g}"
                    )
            if max_name in torch_snapshot and max_name in rust_snapshot:
                torch_worst_gradient = (
                    torch_snapshot[max_name].flatten()[max_flat_index] * pending_unit_scale
                )
                rust_worst_gradient = rust_snapshot[max_name].flatten()[max_flat_index]
                print(
                    f"  final_worst_element {max_name}[{max_flat_index}]: "
                    f"torch_grad={float(torch_worst_gradient.item()):.9g} "
                    f"rust_grad={float(rust_worst_gradient.item()):.9g} "
                    f"diff={abs(float(torch_worst_gradient.item() - rust_worst_gradient.item())):.9g}"
                )

    if rust_result["updates"] != len(updates):
        raise AssertionError("Vulkan verifier did not execute every optimizer update")
    expected_native_fp16_low_rank_parameter_grad = (
        native_fp16_low_rank_parameter_grad and args.dynamic_loss_scale is not None
    )
    for tower in ("h", "l"):
        field = f"{tower}_low_rank_native_fp16_parameter_grad_compute_active"
        if bool(rust_result[field]) != expected_native_fp16_low_rank_parameter_grad:
            raise AssertionError(
                "Vulkan low-rank native-FP16 parameter-gradient mode mismatch: "
                f"{field}={rust_result[field]!r}, expected "
                f"{expected_native_fp16_low_rank_parameter_grad}"
            )
    expected_optimizer_steps = (
        len(updates) + args.accumulation_steps - 1
    ) // args.accumulation_steps
    expected_queue_submissions = (
        expected_optimizer_steps if args.budgeted_windows else len(updates)
    )
    # Both ordinary clipped and dynamic-loss-scaled clipped boundaries keep
    # normalization, finite detection, global norm, clipping, and predicated
    # AdamW in one Vulkan submission. Telemetry is observed only after that
    # submission retires and never participates in the mutation decision.
    expected_queue_submissions += expected_optimizer_steps
    if rust_result["queue_submissions"] != expected_queue_submissions:
        raise AssertionError(
            "Vulkan labeled submission count did not match the selected execution path: "
            f"expected {expected_queue_submissions}, got {rust_result['queue_submissions']}"
        )
    if rust_result["gradient_accumulation_steps"] != args.accumulation_steps:
        raise AssertionError("Vulkan verifier did not preserve the requested accumulation geometry")
    if rust_result["optimizer_step"] != expected_optimizer_steps:
        raise AssertionError(
            "Vulkan AdamW step counter did not match labeled accumulation windows: "
            f"expected {expected_optimizer_steps}, got {rust_result['optimizer_step']}"
        )
    if len(rust_optimizer_window_gradient_norms) != expected_optimizer_steps:
        raise AssertionError(
            "Vulkan certificate did not report one global gradient norm per optimizer window: "
            f"expected {expected_optimizer_steps}, got {len(rust_optimizer_window_gradient_norms)}"
        )
    if len(rust_optimizer_window_clip_coefficients) != expected_optimizer_steps:
        raise AssertionError(
            "Vulkan certificate did not report one clip coefficient per optimizer window: "
            f"expected {expected_optimizer_steps}, got {len(rust_optimizer_window_clip_coefficients)}"
        )
    if len(pytorch_optimizer_window_gradient_norms) != expected_optimizer_steps:
        raise AssertionError(
            "PyTorch certificate did not report one global gradient norm per optimizer window"
        )
    if not all(math.isfinite(value) for value in rust_optimizer_window_gradient_norms):
        raise AssertionError("Vulkan global gradient safety norms contain a non-finite value")
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in rust_optimizer_window_clip_coefficients
    ):
        raise AssertionError("Vulkan gradient clip coefficients are outside [0, 1]")
    if not rust_result["losses_finite"]:
        raise AssertionError("masked/TBPTT Vulkan token losses are non-finite")
    if args.learning_rate != 0.0 and len(changed) < 8:
        raise AssertionError("PyTorch optimizer trajectory changed too few tensors to be meaningful")
    parameter_tolerance = (
        5.0e-4
        if args.precision == "fp16-storage-fp16-lm-backward"
        else 8.0e-6
    )
    if max_diff > parameter_tolerance:
        raise AssertionError(
            "masked TBPTT optimizer trajectory diverged: "
            f"max parameter diff {max_diff:.9g} at {max_name} "
            f"(tolerance {parameter_tolerance:.9g})"
        )
    if args.precision == "fp32":
        state_tolerance = 4.0e-6
        state_relative_tolerance = None
    elif args.precision == "fp16-storage-fp16-lm-backward":
        state_tolerance = 2.0e-3
        state_relative_tolerance = 1.0e-3
    else:
        state_tolerance = 8.0e-6
        state_relative_tolerance = None
    if max_state_diff > state_tolerance or (
        state_relative_tolerance is not None
        and max_state_relative_to_reference > state_relative_tolerance
    ):
        relative_limit = (
            "disabled"
            if state_relative_tolerance is None
            else f"{state_relative_tolerance:.9g}"
        )
        raise AssertionError(
            "masked recurrent row freezing diverged: "
            f"max state diff {max_state_diff:.9g} at {max_state_name} "
            f"(tolerance {state_tolerance:.9g}, "
            f"relative-to-reference {max_state_relative_to_reference:.9g}, "
            f"relative tolerance {relative_limit})"
        )
    if args.safety_clamp_stress:
        recurrent_ceiling = float(config.recurrent_state_clamp)
        observed_recurrent_peak = max(
            max(float(state.abs().max().item()) for state in pytorch_final_h),
            max(float(state.abs().max().item()) for state in pytorch_final_l),
        )
        if observed_recurrent_peak < recurrent_ceiling * 0.999:
            raise AssertionError(
                "safety clamp stress trajectory never reached the recurrent clamp boundary: "
                f"peak={observed_recurrent_peak:.9g}, ceiling={recurrent_ceiling:.9g}"
            )

    if args.report_json is not None:
        qualification = {
            "schema_version": 1,
            "status": "passed",
            "vulkan_device": selected_device,
            "update_count": len(updates),
            "optimizer_window_timing": {
                "window_ms": window_ms,
                "training_elapsed_ms": float(rust_result.get("training_elapsed_ms", 0.0)),
                "mean_window_ms": (
                    sum(window_ms) / len(window_ms) if window_ms else None
                ),
                "min_window_ms": min(window_ms) if window_ms else None,
                "max_window_ms": max(window_ms) if window_ms else None,
            },
            "training_precision": args.precision,
            "checkpoint_dtype": args.checkpoint_dtype,
            "gradient_safety": {
                "grad_clip": args.grad_clip,
                "pytorch_window_global_l2_norms": pytorch_optimizer_window_gradient_norms,
                "vulkan_window_global_l2_norms": rust_optimizer_window_gradient_norms,
                "vulkan_window_clip_coefficients": rust_optimizer_window_clip_coefficients,
                "all_vulkan_windows_stepped": True,
            },
            "clamp_safety": {
                "stress_mode": args.safety_clamp_stress,
                "recurrent_state_clamp": float(getattr(config, "recurrent_state_clamp", 50.0)),
                "context_state_clamp": float(getattr(config, "context_state_clamp", 50.0)),
                "drift_state_clamp": float(getattr(config, "drift_state_clamp", 5.0)),
                "drift_norm_clamp": float(getattr(config, "drift_norm_clamp", 0.0)),
                "activation_clamp": float(getattr(config, "activation_clamp", 100.0)),
                "halt_logit_clamp": float(getattr(config, "halt_logit_clamp", 30.0)),
                "rwkv_channel_mix_key_clamp": float(
                    getattr(config, "rwkv_channel_mix_key_clamp", 12.0)
                ),
                "rwkv_channel_mix_deepembed_clamp": float(
                    getattr(config, "rwkv_channel_mix_deepembed_clamp", 4.0)
                ),
            },
            "tape_geometry": (
                {
                    "sequence_microbatch_size": args.sequence_microbatch_size,
                    "state_checkpoint_stride": args.state_checkpoint_stride,
                }
                if args.sequence_microbatch_size is not None
                else None
            ),
            "runtime_profile": {
                "path": str(args.runtime_profile) if args.runtime_profile is not None else None,
                "selected_device_member_match": runtime_profile_device_member_match,
                "topology_device_count": runtime_profile_topology_device_count,
                "gradient_stream_chunk_values": (
                    runtime_profile_payload["winning_arm"].get("gradient_stream_chunk_values")
                    if runtime_profile_payload is not None
                    else None
                ),
                "optimizer_broadcast_overlap": (
                    runtime_profile_payload["winning_arm"].get("optimizer_broadcast_overlap")
                    if runtime_profile_payload is not None
                    else None
                ),
            },
            "vulkan_execution_plans": {
                "qualified": qualified_budgeted_plans,
                "automatic_budgeted_control": automatic_budgeted_plans,
            },
            "vulkan_plan_synthesis_evidence": {
                "qualified": qualified_synthesis_evidence,
                "automatic_budgeted_control": automatic_synthesis_evidence,
            },
            "parity": {
                "pytorch_vulkan_parameter_max_abs_diff": max_diff,
                "pytorch_vulkan_parameter_max_abs_diff_tensor": max_name,
                "pytorch_vulkan_recurrent_state_max_abs_diff": max_state_diff,
                "pytorch_vulkan_recurrent_state_max_abs_diff_tensor": max_state_name,
                "runtime_profile_vs_budgeted_control_parameter_max_abs_diff": profile_control_parameter_diff,
                "runtime_profile_vs_budgeted_control_loss_max_abs_diff": profile_control_loss_diff,
                "native_rust_inference_max_abs_diff": native_inference_diff,
                "cuda_inference_max_abs_diff": cuda_inference_diff,
                "cuda_status": "passed" if cuda_inference_diff is not None else "skipped-no-cuda-device",
            },
            "architecture": {
                "revision": native_metadata.get("architecture_revision"),
                "contract_sha256": native_metadata["architecture_contract_sha256"],
            },
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(qualification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"qualification_report={args.report_json}")


if __name__ == "__main__":
    main()
