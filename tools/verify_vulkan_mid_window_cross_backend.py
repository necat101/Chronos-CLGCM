#!/usr/bin/env python3
"""Prove a labeled Vulkan accumulation window can cross PyTorch/CUDA mid-step."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


TRAINING_PRECISION_ENV = "HIERARCHOS_VULKAN_TRAINING_PRECISION"


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore, load_full_model_with_config
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_labeled_sequence_parity import (
    INFERENCE_TOKENS,
    TBPTT_CHUNK_SIZE,
    UpdateFixture,
    _build_updates,
    _install_lm_execution_oracle,
    _max_abs,
    _native_inference,
    _optimizer,
    _pytorch_inference_logits,
    _rust_update_payload,
    _shifted_supervision_mass,
    _train_pytorch_update,
)
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture
import tools.verify_vulkan_worker_refinement_loss_parity as worker_parity
from tools.vulkan_optimizer_bridge import (
    PORTABLE_PARAMETER_STATE_FORMAT,
    VULKAN_TRAINING_FORMAT,
    load_vulkan_training_package_into_torch,
    read_vulkan_pending_gradient_checkpoint,
    read_vulkan_training_manifest,
    read_vulkan_training_replay,
    save_torch_adamw_as_vulkan,
    save_torch_pending_gradients_as_vulkan,
    write_closed_torch_training_manifest_as_vulkan,
    write_vulkan_training_replay,
)


PRECISION_CHOICES = (
    "fp32",
    "fp16-storage-fp32-compute",
    "fp16-storage-parity",
    "fp16-storage-fp16-lm-backward",
)
DEFAULT_OPEN_PRECISION = "fp32"
DEFAULT_RESUME_PRECISION = "fp16-storage-fp32-compute"
DYNAMIC_LOSS_SCALE = 8.0
DYNAMIC_LOSS_SCALE_GROWTH_TRACKER = 17
GRAD_CLIP = 0.1
AGGRESSIVE_FP16_PRECISIONS = (
    "fp16-storage-parity",
    "fp16-storage-fp16-lm-backward",
)
OVERFLOW_PROOF_LOSS_SCALE = 44_000.0
OVERFLOW_PROOF_BACKED_OFF_SCALE = OVERFLOW_PROOF_LOSS_SCALE * 0.5


def _pending_gradient_reference_tolerance(precision: str, *, cuda_available: bool) -> float:
    # The aggressive storage policies deliberately expose fp16 execution
    # quantization in the raw, pre-clipping gradient numerator.  The closed
    # optimizer trajectory below remains the stronger semantic certificate;
    # this bound only prevents the open-window carrier check from pretending
    # those fp16 numerators are fp32-exact.
    if precision in AGGRESSIVE_FP16_PRECISIONS:
        return 1.0e-3
    return 3.0e-4 if cuda_available else 3.0e-5


def _run(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=process_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _vulkan_precision_env(precision: str) -> dict[str, str]:
    env = {TRAINING_PRECISION_ENV: precision}
    if precision == "fp16-storage-parity":
        env[worker_parity.DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV] = "1"
    elif precision == "fp16-storage-fp16-lm-backward":
        env[worker_parity.DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV] = "0"
    return env


def _install_pytorch_precision_oracle(
    model: HierarchosCore,
    precision: str,
    *,
    dynamic_loss_scale_active: bool,
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Mirror the selected Vulkan execution-storage/backward policy in PyTorch."""

    fp16_execution_masters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    if precision != "fp32":
        fp16_execution_masters = worker_parity.install_fp16_execution_storage(model)

    native_fp16_policy = precision == "fp16-storage-fp16-lm-backward"
    source_scaled_fp32_source_adjoint_guard = (
        native_fp16_policy and dynamic_loss_scale_active
    )
    native_fp16_lm_input_grad = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and not worker_parity.env_flag_enabled(
            worker_parity.DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV,
            default=False,
        )
    )
    _install_lm_execution_oracle(
        model,
        precision,
        native_input_grad=native_fp16_lm_input_grad,
    )
    native_fp16_low_rank_backward = (
        native_fp16_policy
        and worker_parity.env_flag_enabled(
            worker_parity.NATIVE_FP16_LOW_RANK_BACKWARD_ENV,
            default=True,
        )
    )
    native_fp16_low_rank_parameter_grad = (
        native_fp16_low_rank_backward
        and worker_parity.env_flag_enabled(
            worker_parity.NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_ENV,
            default=False,
        )
    )
    native_fp16_out_norm_backward = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and worker_parity.env_flag_enabled(
            worker_parity.NATIVE_FP16_OUT_NORM_BACKWARD_ENV,
            default=True,
        )
    )
    native_fp16_projection_backward = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and worker_parity.env_flag_enabled(
            worker_parity.NATIVE_FP16_PROJECTION_BACKWARD_ENV,
            default=True,
        )
    )
    native_fp16_recurrent_projection_backward = (
        dynamic_loss_scale_active
        and worker_parity.env_flag_enabled(
            worker_parity.NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD_ENV,
            default=False,
        )
    )
    worker_parity.install_native_fp16_backward_oracle(
        model,
        include_out_norm=native_fp16_out_norm_backward,
        include_projections=native_fp16_projection_backward,
        include_low_rank=native_fp16_low_rank_backward,
        include_low_rank_inter_stage=False,
        include_low_rank_parameter_grad=(
            native_fp16_low_rank_parameter_grad and dynamic_loss_scale_active
        ),
        include_recurrent_projection=native_fp16_recurrent_projection_backward,
    )
    return fp16_execution_masters


def _fixture_to_device(fixture: UpdateFixture, device: torch.device) -> UpdateFixture:
    return replace(
        fixture,
        input_ids=fixture.input_ids.to(device),
        labels=fixture.labels.to(device),
        attention_mask=fixture.attention_mask.to(device),
        loss_weights=fixture.loss_weights.to(device),
        previous_context=fixture.previous_context.to(device),
        target_context=fixture.target_context.to(device),
        h_state=fixture.h_state.to(device),
        l_state=fixture.l_state.to(device),
    )


def _labeled_case(
    updates: list[UpdateFixture],
    config,
    objective: dict[str, float],
    optimizer_case: dict[str, float],
    *,
    accumulation_steps: int,
    leave_open: bool,
    resume_open: bool = False,
    dynamic_loss_scale: float | None = None,
    dynamic_loss_scale_growth_tracker: int = 0,
    expected_dynamic_loss_scale_overflows: int = 0,
    grad_clip: float = 1.0,
) -> dict[str, object]:
    if not updates:
        raise AssertionError("labeled continuation case requires at least one update")
    batch, tokens = updates[0].input_ids.shape
    return {
        "batch": batch,
        "tokens": tokens,
        "max_h_steps": config.max_h_steps,
        "max_l_steps": config.max_l_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "leave_final_accumulation_open": leave_open,
        "resume_open_accumulation": resume_open,
        "dynamic_loss_scale": dynamic_loss_scale,
        "dynamic_loss_scale_growth_tracker": dynamic_loss_scale_growth_tracker,
        "expected_dynamic_loss_scale_overflows": expected_dynamic_loss_scale_overflows,
        "grad_clip": grad_clip,
        **_rust_update_payload(updates[0]),
        "additional_updates": [_rust_update_payload(update) for update in updates[1:]],
        "objective": objective,
        "optimizer": optimizer_case,
    }


def _run_labeled_vulkan(
    model_dir: Path,
    case: dict[str, object],
    output_package: Path,
    case_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    case_path.write_text(json.dumps(case), encoding="utf-8")
    completed = _run(
        [
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
            str(output_package),
        ],
        env=env,
    )
    return json.loads(completed.stdout)


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _parameter_max_abs_diff(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> tuple[float, str]:
    if lhs.keys() != rhs.keys():
        raise AssertionError(
            "parameter registry changed across backend handoff: "
            f"missing={sorted(lhs.keys() - rhs.keys())} extra={sorted(rhs.keys() - lhs.keys())}"
        )
    maximum = 0.0
    worst = "<none>"
    for name in lhs:
        if lhs[name].shape != rhs[name].shape:
            raise AssertionError(f"parameter shape changed for {name!r}")
        diff = _max_abs(lhs[name], rhs[name])
        if diff > maximum:
            maximum = diff
            worst = name
    return maximum, worst


def _pending_gradient_max_abs_diff(lhs, rhs) -> float:
    if lhs.slot_names != rhs.slot_names:
        raise AssertionError("pending-gradient slot registry changed across PyTorch re-export")
    maximum = 0.0
    for name in lhs.slot_names:
        maximum = max(maximum, _max_abs(lhs.gradients[name], rhs.gradients[name]))
    return maximum


def _pending_vs_torch_gradient_max_abs(
    model: torch.nn.Module,
    pending,
) -> tuple[float, str]:
    aliases = dict(model.named_parameters(remove_duplicate=False))
    maximum = 0.0
    worst = "<none>"
    for name in pending.slot_names:
        parameter = aliases.get(name)
        if parameter is None and name == "lm_head.weight":
            parameter = getattr(getattr(model, "lm_head", None), "weight", None)
        if parameter is None:
            raise AssertionError(f"PyTorch model has no parameter alias for Vulkan slot {name!r}")
        if parameter.grad is None:
            raise AssertionError(f"PyTorch reference produced no gradient for Vulkan slot {name!r}")
        diff = _max_abs(
            parameter.grad.detach().float().cpu(),
            pending.gradients[name].reshape(parameter.shape).float(),
        )
        if diff > maximum:
            maximum = diff
            worst = name
    return maximum, worst


def _assert_token_tape_replay(
    replay_state: dict[str, object] | None,
    *,
    expected_h: torch.Tensor,
    expected_l: torch.Tensor,
    expected_tokens: int,
    expected_batch: int,
    label: str,
) -> float:
    if not isinstance(replay_state, dict):
        raise AssertionError(f"{label} omitted portable recurrent replay state")
    tape = replay_state.get("token_tape_replay")
    if not isinstance(tape, dict):
        raise AssertionError(f"{label} omitted token_tape_replay")
    if int(tape.get("tokens", -1)) != expected_tokens or int(tape.get("batch", -1)) != expected_batch:
        raise AssertionError(
            f"{label} token-tape geometry changed: "
            f"tokens={tape.get('tokens')!r} batch={tape.get('batch')!r}"
        )
    actual_h = tape.get("final_h_packed_state")
    actual_l = tape.get("final_l_packed_state")
    if not torch.is_tensor(actual_h) or not torch.is_tensor(actual_l):
        raise AssertionError(f"{label} token-tape replay is missing H/L tensor carriers")
    h_diff = _max_abs(actual_h.reshape(-1).float(), expected_h.reshape(-1).float())
    l_diff = _max_abs(actual_l.reshape(-1).float(), expected_l.reshape(-1).float())
    maximum = max(h_diff, l_diff)
    if maximum > 1.0e-7:
        raise AssertionError(
            f"{label} recurrent token-tape replay drifted by {maximum:.9g}"
        )
    return maximum


def _session_loss_scaling(session: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(session, dict):
        raise AssertionError("portable training session is missing")
    policy = session.get("execution_policy")
    if not isinstance(policy, dict):
        raise AssertionError("portable execution policy is missing")
    scaling = policy.get("loss_scaling")
    if not isinstance(scaling, dict) or scaling.get("mode") != "dynamic":
        raise AssertionError("portable dynamic-loss-scaler state is missing")
    return scaling


def _closed_pytorch_session_after_success(
    session: dict[str, object] | None,
    bridge_device: torch.device,
) -> dict[str, object]:
    if not isinstance(session, dict):
        raise AssertionError("cannot advance a missing portable training session")
    closed = deepcopy(session)
    policy = closed.get("execution_policy")
    if not isinstance(policy, dict):
        raise AssertionError("cannot advance a training session without execution_policy")
    scaling = policy.get("loss_scaling")
    if not isinstance(scaling, dict) or scaling.get("mode") != "dynamic":
        raise AssertionError("cannot advance a training session without dynamic loss scaling")
    tracker = int(scaling.get("growth_tracker", 0)) + 1
    interval = int(scaling.get("growth_interval", 0))
    scale = float(scaling["scale"])
    if interval <= 0:
        raise AssertionError(f"invalid portable loss-scaler growth interval {interval}")
    if tracker >= interval:
        scale *= float(scaling.get("growth_factor", 1.0))
        tracker = 0
    scaling["scale"] = scale
    scaling["growth_tracker"] = tracker
    scaling["pending_gradients_scaled"] = False
    policy["source_backend"] = f"pytorch-{bridge_device.type}"
    policy["compute_dtype"] = "float32"
    policy["autocast_enabled"] = False
    return closed


def _prove_dynamic_loss_scale_overflow_resume(
    *,
    temp: Path,
    source_package: Path,
    updates: list[UpdateFixture],
    config,
    objective: dict[str, float],
    optimizer_case: dict[str, float],
    precision: str,
    env: dict[str, str],
) -> dict[str, object]:
    """Prove a saved fp16 accumulation can overflow, back off, and resume."""

    if precision not in AGGRESSIVE_FP16_PRECISIONS:
        raise AssertionError(
            f"overflow/backoff proof requires an aggressive fp16 policy, got {precision!r}"
        )

    open_package = temp / "overflow-open-window"
    backed_off_package = temp / "overflow-backed-off"
    followup_package = temp / "overflow-followup"

    # This ordering is intentional.  On the deterministic tiny fixture the
    # second update can be checkpointed with finite pending fp16 gradients at
    # this scale; adding the first update crosses the native fp16 range only
    # when the restored window is closed.
    open_result = _run_labeled_vulkan(
        source_package,
        _labeled_case(
            [updates[1]],
            config,
            objective,
            optimizer_case,
            accumulation_steps=2,
            leave_open=True,
            dynamic_loss_scale=OVERFLOW_PROOF_LOSS_SCALE,
            dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER,
            grad_clip=GRAD_CLIP,
        ),
        open_package,
        temp / "overflow-open-case.json",
        env=env,
    )
    if not bool(open_result["accumulation_open"]):
        raise AssertionError("overflow proof failed to preserve the first microbatch as an open window")
    if int(open_result["optimizer_step"]) != 0:
        raise AssertionError("overflow proof advanced AdamW before the restored window was closed")
    if int(open_result["dynamic_loss_scale_overflow_count"]) != 0:
        raise AssertionError("overflow proof reported an overflow before the checkpoint boundary")
    if float(open_result["dynamic_loss_scale_after"]) != OVERFLOW_PROOF_LOSS_SCALE:
        raise AssertionError("open overflow-stress checkpoint changed its loss scale")
    if int(open_result["pending_gradient_tensor_count"]) <= 0:
        raise AssertionError("open overflow-stress checkpoint contains no pending gradients")

    overflow_result = _run_labeled_vulkan(
        open_package,
        _labeled_case(
            [updates[0]],
            config,
            objective,
            optimizer_case,
            accumulation_steps=2,
            leave_open=False,
            resume_open=True,
            dynamic_loss_scale=OVERFLOW_PROOF_LOSS_SCALE,
            dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER,
            expected_dynamic_loss_scale_overflows=1,
            grad_clip=GRAD_CLIP,
        ),
        backed_off_package,
        temp / "overflow-close-case.json",
        env=env,
    )
    if bool(overflow_result["accumulation_open"]):
        raise AssertionError("overflowed restored window remained open after the safety boundary")
    if int(overflow_result["optimizer_step"]) != 0:
        raise AssertionError("dynamic-loss-scale overflow incorrectly advanced AdamW")
    if int(overflow_result["dynamic_loss_scale_overflow_count"]) != 1:
        raise AssertionError("restored overflow proof did not observe exactly one real overflow")
    if overflow_result["dynamic_loss_scale_window_overflowed"] != [True]:
        raise AssertionError("restored overflow proof did not mark its closed window overflowed")
    if overflow_result["dynamic_loss_scale_window_stepped"] != [False]:
        raise AssertionError("overflowed restored window incorrectly reported an optimizer step")
    if overflow_result["dynamic_loss_scale_window_scale_before"] != [OVERFLOW_PROOF_LOSS_SCALE]:
        raise AssertionError("overflowed restored window did not begin at the checkpointed scale")
    if overflow_result["dynamic_loss_scale_window_scale_after"] != [OVERFLOW_PROOF_BACKED_OFF_SCALE]:
        raise AssertionError("overflowed restored window did not apply the configured 0.5 backoff")
    if float(overflow_result["dynamic_loss_scale_after"]) != OVERFLOW_PROOF_BACKED_OFF_SCALE:
        raise AssertionError("backed-off checkpoint did not persist the reduced dynamic loss scale")
    if int(overflow_result["dynamic_loss_scale_growth_tracker"]) != 0:
        raise AssertionError("loss-scale overflow did not reset the growth tracker")
    if int(overflow_result["pending_gradient_tensor_count"]) != 0:
        raise AssertionError("overflowed checkpoint retained gradients that should have been discarded")

    backed_off_manifest = read_vulkan_training_manifest(backed_off_package)
    backed_off_session = backed_off_manifest.get("training_session")
    backed_off_scaling = _session_loss_scaling(
        backed_off_session if isinstance(backed_off_session, dict) else None
    )
    if float(backed_off_scaling["scale"]) != OVERFLOW_PROOF_BACKED_OFF_SCALE:
        raise AssertionError("portable training session lost the backed-off loss scale")
    if int(backed_off_scaling["growth_tracker"]) != 0:
        raise AssertionError("portable training session lost the overflow-reset growth tracker")
    if backed_off_scaling.get("pending_gradients_scaled") is not False:
        raise AssertionError("backed-off checkpoint persisted scaled pending-gradient state")

    source_model, _ = load_full_model_with_config(str(source_package), torch.device("cpu"))
    backed_off_model, _ = load_full_model_with_config(str(backed_off_package), torch.device("cpu"))
    skipped_step_diff, skipped_step_worst = _parameter_max_abs_diff(
        _parameter_snapshot(source_model),
        _parameter_snapshot(backed_off_model),
    )
    if skipped_step_diff != 0.0:
        raise AssertionError(
            "overflowed optimizer window mutated FP32 master parameters: "
            f"{skipped_step_diff:.9g} at {skipped_step_worst}"
        )

    followup_result = _run_labeled_vulkan(
        backed_off_package,
        _labeled_case(
            [updates[2]],
            config,
            objective,
            optimizer_case,
            accumulation_steps=2,
            leave_open=False,
            dynamic_loss_scale=OVERFLOW_PROOF_BACKED_OFF_SCALE,
            dynamic_loss_scale_growth_tracker=0,
            grad_clip=GRAD_CLIP,
        ),
        followup_package,
        temp / "overflow-followup-case.json",
        env=env,
    )
    if bool(followup_result["accumulation_open"]):
        raise AssertionError("backed-off continuation unexpectedly left its optimizer window open")
    if int(followup_result["optimizer_step"]) != 1:
        raise AssertionError("backed-off continuation did not recover with exactly one AdamW step")
    if int(followup_result["dynamic_loss_scale_overflow_count"]) != 0:
        raise AssertionError("backed-off continuation overflowed again instead of recovering")
    if followup_result["dynamic_loss_scale_window_overflowed"] != [False]:
        raise AssertionError("backed-off continuation did not report a finite optimizer window")
    if followup_result["dynamic_loss_scale_window_stepped"] != [True]:
        raise AssertionError("backed-off continuation did not step after the finite window")
    if followup_result["dynamic_loss_scale_window_scale_before"] != [OVERFLOW_PROOF_BACKED_OFF_SCALE]:
        raise AssertionError("backed-off continuation did not restore the reduced loss scale")
    if followup_result["dynamic_loss_scale_window_scale_after"] != [OVERFLOW_PROOF_BACKED_OFF_SCALE]:
        raise AssertionError("successful backed-off continuation changed a non-growing scale")
    if float(followup_result["dynamic_loss_scale_after"]) != OVERFLOW_PROOF_BACKED_OFF_SCALE:
        raise AssertionError("successful backed-off continuation did not preserve its scale")
    if int(followup_result["dynamic_loss_scale_growth_tracker"]) != 1:
        raise AssertionError("successful backed-off continuation did not advance scaler history")
    if int(followup_result["pending_gradient_tensor_count"]) != 0:
        raise AssertionError("successful backed-off continuation exported stray pending gradients")

    followup_model, _ = load_full_model_with_config(str(followup_package), torch.device("cpu"))
    recovered_step_diff, recovered_step_worst = _parameter_max_abs_diff(
        _parameter_snapshot(backed_off_model),
        _parameter_snapshot(followup_model),
    )
    if recovered_step_diff <= 0.0:
        raise AssertionError("backed-off continuation claimed a step without changing model parameters")

    native_inference, _ = _native_inference(followup_package)
    pytorch_inference = _pytorch_inference_logits(followup_model, torch.device("cpu"))
    inference_diff = _max_abs(pytorch_inference, native_inference)
    torch.testing.assert_close(native_inference, pytorch_inference, rtol=2.0e-4, atol=2.0e-5)

    return {
        "open_scale": OVERFLOW_PROOF_LOSS_SCALE,
        "backed_off_scale": OVERFLOW_PROOF_BACKED_OFF_SCALE,
        "skipped_step_parameter_max_abs": skipped_step_diff,
        "recovered_step_parameter_max_abs": recovered_step_diff,
        "recovered_step_parameter_tensor": recovered_step_worst,
        "native_inference_max_abs": inference_diff,
    }


def _assert_ltm_controller_runtime_parity(
    pytorch_config,
    vulkan_controller: dict[str, object],
    *,
    float_tolerance: float,
) -> float:
    field_map = {
        "updates": "val_proj_alignment_updates",
        "last": "val_proj_alignment_last",
        "ema": "val_proj_alignment_ema",
        "best": "val_proj_alignment_best",
        "writer_norm": "val_proj_writer_norm",
        "ready": "val_proj_trained",
    }
    maximum = 0.0
    for controller_name, config_name in field_map.items():
        actual = getattr(pytorch_config, config_name)
        expected = vulkan_controller[controller_name]
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise AssertionError(
                    "LTM controller optional state differs across backends: "
                    f"{config_name}={actual!r} Vulkan.{controller_name}={expected!r}"
                )
            continue
        if isinstance(expected, bool):
            if bool(actual) != expected:
                raise AssertionError(
                    f"LTM controller readiness differs: PyTorch={actual!r} Vulkan={expected!r}"
                )
            continue
        if isinstance(expected, int):
            if int(actual) != expected:
                raise AssertionError(
                    "LTM controller integer state differs across backends: "
                    f"{config_name}={actual!r} Vulkan.{controller_name}={expected!r}"
                )
            continue
        diff = abs(float(actual) - float(expected))
        maximum = max(maximum, diff)
        if diff > float_tolerance:
            raise AssertionError(
                "LTM controller floating state differs across backends: "
                f"{config_name}={actual!r} Vulkan.{controller_name}={expected!r} "
                f"diff={diff:.9g} limit={float_tolerance:.9g}"
            )
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail instead of using the CPU bridge when CUDA is unavailable",
    )
    parser.add_argument(
        "--open-precision",
        choices=PRECISION_CHOICES,
        default=DEFAULT_OPEN_PRECISION,
        help="Vulkan precision policy used to create the open accumulation checkpoint",
    )
    parser.add_argument(
        "--resume-precision",
        choices=PRECISION_CHOICES,
        default=DEFAULT_RESUME_PRECISION,
        help="Vulkan precision policy used for resumed/returning Vulkan legs",
    )
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false")
    bridge_device = torch.device("cuda" if cuda_available else "cpu")

    torch.manual_seed(20260819)
    if cuda_available:
        torch.cuda.manual_seed_all(20260819)
    config = tiny_coherent_config(32)
    config.z_loss_weight = 0.003
    # Exercise the adaptive LTM writer/readiness controller across the same
    # open-window handoff instead of keeping it outside the session contract.
    config.ltm_value_alignment_weight = 0.021
    config.ltm_value_alignment_stride = 2
    config.ltm_value_alignment_min_updates = 1
    config.memory_gate_warmup_steps = 0.0
    config.memory_gate_warmup_floor = 0.0
    config.detach_every_n_steps = 32

    seed_model = HierarchosCore(config).train()
    _make_nontrivial_memory_fixture(seed_model, config)
    updates = _build_updates(seed_model, config, 3)
    objective = {
        "z_loss_weight": config.z_loss_weight,
        "ponder_loss_weight": 0.013,
        "commitment_loss_weight": 0.37,
        "max_ce_loss_for_backward": 0.0,
        "max_ponder_cost_for_backward": 0.0,
        "max_commitment_cost_for_backward": 2.0,
    }
    optimizer_case = {
        "lr": 3.0e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "weight_decay": 0.07,
    }
    open_precision = args.open_precision
    resume_precision = args.resume_precision
    open_vulkan_env = _vulkan_precision_env(open_precision)
    resume_vulkan_env = _vulkan_precision_env(resume_precision)

    first_mass = _shifted_supervision_mass(updates[0])
    second_mass = _shifted_supervision_mass(updates[1])
    window_mass = first_mass + second_mass
    overflow_proof: dict[str, object] | None = None

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-mid-window-") as temp_dir:
        temp = Path(temp_dir)
        source_package = temp / "source"
        open_package = temp / "vulkan-open-window"
        vulkan_resumed_package = temp / "vulkan-current-resumed"
        vulkan_cross_precision_package = temp / "vulkan-cross-precision-resumed"
        control_package = temp / "vulkan-control-closed"
        pytorch_package = temp / f"pytorch-{bridge_device.type}-closed"
        returned_package = temp / "vulkan-returned"
        export_model(seed_model.eval(), config, source_package)

        open_result = _run_labeled_vulkan(
            source_package,
            _labeled_case(
                [updates[0]],
                config,
                objective,
                optimizer_case,
                accumulation_steps=2,
                leave_open=True,
                dynamic_loss_scale=DYNAMIC_LOSS_SCALE,
                dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER,
                grad_clip=GRAD_CLIP,
            ),
            open_package,
            temp / "open-case.json",
            env=open_vulkan_env,
        )
        if open_result["training_precision_policy"] != open_precision:
            raise AssertionError(
                "open Vulkan leg did not honor requested precision policy: "
                f"{open_result['training_precision_policy']!r} != {open_precision!r}"
            )
        if not bool(open_result["accumulation_open"]):
            raise AssertionError("Vulkan labeled runner closed the requested mid-window checkpoint")
        if int(open_result["optimizer_step"]) != 0:
            raise AssertionError("mid-window Vulkan checkpoint advanced AdamW before the window closed")
        if int(open_result["pending_gradient_tensor_count"]) <= 0:
            raise AssertionError("mid-window Vulkan checkpoint contains no pending gradients")

        open_manifest = read_vulkan_training_manifest(open_package)
        if open_manifest.get("format") != VULKAN_TRAINING_FORMAT:
            raise AssertionError(
                f"open Vulkan package did not use current portable training state: {open_manifest.get('format')!r}"
            )
        parameter_state = open_manifest.get("parameter_state")
        if not isinstance(parameter_state, dict) or parameter_state.get("format") != PORTABLE_PARAMETER_STATE_FORMAT:
            raise AssertionError("open Vulkan package omitted the portable FP32 master contract")
        mirrors = parameter_state.get("execution_mirrors")
        if not isinstance(mirrors, dict) or mirrors.get("persistence") != "derived":
            raise AssertionError("open Vulkan package did not mark execution mirrors as derived state")
        if parameter_state.get("parameter_aliases") != [
            {"canonical": "lm_head.weight", "alias": "tok_emb.weight"}
        ]:
            raise AssertionError("open Vulkan package omitted the tied lm_head/tok_emb master alias")
        if open_manifest.get("val_proj_gradient_weight_applied") is not True:
            raise AssertionError(
                "open Vulkan package did not canonicalize val_proj pending objective weight"
            )
        open_session = open_manifest.get("training_session")
        open_scaling = _session_loss_scaling(open_session if isinstance(open_session, dict) else None)
        if float(open_scaling["scale"]) != DYNAMIC_LOSS_SCALE:
            raise AssertionError("open Vulkan checkpoint changed the dynamic loss scale")
        if int(open_scaling["growth_tracker"]) != DYNAMIC_LOSS_SCALE_GROWTH_TRACKER:
            raise AssertionError("open Vulkan checkpoint changed the loss-scaler growth tracker")
        if open_scaling.get("pending_gradients_scaled") is not False:
            raise AssertionError("portable pending gradients must be canonical and unscaled")
        if not isinstance(open_session, dict):
            raise AssertionError("open Vulkan checkpoint omitted its portable training session")
        effective_training_config = open_session.get("effective_training_config")
        if not isinstance(effective_training_config, dict) or abs(
            float(effective_training_config.get("grad_clip", -1.0)) - GRAD_CLIP
        ) > 1.0e-8:
            raise AssertionError("open Vulkan checkpoint did not persist gradient clipping state")
        open_replay = read_vulkan_training_replay(open_package, open_manifest)
        open_replay_max_abs = _assert_token_tape_replay(
            open_replay,
            expected_h=torch.tensor(open_result["final_h_packed_states"][-1]),
            expected_l=torch.tensor(open_result["final_l_packed_states"][-1]),
            expected_tokens=int(open_result["tokens"]),
            expected_batch=int(open_result["batch"]),
            label="open Vulkan checkpoint",
        )

        vulkan_resumed_result = _run_labeled_vulkan(
            open_package,
            _labeled_case(
                [updates[1]],
                config,
                objective,
                optimizer_case,
                accumulation_steps=2,
                leave_open=False,
                resume_open=True,
                dynamic_loss_scale=DYNAMIC_LOSS_SCALE,
                dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER,
                grad_clip=GRAD_CLIP,
            ),
            vulkan_resumed_package,
            temp / "vulkan-current-resume-case.json",
            env=open_vulkan_env,
        )
        if bool(vulkan_resumed_result["accumulation_open"]):
            raise AssertionError("current Vulkan self-resume did not close the restored accumulation window")
        if int(vulkan_resumed_result["optimizer_step"]) != 1:
            raise AssertionError(
                "current Vulkan self-resume did not restore step 0 and close exactly one optimizer window"
            )
        if int(vulkan_resumed_result["dynamic_loss_scale_growth_tracker"]) != (
            DYNAMIC_LOSS_SCALE_GROWTH_TRACKER + 1
        ):
            raise AssertionError("same-precision Vulkan resume did not advance scaler history")

        cross_precision_result = _run_labeled_vulkan(
            open_package,
            _labeled_case(
                [updates[1]],
                config,
                objective,
                optimizer_case,
                accumulation_steps=2,
                leave_open=False,
                resume_open=True,
                dynamic_loss_scale=DYNAMIC_LOSS_SCALE,
                dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER,
                grad_clip=GRAD_CLIP,
            ),
            vulkan_cross_precision_package,
            temp / "vulkan-cross-precision-resume-case.json",
            env=resume_vulkan_env,
        )
        if cross_precision_result["training_precision_policy"] != resume_precision:
            raise AssertionError(
                "cross-precision Vulkan leg did not rebuild execution mirrors under the destination policy: "
                f"{cross_precision_result['training_precision_policy']!r} != {resume_precision!r}"
            )
        if bool(cross_precision_result["accumulation_open"]):
            raise AssertionError("cross-precision Vulkan resume did not close the restored window")
        if int(cross_precision_result["optimizer_step"]) != 1:
            raise AssertionError("cross-precision Vulkan resume did not advance AdamW exactly once")
        if int(cross_precision_result["dynamic_loss_scale_growth_tracker"]) != (
            DYNAMIC_LOSS_SCALE_GROWTH_TRACKER + 1
        ):
            raise AssertionError("cross-precision Vulkan resume did not preserve scaler history")
        if not cross_precision_result["optimizer_window_clip_coefficients"]:
            raise AssertionError("cross-precision Vulkan resume did not report clipping state")
        if float(cross_precision_result["optimizer_window_clip_coefficients"][0]) >= 1.0:
            raise AssertionError(
                "gradient clipping was not active; the cross-precision checkpoint did not exercise the clipping boundary"
            )
        pytorch_model, pytorch_config = load_full_model_with_config(
            str(open_package), bridge_device
        )
        pytorch_model.train()
        optimizer = _optimizer(pytorch_model, optimizer_case)
        loaded = load_vulkan_training_package_into_torch(
            pytorch_model, optimizer, open_package
        )
        pytorch_fp16_execution_masters = _install_pytorch_precision_oracle(
            pytorch_model,
            open_precision,
            dynamic_loss_scale_active=True,
        )
        if loaded.ltm_alignment_controller is None:
            raise AssertionError("Vulkan package did not restore its LTM alignment controller")
        if int(loaded.ltm_alignment_controller["sampled_rows_in_window"]) <= 0:
            raise AssertionError(
                "open Vulkan accumulation package lost LTM sampled-row controller state"
            )
        if loaded.pending_gradients is None:
            raise AssertionError("PyTorch did not receive Vulkan's open gradient numerator")
        loaded_scaling = _session_loss_scaling(loaded.session_state)
        if float(loaded_scaling["scale"]) != DYNAMIC_LOSS_SCALE:
            raise AssertionError("PyTorch did not restore the Vulkan dynamic loss scale")
        if int(loaded_scaling["growth_tracker"]) != DYNAMIC_LOSS_SCALE_GROWTH_TRACKER:
            raise AssertionError("PyTorch did not restore the Vulkan loss-scaler growth tracker")
        loaded_replay_max_abs = _assert_token_tape_replay(
            loaded.replay_state,
            expected_h=torch.tensor(open_result["final_h_packed_states"][-1]),
            expected_l=torch.tensor(open_result["final_l_packed_states"][-1]),
            expected_tokens=int(open_result["tokens"]),
            expected_batch=int(open_result["batch"]),
            label="Vulkan -> PyTorch replay handoff",
        )
        if loaded.pytorch_accumulation_normalization != "weighted-token":
            raise AssertionError("Vulkan weighted-token normalization did not map into PyTorch")
        if abs(loaded.consumed_weighted_token_mass - first_mass) > 2.0e-6:
            raise AssertionError(
                "Vulkan/PyTorch consumed supervision mass changed: "
                f"{loaded.consumed_weighted_token_mass} vs {first_mass}"
            )

        first_reference_model, first_reference_config = load_full_model_with_config(
            str(source_package), bridge_device
        )
        first_reference_model.train()
        first_reference_optimizer = _optimizer(first_reference_model, optimizer_case)
        first_reference_fp16_execution_masters = _install_pytorch_precision_oracle(
            first_reference_model,
            open_precision,
            dynamic_loss_scale_active=True,
        )
        _train_pytorch_update(
            first_reference_model,
            first_reference_optimizer,
            _fixture_to_device(updates[0], bridge_device),
            objective,
            first_reference_config,
            zero_grad=True,
            step_optimizer=False,
            loss_scale=first_mass * DYNAMIC_LOSS_SCALE,
            fp16_execution_masters=first_reference_fp16_execution_masters,
        )
        for parameter in first_reference_model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(DYNAMIC_LOSS_SCALE)
        pending_reference_max_abs, pending_reference_worst = (
            _pending_vs_torch_gradient_max_abs(first_reference_model, loaded.pending_gradients)
        )
        pending_reference_tolerance = _pending_gradient_reference_tolerance(
            open_precision,
            cuda_available=cuda_available,
        )
        if pending_reference_max_abs > pending_reference_tolerance:
            raise AssertionError(
                "Vulkan open-window pending numerator differs from fresh PyTorch first microbatch: "
                f"{pending_reference_max_abs:.9g} at {pending_reference_worst} "
                f"(limit {pending_reference_tolerance:.9g})"
            )

        # Before mutating the live window, prove Python can emit the exact same
        # canonical pending registry that a future Vulkan continuation can read.
        reexported_pending = save_torch_pending_gradients_as_vulkan(
            pytorch_model,
            optimizer,
            temp / "pytorch-open-gradients.safetensors",
            template_checkpoint=open_package / "optimizer.safetensors",
        )
        reread_pending = read_vulkan_pending_gradient_checkpoint(
            temp / "pytorch-open-gradients.safetensors"
        )
        pending_reexport_max_abs = _pending_gradient_max_abs_diff(
            reexported_pending, reread_pending
        )
        if pending_reexport_max_abs != 0.0:
            raise AssertionError(
                f"PyTorch pending-gradient re-export drifted by {pending_reexport_max_abs:.9g}"
            )

        # The portable gradient file is deliberately unscaled. Rehydrate it
        # into the GradScaler domain before accumulating the second microbatch,
        # then combine unscale + weighted-token normalization in the divisor
        # immediately before the clipping/AdamW boundary.
        for parameter in pytorch_model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(DYNAMIC_LOSS_SCALE)
        (
            _,
            pytorch_final_h,
            pytorch_final_l,
            pytorch_window_gradient_norm,
        ) = _train_pytorch_update(
            pytorch_model,
            optimizer,
            _fixture_to_device(updates[1], bridge_device),
            objective,
            pytorch_config,
            zero_grad=False,
            step_optimizer=True,
            loss_scale=second_mass * DYNAMIC_LOSS_SCALE,
            gradient_divisor_before_step=window_mass * DYNAMIC_LOSS_SCALE,
            grad_clip=GRAD_CLIP,
            fp16_execution_masters=pytorch_fp16_execution_masters,
            update_ltm_controller=True,
        )
        if pytorch_window_gradient_norm is None or pytorch_window_gradient_norm <= GRAD_CLIP:
            raise AssertionError(
                "PyTorch closure did not exercise the same active gradient-clipping boundary"
            )

        # The same two labeled microbatches, kept entirely in Vulkan, are the
        # numerical oracle for closing the backend-spanning window.
        control_result = _run_labeled_vulkan(
            source_package,
            _labeled_case(
                updates[:2],
                config,
                objective,
                optimizer_case,
                accumulation_steps=2,
                leave_open=False,
                dynamic_loss_scale=DYNAMIC_LOSS_SCALE,
                dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER,
                grad_clip=GRAD_CLIP,
            ),
            control_package,
            temp / "control-case.json",
            env=open_vulkan_env,
        )
        if bool(control_result["accumulation_open"]) or int(control_result["optimizer_step"]) != 1:
            raise AssertionError("all-Vulkan control did not close exactly one accumulation window")
        control_manifest = read_vulkan_training_manifest(control_package)
        control_controller = control_manifest.get("ltm_alignment_controller")
        if not isinstance(control_controller, dict):
            raise AssertionError("all-Vulkan control did not export LTM controller state")
        controller_tolerance = 7.5e-4 if cuda_available else 5.0e-4
        controller_runtime_max_abs = _assert_ltm_controller_runtime_parity(
            pytorch_config,
            control_controller,
            float_tolerance=controller_tolerance,
        )
        control_model, _ = load_full_model_with_config(str(control_package), torch.device("cpu"))
        vulkan_resumed_model, _ = load_full_model_with_config(
            str(vulkan_resumed_package), torch.device("cpu")
        )
        vulkan_resume_diff, vulkan_resume_worst = _parameter_max_abs_diff(
            _parameter_snapshot(vulkan_resumed_model), _parameter_snapshot(control_model)
        )
        if vulkan_resume_diff > 8.0e-6:
            raise AssertionError(
                "Vulkan open-checkpoint self-resume diverged from uninterrupted Vulkan control: "
                f"{vulkan_resume_diff:.9g} at {vulkan_resume_worst}"
            )
        cross_precision_model, _ = load_full_model_with_config(
            str(vulkan_cross_precision_package), torch.device("cpu")
        )
        cross_precision_diff, cross_precision_worst = _parameter_max_abs_diff(
            _parameter_snapshot(cross_precision_model), _parameter_snapshot(control_model)
        )
        if cross_precision_diff > 7.5e-4:
            raise AssertionError(
                "cross-precision Vulkan open-window resume drifted too far from the FP32 control: "
                f"{cross_precision_diff:.9g} at {cross_precision_worst}"
            )
        cross_backend_diff, cross_backend_worst = _parameter_max_abs_diff(
            _parameter_snapshot(pytorch_model), _parameter_snapshot(control_model)
        )
        trajectory_tolerance = 5.0e-4 if cuda_available else 8.0e-6
        if cross_backend_diff > trajectory_tolerance:
            raise AssertionError(
                "Vulkan -> PyTorch accumulation closure diverged from all-Vulkan control: "
                f"{cross_backend_diff:.9g} at {cross_backend_worst} "
                f"(limit {trajectory_tolerance:.9g})"
            )

        export_model(pytorch_model, pytorch_config, pytorch_package)
        pytorch_optimizer = save_torch_adamw_as_vulkan(
            pytorch_model,
            optimizer,
            pytorch_package / "optimizer.safetensors",
            template_checkpoint=open_package / "optimizer.safetensors",
        )
        if pytorch_optimizer.step != 1:
            raise AssertionError(
                f"PyTorch closure did not advance portable AdamW exactly once: {pytorch_optimizer.step}"
            )
        pytorch_session = _closed_pytorch_session_after_success(
            loaded.session_state, bridge_device
        )
        pytorch_scaling = _session_loss_scaling(pytorch_session)
        if int(pytorch_scaling["growth_tracker"]) != DYNAMIC_LOSS_SCALE_GROWTH_TRACKER + 1:
            raise AssertionError("PyTorch closure did not advance portable scaler history")
        pytorch_manifest = write_closed_torch_training_manifest_as_vulkan(
            pytorch_model,
            pytorch_package,
            open_manifest,
            pytorch_optimizer,
            training_session=pytorch_session,
        )
        pytorch_manifest = write_vulkan_training_replay(
            pytorch_package,
            {
                "completed_epoch": int(pytorch_session["completed_epoch"]),
                "mid_epoch_step": int(pytorch_session["mid_epoch_step"]),
                "optimizer_grouping_version": int(
                    pytorch_session["optimizer_grouping_version"]
                ),
                "effective_training_config": pytorch_session[
                    "effective_training_config"
                ],
                "execution_policy": pytorch_session["execution_policy"],
                "token_tape_replay": {
                    "final_h_packed_state": pytorch_final_h,
                    "final_l_packed_state": pytorch_final_l,
                    "tokens": int(updates[1].input_ids.shape[1]),
                    "batch": int(updates[1].input_ids.shape[0]),
                },
            },
        )
        pytorch_replay = read_vulkan_training_replay(pytorch_package, pytorch_manifest)
        pytorch_replay_max_abs = _assert_token_tape_replay(
            pytorch_replay,
            expected_h=pytorch_final_h,
            expected_l=pytorch_final_l,
            expected_tokens=int(updates[1].input_ids.shape[1]),
            expected_batch=int(updates[1].input_ids.shape[0]),
            label="PyTorch -> Vulkan replay handoff",
        )
        reexported_controller = pytorch_manifest.get("ltm_alignment_controller")
        if not isinstance(reexported_controller, dict):
            raise AssertionError("PyTorch re-export omitted the portable LTM controller")
        for field in ("updates", "ready"):
            if reexported_controller[field] != control_controller[field]:
                raise AssertionError(
                    "PyTorch portable LTM controller re-export differs from Vulkan control: "
                    f"{field}={reexported_controller[field]!r} vs {control_controller[field]!r}"
                )

        returned_result = _run_labeled_vulkan(
            pytorch_package,
            _labeled_case(
                [updates[2]],
                config,
                objective,
                optimizer_case,
                accumulation_steps=2,
                leave_open=False,
                dynamic_loss_scale=DYNAMIC_LOSS_SCALE,
                dynamic_loss_scale_growth_tracker=DYNAMIC_LOSS_SCALE_GROWTH_TRACKER + 1,
                grad_clip=GRAD_CLIP,
            ),
            returned_package,
            temp / "return-case.json",
            env=resume_vulkan_env,
        )
        if bool(returned_result["accumulation_open"]):
            raise AssertionError("returning Vulkan continuation unexpectedly left a window open")
        if int(returned_result["optimizer_step"]) != 2:
            raise AssertionError(
                "returning Vulkan continuation did not restore PyTorch AdamW step 1 and advance to 2"
            )
        if returned_result["training_precision_policy"] != resume_precision:
            raise AssertionError("returning Vulkan leg did not honor the cross-precision destination policy")
        if int(returned_result["dynamic_loss_scale_growth_tracker"]) != (
            DYNAMIC_LOSS_SCALE_GROWTH_TRACKER + 2
        ):
            raise AssertionError("returning Vulkan leg did not continue portable scaler history")
        returned_manifest = read_vulkan_training_manifest(returned_package)
        returned_replay = read_vulkan_training_replay(returned_package, returned_manifest)
        returned_replay_max_abs = _assert_token_tape_replay(
            returned_replay,
            expected_h=torch.tensor(returned_result["final_h_packed_states"][-1]),
            expected_l=torch.tensor(returned_result["final_l_packed_states"][-1]),
            expected_tokens=int(returned_result["tokens"]),
            expected_batch=int(returned_result["batch"]),
            label="returned Vulkan replay",
        )

        returned_model, _ = load_full_model_with_config(str(returned_package), bridge_device)
        pytorch_inference = _pytorch_inference_logits(returned_model, bridge_device)
        native_inference, native_metadata = _native_inference(returned_package)
        native_inference_diff = _max_abs(pytorch_inference, native_inference)
        torch.testing.assert_close(
            native_inference,
            pytorch_inference,
            rtol=3.0e-4,
            atol=4.0e-5,
        )

        if open_precision in AGGRESSIVE_FP16_PRECISIONS:
            overflow_proof = _prove_dynamic_loss_scale_overflow_resume(
                temp=temp,
                source_package=source_package,
                updates=updates,
                config=config,
                objective=objective,
                optimizer_case=optimizer_case,
                precision=open_precision,
                env=open_vulkan_env,
            )

    print(f"bridge_device={bridge_device.type}")
    if cuda_available:
        print(f"nvidia_device={torch.cuda.get_device_name(0)}")
    else:
        print("cuda_training_continuation=SKIPPED(no CUDA device; identical bridge exercised on CPU)")
    print(f"tbptt_chunk_size={TBPTT_CHUNK_SIZE}")
    print(f"inference_tokens={','.join(map(str, INFERENCE_TOKENS))}")
    print(f"vulkan_open_optimizer_step={open_result['optimizer_step']}")
    print(f"vulkan_open_pending_tensors={open_result['pending_gradient_tensor_count']}")
    print(f"open_vulkan_training_precision={open_precision}")
    print(f"resume_vulkan_training_precision={resume_precision}")
    print(f"dynamic_loss_scale={DYNAMIC_LOSS_SCALE:.9g}")
    print(f"grad_clip={GRAD_CLIP:.9g}")
    print(f"consumed_supervision_mass={first_mass:.9g}")
    print(f"closing_supervision_mass={second_mass:.9g}")
    print(f"window_supervision_mass={window_mass:.9g}")
    print(f"pending_gradient_reexport_max_abs={pending_reexport_max_abs:.9g}")
    print(f"open_replay_max_abs={open_replay_max_abs:.9g}")
    print(f"loaded_replay_max_abs={loaded_replay_max_abs:.9g}")
    print(f"pytorch_replay_max_abs={pytorch_replay_max_abs:.9g}")
    print(f"returned_replay_max_abs={returned_replay_max_abs:.9g}")
    print(
        "pending_vs_fresh_pytorch_max_abs="
        f"{pending_reference_max_abs:.9g} tensor={pending_reference_worst}"
    )
    print(f"ltm_controller_runtime_max_abs={controller_runtime_max_abs:.9g}")
    print(
        "vulkan_current_self_resume_parameter_max_abs="
        f"{vulkan_resume_diff:.9g} tensor={vulkan_resume_worst}"
    )
    print(
        "vulkan_cross_precision_resume_parameter_max_abs="
        f"{cross_precision_diff:.9g} tensor={cross_precision_worst}"
    )
    print(
        "cross_backend_window_parameter_max_abs="
        f"{cross_backend_diff:.9g} tensor={cross_backend_worst}"
    )
    print(f"returned_vulkan_optimizer_step={returned_result['optimizer_step']}")
    print(f"native_vs_{bridge_device.type}_inference_max_abs={native_inference_diff:.9g}")
    print(
        "native_architecture_contract_sha256="
        f"{native_metadata['architecture_contract_sha256']}"
    )
    if overflow_proof is not None:
        print(
            "dynamic_loss_scale_overflow_resume="
            f"overflow:{overflow_proof['open_scale']:.9g}->"
            f"{overflow_proof['backed_off_scale']:.9g} "
            f"skipped_step_parameter_max_abs={overflow_proof['skipped_step_parameter_max_abs']:.9g} "
            f"recovered_step_parameter_max_abs={overflow_proof['recovered_step_parameter_max_abs']:.9g} "
            f"tensor={overflow_proof['recovered_step_parameter_tensor']} "
            f"native_inference_max_abs={overflow_proof['native_inference_max_abs']:.9g}"
        )
    else:
        print("dynamic_loss_scale_overflow_resume=SKIPPED(non-aggressive fp16 open policy)")
    print("Vulkan -> portable open gradients -> PyTorch -> Vulkan/native session: PASS")


if __name__ == "__main__":
    main()
