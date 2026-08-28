#!/usr/bin/env python3
"""Bridge Hierarchos Vulkan AdamW SafeTensors into a PyTorch optimizer.

The Vulkan trainer deliberately keeps model weights in the ordinary Hierarchos
``model.safetensors`` package.  Its companion ``optimizer.safetensors`` stores
only FP32 AdamW moments, indexed by the exact canonical model tensor name.  This
module maps that state into/out of ``torch.optim.AdamW`` without introducing a
pickle checkpoint or a backend-specific model layout.

The tied token embedding needs special care: PyTorch enumerates the shared
parameter first as ``tok_emb.weight`` while the portable model/Vulkan contract
uses ``lm_head.weight``.  Parameter lookup therefore retains all PyTorch aliases
and resolves the exact Vulkan slot name before deduplicating by object identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from hierarchos.utils.rosa import ROSAState


VULKAN_ADAMW_FORMAT_V1 = "hierarchos-vulkan-adamw-v1"
VULKAN_ADAMW_FORMAT_V2 = "hierarchos-vulkan-adamw-v2"
VULKAN_ADAMW_FORMAT = "hierarchos-vulkan-adamw-v3"
_SUPPORTED_VULKAN_ADAMW_FORMATS = {
    VULKAN_ADAMW_FORMAT_V1,
    VULKAN_ADAMW_FORMAT_V2,
    VULKAN_ADAMW_FORMAT,
}
VULKAN_ADAMW_DECAY_CLASS = "decay"
VULKAN_ADAMW_NO_DECAY_CLASS = "no-decay"
VULKAN_PENDING_GRADIENT_FORMAT = "hierarchos-vulkan-pending-gradients-v1"
VULKAN_TRAINING_FORMAT_V1 = "hierarchos-vulkan-training-state-v1"
VULKAN_TRAINING_FORMAT_V2 = "hierarchos-vulkan-training-state-v2"
VULKAN_TRAINING_FORMAT_V3 = "hierarchos-vulkan-training-state-v3"
VULKAN_TRAINING_FORMAT_V4 = "hierarchos-vulkan-training-state-v4"
VULKAN_TRAINING_FORMAT_V5 = "hierarchos-vulkan-training-state-v5"
VULKAN_TRAINING_FORMAT = "hierarchos-vulkan-training-state-v6"
_SUPPORTED_VULKAN_TRAINING_FORMATS = {
    VULKAN_TRAINING_FORMAT_V1,
    VULKAN_TRAINING_FORMAT_V2,
    VULKAN_TRAINING_FORMAT_V3,
    VULKAN_TRAINING_FORMAT_V4,
    VULKAN_TRAINING_FORMAT_V5,
    VULKAN_TRAINING_FORMAT,
}
VULKAN_PORTABLE_REPLAY_FORMAT = "hierarchos-portable-training-replay-v1"
VULKAN_PORTABLE_REPLAY_FILENAME = "training_replay.json"
VULKAN_PORTABLE_REPLAY_TENSOR_FILENAME = "training_replay.safetensors"
VULKAN_TRAINING_SESSION_FORMAT_V1 = "hierarchos-portable-training-session-v1"
VULKAN_TRAINING_SESSION_FORMAT = "hierarchos-portable-training-session-v2"
_SUPPORTED_VULKAN_TRAINING_SESSION_FORMATS = {
    VULKAN_TRAINING_SESSION_FORMAT_V1,
    VULKAN_TRAINING_SESSION_FORMAT,
}
PORTABLE_DATA_STREAM_CURSOR_FORMAT = "hierarchos-data-stream-rng-cursor-v1"
PORTABLE_EXECUTION_POLICY_FORMAT = "hierarchos-training-execution-policy-v1"
PORTABLE_PARAMETER_STATE_FORMAT_V1 = "hierarchos-portable-parameter-state-v1"
PORTABLE_PARAMETER_STATE_FORMAT = "hierarchos-portable-parameter-state-v2"
PORTABLE_SAMPLER_RNG_ALGORITHM = "splitmix64-fisher-yates-v1"
CANONICAL_COUNTER_RNG_ALGORITHM = "philox4x32-10-word-v1"

_PORTABLE_REPLAY_CHECKPOINT_KEYS = (
    "running_states",
    "run_identity",
    "best_metric_state",
    # Native Vulkan labeled/TBPTT checkpoints expose the recurrent carrier as
    # a backend-neutral token-tape replay record.  Keep that record when the
    # package crosses PyTorch so a later Vulkan destination does not lose the
    # exact H/L continuation boundary merely because Python rewrote the
    # checkpoint envelope.
    "token_tape_replay",
)


@dataclass(frozen=True)
class VulkanAdamWCheckpoint:
    step: int
    slot_names: tuple[str, ...]
    slot_steps: dict[str, int]
    slot_decay_classes: dict[str, str | None]
    exp_avg: dict[str, torch.Tensor]
    exp_avg_sq: dict[str, torch.Tensor]


@dataclass(frozen=True)
class VulkanPendingGradientCheckpoint:
    slot_names: tuple[str, ...]
    gradients: dict[str, torch.Tensor]


@dataclass(frozen=True)
class VulkanTrainingPackage:
    manifest: dict[str, object]
    parameter_state: dict[str, object] | None
    optimizer: VulkanAdamWCheckpoint
    pending_gradients: VulkanPendingGradientCheckpoint | None
    pytorch_accumulation_normalization: str | None
    consumed_weighted_token_mass: float
    target_weighted_token_mass: float | None
    replay_state: dict[str, object] | None
    session_state: dict[str, object] | None
    ltm_alignment_controller: dict[str, object] | None


_LTM_ALIGNMENT_POLICY_FIELDS = {
    "weight": "ltm_value_alignment_weight",
    "stride": "ltm_value_alignment_stride",
    "min_updates": "ltm_value_alignment_min_updates",
    "ready_threshold": "ltm_value_alignment_ready_threshold",
    "ema_decay": "ltm_value_alignment_ema_decay",
    "writer_max_norm": "ltm_value_writer_max_norm",
}

_LTM_ALIGNMENT_RUNTIME_FIELDS = {
    "updates": "val_proj_alignment_updates",
    "last": "val_proj_alignment_last",
    "ema": "val_proj_alignment_ema",
    "best": "val_proj_alignment_best",
    "writer_norm": "val_proj_writer_norm",
    "ready": "val_proj_trained",
}


def _model_config_candidates(model: torch.nn.Module):
    candidates = [model, getattr(model, "_orig_mod", None)]
    base_model = getattr(model, "base_model", None)
    candidates.extend(
        [
            base_model,
            getattr(base_model, "model", None),
            getattr(getattr(base_model, "model", None), "_orig_mod", None),
        ]
    )
    seen_models: set[int] = set()
    seen_configs: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen_models:
            continue
        seen_models.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is None or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        yield config


def _config_read(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_write(config, name: str, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _finite_nonnegative_float(value, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Vulkan LTM alignment controller {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"Vulkan LTM alignment controller {field} must be finite and non-negative"
        )
    return parsed


def _nonnegative_int(value, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Vulkan LTM alignment controller {field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Vulkan LTM alignment controller {field} must be an integer") from exc
    if parsed < (1 if positive else 0) or parsed != value:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(
            f"Vulkan LTM alignment controller {field} must be a {qualifier} integer"
        )
    return parsed


def _parse_ltm_alignment_controller(
    manifest: dict[str, object],
) -> dict[str, object] | None:
    raw = manifest.get("ltm_alignment_controller")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Vulkan training manifest ltm_alignment_controller must be an object")

    parsed: dict[str, object] = {
        "weight": _finite_nonnegative_float(raw.get("weight"), "weight"),
        "stride": _nonnegative_int(raw.get("stride"), "stride", positive=True),
        "min_updates": _nonnegative_int(
            raw.get("min_updates"), "min_updates", positive=True
        ),
        "ready_threshold": _finite_nonnegative_float(
            raw.get("ready_threshold"), "ready_threshold"
        ),
        "ema_decay": _finite_nonnegative_float(raw.get("ema_decay"), "ema_decay"),
        "writer_max_norm": _finite_nonnegative_float(
            raw.get("writer_max_norm"), "writer_max_norm"
        ),
        "updates": _nonnegative_int(raw.get("updates"), "updates"),
        "ready": raw.get("ready"),
    }
    if parsed["ema_decay"] >= 1.0:
        raise ValueError("Vulkan LTM alignment controller ema_decay must be less than 1")
    if not isinstance(parsed["ready"], bool):
        raise ValueError("Vulkan LTM alignment controller ready must be boolean")
    for field in ("last", "ema", "best", "writer_norm"):
        value = raw.get(field)
        parsed[field] = (
            None if value is None else _finite_nonnegative_float(value, field)
        )
    for field in (
        "sampled_rows_in_window",
        "sampled_rows_in_controller_microbatch",
        "last_step_sampled_rows",
        "last_step_controller_sampled_rows",
    ):
        parsed[field] = _nonnegative_int(raw.get(field, 0), field)
    if not bool(manifest.get("accumulation_open", False)) and (
        parsed["sampled_rows_in_window"] != 0
        or parsed["sampled_rows_in_controller_microbatch"] != 0
    ):
        raise ValueError(
            "closed Vulkan training package has non-zero LTM accumulation counters"
        )
    return parsed


def _validate_ltm_alignment_controller_against_torch(
    model: torch.nn.Module,
    controller: dict[str, object],
) -> list[object]:
    configs = list(_model_config_candidates(model))
    if not configs:
        raise ValueError("PyTorch model has no config for Vulkan LTM controller restore")
    for config in configs:
        for controller_name, config_name in _LTM_ALIGNMENT_POLICY_FIELDS.items():
            current = _config_read(config, config_name, None)
            if current is None:
                raise ValueError(
                    f"PyTorch config is missing LTM alignment policy field {config_name!r}"
                )
            expected = controller[controller_name]
            if isinstance(expected, int):
                if int(current) != expected:
                    raise ValueError(
                        "Vulkan/PyTorch LTM alignment policy mismatch for "
                        f"{config_name}: model={current!r} checkpoint={expected!r}"
                    )
            else:
                current_float = float(current)
                if not math.isclose(
                    current_float,
                    float(expected),
                    rel_tol=2.0e-6,
                    abs_tol=2.0e-8,
                ):
                    raise ValueError(
                        "Vulkan/PyTorch LTM alignment policy mismatch for "
                        f"{config_name}: model={current!r} checkpoint={expected!r}"
                    )
    return configs


def _apply_ltm_alignment_controller_to_torch(
    configs: Iterable[object],
    controller: dict[str, object],
) -> None:
    for config in configs:
        for controller_name, config_name in _LTM_ALIGNMENT_RUNTIME_FIELDS.items():
            _config_write(config, config_name, controller[controller_name])


def restore_vulkan_ltm_alignment_controller_into_torch(
    model: torch.nn.Module,
    manifest: dict[str, object],
) -> dict[str, object] | None:
    """Restore portable Vulkan LTM writer-controller state into PyTorch config.

    Policy fields are validated against the model package instead of silently
    overriding architecture semantics. Runtime readiness/EMA fields are then
    copied into every live model config wrapper so normal PyTorch training and
    inference observe the exact controller state exported by Vulkan.
    """

    controller = _parse_ltm_alignment_controller(manifest)
    if controller is None:
        return None
    configs = _validate_ltm_alignment_controller_against_torch(model, controller)
    _apply_ltm_alignment_controller_to_torch(configs, controller)
    return controller


def capture_torch_ltm_alignment_controller(
    model: torch.nn.Module,
    *,
    template_controller: dict[str, object] | None = None,
    accumulation_open: bool = False,
) -> dict[str, object]:
    """Encode current PyTorch LTM readiness state in Vulkan's portable schema."""

    configs = list(_model_config_candidates(model))
    if not configs:
        raise ValueError("PyTorch model has no config for portable LTM controller capture")
    config = configs[0]
    controller: dict[str, object] = {}
    for controller_name, config_name in _LTM_ALIGNMENT_POLICY_FIELDS.items():
        value = _config_read(config, config_name, None)
        if controller_name in {"stride", "min_updates"}:
            controller[controller_name] = _nonnegative_int(
                value, config_name, positive=True
            )
        else:
            controller[controller_name] = _finite_nonnegative_float(value, config_name)
    for controller_name, config_name in _LTM_ALIGNMENT_RUNTIME_FIELDS.items():
        value = _config_read(config, config_name, None)
        if controller_name == "updates":
            controller[controller_name] = _nonnegative_int(value or 0, config_name)
        elif controller_name == "ready":
            controller[controller_name] = bool(value)
        else:
            controller[controller_name] = (
                None
                if value is None
                else _finite_nonnegative_float(value, config_name)
            )

    template = template_controller or {}
    if accumulation_open:
        controller["sampled_rows_in_window"] = _nonnegative_int(
            template.get("sampled_rows_in_window", 0), "sampled_rows_in_window"
        )
        controller["sampled_rows_in_controller_microbatch"] = _nonnegative_int(
            template.get("sampled_rows_in_controller_microbatch", 0),
            "sampled_rows_in_controller_microbatch",
        )
    else:
        controller["sampled_rows_in_window"] = 0
        controller["sampled_rows_in_controller_microbatch"] = 0
    controller["last_step_sampled_rows"] = _nonnegative_int(
        template.get("last_step_sampled_rows", 0), "last_step_sampled_rows"
    )
    controller["last_step_controller_sampled_rows"] = _nonnegative_int(
        template.get("last_step_controller_sampled_rows", 0),
        "last_step_controller_sampled_rows",
    )
    return controller


def _canonical_parameter_state(
    master_file: object,
    *,
    parameter_format: str = PORTABLE_PARAMETER_STATE_FORMAT,
) -> dict[str, object]:
    if not isinstance(master_file, str) or not master_file or Path(master_file).name != master_file:
        raise ValueError("Vulkan training manifest model_file must be a package-local filename")
    if parameter_format not in {PORTABLE_PARAMETER_STATE_FORMAT_V1, PORTABLE_PARAMETER_STATE_FORMAT}:
        raise ValueError(f"unsupported portable parameter-state format {parameter_format!r}")
    state: dict[str, object] = {
        "format": parameter_format,
        "master_file": master_file,
        "trainable_master_dtype": "float32",
        "layout": "pytorch-row-major",
        "optimizer_binding": "canonical-tensor-name",
        "execution_mirrors": {
            "persistence": "derived",
            "rebuild_from": "trainable-fp32-master",
            "rebuild_on_load": True,
            "destination_policy": "runtime-selected",
        },
    }
    if parameter_format == PORTABLE_PARAMETER_STATE_FORMAT:
        state["parameter_aliases"] = [
            {"canonical": "lm_head.weight", "alias": "tok_emb.weight"}
        ]
    return state


def _normalize_parameter_state(manifest: dict[str, object]) -> dict[str, object] | None:
    raw = manifest.get("parameter_state")
    checkpoint_format = manifest.get("format")
    if raw is None:
        if checkpoint_format in {VULKAN_TRAINING_FORMAT_V5, VULKAN_TRAINING_FORMAT}:
            raise ValueError("Vulkan v5+ training manifest is missing portable parameter_state")
        return None
    if not isinstance(raw, dict) or raw.get("format") not in {
        PORTABLE_PARAMETER_STATE_FORMAT_V1,
        PORTABLE_PARAMETER_STATE_FORMAT,
    }:
        raise ValueError("Vulkan training manifest parameter_state has an unsupported format")
    model_file = manifest.get("model_file")
    parameter_format = str(raw["format"])
    if checkpoint_format == VULKAN_TRAINING_FORMAT and parameter_format != PORTABLE_PARAMETER_STATE_FORMAT:
        raise ValueError("Vulkan v6 training manifest requires portable parameter-state v2")
    expected = _canonical_parameter_state(model_file, parameter_format=parameter_format)
    if raw != expected:
        raise ValueError(
            "Vulkan parameter_state must declare canonical FP32 masters, tied-parameter aliases, and derived runtime-selected execution mirrors"
        )
    return expected


def write_closed_torch_training_manifest_as_vulkan(
    model: torch.nn.Module,
    package_dir: str | Path,
    source_manifest: dict[str, object],
    optimizer_checkpoint: VulkanAdamWCheckpoint,
    *,
    training_session: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write a closed PyTorch continuation using Vulkan's backend-neutral envelope.

    ``training_precision_policy`` is retained as producer provenance only. The
    portable parameter contract makes execution mirrors derived state, so a
    later Vulkan destination is free to rebuild them under a different runtime
    precision policy from the canonical FP32 masters.
    """

    package_dir = Path(package_dir)
    manifest = dict(source_manifest)
    manifest["parameter_state"] = _canonical_parameter_state(manifest.get("model_file"))
    _validate_torch_parameter_alias_contract(model, manifest["parameter_state"])
    _validate_torch_master_dtype_contract(
        model, manifest["parameter_state"], optimizer_checkpoint.slot_names
    )
    _validate_portable_master_file(
        package_dir, manifest["parameter_state"], optimizer_checkpoint.slot_names
    )
    manifest["format"] = VULKAN_TRAINING_FORMAT
    manifest["optimizer_step"] = optimizer_checkpoint.step
    manifest["optimizer_tensor_count"] = len(optimizer_checkpoint.slot_names)
    manifest["gradient_file"] = None
    manifest["gradient_tensor_count"] = 0
    manifest["accumulation_open"] = False
    manifest["accumulation_normalization"] = None
    manifest["accumulation_consumed_token_count"] = 0
    manifest["accumulation_consumed_supervision_mass"] = None
    manifest["accumulation_target_token_count"] = None
    manifest["accumulation_target_supervision_mass"] = None
    manifest["lm_head_gradient_topology"] = None
    manifest["val_proj_active_in_window"] = False
    manifest["val_proj_gradient_weight_applied"] = False
    manifest["ltm_alignment_pytorch_tbptt_weighting_in_window"] = None
    manifest["portable_replay_file"] = None
    manifest["portable_replay_tensor_file"] = None
    manifest["ltm_alignment_controller"] = capture_torch_ltm_alignment_controller(
        model,
        template_controller=_parse_ltm_alignment_controller(source_manifest),
        accumulation_open=False,
    )
    if training_session is not None:
        candidate = dict(manifest)
        candidate["training_session"] = training_session
        candidate["completed_epoch"] = training_session.get("completed_epoch")
        candidate["mid_epoch_step"] = training_session.get("mid_epoch_step")
        normalized_session = _parse_vulkan_training_session(candidate)
        if normalized_session is None:
            raise ValueError("portable PyTorch continuation supplied an empty training_session")
        manifest["training_session"] = normalized_session
        manifest["completed_epoch"] = normalized_session["completed_epoch"]
        manifest["mid_epoch_step"] = normalized_session["mid_epoch_step"]
    path = package_dir / "training_state.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_vulkan_training_manifest(package_dir: str | Path) -> dict[str, object]:
    """Read and validate the JSON envelope for a portable Vulkan training package."""

    package_dir = Path(package_dir)
    manifest_path = package_dir / "training_state.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read Vulkan training manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("Vulkan training manifest must be a JSON object")
    checkpoint_format = manifest.get("format")
    if checkpoint_format not in _SUPPORTED_VULKAN_TRAINING_FORMATS:
        raise ValueError(
            f"unsupported Hierarchos Vulkan training manifest format {checkpoint_format!r}"
        )
    _normalize_parameter_state(manifest)
    return manifest


def is_vulkan_training_package(path: str | Path | None) -> bool:
    if path is None:
        return False
    package_dir = Path(path).expanduser()
    if not package_dir.is_dir() or not (package_dir / "training_state.json").is_file():
        return False
    try:
        read_vulkan_training_manifest(package_dir)
    except ValueError:
        return False
    return True


def _nonnegative_session_int(value, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Vulkan training session {field} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"Vulkan training session {field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Vulkan training session {field} must be an integer") from exc
    if parsed < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"Vulkan training session {field} must be {qualifier}")
    return parsed


def _u64_session_int(value, field: str) -> int:
    parsed = _nonnegative_session_int(value, field)
    if parsed > (1 << 64) - 1:
        raise ValueError(f"Vulkan training session {field} exceeds unsigned 64-bit range")
    return parsed


def _finite_session_float(value, field: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Vulkan training session {field} must be numeric") from exc
    if not math.isfinite(parsed) or (parsed <= 0.0 if positive else parsed < 0.0):
        qualifier = "finite and positive" if positive else "finite and non-negative"
        raise ValueError(f"Vulkan training session {field} must be {qualifier}")
    return parsed


def _session_lr_list(value, field: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Vulkan training session {field} must be a non-empty LR list")
    return [
        _finite_session_float(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def _normalize_training_session_lr_state(
    raw: object,
    field: str,
    *,
    scheduler_state_dict: object = None,
    require_group_lrs: bool = False,
) -> dict[str, object] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Vulkan training session {field} must be an object")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"Vulkan training session {field}.enabled must be boolean")
    normalized: dict[str, object] = {"enabled": enabled}
    if not enabled:
        return normalized

    step = _nonnegative_session_int(raw.get("step", 0), f"{field}.step")
    total_steps = _nonnegative_session_int(
        raw.get("total_steps"), f"{field}.total_steps", minimum=1
    )
    # LambdaLR.last_epoch continues increasing after the configured curve
    # horizon; only the LR lambda clamps to its terminal value.  A Vulkan
    # checkpoint produced after extending an exact-resume run can therefore
    # legitimately carry step > total_steps and must remain loadable by
    # PyTorch/CUDA.
    max_lr = _finite_session_float(raw.get("max_lr"), f"{field}.max_lr", positive=True)
    min_lr = _finite_session_float(raw.get("min_lr", 0.0), f"{field}.min_lr")
    if min_lr > max_lr:
        raise ValueError(f"Vulkan training session {field}.min_lr exceeds max_lr")
    normalized.update(
        {
            "step": step,
            "total_steps": total_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
        }
    )

    for name in ("warmup_steps", "resolved_warmup_steps"):
        if name in raw and raw[name] is not None:
            warmup = _nonnegative_session_int(raw[name], f"{field}.{name}")
            if warmup > total_steps:
                raise ValueError(
                    f"Vulkan training session {field}.{name} exceeds total_steps"
                )
            normalized[name] = warmup
    if "warmup_ratio" in raw and raw["warmup_ratio"] is not None:
        ratio = _finite_session_float(raw["warmup_ratio"], f"{field}.warmup_ratio")
        if ratio > 1.0:
            raise ValueError(f"Vulkan training session {field}.warmup_ratio must be <= 1")
        normalized["warmup_ratio"] = ratio

    scheduler_payload = scheduler_state_dict if isinstance(scheduler_state_dict, dict) else {}
    base_lrs = _session_lr_list(
        raw.get("base_lrs", scheduler_payload.get("base_lrs")),
        f"{field}.base_lrs",
    )
    last_lrs = _session_lr_list(
        raw.get("last_lrs", scheduler_payload.get("_last_lr")),
        f"{field}.last_lrs",
    )
    if (base_lrs is None) != (last_lrs is None):
        raise ValueError(
            f"Vulkan training session {field} must provide base_lrs and last_lrs together"
        )
    if require_group_lrs and base_lrs is None:
        raise ValueError(
            f"Vulkan training session {field} must carry base_lrs/last_lrs "
            "for cross-backend resume"
        )
    if base_lrs is not None:
        if len(base_lrs) != len(last_lrs):
            raise ValueError(
                f"Vulkan training session {field} base_lrs/last_lrs group counts differ"
            )
        normalized["base_lrs"] = base_lrs
        normalized["last_lrs"] = last_lrs
        step_count = raw.get("step_count", scheduler_payload.get("_step_count", step + 1))
        normalized["step_count"] = _nonnegative_session_int(
            step_count, f"{field}.step_count", minimum=1
        )
    return normalized


def _json_native_session_mapping(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Vulkan training session {field} must be an object")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Vulkan training session {field} must contain only JSON-native finite values"
        ) from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Vulkan training session {field} must decode to an object")
    return decoded


def _normalize_data_stream_cursor(raw: object) -> dict[str, object] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("format") != PORTABLE_DATA_STREAM_CURSOR_FORMAT:
        raise ValueError("Vulkan training session data_stream_cursor has an unsupported format")
    sampler_kind = raw.get("sampler_kind")
    if sampler_kind not in {"epoch-shuffle", "length-grouped-batch"}:
        raise ValueError("Vulkan training session data_stream_cursor has an unsupported sampler_kind")
    if raw.get("rng_algorithm") != PORTABLE_SAMPLER_RNG_ALGORITHM:
        raise ValueError("Vulkan training session data_stream_cursor has an unsupported RNG algorithm")
    normalized = {
        "format": PORTABLE_DATA_STREAM_CURSOR_FORMAT,
        "sampler_kind": sampler_kind,
        "rng_algorithm": PORTABLE_SAMPLER_RNG_ALGORITHM,
        "seed": _nonnegative_session_int(raw.get("seed"), "data_stream_cursor.seed"),
        "epoch": _nonnegative_session_int(raw.get("epoch", 0), "data_stream_cursor.epoch"),
        "batch_cursor": _nonnegative_session_int(
            raw.get("batch_cursor", 0), "data_stream_cursor.batch_cursor"
        ),
        "dataset_size": _nonnegative_session_int(
            raw.get("dataset_size"), "data_stream_cursor.dataset_size", minimum=1
        ),
        "batch_size": _nonnegative_session_int(
            raw.get("batch_size"), "data_stream_cursor.batch_size", minimum=1
        ),
        "shuffle": raw.get("shuffle"),
        "drop_last": raw.get("drop_last", False),
    }
    if not isinstance(normalized["shuffle"], bool) or not isinstance(normalized["drop_last"], bool):
        raise ValueError("Vulkan training session data_stream_cursor shuffle/drop_last must be boolean")
    if sampler_kind == "length-grouped-batch":
        normalized["bucket_size"] = _nonnegative_session_int(
            raw.get("bucket_size"), "data_stream_cursor.bucket_size", minimum=1
        )
        preserve_order = raw.get("preserve_order", False)
        if not isinstance(preserve_order, bool):
            raise ValueError("Vulkan training session data_stream_cursor preserve_order must be boolean")
        normalized["preserve_order"] = preserve_order
    return normalized


def _normalize_execution_policy(raw: object) -> dict[str, object] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("format") != PORTABLE_EXECUTION_POLICY_FORMAT:
        raise ValueError("Vulkan training session execution_policy has an unsupported format")
    compute_dtype = raw.get("compute_dtype")
    if compute_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Vulkan training session execution_policy compute_dtype is unsupported")
    autocast_enabled = raw.get("autocast_enabled", compute_dtype != "float32")
    if not isinstance(autocast_enabled, bool):
        raise ValueError("Vulkan training session execution_policy autocast_enabled must be boolean")
    stochastic_rng = raw.get("stochastic_rng", {"mode": "none", "state_required": False})
    if not isinstance(stochastic_rng, dict):
        raise ValueError("Vulkan training session execution_policy stochastic_rng must be an object")
    rng_mode = stochastic_rng.get("mode")
    if rng_mode not in {"none", "backend-native", "canonical-counter"}:
        raise ValueError("Vulkan training session execution_policy stochastic_rng mode is unsupported")
    state_required = stochastic_rng.get("state_required", rng_mode == "backend-native")
    if not isinstance(state_required, bool):
        raise ValueError("Vulkan training session execution_policy RNG state_required must be boolean")
    canonical_counter = stochastic_rng.get("canonical_counter")
    normalized_stochastic_rng: dict[str, object] = {
        "mode": rng_mode,
        "state_required": state_required,
    }
    if rng_mode == "canonical-counter":
        if state_required:
            raise ValueError(
                "Vulkan canonical-counter RNG must not require backend-native RNG state"
            )
        if not isinstance(canonical_counter, dict):
            raise ValueError(
                "Vulkan canonical-counter RNG is missing canonical_counter state"
            )
        if canonical_counter.get("algorithm") != CANONICAL_COUNTER_RNG_ALGORITHM:
            raise ValueError("Vulkan canonical-counter RNG algorithm is unsupported")
        normalized_stochastic_rng["canonical_counter"] = {
            "algorithm": CANONICAL_COUNTER_RNG_ALGORITHM,
            "seed": _u64_session_int(
                canonical_counter.get("seed"),
                "execution_policy.stochastic_rng.canonical_counter.seed",
            ),
            "next_word": _u64_session_int(
                canonical_counter.get("next_word", 0),
                "execution_policy.stochastic_rng.canonical_counter.next_word",
            ),
        }
    elif canonical_counter is not None:
        raise ValueError(
            "Vulkan non-canonical stochastic RNG cannot carry canonical_counter state"
        )
    if rng_mode == "none" and state_required:
        raise ValueError("Vulkan stochastic RNG mode=none cannot require RNG state")
    loss_scaling = raw.get("loss_scaling", {"mode": "none", "pending_gradients_scaled": False})
    if not isinstance(loss_scaling, dict):
        raise ValueError("Vulkan training session execution_policy loss_scaling must be an object")
    scaling_mode = loss_scaling.get("mode")
    if scaling_mode not in {"none", "dynamic", "static"}:
        raise ValueError("Vulkan training session execution_policy loss_scaling mode is unsupported")
    pending_scaled = loss_scaling.get("pending_gradients_scaled", False)
    if not isinstance(pending_scaled, bool):
        raise ValueError("Vulkan training session pending_gradients_scaled must be boolean")
    normalized_scaling: dict[str, object] = {
        "mode": scaling_mode,
        "pending_gradients_scaled": pending_scaled,
    }
    if scaling_mode in {"dynamic", "static"}:
        normalized_scaling["scale"] = _finite_session_float(
            loss_scaling.get("scale"), "execution_policy.loss_scaling.scale", positive=True
        )
    if scaling_mode == "dynamic":
        normalized_scaling.update({
            "growth_factor": _finite_session_float(
                loss_scaling.get("growth_factor", 2.0),
                "execution_policy.loss_scaling.growth_factor",
                positive=True,
            ),
            "backoff_factor": _finite_session_float(
                loss_scaling.get("backoff_factor", 0.5),
                "execution_policy.loss_scaling.backoff_factor",
                positive=True,
            ),
            "growth_interval": _nonnegative_session_int(
                loss_scaling.get("growth_interval", 2000),
                "execution_policy.loss_scaling.growth_interval",
                minimum=1,
            ),
            "growth_tracker": _nonnegative_session_int(
                loss_scaling.get("growth_tracker", 0),
                "execution_policy.loss_scaling.growth_tracker",
            ),
        })
    return {
        "format": PORTABLE_EXECUTION_POLICY_FORMAT,
        "source_backend": str(raw.get("source_backend", "unknown")),
        "compute_dtype": compute_dtype,
        "autocast_enabled": autocast_enabled,
        "stochastic_rng": normalized_stochastic_rng,
        "loss_scaling": normalized_scaling,
    }


def _scaler_state_from_execution_policy(policy: dict[str, object] | None):
    if not policy:
        return None
    scaling = policy.get("loss_scaling")
    if not isinstance(scaling, dict) or scaling.get("mode") != "dynamic":
        return None
    return {
        "scale": float(scaling["scale"]),
        "growth_factor": float(scaling["growth_factor"]),
        "backoff_factor": float(scaling["backoff_factor"]),
        "growth_interval": int(scaling["growth_interval"]),
        "_growth_tracker": int(scaling["growth_tracker"]),
    }


def _capture_vulkan_training_session(
    checkpoint_state: dict[str, object],
    completed_epoch: int,
    mid_epoch_step: int,
    *,
    template_session: dict[str, object] | None = None,
) -> dict[str, object]:
    template_session = template_session or {}
    optimizer_grouping_version = _nonnegative_session_int(
        checkpoint_state.get(
            "optimizer_grouping_version",
            template_session.get("optimizer_grouping_version", 2),
        ),
        "optimizer_grouping_version",
        minimum=1,
    )
    if "lr_scheduler_state" in checkpoint_state:
        main_lr = _normalize_training_session_lr_state(
            checkpoint_state.get("lr_scheduler_state"),
            "main_lr_scheduler",
            scheduler_state_dict=checkpoint_state.get("scheduler_state_dict"),
            require_group_lrs=True,
        )
    else:
        main_lr = _normalize_training_session_lr_state(
            template_session.get("main_lr_scheduler"),
            "main_lr_scheduler",
            require_group_lrs=True,
        )
    ltm_source = (
        checkpoint_state.get("ltm_scheduler_state")
        if "ltm_scheduler_state" in checkpoint_state
        else template_session.get("ltm_lr_scheduler")
    )
    ltm_lr = _normalize_training_session_lr_state(ltm_source, "ltm_lr_scheduler")
    error_budget = checkpoint_state.get("error_budget_state")
    if error_budget is None:
        skipped_train_batches = _nonnegative_session_int(
            template_session.get("skipped_train_batches", 0),
            "skipped_train_batches",
        )
    elif isinstance(error_budget, dict):
        skipped_train_batches = _nonnegative_session_int(
            error_budget.get("skipped_train_batches", 0),
            "error_budget.skipped_train_batches",
        )
    else:
        raise ValueError("Vulkan training session error_budget_state must be an object")
    data_stream_cursor = _normalize_data_stream_cursor(
        checkpoint_state.get(
            "data_stream_cursor",
            template_session.get("data_stream_cursor"),
        )
    )
    execution_policy = _normalize_execution_policy(
        checkpoint_state.get(
            "execution_policy",
            template_session.get("execution_policy"),
        )
    )
    if execution_policy is not None:
        # The Vulkan package boundary stores canonical unscaled pending
        # gradients even when the source PyTorch accumulation buffer lived in a
        # GradScaler domain. Preserve the scaler history, but describe the
        # serialized gradient file rather than the source process's scratch
        # representation.
        execution_policy = dict(execution_policy)
        loss_scaling = dict(execution_policy["loss_scaling"])
        loss_scaling["pending_gradients_scaled"] = False
        execution_policy["loss_scaling"] = loss_scaling
    if data_stream_cursor is not None and data_stream_cursor["batch_cursor"] != mid_epoch_step:
        raise ValueError(
            "Vulkan training session data_stream_cursor batch_cursor disagrees with mid_epoch_step"
        )
    return {
        "format": VULKAN_TRAINING_SESSION_FORMAT,
        "completed_epoch": completed_epoch,
        "mid_epoch_step": mid_epoch_step,
        "optimizer_grouping_version": optimizer_grouping_version,
        "main_lr_scheduler": main_lr,
        "ltm_lr_scheduler": ltm_lr,
        "effective_training_config": _json_native_session_mapping(
            checkpoint_state.get(
                "effective_training_config",
                template_session.get("effective_training_config"),
            ),
            "effective_training_config",
        ),
        "skipped_train_batches": skipped_train_batches,
        "data_stream_cursor": data_stream_cursor,
        "execution_policy": execution_policy,
    }


def _parse_vulkan_training_session(
    manifest: dict[str, object],
) -> dict[str, object] | None:
    raw = manifest.get("training_session")
    if raw is None:
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("format") not in _SUPPORTED_VULKAN_TRAINING_SESSION_FORMATS
    ):
        raise ValueError("Vulkan training manifest has an unsupported training_session")
    completed_epoch = _nonnegative_session_int(raw.get("completed_epoch", 0), "completed_epoch")
    mid_epoch_step = _nonnegative_session_int(raw.get("mid_epoch_step", 0), "mid_epoch_step")
    for manifest_field, session_value in (
        ("completed_epoch", completed_epoch),
        ("mid_epoch_step", mid_epoch_step),
    ):
        manifest_value = manifest.get(manifest_field)
        if manifest_value is not None and _nonnegative_session_int(
            manifest_value, manifest_field
        ) != session_value:
            raise ValueError(
                f"Vulkan training manifest {manifest_field} disagrees with training_session"
            )
    data_stream_cursor = _normalize_data_stream_cursor(raw.get("data_stream_cursor"))
    execution_policy = _normalize_execution_policy(raw.get("execution_policy"))
    if data_stream_cursor is not None and data_stream_cursor["batch_cursor"] != mid_epoch_step:
        raise ValueError(
            "Vulkan training session data_stream_cursor batch_cursor disagrees with mid_epoch_step"
        )
    return {
        "format": VULKAN_TRAINING_SESSION_FORMAT,
        "completed_epoch": completed_epoch,
        "mid_epoch_step": mid_epoch_step,
        "optimizer_grouping_version": _nonnegative_session_int(
            raw.get("optimizer_grouping_version", 2),
            "optimizer_grouping_version",
            minimum=1,
        ),
        "main_lr_scheduler": _normalize_training_session_lr_state(
            raw.get("main_lr_scheduler"),
            "main_lr_scheduler",
            require_group_lrs=True,
        ),
        "ltm_lr_scheduler": _normalize_training_session_lr_state(
            raw.get("ltm_lr_scheduler"), "ltm_lr_scheduler"
        ),
        "effective_training_config": _json_native_session_mapping(
            raw.get("effective_training_config"), "effective_training_config"
        ),
        "skipped_train_batches": _nonnegative_session_int(
            raw.get("skipped_train_batches", 0), "skipped_train_batches"
        ),
        "data_stream_cursor": data_stream_cursor,
        "execution_policy": execution_policy,
    }


def _scheduler_state_dict_from_native_session(
    main_lr_scheduler: dict[str, object] | None,
) -> dict[str, object] | None:
    if not main_lr_scheduler or not main_lr_scheduler.get("enabled", True):
        return None
    base_lrs = main_lr_scheduler.get("base_lrs")
    last_lrs = main_lr_scheduler.get("last_lrs")
    if not isinstance(base_lrs, list) or not isinstance(last_lrs, list):
        return None
    step = int(main_lr_scheduler["step"])
    return {
        "base_lrs": list(base_lrs),
        "last_epoch": step,
        "verbose": False,
        "_step_count": int(main_lr_scheduler.get("step_count", step + 1)),
        "_get_lr_called_within_step": False,
        "_last_lr": list(last_lrs),
        "lr_lambdas": [None] * len(base_lrs),
    }


def _merge_training_session_into_checkpoint(
    state: dict[str, object],
    session: dict[str, object] | None,
) -> dict[str, object]:
    if session is None:
        return state
    promoted = {
        "completed_epoch": session["completed_epoch"],
        "mid_epoch_step": session["mid_epoch_step"],
        "optimizer_grouping_version": session["optimizer_grouping_version"],
        "effective_training_config": session["effective_training_config"],
        "lr_scheduler_state": session["main_lr_scheduler"],
        "ltm_scheduler_state": session["ltm_lr_scheduler"],
        "error_budget_state": {
            "skipped_train_batches": session["skipped_train_batches"]
        },
    }
    if session.get("data_stream_cursor") is not None:
        promoted["data_stream_cursor"] = session["data_stream_cursor"]
    if session.get("execution_policy") is not None:
        promoted["execution_policy"] = session["execution_policy"]
    scheduler_state = _scheduler_state_dict_from_native_session(
        session.get("main_lr_scheduler")
    )
    if scheduler_state is not None:
        promoted["scheduler_state_dict"] = scheduler_state
    scaler_state = _scaler_state_from_execution_policy(session.get("execution_policy"))
    if scaler_state is not None:
        promoted["scaler_state_dict"] = scaler_state
    for key, value in promoted.items():
        if key in state and state[key] != value:
            raise ValueError(
                f"Vulkan portable replay {key} disagrees with native training_session"
            )
        state[key] = value
    state["_vulkan_native_training_session"] = session
    return state


def _portable_replay_encode(value, tensors: dict[str, torch.Tensor]):
    if torch.is_tensor(value):
        name = f"state_{len(tensors):06d}"
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("portable replay refuses non-finite tensor state")
        tensors[name] = tensor
        return {"__kind__": "tensor", "name": name}
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "numpy.ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.reshape(-1).tolist(),
        }
    if isinstance(value, np.generic):
        return _portable_replay_encode(value.item(), tensors)
    if isinstance(value, ROSAState):
        transitions = [
            [
                int(state),
                [[int(symbol), int(target)] for symbol, target in sorted(edges.items())],
            ]
            for state, edges in sorted(value.transitions.items())
        ]
        return {
            "__kind__": "rosa_state",
            "transitions": transitions,
            "suffix_links": [int(item) for item in value.suffix_links],
            "lengths": [int(item) for item in value.lengths],
            "endpos": [int(item) for item in value.endpos],
            "last_state": int(value.last_state),
            "num_states": int(value.num_states),
            "tokens": [int(item) for item in value.tokens],
        }
    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [_portable_replay_encode(item, tensors) for item in value],
        }
    if isinstance(value, list):
        return {
            "__kind__": "list",
            "items": [_portable_replay_encode(item, tensors) for item in value],
        }
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise ValueError(
                    "portable replay dictionaries support only string/integer keys; "
                    f"got {type(key).__name__}"
                )
            items.append(
                [
                    _portable_replay_encode(key, tensors),
                    _portable_replay_encode(item, tensors),
                ]
            )
        return {"__kind__": "dict", "items": items}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("portable replay refuses non-finite scalar state")
        return value
    raise ValueError(
        f"portable replay does not support state object {type(value).__name__}"
    )


def _portable_replay_decode(value, tensors: dict[str, torch.Tensor]):
    if not isinstance(value, dict) or "__kind__" not in value:
        return value
    kind = value.get("__kind__")
    if kind == "tensor":
        name = value.get("name")
        if not isinstance(name, str) or name not in tensors:
            raise ValueError(f"portable replay references missing tensor {name!r}")
        return tensors[name].clone()
    if kind == "numpy.ndarray":
        try:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            array = np.asarray(value["data"], dtype=dtype)
            return array.reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed NumPy array in portable replay") from exc
    if kind == "rosa_state":
        try:
            transitions = {
                int(state): {int(symbol): int(target) for symbol, target in edges}
                for state, edges in value["transitions"]
            }
            return ROSAState(
                transitions=transitions,
                suffix_links=[int(item) for item in value["suffix_links"]],
                lengths=[int(item) for item in value["lengths"]],
                endpos=[int(item) for item in value["endpos"]],
                last_state=int(value["last_state"]),
                num_states=int(value["num_states"]),
                tokens=[int(item) for item in value["tokens"]],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed ROSA state in portable replay") from exc
    if kind in {"tuple", "list"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError(f"portable replay {kind} is missing its item list")
        decoded = [_portable_replay_decode(item, tensors) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("portable replay dict is missing its item list")
        decoded = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("portable replay dict contains a malformed entry")
            key = _portable_replay_decode(pair[0], tensors)
            if not isinstance(key, (str, int)):
                raise ValueError("portable replay decoded an unsupported dictionary key")
            decoded[key] = _portable_replay_decode(pair[1], tensors)
        return decoded
    raise ValueError(f"unsupported portable replay node kind {kind!r}")


def write_vulkan_training_replay(
    package_dir: str | Path,
    checkpoint_state: dict[str, object],
) -> dict[str, object]:
    """Attach pickle-free DataLoader/RNG/recurrent replay state to a Vulkan package.

    The model, AdamW moments, and optional pending gradients remain in their
    native SafeTensors files. This companion stores only host replay state so a
    subsequent ``--resume-from-ckpt PACKAGE`` can reconstruct the same data
    cursor and recurrent carrier on PyTorch CPU/CUDA.
    """

    if not isinstance(checkpoint_state, dict):
        raise ValueError("portable Vulkan replay state must be a checkpoint mapping")
    package_dir = Path(package_dir)
    manifest = read_vulkan_training_manifest(package_dir)
    try:
        completed_epoch = int(checkpoint_state.get("completed_epoch", 0) or 0)
        mid_epoch_step = int(checkpoint_state.get("mid_epoch_step", 0) or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("portable Vulkan replay cursor must use integer epoch/step values") from exc
    if completed_epoch < 0 or mid_epoch_step < 0:
        raise ValueError("portable Vulkan replay epoch/step must be non-negative")
    data_stream_cursor = _normalize_data_stream_cursor(
        checkpoint_state.get("data_stream_cursor")
    )
    execution_policy = _normalize_execution_policy(
        checkpoint_state.get("execution_policy")
    )
    if data_stream_cursor is not None and data_stream_cursor["batch_cursor"] != mid_epoch_step:
        raise ValueError(
            "portable Vulkan data_stream_cursor batch_cursor disagrees with mid_epoch_step"
        )
    if mid_epoch_step > 0:
        missing = []
        if not checkpoint_state.get("run_identity"):
            missing.append("run_identity")
        if data_stream_cursor is None and not checkpoint_state.get("data_state"):
            missing.append("data_state/data_stream_cursor")
        stochastic_rng = (
            execution_policy.get("stochastic_rng")
            if isinstance(execution_policy, dict)
            else None
        )
        backend_rng_required = not (
            isinstance(stochastic_rng, dict)
            and stochastic_rng.get("mode") in {"none", "canonical-counter"}
            and not bool(stochastic_rng.get("state_required", False))
        )
        if backend_rng_required and not checkpoint_state.get("rng_state"):
            missing.append("rng_state")
        if missing:
            raise ValueError(
                "exact mid-epoch Vulkan replay is missing required host state: "
                + ", ".join(missing)
            )
        effective = checkpoint_state.get("effective_training_config")
        persist_state = bool(
            isinstance(effective, dict) and effective.get("persist_state", False)
        )
        if persist_state and checkpoint_state.get("running_states") is None:
            raise ValueError(
                "exact persisted-state Vulkan replay requires running_states"
            )

    training_session = _capture_vulkan_training_session(
        checkpoint_state,
        completed_epoch,
        mid_epoch_step,
        template_session=_parse_vulkan_training_session(manifest),
    )

    replay_payload = {
        key: checkpoint_state[key]
        for key in _PORTABLE_REPLAY_CHECKPOINT_KEYS
        if key in checkpoint_state
    }
    # v2 sessions own the deterministic sampler cursor and numerical execution
    # policy. Keep backend-shaped Python/PyTorch blobs only for legacy or
    # explicitly backend-native stochastic state.
    if data_stream_cursor is None and "data_state" in checkpoint_state:
        replay_payload["data_state"] = checkpoint_state["data_state"]
    stochastic_rng = (
        execution_policy.get("stochastic_rng")
        if isinstance(execution_policy, dict)
        else None
    )
    backend_rng_required = not (
        isinstance(stochastic_rng, dict)
        and stochastic_rng.get("mode") in {"none", "canonical-counter"}
        and not bool(stochastic_rng.get("state_required", False))
    )
    if backend_rng_required and "rng_state" in checkpoint_state:
        replay_payload["rng_state"] = checkpoint_state["rng_state"]
    if execution_policy is None and "scaler_state_dict" in checkpoint_state:
        replay_payload["scaler_state_dict"] = checkpoint_state["scaler_state_dict"]
    tensors: dict[str, torch.Tensor] = {}
    encoded = _portable_replay_encode(replay_payload, tensors)
    replay_document = {
        "format": VULKAN_PORTABLE_REPLAY_FORMAT,
        "state": encoded,
    }

    replay_path = package_dir / VULKAN_PORTABLE_REPLAY_FILENAME
    tensor_path = package_dir / VULKAN_PORTABLE_REPLAY_TENSOR_FILENAME
    replay_temp = replay_path.with_name(replay_path.name + ".tmp")
    replay_temp.write_text(
        json.dumps(replay_document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    replay_temp.replace(replay_path)
    if tensors:
        tensor_temp = tensor_path.with_name(tensor_path.name + ".tmp")
        save_file(
            tensors,
            str(tensor_temp),
            metadata={"format": VULKAN_PORTABLE_REPLAY_FORMAT},
        )
        tensor_temp.replace(tensor_path)
        tensor_member: str | None = VULKAN_PORTABLE_REPLAY_TENSOR_FILENAME
    else:
        tensor_path.unlink(missing_ok=True)
        tensor_member = None

    # v4 additionally guarantees canonical objective-weighted val_proj pending
    # gradients. Attaching replay metadata must not silently relabel an older
    # open package whose gradient file still uses the pre-v4 Vulkan convention.
    if (
        not bool(manifest.get("accumulation_open", False))
        or not bool(manifest.get("val_proj_active_in_window", False))
        or bool(manifest.get("val_proj_gradient_weight_applied", False))
    ):
        manifest["parameter_state"] = _canonical_parameter_state(
            manifest.get("model_file")
        )
        manifest["format"] = VULKAN_TRAINING_FORMAT
    manifest["completed_epoch"] = completed_epoch
    manifest["mid_epoch_step"] = mid_epoch_step
    manifest["training_session"] = training_session
    manifest["portable_replay_file"] = VULKAN_PORTABLE_REPLAY_FILENAME
    manifest["portable_replay_tensor_file"] = tensor_member
    manifest_path = package_dir / "training_state.json"
    manifest_temp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_temp.replace(manifest_path)
    return manifest


def read_vulkan_training_replay(
    package_dir: str | Path,
    manifest: dict[str, object] | None = None,
) -> dict[str, object] | None:
    package_dir = Path(package_dir)
    manifest = manifest or read_vulkan_training_manifest(package_dir)
    session = _parse_vulkan_training_session(manifest)
    replay_name = manifest.get("portable_replay_file")
    if replay_name is None:
        # A native session is necessary trajectory metadata, but it is not a
        # substitute for host RNG/DataLoader/recurrent replay. Exact PyTorch
        # --resume-from-ckpt must still fail closed when that sidecar is absent.
        return None
    replay_path = _package_local_member(
        package_dir, replay_name, "portable_replay_file"
    )
    try:
        replay_document = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read Vulkan portable replay {replay_path}: {exc}") from exc
    if not isinstance(replay_document, dict) or replay_document.get("format") != VULKAN_PORTABLE_REPLAY_FORMAT:
        raise ValueError(
            f"unsupported Vulkan portable replay format in {replay_path}"
        )

    tensor_name = manifest.get("portable_replay_tensor_file")
    tensors: dict[str, torch.Tensor] = {}
    if tensor_name is not None:
        tensor_path = _package_local_member(
            package_dir, tensor_name, "portable_replay_tensor_file"
        )
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != VULKAN_PORTABLE_REPLAY_FORMAT:
                raise ValueError(
                    f"unsupported portable replay tensor format in {tensor_path}"
                )
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    state = _portable_replay_decode(replay_document.get("state"), tensors)
    if not isinstance(state, dict):
        raise ValueError("Vulkan portable replay root must decode to a mapping")
    if session is not None:
        return _merge_training_session_into_checkpoint(state, session)
    try:
        state["completed_epoch"] = int(manifest.get("completed_epoch", 0) or 0)
        state["mid_epoch_step"] = int(manifest.get("mid_epoch_step", 0) or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Vulkan portable replay manifest has malformed epoch/step") from exc
    return state


def read_vulkan_adamw_checkpoint(path: str | Path) -> VulkanAdamWCheckpoint:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        checkpoint_format = metadata.get("format")
        if checkpoint_format not in _SUPPORTED_VULKAN_ADAMW_FORMATS:
            raise ValueError(
                f"{path} is not a supported Hierarchos Vulkan AdamW checkpoint: "
                f"format={checkpoint_format!r}"
            )
        if metadata.get("layout") != "pytorch-row-major":
            raise ValueError(
                f"unsupported Vulkan optimizer layout {metadata.get('layout')!r}"
            )
        try:
            step = int(metadata["step"])
            slot_names_raw = json.loads(metadata["slot_names"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Vulkan AdamW metadata in {path}: {exc}") from exc
        if step < 0:
            raise ValueError(f"Vulkan AdamW step must be non-negative, got {step}")
        if not isinstance(slot_names_raw, list) or not all(
            isinstance(name, str) and name for name in slot_names_raw
        ):
            raise ValueError("Vulkan AdamW slot_names must be a JSON list of names")
        if len(slot_names_raw) != len(set(slot_names_raw)):
            raise ValueError("Vulkan AdamW slot_names contains duplicates")
        if checkpoint_format in {VULKAN_ADAMW_FORMAT_V2, VULKAN_ADAMW_FORMAT}:
            try:
                slot_steps_raw = json.loads(metadata["slot_steps"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Vulkan AdamW slot_steps in {path}: {exc}") from exc
            if (
                not isinstance(slot_steps_raw, list)
                or len(slot_steps_raw) != len(slot_names_raw)
                or not all(isinstance(value, int) for value in slot_steps_raw)
            ):
                raise ValueError(
                    "Vulkan AdamW v2+ slot_steps must be a JSON integer list matching slot_names"
                )
        else:
            slot_steps_raw = [step] * len(slot_names_raw)
        if any(value < 0 or value > step for value in slot_steps_raw):
            raise ValueError(
                f"Vulkan AdamW slot steps must lie in 0..={step}: {slot_steps_raw}"
            )
        slot_steps = dict(zip(slot_names_raw, slot_steps_raw))
        if checkpoint_format == VULKAN_ADAMW_FORMAT:
            try:
                slot_decay_classes_raw = json.loads(metadata["slot_decay_classes"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid Vulkan AdamW slot_decay_classes in {path}: {exc}"
                ) from exc
            if (
                not isinstance(slot_decay_classes_raw, list)
                or len(slot_decay_classes_raw) != len(slot_names_raw)
                or not all(
                    value in {VULKAN_ADAMW_DECAY_CLASS, VULKAN_ADAMW_NO_DECAY_CLASS}
                    for value in slot_decay_classes_raw
                )
            ):
                raise ValueError(
                    "Vulkan AdamW v3 slot_decay_classes must be a JSON list of "
                    "'decay'/'no-decay' labels matching slot_names"
                )
        else:
            slot_decay_classes_raw = [None] * len(slot_names_raw)
        slot_decay_classes = dict(zip(slot_names_raw, slot_decay_classes_raw))

        exp_avg: dict[str, torch.Tensor] = {}
        exp_avg_sq: dict[str, torch.Tensor] = {}
        for name in slot_names_raw:
            first_key = f"optimizer.{name}.exp_avg"
            second_key = f"optimizer.{name}.exp_avg_sq"
            try:
                first = handle.get_tensor(first_key).float().contiguous()
                second = handle.get_tensor(second_key).float().contiguous()
            except Exception as exc:  # safetensors reports missing tensors by key
                raise ValueError(f"missing optimizer tensors for slot {name!r}") from exc
            if first.ndim != 1 or second.ndim != 1 or first.numel() != second.numel():
                raise ValueError(
                    f"invalid flattened AdamW moments for {name!r}: "
                    f"exp_avg={tuple(first.shape)} exp_avg_sq={tuple(second.shape)}"
                )
            if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                raise ValueError(f"non-finite AdamW moments in Vulkan slot {name!r}")
            exp_avg[name] = first
            exp_avg_sq[name] = second

    return VulkanAdamWCheckpoint(
        step=step,
        slot_names=tuple(slot_names_raw),
        slot_steps=slot_steps,
        slot_decay_classes=slot_decay_classes,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
    )


def read_vulkan_pending_gradient_checkpoint(
    path: str | Path,
) -> VulkanPendingGradientCheckpoint:
    """Read the canonical in-flight Vulkan gradient registry.

    Tensor keys are exact model parameter names and each payload is flattened
    FP32 in PyTorch row-major order. Keeping the file free of backend-specific
    prefixes lets the same registry materialize as either Vulkan device buffers
    or ordinary ``torch.nn.Parameter.grad`` tensors.
    """

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != VULKAN_PENDING_GRADIENT_FORMAT:
            raise ValueError(
                f"{path} is not a Hierarchos Vulkan pending-gradient checkpoint: "
                f"format={metadata.get('format')!r}"
            )
        if metadata.get("layout") != "pytorch-row-major":
            raise ValueError(
                f"unsupported Vulkan pending-gradient layout {metadata.get('layout')!r}"
            )
        try:
            slot_names_raw = json.loads(metadata["slot_names"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid pending-gradient metadata in {path}: {exc}") from exc
        if not isinstance(slot_names_raw, list) or not slot_names_raw or not all(
            isinstance(name, str) and name for name in slot_names_raw
        ):
            raise ValueError("Vulkan pending-gradient slot_names must be a non-empty name list")
        if len(slot_names_raw) != len(set(slot_names_raw)):
            raise ValueError("Vulkan pending-gradient slot_names contains duplicates")
        actual_keys = set(handle.keys())
        if actual_keys != set(slot_names_raw):
            missing = sorted(set(slot_names_raw) - actual_keys)
            extra = sorted(actual_keys - set(slot_names_raw))
            raise ValueError(
                "Vulkan pending-gradient tensors do not match slot_names: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        gradients: dict[str, torch.Tensor] = {}
        for name in slot_names_raw:
            gradient = handle.get_tensor(name).float().reshape(-1).contiguous()
            if gradient.numel() == 0:
                raise ValueError(f"Vulkan pending gradient {name!r} is empty")
            if not torch.isfinite(gradient).all():
                raise ValueError(f"non-finite Vulkan pending gradient {name!r}")
            gradients[name] = gradient
    return VulkanPendingGradientCheckpoint(
        slot_names=tuple(slot_names_raw),
        gradients=gradients,
    )


def _clean_parameter_name(name: str) -> str:
    while name.startswith("_orig_mod."):
        name = name[len("_orig_mod.") :]
    return name


def _named_parameter_aliases(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    aliases: dict[str, torch.nn.Parameter] = {}
    try:
        named = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch only
        named = model.named_parameters()
    for raw_name, parameter in named:
        for name in (raw_name, _clean_parameter_name(raw_name)):
            existing = aliases.get(name)
            if existing is not None and existing is not parameter:
                raise ValueError(f"ambiguous model parameter alias {name!r}")
            aliases[name] = parameter
    return aliases


def _validate_torch_parameter_alias_contract(
    model: torch.nn.Module,
    parameter_state: dict[str, object] | None,
) -> None:
    if parameter_state is None:
        return
    aliases = parameter_state.get("parameter_aliases", [])
    if not aliases:
        return
    named = _named_parameter_aliases(model)
    for entry in aliases:
        if not isinstance(entry, dict):
            raise ValueError("portable parameter alias entry must be an object")
        canonical = entry.get("canonical")
        alias = entry.get("alias")
        if not isinstance(canonical, str) or not canonical or not isinstance(alias, str) or not alias:
            raise ValueError("portable parameter alias entry must contain canonical and alias names")
        canonical_parameter = named.get(canonical)
        alias_parameter = named.get(alias)
        if canonical_parameter is None or alias_parameter is None:
            raise ValueError(
                f"PyTorch model is missing portable tied-parameter alias {alias!r} -> {canonical!r}"
            )
        if canonical_parameter is not alias_parameter:
            raise ValueError(
                f"PyTorch parameters {alias!r} and {canonical!r} must share one master object"
            )


def _validate_torch_master_dtype_contract(
    model: torch.nn.Module,
    parameter_state: dict[str, object] | None,
    slot_names: Iterable[str],
) -> None:
    if parameter_state is None:
        return
    if parameter_state.get("trainable_master_dtype") != "float32":
        raise ValueError("portable parameter state does not declare FP32 trainable masters")
    named = _named_parameter_aliases(model)
    for name in slot_names:
        parameter = named.get(name)
        if parameter is None:
            raise ValueError(f"portable FP32 master slot {name!r} has no PyTorch parameter")
        if parameter.dtype != torch.float32:
            raise ValueError(
                f"portable FP32 master slot {name!r} resolved to PyTorch dtype "
                f"{parameter.dtype}; use FP32 model parameters with CUDA/CPU autocast or "
                "an explicit FP32-master optimizer representation"
            )


def _validate_portable_master_file(
    package_dir: Path,
    parameter_state: dict[str, object] | None,
    slot_names: Iterable[str],
) -> None:
    if parameter_state is None:
        return
    master_file = parameter_state.get("master_file")
    if not isinstance(master_file, str) or not master_file:
        raise ValueError("portable parameter state is missing master_file")
    master_path = package_dir / master_file
    try:
        with safe_open(master_path, framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            for name in slot_names:
                if name not in available:
                    raise ValueError(
                        f"portable FP32 master file {master_path} is missing optimizer slot {name!r}"
                    )
                dtype = tensors.get_slice(name).get_dtype()
                if dtype != "F32":
                    raise ValueError(
                        f"portable FP32 master slot {name!r} is stored as {dtype} in {master_path}; "
                        "lower-precision storage is reserved for derived execution mirrors"
                    )
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"unable to validate portable FP32 master file {master_path}: {exc}") from exc


def _optimizer_groups_by_parameter(
    optimizer: torch.optim.Optimizer,
) -> dict[int, dict[str, object]]:
    groups: dict[int, dict[str, object]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in groups:
                raise ValueError("a parameter appears in more than one optimizer group")
            groups[parameter_id] = group
    return groups


def _resolve_vulkan_slots(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    slot_names: Iterable[str],
) -> tuple[dict[str, torch.nn.Parameter], dict[int, dict[str, object]]]:
    aliases = _named_parameter_aliases(model)
    optimizer_groups = _optimizer_groups_by_parameter(optimizer)
    resolved: dict[str, torch.nn.Parameter] = {}
    seen_parameter_ids: dict[int, str] = {}
    for name in slot_names:
        parameter = aliases.get(name)
        if parameter is None:
            raise ValueError(f"Vulkan optimizer slot {name!r} has no PyTorch parameter alias")
        if id(parameter) not in optimizer_groups:
            raise ValueError(f"Vulkan optimizer slot {name!r} is absent from the PyTorch optimizer")
        previous = seen_parameter_ids.get(id(parameter))
        if previous is not None:
            raise ValueError(
                f"Vulkan slots {previous!r} and {name!r} resolve to the same PyTorch parameter"
            )
        seen_parameter_ids[id(parameter)] = name
        resolved[name] = parameter
    return resolved, optimizer_groups


def _step_tensor_device(parameter: torch.nn.Parameter, group: dict[str, object]) -> torch.device:
    if bool(group.get("capturable", False)) or bool(group.get("fused", False)):
        return parameter.device
    return torch.device("cpu")


def _torch_adamw_decay_class(group: dict[str, object]) -> str:
    try:
        weight_decay = float(group.get("weight_decay", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("PyTorch AdamW weight_decay must be numeric") from exc
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(
            f"PyTorch AdamW weight_decay must be finite and non-negative, got {weight_decay}"
        )
    return (
        VULKAN_ADAMW_NO_DECAY_CLASS
        if weight_decay == 0.0
        else VULKAN_ADAMW_DECAY_CLASS
    )


def _validate_torch_adamw_group_semantics(
    name: str,
    group: dict[str, object],
    expected_decay_class: str | None,
) -> str:
    if bool(group.get("amsgrad", False)):
        raise ValueError("Vulkan AdamW does not carry AMSGrad max_exp_avg_sq state")
    if bool(group.get("maximize", False)):
        raise ValueError("Vulkan AdamW does not support PyTorch maximize=True semantics")
    actual_decay_class = _torch_adamw_decay_class(group)
    if (
        expected_decay_class == VULKAN_ADAMW_NO_DECAY_CLASS
        and actual_decay_class != VULKAN_ADAMW_NO_DECAY_CLASS
    ):
        raise ValueError(
            f"Vulkan AdamW slot {name!r} is declared no-decay but the PyTorch "
            f"optimizer group has weight_decay={group.get('weight_decay')!r}"
        )
    return expected_decay_class or actual_decay_class


def restore_vulkan_pending_gradients_into_torch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: VulkanPendingGradientCheckpoint,
) -> None:
    """Materialize a canonical Vulkan pending-gradient registry as ``.grad``.

    Resolution intentionally goes through the same alias table as the AdamW
    bridge, so the canonical ``lm_head.weight`` entry lands on PyTorch's tied
    ``tok_emb.weight`` parameter object without duplicating the gradient.
    """

    resolved, _ = _resolve_vulkan_slots(model, optimizer, checkpoint.slot_names)
    for name in checkpoint.slot_names:
        parameter = resolved[name]
        gradient = checkpoint.gradients[name]
        if gradient.numel() != parameter.numel():
            raise ValueError(
                f"Vulkan pending gradient {name!r} has {gradient.numel()} elements but "
                f"PyTorch parameter shape {tuple(parameter.shape)} has {parameter.numel()}"
            )
        parameter.grad = gradient.reshape(parameter.shape).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )


def save_torch_pending_gradients_as_vulkan(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    template_checkpoint: str | Path,
) -> VulkanPendingGradientCheckpoint:
    """Write live PyTorch ``.grad`` tensors using Vulkan's canonical registry.

    The optimizer checkpoint is used only as the schema authority. This keeps
    tied parameters (notably ``lm_head.weight``/``tok_emb.weight``) represented
    exactly once and prevents a PyTorch-only parameter from silently entering a
    package that the native Vulkan graph cannot restore.
    """

    template = read_vulkan_adamw_checkpoint(template_checkpoint)
    resolved, _ = _resolve_vulkan_slots(model, optimizer, template.slot_names)
    tensor_map: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    for name in template.slot_names:
        parameter = resolved[name]
        if parameter.grad is None:
            raise ValueError(f"PyTorch pending gradient for {name!r} is missing")
        gradient = parameter.grad.detach().float().cpu().reshape(-1).contiguous()
        if gradient.numel() != parameter.numel():
            raise ValueError(f"PyTorch pending gradient shape mismatch for {name!r}")
        if not torch.isfinite(gradient).all():
            raise ValueError(f"refusing to export non-finite pending gradient for {name!r}")
        tensor_map[name] = gradient
        gradients[name] = gradient

    metadata = {
        "format": VULKAN_PENDING_GRADIENT_FORMAT,
        "slot_names": json.dumps(list(template.slot_names), separators=(",", ":")),
        "layout": "pytorch-row-major",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensor_map, str(path), metadata=metadata)
    return VulkanPendingGradientCheckpoint(
        slot_names=template.slot_names,
        gradients=gradients,
    )


def _package_local_member(package_dir: Path, raw_name: object, field: str) -> Path:
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError(f"Vulkan training manifest {field} must be a filename")
    member = Path(raw_name)
    if member.name != raw_name or member.is_absolute():
        raise ValueError(f"Vulkan training manifest {field} must be package-local: {raw_name!r}")
    return package_dir / member


def load_vulkan_training_package_into_torch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    package_dir: str | Path,
    *,
    allowed_untracked_parameters: Iterable[str] = (),
) -> VulkanTrainingPackage:
    """Restore Vulkan optimizer + in-flight accumulation state into PyTorch.

    This is the cross-backend continuation boundary for a native training
    package. Model masters are still loaded from the package's ordinary
    ``model.safetensors`` by the normal Hierarchos loader; this function restores
    the training-only state that PyTorch needs on CPU or CUDA.

    Exact mid-window PyTorch continuation currently maps Vulkan's
    ``mean-by-supervision-weight`` contract to the trainer's ``weighted-token``
    contract. Other Vulkan normalization modes fail closed rather than silently
    changing the optimizer trajectory.
    """

    package_dir = Path(package_dir)
    manifest = read_vulkan_training_manifest(package_dir)
    checkpoint_format = manifest.get("format")
    parameter_state = _normalize_parameter_state(manifest)
    _validate_torch_parameter_alias_contract(model, parameter_state)
    session_state = _parse_vulkan_training_session(manifest)
    replay_state = read_vulkan_training_replay(package_dir, manifest)
    ltm_alignment_controller = _parse_ltm_alignment_controller(manifest)
    ltm_config_targets = (
        _validate_ltm_alignment_controller_against_torch(model, ltm_alignment_controller)
        if ltm_alignment_controller is not None
        else []
    )

    optimizer_path = _package_local_member(
        package_dir,
        manifest.get("optimizer_file"),
        "optimizer_file",
    )
    optimizer_checkpoint = read_vulkan_adamw_checkpoint(optimizer_path)
    if int(manifest.get("optimizer_step", -1)) != optimizer_checkpoint.step:
        raise ValueError(
            "Vulkan training manifest optimizer_step does not match optimizer.safetensors"
        )
    if int(manifest.get("optimizer_tensor_count", -1)) != len(optimizer_checkpoint.slot_names):
        raise ValueError(
            "Vulkan training manifest optimizer_tensor_count does not match optimizer.safetensors"
        )
    _validate_torch_master_dtype_contract(
        model, parameter_state, optimizer_checkpoint.slot_names
    )
    _validate_portable_master_file(
        package_dir, parameter_state, optimizer_checkpoint.slot_names
    )

    accumulation_open = manifest.get("accumulation_open", False)
    if not isinstance(accumulation_open, bool):
        raise ValueError("Vulkan training manifest accumulation_open must be boolean")
    val_proj_gradient_weight_applied = manifest.get(
        "val_proj_gradient_weight_applied", False
    )
    if not isinstance(val_proj_gradient_weight_applied, bool):
        raise ValueError(
            "Vulkan training manifest val_proj_gradient_weight_applied must be boolean"
        )
    pending_checkpoint: VulkanPendingGradientCheckpoint | None = None
    pytorch_normalization: str | None = None
    consumed_weighted_token_mass = 0.0
    target_weighted_token_mass: float | None = None
    if accumulation_open:
        if checkpoint_format == VULKAN_TRAINING_FORMAT_V1:
            raise ValueError("Vulkan v1 training packages cannot resume an open accumulation window")
        gradient_path = _package_local_member(
            package_dir,
            manifest.get("gradient_file"),
            "gradient_file",
        )
        pending_checkpoint = read_vulkan_pending_gradient_checkpoint(gradient_path)
        if int(manifest.get("gradient_tensor_count", -1)) != len(pending_checkpoint.slot_names):
            raise ValueError(
                "Vulkan training manifest gradient_tensor_count does not match gradients.safetensors"
            )
        if pending_checkpoint.slot_names != optimizer_checkpoint.slot_names:
            raise ValueError(
                "Vulkan pending-gradient registry does not match the canonical AdamW slot registry"
            )
        val_proj_active = manifest.get("val_proj_active_in_window", False)
        if not isinstance(val_proj_active, bool):
            raise ValueError("Vulkan training manifest val_proj_active_in_window must be boolean")
        pytorch_tbptt_ltm_weighting = manifest.get(
            "ltm_alignment_pytorch_tbptt_weighting_in_window"
        )
        if pytorch_tbptt_ltm_weighting not in {None, True, False}:
            raise ValueError(
                "Vulkan training manifest LTM TBPTT weighting marker must be boolean or null"
            )
        if val_proj_gradient_weight_applied and (
            not val_proj_active or pytorch_tbptt_ltm_weighting is not True
        ):
            raise ValueError(
                "weighted val_proj pending gradient requires an active PyTorch-TBPTT LTM window"
            )
        if val_proj_active and pytorch_tbptt_ltm_weighting is True:
            if ltm_alignment_controller is None:
                raise ValueError(
                    "active LTM val_proj window is missing its portable controller state"
                )
            if checkpoint_format in {
                VULKAN_TRAINING_FORMAT_V4,
                VULKAN_TRAINING_FORMAT_V5,
                VULKAN_TRAINING_FORMAT,
            } and not val_proj_gradient_weight_applied:
                raise ValueError(
                    "Vulkan v4+ active LTM window is missing canonical val_proj objective weighting"
                )
            if not val_proj_gradient_weight_applied:
                # v1-v3 stored Vulkan's internal pre-objective-weight val_proj
                # accumulator. Convert that legacy representation before
                # exposing the pending registry to PyTorch.
                weight = float(ltm_alignment_controller["weight"])
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(
                        "legacy active LTM val_proj gradient requires a finite positive alignment weight"
                    )
                gradient = pending_checkpoint.gradients.get("val_proj.weight")
                if gradient is None:
                    raise ValueError(
                        "active LTM pending-gradient registry is missing val_proj.weight"
                    )
                canonical_gradients = dict(pending_checkpoint.gradients)
                canonical_gradients["val_proj.weight"] = gradient * weight
                pending_checkpoint = VulkanPendingGradientCheckpoint(
                    slot_names=pending_checkpoint.slot_names,
                    gradients=canonical_gradients,
                )
        normalization = manifest.get("accumulation_normalization")
        if normalization != "mean-by-supervision-weight":
            raise ValueError(
                "exact PyTorch mid-window resume currently requires Vulkan "
                f"mean-by-supervision-weight normalization, got {normalization!r}"
            )
        pytorch_normalization = "weighted-token"
        try:
            consumed_weighted_token_mass = float(
                manifest["accumulation_consumed_supervision_mass"]
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Vulkan weighted-token checkpoint is missing consumed supervision mass"
            ) from exc
        if not torch.isfinite(torch.tensor(consumed_weighted_token_mass)).item() or consumed_weighted_token_mass <= 0.0:
            raise ValueError(
                "Vulkan consumed supervision mass must be finite and positive for PyTorch resume"
            )
        raw_target_mass = manifest.get("accumulation_target_supervision_mass")
        if raw_target_mass is not None:
            try:
                target_weighted_token_mass = float(raw_target_mass)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Vulkan target supervision mass is malformed") from exc
            if (
                not torch.isfinite(torch.tensor(target_weighted_token_mass)).item()
                or target_weighted_token_mass <= 0.0
                or target_weighted_token_mass < consumed_weighted_token_mass
            ):
                raise ValueError(
                    "Vulkan target supervision mass must be finite, positive, and no smaller than consumed mass"
                )
    elif manifest.get("gradient_file") is not None or int(manifest.get("gradient_tensor_count", 0)) != 0:
        raise ValueError("closed Vulkan training package unexpectedly declares pending gradients")
    elif val_proj_gradient_weight_applied:
        raise ValueError("closed Vulkan training package cannot declare a weighted pending val_proj gradient")

    # All portable companions are validated before mutating the live optimizer
    # or parameter gradients.
    loaded_optimizer = load_vulkan_adamw_into_torch(
        model,
        optimizer,
        optimizer_path,
        allowed_untracked_parameters=allowed_untracked_parameters,
    )
    if pending_checkpoint is not None:
        restore_vulkan_pending_gradients_into_torch(model, optimizer, pending_checkpoint)
    if ltm_alignment_controller is not None:
        _apply_ltm_alignment_controller_to_torch(
            ltm_config_targets, ltm_alignment_controller
        )

    return VulkanTrainingPackage(
        manifest=manifest,
        parameter_state=parameter_state,
        optimizer=loaded_optimizer,
        pending_gradients=pending_checkpoint,
        pytorch_accumulation_normalization=pytorch_normalization,
        consumed_weighted_token_mass=consumed_weighted_token_mass,
        target_weighted_token_mass=target_weighted_token_mass,
        replay_state=replay_state,
        session_state=session_state,
        ltm_alignment_controller=ltm_alignment_controller,
    )


def load_vulkan_adamw_into_torch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    allowed_untracked_parameters: Iterable[str] = (),
) -> VulkanAdamWCheckpoint:
    """Load Vulkan AdamW moments into an existing PyTorch optimizer.

    ``allowed_untracked_parameters`` is intentionally explicit for future or
    experimental objectives whose learned tensors are not yet owned by a Vulkan
    graph. Any other extra optimizer parameter fails closed so cross-backend
    continuation cannot silently reset its moments.
    """

    checkpoint = read_vulkan_adamw_checkpoint(path)
    resolved, optimizer_groups = _resolve_vulkan_slots(
        model, optimizer, checkpoint.slot_names
    )
    aliases = _named_parameter_aliases(model)
    allowed_ids: set[int] = set()
    for name in allowed_untracked_parameters:
        parameter = aliases.get(name)
        if parameter is None:
            raise ValueError(f"allowed untracked parameter {name!r} does not exist")
        allowed_ids.add(id(parameter))
    tracked_ids = {id(parameter) for parameter in resolved.values()}
    unexpected = [
        parameter
        for parameter_id in optimizer_groups
        if parameter_id not in tracked_ids and parameter_id not in allowed_ids
        for parameter in [next(
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if id(p) == parameter_id
        )]
    ]
    if unexpected:
        names_by_id: dict[int, list[str]] = {}
        for name, parameter in aliases.items():
            names_by_id.setdefault(id(parameter), []).append(name)
        labels = [sorted(names_by_id.get(id(parameter), ["<unnamed>"])) for parameter in unexpected]
        raise ValueError(
            "PyTorch optimizer contains parameters absent from the Vulkan checkpoint; "
            f"pass only intentional gaps via allowed_untracked_parameters: {labels}"
        )

    for name, parameter in resolved.items():
        group = optimizer_groups[id(parameter)]
        _validate_torch_adamw_group_semantics(
            name, group, checkpoint.slot_decay_classes[name]
        )
        first = checkpoint.exp_avg[name]
        second = checkpoint.exp_avg_sq[name]
        if first.numel() != parameter.numel():
            raise ValueError(
                f"Vulkan slot {name!r} has {first.numel()} elements but PyTorch parameter "
                f"shape {tuple(parameter.shape)} has {parameter.numel()}"
            )
        optimizer.state[parameter] = {
            "step": torch.tensor(
                float(checkpoint.slot_steps[name]),
                dtype=torch.float32,
                device=_step_tensor_device(parameter, group),
            ),
            "exp_avg": first.reshape(parameter.shape).to(
                device=parameter.device, dtype=parameter.dtype
            ),
            "exp_avg_sq": second.reshape(parameter.shape).to(
                device=parameter.device, dtype=parameter.dtype
            ),
        }
    return checkpoint


def save_torch_adamw_as_vulkan(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    template_checkpoint: str | Path,
) -> VulkanAdamWCheckpoint:
    """Write current PyTorch AdamW state using an existing Vulkan slot schema.

    Requiring a template prevents PyTorch-only parameters from being smuggled
    into a checkpoint that the native Vulkan registry cannot consume.  It also
    preserves the canonical ``lm_head.weight`` alias for the tied embedding.
    """

    template = read_vulkan_adamw_checkpoint(template_checkpoint)
    resolved, optimizer_groups = _resolve_vulkan_slots(model, optimizer, template.slot_names)
    tensor_map: dict[str, torch.Tensor] = {}
    slot_steps: dict[str, int] = {}
    step_deltas: list[int] = []
    exp_avg: dict[str, torch.Tensor] = {}
    exp_avg_sq: dict[str, torch.Tensor] = {}
    slot_decay_classes: dict[str, str | None] = {}
    for name in template.slot_names:
        parameter = resolved[name]
        group = optimizer_groups[id(parameter)]
        slot_decay_classes[name] = _validate_torch_adamw_group_semantics(
            name, group, template.slot_decay_classes[name]
        )
        state = optimizer.state.get(parameter)
        if not state or "step" not in state or "exp_avg" not in state or "exp_avg_sq" not in state:
            raise ValueError(f"PyTorch optimizer state for {name!r} is not initialized")
        raw_step = state["step"]
        current_step = int(raw_step.item() if torch.is_tensor(raw_step) else raw_step)
        template_step = template.slot_steps[name]
        if current_step < template_step:
            raise ValueError(
                f"PyTorch AdamW step for {name!r} moved backwards: "
                f"template={template_step} current={current_step}"
            )
        slot_steps[name] = current_step
        step_deltas.append(current_step - template_step)
        first = state["exp_avg"].detach().float().cpu().reshape(-1).contiguous()
        second = state["exp_avg_sq"].detach().float().cpu().reshape(-1).contiguous()
        if first.numel() != parameter.numel() or second.numel() != parameter.numel():
            raise ValueError(f"PyTorch AdamW moment shape mismatch for {name!r}")
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError(f"refusing to export non-finite AdamW moments for {name!r}")
        tensor_map[f"optimizer.{name}.exp_avg"] = first
        tensor_map[f"optimizer.{name}.exp_avg_sq"] = second
        exp_avg[name] = first
        exp_avg_sq[name] = second
    step = template.step + max(step_deltas, default=0)
    if any(slot_step > step for slot_step in slot_steps.values()):
        raise ValueError(
            f"PyTorch AdamW slot step exceeds inferred outer optimizer step {step}: {slot_steps}"
        )

    metadata = {
        "format": VULKAN_ADAMW_FORMAT,
        "step": str(step),
        "slot_names": json.dumps(list(template.slot_names), separators=(",", ":")),
        "slot_steps": json.dumps(
            [slot_steps[name] for name in template.slot_names], separators=(",", ":")
        ),
        "slot_decay_classes": json.dumps(
            [slot_decay_classes[name] for name in template.slot_names],
            separators=(",", ":"),
        ),
        "layout": "pytorch-row-major",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensor_map, str(path), metadata=metadata)
    return VulkanAdamWCheckpoint(
        step=step,
        slot_names=template.slot_names,
        slot_steps=slot_steps,
        slot_decay_classes=slot_decay_classes,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
    )
