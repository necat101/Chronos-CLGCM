import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import time
import random
from tqdm import tqdm
import sys
import traceback
import numpy as np
import math

from .optimizers import DirectMLAdamW
from ..utils.device import is_directml_device, set_threads
from ..utils.checkpoint import (
    TRANSIENT_LTM_STATE_KEYS,
    load_checkpoint_payload_compatible,
    save_checkpoint_safely,
    load_full_model_with_config,
    load_model_state_dict_compatible,
    sanitize_model_state_dict,
    _infer_arch_flags_from_state_dict,
    _reject_unsupported_rwkv_state_dict,
)
from ..models.core import HierarchosCore

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def validate_loss(loss: torch.Tensor, name: str = "loss") -> bool:
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(f"ERROR: Invalid loss ({loss.item()}) detected in {name}!")
        return False
    return True

def _tensor_is_nonfinite(tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return not bool(torch.isfinite(tensor).all().item())

def _describe_tensor_issue(name: str, tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    nan_count = int(torch.isnan(detached).sum().item())
    inf_count = int(torch.isinf(detached).sum().item())
    finite = detached[torch.isfinite(detached)]
    if finite.numel() > 0:
        finite_min = float(finite.min().item())
        finite_max = float(finite.max().item())
        return f"{name} has {nan_count} NaN and {inf_count} Inf values; finite range=[{finite_min:.4e}, {finite_max:.4e}]"
    return f"{name} has {nan_count} NaN and {inf_count} Inf values; no finite values"

def _is_transient_ltm_state_name(name: str) -> bool:
    clean_name = str(name).replace("_orig_mod.", "")
    return any(clean_name.endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS)

INTENTIONAL_NONFINITE_BUFFER_KEYS = (
    "ltm.neg_inf",
)

def _is_intentional_nonfinite_buffer_name(name: str) -> bool:
    clean_name = str(name).replace("_orig_mod.", "")
    return any(clean_name.endswith(suffix) for suffix in INTENTIONAL_NONFINITE_BUFFER_KEYS)

def _find_first_nonfinite_model_tensor(model, include_grads: bool = False, include_transient_ltm: bool = False):
    for name, param in model.named_parameters():
        if _tensor_is_nonfinite(param):
            return _describe_tensor_issue(f"parameter {name}", param)
        if include_grads and param.grad is not None and _tensor_is_nonfinite(param.grad):
            return _describe_tensor_issue(f"gradient {name}", param.grad)
    for name, buffer in model.named_buffers():
        if _is_intentional_nonfinite_buffer_name(name):
            continue
        if not include_transient_ltm and _is_transient_ltm_state_name(name):
            continue
        if _tensor_is_nonfinite(buffer):
            return _describe_tensor_issue(f"buffer {name}", buffer)
    return None

def _find_first_nonfinite_optimizer_tensor(optimizer):
    if optimizer is None:
        return None
    for param_idx, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if _tensor_is_nonfinite(value):
                return _describe_tensor_issue(f"optimizer state[{param_idx}].{key}", value)
    return None

def _find_first_nonfinite_payload_tensor(value, path: str = "checkpoint"):
    if _is_intentional_nonfinite_buffer_name(path):
        return None
    if _is_transient_ltm_state_name(path):
        return None
    if torch.is_tensor(value):
        if _tensor_is_nonfinite(value):
            return _describe_tensor_issue(path, value)
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return f"{path} is non-finite ({value!r})"
    if isinstance(value, dict):
        for key, item in value.items():
            issue = _find_first_nonfinite_payload_tensor(item, f"{path}.{key}")
            if issue:
                return issue
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            issue = _find_first_nonfinite_payload_tensor(item, f"{path}[{idx}]")
            if issue:
                return issue
    return None

def _sanitize_payload_nonfinite_(value, path: str = "checkpoint", max_abs: float = 1.0) -> int:
    if _is_intentional_nonfinite_buffer_name(path):
        return 0
    if torch.is_tensor(value):
        if path.endswith("[0]") and "running_states[5]" in path:
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=0.0, neginf=0.0)
        if path.endswith("[1]") and "running_states[5]" in path:
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=max_abs, neginf=-max_abs)
        if ".model_state_dict." in path or ".grad_state_dict." in path:
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=1.0, neginf=-1.0)
        return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=1.0, neginf=-1.0)
    cleaned = 0
    if isinstance(value, dict):
        for key, item in value.items():
            cleaned += _sanitize_payload_nonfinite_(item, f"{path}.{key}", max_abs=max_abs)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            cleaned += _sanitize_payload_nonfinite_(item, f"{path}[{idx}]", max_abs=max_abs)
    return cleaned

def training_state_is_finite(model, optimizer=None, check_optimizer: bool = True, include_grads: bool = False) -> bool:
    issue = _find_first_nonfinite_model_tensor(model, include_grads=include_grads)
    if issue:
        print(f"CRITICAL: Non-finite training state detected: {issue}")
        return False
    if check_optimizer:
        issue = _find_first_nonfinite_optimizer_tensor(optimizer)
        if issue:
            print(f"CRITICAL: Non-finite training state detected: {issue}")
            return False
    return True

def _checkpoint_grad_clip(checkpoint_dict) -> float:
    config = checkpoint_dict.get("config") if isinstance(checkpoint_dict, dict) else None
    if isinstance(config, dict):
        try:
            return float(config.get("grad_clip", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0
    return 1.0

def _sanitize_ltm_payload_state_(value, path: str = "checkpoint", max_abs: float = 1.0) -> int:
    if _is_intentional_nonfinite_buffer_name(path):
        return 0
    if torch.is_tensor(value):
        clean_path = path.replace("_orig_mod.", "")
        if clean_path.endswith("ltm.fast_vals"):
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                changed = int(value.numel())
                value.zero_()
                return changed
            return 0
        if clean_path.endswith("ltm._mom_vals"):
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=max_abs, neginf=-max_abs)
        if clean_path.endswith("ltm.timestamps"):
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=0.0, neginf=0.0)
        return 0
    cleaned = 0
    if isinstance(value, dict):
        for key, item in value.items():
            cleaned += _sanitize_ltm_payload_state_(item, f"{path}.{key}", max_abs=max_abs)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            cleaned += _sanitize_ltm_payload_state_(item, f"{path}[{idx}]", max_abs=max_abs)
    return cleaned

def _component_is_finite(value) -> bool:
    return value is not None and torch.is_tensor(value) and bool(torch.isfinite(value).all().item())

def _reset_after_nonfinite(optimizer, model=None):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if model is not None and hasattr(model, "reset_memory"):
        model.reset_memory()
    return (None, None, None, None, None, None)

def _sanitize_tensor_nonfinite_(tensor: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> int:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return 0
    bad_count = int((~torch.isfinite(tensor)).sum().item())
    if bad_count:
        tensor.nan_to_num_(nan=nan, posinf=posinf, neginf=neginf)
    return bad_count

def _positive_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0.0 else default

def _nonnegative_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0.0 else default

def _cap_loss_component_for_backward(
    value: torch.Tensor,
    ceiling: float,
    *,
    preserve_gradient: bool = False,
) -> torch.Tensor:
    ceiling = _positive_float(ceiling, 0.0)
    if ceiling <= 0.0 or not torch.is_tensor(value):
        return value
    capped = torch.minimum(value, value.new_tensor(ceiling))
    if preserve_gradient:
        return value + (capped - value).detach()
    return capped

def _clamp_tensor_finite_magnitude_(tensor: torch.Tensor, max_abs: float) -> int:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return 0
    max_abs = _positive_float(max_abs, 0.0)
    if max_abs <= 0.0:
        return 0
    over = torch.abs(tensor) > max_abs
    over_count = int(over.sum().item())
    if over_count:
        tensor.clamp_(min=-max_abs, max=max_abs)
    return over_count

def _detach_finite_clamp(tensor: torch.Tensor, max_abs: float) -> torch.Tensor:
    max_abs = _positive_float(max_abs, 1.0)
    detached = tensor.detach()
    return torch.clamp(
        torch.nan_to_num(detached, nan=0.0, posinf=max_abs, neginf=-max_abs),
        min=-max_abs,
        max=max_abs,
    )

def _detach_finite_l2_clamp(tensor: torch.Tensor, max_abs: float, max_norm: float) -> torch.Tensor:
    detached = _detach_finite_clamp(tensor, max_abs)
    max_norm = _nonnegative_float(max_norm, 0.0)
    if max_norm <= 0.0 or not torch.is_tensor(detached) or not detached.is_floating_point():
        return detached
    norm = torch.linalg.vector_norm(detached.float(), ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(detached.new_tensor(max_norm) / (norm.to(dtype=detached.dtype) + 1e-6), max=1.0)
    return detached * scale

_RUNTIME_MODEL_CONFIG_KEYS = (
    "ltm_lr",
    "min_ltm_lr",
    "disable_ltm_lr_schedule",
    "ltm_training_mode",
    "ltm_score_grad_scale",
    "halt_logit_clamp",
    "recurrent_state_clamp",
    "context_state_clamp",
    "drift_state_clamp",
    "drift_norm_clamp",
    "drift_delta_scale",
    "activation_clamp",
    "rwkv_channel_mix_key_clamp",
    "rwkv_channel_mix_deepembed_clamp",
    "commitment_loss_weight",
    "max_commitment_cost_for_backward",
    "commitment_threshold",
    "l_conv_atol",
    "detach_every_n_steps",
    "full_sample_bptt",
    "full_sample_activation_checkpointing",
    "full_sample_checkpoint_segment_size",
    "inference_logit_parity",
    "inference_recurrence_mode",
    "h_halt_thresh",
    "gradient_checkpointing",
    "debug_numerics",
    "isolate_batch_ltm",
    "memory_gate_warmup_steps",
    "memory_gate_warmup_floor",
    "cuda_chunked_lm_loss",
    "cuda_loss_chunk_rows",
    "cpu_chunked_lm_loss",
    "cpu_loss_chunk_rows",
)

def _normalize_detach_every_n_steps(value):
    """Normalize the documented 0/negative sentinel without changing checkpoints."""
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"detach_every_n_steps must be an integer or None, got {value!r}"
        ) from exc
    return value if value > 0 else None


def _apply_runtime_model_config_overrides(model_config, args):
    for key in _RUNTIME_MODEL_CONFIG_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            if key == "detach_every_n_steps":
                value = _normalize_detach_every_n_steps(value)
            model_config[key] = value
    if hasattr(args, "compile_static_worker_loop") and getattr(args, "compile_static_worker_loop") is not None:
        model_config.compile_static_worker_loop = getattr(args, "compile_static_worker_loop")
    return model_config


def configure_full_sample_bptt(args, config=None, *, announce=False):
    """Apply exact per-sample BPTT invariants without changing model tensors.

    The persisted ``training_chunk_size`` remains untouched because it also keys
    token caches, ROSA precomputation, LTM decay calibration, and compile shapes.
    Without activation checkpointing, ``train_step`` uses one whole-sample
    forward. With checkpointing, it retains attached recurrent boundary states
    between activation-only temporal segments and performs one backward pass after
    the complete sample graph has been assembled. Segmenting therefore bounds
    saved activations without truncating the gradient horizon.
    """
    enabled = bool(getattr(args, "full_sample_bptt", False))
    inference_recurrence_mode = "full-sample" if enabled else "tbptt"
    args.inference_recurrence_mode = inference_recurrence_mode
    # Training always uses the fixed Manager/Worker refinement policy.  Keep
    # that policy at inference independently from whether recurrent gradients
    # were full-sample or truncated at training_chunk_size boundaries.
    args.inference_logit_parity = True
    checkpoint_enabled = getattr(
        args,
        "full_sample_activation_checkpointing",
        None,
    )
    checkpoint_enabled = enabled if checkpoint_enabled is None else bool(checkpoint_enabled)
    checkpoint_segment_size = int(
        getattr(args, "full_sample_checkpoint_segment_size", 128) or 128
    )
    if checkpoint_segment_size <= 0:
        raise ValueError("full_sample_checkpoint_segment_size must be positive")

    if enabled:
        args.detach_every_n_steps = 0
        args.persist_state = False
        args.full_sample_activation_checkpointing = checkpoint_enabled
        args.full_sample_checkpoint_segment_size = checkpoint_segment_size
        # With one isolated forward, the legacy Titans write happens only after
        # backward and cannot affect the sample that produced it. The next sample
        # resets working memory, so retaining every raw retrieval gradient and
        # then writing a terminal state is pure overhead. Read-only still routes
        # LTM per token and keeps all ordinary parameter/score gradients.
        terminal_ltm_update_disabled = (
            normalize_ltm_training_mode(
                getattr(args, "ltm_training_mode", "inner-update")
            ) == "inner-update"
        )
        if terminal_ltm_update_disabled:
            args.ltm_training_mode = "read-only"
            if announce:
                print(
                    "INFO: Skipping the terminal post-backward LTM fast-memory write: "
                    "it is discarded before the next isolated sample and cannot "
                    "change this sample's gradients. Per-token LTM routing remains active."
                )

        # Attached temporal-segment checkpoints already rematerialize every
        # WorkerLoop invocation. Nesting the older per-token WorkerLoop
        # checkpoint adds redundant recomputation without extending the
        # gradient horizon.
        nested_worker_checkpoint = bool(getattr(args, "gradient_checkpointing", False))
        if checkpoint_enabled and nested_worker_checkpoint:
            args.gradient_checkpointing = False
            if announce:
                print(
                    "INFO: Attached full-sample activation checkpointing supersedes the "
                    "nested WorkerLoop checkpoint."
                )

        if announce:
            memory_mode = (
                "attached temporal-segment activation recomputation"
                if checkpoint_enabled
                else "saved full-sample activations"
            )
            print(
                "INFO: Full-sample BPTT enabled: one end-to-end attached graph per "
                f"trimmed sample batch, recurrent detach disabled, {memory_mode}."
            )
            print(
                "INFO: Configured training_chunk_size remains cache/ROSA/LTM/compile "
                "geometry; activation-only segment length is "
                f"{checkpoint_segment_size}."
            )
    elif checkpoint_enabled:
        raise ValueError(
            "full_sample_activation_checkpointing requires full_sample_bptt"
        )

    if config is not None:
        config.full_sample_bptt = enabled
        config.inference_recurrence_mode = inference_recurrence_mode
        config.inference_logit_parity = True
        config.full_sample_activation_checkpointing = bool(checkpoint_enabled)
        config.full_sample_checkpoint_segment_size = checkpoint_segment_size
        config.detach_every_n_steps = _normalize_detach_every_n_steps(
            getattr(args, "detach_every_n_steps", None)
        )
        if enabled:
            config.persist_state = False
            config.ltm_training_mode = getattr(args, "ltm_training_mode", "read-only")
            if checkpoint_enabled:
                config.gradient_checkpointing = False

    return enabled


def _checkpointed_training_model_call(model, model_kwargs):
    """Checkpoint one pure training forward while retaining nested outputs.

    Non-reentrant checkpointing supports dictionaries/lists and records the
    autograd graph needed by the gradient-derived LTM tensors. Training forwards
    suppress Hebbian writes; the trainer commits any LTM inner update exactly
    once after backward, so rematerialization has no persistent model side effect.
    """
    # Pass every value explicitly. In attached segmented BPTT, recurrent states
    # are differentiable inputs from the preceding segment; making them checkpoint
    # arguments avoids relying on closure capture and lets non-reentrant
    # checkpointing discover tensors nested inside the LTM state tuple.
    keys = tuple(model_kwargs)
    values = tuple(model_kwargs[key] for key in keys)

    def run(*replay_values):
        return model(**dict(zip(keys, replay_values)))

    return activation_checkpoint(
        run,
        *values,
        use_reentrant=False,
        # Hierarchos' training forward has no dropout or random sampling. Avoid
        # stashing/restoring CPU+CUDA RNG state roughly max_length/segment_size
        # times per batch; direct-vs-segmented gradient tests guard this premise.
        preserve_rng_state=False,
        determinism_check="default",
    )


def _restore_resume_component_state(component, state, component_name: str, checkpoint_path: str):
    """Restore exact continuation state or stop before silently changing dynamics."""
    if state is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no {component_name} state. Exact --resume-from-ckpt "
            "cannot continue safely; use --override-scheduling to intentionally start fresh "
            "optimizer/scheduler state, or use --model-path for a weights-only continuation."
        )
    issue = _find_first_nonfinite_payload_tensor(state, f"{component_name}_state")
    if issue:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has corrupt {component_name} state: {issue}. "
            "Refusing to rewrite non-finite continuation state; use an earlier checkpoint, "
            "or --override-scheduling only when a fresh optimizer/scheduler is intentional."
        )
    try:
        component.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(
            f"Could not restore {component_name} state from {checkpoint_path}: {exc}. "
            "Refusing a silent fresh-state resume. Use --override-scheduling only if that reset "
            "is intentional."
        ) from exc

def _print_runtime_stability_config(model):
    config = getattr(model, "config", None)
    if config is None:
        return
    threshold = _nonnegative_float(getattr(config, "commitment_threshold", 0.05), 0.05)
    norm_clamp = _nonnegative_float(getattr(config, "drift_norm_clamp", 0.0), 0.0)
    print(
        "INFO: Runtime stability config: "
        f"drift_state_clamp={getattr(config, 'drift_state_clamp', 5.0)}, "
        f"drift_norm_clamp={norm_clamp:g}, "
        f"drift_delta_scale={getattr(config, 'drift_delta_scale', 1.0)}, "
        f"rwkv_channel_mix_key_clamp={getattr(config, 'rwkv_channel_mix_key_clamp', 12.0)}, "
        f"rwkv_channel_mix_deepembed_clamp={getattr(config, 'rwkv_channel_mix_deepembed_clamp', 4.0)}, "
        f"commitment_threshold={threshold:g}, "
        f"max_commitment_cost_for_backward={getattr(config, 'max_commitment_cost_for_backward', 'trainer-only')}"
    )
    if norm_clamp > 0.0:
        max_commit = max(0.0, norm_clamp * norm_clamp - threshold)
        print(f"INFO: Drift norm clamp invariant: raw/displayed commit should stay <= {max_commit:.4g}.")

def _clamp_running_states_for_resume(running_states, args):
    if not running_states or len(running_states) < 5:
        return running_states
    states = list(running_states)
    if torch.is_tensor(states[4]):
        states[4] = _detach_finite_l2_clamp(
            states[4],
            getattr(args, "drift_state_clamp", 5.0),
            getattr(args, "drift_norm_clamp", 0.0),
        )
    return tuple(states)

def _sanitize_optimizer_state_(optimizer) -> int:
    if optimizer is None:
        return 0
    cleaned = 0
    for state in optimizer.state.values():
        for value in state.values():
            cleaned += _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=0.0, neginf=0.0)
    if cleaned:
        print(f"WARNING: Reset {cleaned} non-finite optimizer state value(s) to 0.0 for recovery resume.")
    return cleaned

def _sanitize_model_nonfinite_(model, *, include_transient_ltm: bool = False, log_prefix: str = "model") -> int:
    cleaned = 0
    first_issue = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if first_issue is None and _tensor_is_nonfinite(param):
                first_issue = _describe_tensor_issue(f"parameter {name}", param)
            cleaned += _sanitize_tensor_nonfinite_(param, nan=0.0, posinf=1.0, neginf=-1.0)
        for name, buffer in model.named_buffers():
            if _is_intentional_nonfinite_buffer_name(name):
                continue
            if not include_transient_ltm and _is_transient_ltm_state_name(name):
                continue
            if first_issue is None and _tensor_is_nonfinite(buffer):
                first_issue = _describe_tensor_issue(f"buffer {name}", buffer)
            cleaned += _sanitize_tensor_nonfinite_(buffer, nan=0.0, posinf=1.0, neginf=-1.0)
    if cleaned:
        detail = f" First repaired tensor: {first_issue}." if first_issue else ""
        print(f"WARNING: Sanitized {cleaned} non-finite {log_prefix} parameter/buffer value(s): NaN->0, +Inf->1, -Inf->-1.{detail}")
    return cleaned

def _clamp_model_finite_magnitude_(model, max_abs: float, *, include_transient_ltm: bool = False, log_prefix: str = "model") -> int:
    max_abs = _positive_float(max_abs, 0.0)
    if max_abs <= 0.0:
        return 0
    clamped = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            clamped += _clamp_tensor_finite_magnitude_(param, max_abs)
        for name, buffer in model.named_buffers():
            if _is_intentional_nonfinite_buffer_name(name):
                continue
            if not include_transient_ltm and _is_transient_ltm_state_name(name):
                continue
            clamped += _clamp_tensor_finite_magnitude_(buffer, max_abs)
    if clamped:
        print(f"WARNING: Clamped {clamped} finite {log_prefix} parameter/buffer value(s) to +/-{max_abs:g}.")
    return clamped

def _sanitize_model_transient_state_(model, max_abs: float = 1.0) -> int:
    cleaned = 0
    max_abs = float(max_abs or 0.0)
    if max_abs <= 0.0:
        max_abs = 1.0
    for name, buffer in model.named_buffers():
        clean_name = str(name).replace("_orig_mod.", "")
        if clean_name.endswith("ltm.fast_vals"):
            if torch.is_tensor(buffer) and buffer.is_floating_point():
                changed = int(torch.count_nonzero(buffer).item())
                if changed:
                    buffer.zero_()
                    cleaned += changed
        elif clean_name.endswith("ltm._mom_vals"):
            cleaned += _sanitize_tensor_nonfinite_(buffer, nan=0.0, posinf=max_abs, neginf=-max_abs)
        elif clean_name.endswith("ltm.timestamps"):
            cleaned += _sanitize_tensor_nonfinite_(buffer, nan=0.0, posinf=0.0, neginf=0.0)
        elif clean_name.endswith("ltm.sources"):
            if torch.is_tensor(buffer) and not bool(torch.isfinite(buffer.float()).all().item()):
                buffer.fill_(0)
                cleaned += int(buffer.numel())
    if cleaned:
        print(
            f"WARNING: Sanitized transient LTM state ({cleaned} value(s)): "
            "fast_vals reset; _mom_vals saturated; metadata reset."
        )
    return cleaned

def _sanitize_gradient_nonfinite_(model, max_abs: float) -> int:
    cleaned = 0
    max_abs = _positive_float(max_abs, 1.0)
    for param in model.parameters():
        if param.grad is not None:
            cleaned += _sanitize_tensor_nonfinite_(param.grad, nan=0.0, posinf=max_abs, neginf=-max_abs)
    if cleaned:
        print(f"WARNING: Sanitized {cleaned} non-finite gradient value(s): NaN->0, +Inf->{max_abs:g}, -Inf->{-max_abs:g} before clipping.")
    return cleaned

def _find_first_nonfinite_gradient_tensor(model):
    for name, param in model.named_parameters():
        if param.grad is not None and _tensor_is_nonfinite(param.grad):
            return _describe_tensor_issue(f"gradient {name}", param.grad)
    return None

def _summarize_nonfinite_gradient_tensors(model, limit: int = 8):
    rows = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None or not torch.is_tensor(grad) or not grad.is_floating_point():
            continue
        detached = grad.detach()
        nan_count = int(torch.isnan(detached).sum().item())
        inf_count = int(torch.isinf(detached).sum().item())
        total = nan_count + inf_count
        if total > 0:
            rows.append((total, nan_count, inf_count, name, tuple(detached.shape)))
    if not rows:
        return None
    rows.sort(key=lambda item: item[0], reverse=True)
    parts = []
    for total, nan_count, inf_count, name, shape in rows[:max(1, int(limit or 1))]:
        parts.append(
            f"{name} shape={shape} nonfinite={total} "
            f"(NaN={nan_count}, Inf={inf_count})"
        )
    return "; ".join(parts)

def _manual_clip_grad_norm_(params, max_norm: float, *, return_finite: bool = False):
    params = list(params)
    norms = []
    dense_float32_groups = {}
    for param in params:
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce()._values()
        if not grad.is_sparse and grad.dtype == torch.float32:
            dense_float32_groups.setdefault(grad.device, []).append(grad)
        else:
            norms.append(grad.float().norm(2))
    for device_grads in dense_float32_groups.values():
        try:
            norms.extend(torch._foreach_norm(device_grads, 2.0))
        except (RuntimeError, TypeError):
            norms.extend(grad.norm(2) for grad in device_grads)
    if norms:
        total_norm = torch.stack(norms).norm(2)
    else:
        total_norm = torch.zeros(())
    total_norm_value = float(total_norm.item())
    total_norm_finite = math.isfinite(total_norm_value)
    if torch.is_tensor(total_norm) and not total_norm_finite:
        # Fast fp32 norm can overflow when gradients are finite but huge. Fall back
        # to a max-scaled norm so finite gradients still get clipped instead of
        # being treated like NaN/Inf gradients.
        max_abs = None
        for param in params:
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce()._values()
            local_max = grad.float().abs().max()
            max_abs = local_max if max_abs is None else torch.maximum(max_abs.to(local_max.device), local_max)
        max_abs_value = float(max_abs.item()) if max_abs is not None else 0.0
        if max_abs is not None and math.isfinite(max_abs_value) and max_abs_value > 0.0:
            sum_sq = None
            for param in params:
                grad = param.grad.detach()
                if grad.is_sparse:
                    grad = grad.coalesce()._values()
                scaled = grad.float() / max_abs.to(grad.device)
                local_sq = scaled.pow(2).sum(dtype=torch.float64)
                sum_sq = local_sq if sum_sq is None else sum_sq + local_sq.to(sum_sq.device)
            if sum_sq is not None:
                total_norm = max_abs.to(dtype=torch.float64) * torch.sqrt(sum_sq.to(max_abs.device))
                total_norm_value = float(total_norm.item())
                total_norm_finite = math.isfinite(total_norm_value)
    if max_norm > 0.0 and total_norm_finite:
        # This public foreach-backed helper applies the same L2 coefficient while
        # avoiding a host-side `clip_coef < 1` branch and one kernel per tensor.
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=None)
    if return_finite:
        return total_norm, total_norm_finite
    return total_norm


def _prepare_ltm_update_gradients(grads: torch.Tensor, max_norm: float, *, inplace: bool = False):
    """Reject NaN/Inf LTM gradients, then clip finite gradients like model grads."""
    if not torch.is_tensor(grads):
        return None
    prepared = grads.detach()
    if prepared.dtype != torch.float32:
        prepared = prepared.float()
    elif not inplace:
        prepared = prepared.clone()
    max_norm = _positive_float(max_norm, 1.0)
    values_are_finite = torch.isfinite(prepared).all()
    if max_norm > 0:
        prepared.clamp_(min=-max_norm, max=max_norm)
        grad_norm = prepared.norm()
        checks_are_finite = torch.stack((values_are_finite, torch.isfinite(grad_norm))).all()
        if not bool(checks_are_finite.item()):
            return None
        prepared.mul_(torch.clamp(prepared.new_tensor(max_norm) / (grad_norm + 1e-8), max=1.0))
    elif not bool(values_are_finite.item()):
        return None
    return prepared

def save_training_checkpoint_if_finite(checkpoint_dict, path: str, model, optimizer=None) -> bool:
    model_issue = _find_first_nonfinite_model_tensor(model, include_grads=True)
    if model_issue:
        raise RuntimeError(
            f"Refusing to save a checkpoint with non-finite learned/gradient state: {model_issue}. "
            "The previous atomic checkpoint was left untouched."
        )
    optimizer_issue = _find_first_nonfinite_optimizer_tensor(optimizer)
    if optimizer_issue:
        raise RuntimeError(
            f"Refusing to save a checkpoint with non-finite optimizer state: {optimizer_issue}. "
            "The previous atomic checkpoint was left untouched."
        )

    max_abs = _checkpoint_grad_clip(checkpoint_dict)
    if isinstance(checkpoint_dict, dict) and model is not None and "model_state_dict" in checkpoint_dict:
        reset_transient_ltm = bool(checkpoint_dict.get("training_complete", False))
        checkpoint_dict["model_state_dict"] = sanitize_model_state_dict(
            model,
            reset_transient_ltm=reset_transient_ltm,
        )

    running_states = checkpoint_dict.get("running_states") if isinstance(checkpoint_dict, dict) else None
    running_issue = _find_first_nonfinite_payload_tensor(running_states, "running_states")
    if running_issue:
        checkpoint_dict["running_states"] = None
        print(
            f"WARNING: Dropping non-finite transient running state from checkpoint: {running_issue}. "
            "Learned weights and optimizer state remain unchanged."
        )

    ltm_cleaned = _sanitize_ltm_payload_state_(checkpoint_dict, max_abs=max_abs)
    if ltm_cleaned:
        print(f"WARNING: Sanitized {ltm_cleaned} transient LTM checkpoint value(s) before saving.")
    issue = _find_first_nonfinite_payload_tensor(checkpoint_dict)
    if issue:
        raise RuntimeError(
            f"Refusing to save a checkpoint with non-finite payload state: {issue}. "
            "The previous atomic checkpoint was left untouched."
        )
    save_checkpoint_safely(checkpoint_dict, path)
    return True

def _clip_gradients_and_check(model, max_norm: float, max_sanitized_values: int = 0):
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return True, None
    max_norm = float(max_norm or 0.0)
    # The old healthy path scanned every parameter twice with `.item()`, which
    # serialized the CPU with CUDA once per gradient tensor. The global L2 norm
    # is non-finite whenever any dense gradient contains NaN/Inf, so use its one
    # host decision as the fast path and reserve detailed per-tensor scans for
    # the exceptional diagnostic path.
    total_norm, total_norm_finite = _manual_clip_grad_norm_(
        params,
        max_norm,
        return_finite=True,
    )
    if not total_norm_finite:
        issue = _find_first_nonfinite_gradient_tensor(model)
        summary = _summarize_nonfinite_gradient_tensors(model)
        if issue and summary:
            return False, f"{issue}. Top non-finite gradient tensors: {summary}"
        if issue:
            return False, issue
        return False, "gradient norm became non-finite during finite-gradient clipping"
    return True, total_norm

def _has_pending_gradients(model) -> bool:
    return any(p.requires_grad and p.grad is not None for p in model.parameters())

def set_model_training_step(model, step: int):
    setter = getattr(model, "set_training_step", None)
    if callable(setter):
        setter(step)
        return
    base_model = getattr(model, "base_model", None)
    inner_model = getattr(base_model, "model", None)
    setter = getattr(inner_model, "set_training_step", None)
    if callable(setter):
        setter(step)


def mark_val_proj_trained(model):
    """Record a successful writer update through compiled and PEFT wrappers."""
    candidates = [model, getattr(model, "_orig_mod", None)]
    base_model = getattr(model, "base_model", None)
    candidates.extend(
        [
            base_model,
            getattr(base_model, "model", None),
            getattr(getattr(base_model, "model", None), "_orig_mod", None),
        ]
    )
    seen = set()
    seen_configs = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is None or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        minimum_updates = max(
            1,
            int(
                config.get("ltm_value_alignment_min_updates", 100)
                if isinstance(config, dict)
                else getattr(config, "ltm_value_alignment_min_updates", 100)
            ),
        )
        if isinstance(config, dict):
            updates = int(config.get("val_proj_alignment_updates", 0) or 0) + 1
            config["val_proj_alignment_updates"] = updates
            if updates >= minimum_updates:
                config["val_proj_trained"] = True
        else:
            try:
                updates = int(getattr(config, "val_proj_alignment_updates", 0) or 0) + 1
                setattr(config, "val_proj_alignment_updates", updates)
                if updates >= minimum_updates:
                    setattr(config, "val_proj_trained", True)
            except Exception:
                pass


def compute_chunk_training_weights(labels: torch.Tensor, attention_mask: torch.Tensor = None,
                                   chunk_size: int = 128, loss_weights: torch.Tensor = None,
                                   h_stride: int = None):
    """
    Build TBPTT chunk weights that match the causal objective.

    CrossEntropy is averaged over valid shifted labels inside each chunk. Each
    input chunk receives a one-token label lookahead so the last hidden state in
    a chunk is trained to predict the first token of the next chunk. When
    loss_weights are present, CE chunks are aggregated by supervised weight
    mass rather than raw label count. Auxiliary costs are token-level dynamics,
    so they use real attention-mask tokens instead.
    """
    B, T = labels.shape
    if chunk_size <= 0 or chunk_size > T:
        chunk_size = T

    chunks = []
    total_valid_predictions = 0
    total_prediction_weight = 0.0
    total_real_tokens = 0
    total_ponder_tokens = 0

    for start_t in range(0, T, chunk_size):
        end_t = min(start_t + chunk_size, T)
        loss_label_start = start_t + 1
        loss_label_end = min(end_t + 1, T)

        if loss_label_end > loss_label_start:
            valid_mask = labels[:, loss_label_start:loss_label_end] != -100
            valid_predictions = int(valid_mask.sum().item())
            if loss_weights is not None:
                weight_slice = loss_weights[:, loss_label_start:loss_label_end].float()
                prediction_weight = float((weight_slice * valid_mask.float()).sum().item())
            else:
                prediction_weight = float(valid_predictions)
        else:
            valid_predictions = 0
            prediction_weight = 0.0

        if attention_mask is not None:
            real_tokens = int(attention_mask[:, start_t:end_t].sum().item())
        else:
            real_tokens = B * (end_t - start_t)

        # Manager ponder cost is emitted only at absolute h_stride positions.
        # Tracking that mass separately makes attached segmented full BPTT
        # mathematically match a single whole-sample forward, including a short
        # final segment whose manager-step density can differ from earlier ones.
        if h_stride is not None and int(h_stride) > 0:
            ponder_positions = [
                position - start_t
                for position in range(start_t, end_t)
                if position % int(h_stride) == 0
            ]
            if ponder_positions:
                if attention_mask is not None:
                    ponder_tokens = int(
                        attention_mask[:, start_t:end_t][:, ponder_positions].sum().item()
                    )
                else:
                    ponder_tokens = B * len(ponder_positions)
            else:
                ponder_tokens = 0
        else:
            # Preserve the historical TBPTT weighting when callers do not
            # provide architecture stride information.
            ponder_tokens = real_tokens

        chunks.append({
            "start": start_t,
            "end": end_t,
            "valid_predictions": valid_predictions,
            "prediction_weight": prediction_weight,
            "real_tokens": real_tokens,
            "ponder_tokens": ponder_tokens,
        })
        total_valid_predictions += valid_predictions
        total_prediction_weight += prediction_weight
        total_real_tokens += real_tokens
        total_ponder_tokens += ponder_tokens

    for chunk in chunks:
        chunk["label_ratio"] = (
            chunk["prediction_weight"] / float(total_prediction_weight)
            if total_prediction_weight > 0 else 0.0
        )
        chunk["token_ratio"] = (
            chunk["real_tokens"] / float(total_real_tokens)
            if total_real_tokens > 0 else 0.0
        )
        chunk["ponder_ratio"] = (
            chunk["ponder_tokens"] / float(total_ponder_tokens)
            if total_ponder_tokens > 0 else 0.0
        )

    return chunks


def compute_supervision_coverage_stats(labels: torch.Tensor, attention_mask: torch.Tensor = None):
    if labels is None or labels.ndim != 2:
        return {
            "active_tokens": 0,
            "active_masked_labels": 0,
            "shifted_active_predictions": 0,
            "shifted_supervised_predictions": 0,
            "active_label_coverage": 0.0,
            "shifted_supervision_coverage": 0.0,
        }

    labels_cpu = labels.detach().cpu()
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape == labels.shape:
        active_mask = attention_mask.detach().cpu().to(dtype=torch.bool)
    else:
        active_mask = torch.ones_like(labels_cpu, dtype=torch.bool)

    active_tokens = int(active_mask.sum().item())
    active_masked_labels = int(((labels_cpu == -100) & active_mask).sum().item())
    active_label_coverage = (
        (active_tokens - active_masked_labels) / float(active_tokens)
        if active_tokens > 0 else 0.0
    )

    if labels_cpu.shape[1] > 1:
        shifted_active = active_mask[:, 1:]
        shifted_labels = labels_cpu[:, 1:]
        shifted_active_predictions = int(shifted_active.sum().item())
        shifted_supervised_predictions = int(((shifted_labels != -100) & shifted_active).sum().item())
    else:
        shifted_active_predictions = 0
        shifted_supervised_predictions = 0

    shifted_supervision_coverage = (
        shifted_supervised_predictions / float(shifted_active_predictions)
        if shifted_active_predictions > 0 else 0.0
    )

    return {
        "active_tokens": active_tokens,
        "active_masked_labels": active_masked_labels,
        "shifted_active_predictions": shifted_active_predictions,
        "shifted_supervised_predictions": shifted_supervised_predictions,
        "active_label_coverage": active_label_coverage,
        "shifted_supervision_coverage": shifted_supervision_coverage,
    }


def audit_supervision_coverage_once(args, labels: torch.Tensor, attention_mask: torch.Tensor = None, *, step: int = 0):
    if getattr(args, "_supervision_audit_done", False):
        return None
    args._supervision_audit_done = True

    stats = compute_supervision_coverage_stats(labels, attention_mask)
    print(
        "INFO: Supervision coverage audit "
        f"(step {step + 1}): active_tokens={stats['active_tokens']}, "
        f"active_label_coverage={stats['active_label_coverage']:.4f}, "
        f"shifted_ce_coverage={stats['shifted_supervision_coverage']:.4f}, "
        f"masked_active_labels={stats['active_masked_labels']}."
    )

    prompt_completion_mode = (
        getattr(args, 'alpaca', False)
        or getattr(args, 'kayla', False)
        or bool(getattr(args, 'prompt_column', None))
        or bool(getattr(args, 'completion_column', None))
    )
    if (
        prompt_completion_mode
        and getattr(args, 'train_prompt_tokens', True)
        and getattr(args, 'strict_all_token_loss', True)
        and stats["active_masked_labels"] > 0
    ):
        raise RuntimeError(
            "All-token loss audit failed: real prompt/completion tokens still have "
            "label=-100 even though train_prompt_tokens=True. This usually means "
            "a stale masked HF token cache/PT shard cache, legacy pretokenized data, "
            "or --mask-prompt-tokens. Refresh/delete the token cache or pass "
            "--allow-masked-active-labels only for intentional legacy SFT."
        )
    return stats

def compute_update_steps(dataloader_len: int, accumulation_steps: int) -> int:
    """Count optimizer updates, including the final partial accumulation window."""
    accumulation_steps = max(1, int(accumulation_steps))
    dataloader_len = max(0, int(dataloader_len))
    if dataloader_len <= 0:
        return 0
    return (dataloader_len + accumulation_steps - 1) // accumulation_steps

def accumulation_divisor_for_step(step: int, dataloader_len: int, accumulation_steps: int) -> int:
    """Scale loss by the real accumulation window size for this dataloader step."""
    accumulation_steps = max(1, int(accumulation_steps))
    dataloader_len = max(1, int(dataloader_len))
    step = max(0, min(int(step), dataloader_len - 1))
    window_start = (step // accumulation_steps) * accumulation_steps
    window_end = min(window_start + accumulation_steps, dataloader_len)
    return max(1, window_end - window_start)

def should_step_accumulation(step: int, dataloader_len: int, accumulation_steps: int) -> bool:
    accumulation_steps = max(1, int(accumulation_steps))
    dataloader_len = max(1, int(dataloader_len))
    return ((int(step) + 1) % accumulation_steps == 0) or (int(step) + 1 >= dataloader_len)

def compute_remaining_update_steps(dataloader_len: int, accumulation_steps: int, start_epoch: int,
                                   total_epochs: int, start_step: int = 0) -> int:
    """Count optimizer updates that will actually run after an epoch/mid-epoch resume."""
    accumulation_steps = max(1, int(accumulation_steps))
    dataloader_len = max(0, int(dataloader_len))
    start_step = max(0, min(int(start_step), dataloader_len))
    remaining_epochs_after_current = max(0, int(total_epochs) - int(start_epoch) - 1)
    updates_per_full_epoch = compute_update_steps(dataloader_len, accumulation_steps)
    if start_step >= dataloader_len:
        updates_already_done_this_epoch = updates_per_full_epoch
    else:
        updates_already_done_this_epoch = start_step // accumulation_steps
    updates_left_this_epoch = max(0, updates_per_full_epoch - updates_already_done_this_epoch)
    remaining_updates = updates_left_this_epoch + (remaining_epochs_after_current * updates_per_full_epoch)
    return max(1, remaining_updates)

def resolve_lr_warmup_steps(args, num_update_steps: int) -> int:
    total_steps = max(1, int(num_update_steps or 1))
    max_warmup = max(0, total_steps - 1)
    try:
        explicit_steps = int(getattr(args, 'warmup_steps', 0) or 0)
    except (TypeError, ValueError):
        explicit_steps = 0
    if explicit_steps > 0:
        return min(explicit_steps, max_warmup)

    warmup_ratio = _nonnegative_float(getattr(args, 'warmup_ratio', 0.0), 0.0)
    if warmup_ratio <= 0.0:
        return 0
    return min(int(math.ceil(total_steps * min(warmup_ratio, 1.0))), max_warmup)

def build_lr_scheduler(optimizer, args, num_update_steps: int):
    if getattr(args, 'disable_lr_schedule', False) or num_update_steps <= 0:
        return None

    total_steps = max(1, int(num_update_steps or 1))
    max_lr = _positive_float(getattr(args, 'starting_lr', 1e-4), 1e-4)
    min_lr = min(_nonnegative_float(getattr(args, 'min_lr', 0.0), 0.0), max_lr)
    min_factor = min_lr / max_lr if max_lr > 0.0 else 0.0
    warmup_steps = resolve_lr_warmup_steps(args, total_steps)

    def lr_lambda(current_step: int):
        current_step = max(0, int(current_step))
        if warmup_steps > 0 and current_step < warmup_steps:
            progress = (current_step + 1) / float(warmup_steps)
            return min_factor + (1.0 - min_factor) * min(progress, 1.0)

        decay_steps = max(1, total_steps - warmup_steps)
        decay_step = min(max(0, current_step - warmup_steps), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / float(decay_steps)))
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(
        f"INFO: Warmup+Cosine LR scheduler ENABLED. Total updates: {total_steps}, "
        f"Warmup updates: {warmup_steps}, Max LR: {max_lr:.2e}, Min LR: {min_lr:.2e}"
    )
    return scheduler

def _resolve_ltm_lr_bounds(args):
    max_lr = _positive_float(getattr(args, 'ltm_lr', 1e-3), 1e-3)
    min_ltm_lr = getattr(args, 'min_ltm_lr', None)
    if min_ltm_lr is None:
        min_lr = _nonnegative_float(getattr(args, 'min_lr', 0.0), 0.0)
    else:
        min_lr = _nonnegative_float(min_ltm_lr, 0.0)
    return max_lr, min(min_lr, max_lr)

def _cosine_annealed_value(max_value: float, min_value: float, step: int, total_steps: int) -> float:
    max_value = _nonnegative_float(max_value, 0.0)
    min_value = min(_nonnegative_float(min_value, 0.0), max_value)
    total_steps = max(1, int(total_steps or 1))
    step = max(0, min(int(step or 0), total_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / float(total_steps)))
    return min_value + (max_value - min_value) * cosine

def configure_ltm_lr_schedule(args, num_update_steps: int, checkpoint=None, *, override_schedule: bool = False, scheduler=None):
    max_lr, min_lr = _resolve_ltm_lr_bounds(args)
    schedule_enabled = not bool(getattr(args, 'disable_ltm_lr_schedule', False))
    total_steps = max(1, int(num_update_steps or 1))
    current_step = 0

    state = checkpoint.get('ltm_scheduler_state') if isinstance(checkpoint, dict) else None
    if schedule_enabled and state and not override_schedule:
        total_steps = max(1, int(state.get('total_steps', total_steps) or total_steps))
        current_step = max(0, int(state.get('step', 0) or 0))
    elif schedule_enabled and scheduler is not None and not override_schedule:
        current_step = max(0, int(getattr(scheduler, 'last_epoch', 0) or 0))

    args._ltm_lr_schedule_enabled = schedule_enabled
    args._ltm_lr_schedule_total_steps = total_steps
    args._ltm_lr_schedule_step = min(current_step, total_steps)
    args._ltm_lr_max = max_lr
    args._ltm_lr_min = min_lr
    args._current_ltm_lr = get_current_ltm_lr(args)
    if schedule_enabled:
        print(
            f"INFO: Cosine Annealing LTM LR scheduler ENABLED. "
            f"Total steps: {total_steps}, Max LTM LR: {max_lr:.2e}, Min LTM LR: {min_lr:.2e}"
        )
    else:
        print(f"INFO: LTM LR scheduler disabled. Fixed LTM LR: {max_lr:.2e}")
    return args._current_ltm_lr

def get_current_ltm_lr(args) -> float:
    max_lr = _positive_float(getattr(args, '_ltm_lr_max', getattr(args, 'ltm_lr', 1e-3)), 1e-3)
    min_lr = _nonnegative_float(getattr(args, '_ltm_lr_min', getattr(args, 'min_ltm_lr', 0.0) or 0.0), 0.0)
    if not bool(getattr(args, '_ltm_lr_schedule_enabled', True)):
        return max_lr
    return _cosine_annealed_value(
        max_lr,
        min_lr,
        getattr(args, '_ltm_lr_schedule_step', 0),
        getattr(args, '_ltm_lr_schedule_total_steps', 1),
    )

def advance_ltm_lr_schedule(args):
    if bool(getattr(args, '_ltm_lr_schedule_enabled', True)):
        total_steps = max(1, int(getattr(args, '_ltm_lr_schedule_total_steps', 1) or 1))
        current_step = max(0, int(getattr(args, '_ltm_lr_schedule_step', 0) or 0))
        args._ltm_lr_schedule_step = min(current_step + 1, total_steps)
    args._current_ltm_lr = get_current_ltm_lr(args)
    return args._current_ltm_lr

def capture_ltm_lr_scheduler_state(args):
    return {
        "enabled": bool(getattr(args, '_ltm_lr_schedule_enabled', True)),
        "step": int(getattr(args, '_ltm_lr_schedule_step', 0) or 0),
        "total_steps": int(getattr(args, '_ltm_lr_schedule_total_steps', 1) or 1),
        "max_lr": float(getattr(args, '_ltm_lr_max', getattr(args, 'ltm_lr', 1e-3))),
        "min_lr": float(getattr(args, '_ltm_lr_min', getattr(args, 'min_ltm_lr', 0.0) or 0.0)),
    }

def capture_main_lr_scheduler_state(args, scheduler=None, num_update_steps: int = None):
    if getattr(args, 'disable_lr_schedule', False):
        return {"enabled": False}
    total_steps = int(num_update_steps or getattr(args, '_main_lr_schedule_total_steps', 0) or 0)
    step = int(getattr(scheduler, 'last_epoch', 0) or 0) if scheduler is not None else 0
    return {
        "enabled": scheduler is not None,
        "step": max(0, step),
        "total_steps": max(1, total_steps),
        "max_lr": float(getattr(args, 'starting_lr', 1e-4)),
        "min_lr": float(getattr(args, 'min_lr', 0.0)),
        "warmup_steps": int(getattr(args, 'warmup_steps', 0) or 0),
        "warmup_ratio": float(getattr(args, 'warmup_ratio', 0.0) or 0.0),
        "override_scheduling": bool(getattr(args, 'override_scheduling', False)),
    }

def build_hierarchos_optimizer(model, args, device):
    """RWKV-style AdamW grouping: decay matrices/embeddings, never norms or scalars."""
    lr = args.starting_lr
    weight_decay = float(getattr(args, "rwkv_weight_decay", 0.1))
    decay = []
    no_decay = []
    deepembed_no_decay = []
    val_proj_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name.replace("_orig_mod.", "")
        if clean_name.startswith(("h_deepemb.", "l_deepemb.")):
            # DeepEmbed is initialized to 1.0 and multiplicatively gates the RWKV
            # channel-mix FFN. Decoupled weight decay quietly pulls that identity
            # prior toward zero over long runs, effectively weakening the FFN path.
            no_decay.append(param)
            deepembed_no_decay.append(clean_name)
        elif clean_name.startswith("val_proj.") or ".val_proj." in clean_name:
            # The optional alignment objective makes this the fast-memory
            # encoder. Pulling it toward zero would quietly weaken writes.
            no_decay.append(param)
            val_proj_no_decay.append(clean_name)
        elif (".weight" in name or "emb" in name) and ("ln" not in name and "norm" not in name):
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if deepembed_no_decay:
        print(f"INFO: DeepEmbed weights excluded from AdamW decay ({len(deepembed_no_decay)} tensor(s)).")
    if val_proj_no_decay:
        print(f"INFO: LTM val_proj excluded from AdamW decay ({len(val_proj_no_decay)} tensor(s)).")
    if is_directml_device(device):
        return DirectMLAdamW(param_groups, lr=lr)
    if device.type == 'cuda':
        return torch.optim.AdamW(param_groups, lr=lr, fused=True)
    return torch.optim.AdamW(param_groups, lr=lr)

def estimate_cuda_loss_chunk_rows(free_bytes: int, batch_size: int, chunk_size: int,
                                  vocab_size: int, requested_rows: int = 0) -> int:
    """
    Pick a CUDA lm_head loss chunk size from live free VRAM and current batch shape.

    On 96GB-class GPUs this targets enough rows to cover batch=64 at a
    256-token TBPTT chunk in one loss pass while still reserving most VRAM for
    activations, optimizer state, CUDA graphs, and fragmentation.
    """
    requested_rows = int(requested_rows or 0)
    if requested_rows > 0:
        return requested_rows

    free_bytes = max(0, int(free_bytes or 0))
    batch_size = max(1, int(batch_size or 1))
    chunk_size = max(1, int(chunk_size or 1))
    vocab_size = max(1, int(vocab_size or 1))

    free_gb = free_bytes / float(1024 ** 3)
    if free_gb >= 72.0:
        base_rows = 16834
    elif free_gb >= 48.0:
        base_rows = 12288
    elif free_gb >= 24.0:
        base_rows = 8192
    elif free_gb >= 12.0:
        base_rows = 4096
    else:
        base_rows = 2048

    batch_rows = batch_size * max(1, chunk_size - 1)
    batch_target_rows = int(math.ceil(batch_rows * 1.05))

    # FP32 logits dominate; reserve room for backward/temp buffers and leave most
    # free VRAM for activations, optimizer state, CUDA graphs, and fragmentation.
    estimated_bytes_per_row = vocab_size * 4 * 3
    memory_budget = max(512 * 1024 ** 2, int(free_bytes * 0.20))
    memory_cap_rows = max(512, memory_budget // max(1, estimated_bytes_per_row))

    rows = max(base_rows, min(batch_target_rows, memory_cap_rows))
    rows = min(rows, memory_cap_rows)
    return max(512, int(rows))

def tune_cuda_loss_chunk_rows_once(model, args, batch_size: int, chunk_size: int):
    """Auto-tune CUDA loss chunking once after startup/model allocation."""
    if not (torch.cuda.is_available() and getattr(args, 'cuda_chunked_lm_loss', True)):
        return
    if not getattr(args, '_auto_cuda_loss_chunk_rows', False):
        return

    device = next(model.parameters()).device
    if device.type != 'cuda':
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    vocab_size = int(getattr(model.config, 'vocab_size', getattr(args, 'vocab_size', 1)))
    rows = estimate_cuda_loss_chunk_rows(
        free_bytes=free_bytes,
        batch_size=batch_size,
        chunk_size=chunk_size,
        vocab_size=vocab_size,
    )

    previous = int(getattr(args, 'cuda_loss_chunk_rows', 0) or 0)
    if rows != previous:
        args.cuda_loss_chunk_rows = rows
        if hasattr(model, 'config'):
            model.config.cuda_loss_chunk_rows = rows
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        print(
            f"INFO: Startup CUDA loss chunk rows set to {rows} "
            f"(free VRAM {free_gb:.1f}/{total_gb:.1f} GB, batch={batch_size}, chunk={chunk_size})."
        )

def trim_trailing_padding(input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None):
    """Remove trailing columns that are padding for the entire batch."""
    if attention_mask is None:
        return input_ids, labels, attention_mask
    if not isinstance(attention_mask, torch.Tensor):
        return input_ids, labels, attention_mask
    if input_ids.ndim != 2 or labels.ndim != 2 or attention_mask.ndim != 2:
        return input_ids, labels, attention_mask
    if attention_mask.shape[1] != input_ids.shape[1] or labels.shape[1] != input_ids.shape[1]:
        return input_ids, labels, attention_mask

    active_columns = attention_mask.bool().any(dim=0)
    if not bool(active_columns.any().item()):
        return input_ids, labels, attention_mask
    trim_to = int(active_columns.nonzero(as_tuple=False)[-1].item()) + 1
    if trim_to >= input_ids.shape[1]:
        return input_ids, labels, attention_mask
    return (
        input_ids[:, :trim_to].contiguous(),
        labels[:, :trim_to].contiguous(),
        attention_mask[:, :trim_to].contiguous(),
    )

def pad_training_batch_to_multiple(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor = None,
    multiple: int = 128,
    pad_token_id: int = 0,
):
    """Pad sequence length to a chunk multiple so torch.compile sees stable shapes."""
    multiple = int(multiple or 0)
    if multiple <= 1 or input_ids.ndim != 2 or labels.ndim != 2:
        return input_ids, labels, attention_mask
    T = input_ids.shape[1]
    target_T = int(math.ceil(T / multiple) * multiple)
    pad_cols = target_T - T
    if pad_cols <= 0:
        return input_ids, labels, attention_mask

    ids_pad = input_ids.new_full((input_ids.shape[0], pad_cols), int(pad_token_id))
    label_pad = labels.new_full((labels.shape[0], pad_cols), -100)
    input_ids = torch.cat([input_ids, ids_pad], dim=1).contiguous()
    labels = torch.cat([labels, label_pad], dim=1).contiguous()
    if attention_mask is not None:
        mask_pad = attention_mask.new_zeros((attention_mask.shape[0], pad_cols))
        attention_mask = torch.cat([attention_mask, mask_pad], dim=1).contiguous()
    return input_ids, labels, attention_mask

def set_dataloader_epoch(dataloader, epoch: int):
    """Let length-grouped or distributed samplers reshuffle per epoch."""
    for sampler in (
        getattr(dataloader, "batch_sampler", None),
        getattr(dataloader, "sampler", None),
        getattr(dataloader, "dataset", None),
    ):
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


class CUDABatchPrefetcher:
    """Overlap one pinned host batch transfer with the current CUDA step."""

    def __init__(self, iterable, device):
        self.iterator = iter(iterable)
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self._next = None
        self._preload()

    def __iter__(self):
        return self

    def _preload(self):
        try:
            cpu_batch = next(self.iterator)
        except StopIteration:
            self._next = None
            return

        if cpu_batch is None:
            self._next = (None, {})
            return

        device_tensors = {}
        with torch.cuda.stream(self.stream):
            for key, value in cpu_batch.items():
                if torch.is_tensor(value):
                    device_tensors[key] = value.to(self.device, non_blocking=True)
        self._next = (cpu_batch, device_tensors)

    def __next__(self):
        while self._next is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.stream)
            cpu_batch, device_tensors = self._next
            if cpu_batch is not None:
                for tensor in device_tensors.values():
                    tensor.record_stream(current_stream)
                cpu_batch["_cuda_prefetched_tensors"] = device_tensors
            self._preload()
            if cpu_batch is not None:
                return cpu_batch
        raise StopIteration


def _batch_tensor_to_device(batch, key, cpu_tensor, device, *, pad_value=0):
    """Use a lookahead copy and reproduce any CPU-side trim/pad on device."""
    if cpu_tensor is None:
        return None
    prefetched = batch.get("_cuda_prefetched_tensors")
    candidate = prefetched.get(key) if isinstance(prefetched, dict) else None
    if (
        torch.is_tensor(candidate)
        and candidate.device == device
        and candidate.ndim == cpu_tensor.ndim
        and candidate.shape[0] == cpu_tensor.shape[0]
    ):
        if candidate.shape == cpu_tensor.shape:
            return candidate
        if candidate.ndim == 2 and candidate.shape[1] >= cpu_tensor.shape[1]:
            return candidate[:, :cpu_tensor.shape[1]].contiguous()
        if candidate.ndim == 2 and candidate.shape[1] < cpu_tensor.shape[1]:
            pad_cols = cpu_tensor.shape[1] - candidate.shape[1]
            padding = candidate.new_full(
                (candidate.shape[0], pad_cols),
                pad_value,
            )
            return torch.cat((candidate, padding), dim=1)
    return cpu_tensor.to(device, non_blocking=(device.type == "cuda"))


def should_update_progress(step: int, args, total_steps: int = None, first_step: int = 0) -> bool:
    """Throttle CUDA-to-CPU metric syncs caused by progress-bar scalar logging."""
    interval = max(1, int(getattr(args, 'progress_log_steps', 10) or 1))
    return (
        step == first_step
        or (step + 1) % interval == 0
        or (total_steps is not None and (step + 1) >= int(total_steps))
    )

def _state_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    return value

def _state_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_state_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_state_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _state_to_device(item, device) for key, item in value.items()}
    return value

def normalize_ltm_training_mode(value) -> str:
    mode = str(value or "inner-update").strip().lower().replace("_", "-")
    aliases = {
        "inner": "inner-update",
        "gradient": "inner-update",
        "grad": "inner-update",
        "titans": "inner-update",
        "titans-inner-update": "inner-update",
        "read-only": "read-only",
        "readonly": "read-only",
        "inference": "read-only",
        "inference-like": "read-only",
        "off": "read-only",
        "none": "read-only",
    }
    return aliases.get(mode, "inner-update")

def ltm_inner_updates_enabled(args) -> bool:
    return normalize_ltm_training_mode(getattr(args, "ltm_training_mode", "inner-update")) == "inner-update"


def configure_finetune_ltm_mode(args) -> str:
    """Keep full-sequence LoRA batches isolated from supervised fast-memory writes."""
    mode = normalize_ltm_training_mode(getattr(args, "ltm_training_mode", "inner-update"))
    if mode == "inner-update":
        print(
            "WARNING: LoRA fine-tuning does not use TBPTT chunk boundaries, so a "
            "post-backward LTM write cannot affect the sample that produced it and "
            "would leak into unrelated batches. Using read-only LTM for fine-tuning."
        )
        mode = "read-only"
    args.ltm_training_mode = mode
    return mode

def detach_ltm_state_from_outputs(outputs):
    curr_ltm = outputs.get("ltm_memory_state") if isinstance(outputs, dict) else None
    if curr_ltm is None:
        return None
    return (
        curr_ltm[0].detach() if len(curr_ltm) >= 1 and isinstance(curr_ltm[0], torch.Tensor) else None,
        curr_ltm[1].detach() if len(curr_ltm) >= 2 and isinstance(curr_ltm[1], torch.Tensor) else None,
        curr_ltm[2].detach() if len(curr_ltm) >= 3 and isinstance(curr_ltm[2], torch.Tensor) else None,
        curr_ltm[3] if len(curr_ltm) >= 4 else None,
        curr_ltm[4].detach() if len(curr_ltm) >= 5 and isinstance(curr_ltm[4], torch.Tensor) else None,
        curr_ltm[5].detach() if len(curr_ltm) >= 6 and isinstance(curr_ltm[5], torch.Tensor) else None,
    )

def capture_model_grad_state(model):
    grad_state = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_state[name.replace("_orig_mod.", "")] = param.grad.detach().cpu().clone()
    return grad_state or None

def restore_model_grad_state(model, grad_state, device):
    if not grad_state:
        return False
    issue = _find_first_nonfinite_payload_tensor(grad_state, "grad_state_dict")
    if issue:
        raise RuntimeError(f"Pending accumulation gradients are non-finite and cannot be resumed safely: {issue}")
    restored = 0
    clean_grad_state = {str(k).replace("_orig_mod.", ""): v for k, v in grad_state.items()}
    for name, param in model.named_parameters():
        clean_name = name.replace("_orig_mod.", "")
        grad = clean_grad_state.get(clean_name)
        if grad is None:
            continue
        param.grad = grad.to(device=device, dtype=param.dtype)
        restored += 1
    if restored:
        print(f"INFO: Restored {restored} pending gradient tensor(s) for accumulation parity.")
    return restored > 0

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state

def restore_rng_state(state):
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_all"])
        print("INFO: Restored RNG state from checkpoint.")
    except Exception as exc:
        print(f"Warning: Could not restore RNG state: {exc}")

def _capture_loader_component_state(component):
    if component is None:
        return None
    state = {"class": component.__class__.__name__}
    found = False
    for attr in ("seed", "epoch"):
        if hasattr(component, attr):
            try:
                state[attr] = int(getattr(component, attr))
                found = True
            except Exception:
                pass
    generator = getattr(component, "generator", None)
    if isinstance(generator, torch.Generator):
        try:
            state["generator_state"] = generator.get_state()
            found = True
        except Exception:
            pass
    return state if found else None

def capture_dataloader_state(dataloader):
    if dataloader is None:
        return None
    state = {}
    for name in ("batch_sampler", "sampler", "dataset"):
        component_state = _capture_loader_component_state(getattr(dataloader, name, None))
        if component_state:
            state[name] = component_state
    return state or None

def _restore_loader_component_state(component, state):
    if component is None or not state:
        return
    for attr in ("seed", "epoch"):
        if attr in state and hasattr(component, attr):
            try:
                setattr(component, attr, int(state[attr]))
            except Exception:
                pass
    generator = getattr(component, "generator", None)
    if isinstance(generator, torch.Generator) and "generator_state" in state:
        try:
            generator.set_state(state["generator_state"])
        except Exception:
            pass

def restore_dataloader_state(dataloader, state):
    if dataloader is None or not state:
        return
    for name in ("batch_sampler", "sampler", "dataset"):
        _restore_loader_component_state(getattr(dataloader, name, None), state.get(name))
    print("INFO: Restored dataloader sampler state from checkpoint.")

def build_training_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    dataloader,
    completed_epoch: int,
    mid_epoch_step: int = 0,
    running_states=None,
):
    grad_state = capture_model_grad_state(model)
    checkpoint = {
        "checkpoint_version": 2,
        "checkpoint_kind": "training",
        "completed_epoch": int(completed_epoch),
        "mid_epoch_step": int(mid_epoch_step or 0),
        "model_state_dict": sanitize_model_state_dict(model, reset_transient_ltm=False),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "lr_scheduler_state": capture_main_lr_scheduler_state(args, scheduler),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": dict(model.config),
        "rng_state": capture_rng_state(),
        "data_state": capture_dataloader_state(dataloader),
        "grad_state_dict": grad_state,
        "grad_accumulation_active": bool(grad_state),
        "ltm_scheduler_state": capture_ltm_lr_scheduler_state(args),
        "training_complete": False,
    }
    if running_states is not None:
        checkpoint["running_states"] = _state_to_cpu(running_states)
    return checkpoint

def train_step(model, batch, optimizer, scaler, accumulation_steps, step, args, running_states,
               collect_metrics=True, force_optimizer_step=False, accumulation_divisor=None):
    """Training step with temporal chunking to match original hierarchos.py."""
    args._optimizer_step_was_taken = False
    args._train_step_had_backward = False
    args._train_step_had_nonfinite = False
    device = next(model.parameters()).device
    set_model_training_step(model, getattr(args, '_current_global_step', step))
    _nb = (device.type == 'cuda')  # non_blocking for async CUDA transfer
    full_input_ids = batch['input_ids']
    full_attention_mask = batch.get('attention_mask')
    full_labels = batch['labels']
    full_rosa_ids = batch.get('rosa_ids')
    full_loss_weights = batch.get('loss_weights')
    loss_divisor = max(1, int(accumulation_divisor or accumulation_steps or 1))
    full_input_ids, full_labels, full_attention_mask = trim_trailing_padding(
        full_input_ids, full_labels, full_attention_mask
    )
    if full_rosa_ids is not None:
        full_rosa_ids = full_rosa_ids[:, :full_input_ids.shape[1]].contiguous()
    if full_loss_weights is not None:
        full_loss_weights = full_loss_weights[:, :full_input_ids.shape[1]].contiguous()
    chunk_size = getattr(args, 'training_chunk_size', 128)
    padding_metric_steps = int(getattr(args, 'padding_metric_steps', 0) or 0)
    collect_padding_metrics = (
        collect_metrics
        and bool(getattr(args, 'padding_metrics', True))
        and (padding_metric_steps < 0 or step < padding_metric_steps)
    )
    padding_stats = None
    if collect_padding_metrics:
        pre_static_tokens = int(full_input_ids.numel())
        pre_static_seq_len = int(full_input_ids.shape[1]) if full_input_ids.ndim == 2 else 0
        if isinstance(full_attention_mask, torch.Tensor):
            real_tokens = int(full_attention_mask.sum().item())
        else:
            real_tokens = pre_static_tokens
    if (
        device.type == 'cuda'
        and getattr(args, 'compile', False)
        and getattr(args, 'compile_pad_to_chunk_size', True)
    ):
        pre_pad_seq_len = full_input_ids.shape[1]
        full_input_ids, full_labels, full_attention_mask = pad_training_batch_to_multiple(
            full_input_ids,
            full_labels,
            full_attention_mask,
            multiple=chunk_size,
            pad_token_id=getattr(args, 'pad_token_id', 0),
        )
        if full_loss_weights is not None and full_input_ids.shape[1] > pre_pad_seq_len:
            pad_cols = full_input_ids.shape[1] - pre_pad_seq_len
            weight_pad = full_loss_weights.new_zeros((full_loss_weights.shape[0], pad_cols))
            full_loss_weights = torch.cat([full_loss_weights, weight_pad], dim=1).contiguous()
        if full_rosa_ids is not None and full_input_ids.shape[1] > pre_pad_seq_len:
            pad_cols = full_input_ids.shape[1] - pre_pad_seq_len
            rosa_sentinel = int(getattr(model.config, 'vocab_size', getattr(args, 'vocab_size', 0)))
            rosa_pad = full_rosa_ids.new_full((full_rosa_ids.shape[0], pad_cols), rosa_sentinel)
            full_rosa_ids = torch.cat([full_rosa_ids, rosa_pad], dim=1).contiguous()
    if collect_padding_metrics:
        total_tokens = int(full_input_ids.numel())
        padded_seq_len = int(full_input_ids.shape[1]) if full_input_ids.ndim == 2 else 0
        padding_tokens = max(0, total_tokens - real_tokens)
        padding_stats = {
            "token_efficiency": real_tokens / float(max(1, total_tokens)),
            "padding_fraction": padding_tokens / float(max(1, total_tokens)),
            "bucket_padding_tokens": max(0, pre_static_tokens - real_tokens),
            "compile_padding_tokens": max(0, total_tokens - pre_static_tokens),
            "seq_len": padded_seq_len,
            "pre_static_seq_len": pre_static_seq_len,
        }
    
    # --- [NEW] Track Sequence Poisoning (Parity Fix) ---
    if full_labels.is_floating_point() and torch.isnan(full_labels).any():
        print(f"\nCRITICAL: NaNs detected in labels at step {step}! Skipping batch.")
        optimizer.zero_grad(set_to_none=True)
        return None, running_states
    # --------------------------------------------------
    audit_supervision_coverage_once(args, full_labels, full_attention_mask, step=step)
    
    B, T = full_input_ids.shape
    h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state = running_states
    
    autocast_device = 'cpu' if is_directml_device(device) else device.type
    amp_dtype_str = getattr(args, 'amp_dtype', None) or getattr(model.config if hasattr(model, 'config') else args, 'amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    
    # --- INFINTIE CONTEXT: STATE RECURRENCE ---
    # Default behavior (persist_state=False): carry states ONLY between TBPTT chunks 
    # of the SAME sequence. Cross-batch persistence is disabled by default.
    persist_enabled = getattr(args, 'persist_state', False)
    
    # Safety Check: If batch size changed (e.g. last batch of epoch), we MUST reset
    if persist_enabled and h_state is not None:
        if h_state.shape[0] != B:
            print(f"INFO: Batch size changed from {h_state.shape[0]} to {B}. Resetting states.")
            persist_enabled = False

    if not persist_enabled:
        # Reset states if explicitly disabled or if batch size changed
        h_state = None
        l_state = None
        prev_ctx = None
        target_ctx = None
        drift_state = None
        ltm_state = None
        model.reset_memory() 
    else:
        # If persisting, we detach states to prevent memory issues (TBPTT truncation)
        # but the VALUES are carried forward.
        if h_state is not None: h_state = h_state.detach()
        if l_state is not None: l_state = l_state.detach()
        if prev_ctx is not None: prev_ctx = prev_ctx.detach()
        if target_ctx is not None: target_ctx = target_ctx.detach()
        if drift_state is not None: drift_state = drift_state.detach()
        if ltm_state is not None: 
            ltm_state = tuple(s.detach() if isinstance(s, torch.Tensor) else s for s in ltm_state)

    # Temporal chunking is critical for RWKV-based models. Exact full-sample BPTT
    # uses a single forward when activations are saved normally. When activation
    # checkpointing is enabled it instead builds one attached graph from bounded
    # activation-only temporal segments, carries recurrent states without detaching,
    # and backpropagates only after the last segment. This bounds peak activations
    # while preserving the complete gradient horizon.
    # Build weights before moving labels/masks to CUDA so accounting cannot sync GPU.
    full_sample_bptt = bool(getattr(args, "full_sample_bptt", False))
    if full_sample_bptt:
        if bool(getattr(args, "persist_state", False)):
            raise ValueError(
                "full_sample_bptt requires persist_state=False so unrelated samples "
                "cannot share a graph or recurrent values"
            )
        if bool(getattr(args, "full_sample_activation_checkpointing", False)):
            chunk_size = int(
                getattr(args, "full_sample_checkpoint_segment_size", 128) or 128
            )
        else:
            chunk_size = T
    elif chunk_size <= 0 or chunk_size > T:
        chunk_size = T
    chunk_plan = compute_chunk_training_weights(
        full_labels,
        full_attention_mask,
        chunk_size,
        loss_weights=full_loss_weights,
        h_stride=(
            getattr(model.config, "h_stride", None)
            if full_sample_bptt
            else None
        ),
    )
    num_chunks = len(chunk_plan)

    full_input_ids = _batch_tensor_to_device(
        batch,
        "input_ids",
        full_input_ids,
        device,
        pad_value=int(getattr(args, 'pad_token_id', 0)),
    )
    full_attention_mask = _batch_tensor_to_device(
        batch,
        "attention_mask",
        full_attention_mask,
        device,
        pad_value=0,
    )
    full_labels = _batch_tensor_to_device(
        batch,
        "labels",
        full_labels,
        device,
        pad_value=-100,
    )
    # Compact caches keep labels as int32 through pinned-memory transfer to
    # halve PCIe traffic. Cross-entropy requires int64 targets, so widen only
    # after the tensor reaches the accelerator.
    full_labels = full_labels.to(dtype=torch.long)
    rosa_sentinel = int(getattr(model.config, 'vocab_size', getattr(args, 'vocab_size', 0)))
    full_rosa_ids = _batch_tensor_to_device(
        batch,
        "rosa_ids",
        full_rosa_ids,
        device,
        pad_value=rosa_sentinel,
    )
    full_loss_weights = _batch_tensor_to_device(
        batch,
        "loss_weights",
        full_loss_weights,
        device,
        pad_value=0.0,
    )
    
    # Reporting math is intentionally absent on non-log steps. At the default
    # progress interval this avoids four allocations plus several detached CUDA
    # scalar kernels on 24 out of every 25 batches.
    total_loss = torch.zeros((), device=device, dtype=torch.float32) if collect_metrics else None
    total_ponder = torch.zeros((), device=device, dtype=torch.float32) if collect_metrics else None
    total_commit = torch.zeros((), device=device, dtype=torch.float32) if collect_metrics else None
    total_ltm_value_alignment = (
        torch.zeros((), device=device, dtype=torch.float32) if collect_metrics else None
    )
    has_ponder = False
    has_commitment = False
    has_ltm_value_alignment = False
    chunks_processed = 0
    final_outputs = None
    full_ce_terms = []
    full_ponder_terms = []
    full_commitment_terms = []
    full_ltm_alignment_terms = []
    fast_lm_loss = (
        (device.type == 'cuda' and getattr(args, 'cuda_chunked_lm_loss', True))
        or (device.type == 'cpu' and getattr(args, 'cpu_chunked_lm_loss', True))
    )
    use_ltm_inner_updates = ltm_inner_updates_enabled(args)
    if full_sample_bptt and use_ltm_inner_updates:
        raise RuntimeError(
            "full_sample_bptt requires read-only LTM during the attached graph; "
            "run configure_full_sample_bptt before training"
        )
    ltm_value_alignment_weight = _nonnegative_float(
        getattr(args, 'ltm_value_alignment_weight', 0.0),
        0.0,
    )
    
    try:
        for chunk_idx, chunk_info in enumerate(chunk_plan):
            start_t = chunk_info["start"]
            end_t = chunk_info["end"]
            label_ratio = chunk_info["label_ratio"]
            token_ratio = chunk_info["token_ratio"]
            ponder_ratio = chunk_info.get("ponder_ratio", token_ratio)

            # Dynamic padding can create trailing chunks with no real tokens and
            # no supervised labels. Skip them entirely so padding cannot decay or
            # momentum-step LTM state through a zero-gradient update.
            if label_ratio == 0.0 and token_ratio == 0.0:
                continue
            
            # Slice tensors for this chunk
            input_ids = full_input_ids[:, start_t:end_t]
            attention_mask = full_attention_mask[:, start_t:end_t] if full_attention_mask is not None else None
            loss_end_t = min(end_t + 1, T)
            labels = full_labels[:, start_t:loss_end_t]
            rosa_ids = full_rosa_ids[:, start_t:end_t] if full_rosa_ids is not None else None
            loss_weights = full_loss_weights[:, start_t:loss_end_t] if full_loss_weights is not None else None
            
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=args.amp):
                model_kwargs = dict(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    h_state=h_state, l_state=l_state, prev_context=prev_ctx,
                    target_context=target_ctx, drift_state=drift_state, ltm_memory_state=ltm_state,
                    global_pos_offset=start_t,
                    return_logits=not fast_lm_loss,
                    return_topk_values=False,
                    return_raw_topk_values=use_ltm_inner_updates,
                    return_topk_indices=use_ltm_inner_updates,
                    compute_ltm_value_alignment=ltm_value_alignment_weight > 0.0,
                    rosa_ids=rosa_ids,
                    loss_weights=loss_weights,
                )
                if bool(getattr(args, "full_sample_activation_checkpointing", False)):
                    if not full_sample_bptt:
                        raise RuntimeError(
                            "Full-sample activation checkpointing requires full-sample BPTT"
                        )
                    outputs = _checkpointed_training_model_call(model, model_kwargs)
                else:
                    outputs = model(**model_kwargs)
                
                # LTM fast-memory update needs gradients from the exact tensors used
                # by the forward graph. retrieve_topk already keeps them float32.
                if use_ltm_inner_updates and outputs.get("raw_topk_vals") is not None:
                    for t_val in outputs["raw_topk_vals"]:
                        if t_val.requires_grad:
                            t_val.retain_grad()


                ce_loss = outputs['loss']
                ponder_cost = outputs.get('ponder_cost')
                commitment_cost = outputs.get('commitment_cost')
                ltm_value_alignment_cost = outputs.get('ltm_value_alignment_cost')

                if full_sample_bptt:
                    # Defer loss composition and backward until every attached
                    # segment exists. Aggregating raw components first exactly
                    # matches a single whole-sample forward: caps and adaptive
                    # ponder targets are applied once to global sample means.
                    full_ce_terms.append(ce_loss * label_ratio)
                    if ponder_cost is not None:
                        full_ponder_terms.append(ponder_cost * ponder_ratio)
                    if commitment_cost is not None:
                        full_commitment_terms.append(commitment_cost * token_ratio)
                    if ltm_value_alignment_cost is not None:
                        full_ltm_alignment_terms.append(
                            ltm_value_alignment_cost * token_ratio
                        )
                else:
                    finite_components = [("ce", ce_loss)]
                    if ponder_cost is not None:
                        finite_components.append(("ponder", ponder_cost))
                    if commitment_cost is not None:
                        finite_components.append(("commitment", commitment_cost))
                    if ltm_value_alignment_weight > 0.0:
                        finite_components.append(("alignment", ltm_value_alignment_cost))
                    finite_checks = [
                        torch.isfinite(value).all()
                        if value is not None else torch.tensor(False, device=device)
                        for _name, value in finite_components
                    ]

                    ce_loss_for_backward = _cap_loss_component_for_backward(
                        ce_loss,
                        getattr(args, 'max_ce_loss_for_backward', 0.0),
                    )

                    aux_loss = torch.zeros_like(ce_loss)

                    # --- ACT Sensitivity: Adaptive Ponder Loss ---
                    if ponder_cost is not None:
                        ponder_weight = getattr(args, 'ponder_loss_weight', 0.01)
                        ponder_cost_for_backward = _cap_loss_component_for_backward(
                            ponder_cost,
                            getattr(args, 'max_ponder_cost_for_backward', 0.0),
                        )

                        if getattr(args, 'encourage_thinking', False):
                            # RECOVERY MODE: Invert ponder penalty to REWARD thinking
                            aux_loss = aux_loss - (abs(ponder_weight) * ponder_cost_for_backward)
                        elif getattr(args, 'adaptive_ponder', False):
                            max_h_steps = getattr(args, 'max_h_steps', 5)
                            target_scale = getattr(args, 'ponder_target_scale', 0.5)
                            target_ponder = torch.clamp(
                                ce_loss.detach() * target_scale,
                                min=1.0,
                                max=float(max_h_steps),
                            )
                            ponder_diff = target_ponder - ponder_cost_for_backward
                            aux_loss = aux_loss + torch.relu(ponder_diff) * ponder_weight
                        else:
                            aux_loss = aux_loss + (ponder_weight * ponder_cost_for_backward)

                    if commitment_cost is not None:
                        commitment_cost_for_backward = _cap_loss_component_for_backward(
                            commitment_cost,
                            getattr(args, 'max_commitment_cost_for_backward', 2.0),
                            preserve_gradient=True,
                        )
                        aux_loss = aux_loss + (
                            getattr(args, 'commitment_loss_weight', 0.5)
                            * commitment_cost_for_backward
                        )

                    if ltm_value_alignment_weight > 0.0:
                        aux_loss = aux_loss + (
                            ltm_value_alignment_weight * ltm_value_alignment_cost
                        )

                    # Historical TBPTT semantics: CE is weighted by supervised
                    # labels while token-level auxiliary costs use real tokens.
                    chunk_loss = (
                        (ce_loss_for_backward * label_ratio)
                        + (aux_loss * token_ratio)
                    ) / loss_divisor

                    all_loss_checks = finite_checks + [torch.isfinite(chunk_loss).all()]
                    losses_valid = bool(torch.stack(all_loss_checks).all().item())
                    if not losses_valid:
                        flag_values = torch.stack(all_loss_checks).tolist()
                        component_flags = {
                            name: bool(flag)
                            for (name, _value), flag in zip(finite_components, flag_values)
                        }
                        print(
                            f"\nCRITICAL: Non-finite training loss at step {step+1}, "
                            f"chunk {chunk_idx} ({start_t}:{end_t})."
                        )
                        if not component_flags.get("ce", True) and ce_loss is not None:
                            print("  " + _describe_tensor_issue("cross_entropy_loss", ce_loss))
                        if not component_flags.get("ponder", True) and ponder_cost is not None:
                            print("  " + _describe_tensor_issue("ponder_cost", ponder_cost))
                        if not component_flags.get("commitment", True) and commitment_cost is not None:
                            print("  " + _describe_tensor_issue("commitment_cost", commitment_cost))
                        if not component_flags.get("alignment", True):
                            print("  " + _describe_tensor_issue("ltm_value_alignment_cost", ltm_value_alignment_cost))
                        if not bool(flag_values[-1]):
                            print("  " + _describe_tensor_issue("chunk_loss", chunk_loss))
                        print("  Skipping this batch and clearing recurrent/LTM state before it can poison optimizer state.")
                        args._train_step_had_nonfinite = True
                        return None, _reset_after_nonfinite(optimizer, model)

            if not full_sample_bptt:
                # Historical TBPTT backpropagates each detached temporal chunk.
                if scaler is not None:
                    scaler.scale(chunk_loss).backward()
                else:
                    chunk_loss.backward()
                args._train_step_had_backward = True
            
            # --- GRADIENT-BASED LTM UPDATE (Titans Parity) ---
            if use_ltm_inner_updates and outputs.get("raw_topk_vals") is not None:
                with torch.no_grad():
                    valid_grads = []
                    current_scale = (
                        scaler.get_scale()
                        if getattr(args, 'amp', False) and scaler is not None
                        else None
                    )
                    for t_val in outputs["raw_topk_vals"]:
                        g = t_val.grad
                        if g is not None:
                            # Unscale if using AMP
                            if current_scale is not None:
                                if current_scale > 1e-6:
                                    g = g / current_scale
                            valid_grads.append(g.detach())
                        else:
                            valid_grads.append(torch.zeros_like(t_val))
                    
                    ltm_grads_tensor = torch.stack(valid_grads, dim=1)
                    # Clear intermediate grads immediately to free memory
                    for t_val in outputs["raw_topk_vals"]:
                        t_val.grad = None
                    outputs["raw_topk_vals"] = None  # Free tensor references (BUG #3: memory leak fix)
                    _clip_val = _positive_float(getattr(args, 'grad_clip', 1.0), 1.0)
                    # torch.stack above produced a fresh disposable fp32 tensor;
                    # clip it in place instead of cloning tens of MiB per chunk.
                    ltm_grads_tensor = _prepare_ltm_update_gradients(
                        ltm_grads_tensor,
                        _clip_val,
                        inplace=True,
                    )
                    if ltm_grads_tensor is None:
                        print(
                            f"\nCRITICAL: Non-finite LTM gradient at step {step+1}, "
                            f"chunk {chunk_idx} ({start_t}:{end_t})."
                        )
                        print("  Skipping this batch; poisoned fast-memory updates are never sanitized or applied.")
                        args._train_step_had_nonfinite = True
                        return None, _reset_after_nonfinite(optimizer, model)

                    if ltm_grads_tensor is not None:
                        # Unpack current LTM state for the update
                        curr_ltm = outputs.get('ltm_memory_state')
                        curr_fast = curr_ltm[0] if curr_ltm is not None else None
                        curr_mom = curr_ltm[1] if curr_ltm is not None else None
                        curr_past_tokens = curr_ltm[2] if curr_ltm is not None and len(curr_ltm) >= 3 else None
                        curr_rosa_states = curr_ltm[3] if curr_ltm is not None and len(curr_ltm) >= 4 else None
                        curr_timestamps = curr_ltm[4] if curr_ltm is not None and len(curr_ltm) >= 5 else None
                        curr_sources = curr_ltm[5] if curr_ltm is not None and len(curr_ltm) >= 6 else None
                        
                        # Titans inner_update (Gradient-based)
                        # Pass fast_vals/mom_vals from forward pass state, not module defaults
                        new_fast, new_mom = model.ltm.inner_update(
                            outputs["topk_idx"],
                            ltm_grads_tensor,
                            current_lr=get_current_ltm_lr(args),
                            source=2, # SRC_TRAINING_DATA
                            timestamp=float(end_t),
                            tokens_covered=end_t - start_t,
                            fast_vals=curr_fast,
                            mom_vals=curr_mom,
                            timestamps=curr_timestamps,
                            sources=curr_sources,
                            inplace=True
                        )
                        ltm_state = (new_fast.detach(), new_mom.detach(), 
                                     curr_past_tokens.detach() if curr_past_tokens is not None else None,
                                     curr_rosa_states,
                                     curr_timestamps.detach() if isinstance(curr_timestamps, torch.Tensor) else curr_timestamps,
                                     curr_sources.detach() if isinstance(curr_sources, torch.Tensor) else curr_sources)  # ROSA automaton states (plain Python, no detach needed)
                    else:
                        ltm_state = detach_ltm_state_from_outputs(outputs)
            else:
                # In read-only/inference-like LTM mode, carry ROSA/past-token
                # continuity across TBPTT chunks but do not write supervised fast
                # memory that chat-time generation cannot reproduce.
                if full_sample_bptt:
                    # Keep tensor state attached until the single end-to-end
                    # backward. The current read-only LTM payload is mostly
                    # non-differentiable cache metadata, but avoiding a blanket
                    # detach preserves any trainable routing path added later.
                    ltm_state = outputs.get('ltm_memory_state')
                else:
                    ltm_state = detach_ltm_state_from_outputs(outputs)

            # Carry exact attached states for full BPTT; historical TBPTT detaches
            # at every boundary. Do not carry final_drift into an attached segment:
            # core.forward treats a non-None local-t=0 drift as an external seed,
            # while an uninterrupted absolute-t>0 step derives drift from l_state.
            recurrent_state_clamp = getattr(args, 'recurrent_state_clamp', 50.0)
            context_state_clamp = getattr(args, 'context_state_clamp', 50.0)
            drift_state_clamp = getattr(args, 'drift_state_clamp', 5.0)
            drift_norm_clamp = getattr(args, 'drift_norm_clamp', 0.0)
            if full_sample_bptt:
                h_state = outputs.get('h_state')
                l_state = outputs.get('l_state')
                prev_ctx = outputs.get('prev_context')
                target_ctx = outputs.get('target_context')
                drift_state = outputs.get('drift_state') if chunk_idx == num_chunks - 1 else None
            else:
                if outputs.get('h_state') is not None:
                    h_state = _detach_finite_clamp(outputs['h_state'], recurrent_state_clamp)
                if outputs.get('l_state') is not None:
                    l_state = _detach_finite_clamp(outputs['l_state'], recurrent_state_clamp)
                if outputs.get('prev_context') is not None:
                    prev_ctx = _detach_finite_clamp(outputs['prev_context'], context_state_clamp)
                if outputs.get('target_context') is not None:
                    target_ctx = _detach_finite_clamp(outputs['target_context'], context_state_clamp)
                if outputs.get('drift_state') is not None:
                    drift_state = _detach_finite_l2_clamp(
                        outputs['drift_state'],
                        drift_state_clamp,
                        drift_norm_clamp,
                    )
            
            # Accumulate display-only scalars only when this batch will actually
            # update tqdm; these values never participate in backward or state.
            if collect_metrics:
                total_loss = total_loss + ce_loss.detach().float() * label_ratio
                if ponder_cost is not None:
                    ponder_display_ratio = ponder_ratio if full_sample_bptt else token_ratio
                    total_ponder = (
                        total_ponder
                        + ponder_cost.detach().float() * ponder_display_ratio
                    )
                    has_ponder = True
                if commitment_cost is not None:
                    total_commit = total_commit + commitment_cost.detach().float() * token_ratio
                    has_commitment = True
                if ltm_value_alignment_cost is not None:
                    total_ltm_value_alignment = (
                        total_ltm_value_alignment
                        + ltm_value_alignment_cost.detach().float() * token_ratio
                    )
                    has_ltm_value_alignment = True
            
            chunks_processed += 1
            # final_outputs = outputs # REMOVED: Memory Leak Fix

        if full_sample_bptt and chunks_processed > 0:
            # Reconstruct the exact whole-sample scalar objective before the one
            # backward pass. Each component was normalized by the same mass used
            # in an uninterrupted forward (supervised label weight, manager-step
            # weight, or real-token weight respectively).
            whole_ce_loss = torch.stack(full_ce_terms).sum()
            whole_ponder_cost = (
                torch.stack(full_ponder_terms).sum()
                if full_ponder_terms
                else None
            )
            whole_commitment_cost = (
                torch.stack(full_commitment_terms).sum()
                if full_commitment_terms
                else None
            )
            whole_ltm_alignment_cost = (
                torch.stack(full_ltm_alignment_terms).sum()
                if full_ltm_alignment_terms
                else None
            )

            finite_components = [("ce", whole_ce_loss)]
            if whole_ponder_cost is not None:
                finite_components.append(("ponder", whole_ponder_cost))
            if whole_commitment_cost is not None:
                finite_components.append(("commitment", whole_commitment_cost))
            if ltm_value_alignment_weight > 0.0:
                finite_components.append(("alignment", whole_ltm_alignment_cost))

            ce_loss_for_backward = _cap_loss_component_for_backward(
                whole_ce_loss,
                getattr(args, 'max_ce_loss_for_backward', 0.0),
            )
            aux_loss = torch.zeros_like(whole_ce_loss)

            if whole_ponder_cost is not None:
                ponder_weight = getattr(args, 'ponder_loss_weight', 0.01)
                ponder_cost_for_backward = _cap_loss_component_for_backward(
                    whole_ponder_cost,
                    getattr(args, 'max_ponder_cost_for_backward', 0.0),
                )
                if getattr(args, 'encourage_thinking', False):
                    aux_loss = aux_loss - (
                        abs(ponder_weight) * ponder_cost_for_backward
                    )
                elif getattr(args, 'adaptive_ponder', False):
                    max_h_steps = getattr(args, 'max_h_steps', 5)
                    target_scale = getattr(args, 'ponder_target_scale', 0.5)
                    target_ponder = torch.clamp(
                        whole_ce_loss.detach() * target_scale,
                        min=1.0,
                        max=float(max_h_steps),
                    )
                    aux_loss = aux_loss + (
                        torch.relu(target_ponder - ponder_cost_for_backward)
                        * ponder_weight
                    )
                else:
                    aux_loss = aux_loss + (
                        ponder_weight * ponder_cost_for_backward
                    )

            if whole_commitment_cost is not None:
                commitment_cost_for_backward = _cap_loss_component_for_backward(
                    whole_commitment_cost,
                    getattr(args, 'max_commitment_cost_for_backward', 2.0),
                    preserve_gradient=True,
                )
                aux_loss = aux_loss + (
                    getattr(args, 'commitment_loss_weight', 0.5)
                    * commitment_cost_for_backward
                )

            if ltm_value_alignment_weight > 0.0:
                aux_loss = aux_loss + (
                    ltm_value_alignment_weight * whole_ltm_alignment_cost
                )

            whole_sample_loss = (ce_loss_for_backward + aux_loss) / loss_divisor
            all_loss_checks = [
                torch.isfinite(value).all()
                if value is not None else torch.tensor(False, device=device)
                for _name, value in finite_components
            ] + [torch.isfinite(whole_sample_loss).all()]
            losses_valid = bool(torch.stack(all_loss_checks).all().item())
            if not losses_valid:
                flag_values = torch.stack(all_loss_checks).tolist()
                component_flags = {
                    name: bool(flag)
                    for (name, _value), flag in zip(finite_components, flag_values)
                }
                print(
                    f"\nCRITICAL: Non-finite full-sample training loss at step {step+1}."
                )
                if not component_flags.get("ce", True):
                    print("  " + _describe_tensor_issue("cross_entropy_loss", whole_ce_loss))
                if not component_flags.get("ponder", True):
                    print("  " + _describe_tensor_issue("ponder_cost", whole_ponder_cost))
                if not component_flags.get("commitment", True):
                    print("  " + _describe_tensor_issue("commitment_cost", whole_commitment_cost))
                if not component_flags.get("alignment", True):
                    print("  " + _describe_tensor_issue("ltm_value_alignment_cost", whole_ltm_alignment_cost))
                if not bool(flag_values[-1]):
                    print("  " + _describe_tensor_issue("whole_sample_loss", whole_sample_loss))
                print("  Skipping this batch and clearing recurrent/LTM state before it can poison optimizer state.")
                args._train_step_had_nonfinite = True
                return None, _reset_after_nonfinite(optimizer, model)

            if scaler is not None:
                scaler.scale(whole_sample_loss).backward()
            else:
                whole_sample_loss.backward()
            args._train_step_had_backward = True

            # Backward has consumed the attached segment graph. Return only
            # detached finite terminal state (normally discarded because
            # persist_state is forced off) so no graph can leak into a new sample.
            if h_state is not None:
                h_state = _detach_finite_clamp(h_state, recurrent_state_clamp)
            if l_state is not None:
                l_state = _detach_finite_clamp(l_state, recurrent_state_clamp)
            if prev_ctx is not None:
                prev_ctx = _detach_finite_clamp(prev_ctx, context_state_clamp)
            if target_ctx is not None:
                target_ctx = _detach_finite_clamp(target_ctx, context_state_clamp)
            if drift_state is not None:
                drift_state = _detach_finite_l2_clamp(
                    drift_state,
                    drift_state_clamp,
                    drift_norm_clamp,
                )
            ltm_state = detach_ltm_state_from_outputs(
                {"ltm_memory_state": ltm_state}
            )
            outputs = None
            full_ce_terms.clear()
            full_ponder_terms.clear()
            full_commitment_terms.clear()
            full_ltm_alignment_terms.clear()

        # Optimizer step after all chunks
        should_step_optimizer = ((step + 1) % accumulation_steps == 0) or bool(force_optimizer_step)
        if should_step_optimizer and _has_pending_gradients(model):
            if scaler is not None:
                scaler.unscale_(optimizer)
                grads_ok, grad_issue = _clip_gradients_and_check(
                    model,
                    getattr(args, 'grad_clip', 1.0),
                    getattr(args, 'max_sanitized_gradient_values', 0),
                )
                if not grads_ok:
                    print(f"\nCRITICAL: Non-finite gradient at step {step+1}. {grad_issue}")
                    print("  Skipping optimizer step and clearing accumulated gradients.")
                    args._train_step_had_nonfinite = True
                    return None, _reset_after_nonfinite(optimizer, model)
                scaler.step(optimizer)
                scaler.update()
            else:
                grads_ok, grad_issue = _clip_gradients_and_check(
                    model,
                    getattr(args, 'grad_clip', 1.0),
                    getattr(args, 'max_sanitized_gradient_values', 0),
                )
                if not grads_ok:
                    print(f"\nCRITICAL: Non-finite gradient at step {step+1}. {grad_issue}")
                    print("  Skipping optimizer step and clearing accumulated gradients.")
                    args._train_step_had_nonfinite = True
                    return None, _reset_after_nonfinite(optimizer, model)
                optimizer.step()
            if ltm_value_alignment_weight > 0.0:
                mark_val_proj_trained(model)
            optimizer.zero_grad(set_to_none=True)
            args._optimizer_step_was_taken = True
        elif should_step_optimizer:
            optimizer.zero_grad(set_to_none=True)
        
        if chunks_processed == 0:
            return None, running_states
            
        avg_outputs = None
        if collect_metrics:
            # Keep scalars on-device until the throttled progress update calls
            # .item(); doing this every batch forces a CUDA sync on fast GPUs.
            avg_outputs = {
                'loss': total_loss.detach(),
                'ponder_cost': total_ponder.detach() if has_ponder else None,
                'commitment_cost': total_commit.detach() if has_commitment else None,
                'ltm_value_alignment_cost': (
                    total_ltm_value_alignment.detach()
                    if has_ltm_value_alignment
                    else None
                ),
            }
            if padding_stats is not None:
                avg_outputs.update(padding_stats)
        
        next_states = (h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state)
        return avg_outputs, next_states
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "reset_memory"):
                model.reset_memory()
            torch.cuda.empty_cache()
            if bool(getattr(args, "full_sample_bptt", False)):
                raise RuntimeError(
                    "Full-sample BPTT ran out of memory even with its configured "
                    "activation policy. Refusing to silently truncate the gradient. "
                    "First lower --full-sample-checkpoint-segment-size (this preserves "
                    "the objective and outer batch); otherwise lower --batch-size with "
                    "matching accumulation, shorten --max_length, or enable full-sample "
                    "activation checkpointing."
                ) from e
            print("WARNING: OOM detected. Clearing cache and skipping the batch.")
            return None, running_states
        raise e


def configure_cuda_training_runtime(args, config, device, *, mode_name="training"):
    """Apply the shared throughput-safe CUDA policy to pretraining and LoRA."""
    if getattr(device, "type", None) != "cuda":
        return False

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    gpu_capability = torch.cuda.get_device_capability(device)
    print(
        f"INFO: CUDA GPU: {gpu_name} "
        f"({gpu_mem:.1f} GB, SM {gpu_capability[0]}.{gpu_capability[1]})"
    )

    if gpu_capability[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"INFO: TF32 matmul enabled for CUDA {mode_name}.")
    torch.backends.cudnn.benchmark = True

    amp_was_explicitly_set = any(
        value in sys.argv for value in ("--amp", "--no-amp", "--no_amp")
    )
    if not amp_was_explicitly_set:
        args.amp = True
        config.amp = True
        print(f"INFO: AMP auto-enabled for CUDA {mode_name} (use --no-amp to disable).")

    compile_was_explicitly_disabled = any(
        value in sys.argv for value in ("--no-compile", "--no_compile")
    )
    if compile_was_explicitly_disabled:
        args.compile = False
        config.compile = False
        print("INFO: torch.compile explicitly disabled.")
    elif not getattr(args, "compile", False) and not getattr(args, "force_compile", False):
        args.compile = True
        config.compile = True
        print(f"INFO: torch.compile auto-enabled for CUDA {mode_name}.")

    # Core.compile() already selects the fixed masked WorkerLoop for CUDA when
    # this option is auto. Persist that decision in exported config so parity
    # inference can execute the same refinement dynamics instead of a distinct
    # eval-only early-exit loop.
    if (
        (getattr(args, "compile", False) or getattr(args, "force_compile", False))
        and getattr(args, "compile_static_worker_loop", None) is None
    ):
        args.compile_static_worker_loop = True
        config.compile_static_worker_loop = True
        print(f"INFO: Fixed masked WorkerLoop enabled for CUDA {mode_name} parity.")

    if gpu_capability[0] >= 8 and torch.cuda.is_bf16_supported():
        args.amp_dtype = "bfloat16"
        config.amp_dtype = "bfloat16"
        print(f"INFO: Using bfloat16 AMP for CUDA {mode_name}.")
    else:
        args.amp_dtype = "float16"
        config.amp_dtype = "float16"
        print(f"INFO: Using float16 AMP with GradScaler for CUDA {mode_name}.")
    return compile_was_explicitly_disabled


def train(args, device, tokenizer, dataloader, dataloader_len, model_override=None):
    print("Running in TRAIN mode...")
    if dataloader_len <= 0:
        print("ERROR: dataloader_len must be > 0. If automatic detection failed, please specify --dataset-size.")
        return
    
    
    if getattr(args, 'out_dir', None):
        os.makedirs(args.out_dir, exist_ok=True)
    config = AttrDict(vars(args))
    configure_full_sample_bptt(args, config, announce=True)
    _apply_runtime_model_config_overrides(config, args)
    args.ltm_training_mode = normalize_ltm_training_mode(getattr(args, "ltm_training_mode", "inner-update"))
    config.ltm_training_mode = args.ltm_training_mode

    prompt_completion_mode = (
        getattr(args, 'alpaca', False)
        or getattr(args, 'kayla', False)
        or bool(getattr(args, 'prompt_column', None))
        or bool(getattr(args, 'completion_column', None))
    )
    if prompt_completion_mode and not getattr(args, 'train_prompt_tokens', True):
        print(
            "WARNING: Prompt/instruction tokens are masked from CE loss. With TBPTT "
            f"chunks of {getattr(args, 'training_chunk_size', 128)} and recurrent detach "
            f"every {getattr(args, 'detach_every_n_steps', 32)} tokens, long prompt-only "
            "chunks may not receive semantic answer-loss gradients. Use "
            "--train-prompt-tokens if you want prompt tokens included in the LM objective."
        )
    if prompt_completion_mode:
        if getattr(args, 'alpaca', False):
            prompt_shape = "Alpaca [optional ### Previous Context]/### Instruction/### Response"
        elif getattr(args, 'kayla', False):
            prompt_shape = "Kayla Instruction/Feelings/Thought Process/Response"
        else:
            prompt_shape = "User/Assistant prompt-completion"
        prompt_weight = float(getattr(args, 'prompt_loss_weight', 1.0) or 0.0)
        response_weight = float(getattr(args, 'response_loss_weight', 1.0) or 0.0)
        boundary_weight = float(getattr(args, 'response_boundary_loss_weight', 1.0) or 0.0)
        boundary_tokens = int(getattr(args, 'response_boundary_tokens', 0) or 0)
        if not getattr(args, 'train_prompt_tokens', True):
            prompt_weight = 0.0
        print(f"INFO: Training prompt shape: {prompt_shape}")
        print(
            "INFO: Token loss weights: "
            f"prompt={prompt_weight:g}, response={response_weight:g}, "
            f"response_boundary={boundary_weight:g}x first {boundary_tokens} non-EOS response token(s)"
        )
        print(
            "INFO: Response data guardrails: "
            f"drop_empty_completions={bool(getattr(args, 'drop_empty_completions', True))}, "
            f"min_response_tokens={int(getattr(args, 'min_response_tokens', 1) or 0)}"
        )
        if bool(getattr(args, "assistant_recovery", False)):
            print("INFO: Assistant recovery preset ACTIVE.")
    if ltm_inner_updates_enabled(args):
        print("INFO: Training LTM mode: inner-update (supervised gradient fast-memory updates between TBPTT chunks).")
    else:
        print(
            "INFO: Training LTM mode: read-only/inference-like "
            "(ROSA/history state carries across chunks; supervised LTM fast-memory writes are disabled)."
        )

    # Device stability
    if is_directml_device(device):
        args.compile = False
        args.amp = False
        config.compile = False
        config.amp = False

    _compile_was_explicitly_disabled = any(a in sys.argv for a in ('--no-compile', '--no_compile'))
    if getattr(args, 'force_compile', False) and _compile_was_explicitly_disabled:
        args.force_compile = False
        args.compile = False
        config.compile = False
        print("WARNING: Both --force-compile and --no-compile were set; using --no-compile.")
    elif getattr(args, 'force_compile', False):
        args.compile = True
        config.compile = True

    # =================================================================
    # CUDA DATACENTER OPTIMIZATIONS
    # =================================================================
    if device.type == 'cuda':
        _compile_was_explicitly_disabled = configure_cuda_training_runtime(
            args,
            config,
            device,
            mode_name="training",
        )

        # TF32 matmul (Ampere+, SM >= 8.0) — 3-8x faster matmuls with negligible accuracy loss
        # Common TF32/cuDNN setup was applied by configure_cuda_training_runtime.

        # cuDNN benchmark — auto-tunes convolution algorithms for the hardware
        

        # Auto-enable AMP on CUDA unless the user explicitly passed --amp or --no-amp.
        # Must check all argparse-accepted forms (hyphen and underscore variants).
        

        # Auto-enable torch.compile on CUDA unless the user explicitly disables it.
        

        # Prefer bfloat16 on Ampere+ for better dynamic range (no GradScaler needed)
        

        if getattr(args, 'cuda_chunked_lm_loss', True):
            loss_chunk_rows = int(getattr(args, 'cuda_loss_chunk_rows', 0) or 0)
            if loss_chunk_rows <= 0:
                args._auto_cuda_loss_chunk_rows = True
                config.cuda_loss_chunk_rows = 0
                print("INFO: CUDA chunked LM loss enabled (startup auto rows from free VRAM and batch shape).")
            else:
                args._auto_cuda_loss_chunk_rows = False
                config.cuda_loss_chunk_rows = loss_chunk_rows
                print(f"INFO: CUDA chunked LM loss enabled ({loss_chunk_rows} fixed rows/chunk, logits omitted in train_step).")
            config.cuda_loss_chunk_rows = loss_chunk_rows
            config.cuda_chunked_lm_loss = True
        else:
            args._auto_cuda_loss_chunk_rows = False
            config.cuda_chunked_lm_loss = False

        # Multi-GPU info
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"INFO: {num_gpus} GPUs detected. For multi-GPU training, wrap with DistributedDataParallel.")
    # =================================================================

    model = None
    optimizer = None
    start_epoch = 0
    start_step = 0
    scaler = None
    scheduler = None
    checkpoint = None
    # NOTE: use_amp MUST be read AFTER the CUDA block above, which may auto-enable AMP.
    use_amp = getattr(args, 'amp', False)
    epoch_offset = (
        int(getattr(args, 'base_completed_epoch', 0) or 0)
        if getattr(args, 'model_path', None) and not getattr(args, 'resume_from_ckpt', None)
        else 0
    )
    
    # 1. Loading/Resuming Logic
    if model_override is not None:
        if getattr(args, "resume_from_ckpt", None):
            raise ValueError(
                "model_override cannot be combined with resume_from_ckpt; use the "
                "checkpoint loader for an exact same-run resume"
            )
        print("Using preloaded model as a weights-only training base.")
        model = model_override.to(device)
        loaded_config = dict(getattr(model, "config", {}) or {})
        runtime_config = AttrDict(loaded_config)
        runtime_config.update(dict(config))
        for key in (
            'vocab_size',
            'context_dim',
            'persistent_dim',
            'ltm_slots',
            'ltm_key_dim',
            'ltm_val_dim',
            'ltm_topk',
            'h_hidden',
            'l_hidden',
            'h_stride',
            'max_h_steps',
            'max_l_steps',
            'use_deepembed',
            'use_rosa',
            'memory_token_routers',
            'rosa_max_context',
            'rwkv_head_size',
        ):
            if key in loaded_config:
                runtime_config[key] = loaded_config[key]
        model_config = runtime_config
        model_config.compile = args.compile
        _apply_runtime_model_config_overrides(model_config, args)
        model.config = model_config
        optimizer = build_hierarchos_optimizer(model, args, device)
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16':
            scaler = GradScaler()

    elif args.resume_from_ckpt:
        print(f"Resuming from checkpoint: {args.resume_from_ckpt}")
        checkpoint = load_checkpoint_payload_compatible(
            args.resume_from_ckpt,
            map_location='cpu',
        )
        
        saved_config = checkpoint.get('config', {})
        model_config = AttrDict(saved_config)
        state_dict = sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False)
        _reject_unsupported_rwkv_state_dict(state_dict, args.resume_from_ckpt)
        checkpoint['model_state_dict'] = state_dict
        _infer_arch_flags_from_state_dict(model_config, state_dict)
        
        # ARCH Detection (Safely handling compiled checkpoints with '_orig_mod.' prefix)
        state_dict_keys = set()
        for k in state_dict.keys():
            clean_k = k.replace('_orig_mod.', '')
            state_dict_keys.add(clean_k)
            
        # 1. Detect vocab_size and context_dim from tok_emb/lm_head
        if 'tok_emb.weight' in state_dict_keys:
            key = 'tok_emb.weight' if 'tok_emb.weight' in state_dict else '_orig_mod.tok_emb.weight'
            model_config.vocab_size = state_dict[key].shape[0]
            model_config.context_dim = state_dict[key].shape[1]
        elif 'lm_head.weight' in state_dict_keys:
            key = 'lm_head.weight' if 'lm_head.weight' in state_dict else '_orig_mod.lm_head.weight'
            model_config.vocab_size = state_dict[key].shape[0]
            model_config.context_dim = state_dict[key].shape[1]
        
        # 2. Detect persistent_dim
        if 'persistent' in state_dict_keys:
            key = 'persistent' if 'persistent' in state_dict else '_orig_mod.persistent'
            model_config.persistent_dim = state_dict[key].shape[0]
        
        # 3. Detect LTM dims
        if 'val_proj.weight' in state_dict_keys:
            key = 'val_proj.weight' if 'val_proj.weight' in state_dict else '_orig_mod.val_proj.weight'
            model_config.ltm_val_dim = state_dict[key].shape[0]
        if 'qproj.weight' in state_dict_keys:
            key = 'qproj.weight' if 'qproj.weight' in state_dict else '_orig_mod.qproj.weight'
            model_config.ltm_key_dim = state_dict[key].shape[0]

        # 4. Detect RNN hidden sizes and RWKV matrix-state head geometry.
        if 'h_rnn.key.weight' in state_dict_keys:
            model_config.h_hidden = state_dict['h_rnn.key.weight'].shape[0]
        if 'l_rnn.key.weight' in state_dict_keys:
            model_config.l_hidden = state_dict['l_rnn.key.weight'].shape[0]
        if 'h_rnn.r_k' in state_dict_keys:
            model_config.rwkv_head_size = state_dict['h_rnn.r_k'].shape[1]

        # ARCH defaults / Fallbacks
        arch_defaults = {
            'ltm_slots': 1024, 'ltm_key_dim': 128, 'ltm_val_dim': 128, 'ltm_topk': 4,
            'h_stride': 4, 'max_h_steps': 5, 'max_l_steps': 5,
            'h_hidden': model_config.get('context_dim', 448),
            'l_hidden': model_config.get('context_dim', 448),
            'rwkv_head_size': getattr(args, 'rwkv_head_size', None),
        }
        for k, v in arch_defaults.items():
            if k not in model_config:
                model_config[k] = getattr(args, k, v) if hasattr(args, k) else v

        # Runtime overrides
        model_config.compile = args.compile
        model_config.max_length = args.max_length or model_config.get('max_length', 1024)
        _apply_runtime_model_config_overrides(model_config, args)

        print(f"INFO: Final Adjusted ARCH: context_dim={model_config.context_dim}, persistent={model_config.get('persistent_dim', 128)}, ltm_val={model_config.get('ltm_val_dim', 128)}, h_hidden={model_config.h_hidden}, l_hidden={model_config.l_hidden}, vocab_size={model_config.vocab_size}")
        
        model = HierarchosCore(model_config).to(device)
        
        load_model_state_dict_compatible(model, state_dict, args.resume_from_ckpt)
        print(f"INFO: Model state_dict loaded coherently ({len(state_dict)} tensors).")
        
        # --- SURGICAL FIX: Reset h_halt_proj.bias to encourage pondering ---
        reset_bias = getattr(args, 'reset_halt_bias', None)
        if reset_bias is not None:
            with torch.no_grad():
                if hasattr(model, 'h_halt_proj') and model.h_halt_proj.bias is not None:
                    old_bias = model.h_halt_proj.bias.item()
                    model.h_halt_proj.bias.fill_(reset_bias)
                    print(f"INFO: SURGICAL FIX - Reset h_halt_proj.bias from {old_bias:.4f} to {reset_bias:.4f}")
                    print(f"      Initial halt probability: {torch.sigmoid(torch.tensor(reset_bias)).item():.2%}")
                else:
                    print("WARNING: h_halt_proj.bias not found, surgical fix skipped.")
        
        optimizer = build_hierarchos_optimizer(model, args, device)
        
        if not getattr(args, 'override_scheduling', False):
            _restore_resume_component_state(
                optimizer,
                checkpoint.get('optimizer_state_dict'),
                "optimizer",
                args.resume_from_ckpt,
            )
        
        # Original script uses 'completed_epoch', modular uses 'epoch', check both for compatibility.
        start_epoch = int(checkpoint.get('completed_epoch', checkpoint.get('epoch', 0)) or 0)
        start_step = int(checkpoint.get('mid_epoch_step', 0) or 0)
        if start_step >= dataloader_len:
            start_epoch += 1
            start_step = 0
        if start_epoch >= int(getattr(args, 'epochs', 0) or 0):
            raise ValueError(
                f"Checkpoint is already at completed_epoch={start_epoch}, but --epochs={args.epochs}. "
                "--epochs is the total target epoch for --resume-from-ckpt; use a larger value "
                "(for example 14 after an epoch-11 checkpoint for three more epochs), or use "
                "--model-path for a three-epoch continuation from an inference export."
            )
        if start_step > 0:
            print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}, step {start_step}.")
        else:
            print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}.")
        # BFloat16 does NOT use GradScaler — its dynamic range makes scaling unnecessary.
        # Only create scaler for float16 AMP.
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16':
            scaler = GradScaler()
            if not getattr(args, 'override_scheduling', False):
                _restore_resume_component_state(
                    scaler,
                    checkpoint.get('scaler_state_dict'),
                    "AMP scaler",
                    args.resume_from_ckpt,
                )
    
    elif args.model_path:
        print(f"Loading base model from: {args.model_path}")
        model, model_config = load_full_model_with_config(args.model_path, device)
        loaded_config = dict(model_config)
        runtime_config = AttrDict(loaded_config)
        runtime_config.update(dict(config))
        for key in (
            'vocab_size',
            'context_dim',
            'persistent_dim',
            'ltm_slots',
            'ltm_key_dim',
            'ltm_val_dim',
            'ltm_topk',
            'h_hidden',
            'l_hidden',
            'h_stride',
            'max_h_steps',
            'max_l_steps',
            'use_deepembed',
            'use_rosa',
            'memory_token_routers',
            'rosa_max_context',
            'rwkv_head_size',
        ):
            if key in loaded_config:
                runtime_config[key] = loaded_config[key]
        model_config = runtime_config
        model_config.compile = args.compile
        _apply_runtime_model_config_overrides(model_config, args)
        model.config = model_config
        
        optimizer = build_hierarchos_optimizer(model, args, device)
        
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16': scaler = GradScaler()
    
    else:
        print("Starting training from scratch.")
        if 'vocab_size' not in config: config.vocab_size = len(tokenizer)
        model = HierarchosCore(config).to(device)
        optimizer = build_hierarchos_optimizer(model, args, device)
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16': scaler = GradScaler()

    model_vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if model_vocab > 0 and len(tokenizer) != model_vocab:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary ({model_vocab}). "
            "Use the exact tokenizer from the checkpoint's original training run."
        )

    # --- [NEW] Sync LTM reference chunk size (Parity Fix) ---
    training_chunk_size = getattr(args, 'training_chunk_size', 128)
    if hasattr(model, 'ltm'):
        if not hasattr(model.ltm, 'reference_chunk_len'):
            model.ltm.reference_chunk_len = training_chunk_size
        
        if model.ltm.reference_chunk_len != training_chunk_size:
            print(f"INFO: Updating LTM reference chunk length from {model.ltm.reference_chunk_len} to {training_chunk_size}")
            model.ltm.reference_chunk_len = training_chunk_size
        model.ltm.cpu_gather_retrieval = bool(getattr(args, 'ltm_cpu_gather_retrieval', True))
        model.ltm.cpu_sparse_update = bool(getattr(args, 'ltm_cpu_sparse_update', True))
    if hasattr(model, 'config'):
        _apply_runtime_model_config_overrides(model.config, args)
        model.config.cpu_chunked_lm_loss = bool(getattr(args, 'cpu_chunked_lm_loss', True))
        model.config.cpu_loss_chunk_rows = int(getattr(args, 'cpu_loss_chunk_rows', 0) or 0)
        if hasattr(model, "refresh_runtime_config"):
            model.refresh_runtime_config()
        _print_runtime_stability_config(model)
    # ----------------------------------------------------

    # --- Print Model Stats ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimate file size: params * 4 bytes (float32) + ~10% overhead
    estimated_bytes = total_params * 4 * 1.1
    if estimated_bytes >= 1e9:
        size_str = f"{estimated_bytes / 1e9:.2f} GB"
    else:
        size_str = f"{estimated_bytes / 1e6:.2f} MB"
    print(f"INFO: Model Parameters: {total_params:,} total ({trainable_params:,} trainable)")
    print(f"INFO: Estimated checkpoint size: ~{size_str}")
    # --------------------------

    # Compile
    model.compile()
    tune_cuda_loss_chunk_rows_once(
        model,
        args,
        batch_size=getattr(args, 'batch_size', 1),
        chunk_size=getattr(args, 'training_chunk_size', 128),
    )

    # Scheduler
    # When override_scheduling is used during resume, calculate T_max based on REMAINING epochs
    # so that the LR decays properly to min_lr by the final epoch.
    full_run_update_steps = compute_update_steps(dataloader_len, args.accumulation_steps) * args.epochs
    saved_main_lr_state = checkpoint.get('lr_scheduler_state') if isinstance(checkpoint, dict) else None
    saved_ltm_lr_state = checkpoint.get('ltm_scheduler_state') if isinstance(checkpoint, dict) else None
    if getattr(args, 'override_scheduling', False) and args.resume_from_ckpt:
        num_update_steps = compute_remaining_update_steps(
            dataloader_len,
            args.accumulation_steps,
            start_epoch,
            args.epochs,
            start_step,
        )
        print(f"INFO: --override-scheduling: Calculating LR schedule for remaining work ({num_update_steps} update steps)")
    elif (
        args.resume_from_ckpt
        and isinstance(saved_main_lr_state, dict)
        and saved_main_lr_state.get("enabled", True)
        and int(saved_main_lr_state.get("total_steps", 0) or 0) > 0
    ):
        num_update_steps = max(1, int(saved_main_lr_state.get("total_steps", 1) or 1))
        print(f"INFO: Continuing saved main LR schedule ({num_update_steps} total updates).")
    elif (
        args.resume_from_ckpt
        and isinstance(saved_ltm_lr_state, dict)
        and int(saved_ltm_lr_state.get("total_steps", 0) or 0) > 0
        and int(saved_ltm_lr_state.get("total_steps", 0) or 0) != full_run_update_steps
    ):
        num_update_steps = max(1, int(saved_ltm_lr_state.get("total_steps", 1) or 1))
        print(
            "INFO: Main LR scheduler metadata missing; using saved LTM remaining-work "
            f"schedule length as fallback ({num_update_steps} total updates)."
        )
    else:
        num_update_steps = full_run_update_steps
        if args.resume_from_ckpt and not getattr(args, 'override_scheduling', False):
            print(
                "INFO: Resume scheduler using checkpoint/full-run schedule state. "
                "Pass --override-scheduling to rebuild LR decay over only the remaining work."
            )
    args._main_lr_schedule_total_steps = int(num_update_steps)
    
    if not getattr(args, 'disable_lr_schedule', False) and num_update_steps > 0:
        scheduler = build_lr_scheduler(optimizer, args, num_update_steps)
        if args.resume_from_ckpt and not getattr(args, 'override_scheduling', False):
            _restore_resume_component_state(
                scheduler,
                checkpoint.get('scheduler_state_dict'),
                "main LR scheduler",
                args.resume_from_ckpt,
            )
    configure_ltm_lr_schedule(
        args,
        num_update_steps,
        checkpoint=checkpoint,
        override_schedule=bool(getattr(args, 'override_scheduling', False)),
        scheduler=scheduler,
    )
    if hasattr(model, 'config'):
        model.config.ltm_lr = getattr(args, 'ltm_lr', getattr(model.config, 'ltm_lr', 1e-3))
        model.config.min_ltm_lr = getattr(args, 'min_ltm_lr', getattr(model.config, 'min_ltm_lr', None))
        model.config.disable_ltm_lr_schedule = getattr(args, 'disable_ltm_lr_schedule', False)
    if checkpoint:
        restore_dataloader_state(dataloader, checkpoint.get('data_state'))
        restore_rng_state(checkpoint.get('rng_state'))
        if getattr(args, 'override_scheduling', False):
            if checkpoint.get('grad_state_dict'):
                print(
                    "INFO: Ignoring pending gradient accumulation from checkpoint "
                    "because --override-scheduling requested a fresh optimizer/scheduler path."
                )
        else:
            restore_model_grad_state(model, checkpoint.get('grad_state_dict'), device)
        if checkpoint.get('running_states') is not None:
            running_issue = _find_first_nonfinite_payload_tensor(checkpoint['running_states'], "running_states")
            if running_issue:
                checkpoint['running_states'] = None
                print(
                    f"WARNING: Discarding non-finite transient running state before resume: {running_issue}. "
                    "Learned weights and optimizer state were not modified."
                )
        if start_step > 0:
            if checkpoint.get('data_state') is None:
                print("Warning: Mid-epoch checkpoint has no dataloader state; resume will skip to the saved step but exact batch order may differ.")
            if checkpoint.get('rng_state') is None:
                print("Warning: Mid-epoch checkpoint has no RNG state; stochastic components may not be exactly replayed.")
            if getattr(args, 'persist_state', False) and 'running_states' not in checkpoint:
                print("Warning: Mid-epoch checkpoint has no running RWKV/LTM states while --persist-state is enabled; continuing with reset states.")

    startup_issue = _find_first_nonfinite_model_tensor(model, include_grads=True)
    if startup_issue:
        raise RuntimeError(f"Non-finite startup model/gradient state is not recoverable safely: {startup_issue}")
    optimizer_issue = _find_first_nonfinite_optimizer_tensor(optimizer)
    if optimizer_issue:
        raise RuntimeError(f"Non-finite startup optimizer state is not recoverable safely: {optimizer_issue}")
    _clamp_model_finite_magnitude_(
        model,
        getattr(args, 'startup_weight_max_abs', 0.0),
        log_prefix="startup model",
    )
    _sanitize_model_transient_state_(model, max_abs=getattr(args, 'grad_clip', 1.0))
    # --- Evaluation Confirmation ---
    eval_tasks = getattr(args, 'eval_tasks', None)
    if eval_tasks:
        eval_every = getattr(args, 'eval_every_epoch', 1)
        print(f"INFO: Evaluation ENABLED - will run {eval_tasks} every {eval_every} epoch(s)")
    
    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        absolute_epoch = epoch_offset + epoch
        model.train()
        set_dataloader_epoch(dataloader, absolute_epoch)
        if epoch_offset:
            epoch_desc = f"Epoch {absolute_epoch+1} ({epoch+1}/{args.epochs} session)"
        else:
            epoch_desc = f"Epoch {epoch+1}/{args.epochs}"
        pbar = tqdm(dataloader, desc=epoch_desc, total=dataloader_len)
        batch_source = (
            CUDABatchPrefetcher(pbar, device)
            if device.type == 'cuda' and bool(getattr(args, 'cuda_prefetch', True))
            else pbar
        )
        
        # Restore recurrent/LTM states only for true mid-epoch checkpoints.
        if epoch == start_epoch and start_step > 0 and checkpoint and 'running_states' in checkpoint:
            running_states = _state_to_device(
                _clamp_running_states_for_resume(checkpoint['running_states'], args),
                device,
            )
            print(f"INFO: Restored RNN/LTM running states from checkpoint on {device}.")
        else:
            running_states = (None, None, None, None, None, None)
        
        for step, batch in enumerate(batch_source):
            # Mid-Epoch Resumption: Skip steps already processed
            if epoch == start_epoch and step < start_step:
                if step == start_step - 1: # Print only once when we are about to start processing
                    print(f"INFO: Resuming from mid-epoch step {start_step}...")
                continue

            if batch is None:
                continue
            
            # --- FIXED: Sequence-Level State Reset ---
            # If not persisting across batches, we must start each sequence with a clean slate.
            # Local sequence context is still preserved via trainer.train_step's chunk loop.
            if not getattr(args, 'persist_state', False):
                running_states = (None, None, None, None, None, None)

            first_logged_step = start_step if epoch == start_epoch else 0
            collect_metrics = should_update_progress(step, args, dataloader_len, first_logged_step)
            args._current_global_step = absolute_epoch * dataloader_len + step
            force_optimizer_step = should_step_accumulation(step, dataloader_len, args.accumulation_steps)
            accumulation_divisor = accumulation_divisor_for_step(step, dataloader_len, args.accumulation_steps)
            outputs, running_states = train_step(
                model,
                batch,
                optimizer,
                scaler,
                args.accumulation_steps,
                step,
                args,
                running_states,
                collect_metrics=collect_metrics,
                force_optimizer_step=force_optimizer_step,
                accumulation_divisor=accumulation_divisor,
            )
            if outputs:
                postfix = {"loss": f"{outputs['loss'].item():.4f}"}
                if outputs.get('ponder_cost') is not None:
                    postfix["ponder"] = f"{outputs['ponder_cost'].item():.2f}"
                if outputs.get('commitment_cost') is not None:
                    postfix["commit"] = f"{outputs['commitment_cost'].item():.2e}"
                if outputs.get('token_efficiency') is not None:
                    postfix["tok_eff"] = f"{outputs['token_efficiency'] * 100.0:.1f}%"
                    postfix["seq"] = int(outputs.get('seq_len', 0) or 0)
                if scheduler:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                postfix["ltm_lr"] = f"{get_current_ltm_lr(args):.2e}" if ltm_inner_updates_enabled(args) else "off"
                pbar.set_postfix(postfix)

            if scheduler and getattr(args, '_optimizer_step_was_taken', False):
                scheduler.step()
            if getattr(args, '_optimizer_step_was_taken', False) and ltm_inner_updates_enabled(args):
                advance_ltm_lr_schedule(args)

            # Periodic Checkpointing (Progress Protection)
            if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
                print(f"\n[Step {step+1}] Periodic Checkpoint: Saving to {args.out_dir}...")
                save_training_checkpoint_if_finite(
                    build_training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        dataloader,
                        completed_epoch=absolute_epoch,
                        mid_epoch_step=step + 1,
                        running_states=running_states,
                    ),
                    os.path.join(args.out_dir, f"hierarchos_epoch_{absolute_epoch+1}_step_{step+1}.pt"),
                    model,
                    optimizer,
                )
            
            # --- STEP-BASED EVALUATION (runs every N steps) ---
            eval_steps = getattr(args, 'eval_steps', None)
            if eval_steps and eval_tasks and (step + 1) % eval_steps == 0:
                try:
                    from ..evaluation import run_eval, format_results, is_lm_eval_available
                    
                    if is_lm_eval_available():
                        print(f"\n[Step {step+1}] Running evaluation (--eval-steps triggered)...")
                        model.eval()
                        
                        eval_results = run_eval(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            tasks=eval_tasks,
                            batch_size=getattr(args, 'eval_batch_size', 1),
                            limit=getattr(args, 'eval_limit', None),
                            verbosity="WARNING"
                        )
                        
                        if eval_results:
                            print(format_results(eval_results, tasks=eval_tasks))
                            
                            eval_path = os.path.join(args.out_dir, f"eval_epoch_{absolute_epoch+1}_step_{step+1}.json")
                            from ..evaluation import save_results
                            save_results(eval_results, eval_path)
                        
                        model.train()
                    else:
                        print("WARNING: lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
                except Exception as e:
                    print(f"WARNING: Step-based evaluation failed: {e}")
                    model.train()
        
        save_training_checkpoint_if_finite(
            build_training_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                dataloader,
                completed_epoch=absolute_epoch + 1,
                mid_epoch_step=0,
            ),
            os.path.join(args.out_dir, f"hierarchos_epoch_{absolute_epoch+1}.pt"),
            model,
            optimizer,
        )
        
        # --- OPTIONAL EVALUATION (lm-evaluation-harness) ---
        eval_tasks = getattr(args, 'eval_tasks', None)
        if eval_tasks:
            eval_every = getattr(args, 'eval_every_epoch', 1)
            if (epoch + 1) % eval_every == 0:
                try:
                    from ..evaluation import run_eval, format_results, is_lm_eval_available
                    
                    if is_lm_eval_available():
                        print(f"\n[Epoch {absolute_epoch+1}] Running evaluation on: {eval_tasks}")
                        model.eval()
                        
                        eval_results = run_eval(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            tasks=eval_tasks,
                            batch_size=getattr(args, 'eval_batch_size', 1),
                            limit=getattr(args, 'eval_limit', None),
                            verbosity="WARNING"  # Reduce lm-eval verbosity
                        )
                        
                        if eval_results:
                            print(format_results(eval_results, tasks=eval_tasks))
                            
                            # Save results to file
                            eval_path = os.path.join(args.out_dir, f"eval_epoch_{absolute_epoch+1}.json")
                            from ..evaluation import save_results
                            save_results(eval_results, eval_path)
                        
                        model.train()  # Back to training mode
                    else:
                        print("WARNING: lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
                except Exception as e:
                    print(f"WARNING: Evaluation failed: {e}")
        model.train()  # Ensure model is back in training mode

    # --- FINAL INFERENCE MODEL EXPORT ---
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save inference-ready model (no optimizer/scheduler state = smaller file)
    final_model_path = os.path.join(args.out_dir, "hierarchos.pt")
    print(f"Saving final inference model to: {final_model_path}")
    
    # Clean state dict (remove _orig_mod. prefix from compiled models)
    clean_state_dict = sanitize_model_state_dict(model)
    
    final_completed_epoch = epoch_offset + int(getattr(args, 'epochs', 0) or 0)
    if hasattr(model, 'config'):
        model.config.completed_epoch = final_completed_epoch
    final_config = dict(model.config)
    final_config['completed_epoch'] = final_completed_epoch
    final_checkpoint = {
        'model_state_dict': clean_state_dict,
        'config': final_config,
        'completed_epoch': final_completed_epoch,
        'training_complete': True,
    }
    save_training_checkpoint_if_finite(final_checkpoint, final_model_path, model, optimizer=None)
    
    # Save tokenizer files into the directory (HuggingFace-style portability)
    try:
        if tokenizer:
            tokenizer.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer: {e}")
    
    # Save config as JSON for easy inspection
    try:
        import json as json_module
        config_path = os.path.join(args.out_dir, "hierarchos_config.json")
        with open(config_path, 'w') as f:
            json_module.dump(final_config, f, indent=2, default=str)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Warning: Failed to save config JSON: {e}")
    
    # Calculate final model size
    model_size_bytes = os.path.getsize(final_model_path)
    if model_size_bytes >= 1e9:
        size_str = f"{model_size_bytes / 1e9:.2f} GB"
    else:
        size_str = f"{model_size_bytes / 1e6:.2f} MB"
    
    print(f"Final model size: {size_str}")
    print(f"Total epochs completed: {final_completed_epoch}")
    print(f"\nTo use the model for inference, run:")
    print(f"  python hierarchos_cli.py chat --model-path \"{args.out_dir}\"")
    print("="*60 + "\n")

def finetune(args, device, tokenizer, dataloader, dataloader_len):
    """
    LoRA-based fine-tuning with PEFT support.
    
    Ported from hierarchos.py monolith for full feature parity.
    """
    # Try importing PEFT
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        _HAS_PEFT = True
    except ImportError:
        _HAS_PEFT = False
    
    if not _HAS_PEFT:
        raise ImportError("Please install 'peft' for fine-tuning: pip install peft")
    
    print("Running in FINETUNE mode with LoRA...")
    if dataloader_len <= 0:
        print("ERROR: dataloader_len must be > 0. If automatic detection failed, please specify --dataset-size.")
        return
    configure_finetune_ltm_mode(args)
    print("INFO: Fine-tune LTM mode: read-only/inference-like (supervised LTM fast-memory writes disabled).")

    # Load the base model and its config
    model, model_config = load_full_model_with_config(args.model_path, device)
    if len(tokenizer) != int(model_config.vocab_size):
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary "
            f"({int(model_config.vocab_size)})."
        )

    # Ensure max_length from CLI is used if provided
    if args.max_length and args.max_length != model_config.get('max_length', 1024):
        print(f"INFO: Overriding loaded model max_length ({model_config.get('max_length')}) with CLI value ({args.max_length})")
        model_config.max_length = args.max_length
    elif 'max_length' not in model_config:
        print("Warning: max_length missing from loaded config. Using default 1024.")
        model_config.max_length = 1024

    # Ensure gradient_checkpointing flag from CLI is used
    gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
    if gradient_checkpointing != model_config.get('gradient_checkpointing', False):
        print(f"INFO: Setting gradient_checkpointing to {gradient_checkpointing}")
        model_config.gradient_checkpointing = gradient_checkpointing
    elif 'gradient_checkpointing' not in model_config:
        model_config.gradient_checkpointing = gradient_checkpointing

    # Ensure h_stride flag from CLI is used
    h_stride = getattr(args, 'h_stride', 4)
    if h_stride != model_config.get('h_stride', 4):
        print(f"INFO: Overriding model h_stride ({model_config.get('h_stride', 4)}) with CLI value ({h_stride})")
        model_config.h_stride = h_stride
    elif 'h_stride' not in model_config:
        model_config.h_stride = h_stride

    _apply_runtime_model_config_overrides(model_config, args)
    if device.type == 'cuda':
        configure_cuda_training_runtime(
            args,
            model_config,
            device,
            mode_name="LoRA fine-tuning",
        )
    model.config = model_config
    if hasattr(model, "refresh_runtime_config"):
        model.refresh_runtime_config()
    _print_runtime_stability_config(model)

    # Determine LoRA rank
    lora_r = getattr(args, 'lora_r', 8)
    finetune_unlock_percent = getattr(args, 'finetune_unlock_percent', None)
    
    if finetune_unlock_percent is not None:
        if lora_r != 8:  # Default value check
            print(f"Warning: Both --lora_r ({lora_r}) and --finetune-unlock-percent were specified. Prioritizing --lora_r.")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            target_modules = ["qproj", "in_proj", "val_proj", "h_to_context", "l_to_out", "h_halt_proj", "W_ir", "W_hr", "W_iz", "W_hz", "W_in", "W_hn"]
            lora_param_sum_per_r = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(tm in name for tm in target_modules):
                    lora_param_sum_per_r += module.in_features + module.out_features

            target_trainable_count = total_params * (finetune_unlock_percent / 100.0)
            if lora_param_sum_per_r > 0:
                estimated_r = target_trainable_count / lora_param_sum_per_r
                lora_r = max(1, int(round(estimated_r)))
                print(f"Targeting ~{finetune_unlock_percent}% trainable parameters. Estimated LoRA rank 'r' = {lora_r}")
            else:
                print("Warning: Could not find target modules for LoRA. Using default r=8.")

    lora_alpha = getattr(args, 'lora_alpha', 16)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            # RWKV time-mixing
            "key", "value", "receptance", "output",
            # RWKV channel-mixing
            "key_cm", "receptance_cm", "value_cm",
            # Hierarchos-specific layers
            "qproj", "in_proj", "h_to_context",
            "l_input_proj", "l_to_out", "h_halt_proj",
            "context_drift_proj", "l_feedback_proj", "val_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"],  # LTM still updated directly
    )
    model = get_peft_model(model, lora_config)

    # Resumption Logic
    start_epoch = 0
    start_step = 0
    checkpoint = None
    if getattr(args, 'resume_from_ckpt', None):
        print(f"Resuming LoRA finetune from: {args.resume_from_ckpt}")
        checkpoint = load_checkpoint_payload_compatible(
            args.resume_from_ckpt,
            map_location='cpu',
        )
        if 'model_state_dict' in checkpoint:
            state_dict = sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False)
            _reject_unsupported_rwkv_state_dict(state_dict, args.resume_from_ckpt)
            checkpoint['model_state_dict'] = state_dict
            load_model_state_dict_compatible(model, state_dict, args.resume_from_ckpt)
        start_epoch = checkpoint.get('completed_epoch', 0)
        start_step = checkpoint.get('mid_epoch_step', 0)
        print(f"INFO: Resuming from epoch {start_epoch+1}, step {start_step}.")

    model.print_trainable_parameters()

    # Optimizer selection
    if is_directml_device(device):
        print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
        optimizer = build_hierarchos_optimizer(model, args, device)
    else:
        optimizer = build_hierarchos_optimizer(model, args, device)
    
    if checkpoint and not getattr(args, 'override_scheduling', False):
        _restore_resume_component_state(
            optimizer,
            checkpoint.get('optimizer_state_dict'),
            "optimizer",
            args.resume_from_ckpt,
        )
        print("Successfully loaded optimizer state.")

    # Compile the Hierarchos hot path after PEFT has injected LoRA modules, so
    # the optimized graph includes the adapters instead of compiling the base
    # Linear layers before they are replaced.
    peft_base_model = getattr(getattr(model, "base_model", None), "model", None)
    compile_base_model = getattr(peft_base_model, "compile", None)
    if callable(compile_base_model):
        compile_base_model()

    os.makedirs(args.out_dir, exist_ok=True)

    # AMP setup (BFloat16 does NOT use GradScaler — only float16 needs it)
    scaler = None
    use_amp = getattr(args, 'amp', False)
    amp_dtype_str = getattr(args, 'amp_dtype', None) or getattr(model.config if hasattr(model, 'config') else args, 'amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    if use_amp:
        if amp_dtype_str == 'float16':
            scaler = GradScaler()
            if checkpoint and not getattr(args, 'override_scheduling', False):
                _restore_resume_component_state(
                    scaler,
                    checkpoint.get('scaler_state_dict'),
                    "AMP scaler",
                    args.resume_from_ckpt,
                )
        print(f"INFO: Automatic Mixed Precision (AMP) ENABLED for fine-tuning ({amp_dtype_str}).")

    # Scheduler setup
    scheduler = None
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    full_run_update_steps = compute_update_steps(dataloader_len, accumulation_steps) * args.epochs if dataloader_len > 0 else 0
    saved_main_lr_state = checkpoint.get('lr_scheduler_state') if isinstance(checkpoint, dict) else None
    if (
        checkpoint
        and not getattr(args, 'override_scheduling', False)
        and isinstance(saved_main_lr_state, dict)
        and saved_main_lr_state.get("enabled", True)
        and int(saved_main_lr_state.get("total_steps", 0) or 0) > 0
    ):
        num_update_steps = max(1, int(saved_main_lr_state.get("total_steps", 1) or 1))
        print(f"INFO: Continuing saved main LR schedule ({num_update_steps} total updates).")
    else:
        num_update_steps = full_run_update_steps
    args._main_lr_schedule_total_steps = int(num_update_steps)
    if not getattr(args, 'disable_lr_schedule', False):
        if num_update_steps > 0:
            scheduler = build_lr_scheduler(optimizer, args, num_update_steps)
            if checkpoint and not getattr(args, 'override_scheduling', False):
                _restore_resume_component_state(
                    scheduler,
                    checkpoint.get('scheduler_state_dict'),
                    "main LR scheduler",
                    args.resume_from_ckpt,
                )
        else:
            print("Warning: Cannot enable LR schedule, dataset might be too small.")
    configure_ltm_lr_schedule(
        args,
        num_update_steps,
        checkpoint=checkpoint,
        override_schedule=bool(getattr(args, 'override_scheduling', False)),
        scheduler=scheduler,
    )
    if hasattr(model, 'config'):
        model.config.ltm_lr = getattr(args, 'ltm_lr', getattr(model.config, 'ltm_lr', 1e-3))
        model.config.min_ltm_lr = getattr(args, 'min_ltm_lr', getattr(model.config, 'min_ltm_lr', None))
        model.config.disable_ltm_lr_schedule = getattr(args, 'disable_ltm_lr_schedule', False)

    startup_issue = _find_first_nonfinite_model_tensor(model, include_grads=True)
    if startup_issue:
        raise RuntimeError(f"Non-finite fine-tune model/gradient state is not recoverable safely: {startup_issue}")
    optimizer_issue = _find_first_nonfinite_optimizer_tensor(optimizer)
    if optimizer_issue:
        raise RuntimeError(f"Non-finite fine-tune optimizer state is not recoverable safely: {optimizer_issue}")
    _clamp_model_finite_magnitude_(
        model,
        getattr(args, 'startup_weight_max_abs', 0.0),
        log_prefix="fine-tune startup model",
    )
    _sanitize_model_transient_state_(model, max_abs=getattr(args, 'grad_clip', 1.0))

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    ponder_loss_weight = getattr(args, 'ponder_loss_weight', 0.01)
    commitment_loss_weight = getattr(args, 'commitment_loss_weight', 0.5)
    grad_clip = getattr(args, 'grad_clip', 1.0)
    use_ltm_inner_updates = ltm_inner_updates_enabled(args)
    ltm_value_alignment_weight = _nonnegative_float(
        getattr(args, 'ltm_value_alignment_weight', 0.0),
        0.0,
    )

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        set_dataloader_epoch(dataloader, epoch)
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}", total=dataloader_len)
        batch_source = (
            CUDABatchPrefetcher(pbar, device)
            if device.type == 'cuda' and bool(getattr(args, 'cuda_prefetch', True))
            else pbar
        )
        metric_totals = torch.zeros(3, device=device, dtype=torch.float32)
        
        backward_called_in_cycle = False
        steps_in_epoch = 0

        for i, batch in enumerate(batch_source):
            # Mid-Epoch Resumption: Skip steps already processed
            if epoch == start_epoch and i < start_step:
                if i == start_step - 1:
                    print(f"INFO: Resuming from mid-epoch step {start_step}...")
                continue

            if batch is None:
                continue

            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch["labels"]
            rosa_ids = batch.get("rosa_ids")
            loss_weights = batch.get("loss_weights")
            input_ids, labels, attention_mask = trim_trailing_padding(
                input_ids, labels, attention_mask
            )
            if rosa_ids is not None:
                rosa_ids = rosa_ids[:, :input_ids.shape[1]].contiguous()
            if loss_weights is not None:
                loss_weights = loss_weights[:, :input_ids.shape[1]].contiguous()
            input_ids = _batch_tensor_to_device(
                batch,
                "input_ids",
                input_ids,
                device,
                pad_value=int(getattr(args, 'pad_token_id', 0)),
            )
            attention_mask = _batch_tensor_to_device(
                batch,
                "attention_mask",
                attention_mask,
                device,
                pad_value=0,
            )
            labels = _batch_tensor_to_device(
                batch,
                "labels",
                labels,
                device,
                pad_value=-100,
            )
            labels = labels.to(dtype=torch.long)
            rosa_sentinel = int(
                getattr(model_config, 'vocab_size', getattr(args, 'vocab_size', 0))
            )
            rosa_ids = _batch_tensor_to_device(
                batch,
                "rosa_ids",
                rosa_ids,
                device,
                pad_value=rosa_sentinel,
            )
            loss_weights = _batch_tensor_to_device(
                batch,
                "loss_weights",
                loss_weights,
                device,
                pad_value=0.0,
            )
            set_model_training_step(model, epoch * dataloader_len + i)

            autocast_device_type = 'cpu' if is_directml_device(device) else device.type
            with autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    rosa_ids=rosa_ids,
                    loss_weights=loss_weights,
                    return_topk_values=False,
                    return_raw_topk_values=use_ltm_inner_updates,
                    return_topk_indices=use_ltm_inner_updates,
                    compute_ltm_value_alignment=ltm_value_alignment_weight > 0.0,
                )
                
                if use_ltm_inner_updates and outputs.get("raw_topk_vals") is not None:
                    for t_val in outputs["raw_topk_vals"]:
                        if t_val.requires_grad:
                            t_val.retain_grad()
                
                cross_entropy_loss = outputs.get("loss")
                ponder_cost = outputs.get("ponder_cost")
                commitment_cost = outputs.get("commitment_cost")
                ltm_value_alignment_cost = outputs.get("ltm_value_alignment_cost")

                combined_loss = None
                if cross_entropy_loss is not None:
                    loss_accum = _cap_loss_component_for_backward(
                        cross_entropy_loss,
                        getattr(args, 'max_ce_loss_for_backward', 0.0),
                    )
                    zero = torch.zeros_like(cross_entropy_loss)
                    pc_valid = (
                        torch.isfinite(ponder_cost).all()
                        if ponder_cost is not None else torch.tensor(False, device=device)
                    )
                    cc_valid = (
                        torch.isfinite(commitment_cost).all()
                        if commitment_cost is not None else torch.tensor(False, device=device)
                    )
                    safe_ponder = (
                        torch.where(pc_valid, ponder_cost, zero)
                        if ponder_cost is not None else zero
                    )
                    safe_commitment = (
                        torch.where(cc_valid, commitment_cost, zero)
                        if commitment_cost is not None else zero
                    )
                    loss_accum = loss_accum + (
                        ponder_loss_weight
                        * _cap_loss_component_for_backward(
                            safe_ponder,
                            getattr(args, 'max_ponder_cost_for_backward', 0.0),
                        )
                    )
                    loss_accum = loss_accum + (
                        commitment_loss_weight
                        * _cap_loss_component_for_backward(
                            safe_commitment,
                            getattr(args, 'max_commitment_cost_for_backward', 2.0),
                            preserve_gradient=True,
                        )
                    )
                    if ltm_value_alignment_weight > 0.0:
                        alignment_valid = (
                            torch.isfinite(ltm_value_alignment_cost).all()
                            if ltm_value_alignment_cost is not None
                            else torch.tensor(False, device=device)
                        )
                        safe_alignment = (
                            torch.where(alignment_valid, ltm_value_alignment_cost, zero)
                            if ltm_value_alignment_cost is not None else zero
                        )
                        loss_accum = loss_accum + (
                            ltm_value_alignment_weight * safe_alignment
                        )
                    combined_loss = loss_accum

                combined_valid = (
                    combined_loss is not None
                    and bool(torch.isfinite(combined_loss).all().item())
                )
                if combined_loss is not None and not combined_valid:
                    print(
                        f"\nCRITICAL: Non-finite fine-tune loss at step {i+1}; "
                        "skipping batch and clearing gradients."
                    )
                    print("  " + _describe_tensor_issue("combined_loss", combined_loss))
                    combined_loss = None
                    optimizer.zero_grad(set_to_none=True)

            if combined_loss is not None:
                accumulation_divisor = accumulation_divisor_for_step(i, dataloader_len, accumulation_steps)
                loss_to_backward = combined_loss / accumulation_divisor

                if use_amp and scaler:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                backward_called_in_cycle = True

                metric_totals.add_(torch.stack([
                    cross_entropy_loss.detach().float(),
                    safe_ponder.detach().float(),
                    safe_commitment.detach().float(),
                ]))
                steps_in_epoch += 1

            if should_step_accumulation(i, dataloader_len, accumulation_steps):
                if backward_called_in_cycle:
                    # LTM Update
                    ltm_grads = None
                    if use_ltm_inner_updates and outputs.get("raw_topk_vals") is not None:
                        valid_grads = []
                        for t_val in outputs["raw_topk_vals"]:
                            g = t_val.grad
                            if g is not None:
                                valid_grads.append(g.detach())
                            else:
                                valid_grads.append(torch.zeros_like(t_val))
                        ltm_grads = torch.stack(valid_grads, dim=1)
                        for t_val in outputs["raw_topk_vals"]:
                            t_val.grad = None
                        outputs["raw_topk_vals"] = None

                    if ltm_grads is not None:
                        ltm_grads_copy = ltm_grads.detach().clone()

                        valid_update = True
                        if use_amp and scaler:
                            current_scale = scaler.get_scale()
                            if current_scale > 1e-6:
                                ltm_grads_copy = ltm_grads_copy / current_scale
                            else:
                                valid_update = False

                        if valid_update:
                            ltm_clip = _positive_float(grad_clip, 1.0)
                            ltm_grads_copy = _prepare_ltm_update_gradients(
                                ltm_grads_copy,
                                ltm_clip,
                                inplace=True,
                            )
                            if ltm_grads_copy is None:
                                print(
                                    f"\nCRITICAL: Non-finite fine-tune LTM gradient at step {i+1}. "
                                    "Skipping optimizer and memory updates."
                                )
                                _reset_after_nonfinite(optimizer, model)
                                backward_called_in_cycle = False
                                continue
                            base_model = getattr(getattr(model, "base_model", None), "model", model)
                            base_ltm = getattr(base_model, "ltm", None)
                            topk_idx = outputs.get("topk_idx")
                            if base_ltm is not None and topk_idx is not None:
                                base_ltm.inner_update(
                                    topk_idx,
                                    ltm_grads_copy,
                                    current_lr=get_current_ltm_lr(args),
                                    timestamp=float(i + 1),
                                    source=2  # SRC_TRAINING_DATA
                                )

                    # Optimizer step
                    if use_amp and scaler:
                        scaler.unscale_(optimizer)
                        grads_ok, grad_issue = _clip_gradients_and_check(
                            model,
                            grad_clip,
                            getattr(args, 'max_sanitized_gradient_values', 0),
                        )
                        if not grads_ok:
                            print(f"\nCRITICAL: Non-finite fine-tune gradient at step {i+1}. {grad_issue}")
                            print("  Skipping optimizer step and clearing accumulated gradients.")
                            _reset_after_nonfinite(optimizer, model)
                            backward_called_in_cycle = False
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grads_ok, grad_issue = _clip_gradients_and_check(
                            model,
                            grad_clip,
                            getattr(args, 'max_sanitized_gradient_values', 0),
                        )
                        if not grads_ok:
                            print(f"\nCRITICAL: Non-finite fine-tune gradient at step {i+1}. {grad_issue}")
                            print("  Skipping optimizer step and clearing accumulated gradients.")
                            _reset_after_nonfinite(optimizer, model)
                            backward_called_in_cycle = False
                            continue
                        optimizer.step()

                    if ltm_value_alignment_weight > 0.0:
                        mark_val_proj_trained(model)

                    if scheduler:
                        scheduler.step()
                    if use_ltm_inner_updates:
                        advance_ltm_lr_schedule(args)

                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False
                    global_step += 1

                    # Periodic Checkpointing (Progress Protection)
                    if getattr(args, 'save_steps', 0) > 0 and (i + 1) % args.save_steps == 0:
                        ckpt_path = os.path.join(args.out_dir, f"hierarchos_finetune_epoch_{epoch+1}_step_{i+1}.pt")
                        print(f"\n[Step {i+1}] Periodic Checkpoint: Saving to {ckpt_path}...")
                        save_training_checkpoint_if_finite({
                            'completed_epoch': epoch,
                            'mid_epoch_step': i + 1,
                            'model_state_dict': sanitize_model_state_dict(model),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'lr_scheduler_state': capture_main_lr_scheduler_state(args, scheduler),
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'ltm_scheduler_state': capture_ltm_lr_scheduler_state(args),
                            'config': dict(model.config),
                        }, ckpt_path, model, optimizer)
                else:
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss.")
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False

            # One device-to-host metric read at the same throttled cadence as
            # pretraining, instead of three `.item()` synchronizations per batch.
            first_logged_step = start_step if epoch == start_epoch else 0
            if should_update_progress(i, args, dataloader_len, first_logged_step):
                divisor = max(1, steps_in_epoch)
                avg_loss, avg_ponder, avg_commit = (
                    metric_totals / divisor
                ).tolist()
                current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "ponder": f"{avg_ponder:.2f}",
                    "commit": f"{avg_commit:.2e}",
                    "lr": f"{current_lr:.2e}",
                    "ltm_lr": f"{get_current_ltm_lr(args):.2e}" if use_ltm_inner_updates else "off"
                })

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    
    # Save tokenizer
    try:
        if tokenizer:
            tokenizer.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer with adapter. Error: {e}")
