import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import time
import random
from tqdm import tqdm
import sys
import traceback
import numpy as np
import math
import json
import hashlib
import itertools

from .optimizers import DirectMLAdamW
from .objectives import adaptive_ponder_objective, resolve_ponder_objective
from ..utils.device import is_directml_device, set_threads
from ..utils.tokenizer import validate_inference_tokenizer_identity
from ..utils.checkpoint import (
    TRANSIENT_LTM_STATE_KEYS,
    load_checkpoint_payload_compatible,
    save_checkpoint_safely,
    load_full_model_with_config,
    load_model_state_dict_compatible,
    sanitize_model_state_dict,
    _infer_arch_flags_from_state_dict,
    _reject_unsupported_rwkv_state_dict,
    validate_checkpoint_architecture_contract,
)
from ..models.core import HierarchosCore, _validate_sequence_mask_contract
from ..models.revisions import (
    architecture_contract,
    architecture_contract_hash,
    architecture_default_training_chunk_size,
    normalize_ltm_training_mode as _normalize_ltm_training_mode_contract,
)
from ..evaluation.selection import BestMetric, extract_selection_metric
from ..utils.rosa import rosa_batch_parallel, rosa_context_mode

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _resolved_training_chunk_size(args, model_config=None) -> int:
    """Resolve user input first, then the model's revisioned chunk geometry."""

    for source in (args, model_config):
        if source is None:
            continue
        value = (
            source.get("training_chunk_size")
            if isinstance(source, dict)
            else getattr(source, "training_chunk_size", None)
        )
        if value is not None:
            return int(value)
    return architecture_default_training_chunk_size(model_config or args)

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


def _reject_nonfinite_train_batch(
    optimizer,
    model,
    *,
    had_accumulated_gradients_on_entry: bool,
):
    """Reject a poisoned batch without silently dropping earlier microbatches."""
    if had_accumulated_gradients_on_entry:
        raise RuntimeError(
            "A non-finite trajectory occurred after valid gradients had already "
            "been accumulated from an earlier microbatch. Refusing to clear "
            "those gradients and silently change the accumulation objective; "
            "resume from the last verified checkpoint after diagnosing the batch."
        )
    return None, _reset_after_nonfinite(optimizer, model)

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


def _adaptive_ponder_loss(args, model, expected_steps, difficulty):
    model_config = getattr(model, "config", None)
    architecture_revision = (
        model_config.get("architecture_revision")
        if isinstance(model_config, dict)
        else getattr(model_config, "architecture_revision", None)
    )
    mode = resolve_ponder_objective(
        getattr(args, "ponder_objective", "auto"),
        architecture_revision=architecture_revision,
    )
    result = adaptive_ponder_objective(
        expected_steps,
        difficulty,
        max_steps=int(getattr(args, "max_h_steps", 5) or 5),
        target_scale=float(getattr(args, "ponder_target_scale", 0.5) or 0.0),
        min_steps=float(getattr(args, "min_h_steps", 1) or 1),
        mode=mode,
        huber_beta=float(getattr(args, "ponder_huber_beta", 0.5) or 0.5),
    )
    return result.loss

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
    clamped = torch.clamp(detached, min=-max_abs, max=max_abs)
    # A TBPTT boundary is not a recovery boundary. Preserve NaN/Inf so a bad
    # trajectory can never be rewritten into an apparently valid carrier.
    return torch.where(torch.isfinite(detached), clamped, detached)

def _detach_finite_l2_clamp(tensor: torch.Tensor, max_abs: float, max_norm: float) -> torch.Tensor:
    detached = _detach_finite_clamp(tensor, max_abs)
    max_norm = _nonnegative_float(max_norm, 0.0)
    if max_norm <= 0.0 or not torch.is_tensor(detached) or not detached.is_floating_point():
        return detached
    norm = torch.linalg.vector_norm(detached.float(), ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(detached.new_tensor(max_norm) / (norm.to(dtype=detached.dtype) + 1e-6), max=1.0)
    return detached * scale


_RECURRENT_CARRIER_OUTPUT_KEYS = (
    "h_state",
    "l_state",
    "prev_context",
    "target_context",
    "drift_state",
)


def _recurrent_carrier_finite_checks(outputs):
    """Build device-side checks without one host sync per recurrent carrier."""

    checks = []
    for name in _RECURRENT_CARRIER_OUTPUT_KEYS:
        value = outputs.get(name)
        if torch.is_tensor(value) and value.is_floating_point():
            checks.append((name, value, torch.isfinite(value.detach()).all()))
    return checks

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
    "act_depth_temperature",
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

    # Base pretraining has no stochastic forward modules, so retaining every
    # CPU/CUDA RNG state per temporal segment is avoidable overhead. PEFT LoRA
    # adapters may contain active dropout, however; rematerializing those
    # segments with a fresh mask would produce gradients for a different
    # function than the original forward. Fine-tune setup records the policy
    # once rather than rescanning the module tree for every segment.
    preserve_rng_state = bool(
        getattr(model, "_hierarchos_checkpoint_preserve_rng_state", False)
    )
    return activation_checkpoint(
        run,
        *values,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
        determinism_check="default",
    )


def _checkpoint_replay_safe_rosa_ids(model, input_ids: torch.Tensor):
    """Precompute deterministic ROSA inputs before activation checkpointing.

    ``ROSAState`` is a mutable Python suffix automaton. Passing a live state
    through ``torch.utils.checkpoint`` is not safe: backward rematerialization
    can observe the history after later temporal segments have already mutated
    it. A complete prediction stream is deterministic and cheap to retain, so
    checkpointed full-sample BPTT computes it once and replays only tensors.
    This also avoids repeating CPU suffix work during backward.
    """

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            "Checkpointed ROSA precomputation requires model.config"
        )
    vocab_size = int(getattr(config, "vocab_size"))
    enforce_max_context = bool(
        getattr(config, "enforce_rosa_max_context", False)
    )
    rosa_max_context = int(getattr(config, "rosa_max_context", 512) or 0)
    effective_max_context = (
        max(1, rosa_max_context)
        if enforce_max_context and rosa_max_context > 0
        else 0
    )
    token_rows = input_ids.detach().to(
        device="cpu",
        dtype=torch.long,
    ).tolist()
    prediction_rows, _states = rosa_batch_parallel(
        token_rows,
        states=None,
        max_context=effective_max_context,
    )
    rosa_ids = torch.tensor(prediction_rows, dtype=torch.long)
    return rosa_ids.masked_fill_(rosa_ids < 0, vocab_size), rosa_context_mode(
        enforce_max_context
    )


def configure_checkpoint_rng_policy(model) -> bool:
    """Record whether activation rematerialization must replay RNG exactly."""
    preserve_rng_state = any(
        isinstance(module, nn.modules.dropout._DropoutNd)
        and module.training
        and float(module.p) > 0.0
        for module in model.modules()
    )
    model._hierarchos_checkpoint_preserve_rng_state = preserve_rng_state
    return preserve_rng_state


def _restore_resume_component_state(component, state, component_name: str, checkpoint_path: str):
    """Restore exact continuation state or stop before silently changing dynamics."""
    if state is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no {component_name} state. Exact --resume-from-ckpt "
            "cannot continue safely; use --reset-optimizer-state and/or "
            "--rebuild-lr-schedule to intentionally start fresh "
            "optimizer/scheduler state, or use --model-path for a weights-only continuation."
        )
    issue = _find_first_nonfinite_payload_tensor(state, f"{component_name}_state")
    if issue:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has corrupt {component_name} state: {issue}. "
            "Refusing to rewrite non-finite continuation state; use an earlier checkpoint, "
            "or explicit reset/rebuild flags only when fresh state is intentional."
        )
    try:
        component.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(
            f"Could not restore {component_name} state from {checkpoint_path}: {exc}. "
            "Refusing a silent fresh-state resume. Use explicit reset/rebuild flags only if that reset "
            "is intentional."
        ) from exc


def restore_scheduler_state_and_live_lrs(
    scheduler,
    optimizer,
    state,
    checkpoint_path: str,
):
    """Restore scheduler counters and the LR used by the very next update.

    LambdaLR construction immediately rewrites optimizer group LRs. Loading its
    state restores ``_last_lr`` only inside the scheduler, so without this
    explicit copy logs report one LR while the optimizer applies another.
    """
    _restore_resume_component_state(
        scheduler,
        state,
        "main LR scheduler",
        checkpoint_path,
    )
    last_lrs = state.get("_last_lr") if isinstance(state, dict) else None
    if last_lrs is None:
        last_lrs = list(scheduler.get_last_lr())
    base_lrs = (
        state.get("base_lrs")
        if isinstance(state, dict)
        else getattr(scheduler, "base_lrs", None)
    )
    if len(last_lrs) != len(optimizer.param_groups):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} scheduler has {len(last_lrs)} LR groups, "
            f"but the optimizer has {len(optimizer.param_groups)}. Refusing an inexact resume."
        )
    if base_lrs is not None and len(base_lrs) != len(optimizer.param_groups):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} scheduler base LR group count does not "
            "match the optimizer."
        )
    for idx, (group, live_lr) in enumerate(zip(optimizer.param_groups, last_lrs)):
        group["lr"] = float(live_lr)
        if base_lrs is not None:
            group["initial_lr"] = float(base_lrs[idx])
    scheduler._last_lr = [float(value) for value in last_lrs]
    live_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if live_lrs != [float(value) for value in scheduler.get_last_lr()]:
        raise RuntimeError(
            f"Scheduler/optimizer live LR parity failed after restoring {checkpoint_path}."
        )
    print(
        "INFO: Restored scheduler counters and live optimizer LR(s): "
        + ", ".join(f"{value:.8e}" for value in live_lrs)
    )
    return live_lrs

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


def validate_exact_running_states(checkpoint, args, start_step: int, source: str):
    """Require a usable finite recurrent carrier when exact replay needs it."""
    if (
        not isinstance(checkpoint, dict)
        or int(start_step or 0) <= 0
        or not bool(getattr(args, "persist_state", False))
    ):
        return False
    running_states = checkpoint.get("running_states")
    if not isinstance(running_states, (list, tuple)) or len(running_states) != 6:
        raise RuntimeError(
            f"Exact mid-epoch resume with persist_state=True requires a six-part "
            f"running_states carrier in {source}. Resume from an epoch boundary, "
            "disable cross-batch persistence for a new run, or use --model-path "
            "for an intentional weights-only continuation."
        )
    state_names = (
        "h_state",
        "l_state",
        "prev_context",
        "target_context",
        "drift_state",
    )
    for index, state_name in enumerate(state_names):
        if not torch.is_tensor(running_states[index]):
            raise RuntimeError(
                f"Exact persisted running_states in {source} has no tensor "
                f"{state_name}; refusing to reconstruct recurrent history."
            )
    ltm_state = running_states[5]
    if not isinstance(ltm_state, (list, tuple)) or len(ltm_state) != 7:
        raise RuntimeError(
            f"Exact persisted running_states in {source} requires the seven-part "
            "coherent-v9 LTM carrier."
        )
    if not torch.is_tensor(ltm_state[0]) or not torch.is_tensor(ltm_state[1]):
        raise RuntimeError(
            f"Exact persisted running_states in {source} is missing the LTM "
            "fast-value or momentum tensor."
        )
    issue = _find_first_nonfinite_payload_tensor(
        running_states,
        "running_states",
    )
    if issue:
        raise RuntimeError(
            f"Checkpoint {source} has non-finite persisted running state: {issue}. "
            "Refusing to replace exact recurrent history with reset state."
        )
    return True

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

def _training_step_model_candidates(model):
    candidates = [model, getattr(model, "_orig_mod", None)]
    base_model = getattr(model, "base_model", None)
    inner_model = getattr(base_model, "model", None)
    candidates.extend([
        base_model,
        inner_model,
        getattr(inner_model, "_orig_mod", None),
    ])
    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate


def set_model_training_step(model, step: int):
    for candidate in _training_step_model_candidates(model):
        setter = getattr(candidate, "set_training_step", None)
        if callable(setter):
            setter(step)
            return


def get_model_training_step(model):
    """Read the persisted memory-gate curriculum position through wrappers."""
    for candidate in _training_step_model_candidates(model):
        value = getattr(candidate, "memory_gate_warmup_step", None)
        if torch.is_tensor(value) and value.numel() == 1:
            raw_value = float(value.detach().cpu().item())
            if (
                not math.isfinite(raw_value)
                or raw_value < 0.0
                or not raw_value.is_integer()
            ):
                raise RuntimeError(
                    "Loaded memory_gate_warmup_step must be a finite, "
                    "nonnegative integer"
                )
            return int(raw_value)
    return None


def resolve_training_step_offset(model, next_local_step: int) -> int:
    """Continue a persisted gate curriculum independent of new loader geometry."""
    saved_step = get_model_training_step(model)
    if saved_step is None:
        return 0
    return int(saved_step) + 1 - max(0, int(next_local_step or 0))


def mark_val_proj_trained(model, alignment_cost=None):
    """Record one successful writer update and quality-gate its capability.

    Update count alone cannot prove that a writer maps values into the learned
    LTM read channel.  Readiness therefore requires a finite alignment EMA below
    the checkpointed threshold and a finite, bounded writer norm.
    """
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
        def read(name, default=None):
            return (
                config.get(name, default)
                if isinstance(config, dict)
                else getattr(config, name, default)
            )

        def write(name, value):
            if isinstance(config, dict):
                config[name] = value
            else:
                setattr(config, name, value)

        updates = int(read("val_proj_alignment_updates", 0) or 0) + 1
        write("val_proj_alignment_updates", updates)

        score = None
        if torch.is_tensor(alignment_cost):
            if alignment_cost.numel() == 1:
                score = float(alignment_cost.detach().float().item())
        elif alignment_cost is not None:
            try:
                score = float(alignment_cost)
            except (TypeError, ValueError):
                score = None

        score_is_finite = score is not None and math.isfinite(score) and score >= 0.0
        if score_is_finite:
            decay = float(read("ltm_value_alignment_ema_decay", 0.95) or 0.0)
            decay = min(max(decay, 0.0), 0.999999)
            previous_ema = read("val_proj_alignment_ema", None)
            if previous_ema is None or not math.isfinite(float(previous_ema)):
                ema = score
            else:
                ema = decay * float(previous_ema) + (1.0 - decay) * score
            previous_best = read("val_proj_alignment_best", None)
            best = (
                score
                if previous_best is None or not math.isfinite(float(previous_best))
                else min(float(previous_best), score)
            )
            write("val_proj_alignment_last", score)
            write("val_proj_alignment_ema", ema)
            write("val_proj_alignment_best", best)
        else:
            ema = float("inf")

        writer = getattr(candidate, "val_proj", None)
        writer_weight = getattr(writer, "weight", None)
        if torch.is_tensor(writer_weight):
            writer_norm = float(writer_weight.detach().float().norm().item())
            writer_norm_is_finite = math.isfinite(writer_norm)
            write("val_proj_writer_norm", writer_norm)
        else:
            writer_norm = 0.0
            writer_norm_is_finite = True

        ready_threshold = float(
            read("ltm_value_alignment_ready_threshold", 0.95) or 0.0
        )
        writer_max_norm = float(read("ltm_value_writer_max_norm", 64.0) or 0.0)
        quality_ready = score_is_finite and ema <= ready_threshold
        norm_ready = writer_norm_is_finite and (
            writer_max_norm <= 0.0 or writer_norm <= writer_max_norm
        )
        write(
            "val_proj_trained",
            bool(updates >= minimum_updates and quality_ready and norm_ready),
        )


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

def supervised_weight_mass(batch) -> float:
    labels = batch.get("labels") if isinstance(batch, dict) else None
    if not torch.is_tensor(labels) or labels.ndim != 2 or labels.shape[1] <= 1:
        return 0.0
    active = labels[:, 1:].ne(-100)
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.shape == labels.shape:
        active = active & attention_mask[:, 1:].ne(0)
    loss_weights = batch.get("loss_weights")
    if torch.is_tensor(loss_weights) and loss_weights.shape == labels.shape:
        mass = (loss_weights[:, 1:].to(dtype=torch.float64) * active).sum()
    else:
        mass = active.sum().to(dtype=torch.float64)
    return float(mass.item())

def _divide_pending_gradients_(model, divisor: float):
    divisor = float(divisor)
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise RuntimeError(
            f"Invalid weighted-token accumulation mass {divisor!r}; refusing optimizer step."
        )
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(divisor)

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

def reset_optimizer_state_requested(args) -> bool:
    return bool(
        getattr(args, "reset_optimizer_state", False)
        or getattr(args, "override_scheduling", False)
    )

def rebuild_lr_schedule_requested(args) -> bool:
    return bool(
        getattr(args, "rebuild_lr_schedule", False)
        or getattr(args, "override_scheduling", False)
    )

def build_lr_scheduler(
    optimizer,
    args,
    num_update_steps: int,
    *,
    resume_schedule_state=None,
):
    restoring_saved_curve = (
        isinstance(resume_schedule_state, dict)
        and not rebuild_lr_schedule_requested(args)
    )
    schedule_enabled = (
        bool(resume_schedule_state.get("enabled", True))
        if restoring_saved_curve
        else not bool(getattr(args, 'disable_lr_schedule', False))
    )
    if not schedule_enabled or num_update_steps <= 0:
        return None

    total_steps = max(
        1,
        int(
            (
                resume_schedule_state.get("total_steps")
                if restoring_saved_curve
                else num_update_steps
            )
            or 1
        ),
    )
    max_lr = _positive_float(
        (
            resume_schedule_state.get("max_lr")
            if restoring_saved_curve
            else getattr(args, 'starting_lr', 1e-4)
        ),
        1e-4,
    )
    min_lr = min(
        _nonnegative_float(
            (
                resume_schedule_state.get("min_lr")
                if restoring_saved_curve
                else getattr(args, 'min_lr', 0.0)
            ),
            0.0,
        ),
        max_lr,
    )
    min_factor = min_lr / max_lr if max_lr > 0.0 else 0.0
    if restoring_saved_curve:
        saved_resolved_warmup = resume_schedule_state.get(
            "resolved_warmup_steps"
        )
        if saved_resolved_warmup is not None:
            warmup_steps = min(
                max(0, int(saved_resolved_warmup or 0)),
                max(0, total_steps - 1),
            )
        else:
            saved_warmup_args = AttrDict(
                warmup_steps=resume_schedule_state.get("warmup_steps", 0),
                warmup_ratio=resume_schedule_state.get("warmup_ratio", 0.0),
            )
            warmup_steps = resolve_lr_warmup_steps(
                saved_warmup_args,
                total_steps,
            )
    else:
        warmup_steps = resolve_lr_warmup_steps(args, total_steps)
    if rebuild_lr_schedule_requested(args):
        # Optimizer state_dicts retain the old scheduler's ``initial_lr`` in
        # every parameter group. LambdaLR otherwise reuses that stale base and
        # a requested new --starting-lr only changes the dimensionless lambda,
        # so the rebuilt schedule can warm to the wrong peak. Reset only LR
        # metadata; Adam moments remain untouched.
        for group in optimizer.param_groups:
            group["lr"] = max_lr
            group["initial_lr"] = max_lr

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
    if state and not override_schedule:
        schedule_enabled = bool(state.get("enabled", schedule_enabled))
        max_lr = _positive_float(state.get("max_lr", max_lr), max_lr)
        min_lr = min(
            _nonnegative_float(state.get("min_lr", min_lr), min_lr),
            max_lr,
        )
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
        "resolved_warmup_steps": int(
            resolve_lr_warmup_steps(args, max(1, total_steps))
        ),
        "override_scheduling": bool(getattr(args, 'override_scheduling', False)),
        "rebuild_lr_schedule": rebuild_lr_schedule_requested(args),
        "reset_optimizer_state": reset_optimizer_state_requested(args),
    }

def build_hierarchos_optimizer(model, args, device):
    """RWKV-style AdamW grouping: decay matrices/embeddings, never norms or scalars."""
    lr = args.starting_lr
    weight_decay = float(getattr(args, "rwkv_weight_decay", 0.1))
    grouping_version = int(getattr(args, "_optimizer_grouping_version", 2) or 2)
    args._optimizer_grouping_version = grouping_version
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
        elif grouping_version <= 1:
            # Exact-resume compatibility for checkpoints created before raw
            # RWKV nn.Parameter matrices were classified by dimensionality.
            should_decay = (
                (".weight" in clean_name or "emb" in clean_name)
                and "ln" not in clean_name
                and "norm" not in clean_name
            )
            (decay if should_decay else no_decay).append(param)
        elif param.ndim >= 2:
            # RWKV's w1/w2/a1/a2/v1/v2/g1/g2/r_k tensors are raw
            # nn.Parameters, so name-only ".weight" tests silently omitted
            # them. Dimensionality expresses the documented matrix policy.
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
    print(
        f"INFO: Optimizer grouping v{grouping_version}: "
        f"{len(decay)} matrix/embedding tensor(s) with decay, "
        f"{len(no_decay)} norm/vector/special tensor(s) without decay."
    )
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


def set_dataloader_start_batch(dataloader, start_batch: int) -> bool:
    """Apply a deterministic map-style resume cursor before dataset fetching.

    Returns True only when the loader's sampler can begin directly at the
    requested batch. Iterable datasets and third-party samplers fall back to
    CPU-side iterator skipping in ``train``; importantly, that fallback still
    happens before CUDA prefetch/H2D transfer.
    """
    start_batch = max(0, int(start_batch or 0))
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    set_start_batch = getattr(batch_sampler, "set_start_batch", None)
    if callable(set_start_batch):
        set_start_batch(start_batch)
        return True

    # DataLoader wraps ordinary sample samplers in torch's BatchSampler. Our
    # EpochShuffleSampler exposes an index cursor so it can avoid materializing
    # every skipped record while preserving the exact epoch permutation.
    sample_sampler = getattr(batch_sampler, "sampler", None)
    if sample_sampler is None:
        sample_sampler = getattr(dataloader, "sampler", None)
    set_start_index = getattr(sample_sampler, "set_start_index", None)
    batch_size = getattr(batch_sampler, "batch_size", None)
    if batch_size is None:
        batch_size = getattr(dataloader, "batch_size", None)
    if callable(set_start_index) and batch_size is not None:
        set_start_index(start_batch * max(1, int(batch_size)))
        return True
    return start_batch == 0


def host_batches_from_resume(
    iterable,
    start_batch: int,
    sampler_cursor_applied: bool,
    total_batches: int = None,
):
    """Place cursor skipping and the declared epoch bound before H2D prefetch."""
    start_batch = max(0, int(start_batch or 0))
    if start_batch > 0 and not sampler_cursor_applied:
        iterable = itertools.islice(iterable, start_batch, None)
    if total_batches is not None:
        remaining = max(0, int(total_batches) - start_batch)
        iterable = itertools.islice(iterable, remaining)
    return iterable


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
    return _normalize_ltm_training_mode_contract(value)

def ltm_inner_updates_enabled(args) -> bool:
    return normalize_ltm_training_mode(getattr(args, "ltm_training_mode", "inner-update")) == "inner-update"


def configure_finetune_ltm_mode(args) -> str:
    """Keep LoRA samples isolated from target-gradient fast-memory writes."""
    mode = normalize_ltm_training_mode(getattr(args, "ltm_training_mode", "inner-update"))
    if mode == "inner-update":
        print(
            "WARNING: Target-gradient LTM writes are unavailable during ordinary "
            "autoregressive inference and would leak into unrelated batches as "
            "label-derived state. Using read-only LTM for fine-tuning."
        )
        mode = "read-only"
    args.ltm_training_mode = mode
    return mode


def ensure_finetune_training_mode(model):
    """Put the complete PEFT/base hierarchy into training mode, or fail."""
    model.train()
    inactive_trainable_modules = []
    for name, module in model.named_modules():
        owns_trainable_parameter = any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=False)
        )
        if owns_trainable_parameter and not module.training:
            inactive_trainable_modules.append(name or "<root>")
    if inactive_trainable_modules:
        preview = ", ".join(inactive_trainable_modules[:8])
        raise RuntimeError(
            "Fine-tune model contains trainable modules left in eval mode after "
            f"model.train(): {preview}"
        )
    return model


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
        curr_ltm[6].detach() if len(curr_ltm) >= 7 and isinstance(curr_ltm[6], torch.Tensor) else None,
    )

def capture_model_grad_state(model):
    grad_state = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_state[name.replace("_orig_mod.", "")] = param.grad.detach().cpu().clone()
    return grad_state or None

def restore_model_grad_state(
    model,
    grad_state,
    device,
    *,
    expected_keys=None,
    strict: bool = False,
):
    if not grad_state:
        if strict:
            raise RuntimeError(
                "Checkpoint declares active gradient accumulation but has no "
                "saved gradient mapping."
            )
        return False
    if not isinstance(grad_state, dict):
        if strict:
            raise RuntimeError("Saved gradient state must be a parameter-name mapping.")
        return False
    issue = _find_first_nonfinite_payload_tensor(grad_state, "grad_state_dict")
    if issue:
        raise RuntimeError(f"Pending accumulation gradients are non-finite and cannot be resumed safely: {issue}")
    restored = 0
    clean_grad_state = {}
    for key, grad in grad_state.items():
        clean_key = str(key).replace("_orig_mod.", "")
        if clean_key in clean_grad_state:
            raise RuntimeError(
                f"Multiple saved gradient keys collapse to {clean_key!r}."
            )
        if not torch.is_tensor(grad):
            raise RuntimeError(
                f"Saved gradient {clean_key!r} is not a tensor."
            )
        clean_grad_state[clean_key] = grad
    if strict or expected_keys is not None:
        if not isinstance(expected_keys, (list, tuple)) or not expected_keys:
            raise RuntimeError(
                "Active gradient accumulation is missing its saved gradient-key manifest."
            )
        normalized_expected = {
            str(key).replace("_orig_mod.", "")
            for key in expected_keys
        }
        if len(normalized_expected) != len(expected_keys):
            raise RuntimeError(
                "Saved gradient-key manifest contains duplicate/colliding entries."
            )
        actual_keys = set(clean_grad_state)
        if actual_keys != normalized_expected:
            missing = sorted(normalized_expected - actual_keys)
            extra = sorted(actual_keys - normalized_expected)
            raise RuntimeError(
                "Saved gradient mapping is partial or inconsistent with its manifest: "
                f"missing={missing[:8]}, extra={extra[:8]}."
            )
    parameters = {
        name.replace("_orig_mod.", ""): param
        for name, param in model.named_parameters()
    }
    for clean_name, grad in clean_grad_state.items():
        param = parameters.get(clean_name)
        if param is None:
            raise RuntimeError(
                f"Saved gradient {clean_name!r} has no matching current parameter."
            )
        if tuple(grad.shape) != tuple(param.shape):
            raise RuntimeError(
                f"Saved gradient {clean_name!r} shape mismatch: "
                f"saved={tuple(grad.shape)}, current={tuple(param.shape)}."
            )
        param.grad = grad.to(device=device, dtype=param.dtype)
        restored += 1
    if restored:
        print(f"INFO: Restored {restored} pending gradient tensor(s) for accumulation parity.")
    return restored > 0


def restore_checkpoint_gradient_accumulation(
    model,
    checkpoint,
    args,
    device,
    *,
    scope: str = "Checkpoint",
):
    """Restore a declared partial accumulation window without silent gaps."""
    if not isinstance(checkpoint, dict):
        return False
    saved_grad_state = checkpoint.get("grad_state_dict")
    if reset_optimizer_state_requested(args):
        if saved_grad_state:
            print(
                f"INFO: Ignoring pending {scope.lower()} gradient accumulation "
                "because optimizer-state reset requested a fresh optimizer path."
            )
        return False

    declared_active = checkpoint.get("grad_accumulation_active")
    if declared_active is not None and not isinstance(declared_active, bool):
        raise RuntimeError(
            f"{scope} has a malformed grad_accumulation_active flag."
        )
    if saved_grad_state and declared_active is not True:
        raise RuntimeError(
            f"{scope} contains pending gradients but does not declare active "
            "gradient accumulation. Refusing an ambiguous resume."
        )
    if declared_active is not True:
        return False

    restored = restore_model_grad_state(
        model,
        saved_grad_state,
        device,
        expected_keys=checkpoint.get("grad_state_keys"),
        strict=True,
    )
    accumulation_state = checkpoint.get("accumulation_state")
    if not isinstance(accumulation_state, dict):
        raise RuntimeError(
            f"{scope} declares active gradient accumulation but has no "
            "accumulation_state mapping."
        )
    if (
        "normalization" not in accumulation_state
        or "weighted_token_mass" not in accumulation_state
    ):
        raise RuntimeError(
            f"{scope} active accumulation_state is missing normalization or "
            "weighted_token_mass metadata."
        )
    saved_normalization = str(accumulation_state["normalization"])
    current_normalization = str(
        getattr(args, "accumulation_normalization", "microbatch")
    )
    if saved_normalization != current_normalization:
        raise RuntimeError(
            f"Pending {scope.lower()} gradient accumulation normalization "
            f"mismatch: checkpoint={saved_normalization!r}, "
            f"current={current_normalization!r}."
        )
    try:
        args._accumulation_weighted_token_mass = float(
            accumulation_state["weighted_token_mass"]
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{scope} has malformed accumulated weighted-token mass."
        ) from exc
    if not math.isfinite(args._accumulation_weighted_token_mass):
        raise RuntimeError(
            f"{scope} has non-finite accumulated weighted-token mass."
        )
    if (
        saved_normalization == "weighted-token"
        and args._accumulation_weighted_token_mass <= 0.0
    ):
        raise RuntimeError(
            f"{scope} contains pending weighted-token gradients but no positive "
            "accumulated token mass."
        )
    return restored

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

def restore_rng_state(
    state,
    *,
    strict: bool = False,
    require_cuda: bool = False,
):
    if not state:
        if strict:
            raise RuntimeError(
                "Exact mid-epoch resume requires saved RNG state, but none was found."
            )
        return False
    if strict:
        required_keys = {"python", "numpy", "torch"}
        if not isinstance(state, dict) or not required_keys.issubset(state):
            raise RuntimeError(
                "Exact mid-epoch resume requires complete Python/NumPy/PyTorch "
                "RNG state."
            )
        if require_cuda and "cuda_all" not in state:
            raise RuntimeError(
                "Exact CUDA mid-epoch resume requires saved RNG state for every "
                "CUDA device."
            )
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if (require_cuda or torch.cuda.is_available()) and "cuda_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_all"])
        print("INFO: Restored RNG state from checkpoint.")
        return True
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "Could not restore RNG state for exact mid-epoch resume."
            ) from exc
        print(f"Warning: Could not restore RNG state: {exc}")
        return False

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
    for name, component in (
        ("dataloader", dataloader),
        ("batch_sampler", getattr(dataloader, "batch_sampler", None)),
        ("sampler", getattr(dataloader, "sampler", None)),
        ("dataset", getattr(dataloader, "dataset", None)),
    ):
        component_state = _capture_loader_component_state(component)
        if component_state:
            state[name] = component_state
    return state or None

def _restore_loader_component_state(component, state, *, strict: bool = False):
    if component is None or not state:
        if strict and state:
            raise RuntimeError(
                "Saved dataloader state has no matching runtime component."
            )
        return
    if not isinstance(state, dict):
        if strict:
            raise RuntimeError("Saved dataloader component state must be a mapping.")
        return
    saved_class = state.get("class")
    if (
        strict
        and saved_class
        and str(saved_class) != component.__class__.__name__
    ):
        raise RuntimeError(
            "Dataloader component class changed across exact resume: "
            f"saved={saved_class!r}, current={component.__class__.__name__!r}."
        )
    for attr in ("seed", "epoch"):
        if attr in state:
            if not hasattr(component, attr):
                if strict:
                    raise RuntimeError(
                        f"Saved dataloader {attr} has no runtime target."
                    )
                continue
            try:
                setattr(component, attr, int(state[attr]))
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"Could not restore dataloader {attr} for exact resume."
                    ) from exc
    generator = getattr(component, "generator", None)
    if "generator_state" in state:
        if not isinstance(generator, torch.Generator):
            if strict:
                raise RuntimeError(
                    "Saved dataloader generator state has no runtime generator target."
                )
            return
        try:
            generator.set_state(state["generator_state"])
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    "Could not restore dataloader generator state for exact resume."
                ) from exc

def restore_dataloader_state(dataloader, state, *, strict: bool = False):
    if dataloader is None or not state:
        if strict:
            raise RuntimeError(
                "Exact mid-epoch resume requires saved dataloader state."
            )
        return
    if not isinstance(state, dict):
        if strict:
            raise RuntimeError("Saved dataloader state must be a mapping.")
        return
    for name, component in (
        ("dataloader", dataloader),
        ("batch_sampler", getattr(dataloader, "batch_sampler", None)),
        ("sampler", getattr(dataloader, "sampler", None)),
        ("dataset", getattr(dataloader, "dataset", None)),
    ):
        saved_component_state = state.get(name)
        if (
            strict
            and _capture_loader_component_state(component) is not None
            and not isinstance(saved_component_state, dict)
        ):
            raise RuntimeError(
                f"Exact mid-epoch resume is missing saved {name} state."
            )
        _restore_loader_component_state(
            component,
            saved_component_state,
            strict=strict,
        )
    print("INFO: Restored dataloader sampler state from checkpoint.")


_EXACT_RESUME_SCHEDULE_KEYS = (
    "starting_lr",
    "min_lr",
    "warmup_steps",
    "warmup_ratio",
    "disable_lr_schedule",
    "ltm_lr",
    "min_ltm_lr",
    "disable_ltm_lr_schedule",
)


_EXACT_RESUME_ARG_KEYS = (
    "seed",
    "batch_size",
    "accumulation_steps",
    "accumulation_normalization",
    "max_length",
    "training_chunk_size",
    "full_sample_bptt",
    "full_sample_activation_checkpointing",
    "full_sample_checkpoint_segment_size",
    # Exact continuation includes numerical execution policy. These settings
    # can change precision, reduction order, optimizer implementation, or the
    # compiled recurrent control path even when the mathematical objective is
    # nominally unchanged.
    "_resolved_training_backend",
    "amp",
    "amp_dtype",
    "cuda_chunked_lm_loss",
    "cuda_loss_chunk_rows",
    "cpu_chunked_lm_loss",
    "cpu_loss_chunk_rows",
    "gradient_checkpointing",
    "compile",
    "force_compile",
    "compile_mode",
    "compile_static_worker_loop",
    "compile_pad_to_chunk_size",
    "detach_every_n_steps",
    "persist_state",
    "length_bucketing",
    "length_bucket_size",
    "streaming_datasets",
    "hf_streaming_shuffle_buffer",
    "hf_auto_shard",
    "train_prompt_tokens",
    "prompt_loss_weight",
    "response_loss_weight",
    "response_boundary_loss_weight",
    "response_boundary_tokens",
    "min_response_tokens",
    "drop_empty_completions",
    "strict_all_token_loss",
    # Optimizer and backward transforms are part of an exact continuation.
    # PyTorch's optimizer state does not encode external gradient clipping or
    # loss-component caps, so changing any of these while applying a saved
    # cursor/accumulation window would silently change the objective.
    "grad_clip",
    "max_ce_loss_for_backward",
    "max_ponder_cost_for_backward",
    "max_commitment_cost_for_backward",
    "rwkv_weight_decay",
    "startup_weight_max_abs",
    "ponder_loss_weight",
    "adaptive_ponder",
    "ponder_target_scale",
    "ponder_objective",
    "ponder_huber_beta",
    "encourage_thinking",
    "commitment_loss_weight",
    "commitment_threshold",
    "ltm_value_alignment_weight",
    "ltm_value_alignment_stride",
    "ltm_value_alignment_min_updates",
    "ltm_value_alignment_ready_threshold",
    "ltm_value_alignment_ema_decay",
    "ltm_value_writer_max_norm",
    # PEFT geometry/scaling must remain fixed for a true fine-tune resume.
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "finetune_unlock_percent",
    "alpaca",
    "kayla",
    "text_column",
    "prompt_column",
    "completion_column",
    "use_rosa",
    "rosa_max_context",
    "eval_tasks",
    "eval_limit",
    "best_checkpoint_metric",
    "best_checkpoint_mode",
    *_EXACT_RESUME_SCHEDULE_KEYS,
)


def _json_identity_digest(value):
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()


def _looks_like_hex_digest(value, lengths=(40, 64)):
    normalized = str(value or "").strip().lower()
    return (
        len(normalized) in set(lengths)
        and all(character in "0123456789abcdef" for character in normalized)
    )


def _infer_data_replay_guarantee(identity):
    """Classify whether an identity proves the ordered source is immutable."""
    if not isinstance(identity, dict):
        return "unproven-mutable-source"
    dataset_identity = identity.get("dataset")
    if isinstance(dataset_identity, dict):
        recorded = dataset_identity.get("replay_guarantee")
        if recorded:
            return str(recorded)
    else:
        dataset_identity = {}
    cache_identity = identity.get("token_cache")
    if (
        isinstance(cache_identity, dict)
        and _looks_like_hex_digest(
            cache_identity.get("ordered_record_sha256"),
            lengths=(64,),
        )
    ):
        return "content-addressed-token-cache"
    revision = dataset_identity.get("hf_dataset_revision")
    dataset_class = str(
        (identity.get("loader") or {}).get("dataset_class")
        if isinstance(identity.get("loader"), dict)
        else ""
    )
    if (
        _looks_like_hex_digest(revision)
        and not dataset_class.endswith(".PTChunkedDataset")
    ):
        return "immutable-hf-revision"
    return "unproven-mutable-source"


def build_exact_resume_identity(
    args,
    tokenizer,
    dataloader,
    dataloader_len,
    architecture_config=None,
):
    """Bind a cursor/optimizer checkpoint to the exact data objective."""
    objective = {
        key: (
            _normalize_detach_every_n_steps(getattr(args, key, None))
            if key == "detach_every_n_steps"
            else (
                getattr(args, "min_lr", None)
                if (
                    key == "min_ltm_lr"
                    and getattr(args, key, None) is None
                )
                else getattr(args, key, None)
            )
        )
        for key in _EXACT_RESUME_ARG_KEYS
    }
    dataset_object = (
        getattr(dataloader, "dataset", None)
        if dataloader is not None
        else None
    )
    dataset_class = (
        f"{type(dataset_object).__module__}.{type(dataset_object).__qualname__}"
        if dataset_object is not None
        else None
    )
    dataset = {
        "hf_dataset": getattr(args, "hf_dataset", None),
        "hf_dataset_config": getattr(args, "hf_dataset_config", None),
        "hf_dataset_split": getattr(args, "hf_dataset_split", None),
        "hf_dataset_revision": (
            getattr(args, "_resolved_hf_dataset_revision", None)
            or getattr(args, "hf_dataset_revision", None)
        ),
        "local_train": (
            os.path.abspath(os.path.expanduser(str(getattr(args, "train", ""))))
            if isinstance(getattr(args, "train", None), str)
            else None
        ),
        "pre_chunked_dataset": bool(getattr(args, "pre_chunked_dataset", False)),
        "pre_pt_dataset": bool(getattr(args, "pre_pt_dataset", False)),
    }
    tokenizer_identity = getattr(args, "_tokenizer_identity", None)
    if not isinstance(tokenizer_identity, dict):
        tokenizer_identity = {
            "vocab_size": int(len(tokenizer)) if tokenizer is not None else None,
            "source": getattr(args, "tokenizer_path", None),
        }
    cache_identity = getattr(args, "_token_cache_identity", None)
    loader_identity = {
        "dataloader_len": int(dataloader_len),
        "dataloader_class": (
            f"{type(dataloader).__module__}.{type(dataloader).__qualname__}"
            if dataloader is not None else None
        ),
        "dataset_class": dataset_class,
        "dataset_len": None,
        "iterable_dataset": bool(
            dataloader is not None
            and isinstance(getattr(dataloader, "dataset", None), IterableDataset)
        ),
        # Worker count is semantically irrelevant for deterministic map-style
        # token caches, but it changes sharding/interleaving for IterableDataset
        # inputs. Bind only the latter so an exact raw-stream resume cannot
        # silently consume a different record order while cached Colab resumes
        # remain portable across machines with different CPU counts.
        "iterable_num_workers": (
            int(getattr(dataloader, "num_workers", 0) or 0)
            if (
                dataloader is not None
                and isinstance(getattr(dataloader, "dataset", None), IterableDataset)
            )
            else None
        ),
    }
    try:
        loader_identity["dataset_len"] = int(len(dataloader.dataset))
    except Exception:
        pass
    identity = {
        "version": 1,
        "objective": objective,
        "dataset": dataset,
        "tokenizer": tokenizer_identity,
        "token_cache": cache_identity,
        "loader": loader_identity,
        "optimizer_grouping_version": int(
            getattr(args, "_optimizer_grouping_version", 2) or 2
        ),
    }
    dataset["replay_guarantee"] = _infer_data_replay_guarantee(identity)
    if architecture_config is not None:
        identity["architecture_contract"] = architecture_contract(architecture_config)
        identity["architecture_contract_sha256"] = architecture_contract_hash(
            architecture_config
        )
    identity["sha256"] = _json_identity_digest(identity)
    return identity


def _identity_mismatches(saved, current, path="run_identity"):
    mismatches = []
    if isinstance(saved, dict) and isinstance(current, dict):
        keys = sorted(set(saved) | set(current))
        for key in keys:
            if key == "sha256":
                continue
            mismatches.extend(_identity_mismatches(
                saved.get(key),
                current.get(key),
                f"{path}.{key}",
            ))
        return mismatches
    if saved != current:
        mismatches.append((path, saved, current))
    return mismatches


def validate_exact_resume_identity(
    checkpoint,
    current_identity,
    checkpoint_path,
    *,
    allow_schedule_rebuild=False,
):
    loader_identity = (
        current_identity.get("loader")
        if isinstance(current_identity, dict)
        else None
    )
    is_iterable = bool(
        isinstance(loader_identity, dict)
        and loader_identity.get("iterable_dataset")
    )
    mid_epoch_step = int(
        (checkpoint.get("mid_epoch_step", 0) or 0)
        if isinstance(checkpoint, dict)
        else 0
    )
    saved = checkpoint.get("run_identity") if isinstance(checkpoint, dict) else None
    data_replay_guarantee = _infer_data_replay_guarantee(current_identity)
    if mid_epoch_step > 0 and not isinstance(saved, dict):
        raise RuntimeError(
            "Exact mid-epoch resume cannot be proven because this legacy "
            "checkpoint has no saved run/data identity. Resume from an epoch "
            "boundary, or use --model-path for an intentional weights-only "
            "continuation."
        )
    if is_iterable and mid_epoch_step > 0:
        raise RuntimeError(
            "Exact mid-epoch resume cannot be proven for an IterableDataset: "
            "worker-local RNG state and prefetched shard cursors are not "
            "serializable by PyTorch DataLoader. Build/use the random-access "
            "token cache for paid resumable training, or use --model-path for "
            "an intentional weights-only continuation."
        )
    if (
        mid_epoch_step > 0
        and data_replay_guarantee
        not in {"content-addressed-token-cache", "immutable-hf-revision"}
    ):
        raise RuntimeError(
            "Exact mid-epoch resume cannot be proven for a mutable or "
            "content-unverified dataset source. Use a content-addressed "
            "random-access token cache (recommended), or an immutable pinned "
            "Hugging Face revision with an indexed map-style loader."
        )
    if not isinstance(saved, dict):
        if not allow_schedule_rebuild:
            saved_schedule = {}
            effective_config = checkpoint.get("effective_training_config")
            if isinstance(effective_config, dict):
                saved_schedule.update({
                    key: effective_config[key]
                    for key in _EXACT_RESUME_SCHEDULE_KEYS
                    if key in effective_config
                })
            main_state = checkpoint.get("lr_scheduler_state")
            if isinstance(main_state, dict):
                main_mapping = {
                    "starting_lr": main_state.get("max_lr"),
                    "min_lr": main_state.get("min_lr"),
                    "warmup_steps": main_state.get("warmup_steps"),
                    "warmup_ratio": main_state.get("warmup_ratio"),
                    "disable_lr_schedule": not bool(
                        main_state.get("enabled", True)
                    ),
                }
                for key, value in main_mapping.items():
                    if key not in saved_schedule and value is not None:
                        saved_schedule[key] = value
            ltm_state = checkpoint.get("ltm_scheduler_state")
            if isinstance(ltm_state, dict):
                ltm_mapping = {
                    "ltm_lr": ltm_state.get("max_lr"),
                    "min_ltm_lr": ltm_state.get("min_lr"),
                    "disable_ltm_lr_schedule": not bool(
                        ltm_state.get("enabled", True)
                    ),
                }
                for key, value in ltm_mapping.items():
                    if key not in saved_schedule and value is not None:
                        saved_schedule[key] = value
            current_objective = (
                current_identity.get("objective")
                if isinstance(current_identity, dict)
                else None
            )
            schedule_mismatches = []
            if isinstance(current_objective, dict):
                for key, old in saved_schedule.items():
                    current_value = current_objective.get(key)
                    if key == "min_ltm_lr" and current_value is None:
                        current_value = current_objective.get("min_lr")
                    if current_value != old:
                        schedule_mismatches.append(
                            (key, old, current_value)
                        )
            if schedule_mismatches:
                details = "; ".join(
                    f"{key}: saved={old!r}, current={new!r}"
                    for key, old, new in schedule_mismatches
                )
                raise RuntimeError(
                    f"Resume schedule mismatch for legacy checkpoint "
                    f"{checkpoint_path}: {details}. PyTorch scheduler state "
                    "does not serialize its lambda closure. Pass "
                    "--rebuild-lr-schedule for an intentional new curve."
                )
        print(
            "WARNING: Legacy checkpoint has no exact run identity. Dataset/cache/"
            "tokenizer compatibility cannot be proven; this one compatibility resume "
            "is allowed, but the new checkpoint will be identity-bound."
        )
        return False
    # Identities written immediately before replay-guarantee metadata can still
    # be classified from their already-saved token-cache digest/HF revision.
    saved_for_compare = dict(saved)
    saved_objective = dict(saved_for_compare.get("objective") or {})
    saved_effective_config = (
        checkpoint.get("effective_training_config")
        if isinstance(checkpoint, dict)
        else None
    )
    if isinstance(saved_effective_config, dict):
        # Checkpoints from the first run-identity schema stored schedule
        # semantics in effective_training_config but omitted them from the
        # identity objective. Promote that already-authenticated metadata only
        # in the comparison copy; the persisted self-digest remains untouched.
        for key in _EXACT_RESUME_SCHEDULE_KEYS:
            if key not in saved_objective and key in saved_effective_config:
                saved_objective[key] = saved_effective_config[key]
    if saved_objective.get("min_ltm_lr") is None:
        # Public None means "inherit min_lr"; compare the effective curve
        # rather than treating equivalent serialized spellings as drift.
        saved_objective["min_ltm_lr"] = saved_objective.get("min_lr")
    saved_for_compare["objective"] = saved_objective
    saved_dataset = dict(saved_for_compare.get("dataset") or {})
    saved_dataset.setdefault(
        "replay_guarantee",
        _infer_data_replay_guarantee(saved_for_compare),
    )
    saved_for_compare["dataset"] = saved_dataset
    current_for_compare = current_identity
    saved_tokenizer = saved_for_compare.get("tokenizer")
    current_tokenizer = (
        current_identity.get("tokenizer")
        if isinstance(current_identity, dict)
        else None
    )
    if (
        isinstance(saved_tokenizer, dict)
        and "behavior_sha256_v2" not in saved_tokenizer
        and isinstance(current_tokenizer, dict)
        and "behavior_sha256_v2" in current_tokenizer
    ):
        # Vocabulary-only tokenizer identities predate the behavior-level v2
        # digest. Preserve those checkpoints' established resume contract by
        # comparing the fields they actually authenticated. New v2 checkpoints
        # retain the field and therefore fail closed on any tokenizer-rule drift.
        current_for_compare = dict(current_identity)
        current_tokenizer = dict(current_tokenizer)
        current_tokenizer.pop("behavior_sha256_v2", None)
        current_for_compare["tokenizer"] = current_tokenizer
    mismatches = _identity_mismatches(saved_for_compare, current_for_compare)
    schedule_prefixes = tuple(
        f"run_identity.objective.{key}"
        for key in _EXACT_RESUME_SCHEDULE_KEYS
    ) + (
        # ltm_lr is also recorded in the learned-function contract because it
        # changes Titans fast-memory updates. An explicit schedule rebuild is
        # the one supported same-checkpoint operation allowed to change it.
        "run_identity.architecture_contract.ltm_lr",
        # The full contract remains field-by-field compared, so ignoring its
        # derived digest here cannot hide a non-schedule architecture change.
        "run_identity.architecture_contract_sha256",
    )
    if allow_schedule_rebuild:
        mismatches = [
            mismatch
            for mismatch in mismatches
            if not mismatch[0].startswith(schedule_prefixes)
        ]
    if mismatches:
        details = "; ".join(
            f"{path}: saved={old!r}, current={new!r}"
            for path, old, new in mismatches[:12]
        )
        if len(mismatches) > 12:
            details += f"; ... {len(mismatches) - 12} more"
        schedule_only = all(
            path.startswith(schedule_prefixes)
            for path, _old, _new in mismatches
        )
        recovery = (
            "Pass --rebuild-lr-schedule for an intentional new schedule, or "
            "use --model-path for a deliberate weights-only continuation."
            if schedule_only
            else
            "Use --model-path for a deliberate weights-only continuation."
        )
        raise RuntimeError(
            f"Exact resume identity mismatch for {checkpoint_path}: {details}. "
            "Refusing to apply a saved mid-epoch cursor/optimizer to a different "
            f"data objective. {recovery}"
        )
    if mid_epoch_step > 0:
        data_state = checkpoint.get("data_state")
        if not isinstance(data_state, dict) or not data_state:
            raise RuntimeError(
                "Exact mid-epoch resume requires a non-empty saved dataloader "
                f"state in {checkpoint_path}. Resume from an epoch boundary, or "
                "use --model-path for an intentional weights-only continuation."
            )
        rng_state = checkpoint.get("rng_state")
        required_rng_keys = {"python", "numpy", "torch"}
        if (
            not isinstance(rng_state, dict)
            or not required_rng_keys.issubset(rng_state)
        ):
            raise RuntimeError(
                "Exact mid-epoch resume requires complete Python/NumPy/PyTorch "
                f"RNG state in {checkpoint_path}. Resume from an epoch boundary, "
                "or use --model-path for an intentional weights-only continuation."
            )
    if is_iterable:
        print(
            "WARNING: IterableDataset identity/topology matched at an epoch "
            "boundary, but arbitrary worker-local stochastic replay cannot be "
            "proven. Use a random-access token cache when bit-exact data replay "
            "is required."
        )
        return False
    if data_replay_guarantee == "unproven-mutable-source":
        print(
            "WARNING: Run identity matched at an epoch boundary, but the "
            "dataset source is mutable or lacks a content digest. Exact data "
            "replay cannot be proven; use a content-addressed token cache for "
            "future resumable training."
        )
        return False
    print(
        "INFO: Exact resume identity verified: "
        f"{str(saved.get('sha256') or current_identity.get('sha256'))[:16]}."
    )
    return True


def capture_effective_training_config(args):
    keys = set(_EXACT_RESUME_ARG_KEYS) | {
        "starting_lr",
        "min_lr",
        "warmup_steps",
        "warmup_ratio",
        "disable_lr_schedule",
        "ltm_lr",
        "min_ltm_lr",
        "disable_ltm_lr_schedule",
        "rwkv_weight_decay",
        "max_skipped_train_batches",
    }
    effective = {}
    for key in sorted(keys):
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if key == "detach_every_n_steps":
            value = _normalize_detach_every_n_steps(value)
        effective[key] = value
    return effective


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
    model_architecture_contract = architecture_contract(model.config)
    model_architecture_hash = architecture_contract_hash(model.config)
    saved_config = dict(model.config)
    saved_config["architecture_contract_sha256"] = model_architecture_hash
    checkpoint = {
        "checkpoint_version": 4,
        "checkpoint_kind": "training",
        "completed_epoch": int(completed_epoch),
        "mid_epoch_step": int(mid_epoch_step or 0),
        "model_state_dict": sanitize_model_state_dict(model, reset_transient_ltm=False),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "lr_scheduler_state": capture_main_lr_scheduler_state(args, scheduler),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": saved_config,
        "architecture_contract": model_architecture_contract,
        "architecture_contract_sha256": model_architecture_hash,
        "rng_state": capture_rng_state(),
        "data_state": capture_dataloader_state(dataloader),
        "grad_state_dict": grad_state,
        "grad_accumulation_active": bool(grad_state),
        "grad_state_keys": tuple(sorted(grad_state)) if grad_state else (),
        "ltm_scheduler_state": capture_ltm_lr_scheduler_state(args),
        "run_identity": getattr(args, "_run_identity", None),
        "effective_training_config": capture_effective_training_config(args),
        "optimizer_grouping_version": int(
            getattr(args, "_optimizer_grouping_version", 2) or 2
        ),
        "accumulation_state": {
            "normalization": str(
                getattr(args, "accumulation_normalization", "microbatch")
            ),
            "weighted_token_mass": float(
                getattr(args, "_accumulation_weighted_token_mass", 0.0) or 0.0
            ),
        },
        "error_budget_state": {
            "skipped_train_batches": int(
                getattr(args, "_skipped_train_batches", 0) or 0
            ),
        },
        "best_metric_state": getattr(args, "_best_metric_state", None),
        "training_complete": False,
    }
    # Independent shuffled samples never consume the terminal recurrent/ROSA
    # carrier from the previous batch. Avoid copying and serializing that state
    # into periodic checkpoints when the next step is guaranteed to reset it.
    # Keep the historical builder behavior for callers that do not expose a
    # persist_state setting, and retain exact state for explicitly contiguous
    # streams.
    save_running_states = running_states is not None and (
        not hasattr(args, "persist_state")
        or bool(getattr(args, "persist_state", False))
    )
    if save_running_states:
        checkpoint["running_states"] = _state_to_cpu(running_states)
    return checkpoint


def initialize_best_metric_tracker(args, checkpoint=None):
    selector = getattr(args, "best_checkpoint_metric", None)
    if not selector:
        args._best_metric_tracker = None
        args._best_metric_state = None
        return None

    mode = getattr(args, "best_checkpoint_mode", "max")
    tracker = BestMetric(selector, mode=mode)
    saved_state = (
        checkpoint.get("best_metric_state")
        if isinstance(checkpoint, dict)
        else None
    )
    if isinstance(saved_state, dict):
        restored = BestMetric.from_state_dict(saved_state)
        if (
            restored.selector != tracker.selector
            or restored.mode != tracker.mode
        ):
            raise RuntimeError(
                "Best-checkpoint selector changed across exact resume: "
                f"saved={restored.selector!r}/{restored.mode}, "
                f"current={tracker.selector!r}/{tracker.mode}."
            )
        tracker = restored
    args._best_metric_tracker = tracker
    args._best_metric_state = tracker.state_dict()
    return tracker


def save_best_checkpoint_if_improved(
    eval_results,
    *,
    args,
    model,
    optimizer,
    scheduler,
    scaler,
    dataloader,
    completed_epoch: int,
    mid_epoch_step: int,
    running_states=None,
):
    """Persist an exact-resume checkpoint for a strictly improved eval metric."""
    tracker = getattr(args, "_best_metric_tracker", None)
    if tracker is None:
        return False
    candidate = extract_selection_metric(eval_results, tracker.selector)
    if not tracker.update(
        candidate,
        epoch=int(completed_epoch),
        step=(
            int(mid_epoch_step)
            if int(mid_epoch_step or 0) > 0
            else None
        ),
    ):
        return False

    args._best_metric_state = tracker.state_dict()
    best_path = os.path.join(args.out_dir, "hierarchos_best.pt")
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        args,
        dataloader,
        completed_epoch=completed_epoch,
        mid_epoch_step=mid_epoch_step,
        running_states=running_states,
    )
    checkpoint["checkpoint_kind"] = "best-training"
    checkpoint["selection_metric"] = tracker.state_dict()
    checkpoint["evaluation_results"] = eval_results
    save_training_checkpoint_if_finite(
        checkpoint,
        best_path,
        model,
        optimizer,
    )
    print(
        "INFO: New best checkpoint: "
        f"{tracker.selector}={candidate:.8g} ({tracker.mode}) -> {best_path}"
    )
    return True


def train_step_skip_reason(args):
    """Return the canonical reason a batch failed to produce a valid backward."""
    if getattr(args, "_train_step_had_nonfinite", False):
        return "nonfinite loss/gradient"
    if getattr(args, "_train_step_had_oom", False):
        return "out of memory"
    if getattr(args, "_train_step_had_empty_supervision", False):
        return "empty supervised-token mass"
    if not getattr(args, "_train_step_had_backward", False):
        return "no backward-producing chunk"
    return None


def account_skipped_training_batch(
    args,
    *,
    reason,
    epoch: int,
    step: int,
    scope: str = "Training",
):
    """Consume the shared fail-closed batch skip budget or raise."""
    if reason is None:
        return False
    args._accumulation_weighted_token_mass = 0.0
    args._skipped_train_batches = int(
        getattr(args, "_skipped_train_batches", 0) or 0
    ) + 1
    max_skipped = max(
        0,
        int(getattr(args, "max_skipped_train_batches", 0) or 0),
    )
    if args._skipped_train_batches > max_skipped:
        raise RuntimeError(
            f"{scope} skip/error budget exceeded at "
            f"epoch={int(epoch)}, step={int(step)}: {reason}. "
            f"Observed {args._skipped_train_batches}, allowed {max_skipped}."
        )
    print(
        f"WARNING: {scope} batch skipped within explicit budget "
        f"({args._skipped_train_batches}/{max_skipped}): {reason}."
    )
    return True


def train_step(model, batch, optimizer, scaler, accumulation_steps, step, args, running_states,
               collect_metrics=True, force_optimizer_step=False, accumulation_divisor=None):
    """Training step with temporal chunking to match original hierarchos.py."""
    args._optimizer_step_was_taken = False
    args._train_step_had_backward = False
    args._train_step_had_nonfinite = False
    args._train_step_had_oom = False
    args._train_step_had_empty_supervision = False
    had_accumulated_gradients_on_entry = _has_pending_gradients(model)
    device = next(model.parameters()).device
    set_model_training_step(model, getattr(args, '_current_global_step', step))
    _nb = (device.type == 'cuda')  # non_blocking for async CUDA transfer
    full_input_ids = batch['input_ids']
    full_attention_mask = batch.get('attention_mask')
    full_labels = batch['labels']
    full_rosa_ids = batch.get('rosa_ids')
    full_rosa_context_mode = batch.get("rosa_ids_context_mode")
    full_loss_weights = batch.get('loss_weights')
    accumulation_normalization = str(
        getattr(args, "accumulation_normalization", "microbatch")
    )
    if accumulation_normalization == "weighted-token":
        batch_weight_mass = supervised_weight_mass(batch)
        if batch_weight_mass <= 0.0:
            args._train_step_had_empty_supervision = True
            if _has_pending_gradients(model):
                raise RuntimeError(
                    "Encountered an empty-supervision microbatch inside an active "
                    "accumulation window. Refusing to discard or renormalize earlier "
                    "valid gradients."
                )
            return None, running_states
        args._accumulation_weighted_token_mass = float(
            getattr(args, "_accumulation_weighted_token_mass", 0.0) or 0.0
        ) + batch_weight_mass
        loss_divisor = 1.0 / batch_weight_mass
    elif accumulation_normalization == "microbatch":
        loss_divisor = float(max(1, int(
            accumulation_divisor or accumulation_steps or 1
        )))
    else:
        raise ValueError(
            "accumulation_normalization must be 'weighted-token' or 'microbatch'"
        )
    full_input_ids, full_labels, full_attention_mask = trim_trailing_padding(
        full_input_ids, full_labels, full_attention_mask
    )
    if full_rosa_ids is not None:
        full_rosa_ids = full_rosa_ids[:, :full_input_ids.shape[1]].contiguous()
    if full_loss_weights is not None:
        full_loss_weights = full_loss_weights[:, :full_input_ids.shape[1]].contiguous()
    chunk_size = _resolved_training_chunk_size(
        args,
        getattr(model, "config", None),
    )
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

    # Non-reentrant activation checkpointing may rematerialize a segment during
    # backward. Live ROSA mutates a Python automaton and therefore is not a pure
    # replay input. Materialize the exact deterministic prediction stream once,
    # including when an old/mismatched cache marker would make the core fall
    # back to live ROSA. Ordinary TBPTT and non-checkpointed full BPTT retain the
    # asynchronous incremental path.
    checkpointed_full_sample = bool(
        getattr(args, "full_sample_bptt", False)
        and getattr(args, "full_sample_activation_checkpointing", False)
    )
    model_uses_rosa = bool(
        getattr(
            model,
            "use_rosa",
            getattr(getattr(model, "config", None), "use_rosa", False),
        )
    )
    if checkpointed_full_sample and model_uses_rosa:
        expected_rosa_context_mode = rosa_context_mode(
            bool(
                getattr(
                    getattr(model, "config", None),
                    "enforce_rosa_max_context",
                    False,
                )
            )
        )
        if (
            full_rosa_ids is None
            or full_rosa_context_mode != expected_rosa_context_mode
        ):
            full_rosa_ids, full_rosa_context_mode = (
                _checkpoint_replay_safe_rosa_ids(model, full_input_ids)
            )

    # Validate the immutable batch contract once while it is still on CPU.
    # The recurrent core otherwise has to materialize validation scalars back on
    # the host for every accelerator TBPTT chunk. Long samples can contain dozens
    # of chunks, so those synchronizations noticeably reduce utilization. Direct
    # model calls and non-CPU batches retain the core's ordinary fail-closed audit.
    prevalidated_active_lengths = None
    validation_tensors = (
        full_input_ids,
        full_attention_mask,
        full_labels,
        full_loss_weights,
    )
    can_prevalidate_on_host = all(
        value is None
        or (
            torch.is_tensor(value)
            and value.device.type == "cpu"
        )
        for value in validation_tensors
    )
    if can_prevalidate_on_host:
        _validate_sequence_mask_contract(
            full_input_ids,
            full_attention_mask,
            full_labels,
            full_loss_weights,
        )
        if full_attention_mask is not None:
            prevalidated_active_lengths = full_attention_mask.to(
                dtype=torch.long
            ).sum(dim=1)
    
    # A malformed floating-point label tensor must reject the whole
    # microbatch. In particular, never clear valid gradients from an earlier
    # accumulation microbatch and then continue with a changed objective.
    if (
        full_labels.is_floating_point()
        and not bool(torch.isfinite(full_labels).all().item())
    ):
        print(
            f"\nCRITICAL: Non-finite labels detected at step {step + 1}; "
            "rejecting the complete batch."
        )
        args._train_step_had_nonfinite = True
        return _reject_nonfinite_train_batch(
            optimizer,
            model,
            had_accumulated_gradients_on_entry=(
                had_accumulated_gradients_on_entry
            ),
        )
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
    full_recurrent_carrier_checks = []
    writer_alignment_score = None
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

            prevalidated_mask_metadata = None
            if can_prevalidate_on_host:
                chunk_length = end_t - start_t
                if prevalidated_active_lengths is None:
                    first_padding_column = chunk_length
                else:
                    local_active_lengths = (
                        prevalidated_active_lengths - start_t
                    ).clamp(min=0, max=chunk_length)
                    first_padding_column = int(
                        local_active_lengths.min().item()
                    )
                prevalidated_mask_metadata = (
                    first_padding_column < chunk_length,
                    first_padding_column,
                )
            
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
                    return_step_telemetry=False,
                    return_numerics=False,
                    compute_ltm_value_alignment=ltm_value_alignment_weight > 0.0,
                    rosa_ids=rosa_ids,
                    rosa_ids_context_mode=full_rosa_context_mode,
                    # Cached predictions are complete for this sample. Advancing
                    # a duplicate CPU token history cannot alter the forward and
                    # adds one D2H transfer plus concatenation per TBPTT chunk.
                    advance_cached_rosa_history=rosa_ids is None,
                    loss_weights=loss_weights,
                    _prevalidated_mask_metadata=prevalidated_mask_metadata,
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
                if ltm_value_alignment_cost is not None:
                    weighted_writer_score = (
                        ltm_value_alignment_cost.detach().float() * token_ratio
                    )
                    writer_alignment_score = (
                        weighted_writer_score
                        if writer_alignment_score is None
                        else writer_alignment_score + weighted_writer_score
                    )
                recurrent_carrier_checks = _recurrent_carrier_finite_checks(outputs)

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
                    # Keep only scalar device-side audits, not extra references to
                    # checkpointed activations. One aggregate host check runs before
                    # backward after every segment has been assembled.
                    full_recurrent_carrier_checks.extend(
                        (
                            f"chunk {chunk_idx} {name}",
                            finite_check,
                        )
                        for name, _value, finite_check in recurrent_carrier_checks
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
                    ] + [
                        finite_check
                        for _name, _value, finite_check in recurrent_carrier_checks
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
                            aux_loss = aux_loss + (
                                _adaptive_ponder_loss(
                                    args,
                                    model,
                                    ponder_cost_for_backward,
                                    ce_loss,
                                )
                                * ponder_weight
                            )
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
                            for (name, _value), flag in zip(
                                finite_components,
                                flag_values[:len(finite_components)],
                            )
                        }
                        print(
                            f"\nCRITICAL: Non-finite training trajectory at step {step+1}, "
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
                        carrier_flag_start = len(finite_components)
                        carrier_flag_values = flag_values[
                            carrier_flag_start:
                            carrier_flag_start + len(recurrent_carrier_checks)
                        ]
                        for (name, value, _finite_check), flag in zip(
                            recurrent_carrier_checks,
                            carrier_flag_values,
                        ):
                            if not bool(flag):
                                print("  " + _describe_tensor_issue(name, value))
                        if not bool(flag_values[-1]):
                            print("  " + _describe_tensor_issue("chunk_loss", chunk_loss))
                        print("  Skipping this batch and clearing recurrent/LTM state before it can poison optimizer state.")
                        args._train_step_had_nonfinite = True
                        return _reject_nonfinite_train_batch(
                            optimizer,
                            model,
                            had_accumulated_gradients_on_entry=(
                                had_accumulated_gradients_on_entry
                            ),
                        )

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
                        return _reject_nonfinite_train_batch(
                            optimizer,
                            model,
                            had_accumulated_gradients_on_entry=(
                                had_accumulated_gradients_on_entry
                            ),
                        )

                    if ltm_grads_tensor is not None:
                        # Unpack current LTM state for the update
                        curr_ltm = outputs.get('ltm_memory_state')
                        curr_fast = curr_ltm[0] if curr_ltm is not None else None
                        curr_mom = curr_ltm[1] if curr_ltm is not None else None
                        curr_past_tokens = curr_ltm[2] if curr_ltm is not None and len(curr_ltm) >= 3 else None
                        curr_rosa_states = curr_ltm[3] if curr_ltm is not None and len(curr_ltm) >= 4 else None
                        curr_timestamps = curr_ltm[4] if curr_ltm is not None and len(curr_ltm) >= 5 else None
                        curr_sources = curr_ltm[5] if curr_ltm is not None and len(curr_ltm) >= 6 else None
                        curr_wallclock_timestamps = curr_ltm[6] if curr_ltm is not None and len(curr_ltm) >= 7 else None
                        
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
                            wallclock_timestamps=curr_wallclock_timestamps,
                            inplace=True
                        )
                        ltm_state = (new_fast.detach(), new_mom.detach(), 
                                     curr_past_tokens.detach() if curr_past_tokens is not None else None,
                                     curr_rosa_states,
                                     curr_timestamps.detach() if isinstance(curr_timestamps, torch.Tensor) else curr_timestamps,
                                     curr_sources.detach() if isinstance(curr_sources, torch.Tensor) else curr_sources,
                                     curr_wallclock_timestamps.detach() if isinstance(curr_wallclock_timestamps, torch.Tensor) else curr_wallclock_timestamps)  # ROSA automaton states (plain Python, no detach needed)
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
                    aux_loss = aux_loss + (
                        _adaptive_ponder_loss(
                            args,
                            model,
                            ponder_cost_for_backward,
                            whole_ce_loss,
                        )
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
            component_finite_checks = [
                torch.isfinite(value).all()
                if value is not None else torch.tensor(False, device=device)
                for _name, value in finite_components
            ]
            all_loss_checks = (
                component_finite_checks
                + [check for _name, check in full_recurrent_carrier_checks]
                + [torch.isfinite(whole_sample_loss).all()]
            )
            losses_valid = bool(torch.stack(all_loss_checks).all().item())
            if not losses_valid:
                flag_values = torch.stack(all_loss_checks).tolist()
                component_flags = {
                    name: bool(flag)
                    for (name, _value), flag in zip(
                        finite_components,
                        flag_values[:len(finite_components)],
                    )
                }
                print(
                    f"\nCRITICAL: Non-finite full-sample training trajectory at step {step+1}."
                )
                if not component_flags.get("ce", True):
                    print("  " + _describe_tensor_issue("cross_entropy_loss", whole_ce_loss))
                if not component_flags.get("ponder", True):
                    print("  " + _describe_tensor_issue("ponder_cost", whole_ponder_cost))
                if not component_flags.get("commitment", True):
                    print("  " + _describe_tensor_issue("commitment_cost", whole_commitment_cost))
                if not component_flags.get("alignment", True):
                    print("  " + _describe_tensor_issue("ltm_value_alignment_cost", whole_ltm_alignment_cost))
                carrier_flag_start = len(finite_components)
                carrier_flag_values = flag_values[
                    carrier_flag_start:
                    carrier_flag_start + len(full_recurrent_carrier_checks)
                ]
                for (name, _finite_check), flag in zip(
                    full_recurrent_carrier_checks,
                    carrier_flag_values,
                ):
                    if not bool(flag):
                        print(f"  {name} contains NaN or Inf")
                if not bool(flag_values[-1]):
                    print("  " + _describe_tensor_issue("whole_sample_loss", whole_sample_loss))
                print("  Skipping this batch and clearing recurrent/LTM state before it can poison optimizer state.")
                args._train_step_had_nonfinite = True
                return _reject_nonfinite_train_batch(
                    optimizer,
                    model,
                    had_accumulated_gradients_on_entry=(
                        had_accumulated_gradients_on_entry
                    ),
                )

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
            full_recurrent_carrier_checks.clear()

        # Optimizer step after all chunks
        should_step_optimizer = ((step + 1) % accumulation_steps == 0) or bool(force_optimizer_step)
        if should_step_optimizer and _has_pending_gradients(model):
            if scaler is not None:
                scaler.unscale_(optimizer)
                if accumulation_normalization == "weighted-token":
                    _divide_pending_gradients_(
                        model,
                        getattr(args, "_accumulation_weighted_token_mass", 0.0),
                    )
                grads_ok, grad_issue = _clip_gradients_and_check(
                    model,
                    getattr(args, 'grad_clip', 1.0),
                    getattr(args, 'max_sanitized_gradient_values', 0),
                )
                if not grads_ok:
                    print(f"\nCRITICAL: Non-finite gradient at step {step+1}. {grad_issue}")
                    print("  Skipping optimizer step and clearing accumulated gradients.")
                    args._train_step_had_nonfinite = True
                    return _reject_nonfinite_train_batch(
                        optimizer,
                        model,
                        had_accumulated_gradients_on_entry=(
                            had_accumulated_gradients_on_entry
                        ),
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if accumulation_normalization == "weighted-token":
                    _divide_pending_gradients_(
                        model,
                        getattr(args, "_accumulation_weighted_token_mass", 0.0),
                    )
                grads_ok, grad_issue = _clip_gradients_and_check(
                    model,
                    getattr(args, 'grad_clip', 1.0),
                    getattr(args, 'max_sanitized_gradient_values', 0),
                )
                if not grads_ok:
                    print(f"\nCRITICAL: Non-finite gradient at step {step+1}. {grad_issue}")
                    print("  Skipping optimizer step and clearing accumulated gradients.")
                    args._train_step_had_nonfinite = True
                    return _reject_nonfinite_train_batch(
                        optimizer,
                        model,
                        had_accumulated_gradients_on_entry=(
                            had_accumulated_gradients_on_entry
                        ),
                    )
                optimizer.step()
            if ltm_value_alignment_weight > 0.0:
                mark_val_proj_trained(
                    model,
                    alignment_cost=writer_alignment_score,
                )
            optimizer.zero_grad(set_to_none=True)
            args._accumulation_weighted_token_mass = 0.0
            args._optimizer_step_was_taken = True
        elif should_step_optimizer:
            optimizer.zero_grad(set_to_none=True)
            args._accumulation_weighted_token_mass = 0.0
        
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
            
    except FloatingPointError as e:
        print(
            f"\nCRITICAL: Non-finite model trajectory at training step {step + 1}: "
            f"{e}"
        )
        print(
            "  Rejecting the complete batch; non-finite intermediates are never "
            "rewritten into apparently valid recurrent states."
        )
        args._train_step_had_nonfinite = True
        return _reject_nonfinite_train_batch(
            optimizer,
            model,
            had_accumulated_gradients_on_entry=(
                had_accumulated_gradients_on_entry
            ),
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if had_accumulated_gradients_on_entry:
                raise RuntimeError(
                    "An out-of-memory failure occurred after valid gradients "
                    "had already accumulated from an earlier microbatch. "
                    "Refusing to clear them and silently change the "
                    "accumulation objective; reduce memory use or resume from "
                    "the last verified checkpoint."
                ) from e
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
            args._train_step_had_oom = True
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

    explicit_dests = set(
        getattr(args, "_explicit_cli_dests", ()) or ()
    )
    amp_was_explicitly_set = "amp" in explicit_dests or any(
        value in sys.argv for value in ("--amp", "--no-amp", "--no_amp")
    )
    if not amp_was_explicitly_set:
        args.amp = True
        config.amp = True
        print(f"INFO: AMP auto-enabled for CUDA {mode_name} (use --no-amp to disable).")

    compile_was_explicitly_disabled = (
        "compile" in explicit_dests and not bool(getattr(args, "compile", False))
    ) or any(
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
    if (
        bool(getattr(args, "persist_state", False))
        and not bool(getattr(args, "full_sample_bptt", False))
        and not bool(
            getattr(
                getattr(dataloader, "dataset", None),
                "guarantees_contiguous_state",
                False,
            )
        )
    ):
        raise ValueError(
            "persist_state requires a dataset that explicitly guarantees contiguous "
            "same-sequence lane ordering; shuffled independent samples are unsafe."
        )
    print("Running in TRAIN mode...")
    if dataloader_len <= 0:
        print("ERROR: dataloader_len must be > 0. If automatic detection failed, please specify --dataset-size.")
        return
    
    
    if getattr(args, 'out_dir', None):
        os.makedirs(args.out_dir, exist_ok=True)
    args._resolved_training_backend = (
        "directml" if is_directml_device(device) else str(device.type)
    )
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
            f"chunks of {_resolved_training_chunk_size(args, config)} and recurrent detach "
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

    _compile_was_explicitly_disabled = (
        "compile" in set(getattr(args, "_explicit_cli_dests", ()) or ())
        and not bool(getattr(args, "compile", False))
    ) or any(a in sys.argv for a in ('--no-compile', '--no_compile'))
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
        if validate_inference_tokenizer_identity(
            tokenizer,
            getattr(model, "_hierarchos_checkpoint_metadata", None),
        ):
            print("INFO: Preloaded-model tokenizer content fingerprint verified.")
        else:
            print(
                "WARNING: Preloaded model has no tokenizer content fingerprint; "
                "only vocabulary-size compatibility can be checked."
            )
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
            'min_h_steps',
            'architecture_revision',
            'core_recurrence_version',
            'drift_recurrence_mode',
            'rwkv_state_readout_mode',
            'manager_state_commit_mode',
            'manager_compute_mode',
            'commitment_cost_mode',
            'use_deepembed',
            'deepembed_mode',
            'use_rosa',
            'rosa_embedding_mode',
            'token_adapter_rank',
            'memory_token_routers',
            'rosa_max_context',
            'enforce_rosa_max_context',
            'rosa_zero_no_prediction',
            'rwkv_head_size',
            'h_rwkv_head_size',
            'l_rwkv_head_size',
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
        args._optimizer_grouping_version = int(
            checkpoint.get("optimizer_grouping_version", 1) or 1
        )
        if args._optimizer_grouping_version <= 1:
            print(
                "INFO: Exact resume preserves legacy optimizer grouping v1. "
                "Use --model-path for a new run with corrected matrix grouping v2."
            )
        
        saved_config = checkpoint.get('config', {})
        model_config = AttrDict(saved_config)
        state_dict = sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False)
        _reject_unsupported_rwkv_state_dict(state_dict, args.resume_from_ckpt)
        checkpoint['model_state_dict'] = state_dict
        _infer_arch_flags_from_state_dict(model_config, state_dict)
        validate_checkpoint_architecture_contract(
            checkpoint,
            model_config,
            args.resume_from_ckpt,
        )
        
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
            model_config.h_rwkv_head_size = state_dict['h_rnn.r_k'].shape[1]
        if 'l_rnn.r_k' in state_dict_keys:
            model_config.l_rwkv_head_size = state_dict['l_rnn.r_k'].shape[1]
        if (
            model_config.get("h_rwkv_head_size") is not None
            and model_config.get("h_rwkv_head_size")
            == model_config.get("l_rwkv_head_size")
        ):
            model_config.rwkv_head_size = model_config.h_rwkv_head_size
        else:
            model_config.rwkv_head_size = None

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
        
        if not reset_optimizer_state_requested(args):
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
            if not reset_optimizer_state_requested(args):
                _restore_resume_component_state(
                    scaler,
                    checkpoint.get('scaler_state_dict'),
                    "AMP scaler",
                    args.resume_from_ckpt,
                )
    
    elif args.model_path:
        print(f"Loading base model from: {args.model_path}")
        model, model_config = load_full_model_with_config(args.model_path, device)
        if validate_inference_tokenizer_identity(
            tokenizer,
            getattr(model, "_hierarchos_checkpoint_metadata", None),
        ):
            print("INFO: Base-model tokenizer content fingerprint verified.")
        else:
            print(
                "WARNING: Legacy base model has no tokenizer content fingerprint; "
                "only vocabulary-size compatibility can be checked."
            )
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
            'min_h_steps',
            'architecture_revision',
            'core_recurrence_version',
            'drift_recurrence_mode',
            'rwkv_state_readout_mode',
            'manager_state_commit_mode',
            'manager_compute_mode',
            'commitment_cost_mode',
            'use_deepembed',
            'deepembed_mode',
            'use_rosa',
            'rosa_embedding_mode',
            'token_adapter_rank',
            'memory_token_routers',
            'rosa_max_context',
            'enforce_rosa_max_context',
            'rosa_zero_no_prediction',
            'rwkv_head_size',
            'h_rwkv_head_size',
            'l_rwkv_head_size',
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
    continuing_loaded_weights = bool(
        model_override is not None
        or getattr(args, "resume_from_ckpt", None)
        or getattr(args, "model_path", None)
    )
    args._training_step_offset = 0
    if continuing_loaded_weights:
        next_local_step = (
            int(start_epoch) * int(dataloader_len)
            + int(start_step)
        )
        saved_training_step = get_model_training_step(model)
        if saved_training_step is None:
            print(
                "WARNING: Loaded model has no persisted memory-gate curriculum "
                "step; continuation starts from the local session step."
            )
        else:
            args._training_step_offset = resolve_training_step_offset(
                model,
                next_local_step,
            )
            print(
                "INFO: Memory-gate curriculum continuation: "
                f"saved_step={saved_training_step}, "
                f"next_step={args._training_step_offset + next_local_step}."
            )

    # --- [NEW] Sync LTM reference chunk size (Parity Fix) ---
    training_chunk_size = _resolved_training_chunk_size(
        args,
        getattr(model, "config", None),
    )
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
    if (
        checkpoint
        and not checkpoint.get("run_identity")
        and "accumulation_normalization"
        not in set(getattr(args, "_explicit_cli_dests", ()) or ())
    ):
        args.accumulation_normalization = "microbatch"
        print(
            "INFO: Legacy checkpoint preserves equal-microbatch accumulation. "
            "New runs default to weighted-token normalization."
        )
    args._run_identity = build_exact_resume_identity(
        args,
        tokenizer,
        dataloader,
        dataloader_len,
        architecture_config=getattr(model, "config", None),
    )
    if checkpoint:
        validate_exact_resume_identity(
            checkpoint,
            args._run_identity,
            args.resume_from_ckpt,
            allow_schedule_rebuild=rebuild_lr_schedule_requested(args),
        )
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
        chunk_size=_resolved_training_chunk_size(
            args,
            getattr(model, "config", None),
        ),
    )

    # Scheduler
    # A schedule rebuild is independent from optimizer-moment reset. The legacy
    # --override-scheduling flag requests both through the compatibility helpers.
    full_run_update_steps = compute_update_steps(dataloader_len, args.accumulation_steps) * args.epochs
    saved_main_lr_state = checkpoint.get('lr_scheduler_state') if isinstance(checkpoint, dict) else None
    saved_ltm_lr_state = checkpoint.get('ltm_scheduler_state') if isinstance(checkpoint, dict) else None
    if rebuild_lr_schedule_requested(args) and args.resume_from_ckpt:
        num_update_steps = compute_remaining_update_steps(
            dataloader_len,
            args.accumulation_steps,
            start_epoch,
            args.epochs,
            start_step,
        )
        print(f"INFO: Rebuilding LR schedule for remaining work ({num_update_steps} update steps)")
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
        if args.resume_from_ckpt and not rebuild_lr_schedule_requested(args):
            print(
                "INFO: Resume scheduler using checkpoint/full-run schedule state. "
                "Pass --rebuild-lr-schedule to rebuild LR decay over only the remaining work."
            )
    args._main_lr_schedule_total_steps = int(num_update_steps)

    resume_main_schedule_state = (
        saved_main_lr_state
        if (
            args.resume_from_ckpt
            and not rebuild_lr_schedule_requested(args)
            and isinstance(saved_main_lr_state, dict)
        )
        else None
    )
    main_schedule_enabled = (
        bool(resume_main_schedule_state.get("enabled", True))
        if resume_main_schedule_state is not None
        else not bool(getattr(args, 'disable_lr_schedule', False))
    )
    if main_schedule_enabled and num_update_steps > 0:
        scheduler = build_lr_scheduler(
            optimizer,
            args,
            num_update_steps,
            resume_schedule_state=resume_main_schedule_state,
        )
        if args.resume_from_ckpt and not rebuild_lr_schedule_requested(args):
            restore_scheduler_state_and_live_lrs(
                scheduler,
                optimizer,
                checkpoint.get('scheduler_state_dict'),
                args.resume_from_ckpt,
            )
    configure_ltm_lr_schedule(
        args,
        num_update_steps,
        checkpoint=checkpoint,
        override_schedule=rebuild_lr_schedule_requested(args),
        scheduler=scheduler,
    )
    if hasattr(model, 'config'):
        model.config.ltm_lr = float(getattr(args, '_ltm_lr_max', 1e-3))
        model.config.min_ltm_lr = float(getattr(args, '_ltm_lr_min', 0.0))
        model.config.disable_ltm_lr_schedule = not bool(
            getattr(args, '_ltm_lr_schedule_enabled', True)
        )
    args._accumulation_weighted_token_mass = 0.0
    if checkpoint:
        exact_mid_epoch = start_step > 0
        restore_dataloader_state(
            dataloader,
            checkpoint.get('data_state'),
            strict=exact_mid_epoch,
        )
        restore_rng_state(
            checkpoint.get('rng_state'),
            strict=exact_mid_epoch,
            require_cuda=exact_mid_epoch and device.type == "cuda",
        )
        restored_pending_grads = restore_checkpoint_gradient_accumulation(
            model,
            checkpoint,
            args,
            device,
            scope="Checkpoint",
        )
        exact_running_states = validate_exact_running_states(
            checkpoint,
            args,
            start_step,
            args.resume_from_ckpt,
        )
        if not exact_running_states and checkpoint.get('running_states') is not None:
            running_issue = _find_first_nonfinite_payload_tensor(checkpoint['running_states'], "running_states")
            if running_issue:
                checkpoint['running_states'] = None
                print(
                    f"WARNING: Discarding non-finite transient running state before resume: {running_issue}. "
                    "Learned weights and optimizer state were not modified."
                )

    error_budget_state = (
        checkpoint.get("error_budget_state")
        if isinstance(checkpoint, dict)
        else None
    )
    args._skipped_train_batches = int(
        error_budget_state.get("skipped_train_batches", 0)
        if isinstance(error_budget_state, dict)
        else 0
    )
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
    best_metric_tracker = initialize_best_metric_tracker(args, checkpoint)
    if best_metric_tracker is not None and not eval_tasks:
        raise ValueError(
            "--best-checkpoint-metric requires at least one immutable "
            "--eval-tasks benchmark."
        )
    if eval_tasks:
        eval_every = getattr(args, 'eval_every_epoch', 1)
        print(f"INFO: Evaluation ENABLED - will run {eval_tasks} every {eval_every} epoch(s)")
        if best_metric_tracker is not None:
            from ..evaluation import is_lm_eval_available
            if not is_lm_eval_available():
                raise RuntimeError(
                    "--best-checkpoint-metric was requested, but lm-eval is not "
                    "installed. Refusing to train without the promised selection metric."
                )
            print(
                "INFO: Best-checkpoint selection ENABLED - "
                f"{best_metric_tracker.selector} ({best_metric_tracker.mode})."
            )
    
    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        absolute_epoch = epoch_offset + epoch
        model.train()
        set_dataloader_epoch(dataloader, absolute_epoch)
        epoch_resume_step = (
            start_step if epoch == start_epoch and start_step > 0 else 0
        )
        sampler_cursor_applied = set_dataloader_start_batch(
            dataloader,
            epoch_resume_step,
        )
        if epoch_offset:
            epoch_desc = f"Epoch {absolute_epoch+1} ({epoch+1}/{args.epochs} session)"
        else:
            epoch_desc = f"Epoch {epoch+1}/{args.epochs}"
        pbar = tqdm(
            dataloader,
            desc=epoch_desc,
            total=dataloader_len,
            initial=(epoch_resume_step if sampler_cursor_applied else 0),
        )
        host_batch_source = pbar
        if epoch_resume_step > 0:
            if sampler_cursor_applied:
                print(
                    "INFO: Resuming directly from deterministic sampler batch "
                    f"{epoch_resume_step}; skipped records will not be read or transferred."
                )
            else:
                print(
                    "INFO: Sampler has no direct resume cursor; discarding "
                    f"{epoch_resume_step} batch(es) on the host before device prefetch."
                )
        host_batch_source = host_batches_from_resume(
            host_batch_source,
            epoch_resume_step,
            sampler_cursor_applied,
            total_batches=dataloader_len,
        )
        batch_source = (
            CUDABatchPrefetcher(host_batch_source, device)
            if device.type == 'cuda' and bool(getattr(args, 'cuda_prefetch', True))
            else host_batch_source
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
        
        for step, batch in enumerate(batch_source, start=epoch_resume_step):
            if batch is None:
                continue
            
            # --- FIXED: Sequence-Level State Reset ---
            # If not persisting across batches, we must start each sequence with a clean slate.
            # Local sequence context is still preserved via trainer.train_step's chunk loop.
            if not getattr(args, 'persist_state', False):
                running_states = (None, None, None, None, None, None)

            first_logged_step = start_step if epoch == start_epoch else 0
            collect_metrics = should_update_progress(step, args, dataloader_len, first_logged_step)
            args._current_global_step = (
                int(getattr(args, "_training_step_offset", 0) or 0)
                + epoch * dataloader_len
                + step
            )
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
            account_skipped_training_batch(
                args,
                reason=train_step_skip_reason(args),
                epoch=absolute_epoch + 1,
                step=step + 1,
                scope="Training",
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
                            save_best_checkpoint_if_improved(
                                eval_results,
                                args=args,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                dataloader=dataloader,
                                completed_epoch=absolute_epoch,
                                mid_epoch_step=step + 1,
                                running_states=running_states,
                            )
                        elif getattr(args, "_best_metric_tracker", None) is not None:
                            raise RuntimeError(
                                "Evaluation returned no results for the configured "
                                "best-checkpoint selector."
                            )
                        
                        model.train()
                    else:
                        print("WARNING: lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
                except Exception as e:
                    if getattr(args, "_best_metric_tracker", None) is not None:
                        raise RuntimeError(
                            "Best-checkpoint evaluation failed; refusing to "
                            "continue without a trustworthy selection signal."
                        ) from e
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
                            save_best_checkpoint_if_improved(
                                eval_results,
                                args=args,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                dataloader=dataloader,
                                completed_epoch=absolute_epoch + 1,
                                mid_epoch_step=0,
                            )
                        elif getattr(args, "_best_metric_tracker", None) is not None:
                            raise RuntimeError(
                                "Evaluation returned no results for the configured "
                                "best-checkpoint selector."
                            )
                        
                        model.train()  # Back to training mode
                    else:
                        print("WARNING: lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
                except Exception as e:
                    if getattr(args, "_best_metric_tracker", None) is not None:
                        raise RuntimeError(
                            "Best-checkpoint evaluation failed; refusing to "
                            "continue without a trustworthy selection signal."
                        ) from e
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
    effective_training_config = capture_effective_training_config(args)
    final_config.update(effective_training_config)
    final_architecture_contract = architecture_contract(model.config)
    final_architecture_hash = architecture_contract_hash(model.config)
    final_config["architecture_contract_sha256"] = final_architecture_hash
    final_checkpoint = {
        'checkpoint_version': 4,
        'checkpoint_kind': 'inference',
        'model_state_dict': clean_state_dict,
        'config': final_config,
        'architecture_contract': final_architecture_contract,
        'architecture_contract_sha256': final_architecture_hash,
        'completed_epoch': final_completed_epoch,
        'run_identity': getattr(args, "_run_identity", None),
        'best_metric_state': getattr(args, "_best_metric_state", None),
        'effective_training_config': effective_training_config,
        'optimizer_grouping_version': int(
            getattr(args, "_optimizer_grouping_version", 2) or 2
        ),
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
    if (
        bool(getattr(args, "persist_state", False))
        and not bool(getattr(args, "full_sample_bptt", False))
        and not bool(
            getattr(
                getattr(dataloader, "dataset", None),
                "guarantees_contiguous_state",
                False,
            )
        )
    ):
        raise ValueError(
            "persist_state requires a dataset that explicitly guarantees contiguous "
            "same-sequence lane ordering; shuffled independent fine-tune samples are unsafe."
        )
    configure_finetune_ltm_mode(args)
    args._resolved_training_backend = (
        "directml" if is_directml_device(device) else str(device.type)
    )
    print("INFO: Fine-tune LTM mode: read-only/inference-like (supervised LTM fast-memory writes disabled).")

    # Load the base model and its config
    model, model_config = load_full_model_with_config(args.model_path, device)
    # PEFT's CausalLM wrapper reads this field directly during forward. Older
    # Hierarchos checkpoints did not serialize it even though the strict loader
    # can infer the model family.
    if not model_config.get("model_type"):
        model_config.model_type = "hierarchos"
    # The adapter manifest binds to the untouched source function. Runtime
    # fine-tune overrides are separately authenticated by the run identity.
    base_model_config_for_manifest = AttrDict(dict(model_config))
    if validate_inference_tokenizer_identity(
        tokenizer,
        getattr(model, "_hierarchos_checkpoint_metadata", None),
    ):
        print("INFO: Fine-tune tokenizer content fingerprint verified.")
    else:
        print(
            "WARNING: Legacy fine-tune base model has no tokenizer content "
            "fingerprint; only vocabulary-size compatibility can be checked."
        )
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

    # LoRA now uses the same recurrent training contract as pretraining:
    # configured TBPTT boundaries by default, or one attached full-sample graph
    # (optionally segmented only for activation rematerialization).
    configure_full_sample_bptt(args, model_config, announce=True)
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

    args.lora_r = int(lora_r)
    lora_alpha = int(getattr(args, 'lora_alpha', 16))
    args.lora_alpha = lora_alpha
    lora_dropout = float(getattr(args, "lora_dropout", 0.05))
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    args.lora_dropout = lora_dropout
    
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
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Save/train the slow LTM key/value parameters with the adapter. The
        # read-only policy disables label-derived fast-memory writes, not
        # ordinary optimizer gradients into these slow learned parameters.
        modules_to_save=["ltm"],
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
        args._optimizer_grouping_version = int(
            checkpoint.get("optimizer_grouping_version", 1) or 1
        )
        validate_checkpoint_architecture_contract(
            checkpoint,
            dict(model.config),
            args.resume_from_ckpt,
        )
        if 'model_state_dict' in checkpoint:
            state_dict = sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False)
            _reject_unsupported_rwkv_state_dict(state_dict, args.resume_from_ckpt)
            checkpoint['model_state_dict'] = state_dict
            load_model_state_dict_compatible(model, state_dict, args.resume_from_ckpt)
        start_epoch = checkpoint.get('completed_epoch', 0)
        start_step = checkpoint.get('mid_epoch_step', 0)
        if start_step >= dataloader_len:
            start_epoch += 1
            start_step = 0
        if start_epoch >= int(getattr(args, "epochs", 0) or 0):
            raise ValueError(
                f"Fine-tune checkpoint is already at completed_epoch={start_epoch}, "
                f"but --epochs={args.epochs}; provide a larger total target epoch."
            )
        print(f"INFO: Resuming from epoch {start_epoch+1}, step {start_step}.")

    model.print_trainable_parameters()

    # Optimizer selection
    if is_directml_device(device):
        print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
        optimizer = build_hierarchos_optimizer(model, args, device)
    else:
        optimizer = build_hierarchos_optimizer(model, args, device)
    
    if checkpoint and not reset_optimizer_state_requested(args):
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
    # Full-precision checkpoint loading intentionally returns an eval-mode
    # model. PEFT wrapping does not guarantee that the already-loaded base is
    # flipped back recursively, which would enable inference-only recurrence,
    # disable training detachment/checkpointing, and change LoRA dropout.
    ensure_finetune_training_mode(model)
    checkpoint_replays_rng = configure_checkpoint_rng_policy(model)
    if checkpoint_replays_rng and bool(
        getattr(args, "full_sample_activation_checkpointing", False)
    ):
        print(
            "INFO: Activation rematerialization will replay RNG state exactly "
            "for active LoRA dropout."
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # AMP setup (BFloat16 does NOT use GradScaler — only float16 needs it)
    scaler = None
    use_amp = getattr(args, 'amp', False)
    amp_dtype_str = getattr(args, 'amp_dtype', None) or getattr(model.config if hasattr(model, 'config') else args, 'amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    if use_amp:
        if amp_dtype_str == 'float16':
            scaler = GradScaler()
            if checkpoint and not reset_optimizer_state_requested(args):
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
        and rebuild_lr_schedule_requested(args)
    ):
        num_update_steps = compute_remaining_update_steps(
            dataloader_len,
            accumulation_steps,
            start_epoch,
            args.epochs,
            start_step,
        )
        print(
            "INFO: Rebuilding fine-tune LR schedule for remaining work "
            f"({num_update_steps} update steps)."
        )
    elif (
        checkpoint
        and not rebuild_lr_schedule_requested(args)
        and isinstance(saved_main_lr_state, dict)
        and saved_main_lr_state.get("enabled", True)
        and int(saved_main_lr_state.get("total_steps", 0) or 0) > 0
    ):
        num_update_steps = max(1, int(saved_main_lr_state.get("total_steps", 1) or 1))
        print(f"INFO: Continuing saved main LR schedule ({num_update_steps} total updates).")
    else:
        num_update_steps = full_run_update_steps
    args._main_lr_schedule_total_steps = int(num_update_steps)
    resume_main_schedule_state = (
        saved_main_lr_state
        if (
            checkpoint
            and not rebuild_lr_schedule_requested(args)
            and isinstance(saved_main_lr_state, dict)
        )
        else None
    )
    main_schedule_enabled = (
        bool(resume_main_schedule_state.get("enabled", True))
        if resume_main_schedule_state is not None
        else not bool(getattr(args, 'disable_lr_schedule', False))
    )
    if main_schedule_enabled:
        if num_update_steps > 0:
            scheduler = build_lr_scheduler(
                optimizer,
                args,
                num_update_steps,
                resume_schedule_state=resume_main_schedule_state,
            )
            if checkpoint and not rebuild_lr_schedule_requested(args):
                restore_scheduler_state_and_live_lrs(
                    scheduler,
                    optimizer,
                    checkpoint.get('scheduler_state_dict'),
                    args.resume_from_ckpt,
                )
        else:
            print("Warning: Cannot enable LR schedule, dataset might be too small.")
    configure_ltm_lr_schedule(
        args,
        num_update_steps,
        checkpoint=checkpoint,
        override_schedule=rebuild_lr_schedule_requested(args),
        scheduler=scheduler,
    )
    if hasattr(model, 'config'):
        model.config.ltm_lr = float(getattr(args, '_ltm_lr_max', 1e-3))
        model.config.min_ltm_lr = float(getattr(args, '_ltm_lr_min', 0.0))
        model.config.disable_ltm_lr_schedule = not bool(
            getattr(args, '_ltm_lr_schedule_enabled', True)
        )

    args._run_identity = build_exact_resume_identity(
        args,
        tokenizer,
        dataloader,
        dataloader_len,
        architecture_config=getattr(model, "config", None),
    )
    if checkpoint:
        validate_exact_resume_identity(
            checkpoint,
            args._run_identity,
            args.resume_from_ckpt,
            allow_schedule_rebuild=rebuild_lr_schedule_requested(args),
        )
        exact_mid_epoch = int(start_step or 0) > 0
        restore_dataloader_state(
            dataloader,
            checkpoint.get("data_state"),
            strict=exact_mid_epoch,
        )
        restore_rng_state(
            checkpoint.get("rng_state"),
            strict=exact_mid_epoch,
            require_cuda=exact_mid_epoch and device.type == "cuda",
        )
        validate_exact_running_states(
            checkpoint,
            args,
            start_step,
            args.resume_from_ckpt,
        )
    args._accumulation_weighted_token_mass = 0.0
    restored_pending_grads = False
    if checkpoint:
        restored_pending_grads = restore_checkpoint_gradient_accumulation(
            model,
            checkpoint,
            args,
            device,
            scope="Fine-tune checkpoint",
        )

    error_budget_state = (
        checkpoint.get("error_budget_state")
        if isinstance(checkpoint, dict)
        else None
    )
    args._skipped_train_batches = int(
        error_budget_state.get("skipped_train_batches", 0)
        if isinstance(error_budget_state, dict)
        else 0
    )

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

    if not restored_pending_grads:
        optimizer.zero_grad(set_to_none=True)
    next_local_step = (
        int(start_epoch) * int(dataloader_len)
        + int(start_step)
    )
    saved_training_step = get_model_training_step(model)
    args._training_step_offset = (
        resolve_training_step_offset(model, next_local_step)
        if saved_training_step is not None
        else 0
    )
    if saved_training_step is None:
        print(
            "WARNING: Fine-tune base has no persisted memory-gate curriculum "
            "step; continuation starts from the local session step."
        )
    else:
        print(
            "INFO: Fine-tune memory-gate curriculum continuation: "
            f"saved_step={saved_training_step}, "
            f"next_step={args._training_step_offset + next_local_step}."
        )
    accumulation_normalization = str(
        getattr(args, "accumulation_normalization", "microbatch")
    )
    if accumulation_normalization not in {"microbatch", "weighted-token"}:
        raise ValueError(
            "accumulation_normalization must be 'microbatch' or 'weighted-token'"
        )

    for epoch in range(start_epoch, args.epochs):
        # Defensive reset after any evaluation/callback executed between epochs.
        ensure_finetune_training_mode(model)
        configure_checkpoint_rng_policy(model)
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        set_dataloader_epoch(dataloader, epoch)
        epoch_resume_step = (
            start_step if epoch == start_epoch and start_step > 0 else 0
        )
        sampler_cursor_applied = set_dataloader_start_batch(
            dataloader,
            epoch_resume_step,
        )
        pbar = tqdm(
            dataloader,
            desc=f"Finetune Epoch {epoch + 1}",
            total=dataloader_len,
            initial=(epoch_resume_step if sampler_cursor_applied else 0),
        )
        host_batch_source = host_batches_from_resume(
            pbar,
            epoch_resume_step,
            sampler_cursor_applied,
            total_batches=dataloader_len,
        )
        batch_source = (
            CUDABatchPrefetcher(host_batch_source, device)
            if device.type == 'cuda' and bool(getattr(args, 'cuda_prefetch', True))
            else host_batch_source
        )

        # A mid-epoch exact resume may need recurrent state only when the
        # dataset explicitly permits cross-batch persistence. Ordinary LoRA
        # samples start from a clean state and retain recurrence only across
        # their canonical temporal chunks.
        if (
            epoch == start_epoch
            and start_step > 0
            and isinstance(checkpoint, dict)
            and checkpoint.get("running_states") is not None
        ):
            running_states = _state_to_device(
                _clamp_running_states_for_resume(
                    checkpoint["running_states"],
                    args,
                ),
                device,
            )
            print(
                "INFO: Restored fine-tune RNN/LTM running states from "
                f"checkpoint on {device}."
            )
        else:
            running_states = (None, None, None, None, None, None)

        for i, batch in enumerate(batch_source, start=epoch_resume_step):
            if batch is None:
                continue

            if not bool(getattr(args, "persist_state", False)):
                running_states = (None, None, None, None, None, None)

            first_logged_step = (
                start_step if epoch == start_epoch else 0
            )
            collect_metrics = should_update_progress(
                i,
                args,
                dataloader_len,
                first_logged_step,
            )
            args._current_global_step = (
                int(getattr(args, "_training_step_offset", 0) or 0)
                + epoch * dataloader_len
                + i
            )
            force_optimizer_step = should_step_accumulation(
                i,
                dataloader_len,
                accumulation_steps,
            )
            accumulation_divisor = accumulation_divisor_for_step(
                i,
                dataloader_len,
                accumulation_steps,
            )
            outputs, running_states = train_step(
                model,
                batch,
                optimizer,
                scaler,
                accumulation_steps,
                i,
                args,
                running_states,
                collect_metrics=collect_metrics,
                force_optimizer_step=force_optimizer_step,
                accumulation_divisor=accumulation_divisor,
            )

            account_skipped_training_batch(
                args,
                reason=train_step_skip_reason(args),
                epoch=epoch + 1,
                step=i + 1,
                scope="Fine-tune",
            )

            if outputs:
                postfix = {"loss": f"{outputs['loss'].item():.4f}"}
                if outputs.get("ponder_cost") is not None:
                    postfix["ponder"] = (
                        f"{outputs['ponder_cost'].item():.2f}"
                    )
                if outputs.get("commitment_cost") is not None:
                    postfix["commit"] = (
                        f"{outputs['commitment_cost'].item():.2e}"
                    )
                if outputs.get("token_efficiency") is not None:
                    postfix["tok_eff"] = (
                        f"{outputs['token_efficiency'] * 100.0:.1f}%"
                    )
                    postfix["seq"] = int(
                        outputs.get("seq_len", 0) or 0
                    )
                if scheduler:
                    postfix["lr"] = (
                        f"{scheduler.get_last_lr()[0]:.2e}"
                    )
                postfix["ltm_lr"] = "off"
                pbar.set_postfix(postfix)

            if scheduler and getattr(
                args,
                "_optimizer_step_was_taken",
                False,
            ):
                scheduler.step()

            # Save after the scheduler advance. A checkpoint inside an
            # accumulation window also carries the exact pending adapter
            # gradients and weighted-token denominator.
            if (
                getattr(args, "save_steps", 0) > 0
                and (i + 1) % args.save_steps == 0
            ):
                ckpt_path = os.path.join(
                    args.out_dir,
                    f"hierarchos_finetune_epoch_{epoch+1}_step_{i+1}.pt",
                )
                print(
                    f"\n[Step {i+1}] Periodic Checkpoint: "
                    f"Saving to {ckpt_path}..."
                )
                finetune_checkpoint = build_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    dataloader,
                    completed_epoch=epoch,
                    mid_epoch_step=i + 1,
                    running_states=running_states,
                )
                finetune_checkpoint["checkpoint_kind"] = (
                    "finetune-training"
                )
                save_training_checkpoint_if_finite(
                    finetune_checkpoint,
                    ckpt_path,
                    model,
                    optimizer,
                )

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=True)
    
    # Save tokenizer
    try:
        if tokenizer:
            tokenizer.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer with adapter. Error: {e}")

    # A same-shape LoRA can otherwise be applied to the wrong base checkpoint
    # without any loader error. Publish an atomic, checksummed binding only
    # after PEFT weights/config and tokenizer assets exist.
    from ..utils.lora_merge import write_hierarchos_adapter_manifest

    adapter_manifest_path = write_hierarchos_adapter_manifest(
        args.out_dir,
        base_model_path=args.model_path,
        model_config=base_model_config_for_manifest,
        tokenizer=tokenizer,
        lora_config=lora_config,
        finetune_run_identity=args._run_identity,
    )
    print(f"Bound adapter manifest saved to {adapter_manifest_path}")
