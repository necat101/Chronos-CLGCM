"""
HierarchosCore - Full parity version with original forward method.
This is a direct port from hierarchos.py to achieve exact training parity.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .rwkv_cell import RWKVCell
from .ltm import LTMModule
from .act import (
    hard_act_depth_straight_through,
    hard_act_selection,
    normalized_act_weights,
)
from .revisions import (
    apply_architecture_revision_defaults,
    architecture_default_training_chunk_size,
    validate_architecture_numeric_contract,
)
from .shared_adapters import (
    SharedTokenAdapter,
    resolve_adapter_rank,
    resolve_token_adapter_modes,
    shared_token_lookup,
)
from ..utils.device import setup_msvc_environment, is_directml_device
from ..utils.rosa import (
    ROSA,
    ROSAState,
    ROSA_BOUNDED_CONTEXT_MODE,
    ROSA_UNBOUNDED_CONTEXT_MODE,
    rosa_async_pipeline,
)


def _config_value(config, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _set_config_value(config, name: str, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _normalize_detach_frequency(config):
    """Normalize the public 0 sentinel used to request untruncated BPTT."""
    detach_every = _config_value(config, "detach_every_n_steps", 32)
    if detach_every is not None:
        try:
            detach_every = int(detach_every)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"detach_every_n_steps must be an integer or None, got {detach_every!r}"
            ) from exc
        if detach_every <= 0:
            detach_every = None
    _set_config_value(config, "detach_every_n_steps", detach_every)
    return detach_every


def _positive_config_int(config, name: str, default=None) -> int:
    value = _config_value(config, name, None)
    if value is None and default is not None:
        value = default
        _set_config_value(config, name, value)
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Hierarchos config '{name}' must be a positive integer, got {value!r}") from exc
    if value <= 0:
        raise ValueError(f"Hierarchos config '{name}' must be a positive integer, got {value!r}")
    return value


def _resolve_recurrence_contract(config) -> int:
    """
    Resolve stateful behavior without silently changing historical checkpoints.

    Checkpoints written before this contract existed have no version field and
    therefore remain on version 1. New training runs must opt into version 2 to
    get chunk-invariant drift seeding, an explicit RWKV output readout, and an
    ACT-consistent manager-state transition.
    """
    raw_version = _config_value(config, "core_recurrence_version", 1)
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"core_recurrence_version must be 1 or 2, got {raw_version!r}"
        ) from exc
    if version not in (1, 2):
        raise ValueError(f"core_recurrence_version must be 1 or 2, got {version!r}")

    defaults = {
        "drift_recurrence_mode": (
            "legacy-chunk-seeded" if version == 1 else "state-derived"
        ),
        "rwkv_state_readout_mode": (
            "legacy-input-cache" if version == 1 else "explicit-output"
        ),
        "manager_state_commit_mode": (
            "legacy-real-step" if version == 1 else "act-weighted"
        ),
        "commitment_cost_mode": (
            "sum-square" if version == 1 else "mean-square"
        ),
    }
    allowed = {
        "drift_recurrence_mode": {"legacy-chunk-seeded", "state-derived"},
        "rwkv_state_readout_mode": {"legacy-input-cache", "explicit-output"},
        "manager_state_commit_mode": {
            "legacy-real-step",
            "act-weighted",
            "last-shadow",
            "hard-selected",
        },
        "commitment_cost_mode": {"sum-square", "mean-square"},
    }
    _set_config_value(config, "core_recurrence_version", version)
    for name, default in defaults.items():
        value = str(_config_value(config, name, default) or default)
        if value not in allowed[name]:
            choices = ", ".join(sorted(allowed[name]))
            raise ValueError(
                f"Hierarchos config '{name}' must be one of {choices}; got {value!r}"
            )
        _set_config_value(config, name, value)

    manager_compute_mode = str(
        _config_value(config, "manager_compute_mode", "soft-act") or "soft-act"
    ).strip().lower().replace("_", "-")
    if manager_compute_mode not in {"soft-act", "hard-masked"}:
        raise ValueError(
            "Hierarchos config 'manager_compute_mode' must be 'soft-act' or "
            f"'hard-masked'; got {manager_compute_mode!r}"
        )
    _set_config_value(config, "manager_compute_mode", manager_compute_mode)
    manager_state_commit_mode = _config_value(
        config,
        "manager_state_commit_mode",
        "legacy-real-step",
    )
    if (
        manager_compute_mode == "hard-masked"
        and manager_state_commit_mode != "hard-selected"
    ):
        raise ValueError(
            "manager_compute_mode='hard-masked' requires "
            "manager_state_commit_mode='hard-selected' so the recurrent state "
            "matches the selected manager output"
        )
    try:
        halt_threshold = float(_config_value(config, "h_halt_thresh", 0.9))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "h_halt_thresh must be a finite probability in [0, 1]"
        ) from exc
    if not math.isfinite(halt_threshold) or not 0.0 <= halt_threshold <= 1.0:
        raise ValueError(
            "h_halt_thresh must be a finite probability in [0, 1], "
            f"got {halt_threshold!r}"
        )
    _set_config_value(config, "h_halt_thresh", halt_threshold)

    try:
        act_depth_temperature = float(
            _config_value(config, "act_depth_temperature", 0.05)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "act_depth_temperature must be a finite positive number"
        ) from exc
    if (
        not math.isfinite(act_depth_temperature)
        or act_depth_temperature <= 0.0
    ):
        raise ValueError(
            "act_depth_temperature must be a finite positive number, "
            f"got {act_depth_temperature!r}"
        )
    _set_config_value(
        config,
        "act_depth_temperature",
        act_depth_temperature,
    )

    inference_logit_clamp = _config_value(
        config,
        "inference_logit_clamp",
        30.0 if version == 1 else 0.0,
    )
    try:
        inference_logit_clamp = float(inference_logit_clamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "inference_logit_clamp must be a finite non-negative number, "
            f"got {inference_logit_clamp!r}"
        ) from exc
    if not math.isfinite(inference_logit_clamp) or inference_logit_clamp < 0.0:
        raise ValueError(
            "inference_logit_clamp must be a finite non-negative number, "
            f"got {inference_logit_clamp!r}"
        )
    _set_config_value(config, "inference_logit_clamp", inference_logit_clamp)
    return version


def _validate_architecture_config(config) -> None:
    """Fail before allocation when a requested geometry cannot execute coherently."""
    _resolve_recurrence_contract(config)
    context_dim = _positive_config_int(config, "context_dim")
    defaults = {
        "vocab_size": None,
        "context_dim": None,
        "h_hidden": context_dim,
        "l_hidden": context_dim,
        "h_stride": 4,
        "max_h_steps": 5,
        "max_l_steps": 5,
        "ltm_slots": 1024,
        "ltm_key_dim": 128,
        "ltm_val_dim": 128,
        "ltm_topk": 4,
    }
    values = {
        name: _positive_config_int(config, name, default)
        for name, default in defaults.items()
    }
    min_h_steps = _positive_config_int(config, "min_h_steps", 1)
    if min_h_steps > values["max_h_steps"]:
        raise ValueError(
            "Hierarchos config 'min_h_steps' cannot exceed max_h_steps; "
            f"got min_h_steps={min_h_steps}, max_h_steps={values['max_h_steps']}"
        )

    # The manager input is enc + l_feedback and therefore has context_dim
    # features. Supporting a different manager width would require a new learned
    # projection and would break existing checkpoint layouts.
    if values["h_hidden"] != values["context_dim"]:
        raise ValueError(
            "Hierarchos currently requires h_hidden == context_dim because the "
            f"manager consumes context-width residuals; got h_hidden={values['h_hidden']} "
            f"and context_dim={values['context_dim']}."
        )

    shared_requested_head = _config_value(config, "rwkv_head_size", None)
    for cell_prefix, width_name in (
        ("h", "h_hidden"),
        ("l", "l_hidden"),
    ):
        field_name = f"{cell_prefix}_rwkv_head_size"
        requested_head = _config_value(
            config,
            field_name,
            shared_requested_head,
        )
        if requested_head in (None, 0, "", "auto"):
            continue
        try:
            requested_head = int(requested_head)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be a positive divisor or auto, "
                f"got {requested_head!r}"
            ) from exc
        if requested_head <= 0:
            raise ValueError(
                f"{field_name} must be a positive divisor or auto, "
                f"got {requested_head!r}"
            )
        if values[width_name] % requested_head != 0:
            raise ValueError(
                f"{field_name}={requested_head} does not divide "
                f"{width_name}={values[width_name]}."
            )

    _normalize_detach_frequency(config)

def _config_float(config, name: str, default: float) -> float:
    try:
        if isinstance(config, dict):
            raw_value = config.get(name, default)
        else:
            raw_value = getattr(config, name, default)
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0.0 else default

def _config_nonnegative_float(config, name: str, default: float) -> float:
    try:
        if isinstance(config, dict):
            raw_value = config.get(name, default)
        else:
            raw_value = getattr(config, name, default)
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0.0 else default

def _finite_clamp(tensor: torch.Tensor, max_abs: float, *, nan: float = 0.0) -> torch.Tensor:
    """Clamp finite values without concealing NaN/Inf trajectories.

    Earlier revisions converted non-finite intermediates to ordinary numbers,
    which could let a corrupted recurrent trajectory reach the optimizer as if
    it were valid. Preserve non-finites so the raw-logit/loss guards can reject
    the batch. This is identical to the historical clamp for every finite input.
    """
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    max_abs = float(max_abs)
    clamped = torch.clamp(
        tensor,
        min=-max_abs,
        max=max_abs,
    )
    return torch.where(torch.isfinite(tensor), clamped, tensor)


def _validate_sequence_mask_contract(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    loss_weights: Optional[torch.Tensor],
) -> tuple[bool, int]:
    """
    Validate the recurrence-safe padding contract.

    The recurrent core currently advances every row at every tensor column.
    Consequently only right padding is coherent: a masked token may never be
    followed by an active token in the same call. Masked labels/weights must
    also be inert so padding cannot contribute a language-model gradient.
    """
    batch_size, sequence_length = input_ids.shape
    invalid_checks = []
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        binary_mask = (attention_mask == 0) | (attention_mask == 1)
        invalid_checks.append((
            ~binary_mask.all(),
            "attention_mask must contain only 0/1 values",
        ))
        active = attention_mask.to(dtype=torch.bool)
        if sequence_length > 1:
            masked_then_active = (~active[:, :-1]) & active[:, 1:]
            invalid_checks.append((
                masked_then_active.any(),
                (
                    "Hierarchos recurrence accepts right padding only; left padding "
                    "or holes in attention_mask would advance hidden state through "
                    "masked tokens before later active tokens"
                ),
            ))
    else:
        active = None

    if labels is not None:
        if labels.ndim != 2 or labels.shape[0] != batch_size:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} is incompatible with "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        if labels.shape[1] > sequence_length + 1:
            raise ValueError(
                "labels may be at most one token longer than input_ids for "
                "chunk-boundary lookahead loss"
            )
        if active is not None:
            checked_columns = min(sequence_length, labels.shape[1])
            invalid_masked_labels = (
                (~active[:, :checked_columns])
                & (labels[:, :checked_columns] != -100)
            )
            invalid_checks.append((
                invalid_masked_labels.any(),
                "labels at attention_mask=0 positions must use ignore_index=-100",
            ))
            if labels.shape[1] == sequence_length + 1:
                invalid_lookahead = (
                    (~active[:, -1])
                    & (labels[:, sequence_length] != -100)
                )
                invalid_checks.append((
                    invalid_lookahead.any(),
                    (
                        "lookahead labels after a masked final input token must "
                        "use ignore_index=-100"
                    ),
                ))

    if loss_weights is not None:
        if loss_weights.ndim != 2 or loss_weights.shape[0] != batch_size:
            raise ValueError(
                f"loss_weights shape {tuple(loss_weights.shape)} is incompatible with "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        if loss_weights.shape[1] > sequence_length + 1:
            raise ValueError(
                "loss_weights may be at most one token longer than input_ids"
            )
        if labels is not None and loss_weights.shape[1] < min(
            sequence_length,
            labels.shape[1],
        ):
            raise ValueError(
                "loss_weights must cover every in-chunk label column"
            )
        invalid_checks.extend((
            (
                ~torch.isfinite(loss_weights).all(),
                "loss_weights must be finite",
            ),
            (
                (loss_weights < 0).any(),
                "loss_weights must be non-negative",
            ),
        ))
        if active is not None:
            checked_columns = min(sequence_length, loss_weights.shape[1])
            invalid_masked_weights = (
                (~active[:, :checked_columns])
                & (loss_weights[:, :checked_columns] != 0)
            )
            invalid_checks.append((
                invalid_masked_weights.any(),
                "loss_weights at attention_mask=0 positions must be zero",
            ))
            if loss_weights.shape[1] == sequence_length + 1:
                invalid_lookahead_weight = (
                    (~active[:, -1])
                    & (loss_weights[:, sequence_length] != 0)
                )
                invalid_checks.append((
                    invalid_lookahead_weight.any(),
                    (
                        "lookahead loss_weights after a masked final input token "
                        "must be zero"
                    ),
                ))

    mask_has_padding = (
        (~active).any()
        if active is not None
        else torch.zeros((), device=input_ids.device, dtype=torch.bool)
    )
    first_padding_column = (
        active.to(dtype=torch.long).sum(dim=1).min()
        if active is not None
        else torch.tensor(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
    )
    if invalid_checks:
        # One device-to-host transfer validates the complete mask/label/weight
        # contract and returns the padding fast-path decision.
        check_values = torch.stack(
            [
                check.to(device=input_ids.device, dtype=torch.long)
                for check, _message in invalid_checks
            ] + [
                mask_has_padding.to(device=input_ids.device, dtype=torch.long),
                first_padding_column.to(device=input_ids.device, dtype=torch.long),
            ]
        ).tolist()
        for is_invalid, (_check, message) in zip(
            check_values[:-2],
            invalid_checks,
        ):
            if bool(is_invalid):
                raise ValueError(message)
        return bool(check_values[-2]), int(check_values[-1])
    return False, sequence_length


def _logit_numerics(
    logits: torch.Tensor,
    saturation_threshold: float,
) -> dict:
    """Detached raw-logit health metrics; never sanitize the training graph."""
    raw = logits.detach().float()
    finite = torch.isfinite(raw)
    finite_abs = torch.where(finite, raw.abs(), torch.zeros_like(raw))
    if finite_abs.numel() == 0:
        max_abs = torch.zeros((), device=raw.device, dtype=torch.float32)
        saturation_fraction = torch.zeros(
            (),
            device=raw.device,
            dtype=torch.float32,
        )
    else:
        max_abs = finite_abs.amax()
        saturation_fraction = (
            finite_abs >= float(saturation_threshold)
        ).float().mean()
    return {
        "raw_logit_max_abs": max_abs,
        "raw_logit_nonfinite_count": (~finite).sum(),
        "raw_logit_saturation_fraction": saturation_fraction,
        "raw_logit_saturation_threshold": torch.tensor(
            float(saturation_threshold),
            device=raw.device,
            dtype=torch.float32,
        ),
    }


def _keep_active_rows(
    updated: torch.Tensor,
    previous: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Commit ``updated`` only for active batch rows."""
    row_mask = active_rows.view(
        active_rows.shape[0],
        *([1] * (updated.ndim - 1)),
    )
    return torch.where(row_mask, updated, previous)


def _l2_norm_clamp(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    if (
        tensor is None
        or not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or max_norm <= 0.0
    ):
        return tensor
    norm = torch.linalg.vector_norm(tensor.float(), ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(tensor.new_tensor(float(max_norm)) / (norm.to(dtype=tensor.dtype) + 1e-6), max=1.0)
    return tensor * scale

def _quiet_torch_compile_logs():
    """Keep useful compiler warnings while hiding routine autotune chatter."""
    try:
        import torch._inductor.config as inductor_config
        if hasattr(inductor_config, "verbose_progress"):
            inductor_config.verbose_progress = False
    except Exception:
        pass
    for logger_name in (
        "torch._dynamo",
        "torch._inductor",
        "torch._inductor.select_algorithm",
        "torch._inductor.cudagraph_trees",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def _resolve_compile_kwargs(config, device_type: str, fullgraph: bool = False):
    compile_mode = getattr(config, 'compile_mode', 'reduce-overhead')
    if compile_mode in (None, '', 'default'):
        compile_mode = None
    compile_backend = getattr(config, 'compile_backend', None)
    if compile_backend in (None, '', 'default'):
        compile_backend = None
    compile_dynamic = bool(getattr(config, 'compile_dynamic', False))
    compile_cudagraphs = bool(getattr(config, 'compile_cudagraphs', False))

    # Some PyTorch builds reject passing both mode=... and options=....
    # CUDA-graph preference is encoded with the mode when possible; options are
    # used only for default-mode compile where PyTorch accepts them.
    effective_mode = compile_mode
    effective_cudagraphs = compile_cudagraphs
    if device_type == 'cuda':
        if effective_mode == 'max-autotune' and not compile_cudagraphs:
            effective_mode = 'max-autotune-no-cudagraphs'
            effective_cudagraphs = False
        elif effective_mode == 'max-autotune-no-cudagraphs':
            effective_cudagraphs = False

    kwargs = {
        "dynamic": compile_dynamic,
        "fullgraph": bool(fullgraph),
    }
    if compile_backend is not None:
        kwargs["backend"] = compile_backend
    if effective_mode is not None:
        kwargs["mode"] = effective_mode
    elif device_type == 'cuda':
        kwargs["options"] = {"triton.cudagraphs": effective_cudagraphs}

    return kwargs, effective_mode, effective_cudagraphs

class WorkerLoop:
    """
    Encapsulates the Worker's iterative refinement loop.
    Direct port from original hierarchos.py for full parity.
    NOTE: This is a plain class, NOT nn.Module, to avoid state_dict key prefixing.
    """
    def __init__(self, config, l_rnn, l_input_proj, context_drift_proj, l_to_out):
        self.config = config
        self.l_rnn = l_rnn
        self.l_input_proj = l_input_proj
        self.context_drift_proj = context_drift_proj
        self.l_to_out = l_to_out
        # Serialized config records whether compilation was requested while a
        # checkpoint trained; it does not prove this live object was compiled.
        self._runtime_compiled = False
        self.refresh_runtime_config()

    def refresh_runtime_config(self):
        config = self.config
        self.max_l_steps = config.max_l_steps
        self.l_conv_atol = getattr(config, 'l_conv_atol', 0.01)
        self.commitment_threshold = getattr(config, 'commitment_threshold', 0.1)
        self.commitment_cost_mode = getattr(
            config,
            'commitment_cost_mode',
            'sum-square',
        )
        self.recurrent_state_clamp = _config_float(config, 'recurrent_state_clamp', 50.0)
        self.context_state_clamp = _config_float(config, 'context_state_clamp', 50.0)
        self.drift_state_clamp = _config_float(config, 'drift_state_clamp', 5.0)
        self.drift_norm_clamp = _config_nonnegative_float(config, 'drift_norm_clamp', 0.0)
        # Zero is a coherent ablation that disables iterative drift. The old
        # positive-only parser silently turned an explicit 0 into 1.0 while the
        # architecture contract still recorded 0.
        self.drift_delta_scale = _config_nonnegative_float(
            config,
            'drift_delta_scale',
            1.0,
        )
        self.activation_clamp = _config_float(config, 'activation_clamp', 100.0)
        static_loop = getattr(config, 'compile_static_worker_loop', False)
        self.compile_static_worker_loop = (
            bool(static_loop)
            if static_loop is not None
            else bool(self._runtime_compiled)
        )
        # An explicit static-loop request remains meaningful in eager mode when
        # compilation itself is disabled. When config.compile is true, however,
        # only compile() may activate the fixed loop: checkpoint-loaded eager
        # inference must not inherit a historical training execution mode.
        self._force_static_worker_loop_eager = (
            self.compile_static_worker_loop
            and not bool(getattr(config, 'compile', False))
        )

    def __call__(self, enc: torch.Tensor, static_context: torch.Tensor, l_state: torch.Tensor, 
                initial_drift: torch.Tensor, timestep: Optional[int] = None, l_deepemb_vec: Optional[torch.Tensor] = None):
        # Handle batch dimension squeezing for torch.compile compatibility
        if enc.dim() == 3 and enc.shape[0] == 1: enc = enc.squeeze(0)
        if static_context.dim() == 3 and static_context.shape[0] == 1: static_context = static_context.squeeze(0)
        if l_state.dim() == 4 and l_state.shape[0] == 1: l_state = l_state.squeeze(0)
        if initial_drift.dim() == 3 and initial_drift.shape[0] == 1: initial_drift = initial_drift.squeeze(0)

        enc = _finite_clamp(enc, self.activation_clamp)
        static_context = _finite_clamp(static_context, self.context_state_clamp)
        l_state = _finite_clamp(l_state, self.recurrent_state_clamp)
        current_drift = _l2_norm_clamp(_finite_clamp(initial_drift, self.drift_state_clamp), self.drift_norm_clamp)
        current_enc = enc
        effective_l_steps = torch.zeros(
            enc.shape[0],
            device=enc.device,
            dtype=enc.dtype,
        )

        # Shadow state for exploration (pondering)
        shadow_l_state = l_state
        
        # Initialize dynamic_context
        dynamic_context = static_context + current_drift

        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = self.l_input_proj(l_input_vec)
        l_input = _finite_clamp(l_input, self.recurrent_state_clamp)
        
        check_idx = [0, 1, 2]
        # Full-sample BPTT checkpoints must execute the same refinement policy
        # during inference that produced their training logits.  The legacy eval
        # path also stops on state convergence, while training stops/masks only
        # on drift convergence; that can change the final drift and logits.
        inference_logit_parity = bool(
            getattr(self.config, 'inference_logit_parity', False)
            or getattr(self.config, 'full_sample_bptt', False)
        )
        use_training_refinement = self.l_rnn.training or inference_logit_parity

        if not use_training_refinement:
            prev_shadow = shadow_l_state.clone()
            for step_idx in range(self.max_l_steps):
                effective_l_steps = effective_l_steps + 1.0
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1), deepemb_vec=l_deepemb_vec)
                l_out = _finite_clamp(l_out, self.activation_clamp)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out)) * self.drift_delta_scale
                current_drift = _l2_norm_clamp(_finite_clamp(current_drift + drift_delta, self.drift_state_clamp), self.drift_norm_clamp)
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = _finite_clamp(self.l_input_proj(l_input_vec), self.recurrent_state_clamp)
                
                drift_converged = torch.mean(torch.abs(drift_delta)) < self.l_conv_atol
                state_converged = torch.allclose(shadow_l_state[..., check_idx], prev_shadow[..., check_idx], atol=self.l_conv_atol)
                if drift_converged or state_converged: break
                prev_shadow = shadow_l_state.clone()
        else:
            # One canonical training/parity policy is used in eager and compiled
            # execution. Convergence is row-local and converged rows are frozen;
            # no row may change a peer's refinement depth through a batch mean.
            active = torch.ones(enc.shape[0], device=enc.device, dtype=enc.dtype)
            drift_cost_sum = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)
            drift_cost_count = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)
            fixed_runtime_loop = bool(
                (self._runtime_compiled and self.compile_static_worker_loop)
                or self._force_static_worker_loop_eager
            )

            for step_idx in range(self.max_l_steps):
                if (
                    step_idx > 0
                    and not fixed_runtime_loop
                    and not bool((active > 0).any().item())
                ):
                    # In eager execution, later candidates cannot affect output,
                    # state, commitment, or gradients once every row has frozen.
                    # Compiled execution deliberately keeps the fixed loop shape.
                    break
                prev_shadow = shadow_l_state
                prev_drift = current_drift
                prev_l_input = l_input
                effective_l_steps = effective_l_steps + active
                active_rows = active > 0

                l_out, candidate_shadow = self.l_rnn(
                    l_input,
                    shadow_l_state,
                    timestep=None,
                    deepemb_vec=l_deepemb_vec,
                )
                l_out = _finite_clamp(l_out, self.activation_clamp)

                drift_delta = torch.tanh(self.context_drift_proj(l_out)) * self.drift_delta_scale
                candidate_drift = _l2_norm_clamp(
                    _finite_clamp(
                        current_drift + drift_delta,
                        self.drift_state_clamp,
                    ),
                    self.drift_norm_clamp,
                )
                if self.commitment_cost_mode == 'mean-square':
                    drift_sq = torch.mean(candidate_drift ** 2, dim=-1)
                else:
                    drift_sq = torch.sum(candidate_drift ** 2, dim=-1)
                hinge_cost = torch.relu(drift_sq - self.commitment_threshold)
                hinge_cost = torch.clamp(hinge_cost, max=100.0)
                drift_cost_sum = drift_cost_sum + torch.where(
                    active_rows,
                    hinge_cost,
                    torch.zeros_like(hinge_cost),
                )
                drift_cost_count = drift_cost_count + active

                candidate_dynamic = static_context + candidate_drift
                candidate_input_vec = torch.cat(
                    [current_enc, candidate_dynamic],
                    dim=-1,
                )
                candidate_l_input = _finite_clamp(
                    self.l_input_proj(candidate_input_vec),
                    self.recurrent_state_clamp,
                )

                shadow_l_state = _keep_active_rows(
                    candidate_shadow,
                    prev_shadow,
                    active_rows,
                )
                current_drift = _keep_active_rows(
                    candidate_drift,
                    prev_drift,
                    active_rows,
                )
                l_input = _keep_active_rows(
                    candidate_l_input,
                    prev_l_input,
                    active_rows,
                )

                still_active = (
                    torch.mean(torch.abs(drift_delta), dim=-1) >= self.l_conv_atol
                ).to(dtype=enc.dtype)
                active = active * still_active

            commitment_cost_static = drift_cost_sum / torch.clamp(
                drift_cost_count,
                min=1.0,
            )

        # Use original l_state (not shadow) for the actual state update
        ts = timestep if timestep is not None else 0
        
        # l_input is already aligned with final current_drift: it is initialized
        # before refinement and updated alongside every accepted candidate.
        # Re-projecting the same [enc, context] pair here added one full linear
        # layer per token without changing the committed transition.
        final_l_out, next_l_state = self.l_rnn(l_input, l_state, timestep=None, deepemb_vec=l_deepemb_vec)
        final_l_out = _finite_clamp(final_l_out, self.activation_clamp)
        
        final_enc = current_enc + self.l_to_out(final_l_out)
        final_enc = _finite_clamp(final_enc, self.activation_clamp)
        commitment_cost = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)
        if use_training_refinement:
            commitment_cost = commitment_cost_static

        final_drift = _l2_norm_clamp(_finite_clamp(current_drift, self.drift_state_clamp), self.drift_norm_clamp)
        # The committed worker transition is one additional recurrent step after
        # refinement, even when the row converged immediately.
        effective_l_steps = effective_l_steps + 1.0
        return final_enc, next_l_state, commitment_cost, final_drift, effective_l_steps


class HierarchosCore(nn.Module):
    """
    Full parity version of HierarchosCore - direct port from hierarchos.py.
    """
    
    def reset_memory(self):
        """Resets the short-term 'fast' associative memory."""
        self.ltm.reset_working_memory()

    def refresh_runtime_config(self):
        detach_freq = _normalize_detach_frequency(self.config)
        rwkv_state_clamp = _config_float(
            self.config,
            'recurrent_state_clamp',
            50.0,
        )
        rwkv_channel_mix_key_clamp = _config_nonnegative_float(
            self.config,
            'rwkv_channel_mix_key_clamp',
            12.0,
        )
        rwkv_channel_mix_deepembed_clamp = _config_nonnegative_float(
            self.config,
            'rwkv_channel_mix_deepembed_clamp',
            4.0,
        )
        for cell_name in ("h_rnn", "l_rnn"):
            cell = getattr(self, cell_name, None)
            if cell is not None:
                cell.detach_every_n_steps = detach_freq
                cell.state_clamp = rwkv_state_clamp
                if hasattr(cell, "channel_mix_key_clamp"):
                    cell.channel_mix_key_clamp = rwkv_channel_mix_key_clamp
                if hasattr(cell, "channel_mix_deepembed_clamp"):
                    cell.channel_mix_deepembed_clamp = rwkv_channel_mix_deepembed_clamp
        if hasattr(self, "worker_loop_module"):
            self.worker_loop_module.config = self.config
            self.worker_loop_module.refresh_runtime_config()

    def set_training_step(self, step: int):
        if hasattr(self, "memory_gate_warmup_step"):
            with torch.no_grad():
                self.memory_gate_warmup_step.fill_(float(max(0, int(step or 0))))

    def _memory_gate_warmup_floor(
        self,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Resolve the scalar gate floor once per forward pass.

        The schedule position is constant for every token in one model call.
        Computing its scalar tensor arithmetic inside the recurrent token loop
        launched several tiny accelerator kernels per token.
        """
        warmup_steps = float(getattr(self.config, 'memory_gate_warmup_steps', 0) or 0)
        warmup_floor = float(getattr(self.config, 'memory_gate_warmup_floor', 0.0) or 0.0)
        if warmup_steps <= 0.0 or warmup_floor <= 0.0:
            return None
        warmup_floor = min(max(warmup_floor, 0.0), 0.95)
        step = self.memory_gate_warmup_step.to(
            device=reference.device,
            dtype=torch.float32,
        )
        progress = torch.clamp(step / warmup_steps, min=0.0, max=1.0)
        return reference.new_tensor(warmup_floor) * (
            1.0 - progress.to(dtype=reference.dtype)
        )

    def _apply_memory_gate_warmup(
        self,
        gate: torch.Tensor,
        *,
        floor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if floor is None:
            floor = self._memory_gate_warmup_floor(gate)
        if floor is None:
            return gate
        return floor + (1.0 - floor) * gate
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Resolve a named learned-function contract before allocation. Historical
        # configs may opt into individual recurrence-v2 fixes without carrying a
        # named revision, so preserve that supported path while still applying
        # the same fail-closed numeric validation.
        requested_revision = _config_value(
            self.config,
            "architecture_revision",
            None,
        )
        if requested_revision not in (None, "", "auto"):
            apply_architecture_revision_defaults(self.config)
        else:
            validate_architecture_numeric_contract(self.config)
        _validate_architecture_config(self.config)
        
        # Tokenizer-dependent
        if not torch.cuda.is_available():
            torch.set_flush_denormal(True)
            
        self.tok_emb = nn.Embedding(config.vocab_size, config.context_dim)
        
        # Historical checkpoints retain their vocabulary-sized auxiliary
        # tables. Coherent-v9 reuses the tied token embedding through
        # neutral-initialized low-rank adapters.
        self.deepembed_mode, self.rosa_embedding_mode = resolve_token_adapter_modes(
            self.config
        )
        self.use_deepembed = self.deepembed_mode != "off"
        self.use_rosa = self.rosa_embedding_mode != "off"
        adapter_rank = None
        if (
            self.deepembed_mode == "shared-factorized"
            or self.rosa_embedding_mode == "shared-factorized"
        ):
            adapter_rank = resolve_adapter_rank(
                self.config,
                input_dim=config.context_dim,
            )

        if self.deepembed_mode == "legacy-table":
            self.h_deepemb = nn.Embedding(config.vocab_size, config.h_hidden * 4)
            self.l_deepemb = nn.Embedding(config.vocab_size, config.l_hidden * 4)
            nn.init.ones_(self.h_deepemb.weight)
            nn.init.ones_(self.l_deepemb.weight)
        elif self.deepembed_mode == "shared-factorized":
            self.h_deepembed_adapter = SharedTokenAdapter(
                config.context_dim,
                config.h_hidden * 4,
                adapter_rank,
                output_bias=1.0,
            )
            self.l_deepembed_adapter = SharedTokenAdapter(
                config.context_dim,
                config.l_hidden * 4,
                adapter_rank,
                output_bias=1.0,
            )
        
        self.memory_token_routers = bool(getattr(config, 'memory_token_routers', True))
        # This schedule position is part of the forward function. Persist it so
        # an early checkpoint produces identical gates in training, evaluation,
        # and chat instead of appearing to "warm up again" after reload.
        self.register_buffer(
            "memory_gate_warmup_step",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )
        if self.use_rosa:
            if self.rosa_embedding_mode == "legacy-table":
                self.rosa_emb = nn.Embedding(config.vocab_size + 1, config.context_dim)
                nn.init.zeros_(self.rosa_emb.weight)
            else:
                self.rosa_adapter = SharedTokenAdapter(
                    config.context_dim,
                    config.context_dim,
                    adapter_rank,
                    output_bias=0.0,
                )
            # Learnable gate: sigmoid(-1.0) ≈ 0.27 initial injection strength
            self.rosa_gate_logit = nn.Parameter(torch.tensor(-1.0))
            if self.memory_token_routers:
                self.rosa_router = nn.Linear(config.context_dim, 1)
                nn.init.zeros_(self.rosa_router.weight)
                nn.init.zeros_(self.rosa_router.bias)
        
        # Global Learnable State
        self.persistent_dim = getattr(config, 'persistent_dim', 128)
        self.persistent = nn.Parameter(torch.randn(self.persistent_dim) * 0.02)
        
        # Learnable LTM Gate
        self.ltm_gate_logit = nn.Parameter(torch.tensor(-2.0))
        if self.memory_token_routers:
            self.ltm_router = nn.Linear(config.context_dim, 1)
            nn.init.zeros_(self.ltm_router.weight)
            nn.init.zeros_(self.ltm_router.bias)

        # LTM System
        self.ltm = LTMModule(
            n_slots=config.ltm_slots, 
            key_dim=config.ltm_key_dim, 
            val_dim=config.ltm_val_dim,
            lr=getattr(config, 'ltm_lr', 1e-3),
            momentum=getattr(config, 'ltm_momentum', 0.9),
            wd=getattr(config, 'ltm_weight_decay', 1e-4),
            forget_rate=getattr(config, 'ltm_forget_rate', 0.01),
            reference_chunk_len=_config_value(
                config,
                'reference_chunk_len',
                _config_value(
                    config,
                    'training_chunk_size',
                    architecture_default_training_chunk_size(config),
                ),
            ),
            score_grad_scale=getattr(config, 'ltm_score_grad_scale', 1.0),
            cpu_gather_retrieval=getattr(config, 'ltm_cpu_gather_retrieval', True),
            cpu_sparse_update=getattr(config, 'ltm_cpu_sparse_update', True)
        )
        self.qproj = nn.Linear(config.context_dim * 2, config.ltm_key_dim, bias=False)
        self.val_proj = nn.Linear(config.context_dim, config.ltm_val_dim, bias=False)
        
        # Encoder Projection
        in_dim = config.context_dim + self.persistent_dim + config.ltm_val_dim * config.ltm_topk
        self.in_proj = nn.Linear(in_dim, config.context_dim)
        
        # Manager Components
        self.l_feedback_proj = nn.Linear(config.l_hidden, config.h_hidden, bias=False)
        # Initialize with small weights to introduce feedback gradually
        nn.init.normal_(self.l_feedback_proj.weight, mean=0.0, std=0.01)

        rwkv_head_size = getattr(config, 'rwkv_head_size', None)
        h_rwkv_head_size = getattr(
            config,
            "h_rwkv_head_size",
            rwkv_head_size,
        )
        l_rwkv_head_size = getattr(
            config,
            "l_rwkv_head_size",
            rwkv_head_size,
        )
        rwkv_channel_mix_key_clamp = _config_nonnegative_float(
            config,
            'rwkv_channel_mix_key_clamp',
            12.0,
        )
        rwkv_channel_mix_deepembed_clamp = _config_nonnegative_float(
            config,
            'rwkv_channel_mix_deepembed_clamp',
            4.0,
        )
        rwkv_state_clamp = _config_float(
            config,
            'recurrent_state_clamp',
            50.0,
        )
        rwkv_state_readout_mode = _config_value(
            config,
            'rwkv_state_readout_mode',
            'legacy-input-cache',
        )
        self.config.rwkv_channel_mix_key_clamp = rwkv_channel_mix_key_clamp
        self.config.rwkv_channel_mix_deepembed_clamp = rwkv_channel_mix_deepembed_clamp
        self.h_rnn = RWKVCell(
            config.h_hidden,
            head_size=h_rwkv_head_size,
            layer_id=0,
            n_layer=getattr(config, 'rwkv_n_layer_hint', 2),
            channel_mix_key_clamp=rwkv_channel_mix_key_clamp,
            channel_mix_deepembed_clamp=rwkv_channel_mix_deepembed_clamp,
            state_readout_mode=rwkv_state_readout_mode,
            state_clamp=rwkv_state_clamp,
        )
        _set_config_value(
            config,
            "h_rwkv_head_size",
            int(self.h_rnn.head_size),
        )
        self.h_to_context = nn.Linear(config.h_hidden, config.context_dim)
        self.h_halt_proj = nn.Linear(config.h_hidden, 1)
        # Initialize bias to encourage max_h_steps pondering steps initially
        # Formula: logit(1/N) = -log(N-1). This sets initial halt prob to 1/max_h_steps.
        with torch.no_grad():
            initial_steps = max(2.0, float(config.max_h_steps))
            initial_bias = -math.log(initial_steps - 1.0)
            self.h_halt_proj.bias.fill_(initial_bias)
        
        # Worker Components
        self.l_input_proj = nn.Linear(config.context_dim * 2, config.l_hidden)
        self.l_rnn = RWKVCell(
            config.l_hidden,
            head_size=l_rwkv_head_size,
            layer_id=0,
            n_layer=getattr(config, 'rwkv_n_layer_hint', 2),
            channel_mix_key_clamp=rwkv_channel_mix_key_clamp,
            channel_mix_deepembed_clamp=rwkv_channel_mix_deepembed_clamp,
            state_readout_mode=rwkv_state_readout_mode,
            state_clamp=rwkv_state_clamp,
        )
        _set_config_value(
            config,
            "l_rwkv_head_size",
            int(self.l_rnn.head_size),
        )
        # Preserve the historical shared field when both cells really share
        # geometry; explicit per-cell fields make asymmetric widths unambiguous.
        _set_config_value(
            config,
            "rwkv_head_size",
            (
                int(self.h_rnn.head_size)
                if self.h_rnn.head_size == self.l_rnn.head_size
                else None
            ),
        )
        
        # Configure truncated BPTT for RWKV cells
        detach_freq = getattr(config, 'detach_every_n_steps', 32)
        self.h_rnn.detach_every_n_steps = detach_freq
        self.l_rnn.detach_every_n_steps = detach_freq

        self.context_drift_proj = nn.Linear(config.l_hidden, config.context_dim, bias=False)
        nn.init.normal_(self.context_drift_proj.weight, mean=0.0, std=0.01)

        self.l_to_out = nn.Linear(config.l_hidden, config.context_dim)
        
        # Output Head
        self.out_norm = nn.LayerNorm(config.context_dim)
        self.lm_head = nn.Linear(config.context_dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying
        
        # Worker Loop Wrapper - pass actual module references
        self.worker_loop_module = WorkerLoop(config, self.l_rnn, self.l_input_proj, 
                                             self.context_drift_proj, self.l_to_out)
        
        # Sinusoidal Encoding for Timestamps
        half_dim = config.ltm_val_dim // 2
        if half_dim <= 0:
            emb = torch.empty(0, dtype=torch.float32)
        elif half_dim == 1:
            emb = torch.ones(1, dtype=torch.float32)
        else:
            scale = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
        self.register_buffer('time_freqs', emb)

    def compile(self):
        """Applies torch.compile to the worker loop if enabled in config (Robust Parity)."""
        if not getattr(self.config, 'compile', False):
            return
        eager_worker_loop = self.worker_loop_module
        
        device = next(self.parameters()).device
        device_type = 'cpu'
        if device.type == 'cuda': device_type = 'cuda'
        elif is_directml_device(device): device_type = 'dml'

        # Check for DirectML (doesn't support torch.compile)
        if device_type == 'dml':
            print("INFO: DirectML detected - torch.compile is not supported. Using eager mode.")
            self.config.compile = False
            return

        # Check for Windows CPU + Compile (Known Hang Issue)
        if os.name == 'nt' and device_type == 'cpu' and not getattr(self.config, 'force_compile', False):
            print("WARNING: torch.compile on Windows CPU is known to hang with complex RNN loops.")
            print("         Disabling compilation for stability. Use force_compile=True to override.")
            self.config.compile = False
            return

        try:
            if hasattr(torch, "compile"):
                compile_kwargs, compile_mode, compile_cudagraphs = _resolve_compile_kwargs(
                    self.config,
                    device_type,
                    fullgraph=False,
                )
                compile_dynamic = bool(compile_kwargs.get("dynamic", False))
                compile_fullgraph_worker = bool(getattr(self.config, 'compile_fullgraph_worker', False))
                if bool(getattr(self.config, 'compile_quiet', True)):
                    _quiet_torch_compile_logs()

                print(
                    "INFO: Compiling RWKV hot path "
                    f"(mode={compile_mode or 'default'}, dynamic={compile_dynamic}, "
                    f"cudagraphs={compile_cudagraphs})."
                )
                if os.name == 'nt' and device_type != 'dml':
                    setup_msvc_environment()

                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                dynamo.config.cache_size_limit = max(getattr(dynamo.config, 'cache_size_limit', 8), 64)

                if hasattr(self.h_rnn, "allow_legacy_state_migration"):
                    self.h_rnn.allow_legacy_state_migration = False
                if hasattr(self.l_rnn, "allow_legacy_state_migration"):
                    self.l_rnn.allow_legacy_state_migration = False
                static_loop = getattr(self.config, 'compile_static_worker_loop', None)
                eager_worker_loop.compile_static_worker_loop = True if static_loop is None else bool(static_loop)
                eager_worker_loop._runtime_compiled = True
                eager_worker_loop._force_static_worker_loop_eager = False

                if bool(getattr(self.config, 'compile_h_rnn', True)):
                    self.h_rnn.compile_forward(**compile_kwargs)

                worker_compile_kwargs, _, _ = _resolve_compile_kwargs(
                    self.config,
                    device_type,
                    fullgraph=compile_fullgraph_worker,
                )
                self.worker_loop_module = torch.compile(
                    eager_worker_loop,
                    **worker_compile_kwargs,
                )
                print("INFO: RWKV hot path compiled successfully.")
                if device_type == 'cuda' and compile_mode in ('max-autotune', 'max-autotune-no-cudagraphs'):
                    print(
                        "INFO: The first CUDA train step may spend several minutes autotuning kernels; "
                        "judge steady-state throughput after steps 3-5."
                    )
        except Exception as e:
            print(f"Warning: Compilation failed! Falling back to eager mode. {e}")
            eager_worker_loop._runtime_compiled = False
            eager_worker_loop._force_static_worker_loop_eager = False
            self.worker_loop_module = eager_worker_loop
            self.config.compile = False

    def forward(self, input_ids, attention_mask=None, labels=None, 
                h_state=None, l_state=None, 
                prev_context=None, target_context=None,
                drift_state=None, ltm_memory_state=None,
                global_pos_offset=0, min_timestamp=0.0, source_filter=None,
                min_wallclock_timestamp=0.0, **kwargs):
        """
        Full forward method - direct port from hierarchos.py for exact parity.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        B, T = input_ids.shape
        if B <= 0 or T <= 0:
            raise ValueError("input_ids must contain at least one batch row and one token")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        device = input_ids.device
        recurrent_state_clamp = _config_float(self.config, 'recurrent_state_clamp', 50.0)
        context_state_clamp = _config_float(self.config, 'context_state_clamp', 50.0)
        drift_state_clamp = _config_float(self.config, 'drift_state_clamp', 5.0)
        drift_norm_clamp = _config_nonnegative_float(self.config, 'drift_norm_clamp', 0.0)
        activation_clamp = _config_float(self.config, 'activation_clamp', 100.0)
        halt_logit_clamp = _config_float(self.config, 'halt_logit_clamp', 30.0)
        allow_hebbian_update = kwargs.pop("allow_hebbian_update", False)
        return_logits = kwargs.pop("return_logits", True)
        return_topk_values = kwargs.pop("return_topk_values", True)
        return_raw_topk_values = kwargs.pop("return_raw_topk_values", True)
        return_topk_indices = kwargs.pop("return_topk_indices", True)
        return_step_telemetry = kwargs.pop("return_step_telemetry", True)
        return_last_logit_only = kwargs.pop("return_last_logit_only", False)
        return_numerics = kwargs.pop("return_numerics", True)
        if return_last_logit_only and labels is not None:
            raise ValueError(
                "return_last_logit_only is an inference-only optimization and "
                "cannot be combined with labels"
            )
        compute_ltm_value_alignment = bool(kwargs.pop("compute_ltm_value_alignment", False))
        cached_rosa_ids = kwargs.pop("rosa_ids", None)
        cached_rosa_context_mode = kwargs.pop("rosa_ids_context_mode", None)
        advance_cached_rosa_history = bool(
            kwargs.pop("advance_cached_rosa_history", True)
        )
        loss_weights = kwargs.pop("loss_weights", None)
        prevalidated_mask_metadata = kwargs.pop(
            "_prevalidated_mask_metadata",
            None,
        )
        if prevalidated_mask_metadata is None:
            mask_has_any_padding, first_padding_column = (
                _validate_sequence_mask_contract(
                    input_ids,
                    attention_mask,
                    labels,
                    loss_weights,
                )
            )
        else:
            # The canonical trainer validates the complete CPU batch once before
            # transfer, then supplies the exact right-padding geometry for each
            # TBPTT chunk. Repeating the tensor contract audit here would force a
            # device-to-host synchronization for every chunk on CUDA. Keep this
            # hook private and validate its cheap structural invariants so direct
            # model callers continue to use the fail-closed path above.
            if (
                not isinstance(prevalidated_mask_metadata, (tuple, list))
                or len(prevalidated_mask_metadata) != 2
            ):
                raise ValueError(
                    "_prevalidated_mask_metadata must be a "
                    "(mask_has_padding, first_padding_column) pair"
                )
            mask_has_any_padding = bool(prevalidated_mask_metadata[0])
            try:
                first_padding_column = int(prevalidated_mask_metadata[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "_prevalidated_mask_metadata first_padding_column must be "
                    "an integer"
                ) from exc
            if first_padding_column < 0 or first_padding_column > input_ids.shape[1]:
                raise ValueError(
                    "_prevalidated_mask_metadata first_padding_column is outside "
                    f"the chunk: {first_padding_column} for length {input_ids.shape[1]}"
                )
            if mask_has_any_padding != (
                first_padding_column < input_ids.shape[1]
            ):
                raise ValueError(
                    "_prevalidated_mask_metadata is internally inconsistent"
                )
        if (
            not self.training
            and self.use_rosa
            and mask_has_any_padding
        ):
            raise ValueError(
                "Padded stateful inference with ROSA is not coherent because its "
                "suffix-automaton history is rectangular and would persist pad "
                "tokens. Group inference rows by equal active length (or disable "
                "ROSA) instead."
            )
        memory_write_source = int(
            kwargs.pop("memory_write_source", self.ltm.SRC_USER_INTERACTION)
        )
        memory_write_timestamp = kwargs.pop("memory_write_timestamp", None)
        memory_write_wallclock_timestamp = kwargs.pop(
            "memory_write_wallclock_timestamp",
            None,
        )
        suppress_hebbian = kwargs.pop("suppress_hebbian", getattr(self, "suppress_hebbian", True))
        hebbian_writer_ready = bool(getattr(self.config, "val_proj_trained", False))
        allow_untrained_writer = bool(
            getattr(self.config, "allow_untrained_hebbian_writer", False)
        )
        if allow_hebbian_update and (hebbian_writer_ready or allow_untrained_writer):
            suppress_hebbian = False
        elif not hebbian_writer_ready and not allow_untrained_writer:
            # Historical checkpoints never optimized val_proj. Silently writing its
            # random projection into fast memory can degrade later generations.
            suppress_hebbian = True

        # Unpack LTM Memory State early so we can use past_tokens + ROSA states
        rosa_states = None
        memory_timestamps = None
        memory_sources = None
        memory_wallclock_timestamps = None
        if ltm_memory_state is None:
            isolate_batch_ltm = getattr(self.config, 'isolate_batch_ltm', True)
            training_memory_writable = (
                self.training
                and str(
                    getattr(self.config, "ltm_training_mode", "inner-update")
                    or "inner-update"
                ).strip().lower().replace("_", "-")
                == "inner-update"
            )
            inference_memory_writable = not self.training and not suppress_hebbian
            isolate_runtime_ltm = isolate_batch_ltm and (
                training_memory_writable
                or (B > 1 and inference_memory_writable)
            )
            if isolate_runtime_ltm:
                curr_fast_vals = self.ltm.fast_vals.unsqueeze(0).expand(B, -1, -1).clone()
                curr_mom_vals = self.ltm._mom_vals.unsqueeze(0).expand(B, -1, -1).clone()
                memory_timestamps = self.ltm.timestamps.unsqueeze(0).expand(B, -1).clone()
                memory_sources = self.ltm.sources.unsqueeze(0).expand(B, -1).clone()
                memory_wallclock_timestamps = (
                    self.ltm.wallclock_timestamps.unsqueeze(0).expand(B, -1).clone()
                )
            else:
                # A read-only store is shared by definition. Keeping it 2-D is
                # important: broadcasting ``vals + fast_vals`` from a zero-stride
                # [B, slots, dim] view would still allocate B full copies at every
                # token. Writable training/chat memory takes the isolated branch
                # above and retains its [B, slots, dim] representation.
                curr_fast_vals = self.ltm.fast_vals
                curr_mom_vals = self.ltm._mom_vals
                memory_timestamps = self.ltm.timestamps
                memory_sources = self.ltm.sources
                memory_wallclock_timestamps = self.ltm.wallclock_timestamps
            past_tokens = None
        else:
            if not isinstance(ltm_memory_state, (tuple, list)):
                raise ValueError(
                    "ltm_memory_state must be a tuple/list state carrier"
                )
            state_width = len(ltm_memory_state)
            if state_width not in (2, 3, 4, 6, 7):
                raise ValueError(
                    "ltm_memory_state must contain exactly 2, 3, 4, 6, or 7 "
                    f"fields, got {state_width}"
                )
            if state_width == 7:
                (
                    curr_fast_vals,
                    curr_mom_vals,
                    past_tokens,
                    rosa_states,
                    memory_timestamps,
                    memory_sources,
                    memory_wallclock_timestamps,
                ) = ltm_memory_state
            elif state_width == 6:
                curr_fast_vals, curr_mom_vals, past_tokens, rosa_states, memory_timestamps, memory_sources = ltm_memory_state
            elif state_width == 4:
                curr_fast_vals, curr_mom_vals, past_tokens, rosa_states = ltm_memory_state
            elif state_width == 3:
                curr_fast_vals, curr_mom_vals, past_tokens = ltm_memory_state
            else:
                curr_fast_vals, curr_mom_vals = ltm_memory_state
                past_tokens = None
            if not torch.is_tensor(curr_fast_vals) or not torch.is_tensor(
                curr_mom_vals
            ):
                raise ValueError(
                    "ltm_memory_state fast and momentum values must be tensors"
                )
            if not curr_fast_vals.is_floating_point() or not curr_mom_vals.is_floating_point():
                raise ValueError(
                    "LTM fast and momentum state tensors must use floating-point dtypes"
                )
            expected_memory_shape = (
                int(self.config.ltm_slots),
                int(self.config.ltm_val_dim),
            )
            if tuple(curr_fast_vals.shape[-2:]) != expected_memory_shape:
                raise ValueError(
                    "LTM fast-state shape mismatch: "
                    f"got {tuple(curr_fast_vals.shape)}, expected "
                    f"[slots, value_dim]={expected_memory_shape} with an optional "
                    "leading batch dimension"
                )
            if tuple(curr_mom_vals.shape) != tuple(curr_fast_vals.shape):
                raise ValueError(
                    "LTM momentum shape must exactly match fast-state shape; "
                    f"got momentum={tuple(curr_mom_vals.shape)}, "
                    f"fast={tuple(curr_fast_vals.shape)}"
                )
            if curr_fast_vals.dim() not in (2, 3):
                raise ValueError(
                    "LTM fast state must have shape [slots, value_dim] or "
                    f"[batch, slots, value_dim], got {tuple(curr_fast_vals.shape)}"
                )
            if curr_fast_vals.dim() == 3 and curr_fast_vals.shape[0] not in (1, B):
                raise ValueError(
                    f"LTM state batch {curr_fast_vals.shape[0]} cannot serve input "
                    f"batch {B}"
                )

            if (memory_timestamps is None) != (memory_sources is None):
                raise ValueError(
                    "LTM timestamps and sources must either both be present or "
                    "both be omitted"
                )

            def _validate_slot_metadata(name, value, *, floating):
                if value is None:
                    return
                if not torch.is_tensor(value):
                    raise ValueError(f"LTM {name} must be a tensor or None")
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LTM {name} must have shape [slots] or [batch, slots], "
                        f"got {tuple(value.shape)}"
                    )
                if int(value.shape[-1]) != expected_memory_shape[0]:
                    raise ValueError(
                        f"LTM {name} slot count {value.shape[-1]} does not match "
                        f"configured slot count {expected_memory_shape[0]}"
                    )
                if value.dim() == 2 and int(value.shape[0]) not in (1, B):
                    raise ValueError(
                        f"LTM {name} batch {value.shape[0]} cannot serve input "
                        f"batch {B}"
                    )
                if floating and not value.is_floating_point():
                    raise ValueError(f"LTM {name} must use a floating-point dtype")
                if not floating and value.is_floating_point():
                    raise ValueError(f"LTM {name} must use an integer dtype")

            _validate_slot_metadata(
                "timestamps",
                memory_timestamps,
                floating=True,
            )
            _validate_slot_metadata(
                "sources",
                memory_sources,
                floating=False,
            )
            _validate_slot_metadata(
                "wallclock_timestamps",
                memory_wallclock_timestamps,
                floating=True,
            )

            # A shared read-only store is safe and cheap, but it must never be
            # reused for a later batched write: that would aggregate independent
            # rows into one fast-memory trajectory. Expand-and-clone only at the
            # moment writable row isolation becomes necessary.
            isolate_batch_ltm = bool(
                getattr(self.config, "isolate_batch_ltm", True)
            )
            training_memory_writable = (
                self.training
                and str(
                    getattr(self.config, "ltm_training_mode", "inner-update")
                    or "inner-update"
                ).strip().lower().replace("_", "-")
                == "inner-update"
            )
            inference_memory_writable = not self.training and not suppress_hebbian
            if (
                isolate_batch_ltm
                and B > 1
                and (training_memory_writable or inference_memory_writable)
                and (
                    curr_fast_vals.dim() == 2
                    or curr_fast_vals.shape[0] == 1
                )
            ):
                if curr_fast_vals.dim() == 2:
                    curr_fast_vals = (
                        curr_fast_vals.unsqueeze(0).expand(B, -1, -1).clone()
                    )
                    curr_mom_vals = (
                        curr_mom_vals.unsqueeze(0).expand(B, -1, -1).clone()
                    )
                else:
                    curr_fast_vals = curr_fast_vals.expand(B, -1, -1).clone()
                    curr_mom_vals = curr_mom_vals.expand(B, -1, -1).clone()

            memory_state_writable = (
                training_memory_writable or inference_memory_writable
            )

            def _isolate_metadata(value):
                if value is None:
                    return None
                if value.dim() == 1:
                    return value.unsqueeze(0).expand(B, -1).clone()
                if value.shape[0] == 1 and B > 1:
                    return value.expand(B, -1).clone()
                return value

            # Batched writable memory requires batched writable metadata too.
            # Retrieval can cheaply broadcast a one-dimensional read-only
            # metadata vector, so avoid this allocation unless writes are live.
            if memory_state_writable and curr_fast_vals.dim() == 3:
                memory_timestamps = _isolate_metadata(memory_timestamps)
                memory_sources = _isolate_metadata(memory_sources)
                memory_wallclock_timestamps = _isolate_metadata(
                    memory_wallclock_timestamps
                )
            if memory_timestamps is None:
                if curr_fast_vals.dim() == 3:
                    memory_timestamps = self.ltm.timestamps.unsqueeze(0).expand(curr_fast_vals.shape[0], -1).clone()
                    memory_sources = self.ltm.sources.unsqueeze(0).expand(curr_fast_vals.shape[0], -1).clone()
                    memory_wallclock_timestamps = (
                        self.ltm.wallclock_timestamps.unsqueeze(0)
                        .expand(curr_fast_vals.shape[0], -1)
                        .clone()
                    )
                else:
                    memory_timestamps = self.ltm.timestamps
                    memory_sources = self.ltm.sources
                    memory_wallclock_timestamps = self.ltm.wallclock_timestamps
            elif memory_wallclock_timestamps is None:
                if curr_fast_vals.dim() == 3:
                    memory_wallclock_timestamps = (
                        self.ltm.wallclock_timestamps.unsqueeze(0)
                        .expand(curr_fast_vals.shape[0], -1)
                        .clone()
                    )
                else:
                    memory_wallclock_timestamps = self.ltm.wallclock_timestamps

        # V8 ROSA Precomputation (only when enabled). Launch this before the
        # embedding lookup so CPU suffix-automaton work can overlap CUDA kernels.
        new_rosa_states = None
        rosa_finalize = None
        if self.use_rosa:
            rosa_max_ctx = getattr(self.config, 'rosa_max_context', 512)
            enforce_rosa_max_context = bool(
                getattr(self.config, "enforce_rosa_max_context", False)
            )
            expected_rosa_context_mode = (
                ROSA_BOUNDED_CONTEXT_MODE
                if enforce_rosa_max_context
                else ROSA_UNBOUNDED_CONTEXT_MODE
            )
            actual_rosa_context_mode = (
                cached_rosa_context_mode
                or ROSA_UNBOUNDED_CONTEXT_MODE
            )
            if (
                cached_rosa_ids is not None
                and actual_rosa_context_mode != expected_rosa_context_mode
            ):
                # Existing caches without a marker are legacy-unbounded. Any
                # cache/config semantic mismatch is recomputed rather than
                # silently changing the model's exact-memory input.
                cached_rosa_ids = None

            if cached_rosa_ids is None:
                # --- Datacenter-Optimized Async ROSA Pipeline ---
                # Launch CPU suffix automaton work immediately (overlaps with GPU tok_emb)
                # Uses bounded parallel batch threads, pinned memory, CUDA streams,
                # and persistent incremental automaton state across TBPTT chunks.
                rosa_finalize = rosa_async_pipeline(
                    input_ids=input_ids,
                    past_tokens=past_tokens,
                    rosa_states=rosa_states,
                    vocab_size=self.config.vocab_size,
                    device=device,
                    rosa_max_ctx=rosa_max_ctx,
                    enforce_max_context=enforce_rosa_max_context,
                )
            else:
                if cached_rosa_ids.shape != input_ids.shape:
                    raise ValueError(
                        f"Cached ROSA shape {tuple(cached_rosa_ids.shape)} does not match "
                        f"input_ids shape {tuple(input_ids.shape)}"
                    )
                no_prediction = int(self.config.vocab_size)
                cached_rosa_ids = cached_rosa_ids.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda"))
                cached_rosa_ids = torch.where(
                    (cached_rosa_ids >= 0) & (cached_rosa_ids <= no_prediction),
                    cached_rosa_ids,
                    torch.full_like(cached_rosa_ids, no_prediction),
                )

        x = self.tok_emb(input_ids)
        shared_token_features = x
        memory_gate_warmup_floor = self._memory_gate_warmup_floor(x)

        if self.use_rosa:
            if cached_rosa_ids is None:
                # Finalize: wait for CPU work, async H2D transfer
                rosa_batch_tensor, rosa_past_tokens, new_rosa_states = rosa_finalize()
                # The incremental automata are the authoritative live history.
                # Keeping a second growing token tensor here makes autoregressive
                # inference quadratic in host-copy traffic. Serialization
                # materializes a compatibility tensor once, when needed.
                new_past_tokens = (
                    rosa_past_tokens.detach().cpu()
                    if torch.is_tensor(rosa_past_tokens)
                    else None
                )
            else:
                rosa_batch_tensor = cached_rosa_ids
                if not advance_cached_rosa_history:
                    # A complete token cache already contains the deterministic
                    # ROSA prediction for every training position. Rebuilding a
                    # second history here cannot affect those predictions, but on
                    # CUDA it forces one device-to-host copy per TBPTT chunk and
                    # grows checkpoint-only metadata. Preserve any incoming
                    # compatibility history without advancing it; live chat and
                    # mixed cached/live callers retain the historical default.
                    new_past_tokens = past_tokens
                    new_rosa_states = None
                else:
                    input_ids_cpu = input_ids.detach().to(
                        device="cpu",
                        dtype=torch.long,
                    )
                    if torch.is_tensor(past_tokens):
                        past_tokens_cpu = past_tokens.detach().to(
                            device="cpu",
                            dtype=torch.long,
                        )
                        if past_tokens_cpu.dim() == 1:
                            past_tokens_cpu = past_tokens_cpu.unsqueeze(0)
                        if (
                            past_tokens_cpu.shape[0] == 1
                            and input_ids_cpu.shape[0] > 1
                        ):
                            past_tokens_cpu = past_tokens_cpu.expand(
                                input_ids_cpu.shape[0],
                                -1,
                            )
                        new_past_tokens = torch.cat(
                            [past_tokens_cpu, input_ids_cpu],
                            dim=1,
                        )
                    elif rosa_states is not None and any(
                        state is not None for state in rosa_states
                    ):
                        if len(rosa_states) != input_ids_cpu.shape[0]:
                            raise ValueError(
                                f"ROSA state count {len(rosa_states)} does not match "
                                f"batch size {input_ids_cpu.shape[0]}"
                            )
                        state_rows = []
                        history_length = None
                        for row, state in enumerate(rosa_states):
                            if not isinstance(state, ROSAState):
                                raise ValueError(
                                    "Cannot switch from live to cached ROSA with a "
                                    f"missing or invalid state at batch row {row}"
                                )
                            row_tokens = torch.tensor(
                                state.tokens,
                                dtype=torch.long,
                                device="cpu",
                            )
                            if history_length is None:
                                history_length = int(row_tokens.numel())
                            elif int(row_tokens.numel()) != history_length:
                                raise ValueError(
                                    "Cannot materialize batched ROSA history with "
                                    "different per-row lengths"
                                )
                            state_rows.append(row_tokens)
                        past_tokens_cpu = torch.stack(state_rows, dim=0)
                        new_past_tokens = torch.cat(
                            [past_tokens_cpu, input_ids_cpu],
                            dim=1,
                        )
                    else:
                        new_past_tokens = input_ids_cpu
                    if enforce_rosa_max_context and int(rosa_max_ctx or 0) > 0:
                        context_cap = int(rosa_max_ctx)
                        combined_length = int(new_past_tokens.shape[1])
                        active_length = (
                            ((combined_length - 1) % context_cap) + 1
                            if combined_length > 0
                            else 0
                        )
                        new_past_tokens = new_past_tokens[
                            :,
                            -active_length:,
                        ].contiguous()
                    # Cached IDs never advance the Python automata. Drop any
                    # stale automaton so a later live call rebuilds from the
                    # complete, checked tensor history above.
                    new_rosa_states = None

            if self.rosa_embedding_mode == "legacy-table":
                rosa_embs = self.rosa_emb(rosa_batch_tensor)
                if bool(getattr(self.config, "rosa_zero_no_prediction", False)):
                    rosa_valid = (
                        rosa_batch_tensor != int(self.config.vocab_size)
                    ).unsqueeze(-1)
                    rosa_embs = rosa_embs * rosa_valid.to(dtype=rosa_embs.dtype)
            else:
                rosa_features, rosa_valid = shared_token_lookup(
                    rosa_batch_tensor,
                    self.tok_emb.weight,
                    vocab_size=int(self.config.vocab_size),
                )
                rosa_embs = self.rosa_adapter(rosa_features) * rosa_valid
            # Per-token router controls exact-memory injection without branching.
            if self.memory_token_routers and hasattr(self, "rosa_router"):
                rosa_gate_logits = self.rosa_gate_logit + self.rosa_router(x)
            else:
                rosa_gate_logits = self.rosa_gate_logit
            rosa_gate = torch.sigmoid(_finite_clamp(rosa_gate_logits, 50.0))
            rosa_gate = self._apply_memory_gate_warmup(
                rosa_gate,
                floor=memory_gate_warmup_floor,
            )
            x = x + rosa_gate * rosa_embs  # Gated Neurosymbolic Inner Monologue Mix
        else:
            new_past_tokens = None

        # DeepEmbed is token-local, so compute it as two batched GEMMs/lookups
        # instead of launching two tiny kernels inside every recurrent token
        # iteration. The shared adapters intentionally consume the raw tied
        # embedding, before ROSA modifies ``x``.
        if self.deepembed_mode == "legacy-table":
            h_deepemb_all = self.h_deepemb(input_ids)
            l_deepemb_all = self.l_deepemb(input_ids)
        elif self.deepembed_mode == "shared-factorized":
            # Both affine-free LayerNorms are identical. Normalize the tied
            # embedding once, then feed the two learned low-rank projections.
            normalized_token_features = self.h_deepembed_adapter.norm(
                shared_token_features
            )
            h_deepemb_all = self.h_deepembed_adapter.forward_normalized(
                normalized_token_features
            )
            l_deepemb_all = self.l_deepembed_adapter.forward_normalized(
                normalized_token_features
            )
        else:
            h_deepemb_all = None
            l_deepemb_all = None


        # ==================================================================
        # 1. STATE INITIALIZATION (With Context Recovery)
        # ==================================================================
        l_state_was_provided = l_state is not None
        if h_state is None:
            if prev_context is not None or target_context is not None:
                raise ValueError(
                    "prev_context/target_context cannot be restored without an "
                    "H recurrent state; pass the complete state tuple or reset all "
                    "three values"
                )
            h_state = self.h_rnn.initial_state(B, device=device)
            prev_context = torch.zeros(B, self.config.context_dim, device=device)
            target_context = torch.zeros(B, self.config.context_dim, device=device)
        else:
            # Validate/migrate before state_hidden is allowed to derive a public
            # context. Waiting for the first recurrent call could otherwise use
            # a malformed state once before the cell rejects it.
            h_state = self.h_rnn._prepare_state(h_state, x[:, 0])
            if prev_context is None:
                prev_context = self.h_to_context(self.h_rnn.state_hidden(h_state))
            else:
                prev_context = prev_context.to(device)
                if prev_context.shape != (B, self.config.context_dim):
                    raise ValueError(
                        "prev_context must have shape "
                        f"{(B, self.config.context_dim)}, got "
                        f"{tuple(prev_context.shape)}"
                    )
            if target_context is None:
                target_context = self.h_to_context(self.h_rnn.state_hidden(h_state))
            else:
                target_context = target_context.to(device)
                if target_context.shape != (B, self.config.context_dim):
                    raise ValueError(
                        "target_context must have shape "
                        f"{(B, self.config.context_dim)}, got "
                        f"{tuple(target_context.shape)}"
                    )
        h_state = _finite_clamp(h_state, recurrent_state_clamp)
        prev_context = _finite_clamp(prev_context, context_state_clamp)
        target_context = _finite_clamp(target_context, context_state_clamp)

        if l_state is None:
            l_state = self.l_rnn.initial_state(B, device=device)
        else:
            l_state = self.l_rnn._prepare_state(l_state, x[:, 0])
        l_state = _finite_clamp(l_state, recurrent_state_clamp)

        # (ltm_memory_state already unpacked above)
        if ltm_memory_state is not None:
            # Ensure they are on the correct device
            curr_fast_vals = curr_fast_vals.to(device)
            curr_mom_vals = curr_mom_vals.to(device)
            memory_timestamps = memory_timestamps.to(device)
            memory_sources = memory_sources.to(device)
            memory_wallclock_timestamps = memory_wallclock_timestamps.to(device)

        drift_seed = None
        if drift_state is not None:
            if not torch.is_tensor(drift_state):
                raise ValueError("drift_state must be a tensor or None")
            drift_seed = drift_state.to(device)
            if drift_seed.dim() == 1:
                drift_seed = drift_seed.unsqueeze(0)
            if drift_seed.shape[0] == 1 and B > 1:
                drift_seed = drift_seed.expand(B, -1)
            if drift_seed.shape != (B, self.config.context_dim):
                raise ValueError(
                    "drift_state must have shape "
                    f"{(B, self.config.context_dim)} (or one broadcast row), "
                    f"got {tuple(drift_seed.shape)}"
                )
            drift_seed = _finite_clamp(drift_seed, drift_state_clamp)

        final_embs = [] if not return_last_logit_only else None
        last_final_emb = None
        collect_auxiliary_costs = labels is not None
        ponder_costs = []
        ponder_weights = []
        commitment_costs = []
        commitment_weights = []
        ltm_value_alignment_costs = []
        ltm_value_alignment_weights = []
        all_topk_vals = []
        all_topk_idx = []
        aux_attention_mask = attention_mask.to(device=device, dtype=torch.float32) if attention_mask is not None else None
        all_token_rows_active = torch.ones(B, device=device, dtype=torch.bool)

        ltm_value_readout = None
        if compute_ltm_value_alignment:
            memory_offset = int(self.config.context_dim + self.config.persistent_dim)
            memory_width = int(self.config.ltm_topk * self.config.ltm_val_dim)
            memory_weights = self.in_proj.weight[:, memory_offset:memory_offset + memory_width]
            if memory_weights.shape[1] != memory_width:
                raise RuntimeError(
                    f"LTM value readout width {memory_weights.shape[1]} does not match "
                    f"ltm_topk * ltm_val_dim ({memory_width})"
                )
            # A Hebbian write stores the same projected value in each selected
            # slot. Summing the corresponding in_proj blocks gives the exact
            # linear readback for that repeated value. Detach the readout and
            # target so this auxiliary trains val_proj rather than moving the
            # already-learned language path to accommodate a random writer.
            ltm_value_readout = memory_weights.reshape(
                self.config.context_dim,
                self.config.ltm_topk,
                self.config.ltm_val_dim,
            ).sum(dim=1).detach()

        stride = self.config.h_stride
        final_drift = None
        detach_freq = getattr(self.config, 'detach_every_n_steps', 32)
        drift_recurrence_mode = getattr(
            self.config,
            'drift_recurrence_mode',
            'legacy-chunk-seeded',
        )
        manager_state_commit_mode = getattr(
            self.config,
            'manager_state_commit_mode',
            'legacy-real-step',
        )
        manager_compute_mode = getattr(
            self.config,
            'manager_compute_mode',
            'soft-act',
        )
        h_halt_threshold = float(getattr(self.config, 'h_halt_thresh', 0.9))
        min_h_steps = int(getattr(self.config, 'min_h_steps', 1))
        h_effective_steps = [] if return_step_telemetry else None
        l_effective_steps = [] if return_step_telemetry else None
        persistent_batch = self.persistent.unsqueeze(0).expand(B, -1)
        time_frequencies = self.time_freqs.view(1, 1, -1)

        # ==================================================================
        # 2. MAIN TIME LOOP
        # ==================================================================
        for t in range(T):
            if (
                self.training
                and detach_freq is not None
                and t > 0
                and t % detach_freq == 0
            ):
                # This is the temporal graph boundary. Cut every recurrent
                # carrier before it participates in the boundary token, rather
                # than detaching H and L after feedback/drift already consumed
                # their old graphs.
                h_state = h_state.detach()
                l_state = l_state.detach()
                prev_context = prev_context.detach()
                target_context = target_context.detach()
                if final_drift is not None:
                    final_drift = final_drift.detach()

            if not mask_has_any_padding or t < first_padding_column:
                token_active_rows = all_token_rows_active
            else:
                token_active_rows = aux_attention_mask[:, t].to(
                    device=device,
                    dtype=torch.bool,
                )
            # If the batch contains any padding, use the row mask
            # unconditionally. This avoids a device-to-host synchronization at
            # every token while preserving exact state freezing.
            has_inactive_rows = (
                mask_has_any_padding and t >= first_padding_column
            )
            if has_inactive_rows:
                previous_h_state = h_state
                previous_l_state = l_state
                previous_prev_context = prev_context
                previous_target_context = target_context
                if final_drift is not None:
                    previous_drift = final_drift
                elif drift_seed is not None:
                    previous_drift = drift_seed
                else:
                    previous_drift = torch.zeros(
                        B,
                        self.config.context_dim,
                        device=device,
                        dtype=prev_context.dtype,
                    )

            token_x = x[:, t]
            abs_t = global_pos_offset + t
            
            if h_deepemb_all is not None:
                h_deepemb_vec = h_deepemb_all[:, t]
                l_deepemb_vec = l_deepemb_all[:, t]
            else:
                h_deepemb_vec = None
                l_deepemb_vec = None
            
            # --- LTM Retrieval ---
            q_in = torch.cat([token_x, prev_context], dim=-1)
            q = _finite_clamp(self.qproj(q_in), 12.0)
            
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(
                q, self.config.ltm_topk, min_timestamp, source_filter, fast_vals=curr_fast_vals,
                timestamps=memory_timestamps, sources=memory_sources,
                min_wallclock_timestamp=min_wallclock_timestamp,
                wallclock_timestamps=memory_wallclock_timestamps,
            )
            topk_vals = topk_vals + (
                q.sum(dim=-1, keepdim=True) * 0.0
            ).unsqueeze(-1)
            
            if return_topk_values or return_raw_topk_values:
                all_topk_vals.append(topk_vals)
            if return_topk_indices:
                all_topk_idx.append(topk_idx)
            
            # Positional encoding
            args = topk_ts.unsqueeze(-1) * time_frequencies
            pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.config.ltm_val_dim % 2 == 1:
                pe = torch.cat([pe, pe.new_zeros(*pe.shape[:-1], 1)], dim=-1)
            valid_memory = (topk_idx >= 0).unsqueeze(-1)
            topk_vals = (topk_vals + pe) * valid_memory.to(dtype=topk_vals.dtype)
            
            if self.memory_token_routers and hasattr(self, "ltm_router"):
                gate_input = self.ltm_gate_logit + self.ltm_router(token_x)
            else:
                gate_input = self.ltm_gate_logit
            gate = torch.sigmoid(_finite_clamp(gate_input, 50.0))
            gate = self._apply_memory_gate_warmup(
                gate,
                floor=memory_gate_warmup_floor,
            )
            if gate.dim() == 2:
                gate = gate.unsqueeze(1)
            gated_vals = topk_vals * gate
            mac_in = torch.cat(
                [token_x, persistent_batch, gated_vals.view(B, -1)],
                dim=-1,
            )
            
            enc = F.gelu(self.in_proj(mac_in))
            enc = _finite_clamp(enc, 30.0)

            # ==================================================================
            # 3. HIERARCHICAL MANAGER (Continuous Watch, Strided Plan)
            # ==================================================================
            l_feedback = self.l_feedback_proj(self.l_rnn.state_hidden(l_state).to(device))
            enc_with_feedback = _finite_clamp(enc + l_feedback, activation_clamp)

            h_out_real, h_state = self.h_rnn(enc_with_feedback, h_state, timestep=None, deepemb_vec=h_deepemb_vec)
            h_out_real = _finite_clamp(h_out_real, activation_clamp)
            
            if getattr(self.config, 'debug_numerics', False) and (torch.isnan(h_out_real).any() or torch.isinf(h_out_real).any()):
                print(f"WARNING: NaN/Inf detected in h_out_real at step {t}")
            
            step_ponder_cost = torch.zeros(B, device=device, dtype=enc.dtype)
            h_steps_this_token = torch.ones(
                B,
                device=device,
                dtype=torch.float32,
            )
            
            # PLANNING STEP (Strided with ACT)
            if abs_t % stride == 0:
                prev_context = _finite_clamp(target_context, context_state_clamp)

                # Pondering on Shadow State
                h_step_outputs = [h_out_real]
                h_step_states = [h_state]
                halt_logit = _finite_clamp(self.h_halt_proj(h_out_real).squeeze(-1), halt_logit_clamp)
                h_halt_probs = [torch.sigmoid(halt_logit).clamp(1e-6, 1.0 - 1e-6)]
                
                shadow_h_state = h_state
                current_enc_h = enc_with_feedback
                hard_survival = 1.0 - h_halt_probs[0]
                hard_halted = (
                    (1.0 - hard_survival) >= h_halt_threshold
                    if min_h_steps <= 1
                    else torch.zeros_like(h_halt_probs[0], dtype=torch.bool)
                )

                inference_logit_parity = bool(
                    getattr(self.config, 'inference_logit_parity', False)
                    or getattr(self.config, 'full_sample_bptt', False)
                )
                for step_idx in range(self.config.max_h_steps - 1):
                    if not self.training:
                        if not bool(token_active_rows.any().item()):
                            break
                        if manager_compute_mode == "hard-masked":
                            # Hard selection is invariant to uncomputed later
                            # candidates after every active row has selected a
                            # state, so this eager exit preserves train logits.
                            if bool(hard_halted[token_active_rows].all().item()):
                                break
                        elif (
                            not inference_logit_parity
                            and h_halt_probs[-1][token_active_rows].mean()
                            > h_halt_threshold
                        ):
                            break
                    h_out_ponder, shadow_h_state = self.h_rnn(current_enc_h, shadow_h_state, timestep=None, deepemb_vec=h_deepemb_vec)
                    h_out_ponder = _finite_clamp(h_out_ponder, activation_clamp)
                    halt_logit = _finite_clamp(self.h_halt_proj(h_out_ponder).squeeze(-1), halt_logit_clamp)
                    h_step_outputs.append(h_out_ponder)
                    h_step_states.append(shadow_h_state)
                    h_halt_probs.append(torch.sigmoid(halt_logit).clamp(1e-6, 1.0 - 1e-6))
                    hard_survival = hard_survival * (1.0 - h_halt_probs[-1])
                    completed_steps = step_idx + 2
                    if completed_steps >= min_h_steps:
                        hard_halted = hard_halted | (
                            (1.0 - hard_survival) >= h_halt_threshold
                        )

                h_stack = torch.stack(h_step_outputs, dim=0).float()
                halt_stack = torch.stack(h_halt_probs, dim=0).float()
                act = (
                    None
                    if manager_compute_mode == "hard-masked"
                    else normalized_act_weights(halt_stack)
                )
                h_state_stack = torch.stack(h_step_states, dim=0)
                if manager_compute_mode == "hard-masked":
                    selection = hard_act_selection(
                        h_stack,
                        h_state_stack,
                        halt_stack,
                        threshold=h_halt_threshold,
                        min_steps=min(min_h_steps, len(h_step_outputs)),
                    )
                    final_h_out = selection.output
                    h_state = selection.state.to(dtype=h_step_states[0].dtype)
                    h_steps_this_token = selection.executed_steps
                    step_ponder_cost = hard_act_depth_straight_through(
                        halt_stack,
                        selection.executed_steps,
                        threshold=h_halt_threshold,
                        min_steps=min(
                            min_h_steps,
                            len(h_step_outputs),
                        ),
                        temperature=float(
                            getattr(
                                self.config,
                                "act_depth_temperature",
                                0.05,
                            )
                        ),
                    ).to(enc.dtype)
                else:
                    final_h_out = (
                        (act.weights.unsqueeze(-1) * h_stack).sum(dim=0)
                        + act.remainder.unsqueeze(-1) * h_stack[-1]
                    )
                    step_ponder_cost = act.expected_steps.to(enc.dtype)
                # ACT helpers defensively define a selection for malformed
                # probabilities, but the model must not turn a NaN halt head
                # into valid-looking logits. This zero-valued dependency keeps
                # finite trajectories unchanged and propagates any non-finite
                # halt value to the raw-logit rejection guard without a host
                # synchronization.
                final_h_out = final_h_out + (
                    halt_stack.sum(dim=0) * 0.0
                ).unsqueeze(-1)
                if getattr(self.config, 'debug_numerics', False):
                    if bool((~torch.isfinite(final_h_out)).any().item()):
                        print(
                            "WARNING: Non-finite manager output at token step "
                            f"{abs_t}; preserving it for fail-closed rejection."
                        )
                final_h_out = _finite_clamp(final_h_out, activation_clamp).to(enc.dtype)  # Cast back to working precision

                if manager_state_commit_mode == 'act-weighted':
                    h_state_stack_float = h_state_stack.float()
                    state_weights = act.weights.unsqueeze(-1).unsqueeze(-1)
                    state_remainder = act.remainder.unsqueeze(-1).unsqueeze(-1)
                    h_state = (
                        (state_weights * h_state_stack_float).sum(dim=0)
                        + state_remainder * h_state_stack_float[-1]
                    ).to(dtype=h_step_states[0].dtype)
                    h_state = _finite_clamp(h_state, recurrent_state_clamp)
                elif manager_state_commit_mode == 'last-shadow':
                    h_state = _finite_clamp(
                        h_step_states[-1],
                        recurrent_state_clamp,
                    )
                
                target_context = self.h_to_context(final_h_out)
                target_context = _finite_clamp(target_context, context_state_clamp)
                
                if collect_auxiliary_costs:
                    ponder_costs.append(step_ponder_cost)
                    if aux_attention_mask is not None:
                        ponder_weights.append(aux_attention_mask[:, t])
                if manager_compute_mode != "hard-masked":
                    h_steps_this_token = torch.full(
                        (B,),
                        float(len(h_step_outputs)),
                        device=device,
                        dtype=torch.float32,
                    )

            if return_step_telemetry:
                h_effective_steps.append(
                    h_steps_this_token * token_active_rows.float()
                )
            
            # LERP (Interpolation)
            step_in_stride = abs_t % stride
            alpha = step_in_stride / float(stride)
            sliding_context = (prev_context.float() + alpha * (target_context.float() - prev_context.float())).to(prev_context.dtype)
            sliding_context = _finite_clamp(sliding_context, context_state_clamp)

            # ==================================================================
            # 4. WORKER STEP
            # ==================================================================
            use_external_drift_seed = (
                t == 0
                and drift_seed is not None
                and (
                    drift_recurrence_mode == 'legacy-chunk-seeded'
                    or not l_state_was_provided
                )
            )
            if use_external_drift_seed:
                initial_drift = _l2_norm_clamp(_finite_clamp(drift_seed, drift_state_clamp), drift_norm_clamp)
            elif self.context_drift_proj is not None:
                prev_worker_h = self.l_rnn.state_hidden(l_state).to(device)
                initial_drift = torch.tanh(self.context_drift_proj(prev_worker_h))
                initial_drift = _l2_norm_clamp(_finite_clamp(initial_drift, drift_state_clamp), drift_norm_clamp)
            else:
                initial_drift = torch.zeros(B, self.config.context_dim, device=device)

            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                enc, l_state, cc, final_drift, l_steps_this_token = checkpoint(
                    self.worker_loop_module, enc, sliding_context, l_state, initial_drift, None, l_deepemb_vec,
                    use_reentrant=False
                )
            else:
                enc, l_state, cc, final_drift, l_steps_this_token = self.worker_loop_module(
                    enc, sliding_context, l_state, initial_drift, timestep=None, l_deepemb_vec=l_deepemb_vec
                )
            enc = _finite_clamp(enc, activation_clamp)
            final_drift = _l2_norm_clamp(_finite_clamp(final_drift, drift_state_clamp), drift_norm_clamp)
            if return_step_telemetry:
                l_effective_steps.append(
                    l_steps_this_token.float() * token_active_rows.float()
                )
            if has_inactive_rows:
                h_state = _keep_active_rows(
                    h_state,
                    previous_h_state,
                    token_active_rows,
                )
                l_state = _keep_active_rows(
                    l_state,
                    previous_l_state,
                    token_active_rows,
                )
                prev_context = _keep_active_rows(
                    prev_context,
                    previous_prev_context,
                    token_active_rows,
                )
                target_context = _keep_active_rows(
                    target_context,
                    previous_target_context,
                    token_active_rows,
                )
                final_drift = _keep_active_rows(
                    final_drift,
                    previous_drift,
                    token_active_rows,
                )
                cc = torch.where(
                    token_active_rows,
                    cc,
                    torch.zeros_like(cc),
                )

            if ltm_value_readout is not None:
                value_to_store = self.val_proj(enc.detach())
                memory_readback = F.linear(value_to_store, ltm_value_readout)
                target_value = enc.detach().float()
                squared_error = (memory_readback.float() - target_value).square().mean(dim=-1)
                target_energy = target_value.square().mean(dim=-1).clamp_min(1e-4)
                alignment_cost = squared_error / target_energy
                ltm_value_alignment_costs.append(alignment_cost)
                if aux_attention_mask is not None:
                    ltm_value_alignment_weights.append(aux_attention_mask[:, t])
            
            if return_last_logit_only:
                if has_inactive_rows:
                    if last_final_emb is None:
                        last_final_emb = torch.zeros_like(enc)
                    last_final_emb = torch.where(
                        token_active_rows.unsqueeze(-1),
                        enc,
                        last_final_emb,
                    )
                else:
                    last_final_emb = enc
            else:
                final_embs.append(enc)
            if collect_auxiliary_costs:
                commitment_costs.append(cc)
                if aux_attention_mask is not None:
                    commitment_weights.append(aux_attention_mask[:, t])

            # ==================================================================
            # 5. MEMORY UPDATE (Differentiable Hebbian — Inference Only)
            # ==================================================================
            # During training, the trainer handles LTM updates via gradient-based
            # Titans inner_update after backward(). Running Hebbian here too would
            # cause double-decay and conflicting update signals on the same slots.
            if not self.training and not suppress_hebbian:
                val_to_store = self.val_proj(enc)
                val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                write_topk_idx = topk_idx.masked_fill(
                    ~token_active_rows.unsqueeze(-1),
                    -1,
                )
                
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    write_topk_idx, None, val_expanded,
                    current_lr=getattr(self.config, 'ltm_lr', 0.001),  # COH #1: Use config LR, not hardcoded 0.01
                    timestamp=(
                        float(memory_write_timestamp)
                        if memory_write_timestamp is not None
                        else float(abs_t)
                    ),
                    source=memory_write_source,
                    tokens_covered=1,
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals,
                    timestamps=memory_timestamps,
                    sources=memory_sources,
                    wallclock_timestamp=memory_write_wallclock_timestamp,
                    wallclock_timestamps=memory_wallclock_timestamps,
                    inplace=True
                )

        # ==================================================================
        # 5. FINAL OUTPUTS
        # ==================================================================
        if return_last_logit_only:
            final = _finite_clamp(
                self.out_norm(last_final_emb.unsqueeze(1)),
                activation_clamp,
            )
        else:
            final = _finite_clamp(
                self.out_norm(torch.stack(final_embs, dim=1)),
                activation_clamp,
            )
        logits = None

        loss = None
        ponder_cost_out = None
        commitment_cost_out = None
        ltm_value_alignment_cost_out = None
        logit_numerics = None
        logit_saturation_threshold = _config_float(
            self.config,
            'logit_saturation_threshold',
            30.0,
        )

        if labels is not None and not return_logits:
            chunked_loss_result = self._compute_cuda_chunked_lm_loss(
                final,
                labels,
                getattr(self.config, 'z_loss_weight', 1e-4),
                loss_weights=loss_weights,
                return_telemetry=return_numerics,
            )
            if return_numerics:
                loss, logit_numerics = chunked_loss_result
            else:
                loss = chunked_loss_result
        else:
            raw_logits = self.lm_head(final)
            if return_numerics:
                logit_numerics = _logit_numerics(
                    raw_logits,
                    logit_saturation_threshold,
                )
                has_nonfinite_logits = bool(
                    (logit_numerics["raw_logit_nonfinite_count"] > 0).item()
                )
            else:
                has_nonfinite_logits = not bool(
                    torch.isfinite(raw_logits.detach()).all().item()
                )
            inference_logit_clamp = float(
                getattr(self.config, 'inference_logit_clamp', 30.0)
            )
            if has_nonfinite_logits:
                if self.training or inference_logit_clamp <= 0.0:
                    raise FloatingPointError(
                        "Non-finite raw language-model logits detected; refusing "
                        "to hide the numerical failure with sanitization"
                    )
                if getattr(self.config, 'debug_numerics', False):
                    print(
                        "WARNING: NaN/Inf detected in legacy inference logits; "
                        "sanitizing under the configured compatibility clamp."
                    )
                raw_logits = torch.nan_to_num(
                    raw_logits,
                    nan=0.0,
                    posinf=inference_logit_clamp,
                    neginf=-inference_logit_clamp,
                )

            # Training always optimizes the raw logits. Historical inference
            # retains its old clamp unless a corrected recurrence contract (or
            # explicit inference_logit_clamp=0) disables it.
            if not self.training and inference_logit_clamp > 0.0:
                logits = torch.clamp(
                    raw_logits,
                    min=-inference_logit_clamp,
                    max=inference_logit_clamp,
                )
            else:
                logits = raw_logits

            if labels is not None:
                if labels.ndim != 2 or labels.shape[0] != logits.shape[0]:
                    raise ValueError(
                        f"labels shape {tuple(labels.shape)} is incompatible with "
                        f"logits shape {tuple(logits.shape)}"
                    )
                if labels.shape[1] > logits.shape[1] + 1:
                    raise ValueError(
                        "labels may be at most one token longer than input_ids for "
                        "chunk-boundary lookahead loss"
                    )
                loss_hidden_len = min(logits.shape[1], max(0, labels.shape[1] - 1))
                shift_logits = logits[..., :loss_hidden_len, :].contiguous()
                shift_labels = labels[..., 1:1 + loss_hidden_len].contiguous()
                shift_weights = None
                if loss_weights is not None:
                    loss_weights = loss_weights.to(device=device, dtype=torch.float32)
                    if loss_weights.ndim != 2 or loss_weights.shape[0] != logits.shape[0]:
                        raise ValueError(
                            f"loss_weights shape {tuple(loss_weights.shape)} is incompatible with "
                            f"logits shape {tuple(logits.shape)}"
                        )
                    if loss_weights.shape[1] < labels.shape[1]:
                        pad_cols = labels.shape[1] - loss_weights.shape[1]
                        loss_weights = F.pad(loss_weights, (0, pad_cols), value=0.0)
                    shift_weights = loss_weights[..., 1:1 + loss_hidden_len].contiguous()
                
                valid_mask = shift_labels != -100
                flat_logits = shift_logits.view(-1, self.config.vocab_size).float()
                flat_labels = shift_labels.view(-1)
                flat_ce = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    reduction="none",
                    ignore_index=-100,
                )
                valid_weight = valid_mask.view(-1).float()
                if shift_weights is not None:
                    valid_weight = valid_weight * shift_weights.view(-1).float()
                denom = valid_weight.sum().clamp_min(1e-8)

                # Base CE loss, optionally weighted toward assistant response
                # tokens. Cross-entropy emits zero for ignored rows, so the same
                # tensor-only reduction also yields a connected zero loss for an
                # empty supervised slice without a device-to-host branch.
                loss = (flat_ce * valid_weight).sum() / denom

                # Z-Loss Regularization built to prevent exploding logits
                z_loss_weight = getattr(self.config, 'z_loss_weight', 1e-4)
                if z_loss_weight > 0:
                    # AMP FIX: Disable autocast for the z-loss block. Boolean indexing
                    # (flat_logits[valid_mask_flat]) uses masked_scatter_ in its backward
                    # pass. Under BFloat16 AMP, logsumexp can produce BF16 gradients that
                    # flow back into the float32 flat_logits via masked_scatter_, crashing
                    # with "expected self and source to have same dtypes".
                    _zloss_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'
                    with torch.amp.autocast(device_type=_zloss_device, enabled=False):
                        row_z = torch.logsumexp(flat_logits, dim=-1).pow(2)
                        z_loss = ((row_z * valid_weight).sum() / denom) * z_loss_weight
                    loss = loss + z_loss

        if labels is not None:
            
            # Compute auxiliary costs for reporting (trainer handles loss composition)
            def _weighted_aux_mean(costs, weights):
                if not costs:
                    return None
                cost_tensor = torch.stack([c.float().view(B) for c in costs], dim=0)
                if not weights:
                    return cost_tensor.mean()
                weight_tensor = torch.stack(weights, dim=0).float()
                denom = weight_tensor.sum()
                weighted_sum = (cost_tensor * weight_tensor).sum()
                safe_mean = weighted_sum / denom.clamp_min(1.0)
                return torch.where(
                    denom > 0,
                    safe_mean,
                    torch.zeros((), device=device, dtype=cost_tensor.dtype),
                )

            ponder_cost_out = _weighted_aux_mean(ponder_costs, ponder_weights)
            commitment_cost_out = _weighted_aux_mean(commitment_costs, commitment_weights)
            ltm_value_alignment_cost_out = _weighted_aux_mean(
                ltm_value_alignment_costs,
                ltm_value_alignment_weights,
            )

        h_state = _finite_clamp(h_state, recurrent_state_clamp)
        l_state = _finite_clamp(l_state, recurrent_state_clamp)
        prev_context = _finite_clamp(prev_context, context_state_clamp)
        target_context = _finite_clamp(target_context, context_state_clamp)
        final_drift = _l2_norm_clamp(_finite_clamp(final_drift, drift_state_clamp), drift_norm_clamp)

        return {
            "loss": loss, 
            "logits": logits, 
            "ponder_cost": ponder_cost_out, 
            "commitment_cost": commitment_cost_out,
            "ltm_value_alignment_cost": ltm_value_alignment_cost_out,
            "numerics": logit_numerics,
            "step_telemetry": (
                {
                    "h_effective_steps": torch.stack(h_effective_steps, dim=1),
                    "l_effective_steps": torch.stack(l_effective_steps, dim=1),
                }
                if return_step_telemetry
                else None
            ),
            "topk_vals": torch.stack(all_topk_vals, dim=1) if (return_topk_values and all_topk_vals) else None, 
            "raw_topk_vals": all_topk_vals if return_raw_topk_values else None,
            "topk_idx": torch.stack(all_topk_idx, dim=1) if all_topk_idx else None,
            "h_state": h_state,
            "l_state": l_state,
            "prev_context": prev_context,
            "target_context": target_context,
            "drift_state": final_drift,
            "ltm_memory_state": (
                curr_fast_vals,
                curr_mom_vals,
                new_past_tokens,
                new_rosa_states,
                memory_timestamps,
                memory_sources,
                memory_wallclock_timestamps,
            ),
        }

    def _compute_cuda_chunked_lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor,
                                      z_loss_weight: float = 1e-4,
                                      loss_weights: Optional[torch.Tensor] = None,
                                      return_telemetry: bool = False):
        """
        Memory-friendly supervised-row loss path for large vocabularies.

        This intentionally recomputes lm_head by row chunks instead of materializing
        the full shifted logits tensor for loss calculation. The reduction matches
        PyTorch's mean cross-entropy with ignore_index=-100, and the z-loss is
        averaged over the same valid-token rows as the dense path.
        """
        if labels.ndim != 2 or labels.shape[0] != hidden.shape[0]:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} is incompatible with "
                f"hidden shape {tuple(hidden.shape)}"
            )
        if labels.shape[1] > hidden.shape[1] + 1:
            raise ValueError(
                "labels may be at most one token longer than hidden for "
                "chunk-boundary lookahead loss"
            )
        loss_hidden_len = min(hidden.shape[1], max(0, labels.shape[1] - 1))
        shift_hidden = hidden[:, :loss_hidden_len, :].contiguous()
        shift_labels = labels[:, 1:1 + loss_hidden_len].contiguous()
        shift_weights = None
        if loss_weights is not None:
            loss_weights = loss_weights.to(device=hidden.device, dtype=torch.float32)
            if loss_weights.ndim != 2 or loss_weights.shape[0] != hidden.shape[0]:
                raise ValueError(
                    f"loss_weights shape {tuple(loss_weights.shape)} is incompatible with "
                    f"hidden shape {tuple(hidden.shape)}"
                )
            if loss_weights.shape[1] < labels.shape[1]:
                pad_cols = labels.shape[1] - loss_weights.shape[1]
                loss_weights = F.pad(loss_weights, (0, pad_cols), value=0.0)
            shift_weights = loss_weights[:, 1:1 + loss_hidden_len].contiguous()
        flat_hidden = shift_hidden.view(-1, hidden.shape[-1])
        flat_labels = shift_labels.view(-1)

        valid_mask = flat_labels != -100
        valid_hidden = flat_hidden[valid_mask]
        valid_labels = flat_labels[valid_mask]
        valid_weights = None
        if shift_weights is not None:
            valid_weights = shift_weights.view(-1)[valid_mask].float()
        valid_count = valid_labels.shape[0]
        if valid_count == 0:
            zero_loss = hidden.sum() * 0.0
            if not return_telemetry:
                return zero_loss
            empty_numerics = _logit_numerics(
                hidden.new_empty((0, self.config.vocab_size)),
                _config_float(
                    self.config,
                    'logit_saturation_threshold',
                    30.0,
                ),
            )
            return zero_loss, empty_numerics
        if valid_weights is None:
            denom = torch.tensor(float(valid_count), device=hidden.device, dtype=torch.float32)
        else:
            denom = valid_weights.sum().clamp_min(1e-8)

        if hidden.device.type == "cpu":
            chunk_rows = int(getattr(self.config, "cpu_loss_chunk_rows", 0) or 0)
        else:
            chunk_rows = int(getattr(self.config, "cuda_loss_chunk_rows", 0) or 0)
        if chunk_rows <= 0:
            chunk_rows = flat_hidden.shape[0]

        total_ce = torch.zeros((), device=hidden.device, dtype=torch.float32)
        total_z = torch.zeros((), device=hidden.device, dtype=torch.float32)
        saturation_threshold = _config_float(
            self.config,
            'logit_saturation_threshold',
            30.0,
        )
        raw_logit_max_abs = (
            torch.zeros((), device=hidden.device, dtype=torch.float32)
            if return_telemetry
            else None
        )
        raw_logit_nonfinite_count = torch.zeros(
            (),
            device=hidden.device,
            dtype=torch.long,
        )
        raw_logit_saturation_count = (
            torch.zeros((), device=hidden.device, dtype=torch.long)
            if return_telemetry
            else None
        )
        raw_logit_count = 0

        for start in range(0, valid_count, chunk_rows):
            end = min(start + chunk_rows, valid_count)
            chunk_hidden = valid_hidden[start:end]
            chunk_labels = valid_labels[start:end]
            chunk_weights = valid_weights[start:end] if valid_weights is not None else None
            chunk_logits = self.lm_head(chunk_hidden).float()
            chunk_finite = torch.isfinite(chunk_logits.detach())
            chunk_nonfinite_count = (~chunk_finite).sum()
            raw_logit_nonfinite_count = (
                raw_logit_nonfinite_count + chunk_nonfinite_count
            )
            if return_telemetry:
                finite_abs = torch.where(
                    chunk_finite,
                    chunk_logits.detach().abs(),
                    torch.zeros_like(chunk_logits.detach()),
                )
                raw_logit_max_abs = torch.maximum(
                    raw_logit_max_abs,
                    finite_abs.amax(),
                )
                raw_logit_saturation_count = (
                    raw_logit_saturation_count
                    + (finite_abs >= saturation_threshold).sum()
                )
                raw_logit_count += chunk_logits.numel()
            if chunk_weights is None:
                total_ce = total_ce + F.cross_entropy(chunk_logits, chunk_labels, reduction="sum")
            else:
                chunk_ce = F.cross_entropy(chunk_logits, chunk_labels, reduction="none")
                total_ce = total_ce + (chunk_ce * chunk_weights).sum()

            if z_loss_weight > 0:
                row_z = torch.logsumexp(chunk_logits, dim=-1).pow(2)
                if chunk_weights is None:
                    total_z = total_z + row_z.sum()
                else:
                    total_z = total_z + (row_z * chunk_weights).sum()

        # Synchronize once after all row chunks, rather than once per chunk.
        # The failure contract is unchanged: no non-finite training trajectory is
        # returned to the trainer or allowed to reach an optimizer step.
        if bool((raw_logit_nonfinite_count > 0).item()):
            raise FloatingPointError(
                "Non-finite raw language-model logits detected in chunked "
                "training loss; refusing to sanitize the training graph"
            )

        loss = total_ce / denom
        if z_loss_weight > 0:
            loss = loss + (total_z / denom) * z_loss_weight
        if not return_telemetry:
            return loss
        numerics = {
            "raw_logit_max_abs": raw_logit_max_abs,
            "raw_logit_nonfinite_count": raw_logit_nonfinite_count,
            "raw_logit_saturation_fraction": (
                raw_logit_saturation_count.float()
                / max(1, raw_logit_count)
            ),
            "raw_logit_saturation_threshold": torch.tensor(
                saturation_threshold,
                device=hidden.device,
                dtype=torch.float32,
            ),
        }
        return loss, numerics

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def update_memory(self, topk_idx, grads, timestamp, lr=1e-3):
        """Updates the LTM memory using gradients (Titans style)."""
        self.ltm.inner_update(topk_idx, grads, current_lr=lr, timestamp=timestamp, inplace=True)

    def update_memory_hebbian(self, topk_idx, vals, timestamp, lr=1e-3, tokens_covered=1):
        """Updates the LTM memory using Hebbian rule (Fallback for Inference)."""
        self.ltm.update_memory_hebbian(topk_idx, None, vals, current_lr=lr, 
                                       timestamp=timestamp, tokens_covered=tokens_covered, inplace=True)
