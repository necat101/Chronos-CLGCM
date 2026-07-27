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
from ..utils.device import setup_msvc_environment, is_directml_device
from ..utils.rosa import ROSA, rosa_async_pipeline, ROSAState


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


def _validate_architecture_config(config) -> None:
    """Fail before allocation when a requested geometry cannot execute coherently."""
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

    # The manager input is enc + l_feedback and therefore has context_dim
    # features. Supporting a different manager width would require a new learned
    # projection and would break existing checkpoint layouts.
    if values["h_hidden"] != values["context_dim"]:
        raise ValueError(
            "Hierarchos currently requires h_hidden == context_dim because the "
            f"manager consumes context-width residuals; got h_hidden={values['h_hidden']} "
            f"and context_dim={values['context_dim']}."
        )

    requested_head = _config_value(config, "rwkv_head_size", None)
    if requested_head not in (None, 0, "", "auto"):
        try:
            requested_head = int(requested_head)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rwkv_head_size must be a positive divisor or auto, got {requested_head!r}") from exc
        if requested_head <= 0:
            raise ValueError(f"rwkv_head_size must be a positive divisor or auto, got {requested_head!r}")
        for width_name in ("h_hidden", "l_hidden"):
            if values[width_name] % requested_head != 0:
                raise ValueError(
                    f"rwkv_head_size={requested_head} does not divide {width_name}={values[width_name]}."
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
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    max_abs = float(max_abs)
    return torch.clamp(
        torch.nan_to_num(tensor, nan=nan, posinf=max_abs, neginf=-max_abs),
        min=-max_abs,
        max=max_abs,
    )

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
        self.refresh_runtime_config()

    def refresh_runtime_config(self):
        config = self.config
        self.max_l_steps = config.max_l_steps
        self.l_conv_atol = getattr(config, 'l_conv_atol', 0.01)
        self.commitment_threshold = getattr(config, 'commitment_threshold', 0.1)
        self.recurrent_state_clamp = _config_float(config, 'recurrent_state_clamp', 50.0)
        self.context_state_clamp = _config_float(config, 'context_state_clamp', 50.0)
        self.drift_state_clamp = _config_float(config, 'drift_state_clamp', 5.0)
        self.drift_norm_clamp = _config_nonnegative_float(config, 'drift_norm_clamp', 0.0)
        self.drift_delta_scale = _config_float(config, 'drift_delta_scale', 1.0)
        self.activation_clamp = _config_float(config, 'activation_clamp', 100.0)
        static_loop = getattr(config, 'compile_static_worker_loop', False)
        self.compile_static_worker_loop = bool(static_loop) if static_loop is not None else False

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
        drift_costs = [] 
        current_enc = enc

        # Shadow state for exploration (pondering)
        shadow_l_state = l_state.clone()
        
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
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1), deepemb_vec=l_deepemb_vec)
                l_out = _finite_clamp(l_out, self.activation_clamp)
                shadow_l_state = _finite_clamp(shadow_l_state, self.recurrent_state_clamp)
                
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
            commitment_cost_static = None
            if self.compile_static_worker_loop:
                # Convergence must be sample-local. A scalar active flag coupled
                # every row in the batch, so a padded/easy peer could change a
                # valid row's refinement depth and logits.
                active = torch.ones(enc.shape[0], device=enc.device, dtype=enc.dtype)
                drift_cost_sum = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)
                drift_cost_count = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)

                for step_idx in range(self.max_l_steps):
                    prev_shadow = shadow_l_state
                    prev_drift = current_drift
                    prev_l_input = l_input

                    l_out, candidate_shadow = self.l_rnn(l_input, shadow_l_state, timestep=None, deepemb_vec=l_deepemb_vec)
                    l_out = _finite_clamp(l_out, self.activation_clamp)
                    candidate_shadow = _finite_clamp(candidate_shadow, self.recurrent_state_clamp)

                    drift_delta = torch.tanh(self.context_drift_proj(l_out)) * self.drift_delta_scale
                    candidate_drift = _l2_norm_clamp(_finite_clamp(current_drift + drift_delta, self.drift_state_clamp), self.drift_norm_clamp)
                    drift_sq = torch.sum(candidate_drift ** 2, dim=-1)
                    hinge_cost = torch.relu(drift_sq - self.commitment_threshold)
                    hinge_cost = torch.clamp(hinge_cost, max=100.0)

                    drift_cost_sum = drift_cost_sum + hinge_cost * active
                    drift_cost_count = drift_cost_count + active

                    candidate_dynamic = static_context + candidate_drift
                    candidate_input_vec = torch.cat([current_enc, candidate_dynamic], dim=-1)
                    candidate_l_input = _finite_clamp(self.l_input_proj(candidate_input_vec), self.recurrent_state_clamp)

                    keep2 = active.unsqueeze(-1)
                    keep3 = active.view(-1, 1, 1)
                    shadow_l_state = candidate_shadow * keep3 + prev_shadow * (1.0 - keep3)
                    current_drift = candidate_drift * keep2 + prev_drift * (1.0 - keep2)
                    l_input = candidate_l_input * keep2 + prev_l_input * (1.0 - keep2)

                    still_active = (
                        torch.mean(torch.abs(drift_delta), dim=-1) >= self.l_conv_atol
                    ).to(dtype=enc.dtype)
                    active = active * still_active

                commitment_cost_static = drift_cost_sum / torch.clamp(drift_cost_count, min=1.0)
            else:
                commitment_cost_static = None

        if use_training_refinement and not self.compile_static_worker_loop:
            for step_idx in range(self.max_l_steps):
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=None, deepemb_vec=l_deepemb_vec)
                l_out = _finite_clamp(l_out, self.activation_clamp)
                shadow_l_state = _finite_clamp(shadow_l_state, self.recurrent_state_clamp)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out)) * self.drift_delta_scale
                current_drift = _l2_norm_clamp(_finite_clamp(current_drift + drift_delta, self.drift_state_clamp), self.drift_norm_clamp)
                drift_sq = torch.sum(current_drift ** 2, dim=-1)
                hinge_cost = torch.relu(drift_sq - self.commitment_threshold)
                hinge_cost = torch.clamp(hinge_cost, max=100.0)
                drift_costs.append(hinge_cost)
                
                # Update dynamic_context BEFORE convergence check (matches inference path)
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = _finite_clamp(self.l_input_proj(l_input_vec), self.recurrent_state_clamp)
                
                if torch.mean(torch.abs(drift_delta)) < self.l_conv_atol:
                   break

        # Use original l_state (not shadow) for the actual state update
        ts = timestep if timestep is not None else 0
        
        # Recalculate l_input with final drift
        dynamic_context = static_context + current_drift
        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = _finite_clamp(self.l_input_proj(l_input_vec), self.recurrent_state_clamp)
        
        final_l_out, next_l_state = self.l_rnn(l_input, l_state, timestep=None, deepemb_vec=l_deepemb_vec)
        final_l_out = _finite_clamp(final_l_out, self.activation_clamp)
        next_l_state = _finite_clamp(next_l_state, self.recurrent_state_clamp)
        
        final_enc = current_enc + self.l_to_out(final_l_out)
        final_enc = _finite_clamp(final_enc, self.activation_clamp)
        commitment_cost = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)
        if use_training_refinement and self.compile_static_worker_loop and commitment_cost_static is not None:
            commitment_cost = commitment_cost_static
        elif len(drift_costs) > 0:
            commitment_cost = torch.stack(drift_costs, dim=0).mean(dim=0)

        final_drift = _l2_norm_clamp(_finite_clamp(current_drift, self.drift_state_clamp), self.drift_norm_clamp)
        return final_enc, next_l_state, commitment_cost, final_drift


class HierarchosCore(nn.Module):
    """
    Full parity version of HierarchosCore - direct port from hierarchos.py.
    """
    
    def reset_memory(self):
        """Resets the short-term 'fast' associative memory."""
        self.ltm.reset_working_memory()

    def refresh_runtime_config(self):
        detach_freq = _normalize_detach_frequency(self.config)
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

    def _apply_memory_gate_warmup(self, gate: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return gate
        warmup_steps = float(getattr(self.config, 'memory_gate_warmup_steps', 0) or 0)
        warmup_floor = float(getattr(self.config, 'memory_gate_warmup_floor', 0.0) or 0.0)
        if warmup_steps <= 0.0 or warmup_floor <= 0.0:
            return gate
        warmup_floor = min(max(warmup_floor, 0.0), 0.95)
        step = self.memory_gate_warmup_step.to(device=gate.device, dtype=torch.float32)
        progress = torch.clamp(step / warmup_steps, min=0.0, max=1.0)
        floor = gate.new_tensor(warmup_floor) * (1.0 - progress.to(dtype=gate.dtype))
        return floor + (1.0 - floor) * gate
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        _validate_architecture_config(self.config)
        
        # Tokenizer-dependent
        if not torch.cuda.is_available():
            torch.set_flush_denormal(True)
            
        self.tok_emb = nn.Embedding(config.vocab_size, config.context_dim)
        
        # V8 DeepEmbed (4x scale) - default to True
        self.use_deepembed = getattr(config, 'use_deepembed', True)
        if self.use_deepembed:
            self.h_deepemb = nn.Embedding(config.vocab_size, config.h_hidden * 4)
            self.l_deepemb = nn.Embedding(config.vocab_size, config.l_hidden * 4)
            nn.init.ones_(self.h_deepemb.weight)
            nn.init.ones_(self.l_deepemb.weight)
        
        # V8 ROSA Embed - default to True
        self.memory_token_routers = bool(getattr(config, 'memory_token_routers', True))
        self.register_buffer("memory_gate_warmup_step", torch.zeros((), dtype=torch.float32), persistent=False)
        self.use_rosa = getattr(config, 'use_rosa', True)
        if self.use_rosa:
            self.rosa_emb = nn.Embedding(config.vocab_size + 1, config.context_dim)
            nn.init.zeros_(self.rosa_emb.weight)
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
            reference_chunk_len=getattr(config, 'reference_chunk_len', getattr(config, 'training_chunk_size', 128)),
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
        self.config.rwkv_channel_mix_key_clamp = rwkv_channel_mix_key_clamp
        self.config.rwkv_channel_mix_deepembed_clamp = rwkv_channel_mix_deepembed_clamp
        self.h_rnn = RWKVCell(
            config.h_hidden,
            head_size=rwkv_head_size,
            layer_id=0,
            n_layer=getattr(config, 'rwkv_n_layer_hint', 2),
            channel_mix_key_clamp=rwkv_channel_mix_key_clamp,
            channel_mix_deepembed_clamp=rwkv_channel_mix_deepembed_clamp,
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
            head_size=rwkv_head_size,
            layer_id=0,
            n_layer=getattr(config, 'rwkv_n_layer_hint', 2),
            channel_mix_key_clamp=rwkv_channel_mix_key_clamp,
            channel_mix_deepembed_clamp=rwkv_channel_mix_deepembed_clamp,
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
                self.worker_loop_module.compile_static_worker_loop = True if static_loop is None else bool(static_loop)

                if bool(getattr(self.config, 'compile_h_rnn', True)):
                    self.h_rnn.compile_forward(**compile_kwargs)

                worker_compile_kwargs, _, _ = _resolve_compile_kwargs(
                    self.config,
                    device_type,
                    fullgraph=compile_fullgraph_worker,
                )
                self.worker_loop_module = torch.compile(
                    self.worker_loop_module,
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
            self.config.compile = False

    def forward(self, input_ids, attention_mask=None, labels=None, 
                h_state=None, l_state=None, 
                prev_context=None, target_context=None,
                drift_state=None, ltm_memory_state=None,
                global_pos_offset=0, min_timestamp=0.0, source_filter=None, **kwargs):
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
        compute_ltm_value_alignment = bool(kwargs.pop("compute_ltm_value_alignment", False))
        cached_rosa_ids = kwargs.pop("rosa_ids", None)
        loss_weights = kwargs.pop("loss_weights", None)
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
        if ltm_memory_state is None:
            isolate_batch_ltm = getattr(self.config, 'isolate_batch_ltm', True)
            isolate_runtime_ltm = isolate_batch_ltm and (self.training or B > 1)
            if isolate_runtime_ltm:
                curr_fast_vals = self.ltm.fast_vals.unsqueeze(0).expand(B, -1, -1).clone()
                curr_mom_vals = self.ltm._mom_vals.unsqueeze(0).expand(B, -1, -1).clone()
                memory_timestamps = self.ltm.timestamps.unsqueeze(0).expand(B, -1).clone()
                memory_sources = self.ltm.sources.unsqueeze(0).expand(B, -1).clone()
            else:
                curr_fast_vals = self.ltm.fast_vals
                curr_mom_vals = self.ltm._mom_vals
                memory_timestamps = self.ltm.timestamps
                memory_sources = self.ltm.sources
            past_tokens = None
        else:
            if len(ltm_memory_state) >= 6:
                curr_fast_vals, curr_mom_vals, past_tokens, rosa_states, memory_timestamps, memory_sources = ltm_memory_state[:6]
            elif len(ltm_memory_state) >= 4:
                curr_fast_vals, curr_mom_vals, past_tokens, rosa_states = ltm_memory_state[:4]
            elif len(ltm_memory_state) >= 3:
                curr_fast_vals, curr_mom_vals, past_tokens = ltm_memory_state[:3]
            else:
                curr_fast_vals, curr_mom_vals = ltm_memory_state
                past_tokens = None
            if memory_timestamps is None:
                if curr_fast_vals.dim() == 3:
                    memory_timestamps = self.ltm.timestamps.unsqueeze(0).expand(curr_fast_vals.shape[0], -1).clone()
                    memory_sources = self.ltm.sources.unsqueeze(0).expand(curr_fast_vals.shape[0], -1).clone()
                else:
                    memory_timestamps = self.ltm.timestamps
                    memory_sources = self.ltm.sources

        # V8 ROSA Precomputation (only when enabled). Launch this before the
        # embedding lookup so CPU suffix-automaton work can overlap CUDA kernels.
        new_rosa_states = None
        rosa_finalize = None
        if self.use_rosa:
            rosa_max_ctx = getattr(self.config, 'rosa_max_context', 512)

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

        if self.use_rosa:
            if cached_rosa_ids is None:
                # Finalize: wait for CPU work, async H2D transfer
                rosa_batch_tensor, rosa_past_tokens, new_rosa_states = rosa_finalize()
                # Store complete past_tokens for cross-chunk/chat continuity.
                # Detach and move to CPU immediately to prevent GPU memory leak across chunks
                new_past_tokens = rosa_past_tokens.detach().cpu()
            else:
                rosa_batch_tensor = cached_rosa_ids
                input_ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.long)
                if torch.is_tensor(past_tokens):
                    past_tokens_cpu = past_tokens.detach().to(device="cpu", dtype=torch.long)
                    if past_tokens_cpu.dim() == 1:
                        past_tokens_cpu = past_tokens_cpu.unsqueeze(0)
                    if past_tokens_cpu.shape[0] == 1 and input_ids_cpu.shape[0] > 1:
                        past_tokens_cpu = past_tokens_cpu.expand(input_ids_cpu.shape[0], -1)
                    new_past_tokens = torch.cat([past_tokens_cpu, input_ids_cpu], dim=1)
                else:
                    new_past_tokens = input_ids_cpu
                new_rosa_states = rosa_states

            rosa_embs = self.rosa_emb(rosa_batch_tensor)
            # Per-token router controls exact-memory injection without branching.
            if self.memory_token_routers and hasattr(self, "rosa_router"):
                rosa_gate_logits = self.rosa_gate_logit + self.rosa_router(x)
            else:
                rosa_gate_logits = self.rosa_gate_logit
            rosa_gate = torch.sigmoid(_finite_clamp(rosa_gate_logits, 50.0))
            rosa_gate = self._apply_memory_gate_warmup(rosa_gate)
            x = x + rosa_gate * rosa_embs  # Gated Neurosymbolic Inner Monologue Mix
        else:
            new_past_tokens = None


        # ==================================================================
        # 1. STATE INITIALIZATION (With Context Recovery)
        # ==================================================================
        if h_state is None:
            h_state = self.h_rnn.initial_state(B, device=device)
            prev_context = torch.zeros(B, self.config.context_dim, device=device)
            target_context = torch.zeros(B, self.config.context_dim, device=device)
        else:
            h_state = h_state.to(device)
            if prev_context is None:
                prev_context = self.h_to_context(self.h_rnn.state_hidden(h_state))
            else:
                prev_context = prev_context.to(device)
            if target_context is None:
                target_context = self.h_to_context(self.h_rnn.state_hidden(h_state))
            else:
                target_context = target_context.to(device)
        h_state = _finite_clamp(h_state, recurrent_state_clamp)
        prev_context = _finite_clamp(prev_context, context_state_clamp)
        target_context = _finite_clamp(target_context, context_state_clamp)

        if l_state is None:
            l_state = self.l_rnn.initial_state(B, device=device)
        else:
            l_state = l_state.to(device)
        l_state = _finite_clamp(l_state, recurrent_state_clamp)

        # (ltm_memory_state already unpacked above)
        if ltm_memory_state is not None:
            # Ensure they are on the correct device
            curr_fast_vals = curr_fast_vals.to(device)
            curr_mom_vals = curr_mom_vals.to(device)
            memory_timestamps = memory_timestamps.to(device)
            memory_sources = memory_sources.to(device)

        drift_seed = None
        if drift_state is not None:
            drift_seed = drift_state.to(device)
            if drift_seed.dim() == 1:
                drift_seed = drift_seed.unsqueeze(0)
            if drift_seed.shape[0] == 1 and B > 1:
                drift_seed = drift_seed.expand(B, -1)
            if drift_seed.shape != (B, self.config.context_dim):
                drift_seed = None
            else:
                drift_seed = _finite_clamp(drift_seed, drift_state_clamp)

        final_embs = []
        ponder_costs = []
        ponder_weights = []
        commitment_costs = []
        commitment_weights = []
        ltm_value_alignment_costs = []
        ltm_value_alignment_weights = []
        all_topk_vals = []
        all_topk_idx = []
        aux_attention_mask = attention_mask.to(device=device, dtype=torch.float32) if attention_mask is not None else None

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

        # ==================================================================
        # 2. MAIN TIME LOOP
        # ==================================================================
        for t in range(T):
            token_x_idx = input_ids[:, t]
            token_x = x[:, t]
            abs_t = global_pos_offset + t
            
            h_deepemb_vec = self.h_deepemb(token_x_idx) if self.use_deepembed else None
            l_deepemb_vec = self.l_deepemb(token_x_idx) if self.use_deepembed else None
            
            # --- LTM Retrieval ---
            p = self.persistent.unsqueeze(0).expand(B, -1)
            q_in = torch.cat([token_x, prev_context], dim=-1)
            q = _finite_clamp(self.qproj(q_in), 12.0)
            
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(
                q, self.config.ltm_topk, min_timestamp, source_filter, fast_vals=curr_fast_vals,
                timestamps=memory_timestamps, sources=memory_sources
            )
            
            if return_topk_values or return_raw_topk_values:
                all_topk_vals.append(topk_vals)
            if return_topk_indices:
                all_topk_idx.append(topk_idx)
            
            # Positional encoding
            args = topk_ts.unsqueeze(-1) * self.time_freqs.unsqueeze(0).unsqueeze(0)
            pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.config.ltm_val_dim % 2 == 1:
                pe = torch.cat([pe, pe.new_zeros(*pe.shape[:-1], 1)], dim=-1)
            topk_vals = topk_vals + pe
            
            if self.memory_token_routers and hasattr(self, "ltm_router"):
                gate_input = self.ltm_gate_logit + self.ltm_router(token_x)
            else:
                gate_input = self.ltm_gate_logit
            gate = torch.sigmoid(_finite_clamp(gate_input, 50.0))
            gate = self._apply_memory_gate_warmup(gate)
            if gate.dim() == 2:
                gate = gate.unsqueeze(1)
            gated_vals = topk_vals * gate
            mac_in = torch.cat([token_x, p, gated_vals.view(B, -1)], dim=-1)
            
            enc = F.gelu(self.in_proj(mac_in))
            enc = _finite_clamp(enc, 30.0)

            # ==================================================================
            # 3. HIERARCHICAL MANAGER (Continuous Watch, Strided Plan)
            # ==================================================================
            l_feedback = self.l_feedback_proj(self.l_rnn.state_hidden(l_state).to(device))
            enc_with_feedback = _finite_clamp(enc + l_feedback, activation_clamp)

            detach_freq = getattr(self.config, 'detach_every_n_steps', 32)
            if self.training and detach_freq is not None and t > 0 and t % detach_freq == 0:
                h_state = h_state.detach()
            
            h_out_real, h_state = self.h_rnn(enc_with_feedback, h_state, timestep=None, deepemb_vec=h_deepemb_vec)
            h_out_real = _finite_clamp(h_out_real, activation_clamp)
            h_state = _finite_clamp(h_state, recurrent_state_clamp)
            
            if getattr(self.config, 'debug_numerics', False) and (torch.isnan(h_out_real).any() or torch.isinf(h_out_real).any()):
                print(f"WARNING: NaN/Inf detected in h_out_real at step {t}")
            
            step_ponder_cost = torch.zeros(B, device=device, dtype=enc.dtype)
            
            # PLANNING STEP (Strided with ACT)
            if abs_t % stride == 0:
                prev_context = _finite_clamp(target_context, context_state_clamp)

                # Pondering on Shadow State
                h_step_outputs = [h_out_real]
                halt_logit = _finite_clamp(self.h_halt_proj(h_out_real).squeeze(-1), halt_logit_clamp)
                h_halt_probs = [torch.sigmoid(halt_logit).clamp(1e-6, 1.0 - 1e-6)]
                
                shadow_h_state = _finite_clamp(h_state.clone(), recurrent_state_clamp)
                current_enc_h = enc_with_feedback

                inference_logit_parity = bool(
                    getattr(self.config, 'inference_logit_parity', False)
                    or getattr(self.config, 'full_sample_bptt', False)
                )
                for step_idx in range(self.config.max_h_steps - 1):
                    if (
                        not self.training
                        and not inference_logit_parity
                        and h_halt_probs[-1].mean() > getattr(self.config, 'h_halt_thresh', 0.9)
                    ):
                        break
                    h_out_ponder, shadow_h_state = self.h_rnn(current_enc_h, shadow_h_state, timestep=None, deepemb_vec=h_deepemb_vec)
                    h_out_ponder = _finite_clamp(h_out_ponder, activation_clamp)
                    shadow_h_state = _finite_clamp(shadow_h_state, recurrent_state_clamp)
                    halt_logit = _finite_clamp(self.h_halt_proj(h_out_ponder).squeeze(-1), halt_logit_clamp)
                    h_step_outputs.append(h_out_ponder)
                    h_halt_probs.append(torch.sigmoid(halt_logit).clamp(1e-6, 1.0 - 1e-6))

                # BUG #4 FIX: Force ACT weighting to float32 for numerical stability.
                # BFloat16's limited precision (~7 bits) causes underflow in cumprod chains,
                # leading to NaN weights. Mirrors the autocast(enabled=False) pattern in rwkv_cell.py.
                h_stack = torch.stack(h_step_outputs, dim=0).float()
                halt_stack = torch.stack(h_halt_probs, dim=0).float()
                halt_stack = torch.nan_to_num(halt_stack, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
                halt_stack = halt_stack.clamp(1e-6, 1.0 - 1e-6)
                remain = 1.0 - halt_stack
                remain_shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
                cum_remain = torch.cumprod(remain_shifted, dim=0)
                
                weights = halt_stack * cum_remain
                remainder = cum_remain[-1] * (1.0 - halt_stack[-1])
                total = weights.sum(dim=0) + remainder + 1e-8
                weights = weights / total.unsqueeze(0)
                remainder = remainder / total
                final_h_out = (weights.unsqueeze(-1) * h_stack).sum(dim=0) + remainder.unsqueeze(-1) * h_stack[-1]
                if torch.isnan(final_h_out).any() or torch.isinf(final_h_out).any():
                    print(f"WARNING: Non-finite manager output at token step {abs_t}; clamping for stability.")
                    final_h_out = torch.nan_to_num(final_h_out, nan=0.0, posinf=10.0, neginf=-10.0)
                final_h_out = _finite_clamp(final_h_out, activation_clamp).to(enc.dtype)  # Cast back to working precision
                
                target_context = self.h_to_context(final_h_out)
                target_context = _finite_clamp(target_context, context_state_clamp)
                
                step_ponder_cost = cum_remain.sum(dim=0).to(enc.dtype)
                ponder_costs.append(step_ponder_cost)
                if aux_attention_mask is not None:
                    ponder_weights.append(aux_attention_mask[:, t])
                else:
                    ponder_weights.append(torch.ones(B, device=device, dtype=torch.float32))
            
            # LERP (Interpolation)
            step_in_stride = abs_t % stride
            alpha = step_in_stride / float(stride)
            sliding_context = (prev_context.float() + alpha * (target_context.float() - prev_context.float())).to(prev_context.dtype)
            sliding_context = _finite_clamp(sliding_context, context_state_clamp)

            # ==================================================================
            # 4. WORKER STEP
            # ==================================================================
            if t == 0 and drift_seed is not None:
                initial_drift = _l2_norm_clamp(_finite_clamp(drift_seed, drift_state_clamp), drift_norm_clamp)
            elif self.context_drift_proj is not None:
                prev_worker_h = self.l_rnn.state_hidden(l_state).to(device)
                initial_drift = torch.tanh(self.context_drift_proj(prev_worker_h))
                initial_drift = _l2_norm_clamp(_finite_clamp(initial_drift, drift_state_clamp), drift_norm_clamp)
            else:
                initial_drift = torch.zeros(B, self.config.context_dim, device=device)

            if self.training and detach_freq is not None and t > 0 and t % detach_freq == 0:
                l_state = l_state.detach()

            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                enc, l_state, cc, final_drift = checkpoint(
                    self.worker_loop_module, enc, sliding_context, l_state, initial_drift, None, l_deepemb_vec,
                    use_reentrant=False
                )
            else:
                enc, l_state, cc, final_drift = self.worker_loop_module(
                    enc, sliding_context, l_state, initial_drift, timestep=None, l_deepemb_vec=l_deepemb_vec
                )
            enc = _finite_clamp(enc, activation_clamp)
            l_state = _finite_clamp(l_state, recurrent_state_clamp)
            final_drift = _l2_norm_clamp(_finite_clamp(final_drift, drift_state_clamp), drift_norm_clamp)

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
                else:
                    ltm_value_alignment_weights.append(
                        torch.ones(B, device=device, dtype=torch.float32)
                    )
            
            final_embs.append(enc)
            commitment_costs.append(cc)
            if aux_attention_mask is not None:
                commitment_weights.append(aux_attention_mask[:, t])
            else:
                commitment_weights.append(torch.ones(B, device=device, dtype=torch.float32))

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
                
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    topk_idx, None, val_expanded,
                    current_lr=getattr(self.config, 'ltm_lr', 0.001),  # COH #1: Use config LR, not hardcoded 0.01
                    timestamp=0.0,
                    tokens_covered=1,
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals,
                    timestamps=memory_timestamps,
                    sources=memory_sources,
                    inplace=True
                )

        # ==================================================================
        # 5. FINAL OUTPUTS
        # ==================================================================
        final = _finite_clamp(self.out_norm(torch.stack(final_embs, dim=1)), activation_clamp)
        logits = None

        loss = None
        ponder_cost_out = None
        commitment_cost_out = None
        ltm_value_alignment_cost_out = None

        if labels is not None and not return_logits:
            loss = self._compute_cuda_chunked_lm_loss(
                final,
                labels,
                getattr(self.config, 'z_loss_weight', 1e-4),
                loss_weights=loss_weights,
            )
        else:
            logits = self.lm_head(final)
            
            if getattr(self.config, 'debug_numerics', False) and (torch.isnan(logits).any() or torch.isinf(logits).any()):
                print("WARNING: NaN/Inf detected in logits. Replacing with zeros and clamping...")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            
            logits = _finite_clamp(logits, 30.0)

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
                if not valid_mask.any():
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
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
                    
                    # Base CE loss, optionally weighted toward assistant response tokens.
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
                weight_tensor = torch.stack(weights, dim=0).float()
                denom = weight_tensor.sum()
                if denom <= 0:
                    return torch.zeros((), device=device, dtype=cost_tensor.dtype)
                return (cost_tensor * weight_tensor).sum() / denom

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
            "topk_vals": torch.stack(all_topk_vals, dim=1) if (return_topk_values and all_topk_vals) else None, 
            "raw_topk_vals": all_topk_vals if return_raw_topk_values else None,
            "topk_idx": torch.stack(all_topk_idx, dim=1) if all_topk_idx else None,
            "h_state": h_state,
            "l_state": l_state,
            "prev_context": prev_context,
            "target_context": target_context,
            "drift_state": final_drift,
            "ltm_memory_state": (curr_fast_vals, curr_mom_vals, new_past_tokens, new_rosa_states, memory_timestamps, memory_sources),
        }

    def _compute_cuda_chunked_lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor,
                                      z_loss_weight: float = 1e-4,
                                      loss_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            return hidden.sum() * 0.0
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

        for start in range(0, valid_count, chunk_rows):
            end = min(start + chunk_rows, valid_count)
            chunk_hidden = valid_hidden[start:end]
            chunk_labels = valid_labels[start:end]
            chunk_weights = valid_weights[start:end] if valid_weights is not None else None
            chunk_logits = torch.clamp(self.lm_head(chunk_hidden), min=-30.0, max=30.0).float()

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

        loss = total_ce / denom
        if z_loss_weight > 0:
            loss = loss + (total_z / denom) * z_loss_weight
        return loss

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def update_memory(self, topk_idx, grads, timestamp, lr=1e-3):
        """Updates the LTM memory using gradients (Titans style)."""
        self.ltm.inner_update(topk_idx, grads, current_lr=lr, timestamp=timestamp, inplace=True)

    def update_memory_hebbian(self, topk_idx, vals, timestamp, lr=1e-3, tokens_covered=1):
        """Updates the LTM memory using Hebbian rule (Fallback for Inference)."""
        self.ltm.update_memory_hebbian(topk_idx, None, vals, current_lr=lr, 
                                       timestamp=timestamp, tokens_covered=tokens_covered, inplace=True)
