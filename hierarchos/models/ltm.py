import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(numeric)


def _bounded_float(
    name: str,
    value,
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")
    return numeric


class LTMModule(nn.Module):
    """
    Titans-style Neural Memory Module with Dual-Store Architecture.
    
    MERGED VERSION (V3.2 - Time-Invariant Decay): 
    1. Retains 'Fast State' (Titans) for test-time updates.
    2. Scales forgetting rate based on token count to fix training/inference mismatch.
    """
    # SOURCE_ID definitions
    SRC_UNKNOWN = 0
    SRC_USER_INTERACTION = 1
    SRC_TRAINING_DATA = 2
    SRC_CORRECTION = 3 

    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4, forget_rate=0.01, reference_chunk_len=128, score_grad_scale=1.0, cpu_gather_retrieval=True, cpu_sparse_update=True):
        super().__init__()

        n_slots = _positive_int("n_slots", n_slots)
        key_dim = _positive_int("key_dim", key_dim)
        val_dim = _positive_int("val_dim", val_dim)
        reference_chunk_len = _positive_int(
            "reference_chunk_len",
            reference_chunk_len,
        )
        lr = _bounded_float("lr", lr, minimum=0.0)
        momentum = _bounded_float(
            "momentum",
            momentum,
            minimum=0.0,
            maximum=1.0,
        )
        wd = _bounded_float("wd", wd, minimum=0.0)
        forget_rate = _bounded_float(
            "forget_rate",
            forget_rate,
            minimum=0.0,
            maximum=1.0,
        )
        score_grad_scale = _bounded_float(
            "score_grad_scale",
            score_grad_scale,
            minimum=0.0,
        )
        
        # --- Slow Weights (Long-Term Consolidation) ---
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        
        # Vals represent the consolidated content. 
        vals_init = torch.empty(n_slots, val_dim)
        nn.init.orthogonal_(vals_init)
        self.vals = nn.Parameter(vals_init * 0.02)
        
        # --- Fast State (Associative Working Memory - Titans Feature) ---
        self.register_buffer("fast_vals", torch.zeros(n_slots, val_dim))
        
        # --- Metadata & Optimizer Stats ---
        self.register_buffer("_mom_vals", torch.zeros_like(self.vals.data))
        self.lr, self.momentum, self.weight_decay = lr, momentum, wd
        
        # Base forget rate per REFERENCE chunk size
        self.forget_rate = forget_rate 
        self.reference_chunk_len = reference_chunk_len
        self.score_grad_scale = float(score_grad_scale)
        self.cpu_gather_retrieval = bool(cpu_gather_retrieval)
        self.cpu_sparse_update = bool(cpu_sparse_update)

        # Buffers for tracking history context
        self.register_buffer("timestamps", torch.zeros(n_slots, dtype=torch.float32))
        self.register_buffer("sources", torch.full((n_slots,), self.SRC_UNKNOWN, dtype=torch.long))
        # Wall-clock metadata is intentionally runtime-only. ``timestamps`` is the
        # model-facing token-position clock used by the sinusoidal read feature;
        # wall-clock seconds are used only by user-facing recency filters.
        self.register_buffer(
            "wallclock_timestamps",
            torch.zeros(n_slots, dtype=torch.float64),
            persistent=False,
        )

        # Buffer for neg_inf to avoid creation in hot loop
        self.register_buffer("neg_inf", torch.tensor(-float('inf')), persistent=False)

        # Pre-allocate buffers for update calculations
        self.register_buffer("update_counts", torch.zeros(n_slots, dtype=torch.float32), persistent=False)
        self.register_buffer("update_slots", torch.zeros(n_slots, val_dim, dtype=torch.float32), persistent=False)

        # <<< NEW: Online Learning Accumulator >>>
        self.register_buffer("ltm_deltas", torch.zeros(n_slots, val_dim), persistent=False)
        self.accumulate_deltas = False
        # Internal architecture switch: CUDA stays on the existing gather/scatter path.
        # CPU can opt into the same sparse math to avoid large one-hot tensors.
        self._cuda_friendly_math = True

    def reset_working_memory(self):
        """Zeros out the Fast State (Working Memory), momentum, and slot metadata."""
        self.fast_vals.zero_()
        self._mom_vals.zero_()
        self.timestamps.zero_()
        self.sources.fill_(self.SRC_UNKNOWN)
        self.wallclock_timestamps.zero_()

    def get_effective_memory(self):
        """Returns the combined memory (Slow + Fast)."""
        return self.vals + self.fast_vals

    def _use_cuda_math(self, tensor: torch.Tensor) -> bool:
        return bool(self._cuda_friendly_math and tensor.device.type == "cuda")

    def _use_gather_retrieval(self, tensor: torch.Tensor) -> bool:
        return self._use_cuda_math(tensor) or bool(
            self.cpu_gather_retrieval and tensor.device.type == "cpu"
        )

    def _use_sparse_update(self, tensor: torch.Tensor) -> bool:
        return self._use_cuda_math(tensor) or bool(
            self.cpu_sparse_update and tensor.device.type == "cpu"
        )

    def _capture_fast_state_for_delta(self, fast_vals: torch.Tensor) -> Optional[torch.Tensor]:
        if not getattr(self, "accumulate_deltas", False):
            return None
        return fast_vals.detach().float().clone()

    def _record_fast_state_delta(
        self,
        before: Optional[torch.Tensor],
        after: torch.Tensor,
    ) -> None:
        """Accumulate the exact effective fast-state change, including decay."""
        if before is None:
            return
        delta = after.detach().float() - before.to(device=after.device, dtype=torch.float32)
        if delta.dim() == 3:
            # Persistent overlays are model-global. Batched online writes are
            # unusual, but summing independent row deltas preserves every update.
            delta = delta.sum(dim=0)
        self.ltm_deltas.add_(
            delta.to(device=self.ltm_deltas.device, dtype=self.ltm_deltas.dtype)
        )

    def _stamp_wallclock_metadata(
        self,
        topk_idx: torch.LongTensor,
        wallclock_timestamp: Optional[float],
        wallclock_timestamps: Optional[torch.Tensor],
    ) -> None:
        if wallclock_timestamp is None:
            return
        target = (
            wallclock_timestamps
            if wallclock_timestamps is not None
            else self.wallclock_timestamps
        )
        if not torch.is_tensor(target):
            return
        valid = topk_idx >= 0
        if not bool(valid.any().item()):
            return
        indices = topk_idx.to(device=target.device, dtype=torch.long)
        valid = valid.to(device=target.device)
        with torch.no_grad():
            if target.dim() == 1:
                target.index_fill_(0, torch.unique(indices[valid]), float(wallclock_timestamp))
                return
            if target.dim() != 2:
                raise ValueError(
                    "LTM wall-clock metadata must have shape [slots] or [batch, slots]"
                )
            if indices.shape[0] != target.shape[0]:
                raise ValueError(
                    f"LTM wall-clock batch {target.shape[0]} does not match "
                    f"top-k batch {indices.shape[0]}"
                )
            for batch_idx in range(target.shape[0]):
                row_valid = valid[batch_idx]
                if bool(row_valid.any().item()):
                    target[batch_idx].index_fill_(
                        0,
                        torch.unique(indices[batch_idx][row_valid]),
                        float(wallclock_timestamp),
                    )

    @staticmethod
    def _expand_slot_tensor_for_index(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Broadcast a (..., slots, value_dim) tensor to index leading dims."""
        if tensor.dim() == 2:
            return tensor
        index_leading = tuple(index.shape[:-1])
        tensor_leading = tuple(tensor.shape[:-2])
        if len(tensor_leading) > len(index_leading):
            raise ValueError("LTM memory has more leading dimensions than top-k indices")
        view_shape = tensor_leading + (1,) * (len(index_leading) - len(tensor_leading)) + tuple(tensor.shape[-2:])
        expand_shape = index_leading + tuple(tensor.shape[-2:])
        return tensor.reshape(view_shape).expand(expand_shape)

    @staticmethod
    def _expand_slot_metadata_for_index(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Broadcast a (..., slots) metadata tensor to index leading dims."""
        if tensor.dim() == 1:
            return tensor
        index_leading = tuple(index.shape[:-1])
        tensor_leading = tuple(tensor.shape[:-1])
        if len(tensor_leading) > len(index_leading):
            raise ValueError("LTM metadata has more leading dimensions than top-k indices")
        view_shape = tensor_leading + (1,) * (len(index_leading) - len(tensor_leading)) + (tensor.shape[-1],)
        expand_shape = index_leading + (tensor.shape[-1],)
        return tensor.reshape(view_shape).expand(expand_shape)

    def _gather_slot_values(
        self,
        idx_clamped: torch.LongTensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Gather selected slots without materializing a broadcast full store."""
        memory = memory.float()
        if memory.dim() == 2:
            flat_idx = idx_clamped.reshape(-1)
            gathered_vals = memory.index_select(0, flat_idx)
            gathered_vals = gathered_vals.view(
                *idx_clamped.shape,
                memory.shape[-1],
            )
        else:
            memory_view = self._expand_slot_tensor_for_index(memory, idx_clamped)
            gather_idx = idx_clamped.unsqueeze(-1).expand(*idx_clamped.shape, memory_view.shape[-1])
            gathered_vals = torch.gather(memory_view, dim=-2, index=gather_idx)
        return gathered_vals.contiguous()

    def _gather_topk_cuda(self, idx_clamped: torch.LongTensor, memory: torch.Tensor,
                          current_timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gathered_vals = self._gather_slot_values(idx_clamped, memory)
        current_timestamps = current_timestamps.to(device=idx_clamped.device, dtype=torch.float32)
        if current_timestamps.dim() == 1:
            ts_retrieved = current_timestamps.index_select(0, idx_clamped.reshape(-1))
            ts_retrieved = ts_retrieved.view_as(idx_clamped)
        else:
            ts_view = self._expand_slot_metadata_for_index(current_timestamps, idx_clamped)
            ts_retrieved = torch.gather(ts_view, dim=-1, index=idx_clamped)

        return gathered_vals, ts_retrieved

    def _inject_score_gradients(self, gathered_vals: torch.Tensor, selected_sim: torch.Tensor) -> torch.Tensor:
        """Give the address path gradients without changing retrieved values."""
        scale = getattr(self, "score_grad_scale", 1.0)
        if selected_sim is None or scale == 0.0 or not torch.is_grad_enabled() or not selected_sim.requires_grad:
            return gathered_vals
        score_signal = (selected_sim - selected_sim.detach()).unsqueeze(-1)
        return gathered_vals + score_signal.to(dtype=gathered_vals.dtype) * gathered_vals.detach() * scale

    def _inner_update_flat_cuda(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor,
                                current_lr: float, timestamp: float, source: int,
                                tokens_covered: int, curr_fast: torch.Tensor,
                                curr_mom: torch.Tensor, timestamps: Optional[torch.Tensor],
                                sources: Optional[torch.Tensor],
                                inplace: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if grads_tensor is None:
            return curr_fast, curr_mom

        fast_before = self._capture_fast_state_for_delta(curr_fast)
        device = curr_fast.device
        topk_idx = topk_idx.to(device=device, dtype=torch.long)
        grads_tensor = grads_tensor.to(device=device).float()
        valid_mask = topk_idx >= 0
        has_valid = valid_mask.any()

        curr_fast = curr_fast.float() if inplace else curr_fast.float().clone()
        curr_mom = curr_mom.float() if inplace else curr_mom.float().clone()

        idx_flat = topk_idx[valid_mask].reshape(-1)
        grads_flat = grads_tensor[valid_mask].reshape(-1, self.vals.size(1)).contiguous()

        num_slots = self.vals.size(0)
        unique_idx, inverse, counts = torch.unique(
            idx_flat,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        slot_grads = grads_flat.new_zeros((unique_idx.numel(), self.vals.size(1)))
        slot_grads.index_add_(0, inverse, grads_flat)
        slot_grads = slot_grads / (counts.to(dtype=torch.float32).unsqueeze(-1) + 1e-8)

        momentum_factor = torch.where(
            has_valid,
            curr_mom.new_tensor(float(self.momentum)),
            curr_mom.new_tensor(1.0),
        )
        curr_mom.mul_(momentum_factor)
        touched_mom = curr_mom.index_select(0, unique_idx).add(slot_grads)
        touched_mom.clamp_(min=-50.0, max=50.0)
        curr_mom.index_copy_(0, unique_idx, touched_mom)

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))
        retention_rate = torch.where(
            has_valid,
            curr_fast.new_tensor(float(retention_rate)),
            curr_fast.new_tensor(1.0),
        )

        touched_fast = curr_fast.index_select(0, unique_idx)
        update_step = (touched_mom + self.weight_decay * touched_fast).mul(-current_lr)
        curr_fast.mul_(retention_rate)
        touched_fast = touched_fast.mul(retention_rate).add(update_step)
        touched_fast.clamp_(min=-50.0, max=50.0)
        curr_fast.index_copy_(0, unique_idx, touched_fast)

        target_timestamps = timestamps if timestamps is not None else self.timestamps
        target_sources = sources if sources is not None else self.sources
        with torch.no_grad():
            target_timestamps = target_timestamps.to(device=device)
            target_sources = target_sources.to(device=device)
            target_timestamps.index_fill_(0, unique_idx, float(timestamp))
            target_sources.index_fill_(0, unique_idx, int(source))

        self._record_fast_state_delta(fast_before, curr_fast)
        return curr_fast, curr_mom

    def _inner_update_batched_cuda(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor,
                                   current_lr: float, timestamp: float, source: int,
                                   tokens_covered: int, curr_fast: torch.Tensor,
                                   curr_mom: torch.Tensor, timestamps: Optional[torch.Tensor],
                                   sources: Optional[torch.Tensor],
                                   inplace: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if grads_tensor is None:
            return curr_fast, curr_mom

        fast_before = self._capture_fast_state_for_delta(curr_fast)
        device = curr_fast.device
        topk_idx = topk_idx.to(device=device, dtype=torch.long)
        grads_tensor = grads_tensor.to(device=device).float()
        valid_mask = topk_idx >= 0

        curr_fast = curr_fast.float() if inplace else curr_fast.float().clone()
        curr_mom = curr_mom.float() if inplace else curr_mom.float().clone()

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))

        batch_size, num_slots, val_dim = curr_fast.shape
        idx_safe = topk_idx.clamp(min=0)
        flat_idx = idx_safe.reshape(batch_size, -1)
        valid_flat = valid_mask.reshape(batch_size, -1)
        batch_has_valid = valid_flat.any(dim=1)
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_slots
        linear_idx = (flat_idx + batch_offsets)[valid_flat].reshape(-1)
        grads_flat = grads_tensor.reshape(batch_size, -1, val_dim)[valid_flat].reshape(-1, val_dim).contiguous()

        unique_linear, inverse, counts = torch.unique(
            linear_idx,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        slot_grads = grads_flat.new_zeros((unique_linear.numel(), val_dim))
        slot_grads.index_add_(0, inverse, grads_flat)
        slot_grads = slot_grads / (counts.to(dtype=torch.float32).unsqueeze(-1) + 1e-8)

        batch_idx = torch.div(unique_linear, num_slots, rounding_mode='floor')
        slot_idx = unique_linear.remainder(num_slots)

        momentum_factor = torch.where(
            batch_has_valid.view(batch_size, 1, 1),
            curr_mom.new_tensor(float(self.momentum)),
            curr_mom.new_tensor(1.0),
        )
        curr_mom.mul_(momentum_factor)
        touched_mom = curr_mom[batch_idx, slot_idx].add(slot_grads)
        touched_mom.clamp_(min=-50.0, max=50.0)
        curr_mom[batch_idx, slot_idx] = touched_mom

        touched_fast = curr_fast[batch_idx, slot_idx]
        update_step = (touched_mom + self.weight_decay * touched_fast).mul(-current_lr)
        retention = torch.where(
            batch_has_valid.view(batch_size, 1, 1),
            curr_fast.new_tensor(float(retention_rate)),
            curr_fast.new_tensor(1.0),
        )
        curr_fast.mul_(retention)
        touched_fast = touched_fast.mul(float(retention_rate)).add(update_step)
        touched_fast.clamp_(min=-50.0, max=50.0)
        curr_fast[batch_idx, slot_idx] = touched_fast

        if timestamps is not None and sources is not None:
            with torch.no_grad():
                timestamps = timestamps.to(device=device)
                sources = sources.to(device=device)
                timestamps[batch_idx, slot_idx] = float(timestamp)
                sources[batch_idx, slot_idx] = int(source)

        self._record_fast_state_delta(fast_before, curr_fast)
        return curr_fast, curr_mom

    def retrieve_topk(self, queries: torch.Tensor, topk: int = 4, min_timestamp: float = 0.0,
                      source_filter: Optional[int] = None, fast_vals: Optional[torch.Tensor] = None,
                      timestamps: Optional[torch.Tensor] = None,
                      sources: Optional[torch.Tensor] = None,
                      min_wallclock_timestamp: float = 0.0,
                      wallclock_timestamps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        # CRITICAL AMP FIX: Disable autocast for the entire retrieval path.
        # Under BFloat16 AMP, torch.matmul is in the BF16-eligible list and will
        # silently recast explicit .float() operands back to BF16. The backward pass
        # of this BF16 matmul then propagates BF16 gradients into float32 parameters
        # via masked_scatter_, which crashes with:
        #   "masked_scatter_: expected self and source to have same dtypes but got BFloat16 and Float"
        # This mirrors the autocast(enabled=False) pattern used in rwkv_cell.py for WKV stability.
        _device_type = queries.device.type if queries.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=_device_type, enabled=False):
            topk = int(topk)
            if topk <= 0:
                raise ValueError(f"topk must be positive, got {topk}")

            # Cast queries to float32 to ensure all downstream ops stay in float32
            queries = queries.float()

            scale_factor = self.keys.shape[1] ** -0.5
            sim = (queries @ self.keys.t()) * scale_factor

            current_timestamps = timestamps if timestamps is not None else self.timestamps
            current_sources = sources if sources is not None else self.sources
            current_wallclock = (
                wallclock_timestamps
                if wallclock_timestamps is not None
                else self.wallclock_timestamps
            )
            current_timestamps = current_timestamps.to(device=queries.device)
            current_sources = current_sources.to(device=queries.device)
            current_wallclock = current_wallclock.to(device=queries.device)

            with torch.no_grad():
                valid_mask = torch.isfinite(sim)
                if min_timestamp > 0.0:
                    valid_mask = valid_mask & (current_timestamps >= min_timestamp)
                if source_filter is not None:
                    valid_mask = valid_mask & (current_sources == source_filter)
                if min_wallclock_timestamp > 0.0:
                    valid_mask = valid_mask & (
                        current_wallclock >= float(min_wallclock_timestamp)
                    )

            # Keep the similarity graph alive for address learning. The old
            # filtered branch performed these operations inside no_grad(), which
            # detached query/key score gradients along with the boolean mask.
            sim = torch.nan_to_num(
                sim,
                nan=-torch.inf,
                posinf=torch.finfo(sim.dtype).max,
                neginf=-torch.inf,
            )
            sim = torch.where(valid_mask, sim, self.neg_inf.to(dtype=sim.dtype))

            use_gather_retrieval = self._use_gather_retrieval(queries)
            effective_topk = min(topk, self.vals.shape[0])

            if effective_topk <= 0:
                query_shape = list(queries.shape)
                vals_shape = query_shape[:-1] + [topk, self.vals.shape[-1]]
                idx_shape = query_shape[:-1] + [topk]
                
                return (torch.zeros(vals_shape, device=queries.device, dtype=self.vals.dtype), 
                        torch.full(idx_shape, -1, device=queries.device, dtype=torch.long), 
                        torch.zeros(idx_shape, device=queries.device, dtype=torch.float32))

            sim_detached = sim.detach()
            _, idx = torch.topk(sim_detached, k=effective_topk, dim=-1)
            selected_valid = torch.gather(valid_mask, dim=-1, index=idx)
            idx_ret = torch.where(selected_valid, idx, torch.full_like(idx, -1))
            
            current_fast_vals = fast_vals if fast_vals is not None else self.fast_vals
            
            effective_memory_size = self.vals.shape[0]
            idx_clamped = idx.clamp(min=0, max=effective_memory_size-1)
            selected_sim = None
            if torch.is_grad_enabled() and getattr(self, "score_grad_scale", 1.0) != 0.0 and sim.requires_grad:
                selected_sim = torch.gather(sim, dim=-1, index=idx_clamped)
                selected_sim = torch.where(selected_valid, selected_sim, torch.zeros_like(selected_sim))

            if use_gather_retrieval:
                slow_vals, ts_retrieved = self._gather_topk_cuda(
                    idx_clamped,
                    self.vals,
                    current_timestamps,
                )
                fast_selected = self._gather_slot_values(
                    idx_clamped,
                    current_fast_vals,
                )
                weighted_vals = slow_vals + fast_selected
                weighted_vals = self._inject_score_gradients(weighted_vals, selected_sim)
                weighted_vals = weighted_vals * selected_valid.unsqueeze(-1).to(weighted_vals.dtype)
                ts_retrieved = torch.where(selected_valid, ts_retrieved, torch.zeros_like(ts_retrieved))
                if effective_topk < topk:
                    pad_size = topk - effective_topk
                    batch_shape_list = list(idx_clamped.shape[:-1])

                    idx_pad = torch.full(batch_shape_list + [pad_size], -1, device=idx.device, dtype=idx.dtype)
                    idx_ret = torch.cat([idx_ret, idx_pad], dim=-1)

                    vals_pad = torch.zeros(batch_shape_list + [pad_size, weighted_vals.shape[-1]],
                                           device=weighted_vals.device, dtype=weighted_vals.dtype)
                    vals_ret = torch.cat([weighted_vals, vals_pad], dim=-2)

                    ts_pad = torch.zeros(batch_shape_list + [pad_size],
                                         device=ts_retrieved.device, dtype=ts_retrieved.dtype)
                    ts_ret = torch.cat([ts_retrieved, ts_pad], dim=-1)

                    return vals_ret, idx_ret, ts_ret
                return weighted_vals, idx_ret, ts_retrieved
            
            num_classes = effective_memory_size
            range_tensor = torch.arange(num_classes, device=idx_clamped.device)
            view_shape = [1] * idx.ndim + [-1]
            range_tensor = range_tensor.view(*view_shape)
            
            one_hot = (idx_clamped.unsqueeze(-1) == range_tensor).float()

            gathered_vals = (
                torch.matmul(one_hot, self.vals.float())
                + torch.matmul(one_hot, current_fast_vals.float())
            )
            
            # --- FINAL LTM FIX: Clean Concatenation Signal ---
            # With normalization handling the bleeding, we can now use the 
            # raw memories as the model expects. Softmax is no longer needed.
            weighted_vals = self._inject_score_gradients(gathered_vals.contiguous(), selected_sim)
            
            ts_for_gather = current_timestamps.to(device=one_hot.device, dtype=one_hot.dtype)
            while ts_for_gather.ndim < one_hot.ndim:
                ts_for_gather = ts_for_gather.unsqueeze(-2)
            ts_retrieved = (one_hot * ts_for_gather).sum(dim=-1)
            weighted_vals = weighted_vals * selected_valid.unsqueeze(-1).to(weighted_vals.dtype)
            ts_retrieved = torch.where(selected_valid, ts_retrieved, torch.zeros_like(ts_retrieved))
            
            if effective_topk < topk:
                pad_size = topk - effective_topk
                batch_shape_list = list(idx_clamped.shape[:-1])
                
                idx_pad = torch.full(batch_shape_list + [pad_size], -1, device=idx.device, dtype=idx.dtype)
                idx_ret = torch.cat([idx_ret, idx_pad], dim=-1)
                
                vals_pad = torch.zeros(batch_shape_list + [pad_size, weighted_vals.shape[-1]], 
                                       device=weighted_vals.device, dtype=weighted_vals.dtype)
                vals_ret = torch.cat([weighted_vals, vals_pad], dim=-2)
                
                ts_pad = torch.zeros(batch_shape_list + [pad_size], 
                                   device=ts_retrieved.device, dtype=ts_retrieved.dtype)
                ts_ret = torch.cat([ts_retrieved, ts_pad], dim=-1)
                
                return vals_ret, idx_ret, ts_ret
            else:
                return weighted_vals, idx_ret, ts_retrieved

    def inner_update(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor, current_lr: float, timestamp: float, 
                    source: int = SRC_USER_INTERACTION, tokens_covered: int = None,
                    fast_vals: Optional[torch.Tensor] = None, mom_vals: Optional[torch.Tensor] = None,
                    timestamps: Optional[torch.Tensor] = None, sources: Optional[torch.Tensor] = None,
                    wallclock_timestamp: Optional[float] = None,
                    wallclock_timestamps: Optional[torch.Tensor] = None,
                    inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Fast Weight Associative Update (Titans Style).
        """
        try:
            current_lr = float(current_lr)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LTM current_lr must be a finite nonnegative number, got {current_lr!r}") from exc
        if not math.isfinite(current_lr) or current_lr < 0.0:
            raise ValueError(f"LTM current_lr must be a finite nonnegative number, got {current_lr!r}")
        if torch.is_tensor(grads_tensor) and not bool(torch.isfinite(grads_tensor).all().item()):
            raise ValueError("LTM inner_update rejected non-finite gradients")
        curr_fast = fast_vals if fast_vals is not None else self.fast_vals
        curr_mom = mom_vals if mom_vals is not None else self._mom_vals

        if curr_fast.dim() == 3:
            if self._use_sparse_update(curr_fast):
                result = self._inner_update_batched_cuda(
                    topk_idx=topk_idx,
                    grads_tensor=grads_tensor,
                    current_lr=current_lr,
                    timestamp=timestamp,
                    source=source,
                    tokens_covered=tokens_covered,
                    curr_fast=curr_fast,
                    curr_mom=curr_mom,
                    timestamps=timestamps,
                    sources=sources,
                    inplace=inplace
                )
            else:
                result = self._inner_update_batched(
                    topk_idx=topk_idx,
                    grads_tensor=grads_tensor,
                    current_lr=current_lr,
                    timestamp=timestamp,
                    source=source,
                    tokens_covered=tokens_covered,
                    curr_fast=curr_fast,
                    curr_mom=curr_mom,
                    timestamps=timestamps,
                    sources=sources,
                    inplace=inplace
                )
            self._stamp_wallclock_metadata(
                topk_idx,
                wallclock_timestamp,
                wallclock_timestamps,
            )
            return result
        
        if grads_tensor is None: 
            return curr_fast, curr_mom

        if self._use_sparse_update(curr_fast):
            result = self._inner_update_flat_cuda(
                topk_idx=topk_idx,
                grads_tensor=grads_tensor,
                current_lr=current_lr,
                timestamp=timestamp,
                source=source,
                tokens_covered=tokens_covered,
                curr_fast=curr_fast,
                curr_mom=curr_mom,
                timestamps=timestamps,
                sources=sources,
                inplace=inplace
            )
            self._stamp_wallclock_metadata(
                topk_idx,
                wallclock_timestamp,
                wallclock_timestamps,
            )
            return result

        fast_before = self._capture_fast_state_for_delta(curr_fast)
        device = curr_fast.device
        valid_mask = topk_idx >= 0
        if not valid_mask.any(): 
            return curr_fast, curr_mom

        # AMP FIX: Cast grads_tensor to float32 BEFORE boolean indexing.
        # Under BFloat16 AMP, grads_tensor arrives as BFloat16 but downstream
        # computations promote to float32. The backward pass of boolean indexing
        # uses masked_scatter_ which requires self and source to have the same dtype.
        # If grads_tensor is BFloat16 but the incoming gradient is float32, it crashes.
        grads_tensor = grads_tensor.float()
        curr_fast = curr_fast.float()
        curr_mom = curr_mom.float()

        idx_flat = topk_idx[valid_mask].view(-1)
        grads_flat = grads_tensor[valid_mask].view(-1, self.vals.size(1))
        grads_flat = grads_flat.contiguous()

        num_classes = self.vals.size(0)
        range_tensor = torch.arange(num_classes, device=device).unsqueeze(0)
        one_hot = (idx_flat.unsqueeze(1) == range_tensor).float()
        
        counts = one_hot.sum(dim=0)
        slot_grads = torch.matmul(one_hot.t(), grads_flat.to(device))
        
        # <<< STABILITY FIX: div + 1e-8 for safe division >>>
        slot_grads = slot_grads / (counts.unsqueeze(-1) + 1e-8)
        
        nonzero_mask = counts > 0

        if inplace:
            curr_mom.mul_(self.momentum).add_(slot_grads)
            curr_mom.clamp_(min=-50.0, max=50.0)
            new_mom = curr_mom
        else:
            new_mom = self.momentum * curr_mom + slot_grads
            new_mom = torch.clamp(new_mom, min=-50.0, max=50.0)
        
        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        
        decay_scaler = tokens_covered / float(self.reference_chunk_len)
        retention_rate = (1.0 - self.forget_rate) ** decay_scaler
        
        if inplace:
            update_step = (new_mom + self.weight_decay * curr_fast) * (-current_lr)
            
            # Global decay
            curr_fast.mul_(retention_rate)
            
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(curr_fast)
            update_step = torch.where(nonzero_mask_expanded, update_step, torch.zeros_like(update_step))
            curr_fast.add_(update_step)
            curr_fast.clamp_(min=-50.0, max=50.0)
            new_fast = curr_fast

        else:
            update_delta = (new_mom + self.weight_decay * curr_fast)
            update_step = update_delta * (-current_lr)
            
            # Apply the same global decay/update rule as the trainer's batched path.
            decayed_fast = curr_fast * retention_rate
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(decayed_fast)
            
            update_mask = torch.zeros_like(decayed_fast)
            update_step = update_step.to(dtype=update_mask.dtype)
            update_mask = torch.where(nonzero_mask_expanded, update_step, update_mask)
            
            new_fast = decayed_fast + update_mask
            new_fast = torch.clamp(new_fast, min=-50.0, max=50.0)

        target_timestamps = timestamps if timestamps is not None else self.timestamps
        target_sources = sources if sources is not None else self.sources
        with torch.no_grad():
            target_timestamps.data[nonzero_mask] = timestamp
            target_sources.data[nonzero_mask] = source

        self._record_fast_state_delta(fast_before, new_fast)
        self._stamp_wallclock_metadata(
            topk_idx,
            wallclock_timestamp,
            wallclock_timestamps,
        )
        return new_fast, new_mom

    def _inner_update_batched(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor,
                              current_lr: float, timestamp: float, source: int,
                              tokens_covered: int, curr_fast: torch.Tensor,
                              curr_mom: torch.Tensor, timestamps: Optional[torch.Tensor],
                              sources: Optional[torch.Tensor], inplace: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if grads_tensor is None:
            return curr_fast, curr_mom

        fast_before = self._capture_fast_state_for_delta(curr_fast)
        valid_mask_all = topk_idx >= 0
        if not valid_mask_all.any():
            return curr_fast, curr_mom

        grads_tensor = grads_tensor.float()
        if not inplace:
            curr_fast = curr_fast.float().clone()
            curr_mom = curr_mom.float().clone()
        else:
            curr_fast = curr_fast.float()
            curr_mom = curr_mom.float()

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))

        num_slots = self.vals.size(0)
        range_tensor = torch.arange(num_slots, device=curr_fast.device).unsqueeze(0)

        for batch_idx in range(curr_fast.shape[0]):
            valid_mask = valid_mask_all[batch_idx]
            if not valid_mask.any():
                continue

            idx_flat = topk_idx[batch_idx][valid_mask].view(-1).to(curr_fast.device)
            grads_flat = grads_tensor[batch_idx][valid_mask].view(-1, self.vals.size(1)).to(curr_fast.device)
            one_hot = (idx_flat.unsqueeze(1) == range_tensor).float()
            counts = one_hot.sum(dim=0)
            slot_grads = torch.matmul(one_hot.t(), grads_flat)
            slot_grads = slot_grads / (counts.unsqueeze(-1) + 1e-8)
            nonzero_mask = counts > 0

            curr_mom[batch_idx].mul_(self.momentum).add_(slot_grads)
            curr_mom[batch_idx].clamp_(min=-50.0, max=50.0)

            update_step = curr_mom[batch_idx] + self.weight_decay * curr_fast[batch_idx]
            update_step = update_step.mul(-current_lr)
            curr_fast[batch_idx].mul_(retention_rate)
            update_step = torch.where(nonzero_mask.unsqueeze(-1), update_step, torch.zeros_like(update_step))
            curr_fast[batch_idx].add_(update_step)
            curr_fast[batch_idx].clamp_(min=-50.0, max=50.0)

            if timestamps is not None and sources is not None:
                with torch.no_grad():
                    timestamps[batch_idx].data[nonzero_mask] = timestamp
                    sources[batch_idx].data[nonzero_mask] = source

        self._record_fast_state_delta(fast_before, curr_fast)
        return curr_fast, curr_mom

    def update_memory_hebbian(self, topk_idx: torch.LongTensor, keys: torch.Tensor, vals: torch.Tensor, current_lr: float, timestamp: float, 
                              source: int = SRC_USER_INTERACTION, tokens_covered: int = None, fast_vals: Optional[torch.Tensor] = None,
                              mom_vals: Optional[torch.Tensor] = None, timestamps: Optional[torch.Tensor] = None,
                              sources: Optional[torch.Tensor] = None,
                              wallclock_timestamp: Optional[float] = None,
                              wallclock_timestamps: Optional[torch.Tensor] = None,
                              inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Hebbian-style associative update, unified with Titans Momentum logic.
        """
        pseudo_grads = -vals
        return self.inner_update(
            topk_idx=topk_idx,
            grads_tensor=pseudo_grads,
            current_lr=current_lr,
            timestamp=timestamp,
            source=source,
            tokens_covered=tokens_covered,
            fast_vals=fast_vals,
            mom_vals=mom_vals,
            timestamps=timestamps,
            sources=sources,
            wallclock_timestamp=wallclock_timestamp,
            wallclock_timestamps=wallclock_timestamps,
            inplace=inplace
        )
