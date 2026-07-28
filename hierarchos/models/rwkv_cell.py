import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_preserve_nonfinite(tensor: torch.Tensor, minimum: float, maximum: float):
    """Clamp finite values while leaving NaN/Inf visible to outer guards."""
    clamped = torch.clamp(tensor, min=minimum, max=maximum)
    return torch.where(torch.isfinite(tensor), clamped, tensor)


def _choose_head_size(n_embd: int, requested=None) -> int:
    n_embd = int(n_embd)
    if requested:
        requested = int(requested)
        if requested > 0 and n_embd % requested == 0:
            return requested

    preferred = 64
    candidates = [
        size
        for size in range(1, min(n_embd, 128) + 1)
        if n_embd % size == 0
    ]
    if not candidates:
        return 1

    real_matrix_heads = [size for size in candidates if size >= 16]
    if real_matrix_heads:
        candidates = real_matrix_heads

    return min(candidates, key=lambda size: (abs(math.log(size / preferred)), size > preferred))


def _rwkv_lora_rank(width: int, scale: float) -> int:
    if width < 128:
        return 8
    return max(32, int(round((scale * math.sqrt(width)) / 32.0) * 32))


def _ortho_init(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    with torch.no_grad():
        shape = tensor.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(tensor, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(tensor[i], gain=gain * scale)
        else:
            raise ValueError(f"Unsupported orthogonal init shape: {shape}")
    return tensor


class RWKVCell(nn.Module):
    """
    RWKV7/Heron-style recurrent block for Hierarchos H/L modules.

    Public interface intentionally stays compatible with the old Hierarchos cell:
    forward(x, state, timestep=None, deepemb_vec=None) -> (x, state).

    Legacy packed state layout, shape [B, C, 3 + head_size]:
      slot 0: previous time-mix input (LayerNorm output)
      slot 1: previous channel-mix input (LayerNorm output)
      slot 2: latest v_first value, reserved for stack-compatible checkpoints
      slots 3: per-head matrix state reshaped as [B, C, head_size]

    Corrected ``explicit-output`` state layout, shape [B, C, 4 + head_size]:
      slots 0..2: same recurrent caches as the legacy layout
      slot 3: the block's most recent output (the public hidden readout)
      slots 4: per-head matrix state reshaped as [B, C, head_size]

    The legacy layout remains the default so checkpoints without an explicit
    recurrence-version declaration keep byte-for-byte compatible behavior.
    """

    def __init__(
        self,
        n_embd,
        head_size=None,
        layer_id: int = 0,
        n_layer: int = 2,
        channel_mix_key_clamp: float = 12.0,
        channel_mix_deepembed_clamp: float = 4.0,
        state_readout_mode: str = "legacy-input-cache",
        state_clamp: float = 50.0,
    ):
        super().__init__()
        self.n_embd = int(n_embd)
        self.head_size = _choose_head_size(self.n_embd, head_size)
        self.n_head = self.n_embd // self.head_size
        self.layer_id = int(layer_id)
        self.n_layer = max(2, int(n_layer))
        self.state_readout_mode = str(state_readout_mode or "legacy-input-cache")
        if self.state_readout_mode not in {"legacy-input-cache", "explicit-output"}:
            raise ValueError(
                "state_readout_mode must be 'legacy-input-cache' or "
                f"'explicit-output', got {self.state_readout_mode!r}"
            )
        self.matrix_offset = 4 if self.state_readout_mode == "explicit-output" else 3
        self.state_size = self.matrix_offset + self.head_size
        self.state_clamp = None if state_clamp is None else float(state_clamp)
        self.allow_legacy_state_migration = True
        self._compiled_impl = None
        try:
            self.channel_mix_key_clamp = float(channel_mix_key_clamp or 0.0)
        except (TypeError, ValueError):
            self.channel_mix_key_clamp = 12.0
        try:
            self.channel_mix_deepembed_clamp = float(channel_mix_deepembed_clamp or 0.0)
        except (TypeError, ValueError):
            self.channel_mix_deepembed_clamp = 4.0

        C = self.n_embd
        H = self.n_head
        N = self.head_size

        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / max(1, self.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)
            ddd = torch.arange(C, dtype=torch.float32).view(1, C) / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            linear = torch.arange(C, dtype=torch.float32) / max(1, C - 1) - 0.5
            zigzag = torch.empty(C, dtype=torch.float32)
            www = torch.empty(C, dtype=torch.float32)
            for n in range(C):
                if N > 1:
                    z = ((n % N) - ((N - 1) / 2.0)) / ((N - 1) / 2.0)
                else:
                    z = 0.0
                zigzag[n] = z * abs(z)
                www[n] = -6 + 6 * (n / max(1, C - 1)) ** (1 + ratio_0_to_1 ** 0.3)

            d_decay = _rwkv_lora_rank(C, 2.5)
            self.w1 = nn.Parameter(torch.zeros(C, d_decay))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(d_decay, C), 0.1))
            self.w0 = nn.Parameter((www + 0.5 + zigzag * 2.5).view(1, C))

            d_aaa = _rwkv_lora_rank(C, 2.5)
            self.a1 = nn.Parameter(torch.zeros(C, d_aaa))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(d_aaa, C), 0.1))
            self.a0 = nn.Parameter((-0.19 + zigzag * 0.3 + linear * 0.4).view(1, C))

            if self.layer_id > 0:
                d_mv = _rwkv_lora_rank(C, 1.7)
                self.v1 = nn.Parameter(torch.zeros(C, d_mv))
                self.v2 = nn.Parameter(_ortho_init(torch.zeros(d_mv, C), 0.1))
                self.v0 = nn.Parameter((0.73 - linear * 0.4).view(1, C))

            d_gate = _rwkv_lora_rank(C, 5.0)
            self.g1 = nn.Parameter(torch.zeros(C, d_gate))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(d_gate, C), 0.1))

            self.k_k = nn.Parameter((0.71 - linear * 0.1).view(1, C))
            self.k_a = nn.Parameter(torch.ones(1, C) * 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self.x_k_cm = nn.Parameter(torch.zeros(1, C))
        self.key_cm = nn.Linear(C, C * 4, bias=False)
        self.value_cm = nn.Linear(C * 4, C, bias=False)

        with torch.no_grad():
            self.receptance.weight.data.uniform_(-0.5 / math.sqrt(C), 0.5 / math.sqrt(C))
            self.key.weight.data.uniform_(-0.05 / math.sqrt(C), 0.05 / math.sqrt(C))
            self.value.weight.data.uniform_(-0.5 / math.sqrt(C), 0.5 / math.sqrt(C))
            init_scale = 0.01 / math.sqrt(C)
            self.output.weight.data.uniform_(-init_scale, init_scale)
            self.value_cm.weight.data.uniform_(-init_scale, init_scale)
            nn.init.orthogonal_(self.key_cm.weight.data, gain=math.sqrt(4.0))

    def initial_state(self, batch_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(
            int(batch_size),
            self.n_embd,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def _empty_state(self, batch_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.empty(
            int(batch_size),
            self.n_embd,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def state_hidden(self, state: torch.Tensor) -> torch.Tensor:
        if state is None:
            raise ValueError("state_hidden requires a non-None state")
        if state.dim() == 4 and state.shape[0] == 1:
            state = state.squeeze(0)
        hidden_slot = 3 if self.state_readout_mode == "explicit-output" else 0
        if state.shape[-1] <= hidden_slot:
            raise ValueError(
                f"RWKV state has {state.shape[-1]} slots, but "
                f"{self.state_readout_mode!r} requires slot {hidden_slot}"
            )
        return state[:, :, hidden_slot]

    def compile_forward(self, **compile_kwargs) -> None:
        if not hasattr(torch, "compile"):
            return
        self.allow_legacy_state_migration = False
        object.__setattr__(
            self,
            "_compiled_impl",
            torch.compile(self._forward_impl, **compile_kwargs),
        )

    def _prepare_state(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if state is None:
            return self.initial_state(B, device=x.device, dtype=torch.float32)
        if state.dim() == 4 and state.shape[0] == 1:
            state = state.squeeze(0)
        state = state.to(device=x.device, dtype=torch.float32)
        if state.shape == (B, self.n_embd, self.state_size):
            if not self.allow_legacy_state_migration:
                return state
            # Old scalar-WKV states also used 5 slots and stored pp=-1e30 in
            # slot 3. For tiny head_size=2 this collides with the new packed
            # shape, so treat that sentinel as a legacy state to migrate.
            state_tail = state[:, :, self.matrix_offset:]
            scalar_wkv_sentinel = (
                state.shape[-1] >= 5
                and state[:, :, 3].amin() <= -1e20
            )
            if (
                not scalar_wkv_sentinel
                and (state_tail.numel() == 0 or state_tail.amin() > -1e20)
            ):
                return state

        migrated = self.initial_state(B, device=x.device, dtype=torch.float32)
        if state.dim() != 3:
            return migrated

        common_b = min(B, state.shape[0])
        common_c = min(self.n_embd, state.shape[1])
        source = state[:common_b, :common_c]
        source_slots = int(state.shape[-1])
        scalar_wkv_sentinel = (
            source_slots >= 5 and source[:, :, 3].amin() <= -1e20
        )

        if source_slots > 0:
            migrated[:common_b, :common_c, 0] = source[:, :, 0]
        if scalar_wkv_sentinel:
            # Historical scalar-WKV layout: slot 4 was the channel-mix cache.
            migrated[:common_b, :common_c, 1] = source[:, :, 4]
            if self.state_readout_mode == "explicit-output":
                migrated[:common_b, :common_c, 3] = source[:, :, 0]
            return migrated

        prefix_slots = min(3, source_slots)
        if prefix_slots > 0:
            migrated[:common_b, :common_c, :prefix_slots] = source[:, :, :prefix_slots]

        explicit_source = source_slots == 4 + self.head_size
        source_matrix_offset = 4 if explicit_source else 3
        if self.state_readout_mode == "explicit-output":
            if explicit_source:
                migrated[:common_b, :common_c, 3] = source[:, :, 3]
            elif source_slots > 0:
                # An old state cannot recover the historical block output.
                # Seeding with its old public readout makes migration continuous
                # while all future states use the corrected explicit output.
                migrated[:common_b, :common_c, 3] = source[:, :, 0]

        available_matrix = max(0, source_slots - source_matrix_offset)
        matrix_slots = min(self.head_size, available_matrix)
        if matrix_slots > 0:
            migrated[
                :common_b,
                :common_c,
                self.matrix_offset:self.matrix_offset + matrix_slots,
            ] = source[
                :,
                :,
                source_matrix_offset:source_matrix_offset + matrix_slots,
            ]
        return migrated

    def _mix(self, x: torch.Tensor, previous: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        return x + (previous - x) * coeff.to(dtype=x.dtype, device=x.device)

    def forward(self, x, state, timestep=None, deepemb_vec=None):
        if self._compiled_impl is not None:
            return self._compiled_impl(x, state, timestep, deepemb_vec)
        return self._forward_impl(x, state, timestep, deepemb_vec)

    def _forward_impl(self, x, state, timestep=None, deepemb_vec=None):
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if deepemb_vec is not None and deepemb_vec.dim() == 3 and deepemb_vec.shape[0] == 1:
            deepemb_vec = deepemb_vec.squeeze(0)

        state = self._prepare_state(state, x)

        detach_every_n_steps = getattr(self, "detach_every_n_steps", None)
        if self.training and detach_every_n_steps is not None and timestep is not None:
            if timestep > 0 and timestep % detach_every_n_steps == 0:
                state = state.detach()

        B, C = x.shape
        H, N = self.n_head, self.head_size
        x_dtype = x.dtype
        x_resid_tm = x

        prev_tm = state[:, :, 0].to(dtype=x_dtype)
        prev_cm = state[:, :, 1].to(dtype=x_dtype)
        matrix_state = state[:, :, self.matrix_offset:].contiguous().view(B, H, N, N)

        x_norm = self.ln1(x)
        xr = self._mix(x_norm, prev_tm, self.x_r)
        xw = self._mix(x_norm, prev_tm, self.x_w)
        xk = self._mix(x_norm, prev_tm, self.x_k)
        xv = self._mix(x_norm, prev_tm, self.x_v)
        xa = self._mix(x_norm, prev_tm, self.x_a)
        xg = self._mix(x_norm, prev_tm, self.x_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v_first = state[:, :, 2].to(dtype=x_dtype)
            v = v + (v_first - v) * torch.sigmoid(
                self.v0.to(x_dtype) + (xv @ self.v1.to(x_dtype)) @ self.v2.to(x_dtype)
            )
        a = torch.sigmoid(self.a0.to(x_dtype) + (xa @ self.a1.to(x_dtype)) @ self.a2.to(x_dtype))
        g = torch.sigmoid(xg @ self.g1.to(x_dtype)) @ self.g2.to(x_dtype)
        w = -F.softplus(-(self.w0.to(x_dtype) + torch.tanh(xw @ self.w1.to(x_dtype)) @ self.w2.to(x_dtype))) - 0.5

        kk = k * self.k_k.to(x_dtype)
        kk = F.normalize(kk.view(B, H, N), dim=-1, p=2.0).view(B, C)
        k = k * (1 + (a - 1) * self.k_a.to(x_dtype))
        state_a = -kk
        state_b = kk * a

        autocast_device = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            r_f = r.float().view(B, H, N)
            k_f = k.float().view(B, H, N)
            v_f = v.float().view(B, H, N)
            a_f = state_a.float().view(B, H, N)
            b_f = state_b.float().view(B, H, N)
            bounded_w = _clamp_preserve_nonfinite(w.float(), -60.0, 30.0)
            w_decay = torch.exp(-torch.exp(bounded_w)).view(B, H, N)
            state_f = matrix_state.float()

            sa = torch.matmul(state_f, a_f.unsqueeze(-1)).squeeze(-1)
            state_f = (
                state_f * w_decay.unsqueeze(-2)
                + sa.unsqueeze(-1) * b_f.unsqueeze(-2)
                + v_f.unsqueeze(-1) * k_f.unsqueeze(-2)
            )
            tmix = torch.matmul(state_f, r_f.unsqueeze(-1)).squeeze(-1).reshape(B, C)

        tmix = self.ln_x(tmix.to(dtype=x_dtype))
        bonus = (
            (
                r.view(B, H, N)
                * k.view(B, H, N)
                * self.r_k.to(dtype=x_dtype, device=x.device)
            ).sum(dim=-1, keepdim=True)
            * v.view(B, H, N)
        ).view(B, C)
        tmix = tmix + bonus
        x = x_resid_tm + self.output(tmix * g)

        x_resid_cm = x
        x_norm2 = self.ln2(x)
        xk_cm = self._mix(x_norm2, prev_cm, self.x_k_cm)
        cm_key = self.key_cm(xk_cm)
        if self.channel_mix_key_clamp > 0.0:
            cm_key = _clamp_preserve_nonfinite(
                cm_key,
                -self.channel_mix_key_clamp,
                self.channel_mix_key_clamp,
            )
        ffn = torch.square(torch.relu(cm_key))
        if deepemb_vec is not None:
            deepemb = deepemb_vec.to(dtype=ffn.dtype, device=ffn.device)
            if self.channel_mix_deepembed_clamp > 0.0:
                clamped_deepemb = _clamp_preserve_nonfinite(
                    deepemb,
                    -self.channel_mix_deepembed_clamp,
                    self.channel_mix_deepembed_clamp,
                )
                # Do not turn a broken adapter trajectory into a plausible
                # multiplicative gate. Preserve NaN/Inf so the model-level
                # numerical guard rejects the forward/batch.
                deepemb = torch.where(
                    torch.isfinite(deepemb),
                    clamped_deepemb,
                    deepemb,
                )
            ffn = ffn * deepemb
            if self.channel_mix_key_clamp > 0.0 and self.channel_mix_deepembed_clamp > 0.0:
                ffn_limit = (
                    self.channel_mix_key_clamp
                    * self.channel_mix_key_clamp
                    * self.channel_mix_deepembed_clamp
                )
                ffn = _clamp_preserve_nonfinite(
                    ffn,
                    -ffn_limit,
                    ffn_limit,
                )
        x = x_resid_cm + self.value_cm(ffn)

        state_parts = [
            x_norm.float().unsqueeze(-1),
            x_norm2.float().unsqueeze(-1),
            v_first.float().unsqueeze(-1),
        ]
        if self.state_readout_mode == "explicit-output":
            state_parts.append(x.float().unsqueeze(-1))
        state_parts.append(state_f.reshape(B, C, N))
        new_state = torch.cat(state_parts, dim=-1)
        if self.state_clamp is not None:
            new_state = _clamp_preserve_nonfinite(
                new_state,
                -self.state_clamp,
                self.state_clamp,
            )

        return x, new_state
