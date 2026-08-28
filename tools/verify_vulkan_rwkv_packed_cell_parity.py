#!/usr/bin/env python3
"""Verify the packed/clamped single-submit Vulkan RWKV cell against PyTorch."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(20260820)
    batch = 2
    heads = 2
    head_size = 64
    width = heads * head_size
    hidden_width = width * 4
    input_dim = 32
    adapter_rank = 16
    w_rank = 32
    a_rank = 32
    g_rank = 64
    matrix_offset = 4
    state_size = matrix_offset + head_size
    state_clamp = 0.75

    x = (torch.randn(batch, width) * 0.25).requires_grad_()
    token_features = (torch.randn(batch, input_dim) * 0.3).requires_grad_()
    packed_state = (torch.randn(batch, width, state_size) * 0.28).requires_grad_()
    previous_tm = packed_state[:, :, 0]
    previous_cm = packed_state[:, :, 1]
    matrix_state = packed_state[:, :, matrix_offset:].reshape(
        batch, heads, head_size, head_size
    )

    def vector(base: float = 0.0, scale: float = 1.0) -> torch.Tensor:
        return (base + torch.rand(width) * scale).requires_grad_()

    def matrix(rows: int, cols: int, scale: float) -> torch.Tensor:
        return (
            torch.randn(rows, cols) * (scale / math.sqrt(max(1, rows)))
        ).requires_grad_()

    ln1_weight = (0.9 + torch.rand(width) * 0.2).requires_grad_()
    ln1_bias = (torch.randn(width) * 0.04).requires_grad_()
    mix_r, mix_k, mix_v = vector(), vector(), vector()
    mix_w, mix_a, mix_g = vector(), vector(), vector()
    receptance_weight = matrix(width, width, 0.35)
    key_weight = matrix(width, width, 0.35)
    value_weight = matrix(width, width, 0.35)
    k_k = vector(0.65, 0.12)
    k_a = vector(0.95, 0.12)
    w0 = (-2.5 + torch.randn(1, width) * 0.4).requires_grad_()
    w1 = matrix(width, w_rank, 0.18)
    w2 = matrix(w_rank, width, 0.18)
    a0 = (torch.randn(1, width) * 0.25).requires_grad_()
    a1 = matrix(width, a_rank, 0.18)
    a2 = matrix(a_rank, width, 0.18)
    g1 = matrix(width, g_rank, 0.18)
    g2 = matrix(g_rank, width, 0.18)
    r_k = (torch.randn(heads, head_size) * 0.08).requires_grad_()
    group_norm_weight = (0.9 + torch.rand(width) * 0.2).requires_grad_()
    group_norm_bias = (torch.randn(width) * 0.08).requires_grad_()
    output_weight = matrix(width, width, 0.24)

    ln2_weight = (0.9 + torch.rand(width) * 0.2).requires_grad_()
    ln2_bias = (torch.randn(width) * 0.04).requires_grad_()
    mix_k_cm = (torch.randn(width) * 0.12).requires_grad_()
    key_cm_weight = matrix(hidden_width, width, 0.28)
    value_cm_weight = matrix(width, hidden_width, 0.18)
    adapter_down = (
        torch.randn(adapter_rank, input_dim) * (0.12 / math.sqrt(input_dim))
    ).requires_grad_()
    adapter_up = (
        torch.randn(hidden_width, adapter_rank) * (0.06 / math.sqrt(adapter_rank))
    ).requires_grad_()
    adapter_bias = (torch.ones(hidden_width) + torch.randn(hidden_width) * 0.03).requires_grad_()

    token_norm = F.layer_norm(token_features, (input_dim,), eps=1.0e-5)
    deepembed = adapter_bias + F.linear(
        F.silu(F.linear(token_norm, adapter_down)), adapter_up
    )
    x_norm = F.layer_norm(x, (width,), ln1_weight, ln1_bias, 1.0e-5)
    delta = previous_tm - x_norm
    xr = x_norm + delta * mix_r
    xk = x_norm + delta * mix_k
    xv = x_norm + delta * mix_v
    xw = x_norm + delta * mix_w
    xa = x_norm + delta * mix_a
    xg = x_norm + delta * mix_g
    r = F.linear(xr, receptance_weight)
    raw_k = F.linear(xk, key_weight)
    v = F.linear(xv, value_weight)
    w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2
    kk = F.normalize(
        (raw_k * k_k).view(batch, heads, head_size), dim=-1, p=2.0, eps=1.0e-12
    ).view(batch, width)
    scaled_k = raw_k * (1.0 + (a - 1.0) * k_a)
    r_h = r.view(batch, heads, head_size)
    k_h = scaled_k.view(batch, heads, head_size)
    v_h = v.view(batch, heads, head_size)
    kk_h = kk.view(batch, heads, head_size)
    a_h = a.view(batch, heads, head_size)
    decay = torch.exp(
        -torch.exp(torch.clamp(w, -60.0, 30.0).view(batch, heads, head_size))
    )
    sa = torch.matmul(matrix_state, (-kk_h).unsqueeze(-1)).squeeze(-1)
    new_matrix_state = (
        matrix_state * decay.unsqueeze(-2)
        + sa.unsqueeze(-1) * (kk_h * a_h).unsqueeze(-2)
        + v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    )
    tmix = torch.matmul(new_matrix_state, r_h.unsqueeze(-1)).squeeze(-1).reshape(batch, width)
    group_normed = F.group_norm(
        tmix,
        heads,
        weight=group_norm_weight,
        bias=group_norm_bias,
        eps=64e-5,
    )
    bonus = ((r_h * k_h * r_k).sum(dim=-1, keepdim=True) * v_h).reshape(batch, width)
    x_after_time = x + F.linear((group_normed + bonus) * g, output_weight)
    x_norm2 = F.layer_norm(x_after_time, (width,), ln2_weight, ln2_bias, 1.0e-5)
    mixed_cm = x_norm2 + (previous_cm - x_norm2) * mix_k_cm
    cm_key = torch.clamp(F.linear(mixed_cm, key_cm_weight), -12.0, 12.0)
    ffn = torch.square(torch.relu(cm_key)) * torch.clamp(deepembed, -4.0, 4.0)
    ffn = torch.clamp(ffn, -576.0, 576.0)
    output = x_after_time + F.linear(ffn, value_cm_weight)

    packed_new_state = torch.cat(
        [
            x_norm.unsqueeze(-1),
            x_norm2.unsqueeze(-1),
            v.unsqueeze(-1),
            output.unsqueeze(-1),
            new_matrix_state.reshape(batch, width, head_size),
        ],
        dim=-1,
    )
    packed_new_state = torch.clamp(packed_new_state, -state_clamp, state_clamp)
    grad_output = torch.randn_like(output) * 0.03
    grad_packed_new_state = torch.randn_like(packed_new_state) * 0.02
    objective = (output * grad_output).sum() + (
        packed_new_state * grad_packed_new_state
    ).sum()
    objective.backward()

    tensors = {
        "h_rnn.ln1.weight": ln1_weight.detach().contiguous(),
        "h_rnn.ln1.bias": ln1_bias.detach().contiguous(),
        "h_rnn.x_r": mix_r.detach().view(1, width).contiguous(),
        "h_rnn.x_k": mix_k.detach().view(1, width).contiguous(),
        "h_rnn.x_v": mix_v.detach().view(1, width).contiguous(),
        "h_rnn.x_w": mix_w.detach().view(1, width).contiguous(),
        "h_rnn.x_a": mix_a.detach().view(1, width).contiguous(),
        "h_rnn.x_g": mix_g.detach().view(1, width).contiguous(),
        "h_rnn.receptance.weight": receptance_weight.detach().contiguous(),
        "h_rnn.key.weight": key_weight.detach().contiguous(),
        "h_rnn.value.weight": value_weight.detach().contiguous(),
        "h_rnn.k_k": k_k.detach().view(1, width).contiguous(),
        "h_rnn.k_a": k_a.detach().view(1, width).contiguous(),
        "h_rnn.w0": w0.detach().contiguous(),
        "h_rnn.w1": w1.detach().contiguous(),
        "h_rnn.w2": w2.detach().contiguous(),
        "h_rnn.a0": a0.detach().contiguous(),
        "h_rnn.a1": a1.detach().contiguous(),
        "h_rnn.a2": a2.detach().contiguous(),
        "h_rnn.g1": g1.detach().contiguous(),
        "h_rnn.g2": g2.detach().contiguous(),
        "h_rnn.r_k": r_k.detach().contiguous(),
        "h_rnn.ln_x.weight": group_norm_weight.detach().contiguous(),
        "h_rnn.ln_x.bias": group_norm_bias.detach().contiguous(),
        "h_rnn.output.weight": output_weight.detach().contiguous(),
        "h_rnn.ln2.weight": ln2_weight.detach().contiguous(),
        "h_rnn.ln2.bias": ln2_bias.detach().contiguous(),
        "h_rnn.x_k_cm": mix_k_cm.detach().view(1, width).contiguous(),
        "h_rnn.key_cm.weight": key_cm_weight.detach().contiguous(),
        "h_rnn.value_cm.weight": value_cm_weight.detach().contiguous(),
        "h_deepembed_adapter.down.weight": adapter_down.detach().contiguous(),
        "h_deepembed_adapter.up.weight": adapter_up.detach().contiguous(),
        "h_deepembed_adapter.bias": adapter_bias.detach().contiguous(),
    }
    case = {
        "batch": batch,
        "width": width,
        "head_size": head_size,
        "input_dim": input_dim,
        "state_mode": "explicit-output",
        "state_clamp": state_clamp,
        "x": x.detach().flatten().tolist(),
        "token_features": token_features.detach().flatten().tolist(),
        "packed_state": packed_state.detach().flatten().tolist(),
        "grad_output": grad_output.flatten().tolist(),
        "grad_packed_new_state": grad_packed_new_state.flatten().tolist(),
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-packed-cell-") as temp_dir:
        temp = Path(temp_dir)
        package_dir = temp / "model-package"
        package_dir.mkdir()
        save_file(
            tensors,
            str(package_dir / "model.safetensors"),
            metadata={"format": "pt", "layout": "pytorch-row-major"},
        )
        case_path = temp / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-rwkv-packed-cell-step",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(package_dir),
                "--cell-prefix",
                "h_rnn",
                "--adapter-prefix",
                "h_deepembed_adapter",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan packed RWKV cell parity runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    comparisons = {
        "output": (result["output"], output.detach()),
        "packed_new_state": (result["packed_new_state"], packed_new_state.detach()),
        "grad_x": (result["grad_x"], x.grad),
        "grad_packed_state": (result["grad_packed_state"], packed_state.grad),
        "token_feature_grad": (result["token_feature_grad"], token_features.grad),
    }
    diffs: dict[str, float] = {}
    for name, (actual_values, expected) in comparisons.items():
        assert expected is not None
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=2.0e-3, atol=2.0e-5)
        diffs[name] = (actual - expected).abs().max().item()
    print(
        f"device={result['device']} width={result['width']} state_size={result['state_size']}"
    )
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    print("Hierarchos Vulkan packed/clamped full RWKV cell PyTorch parity: PASS")


if __name__ == "__main__":
    main()
