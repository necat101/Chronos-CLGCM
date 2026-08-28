#!/usr/bin/env python3
"""Verify the single-submit Vulkan RWKV a/w/g + recurrent core against PyTorch."""

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
    torch.manual_seed(20260812)
    batch = 2
    heads = 2
    head_size = 64
    width = heads * head_size
    w_rank = 32
    a_rank = 32
    g_rank = 64

    state = (
        torch.randn(batch, heads, head_size, head_size, dtype=torch.float32) * 0.12
    ).requires_grad_()
    x_norm = (torch.randn(batch, width, dtype=torch.float32) * 0.35).requires_grad_()
    previous = (torch.randn(batch, width, dtype=torch.float32) * 0.3).requires_grad_()

    def vector(base: float = 0.0, scale: float = 1.0) -> torch.Tensor:
        return (base + torch.rand(width, dtype=torch.float32) * scale).requires_grad_()

    mix_r = vector()
    mix_k = vector()
    mix_v = vector()
    mix_w = vector()
    mix_a = vector()
    mix_g = vector()

    def matrix(rows: int, cols: int, scale: float = 0.18) -> torch.Tensor:
        return (
            torch.randn(rows, cols, dtype=torch.float32) * (scale / math.sqrt(rows))
        ).requires_grad_()

    receptance_weight = matrix(width, width, 0.35)
    key_weight = matrix(width, width, 0.35)
    value_weight = matrix(width, width, 0.35)
    k_k = vector(0.65, 0.12)
    k_a = vector(0.95, 0.12)
    w0 = (-2.5 + torch.randn(1, width, dtype=torch.float32) * 0.4).requires_grad_()
    w1 = matrix(width, w_rank)
    w2 = matrix(w_rank, width)
    a0 = (torch.randn(1, width, dtype=torch.float32) * 0.25).requires_grad_()
    a1 = matrix(width, a_rank)
    a2 = matrix(a_rank, width)
    g1 = matrix(width, g_rank)
    g2 = matrix(g_rank, width)

    delta = previous - x_norm
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
    sa = torch.matmul(state, (-kk_h).unsqueeze(-1)).squeeze(-1)
    new_state = (
        state * decay.unsqueeze(-2)
        + sa.unsqueeze(-1) * (kk_h * a_h).unsqueeze(-2)
        + v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    )
    tmix = torch.matmul(new_state, r_h.unsqueeze(-1)).squeeze(-1).reshape(batch, width)

    grad_new_state = torch.randn_like(new_state) * 0.025
    grad_tmix = torch.randn_like(tmix) * 0.04
    grad_g = torch.randn_like(g) * 0.045
    objective = (
        (new_state * grad_new_state).sum()
        + (tmix * grad_tmix).sum()
        + (g * grad_g).sum()
    )
    objective.backward()

    case = {
        "batch": batch,
        "width": width,
        "head_size": head_size,
        "state": state.detach().flatten().tolist(),
        "x_norm": x_norm.detach().flatten().tolist(),
        "previous": previous.detach().flatten().tolist(),
        "grad_new_state": grad_new_state.flatten().tolist(),
        "grad_tmix": grad_tmix.flatten().tolist(),
        "grad_g": grad_g.flatten().tolist(),
    }

    tensors = {
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
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-rwkv-fused-") as temp_dir:
        temp = Path(temp_dir)
        case_path = temp / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        package_dir = temp / "model-package"
        package_dir.mkdir()
        save_file(
            tensors,
            str(package_dir / "model.safetensors"),
            metadata={"format": "pt", "layout": "pytorch-row-major"},
        )
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-rwkv-fused-time-mix-step",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(package_dir),
                "--prefix",
                "h_rnn",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan fused RWKV parity runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    comparisons = {
        "new_state": (result["new_state"], new_state.detach()),
        "tmix": (result["tmix"], tmix.detach()),
        "scaled_k": (result["scaled_k"], scaled_k.detach()),
        "kk": (result["kk"], kk.detach()),
        "a": (result["a"], a.detach()),
        "w": (result["w"], w.detach()),
        "g": (result["g"], g.detach()),
        "grad_state": (result["grad_state"], state.grad),
        "grad_x_norm": (result["grad_x_norm"], x_norm.grad),
        "grad_previous": (result["grad_previous"], previous.grad),
        "grad_mix_r": (result["grad_mix_r"], mix_r.grad),
        "grad_mix_k": (result["grad_mix_k"], mix_k.grad),
        "grad_mix_v": (result["grad_mix_v"], mix_v.grad),
        "grad_mix_w": (result["grad_mix_w"], mix_w.grad),
        "grad_mix_a": (result["grad_mix_a"], mix_a.grad),
        "grad_mix_g": (result["grad_mix_g"], mix_g.grad),
        "grad_receptance_weight": (
            result["grad_receptance_weight"],
            receptance_weight.grad,
        ),
        "grad_key_weight": (result["grad_key_weight"], key_weight.grad),
        "grad_value_weight": (result["grad_value_weight"], value_weight.grad),
        "grad_k_k": (result["grad_k_k"], k_k.grad),
        "grad_k_a": (result["grad_k_a"], k_a.grad),
        "grad_w0": (result["grad_w0"], w0.grad),
        "grad_w1": (result["grad_w1"], w1.grad),
        "grad_w2": (result["grad_w2"], w2.grad),
        "grad_a0": (result["grad_a0"], a0.grad),
        "grad_a1": (result["grad_a1"], a1.grad),
        "grad_a2": (result["grad_a2"], a2.grad),
        "grad_g1": (result["grad_g1"], g1.grad),
        "grad_g2": (result["grad_g2"], g2.grad),
    }

    diffs: dict[str, float] = {}
    for name, (actual_values, expected) in comparisons.items():
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=4.0e-4, atol=4.0e-6)
        diffs[name] = (actual - expected).abs().max().item()

    print(f"device={result['device']} heads={result['heads']} head_size={head_size}")
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    print("Hierarchos Vulkan single-submit a/w/g + recurrent-core parity: PASS")


if __name__ == "__main__":
    main()
