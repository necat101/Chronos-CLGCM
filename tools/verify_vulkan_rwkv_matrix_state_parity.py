#!/usr/bin/env python3
"""Verify fused Vulkan RWKV-v8 matrix-state forward/backward vs PyTorch."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(20260818)
    # Hierarchos' automatic RWKV geometry explicitly prefers 64-wide heads.
    # Exercise that production-relevant matrix size rather than a toy 4x4 state.
    batch = 2
    heads = 2
    head_size = 64
    width = heads * head_size

    state = (torch.randn(batch, heads, head_size, head_size, dtype=torch.float32) * 0.2).requires_grad_()
    r = (torch.randn(batch, width, dtype=torch.float32) * 0.15).requires_grad_()
    k = (torch.randn(batch, width, dtype=torch.float32) * 0.12).requires_grad_()
    v = (torch.randn(batch, width, dtype=torch.float32) * 0.14).requires_grad_()
    kk_raw = torch.randn(batch, heads, head_size, dtype=torch.float32)
    kk = F.normalize(kk_raw, dim=-1).reshape(batch, width).detach().requires_grad_()
    a = (0.15 + torch.rand(batch, width, dtype=torch.float32) * 0.75).requires_grad_()
    w_values = -0.5 - torch.rand(batch, width, dtype=torch.float32) * 3.0
    # Deliberately cross both production decay-clamp boundaries.  Ordinary
    # random RWKV initialization lives well inside [-60, 30] and therefore
    # cannot catch a fused-kernel regression in the clamp derivative.
    w_values.view(-1)[:8] = torch.tensor(
        [-80.0, -60.0, -59.0, -3.0, 29.0, 30.0, 45.0, 80.0],
        dtype=torch.float32,
    )
    w = w_values.requires_grad_()
    grad_new_state = torch.randn_like(state) * 0.03
    grad_tmix = torch.randn(batch, width, dtype=torch.float32) * 0.04

    r_h = r.view(batch, heads, head_size)
    k_h = k.view(batch, heads, head_size)
    v_h = v.view(batch, heads, head_size)
    kk_h = kk.view(batch, heads, head_size)
    a_h = a.view(batch, heads, head_size)
    w_h = w.view(batch, heads, head_size)
    decay = torch.exp(-torch.exp(torch.clamp(w_h, -60.0, 30.0)))
    sa = torch.matmul(state, (-kk_h).unsqueeze(-1)).squeeze(-1)
    new_state = (
        state * decay.unsqueeze(-2)
        + sa.unsqueeze(-1) * (kk_h * a_h).unsqueeze(-2)
        + v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    )
    tmix = torch.matmul(new_state, r_h.unsqueeze(-1)).squeeze(-1).reshape(batch, width)
    objective = (new_state * grad_new_state).sum() + (tmix * grad_tmix).sum()
    objective.backward()
    saturated_grad = w.grad.detach().view(-1)[:8]
    assert saturated_grad[0].item() == 0.0
    assert saturated_grad[6].item() == 0.0
    assert saturated_grad[7].item() == 0.0
    assert torch.isfinite(saturated_grad[1:6]).all()

    case = {
        "batch": batch,
        "width": width,
        "head_size": head_size,
        "state": state.detach().flatten().tolist(),
        "r": r.detach().flatten().tolist(),
        "k": k.detach().flatten().tolist(),
        "v": v.detach().flatten().tolist(),
        "kk": kk.detach().flatten().tolist(),
        "a": a.detach().flatten().tolist(),
        "w": w.detach().flatten().tolist(),
        "grad_new_state": grad_new_state.flatten().tolist(),
        "grad_tmix": grad_tmix.flatten().tolist(),
    }
    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-rwkv-state-") as temp_dir:
        case_path = Path(temp_dir) / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        command = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-rwkv-state-step",
            "--",
            "--case",
            str(case_path),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan RWKV matrix-state parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    comparisons = {
        "new_state": (result["new_state"], new_state.detach()),
        "tmix": (result["tmix"], tmix.detach()),
        "grad_state": (result["grad_state"], state.grad),
        "grad_r": (result["grad_r"], r.grad),
        "grad_k": (result["grad_k"], k.grad),
        "grad_v": (result["grad_v"], v.grad),
        "grad_kk": (result["grad_kk"], kk.grad),
        "grad_a": (result["grad_a"], a.grad),
        "grad_w": (result["grad_w"], w.grad),
    }
    diffs = {}
    for name, (actual_values, expected) in comparisons.items():
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-7)
        diffs[name] = (actual - expected).abs().max().item()

    print(f"device={result['device']} heads={result['heads']} head_size={head_size}")
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    print("Hierarchos Vulkan RWKV-v8 matrix-state forward/backward parity: PASS")


if __name__ == "__main__":
    main()
