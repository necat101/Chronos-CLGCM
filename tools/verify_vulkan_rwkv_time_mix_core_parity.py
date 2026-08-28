#!/usr/bin/env python3
"""Verify the composed Vulkan RWKV r/k/v time-mix core against PyTorch."""

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

    state = (
        torch.randn(batch, heads, head_size, head_size, dtype=torch.float32) * 0.15
    ).requires_grad_()
    x_norm = (torch.randn(batch, width, dtype=torch.float32) * 0.4).requires_grad_()
    previous = (torch.randn(batch, width, dtype=torch.float32) * 0.35).requires_grad_()
    mix_r = torch.rand(width, dtype=torch.float32).requires_grad_()
    mix_k = torch.rand(width, dtype=torch.float32).requires_grad_()
    mix_v = torch.rand(width, dtype=torch.float32).requires_grad_()

    weight_scale = 0.35 / math.sqrt(width)
    receptance_weight = (
        torch.randn(width, width, dtype=torch.float32) * weight_scale
    ).requires_grad_()
    key_weight = (
        torch.randn(width, width, dtype=torch.float32) * weight_scale
    ).requires_grad_()
    value_weight = (
        torch.randn(width, width, dtype=torch.float32) * weight_scale
    ).requires_grad_()
    k_k = (0.65 + torch.rand(width, dtype=torch.float32) * 0.12).requires_grad_()
    k_a = (0.95 + torch.rand(width, dtype=torch.float32) * 0.12).requires_grad_()
    a = (0.15 + torch.rand(batch, width, dtype=torch.float32) * 0.75).requires_grad_()
    w = (-0.55 - torch.rand(batch, width, dtype=torch.float32) * 3.0).requires_grad_()

    delta = previous - x_norm
    xr = x_norm + delta * mix_r
    xk = x_norm + delta * mix_k
    xv = x_norm + delta * mix_v
    r = F.linear(xr, receptance_weight)
    raw_k = F.linear(xk, key_weight)
    v = F.linear(xv, value_weight)

    kk = F.normalize(
        (raw_k * k_k).view(batch, heads, head_size), dim=-1, p=2.0, eps=1.0e-12
    ).view(batch, width)
    scaled_k = raw_k * (1.0 + (a - 1.0) * k_a)

    r_h = r.view(batch, heads, head_size)
    k_h = scaled_k.view(batch, heads, head_size)
    v_h = v.view(batch, heads, head_size)
    kk_h = kk.view(batch, heads, head_size)
    a_h = a.view(batch, heads, head_size)
    bounded_w = torch.clamp(w, -60.0, 30.0).view(batch, heads, head_size)
    decay = torch.exp(-torch.exp(bounded_w))
    sa = torch.matmul(state, (-kk_h).unsqueeze(-1)).squeeze(-1)
    new_state = (
        state * decay.unsqueeze(-2)
        + sa.unsqueeze(-1) * (kk_h * a_h).unsqueeze(-2)
        + v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    )
    tmix = torch.matmul(new_state, r_h.unsqueeze(-1)).squeeze(-1).reshape(batch, width)

    grad_new_state = torch.randn_like(new_state) * 0.025
    grad_tmix = torch.randn_like(tmix) * 0.04
    objective = (new_state * grad_new_state).sum() + (tmix * grad_tmix).sum()
    objective.backward()

    case = {
        "batch": batch,
        "width": width,
        "head_size": head_size,
        "state": state.detach().flatten().tolist(),
        "x_norm": x_norm.detach().flatten().tolist(),
        "previous": previous.detach().flatten().tolist(),
        "mix_r": mix_r.detach().tolist(),
        "mix_k": mix_k.detach().tolist(),
        "mix_v": mix_v.detach().tolist(),
        "receptance_weight": receptance_weight.detach().flatten().tolist(),
        "key_weight": key_weight.detach().flatten().tolist(),
        "value_weight": value_weight.detach().flatten().tolist(),
        "k_k": k_k.detach().tolist(),
        "k_a": k_a.detach().tolist(),
        "a": a.detach().flatten().tolist(),
        "w": w.detach().flatten().tolist(),
        "grad_new_state": grad_new_state.flatten().tolist(),
        "grad_tmix": grad_tmix.flatten().tolist(),
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-rwkv-time-mix-core-") as temp_dir:
        case_path = Path(temp_dir) / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        base_command = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-rwkv-time-mix-core-step",
            "--",
            "--case",
            str(case_path),
        ]

        def run(extra_args: list[str]) -> dict:
            completed = subprocess.run(
                [*base_command, *extra_args],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Vulkan RWKV time-mix core parity runner failed:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            return json.loads(completed.stdout)

        result = run([])

        package_dir = Path(temp_dir) / "model-package"
        package_dir.mkdir()
        save_file(
            {
                "h_rnn.x_r": mix_r.detach().view(1, width).contiguous(),
                "h_rnn.x_k": mix_k.detach().view(1, width).contiguous(),
                "h_rnn.x_v": mix_v.detach().view(1, width).contiguous(),
                "h_rnn.receptance.weight": receptance_weight.detach().contiguous(),
                "h_rnn.key.weight": key_weight.detach().contiguous(),
                "h_rnn.value.weight": value_weight.detach().contiguous(),
                "h_rnn.k_k": k_k.detach().view(1, width).contiguous(),
                "h_rnn.k_a": k_a.detach().view(1, width).contiguous(),
            },
            str(package_dir / "model.safetensors"),
            metadata={"format": "pt", "layout": "pytorch-row-major"},
        )
        package_result = run(["--model-dir", str(package_dir), "--prefix", "h_rnn"])

    comparisons = {
        "new_state": (result["new_state"], new_state.detach()),
        "tmix": (result["tmix"], tmix.detach()),
        "scaled_k": (result["scaled_k"], scaled_k.detach()),
        "kk": (result["kk"], kk.detach()),
        "grad_state": (result["grad_state"], state.grad),
        "grad_x_norm": (result["grad_x_norm"], x_norm.grad),
        "grad_previous": (result["grad_previous"], previous.grad),
        "grad_a": (result["grad_a"], a.grad),
        "grad_w": (result["grad_w"], w.grad),
        "grad_mix_r": (result["grad_mix_r"], mix_r.grad),
        "grad_mix_k": (result["grad_mix_k"], mix_k.grad),
        "grad_mix_v": (result["grad_mix_v"], mix_v.grad),
        "grad_receptance_weight": (
            result["grad_receptance_weight"],
            receptance_weight.grad,
        ),
        "grad_key_weight": (result["grad_key_weight"], key_weight.grad),
        "grad_value_weight": (result["grad_value_weight"], value_weight.grad),
        "grad_k_k": (result["grad_k_k"], k_k.grad),
        "grad_k_a": (result["grad_k_a"], k_a.grad),
    }

    diffs = {}
    for name, (actual_values, expected) in comparisons.items():
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=2.5e-4, atol=2.5e-6)
        diffs[name] = (actual - expected).abs().max().item()

    for name in comparisons:
        inline = torch.tensor(result[name], dtype=torch.float32)
        packaged = torch.tensor(package_result[name], dtype=torch.float32)
        torch.testing.assert_close(packaged, inline, rtol=0.0, atol=0.0)

    print(f"device={result['device']} heads={result['heads']} head_size={head_size}")
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    print("Hierarchos safetensors -> Vulkan r/k/v core interchange: PASS")
    print("Hierarchos Vulkan composed RWKV r/k/v time-mix core parity: PASS")


if __name__ == "__main__":
    main()
