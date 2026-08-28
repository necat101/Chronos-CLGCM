#!/usr/bin/env python3
"""Verify Vulkan RWKV a/w/g low-rank branches against PyTorch autograd."""

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
    width = 128
    w_rank = 32
    a_rank = 32
    g_rank = 64

    x_norm = (torch.randn(batch, width, dtype=torch.float32) * 0.35).requires_grad_()
    previous = (torch.randn(batch, width, dtype=torch.float32) * 0.3).requires_grad_()
    mix_w = torch.rand(width, dtype=torch.float32).requires_grad_()
    mix_a = torch.rand(width, dtype=torch.float32).requires_grad_()
    mix_g = torch.rand(width, dtype=torch.float32).requires_grad_()

    def matrix(rows: int, cols: int, scale: float = 0.16) -> torch.Tensor:
        return (torch.randn(rows, cols, dtype=torch.float32) * (scale / math.sqrt(rows))).requires_grad_()

    w0 = (-2.5 + torch.randn(1, width, dtype=torch.float32) * 0.4).requires_grad_()
    w1 = matrix(width, w_rank)
    w2 = matrix(w_rank, width)
    a0 = (torch.randn(1, width, dtype=torch.float32) * 0.25).requires_grad_()
    a1 = matrix(width, a_rank)
    a2 = matrix(a_rank, width)
    g1 = matrix(width, g_rank)
    g2 = matrix(g_rank, width)

    delta = previous - x_norm
    xw = x_norm + delta * mix_w
    xa = x_norm + delta * mix_a
    xg = x_norm + delta * mix_g
    w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g2_input = torch.sigmoid(xg @ g1)
    g = g2_input @ g2

    grad_a = torch.randn_like(a) * 0.04
    grad_w = torch.randn_like(w) * 0.035
    grad_g = torch.randn_like(g) * 0.045
    objective = (a * grad_a).sum() + (w * grad_w).sum() + (g * grad_g).sum()
    objective.backward()

    case = {
        "batch": batch,
        "width": width,
        "w_rank": w_rank,
        "a_rank": a_rank,
        "g_rank": g_rank,
        "x_norm": x_norm.detach().flatten().tolist(),
        "previous": previous.detach().flatten().tolist(),
        "mix_w": mix_w.detach().tolist(),
        "mix_a": mix_a.detach().tolist(),
        "mix_g": mix_g.detach().tolist(),
        "w0": w0.detach().flatten().tolist(),
        "w1": w1.detach().flatten().tolist(),
        "w2": w2.detach().flatten().tolist(),
        "a0": a0.detach().flatten().tolist(),
        "a1": a1.detach().flatten().tolist(),
        "a2": a2.detach().flatten().tolist(),
        "g1": g1.detach().flatten().tolist(),
        "g2": g2.detach().flatten().tolist(),
        "grad_a": grad_a.flatten().tolist(),
        "grad_w": grad_w.flatten().tolist(),
        "grad_g": grad_g.flatten().tolist(),
        "g2_input": g2_input.detach().flatten().tolist(),
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-rwkv-low-rank-") as temp_dir:
        temp = Path(temp_dir)
        case_path = temp / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        base_command = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-rwkv-low-rank-step",
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
                    "Vulkan RWKV low-rank parity runner failed:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            return json.loads(completed.stdout)

        result = run([])
        dw_diagnostic = run(["--native-fp16-dw-diagnostic"])

        package_dir = temp / "model-package"
        package_dir.mkdir()
        save_file(
            {
                "h_rnn.x_w": mix_w.detach().view(1, width).contiguous(),
                "h_rnn.x_a": mix_a.detach().view(1, width).contiguous(),
                "h_rnn.x_g": mix_g.detach().view(1, width).contiguous(),
                "h_rnn.w0": w0.detach().contiguous(),
                "h_rnn.w1": w1.detach().contiguous(),
                "h_rnn.w2": w2.detach().contiguous(),
                "h_rnn.a0": a0.detach().contiguous(),
                "h_rnn.a1": a1.detach().contiguous(),
                "h_rnn.a2": a2.detach().contiguous(),
                "h_rnn.g1": g1.detach().contiguous(),
                "h_rnn.g2": g2.detach().contiguous(),
            },
            str(package_dir / "model.safetensors"),
            metadata={"format": "pt", "layout": "pytorch-row-major"},
        )
        package_result = run(["--model-dir", str(package_dir), "--prefix", "h_rnn"])

    comparisons = {
        "a": (result["a"], a.detach()),
        "w": (result["w"], w.detach()),
        "g": (result["g"], g.detach()),
        "grad_x_norm": (result["grad_x_norm"], x_norm.grad),
        "grad_previous": (result["grad_previous"], previous.grad),
        "grad_mix_w": (result["grad_mix_w"], mix_w.grad),
        "grad_mix_a": (result["grad_mix_a"], mix_a.grad),
        "grad_mix_g": (result["grad_mix_g"], mix_g.grad),
        "grad_w0": (result["grad_w0"], w0.grad),
        "grad_w1": (result["grad_w1"], w1.grad),
        "grad_w2": (result["grad_w2"], w2.grad),
        "grad_a0": (result["grad_a0"], a0.grad),
        "grad_a1": (result["grad_a1"], a1.grad),
        "grad_a2": (result["grad_a2"], a2.grad),
        "grad_g1": (result["grad_g1"], g1.grad),
        "grad_g2": (result["grad_g2"], g2.grad),
    }

    diffs = {}
    for name, (actual_values, expected) in comparisons.items():
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=3.0e-4, atol=3.0e-6)
        diffs[name] = (actual - expected).abs().max().item()

    for name in comparisons:
        inline = torch.tensor(result[name], dtype=torch.float32)
        packaged = torch.tensor(package_result[name], dtype=torch.float32)
        torch.testing.assert_close(packaged, inline, rtol=0.0, atol=0.0)

    diagnostic_input = g2_input.detach().to(dtype=torch.float32)
    diagnostic_grad = grad_g.detach().to(dtype=torch.float32)
    expected_portable = diagnostic_input.transpose(0, 1) @ diagnostic_grad

    # Mirror the shader literally: each source is rounded to IEEE FP16, the
    # product is rounded in FP16, then widened and accumulated serially in FP32.
    input_half = diagnostic_input.to(dtype=torch.float16)
    grad_half = diagnostic_grad.to(dtype=torch.float16)
    expected_native = torch.zeros(g_rank, width, dtype=torch.float32)
    expected_widened_product = torch.zeros(g_rank, width, dtype=torch.float32)
    for row in range(batch):
        product_half = input_half[row].unsqueeze(1) * grad_half[row].unsqueeze(0)
        expected_native += product_half.to(dtype=torch.float32)
        expected_widened_product += input_half[row].to(dtype=torch.float32).unsqueeze(
            1
        ) * grad_half[row].to(dtype=torch.float32).unsqueeze(0)

    diagnostic_native = torch.tensor(
        dw_diagnostic["native_fp16_grad"], dtype=torch.float32
    ).reshape(g_rank, width)
    diagnostic_native_vs_portable = (
        diagnostic_native - expected_portable
    ).abs().max().item()
    diagnostic_native_vs_reference = (
        diagnostic_native - expected_native
    ).abs().max().item()
    diagnostic_native_vs_widened_product = (
        diagnostic_native - expected_widened_product
    ).abs().max().item()
    if diagnostic_native_vs_reference == 0.0:
        diagnostic_semantics = "serial-fp32-sum-of-fp16-products"
    elif diagnostic_native_vs_widened_product == 0.0:
        diagnostic_semantics = "serial-fp32-sum-of-fp32-products-of-fp16-inputs"
    else:
        diagnostic_semantics = "device-specific-other"
        raise AssertionError(
            "native FP16 dW matched neither modeled arithmetic contract: "
            f"half_product_diff={diagnostic_native_vs_reference:.9g} "
            f"widened_input_diff={diagnostic_native_vs_widened_product:.9g}"
        )

    assert (result["w_rank"], result["a_rank"], result["g_rank"]) == (
        w_rank,
        a_rank,
        g_rank,
    )
    assert (
        package_result["w_rank"],
        package_result["a_rank"],
        package_result["g_rank"],
    ) == (w_rank, a_rank, g_rank)

    print(
        f"device={result['device']} width={width} "
        f"ranks=w{w_rank}/a{a_rank}/g{g_rank}"
    )
    print(
        "native_fp16_dw_diagnostic=g2 "
        f"semantics={diagnostic_semantics} "
        f"max_abs_native_vs_portable={diagnostic_native_vs_portable:.9g} "
        f"max_abs_native_vs_serial_half_reference={diagnostic_native_vs_reference:.9g} "
        "max_abs_native_vs_fp32_product_of_half_inputs="
        f"{diagnostic_native_vs_widened_product:.9g}"
    )
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    print("Hierarchos safetensors -> Vulkan a/w/g low-rank interchange: PASS")
    print("Hierarchos Vulkan RWKV a/w/g low-rank autograd parity: PASS")


if __name__ == "__main__":
    main()
