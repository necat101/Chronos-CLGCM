#!/usr/bin/env python3
"""Verify Vulkan Hierarchos manager/worker projection primitive parity."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def run_case(*, label: str, has_bias: bool, matrix_weight_decay: float) -> None:
    torch.manual_seed(20260816 if has_bias else 20260817)
    rows = 8
    input_dim = 19
    output_dim = 13
    steps = 3
    lr = 1.8e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8

    input_tensor = torch.randn(rows, input_dim, dtype=torch.float32, requires_grad=True)
    initial_weight = torch.randn(output_dim, input_dim, dtype=torch.float32) * 0.06
    initial_bias = torch.randn(output_dim, dtype=torch.float32) * 0.02 if has_bias else None
    grad_output = torch.randn(rows, output_dim, dtype=torch.float32) * 0.04

    weight = torch.nn.Parameter(initial_weight.clone())
    bias = torch.nn.Parameter(initial_bias.clone()) if initial_bias is not None else None
    groups = [{"params": [weight], "weight_decay": matrix_weight_decay}]
    if bias is not None:
        groups.append({"params": [bias], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )

    last_output = None
    last_input_grad = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        input_tensor.grad = None
        last_output = F.linear(input_tensor, weight, bias)
        last_output.backward(grad_output)
        last_input_grad = input_tensor.grad.detach().clone()
        optimizer.step()
    assert last_output is not None and last_input_grad is not None

    case = {
        "rows": rows,
        "steps": steps,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "input": input_tensor.detach().flatten().tolist(),
        "grad_output": grad_output.flatten().tolist(),
        "weight": initial_weight.flatten().tolist(),
        "bias": initial_bias.tolist() if initial_bias is not None else None,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "matrix_weight_decay": matrix_weight_decay,
    }
    with tempfile.TemporaryDirectory(prefix=f"hierarchos-vulkan-projection-{label}-") as temp_dir:
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
            "hierarchos-vulkan-projection-step",
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
                f"Vulkan {label} projection parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    actual_output = torch.tensor(result["output"], dtype=torch.float32).reshape_as(last_output)
    actual_input_grad = torch.tensor(result["input_grad"], dtype=torch.float32).reshape_as(input_tensor)
    actual_weight = torch.tensor(result["weight"], dtype=torch.float32).reshape_as(weight)
    torch.testing.assert_close(actual_output, last_output.detach(), rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual_input_grad, last_input_grad, rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual_weight, weight.detach(), rtol=8e-4, atol=8e-6)
    max_bias_diff = 0.0
    if bias is not None:
        actual_bias = torch.tensor(result["bias"], dtype=torch.float32)
        torch.testing.assert_close(actual_bias, bias.detach(), rtol=8e-4, atol=8e-6)
        max_bias_diff = (actual_bias - bias.detach()).abs().max().item()
    else:
        assert result["bias"] is None

    print(
        f"{label}: device={result['device']} "
        f"max_output_diff={(actual_output - last_output.detach()).abs().max().item():.9g} "
        f"max_input_grad_diff={(actual_input_grad - last_input_grad).abs().max().item():.9g} "
        f"max_weight_diff={(actual_weight - weight.detach()).abs().max().item():.9g} "
        f"max_bias_diff={max_bias_diff:.9g}"
    )


def main() -> None:
    # h_to_context/l_input_proj/l_to_out/in_proj/router-style affine projection.
    run_case(label="affine-manager-worker", has_bias=True, matrix_weight_decay=0.1)
    # l_feedback_proj/context_drift_proj/qproj-style biasless projection.
    run_case(label="biasless-manager-worker", has_bias=False, matrix_weight_decay=0.1)
    # val_proj is a project-specific exception: a matrix explicitly excluded from decay.
    run_case(label="val-proj-no-decay", has_bias=False, matrix_weight_decay=0.0)
    print("Hierarchos Vulkan projection PyTorch parity: PASS")


if __name__ == "__main__":
    main()
