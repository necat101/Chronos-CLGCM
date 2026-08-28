#!/usr/bin/env python3
"""Verify coherent-v9 SharedTokenAdapter Vulkan forward/backward parity."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def run_case(*, label: str, matrix_weight_decay: float, output_bias: float) -> None:
    torch.manual_seed(20260814 if matrix_weight_decay == 0.0 else 20260815)
    rows = 6
    input_dim = 18
    rank = 7
    output_dim = 22
    steps = 3
    lr = 1.5e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8

    input_tensor = torch.randn(rows, input_dim, dtype=torch.float32, requires_grad=True)
    initial_down = torch.randn(rank, input_dim, dtype=torch.float32) * 0.08
    initial_up = torch.randn(output_dim, rank, dtype=torch.float32) * 0.04
    initial_bias = torch.full((output_dim,), output_bias, dtype=torch.float32)
    grad_output = torch.randn(rows, output_dim, dtype=torch.float32) * 0.05

    down = torch.nn.Parameter(initial_down.clone())
    up = torch.nn.Parameter(initial_up.clone())
    bias = torch.nn.Parameter(initial_bias.clone())
    optimizer = torch.optim.AdamW(
        [
            {"params": [down, up], "weight_decay": matrix_weight_decay},
            {"params": [bias], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )

    last_output = None
    last_input_grad = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        input_tensor.grad = None
        normalized = F.layer_norm(input_tensor, (input_dim,), eps=1.0e-5)
        last_output = bias + F.linear(F.silu(F.linear(normalized, down)), up)
        last_output.backward(grad_output)
        last_input_grad = input_tensor.grad.detach().clone()
        optimizer.step()
    assert last_output is not None and last_input_grad is not None

    case = {
        "rows": rows,
        "steps": steps,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": rank,
        "input": input_tensor.detach().flatten().tolist(),
        "grad_output": grad_output.flatten().tolist(),
        "down_weight": initial_down.flatten().tolist(),
        "up_weight": initial_up.flatten().tolist(),
        "bias": initial_bias.tolist(),
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "matrix_weight_decay": matrix_weight_decay,
    }

    with tempfile.TemporaryDirectory(prefix=f"hierarchos-vulkan-adapter-{label}-") as temp_dir:
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
            "hierarchos-vulkan-adapter-step",
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
                f"Vulkan {label} adapter parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    actual_output = torch.tensor(result["output"], dtype=torch.float32).reshape_as(last_output)
    actual_input_grad = torch.tensor(result["input_grad"], dtype=torch.float32).reshape_as(input_tensor)
    actual_down = torch.tensor(result["down_weight"], dtype=torch.float32).reshape_as(down)
    actual_up = torch.tensor(result["up_weight"], dtype=torch.float32).reshape_as(up)
    actual_bias = torch.tensor(result["bias"], dtype=torch.float32)

    torch.testing.assert_close(actual_output, last_output.detach(), rtol=6e-4, atol=6e-6)
    torch.testing.assert_close(actual_input_grad, last_input_grad, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(actual_down, down.detach(), rtol=8e-4, atol=8e-6)
    torch.testing.assert_close(actual_up, up.detach(), rtol=8e-4, atol=8e-6)
    torch.testing.assert_close(actual_bias, bias.detach(), rtol=8e-4, atol=8e-6)

    print(
        f"{label}: device={result['device']} "
        f"max_output_diff={(actual_output - last_output.detach()).abs().max().item():.9g} "
        f"max_input_grad_diff={(actual_input_grad - last_input_grad).abs().max().item():.9g} "
        f"max_down_diff={(actual_down - down.detach()).abs().max().item():.9g} "
        f"max_up_diff={(actual_up - up.detach()).abs().max().item():.9g} "
        f"max_bias_diff={(actual_bias - bias.detach()).abs().max().item():.9g}"
    )


def main() -> None:
    # DeepEmbed adapters are explicitly excluded from AdamW decay in the Python trainer.
    run_case(label="deepembed-no-decay", matrix_weight_decay=0.0, output_bias=1.0)
    # ROSA adapter matrices follow the normal matrix-decay group; its bias is a vector.
    run_case(label="rosa-matrix-decay", matrix_weight_decay=0.1, output_bias=0.0)
    print("Hierarchos Vulkan SharedTokenAdapter PyTorch parity: PASS")


if __name__ == "__main__":
    main()
