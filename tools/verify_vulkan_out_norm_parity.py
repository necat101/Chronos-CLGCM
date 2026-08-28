#!/usr/bin/env python3
"""Verify Vulkan out_norm + tied LM-head training against PyTorch AdamW."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--activation-clamp", type=float, default=0.35)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not math.isfinite(args.activation_clamp) or args.activation_clamp <= 0.0:
        parser.error("--activation-clamp must be finite and positive")

    torch.manual_seed(20260812)
    rows = 7
    context_dim = 16
    vocab_size = 29
    steps = args.steps
    lr = 2.0e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    weight_decay = 0.1

    hidden = torch.randn(rows, context_dim, dtype=torch.float32, requires_grad=True)
    initial_lm = torch.randn(vocab_size, context_dim, dtype=torch.float32) * 0.06
    initial_norm_weight = 0.9 + torch.rand(context_dim, dtype=torch.float32) * 0.2
    initial_norm_bias = torch.randn(context_dim, dtype=torch.float32) * 0.02
    targets = torch.tensor([1, 7, 3, 19, 4, 11, 23], dtype=torch.long)

    lm_weight = torch.nn.Parameter(initial_lm.clone())
    norm_weight = torch.nn.Parameter(initial_norm_weight.clone())
    norm_bias = torch.nn.Parameter(initial_norm_bias.clone())
    optimizer = torch.optim.AdamW(
        [
            {"params": [lm_weight], "weight_decay": weight_decay},
            {"params": [norm_weight, norm_bias], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )

    loss = None
    last_input_grad = None
    clamp_saturated = False
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        hidden.grad = None
        raw_normalized = F.layer_norm(
            hidden,
            (context_dim,),
            norm_weight,
            norm_bias,
            eps=1.0e-5,
        )
        clamp_saturated = clamp_saturated or bool(
            (raw_normalized.detach().abs() > args.activation_clamp).any().item()
        )
        normalized = torch.where(
            torch.isfinite(raw_normalized),
            torch.clamp(
                raw_normalized,
                min=-args.activation_clamp,
                max=args.activation_clamp,
            ),
            raw_normalized,
        )
        loss = F.cross_entropy(normalized @ lm_weight.t(), targets, reduction="mean")
        loss.backward()
        last_input_grad = hidden.grad.detach().clone()
        optimizer.step()
    assert loss is not None and last_input_grad is not None
    if not clamp_saturated:
        raise AssertionError("out_norm parity fixture did not activate the clamp")

    case = {
        "rows": rows,
        "steps": steps,
        "context_dim": context_dim,
        "vocab_size": vocab_size,
        "hidden": hidden.detach().flatten().tolist(),
        "targets": targets.tolist(),
        "lm_weight": initial_lm.flatten().tolist(),
        "norm_weight": initial_norm_weight.tolist(),
        "norm_bias": initial_norm_bias.tolist(),
        "activation_clamp": args.activation_clamp,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-out-norm-") as temp_dir:
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
            "hierarchos-vulkan-out-norm-step",
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
                "Vulkan out_norm parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    if abs(float(result["activation_clamp"]) - args.activation_clamp) > 1.0e-7:
        raise AssertionError(
            "Vulkan out_norm activation clamp mismatch: "
            f"rust={result['activation_clamp']} torch={args.activation_clamp}"
        )

    actual_lm = torch.tensor(result["lm_weight"], dtype=torch.float32).reshape_as(lm_weight)
    actual_norm_weight = torch.tensor(result["norm_weight"], dtype=torch.float32)
    actual_norm_bias = torch.tensor(result["norm_bias"], dtype=torch.float32)
    actual_input_grad = torch.tensor(result["input_grad"], dtype=torch.float32).reshape_as(hidden)

    torch.testing.assert_close(
        torch.tensor(result["loss"], dtype=torch.float32),
        loss.detach(),
        rtol=4e-5,
        atol=4e-6,
    )
    torch.testing.assert_close(actual_lm, lm_weight.detach(), rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual_norm_weight, norm_weight.detach(), rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual_norm_bias, norm_bias.detach(), rtol=5e-4, atol=5e-6)
    torch.testing.assert_close(actual_input_grad, last_input_grad, rtol=8e-4, atol=8e-6)

    print(f"device={result['device']}")
    print(f"activation_clamp={result['activation_clamp']:.9g} saturated={clamp_saturated}")
    print(f"pytorch_loss={loss.item():.9g} vulkan_loss={result['loss']:.9g}")
    print(f"max_abs_lm_diff={(actual_lm - lm_weight.detach()).abs().max().item():.9g}")
    print(
        "max_abs_norm_weight_diff="
        f"{(actual_norm_weight - norm_weight.detach()).abs().max().item():.9g}"
    )
    print(
        "max_abs_norm_bias_diff="
        f"{(actual_norm_bias - norm_bias.detach()).abs().max().item():.9g}"
    )
    print(
        "max_abs_input_grad_diff="
        f"{(actual_input_grad - last_input_grad).abs().max().item():.9g}"
    )
    print("Hierarchos Vulkan out_norm + LM-head PyTorch parity: PASS")


if __name__ == "__main__":
    main()
