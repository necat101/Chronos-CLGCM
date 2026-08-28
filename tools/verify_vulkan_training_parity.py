#!/usr/bin/env python3
"""Verify repeated Hierarchos LM-head training steps match PyTorch AdamW."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(20260812)
    rows = 5
    context_dim = 12
    vocab_size = 23
    lr = 3.0e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    weight_decay = 0.1
    steps = 3

    hidden = torch.randn(rows, context_dim, dtype=torch.float32)
    initial_weight = torch.randn(vocab_size, context_dim, dtype=torch.float32) * 0.08
    targets = torch.tensor([1, 7, 3, 19, 4], dtype=torch.long)

    reference_weight = torch.nn.Parameter(initial_weight.clone())
    optimizer = torch.optim.AdamW(
        [reference_weight],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(hidden @ reference_weight.t(), targets, reduction="mean")
        loss.backward()
        optimizer.step()
    assert loss is not None

    case = {
        "rows": rows,
        "steps": steps,
        "context_dim": context_dim,
        "vocab_size": vocab_size,
        "hidden": hidden.flatten().tolist(),
        "targets": targets.tolist(),
        "weight": initial_weight.flatten().tolist(),
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-parity-") as temp_dir:
        temp = Path(temp_dir)
        case_path = temp / "case.json"
        output_path = temp / "vulkan-head.safetensors"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        command = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-head-step",
            "--",
            "--case",
            str(case_path),
            "--output-safetensors",
            str(output_path),
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
                "Vulkan parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)
        actual_weight = torch.tensor(result["weights"], dtype=torch.float32).reshape(
            vocab_size, context_dim
        )
        vulkan_file = load_file(str(output_path))["lm_head.weight"]

    torch.testing.assert_close(
        torch.tensor(result["loss"], dtype=torch.float32),
        loss.detach(),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(actual_weight, reference_weight.detach(), rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(vulkan_file, actual_weight, rtol=0.0, atol=0.0)
    max_abs = (actual_weight - reference_weight.detach()).abs().max().item()
    print(f"device={result['device']}")
    print(f"pytorch_loss={loss.item():.9g} vulkan_loss={result['loss']:.9g}")
    print(f"max_abs_weight_diff={max_abs:.9g}")
    print("Hierarchos Vulkan LM-head PyTorch parity: PASS")


if __name__ == "__main__":
    main()
