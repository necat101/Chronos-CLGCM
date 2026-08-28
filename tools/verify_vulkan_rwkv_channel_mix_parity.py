#!/usr/bin/env python3
"""Verify Vulkan RWKV channel-mix ReLU2/DeepEmbed against PyTorch autograd."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if args.width <= 0 or args.batch <= 0:
        parser.error("--width and --batch must be positive")

    torch.manual_seed(20260812)
    batch = args.batch
    width = args.width
    hidden = width * 4
    key_clamp = 12.0
    deepembed_clamp = 4.0

    x = (torch.randn(batch, width, dtype=torch.float32) * 0.45).requires_grad_()
    previous = (torch.randn(batch, width, dtype=torch.float32) * 0.35).requires_grad_()
    deepembed_values = 0.9 + torch.randn(batch, hidden, dtype=torch.float32) * 6.0
    deepembed_values.view(-1)[:8] = torch.tensor(
        [-8.0, -4.0, -3.9, 0.0, 3.9, 4.0, 8.0, 12.0],
        dtype=torch.float32,
    )
    deepembed = deepembed_values.requires_grad_()
    mix_k = torch.rand(width, dtype=torch.float32).requires_grad_()
    ln_weight = (0.9 + torch.rand(width, dtype=torch.float32) * 0.2).requires_grad_()
    ln_bias = (torch.randn(width, dtype=torch.float32) * 0.07).requires_grad_()

    def matrix(rows: int, cols: int, scale: float) -> torch.Tensor:
        return (
            torch.randn(rows, cols, dtype=torch.float32) * (scale / math.sqrt(cols))
        ).requires_grad_()

    # Production uses a ±12 key clamp.  A normal initialization-scale parity
    # fixture never reaches it, so use a deliberately hot projection here and
    # certify both saturated and unsaturated backward lanes.
    key_weight = matrix(hidden, width, 20.0)
    value_weight = matrix(width, hidden, 0.08)

    normalized = F.layer_norm(x, (width,), ln_weight, ln_bias, eps=1.0e-5)
    mixed = normalized + (previous - normalized) * mix_k
    cm_key = F.linear(mixed, key_weight)
    raw_cm_key = cm_key
    cm_key = torch.clamp(cm_key, -key_clamp, key_clamp)
    ffn = torch.square(torch.relu(cm_key))
    deep_clamped = torch.clamp(deepembed, -deepembed_clamp, deepembed_clamp)
    ffn = ffn * deep_clamped
    ffn_limit = key_clamp * key_clamp * deepembed_clamp
    ffn = torch.clamp(ffn, -ffn_limit, ffn_limit)
    output = x + F.linear(ffn, value_weight)

    grad_output = torch.randn_like(output) * 0.05
    (output * grad_output).sum().backward()
    key_saturated = int((raw_cm_key.detach().abs() > key_clamp).sum().item())
    deep_saturated = int((deepembed.detach().abs() > deepembed_clamp).sum().item())
    assert key_saturated > 0, "channel-mix fixture failed to exercise key clamping"
    assert deep_saturated > 0, "channel-mix fixture failed to exercise deep-embed clamping"

    case = {
        "batch": batch,
        "width": width,
        "x": x.detach().flatten().tolist(),
        "previous": previous.detach().flatten().tolist(),
        "deepembed": deepembed.detach().flatten().tolist(),
        "grad_output": grad_output.flatten().tolist(),
    }
    tensors = {
        "h_rnn.ln2.weight": ln_weight.detach().contiguous(),
        "h_rnn.ln2.bias": ln_bias.detach().contiguous(),
        "h_rnn.x_k_cm": mix_k.detach().view(1, width).contiguous(),
        "h_rnn.key_cm.weight": key_weight.detach().contiguous(),
        "h_rnn.value_cm.weight": value_weight.detach().contiguous(),
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-rwkv-channel-mix-") as temp_dir:
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
                "hierarchos-vulkan-rwkv-channel-mix-step",
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
                "Vulkan RWKV channel-mix parity runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    comparisons = {
        "output": (result["output"], output.detach()),
        "grad_x": (result["grad_x"], x.grad),
        "grad_previous": (result["grad_previous"], previous.grad),
        "grad_deepembed": (result["grad_deepembed"], deepembed.grad),
        "grad_mix_k": (result["grad_mix_k"], mix_k.grad),
        "grad_key_weight": (result["grad_key_weight"], key_weight.grad),
        "grad_value_weight": (result["grad_value_weight"], value_weight.grad),
        "grad_layer_norm_weight": (result["grad_layer_norm_weight"], ln_weight.grad),
        "grad_layer_norm_bias": (result["grad_layer_norm_bias"], ln_bias.grad),
    }
    diffs: dict[str, float] = {}
    materialized: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (actual_values, expected) in comparisons.items():
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        diffs[name] = (actual - expected).abs().max().item()
        materialized[name] = (actual, expected)

    print(
        f"device={result['device']} width={width} hidden={hidden} "
        f"key_saturated={key_saturated} deep_saturated={deep_saturated}"
    )
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    for actual, expected in materialized.values():
        torch.testing.assert_close(actual, expected, rtol=5.0e-4, atol=5.0e-6)
    print("Hierarchos Vulkan RWKV channel-mix ReLU2/DeepEmbed parity: PASS")


if __name__ == "__main__":
    main()
