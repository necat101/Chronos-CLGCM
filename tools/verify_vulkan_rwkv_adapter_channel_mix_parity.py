#!/usr/bin/env python3
"""Verify single-submit SharedTokenAdapter -> RWKV channel-mix parity."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(20260816)
    batch = 5
    input_dim = 14
    width = 16
    hidden_width = width * 4
    rank = 9
    key_clamp = 12.0
    deepembed_clamp = 4.0

    token_features = torch.randn(batch, input_dim, dtype=torch.float32, requires_grad=True)
    x = torch.randn(batch, width, dtype=torch.float32, requires_grad=True)
    previous = torch.randn(batch, width, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(batch, width, dtype=torch.float32) * 0.08

    adapter_down = torch.nn.Parameter(torch.randn(rank, input_dim, dtype=torch.float32) * 0.07)
    adapter_up = torch.nn.Parameter(torch.randn(hidden_width, rank, dtype=torch.float32) * 0.035)
    adapter_bias = torch.nn.Parameter(torch.ones(hidden_width, dtype=torch.float32))
    layer_norm_weight = torch.nn.Parameter(torch.randn(width, dtype=torch.float32) * 0.05 + 1.0)
    layer_norm_bias = torch.nn.Parameter(torch.randn(width, dtype=torch.float32) * 0.03)
    mix_k = torch.nn.Parameter(torch.randn(width, dtype=torch.float32) * 0.15)
    key_weight = torch.nn.Parameter(torch.randn(hidden_width, width, dtype=torch.float32) * 0.045)
    value_weight = torch.nn.Parameter(torch.randn(width, hidden_width, dtype=torch.float32) * 0.025)

    token_norm = F.layer_norm(token_features, (input_dim,), eps=1.0e-5)
    deepembed = adapter_bias + F.linear(F.silu(F.linear(token_norm, adapter_down)), adapter_up)
    deepembed.retain_grad()

    x_norm = F.layer_norm(
        x,
        (width,),
        weight=layer_norm_weight,
        bias=layer_norm_bias,
        eps=1.0e-5,
    )
    mixed = x_norm + (previous - x_norm) * mix_k
    cm_key = F.linear(mixed, key_weight)
    cm_key = torch.clamp(cm_key, -key_clamp, key_clamp)
    ffn = torch.square(torch.relu(cm_key))
    deepembed_used = torch.clamp(deepembed, -deepembed_clamp, deepembed_clamp)
    ffn = ffn * deepembed_used
    ffn = torch.clamp(
        ffn,
        -(key_clamp * key_clamp * deepembed_clamp),
        key_clamp * key_clamp * deepembed_clamp,
    )
    output = x + F.linear(ffn, value_weight)
    output.backward(grad_output)

    case = {
        "batch": batch,
        "input_dim": input_dim,
        "width": width,
        "rank": rank,
        "token_features": token_features.detach().flatten().tolist(),
        "x": x.detach().flatten().tolist(),
        "previous": previous.detach().flatten().tolist(),
        "grad_output": grad_output.flatten().tolist(),
        "adapter_down_weight": adapter_down.detach().flatten().tolist(),
        "adapter_up_weight": adapter_up.detach().flatten().tolist(),
        "adapter_bias": adapter_bias.detach().tolist(),
        "layer_norm_weight": layer_norm_weight.detach().tolist(),
        "layer_norm_bias": layer_norm_bias.detach().tolist(),
        "mix_k": mix_k.detach().tolist(),
        "key_weight": key_weight.detach().flatten().tolist(),
        "value_weight": value_weight.detach().flatten().tolist(),
        "key_clamp": key_clamp,
        "deepembed_clamp": deepembed_clamp,
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-adapter-channel-") as temp_dir:
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
            "hierarchos-vulkan-rwkv-adapter-channel-mix-step",
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
                "Vulkan adapter/channel-mix parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    comparisons = {
        "output": (result["output"], output.detach()),
        "grad_x": (result["grad_x"], x.grad),
        "grad_previous": (result["grad_previous"], previous.grad),
        "grad_deepembed": (result["grad_deepembed"], deepembed.grad),
        "token_feature_grad": (result["token_feature_grad"], token_features.grad),
        "grad_mix_k": (result["grad_mix_k"], mix_k.grad),
        "grad_key_weight": (result["grad_key_weight"], key_weight.grad),
        "grad_value_weight": (result["grad_value_weight"], value_weight.grad),
        "grad_layer_norm_weight": (result["grad_layer_norm_weight"], layer_norm_weight.grad),
        "grad_layer_norm_bias": (result["grad_layer_norm_bias"], layer_norm_bias.grad),
    }
    max_diffs: dict[str, float] = {}
    for name, (actual_values, expected) in comparisons.items():
        assert expected is not None
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=2.5e-3, atol=1.5e-5)
        max_diffs[name] = (actual - expected).abs().max().item()

    print(
        f"device={result['device']} "
        + " ".join(f"max_{name}_diff={diff:.9g}" for name, diff in max_diffs.items())
    )
    print("Hierarchos Vulkan SharedTokenAdapter -> channel-mix PyTorch parity: PASS")


if __name__ == "__main__":
    main()
