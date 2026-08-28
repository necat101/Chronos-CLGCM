#!/usr/bin/env python3
"""Verify the Vulkan token front-end against PyTorch and native Rust inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _reference(
    model: HierarchosCore,
    token_ids: torch.Tensor,
    token_residual: torch.Tensor,
    gated_ltm_values: torch.Tensor,
    grad_enc: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    model.zero_grad(set_to_none=True)
    residual = token_residual.detach().clone().requires_grad_(True)
    ltm = gated_ltm_values.detach().clone().requires_grad_(True)
    token_features = F.embedding(token_ids, model.lm_head.weight)
    token_features.retain_grad()
    persistent_batch = model.persistent.unsqueeze(0).expand(token_ids.numel(), -1)
    mac_in = torch.cat([token_features + residual, persistent_batch, ltm], dim=-1)
    raw_enc = F.gelu(model.in_proj(mac_in))
    enc = raw_enc
    enc = torch.where(torch.isfinite(enc), enc.clamp(-30.0, 30.0), enc)
    enc.backward(grad_enc)
    return (
        token_features.detach(),
        raw_enc.detach(),
        enc.detach(),
        token_features.grad.detach(),
        ltm.grad.detach(),
        model.persistent.grad.detach(),
        model.in_proj.weight.grad.detach(),
        model.in_proj.bias.grad.detach(),
        model.lm_head.weight.grad.detach(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clamp-stress",
        action="store_true",
        help=(
            "force a mixed saturated/in-range GELU->finite_clamp(30) fixture so "
            "the Vulkan backward mask is qualified, not only the identity path"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(20260814)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).eval()

    if args.clamp_stress:
        # Keep alternating output coordinates in two regimes: the +40 bias
        # guarantees GELU is above the +30 clamp, while the small bias leaves
        # the neighboring row on the ordinary differentiable path.  Retaining
        # the initialized weights keeps non-zero upstream gradients alive, so
        # this catches an implementation that accidentally masks the whole
        # projection instead of only saturated coordinates.
        with torch.no_grad():
            model.in_proj.bias[0::2].fill_(40.0)
            model.in_proj.bias[1::2].fill_(0.05)

    token_ids = torch.tensor([2, 5, 2, 7, 5], dtype=torch.long)
    rows = token_ids.numel()
    ltm_dim = config.ltm_topk * config.ltm_val_dim
    token_residual = torch.randn(rows, config.context_dim, dtype=torch.float32) * 0.03
    gated_ltm_values = torch.randn(rows, ltm_dim, dtype=torch.float32) * 0.04
    grad_enc = torch.randn(rows, config.context_dim, dtype=torch.float32) * 0.05

    (
        expected_token_features,
        expected_raw_enc,
        expected_enc,
        expected_grad_token,
        expected_grad_ltm,
        expected_grad_persistent,
        expected_grad_in_weight,
        expected_grad_in_bias,
        expected_grad_lm_head,
    ) = _reference(model, token_ids, token_residual, gated_ltm_values, grad_enc)

    if args.clamp_stress:
        saturated = expected_raw_enc > 30.0
        in_range = expected_raw_enc.abs() < 30.0
        if not bool(saturated.any().item()) or not bool(in_range.any().item()):
            raise AssertionError(
                "token front-end clamp-stress fixture must contain both saturated "
                "and in-range GELU outputs"
            )
        saturated_columns = saturated.all(dim=0)
        if not bool(saturated_columns.any().item()):
            raise AssertionError(
                "token front-end clamp-stress fixture no longer has an output "
                "column saturated for every row"
            )
        if not torch.equal(
            expected_grad_in_bias[saturated_columns],
            torch.zeros_like(expected_grad_in_bias[saturated_columns]),
        ):
            raise AssertionError(
                "PyTorch clamp-stress reference did not mask saturated in_proj.bias gradients"
            )
        if not bool((expected_grad_in_bias[~saturated_columns].abs() > 0).any().item()):
            raise AssertionError(
                "token front-end clamp-stress fixture accidentally removed the in-range gradient path"
            )

    case = {
        "token_ids": token_ids.tolist(),
        "token_residual": token_residual.flatten().tolist(),
        "gated_ltm_values": gated_ltm_values.flatten().tolist(),
        "grad_enc": grad_enc.flatten().tolist(),
    }
    native_case = {
        "token_ids": case["token_ids"],
        "token_residual": case["token_residual"],
        "gated_ltm_values": case["gated_ltm_values"],
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-token-frontend-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        case_path = temp / "case.json"
        native_case_path = temp / "native-case.json"
        export_model(model, config, model_dir)
        case_path.write_text(json.dumps(case), encoding="utf-8")
        native_case_path.write_text(json.dumps(native_case), encoding="utf-8")

        vulkan_result = json.loads(
            _run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--release",
                    "--manifest-path",
                    str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                    "--bin",
                    "hierarchos-vulkan-token-frontend-parity",
                    "--",
                    "--model",
                    str(model_dir),
                    "--case",
                    str(case_path),
                ]
            ).stdout
        )
        native_result = json.loads(
            _run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--release",
                    "--manifest-path",
                    str(ROOT / "hierarchos-inference" / "Cargo.toml"),
                    "--bin",
                    "frontend-probe",
                    "--",
                    "--model",
                    str(model_dir),
                    "--case",
                    str(native_case_path),
                ]
            ).stdout
        )

    actual_token_features = torch.tensor(vulkan_result["token_features"]).reshape_as(
        expected_token_features
    )
    actual_enc = torch.tensor(vulkan_result["enc"]).reshape_as(expected_enc)
    actual_grad_token = torch.tensor(vulkan_result["grad_token_features"]).reshape_as(
        expected_grad_token
    )
    actual_grad_ltm = torch.tensor(vulkan_result["grad_gated_ltm_values"]).reshape_as(
        expected_grad_ltm
    )
    actual_grad_persistent = torch.tensor(vulkan_result["grad_persistent"])
    actual_grad_in_weight = torch.tensor(vulkan_result["grad_in_proj_weight"]).reshape_as(
        expected_grad_in_weight
    )
    actual_grad_in_bias = torch.tensor(vulkan_result["grad_in_proj_bias"])
    actual_grad_lm_head = torch.tensor(vulkan_result["grad_lm_head_weight"]).reshape_as(
        expected_grad_lm_head
    )
    native_enc = torch.tensor(native_result["enc"]).reshape_as(expected_enc)

    torch.testing.assert_close(actual_token_features, expected_token_features, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_enc, expected_enc, rtol=4e-4, atol=4e-6)
    torch.testing.assert_close(actual_grad_token, expected_grad_token, rtol=6e-4, atol=6e-6)
    torch.testing.assert_close(actual_grad_ltm, expected_grad_ltm, rtol=6e-4, atol=6e-6)
    torch.testing.assert_close(
        actual_grad_persistent, expected_grad_persistent, rtol=8e-4, atol=8e-6
    )
    torch.testing.assert_close(
        actual_grad_in_weight, expected_grad_in_weight, rtol=8e-4, atol=8e-6
    )
    torch.testing.assert_close(actual_grad_in_bias, expected_grad_in_bias, rtol=8e-4, atol=8e-6)
    torch.testing.assert_close(actual_grad_lm_head, expected_grad_lm_head, rtol=8e-4, atol=8e-6)
    torch.testing.assert_close(native_enc, expected_enc, rtol=4e-4, atol=4e-6)

    if torch.cuda.is_available():
        cuda_model = HierarchosCore(config).cuda().eval()
        cuda_model.load_state_dict(model.state_dict())
        cuda_reference = _reference(
            cuda_model,
            token_ids.cuda(),
            token_residual.cuda(),
            gated_ltm_values.cuda(),
            grad_enc.cuda(),
        )
        torch.testing.assert_close(cuda_reference[2].cpu(), expected_enc, rtol=5e-4, atol=5e-6)
        torch.testing.assert_close(
            cuda_reference[6].cpu(), expected_grad_in_weight, rtol=8e-4, atol=8e-6
        )
        cuda_status = "PASS"
    else:
        cuda_status = "SKIP(no CUDA device present)"

    print(f"device={vulkan_result['device']}")
    print(f"queue_submissions={vulkan_result['queue_submissions']}")
    print(f"vulkan_enc_max_abs={(actual_enc - expected_enc).abs().max().item():.9g}")
    print(f"native_enc_max_abs={(native_enc - expected_enc).abs().max().item():.9g}")
    print(
        "in_proj_grad_max_abs="
        f"{(actual_grad_in_weight - expected_grad_in_weight).abs().max().item():.9g}"
    )
    print(
        "lm_head_embedding_grad_max_abs="
        f"{(actual_grad_lm_head - expected_grad_lm_head).abs().max().item():.9g}"
    )
    print(f"clamp_stress={'PASS' if args.clamp_stress else 'disabled'}")
    print(f"cuda_reference={cuda_status}")
    print("Vulkan token front-end PyTorch/native Rust parity: PASS")


if __name__ == "__main__":
    main()
