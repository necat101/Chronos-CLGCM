#!/usr/bin/env python3
"""Prove Vulkan-trained weights round-trip through PyTorch and native Rust inference."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


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


def main() -> None:
    torch.manual_seed(20260812)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).eval()

    rows = 4
    hidden = torch.randn(rows, config.context_dim, dtype=torch.float32)
    targets = torch.tensor([2, 5, 11, 17], dtype=torch.long)
    lr = 2.0e-3
    weight_decay = 0.1

    reference_weight = torch.nn.Parameter(model.lm_head.weight.detach().cpu().clone())
    optimizer = torch.optim.AdamW(
        [reference_weight],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=weight_decay,
    )
    reference_loss = F.cross_entropy(hidden @ reference_weight.t(), targets)
    reference_loss.backward()
    optimizer.step()

    case = {
        "hidden": hidden.flatten().tolist(),
        "targets": targets.tolist(),
        "lr": lr,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "weight_decay": weight_decay,
    }

    tokens = [1, 2, 1, 3]
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-interchange-") as temp_dir:
        temp = Path(temp_dir)
        source_model = temp / "source-model"
        trained_model = temp / "trained-model"
        case_path = temp / "case.json"
        export_model(model, config, source_model)
        case_path.write_text(json.dumps(case), encoding="utf-8")

        package_result = _run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-package-step",
                "--",
                "--model",
                str(source_model),
                "--case",
                str(case_path),
                "--output-model",
                str(trained_model),
            ]
        )
        package_json = json.loads(package_result.stdout)
        trained_state = load_file(str(trained_model / "model.safetensors"))
        trained_weight = trained_state["lm_head.weight"]
        torch.testing.assert_close(trained_weight, reference_weight.detach(), rtol=3e-4, atol=3e-6)

        # Consume the Vulkan-written SafeTensors weight in the original PyTorch
        # model. lm_head and tok_emb are tied, matching the Rust runtime's use of
        # lm_head.weight for both lookup and output projection.
        with torch.no_grad():
            model.lm_head.weight.copy_(trained_weight)
        model.reset_memory()
        with torch.no_grad():
            expected = model(
                input_ids,
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
            )["logits"][0].float().cpu()

        rust_result = _run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-inference" / "Cargo.toml"),
                "--bin",
                "hierarchos-infer",
                "--",
                "--model",
                str(trained_model),
                "--tokens",
                ",".join(map(str, tokens)),
            ]
        )
        actual = torch.tensor(json.loads(rust_result.stdout)["logits"], dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

        if torch.cuda.is_available():
            cuda_model = HierarchosCore(config).cuda().eval()
            cuda_model.load_state_dict(model.state_dict())
            cuda_model.reset_memory()
            with torch.no_grad():
                cuda_logits = cuda_model(
                    input_ids.cuda(),
                    return_topk_values=False,
                    return_raw_topk_values=False,
                    return_topk_indices=False,
                    return_step_telemetry=False,
                    return_numerics=False,
                )["logits"][0].float().cpu()
            torch.testing.assert_close(cuda_logits, expected, rtol=3e-4, atol=3e-5)
            cuda_status = "PASS"
        else:
            cuda_status = "SKIP(no CUDA device present)"

    max_abs = (actual - expected).abs().max().item()
    print(f"device={package_json['device']}")
    print(f"head_step_loss={package_json['loss']:.9g}")
    print(f"native_vs_pytorch_max_abs={max_abs:.9g}")
    print(f"cuda_runtime_check={cuda_status}")
    print("Vulkan -> SafeTensors -> PyTorch/native Rust interchange: PASS")


if __name__ == "__main__":
    main()
