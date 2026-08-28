#!/usr/bin/env python3
"""Verify FP16/BF16 SafeTensors can cross PyTorch -> Vulkan -> native Rust/CUDA."""

from __future__ import annotations

import json
import gc
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


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


def _convert_package(source: Path, destination: Path, dtype: torch.dtype) -> None:
    shutil.copytree(source, destination)
    model_path = destination / "model.safetensors"
    state = load_file(str(model_path))
    converted = {
        name: (
            tensor.to(dtype=dtype).clone()
            if tensor.is_floating_point()
            else tensor.clone()
        )
        for name, tensor in state.items()
    }
    del state
    gc.collect()
    with safe_open(str(model_path), framework="pt", device="cpu") as tensors:
        metadata = tensors.metadata()
    rewritten = model_path.with_name("model.converted.safetensors")
    save_file(converted, str(rewritten), metadata=metadata)
    model_path.unlink()
    rewritten.replace(model_path)


def _load_package_into_model(
    model: HierarchosCore,
    package_state: dict[str, torch.Tensor],
) -> None:
    destination = model.state_dict()
    missing = []
    with torch.no_grad():
        for name, tensor in destination.items():
            source = package_state.get(name)
            if source is None:
                missing.append(name)
                continue
            tensor.copy_(source.to(dtype=tensor.dtype))

        # Hierarchos exports the canonical tied matrix once as lm_head.weight.
        # Keep the embedding alias synchronized when state_dict exposes both.
        if "lm_head.weight" in package_state and "tok_emb.weight" in destination:
            destination["tok_emb.weight"].copy_(
                package_state["lm_head.weight"].to(dtype=destination["tok_emb.weight"].dtype)
            )

    # `ltm.sources` is runtime memory state rather than a persisted model
    # parameter; `tok_emb.weight` may be omitted because lm_head.weight is the
    # canonical serialization of the tied token matrix.
    allowed_missing = {"tok_emb.weight", "ltm.sources"}
    unexpected_missing = [name for name in missing if name not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(
            "mixed package is missing model parameters: " + ", ".join(unexpected_missing[:8])
        )
    model.load_state_dict(destination)


def _exercise_dtype(
    base_package: Path,
    temp: Path,
    config,
    dtype: torch.dtype,
    dtype_label: str,
) -> tuple[float, str, str]:
    source_model = temp / f"source-{dtype_label}"
    trained_model = temp / f"trained-{dtype_label}"
    case_path = temp / f"case-{dtype_label}.json"
    _convert_package(base_package, source_model, dtype)

    source_state = load_file(str(source_model / "model.safetensors"))
    initial_weight = source_state["lm_head.weight"].float().clone()
    rows = 4
    hidden = torch.randn(rows, config.context_dim, dtype=torch.float32)
    targets = torch.tensor([2, 5, 11, 17], dtype=torch.long)
    lr = 2.0e-3
    weight_decay = 0.1
    reference_weight = torch.nn.Parameter(initial_weight)
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
    case_path.write_text(
        json.dumps(
            {
                "hidden": hidden.flatten().tolist(),
                "targets": targets.tolist(),
                "lr": lr,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1.0e-8,
                "weight_decay": weight_decay,
            }
        ),
        encoding="utf-8",
    )

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
    torch.testing.assert_close(
        trained_state["lm_head.weight"],
        reference_weight.detach(),
        rtol=3e-4,
        atol=3e-6,
    )
    assert trained_state["lm_head.weight"].dtype == torch.float32
    untouched_dtypes = {
        tensor.dtype
        for name, tensor in trained_state.items()
        if name != "lm_head.weight" and tensor.is_floating_point()
    }
    assert untouched_dtypes == {dtype}, untouched_dtypes

    tokens = [1, 2, 1, 3]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    reference_model = HierarchosCore(config).eval()
    _load_package_into_model(reference_model, trained_state)
    reference_model.reset_memory()
    with torch.no_grad():
        expected = reference_model(
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
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-5)

    if torch.cuda.is_available():
        cuda_model = HierarchosCore(config).cuda().eval()
        _load_package_into_model(cuda_model, trained_state)
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
        torch.testing.assert_close(cuda_logits, expected, rtol=4e-4, atol=5e-5)
        cuda_status = "PASS"
    else:
        cuda_status = "SKIP(no CUDA device present)"

    max_abs = (actual - expected).abs().max().item()
    return max_abs, cuda_status, package_json["device"]


def main() -> None:
    torch.manual_seed(20260820)
    config = tiny_coherent_config(32)
    base_model = HierarchosCore(config).eval()
    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-mixed-interchange-") as temp_dir:
        temp = Path(temp_dir)
        base_package = temp / "base-model"
        export_model(base_model, config, base_package)
        results = []
        for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            results.append((label, *_exercise_dtype(base_package, temp, config, dtype, label)))

    for label, max_abs, cuda_status, device in results:
        print(f"dtype={label} device={device}")
        print(f"native_vs_pytorch_max_abs={max_abs:.9g}")
        print(f"cuda_runtime_check={cuda_status}")
    print("PyTorch mixed SafeTensors -> Vulkan training -> native Rust/CUDA interchange: PASS")


if __name__ == "__main__":
    main()
