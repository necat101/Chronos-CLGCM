#!/usr/bin/env python3
"""Exercise exact mid-epoch resume through the real Hierarchos Vulkan trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore, load_full_model_with_config
from hierarchos.training.trainer import build_hierarchos_optimizer
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.vulkan_optimizer_bridge import (
    load_vulkan_training_package_into_torch,
    read_vulkan_training_manifest,
    read_vulkan_training_replay,
)


def _run(command: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if expect_success and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not expect_success and completed.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}")
    return completed


def _trainer_command(
    *,
    source_flag: str,
    source: Path,
    dataset: Path,
    output: Path,
    device_index: int,
    release: bool,
) -> list[str]:
    command = ["cargo", "run", "--quiet"]
    if release:
        command.append("--release")
    command.extend(
        [
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-train",
            "--",
            source_flag,
            str(source),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "2",
            "--save-steps",
            "1",
            "--persist-state",
            "--no-shuffle",
            "--max-commitment-cost-for-backward",
            "0",
            "--weight-decay",
            "0",
            "--device-index",
            str(device_index),
        ]
    )
    return command


def _assert_tensor_packages_equal(left: Path, right: Path, filename: str) -> None:
    left_state = load_file(str(left / filename))
    right_state = load_file(str(right / filename))
    if left_state.keys() != right_state.keys():
        raise AssertionError(f"{filename} tensor registries differ")
    mismatches = [name for name in left_state if not torch.equal(left_state[name], right_state[name])]
    if mismatches:
        worst_name = max(
            mismatches,
            key=lambda name: float((left_state[name] - right_state[name]).abs().max().item()),
        )
        worst = float((left_state[worst_name] - right_state[worst_name]).abs().max().item())
        raise AssertionError(
            f"{filename} diverged after exact resume: tensor={worst_name!r} max_abs={worst:.9g}"
        )


def _write_dataset(path: Path, *, first_token: int = 2) -> None:
    rows = [
        {"input_ids": [first_token, 3, 4, 5], "labels": [first_token, 3, 4, 5]},
        {"input_ids": [3, 4, 5, 6], "labels": [3, 4, 5, 6]},
        {"input_ids": [4, 5, 6, 7], "labels": [4, 5, 6, 7]},
        {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless the final portable package also executes on CUDA",
    )
    args = parser.parse_args()

    if args.device_index < 0:
        parser.error("--device-index must be non-negative")

    torch.manual_seed(20260826)
    config = tiny_coherent_config(16)
    config.max_h_steps = 1
    config.max_l_steps = 1
    model = HierarchosCore(config).eval()

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-real-trainer-resume-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        dataset = temp / "dataset.jsonl"
        mismatch_dataset = temp / "dataset-mismatch.jsonl"
        uninterrupted = temp / "uninterrupted"
        resumed = temp / "resumed"
        export_model(model, config, model_dir)
        _write_dataset(dataset)
        _write_dataset(mismatch_dataset, first_token=9)

        _run(
            _trainer_command(
                source_flag="--model",
                source=model_dir,
                dataset=dataset,
                output=uninterrupted,
                device_index=args.device_index,
                release=args.release,
            )
        )

        open_checkpoint = uninterrupted / "checkpoint-epoch-1-step-1"
        manifest = read_vulkan_training_manifest(open_checkpoint)
        replay = read_vulkan_training_replay(open_checkpoint, manifest)
        if replay is None:
            raise AssertionError("real trainer omitted portable replay state")
        if manifest.get("mid_epoch_step") != 1 or manifest.get("gradient_file") != "gradients.safetensors":
            raise AssertionError("step-1 checkpoint is not an open exact-resume accumulation window")
        if not isinstance(replay.get("run_identity"), dict):
            raise AssertionError("real trainer checkpoint omitted run_identity")
        running_states = replay.get("running_states")
        if not isinstance(running_states, tuple) or len(running_states) != 6:
            raise AssertionError("real trainer checkpoint omitted the six-part running_states carrier")

        torch_model, _ = load_full_model_with_config(str(open_checkpoint), torch.device("cpu"))
        optimizer_args = argparse.Namespace(
            starting_lr=1.0e-4,
            rwkv_weight_decay=0.0,
            adamw_eps=1.0e-8,
            _optimizer_grouping_version=2,
        )
        torch_optimizer = build_hierarchos_optimizer(
            torch_model,
            optimizer_args,
            torch.device("cpu"),
        )
        bridged = load_vulkan_training_package_into_torch(
            torch_model,
            torch_optimizer,
            open_checkpoint,
        )
        if bridged.pending_gradients is None:
            raise AssertionError("PyTorch bridge lost the open Vulkan gradient window")
        if bridged.pytorch_accumulation_normalization != "weighted-token":
            raise AssertionError("PyTorch bridge changed the accumulation normalization contract")
        if sum(parameter.grad is not None for parameter in torch_model.parameters()) != len(
            bridged.pending_gradients.gradients
        ):
            raise AssertionError("PyTorch bridge did not hydrate every pending Vulkan gradient")

        _run(
            _trainer_command(
                source_flag="--resume-from-ckpt",
                source=open_checkpoint,
                dataset=dataset,
                output=resumed,
                device_index=args.device_index,
                release=args.release,
            )
        )
        _assert_tensor_packages_equal(uninterrupted, resumed, "model.safetensors")
        _assert_tensor_packages_equal(uninterrupted, resumed, "optimizer.safetensors")

        mismatch = _run(
            _trainer_command(
                source_flag="--resume-from-ckpt",
                source=open_checkpoint,
                dataset=mismatch_dataset,
                output=temp / "mismatch-must-fail",
                device_index=args.device_index,
                release=args.release,
            ),
            expect_success=False,
        )
        if "exact native resume identity mismatch" not in mismatch.stderr:
            raise AssertionError(
                "changed dataset failed for the wrong reason instead of the run-identity boundary"
            )

        pytorch_model, _ = load_full_model_with_config(str(resumed), torch.device("cpu"))
        pytorch_model.eval()
        input_ids = torch.tensor([[2, 5, 2]], dtype=torch.long)
        with torch.no_grad():
            expected = pytorch_model(
                input_ids,
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
            )["logits"][0].float().cpu()

        native = _run(
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(ROOT / "hierarchos-inference" / "Cargo.toml"),
                "--bin",
                "hierarchos-infer",
                "--",
                "--model",
                str(resumed),
                "--tokens",
                "2,5,2",
            ]
        )
        native_logits = torch.tensor(json.loads(native.stdout)["logits"], dtype=torch.float32)
        torch.testing.assert_close(native_logits, expected, rtol=2.0e-4, atol=2.0e-5)
        native_max_abs = float((native_logits - expected).abs().max().item())

        if torch.cuda.is_available():
            cuda_model, _ = load_full_model_with_config(str(resumed), torch.device("cuda"))
            cuda_model.eval()
            with torch.no_grad():
                cuda_logits = cuda_model(
                    input_ids.cuda(),
                    return_topk_values=False,
                    return_raw_topk_values=False,
                    return_topk_indices=False,
                    return_step_telemetry=False,
                    return_numerics=False,
                )["logits"][0].float().cpu()
            torch.testing.assert_close(cuda_logits, expected, rtol=3.0e-4, atol=3.0e-5)
            cuda_status = "PASS"
        elif args.require_cuda:
            raise RuntimeError("--require-cuda requested but torch.cuda.is_available() is false")
        else:
            cuda_status = "SKIP(no CUDA device present)"

        print("real_trainer_periodic_checkpoint=PASS")
        print("real_trainer_open_window_resume=PASS")
        print("uninterrupted_vs_resumed_model=BIT_EXACT")
        print("uninterrupted_vs_resumed_optimizer=BIT_EXACT")
        print("pytorch_open_window_bridge=PASS")
        print("dataset_identity_fail_closed=PASS")
        print(f"native_vs_pytorch_max_abs={native_max_abs:.9g}")
        print("native_rust_inference_reload=PASS")
        print(f"cuda_runtime_check={cuda_status}")


if __name__ == "__main__":
    main()
