#!/usr/bin/env python3
"""Benchmark the real raw-token Hierarchos Vulkan training submission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture


def _case_for_model(model: HierarchosCore, config: object) -> dict[str, object]:
    torch.manual_seed(20260814)
    batch = 2
    token_rows = [[2, 7], [5, 8], [2, 7]]
    reset_rows = [[1, 1], [0, 0], [0, 1]]
    target_rows = [[5, 9], [2, 7], [8, 4]]
    steps: list[dict[str, object]] = []
    for index, (token_ids, reset_lanes, targets) in enumerate(
        zip(token_rows, reset_rows, target_rows, strict=True)
    ):
        scale = 1.0 + index * 0.15
        steps.append(
            {
                "token_ids": token_ids,
                "rosa_reset_lanes": reset_lanes,
                "previous_context": (
                    torch.randn(batch, config.context_dim) * (0.035 * scale)
                )
                .flatten()
                .tolist(),
                "target_context": (
                    torch.randn(batch, config.context_dim) * (0.03 * scale)
                )
                .flatten()
                .tolist(),
                "context_alpha": 0.25 + 0.125 * index,
                "h_token_ids": token_ids,
                "l_token_ids": token_ids,
                "h_to_context_grad": (
                    torch.randn(batch, config.context_dim) * 0.004
                )
                .flatten()
                .tolist(),
                "h_depth_grad": [0.003 + index * 0.0004, -0.0025],
                "final_drift_grad": (
                    torch.randn(batch, config.context_dim) * 0.0035
                )
                .flatten()
                .tolist(),
                "commitment_cost_grad": [0.05, 0.035 + index * 0.005],
                "targets": targets,
            }
        )

    h_state = (
        torch.randn(batch, config.h_hidden, model.h_rnn.state_size, dtype=torch.float32)
        * 0.01
    )
    l_state = (
        torch.randn(batch, config.l_hidden, model.l_rnn.state_size, dtype=torch.float32)
        * 0.01
    )
    case = {
        "h_initial_packed_state": h_state.flatten().tolist(),
        "l_initial_packed_state": l_state.flatten().tolist(),
        "steps": steps,
        "optimizer": {
            "lr": 3.0e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "weight_decay": 0.0,
        },
    }
    return case


def _fixture(
    width: int = 32,
    vocab_size: int = 64,
) -> tuple[HierarchosCore, object, dict[str, object]]:
    torch.manual_seed(20260814)
    config = tiny_coherent_config(width)
    config.vocab_size = vocab_size
    config.max_h_steps = max(1, config.max_h_steps)
    config.max_l_steps = max(1, config.max_l_steps)
    model = HierarchosCore(config).eval()
    _make_nontrivial_memory_fixture(model, config)
    case = _case_for_model(model, config)
    return model, config, case


def _run_benchmark(
    model_dir: Path,
    case_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--manifest-path",
        str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
        "--bin",
        "hierarchos-vulkan-training-submission-bench",
        "--",
        "--model",
        str(model_dir),
        "--case",
        str(case_path),
        "--tokens",
        str(args.tokens),
        "--sequences",
        str(args.sequences),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--training-step",
        str(args.training_step),
        "--normalization",
        args.normalization,
        "--readback",
        args.readback,
        "--budget-fraction",
        str(args.budget_fraction),
        "--reserve-mib",
        str(args.reserve_mib),
    ]
    env = os.environ.copy()
    if args.required_subgroup_size is not None:
        env["HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE"] = str(args.required_subgroup_size)
    if args.low_rank_first_stage_arm == "portable":
        env.pop("HIERARCHOS_RWKV_LOW_RANK_ENABLE_SUBGROUP_PACKED_SHARE", None)
    elif args.low_rank_first_stage_arm == "subgroup-packed-share":
        env["HIERARCHOS_RWKV_LOW_RANK_ENABLE_SUBGROUP_PACKED_SHARE"] = "1"
    if args.autotune_log:
        env["HIERARCHOS_RWKV_BACKWARD_SEGMENT_AUTOTUNE_LOG"] = "1"
        env["HIERARCHOS_VULKAN_TAPE_PROFILE_LOG"] = "1"
        env["HIERARCHOS_VULKAN_LM_AUTOTUNE_LOG"] = "1"
    if args.reautotune:
        env["HIERARCHOS_RWKV_BACKWARD_SEGMENT_REAUTOTUNE"] = "1"
        env["HIERARCHOS_VULKAN_LM_REAUTOTUNE"] = "1"
    if args.microbatch_size is not None:
        command.extend(["--microbatch-size", str(args.microbatch_size)])
        command.extend(["--checkpoint-stride", str(args.checkpoint_stride)])
    h_kernel_geometry = getattr(args, "h_kernel_geometry", None)
    l_kernel_geometry = getattr(args, "l_kernel_geometry", None)
    if h_kernel_geometry is not None:
        command.extend(["--h-kernel-geometry", str(h_kernel_geometry)])
        command.extend(["--l-kernel-geometry", str(l_kernel_geometry)])
    command.extend(["--numerics", args.numerics])
    command.extend(["--precision", args.precision])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}"
        )
    result = json.loads(completed.stdout)
    expects_fp16_parameter_storage = args.precision != "fp32"
    for tower in ("h", "l"):
        field = f"{tower}_low_rank_fp16_parameter_storage_active"
        if bool(result.get(field)) != expects_fp16_parameter_storage:
            raise RuntimeError(
                f"benchmark precision contract failed: {field}={result.get(field)!r} "
                f"for requested precision {args.precision!r}"
            )
    if args.low_rank_first_stage_arm is not None and expects_fp16_parameter_storage:
        for tower in ("h", "l"):
            field = f"{tower}_low_rank_fp16_full_forward_first_stage_arm"
            if result.get(field) != args.low_rank_first_stage_arm:
                raise RuntimeError(
                    f"benchmark low-rank first-stage arm contract failed: "
                    f"{field}={result.get(field)!r}, requested {args.low_rank_first_stage_arm!r}"
                )
    for field in (
        "projection_fp16_parameter_storage_active",
        "lm_head_fp16_parameter_storage_active",
    ):
        if bool(result.get(field)) != expects_fp16_parameter_storage:
            raise RuntimeError(
                f"benchmark precision contract failed: {field}={result.get(field)!r} "
                f"for requested precision {args.precision!r}"
            )
    expects_native_fp16_lm_backward = args.precision in (
        "fp16-storage-parity",
        "fp16-storage-fp16-lm-backward",
    )
    if (
        bool(result.get("lm_head_native_fp16_backward_compute_active"))
        != expects_native_fp16_lm_backward
    ):
        raise RuntimeError(
            "benchmark native-FP16 LM backward contract failed: "
            f"active={result.get('lm_head_native_fp16_backward_compute_active')!r} "
            f"for requested precision {args.precision!r}"
        )
    lm_execution_arm = result.get("lm_head_execution_arm")
    expected_lm_arms = (
        {
            "fp16-packed",
            "fp16-ce-tape",
            "fp16-ce-tape-rows8",
            "fp16-ce-tape-rows16",
            "fp16-ce-tape-rows16-dot4",
            "fp16-ce-tape-rows16-fused-adjoints",
            "fp16-ce-tape-rows16-dot4-fused-adjoints",
            "fp16-ce-tape-rows16-cluster4-fused-adjoints",
            "fp16-native",
            "fp16-native-reuse64",
            "fp16-native-reuse128",
            "fp16-native-reuse224",
        }
        if expects_fp16_parameter_storage
        else {"fp32"}
    )
    if lm_execution_arm not in expected_lm_arms:
        raise RuntimeError(
            "benchmark LM execution-arm contract failed: "
            f"lm_head_execution_arm={lm_execution_arm!r} "
            f"for requested precision {args.precision!r}"
        )
    if expects_native_fp16_lm_backward and lm_execution_arm != "fp16-native":
        raise RuntimeError(
            "native-FP16 LM backward currently requires the explicit fp16-native arm; "
            f"got {lm_execution_arm!r}"
        )
    lm_weight_grad_topology = result.get("lm_head_weight_grad_topology")
    expected_lm_topologies = {"dw-vocab4", "dw-vocab8", "dw-vocab16"}
    if expects_fp16_parameter_storage:
        if lm_weight_grad_topology not in expected_lm_topologies:
            raise RuntimeError(
                "benchmark LM dW-topology contract failed: "
                f"lm_head_weight_grad_topology={lm_weight_grad_topology!r}"
            )
    elif lm_weight_grad_topology is not None:
        raise RuntimeError(
            "benchmark FP32 run unexpectedly reported an LM FP16 dW topology: "
            f"{lm_weight_grad_topology!r}"
        )
    fused_adjoint_topology = result.get("lm_head_fused_adjoint_topology")
    fused_arms = {
        "fp16-ce-tape-rows16-fused-adjoints",
        "fp16-ce-tape-rows16-dot4-fused-adjoints",
        "fp16-ce-tape-rows16-cluster4-fused-adjoints",
    }
    if lm_execution_arm in fused_arms:
        if fused_adjoint_topology not in {
            "fused-shared-hidden",
            "fused-private-hidden",
            "fused-private-hidden-tile256-wg256",
        }:
            raise RuntimeError(
                "benchmark LM fused-adjoint topology contract failed: "
                f"lm_head_fused_adjoint_topology={fused_adjoint_topology!r}"
            )
    elif fused_adjoint_topology is not None:
        raise RuntimeError(
            "benchmark non-fused LM arm unexpectedly reported a fused-adjoint topology: "
            f"{fused_adjoint_topology!r}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument(
        "--fixture-width",
        type=int,
        default=32,
        help="width for the deterministic fixture when --model/--case are omitted",
    )
    parser.add_argument(
        "--fixture-vocab",
        type=int,
        default=64,
        help="vocabulary size for the deterministic fixture; 50257 exercises production LM-head geometry",
    )
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--training-step", type=int, default=0)
    parser.add_argument("--normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument(
        "--readback",
        choices=("full", "loss-only"),
        default="full",
        help="host diagnostic readback policy; loss-only avoids per-token ACT/state copies",
    )
    parser.add_argument("--budget-fraction", type=float, default=0.85)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--checkpoint-stride", type=int)
    parser.add_argument("--h-kernel-geometry")
    parser.add_argument("--l-kernel-geometry")
    parser.add_argument(
        "--numerics",
        choices=(
            "strict",
            "fast-subgroup",
            "fast-recurrent-tree",
            "fast-recurrent-tiled",
            "fast-recurrent-subgroup",
        ),
        default="strict",
    )
    parser.add_argument(
        "--precision",
        choices=(
            "fp32",
            "fp16-storage-fp32-compute",
            "fp16-storage-parity",
            "fp16-storage-fp16-lm-backward",
        ),
        default="fp32",
        help="trainable execution-storage precision arm for the Vulkan graph",
    )
    parser.add_argument(
        "--required-subgroup-size",
        type=int,
        help=(
            "request an exact compute subgroup width through "
            "HIERARCHOS_VULKAN_REQUIRED_SUBGROUP_SIZE; useful for cross-vendor wave32/wave64 A/Bs"
        ),
    )
    parser.add_argument(
        "--low-rank-first-stage-arm",
        choices=("portable", "subgroup-packed-share"),
        help=(
            "select the full FP16 RWKV low-rank first-stage load arm; omitted preserves the inherited environment"
        ),
    )
    parser.add_argument("--autotune-log", action="store_true")
    parser.add_argument("--reautotune", action="store_true")
    args = parser.parse_args()
    if (args.model is None) != (args.case is None):
        parser.error("--model and --case must be supplied together")
    if args.fixture_width <= 0:
        parser.error("--fixture-width must be positive")
    if args.fixture_vocab <= 9:
        parser.error("--fixture-vocab must be at least 10 for the deterministic token fixture")
    if args.required_subgroup_size is not None and args.required_subgroup_size <= 0:
        parser.error("--required-subgroup-size must be positive")
    if (args.microbatch_size is None) != (args.checkpoint_stride is None):
        parser.error("--microbatch-size and --checkpoint-stride must be supplied together")
    if args.microbatch_size is not None and args.microbatch_size <= 0:
        parser.error("--microbatch-size must be positive")
    if args.checkpoint_stride is not None and args.checkpoint_stride <= 0:
        parser.error("--checkpoint-stride must be positive")
    if (args.h_kernel_geometry is None) != (args.l_kernel_geometry is None):
        parser.error("--h-kernel-geometry and --l-kernel-geometry must be supplied together")
    if args.h_kernel_geometry is not None and args.microbatch_size is None:
        parser.error(
            "forced kernel geometry requires --microbatch-size and --checkpoint-stride "
            "so automatic policy selection cannot replace the requested arm"
        )

    if args.model is not None and args.case is not None:
        result = _run_benchmark(args.model, args.case, args)
    else:
        model, config, case = _fixture(args.fixture_width, args.fixture_vocab)
        with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-training-bench-") as temp_dir:
            temp = Path(temp_dir)
            model_dir = temp / "model"
            case_path = temp / "case.json"
            export_model(model, config, model_dir)
            case_path.write_text(json.dumps(case), encoding="utf-8")
            result = _run_benchmark(model_dir, case_path, args)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
