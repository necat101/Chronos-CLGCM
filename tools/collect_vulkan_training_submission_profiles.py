#!/usr/bin/env python3
"""Collect persistent, plan-aware Hierarchos Vulkan training profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hierarchos.utils.checkpoint import load_full_model_with_config
from tools.benchmark_vulkan_training_submission import _case_for_model, _fixture, _run_benchmark
from tools.export_rust_inference import export_model


DEFAULT_OUTPUT = ROOT / "benchmark_results" / "vulkan_training_submission_profiles.v1.jsonl"
SCHEMA_VERSION = 1


def _positive_csv(raw: str, name: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} contains non-integer value {part!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive; got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one positive integer")
    return values


def _microbatch_csv(raw: str) -> tuple[bool, list[int]]:
    include_auto = False
    numeric: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "auto":
            include_auto = True
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--microbatches expects 'auto' or positive integers; got {part!r}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"--microbatches values must be positive; got {value}"
            )
        if value not in seen:
            seen.add(value)
            numeric.append(value)
    if not include_auto and not numeric:
        raise argparse.ArgumentTypeError("--microbatches must contain 'auto' or a positive integer")
    return include_auto, numeric


def _kernel_geometry_csv(raw: str) -> list[str | None]:
    aliases = {
        "32": "rwkv-state-bwd-wg32",
        "wg32": "rwkv-state-bwd-wg32",
        "rwkv-state-bwd-wg32": "rwkv-state-bwd-wg32",
        "64": "rwkv-state-bwd-wg64",
        "wg64": "rwkv-state-bwd-wg64",
        "rwkv-state-bwd-wg64": "rwkv-state-bwd-wg64",
        "128": "rwkv-state-bwd-wg128",
        "wg128": "rwkv-state-bwd-wg128",
        "rwkv-state-bwd-wg128": "rwkv-state-bwd-wg128",
    }
    values: list[str | None] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        value = None if part == "auto" else aliases.get(part)
        if part != "auto" and value is None:
            raise argparse.ArgumentTypeError(
                "--kernel-geometries expects auto, 32/wg32, 64/wg64, or 128/wg128; "
                f"got {part!r}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--kernel-geometries must contain at least one value")
    return values


def _numerics_csv(raw: str) -> list[str]:
    aliases = {
        "strict": "strict",
        "strict-parity": "strict",
        "fast-subgroup": "fast-subgroup",
        "subgroup": "fast-subgroup",
        "fast-recurrent-tree": "fast-recurrent-tree",
        "recurrent-tree": "fast-recurrent-tree",
        "tree": "fast-recurrent-tree",
        "fast-recurrent-tiled": "fast-recurrent-tiled",
        "recurrent-tiled": "fast-recurrent-tiled",
        "tiled": "fast-recurrent-tiled",
        "fast-recurrent-subgroup": "fast-recurrent-subgroup",
        "recurrent-subgroup": "fast-recurrent-subgroup",
        "subgroup-recurrent": "fast-recurrent-subgroup",
    }
    values: list[str] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        value = aliases.get(part)
        if value is None:
            raise argparse.ArgumentTypeError(
                "--numerics expects strict, fast-subgroup, fast-recurrent-tree, "
                f"fast-recurrent-tiled, or fast-recurrent-subgroup; got {part!r}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--numerics must contain at least one policy")
    return values


def _precision_csv(raw: str) -> list[str]:
    aliases = {
        "fp32": "fp32",
        "fp16": "fp16-storage-fp32-compute",
        "fp16-storage-fp32-compute": "fp16-storage-fp32-compute",
        "fp16-parity": "fp16-storage-parity",
        "fp16-storage-parity": "fp16-storage-parity",
        "fp16-lm-backward": "fp16-storage-fp16-lm-backward",
        "fp16-storage-fp16-lm-backward": "fp16-storage-fp16-lm-backward",
    }
    values: list[str] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        value = aliases.get(part)
        if value is None:
            raise argparse.ArgumentTypeError(
                "--precisions expects fp32, fp16-storage-fp32-compute, "
                "fp16-storage-parity, or "
                "fp16-storage-fp16-lm-backward; "
                f"got {part!r}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--precisions must contain at least one policy")
    return values


def _plan_matrix(
    tokens: Iterable[int],
    sequences: Iterable[int],
    include_auto: bool,
    microbatches: Iterable[int],
    checkpoint_strides: Iterable[int],
    kernel_geometries: Iterable[str | None],
    numerics_policies: Iterable[str],
    precision_policies: Iterable[str],
) -> list[tuple[int, int, int | None, int | None, str | None, str, str]]:
    matrix: list[tuple[int, int, int | None, int | None, str | None, str, str]] = []
    for token_count in tokens:
        for sequence_count in sequences:
            if include_auto:
                # Automatic tape planning is itself allowed to choose kernel
                # geometry, so there is no meaningful "forced" geometry row in
                # this branch. Exact plans below are the deterministic geometry
                # profiling surface.
                for numerics in numerics_policies:
                    for precision in precision_policies:
                        matrix.append(
                            (token_count, sequence_count, None, None, None, numerics, precision)
                        )
            for microbatch_size in microbatches:
                if microbatch_size > sequence_count:
                    continue
                for checkpoint_stride in checkpoint_strides:
                    if checkpoint_stride > token_count:
                        continue
                    for kernel_geometry in kernel_geometries:
                        for numerics in numerics_policies:
                            for precision in precision_policies:
                                matrix.append(
                                    (
                                        token_count,
                                        sequence_count,
                                        microbatch_size,
                                        checkpoint_stride,
                                        kernel_geometry,
                                        numerics,
                                        precision,
                                    )
                                )
    return matrix


def _benchmark_args(
    args: argparse.Namespace,
    tokens: int,
    sequences: int,
    microbatch_size: int | None,
    checkpoint_stride: int | None,
    kernel_geometry: str | None,
    numerics: str,
    precision: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=tokens,
        sequences=sequences,
        warmup=args.warmup,
        iterations=args.iterations,
        training_step=args.training_step,
        normalization=args.normalization,
        budget_fraction=args.budget_fraction,
        reserve_mib=args.reserve_mib,
        autotune_log=args.autotune_log,
        reautotune=args.reautotune,
        microbatch_size=microbatch_size,
        checkpoint_stride=checkpoint_stride,
        h_kernel_geometry=kernel_geometry,
        l_kernel_geometry=kernel_geometry,
        numerics=numerics,
        precision=precision,
    )


def _profile_key(result: dict[str, object]) -> dict[str, object]:
    samples = result.get("samples")
    first_sample = samples[0] if isinstance(samples, list) and samples else {}
    return {
        "device": result.get("device"),
        "subgroup_size": result.get("subgroup_size"),
        "training_precision_policy": result.get("training_precision_policy", "fp32"),
        "h_low_rank_fp16_parameter_storage_active": result.get(
            "h_low_rank_fp16_parameter_storage_active", False
        ),
        "l_low_rank_fp16_parameter_storage_active": result.get(
            "l_low_rank_fp16_parameter_storage_active", False
        ),
        "projection_fp16_parameter_storage_active": result.get(
            "projection_fp16_parameter_storage_active", False
        ),
        "lm_head_fp16_parameter_storage_active": result.get(
            "lm_head_fp16_parameter_storage_active", False
        ),
        "lm_head_execution_arm": result.get("lm_head_execution_arm", "fp32"),
        "backward_kernel_geometry_revision": result.get("backward_kernel_geometry_revision", 0),
        "architecture_revision": result.get("architecture_revision"),
        "batch": result.get("batch"),
        "context_dim": result.get("context_dim"),
        "persistent_dim": result.get("persistent_dim"),
        "ltm_slots": result.get("ltm_slots"),
        "ltm_key_dim": result.get("ltm_key_dim"),
        "ltm_val_dim": result.get("ltm_val_dim"),
        "ltm_topk": result.get("ltm_topk"),
        "vocab_size": result.get("vocab_size"),
        "h_hidden": result.get("h_hidden"),
        "l_hidden": result.get("l_hidden"),
        "h_width": result.get("h_width"),
        "l_width": result.get("l_width"),
        "h_state_size": result.get("h_state_size"),
        "l_state_size": result.get("l_state_size"),
        "h_rwkv_head_size": result.get("h_rwkv_head_size"),
        "l_rwkv_head_size": result.get("l_rwkv_head_size"),
        "h_low_rank_ranks": result.get("h_low_rank_ranks"),
        "l_low_rank_ranks": result.get("l_low_rank_ranks"),
        "token_adapter_rank": result.get("token_adapter_rank"),
        "max_h_steps": result.get("max_h_steps"),
        "max_l_steps": result.get("max_l_steps"),
        "tokens_per_sequence": result.get("tokens_per_sequence"),
        "sequences": result.get("sequences"),
        "sequence_microbatch_size": (
            first_sample.get("sequence_microbatch_size")
            if isinstance(first_sample, dict)
            else None
        ),
        "state_checkpoint_stride": (
            first_sample.get("state_checkpoint_stride")
            if isinstance(first_sample, dict)
            else None
        ),
        "device_local_pressure_bucket": (
            first_sample.get("device_local_pressure_bucket")
            if isinstance(first_sample, dict)
            else None
        ),
        "h_backward_segment_schedule": (
            first_sample.get("h_backward_segment_schedule")
            if isinstance(first_sample, dict)
            and first_sample.get("h_backward_segment_schedule") is not None
            else result.get("h_backward_schedule")
        ),
        "l_backward_segment_schedule": (
            first_sample.get("l_backward_segment_schedule")
            if isinstance(first_sample, dict)
            and first_sample.get("l_backward_segment_schedule") is not None
            else result.get("l_backward_schedule")
        ),
        "h_backward_kernel_geometry": (
            first_sample.get("h_backward_kernel_geometry")
            if isinstance(first_sample, dict)
            and first_sample.get("h_backward_kernel_geometry") is not None
            else result.get("h_backward_kernel_geometry")
        ),
        "l_backward_kernel_geometry": (
            first_sample.get("l_backward_kernel_geometry")
            if isinstance(first_sample, dict)
            and first_sample.get("l_backward_kernel_geometry") is not None
            else result.get("l_backward_kernel_geometry")
        ),
        "rwkv_numerics_policy": (
            first_sample.get("rwkv_numerics_policy")
            if isinstance(first_sample, dict)
            and first_sample.get("rwkv_numerics_policy") is not None
            else result.get("numerics_policy", "strict-parity")
        ),
    }


def _append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_matrix(
    model_dir: Path,
    case_path: Path,
    args: argparse.Namespace,
    model_source: str,
    case_source: str,
) -> tuple[int, int]:
    include_auto, microbatches = _microbatch_csv(args.microbatches)
    matrix = _plan_matrix(
        _positive_csv(args.tokens, "--tokens"),
        _positive_csv(args.sequences, "--sequences"),
        include_auto,
        microbatches,
        _positive_csv(args.checkpoint_strides, "--checkpoint-strides"),
        _kernel_geometry_csv(args.kernel_geometries),
        _numerics_csv(args.numerics),
        _precision_csv(args.precisions),
    )
    if args.max_cases > 0:
        matrix = matrix[: args.max_cases]

    ok = 0
    rejected = 0
    for index, (
        tokens,
        sequences,
        microbatch_size,
        checkpoint_stride,
        kernel_geometry,
        numerics,
        precision,
    ) in enumerate(
        matrix, 1
    ):
        mode = "auto" if microbatch_size is None else f"mb={microbatch_size},stride={checkpoint_stride}"
        geometry_mode = kernel_geometry or "auto"
        print(
            f"[{index}/{len(matrix)}] tokens={tokens} sequences={sequences} "
            f"plan={mode} kernel_geometry={geometry_mode} numerics={numerics} precision={precision}",
            flush=True,
        )
        collected_at = datetime.now(timezone.utc).isoformat()
        request = {
            "tokens_per_sequence": tokens,
            "sequences": sequences,
            "sequence_microbatch_size": microbatch_size,
            "state_checkpoint_stride": checkpoint_stride,
            "h_backward_kernel_geometry": kernel_geometry,
            "l_backward_kernel_geometry": kernel_geometry,
            "rwkv_numerics_policy": numerics,
            "training_precision_policy": precision,
            "normalization": args.normalization,
            "budget_fraction": args.budget_fraction,
            "reserve_mib": args.reserve_mib,
        }
        try:
            result = _run_benchmark(
                model_dir,
                case_path,
                _benchmark_args(
                    args,
                    tokens,
                    sequences,
                    microbatch_size,
                    checkpoint_stride,
                    kernel_geometry,
                    numerics,
                    precision,
                ),
            )
        except RuntimeError as exc:
            rejected += 1
            record = {
                "schema_version": SCHEMA_VERSION,
                "collected_at_utc": collected_at,
                "status": "rejected",
                "model_source": model_source,
                "case_source": case_source,
                "request": request,
                "error": str(exc)[-4000:],
            }
            _append_record(args.output, record)
            if args.fail_fast:
                raise
            continue

        ok += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "collected_at_utc": collected_at,
            "status": "ok",
            "model_source": model_source,
            "case_source": case_source,
            "request": request,
            "profile_key": _profile_key(result),
            "result": result,
        }
        _append_record(args.output, record)

    return ok, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument(
        "--fixture-width",
        type=int,
        default=32,
        help="width for the deterministic fixture when no external model is supplied",
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        help=(
            "PyTorch Hierarchos model directory or .pt checkpoint. It is exported to the "
            "canonical SafeTensors Rust package in a temporary directory and paired with a "
            "deterministic profiling case."
        ),
    )
    parser.add_argument("--tokens", default="4,8,16")
    parser.add_argument("--sequences", default="1,2,4")
    parser.add_argument("--microbatches", default="auto,1,2,4")
    parser.add_argument("--checkpoint-strides", default="1,2,4")
    parser.add_argument(
        "--kernel-geometries",
        default="auto",
        help="comma-separated auto/32/64/128 compiled RWKV backward local-size variants",
    )
    parser.add_argument(
        "--numerics",
        default="strict",
        help=(
            "comma-separated strict/fast-subgroup/fast-recurrent-tree/fast-recurrent-tiled/"
            "fast-recurrent-subgroup "
            "RWKV reduction policies"
        ),
    )
    parser.add_argument(
        "--precisions",
        default="fp32",
        help=(
            "comma-separated fp32/fp16-storage-fp32-compute/"
            "fp16-storage-parity/"
            "fp16-storage-fp16-lm-backward training precision arms"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--training-step", type=int, default=0)
    parser.add_argument("--normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--budget-fraction", type=float, default=0.85)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--autotune-log", action="store_true")
    parser.add_argument("--reautotune", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if (args.model is None) != (args.case is None):
        parser.error("--model and --case must be supplied together")
    if args.fixture_width <= 0:
        parser.error("--fixture-width must be positive")
    if args.source_model is not None and args.model is not None:
        parser.error("--source-model cannot be combined with --model/--case")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.max_cases < 0:
        parser.error("--max-cases must be non-negative")
    if not (0.0 < args.budget_fraction <= 1.0):
        parser.error("--budget-fraction must be in (0, 1]")
    if args.reserve_mib < 0:
        parser.error("--reserve-mib must be non-negative")

    if args.source_model is not None:
        source_model = args.source_model.resolve()
        model, config = load_full_model_with_config(str(source_model), torch.device("cpu"))
        model.eval()
        case = _case_for_model(model, config)
        with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-source-profile-") as temp_dir:
            temp = Path(temp_dir)
            model_dir = temp / "model"
            case_path = temp / "case.json"
            export_model(model, config, model_dir)
            case_path.write_text(json.dumps(case), encoding="utf-8")
            ok, rejected = _run_matrix(
                model_dir,
                case_path,
                args,
                str(source_model),
                "deterministic-profile-case-from-source-model",
            )
    elif args.model is not None and args.case is not None:
        ok, rejected = _run_matrix(
            args.model,
            args.case,
            args,
            str(args.model.resolve()),
            str(args.case.resolve()),
        )
    else:
        model, config, case = _fixture(args.fixture_width)
        with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-profile-matrix-") as temp_dir:
            temp = Path(temp_dir)
            model_dir = temp / "model"
            case_path = temp / "case.json"
            export_model(model, config, model_dir)
            case_path.write_text(json.dumps(case), encoding="utf-8")
            ok, rejected = _run_matrix(
                model_dir,
                case_path,
                args,
                "deterministic-tiny-coherent-fixture",
                "deterministic-tiny-coherent-fixture",
            )

    print(
        f"wrote {ok} successful and {rejected} rejected profiles to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
