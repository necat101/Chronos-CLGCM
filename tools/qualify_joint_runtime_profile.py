#!/usr/bin/env python3
"""A/B qualify an imported multi-device Hierarchos Vulkan runtime profile.

This harness is deliberately stricter than the single-device parity utility:
it launches the real multi-adapter trainer twice against the same canonical
model and dataset.  The control run may resume online exploration from the
imported profile; the locked run must execute exactly the imported winning
transport width, tape geometry, and optimizer/broadcast-overlap coordinate.
The resulting portable training checkpoints are then compared across model,
optimizer, replay tensors, and replay/session JSON.  The runtime profile sidecar
is deliberately excluded because scheduler evidence is expected to differ.

Run this on a host exposing every physical adapter recorded by the profile.
The profile key remains authoritative for ordered device/driver UUIDs, batch
geometry, token width, and transport identity; the Rust trainer performs its
own independent exact-match validation as well.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "hierarchos-vulkan" / "Cargo.toml"
SCHEMA_VERSION = 1
JOINT_RUNTIME_EXPLORE_EVERY_ENV = "HIERARCHOS_VULKAN_JOINT_RUNTIME_AUTOTUNE_EXPLORE_EVERY"


def _run(
    command: list[str],
    *,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if env_overrides:
        environment.update(env_overrides)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    combined = (completed.stdout or "")
    if completed.stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{combined}"
        )
    return completed


def _load_profile(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported joint-runtime profile schema {payload.get('schema_version')!r}"
        )
    profile_key = payload.get("profile_key")
    winning_arm = payload.get("winning_arm")
    if not isinstance(profile_key, dict) or not isinstance(winning_arm, dict):
        raise ValueError("joint-runtime profile is missing profile_key or winning_arm")
    tape_geometry = winning_arm.get("tape_geometry")
    if not isinstance(tape_geometry, dict):
        raise ValueError("joint-runtime winning arm is missing tape_geometry")
    for name in (
        "gradient_stream_chunk_values",
        "optimizer_broadcast_overlap",
    ):
        if name not in winning_arm:
            raise ValueError(f"joint-runtime winning arm is missing {name}")
    for name in ("sequence_microbatch_size", "state_checkpoint_stride"):
        if name not in tape_geometry:
            raise ValueError(f"joint-runtime tape geometry is missing {name}")
    return payload, profile_key, winning_arm


def _device_catalog(log_path: Path) -> list[dict[str, Any]]:
    completed = _run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(MANIFEST),
            "--bin",
            "hierarchos-vulkan-devices",
        ],
        log_path=log_path,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Vulkan device catalog did not return a JSON list")
    return payload


def _profile_device_indices(
    profile_key: dict[str, Any], catalog: list[dict[str, Any]]
) -> list[int]:
    device_uuids = profile_key.get("device_uuids")
    driver_uuids = profile_key.get("driver_uuids")
    if not isinstance(device_uuids, list) or not isinstance(driver_uuids, list):
        raise ValueError("runtime profile is missing ordered device/driver UUID arrays")
    if len(device_uuids) != len(driver_uuids):
        raise ValueError("runtime profile device/driver UUID arrays have different lengths")
    if len(device_uuids) < 2:
        raise ValueError("joint-runtime A/B qualification requires at least two profile devices")

    selected: list[int] = []
    for position, fingerprint in enumerate(zip(device_uuids, driver_uuids, strict=True)):
        matches = [
            device
            for device in catalog
            if (device.get("device_uuid"), device.get("driver_uuid")) == fingerprint
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "runtime profile device fingerprint did not resolve uniquely at ordered "
                f"position {position}: device_uuid={fingerprint[0]!r} "
                f"driver_uuid={fingerprint[1]!r} matches={len(matches)}"
            )
        selected.append(int(matches[0]["index"]))
    return selected


def _parse_indices(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("--device-indices requires at least two comma-separated indices")
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("--device-indices must contain distinct non-negative indices")
    return indices


def _validate_explicit_indices(
    indices: list[int],
    expected: list[int],
    profile_key: dict[str, Any],
) -> None:
    if indices != expected:
        raise ValueError(
            "--device-indices does not match the runtime profile's ordered UUID topology: "
            f"requested={indices} resolved_profile_order={expected} "
            f"device_uuids={profile_key.get('device_uuids')!r}"
        )


def _dataset_geometry(path: Path) -> tuple[int, int]:
    rows = 0
    maximum_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            input_ids = payload.get("input_ids")
            if not isinstance(input_ids, list) or not input_ids:
                raise ValueError(
                    f"dataset line {line_number} must contain a non-empty input_ids list"
                )
            rows += 1
            maximum_tokens = max(maximum_tokens, len(input_ids))
    if rows == 0:
        raise ValueError("dataset contains no non-empty JSONL rows")
    return rows, maximum_tokens


def _trainer_command(
    *,
    model: Path,
    dataset: Path,
    output: Path,
    runtime_profile: Path,
    device_indices: list[int],
    profile_key: dict[str, Any],
    winning_arm: dict[str, Any],
    epochs: int,
    locked: bool,
) -> list[str]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--manifest-path",
        str(MANIFEST),
        "--bin",
        "hierarchos-vulkan-train",
        "--",
        "--model",
        str(model),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(int(profile_key["batch_size"])),
        "--gradient-accumulation-steps",
        str(int(profile_key["gradient_accumulation_steps"])),
        "--device-indices",
        ",".join(map(str, device_indices)),
        "--gradient-stream-chunk-values",
        str(int(winning_arm["gradient_stream_chunk_values"])),
        "--joint-runtime-profile",
        str(runtime_profile),
        "--no-shuffle",
    ]
    if locked:
        command.append("--lock-joint-runtime-profile")
    return command


def _parse_training_report(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Vulkan trainer stdout was not a single JSON report") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Vulkan trainer report was not a JSON object")
    return payload


def _canonical_arm(arm: dict[str, Any]) -> dict[str, Any]:
    tape = arm.get("tape_geometry")
    if not isinstance(tape, dict):
        raise ValueError("runtime arm is missing tape_geometry")
    return {
        "gradient_stream_chunk_values": int(arm["gradient_stream_chunk_values"]),
        "tape_geometry": {
            "sequence_microbatch_size": int(tape["sequence_microbatch_size"]),
            "state_checkpoint_stride": int(tape["state_checkpoint_stride"]),
        },
        "optimizer_broadcast_overlap": bool(arm["optimizer_broadcast_overlap"]),
    }


def _canonical_report_arm(arm: dict[str, Any]) -> dict[str, Any]:
    if isinstance(arm.get("tape_geometry"), dict):
        return _canonical_arm(arm)
    return {
        "gradient_stream_chunk_values": int(arm["gradient_stream_chunk_values"]),
        "tape_geometry": {
            "sequence_microbatch_size": int(arm["sequence_microbatch_size"]),
            "state_checkpoint_stride": int(arm["state_checkpoint_stride"]),
        },
        "optimizer_broadcast_overlap": bool(arm["optimizer_broadcast_overlap"]),
    }


def _runtime_activity(autotune: dict[str, Any]) -> list[dict[str, Any]]:
    arms = autotune.get("arms")
    if not isinstance(arms, list) or not arms:
        raise AssertionError("joint-runtime report is missing per-arm execution evidence")
    activity: list[dict[str, Any]] = []
    for raw_arm in arms:
        if not isinstance(raw_arm, dict):
            raise AssertionError("joint-runtime report contains a non-object arm")
        selected = int(raw_arm.get("current_run_selected_windows", 0))
        scored = int(raw_arm.get("current_run_scored_windows", 0))
        if selected < 0 or scored < 0 or scored > selected:
            raise AssertionError(
                f"invalid current-run execution counters selected={selected} scored={scored}"
            )
        if selected or scored:
            raw_weights = raw_arm.get("learned_device_workload_weights")
            if raw_weights is None:
                workload_weights: list[float] | None = None
            elif isinstance(raw_weights, list):
                workload_weights = [float(value) for value in raw_weights]
                if any(
                    not math.isfinite(value) or value <= 0.0
                    for value in workload_weights
                ):
                    raise AssertionError(
                        "joint-runtime report contains an invalid learned device workload weight"
                    )
            else:
                raise AssertionError(
                    "joint-runtime report learned_device_workload_weights is not a list"
                )

            raw_devices = raw_arm.get("devices")
            lane_evidence: list[dict[str, Any]] = []
            if raw_devices is not None:
                if not isinstance(raw_devices, list):
                    raise AssertionError("joint-runtime report devices field is not a list")
                for raw_device in raw_devices:
                    if not isinstance(raw_device, dict):
                        raise AssertionError(
                            "joint-runtime report contains a non-object device lane"
                        )
                    lane_evidence.append(
                        {
                            "lane_index": int(raw_device["lane_index"]),
                            "windows": int(raw_device.get("windows", 0)),
                            "tokens_per_second": raw_device.get("tokens_per_second"),
                            "adaptive_tokens_per_second": raw_device.get(
                                "adaptive_tokens_per_second"
                            ),
                            "confidence_throughput_samples": int(
                                raw_device.get("confidence_throughput_samples", 0)
                            ),
                            "confidence_throughput_mean_tokens_per_second": raw_device.get(
                                "confidence_throughput_mean_tokens_per_second"
                            ),
                            "throughput_relative_uncertainty": raw_device.get(
                                "throughput_relative_uncertainty"
                            ),
                            "confidence_adjusted_tokens_per_second": raw_device.get(
                                "confidence_adjusted_tokens_per_second"
                            ),
                            "peak_device_local_usage_ratio": raw_device.get(
                                "peak_device_local_usage_ratio"
                            ),
                            "max_device_local_pressure_bucket": raw_device.get(
                                "max_device_local_pressure_bucket"
                            ),
                            "high_memory_pressure_windows": int(
                                raw_device.get("high_memory_pressure_windows", 0)
                            ),
                            "kernel_gpu_ns_per_token": raw_device.get(
                                "kernel_gpu_ns_per_token"
                            ),
                        }
                    )
                lane_evidence.sort(key=lambda lane: lane["lane_index"])
                if workload_weights is not None and len(workload_weights) != len(
                    lane_evidence
                ):
                    raise AssertionError(
                        "joint-runtime learned workload weights do not match the reported lane count"
                    )
            activity.append(
                {
                    "coordinate": _canonical_report_arm(raw_arm),
                    "selected_windows": selected,
                    "scored_windows": scored,
                    "learned_device_workload_weights": workload_weights,
                    "device_lane_evidence": lane_evidence,
                }
            )
    return activity


def _assert_locked_report(
    report: dict[str, Any], winning_arm: dict[str, Any], expected_selection_windows: int
) -> dict[str, Any]:
    expected = _canonical_arm(winning_arm)
    if report.get("joint_runtime_profile_loaded") is not True:
        raise AssertionError("locked run did not report a loaded joint-runtime profile")
    if report.get("joint_runtime_profile_locked") is not True:
        raise AssertionError("locked run did not report joint-runtime profile locking")
    actual = report.get("joint_runtime_locked_arm")
    if not isinstance(actual, dict) or _canonical_arm(actual) != expected:
        raise AssertionError(
            f"locked run did not execute the imported winning arm: expected={expected} actual={actual}"
        )

    autotune = report.get("joint_runtime_autotune")
    if not isinstance(autotune, dict):
        raise AssertionError("locked run did not emit joint-runtime autotune evidence")
    selectors = autotune.get("phase_selectors")
    if not isinstance(selectors, dict):
        raise AssertionError("locked run is missing factorized phase-selector evidence")
    selected_windows = int(autotune.get("current_run_selected_windows", -1))
    scored_windows = int(autotune.get("current_run_scored_windows", -1))
    selection_steps = int(autotune.get("selection_steps", -1))
    if selected_windows != expected_selection_windows or selection_steps != expected_selection_windows:
        raise AssertionError(
            "locked run did not execute exactly one imported coordinate per optimizer window: "
            f"expected={expected_selection_windows} selected={selected_windows} "
            f"selection_steps={selection_steps}"
        )
    expected_scored_windows = max(0, expected_selection_windows - 1)
    if scored_windows != expected_scored_windows:
        raise AssertionError(
            "locked run scored an unexpected number of windows after its one mandatory dwell: "
            f"expected={expected_scored_windows} actual={scored_windows}"
        )
    activity = _runtime_activity(autotune)
    if sum(item["selected_windows"] for item in activity) != selected_windows:
        raise AssertionError("locked per-arm selected-window evidence does not sum to the run total")
    if sum(item["scored_windows"] for item in activity) != scored_windows:
        raise AssertionError("locked per-arm scored-window evidence does not sum to the run total")
    if not activity or any(item["coordinate"] != expected for item in activity):
        raise AssertionError(
            f"locked run selected a coordinate outside the imported winner: expected={expected} activity={activity}"
        )
    return {
        "selected_windows": selected_windows,
        "scored_windows": scored_windows,
        "executed_coordinates": activity,
        "factorized_recommendation": {
            "gradient_stream_chunk_values": selectors.get(
                "selected_gradient_stream_chunk_values"
            ),
            "optimizer_broadcast_overlap": selectors.get(
                "selected_optimizer_broadcast_overlap"
            ),
            "tape_geometry": selectors.get("selected_tape_geometry"),
        },
    }


def _assert_control_report(
    report: dict[str, Any], winning_arm: dict[str, Any], expected_selection_windows: int
) -> dict[str, Any]:
    if report.get("joint_runtime_profile_loaded") is not True:
        raise AssertionError("control run did not report a loaded joint-runtime profile")
    if report.get("joint_runtime_profile_locked") is not False:
        raise AssertionError("control run unexpectedly locked the imported profile")
    if report.get("joint_runtime_locked_arm") is not None:
        raise AssertionError("control run unexpectedly reported a locked runtime arm")
    autotune = report.get("joint_runtime_autotune")
    if not isinstance(autotune, dict):
        raise AssertionError("control run did not emit joint-runtime autotune evidence")
    selected_windows = int(autotune.get("current_run_selected_windows", -1))
    scored_windows = int(autotune.get("current_run_scored_windows", -1))
    selection_steps = int(autotune.get("selection_steps", -1))
    if selected_windows != expected_selection_windows or selection_steps != expected_selection_windows:
        raise AssertionError(
            "control run did not execute the expected optimizer-window schedule: "
            f"expected={expected_selection_windows} selected={selected_windows} "
            f"selection_steps={selection_steps}"
        )
    activity = _runtime_activity(autotune)
    if sum(item["selected_windows"] for item in activity) != selected_windows:
        raise AssertionError("control per-arm selected-window evidence does not sum to the run total")
    if sum(item["scored_windows"] for item in activity) != scored_windows:
        raise AssertionError("control per-arm scored-window evidence does not sum to the run total")
    expected = _canonical_arm(winning_arm)
    if not any(item["coordinate"] == expected for item in activity):
        raise AssertionError("control run never executed the imported winning coordinate")
    explored = [
        item
        for item in activity
        if item["coordinate"] != expected and item["scored_windows"] > 0
    ]
    if not explored:
        raise AssertionError(
            "control run never scored a non-winning coordinate; increase dataset/epochs or reduce "
            "--control-explore-every so this is a real scheduler A/B qualification"
        )
    return {
        "selected_windows": selected_windows,
        "scored_windows": scored_windows,
        "executed_coordinates": activity,
        "non_winning_scored_coordinates": explored,
    }


def _compare_safetensors(lhs: Path, rhs: Path) -> dict[str, Any]:
    maximum = 0.0
    maximum_tensor: str | None = None
    tensor_count = 0
    with safe_open(str(lhs), framework="pt", device="cpu") as lhs_handle, safe_open(
        str(rhs), framework="pt", device="cpu"
    ) as rhs_handle:
        lhs_keys = set(lhs_handle.keys())
        rhs_keys = set(rhs_handle.keys())
        if lhs_keys != rhs_keys:
            raise AssertionError(
                f"SafeTensors registry differs for {lhs.name}: "
                f"missing={sorted(lhs_keys - rhs_keys)} extra={sorted(rhs_keys - lhs_keys)}"
            )
        for name in sorted(lhs_keys):
            lhs_tensor = lhs_handle.get_tensor(name)
            rhs_tensor = rhs_handle.get_tensor(name)
            tensor_count += 1
            if lhs_tensor.shape != rhs_tensor.shape or lhs_tensor.dtype != rhs_tensor.dtype:
                raise AssertionError(
                    f"SafeTensors metadata differs for {lhs.name}:{name}: "
                    f"{lhs_tensor.shape}/{lhs_tensor.dtype} vs {rhs_tensor.shape}/{rhs_tensor.dtype}"
                )
            if lhs_tensor.is_floating_point():
                diff = float(
                    (lhs_tensor.float() - rhs_tensor.float()).abs().max().item()
                )
                if not math.isfinite(diff):
                    raise AssertionError(f"non-finite checkpoint diff for {lhs.name}:{name}")
                if diff > maximum:
                    maximum = diff
                    maximum_tensor = name
            elif not torch.equal(lhs_tensor, rhs_tensor):
                raise AssertionError(f"non-floating tensor differs for {lhs.name}:{name}")
    return {
        "file": lhs.name,
        "tensor_count": tensor_count,
        "max_abs_diff": maximum,
        "max_abs_diff_tensor": maximum_tensor,
    }


def _compare_json(lhs: Path, rhs: Path) -> dict[str, Any]:
    left = json.loads(lhs.read_text(encoding="utf-8"))
    right = json.loads(rhs.read_text(encoding="utf-8"))
    if left != right:
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left) | set(right))
            differing = [key for key in keys if left.get(key) != right.get(key)]
            detail = f" differing_top_level_keys={differing}"
        else:
            detail = ""
        raise AssertionError(f"portable checkpoint JSON differs for {lhs.name}.{detail}")
    return {"file": lhs.name, "semantic_equal": True}


def _checkpoint_parity(control: Path, locked: Path, atol: float) -> dict[str, Any]:
    required_tensors = ["model.safetensors", "optimizer.safetensors"]
    optional_tensors = ["gradients.safetensors", "training_replay.safetensors"]
    tensor_comparisons: list[dict[str, Any]] = []
    for filename in required_tensors:
        lhs = control / filename
        rhs = locked / filename
        if not lhs.is_file() or not rhs.is_file():
            raise AssertionError(f"A/B checkpoint is missing required {filename}")
        result = _compare_safetensors(lhs, rhs)
        if result["max_abs_diff"] > atol:
            raise AssertionError(
                f"locked scheduler changed {filename} by {result['max_abs_diff']:.9g} "
                f"> atol={atol:.9g} at tensor {result['max_abs_diff_tensor']!r}"
            )
        tensor_comparisons.append(result)
    for filename in optional_tensors:
        lhs = control / filename
        rhs = locked / filename
        if lhs.is_file() != rhs.is_file():
            raise AssertionError(
                f"portable checkpoint optional tensor presence differs for {filename}"
            )
        if lhs.is_file():
            result = _compare_safetensors(lhs, rhs)
            if result["max_abs_diff"] > atol:
                raise AssertionError(
                    f"locked scheduler changed {filename} by {result['max_abs_diff']:.9g} "
                    f"> atol={atol:.9g} at tensor {result['max_abs_diff_tensor']!r}"
                )
            tensor_comparisons.append(result)

    json_comparisons: list[dict[str, Any]] = []
    for filename in ("training_state.json", "training_replay.json"):
        lhs = control / filename
        rhs = locked / filename
        if not lhs.is_file() or not rhs.is_file():
            raise AssertionError(f"A/B checkpoint is missing required portable state {filename}")
        json_comparisons.append(_compare_json(lhs, rhs))
    return {
        "atol": atol,
        "tensor_files": tensor_comparisons,
        "json_files": json_comparisons,
        "excluded_runtime_only_files": ["vulkan_joint_runtime_profile.v1.json"],
        "max_abs_diff": max(result["max_abs_diff"] for result in tensor_comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--device-indices",
        default=None,
        help="ordered physical-device indices; defaults to exact UUID+driver resolution from the profile",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--control-explore-every",
        type=int,
        default=1,
        help=(
            "factorized scheduler exploration cadence for the unlocked control; "
            "qualification defaults to 1 so a short run still exercises a non-winning coordinate"
        ),
    )
    parser.add_argument(
        "--checkpoint-atol",
        type=float,
        default=8.0e-6,
        help="maximum scheduler-neutral floating checkpoint difference (default: 8e-6)",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.control_explore_every <= 0:
        parser.error("--control-explore-every must be positive")
    if not math.isfinite(args.checkpoint_atol) or args.checkpoint_atol < 0.0:
        parser.error("--checkpoint-atol must be finite and non-negative")
    args.model = args.model.resolve()
    args.dataset = args.dataset.resolve()
    args.runtime_profile = args.runtime_profile.resolve()
    args.output_root = args.output_root.resolve()
    for label, path in (
        ("--model", args.model),
        ("--dataset", args.dataset),
        ("--runtime-profile", args.runtime_profile),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")

    try:
        _, profile_key, winning_arm = _load_profile(args.runtime_profile)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(f"invalid --runtime-profile: {exc}")

    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(
            f"--output-root must be absent or empty so A/B outputs cannot overwrite prior evidence: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"

    catalog = _device_catalog(log_dir / "device-catalog.log")
    resolved_indices = _profile_device_indices(profile_key, catalog)
    if args.device_indices is None:
        device_indices = resolved_indices
    else:
        try:
            device_indices = _parse_indices(args.device_indices)
            _validate_explicit_indices(device_indices, resolved_indices, profile_key)
        except ValueError as exc:
            parser.error(str(exc))

    rows, maximum_tokens = _dataset_geometry(args.dataset)
    expected_tokens = int(profile_key["tokens_per_sequence"])
    if maximum_tokens != expected_tokens:
        parser.error(
            "dataset maximum token width does not match the runtime profile key: "
            f"dataset={maximum_tokens} profile={expected_tokens}"
        )
    batch_size = int(profile_key["batch_size"])
    accumulation_steps = int(profile_key["gradient_accumulation_steps"])
    complete_batches = rows // batch_size
    if complete_batches < accumulation_steps:
        parser.error(
            "dataset is too small to execute one complete optimizer window under the profile key: "
            f"rows={rows} batch_size={batch_size} accumulation_steps={accumulation_steps}"
        )
    optimizer_windows_per_epoch = (
        complete_batches + accumulation_steps - 1
    ) // accumulation_steps
    expected_selection_windows = optimizer_windows_per_epoch * args.epochs
    # The factorized selector advances one factor every selection and defines an
    # exploration round as four factor selections.  A switched coordinate gets
    # one unscored dwell and then a forced follow-up scoring window.  Requiring
    # this many windows makes the unlocked side a real scheduler A/B rather than
    # merely replaying the imported winner.
    minimum_ab_windows = 4 * args.control_explore_every + 2
    if expected_selection_windows < minimum_ab_windows:
        parser.error(
            "dataset/epochs are too small for a guaranteed unlocked exploration + scored follow-up: "
            f"optimizer_windows={expected_selection_windows} required={minimum_ab_windows} "
            f"control_explore_every={args.control_explore_every}"
        )

    control_dir = args.output_root / "profile-unlocked"
    locked_dir = args.output_root / "profile-locked"
    control_command = _trainer_command(
        model=args.model,
        dataset=args.dataset,
        output=control_dir,
        runtime_profile=args.runtime_profile,
        device_indices=device_indices,
        profile_key=profile_key,
        winning_arm=winning_arm,
        epochs=args.epochs,
        locked=False,
    )
    locked_command = _trainer_command(
        model=args.model,
        dataset=args.dataset,
        output=locked_dir,
        runtime_profile=args.runtime_profile,
        device_indices=device_indices,
        profile_key=profile_key,
        winning_arm=winning_arm,
        epochs=args.epochs,
        locked=True,
    )

    control_report = _parse_training_report(
        _run(
            control_command,
            log_path=log_dir / "profile-unlocked.log",
            env_overrides={
                JOINT_RUNTIME_EXPLORE_EVERY_ENV: str(args.control_explore_every),
            },
        )
    )
    control_evidence = _assert_control_report(
        control_report, winning_arm, expected_selection_windows
    )
    locked_report = _parse_training_report(
        _run(locked_command, log_path=log_dir / "profile-locked.log")
    )
    locked_evidence = _assert_locked_report(
        locked_report, winning_arm, expected_selection_windows
    )
    parity = _checkpoint_parity(control_dir, locked_dir, args.checkpoint_atol)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_profile": str(args.runtime_profile),
        "model": str(args.model),
        "dataset": str(args.dataset),
        "dataset_rows": rows,
        "optimizer_windows_per_epoch": optimizer_windows_per_epoch,
        "expected_selection_windows": expected_selection_windows,
        "profile_key": profile_key,
        "imported_winning_arm": _canonical_arm(winning_arm),
        "resolved_device_indices": device_indices,
        "control": {
            "mode": "imported-profile-online-exploration",
            "optimizer_step": control_report.get("optimizer_step"),
            "joint_runtime_profile_loaded": control_report.get(
                "joint_runtime_profile_loaded"
            ),
            "joint_runtime_profile_locked": control_report.get(
                "joint_runtime_profile_locked"
            ),
            "explore_every": args.control_explore_every,
            "execution_evidence": control_evidence,
        },
        "locked": {
            "mode": "exact-imported-winning-arm",
            "optimizer_step": locked_report.get("optimizer_step"),
            "reported_locked_arm": locked_report.get("joint_runtime_locked_arm"),
            "joint_runtime_profile_loaded": locked_report.get(
                "joint_runtime_profile_loaded"
            ),
            "joint_runtime_profile_locked": locked_report.get(
                "joint_runtime_profile_locked"
            ),
            "execution_evidence": locked_evidence,
        },
        "checkpoint_parity": parity,
        "logs": str(log_dir),
        "contract": {
            "locked_coordinates": [
                "gradient_stream_chunk_values",
                "tape_geometry.sequence_microbatch_size",
                "tape_geometry.state_checkpoint_stride",
                "optimizer_broadcast_overlap",
            ],
            "heterogeneous_shard_evidence": [
                "learned_device_workload_weights",
                "device_lane_evidence.confidence_throughput_samples",
                "device_lane_evidence.throughput_relative_uncertainty",
                "device_lane_evidence.confidence_adjusted_tokens_per_second",
                "device_lane_evidence.peak_device_local_usage_ratio",
                "device_lane_evidence.kernel_gpu_ns_per_token",
            ],
            "checkpoint_abi": "canonical-safetensors-pytorch-framework-readable",
            "scheduler_math_requirement": "locked-vs-unlocked-checkpoint-neutral",
        },
    }
    report_path = args.output_root / "joint_runtime_profile_ab_qualification.v1.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Vulkan joint-runtime imported-profile A/B qualification: PASS")
    print(f"report={report_path}")
    print(f"checkpoint_max_abs_diff={parity['max_abs_diff']:.9g}")


if __name__ == "__main__":
    main()
