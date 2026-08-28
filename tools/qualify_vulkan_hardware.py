#!/usr/bin/env python3
"""Collect a reproducible Hierarchos Vulkan hardware qualification certificate.

The certificate intentionally separates hardware/runtime measurements from the
portable model ABI. Microprofiles are tied to a Vulkan physical-device index;
the parity leg independently proves that the same canonical SafeTensors path is
interchangeable with PyTorch and native Rust inference (and CUDA when present).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "hierarchos-vulkan" / "Cargo.toml"
DEVICE_INDEX_ENV = "HIERARCHOS_VULKAN_MICROPROFILE_DEVICE_INDEX"
PROFILE_KERNELS_ENV = "HIERARCHOS_VULKAN_PROFILE_KERNELS"
QUALIFICATION_DEVICE_INDEX_ENV = "HIERARCHOS_VULKAN_QUALIFICATION_DEVICE_INDEX"
SCHEMA_VERSION = 1


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    combined = ""
    if completed.stdout:
        combined += completed.stdout
    if completed.stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += completed.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{combined}"
        )
    return completed


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stdout or "") + "\n" + (completed.stderr or "")


def _device_catalog() -> list[dict[str, object]]:
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
        ]
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Vulkan device catalog did not return a JSON list")
    return payload


def _selected_device(catalog: list[dict[str, object]], index: int) -> dict[str, object]:
    for device in catalog:
        if device.get("index") == index:
            return device
    available = ", ".join(str(device.get("index")) for device in catalog) or "none"
    raise RuntimeError(f"Vulkan physical-device index {index} is unavailable; found {available}")


def _runtime_profile_summary(path: Path, selected_device: dict[str, object]) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported joint-runtime profile schema {payload.get('schema_version')!r}"
        )
    profile_key = payload.get("profile_key")
    winning_arm = payload.get("winning_arm")
    if not isinstance(profile_key, dict) or not isinstance(winning_arm, dict):
        raise RuntimeError("joint-runtime profile is missing profile_key or winning_arm")
    device_uuids = profile_key.get("device_uuids")
    driver_uuids = profile_key.get("driver_uuids")
    if not isinstance(device_uuids, list) or not isinstance(driver_uuids, list):
        raise RuntimeError("joint-runtime profile is missing device/driver UUID arrays")
    if len(device_uuids) != len(driver_uuids):
        raise RuntimeError("joint-runtime profile device/driver UUID arrays differ in length")
    fingerprint = (selected_device.get("device_uuid"), selected_device.get("driver_uuid"))
    topology = list(zip(device_uuids, driver_uuids, strict=True))
    return {
        "path": str(path.resolve()),
        "profile_key": profile_key,
        "winning_arm": winning_arm,
        "selected_device_member_match": fingerprint in topology,
        "topology_device_count": len(topology),
    }


def _parse_width448(output: str) -> dict[str, object]:
    header = re.search(
        r"Hierarchos LM width448 microprofile device_index=(\d+) device=(.+?) "
        r"subgroup=(\d+) shared=(\d+)B rows=(\d+) vocab=(\d+)",
        output,
    )
    if header is None:
        raise RuntimeError("width448 microprofile output did not contain its hardware header")
    arms: dict[str, float] = {}
    for match in re.finditer(r"(?m)^\s{2}(.+)=([0-9]+(?:\.[0-9]+)?)ms\s*$", output):
        arms[match.group(1)] = float(match.group(2))
    selector = re.search(r"Hierarchos LM width448 selector selected=(.+)", output)
    if not arms or selector is None:
        raise RuntimeError("width448 microprofile output did not contain measured arms/selector")
    fastest_name, fastest_ms = min(arms.items(), key=lambda item: item[1])
    return {
        "device_index": int(header.group(1)),
        "device_name": header.group(2),
        "subgroup_size": int(header.group(3)),
        "shared_memory_bytes": int(header.group(4)),
        "rows": int(header.group(5)),
        "vocab_size": int(header.group(6)),
        "measured_ms": arms,
        "fastest_measured_arm": fastest_name,
        "fastest_measured_ms": fastest_ms,
        "selector": selector.group(1).strip(),
    }


def _parse_seam(output: str) -> dict[str, object]:
    match = re.search(
        r"Hierarchos LM rows16 forward->stats seam device_index=(\d+) device=(.+?) "
        r"rows=(\d+) vocab=(\d+) tile_partials=(\d+)B repetitions=(\d+) "
        r"serial_host_pair_ms=([0-9.eE+-]+) dot4_host_pair_ms=([0-9.eE+-]+) "
        r"logit_max_abs=([0-9.eE+-]+) stats_max_abs=([0-9.eE+-]+)",
        output,
    )
    if match is None:
        raise RuntimeError("rows16 seam microprofile output did not contain its summary")
    serial_ms = float(match.group(7))
    dot4_ms = float(match.group(8))
    return {
        "device_index": int(match.group(1)),
        "device_name": match.group(2),
        "rows": int(match.group(3)),
        "vocab_size": int(match.group(4)),
        "tile_partials_bytes": int(match.group(5)),
        "repetitions": int(match.group(6)),
        "serial_host_pair_ms": serial_ms,
        "dot4_host_pair_ms": dot4_ms,
        "dot4_speedup": serial_ms / dot4_ms,
        "logit_max_abs": float(match.group(9)),
        "stats_max_abs": float(match.group(10)),
    }


def _microprofile_env(device_index: int) -> dict[str, str]:
    env = os.environ.copy()
    env[DEVICE_INDEX_ENV] = str(device_index)
    return env


def _run_microprofiles(
    *,
    device_index: int,
    width_vocab: int,
    seam_vocab: int,
    seam_repetitions: int,
    log_dir: Path,
) -> dict[str, object]:
    env = _microprofile_env(device_index)
    env["HIERARCHOS_VULKAN_LM_MICROPROFILE_VOCAB_SIZE"] = str(width_vocab)
    width_command = [
        "cargo",
        "test",
        "--manifest-path",
        str(MANIFEST),
        "lm_width448_backward_topology_microprofile",
        "--lib",
        "--release",
        "--",
        "--ignored",
        "--nocapture",
    ]
    width = _run(width_command, env=env, log_path=log_dir / "width448.log")

    seam_env = _microprofile_env(device_index)
    seam_env[PROFILE_KERNELS_ENV] = "1"
    seam_env["HIERARCHOS_VULKAN_LM_SEAM_PROFILE_VOCAB_SIZE"] = str(seam_vocab)
    seam_env["HIERARCHOS_VULKAN_LM_SEAM_PROFILE_REPETITIONS"] = str(seam_repetitions)
    seam_command = [
        "cargo",
        "test",
        "--manifest-path",
        str(MANIFEST),
        "lm_rows16_forward_stats_seam_microprofile",
        "--lib",
        "--release",
        "--",
        "--ignored",
        "--nocapture",
    ]
    seam = _run(seam_command, env=seam_env, log_path=log_dir / "rows16-seam.log")
    return {
        "width448_backward": _parse_width448(_combined_output(width)),
        "rows16_forward_stats_seam": _parse_seam(_combined_output(seam)),
    }


def _run_parity(
    *,
    device_index: int,
    runtime_profile: Path | None,
    report_path: Path,
    log_path: Path,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "verify_vulkan_labeled_sequence_parity.py"),
        "--device-index",
        str(device_index),
        "--budgeted-windows",
        "--report-json",
        str(report_path),
    ]
    if runtime_profile is not None:
        command.extend(
            [
                "--runtime-profile",
                str(runtime_profile),
                "--require-runtime-profile-device-match",
            ]
        )
    _run(command, log_path=log_path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_trajectory(*, cycles: int, require_cuda: bool, log_path: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "verify_vulkan_cuda_vulkan_trajectory.py"),
        "--cycles",
        str(cycles),
    ]
    command.append("--require-cuda" if require_cuda else "--cpu-fallback")
    completed = _run(command, log_path=log_path)
    output = _combined_output(completed)
    if "PASS" not in output:
        raise RuntimeError("cross-backend trajectory command completed without a PASS marker")
    return {
        "cycles": cycles,
        "cuda_required": require_cuda,
        "status": "passed",
        "mode": "cuda" if require_cuda else "cpu-fallback-when-needed",
    }


def _run_numerical_safety(*, device_index: int, log_dir: Path) -> dict[str, object]:
    env = os.environ.copy()
    env[QUALIFICATION_DEVICE_INDEX_ENV] = str(device_index)
    checks = [
        (
            "finite_clamp_boundary_and_nonfinite",
            [
                "cargo",
                "test",
                "--release",
                "--manifest-path",
                str(MANIFEST),
                "finite_clamp_matches_pytorch_boundaries_and_preserves_nonfinite_branch",
                "--lib",
            ],
        ),
        (
            "optimizer_gradient_numerics",
            [
                "cargo",
                "test",
                "--release",
                "--manifest-path",
                str(MANIFEST),
                "training_numerics::tests",
                "--lib",
            ],
        ),
        (
            "labeled_objective_backward_cap_policy",
            [
                "cargo",
                "test",
                "--release",
                "--manifest-path",
                str(MANIFEST),
                "labeled_sequence_objective_matches_pytorch_backward_cap_policy",
                "--lib",
            ],
        ),
        (
            "native_resume_numerical_policy",
            [
                "cargo",
                "test",
                "--release",
                "--manifest-path",
                str(MANIFEST),
                "resume_numerical_policy_scalar_rejects_safety_drift",
                "--bin",
                "hierarchos-vulkan-train",
            ],
        ),
        (
            "rwkv_decay_clamp_parity",
            [sys.executable, str(ROOT / "tools" / "verify_vulkan_rwkv_matrix_state_parity.py")],
        ),
        (
            "rwkv_channel_mix_clamp_parity",
            [sys.executable, str(ROOT / "tools" / "verify_vulkan_rwkv_channel_mix_parity.py")],
        ),
        (
            "rwkv_packed_state_clamp_parity",
            [sys.executable, str(ROOT / "tools" / "verify_vulkan_rwkv_packed_state_parity.py")],
        ),
        (
            "token_frontend_activation_clamp_parity",
            [
                sys.executable,
                str(ROOT / "tools" / "verify_vulkan_token_frontend_parity.py"),
                "--clamp-stress",
            ],
        ),
        (
            "token_memory_frontend_clamp_parity",
            [
                sys.executable,
                str(ROOT / "tools" / "verify_vulkan_token_memory_frontend_parity.py"),
                "--clamp-stress",
            ],
        ),
        (
            "manager_control_clamp_parity",
            [sys.executable, str(ROOT / "tools" / "verify_vulkan_control_parity.py")],
        ),
        (
            "worker_refinement_clamp_parity",
            [
                sys.executable,
                str(ROOT / "tools" / "verify_vulkan_worker_refinement_loss_parity.py"),
                "--precision",
                "fp32",
            ],
        ),
    ]
    passed = []
    for name, command in checks:
        completed = _run(command, env=env, log_path=log_dir / f"safety-{name}.log")
        output = _combined_output(completed)
        if "test result: ok" not in output and "PASS" not in output:
            raise RuntimeError(f"numerical-safety check {name!r} completed without a PASS marker")
        passed.append(name)

    # The isolated norm/clamp tests above prove the primitives. Exercise the
    # same boundary inside a real optimizer window as well: six tokens span two
    # historical PyTorch TBPTT chunks, two updates accumulate into one AdamW
    # step, and the deliberately tiny clip threshold guarantees that clipping
    # is active rather than merely configured. The verifier also checks the
    # post-update native-Rust inference package against the PyTorch checkpoint.
    active_clip_report = log_dir / "safety-full-graph-active-gradient-clip.json"
    active_clip_name = "full_graph_rwkv_active_gradient_clip_parity"
    active_clip = _run(
        [
            sys.executable,
            str(ROOT / "tools" / "verify_vulkan_labeled_sequence_parity.py"),
            "--device-index",
            str(device_index),
            "--update-count",
            "2",
            "--accumulation-steps",
            "2",
            "--grad-clip",
            "0.01",
            "--report-json",
            str(active_clip_report),
        ],
        env=env,
        log_path=log_dir / f"safety-{active_clip_name}.log",
    )
    active_clip_output = _combined_output(active_clip)
    active_clip_payload = json.loads(active_clip_report.read_text(encoding="utf-8"))
    if active_clip_payload.get("status") != "passed":
        raise RuntimeError(
            f"numerical-safety check {active_clip_name!r} did not produce a passed report"
        )
    gradient_safety = active_clip_payload.get("gradient_safety")
    if not isinstance(gradient_safety, dict):
        raise RuntimeError(
            f"numerical-safety check {active_clip_name!r} omitted gradient_safety evidence"
        )
    clip_coefficients = gradient_safety.get("vulkan_window_clip_coefficients")
    vulkan_norms = gradient_safety.get("vulkan_window_global_l2_norms")
    pytorch_norms = gradient_safety.get("pytorch_window_global_l2_norms")
    if (
        not isinstance(clip_coefficients, list)
        or not clip_coefficients
        or not all(isinstance(value, (int, float)) for value in clip_coefficients)
        or not any(0.0 <= float(value) < 0.999999 for value in clip_coefficients)
    ):
        raise RuntimeError(
            f"numerical-safety check {active_clip_name!r} never exercised active clipping: "
            f"{clip_coefficients!r}"
        )
    for backend, norms in (("Vulkan", vulkan_norms), ("PyTorch", pytorch_norms)):
        if (
            not isinstance(norms, list)
            or not norms
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in norms
            )
        ):
            raise RuntimeError(
                f"numerical-safety check {active_clip_name!r} has invalid {backend} norms: "
                f"{norms!r}"
            )
    if "native_inference_max_abs_diff=" not in active_clip_output:
        raise RuntimeError(
            f"numerical-safety check {active_clip_name!r} omitted native-Rust inference parity"
        )
    passed.append(active_clip_name)
    return {
        "status": "passed",
        "device_index": device_index,
        "device_selection_env": QUALIFICATION_DEVICE_INDEX_ENV,
        "checks": passed,
        "active_gradient_clip": {
            "threshold": 0.01,
            "vulkan_window_global_l2_norms": vulkan_norms,
            "pytorch_window_global_l2_norms": pytorch_norms,
            "vulkan_window_clip_coefficients": clip_coefficients,
            "report": str(active_clip_report),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--runtime-profile", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="qualification JSON (default: benchmark_results/vulkan_hardware_qualification.v1.deviceN.json)",
    )
    parser.add_argument("--skip-microprofiles", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument(
        "--skip-numerical-safety",
        action="store_true",
        help="skip clamp/non-finite/gradient-norm safety qualification (not recommended for release certificates)",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--trajectory-cycles",
        type=int,
        default=0,
        help="also exercise repeated Vulkan <-> PyTorch training-state handoffs; 0 disables",
    )
    parser.add_argument("--width448-vocab-size", type=int, default=50_257)
    parser.add_argument("--seam-vocab-size", type=int, default=50_257)
    parser.add_argument("--seam-repetitions", type=int, default=32)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short diagnostic geometry (vocab 2048, seam repetitions 8); parity remains full",
    )
    args = parser.parse_args()
    if args.device_index < 0:
        parser.error("--device-index must be non-negative")
    if args.trajectory_cycles < 0:
        parser.error("--trajectory-cycles must be non-negative")
    for name, value in (
        ("--width448-vocab-size", args.width448_vocab_size),
        ("--seam-vocab-size", args.seam_vocab_size),
        ("--seam-repetitions", args.seam_repetitions),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.require_cuda and args.skip_parity and args.trajectory_cycles == 0:
        parser.error("--require-cuda needs the parity leg or --trajectory-cycles")

    if args.quick:
        args.width448_vocab_size = 2_048
        args.seam_vocab_size = 2_048
        args.seam_repetitions = 8

    output = args.output or (
        ROOT
        / "benchmark_results"
        / f"vulkan_hardware_qualification.v1.device{args.device_index}.json"
    )
    output = output.resolve()
    log_dir = output.with_suffix("").with_name(output.stem + ".logs")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    catalog = _device_catalog()
    selected_device = _selected_device(catalog, args.device_index)
    runtime_profile = None
    if args.runtime_profile is not None:
        args.runtime_profile = args.runtime_profile.resolve()
        runtime_profile = _runtime_profile_summary(args.runtime_profile, selected_device)
        if not runtime_profile["selected_device_member_match"]:
            raise RuntimeError(
                "selected Vulkan device UUID+driver UUID is not a member of --runtime-profile"
            )

    numerical_safety = None
    if not args.skip_numerical_safety:
        numerical_safety = _run_numerical_safety(
            device_index=args.device_index,
            log_dir=log_dir,
        )

    microprofiles = None
    if not args.skip_microprofiles:
        microprofiles = _run_microprofiles(
            device_index=args.device_index,
            width_vocab=args.width448_vocab_size,
            seam_vocab=args.seam_vocab_size,
            seam_repetitions=args.seam_repetitions,
            log_dir=log_dir,
        )

    parity = None
    parity_report_path = log_dir / "pytorch-vulkan-native-parity.json"
    if not args.skip_parity:
        parity = _run_parity(
            device_index=args.device_index,
            runtime_profile=args.runtime_profile,
            report_path=parity_report_path,
            log_path=log_dir / "pytorch-vulkan-native-parity.log",
        )
        cuda_status = parity.get("parity", {}).get("cuda_status")
        if args.require_cuda and cuda_status != "passed":
            raise RuntimeError(f"CUDA inference qualification required, got {cuda_status!r}")

    trajectory = None
    if args.trajectory_cycles:
        trajectory = _run_trajectory(
            cycles=args.trajectory_cycles,
            require_cuda=args.require_cuda,
            log_path=log_dir / "cross-backend-trajectory.log",
        )

    certificate = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vulkan_device": selected_device,
        "numerical_safety": numerical_safety,
        "runtime_profile": runtime_profile,
        "microprofiles": microprofiles,
        "portable_checkpoint_parity": parity,
        "cross_backend_trajectory": trajectory,
        "logs": str(log_dir),
        "contract": {
            "model_checkpoint_abi": "canonical-safetensors-pytorch-row-major",
            "runtime_profile_scope": "scheduler-only-no-model-tensor-layout-change",
            "locked_profile_replay_flag": "--lock-joint-runtime-profile",
            "microprofile_device_env": DEVICE_INDEX_ENV,
            "qualification_device_env": QUALIFICATION_DEVICE_INDEX_ENV,
        },
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Vulkan hardware qualification: PASS device={selected_device.get('name')}")
    print(f"qualification_report={output}")
    print(f"qualification_logs={log_dir}")


if __name__ == "__main__":
    main()
