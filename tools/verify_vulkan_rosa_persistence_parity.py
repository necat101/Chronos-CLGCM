#!/usr/bin/env python3
"""Verify bounded ROSA persistent Vulkan state across arbitrary TBPTT chunks."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from hierarchos.utils.rosa import ROSAState, _rosa_incremental
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


def _chunk(values: list[int], widths: list[int]) -> list[list[int]]:
    chunks: list[list[int]] = []
    offset = 0
    for width in widths:
        chunks.append(values[offset : offset + width])
        offset += width
    if offset < len(values):
        chunks.append(values[offset:])
    return chunks


def main() -> None:
    random.seed(20260814)
    torch.manual_seed(20260814)
    config = tiny_coherent_config(32)
    # Exercise the production coherent-v9 bound so the private Vulkan
    # match-state packing is covered at its intended 16-bit logical width.
    config.rosa_max_context = 512
    model = HierarchosCore(config).eval()
    if not config.enforce_rosa_max_context or config.rosa_max_context != 512:
        raise AssertionError("coherent fixture must resolve bounded ROSA cap=512")

    # Cross the full default-width segment using awkward caller chunk boundaries
    # so persistence, automatic rollover, and the packed generation stride are
    # all exercised together.
    prefix = [2, 5, 2, 5, 2, 5, 7, 2, 5, 2, 5]
    tail = [random.randrange(2, 12) for _ in range(518)]
    tokens = prefix + tail
    chunks = _chunk(tokens, [1, 4, 17, 63, 129, 157])
    reset_tokens = [3, 4, 3, 4, 3, 4, 9, 3, 4]
    lane_tokens = [
        [2, 7],
        [5, 8],
        [2, 7],
        [5, 8],
        [2, 7],
        [5, 8],
        [9, 7],
        [2, 8],
        [5, 7],
        [2, 8],
        [5, 7],
        [2, 8],
    ]
    lane_resets = [[0, 0] for _ in lane_tokens]
    lane_resets[0] = [1, 1]
    lane_resets[6] = [0, 1]
    lane_resets[9] = [1, 0]

    expected_state = ROSAState.new()
    expected_chunks = [
        _rosa_incremental(expected_state, chunk, config.rosa_max_context)
        for chunk in chunks
    ]
    expected_after_reset = _rosa_incremental(
        ROSAState.new(), reset_tokens, config.rosa_max_context
    )
    lane_states = [ROSAState.new(), ROSAState.new()]
    expected_lane_predictions: list[list[int]] = []
    for step_tokens, step_resets in zip(lane_tokens, lane_resets, strict=True):
        step_predictions: list[int] = []
        for lane, (token, reset) in enumerate(
            zip(step_tokens, step_resets, strict=True)
        ):
            if reset:
                lane_states[lane] = ROSAState.new()
            step_predictions.append(
                _rosa_incremental(
                    lane_states[lane], [token], config.rosa_max_context
                )[0]
            )
        expected_lane_predictions.append(step_predictions)

    case = {
        "chunks": chunks,
        "reset_tokens": reset_tokens,
        "lane_tokens": lane_tokens,
        "lane_resets": lane_resets,
    }
    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-rosa-persistence-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        case_path = temp / "case.json"
        export_model(model, config, model_dir)
        case_path.write_text(json.dumps(case), encoding="utf-8")
        result = json.loads(
            _run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--release",
                    "--manifest-path",
                    str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                    "--bin",
                    "hierarchos-vulkan-rosa-persistence-parity",
                    "--",
                    "--model",
                    str(model_dir),
                    "--case",
                    str(case_path),
                ]
            ).stdout
        )

    if result["predictions"] != expected_chunks:
        raise AssertionError(
            f"persistent Vulkan ROSA mismatch\nexpected={expected_chunks}\nactual={result['predictions']}"
        )
    if result["after_reset"] != expected_after_reset:
        raise AssertionError(
            f"explicit reset mismatch expected={expected_after_reset} actual={result['after_reset']}"
        )
    if result["lane_predictions"] != expected_lane_predictions:
        raise AssertionError(
            "independent lane mismatch\n"
            f"expected={expected_lane_predictions}\n"
            f"actual={result['lane_predictions']}"
        )

    flat_expected = [value for chunk in expected_chunks for value in chunk]
    flat_actual = [value for chunk in result["predictions"] for value in chunk]
    print(f"device={result['device']}")
    print(
        "rosa_kernel="
        f"{result['rosa_kernel_label']} "
        f"width={result['rosa_workgroup_size']} "
        f"autotuned={result['rosa_autotuned']}"
    )
    print(f"tokens={len(tokens)}")
    print(f"chunks={[len(chunk) for chunk in chunks]}")
    print(f"automatic_segment_resets={len(tokens) // config.rosa_max_context}")
    print(f"prediction_exact={flat_actual == flat_expected}")
    print(f"explicit_reset_exact={result['after_reset'] == expected_after_reset}")
    print(f"lane_steps={len(lane_tokens)} lanes={len(lane_tokens[0])}")
    print(f"lane_prediction_exact={result['lane_predictions'] == expected_lane_predictions}")
    print("Persistent bounded ROSA Vulkan/Python parity: PASS")


if __name__ == "__main__":
    main()
