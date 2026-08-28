#!/usr/bin/env python3
"""PyTorch parity for device-resident Vulkan H pondering + hard-ACT selection."""

from __future__ import annotations

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
from hierarchos.models.act import hard_act_selection
from hierarchos.models.core import _finite_clamp
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config


def assert_close(label: str, actual: list[float], expected: torch.Tensor) -> float:
    actual_tensor = torch.tensor(actual, dtype=torch.float32).reshape_as(expected)
    expected = expected.detach().float()
    diff = float((actual_tensor - expected).abs().max().item()) if expected.numel() else 0.0
    try:
        torch.testing.assert_close(actual_tensor, expected, rtol=3.0e-4, atol=3.0e-5)
    except AssertionError as exc:
        raise AssertionError(f"{label} parity failed; max_abs_diff={diff:.9g}\n{exc}") from exc
    return diff


def main() -> None:
    torch.manual_seed(20260827)
    config = tiny_coherent_config(32)
    config.max_h_steps = 4
    config.min_h_steps = 2
    # A lower threshold makes different rows more likely to select different
    # candidates while still exercising cumulative-hazard semantics.
    config.h_halt_thresh = 0.62
    model = HierarchosCore(config).eval()

    batch = 3
    steps = config.max_h_steps
    residual = torch.randn(batch, config.h_hidden) * 0.09
    initial_state = torch.randn(batch, config.h_hidden, model.h_rnn.state_size) * 0.025
    token_ids = [3, 5, 7]
    ids = torch.tensor(token_ids, dtype=torch.long)
    deep = model.h_deepembed_adapter(F.embedding(ids, model.lm_head.weight))

    outputs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    state = initial_state
    with torch.no_grad():
        for _ in range(steps):
            output, state = model.h_rnn(
                residual,
                state,
                timestep=None,
                deepemb_vec=deep,
            )
            outputs.append(output)
            states.append(state)
            logits.append(
                _finite_clamp(
                    model.h_halt_proj(output).squeeze(-1),
                    config.halt_logit_clamp,
                )
            )
        output_stack = torch.stack(outputs, dim=0).float()
        state_stack = torch.stack(states, dim=0).float()
        halt_probabilities = torch.sigmoid(torch.stack(logits, dim=0)).clamp(
            1.0e-6, 1.0 - 1.0e-6
        )
        selection = hard_act_selection(
            output_stack,
            state_stack,
            halt_probabilities,
            threshold=config.h_halt_thresh,
            min_steps=config.min_h_steps,
        )

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-manager-act-parity-") as temp:
        temp_path = Path(temp)
        model_dir = temp_path / "model"
        export_model(model, config, model_dir)
        case = {
            "batch": batch,
            "steps": steps,
            "h_residual_input": residual.flatten().tolist(),
            "h_token_ids": token_ids,
            "h_initial_packed_state": initial_state.flatten().tolist(),
        }
        case_path = temp_path / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-manager-hard-act-parity",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(model_dir),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan manager hard-ACT parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    if result["queue_submissions"] != 1:
        raise AssertionError(result)
    if result["selected_index"] != selection.selected_index.tolist():
        raise AssertionError(
            f"selected_index mismatch: rust={result['selected_index']} "
            f"torch={selection.selected_index.tolist()}"
        )
    probability_diff = assert_close(
        "halt_probabilities", result["halt_probabilities"], halt_probabilities
    )
    depth_diff = assert_close(
        "executed_steps", result["executed_steps"], selection.executed_steps
    )
    output_diff = assert_close(
        "selected_output", result["selected_output"], selection.output
    )
    state_diff = assert_close(
        "selected_packed_state", result["selected_packed_state"], selection.state
    )
    print(
        f"device={result['device']} steps={steps} queue_submissions={result['queue_submissions']}"
    )
    print(f"selected_index={result['selected_index']}")
    print(f"halt_probabilities_max_abs_diff={probability_diff:.9g}")
    print(f"executed_steps_max_abs_diff={depth_diff:.9g}")
    print(f"selected_output_max_abs_diff={output_diff:.9g}")
    print(f"selected_packed_state_max_abs_diff={state_diff:.9g}")
    print("Hierarchos Vulkan device-resident H pondering + hard-ACT PyTorch parity: PASS")


if __name__ == "__main__":
    main()
