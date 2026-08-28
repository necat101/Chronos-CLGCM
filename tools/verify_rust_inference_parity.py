#!/usr/bin/env python3
"""End-to-end tiny coherent-v9 Python <-> Rust logit parity smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from hierarchos.inference.runtime_state_interchange import (
    load_runtime_state_interchange,
    save_runtime_state_interchange,
)
from hierarchos.utils.checkpoint import AttrDict
from tools.export_rust_inference import export_model


def tiny_coherent_config(width: int = 32) -> AttrDict:
    return AttrDict(
        architecture_revision="coherent-v9",
        vocab_size=64,
        context_dim=width,
        persistent_dim=8,
        ltm_slots=16,
        ltm_key_dim=8,
        ltm_val_dim=8,
        ltm_topk=2,
        h_hidden=width,
        l_hidden=width,
        max_h_steps=3,
        max_l_steps=2,
        min_h_steps=1,
        h_stride=2,
        h_halt_thresh=0.9,
        l_conv_atol=1e-4,
        use_deepembed=True,
        use_rosa=True,
        rosa_max_context=8,
        compile=False,
        gradient_checkpointing=False,
        detach_every_n_steps=0,
    )


def _run_rust(
    export_dir: Path,
    tokens: list[int],
    *,
    load_state: Path | None = None,
    save_state: Path | None = None,
) -> dict:
    command = [
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
        str(export_dir),
        "--tokens",
        ",".join(map(str, tokens)),
    ]
    if load_state is not None:
        command.extend(["--load-state", str(load_state)])
    if save_state is not None:
        command.extend(["--save-state", str(save_state)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Rust parity runner failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise AssertionError("Rust parity runner emitted a non-object payload")
    return payload


def _assert_portable_runtime_state(
    path: Path, *, expected_tokens: list[int], expected_width: int
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "hierarchos_runtime_state_interchange":
        raise AssertionError("Rust runtime state has the wrong interchange kind")
    if payload.get("schema_version") != 1:
        raise AssertionError("Rust runtime state has the wrong interchange schema")
    if payload.get("position") != len(expected_tokens):
        raise AssertionError("Rust runtime state position does not match its prefix")
    if payload.get("history") != expected_tokens:
        raise AssertionError("Rust runtime state did not preserve exact token history")

    for branch in ("h_state", "l_state"):
        state = payload.get(branch)
        if not isinstance(state, dict):
            raise AssertionError(f"Rust runtime state is missing {branch}")
        if state.get("layout") != "rwkv_v8_matrix_packed":
            raise AssertionError(f"Rust runtime {branch} is not PyTorch/Vulkan packed RWKV-v8")
        if state.get("state_readout_mode") != "explicit-output":
            raise AssertionError(f"Rust runtime {branch} lost coherent-v9 explicit output")
        shape = state.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[:2] != [1, expected_width]:
            raise AssertionError(f"Rust runtime {branch} has invalid packed shape {shape!r}")
        values = state.get("values")
        if not isinstance(values, list) or len(values) != shape[0] * shape[1] * shape[2]:
            raise AssertionError(f"Rust runtime {branch} packed storage disagrees with shape")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="context/H/L width; 192 exercises multiple 64-wide RWKV heads",
    )
    args = parser.parse_args()
    torch.manual_seed(20260812)
    config = tiny_coherent_config(args.width)
    model = HierarchosCore(config).eval()
    tokens = [1, 2, 1, 2, 3, 1]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        expected = model(
            input_ids,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
            return_step_telemetry=False,
            return_numerics=False,
        )["logits"][0].float().cpu()

    with tempfile.TemporaryDirectory(prefix="hierarchos-rust-parity-") as temp:
        export_dir = Path(temp) / "model"
        export_model(model, config, export_dir)
        uninterrupted = _run_rust(export_dir, tokens)
        actual = torch.tensor(uninterrupted["logits"], dtype=torch.float32)

        split = 4
        first_state = Path(temp) / "runtime-prefix.json"
        final_state = Path(temp) / "runtime-final.json"
        prefix = _run_rust(export_dir, tokens[:split], save_state=first_state)
        first_state_payload = _assert_portable_runtime_state(
            first_state, expected_tokens=tokens[:split], expected_width=args.width
        )
        resumed = _run_rust(
            export_dir,
            tokens[split:],
            load_state=first_state,
            save_state=final_state,
        )
        final_state_payload = _assert_portable_runtime_state(
            final_state, expected_tokens=tokens, expected_width=args.width
        )
        resumed_logits = torch.tensor(prefix["logits"] + resumed["logits"], dtype=torch.float32)
        if not torch.equal(resumed_logits, actual):
            resume_diff = float((resumed_logits - actual).abs().max().item())
            raise AssertionError(
                f"Rust JSON runtime-state resume changed logits (max_abs={resume_diff:.9g})"
            )
        if resumed.get("state_position") != len(tokens):
            raise AssertionError("resumed Rust runner reported the wrong absolute state position")
        if first_state_payload.get("architecture_contract_sha256") != final_state_payload.get(
            "architecture_contract_sha256"
        ):
            raise AssertionError("runtime-state learned-function fingerprint changed across resume")

        # Rust -> PyTorch: restore the exact native snapshot without replaying
        # its prefix, then continue at the absolute token position.  This is the
        # same route a CUDA inference target uses because the carrier tensors
        # are ordinary PyTorch tensors and can be materialized directly there.
        rust_state_for_python = load_runtime_state_interchange(model, first_state)
        with torch.no_grad():
            python_from_rust = model(
                torch.tensor([tokens[split:]], dtype=torch.long),
                **rust_state_for_python.model_kwargs(),
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
            )["logits"][0].float().cpu()
        rust_suffix = torch.tensor(resumed["logits"], dtype=torch.float32)
        torch.testing.assert_close(
            python_from_rust,
            rust_suffix,
            rtol=2e-4,
            atol=2e-5,
        )

        # PyTorch -> Rust: export the PyTorch live state in the same schema and
        # prove native Rust can continue it directly.  The PyTorch state is
        # produced by the real model forward, including the live ROSA automaton.
        with torch.no_grad():
            python_prefix = model(
                torch.tensor([tokens[:split]], dtype=torch.long),
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
            )
        python_state = Path(temp) / "runtime-python-prefix.json"
        save_runtime_state_interchange(
            model,
            python_state,
            h_state=python_prefix["h_state"],
            l_state=python_prefix["l_state"],
            prev_context=python_prefix["prev_context"],
            target_context=python_prefix["target_context"],
            drift_state=python_prefix["drift_state"],
            ltm_memory_state=python_prefix["ltm_memory_state"],
            position=split,
            history=tokens[:split],
        )
        _assert_portable_runtime_state(
            python_state, expected_tokens=tokens[:split], expected_width=args.width
        )
        rust_from_python = _run_rust(
            export_dir,
            tokens[split:],
            load_state=python_state,
        )
        rust_from_python_suffix = torch.tensor(
            rust_from_python["logits"], dtype=torch.float32
        )
        torch.testing.assert_close(
            rust_from_python_suffix,
            expected[split:],
            rtol=2e-4,
            atol=2e-5,
        )

    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print(f"max_abs={max_abs:.9g} mean_abs={mean_abs:.9g}")
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    print("Rust packed runtime-state save/resume: PASS (bit-exact logits)")
    print("Rust -> PyTorch packed runtime-state continuation: PASS")
    print("PyTorch -> Rust packed runtime-state continuation: PASS")
    print("Rust coherent-v9 FP32 parity: PASS")


if __name__ == "__main__":
    main()
