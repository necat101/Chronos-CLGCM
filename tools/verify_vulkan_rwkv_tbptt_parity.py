#!/usr/bin/env python3
"""Verify multi-step packed-state Vulkan TBPTT scheduling against PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
BATCH = 1
STEPS = 4
HEADS = 2
HEAD_SIZE = 64
WIDTH = HEADS * HEAD_SIZE
HIDDEN = WIDTH * 4
INPUT_DIM = 24
STATE_OFFSET = 4
STATE_SIZE = STATE_OFFSET + HEAD_SIZE
STATE_CLAMP = 0.75
DETACH_EVERY = 2


def matrix(rows: int, cols: int, scale: float) -> torch.Tensor:
    return (
        torch.randn(rows, cols) * (scale / math.sqrt(max(1, rows)))
    ).requires_grad_()


def vector(base: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    return (base + torch.rand(WIDTH) * scale).requires_grad_()


def make_parameters() -> dict[str, torch.Tensor]:
    adapter_rank = 12
    return {
        "ln1_w": (0.9 + torch.rand(WIDTH) * 0.2).requires_grad_(),
        "ln1_b": (torch.randn(WIDTH) * 0.04).requires_grad_(),
        "mix_r": vector(),
        "mix_k": vector(),
        "mix_v": vector(),
        "mix_w": vector(),
        "mix_a": vector(),
        "mix_g": vector(),
        "r_weight": matrix(WIDTH, WIDTH, 0.35),
        "k_weight": matrix(WIDTH, WIDTH, 0.35),
        "v_weight": matrix(WIDTH, WIDTH, 0.35),
        "k_k": vector(0.65, 0.12),
        "k_a": vector(0.95, 0.12),
        "w0": (-2.5 + torch.randn(1, WIDTH) * 0.4).requires_grad_(),
        "w1": matrix(WIDTH, 32, 0.18),
        "w2": matrix(32, WIDTH, 0.18),
        "a0": (torch.randn(1, WIDTH) * 0.25).requires_grad_(),
        "a1": matrix(WIDTH, 32, 0.18),
        "a2": matrix(32, WIDTH, 0.18),
        "g1": matrix(WIDTH, 64, 0.18),
        "g2": matrix(64, WIDTH, 0.18),
        "r_k": (torch.randn(HEADS, HEAD_SIZE) * 0.08).requires_grad_(),
        "gn_w": (0.9 + torch.rand(WIDTH) * 0.2).requires_grad_(),
        "gn_b": (torch.randn(WIDTH) * 0.08).requires_grad_(),
        "out_weight": matrix(WIDTH, WIDTH, 0.24),
        "ln2_w": (0.9 + torch.rand(WIDTH) * 0.2).requires_grad_(),
        "ln2_b": (torch.randn(WIDTH) * 0.04).requires_grad_(),
        "mix_k_cm": (torch.randn(WIDTH) * 0.12).requires_grad_(),
        "key_cm": matrix(HIDDEN, WIDTH, 0.28),
        "value_cm": matrix(WIDTH, HIDDEN, 0.18),
        "adapter_down": (
            torch.randn(adapter_rank, INPUT_DIM) * (0.12 / math.sqrt(INPUT_DIM))
        ).requires_grad_(),
        "adapter_up": (
            torch.randn(HIDDEN, adapter_rank) * (0.06 / math.sqrt(adapter_rank))
        ).requires_grad_(),
        "adapter_bias": (
            torch.ones(HIDDEN) + torch.randn(HIDDEN) * 0.03
        ).requires_grad_(),
    }


def cell_step(
    x: torch.Tensor,
    token_features: torch.Tensor,
    packed_state: torch.Tensor,
    p: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    previous_tm = packed_state[:, :, 0]
    previous_cm = packed_state[:, :, 1]
    matrix_state = packed_state[:, :, STATE_OFFSET:].reshape(
        BATCH, HEADS, HEAD_SIZE, HEAD_SIZE
    )

    token_norm = F.layer_norm(token_features, (INPUT_DIM,), eps=1.0e-5)
    deepembed = p["adapter_bias"] + F.linear(
        F.silu(F.linear(token_norm, p["adapter_down"])), p["adapter_up"]
    )
    x_norm = F.layer_norm(x, (WIDTH,), p["ln1_w"], p["ln1_b"], 1.0e-5)
    delta = previous_tm - x_norm
    xr = x_norm + delta * p["mix_r"]
    xk = x_norm + delta * p["mix_k"]
    xv = x_norm + delta * p["mix_v"]
    xw = x_norm + delta * p["mix_w"]
    xa = x_norm + delta * p["mix_a"]
    xg = x_norm + delta * p["mix_g"]
    r = F.linear(xr, p["r_weight"])
    raw_k = F.linear(xk, p["k_weight"])
    v = F.linear(xv, p["v_weight"])
    w = -F.softplus(-(p["w0"] + torch.tanh(xw @ p["w1"]) @ p["w2"])) - 0.5
    a = torch.sigmoid(p["a0"] + (xa @ p["a1"]) @ p["a2"])
    g = torch.sigmoid(xg @ p["g1"]) @ p["g2"]
    kk = F.normalize(
        (raw_k * p["k_k"]).view(BATCH, HEADS, HEAD_SIZE),
        dim=-1,
        p=2.0,
        eps=1.0e-12,
    ).view(BATCH, WIDTH)
    scaled_k = raw_k * (1.0 + (a - 1.0) * p["k_a"])
    r_h = r.view(BATCH, HEADS, HEAD_SIZE)
    k_h = scaled_k.view(BATCH, HEADS, HEAD_SIZE)
    v_h = v.view(BATCH, HEADS, HEAD_SIZE)
    kk_h = kk.view(BATCH, HEADS, HEAD_SIZE)
    a_h = a.view(BATCH, HEADS, HEAD_SIZE)
    decay = torch.exp(
        -torch.exp(torch.clamp(w, -60.0, 30.0).view(BATCH, HEADS, HEAD_SIZE))
    )
    sa = torch.matmul(matrix_state, (-kk_h).unsqueeze(-1)).squeeze(-1)
    new_matrix_state = (
        matrix_state * decay.unsqueeze(-2)
        + sa.unsqueeze(-1) * (kk_h * a_h).unsqueeze(-2)
        + v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    )
    tmix = torch.matmul(new_matrix_state, r_h.unsqueeze(-1)).squeeze(-1).reshape(
        BATCH, WIDTH
    )
    group_normed = F.group_norm(
        tmix,
        HEADS,
        weight=p["gn_w"],
        bias=p["gn_b"],
        eps=64e-5,
    )
    bonus = ((r_h * k_h * p["r_k"]).sum(dim=-1, keepdim=True) * v_h).reshape(
        BATCH, WIDTH
    )
    x_after_time = x + F.linear((group_normed + bonus) * g, p["out_weight"])
    x_norm2 = F.layer_norm(
        x_after_time, (WIDTH,), p["ln2_w"], p["ln2_b"], 1.0e-5
    )
    mixed_cm = x_norm2 + (previous_cm - x_norm2) * p["mix_k_cm"]
    cm_key = torch.clamp(F.linear(mixed_cm, p["key_cm"]), -12.0, 12.0)
    ffn = torch.square(torch.relu(cm_key)) * torch.clamp(deepembed, -4.0, 4.0)
    ffn = torch.clamp(ffn, -576.0, 576.0)
    output = x_after_time + F.linear(ffn, p["value_cm"])
    packed_new_state = torch.cat(
        [
            x_norm.unsqueeze(-1),
            x_norm2.unsqueeze(-1),
            v.unsqueeze(-1),
            output.unsqueeze(-1),
            new_matrix_state.reshape(BATCH, WIDTH, HEAD_SIZE),
        ],
        dim=-1,
    )
    return output, torch.clamp(packed_new_state, -STATE_CLAMP, STATE_CLAMP)


def package_tensors(p: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def d(name: str) -> torch.Tensor:
        return p[name].detach().contiguous()

    return {
        "h_rnn.ln1.weight": d("ln1_w"),
        "h_rnn.ln1.bias": d("ln1_b"),
        "h_rnn.x_r": d("mix_r").view(1, WIDTH),
        "h_rnn.x_k": d("mix_k").view(1, WIDTH),
        "h_rnn.x_v": d("mix_v").view(1, WIDTH),
        "h_rnn.x_w": d("mix_w").view(1, WIDTH),
        "h_rnn.x_a": d("mix_a").view(1, WIDTH),
        "h_rnn.x_g": d("mix_g").view(1, WIDTH),
        "h_rnn.receptance.weight": d("r_weight"),
        "h_rnn.key.weight": d("k_weight"),
        "h_rnn.value.weight": d("v_weight"),
        "h_rnn.k_k": d("k_k").view(1, WIDTH),
        "h_rnn.k_a": d("k_a").view(1, WIDTH),
        "h_rnn.w0": d("w0"),
        "h_rnn.w1": d("w1"),
        "h_rnn.w2": d("w2"),
        "h_rnn.a0": d("a0"),
        "h_rnn.a1": d("a1"),
        "h_rnn.a2": d("a2"),
        "h_rnn.g1": d("g1"),
        "h_rnn.g2": d("g2"),
        "h_rnn.r_k": d("r_k"),
        "h_rnn.ln_x.weight": d("gn_w"),
        "h_rnn.ln_x.bias": d("gn_b"),
        "h_rnn.output.weight": d("out_weight"),
        "h_rnn.ln2.weight": d("ln2_w"),
        "h_rnn.ln2.bias": d("ln2_b"),
        "h_rnn.x_k_cm": d("mix_k_cm").view(1, WIDTH),
        "h_rnn.key_cm.weight": d("key_cm"),
        "h_rnn.value_cm.weight": d("value_cm"),
        "h_deepembed_adapter.down.weight": d("adapter_down"),
        "h_deepembed_adapter.up.weight": d("adapter_up"),
        "h_deepembed_adapter.bias": d("adapter_bias"),
    }


def export_benchmark_package(
    package_dir: Path, p: dict[str, torch.Tensor], *, vocab_size: int = 29
) -> None:
    """Export the canonical PyTorch-row-major package used by the Rust A/B gate."""
    package_dir.mkdir(parents=True, exist_ok=True)
    tensors = package_tensors(p)
    tensors["lm_head.weight"] = (
        torch.randn(vocab_size, INPUT_DIM) * (0.2 / math.sqrt(INPUT_DIM))
    ).contiguous()
    save_file(
        tensors,
        str(package_dir / "model.safetensors"),
        metadata={"format": "pt", "layout": "pytorch-row-major"},
    )


def main() -> None:
    global HEADS, HEAD_SIZE, WIDTH, HIDDEN, STATE_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--head-size", type=int, default=HEAD_SIZE)
    parser.add_argument(
        "--kernel-geometry",
        choices=("rwkv-state-bwd-wg32", "rwkv-state-bwd-wg64", "rwkv-state-bwd-wg128"),
        default=None,
    )
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
        "--export-benchmark-package",
        type=Path,
        default=None,
        help="also export a tied-embedding SafeTensors package for the Rust packed-TBPTT A/B benchmark",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="export --export-benchmark-package and skip the PyTorch/Vulkan parity run",
    )
    args = parser.parse_args()
    if args.heads <= 0 or args.head_size <= 0:
        parser.error("--heads and --head-size must be positive")
    if args.export_only and args.export_benchmark_package is None:
        parser.error("--export-only requires --export-benchmark-package")
    HEADS = args.heads
    HEAD_SIZE = args.head_size
    WIDTH = HEADS * HEAD_SIZE
    HIDDEN = WIDTH * 4
    STATE_SIZE = STATE_OFFSET + HEAD_SIZE
    torch.manual_seed(20260821)
    p = make_parameters()
    if args.export_benchmark_package is not None:
        export_benchmark_package(args.export_benchmark_package, p)
        print(
            "Exported packed-TBPTT benchmark package "
            f"width={WIDTH} head_size={HEAD_SIZE} to {args.export_benchmark_package}"
        )
        if args.export_only:
            return
    x_sequence = (torch.randn(STEPS, BATCH, WIDTH) * 0.22).requires_grad_()
    token_sequence = (torch.randn(STEPS, BATCH, INPUT_DIM) * 0.28).requires_grad_()
    initial_state = (
        torch.randn(BATCH, WIDTH, STATE_SIZE) * 0.24
    ).requires_grad_()
    grad_output = torch.randn(STEPS, BATCH, WIDTH) * 0.025
    final_state_grad = torch.randn(BATCH, WIDTH, STATE_SIZE) * 0.012

    state = initial_state
    outputs: list[torch.Tensor] = []
    for timestep in range(STEPS):
        if timestep > 0 and timestep % DETACH_EVERY == 0:
            state = state.detach()
        output, state = cell_step(x_sequence[timestep], token_sequence[timestep], state, p)
        outputs.append(output)
    output_sequence = torch.stack(outputs, dim=0)
    objective = (output_sequence * grad_output).sum() + (state * final_state_grad).sum()
    objective.backward()

    case = {
        "batch": BATCH,
        "steps": STEPS,
        "width": WIDTH,
        "head_size": HEAD_SIZE,
        "input_dim": INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": STATE_CLAMP,
        "detach_every_n_steps": DETACH_EVERY,
        "x_sequence": x_sequence.detach().flatten().tolist(),
        "token_feature_sequence": token_sequence.detach().flatten().tolist(),
        "initial_packed_state": initial_state.detach().flatten().tolist(),
        "grad_output_sequence": grad_output.flatten().tolist(),
        "final_packed_state_grad": final_state_grad.flatten().tolist(),
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-tbptt-") as temp_dir:
        temp = Path(temp_dir)
        package_dir = temp / "model-package"
        package_dir.mkdir()
        save_file(
            package_tensors(p),
            str(package_dir / "model.safetensors"),
            metadata={"format": "pt", "layout": "pytorch-row-major"},
        )
        case_path = temp / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        command = [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-rwkv-tbptt-sequence-step",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(package_dir),
                "--cell-prefix",
                "h_rnn",
                "--adapter-prefix",
                "h_deepembed_adapter",
            ]
        if args.kernel_geometry is not None:
            command.extend(["--kernel-geometry", args.kernel_geometry])
        command.extend(["--numerics", args.numerics])
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan TBPTT parity runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    expected_numerics = "strict-parity" if args.numerics == "strict" else args.numerics
    if result["numerics_policy"] != expected_numerics:
        raise AssertionError(
            f"Vulkan numerics policy mismatch: {result['numerics_policy']} != {expected_numerics}"
        )
    if args.kernel_geometry is not None and result["backward_kernel_geometry"] != args.kernel_geometry:
        raise AssertionError(
            "Vulkan backward geometry mismatch: "
            f"{result['backward_kernel_geometry']} != {args.kernel_geometry}"
        )

    comparisons = {
        "outputs": (result["outputs"], output_sequence.detach()),
        "final_packed_state": (result["final_packed_state"], state.detach()),
        "grad_x": (result["grad_x"], x_sequence.grad),
        "token_feature_grad": (result["token_feature_grad"], token_sequence.grad),
        "grad_initial_packed_state": (
            result["grad_initial_packed_state"],
            initial_state.grad,
        ),
    }
    diffs: dict[str, float] = {}
    for name, (actual_values, expected) in comparisons.items():
        assert expected is not None
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=2.5e-3, atol=2.5e-5)
        diffs[name] = (actual - expected).abs().max().item()

    print(
        f"device={result['device']} numerics={result['numerics_policy']} "
        f"geometry={result['backward_kernel_geometry']} steps={result['steps']} "
        f"detach_every={DETACH_EVERY} state_size={result['state_size']}"
    )
    print(" ".join(f"max_abs_{name}={value:.9g}" for name, value in diffs.items()))
    print("Hierarchos Vulkan packed-state TBPTT PyTorch parity: PASS")


if __name__ == "__main__":
    main()
