#!/usr/bin/env python3
"""Export a full-precision Hierarchos checkpoint for the pure Rust runtime.

This is deliberately a development/conversion utility. The produced `.hrf32`
artifact contains only little-endian float32 tensors plus normalized JSON config;
loading and inference from that point onward require no Python or PyTorch.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

# Running a tool by path makes Python use `tools/` as sys.path[0]. Add the
# checkout root explicitly so the exporter works without installing Hierarchos.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hierarchos.utils.checkpoint import load_full_model_with_config


MAGIC = b"HRF32\x00\x01\x00"
FORMAT_VERSION = 1

# The Rust milestone is read-only inference. These tensors either duplicate a
# huge tied matrix or are mutable training/online-learning state that ordinary
# generation suppresses.
SKIP_TENSORS = {
    "lm_head.weight",  # tied to tok_emb.weight
    "ltm.fast_vals",
    "ltm._mom_vals",
    "time_freqs",  # regenerated from ltm_val_dim when legacy-v8 needs it
}


def _json_config(config) -> bytes:
    payload = dict(config)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exportable_tensors(model):
    for name, tensor in model.state_dict().items():
        if name in SKIP_TENSORS:
            continue
        # sources is int64 metadata and the current Rust read-only path does not
        # filter by source. Everything consumed by model math is exported f32.
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        if tensor.is_complex():
            raise ValueError(f"Rust inference export does not support complex tensor {name!r}")
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Refusing to export non-finite tensor {name!r}")
        yield name, value


def export_model(model_path: Path, output_path: Path) -> None:
    model, config = load_full_model_with_config(str(model_path), torch.device("cpu"))
    model.eval()
    config_bytes = _json_config(config)
    tensors = list(_exportable_tensors(model))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", FORMAT_VERSION))
        handle.write(struct.pack("<Q", len(config_bytes)))
        handle.write(struct.pack("<I", len(tensors)))
        handle.write(config_bytes)

        for name, tensor in tensors:
            encoded_name = name.encode("utf-8")
            if not encoded_name or len(encoded_name) > 0xFFFF:
                raise ValueError(f"Tensor name is too long for HRF32: {name!r}")
            if tensor.ndim > 0xFF:
                raise ValueError(f"Tensor has too many dimensions for HRF32: {name!r}")
            handle.write(struct.pack("<H", len(encoded_name)))
            handle.write(encoded_name)
            handle.write(struct.pack("<B", tensor.ndim))
            for dim in tensor.shape:
                handle.write(struct.pack("<Q", int(dim)))
            handle.write(struct.pack("<Q", tensor.numel()))
            handle.write(tensor.numpy().tobytes(order="C"))

    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Exported {len(tensors)} float32 tensors to {output_path} "
        f"({size_mib:.2f} MiB)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Hierarchos model directory or .pt checkpoint")
    parser.add_argument("output", type=Path, help="Destination .hrf32 file")
    args = parser.parse_args()
    export_model(args.model, args.output)


if __name__ == "__main__":
    main()
