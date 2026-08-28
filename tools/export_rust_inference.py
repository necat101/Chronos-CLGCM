#!/usr/bin/env python3
"""Export a validated Hierarchos checkpoint for the pure-Rust FP32 runtime."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos.models.revisions import architecture_contract, architecture_contract_hash
from hierarchos.utils.checkpoint import load_full_model_with_config


RUST_FORMAT_VERSION = 1
TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def _value(config, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _resolved_rust_config(config) -> dict[str, Any]:
    fields = (
        "architecture_revision",
        "vocab_size",
        "context_dim",
        "persistent_dim",
        "ltm_slots",
        "ltm_key_dim",
        "ltm_val_dim",
        "ltm_topk",
        "h_hidden",
        "l_hidden",
        "h_stride",
        "max_h_steps",
        "max_l_steps",
        "min_h_steps",
        "h_halt_thresh",
        "l_conv_atol",
        "commitment_cost_mode",
        "commitment_threshold",
        "act_depth_temperature",
        "h_rwkv_head_size",
        "l_rwkv_head_size",
        "token_adapter_rank",
        "rosa_max_context",
        "use_deepembed",
        "use_rosa",
        "memory_token_routers",
        "enforce_rosa_max_context",
        "inference_logit_parity",
        "deepembed_mode",
        "rosa_embedding_mode",
        "rwkv_state_readout_mode",
        "manager_compute_mode",
        "manager_state_commit_mode",
        "ltm_time_feature_mode",
        "ltm_score_grad_scale",
        "ltm_value_alignment_weight",
        "ltm_value_alignment_stride",
        "ltm_value_alignment_min_updates",
        "ltm_value_alignment_ready_threshold",
        "ltm_value_alignment_ema_decay",
        "ltm_value_writer_max_norm",
        "val_proj_alignment_updates",
        "val_proj_alignment_last",
        "val_proj_alignment_ema",
        "val_proj_alignment_best",
        "val_proj_writer_norm",
        "val_proj_trained",
        "recurrent_state_clamp",
        "context_state_clamp",
        "drift_state_clamp",
        "drift_norm_clamp",
        "activation_clamp",
        "halt_logit_clamp",
        "rwkv_channel_mix_key_clamp",
        "rwkv_channel_mix_deepembed_clamp",
        "drift_delta_scale",
        "inference_logit_clamp",
        "memory_gate_warmup_steps",
        "memory_gate_warmup_floor",
    )
    result = {"format_version": RUST_FORMAT_VERSION}
    for field in fields:
        result[field] = _value(config, field)
    # Historical/PyTorch configs may represent not-yet-initialized writer
    # readiness fields as None. The portable Rust contract uses concrete policy
    # defaults while retaining None only for genuinely optional quality metrics.
    controller_defaults = {
        "ltm_value_alignment_weight": 0.0,
        "ltm_value_alignment_stride": 1,
        "ltm_value_alignment_min_updates": 100,
        "ltm_value_alignment_ready_threshold": 0.95,
        "ltm_value_alignment_ema_decay": 0.95,
        "ltm_value_writer_max_norm": 64.0,
        "val_proj_alignment_updates": 0,
        "val_proj_trained": False,
    }
    for field, default in controller_defaults.items():
        if result[field] is None:
            result[field] = default
    result["architecture_contract"] = architecture_contract(config)
    result["architecture_contract_sha256"] = architecture_contract_hash(config)
    return result


def _resolved_pytorch_config(config) -> dict[str, Any]:
    """Preserve the resolved Hierarchos config for direct PyTorch reloads.

    The native package is also the cross-backend training interchange package,
    so it must carry the config filename understood by Hierarchos' ordinary
    PyTorch loader rather than requiring callers to reconstruct architecture
    state out-of-band.
    """
    if isinstance(config, dict):
        result = dict(config)
    elif hasattr(config, "to_dict"):
        result = dict(config.to_dict())
    else:
        result = dict(vars(config))
    result.setdefault("model_type", "hierarchos")
    result["architecture_contract_sha256"] = architecture_contract_hash(config)
    # Fail here instead of publishing a package whose model config cannot be
    # consumed by the standard loader.
    json.dumps(result)
    return result


def _validate_supported(config) -> None:
    checks = {
        "architecture_revision": "coherent-v9",
        "deepembed_mode": "shared-factorized",
        "rosa_embedding_mode": "shared-factorized",
        "rwkv_state_readout_mode": "explicit-output",
        "manager_compute_mode": "hard-masked",
        "manager_state_commit_mode": "hard-selected",
        "ltm_time_feature_mode": "metadata-only",
    }
    mismatches = [
        f"{name}={_value(config, name)!r} (expected {expected!r})"
        for name, expected in checks.items()
        if _value(config, name) != expected
    ]
    if not bool(_value(config, "inference_logit_parity", False)):
        mismatches.append("inference_logit_parity must be true")
    if float(_value(config, "inference_logit_clamp", 0.0) or 0.0) != 0.0:
        mismatches.append("inference_logit_clamp must be 0 for coherent-v9 parity")
    for name in ("use_deepembed", "use_rosa", "memory_token_routers"):
        if not bool(_value(config, name, False)):
            mismatches.append(f"{name} must be true in the phase-1 Rust runtime")
    if mismatches:
        raise ValueError(
            "The phase-1 Rust runtime intentionally fails closed on non-coherent-v9 "
            "learned functions:\n  - " + "\n  - ".join(mismatches)
        )


def export_model(model, config, output_dir: str | Path) -> Path:
    """Export an already-loaded/eval model. Useful for parity tests as well as CLI."""
    _validate_supported(config)
    model.eval()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    exported: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name == "tok_emb.weight":
            # lm_head.weight is the canonical alias for the tied storage.
            continue
        if not tensor.is_floating_point():
            # sources are inference metadata, not learned FP32 model weights.
            continue
        value = tensor.detach().cpu().float().contiguous()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Refusing to export non-finite tensor {name!r}")
        # clone breaks any residual shared-storage alias before safetensors checks.
        exported[name] = value.clone()

    weights_path = output / "model.safetensors"
    save_file(
        exported,
        str(weights_path),
        metadata={
            "format": "hierarchos-rust-fp32-v1",
            "architecture_revision": str(_value(config, "architecture_revision")),
            "architecture_contract_sha256": architecture_contract_hash(config),
        },
    )
    rust_config_path = output / "hierarchos_rust_config.json"
    rust_config_path.write_text(
        json.dumps(_resolved_rust_config(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pytorch_config_path = output / "hierarchos_config.json"
    pytorch_config_path.write_text(
        json.dumps(_resolved_pytorch_config(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def copy_tokenizer_assets(source: str | Path, output_dir: str | Path) -> Path:
    """Copy tokenizer files needed by the native text runtime."""
    source = Path(source)
    source_dir = source if source.is_dir() else source.parent
    output = Path(output_dir)
    tokenizer_json = source_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(
            f"No tokenizer.json found in {source_dir}. "
            "Pass --tokenizer-source with a model/tokenizer directory so the "
            "native package can encode text without Python."
        )
    for name in TOKENIZER_ASSETS:
        candidate = source_dir / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)
    return output / "tokenizer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hierarchos model directory or direct .pt checkpoint")
    parser.add_argument("output", help="Output directory for the Rust model package")
    parser.add_argument(
        "--tokenizer-source",
        default=None,
        help="Directory containing tokenizer.json (defaults to the model directory)",
    )
    args = parser.parse_args()

    model, config = load_full_model_with_config(args.model, torch.device("cpu"))
    output = export_model(model, config, args.output)
    tokenizer_path = copy_tokenizer_assets(args.tokenizer_source or args.model, output)
    print(f"Rust FP32 model exported to {output}")
    print(f"Native tokenizer copied to {tokenizer_path}")


if __name__ == "__main__":
    main()
