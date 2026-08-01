import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from hierarchos import AttrDict, HierarchosCore
from hierarchos.models.revisions import (
    architecture_contract,
    architecture_contract_hash,
    normalize_architecture_revision,
)
from hierarchos.utils.checkpoint import (
    _infer_arch_flags_from_state_dict,
    _reject_unsupported_rwkv_state_dict,
    _resolve_weights_path,
    _validate_state_dict_finite,
    _validate_tied_embedding_state_dict,
    load_checkpoint_payload_compatible,
    sanitize_model_state_dict,
    save_checkpoint_safely,
    validate_checkpoint_architecture_contract,
)
from hierarchos.utils.tokenizer import (
    checkpoint_tokenizer_identity,
    tokenizer_identity,
    tokenizer_vocab_size,
    validate_inference_tokenizer_identity,
)


MODEL_WEIGHTS_NAME = "hierarchos.pt"
EXPANSION_PROVENANCE_VERSION = 1
EXPANSION_CHECKPOINT_VERSION = 4
EXPANSION_CHECKPOINT_KIND = "inference-expanded"
EXPANSION_MAPPING_VERSION = "segment-aware-v1"

SOURCE_METADATA_KEYS = (
    "checkpoint_version",
    "checkpoint_kind",
    "completed_epoch",
    "architecture_contract",
    "architecture_contract_sha256",
    "tokenizer_identity",
    "run_identity",
    "training_complete",
)

LATEST_CONFIG_DEFAULTS = {
    "max_length": 1024,
    "ltm_lr": 1e-3,
    "ltm_topk": 4,
    "max_h_steps": 5,
    "max_l_steps": 5,
    "h_stride": 4,
    "l_conv_atol": 1e-4,
    "commitment_threshold": 0.05,
    "detach_every_n_steps": 32,
    "h_halt_thresh": 0.9,
    "memory_token_routers": True,
    "memory_gate_warmup_steps": 2000,
    "memory_gate_warmup_floor": 0.10,
    "gradient_checkpointing": False,
    "compile": False,
    "compile_mode": "max-autotune-no-cudagraphs",
    "compile_backend": None,
    "compile_dynamic": False,
    "compile_fullgraph_worker": False,
    "compile_cudagraphs": False,
    "compile_pad_to_chunk_size": True,
    "compile_static_worker_loop": None,
    "compile_h_rnn": True,
    "compile_quiet": True,
    "use_deepembed": True,
    "use_rosa": True,
    "rosa_max_context": 512,
    "rwkv_head_size": None,
}

ARCH_UPDATE_KEYS = [
    "vocab_size",
    "context_dim",
    "persistent_dim",
    "ltm_slots",
    "ltm_key_dim",
    "ltm_val_dim",
    "ltm_lr",
    "ltm_topk",
    "h_hidden",
    "l_hidden",
    "h_stride",
    "max_h_steps",
    "max_l_steps",
    "l_conv_atol",
    "commitment_threshold",
    "detach_every_n_steps",
    "h_halt_thresh",
    "memory_token_routers",
    "memory_gate_warmup_steps",
    "memory_gate_warmup_floor",
    "ltm_forget_rate",
    "use_deepembed",
    "use_rosa",
    "token_adapter_rank",
    "rosa_max_context",
    "rwkv_head_size",
]

DERIVED_OR_RUNTIME_KEYS = {
    "time_freqs",
    "ltm.neg_inf",
    "ltm.update_counts",
    "ltm.update_slots",
    "ltm.ltm_deltas",
}

LOW_VARIANCE_NEW_WEIGHTS = {
    "context_drift_proj.weight",
    "l_feedback_proj.weight",
}


def _plain_dict(value: Any) -> Any:
    if isinstance(value, AttrDict):
        return {k: _plain_dict(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_dict(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain_dict(v) for v in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(8 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _verified_checkpoint_digest(path: Path) -> str:
    """Reuse a sidecar already verified by the safe loader, or hash once."""
    checksum_path = Path(str(path) + ".sha256")
    if checksum_path.exists():
        with open(checksum_path, "r", encoding="utf-8") as checksum_file:
            digest = checksum_file.read().strip().split()[0].lower()
        if (
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            return digest
    return _sha256_file(path)


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as destination:
            json.dump(_json_safe(dict(value)), destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _path_contains(parent: Path, child: Path) -> bool:
    parent_real = os.path.realpath(os.path.abspath(os.fspath(parent)))
    child_real = os.path.realpath(os.path.abspath(os.fspath(child)))
    try:
        return os.path.commonpath([parent_real, child_real]) == parent_real
    except ValueError:
        return False


def _publish_directory_atomically(
    staging_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    backup_path = output_dir.with_name(output_dir.name + ".pre-expansion-backup")
    published = False
    moved_existing = False
    try:
        if os.path.lexists(output_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Expansion output already exists: {output_dir}. "
                    "Pass --overwrite-output to replace it atomically."
                )
            if os.path.lexists(backup_path):
                raise FileExistsError(
                    "Cannot overwrite expansion output because its recovery "
                    f"path already exists: {backup_path}"
                )
            os.replace(output_dir, backup_path)
            moved_existing = True
        os.replace(staging_dir, output_dir)
        published = True
    except Exception:
        if moved_existing and not os.path.lexists(output_dir) and os.path.lexists(backup_path):
            os.replace(backup_path, output_dir)
        raise
    finally:
        if published and moved_existing and os.path.lexists(backup_path):
            try:
                if os.path.isdir(backup_path) and not os.path.islink(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
            except OSError as exc:
                print(
                    f"[WARN] Expanded package was published, but the old output "
                    f"backup could not be removed: {backup_path} ({exc})"
                )


def _load_torch(path: Path, device: str) -> Any:
    return load_checkpoint_payload_compatible(str(path), map_location=device)


def _resolve_checkpoint_path(model_path: str) -> Tuple[Path, Path]:
    checkpoint_path, model_root = _resolve_weights_path(model_path)
    return Path(checkpoint_path), Path(model_root)


def _resolve_output_paths(output_target: str) -> Tuple[Path, Path]:
    path = Path(output_target).expanduser()
    if path.suffix.lower() == ".pt":
        output_dir = path.with_suffix("")
        print(
            f"Treating '{output_target}' as a legacy output file path. "
            "Expansion now publishes an authenticated model package, so weights "
            f"will be saved as '{output_dir / MODEL_WEIGHTS_NAME}'."
        )
        return output_dir, output_dir / MODEL_WEIGHTS_NAME

    return path, path / MODEL_WEIGHTS_NAME


def _validate_output_location(
    source_artifact: Mapping[str, Any],
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    output_abs = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    if not output_abs.name:
        raise ValueError("Refusing to use a filesystem root as expansion output.")

    checkpoint_path = Path(source_artifact["checkpoint_path"])
    source_root = Path(source_artifact["model_root"])
    if _path_contains(output_abs, checkpoint_path):
        raise ValueError(
            "Expansion output cannot contain the source checkpoint; publishing "
            "would overwrite the only authenticated input artifact."
        )
    if bool(source_artifact.get("source_is_directory")) and (
        _path_contains(output_abs, source_root)
        or _path_contains(source_root, output_abs)
    ):
        raise ValueError(
            "Expansion output and source model package cannot overlap."
        )
    if os.path.lexists(output_abs) and not overwrite:
        raise FileExistsError(
            f"Expansion output already exists: {output_abs}. "
            "Pass --overwrite-output to replace it atomically."
        )
    if os.path.lexists(output_abs) and overwrite:
        if not output_abs.is_dir() or output_abs.is_symlink():
            raise ValueError(
                "Expansion may overwrite only a real model-package directory, "
                f"not a file or symlink: {output_abs}"
            )
        entries = list(output_abs.iterdir())
        known_weights = (
            output_abs / MODEL_WEIGHTS_NAME,
            output_abs / "Hierarchos.pt",
            output_abs / "model.pt",
            output_abs / "hierarchos_final.pt",
        )
        if entries and not any(candidate.is_file() for candidate in known_weights):
            raise ValueError(
                "Refusing to replace a non-empty directory that is not a "
                f"recognized Hierarchos model package: {output_abs}"
            )
    return output_abs


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _normalize_state_dict_keys(checkpoint[key])

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return _normalize_state_dict_keys(checkpoint)

    raise ValueError("Could not find a model state dict in the checkpoint.")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    source_keys = {}
    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        if clean_key in normalized:
            previous = normalized[clean_key]
            values_match = (
                torch.is_tensor(previous)
                and torch.is_tensor(value)
                and previous.shape == value.shape
                and previous.dtype == value.dtype
                and bool(torch.equal(previous, value))
            )
            if not values_match:
                raise ValueError(
                    "Conflicting source checkpoint keys collapse to the same "
                    f"name: {source_keys[clean_key]!r} and {key!r} -> {clean_key!r}."
                )
            continue
        normalized[clean_key] = value
        source_keys[clean_key] = key
    return normalized


def _load_config(checkpoint: Any, model_root: Path) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and checkpoint.get("config"):
        return _plain_dict(checkpoint["config"])

    for name in ("hierarchos_config.json", "config.json"):
        config_path = model_root / name
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

    raise ValueError(
        "Could not find config in the checkpoint or alongside it. "
        "Expected checkpoint['config'] or hierarchos_config.json."
    )


def _small_checkpoint_metadata(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    return {
        key: copy.deepcopy(checkpoint[key])
        for key in SOURCE_METADATA_KEYS
        if key in checkpoint
    }


def _state_vocab_size(state_dict: Mapping[str, torch.Tensor]) -> int:
    tok_weight = state_dict.get("tok_emb.weight")
    head_weight = state_dict.get("lm_head.weight")
    tensor = tok_weight if torch.is_tensor(tok_weight) else head_weight
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        raise ValueError(
            "Source checkpoint does not contain a two-dimensional tied token "
            "embedding/language-head tensor."
        )
    return int(tensor.shape[0])


def _load_source_artifact(
    model_path: str,
    device: str,
    *,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Load and authenticate one expansion source without retaining optimizer state."""
    checkpoint_path, model_root = _resolve_checkpoint_path(model_path)
    print(f"Loading source checkpoint once: {checkpoint_path}")
    checkpoint = _load_torch(checkpoint_path, device)
    state_dict = _extract_state_dict(checkpoint)
    config = _infer_missing_config(_load_config(checkpoint, model_root), state_dict)

    source_label = str(checkpoint_path)
    _reject_unsupported_rwkv_state_dict(state_dict, source_label)
    _validate_tied_embedding_state_dict(state_dict, source_label)
    _validate_state_dict_finite(state_dict, source_label)
    _infer_arch_flags_from_state_dict(config, state_dict)
    if isinstance(checkpoint, dict):
        validate_checkpoint_architecture_contract(
            checkpoint,
            config,
            source_label,
        )

    config["architecture_revision"] = normalize_architecture_revision(
        config.get("architecture_revision")
    )
    source_contract = architecture_contract(config)
    source_contract_hash = architecture_contract_hash(config)

    print(f"Loading and authenticating tokenizer from: {model_root}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_root),
            trust_remote_code=bool(trust_remote_code),
            local_files_only=True,
        )
    except Exception as exc:
        raise ValueError(
            "Expansion requires the source tokenizer in the model package so "
            "token IDs can be bound to the transplanted embedding rows. "
            f"Could not load a local tokenizer from '{model_root}': {exc}"
        ) from exc

    validate_inference_tokenizer_identity(tokenizer, checkpoint)
    computed_tokenizer_identity = tokenizer_identity(tokenizer)
    saved_tokenizer_identity = checkpoint_tokenizer_identity(checkpoint)
    if saved_tokenizer_identity is not None:
        saved_digest = str(saved_tokenizer_identity.get("sha256") or "").lower()
        if saved_digest and saved_digest != computed_tokenizer_identity["sha256"].lower():
            raise ValueError(
                "Source tokenizer fingerprint changed after checkpoint validation."
            )

    tensor_vocab_size = _state_vocab_size(state_dict)
    config_vocab_size = int(config.get("vocab_size", tensor_vocab_size))
    actual_tokenizer_vocab_size = tokenizer_vocab_size(tokenizer)
    if config_vocab_size != tensor_vocab_size:
        raise ValueError(
            "Source config vocabulary size does not match its embedding tensor "
            f"({config_vocab_size} != {tensor_vocab_size})."
        )
    if actual_tokenizer_vocab_size != tensor_vocab_size:
        raise ValueError(
            "Source tokenizer vocabulary size does not match its embedding rows "
            f"({actual_tokenizer_vocab_size} != {tensor_vocab_size})."
        )

    metadata = _small_checkpoint_metadata(checkpoint)
    checkpoint_digest = _verified_checkpoint_digest(checkpoint_path)
    # Exact-resume checkpoints can contain optimizer/scaler/scheduler/RNG state
    # several times larger than the learned weights. Keep only the state dict and
    # small authenticated metadata before allocating the expanded model.
    del checkpoint
    gc.collect()

    return {
        "checkpoint_path": checkpoint_path.resolve(),
        "model_root": model_root.resolve(),
        "source_is_directory": Path(model_path).expanduser().is_dir(),
        "checkpoint_sha256": checkpoint_digest,
        "state_dict": state_dict,
        "config": config,
        "checkpoint_metadata": metadata,
        "source_architecture_contract": source_contract,
        "source_architecture_contract_sha256": source_contract_hash,
        "tokenizer": tokenizer,
        "tokenizer_identity": computed_tokenizer_identity,
    }


def _infer_missing_config(config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    inferred = copy.deepcopy(config)

    if "tok_emb.weight" in state_dict:
        inferred.setdefault("vocab_size", int(state_dict["tok_emb.weight"].shape[0]))
        inferred.setdefault("context_dim", int(state_dict["tok_emb.weight"].shape[1]))
    elif "lm_head.weight" in state_dict:
        inferred.setdefault("vocab_size", int(state_dict["lm_head.weight"].shape[0]))
        inferred.setdefault("context_dim", int(state_dict["lm_head.weight"].shape[1]))

    if "persistent" in state_dict and state_dict["persistent"].ndim == 1:
        inferred.setdefault("persistent_dim", int(state_dict["persistent"].shape[0]))

    if "ltm.keys" in state_dict and state_dict["ltm.keys"].ndim == 2:
        inferred.setdefault("ltm_slots", int(state_dict["ltm.keys"].shape[0]))
        inferred.setdefault("ltm_key_dim", int(state_dict["ltm.keys"].shape[1]))

    if "ltm.vals" in state_dict and state_dict["ltm.vals"].ndim == 2:
        inferred.setdefault("ltm_slots", int(state_dict["ltm.vals"].shape[0]))
        inferred.setdefault("ltm_val_dim", int(state_dict["ltm.vals"].shape[1]))

    if "qproj.weight" in state_dict and state_dict["qproj.weight"].ndim == 2:
        inferred.setdefault("ltm_key_dim", int(state_dict["qproj.weight"].shape[0]))

    if "h_rnn.key.weight" in state_dict and state_dict["h_rnn.key.weight"].ndim == 2:
        inferred.setdefault("h_hidden", int(state_dict["h_rnn.key.weight"].shape[0]))

    if "h_rnn.r_k" in state_dict and state_dict["h_rnn.r_k"].ndim == 2:
        inferred.setdefault("rwkv_head_size", int(state_dict["h_rnn.r_k"].shape[1]))

    if "l_rnn.key.weight" in state_dict and state_dict["l_rnn.key.weight"].ndim == 2:
        inferred.setdefault("l_hidden", int(state_dict["l_rnn.key.weight"].shape[0]))

    if "h_hidden" not in inferred and "context_dim" in inferred:
        inferred["h_hidden"] = inferred["context_dim"]
    if "l_hidden" not in inferred and "context_dim" in inferred:
        inferred["l_hidden"] = inferred["context_dim"]

    for key, value in LATEST_CONFIG_DEFAULTS.items():
        if key not in inferred:
            inferred[key] = value

    inferred["compile"] = False
    inferred.setdefault("model_type", "hierarchos")
    return inferred


def _source_key_for(new_key: str, old_state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    if new_key in old_state_dict:
        return new_key

    tied_aliases = {
        "tok_emb.weight": "lm_head.weight",
        "lm_head.weight": "tok_emb.weight",
    }
    alias = tied_aliases.get(new_key)
    if alias in old_state_dict:
        return alias

    return None


def _copy_overlap_(target: torch.Tensor, source: torch.Tensor) -> Optional[Tuple[int, ...]]:
    if target.shape == source.shape:
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        return tuple(target.shape)

    if target.ndim != source.ndim:
        return None

    slices = tuple(slice(0, min(new_dim, old_dim)) for new_dim, old_dim in zip(target.shape, source.shape))
    if any(s.stop == 0 for s in slices):
        return None

    source_view = source.to(device=target.device, dtype=target.dtype)
    target[slices].copy_(source_view[slices])
    return tuple(s.stop for s in slices)


def _required_config_int(config: Mapping[str, Any], name: str) -> int:
    try:
        value = int(config[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Expansion config requires integer field {name!r}.") from exc
    if value <= 0:
        raise ValueError(f"Expansion config field {name!r} must be positive, got {value}.")
    return value


def _copy_column_segments_(
    target: torch.Tensor,
    source: torch.Tensor,
    segments: Tuple[Tuple[int, int, int, int], ...],
) -> int:
    """Copy semantic column blocks while preserving the initialized expansion area."""
    if target.ndim != 2 or source.ndim != 2:
        raise ValueError("Segment-aware transplantation requires two-dimensional tensors.")
    rows = min(int(target.shape[0]), int(source.shape[0]))
    if rows <= 0:
        return 0
    source_view = source.to(device=target.device, dtype=target.dtype)
    copied = 0
    for target_start, target_width, source_start, source_width in segments:
        width = min(int(target_width), int(source_width))
        if width <= 0:
            continue
        if target_start < 0 or target_start + width > target.shape[1]:
            raise ValueError("Target segment exceeds projection geometry during expansion.")
        if source_start < 0 or source_start + width > source.shape[1]:
            raise ValueError("Source segment exceeds projection geometry during expansion.")
        target[:rows, target_start:target_start + width].copy_(
            source_view[:rows, source_start:source_start + width]
        )
        copied += rows * width
    return copied


def _copy_named_tensor_(
    name: str,
    target: torch.Tensor,
    source: torch.Tensor,
    source_config: Mapping[str, Any],
    target_config: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    """Copy one tensor, using semantic blocks for concatenated projections."""
    old_context = _required_config_int(source_config, "context_dim")
    new_context = _required_config_int(target_config, "context_dim")

    if name == "qproj.weight":
        expected_target_width = 2 * new_context
        if target.ndim != 2 or int(target.shape[1]) != expected_target_width:
            raise ValueError(
                f"Unexpected target qproj geometry {list(target.shape)}; expected "
                f"input width {expected_target_width}."
            )
        if source.ndim != 2:
            raise ValueError(f"Unexpected source qproj geometry {list(source.shape)}.")
        if int(source.shape[1]) == 2 * old_context:
            if old_context == new_context and source.shape == target.shape:
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                return "exact", f"exact {list(target.shape)}"
            copied = _copy_column_segments_(
                target,
                source,
                (
                    (0, new_context, 0, old_context),
                    (new_context, new_context, old_context, old_context),
                ),
            )
            return "resized", f"semantic token/context blocks ({copied} elements)"
        if int(source.shape[1]) == old_context:
            rows = min(int(target.shape[0]), int(source.shape[0]))
            # The supported legacy qproj had no previous-context block. Zero the
            # corresponding rows so expansion preserves that learned function.
            target[:rows, new_context:2 * new_context].zero_()
            copied = _copy_column_segments_(
                target,
                source,
                ((0, new_context, 0, old_context),),
            )
            return "resized", f"legacy token-only query plus zero context block ({copied} elements)"
        raise ValueError(
            "Unsupported qproj input geometry during expansion: "
            f"source={list(source.shape)}, source_context_dim={old_context}."
        )

    if name == "l_input_proj.weight":
        if target.ndim != 2 or source.ndim != 2:
            raise ValueError("Worker input projection must be two-dimensional.")
        if int(source.shape[1]) != 2 * old_context or int(target.shape[1]) != 2 * new_context:
            raise ValueError(
                "Unexpected worker input projection geometry during expansion: "
                f"source={list(source.shape)}, target={list(target.shape)}."
            )
        if old_context == new_context and source.shape == target.shape:
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            return "exact", f"exact {list(target.shape)}"
        copied = _copy_column_segments_(
            target,
            source,
            (
                (0, new_context, 0, old_context),
                (new_context, new_context, old_context, old_context),
            ),
        )
        return "resized", f"semantic encoding/context blocks ({copied} elements)"

    if name == "in_proj.weight":
        old_persistent = _required_config_int(source_config, "persistent_dim")
        new_persistent = _required_config_int(target_config, "persistent_dim")
        old_topk = _required_config_int(source_config, "ltm_topk")
        new_topk = _required_config_int(target_config, "ltm_topk")
        old_value_dim = _required_config_int(source_config, "ltm_val_dim")
        new_value_dim = _required_config_int(target_config, "ltm_val_dim")
        expected_source_width = old_context + old_persistent + old_topk * old_value_dim
        expected_target_width = new_context + new_persistent + new_topk * new_value_dim
        if (
            source.ndim != 2
            or target.ndim != 2
            or int(source.shape[1]) != expected_source_width
            or int(target.shape[1]) != expected_target_width
        ):
            raise ValueError(
                "Unexpected memory-input projection geometry during expansion: "
                f"source={list(source.shape)} (expected width {expected_source_width}), "
                f"target={list(target.shape)} (expected width {expected_target_width})."
            )
        unchanged = (
            old_context == new_context
            and old_persistent == new_persistent
            and old_topk == new_topk
            and old_value_dim == new_value_dim
            and source.shape == target.shape
        )
        if unchanged:
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            return "exact", f"exact {list(target.shape)}"

        segments = [
            (0, new_context, 0, old_context),
            (new_context, new_persistent, old_context, old_persistent),
        ]
        old_memory_start = old_context + old_persistent
        new_memory_start = new_context + new_persistent
        for slot in range(min(old_topk, new_topk)):
            segments.append(
                (
                    new_memory_start + slot * new_value_dim,
                    new_value_dim,
                    old_memory_start + slot * old_value_dim,
                    old_value_dim,
                )
            )
        copied = _copy_column_segments_(target, source, tuple(segments))
        return "resized", f"semantic token/persistent/memory-slot blocks ({copied} elements)"

    if target.shape == source.shape:
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        return "exact", f"exact {list(target.shape)}"

    copied_shape = _copy_overlap_(target, source)
    if copied_shape is None:
        return None
    return "resized", f"prefix overlap {list(copied_shape)}"


def _maybe_reinitialize_missing_weight(name: str, tensor: torch.Tensor) -> bool:
    if name in LOW_VARIANCE_NEW_WEIGHTS and tensor.is_floating_point():
        nn.init.normal_(tensor, mean=0.0, std=0.01)
        return True
    return False


def _layer_note(name: str) -> str:
    if name.startswith(("h_rnn.", "l_rnn.")):
        return "RWKV cell parameter"
    if name.startswith("val_proj."):
        return "latest LTM value-projection layer"
    if name.startswith(("h_deepemb.", "l_deepemb.")):
        return "latest RWKV-v8 DeepEmbed layer"
    if name.startswith("rosa_emb.") or name == "rosa_gate_logit":
        return "latest ROSA state layer"
    if name.startswith("l_feedback_proj."):
        return "latest worker-to-manager feedback layer"
    if name.startswith("context_drift_proj."):
        return "context-drift layer"
    if name.startswith("h_halt_proj."):
        return "manager halt layer"
    if name == "ltm_gate_logit":
        return "LTM gate"
    return "new/current layer"


def scan_dataset_for_max_length(dataset_path: str, tokenizer, kayla_mode: bool, alpaca_mode: bool = False) -> int:
    """Scan a JSON or JSONL dataset and return the max token length rounded to 8."""
    max_found_length = 0
    print(f"Scanning dataset '{dataset_path}' to determine max length...")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    def get_text_from_obj(obj: Dict[str, Any], kayla: bool, alpaca: bool) -> str:
        try:
            if kayla:
                feelings_part = f"### Feelings:\n{obj.get('feelings')}\n\n" if obj.get("feelings") else ""
                return (
                    f"### Instruction:\n{obj.get('Instruction', '')}\n\n"
                    f"{feelings_part}"
                    f"### Thought Process:\n{obj.get('thought-process', '')}\n\n"
                    f"### Response:\n{obj.get('output', '')}"
                )
            if alpaca:
                input_part = f"### Input:\n{obj.get('input', '')}\n\n" if obj.get("input") else ""
                return (
                    f"### Instruction:\n{obj.get('instruction', '')}\n\n"
                    f"{input_part}"
                    f"### Response:\n{obj.get('output', '') or obj.get('response', '')}"
                )
            return (
                f"### Instruction:\n{obj.get('instruction', '')}\n\n"
                f"### Response:\n{obj.get('output', '') or obj.get('response', '')}"
            )
        except Exception:
            return ""

    with open(dataset_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list):
                for obj in tqdm(data, desc="Scanning JSON"):
                    if not isinstance(obj, dict):
                        continue
                    length = len(tokenizer.encode(get_text_from_obj(obj, kayla_mode, alpaca_mode))) + 1
                    max_found_length = max(max_found_length, length)
        except json.JSONDecodeError:
            f.seek(0)
            for line in tqdm(f, desc="Scanning JSONL"):
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    length = len(tokenizer.encode(get_text_from_obj(obj, kayla_mode, alpaca_mode))) + 1
                    max_found_length = max(max_found_length, length)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue

    if max_found_length > 0:
        adjusted_length = (max_found_length + 16 + 7) & -8
        print(f"[OK] Auto-scan complete. max_length={adjusted_length} (found max: {max_found_length}).")
        return adjusted_length

    print("[WARN] Auto-scan did not find any valid entries.")
    return 0


def _config_changes(
    source_config: Mapping[str, Any],
    target_config: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    changes = {}
    for key in sorted(set(source_config) | set(target_config)):
        if str(key).startswith("_"):
            continue
        old_value = _json_safe(source_config.get(key))
        new_value = _json_safe(target_config.get(key))
        if old_value != new_value:
            changes[str(key)] = {"source": old_value, "expanded": new_value}
    return changes


def _build_expansion_provenance(
    source_artifact: Mapping[str, Any],
    target_config: Mapping[str, Any],
    target_contract: Mapping[str, Any],
    target_contract_hash: str,
    stats: Mapping[str, int],
) -> Dict[str, Any]:
    metadata = source_artifact.get("checkpoint_metadata") or {}
    source_run_identity = metadata.get("run_identity")
    source_run_digest = (
        source_run_identity.get("sha256")
        if isinstance(source_run_identity, dict)
        else None
    )
    provenance = {
        "version": EXPANSION_PROVENANCE_VERSION,
        "mapping_version": EXPANSION_MAPPING_VERSION,
        "source": {
            "checkpoint_name": Path(source_artifact["checkpoint_path"]).name,
            "checkpoint_sha256": source_artifact["checkpoint_sha256"],
            "checkpoint_version": metadata.get("checkpoint_version"),
            "checkpoint_kind": metadata.get("checkpoint_kind"),
            "run_identity_sha256": source_run_digest,
            "architecture_contract": source_artifact[
                "source_architecture_contract"
            ],
            "architecture_contract_sha256": source_artifact[
                "source_architecture_contract_sha256"
            ],
            "tokenizer_identity": source_artifact["tokenizer_identity"],
        },
        "expanded": {
            "checkpoint_kind": EXPANSION_CHECKPOINT_KIND,
            "architecture_contract": dict(target_contract),
            "architecture_contract_sha256": target_contract_hash,
            "tokenizer_identity": source_artifact["tokenizer_identity"],
            "transient_ltm_reset": True,
            "config_changes": _config_changes(
                source_artifact["config"],
                target_config,
            ),
            "transplant_stats": dict(stats),
        },
    }
    provenance["sha256"] = _sha256_json(provenance)
    return provenance


def transplant_weights(
    old_model_path: str,
    new_config: Dict[str, Any],
    output_target: str,
    device: str,
    *,
    trust_remote_code: bool = False,
    overwrite_output: bool = False,
    source_artifact: Optional[Dict[str, Any]] = None,
) -> None:
    """Create and atomically publish an authenticated expanded model package."""
    if source_artifact is None:
        source_artifact = _load_source_artifact(
            old_model_path,
            device,
            trust_remote_code=trust_remote_code,
        )
    old_state_dict = source_artifact["state_dict"]
    source_config = source_artifact["config"]
    output_dir, _ = _resolve_output_paths(output_target)
    output_dir = _validate_output_location(
        source_artifact,
        output_dir,
        overwrite=overwrite_output,
    )

    target_vocab_size = int(new_config.get("vocab_size", 0) or 0)
    bound_vocab_size = int(source_artifact["tokenizer_identity"]["vocab_size"])
    if target_vocab_size != bound_vocab_size:
        raise ValueError(
            "Model expansion cannot change vocabulary size without an audited "
            "tokenizer-ID migration. Keep --vocab-size equal to the source "
            f"tokenizer size ({bound_vocab_size}); got {target_vocab_size}."
        )

    print("Initializing latest HierarchosCore layout...")
    new_model = HierarchosCore(AttrDict(new_config)).to(device)
    new_state_dict = new_model.state_dict()

    stats = {
        "copied": 0,
        "resized": 0,
        "initialized": 0,
        "skipped": 0,
    }

    print("Transplanting weights into latest layer set...")
    for name, new_tensor in tqdm(new_state_dict.items()):
        source_key = _source_key_for(name, old_state_dict)

        if source_key is None:
            if _maybe_reinitialize_missing_weight(name, new_tensor):
                print(f"  - Initialized missing {name} ({_layer_note(name)}) with low-variance weights.")
            else:
                print(f"  - Missing {name} ({_layer_note(name)}); keeping latest initialization.")
            stats["initialized"] += 1
            continue

        old_tensor = old_state_dict[source_key]
        if not torch.is_tensor(old_tensor):
            print(f"  - Skipping {name}: source value is not a tensor.")
            stats["skipped"] += 1
            continue

        if name in DERIVED_OR_RUNTIME_KEYS and old_tensor.shape != new_tensor.shape:
            print(
                f"  - Keeping derived/runtime {name} initialized by latest code "
                f"(old {list(old_tensor.shape)} -> new {list(new_tensor.shape)})."
            )
            stats["initialized"] += 1
            continue

        copy_result = _copy_named_tensor_(
            name,
            new_tensor,
            old_tensor,
            source_config,
            new_config,
        )
        if copy_result is None:
            print(
                f"  - Could not map {name} from {source_key} "
                f"(old {list(old_tensor.shape)} -> new {list(new_tensor.shape)}); keeping latest initialization."
            )
            stats["skipped"] += 1
        elif copy_result[0] == "exact":
            stats["copied"] += 1
        else:
            print(
                f"  - Partially copied {name} from {source_key}: "
                f"old {list(old_tensor.shape)} -> new {list(new_tensor.shape)}; "
                f"{copy_result[1]}."
            )
            stats["resized"] += 1

    new_model.load_state_dict(new_state_dict, strict=True)
    if hasattr(new_model, "reset_memory"):
        new_model.reset_memory()
        print("Reset transient LTM working memory before saving expanded model.")

    print(
        "Transplant summary: "
        f"{stats['copied']} copied, {stats['resized']} resized, "
        f"{stats['initialized']} initialized, {stats['skipped']} skipped."
    )

    # Release the source weights before serialization. This is especially
    # important when the source was an exact-resume checkpoint whose optimizer
    # payload was already discarded and the expanded model is larger.
    source_artifact["state_dict"] = {}
    del old_state_dict
    gc.collect()

    final_config = _plain_dict(new_model.config)
    for key in list(final_config):
        if str(key).startswith("_"):
            final_config.pop(key, None)
    final_config["compile"] = False
    final_config["tokenizer_identity_sha256"] = source_artifact[
        "tokenizer_identity"
    ]["sha256"]
    target_contract = architecture_contract(final_config)
    target_contract_hash = architecture_contract_hash(final_config)
    final_config["architecture_contract_sha256"] = target_contract_hash
    final_config = _json_safe(final_config)

    output_state_dict = sanitize_model_state_dict(
        new_model,
        reset_transient_ltm=True,
    )
    _validate_tied_embedding_state_dict(
        output_state_dict,
        "expanded model before publication",
    )
    _validate_state_dict_finite(
        output_state_dict,
        "expanded model before publication",
    )
    provenance = _build_expansion_provenance(
        source_artifact,
        final_config,
        target_contract,
        target_contract_hash,
        stats,
    )
    source_metadata = source_artifact.get("checkpoint_metadata") or {}
    checkpoint_payload = {
        "checkpoint_version": EXPANSION_CHECKPOINT_VERSION,
        "checkpoint_kind": EXPANSION_CHECKPOINT_KIND,
        "derived_from_checkpoint_kind": source_metadata.get("checkpoint_kind"),
        "derived_from_checkpoint_version": source_metadata.get("checkpoint_version"),
        "derived_from_checkpoint_sha256": source_artifact["checkpoint_sha256"],
        "model_state_dict": output_state_dict,
        "config": final_config,
        "architecture_contract": target_contract,
        "architecture_contract_sha256": target_contract_hash,
        "tokenizer_identity": copy.deepcopy(source_artifact["tokenizer_identity"]),
        "expansion_provenance": provenance,
        "training_complete": True,
    }
    validate_checkpoint_architecture_contract(
        checkpoint_payload,
        final_config,
        "expanded model before publication",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.expansion-staging-",
            dir=str(output_dir.parent),
        )
    )
    published = False
    try:
        print(f"\nBuilding authenticated expanded package in: {staging_dir}")
        staging_weights_path = staging_dir / MODEL_WEIGHTS_NAME
        save_checkpoint_safely(checkpoint_payload, str(staging_weights_path))
        _write_json(staging_dir / "hierarchos_config.json", final_config)
        _write_json(staging_dir / "expansion_provenance.json", provenance)

        print("Copying and verifying tokenizer files...")
        source_artifact["tokenizer"].save_pretrained(str(staging_dir))
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            str(staging_dir),
            trust_remote_code=bool(trust_remote_code),
            local_files_only=True,
        )
        if not validate_inference_tokenizer_identity(
            reloaded_tokenizer,
            checkpoint_payload,
        ):
            raise ValueError(
                "Expanded package tokenizer did not retain a strong identity proof."
            )
        if tokenizer_vocab_size(reloaded_tokenizer) != target_vocab_size:
            raise ValueError(
                "Reloaded expanded tokenizer vocabulary size changed during publication."
            )

        _publish_directory_atomically(
            staging_dir,
            output_dir,
            overwrite=overwrite_output,
        )
        published = True
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(f"[OK] Authenticated expanded model package published to: {output_dir}")


def build_expanded_config(
    args: argparse.Namespace,
    device: str,
    *,
    source_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if source_artifact is None:
        source_artifact = _load_source_artifact(
            args.old_model_path,
            device,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
    final_config = copy.deepcopy(source_artifact["config"])
    source_revision = normalize_architecture_revision(
        final_config.get("architecture_revision")
    )

    updated_dims = {key: getattr(args, key) for key in ARCH_UPDATE_KEYS if getattr(args, key, None) is not None}
    if updated_dims:
        print("Updating model/config values:")
        for key, value in updated_dims.items():
            print(f"  - {key}: {final_config.get(key, 'N/A')} -> {value}")
        final_config.update(updated_dims)

    current_ctx = final_config.get("context_dim")
    if args.context_dim is not None and current_ctx is not None:
        if args.h_hidden is None and final_config.get("h_hidden") != current_ctx:
            print(f"  - [Auto-Sync] h_hidden: {final_config.get('h_hidden', 'N/A')} -> {current_ctx}")
            final_config["h_hidden"] = current_ctx
        if args.l_hidden is None and final_config.get("l_hidden") != current_ctx:
            print(f"  - [Auto-Sync] l_hidden: {final_config.get('l_hidden', 'N/A')} -> {current_ctx}")
            final_config["l_hidden"] = current_ctx

    if getattr(args, "rwkv_head_size", None) is not None:
        final_config["h_rwkv_head_size"] = int(args.rwkv_head_size)
        final_config["l_rwkv_head_size"] = int(args.rwkv_head_size)

    if getattr(args, "use_deepembed", None) is not None:
        if not args.use_deepembed:
            final_config["deepembed_mode"] = "off"
        elif final_config.get("deepembed_mode") in (None, "", "auto", "off"):
            final_config["deepembed_mode"] = (
                "shared-factorized"
                if source_revision == "coherent-v9"
                else "legacy-table"
            )
    if getattr(args, "use_rosa", None) is not None:
        if not args.use_rosa:
            final_config["rosa_embedding_mode"] = "off"
        elif final_config.get("rosa_embedding_mode") in (None, "", "auto", "off"):
            final_config["rosa_embedding_mode"] = (
                "shared-factorized"
                if source_revision == "coherent-v9"
                else "legacy-table"
            )

    if (
        getattr(args, "token_adapter_rank", None) is None
        and current_ctx is not None
        and final_config.get("token_adapter_rank") is not None
        and int(final_config["token_adapter_rank"]) > int(current_ctx)
    ):
        print(
            f"  - [Auto-Cap] token_adapter_rank: "
            f"{final_config['token_adapter_rank']} -> {current_ctx}"
        )
        final_config["token_adapter_rank"] = int(current_ctx)

    new_max_len = None
    if args.auto_max_length:
        if not args.dataset_for_length:
            raise ValueError("--auto-max-length requires --dataset-for-length.")

        try:
            tokenizer = source_artifact["tokenizer"]
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            determined_len = scan_dataset_for_max_length(args.dataset_for_length, tokenizer, args.kayla, args.alpaca)
            if determined_len > 0:
                new_max_len = determined_len
        except Exception as exc:
            print(f"[WARN] Error during auto-scan for max length: {exc}. Falling back.")
    elif args.new_max_length is not None:
        new_max_len = args.new_max_length

    if new_max_len is not None:
        print(f"Updating max_length: {final_config.get('max_length', 'N/A')} -> {new_max_len}")
        final_config["max_length"] = new_max_len

    for key, value in LATEST_CONFIG_DEFAULTS.items():
        if key not in final_config:
            final_config[key] = value

    target_revision = normalize_architecture_revision(
        final_config.get("architecture_revision")
    )
    if target_revision != source_revision:
        raise ValueError(
            "Expansion may resize a model only within its authenticated "
            f"architecture revision ({source_revision}); got {target_revision}."
        )
    expected_vocab = int(source_artifact["tokenizer_identity"]["vocab_size"])
    if int(final_config.get("vocab_size", 0) or 0) != expected_vocab:
        raise ValueError(
            "Expansion cannot alter tokenizer row IDs. The target vocab_size must "
            f"remain {expected_vocab}."
        )

    final_config["compile"] = False
    final_config.setdefault("model_type", "hierarchos")
    architecture_contract(final_config)
    final_config["architecture_contract_sha256"] = architecture_contract_hash(
        final_config
    )
    return final_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a trained Hierarchos model into the latest architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--old-model-path",
        type=str,
        required=True,
        help="Path to a trained model directory or .pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        "--output-path",
        dest="output_dir",
        type=str,
        required=True,
        help="Directory for the expanded model. Legacy .pt output paths are treated as their parent directory.",
    )

    dim_group = parser.add_argument_group("Latest Architecture Overrides")
    dim_group.add_argument("--vocab_size", "--vocab-size", dest="vocab_size", type=int)
    dim_group.add_argument("--context_dim", "--context-dim", dest="context_dim", type=int)
    dim_group.add_argument("--persistent_dim", "--persistent-dim", dest="persistent_dim", type=int)
    dim_group.add_argument("--ltm_slots", "--ltm-slots", dest="ltm_slots", type=int)
    dim_group.add_argument("--ltm_key_dim", "--ltm-key-dim", dest="ltm_key_dim", type=int)
    dim_group.add_argument("--ltm_val_dim", "--ltm-val-dim", dest="ltm_val_dim", type=int)
    dim_group.add_argument("--ltm_lr", "--ltm-lr", dest="ltm_lr", type=float)
    dim_group.add_argument("--ltm_topk", "--ltm-topk", dest="ltm_topk", type=int)
    dim_group.add_argument("--h_hidden", "--h-hidden", dest="h_hidden", type=int)
    dim_group.add_argument("--l_hidden", "--l-hidden", dest="l_hidden", type=int)
    dim_group.add_argument("--h_stride", "--h-stride", dest="h_stride", type=int)
    dim_group.add_argument("--max_h_steps", "--max-h-steps", dest="max_h_steps", type=int)
    dim_group.add_argument("--max_l_steps", "--max-l-steps", dest="max_l_steps", type=int)
    dim_group.add_argument("--l_conv_atol", "--l-conv-atol", dest="l_conv_atol", type=float)
    dim_group.add_argument("--commitment_threshold", "--commitment-threshold", dest="commitment_threshold", type=float)
    dim_group.add_argument("--detach_every_n_steps", "--detach-every-n-steps", dest="detach_every_n_steps", type=int)
    dim_group.add_argument("--h_halt_thresh", "--h-halt-thresh", dest="h_halt_thresh", type=float)
    dim_group.add_argument("--ltm_forget_rate", "--ltm-forget-rate", dest="ltm_forget_rate", type=float)
    dim_group.add_argument(
        "--token_adapter_rank",
        "--token-adapter-rank",
        dest="token_adapter_rank",
        type=int,
    )
    dim_group.add_argument("--rosa_max_context", "--rosa-max-context", dest="rosa_max_context", type=int)
    dim_group.add_argument("--rwkv_head_size", "--rwkv-head-size", dest="rwkv_head_size", type=int)
    deepembed_group = dim_group.add_mutually_exclusive_group()
    deepembed_group.add_argument("--use-deepembed", dest="use_deepembed", action="store_true", default=None)
    deepembed_group.add_argument("--no-deepembed", dest="use_deepembed", action="store_false")
    rosa_group = dim_group.add_mutually_exclusive_group()
    rosa_group.add_argument("--use-rosa", dest="use_rosa", action="store_true", default=None)
    rosa_group.add_argument("--no-rosa", dest="use_rosa", action="store_false")
    router_group = dim_group.add_mutually_exclusive_group()
    router_group.add_argument(
        "--memory-token-routers",
        dest="memory_token_routers",
        action="store_true",
        default=None,
    )
    router_group.add_argument(
        "--no-memory-token-routers",
        dest="memory_token_routers",
        action="store_false",
    )
    dim_group.add_argument(
        "--memory-gate-warmup-steps",
        dest="memory_gate_warmup_steps",
        type=int,
    )
    dim_group.add_argument(
        "--memory-gate-warmup-floor",
        dest="memory_gate_warmup_floor",
        type=float,
    )

    length_group = parser.add_argument_group("Sequence Length Expansion")
    length_group.add_argument("--new-max-length", type=int, help="Manually specify the new maximum sequence length.")
    length_group.add_argument(
        "--auto-max-length",
        action="store_true",
        help="Scan a dataset to determine max_length. Requires --dataset-for-length.",
    )
    length_group.add_argument("--dataset-for-length", type=str, help="Dataset (.jsonl or .json) for --auto-max-length.")
    scan_format_group = length_group.add_mutually_exclusive_group()
    scan_format_group.add_argument("--kayla", action="store_true", help="Use Kayla formatting for auto length scanning.")
    scan_format_group.add_argument("--alpaca", action="store_true", help="Use Alpaca instruction/input/output formatting for auto length scanning.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Explicitly allow tokenizer repositories to execute custom Python "
            "code. Disabled by default; use only for a repository you trust."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Atomically replace an existing output package after validation.",
    )

    args = parser.parse_args()
    device = "cpu"

    source_artifact = _load_source_artifact(
        args.old_model_path,
        device,
        trust_remote_code=args.trust_remote_code,
    )
    final_config = build_expanded_config(
        args,
        device,
        source_artifact=source_artifact,
    )
    transplant_weights(
        args.old_model_path,
        final_config,
        args.output_dir,
        device,
        trust_remote_code=args.trust_remote_code,
        overwrite_output=args.overwrite_output,
        source_artifact=source_artifact,
    )


if __name__ == "__main__":
    main()
