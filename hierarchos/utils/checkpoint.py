import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .device import is_directml_device

TRANSIENT_LTM_STATE_KEYS = (
    "ltm.fast_vals",
    "ltm._mom_vals",
    "ltm.timestamps",
    "ltm.sources",
)

DETERMINISTIC_STATE_KEYS = (
    "time_freqs",
)


def _clean_state_dict_key(key: str) -> str:
    """Remove torch.compile wrapper path components without rewriting real names."""
    return ".".join(part for part in str(key).split(".") if part != "_orig_mod")


def _state_values_equal(left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            return False
        if left is right:
            return True
        try:
            if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
                return (
                    left.storage_offset() == right.storage_offset()
                    and left.stride() == right.stride()
                )
        except (AttributeError, RuntimeError):
            pass
        return bool(torch.equal(left, right))
    return left == right


def sanitize_model_state_dict(model_or_state_dict, reset_transient_ltm: bool = True) -> Dict[str, torch.Tensor]:
    """Return a save-ready state_dict with compile prefixes removed and transient LTM state zeroed."""
    source_state = model_or_state_dict.state_dict() if hasattr(model_or_state_dict, "state_dict") else model_or_state_dict
    clean_state = {}
    source_keys = {}
    for key, value in source_state.items():
        clean_key = _clean_state_dict_key(key)
        is_transient_ltm = any(clean_key.endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS)
        if reset_transient_ltm and is_transient_ltm:
            clean_value = torch.zeros_like(value)
        elif is_transient_ltm:
            # Checkpoint validation may repair/reset transient working memory.
            # Clone only these small buffers so saving can never mutate the live
            # model while avoiding a full learned-weight copy.
            clean_value = value.detach().clone()
        else:
            clean_value = value

        if clean_key in clean_state:
            if not _state_values_equal(clean_state[clean_key], clean_value):
                raise ValueError(
                    "Conflicting checkpoint keys collapse to the same name after removing "
                    f"torch.compile prefixes: {source_keys[clean_key]!r} and {key!r} -> {clean_key!r}."
                )
            continue

        clean_state[clean_key] = clean_value
        source_keys[clean_key] = key
    return clean_state


# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _legacy_numpy_checkpoint_safe_globals():
    """Return the narrow NumPy allowlist needed by saved MT19937 RNG state."""
    import numpy as np

    try:
        from numpy._core.multiarray import _reconstruct as numpy_reconstruct
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import _reconstruct as numpy_reconstruct

    # Training checkpoints saved before v0.21 contain np.random.get_state().
    # NumPy changed the pickle module path at 2.0, so accept both spellings for
    # this one function. The dynamically-created uint32 dtype class is not
    # reported by get_unsafe_globals_in_checkpoint(), but PyTorch requires it.
    reconstruct_globals = [numpy_reconstruct]
    if hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        # PyTorch 2.6+ accepts an explicit pickle path alongside the callable,
        # which makes NumPy 1.x-created checkpoints portable to NumPy 2.x and
        # vice versa. PyTorch 2.5's safe_globals accepts callables only.
        reconstruct_globals = [
            (numpy_reconstruct, "numpy._core.multiarray._reconstruct"),
            (numpy_reconstruct, "numpy.core.multiarray._reconstruct"),
        ]

    return [
        *reconstruct_globals,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]


def load_checkpoint_payload_compatible(path: str, map_location="cpu"):
    """Load Hierarchos payloads safely, including legacy NumPy RNG metadata."""
    try:
        from torch.serialization import safe_globals
    except (ImportError, AttributeError):
        # PyTorch releases predating safe_globals retain their legacy loader.
        return torch.load(path, map_location=map_location)

    allowed_globals = [AttrDict, *_legacy_numpy_checkpoint_safe_globals()]
    with safe_globals(allowed_globals):
        return torch.load(path, map_location=map_location, weights_only=True)

def _resolve_weights_path(model_path: str) -> Tuple[str, str]:
    """Resolve a Hierarchos model source to (weights_path, model_dir)."""
    if not model_path:
        raise FileNotFoundError("No model path was provided.")

    resolved = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isfile(resolved):
        if not resolved.lower().endswith(".pt"):
            raise FileNotFoundError(f"Model file must be a .pt checkpoint: {resolved}")
        return resolved, os.path.dirname(resolved)

    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    preferred = ("hierarchos.pt", "model.pt", "hierarchos_final.pt")
    preferred_candidates = [
        os.path.join(resolved, name)
        for name in preferred
        if os.path.exists(os.path.join(resolved, name))
    ]
    if preferred_candidates:
        # Converter runs used to write model.pt while training exports write
        # hierarchos.pt. If both exist, loading by fixed name order can silently
        # pick an older stale export. Prefer the most recently written known
        # checkpoint while keeping deterministic name tie-breaking.
        return max(
            preferred_candidates,
            key=lambda path: (os.path.getmtime(path), path),
        ), resolved

    pt_files = sorted(
        f for f in os.listdir(resolved)
        if f.lower().endswith(".pt")
    )
    if pt_files:
        pt_paths = [os.path.join(resolved, name) for name in pt_files]
        return max(pt_paths, key=lambda path: (os.path.getmtime(path), path)), resolved

    # Browser/Hugging Face downloads commonly wrap a model directory in one
    # same-named folder. Accept that layout only when it resolves unambiguously.
    nested_candidates = []
    for name in sorted(os.listdir(resolved)):
        nested_dir = os.path.join(resolved, name)
        if not os.path.isdir(nested_dir):
            continue
        nested_preferred = [
            os.path.join(nested_dir, filename)
            for filename in preferred
            if os.path.isfile(os.path.join(nested_dir, filename))
        ]
        if nested_preferred:
            nested_candidates.append(
                (
                    max(nested_preferred, key=lambda path: (os.path.getmtime(path), path)),
                    nested_dir,
                )
            )
    if len(nested_candidates) == 1:
        return nested_candidates[0]
    if len(nested_candidates) > 1:
        raise FileNotFoundError(
            f"Multiple nested model directories found in '{model_path}'; "
            "pass the intended directory explicitly."
        )

    raise FileNotFoundError(f"Model weights file not found in '{model_path}'")


def _load_json_config(model_dir: str) -> Optional[Dict[str, Any]]:
    for name in ("hierarchos_config.json", "config.json"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return dict(loaded)
    return None


def _has_v8_rwkv_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("h_rnn.x_r") or key.startswith("h_rnn.r_k")
        for key in state_dict
    )


def _has_legacy_rwkv_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("h_rnn.time_decay") or key.startswith("h_rnn.time_mix_")
        for key in state_dict
    )


def _reject_unsupported_rwkv_state_dict(state_dict: Dict[str, torch.Tensor], source: str = "checkpoint") -> None:
    if _has_legacy_rwkv_state_dict(state_dict):
        raise ValueError(
            f"Unsupported legacy scalar-RWKV checkpoint in {source}. "
            "The active modular Hierarchos path is v8-only and requires "
            "matrix-state RWKV keys such as 'h_rnn.x_r' and 'h_rnn.r_k'. "
            "Do not continue paid v8 training from this checkpoint."
        )


def _reject_rwkv_load_mismatch(missing_keys, unexpected_keys, source: str = "checkpoint") -> None:
    critical_prefixes = ("h_rnn.", "l_rnn.")
    missing = [key for key in missing_keys if str(key).startswith(critical_prefixes)]
    unexpected = [key for key in unexpected_keys if str(key).startswith(critical_prefixes)]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing RWKV keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            details.append(f"unexpected RWKV keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        raise ValueError(
            f"RWKV v8 checkpoint mismatch in {source}; "
            + "; ".join(details)
            + ". Refusing to continue because partial recurrent-block loading can produce incoherence."
        )


def _is_transient_ltm_state_key(key: str) -> bool:
    return any(str(key).endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS)


def _validate_tied_embedding_state_dict(state_dict: Dict[str, torch.Tensor], source: str) -> None:
    """Reject ambiguous checkpoints whose two aliases contain different weights."""
    tok_weight = state_dict.get("tok_emb.weight")
    head_weight = state_dict.get("lm_head.weight")
    if tok_weight is None or head_weight is None:
        return
    if not _state_values_equal(tok_weight, head_weight):
        raise ValueError(
            f"Tied embedding mismatch in {source}: 'tok_emb.weight' and 'lm_head.weight' "
            "contain different values. Loading order would otherwise choose one silently."
        )


def _validate_state_dict_finite(
    state_dict: Dict[str, torch.Tensor],
    source: str,
    allow_nonfinite_transient_ltm: bool = True,
) -> None:
    """Reject NaN/Inf learned tensors without allocating a checkpoint-sized mask."""
    chunk_elements = 1_048_576
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or not (value.is_floating_point() or value.is_complex()):
            continue
        if allow_nonfinite_transient_ltm and _is_transient_ltm_state_key(key):
            continue
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            if not bool(torch.isfinite(flat[start:start + chunk_elements]).all().item()):
                raise ValueError(
                    f"Non-finite tensor '{key}' in {source}. Refusing to load NaN/Inf "
                    "learned weights because they can make an otherwise complete checkpoint incoherent."
                )


def _adapt_legacy_qproj_weight(model, state_dict: Dict[str, torch.Tensor], source: str) -> Dict[str, torch.Tensor]:
    """Adapt the one supported context-only qproj layout deterministically."""
    old_weight = state_dict.get("qproj.weight")
    new_weight = getattr(getattr(model, "qproj", None), "weight", None)
    if not torch.is_tensor(old_weight) or not torch.is_tensor(new_weight) or old_weight.shape == new_weight.shape:
        return state_dict

    if (
        old_weight.ndim == 2
        and new_weight.ndim == 2
        and old_weight.shape[0] == new_weight.shape[0]
        and old_weight.shape[1] * 2 == new_weight.shape[1]
    ):
        adapted = old_weight.new_zeros(new_weight.shape)
        adapted[:, :old_weight.shape[1]].copy_(old_weight)
        adapted_state = dict(state_dict)
        adapted_state["qproj.weight"] = adapted
        print(
            f"INFO: Deterministically adapted qproj.weight from {tuple(old_weight.shape)} "
            f"to {tuple(new_weight.shape)} (new context columns initialized to zero)."
        )
        return adapted_state

    raise ValueError(
        f"Unsupported qproj.weight shape in {source}: checkpoint={tuple(old_weight.shape)}, "
        f"model={tuple(new_weight.shape)}. Refusing to leave qproj randomly initialized."
    )


def _reject_model_load_mismatch(model, state_dict, missing_keys, unexpected_keys, source: str) -> None:
    """Allow only deterministic/transient omissions and a single tied-weight alias."""
    _reject_rwkv_load_mismatch(missing_keys, unexpected_keys, source)

    state_keys = set(state_dict)
    allowed_missing = set(TRANSIENT_LTM_STATE_KEYS) | set(DETERMINISTIC_STATE_KEYS)
    if "tok_emb.weight" in state_keys and "lm_head.weight" not in state_keys:
        allowed_missing.add("lm_head.weight")
    if "lm_head.weight" in state_keys and "tok_emb.weight" not in state_keys:
        allowed_missing.add("tok_emb.weight")

    missing = [key for key in missing_keys if key not in allowed_missing]
    unexpected = list(unexpected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        raise ValueError(
            f"Incomplete Hierarchos checkpoint load in {source}; "
            + "; ".join(details)
            + ". Refusing to run with randomly initialized or unused learned tensors."
        )


def load_model_state_dict_compatible(model, state_dict: Dict[str, torch.Tensor], source: str = "checkpoint"):
    """Load a checkpoint completely while preserving documented legacy compatibility."""
    _validate_tied_embedding_state_dict(state_dict, source)
    _validate_state_dict_finite(state_dict, source)
    compatible_state = _adapt_legacy_qproj_weight(model, state_dict, source)
    load_result = model.load_state_dict(compatible_state, strict=False)
    _reject_model_load_mismatch(
        model,
        compatible_state,
        load_result.missing_keys,
        load_result.unexpected_keys,
        source,
    )
    return load_result


def _infer_arch_flags_from_state_dict(config_dict: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> None:
    """Backfill architecture toggles for checkpoints saved before these flags existed."""
    if "use_deepembed" not in config_dict:
        config_dict["use_deepembed"] = any(
            key.startswith("h_deepemb.") or key.startswith("l_deepemb.")
            for key in state_dict
        )

    if "use_rosa" not in config_dict:
        config_dict["use_rosa"] = any(
            key.startswith("rosa_emb.") or key == "rosa_gate_logit"
            for key in state_dict
        )

    if config_dict.get("use_rosa", True) and "rosa_max_context" not in config_dict:
        config_dict["rosa_max_context"] = 512

    if "memory_token_routers" not in config_dict:
        has_router_weights = any(
            key.startswith("rosa_router.") or key.startswith("ltm_router.")
            for key in state_dict
        )
        config_dict["memory_token_routers"] = has_router_weights

    if "rwkv_head_size" not in config_dict:
        head_shape = state_dict.get("h_rnn.r_k")
        if torch.is_tensor(head_shape) and head_shape.ndim == 2:
            config_dict["rwkv_head_size"] = int(head_shape.shape[1])

    if "rwkv_channel_mix_key_clamp" not in config_dict:
        config_dict["rwkv_channel_mix_key_clamp"] = 12.0

    if "rwkv_channel_mix_deepembed_clamp" not in config_dict:
        config_dict["rwkv_channel_mix_deepembed_clamp"] = 4.0

    # Refinement parity and recurrent geometry are independent. Some TBPTT
    # exports persisted full_sample_bptt=False and inference_logit_parity=True
    # before the explicit recurrence field was introduced; treating the parity
    # flag as full-sample recurrence silently disabled their training boundaries.
    if (
        "inference_recurrence_mode" not in config_dict
        and "full_sample_bptt" in config_dict
    ):
        config_dict["inference_recurrence_mode"] = (
            "full-sample" if bool(config_dict["full_sample_bptt"]) else "tbptt"
        )


def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model from a directory or direct .pt file."""
    weights_path, model_dir = _resolve_weights_path(model_path)

    try:
        checkpoint = load_checkpoint_payload_compatible(weights_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like .pt file.")

    config_dict = checkpoint.get('config') or _load_json_config(model_dir)
    if config_dict is None:
        raise ValueError(
            "Model config not found. Include 'config' in the checkpoint or add "
            "hierarchos_config.json next to the .pt file."
        )

    config_dict = dict(config_dict)
    if 'model_type' not in config_dict: config_dict['model_type'] = 'hierarchos'
    
    # Strip _orig_mod. prefix from compiled model checkpoints (torch.compile adds this)
    # Without this, strict=False silently drops ALL weights from compiled checkpoints
    if 'model_state_dict' in checkpoint:
        state_source = checkpoint['model_state_dict']
    else:
        state_source = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
        if not state_source:
            raise ValueError("Model state_dict not found in checkpoint.")
    state_dict = sanitize_model_state_dict(state_source, reset_transient_ltm=True)
    _reject_unsupported_rwkv_state_dict(state_dict, weights_path)
    _infer_arch_flags_from_state_dict(config_dict, state_dict)

    config = AttrDict(config_dict)

    from ..models.core import HierarchosCore
    model = HierarchosCore(config)
    
    load_result = load_model_state_dict_compatible(model, state_dict, weights_path)
    allowed_missing = [
        key for key in load_result.missing_keys
        if key in TRANSIENT_LTM_STATE_KEYS or key in DETERMINISTIC_STATE_KEYS
    ]
    if allowed_missing:
        print(f"INFO: Reinitialized {len(allowed_missing)} deterministic/transient state tensor(s).")
    print(f"INFO: All {len(state_dict)} checkpoint tensors loaded coherently.")
    model.to(device)
    if checkpoint.get('training_complete', False) and hasattr(model, 'reset_memory'):
        model.reset_memory()
    model.eval()
    return model, config

def save_checkpoint_safely(checkpoint_dict: Dict[str, Any], path: str):
    """Saves a checkpoint safely with validation and backup."""
    temp_path = path + ".tmp"
    backup_path = path + ".bak"
    moved_existing_to_backup = False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        torch.save(checkpoint_dict, temp_path)
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("Failed to save checkpoint: Temp file is missing or empty.")

        if os.path.exists(path):
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.replace(path, backup_path)
            moved_existing_to_backup = True

        os.replace(temp_path, path)
        print(f"INFO: Checkpoint saved safely to {path}")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint safely: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if moved_existing_to_backup and not os.path.exists(path) and os.path.exists(backup_path):
            try:
                os.replace(backup_path, path)
                print(f"INFO: Restored previous checkpoint after failed save: {path}")
            except OSError as restore_error:
                print(f"CRITICAL: Could not restore checkpoint backup '{backup_path}': {restore_error}")
        raise
