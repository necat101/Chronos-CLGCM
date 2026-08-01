import json
import os
import hashlib
import zipfile
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .device import is_directml_device
from ..models.revisions import (
    COHERENT_REVISION,
    normalize_architecture_revision,
    validate_architecture_contract,
)
from .safe_loading import load_tensor_payload_safely

TRANSIENT_LTM_STATE_KEYS = (
    "ltm.fast_vals",
    "ltm._mom_vals",
    "ltm.timestamps",
    "ltm.sources",
    "ltm.wallclock_timestamps",
)

DETERMINISTIC_STATE_KEYS = (
    "time_freqs",
)

RUNTIME_CHECKPOINT_METADATA_KEYS = (
    "checkpoint_version",
    "checkpoint_kind",
    "derived_from_checkpoint_kind",
    "derived_from_checkpoint_version",
    "derived_from_checkpoint_sha256",
    "completed_epoch",
    "tokenizer_identity",
    "expansion_provenance",
    "run_identity",
    "best_metric_state",
    "selection_metric",
    "effective_training_config",
    "optimizer_grouping_version",
    "training_complete",
)

LTM_PERSISTENT_METADATA_VERSION = 1
LTM_WALLCLOCK_SEMANTICS = "unix-seconds-utc-user-memory-write-v1"


def _validate_run_identity_digest(checkpoint: Dict[str, Any], source: str) -> bool:
    """Verify the self-digest on exact-run metadata when one is present."""
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, dict):
        return False
    saved_digest = run_identity.get("sha256")
    if saved_digest is None:
        return False
    if (
        not isinstance(saved_digest, str)
        or len(saved_digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in saved_digest)
    ):
        raise ValueError(f"Run identity in {source} has an invalid SHA-256 digest.")
    digest_payload = {
        key: value
        for key, value in run_identity.items()
        if key != "sha256"
    }
    actual_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if actual_digest != saved_digest.lower():
        raise ValueError(
            f"Run identity SHA-256 verification failed for {source}: "
            "checkpoint provenance/tokenizer/objective metadata is inconsistent."
        )
    return True


def _validate_expansion_provenance(checkpoint: Dict[str, Any], source: str) -> bool:
    """Verify the self-digest and output bindings of expansion lineage metadata."""
    provenance = checkpoint.get("expansion_provenance")
    if provenance is None:
        return False
    if not isinstance(provenance, dict):
        raise ValueError(f"Expansion provenance in {source} must be a dictionary.")
    saved_digest = provenance.get("sha256")
    if (
        not isinstance(saved_digest, str)
        or len(saved_digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in saved_digest)
    ):
        raise ValueError(f"Expansion provenance in {source} has an invalid SHA-256 digest.")
    digest_payload = {
        key: value
        for key, value in provenance.items()
        if key != "sha256"
    }
    actual_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if actual_digest != saved_digest.lower():
        raise ValueError(
            f"Expansion provenance SHA-256 verification failed for {source}."
        )

    source_info = provenance.get("source")
    expanded_info = provenance.get("expanded")
    if not isinstance(source_info, dict) or not isinstance(expanded_info, dict):
        raise ValueError(
            f"Expansion provenance in {source} must contain source and expanded mappings."
        )
    source_checkpoint_digest = source_info.get("checkpoint_sha256")
    if (
        not isinstance(source_checkpoint_digest, str)
        or len(source_checkpoint_digest) != 64
        or any(
            char not in "0123456789abcdefABCDEF"
            for char in source_checkpoint_digest
        )
    ):
        raise ValueError(
            f"Expansion provenance in {source} has an invalid source checkpoint digest."
        )

    checkpoint_contract_hash = checkpoint.get("architecture_contract_sha256")
    provenance_contract_hash = expanded_info.get(
        "architecture_contract_sha256"
    )
    if (
        checkpoint_contract_hash is not None
        and str(checkpoint_contract_hash).strip().lower()
        != str(provenance_contract_hash or "").strip().lower()
    ):
        raise ValueError(
            f"Expansion provenance architecture hash disagrees with the checkpoint in {source}."
        )
    direct_tokenizer = checkpoint.get("tokenizer_identity")
    provenance_tokenizer = expanded_info.get("tokenizer_identity")
    if (
        isinstance(direct_tokenizer, dict)
        and isinstance(provenance_tokenizer, dict)
        and direct_tokenizer != provenance_tokenizer
    ):
        raise ValueError(
            f"Expansion provenance tokenizer identity disagrees with the checkpoint in {source}."
        )
    return True


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
    checksum_path = path + ".sha256"
    checkpoint_source = path
    verified_checkpoint_file = None
    if os.path.exists(checksum_path):
        with open(checksum_path, "r", encoding="utf-8") as checksum_file:
            checksum_parts = checksum_file.read().strip().split()
        if not checksum_parts:
            raise RuntimeError(f"Checkpoint SHA-256 sidecar is empty: {checksum_path}")
        expected = checksum_parts[0].lower()
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise RuntimeError(
                f"Checkpoint SHA-256 sidecar is malformed: {checksum_path}"
            )
        hasher = hashlib.sha256()
        # Hash and deserialize the same open file description. Reopening by
        # path after verification leaves a TOCTOU window where a replacement
        # file can bypass the sidecar identity check.
        verified_checkpoint_file = open(path, "rb")
        try:
            while True:
                chunk = verified_checkpoint_file.read(8 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
        except Exception:
            verified_checkpoint_file.close()
            raise
        actual = hasher.hexdigest()
        if actual != expected:
            verified_checkpoint_file.close()
            raise RuntimeError(
                f"Checkpoint SHA-256 verification failed for {path}: "
                f"expected={expected!r}, actual={actual!r}"
            )
        verified_checkpoint_file.seek(0)
        checkpoint_source = verified_checkpoint_file
    # Current exact-resume checkpoints can contain the deterministic,
    # project-owned ROSA automaton carried at a TBPTT boundary. Keep the
    # allowlist narrow: arbitrary user classes must remain rejected.
    from .rosa import ROSAState

    allowed_globals = [
        AttrDict,
        ROSAState,
        *_legacy_numpy_checkpoint_safe_globals(),
    ]
    try:
        return load_tensor_payload_safely(
            checkpoint_source,
            map_location=map_location,
            allowed_globals=allowed_globals,
        )
    finally:
        if verified_checkpoint_file is not None:
            verified_checkpoint_file.close()

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
    model_config = getattr(model, "config", {})
    model_revision = normalize_architecture_revision(
        model_config.get("architecture_revision")
        if isinstance(model_config, dict)
        else getattr(model_config, "architecture_revision", None)
    )
    if model_revision != COHERENT_REVISION:
        # This buffer did not exist in historical checkpoints. In coherent-v9
        # it is schedule state and is required for train/eval/chat logit parity.
        allowed_missing.add("memory_gate_warmup_step")
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
    has_legacy_deepembed = any(
        key.startswith("h_deepemb.") or key.startswith("l_deepemb.")
        for key in state_dict
    )
    has_shared_deepembed = any(
        key.startswith("h_deepembed_adapter.")
        or key.startswith("l_deepembed_adapter.")
        for key in state_dict
    )
    if "deepembed_mode" not in config_dict:
        if has_shared_deepembed:
            config_dict["deepembed_mode"] = "shared-factorized"
        elif has_legacy_deepembed:
            config_dict["deepembed_mode"] = "legacy-table"
        else:
            config_dict["deepembed_mode"] = "off"
    if "use_deepembed" not in config_dict:
        config_dict["use_deepembed"] = bool(
            has_legacy_deepembed or has_shared_deepembed
        )

    has_legacy_rosa = any(key.startswith("rosa_emb.") for key in state_dict)
    has_shared_rosa = any(key.startswith("rosa_adapter.") for key in state_dict)
    if "rosa_embedding_mode" not in config_dict:
        if has_shared_rosa:
            config_dict["rosa_embedding_mode"] = "shared-factorized"
        elif has_legacy_rosa:
            config_dict["rosa_embedding_mode"] = "legacy-table"
        else:
            config_dict["rosa_embedding_mode"] = "off"
    if "use_rosa" not in config_dict:
        config_dict["use_rosa"] = bool(has_legacy_rosa or has_shared_rosa)

    if (
        "architecture_revision" not in config_dict
        and (has_shared_deepembed or has_shared_rosa)
    ):
        config_dict["architecture_revision"] = "coherent-v9"

    if config_dict.get("use_rosa", True) and "rosa_max_context" not in config_dict:
        config_dict["rosa_max_context"] = 512

    if "memory_token_routers" not in config_dict:
        has_router_weights = any(
            key.startswith("rosa_router.") or key.startswith("ltm_router.")
            for key in state_dict
        )
        config_dict["memory_token_routers"] = has_router_weights

    h_head_shape = state_dict.get("h_rnn.r_k")
    l_head_shape = state_dict.get("l_rnn.r_k")
    if (
        "h_rwkv_head_size" not in config_dict
        and torch.is_tensor(h_head_shape)
        and h_head_shape.ndim == 2
    ):
        config_dict["h_rwkv_head_size"] = int(h_head_shape.shape[1])
    if (
        "l_rwkv_head_size" not in config_dict
        and torch.is_tensor(l_head_shape)
        and l_head_shape.ndim == 2
    ):
        config_dict["l_rwkv_head_size"] = int(l_head_shape.shape[1])
    if "rwkv_head_size" not in config_dict:
        h_head = config_dict.get("h_rwkv_head_size")
        l_head = config_dict.get("l_rwkv_head_size")
        if h_head is not None and (l_head is None or h_head == l_head):
            config_dict["rwkv_head_size"] = int(h_head)
        elif l_head is not None and h_head is None:
            config_dict["rwkv_head_size"] = int(l_head)
        else:
            config_dict["rwkv_head_size"] = None

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


def validate_checkpoint_architecture_contract(
    checkpoint: Dict[str, Any],
    config_dict: Dict[str, Any],
    source: str = "checkpoint",
) -> bool:
    """Verify every persisted copy of the learned-function contract.

    The redundant copies are intentional: training checkpoints, inference
    exports, and exact-resume identity each remain independently inspectable.
    Any disagreement is treated as corruption/configuration drift.
    """
    _validate_run_identity_digest(checkpoint, source)
    _validate_expansion_provenance(checkpoint, source)

    contracts = []
    hashes = []

    if isinstance(checkpoint.get("architecture_contract"), dict):
        contracts.append(("checkpoint", checkpoint["architecture_contract"]))
    if checkpoint.get("architecture_contract_sha256") is not None:
        hashes.append(("checkpoint", checkpoint["architecture_contract_sha256"]))

    config_hash = config_dict.get("architecture_contract_sha256")
    if config_hash is not None:
        hashes.append(("config", config_hash))

    run_identity = checkpoint.get("run_identity")
    if isinstance(run_identity, dict):
        if isinstance(run_identity.get("architecture_contract"), dict):
            contracts.append(("run_identity", run_identity["architecture_contract"]))
        if run_identity.get("architecture_contract_sha256") is not None:
            hashes.append((
                "run_identity",
                run_identity["architecture_contract_sha256"],
            ))

    if len(contracts) > 1:
        canonical_contract = contracts[0][1]
        disagreements = [
            name for name, value in contracts[1:]
            if value != canonical_contract
        ]
        if disagreements:
            raise ValueError(
                f"Conflicting architecture contracts in {source}: "
                f"{contracts[0][0]} disagrees with {', '.join(disagreements)}."
            )
    if len(hashes) > 1:
        canonical_hash = str(hashes[0][1]).strip().lower()
        disagreements = [
            name for name, value in hashes[1:]
            if str(value).strip().lower() != canonical_hash
        ]
        if disagreements:
            raise ValueError(
                f"Conflicting architecture contract hashes in {source}: "
                f"{hashes[0][0]} disagrees with {', '.join(disagreements)}."
            )

    expected_contract = contracts[0][1] if contracts else None
    expected_hash = hashes[0][1] if hashes else None
    if expected_contract is None and expected_hash is None:
        revision = normalize_architecture_revision(
            config_dict.get("architecture_revision")
        )
        checkpoint_version = int(checkpoint.get("checkpoint_version", 0) or 0)
        if revision == COHERENT_REVISION or checkpoint_version >= 4:
            raise ValueError(
                f"Missing architecture contract metadata in {source}. "
                "coherent-v9 and checkpoint format v4+ require a serialized "
                "contract so recurrent/inference semantics cannot drift silently."
            )
        print(
            "WARNING: Legacy checkpoint has no architecture contract metadata; "
            "resolved legacy settings will be used for compatibility."
        )
        return False

    validate_architecture_contract(
        config_dict,
        expected_contract=expected_contract,
        expected_hash=expected_hash,
        source=source,
    )
    print(f"INFO: Architecture contract verified for {source}.")
    return True


def _restore_persistent_ltm_metadata(model, checkpoint: Dict[str, Any], source: str) -> bool:
    """Restore metadata for an explicitly consolidated inference LTM export."""
    metadata = checkpoint.get("ltm_persistent_metadata")
    if metadata is None:
        return False
    if checkpoint.get("checkpoint_kind") != "inference-ltm-consolidated":
        raise ValueError(
            f"Unexpected persistent LTM metadata in {source}: only an "
            "inference-ltm-consolidated checkpoint may carry it."
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"Persistent LTM metadata in {source} must be a dictionary.")
    try:
        version = int(metadata.get("version", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Persistent LTM metadata in {source} has an invalid version.") from exc
    if version != LTM_PERSISTENT_METADATA_VERSION:
        raise ValueError(
            f"Unsupported persistent LTM metadata version {version} in {source}."
        )
    if metadata.get("wallclock_semantics") != LTM_WALLCLOCK_SEMANTICS:
        raise ValueError(
            f"Persistent LTM metadata in {source} has unsupported wall-clock semantics."
        )
    ltm = getattr(model, "ltm", None)
    if ltm is None:
        raise ValueError(f"Persistent LTM metadata in {source} has no target LTM module.")

    with torch.no_grad():
        for name in ("timestamps", "sources", "wallclock_timestamps"):
            value = metadata.get(name)
            target = getattr(ltm, name, None)
            if not torch.is_tensor(value) or not torch.is_tensor(target):
                raise ValueError(
                    f"Persistent LTM metadata {name!r} is missing or is not a tensor "
                    f"in {source}."
                )
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Persistent LTM metadata {name!r} shape mismatch in {source}: "
                    f"saved={tuple(value.shape)}, expected={tuple(target.shape)}."
                )
            if name == "sources":
                if value.dtype != torch.long:
                    raise ValueError(
                        f"Persistent LTM metadata 'sources' must use torch.int64 in {source}."
                    )
                min_source = int(value.min().item()) if value.numel() else 0
                max_source = int(value.max().item()) if value.numel() else 0
                allowed_max = max(
                    int(getattr(ltm, "SRC_UNKNOWN", 0)),
                    int(getattr(ltm, "SRC_USER_INTERACTION", 1)),
                    int(getattr(ltm, "SRC_TRAINING_DATA", 2)),
                    int(getattr(ltm, "SRC_CORRECTION", 3)),
                )
                if min_source < 0 or max_source > allowed_max:
                    raise ValueError(
                        f"Persistent LTM metadata 'sources' contains an unknown source "
                        f"identifier in {source}."
                    )
            else:
                if not value.is_floating_point():
                    raise ValueError(
                        f"Persistent LTM metadata {name!r} must be floating point in {source}."
                    )
                finite_nonnegative = torch.isfinite(value) & (value >= 0)
                if not bool(finite_nonnegative.all().item()):
                    raise ValueError(
                        f"Persistent LTM metadata {name!r} contains a non-finite or "
                        f"negative value in {source}."
                    )
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return True


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
    validate_checkpoint_architecture_contract(
        checkpoint,
        config_dict,
        weights_path,
    )

    config = AttrDict(config_dict)

    from ..models.core import HierarchosCore
    model = HierarchosCore(config)
    
    load_result = load_model_state_dict_compatible(model, state_dict, weights_path)
    if (
        "memory_gate_warmup_step" in load_result.missing_keys
        and normalize_architecture_revision(
            config_dict.get("architecture_revision")
        ) != COHERENT_REVISION
        and hasattr(model, "memory_gate_warmup_step")
    ):
        # Historical eval/chat bypassed the warmup floor entirely. Treat a
        # legacy inference export with no schedule buffer as warmup-complete so
        # this parity fix does not restart its gate curriculum at inference.
        with torch.no_grad():
            model.memory_gate_warmup_step.fill_(
                float(config_dict.get("memory_gate_warmup_steps", 0) or 0)
            )
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
    _restore_persistent_ltm_metadata(model, checkpoint, weights_path)
    # Retain only small identity metadata needed by inference-side re-exports.
    # Keeping it on the loaded model avoids rereading a multi-gigabyte checkpoint
    # merely to preserve its contract and training provenance after an LTM edit.
    model._hierarchos_checkpoint_metadata = {
        key: copy.deepcopy(checkpoint[key])
        for key in RUNTIME_CHECKPOINT_METADATA_KEYS
        if key in checkpoint
    }
    model._hierarchos_checkpoint_metadata["source_weights_path"] = weights_path
    model.eval()
    return model, config

def save_checkpoint_safely(checkpoint_dict: Dict[str, Any], path: str):
    """Save atomically with ZIP readback, SHA-256 identity, and backup."""
    temp_path = path + ".tmp"
    temp_checksum_path = temp_path + ".sha256"
    checksum_path = path + ".sha256"
    backup_path = path + ".bak"
    backup_checksum_path = backup_path + ".sha256"
    moved_existing_to_backup = False
    published_new_checkpoint = False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_checksum_path):
            os.remove(temp_checksum_path)
        torch.save(checkpoint_dict, temp_path)
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("Failed to save checkpoint: Temp file is missing or empty.")
        # torch.save's default container is ZIP. CRC-test every member without
        # materializing a second copy of all tensors in RAM.
        if zipfile.is_zipfile(temp_path):
            with zipfile.ZipFile(temp_path, "r") as archive:
                corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise RuntimeError(
                    f"Checkpoint readback found a corrupt ZIP member: {corrupt_member}"
                )
        # Windows requires a writable descriptor for fsync/FlushFileBuffers.
        with open(temp_path, "r+b") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        hasher = hashlib.sha256()
        with open(temp_path, "rb") as checkpoint_file:
            while True:
                chunk = checkpoint_file.read(8 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
        digest = hasher.hexdigest()
        with open(temp_checksum_path, "w", encoding="utf-8") as checksum_file:
            checksum_file.write(f"{digest}  {os.path.basename(path)}\n")
            checksum_file.flush()
            os.fsync(checksum_file.fileno())

        if os.path.exists(path):
            if os.path.exists(backup_path):
                os.remove(backup_path)
            if os.path.exists(backup_checksum_path):
                os.remove(backup_checksum_path)
            os.replace(path, backup_path)
            moved_existing_to_backup = True
            if os.path.exists(checksum_path):
                os.replace(checksum_path, backup_checksum_path)

        os.replace(temp_path, path)
        published_new_checkpoint = True
        os.replace(temp_checksum_path, checksum_path)
        print(
            f"INFO: Checkpoint saved safely to {path} "
            f"(sha256={digest[:16]}..., ZIP readback passed)"
        )
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint safely: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_checksum_path):
            os.remove(temp_checksum_path)
        if published_new_checkpoint and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as remove_error:
                print(
                    f"CRITICAL: Could not remove incompletely published checkpoint "
                    f"'{path}': {remove_error}"
                )
        if published_new_checkpoint and os.path.exists(checksum_path):
            try:
                os.remove(checksum_path)
            except OSError:
                pass
        if moved_existing_to_backup and not os.path.exists(path) and os.path.exists(backup_path):
            try:
                os.replace(backup_path, path)
                if (
                    not os.path.exists(checksum_path)
                    and os.path.exists(backup_checksum_path)
                ):
                    os.replace(backup_checksum_path, checksum_path)
                print(f"INFO: Restored previous checkpoint after failed save: {path}")
            except OSError as restore_error:
                print(f"CRITICAL: Could not restore checkpoint backup '{backup_path}': {restore_error}")
        raise
