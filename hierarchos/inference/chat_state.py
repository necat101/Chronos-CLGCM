"""Helpers for Hierarchos chat runtime state files.

This module is intentionally inference-only. It knows how to describe and
normalize recurrent chat state tensors, including the RWKV v8 packed matrix
state layout, without touching model weights or training state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..models.revisions import (
    architecture_contract,
    architecture_contract_hash,
    validate_architecture_contract,
)


CHAT_STATE_KIND = "hierarchos_chat_runtime_state"
CHAT_STATE_VERSION = 4
LEGACY_CHAT_STATE_VERSIONS = frozenset({1, 2, 3})
RWKV_V8_LAYOUT = "rwkv_v8_matrix_packed"
LEGACY_SCALAR_LAYOUT = "legacy_scalar_wkv"

_SIGNATURE_KEYS = (
    "context_dim",
    "h_hidden",
    "l_hidden",
    "h_stride",
    "max_h_steps",
    "max_l_steps",
    "vocab_size",
)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _maybe_int(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def tensor_to_cpu(value: Any) -> Optional[torch.Tensor]:
    return value.detach().cpu().clone() if torch.is_tensor(value) else None


def clear_ltm_working_memory(model: Any) -> bool:
    """Clear transient LTM working memory for a fresh inference session."""
    ltm = getattr(model, "ltm", None)
    if ltm is None:
        return False

    if hasattr(ltm, "reset_working_memory"):
        ltm.reset_working_memory()
        return True

    cleared = False
    with torch.no_grad():
        for attr in ("fast_vals", "_mom_vals", "timestamps", "wallclock_timestamps"):
            value = getattr(ltm, attr, None)
            if torch.is_tensor(value):
                value.zero_()
                cleared = True

        sources = getattr(ltm, "sources", None)
        if torch.is_tensor(sources):
            sources.fill_(int(getattr(ltm, "SRC_UNKNOWN", 0)))
            cleared = True

    return cleared


def chat_state_config_signature(config: Any, model: Any = None) -> Dict[str, Any]:
    """Small architecture fingerprint for model-neutral chat state files."""
    signature: Dict[str, Any] = {}
    for key in _SIGNATURE_KEYS:
        value = _config_value(config, key, None)
        if value is not None:
            signature[key] = _maybe_int(value)

    rwkv_head_size = _config_value(config, "rwkv_head_size", None)
    if rwkv_head_size is None and model is not None:
        h_rnn = getattr(model, "h_rnn", None)
        rwkv_head_size = getattr(h_rnn, "head_size", None)
    if rwkv_head_size is not None:
        signature["rwkv_head_size"] = _maybe_int(rwkv_head_size)

    return signature


def chat_state_architecture_metadata(config: Any) -> Dict[str, Any]:
    """Return the exact learned-function contract for a new state file."""
    return {
        "architecture_contract": architecture_contract(config),
        "architecture_contract_sha256": architecture_contract_hash(config),
    }


def _cell_state_spec(cell: Any, fallback_hidden: Any = None) -> Dict[str, Any]:
    hidden = getattr(cell, "n_embd", fallback_hidden)
    state_size = getattr(cell, "state_size", None)
    head_size = getattr(cell, "head_size", None)
    n_head = getattr(cell, "n_head", None)

    if state_size is None and hidden is not None:
        state_size = 5

    layout = RWKV_V8_LAYOUT if state_size is not None and head_size is not None else LEGACY_SCALAR_LAYOUT
    spec: Dict[str, Any] = {"layout": layout}

    for key, value in (
        ("hidden", hidden),
        ("state_size", state_size),
        ("head_size", head_size),
        ("n_head", n_head),
        ("matrix_offset", getattr(cell, "matrix_offset", None)),
        ("state_readout_mode", getattr(cell, "state_readout_mode", None)),
        ("layer_id", getattr(cell, "layer_id", None)),
        ("n_layer", getattr(cell, "n_layer", None)),
    ):
        if value is not None:
            spec[key] = _maybe_int(value)

    return spec


def recurrent_state_layout(model: Any = None, config: Any = None) -> Dict[str, Dict[str, Any]]:
    """Return expected recurrent state layout for the loaded inference model."""
    layout: Dict[str, Dict[str, Any]] = {}
    for label, attr, hidden_key in (
        ("h", "h_rnn", "h_hidden"),
        ("l", "l_rnn", "l_hidden"),
    ):
        cell = getattr(model, attr, None) if model is not None else None
        fallback_hidden = _config_value(config, hidden_key, _config_value(config, "context_dim", None))
        if cell is None and fallback_hidden is None:
            continue
        layout[label] = _cell_state_spec(cell, fallback_hidden)
    return layout


def _shape(value: Any) -> Optional[list[int]]:
    if torch.is_tensor(value):
        return [int(dim) for dim in value.shape]
    return None


def _cell_exact_shape(cell: Any, batch_size: int) -> list[int]:
    hidden = getattr(cell, "n_embd", None)
    state_size = getattr(cell, "state_size", None)
    if hidden is None:
        raise RuntimeError("Recurrent cell does not expose n_embd geometry.")
    if state_size is None:
        # The supported legacy quantized RWKV cell has an exact five-slot
        # scalar state but predates the explicit ``state_size`` attribute.
        state_size = 5
    return [int(batch_size), int(hidden), int(state_size)]


def recurrent_state_metadata(
    *,
    model: Any = None,
    config: Any = None,
    h_state: Any = None,
    l_state: Any = None,
) -> Dict[str, Any]:
    return {
        "recurrent_state_layout": recurrent_state_layout(model, config),
        "recurrent_state_shapes": {
            "h_state": _shape(h_state),
            "l_state": _shape(l_state),
        },
    }


def _payload_version(payload: Dict[str, Any]) -> int:
    version = payload.get("version")
    if isinstance(version, bool):
        raise RuntimeError("Chat state has an invalid boolean version.")
    try:
        version = int(version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Chat state is missing a valid version.") from exc
    if version > CHAT_STATE_VERSION:
        raise RuntimeError(
            f"Chat state version {version} is newer than supported version "
            f"{CHAT_STATE_VERSION}."
        )
    if version != CHAT_STATE_VERSION and version not in LEGACY_CHAT_STATE_VERSIONS:
        raise RuntimeError(f"Unsupported chat state version: {version}.")
    return version


def _validate_strict_recurrent_shapes(
    payload: Dict[str, Any],
    model: Any,
) -> None:
    saved_shapes = payload.get("recurrent_state_shapes")
    if not isinstance(saved_shapes, dict):
        raise RuntimeError(
            "Version-4 chat state is missing recurrent_state_shapes metadata."
        )

    for state_name, module_name in (
        ("h_state", "h_rnn"),
        ("l_state", "l_rnn"),
    ):
        state = payload.get(state_name)
        actual_shape = _shape(state)
        declared_shape = saved_shapes.get(state_name)
        if declared_shape != actual_shape:
            raise RuntimeError(
                f"Chat state {state_name} shape metadata mismatch: "
                f"declared={declared_shape}, actual={actual_shape}."
            )
        if state is None:
            continue
        if state.dim() != 3 or int(state.shape[0]) <= 0:
            raise RuntimeError(
                f"Chat state {state_name} must have exact [B, C, S] shape; "
                f"got {actual_shape}."
            )
        cell = getattr(model, module_name, None) if model is not None else None
        if cell is None:
            raise RuntimeError(
                f"Cannot verify strict {state_name} geometry without model.{module_name}."
            )
        expected_shape = _cell_exact_shape(cell, int(state.shape[0]))
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Chat state recurrent tensor shape mismatch for {state_name}: "
                f"saved={actual_shape}, current={expected_shape}."
            )


def validate_chat_state_payload_compatible(
    payload: Dict[str, Any],
    config: Any,
    model: Any = None,
) -> bool:
    """Raise on incompatibility and return whether legacy migration is allowed."""
    version = _payload_version(payload)
    allow_legacy_migration = version in LEGACY_CHAT_STATE_VERSIONS
    saved = payload.get("config_signature") or {}
    current = chat_state_config_signature(config, model)
    signature_keys = (
        ("context_dim", "h_hidden", "l_hidden", "vocab_size")
        if allow_legacy_migration
        else _SIGNATURE_KEYS
    )
    for key in signature_keys:
        if key in saved and key in current and saved[key] != current[key]:
            raise RuntimeError(
                f"Chat state was saved for {key}={saved[key]}, "
                f"but the loaded model has {key}={current[key]}."
            )

    saved_layout = payload.get("recurrent_state_layout") or {}
    current_layout = recurrent_state_layout(model, config)
    for label in ("h", "l"):
        saved_spec = saved_layout.get(label) or {}
        current_spec = current_layout.get(label) or {}
        if not saved_spec or not current_spec:
            continue

        # Only explicitly versioned legacy files may migrate. A current-format
        # file must describe the exact active recurrent representation.
        if (
            allow_legacy_migration
            and saved_spec.get("layout") == LEGACY_SCALAR_LAYOUT
        ):
            continue

        layout_keys = (
            "layout",
            "hidden",
            "state_size",
            "head_size",
            "n_head",
            "matrix_offset",
            "state_readout_mode",
        )
        for key in layout_keys:
            if key in saved_spec and key in current_spec and saved_spec[key] != current_spec[key]:
                raise RuntimeError(
                    f"Chat state recurrent layout mismatch for {label}_state: "
                    f"saved {key}={saved_spec[key]}, current {key}={current_spec[key]}."
                )

    if allow_legacy_migration:
        return True

    if not isinstance(payload.get("recurrent_state_layout"), dict):
        raise RuntimeError(
            "Version-4 chat state is missing recurrent_state_layout metadata."
        )
    _validate_strict_recurrent_shapes(payload, model)

    expected_contract = payload.get("architecture_contract")
    expected_hash = payload.get("architecture_contract_sha256")
    if not isinstance(expected_contract, dict) or expected_hash is None:
        raise RuntimeError(
            "Version-4 chat state is missing its architecture contract/hash."
        )
    try:
        validate_architecture_contract(
            config,
            expected_contract=expected_contract,
            expected_hash=expected_hash,
            source="chat runtime state",
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    return False


def _legacy_initial_state(batch_size: int, hidden: int, device: Any = None) -> torch.Tensor:
    state = torch.zeros(int(batch_size), int(hidden), 5, device=device, dtype=torch.float32)
    state[:, :, 3] = -1e30
    return state


def normalize_recurrent_state_for_model(
    state: Any,
    model: Any,
    module_name: str,
    *,
    device: Any = None,
    batch_size: Optional[int] = None,
    allow_legacy_migration: bool = False,
) -> Optional[torch.Tensor]:
    """Convert a loaded recurrent state tensor to the active model layout.

    Version-1 chat files saved legacy 5-slot RWKV state. RWKV v8 uses
    [B, C, 3 + head_size], so this helper performs the same conservative
    migration the cell uses at runtime and returns a tensor ready to pass into
    inference.
    """
    if not torch.is_tensor(state):
        return None

    cell = getattr(model, module_name, None) if model is not None else None
    target_device = device if device is not None else state.device
    if not allow_legacy_migration:
        if cell is None:
            raise RuntimeError(
                f"Cannot verify strict recurrent state without model.{module_name}."
            )
        expected = tuple(
            _cell_exact_shape(
                cell,
                int(state.shape[0]) if state.dim() >= 1 else 0,
            )
        )
        if state.dim() != 3 or tuple(state.shape) != expected:
            raise RuntimeError(
                f"Chat state recurrent tensor shape mismatch for {module_name}: "
                f"saved={tuple(state.shape)}, current={expected}."
            )
        return state.to(device=target_device, dtype=torch.float32).detach()

    squeezed = state.squeeze(0) if state.dim() == 4 and state.shape[0] == 1 else state
    inferred_batch = int(batch_size or (squeezed.shape[0] if squeezed.dim() >= 1 else 1))

    if cell is not None and hasattr(cell, "_prepare_state") and hasattr(cell, "n_embd"):
        dummy = torch.zeros(
            inferred_batch,
            int(cell.n_embd),
            device=target_device,
            dtype=torch.float32,
        )
        old_migration_flag = getattr(cell, "allow_legacy_state_migration", None)
        if old_migration_flag is not None:
            cell.allow_legacy_state_migration = True
        try:
            prepared = cell._prepare_state(state, dummy)
        finally:
            if old_migration_flag is not None:
                cell.allow_legacy_state_migration = old_migration_flag
        return prepared.detach()

    if cell is None:
        return state.to(device=target_device, dtype=torch.float32).detach()

    hidden = int(getattr(cell, "n_embd", squeezed.shape[1] if squeezed.dim() >= 2 else 0))
    if squeezed.dim() == 3 and squeezed.shape == (inferred_batch, hidden, 5):
        return squeezed.to(device=target_device, dtype=torch.float32).detach()

    migrated = _legacy_initial_state(inferred_batch, hidden, device=target_device)
    source = squeezed.to(device=target_device, dtype=torch.float32)
    if source.dim() == 3 and hidden > 0:
        common_b = min(inferred_batch, source.shape[0])
        common_c = min(hidden, source.shape[1])
        if source.shape[-1] > 0:
            migrated[:common_b, :common_c, 0] = source[:common_b, :common_c, 0]
        if source.shape[-1] > 1:
            migrated[:common_b, :common_c, 4] = source[:common_b, :common_c, 1]
    return migrated.detach()
