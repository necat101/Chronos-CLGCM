#!/usr/bin/env python3
"""
hierarchos_bridge_server.py — JSON-RPC Bridge Server for the Hierarchos Rust GUI.

Reads JSON requests from stdin (one per line), writes JSON responses/events to stdout.
Hooks into the modular hierarchos package (not the deprecated monolith).
"""

import sys
import os
import json
import threading
import traceback
import time
import copy
from functools import wraps
from types import SimpleNamespace

# Ensure stdout is line-buffered for real-time streaming to the Rust GUI
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Globals ──────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_tokenizer_identity = None
_model_identity = None
_device = None
_config = {}          # AttrDict / dict of model config
_model_dir = None
_h_state = None
_l_state = None
_prev_context = None
_target_context = None
_drift_state = None
_ltm_state = None
_total_tokens_generated = 0
_pending_feedback = None
_ltm_token_clock = 0
_ltm_overlay_write_blocked_reason = None
_stop_generation = threading.Event()
_stop_training = threading.Event()
_cpu_threads = max(1, (os.cpu_count() or 2) // 2)
_operation_lock = threading.Lock()
_active_operation = None


class _TrainingCancelled(RuntimeError):
    """Internal cooperative-cancellation signal for GUI training."""


def emit(event_type: str, data: dict = None):
    """Send a JSON event to stdout (consumed by the Rust GUI)."""
    msg = {"event": event_type}
    if data:
        msg.update(data)
    try:
        print(json.dumps(msg, default=str), flush=True)
    except Exception:
        pass


def emit_error(msg: str, operation: str = None):
    payload = {"message": msg}
    if operation:
        payload["operation"] = operation
    emit("error", payload)


def emit_status(msg: str):
    emit("status", {"message": msg})


def emit_load_progress(progress: float, label: str):
    """Publish approximate load progress for the GUI progress bar."""
    try:
        progress = max(0.0, min(1.0, float(progress)))
    except Exception:
        progress = 0.0
    emit("load_progress", {"progress": progress, "label": label})


def _try_begin_operation(operation: str):
    """Claim the bridge's single mutable model/runtime lane."""
    global _active_operation
    with _operation_lock:
        if _active_operation is not None:
            return False, _active_operation
        _active_operation = str(operation)
        return True, None


def _finish_operation(operation: str) -> None:
    global _active_operation
    with _operation_lock:
        if _active_operation == operation:
            _active_operation = None


def _current_operation():
    with _operation_lock:
        return _active_operation


def _reject_busy(operation: str, active: str, *, terminal_event: str = None):
    message = (
        f"Cannot start {operation} while {active} is active. "
        "Stop or finish the current operation first."
    )
    emit_error(message, operation=operation)
    if terminal_event:
        emit(terminal_event, {"status": "rejected"})


def _exclusive_operation(operation: str):
    """Serialize a short model/runtime handler with long-running workers."""
    def decorate(handler):
        @wraps(handler)
        def wrapped(params):
            claimed, active = _try_begin_operation(operation)
            if not claimed:
                _reject_busy(operation, active)
                return None
            try:
                return handler(params)
            finally:
                _finish_operation(operation)

        wrapped._bridge_exclusive_operation = operation
        return wrapped

    return decorate


def _emit_backend_runtime_info():
    """Publish the PyTorch/CUDA runtime that was bundled into this backend."""
    try:
        from hierarchos.utils.device import cuda_diagnostics

        diag = cuda_diagnostics()
        emit("backend_info", diag)
        emit_load_progress(0.16, "Inspecting PyTorch runtime")

        torch_version = diag.get("torch_version", "unknown")
        cuda_version = diag.get("cuda_version") or "none"
        if diag.get("cuda_available"):
            total_mb = diag.get("total_memory_mb") or 0
            total_gb = total_mb / 1024.0 if total_mb else 0.0
            emit_status(
                f"PyTorch {torch_version} CUDA {cuda_version} ready: "
                f"{diag.get('device_name')} ({total_gb:.1f} GB VRAM). CPU fallback is bundled."
            )
        elif diag.get("cuda_built"):
            reason = diag.get("driver_error") or "no NVIDIA CUDA device visible"
            emit_status(
                f"PyTorch {torch_version} includes CUDA {cuda_version}; "
                f"CUDA is inactive on this machine ({reason}). CPU mode is available."
            )
        else:
            emit_status(
                f"PyTorch {torch_version} is CPU-only; CUDA selection will require a CUDA backend build."
            )
    except Exception as exc:
        emit_status(f"Could not inspect PyTorch runtime: {exc}")


def _release_loaded_model():
    """Release the current model before a replacement load to reduce VRAM spikes."""
    global _model, _tokenizer, _tokenizer_identity, _model_identity
    global _config, _model_dir
    global _h_state, _l_state, _prev_context, _target_context, _drift_state, _ltm_state
    global _ltm_token_clock, _ltm_overlay_write_blocked_reason

    old_device = _device
    _model = None
    _tokenizer = None
    _tokenizer_identity = None
    _model_identity = None
    _config = {}
    _model_dir = None
    _reset_runtime_state()
    _ltm_token_clock = 0
    _ltm_overlay_write_blocked_reason = None

    try:
        import torch

        if getattr(old_device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    emit("model_unloaded", {})


def _normalize_device(device_str: str):
    requested = (device_str or "auto").strip().lower()
    aliases = {
        "auto": None,
        "automatic": None,
        "cuda": "cuda",
        "gpu": "cuda",
        "cpu": "cpu",
        "dml": "dml",
        "directml": "dml",
    }
    return aliases.get(requested, requested)


def _looks_like_hf_repo_id(value: str) -> bool:
    if not value or os.path.exists(value):
        return False
    if value.lower().endswith(".pt"):
        return False
    if os.path.isabs(value):
        return False
    return all(c.isalnum() or c in "-_./" for c in value)


def _resolve_model_source(model_ref: str, cache_dir: str = None) -> str:
    """Return a local folder/file for a local path or Hugging Face repo id."""
    model_ref = (model_ref or "").strip().strip('"')
    if not model_ref:
        raise FileNotFoundError("No model source provided.")

    expanded = os.path.abspath(os.path.expanduser(model_ref))
    if os.path.exists(expanded):
        return expanded

    if not _looks_like_hf_repo_id(model_ref):
        raise FileNotFoundError(f"Model path not found: {model_ref}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Loading a Hugging Face repo id requires huggingface_hub "
            "(installed with transformers)."
        ) from exc

    safe_name = model_ref.replace("/", "_")
    local_dir = None
    if cache_dir:
        local_dir = os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), safe_name)
        os.makedirs(local_dir, exist_ok=True)

    emit_load_progress(0.28, "Downloading model snapshot")
    emit_status(f"Downloading Hugging Face model: {model_ref}")
    kwargs = {
        "repo_id": model_ref,
        "allow_patterns": [
            "*.pt",
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    return snapshot_download(**kwargs)


def _has_tokenizer_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "tokenizer.model",
    }
    try:
        return any(name.lower() in names or name.lower().startswith("tokenizer.") for name in os.listdir(path))
    except OSError:
        return False


def _ltm_updates_path() -> str:
    if not _model_dir:
        raise RuntimeError("No model directory is loaded.")
    return os.path.join(_model_dir, "hierarchos_ltm_updates.pt")


def _checkpoint_file_identity(model) -> dict:
    """Build a cheap identity for state/overlay compatibility checks.

    Safe checkpoints already have a SHA-256 sidecar, so use that without
    re-hashing multi-gigabyte weights. Legacy files fall back to run metadata
    plus stable file attributes. This is a compatibility guard, not an
    authentication primitive.
    """
    metadata = getattr(model, "_hierarchos_checkpoint_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    identity = {
        "version": 1,
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "completed_epoch": metadata.get("completed_epoch"),
    }
    run_identity = metadata.get("run_identity")
    if isinstance(run_identity, dict):
        identity["run_identity_sha256"] = run_identity.get("sha256")

    source = metadata.get("source_weights_path")
    if isinstance(source, str) and source:
        source = os.path.abspath(source)
        checksum_path = source + ".sha256"
        if os.path.isfile(checksum_path):
            try:
                with open(checksum_path, "r", encoding="utf-8") as checksum_file:
                    digest = checksum_file.read().strip().split()[0].lower()
                if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                    identity["checkpoint_sha256"] = digest
            except OSError:
                pass
        if "checkpoint_sha256" not in identity:
            try:
                stat = os.stat(source)
                identity["source_basename"] = os.path.basename(source)
                identity["source_size"] = int(stat.st_size)
                identity["source_mtime_ns"] = int(stat.st_mtime_ns)
            except OSError:
                pass

    return {
        key: value
        for key, value in identity.items()
        if value is not None
    }


def _bridge_runtime_identity() -> dict:
    """Bind GUI continuation state to the loaded weights and token language."""
    identity = {
        "version": 2,
        "model": copy.deepcopy(_model_identity) if isinstance(_model_identity, dict) else {},
    }
    if isinstance(_tokenizer_identity, dict):
        identity["tokenizer_sha256"] = _tokenizer_identity.get("sha256")
        identity["tokenizer_vocab_size"] = _tokenizer_identity.get("vocab_size")
        identity["tokenizer_behavior_sha256_v2"] = _tokenizer_identity.get(
            "behavior_sha256_v2"
        )
    return identity


def _validate_bridge_runtime_identity(payload: dict) -> None:
    saved = payload.get("bridge_runtime_identity")
    if saved is None:
        # CLI and legacy GUI state files predate this bridge-only binding. The
        # shared architecture/state validator still checks their tensor layout.
        # Legacy bridge files do carry model_dir, so use it to prevent the GUI
        # from auto-loading one model's hidden state into another model.
        saved_model_dir = payload.get("model_dir")
        if isinstance(saved_model_dir, str) and saved_model_dir and _model_dir:
            if os.path.normcase(os.path.abspath(saved_model_dir)) != os.path.normcase(
                os.path.abspath(_model_dir)
            ):
                raise RuntimeError(
                    "Legacy GUI chat state belongs to a different model directory. "
                    "Reset this chat state before continuing."
                )
        return
    if not isinstance(saved, dict):
        raise RuntimeError("Chat state has malformed bridge runtime identity metadata.")
    current = _bridge_runtime_identity()
    saved_version = int(saved.get("version", 1) or 1)
    if saved_version == 1:
        # Version 1 authenticated weights and the vocabulary mapping but
        # predates behavior-level tokenizer fingerprints. Compare exactly the
        # fields it actually promised so existing GUI continuations remain usable.
        current_v1 = {
            "version": 1,
            "model": current.get("model", {}),
        }
        if isinstance(_tokenizer_identity, dict):
            current_v1["tokenizer_sha256"] = current.get("tokenizer_sha256")
            current_v1["tokenizer_vocab_size"] = current.get(
                "tokenizer_vocab_size"
            )
        current = current_v1
    elif saved_version != 2:
        raise RuntimeError(
            f"Unsupported bridge runtime identity version {saved_version}."
        )
    if saved != current:
        raise RuntimeError(
            "Chat state belongs to different model weights or tokenizer. "
            "Reset this GUI chat state before continuing with the loaded model."
        )


def _normalize_ltm_tensor(value):
    import torch
    if value is None or not torch.is_tensor(value):
        return None
    tensor = value.detach()
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.cpu()


def _copy_ltm_tensor(module, attr: str, value) -> bool:
    import torch
    tensor = _normalize_ltm_tensor(value)
    if tensor is None or not hasattr(module, attr):
        return False
    target = getattr(module, attr)
    if not torch.is_tensor(target) or tuple(target.shape) != tuple(tensor.shape):
        return False
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"Refusing to copy non-finite runtime LTM tensor {attr}.")
    with torch.no_grad():
        converted = tensor.to(device=target.device, dtype=target.dtype)
        if converted.is_floating_point() and not bool(torch.isfinite(converted).all().item()):
            raise ValueError(f"Runtime LTM tensor {attr} overflows the model dtype.")
        target.copy_(converted)
    return True


def _sync_runtime_ltm_to_module():
    """Copy runtime LTM state back into the model LTM buffers before saving."""
    if _model is None or not hasattr(_model, "ltm") or _ltm_state is None:
        return

    ltm = _model.ltm
    if len(_ltm_state) >= 1:
        _copy_ltm_tensor(ltm, "fast_vals", _ltm_state[0])
    if len(_ltm_state) >= 2:
        _copy_ltm_tensor(ltm, "_mom_vals", _ltm_state[1])
    if len(_ltm_state) >= 5:
        _copy_ltm_tensor(ltm, "timestamps", _ltm_state[4])
    if len(_ltm_state) >= 6:
        _copy_ltm_tensor(ltm, "sources", _ltm_state[5])
    if len(_ltm_state) >= 7:
        _copy_ltm_tensor(ltm, "wallclock_timestamps", _ltm_state[6])


def _refresh_ltm_token_clock() -> int:
    """Resume the monotonic online-memory clock from validated slot metadata."""
    global _ltm_token_clock
    try:
        import torch

        timestamps = getattr(getattr(_model, "ltm", None), "timestamps", None)
        if torch.is_tensor(timestamps) and timestamps.numel():
            finite = timestamps.detach().float()
            finite = finite[torch.isfinite(finite)]
            if finite.numel():
                _ltm_token_clock = max(0, int(finite.max().item()))
                return _ltm_token_clock
    except Exception:
        pass
    _ltm_token_clock = 0
    return _ltm_token_clock


def _apply_saved_ltm_updates():
    """Load durable LTM sidecar data, if present, without restoring working memory."""
    global _ltm_state, _ltm_overlay_write_blocked_reason
    if _model is None or not hasattr(_model, "ltm") or not _model_dir:
        return

    ltm = _model.ltm
    if hasattr(ltm, "accumulate_deltas"):
        ltm.accumulate_deltas = True

    path = _ltm_updates_path()
    if not os.path.exists(path):
        _ltm_overlay_write_blocked_reason = None
        _refresh_ltm_token_clock()
        return

    import torch
    from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

    payload = load_checkpoint_payload_compatible(path, map_location="cpu")
    target_vals = getattr(ltm, "vals", None)
    if not torch.is_tensor(target_vals):
        raise ValueError("Loaded model does not expose persistent LTM values.")

    raw_version = payload.get("version", 0) if isinstance(payload, dict) else 1
    if isinstance(raw_version, bool):
        raise ValueError("LTM sidecar version cannot be boolean.")
    try:
        payload_version = int(raw_version or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LTM sidecar has an invalid version.") from exc

    if payload_version == 3:
        from hierarchos.inference.chat import load_ltm_delta_overlay

        if payload.get("bridge_runtime_identity") is not None:
            _validate_bridge_runtime_identity(payload)
        loaded_version = load_ltm_delta_overlay(
            _model,
            path,
            tokenizer=_tokenizer,
        )
        if loaded_version != 3:
            raise RuntimeError(
                f"Expected LTM overlay v3 but shared loader returned v{loaded_version}."
            )
        _ltm_state = None
        _ltm_overlay_write_blocked_reason = None
        _refresh_ltm_token_clock()
        emit_status(f"Loaded identity-bound LTM overlay v3 from {path}.")
        return

    metadata = {}
    legacy_bridge_payload = False
    if torch.is_tensor(payload):
        delta = payload
        version = 1
    elif isinstance(payload, dict):
        version = int(payload.get("version", 0) or 0)
        if version == 2 and "delta" in payload:
            delta = payload.get("delta")
            metadata = payload
            saved_base = payload.get("base_model_identity")
            if saved_base is not None and saved_base != _model_identity:
                raise ValueError(
                    "Saved LTM updates belong to different base model weights."
                )
        elif version == 1 and "ltm_deltas" in payload:
            # Compatibility with bridge v1. Its fast/momentum/timestamp fields
            # were transient snapshots and are intentionally not restored.
            delta = payload.get("ltm_deltas")
            legacy_bridge_payload = True
        else:
            raise ValueError(f"Unsupported LTM sidecar version {version}.")
    else:
        raise ValueError("LTM sidecar must be a tensor or versioned dictionary.")

    if not torch.is_tensor(delta) or not delta.is_floating_point():
        raise ValueError("LTM sidecar delta must be a floating-point tensor.")
    if tuple(delta.shape) != tuple(target_vals.shape):
        raise ValueError(
            f"LTM sidecar delta shape {tuple(delta.shape)} does not match "
            f"model shape {tuple(target_vals.shape)}."
        )
    if not bool(torch.isfinite(delta).all().item()):
        raise ValueError("LTM sidecar delta contains non-finite values.")

    converted_delta = delta.to(device=target_vals.device, dtype=target_vals.dtype)
    if not bool(torch.isfinite(converted_delta).all().item()):
        raise ValueError("LTM sidecar delta overflows the model value dtype.")
    candidate_vals = target_vals.detach() + converted_delta
    if not bool(torch.isfinite(candidate_vals).all().item()):
        raise ValueError("LTM sidecar would make persistent memory non-finite.")

    validated_metadata = {}
    for key, attr in (
        ("timestamps", "timestamps"),
        ("sources", "sources"),
        ("wallclock_timestamps", "wallclock_timestamps"),
    ):
        value = metadata.get(key)
        target = getattr(ltm, attr, None)
        if value is None:
            continue
        if not torch.is_tensor(value) or not torch.is_tensor(target):
            raise ValueError(f"LTM sidecar {key} has no compatible model target.")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"LTM sidecar {key} shape {tuple(value.shape)} does not match "
                f"model shape {tuple(target.shape)}."
            )
        if key == "sources":
            if value.dtype != torch.int64:
                raise ValueError("LTM sidecar sources must use torch.int64.")
            allowed_sources = {
                int(getattr(ltm, "SRC_UNKNOWN", 0)),
                int(getattr(ltm, "SRC_USER_INTERACTION", 1)),
                int(getattr(ltm, "SRC_TRAINING_DATA", 2)),
                int(getattr(ltm, "SRC_CORRECTION", 3)),
            }
            if value.numel() and not set(int(item) for item in value.unique().tolist()) <= allowed_sources:
                raise ValueError("LTM sidecar contains an unknown source identifier.")
        else:
            if not value.is_floating_point():
                raise ValueError(f"LTM sidecar {key} must be floating point.")
            if not bool((torch.isfinite(value) & (value >= 0)).all().item()):
                raise ValueError(f"LTM sidecar {key} contains invalid values.")
        validated_metadata[attr] = value

    with torch.no_grad():
        target_vals.copy_(candidate_vals)
        accumulator = getattr(ltm, "ltm_deltas", None)
        if torch.is_tensor(accumulator):
            accumulator.copy_(
                converted_delta.to(device=accumulator.device, dtype=accumulator.dtype)
            )
        for attr, value in validated_metadata.items():
            target = getattr(ltm, attr)
            target.copy_(value.to(device=target.device, dtype=target.dtype))

    _ltm_state = None
    _ltm_overlay_write_blocked_reason = None
    _refresh_ltm_token_clock()
    emit_status(f"Loaded saved LTM updates from {path}.")
    if legacy_bridge_payload:
        emit_status(
            "Migrated legacy bridge LTM deltas; discarded its transient "
            "fast/momentum metadata."
        )


def _reset_runtime_state():
    global _h_state, _l_state, _prev_context, _target_context
    global _drift_state, _ltm_state, _total_tokens_generated, _pending_feedback
    _h_state = None
    _l_state = None
    _prev_context = None
    _target_context = None
    _drift_state = None
    _ltm_state = None
    _total_tokens_generated = 0
    _pending_feedback = None


def _chat_state_config_signature():
    """Small architecture fingerprint for model-neutral chat state files."""
    if not _config:
        return {}
    from hierarchos.inference.chat_state import chat_state_config_signature

    return chat_state_config_signature(_config, _model)


def _tensor_to_cpu(value):
    from hierarchos.inference.chat_state import tensor_to_cpu

    return tensor_to_cpu(value)


def _rosa_past_tokens_from_ltm_state(ltm_state):
    if ltm_state is None or not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 3:
        return None
    return _tensor_to_cpu(ltm_state[2])


def _rosa_states_from_ltm_state(ltm_state):
    if ltm_state is None or not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 4:
        return None
    rosa_states = ltm_state[3]
    if rosa_states is None:
        return None
    try:
        return copy.deepcopy(rosa_states)
    except Exception:
        return rosa_states


def _ltm_state_from_rosa_context(rosa_past_tokens, rosa_states=None):
    """Resume ROSA context without putting per-chat LTM values in state files."""
    if _model is None or not hasattr(_model, "ltm"):
        return None
    try:
        import torch

        if not torch.is_tensor(rosa_past_tokens):
            return None
        ltm = _model.ltm
        fast_vals = getattr(ltm, "fast_vals", None)
        mom_vals = getattr(ltm, "_mom_vals", None)
        if not torch.is_tensor(fast_vals) or not torch.is_tensor(mom_vals):
            return None
        return (
            fast_vals,
            torch.zeros_like(mom_vals),
            rosa_past_tokens.detach().cpu().clone(),
            copy.deepcopy(rosa_states) if rosa_states is not None else None,
            getattr(ltm, "timestamps", None),
            getattr(ltm, "sources", None),
            getattr(ltm, "wallclock_timestamps", None),
        )
    except Exception:
        return None


def _ltm_state_from_rosa_past(rosa_past_tokens):
    return _ltm_state_from_rosa_context(rosa_past_tokens, None)


def _normalize_chat_state_path(path: str) -> str:
    raw_path = os.fspath(path) if path is not None else ""
    raw_path = raw_path.strip().strip('"')
    if not raw_path:
        raise ValueError("No chat runtime state path provided.")
    return os.path.abspath(os.path.expanduser(raw_path))


def _write_chat_runtime_state(path: str):
    """Persist only tiny hierarchical continuation state, never LTM/model weights."""
    path = _normalize_chat_state_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    import torch
    from hierarchos.inference.chat_state import (
        CHAT_STATE_KIND,
        CHAT_STATE_VERSION,
        chat_state_architecture_metadata,
        recurrent_state_metadata,
        validate_chat_state_payload_compatible,
    )

    payload = {
        "version": CHAT_STATE_VERSION,
        "kind": CHAT_STATE_KIND,
        "saved_at": time.time(),
        "model_dir": _model_dir,
        "config_signature": _chat_state_config_signature(),
        "bridge_runtime_identity": _bridge_runtime_identity(),
        "total_tokens_generated": int(_total_tokens_generated),
        "h_state": _tensor_to_cpu(_h_state),
        "l_state": _tensor_to_cpu(_l_state),
        "prev_context": _tensor_to_cpu(_prev_context),
        "target_context": _tensor_to_cpu(_target_context),
        "drift_state": _tensor_to_cpu(_drift_state),
        "rosa_past_tokens": _rosa_past_tokens_from_ltm_state(_ltm_state),
        "rosa_states": _rosa_states_from_ltm_state(_ltm_state),
    }
    payload.update(chat_state_architecture_metadata(_config))
    payload.update(
        recurrent_state_metadata(
            model=_model,
            config=_config,
            h_state=_h_state,
            l_state=_l_state,
        )
    )
    # Fail before replacing the prior autosave if any runtime tensor has
    # become non-finite, malformed, or inconsistent with the loaded model.
    validate_chat_state_payload_compatible(payload, _config, model=_model)

    temp_path = path + ".tmp"
    try:
        torch.save(payload, temp_path)
        with open(temp_path, "r+b") as state_file:
            os.fsync(state_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise
    return path


def _validate_chat_state_signature(payload: dict):
    from hierarchos.inference.chat_state import validate_chat_state_payload_compatible

    return validate_chat_state_payload_compatible(payload, _config, model=_model)


def _load_chat_runtime_state(path: str):
    """Restore hierarchical continuation state without restoring LTM working memory."""
    global _h_state, _l_state, _prev_context, _target_context
    global _drift_state, _ltm_state, _total_tokens_generated, _pending_feedback

    path = _normalize_chat_state_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chat runtime state file not found: {path}")

    import torch
    from hierarchos.inference.chat_state import (
        CHAT_STATE_KIND,
        CHAT_STATE_VERSION,
        chat_state_architecture_metadata,
        normalize_recurrent_state_for_model,
        recurrent_state_metadata,
        validate_chat_state_payload_compatible,
    )
    from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

    payload = load_checkpoint_payload_compatible(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("kind") != CHAT_STATE_KIND:
        raise RuntimeError(f"Not a Hierarchos chat runtime state file: {path}")

    allow_legacy_migration = _validate_chat_state_signature(payload)
    _validate_bridge_runtime_identity(payload)

    target_device = _device if _device is not None else "cpu"

    def _device_tensor(name):
        value = payload.get(name)
        if not torch.is_tensor(value):
            return None
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Chat state {name} contains non-finite values.")
        return value.to(target_device).detach()

    h_state = normalize_recurrent_state_for_model(
        payload.get("h_state"),
        _model,
        "h_rnn",
        device=target_device,
        allow_legacy_migration=allow_legacy_migration,
    )
    l_state = normalize_recurrent_state_for_model(
        payload.get("l_state"),
        _model,
        "l_rnn",
        device=target_device,
        allow_legacy_migration=allow_legacy_migration,
    )
    prev_context = _device_tensor("prev_context")
    target_context = _device_tensor("target_context")
    drift_state = _device_tensor("drift_state")
    rosa_past_tokens = payload.get("rosa_past_tokens")
    if torch.is_tensor(rosa_past_tokens):
        rosa_past_tokens = rosa_past_tokens.detach().cpu()
    elif rosa_past_tokens is not None:
        raise RuntimeError("Chat state rosa_past_tokens must be a tensor.")
    rosa_states = payload.get("rosa_states")
    if allow_legacy_migration:
        # Rebuild legacy automata from the validated token history rather than
        # trusting stale derived transition tables.
        rosa_states = None
    elif rosa_states is not None:
        rosa_states = copy.deepcopy(rosa_states)

    total_tokens = payload.get("total_tokens_generated", 0)
    if isinstance(total_tokens, bool):
        raise RuntimeError("Chat state total_tokens_generated cannot be boolean.")
    try:
        total_tokens = int(total_tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Chat state total_tokens_generated must be a nonnegative integer."
        ) from exc
    if total_tokens < 0:
        raise RuntimeError(
            "Chat state total_tokens_generated must be a nonnegative integer."
        )

    if allow_legacy_migration:
        # Re-validate the migrated tensors as a complete current-format state.
        migrated = {
            "version": CHAT_STATE_VERSION,
            "kind": CHAT_STATE_KIND,
            "config_signature": _chat_state_config_signature(),
            "total_tokens_generated": total_tokens,
            "h_state": h_state,
            "l_state": l_state,
            "prev_context": prev_context,
            "target_context": target_context,
            "drift_state": drift_state,
            "rosa_past_tokens": rosa_past_tokens,
            "rosa_states": None,
        }
        migrated.update(chat_state_architecture_metadata(_config))
        migrated.update(
            recurrent_state_metadata(
                model=_model,
                config=_config,
                h_state=h_state,
                l_state=l_state,
            )
        )
        validate_chat_state_payload_compatible(migrated, _config, model=_model)

    # Assign only after the complete payload has passed validation so a bad
    # autosave cannot partially replace the active chat state.
    _h_state = h_state
    _l_state = l_state
    _prev_context = prev_context
    _target_context = target_context
    _drift_state = drift_state
    _total_tokens_generated = total_tokens

    # LTM fast/momentum state is deliberately not per-chat. The model/LTM sidecar
    # owns persistent memory; chat files keep full ROSA token history/state.
    _ltm_state = _ltm_state_from_rosa_context(
        rosa_past_tokens,
        rosa_states,
    )
    _pending_feedback = None
    return path


def _apply_thread_count(value=None) -> int:
    """Clamp and apply the PyTorch CPU thread count used by chat/inference."""
    global _cpu_threads
    if value is None:
        value = _cpu_threads
    try:
        threads = int(value)
    except Exception:
        threads = _cpu_threads

    max_threads = max(1, os.cpu_count() or 1)
    threads = max(1, min(threads, max_threads))

    try:
        from hierarchos import set_threads

        set_threads(threads)
    finally:
        _cpu_threads = threads
    return threads


def _iter_training_source_objects(path: str):
    """Yield JSON training objects without probing JSONL via whole-file loads."""
    is_jsonl = str(path).lower().endswith((".jsonl", ".ndjson"))
    with open(path, "r", encoding="utf-8") as source:
        if is_jsonl:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            return

        try:
            data = json.load(source)
        except json.JSONDecodeError:
            # Retain compatibility with JSON-lines content using a .json
            # suffix, but only after the declared JSON document failed.
            source.seek(0)
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            return

        if not isinstance(data, list):
            data = [data]
        yield from data


def _exact_dataloader_batches(dataloader, batch_size: int):
    """Return an exact step count, streaming iterable datasets at O(1) RAM."""
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        batches = int(len(dataloader))
        if batches <= 0:
            raise ValueError("Training dataloader contains no usable batches.")
        return batches, None

    try:
        sample_count = int(len(dataset))
    except (TypeError, AttributeError, NotImplementedError):
        # StreamingJSONLDataset intentionally has no approximate __len__. Scan
        # its own iterator once so malformed/dropped rows cannot corrupt the LR
        # schedule, checkpoint step identity, or GUI progress. Disable its
        # length bucket on the shallow copy to keep the count O(1) in memory.
        import torch

        num_workers = max(0, int(getattr(dataloader, "num_workers", 0) or 0))
        logical_workers = max(1, num_workers)
        worker_sample_counts = []
        original_get_worker_info = torch.utils.data.get_worker_info
        try:
            for worker_id in range(logical_workers):
                counting_dataset = copy.copy(dataset)
                if hasattr(counting_dataset, "bucket_size"):
                    counting_dataset.bucket_size = 0
                if hasattr(counting_dataset, "shuffle_buckets"):
                    counting_dataset.shuffle_buckets = False
                worker_info = (
                    None
                    if num_workers == 0
                    else SimpleNamespace(id=worker_id, num_workers=num_workers)
                )
                torch.utils.data.get_worker_info = lambda info=worker_info: info
                worker_sample_counts.append(
                    sum(
                        1
                        for sample in counting_dataset
                        if sample is not None
                        and not (
                            isinstance(sample, dict)
                            and sample.get("_audit_only", False)
                        )
                    )
                )
        finally:
            torch.utils.data.get_worker_info = original_get_worker_info

        sample_count = sum(worker_sample_counts)
        if sample_count <= 0:
            raise ValueError("Training dataset contains no usable samples.")
        batch_size = max(1, int(batch_size))
        # IterableDataset auto-batching happens inside each worker. Every
        # worker can therefore contribute its own partial final batch.
        batches = sum(
            (count + batch_size - 1) // batch_size
            for count in worker_sample_counts
            if count > 0
        )
        return batches, sample_count

    if sample_count <= 0:
        raise ValueError("Training dataset contains no usable samples.")
    try:
        batches = int(len(dataloader))
    except (TypeError, AttributeError, NotImplementedError):
        batch_size = max(1, int(batch_size))
        batches = (sample_count + batch_size - 1) // batch_size
    if batches <= 0:
        raise ValueError("Training dataloader contains no usable batches.")
    return batches, sample_count


def _sample_next_token(logits, generated_ids, sampling, tokenizer=None):
    import torch
    from hierarchos.inference.chat import (
        sample_next_token,
        should_stop_generation_from_uncertainty,
    )

    if logits is None or not torch.is_tensor(logits):
        raise RuntimeError("Model did not return tensor logits.")

    if should_stop_generation_from_uncertainty(logits, generated_ids, tokenizer, sampling):
        return None

    next_logits = logits[:, -1, :] if logits.dim() == 3 else logits
    return sample_next_token(
        next_logits,
        temperature=sampling.get("temperature", 0.7),
        top_k=sampling.get("top_k", 40),
        top_p=sampling.get("top_p", 0.9),
        repetition_penalty=sampling.get("repetition_penalty", 1.2),
        previous_tokens=generated_ids,
        # Every bridge call consumes logits returned directly by
        # HierarchosCore, which already rejected non-finite output. Avoid a
        # second vocabulary scan and device synchronization for every token.
        _logits_prevalidated=True,
    )


def _validated_online_learning_rate(
    value,
    *,
    default: float,
    maximum: float,
    label: str,
) -> float:
    """Validate an online-memory rate and cap it at the bridge safety limit."""
    import math

    raw_value = default if value is None else value
    if isinstance(raw_value, bool):
        raise ValueError(f"{label} must be a finite positive number.")
    try:
        rate = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite positive number.") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{label} must be a finite positive number.")
    return min(rate, float(maximum))


def _apply_bridge_online_ltm_transaction(
    input_ids,
    label_ids,
    *,
    source_id: int,
    penalty: bool,
    learning_rate: float,
    learn_input_tokens: bool = False,
):
    """Commit and persist one bounded online-memory transaction."""
    global _ltm_state, _ltm_token_clock

    import torch
    from hierarchos.inference.chat import (
        apply_online_feedback_transaction,
        save_ltm_delta_overlay_atomic,
    )

    ltm = getattr(_model, "ltm", None)
    if ltm is None:
        raise ValueError("Loaded model does not expose online LTM memory.")
    if hasattr(ltm, "accumulate_deltas"):
        ltm.accumulate_deltas = True
    if _ltm_overlay_write_blocked_reason:
        return {
            "committed": False,
            "reason": "persistence-blocked",
            "ltm_state": _ltm_state,
            "token_clock": int(_ltm_token_clock),
            "persistence_error": (
                "sidecar overwrite is blocked because the existing overlay "
                f"failed validation: {_ltm_overlay_write_blocked_reason}"
            ),
        }

    # The shared transaction commits to module buffers once its objective gate
    # accepts. Snapshot every mutated durable/runtime tensor so the GUI can
    # extend that transaction boundary through the atomic sidecar replace.
    rollback_tensors = {}
    for attr in (
        "fast_vals",
        "_mom_vals",
        "timestamps",
        "sources",
        "wallclock_timestamps",
        "ltm_deltas",
    ):
        value = getattr(ltm, attr, None)
        if torch.is_tensor(value):
            rollback_tensors[attr] = value.detach().clone()
    prior_ltm_state = _ltm_state
    prior_token_clock = int(_ltm_token_clock)

    def _rollback_module_state():
        global _ltm_state, _ltm_token_clock
        with torch.no_grad():
            for attr, snapshot in rollback_tensors.items():
                target = getattr(ltm, attr, None)
                if torch.is_tensor(target) and tuple(target.shape) == tuple(snapshot.shape):
                    target.copy_(snapshot.to(device=target.device, dtype=target.dtype))
        _ltm_state = prior_ltm_state
        _ltm_token_clock = prior_token_clock

    model_config = getattr(_model, "config", _config)
    try:
        result = apply_online_feedback_transaction(
            _model,
            input_ids,
            label_ids,
            config=model_config,
            ltm_state=_ltm_state,
            source_id=int(source_id),
            penalty=bool(penalty),
            learning_rate=float(learning_rate),
            grad_clip=float(_config.get("online_ltm_grad_clip", 0.75)),
            max_delta_norm=float(
                _config.get("online_ltm_max_delta_norm", 1.0)
            ),
            max_fast_norm=float(
                _config.get("online_ltm_max_fast_norm", 64.0)
            ),
            max_slot_norm=float(
                _config.get("online_ltm_max_slot_norm", 4.0)
            ),
            token_clock=int(_ltm_token_clock),
            learn_input_tokens=bool(learn_input_tokens),
        )
    except Exception:
        _rollback_module_state()
        raise
    result = dict(result or {})
    if not bool(result.get("committed", False)):
        return result

    accepted_state = result.get("ltm_state")
    if not isinstance(accepted_state, (tuple, list)):
        raise RuntimeError("Accepted feedback did not return an LTM state.")
    accepted_state = tuple(accepted_state)
    try:
        persisted_path = save_ltm_delta_overlay_atomic(
            _model,
            _ltm_updates_path(),
            accepted_state,
            tokenizer=_tokenizer,
            extra_metadata={
                "total_tokens_generated": int(_total_tokens_generated),
                "base_model_identity": copy.deepcopy(_model_identity),
                "bridge_runtime_identity": _bridge_runtime_identity(),
            },
        )
    except Exception as exc:
        _rollback_module_state()
        result.update(
            {
                "committed": False,
                "reason": "persistence-failed",
                "ltm_state": prior_ltm_state,
                "token_clock": prior_token_clock,
                "persisted": False,
                "path": None,
                "persistence_error": str(exc),
                "rolled_back": True,
            }
        )
        return result

    _ltm_state = accepted_state
    _ltm_token_clock = int(result.get("token_clock", _ltm_token_clock))

    result.update(
        {
            "persisted": True,
            "path": persisted_path,
            "persistence_error": None,
        }
    )
    return result


# ── Handlers ─────────────────────────────────────────────────────────────────

def handle_load_model(params: dict):
    """Load a Hierarchos model from a local folder/file or Hugging Face repo."""
    global _model, _tokenizer, _tokenizer_identity, _model_identity
    global _device, _config, _model_dir, _cpu_threads
    global _ltm_token_clock, _ltm_overlay_write_blocked_reason
    from transformers import AutoTokenizer
    import argparse

    from hierarchos import (
        configure_torch_runtime,
        cuda_diagnostics,
        describe_device,
        pick_device,
        load_full_model_with_config,
        set_threads,
    )

    model_ref = params.get("model_path", "")
    device_str = params.get("device", "auto")
    tokenizer_ref = (params.get("tokenizer_path") or "").strip()
    cache_dir = params.get("cache_dir")

    claimed, active = _try_begin_operation("model_loading")
    if not claimed:
        _reject_busy("model loading", active)
        return

    emit_load_progress(0.22, "Resolving model source")
    emit_status(f"Resolving model source: {model_ref}")

    replacement_started = False
    try:
        resolved_model_path = _resolve_model_source(model_ref, cache_dir)
        emit_load_progress(0.32, "Preparing model runtime")

        if _model is not None:
            emit_status("Unloading current model before replacement load.")
            emit_load_progress(0.34, "Unloading previous model")
            _release_loaded_model()
            replacement_started = True

        _model_dir = resolved_model_path if os.path.isdir(resolved_model_path) else os.path.dirname(resolved_model_path)

        emit_load_progress(0.38, "Selecting compute device")
        requested_threads = params.get("cpu_threads", params.get("threads", _cpu_threads))
        ns = argparse.Namespace(
            device=_normalize_device(device_str),
            threads=_apply_thread_count(requested_threads),
        )
        _device = pick_device(ns)
        runtime_diag = configure_torch_runtime(_device)
        device_label = describe_device(_device)

        emit_load_progress(0.45, "Loading model weights")
        emit_status(f"Loading model on {device_label} from {resolved_model_path}")
        model, cfg = load_full_model_with_config(resolved_model_path, _device)
        emit_load_progress(0.72, "Weights loaded")
        from hierarchos.inference.chat_state import clear_ltm_working_memory

        clear_ltm_working_memory(model)
        _model = model
        _model.eval()
        _model.suppress_hebbian = True
        if hasattr(_model.ltm, "accumulate_deltas"):
            _model.ltm.accumulate_deltas = True
        _config = dict(cfg) if hasattr(cfg, 'items') else {}
        _model_identity = _checkpoint_file_identity(_model)
        _reset_runtime_state()
        _ltm_token_clock = 0
        _ltm_overlay_write_blocked_reason = None

        emit_load_progress(0.76, "Loading tokenizer")
        tokenizer_candidates = []
        if tokenizer_ref:
            tokenizer_candidates.append(tokenizer_ref)
        model_dir_has_tokenizer = _has_tokenizer_files(_model_dir)
        if _model_dir and model_dir_has_tokenizer:
            tokenizer_candidates.append(_model_dir)
        if _config.get("tokenizer_name"):
            tokenizer_candidates.append(_config.get("tokenizer_name"))
        if not tokenizer_candidates:
            tokenizer_candidates.append("openai-community/gpt2")

        trust_remote_code_value = params.get("trust_remote_code", False)
        if isinstance(trust_remote_code_value, str):
            trust_remote_code = trust_remote_code_value.strip().lower() in {
                "1", "true", "yes", "on",
            }
        else:
            trust_remote_code = trust_remote_code_value is True
        if trust_remote_code:
            emit_status(
                "WARNING: tokenizer trust_remote_code was explicitly enabled; "
                "the tokenizer repository may execute Python code."
            )

        from hierarchos.utils.tokenizer import (
            tokenizer_identity,
            validate_inference_tokenizer_identity,
        )

        last_tokenizer_error = None
        last_tokenizer_path = None
        for tok_path in tokenizer_candidates:
            try:
                candidate = AutoTokenizer.from_pretrained(
                    tok_path,
                    trust_remote_code=trust_remote_code,
                )
                if candidate.pad_token is None:
                    if candidate.eos_token:
                        candidate.pad_token = candidate.eos_token
                    else:
                        candidate.add_special_tokens({'pad_token': '[PAD]'})
                model_vocab = int(_config.get("vocab_size", len(candidate)))
                if len(candidate) != model_vocab:
                    raise ValueError(
                        f"tokenizer vocabulary {len(candidate)} does not match "
                        f"checkpoint vocabulary {model_vocab}"
                    )
                eos_id = candidate.eos_token_id
                if eos_id is None or not 0 <= int(eos_id) < model_vocab:
                    raise ValueError(
                        "tokenizer has no valid EOS token for the checkpoint vocabulary"
                    )
                identity_verified = validate_inference_tokenizer_identity(
                    candidate,
                    getattr(_model, "_hierarchos_checkpoint_metadata", {}),
                )
                _tokenizer = candidate
                _tokenizer_identity = tokenizer_identity(candidate)
                last_tokenizer_path = tok_path
                break
            except Exception as exc:
                last_tokenizer_error = exc
                _tokenizer = None

        if _tokenizer is None:
            raise RuntimeError(f"Failed to load tokenizer: {last_tokenizer_error}")
        emit_load_progress(0.90, "Tokenizer ready")
        emit_status(f"Tokenizer loaded from {last_tokenizer_path}.")
        if identity_verified:
            emit_status("Training tokenizer content fingerprint verified.")
        else:
            emit_status(
                "WARNING: Legacy checkpoint has no strong tokenizer fingerprint; "
                "only vocabulary compatibility could be checked."
            )

        emit_load_progress(0.92, "Checking saved LTM updates")
        try:
            _apply_saved_ltm_updates()
        except Exception as exc:
            clear_ltm_working_memory(_model)
            if hasattr(_model.ltm, "accumulate_deltas"):
                _model.ltm.accumulate_deltas = True
            _ltm_token_clock = 0
            _ltm_overlay_write_blocked_reason = str(exc)
            emit_status(
                "WARNING: Ignored incompatible or corrupt saved LTM updates and "
                "disabled sidecar overwrites for this load: "
                f"{exc}"
            )

        emit_load_progress(0.94, "Finalizing model")
        total_params = sum(p.numel() for p in _model.parameters())

        model_config = {
            "context_dim": int(_config.get("context_dim", 448)),
            "h_hidden": int(_config.get("h_hidden", _config.get("context_dim", 448))),
            "l_hidden": int(_config.get("l_hidden", _config.get("context_dim", 448))),
            "ltm_slots": int(_config.get("ltm_slots", 1024)),
            "ltm_key_dim": int(_config.get("ltm_key_dim", 128)),
            "ltm_val_dim": int(_config.get("ltm_val_dim", 128)),
            "ltm_topk": int(_config.get("ltm_topk", 4)),
            "vocab_size": int(_config.get("vocab_size", len(_tokenizer))),
            "max_length": int(_config.get("max_length", 1024)),
            "h_stride": int(_config.get("h_stride", 4)),
            "max_h_steps": int(_config.get("max_h_steps", 5)),
            "max_l_steps": int(_config.get("max_l_steps", 5)),
            "persistent_dim": int(_config.get("persistent_dim", 128)),
            "training_chunk_size": int(_config.get("training_chunk_size", 256)),
            "full_sample_bptt": bool(_config.get("full_sample_bptt", False)),
            "full_sample_activation_checkpointing": bool(
                _config.get("full_sample_activation_checkpointing", True)
            ),
            "full_sample_checkpoint_segment_size": int(
                _config.get("full_sample_checkpoint_segment_size", 128)
            ),
            "is_quantized": bool(_config.get("is_quantized", False)),
            "device": str(_device),
            "device_label": device_label,
            "torch_version": runtime_diag.get("torch_version"),
            "cuda_built": bool(runtime_diag.get("cuda_built", False)),
            "cuda_available": bool(runtime_diag.get("cuda_available", False)),
            "cuda_version": runtime_diag.get("cuda_version"),
            "cuda_device_name": runtime_diag.get("device_name"),
            "vram_total_mb": runtime_diag.get("total_memory_mb"),
            "total_params": int(total_params),
        }

        emit_load_progress(1.0, "Model ready")
        emit("model_loaded", {"config": model_config})
        if str(_device).startswith("cuda"):
            emit_status(f"CUDA acceleration active on {device_label}.")
        elif cuda_diagnostics().get("cuda_available"):
            emit_status("Model is running on CPU even though CUDA is available; select Auto or CUDA for GPU acceleration.")
        emit_status(f"Model loaded successfully from {_model_dir}.")

    except Exception as e:
        if _model is not None and (replacement_started or _tokenizer is None):
            _release_loaded_model()
        emit_error(
            f"Failed to load model: {e}\n{traceback.format_exc()}",
            operation="model_loading",
        )
    finally:
        _finish_operation("model_loading")


def handle_generate(params: dict):
    """Stream generation with the same runtime states as CLI chat mode."""
    global _model, _tokenizer, _device
    global _h_state, _l_state, _prev_context, _target_context
    global _drift_state, _ltm_state, _total_tokens_generated, _cpu_threads
    global _pending_feedback

    if _model is None:
        emit_error("No model loaded.", operation="generation")
        emit("generation_complete", {"status": "rejected"})
        return

    message = params.get("message", "")
    sampling = params.get("sampling", {})
    if not isinstance(sampling, dict):
        emit_error(
            "Generation sampling settings must be a JSON object.",
            operation="generation",
        )
        emit("generation_complete", {"status": "rejected"})
        return

    online_learning = params.get("online_learning", {})
    if not isinstance(online_learning, dict):
        emit_error(
            "Generation online_learning settings must be a JSON object.",
            operation="generation",
        )
        emit("generation_complete", {"status": "rejected"})
        return
    passive_learning = online_learning.get("passive_learning", False)
    if not isinstance(passive_learning, bool):
        emit_error(
            "passive_learning must be a boolean.",
            operation="generation",
        )
        emit("generation_complete", {"status": "rejected"})
        return
    passive_learning_rate = None
    if passive_learning:
        try:
            passive_learning_rate = _validated_online_learning_rate(
                online_learning.get("passive_lr"),
                default=float(_config.get("passive_ltm_lr", 5e-6)),
                maximum=1e-3,
                label="Passive LTM learning rate",
            )
        except ValueError as exc:
            emit_error(str(exc), operation="generation")
            emit("generation_complete", {"status": "rejected"})
            return

    claimed, active = _try_begin_operation("generation")
    if not claimed:
        _reject_busy("generation", active, terminal_event="generation_complete")
        return

    # Once a new causal turn begins, feedback for the prior turn is no longer
    # eligible. A replacement is published only after this generation commits.
    _pending_feedback = None
    _stop_generation.clear()
    try:
        _apply_thread_count(sampling.get("cpu_threads", _cpu_threads))
    except Exception as exc:
        _finish_operation("generation")
        emit_error(f"Could not configure generation runtime: {exc}", operation="generation")
        emit("generation_complete", {"status": "error"})
        return

    def _gen():
        global _h_state, _l_state, _prev_context, _target_context
        global _drift_state, _ltm_state, _total_tokens_generated
        global _pending_feedback

        status = "error"
        try:
            import torch
            from hierarchos.inference.chat import (
                advance_chat_model_state,
                boundary_drift_seed,
                resolve_inference_prefill_chunk_size,
                tbptt_chunk_ranges,
                uses_full_sample_inference_recurrence,
                wrap_for_hierarchos,
                zero_ltm_momentum_state,
            )

            max_new = max(0, int(sampling.get("max_new_tokens", 512)))
            if max_new > 65536:
                raise ValueError("max_new_tokens exceeds the bridge safety limit of 65536.")
            alpaca_mode = bool(_config.get("alpaca", False))
            prompt = wrap_for_hierarchos(message, alpaca_mode=alpaca_mode)
            prompt_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)
            response_ids = []

            _model.eval()
            # Match the stable CLI chat path: do not mutate LTM on every
            # autoregressive token. Per-token Hebbian writes can compound and
            # pull the model into repeated/off-topic text.
            _model.suppress_hebbian = True
            model_config = getattr(_model, "config", _config)
            prefill_chunk_size = resolve_inference_prefill_chunk_size(model_config)
            exact_full_sample = uses_full_sample_inference_recurrence(model_config)

            with torch.inference_mode():
                prompt_offset = _total_tokens_generated
                outputs = None
                for prefill_start, prefill_end in tbptt_chunk_ranges(
                    int(prompt_ids.shape[1]),
                    prefill_chunk_size,
                    prompt_offset,
                ):
                    absolute_start = prompt_offset + prefill_start
                    outputs, runtime_state = advance_chat_model_state(
                        _model,
                        prompt_ids[:, prefill_start:prefill_end],
                        device=_device,
                        h_state=_h_state,
                        l_state=_l_state,
                        prev_context=_prev_context,
                        target_context=_target_context,
                        drift_state=_drift_state,
                        drift_seed=boundary_drift_seed(
                            _drift_state,
                            absolute_start,
                            prefill_chunk_size,
                            exact_full_sample=exact_full_sample,
                        ),
                        ltm_state=_ltm_state,
                        global_pos_offset=absolute_start,
                        return_last_logit_only=True,
                    )

                    (
                        _h_state,
                        _l_state,
                        _prev_context,
                        _target_context,
                        _drift_state,
                        _ltm_state,
                    ) = runtime_state

                if outputs is None:
                    raise RuntimeError("Tokenizer produced an empty formatted prompt.")
                logits = outputs["logits"]
                _total_tokens_generated += prompt_ids.shape[1]

                current_ids = (
                    _sample_next_token(logits, response_ids, sampling, _tokenizer)
                    if max_new > 0
                    else None
                )

                for _ in range(max_new):
                    if _stop_generation.is_set() or current_ids is None:
                        break

                    next_token = int(current_ids.item())
                    terminal_eos = (
                        _tokenizer.eos_token_id is not None
                        and next_token == int(_tokenizer.eos_token_id)
                    )
                    if not terminal_eos:
                        response_ids.append(next_token)
                        token_str = _tokenizer.decode([next_token])
                        if token_str:
                            emit("token", {"text": token_str})

                    # Every sampled token, including terminal EOS, must be
                    # consumed exactly once before recurrent state is saved or
                    # carried to the next GUI turn.
                    outputs, runtime_state = advance_chat_model_state(
                        _model,
                        current_ids,
                        device=_device,
                        h_state=_h_state,
                        l_state=_l_state,
                        prev_context=_prev_context,
                        target_context=_target_context,
                        drift_state=_drift_state,
                        drift_seed=boundary_drift_seed(
                            _drift_state,
                            _total_tokens_generated,
                            prefill_chunk_size,
                            exact_full_sample=exact_full_sample,
                        ),
                        ltm_state=_ltm_state,
                        global_pos_offset=_total_tokens_generated,
                        return_last_logit_only=True,
                    )

                    logits = outputs["logits"]
                    (
                        _h_state,
                        _l_state,
                        _prev_context,
                        _target_context,
                        _drift_state,
                        _ltm_state,
                    ) = runtime_state
                    _total_tokens_generated += 1

                    if terminal_eos:
                        break
                    current_ids = _sample_next_token(logits, response_ids, sampling, _tokenizer)

            _ltm_state = zero_ltm_momentum_state(_model, _ltm_state)
            _model.suppress_hebbian = True
            status = "stopped" if _stop_generation.is_set() else "completed"
            if status == "completed" and passive_learning:
                try:
                    from hierarchos.models.ltm import LTMModule

                    passive_result = _apply_bridge_online_ltm_transaction(
                        prompt_ids[0].detach().to("cpu"),
                        None,
                        source_id=LTMModule.SRC_USER_INTERACTION,
                        penalty=False,
                        learning_rate=passive_learning_rate,
                        learn_input_tokens=True,
                    )
                    if bool(passive_result.get("committed", False)):
                        emit_status(
                            "Passive prompt-only LTM update committed; generated "
                            "response tokens were not used as training targets."
                        )
                        if passive_result.get("persisted"):
                            emit_status(
                                "Online LTM overlay autosaved to "
                                f"{passive_result.get('path')}."
                            )
                        elif passive_result.get("persistence_error"):
                            emit_error(
                                "Passive prompt learning committed in memory but "
                                "could not be persisted: "
                                f"{passive_result.get('persistence_error')}",
                                operation="passive_learning",
                            )
                    else:
                        emit_status(
                            "Passive prompt-only LTM update was safely rejected "
                            f"({passive_result.get('reason') or 'unknown reason'}); "
                            "memory is unchanged."
                        )
                except Exception as exc:
                    # Generation already completed successfully. Keep the answer
                    # and report the opt-in memory update failure independently.
                    emit_error(
                        f"Passive prompt learning failed: {exc}",
                        operation="passive_learning",
                    )
            if status == "completed" and response_ids:
                _pending_feedback = {
                    "prompt_ids": prompt_ids[0].detach().to("cpu").clone(),
                    "response_ids": torch.tensor(response_ids, dtype=torch.long),
                    "created_at": time.time(),
                }

        except Exception as e:
            emit_error(
                f"Generation error: {e}\n{traceback.format_exc()}",
                operation="generation",
            )
        finally:
            _stop_generation.clear()
            _finish_operation("generation")
            emit("generation_complete", {"status": status})

    try:
        threading.Thread(target=_gen, daemon=True).start()
    except Exception as exc:
        _finish_operation("generation")
        emit_error(f"Could not start generation worker: {exc}", operation="generation")
        emit("generation_complete", {"status": "error"})


def handle_start_training(params: dict):
    """Start a training run using the modular hierarchos.training.trainer."""
    global _model, _tokenizer, _device, _config
    if _model is None:
        emit_error("No model loaded — cannot train.", operation="training")
        emit("training_complete", {"status": "rejected"})
        return

    claimed, active = _try_begin_operation("training")
    if not claimed:
        _reject_busy("training", active, terminal_event="training_complete")
        return

    _stop_training.clear()

    def _train():
        global _ltm_token_clock
        status = "error"
        trainer_module = None
        original_trainer_tqdm = None
        try:
            import torch
            import argparse
            import time

            # Import from the modular package (same as hierarchos_cli.py)
            from hierarchos import (
                train as hierarchos_train,
                OriginalJSONLDataset,
                create_dataloader_for_jsonl,
                create_map_style_dataloader,
                process_text_sample,
                process_tokenized_sample,
            )
            from hierarchos.models.revisions import (
                architecture_default_commitment_threshold,
            )

            data_path = params.get("data_path", "")
            if not data_path or not os.path.exists(data_path):
                emit_error(f"Training data not found: {data_path}")
                return

            emit_status("Preparing training...")

            def _int_param(name, default, minimum=1):
                try:
                    fallback = int(default)
                except (TypeError, ValueError, OverflowError):
                    fallback = int(minimum)
                try:
                    value = int(params.get(name, default))
                except Exception:
                    value = fallback
                try:
                    value = int(value)
                except Exception:
                    value = fallback
                return max(minimum, value)

            def _bool_param(name, default=False):
                value = params.get(name, default)
                if isinstance(value, bool):
                    return value
                if value is None:
                    return bool(default)
                if isinstance(value, (int, float)):
                    return bool(value)
                text = str(value).strip().lower()
                if text in {"1", "true", "t", "yes", "y", "on"}:
                    return True
                if text in {"0", "false", "f", "no", "n", "off"}:
                    return False
                return bool(default)

            def _float_param(name, default):
                try:
                    fallback = float(default)
                except (TypeError, ValueError, OverflowError):
                    fallback = 0.0
                try:
                    value = float(params.get(name, fallback))
                except (TypeError, ValueError, OverflowError):
                    value = fallback
                if not value == value or value in (float("inf"), -float("inf")):
                    raise ValueError(f"{name} must be finite.")
                return value

            train_alpaca = _bool_param(
                "alpaca",
                bool(_config.get("alpaca", False)),
            )
            train_kayla = _bool_param(
                "kayla",
                bool(_config.get("kayla", False)),
            )

            context_dim = _int_param("context_dim", _config.get("context_dim", 448), 32)
            train_arch = {
                "context_dim": context_dim,
                "persistent_dim": _int_param("persistent_dim", _config.get("persistent_dim", 128), 1),
                "ltm_slots": _int_param("ltm_slots", _config.get("ltm_slots", 1024), 1),
                "ltm_key_dim": _int_param("ltm_key_dim", _config.get("ltm_key_dim", 128), 8),
                "ltm_val_dim": _int_param("ltm_val_dim", _config.get("ltm_val_dim", 128), 8),
                "h_hidden": _int_param("h_hidden", _config.get("h_hidden", context_dim), 32),
                "l_hidden": _int_param("l_hidden", _config.get("l_hidden", context_dim), 32),
                "h_stride": _int_param("h_stride", _config.get("h_stride", 4), 1),
                "max_h_steps": _int_param("max_h_steps", _config.get("max_h_steps", 5), 1),
                "max_l_steps": _int_param("max_l_steps", _config.get("max_l_steps", 5), 1),
                "ltm_topk": _int_param("ltm_topk", _config.get("ltm_topk", 4), 1),
                "max_length": _int_param("max_length", _config.get("max_length", 1024), 32),
            }
            # A preloaded model's tensor geometry is immutable. The trainer
            # intentionally keeps these loaded values authoritative, so fail
            # loudly instead of displaying/saving a GUI value that was ignored.
            immutable_geometry = (
                "context_dim",
                "persistent_dim",
                "ltm_slots",
                "ltm_key_dim",
                "ltm_val_dim",
                "h_hidden",
                "l_hidden",
                "h_stride",
                "max_h_steps",
                "max_l_steps",
                "ltm_topk",
            )
            for key in immutable_geometry:
                loaded_value = int(_config.get(key, train_arch[key]))
                if key in params and int(params[key]) != loaded_value:
                    raise ValueError(
                        f"{key} is fixed by the loaded model "
                        f"({loaded_value}); received {params[key]!r}."
                    )
                train_arch[key] = loaded_value
            train_arch["ltm_topk"] = min(
                train_arch["ltm_topk"], train_arch["ltm_slots"]
            )
            rwkv_head_size = params.get("rwkv_head_size", _config.get("rwkv_head_size", None))
            try:
                rwkv_head_size = int(rwkv_head_size) if rwkv_head_size is not None else None
            except Exception:
                rwkv_head_size = None
            if rwkv_head_size is not None and rwkv_head_size <= 0:
                rwkv_head_size = None
            train_arch["rwkv_head_size"] = rwkv_head_size

            def _scan_auto_max_length(path: str) -> int:
                max_found = 0
                scanned = 0
                # The GUI caps training length at 32K, so tokenizing beyond
                # that point only wastes RAM/CPU and cannot change the result.
                scan_max_length = 32768

                def consider(obj):
                    nonlocal max_found, scanned
                    if not isinstance(obj, dict):
                        return
                    processed = process_tokenized_sample(obj, scan_max_length)
                    if processed is None:
                        processed = process_text_sample(
                            _tokenizer,
                            obj,
                            scan_max_length,
                            train_kayla,
                            alpaca_mode=train_alpaca,
                        )
                    if processed:
                        max_found = max(max_found, len(processed["input_ids"]))
                        scanned += 1

                emit_status("Scanning dataset for auto max length...")
                for obj in _iter_training_source_objects(path):
                    consider(obj)

                if max_found <= 0:
                    raise ValueError("No valid text or instruction/output samples found while scanning.")

                auto_len = (max_found + 16 + 7) & -8
                capped_len = min(max(auto_len, 32), 32768)
                if capped_len != auto_len:
                    emit_status(
                        f"Auto max length found {max_found} tokens; capped at {capped_len}."
                    )
                else:
                    emit_status(
                        f"Auto max length found {max_found} tokens across {scanned} samples; using {capped_len}."
                    )
                return capped_len

            if bool(params.get("auto_max_length", False)):
                try:
                    train_arch["max_length"] = _scan_auto_max_length(data_path)
                except Exception as exc:
                    emit_error(f"Auto max length scan failed: {exc}")
                    return

            train_batch_size = int(params.get("batch_size", 64))
            train_chunk_size = int(
                params.get(
                    "training_chunk_size",
                    _config.get("training_chunk_size", 256),
                )
            )
            full_sample_bptt = _bool_param(
                "full_sample_bptt",
                bool(_config.get("full_sample_bptt", False)),
            )
            full_sample_activation_checkpointing = _bool_param(
                "full_sample_activation_checkpointing",
                bool(_config.get("full_sample_activation_checkpointing", True)),
            )
            full_sample_checkpoint_segment_size = max(
                1,
                int(
                    params.get(
                        "full_sample_checkpoint_segment_size",
                        _config.get("full_sample_checkpoint_segment_size", 128),
                    )
                ),
            )
            requested_amp = _bool_param("amp", True)
            effective_amp = requested_amp and str(_device).startswith("cuda")
            default_workers = 8 if str(_device).startswith("cuda") and train_batch_size >= 64 else 0
            default_bucket_size = 8192 if str(_device).startswith("cuda") and train_batch_size >= 64 else None

            # Build args namespace matching what hierarchos.training.trainer.train() expects
            # (mirrors the argparse in hierarchos_cli.py)
            checkpoint_metadata = getattr(
                _model,
                "_hierarchos_checkpoint_metadata",
                {},
            )
            if not isinstance(checkpoint_metadata, dict):
                checkpoint_metadata = {}
            source_weights_path = checkpoint_metadata.get("source_weights_path")
            train_args = argparse.Namespace(
                mode="train",
                # model_override remains authoritative; retaining the source
                # path/base epoch makes weights-only continuation provenance
                # and completed_epoch accounting honest.
                model_path=source_weights_path or _model_dir,
                base_completed_epoch=int(
                    _config.get(
                        "completed_epoch",
                        checkpoint_metadata.get("completed_epoch", 0),
                    )
                    or 0
                ),
                out_dir=params.get("out_dir", "./hierarchos_model"),
                resume_from_ckpt=None,
                # Architecture (defaults come from loaded config, GUI may override)
                context_dim=train_arch["context_dim"],
                persistent_dim=train_arch["persistent_dim"],
                ltm_slots=train_arch["ltm_slots"],
                ltm_key_dim=train_arch["ltm_key_dim"],
                ltm_val_dim=train_arch["ltm_val_dim"],
                h_hidden=train_arch["h_hidden"],
                l_hidden=train_arch["l_hidden"],
                h_stride=train_arch["h_stride"],
                max_h_steps=train_arch["max_h_steps"],
                max_l_steps=train_arch["max_l_steps"],
                ltm_topk=train_arch["ltm_topk"],
                max_length=train_arch["max_length"],
                auto_max_length=bool(params.get("auto_max_length", False)),
                vocab_size=_config.get("vocab_size", len(_tokenizer)),
                use_deepembed=bool(_config.get("use_deepembed", True)),
                use_rosa=bool(_config.get("use_rosa", True)),
                rosa_max_context=int(_config.get("rosa_max_context", 512)),
                rwkv_head_size=train_arch["rwkv_head_size"],
                # Training hyperparams from GUI
                epochs=int(params.get("epochs", 3)),
                batch_size=train_batch_size,
                accumulation_steps=int(params.get("accumulation_steps", 1)),
                accumulation_normalization=str(
                    params.get(
                        "accumulation_normalization",
                        _config.get("accumulation_normalization", "weighted-token"),
                    )
                ),
                starting_lr=_float_param("learning_rate", 1e-4),
                min_lr=_float_param("min_lr", 1e-6),
                warmup_steps=_int_param(
                    "warmup_steps",
                    _config.get("warmup_steps", 0),
                    0,
                ),
                warmup_ratio=_float_param(
                    "warmup_ratio",
                    _config.get("warmup_ratio", 0.0),
                ),
                training_chunk_size=train_chunk_size,
                full_sample_bptt=full_sample_bptt,
                full_sample_activation_checkpointing=(
                    full_sample_activation_checkpointing if full_sample_bptt else False
                ),
                full_sample_checkpoint_segment_size=full_sample_checkpoint_segment_size,
                grad_clip=float(params.get("grad_clip", 1.0)),
                persist_state=(
                    False if full_sample_bptt else bool(params.get("persist_state", False))
                ),
                amp=effective_amp,
                save_steps=int(params.get("save_steps", 0)),
                num_workers=max(0, int(params.get("num_workers", default_workers))),
                prefetch_factor=params.get("prefetch_factor", None),
                length_bucketing=bool(params.get("length_bucketing", True)),
                length_bucket_size=params.get("length_bucket_size", default_bucket_size),
                progress_log_steps=int(params.get("progress_log_steps", 25)),
                # Semantic training settings not exposed in the GUI inherit
                # the checkpoint. Hard-coded defaults here previously changed
                # the objective and stability envelope on continuation.
                disable_lr_schedule=_bool_param(
                    "disable_lr_schedule",
                    bool(_config.get("disable_lr_schedule", False)),
                ),
                ltm_lr=_float_param("ltm_lr", _config.get("ltm_lr", 1e-3)),
                min_ltm_lr=_float_param(
                    "min_ltm_lr",
                    _config.get("min_ltm_lr", _config.get("min_lr", 1e-6)),
                ),
                disable_ltm_lr_schedule=_bool_param(
                    "disable_ltm_lr_schedule",
                    bool(_config.get("disable_ltm_lr_schedule", False)),
                ),
                ltm_training_mode=str(
                    params.get(
                        "ltm_training_mode",
                        _config.get("ltm_training_mode", "inner-update"),
                    )
                ),
                rwkv_weight_decay=_float_param(
                    "rwkv_weight_decay",
                    _config.get("rwkv_weight_decay", 0.1),
                ),
                ltm_score_grad_scale=_float_param(
                    "ltm_score_grad_scale",
                    _config.get("ltm_score_grad_scale", 1.0),
                ),
                ltm_cpu_gather_retrieval=_bool_param(
                    "ltm_cpu_gather_retrieval",
                    bool(_config.get("ltm_cpu_gather_retrieval", True)),
                ),
                ltm_cpu_sparse_update=_bool_param(
                    "ltm_cpu_sparse_update",
                    bool(_config.get("ltm_cpu_sparse_update", True)),
                ),
                kayla=train_kayla,
                alpaca=train_alpaca,
                train_prompt_tokens=_bool_param(
                    "train_prompt_tokens",
                    bool(_config.get("train_prompt_tokens", True)),
                ),
                prompt_loss_weight=_float_param(
                    "prompt_loss_weight",
                    _config.get("prompt_loss_weight", 1.0),
                ),
                response_loss_weight=_float_param(
                    "response_loss_weight",
                    _config.get("response_loss_weight", 1.0),
                ),
                response_boundary_loss_weight=_float_param(
                    "response_boundary_loss_weight",
                    _config.get("response_boundary_loss_weight", 1.0),
                ),
                response_boundary_tokens=_int_param(
                    "response_boundary_tokens",
                    _config.get("response_boundary_tokens", 0),
                    0,
                ),
                min_response_tokens=_int_param(
                    "min_response_tokens",
                    _config.get("min_response_tokens", 1),
                    0,
                ),
                drop_empty_completions=_bool_param(
                    "drop_empty_completions",
                    bool(_config.get("drop_empty_completions", True)),
                ),
                lora_r=8, lora_alpha=16,
                ponder_loss_weight=_float_param(
                    "ponder_loss_weight",
                    _config.get("ponder_loss_weight", 0.01),
                ),
                adaptive_ponder=_bool_param(
                    "adaptive_ponder",
                    bool(_config.get("adaptive_ponder", False)),
                ),
                ponder_target_scale=_float_param(
                    "ponder_target_scale",
                    _config.get("ponder_target_scale", 0.5),
                ),
                ponder_objective=str(
                    params.get(
                        "ponder_objective",
                        _config.get("ponder_objective", "auto"),
                    )
                ),
                ponder_huber_beta=_float_param(
                    "ponder_huber_beta",
                    _config.get("ponder_huber_beta", 0.5),
                ),
                commitment_loss_weight=_float_param(
                    "commitment_loss_weight",
                    _config.get("commitment_loss_weight", 0.5),
                ),
                max_commitment_cost_for_backward=_float_param(
                    "max_commitment_cost_for_backward",
                    _config.get("max_commitment_cost_for_backward", 2.0),
                ),
                max_ce_loss_for_backward=_float_param(
                    "max_ce_loss_for_backward",
                    _config.get("max_ce_loss_for_backward", 0.0),
                ),
                max_ponder_cost_for_backward=_float_param(
                    "max_ponder_cost_for_backward",
                    _config.get("max_ponder_cost_for_backward", 0.0),
                ),
                commitment_threshold=_float_param(
                    "commitment_threshold",
                    _config.get(
                        "commitment_threshold",
                        architecture_default_commitment_threshold(_config),
                    ),
                ),
                l_conv_atol=_float_param(
                    "l_conv_atol",
                    _config.get("l_conv_atol", 1e-4),
                ),
                detach_every_n_steps=(
                    0 if full_sample_bptt else _config.get("detach_every_n_steps", 32)
                ),
                gradient_checkpointing=_bool_param(
                    "gradient_checkpointing",
                    bool(_config.get("gradient_checkpointing", False)),
                ),
                h_halt_thresh=_float_param(
                    "h_halt_thresh",
                    _config.get("h_halt_thresh", 0.9),
                ),
                act_depth_temperature=_float_param(
                    "act_depth_temperature",
                    _config.get("act_depth_temperature", 0.05),
                ),
                encourage_thinking=_bool_param(
                    "encourage_thinking",
                    bool(_config.get("encourage_thinking", False)),
                ),
                memory_token_routers=bool(_config.get("memory_token_routers", True)),
                memory_gate_warmup_steps=_int_param(
                    "memory_gate_warmup_steps",
                    _config.get("memory_gate_warmup_steps", 2000),
                    0,
                ),
                memory_gate_warmup_floor=_float_param(
                    "memory_gate_warmup_floor",
                    _config.get("memory_gate_warmup_floor", 0.10),
                ),
                halt_logit_clamp=_float_param(
                    "halt_logit_clamp",
                    _config.get("halt_logit_clamp", 30.0),
                ),
                recurrent_state_clamp=_float_param(
                    "recurrent_state_clamp",
                    _config.get("recurrent_state_clamp", 50.0),
                ),
                context_state_clamp=_float_param(
                    "context_state_clamp",
                    _config.get("context_state_clamp", 50.0),
                ),
                activation_clamp=_float_param(
                    "activation_clamp",
                    _config.get("activation_clamp", 100.0),
                ),
                drift_state_clamp=_float_param(
                    "drift_state_clamp",
                    _config.get("drift_state_clamp", 5.0),
                ),
                drift_norm_clamp=_float_param(
                    "drift_norm_clamp",
                    _config.get("drift_norm_clamp", 0.0),
                ),
                drift_delta_scale=_float_param(
                    "drift_delta_scale",
                    _config.get("drift_delta_scale", 1.0),
                ),
                rwkv_channel_mix_key_clamp=_float_param(
                    "rwkv_channel_mix_key_clamp",
                    _config.get("rwkv_channel_mix_key_clamp", 12.0),
                ),
                rwkv_channel_mix_deepembed_clamp=_float_param(
                    "rwkv_channel_mix_deepembed_clamp",
                    _config.get("rwkv_channel_mix_deepembed_clamp", 4.0),
                ),
                startup_weight_max_abs=_float_param(
                    "startup_weight_max_abs",
                    _config.get("startup_weight_max_abs", 0.0),
                ),
                reset_halt_bias=None,
                override_scheduling=_bool_param(
                    "override_scheduling",
                    _bool_param("override-scheduling", False),
                ),
                compile=_bool_param("compile", bool(_config.get("compile", False))),
                force_compile=_bool_param(
                    "force_compile",
                    bool(_config.get("force_compile", False)),
                ),
                compile_mode=str(
                    params.get(
                        "compile_mode",
                        _config.get("compile_mode", "max-autotune-no-cudagraphs"),
                    )
                ),
                compile_backend=params.get(
                    "compile_backend",
                    _config.get("compile_backend"),
                ),
                compile_dynamic=_bool_param(
                    "compile_dynamic",
                    bool(_config.get("compile_dynamic", False)),
                ),
                compile_fullgraph_worker=_bool_param(
                    "compile_fullgraph_worker",
                    bool(_config.get("compile_fullgraph_worker", False)),
                ),
                compile_cudagraphs=_bool_param(
                    "compile_cudagraphs",
                    bool(_config.get("compile_cudagraphs", False)),
                ),
                compile_pad_to_chunk_size=_bool_param(
                    "compile_pad_to_chunk_size",
                    bool(_config.get("compile_pad_to_chunk_size", True)),
                ),
                compile_static_worker_loop=params.get(
                    "compile_static_worker_loop",
                    _config.get("compile_static_worker_loop"),
                ),
                compile_h_rnn=_bool_param(
                    "compile_h_rnn",
                    bool(_config.get("compile_h_rnn", True)),
                ),
                compile_quiet=True,
                debug_numerics=_bool_param(
                    "debug_numerics",
                    bool(_config.get("debug_numerics", False)),
                ),
                isolate_batch_ltm=_bool_param(
                    "isolate_batch_ltm",
                    bool(_config.get("isolate_batch_ltm", True)),
                ),
                cuda_chunked_lm_loss=_bool_param(
                    "cuda_chunked_lm_loss",
                    bool(_config.get("cuda_chunked_lm_loss", True)),
                ),
                cuda_loss_chunk_rows=_int_param(
                    "cuda_loss_chunk_rows",
                    _config.get("cuda_loss_chunk_rows", 0),
                    0,
                ),
                cpu_chunked_lm_loss=_bool_param(
                    "cpu_chunked_lm_loss",
                    bool(_config.get("cpu_chunked_lm_loss", True)),
                ),
                cpu_loss_chunk_rows=_int_param(
                    "cpu_loss_chunk_rows",
                    _config.get("cpu_loss_chunk_rows", 0),
                    0,
                ),
                eval_tasks=None, eval_every_epoch=1, eval_batch_size=1,
                eval_limit=None, eval_steps=None,
            )
            # The GUI exposes AMP as an explicit user choice. Preserve both
            # checked and unchecked states when CUDA runtime defaults are
            # applied later by the shared trainer.
            train_args._explicit_cli_dests = frozenset({"amp"})

            emit_status(
                "Architecture: "
                f"context={train_args.context_dim}, "
                f"H={train_args.h_hidden}, L={train_args.l_hidden}, "
                f"LTM={train_args.ltm_slots} slots/{train_args.ltm_val_dim} value dim, "
                f"max_len={train_args.max_length}"
            )
            emit_status(
                "GUI training is a weights-only continuation: optimizer and LR "
                "scheduler state start fresh. Use the CLI checkpoint-resume path "
                "when exact optimizer/scheduler continuation is required."
            )

            # Build dataloader (same pattern as hierarchos_cli.py)
            if data_path.lower().endswith((".jsonl", ".ndjson")):
                emit_status("Streaming JSONL dataset...")
                dataloader = create_dataloader_for_jsonl(
                    data_path,
                    _tokenizer,
                    train_args.max_length,
                    train_args.batch_size,
                    _tokenizer.pad_token_id,
                    num_workers=train_args.num_workers,
                    kayla_mode=train_args.kayla,
                    alpaca_mode=train_args.alpaca,
                    train_prompt_tokens=train_args.train_prompt_tokens,
                    prompt_loss_weight=train_args.prompt_loss_weight,
                    response_loss_weight=train_args.response_loss_weight,
                    response_boundary_loss_weight=train_args.response_boundary_loss_weight,
                    response_boundary_tokens=train_args.response_boundary_tokens,
                    min_response_tokens=train_args.min_response_tokens,
                    drop_empty_completions=train_args.drop_empty_completions,
                    use_length_bucketing=train_args.length_bucketing,
                    bucket_size=train_args.length_bucket_size,
                    device=_device,
                    prefetch_factor=train_args.prefetch_factor,
                )
                emit_status(
                    "Validating streaming dataset size for an exact training schedule..."
                )
                dataloader_len, usable_sample_count = _exact_dataloader_batches(
                    dataloader,
                    train_args.batch_size,
                )
                if usable_sample_count is not None:
                    emit_status(
                        f"Validated {usable_sample_count} usable samples "
                        f"({dataloader_len} batches/epoch)."
                    )
            else:
                dataset = OriginalJSONLDataset(
                    data_path, _tokenizer, train_args.max_length, train_args.kayla,
                    alpaca_mode=train_args.alpaca,
                    train_prompt_tokens=train_args.train_prompt_tokens,
                    prompt_loss_weight=train_args.prompt_loss_weight,
                    response_loss_weight=train_args.response_loss_weight,
                    response_boundary_loss_weight=train_args.response_boundary_loss_weight,
                    response_boundary_tokens=train_args.response_boundary_tokens,
                    min_response_tokens=train_args.min_response_tokens,
                    drop_empty_completions=train_args.drop_empty_completions,
                )
                dataloader = create_map_style_dataloader(
                    dataset,
                    train_args.batch_size,
                    _tokenizer.pad_token_id,
                    train_args.num_workers,
                    use_length_bucketing=train_args.length_bucketing,
                    bucket_size=train_args.length_bucket_size,
                    device=_device,
                    prefetch_factor=train_args.prefetch_factor,
                )
                dataloader_len = len(dataloader)

            emit_status(f"Training started — {dataloader_len} batches/epoch × {train_args.epochs} epochs")

            # The trainer binds ``tqdm`` at module import time (``from tqdm
            # import tqdm``), so patch that exact binding. Patching the tqdm
            # package object does not affect the already-imported trainer.
            import hierarchos.training.trainer as trainer_module
            original_trainer_tqdm = trainer_module.tqdm

            class GUITqdm(original_trainer_tqdm):
                """tqdm wrapper that emits training metrics to the GUI bridge."""
                def __init__(self, *a, **kw):
                    self._gui_epoch = 0
                    self._gui_step = 0
                    self._gui_start_time = time.time()
                    self._gui_tokens_processed = 0
                    # Extract epoch from desc if available
                    desc = kw.get('desc', '') or (a[1] if len(a) > 1 else '')
                    if 'Epoch' in str(desc):
                        try:
                            parts = str(desc).split('/')
                            self._gui_epoch = int(parts[0].split()[-1]) - 1
                        except (ValueError, IndexError):
                            pass
                    super().__init__(*a, **kw)

                def __iter__(self):
                    for item in super().__iter__():
                        if _stop_training.is_set():
                            self.close()
                            raise _TrainingCancelled("Training stopped by user.")
                        yield item

                def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
                    if _stop_training.is_set():
                        self.close()
                        raise _TrainingCancelled("Training stopped by user.")

                    super().set_postfix(ordered_dict, refresh, **kwargs)

                    # Extract metrics from postfix
                    postfix = ordered_dict or kwargs
                    self._gui_step += 1

                    loss = 0.0
                    lr = 0.0
                    ponder_cost = None
                    commitment_cost = None

                    if 'loss' in postfix:
                        try: loss = float(postfix['loss'])
                        except: pass
                    if 'lr' in postfix:
                        try: lr = float(postfix['lr'])
                        except: pass
                    if 'ponder' in postfix:
                        try: ponder_cost = float(postfix['ponder'])
                        except: pass
                    if 'commit' in postfix:
                        try: commitment_cost = float(postfix['commit'])
                        except: pass

                    # Estimate tokens/sec
                    elapsed = time.time() - self._gui_start_time
                    chunk_size = train_args.training_chunk_size
                    batch_size = train_args.batch_size
                    tokens_this_step = chunk_size * batch_size
                    self._gui_tokens_processed += tokens_this_step
                    tps = self._gui_tokens_processed / max(elapsed, 0.01)

                    emit("training_metrics", {
                        "epoch": self._gui_epoch,
                        "step": self._gui_step,
                        "loss": loss,
                        "lr": lr,
                        "ponder_cost": ponder_cost,
                        "commitment_cost": commitment_cost,
                        "tokens_per_sec": round(tps, 1),
                    })

            trainer_module.tqdm = GUITqdm

            hierarchos_train(
                train_args,
                _device,
                _tokenizer,
                dataloader,
                dataloader_len,
                model_override=_model,
            )
            if _stop_training.is_set():
                status = "stopped"
                emit_status("Training stopped by user.")
            else:
                status = "completed"
                emit_status("Training complete!")

        except _TrainingCancelled:
            status = "stopped"
            emit_status("Training stopped by user.")
        except Exception as e:
            emit_error(
                f"Training error: {e}\n{traceback.format_exc()}",
                operation="training",
            )
            emit_status("Training stopped due to error.")
        finally:
            try:
                if trainer_module is not None and original_trainer_tqdm is not None:
                    try:
                        trainer_module.tqdm = original_trainer_tqdm
                    except Exception as restore_exc:
                        emit_error(
                            f"Training progress-hook restore error: {restore_exc}",
                            operation="training",
                        )
                        status = "error"

                # Any update, cancellation, or partial failure invalidates
                # recurrence produced by the pre-training weights.
                try:
                    _reset_runtime_state()
                    from hierarchos.inference.chat_state import clear_ltm_working_memory

                    if _model is not None:
                        clear_ltm_working_memory(_model)
                        _ltm_token_clock = 0
                        _model.eval()
                        _model.suppress_hebbian = True
                except Exception as cleanup_exc:
                    emit_error(
                        f"Training cleanup error: {cleanup_exc}",
                        operation="training",
                    )
                    status = "error"
            finally:
                # Terminal cleanup must not be skipped even if lifecycle
                # restoration itself encounters an unexpected failure.
                _stop_training.clear()
                _finish_operation("training")
                emit("training_complete", {"status": status})

    try:
        threading.Thread(target=_train, daemon=True).start()
    except Exception as exc:
        _finish_operation("training")
        emit_error(f"Could not start training worker: {exc}", operation="training")
        emit("training_complete", {"status": "error"})


def handle_stop_generation(_params):
    _stop_generation.set()
    emit_status("Generation stop requested.")


def handle_stop_training(_params):
    _stop_training.set()
    emit_status("Training stop requested.")


@_exclusive_operation("model_inspection")
def handle_get_model_info(_params):
    """Return parameter-level model inspection data."""
    global _model
    if _model is None:
        emit_error("No model loaded.")
        return

    import torch
    layers = []
    total_params = 0
    trainable_params = 0
    for name, param in _model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
        p = param.detach()
        # Reduce on the owning device and transfer only scalars. The previous
        # inspector path copied every full parameter to CPU float32 immediately
        # after model load, temporarily duplicating multi-gigabyte models.
        mean_val = float(p.mean(dtype=torch.float32).item())
        std_val = (
            0.0
            if p.numel() <= 1
            else float(p.std(unbiased=False).item())
        )
        layers.append({
            "name": name,
            "param_count": count,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "min": round(float(p.min().item()), 6),
            "max": round(float(p.max().item()), 6),
        })

    emit("model_info", {
        "layers": layers,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })


@_exclusive_operation("ltm_snapshot")
def handle_get_ltm_snapshot(_params):
    """Return current LTM memory state for heatmap visualization."""
    global _model
    if _model is None:
        emit_error("No model loaded.")
        return

    try:
        import torch

        # Find the LTM module (hierarchos.models.ltm.LTMModule)
        ltm = None
        for name, module in _model.named_modules():
            cls_name = type(module).__name__.lower()
            if 'ltm' in cls_name or 'longtermmemory' in cls_name:
                ltm = module
                break

        if ltm is None and hasattr(_model, 'ltm'):
            ltm = _model.ltm

        if ltm is None:
            emit_status("No LTM module found in model.")
            return

        def _runtime_or_module(index, attr):
            target = getattr(ltm, attr, None)
            if isinstance(_ltm_state, (tuple, list)) and len(_ltm_state) > index:
                value = _ltm_state[index]
                if torch.is_tensor(value):
                    if (
                        torch.is_tensor(target)
                        and value.dim() == target.dim() + 1
                        and value.shape[0] == 1
                    ):
                        value = value.squeeze(0)
                    return value
            return target

        def _matrix_preview(value):
            if not torch.is_tensor(value):
                return []
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                raise ValueError("LTM snapshot contains non-finite memory values.")
            rows = min(int(value.shape[0]), 64) if value.dim() else 0
            if value.dim() > 1:
                cols = min(int(value.shape[1]), 32)
                preview = value.detach()[:rows, :cols].to(device="cpu", dtype=torch.float32)
                return preview.tolist()
            preview = value.detach()[:rows].to(device="cpu", dtype=torch.float32)
            return [[float(item)] for item in preview.tolist()]

        fast_vals = _matrix_preview(_runtime_or_module(0, "fast_vals"))
        # Hierarchos calls the trainable/consolidated value bank ``vals``.
        # The old bridge looked for nonexistent aliases and displayed zeros.
        slow_vals = _matrix_preview(getattr(ltm, "vals", None))
        rows = min(
            64,
            len(fast_vals) if fast_vals else len(slow_vals),
        )

        timestamp_tensor = _runtime_or_module(4, "timestamps")
        source_tensor = _runtime_or_module(5, "sources")
        timestamps = []
        sources = []
        if torch.is_tensor(timestamp_tensor):
            timestamp_tensor = timestamp_tensor.detach().reshape(-1)[:rows]
            if not timestamp_tensor.is_floating_point() or not bool(
                (torch.isfinite(timestamp_tensor) & (timestamp_tensor >= 0)).all().item()
            ):
                raise ValueError("LTM snapshot contains invalid timestamps.")
            timestamps = timestamp_tensor.cpu().tolist()
        if torch.is_tensor(source_tensor):
            source_tensor = source_tensor.detach().reshape(-1)[:rows]
            if source_tensor.dtype != torch.int64:
                raise ValueError("LTM snapshot sources must use torch.int64.")
            sources = [int(item) for item in source_tensor.cpu().tolist()]

        emit("ltm_snapshot", {
            "fast_vals": fast_vals,
            "slow_vals": slow_vals,
            "timestamps": timestamps,
            "sources": sources,
        })

    except Exception as e:
        emit_error(f"LTM snapshot error: {e}")


@_exclusive_operation("ltm_save")
def handle_save_ltm_updates(_params):
    """Persist runtime LTM state next to the loaded model without overwriting base weights."""
    if _model is None:
        emit_error("No model loaded.")
        return
    if not hasattr(_model, "ltm"):
        emit_error("No LTM module found in model.")
        return

    try:
        from hierarchos.inference.chat import save_ltm_delta_overlay_atomic

        if _ltm_overlay_write_blocked_reason:
            raise RuntimeError(
                "Refusing to overwrite an LTM sidecar that failed validation: "
                f"{_ltm_overlay_write_blocked_reason}"
            )
        path = save_ltm_delta_overlay_atomic(
            _model,
            _ltm_updates_path(),
            _ltm_state,
            tokenizer=_tokenizer,
            extra_metadata={
                "total_tokens_generated": int(_total_tokens_generated),
                "base_model_identity": copy.deepcopy(_model_identity),
                "bridge_runtime_identity": _bridge_runtime_identity(),
            },
        )
        emit("ltm_saved", {"path": path})
        emit_status(f"Identity-bound LTM overlay v3 saved to {path}.")

    except Exception as e:
        emit_error(f"Failed to save LTM updates: {e}\n{traceback.format_exc()}")


@_exclusive_operation("chat_state_save")
def handle_save_chat_runtime_state(params: dict):
    if _model is None:
        emit_error("No model loaded.")
        return
    try:
        path = _write_chat_runtime_state(params.get("path", ""))
        emit("chat_state_saved", {"path": path})
    except Exception as e:
        emit_error(f"Failed to save chat runtime state: {e}\n{traceback.format_exc()}")


@_exclusive_operation("chat_state_load")
def handle_load_chat_runtime_state(params: dict):
    if _model is None:
        emit_error("No model loaded.")
        return
    try:
        path = _load_chat_runtime_state(params.get("path", ""))
        emit("chat_state_loaded", {"path": path})
        emit_status(f"Restored chat runtime state from {path}.")
    except Exception as e:
        emit_error(f"Failed to load chat runtime state: {e}\n{traceback.format_exc()}")


@_exclusive_operation("chat_state_reset")
def handle_reset_chat_runtime_state(params: dict):
    if _model is None:
        emit_error("No model loaded.")
        return
    try:
        _reset_runtime_state()
        path = _write_chat_runtime_state(params.get("path", ""))
        emit("chat_state_saved", {"path": path})
    except Exception as e:
        emit_error(f"Failed to reset chat runtime state: {e}\n{traceback.format_exc()}")


@_exclusive_operation("feedback")
def handle_send_feedback(params: dict):
    """Apply one explicit, bounded update to the last completed GUI exchange."""
    global _pending_feedback, _ltm_state, _ltm_token_clock

    if _model is None or _tokenizer is None:
        emit_error("No model/tokenizer is loaded.", operation="feedback")
        emit("feedback_complete", {"status": "rejected", "reason": "no-model"})
        return

    positive = params.get("positive", True)
    if not isinstance(positive, bool):
        emit_error("Feedback polarity must be a boolean.", operation="feedback")
        emit(
            "feedback_complete",
            {"status": "rejected", "reason": "invalid-polarity"},
        )
        return

    try:
        learning_rate = _validated_online_learning_rate(
            params.get("learning_rate"),
            default=float(_config.get("online_ltm_lr", 1e-3)),
            maximum=0.1,
            label="Explicit feedback LTM learning rate",
        )
    except ValueError as exc:
        emit_error(str(exc), operation="feedback")
        emit(
            "feedback_complete",
            {"status": "rejected", "reason": "invalid-learning-rate"},
        )
        return

    pending = _pending_feedback
    if not isinstance(pending, dict):
        emit_status("No completed assistant response is pending feedback.")
        emit(
            "feedback_complete",
            {"status": "rejected", "reason": "nothing-pending"},
        )
        return

    # A completed exchange is single-use: two rapid button presses must not
    # compound the same answer or apply contradictory updates.
    _pending_feedback = None

    try:
        import torch
        from hierarchos.models.ltm import LTMModule

        prompt_ids = pending.get("prompt_ids")
        response_ids = pending.get("response_ids")
        if (
            not torch.is_tensor(prompt_ids)
            or prompt_ids.ndim != 1
            or not torch.is_tensor(response_ids)
            or response_ids.ndim != 1
            or response_ids.numel() == 0
        ):
            raise ValueError("Pending GUI feedback tensors are malformed.")

        source_id = (
            LTMModule.SRC_USER_INTERACTION
            if positive
            else LTMModule.SRC_CORRECTION
        )
        result = _apply_bridge_online_ltm_transaction(
            prompt_ids,
            response_ids,
            source_id=source_id,
            penalty=not positive,
            learning_rate=learning_rate,
        )
        if not bool(result.get("committed", False)):
            reason = str(result.get("reason") or "rejected")
            emit_status(
                f"Explicit {'positive' if positive else 'negative'} feedback "
                f"was safely rejected ({reason}); LTM memory is unchanged."
            )
            emit(
                "feedback_complete",
                {
                    "status": "rejected",
                    "reason": reason,
                    "loss_before": result.get("loss_before"),
                },
            )
            return

        persisted = bool(result.get("persisted", False))
        persisted_path = result.get("path")
        persistence_error = result.get("persistence_error")

        delta_norm = float(result.get("delta_norm", 0.0) or 0.0)
        loss_before = result.get("loss_before")
        loss_after = result.get("loss_after")
        polarity = "Positive" if positive else "Negative"
        emit_status(
            f"{polarity} feedback committed to fast LTM memory "
            f"(delta norm {delta_norm:.3e}, objective "
            f"{float(loss_before):.6f} -> {float(loss_after):.6f})."
        )
        if persisted:
            emit_status(f"Online LTM overlay autosaved to {persisted_path}.")
        elif persistence_error:
            emit_error(
                "Feedback was committed in memory but could not be persisted: "
                f"{persistence_error}",
                operation="feedback",
            )
        emit(
            "feedback_complete",
            {
                "status": "accepted",
                "positive": positive,
                "delta_norm": delta_norm,
                "fast_norm": result.get("fast_norm"),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "persisted": persisted,
                "path": persisted_path,
            },
        )
    except Exception as exc:
        emit_error(
            f"Feedback update failed: {exc}\n{traceback.format_exc()}",
            operation="feedback",
        )
        emit(
            "feedback_complete",
            {"status": "error", "reason": str(exc)},
        )


@_exclusive_operation("command")
def handle_execute_command(params: dict):
    global _model, _device, _ltm_state, _pending_feedback, _ltm_token_clock
    command = params.get("command", "").strip()

    if command == "/reset":
        _reset_runtime_state()
        emit_status("RNN and hierarchical states reset. LTM memory was left unchanged.")
    elif command == "/reset_ltm":
        if _model is not None and hasattr(_model, 'ltm'):
            if hasattr(_model.ltm, 'reset_working_memory'):
                _model.ltm.reset_working_memory()
            elif hasattr(_model.ltm, 'reset_memory'):
                _model.ltm.reset_memory()
        _ltm_state = None
        _pending_feedback = None
        _ltm_token_clock = 0
        emit_status("LTM working memory cleared.")
    elif command == "/status":
        if _model is not None:
            total = sum(p.numel() for p in _model.parameters())
            emit_status(f"Model: active | Device: {_device} | Params: {total/1e6:.1f}M")
        else:
            emit_status("No model loaded.")
    else:
        emit_status(f"Unknown command: {command}")


def handle_ping(_params):
    emit("pong", {})


@_exclusive_operation("thread_configuration")
def handle_set_threads(params: dict):
    threads = _apply_thread_count(params.get("threads", params.get("cpu_threads")))
    emit("threads_set", {"threads": threads})
    emit_status(f"CPU chat threads set to {threads}.")


# ── Dispatch ─────────────────────────────────────────────────────────────────

HANDLERS = {
    "load_model": handle_load_model,
    "generate": handle_generate,
    "start_training": handle_start_training,
    "stop_generation": handle_stop_generation,
    "stop_training": handle_stop_training,
    "get_model_info": handle_get_model_info,
    "get_ltm_snapshot": handle_get_ltm_snapshot,
    "save_ltm_updates": handle_save_ltm_updates,
    "save_chat_runtime_state": handle_save_chat_runtime_state,
    "load_chat_runtime_state": handle_load_chat_runtime_state,
    "reset_chat_runtime_state": handle_reset_chat_runtime_state,
    "send_feedback": handle_send_feedback,
    "execute_command": handle_execute_command,
    "set_threads": handle_set_threads,
    "ping": handle_ping,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # The packaged backend also serves as the compatibility runtime for the
    # native Rust CLI.  Keep this path completely separate from the line-based
    # GUI bridge protocol: HierarchosCLI.exe launches us with ``--cli`` and the
    # remaining argv is handed verbatim to the canonical Python CLI parser.
    # PyInstaller carries ``hierarchos_cli`` as a hidden import so this works in
    # a release without a system Python installation.
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from hierarchos_cli import main as cli_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        cli_main()
        return

    emit_status("Hierarchos bridge server started.")
    _emit_backend_runtime_info()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit_error(f"Invalid JSON: {line[:100]}")
            continue

        method = request.get("method", "")
        params = request.get("params", {})

        handler = HANDLERS.get(method)
        if handler:
            try:
                handler(params)
            except Exception as e:
                emit_error(f"Handler error [{method}]: {e}\n{traceback.format_exc()}")
        else:
            emit_error(f"Unknown method: {method}")

    emit_status("Bridge server shutting down.")


if __name__ == "__main__":
    main()
