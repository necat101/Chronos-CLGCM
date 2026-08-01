"""Strict, atomic LoRA-to-full-checkpoint export for modular Hierarchos.

The historical monolith exposed a ``merge-lora`` command, but the modular CLI
never dispatched to it.  This module deliberately supports the adapter format
produced by :func:`hierarchos.training.trainer.finetune`: a local PEFT LoRA
adapter saved as safetensors, with an optional ``ltm`` module-to-save.

Merging is performed on CPU.  The result is reconstructed through a fresh
HierarchosCore before publication, then written into a staging directory and
renamed into place as one package.  No optimizer state or transient LTM working
memory is carried into the inference checkpoint.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from typing import Any, Dict, Mapping, Tuple

import torch

from ..models.revisions import (
    architecture_contract,
    architecture_contract_hash,
)
from .checkpoint import (
    AttrDict,
    TRANSIENT_LTM_STATE_KEYS,
    _reject_unsupported_rwkv_state_dict,
    _resolve_weights_path,
    _validate_run_identity_digest,
    _validate_state_dict_finite,
    _validate_tied_embedding_state_dict,
    load_full_model_with_config,
    load_model_state_dict_compatible,
    sanitize_model_state_dict,
    save_checkpoint_safely,
)
from .tokenizer import (
    tokenizer_identity,
    tokenizer_vocab_size,
    validate_inference_tokenizer_identity,
)


MERGE_PROVENANCE_VERSION = 1
ADAPTER_MANIFEST_VERSION = 1
ADAPTER_MANIFEST_FORMAT = "hierarchos-peft-lora-v1"
ADAPTER_MANIFEST_NAME = "hierarchos_adapter_manifest.json"
_ADAPTER_CONFIG_NAME = "adapter_config.json"
_SAFE_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
_UNSAFE_ADAPTER_WEIGHT_NAMES = (
    "adapter_model.bin",
    "adapter_model.pt",
    "pytorch_model.bin",
)
_SUPPORTED_MODULES_TO_SAVE = frozenset({"ltm"})
_PEFT_STATE_PREFIX = "base_model.model."


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(8 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_lora_geometry(config_or_object) -> Dict[str, Any]:
    if hasattr(config_or_object, "to_dict"):
        source = config_or_object.to_dict()
    elif isinstance(config_or_object, Mapping):
        source = dict(config_or_object)
    else:
        raise ValueError("LoRA config must be a mapping or expose to_dict().")

    def _sorted_strings(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return sorted(str(item) for item in value)

    def _normalized_mapping(value):
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): value[key]
            for key in sorted(value, key=lambda item: str(item))
        }

    def _enum_string(value):
        value = getattr(value, "value", value)
        return str(value or "").upper()

    try:
        rank = int(source.get("r", 0))
        alpha = float(source.get("lora_alpha", 0.0))
        dropout = float(source.get("lora_dropout", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LoRA geometry contains invalid numeric values.") from exc
    return {
        "peft_type": _enum_string(source.get("peft_type")),
        "task_type": _enum_string(source.get("task_type")),
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": _sorted_strings(source.get("target_modules")),
        "modules_to_save": _sorted_strings(source.get("modules_to_save")),
        "rank_pattern": _normalized_mapping(source.get("rank_pattern")),
        "alpha_pattern": _normalized_mapping(source.get("alpha_pattern")),
        "bias": str(source.get("bias") or "none").lower(),
        "fan_in_fan_out": bool(source.get("fan_in_fan_out", False)),
        "use_rslora": bool(source.get("use_rslora", False)),
        "use_dora": bool(source.get("use_dora", False)),
    }


def _write_text_file_atomically(path: str, text: str) -> None:
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as destination:
            destination.write(text)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def write_hierarchos_adapter_manifest(
    adapter_dir: str,
    *,
    base_model_path: str,
    model_config,
    tokenizer,
    lora_config,
    finetune_run_identity: Mapping[str, Any],
) -> str:
    """Bind a saved PEFT adapter to the exact base/function/token language.

    Call this only after ``PeftModel.save_pretrained`` and tokenizer saving have
    completed.  The manifest and its sidecar are written atomically.
    """

    adapter_dir = os.path.realpath(
        os.path.abspath(os.path.expanduser(adapter_dir))
    )
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"Cannot write adapter manifest; directory not found: {adapter_dir}"
        )
    weights_path, _ = _resolve_weights_path(base_model_path)
    adapter_config_path = os.path.join(adapter_dir, _ADAPTER_CONFIG_NAME)
    adapter_weights_path = os.path.join(
        adapter_dir,
        _SAFE_ADAPTER_WEIGHTS_NAME,
    )
    if not os.path.isfile(adapter_config_path) or not os.path.isfile(
        adapter_weights_path
    ):
        raise FileNotFoundError(
            "Cannot bind adapter before adapter_config.json and "
            "adapter_model.safetensors have both been saved."
        )
    if not isinstance(finetune_run_identity, Mapping):
        raise ValueError(
            "A complete finetune run identity is required for a bound adapter."
        )
    run_identity_copy = copy.deepcopy(dict(finetune_run_identity))
    if not _validate_run_identity_digest(
        {"run_identity": run_identity_copy},
        "LoRA adapter manifest",
    ):
        raise ValueError(
            "Finetune run identity must contain a valid self SHA-256 digest."
        )

    config_dict = dict(model_config)
    contract = architecture_contract(config_dict)
    contract_hash = architecture_contract_hash(config_dict)
    identity = tokenizer_identity(tokenizer)
    manifest = {
        "manifest_version": ADAPTER_MANIFEST_VERSION,
        "format": ADAPTER_MANIFEST_FORMAT,
        "base_checkpoint": {
            "filename": os.path.basename(weights_path),
            "sha256": _sha256_file(weights_path),
        },
        "architecture_contract": contract,
        "architecture_contract_sha256": contract_hash,
        "tokenizer_identity": identity,
        "lora_geometry": _canonical_lora_geometry(lora_config),
        "adapter_files": {
            _ADAPTER_CONFIG_NAME: _sha256_file(adapter_config_path),
            _SAFE_ADAPTER_WEIGHTS_NAME: _sha256_file(adapter_weights_path),
        },
        "finetune_run_identity": run_identity_copy,
    }
    manifest_path = os.path.join(adapter_dir, ADAPTER_MANIFEST_NAME)
    serialized = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )
    _write_text_file_atomically(manifest_path, serialized)
    digest = _sha256_file(manifest_path)
    _write_text_file_atomically(
        manifest_path + ".sha256",
        f"{digest}  {ADAPTER_MANIFEST_NAME}\n",
    )
    return manifest_path


def _load_bound_adapter_manifest(adapter_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(adapter_dir, ADAPTER_MANIFEST_NAME)
    checksum_path = manifest_path + ".sha256"
    if not os.path.isfile(manifest_path):
        raise ValueError(
            f"Bound Hierarchos adapter manifest is required: {manifest_path}. "
            "Unbound legacy adapters can belong to different base weights even "
            "when every tensor shape matches, so they are not mergeable safely."
        )
    if not os.path.isfile(checksum_path):
        raise ValueError(
            f"Adapter manifest checksum is required: {checksum_path}"
        )
    with open(checksum_path, "r", encoding="utf-8") as source:
        expected_digest = source.read().strip().split()[0].lower()
    actual_digest = _sha256_file(manifest_path)
    if (
        len(expected_digest) != 64
        or any(char not in "0123456789abcdef" for char in expected_digest)
        or actual_digest != expected_digest
    ):
        raise ValueError(
            "Adapter manifest SHA-256 verification failed; refusing to merge."
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid adapter manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Adapter manifest must contain a JSON object.")
    if int(manifest.get("manifest_version", 0) or 0) != ADAPTER_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported adapter manifest version "
            f"{manifest.get('manifest_version')!r}."
        )
    if manifest.get("format") != ADAPTER_MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported adapter manifest format {manifest.get('format')!r}."
        )
    return manifest


def _validate_bound_adapter_manifest(
    manifest: Mapping[str, Any],
    *,
    weights_path: str,
    base_config: Mapping[str, Any],
    tokenizer,
    adapter_dir: str,
    adapter_config: Mapping[str, Any],
    adapter_weights_path: str,
    base_checkpoint_sha256: str,
) -> Dict[str, Any]:
    base_binding = manifest.get("base_checkpoint")
    if not isinstance(base_binding, Mapping):
        raise ValueError("Adapter manifest has no base_checkpoint binding.")
    if str(base_binding.get("sha256", "")).lower() != base_checkpoint_sha256:
        raise ValueError(
            "LoRA adapter was not trained from this exact base checkpoint "
            "(base SHA-256 mismatch)."
        )
    if base_binding.get("filename") != os.path.basename(weights_path):
        raise ValueError(
            "LoRA adapter base checkpoint filename disagrees with the selected base."
        )

    expected_contract = architecture_contract(base_config)
    expected_contract_hash = architecture_contract_hash(base_config)
    if manifest.get("architecture_contract") != expected_contract:
        raise ValueError(
            "LoRA adapter architecture contract differs from the selected base."
        )
    if (
        str(manifest.get("architecture_contract_sha256", "")).lower()
        != expected_contract_hash
    ):
        raise ValueError(
            "LoRA adapter architecture contract hash differs from the selected base."
        )

    manifest_tokenizer = manifest.get("tokenizer_identity")
    if not isinstance(manifest_tokenizer, Mapping):
        raise ValueError("LoRA adapter manifest has no tokenizer identity.")
    _assert_same_tokenizer_content(
        manifest_tokenizer,
        tokenizer_identity(tokenizer),
    )
    if _canonical_lora_geometry(adapter_config) != manifest.get(
        "lora_geometry"
    ):
        raise ValueError(
            "LoRA adapter geometry/config differs from its bound manifest."
        )

    adapter_files = manifest.get("adapter_files")
    if not isinstance(adapter_files, Mapping):
        raise ValueError("LoRA adapter manifest has no adapter file hashes.")
    actual_adapter_hashes = {
        _ADAPTER_CONFIG_NAME: _sha256_file(
            os.path.join(adapter_dir, _ADAPTER_CONFIG_NAME)
        ),
        _SAFE_ADAPTER_WEIGHTS_NAME: _sha256_file(adapter_weights_path),
    }
    for filename, actual_digest in actual_adapter_hashes.items():
        if str(adapter_files.get(filename, "")).lower() != actual_digest:
            raise ValueError(
                f"LoRA adapter file hash mismatch for {filename}; refusing to merge."
            )

    finetune_identity = manifest.get("finetune_run_identity")
    if not isinstance(finetune_identity, dict):
        raise ValueError("LoRA adapter manifest has no finetune run identity.")
    if not _validate_run_identity_digest(
        {"run_identity": finetune_identity},
        "LoRA adapter manifest",
    ):
        raise ValueError(
            "LoRA adapter finetune run identity has no authenticated digest."
        )
    finetune_contract = finetune_identity.get("architecture_contract")
    finetune_contract_hash = finetune_identity.get(
        "architecture_contract_sha256"
    )
    if not isinstance(finetune_contract, dict) or not isinstance(
        finetune_contract_hash, str
    ):
        raise ValueError(
            "LoRA adapter finetune identity has no effective architecture contract."
        )
    if architecture_contract(finetune_contract) != finetune_contract:
        raise ValueError(
            "LoRA adapter finetune architecture contract is not canonical."
        )
    if (
        architecture_contract_hash(finetune_contract)
        != finetune_contract_hash.strip().lower()
    ):
        raise ValueError(
            "LoRA adapter finetune architecture contract hash is invalid."
        )
    finetune_tokenizer = finetune_identity.get("tokenizer")
    if not isinstance(finetune_tokenizer, Mapping):
        raise ValueError(
            "LoRA adapter finetune identity has no tokenizer fingerprint."
        )
    _assert_same_tokenizer_content(manifest_tokenizer, finetune_tokenizer)

    objective = finetune_identity.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("LoRA adapter finetune identity has no objective metadata.")
    manifest_geometry = manifest.get("lora_geometry")
    identity_geometry = {
        "r": objective.get("lora_r"),
        "lora_alpha": objective.get("lora_alpha"),
        "lora_dropout": objective.get("lora_dropout"),
    }
    for geometry_key, identity_value in identity_geometry.items():
        if identity_value is None:
            raise ValueError(
                f"LoRA adapter finetune identity is missing {geometry_key!r}."
            )
        manifest_value = manifest_geometry.get(geometry_key)
        if float(identity_value) != float(manifest_value):
            raise ValueError(
                f"LoRA adapter {geometry_key} disagrees with its finetune identity."
            )

    # The adapter-local tokenizer is the most direct proof of what finetuning
    # actually consumed.  Reload it without remote code and compare content if
    # the adapter package contains tokenizer assets.
    tokenizer_markers = ("tokenizer.json", "tokenizer_config.json")
    if any(
        os.path.isfile(os.path.join(adapter_dir, marker))
        for marker in tokenizer_markers
    ):
        adapter_tokenizer = _reload_saved_tokenizer(tokenizer, adapter_dir)
        _assert_same_tokenizer_content(
            manifest_tokenizer,
            tokenizer_identity(adapter_tokenizer),
        )
    return finetune_identity


def _load_peft_api():
    try:
        from peft import PeftModel
        import peft
    except ImportError as exc:
        raise ImportError(
            "LoRA merging requires the optional 'peft' package. "
            "Install it with: pip install peft safetensors"
        ) from exc
    return PeftModel, str(getattr(peft, "__version__", "unknown"))


def _read_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(adapter_dir, _ADAPTER_CONFIG_NAME)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"PEFT adapter config not found: {config_path}"
        )
    try:
        with open(config_path, "r", encoding="utf-8") as source:
            config = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read a valid PEFT adapter config from {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("PEFT adapter_config.json must contain a JSON object.")

    peft_type = str(config.get("peft_type") or "").strip().upper()
    if peft_type != "LORA":
        raise ValueError(
            f"Unsupported PEFT adapter type {peft_type or '<missing>'!r}; "
            "merge-lora accepts LoRA adapters only."
        )
    task_type = str(config.get("task_type") or "").strip().upper()
    if task_type != "CAUSAL_LM":
        raise ValueError(
            f"Unsupported PEFT task type {task_type or '<missing>'!r}; "
            "Hierarchos LoRA adapters must use CAUSAL_LM."
        )
    if str(config.get("bias") or "none").strip().lower() != "none":
        raise ValueError(
            "This merge path accepts the project's bias='none' LoRA format only."
        )

    modules_to_save = config.get("modules_to_save") or []
    if not isinstance(modules_to_save, (list, tuple, set)):
        raise ValueError("PEFT modules_to_save must be a list when present.")
    normalized_modules = {str(name) for name in modules_to_save}
    unsupported_modules = normalized_modules - _SUPPORTED_MODULES_TO_SAVE
    if unsupported_modules:
        raise ValueError(
            "Unsupported PEFT modules_to_save entries: "
            + ", ".join(sorted(unsupported_modules))
            + ". The modular Hierarchos finetuner saves only the optional 'ltm' module."
        )

    unsupported_features = {
        "use_dora": bool(config.get("use_dora", False)),
        "use_qalora": bool(config.get("use_qalora", False)),
        "lora_bias": bool(config.get("lora_bias", False)),
        "target_parameters": bool(config.get("target_parameters")),
        "trainable_token_indices": bool(config.get("trainable_token_indices")),
        "alora_invocation_tokens": bool(config.get("alora_invocation_tokens")),
        "arrow_config": bool(config.get("arrow_config")),
        "corda_config": bool(config.get("corda_config")),
    }
    enabled_unsupported = [
        name for name, enabled in unsupported_features.items() if enabled
    ]
    if enabled_unsupported:
        raise ValueError(
            "Unsupported nonstandard LoRA feature(s): "
            + ", ".join(enabled_unsupported)
            + ". Re-export a standard Hierarchos PEFT LoRA adapter."
        )

    target_modules = config.get("target_modules")
    if not target_modules:
        raise ValueError("PEFT LoRA adapter has no target_modules.")
    try:
        rank = int(config.get("r", 0))
        alpha = float(config.get("lora_alpha", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("PEFT LoRA rank/alpha metadata is invalid.") from exc
    if rank <= 0 or not torch.isfinite(torch.tensor(alpha)) or alpha <= 0.0:
        raise ValueError("PEFT LoRA rank and alpha must both be finite and positive.")
    return config


def _load_and_validate_adapter_tensors(
    adapter_dir: str,
    adapter_config: Mapping[str, Any],
) -> Tuple[Dict[str, torch.Tensor], str]:
    for unsafe_name in _UNSAFE_ADAPTER_WEIGHT_NAMES:
        unsafe_path = os.path.join(adapter_dir, unsafe_name)
        if os.path.exists(unsafe_path):
            raise ValueError(
                f"Unsafe pickle-based adapter weights are not supported: {unsafe_path}. "
                f"Re-save the adapter as {_SAFE_ADAPTER_WEIGHTS_NAME}."
            )

    weights_path = os.path.join(adapter_dir, _SAFE_ADAPTER_WEIGHTS_NAME)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Safe PEFT adapter weights not found: {weights_path}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Safe LoRA merging requires 'safetensors'. Install it with: "
            "pip install safetensors"
        ) from exc

    try:
        tensors = load_file(weights_path, device="cpu")
    except Exception as exc:
        raise ValueError(
            f"Could not load safe adapter weights from {weights_path}: {exc}"
        ) from exc
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("PEFT adapter safetensors file is empty.")

    lora_a_prefixes = set()
    lora_b_prefixes = set()
    modules_to_save = {
        str(name) for name in (adapter_config.get("modules_to_save") or [])
    }
    for key, value in tensors.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("PEFT adapter contains a non-tensor or invalid key.")
        if not key.startswith(_PEFT_STATE_PREFIX):
            raise ValueError(
                f"Unsupported adapter tensor key {key!r}; expected modular "
                f"Hierarchos keys under {_PEFT_STATE_PREFIX!r}."
            )
        if value.numel() == 0:
            raise ValueError(f"PEFT adapter tensor {key!r} is empty.")
        if value.is_floating_point() or value.is_complex():
            flat = value.detach().reshape(-1)
            for start in range(0, flat.numel(), 1_048_576):
                if not bool(
                    torch.isfinite(flat[start : start + 1_048_576]).all().item()
                ):
                    raise ValueError(
                        f"Non-finite tensor {key!r} in PEFT adapter. "
                        "Refusing to merge NaN/Inf weights."
                    )

        if key.endswith(".lora_A.weight"):
            lora_a_prefixes.add(key[: -len(".lora_A.weight")])
        elif key.endswith(".lora_B.weight"):
            lora_b_prefixes.add(key[: -len(".lora_B.weight")])
        elif (
            "ltm" in modules_to_save
            and key.startswith(f"{_PEFT_STATE_PREFIX}ltm.")
        ):
            pass
        else:
            raise ValueError(
                f"Unsupported tensor {key!r} in adapter. Expected standard "
                "LoRA A/B matrices and, optionally, the saved LTM module."
            )

    if not lora_a_prefixes or lora_a_prefixes != lora_b_prefixes:
        missing_b = sorted(lora_a_prefixes - lora_b_prefixes)
        missing_a = sorted(lora_b_prefixes - lora_a_prefixes)
        raise ValueError(
            "PEFT adapter has incomplete LoRA A/B matrix pairs"
            + (f"; missing B for {missing_b[:4]}" if missing_b else "")
            + (f"; missing A for {missing_a[:4]}" if missing_a else "")
            + "."
        )
    return tensors, weights_path


def _loaded_adapter_key(
    saved_key: str,
    adapter_config: Mapping[str, Any],
) -> str:
    if saved_key.endswith(".lora_A.weight"):
        return saved_key[: -len(".weight")] + ".default.weight"
    if saved_key.endswith(".lora_B.weight"):
        return saved_key[: -len(".weight")] + ".default.weight"
    modules_to_save = {
        str(name) for name in (adapter_config.get("modules_to_save") or [])
    }
    if (
        "ltm" in modules_to_save
        and saved_key.startswith(f"{_PEFT_STATE_PREFIX}ltm.")
    ):
        suffix = saved_key[len(f"{_PEFT_STATE_PREFIX}ltm.") :]
        return (
            f"{_PEFT_STATE_PREFIX}ltm.modules_to_save.default.{suffix}"
        )
    raise ValueError(f"Unsupported saved adapter key: {saved_key!r}")


def _validate_adapter_loaded_exactly(
    peft_model,
    adapter_tensors: Mapping[str, torch.Tensor],
    adapter_config: Mapping[str, Any],
) -> None:
    loaded_state = peft_model.state_dict()
    expected_keys = {
        _loaded_adapter_key(key, adapter_config)
        for key in adapter_tensors
    }
    actual_keys = {
        key
        for key in loaded_state
        if ".lora_A." in key
        or ".lora_B." in key
        or ".modules_to_save.default." in key
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "PEFT adapter did not load exactly"
            + (f"; missing keys={missing[:8]}" if missing else "")
            + (f"; unexpected keys={unexpected[:8]}" if unexpected else "")
            + "."
        )

    for saved_key, saved_value in adapter_tensors.items():
        loaded_key = _loaded_adapter_key(saved_key, adapter_config)
        loaded_value = loaded_state[loaded_key].detach().cpu()
        expected_value = saved_value.detach().to(dtype=loaded_value.dtype)
        if (
            tuple(loaded_value.shape) != tuple(expected_value.shape)
            or not torch.equal(loaded_value, expected_value)
        ):
            raise ValueError(
                f"PEFT adapter tensor {saved_key!r} was not restored exactly "
                f"into {loaded_key!r}."
            )


def _model_state_signature(model) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    state = sanitize_model_state_dict(model, reset_transient_ltm=False)
    return {
        key: (tuple(value.shape), value.dtype)
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def _assert_tied_embedding_parameters(model, source: str) -> None:
    token_weight = getattr(getattr(model, "tok_emb", None), "weight", None)
    head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    if not torch.is_tensor(token_weight) or not torch.is_tensor(head_weight):
        raise ValueError(f"Missing tied embedding/head weights in {source}.")
    if tuple(token_weight.shape) != tuple(head_weight.shape):
        raise ValueError(f"Tied embedding/head geometry differs in {source}.")
    if token_weight is not head_weight:
        try:
            shares_storage = (
                token_weight.untyped_storage().data_ptr()
                == head_weight.untyped_storage().data_ptr()
            )
        except (AttributeError, RuntimeError):
            shares_storage = False
        if not shares_storage:
            raise ValueError(
                f"Token embedding and language-model head are not tied in {source}."
            )


def _assert_transient_ltm_is_zero(
    state_dict: Mapping[str, torch.Tensor],
    source: str,
) -> None:
    for key, value in state_dict.items():
        if not any(str(key).endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS):
            continue
        if not torch.is_tensor(value) or bool(torch.count_nonzero(value).item()):
            raise ValueError(
                f"Transient LTM tensor {key!r} was not reset in {source}."
            )


def _tokenizer_source_name(tokenizer, config: Mapping[str, Any]) -> str:
    return str(
        config.get("tokenizer_name")
        or config.get("tokenizer_path")
        or getattr(tokenizer, "name_or_path", None)
        or type(tokenizer).__name__
    )


def _align_tokenizer_runtime_length(tokenizer, config: Mapping[str, Any]) -> None:
    max_length = config.get("max_length")
    if max_length is None:
        return
    try:
        max_length = int(max_length)
    except (TypeError, ValueError, OverflowError):
        return
    if max_length <= 0:
        return
    try:
        tokenizer.model_max_length = max_length
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            init_kwargs["model_max_length"] = max_length
    except Exception:
        pass


def _reload_saved_tokenizer(tokenizer, directory: str):
    tokenizer_class = type(tokenizer)
    loader = getattr(tokenizer_class, "from_pretrained", None)
    if not callable(loader):
        raise ValueError(
            f"Tokenizer class {tokenizer_class.__name__} cannot be reloaded "
            "to verify the merged package."
        )
    try:
        return loader(directory, local_files_only=True)
    except Exception as exc:
        raise ValueError(
            f"Saved tokenizer could not be reloaded locally from {directory}: {exc}"
        ) from exc


def _assert_same_tokenizer_content(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    if int(expected.get("vocab_size", -1)) != int(actual.get("vocab_size", -2)):
        raise ValueError("Saved tokenizer vocabulary size changed during export.")
    if str(expected.get("sha256", "")).lower() != str(
        actual.get("sha256", "")
    ).lower():
        raise ValueError("Saved tokenizer content fingerprint changed during export.")


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True, default=str)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def _path_contains(container: str, item: str) -> bool:
    try:
        return os.path.commonpath((container, item)) == container
    except ValueError:
        return False


def _validate_merge_paths(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    *,
    overwrite: bool,
) -> Tuple[str, str, str, str]:
    if not base_model_path:
        raise ValueError("merge-lora requires a base --model-path.")
    if not adapter_path:
        raise ValueError("merge-lora requires --lora-adapter-path.")
    if not output_dir:
        raise ValueError("merge-lora requires a non-empty --out-dir.")

    base_source = os.path.realpath(
        os.path.abspath(os.path.expanduser(base_model_path))
    )
    base_source_is_directory = os.path.isdir(base_source)
    weights_path, base_dir = _resolve_weights_path(base_model_path)
    adapter_dir = os.path.realpath(os.path.abspath(os.path.expanduser(adapter_path)))
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")

    output_abs = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.basename(os.path.normpath(output_abs)):
        raise ValueError("Refusing to use a filesystem root as merge output.")
    output_real = os.path.realpath(output_abs)
    weights_real = os.path.realpath(weights_path)
    adapter_real = os.path.realpath(adapter_dir)
    if (
        output_real == adapter_real
        or _path_contains(output_real, adapter_real)
        or _path_contains(adapter_real, output_real)
    ):
        raise ValueError(
            "Merge output and source LoRA adapter cannot overlap."
        )
    base_dir_real = os.path.realpath(base_dir)
    overlaps_base_package = _path_contains(output_real, weights_real)
    if base_source_is_directory:
        overlaps_base_package = (
            overlaps_base_package
            or output_real == base_dir_real
            or _path_contains(output_real, base_dir_real)
            or _path_contains(base_dir_real, output_real)
        )
    if overlaps_base_package:
        raise ValueError(
            "Merge output and source base-model package cannot overlap."
        )
    if os.path.lexists(output_abs) and not overwrite:
        raise FileExistsError(
            f"Merge output already exists: {output_abs}. "
            "Pass --overwrite-merge-output to replace it atomically."
        )
    return weights_path, base_dir, adapter_dir, output_abs


def _publish_directory_atomically(
    staging_dir: str,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    backup_path = None
    published = False
    try:
        if os.path.lexists(output_dir):
            if not overwrite:
                raise FileExistsError(f"Merge output already exists: {output_dir}")
            backup_path = output_dir + ".pre-merge-backup"
            if os.path.lexists(backup_path):
                raise FileExistsError(
                    f"Cannot overwrite merge output because backup path exists: "
                    f"{backup_path}"
                )
            os.replace(output_dir, backup_path)
        os.replace(staging_dir, output_dir)
        published = True
    except Exception:
        if (
            backup_path is not None
            and not os.path.lexists(output_dir)
            and os.path.lexists(backup_path)
        ):
            os.replace(backup_path, output_dir)
        raise
    finally:
        if published and backup_path is not None and os.path.lexists(backup_path):
            try:
                if os.path.isdir(backup_path) and not os.path.islink(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
            except OSError as exc:
                print(
                    "WARNING: Merged package is valid, but the previous output "
                    f"backup could not be removed: {backup_path} ({exc})"
                )


def merge_lora_adapter(
    *,
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    tokenizer,
    overwrite: bool = False,
) -> str:
    """Merge one local Hierarchos PEFT LoRA adapter into a full checkpoint.

    Returns the absolute output directory.  All expensive work happens before
    the staged package is atomically published.
    """

    if tokenizer is None:
        raise ValueError("merge-lora requires the exact base-model tokenizer.")
    weights_path, _, adapter_dir, output_abs = _validate_merge_paths(
        base_model_path,
        adapter_path,
        output_dir,
        overwrite=bool(overwrite),
    )
    adapter_config = _read_adapter_config(adapter_dir)
    adapter_tensors, adapter_weights_path = _load_and_validate_adapter_tensors(
        adapter_dir,
        adapter_config,
    )
    adapter_manifest = _load_bound_adapter_manifest(adapter_dir)
    PeftModel, peft_version = _load_peft_api()
    base_digest = _sha256_file(weights_path)

    print(f"INFO: Loading full-precision base checkpoint on CPU: {weights_path}")
    base_model, loaded_config = load_full_model_with_config(
        base_model_path,
        torch.device("cpu"),
    )
    from ..models.core import HierarchosCore

    if not isinstance(base_model, HierarchosCore):
        raise ValueError(
            "Base checkpoint did not load as a modular HierarchosCore model."
        )
    base_config = dict(loaded_config)
    base_contract = architecture_contract(base_config)
    base_contract_hash = architecture_contract_hash(base_config)
    base_signature = _model_state_signature(base_model)
    _assert_tied_embedding_parameters(base_model, "base checkpoint")

    metadata = copy.deepcopy(
        getattr(base_model, "_hierarchos_checkpoint_metadata", {}) or {}
    )
    model_vocab_size = int(base_config.get("vocab_size", 0) or 0)
    actual_vocab_size = tokenizer_vocab_size(tokenizer)
    if model_vocab_size <= 0 or actual_vocab_size != model_vocab_size:
        raise ValueError(
            "Tokenizer/base vocabulary mismatch: "
            f"tokenizer={actual_vocab_size}, checkpoint={model_vocab_size}."
        )
    if validate_inference_tokenizer_identity(tokenizer, metadata):
        print("INFO: Base tokenizer content fingerprint verified.")
    else:
        print(
            "WARNING: Legacy base checkpoint has no tokenizer fingerprint; "
            "the merged package will record and enforce the supplied tokenizer."
        )
    _align_tokenizer_runtime_length(tokenizer, base_config)
    merged_tokenizer_identity = tokenizer_identity(tokenizer)
    finetune_run_identity = _validate_bound_adapter_manifest(
        adapter_manifest,
        weights_path=weights_path,
        base_config=base_config,
        tokenizer=tokenizer,
        adapter_dir=adapter_dir,
        adapter_config=adapter_config,
        adapter_weights_path=adapter_weights_path,
        base_checkpoint_sha256=base_digest,
    )

    print(f"INFO: Loading verified PEFT adapter: {adapter_dir}")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        peft_model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            is_trainable=False,
            autocast_adapter_dtype=False,
            local_files_only=True,
        )
    missing_adapter_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if "missing adapter keys" in str(warning.message).lower()
    ]
    if missing_adapter_warnings:
        raise ValueError(
            "PEFT reported an incomplete adapter load: "
            + " | ".join(missing_adapter_warnings)
        )
    _validate_adapter_loaded_exactly(
        peft_model,
        adapter_tensors,
        adapter_config,
    )
    del adapter_tensors

    print("INFO: Merging LoRA weights with PEFT safe_merge=True.")
    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError as exc:
        raise RuntimeError(
            "Installed PEFT does not support safe_merge=True. Upgrade PEFT "
            "instead of performing an unchecked merge."
        ) from exc
    if not isinstance(merged_model, HierarchosCore):
        raise ValueError(
            "PEFT merge did not return the original modular HierarchosCore."
        )
    if any(
        "lora_" in key
        or ".modules_to_save." in key
        or key.startswith("base_model.model.")
        for key in merged_model.state_dict()
    ):
        raise ValueError("PEFT adapter wrappers remain after merge_and_unload.")

    merged_config = dict(getattr(merged_model, "config", {}) or {})
    if architecture_contract(merged_config) != base_contract:
        raise ValueError(
            "PEFT merge changed the serialized Hierarchos architecture contract."
        )
    if architecture_contract_hash(merged_config) != base_contract_hash:
        raise ValueError("PEFT merge changed the architecture contract hash.")

    merged_signature = _model_state_signature(merged_model)
    if merged_signature != base_signature:
        missing = sorted(set(base_signature) - set(merged_signature))
        unexpected = sorted(set(merged_signature) - set(base_signature))
        changed = sorted(
            key
            for key in set(base_signature) & set(merged_signature)
            if base_signature[key] != merged_signature[key]
        )
        raise ValueError(
            "Merged model state geometry differs from the base architecture"
            + (f"; missing={missing[:8]}" if missing else "")
            + (f"; unexpected={unexpected[:8]}" if unexpected else "")
            + (f"; changed={changed[:8]}" if changed else "")
            + "."
        )
    for name, parameter in merged_model.named_parameters():
        if not (parameter.is_floating_point() or parameter.is_complex()):
            raise ValueError(
                f"Merged learned parameter {name!r} is not full precision."
            )
    _assert_tied_embedding_parameters(merged_model, "merged model")

    raw_merged_state = sanitize_model_state_dict(
        merged_model,
        reset_transient_ltm=False,
    )
    _reject_unsupported_rwkv_state_dict(raw_merged_state, "merged LoRA model")
    _validate_state_dict_finite(
        raw_merged_state,
        "merged LoRA model",
        allow_nonfinite_transient_ltm=False,
    )
    _validate_tied_embedding_state_dict(raw_merged_state, "merged LoRA model")

    clean_state = sanitize_model_state_dict(
        merged_model,
        reset_transient_ltm=True,
    )
    _validate_state_dict_finite(
        clean_state,
        "sanitized merged LoRA model",
        allow_nonfinite_transient_ltm=False,
    )
    _validate_tied_embedding_state_dict(
        clean_state,
        "sanitized merged LoRA model",
    )
    _assert_transient_ltm_is_zero(clean_state, "sanitized merged LoRA model")

    # Reconstruct through a fresh model before any output is published.  This
    # catches key/shape drift that PEFT's permissive state loading could hide.
    export_config = dict(base_config)
    effective_contract = finetune_run_identity["architecture_contract"]
    export_config.update(
        {
            key: value
            for key, value in effective_contract.items()
            if key != "architecture_contract_schema_version"
        }
    )
    finetune_objective = finetune_run_identity.get("objective") or {}
    try:
        effective_max_length = int(
            finetune_objective.get(
                "max_length",
                export_config.get("max_length", 0),
            )
            or 0
        )
    except (TypeError, ValueError, OverflowError):
        effective_max_length = 0
    if effective_max_length > 0:
        export_config["max_length"] = effective_max_length
    export_config["compile"] = False
    export_config["force_compile"] = False
    export_config["gradient_checkpointing"] = False
    tokenizer_name = _tokenizer_source_name(tokenizer, export_config)
    export_config["tokenizer_name"] = tokenizer_name
    export_config["tokenizer_identity_sha256"] = merged_tokenizer_identity[
        "sha256"
    ]
    export_contract = architecture_contract(export_config)
    export_contract_hash = architecture_contract_hash(export_config)
    if (
        export_contract != effective_contract
        or export_contract_hash
        != finetune_run_identity["architecture_contract_sha256"]
    ):
        raise ValueError(
            "Merged export config does not reproduce the effective finetune "
            "architecture contract."
        )
    export_config["architecture_contract_sha256"] = export_contract_hash

    verification_model = HierarchosCore(AttrDict(dict(export_config)))
    load_model_state_dict_compatible(
        verification_model,
        clean_state,
        "merged LoRA verification model",
    )
    _assert_tied_embedding_parameters(
        verification_model,
        "merged LoRA verification model",
    )
    verification_state = sanitize_model_state_dict(
        verification_model,
        reset_transient_ltm=False,
    )
    _validate_state_dict_finite(
        verification_state,
        "merged LoRA verification model",
        allow_nonfinite_transient_ltm=False,
    )
    _assert_transient_ltm_is_zero(
        verification_state,
        "merged LoRA verification model",
    )
    del verification_state, verification_model
    gc.collect()

    adapter_config_path = os.path.join(adapter_dir, _ADAPTER_CONFIG_NAME)
    merge_provenance = {
        "version": MERGE_PROVENANCE_VERSION,
        "merge_method": "peft.merge_and_unload(safe_merge=True)",
        "peft_version": peft_version,
        "base_checkpoint": os.path.basename(weights_path),
        "base_checkpoint_sha256": base_digest,
        "adapter_directory": os.path.basename(adapter_dir),
        "adapter_config_sha256": _sha256_file(adapter_config_path),
        "adapter_weights_sha256": _sha256_file(adapter_weights_path),
        "adapter_peft_type": "LORA",
        "adapter_task_type": "CAUSAL_LM",
        "base_architecture_contract_sha256": base_contract_hash,
        "architecture_contract_sha256": export_contract_hash,
        "tokenizer_sha256": merged_tokenizer_identity["sha256"],
        "transient_ltm_reset": True,
        "finetune_run_identity": copy.deepcopy(finetune_run_identity),
    }
    # Preserve source-run provenance as a nested historical record only.  A
    # merged adapter is not an exact continuation checkpoint: it has neither
    # the adapter optimizer/scheduler nor its data cursor, so copying those
    # fields to the active checkpoint namespace would misrepresent resumability.
    for source_key in (
        "expansion_provenance",
        "run_identity",
        "effective_training_config",
        "optimizer_grouping_version",
        "best_metric_state",
        "selection_metric",
    ):
        if source_key in metadata:
            merge_provenance[f"base_{source_key}"] = copy.deepcopy(
                metadata[source_key]
            )
    completed_epoch = metadata.get("completed_epoch", "unknown")
    try:
        source_checkpoint_version = int(
            metadata.get("checkpoint_version", 0) or 0
        )
    except (TypeError, ValueError, OverflowError):
        source_checkpoint_version = 0
    output_checkpoint = {
        "checkpoint_version": max(4, source_checkpoint_version),
        "checkpoint_kind": "inference",
        "model_state_dict": clean_state,
        "config": export_config,
        "architecture_contract": export_contract,
        "architecture_contract_sha256": export_contract_hash,
        "completed_epoch": completed_epoch,
        "training_complete": True,
        "converted_from": os.path.basename(weights_path),
        "merged_lora_adapter": os.path.basename(adapter_dir),
        "tokenizer_name": tokenizer_name,
        "tokenizer_identity": merged_tokenizer_identity,
        "merge_provenance": merge_provenance,
    }

    output_parent = os.path.dirname(output_abs)
    os.makedirs(output_parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(output_abs)}.merge-",
        dir=output_parent,
    )
    try:
        tokenizer.save_pretrained(staging_dir)
        reloaded_tokenizer = _reload_saved_tokenizer(tokenizer, staging_dir)
        reloaded_identity = tokenizer_identity(reloaded_tokenizer)
        _assert_same_tokenizer_content(
            merged_tokenizer_identity,
            reloaded_identity,
        )

        config_json = dict(export_config)
        config_json.update(
            {
                "completed_epoch": completed_epoch,
                "converted_from": os.path.basename(weights_path),
                "merged_lora_adapter": os.path.basename(adapter_dir),
                "merge_provenance": merge_provenance,
                "tokenizer_identity": merged_tokenizer_identity,
            }
        )
        _write_json(
            os.path.join(staging_dir, "hierarchos_config.json"),
            config_json,
        )
        checkpoint_path = os.path.join(staging_dir, "hierarchos.pt")
        save_checkpoint_safely(output_checkpoint, checkpoint_path)

        # Final package readback uses the same strict path as chat/benchmark.
        del clean_state, output_checkpoint, merged_model, peft_model, base_model
        gc.collect()
        reloaded_model, reloaded_config = load_full_model_with_config(
            staging_dir,
            torch.device("cpu"),
        )
        if architecture_contract_hash(dict(reloaded_config)) != export_contract_hash:
            raise ValueError(
                "Merged package architecture hash changed during serialization."
            )
        _assert_tied_embedding_parameters(
            reloaded_model,
            "serialized merged package",
        )
        if not validate_inference_tokenizer_identity(
            reloaded_tokenizer,
            getattr(reloaded_model, "_hierarchos_checkpoint_metadata", {}),
        ):
            raise ValueError(
                "Serialized merged package did not retain a tokenizer fingerprint."
            )
        serialized_state = sanitize_model_state_dict(
            reloaded_model,
            reset_transient_ltm=False,
        )
        _validate_state_dict_finite(
            serialized_state,
            "serialized merged package",
            allow_nonfinite_transient_ltm=False,
        )
        _assert_transient_ltm_is_zero(
            serialized_state,
            "serialized merged package",
        )
        del serialized_state, reloaded_model, reloaded_tokenizer
        gc.collect()

        _publish_directory_atomically(
            staging_dir,
            output_abs,
            overwrite=bool(overwrite),
        )
        staging_dir = ""
    finally:
        if staging_dir and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)

    print(f"INFO: Standalone merged model published atomically to: {output_abs}")
    return output_abs
