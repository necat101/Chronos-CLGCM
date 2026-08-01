import os
import sys
import argparse
import json
import hashlib
import shutil
import random
import math
import time
from array import array
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import signal
import traceback

from hierarchos.utils.checkpoint import (
    _infer_arch_flags_from_state_dict,
    _reject_unsupported_rwkv_state_dict,
    _resolve_weights_path,
    load_checkpoint_payload_compatible,
    save_checkpoint_safely,
    sanitize_model_state_dict,
    validate_checkpoint_architecture_contract,
)
from hierarchos.inference.chat_state import clear_ltm_working_memory
from hierarchos import (
    pick_device,
    set_threads,
    load_full_model_with_config,
    HierarchosCore,
    train,
    finetune,
    chat,
    is_directml_device,
    setup_msvc_environment,
    create_dataloader_for_jsonl,
    create_dataloader_for_hf_streaming,
    create_map_style_dataloader,
    HuggingFaceMapStyleDataset,
    OriginalJSONLDataset,
    process_text_sample,
    create_dataloader_for_chunked,
    create_dataloader_pt_chunked,
    create_dataloader_for_tokenized_cache,
    format_results,
    format_benchmark_catalog,
    resolve_task_names,
    run_post_training_benchmarks,
    write_benchmark_artifacts,
)
from hierarchos.utils.rosa import (
    precompute_rosa_ids_for_chunks,
    rosa_context_mode,
)
from hierarchos.utils.tokenizer import (
    tokenizer_identity as _tokenizer_identity,
    tokenizer_vocab_size as _tokenizer_vocab_size,
    validate_inference_tokenizer_identity,
)
from hierarchos.utils.lora_merge import merge_lora_adapter
from hierarchos.utils.safe_loading import load_tensor_payload_safely
from hierarchos.models.revisions import (
    COHERENT_REVISION,
    COHERENT_TRAINING_CHUNK_SIZE,
    apply_architecture_revision_defaults,
    architecture_contract,
    architecture_contract_hash,
)

HF_CACHE_FORMATTER_VERSION = "alpaca-previous-context-v6-min-response-filter"
_RETRY_CACHE_DIRECTORY_PERMISSION_ERRORS = os.name == "nt"
DEFAULT_TRAINING_CHUNK_SIZE = COHERENT_TRAINING_CHUNK_SIZE
DEFAULT_CHAT_TEMPERATURE = 0.7


def _replace_cache_directory_atomically(tmp_dir, cache_dir, *, attempts=5):
    """Publish a completed cache despite brief Windows handle contention."""
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            os.replace(tmp_dir, cache_dir)
            return
        except PermissionError:
            if (
                not _RETRY_CACHE_DIRECTORY_PERMISSION_ERRORS
                or attempt + 1 >= attempts
            ):
                raise
            # Antivirus/indexers can briefly retain a directory handle after
            # its final file closes. Keep the retry bounded and never mark the
            # temporary cache complete unless the atomic rename succeeds.
            time.sleep(0.05 * (2 ** attempt))


def set_global_training_seed(seed):
    """Seed every host/device RNG before tokenizer and loader construction."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"INFO: Global training seed set to {seed}.")
    return seed


def load_hf_dataset(
    dataset_name,
    dataset_config=None,
    split="train",
    streaming=False,
    revision=None,
    num_proc=None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Hugging Face dataset loading requires the 'datasets' package.") from exc
    kwargs = {"split": split, "streaming": bool(streaming)}
    if revision:
        kwargs["revision"] = revision
    if not streaming and int(num_proc or 0) > 0:
        kwargs["num_proc"] = int(num_proc)
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def resolve_hf_dataset_revision(args, *, announce=False):
    """Pin a Hub dataset to the commit used to key and build token caches.

    Resolving before ``load_dataset`` prevents a mutable ``main`` branch from
    silently reusing tokens from an older release. Explicit revisions still go
    through the Hub so a branch/tag is converted to an immutable commit SHA.
    Offline callers retain their explicit revision; callers without one receive
    a warning because no implementation can detect a remote update while offline.
    """
    existing = getattr(args, "_resolved_hf_dataset_revision", None)
    if existing:
        return existing
    dataset_name = getattr(args, "hf_dataset", None)
    requested = getattr(args, "hf_dataset_revision", None)
    if not dataset_name or os.path.exists(os.path.expanduser(str(dataset_name))):
        if requested:
            args._resolved_hf_dataset_revision = str(requested)
        return requested

    resolved = None
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id=dataset_name, revision=requested)
        resolved = getattr(info, "sha", None)
    except Exception as exc:
        resolved = requested
        if announce:
            if requested:
                print(
                    "WARNING: Could not resolve the HF dataset revision to a commit SHA; "
                    f"using explicit revision {requested!r}. {exc}"
                )
            else:
                print(
                    "WARNING: Could not resolve the mutable HF dataset revision. "
                    "Use --hf-dataset-revision with an immutable commit SHA or a fresh "
                    f"cache root before reusing a cache offline. {exc}"
                )

    if resolved:
        args._resolved_hf_dataset_revision = str(resolved)
        if announce:
            print(f"INFO: Pinned HF dataset revision: {resolved}")
    return resolved


def _hf_revision_for_load(args):
    return (
        getattr(args, "_resolved_hf_dataset_revision", None)
        or getattr(args, "hf_dataset_revision", None)
    )


def _parse_hf_split_count(base_count, selector):
    selector = selector.strip()
    if not selector:
        return base_count
    if ":" not in selector:
        return base_count

    start_text, end_text = selector.split(":", 1)
    start_text = start_text.strip()
    end_text = end_text.strip()
    percent_mode = start_text.endswith("%") or end_text.endswith("%")

    if percent_mode:
        start_pct = float(start_text[:-1]) if start_text.endswith("%") and start_text[:-1] else 0.0
        end_pct = float(end_text[:-1]) if end_text.endswith("%") and end_text[:-1] else 100.0
        start = int(base_count * (start_pct / 100.0))
        end = int(base_count * (end_pct / 100.0))
    else:
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else base_count

    start = max(0, min(base_count, start))
    end = max(start, min(base_count, end))
    return end - start


def estimate_hf_dataset_size(
    dataset_name,
    dataset_config=None,
    split="train",
    revision=None,
):
    from datasets import load_dataset_builder

    kwargs = {"revision": revision} if revision else {}
    builder = load_dataset_builder(dataset_name, dataset_config, **kwargs)
    info = builder.info
    split = split or "train"
    if split in info.splits:
        return info.splits[split].num_examples

    if "[" in split and split.endswith("]"):
        base_split, selector = split.split("[", 1)
        selector = selector[:-1]
        if base_split in info.splits:
            return _parse_hf_split_count(info.splits[base_split].num_examples, selector)

    return None


def _steps_from_samples(sample_count, batch_size):
    sample_count = max(0, int(sample_count or 0))
    batch_size = max(1, int(batch_size or 1))
    return max(1, (sample_count + batch_size - 1) // batch_size)


def _batch_effective_lengths(batch):
    """Return the last non-padding position for each collated cache row."""
    input_ids = batch["input_ids"]
    masks = batch.get("attention_mask")
    if masks is None:
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    width = int(masks.shape[1])
    positions = torch.arange(1, width + 1, device=masks.device, dtype=torch.long)
    return torch.where(masks.ne(0), positions, 0).amax(dim=1).tolist()


def _is_jsonl_path(path):
    return str(path).lower().endswith((".jsonl", ".ndjson"))


def _ensure_tokenizer_identity(args, tokenizer):
    identity = getattr(args, "_tokenizer_identity", None)
    if not isinstance(identity, dict) or not identity.get("sha256"):
        identity = _tokenizer_identity(tokenizer)
        args._tokenizer_identity = identity
    return identity


def _new_cache_audit():
    return {
        "version": 1,
        "records": 0,
        "accepted": 0,
        "rejected": 0,
        "truncated": 0,
        "original_tokens": 0,
        "retained_tokens": 0,
        "supervised_tokens": 0,
        "weighted_tokens": 0.0,
        "original_response_tokens": 0,
        "retained_response_tokens": 0,
        "rejection_reasons": {},
        "response_length_histogram": {
            "0": 0,
            "1-7": 0,
            "8-31": 0,
            "32-127": 0,
            "128-511": 0,
            "512+": 0,
        },
        "by_schema": {},
        "by_source": {},
    }


def _bounded_counter_bucket(mapping, name, *, limit=4096):
    name = str(name or "unknown")[:256]
    if name not in mapping and len(mapping) >= limit:
        name = "__other__"
    return mapping.setdefault(name, {
        "records": 0,
        "accepted": 0,
        "rejected": 0,
        "truncated": 0,
        "retained_tokens": 0,
        "supervised_tokens": 0,
        "weighted_tokens": 0.0,
        "retained_response_tokens": 0,
    })


def _response_length_bucket(length):
    length = max(0, int(length or 0))
    if length == 0:
        return "0"
    if length < 8:
        return "1-7"
    if length < 32:
        return "8-31"
    if length < 128:
        return "32-127"
    if length < 512:
        return "128-511"
    return "512+"


def _consume_cache_audit(batch, audit):
    records = batch.pop("_audit_records", None) if isinstance(batch, dict) else None
    if not records:
        # Third-party/legacy test loaders may not emit audit metadata. Preserve
        # compatibility while still reporting exact retained/loss mass.
        if not isinstance(batch, dict) or "input_ids" not in batch:
            return
        lengths = _batch_effective_lengths(batch)
        labels = batch.get("labels")
        weights = batch.get("loss_weights")
        records = []
        for row, length in enumerate(lengths):
            length = max(0, int(length))
            active = labels[row, 1:length].ne(-100) if labels is not None else torch.ones(
                max(0, length - 1), dtype=torch.bool
            )
            weighted = (
                float(weights[row, 1:length][active].sum().item())
                if weights is not None else float(active.sum().item())
            )
            records.append({
                "accepted": True,
                "schema": "unknown",
                "source": "unknown",
                "original_tokens": length,
                "retained_tokens": length,
                "original_response_tokens": 0,
                "retained_response_tokens": 0,
                "supervised_tokens": int(active.sum().item()),
                "weighted_tokens": weighted,
                "truncated": False,
            })

    for record in records:
        if not isinstance(record, dict):
            continue
        accepted = bool(record.get("accepted", False))
        truncated = bool(record.get("truncated", False))
        retained = max(0, int(record.get("retained_tokens", 0) or 0))
        supervised = max(0, int(record.get("supervised_tokens", 0) or 0))
        weighted = max(0.0, float(record.get("weighted_tokens", 0.0) or 0.0))
        retained_response = max(0, int(record.get("retained_response_tokens", 0) or 0))
        audit["records"] += 1
        audit["accepted" if accepted else "rejected"] += 1
        audit["truncated"] += int(truncated)
        audit["original_tokens"] += max(0, int(record.get("original_tokens", retained) or 0))
        audit["retained_tokens"] += retained
        audit["supervised_tokens"] += supervised
        audit["weighted_tokens"] += weighted
        audit["original_response_tokens"] += max(
            0, int(record.get("original_response_tokens", retained_response) or 0)
        )
        audit["retained_response_tokens"] += retained_response
        audit["response_length_histogram"][
            _response_length_bucket(retained_response)
        ] += int(accepted)
        if not accepted:
            reason = str(record.get("rejection_reason") or "unknown")[:256]
            audit["rejection_reasons"][reason] = (
                int(audit["rejection_reasons"].get(reason, 0)) + 1
            )
        for dimension, key in (
            ("by_schema", record.get("schema")),
            ("by_source", record.get("source")),
        ):
            bucket = _bounded_counter_bucket(audit[dimension], key)
            bucket["records"] += 1
            bucket["accepted" if accepted else "rejected"] += 1
            bucket["truncated"] += int(truncated)
            bucket["retained_tokens"] += retained
            bucket["supervised_tokens"] += supervised
            bucket["weighted_tokens"] += weighted
            bucket["retained_response_tokens"] += retained_response


def _enforce_cache_audit_budgets(args, audit):
    total = max(1, int(audit.get("records", 0) or 0))
    rejected = max(0, int(audit.get("rejected", 0) or 0))
    truncated = max(0, int(audit.get("truncated", 0) or 0))
    max_rejected_count = max(0, int(getattr(args, "max_cache_rejected_samples", 0) or 0))
    max_rejected_fraction = max(
        0.0, float(getattr(args, "max_cache_rejected_fraction", 0.0) or 0.0)
    )
    allowed_rejected = max(
        max_rejected_count,
        int(math.floor(total * min(1.0, max_rejected_fraction))),
    )
    raw_truncated_count = getattr(args, "max_cache_truncated_samples", -1)
    max_truncated_count = int(
        -1 if raw_truncated_count is None else raw_truncated_count
    )
    max_truncated_fraction = max(
        0.0, float(getattr(args, "max_cache_truncated_fraction", 0.05) or 0.0)
    )
    allowed_truncated = max(
        max(0, max_truncated_count),
        int(math.floor(total * min(1.0, max_truncated_fraction))),
    )
    failures = []
    if rejected > allowed_rejected:
        failures.append(
            f"rejected {rejected}/{total} rows (allowed {allowed_rejected}); "
            f"reasons={audit.get('rejection_reasons', {})}"
        )
    if truncated > allowed_truncated:
        failures.append(
            f"truncated {truncated}/{total} accepted rows (allowed {allowed_truncated})"
        )
    if failures:
        raise RuntimeError(
            "Token-cache data quality budget exceeded: " + "; ".join(failures)
            + ". Inspect cache_audit.json, repair the dataset/format flags, or "
            "set explicit --max-cache-*-samples/fraction budgets after review."
        )


def _persist_and_enforce_cache_audit(args, audit, *, tmp_dir, cache_dir):
    """Persist audit evidence before enforcing budgets and clean failed builds."""

    audit_path = os.path.join(tmp_dir, "cache_audit.json")
    with open(audit_path, "w", encoding="utf-8") as audit_file:
        json.dump(audit, audit_file, indent=2, sort_keys=True)

    failed_audit_path = cache_dir + ".failed-cache-audit.json"
    try:
        _enforce_cache_audit_budgets(args, audit)
    except Exception as exc:
        preserved_path = None
        try:
            shutil.copy2(audit_path, failed_audit_path)
            preserved_path = failed_audit_path
        except OSError:
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        suffix = (
            f" Full audit saved to {preserved_path}."
            if preserved_path is not None
            else " The temporary cache was removed; the audit could not be preserved."
        )
        raise RuntimeError(f"{exc}{suffix}") from exc

    try:
        if os.path.exists(failed_audit_path):
            os.remove(failed_audit_path)
    except OSError:
        pass
    return audit_path


def _jsonl_source_files(path):
    if _is_jsonl_path(path):
        return [path]
    if not path or not os.path.isdir(path):
        return []
    paths = []
    for _root, _dirs, files in os.walk(path):
        for filename in files:
            lower = filename.lower()
            if lower != "manifest.jsonl" and lower.endswith((".jsonl", ".ndjson")):
                paths.append(os.path.join(_root, filename))
    paths.sort()
    return paths


def _is_jsonl_source(path):
    return bool(_jsonl_source_files(path))


def count_jsonl_source_rows(path):
    count = 0
    for jsonl_path in _jsonl_source_files(path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            count += sum(1 for line in f if line.strip())
    return count


def _local_text_columns(args):
    return args.text_column, args.prompt_column, args.completion_column


def _length_bucketing_enabled(args, default=True):
    value = getattr(args, "length_bucketing", default)
    if value is None:
        return bool(default)
    return bool(value)


def _bool_cli_literal(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _normalize_optional_bool_flags(argv, flag_names):
    """Allow boolean endpoint forms like --flag true without breaking bare flags."""
    normalized = []
    i = 0
    flag_names = tuple(flag_names)
    while i < len(argv):
        token = argv[i]
        matched_inline = None
        for flag in flag_names:
            prefix = f"{flag}="
            if token.startswith(prefix):
                matched_inline = (flag, token[len(prefix):])
                break

        if matched_inline:
            flag, raw_value = matched_inline
            parsed = _bool_cli_literal(raw_value)
            if parsed is None:
                normalized.append(token)
            elif parsed:
                normalized.append(flag)
            i += 1
            continue

        if token in flag_names:
            if i + 1 < len(argv):
                parsed = _bool_cli_literal(argv[i + 1])
                if parsed is not None:
                    if parsed:
                        normalized.append(token)
                    i += 2
                    continue
            normalized.append(token)
            i += 1
            continue

        normalized.append(token)
        i += 1
    return normalized

def _rounded_bucket_size(bucket_size, batch_size, sample_count):
    batch_size = max(1, int(batch_size or 1))
    sample_count = max(batch_size, int(sample_count or batch_size))
    bucket_size = max(batch_size, int(bucket_size or batch_size))
    bucket_size = min(bucket_size, sample_count)
    return max(batch_size, (bucket_size // batch_size) * batch_size)


def _default_bucket_candidates(sample_count, batch_size):
    batch_size = max(1, int(batch_size or 1))
    sample_count = max(batch_size, int(sample_count or batch_size))
    bases = [
        batch_size * 50,
        8192,
        32768,
        65536,
        131072,
        262144,
        524288,
        sample_count,
    ]
    candidates = []
    seen = set()
    for candidate in bases:
        rounded = _rounded_bucket_size(candidate, batch_size, sample_count)
        if rounded not in seen:
            seen.add(rounded)
            candidates.append(rounded)
    return candidates


def estimate_bucket_token_efficiency(lengths, batch_size, chunk_size, bucket_size, permutation=None):
    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
    batch_size = max(1, int(batch_size or 1))
    chunk_size = max(1, int(chunk_size or 1))
    if lengths_t.numel() < batch_size:
        return {
            "bucket_size": batch_size,
            "token_efficiency": 1.0,
            "real_tokens": int(lengths_t.sum().item()) if lengths_t.numel() else 0,
            "compiled_tokens": int(lengths_t.sum().item()) if lengths_t.numel() else 0,
            "batches": 0,
        }

    full_count = (int(lengths_t.numel()) // batch_size) * batch_size
    if permutation is None:
        generator = torch.Generator()
        generator.manual_seed(12345)
        permutation = torch.randperm(int(lengths_t.numel()), generator=generator)
    else:
        permutation = torch.as_tensor(permutation, dtype=torch.long, device="cpu")

    ordered = lengths_t[permutation[:full_count]].clone()
    bucket_size = _rounded_bucket_size(bucket_size, batch_size, full_count)
    for start in range(0, full_count, bucket_size):
        end = min(start + bucket_size, full_count)
        ordered[start:end] = torch.sort(ordered[start:end], descending=True).values

    batches = ordered.view(-1, batch_size)
    max_lengths = batches.max(dim=1).values
    compiled_lengths = ((max_lengths + chunk_size - 1) // chunk_size) * chunk_size
    real_tokens = int(batches.sum().item())
    compiled_tokens = int(compiled_lengths.sum().item()) * batch_size
    token_efficiency = real_tokens / float(max(1, compiled_tokens))
    return {
        "bucket_size": bucket_size,
        "token_efficiency": token_efficiency,
        "real_tokens": real_tokens,
        "compiled_tokens": compiled_tokens,
        "batches": int(batches.shape[0]),
    }


def choose_length_bucket_size_from_lengths(
    lengths,
    batch_size,
    chunk_size=256,
    candidates=None,
    tolerance=0.005,
    seed=12345,
):
    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
    lengths_t = torch.clamp(lengths_t, min=1)
    batch_size = max(1, int(batch_size or 1))
    if lengths_t.numel() < batch_size * 2:
        bucket_size = _rounded_bucket_size(lengths_t.numel(), batch_size, max(batch_size, int(lengths_t.numel())))
        result = estimate_bucket_token_efficiency(lengths_t, batch_size, chunk_size, bucket_size)
        return bucket_size, {"chosen": result, "best": result, "results": [result]}

    full_count = (int(lengths_t.numel()) // batch_size) * batch_size
    if candidates is None:
        candidates = _default_bucket_candidates(full_count, batch_size)
    rounded_candidates = []
    seen = set()
    for candidate in candidates:
        rounded = _rounded_bucket_size(candidate, batch_size, full_count)
        if rounded not in seen:
            seen.add(rounded)
            rounded_candidates.append(rounded)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(int(lengths_t.numel()), generator=generator)
    results = [
        estimate_bucket_token_efficiency(
            lengths_t,
            batch_size,
            chunk_size,
            candidate,
            permutation=permutation,
        )
        for candidate in rounded_candidates
    ]
    best = max(results, key=lambda item: item["token_efficiency"])
    threshold = best["token_efficiency"] - max(0.0, float(tolerance or 0.0))
    chosen = min(
        (item for item in results if item["token_efficiency"] >= threshold),
        key=lambda item: item["bucket_size"],
    )
    return chosen["bucket_size"], {"chosen": chosen, "best": best, "results": results}


def auto_tune_length_bucket_size_from_token_cache(args, cache_dir):
    index_path = os.path.join(cache_dir, "index.pt")
    if not os.path.exists(index_path):
        return None
    tuning_path = os.path.join(cache_dir, "bucket_tuning.json")
    try:
        index = load_tensor_payload_safely(index_path, map_location="cpu")
        if not isinstance(index, dict) or not torch.is_tensor(index.get("lengths")):
            raise ValueError(
                f"Token-cache index has no tensor lengths payload: {index_path}"
            )
        lengths = index["lengths"].to(dtype=torch.long, device="cpu")
        max_length = int(getattr(args, "max_length", 0) or 0)
        if max_length > 0:
            lengths = torch.clamp(lengths, max=max_length)
        sample_limit = max(
            0,
            int(getattr(args, "length_bucket_auto_sample_size", 1_000_000) or 0),
        )
        settings = {
            "version": 1,
            "cache_key": index.get("cache_key"),
            "source_samples": int(lengths.numel()),
            "batch_size": int(getattr(args, "batch_size", 1) or 1),
            "training_chunk_size": int(
                getattr(args, "training_chunk_size", DEFAULT_TRAINING_CHUNK_SIZE)
                or DEFAULT_TRAINING_CHUNK_SIZE
            ),
            "max_length": max_length,
            "tolerance": float(getattr(args, "length_bucket_auto_tolerance", 0.005) or 0.0),
            "sample_limit": sample_limit,
        }
        if os.path.exists(tuning_path):
            try:
                with open(tuning_path, "r", encoding="utf-8") as tuning_file:
                    cached_tuning = json.load(tuning_file)
                if cached_tuning.get("settings") == settings:
                    bucket_size = int(cached_tuning["bucket_size"])
                    args.length_bucket_size = bucket_size
                    print(
                        "INFO: Reusing persisted HF token-cache length bucket window "
                        f"{bucket_size} (tuned from {cached_tuning.get('sampled_lengths', 0)} lengths)."
                    )
                    return bucket_size
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                pass

        source_length_count = int(lengths.numel())
        if sample_limit > 0 and source_length_count > sample_limit:
            # Evenly spaced deterministic sampling covers the entire cache without
            # allocating a full-size randperm. choose_length_bucket_size shuffles
            # the compact sample before evaluating candidate windows.
            sample_indices = torch.linspace(
                0,
                source_length_count - 1,
                steps=sample_limit,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
            lengths = lengths.index_select(0, sample_indices)
        bucket_size, summary = choose_length_bucket_size_from_lengths(
            lengths,
            batch_size=settings["batch_size"],
            chunk_size=settings["training_chunk_size"],
            tolerance=settings["tolerance"],
        )
    except Exception as exc:
        print(f"WARNING: Length bucket auto-tune failed; using fallback. {exc}")
        return None

    args.length_bucket_size = int(bucket_size)
    chosen = summary["chosen"]
    best = summary["best"]
    result_text = ", ".join(
        f"{item['bucket_size']}={item['token_efficiency'] * 100.0:.1f}%"
        for item in summary["results"]
    )
    print(
        "INFO: Auto-tuned HF token-cache length bucket window to "
        f"{bucket_size} (estimated tok_eff={chosen['token_efficiency'] * 100.0:.1f}%; "
        f"best={best['bucket_size']} at {best['token_efficiency'] * 100.0:.1f}%; "
        f"sampled={int(lengths.numel())}/{settings['source_samples']}; candidates: {result_text})."
    )
    try:
        tuning_payload = {
            "settings": settings,
            "bucket_size": int(bucket_size),
            "sampled_lengths": int(lengths.numel()),
            "chosen": chosen,
            "best": best,
        }
        tmp_tuning_path = tuning_path + ".tmp"
        with open(tmp_tuning_path, "w", encoding="utf-8") as tuning_file:
            json.dump(tuning_payload, tuning_file, indent=2)
        os.replace(tmp_tuning_path, tuning_path)
    except OSError as exc:
        print(f"WARNING: Could not persist length bucket auto-tune result: {exc}")
    return int(bucket_size)


def resolve_length_bucket_size(length_bucket_size, device, batch_size, hf_token_cache=False, auto_tune=True):
    if length_bucket_size is not None:
        return length_bucket_size, None
    device_type = getattr(device, "type", str(device)).lower()
    if device_type == "cuda" and int(batch_size or 1) >= 64:
        if bool(hf_token_cache) and bool(auto_tune):
            return (
                None,
                "INFO: Length bucket window will auto-tune after the HF token cache is ready.",
            )
        if bool(hf_token_cache):
            return (
                65536,
                "INFO: Length bucket window auto-set to 65536 for CUDA batch>=64 with HF token cache.",
            )
        return (
            8192,
            "INFO: Length bucket window auto-set to 8192 for CUDA batch>=64.",
        )
    return length_bucket_size, None


def _loss_weight_kwargs(args):
    return {
        "prompt_loss_weight": float(getattr(args, "prompt_loss_weight", 1.0) or 0.0),
        "response_loss_weight": float(getattr(args, "response_loss_weight", 1.0) or 0.0),
        "response_boundary_loss_weight": float(getattr(args, "response_boundary_loss_weight", 1.0) or 0.0),
        "response_boundary_tokens": int(getattr(args, "response_boundary_tokens", 0) or 0),
    }


def _response_quality_kwargs(args):
    return {
        "min_response_tokens": int(getattr(args, "min_response_tokens", 1) or 0),
        "drop_empty_completions": bool(getattr(args, "drop_empty_completions", True)),
    }


def _uses_weighted_token_loss(args):
    weights = _loss_weight_kwargs(args)
    return (
        abs(weights["prompt_loss_weight"] - 1.0) > 1e-8
        or abs(weights["response_loss_weight"] - 1.0) > 1e-8
        or (
            weights["response_boundary_tokens"] > 0
            and abs(weights["response_boundary_loss_weight"] - 1.0) > 1e-8
        )
    )


def _token_cache_loss_weight_palette(args):
    """Return the exact float32 values emitted by the current formatter."""
    weights = _loss_weight_kwargs(args)
    prompt_weight = (
        weights["prompt_loss_weight"]
        if bool(getattr(args, "train_prompt_tokens", True))
        else 0.0
    )
    candidates = torch.tensor(
        [
            0.0,
            1.0,
            prompt_weight,
            weights["response_loss_weight"],
            weights["response_loss_weight"] * weights["response_boundary_loss_weight"],
        ],
        dtype=torch.float32,
    )
    palette = []
    seen_bits = set()
    for value, bits in zip(candidates.tolist(), candidates.view(torch.int32).tolist()):
        if bits not in seen_bits:
            seen_bits.add(bits)
            palette.append(float(value))
    if len(palette) > 255:
        raise ValueError("Token-cache loss-weight palette exceeds uint8 capacity")
    return palette


def _token_cache_storage_layout(args, tokenizer, *, has_rosa_ids, has_loss_weights):
    vocab_size = _tokenizer_vocab_size(tokenizer)
    # 65535 is reserved for ignore_index in the compact label stream.
    compact_u16 = 0 < vocab_size <= 65534
    token_dtype = "uint16" if compact_u16 else "int32"
    token_bytes = 2 if compact_u16 else 4
    labels_elided = bool(getattr(args, "train_prompt_tokens", True))
    return {
        "storage_schema_version": 6,
        "byte_order": "little",
        "token_dtype": token_dtype,
        "label_dtype": None if labels_elided else token_dtype,
        "label_encoding": "input_ids_alias" if labels_elided else None,
        "rosa_dtype": token_dtype if has_rosa_ids else None,
        "token_bytes": token_bytes,
        "label_bytes": 0 if labels_elided else token_bytes,
        "rosa_bytes": token_bytes if has_rosa_ids else 0,
        "label_ignore_sentinel": (
            None if labels_elided else (65535 if compact_u16 else -100)
        ),
        "has_rosa_ids": bool(has_rosa_ids),
        "has_loss_weights": bool(has_loss_weights),
        "loss_weight_encoding": (
            "float32_palette_rle" if has_loss_weights else None
        ),
        "loss_weight_palette": (
            _token_cache_loss_weight_palette(args) if has_loss_weights else None
        ),
    }


def _encode_cache_integer_tensor(values, *, layout, field_name, allow_ignore=False):
    values = values.detach().cpu().to(dtype=torch.int64).contiguous()
    if layout["token_dtype"] == "uint16":
        valid = (values >= 0) & (values <= 65534)
        if allow_ignore:
            valid = valid | (values == -100)
        if not bool(valid.all().item()):
            bad = values[~valid][:8].tolist()
            raise ValueError(
                f"Cannot encode {field_name} in compact uint16 token cache; "
                f"invalid values include {bad}"
            )
        if allow_ignore:
            values = torch.where(
                values == -100,
                torch.full_like(values, layout["label_ignore_sentinel"]),
                values,
            )
        return values.to(dtype=torch.uint16).contiguous()
    if allow_ignore:
        valid = (values == -100) | (
            (values >= torch.iinfo(torch.int32).min)
            & (values <= torch.iinfo(torch.int32).max)
        )
    else:
        valid = (values >= 0) & (values <= torch.iinfo(torch.int32).max)
    if not bool(valid.all().item()):
        raise ValueError(f"Cannot encode out-of-range {field_name} in int32 token cache")
    return values.to(dtype=torch.int32).contiguous()


def _append_loss_weight_runs(
    batch,
    row,
    length,
    *,
    palette,
    run_offsets,
    run_ends,
    run_codes,
):
    cached_codes = batch.get("_cache_loss_weight_codes")
    if cached_codes is not None:
        codes = cached_codes[row, :length]
    else:
        cached_weights = batch.get("loss_weights")
        if cached_weights is None:
            weights = torch.ones(length, dtype=torch.float32)
        else:
            weights = cached_weights[row, :length].detach().cpu().to(dtype=torch.float32).contiguous()
        palette_tensor = (
            palette
            if isinstance(palette, torch.Tensor)
            else torch.tensor(palette, dtype=torch.float32)
        )
        matches = weights.unsqueeze(-1).eq(palette_tensor.view(1, -1))
        matched = matches.any(dim=-1)
        if not bool(matched.all().item()):
            unknown = weights[~matched][:8].tolist()
            raise ValueError(
                "Loss weights are not representable by the exact cache palette; "
                f"unexpected values include {unknown}"
            )
        codes = matches.to(dtype=torch.uint8).argmax(dim=-1).to(dtype=torch.uint8)
    if length > 1:
        changes = torch.nonzero(codes[1:] != codes[:-1], as_tuple=False).flatten() + 1
    else:
        changes = torch.empty(0, dtype=torch.long)
    ends = torch.cat([changes, torch.tensor([length], dtype=torch.long)])
    starts = torch.cat([torch.tensor([0], dtype=torch.long), changes])
    sample_codes = codes.index_select(0, starts)
    run_ends.extend(int(value) for value in ends.tolist())
    run_codes.extend(int(value) for value in sample_codes.tolist())
    run_offsets.append(len(run_ends))


def _append_token_cache_record(
    write_buffer,
    batch,
    row,
    length,
    *,
    layout,
    precompute_rosa,
    identity_hasher=None,
):
    labels_elided = layout.get("label_encoding") == "input_ids_alias"
    if batch.get("_cache_storage_encoded", False):
        input_ids = batch["input_ids"][row, :length]
        labels = None if labels_elided else batch["labels"][row, :length]
    else:
        input_ids = _encode_cache_integer_tensor(
            batch["input_ids"][row, :length],
            layout=layout,
            field_name="input_ids",
        )
        labels = None
        if labels_elided:
            raw_inputs = batch["input_ids"][row, :length].detach().cpu().to(dtype=torch.long)
            raw_labels = batch["labels"][row, :length].detach().cpu().to(dtype=torch.long)
            if not torch.equal(raw_inputs, raw_labels):
                raise ValueError(
                    "Cannot elide token-cache labels because labels differ from input_ids"
                )
        else:
            labels = _encode_cache_integer_tensor(
                batch["labels"][row, :length],
                layout=layout,
                field_name="labels",
                allow_ignore=True,
            )
    parts = [input_ids.numpy().tobytes()]
    if labels is not None:
        parts.append(labels.numpy().tobytes())
    if precompute_rosa:
        cached_rosa = batch.get("rosa_ids")
        if cached_rosa is None:
            raise RuntimeError("ROSA cache builder did not return precomputed ids")
        if batch.get("_cache_storage_encoded", False):
            rosa_ids = cached_rosa[row, :length]
        else:
            rosa_ids = _encode_cache_integer_tensor(
                cached_rosa[row, :length],
                layout=layout,
                field_name="rosa_ids",
            )
        parts.append(rosa_ids.numpy().tobytes())
    for part in parts:
        write_buffer.extend(part)
    if identity_hasher is not None:
        identity_hasher.update(int(length).to_bytes(8, "little", signed=False))
        for part in parts:
            identity_hasher.update(part)
        if layout.get("has_loss_weights"):
            codes = batch.get("_cache_loss_weight_codes")
            if codes is not None:
                identity_hasher.update(
                    codes[row, :length].detach().cpu().contiguous().numpy().tobytes()
                )
            else:
                weights = batch.get("loss_weights")
                if weights is None:
                    identity_hasher.update(
                        torch.ones(length, dtype=torch.float32).numpy().tobytes()
                    )
                else:
                    identity_hasher.update(
                        weights[row, :length].detach().cpu().to(
                            dtype=torch.float32
                        ).contiguous().numpy().tobytes()
                    )
    return sum(len(part) for part in parts)


def _prepare_token_cache_batch(batch, lengths, *, layout, precompute_rosa):
    """Vectorize compact dtype conversion and palette lookup once per batch."""
    lengths = [max(0, int(length)) for length in lengths]
    max_length = max(lengths, default=0)
    if max_length <= 0:
        return batch
    prepared = {
        "_cache_storage_encoded": True,
        "input_ids": _encode_cache_integer_tensor(
            batch["input_ids"][:, :max_length],
            layout=layout,
            field_name="input_ids",
        ),
    }
    if layout.get("label_encoding") == "input_ids_alias":
        raw_inputs = batch["input_ids"][:, :max_length].detach().cpu().to(dtype=torch.long)
        raw_labels = batch["labels"][:, :max_length].detach().cpu().to(dtype=torch.long)
        valid_positions = (
            torch.arange(max_length, dtype=torch.long).unsqueeze(0)
            < torch.tensor(lengths, dtype=torch.long).unsqueeze(1)
        )
        mismatch = valid_positions & raw_labels.ne(raw_inputs)
        if bool(mismatch.any().item()):
            coordinates = torch.nonzero(mismatch, as_tuple=False)[:8].tolist()
            raise ValueError(
                "Cannot elide token-cache labels: labels differ from input_ids at "
                f"real-token coordinates {coordinates}. Disable --train-prompt-tokens "
                "or rebuild with an explicit-label cache format."
            )
    else:
        prepared["labels"] = _encode_cache_integer_tensor(
            batch["labels"][:, :max_length],
            layout=layout,
            field_name="labels",
            allow_ignore=True,
        )
    if precompute_rosa:
        cached_rosa = batch.get("rosa_ids")
        if cached_rosa is None:
            raise RuntimeError("ROSA cache builder did not return precomputed ids")
        prepared["rosa_ids"] = _encode_cache_integer_tensor(
            cached_rosa[:, :max_length],
            layout=layout,
            field_name="rosa_ids",
        )
    if layout["has_loss_weights"]:
        cached_weights = batch.get("loss_weights")
        if cached_weights is None:
            weights = torch.ones(
                (len(lengths), max_length),
                dtype=torch.float32,
            )
        else:
            weights = cached_weights[:, :max_length].detach().cpu().to(
                dtype=torch.float32
            ).contiguous()
        palette = torch.tensor(layout["loss_weight_palette"], dtype=torch.float32)
        matches = weights.unsqueeze(-1).eq(palette.view(1, 1, -1))
        valid_positions = (
            torch.arange(max_length, dtype=torch.long).unsqueeze(0)
            < torch.tensor(lengths, dtype=torch.long).unsqueeze(1)
        )
        matched = matches.any(dim=-1)
        invalid = valid_positions & ~matched
        if bool(invalid.any().item()):
            unknown = weights[invalid][:8].tolist()
            raise ValueError(
                "Loss weights are not representable by the exact cache palette; "
                f"unexpected values include {unknown}"
            )
        prepared["_cache_loss_weight_codes"] = matches.to(
            dtype=torch.uint8
        ).argmax(dim=-1).to(dtype=torch.uint8)
    return prepared


def _token_cache_index_layout(layout, run_offsets, run_ends, run_codes):
    metadata = {
        "storage_schema_version": layout["storage_schema_version"],
        "byte_order": layout["byte_order"],
        "token_dtype": layout["token_dtype"],
        "label_dtype": layout["label_dtype"],
        "label_encoding": layout["label_encoding"],
        "rosa_dtype": layout["rosa_dtype"],
        "label_ignore_sentinel": layout["label_ignore_sentinel"],
        "loss_weight_encoding": layout["loss_weight_encoding"],
        "loss_weight_palette": layout["loss_weight_palette"],
    }
    if layout["has_loss_weights"]:
        metadata.update({
            "loss_run_offsets": torch.tensor(run_offsets, dtype=torch.long),
            "loss_run_ends": torch.tensor(run_ends, dtype=torch.int32),
            "loss_run_codes": torch.tensor(run_codes, dtype=torch.uint8),
        })
    return metadata


def _hf_cache_key_payload(args, *, format_name, num_shards=None):
    payload = {
        "format": format_name,
        "formatter": HF_CACHE_FORMATTER_VERSION,
        "dataset": args.hf_dataset,
        "config": args.hf_dataset_config,
        "split": args.hf_dataset_split,
        "dataset_revision": (
            getattr(args, "_resolved_hf_dataset_revision", None)
            or getattr(args, "hf_dataset_revision", None)
        ),
        "tokenizer": (
            getattr(args, "tokenizer_path", None)
            or getattr(args, "model_path", None)
            or "openai-community/gpt2"
        ),
        "tokenizer_identity": getattr(args, "_tokenizer_identity", None),
        "max_length": int(getattr(args, "max_length", 0) or 0),
        "kayla": bool(getattr(args, "kayla", False)),
        "alpaca": bool(getattr(args, "alpaca", False)),
        "alpaca_input_field": "input",
        "alpaca_input_role": "previous_context",
        "train_prompt_tokens": bool(getattr(args, "train_prompt_tokens", True)),
        "prompt_loss_weight": float(getattr(args, "prompt_loss_weight", 1.0) or 0.0),
        "response_loss_weight": float(getattr(args, "response_loss_weight", 1.0) or 0.0),
        "response_boundary_loss_weight": float(getattr(args, "response_boundary_loss_weight", 1.0) or 0.0),
        "response_boundary_tokens": int(getattr(args, "response_boundary_tokens", 0) or 0),
        "min_response_tokens": int(getattr(args, "min_response_tokens", 1) or 0),
        "drop_empty_completions": bool(getattr(args, "drop_empty_completions", True)),
        "text_column": getattr(args, "text_column", None),
        "prompt_column": getattr(args, "prompt_column", None),
        "completion_column": getattr(args, "completion_column", None),
        "precompute_rosa": bool(getattr(args, "use_rosa", True)),
        "rosa_max_context": int(getattr(args, "rosa_max_context", 512) or 512),
        "enforce_rosa_max_context": bool(
            getattr(args, "enforce_rosa_max_context", False)
        ),
        "rosa_ids_context_mode": rosa_context_mode(
            getattr(args, "enforce_rosa_max_context", False)
        ),
        "training_chunk_size": int(
            getattr(args, "training_chunk_size", DEFAULT_TRAINING_CHUNK_SIZE)
            or DEFAULT_TRAINING_CHUNK_SIZE
        ),
    }
    if num_shards is not None:
        payload["num_shards"] = int(num_shards)
        payload["chunks_per_file"] = int(getattr(args, "hf_cache_chunks_per_file", 2048) or 2048)
    return payload


def _hf_shard_cache_key(args, num_shards):
    payload = _hf_cache_key_payload(
        args,
        format_name="tokenized-pt-shards-v3",
        num_shards=num_shards,
    )
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _legacy_hf_shard_cache_key(args, num_shards):
    payload = {
        "format": "tokenized-pt-shards-v2",
        "formatter": "alpaca-input-section-v1",
        "dataset": args.hf_dataset,
        "config": args.hf_dataset_config,
        "split": args.hf_dataset_split,
        "num_shards": int(num_shards),
        "chunks_per_file": int(getattr(args, "hf_cache_chunks_per_file", 2048) or 2048),
        "tokenizer": (
            getattr(args, "tokenizer_path", None)
            or getattr(args, "model_path", None)
            or "openai-community/gpt2"
        ),
        "max_length": int(getattr(args, "max_length", 0) or 0),
        "kayla": bool(getattr(args, "kayla", False)),
        "alpaca": bool(getattr(args, "alpaca", False)),
        "train_prompt_tokens": bool(getattr(args, "train_prompt_tokens", True)),
        "text_column": getattr(args, "text_column", None),
        "prompt_column": getattr(args, "prompt_column", None),
        "completion_column": getattr(args, "completion_column", None),
        "precompute_rosa": bool(getattr(args, "use_rosa", True)),
        "rosa_max_context": int(getattr(args, "rosa_max_context", 512) or 512),
        "training_chunk_size": int(
            getattr(args, "training_chunk_size", DEFAULT_TRAINING_CHUNK_SIZE)
            or DEFAULT_TRAINING_CHUNK_SIZE
        ),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _estimate_hf_total_for_progress(args):
    total_samples = getattr(args, "dataset_size", 0) or None
    if total_samples is not None:
        return total_samples
    try:
        return estimate_hf_dataset_size(
            args.hf_dataset,
            args.hf_dataset_config,
            args.hf_dataset_split,
            revision=_hf_revision_for_load(args),
        )
    except Exception:
        return None


def _processed_sample_to_cached_item(processed, args=None, tokenizer=None):
    input_ids = processed["input_ids"].detach().cpu().to(dtype=torch.int32)
    labels = processed["labels"].detach().cpu().to(dtype=torch.int32)
    length = int(processed.get("_length", input_ids.numel()))
    length = max(1, min(int(input_ids.numel()), length))
    item = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones(input_ids.numel(), dtype=torch.uint8),
        "_length": length,
    }
    if "loss_weights" in processed:
        item["loss_weights"] = processed["loss_weights"].detach().cpu().to(dtype=torch.float32)
    if args is not None and tokenizer is not None and bool(getattr(args, "use_rosa", True)):
        rosa_sentinel = _tokenizer_vocab_size(tokenizer)
        item["rosa_ids"] = torch.tensor(
            precompute_rosa_ids_for_chunks(
                input_ids[:length].tolist(),
                vocab_size=rosa_sentinel,
                chunk_size=getattr(
                    args,
                    "training_chunk_size",
                    DEFAULT_TRAINING_CHUNK_SIZE,
                ),
                rosa_max_ctx=getattr(args, "rosa_max_context", 512),
                enforce_max_context=bool(
                    getattr(args, "enforce_rosa_max_context", False)
                ),
            ),
            dtype=torch.int32,
        )
        item["_rosa_sentinel"] = int(rosa_sentinel)
        item["_rosa_context_mode"] = rosa_context_mode(
            getattr(args, "enforce_rosa_max_context", False)
        )
    return item


def materialize_hf_dataset_pt_cache(args, tokenizer, num_shards):
    resolve_hf_dataset_revision(args, announce=True)
    num_shards = max(1, int(num_shards or 1))
    cache_root = getattr(args, "hf_shard_cache_dir", None) or os.path.join(os.getcwd(), ".hierarchos_hf_shards")
    shard_key = _hf_shard_cache_key(args, num_shards)
    shard_dir = os.path.join(cache_root, shard_key)
    legacy_shard_dir = os.path.join(cache_root, _legacy_hf_shard_cache_key(args, num_shards))
    success_path = os.path.join(shard_dir, "_SUCCESS")
    manifest_path = os.path.join(shard_dir, "manifest.jsonl")

    if os.path.exists(legacy_shard_dir) and legacy_shard_dir != shard_dir:
        print(
            "INFO: Ignoring legacy HF PT shard cache built before all-token "
            f"coherence cache versioning: {legacy_shard_dir}"
        )

    if (
        not getattr(args, "refresh_hf_shards", False)
        and os.path.exists(success_path)
        and os.path.exists(manifest_path)
    ):
        print(f"INFO: Reusing cached HF tokenized PT shards from {shard_dir}")
        return shard_dir

    tmp_dir = shard_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if getattr(args, "refresh_hf_shards", False) and os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print(
        "INFO: Materializing HF dataset into "
        f"{num_shards} tokenized PT shard stream(s) for multi-worker loading..."
    )
    hf_dataset = load_hf_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
        streaming=True,
        revision=_hf_revision_for_load(args),
    )

    chunks_per_file = max(1, int(getattr(args, "hf_cache_chunks_per_file", 2048) or 2048))
    shard_buffers = [[] for _ in range(num_shards)]
    shard_counts = [0] * int(num_shards)
    shard_parts = [0] * int(num_shards)
    accepted = 0
    skipped = 0

    def flush_shard(manifest_file, shard_idx):
        buffer = shard_buffers[shard_idx]
        if not buffer:
            return
        filename = f"shard_{shard_idx:05d}_part_{shard_parts[shard_idx]:05d}.pt"
        path = os.path.join(tmp_dir, filename)
        torch.save(buffer, path)
        for item_idx, item in enumerate(buffer):
            length = int(item.get("_length", item["input_ids"].numel()))
            manifest_file.write(json.dumps({
                "file_path": filename,
                "index_in_file": item_idx,
                "length": length,
                "valid_length": length,
            }) + "\n")
        shard_buffers[shard_idx] = []
        shard_parts[shard_idx] += 1

    with open(os.path.join(tmp_dir, "manifest.jsonl"), "w", encoding="utf-8") as manifest_file:
        total_samples = _estimate_hf_total_for_progress(args)
        skipped = 0
        for sample_idx, sample in enumerate(tqdm(
            hf_dataset,
            desc="Tokenizing/sharding HF dataset",
            unit="sample",
            total=total_samples,
        )):
            processed = process_text_sample(
                tokenizer,
                sample,
                getattr(args, "max_length", 1024),
                getattr(args, "kayla", False),
                text_column=getattr(args, "text_column", None),
                prompt_column=getattr(args, "prompt_column", None),
                completion_column=getattr(args, "completion_column", None),
                alpaca_mode=getattr(args, "alpaca", False),
                train_prompt_tokens=getattr(args, "train_prompt_tokens", True),
                **_loss_weight_kwargs(args),
                **_response_quality_kwargs(args),
            )
            if processed is None:
                skipped += 1
                continue
            item = _processed_sample_to_cached_item(processed, args=args, tokenizer=tokenizer)
            item["_source_idx"] = sample_idx
            shard_idx = accepted % num_shards
            shard_buffers[shard_idx].append(item)
            shard_counts[shard_idx] += 1
            accepted += 1
            if len(shard_buffers[shard_idx]) >= chunks_per_file:
                flush_shard(manifest_file, shard_idx)

        for shard_idx in range(num_shards):
            flush_shard(manifest_file, shard_idx)

    with open(os.path.join(tmp_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({
            "format": "tokenized-pt-shards-v3",
            "formatter": HF_CACHE_FORMATTER_VERSION,
            "cache_key": shard_key,
            "cache_payload": _hf_cache_key_payload(
                args,
                format_name="tokenized-pt-shards-v3",
                num_shards=num_shards,
            ),
            "num_shards": num_shards,
            "counts": shard_counts,
            "parts": shard_parts,
            "skipped": skipped,
            "max_length": getattr(args, "max_length", 1024),
            "train_prompt_tokens": bool(getattr(args, "train_prompt_tokens", True)),
            "chunks_per_file": chunks_per_file,
            "has_rosa_ids": bool(getattr(args, "use_rosa", True)),
            "rosa_max_context": int(
                getattr(args, "rosa_max_context", 512) or 512
            ),
            "rosa_training_chunk_size": int(
                getattr(
                    args,
                    "training_chunk_size",
                    DEFAULT_TRAINING_CHUNK_SIZE,
                )
                or DEFAULT_TRAINING_CHUNK_SIZE
            ),
            "enforce_rosa_max_context": bool(
                getattr(args, "enforce_rosa_max_context", False)
            ),
            "rosa_ids_context_mode": rosa_context_mode(
                getattr(args, "enforce_rosa_max_context", False)
            ),
        }, f, indent=2)

    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    _replace_cache_directory_atomically(tmp_dir, shard_dir)
    print(
        f"INFO: HF tokenized PT shards ready in {shard_dir} "
        f"({sum(shard_counts)} samples, skipped {skipped})."
    )
    return shard_dir


def materialize_hf_dataset_shards(args, tokenizer, num_shards):
    return materialize_hf_dataset_pt_cache(args, tokenizer, num_shards)


def _hf_token_cache_key(args):
    payload = _hf_cache_key_payload(args, format_name=_hf_token_cache_format(args))
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _hf_token_cache_format(args):
    format_name = (
        "map-token-bin-v6-compact-weighted"
        if _uses_weighted_token_loss(args)
        else "map-token-bin-v6-compact"
    )
    if bool(getattr(args, "train_prompt_tokens", True)):
        format_name += "-label-alias"
    return format_name


def _legacy_hf_token_cache_key(args):
    payload = {
        "format": "map-token-bin-v3",
        "formatter": "alpaca-input-section-v1",
        "dataset": args.hf_dataset,
        "config": args.hf_dataset_config,
        "split": args.hf_dataset_split,
        "tokenizer": (
            getattr(args, "tokenizer_path", None)
            or getattr(args, "model_path", None)
            or "openai-community/gpt2"
        ),
        "max_length": int(getattr(args, "max_length", 0) or 0),
        "kayla": bool(getattr(args, "kayla", False)),
        "alpaca": bool(getattr(args, "alpaca", False)),
        "train_prompt_tokens": bool(getattr(args, "train_prompt_tokens", True)),
        "text_column": getattr(args, "text_column", None),
        "prompt_column": getattr(args, "prompt_column", None),
        "completion_column": getattr(args, "completion_column", None),
        "precompute_rosa": bool(getattr(args, "use_rosa", True)),
        "rosa_max_context": int(getattr(args, "rosa_max_context", 512) or 512),
        "training_chunk_size": int(
            getattr(args, "training_chunk_size", DEFAULT_TRAINING_CHUNK_SIZE)
            or DEFAULT_TRAINING_CHUNK_SIZE
        ),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _file_sha256(path, *, chunk_bytes=8 << 20):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256_digest(value):
    normalized = str(value or "").strip()
    return (
        len(normalized) == 64
        and all(character in "0123456789abcdefABCDEF" for character in normalized)
    )


def _read_token_cache_identity(cache_dir):
    index_path = os.path.join(cache_dir, "index.pt")
    success_path = os.path.join(cache_dir, "_SUCCESS")
    data_path = os.path.join(cache_dir, "tokens.bin")
    with open(success_path, "r", encoding="utf-8") as stream:
        success = json.load(stream)
    index = load_tensor_payload_safely(index_path, map_location="cpu")
    if not isinstance(index, dict) or not torch.is_tensor(index.get("lengths")):
        raise ValueError(
            f"Token-cache index has no tensor lengths payload: {index_path}"
        )
    for field_name in ("cache_key", "format", "cache_payload", "audit_sha256"):
        index_value = index.get(field_name)
        success_value = success.get(field_name)
        if (
            index_value is not None
            and success_value is not None
            and index_value != success_value
        ):
            raise RuntimeError(
                f"Token cache {field_name} metadata disagrees between index.pt "
                f"and _SUCCESS: {cache_dir}"
            )
    length_count = int(index["lengths"].numel())
    if success.get("samples") is not None:
        try:
            success_samples = int(success["samples"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Token cache has an invalid sample count in _SUCCESS: {cache_dir}"
            ) from exc
        if success_samples != length_count:
            raise RuntimeError(
                f"Token cache sample count mismatch: _SUCCESS={success_samples}, "
                f"index.pt={length_count}"
            )
    if success.get("bytes") is not None:
        try:
            success_bytes = int(success["bytes"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Token cache has an invalid byte count in _SUCCESS: {cache_dir}"
            ) from exc
        actual_bytes = os.path.getsize(data_path)
        if success_bytes != actual_bytes:
            raise RuntimeError(
                f"Token cache byte count mismatch: _SUCCESS={success_bytes}, "
                f"tokens.bin={actual_bytes}"
            )
    audit_path = os.path.join(cache_dir, str(success.get("audit_file") or "cache_audit.json"))
    expected_audit_hash = success.get("audit_sha256") or index.get("audit_sha256")
    if expected_audit_hash and not _is_sha256_digest(expected_audit_hash):
        raise RuntimeError(
            f"Token cache has a malformed audit checksum: {cache_dir}"
        )
    if expected_audit_hash and (
        not os.path.exists(audit_path)
        or _file_sha256(audit_path) != str(expected_audit_hash)
    ):
        raise RuntimeError(
            f"Token cache audit manifest checksum failed: {audit_path}"
        )
    ordered_hash = index.get("ordered_record_sha256")
    success_hash = success.get("ordered_record_sha256")
    if ordered_hash and success_hash and str(ordered_hash) != str(success_hash):
        raise RuntimeError(
            f"Token cache identity mismatch between index.pt and _SUCCESS: {cache_dir}"
        )
    if ordered_hash or success_hash:
        digest = str(ordered_hash or success_hash)
        if not _is_sha256_digest(digest):
            raise RuntimeError(
                f"Token cache has a malformed ordered-record identity: {cache_dir}"
            )
        algorithm = str(
            index.get("ordered_record_hash_algorithm")
            or success.get("ordered_record_hash_algorithm")
            or "record-stream-v1"
        )
    else:
        # Legacy caches did not persist a logical record hash. Hash their
        # physical ordered stream plus lengths and payload once so new
        # checkpoints can still bind to that exact cache without rewriting it.
        hasher = hashlib.sha256()
        hasher.update(b"legacy-token-cache-file-v1\0")
        hasher.update(_file_sha256(data_path).encode("ascii"))
        lengths = index.get("lengths")
        if torch.is_tensor(lengths):
            hasher.update(
                lengths.detach().cpu().to(dtype=torch.int64).contiguous().numpy().tobytes()
            )
        hasher.update(json.dumps(
            success.get("cache_payload") or index.get("cache_payload") or {},
            sort_keys=True,
            default=str,
        ).encode("utf-8"))
        digest = hasher.hexdigest()
        algorithm = "legacy-token-cache-file-v1"
    index_tokens_hash = index.get("tokens_sha256")
    success_tokens_hash = success.get("tokens_sha256")
    if (
        index_tokens_hash
        and success_tokens_hash
        and str(index_tokens_hash) != str(success_tokens_hash)
    ):
        raise RuntimeError(
            f"Token cache binary checksum metadata disagrees in {cache_dir}"
        )
    expected_tokens_hash = index_tokens_hash or success_tokens_hash
    if expected_tokens_hash:
        if not _is_sha256_digest(expected_tokens_hash):
            raise RuntimeError(
                f"Token cache has a malformed binary checksum: {cache_dir}"
            )
        actual_tokens_hash = _file_sha256(data_path)
        if actual_tokens_hash != str(expected_tokens_hash):
            raise RuntimeError(
                "Token cache binary checksum failed for "
                f"{data_path}: expected={str(expected_tokens_hash)!r}, "
                f"actual={actual_tokens_hash!r}. Rebuild the cache before training."
            )
    samples = length_count
    identity = {
        "cache_key": str(success.get("cache_key") or index.get("cache_key") or ""),
        "format": str(success.get("format") or index.get("format") or ""),
        "ordered_record_sha256": digest,
        "ordered_record_hash_algorithm": algorithm,
        "samples": samples,
        "cache_payload_sha256": hashlib.sha256(json.dumps(
            success.get("cache_payload") or index.get("cache_payload") or {},
            sort_keys=True,
            default=str,
        ).encode("utf-8")).hexdigest(),
        "audit_sha256": success.get("audit_sha256") or index.get("audit_sha256"),
    }
    # Preserve exact-resume compatibility for pre-checksum v6 caches: only new
    # caches that persisted a verifiable physical digest add this identity field.
    if expected_tokens_hash:
        identity["tokens_sha256"] = str(expected_tokens_hash)
    return identity


def _attach_token_cache_identity(args, cache_dir):
    audit_path = os.path.join(cache_dir, "cache_audit.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as audit_file:
            cache_audit = json.load(audit_file)
        _enforce_cache_audit_budgets(args, cache_audit)
    else:
        print(
            "WARNING: Reused legacy token cache has no cache_audit.json; "
            "schema/rejection/truncation quality budgets cannot be revalidated."
        )
    identity = _read_token_cache_identity(cache_dir)
    args._token_cache_identity = identity
    args._token_cache_dir = os.path.abspath(cache_dir)
    print(
        "INFO: Token cache ordered identity: "
        f"{identity['ordered_record_sha256'][:16]} "
        f"({identity['samples']} samples, {identity['ordered_record_hash_algorithm']})."
    )
    return cache_dir


def materialize_hf_token_cache(args, tokenizer):
    resolve_hf_dataset_revision(args, announce=True)
    _ensure_tokenizer_identity(args, tokenizer)
    cache_root = (
        getattr(args, "hf_token_cache_dir", None)
        or getattr(args, "hf_shard_cache_dir", None)
        or os.path.join(os.getcwd(), ".hierarchos_token_cache")
    )
    cache_key = _hf_token_cache_key(args)
    cache_format = _hf_token_cache_format(args)
    write_loss_weights = _uses_weighted_token_loss(args)
    cache_dir = os.path.join(cache_root, cache_key)
    legacy_cache_dir = os.path.join(cache_root, _legacy_hf_token_cache_key(args))
    success_path = os.path.join(cache_dir, "_SUCCESS")
    index_path = os.path.join(cache_dir, "index.pt")
    data_path = os.path.join(cache_dir, "tokens.bin")

    if os.path.exists(legacy_cache_dir) and legacy_cache_dir != cache_dir:
        print(
            "INFO: Ignoring legacy HF token cache built before all-token "
            f"coherence cache versioning: {legacy_cache_dir}"
        )

    if (
        not getattr(args, "refresh_hf_token_cache", False)
        and os.path.exists(success_path)
        and os.path.exists(index_path)
        and os.path.exists(data_path)
    ):
        print(f"INFO: Reusing HF random-access token cache from {cache_dir}")
        return _attach_token_cache_identity(args, cache_dir)

    tmp_dir = cache_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if getattr(args, "refresh_hf_token_cache", False) and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    if write_loss_weights:
        print(
            "INFO: Building weighted HF random-access token cache "
            "(lossless float32-palette RLE loss weights)."
        )
    else:
        print("INFO: Building HF random-access token cache...")
    build_batch_size = max(
        1,
        int(
            getattr(args, "token_cache_build_batch_size", 0)
            or min(512, max(64, int(getattr(args, "batch_size", 1) or 1) * 4))
        ),
    )
    build_workers = max(0, int(getattr(args, "num_workers", 0) or 0))
    hf_dataset = load_hf_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
        streaming=False,
        revision=_hf_revision_for_load(args),
        num_proc=build_workers,
    )
    try:
        total_samples = len(hf_dataset)
    except Exception:
        total_samples = _estimate_hf_total_for_progress(args)

    precompute_rosa = bool(getattr(args, "use_rosa", True))
    rosa_sentinel = _tokenizer_vocab_size(tokenizer)
    rosa_chunk_size = int(
        getattr(args, "training_chunk_size", DEFAULT_TRAINING_CHUNK_SIZE)
        or DEFAULT_TRAINING_CHUNK_SIZE
    )
    rosa_max_context = int(getattr(args, "rosa_max_context", 512) or 512)
    enforce_rosa_max_context = bool(
        getattr(args, "enforce_rosa_max_context", False)
    )
    cached_rosa_context_mode = rosa_context_mode(enforce_rosa_max_context)
    storage_layout = _token_cache_storage_layout(
        args,
        tokenizer,
        has_rosa_ids=precompute_rosa,
        has_loss_weights=write_loss_weights,
    )
    loss_run_offsets = array("q", [0])
    loss_run_ends = array("i")
    loss_run_codes = array("B")
    loss_weight_palette = (
        torch.tensor(storage_layout["loss_weight_palette"], dtype=torch.float32)
        if write_loss_weights else None
    )
    cache_dataset = HuggingFaceMapStyleDataset(
        hf_dataset,
        tokenizer,
        getattr(args, "max_length", 1024),
        getattr(args, "kayla", False),
        getattr(args, "text_column", None),
        getattr(args, "prompt_column", None),
        getattr(args, "completion_column", None),
        getattr(args, "alpaca", False),
        getattr(args, "train_prompt_tokens", True),
        **_loss_weight_kwargs(args),
        **_response_quality_kwargs(args),
        precompute_rosa=precompute_rosa,
        rosa_vocab_size=rosa_sentinel,
        rosa_chunk_size=rosa_chunk_size,
        rosa_max_context=rosa_max_context,
        enforce_rosa_max_context=enforce_rosa_max_context,
        include_audit=True,
        audit_source_default=getattr(args, "hf_dataset", None),
    )
    builder = create_map_style_dataloader(
        cache_dataset,
        build_batch_size,
        tokenizer.pad_token_id,
        num_workers=build_workers,
        shuffle=False,
        use_length_bucketing=False,
        device=torch.device("cpu"),
        prefetch_factor=getattr(args, "prefetch_factor", None),
        in_order=True,
    )
    print(
        "INFO: Parallel HF cache preprocessing configured with "
        f"batch_size={build_batch_size}, workers={build_workers}."
    )

    offsets = array("q")
    lengths = array("i")
    cache_audit = _new_cache_audit()
    ordered_record_hasher = hashlib.sha256()
    ordered_record_hasher.update(b"hierarchos-token-cache-record-stream-v1\0")
    tokens_hasher = hashlib.sha256()
    total_bytes = 0
    write_buffer = bytearray()
    flush_bytes = max(
        1 << 20,
        int(getattr(args, "token_cache_write_buffer_mb", 64) or 64) * (1 << 20),
    )
    tmp_data_path = os.path.join(tmp_dir, "tokens.bin")

    try:
        with open(tmp_data_path, "wb", buffering=flush_bytes) as data_file:
            for batch in tqdm(
                builder,
                desc="Tokenizing HF cache",
                unit="batch",
                total=(
                    _steps_from_samples(total_samples, build_batch_size)
                    if total_samples is not None else None
                ),
            ):
                if batch is None:
                    continue
                _consume_cache_audit(batch, cache_audit)
                if "input_ids" not in batch:
                    continue
                batch_lengths = _batch_effective_lengths(batch)
                cache_batch = _prepare_token_cache_batch(
                    batch,
                    batch_lengths,
                    layout=storage_layout,
                    precompute_rosa=precompute_rosa,
                )
                for row, length in enumerate(batch_lengths):
                    length = max(0, int(length))
                    if length <= 0:
                        continue
                    offsets.append(total_bytes)
                    lengths.append(length)
                    total_bytes += _append_token_cache_record(
                        write_buffer,
                        cache_batch,
                        row,
                        length,
                        layout=storage_layout,
                        precompute_rosa=precompute_rosa,
                        identity_hasher=ordered_record_hasher,
                    )
                    if write_loss_weights:
                        _append_loss_weight_runs(
                            cache_batch,
                            row,
                            length,
                            palette=loss_weight_palette,
                            run_offsets=loss_run_offsets,
                            run_ends=loss_run_ends,
                            run_codes=loss_run_codes,
                        )

                    if len(write_buffer) >= flush_bytes:
                        tokens_hasher.update(write_buffer)
                        data_file.write(write_buffer)
                        write_buffer.clear()
            if write_buffer:
                tokens_hasher.update(write_buffer)
                data_file.write(write_buffer)
                write_buffer.clear()
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        del builder

    skipped = int(cache_audit.get("rejected", 0) or 0)
    if total_samples is not None and int(cache_audit.get("records", 0) or 0) < int(total_samples):
        missing_records = max(0, int(total_samples) - int(cache_audit.get("records", 0) or 0))
        skipped += missing_records
        cache_audit["records"] += missing_records
        cache_audit["rejected"] += missing_records
        cache_audit["rejection_reasons"]["missing_from_loader"] = missing_records

    if not lengths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("HF token cache build produced no usable samples.")

    ordered_record_sha256 = ordered_record_hasher.hexdigest()
    tokens_sha256 = tokens_hasher.hexdigest()
    audit_path = _persist_and_enforce_cache_audit(
        args,
        cache_audit,
        tmp_dir=tmp_dir,
        cache_dir=cache_dir,
    )
    audit_sha256 = _file_sha256(audit_path)

    index = {
        "format": cache_format,
        "formatter": HF_CACHE_FORMATTER_VERSION,
        "cache_key": cache_key,
        "cache_payload": _hf_cache_key_payload(args, format_name=cache_format),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
        "has_loss_weights": write_loss_weights,
        "loss_weight_dtype": None,
        "has_rosa_ids": precompute_rosa,
        "rosa_sentinel": int(rosa_sentinel),
        "rosa_max_context": int(rosa_max_context),
        "enforce_rosa_max_context": enforce_rosa_max_context,
        "rosa_ids_context_mode": cached_rosa_context_mode,
        "rosa_training_chunk_size": int(rosa_chunk_size),
        "ordered_record_sha256": ordered_record_sha256,
        "ordered_record_hash_algorithm": "record-stream-v1",
        "tokens_sha256": tokens_sha256,
        "audit_sha256": audit_sha256,
    }
    index.update(_token_cache_index_layout(
        storage_layout,
        loss_run_offsets,
        loss_run_ends,
        loss_run_codes,
    ))
    torch.save(index, os.path.join(tmp_dir, "index.pt"))
    with open(os.path.join(tmp_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({
            "format": cache_format,
            "formatter": HF_CACHE_FORMATTER_VERSION,
            "cache_key": cache_key,
            "cache_payload": _hf_cache_key_payload(args, format_name=cache_format),
            "samples": len(lengths),
            "skipped": skipped,
            "bytes": total_bytes,
            "max_length": getattr(args, "max_length", 1024),
            "train_prompt_tokens": bool(getattr(args, "train_prompt_tokens", True)),
            "has_loss_weights": write_loss_weights,
            "loss_weight_dtype": None,
            "loss_weight_encoding": storage_layout["loss_weight_encoding"],
            "loss_weight_palette": storage_layout["loss_weight_palette"],
            "storage_schema_version": storage_layout["storage_schema_version"],
            "token_dtype": storage_layout["token_dtype"],
            "label_dtype": storage_layout["label_dtype"],
            "label_encoding": storage_layout["label_encoding"],
            "rosa_dtype": storage_layout["rosa_dtype"],
            "has_rosa_ids": precompute_rosa,
            "rosa_max_context": int(rosa_max_context),
            "enforce_rosa_max_context": enforce_rosa_max_context,
            "rosa_ids_context_mode": cached_rosa_context_mode,
            "rosa_training_chunk_size": int(rosa_chunk_size),
            "ordered_record_sha256": ordered_record_sha256,
            "ordered_record_hash_algorithm": "record-stream-v1",
            "tokens_sha256": tokens_sha256,
            "audit_sha256": audit_sha256,
            "audit_file": "cache_audit.json",
            "audit": {
                "records": cache_audit["records"],
                "accepted": cache_audit["accepted"],
                "rejected": cache_audit["rejected"],
                "truncated": cache_audit["truncated"],
                "retained_tokens": cache_audit["retained_tokens"],
                "weighted_tokens": cache_audit["weighted_tokens"],
            },
        }, f, indent=2)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    _replace_cache_directory_atomically(tmp_dir, cache_dir)
    print(
        f"INFO: HF random-access token cache ready in {cache_dir} "
        f"({len(lengths)} samples, skipped {skipped}, {total_bytes / (1024 ** 3):.2f} GiB)."
    )
    return _attach_token_cache_identity(args, cache_dir)


def _local_token_cache_payload(args):
    source_arg = os.path.abspath(os.path.expanduser(args.train))
    source_is_directory = os.path.isdir(source_arg)
    source_root = source_arg if source_is_directory else os.path.dirname(source_arg)
    source_files = []
    for path in _jsonl_source_files(args.train):
        resolved = os.path.abspath(os.path.expanduser(path))
        stat = os.stat(resolved)
        source_files.append({
            # Keep cache keys portable across a CPU preprocessing machine and
            # the rented GPU host. Size/mtime plus the source-relative shard
            # layout still invalidates ordinary edits without hashing 5B tokens.
            "relative_path": os.path.normcase(os.path.relpath(resolved, source_root)),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    payload = _hf_cache_key_payload(
        args,
        format_name=_hf_token_cache_format(args),
    )
    payload.pop("dataset", None)
    payload.pop("config", None)
    payload.pop("split", None)
    payload["source_kind"] = "local-jsonl"
    payload["source_layout"] = "directory" if source_is_directory else "single-file"
    payload["source_files"] = source_files
    return payload


def materialize_local_token_cache(args, tokenizer):
    """Build/reuse the mmap token cache for local JSONL exactly once."""
    _ensure_tokenizer_identity(args, tokenizer)
    cache_root = (
        getattr(args, "local_token_cache_dir", None)
        or getattr(args, "hf_token_cache_dir", None)
        or os.path.join(os.getcwd(), ".hierarchos_token_cache")
    )
    payload = _local_token_cache_payload(args)
    cache_key = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = os.path.join(cache_root, "local-" + cache_key)
    success_path = os.path.join(cache_dir, "_SUCCESS")
    index_path = os.path.join(cache_dir, "index.pt")
    data_path = os.path.join(cache_dir, "tokens.bin")

    if (
        not getattr(args, "refresh_local_token_cache", False)
        and os.path.exists(success_path)
        and os.path.exists(index_path)
        and os.path.exists(data_path)
    ):
        print(f"INFO: Reusing local JSONL random-access token cache from {cache_dir}")
        return _attach_token_cache_identity(args, cache_dir)

    tmp_dir = cache_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if getattr(args, "refresh_local_token_cache", False) and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    build_batch_size = max(
        1,
        int(
            getattr(args, "token_cache_build_batch_size", 0)
            or min(512, max(64, int(getattr(args, "batch_size", 1) or 1) * 4))
        ),
    )
    requested_build_workers = max(0, int(getattr(args, "num_workers", 0) or 0))
    # Iterable JSONL workers own disjoint byte ranges/shards, so changing the
    # worker count changes interleaving even with DataLoader FIFO delivery.
    # Use one deterministic producer for cache creation; training remains
    # multi-worker over the finished random-access cache.
    build_workers = 0
    if requested_build_workers:
        print(
            "INFO: Local token-cache build uses one deterministic producer "
            f"(requested training workers={requested_build_workers})."
        )
    text_column, prompt_column, completion_column = _local_text_columns(args)
    precompute_rosa = bool(getattr(args, "use_rosa", True))
    rosa_sentinel = _tokenizer_vocab_size(tokenizer)
    rosa_chunk_size = int(
        getattr(args, "training_chunk_size", DEFAULT_TRAINING_CHUNK_SIZE)
        or DEFAULT_TRAINING_CHUNK_SIZE
    )
    rosa_max_context = int(getattr(args, "rosa_max_context", 512) or 512)
    enforce_rosa_max_context = bool(
        getattr(args, "enforce_rosa_max_context", False)
    )
    cached_rosa_context_mode = rosa_context_mode(enforce_rosa_max_context)
    write_loss_weights = _uses_weighted_token_loss(args)
    storage_layout = _token_cache_storage_layout(
        args,
        tokenizer,
        has_rosa_ids=precompute_rosa,
        has_loss_weights=write_loss_weights,
    )
    loss_run_offsets = array("q", [0])
    loss_run_ends = array("i")
    loss_run_codes = array("B")
    loss_weight_palette = (
        torch.tensor(storage_layout["loss_weight_palette"], dtype=torch.float32)
        if write_loss_weights else None
    )
    print(
        "INFO: Building local JSONL random-access token cache with "
        f"batch_size={build_batch_size}, workers={build_workers}..."
    )
    builder = create_dataloader_for_jsonl(
        args.train,
        tokenizer,
        args.max_length,
        build_batch_size,
        tokenizer.pad_token_id,
        num_workers=build_workers,
        kayla_mode=args.kayla,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=getattr(args, "alpaca", False),
        train_prompt_tokens=getattr(args, "train_prompt_tokens", True),
        **_loss_weight_kwargs(args),
        **_response_quality_kwargs(args),
        precompute_rosa=precompute_rosa,
        rosa_vocab_size=rosa_sentinel,
        rosa_chunk_size=rosa_chunk_size,
        rosa_max_context=rosa_max_context,
        enforce_rosa_max_context=enforce_rosa_max_context,
        use_length_bucketing=False,
        device=torch.device("cpu"),
        prefetch_factor=getattr(args, "prefetch_factor", None),
        in_order=True,
        include_audit=True,
        audit_source_default=os.path.abspath(os.path.expanduser(args.train)),
    )

    offsets = array("q")
    lengths = array("i")
    cache_audit = _new_cache_audit()
    ordered_record_hasher = hashlib.sha256()
    ordered_record_hasher.update(b"hierarchos-token-cache-record-stream-v1\0")
    tokens_hasher = hashlib.sha256()
    total_bytes = 0
    write_buffer = bytearray()
    flush_bytes = max(
        1 << 20,
        int(getattr(args, "token_cache_write_buffer_mb", 64) or 64) * (1 << 20),
    )
    total_hint = getattr(args, "dataset_size", None)
    tmp_data_path = os.path.join(tmp_dir, "tokens.bin")

    try:
        with open(tmp_data_path, "wb", buffering=flush_bytes) as data_file:
            progress = tqdm(
                builder,
                desc="Caching local JSONL",
                unit="batch",
                total=(
                    _steps_from_samples(total_hint, build_batch_size)
                    if total_hint else None
                ),
            )
            for batch in progress:
                if batch is None:
                    continue
                _consume_cache_audit(batch, cache_audit)
                if "input_ids" not in batch:
                    continue
                batch_lengths = _batch_effective_lengths(batch)
                cache_batch = _prepare_token_cache_batch(
                    batch,
                    batch_lengths,
                    layout=storage_layout,
                    precompute_rosa=precompute_rosa,
                )
                for row, length in enumerate(batch_lengths):
                    length = max(0, int(length))
                    if length <= 0:
                        continue
                    offsets.append(total_bytes)
                    lengths.append(length)
                    total_bytes += _append_token_cache_record(
                        write_buffer,
                        cache_batch,
                        row,
                        length,
                        layout=storage_layout,
                        precompute_rosa=precompute_rosa,
                        identity_hasher=ordered_record_hasher,
                    )
                    if write_loss_weights:
                        _append_loss_weight_runs(
                            cache_batch,
                            row,
                            length,
                            palette=loss_weight_palette,
                            run_offsets=loss_run_offsets,
                            run_ends=loss_run_ends,
                            run_codes=loss_run_codes,
                        )

                    if len(write_buffer) >= flush_bytes:
                        tokens_hasher.update(write_buffer)
                        data_file.write(write_buffer)
                        write_buffer.clear()
            if write_buffer:
                tokens_hasher.update(write_buffer)
                data_file.write(write_buffer)
                write_buffer.clear()
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        del builder

    if not lengths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Local JSONL token cache build produced no usable samples.")

    ordered_record_sha256 = ordered_record_hasher.hexdigest()
    tokens_sha256 = tokens_hasher.hexdigest()
    audit_path = _persist_and_enforce_cache_audit(
        args,
        cache_audit,
        tmp_dir=tmp_dir,
        cache_dir=cache_dir,
    )
    audit_sha256 = _file_sha256(audit_path)

    index = {
        "format": _hf_token_cache_format(args),
        "formatter": HF_CACHE_FORMATTER_VERSION,
        "cache_key": cache_key,
        "cache_payload": payload,
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
        "has_loss_weights": write_loss_weights,
        "loss_weight_dtype": None,
        "has_rosa_ids": precompute_rosa,
        "rosa_sentinel": int(rosa_sentinel),
        "rosa_max_context": int(rosa_max_context),
        "enforce_rosa_max_context": enforce_rosa_max_context,
        "rosa_ids_context_mode": cached_rosa_context_mode,
        "rosa_training_chunk_size": int(rosa_chunk_size),
        "ordered_record_sha256": ordered_record_sha256,
        "ordered_record_hash_algorithm": "record-stream-v1",
        "tokens_sha256": tokens_sha256,
        "audit_sha256": audit_sha256,
    }
    index.update(_token_cache_index_layout(
        storage_layout,
        loss_run_offsets,
        loss_run_ends,
        loss_run_codes,
    ))
    torch.save(index, os.path.join(tmp_dir, "index.pt"))
    with open(os.path.join(tmp_dir, "_SUCCESS"), "w", encoding="utf-8") as success_file:
        json.dump({
            "format": _hf_token_cache_format(args),
            "formatter": HF_CACHE_FORMATTER_VERSION,
            "cache_key": cache_key,
            "cache_payload": payload,
            "samples": len(lengths),
            "bytes": total_bytes,
            "build_batch_size": build_batch_size,
            "build_workers": build_workers,
            "has_loss_weights": write_loss_weights,
            "loss_weight_dtype": None,
            "loss_weight_encoding": storage_layout["loss_weight_encoding"],
            "loss_weight_palette": storage_layout["loss_weight_palette"],
            "storage_schema_version": storage_layout["storage_schema_version"],
            "token_dtype": storage_layout["token_dtype"],
            "label_dtype": storage_layout["label_dtype"],
            "label_encoding": storage_layout["label_encoding"],
            "rosa_dtype": storage_layout["rosa_dtype"],
            "has_rosa_ids": precompute_rosa,
            "rosa_max_context": int(rosa_max_context),
            "rosa_training_chunk_size": int(rosa_chunk_size),
            "enforce_rosa_max_context": enforce_rosa_max_context,
            "rosa_ids_context_mode": cached_rosa_context_mode,
            "ordered_record_sha256": ordered_record_sha256,
            "ordered_record_hash_algorithm": "record-stream-v1",
            "tokens_sha256": tokens_sha256,
            "audit_sha256": audit_sha256,
            "audit_file": "cache_audit.json",
            "audit": {
                "records": cache_audit["records"],
                "accepted": cache_audit["accepted"],
                "rejected": cache_audit["rejected"],
                "truncated": cache_audit["truncated"],
                "retained_tokens": cache_audit["retained_tokens"],
                "weighted_tokens": cache_audit["weighted_tokens"],
            },
        }, success_file, indent=2)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    _replace_cache_directory_atomically(tmp_dir, cache_dir)
    print(
        f"INFO: Local JSONL token cache ready in {cache_dir} "
        f"({len(lengths)} samples, {total_bytes / (1024 ** 3):.2f} GiB)."
    )
    return _attach_token_cache_identity(args, cache_dir)


def create_hf_training_dataloader(args, tokenizer, device):
    resolve_hf_dataset_revision(args, announce=not bool(
        getattr(args, "_resolved_hf_dataset_revision", None)
    ))

    def create_indexed_hf_dataloader(reason=None):
        if reason:
            print(f"INFO: {reason}")
        hf_dataset = load_hf_dataset(
            args.hf_dataset,
            args.hf_dataset_config,
            split=args.hf_dataset_split,
            revision=_hf_revision_for_load(args),
        )
        dataset = HuggingFaceMapStyleDataset(
            hf_dataset,
            tokenizer,
            args.max_length,
            args.kayla,
            args.text_column,
            args.prompt_column,
            args.completion_column,
            getattr(args, "alpaca", False),
            getattr(args, "train_prompt_tokens", True),
            **_loss_weight_kwargs(args),
            **_response_quality_kwargs(args),
        )
        return create_map_style_dataloader(
            dataset,
            args.batch_size,
            tokenizer.pad_token_id,
            args.num_workers,
            use_length_bucketing=_length_bucketing_enabled(args),
            bucket_size=getattr(args, "length_bucket_size", None),
            device=device,
            prefetch_factor=args.prefetch_factor,
        )

    if getattr(args, "hf_token_cache", False):
        cache_dir = materialize_hf_token_cache(args, tokenizer)
        cache_length_bucketing = _length_bucketing_enabled(args, default=True)
        if (
            cache_length_bucketing
            and getattr(args, "length_bucket_size", None) is None
            and getattr(args, "auto_length_bucket_size", True)
        ):
            auto_tune_length_bucket_size_from_token_cache(args, cache_dir)
        if cache_length_bucketing:
            print("INFO: Using HF random-access token cache with automatic per-step length bucketing.")
        else:
            print("INFO: Using HF random-access token cache without length bucketing.")
        return create_dataloader_for_tokenized_cache(
            cache_dir,
            args.max_length,
            args.batch_size,
            tokenizer.pad_token_id,
            num_workers=args.num_workers,
            use_length_bucketing=cache_length_bucketing,
            bucket_size=getattr(args, "length_bucket_size", None),
            device=device,
            prefetch_factor=args.prefetch_factor,
        )

    if getattr(args, "streaming_datasets", True):
        try:
            hf_dataset = load_hf_dataset(
                args.hf_dataset,
                args.hf_dataset_config,
                split=args.hf_dataset_split,
                streaming=True,
                revision=_hf_revision_for_load(args),
            )
            hf_num_workers = args.num_workers
            hf_num_shards = getattr(hf_dataset, "num_shards", None)
            try:
                hf_num_shards = int(hf_num_shards) if hf_num_shards is not None else None
            except (TypeError, ValueError):
                hf_num_shards = None
            if hf_num_shards is not None and hf_num_shards > 0 and hf_num_workers > hf_num_shards:
                if getattr(args, "hf_auto_shard", False) and hf_num_workers > 1:
                    shard_dir = materialize_hf_dataset_shards(args, tokenizer, hf_num_workers)
                    print(
                        "INFO: Using local tokenized PT shard cache for HF dataset "
                        f"({hf_num_workers} worker shard(s))."
                    )
                    return create_dataloader_pt_chunked(
                        shard_dir,
                        args.max_length,
                        args.batch_size,
                        hf_num_workers,
                        use_length_bucketing=_length_bucketing_enabled(args),
                        bucket_size=getattr(args, "length_bucket_size", None),
                        device=device,
                        prefetch_factor=args.prefetch_factor,
                        cache_size=max(getattr(args, "pt_cache_size", 2), hf_num_workers),
                    )
                return create_indexed_hf_dataloader(
                    "HF streaming exposes "
                    f"{hf_num_shards} physical shard(s), but --num_workers={hf_num_workers}. "
                    "Using indexed HF loading with length bucketing so PyTorch can use all requested workers."
                )
            dataloader = create_dataloader_for_hf_streaming(
                hf_dataset,
                tokenizer,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=hf_num_workers,
                kayla_mode=args.kayla,
                text_column=args.text_column,
                prompt_column=args.prompt_column,
                completion_column=args.completion_column,
                alpaca_mode=getattr(args, "alpaca", False),
                train_prompt_tokens=getattr(args, "train_prompt_tokens", True),
                **_loss_weight_kwargs(args),
                **_response_quality_kwargs(args),
                use_length_bucketing=_length_bucketing_enabled(args),
                bucket_size=getattr(args, "length_bucket_size", None),
                shuffle_buffer_size=getattr(args, "hf_streaming_shuffle_buffer", 10000),
                device=device,
                prefetch_factor=args.prefetch_factor,
            )
            print("INFO: Hugging Face dataset streaming enabled.")
            return dataloader
        except Exception as exc:
            print(f"WARNING: HF streaming loader failed ({exc}); falling back to map-style loading.")

    return create_indexed_hf_dataloader()


def create_local_training_dataloader(args, tokenizer, device):
    text_column, prompt_column, completion_column = _local_text_columns(args)
    if getattr(args, "streaming_datasets", True) and _is_jsonl_source(args.train):
        if getattr(args, "local_token_cache", False):
            cache_dir = materialize_local_token_cache(args, tokenizer)
            cache_length_bucketing = _length_bucketing_enabled(args, default=True)
            if (
                cache_length_bucketing
                and getattr(args, "length_bucket_size", None) is None
                and getattr(args, "auto_length_bucket_size", True)
            ):
                auto_tune_length_bucket_size_from_token_cache(args, cache_dir)
            print("INFO: Using local JSONL random-access binary token cache.")
            return create_dataloader_for_tokenized_cache(
                cache_dir,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=args.num_workers,
                use_length_bucketing=cache_length_bucketing,
                bucket_size=getattr(args, "length_bucket_size", None),
                device=device,
                prefetch_factor=args.prefetch_factor,
            )
        print("INFO: Local JSONL dataset streaming enabled.")
        return create_dataloader_for_jsonl(
            args.train,
            tokenizer,
            args.max_length,
            args.batch_size,
            tokenizer.pad_token_id,
            num_workers=args.num_workers,
            kayla_mode=args.kayla,
            text_column=text_column,
            prompt_column=prompt_column,
            completion_column=completion_column,
            alpaca_mode=getattr(args, "alpaca", False),
            train_prompt_tokens=getattr(args, "train_prompt_tokens", True),
            **_loss_weight_kwargs(args),
            **_response_quality_kwargs(args),
            use_length_bucketing=_length_bucketing_enabled(args),
            bucket_size=getattr(args, "length_bucket_size", None),
            device=device,
            prefetch_factor=args.prefetch_factor,
        )

    dataset = OriginalJSONLDataset(
        args.train,
        tokenizer,
        args.max_length,
        args.kayla,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=getattr(args, "alpaca", False),
        train_prompt_tokens=getattr(args, "train_prompt_tokens", True),
        **_loss_weight_kwargs(args),
        **_response_quality_kwargs(args),
    )
    return create_map_style_dataloader(
        dataset,
        args.batch_size,
        tokenizer.pad_token_id,
        args.num_workers,
        use_length_bucketing=_length_bucketing_enabled(args),
        bucket_size=getattr(args, "length_bucket_size", None),
        device=device,
        prefetch_factor=args.prefetch_factor,
    )


def configure_tokenizer_length(tokenizer, max_length):
    """Keep tokenizer warnings aligned with Hierarchos' runtime context length."""
    if tokenizer is None or not max_length:
        return

    try:
        tokenizer.model_max_length = int(max_length)
        if hasattr(tokenizer, "init_kwargs"):
            tokenizer.init_kwargs["model_max_length"] = int(max_length)
    except Exception:
        pass


def resolve_tokenizer_source(tokenizer_path=None, model_path=None):
    """Resolve tokenizer assets without treating a direct .pt file as a HF directory."""
    if tokenizer_path:
        return tokenizer_path
    if model_path:
        resolved = os.path.abspath(os.path.expanduser(model_path))
        if os.path.isfile(resolved):
            return os.path.dirname(resolved)
        if os.path.isdir(resolved):
            tokenizer_markers = ("tokenizer.json", "tokenizer_config.json")
            if any(os.path.isfile(os.path.join(resolved, name)) for name in tokenizer_markers):
                return resolved
            nested = []
            for name in sorted(os.listdir(resolved)):
                candidate = os.path.join(resolved, name)
                if os.path.isdir(candidate) and any(
                    os.path.isfile(os.path.join(candidate, marker))
                    for marker in tokenizer_markers
                ):
                    nested.append(candidate)
            if len(nested) == 1:
                return nested[0]
        return model_path
    return "openai-community/gpt2"


def resolve_export_tokenizer_source(explicit_tokenizer_path, config):
    """Resolve the exact tokenizer recorded by training for inference export."""
    config = config or {}
    return (
        explicit_tokenizer_path
        or config.get("tokenizer_path")
        or config.get("tokenizer_name")
        or "openai-community/gpt2"
    )


def validate_tokenizer_vocab(tokenizer, config, source: str = "model"):
    tokenizer_vocab = len(tokenizer)
    model_vocab = config.get("vocab_size") if isinstance(config, dict) else getattr(config, "vocab_size", None)
    if model_vocab is not None and int(model_vocab) != int(tokenizer_vocab):
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch for {source}: tokenizer={tokenizer_vocab}, "
            f"checkpoint={int(model_vocab)}. Use the exact tokenizer from training."
        )


def resolve_num_workers(requested_workers, device, batch_size, *, token_cache_only=False):
    requested_workers = int(requested_workers)
    if requested_workers >= 0:
        return requested_workers

    cpu_count = os.cpu_count() or 1
    if getattr(device, "type", "cpu") == "cuda":
        # 96GB Blackwell-class runs with token caches and batch>=64 can keep the
        # GPU busier with a modestly wider loader pool. Worker threads are capped
        # so pinned-memory queues stay bounded and CPU preprocessing cannot run
        # away from the training loop.
        target = 8 if int(batch_size or 1) >= 64 else 4
        by_cpu = max(1, cpu_count // 3)
        if cpu_count >= 8:
            by_cpu = max(4, by_cpu)
        by_batch = max(1, int(batch_size or 1) * 2)
        return min(target, by_cpu, by_batch)

    if token_cache_only:
        # Cache-only preprocessing has no model competing for CPU cores. A modest
        # pool speeds tokenizer work while avoiding RAM/thread oversubscription.
        return min(8, max(1, cpu_count - 1), max(1, int(batch_size or 1)))

    # CPU and DirectML training usually want the cores for model math. Users can
    # still override this for tokenizer-heavy Hugging Face datasets.
    return 0


_CONTINUATION_SKIP_CONFIG_KEYS = {
    "mode",
    "model_path",
    "out_dir",
    "resume_from_ckpt",
    "shadow_model_path",
    "epochs",
    "ckpt_input",
    "inf_output",
    "ckpt_tok_path",
    "completed_epoch",
    "training_complete",
    "assistant_recovery",
    "amp_dtype",
    "vocab_size",
    "device",
    "refresh_hf_token_cache",
    "refresh_hf_shards",
    "refresh_local_token_cache",
    "max_ce_loss_for_backward",
    "startup_weight_max_abs",
}

_CONTINUATION_SUMMARY_KEYS = (
    "hf_dataset",
    "hf_dataset_split",
    "alpaca",
    "max_length",
    "training_chunk_size",
    "full_sample_bptt",
    "full_sample_activation_checkpointing",
    "full_sample_checkpoint_segment_size",
    "batch_size",
    "starting_lr",
    "min_lr",
    "warmup_steps",
    "warmup_ratio",
    "ltm_lr",
    "min_ltm_lr",
    "ltm_training_mode",
    "ltm_value_alignment_weight",
    "ltm_value_alignment_stride",
    "ltm_value_alignment_min_updates",
    "ltm_value_alignment_ready_threshold",
    "ltm_value_alignment_ema_decay",
    "ltm_value_writer_max_norm",
    "compile",
    "force_compile",
    "amp",
    "train_prompt_tokens",
    "prompt_loss_weight",
    "response_loss_weight",
    "response_boundary_loss_weight",
    "response_boundary_tokens",
    "min_response_tokens",
    "rwkv_channel_mix_key_clamp",
    "rwkv_channel_mix_deepembed_clamp",
    "drift_norm_clamp",
    "drift_delta_scale",
    "drop_empty_completions",
)


def _explicit_cli_dests(parser, argv):
    option_to_dest = {}
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS:
            continue
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def _parser_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS:
            continue
        defaults[action.dest] = parser.get_default(action.dest)
    return defaults


def _read_model_config_defaults(model_path):
    if not model_path:
        return None, None

    resolved = os.path.abspath(os.path.expanduser(model_path))
    local_checkpoint = None
    local_checkpoint_path = None
    if os.path.exists(resolved):
        try:
            local_checkpoint_path, _ = _resolve_weights_path(resolved)
            local_checkpoint = load_checkpoint_payload_compatible(
                local_checkpoint_path,
                map_location="cpu",
            )
            if isinstance(local_checkpoint, dict):
                if "model_state_dict" in local_checkpoint:
                    state_source = local_checkpoint["model_state_dict"]
                else:
                    state_source = {k: v for k, v in local_checkpoint.items() if torch.is_tensor(v)}
                if state_source:
                    inspected_state = sanitize_model_state_dict(
                        state_source,
                        reset_transient_ltm=False,
                    )
                    _reject_unsupported_rwkv_state_dict(
                        inspected_state,
                        local_checkpoint_path,
                    )
                    if isinstance(local_checkpoint.get("config"), dict):
                        inspected_config = dict(local_checkpoint["config"])
                        _infer_arch_flags_from_state_dict(
                            inspected_config,
                            inspected_state,
                        )
                        validate_checkpoint_architecture_contract(
                            local_checkpoint,
                            inspected_config,
                            local_checkpoint_path,
                        )
        except FileNotFoundError:
            local_checkpoint = None
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Could not safely inspect the continuation checkpoint before dataset "
                f"preparation: {resolved}. {exc}"
            ) from exc

    if local_checkpoint is not None and isinstance(local_checkpoint, dict):
        if isinstance(local_checkpoint.get("config"), dict):
            config = dict(local_checkpoint["config"])
            effective = local_checkpoint.get("effective_training_config")
            if isinstance(effective, dict):
                config.update({
                    key: value
                    for key, value in effective.items()
                    if value is not None
                })
            run_identity = local_checkpoint.get("run_identity")
            if isinstance(run_identity, dict):
                objective = run_identity.get("objective")
                if isinstance(objective, dict):
                    config.update({
                        key: value
                        for key, value in objective.items()
                        if value is not None
                    })
            if "completed_epoch" not in config and local_checkpoint.get("completed_epoch") is not None:
                config["completed_epoch"] = local_checkpoint.get("completed_epoch")
            return config, local_checkpoint_path

    model_dir = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
    for name in ("hierarchos_config.json", "config.json"):
        candidate = os.path.join(model_dir, name)
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return dict(json.load(f)), candidate

    if os.path.isfile(resolved) and resolved.lower().endswith(".pt"):
        try:
            checkpoint = load_checkpoint_payload_compatible(
                resolved,
                map_location="cpu",
            )
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
                config = dict(checkpoint["config"])
                if "completed_epoch" not in config and checkpoint.get("completed_epoch") is not None:
                    config["completed_epoch"] = checkpoint.get("completed_epoch")
                return config, resolved
        except Exception as exc:
            print(f"WARNING: Could not read continuation config from {resolved}: {exc}")

    return None, None


def _hydrate_training_args_from_model_config(args, parser, explicit_dests):
    """Use saved model metadata as defaults for continuation and LoRA tuning."""
    if args.mode not in ("train", "finetune"):
        return

    continuation_source = args.resume_from_ckpt or args.model_path
    if not continuation_source:
        return

    saved_config, source = _read_model_config_defaults(continuation_source)
    if not saved_config:
        return

    defaults = _parser_defaults(parser)
    hydrated = []

    if (
        "architecture_revision" not in saved_config
        and "architecture_revision" not in explicit_dests
    ):
        # Checkpoints predating named revisions are the legacy tensor/function
        # contract. Do not let the new-from-scratch CLI default alter their data
        # preprocessing before the strict loader gets a chance to validate them.
        args.architecture_revision = "legacy-v8"
        hydrated.append("architecture_revision")

    completed_epoch = saved_config.get("completed_epoch")
    completed_epoch_int = None
    if completed_epoch is not None:
        try:
            completed_epoch_int = int(completed_epoch)
        except (TypeError, ValueError):
            completed_epoch_int = None

    if completed_epoch_int is not None and args.resume_from_ckpt:
        args.resume_completed_epoch = completed_epoch_int
    if completed_epoch_int is not None and args.model_path and not args.resume_from_ckpt:
        try:
            args.base_completed_epoch = completed_epoch_int
        except (TypeError, ValueError):
            args.base_completed_epoch = 0

    for key, value in saved_config.items():
        if key in _CONTINUATION_SKIP_CONFIG_KEYS:
            continue
        if key in explicit_dests or not hasattr(args, key) or value is None:
            continue
        if key == "train" and not isinstance(value, str):
            continue

        current = getattr(args, key)
        default = defaults.get(key, None)
        if current is None or current == default:
            setattr(args, key, value)
            hydrated.append(key)

    if not hydrated and getattr(args, "base_completed_epoch", 0) <= 0:
        return

    summary = []
    for key in _CONTINUATION_SUMMARY_KEYS:
        if hasattr(args, key):
            summary.append(f"{key}={getattr(args, key)!r}")
    base_epoch = int(getattr(args, "base_completed_epoch", 0) or 0)
    if base_epoch > 0:
        summary.append(f"base_completed_epoch={base_epoch}")
    print(f"INFO: Hydrated continuation training defaults from {source}.")
    print("INFO: Continuation defaults: " + ", ".join(summary))


def _validate_resume_epoch_target(args):
    if args.mode != "train" or not args.resume_from_ckpt:
        return
    completed = getattr(args, "resume_completed_epoch", None)
    if completed is None:
        return
    if int(getattr(args, "epochs", 0) or 0) <= int(completed):
        print(
            f"ERROR: --resume-from-ckpt is already at completed_epoch={completed}, "
            f"but --epochs={args.epochs}."
        )
        print(
            "       For a true training checkpoint, --epochs is the total target epoch. "
            f"Use --epochs {int(completed) + 3} for three more epochs from epoch {completed}, "
            "or use --model-path with an inference export for a three-epoch continuation."
        )
        sys.exit(1)


def _set_default_if_not_explicit(args, explicit_dests, name, value, hydrated):
    if name in explicit_dests:
        return
    setattr(args, name, value)
    hydrated.append(f"{name}={value!r}")


def _apply_assistant_recovery_defaults(args, explicit_dests):
    if not bool(getattr(args, "assistant_recovery", False)):
        return
    if args.mode != "train":
        print("WARNING: --assistant-recovery only affects train mode; ignoring for this mode.")
        return
    if getattr(args, "kayla", False):
        print("ERROR: --assistant-recovery targets Alpaca instruction/input/output data and cannot be combined with --kayla.")
        sys.exit(1)

    applied = []
    if "alpaca" not in explicit_dests:
        args.alpaca = True
        applied.append("alpaca=True")
    _set_default_if_not_explicit(args, explicit_dests, "epochs", 4, applied)
    _set_default_if_not_explicit(args, explicit_dests, "starting_lr", 6e-5, applied)
    _set_default_if_not_explicit(args, explicit_dests, "min_lr", 1e-6, applied)
    _set_default_if_not_explicit(args, explicit_dests, "warmup_ratio", 0.03, applied)
    _set_default_if_not_explicit(args, explicit_dests, "ltm_lr", 3e-4, applied)
    _set_default_if_not_explicit(args, explicit_dests, "min_ltm_lr", 1e-5, applied)
    _set_default_if_not_explicit(args, explicit_dests, "ltm_training_mode", "read-only", applied)
    if str(getattr(args, "architecture_revision", "") or "").strip().lower().replace("_", "-") in {
        COHERENT_REVISION,
        "coherent",
        "v9",
        "v9-coherent",
    }:
        _set_default_if_not_explicit(
            args,
            explicit_dests,
            "ltm_value_alignment_weight",
            0.01,
            applied,
        )
    _set_default_if_not_explicit(args, explicit_dests, "prompt_loss_weight", 0.10, applied)
    _set_default_if_not_explicit(args, explicit_dests, "response_loss_weight", 1.0, applied)
    _set_default_if_not_explicit(args, explicit_dests, "response_boundary_loss_weight", 2.0, applied)
    _set_default_if_not_explicit(args, explicit_dests, "response_boundary_tokens", 32, applied)
    _set_default_if_not_explicit(args, explicit_dests, "min_response_tokens", 16, applied)
    _set_default_if_not_explicit(args, explicit_dests, "ponder_loss_weight", 0.003, applied)
    _set_default_if_not_explicit(args, explicit_dests, "memory_gate_warmup_steps", 5000, applied)
    if applied:
        print("INFO: Assistant recovery defaults applied: " + ", ".join(applied))


def _apply_benchmark_preset_defaults(args, explicit_dests):
    if args.mode != "benchmark":
        return
    preset = str(getattr(args, "benchmark_preset", "") or "").strip().lower()
    if not preset:
        return
    if preset in ("ally", "handheld", "local"):
        preset = "rog-ally"
        args.benchmark_preset = preset
    if preset != "rog-ally":
        print(f"ERROR: Unknown benchmark preset: {args.benchmark_preset}")
        sys.exit(1)

    applied = []
    if (
        "benchmark_suite" not in explicit_dests
        and "benchmark" not in explicit_dests
        and "eval_tasks" not in explicit_dests
        and not bool(getattr(args, "benchmark_all", False))
        and not getattr(args, "benchmark_suite", None)
        and not getattr(args, "benchmark", None)
        and not getattr(args, "eval_tasks", None)
    ):
        args.benchmark_suite = ["rog-ally"]
        applied.append("benchmark_suite=['rog-ally']")

    _set_default_if_not_explicit(args, explicit_dests, "device", "cpu", applied)
    _set_default_if_not_explicit(args, explicit_dests, "threads", min(8, max(1, os.cpu_count() or 8)), applied)
    _set_default_if_not_explicit(args, explicit_dests, "eval_batch_size", 1, applied)
    _set_default_if_not_explicit(args, explicit_dests, "eval_limit", 25, applied)
    _set_default_if_not_explicit(args, explicit_dests, "max_new_tokens", 64, applied)
    _set_default_if_not_explicit(args, explicit_dests, "benchmark_sequential", True, applied)
    _set_default_if_not_explicit(args, explicit_dests, "benchmark_out_dir", "./benchmark_results/rog_ally", applied)

    if applied:
        print(f"INFO: Benchmark preset '{preset}' applied: " + ", ".join(applied))


def main():
    parser = argparse.ArgumentParser(description="hierarchos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "benchmark", "quantize", "merge-lora", "ckpt-2-inf"], help="Operation mode.")

    # --- Data and Path Arguments ---
    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, nargs='?', default=None, const=True, help="Path to local training data.")
    path_group.add_argument("--hf_dataset", type=str, default=None, help="Name or path to a Hugging Face dataset.")
    path_group.add_argument("--hf_dataset_config", type=str, default=None, help="Optional configuration name for the HF dataset.")
    path_group.add_argument("--hf_dataset_split", type=str, default="train", help="Dataset split to use.")
    path_group.add_argument("--hf-dataset-revision", "--hf_dataset_revision", dest="hf_dataset_revision", type=str, default=None, help="HF dataset commit/tag/branch. Hub datasets are automatically pinned to a commit SHA before token-cache lookup.")
    path_group.add_argument("--text_column", type=str, default=None, help="Column name for text completion data.")
    path_group.add_argument("--prompt_column", type=str, default=None, help="Column name for prompt/instruction.")
    path_group.add_argument("--completion_column", type=str, default=None, help="Column name for completion/response.")
    
    path_group.add_argument("--model-path", type=str, default=None, help="Path to the model directory.")
    path_group.add_argument("--out-dir", type=str, default="./hierarchos_model", help="Directory to save the new model/adapter.")
    path_group.add_argument("--tokenizer-path", type=str, default=None, help="Path or HF name of the tokenizer.") 
    path_group.add_argument(
        "--lora-adapter-path",
        type=str,
        default=None,
        help="[merge-lora] Local PEFT LoRA adapter directory saved as safetensors.",
    )
    path_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Explicitly allow Hugging Face tokenizer repositories to execute "
            "custom Python code. Disabled by default."
        ),
    )
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="Path to a training checkpoint .pt file.")
    path_group.add_argument(
        "--shadow-model-path",
        type=str,
        default=None,
        help="Reserved legacy argument; quantized online learning is unsupported.",
    )

    data_fmt_group = parser.add_mutually_exclusive_group()
    data_fmt_group.add_argument("--pre_chunked_dataset", action="store_true")
    data_fmt_group.add_argument("--pre_pt_dataset", action="store_true")

    # --- Architecture Arguments ---
    arch_group = parser.add_argument_group('Architecture')
    arch_group.add_argument(
        "--architecture-revision",
        choices=("coherent-v9", "legacy-v8"),
        default="coherent-v9",
        help=(
            "Learned-function contract for a new model. coherent-v9 enables "
            "corrected recurrence, hard per-row ACT, bounded ROSA, and shared "
            "factorized token adapters. Loaded checkpoints keep their saved contract."
        ),
    )
    arch_group.add_argument("--context_dim", type=int, default=448)
    arch_group.add_argument("--persistent_dim", type=int, default=128)
    arch_group.add_argument("--ltm_slots", type=int, default=1024)
    arch_group.add_argument("--ltm_key_dim", type=int, default=128)
    arch_group.add_argument("--ltm_val_dim", type=int, default=128)
    arch_group.add_argument(
        "--h_hidden",
        type=int,
        default=None,
        help=(
            "H-module hidden size. Defaults to context_dim; the complete "
            "448/448/448 coherent-v9 model has ~30.2M parameters with the "
            "GPT-2 vocabulary (legacy-v8 was ~232.5M)."
        ),
    )
    arch_group.add_argument(
        "--l_hidden",
        type=int,
        default=None,
        help=(
            "L-module hidden size. Defaults to context_dim; the complete "
            "448/448/448 coherent-v9 model has ~30.2M parameters with the "
            "GPT-2 vocabulary (legacy-v8 was ~232.5M)."
        ),
    )
    arch_group.add_argument("--h_stride", type=int, default=4)
    arch_group.add_argument("--max_h_steps", type=int, default=5)
    arch_group.add_argument("--max_l_steps", type=int, default=5)
    arch_group.add_argument("--ltm_topk", type=int, default=4)
    arch_group.add_argument("--max_length", type=int, default=1024)
    arch_group.add_argument("--auto-max-length", action="store_true")
    arch_group.add_argument("--use-deepembed", dest="use_deepembed", action="store_true", default=True, help="Enable DeepEmbed channel-mix modulation (default).")
    arch_group.add_argument("--no-deepembed", dest="use_deepembed", action="store_false", help="Disable DeepEmbed for ablations or legacy checkpoints.")
    arch_group.add_argument("--deepembed-mode", choices=("legacy-table", "shared-factorized", "off"), default=None, help="Override the revision's DeepEmbed representation for an ablation.")
    arch_group.add_argument("--use-rosa", dest="use_rosa", action="store_true", default=True, help="Enable the ROSA embedding path (default).")
    arch_group.add_argument("--no-rosa", dest="use_rosa", action="store_false", help="Disable ROSA for ablations or legacy checkpoints.")
    arch_group.add_argument("--rosa-embedding-mode", choices=("legacy-table", "shared-factorized", "off"), default=None, help="Override the revision's ROSA token representation for an ablation.")
    arch_group.add_argument("--token-adapter-rank", type=int, default=None, help="Rank of coherent-v9 shared token adapters (default: min(64, context_dim)).")
    arch_group.add_argument("--rosa-max-context", dest="rosa_max_context", type=int, default=512, help="ROSA context bound used when --enforce-rosa-max-context is enabled; legacy checkpoints retain full history.")
    rosa_bound_group = arch_group.add_mutually_exclusive_group()
    rosa_bound_group.add_argument("--enforce-rosa-max-context", dest="enforce_rosa_max_context", action="store_true", default=None, help="Enable versioned, chunk-invariant bounded ROSA state (coherent-v9 default).")
    rosa_bound_group.add_argument("--no-enforce-rosa-max-context", dest="enforce_rosa_max_context", action="store_false", help="Retain unbounded legacy ROSA history.")
    rosa_sentinel_group = arch_group.add_mutually_exclusive_group()
    rosa_sentinel_group.add_argument("--rosa-zero-no-prediction", dest="rosa_zero_no_prediction", action="store_true", default=None, help="Zero-mask the ROSA no-prediction sentinel (coherent-v9 default).")
    rosa_sentinel_group.add_argument("--no-rosa-zero-no-prediction", dest="rosa_zero_no_prediction", action="store_false", help="Use the legacy trainable no-prediction sentinel.")
    arch_group.add_argument("--core-recurrence-version", type=int, choices=(1, 2), default=None, help="Advanced recurrent-state contract override; normally selected by --architecture-revision.")
    arch_group.add_argument("--manager-compute-mode", choices=("soft-act", "hard-masked"), default=None, help="Advanced ACT semantic override; coherent-v9 uses train/inference-parity hard selection.")
    arch_group.add_argument("--min-h-steps", type=int, default=1, help="Minimum manager steps before a coherent-v9 hard halt may be selected.")
    arch_group.add_argument("--rwkv-head-size", "--rwkv_head_size", dest="rwkv_head_size", type=int, default=None, help="RWKV matrix-state head size. Default auto-selects 64 when divisible, else a smaller divisor.")
    arch_group.add_argument("--rwkv-channel-mix-key-clamp", "--rwkv_channel_mix_key_clamp", dest="rwkv_channel_mix_key_clamp", type=float, default=12.0, help="Clamp RWKV channel-mix key preactivation before squared ReLU; 0 disables.")
    arch_group.add_argument("--rwkv-channel-mix-deepembed-clamp", "--rwkv_channel_mix_deepembed_clamp", dest="rwkv_channel_mix_deepembed_clamp", type=float, default=4.0, help="Clamp DeepEmbed's multiplicative RWKV channel-mix modulation before value projection; 0 disables.")

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Global Python/NumPy/PyTorch/DataLoader seed persisted in exact-resume identity.",
    )
    train_group.add_argument("--batch_size", type=int, default=64)
    train_group.add_argument("--accumulation-steps", "--accumulation_steps", dest="accumulation_steps", type=int, default=1)
    train_group.add_argument(
        "--accumulation-normalization",
        choices=("weighted-token", "microbatch"),
        default="weighted-token",
        help=(
            "How accumulated microbatches are combined. weighted-token matches one "
            "batch using supervised loss-weight mass; microbatch preserves the legacy "
            "equal-microbatch objective."
        ),
    )
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6)
    train_group.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=0, help="Optimizer update steps spent linearly warming from --min-lr to --starting-lr. Overrides --warmup-ratio when >0.")
    train_group.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.0, help="Fraction of optimizer updates used for LR warmup before cosine decay.")
    train_group.add_argument("--disable-lr-schedule", action="store_true")
    train_group.add_argument("--ltm-lr", "--ltm_lr", dest="ltm_lr", type=float, default=1e-3, help="Maximum LTM inner-update learning rate.")
    train_group.add_argument("--min-ltm-lr", "--min_ltm_lr", dest="min_ltm_lr", type=float, default=None, help="Minimum LTM LR for cosine annealing. Defaults to --min-lr.")
    train_group.add_argument("--disable-ltm-lr-schedule", "--disable_ltm_lr_schedule", dest="disable_ltm_lr_schedule", action="store_true", help="Keep LTM inner-update LR fixed at --ltm-lr.")
    train_group.add_argument(
        "--ltm-training-mode",
        "--ltm_training_mode",
        dest="ltm_training_mode",
        choices=("inner-update", "read-only"),
        default=None,
        help=(
            "How training handles LTM fast memory across TBPTT chunks. "
            "read-only (the coherent-v9 default) carries ROSA/history state "
            "without target-gradient writes and matches normal chat inference; "
            "inner-update is an explicit legacy/Titans ablation."
        ),
    )
    train_group.add_argument(
        "--inference-like-ltm-training",
        dest="ltm_training_mode",
        action="store_const",
        const="read-only",
        help="Shortcut for --ltm-training-mode read-only; recommended for assistant rescue/coherence runs.",
    )
    train_group.add_argument("--rwkv-weight-decay", "--rwkv_weight_decay", dest="rwkv_weight_decay", type=float, default=0.1, help="AdamW decay for RWKV matrices/embeddings; norms/scalars use 0 decay.")
    train_group.add_argument("--ltm-score-grad-scale", "--ltm_score_grad_scale", type=float, default=1.0, help="Straight-through gradient scale for LTM query/key addressing. Set 0 to keep retrieval addressing frozen.")
    train_group.add_argument(
        "--ltm-value-alignment-weight",
        "--ltm_value_alignment_weight",
        dest="ltm_value_alignment_weight",
        type=float,
        default=None,
        help=(
            "Leakage-free auxiliary weight that trains val_proj for bounded online "
            "writes. Fresh coherent-v9 runs default to 0.01; legacy-v8 defaults to 0."
        ),
    )
    train_group.add_argument(
        "--ltm-value-alignment-stride",
        "--ltm_value_alignment_stride",
        dest="ltm_value_alignment_stride",
        type=int,
        default=None,
        help="Train the value writer on every Nth causal token (coherent-v9 default: 8).",
    )
    train_group.add_argument(
        "--ltm-value-alignment-min-updates",
        "--ltm_value_alignment_min_updates",
        dest="ltm_value_alignment_min_updates",
        type=int,
        default=100,
        help="Successful writer-training optimizer updates required before Hebbian writes are enabled (default: 100).",
    )
    train_group.add_argument(
        "--ltm-value-alignment-ready-threshold",
        "--ltm_value_alignment_ready_threshold",
        dest="ltm_value_alignment_ready_threshold",
        type=float,
        default=None,
        help="Maximum finite alignment-loss EMA for declaring the writer ready (v9 default: 0.95).",
    )
    train_group.add_argument(
        "--ltm-value-alignment-ema-decay",
        "--ltm_value_alignment_ema_decay",
        dest="ltm_value_alignment_ema_decay",
        type=float,
        default=None,
        help="EMA decay used by the writer quality gate (v9 default: 0.95).",
    )
    train_group.add_argument(
        "--ltm-value-writer-max-norm",
        "--ltm_value_writer_max_norm",
        dest="ltm_value_writer_max_norm",
        type=float,
        default=None,
        help="Maximum finite val_proj weight norm allowed by the writer readiness gate.",
    )
    format_group = train_group.add_mutually_exclusive_group()
    format_group.add_argument("--kayla", action="store_true")
    format_group.add_argument(
        "--alpaca",
        action="store_true",
        help="Use Alpaca instruction/input/output formatting and default columns instruction/output.",
    )
    train_group.add_argument(
        "--train-prompt-tokens",
        action="store_true",
        dest="train_prompt_tokens",
        default=True,
        help=(
            "Include prompt/instruction tokens in the LM loss for prompt-completion data. "
            "Default: ON, so the LM objective covers all tokens."
        ),
    )
    train_group.add_argument(
        "--mask-prompt-tokens",
        action="store_false",
        dest="train_prompt_tokens",
        help="Legacy SFT behavior: mask prompt/instruction tokens from CE loss.",
    )
    train_group.add_argument(
        "--allow-masked-active-labels",
        action="store_false",
        dest="strict_all_token_loss",
        default=True,
        help=(
            "Disable the trainer fail-fast audit that rejects label=-100 on real "
            "prompt/completion tokens when --train-prompt-tokens is active."
        ),
    )
    train_group.add_argument("--prompt-loss-weight", type=float, default=1.0, help="Per-token CE weight for prompt/instruction/input tokens when prompt tokens are trained.")
    train_group.add_argument("--response-loss-weight", type=float, default=1.0, help="Per-token CE weight for completion/assistant response tokens.")
    train_group.add_argument("--response-boundary-loss-weight", type=float, default=1.0, help="Multiplier for the first N response tokens after ### Response.")
    train_group.add_argument("--response-boundary-tokens", type=int, default=0, help="Number of initial response tokens to multiply by --response-boundary-loss-weight.")
    train_group.add_argument(
        "--min-response-tokens",
        type=int,
        default=1,
        help=(
            "Reject prompt-completion rows whose tokenized assistant output is shorter "
            "than this many non-EOS tokens, and preserve at least this many when truncating."
        ),
    )
    train_group.add_argument("--allow-empty-completions", dest="drop_empty_completions", action="store_false", help="Keep prompt-completion rows with blank completions instead of dropping them.")
    train_group.set_defaults(drop_empty_completions=True)
    train_group.add_argument(
        "--assistant-recovery",
        action="store_true",
        help=(
            "Apply large-assistant SFT defaults for Alpaca training: "
            "4 epochs, warmup+cosine LR, response-weighted loss, boosted response-boundary tokens, "
            "and response-preserving truncation. "
            "Explicit CLI values still override the preset."
        ),
    )
    train_group.add_argument("--lora_r", type=int, default=8)
    train_group.add_argument("--lora_alpha", type=int, default=16)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--max-sanitized-gradient-values", type=int, default=0, help="Deprecated compatibility no-op. Optimizer steps now always reject NaN/Inf gradients; finite gradients are still clipped with --grad-clip.")
    train_group.add_argument("--startup-weight-max-abs", type=float, default=0.0, help="Explicit emergency clamp for finite loaded weights/buffers; changes checkpoint values, so 0 is the compatibility-safe default.")
    train_group.add_argument("--max-ce-loss-for-backward", type=float, default=0.0, help="Clamp finite CE loss used for backward (0 disables; recommended for from-scratch training).")
    train_group.add_argument("--max-commitment-cost-for-backward", type=float, default=2.0, help="Clamp finite commitment cost used for backward to prevent auxiliary loss explosions (0 disables).")
    train_group.add_argument("--max-ponder-cost-for-backward", type=float, default=0.0, help="Clamp finite ponder cost used for backward (0 disables).")
    train_group.add_argument("--halt-logit-clamp", type=float, default=30.0, help="Clamp finite ACT halt logits; NaN/Inf trajectories are rejected.")
    train_group.add_argument("--recurrent-state-clamp", type=float, default=50.0, help="Clamp finite H/L recurrent states; NaN/Inf trajectories are rejected.")
    train_group.add_argument("--context-state-clamp", type=float, default=50.0, help="Clamp finite manager context states; NaN/Inf trajectories are rejected.")
    train_group.add_argument("--drift-state-clamp", type=float, default=5.0, help="Clamp finite worker drift states; NaN/Inf trajectories are rejected.")
    train_group.add_argument("--drift-norm-clamp", type=float, default=0.0, help="Optional L2 norm clamp for worker drift states (0 disables). Useful for commit/drift rescue resumes.")
    train_group.add_argument("--drift-delta-scale", type=float, default=1.0, help="Scale applied to each worker drift update before accumulation. Values below 1 tame commit/drift growth.")
    train_group.add_argument("--activation-clamp", type=float, default=100.0, help="Clamp finite manager/worker activations; NaN/Inf trajectories are rejected.")
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01)
    train_group.add_argument("--commitment-loss-weight", type=float, default=0.5)
    train_group.add_argument("--commitment-threshold", type=float, default=0.05)
    train_group.add_argument("--l_conv_atol", "--l-conv-atol", type=float, default=1e-4, help="Converge tolerance for WorkerLoop. Default: 1e-4.")
    train_group.add_argument("--detach_every_n_steps", "--detach-every-n-steps", type=int, default=32, help="RWKV state detachment frequency. Use 0 to disable; default: 32.")
    train_group.add_argument("--h_halt_thresh", "--h-halt-thresh", type=float, default=0.9, help="H-RNN halt probability threshold. Default: 0.9.")
    train_group.add_argument("--act-depth-temperature", type=float, default=0.05, help="Temperature of the coherent-v9 straight-through hard-depth surrogate; forward ponder cost remains the actual executed manager depth.")
    train_group.add_argument("--encourage-thinking", action="store_true", help="Invert ponder loss to REWARD thinking (for recovery training).")
    adaptive_ponder_group = train_group.add_mutually_exclusive_group()
    adaptive_ponder_group.add_argument("--adaptive-ponder", dest="adaptive_ponder", action="store_true", default=None, help="Match ACT depth to detached CE difficulty (coherent-v9 default).")
    adaptive_ponder_group.add_argument("--no-adaptive-ponder", dest="adaptive_ponder", action="store_false", help="Use a fixed compute penalty instead of difficulty-adaptive ACT.")
    train_group.add_argument("--ponder-target-scale", type=float, default=0.5, help="Scaling factor for adaptive ponder target. Default: 0.5.")
    train_group.add_argument(
        "--ponder-objective",
        choices=("auto", "symmetric-huber", "legacy-one-sided"),
        default=None,
        help="Adaptive ACT target loss. auto selects symmetric-huber for coherent-v9 and the historical one-sided loss for legacy-v8.",
    )
    train_group.add_argument(
        "--ponder-huber-beta",
        type=float,
        default=0.5,
        help="Transition width for the coherent-v9 symmetric adaptive-ponder loss.",
    )
    train_group.add_argument("--no-memory-token-routers", dest="memory_token_routers", action="store_false", help="Disable per-token ROSA/LTM memory routers and use scalar gates only.")
    train_group.add_argument("--memory-gate-warmup-steps", "--memory_gate_warmup_steps", dest="memory_gate_warmup_steps", type=int, default=2000, help="Training batches used to softly keep ROSA/LTM gates open before decaying to learned gates.")
    train_group.add_argument("--memory-gate-warmup-floor", "--memory_gate_warmup_floor", dest="memory_gate_warmup_floor", type=float, default=0.10, help="Initial minimum memory gate floor during warmup. Decays to 0 over --memory-gate-warmup-steps.")
    train_group.add_argument("--reset-halt-bias", type=float, default=None, metavar="BIAS", help="SURGICAL FIX: Reset h_halt_proj.bias to this value on load (e.g., -2.0 for ~12%% halt prob).")
    train_group.add_argument(
        "--override-scheduling",
        "--override_scheduling",
        dest="override_scheduling",
        action="store_true",
        help=(
            "Legacy compatibility alias for BOTH --reset-optimizer-state and "
            "--rebuild-lr-schedule. Prefer the explicit flags."
        ),
    )
    train_group.add_argument(
        "--rebuild-lr-schedule",
        action="store_true",
        help="On resume, keep optimizer/scaler moments but rebuild LR schedules over remaining work.",
    )
    train_group.add_argument(
        "--reset-optimizer-state",
        action="store_true",
        help="On resume, intentionally discard optimizer/scaler moments while preserving the saved LR schedule unless rebuilt explicitly.",
    )
    train_group.add_argument("--persist-state", action="store_true", default=False, help="Persist RNN/LTM states between batches. Default: False.")
    train_group.add_argument("--no-persist-state", dest="persist_state", action="store_false", help="Disable recurrent-value persistence between unrelated DataLoader batches; within-sample recurrence remains active.")
    train_group.add_argument(
        "--training-chunk-size",
        "--training_chunk_size",
        type=int,
        default=None,
        help=(
            "TBPTT chunk size. Revision default: 256 for coherent-v9 and 128 "
            "for legacy-v8; reduce it further if memory gets tight."
        ),
    )
    full_bptt_group = train_group.add_mutually_exclusive_group()
    full_bptt_group.add_argument(
        "--full-sample-bptt",
        "--full_sample_bptt",
        dest="full_sample_bptt",
        action="store_true",
        default=False,
        help=(
            "Use one attached autograd graph across every trimmed sample (exact per-sample BPTT). "
            "Disables recurrent detachment and cross-sample state persistence without changing "
            "training_chunk_size cache/compile metadata."
        ),
    )
    full_bptt_group.add_argument(
        "--no-full-sample-bptt",
        "--no_full_sample_bptt",
        dest="full_sample_bptt",
        action="store_false",
        help="Disable an inherited full-sample BPTT setting and use configured TBPTT chunking.",
    )
    full_checkpoint_group = train_group.add_mutually_exclusive_group()
    full_checkpoint_group.add_argument(
        "--full-sample-activation-checkpointing",
        "--full_sample_activation_checkpointing",
        dest="full_sample_activation_checkpointing",
        action="store_true",
        help=(
            "Recompute attached temporal segments during one full-sample backward "
            "to bound activation VRAM without truncating gradients."
        ),
    )
    full_checkpoint_group.add_argument(
        "--no-full-sample-activation-checkpointing",
        "--no_full_sample_activation_checkpointing",
        dest="full_sample_activation_checkpointing",
        action="store_false",
        help="Keep full-sample activations instead of recomputing them during backward.",
    )
    train_group.add_argument(
        "--full-sample-checkpoint-segment-size",
        "--full_sample_checkpoint_segment_size",
        dest="full_sample_checkpoint_segment_size",
        type=int,
        default=128,
        help=(
            "Tokens per attached activation-checkpoint segment in full-sample BPTT. "
            "This bounds activation memory without detaching gradients and does not "
            "change training_chunk_size cache/ROSA/LTM metadata."
        ),
    )
    train_group.add_argument("--cuda-loss-chunk-rows", "--cuda_loss_chunk_rows", type=int, default=0, help="Rows per lm_head loss chunk on CUDA (0 = auto).")
    train_group.add_argument("--no-cuda-chunked-lm-loss", dest="cuda_chunked_lm_loss", action="store_false", help="Disable CUDA chunked LM loss and return full logits during training.")
    train_group.add_argument("--cpu-loss-chunk-rows", "--cpu_loss_chunk_rows", type=int, default=0, help="Rows per lm_head loss chunk on CPU (0 = all supervised rows).")
    train_group.add_argument("--no-cpu-chunked-lm-loss", dest="cpu_chunked_lm_loss", action="store_false", help="Disable CPU supervised-row LM loss and return full logits during training.")
    train_group.add_argument("--no-ltm-cpu-gather-retrieval", dest="ltm_cpu_gather_retrieval", action="store_false", help="Use the old dense one-hot CPU retrieval path.")
    train_group.add_argument("--no-ltm-cpu-sparse-update", dest="ltm_cpu_sparse_update", action="store_false", help="Use the old dense one-hot CPU LTM update path.")
    train_group.set_defaults(
        cuda_chunked_lm_loss=True,
        cpu_chunked_lm_loss=True,
        ltm_cpu_gather_retrieval=True,
        ltm_cpu_sparse_update=True,
        memory_token_routers=True,
        full_sample_activation_checkpointing=None,
    )
    train_group.add_argument("--debug-numerics", action="store_true", help="Enable per-token NaN/Inf debug checks. Slower on CUDA.")
    train_group.add_argument("--save-steps", type=int, default=0, help="Save a checkpoint/adapter every N steps during training/finetuning (0 to disable).")
    train_group.add_argument("--progress-log-steps", "--progress_log_steps", dest="progress_log_steps", type=int, default=25, help="Update tqdm scalar metrics every N steps (1 = every step).")
    train_group.add_argument("--padding-metric-steps", "--padding_metric_steps", dest="padding_metric_steps", type=int, default=0, help="Report tok_eff only for the first N training steps (-1 = always, 0 = never).")
    train_group.add_argument("--no-padding-metrics", dest="padding_metrics", action="store_false", help="Disable tok_eff/seq padding diagnostics during training.")
    train_group.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers (-1 = auto; CUDA uses prefetched workers, CPU/DML uses 0).")
    train_group.add_argument("--prefetch-factor", "--prefetch_factor", dest="prefetch_factor", type=int, default=None, help="Batches prefetched per DataLoader worker (auto keeps total queued batches tied to worker count).")
    train_group.add_argument("--cuda-prefetch", dest="cuda_prefetch", action="store_true", help="Overlap pinned host-to-device transfer for the next batch on a dedicated CUDA stream.")
    train_group.add_argument("--no-cuda-prefetch", dest="cuda_prefetch", action="store_false", help="Disable dedicated-stream CUDA batch lookahead for diagnostics.")
    train_group.add_argument("--pt-cache-size", "--pt_cache_size", dest="pt_cache_size", type=int, default=2, help="Number of .pt chunk files to keep hot per worker for --pre_pt_dataset.")
    train_group.add_argument("--no-length-bucketing", dest="length_bucketing", action="store_false", help="Disable length-aware batching for training datasets.")
    train_group.add_argument("--length-bucket-size", "--length_bucket_size", dest="length_bucket_size", type=int, default=None, help="Samples per length bucket/window. Manual override; by default CUDA HF token-cache runs auto-tune this from cached sample lengths.")
    train_group.add_argument("--no-auto-length-bucket-size", dest="auto_length_bucket_size", action="store_false", help="Disable startup auto-tuning of HF token-cache length bucket size.")
    train_group.add_argument("--length-bucket-auto-tolerance", "--length_bucket_auto_tolerance", dest="length_bucket_auto_tolerance", type=float, default=0.005, help="Pick the smallest bucket within this absolute token-efficiency margin of the best auto-tuned bucket. Use 0 for max padding reduction.")
    train_group.add_argument("--length-bucket-auto-sample-size", "--length_bucket_auto_sample_size", dest="length_bucket_auto_sample_size", type=int, default=1000000, help="Maximum cached lengths sampled once for bucket auto-tuning (0 = all); the result is persisted in the token cache.")
    train_group.add_argument("--no-streaming-datasets", dest="streaming_datasets", action="store_false", help="Disable streaming for raw JSONL/Hugging Face datasets and use map-style loading.")
    train_group.add_argument("--hf-streaming-shuffle-buffer", "--hf_streaming_shuffle_buffer", dest="hf_streaming_shuffle_buffer", type=int, default=10000, help="Buffered shuffle size for Hugging Face streaming datasets.")
    train_group.add_argument("--hf-auto-shard", dest="hf_auto_shard", action="store_true", help="Opt into local tokenized shard caching for single-shard HF streaming datasets.")
    train_group.add_argument("--no-hf-auto-shard", dest="hf_auto_shard", action="store_false", help="Use indexed HF loading instead of local tokenized shard caching for single-shard HF datasets.")
    train_group.set_defaults(hf_auto_shard=False)
    train_group.set_defaults(length_bucketing=True)
    train_group.set_defaults(auto_length_bucket_size=True)
    train_group.set_defaults(padding_metrics=True)
    train_group.set_defaults(cuda_prefetch=True)
    train_group.add_argument("--hf-shard-cache-dir", "--hf_shard_cache_dir", dest="hf_shard_cache_dir", type=str, default=None, help="Directory for cached local HF tokenized shard files.")
    train_group.add_argument("--refresh-hf-shards", "--refresh_hf_shards", dest="refresh_hf_shards", action="store_true", help="Rebuild cached local HF tokenized shards before training.")
    train_group.add_argument("--hf-cache-chunks-per-file", "--hf_cache_chunks_per_file", dest="hf_cache_chunks_per_file", type=int, default=2048, help="Tokenized samples per cached HF .pt shard chunk.")
    train_group.add_argument("--hf-token-cache", "--hf_token_cache", dest="hf_token_cache", action="store_true", help="Build/reuse a random-access binary token cache for HF datasets before training.")
    train_group.add_argument("--no-hf-token-cache", "--no_hf_token_cache", dest="hf_token_cache", action="store_false", help="Disable the default HF random-access token cache.")
    train_group.add_argument("--hf-token-cache-dir", "--hf_token_cache_dir", dest="hf_token_cache_dir", type=str, default=None, help="Directory for random-access HF token caches.")
    train_group.add_argument("--refresh-hf-token-cache", "--refresh_hf_token_cache", dest="refresh_hf_token_cache", action="store_true", help="Rebuild the random-access HF token cache before training.")
    train_group.add_argument("--local-token-cache", dest="local_token_cache", action="store_true", help="Build/reuse a random-access binary token cache for local JSONL instead of tokenizing every epoch.")
    train_group.add_argument("--no-local-token-cache", dest="local_token_cache", action="store_false", help="Stream and tokenize local JSONL on every epoch without a binary cache.")
    train_group.add_argument("--local-token-cache-dir", "--local_token_cache_dir", dest="local_token_cache_dir", type=str, default=None, help="Directory for fingerprinted local JSONL token caches.")
    train_group.add_argument("--refresh-local-token-cache", "--refresh_local_token_cache", dest="refresh_local_token_cache", action="store_true", help="Rebuild the local JSONL token cache once.")
    train_group.add_argument("--token-cache-build-batch-size", type=int, default=0, help="Local-cache preprocessing batch size (0 = auto, up to 512).")
    train_group.add_argument("--token-cache-write-buffer-mb", type=int, default=64, help="Buffered binary-cache write size in MiB.")
    train_group.add_argument("--token-cache-only", action="store_true", help="Build/reuse the selected token cache and exit before allocating the model (useful on a cheaper CPU machine).")
    train_group.add_argument(
        "--max-cache-rejected-samples",
        type=int,
        default=0,
        help="Absolute rejected-row budget during cache construction (default 0).",
    )
    train_group.add_argument(
        "--max-cache-rejected-fraction",
        type=float,
        default=0.0,
        help="Rejected-row fraction budget; the larger count/fraction budget applies.",
    )
    train_group.add_argument(
        "--max-cache-truncated-samples",
        type=int,
        default=-1,
        help="Absolute overlength/truncated-row budget (-1 uses only the fraction budget).",
    )
    train_group.add_argument(
        "--max-cache-truncated-fraction",
        type=float,
        default=0.05,
        help="Maximum fraction of accepted rows that may be truncated before cache construction fails.",
    )
    train_group.add_argument(
        "--max-skipped-train-batches",
        type=int,
        default=0,
        help="Abort after more than this many OOM/nonfinite/unsupervised training batches.",
    )
    train_group.add_argument("--amp", action="store_true", help="Enable mixed precision (auto-enabled on CUDA).")
    train_group.add_argument("--no-amp", dest="amp", action="store_false", help="Explicitly disable mixed precision.")
    train_group.add_argument("--dataset-size", type=int, default=None, help="Force a specific dataset size (total samples) to calculate steps for the LR scheduler.")
    train_group.set_defaults(hf_token_cache=True)
    train_group.set_defaults(local_token_cache=True)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (auto-enabled on CUDA).")
    parser.add_argument("--no-compile", "--no_compile", dest="compile", action="store_false", help="Explicitly disable torch.compile, including CUDA auto-enable. Useful for stability diagnostics/rescue resumes.")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode for the RWKV hot path. max-autotune-no-cudagraphs keeps autotuned kernels without cudagraph warnings from TBPTT submodule calls.",
    )
    parser.add_argument("--compile-dynamic", action="store_true", help="Compile with dynamic shapes. Slower, but useful if batch/sequence geometry changes often.")
    parser.add_argument("--compile-backend", default=None, help="Optional torch.compile backend override, e.g. eager for diagnostics or inductor for production.")
    parser.add_argument("--compile-fullgraph-worker", action="store_true", help="Require the worker loop to compile as one full graph. Useful for diagnostics; falls back on failure.")
    parser.add_argument("--compile-cudagraphs", dest="compile_cudagraphs", action="store_true", help="Opt into CUDA graph capture inside torch.compile. Pair with --compile-mode max-autotune to test it; it can warn with TBPTT submodule compilation.")
    parser.add_argument("--no-compile-cudagraphs", dest="compile_cudagraphs", action="store_false", help="Disable CUDA graph capture inside torch.compile.")
    parser.add_argument("--no-compile-pad-to-chunk-size", dest="compile_pad_to_chunk_size", action="store_false", help="Disable CUDA compile padding to training_chunk_size multiples.")
    parser.add_argument("--compile-static-worker-loop", dest="compile_static_worker_loop", action="store_true", default=None, help="Force the compile-friendly fixed WorkerLoop even outside CUDA auto-compile.")
    parser.add_argument("--no-compile-static-worker-loop", dest="compile_static_worker_loop", action="store_false", help="Keep the old data-dependent WorkerLoop early break under torch.compile.")
    parser.add_argument("--no-compile-h-rnn", dest="compile_h_rnn", action="store_false", help="Compile only the L/worker hot loop, leaving the H RWKV cell eager.")
    parser.add_argument("--verbose-compile", dest="compile_quiet", action="store_false", help="Let torch.compile/Triton emit verbose autotune logs.")
    parser.set_defaults(
        compile_cudagraphs=False,
        compile_pad_to_chunk_size=True,
        compile_h_rnn=True,
        compile_quiet=True,
    )

    # --- Evaluation Arguments (lm-evaluation-harness) ---
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument("--eval-tasks", type=str, nargs='+', default=None,
        help="Benchmark tasks to run during training (e.g., 'hellaswag arc_easy'). Disabled by default. Requires: pip install lm-eval")
    eval_group.add_argument("--eval-every-epoch", type=int, default=1,
        help="Run evaluation every N epochs (default: 1).")
    eval_group.add_argument("--eval-batch-size", type=int, default=1,
        help="Batch size for evaluation (default: 1).")
    eval_group.add_argument("--eval-limit", type=int, default=None,
        help="Limit samples per task for fast evaluation runs (e.g., 10 for quick tests).")
    eval_group.add_argument("--eval-steps", type=int, default=None,
        help="Run evaluation every N training steps (for quick testing). Triggers periodically.")
    eval_group.add_argument(
        "--best-checkpoint-metric",
        type=str,
        default=None,
        metavar="TASK:METRIC",
        help="Save hierarchos_best.pt only when this finite lm-eval metric improves, e.g. hellaswag:acc_norm,none. Requires --eval-tasks and fails closed if evaluation is unavailable.",
    )
    eval_group.add_argument(
        "--best-checkpoint-mode",
        choices=("max", "min"),
        default="max",
        help="Whether higher or lower values improve --best-checkpoint-metric.",
    )
    eval_group.add_argument("--benchmark-suite", type=str, nargs='+', default=None,
        help="Post-training benchmark suite(s) for benchmark mode. Examples: frontier-text, frontier, math, coding, arc-agi-family.")
    eval_group.add_argument("--benchmark", type=str, nargs='+', default=None,
        help="Individual post-training benchmarks or raw lm-eval task ids. Examples: gpqa-diamond aime25 arc-agi mmlu_pro.")
    eval_group.add_argument("--benchmark-all", action="store_true",
        help="Run every registered benchmark sequentially. Runnable lm-eval tasks are chained one at a time; external benchmarks are reported or run when their local path is provided.")
    eval_group.add_argument("--benchmark-sequential", action="store_true",
        help="Run selected lm-eval benchmarks one at a time and merge the scores into one final terminal report.")
    eval_group.add_argument("--benchmark-preset", type=str, default=None, choices=["rog-ally", "ally", "handheld", "local"],
        help="Apply bounded benchmark defaults for local hardware. 'rog-ally' uses CPU, batch=1, limit=25, sequential tasks, and the rog-ally suite unless tasks are explicit.")
    eval_group.add_argument("--benchmark-out-dir", type=str, default="./benchmark_results",
        help="Directory for post-training benchmark reports.")
    eval_group.add_argument("--benchmark-run-name", type=str, default=None,
        help="Optional report folder name inside --benchmark-out-dir.")
    eval_group.add_argument("--list-benchmarks", action="store_true",
        help="List supported benchmark suites and benchmark aliases, then exit.")
    eval_group.add_argument("--strict-benchmarks", action="store_true",
        help="Validate lm-eval task ids before running and fail if any are missing.")
    eval_group.add_argument("--arc-agi-path", type=str, default=None,
        help="Local ARC-AGI JSON file or directory. Enables runnable ARC-AGI/ARC-AGI-2 benchmark entries.")
    eval_group.add_argument("--arc-agi-max-tasks", type=int, default=None,
        help="Limit local ARC-AGI tasks for quick runs.")
    eval_group.add_argument("--arc-agi-max-test-items", type=int, default=None,
        help="Limit total ARC-AGI test items across loaded tasks.")
    eval_group.add_argument("--arc-agi-keep-samples", action="store_true",
        help="Store ARC-AGI raw generations and parsed grids in results.json.")

    # --- Inference & Sampling ---
    infer_group = parser.add_argument_group('Inference')
    infer_group.add_argument("--temperature", type=float, default=DEFAULT_CHAT_TEMPERATURE)
    infer_group.add_argument("--top-k", type=int, default=40)
    infer_group.add_argument("--top-p", type=float, default=0.9)
    infer_group.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty for repeating tokens (1.0=none, >1.0=discourage). Default: 1.2.")
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--entropy-stop-threshold", type=float, default=0.0, help="Stop chat generation when raw logit entropy is this high and the top token is low-confidence. Default 0 disables this guard.")
    infer_group.add_argument("--entropy-stop-min-tokens", type=int, default=3, help="Minimum generated tokens before entropy stop can trigger.")
    infer_group.add_argument("--entropy-stop-top-prob", type=float, default=0.05, help="Entropy stop only triggers when the top raw token probability is at or below this value.")
    infer_group.add_argument("--eos-stop-prob", type=float, default=0.0, help="Stop once EOS has at least this raw probability after generation has started. Default 0 disables this guard.")
    infer_group.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "dml"])
    infer_group.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
    )
    
    # --- Chat-specific (Online Learning) ---
    chat_group = parser.add_argument_group('Chat Online Learning')
    chat_group.add_argument("--enable-quantized-learning", action="store_true", help="Reserved legacy flag; quantized online learning is unsupported by the active matrix-state architecture.")
    chat_group.add_argument("--ltm-lora-path", type=str, default=None, help="Legacy name for the cumulative, identity-bound online-memory overlay path. Defaults beside the model.")
    chat_group.add_argument("--static-ltm-lr", action="store_true", default=True, help="Use a fixed LR for transactional online-memory updates (default).")
    chat_group.add_argument("--dynamic-ltm-lr", dest="static_ltm_lr", action="store_false", help="Cosine-decay the LR across accepted online-memory transactions.")
    chat_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="Accepted online updates over which dynamic LTM LR decays.")
    chat_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="Min LR for LTM cosine annealing.")
    chat_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="Target %% of params to train (overrides lora_r).")
    chat_group.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    chat_group.add_argument("--passive-learning", action="store_true", default=False, help="Enable passive LTM learning from user prompts/observed context. Default OFF for checkpoint-stable inference.")
    chat_group.add_argument("--no-passive-learning", dest="passive_learning", action="store_false", help="Disable passive LTM learning.")
    chat_group.add_argument("--passive-response-learning", action="store_true", default=False, help="Also allow passive learning from self-generated responses after confidence/quality gates. Default: OFF.")
    chat_group.add_argument(
        "--online-adaptation-policy",
        choices=("off", "validated", "prompt", "prompt+response"),
        default="validated",
        help=(
            "Online-memory policy. validated permits only explicit /learn, /reject, "
            "and /correct actions and is the safe default; passive response learning "
            "requires prompt+response."
        ),
    )
    chat_group.add_argument(
        "--natural-feedback-detection",
        action="store_true",
        default=False,
        help="Opt in to heuristic natural-language praise/correction detection. Default OFF.",
    )
    chat_group.add_argument("--passive-lr", type=float, default=5e-6, help="Learning rate for passive LTM updates (default: 5e-6, very conservative).")
    chat_group.add_argument("--online-ltm-grad-clip", type=float, default=0.75, help="Value/global-norm clip for finite online LTM gradients; non-finite gradients are always rejected.")
    chat_group.add_argument("--online-ltm-lr", type=float, default=1e-3, help="Explicit-feedback fast-memory LR, separate from training inner-update LR.")
    chat_group.add_argument("--online-ltm-max-delta-norm", type=float, default=1.0, help="Transactional fast-state delta norm budget per accepted online update.")
    chat_group.add_argument("--online-ltm-max-fast-norm", type=float, default=64.0, help="Total fast-memory norm budget after an online update.")
    chat_group.add_argument("--online-ltm-max-slot-norm", type=float, default=4.0, help="Per-slot fast-memory norm budget after an online update.")
    chat_group.add_argument("--surprise-threshold", type=float, default=1.0, help="Passive learning only writes when loss <= this threshold (default: 1.0; lower is stricter).")
    chat_group.add_argument("--chat-state-file", type=str, nargs="?", const="auto", default=None, help="Autosave a new tiny model/tokenizer-bound hierarchical chat state .pt file. Pass no value for an auto path.")
    chat_group.add_argument("--resume-chat-from-state-file", type=str, default=None, help="Resume and autosave a model/tokenizer-bound hierarchical chat state .pt file.")
    chat_group.add_argument("--carry-chat-state", action="store_true", default=False, help="Carry recurrent/hierarchical state across user turns. Default OFF for Alpaca train/chat parity; use Previous Context text for history instead.")
    chat_group.add_argument("--chat-prefill-chunk-size", type=int, default=None, help="Chunk prompt prefill like TBPTT training. Default: checkpoint training_chunk_size; use 0 for one full prompt forward.")
    chat_group.add_argument("--chat-input-history-turns", type=int, default=0, help="For Alpaca chat, put this many previous turns in the ### Previous Context field. Default 0 matches independent SFT samples.")
    chat_group.add_argument("--chat-input-history-chars", type=int, default=3000, help="Character cap for Alpaca chat ### Previous Context text.")
    
    # --- Utility Arguments ---
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument("--ckpt-input", type=str, default=None, help="Input checkpoint path for ckpt-2-inf mode.")
    util_group.add_argument("--inf-output", type=str, default=None, help="Output inference model path for ckpt-2-inf mode.")
    util_group.add_argument("--ckpt-tok-path", type=str, default=None, help="HuggingFace tokenizer name/path to embed in the inference model (e.g., 'gpt2', 'openai-community/gpt2').")
    util_group.add_argument(
        "--overwrite-merge-output",
        action="store_true",
        help=(
            "[merge-lora] Atomically replace an existing output package after "
            "the merged replacement passes full validation."
        ),
    )
    
    normalized_argv = _normalize_optional_bool_flags(
        sys.argv[1:],
        (
            "--override-scheduling",
            "--override_scheduling",
            "--rebuild-lr-schedule",
            "--reset-optimizer-state",
        ),
    )
    args = parser.parse_args(normalized_argv)
    if args.mode == "merge-lora":
        if not args.model_path:
            parser.error("merge-lora requires --model-path for the full-precision base model.")
        if not args.lora_adapter_path:
            parser.error("merge-lora requires --lora-adapter-path.")
        if not args.out_dir:
            parser.error("merge-lora requires a non-empty --out-dir.")
    if args.mode == "quantize":
        print(
            "ERROR: Quantized export is intentionally disabled for the active "
            "RWKV matrix-state/coherent-v9 architecture."
        )
        print(
            "       The legacy scalar-RWKV exporter cannot preserve DeepEmbed, "
            "ROSA, ACT, Worker, or recurrent-state semantics. Use a full-precision "
            ".pt checkpoint until a v8/v9 parity-tested exporter is available."
        )
        sys.exit(2)
    if args.enable_quantized_learning or args.shadow_model_path:
        parser.error(
            "quantized online learning is unsupported by the active matrix-state "
            "architecture; --enable-quantized-learning and --shadow-model-path "
            "cannot be used"
        )
    explicit_dests = _explicit_cli_dests(parser, normalized_argv)
    args._explicit_cli_dests = sorted(explicit_dests)
    _hydrate_training_args_from_model_config(args, parser, explicit_dests)
    _apply_assistant_recovery_defaults(args, explicit_dests)
    _apply_benchmark_preset_defaults(args, explicit_dests)
    apply_architecture_revision_defaults(args)
    _validate_resume_epoch_target(args)
    if bool(getattr(args, "override_scheduling", False)):
        args.rebuild_lr_schedule = True
        args.reset_optimizer_state = True
        print(
            "WARNING: --override-scheduling is the legacy combined reset. "
            "It aliases --reset-optimizer-state plus --rebuild-lr-schedule."
        )
    if args.mode in ("train", "finetune"):
        set_global_training_seed(getattr(args, "seed", 1337))

    if args.mode == "benchmark" and args.list_benchmarks:
        if format_benchmark_catalog is None:
            print("ERROR: Benchmark registry is unavailable.")
            sys.exit(1)
        print(format_benchmark_catalog(include_external=True))
        return

    if args.mode == "benchmark" and not args.model_path:
        print("ERROR: benchmark mode requires --model-path.")
        print("       Use --list-benchmarks to inspect available benchmark suites without loading a model.")
        sys.exit(1)

    if args.mode in ("train", "finetune") and args.eval_tasks and resolve_task_names is not None:
        resolved_eval_tasks, skipped_eval_specs, _ = resolve_task_names(args.eval_tasks)
        args.eval_tasks = resolved_eval_tasks
        if skipped_eval_specs:
            skipped_names = ", ".join(spec.display_name for spec in skipped_eval_specs)
            print(f"INFO: Skipping non-lm-eval benchmark entries in --eval-tasks: {skipped_names}")
    if getattr(args, "best_checkpoint_metric", None):
        if args.mode != "train":
            parser.error(
                "--best-checkpoint-metric is currently supported by train mode"
            )
        if not args.eval_tasks:
            parser.error(
                "--best-checkpoint-metric requires at least one runnable --eval-tasks entry"
            )

    # Parity: hidden size auto-sync
    if args.mode == 'train' and not args.resume_from_ckpt:
        if args.h_hidden is None: args.h_hidden = args.context_dim
        if args.l_hidden is None: args.l_hidden = args.context_dim

    set_threads(args.threads)
    if args.compile or args.force_compile:
        setup_msvc_environment()
    pt_device = pick_device(args)
    args.num_workers = resolve_num_workers(
        args.num_workers,
        pt_device,
        args.batch_size,
        token_cache_only=bool(getattr(args, "token_cache_only", False)),
    )
    random_access_token_cache = (
        bool(args.hf_dataset) and bool(getattr(args, "hf_token_cache", False))
    ) or (
        bool(getattr(args, "train", None))
        and _is_jsonl_source(args.train)
        and bool(getattr(args, "local_token_cache", False))
    )
    args.length_bucket_size, bucket_message = resolve_length_bucket_size(
        args.length_bucket_size,
        pt_device,
        args.batch_size,
        hf_token_cache=random_access_token_cache and _length_bucketing_enabled(args),
        auto_tune=getattr(args, "auto_length_bucket_size", True),
    )
    if bucket_message:
        print(bucket_message)
    print(f"INFO: DataLoader workers set to {args.num_workers} for device={pt_device}.")
    
    # Tokenizer Loading
    tokenizer = None
    tokenizer_source = resolve_tokenizer_source(
        args.tokenizer_path,
        args.model_path or args.resume_from_ckpt,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        configure_tokenizer_length(tokenizer, args.max_length)
        if args.mode in ("train", "finetune"):
            _ensure_tokenizer_identity(args, tokenizer)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}"); sys.exit(1)

    # --- Auto Max Length Scan ---
    if args.auto_max_length and args.mode in ['train', 'finetune']:
        max_found = 0
        if args.hf_dataset:
            print(f"Scanning HF dataset: {args.hf_dataset}...")
            resolve_hf_dataset_revision(args, announce=True)
            temp_ds = load_hf_dataset(
                args.hf_dataset,
                args.hf_dataset_config,
                split=args.hf_dataset_split,
                revision=_hf_revision_for_load(args),
            )
            for sample in tqdm(temp_ds, desc="Scanning HF"):
                processed = process_text_sample(
                    tokenizer, sample, 9999, args.kayla,
                    args.text_column, args.prompt_column, args.completion_column,
                    args.alpaca, getattr(args, "train_prompt_tokens", True),
                    **_loss_weight_kwargs(args), **_response_quality_kwargs(args),
                )
                if processed: max_found = max(max_found, len(processed['input_ids']))
        elif args.train and isinstance(args.train, str):
            jsonl_files = _jsonl_source_files(args.train)
            if jsonl_files:
                print(f"Scanning local JSONL source: {args.train}...")
                for jsonl_path in jsonl_files:
                    with open(jsonl_path, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"Scanning {os.path.basename(jsonl_path)}"):
                            try:
                                processed = process_text_sample(
                                    tokenizer, json.loads(line), 9999, args.kayla,
                                    args.text_column, args.prompt_column, args.completion_column,
                                    args.alpaca, getattr(args, "train_prompt_tokens", True),
                                    **_loss_weight_kwargs(args), **_response_quality_kwargs(args),
                                )
                                if processed: max_found = max(max_found, len(processed['input_ids']))
                            except: continue
            else:
                print(f"Scanning local file: {args.train}...")
                with open(args.train, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list): data = [data]
                        for obj in tqdm(data, desc="Scanning JSON"):
                            processed = process_text_sample(
                                tokenizer, obj, 9999, args.kayla,
                                args.text_column, args.prompt_column, args.completion_column,
                                args.alpaca, getattr(args, "train_prompt_tokens", True),
                                **_loss_weight_kwargs(args), **_response_quality_kwargs(args),
                            )
                            if processed: max_found = max(max_found, len(processed['input_ids']))
                    except:
                        f.seek(0)
                        for line in tqdm(f, desc="Scanning JSONL"):
                            try:
                                processed = process_text_sample(
                                    tokenizer, json.loads(line), 9999, args.kayla,
                                    args.text_column, args.prompt_column, args.completion_column,
                                    args.alpaca, getattr(args, "train_prompt_tokens", True),
                                    **_loss_weight_kwargs(args), **_response_quality_kwargs(args),
                                )
                                if processed: max_found = max(max_found, len(processed['input_ids']))
                            except: continue
        if max_found > 0:
            args.max_length = (max_found + 16 + 7) & -8 # Align to 8
            print(f"Auto-scan found max length {max_found}. Setting max_length={args.max_length}")

    if getattr(args, "token_cache_only", False):
        if args.mode not in ("train", "finetune"):
            print("ERROR: --token-cache-only requires train or finetune mode.")
            sys.exit(1)
        if args.hf_dataset:
            cache_dir = materialize_hf_token_cache(args, tokenizer)
        elif args.train and isinstance(args.train, str) and _is_jsonl_source(args.train):
            cache_dir = materialize_local_token_cache(args, tokenizer)
        else:
            print("ERROR: --token-cache-only requires --hf-dataset or a JSONL --train source.")
            sys.exit(1)
        print(f"INFO: Token-cache-only preparation complete: {cache_dir}")
        return

    # Execution
    if args.mode == "train":
        dataloader = None
        if args.hf_dataset:
            dataloader = create_hf_training_dataloader(args, tokenizer, pt_device)
        elif args.pre_chunked_dataset:
            dataloader = create_dataloader_for_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor)
        elif args.pre_pt_dataset:
            dataloader = create_dataloader_pt_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor, cache_size=args.pt_cache_size)
        elif args.train and isinstance(args.train, str):
            dataloader = create_local_training_dataloader(args, tokenizer, pt_device)
        
        if dataloader is None:
            print("ERROR: No dataset provided for training. Use --train or --hf_dataset."); sys.exit(1)
        if bool(getattr(args, "persist_state", False)) and not bool(
            getattr(getattr(dataloader, "dataset", None), "guarantees_contiguous_state", False)
        ):
            raise ValueError(
                "--persist-state is unsafe for ordinary shuffled/length-bucketed "
                "independent samples. The selected dataset does not declare contiguous "
                "same-sequence lane ordering; use --no-persist-state."
            )

        if args.dataset_size:
            dataloader_len = _steps_from_samples(args.dataset_size, args.batch_size)
            print(f"INFO: Manual session override: {args.dataset_size} samples -> {dataloader_len} steps per epoch.")
        else:
            try: 
                dataloader_len = len(dataloader)
            except: 
                # Heuristic for IterableDataset (JSONL) or HF Metadata
                if args.hf_dataset:
                    print(f"INFO: Estimating size for HF dataset: {args.hf_dataset}...")
                    try:
                        count = estimate_hf_dataset_size(
                            args.hf_dataset,
                            args.hf_dataset_config,
                            args.hf_dataset_split,
                            revision=_hf_revision_for_load(args),
                        )
                        if count is not None:
                            dataloader_len = _steps_from_samples(count, args.batch_size)
                            print(f"INFO: Automatic HF session detection found {count} samples -> {dataloader_len} steps per epoch.")
                        else:
                            print(f"ERROR: Could not find split '{args.hf_dataset_split}' in HF dataset info. Please use --dataset-size."); sys.exit(1)
                    except Exception as e:
                        print(f"ERROR: Failed to query HF metadata: {e}. Please specify --dataset-size manually."); sys.exit(1)
                elif hasattr(args, 'train') and isinstance(args.train, str) and _is_jsonl_source(args.train):
                    print(f"INFO: Estimating dataset size from JSONL source: {args.train}...")
                    try:
                        count = count_jsonl_source_rows(args.train)
                        dataloader_len = _steps_from_samples(count, args.batch_size)
                        print(f"INFO: Automatic session detection found {count} samples -> {dataloader_len} steps per epoch.")
                    except Exception as e:
                        print(f"ERROR: Could not count JSONL rows in {args.train}: {e}. Please specify --dataset-size manually."); sys.exit(1)
                else:
                    print("ERROR: Dataset size could not be determined automatically. Please use --dataset-size to specify the number of steps per epoch."); sys.exit(1)

        train(args, pt_device, tokenizer, dataloader, dataloader_len)
    elif args.mode == "chat":
        chat(args, pt_device, tokenizer)
    elif args.mode == "benchmark":
        if run_post_training_benchmarks is None or write_benchmark_artifacts is None:
            print("ERROR: Post-training benchmark support is unavailable.")
            sys.exit(1)

        print(f"Loading model for post-training benchmarks: {args.model_path}")
        try:
            model, benchmark_config = load_full_model_with_config(args.model_path, pt_device)
            validate_tokenizer_vocab(tokenizer, benchmark_config, args.model_path)
            if validate_inference_tokenizer_identity(
                tokenizer,
                getattr(model, "_hierarchos_checkpoint_metadata", {}),
            ):
                print("INFO: Benchmark tokenizer content fingerprint verified.")
            else:
                print(
                    "WARNING: This legacy benchmark checkpoint has no strong "
                    "tokenizer fingerprint; only vocabulary size was checked."
                )
            clear_ltm_working_memory(model)
            model.suppress_hebbian = True
        except Exception as e:
            print(f"ERROR: Failed to load model for benchmark mode: {e}")
            sys.exit(1)

        try:
            benchmark_suites = list(args.benchmark_suite or [])
            if args.benchmark_all:
                benchmark_suites.insert(0, "all-common")
            raw_benchmark_tasks = (
                args.eval_tasks
                if args.eval_tasks and not args.benchmark_all and not benchmark_suites and not args.benchmark
                else None
            )
            results, manifest, skipped = run_post_training_benchmarks(
                model=model,
                tokenizer=tokenizer,
                device=pt_device,
                benchmark_names=args.benchmark,
                suite_names=benchmark_suites,
                raw_tasks=raw_benchmark_tasks,
                batch_size=args.eval_batch_size,
                limit=args.eval_limit,
                verbosity="WARNING",
                strict=args.strict_benchmarks,
                arc_agi_path=args.arc_agi_path,
                arc_agi_max_tasks=args.arc_agi_max_tasks,
                arc_agi_max_test_items=args.arc_agi_max_test_items,
                arc_agi_keep_samples=args.arc_agi_keep_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                sequential=args.benchmark_all or args.benchmark_sequential,
            )
        except Exception as e:
            print(f"ERROR: Benchmark run failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        print("\n" + "="*60)
        print("Combined Benchmark Results")
        print("="*60)
        print(format_results(results))
        if skipped:
            print("\nSkipped external benchmarks:")
            for spec in skipped:
                detail = spec.notes or spec.description
                print(f"  - {spec.display_name} ({spec.key}): {detail}")

        run_dir = write_benchmark_artifacts(
            output_dir=args.benchmark_out_dir,
            run_name=args.benchmark_run_name,
            results=results,
            manifest=manifest,
            skipped=skipped,
        )
        print(f"\nBenchmark artifacts saved to: {run_dir}")
    elif args.mode == "finetune":
        # Build dataloader for finetune
        dataloader = None
        if args.hf_dataset:
            dataloader = create_hf_training_dataloader(args, tokenizer, pt_device)
        elif args.pre_chunked_dataset:
            dataloader = create_dataloader_for_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor)
        elif args.pre_pt_dataset:
            dataloader = create_dataloader_pt_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor, cache_size=args.pt_cache_size)
        elif args.train and isinstance(args.train, str):
            dataloader = create_local_training_dataloader(args, tokenizer, pt_device)
        
        if dataloader is None:
            print("ERROR: No dataset provided for finetuning. Use --train or --hf_dataset."); sys.exit(1)

        if args.dataset_size:
            dataloader_len = _steps_from_samples(args.dataset_size, args.batch_size)
            print(f"INFO: Manual session override: {args.dataset_size} samples -> {dataloader_len} steps per epoch.")
        else:
            try: 
                dataloader_len = len(dataloader)
            except: 
                # Heuristic for IterableDataset (JSONL) or HF Metadata
                if args.hf_dataset:
                    print(f"INFO: Estimating size for HF dataset: {args.hf_dataset}...")
                    try:
                        count = estimate_hf_dataset_size(
                            args.hf_dataset,
                            args.hf_dataset_config,
                            args.hf_dataset_split,
                            revision=_hf_revision_for_load(args),
                        )
                        if count is not None:
                            dataloader_len = _steps_from_samples(count, args.batch_size)
                            print(f"INFO: Automatic HF session detection found {count} samples -> {dataloader_len} steps per epoch.")
                        else:
                            print(f"ERROR: Could not find split '{args.hf_dataset_split}' in HF dataset info. Please use --dataset-size."); sys.exit(1)
                    except Exception as e:
                        print(f"ERROR: Failed to query HF metadata: {e}. Please specify --dataset-size manually."); sys.exit(1)
                elif hasattr(args, 'train') and isinstance(args.train, str) and _is_jsonl_source(args.train):
                    print(f"INFO: Estimating dataset size from JSONL source: {args.train}...")
                    try:
                        count = count_jsonl_source_rows(args.train)
                        dataloader_len = _steps_from_samples(count, args.batch_size)
                        print(f"INFO: Automatic session detection found {count} samples -> {dataloader_len} steps per epoch.")
                    except Exception as e:
                        print(f"ERROR: Could not count JSONL rows in {args.train}: {e}. Please specify --dataset-size manually."); sys.exit(1)
                else:
                    print("ERROR: Dataset size could not be determined automatically. Please use --dataset-size to specify the number of steps per epoch."); sys.exit(1)

        finetune(args, pt_device, tokenizer, dataloader, dataloader_len)
    elif args.mode == "merge-lora":
        try:
            merged_output = merge_lora_adapter(
                base_model_path=args.model_path,
                adapter_path=args.lora_adapter_path,
                output_dir=args.out_dir,
                tokenizer=tokenizer,
                overwrite=bool(args.overwrite_merge_output),
            )
        except Exception as exc:
            print(f"ERROR: LoRA merge failed: {exc}")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("LORA MERGE COMPLETE")
        print("=" * 60)
        print(f"Standalone model: {merged_output}")
        print("Adapter wrappers: removed")
        print("Transient LTM state: reset")
        print("Validation: architecture, tensor geometry, finiteness, tied weights, tokenizer")
        print("\nTo use the merged model:")
        print(f'  python hierarchos_cli.py chat --model-path "{merged_output}"')
        print("=" * 60)
    elif args.mode == "ckpt-2-inf":
        # Convert checkpoint to inference model (HuggingFace-style directory)
        ckpt_path = args.ckpt_input or args.resume_from_ckpt or args.model_path
        if not ckpt_path:
            print("ERROR: No checkpoint specified. Use --ckpt-input, --resume-from-ckpt, or --model-path.")
            sys.exit(1)
        
        # Determine output directory (strip .pt extension if provided)
        if args.inf_output:
            output_dir = args.inf_output.replace('.pt', '')
        else:
            base_dir = os.path.dirname(ckpt_path)
            output_dir = os.path.join(base_dir, "hierarchos_final")
        
        print(f"Converting checkpoint to inference model...")
        print(f"  Input:  {ckpt_path}")
        print(f"  Output: {output_dir}/")
        
        try:
            checkpoint = load_checkpoint_payload_compatible(
                ckpt_path,
                map_location='cpu',
            )
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # Strictly validate every learned tensor before producing an inference
        # export. This also applies deterministic legacy qproj adaptation.
        try:
            verified_model, verified_config = load_full_model_with_config(ckpt_path, torch.device("cpu"))
        except Exception as e:
            print(f"ERROR: Checkpoint failed coherent-load validation: {e}")
            sys.exit(1)
        clean_state_dict = sanitize_model_state_dict(verified_model, reset_transient_ltm=True)
        config = dict(verified_config)
        completed_epoch = checkpoint.get('completed_epoch', checkpoint.get('epoch', 'unknown'))
        
        # Handle tokenizer
        tokenizer_name = resolve_export_tokenizer_source(args.ckpt_tok_path, config)
        config['tokenizer_name'] = tokenizer_name
        print(f"  Tokenizer: {tokenizer_name}")
        
        # Load and verify tokenizer
        try:
            inf_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=bool(args.trust_remote_code),
            )
            if inf_tokenizer.pad_token is None:
                if inf_tokenizer.eos_token:
                    inf_tokenizer.pad_token = inf_tokenizer.eos_token
                else:
                    inf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            vocab_size = len(inf_tokenizer)
            print(f"  Vocab size: {vocab_size}")
            
            # Verify vocab size matches model
            model_vocab = config.get('vocab_size')
            if model_vocab and model_vocab != vocab_size:
                print(f"ERROR: Model vocab_size ({model_vocab}) != tokenizer vocab_size ({vocab_size})")
                print("       Refusing to export an inference package with an incompatible tokenizer.")
                sys.exit(1)
            if validate_inference_tokenizer_identity(inf_tokenizer, checkpoint):
                print("  Tokenizer fingerprint: verified against training")
            else:
                print(
                    "WARNING: Source checkpoint has no strong tokenizer fingerprint; "
                    "the export can verify vocabulary size only."
                )
        except Exception as e:
            print(f"ERROR: Failed to load tokenizer '{tokenizer_name}': {e}")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer files
        print(f"  Saving tokenizer files...")
        inf_tokenizer.save_pretrained(output_dir)
        
        # Create inference-ready checkpoint. Use the same canonical filename as
        # training export so chat cannot accidentally load an older sibling.
        model_path = os.path.join(output_dir, "hierarchos.pt")
        exported_architecture_contract = architecture_contract(config)
        exported_architecture_hash = architecture_contract_hash(config)
        config["architecture_contract_sha256"] = exported_architecture_hash
        carries_persistent_ltm = (
            checkpoint.get("checkpoint_kind") == "inference-ltm-consolidated"
            and "ltm_persistent_metadata" in checkpoint
        )
        inference_checkpoint = {
            'checkpoint_version': max(
                4,
                int(checkpoint.get("checkpoint_version", 0) or 0),
            ),
            'checkpoint_kind': (
                "inference-ltm-consolidated"
                if carries_persistent_ltm
                else "inference"
            ),
            'derived_from_checkpoint_kind': checkpoint.get("checkpoint_kind"),
            'derived_from_checkpoint_version': checkpoint.get("checkpoint_version"),
            'model_state_dict': clean_state_dict,
            'config': config,
            'architecture_contract': exported_architecture_contract,
            'architecture_contract_sha256': exported_architecture_hash,
            'completed_epoch': completed_epoch,
            'training_complete': True,
            'converted_from': os.path.basename(ckpt_path),
            'tokenizer_name': tokenizer_name,
        }
        # Keep the exact tokenizer/run proof and any explicitly consolidated
        # user-memory metadata. Dropping either would turn a coherent source
        # checkpoint into an inference package whose parity or retrieval
        # filters could no longer be verified.
        for metadata_key in (
            "tokenizer_identity",
            "expansion_provenance",
            "run_identity",
            "best_metric_state",
            "selection_metric",
            "effective_training_config",
            "optimizer_grouping_version",
            "ltm_persistent_metadata",
        ):
            if metadata_key in checkpoint:
                inference_checkpoint[metadata_key] = checkpoint[metadata_key]
        save_checkpoint_safely(inference_checkpoint, model_path)
        
        # Save config as JSON for easy inspection
        config_path = os.path.join(output_dir, "hierarchos_config.json")
        import json as json_module
        config_to_save = dict(config)
        config_to_save['completed_epoch'] = completed_epoch
        config_to_save['tokenizer_name'] = tokenizer_name
        config_to_save['converted_from'] = os.path.basename(ckpt_path)
        with open(config_path, 'w') as f:
            json_module.dump(config_to_save, f, indent=2, default=str)
        
        # Report
        input_size = os.path.getsize(ckpt_path)
        output_size = os.path.getsize(model_path)
        reduction = (1 - output_size / input_size) * 100
        
        # Count all files in output dir
        total_output_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir))
        
        print(f"\n" + "="*60)
        print(f"CONVERSION COMPLETE!")
        print(f"="*60)
        print(f"Input checkpoint: {input_size / 1e6:.2f} MB")
        print(f"Output directory: {output_dir}/")
        print(f"  - hierarchos.pt: {output_size / 1e6:.2f} MB  ({reduction:.1f}% reduction)")
        print(f"  - Total size:   {total_output_size / 1e6:.2f} MB")
        print(f"  - Epoch:        {completed_epoch}")
        print(f"  - Tokenizer:    {tokenizer_name}")
        print(f"\nTo use the model for inference:")
        print(f"  python hierarchos_cli.py chat --model-path \"{output_dir}\"")
        print(f"="*60)
    else:
        print(f"INFO: Mode '{args.mode}' is not yet fully integrated in the CLI.")

if __name__ == "__main__":
    main()
