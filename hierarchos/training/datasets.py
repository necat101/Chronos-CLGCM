import os
import json
import mmap
import sys
import torch
import traceback
import functools
import itertools
import random
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from typing import Optional, List, Dict, Any
from hierarchos.utils.rosa import (
    ROSA_BOUNDED_CONTEXT_MODE,
    ROSA_UNBOUNDED_CONTEXT_MODE,
)
from hierarchos.utils.safe_loading import load_tensor_payload_safely


PORTABLE_SAMPLER_RNG_ALGORITHM = "splitmix64-fisher-yates-v1"
LEGACY_TORCH_SAMPLER_RNG_ALGORITHM = "torch-generator-randperm-v1"
_U64_MASK = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


class _PortableSplitMix64:
    """Tiny backend-neutral RNG used only for deterministic data ordering.

    SplitMix64 is intentionally used as a cursorable integer generator rather
    than as a numerical-training RNG. Its complete state is one u64 counter,
    which is trivial to reproduce in Rust/Vulkan hosts without importing a
    Python, NumPy, or PyTorch RNG representation.
    """

    def __init__(self, state: int):
        self.state = int(state) & _U64_MASK

    def next_u64(self) -> int:
        self.state = (self.state + _SPLITMIX_GAMMA) & _U64_MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
        return (value ^ (value >> 31)) & _U64_MASK


def _portable_sampler_rng(seed: int, epoch: int) -> _PortableSplitMix64:
    # Give each epoch a disjoint deterministic starting state while keeping the
    # recipe representable as two ordinary unsigned integers in checkpoints.
    state = (int(seed) + (int(epoch) * _SPLITMIX_GAMMA)) & _U64_MASK
    return _PortableSplitMix64(state)


def portable_fisher_yates_permutation(length: int, seed: int, epoch: int = 0):
    """Return the canonical Hierarchos portable permutation for one epoch."""

    length = max(0, int(length))
    values = list(range(length))
    rng = _portable_sampler_rng(seed, epoch)
    for upper in range(length - 1, 0, -1):
        swap_index = rng.next_u64() % (upper + 1)
        values[upper], values[swap_index] = values[swap_index], values[upper]
    return values


def _portable_randperm_tensor(length: int, rng: _PortableSplitMix64):
    values = list(range(max(0, int(length))))
    for upper in range(len(values) - 1, 0, -1):
        swap_index = rng.next_u64() % (upper + 1)
        values[upper], values[swap_index] = values[swap_index], values[upper]
    return torch.tensor(values, dtype=torch.long)


def _shared_epoch_counter():
    """Create an epoch value that persistent DataLoader workers can observe."""
    epoch = torch.zeros((), dtype=torch.int64)
    try:
        epoch.share_memory_()
    except RuntimeError:
        # Single-process and restricted shared-memory environments still work;
        # only cross-process persistent-worker updates lose this fast path.
        pass
    return epoch


def _set_iterable_epoch(dataset, epoch: int):
    epoch = int(epoch)
    dataset.epoch = epoch
    dataset._shared_epoch.fill_(epoch)


def _get_iterable_epoch(dataset):
    return int(dataset._shared_epoch.item())

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class LengthGroupedBatchSampler(Sampler):
    """
    Shuffled batch sampler that keeps similarly sized samples together.

    This preserves epoch-level randomness while reducing the amount of padding
    introduced by the dynamic collator.
    """
    def __init__(self, lengths, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, bucket_size: Optional[int] = None,
                 seed: Optional[int] = None, preserve_order: bool = False,
                 rng_algorithm: str = PORTABLE_SAMPLER_RNG_ALGORITHM):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.lengths = _lengths_to_cpu_tensor(lengths)
        if self.lengths is None:
            raise ValueError("lengths must be a one-dimensional sequence of integers")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.preserve_order = bool(preserve_order)
        target_bucket_size = int(bucket_size or (self.batch_size * 50))
        self.bucket_size = max(self.batch_size, target_bucket_size)
        self.bucket_size = max(
            self.batch_size,
            (self.bucket_size // self.batch_size) * self.batch_size,
        )
        self.seed = int(seed if seed is not None else torch.initial_seed()) % (2**63 - 1)
        self.rng_algorithm = str(rng_algorithm)
        if self.rng_algorithm not in {
            PORTABLE_SAMPLER_RNG_ALGORITHM,
            LEGACY_TORCH_SAMPLER_RNG_ALGORITHM,
        }:
            raise ValueError(f"unsupported sampler RNG algorithm {self.rng_algorithm!r}")
        self.epoch = 0
        # A resume cursor is deliberately separate from epoch/seed state. It
        # changes only where iteration begins, never the deterministic epoch
        # permutation itself. Keeping __len__ at the full-epoch value also
        # preserves scheduler and accumulation geometry across a mid-epoch
        # resume.
        self.start_batch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int):
        self.start_batch = max(0, min(int(start_batch), len(self)))

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = None
        portable_rng = None
        if self.rng_algorithm == PORTABLE_SAMPLER_RNG_ALGORITHM:
            portable_rng = _portable_sampler_rng(self.seed, self.epoch)
        else:
            generator = torch.Generator()
            generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))
        sample_count = len(self.lengths)

        if self.shuffle and not self.preserve_order and sample_count > 1:
            indices = (
                _portable_randperm_tensor(sample_count, portable_rng)
                if portable_rng is not None
                else torch.randperm(sample_count, generator=generator)
            )
        else:
            indices = torch.arange(sample_count, dtype=torch.long)

        # Keep the epoch permutation and sorted windows as compact tensors. The
        # previous implementation expanded every index into Python integers and
        # retained a Python list for every batch before it could yield the first
        # one. At multi-million-sample scale that consumed hundreds of MiB and
        # made epoch startup Python-sort bound.
        for bucket_start in range(0, sample_count, self.bucket_size):
            bucket_end = min(sample_count, bucket_start + self.bucket_size)
            if self.shuffle:
                bucket = indices[bucket_start:bucket_end]
                bucket_lengths = self.lengths.index_select(0, bucket)
                length_order = torch.argsort(
                    bucket_lengths,
                    descending=True,
                    stable=True,
                )
                indices[bucket_start:bucket_end] = bucket.index_select(0, length_order)

        if self.shuffle and self.preserve_order:
            # PT caches use this mode to keep each length window local on disk
            # while randomizing the batches inside that window.
            for bucket_start in range(0, sample_count, self.bucket_size):
                bucket_end = min(sample_count, bucket_start + self.bucket_size)
                bucket_count = bucket_end - bucket_start
                if self.drop_last:
                    batch_count = bucket_count // self.batch_size
                else:
                    batch_count = (bucket_count + self.batch_size - 1) // self.batch_size
                if batch_count > 1:
                    batch_order = (
                        _portable_randperm_tensor(batch_count, portable_rng)
                        if portable_rng is not None
                        else torch.randperm(batch_count, generator=generator)
                    )
                else:
                    batch_order = range(batch_count)
                for local_output_idx, local_batch_idx in enumerate(batch_order):
                    global_batch_idx = (
                        (bucket_start // self.batch_size) + local_output_idx
                    )
                    if global_batch_idx < self.start_batch:
                        continue
                    batch_start = bucket_start + int(local_batch_idx) * self.batch_size
                    batch_end = min(bucket_end, batch_start + self.batch_size)
                    if batch_end - batch_start == self.batch_size or not self.drop_last:
                        yield indices[batch_start:batch_end].tolist()
            return

        batch_count = len(self)
        if self.shuffle and not self.preserve_order and batch_count > 1:
            batch_order = (
                _portable_randperm_tensor(batch_count, portable_rng)
                if portable_rng is not None
                else torch.randperm(batch_count, generator=generator)
            )
        else:
            batch_order = range(batch_count)
        for output_batch_idx, batch_idx in enumerate(batch_order):
            if output_batch_idx < self.start_batch:
                continue
            batch_start = int(batch_idx) * self.batch_size
            batch_end = min(sample_count, batch_start + self.batch_size)
            if batch_end - batch_start == self.batch_size or not self.drop_last:
                yield indices[batch_start:batch_end].tolist()

class EpochShuffleSampler(Sampler):
    """Deterministic map-style sampler with an epoch hook for resume parity."""
    def __init__(self, data_source, shuffle: bool = True, seed: Optional[int] = None,
                 rng_algorithm: str = PORTABLE_SAMPLER_RNG_ALGORITHM):
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed if seed is not None else torch.initial_seed()) % (2**63 - 1)
        self.rng_algorithm = str(rng_algorithm)
        if self.rng_algorithm not in {
            PORTABLE_SAMPLER_RNG_ALGORITHM,
            LEGACY_TORCH_SAMPLER_RNG_ALGORITHM,
        }:
            raise ValueError(f"unsupported sampler RNG algorithm {self.rng_algorithm!r}")
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int):
        self.start_index = max(0, min(int(start_index), len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        length = len(self.data_source)
        if self.shuffle and length > 1:
            if self.rng_algorithm == PORTABLE_SAMPLER_RNG_ALGORITHM:
                indices = portable_fisher_yates_permutation(length, self.seed, self.epoch)
                yield from indices[self.start_index:]
            else:
                generator = torch.Generator()
                generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))
                indices = torch.randperm(length, generator=generator)
                yield from indices[self.start_index:].tolist()
        else:
            yield from range(self.start_index, length)

_INTEGER_LENGTH_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _lengths_to_cpu_tensor(lengths):
    if lengths is None:
        return None
    try:
        if isinstance(lengths, torch.Tensor):
            length_tensor = lengths.detach()
        else:
            try:
                length_tensor = torch.as_tensor(lengths)
            except (TypeError, ValueError, RuntimeError):
                length_tensor = torch.tensor(
                    [int(length) for length in lengths],
                    dtype=torch.long,
                )
        if length_tensor.is_complex():
            return None
        if length_tensor.is_floating_point():
            if not bool(torch.isfinite(length_tensor).all().item()):
                return None
            length_tensor = length_tensor.to(dtype=torch.long)
        elif length_tensor.dtype not in _INTEGER_LENGTH_DTYPES:
            length_tensor = length_tensor.to(dtype=torch.long)
        if length_tensor.device.type != "cpu":
            length_tensor = length_tensor.to(device="cpu")
        length_tensor = length_tensor.reshape(-1)
        if length_tensor.numel() > 0 and bool((length_tensor < 1).any().item()):
            length_tensor = length_tensor.clamp_min(1)
        return length_tensor.contiguous()
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None


def _normalize_sample_lengths(dataset, lengths):
    lengths = _lengths_to_cpu_tensor(lengths)
    if lengths is None or len(lengths) != len(dataset):
        return None
    return lengths

def _get_dataset_sample_lengths(dataset):
    lengths = _normalize_sample_lengths(dataset, getattr(dataset, "sample_lengths", None))
    if lengths is not None:
        return lengths

    getter = getattr(dataset, "get_sample_lengths", None)
    if callable(getter):
        return _normalize_sample_lengths(dataset, getter())
    return None

def _device_type(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return device.type
    device_str = str(device).lower()
    if device_str.startswith("cuda"):
        return "cuda"
    if device_str.startswith("dml") or device_str.startswith("privateuseone"):
        return "dml"
    return "cpu"

def _pin_memory_for_device(device=None):
    return _device_type(device) == "cuda" and torch.cuda.is_available()

def _worker_init_fn(_worker_id):
    # Keep loader workers from each spinning up a full BLAS/OpenMP thread pool.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    # PyTorch gives each worker a deterministic seed derived from the loader's
    # generator. Mirror it into libraries used by dataset/formatter extensions
    # so a global run seed controls every preprocessing path.
    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass

def _resolve_prefetch_factor(num_workers: int, prefetch_factor=None, pin_memory: bool = False):
    if num_workers <= 0:
        return None
    if prefetch_factor is not None:
        return max(1, int(prefetch_factor))

    target_prefetched_batches = 16 if pin_memory else 4
    return max(1, (target_prefetched_batches + num_workers - 1) // num_workers)

def _create_dataloader(dataset, *, batch_size=None, collate_fn=None, num_workers=0,
                       pin_memory=False, shuffle=None, drop_last=False,
                       batch_sampler=None, sampler=None, prefetch_factor=None,
                       in_order=True):
    resolved_num_workers = int(num_workers or 0)
    if resolved_num_workers < 0:
        resolved_num_workers = 0

    kwargs = {
        "dataset": dataset,
        "collate_fn": collate_fn,
        "num_workers": resolved_num_workers,
        "pin_memory": bool(pin_memory),
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = batch_size
        if sampler is not None:
            kwargs["sampler"] = sampler
        elif shuffle is not None:
            kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = drop_last

    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = _worker_init_fn
        kwargs["prefetch_factor"] = _resolve_prefetch_factor(
            kwargs["num_workers"],
            prefetch_factor=prefetch_factor,
            pin_memory=kwargs["pin_memory"],
        )
    else:
        kwargs["persistent_workers"] = False
    if not bool(in_order) and kwargs["num_workers"] > 0:
        # Cache construction consumes every row before training and later uses
        # an explicit shuffling sampler, so FIFO worker delivery only creates
        # head-of-line blocking without adding a semantic guarantee.
        kwargs["in_order"] = False

    try:
        return DataLoader(**kwargs)
    except TypeError as exc:
        # Keep older supported PyTorch releases functional; in_order was added
        # after the original training code and is only a cache-build speed hint.
        if "in_order" not in kwargs:
            raise
        kwargs.pop("in_order", None)
        return DataLoader(**kwargs)


def _load_tensor_artifact_weights_only(path):
    """Load project tensor artifacts without permitting arbitrary pickle code."""
    return load_tensor_payload_safely(path, map_location="cpu")


def _validate_pt_chunk_container(data, path):
    if not isinstance(data, (list, tuple)):
        raise ValueError(
            f"PT chunk {path} must contain a list/tuple of tensor sample mappings"
        )
    return data


def _validate_pt_chunk_item(item, path, index):
    if not isinstance(item, dict):
        raise ValueError(
            f"PT chunk {path} sample {index} must be a mapping"
        )
    input_ids = item.get("input_ids")
    labels = item.get("labels")
    if not torch.is_tensor(input_ids) or not torch.is_tensor(labels):
        raise ValueError(
            f"PT chunk {path} sample {index} is missing tensor input_ids/labels"
        )
    if input_ids.ndim != 1 or labels.ndim != 1 or input_ids.shape != labels.shape:
        raise ValueError(
            f"PT chunk {path} sample {index} has incompatible input/label geometry"
        )
    if (
        input_ids.dtype not in _INTEGER_LENGTH_DTYPES
        or labels.dtype not in _INTEGER_LENGTH_DTYPES
    ):
        raise ValueError(
            f"PT chunk {path} sample {index} input_ids/labels must use integer dtypes"
        )
    attention_mask = item.get("attention_mask")
    if attention_mask is not None and (
        not torch.is_tensor(attention_mask)
        or attention_mask.ndim != 1
        or attention_mask.shape != input_ids.shape
        or (
            attention_mask.dtype != torch.bool
            and attention_mask.dtype not in _INTEGER_LENGTH_DTYPES
        )
    ):
        raise ValueError(
            f"PT chunk {path} sample {index} has an incompatible attention mask"
        )
    for field_name in ("rosa_ids", "loss_weights"):
        value = item.get(field_name)
        if value is not None and (
            not torch.is_tensor(value)
            or value.ndim != 1
            or value.shape != input_ids.shape
        ):
            raise ValueError(
                f"PT chunk {path} sample {index} has incompatible {field_name}"
            )
    rosa_ids = item.get("rosa_ids")
    if rosa_ids is not None and rosa_ids.dtype not in _INTEGER_LENGTH_DTYPES:
        raise ValueError(
            f"PT chunk {path} sample {index} rosa_ids must use an integer dtype"
        )
    loss_weights = item.get("loss_weights")
    if loss_weights is not None and not (
        loss_weights.is_floating_point()
        or loss_weights.dtype in _INTEGER_LENGTH_DTYPES
    ):
        raise ValueError(
            f"PT chunk {path} sample {index} loss_weights must be numeric"
        )
    if loss_weights is not None and (
        not bool(torch.isfinite(loss_weights).all().item())
        or bool((loss_weights < 0).any().item())
    ):
        raise ValueError(
            f"PT chunk {path} sample {index} loss_weights must be finite and nonnegative"
        )
    return item


def _as_long_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)

def _as_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)

def _normalize_loss_weight_args(prompt_loss_weight: float = 1.0,
                                response_loss_weight: float = 1.0,
                                response_boundary_loss_weight: float = 1.0,
                                response_boundary_tokens: int = 0):
    try:
        prompt_loss_weight = max(0.0, float(prompt_loss_weight))
    except (TypeError, ValueError):
        prompt_loss_weight = 1.0
    try:
        response_loss_weight = max(0.0, float(response_loss_weight))
    except (TypeError, ValueError):
        response_loss_weight = 1.0
    try:
        response_boundary_loss_weight = max(0.0, float(response_boundary_loss_weight))
    except (TypeError, ValueError):
        response_boundary_loss_weight = 1.0
    try:
        response_boundary_tokens = max(0, int(response_boundary_tokens))
    except (TypeError, ValueError):
        response_boundary_tokens = 0
    return (
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
    )

def _uses_custom_loss_weights(prompt_loss_weight: float = 1.0,
                              response_loss_weight: float = 1.0,
                              response_boundary_loss_weight: float = 1.0,
                              response_boundary_tokens: int = 0):
    prompt_loss_weight, response_loss_weight, response_boundary_loss_weight, response_boundary_tokens = (
        _normalize_loss_weight_args(
            prompt_loss_weight,
            response_loss_weight,
            response_boundary_loss_weight,
            response_boundary_tokens,
        )
    )
    return (
        abs(prompt_loss_weight - 1.0) > 1e-8
        or abs(response_loss_weight - 1.0) > 1e-8
        or (response_boundary_tokens > 0 and abs(response_boundary_loss_weight - 1.0) > 1e-8)
    )

def _prompt_response_loss_weights(prompt_len: int, response_len: int,
                                  train_prompt_tokens: bool,
                                  prompt_loss_weight: float = 1.0,
                                  response_loss_weight: float = 1.0,
                                  response_boundary_loss_weight: float = 1.0,
                                  response_boundary_tokens: int = 0):
    prompt_loss_weight, response_loss_weight, response_boundary_loss_weight, response_boundary_tokens = (
        _normalize_loss_weight_args(
            prompt_loss_weight,
            response_loss_weight,
            response_boundary_loss_weight,
            response_boundary_tokens,
        )
    )
    if not _uses_custom_loss_weights(
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
    ):
        return None
    prompt_weight = prompt_loss_weight if train_prompt_tokens else 0.0
    weights = [prompt_weight] * max(0, int(prompt_len))
    response_weights = [response_loss_weight] * max(0, int(response_len))
    # The final response position is EOS for prompt/completion rows. Do not
    # boost EOS, or short arithmetic rescue examples teach "end immediately."
    boundary = min(max(0, len(response_weights) - 1), response_boundary_tokens)
    for idx in range(boundary):
        response_weights[idx] = response_loss_weight * response_boundary_loss_weight
    return weights + response_weights

def _compose_prompt_response_sample(prompt_ids, response_ids, eos_token_id, max_length: int,
                                    train_prompt_tokens: bool,
                                    prompt_loss_weight: float = 1.0,
                                    response_loss_weight: float = 1.0,
                                    response_boundary_loss_weight: float = 1.0,
                                    response_boundary_tokens: int = 0,
                                    min_response_tokens: int = 1,
                                    return_audit: bool = False):
    """
    Compose prompt/completion rows while preserving supervised answer tokens.

    Overlong instruction/input fields can otherwise fill the entire sequence and
    leave only an EOS after the response marker. For small assistant models that
    is especially harmful: it teaches "stop" where the answer should start.
    """
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        max_length = 0
    try:
        min_response_tokens = max(0, int(min_response_tokens or 0))
    except (TypeError, ValueError):
        min_response_tokens = 1

    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    original_prompt_tokens = len(prompt_ids)
    original_response_tokens = len(response_ids)
    if eos_token_id is None:
        eos_token_id = response_ids[-1] if response_ids else 0
    eos_token_id = int(eos_token_id)
    full_response_ids = response_ids + [eos_token_id]

    if max_length > 0 and len(prompt_ids) + len(full_response_ids) > max_length:
        if not full_response_ids:
            return None

        min_answer_tokens = min(min_response_tokens, len(response_ids))
        min_response_total = min(max_length, min_answer_tokens + 1)
        if min_response_total <= 0:
            min_response_total = min(max_length, 1)
        prompt_budget = max_length - min_response_total
        if prompt_budget < 0:
            return None
        if len(prompt_ids) > prompt_budget:
            # Preserve the prompt suffix because it contains the response marker.
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        response_budget = max_length - len(prompt_ids)
        if response_budget <= 0:
            return None
        if len(full_response_ids) > response_budget:
            full_response_ids = full_response_ids[:response_budget]
            full_response_ids[-1] = eos_token_id

    ids = prompt_ids + full_response_ids
    if not ids:
        return None

    prompt_labels = prompt_ids if train_prompt_tokens else ([-100] * len(prompt_ids))
    labels = prompt_labels + full_response_ids
    loss_weights = _prompt_response_loss_weights(
        len(prompt_ids),
        len(full_response_ids),
        train_prompt_tokens,
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
    )
    if not return_audit:
        return ids, labels, loss_weights
    audit = {
        "original_prompt_tokens": int(original_prompt_tokens),
        "original_response_tokens": int(original_response_tokens),
        "original_tokens": int(original_prompt_tokens + original_response_tokens + 1),
        "retained_prompt_tokens": int(len(prompt_ids)),
        # full_response_ids always ends in EOS. Exclude it from response-length
        # distributions so a one-token EOS is never reported as an answer.
        "retained_response_tokens": int(max(0, len(full_response_ids) - 1)),
        "retained_tokens": int(len(ids)),
        "truncated": bool(
            len(prompt_ids) != original_prompt_tokens
            or max(0, len(full_response_ids) - 1) != original_response_tokens
        ),
    }
    return ids, labels, loss_weights, audit

def _sample_effective_length(item, fallback_length: Optional[int] = None):
    if item is None:
        return fallback_length or 1

    for key in ("_length", "length", "valid_length", "seq_len"):
        if key in item and item[key] is not None:
            try:
                return max(1, int(item[key]))
            except (TypeError, ValueError):
                pass

    attention_mask = item.get("attention_mask")
    if attention_mask is not None:
        try:
            if isinstance(attention_mask, torch.Tensor):
                active = attention_mask.to(dtype=torch.bool).nonzero(as_tuple=False)
                length = int(active[-1].item()) + 1 if active.numel() > 0 else 0
            else:
                length = 0
                for idx, value in enumerate(attention_mask):
                    if int(value) != 0:
                        length = idx + 1
            if length > 0:
                return length
        except Exception:
            pass

    input_ids = item.get("input_ids")
    if input_ids is not None:
        try:
            return max(1, len(input_ids))
        except TypeError:
            pass

    return fallback_length or 1

def _yield_length_bucket(buffer, batch_size: int, shuffle: bool, generator: torch.Generator):
    if not buffer:
        return
    buffer.sort(key=_sample_effective_length, reverse=True)
    if shuffle and batch_size > 1:
        groups = [buffer[i:i + batch_size] for i in range(0, len(buffer), batch_size)]
        order = torch.randperm(len(groups), generator=generator).tolist() if len(groups) > 1 else [0]
        for group_idx in order:
            for item in groups[group_idx]:
                yield item
    else:
        for item in buffer:
            yield item


def _attach_precomputed_rosa(
    sample,
    vocab_size: int,
    chunk_size: int,
    max_context: int = 512,
    enforce_max_context: bool = False,
):
    """Attach exact cached ROSA ids inside a loader worker."""
    if sample is None:
        return None
    from hierarchos.utils.rosa import (
        precompute_rosa_ids_for_chunks,
        rosa_context_mode,
    )

    input_ids = sample.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        tokens = input_ids.detach().cpu().tolist()
    else:
        tokens = list(input_ids or [])
    sentinel = int(vocab_size)
    sample["rosa_ids"] = torch.tensor(
        precompute_rosa_ids_for_chunks(
            tokens,
            vocab_size=sentinel,
            chunk_size=max(1, int(chunk_size or len(tokens) or 1)),
            rosa_max_ctx=max(1, int(max_context or 512)),
            enforce_max_context=bool(enforce_max_context),
        ),
        dtype=torch.long,
    )
    sample["_rosa_sentinel"] = sentinel
    sample["_rosa_context_mode"] = rosa_context_mode(enforce_max_context)
    return sample

_TEXT_COLUMN_CANDIDATES = ("text", "content")
_PROMPT_COMPLETION_COLUMN_CANDIDATES = (
    ("instruction", "output"),
    ("prompt", "completion"),
    ("question", "answer"),
)

def _column_has_value(sample: dict, column: Optional[str]) -> bool:
    if not column or column not in sample:
        return False
    value = sample.get(column)
    return value is not None and str(value).strip() != ""

def _resolve_text_sample_columns(text_dict: dict, text_column: Optional[str] = None,
                                 prompt_column: Optional[str] = None,
                                 completion_column: Optional[str] = None,
                                 alpaca_mode: bool = False):
    if text_column:
        return text_column, None, None
    if alpaca_mode and not (prompt_column or completion_column):
        return None, "instruction", "output"
    if prompt_column or completion_column:
        return None, prompt_column, completion_column

    for candidate in _TEXT_COLUMN_CANDIDATES:
        if _column_has_value(text_dict, candidate):
            return candidate, None, None

    for candidate_prompt, candidate_completion in _PROMPT_COMPLETION_COLUMN_CANDIDATES:
        if (
            _column_has_value(text_dict, candidate_prompt)
            or _column_has_value(text_dict, candidate_completion)
        ):
            return None, candidate_prompt, candidate_completion

    return None, None, None

def _is_alpaca_prompt_pair(prompt_column: Optional[str], completion_column: Optional[str]) -> bool:
    return (
        str(prompt_column or "").lower() == "instruction"
        and str(completion_column or "").lower() == "output"
    )

def _bounded_audit_value(value, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        text = default
    return text[:256]

def _audit_source_name(text_dict: dict, default: Optional[str] = None) -> str:
    if isinstance(text_dict, dict):
        for key in ("source", "dataset_source", "dataset_name", "origin", "_source"):
            value = text_dict.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return _bounded_audit_value(value)
    return _bounded_audit_value(default, "unknown")

def _audit_schema_and_rejection(
    text_dict,
    *,
    text_column: Optional[str] = None,
    prompt_column: Optional[str] = None,
    completion_column: Optional[str] = None,
    alpaca_mode: bool = False,
    drop_empty_completions: bool = True,
):
    if not isinstance(text_dict, dict):
        return "invalid", "not_an_object"
    resolved_text, resolved_prompt, resolved_completion = _resolve_text_sample_columns(
        text_dict,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
    )
    if resolved_text:
        schema = f"text:{resolved_text}"
        if not _column_has_value(text_dict, resolved_text):
            return schema, "empty_text"
        return schema, None
    if resolved_prompt and resolved_completion:
        if _is_alpaca_prompt_pair(resolved_prompt, resolved_completion):
            schema = "alpaca:instruction-output"
        else:
            schema = f"pair:{resolved_prompt}-{resolved_completion}"
        prompt = str(text_dict.get(resolved_prompt, "") or "").strip()
        completion = str(text_dict.get(resolved_completion, "") or "").strip()
        inp = str(text_dict.get("input", "") or "").strip()
        if drop_empty_completions and not completion:
            return schema, "empty_completion"
        if not prompt and not completion and not inp:
            return schema, "empty_prompt_and_completion"
        return schema, None
    return "unknown", "missing_supported_schema"

def _make_audit_rejection(
    text_dict,
    *,
    source_default: Optional[str] = None,
    text_column: Optional[str] = None,
    prompt_column: Optional[str] = None,
    completion_column: Optional[str] = None,
    alpaca_mode: bool = False,
    drop_empty_completions: bool = True,
    reason: Optional[str] = None,
):
    schema, classified_reason = _audit_schema_and_rejection(
        text_dict,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
        drop_empty_completions=drop_empty_completions,
    )
    return {
        "_audit_only": True,
        "_audit": {
            "accepted": False,
            "schema": schema,
            "source": _audit_source_name(text_dict, source_default),
            "rejection_reason": reason or classified_reason or "formatting_or_tokenization_failure",
        },
    }

def _format_alpaca_prompt(instruction: str, inp: str) -> str:
    prompt = ""
    if inp:
        prompt += f"### Previous Context:\n{inp}\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n"
    return prompt + "### Response:\n"

def _estimate_text_token_length(text_dict: dict, max_length: int, kayla_mode: bool,
                                text_column: Optional[str] = None,
                                prompt_column: Optional[str] = None,
                                completion_column: Optional[str] = None,
                                alpaca_mode: bool = False):
    text_column, prompt_column, completion_column = _resolve_text_sample_columns(
        text_dict,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
    )
    if text_column:
        text_len = len(str(text_dict.get(text_column, "")))
    elif prompt_column and completion_column:
        text_len = len(str(text_dict.get(prompt_column, "")))
        text_len += len(str(text_dict.get(completion_column, "")))
        text_len += len(str(text_dict.get("input", "")))
        if kayla_mode:
            text_len += len(str(text_dict.get("feelings", "")))
            text_len += len(str(text_dict.get("thought-process", "")))
    else:
        return None

    approx_tokens = (text_len + 3) // 4 + 1
    return max(1, min(int(max_length), approx_tokens))

def _streaming_bucket_size(batch_size: int, use_length_bucketing: bool, bucket_size=None):
    if not use_length_bucketing or batch_size <= 1:
        return 0
    if bucket_size is not None:
        return max(batch_size, int(bucket_size))
    # Keep enough lookahead for dynamic padding wins. HF single-shard datasets
    # use the tokenized PT cache path, so this no longer forces repeated HF
    # tokenization on every epoch.
    return max(batch_size, int(batch_size) * 8)

def _resolve_jsonl_paths(path):
    path = os.fspath(path)
    if os.path.isdir(path):
        paths = []
        for root, _dirs, files in os.walk(path):
            for filename in files:
                lower = filename.lower()
                if lower == "manifest.jsonl":
                    continue
                if lower.endswith((".jsonl", ".ndjson")):
                    paths.append(os.path.join(root, filename))
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"No JSONL/NDJSON shard files found in directory: {path}")
        return paths
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return [path]

def _iter_jsonl_lines_for_worker(path: str, worker_id: int, num_workers: int):
    if num_workers <= 1:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                yield line_num, line
        return

    file_size = os.path.getsize(path)
    start_byte = (file_size * worker_id) // num_workers
    end_byte = (file_size * (worker_id + 1)) // num_workers
    with open(path, "rb") as f:
        if start_byte > 0:
            # Only discard a partial line. If the range starts immediately
            # after a newline, seeking to start_byte and calling readline()
            # would incorrectly drop the first complete record in this shard.
            f.seek(start_byte - 1)
            preceding_byte = f.read(1)
            if preceding_byte != b"\n":
                f.readline()
        else:
            f.seek(0)
        while True:
            if worker_id != num_workers - 1 and f.tell() >= end_byte:
                break
            raw_line = f.readline()
            if not raw_line:
                break
            try:
                yield None, raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue


def _assign_jsonl_shards(paths, num_workers: int):
    """Deterministically balance whole JSONL shards by byte size."""
    paths = list(paths)
    num_workers = max(1, int(num_workers))
    assignments = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    sized_paths = []
    for path_idx, path in enumerate(paths):
        try:
            size = max(0, int(os.path.getsize(path)))
        except OSError:
            size = 0
        sized_paths.append((path_idx, path, size))

    # Longest-processing-time scheduling sharply reduces tail-worker idle time
    # for uneven export shards. Stable input indexes make assignment repeatable.
    for path_idx, path, size in sorted(sized_paths, key=lambda row: (-row[2], row[0])):
        worker_id = min(range(num_workers), key=lambda idx: (loads[idx], idx))
        assignments[worker_id].append((path_idx, path))
        loads[worker_id] += size

    # Preserve source order within each worker so only the worker assignment,
    # not the local record order, changes.
    return [
        [path for _path_idx, path in sorted(worker_paths)]
        for worker_paths in assignments
    ]

def _iter_jsonl_shards_for_worker(paths, worker_id: int, num_workers: int):
    paths = list(paths)
    if not paths:
        return

    # Prefer whole-file assignment when there are enough physical shards. This
    # avoids multiple workers seeking/reading the same file at once.
    if num_workers <= 1 or len(paths) >= num_workers:
        assigned_paths = (
            paths
            if num_workers <= 1
            else _assign_jsonl_shards(paths, num_workers)[worker_id]
        )
        for path in assigned_paths:
            yield from _iter_jsonl_lines_for_worker(path, 0, 1)
        return

    # If a dataset has fewer files than workers, still split by byte range so
    # extra workers do useful work without duplicating samples.
    for path in paths:
        yield from _iter_jsonl_lines_for_worker(path, worker_id, num_workers)

class IterableChunkedJSONLDataset(IterableDataset):
    """
    An IterableDataset for loading pre-tokenized, chunked, masked, and padded
    data from a JSONL file line by line. Reduces RAM usage.
    """
    def __init__(self, path: str, max_length: int, bucket_size: int = 0,
                 batch_size: int = 1, shuffle_buckets: bool = True):
        super().__init__()
        self.path = path
        self.paths = _resolve_jsonl_paths(path)
        self.max_length = max_length
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle_buckets = bool(shuffle_buckets)
        self.seed = int(torch.initial_seed()) % (2**63 - 1)
        self.epoch = 0
        self._shared_epoch = _shared_epoch_counter()

    def set_epoch(self, epoch: int):
        _set_iterable_epoch(self, epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        print(f"[Worker {worker_id}/{num_workers}] Opening {len(self.paths)} chunked JSONL shard(s): {self.path}")
        skipped_count = 0
        processed_count = 0
        bucket = []
        generator = torch.Generator()
        epoch = _get_iterable_epoch(self)
        generator.manual_seed((self.seed + epoch + worker_id) % (2**63 - 1))

        try:
            for _line_num, line in _iter_jsonl_shards_for_worker(self.paths, worker_id, num_workers):
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if not all(k in obj for k in ["input_ids", "labels", "attention_mask"]):
                            skipped_count += 1
                            continue
                        seq_len = len(obj["input_ids"])
                        if seq_len != self.max_length:
                            skipped_count += 1
                            continue
                        attention_mask = obj["attention_mask"]
                        try:
                            valid_length = 0
                            for mask_idx, mask_value in enumerate(attention_mask):
                                if int(mask_value) != 0:
                                    valid_length = mask_idx + 1
                        except Exception:
                            valid_length = seq_len
                        processed_count += 1
                        item = {
                            "input_ids": torch.tensor(obj["input_ids"], dtype=torch.long),
                            "labels": torch.tensor(obj["labels"], dtype=torch.long),
                            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                            "_length": valid_length,
                        }
                        if self.bucket_size > 0:
                            bucket.append(item)
                            if len(bucket) >= self.bucket_size:
                                yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)
                                bucket.clear()
                        else:
                            yield item
                    except Exception:
                        skipped_count += 1
                        continue
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR: {e}")
            raise e
        if bucket:
            yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)
        print(f"[Worker {worker_id}] Finished. Processed: {processed_count}, Skipped: {skipped_count}")

def create_dataloader_for_chunked(path, max_length, batch_size, num_workers=0,
                                  use_length_bucketing=True, bucket_size=None,
                                  device=None, prefetch_factor=None):
    streaming_bucket_size = _streaming_bucket_size(batch_size, use_length_bucketing, bucket_size)
    dataset = IterableChunkedJSONLDataset(
        path,
        max_length=max_length,
        bucket_size=streaming_bucket_size,
        batch_size=batch_size,
        shuffle_buckets=True,
    )
    collate_fn_simple = functools.partial(_collate_training_batch, pad_token_id=0)
    pin_memory = _pin_memory_for_device(device)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_simple,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

class PTChunkedDataset(Dataset):
    def __init__(self, directory_path: str, max_length: int, cache_size: int = 2):
        super().__init__()
        self.directory_path = os.path.abspath(directory_path)
        self.max_length = max_length
        self.cache_size = max(1, int(cache_size or 1))
        self.chunk_pointers = []
        self.sample_lengths = []
        self.last_loaded_path = None
        self.last_loaded_data = None
        self._chunk_cache = OrderedDict()
        manifest_file = os.path.join(directory_path, "manifest.jsonl")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("Each PT manifest row must be a JSON object")
                relative_path = entry.get("file_path")
                if not isinstance(relative_path, str) or not relative_path:
                    raise ValueError("PT manifest row is missing a non-empty file_path")
                chunk_path = os.path.abspath(
                    os.path.join(self.directory_path, relative_path)
                )
                try:
                    inside_root = (
                        os.path.normcase(os.path.commonpath([
                            self.directory_path,
                            chunk_path,
                        ]))
                        == os.path.normcase(self.directory_path)
                    )
                except ValueError:
                    inside_root = False
                if not inside_root:
                    raise ValueError(
                        f"PT manifest path escapes its dataset directory: {relative_path!r}"
                    )
                try:
                    index_in_file = int(entry["index_in_file"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "PT manifest row has an invalid index_in_file"
                    ) from exc
                if index_in_file < 0:
                    raise ValueError("PT manifest index_in_file must be nonnegative")
                self.chunk_pointers.append((chunk_path, index_in_file))
                length = entry.get("length", entry.get("seq_len", entry.get("valid_length")))
                try:
                    self.sample_lengths.append(max(1, int(length)))
                except (TypeError, ValueError):
                    self.sample_lengths.append(None)

    def __len__(self): return len(self.chunk_pointers)
    def _get_chunk(self, path):
        cached = self._chunk_cache.get(path)
        if cached is not None:
            self._chunk_cache.move_to_end(path)
            self.last_loaded_path = path
            self.last_loaded_data = cached
            return cached

        data = _validate_pt_chunk_container(
            _load_tensor_artifact_weights_only(path),
            path,
        )
        self._chunk_cache[path] = data
        self._chunk_cache.move_to_end(path)
        while len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        self.last_loaded_path = path
        self.last_loaded_data = data
        return data

    def __getitem__(self, idx):
        path, index = self.chunk_pointers[idx]
        data = self._get_chunk(path)
        if index >= len(data):
            raise IndexError(
                f"PT manifest index {index} is outside chunk {path} "
                f"with {len(data)} sample(s)"
            )
        item = _validate_pt_chunk_item(data[index], path, index)
        length = self.sample_lengths[idx] if idx < len(self.sample_lengths) else None
        if length is not None and "_length" not in item:
            item = dict(item)
            item["_length"] = length
        return item
    def get_sample_lengths(self):
        if self.sample_lengths and all(length is not None for length in self.sample_lengths):
            return self.sample_lengths

        lengths = list(self.sample_lengths) if self.sample_lengths else [None] * len(self.chunk_pointers)
        positions_by_path = {}
        for position, (path, index) in enumerate(self.chunk_pointers):
            if lengths[position] is None:
                positions_by_path.setdefault(path, []).append((position, index))

        for path, entries in positions_by_path.items():
            try:
                data = _validate_pt_chunk_container(
                    _load_tensor_artifact_weights_only(path),
                    path,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not safely inspect PT training chunk {path}: {exc}"
                ) from exc
            for position, index in entries:
                try:
                    if index < 0 or index >= len(data):
                        raise IndexError(index)
                    item = _validate_pt_chunk_item(data[index], path, index)
                    lengths[position] = _sample_effective_length(
                        item,
                        self.max_length,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid PT training sample {path}[{index}]: {exc}"
                    ) from exc

        self.sample_lengths = lengths
        return self.sample_lengths

def create_dataloader_pt_chunked(directory_path, max_length, batch_size, num_workers=0,
                                 use_length_bucketing=True, bucket_size=None,
                                 device=None, prefetch_factor=None, cache_size: int = 2):
    dataset = PTChunkedDataset(directory_path, max_length=max_length, cache_size=cache_size)
    collate_fn_pt = functools.partial(_collate_training_batch, pad_token_id=0)
    use_cuda = _pin_memory_for_device(device)
    drop_last = False
    lengths = _get_dataset_sample_lengths(dataset) if use_length_bucketing and batch_size > 1 else None
    if lengths is not None and len(lengths) > 0:
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            bucket_size=bucket_size,
            preserve_order=True,
        )
        return _create_dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_pt,
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=prefetch_factor,
        )

    sampler = EpochShuffleSampler(dataset, shuffle=True)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_pt,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=prefetch_factor,
    )

class TokenizedBinaryDataset(Dataset):
    """
    Random-access token cache backed by one binary file plus a compact index.

    Legacy records store variable-length int32 input ids followed by int32
    labels and optional inline weights/ROSA ids. Schema v6 records use lossless
    uint16 ids when the vocabulary fits and keep loss weights as palette-coded
    runs in the index. Attention masks are generated from each record length,
    so neither format stores padded max_length tensors.
    """
    def __init__(self, directory_path: str, max_length: Optional[int] = None,
                 pad_token_id: int = 0):
        super().__init__()
        self._file = None
        self._mmap = None
        self.directory_path = directory_path
        self.max_length = int(max_length or 0)
        self.pad_token_id = int(pad_token_id)
        index_path = os.path.join(directory_path, "index.pt")
        data_path = os.path.join(directory_path, "tokens.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Token cache index not found: {index_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Token cache data file not found: {data_path}")

        index = load_tensor_payload_safely(index_path, map_location="cpu")
        if not isinstance(index, dict) or "offsets" not in index or "lengths" not in index:
            raise ValueError("Token cache index must contain offsets and lengths")
        self.offsets = index["offsets"].to(device="cpu", dtype=torch.long).reshape(-1).contiguous()
        raw_lengths = index["lengths"].to(device="cpu", dtype=torch.long).reshape(-1).contiguous()
        if self.offsets.numel() != raw_lengths.numel():
            raise ValueError("Token cache index offsets/lengths size mismatch")
        if self.offsets.numel() == 0:
            raise ValueError("Token cache index contains no samples")
        if bool((self.offsets < 0).any().item()) or bool((raw_lengths <= 0).any().item()):
            raise ValueError("Token cache offsets must be nonnegative and lengths must be positive")
        try:
            self.storage_schema_version = int(index.get("storage_schema_version", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Token cache has an invalid storage schema version") from exc
        if self.storage_schema_version < 0 or self.storage_schema_version > 6:
            raise ValueError(
                f"Unsupported token-cache storage schema version: {self.storage_schema_version}"
            )
        self.has_rosa_ids = bool(index.get("has_rosa_ids", False))
        self.has_loss_weights = bool(index.get("has_loss_weights", False))
        self.rosa_sentinel = int(index.get("rosa_sentinel", 0))
        self.rosa_ids_context_mode = str(
            index.get("rosa_ids_context_mode")
            or ROSA_UNBOUNDED_CONTEXT_MODE
        )
        if self.rosa_ids_context_mode not in {
            ROSA_BOUNDED_CONTEXT_MODE,
            ROSA_UNBOUNDED_CONTEXT_MODE,
        }:
            raise ValueError(
                "Token cache has an unsupported ROSA context mode: "
                f"{self.rosa_ids_context_mode!r}"
            )
        self.label_encoding = index.get("label_encoding")
        if self.label_encoding in ("", "None", "none"):
            self.label_encoding = None
        self.labels_alias_input_ids = self.label_encoding == "input_ids_alias"

        if self.storage_schema_version == 6:
            byte_order = str(index.get("byte_order", ""))
            if byte_order != "little":
                raise ValueError(
                    "Token-cache schema v6 requires byte_order='little'"
                )
            if sys.byteorder != "little":
                raise RuntimeError(
                    "Token-cache schema v6 cannot be memory-mapped on a big-endian host"
                )
            self._token_torch_dtype, self._token_bytes = self._parse_integer_dtype(
                index.get("token_dtype"), "token"
            )
            if self.labels_alias_input_ids:
                if index.get("label_dtype") not in (None, "None", "none", ""):
                    raise ValueError(
                        "Input-aliased token-cache labels cannot declare label_dtype"
                    )
                self._label_torch_dtype = None
                self._label_bytes = 0
            else:
                if self.label_encoding is not None:
                    raise ValueError(
                        f"Unsupported token-cache label encoding: {self.label_encoding}"
                    )
                self._label_torch_dtype, self._label_bytes = self._parse_integer_dtype(
                    index.get("label_dtype"), "label"
                )
            if self.has_rosa_ids:
                self._rosa_torch_dtype, self._rosa_bytes = self._parse_integer_dtype(
                    index.get("rosa_dtype"), "ROSA"
                )
            else:
                if index.get("rosa_dtype") not in (None, "None", "none", ""):
                    raise ValueError(
                        "Token cache declares rosa_dtype without has_rosa_ids"
                    )
                self._rosa_torch_dtype = None
                self._rosa_bytes = 0
            if self.labels_alias_input_ids:
                if index.get("label_ignore_sentinel") not in (None, -100):
                    raise ValueError(
                        "Input-aliased token-cache labels cannot declare a storage sentinel"
                    )
                self.label_ignore_sentinel = -100
            else:
                default_ignore = 65535 if self._label_torch_dtype == torch.uint16 else -100
                self.label_ignore_sentinel = int(
                    index.get("label_ignore_sentinel", default_ignore)
                )
                if self._label_torch_dtype == torch.uint16:
                    if self.label_ignore_sentinel != 65535:
                        raise ValueError(
                            "uint16 token-cache labels must reserve 65535 as ignore_index"
                        )
                elif self.label_ignore_sentinel != -100:
                    raise ValueError(
                        "int32 token-cache labels must use -100 as ignore_index"
                    )
        else:
            # All pre-v6 cache writers used native int32 token streams.
            if self.label_encoding is not None:
                raise ValueError("Legacy token caches cannot declare a label encoding")
            self._token_torch_dtype = torch.int32
            self._label_torch_dtype = torch.int32
            self._rosa_torch_dtype = torch.int32 if self.has_rosa_ids else None
            self._token_bytes = 4
            self._label_bytes = 4
            self._rosa_bytes = 4 if self.has_rosa_ids else 0
            self.label_ignore_sentinel = -100

        self.loss_weight_encoding = index.get("loss_weight_encoding")
        if self.loss_weight_encoding in ("", "None", "none"):
            self.loss_weight_encoding = None
        self._loss_weights_are_rle = (
            self.has_loss_weights
            and self.loss_weight_encoding == "float32_palette_rle"
        )
        self.loss_weight_dtype = str(index.get("loss_weight_dtype", "float16"))
        self._loss_weight_torch_dtype = None
        self._loss_weight_bytes = 0
        self.loss_weight_palette = None
        self.loss_run_offsets = None
        self.loss_run_ends = None
        self.loss_run_codes = None
        if self.has_loss_weights and self._loss_weights_are_rle:
            if self.storage_schema_version != 6:
                raise ValueError(
                    "Palette-RLE loss weights require token-cache storage schema v6"
                )
            self._load_and_validate_loss_weight_runs(index, raw_lengths)
        elif self.has_loss_weights:
            if self.storage_schema_version == 6:
                raise ValueError(
                    "Token-cache schema v6 with loss weights requires "
                    "loss_weight_encoding='float32_palette_rle'"
                )
            if self.loss_weight_encoding is not None:
                raise ValueError(
                    f"Unsupported token-cache loss weight encoding: {self.loss_weight_encoding}"
                )
            if self.loss_weight_dtype == "float32":
                self._loss_weight_torch_dtype = torch.float32
                self._loss_weight_bytes = 4
            elif self.loss_weight_dtype in ("float16", "None", "none"):
                self._loss_weight_torch_dtype = torch.float16
                self._loss_weight_bytes = 2
            else:
                raise ValueError(
                    f"Unsupported token-cache loss weight dtype: {self.loss_weight_dtype}"
                )
        elif self.loss_weight_encoding is not None:
            raise ValueError(
                "Token cache declares a loss-weight encoding without loss weights"
            )
        if int(raw_lengths.max().item()) <= torch.iinfo(torch.int32).max:
            self.lengths = raw_lengths.to(dtype=torch.int32)
        else:
            self.lengths = raw_lengths
        if self.has_rosa_ids and self.rosa_sentinel <= 0:
            raise ValueError("ROSA token cache is missing a valid vocabulary sentinel")

        bytes_per_token = self._token_bytes + self._label_bytes + self._rosa_bytes
        if self.has_loss_weights and not self._loss_weights_are_rle:
            bytes_per_token += self._loss_weight_bytes
        record_bytes = self.lengths.to(dtype=torch.long) * int(bytes_per_token)
        offsets_valid = int(self.offsets[0].item()) == 0
        if self.offsets.numel() > 1:
            offsets_valid = offsets_valid and torch.equal(
                self.offsets[1:],
                self.offsets[:-1] + record_bytes[:-1],
            )
        if not offsets_valid:
            raise ValueError(
                "Token cache offsets do not match the declared record layout; "
                "the cache may be stale or partially written"
            )
        expected_offset = int((self.offsets[-1] + record_bytes[-1]).item())
        data_size = os.path.getsize(data_path)
        if data_size != expected_offset:
            raise ValueError(
                f"Token cache data size mismatch: expected {expected_offset} bytes, "
                f"found {data_size}. Rebuild the cache before training."
            )
        if self.max_length > 0:
            self.sample_lengths = self.lengths.clamp(max=self.max_length).contiguous()
        else:
            self.sample_lengths = self.lengths
        self.data_path = data_path

    @staticmethod
    def _parse_integer_dtype(dtype_name, field_name):
        dtype_name = str(dtype_name)
        if dtype_name == "uint16":
            return torch.uint16, 2
        if dtype_name == "int32":
            return torch.int32, 4
        raise ValueError(
            f"Unsupported token-cache {field_name} dtype: {dtype_name}"
        )

    def _load_and_validate_loss_weight_runs(self, index, raw_lengths):
        palette_value = index.get("loss_weight_palette")
        try:
            palette = torch.as_tensor(palette_value, dtype=torch.float32, device="cpu").reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Token cache has an invalid loss-weight palette") from exc
        if palette.numel() <= 0 or palette.numel() > 255:
            raise ValueError("Token-cache loss-weight palette must contain 1 to 255 values")
        if not bool(torch.isfinite(palette).all().item()) or bool((palette < 0).any().item()):
            raise ValueError("Token-cache loss-weight palette must contain finite nonnegative values")

        required = ("loss_run_offsets", "loss_run_ends", "loss_run_codes")
        if any(name not in index for name in required):
            raise ValueError("Token cache is missing palette-RLE loss-weight metadata")
        try:
            run_offsets = torch.as_tensor(
                index["loss_run_offsets"], device="cpu"
            ).reshape(-1).contiguous()
            run_ends_long = torch.as_tensor(
                index["loss_run_ends"], device="cpu"
            ).reshape(-1).contiguous()
            run_codes_long = torch.as_tensor(
                index["loss_run_codes"], device="cpu"
            ).reshape(-1).contiguous()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Token cache has invalid palette-RLE loss-weight tensors") from exc
        for name, value in (
            ("loss-run offsets", run_offsets),
            ("loss-run ends", run_ends_long),
            ("loss-run codes", run_codes_long),
        ):
            if value.dtype not in _INTEGER_LENGTH_DTYPES:
                raise ValueError(f"Token-cache {name} must use an integer dtype")
        # Offsets participate in indexing and subtraction throughout validation;
        # retain compact int32 ends/uint8 codes but canonicalize offsets to long.
        run_offsets = run_offsets.to(dtype=torch.long)

        sample_count = int(raw_lengths.numel())
        validation_chunk = 1_048_576
        if run_offsets.numel() != sample_count + 1:
            raise ValueError(
                "Token-cache loss-run offsets must contain one entry per sample plus one"
            )
        offsets_nonnegative = True
        for start in range(0, run_offsets.numel(), validation_chunk):
            if bool((run_offsets[start:start + validation_chunk] < 0).any().item()):
                offsets_nonnegative = False
                break
        if int(run_offsets[0].item()) != 0 or not offsets_nonnegative:
            raise ValueError("Token-cache loss-run offsets must start at zero and be nonnegative")
        for start in range(0, sample_count, validation_chunk):
            stop = min(sample_count, start + validation_chunk)
            if bool((run_offsets[start + 1:stop + 1] <= run_offsets[start:stop]).any().item()):
                raise ValueError("Every token-cache sample must contain at least one loss-weight run")
        run_count = int(run_offsets[-1].item())
        if run_count != run_ends_long.numel() or run_count != run_codes_long.numel():
            raise ValueError("Token-cache loss-run offsets/ends/codes size mismatch")
        for start in range(0, run_count, validation_chunk):
            stop = min(run_count, start + validation_chunk)
            ends_chunk = run_ends_long[start:stop]
            codes_chunk = run_codes_long[start:stop]
            if bool((ends_chunk <= 0).any().item()):
                raise ValueError("Token-cache loss-run ends must be positive")
            if bool((codes_chunk < 0).any().item()) or bool(
                (codes_chunk >= palette.numel()).any().item()
            ):
                raise ValueError("Token-cache loss-run code is outside the declared palette")

        for start in range(0, sample_count, validation_chunk):
            stop = min(sample_count, start + validation_chunk)
            final_run_indices = run_offsets[start + 1:stop + 1] - 1
            if not torch.equal(
                run_ends_long.index_select(0, final_run_indices),
                raw_lengths[start:stop],
            ):
                raise ValueError(
                    "Each token-cache sample's final loss run must end at its stored length"
                )
        if run_count > 1:
            sample_boundaries = run_offsets[1:-1]
            for start in range(1, run_count, validation_chunk):
                stop = min(run_count, start + validation_chunk)
                invalid_order = (
                    run_ends_long[start:stop]
                    <= run_ends_long[start - 1:stop - 1]
                )
                if sample_boundaries.numel() > 0:
                    left = int(torch.searchsorted(
                        sample_boundaries,
                        torch.tensor(start, dtype=sample_boundaries.dtype),
                    ).item())
                    right = int(torch.searchsorted(
                        sample_boundaries,
                        torch.tensor(stop, dtype=sample_boundaries.dtype),
                    ).item())
                    if right > left:
                        boundary_positions = (
                            sample_boundaries[left:right] - start
                        )
                        invalid_order[boundary_positions] = False
                if bool(invalid_order.any().item()):
                    raise ValueError(
                        "Token-cache loss-run ends must increase strictly within each sample"
                    )

        self.loss_weight_palette = palette.contiguous()
        self._loss_weight_palette_values = tuple(
            float(value) for value in self.loss_weight_palette.tolist()
        )
        self.loss_run_offsets = run_offsets
        if int(run_ends_long.max().item()) <= torch.iinfo(torch.int32).max:
            self.loss_run_ends = run_ends_long.to(dtype=torch.int32).contiguous()
        else:
            self.loss_run_ends = run_ends_long
        self.loss_run_codes = run_codes_long.to(dtype=torch.uint8).contiguous()

    def __len__(self):
        return int(self.lengths.numel())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        state["_mmap"] = None
        return state

    def _ensure_open(self):
        if self._mmap is not None:
            return
        self._file = open(self.data_path, "rb")
        # Length-grouped epoch shuffling produces random record reads. Tell the
        # Linux kernel not to spend I/O bandwidth on sequential readahead pages
        # that a multi-billion-token cache is unlikely to consume next.
        posix_fadvise = getattr(os, "posix_fadvise", None)
        random_advice = getattr(os, "POSIX_FADV_RANDOM", None)
        if callable(posix_fadvise) and random_advice is not None:
            try:
                posix_fadvise(self._file.fileno(), 0, 0, random_advice)
            except OSError:
                pass
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        madvise = getattr(self._mmap, "madvise", None)
        mmap_random = getattr(mmap, "MADV_RANDOM", None)
        if callable(madvise) and mmap_random is not None:
            try:
                madvise(mmap_random)
            except (OSError, ValueError):
                pass

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self.close()

    def _integer_view(
        self,
        offset,
        count,
        dtype,
        *,
        decode_labels=False,
        promote_uint16=True,
    ):
        view = torch.frombuffer(
            self._mmap,
            dtype=dtype,
            count=count,
            offset=offset,
        )
        # Preserve the legacy scalar-item contract (int32 ids/labels) while
        # decoding the uint16 ignore sentinel before it reaches a collator.
        if dtype == torch.uint16 and promote_uint16:
            view = view.to(dtype=torch.int32)
        if decode_labels and self.label_ignore_sentinel != -100:
            if view.dtype != torch.int32:
                view = view.to(dtype=torch.int32)
            view = view.clone() if view._base is not None else view
            view.masked_fill_(view == self.label_ignore_sentinel, -100)
        return view

    def _copy_rle_loss_weights(self, idx, length, destination):
        run_start = int(self.loss_run_offsets[idx].item())
        run_stop = int(self.loss_run_offsets[idx + 1].item())
        run_ends = self.loss_run_ends[run_start:run_stop].tolist()
        run_codes = self.loss_run_codes[run_start:run_stop].tolist()
        position = 0
        for raw_end, code in zip(run_ends, run_codes):
            end = min(int(raw_end), int(length))
            if end > position:
                destination[position:end].fill_(self._loss_weight_palette_values[int(code)])
                position = end
            if position >= int(length):
                break
        if position != int(length):
            # Initialization validates this invariant; retain a local guard so
            # metadata modified after construction cannot produce silent loss
            # reweighting.
            raise RuntimeError("Token-cache loss runs do not cover the requested sample length")

    def _decode_rle_loss_weights(self, idx, length):
        weights = torch.empty(int(length), dtype=torch.float32)
        self._copy_rle_loss_weights(idx, length, weights)
        return weights

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Token cache index out of range: {idx}")
        self._ensure_open()
        offset = int(self.offsets[idx])
        stored_length = int(self.lengths[idx])
        length = stored_length
        if self.max_length > 0:
            length = min(length, self.max_length)
        if length <= 0:
            return None

        label_offset = offset + (stored_length * self._token_bytes)
        after_labels_offset = label_offset + (stored_length * self._label_bytes)
        input_ids = self._integer_view(
            offset,
            length,
            self._token_torch_dtype,
        )
        if self.labels_alias_input_ids:
            labels = input_ids.clone()
        else:
            labels = self._integer_view(
                label_offset,
                length,
                self._label_torch_dtype,
                decode_labels=True,
            )
        item = {
            "input_ids": input_ids,
            "labels": labels,
            "_length": length,
        }
        if self.has_loss_weights:
            if self._loss_weights_are_rle:
                item["loss_weights"] = self._decode_rle_loss_weights(idx, length)
            else:
                item["loss_weights"] = torch.frombuffer(
                    self._mmap,
                    dtype=self._loss_weight_torch_dtype,
                    count=length,
                    offset=after_labels_offset,
                ).to(dtype=torch.float32)
        if self.has_rosa_ids:
            rosa_offset = after_labels_offset
            if self.has_loss_weights and not self._loss_weights_are_rle:
                rosa_offset += stored_length * self._loss_weight_bytes
            item["rosa_ids"] = self._integer_view(
                rosa_offset,
                length,
                self._rosa_torch_dtype,
            )
            item["_rosa_sentinel"] = self.rosa_sentinel
            item["_rosa_context_mode"] = self.rosa_ids_context_mode
        return item

    def __getitems__(self, indices):
        """Fetch and collate one binary-cache batch directly from the mmap.

        PyTorch's map-style fetcher passes a batch sampler's indices to this
        method when it is available. Legacy caches retain their historical long
        batch tensors. Compact v6 caches keep ids/labels/ROSA as int32 and masks
        as bool through pinned-memory and H2D transfer; labels are widened on
        the GPU immediately before the language-model objective.
        """
        compact_batch = self.storage_schema_version == 6
        integer_batch_dtype = torch.int32 if compact_batch else torch.long
        mask_batch_dtype = torch.bool if compact_batch else torch.long
        if isinstance(indices, torch.Tensor):
            indices = indices.reshape(-1).tolist()
        else:
            indices = list(indices)
        if not indices:
            return {
                "input_ids": torch.empty((0, 0), dtype=integer_batch_dtype),
                "labels": torch.empty((0, 0), dtype=integer_batch_dtype),
                "attention_mask": torch.empty((0, 0), dtype=mask_batch_dtype),
            }

        sample_count = len(self)
        normalized_indices = []
        stored_lengths = []
        effective_lengths = []
        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0:
                idx += sample_count
            if idx < 0 or idx >= sample_count:
                raise IndexError(f"Token cache index out of range: {raw_idx}")
            stored_length = int(self.lengths[idx])
            length = min(stored_length, self.max_length) if self.max_length > 0 else stored_length
            normalized_indices.append(idx)
            stored_lengths.append(stored_length)
            effective_lengths.append(length)

        self._ensure_open()
        batch_size = len(normalized_indices)
        max_length = max(effective_lengths)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=integer_batch_dtype,
        )
        labels = torch.full((batch_size, max_length), -100, dtype=integer_batch_dtype)
        attention_mask = torch.zeros((batch_size, max_length), dtype=mask_batch_dtype)
        loss_weights = (
            torch.zeros((batch_size, max_length), dtype=torch.float32)
            if self.has_loss_weights else None
        )
        rosa_ids = (
            torch.full(
                (batch_size, max_length),
                self.rosa_sentinel,
                dtype=integer_batch_dtype,
            )
            if self.has_rosa_ids else None
        )

        # Read each batch in file-offset order to turn 64 random mmap seeks into
        # the most sequential access pattern available for the same batch. Rows
        # are copied back to their original positions, so sampler order and all
        # model inputs remain unchanged.
        fetch_plan = [
            (row, idx, stored_length, length)
            for row, (idx, stored_length, length) in enumerate(zip(
                normalized_indices,
                stored_lengths,
                effective_lengths,
            ))
        ]
        fetch_plan.sort(key=lambda entry: int(self.offsets[entry[1]]))
        for row, idx, stored_length, length in fetch_plan:
            offset = int(self.offsets[idx])
            label_offset = offset + (stored_length * self._token_bytes)
            after_labels_offset = label_offset + (stored_length * self._label_bytes)

            input_view = self._integer_view(
                offset,
                length,
                self._token_torch_dtype,
                promote_uint16=False,
            )
            input_ids[row, :length].copy_(input_view)
            if self.labels_alias_input_ids:
                labels[row, :length].copy_(input_view)
            else:
                label_view = self._integer_view(
                    label_offset,
                    length,
                    self._label_torch_dtype,
                    promote_uint16=False,
                )
                labels[row, :length].copy_(label_view)
                if self.label_ignore_sentinel != -100:
                    labels[row, :length].masked_fill_(
                        labels[row, :length] == self.label_ignore_sentinel,
                        -100,
                    )
            attention_mask[row, :length] = 1

            if loss_weights is not None:
                if self._loss_weights_are_rle:
                    self._copy_rle_loss_weights(
                        idx,
                        length,
                        loss_weights[row, :length],
                    )
                else:
                    weight_view = torch.frombuffer(
                        self._mmap,
                        dtype=self._loss_weight_torch_dtype,
                        count=length,
                        offset=after_labels_offset,
                    )
                    loss_weights[row, :length].copy_(weight_view)

            if rosa_ids is not None:
                rosa_offset = after_labels_offset
                if self.has_loss_weights and not self._loss_weights_are_rle:
                    rosa_offset += stored_length * self._loss_weight_bytes
                rosa_view = self._integer_view(
                    rosa_offset,
                    length,
                    self._rosa_torch_dtype,
                    promote_uint16=False,
                )
                rosa_ids[row, :length].copy_(rosa_view)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if loss_weights is not None:
            batch["loss_weights"] = loss_weights
        if rosa_ids is not None:
            batch["rosa_ids"] = rosa_ids
            batch["rosa_ids_context_mode"] = self.rosa_ids_context_mode
        return batch

    def get_sample_lengths(self):
        # Preserve the historical public list API for direct callers without
        # retaining a second multi-million-element Python list in memory. Loader
        # construction reads the tensor-valued ``sample_lengths`` attribute
        # directly through _get_dataset_sample_lengths.
        return self.sample_lengths.tolist()


def create_dataloader_for_tokenized_cache(directory_path, max_length, batch_size, pad_token_id,
                                          num_workers=0, use_length_bucketing=True,
                                          bucket_size=None, device=None, prefetch_factor=None):
    dataset = TokenizedBinaryDataset(
        directory_path,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    collate = functools.partial(_collate_tokenized_binary_batch, pad_token_id=pad_token_id)
    use_cuda = _pin_memory_for_device(device)
    # Keep the final incomplete batch: silently dropping up to batch_size-1
    # samples violates complete epoch coverage. One smaller terminal batch may
    # compile a second shape, but that once-per-epoch cost is negligible.
    drop_last = False
    lengths = _get_dataset_sample_lengths(dataset) if use_length_bucketing and batch_size > 1 else None
    if lengths is not None and len(lengths) > 0:
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            bucket_size=bucket_size,
        )
        return _create_dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=prefetch_factor,
        )

    sampler = EpochShuffleSampler(dataset, shuffle=True)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )

def process_text_sample(tokenizer, text_dict: dict, max_length: int, kayla_mode: bool = False,
                         text_column: Optional[str] = None,
                         prompt_column: Optional[str] = None, completion_column: Optional[str] = None,
                         alpaca_mode: bool = False,
                         train_prompt_tokens: bool = True,
                         prompt_loss_weight: float = 1.0,
                         response_loss_weight: float = 1.0,
                         response_boundary_loss_weight: float = 1.0,
                         response_boundary_tokens: int = 0,
                         min_response_tokens: int = 1,
                         drop_empty_completions: bool = True,
                         include_audit: bool = False,
                         audit_source_default: Optional[str] = None):
    requested_columns = (text_column, prompt_column, completion_column)

    def reject(reason):
        if not include_audit:
            return None
        return _make_audit_rejection(
            text_dict,
            source_default=audit_source_default,
            text_column=requested_columns[0],
            prompt_column=requested_columns[1],
            completion_column=requested_columns[2],
            alpaca_mode=alpaca_mode,
            drop_empty_completions=drop_empty_completions,
            reason=reason,
        )

    try:
        if not isinstance(text_dict, dict):
            return reject("not_an_object")
        try:
            minimum_response_tokens = max(0, int(min_response_tokens or 0))
        except (TypeError, ValueError):
            minimum_response_tokens = 1
        prompt_loss_weight, response_loss_weight, response_boundary_loss_weight, response_boundary_tokens = (
            _normalize_loss_weight_args(
                prompt_loss_weight,
                response_loss_weight,
                response_boundary_loss_weight,
                response_boundary_tokens,
            )
        )
        loss_weights = None
        composition_audit = None
        schema, _schema_rejection = _audit_schema_and_rejection(
            text_dict,
            text_column=requested_columns[0],
            prompt_column=requested_columns[1],
            completion_column=requested_columns[2],
            alpaca_mode=alpaca_mode,
            drop_empty_completions=drop_empty_completions,
        )

        text_column, prompt_column, completion_column = _resolve_text_sample_columns(
            text_dict,
            text_column=text_column,
            prompt_column=prompt_column,
            completion_column=completion_column,
            alpaca_mode=alpaca_mode,
        )

        if text_column:
            text = str(text_dict.get(text_column, ""))
            if not text.strip():
                return reject("empty_text")
            ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
            labels = list(ids)
            original_text_tokens = len(ids)
            composition_audit = {
                "original_prompt_tokens": 0,
                "original_response_tokens": max(0, original_text_tokens - 1),
                "original_tokens": original_text_tokens,
                "retained_prompt_tokens": 0,
                "retained_response_tokens": max(0, original_text_tokens - 1),
                "retained_tokens": original_text_tokens,
                "truncated": False,
            }
            if _uses_custom_loss_weights(
                prompt_loss_weight,
                response_loss_weight,
                response_boundary_loss_weight,
                response_boundary_tokens,
            ):
                loss_weights = [response_loss_weight] * len(ids)
        elif prompt_column and completion_column:
            instruction = str(text_dict.get(prompt_column, ""))
            output = str(text_dict.get(completion_column, ""))
            inp = str(text_dict.get('input', "")).strip()
            if drop_empty_completions and not output.strip():
                return reject("empty_completion")
            if not instruction.strip() and not output.strip() and not inp:
                return reject("empty_prompt_and_completion")
            if kayla_mode:
                feelings = str(text_dict.get('feelings', ''))
                thought = str(text_dict.get('thought-process', ''))
                prompt_text = f"### Instruction:\n{instruction}\n\n" + (f"### Feelings:\n{feelings}\n\n" if feelings else "")
                thought_text = f"### Thought Process:\n{thought}\n\n"
                response_text = f"### Response:\n{output}"
                p_ids = tokenizer.encode(prompt_text)
                t_ids = tokenizer.encode(thought_text, add_special_tokens=False)
                r_ids = tokenizer.encode(response_text, add_special_tokens=False)
                output_ids = tokenizer.encode(output, add_special_tokens=False)
                if output.strip() and len(output_ids) < minimum_response_tokens:
                    return reject("response_below_minimum")
                composed = _compose_prompt_response_sample(
                    p_ids,
                    t_ids + r_ids,
                    tokenizer.eos_token_id,
                    max_length,
                    train_prompt_tokens,
                    prompt_loss_weight,
                    response_loss_weight,
                    response_boundary_loss_weight,
                    response_boundary_tokens,
                    minimum_response_tokens,
                    return_audit=include_audit,
                )
                if composed is None:
                    return reject("composition_failed")
                if include_audit:
                    ids, labels, loss_weights, composition_audit = composed
                    composition_audit["original_response_tokens"] = len(output_ids)
                else:
                    ids, labels, loss_weights = composed
            elif alpaca_mode or _is_alpaca_prompt_pair(prompt_column, completion_column):
                prompt = _format_alpaca_prompt(instruction, inp)
                p_ids = tokenizer.encode(prompt)
                c_ids = tokenizer.encode(output, add_special_tokens=False)
                if output.strip() and len(c_ids) < minimum_response_tokens:
                    return reject("response_below_minimum")
                composed = _compose_prompt_response_sample(
                    p_ids,
                    c_ids,
                    tokenizer.eos_token_id,
                    max_length,
                    train_prompt_tokens,
                    prompt_loss_weight,
                    response_loss_weight,
                    response_boundary_loss_weight,
                    response_boundary_tokens,
                    minimum_response_tokens,
                    return_audit=include_audit,
                )
                if composed is None:
                    return reject("composition_failed")
                if include_audit:
                    ids, labels, loss_weights, composition_audit = composed
                else:
                    ids, labels, loss_weights = composed
            else:
                if inp:
                    prompt = f"User: {inp}\n\nUser: {instruction}\n\nAssistant: "
                else:
                    prompt = f"User: {instruction}\n\nAssistant: "
                p_ids = tokenizer.encode(prompt)
                c_ids = tokenizer.encode(output, add_special_tokens=False)
                if output.strip() and len(c_ids) < minimum_response_tokens:
                    return reject("response_below_minimum")
                composed = _compose_prompt_response_sample(
                    p_ids,
                    c_ids,
                    tokenizer.eos_token_id,
                    max_length,
                    train_prompt_tokens,
                    prompt_loss_weight,
                    response_loss_weight,
                    response_boundary_loss_weight,
                    response_boundary_tokens,
                    minimum_response_tokens,
                    return_audit=include_audit,
                )
                if composed is None:
                    return reject("composition_failed")
                if include_audit:
                    ids, labels, loss_weights, composition_audit = composed
                else:
                    ids, labels, loss_weights = composed
        else:
            return reject("missing_supported_schema")
        if len(ids) > max_length:
            ids = ids[:max_length-1] + [tokenizer.eos_token_id]
            labels = labels[:max_length-1] + [tokenizer.eos_token_id]
            if loss_weights is not None:
                eos_weight = loss_weights[-1] if loss_weights else 1.0
                loss_weights = loss_weights[:max_length-1] + [eos_weight]
            if composition_audit is not None:
                composition_audit["retained_tokens"] = len(ids)
                composition_audit["truncated"] = True
        if (
            prompt_column
            and completion_column
            and str(text_dict.get(completion_column, "") or "").strip()
            and composition_audit is not None
            and int(composition_audit.get("retained_response_tokens", 0)) < minimum_response_tokens
        ):
            return reject("retained_response_below_minimum")
        sample = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "_length": len(ids),
        }
        if loss_weights is not None:
            if len(loss_weights) < len(ids):
                loss_weights = loss_weights + ([1.0] * (len(ids) - len(loss_weights)))
            elif len(loss_weights) > len(ids):
                loss_weights = loss_weights[:len(ids)]
            sample["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        if include_audit:
            active_labels = sample["labels"][1:].ne(-100)
            supervised_tokens = int(active_labels.sum().item())
            if "loss_weights" in sample:
                weighted_tokens = float(
                    sample["loss_weights"][1:][active_labels].sum().item()
                )
            else:
                weighted_tokens = float(supervised_tokens)
            composition_audit = dict(composition_audit or {})
            composition_audit.update({
                "accepted": True,
                "schema": schema,
                "source": _audit_source_name(text_dict, audit_source_default),
                "retained_tokens": int(len(ids)),
                "supervised_tokens": supervised_tokens,
                "weighted_tokens": weighted_tokens,
            })
            sample["_audit"] = composition_audit
        return sample
    except Exception as exc:
        return reject(f"formatting_or_tokenization_error:{type(exc).__name__}")


class _BatchEncodeRecorder:
    """Record ``encode`` calls made by the scalar sample formatter.

    Reusing :func:`process_text_sample` as the planner keeps the batched path
    tied to the exact scalar prompt formatting, special-token choices, and
    prompt/completion BPE boundaries. The placeholder ids are never returned to
    callers; they only let the formatter finish after all encode calls have
    been observed.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.requests = []
        self.batchable = True

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def encode(self, text, *args, **kwargs):
        # process_text_sample currently varies only add_special_tokens. If a
        # future formatter adds another encode option, take the safe scalar
        # fallback until the batch implementation explicitly supports it.
        if args or any(key != "add_special_tokens" for key in kwargs):
            self.batchable = False
        add_special_tokens = bool(kwargs.get("add_special_tokens", True))
        self.requests.append((text, add_special_tokens))
        return [0]


class _BatchEncodeReplay:
    """Replay pre-tokenized ids through the unchanged scalar formatter."""

    def __init__(self, tokenizer, requests, encoded_ids):
        self._tokenizer = tokenizer
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._requests = requests
        self._encoded_ids = encoded_ids
        self._position = 0
        self._failed = False

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def encode(self, text, *args, **kwargs):
        if self._position >= len(self._requests):
            self._failed = True
            raise RuntimeError("batched tokenizer replay received an unexpected encode call")
        expected_text, expected_special_tokens = self._requests[self._position]
        add_special_tokens = bool(kwargs.get("add_special_tokens", True))
        if (
            args
            or any(key != "add_special_tokens" for key in kwargs)
            or text != expected_text
            or add_special_tokens != expected_special_tokens
        ):
            self._failed = True
            raise RuntimeError("batched tokenizer replay diverged from scalar formatting")
        token_ids = self._encoded_ids[self._position]
        self._position += 1
        return list(token_ids)

    @property
    def complete(self):
        return not self._failed and self._position == len(self._requests)


def _is_supported_fast_tokenizer(tokenizer):
    """Limit batch encoding to Hugging Face's native fast-tokenizer family."""
    if not bool(getattr(tokenizer, "is_fast", False)) or not callable(tokenizer):
        return False
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        return False
    return (
        isinstance(tokenizer, PreTrainedTokenizerFast)
        and type(tokenizer).__module__.startswith("transformers.")
    )


def _normalize_batched_input_ids(encoded, expected_size: int):
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError):
        raise ValueError("batch tokenizer output is missing input_ids") from None
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()
    elif hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if not isinstance(input_ids, (list, tuple)) or len(input_ids) != expected_size:
        raise ValueError("batch tokenizer returned the wrong number of sequences")

    normalized = []
    for row in input_ids:
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().tolist()
        elif hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise ValueError("batch tokenizer returned malformed token ids")
        normalized.append([int(token_id) for token_id in row])
    return normalized


def process_text_samples_batch(tokenizer, text_dicts, max_length: int,
                               kayla_mode: bool = False,
                               text_column: Optional[str] = None,
                               prompt_column: Optional[str] = None,
                               completion_column: Optional[str] = None,
                               alpaca_mode: bool = False,
                               train_prompt_tokens: bool = True,
                               prompt_loss_weight: float = 1.0,
                               response_loss_weight: float = 1.0,
                               response_boundary_loss_weight: float = 1.0,
                               response_boundary_tokens: int = 0,
                               min_response_tokens: int = 1,
                               drop_empty_completions: bool = True,
                               include_audit: bool = False,
                               audit_source_default: Optional[str] = None):
    """Tokenize a map-style batch while preserving scalar sample semantics.

    Each prompt, completion, thought, or response remains an independent
    tokenizer sequence, so byte-pair merges can never cross a boundary that the
    scalar implementation keeps separate. Slow or custom tokenizers, malformed
    batch output, and any tokenizer exception fall back to the established
    per-sample path.
    """
    samples = list(text_dicts)
    if not samples:
        return []

    scalar_args = (
        max_length,
        kayla_mode,
        text_column,
        prompt_column,
        completion_column,
        alpaca_mode,
        train_prompt_tokens,
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
        min_response_tokens,
        drop_empty_completions,
        include_audit,
        audit_source_default,
    )

    def scalar_fallback():
        return [process_text_sample(tokenizer, sample, *scalar_args) for sample in samples]

    if not _is_supported_fast_tokenizer(tokenizer):
        return scalar_fallback()

    recorders = []
    flat_requests = []
    request_ranges = []
    for sample in samples:
        recorder = _BatchEncodeRecorder(tokenizer)
        process_text_sample(recorder, sample, *scalar_args)
        if not recorder.batchable:
            return scalar_fallback()
        request_start = len(flat_requests)
        flat_requests.extend(recorder.requests)
        request_ranges.append((request_start, len(flat_requests)))
        recorders.append(recorder)

    if not flat_requests:
        return scalar_fallback()

    flat_encoded_ids = [None] * len(flat_requests)
    try:
        for add_special_tokens in (True, False):
            positions = [
                idx
                for idx, (_text, request_special_tokens) in enumerate(flat_requests)
                if request_special_tokens == add_special_tokens
            ]
            if not positions:
                continue
            texts = [flat_requests[idx][0] for idx in positions]
            encoded = tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            encoded_rows = _normalize_batched_input_ids(encoded, len(positions))
            for request_idx, token_ids in zip(positions, encoded_rows):
                flat_encoded_ids[request_idx] = token_ids
        if any(token_ids is None for token_ids in flat_encoded_ids):
            raise ValueError("batch tokenizer did not satisfy every encode request")
    except Exception:
        return scalar_fallback()

    processed_samples = []
    for sample, recorder, (request_start, request_end) in zip(
        samples,
        recorders,
        request_ranges,
    ):
        if request_start == request_end:
            # Invalid rows normally land here. Running the scalar function once
            # gives future formatter extensions an exact fallback as well.
            processed_samples.append(process_text_sample(tokenizer, sample, *scalar_args))
            continue
        replay = _BatchEncodeReplay(
            tokenizer,
            recorder.requests,
            flat_encoded_ids[request_start:request_end],
        )
        processed = process_text_sample(replay, sample, *scalar_args)
        if not replay.complete:
            processed = process_text_sample(tokenizer, sample, *scalar_args)
        processed_samples.append(processed)
    return processed_samples

def process_tokenized_sample(text_dict: dict, max_length: Optional[int] = None):
    """Load a pre-tokenized JSONL row without running the tokenizer again."""
    if not isinstance(text_dict, dict) or "input_ids" not in text_dict:
        return None
    try:
        input_ids = text_dict.get("input_ids")
        labels = text_dict.get("labels", input_ids)
        attention_mask = text_dict.get("attention_mask")
        loss_weights = text_dict.get("loss_weights")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.detach().cpu().tolist()
        if isinstance(loss_weights, torch.Tensor):
            loss_weights = loss_weights.detach().cpu().tolist()

        input_ids = [int(value) for value in input_ids]
        labels = [int(value) for value in labels]
        if loss_weights is not None:
            loss_weights = [float(value) for value in loss_weights]
        if not input_ids:
            return None

        max_len = int(max_length or 0)
        if max_len > 0:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:max_len]
            if loss_weights is not None:
                loss_weights = loss_weights[:max_len]

        if len(labels) < len(input_ids):
            labels = labels + ([-100] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        else:
            attention_mask = [int(value) for value in attention_mask]
            if len(attention_mask) < len(input_ids):
                attention_mask = attention_mask + ([1] * (len(input_ids) - len(attention_mask)))
            elif len(attention_mask) > len(input_ids):
                attention_mask = attention_mask[:len(input_ids)]
        if loss_weights is not None:
            if len(loss_weights) < len(input_ids):
                loss_weights = loss_weights + ([1.0] * (len(input_ids) - len(loss_weights)))
            elif len(loss_weights) > len(input_ids):
                loss_weights = loss_weights[:len(input_ids)]

        length = text_dict.get("_length", text_dict.get("length", text_dict.get("valid_length")))
        try:
            length = int(length)
        except (TypeError, ValueError):
            length = _sample_effective_length(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                fallback_length=len(input_ids),
            )
        length = max(1, min(len(input_ids), length))

        sample = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "_length": length,
        }
        if loss_weights is not None:
            sample["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return sample
    except Exception:
        return None

class OriginalJSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, kayla_mode=False,
                 text_column: Optional[str] = None,
                 prompt_column: Optional[str] = None,
                 completion_column: Optional[str] = None,
                 alpaca_mode: bool = False,
                 train_prompt_tokens: bool = True,
                 prompt_loss_weight: float = 1.0,
                 response_loss_weight: float = 1.0,
                 response_boundary_loss_weight: float = 1.0,
                 response_boundary_tokens: int = 0,
                 min_response_tokens: int = 1,
                 drop_empty_completions: bool = True):
        super().__init__()
        self.tokenizer, self.max_length, self.kayla_mode, self.samples = tokenizer, max_length, kayla_mode, []
        self.sample_lengths = []
        skipped = 0
        line_num = 0

        def add_sample(data):
            processed = process_text_sample(
                tokenizer,
                data,
                max_length,
                kayla_mode,
                text_column=text_column,
                prompt_column=prompt_column,
                completion_column=completion_column,
                alpaca_mode=alpaca_mode,
                train_prompt_tokens=train_prompt_tokens,
                prompt_loss_weight=prompt_loss_weight,
                response_loss_weight=response_loss_weight,
                response_boundary_loss_weight=response_boundary_loss_weight,
                response_boundary_tokens=response_boundary_tokens,
                min_response_tokens=min_response_tokens,
                drop_empty_completions=drop_empty_completions,
            )
            if processed:
                self.samples.append(processed)
                self.sample_lengths.append(len(processed["input_ids"]))
                return True
            return False

        with open(path, "r", encoding="utf-8") as f:
            if not str(path).lower().endswith((".jsonl", ".ndjson")):
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
                    if not isinstance(data, list):
                        raise ValueError("JSON dataset must be an object, a list of objects, or JSONL.")
                    for obj in tqdm(data, desc="Loading JSON"):
                        if not add_sample(obj):
                            skipped += 1
                    if skipped:
                        print(f"WARNING: Skipped {skipped} invalid JSON samples.")
                    return
                except json.JSONDecodeError:
                    f.seek(0)
                except Exception:
                    f.seek(0)

            for line_num, line in enumerate(tqdm(f, desc="Loading JSONL"), 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    if skipped <= 5:
                        print(f"WARNING: Skipping malformed JSON at line {line_num} (char preview: {line[:80]}...)")
                    continue
                if not add_sample(data):
                    skipped += 1
        if skipped:
            print(f"WARNING: Skipped {skipped} malformed JSONL lines out of {line_num} total.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def get_sample_lengths(self): return self.sample_lengths

class StreamingJSONLDataset(IterableDataset):
    """
    Stream raw JSONL text samples from disk and tokenize inside DataLoader
    workers. This avoids materializing a tokenized tensor list before training.
    """
    def __init__(self, path, tokenizer, max_length, kayla_mode=False,
                 text_column: Optional[str] = None,
                 prompt_column: Optional[str] = None,
                 completion_column: Optional[str] = None,
                 bucket_size: int = 0, batch_size: int = 1,
                 shuffle_buckets: bool = True,
                 alpaca_mode: bool = False,
                 train_prompt_tokens: bool = True,
                 prompt_loss_weight: float = 1.0,
                 response_loss_weight: float = 1.0,
                 response_boundary_loss_weight: float = 1.0,
                 response_boundary_tokens: int = 0,
                 min_response_tokens: int = 1,
                 drop_empty_completions: bool = True,
                 precompute_rosa: bool = False,
                 rosa_vocab_size: Optional[int] = None,
                 rosa_chunk_size: int = 256,
                 rosa_max_context: int = 512,
                 include_audit: bool = False,
                 audit_source_default: Optional[str] = None,
                 enforce_rosa_max_context: bool = False):
        super().__init__()
        self.path = path
        self.paths = _resolve_jsonl_paths(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kayla_mode = kayla_mode
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        self.alpaca_mode = alpaca_mode
        self.train_prompt_tokens = bool(train_prompt_tokens)
        self.prompt_loss_weight = prompt_loss_weight
        self.response_loss_weight = response_loss_weight
        self.response_boundary_loss_weight = response_boundary_loss_weight
        self.response_boundary_tokens = response_boundary_tokens
        self.min_response_tokens = min_response_tokens
        self.drop_empty_completions = bool(drop_empty_completions)
        self.precompute_rosa = bool(precompute_rosa)
        self.rosa_vocab_size = int(rosa_vocab_size or 0)
        self.rosa_chunk_size = max(1, int(rosa_chunk_size or 256))
        self.rosa_max_context = max(1, int(rosa_max_context or 512))
        self.enforce_rosa_max_context = bool(enforce_rosa_max_context)
        self.include_audit = bool(include_audit)
        self.audit_source_default = audit_source_default or path
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle_buckets = bool(shuffle_buckets)
        self.seed = int(torch.initial_seed()) % (2**63 - 1)
        self.epoch = 0
        self._shared_epoch = _shared_epoch_counter()

    def set_epoch(self, epoch: int):
        _set_iterable_epoch(self, epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        bucket = []
        generator = torch.Generator()
        epoch = _get_iterable_epoch(self)
        generator.manual_seed((self.seed + epoch + worker_id) % (2**63 - 1))

        for _line_num, line in _iter_jsonl_shards_for_worker(self.paths, worker_id, num_workers):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if self.include_audit:
                    yield _make_audit_rejection(
                        {},
                        source_default=self.audit_source_default,
                        reason="malformed_json",
                    )
                continue
            processed = process_tokenized_sample(obj, self.max_length)
            if processed is None:
                processed = process_text_sample(
                    self.tokenizer,
                    obj,
                    self.max_length,
                    self.kayla_mode,
                    text_column=self.text_column,
                    prompt_column=self.prompt_column,
                    completion_column=self.completion_column,
                    alpaca_mode=self.alpaca_mode,
                    train_prompt_tokens=self.train_prompt_tokens,
                    prompt_loss_weight=self.prompt_loss_weight,
                    response_loss_weight=self.response_loss_weight,
                    response_boundary_loss_weight=self.response_boundary_loss_weight,
                    response_boundary_tokens=self.response_boundary_tokens,
                    min_response_tokens=self.min_response_tokens,
                    drop_empty_completions=self.drop_empty_completions,
                    include_audit=self.include_audit,
                    audit_source_default=self.audit_source_default,
                )
            elif self.include_audit:
                length = int(processed.get("_length", len(processed["input_ids"])))
                active_labels = processed["labels"][1:length].ne(-100)
                weights = processed.get("loss_weights")
                weighted_tokens = (
                    float(weights[1:length][active_labels].sum().item())
                    if weights is not None else float(active_labels.sum().item())
                )
                processed["_audit"] = {
                    "accepted": True,
                    "schema": "pretokenized",
                    "source": _audit_source_name(obj, self.audit_source_default),
                    "original_tokens": length,
                    "retained_tokens": length,
                    "original_response_tokens": 0,
                    "retained_response_tokens": 0,
                    "supervised_tokens": int(active_labels.sum().item()),
                    "weighted_tokens": weighted_tokens,
                    "truncated": False,
                }
            if processed is None:
                continue
            if processed.get("_audit_only", False):
                yield processed
                continue
            if self.precompute_rosa:
                processed = _attach_precomputed_rosa(
                    processed,
                    self.rosa_vocab_size,
                    self.rosa_chunk_size,
                    self.rosa_max_context,
                    self.enforce_rosa_max_context,
                )
            if self.bucket_size > 0:
                bucket.append(processed)
                if len(bucket) >= self.bucket_size:
                    yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)
                    bucket.clear()
            else:
                yield processed

        if bucket:
            yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)

def _maybe_shuffle_hf_dataset(hf_dataset, shuffle: bool, shuffle_buffer_size: int, seed: int):
    if not shuffle or shuffle_buffer_size <= 0:
        return hf_dataset
    shuffle_fn = getattr(hf_dataset, "shuffle", None)
    if not callable(shuffle_fn):
        return hf_dataset
    try:
        return shuffle_fn(buffer_size=int(shuffle_buffer_size), seed=int(seed))
    except TypeError:
        try:
            return shuffle_fn(seed=int(seed))
        except Exception:
            return hf_dataset
    except Exception:
        return hf_dataset

def _iter_hf_worker_samples(hf_dataset, worker_id: int, num_workers: int):
    dataset = hf_dataset
    sharded = False
    if num_workers > 1:
        shard_fn = getattr(dataset, "shard", None)
        if callable(shard_fn):
            try:
                dataset = shard_fn(num_shards=num_workers, index=worker_id, contiguous=False)
                sharded = True
            except TypeError:
                try:
                    dataset = shard_fn(num_shards=num_workers, index=worker_id)
                    sharded = True
                except Exception:
                    sharded = False
            except Exception:
                sharded = False

    iterator = iter(dataset)
    if num_workers > 1 and not sharded:
        iterator = itertools.islice(iterator, worker_id, None, num_workers)
    return iterator

class HuggingFaceStreamingDataset(IterableDataset):
    """
    Iterable wrapper for Hugging Face streaming datasets. It performs buffered
    stream shuffling, worker sharding, tokenization, and optional length buckets
    without building a full tokenized dataset in memory.
    """
    def __init__(self, hf_dataset, tokenizer, max_length, kayla_mode=False,
                 text_column=None, prompt_column=None, completion_column=None,
                 bucket_size: int = 0, batch_size: int = 1,
                 shuffle: bool = True, shuffle_buffer_size: int = 10000,
                 alpaca_mode: bool = False,
                 train_prompt_tokens: bool = True,
                 prompt_loss_weight: float = 1.0,
                 response_loss_weight: float = 1.0,
                 response_boundary_loss_weight: float = 1.0,
                 response_boundary_tokens: int = 0,
                 min_response_tokens: int = 1,
                 drop_empty_completions: bool = True):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kayla_mode = kayla_mode
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        self.alpaca_mode = alpaca_mode
        self.train_prompt_tokens = bool(train_prompt_tokens)
        self.prompt_loss_weight = prompt_loss_weight
        self.response_loss_weight = response_loss_weight
        self.response_boundary_loss_weight = response_boundary_loss_weight
        self.response_boundary_tokens = response_boundary_tokens
        self.min_response_tokens = min_response_tokens
        self.drop_empty_completions = bool(drop_empty_completions)
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size or 0))
        self.seed = int(torch.initial_seed()) % (2**32 - 1)
        self.epoch = 0
        self._shared_epoch = _shared_epoch_counter()

    def set_epoch(self, epoch: int):
        _set_iterable_epoch(self, epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = _get_iterable_epoch(self)
        dataset_seed = int((self.seed + (epoch * 100003)) % (2**32 - 1))
        bucket_seed = int((dataset_seed + worker_id) % (2**32 - 1))
        dataset = _maybe_shuffle_hf_dataset(
            self.hf_dataset,
            shuffle=self.shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            # Every worker must shuffle the same logical stream before it is
            # sharded. Per-worker shuffle seeds can duplicate/omit samples.
            seed=dataset_seed,
        )
        iterator = _iter_hf_worker_samples(dataset, worker_id, num_workers)

        bucket = []
        generator = torch.Generator()
        generator.manual_seed(bucket_seed)

        for sample in iterator:
            processed = process_text_sample(
                self.tokenizer,
                sample,
                self.max_length,
                self.kayla_mode,
                self.text_column,
                self.prompt_column,
                self.completion_column,
                self.alpaca_mode,
                self.train_prompt_tokens,
                self.prompt_loss_weight,
                self.response_loss_weight,
                self.response_boundary_loss_weight,
                self.response_boundary_tokens,
                self.min_response_tokens,
                self.drop_empty_completions,
            )
            if processed is None:
                continue
            if self.bucket_size > 0:
                bucket.append(processed)
                if len(bucket) >= self.bucket_size:
                    yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle, generator)
                    bucket.clear()
            else:
                yield processed

        if bucket:
            yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle, generator)

class HuggingFaceMapStyleDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length, kayla_mode=False, text_column=None, prompt_column=None, completion_column=None, alpaca_mode: bool = False, train_prompt_tokens: bool = True, prompt_loss_weight: float = 1.0, response_loss_weight: float = 1.0, response_boundary_loss_weight: float = 1.0, response_boundary_tokens: int = 0, min_response_tokens: int = 1, drop_empty_completions: bool = True, precompute_rosa: bool = False, rosa_vocab_size: Optional[int] = None, rosa_chunk_size: int = 256, rosa_max_context: int = 512, include_audit: bool = False, audit_source_default: Optional[str] = None, enforce_rosa_max_context: bool = False):
        super().__init__()
        self.hf_dataset, self.tokenizer, self.max_length, self.kayla_mode = hf_dataset, tokenizer, max_length, kayla_mode
        self.text_column, self.prompt_column, self.completion_column = text_column, prompt_column, completion_column
        self.alpaca_mode = alpaca_mode
        self.train_prompt_tokens = bool(train_prompt_tokens)
        self.prompt_loss_weight = prompt_loss_weight
        self.response_loss_weight = response_loss_weight
        self.response_boundary_loss_weight = response_boundary_loss_weight
        self.response_boundary_tokens = response_boundary_tokens
        self.min_response_tokens = min_response_tokens
        self.drop_empty_completions = bool(drop_empty_completions)
        self.precompute_rosa = bool(precompute_rosa)
        self.rosa_vocab_size = int(rosa_vocab_size or 0)
        self.rosa_chunk_size = max(1, int(rosa_chunk_size or 256))
        self.rosa_max_context = max(1, int(rosa_max_context or 512))
        self.enforce_rosa_max_context = bool(enforce_rosa_max_context)
        self.include_audit = bool(include_audit)
        self.audit_source_default = audit_source_default
        self.sample_lengths = None
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        try:
            sample = process_text_sample(self.tokenizer, self.hf_dataset[idx], self.max_length, self.kayla_mode, self.text_column, self.prompt_column, self.completion_column, self.alpaca_mode, self.train_prompt_tokens, self.prompt_loss_weight, self.response_loss_weight, self.response_boundary_loss_weight, self.response_boundary_tokens, self.min_response_tokens, self.drop_empty_completions, self.include_audit, self.audit_source_default)
            if self.precompute_rosa and sample is not None and not sample.get("_audit_only", False):
                sample = _attach_precomputed_rosa(
                    sample,
                    self.rosa_vocab_size,
                    self.rosa_chunk_size,
                    self.rosa_max_context,
                    self.enforce_rosa_max_context,
                )
            return sample
        except Exception:
            # The dynamic collator filters malformed samples. Returning None
            # keeps worker pools alive instead of aborting a multi-hour build.
            return None
    def __getitems__(self, indices):
        """Fetch Arrow rows and fast-tokenize the whole DataLoader batch."""
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        else:
            indices = list(indices)
        if not indices:
            return []

        try:
            # Hugging Face Dataset supports list indexing and returns a
            # column-oriented dict. One Arrow lookup is substantially cheaper
            # than one Python lookup per row during a multi-billion-token build.
            fetched = self.hf_dataset[indices]
            if isinstance(fetched, dict):
                for values in fetched.values():
                    if (
                        isinstance(values, (str, bytes))
                        or not hasattr(values, "__len__")
                        or len(values) != len(indices)
                    ):
                        raise TypeError("dataset batch column has an unsupported shape")
                rows = []
                for position in range(len(indices)):
                    rows.append({key: values[position] for key, values in fetched.items()})
            elif isinstance(fetched, (list, tuple)) and len(fetched) == len(indices):
                rows = list(fetched)
            else:
                raise TypeError("dataset batch indexing returned an unsupported shape")
        except Exception:
            try:
                rows = [self.hf_dataset[idx] for idx in indices]
            except Exception:
                return [self[idx] for idx in indices]

        try:
            processed_rows = process_text_samples_batch(
                self.tokenizer,
                rows,
                self.max_length,
                self.kayla_mode,
                self.text_column,
                self.prompt_column,
                self.completion_column,
                self.alpaca_mode,
                self.train_prompt_tokens,
                self.prompt_loss_weight,
                self.response_loss_weight,
                self.response_boundary_loss_weight,
                self.response_boundary_tokens,
                self.min_response_tokens,
                self.drop_empty_completions,
                self.include_audit,
                self.audit_source_default,
            )
            if self.precompute_rosa:
                processed_rows = [
                    _attach_precomputed_rosa(
                        sample,
                        self.rosa_vocab_size,
                        self.rosa_chunk_size,
                        self.rosa_max_context,
                        self.enforce_rosa_max_context,
                    )
                    if sample is not None and not sample.get("_audit_only", False)
                    else sample
                    for sample in processed_rows
                ]
            return processed_rows
        except Exception:
            # Keep the cache build resilient to third-party Dataset wrappers.
            return [self[idx] for idx in indices]
    def get_sample_lengths(self):
        if self.sample_lengths is not None:
            return self.sample_lengths
        lengths = []
        for idx in range(len(self.hf_dataset)):
            try:
                sample = self.hf_dataset[idx]
                length = _estimate_text_token_length(
                    sample, self.max_length, self.kayla_mode,
                    self.text_column, self.prompt_column, self.completion_column,
                    self.alpaca_mode,
                )
                if length is None:
                    processed = process_text_sample(
                        self.tokenizer,
                        sample,
                        self.max_length,
                        self.kayla_mode,
                        self.text_column,
                        self.prompt_column,
                        self.completion_column,
                        self.alpaca_mode,
                        self.train_prompt_tokens,
                        self.prompt_loss_weight,
                        self.response_loss_weight,
                        self.response_boundary_loss_weight,
                        self.response_boundary_tokens,
                        self.min_response_tokens,
                        self.drop_empty_completions,
                        self.include_audit,
                        self.audit_source_default,
                    )
                    length = len(processed["input_ids"]) if processed is not None else None
            except Exception:
                return None
            if length is None:
                length = self.max_length
            lengths.append(length)
        self.sample_lengths = lengths
        return self.sample_lengths

def _collate_training_batch(batch, pad_token_id):
    audit_records = [
        dict(item["_audit"])
        for item in batch
        if isinstance(item, dict) and isinstance(item.get("_audit"), dict)
    ]
    batch = [
        item for item in batch
        if isinstance(item, dict)
        and not item.get("_audit_only", False)
        and "input_ids" in item
        and "labels" in item
    ]
    if not batch:
        return {"_audit_records": audit_records} if audit_records else None
    ml = max(_sample_effective_length(i) for i in batch)
    ids = torch.full((len(batch), ml), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), ml), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), ml), dtype=torch.long)
    has_loss_weights = any("loss_weights" in item for item in batch)
    loss_weights = torch.zeros((len(batch), ml), dtype=torch.float32) if has_loss_weights else None
    has_rosa_ids = any("rosa_ids" in item for item in batch)
    rosa_ids = None
    if has_rosa_ids:
        sentinel = next((int(item.get("_rosa_sentinel")) for item in batch if "_rosa_sentinel" in item), None)
        if sentinel is None:
            raise ValueError("Cached ROSA samples must include _rosa_sentinel")
        rosa_context_modes = {
            str(
                item.get(
                    "_rosa_context_mode",
                    ROSA_UNBOUNDED_CONTEXT_MODE,
                )
            )
            for item in batch
            if "rosa_ids" in item
        }
        if len(rosa_context_modes) != 1:
            raise ValueError(
                "Cannot mix cached ROSA context modes in one batch"
            )
        rosa_context_mode = next(iter(rosa_context_modes))
        rosa_ids = torch.full((len(batch), ml), sentinel, dtype=torch.long)
    for i, item in enumerate(batch):
        item_ids = _as_long_tensor(item["input_ids"])
        item_labels = _as_long_tensor(item["labels"])
        item_mask = item.get("attention_mask")
        item_mask = _as_long_tensor(item_mask) if item_mask is not None else None
        item_weights = item.get("loss_weights")
        item_weights = _as_float_tensor(item_weights) if item_weights is not None else None

        sl = min(ml, item_ids.numel(), item_labels.numel())
        if item_mask is not None:
            sl = min(sl, item_mask.numel())
        if item_weights is not None:
            sl = min(sl, item_weights.numel())
        ids[i, :sl] = item_ids[:sl]
        labels[i, :sl] = item_labels[:sl]
        mask[i, :sl] = item_mask[:sl] if item_mask is not None else 1
        if loss_weights is not None:
            loss_weights[i, :sl] = item_weights[:sl] if item_weights is not None else 1.0
        if rosa_ids is not None:
            if "rosa_ids" not in item:
                raise ValueError("Cannot mix cached-ROSA and live-ROSA samples in one batch")
            item_rosa = _as_long_tensor(item["rosa_ids"])
            rosa_ids[i, :min(sl, item_rosa.numel())] = item_rosa[:min(sl, item_rosa.numel())]
    batch_out = {"input_ids": ids, "labels": labels, "attention_mask": mask}
    if loss_weights is not None:
        batch_out["loss_weights"] = loss_weights
    if rosa_ids is not None:
        batch_out["rosa_ids"] = rosa_ids
        batch_out["rosa_ids_context_mode"] = rosa_context_mode
    if audit_records:
        batch_out["_audit_records"] = audit_records
    return batch_out

def _collate_fn_dynamic_padding(batch, pad_token_id):
    return _collate_training_batch(batch, pad_token_id)


def _collate_tokenized_binary_batch(batch, pad_token_id):
    # Modern PyTorch forwards a batch sampler's complete index list to
    # Dataset.__getitems__. TokenizedBinaryDataset returns the already-collated
    # dictionary, so the hot path is an identity operation. Keep the generic
    # fallback for older PyTorch versions that only call __getitem__.
    if isinstance(batch, dict):
        return batch
    return _collate_training_batch(batch, pad_token_id)

def create_dataloader_for_jsonl(path, tokenizer, max_length, batch_size, pad_token_id,
                                num_workers=0, kayla_mode=False,
                                text_column: Optional[str] = None,
                                prompt_column: Optional[str] = None,
                                completion_column: Optional[str] = None,
                                use_length_bucketing=True, bucket_size=None,
                                device=None, prefetch_factor=None,
                                alpaca_mode: bool = False,
                                train_prompt_tokens: bool = True,
                                prompt_loss_weight: float = 1.0,
                                response_loss_weight: float = 1.0,
                                response_boundary_loss_weight: float = 1.0,
                                response_boundary_tokens: int = 0,
                                min_response_tokens: int = 1,
                                drop_empty_completions: bool = True,
                                precompute_rosa: bool = False,
                                rosa_vocab_size: Optional[int] = None,
                                rosa_chunk_size: int = 256,
                                rosa_max_context: int = 512,
                                in_order: bool = True,
                                include_audit: bool = False,
                                audit_source_default: Optional[str] = None,
                                enforce_rosa_max_context: bool = False):
    dataset = StreamingJSONLDataset(
        path,
        tokenizer,
        max_length,
        kayla_mode=kayla_mode,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
        train_prompt_tokens=train_prompt_tokens,
        prompt_loss_weight=prompt_loss_weight,
        response_loss_weight=response_loss_weight,
        response_boundary_loss_weight=response_boundary_loss_weight,
        response_boundary_tokens=response_boundary_tokens,
        min_response_tokens=min_response_tokens,
        drop_empty_completions=drop_empty_completions,
        precompute_rosa=precompute_rosa,
        rosa_vocab_size=rosa_vocab_size,
        rosa_chunk_size=rosa_chunk_size,
        rosa_max_context=rosa_max_context,
        enforce_rosa_max_context=enforce_rosa_max_context,
        include_audit=include_audit,
        audit_source_default=audit_source_default,
        bucket_size=_streaming_bucket_size(batch_size, use_length_bucketing, bucket_size),
        batch_size=batch_size,
        shuffle_buckets=True,
    )
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    pin_memory = _pin_memory_for_device(device)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        in_order=in_order,
    )

def create_dataloader_for_hf_streaming(hf_dataset, tokenizer, max_length, batch_size, pad_token_id,
                                       num_workers=0, kayla_mode=False,
                                       text_column=None, prompt_column=None, completion_column=None,
                                       use_length_bucketing=True, bucket_size=None,
                                       shuffle=True, shuffle_buffer_size=10000,
                                       device=None, prefetch_factor=None,
                                       alpaca_mode: bool = False,
                                       train_prompt_tokens: bool = True,
                                       prompt_loss_weight: float = 1.0,
                                       response_loss_weight: float = 1.0,
                                       response_boundary_loss_weight: float = 1.0,
                                       response_boundary_tokens: int = 0,
                                       min_response_tokens: int = 1,
                                       drop_empty_completions: bool = True):
    dataset = HuggingFaceStreamingDataset(
        hf_dataset,
        tokenizer,
        max_length,
        kayla_mode=kayla_mode,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
        train_prompt_tokens=train_prompt_tokens,
        prompt_loss_weight=prompt_loss_weight,
        response_loss_weight=response_loss_weight,
        response_boundary_loss_weight=response_boundary_loss_weight,
        response_boundary_tokens=response_boundary_tokens,
        min_response_tokens=min_response_tokens,
        drop_empty_completions=drop_empty_completions,
        bucket_size=_streaming_bucket_size(batch_size, use_length_bucketing, bucket_size),
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    pin_memory = _pin_memory_for_device(device)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

def create_map_style_dataloader(dataset, batch_size, pad_token_id, num_workers=0, shuffle=True,
                                use_length_bucketing=True, bucket_size=None,
                                device=None, prefetch_factor=None, in_order=True):
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    use_cuda = _pin_memory_for_device(device)
    drop_last = False
    lengths = _get_dataset_sample_lengths(dataset) if use_length_bucketing and shuffle and batch_size > 1 else None
    if lengths is not None and len(lengths) > 0:
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_size=bucket_size,
        )
        return _create_dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=prefetch_factor,
            in_order=in_order,
        )

    sampler = EpochShuffleSampler(dataset, shuffle=shuffle)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        in_order=in_order,
    )
