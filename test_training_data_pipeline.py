import json
import os
import functools

import pytest
import torch

from hierarchos.training.datasets import (
    LengthGroupedBatchSampler,
    PTChunkedDataset,
    TokenizedBinaryDataset,
    _collate_training_batch,
    _normalize_sample_lengths,
    create_dataloader_for_tokenized_cache,
)
import hierarchos.training.datasets as datasets_module


def _write_binary_cache(directory, records, *, with_loss_weights=False, with_rosa=False):
    os.makedirs(directory, exist_ok=True)
    offsets = []
    lengths = []
    total_bytes = 0
    with open(os.path.join(directory, "tokens.bin"), "wb") as data_file:
        for record in records:
            input_ids = torch.tensor(record["input_ids"], dtype=torch.int32).contiguous()
            labels = torch.tensor(record["labels"], dtype=torch.int32).contiguous()
            assert input_ids.numel() == labels.numel()
            length = int(input_ids.numel())
            offsets.append(total_bytes)
            lengths.append(length)

            fields = [input_ids.numpy().tobytes(), labels.numpy().tobytes()]
            if with_loss_weights:
                weights = torch.tensor(record["loss_weights"], dtype=torch.float16).contiguous()
                assert weights.numel() == length
                fields.append(weights.numpy().tobytes())
            if with_rosa:
                rosa_ids = torch.tensor(record["rosa_ids"], dtype=torch.int32).contiguous()
                assert rosa_ids.numel() == length
                fields.append(rosa_ids.numpy().tobytes())

            for field in fields:
                data_file.write(field)
                total_bytes += len(field)

    torch.save(
        {
            "format": "test-token-cache",
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.int32),
            "has_loss_weights": with_loss_weights,
            "loss_weight_dtype": "float16" if with_loss_weights else None,
            "has_rosa_ids": with_rosa,
            "rosa_sentinel": 4096 if with_rosa else 0,
        },
        os.path.join(directory, "index.pt"),
    )
    with open(os.path.join(directory, "_SUCCESS"), "w", encoding="utf-8") as success_file:
        json.dump({"samples": len(records), "bytes": total_bytes}, success_file)


def _write_pt_chunk_dataset(directory, payload, *, file_path="chunk.pt"):
    os.makedirs(directory, exist_ok=True)
    torch.save(payload, os.path.join(directory, file_path))
    with open(
        os.path.join(directory, "manifest.jsonl"),
        "w",
        encoding="utf-8",
    ) as manifest:
        manifest.write(json.dumps({
            "file_path": file_path,
            "index_in_file": 0,
            "length": 3,
        }) + "\n")


def _legacy_length_grouped_batches(
    lengths,
    batch_size,
    *,
    shuffle,
    drop_last,
    bucket_size,
    seed,
    epoch,
    preserve_order,
):
    lengths = [max(1, int(length)) for length in lengths]
    bucket_size = max(batch_size, int(bucket_size or (batch_size * 50)))
    bucket_size = max(batch_size, (bucket_size // batch_size) * batch_size)
    generator = torch.Generator()
    generator.manual_seed((int(seed) + int(epoch)) % (2**63 - 1))

    if shuffle and not preserve_order and len(lengths) > 1:
        indices = torch.randperm(len(lengths), generator=generator).tolist()
    else:
        indices = list(range(len(lengths)))

    batches = []
    for bucket_start in range(0, len(indices), bucket_size):
        bucket = indices[bucket_start:bucket_start + bucket_size]
        if shuffle:
            bucket.sort(key=lengths.__getitem__, reverse=True)
        bucket_batches = []
        for batch_start in range(0, len(bucket), batch_size):
            batch = bucket[batch_start:batch_start + batch_size]
            if len(batch) == batch_size or not drop_last:
                bucket_batches.append(batch)
        if shuffle and preserve_order and len(bucket_batches) > 1:
            order = torch.randperm(len(bucket_batches), generator=generator).tolist()
            batches.extend(bucket_batches[batch_idx] for batch_idx in order)
        else:
            batches.extend(bucket_batches)

    if shuffle and not preserve_order and len(batches) > 1:
        order = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[batch_idx] for batch_idx in order]
    return batches


@pytest.mark.parametrize(
    "shuffle,drop_last,preserve_order",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_tensor_sampler_matches_legacy_order_and_coverage(
    shuffle,
    drop_last,
    preserve_order,
):
    lengths = torch.tensor(
        [9, 1, 7, 7, 3, 12, 4, 4, 8, 2, 11, 6, 5, 5, 10, 2, 9, 3, 8, 1, 6, 12, 7],
        dtype=torch.int32,
    )
    kwargs = {
        "batch_size": 4,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "bucket_size": 8,
        "seed": 7719,
        "preserve_order": preserve_order,
    }
    sampler = LengthGroupedBatchSampler(lengths, **kwargs)
    sampler.set_epoch(3)

    actual = list(sampler)
    expected = _legacy_length_grouped_batches(
        lengths.tolist(),
        epoch=3,
        **kwargs,
    )
    assert actual == expected

    flattened = [idx for batch in actual for idx in batch]
    expected_count = (len(lengths) // 4) * 4 if drop_last else len(lengths)
    assert len(flattened) == expected_count
    assert len(flattened) == len(set(flattened))

    repeated = list(sampler)
    assert repeated == actual
    if shuffle:
        sampler.set_epoch(4)
        assert list(sampler) != actual


def test_length_normalization_preserves_compact_tensor_storage():
    class _SizedDataset:
        def __len__(self):
            return 5

    source = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    normalized = _normalize_sample_lengths(_SizedDataset(), source)
    assert isinstance(normalized, torch.Tensor)
    assert normalized.dtype == torch.int32
    assert normalized.data_ptr() == source.data_ptr()


def test_fused_binary_batch_matches_reference_collation(tmp_path):
    records = [
        {
            "input_ids": [10, 11, 12, 13, 14, 15, 16],
            "labels": [-100, -100, 12, 13, 14, 15, 16],
            "loss_weights": [0.0, 0.0, 1.0, 1.5, 2.0, 0.5, 1.0],
            "rosa_ids": [4096, 4096, 12, 13, 14, 15, 16],
        },
        {
            "input_ids": [20, 21, 22],
            "labels": [20, 21, 22],
            "loss_weights": [1.0, 0.25, 2.0],
            "rosa_ids": [4096, 20, 21],
        },
        {
            "input_ids": [30, 31, 32, 33, 34],
            "labels": [-100, 31, 32, 33, 34],
            "loss_weights": [0.0, 1.0, 1.0, 1.0, 3.0],
            "rosa_ids": [4096, 30, 31, 32, 33],
        },
    ]
    _write_binary_cache(
        tmp_path,
        records,
        with_loss_weights=True,
        with_rosa=True,
    )

    dataset = TokenizedBinaryDataset(tmp_path, max_length=5, pad_token_id=77)
    indices = [2, 0, 1]
    reference = _collate_training_batch([dataset[idx] for idx in indices], pad_token_id=77)
    fused = dataset.__getitems__(indices)

    assert fused.keys() == reference.keys()
    for key in reference:
        if torch.is_tensor(reference[key]):
            assert fused[key].dtype == reference[key].dtype
            assert torch.equal(fused[key], reference[key]), key
        else:
            assert fused[key] == reference[key], key
    assert torch.equal(dataset.sample_lengths, torch.tensor([5, 3, 5], dtype=torch.int32))
    dataset.close()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_token_cache_dataloader_uses_fused_batches_with_exact_outputs(tmp_path, num_workers):
    records = [
        {"input_ids": list(range(1, length + 1)), "labels": list(range(1, length + 1))}
        for length in (2, 7, 4, 6, 3, 5)
    ]
    _write_binary_cache(tmp_path, records)
    loader = create_dataloader_for_tokenized_cache(
        tmp_path,
        max_length=5,
        batch_size=3,
        pad_token_id=99,
        num_workers=num_workers,
        use_length_bucketing=True,
        bucket_size=6,
    )
    loader.batch_sampler.seed = 1234
    loader.batch_sampler.set_epoch(2)
    expected_indices = list(loader.batch_sampler)
    actual_batches = list(loader)

    assert len(actual_batches) == len(expected_indices)
    for batch, indices in zip(actual_batches, expected_indices):
        reference = _collate_training_batch(
            [loader.dataset[idx] for idx in indices],
            pad_token_id=99,
        )
        assert batch.keys() == reference.keys()
        for key in reference:
            assert torch.equal(batch[key], reference[key]), key
    loader.dataset.close()


def test_vectorized_binary_index_validation_rejects_bad_offsets(tmp_path):
    records = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    _write_binary_cache(tmp_path, records)
    index_path = os.path.join(tmp_path, "index.pt")
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    index["offsets"][1] += 4
    torch.save(index, index_path)

    with pytest.raises(ValueError, match="offsets do not match"):
        TokenizedBinaryDataset(tmp_path)


def test_pt_chunk_loader_uses_weights_only_and_validates_samples(
    tmp_path,
    monkeypatch,
):
    sample = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_mask": torch.ones(3, dtype=torch.long),
    }
    _write_pt_chunk_dataset(tmp_path, [sample])
    real_load = torch.load
    load_kwargs = []

    @functools.wraps(real_load)
    def tracking_load(*args, **kwargs):
        load_kwargs.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(datasets_module.torch, "load", tracking_load)
    dataset = PTChunkedDataset(tmp_path, max_length=8)
    loaded = dataset[0]

    assert torch.equal(loaded["input_ids"], sample["input_ids"])
    assert load_kwargs
    assert load_kwargs[0].get("weights_only") is True

    invalid_root = tmp_path / "invalid"
    _write_pt_chunk_dataset(invalid_root, ["not-a-sample"])
    invalid = PTChunkedDataset(invalid_root, max_length=8)
    with pytest.raises(ValueError, match="must be a mapping"):
        invalid[0]


def test_pt_manifest_cannot_escape_dataset_directory(tmp_path):
    outside = tmp_path / "outside.pt"
    torch.save([], outside)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with open(
        dataset_dir / "manifest.jsonl",
        "w",
        encoding="utf-8",
    ) as manifest:
        manifest.write(json.dumps({
            "file_path": "../outside.pt",
            "index_in_file": 0,
            "length": 1,
        }) + "\n")

    with pytest.raises(ValueError, match="escapes"):
        PTChunkedDataset(dataset_dir, max_length=8)
