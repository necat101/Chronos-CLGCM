import json
import os
import shutil
import types
import tempfile

import pytest
import torch

from hierarchos.training.datasets import (
    HuggingFaceStreamingDataset,
    LengthGroupedBatchSampler,
    OriginalJSONLDataset,
    StreamingJSONLDataset,
    TokenizedBinaryDataset,
    _assign_jsonl_shards,
    _iter_hf_worker_samples,
    _iter_jsonl_lines_for_worker,
    _iter_jsonl_shards_for_worker,
    create_dataloader_for_chunked,
    create_dataloader_for_hf_streaming,
    create_dataloader_for_jsonl,
    create_dataloader_for_tokenized_cache,
    process_text_sample,
)
from hierarchos.inference.chat import wrap_for_hierarchos
from hierarchos.training.trainer import pad_training_batch_to_multiple, set_dataloader_epoch


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        text = str(text)
        if not text:
            return []
        return [3 + (ord(ch) % 97) for ch in text]


class RecordingTokenizer(TinyTokenizer):
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((str(text), add_special_tokens))
        return super().encode(text, add_special_tokens=add_special_tokens)


class FakeHFStream:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, buffer_size=10000, seed=0):
        del buffer_size
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if len(self.rows) <= 1:
            return FakeHFStream(self.rows)
        order = torch.randperm(len(self.rows), generator=generator).tolist()
        return FakeHFStream([self.rows[idx] for idx in order])

    def shard(self, num_shards, index, contiguous=False):
        del contiguous
        return FakeHFStream(self.rows[index::num_shards])


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _padding_waste(batches):
    padded_tokens = sum(int(batch["input_ids"].numel()) for batch in batches)
    real_tokens = sum(int(batch["attention_mask"].sum().item()) for batch in batches)
    return padded_tokens - real_tokens


def _batch_length_spreads(batches):
    spreads = []
    for batch in batches:
        lengths = batch["attention_mask"].sum(dim=1)
        spreads.append(int(lengths.max().item() - lengths.min().item()))
    return spreads


def _compile_padded_batches(batches, multiple):
    padded = []
    for batch in batches:
        ids, labels, mask = pad_training_batch_to_multiple(
            batch["input_ids"],
            batch["labels"],
            batch.get("attention_mask"),
            multiple=multiple,
            pad_token_id=0,
        )
        assert ids.shape[1] % multiple == 0
        padded.append({"input_ids": ids, "labels": labels, "attention_mask": mask})
    return padded


def _alternating_length_rows():
    lengths = [8, 180, 12, 170, 16, 160, 20, 150, 24, 140, 28, 130, 32, 120, 36, 110]
    return [{"text": "x" * length} for length in lengths]


def _write_binary_token_cache(directory, rows, tokenizer, max_length=256):
    os.makedirs(directory, exist_ok=True)
    offsets = []
    lengths = []
    total_bytes = 0
    with open(os.path.join(directory, "tokens.bin"), "wb") as f:
        for row in rows:
            processed = process_text_sample(tokenizer, row, max_length)
            assert processed is not None
            input_ids = processed["input_ids"].to(dtype=torch.int32).contiguous()
            labels = processed["labels"].to(dtype=torch.int32).contiguous()
            length = min(int(input_ids.numel()), int(labels.numel()))
            offsets.append(total_bytes)
            lengths.append(length)
            input_bytes = input_ids[:length].numpy().tobytes()
            label_bytes = labels[:length].numpy().tobytes()
            f.write(input_bytes)
            f.write(label_bytes)
            total_bytes += len(input_bytes) + len(label_bytes)
    torch.save({
        "format": "map-token-bin-v1",
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
    }, os.path.join(directory, "index.pt"))
    with open(os.path.join(directory, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({"samples": len(lengths)}, f)


def test_binary_token_cache_rejects_truncated_data():
    tokenizer = TinyTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_binary_token_cache(tmpdir, [{"text": "cache integrity"}], tokenizer)
        data_path = os.path.join(tmpdir, "tokens.bin")
        with open(data_path, "r+b") as data_file:
            data_file.truncate(os.path.getsize(data_path) - 1)

        with pytest.raises(ValueError, match="data size mismatch"):
            TokenizedBinaryDataset(tmpdir)


def test_process_text_sample_autodetects_common_columns():
    tokenizer = TinyTokenizer()
    text_sample = process_text_sample(tokenizer, {"text": "plain text"}, max_length=128)
    instruct_sample = process_text_sample(
        tokenizer,
        {"instruction": "say hi", "output": "hi"},
        max_length=128,
    )

    assert text_sample is not None
    assert instruct_sample is not None
    assert text_sample["input_ids"].dtype == torch.long
    assert instruct_sample["labels"].shape == instruct_sample["input_ids"].shape
    assert not (instruct_sample["labels"] == -100).any()


def test_alpaca_instruction_input_output_uses_previous_context_section():
    tokenizer = RecordingTokenizer()
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Summarize the passage.",
            "input": "The passage to summarize.",
            "output": "A short summary.",
        },
        max_length=256,
        prompt_column="instruction",
        completion_column="output",
    )

    assert processed is not None
    assert tokenizer.calls[0] == (
        "### Previous Context:\nThe passage to summarize.\n\n"
        "### Instruction:\nSummarize the passage.\n\n"
        "### Response:\n",
        True,
    )
    assert tokenizer.calls[1] == ("A short summary.", False)
    assert "User:" not in tokenizer.calls[0][0]
    prompt_ids = torch.tensor(TinyTokenizer().encode(tokenizer.calls[0][0]), dtype=torch.long)
    assert torch.equal(
        processed["labels"][: len(prompt_ids)],
        prompt_ids,
    )


def test_alpaca_prompt_tokens_can_be_masked_for_legacy_sft():
    tokenizer = RecordingTokenizer()
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Explain the protocol.",
            "input": "A long prompt prefix.",
            "output": "Done.",
        },
        max_length=256,
        prompt_column="instruction",
        completion_column="output",
        train_prompt_tokens=False,
    )

    assert processed is not None
    prompt_len = len(TinyTokenizer().encode(tokenizer.calls[0][0]))
    assert torch.equal(
        processed["labels"][:prompt_len],
        torch.full((prompt_len,), -100, dtype=torch.long),
    )


def test_alpaca_flag_defaults_instruction_output_columns():
    tokenizer = RecordingTokenizer()
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Classify sentiment.",
            "input": "I love this.",
            "output": "positive",
        },
        max_length=256,
        alpaca_mode=True,
    )

    assert processed is not None
    assert "### Previous Context:\nI love this.\n\n" in tokenizer.calls[0][0]
    assert tokenizer.calls[1] == ("positive", False)


def test_alpaca_empty_input_omits_previous_context_section():
    tokenizer = RecordingTokenizer()
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Answer directly.",
            "input": "   ",
            "output": "Direct answer.",
        },
        max_length=256,
        alpaca_mode=True,
    )

    assert processed is not None
    assert tokenizer.calls[0] == (
        "### Instruction:\nAnswer directly.\n\n"
        "### Response:\n",
        True,
    )
    assert "### Previous Context:" not in tokenizer.calls[0][0]
    assert "User:" not in tokenizer.calls[0][0]


def test_alpaca_chat_wrapper_uses_context_section_not_user_prompt():
    prompt = wrap_for_hierarchos(
        "Answer the latest request.",
        alpaca_mode=True,
        input_context="Earlier context.",
    )

    assert prompt == (
        "### Previous Context:\nEarlier context.\n\n"
        "### Instruction:\nAnswer the latest request.\n\n"
        "### Response:\n"
    )
    assert "User:" not in prompt
    assert prompt.index("### Previous Context:") < prompt.index("### Instruction:")


def test_alpaca_chat_wrapper_empty_context_is_zero_shot():
    prompt = wrap_for_hierarchos(
        "Answer without prior context.",
        alpaca_mode=True,
        input_context="",
    )

    assert prompt == (
        "### Instruction:\nAnswer without prior context.\n\n"
        "### Response:\n"
    )
    assert "### Previous Context:" not in prompt
    assert "User:" not in prompt


def test_prompt_completion_drops_empty_outputs_by_default():
    tokenizer = RecordingTokenizer()

    assert process_text_sample(
        tokenizer,
        {"instruction": "Say something.", "output": ""},
        max_length=128,
        alpaca_mode=True,
    ) is None

    kept = process_text_sample(
        tokenizer,
        {"instruction": "Say something.", "output": ""},
        max_length=128,
        alpaca_mode=True,
        drop_empty_completions=False,
    )
    assert kept is not None
    assert kept["input_ids"][-1].item() == TinyTokenizer.eos_token_id


def test_prompt_completion_truncation_preserves_response_tokens():
    tokenizer = RecordingTokenizer()
    output = "assistant answer"
    output_ids = TinyTokenizer().encode(output, add_special_tokens=False)
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Use the context.",
            "input": "context " * 200,
            "output": output,
        },
        max_length=len(output_ids) + 12,
        alpaca_mode=True,
        min_response_tokens=len(output_ids),
    )

    assert processed is not None
    assert len(processed["input_ids"]) == len(output_ids) + 12
    assert processed["input_ids"][-(len(output_ids) + 1):].tolist() == (
        output_ids + [TinyTokenizer.eos_token_id]
    )
    assert processed["labels"][-(len(output_ids) + 1):].tolist() == (
        output_ids + [TinyTokenizer.eos_token_id]
    )


def test_streaming_jsonl_dataloader_batches_without_materializing():
    tokenizer = TinyTokenizer()
    rows = [{"text": f"sample {idx}"} for idx in range(11)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        dataloader = create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=64,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        )

        assert isinstance(dataloader.dataset, StreamingJSONLDataset)
        assert not hasattr(dataloader.dataset, "samples")

        batches = list(dataloader)
        assert len(batches) == 3
        assert sum(batch["input_ids"].shape[0] for batch in batches) == len(rows)
        assert all(batch["input_ids"].dtype == torch.long for batch in batches)


def test_jsonl_byte_shards_cover_lines_once():
    rows = [{"text": f"row {idx}", "idx": idx} for idx in range(37)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        seen = []
        for worker_id in range(4):
            for _line_num, line in _iter_jsonl_lines_for_worker(path, worker_id, 4):
                seen.append(json.loads(line)["idx"])

    assert sorted(seen) == list(range(len(rows)))


def test_jsonl_file_shards_assign_whole_files_to_workers():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        expected = []
        for shard_idx in range(4):
            path = os.path.join(tmpdir, f"shard_{shard_idx:05d}.jsonl")
            rows = [{"text": f"row {shard_idx}-{idx}", "idx": shard_idx * 10 + idx} for idx in range(3)]
            expected.extend(row["idx"] for row in rows)
            _write_jsonl(path, rows)
            paths.append(path)

        seen = []
        for worker_id in range(4):
            for _line_num, line in _iter_jsonl_shards_for_worker(paths, worker_id, 4):
                seen.append(json.loads(line)["idx"])

    assert sorted(seen) == sorted(expected)


def test_jsonl_whole_shards_are_size_balanced_and_deterministic():
    sizes = [1000, 900, 800, 700, 600, 500]
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for shard_idx, size in enumerate(sizes):
            path = os.path.join(tmpdir, f"shard_{shard_idx:05d}.jsonl")
            with open(path, "wb") as shard_file:
                shard_file.write(b"x" * size)
            paths.append(path)

        assignments = _assign_jsonl_shards(paths, 3)
        repeated = _assign_jsonl_shards(paths, 3)
        balanced_loads = [sum(os.path.getsize(path) for path in group) for group in assignments]
        modulo_loads = [sum(sizes[idx::3]) for idx in range(3)]

    assert assignments == repeated
    assert sorted(path for group in assignments for path in group) == sorted(paths)
    assert max(balanced_loads) < max(modulo_loads)
    assert max(balanced_loads) - min(balanced_loads) == 0


def test_persistent_jsonl_workers_observe_epoch_updates():
    rows = []
    for row_idx in range(48):
        length = 2 + (row_idx % 12)
        ids = [1000 + row_idx] + [10 + row_idx] * (length - 1)
        rows.append({
            "input_ids": ids,
            "labels": list(ids),
            "attention_mask": [1] * length,
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)
        loader = create_dataloader_for_jsonl(
            path,
            TinyTokenizer(),
            max_length=32,
            batch_size=2,
            pad_token_id=0,
            num_workers=2,
            use_length_bucketing=True,
            bucket_size=24,
        )
        loader.dataset.seed = 9137

        set_dataloader_epoch(loader, 0)
        epoch_zero = [
            int(sample_id)
            for batch in loader
            for sample_id in batch["input_ids"][:, 0].tolist()
        ]
        set_dataloader_epoch(loader, 1)
        epoch_one = [
            int(sample_id)
            for batch in loader
            for sample_id in batch["input_ids"][:, 0].tolist()
        ]

    expected = list(range(1000, 1048))
    assert sorted(epoch_zero) == expected
    assert sorted(epoch_one) == expected
    assert epoch_one != epoch_zero


def test_hf_workers_share_shuffle_seed_before_sharding(monkeypatch):
    import hierarchos.training.datasets as dataset_module

    dataset = HuggingFaceStreamingDataset(
        FakeHFStream([{"text": "a"}, {"text": "b"}]),
        TinyTokenizer(),
        max_length=16,
        batch_size=1,
        shuffle=True,
    )
    dataset.seed = 41
    dataset.set_epoch(3)
    shuffle_seeds = []
    original_shuffle = dataset_module._maybe_shuffle_hf_dataset

    def record_shuffle(hf_dataset, shuffle, shuffle_buffer_size, seed):
        shuffle_seeds.append(seed)
        return original_shuffle(hf_dataset, shuffle, shuffle_buffer_size, seed)

    monkeypatch.setattr(dataset_module, "_maybe_shuffle_hf_dataset", record_shuffle)
    for worker_id in range(2):
        monkeypatch.setattr(
            dataset_module.torch.utils.data,
            "get_worker_info",
            lambda worker_id=worker_id: types.SimpleNamespace(id=worker_id, num_workers=2),
        )
        list(dataset)

    assert len(shuffle_seeds) == 2
    assert shuffle_seeds[0] == shuffle_seeds[1]


def test_streaming_jsonl_directory_shards_batch_once():
    tokenizer = TinyTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        expected_rows = 0
        for shard_idx in range(3):
            rows = [{"text": f"directory shard {shard_idx} sample {idx}"} for idx in range(5)]
            expected_rows += len(rows)
            _write_jsonl(os.path.join(tmpdir, f"shard_{shard_idx:05d}.jsonl"), rows)
        _write_jsonl(os.path.join(tmpdir, "manifest.jsonl"), [{"ignored": True}])

        batches = list(create_dataloader_for_jsonl(
            tmpdir,
            tokenizer,
            max_length=96,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
        ))

    assert sum(batch["input_ids"].shape[0] for batch in batches) == expected_rows


def test_hf_materialization_partitions_rows_without_copying_shards():
    import hierarchos_cli

    original_loader = hierarchos_cli.load_hf_dataset
    rows = [{"text": f"hf row {idx}", "idx": idx} for idx in range(23)]

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: iter(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/hf",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_shard_cache_dir=tmpdir,
                refresh_hf_shards=True,
                dataset_size=len(rows),
            )
            shard_dir = hierarchos_cli.materialize_hf_dataset_shards(args, TinyTokenizer(), 4)

            seen = []
            shard_counts = []
            tokenized_rows = 0
            with open(os.path.join(shard_dir, "manifest.jsonl"), "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
            by_file = {}
            for entry in entries:
                by_file.setdefault(entry["file_path"], []).append(entry["index_in_file"])

            shard_counts_by_prefix = {idx: 0 for idx in range(4)}
            for file_path, indices in by_file.items():
                shard_idx = int(os.path.basename(file_path).split("_")[1])
                shard_items = torch.load(os.path.join(shard_dir, file_path), map_location="cpu")
                shard_counts_by_prefix[shard_idx] += len(indices)
                for index in indices:
                    row = shard_items[index]
                    if "input_ids" in row and "labels" in row:
                        tokenized_rows += 1
                    seen.append(row["_source_idx"])
            shard_counts = list(shard_counts_by_prefix.values())

    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    assert sorted(seen) == list(range(len(rows)))
    assert len(seen) == len(set(seen))
    assert tokenized_rows == len(rows)
    assert max(shard_counts) - min(shard_counts) <= 1


def test_streaming_jsonl_reads_pretokenized_rows_without_tokenizer():
    class ExplodingTokenizer(TinyTokenizer):
        def encode(self, text, add_special_tokens=True):
            raise AssertionError("pre-tokenized JSONL should not call tokenizer.encode")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenized.jsonl")
        _write_jsonl(path, [
            {"input_ids": [10, 11, 12], "labels": [-100, 11, 12], "attention_mask": [1, 1, 1]},
            {"input_ids": [20, 21], "labels": [20, 21], "length": 2},
        ])

        batches = list(create_dataloader_for_jsonl(
            path,
            ExplodingTokenizer(),
            max_length=16,
            batch_size=2,
            pad_token_id=0,
            num_workers=0,
            use_length_bucketing=True,
        ))

    assert len(batches) == 1
    assert int(batches[0]["attention_mask"].sum().item()) == 5


def test_cli_detects_jsonl_shard_directory():
    import hierarchos_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_jsonl(os.path.join(tmpdir, "manifest.jsonl"), [{"ignored": True}])
        _write_jsonl(os.path.join(tmpdir, "shard_00000.jsonl"), [{"text": "visible shard"}])
        _write_jsonl(os.path.join(tmpdir, "shard_00001.ndjson"), [{"text": "second shard"}])

        assert hierarchos_cli._is_jsonl_source(tmpdir)
        assert hierarchos_cli.count_jsonl_source_rows(tmpdir) == 2


def test_cli_prefers_indexed_hf_for_single_streaming_shard_by_default():
    import hierarchos_cli
    from hierarchos.training.datasets import HuggingFaceMapStyleDataset

    class SingleShardStream(FakeHFStream):
        num_shards = 1

    original_loader = hierarchos_cli.load_hf_dataset
    rows = [{"text": f"hf row {idx}"} for idx in range(8)]

    def fake_loader(*args, **kwargs):
        if kwargs.get("streaming"):
            return SingleShardStream(rows)
        return list(rows)

    try:
        hierarchos_cli.load_hf_dataset = fake_loader
        args = types.SimpleNamespace(
            hf_dataset="fake/hf",
            hf_dataset_config=None,
            hf_dataset_split="train",
            max_length=64,
            kayla=False,
            text_column=None,
            prompt_column=None,
            completion_column=None,
            batch_size=4,
            num_workers=4,
            length_bucketing=True,
            length_bucket_size=None,
            streaming_datasets=True,
            hf_auto_shard=False,
            hf_streaming_shuffle_buffer=100,
            prefetch_factor=None,
        )
        dataloader = hierarchos_cli.create_hf_training_dataloader(args, TinyTokenizer(), torch.device("cpu"))
    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    assert isinstance(dataloader.dataset, HuggingFaceMapStyleDataset)


def test_tokenized_binary_cache_is_map_style_and_bucketed():
    tokenizer = TinyTokenizer()
    rows = _alternating_length_rows()

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_binary_token_cache(tmpdir, rows, tokenizer, max_length=256)
        dataset = TokenizedBinaryDataset(tmpdir, max_length=256)
        assert len(dataset) == len(rows)
        assert dataset.get_sample_lengths()
        first = dataset[0]
        assert first["input_ids"].dtype == torch.int32
        assert first["labels"].dtype == torch.int32
        dataset.close()

        unbucketed_loader = create_dataloader_for_tokenized_cache(
            tmpdir,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        )
        unbucketed = list(unbucketed_loader)
        unbucketed_loader.dataset.close()
        bucketed_loader = create_dataloader_for_tokenized_cache(
            tmpdir,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        )
        bucketed = list(bucketed_loader)
        bucketed_loader.dataset.close()

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_cli_builds_and_uses_hf_token_cache():
    import hierarchos_cli
    from hierarchos.training.datasets import TokenizedBinaryDataset

    original_loader = hierarchos_cli.load_hf_dataset
    rows = _alternating_length_rows()

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: list(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/hf",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_token_cache_dir=tmpdir,
                hf_shard_cache_dir=None,
                refresh_hf_token_cache=True,
                tokenizer_path=None,
                model_path=None,
                max_length=96,
                kayla=False,
                text_column=None,
                prompt_column=None,
                completion_column=None,
                batch_size=4,
                num_workers=0,
                length_bucket_size=len(rows),
                    hf_token_cache=True,
                    streaming_datasets=True,
                    prefetch_factor=None,
                    max_cache_truncated_fraction=1.0,
                )
            dataloader = hierarchos_cli.create_hf_training_dataloader(
                args,
                TinyTokenizer(),
                torch.device("cpu"),
            )
            batches = list(dataloader)
            dataloader.dataset.close()
    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    assert isinstance(dataloader.dataset, TokenizedBinaryDataset)
    assert isinstance(dataloader.batch_sampler, LengthGroupedBatchSampler)
    assert sum(batch["input_ids"].shape[0] for batch in batches) == len(rows)
    assert all("rosa_ids" in batch for batch in batches)
    assert all(batch["rosa_ids"].shape == batch["input_ids"].shape for batch in batches)
    assert max(_batch_length_spreads(batches)) <= 32


def test_cli_hf_token_cache_respects_disabled_length_bucketing():
    import hierarchos_cli

    original_loader = hierarchos_cli.load_hf_dataset
    rows = _alternating_length_rows()

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: list(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/hf",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_token_cache_dir=tmpdir,
                hf_shard_cache_dir=None,
                refresh_hf_token_cache=True,
                tokenizer_path=None,
                model_path=None,
                max_length=256,
                kayla=False,
                text_column=None,
                prompt_column=None,
                completion_column=None,
                batch_size=4,
                num_workers=0,
                length_bucketing=False,
                length_bucket_size=len(rows),
                hf_token_cache=True,
                streaming_datasets=True,
                prefetch_factor=None,
            )
            dataloader = hierarchos_cli.create_hf_training_dataloader(
                args,
                TinyTokenizer(),
                torch.device("cpu"),
            )
            try:
                assert not isinstance(dataloader.batch_sampler, LengthGroupedBatchSampler)
            finally:
                dataloader.dataset.close()
    finally:
        hierarchos_cli.load_hf_dataset = original_loader


def test_hf_token_cache_preserves_alpaca_input_and_all_token_labels():
    import hierarchos_cli

    original_loader = hierarchos_cli.load_hf_dataset
    rows = [{
        "instruction": "Answer using the prior context.",
        "input": "Previous turn: the user asked about token caches.",
        "output": "The cache keeps instruction, input, and output tokens.",
    }]

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: list(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/alpaca",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_token_cache_dir=tmpdir,
                hf_shard_cache_dir=None,
                refresh_hf_token_cache=True,
                tokenizer_path=None,
                model_path=None,
                max_length=512,
                kayla=False,
                alpaca=True,
                text_column=None,
                prompt_column=None,
                completion_column=None,
                train_prompt_tokens=True,
                use_rosa=False,
                rosa_max_context=512,
                training_chunk_size=256,
                batch_size=2,
                num_workers=2,
                prefetch_factor=2,
                token_cache_build_batch_size=2,
                token_cache_write_buffer_mb=1,
            )
            cache_dir = hierarchos_cli.materialize_hf_token_cache(args, RecordingTokenizer())
            with open(os.path.join(cache_dir, "_SUCCESS"), "r", encoding="utf-8") as f:
                success = json.load(f)
            index = torch.load(os.path.join(cache_dir, "index.pt"), map_location="cpu")
            dataset = TokenizedBinaryDataset(cache_dir, max_length=512)
            try:
                raw_item = dataset[0]
                item = {
                    "input_ids": raw_item["input_ids"].clone(),
                    "labels": raw_item["labels"].clone(),
                    "_length": raw_item["_length"],
                }
            finally:
                dataset.close()

    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    expected_prompt = (
        "### Previous Context:\nPrevious turn: the user asked about token caches.\n\n"
        "### Instruction:\nAnswer using the prior context.\n\n"
        "### Response:\n"
    )
    expected_ids = (
        TinyTokenizer().encode(expected_prompt)
        + TinyTokenizer().encode("The cache keeps instruction, input, and output tokens.", add_special_tokens=False)
        + [TinyTokenizer.eos_token_id]
    )
    assert success["format"] == hierarchos_cli._hf_token_cache_format(args)
    assert success["formatter"] == hierarchos_cli.HF_CACHE_FORMATTER_VERSION
    assert success["cache_payload"]["alpaca_input_role"] == "previous_context"
    assert success["cache_payload"]["train_prompt_tokens"] is True
    assert index["formatter"] == hierarchos_cli.HF_CACHE_FORMATTER_VERSION
    assert item["_length"] == len(expected_ids)
    assert item["input_ids"].tolist() == expected_ids
    assert item["labels"].tolist() == expected_ids
    assert not (item["labels"] == -100).any()


def test_local_jsonl_token_cache_reuses_exact_records_and_invalidates_on_change():
    import hierarchos_cli

    rows = [
        {"text": "short cache row"},
        {"text": "a somewhat longer cache row"},
        {"text": "final cache row"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "train.jsonl")
        cache_root = os.path.join(tmpdir, "cache")
        _write_jsonl(source_path, rows)
        args = types.SimpleNamespace(
            train=source_path,
            hf_dataset=None,
            hf_dataset_config=None,
            hf_dataset_split="train",
            local_token_cache_dir=cache_root,
            hf_token_cache_dir=None,
            refresh_local_token_cache=False,
            tokenizer_path="tiny-test-tokenizer",
            model_path=None,
            max_length=64,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
            dataset_size=None,
            token_cache_build_batch_size=2,
            token_cache_write_buffer_mb=1,
            kayla=False,
            alpaca=False,
            text_column=None,
            prompt_column=None,
            completion_column=None,
            train_prompt_tokens=True,
            prompt_loss_weight=1.0,
            response_loss_weight=1.0,
            response_boundary_loss_weight=1.0,
            response_boundary_tokens=0,
            min_response_tokens=1,
            drop_empty_completions=True,
            use_rosa=True,
            rosa_max_context=512,
            training_chunk_size=256,
        )
        tokenizer = RecordingTokenizer()

        first_cache = hierarchos_cli.materialize_local_token_cache(args, tokenizer)
        calls_after_build = len(tokenizer.calls)
        reused_cache = hierarchos_cli.materialize_local_token_cache(args, tokenizer)

        assert reused_cache == first_cache
        assert len(tokenizer.calls) == calls_after_build

        portable_source_dir = os.path.join(tmpdir, "portable-copy")
        os.makedirs(portable_source_dir)
        portable_source = os.path.join(portable_source_dir, "train.jsonl")
        shutil.copy2(source_path, portable_source)
        args.train = portable_source
        assert hierarchos_cli.materialize_local_token_cache(args, tokenizer) == first_cache
        assert len(tokenizer.calls) == calls_after_build
        args.train = source_path

        dataset = TokenizedBinaryDataset(first_cache, max_length=64)
        try:
            assert len(dataset) == len(rows)
            for row_idx, row in enumerate(rows):
                expected = process_text_sample(TinyTokenizer(), row, 64)
                actual = dataset[row_idx]
                assert torch.equal(actual["input_ids"], expected["input_ids"])
                assert torch.equal(actual["labels"], expected["labels"])
                assert actual["_length"] == expected["_length"]
                expected_rosa = torch.tensor(
                    hierarchos_cli.precompute_rosa_ids_for_chunks(
                        expected["input_ids"].tolist(),
                        vocab_size=actual["_rosa_sentinel"],
                        chunk_size=args.training_chunk_size,
                        rosa_max_ctx=args.rosa_max_context,
                    ),
                    dtype=torch.int32,
                )
                assert torch.equal(actual["rosa_ids"], expected_rosa)
        finally:
            dataset.close()

        rows.append({"text": "a newly appended row invalidates the fingerprint"})
        _write_jsonl(source_path, rows)
        changed_cache = hierarchos_cli.materialize_local_token_cache(args, tokenizer)
        changed_dataset = TokenizedBinaryDataset(changed_cache, max_length=64)
        try:
            assert changed_cache != first_cache
            assert len(changed_dataset) == len(rows)
        finally:
            changed_dataset.close()


def test_cache_effective_lengths_use_last_real_token_not_mask_sum():
    import hierarchos_cli

    batch = {
        "input_ids": torch.arange(8, dtype=torch.long).reshape(2, 4),
        "attention_mask": torch.tensor([
            [0, 1, 1, 1],
            [1, 0, 1, 0],
        ], dtype=torch.long),
    }
    assert hierarchos_cli._batch_effective_lengths(batch) == [4, 3]


def test_hf_cache_keys_reject_legacy_masked_or_ambiguous_formatter_versions():
    import hierarchos_cli

    args = types.SimpleNamespace(
        hf_dataset="netcat420/Experiment_0.1",
        hf_dataset_config=None,
        hf_dataset_split="train",
        tokenizer_path=None,
        model_path=None,
        max_length=8880,
        kayla=False,
        alpaca=True,
        train_prompt_tokens=True,
        text_column=None,
        prompt_column=None,
        completion_column=None,
        use_rosa=True,
        rosa_max_context=512,
        training_chunk_size=256,
        hf_cache_chunks_per_file=2048,
    )

    assert hierarchos_cli._hf_token_cache_key(args) != hierarchos_cli._legacy_hf_token_cache_key(args)
    assert hierarchos_cli._hf_shard_cache_key(args, 8) != hierarchos_cli._legacy_hf_shard_cache_key(args, 8)
    payload = hierarchos_cli._hf_cache_key_payload(
        args,
        format_name=hierarchos_cli._hf_token_cache_format(args),
    )
    assert payload["formatter"] == hierarchos_cli.HF_CACHE_FORMATTER_VERSION
    assert payload["alpaca_input_field"] == "input"
    assert payload["alpaca_input_role"] == "previous_context"
    assert payload["train_prompt_tokens"] is True
    assert payload["min_response_tokens"] == 1
    assert payload["drop_empty_completions"] is True


def test_hf_streaming_dataloader_batches():
    tokenizer = TinyTokenizer()
    hf_rows = FakeHFStream({"text": f"streamed sample {idx}"} for idx in range(10))

    dataloader = create_dataloader_for_hf_streaming(
        hf_rows,
        tokenizer,
        max_length=64,
        batch_size=3,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        use_length_bucketing=False,
        shuffle=False,
    )

    assert isinstance(dataloader.dataset, HuggingFaceStreamingDataset)
    batches = list(dataloader)
    assert len(batches) == 4
    assert sum(batch["input_ids"].shape[0] for batch in batches) == 10


def test_hf_worker_shards_cover_samples_once():
    hf_rows = FakeHFStream({"text": f"row {idx}", "idx": idx} for idx in range(19))
    seen = []

    for worker_id in range(3):
        for sample in _iter_hf_worker_samples(hf_rows, worker_id, 3):
            seen.append(sample["idx"])

    assert sorted(seen) == list(range(19))


def test_streaming_jsonl_length_buckets_reduce_padding():
    tokenizer = TinyTokenizer()
    rows = _alternating_length_rows()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        unbucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        ))
        bucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        ))

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_hf_streaming_length_buckets_reduce_padding():
    tokenizer = TinyTokenizer()
    hf_rows = FakeHFStream(_alternating_length_rows())

    unbucketed = list(create_dataloader_for_hf_streaming(
        hf_rows,
        tokenizer,
        max_length=256,
        batch_size=4,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        use_length_bucketing=False,
        shuffle=False,
    ))
    bucketed = list(create_dataloader_for_hf_streaming(
        hf_rows,
        tokenizer,
        max_length=256,
        batch_size=4,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        use_length_bucketing=True,
        bucket_size=len(hf_rows.rows),
        shuffle=False,
    ))

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(hf_rows.rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_prechunked_jsonl_streaming_length_buckets_reduce_padding():
    max_length = 64
    valid_lengths = [4, 60, 8, 56, 12, 52, 16, 48, 20, 44, 24, 40]
    rows = []
    for idx, valid_length in enumerate(valid_lengths):
        input_ids = list(range(1, valid_length + 1)) + [0] * (max_length - valid_length)
        labels = list(input_ids)
        attention_mask = [1] * valid_length + [0] * (max_length - valid_length)
        rows.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "idx": idx,
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "chunked.jsonl")
        _write_jsonl(path, rows)

        unbucketed = list(create_dataloader_for_chunked(
            path,
            max_length=max_length,
            batch_size=4,
            num_workers=0,
            use_length_bucketing=False,
        ))
        bucketed = list(create_dataloader_for_chunked(
            path,
            max_length=max_length,
            batch_size=4,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        ))

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_length_bucketing_still_reduces_padding_with_static_compile_chunks():
    tokenizer = TinyTokenizer()
    rows = _alternating_length_rows()
    multiple = 32

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        unbucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        ))
        bucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        ))

    unbucketed_static = _compile_padded_batches(unbucketed, multiple)
    bucketed_static = _compile_padded_batches(bucketed, multiple)

    assert sum(batch["input_ids"].shape[0] for batch in bucketed_static) == len(rows)
    assert _padding_waste(bucketed_static) < _padding_waste(unbucketed_static)


def test_map_style_jsonl_fallback_still_loads_text_samples():
    tokenizer = TinyTokenizer()
    rows = [{"text": f"fallback {idx}"} for idx in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        dataset = OriginalJSONLDataset(path, tokenizer, max_length=64)

    assert len(dataset) == 5
    assert dataset.get_sample_lengths()


def main():
    tests = [
        test_process_text_sample_autodetects_common_columns,
        test_alpaca_instruction_input_output_uses_input_section,
        test_alpaca_prompt_tokens_can_be_masked_for_legacy_sft,
        test_alpaca_flag_defaults_instruction_output_columns,
        test_streaming_jsonl_dataloader_batches_without_materializing,
        test_jsonl_byte_shards_cover_lines_once,
        test_jsonl_file_shards_assign_whole_files_to_workers,
        test_streaming_jsonl_directory_shards_batch_once,
        test_hf_materialization_partitions_rows_without_copying_shards,
        test_streaming_jsonl_reads_pretokenized_rows_without_tokenizer,
        test_cli_detects_jsonl_shard_directory,
        test_cli_prefers_indexed_hf_for_single_streaming_shard_by_default,
        test_tokenized_binary_cache_is_map_style_and_bucketed,
        test_cli_builds_and_uses_hf_token_cache,
        test_cli_hf_token_cache_respects_disabled_length_bucketing,
        test_hf_token_cache_preserves_alpaca_input_and_all_token_labels,
        test_hf_cache_keys_reject_legacy_masked_or_ambiguous_formatter_versions,
        test_hf_streaming_dataloader_batches,
        test_hf_worker_shards_cover_samples_once,
        test_streaming_jsonl_length_buckets_reduce_padding,
        test_hf_streaming_length_buckets_reduce_padding,
        test_prechunked_jsonl_streaming_length_buckets_reduce_padding,
        test_length_bucketing_still_reduces_padding_with_static_compile_chunks,
        test_map_style_jsonl_fallback_still_loads_text_samples,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
