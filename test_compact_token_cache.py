import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

import hierarchos_cli
from hierarchos.training.datasets import TokenizedBinaryDataset
from hierarchos.utils.rosa import precompute_rosa_ids_for_chunks


class _GPT2SizedTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 50257


def _cache_args(source_path, cache_root):
    return SimpleNamespace(
        train=str(source_path),
        local_token_cache_dir=str(cache_root),
        hf_token_cache_dir=None,
        hf_dataset=None,
        hf_dataset_config=None,
        hf_dataset_split="train",
        hf_dataset_revision=None,
        tokenizer_path="openai-community/gpt2",
        model_path=None,
        max_length=8,
        kayla=False,
        alpaca=True,
        train_prompt_tokens=True,
        prompt_loss_weight=0.10,
        response_loss_weight=1.0,
        response_boundary_loss_weight=2.0,
        response_boundary_tokens=2,
        min_response_tokens=1,
        drop_empty_completions=True,
        text_column=None,
        prompt_column=None,
        completion_column=None,
        use_rosa=True,
        training_chunk_size=4,
        rosa_max_context=8,
        enforce_rosa_max_context=False,
        token_cache_build_batch_size=2,
        token_cache_write_buffer_mb=1,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        dataset_size=2,
        refresh_local_token_cache=False,
    )


def _dummy_weighted_batch():
    return {
        "input_ids": torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, 0, 0]],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, -100, -100]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
            dtype=torch.long,
        ),
        "loss_weights": torch.tensor(
            [[0.10, 0.10, 2.0, 2.0, 1.0], [0.10, 2.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "rosa_ids": torch.tensor(
            [[50257, 10, 11, 12, 13], [50257, 20, 21, 50257, 50257]],
            dtype=torch.long,
        ),
    }


def test_cache_directory_publish_retries_transient_windows_permission_error(monkeypatch):
    calls = []
    sleeps = []

    def _flaky_replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError("simulated transient directory handle")

    monkeypatch.setattr(
        hierarchos_cli,
        "_RETRY_CACHE_DIRECTORY_PERMISSION_ERRORS",
        True,
    )
    monkeypatch.setattr(hierarchos_cli.os, "replace", _flaky_replace)
    monkeypatch.setattr(hierarchos_cli.time, "sleep", sleeps.append)

    hierarchos_cli._replace_cache_directory_atomically(
        "cache.tmp",
        "cache",
        attempts=5,
    )

    assert calls == [("cache.tmp", "cache")] * 3
    assert sleeps == [0.05, 0.10]


def test_cache_directory_publish_fails_closed_after_retry_budget(monkeypatch):
    calls = []

    def _blocked_replace(source, destination):
        calls.append((source, destination))
        raise PermissionError("simulated persistent directory handle")

    monkeypatch.setattr(
        hierarchos_cli,
        "_RETRY_CACHE_DIRECTORY_PERMISSION_ERRORS",
        True,
    )
    monkeypatch.setattr(hierarchos_cli.os, "replace", _blocked_replace)
    monkeypatch.setattr(hierarchos_cli.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError, match="persistent directory handle"):
        hierarchos_cli._replace_cache_directory_atomically(
            "cache.tmp",
            "cache",
            attempts=3,
        )

    assert calls == [("cache.tmp", "cache")] * 3


def test_cached_item_uses_versioned_bounded_rosa_precompute(tmp_path):
    args = _cache_args(tmp_path / "source.jsonl", tmp_path / "cache")
    args.enforce_rosa_max_context = True
    tokens = [1, 2, 1, 2, 3, 1, 2, 4, 1, 2]
    processed = {
        "input_ids": torch.tensor(tokens, dtype=torch.long),
        "labels": torch.tensor(tokens, dtype=torch.long),
        "_length": len(tokens),
    }
    item = hierarchos_cli._processed_sample_to_cached_item(
        processed,
        args=args,
        tokenizer=_GPT2SizedTokenizer(),
    )
    expected = precompute_rosa_ids_for_chunks(
        tokens,
        vocab_size=50257,
        chunk_size=args.training_chunk_size,
        rosa_max_ctx=args.rosa_max_context,
        enforce_max_context=True,
    )
    assert item["_rosa_context_mode"] == "bounded-segment-v1"
    assert item["rosa_ids"].tolist() == expected


def test_local_compact_cache_is_lossless_small_and_backward_ready(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    cache_root = tmp_path / "cache"
    args = _cache_args(source_path, cache_root)
    batch = _dummy_weighted_batch()
    build_calls = []
    builder_kwargs = {}

    def _fake_builder(*_args, **kwargs):
        build_calls.append(1)
        builder_kwargs.update(kwargs)
        return [batch]

    monkeypatch.setattr(hierarchos_cli, "create_dataloader_for_jsonl", _fake_builder)
    cache_dir = hierarchos_cli.materialize_local_token_cache(args, _GPT2SizedTokenizer())

    index = torch.load(os.path.join(cache_dir, "index.pt"), map_location="cpu", weights_only=True)
    assert index["storage_schema_version"] == 6
    assert index["token_dtype"] == "uint16"
    assert index["label_dtype"] is None
    assert index["label_encoding"] == "input_ids_alias"
    assert index["rosa_dtype"] == "uint16"
    assert index["label_ignore_sentinel"] is None
    assert index["loss_weight_encoding"] == "float32_palette_rle"
    assert index["rosa_ids_context_mode"] == "legacy-unbounded-v1"
    assert index["enforce_rosa_max_context"] is False
    assert builder_kwargs["in_order"] is True
    assert builder_kwargs["enforce_rosa_max_context"] is False
    assert len(index["ordered_record_sha256"]) == 64
    assert len(index["tokens_sha256"]) == 64

    # Eight real tokens at four bytes/token: uint16 input plus ROSA. Labels are
    # reconstructed exactly from input ids after the writer verifies equality.
    data_path = os.path.join(cache_dir, "tokens.bin")
    assert os.path.getsize(data_path) == 8 * 4
    with open(os.path.join(cache_dir, "_SUCCESS"), "r", encoding="utf-8") as success_file:
        success = json.load(success_file)
    assert success["bytes"] == 8 * 4
    assert success["ordered_record_sha256"] == index["ordered_record_sha256"]
    assert success["tokens_sha256"] == index["tokens_sha256"]
    assert args._token_cache_identity["tokens_sha256"] == index["tokens_sha256"]
    assert success["rosa_ids_context_mode"] == "legacy-unbounded-v1"
    with open(os.path.join(cache_dir, "cache_audit.json"), "r", encoding="utf-8") as audit_file:
        audit = json.load(audit_file)
    assert audit["accepted"] == 2
    assert audit["retained_tokens"] == 8

    dataset = TokenizedBinaryDataset(cache_dir, max_length=4, pad_token_id=99)
    first = dataset[0]
    assert first["input_ids"].dtype == torch.int32
    assert torch.equal(first["input_ids"], torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert torch.equal(first["labels"], torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert torch.equal(first["loss_weights"], batch["loss_weights"][0, :4])
    assert torch.equal(first["rosa_ids"], batch["rosa_ids"][0, :4].to(torch.int32))

    fused = dataset.__getitems__([0, 1])
    assert fused["input_ids"].dtype == torch.int32
    assert fused["labels"].dtype == torch.int32
    assert fused["attention_mask"].dtype == torch.bool
    assert fused["rosa_ids"].dtype == torch.int32
    assert fused["rosa_ids_context_mode"] == "legacy-unbounded-v1"
    assert torch.equal(
        fused["labels"],
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, -100]], dtype=torch.int32),
    )
    assert torch.equal(fused["loss_weights"][0], batch["loss_weights"][0, :4])
    assert torch.equal(fused["loss_weights"][1, :3], batch["loss_weights"][1, :3])
    assert fused["loss_weights"][1, 3].item() == 0.0
    reordered = dataset.__getitems__([1, 0, 1])
    assert torch.equal(reordered["input_ids"][0, :3], torch.tensor([20, 21, 22], dtype=torch.int32))
    assert torch.equal(reordered["input_ids"][1], torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert torch.equal(reordered["input_ids"][2, :3], torch.tensor([20, 21, 22], dtype=torch.int32))
    assert torch.equal(reordered["labels"][1], reordered["input_ids"][1])
    dataset.close()

    # A completed immutable-key cache is reused rather than rebuilt.
    assert hierarchos_cli.materialize_local_token_cache(args, _GPT2SizedTokenizer()) == cache_dir
    assert len(build_calls) == 1


def test_completed_cache_rejects_same_size_binary_corruption(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    args = _cache_args(source_path, tmp_path / "cache")
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [_dummy_weighted_batch()],
    )
    cache_dir = hierarchos_cli.materialize_local_token_cache(
        args,
        _GPT2SizedTokenizer(),
    )
    data_path = os.path.join(cache_dir, "tokens.bin")
    original_size = os.path.getsize(data_path)
    with open(data_path, "r+b") as data_file:
        first = data_file.read(1)
        data_file.seek(0)
        data_file.write(bytes([first[0] ^ 0x01]))
    assert os.path.getsize(data_path) == original_size

    with pytest.raises(RuntimeError, match="binary checksum failed"):
        hierarchos_cli.materialize_local_token_cache(
            args,
            _GPT2SizedTokenizer(),
        )


@pytest.mark.parametrize(
    "field,mutate,match",
    [
        (
            "cache_payload",
            lambda success: {**success["cache_payload"], "max_length": 999},
            "cache_payload metadata disagrees",
        ),
        ("samples", lambda success: int(success["samples"]) + 1, "sample count mismatch"),
        ("bytes", lambda success: int(success["bytes"]) + 1, "byte count mismatch"),
    ],
)
def test_completed_cache_rejects_success_index_metadata_drift(
    tmp_path,
    monkeypatch,
    field,
    mutate,
    match,
):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    args = _cache_args(source_path, tmp_path / "cache")
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [_dummy_weighted_batch()],
    )
    cache_dir = hierarchos_cli.materialize_local_token_cache(
        args,
        _GPT2SizedTokenizer(),
    )
    success_path = os.path.join(cache_dir, "_SUCCESS")
    with open(success_path, "r", encoding="utf-8") as success_file:
        success = json.load(success_file)
    success[field] = mutate(success)
    with open(success_path, "w", encoding="utf-8") as success_file:
        json.dump(success, success_file)

    with pytest.raises(RuntimeError, match=match):
        hierarchos_cli.materialize_local_token_cache(
            args,
            _GPT2SizedTokenizer(),
        )


def test_ordered_cache_identity_is_reproducible_across_fresh_roots(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    batch = _dummy_weighted_batch()
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [batch],
    )
    identities = []
    for name in ("cache-a", "cache-b"):
        args = _cache_args(source_path, tmp_path / name)
        cache_dir = hierarchos_cli.materialize_local_token_cache(
            args,
            _GPT2SizedTokenizer(),
        )
        identities.append(args._token_cache_identity["ordered_record_sha256"])
        assert os.path.exists(os.path.join(cache_dir, "cache_audit.json"))
    assert identities[0] == identities[1]


def test_cache_build_fails_when_rejection_budget_is_exceeded(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    args = _cache_args(source_path, tmp_path / "cache")
    batch = dict(_dummy_weighted_batch())
    batch["_audit_records"] = [
        {
            "accepted": True,
            "schema": "alpaca:instruction-output",
            "source": "fixture",
            "retained_tokens": 5,
            "supervised_tokens": 4,
            "weighted_tokens": 4.0,
            "retained_response_tokens": 2,
        },
        {
            "accepted": False,
            "schema": "alpaca:instruction-output",
            "source": "fixture",
            "rejection_reason": "response_below_minimum",
        },
    ]
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [batch],
    )
    with pytest.raises(RuntimeError, match="data quality budget exceeded"):
        hierarchos_cli.materialize_local_token_cache(
            args,
            _GPT2SizedTokenizer(),
        )


def test_hf_schema_v6_cache_can_move_roots_without_retokenizing(tmp_path, monkeypatch, capsys):
    """A complete immutable-key HF cache is self-contained and relocatable."""
    original_root = tmp_path / "original-cache-root"
    relocated_root = tmp_path / "relocated-cache-root"
    args = _cache_args(tmp_path / "unused.jsonl", original_root)
    revision = "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51"
    args.hf_dataset = "netcat420/Experiment_0.1"
    args.hf_dataset_revision = revision
    args._resolved_hf_dataset_revision = revision
    args.hf_token_cache_dir = str(original_root)
    args.hf_shard_cache_dir = None
    args.refresh_hf_token_cache = False
    args.prefetch_factor = None

    batch = _dummy_weighted_batch()
    load_calls = []

    def _fake_hf_load(*_args, **_kwargs):
        load_calls.append(1)
        return [{}, {}]

    monkeypatch.setattr(hierarchos_cli, "load_hf_dataset", _fake_hf_load)
    monkeypatch.setattr(
        hierarchos_cli,
        "create_map_style_dataloader",
        lambda *_args, **_kwargs: [batch],
    )
    original_leaf = hierarchos_cli.materialize_hf_token_cache(
        args,
        _GPT2SizedTokenizer(),
    )
    assert len(load_calls) == 1
    assert {
        "_SUCCESS",
        "index.pt",
        "tokens.bin",
    }.issubset(set(os.listdir(original_leaf)))
    index = torch.load(
        os.path.join(original_leaf, "index.pt"),
        map_location="cpu",
        weights_only=True,
    )
    assert index["storage_schema_version"] == 6
    assert index["cache_payload"]["dataset_revision"] == revision

    cache_key = os.path.basename(original_leaf)
    relocated_leaf = relocated_root / cache_key
    shutil.copytree(original_leaf, relocated_leaf)
    shutil.rmtree(original_root)

    relocated_args = SimpleNamespace(**vars(args))
    relocated_args.hf_token_cache_dir = str(relocated_root)

    def _must_not_retokenize(*_args, **_kwargs):
        raise AssertionError("a relocated complete cache must not reload the HF dataset")

    monkeypatch.setattr(hierarchos_cli, "load_hf_dataset", _must_not_retokenize)
    reused_leaf = hierarchos_cli.materialize_hf_token_cache(
        relocated_args,
        _GPT2SizedTokenizer(),
    )
    assert reused_leaf == str(relocated_leaf)
    assert "Reusing HF random-access token cache" in capsys.readouterr().out

    dataset = TokenizedBinaryDataset(reused_leaf, max_length=4, pad_token_id=99)
    try:
        assert len(dataset) == 2
        assert torch.equal(
            dataset[0]["input_ids"],
            torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        )
        assert torch.equal(dataset[0]["loss_weights"], batch["loss_weights"][0, :4])
    finally:
        dataset.close()


def test_compact_cache_rejects_corrupt_loss_run_metadata(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    args = _cache_args(source_path, tmp_path / "cache")
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [_dummy_weighted_batch()],
    )
    cache_dir = hierarchos_cli.materialize_local_token_cache(args, _GPT2SizedTokenizer())
    index_path = os.path.join(cache_dir, "index.pt")
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    index["loss_run_ends"][-1] -= 1
    torch.save(index, index_path)

    with pytest.raises(ValueError, match="final loss run"):
        TokenizedBinaryDataset(cache_dir)

    index["loss_run_ends"] = index["loss_run_ends"].to(dtype=torch.float32)
    torch.save(index, index_path)
    with pytest.raises(ValueError, match="loss-run ends must use an integer dtype"):
        TokenizedBinaryDataset(cache_dir)


def test_explicit_label_cache_preserves_masking_and_alias_mode_rejects_it(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    masked = _dummy_weighted_batch()
    masked["labels"] = masked["labels"].clone()
    masked["labels"][0, :2] = -100
    masked["labels"][1, 0] = -100
    masked["loss_weights"] = masked["loss_weights"].clone()
    masked["loss_weights"][0, :2] = 0.0
    masked["loss_weights"][1, 0] = 0.0

    alias_args = _cache_args(source_path, tmp_path / "alias-cache")
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [masked],
    )
    with pytest.raises(ValueError, match="Cannot elide token-cache labels"):
        hierarchos_cli.materialize_local_token_cache(alias_args, _GPT2SizedTokenizer())

    explicit_args = _cache_args(source_path, tmp_path / "explicit-cache")
    explicit_args.train_prompt_tokens = False
    cache_dir = hierarchos_cli.materialize_local_token_cache(
        explicit_args,
        _GPT2SizedTokenizer(),
    )
    index = torch.load(os.path.join(cache_dir, "index.pt"), map_location="cpu", weights_only=True)
    assert index["label_encoding"] is None
    assert index["label_dtype"] == "uint16"
    assert index["label_ignore_sentinel"] == 65535
    assert os.path.getsize(os.path.join(cache_dir, "tokens.bin")) == 8 * 6

    dataset = TokenizedBinaryDataset(cache_dir)
    try:
        assert torch.equal(dataset[0]["labels"], masked["labels"][0])
        assert torch.equal(dataset[1]["labels"], masked["labels"][1, :3])
    finally:
        dataset.close()
