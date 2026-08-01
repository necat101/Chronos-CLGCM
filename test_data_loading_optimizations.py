import json
import os
import types

import torch

import hierarchos_cli


def _hf_args(revision=None):
    return types.SimpleNamespace(
        hf_dataset="owner/dataset",
        hf_dataset_config=None,
        hf_dataset_split="train",
        hf_dataset_revision=revision,
        tokenizer_path="openai-community/gpt2",
        model_path=None,
        max_length=8880,
        kayla=False,
        alpaca=True,
        train_prompt_tokens=True,
        prompt_loss_weight=0.10,
        response_loss_weight=1.0,
        response_boundary_loss_weight=2.0,
        response_boundary_tokens=64,
        min_response_tokens=128,
        drop_empty_completions=True,
        text_column=None,
        prompt_column=None,
        completion_column=None,
        use_rosa=True,
        rosa_max_context=512,
        enforce_rosa_max_context=False,
        training_chunk_size=256,
    )


def test_hf_cache_key_is_pinned_to_resolved_commit(monkeypatch):
    import huggingface_hub

    class _Api:
        def dataset_info(self, repo_id, revision=None):
            assert repo_id == "owner/dataset"
            assert revision == "release"
            return types.SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _Api())
    args = _hf_args("release")
    assert hierarchos_cli.resolve_hf_dataset_revision(args) == "a" * 40
    first_key = hierarchos_cli._hf_token_cache_key(args)
    first_payload = hierarchos_cli._hf_cache_key_payload(
        args,
        format_name=hierarchos_cli._hf_token_cache_format(args),
    )
    assert first_payload["dataset_revision"] == "a" * 40

    args._resolved_hf_dataset_revision = "b" * 40
    assert hierarchos_cli._hf_token_cache_key(args) != first_key


def test_hf_cache_key_versions_bounded_rosa_semantics():
    args = _hf_args("a" * 40)
    args._resolved_hf_dataset_revision = "a" * 40
    legacy_key = hierarchos_cli._hf_token_cache_key(args)
    legacy_payload = hierarchos_cli._hf_cache_key_payload(
        args,
        format_name=hierarchos_cli._hf_token_cache_format(args),
    )
    assert legacy_payload["rosa_ids_context_mode"] == "legacy-unbounded-v1"

    args.enforce_rosa_max_context = True
    bounded_key = hierarchos_cli._hf_token_cache_key(args)
    bounded_payload = hierarchos_cli._hf_cache_key_payload(
        args,
        format_name=hierarchos_cli._hf_token_cache_format(args),
    )
    assert bounded_key != legacy_key
    assert bounded_payload["enforce_rosa_max_context"] is True
    assert bounded_payload["rosa_ids_context_mode"] == "bounded-segment-v1"


def test_nonstreaming_hf_load_parallelizes_arrow_preparation(monkeypatch):
    import datasets

    calls = []

    def _fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    hierarchos_cli.load_hf_dataset(
        "owner/dataset",
        split="train",
        revision="a" * 40,
        num_proc=8,
    )
    assert calls[-1][1]["num_proc"] == 8
    assert calls[-1][1]["revision"] == "a" * 40

    hierarchos_cli.load_hf_dataset(
        "owner/dataset",
        split="train",
        streaming=True,
        num_proc=8,
    )
    assert "num_proc" not in calls[-1][1]


def test_hf_size_estimate_uses_the_same_pinned_revision_as_training(monkeypatch):
    import datasets

    calls = []

    def _fake_builder(*args, **kwargs):
        calls.append((args, kwargs))
        split = types.SimpleNamespace(num_examples=123)
        return types.SimpleNamespace(info=types.SimpleNamespace(splits={"train": split}))

    monkeypatch.setattr(datasets, "load_dataset_builder", _fake_builder)

    assert hierarchos_cli.estimate_hf_dataset_size(
        "owner/dataset",
        split="train",
        revision="c" * 40,
    ) == 123
    assert calls[-1][1] == {"revision": "c" * 40}


def test_bucket_auto_tuning_is_sampled_once_and_persisted(tmp_path, monkeypatch):
    lengths = torch.tensor(([8, 250, 12, 245, 16, 240, 20, 235] * 128), dtype=torch.int32)
    torch.save(
        {
            "cache_key": "immutable-cache",
            "offsets": torch.zeros(lengths.numel(), dtype=torch.long),
            "lengths": lengths,
        },
        tmp_path / "index.pt",
    )
    args = types.SimpleNamespace(
        max_length=8880,
        length_bucket_auto_sample_size=256,
        batch_size=8,
        training_chunk_size=64,
        length_bucket_auto_tolerance=0.005,
        length_bucket_size=None,
    )
    first = hierarchos_cli.auto_tune_length_bucket_size_from_token_cache(args, str(tmp_path))
    assert first == args.length_bucket_size
    tuning_path = tmp_path / "bucket_tuning.json"
    assert tuning_path.exists()
    with tuning_path.open("r", encoding="utf-8") as tuning_file:
        persisted = json.load(tuning_file)
    assert persisted["sampled_lengths"] == 256
    assert persisted["settings"]["source_samples"] == lengths.numel()

    def _must_not_retune(*_args, **_kwargs):
        raise AssertionError("persisted bucket setting was not reused")

    monkeypatch.setattr(
        hierarchos_cli,
        "choose_length_bucket_size_from_lengths",
        _must_not_retune,
    )
    args.length_bucket_size = None
    assert hierarchos_cli.auto_tune_length_bucket_size_from_token_cache(args, str(tmp_path)) == first


def test_cache_only_cpu_auto_workers_do_not_change_cpu_training_default(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    cpu = torch.device("cpu")
    assert hierarchos_cli.resolve_num_workers(-1, cpu, 64) == 0
    assert hierarchos_cli.resolve_num_workers(-1, cpu, 64, token_cache_only=True) == 8
