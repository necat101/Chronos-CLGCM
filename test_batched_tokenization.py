import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from hierarchos.training.datasets import (
    HuggingFaceMapStyleDataset,
    _collate_training_batch,
    create_map_style_dataloader,
    process_text_sample,
    process_text_samples_batch,
)


class _CountingFastTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_calls = []

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, (list, tuple)):
            self.batch_calls.append((list(text), bool(kwargs.get("add_special_tokens", True))))
        return super().__call__(text, *args, **kwargs)


class _FailingFastTokenizer(_CountingFastTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_attempts = 0

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, (list, tuple)):
            self.batch_attempts += 1
            raise RuntimeError("simulated batch tokenizer failure")
        return super().__call__(text, *args, **kwargs)


class _MalformedFastTokenizer(_CountingFastTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_attempts = 0

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, (list, tuple)):
            self.batch_attempts += 1
            return {"input_ids": [[1, 2]]}
        return super().__call__(text, *args, **kwargs)


# Model native AutoTokenizer classes live under transformers.*. Keep these test
# doubles inside the same safety boundary; arbitrary user subclasses fall back.
_CountingFastTokenizer.__module__ = "transformers.testing"
_FailingFastTokenizer.__module__ = "transformers.testing"
_MalformedFastTokenizer.__module__ = "transformers.testing"


def _make_fast_tokenizer(tokenizer_cls=_CountingFastTokenizer):
    vocabulary = {
        "[UNK]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "hello": 3,
        "world": 4,
        "explain": 5,
        "hierarchy": 6,
        "prior": 7,
        "context": 8,
        "final": 9,
        "answer": 10,
        "question": 11,
        "reply": 12,
        "thought": 13,
        "calm": 14,
        "Instruction": 15,
        "Response": 16,
        "Previous": 17,
        "Assistant": 18,
        "User": 19,
        "Feelings": 20,
        "Thought": 21,
        "Process": 22,
        ":": 23,
        "#": 24,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
    )
    return tokenizer_cls(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )


def _assert_sample_equal(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
        return
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            assert actual[key].dtype == expected[key].dtype, key
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def _assert_batches_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_sample, expected_sample in zip(actual, expected):
        _assert_sample_equal(actual_sample, expected_sample)


def test_fast_batch_tokenization_matches_scalar_weighted_alpaca_semantics():
    tokenizer = _make_fast_tokenizer()
    samples = [
        {
            "instruction": "explain hierarchy",
            "input": "prior context",
            "output": "final answer",
        },
        {
            "instruction": "question",
            "input": "",
            "output": "reply final answer",
        },
        {"instruction": "question", "input": "prior", "output": ""},
    ]
    args = (19, False, None, None, None, True, False, 0.1, 1.0, 2.0, 3, 2, True)
    expected = [process_text_sample(tokenizer, sample, *args) for sample in samples]
    tokenizer.batch_calls.clear()

    actual = process_text_samples_batch(tokenizer, samples, *args)

    _assert_batches_equal(actual, expected)
    assert len(tokenizer.batch_calls) == 2
    false_special_texts = [
        text
        for texts, add_special_tokens in tokenizer.batch_calls
        if not add_special_tokens
        for text in texts
    ]
    # Completions are batch entries of their own. They were not concatenated to
    # prompts, so BPE merges cannot cross the scalar prompt/completion boundary.
    assert "final answer" in false_special_texts
    assert "reply final answer" in false_special_texts


def test_min_response_tokens_filters_short_rows_and_audits_truncation():
    tokenizer = _make_fast_tokenizer()
    short = {"instruction": "question", "output": "reply"}
    assert process_text_sample(
        tokenizer,
        short,
        32,
        alpaca_mode=True,
        min_response_tokens=2,
    ) is None
    rejected = process_text_sample(
        tokenizer,
        short,
        32,
        alpaca_mode=True,
        min_response_tokens=2,
        include_audit=True,
        audit_source_default="fixture",
    )
    assert rejected["_audit_only"] is True
    assert rejected["_audit"]["rejection_reason"] == "response_below_minimum"
    assert rejected["_audit"]["source"] == "fixture"

    accepted = process_text_sample(
        tokenizer,
        {
            "instruction": "explain hierarchy prior context",
            "input": "prior context prior context",
            "output": "final answer reply",
        },
        8,
        alpaca_mode=True,
        min_response_tokens=2,
        include_audit=True,
        audit_source_default="fixture",
    )
    audit = accepted["_audit"]
    assert audit["accepted"] is True
    assert audit["truncated"] is True
    assert audit["original_response_tokens"] >= audit["retained_response_tokens"] >= 2
    assert audit["retained_tokens"] == len(accepted["input_ids"])
    assert audit["weighted_tokens"] > 0


def test_fast_batch_tokenization_matches_scalar_for_text_generic_and_kayla_rows():
    tokenizer = _make_fast_tokenizer()

    text_rows = [{"text": "hello world"}, {"text": "final answer"}, {"text": "   "}]
    text_args = (8, False, "text", None, None, False, True, 1.0, 1.0, 1.0, 0, 1, True)
    expected = [process_text_sample(tokenizer, row, *text_args) for row in text_rows]
    actual = process_text_samples_batch(tokenizer, text_rows, *text_args)
    _assert_batches_equal(actual, expected)

    generic_rows = [
        {"question": "hello question", "input": "prior context", "answer": "final answer"},
        {"question": "question", "answer": "reply"},
    ]
    generic_args = (16, False, None, "question", "answer", False, True, 1.0, 1.0, 1.0, 0, 1, True)
    expected = [process_text_sample(tokenizer, row, *generic_args) for row in generic_rows]
    actual = process_text_samples_batch(tokenizer, generic_rows, *generic_args)
    _assert_batches_equal(actual, expected)

    kayla_rows = [
        {
            "instruction": "question",
            "output": "final answer",
            "feelings": "calm",
            "thought-process": "thought",
        }
    ]
    kayla_args = (32, True, None, "instruction", "output", False, True, 1.0, 1.0, 1.0, 0, 1, True)
    tokenizer.batch_calls.clear()
    expected = [process_text_sample(tokenizer, row, *kayla_args) for row in kayla_rows]
    actual = process_text_samples_batch(tokenizer, kayla_rows, *kayla_args)
    _assert_batches_equal(actual, expected)
    false_special_texts = [
        text
        for texts, add_special_tokens in tokenizer.batch_calls
        if not add_special_tokens
        for text in texts
    ]
    assert "### Thought Process:\nthought\n\n" in false_special_texts
    assert "### Response:\nfinal answer" in false_special_texts


class _CustomTokenizer:
    eos_token_id = 91

    def __init__(self, is_fast):
        self.is_fast = is_fast
        self.encode_calls = 0
        self.batch_attempts = 0

    def encode(self, text, add_special_tokens=True):
        self.encode_calls += 1
        ids = [3 + (ord(char) % 71) for char in str(text)]
        return ([1] + ids + [2]) if add_special_tokens else ids

    def __call__(self, *_args, **_kwargs):
        self.batch_attempts += 1
        raise AssertionError("custom tokenizer must use scalar fallback")


def test_slow_and_custom_tokenizers_use_scalar_fallback():
    samples = [
        {"instruction": "hello", "output": "final answer"},
        {"instruction": "question", "output": "reply"},
    ]
    args = (64, False, None, None, None, True, True, 1.0, 1.0, 1.0, 0, 1, True)
    for claims_fast in (False, True):
        expected_tokenizer = _CustomTokenizer(is_fast=claims_fast)
        expected = [process_text_sample(expected_tokenizer, row, *args) for row in samples]
        actual_tokenizer = _CustomTokenizer(is_fast=claims_fast)
        actual = process_text_samples_batch(actual_tokenizer, samples, *args)
        _assert_batches_equal(actual, expected)
        assert actual_tokenizer.batch_attempts == 0
        assert actual_tokenizer.encode_calls == expected_tokenizer.encode_calls


def test_fast_tokenizer_batch_failure_or_malformed_output_falls_back_to_scalar():
    samples = [
        {"instruction": "hello", "output": "final answer"},
        {"instruction": "question", "output": "reply"},
    ]
    args = (64, False, None, None, None, True, False, 0.25, 1.0, 1.5, 2, 1, True)
    reference_tokenizer = _make_fast_tokenizer()
    expected = [process_text_sample(reference_tokenizer, row, *args) for row in samples]
    for tokenizer_cls in (_FailingFastTokenizer, _MalformedFastTokenizer):
        tokenizer = _make_fast_tokenizer(tokenizer_cls)
        actual = process_text_samples_batch(tokenizer, samples, *args)
        _assert_batches_equal(actual, expected)
        assert tokenizer.batch_attempts == 1


class _BatchIndexDataset:
    def __init__(self, rows):
        self.rows = rows
        self.batch_fetches = 0
        self.scalar_fetches = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, list):
            self.batch_fetches += 1
            keys = self.rows[0].keys()
            return {key: [self.rows[row_idx][key] for row_idx in index] for key in keys}
        self.scalar_fetches += 1
        return self.rows[index]


def test_hf_map_dataset_dataloader_uses_batched_arrow_fetch_and_tokenization():
    rows = [
        {"instruction": "hello", "input": "prior context", "output": "final answer"},
        {"instruction": "question", "input": "", "output": "reply"},
        {"instruction": "explain hierarchy", "input": "prior", "output": "final"},
    ]
    source = _BatchIndexDataset(rows)
    tokenizer = _make_fast_tokenizer()
    dataset = HuggingFaceMapStyleDataset(
        source,
        tokenizer,
        max_length=32,
        alpaca_mode=True,
        train_prompt_tokens=False,
        prompt_loss_weight=0.1,
        response_boundary_loss_weight=2.0,
        response_boundary_tokens=2,
        precompute_rosa=True,
        rosa_vocab_size=len(tokenizer),
        rosa_chunk_size=8,
        rosa_max_context=16,
    )
    loader = create_map_style_dataloader(
        dataset,
        batch_size=3,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        shuffle=False,
        use_length_bucketing=False,
        device=torch.device("cpu"),
    )

    actual_batch = next(iter(loader))

    reference_tokenizer = _make_fast_tokenizer()
    reference_dataset = HuggingFaceMapStyleDataset(
        _BatchIndexDataset(rows),
        reference_tokenizer,
        max_length=32,
        alpaca_mode=True,
        train_prompt_tokens=False,
        prompt_loss_weight=0.1,
        response_boundary_loss_weight=2.0,
        response_boundary_tokens=2,
        precompute_rosa=True,
        rosa_vocab_size=len(reference_tokenizer),
        rosa_chunk_size=8,
        rosa_max_context=16,
    )
    expected_batch = _collate_training_batch(
        [reference_dataset[idx] for idx in range(len(rows))],
        reference_tokenizer.pad_token_id,
    )

    assert source.batch_fetches == 1
    assert source.scalar_fetches == 0
    assert len(tokenizer.batch_calls) == 2
    assert actual_batch.keys() == expected_batch.keys()
    for key in expected_batch:
        if torch.is_tensor(expected_batch[key]):
            assert torch.equal(actual_batch[key], expected_batch[key]), key
        else:
            assert actual_batch[key] == expected_batch[key], key
