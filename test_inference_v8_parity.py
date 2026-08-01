from types import SimpleNamespace
import tempfile

import pytest
import torch
import torch.nn as nn

import hierarchos.utils.rosa as rosa_module
from hierarchos.inference.chat import (
    _queue_chat_sampled_token,
    _chat_ltm_state_from_rosa_context,
    load_hierarchical_chat_state,
    save_hierarchical_chat_state,
)
from hierarchos.inference.chat_state import clear_ltm_working_memory
from hierarchos.utils.rosa import ROSA, ROSAState, precompute_rosa_ids_for_chunks, rosa_async_pipeline, rosa_single


class _FakeLTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vals = nn.Parameter(torch.ones(4, 3))
        self.register_buffer("fast_vals", torch.ones(4, 3))
        self.register_buffer("_mom_vals", torch.full((4, 3), 2.0))
        self.register_buffer("timestamps", torch.arange(4, dtype=torch.float32))
        self.register_buffer("sources", torch.arange(4, dtype=torch.long))


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = _FakeLTM()


class _IdentityTokenizer:
    special_tokens_map = {"eos_token": "<eos>"}
    name_or_path = "fixture"

    def __init__(self, behavior):
        self.behavior = behavior
        self.backend_tokenizer = self

    def __len__(self):
        return 3

    def get_vocab(self):
        return {"<eos>": 0, "a": 1, "b": 2}

    def to_str(self):
        return '{"normalizer":{"type":"' + self.behavior + '"}}'


class _LegacyQuantizedCell:
    def __init__(self, hidden):
        self.n_embd = hidden


class _LegacyQuantizedLikeModel(_FakeModel):
    def __init__(self):
        super().__init__()
        self.h_rnn = _LegacyQuantizedCell(3)
        self.l_rnn = _LegacyQuantizedCell(3)


class _UnsupportedSerializedObject:
    pass


def _fake_config():
    return SimpleNamespace(
        context_dim=3,
        h_hidden=3,
        l_hidden=3,
        h_stride=1,
        max_h_steps=1,
        max_l_steps=1,
        vocab_size=100,
        rwkv_head_size=1,
    )


def test_chat_state_preserves_full_rosa_history_and_v8_state():
    model = _FakeModel()
    config = _fake_config()
    past_tokens = (
        torch.arange(2048, dtype=torch.long) % config.vocab_size
    ).reshape(1, 2048)
    _, rosa_state = rosa_single(past_tokens[0].tolist())
    ltm_state = (
        model.ltm.fast_vals,
        model.ltm._mom_vals,
        past_tokens,
        [rosa_state],
        model.ltm.timestamps,
        model.ltm.sources,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    path = tmp.name
    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path="model.pt",
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        ltm_state=ltm_state,
        total_tokens_generated=2048,
    )
    restored = load_hierarchical_chat_state(path, config=config, device=torch.device("cpu"), model=model)

    assert restored["rosa_past_tokens"].shape == (1, 2048)
    assert torch.equal(restored["rosa_past_tokens"], past_tokens)
    assert restored["rosa_states"][0].tokens == past_tokens[0].tolist()

    rehydrated = _chat_ltm_state_from_rosa_context(
        model,
        restored["rosa_past_tokens"],
        restored["rosa_states"],
    )
    assert len(rehydrated) == 7
    assert torch.equal(rehydrated[2], past_tokens)
    assert rehydrated[3][0].tokens == past_tokens[0].tolist()
    assert rehydrated[6] is None


def test_chat_state_materializes_history_from_authoritative_rosa_state(tmp_path):
    model = _FakeModel()
    config = _fake_config()
    tokens = [1, 2, 3, 1, 2, 4]
    _, rosa_state = rosa_single(tokens)
    ltm_state = (
        model.ltm.fast_vals,
        model.ltm._mom_vals,
        None,
        [rosa_state],
        model.ltm.timestamps,
        model.ltm.sources,
    )
    path = tmp_path / "authoritative-rosa-state.pt"

    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path="model.pt",
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        ltm_state=ltm_state,
        total_tokens_generated=len(tokens),
    )
    restored = load_hierarchical_chat_state(
        path,
        config=config,
        device=torch.device("cpu"),
        model=model,
    )

    assert torch.equal(
        restored["rosa_past_tokens"],
        torch.tensor([tokens], dtype=torch.long),
    )
    assert restored["rosa_states"][0].tokens == tokens


def test_chat_state_binds_checkpoint_and_tokenizer_behavior(tmp_path):
    config = _fake_config()
    model = _FakeModel()
    other_model = _FakeModel()
    model._hierarchos_checkpoint_metadata = {
        "source_weights_path": str(tmp_path / "model-a.pt"),
    }
    other_model._hierarchos_checkpoint_metadata = {
        "source_weights_path": str(tmp_path / "model-b.pt"),
    }
    for name, digest in (("model-a.pt", "a" * 64), ("model-b.pt", "b" * 64)):
        weights = tmp_path / name
        weights.write_bytes(name.encode("utf-8"))
        (tmp_path / f"{name}.sha256").write_text(
            f"{digest}  {name}\n",
            encoding="utf-8",
        )
    tokenizer = _IdentityTokenizer("NFC")
    path = tmp_path / "bound-chat-state.pt"

    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path=str(tmp_path),
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        total_tokens_generated=0,
        tokenizer=tokenizer,
    )
    load_hierarchical_chat_state(
        path,
        config=config,
        device="cpu",
        model=model,
        tokenizer=tokenizer,
    )

    with pytest.raises(RuntimeError, match="different model weights"):
        load_hierarchical_chat_state(
            path,
            config=config,
            device="cpu",
            model=other_model,
            tokenizer=tokenizer,
        )
    with pytest.raises(RuntimeError, match="different model weights"):
        load_hierarchical_chat_state(
            path,
            config=config,
            device="cpu",
            model=model,
            tokenizer=_IdentityTokenizer("NFKC"),
        )


def test_current_chat_state_accepts_exact_legacy_quantized_cell_shape():
    model = _LegacyQuantizedLikeModel()
    config = _fake_config()
    h_state = torch.zeros(1, 3, 5)
    l_state = torch.zeros(1, 3, 5)
    h_state[:, :, 3] = -1e30
    l_state[:, :, 3] = -1e30

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    path = tmp.name
    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path="model.pt",
        h_state=h_state,
        l_state=l_state,
        prev_context=torch.zeros(1, 3),
        target_context=torch.zeros(1, 3),
        drift_state=torch.zeros(1, 3),
        total_tokens_generated=3,
    )
    restored = load_hierarchical_chat_state(
        path,
        config=config,
        device=torch.device("cpu"),
        model=model,
    )
    assert torch.equal(restored["h_state"], h_state)
    assert torch.equal(restored["l_state"], l_state)


def test_chat_eos_remains_queued_for_recurrent_state_consumption():
    eos = torch.tensor([[7]], dtype=torch.long)
    current, sampled_token, pending, terminal = _queue_chat_sampled_token(eos, 7)
    assert current is eos
    assert sampled_token == 7
    assert pending is True
    assert terminal is True

    ordinary = torch.tensor([[6]], dtype=torch.long)
    current, sampled_token, pending, terminal = _queue_chat_sampled_token(ordinary, 7)
    assert current is ordinary
    assert sampled_token == 6
    assert pending is True
    assert terminal is False


def test_chat_state_loader_uses_safe_weights_only_deserialization(tmp_path):
    path = tmp_path / "unsupported-chat-state.pt"
    torch.save(
        {
            "kind": "hierarchos_chat_runtime_state",
            "version": 4,
            "unsupported": _UnsupportedSerializedObject(),
        },
        path,
    )
    try:
        load_hierarchical_chat_state(
            path,
            config=_fake_config(),
            device=torch.device("cpu"),
            model=_LegacyQuantizedLikeModel(),
        )
    except Exception as exc:
        assert "Weights only load failed" in str(exc)
    else:
        raise AssertionError("Unsafe chat-state global was accepted")


def test_current_chat_state_rejects_nonfinite_context_and_invalid_offset(tmp_path):
    model = _LegacyQuantizedLikeModel()
    config = _fake_config()
    path = tmp_path / "chat-state.pt"
    h_state = torch.zeros(1, 3, 5)
    l_state = torch.zeros(1, 3, 5)
    h_state[:, :, 3] = -1e30
    l_state[:, :, 3] = -1e30
    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path="model.pt",
        h_state=h_state,
        l_state=l_state,
        prev_context=torch.zeros(1, 3),
        target_context=torch.zeros(1, 3),
        drift_state=torch.zeros(1, 3),
        total_tokens_generated=3,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["prev_context"][0, 0] = float("nan")
    torch.save(payload, path)
    try:
        load_hierarchical_chat_state(
            path,
            config=config,
            device=torch.device("cpu"),
            model=model,
        )
    except RuntimeError as exc:
        assert "prev_context contains non-finite" in str(exc)
    else:
        raise AssertionError("Non-finite context state was accepted")

    payload["prev_context"].zero_()
    payload["total_tokens_generated"] = -1
    torch.save(payload, path)
    try:
        load_hierarchical_chat_state(
            path,
            config=config,
            device=torch.device("cpu"),
            model=model,
        )
    except RuntimeError as exc:
        assert "nonnegative integer" in str(exc)
    else:
        raise AssertionError("Negative chat token offset was accepted")


def test_current_chat_state_rejects_cyclic_rosa_automaton(tmp_path):
    model = _FakeModel()
    config = _fake_config()
    tokens = [1, 2, 1, 3]
    _, rosa_state = rosa_single(tokens)
    path = tmp_path / "cyclic-rosa-state.pt"

    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path="model.pt",
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        ltm_state=(
            model.ltm.fast_vals,
            model.ltm._mom_vals,
            torch.tensor([tokens], dtype=torch.long),
            [rosa_state],
            model.ltm.timestamps,
            model.ltm.sources,
        ),
        total_tokens_generated=len(tokens),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rosa_states"][0].suffix_links[1] = 1
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="invalid suffix link"):
        load_hierarchical_chat_state(
            path,
            config=config,
            device=torch.device("cpu"),
            model=model,
        )


def test_clear_ltm_working_memory_zeros_transient_inference_buffers():
    model = _FakeModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)

    assert clear_ltm_working_memory(model) is True
    assert torch.count_nonzero(model.ltm.fast_vals).item() == 0
    assert torch.count_nonzero(model.ltm._mom_vals).item() == 0
    assert torch.count_nonzero(model.ltm.timestamps).item() == 0
    assert torch.equal(model.ltm.sources, torch.zeros_like(model.ltm.sources))
    assert torch.count_nonzero(model.ltm.vals).item() > 0


def test_rosa_async_pipeline_keeps_history_only_in_authoritative_state():
    past_tokens = torch.arange(20, dtype=torch.long).reshape(1, 20)
    input_ids = torch.tensor([[20, 21]], dtype=torch.long)
    expected = torch.cat([past_tokens, input_ids], dim=1)

    finalize = rosa_async_pipeline(
        input_ids,
        past_tokens,
        rosa_states=None,
        vocab_size=100,
        device=torch.device("cpu"),
        rosa_max_ctx=8,
    )
    rosa_ids, new_past_tokens, new_states = finalize()

    assert rosa_ids.shape == input_ids.shape
    assert new_past_tokens is None
    assert new_states[0].tokens == expected.squeeze(0).tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_single_row_rosa_uses_pipeline_worker(monkeypatch):
    original_pool = rosa_module._get_pipeline_pool()
    submitted = {"count": 0}

    class _PoolSpy:
        def submit(self, fn, *args, **kwargs):
            submitted["count"] += 1
            return original_pool.submit(fn, *args, **kwargs)

    monkeypatch.setattr(rosa_module, "_get_pipeline_pool", lambda: _PoolSpy())
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cuda")
    finalize = rosa_async_pipeline(
        input_ids,
        past_tokens=None,
        rosa_states=None,
        vocab_size=100,
        device=input_ids.device,
    )
    rosa_ids, _, _ = finalize()
    assert submitted["count"] == 1
    assert rosa_ids.device.type == "cuda"


def test_rosa_async_pipeline_extends_uncapped_state_incrementally():
    past_tokens = torch.arange(20, dtype=torch.long).reshape(1, 20)
    input_ids = torch.tensor([[20, 21]], dtype=torch.long)
    state = ROSAState.new()
    rosa_single(past_tokens.squeeze(0).tolist(), state)

    finalize = rosa_async_pipeline(
        input_ids,
        past_tokens,
        rosa_states=[state],
        vocab_size=100,
        device=torch.device("cpu"),
        rosa_max_ctx=8,
    )
    _, new_past_tokens, new_states = finalize()

    assert new_past_tokens is None
    assert new_states[0].tokens == torch.cat(
        [past_tokens, input_ids], dim=1
    ).squeeze(0).tolist()


def test_rosa_precompute_matches_full_history_across_chunks():
    tokens = [1, 2, 3, 1, 2, 4, 1, 2, 3, 5]
    vocab_size = 100
    expected = [vocab_size if pred == -1 else pred for pred in ROSA(tokens)]

    actual = precompute_rosa_ids_for_chunks(
        tokens,
        vocab_size=vocab_size,
        chunk_size=3,
        rosa_max_ctx=2,
    )

    assert actual == expected
