from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hierarchos.inference.chat import (
    bound_chat_turn_history,
    build_effective_chat_input_context,
    load_hierarchical_chat_state,
    resolve_effective_carry_chat_state,
    save_hierarchical_chat_state,
)


class _TinyModel(nn.Module):
    pass


def _config():
    return SimpleNamespace(
        context_dim=8,
        h_hidden=8,
        l_hidden=8,
        h_stride=1,
        max_h_steps=1,
        max_l_steps=1,
        vocab_size=32,
        rwkv_head_size=1,
        training_chunk_size=256,
        full_sample_bptt=False,
        inference_recurrence_mode="tbptt",
    )


def _save(path, *, history=None, chunk_size=256, carry=True, total_tokens=513):
    save_hierarchical_chat_state(
        path,
        config=_config(),
        model=_TinyModel(),
        model_path="model.pt",
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        total_tokens_generated=total_tokens,
        chat_prefill_chunk_size=chunk_size,
        chat_input_history_turns=2,
        chat_input_history_chars=64,
        carry_chat_state=carry,
        chat_turn_history=history or [],
    )


def test_resume_always_enables_carry_and_carried_state_skips_text_replay():
    history = ["User: one\nAssistant: first"]

    assert resolve_effective_carry_chat_state(False, "state.pt") is True
    assert resolve_effective_carry_chat_state(False, None) is False
    assert build_effective_chat_input_context(
        history,
        max_turns=2,
        max_chars=100,
        carry_chat_state=True,
    ) == ""
    assert "User: one" in build_effective_chat_input_context(
        history,
        max_turns=2,
        max_chars=100,
        carry_chat_state=False,
    )


def test_chat_continuity_geometry_and_bounded_history_round_trip(tmp_path):
    path = tmp_path / "continuity.pt"
    history = [
        "User: old\nAssistant: old answer",
        "User: middle\nAssistant: middle answer",
        "User: newest\nAssistant: newest answer",
    ]
    expected_history = bound_chat_turn_history(history, max_turns=2, max_chars=64)
    _save(path, history=history)

    restored = load_hierarchical_chat_state(
        path,
        config=_config(),
        device="cpu",
        model=_TinyModel(),
        expected_chat_prefill_chunk_size=256,
    )

    assert restored["chat_prefill_chunk_size"] == 256
    assert restored["chat_continuity"]["absolute_chunk_phase"] == 1
    assert restored["carry_chat_state"] is True
    assert restored["chat_input_history_turns"] == 2
    assert restored["chat_input_history_chars"] == 64
    assert restored["chat_turn_history"] == expected_history
    assert len("\n\n".join(restored["chat_turn_history"])) <= 64


def test_explicit_resume_prefill_geometry_conflict_fails_closed(tmp_path):
    path = tmp_path / "continuity.pt"
    _save(path)

    with pytest.raises(RuntimeError, match="prefill chunk geometry mismatch"):
        load_hierarchical_chat_state(
            path,
            config=_config(),
            device="cpu",
            model=_TinyModel(),
            expected_chat_prefill_chunk_size=128,
        )


@pytest.mark.parametrize("corruption", ["phase", "history"])
def test_corrupt_chat_continuity_metadata_is_rejected(tmp_path, corruption):
    path = tmp_path / f"corrupt-{corruption}.pt"
    _save(path, history=["User: one\nAssistant: answer"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if corruption == "phase":
        payload["chat_continuity"]["absolute_chunk_phase"] = 2
        expected = "chunk phase is inconsistent"
    else:
        payload["chat_continuity"]["turn_history"].append("x" * 65)
        expected = "turn_history exceeds"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=expected):
        load_hierarchical_chat_state(
            path,
            config=_config(),
            device="cpu",
            model=_TinyModel(),
        )


def test_pre_continuity_version_four_state_remains_loadable(tmp_path):
    path = tmp_path / "old-v4.pt"
    _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["version"] == 4
    del payload["chat_continuity"]
    payload["runtime_identity"]["version"] = 1
    torch.save(payload, path)

    restored = load_hierarchical_chat_state(
        path,
        config=_config(),
        device="cpu",
        model=_TinyModel(),
    )
    assert restored["chat_continuity"] is None
    assert restored["chat_turn_history"] == []

    with pytest.raises(RuntimeError, match="explicit resume-time chunk override"):
        load_hierarchical_chat_state(
            path,
            config=_config(),
            device="cpu",
            model=_TinyModel(),
            expected_chat_prefill_chunk_size=128,
        )


def test_current_cli_state_cannot_strip_required_continuity_metadata(tmp_path):
    path = tmp_path / "stripped-current.pt"
    _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["runtime_identity"]["version"] == 2
    del payload["chat_continuity"]
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="missing required chat_continuity"):
        load_hierarchical_chat_state(
            path,
            config=_config(),
            device="cpu",
            model=_TinyModel(),
        )
