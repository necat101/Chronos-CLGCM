from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import hierarchos.inference.chat as chat_module
from hierarchos.evaluation.lm_eval_wrapper import _score_target_logits
from hierarchos.inference.chat import (
    _queue_chat_sampled_token,
    sample_next_token,
    should_stop_generation_from_uncertainty,
)
from hierarchos.models.core import WorkerLoop


def _legacy_filtered_probabilities(logits, *, temperature, top_k, top_p):
    scores = logits.float().clone()
    scores.div_(temperature)
    threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
    scores.masked_fill_(scores < threshold, -torch.inf)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
    scores.masked_fill_(remove, -torch.inf)
    return F.softmax(scores, dim=-1)


def test_disabled_uncertainty_guards_skip_vocabulary_softmax(monkeypatch):
    def unexpected_softmax(*_args, **_kwargs):
        raise AssertionError("disabled uncertainty guards must not normalize logits")

    monkeypatch.setattr(chat_module.F, "softmax", unexpected_softmax)
    settings = SimpleNamespace(
        entropy_stop_threshold=0.0,
        entropy_stop_min_tokens=3,
        entropy_stop_top_prob=0.05,
        eos_stop_prob=0.0,
    )
    tokenizer = SimpleNamespace(eos_token_id=2)
    assert not should_stop_generation_from_uncertainty(
        torch.randn(1, 128),
        [1, 2, 3],
        tokenizer,
        settings,
    )


def test_top_k_nucleus_sampling_normalizes_only_candidates_with_same_distribution(
    monkeypatch,
):
    logits = torch.tensor(
        [
            [9.0, 7.5, 6.0, 4.5, 3.0, 1.5, 0.0, -1.5],
            [-2.0, 0.0, 2.0, 4.0, 8.0, 6.0, -4.0, 10.0],
        ]
    )
    temperature = 0.8
    top_k = 4
    top_p = 0.72
    expected_full = _legacy_filtered_probabilities(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    candidate_indices = torch.topk(
        logits.float() / temperature,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    expected_candidates = expected_full.gather(1, candidate_indices)
    captured = {}

    def capture_multinomial(probs, num_samples):
        captured["probs"] = probs.detach().clone()
        assert num_samples == 1
        return torch.zeros((probs.shape[0], 1), dtype=torch.long, device=probs.device)

    def unexpected_full_sort(*_args, **_kwargs):
        raise AssertionError("top-k nucleus path must not sort the full vocabulary")

    monkeypatch.setattr(chat_module.torch, "multinomial", capture_multinomial)
    monkeypatch.setattr(chat_module.torch, "sort", unexpected_full_sort)
    sampled = sample_next_token(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    assert captured["probs"].shape == (2, top_k)
    torch.testing.assert_close(captured["probs"], expected_candidates, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(sampled, candidate_indices[:, :1], rtol=0, atol=0)


def test_top_k_sampling_promotes_only_candidates_and_keeps_logits_immutable(
    monkeypatch,
):
    logits = torch.tensor(
        [[8.0, -3.0, 6.0, 1.0, 4.0, -1.0]],
        dtype=torch.bfloat16,
    )
    original = logits.clone()
    real_topk = torch.topk
    observed = {}

    def capture_topk(values, *args, **kwargs):
        observed["topk_input_dtype"] = values.dtype
        return real_topk(values, *args, **kwargs)

    def capture_multinomial(probs, num_samples):
        observed["prob_dtype"] = probs.dtype
        observed["prob_shape"] = tuple(probs.shape)
        return torch.zeros((probs.shape[0], num_samples), dtype=torch.long)

    monkeypatch.setattr(chat_module.torch, "topk", capture_topk)
    monkeypatch.setattr(chat_module.torch, "multinomial", capture_multinomial)

    sample_next_token(logits, temperature=0.7, top_k=3, top_p=1.0)

    assert observed == {
        "topk_input_dtype": torch.bfloat16,
        "prob_dtype": torch.float32,
        "prob_shape": (1, 3),
    }
    torch.testing.assert_close(logits, original, rtol=0, atol=0)


def test_sample_queue_materializes_the_device_scalar_once():
    class CountingSample:
        def __init__(self, value):
            self.value = value
            self.item_calls = 0

        def item(self):
            self.item_calls += 1
            return self.value

    sample = CountingSample(7)
    current, sampled_token, pending, terminal = _queue_chat_sampled_token(sample, 7)
    assert current is sample
    assert sampled_token == 7
    assert pending is True
    assert terminal is True
    assert sample.item_calls == 1


def test_internal_sampler_can_reuse_the_model_finite_audit(monkeypatch):
    logits = torch.tensor([[4.0, 1.0, -2.0]])

    def unexpected_finite_scan(*_args, **_kwargs):
        raise AssertionError("prevalidated model logits must not be rescanned")

    monkeypatch.setattr(chat_module.torch, "isfinite", unexpected_finite_scan)
    sampled = sample_next_token(
        logits,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        _logits_prevalidated=True,
    )
    assert sampled.item() == 0


class _CountingStableRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.inputs = []

    def forward(self, x, state, timestep=None, deepemb_vec=None):
        self.calls += 1
        self.inputs.append(x.detach().clone())
        return x, state


class _CountingLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return super().forward(value)


def _counting_worker(*, compile_requested, static_loop, max_l_steps=4):
    config = SimpleNamespace(
        max_l_steps=max_l_steps,
        l_conv_atol=0.2,
        commitment_threshold=0.0,
        commitment_cost_mode="mean-square",
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        drift_delta_scale=1.0,
        activation_clamp=100.0,
        compile=compile_requested,
        compile_static_worker_loop=static_loop,
        inference_logit_parity=True,
    )
    rnn = _CountingStableRNN()
    rnn.train()
    input_projection = _CountingLinear(4, 2, bias=False)
    drift_projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        input_projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                ]
            )
        )
        drift_projection.weight.zero_()
    loop = WorkerLoop(
        config,
        rnn,
        input_projection,
        drift_projection,
        nn.Identity(),
    )
    return loop, rnn, input_projection


def _run_counting_worker(loop):
    return loop(
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2),
    )


def test_checkpoint_compile_flags_do_not_force_fixed_eager_worker_iterations():
    loop, rnn, input_projection = _counting_worker(
        compile_requested=True,
        static_loop=True,
    )
    outputs = _run_counting_worker(loop)

    # One converged exploratory step plus the required committed transition.
    assert rnn.calls == 2
    # Initial projection plus the one accepted candidate; no duplicate commit projection.
    assert input_projection.calls == 2
    expected_commit_input = F.linear(
        torch.cat([torch.ones(1, 2), outputs[3]], dim=-1),
        input_projection.weight,
        input_projection.bias,
    )
    torch.testing.assert_close(rnn.inputs[-1], expected_commit_input, rtol=0, atol=0)


def test_explicit_eager_static_worker_keeps_fixed_shape_without_commit_reprojection():
    loop, rnn, input_projection = _counting_worker(
        compile_requested=False,
        static_loop=True,
    )
    _run_counting_worker(loop)

    assert rnn.calls == loop.max_l_steps + 1
    assert input_projection.calls == loop.max_l_steps + 1


def _rowwise_target_score_reference(chunk_logits, chunk_targets):
    score_sums = torch.zeros(chunk_logits.shape[0], dtype=torch.float32)
    score_is_greedy = torch.ones(chunk_logits.shape[0], dtype=torch.bool)
    for row in range(chunk_logits.shape[0]):
        active = chunk_targets[row] != -100
        if not bool(active.any().item()):
            continue
        active_logits = chunk_logits[row, active, :].float()
        active_targets = chunk_targets[row, active]
        target_log_probs = F.log_softmax(active_logits, dim=-1).gather(
            dim=-1,
            index=active_targets.unsqueeze(-1),
        ).squeeze(-1)
        score_sums[row].add_(target_log_probs.sum())
        score_is_greedy[row] = (active_logits.argmax(dim=-1) == active_targets).all()
    return score_sums, score_is_greedy


def test_vectorized_lm_eval_scoring_matches_rowwise_reference_and_ignores_padding():
    torch.manual_seed(73)
    logits = torch.randn(3, 5, 11)
    targets = torch.tensor(
        [
            [-100, 2, 4, -100, 7],
            [-100, -100, -100, -100, -100],
            [1, -100, 3, 5, -100],
        ],
        dtype=torch.long,
    )
    # Unscored padding/context logits must remain irrelevant, as in the old loop.
    logits[1].fill_(float("nan"))
    logits[0, 0].fill_(float("nan"))

    expected_scores, expected_greedy = _rowwise_target_score_reference(logits, targets)
    actual_scores, actual_greedy = _score_target_logits(logits, targets)

    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_greedy, expected_greedy, rtol=0, atol=0)


def test_vectorized_lm_eval_scoring_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="batch/time dimensions"):
        _score_target_logits(torch.randn(2, 3, 5), torch.full((2, 2), -100))
