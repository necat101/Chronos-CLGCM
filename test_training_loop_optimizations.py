from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hierarchos.training.trainer import (
    CUDABatchPrefetcher,
    _batch_tensor_to_device,
    _clip_gradients_and_check,
    _prepare_ltm_update_gradients,
    train_step,
)


class _MetricParityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_kwargs = []
        self.config = SimpleNamespace(
            vocab_size=8,
            cpu_chunked_lm_loss=False,
            cuda_chunked_lm_loss=False,
        )

    def reset_memory(self):
        return None

    def forward(self, **_kwargs):
        self.forward_kwargs.append(dict(_kwargs))
        zero = self.weight * 0.0
        return {
            "loss": self.weight,
            "ponder_cost": zero,
            "commitment_cost": zero,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


def _step_args():
    return SimpleNamespace(
        amp=False,
        training_chunk_size=2,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode="read-only",
    )


def test_collect_metrics_flag_does_not_change_training_update():
    with_metrics = _MetricParityModel()
    without_metrics = _MetricParityModel()
    without_metrics.load_state_dict(with_metrics.state_dict())
    optimizer_a = torch.optim.SGD(with_metrics.parameters(), lr=0.1)
    optimizer_b = torch.optim.SGD(without_metrics.parameters(), lr=0.1)
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }
    empty_state = (None, None, None, None, None, None)

    outputs_a, states_a = train_step(
        with_metrics,
        batch,
        optimizer_a,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=_step_args(),
        running_states=empty_state,
        collect_metrics=True,
        force_optimizer_step=True,
    )
    outputs_b, states_b = train_step(
        without_metrics,
        batch,
        optimizer_b,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=_step_args(),
        running_states=empty_state,
        collect_metrics=False,
        force_optimizer_step=True,
    )

    assert outputs_a is not None
    assert outputs_b is None
    assert torch.equal(with_metrics.weight, without_metrics.weight)
    assert states_a == states_b


def test_train_step_disables_duplicate_history_for_cached_rosa_chunks():
    model = _MetricParityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "rosa_ids": torch.tensor([[8, 1, 1, 2]], dtype=torch.long),
        "rosa_ids_context_mode": "bounded-segment-v1",
    }

    train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=_step_args(),
        running_states=(None, None, None, None, None, None),
        collect_metrics=False,
        force_optimizer_step=True,
    )

    assert model.forward_kwargs
    assert all(
        call["advance_cached_rosa_history"] is False
        for call in model.forward_kwargs
    )


def test_train_step_prevalidates_cpu_padding_once_and_passes_chunk_geometry():
    model = _MetricParityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 0]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [[1, 2, 3, -100], [4, 5, -100, -100]],
            dtype=torch.long,
        ),
    }

    train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=_step_args(),
        running_states=(None, None, None, None, None, None),
        collect_metrics=False,
        force_optimizer_step=True,
    )

    assert [
        call["_prevalidated_mask_metadata"]
        for call in model.forward_kwargs
    ] == [(False, 2), (True, 0)]


def test_train_step_host_prevalidation_rejects_non_right_padding_before_forward():
    model = _MetricParityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "input_ids": torch.tensor([[1, 0, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 0, 1]], dtype=torch.long),
        "labels": torch.tensor([[1, -100, 2]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="right padding only"):
        train_step(
            model,
            batch,
            optimizer,
            scaler=None,
            accumulation_steps=1,
            step=0,
            args=_step_args(),
            running_states=(None, None, None, None, None, None),
            collect_metrics=False,
            force_optimizer_step=True,
        )

    assert model.forward_kwargs == []


def test_prefetched_tensor_replays_cpu_padding_and_trimming_exactly():
    raw = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    batch = {"_cuda_prefetched_tensors": {"input_ids": raw.clone()}}
    padded_cpu = torch.tensor(
        [[1, 2, 3, 99, 99], [4, 5, 6, 99, 99]],
        dtype=torch.long,
    )
    padded = _batch_tensor_to_device(
        batch,
        "input_ids",
        padded_cpu,
        torch.device("cpu"),
        pad_value=99,
    )
    assert torch.equal(padded, padded_cpu)

    trimmed_cpu = raw[:, :2].contiguous()
    trimmed = _batch_tensor_to_device(
        batch,
        "input_ids",
        trimmed_cpu,
        torch.device("cpu"),
        pad_value=99,
    )
    assert torch.equal(trimmed, trimmed_cpu)
    assert trimmed.is_contiguous()


def test_foreach_gradient_clip_matches_global_l2_rule():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    original_grads = []
    for index, parameter in enumerate(model.parameters(), 1):
        parameter.grad = torch.full_like(parameter, float(index))
        original_grads.append(parameter.grad.clone())
    expected_norm = torch.linalg.vector_norm(
        torch.cat([gradient.reshape(-1).float() for gradient in original_grads]),
        ord=2,
    )
    expected_scale = min(1.0, 0.75 / (float(expected_norm) + 1e-6))

    ok, total_norm = _clip_gradients_and_check(model, max_norm=0.75)

    assert ok is True
    assert torch.allclose(total_norm.float(), expected_norm.float())
    for parameter, original in zip(model.parameters(), original_grads):
        assert torch.allclose(parameter.grad, original * expected_scale)


def test_ltm_owned_inplace_path_matches_nonmutating_default():
    source = torch.linspace(-2.0, 2.0, 64).reshape(2, 4, 8)
    source_before = source.clone()
    reference = _prepare_ltm_update_gradients(source, 0.75)
    assert torch.equal(source, source_before)

    owned = source.clone()
    pointer = owned.data_ptr()
    optimized = _prepare_ltm_update_gradients(owned, 0.75, inplace=True)
    assert optimized.data_ptr() == pointer
    assert torch.allclose(optimized, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_batch_prefetcher_preserves_values_and_device():
    source = torch.arange(12, dtype=torch.long).reshape(3, 4).pin_memory()
    batches = [{"input_ids": source, "labels": source.clone().pin_memory()}]
    prefetched = next(iter(CUDABatchPrefetcher(batches, torch.device("cuda"))))
    device_tensors = prefetched["_cuda_prefetched_tensors"]
    assert device_tensors["input_ids"].device.type == "cuda"
    assert torch.equal(device_tensors["input_ids"].cpu(), source)
    assert torch.equal(device_tensors["labels"].cpu(), source)
