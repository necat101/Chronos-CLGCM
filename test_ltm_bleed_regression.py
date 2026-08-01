import pytest
import torch
import torch.nn.functional as F

from hierarchos import AttrDict, HierarchosCore, LTMModule
from hierarchos.inference.chat import extract_correction_text, parse_temperature_setting, passive_response_quality
from hierarchos.training.trainer import (
    compute_chunk_training_weights,
    compute_remaining_update_steps,
    estimate_cuda_loss_chunk_rows,
    train_step,
)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    (
        ({"n_slots": 0}, "n_slots"),
        ({"key_dim": 2.5}, "key_dim"),
        ({"val_dim": float("inf")}, "val_dim"),
        ({"reference_chunk_len": 0}, "reference_chunk_len"),
        ({"lr": -1e-3}, "lr"),
        ({"momentum": 1.01}, "momentum"),
        ({"wd": -1e-4}, "wd"),
        ({"forget_rate": 1.01}, "forget_rate"),
        ({"score_grad_scale": float("nan")}, "score_grad_scale"),
    ),
)
def test_ltm_constructor_rejects_invalid_numeric_state_policy(kwargs, field):
    parameters = {"n_slots": 4, "key_dim": 3, "val_dim": 2}
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=field):
        LTMModule(**parameters)


def test_ltm_constructor_preserves_valid_zero_and_boundary_controls():
    ltm = LTMModule(
        n_slots=4,
        key_dim=3,
        val_dim=2,
        lr=0,
        momentum=1,
        wd=0,
        forget_rate=1,
        reference_chunk_len=1,
        score_grad_scale=0,
    )

    assert ltm.lr == 0.0
    assert ltm.momentum == 1.0
    assert ltm.weight_decay == 0.0
    assert ltm.forget_rate == 1.0
    assert ltm.reference_chunk_len == 1
    assert ltm.score_grad_scale == 0.0


def _config():
    return AttrDict(
        vocab_size=128,
        context_dim=24,
        persistent_dim=8,
        ltm_slots=24,
        ltm_key_dim=12,
        ltm_val_dim=12,
        ltm_lr=0.05,
        ltm_topk=2,
        ltm_forget_rate=0.0,
        h_hidden=24,
        l_hidden=24,
        max_h_steps=1,
        max_l_steps=1,
        h_stride=2,
        l_conv_atol=1e-4,
        commitment_threshold=0.05,
        isolate_batch_ltm=True,
        val_proj_trained=True,
        use_deepembed=False,
        use_rosa=False,
        compile=False,
    )


def test_plain_eval_forward_is_read_only_by_default():
    torch.manual_seed(7)
    model = HierarchosCore(_config())
    model.eval()

    input_ids = torch.tensor([[1, 2, 3], [31, 32, 33]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()

    with torch.no_grad():
        out = model(input_ids)

    ltm_state = out["ltm_memory_state"]
    # A read-only batch shares one immutable 2-D memory store. Expanding it to
    # [batch, slots, dim] makes ``slow + fast`` allocate a full batched memory at
    # every token despite never writing it.
    assert ltm_state[0].shape == (model.config.ltm_slots, model.config.ltm_val_dim)
    assert ltm_state[0].data_ptr() == model.ltm.fast_vals.data_ptr()
    assert torch.allclose(model.ltm.fast_vals, initial_fast)
    assert torch.allclose(ltm_state[0], initial_fast)


def test_read_only_training_gathers_slow_and_fast_without_batched_full_store():
    torch.manual_seed(71)
    config = _config()
    config.ltm_training_mode = "read-only"
    model = HierarchosCore(config).train()
    gathered_memories = []
    original_gather = model.ltm._gather_slot_values

    def capture_gather(index, memory):
        gathered_memories.append(memory)
        return original_gather(index, memory)

    model.ltm._gather_slot_values = capture_gather
    input_ids = torch.tensor([[1, 2, 3], [31, 32, 33]], dtype=torch.long)
    outputs = model(
        input_ids,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
        return_step_telemetry=False,
    )

    assert outputs["ltm_memory_state"][0].ndim == 2
    # Each token gathers once from slow values and once from fast values. No
    # [batch, slots, dim] slow+fast temporary is ever materialized.
    assert len(gathered_memories) == input_ids.shape[1] * 2
    assert {
        tuple(memory.shape)
        for memory in gathered_memories
    } == {(model.config.ltm_slots, model.config.ltm_val_dim)}
    assert {
        memory.data_ptr()
        for memory in gathered_memories
    } == {
        model.ltm.vals.data_ptr(),
        model.ltm.fast_vals.data_ptr(),
    }


def test_shared_ltm_state_is_isolated_before_batched_runtime_writes():
    torch.manual_seed(711)
    config = _config()
    config.ltm_training_mode = "read-only"
    model = HierarchosCore(config).eval()
    shared_state = (
        model.ltm.fast_vals.clone(),
        model.ltm._mom_vals.clone(),
        None,
        None,
        model.ltm.timestamps.clone(),
        model.ltm.sources.clone(),
        model.ltm.wallclock_timestamps.clone(),
    )
    input_ids = torch.tensor([[1, 2], [31, 32]], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids,
            ltm_memory_state=shared_state,
            suppress_hebbian=False,
        )

    runtime_state = outputs["ltm_memory_state"]
    assert runtime_state[0].shape == (
        2,
        model.config.ltm_slots,
        model.config.ltm_val_dim,
    )
    assert runtime_state[1].shape == runtime_state[0].shape
    assert runtime_state[4].shape == (2, model.config.ltm_slots)
    assert runtime_state[5].shape == (2, model.config.ltm_slots)
    assert runtime_state[6].shape == (2, model.config.ltm_slots)
    assert runtime_state[0][0].data_ptr() != runtime_state[0][1].data_ptr()


def test_last_logit_only_is_exact_and_does_not_build_step_telemetry():
    torch.manual_seed(72)
    model = HierarchosCore(_config()).eval()
    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)

    with torch.no_grad():
        full = model(input_ids)
        last = model(
            input_ids,
            return_last_logit_only=True,
            return_step_telemetry=False,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
        )

    torch.testing.assert_close(
        last["logits"],
        full["logits"][:, -1:],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(last["h_state"], full["h_state"], rtol=0, atol=0)
    torch.testing.assert_close(last["l_state"], full["l_state"], rtol=0, atol=0)
    assert last["step_telemetry"] is None


def test_last_logit_only_selects_each_rows_last_active_token():
    torch.manual_seed(721)
    model = HierarchosCore(_config()).eval()
    input_ids = torch.tensor(
        [[7, 8, 9, 10], [11, 12, 0, 0]],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0]],
        dtype=torch.long,
    )

    with torch.no_grad():
        full = model(input_ids, attention_mask=attention_mask)
        last = model(
            input_ids,
            attention_mask=attention_mask,
            return_last_logit_only=True,
            return_step_telemetry=False,
        )

    expected = torch.stack(
        [full["logits"][0, 3], full["logits"][1, 1]],
        dim=0,
    ).unsqueeze(1)
    torch.testing.assert_close(last["logits"], expected, rtol=1e-5, atol=1e-6)


def test_fully_filtered_ltm_slots_inject_no_timestamp_positional_signal():
    torch.manual_seed(8)
    model = HierarchosCore(_config()).eval()
    captured_mac_inputs = []

    def capture_mac_input(_module, inputs):
        captured_mac_inputs.append(inputs[0].detach().clone())

    handle = model.in_proj.register_forward_pre_hook(capture_mac_input)
    try:
        with torch.no_grad():
            outputs = model(
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                min_timestamp=1.0,
                suppress_hebbian=True,
            )
    finally:
        handle.remove()

    assert torch.all(outputs["topk_idx"] == -1)
    memory_width = model.config.ltm_topk * model.config.ltm_val_dim
    for mac_input in captured_mac_inputs:
        assert torch.count_nonzero(mac_input[..., -memory_width:]) == 0


def test_ltm_write_metadata_keeps_token_wallclock_and_source_clocks_separate():
    torch.manual_seed(9)
    ltm = LTMModule(
        n_slots=4,
        key_dim=2,
        val_dim=3,
        forget_rate=0.0,
    )
    with torch.no_grad():
        ltm.keys.zero_()
        ltm.keys[0, 0] = 10.0

    topk_idx = torch.tensor([[[0]]], dtype=torch.long)
    grads = torch.ones(1, 1, 1, 3)
    ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.1,
        timestamp=17.0,
        source=LTMModule.SRC_CORRECTION,
        wallclock_timestamp=1_700_000_000.0,
        tokens_covered=1,
        inplace=True,
    )

    assert ltm.timestamps[0].item() == 17.0
    assert ltm.sources[0].item() == LTMModule.SRC_CORRECTION
    assert ltm.wallclock_timestamps[0].item() == 1_700_000_000.0

    query = torch.tensor([[1.0, 0.0]])
    _vals, idx, _ts = ltm.retrieve_topk(
        query,
        topk=1,
        source_filter=LTMModule.SRC_CORRECTION,
        min_wallclock_timestamp=1_699_999_999.0,
    )
    assert idx.item() == 0
    _vals, idx, _ts = ltm.retrieve_topk(
        query,
        topk=1,
        source_filter=LTMModule.SRC_TRAINING_DATA,
    )
    assert idx.item() == -1
    _vals, idx, _ts = ltm.retrieve_topk(
        query,
        topk=1,
        min_wallclock_timestamp=1_700_000_001.0,
    )
    assert idx.item() == -1


def test_core_hebbian_write_propagates_requested_source_and_clocks():
    torch.manual_seed(10)
    model = HierarchosCore(_config()).eval()
    with torch.no_grad():
        outputs = model(
            torch.tensor([[4, 5, 6]], dtype=torch.long),
            allow_hebbian_update=True,
            memory_write_source=LTMModule.SRC_CORRECTION,
            memory_write_timestamp=123.0,
            memory_write_wallclock_timestamp=456.0,
        )

    state = outputs["ltm_memory_state"]
    touched = torch.unique(outputs["topk_idx"][outputs["topk_idx"] >= 0])
    assert touched.numel() > 0
    assert torch.all(state[4][touched] == 123.0)
    assert torch.all(state[5][touched] == LTMModule.SRC_CORRECTION)
    assert torch.all(state[6][touched] == 456.0)


def test_ltm_delta_accumulator_records_global_decay_exactly_and_cumulatively():
    ltm = LTMModule(
        n_slots=5,
        key_dim=2,
        val_dim=3,
        momentum=0.0,
        wd=0.0,
        forget_rate=0.25,
        reference_chunk_len=4,
    )
    with torch.no_grad():
        ltm.fast_vals.fill_(1.0)
    ltm.accumulate_deltas = True
    initial = ltm.fast_vals.clone()
    topk_idx = torch.tensor([[[0]]], dtype=torch.long)
    zero_grads = torch.zeros(1, 1, 1, 3)

    ltm.inner_update(
        topk_idx,
        zero_grads,
        current_lr=0.0,
        timestamp=1.0,
        tokens_covered=4,
        inplace=True,
    )
    torch.testing.assert_close(ltm.ltm_deltas, ltm.fast_vals - initial)

    ltm.inner_update(
        topk_idx,
        zero_grads,
        current_lr=0.0,
        timestamp=2.0,
        tokens_covered=4,
        inplace=True,
    )
    torch.testing.assert_close(ltm.ltm_deltas, ltm.fast_vals - initial)


def test_validation_hebbian_batch_ltm_state_is_isolated_from_global_buffer():
    torch.manual_seed(7)
    model = HierarchosCore(_config())
    model.eval()

    input_ids = torch.tensor([[1, 2, 3], [31, 32, 33]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()

    with torch.no_grad():
        out = model(input_ids, allow_hebbian_update=True)

    ltm_state = out["ltm_memory_state"]
    assert ltm_state[0].shape == (2, model.config.ltm_slots, model.config.ltm_val_dim)
    assert torch.allclose(model.ltm.fast_vals, initial_fast)
    assert not torch.allclose(ltm_state[0], initial_fast.unsqueeze(0).expand_as(ltm_state[0]))


def test_suppressed_hebbian_leaves_ltm_unchanged_during_generation():
    torch.manual_seed(11)
    model = HierarchosCore(_config())
    model.eval()
    model.suppress_hebbian = True

    input_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()
    initial_mom = model.ltm._mom_vals.clone()

    with torch.no_grad():
        out = model(input_ids)

    ltm_state = out["ltm_memory_state"]
    assert torch.allclose(model.ltm.fast_vals, initial_fast)
    assert torch.allclose(model.ltm._mom_vals, initial_mom)
    assert torch.allclose(ltm_state[0], initial_fast)
    assert torch.allclose(ltm_state[1], initial_mom)


def test_explicit_hebbian_still_writes_single_sequence_working_memory():
    torch.manual_seed(13)
    model = HierarchosCore(_config())
    model.eval()

    input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()

    with torch.no_grad():
        model(input_ids, allow_hebbian_update=True)

    diff = (model.ltm.fast_vals - initial_fast).abs().sum()
    assert diff > 0


def test_ltm_2d_update_matches_trainer_batched_update():
    torch.manual_seed(17)
    ltm = LTMModule(
        n_slots=8,
        key_dim=4,
        val_dim=4,
        momentum=0.9,
        wd=0.01,
        forget_rate=0.05,
        reference_chunk_len=4,
    )

    topk_idx = torch.tensor([[[0, 1], [2, 1], [1, -1]]], dtype=torch.long)
    grads = torch.randn(1, 3, 2, 4)
    fast = torch.randn(8, 4)
    mom = torch.randn(8, 4) * 0.1
    timestamps = torch.zeros(8)
    sources = torch.zeros(8, dtype=torch.long)

    fast_2d, mom_2d = ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.03,
        timestamp=5.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )
    fast_3d, mom_3d = ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.03,
        timestamp=5.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        fast_vals=fast.unsqueeze(0).clone(),
        mom_vals=mom.unsqueeze(0).clone(),
        timestamps=timestamps.unsqueeze(0).clone(),
        sources=sources.unsqueeze(0).clone(),
        inplace=True,
    )

    assert torch.allclose(fast_2d, fast_3d[0], atol=1e-6)
    assert torch.allclose(mom_2d, mom_3d[0], atol=1e-6)


def test_cuda_friendly_flat_update_matches_dense_cpu_math():
    torch.manual_seed(18)
    ltm = LTMModule(n_slots=9, key_dim=4, val_dim=5, momentum=0.8, wd=0.03, forget_rate=0.04)

    topk_idx = torch.tensor([[[0, 1, 1], [4, -1, 8]], [[2, 2, 4], [8, 0, -1]]], dtype=torch.long)
    grads = torch.randn(2, 2, 3, 5)
    fast = torch.randn(9, 5)
    mom = torch.randn(9, 5) * 0.1
    timestamps = torch.zeros(9)
    sources = torch.zeros(9, dtype=torch.long)

    dense_fast, dense_mom = ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.025,
        timestamp=11.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )

    cuda_timestamps = timestamps.clone()
    cuda_sources = sources.clone()
    sparse_fast, sparse_mom = ltm._inner_update_flat_cuda(
        topk_idx=topk_idx,
        grads_tensor=grads,
        current_lr=0.025,
        timestamp=11.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        curr_fast=fast.clone(),
        curr_mom=mom.clone(),
        timestamps=cuda_timestamps,
        sources=cuda_sources,
        inplace=True,
    )

    assert torch.allclose(sparse_fast, dense_fast, atol=1e-6)
    assert torch.allclose(sparse_mom, dense_mom, atol=1e-6)


def test_cuda_friendly_batched_update_matches_dense_cpu_math_and_deltas():
    torch.manual_seed(20)
    dense_ltm = LTMModule(n_slots=10, key_dim=4, val_dim=6, momentum=0.7, wd=0.02, forget_rate=0.03)
    sparse_ltm = LTMModule(n_slots=10, key_dim=4, val_dim=6, momentum=0.7, wd=0.02, forget_rate=0.03)
    dense_ltm.accumulate_deltas = True
    sparse_ltm.accumulate_deltas = True

    topk_idx = torch.tensor(
        [
            [[0, 1], [3, 1], [-1, 4]],
            [[2, 2], [9, -1], [0, 9]],
        ],
        dtype=torch.long,
    )
    grads = torch.randn(2, 3, 2, 6)
    fast = torch.randn(2, 10, 6)
    mom = torch.randn(2, 10, 6) * 0.1
    timestamps = torch.zeros(2, 10)
    sources = torch.zeros(2, 10, dtype=torch.long)

    dense_fast, dense_mom = dense_ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.015,
        timestamp=13.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=5,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )

    sparse_fast, sparse_mom = sparse_ltm._inner_update_batched_cuda(
        topk_idx=topk_idx,
        grads_tensor=grads,
        current_lr=0.015,
        timestamp=13.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=5,
        curr_fast=fast.clone(),
        curr_mom=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )

    assert torch.allclose(sparse_fast, dense_fast, atol=1e-6)
    assert torch.allclose(sparse_mom, dense_mom, atol=1e-6)
    assert torch.allclose(sparse_ltm.ltm_deltas, dense_ltm.ltm_deltas, atol=1e-6)


def test_cuda_friendly_update_with_no_valid_slots_is_read_only():
    torch.manual_seed(21)
    ltm = LTMModule(n_slots=5, key_dim=3, val_dim=4, momentum=0.7, wd=0.02, forget_rate=0.1)
    topk_idx = torch.full((2, 3, 2), -1, dtype=torch.long)
    grads = torch.randn(2, 3, 2, 4)
    fast = torch.randn(2, 5, 4)
    mom = torch.randn(2, 5, 4)
    timestamps = torch.randn(2, 5)
    sources = torch.ones(2, 5, dtype=torch.long)

    new_fast, new_mom = ltm._inner_update_batched_cuda(
        topk_idx=topk_idx,
        grads_tensor=grads,
        current_lr=0.02,
        timestamp=9.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        curr_fast=fast.clone(),
        curr_mom=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )

    assert torch.allclose(new_fast, fast)
    assert torch.allclose(new_mom, mom)


def test_cuda_friendly_retrieve_matches_dense_cpu_math_with_filter():
    torch.manual_seed(22)
    dense_ltm = LTMModule(n_slots=7, key_dim=4, val_dim=3)
    sparse_ltm = LTMModule(n_slots=7, key_dim=4, val_dim=3)
    sparse_ltm.load_state_dict(dense_ltm.state_dict())
    sparse_ltm._use_cuda_math = lambda tensor: True

    queries = torch.randn(2, 4)
    fast_vals = torch.randn(7, 3) * 0.1
    timestamps = torch.tensor([0.0, 1.0, 4.0, 2.0, 5.0, 0.5, 3.0])
    sources = torch.tensor([
        LTMModule.SRC_UNKNOWN,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_USER_INTERACTION,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_CORRECTION,
        LTMModule.SRC_TRAINING_DATA,
    ])

    dense_vals, dense_idx, dense_ts = dense_ltm.retrieve_topk(
        queries,
        topk=3,
        min_timestamp=1.0,
        source_filter=LTMModule.SRC_TRAINING_DATA,
        fast_vals=fast_vals,
        timestamps=timestamps,
        sources=sources,
    )
    sparse_vals, sparse_idx, sparse_ts = sparse_ltm.retrieve_topk(
        queries,
        topk=3,
        min_timestamp=1.0,
        source_filter=LTMModule.SRC_TRAINING_DATA,
        fast_vals=fast_vals,
        timestamps=timestamps,
        sources=sources,
    )

    assert torch.allclose(sparse_vals, dense_vals, atol=1e-6)
    assert torch.equal(sparse_idx, dense_idx)
    assert torch.allclose(sparse_ts, dense_ts, atol=1e-6)


def test_cpu_gather_retrieval_default_matches_dense_cpu_math():
    torch.manual_seed(221)
    dense_ltm = LTMModule(n_slots=11, key_dim=5, val_dim=4, cpu_gather_retrieval=False)
    gather_ltm = LTMModule(n_slots=11, key_dim=5, val_dim=4, cpu_gather_retrieval=True)
    gather_ltm.load_state_dict(dense_ltm.state_dict())

    queries = torch.randn(3, 5)
    fast_vals = torch.randn(11, 4) * 0.1
    timestamps = torch.arange(11, dtype=torch.float32)
    sources = torch.tensor([
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_USER_INTERACTION,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_CORRECTION,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_UNKNOWN,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_USER_INTERACTION,
        LTMModule.SRC_TRAINING_DATA,
        LTMModule.SRC_TRAINING_DATA,
    ])

    dense_vals, dense_idx, dense_ts = dense_ltm.retrieve_topk(
        queries,
        topk=4,
        min_timestamp=2.0,
        source_filter=LTMModule.SRC_TRAINING_DATA,
        fast_vals=fast_vals,
        timestamps=timestamps,
        sources=sources,
    )
    gather_vals, gather_idx, gather_ts = gather_ltm.retrieve_topk(
        queries,
        topk=4,
        min_timestamp=2.0,
        source_filter=LTMModule.SRC_TRAINING_DATA,
        fast_vals=fast_vals,
        timestamps=timestamps,
        sources=sources,
    )

    assert torch.allclose(gather_vals, dense_vals, atol=1e-6)
    assert torch.equal(gather_idx, dense_idx)
    assert torch.allclose(gather_ts, dense_ts, atol=1e-6)


def test_batched_gather_adds_only_selected_slow_and_fast_slots():
    torch.manual_seed(223)
    ltm = LTMModule(
        n_slots=9,
        key_dim=5,
        val_dim=4,
        cpu_gather_retrieval=True,
    )
    queries = torch.randn(3, 5)
    fast_vals = torch.randn(3, 9, 4) * 0.1
    timestamps = torch.arange(9, dtype=torch.float32).expand(3, -1).clone()

    with torch.no_grad():
        retrieved, indices, _timestamps = ltm.retrieve_topk(
            queries,
            topk=3,
            fast_vals=fast_vals,
            timestamps=timestamps,
        )
        effective = ltm.vals.unsqueeze(0) + fast_vals
        batch_index = torch.arange(queries.shape[0]).unsqueeze(1)
        expected = effective[batch_index, indices]

    torch.testing.assert_close(retrieved, expected, rtol=0, atol=0)


def test_cpu_sparse_update_default_matches_dense_cpu_math():
    torch.manual_seed(222)
    dense_ltm = LTMModule(n_slots=12, key_dim=4, val_dim=5, momentum=0.8, wd=0.02, forget_rate=0.04, cpu_sparse_update=False)
    sparse_ltm = LTMModule(n_slots=12, key_dim=4, val_dim=5, momentum=0.8, wd=0.02, forget_rate=0.04, cpu_sparse_update=True)
    sparse_ltm.load_state_dict(dense_ltm.state_dict())

    topk_idx = torch.tensor(
        [
            [[0, 1], [5, 1], [11, -1]],
            [[2, 2], [8, -1], [0, 8]],
        ],
        dtype=torch.long,
    )
    grads = torch.randn(2, 3, 2, 5)
    fast = torch.randn(2, 12, 5)
    mom = torch.randn(2, 12, 5) * 0.1
    timestamps = torch.zeros(2, 12)
    sources = torch.zeros(2, 12, dtype=torch.long)

    dense_fast, dense_mom = dense_ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.015,
        timestamp=13.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=5,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )
    sparse_fast, sparse_mom = sparse_ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.015,
        timestamp=13.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=5,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )

    assert torch.allclose(sparse_fast, dense_fast, atol=1e-6)
    assert torch.allclose(sparse_mom, dense_mom, atol=1e-6)


def test_hebbian_update_uses_same_rule_as_negative_value_gradient():
    torch.manual_seed(19)
    ltm = LTMModule(n_slots=8, key_dim=4, val_dim=4, momentum=0.9, wd=0.01, forget_rate=0.02)
    topk_idx = torch.tensor([[[0, 1], [3, 1]]], dtype=torch.long)
    vals = torch.randn(1, 2, 2, 4)
    fast = torch.randn(1, 8, 4)
    mom = torch.randn(1, 8, 4) * 0.1

    fast_grad, mom_grad = ltm.inner_update(
        topk_idx,
        -vals,
        current_lr=0.02,
        timestamp=7.0,
        source=LTMModule.SRC_USER_INTERACTION,
        tokens_covered=2,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        inplace=True,
    )
    fast_hebb, mom_hebb = ltm.update_memory_hebbian(
        topk_idx,
        None,
        vals,
        current_lr=0.02,
        timestamp=7.0,
        source=LTMModule.SRC_USER_INTERACTION,
        tokens_covered=2,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        inplace=True,
    )

    assert torch.allclose(fast_grad, fast_hebb, atol=1e-6)
    assert torch.allclose(mom_grad, mom_hebb, atol=1e-6)


def test_feedback_path_raw_topk_tensors_receive_gradients():
    torch.manual_seed(23)
    model = HierarchosCore(_config())
    model.train()

    input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels, suppress_hebbian=True)
    raw_topk_vals = outputs["raw_topk_vals"]
    for value_tensor in raw_topk_vals:
        if value_tensor.requires_grad:
            value_tensor.retain_grad()

    outputs["loss"].backward()
    grad_norm = sum(
        value_tensor.grad.abs().sum().item()
        for value_tensor in raw_topk_vals
        if value_tensor.grad is not None
    )

    assert grad_norm > 0


def test_memory_addressing_parameters_receive_gradients():
    torch.manual_seed(24)
    model = HierarchosCore(_config())
    model.train()

    input_ids = torch.tensor([[15, 16, 17, 18]], dtype=torch.long)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels, suppress_hebbian=True)
    outputs["loss"].backward()

    assert model.qproj.weight.grad is not None
    assert model.qproj.weight.grad.abs().sum().item() > 0
    assert model.ltm.keys.grad is not None
    assert model.ltm.keys.grad.abs().sum().item() > 0


def test_ltm_score_gradients_do_not_change_retrieved_values():
    torch.manual_seed(25)
    ltm = LTMModule(n_slots=8, key_dim=4, val_dim=4)
    queries = torch.randn(2, 4, requires_grad=True)

    ltm.score_grad_scale = 0.0
    vals_without, idx_without, ts_without = ltm.retrieve_topk(queries, topk=3)

    ltm.score_grad_scale = 1.0
    vals_with, idx_with, ts_with = ltm.retrieve_topk(queries, topk=3)

    assert torch.allclose(vals_with, vals_without, atol=0.0)
    assert torch.equal(idx_with, idx_without)
    assert torch.allclose(ts_with, ts_without, atol=0.0)


def test_drift_state_argument_seeds_next_forward_call():
    torch.manual_seed(26)
    cfg = _config()
    cfg.max_l_steps = 1
    model = HierarchosCore(cfg)
    model.eval()

    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    next_ids = torch.tensor([[5]], dtype=torch.long)

    with torch.no_grad():
        base = model(prefix)
        common_kwargs = dict(
            input_ids=next_ids,
            h_state=base["h_state"],
            l_state=base["l_state"],
            prev_context=base["prev_context"],
            target_context=base["target_context"],
            ltm_memory_state=base["ltm_memory_state"],
            global_pos_offset=prefix.shape[1],
        )
        out_zero = model(**common_kwargs, drift_state=torch.zeros_like(base["drift_state"]))
        out_seeded = model(**common_kwargs, drift_state=torch.ones_like(base["drift_state"]) * 4.0)

    assert not torch.allclose(out_zero["logits"], out_seeded["logits"], atol=1e-6)
    assert not torch.allclose(out_zero["drift_state"], out_seeded["drift_state"], atol=1e-6)


def test_chunked_lm_loss_matches_full_logits_objective_and_gradients():
    torch.manual_seed(27)
    cfg = _config()
    cfg.vocab_size = 257
    cfg.z_loss_weight = 1e-4
    cfg.cuda_loss_chunk_rows = 7
    cfg.cpu_loss_chunk_rows = 7
    model = HierarchosCore(cfg)
    model.train()

    hidden_a = torch.randn(3, 17, cfg.context_dim, requires_grad=True)
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    labels = torch.randint(0, cfg.vocab_size, (3, 17), dtype=torch.long)
    labels[0, :5] = -100
    labels[1, 8:12] = -100
    labels[2, -3:] = -100

    logits = torch.clamp(model.lm_head(hidden_a), min=-30.0, max=30.0)
    flat_logits = logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size).float()
    flat_labels = labels[:, 1:].contiguous().view(-1)
    full_loss = F.cross_entropy(flat_logits, flat_labels)
    valid_logits = flat_logits[flat_labels != -100]
    full_loss = full_loss + torch.logsumexp(valid_logits, dim=-1).pow(2).mean() * cfg.z_loss_weight

    model.zero_grad(set_to_none=True)
    full_loss.backward()
    full_hidden_grad = hidden_a.grad.detach().clone()
    full_head_grad = model.lm_head.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    chunked_loss = model._compute_cuda_chunked_lm_loss(hidden_b, labels, cfg.z_loss_weight)
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, full_loss.detach(), atol=1e-6)
    assert torch.allclose(hidden_b.grad, full_hidden_grad, atol=1e-6)
    assert torch.allclose(model.lm_head.weight.grad, full_head_grad, atol=1e-6)


def test_chunked_lm_loss_supports_one_token_boundary_lookahead():
    torch.manual_seed(271)
    cfg = _config()
    cfg.vocab_size = 257
    cfg.z_loss_weight = 1e-4
    cfg.cuda_loss_chunk_rows = 3
    cfg.cpu_loss_chunk_rows = 3
    model = HierarchosCore(cfg)
    model.train()

    hidden_a = torch.randn(2, 5, cfg.context_dim, requires_grad=True)
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    labels = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    labels[0, 2] = -100
    labels[1, 5] = -100

    logits = torch.clamp(model.lm_head(hidden_a), min=-30.0, max=30.0)
    flat_logits = logits.contiguous().view(-1, cfg.vocab_size).float()
    flat_labels = labels[:, 1:].contiguous().view(-1)
    full_loss = F.cross_entropy(flat_logits, flat_labels)
    valid_logits = flat_logits[flat_labels != -100]
    full_loss = full_loss + torch.logsumexp(valid_logits, dim=-1).pow(2).mean() * cfg.z_loss_weight

    model.zero_grad(set_to_none=True)
    full_loss.backward()
    full_hidden_grad = hidden_a.grad.detach().clone()
    full_head_grad = model.lm_head.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    chunked_loss = model._compute_cuda_chunked_lm_loss(hidden_b, labels, cfg.z_loss_weight)
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, full_loss.detach(), atol=1e-6)
    assert torch.allclose(hidden_b.grad, full_hidden_grad, atol=1e-6)
    assert torch.allclose(model.lm_head.weight.grad, full_head_grad, atol=1e-6)


def test_chunked_lm_loss_all_ignored_rows_is_zero_with_zero_gradients():
    torch.manual_seed(27)
    cfg = _config()
    cfg.vocab_size = 257
    cfg.cuda_loss_chunk_rows = 5
    cfg.cpu_loss_chunk_rows = 5
    model = HierarchosCore(cfg)
    model.train()

    hidden = torch.randn(2, 9, cfg.context_dim, requires_grad=True)
    labels = torch.full((2, 9), -100, dtype=torch.long)

    loss = model._compute_cuda_chunked_lm_loss(hidden, labels, z_loss_weight=1e-4)
    loss.backward()

    assert loss.item() == 0.0
    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() == 0.0


def test_training_forward_can_skip_logits_while_preserving_ltm_gradients():
    torch.manual_seed(28)
    cfg = _config()
    cfg.vocab_size = 257
    cfg.z_loss_weight = 1e-4
    cfg.cuda_loss_chunk_rows = 5
    model = HierarchosCore(cfg)
    model.train()

    input_ids = torch.tensor([[21, 22, 23, 24, 25]], dtype=torch.long)
    labels = input_ids.clone()

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        suppress_hebbian=True,
        return_logits=False,
        return_topk_values=False,
    )

    assert outputs["logits"] is None
    assert outputs["topk_vals"] is None
    assert outputs["loss"] is not None

    for value_tensor in outputs["raw_topk_vals"]:
        if value_tensor.requires_grad:
            value_tensor.retain_grad()

    outputs["loss"].backward()
    grad_norm = sum(
        value_tensor.grad.abs().sum().item()
        for value_tensor in outputs["raw_topk_vals"]
        if value_tensor.grad is not None
    )

    assert grad_norm > 0


def test_train_step_uses_cpu_supervised_row_loss_path_by_default():
    class RecordingCore(HierarchosCore):
        def __init__(self, config):
            super().__init__(config)
            self.seen_return_logits = []

        def forward(self, *args, **kwargs):
            self.seen_return_logits.append(kwargs.get("return_logits", True))
            return super().forward(*args, **kwargs)

    torch.manual_seed(281)
    cfg = _config()
    cfg.vocab_size = 96
    cfg.cpu_chunked_lm_loss = True
    cfg.cpu_loss_chunk_rows = 3
    model = RecordingCore(cfg)
    model.train()

    batch = {
        "input_ids": torch.tensor([[21, 22, 23, 24, 25, 26]], dtype=torch.long),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 23, 24, 25, 26]], dtype=torch.long),
    }
    args = AttrDict(
        training_chunk_size=6,
        amp=False,
        amp_dtype="float16",
        persist_state=False,
        cpu_chunked_lm_loss=True,
        cuda_chunked_lm_loss=True,
        grad_clip=1.0,
        ltm_lr=0.001,
        ponder_loss_weight=0.01,
        commitment_loss_weight=0.5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    outputs, _ = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert model.seen_return_logits == [False]
    assert outputs is not None
    assert torch.isfinite(outputs["loss"]).all()


def test_inplace_ltm_update_accumulates_deltas_like_state_delta():
    torch.manual_seed(29)
    ltm = LTMModule(n_slots=6, key_dim=4, val_dim=4, momentum=0.0, wd=0.0, forget_rate=0.0)
    ltm.accumulate_deltas = True

    topk_idx = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    grads = torch.randn(1, 2, 2, 4)
    before = ltm.fast_vals.clone()

    ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.04,
        timestamp=3.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=2,
        inplace=True,
    )

    assert torch.allclose(ltm.ltm_deltas, ltm.fast_vals - before, atol=1e-6)
    assert ltm.ltm_deltas.abs().sum() > 0


def test_chat_correction_and_passive_quality_gates():
    assert extract_correction_text("its 4") == "4"
    assert extract_correction_text("Actually, the answer is four") == "the answer is four"
    assert passive_response_quality([1, 2, 3, 1, 2, 3, 1, 2, 3])[0] is False
    assert passive_response_quality(list(range(12)))[0] is True


def test_runtime_temperature_setting_grid():
    assert parse_temperature_setting("0") == 0.0
    assert parse_temperature_setting("0.05") == 0.05
    assert parse_temperature_setting("1") == 1.0

    for bad_value in ("-0.05", "1.05", "0.03"):
        try:
            parse_temperature_setting(bad_value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_value} should not be accepted")


def test_training_chunk_weights_follow_supervised_answer_tokens():
    labels = torch.full((1, 260), -100, dtype=torch.long)
    labels[0, 250:260] = torch.arange(10)

    weights = compute_chunk_training_weights(labels, chunk_size=128)

    assert len(weights) == 3
    assert weights[0]["valid_predictions"] == 0
    assert weights[1]["valid_predictions"] == 7
    assert weights[2]["valid_predictions"] == 3
    assert weights[0]["label_ratio"] == 0.0
    assert abs(sum(chunk["label_ratio"] for chunk in weights) - 1.0) < 1e-8


def test_training_chunk_token_weights_ignore_padding_only_chunks():
    labels = torch.full((1, 260), -100, dtype=torch.long)
    attention_mask = torch.zeros((1, 260), dtype=torch.long)
    attention_mask[0, :130] = 1

    weights = compute_chunk_training_weights(labels, attention_mask=attention_mask, chunk_size=128)

    assert weights[0]["real_tokens"] == 128
    assert weights[1]["real_tokens"] == 2
    assert weights[2]["real_tokens"] == 0
    assert weights[2]["token_ratio"] == 0.0
    assert abs(sum(chunk["token_ratio"] for chunk in weights) - 1.0) < 1e-8


def test_cuda_loss_chunk_rows_scale_from_free_vram_and_batch_shape():
    vocab_size = 50257
    rows_96gb = estimate_cuda_loss_chunk_rows(
        free_bytes=int(80 * 1024 ** 3),
        batch_size=82,
        chunk_size=128,
        vocab_size=vocab_size,
    )
    assert rows_96gb == 16834

    larger_batch_rows = estimate_cuda_loss_chunk_rows(
        free_bytes=int(80 * 1024 ** 3),
        batch_size=192,
        chunk_size=128,
        vocab_size=vocab_size,
    )
    assert larger_batch_rows >= int(192 * 127 * 1.05)

    constrained_rows = estimate_cuda_loss_chunk_rows(
        free_bytes=int(8 * 1024 ** 3),
        batch_size=192,
        chunk_size=128,
        vocab_size=vocab_size,
    )
    assert constrained_rows < rows_96gb

    assert estimate_cuda_loss_chunk_rows(
        free_bytes=int(80 * 1024 ** 3),
        batch_size=82,
        chunk_size=128,
        vocab_size=vocab_size,
        requested_rows=12345,
    ) == 12345


def test_masked_padding_tokens_do_not_change_training_costs():
    torch.manual_seed(31)
    model = HierarchosCore(_config())
    model.train()

    input_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0]], dtype=torch.long)
    input_b = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 91, 92, 93, 94, 95]], dtype=torch.long)
    labels = input_a.clone()
    labels[:, 7:] = -100
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)

    out_a = model(input_a, labels=labels, attention_mask=attention_mask, suppress_hebbian=True)
    out_b = model(input_b, labels=labels, attention_mask=attention_mask, suppress_hebbian=True)

    assert torch.allclose(out_a["loss"], out_b["loss"], atol=1e-6)
    assert torch.allclose(out_a["ponder_cost"], out_b["ponder_cost"], atol=1e-6)
    assert torch.allclose(out_a["commitment_cost"], out_b["commitment_cost"], atol=1e-6)


def test_override_scheduler_counts_only_remaining_mid_epoch_work():
    assert compute_remaining_update_steps(
        dataloader_len=2365,
        accumulation_steps=1,
        start_epoch=24,
        total_epochs=27,
        start_step=1200,
    ) == 5895
    assert compute_remaining_update_steps(
        dataloader_len=2365,
        accumulation_steps=1,
        start_epoch=24,
        total_epochs=27,
        start_step=0,
    ) == 7095
