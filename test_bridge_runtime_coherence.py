import importlib

import pytest
import torch

from hierarchos import HierarchosCore
from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible
from test_rwkv_v8_integrity import _tiny_config


class _ImmediateThread:
    def __init__(self, *, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


class _PromptTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self.prompts = []

    def __len__(self):
        return 64

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        self.prompts.append(text)
        ids = [1, 2]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(str(int(token)) for token in tokens)


class _EOSRecordingModel:
    def __init__(self):
        self.config = {
            "alpaca": True,
            "full_sample_bptt": True,
            "training_chunk_size": 0,
            "context_dim": 4,
        }
        self.calls = []
        self.suppress_hebbian = True

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        self.calls.append(input_ids.detach().clone())
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8)
        logits[..., 0] = 10.0
        return {
            "logits": logits,
            "h_state": kwargs.get("h_state"),
            "l_state": kwargs.get("l_state"),
            "prev_context": kwargs.get("prev_context"),
            "target_context": kwargs.get("target_context"),
            "drift_state": kwargs.get("drift_state"),
            "ltm_memory_state": kwargs.get("ltm_memory_state"),
        }


@pytest.fixture(autouse=True)
def _reset_bridge_operation_lane():
    bridge = importlib.import_module("hierarchos_bridge_server")
    bridge._stop_generation.clear()
    bridge._stop_training.clear()
    with bridge._operation_lock:
        bridge._active_operation = None
    yield
    bridge._stop_generation.clear()
    bridge._stop_training.clear()
    with bridge._operation_lock:
        bridge._active_operation = None


def test_bridge_uses_checkpoint_prompt_format_and_advances_terminal_eos(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    model = _EOSRecordingModel()
    tokenizer = _PromptTokenizer()
    emitted = []

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_tokenizer", tokenizer)
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(model.config))
    monkeypatch.setattr(bridge, "_h_state", None)
    monkeypatch.setattr(bridge, "_l_state", None)
    monkeypatch.setattr(bridge, "_prev_context", None)
    monkeypatch.setattr(bridge, "_target_context", None)
    monkeypatch.setattr(bridge, "_drift_state", None)
    monkeypatch.setattr(bridge, "_ltm_state", None)
    monkeypatch.setattr(bridge, "_total_tokens_generated", 0)
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)

    bridge.handle_generate(
        {
            "message": "hello",
            "sampling": {"max_new_tokens": 1, "temperature": 0.0},
        }
    )

    assert not [event for event, _ in emitted if event == "error"]
    assert tokenizer.prompts[0] == "### Instruction:\nhello\n\n### Response:\n"
    assert len(model.calls) == 2
    assert torch.equal(model.calls[-1], torch.tensor([[tokenizer.eos_token_id]]))
    assert bridge._total_tokens_generated == 3
    assert not [event for event, _ in emitted if event == "token"]


def test_bridge_chat_state_is_current_strict_and_round_trips(tmp_path, monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    config = _tiny_config()
    model = HierarchosCore(config).eval()
    context_dim = int(config.context_dim)

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(config))
    monkeypatch.setattr(bridge, "_model_dir", str(tmp_path))
    monkeypatch.setattr(bridge, "_model_identity", {"checkpoint_sha256": "a" * 64})
    monkeypatch.setattr(
        bridge,
        "_tokenizer_identity",
        {
            "sha256": "b" * 64,
            "behavior_sha256_v2": "c" * 64,
            "vocab_size": int(config.vocab_size),
        },
    )
    monkeypatch.setattr(bridge, "_h_state", model.h_rnn.initial_state(1))
    monkeypatch.setattr(bridge, "_l_state", model.l_rnn.initial_state(1))
    monkeypatch.setattr(bridge, "_prev_context", torch.zeros(1, context_dim))
    monkeypatch.setattr(bridge, "_target_context", torch.ones(1, context_dim))
    monkeypatch.setattr(bridge, "_drift_state", torch.full((1, context_dim), 0.5))
    monkeypatch.setattr(bridge, "_ltm_state", None)
    monkeypatch.setattr(bridge, "_total_tokens_generated", 7)

    path = tmp_path / "chat-state.pt"
    bridge._write_chat_runtime_state(str(path))
    payload = load_checkpoint_payload_compatible(str(path), map_location="cpu")
    assert payload["version"] == 4
    assert payload["bridge_runtime_identity"]["model"]["checkpoint_sha256"] == "a" * 64
    assert payload["bridge_runtime_identity"]["version"] == 2
    assert (
        payload["bridge_runtime_identity"]["tokenizer_behavior_sha256_v2"]
        == "c" * 64
    )

    bridge._reset_runtime_state()
    bridge._load_chat_runtime_state(str(path))
    assert bridge._h_state.device.type == "cpu"
    assert bridge._h_state.shape == model.h_rnn.initial_state(1).shape
    assert bridge._total_tokens_generated == 7


def test_bridge_runtime_identity_preserves_v1_and_enforces_v2_behavior(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    monkeypatch.setattr(bridge, "_model_identity", {"checkpoint_sha256": "a" * 64})
    monkeypatch.setattr(
        bridge,
        "_tokenizer_identity",
        {
            "sha256": "b" * 64,
            "behavior_sha256_v2": "c" * 64,
            "vocab_size": 64,
        },
    )
    legacy = {
        "version": 1,
        "model": {"checkpoint_sha256": "a" * 64},
        "tokenizer_sha256": "b" * 64,
        "tokenizer_vocab_size": 64,
    }
    bridge._validate_bridge_runtime_identity({"bridge_runtime_identity": legacy})

    current = bridge._bridge_runtime_identity()
    changed = dict(current)
    changed["tokenizer_behavior_sha256_v2"] = "d" * 64
    with pytest.raises(RuntimeError, match="different model weights or tokenizer"):
        bridge._validate_bridge_runtime_identity(
            {"bridge_runtime_identity": changed}
        )


def test_bridge_rejects_nonfinite_ltm_sidecar_without_mutating_values(
    tmp_path,
    monkeypatch,
):
    bridge = importlib.import_module("hierarchos_bridge_server")
    model = HierarchosCore(_tiny_config()).eval()
    before = model.ltm.vals.detach().clone()

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_model_dir", str(tmp_path))
    monkeypatch.setattr(bridge, "_model_identity", {"checkpoint_sha256": "c" * 64})
    path = tmp_path / "hierarchos_ltm_updates.pt"
    torch.save(
        {
            "version": 2,
            "delta": torch.full_like(model.ltm.vals, float("nan")),
            "base_model_identity": {"checkpoint_sha256": "c" * 64},
        },
        path,
    )

    with pytest.raises(ValueError, match="non-finite"):
        bridge._apply_saved_ltm_updates()
    torch.testing.assert_close(model.ltm.vals, before)


def test_bridge_rejects_generation_while_training_lane_is_active(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    emitted = []
    monkeypatch.setattr(bridge, "_model", _EOSRecordingModel())
    monkeypatch.setattr(bridge, "_tokenizer", _PromptTokenizer())
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    with bridge._operation_lock:
        bridge._active_operation = "training"

    bridge.handle_generate({"message": "must not run", "sampling": {}})

    assert any(event == "error" for event, _ in emitted)
    assert ("generation_complete", {"status": "rejected"}) in emitted
    assert bridge._current_operation() == "training"


def test_training_validation_failure_still_emits_terminal_and_releases_lane(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    model = _EOSRecordingModel()
    emitted = []
    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_tokenizer", _PromptTokenizer())
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(model.config))
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)

    bridge.handle_start_training({"data_path": "definitely-missing.jsonl"})

    assert ("training_complete", {"status": "error"}) in emitted
    assert any(event == "error" for event, _ in emitted)
    assert bridge._current_operation() is None
    assert model.suppress_hebbian


def test_training_cleanup_failure_cannot_leak_operation_lane(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    emitted = []
    monkeypatch.setattr(bridge, "_model", _EOSRecordingModel())
    monkeypatch.setattr(bridge, "_tokenizer", _PromptTokenizer())
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", {})
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        bridge,
        "_reset_runtime_state",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    bridge.handle_start_training({"data_path": "definitely-missing.jsonl"})

    assert bridge._current_operation() is None
    assert ("training_complete", {"status": "error"}) in emitted
    assert any(
        event == "error" and "cleanup failed" in (data or {}).get("message", "")
        for event, data in emitted
    )


def test_every_shared_state_handler_uses_the_operation_lane():
    bridge = importlib.import_module("hierarchos_bridge_server")
    guarded = {
        "get_model_info",
        "get_ltm_snapshot",
        "save_ltm_updates",
        "save_chat_runtime_state",
        "load_chat_runtime_state",
        "reset_chat_runtime_state",
        "execute_command",
        "set_threads",
    }
    for method in guarded:
        assert getattr(
            bridge.HANDLERS[method],
            "_bridge_exclusive_operation",
            None,
        )
    for responsive_method in ("stop_generation", "stop_training", "ping"):
        assert not hasattr(
            bridge.HANDLERS[responsive_method],
            "_bridge_exclusive_operation",
        )


def test_busy_lane_rejects_resets_and_thread_changes_but_allows_stop_and_ping(
    monkeypatch,
):
    bridge = importlib.import_module("hierarchos_bridge_server")
    emitted = []
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge, "_total_tokens_generated", 11)
    monkeypatch.setattr(
        bridge,
        "_apply_thread_count",
        lambda *_args, **_kwargs: pytest.fail("busy thread change reached runtime"),
    )
    with bridge._operation_lock:
        bridge._active_operation = "training"

    bridge.handle_execute_command({"command": "/reset"})
    bridge.handle_set_threads({"threads": 1})
    assert bridge._total_tokens_generated == 11
    assert bridge._current_operation() == "training"
    assert len([event for event, _ in emitted if event == "error"]) == 2

    bridge.handle_stop_generation({})
    bridge.handle_stop_training({})
    bridge.handle_ping({})
    assert bridge._stop_generation.is_set()
    assert bridge._stop_training.is_set()
    assert any(event == "pong" for event, _ in emitted)


def test_jsonl_source_scan_never_uses_whole_file_json_load(tmp_path, monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    path = tmp_path / "large.jsonl"
    path.write_text(
        '\n'.join((
            '{"text": "one"}',
            'not json',
            '',
            '{"text": "two"}',
        )),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge.json,
        "load",
        lambda *_args, **_kwargs: pytest.fail("json.load used for JSONL"),
    )

    assert list(bridge._iter_training_source_objects(str(path))) == [
        {"text": "one"},
        {"text": "two"},
    ]


def test_exact_streaming_schedule_counts_only_usable_samples():
    bridge = importlib.import_module("hierarchos_bridge_server")

    class Dataset:
        bucket_size = 8192
        shuffle_buckets = True

        def __iter__(self):
            yield {"input_ids": [1]}
            yield None
            yield {"_audit_only": True}
            yield {"input_ids": [2]}
            yield {"input_ids": [3]}

    class Loader:
        dataset = Dataset()

    batches, samples = bridge._exact_dataloader_batches(Loader(), batch_size=2)
    assert samples == 3
    assert batches == 2
    assert Loader.dataset.bucket_size == 8192


def test_exact_streaming_schedule_counts_worker_local_partial_batches():
    bridge = importlib.import_module("hierarchos_bridge_server")
    original_get_worker_info = torch.utils.data.get_worker_info

    class ShardedDataset:
        bucket_size = 8192
        shuffle_buckets = True
        counts = (1, 3, 4)

        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            count = sum(self.counts) if info is None else self.counts[info.id]
            for index in range(count):
                yield {"input_ids": [index]}

    class Loader:
        dataset = ShardedDataset()
        num_workers = 3

    batches, samples = bridge._exact_dataloader_batches(Loader(), batch_size=4)
    assert samples == 8
    # Worker-local auto-batching gives ceil(1/4)+ceil(3/4)+ceil(4/4), not
    # the incorrect global ceil(8/4).
    assert batches == 3
    assert Loader.dataset.bucket_size == 8192
    assert torch.utils.data.get_worker_info is original_get_worker_info


@pytest.mark.parametrize("cancel", [False, True])
def test_bridge_training_hooks_real_trainer_binding_and_always_cleans_up(
    tmp_path,
    monkeypatch,
    cancel,
):
    bridge = importlib.import_module("hierarchos_bridge_server")
    import hierarchos
    import hierarchos.training.trainer as trainer_module

    config = _tiny_config()
    model = HierarchosCore(config).eval()
    tokenizer = _PromptTokenizer()
    emitted = []
    original_tqdm = trainer_module.tqdm
    hook_observed = {"value": False}
    args_observed = {}

    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "tiny sample"}\n', encoding="utf-8")

    def fake_train(args, device, tokenizer_arg, dataloader, dataloader_len, model_override=None):
        hook_observed["value"] = trainer_module.tqdm is not original_tqdm
        args_observed["value"] = args
        progress = trainer_module.tqdm(
            [{"batch": 1}],
            desc="Epoch 1/1",
            total=1,
            disable=True,
        )
        if cancel:
            bridge.handle_stop_training({})
        for _ in progress:
            progress.set_postfix({"loss": "1.25", "lr": "0.0001"})

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_tokenizer", tokenizer)
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(config))
    monkeypatch.setattr(bridge, "_model_dir", str(tmp_path))
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(hierarchos, "train", fake_train)
    monkeypatch.setattr(
        hierarchos,
        "create_dataloader_for_jsonl",
        lambda *args, **kwargs: [{"batch": 1}],
    )

    bridge.handle_start_training(
        {
            "data_path": str(data_path),
            "out_dir": str(tmp_path / "out"),
            "epochs": 1,
            "batch_size": 1,
        }
    )

    terminal_status = "stopped" if cancel else "completed"
    assert hook_observed["value"]
    args = args_observed["value"]
    assert args.ponder_objective == "auto"
    assert args.max_commitment_cost_for_backward == pytest.approx(2.0)
    assert args.act_depth_temperature == pytest.approx(0.05)
    assert args.drift_state_clamp == pytest.approx(5.0)
    assert args.drift_norm_clamp == pytest.approx(0.0)
    assert args.drift_delta_scale == pytest.approx(1.0)
    assert args.min_ltm_lr == pytest.approx(args.min_lr)
    assert args.amp is False
    assert "amp" in args._explicit_cli_dests
    assert trainer_module.tqdm is original_tqdm
    assert ("training_complete", {"status": terminal_status}) in emitted
    assert bool(model.suppress_hebbian)
    assert not model.training
    assert bridge._current_operation() is None
    if cancel:
        assert not [event for event, _ in emitted if event == "training_metrics"]
    else:
        assert any(event == "training_metrics" for event, _ in emitted)
