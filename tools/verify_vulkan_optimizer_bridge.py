#!/usr/bin/env python3
"""Verify Vulkan AdamW <-> PyTorch state interchange, including tied weights."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from hierarchos.utils.rosa import ROSAState
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.vulkan_optimizer_bridge import (
    PORTABLE_PARAMETER_STATE_FORMAT,
    VULKAN_ADAMW_FORMAT,
    VULKAN_PENDING_GRADIENT_FORMAT,
    VULKAN_TRAINING_FORMAT,
    VULKAN_TRAINING_SESSION_FORMAT,
    load_vulkan_training_package_into_torch,
    load_vulkan_adamw_into_torch,
    read_vulkan_training_replay,
    read_vulkan_adamw_checkpoint,
    save_torch_adamw_as_vulkan,
    write_vulkan_training_replay,
)


SLOT_NAMES = ("lm_head.weight", "out_norm.weight", "out_norm.bias")
SLOT_DECAY_CLASSES = ("decay", "no-decay", "no-decay")


def _portable_parameter_state() -> dict[str, object]:
    return {
        "format": PORTABLE_PARAMETER_STATE_FORMAT,
        "master_file": "model.safetensors",
        "trainable_master_dtype": "float32",
        "layout": "pytorch-row-major",
        "optimizer_binding": "canonical-tensor-name",
        "parameter_aliases": [
            {"canonical": "lm_head.weight", "alias": "tok_emb.weight"}
        ],
        "execution_mirrors": {
            "persistence": "derived",
            "rebuild_from": "trainable-fp32-master",
            "rebuild_on_load": True,
            "destination_policy": "runtime-selected",
        },
    }


def _optimizer(model: HierarchosCore) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": [model.lm_head.weight], "weight_decay": 0.1},
            {"params": [model.out_norm.weight, model.out_norm.bias], "weight_decay": 0.0},
        ],
        lr=2.5e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )


def _write_template(path: Path, model: HierarchosCore, step: int) -> None:
    parameters = {
        "lm_head.weight": model.lm_head.weight,
        "out_norm.weight": model.out_norm.weight,
        "out_norm.bias": model.out_norm.bias,
    }
    tensors: dict[str, torch.Tensor] = {}
    for index, name in enumerate(SLOT_NAMES, start=1):
        count = parameters[name].numel()
        first = torch.linspace(-1.0, 1.0, count, dtype=torch.float32) * (index * 1.0e-3)
        second = torch.linspace(0.25, 1.25, count, dtype=torch.float32) * (index * 1.0e-4)
        tensors[f"optimizer.{name}.exp_avg"] = first.contiguous()
        tensors[f"optimizer.{name}.exp_avg_sq"] = second.contiguous()
    save_file(
        tensors,
        str(path),
        metadata={
            "format": VULKAN_ADAMW_FORMAT,
            "step": str(step),
            "slot_names": json.dumps(list(SLOT_NAMES), separators=(",", ":")),
            # Simulate an intermittent parameter that has already skipped two
            # outer optimizer steps before this cross-backend handoff.
            "slot_steps": json.dumps([step, step, step - 2], separators=(",", ":")),
            "slot_decay_classes": json.dumps(SLOT_DECAY_CLASSES, separators=(",", ":")),
            "layout": "pytorch-row-major",
        },
    )


def _assign_gradients(model: HierarchosCore, scale: float) -> None:
    parameters = (model.lm_head.weight, model.out_norm.weight, model.out_norm.bias)
    for index, parameter in enumerate(parameters, start=1):
        grad = torch.linspace(-0.5, 0.75, parameter.numel(), dtype=torch.float32)
        parameter.grad = grad.reshape_as(parameter) * (scale * index)


def _canonical_parameters(model: HierarchosCore) -> dict[str, torch.nn.Parameter]:
    return {
        "lm_head.weight": model.lm_head.weight,
        "out_norm.weight": model.out_norm.weight,
        "out_norm.bias": model.out_norm.bias,
    }


def _gradient_payload(model: HierarchosCore, scale: float) -> dict[str, torch.Tensor]:
    payload = {}
    for index, (name, parameter) in enumerate(_canonical_parameters(model).items(), start=1):
        payload[name] = (
            torch.linspace(-0.75, 0.5, parameter.numel(), dtype=torch.float32)
            .mul(scale * index)
            .contiguous()
        )
    return payload


def _assign_gradient_payload(model: HierarchosCore, payload: dict[str, torch.Tensor]) -> None:
    for name, parameter in _canonical_parameters(model).items():
        parameter.grad = payload[name].reshape_as(parameter).to(parameter)


def _add_gradient_payload(model: HierarchosCore, payload: dict[str, torch.Tensor]) -> None:
    for name, parameter in _canonical_parameters(model).items():
        if parameter.grad is None:
            raise AssertionError(f"missing pending gradient before continuation for {name}")
        parameter.grad.add_(payload[name].reshape_as(parameter).to(parameter))


def _divide_gradients(model: HierarchosCore, mass: float) -> None:
    for parameter in _canonical_parameters(model).values():
        if parameter.grad is not None:
            parameter.grad.div_(mass)


def _run_rust_inspector(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-optimizer-inspect",
            "--",
            "--optimizer",
            str(path),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Rust optimizer inspector failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def main() -> None:
    torch.manual_seed(20260818)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).cpu().train()
    # Sanity gate the alias that makes this bridge non-trivial.
    unique_alias = [name for name, parameter in model.named_parameters() if parameter is model.lm_head.weight]
    all_aliases = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter is model.lm_head.weight
    ]
    if unique_alias != ["tok_emb.weight"] or "lm_head.weight" not in all_aliases:
        raise AssertionError(
            f"unexpected tied-weight enumeration: unique={unique_alias} all={all_aliases}"
        )

    optimizer = _optimizer(model)
    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-optimizer-bridge-") as temp_dir:
        temp = Path(temp_dir)
        template = temp / "optimizer-template.safetensors"
        exported = temp / "optimizer-from-pytorch.safetensors"
        _write_template(template, model, step=4)

        loaded = load_vulkan_adamw_into_torch(model, optimizer, template)
        if loaded.step != 4 or loaded.slot_names != SLOT_NAMES:
            raise AssertionError("Vulkan optimizer metadata changed while loading into PyTorch")
        if loaded.slot_steps != {
            "lm_head.weight": 4,
            "out_norm.weight": 4,
            "out_norm.bias": 2,
        }:
            raise AssertionError(f"independent Vulkan Adam steps were not decoded: {loaded.slot_steps}")
        if tuple(loaded.slot_decay_classes[name] for name in SLOT_NAMES) != SLOT_DECAY_CLASSES:
            raise AssertionError(
                f"Vulkan AdamW decay topology was not decoded: {loaded.slot_decay_classes}"
            )
        if int(optimizer.state[model.lm_head.weight]["step"].item()) != 4:
            raise AssertionError("canonical lm_head.weight state did not reach the tied PyTorch parameter")
        if int(optimizer.state[model.out_norm.bias]["step"].item()) != 2:
            raise AssertionError("intermittent slot step did not reach PyTorch AdamW")

        _assign_gradients(model, 1.0e-2)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        written = save_torch_adamw_as_vulkan(
            model,
            optimizer,
            exported,
            template_checkpoint=template,
        )
        if written.step != 5:
            raise AssertionError(f"expected exported AdamW step 5, got {written.step}")
        if written.slot_steps["out_norm.bias"] != 3:
            raise AssertionError(
                f"PyTorch did not preserve independent intermittent step: {written.slot_steps}"
            )
        if tuple(written.slot_decay_classes[name] for name in SLOT_NAMES) != SLOT_DECAY_CLASSES:
            raise AssertionError(
                f"PyTorch did not preserve Vulkan AdamW decay topology: {written.slot_decay_classes}"
            )

        rust = _run_rust_inspector(exported)
        if rust["step"] != 5 or rust["tensor_count"] != len(SLOT_NAMES):
            raise AssertionError(f"Rust rejected or misread PyTorch-exported optimizer state: {rust}")
        if tuple(rust["slot_names"]) != SLOT_NAMES:
            raise AssertionError(f"Rust slot ordering changed: {rust['slot_names']}")
        if rust["slot_steps"] != [5, 5, 3]:
            raise AssertionError(f"Rust lost independent PyTorch Adam steps: {rust['slot_steps']}")
        if rust["slot_decay_classes"] != list(SLOT_DECAY_CLASSES):
            raise AssertionError(
                f"Rust lost PyTorch-exported AdamW decay topology: {rust['slot_decay_classes']}"
            )

        bad_model = HierarchosCore(config).cpu().train()
        bad_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        bad_model.lm_head.weight,
                        bad_model.out_norm.weight,
                        bad_model.out_norm.bias,
                    ],
                    "weight_decay": 0.1,
                }
            ],
            lr=2.5e-4,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        try:
            load_vulkan_adamw_into_torch(bad_model, bad_optimizer, exported)
        except ValueError as exc:
            if "declared no-decay" not in str(exc):
                raise AssertionError(f"unexpected decay-topology rejection: {exc}") from exc
        else:
            raise AssertionError(
                "Vulkan AdamW v3 allowed a no-decay slot to resume in a decayed PyTorch group"
            )

        # Build the training-only half of a v2 package at an arbitrary point in
        # a weighted-token accumulation window, then prove PyTorch can finish
        # the same window with an identical AdamW trajectory.
        package_dir = temp / "mid-window-package"
        package_dir.mkdir()
        shutil.copyfile(exported, package_dir / "optimizer.safetensors")
        save_file(
            {
                name: parameter.detach().cpu().float().contiguous().clone()
                for name, parameter in _canonical_parameters(model).items()
            },
            str(package_dir / "model.safetensors"),
            metadata={"format": "hierarchos-rust-fp32-v1"},
        )
        pending = _gradient_payload(model, 2.0e-2)
        save_file(
            pending,
            str(package_dir / "gradients.safetensors"),
            metadata={
                "format": VULKAN_PENDING_GRADIENT_FORMAT,
                "slot_names": json.dumps(list(SLOT_NAMES), separators=(",", ":")),
                "layout": "pytorch-row-major",
            },
        )
        consumed_mass = 5.5
        closing_mass = 3.75
        target_mass = consumed_mass + closing_mass
        (package_dir / "training_state.json").write_text(
            json.dumps(
                {
                    "format": VULKAN_TRAINING_FORMAT,
                    "model_file": "model.safetensors",
                    "parameter_state": _portable_parameter_state(),
                    "optimizer_file": "optimizer.safetensors",
                    "optimizer_step": 5,
                    "optimizer_tensor_count": len(SLOT_NAMES),
                    "gradient_file": "gradients.safetensors",
                    "gradient_tensor_count": len(SLOT_NAMES),
                    "accumulation_open": True,
                    "accumulation_normalization": "mean-by-supervision-weight",
                    "accumulation_consumed_token_count": 4,
                    "accumulation_consumed_supervision_mass": consumed_mass,
                    "accumulation_target_token_count": 7,
                    "accumulation_target_supervision_mass": target_mass,
                    "lm_head_gradient_topology": "shared-tied",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        replay_rosa = ROSAState(
            transitions={0: {5: 1}, 1: {}},
            suffix_links=[-1, 0],
            lengths=[0, 1],
            endpos=[0, 1],
            last_state=1,
            num_states=2,
            tokens=[5],
        )
        replay_checkpoint = {
            "completed_epoch": 2,
            "mid_epoch_step": 3,
            "rng_state": {
                "python": random.Random(17).getstate(),
                "numpy": np.random.RandomState(19).get_state(),
                "torch": torch.arange(32, dtype=torch.uint8),
            },
            "data_state": {
                "sampler": {
                    "class": "tests.DeterministicSampler",
                    "epoch": 2,
                    "generator_state": torch.arange(16, dtype=torch.uint8),
                }
            },
            "data_stream_cursor": {
                "format": "hierarchos-data-stream-rng-cursor-v1",
                "sampler_kind": "epoch-shuffle",
                "rng_algorithm": "splitmix64-fisher-yates-v1",
                "seed": 321,
                "epoch": 2,
                "batch_cursor": 3,
                "dataset_size": 16,
                "batch_size": 2,
                "shuffle": True,
                "drop_last": False,
            },
            "execution_policy": {
                "format": "hierarchos-training-execution-policy-v1",
                "source_backend": "vulkan",
                "compute_dtype": "float32",
                "autocast_enabled": False,
                "stochastic_rng": {
                    "mode": "canonical-counter",
                    "state_required": False,
                    "canonical_counter": {
                        "algorithm": "philox4x32-10-word-v1",
                        "seed": 17,
                        "next_word": 4096,
                    },
                },
                "loss_scaling": {
                    "mode": "none",
                    "pending_gradients_scaled": False,
                },
            },
            "running_states": (
                torch.tensor([1.0, 2.0]),
                torch.tensor([3.0]),
                torch.tensor([4.0]),
                torch.tensor([5.0]),
                torch.tensor([6.0]),
                (
                    torch.tensor([[7.0]]),
                    torch.tensor([[8.0]]),
                    [5],
                    [replay_rosa],
                    [9],
                    ["bridge-test"],
                    [10.0],
                ),
            ),
            "run_identity": {
                "version": 1,
                "objective": {"persist_state": True},
                "dataset": {"replay_guarantee": "content-addressed-token-cache"},
            },
            "effective_training_config": {
                "persist_state": True,
                "starting_lr": 2.5e-4,
                "min_lr": 2.5e-5,
                "warmup_steps": 2,
                "warmup_ratio": 0.1,
                "disable_lr_schedule": False,
                "ltm_lr": 1.0e-3,
                "min_ltm_lr": 1.0e-4,
                "disable_ltm_lr_schedule": False,
            },
            "scheduler_state_dict": {
                "base_lrs": [2.5e-4, 2.5e-4],
                # Exact LambdaLR continuation keeps the saved 20-step curve,
                # but last_epoch remains monotonic when training is extended.
                "last_epoch": 27,
                "verbose": False,
                "_step_count": 28,
                "_get_lr_called_within_step": False,
                "_last_lr": [2.5e-5, 2.5e-5],
                "lr_lambdas": [None, None],
            },
            "lr_scheduler_state": {
                "enabled": True,
                "step": 27,
                "total_steps": 20,
                "max_lr": 2.5e-4,
                "min_lr": 2.5e-5,
                "warmup_steps": 2,
                "warmup_ratio": 0.1,
                "resolved_warmup_steps": 2,
            },
            "ltm_scheduler_state": {
                "enabled": True,
                "step": 7,
                "total_steps": 20,
                "max_lr": 1.0e-3,
                "min_lr": 1.0e-4,
            },
            "error_budget_state": {"skipped_train_batches": 2},
            "optimizer_grouping_version": 2,
        }
        upgraded_manifest = write_vulkan_training_replay(
            package_dir,
            replay_checkpoint,
        )
        if upgraded_manifest["format"] != VULKAN_TRAINING_FORMAT:
            raise AssertionError("portable replay did not upgrade the package manifest")
        native_session = upgraded_manifest.get("training_session")
        if not isinstance(native_session, dict) or native_session.get("format") != VULKAN_TRAINING_SESSION_FORMAT:
            raise AssertionError("portable replay did not promote trajectory state into native session")
        if native_session["ltm_lr_scheduler"] != replay_checkpoint["ltm_scheduler_state"]:
            raise AssertionError("LTM scheduler state was not promoted into native session")
        if native_session["main_lr_scheduler"]["step"] != 27:
            raise AssertionError("native session clamped the post-horizon LambdaLR counter")
        if native_session["main_lr_scheduler"]["last_lrs"] != [2.5e-5, 2.5e-5]:
            raise AssertionError("native session lost the live optimizer-group LRs")
        if native_session["data_stream_cursor"] != replay_checkpoint["data_stream_cursor"]:
            raise AssertionError("native session lost the backend-neutral data cursor")
        if native_session["execution_policy"] != replay_checkpoint["execution_policy"]:
            raise AssertionError("native session lost the execution/loss-scaling policy")
        raw_replay_text = (package_dir / "training_replay.json").read_text(encoding="utf-8")
        for replay_only_key in (
            "effective_training_config",
            "scheduler_state_dict",
            "lr_scheduler_state",
            "ltm_scheduler_state",
            "error_budget_state",
            "optimizer_grouping_version",
            "rng_state",
            "data_state",
            "scaler_state_dict",
        ):
            if replay_only_key in raw_replay_text:
                raise AssertionError(
                    f"trajectory state {replay_only_key!r} leaked back into host replay"
                )
        replay_roundtrip = read_vulkan_training_replay(package_dir, upgraded_manifest)
        if replay_roundtrip is None:
            raise AssertionError("portable replay unexpectedly decoded as absent")
        if replay_roundtrip["completed_epoch"] != 2 or replay_roundtrip["mid_epoch_step"] != 3:
            raise AssertionError("portable replay cursor did not round-trip")
        if replay_roundtrip["ltm_scheduler_state"] != replay_checkpoint["ltm_scheduler_state"]:
            raise AssertionError("native LTM scheduler state did not materialize for PyTorch resume")
        if replay_roundtrip["scheduler_state_dict"] != replay_checkpoint["scheduler_state_dict"]:
            raise AssertionError("native main LR state did not reconstruct exact LambdaLR live state")
        if replay_roundtrip["error_budget_state"] != replay_checkpoint["error_budget_state"]:
            raise AssertionError("native skipped-batch error budget did not round-trip")
        if "rng_state" in replay_roundtrip or "data_state" in replay_roundtrip:
            raise AssertionError("typed Vulkan session unexpectedly rematerialized Python replay blobs")
        if replay_roundtrip["data_stream_cursor"] != replay_checkpoint["data_stream_cursor"]:
            raise AssertionError("typed data-stream cursor did not round-trip")
        if replay_roundtrip["execution_policy"] != replay_checkpoint["execution_policy"]:
            raise AssertionError("typed execution policy did not round-trip")
        roundtrip_rosa = replay_roundtrip["running_states"][5][3][0]
        if not isinstance(roundtrip_rosa, ROSAState) or roundtrip_rosa.transitions != replay_rosa.transitions:
            raise AssertionError("ROSA recurrent replay state did not round-trip")

        control_model = HierarchosCore(config).cpu().train()
        control_model.load_state_dict(model.state_dict())
        control_optimizer = _optimizer(control_model)
        load_vulkan_adamw_into_torch(control_model, control_optimizer, exported)
        _assign_gradient_payload(control_model, pending)

        cross_model = HierarchosCore(config).cpu().train()
        cross_model.load_state_dict(model.state_dict())
        cross_optimizer = _optimizer(cross_model)
        package = load_vulkan_training_package_into_torch(
            cross_model,
            cross_optimizer,
            package_dir,
        )
        if package.pytorch_accumulation_normalization != "weighted-token":
            raise AssertionError("Vulkan weighted-token normalization did not map into PyTorch")
        if package.parameter_state != _portable_parameter_state():
            raise AssertionError("portable FP32 master/derived-mirror contract changed at PyTorch handoff")
        if package.consumed_weighted_token_mass != consumed_mass:
            raise AssertionError("Vulkan consumed supervision mass changed at PyTorch handoff")
        if package.target_weighted_token_mass != target_mass:
            raise AssertionError("Vulkan target supervision mass changed at PyTorch handoff")
        if package.replay_state is None or package.replay_state["mid_epoch_step"] != 3:
            raise AssertionError("Vulkan package loader did not carry portable replay state")
        if package.session_state is None or package.session_state["ltm_lr_scheduler"]["step"] != 7:
            raise AssertionError("Vulkan package loader did not expose native training session")
        for name, parameter in _canonical_parameters(cross_model).items():
            torch.testing.assert_close(
                parameter.grad,
                pending[name].reshape_as(parameter),
                rtol=0.0,
                atol=0.0,
            )
        if cross_model.lm_head.weight is not cross_model.tok_emb.weight:
            raise AssertionError("test model unexpectedly lost tied lm_head/tok_emb storage")
        if cross_model.lm_head.weight.grad is not cross_model.tok_emb.weight.grad:
            raise AssertionError("canonical lm_head pending gradient did not land on tied PyTorch grad")

        untied_model = HierarchosCore(config).cpu().train()
        untied_model.load_state_dict(model.state_dict())
        untied_model.lm_head.weight = torch.nn.Parameter(
            untied_model.lm_head.weight.detach().clone()
        )
        untied_optimizer = _optimizer(untied_model)
        try:
            load_vulkan_training_package_into_torch(
                untied_model,
                untied_optimizer,
                package_dir,
            )
        except ValueError as exc:
            if "must share one master object" not in str(exc):
                raise
        else:
            raise AssertionError("portable alias contract accepted an untied PyTorch model")

        half_model = HierarchosCore(config).cpu().half().train()
        half_model.load_state_dict(model.state_dict())
        half_optimizer = _optimizer(half_model)
        try:
            load_vulkan_training_package_into_torch(
                half_model,
                half_optimizer,
                package_dir,
            )
        except ValueError as exc:
            if "resolved to PyTorch dtype" not in str(exc):
                raise
        else:
            raise AssertionError("portable FP32 master contract accepted FP16 PyTorch parameters")

        bad_master_package = temp / "bad-master-package"
        shutil.copytree(package_dir, bad_master_package)
        bad_master_tensors = {
            name: parameter.detach().cpu().float().contiguous().clone()
            for name, parameter in _canonical_parameters(model).items()
        }
        bad_master_tensors["lm_head.weight"] = bad_master_tensors["lm_head.weight"].half()
        save_file(
            bad_master_tensors,
            str(bad_master_package / "model.safetensors"),
            metadata={"format": "hierarchos-rust-fp32-v1"},
        )
        bad_master_model = HierarchosCore(config).cpu().train()
        bad_master_model.load_state_dict(model.state_dict())
        bad_master_optimizer = _optimizer(bad_master_model)
        try:
            load_vulkan_training_package_into_torch(
                bad_master_model,
                bad_master_optimizer,
                bad_master_package,
            )
        except ValueError as exc:
            if "stored as F16" not in str(exc):
                raise
        else:
            raise AssertionError("portable FP32 master file contract accepted an F16 optimizer slot")

        closing = _gradient_payload(model, 7.0e-3)
        _add_gradient_payload(control_model, closing)
        _add_gradient_payload(cross_model, closing)
        _divide_gradients(control_model, target_mass)
        _divide_gradients(cross_model, target_mass)
        control_optimizer.step()
        cross_optimizer.step()
        for name, control_parameter in _canonical_parameters(control_model).items():
            torch.testing.assert_close(
                _canonical_parameters(cross_model)[name],
                control_parameter,
                rtol=0.0,
                atol=0.0,
            )

        # Prove a second PyTorch optimizer can continue from the bridge output
        # without trajectory drift.
        resumed_model = HierarchosCore(config).cpu().train()
        resumed_model.load_state_dict(model.state_dict())
        resumed_optimizer = _optimizer(resumed_model)
        resumed = load_vulkan_adamw_into_torch(resumed_model, resumed_optimizer, exported)
        if resumed.step != 5:
            raise AssertionError("round-tripped PyTorch optimizer did not restore step 5")
        _assign_gradients(model, 7.0e-3)
        _assign_gradients(resumed_model, 7.0e-3)
        optimizer.step()
        resumed_optimizer.step()
        torch.testing.assert_close(
            resumed_model.lm_head.weight,
            model.lm_head.weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            resumed_model.out_norm.weight,
            model.out_norm.weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            resumed_model.out_norm.bias,
            model.out_norm.bias,
            rtol=0.0,
            atol=0.0,
        )
        reread = read_vulkan_adamw_checkpoint(exported)
        if reread.slot_names != SLOT_NAMES:
            raise AssertionError("SafeTensors bridge did not preserve canonical slot names")

    print("tied_parameter_alias=tok_emb.weight -> lm_head.weight: PASS")
    print("vulkan_optimizer_to_pytorch=PASS")
    print("pytorch_optimizer_to_vulkan_rust_reader=PASS")
    print("vulkan_mid_window_to_pytorch=bit-exact")
    print("portable_parameter_master_mirror_contract=PASS")
    print("portable_fp32_master_runtime_dtype_contract=PASS")
    print("portable_fp32_master_file_dtype_contract=PASS")
    print("portable_tied_parameter_alias_contract=PASS")
    print("vulkan_typed_stream_execution_replay=PASS")
    print("pytorch_resume_trajectory=bit-exact")
    print("Vulkan <-> PyTorch AdamW optimizer bridge: PASS")


if __name__ == "__main__":
    main()
