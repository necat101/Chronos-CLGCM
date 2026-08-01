import unittest
import os
import tempfile
import copy

import torch

import hierarchos
import hierarchos.models.rwkv_cell as rwkv_cell_module
from hierarchos import AttrDict, HierarchosCore
from hierarchos.inference.chat import load_hierarchical_chat_state, save_hierarchical_chat_state
from hierarchos.inference.chat_state import CHAT_STATE_VERSION
from hierarchos.models.core import _resolve_compile_kwargs
from hierarchos.models.rwkv_cell import RWKVCell
from hierarchos.utils.checkpoint import load_full_model_with_config
from hierarchos.utils.rosa import precompute_rosa_ids_for_chunks


def _tiny_config():
    return AttrDict(
        vocab_size=64,
        context_dim=16,
        persistent_dim=8,
        ltm_slots=16,
        ltm_key_dim=8,
        ltm_val_dim=8,
        ltm_topk=2,
        h_hidden=16,
        l_hidden=16,
        max_h_steps=2,
        max_l_steps=2,
        h_stride=2,
        l_conv_atol=1e-4,
        commitment_threshold=0.05,
        use_deepembed=True,
        use_rosa=True,
        rosa_max_context=32,
        compile=False,
        gradient_checkpointing=False,
        detach_every_n_steps=16,
    )


class RWKVV8IntegrityTests(unittest.TestCase):
    def test_autocast_cached_low_rank_weights_preserve_forward_and_gradients(self):
        torch.manual_seed(123)
        optimized = RWKVCell(
            32,
            head_size=16,
            state_readout_mode="explicit-output",
        ).train()
        reference = copy.deepcopy(optimized)
        optimized_input = torch.randn(
            3,
            32,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        reference_input = optimized_input.detach().clone().requires_grad_(True)

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            optimized_out, optimized_state = optimized(optimized_input, None)
            optimized_loss = (
                optimized_out.float().square().mean()
                + optimized_state.float().square().mean()
            )
        optimized_loss.backward()

        cached_helper = rwkv_cell_module._parameter_matmul
        try:
            rwkv_cell_module._parameter_matmul = (
                lambda left, parameter: left
                @ parameter.to(dtype=left.dtype, device=left.device)
            )
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                reference_out, reference_state = reference(reference_input, None)
                reference_loss = (
                    reference_out.float().square().mean()
                    + reference_state.float().square().mean()
                )
            reference_loss.backward()
        finally:
            rwkv_cell_module._parameter_matmul = cached_helper

        torch.testing.assert_close(optimized_out, reference_out, rtol=0, atol=0)
        torch.testing.assert_close(
            optimized_state,
            reference_state,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            optimized_input.grad,
            reference_input.grad,
            rtol=0,
            atol=0,
        )
        for (name, parameter), (reference_name, reference_parameter) in zip(
            optimized.named_parameters(),
            reference.named_parameters(),
        ):
            self.assertEqual(name, reference_name)
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                rtol=0,
                atol=0,
            )

    def test_import_uses_modular_package_not_legacy_monolith(self):
        loaded = os.path.normcase(os.path.abspath(hierarchos.__file__))
        self.assertTrue(loaded.endswith(os.path.normcase(os.path.join("hierarchos", "__init__.py"))), loaded)

    def test_cuda_compile_kwargs_do_not_mix_mode_and_options(self):
        cfg = AttrDict(
            compile_mode="max-autotune",
            compile_dynamic=False,
            compile_cudagraphs=True,
            compile_backend=None,
        )

        kwargs, mode, cudagraphs = _resolve_compile_kwargs(cfg, "cuda")

        self.assertEqual(mode, "max-autotune")
        self.assertTrue(cudagraphs)
        self.assertEqual(kwargs["mode"], "max-autotune")
        self.assertNotIn("options", kwargs)

        cfg.compile_cudagraphs = False
        kwargs, mode, cudagraphs = _resolve_compile_kwargs(cfg, "cuda")
        self.assertEqual(mode, "max-autotune-no-cudagraphs")
        self.assertFalse(cudagraphs)
        self.assertEqual(kwargs["mode"], "max-autotune-no-cudagraphs")
        self.assertNotIn("options", kwargs)

    def test_cuda_compile_defaults_keep_autotune_without_cudagraphs(self):
        cfg = AttrDict(
            compile_mode="max-autotune-no-cudagraphs",
            compile_dynamic=False,
            compile_backend=None,
        )

        kwargs, mode, cudagraphs = _resolve_compile_kwargs(cfg, "cuda")

        self.assertEqual(mode, "max-autotune-no-cudagraphs")
        self.assertFalse(cudagraphs)
        self.assertEqual(kwargs["mode"], "max-autotune-no-cudagraphs")
        self.assertNotIn("options", kwargs)

    def test_conservative_448_default_stays_near_233m_with_real_rwkv_heads(self):
        cfg = AttrDict(
            vocab_size=50257,
            context_dim=448,
            max_length=1024,
            persistent_dim=128,
            ltm_slots=1024,
            ltm_key_dim=128,
            ltm_val_dim=128,
            ltm_lr=1e-3,
            ltm_topk=4,
            h_hidden=448,
            l_hidden=448,
            h_stride=4,
            max_h_steps=5,
            max_l_steps=5,
            l_conv_atol=1e-4,
            commitment_threshold=0.05,
            detach_every_n_steps=32,
            h_halt_thresh=0.9,
            gradient_checkpointing=False,
            compile=False,
            use_deepembed=True,
            use_rosa=True,
            rosa_max_context=512,
            rwkv_head_size=64,
        )

        with torch.device("meta"):
            model = HierarchosCore(cfg)

        self.assertEqual(model.h_rnn.head_size, 64)
        self.assertEqual(model.l_rnn.head_size, 64)
        self.assertEqual(model.h_rnn.n_head, 7)
        self.assertEqual(model.l_rnn.n_head, 7)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 232516229)

    def test_450_hidden_auto_head_does_not_collapse_to_scalar_state(self):
        cell = RWKVCell(450)
        self.assertEqual(cell.head_size, 75)
        self.assertEqual(cell.n_head, 6)
        self.assertGreaterEqual(cell.state_size, 3 + 64)

    def test_explicit_output_cell_rejects_wrong_runtime_state_layout(self):
        cell = RWKVCell(
            4,
            head_size=2,
            state_readout_mode="explicit-output",
        )
        self.assertFalse(cell.allow_legacy_state_migration)

        x = torch.randn(1, 4)
        wrong_state = torch.zeros(1, 4, cell.state_size - 1)
        with self.assertRaisesRegex(ValueError, "strict 'explicit-output' contract"):
            cell(x, wrong_state)

    def test_legacy_input_cache_cell_still_migrates_old_runtime_state(self):
        cell = RWKVCell(
            4,
            head_size=2,
            state_readout_mode="legacy-input-cache",
        )
        self.assertTrue(cell.allow_legacy_state_migration)

        x = torch.randn(1, 4)
        old_state = torch.zeros(1, 4, cell.state_size - 1)
        output, migrated_state = cell(x, old_state)

        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertEqual(
            tuple(migrated_state.shape),
            (1, 4, cell.state_size),
        )

    def test_rwkv_cell_is_torch_compile_graph_capture_compatible(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is not available")

        torch.manual_seed(5)
        cell = RWKVCell(16, head_size=16)
        cell.train()

        try:
            compiled_cell = torch.compile(cell, backend="eager")
        except Exception as exc:
            self.skipTest(f"torch.compile is not available in this runtime: {exc}")

        x = torch.randn(2, 16, requires_grad=True)
        state = cell.initial_state(2, device=x.device)
        deepemb = torch.ones(2, 64)

        y, new_state = compiled_cell(x, state, timestep=1, deepemb_vec=deepemb)
        loss = y.float().square().mean() + new_state.float().square().mean() * 1e-4
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rwkv_channel_mix_key_clamp_keeps_squared_relu_finite(self):
        torch.manual_seed(23)
        cell = RWKVCell(4, head_size=2, channel_mix_key_clamp=2.0)
        cell.eval()

        with torch.no_grad():
            cell.key_cm.weight.zero_()
            cell.key_cm.weight[:, 0] = 1e20
            cell.value_cm.weight.fill_(1.0)

            x = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
            state = cell.initial_state(1)
            y, new_state = cell(x, state)

        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(new_state).all())
        self.assertLess(float(y.abs().max().item()), 1000.0)

    def test_rwkv_deepembed_clamp_keeps_channel_mix_input_finite(self):
        torch.manual_seed(31)
        cell = RWKVCell(
            4,
            head_size=2,
            channel_mix_key_clamp=2.0,
            channel_mix_deepembed_clamp=3.0,
        )
        cell.eval()

        with torch.no_grad():
            cell.key_cm.weight.zero_()
            cell.key_cm.weight[:, 0] = 1e20
            cell.value_cm.weight.fill_(1.0)

            x = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
            state = cell.initial_state(1)
            deepemb = torch.full((1, 16), 1e20)
            y, new_state = cell(x, state, deepemb_vec=deepemb)

        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(new_state).all())
        self.assertLess(float(y.abs().max().item()), 1000.0)

    def test_legacy_checkpoint_backfills_channel_mix_key_clamp(self):
        torch.manual_seed(29)
        cfg = _tiny_config()
        model = HierarchosCore(cfg)
        legacy_cfg = dict(cfg)
        legacy_cfg.pop("rwkv_channel_mix_key_clamp", None)
        legacy_cfg.pop("rwkv_channel_mix_deepembed_clamp", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": legacy_cfg,
                    "training_complete": True,
                },
                os.path.join(tmpdir, "hierarchos.pt"),
            )
            loaded, loaded_cfg = load_full_model_with_config(tmpdir, "cpu")

        self.assertEqual(loaded_cfg.rwkv_channel_mix_key_clamp, 12.0)
        self.assertEqual(loaded_cfg.rwkv_channel_mix_deepembed_clamp, 4.0)
        self.assertEqual(loaded.h_rnn.channel_mix_key_clamp, 12.0)
        self.assertEqual(loaded.l_rnn.channel_mix_key_clamp, 12.0)
        self.assertEqual(loaded.h_rnn.channel_mix_deepembed_clamp, 4.0)
        self.assertEqual(loaded.l_rnn.channel_mix_deepembed_clamp, 4.0)

        loaded.config.rwkv_channel_mix_key_clamp = 5.0
        loaded.config.rwkv_channel_mix_deepembed_clamp = 2.0
        loaded.refresh_runtime_config()
        self.assertEqual(loaded.h_rnn.channel_mix_key_clamp, 5.0)
        self.assertEqual(loaded.l_rnn.channel_mix_key_clamp, 5.0)
        self.assertEqual(loaded.h_rnn.channel_mix_deepembed_clamp, 2.0)
        self.assertEqual(loaded.l_rnn.channel_mix_deepembed_clamp, 2.0)

    def test_model_compile_hot_path_smoke_with_eager_backend(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is not available")

        torch.manual_seed(17)
        cfg = _tiny_config()
        cfg.compile = True
        cfg.force_compile = True
        cfg.compile_backend = "eager"
        cfg.compile_mode = "default"
        cfg.compile_cudagraphs = False
        cfg.compile_static_worker_loop = True
        cfg.compile_h_rnn = True

        model = HierarchosCore(cfg)
        model.train()
        model.compile()

        self.assertIsNotNone(model.h_rnn._compiled_impl)
        self.assertFalse(model.h_rnn.allow_legacy_state_migration)
        self.assertFalse(model.l_rnn.allow_legacy_state_migration)
        self.assertTrue(model.worker_loop_module.compile_static_worker_loop)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, 0] = -100
        out = model(input_ids, labels=labels)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertTrue(torch.isfinite(model.tok_emb.weight.grad).all())

    def test_cached_rosa_chunked_forward_matches_live_rosa_forward(self):
        torch.manual_seed(23)
        cfg = _tiny_config()
        cfg.rosa_max_context = 7
        model = HierarchosCore(cfg)
        model.eval()

        input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4, 1]], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, 0] = -100
        chunk_size = 4
        cached_rosa = torch.tensor(
            [
                precompute_rosa_ids_for_chunks(
                    input_ids[0].tolist(),
                    vocab_size=cfg.vocab_size,
                    chunk_size=chunk_size,
                    rosa_max_ctx=cfg.rosa_max_context,
                )
            ],
            dtype=torch.long,
        )

        live_states = dict(
            h_state=None,
            l_state=None,
            prev_context=None,
            target_context=None,
            drift_state=None,
            ltm_memory_state=None,
        )
        cached_states = dict(live_states)

        with torch.no_grad():
            for start in range(0, input_ids.shape[1], chunk_size):
                end = min(start + chunk_size, input_ids.shape[1])
                common_kwargs = dict(
                    input_ids=input_ids[:, start:end],
                    labels=labels[:, start:end],
                    global_pos_offset=start,
                    return_topk_values=False,
                )
                live = model(**common_kwargs, **live_states)
                cached = model(
                    **common_kwargs,
                    **cached_states,
                    rosa_ids=cached_rosa[:, start:end],
                )

                torch.testing.assert_close(cached["logits"], live["logits"], rtol=0, atol=0)
                torch.testing.assert_close(cached["loss"], live["loss"], rtol=0, atol=0)

                live_states = {
                    "h_state": live["h_state"],
                    "l_state": live["l_state"],
                    "prev_context": live["prev_context"],
                    "target_context": live["target_context"],
                    "drift_state": live["drift_state"],
                    "ltm_memory_state": live["ltm_memory_state"],
                }
                cached_states = {
                    "h_state": cached["h_state"],
                    "l_state": cached["l_state"],
                    "prev_context": cached["prev_context"],
                    "target_context": cached["target_context"],
                    "drift_state": cached["drift_state"],
                    "ltm_memory_state": cached["ltm_memory_state"],
                }

    def test_time_mix_uses_matrix_state_recurrence(self):
        torch.manual_seed(7)
        cell = RWKVCell(4, head_size=2)
        cell.eval()

        x = torch.randn(2, 4)
        state = cell.initial_state(batch_size=2, device=x.device)
        _, new_state = cell(x, state)

        with torch.no_grad():
            x_norm = cell.ln1(x)
            zero_prev = torch.zeros_like(x_norm)
            xk = cell._mix(x_norm, zero_prev, cell.x_k)
            xv = cell._mix(x_norm, zero_prev, cell.x_v)
            xa = cell._mix(x_norm, zero_prev, cell.x_a)
            k = cell.key(xk)
            v = cell.value(xv)
            a = torch.sigmoid(cell.a0 + (xa @ cell.a1) @ cell.a2)
            kk = k * cell.k_k
            kk = torch.nn.functional.normalize(kk.view(2, cell.n_head, cell.head_size), dim=-1, p=2.0).view(2, 4)
            k = k * (1 + (a - 1) * cell.k_a)
            expected = (
                v.float().view(2, cell.n_head, cell.head_size).unsqueeze(-1)
                * k.float().view(2, cell.n_head, cell.head_size).unsqueeze(-2)
            )

        actual = new_state[:, :, 3:].view(2, cell.n_head, cell.head_size, cell.head_size)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5), (actual, expected))

    def test_matrix_state_recurrence_matches_v7_reference_with_existing_state(self):
        torch.manual_seed(19)
        cell = RWKVCell(4, head_size=2)
        cell.eval()

        x = torch.randn(2, 4)
        state = cell.initial_state(batch_size=2, device=x.device)
        state[:, :, 0] = torch.randn(2, 4) * 0.2
        state[:, :, 1] = torch.randn(2, 4) * 0.2
        matrix_state = torch.randn(2, cell.n_head, cell.head_size, cell.head_size) * 0.05
        state[:, :, 3:] = matrix_state.reshape(2, 4, cell.head_size)

        _, new_state = cell(x, state)

        with torch.no_grad():
            prev_tm = state[:, :, 0]
            x_norm = cell.ln1(x)
            xw = cell._mix(x_norm, prev_tm, cell.x_w)
            xk = cell._mix(x_norm, prev_tm, cell.x_k)
            xv = cell._mix(x_norm, prev_tm, cell.x_v)
            xa = cell._mix(x_norm, prev_tm, cell.x_a)

            k = cell.key(xk)
            v = cell.value(xv)
            a = torch.sigmoid(cell.a0 + (xa @ cell.a1) @ cell.a2)
            w = -torch.nn.functional.softplus(
                -(cell.w0 + torch.tanh(xw @ cell.w1) @ cell.w2)
            ) - 0.5
            kk = k * cell.k_k
            kk = torch.nn.functional.normalize(
                kk.view(2, cell.n_head, cell.head_size), dim=-1, p=2.0
            ).view(2, 4)
            k = k * (1 + (a - 1) * cell.k_a)
            state_a = -kk.view(2, cell.n_head, cell.head_size)
            state_b = (kk * a).view(2, cell.n_head, cell.head_size)
            w_decay = torch.exp(-torch.exp(torch.clamp(w.float(), min=-60.0, max=30.0))).view(
                2, cell.n_head, cell.head_size
            )

            expected = matrix_state.float()
            sa = torch.matmul(expected, state_a.float().unsqueeze(-1)).squeeze(-1)
            expected = (
                expected * w_decay.unsqueeze(-2)
                + sa.unsqueeze(-1) * state_b.float().unsqueeze(-2)
                + v.float().view(2, cell.n_head, cell.head_size).unsqueeze(-1)
                * k.float().view(2, cell.n_head, cell.head_size).unsqueeze(-2)
            )

        actual = new_state[:, :, 3:].view(2, cell.n_head, cell.head_size, cell.head_size)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5), (actual, expected))

    def test_deepembed_4x_matches_wiki_relu_squared_formula(self):
        torch.manual_seed(11)
        cell = RWKVCell(2)
        cell.eval()

        with torch.no_grad():
            cell.output.weight.zero_()
            cell.key_cm.weight.zero_()
            cell.key_cm.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [0.0, 2.0],
                        [-1.0, 0.0],
                        [-2.0, 0.0],
                        [1.0, 1.0],
                        [-1.0, 1.0],
                        [0.5, -0.5],
                        [1.0, 0.0],
                    ]
                )
            )
            cell.value_cm.weight.copy_(
                torch.tensor(
                    [
                        [0.11, -0.07, 0.13, 0.19, -0.05, 0.17, -0.03, 0.23],
                        [-0.02, 0.29, -0.31, 0.37, 0.41, -0.43, 0.47, -0.53],
                    ]
                )
            )

        x = torch.tensor([[-1.0, 2.0]])
        state = torch.zeros(1, 2, 5)
        state[:, :, 3] = -1e30
        deepemb = torch.tensor([[1.0, 1.5, 0.5, 2.0, 3.0, 0.25, 4.0, 0.75]])

        actual, _ = cell(x, state, deepemb_vec=deepemb)

        x_norm2 = cell.ln2(x)
        key_out = cell.key_cm(x_norm2)
        expanded = torch.square(torch.relu(key_out)) * deepemb
        expected = x + cell.value_cm(expanded)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6), (actual, expected))

    def test_deepembed_and_rosa_receive_finite_gradients(self):
        torch.manual_seed(123)
        model = HierarchosCore(_tiny_config())
        model.train()
        with torch.no_grad():
            model.h_rnn.value_cm.weight.normal_(mean=0.0, std=0.02)
            model.l_rnn.value_cm.weight.normal_(mean=0.0, std=0.02)

        input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 3]], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, 0] = -100

        out = model(input_ids, labels=labels)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

        for name, param in [
            ("h_deepemb.weight", model.h_deepemb.weight),
            ("l_deepemb.weight", model.l_deepemb.weight),
            ("rosa_emb.weight", model.rosa_emb.weight),
        ]:
            self.assertIsNotNone(param.grad, f"{name} did not receive a gradient")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{name} gradient is not finite")
            self.assertGreater(param.grad.abs().sum().item(), 0.0, f"{name} gradient is zero")

        self.assertIsNotNone(model.rosa_gate_logit.grad)
        self.assertTrue(torch.isfinite(model.rosa_gate_logit.grad).all())

        # The default ROSA embedding is zero-initialized for a stable no-op start,
        # so the scalar gate can be zero-gradient on step 1. Seed the embedding
        # to prove the gate is connected once ROSA has learned any signal.
        model = HierarchosCore(_tiny_config())
        model.train()
        with torch.no_grad():
            model.rosa_emb.weight.normal_(mean=0.0, std=0.02)

        out = model(input_ids, labels=labels)
        out["loss"].backward()
        self.assertIsNotNone(model.rosa_gate_logit.grad)
        self.assertTrue(torch.isfinite(model.rosa_gate_logit.grad).all())
        self.assertGreater(model.rosa_gate_logit.grad.abs().item(), 0.0)
        self.assertIsNotNone(model.rosa_router.weight.grad)
        self.assertTrue(torch.isfinite(model.rosa_router.weight.grad).all())

    def test_memory_token_routers_start_as_scalar_gates_with_warmup_floor(self):
        torch.manual_seed(19)
        cfg = _tiny_config()
        cfg.memory_gate_warmup_steps = 100
        cfg.memory_gate_warmup_floor = 0.1
        model = HierarchosCore(cfg)
        model.train()

        x = torch.randn(2, 3, cfg.context_dim)
        rosa_gate = torch.sigmoid(model.rosa_gate_logit + model.rosa_router(x))
        expected_rosa = torch.full_like(rosa_gate, torch.sigmoid(model.rosa_gate_logit).item())
        self.assertTrue(torch.allclose(rosa_gate, expected_rosa, atol=1e-6))

        model.set_training_step(0)
        warmed = model._apply_memory_gate_warmup(torch.zeros(2, 1))
        self.assertTrue(torch.allclose(warmed, torch.full_like(warmed, 0.1), atol=1e-6))

        model.set_training_step(100)
        cooled = model._apply_memory_gate_warmup(torch.zeros(2, 1))
        self.assertTrue(torch.allclose(cooled, torch.zeros_like(cooled), atol=1e-6))

    def test_rwkv_cell_large_input_backward_stays_finite(self):
        torch.manual_seed(321)
        cell = RWKVCell(16)
        cell.train()

        batch = 2
        steps = 32
        state = torch.zeros(batch, 16, 5)
        state[:, :, 3] = -1e30
        xs = (torch.randn(steps, batch, 16) * 25.0).requires_grad_(True)
        deepemb = 1.0 + 0.05 * torch.randn(steps, batch, 64)

        outputs = []
        for t in range(steps):
            y, state = cell(xs[t], state, timestep=t, deepemb_vec=deepemb[t])
            outputs.append(y)

        loss = torch.stack(outputs).float().square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(xs.grad).all())
        for name, param in cell.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"{name} gradient is not finite")

    def test_chat_state_export_records_rwkv_v8_layout(self):
        torch.manual_seed(41)
        cfg = _tiny_config()
        model = HierarchosCore(cfg)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, return_topk_values=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-state.pt")
            save_hierarchical_chat_state(
                path,
                config=cfg,
                model=model,
                model_path=tmpdir,
                h_state=out["h_state"],
                l_state=out["l_state"],
                prev_context=out["prev_context"],
                target_context=out["target_context"],
                drift_state=out["drift_state"],
                ltm_state=out["ltm_memory_state"],
                total_tokens_generated=input_ids.shape[1],
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

            self.assertEqual(payload["version"], CHAT_STATE_VERSION)
            self.assertEqual(payload["recurrent_state_layout"]["h"]["layout"], "rwkv_v8_matrix_packed")
            self.assertEqual(payload["recurrent_state_layout"]["h"]["head_size"], model.h_rnn.head_size)
            self.assertEqual(payload["recurrent_state_layout"]["h"]["state_size"], model.h_rnn.state_size)
            self.assertEqual(payload["recurrent_state_shapes"]["h_state"], list(out["h_state"].shape))
            self.assertEqual(len(payload["architecture_contract_sha256"]), 64)
            self.assertIn("manager_compute_mode", payload["architecture_contract"])
            self.assertIn("manager_state_commit_mode", payload["architecture_contract"])

            restored = load_hierarchical_chat_state(path, config=cfg, device="cpu", model=model)
            self.assertEqual(tuple(restored["h_state"].shape), tuple(out["h_state"].shape))
            self.assertEqual(tuple(restored["l_state"].shape), tuple(out["l_state"].shape))

    def test_current_chat_state_rejects_manager_contract_drift(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        cfg.architecture_revision = "coherent-v9"
        model = HierarchosCore(cfg)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor([[1, 2]], dtype=torch.long), return_topk_values=False)

        other_cfg = _tiny_config()
        other_cfg.architecture_revision = "coherent-v9"
        other_cfg.manager_compute_mode = "soft-act"
        other_cfg.manager_state_commit_mode = "last-shadow"
        other_model = HierarchosCore(other_cfg)
        other_model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-state.pt")
            save_hierarchical_chat_state(
                path,
                config=cfg,
                model=model,
                model_path=tmpdir,
                h_state=out["h_state"],
                l_state=out["l_state"],
                prev_context=out["prev_context"],
                target_context=out["target_context"],
                drift_state=out["drift_state"],
                ltm_state=out["ltm_memory_state"],
                total_tokens_generated=2,
            )
            with self.assertRaisesRegex(RuntimeError, "Architecture contract mismatch"):
                load_hierarchical_chat_state(
                    path,
                    config=other_cfg,
                    device="cpu",
                    model=other_model,
                )

    def test_current_chat_state_rejects_shape_tampering_without_migration(self):
        torch.manual_seed(44)
        cfg = _tiny_config()
        model = HierarchosCore(cfg)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor([[1, 2]], dtype=torch.long), return_topk_values=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-state.pt")
            save_hierarchical_chat_state(
                path,
                config=cfg,
                model=model,
                model_path=tmpdir,
                h_state=out["h_state"],
                l_state=out["l_state"],
                prev_context=out["prev_context"],
                target_context=out["target_context"],
                drift_state=out["drift_state"],
                ltm_state=out["ltm_memory_state"],
                total_tokens_generated=2,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["h_state"] = payload["h_state"][:, :-1, :]
            payload["recurrent_state_shapes"]["h_state"] = list(
                payload["h_state"].shape
            )
            torch.save(payload, path)

            with self.assertRaisesRegex(RuntimeError, "tensor shape mismatch"):
                load_hierarchical_chat_state(
                    path,
                    config=cfg,
                    device="cpu",
                    model=model,
                )

    def test_chat_state_loader_migrates_legacy_five_slot_state_to_v8(self):
        torch.manual_seed(43)
        cfg = _tiny_config()
        model = HierarchosCore(cfg)
        model.eval()

        legacy_h = torch.zeros(1, cfg.h_hidden, 5)
        legacy_l = torch.zeros(1, cfg.l_hidden, 5)
        legacy_h[:, :, 0] = torch.randn(1, cfg.h_hidden)
        legacy_h[:, :, 4] = torch.randn(1, cfg.h_hidden)
        legacy_l[:, :, 0] = torch.randn(1, cfg.l_hidden)
        legacy_l[:, :, 4] = torch.randn(1, cfg.l_hidden)
        legacy_h[:, :, 3] = -1e30
        legacy_l[:, :, 3] = -1e30

        payload = {
            "version": 1,
            "kind": "hierarchos_chat_runtime_state",
            "config_signature": {
                "context_dim": cfg.context_dim,
                "h_hidden": cfg.h_hidden,
                "l_hidden": cfg.l_hidden,
                "vocab_size": cfg.vocab_size,
            },
            "h_state": legacy_h,
            "l_state": legacy_l,
            "prev_context": torch.zeros(1, cfg.context_dim),
            "target_context": torch.zeros(1, cfg.context_dim),
            "drift_state": torch.zeros(1, cfg.context_dim),
            "total_tokens_generated": 7,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy-chat-state.pt")
            torch.save(payload, path)
            restored = load_hierarchical_chat_state(path, config=cfg, device="cpu", model=model)

        self.assertEqual(tuple(restored["h_state"].shape), (1, cfg.h_hidden, model.h_rnn.state_size))
        self.assertEqual(tuple(restored["l_state"].shape), (1, cfg.l_hidden, model.l_rnn.state_size))
        torch.testing.assert_close(restored["h_state"][:, :, 0], legacy_h[:, :, 0])
        torch.testing.assert_close(restored["h_state"][:, :, 1], legacy_h[:, :, 4])
        torch.testing.assert_close(restored["l_state"][:, :, 0], legacy_l[:, :, 0])
        torch.testing.assert_close(restored["l_state"][:, :, 1], legacy_l[:, :, 4])
        self.assertEqual(restored["total_tokens_generated"], 7)

    def test_chat_state_loader_rejects_mismatched_v8_head_layout(self):
        torch.manual_seed(47)
        cfg = _tiny_config()
        model = HierarchosCore(cfg)
        model.eval()

        other_cfg = _tiny_config()
        other_cfg.rwkv_head_size = 8
        other_model = HierarchosCore(other_cfg)
        other_model.eval()
        self.assertNotEqual(model.h_rnn.state_size, other_model.h_rnn.state_size)

        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, return_topk_values=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chat-state.pt")
            save_hierarchical_chat_state(
                path,
                config=cfg,
                model=model,
                model_path=tmpdir,
                h_state=out["h_state"],
                l_state=out["l_state"],
                prev_context=out["prev_context"],
                target_context=out["target_context"],
                drift_state=out["drift_state"],
                ltm_state=out["ltm_memory_state"],
                total_tokens_generated=input_ids.shape[1],
            )

            with self.assertRaisesRegex(RuntimeError, "recurrent layout mismatch"):
                load_hierarchical_chat_state(path, config=other_cfg, device="cpu", model=other_model)


if __name__ == "__main__":
    unittest.main()
