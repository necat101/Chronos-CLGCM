import unittest
import hashlib
import os
import tempfile

import torch
from safetensors.torch import save_file

from hierarchos.models.quantized import (
    detect_quantized_rwkv_format,
    load_quantized,
    validate_quantized_rwkv_format,
)
from hierarchos.utils.checkpoint import (
    _infer_arch_flags_from_state_dict,
    load_checkpoint_payload_compatible,
    _reject_unsupported_rwkv_state_dict,
    _resolve_weights_path,
)


class ArchitectureConfigFlagTests(unittest.TestCase):
    def test_infers_deepembed_and_rosa_from_legacy_state_dict(self):
        config = {}
        state = {
            "h_deepemb.weight": torch.empty(2, 4),
            "l_deepemb.weight": torch.empty(2, 4),
            "rosa_emb.weight": torch.empty(3, 2),
            "rosa_gate_logit": torch.empty(()),
            "h_rnn.r_k": torch.empty(7, 64),
        }

        _infer_arch_flags_from_state_dict(config, state)

        self.assertTrue(config["use_deepembed"])
        self.assertTrue(config["use_rosa"])
        self.assertEqual(config["rosa_max_context"], 512)
        self.assertEqual(config["rwkv_head_size"], 64)
        self.assertEqual(config["rwkv_channel_mix_key_clamp"], 12.0)
        self.assertEqual(config["rwkv_channel_mix_deepembed_clamp"], 4.0)

    def test_infers_disabled_when_legacy_state_dict_has_no_optional_modules(self):
        config = {}
        state = {"tok_emb.weight": torch.empty(2, 4)}

        _infer_arch_flags_from_state_dict(config, state)

        self.assertFalse(config["use_deepembed"])
        self.assertFalse(config["use_rosa"])
        self.assertNotIn("rosa_max_context", config)

    def test_infers_coherent_shared_adapters_from_state_dict(self):
        config = {}
        state = {
            "h_deepembed_adapter.down.weight": torch.empty(2, 4),
            "l_deepembed_adapter.down.weight": torch.empty(2, 4),
            "rosa_adapter.down.weight": torch.empty(2, 4),
            "rosa_gate_logit": torch.empty(()),
            "h_rnn.r_k": torch.empty(7, 4),
        }

        _infer_arch_flags_from_state_dict(config, state)

        self.assertEqual(config["architecture_revision"], "coherent-v9")
        self.assertEqual(config["deepembed_mode"], "shared-factorized")
        self.assertEqual(config["rosa_embedding_mode"], "shared-factorized")
        self.assertTrue(config["use_deepembed"])
        self.assertTrue(config["use_rosa"])

    def test_rejects_legacy_scalar_rwkv_checkpoint(self):
        state = {
            "h_rnn.time_decay": torch.empty(4),
            "h_rnn.time_mix_k": torch.empty(1, 1, 4),
            "l_rnn.time_decay": torch.empty(4),
            "rosa_emb.weight": torch.empty(8, 4),
        }

        with self.assertRaisesRegex(ValueError, "v8-only"):
            _reject_unsupported_rwkv_state_dict(state, "legacy.pt")

    def test_does_not_override_explicit_config(self):
        config = {
            "use_deepembed": False,
            "use_rosa": True,
            "rosa_max_context": 128,
            "rwkv_channel_mix_key_clamp": 8.0,
            "rwkv_channel_mix_deepembed_clamp": 2.0,
        }
        state = {
            "h_deepemb.weight": torch.empty(2, 4),
            "rosa_emb.weight": torch.empty(3, 2),
        }

        _infer_arch_flags_from_state_dict(config, state)

        self.assertFalse(config["use_deepembed"])
        self.assertTrue(config["use_rosa"])
        self.assertEqual(config["rosa_max_context"], 128)
        self.assertEqual(config["rwkv_channel_mix_key_clamp"], 8.0)
        self.assertEqual(config["rwkv_channel_mix_deepembed_clamp"], 2.0)

    def test_resolve_weights_prefers_newest_known_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = os.path.join(tmp, "hierarchos.pt")
            new_path = os.path.join(tmp, "model.pt")
            torch.save({"config": {}, "model_state_dict": {}}, old_path)
            torch.save({"config": {}, "model_state_dict": {}}, new_path)
            os.utime(old_path, (1000, 1000))
            os.utime(new_path, (2000, 2000))

            resolved, model_dir = _resolve_weights_path(tmp)

        self.assertEqual(os.path.normcase(resolved), os.path.normcase(new_path))
        self.assertEqual(os.path.normcase(model_dir), os.path.normcase(tmp))

    def test_resolve_weights_accepts_native_model_safetensors_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.safetensors")
            save_file({"weight": torch.ones(1)}, model_path)

            resolved, model_dir = _resolve_weights_path(tmp)

        self.assertEqual(os.path.normcase(resolved), os.path.normcase(model_path))
        self.assertEqual(os.path.normcase(model_dir), os.path.normcase(tmp))

    def test_safetensors_loader_honors_sha256_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.safetensors")
            save_file({"weight": torch.tensor([1.25])}, model_path)
            with open(model_path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            with open(model_path + ".sha256", "w", encoding="utf-8") as handle:
                handle.write(digest + "\n")

            loaded = load_checkpoint_payload_compatible(model_path)
            self.assertTrue(torch.equal(loaded["weight"], torch.tensor([1.25])))

            with open(model_path + ".sha256", "w", encoding="utf-8") as handle:
                handle.write("0" * 64 + "\n")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 verification failed"):
                load_checkpoint_payload_compatible(model_path)

    def test_quantized_loader_rejects_every_legacy_npz_architecture(self):
        legacy_q = {"h_rnn.time_decay": object(), "h_rnn.time_mix_k": object()}
        v8_q = {"h_rnn.x_r": object(), "h_rnn.r_k": object()}
        mixed_q = {"h_rnn.time_decay": object(), "h_rnn.x_r": object()}

        self.assertEqual(detect_quantized_rwkv_format(legacy_q), "legacy-scalar")
        self.assertEqual(detect_quantized_rwkv_format(v8_q), "v8-matrix")
        self.assertEqual(detect_quantized_rwkv_format(mixed_q), "mixed")
        for archive, source in (
            (legacy_q, "legacy.npz"),
            (v8_q, "v8.npz"),
            (mixed_q, "mixed.npz"),
            ({}, "unknown.npz"),
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    ValueError,
                    r"Quantized \.npz inference is intentionally unsupported",
                ):
                    validate_quantized_rwkv_format(archive, source)

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "legacy.npz"), "wb"):
                pass
            with self.assertRaisesRegex(
                ValueError,
                r"Quantized \.npz inference is intentionally unsupported",
            ):
                load_quantized(tmp, device="cpu")


if __name__ == "__main__":
    unittest.main()
