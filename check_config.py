import argparse
import os
import json
from hierarchos.utils.checkpoint import (
    _has_v8_rwkv_state_dict,
    _reject_unsupported_rwkv_state_dict,
    load_checkpoint_payload_compatible,
    sanitize_model_state_dict,
)

DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rog_ally_model",
    "hierarchos_epoch_1.pt",
)

parser = argparse.ArgumentParser(description="Inspect a Hierarchos checkpoint config.")
parser.add_argument(
    "checkpoint",
    nargs="?",
    default=DEFAULT_CKPT_PATH,
    help="Path to a .pt checkpoint. Defaults to rog_ally_model/hierarchos_epoch_1.pt.",
)
args = parser.parse_args()

ckpt_path = os.path.abspath(os.path.expanduser(args.checkpoint))
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

ckpt = load_checkpoint_payload_compatible(ckpt_path, map_location="cpu")

config = ckpt.get('config', {})
state_dict = sanitize_model_state_dict(
    ckpt.get('model_state_dict', {}),
    reset_transient_ltm=False,
)

try:
    _reject_unsupported_rwkv_state_dict(state_dict, ckpt_path)
except ValueError as exc:
    print(f"CRITICAL: {exc}")
    raise SystemExit(1)

if not _has_v8_rwkv_state_dict(state_dict):
    print(
        "CRITICAL: No matrix-state RWKV v8 keys were found. "
        "Expected keys such as 'h_rnn.x_r' and 'h_rnn.r_k'."
    )
    raise SystemExit(1)

print("=" * 60)
print("CHECKPOINT CONFIGURATION")
print(f"Path: {ckpt_path}")
print("=" * 60)
print(json.dumps(config, indent=2) if isinstance(config, dict) else str(config))
print("=" * 60)
print("RWKV architecture: v8 matrix-state OK")

# Also check model state dict keys to understand structure
print("\nModel State Dict Keys (first 30):")
state_keys = list(state_dict.keys())[:30]
for key in state_keys:
    print(f"  {key}")
