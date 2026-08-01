import argparse
import os
import torch
from hierarchos.utils.checkpoint import (
    TRANSIENT_LTM_STATE_KEYS,
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

parser = argparse.ArgumentParser(description="Inspect Hierarchos checkpoint dimensions.")
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

state_dict = sanitize_model_state_dict(ckpt['model_state_dict'], reset_transient_ltm=False)
config = ckpt['config']

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

nonfinite_keys = []
for key, tensor in state_dict.items():
    clean_key = key.replace("_orig_mod.", "")
    if clean_key.endswith("ltm.neg_inf"):
        continue
    if any(clean_key.endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS):
        continue
    if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        nonfinite_keys.append(clean_key)

if nonfinite_keys:
    print(
        "CRITICAL: Non-finite persistent checkpoint tensors detected: "
        + ", ".join(nonfinite_keys[:10])
        + ("..." if len(nonfinite_keys) > 10 else "")
    )
    raise SystemExit(1)

print("=" * 70)
print("CHECKPOINT FILE ANALYSIS")
print(f"Path: {ckpt_path}")
print("=" * 70)

# Count params in state dict
total_params_all = sum(p.numel() for p in state_dict.values())
total_params_no_deltas = sum(p.numel() for k, p in state_dict.items() if 'ltm_deltas' not in k)

print(f"\nTotal params in checkpoint state_dict: {total_params_all:,}")
print(f"(excluding ltm_deltas buffer): {total_params_no_deltas:,}")
print("RWKV architecture: v8 matrix-state OK")
print("Persistent tensor finiteness: OK")

print("\nConfig dimensions:")
print(f"  context_dim:    {config['context_dim']}")
print(f"  h_hidden:       {config['h_hidden']}")
print(f"  l_hidden:       {config['l_hidden']}")
print(f"  persistent_dim: {config['persistent_dim']}")
print(f"  ltm_key_dim:    {config['ltm_key_dim']}")
print(f"  ltm_val_dim:    {config['ltm_val_dim']}")

# Show largest layers
print("\nTop 10 largest layers:")
sorted_layers = sorted(state_dict.items(), key=lambda x: x[1].numel(), reverse=True)[:10]
for name, tensor in sorted_layers:
    print(f"  {name:50s} {tensor.shape!s:30s} {tensor.numel():>12,} params")

print("\n" + "=" * 70)
