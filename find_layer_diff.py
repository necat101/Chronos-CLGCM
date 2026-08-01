import sys
sys.path.insert(0, r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main')

from hierarchos import HierarchosCore, AttrDict
from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

# Load checkpoint
ckpt = load_checkpoint_payload_compatible(
    r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main\rog_ally_model\hierarchos_epoch_1.pt',
    map_location='cpu',
)
checkpoint_state = ckpt['model_state_dict']
config = ckpt['config']

# Create model from config
model = HierarchosCore(config)
current_state = model.state_dict()

print("=" * 70)
print("LAYER-BY-LAYER PARAMETER COMPARISON")
print("=" * 70)

# Compare keys
checkpoint_keys = set(checkpoint_state.keys())
current_keys = set(current_state.keys())

print("\n1. EXTRA LAYERS (in current code, not in checkpoint):")
extra_keys = current_keys - checkpoint_keys
if extra_keys:
    total_extra = 0
    for key in sorted(extra_keys):
        params = current_state[key].numel()
        total_extra += params
        print(f"   {key:60s} {params:>12,} params")
    print(f"\n   TOTAL EXTRA: {total_extra:,} params")
else:
    print("   None")

print("\n2. MISSING LAYERS (in checkpoint, not in current code):")
missing_keys = checkpoint_keys - current_keys
if missing_keys:
    total_missing = 0
    for key in sorted(missing_keys):
        params = checkpoint_state[key].numel()
        total_missing += params
        print(f"   {key:60s} {params:>12,} params")
    print(f"\n   TOTAL MISSING: {total_missing:,} params")
else:
    print("   None")

print("\n3. DIMENSION MISMATCHES (same layer name, different size):")
common_keys = checkpoint_keys & current_keys
mismatches = []
for key in sorted(common_keys):
    ckpt_shape = checkpoint_state[key].shape
    curr_shape = current_state[key].shape
    if ckpt_shape != curr_shape:
        ckpt_params = checkpoint_state[key].numel()
        curr_params = current_state[key].numel()
        diff = curr_params - ckpt_params
        mismatches.append((key, ckpt_shape, curr_shape, diff))
        print(f"   {key:60s}")
        print(f"      Checkpoint: {ckpt_shape} ({ckpt_params:,} params)")
        print(f"      Current:    {curr_shape} ({curr_params:,} params)")
        print(f"      Difference: {diff:+,} params")

if not mismatches:
    print("   None")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
checkpoint_total = sum(p.numel() for p in checkpoint_state.values())
current_total = sum(p.numel() for p in current_state.values())
print(f"Checkpoint total: {checkpoint_total:,} params")
print(f"Current total:    {current_total:,} params")
print(f"Difference:       {current_total - checkpoint_total:+,} params")
print("=" * 70)
