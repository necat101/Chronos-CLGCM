import sys
sys.path.insert(0, r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main')

from hierarchos import HierarchosCore, AttrDict
from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

# Load checkpoint config
ckpt = load_checkpoint_payload_compatible(
    r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main\rog_ally_model\hierarchos_epoch_1.pt',
    map_location='cpu',
)
old_config = ckpt['config']

print("=" * 70)
print("ARCHITECTURE COMPARISON")
print("=" * 70)

# Create model with old config
old_model = HierarchosCore(old_config)
old_params = sum(p.numel() for p in old_model.parameters())

print(f"\n1. OLD Architecture (from checkpoint):")
print(f"   Total Parameters: {old_params:,}")

# Get state dict keys from checkpoint
checkpoint_keys = set(ckpt['model_state_dict'].keys())
current_keys = set(old_model.state_dict().keys())

print(f"\n2. Missing keys (in checkpoint but not in current code):")
missing = checkpoint_keys - current_keys
if missing:
    for key in sorted(missing):
        print(f"   - {key}")
else:
    print("   None")

print(f"\n3. Extra keys (in current code but not in checkpoint):")
extra = current_keys - checkpoint_keys
if extra:
    for key in sorted(extra):
        tensor = old_model.state_dict()[key]
        params = tensor.numel()
        print(f"   - {key:50s} {params:>10,} params")
else:
    print("   None")

# Calculate total extra parameters
if extra:
    total_extra = sum(old_model.state_dict()[key].numel() for key in extra)
    print(f"\n   Total extra parameters: {total_extra:,}")
    print(f"   This explains the difference: {old_params:,} vs expected 24,782,210")

print("\n" + "=" * 70)
