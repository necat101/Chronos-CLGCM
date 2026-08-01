import sys
sys.path.insert(0, r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main')

from hierarchos import HierarchosCore, AttrDict
from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

# Load checkpoint
ckpt = load_checkpoint_payload_compatible(
    r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main\rog_ally_model\hierarchos_epoch_31.pt',
    map_location='cpu',
)
config_dict = ckpt['config']

print("=" * 70)
print("DETAILED CHECKPOINT CONFIG ANALYSIS")
print("=" * 70)

# Print ALL config keys and values
print("\nFull checkpoint config:")
for key in sorted(config_dict.keys()):
    value = config_dict[key]
    if key in ['context_dim', 'persistent_dim', 'h_hidden', 'l_hidden', 'ltm_key_dim', 'ltm_val_dim', 'ltm_slots']:
        print(f"  [ARCH] {key:30s} = {value}")
    else:
        if value is not None and value != False and value != "":
            print(f"         {key:30s} = {value}")

# Create model and count params
print("\n" + "=" * 70)
print("MODEL INITIALIZATION TEST")
print("=" * 70)

# Test 1: Create model EXACTLY as checkpoint config (using AttrDict wrapper)
config_attr = AttrDict(config_dict)
model1 = HierarchosCore(config_attr)
params1 = sum(p.numel() for p in model1.parameters())
print(f"\n1. Model from checkpoint config (AttrDict):  {params1:,} params")

# Test 2: Verify consistency with AttrDict (duplicate check)
model2 = HierarchosCore(config_attr)
params2 = sum(p.numel() for p in model2.parameters())
print(f"2. Model from checkpoint config (AttrDict 2): {params2:,} params")

# Test 3: Check what happens with compile flag
config_with_compile = AttrDict(config_dict.copy())
config_with_compile['compile'] = True
model3 = HierarchosCore(config_with_compile)
params3 = sum(p.numel() for p in model3.parameters())
print(f"3. Model with compile=True: {params3:,} params")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if params1 == params2 == params3:
    print(f"[OK] All tests show consistent param count: {params1:,}")
    print(f"[WARN] But training shows: 44,080,898")
    print(f"\n[INFO] This means the checkpoint config itself might be corrupted")
    print(f"    OR there's a mismatch in how config is loaded during training")
else:
    print(f"[ERROR] Inconsistent param counts detected!")
    print(f"   This suggests initialization is non-deterministic")

# Check actual state dict size
state_dict_params = sum(p.numel() for k, p in ckpt['model_state_dict'].items() if 'ltm_deltas' not in k)
print(f"\n[STATS] Actual checkpoint state_dict params: {state_dict_params:,}")
print(f"   (This is the REAL size saved in the checkpoint)")

print("\n" + "=" * 70)
