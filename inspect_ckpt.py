import sys
from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

ckpt_path = r"C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main\rog_ally_model\hierarchos_epoch_31.pt"
try:
    checkpoint = load_checkpoint_payload_compatible(ckpt_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    print(f"Keys: {list(state_dict.keys())[:20]}...")
    if 'persistent' in state_dict:
        print(f"persistent shape: {state_dict['persistent'].shape}")
    if 'in_proj.weight' in state_dict:
        print(f"in_proj.weight shape: {state_dict['in_proj.weight'].shape}")
    if 'tok_emb.weight' in state_dict:
        print(f"tok_emb.weight shape: {state_dict['tok_emb.weight'].shape}")
    if 'config' in checkpoint:
        print(f"Config: {checkpoint['config']}")
except Exception as e:
    print(f"Error: {e}")
