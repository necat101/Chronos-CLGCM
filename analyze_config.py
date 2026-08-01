"""Analyze training config from checkpoint."""
from hierarchos.utils.checkpoint import load_checkpoint_payload_compatible

ckpt = load_checkpoint_payload_compatible(
    "./rog_ally_model/hierarchos_epoch_60.pt",
    map_location="cpu",
)
c = ckpt.get('config', {})

print('=== TRAINING CONFIG ===')
print(f'context_dim: {c.get("context_dim")}')
print(f'h_hidden: {c.get("h_hidden")}')
print(f'l_hidden: {c.get("l_hidden")}')
print(f'max_length: {c.get("max_length")}')
print(f'ltm_slots: {c.get("ltm_slots")}')
print(f'max_h_steps: {c.get("max_h_steps")}')
print(f'max_l_steps: {c.get("max_l_steps")}')
print(f'epochs: {c.get("epochs")}')
print(f'batch_size: {c.get("batch_size")}')
print(f'accumulation_steps: {c.get("accumulation_steps")}')
print(f'vocab_size: {c.get("vocab_size")}')
print(f'ltm_topk: {c.get("ltm_topk")}')
print(f'h_stride: {c.get("h_stride")}')

# Count parameters
sd = ckpt.get('model_state_dict', {})
total_params = sum(v.numel() for v in sd.values())
print(f'\nTotal params: {total_params:,} (~{total_params/1e6:.1f}M)')

# Show embedding size
if 'tok_emb.weight' in sd:
    print(f'Embedding shape: {sd["tok_emb.weight"].shape}')
