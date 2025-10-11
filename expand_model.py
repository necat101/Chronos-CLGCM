import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
from tqdm import tqdm
# We import your model definition and optimizer
from chronos import ChronosCore, AttrDict, ADAM_OPTIMIZER 

def transplant_weights(old_model_path: str, new_config: dict, output_path: str, device: str, starting_lr: float, min_lr: float):
    """
    Loads weights from a smaller, trained model into a new, larger model
    and creates a full training checkpoint with a fresh optimizer and scheduler.
    """
    print(f"Loading old model from: {old_model_path}")
    checkpoint = torch.load(old_model_path, map_location=device)
    old_state_dict = checkpoint['model_state_dict']
    old_config = AttrDict(checkpoint.get('config', {}))
    
    # Ensure vocab size is present in the new config if it was in the old one
    if 'vocab_size' in old_config and 'vocab_size' not in new_config:
        new_config['vocab_size'] = old_config.vocab_size

    print("Initializing new, larger model...")
    new_model = ChronosCore(new_config).to(device)
    new_state_dict = new_model.state_dict()

    print("Transplanting weights...")
    for name, new_param in tqdm(new_state_dict.items()):
        if name in old_state_dict:
            old_param = old_state_dict[name]
            if new_param.shape == old_param.shape:
                # If shapes match, copy directly
                with torch.no_grad():
                    new_param.copy_(old_param)
            else:
                # If shapes differ, copy the slice that fits
                print(f"  - Slicing weights for '{name}' (new: {list(new_param.shape)}, old: {list(old_param.shape)})")
                with torch.no_grad():
                    slices = tuple(slice(0, dim) for dim in old_param.shape)
                    new_param[slices].copy_(old_param)
        else:
            print(f"  - Layer '{name}' not found in old model. Retaining new initialization.")

    new_model.load_state_dict(new_state_dict)
    
    # <<< START: CORE FIX >>>
    print("\nCreating new optimizer and scheduler states for the expanded model...")
    # Create a new optimizer for the new model
    new_optimizer = ADAM_OPTIMIZER(new_model.parameters(), lr=starting_lr)
    
    # Create a new scheduler. Here we're assuming a large T_max as it will be
    # re-calculated by the training script anyway. We just need a valid state.
    new_scheduler = CosineAnnealingLR(new_optimizer, T_max=10000, eta_min=min_lr)
    print("  - Optimizer and scheduler created.")
    # <<< END: CORE FIX >>>

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving expanded TRAINING CHECKPOINT to: {output_path}")
    # <<< MODIFIED: Save a complete training checkpoint, not just an inference model >>>
    torch.save({
        'completed_epoch': 0, # Reset epoch count
        'model_state_dict': new_model.state_dict(),
        'optimizer_state_dict': new_optimizer.state_dict(),
        'scheduler_state_dict': new_scheduler.state_dict(),
        'config': new_model.config
    }, output_path)
    
    print("✅ Model expansion complete. You can now resume training from this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand a trained Chronos model to a larger architecture and create a new training checkpoint.")
    parser.add_argument("--old-model-path", type=str, required=True, help="Path to the trained .pt checkpoint of the smaller model.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the new, expanded .pt training checkpoint.")
    
    # Add the new, larger architecture arguments
    arch_group = parser.add_argument_group('New Expanded Architecture')
    arch_group.add_argument("--context_dim", type=int, required=True)
    arch_group.add_argument("--persistent_dim", type=int)
    arch_group.add_argument("--ltm_slots", type=int)
    arch_group.add_argument("--ltm_key_dim", type=int)
    arch_group.add_argument("--ltm_val_dim", type=int)
    arch_group.add_argument("--h_hidden", type=int, required=True)
    arch_group.add_argument("--l_hidden", type=int, required=True)
    arch_group.add_argument("--max_length", type=int)

    # Add LR arguments to create a valid optimizer state
    optim_group = parser.add_argument_group('Optimizer Settings for New Checkpoint')
    optim_group.add_argument("--starting-lr", type=float, default=1e-4, help="The learning rate to save in the new optimizer state.")
    optim_group.add_argument("--min-lr", type=float, default=1e-6, help="The min learning rate for the new scheduler state.")
    
    args = parser.parse_args()
    
    device = "cpu" # Surgery is fine to do on the CPU
    
    # Load the old config to inherit any unchanged values
    old_checkpoint = torch.load(args.old_model_path, map_location=device)
    if 'config' not in old_checkpoint:
        raise ValueError("The old model checkpoint does not contain a 'config' dictionary. Cannot expand.")
    final_config = old_checkpoint['config'].copy()

    # Update the config with any new values from the command line
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key in final_config and value is not None:
            print(f"Updating config: '{key}' from '{final_config[key]}' to '{value}'")
            final_config[key] = value

    transplant_weights(args.old_model_path, final_config, args.output_path, device, args.starting_lr, args.min_lr)
