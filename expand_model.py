import torch
import argparse
import os
import json # MODIFICATION: Added for dataset scanning
from tqdm import tqdm
from transformers import AutoTokenizer # MODIFICATION: Added for tokenizer loading
from chronos import ChronosCore, AttrDict # We import your model definition

# --- MODIFICATION START: Add dataset scanning function (adapted from chronos.py) ---
def scan_dataset_for_max_length(dataset_path: str, tokenizer, kayla_mode: bool) -> int:
    """Scans a JSON or JSONL dataset to find the maximum token sequence length."""
    max_found_length = 0
    print(f"Scanning dataset '{dataset_path}' to determine max length...")

    if not os.path.exists(dataset_path):
         raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        def get_text_from_obj(obj, kayla_mode):
            try:
                if kayla_mode:
                    feelings_part = f"### Feelings:\n{obj.get('feelings')}\n\n" if obj.get('feelings') else ""
                    return (f"### Instruction:\n{obj.get('Instruction', '')}\n\n"
                            f"{feelings_part}"
                            f"### Thought Process:\n{obj.get('thought-process', '')}\n\n"
                            f"### Response:\n{obj.get('output', '')}")
                else:
                    return f"### Instruction:\n{obj.get('instruction', '')}\n\n### Response:\n{obj.get('output', '') or obj.get('response', '')}"
            except:
                return ""

        try:
            data = json.load(f)
            if isinstance(data, list):
                for obj in tqdm(data, desc="Scanning JSON"):
                    text = get_text_from_obj(obj, kayla_mode)
                    length = len(tokenizer.encode(text)) + 1 # Add 1 for EOS
                    if length > max_found_length: max_found_length = length
        except json.JSONDecodeError:
            f.seek(0)
            for line in tqdm(f, desc="Scanning JSONL"):
                try:
                    obj = json.loads(line)
                    text = get_text_from_obj(obj, kayla_mode)
                    length = len(tokenizer.encode(text)) + 1 # Add 1 for EOS
                    if length > max_found_length: max_found_length = length
                except (json.JSONDecodeError, AttributeError, TypeError):
                     continue # Skip malformed lines

    if max_found_length > 0:
        # Add a small buffer and round up to a multiple of 8 for efficiency
        adjusted_length = (max_found_length + 16 + 7) & -8
        print(f"✅ Auto-scan complete. Determined required max_length: {adjusted_length} (found max: {max_found_length}).")
        return adjusted_length
    else:
        print("⚠️ WARNING: Auto-scan did not find any valid entries.")
        return 0
# --- MODIFICATION END ---

def transplant_weights(old_model_path: str, new_config: dict, output_dir: str, device: str): # MODIFICATION: Changed output_path to output_dir
    """
    Loads weights from a smaller, trained model into a new, larger model,
    handling changes in dimensions and max_length. Saves as a model directory.
    """
    print(f"Loading old model directory: {old_model_path}")
    # --- MODIFICATION START: Load from directory ---
    old_weights_path = os.path.join(old_model_path, "chronos.pt")
    if not os.path.exists(old_weights_path):
        raise FileNotFoundError(f"'chronos.pt' not found in directory: {old_model_path}")
    checkpoint = torch.load(old_weights_path, map_location=device)
    old_state_dict = checkpoint['model_state_dict']
    old_config = AttrDict(checkpoint.get('config', {}))
    # --- MODIFICATION END ---

    # Ensure vocab size is present in the new config if it was in the old one
    if 'vocab_size' in old_config and 'vocab_size' not in new_config:
        new_config['vocab_size'] = old_config.vocab_size
    # --- MODIFICATION START: Ensure max_length consistency ---
    if 'max_length' not in new_config:
        new_config['max_length'] = old_config.max_length # Default to old if not specified/auto-detected
    print(f"Target max_length for new model: {new_config['max_length']}")
    # --- MODIFICATION END ---


    print("Initializing new, larger model...")
    new_model = ChronosCore(new_config).to(device)
    new_state_dict = new_model.state_dict()

    print("Transplanting weights...")
    for name, new_param in tqdm(new_state_dict.items()):
        if name in old_state_dict:
            old_param = old_state_dict[name]

            # --- MODIFICATION START: Special handling for pos_emb ---
            if name == "pos_emb.weight":
                old_len = old_param.shape[0]
                new_len = new_param.shape[0]
                if new_len >= old_len:
                    print(f"  - Transplanting positional embeddings (new: {new_len}, old: {old_len})")
                    new_state_dict[name][:old_len, :].copy_(old_param)
                else:
                    # Should not happen if expanding, but handle just in case
                    print(f"  - WARNING: New max_length ({new_len}) is smaller than old ({old_len}). Truncating positional embeddings.")
                    new_state_dict[name].copy_(old_param[:new_len, :])
                continue # Skip the general slicing logic below for pos_emb
            # --- MODIFICATION END ---

            if new_param.shape == old_param.shape:
                # If shapes match, copy directly
                new_state_dict[name].copy_(old_param)
            else:
                # If shapes differ (e.g., context_dim changed), copy the slice
                print(f"  - Slicing weights for '{name}' (new: {list(new_param.shape)}, old: {list(old_param.shape)})")
                try:
                    slices = [slice(0, min(new_dim, old_dim)) for new_dim, old_dim in zip(new_param.shape, old_param.shape)]
                    new_state_dict[name][tuple(slices)].copy_(old_param[tuple(slices)]) # Copy matching part
                except IndexError:
                     print(f"  - WARNING: Could not slice weights for '{name}'. Shapes might be incompatible beyond simple expansion. Retaining new initialization.")

        else:
            print(f"  - Layer '{name}' not found in old model. Retaining new initialization.")

    new_model.load_state_dict(new_state_dict)

    # --- MODIFICATION START: Save as a model directory ---
    print(f"\nSaving expanded model directory to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_weights_path = os.path.join(output_dir, "chronos.pt")
    torch.save({
        'model_state_dict': new_model.state_dict(),
        'config': new_model.config # Save the final config used
    }, output_weights_path)

    # Copy tokenizer files from old model directory
    try:
        print("Copying tokenizer files...")
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_path)
        old_tokenizer.save_pretrained(output_dir)
        print("✅ Tokenizer copied successfully.")
    except Exception as e:
         print(f"⚠️ WARNING: Could not load or save tokenizer from '{old_model_path}'. You may need to manually copy tokenizer files. Error: {e}")
    # --- MODIFICATION END ---

    print("✅ Model expansion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand a trained Chronos model to a larger architecture, including max_length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # MODIFICATION: Better help format
    )
    # MODIFICATION: Changed to model directory
    parser.add_argument("--old-model-path", type=str, required=True, help="Path to the trained model *directory* of the smaller model.")
    # MODIFICATION: Changed to output directory
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the new, expanded model *directory*.")

    # --- MODIFICATION START: Arguments for changing dimensions ---
    dim_group = parser.add_argument_group('Architecture Dimension Expansion (Optional)')
    dim_group.add_argument("--context_dim", type=int, help="New context dimension (embedding size).")
    dim_group.add_argument("--persistent_dim", type=int, help="New persistent memory dimension.")
    dim_group.add_argument("--ltm_slots", type=int, help="New number of LTM slots.")
    dim_group.add_argument("--ltm_key_dim", type=int, help="New LTM key dimension.")
    dim_group.add_argument("--ltm_val_dim", type=int, help="New LTM value dimension.")
    dim_group.add_argument("--h_hidden", type=int, help="New H-RNN hidden size.")
    dim_group.add_argument("--l_hidden", type=int, help="New L-RNN hidden size.")
    # --- MODIFICATION END ---

    # --- MODIFICATION START: Arguments for changing max_length ---
    length_group = parser.add_argument_group('Sequence Length Expansion (Optional)')
    length_group.add_argument("--new-max-length", type=int, help="Manually specify the new maximum sequence length.")
    length_group.add_argument("--auto-max-length", action="store_true", help="Automatically determine new max length by scanning a dataset (requires --dataset-for-length). Overrides --new-max-length if both are given.")
    length_group.add_argument("--dataset-for-length", type=str, help="Path to dataset (.jsonl or .json) to scan if using --auto-max-length.")
    length_group.add_argument("--kayla", action="store_true", help="Use Kayla formatting when scanning dataset with --auto-max-length.")
    # --- MODIFICATION END ---

    args = parser.parse_args()

    device = "cpu" # Surgery is fine to do on the CPU

    # --- Load the old config to get base values ---
    print(f"Loading configuration from old model: {args.old_model_path}")
    old_weights_path = os.path.join(args.old_model_path, "chronos.pt")
    if not os.path.exists(old_weights_path):
         raise FileNotFoundError(f"'chronos.pt' not found in directory: {args.old_model_path}")
    old_checkpoint = torch.load(old_weights_path, map_location=device)
    old_config = old_checkpoint.get('config', {})
    if not old_config:
        raise ValueError(f"Could not find 'config' dictionary in old model checkpoint: {old_weights_path}")

    final_config = old_config.copy() # Start with old config

    # --- Update dimensions based on CLI args ---
    updated_dims = {
        k: v for k, v in vars(args).items() if v is not None and k in [
            "context_dim", "persistent_dim", "ltm_slots", "ltm_key_dim",
            "ltm_val_dim", "h_hidden", "l_hidden"
        ]
    }
    if updated_dims:
        print("Updating model dimensions:")
        for k, v in updated_dims.items():
            print(f"  - {k}: {final_config.get(k, 'N/A')} -> {v}")
        final_config.update(updated_dims)

    # --- Determine and update max_length ---
    new_max_len = None
    if args.auto_max_length:
        if not args.dataset_for_length:
            parser.error("--auto-max-length requires --dataset-for-length to be specified.")
        try:
            # --- MODIFICATION START: Load tokenizer for scanning ---
            print(f"Loading tokenizer from old model path: {args.old_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # --- MODIFICATION END ---

            determined_len = scan_dataset_for_max_length(args.dataset_for_length, tokenizer, args.kayla)
            if determined_len > 0:
                new_max_len = determined_len
            else:
                print("Could not determine length automatically, falling back.")
        except Exception as e:
            print(f"Error during auto-scan for max length: {e}. Falling back.")

    elif args.new_max_length is not None:
        new_max_len = args.new_max_length

    if new_max_len is not None:
        old_len = final_config.get('max_length', 'N/A')
        print(f"Updating max_length: {old_len} -> {new_max_len}")
        final_config['max_length'] = new_max_len
    elif 'max_length' not in final_config:
         # Fallback if length wasn't in old config and not specified/auto-detected
         default_len = 1024
         print(f"Warning: max_length not found in old config and not specified/auto-detected. Defaulting to {default_len}.")
         final_config['max_length'] = default_len


    transplant_weights(args.old_model_path, final_config, args.output_dir, device)
