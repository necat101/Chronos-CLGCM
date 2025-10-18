import os
import sys
import json
import argparse
import time
import numpy as np
from typing import Optional, Tuple

# <<< MODIFIED: Set Tokenizers Parallelism Environment Variable >>>
# Set this early, before tokenizers might be implicitly loaded by other imports
# Setting to "true" forces parallelism despite potential fork issues (use with caution)
# Setting to "false" explicitly disables parallelism in worker processes (safer, suppresses warning)
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Or "false" to be safer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.serialization import safe_globals

# --- Optional Imports ---
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    _HAS_PEFT = True
except ImportError:
    print("Warning: 'peft' library not found. LoRA fine-tuning and merging will be unavailable.")
    _HAS_PEFT = False

try:
    import keyboard
    _HAS_KEYBOARD = True
except ImportError:
    print("Warning: 'keyboard' library not found. The Ctrl+X interrupt feature will be disabled in chat mode.")
    _HAS_KEYBOARD = False

# --- Optimizer Selection (Handles CUDA, bitsandbytes, and CPU fallback) ---
_HAS_BNB = False # Default assumption
_HAS_AMP = False # Default assumption

if torch.cuda.is_available():
    # Attempt to use bitsandbytes 8-bit AdamW if available
    try:
        import bitsandbytes as bnb
        ADAM_OPTIMIZER = bnb.optim.AdamW8bit
        _HAS_BNB = True
        print("INFO: CUDA detected and bitsandbytes found. Using bitsandbytes 8-bit AdamW optimizer.")
    except ImportError:
        print("Warning: bitsandbytes not found.")
        print("INFO: Falling back to standard torch.optim.AdamW optimizer.")
        ADAM_OPTIMIZER = torch.optim.AdamW
        _HAS_BNB = False # Explicitly set back to False

    # Check for AMP support (only relevant if CUDA is available)
    try:
        from torch.cuda.amp import GradScaler, autocast # Keep old import for now until warnings fixed
        # from torch.amp import GradScaler, autocast # Future update
        _HAS_AMP = True
        print("INFO: torch.cuda.amp available. AMP support is enabled.")
    except ImportError:
        _HAS_AMP = False # Keep False if import fails
        print("Warning: torch.cuda.amp not found. AMP support will be disabled.")

else: # No CUDA detected
    print("Warning: CUDA not detected. Using CPU training.")
    print("INFO: Falling back to standard torch.optim.AdamW optimizer (bitsandbytes requires CUDA).")
    ADAM_OPTIMIZER = torch.optim.AdamW
    _HAS_BNB = False # bitsandbytes not usable on CPU
    _HAS_AMP = False # AMP not usable on CPU
# --- END Optimizer Selection ---


# --- C++ Kernel Import ---
try:
    import chronos_matmul
    _HAS_KERNEL = True
    print("Successfully imported C++ quantization kernel.")
    if hasattr(chronos_matmul, "VULKAN_SUPPORT") and chronos_matmul.VULKAN_SUPPORT:
        print("INFO: Vulkan support is enabled in the compiled kernel.")
        _HAS_VULKAN = True
    else:
        print("INFO: Vulkan support is disabled in the compiled kernel.")
        _HAS_VULKAN = False
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ERROR: The compiled C++ kernel 'chronos_matmul' was not found.           !!!")
    print("!!!                                                                         !!!")
    print("!!! To fix this, please run the appropriate setup script:                   !!!")
    print("!!!  - On Windows:   Run setup.bat                                          !!!")
    print("!!!  - On Linux/macOS: Run bash setup.sh                                     !!!")
    print("!!!                                                                         !!!")
    print("!!! If you have already run the setup, you may need to activate the         !!!")
    print("!!! virtual environment first:                                              !!!")
    print("!!!  - On Windows:   .\\.venv\\Scripts\\Activate                              !!!")
    print("!!!  - On Linux/macOS: source .venv/bin/activate                             !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1) # Exit the program because the kernel is essential


# --- CONSTANTS ---
MODEL_WEIGHTS_NAME = "chronos.pt"
QUANTIZED_MODEL_WEIGHTS_NAME_TPL = "chronos-{qtype}.npz" # Template for the name


# --- HELPER CLASS: AttrDict ---
class AttrDict(dict):
    """A dictionary that allows for attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# --- Device Utilities ---
def pick_device():
    """Picks the best available device for PyTorch training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Commenting out MPS check for stability, can be re-enabled if needed
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")

def set_threads(n: int):
    try:
        torch.set_num_threads(n)
        os.environ['OMP_NUM_THREADS'] = str(n)
    except Exception as e:
        print(f"Warning: Could not set thread count. {e}")


# --- Dataset & Dataloader ---
class JSONLDataset(Dataset):
    """
    Handles both .jsonl (one JSON object per line) and .json (a list of objects) files.
    Also supports standard and "Kayla" instruction formats.
    """
    def __init__(self, path: str, tokenizer, max_length: int, kayla_mode: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.kayla_mode = kayla_mode
        self._load(path)

    def _process_object(self, obj):
        """Processes a single JSON object into tokenized input_ids and labels."""
        try:
            if self.kayla_mode:
                # --- Context part (masked out in labels) ---
                instruction_text = f"### Instruction:\n{obj.get('Instruction', '')}\n\n"

                feelings_text = ""
                # Check for the feelings key and ensure it has content to include it
                if obj.get('feelings'):
                    feelings_text = f"### Feelings:\n{obj.get('feelings')}\n\n"

                # The full prompt context is instruction + feelings
                prompt_context_text = instruction_text + feelings_text

                # --- Generation part (predicted by the model) ---
                thought_text = f"### Thought Process:\n{obj.get('thought-process', '')}\n\n"
                output_text = f"### Response:\n{obj.get('output', '')}"

                # Tokenize the different parts
                prompt_context_tokens = self.tokenizer.encode(prompt_context_text, add_special_tokens=True)
                thought_tokens = self.tokenizer.encode(thought_text, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)

                # Combine into final input_ids and labels
                input_ids = prompt_context_tokens + thought_tokens + output_tokens + [self.tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_context_tokens)) + thought_tokens + output_tokens + [self.tokenizer.eos_token_id]
            else:
                prompt = f"### Instruction:\n{obj.get('instruction', '')}\n\n### Response:\n"
                completion = obj.get('output', '') or obj.get('response', '')

                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

                input_ids = prompt_tokens + completion_tokens + [self.tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_tokens)) + completion_tokens + [self.tokenizer.eos_token_id]

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}
        except (KeyError, AttributeError, TypeError, ValueError) as e: # Added ValueError
             print(f"Warning: Skipping invalid data entry: {obj}. Error: {e}")
             return None


    def _load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        print(f"Loading and tokenizing dataset from {path}...")
        if self.kayla_mode:
            print("INFO: Kayla-style instruction tuning is ENABLED.")

        with open(path, "r", encoding="utf-8") as f:
            try:
                # Handle .json file (a list of objects)
                data = json.load(f)
                if isinstance(data, list):
                    print("Detected JSON file (list of objects). Processing...")
                    for obj in tqdm(data, desc="Tokenizing samples"):
                        processed = self._process_object(obj)
                        if processed: self.samples.append(processed)
                    return
                else:
                    print("Warning: JSON file does not contain a list of objects. Attempting JSONL parsing.")
                    # Fall through to JSONL parsing
            except json.JSONDecodeError:
                # Expected if it's a JSONL file
                pass # Fall through to JSONL parsing
            except Exception as e:
                print(f"Error loading JSON file: {e}. Attempting JSONL parsing.")
                # Fall through to JSONL parsing

            # Handle .jsonl file (one object per line)
            print("Detected or attempting JSONL file (one object per line). Processing...")
            f.seek(0) # Reset file pointer in case JSON loading failed
            line_num = 0
            for line in tqdm(f, desc="Tokenizing samples"):
                line_num += 1
                line = line.strip()
                if not line: continue # Skip empty lines
                try:
                    obj = json.loads(line)
                    processed = self._process_object(obj)
                    if processed: self.samples.append(processed)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {line[:100]}...") # Show beginning of bad line
                    continue
                except Exception as e: # Catch other potential errors during processing
                    print(f"Warning: Error processing line {line_num}: {e}. Line: {line[:100]}...")
                    continue


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# <<< MODIFIED: Added num_workers parameter >>>
def create_dataloader(path, tokenizer, max_length, batch_size, pad_token_id, kayla_mode=False, num_workers=0):
    """Creates a DataLoader for training or fine-tuning."""
    dataset = JSONLDataset(path, tokenizer, max_length, kayla_mode=kayla_mode)
    if len(dataset) == 0:
         raise ValueError(f"Dataset loaded from {path} is empty. Please check the file format and content.")


    def collate_fn(batch):
        # Filter out None items potentially returned by dataset __getitem__ if _process_object failed
        batch = [item for item in batch if item is not None]
        if not batch: return None # Return None if batch becomes empty

        max_len = max(len(item['input_ids']) for item in batch)

        input_ids_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask_batch = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids_batch[i, :seq_len] = item['input_ids']
            labels_batch[i, :seq_len] = item['labels']
            attention_mask_batch[i, :seq_len] = 1

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    # <<< MODIFIED: Pass num_workers to DataLoader >>>
    # pin_memory is often beneficial when using GPUs and workers
    pin_memory = torch.cuda.is_available() and num_workers > 0
    # Add persistent_workers=True if num_workers > 0, helps reuse workers
    persistent_workers = num_workers > 0
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)


# --- Quantization & Model Serialization ---
def get_q_block_size(qtype: str) -> int:
    """Returns the block size for a given quantization type."""
    if qtype in ["INT4", "Q4_0", "Q8_0"]:
        return 32
    elif qtype == "Q2_K":
        return 256
    else:
        raise ValueError(f"Unknown quantization type: {qtype}")

# <<< CHANGED: This function now saves to a directory with tokenizer files and config
def export_and_quantize_model(output_dir: str, model: nn.Module, tokenizer, qtype: str):
    """Quantizes and exports the model to a directory containing the .npz and tokenizer."""
    if not _HAS_KERNEL:
        print("ERROR: C++ kernel is required for quantization. Aborting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, QUANTIZED_MODEL_WEIGHTS_NAME_TPL.format(qtype=qtype))

    print(f"Quantizing and exporting model to {output_path} with {qtype} weights...")
    state_dict = model.state_dict()
    quantized_tensors = {}

    # <<< FIXED: Save the model config directly into the npz file
    config_to_save = dict(model.config)
    quantized_tensors['_config'] = np.array(config_to_save)

    Q_BLOCK_SIZE = get_q_block_size(qtype)

    for name, tensor in tqdm(state_dict.items(), desc="Quantizing Tensors"):
        # Include a check for 1D tensors (like biases and layernorm weights) and exclude LTM metadata
        if tensor.ndim == 2 and "emb" not in name and "ltm" not in name and "timestamps" not in name and "sources" not in name:
            float_tensor_np = tensor.cpu().float().numpy()
            M, K = float_tensor_np.shape

            pad_cols = 0
            if K % Q_BLOCK_SIZE != 0:
                pad_cols = Q_BLOCK_SIZE - (K % Q_BLOCK_SIZE)
                float_tensor_np = np.pad(float_tensor_np, ((0, 0), (0, pad_cols)), 'constant')

            quantized_data = chronos_matmul.quantize(float_tensor_np, qtype)
            quantized_tensors[name] = {
                "quantized": quantized_data,
                "qtype": qtype,
                "original_shape": [M, K],
            }
        # Keep 1D tensors and explicitly excluded layers as raw numpy arrays
        elif tensor.ndim >= 1 and ("emb" in name or "norm" in name or "bias" in name or "persistent" in name or "ltm." in name):
            # Exclude internal LTM buffers explicitly
            if name != 'ltm._mom_vals' and name != 'ltm.ltm_deltas':
                quantized_tensors[name] = {"raw": tensor.cpu().numpy()}
            else:
                print(f"Skipping internal LTM buffer '{name}' from quantization file.")
        else:
            print(f"Skipping tensor '{name}' with shape {tensor.shape} from quantization.")


    np.savez_compressed(output_path, **quantized_tensors)
    print(f"Model weights successfully exported to {output_path}")

    # <<< FIXED: Also save the tokenizer to make the directory self-contained
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer files saved to {output_dir}")

class QuantizedLinear:
    """A wrapper for a quantized linear layer that uses the C++ kernel for inference."""
    def __init__(self, name: str, q_data: dict):
        self.name = name
        weight_data_key = f'{name}.weight'
        bias_data_key = f'{name}.bias'

        if weight_data_key not in q_data:
            raise KeyError(f"Weight data '{weight_data_key}' not found in quantized file.")

        weight_meta = q_data[weight_data_key].item()
        if 'quantized' not in weight_meta:
             raise ValueError(f"Weight '{weight_data_key}' is not quantized (missing 'quantized' key).")

        self.quantized_w = weight_meta['quantized']
        self.qtype = str(weight_meta['qtype'])
        self.original_shape = weight_meta['original_shape']
        self.M, self.K = self.original_shape

        if bias_data_key in q_data:
            bias_meta = q_data[bias_data_key].item()
            if 'raw' not in bias_meta:
                 raise ValueError(f"Bias '{bias_data_key}' is missing 'raw' data.")
            self.bias = bias_meta['raw']
        else:
            self.bias = None

    def __call__(self, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        if not _HAS_KERNEL: raise ImportError("C++ kernel required for quantized matmul")

        x_np = x.cpu().float().numpy()
        # Ensure x_np is 2D for matmul
        original_ndim = x_np.ndim
        original_shape = x_np.shape
        if original_ndim == 1:
            x_np = x_np.reshape(1, -1)
        elif original_ndim > 2:
            # Flatten leading dimensions if any (e.g., batch, sequence length)
            x_np = x_np.reshape(-1, x_np.shape[-1])
        # Ensure input K matches the quantized weight K (after potential padding)
        expected_k = self.K
        if x_np.shape[-1] != expected_k:
             # This might happen if the original model had K not multiple of block size
             # Pad input if needed (although quantization should handle weight padding)
             pad_k = expected_k - x_np.shape[-1]
             if pad_k > 0:
                 x_np = np.pad(x_np, ((0, 0), (0, pad_k)), 'constant')
             elif pad_k < 0: # Should ideally not happen if original_shape is correct
                 x_np = x_np[..., :expected_k]


        y_np = chronos_matmul.matmul_quantized(x_np, self.quantized_w, self.M, self.qtype, device)

        # Ensure output shape matches original input dimensions + output features M
        if original_ndim > 2:
            output_shape = list(original_shape[:-1]) + [self.M]
            y_np = y_np.reshape(output_shape)
        elif original_ndim == 1:
            y_np = y_np.reshape(-1) # Reshape back to 1D


        if y_np.shape[-1] != self.M:
             # This should ideally not happen if matmul_quantized handles padding correctly,
             # but keep as a safeguard.
             y_np = y_np[..., :self.M]

        if self.bias is not None: y_np += self.bias
        return torch.from_numpy(y_np)

class QuantizedGRUCell:
    """A quantized GRU cell implementation using QuantizedLinear layers."""
    def __init__(self, input_size, hidden_size, name_prefix, q_data):
        self.input_size, self.hidden_size = input_size, hidden_size
        self.W_ir = QuantizedLinear(f'{name_prefix}.W_ir', q_data)
        self.W_hr = QuantizedLinear(f'{name_prefix}.W_hr', q_data)
        self.W_iz = QuantizedLinear(f'{name_prefix}.W_iz', q_data)
        self.W_hz = QuantizedLinear(f'{name_prefix}.W_hz', q_data)
        self.W_in = QuantizedLinear(f'{name_prefix}.W_in', q_data)
        self.W_hn = QuantizedLinear(f'{name_prefix}.W_hn', q_data)

    def __call__(self, x: torch.Tensor, h: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        r = torch.sigmoid(self.W_ir(x, device) + self.W_hr(h, device))
        z = torch.sigmoid(self.W_iz(x, device) + self.W_hz(h, device))
        n = torch.tanh(self.W_in(x, device) + r * self.W_hn(h, device))
        return (1 - z) * n + z * h

class GRUCell(nn.Module):
    """A custom GRU cell implementation using individual Linear layers."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Using individual layers ensures names like "h_rnn.W_ir.weight"
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        return (1 - z) * n + z * h

class LTMModule(nn.Module):
    """Titans Long-Term Memory Module. Capable of test-time updates."""
    ### MODIFICATION START: Define Source IDs for clarity ###
    # SOURCE_ID definitions
    SRC_UNKNOWN = 0
    SRC_USER_INTERACTION = 1
    SRC_TRAINING_DATA = 2
    ### MODIFICATION END ###

    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, val_dim) * 0.02)
        self.register_buffer("_mom_vals", torch.zeros_like(self.vals.data))
        self.lr, self.momentum, self.weight_decay = lr, momentum, wd

        ### MODIFICATION START: Add buffers for metadata ###
        # Buffers are not parameters; they are part of the model's state
        # but are not updated by the optimizer during training.
        self.register_buffer("timestamps", torch.zeros(n_slots, dtype=torch.float32))
        self.register_buffer("sources", torch.full((n_slots,), self.SRC_UNKNOWN, dtype=torch.long))
        ### MODIFICATION END ###

        # Buffer for accumulating deltas if not updating in-place
        self.register_buffer("ltm_deltas", torch.zeros_like(self.vals.data))
        self.accumulate_deltas = False

    ### MODIFICATION START: Enhance retrieval with filtering ###
    def retrieve_topk(self, queries: torch.Tensor, topk: int = 4, min_timestamp: float = 0.0, source_filter: Optional[int] = None) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Retrieves the top-k most similar values from memory, with optional filtering
        by timestamp and source.
        """
        sim = queries @ self.keys.t()

        # Apply filters by creating a mask of valid slots
        if min_timestamp > 0.0 or source_filter is not None:
            with torch.no_grad():
                # Start with all slots being valid
                valid_mask = torch.ones(self.keys.size(0), dtype=torch.bool, device=self.keys.device)

                if min_timestamp > 0.0:
                    # Filter out memories that are older than the specified timestamp
                    valid_mask &= (self.timestamps >= min_timestamp)

                if source_filter is not None:
                    # Filter for a specific source
                    valid_mask &= (self.sources == source_filter)

                # Set the similarity score of invalid slots to negative infinity
                # so they are never chosen by topk. Handle potential NaNs/Infs in sim first.
                sim = torch.nan_to_num(sim, nan=-torch.inf, posinf=torch.finfo(sim.dtype).max, neginf=-torch.inf)
                sim[:, ~valid_mask] = -torch.inf


        # Ensure topk is not greater than the number of valid slots
        num_valid_slots = sim.isfinite().sum(dim=-1).min().item() # Minimum valid slots across batch
        effective_topk = min(topk, int(num_valid_slots))

        if effective_topk <= 0:
            # Handle case where no slots match the filter
            print("Warning: No LTM slots matched the current filter criteria.")
            # Return empty tensors or tensors filled with zeros/dummy values
            query_shape = list(queries.shape)
            # Shape: [Batch, TopK, ValDim]
            vals_shape = query_shape[:-1] + [topk, self.vals.shape[-1]]
            # Shape: [Batch, TopK]
            idx_shape = query_shape[:-1] + [topk]
            return torch.zeros(vals_shape, device=queries.device, dtype=self.vals.dtype), \
                   torch.full(idx_shape, -1, device=queries.device, dtype=torch.long) # Use -1 for invalid index


        _, idx = torch.topk(sim, k=effective_topk, dim=-1)

        # Pad results if effective_topk < topk
        if effective_topk < topk:
            pad_size = topk - effective_topk
            # Pad indices with -1 (or another invalid index)
            idx_pad = torch.full(list(idx.shape[:-1]) + [pad_size], -1, device=idx.device, dtype=idx.dtype)
            idx = torch.cat([idx, idx_pad], dim=-1)

            # Pad retrieved values with zeros
            vals_retrieved = self.vals[idx[..., :effective_topk]] # Get only valid values first
            vals_pad = torch.zeros(list(vals_retrieved.shape[:-2]) + [pad_size, vals_retrieved.shape[-1]], device=vals_retrieved.device, dtype=vals_retrieved.dtype)
            vals_ret = torch.cat([vals_retrieved, vals_pad], dim=-2) # Concatenate along the topk dimension
            return vals_ret, idx
        else:
            # Ensure indices are valid before indexing self.vals
            idx = idx.clamp(min=0, max=self.vals.shape[0]-1) # Clamp indices just in case
            return self.vals[idx], idx

    ### MODIFICATION END ###

    ### MODIFICATION START: Update metadata during memory consolidation ###
    def inner_update(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor, current_lr: float, source: int = SRC_USER_INTERACTION):
        """
        Performs a meta-learning update on the LTM value slots based on the "surprise" gradient.
        Now also updates the timestamp and source metadata for the modified slots.
        """
        with torch.no_grad():
            if grads_tensor is None: return
            device = self.vals.device

            # Filter out invalid indices (e.g., -1 from padding in retrieve_topk)
            valid_mask = topk_idx >= 0
            if not valid_mask.any(): return # No valid indices to update

            idx_flat = topk_idx[valid_mask].view(-1)
            # Ensure grads_tensor has the same shape pattern as topk_idx before masking
            # Ensure grads_tensor is expanded correctly if needed (e.g., if batch size was 1)
            if grads_tensor.shape[0] != topk_idx.shape[0] or grads_tensor.shape[1] != topk_idx.shape[1]:
                print(f"Warning: grads_tensor shape {grads_tensor.shape} mismatch with topk_idx shape {topk_idx.shape}. Skipping LTM update.")
                return # Avoid shape errors
            grads_flat = grads_tensor[valid_mask].view(-1, self.vals.size(1))


            slot_grads = torch.zeros_like(self.vals.data)
            slot_grads.index_add_(0, idx_flat.to(device), grads_flat.to(device))

            counts = torch.zeros(self.vals.size(0), device=device)
            counts.index_add_(0, idx_flat.to(device), torch.ones_like(idx_flat, dtype=torch.float, device=device))
            nonzero_mask = counts > 0
            if nonzero_mask.any():
                slot_grads[nonzero_mask] /= counts[nonzero_mask].unsqueeze(-1)

            self._mom_vals.data.mul_(self.momentum).add_(slot_grads)
            update_delta = (self._mom_vals.data + self.weight_decay * self.vals.data)
            update_delta.mul_(-current_lr)

            final_update = torch.zeros_like(self.vals.data)
            final_update[nonzero_mask] = update_delta[nonzero_mask]

            # --- UPDATE METADATA ---
            current_time = time.time()
            self.timestamps.data[nonzero_mask] = current_time
            self.sources.data[nonzero_mask] = source
            # --- END METADATA UPDATE ---

            if self.accumulate_deltas:
                self.ltm_deltas.data.add_(final_update)
                self.vals.data.add_(final_update)
            else:
                self.vals.data.add_(final_update)
    ### MODIFICATION END ###


class ChronosCore(nn.Module):
    """The full, trainable Chronos model, integrating HRM as the core processor."""
    def __init__(self, config: dict):
        super().__init__()
        # Use AttrDict for config to support both dot-notation and .get() method access
        self.config = AttrDict(config)

        # --- Ensure required config values exist ---
        required_keys = ['vocab_size', 'context_dim', 'max_length', 'persistent_dim',
                         'ltm_slots', 'ltm_key_dim', 'ltm_val_dim', 'ltm_lr',
                         'ltm_topk', 'h_hidden', 'l_hidden', 'max_h_steps',
                         'max_l_steps', 'l_conv_atol']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: '{key}'")

        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.context_dim)
        self.pos_emb = nn.Embedding(self.config.max_length, self.config.context_dim)

        self.persistent = nn.Parameter(torch.randn(self.config.persistent_dim) * 0.02)
        self.ltm = LTMModule(
            n_slots=self.config.ltm_slots,
            key_dim=self.config.ltm_key_dim,
            val_dim=self.config.ltm_val_dim,
            lr=self.config.ltm_lr # Note: lr here is mainly for potential direct training, chat uses its own LR
        )
        self.qproj = nn.Linear(self.config.context_dim, self.config.ltm_key_dim, bias=False)

        in_dim = self.config.context_dim + self.config.persistent_dim + (self.config.ltm_val_dim * self.config.ltm_topk)
        self.in_proj = nn.Linear(in_dim, self.config.context_dim)

        self.h_rnn = GRUCell(self.config.context_dim, self.config.h_hidden)
        self.h_to_context = nn.Linear(self.config.h_hidden, self.config.context_dim)
        self.l_rnn = GRUCell(self.config.context_dim * 2, self.config.l_hidden)
        self.l_to_out = nn.Linear(self.config.l_hidden, self.config.context_dim)

        ### MODIFICATION START ###
        # Add the halting projection layer
        self.h_halt_proj = nn.Linear(self.config.h_hidden, 1)
        ### MODIFICATION END ###

        self.out_norm = nn.LayerNorm(self.config.context_dim)
        self.lm_head = nn.Linear(self.config.context_dim, self.config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight # Weight tying

        # Ensure model config reflects its parameters
        self.config.model_type = 'chronos'


    # Methods to satisfy the PEFT library
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _get_prompt_embedding(self, prompt_embedding):
        return prompt_embedding

    ### MODIFICATION START: Add filtering args to forward method ###
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, labels: Optional[torch.LongTensor] = None, min_timestamp: float = 0.0, source_filter: Optional[int] = None, **kwargs):
    ### MODIFICATION END ###
        B, T = input_ids.shape
        device = input_ids.device

        tok_embs = self.tok_emb(input_ids)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = tok_embs + self.pos_emb(pos)

        if attention_mask is None: attention_mask = torch.ones_like(input_ids)

        all_topk_vals = []
        all_topk_idx = []
        final_token_embeddings = []
        all_ponder_costs = []

        h_state = torch.zeros(B, self.config.h_hidden, device=device)
        l_state = torch.zeros(B, self.config.l_hidden, device=device)

        for t in range(T):
            token_emb = x[:, t, :]

            p_read = self.persistent.unsqueeze(0).expand(B, -1)
            query = self.qproj(token_emb)

            ### MODIFICATION START: Pass filters to retrieve_topk ###
            topk_vals, topk_idx = self.ltm.retrieve_topk(
                query,
                topk=self.config.ltm_topk,
                min_timestamp=min_timestamp,
                source_filter=source_filter
            )
            ### MODIFICATION END ###

            # NOTE: This UserWarning is expected and necessary.
            # We must retain the grad on a non-leaf tensor (the retrieved values)
            # to calculate the "surprise" gradient for the LTM update.
            if self.training or torch.is_grad_enabled():
                # Only retain grad if the retrieved values are not just zeros (e.g., from failed filter)
                # and if the tensor actually requires grad (might be detached if loaded)
                if topk_vals.requires_grad:
                    topk_vals.retain_grad()


            all_topk_vals.append(topk_vals)
            all_topk_idx.append(topk_idx)

            ltm_summary_flat = topk_vals.view(B, -1)

            mac_input = torch.cat([token_emb, p_read, ltm_summary_flat], dim=-1)
            enc = F.gelu(self.in_proj(mac_input))

            ### MODIFICATION START: Adaptive HRM Loop ###
            step_outputs = []
            halt_probs = []

            # The initial 'enc' is passed to the first H-step
            current_enc = enc

            for h_step in range(self.config.max_h_steps):
                h_state = self.h_rnn(current_enc, h_state)
                context = self.h_to_context(h_state)
                l_input = torch.cat([current_enc, context], dim=-1)

                # L-module converges to equilibrium
                l_state_prev = torch.zeros_like(l_state) # Initialize for the first check
                for _ in range(self.config.max_l_steps):
                    l_state_prev = l_state.clone() # Must clone to prevent reference issues
                    l_state = self.l_rnn(l_input, l_state)
                    # Check for convergence
                    if torch.allclose(l_state, l_state_prev, atol=self.config.l_conv_atol):
                        break

                # The output for this H-step is the converged L-state applied as a residual
                step_update = self.l_to_out(l_state)
                current_enc = current_enc + step_update

                # Calculate halt probability for this step
                halt_logit = self.h_halt_proj(h_state).squeeze(-1) # Shape: (B,)
                halt_prob = torch.sigmoid(halt_logit)

                step_outputs.append(current_enc)
                halt_probs.append(halt_prob)

                # For inference, we can exit early for efficiency
                # Use getattr for safe access to h_halt_thresh, provide default if missing
                halt_thresh = getattr(self.config, 'h_halt_thresh', 0.9)
                if not self.training and (halt_prob.mean() > halt_thresh):
                    break

            # After the loop, calculate the final output and ponder cost using ACT logic
            step_outputs_t = torch.stack(step_outputs, dim=0) # Shape: (H, B, D)
            halt_probs_t = torch.stack(halt_probs, dim=0)     # Shape: (H, B)
            num_steps_taken = halt_probs_t.shape[0]

            # Calculate probabilities for weighting
            unhalt_probs = 1.0 - halt_probs_t
            # Cumulative product of unhalting probabilities up to the previous step
            unhalt_probs_shifted = torch.cat([torch.ones_like(unhalt_probs[:1]), unhalt_probs[:-1]], dim=0)
            cum_unhalt_probs = torch.cumprod(unhalt_probs_shifted, dim=0)

            # Weight for each step is p_h * product_{i<h}(1-p_i)
            weights = halt_probs_t * cum_unhalt_probs

            # Remainder is the probability of not having halted after all steps
            remainder = cum_unhalt_probs[-1] * (1.0 - halt_probs_t[-1])

            # Normalize weights to ensure they sum to 1
            total_prob_sum = weights.sum(dim=0) + remainder + 1e-8 # Add epsilon for stability
            weights = weights / total_prob_sum.unsqueeze(0)

            # Weighted average of step outputs
            final_enc = (weights.unsqueeze(-1) * step_outputs_t).sum(dim=0)

            # Ponder cost: number of steps executed + probability of not halting
            ponder_cost = num_steps_taken + remainder
            all_ponder_costs.append(ponder_cost)
            ### MODIFICATION END ###

            final_token_embeddings.append(final_enc)

        final_embeddings = torch.stack(final_token_embeddings, dim=1)
        final_embeddings = self.out_norm(final_embeddings)
        logits = self.lm_head(final_embeddings)

        loss = None
        ponder_cost_out = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            ### MODIFICATION START ###
            # Average the ponder cost across the sequence length and batch
            if all_ponder_costs:
                ponder_cost_out = torch.stack(all_ponder_costs, dim=1).mean() # Mean over sequence, then batch
            else: # Handle edge case T=0
                ponder_cost_out = torch.tensor(0.0, device=device)
            ### MODIFICATION END ###

        # Ensure returned tensors are correctly shaped even if loop didn't run
        seq_topk_vals = torch.stack(all_topk_vals, dim=1) if all_topk_vals else torch.empty(B, 0, self.config.ltm_topk, self.config.ltm_val_dim, device=device)
        seq_topk_idx = torch.stack(all_topk_idx, dim=1) if all_topk_idx else torch.empty(B, 0, self.config.ltm_topk, dtype=torch.long, device=device)


        return {"loss": loss, "logits": logits, "topk_vals": seq_topk_vals, "topk_idx": seq_topk_idx, "ponder_cost": ponder_cost_out}


class QuantizedChronos:
    """The quantized Chronos model for CPU/Vulkan inference."""
    def __init__(self, config: dict, q_data: dict):
        self.config = AttrDict(config)
        self.qtype = None # Will be determined from the first quantized layer

        # Load raw (non-quantized) parameters first
        try:
             self.tok_emb = nn.Embedding.from_pretrained(torch.from_numpy(q_data['tok_emb.weight'].item()['raw']))
             self.pos_emb = nn.Embedding.from_pretrained(torch.from_numpy(q_data['pos_emb.weight'].item()['raw']))
             self.persistent = torch.from_numpy(q_data['persistent'].item()['raw'])
             self.out_norm = nn.LayerNorm(self.config.context_dim)
             self.out_norm.load_state_dict({
                 'weight': torch.from_numpy(q_data['out_norm.weight'].item()['raw']),
                 'bias': torch.from_numpy(q_data['out_norm.bias'].item()['raw'])
             })
             self.ltm = LTMModule(n_slots=self.config.ltm_slots, key_dim=self.config.ltm_key_dim, val_dim=self.config.ltm_val_dim)
             # Load LTM state, ignoring momentum buffer etc.
             ltm_state = {k: torch.from_numpy(v.item()['raw']) for k, v in q_data.items() if k.startswith('ltm.') and k != 'ltm.ltm_deltas' and k != 'ltm._mom_vals'}
             self.ltm.load_state_dict(ltm_state, strict=False)

        except KeyError as e:
             raise KeyError(f"Missing expected raw parameter in quantized file: {e}")
        except Exception as e:
             raise RuntimeError(f"Error loading raw parameters: {e}")


        # Initialize quantized layers, determining qtype from the first one found
        quantized_layers = {}
        expected_quantized = ['qproj', 'in_proj', 'h_rnn', 'h_to_context', 'l_rnn', 'l_to_out', 'lm_head', 'h_halt_proj']

        for layer_base_name in expected_quantized:
             try:
                 if layer_base_name in ['h_rnn', 'l_rnn']:
                     input_sz = self.config.context_dim if layer_base_name == 'h_rnn' else self.config.context_dim * 2
                     hidden_sz = self.config.h_hidden if layer_base_name == 'h_rnn' else self.config.l_hidden
                     quantized_layers[layer_base_name] = QuantizedGRUCell(input_sz, hidden_sz, layer_base_name, q_data)
                     # Determine qtype from the first sub-layer if not already set
                     if self.qtype is None:
                         self.qtype = quantized_layers[layer_base_name].W_ir.qtype
                 else:
                     quantized_layers[layer_base_name] = QuantizedLinear(layer_base_name, q_data)
                     if self.qtype is None:
                         self.qtype = quantized_layers[layer_base_name].qtype
             except KeyError as e:
                 raise KeyError(f"Missing expected quantized layer data for '{layer_base_name}' in file: {e}")
             except Exception as e:
                 raise RuntimeError(f"Error initializing quantized layer '{layer_base_name}': {e}")


        # Assign quantized layers to attributes
        self.qproj = quantized_layers['qproj']
        self.in_proj = quantized_layers['in_proj']
        self.h_rnn = quantized_layers['h_rnn']
        self.h_to_context = quantized_layers['h_to_context']
        self.l_rnn = quantized_layers['l_rnn']
        self.l_to_out = quantized_layers['l_to_out']
        self.lm_head = quantized_layers['lm_head']
        self.h_halt_proj = quantized_layers['h_halt_proj']


        if self.qtype is None:
             raise ValueError("Could not determine quantization type from the loaded model file.")

        print(f"Initialized QuantizedChronos model ({self.qtype}) from config.")


    ### MODIFICATION START: Add filtering args to call method ###
    def __call__(self, input_ids: torch.LongTensor, h_state: torch.Tensor, l_state: torch.Tensor, device: str = "cpu", min_timestamp: float = 0.0, source_filter: Optional[int] = None):
    ### MODIFICATION END ###
        B, T = input_ids.shape
        # Use T-1 for single-token generation, but allow for longer sequences during initial prompt processing.
        current_pos_start = T - 1 if T > 1 else 0

        # Process the entire input_ids sequence token-by-token (or just the last token if T > 1)
        for t in range(current_pos_start, T):
            # Ensure inputs to embeddings are LongTensors on CPU
            current_token_ids = input_ids[:, t].cpu().long()
            pos_ids = torch.tensor([t], dtype=torch.long).cpu()

            token_emb = self.tok_emb(current_token_ids) + self.pos_emb(pos_ids)

            p_read = self.persistent.unsqueeze(0).expand(B, -1)
            query = self.qproj(token_emb, device=device) # Query projection happens on target device

            ### MODIFICATION START: Pass filters to retrieve_topk ###
            topk_vals, _ = self.ltm.retrieve_topk(
                query,
                topk=self.config.ltm_topk,
                min_timestamp=min_timestamp,
                source_filter=source_filter
            )
            ### MODIFICATION END ###

            # Ensure topk_vals is on CPU for concatenation if needed, though subsequent ops use target device
            ltm_summary_flat = topk_vals.view(B, -1).cpu()


            mac_input = torch.cat([token_emb.cpu(), p_read.cpu(), ltm_summary_flat], dim=-1)
            enc = F.gelu(self.in_proj(mac_input, device=device)) # Input projection on target device

            # Ensure RNN states are on CPU before passing to QuantizedGRUCell
            h_state = h_state.cpu()
            l_state = l_state.cpu()

            ### MODIFICATION START: Adaptive HRM Loop for Inference ###
            for _ in range(self.config.max_h_steps):
                h_state = self.h_rnn(enc, h_state, device=device) # GRU cell computes on target device

                # Check for halt condition
                halt_logit = self.h_halt_proj(h_state, device=device) # Projection on target device
                halt_prob = torch.sigmoid(halt_logit)

                context = self.h_to_context(h_state, device=device) # Context projection on target device
                # L-module runs for its max steps; convergence check is omitted for speed in quantized inference
                for _ in range(self.config.max_l_steps):
                    l_input = torch.cat([enc.cpu(), context.cpu()], dim=-1) # Concat on CPU
                    l_state = self.l_rnn(l_input, l_state, device=device) # GRU cell on target device

                # Add residual on target device
                enc = enc + self.l_to_out(l_state, device=device)

                # Use getattr for safe access, provide default if missing
                halt_thresh = getattr(self.config, 'h_halt_thresh', 0.9)
                # Early exit based on halt probability (checked on CPU)
                if halt_prob.cpu().item() > halt_thresh:
                    break
            ### MODIFICATION END ###

        # Final operations
        final_embedding = self.out_norm(enc.cpu()).to(enc.dtype) # Norm on CPU, ensure correct dtype
        logits = self.lm_head(final_embedding, device=device) # LM head on target device

        # Return states on CPU as expected by the chat loop for quantized models
        return {"logits": logits.unsqueeze(1), "h_state": h_state.cpu(), "l_state": l_state.cpu()}


# <<< FIXED: This function now loads a portable, self-contained model directory
def load_quantized(model_path: str):
    """Loads a quantized model directory, automatically finding the .npz and tokenizer."""
    print(f"Loading quantized model from directory: {model_path}")

    # Find the .npz file in the directory
    npz_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No quantized model .npz file found in {model_path}")
    if len(npz_files) > 1:
        print(f"Warning: Multiple .npz files found. Using the first one: {npz_files[0]}")

    weights_path = os.path.join(model_path, npz_files[0])
    q_data = np.load(weights_path, allow_pickle=True)

    if '_config' not in q_data:
        raise ValueError("Quantized model file is missing '_config' data. Please re-quantize the model.")

    config = q_data['_config'].item()
    # Ensure config gets 'model_type' if missing, useful for HF compatibility downstream
    if 'model_type' not in config:
        config['model_type'] = 'chronos'

    return QuantizedChronos(config, q_data), AttrDict(config)


# <<< FIXED: This function now loads a portable, self-contained model directory
def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model and its config from a directory."""
    weights_path = os.path.join(model_path, MODEL_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{MODEL_WEIGHTS_NAME}' not found in '{model_path}'")

    # Load checkpoint safely, allowing pickles only if necessary (e.g., for optimizer state)
    # For inference-only loading, weights_only=True is safer. For resuming, False is needed.
    # We load with weights_only=False first to get config, then potentially reload if only weights are needed.
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except RuntimeError as e:
        # If weights_only=False fails (e.g., untrusted pickle), try weights_only=True
        print(f"Warning: Failed to load checkpoint allowing pickles ({e}). Attempting weights_only=True.")
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
            if 'config' not in checkpoint: # Config might be missing in weights_only load
                 raise ValueError("Config missing even in weights_only load.")
        except Exception as inner_e:
             raise RuntimeError(f"Failed to load checkpoint even with weights_only=True: {inner_e}")


    # Config is always loaded from the checkpoint
    if 'config' not in checkpoint:
        raise ValueError("Model config not found in checkpoint. The model file is likely corrupted or from an old version.")
    config = AttrDict(checkpoint['config'])

    # Ensure model_type is present for HuggingFace compatibility
    if 'model_type' not in config:
        config['model_type'] = 'chronos'

    # Ensure vocab_size is present before creating model
    if 'vocab_size' not in config:
        # Attempt to infer from tokenizer if possible (might require loading tokenizer first)
        # Or raise a more specific error
        raise ValueError("Cannot initialize model: 'vocab_size' missing from checkpoint config.")


    model = ChronosCore(config).to(device)

    # Load state dict, be flexible with missing/extra keys if needed
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"Warning: Non-strict state dict loading due to mismatch: {e}. Trying strict=False.")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    return model, config


# <<< Corrected train function >>>
def train(args, device, tokenizer):
    print("Running in TRAIN mode...")
    config = vars(args)
    # Ensure train data path is saved in config for potential resume with auto-max-length
    config['train_data_path'] = args.train
    config['model_type'] = 'chronos' # Ensure model_type is set

    # --- Determine vocab_size ---
    # Needs to be known BEFORE model initialization
    current_vocab_size = None
    if tokenizer:
        current_vocab_size = len(tokenizer)
        # If starting fresh, ensure config gets the vocab_size
        if not args.resume_from_ckpt and 'vocab_size' not in config:
            config['vocab_size'] = current_vocab_size
    elif not args.resume_from_ckpt:
        # Should not happen if tokenizer loading in main is correct
        raise RuntimeError("Tokenizer not loaded, cannot determine vocab_size for new model.")
    # If resuming, vocab_size *should* come from the checkpoint config,
    # but we'll add a check later.

    model = None # Initialize model variable
    optimizer = None # Initialize optimizer variable
    start_epoch = 0
    model_config = None # Initialize model_config
    scaler = None # Initialize scaler
    scheduler = None # Initialize scheduler
    use_amp = args.amp and _HAS_AMP # Determine AMP usage early

    # --- Create DataLoader ---
    # Moved dataloader creation earlier to use its length for scheduler setup
    try:
        dataloader = create_dataloader(
            args.train, tokenizer, args.max_length, args.batch_size,
            tokenizer.pad_token_id if tokenizer else 0, # Use 0 if tokenizer is None temporarily
            kayla_mode=args.kayla, num_workers=args.num_workers
        )
        if dataloader is None or len(dataloader) == 0:
            raise ValueError("DataLoader creation failed or resulted in an empty loader.")
        dataloader_len = len(dataloader)
        print(f"INFO: DataLoader created with {dataloader_len} batches.")
    except Exception as e:
        print(f"ERROR creating DataLoader: {e}")
        sys.exit(1)


    # <<< Resume logic >>>
    if args.resume_from_ckpt:
        if not os.path.exists(args.resume_from_ckpt):
            raise FileNotFoundError(f"Checkpoint to resume from not found at {args.resume_from_ckpt}")

        print(f"Resuming training from checkpoint: {args.resume_from_ckpt}")

        # weights_only=False is crucial for loading optimizer, scheduler, scaler
        try:
            checkpoint = torch.load(args.resume_from_ckpt, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load training checkpoint (needs optimizer/scheduler state): {e}")


        if 'optimizer_state_dict' not in checkpoint:
            raise ValueError("The specified checkpoint is a final inference model, not a training checkpoint. Cannot resume.")

        # Load config from checkpoint to ensure consistency
        if 'config' in checkpoint:
            model_config = AttrDict(checkpoint['config'])
            # <<< START PATCH for vocab_size >>>
            if 'vocab_size' not in model_config:
                if current_vocab_size:
                    print(f"Warning: 'vocab_size' not found in checkpoint config. Setting from loaded tokenizer ({current_vocab_size}).")
                    model_config['vocab_size'] = current_vocab_size
                else:
                    # This case should ideally not happen if tokenizer loading is correct
                    raise ValueError("Cannot determine vocab_size: Not found in checkpoint and tokenizer not loaded.")
            elif model_config.vocab_size != current_vocab_size and current_vocab_size is not None:
                # Warn if checkpoint vocab size differs from currently loaded tokenizer
                print(f"Warning: Checkpoint vocab_size ({model_config.vocab_size}) differs from loaded tokenizer ({current_vocab_size}). Using checkpoint value.")
            # <<< END PATCH for vocab_size >>>

            # Ensure model_type is present for HuggingFace compatibility
            if 'model_type' not in model_config:
                model_config['model_type'] = 'chronos'

            print("INFO: Re-initializing model architecture from checkpoint config.")
            model = ChronosCore(model_config).to(device) # Create model AFTER potentially fixing vocab_size
        else:
            print("Warning: Config not found in checkpoint. Using current CLI args for model architecture.")
            # Ensure CLI args config has vocab_size if falling back
            cli_config = config # Use the initial config from vars(args)
            if 'vocab_size' not in cli_config and current_vocab_size:
                cli_config['vocab_size'] = current_vocab_size
            elif 'vocab_size' not in cli_config:
                raise ValueError("Cannot determine vocab_size: Not found in checkpoint or CLI args, and tokenizer not loaded.")
            model_config = AttrDict(cli_config) # Fallback, might cause issues if arch changed
            model = ChronosCore(model_config).to(device)


        # --- START FIX: Optimizer Initialization/Loading Logic ---
        # Initialize optimizer AFTER model creation
        # Initialize with the NEW LR from args if overriding, otherwise use LR from checkpoint config (or args as fallback)
        initial_lr_for_optim = args.starting_lr if args.override_scheduling else checkpoint.get('config', {}).get('starting_lr', args.starting_lr)
        optimizer = ADAM_OPTIMIZER(model.parameters(), lr=initial_lr_for_optim) # Use determined LR
        # --- END FIX ---

        # Load model state dict with flexibility
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Non-strict model state dict loading: {e}. Trying strict=False.")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # --- START FIX: Conditional Optimizer State Loading ---
        # Load optimizer state ONLY IF NOT OVERRIDING SCHEDULING
        if not args.override_scheduling: # <<<--- ADDED THIS CONDITION
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Starting optimizer from scratch.")
                # Re-initialize optimizer if loading failed severely (already done above, ensure LR is right)
                optimizer = ADAM_OPTIMIZER(model.parameters(), lr=initial_lr_for_optim) # Re-init with correct LR
        else:
            print("INFO: --override-scheduling detected. Skipping loading optimizer state.") # <<<--- ADDED INFO MSG
        # --- END FIX ---

        start_epoch = checkpoint.get('completed_epoch', 0) # Use get for safety

        # --- Initialize AMP GradScaler ---
        if use_amp:
            scaler = GradScaler()
            print("INFO: Automatic Mixed Precision (AMP) ENABLED for training.")
            # --- Resume GradScaler state ---
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                print("Resuming GradScaler state.")
                try:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except Exception as e:
                    print(f"Warning: Failed to load scaler state: {e}. Continuing with a fresh scaler.")
            else:
                 # No warning needed if overriding, expected to start fresh scaler anyway
                 if not args.override_scheduling:
                    print("Warning: Scaler state not found in checkpoint. Initializing a fresh scaler.")
        # --- End AMP Init/Resume ---

        # --- Initialize Scheduler (AFTER optimizer is potentially re-initialized) ---
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0

        if not args.disable_lr_schedule and num_update_steps > 0:
            print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            # Initialize scheduler with the potentially fresh optimizer
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)

            ### --- Scheduler Resuming Logic --- ###
            checkpoint_has_scheduler = 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None

            if checkpoint_has_scheduler and not args.override_scheduling:
                # DEFAULT: Resume from checkpoint, warn if user tried to change LR.
                loaded_config = checkpoint.get('config', {})
                old_lr = loaded_config.get('starting_lr')
                old_min_lr = loaded_config.get('min_lr')
                lr_mismatch = (old_lr is not None and not np.isclose(old_lr, args.starting_lr))
                min_lr_mismatch = (old_min_lr is not None and not np.isclose(old_min_lr, args.min_lr))

                if lr_mismatch or min_lr_mismatch:
                    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!! WARNING: New LR flags detected but --override-scheduling was not set.             !!!")
                    print(f"!!!   Your new LR ({args.starting_lr}) / Min LR ({args.min_lr}) WILL BE IGNORED.             !!!")
                    print(f"!!!   Loading old schedule (LR: {old_lr}, Min LR: {old_min_lr}).                       !!!")
                    print("!!!   To use your new LR flags, add --override-scheduling to your command.            !!!")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

                print("Resuming learning rate scheduler state.")
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warning: Failed to load scheduler state: {e}. Continuing with potentially incorrect LR.")

            # --- START FIX: Combined logic for overriding or no scheduler state ---
            elif args.override_scheduling or not checkpoint_has_scheduler:
                # OVERRIDE or NO SCHEDULER IN CKPT: Start scheduler fresh based on args and start_epoch
                if args.override_scheduling and checkpoint_has_scheduler:
                    print("INFO: --override-scheduling detected. Ignoring checkpoint's scheduler state.")
                elif not checkpoint_has_scheduler:
                    print("Warning: No scheduler state found in checkpoint. Initializing new schedule.")

                print(f"INFO: Initializing new schedule with Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
                # Optimizer LR should already be correct from initialization above

                # Set the step count to where it should be for the resumed epoch
                steps_per_epoch = dataloader_len // args.accumulation_steps if dataloader_len > 0 else 0
                # Ensure last_epoch doesn't go below -1
                scheduler.last_epoch = max(-1, start_epoch * steps_per_epoch - 1)
            # --- END FIX ---
            ### --- End Scheduler Resuming Logic --- ###

        print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}.") # Adjusted message


    else: # Not resuming, starting fresh
        # Need to ensure vocab_size is in the config used to create the model
        if 'vocab_size' not in config:
            if current_vocab_size:
                config['vocab_size'] = current_vocab_size
            else:
                 raise ValueError("Cannot determine vocab_size for new model.")
        model = ChronosCore(config).to(device)
        optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
        model_config = AttrDict(config) # Use the CLI args config

        # --- Initialize AMP GradScaler ---
        if use_amp:
            scaler = GradScaler()
            print("INFO: Automatic Mixed Precision (AMP) ENABLED for training.")
        # --- Initialize Scheduler ---
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0
        if not args.disable_lr_schedule and num_update_steps > 0:
            print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)

    # --- Start Training Loop ---
    model.train()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Zero gradients *before* starting the loop, especially important when resuming
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1} / {args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        total_loss = 0.0
        total_ponder_cost = 0.0
        # Flag to track if backward was called in the current accumulation cycle
        backward_called_in_cycle = False

        for i, batch in enumerate(pbar):
            # Handle potential None batch from collate_fn if it was empty
            if batch is None:
                print(f"Warning: Skipping empty batch at step {i}.")
                continue

            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            # --- AMP autocast context ---
            # Use autocast context manager for forward pass
            with autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                cross_entropy_loss = outputs["loss"]
                ponder_cost = outputs["ponder_cost"]

                combined_loss = None
                # Check for valid loss components BEFORE combining
                ce_valid = cross_entropy_loss is not None and not torch.isnan(cross_entropy_loss) and not torch.isinf(cross_entropy_loss)
                pc_valid = ponder_cost is not None and not torch.isnan(ponder_cost) and not torch.isinf(ponder_cost)

                if ce_valid and pc_valid:
                    combined_loss = cross_entropy_loss + args.ponder_loss_weight * ponder_cost
                elif ce_valid: # Only CE is valid
                    if i % args.accumulation_steps == 0: # Print only once per update step
                        print(f"\nWarning: Ponder cost is NaN/Inf at step {i}. Using only CrossEntropy loss for this step.")
                    combined_loss = cross_entropy_loss
                elif not ce_valid: # CE loss is invalid, skip backward
                    if i % args.accumulation_steps == 0:
                        print(f"\nWarning: CrossEntropy loss is NaN/Inf at step {i}. Skipping backward pass for this step.")
                    combined_loss = None # Ensure it's None


            # --- Backward Pass ---
            if combined_loss is not None:
                loss_to_backward = combined_loss / args.accumulation_steps

                if use_amp:
                    # scaler.scale automatically checks for enabled state
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                backward_called_in_cycle = True # Mark that backward was successful

                # Accumulate display stats only if loss was valid
                if ce_valid:
                    total_loss += cross_entropy_loss.item()
                if pc_valid:
                    total_ponder_cost += ponder_cost.item()


            # --- Optimizer Step (End of Accumulation Cycle) ---
            if (i + 1) % args.accumulation_steps == 0:
                # Only proceed if backward was called at least once in this cycle
                if backward_called_in_cycle:
                    # --- LTM Update (Before Optimizer Step) ---
                    # Accessing .grad after backward is fine, but check if topk_vals was valid
                    ltm_grads = None
                    if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad:
                        ltm_grads = outputs["topk_vals"].grad

                    if ltm_grads is not None:
                        # Make a copy before potentially modifying in-place with unscaling
                        ltm_grads_copy = ltm_grads.detach().clone()
                        if use_amp:
                            # Manually unscale LTM grads *if* the scaler is currently scaled
                            # We need the scale *before* unscale_(optimizer) is called
                            current_scale = scaler.get_scale()
                            if current_scale != 1.0: # Check if scaling is active
                                # Use _assert_state to avoid errors if scale is 1.0
                                if scaler._enabled and scaler._scale is not None:
                                    assert current_scale > 0.0 # Should always be true if scale != 1.0
                                    ltm_grads_copy = ltm_grads_copy / current_scale # Unscale the copy
                                else: # If scaler somehow disabled or scale is None, don't unscale
                                    print(f"\nWarning: Scaler state inconsistent at step {i}, cannot unscale LTM grads.")

                        model.ltm.inner_update(outputs["topk_idx"], ltm_grads_copy, current_lr=args.ltm_lr, source=LTMModule.SRC_TRAINING_DATA)


                    # --- Optimizer Step ---
                    if use_amp:
                        # Unscale gradients - performs inf/NaN checks
                        scaler.unscale_(optimizer)

                        if args.grad_clip > 0:
                            # Clip gradients *after* unscaling
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                        # Step the optimizer - scaler skips if unscale_ found inf/NaN
                        scaler.step(optimizer)

                        # Update the scale factor for the next iteration
                        scaler.update()
                    else: # Not using AMP
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()

                    # Step the learning rate scheduler *after* the optimizer step
                    if scheduler:
                        scheduler.step()

                    # Zero gradients for the next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)

                    # Reset flag for the next cycle
                    backward_called_in_cycle = False

                else: # Backward was not called in this cycle (all losses were NaN/Inf)
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss(es) in accumulation cycle.")
                    # Still need to zero gradients that might exist from previous cycles if resuming
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False # Reset flag

            # --- Update Progress Bar ---
            display_steps = i + 1
            avg_loss = total_loss / display_steps if display_steps > 0 else 0.0
            avg_ponder = total_ponder_cost / display_steps if display_steps > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "lr": f"{current_lr:.2e}"
            })

        # --- End of Epoch ---
        ckpt_path = os.path.join(args.out_dir, f"chronos_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} complete. Saving training checkpoint to {ckpt_path}")
        # Make sure config saved in checkpoint reflects current args if scheduling was overridden
        # Also include train_data_path and ensure vocab_size is present
        config_to_save = dict(model.config) # Get config directly from model
        config_to_save['starting_lr'] = args.starting_lr
        config_to_save['min_lr'] = args.min_lr
        config_to_save['disable_lr_schedule'] = args.disable_lr_schedule
        config_to_save['train_data_path'] = args.train # Save train path
        # Ensure vocab_size is saved (should be in model.config by now)
        if 'vocab_size' not in config_to_save:
            print(f"CRITICAL WARNING: vocab_size missing from model config before saving epoch {epoch+1} checkpoint!")


        # Prepare state dicts for saving
        scaler_state = scaler.state_dict() if use_amp and scaler is not None else None
        scheduler_state = scheduler.state_dict() if scheduler is not None else None

        torch.save({
            'completed_epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'scaler_state_dict': scaler_state,
            'config': config_to_save, # Save potentially updated config
        }, ckpt_path)

    # --- End of Training ---
    final_save_path = os.path.join(args.out_dir, MODEL_WEIGHTS_NAME)
    print(f"\nTraining finished. Saving final inference model to {final_save_path}")
    # Make sure final config reflects current args if scheduling was overridden
    final_config_to_save = dict(model.config) # Get config from model
    final_config_to_save['starting_lr'] = args.starting_lr
    final_config_to_save['min_lr'] = args.min_lr
    final_config_to_save['disable_lr_schedule'] = args.disable_lr_schedule
    final_config_to_save['train_data_path'] = args.train # Also save train path here
    # Ensure vocab_size is saved
    if 'vocab_size' not in final_config_to_save:
        print(f"CRITICAL WARNING: vocab_size missing from model config before saving final model!")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': final_config_to_save
    }, final_save_path)

    tokenizer.save_pretrained(args.out_dir)
    print(f"Tokenizer files saved to {args.out_dir}")

    if args.quantize_on_complete:
        print("\n--- Training Complete: Starting On-the-Fly Quantization ---")
        # Quantize to a new directory for clarity, e.g., './my_model-INT4'
        quantize_out_dir = args.out_dir.rstrip('/\\') + f"-{args.qtype}"
        quantize(args, device, model, tokenizer, quantize_out_dir)

# --- finetune, merge_lora, quantize, chat functions remain the same as the previous version ---

# <<< FINETUNE Function (Keep from previous version) >>>
def finetune(args, device, tokenizer):
    if not _HAS_PEFT: raise ImportError("Please install 'peft' for fine-tuning.")
    print("Running in FINETUNE mode with LoRA...")

    # Load the base model and its config from the specified directory
    model, model_config = load_full_model_with_config(args.model_path, device)

    lora_r = args.lora_r
    if args.finetune_unlock_percent is not None: # Check if flag was actually used
        if args.lora_r != 8: # Default value check
            print(f"Warning: Both --lora_r ({args.lora_r}) and --finetune-unlock-percent were specified. Prioritizing --lora_r.")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            target_modules = ["qproj", "in_proj", "h_to_context", "l_to_out", "h_halt_proj"]
            lora_param_sum_per_r = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(tm in name for tm in target_modules):
                    lora_param_sum_per_r += module.in_features + module.out_features

            target_trainable_count = total_params * (args.finetune_unlock_percent / 100.0)
            if lora_param_sum_per_r > 0:
                estimated_r = target_trainable_count / lora_param_sum_per_r
                lora_r = max(1, int(round(estimated_r)))
                print(f"Targeting ~{args.finetune_unlock_percent}% trainable parameters. Estimated LoRA rank 'r' = {lora_r}")
            else:
                print("Warning: Could not find target modules for LoRA. Using default r=8.")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qproj", "in_proj", "h_to_context", "l_to_out", "h_halt_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"], # Ensure LTM can still be updated directly
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataloader = create_dataloader(
        args.train, tokenizer, model_config.max_length, args.batch_size,
        tokenizer.pad_token_id, kayla_mode=args.kayla, num_workers=args.num_workers
    )
    optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr) # Only trainable params will have grads
    os.makedirs(args.out_dir, exist_ok=True)

    scaler = None
    use_amp = args.amp and _HAS_AMP
    if use_amp:
        scaler = GradScaler()
        print("INFO: Automatic Mixed Precision (AMP) ENABLED for fine-tuning.")

    scheduler = None
    if not args.disable_lr_schedule:
        dataloader_len = len(dataloader)
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0
        if num_update_steps > 0:
            print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED for finetuning. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
        else:
            print("Warning: Cannot enable LR schedule, dataset might be too small or empty.")


    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}")
        total_loss = 0.0
        total_ponder_cost = 0.0
        backward_called_in_cycle = False

        for i, batch in enumerate(pbar):
            if batch is None: continue # Skip empty batches

            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            with autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                cross_entropy_loss = outputs["loss"]
                ponder_cost = outputs["ponder_cost"]

                combined_loss = None
                ce_valid = cross_entropy_loss is not None and not torch.isnan(cross_entropy_loss) and not torch.isinf(cross_entropy_loss)
                pc_valid = ponder_cost is not None and not torch.isnan(ponder_cost) and not torch.isinf(ponder_cost)

                if ce_valid and pc_valid:
                    combined_loss = cross_entropy_loss + args.ponder_loss_weight * ponder_cost
                elif ce_valid:
                    if i % args.accumulation_steps == 0:
                        print(f"\nWarning: Ponder cost is NaN/Inf at step {i}. Using only CrossEntropy loss.")
                    combined_loss = cross_entropy_loss
                elif not ce_valid:
                    if i % args.accumulation_steps == 0:
                        print(f"\nWarning: CrossEntropy loss is NaN/Inf at step {i}. Skipping backward pass.")
                    combined_loss = None


            if combined_loss is not None:
                loss_to_backward = combined_loss / args.accumulation_steps

                if use_amp:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                backward_called_in_cycle = True

                if ce_valid: total_loss += cross_entropy_loss.item()
                if pc_valid: total_ponder_cost += ponder_cost.item()


            if (i + 1) % args.accumulation_steps == 0:
                if backward_called_in_cycle:
                    # LTM Update (Needs careful handling with PEFT)
                    ltm_grads = None
                    if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad:
                        ltm_grads = outputs["topk_vals"].grad

                    if ltm_grads is not None:
                        # Access the base model's LTM module directly
                        base_ltm = model.base_model.ltm
                        ltm_grads_copy = ltm_grads.detach().clone() # Use a copy

                        if use_amp:
                            current_scale = scaler.get_scale()
                            if current_scale != 1.0 and scaler._enabled and scaler._scale is not None:
                                assert current_scale > 0.0
                                ltm_grads_copy = ltm_grads_copy / current_scale
                            elif current_scale != 1.0:
                                print(f"\nWarning: Scaler state inconsistent at step {i}, cannot unscale LTM grads.")

                        base_ltm.inner_update(outputs["topk_idx"], ltm_grads_copy, current_lr=args.ltm_lr, source=LTMModule.SRC_TRAINING_DATA)

                    # Optimizer Step
                    if use_amp:
                        scaler.unscale_(optimizer)
                        if args.grad_clip > 0:
                            # Only clip trainable parameters (those adapted by LoRA + saved modules like LTM)
                            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.grad_clip)
                        optimizer.step()

                    if scheduler:
                        scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False
                else:
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss(es) in accumulation cycle.")
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False

            display_steps = i + 1
            avg_loss = total_loss / display_steps if display_steps > 0 else 0.0
            avg_ponder = total_ponder_cost / display_steps if display_steps > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "lr": f"{current_lr:.2e}"
            })

    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    # Note: Only the adapter (+ saved modules like LTM) is saved here.

# <<< MERGE LORA Function (Keep from previous version) >>>
def merge_lora(args, device, tokenizer):
    if not _HAS_PEFT: raise ImportError("Please install 'peft' for merging.")
    print("Running in MERGE-LORA mode...")

    print(f"Loading base model from {args.model_path}...")
    base_model, _ = load_full_model_with_config(args.model_path, device)

    print(f"Loading LoRA adapter from {args.lora_adapter_path}...")
    # Load adapter onto the base model
    try:
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        print("Ensure the adapter path is correct and compatible with the base model.")
        sys.exit(1)


    print("Merging adapter into the base model...")
    try:
        # merge_and_unload returns the merged base model
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Error merging LoRA adapter: {e}")
        sys.exit(1)

    # Save the merged model to a new, self-contained directory
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, MODEL_WEIGHTS_NAME)
    print(f"Saving merged model to {output_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(model.config) # Save config from the (now merged) model
    }, output_path)

    # Copy tokenizer files from the original base model directory
    try:
        # Load the tokenizer associated with the original base model
        original_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        original_tokenizer.save_pretrained(args.out_dir)
        print(f"Tokenizer files copied to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Could not copy tokenizer files from {args.model_path}: {e}")

    print("Merge complete.")

# <<< QUANTIZE Function (Keep from previous version) >>>
def quantize(args, device, model=None, tokenizer=None, out_dir=None):
    print(f"Running in QUANTIZE mode with {args.qtype} precision...")

    # Allow passing in an already-loaded model (e.g., from train --quantize-on-complete)
    if model is None or tokenizer is None:
        if not args.model_path:
            raise ValueError("--model-path is required for quantize mode when model/tokenizer not provided.")
        print(f"Loading full-precision model from {args.model_path}...")
        # Tokenizer is loaded from the same directory as the model
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model, _ = load_full_model_with_config(args.model_path, device)
        except Exception as e:
            print(f"Error loading model or tokenizer from {args.model_path}: {e}")
            sys.exit(1)


    # Determine output directory
    if out_dir is None:
        if not args.out_dir:
            # Default to creating a new dir next to the source, e.g., './my_model-INT4'
            # Use model_path if available, otherwise fallback (though less likely now)
            source_dir = args.model_path if args.model_path else "./chronos_model"
            out_dir = source_dir.rstrip('/\\') + f"-{args.qtype}"
        else:
            out_dir = args.out_dir

    export_and_quantize_model(out_dir, model, tokenizer, qtype=args.qtype)

# <<< CHAT Function (Keep from previous version) >>>
def chat(args, device, tokenizer):
    print("Running in CHAT mode...")

    model = None
    shadow_model = None
    config = None
    is_quantized = False
    inference_device = "cpu" # Default for quantized models
    ltm_has_been_updated = False # Flag to track if we need to save

    # Determine if loading quantized or full model from the same --model-path
    if not args.model_path or not os.path.isdir(args.model_path):
        print(f"Error: Model directory not found or invalid at {args.model_path}")
        sys.exit(1)

    try:
        npz_files = [f for f in os.listdir(args.model_path) if f.endswith('.npz')]
    except FileNotFoundError:
        print(f"Error: Model directory not found at {args.model_path}")
        sys.exit(1)

    if npz_files:
        if not _HAS_KERNEL:
            print("ERROR: Cannot run quantized chat without the C++ kernel.")
            return
        model, config = load_quantized(args.model_path)
        is_quantized = True
        print(f"Loaded quantized model with {model.qtype} weights.")

        if args.device == 'vulkan':
            if not _HAS_VULKAN:
                print("WARNING: Vulkan support not found in kernel. Falling back to CPU.")
            else:
                inference_device = "vulkan"
                print("INFO: Using Vulkan for inference.")
        else: # Default to CPU if not Vulkan
            inference_device = "cpu"


        if args.enable_quantized_learning:
            if not args.shadow_model_path:
                raise ValueError("To enable learning on a quantized model, you must provide the original full-precision model directory via --shadow-model-path.")
            print("Loading full-precision 'shadow' model for online learning...")
            # Shadow model is loaded with its own config, which should match the quantized one
            try:
                shadow_model, shadow_config = load_full_model_with_config(args.shadow_model_path, device)
                # Basic config check
                if shadow_config.context_dim != config.context_dim or shadow_config.ltm_slots != config.ltm_slots:
                    print("Warning: Shadow model config differs significantly from quantized config. Learning might be unstable.")
            except Exception as e:
                print(f"Error loading shadow model from {args.shadow_model_path}: {e}")
                sys.exit(1)


            # Sync the quantized model's initial LTM state to the shadow model
            shadow_model.ltm.load_state_dict(model.ltm.state_dict())
            shadow_model.eval()

    else: # Load full precision model
        try:
            model, config = load_full_model_with_config(args.model_path, device)
            inference_device = device # Use the main PyTorch device
        except Exception as e:
            print(f"Error loading full precision model from {args.model_path}: {e}")
            sys.exit(1)


    if args.ltm_lora_path:
        print(f"LTM online learning is ACTIVE. Updates will be stored separately at: {args.ltm_lora_path}")
        # The model to update is the shadow model if it exists, otherwise the base model
        updatable_model = shadow_model if is_quantized and args.enable_quantized_learning else model
        if updatable_model is None:
            print("Warning: LTM LoRA path specified but no updatable model found (e.g., quantized without learning enabled). Updates will NOT be saved.")
        else:
            updatable_model.ltm.accumulate_deltas = True
            if os.path.exists(args.ltm_lora_path):
                print("Loading existing LTM deltas...")
                try:
                    deltas = torch.load(args.ltm_lora_path)
                    # Apply loaded deltas to the LTM values and the delta accumulator
                    updatable_model.ltm.vals.data.add_(deltas.to(updatable_model.ltm.vals.device))
                    updatable_model.ltm.ltm_deltas.data = deltas.to(updatable_model.ltm.ltm_deltas.device)
                    # If quantized, sync the now-updated shadow LTM back to the live model
                    if is_quantized and args.enable_quantized_learning:
                        model.ltm.load_state_dict(updatable_model.ltm.state_dict())
                except Exception as e:
                    print(f"Warning: Failed to load or apply LTM deltas from {args.ltm_lora_path}: {e}")

    elif not is_quantized or args.enable_quantized_learning:
        print("LTM online learning is ACTIVE. Updates will modify model weights directly in memory.")

    if not is_quantized:
        model.eval()

    ltm_scheduler = None
    # Setup LTM scheduler if not in static mode and learning is enabled
    if not args.static_ltm_lr and (not is_quantized or args.enable_quantized_learning):
        print("INFO: Using Cosine Annealing schedule for LTM updates.")
        print(f"             - Max LR: {args.ltm_lr:.2e}, Min LR: {args.ltm_schedule_min_lr:.2e}, Cycle Steps: {args.ltm_schedule_steps}")
        # Schedulers need an optimizer, so we create a dummy one for the LTM LR.
        # We will call scheduler.step() manually, but never optimizer.step().
        dummy_param = nn.Parameter(torch.tensor(0.0)) # Needs to be Parameter
        # Use the main LTM LR as the MAX LR for the schedule
        ltm_optimizer = torch.optim.SGD([dummy_param], lr=args.ltm_lr)
        ltm_scheduler = CosineAnnealingLR(
            ltm_optimizer,
            T_max=args.ltm_schedule_steps,
            eta_min=args.ltm_schedule_min_lr
        )

    # --- MODIFICATION START: Initialize AMP scaler and dummy optimizer for chat learning ---
    scaler = None
    dummy_optimizer = None
    # Enable AMP for learning if requested AND possible (CUDA available AND (full model OR quantized learning enabled))
    use_amp = args.amp and _HAS_AMP and (not is_quantized or args.enable_quantized_learning)

    if use_amp:
        scaler = GradScaler()
        # Create a dummy optimizer for the scaler to track state (NaNs/Infs)
        # This is necessary because chat learning doesn't have a persistent optimizer
        dummy_param_amp = nn.Parameter(torch.tensor(0.0)).to(device) # Needs to be Parameter and on device
        dummy_optimizer = torch.optim.SGD([dummy_param_amp], lr=1.0) # Dummy optimizer for AMP scaler
        print("INFO: Automatic Mixed Precision (AMP) ENABLED for online learning.")
    # --- MODIFICATION END ---

    print("\nWelcome to Chronos Chat. Type 'exit' or 'quit' to end.")
    ### MODIFICATION START: Add help text for new filter command ###
    print("Use '/filter time=-<seconds>' or '/filter source=<id>' to constrain memory.")
    print("Example: /filter time=-3600  (memories from the last hour)")
    ### MODIFICATION END ###
    if _HAS_KEYBOARD:
        print("Press Ctrl+X to stop generation at any time.")
    print("="*50)

    try:
        ### MODIFICATION START: Initialize filter variables ###
        min_ts_filter = 0.0
        source_id_filter = None
        ### MODIFICATION END ###
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ["exit", "quit"]:
                break

            ### MODIFICATION START: Simple command parser for filtering ###
            if prompt.startswith('/filter'):
                parts = prompt.split()
                try:
                    for part in parts[1:]:
                        key, value = part.split('=')
                        if key == 'time':
                            # time=-3600 means "minimum timestamp is 3600 seconds ago"
                            time_offset = float(value)
                            if time_offset <= 0:
                                min_ts_filter = time.time() + time_offset # Add negative offset
                                print(f"[INFO: Memory filtered to events after {time.ctime(min_ts_filter)}]")
                            else:
                                print("[ERROR: Time filter requires a negative offset, e.g., time=-3600]")
                        elif key == 'source':
                            src_id = int(value)
                            if src_id in [LTMModule.SRC_UNKNOWN, LTMModule.SRC_USER_INTERACTION, LTMModule.SRC_TRAINING_DATA]:
                                source_id_filter = src_id
                                print(f"[INFO: Memory filtered to source ID: {source_id_filter}]")
                            else:
                                print(f"[ERROR: Invalid source ID. Use {LTMModule.SRC_UNKNOWN}, {LTMModule.SRC_USER_INTERACTION}, or {LTMModule.SRC_TRAINING_DATA}]")
                except Exception as e: # Catch broader errors like split issues
                    print(f"[ERROR: Invalid filter format. Use 'time=-<seconds>' or 'source=<id>'. Details: {e}]")
                continue
            elif prompt == '/filter reset':
                min_ts_filter = 0.0
                source_id_filter = None
                print("[INFO: Memory filters have been reset.]")
                continue
            ### MODIFICATION END ###

            prompt_format = f"### Instruction:\n{prompt}\n\n### Response:\n"
            # Always use the main device for tokenization, even for quantized models
            # Ensure tokenizer exists before encoding
            if tokenizer is None:
                print("Error: Tokenizer not loaded. Cannot proceed.")
                break
            prompt_ids = tokenizer.encode(prompt_format, return_tensors="pt").to(device)


            print("\nChronos: ", end="", flush=True)
            response_ids = []

            # Initialize RNN states on the appropriate device
            rnn_device = "cpu" if is_quantized else device
            h_state = torch.zeros(1, config.h_hidden, device=rnn_device)
            l_state = torch.zeros(1, config.l_hidden, device=rnn_device)

            current_ids = prompt_ids
            # Generation loop uses no_grad
            with torch.no_grad():
                for i in range(args.max_new_tokens):
                    if _HAS_KEYBOARD and keyboard.is_pressed('ctrl+x'):
                        print("\n[Generation interrupted by user.]", end="", flush=True)
                        break

                    # Input for the model call
                    # Pass CPU tensors to quantized model, CUDA/MPS to full model
                    model_input_ids = current_ids.cpu() if is_quantized else current_ids.to(device)


                    if is_quantized:
                        # Quantized model expects CPU input ids and states
                        outputs = model(model_input_ids, h_state.cpu(), l_state.cpu(), device=inference_device, min_timestamp=min_ts_filter, source_filter=source_id_filter)
                        # Update CPU states
                        h_state, l_state = outputs['h_state'], outputs['l_state']
                    else:
                        # Full model expects inputs on its device
                        outputs = model(model_input_ids.to(device), min_timestamp=min_ts_filter, source_filter=source_id_filter)
                        # No explicit state passing needed for full model forward


                    logits = outputs["logits"].to(device) # Ensure logits are on main device for sampling
                    next_token_logits = logits[:, -1, :]

                    # Simple argmax sampling (can be replaced with more sophisticated sampling)
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)


                    if next_token_id.item() == tokenizer.eos_token_id:
                        break

                    response_ids.append(next_token_id.item())
                    decoded_token = tokenizer.decode([next_token_id.item()])

                    # Stop generation if a special token like ### is encountered
                    # (This check might need refinement based on tokenizer specifics)
                    if "###" in decoded_token and len(decoded_token) <= 5:
                        break

                    print(decoded_token, end="", flush=True)

                    # Append the new token for the next iteration's input
                    current_ids = torch.cat([current_ids, next_token_id.to(current_ids.device)], dim=1)
                    # Truncate input_ids if they exceed max_length (important for long chats)
                    if current_ids.shape[1] > config.max_length:
                        current_ids = current_ids[:, -config.max_length:]


            # --- Online Learning Step ---
            learning_enabled = not is_quantized or args.enable_quantized_learning
            if len(response_ids) > 0 and learning_enabled:
                update_model = shadow_model if is_quantized else model
                target_device = device # Learning happens on the main device

                if is_quantized:
                    print("\n[Updating LTM via shadow model...]", end="", flush=True)

                update_model.train() # Set to train mode to enable gradients
                with torch.enable_grad(): # Ensure grads are enabled
                    # Prepare inputs for the learning pass
                    full_sequence = torch.cat([prompt_ids[0], torch.tensor(response_ids, device=target_device)], dim=0).unsqueeze(0)
                    # Create labels, masking out the prompt part
                    labels = torch.cat([torch.full_like(prompt_ids[0], -100), torch.tensor(response_ids, device=target_device)], dim=0).unsqueeze(0)

                    # Ensure sequence length doesn't exceed model capacity for learning pass
                    if full_sequence.shape[1] > config.max_length:
                        full_sequence = full_sequence[:, -config.max_length:]
                        labels = labels[:, -config.max_length:]


                    # Zero grads before the learning forward pass
                    optimizer_to_zero = dummy_optimizer if use_amp else None # For AMP scaler state
                    if optimizer_to_zero: optimizer_to_zero.zero_grad(set_to_none=True)
                    update_model.zero_grad(set_to_none=True)

                    # Forward pass for learning with AMP context
                    with autocast(enabled=use_amp):
                        outputs = update_model(input_ids=full_sequence, labels=labels)
                        cross_entropy_loss = outputs["loss"]
                        ponder_cost = outputs["ponder_cost"]
                        # Use get with default for ponder weight in case it's missing in older configs
                        ponder_loss_weight = update_model.config.get('ponder_loss_weight', 0.01)

                        combined_loss = None
                        ce_valid = cross_entropy_loss is not None and not torch.isnan(cross_entropy_loss) and not torch.isinf(cross_entropy_loss)
                        pc_valid = ponder_cost is not None and not torch.isnan(ponder_cost) and not torch.isinf(ponder_cost)

                        if ce_valid and pc_valid:
                            print(f"[CE Loss: {cross_entropy_loss.item():.3f}, Ponder Cost: {ponder_cost.item():.2f}]", end="", flush=True)
                            combined_loss = cross_entropy_loss + ponder_loss_weight * ponder_cost
                        elif ce_valid:
                            print(f"[CE Loss: {cross_entropy_loss.item():.3f}, Ponder Cost: NaN/Inf]", end="", flush=True)
                            combined_loss = cross_entropy_loss
                        else:
                            print(f"[CE Loss: NaN/Inf, Skipping LTM update]", end="", flush=True)
                            combined_loss = None # Ensure it's None

                    # Backward pass for LTM gradients
                    if combined_loss is not None:
                        if use_amp:
                            scaler.scale(combined_loss).backward()
                        else:
                            combined_loss.backward()

                        ltm_grads = None
                        if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad:
                            ltm_grads = outputs["topk_vals"].grad

                        if ltm_grads is not None:
                            ltm_grads_copy = ltm_grads.detach().clone() # Use a copy

                            # Determine the current LTM learning rate
                            current_ltm_lr = args.ltm_lr # Default to static LR
                            if ltm_scheduler:
                                current_ltm_lr = ltm_scheduler.get_last_lr()[0]
                                print(f"[LTM LR: {current_ltm_lr:.2e}]", end="", flush=True)
                                ltm_scheduler.step() # Step scheduler *after* using the LR

                            # Unscale LTM grads if using AMP
                            if use_amp:
                                current_scale = scaler.get_scale()
                                if current_scale != 1.0 and scaler._enabled and scaler._scale is not None:
                                    assert current_scale > 0.0
                                    ltm_grads_copy = ltm_grads_copy / current_scale
                                elif current_scale != 1.0:
                                    print(f"\nWarning: Scaler inconsistent during LTM update, grads might be scaled.")

                            # Perform the LTM inner update
                            update_model.ltm.inner_update(outputs["topk_idx"], ltm_grads_copy, current_lr=current_ltm_lr, source=LTMModule.SRC_USER_INTERACTION)
                            ltm_has_been_updated = True # Mark that a change has occurred

                        # --- Necessary steps for AMP scaler state even without optimizer step ---
                        if use_amp:
                            # Unscale the dummy optimizer to check for infs/NaNs
                            scaler.unscale_(dummy_optimizer)
                            # Step dummy optimizer (will be skipped by scaler if infs found)
                            scaler.step(dummy_optimizer)
                            # Update scale for next iteration
                            scaler.update()

                        # --- Important: Zero grads on the update_model ---
                        # We don't step a real optimizer, but need to clear grads
                        update_model.zero_grad(set_to_none=True)

                        # Copy the updated LTM weights back to the live quantized model
                        if is_quantized:
                            model.ltm.load_state_dict(update_model.ltm.state_dict())

                    else: # Combined loss was None (e.g., NaN)
                        # Still need to handle AMP scaler state if used
                        if use_amp:
                            # Even if backward wasn't called, step/update cycle might be needed
                            # If backward wasn't called, unscale_ is a no-op, step is skipped, update proceeds
                            scaler.unscale_(dummy_optimizer)
                            scaler.step(dummy_optimizer)
                            scaler.update()
                        # Zero any potential stale grads
                        update_model.zero_grad(set_to_none=True)


                update_model.eval() # Set back to eval mode after learning step
                if is_quantized:
                    print("[Done]", end="", flush=True)

            print("\n\n" + "="*50)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected. Exiting chat.]")

    finally:
        # --- SAVE ON EXIT LOGIC ---
        updatable_model = shadow_model if is_quantized and args.enable_quantized_learning else model

        # Check if we have a model capable of updates
        can_update = updatable_model is not None and (not is_quantized or args.enable_quantized_learning)

        if can_update and args.ltm_lora_path and hasattr(updatable_model.ltm, 'accumulate_deltas') and updatable_model.ltm.accumulate_deltas:
            # Save accumulated deltas if they exist
            if torch.any(updatable_model.ltm.ltm_deltas != 0):
                print(f"\nSaving LTM memory deltas to {args.ltm_lora_path}...")
                try:
                    torch.save(updatable_model.ltm.ltm_deltas.cpu(), args.ltm_lora_path)
                    print("✅ Deltas saved.")
                except Exception as e:
                    print(f"Error saving LTM deltas: {e}")
            else:
                print("\nNo new LTM updates to save as LoRA.")

        elif can_update and not args.ltm_lora_path and ltm_has_been_updated:
            # Prompt to save directly incorporated updates
            if not is_quantized: # Save full precision model directly
                while True:
                    response = input(f"Do you want to save the learned LTM updates back to '{args.model_path}'? (y/n): ").lower()
                    if response in ["y", "yes"]:
                        print(f"\nSaving updated model to {args.model_path}...")
                        output_weights_path = os.path.join(args.model_path, MODEL_WEIGHTS_NAME)
                        try:
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'config': dict(model.config) # Save current config
                            }, output_weights_path)
                            print("✅ Save complete.")
                        except Exception as e:
                            print(f"Error saving model: {e}")
                        break
                    elif response in ["n", "no"]:
                        print("Changes will be discarded. Exiting.")
                        break
            else: # Need to re-quantize the shadow model
                output_dir = args.model_path # Overwrite the existing quantized model dir
                while True:
                    response = input(f"Do you want to save these LTM changes by re-quantizing the model to '{output_dir}'? (y/n): ").lower()
                    if response in ["y", "yes"]:
                        print(f"\nRe-quantizing model with updated LTM to {output_dir}...")
                        try:
                            # Ensure the shadow model is on CPU for quantization if needed
                            shadow_model.cpu()
                            # Make sure tokenizer is available for export
                            if tokenizer is None:
                                print("Error: Cannot re-quantize without a loaded tokenizer.")
                                break
                            export_and_quantize_model(output_dir, shadow_model, tokenizer, model.qtype)
                            print("✅ Re-quantization complete.")
                            # Move shadow model back to original device if needed
                            shadow_model.to(device)
                        except Exception as e:
                            print(f"Error during re-quantization: {e}")
                        break
                    elif response in ["n", "no"]:
                        print("Changes will be discarded. Exiting.")
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")

        elif ltm_has_been_updated:
            print("\nLTM was updated, but saving is disabled (e.g., quantized mode without --enable-quantized-learning). Changes will be lost.")


# <<< main function with REVERTED tokenizer logic >>>
def main():
    parser = argparse.ArgumentParser(description="Chronos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora"], help="Operation mode.")

    # --- Data and Path Arguments (Universal) ---
    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, default=None, help="[Train/Finetune] Path to training JSON or JSONL file.")
    # <<< REVERTED CHANGE: Made --model-path NOT required for train mode >>>
    path_group.add_argument("--model-path", type=str, default=None, help="Path to the model directory (required for all modes except 'train' unless resuming).")
    path_group.add_argument("--out-dir", type=str, default="./chronos_model", help="[Train/Finetune/Merge/Quantize] Directory to save the new model/adapter.")
    path_group.add_argument("--lora-adapter-path", type=str, default=None, help="[Merge/Finetune] Path to the LoRA adapter directory.") # Corrected help text
    path_group.add_argument("--tokenizer-path", type=str, default="microsoft/phi-2", help="[Train] Path or HF name of the tokenizer to use for a new model.")
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="[Train] Path to a specific training checkpoint .pt file to resume from.")
    path_group.add_argument("--shadow-model-path", type=str, default=None, help="[Chat] Path to the original full-precision model dir, required for online learning with a quantized model.")


    # --- Model Architecture Arguments (for Training) ---
    arch_group = parser.add_argument_group('Architecture (for --mode train)')
    arch_group.add_argument("--context_dim", type=int, default=512)
    arch_group.add_argument("--persistent_dim", type=int, default=128)
    arch_group.add_argument("--ltm_slots", type=int, default=2048)
    arch_group.add_argument("--ltm_key_dim", type=int, default=128)
    arch_group.add_argument("--ltm_val_dim", type=int, default=128)
    arch_group.add_argument("--h_hidden", type=int, default=512)
    arch_group.add_argument("--l_hidden", type=int, default=512)
    ### MODIFICATION START ###
    arch_group.add_argument("--max_h_steps", type=int, default=10, help="[HRM] Maximum number of high-level refinement steps.")
    arch_group.add_argument("--max_l_steps", type=int, default=10, help="[HRM] Maximum number of low-level iterations before forcing completion.")
    arch_group.add_argument("--l_conv_atol", type=float, default=1e-5, help="[HRM] Absolute tolerance for checking L-module state convergence.")
    ### MODIFICATION END ###
    arch_group.add_argument("--ltm_topk", type=int, default=4, help="Number of LTM slots to retrieve per token.")
    arch_group.add_argument("--max_length", type=int, default=1024)
    arch_group.add_argument("--auto-max-length", action="store_true", help="Automatically scan the dataset to find the longest sequence and set it as max_length.")

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training and Finetuning')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--accumulation-steps", type=int, default=1, help="Simulates a larger batch size.")
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine annealing.")
    train_group.add_argument("--disable-lr-schedule", action="store_true", help="Use a fixed LR instead of cosine annealing.")
    train_group.add_argument("--ltm_lr", type=float, default=1e-2, help="[Static] LR for LTM updates, or [Scheduled] MAX LR for the LTM cosine schedule.")
    train_group.add_argument("--kayla", action="store_true", help="Enable Kayla-style instruction tuning (with thought-process).")
    train_group.add_argument("--lora_r", type=int, default=8, help="[Finetune] LoRA rank.")
    train_group.add_argument("--lora_alpha", type=int, default=16, help="[Finetune] LoRA alpha.")
    train_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="[Finetune] Target percentage of params to train (e.g., 1.5 for 1.5%). Overrides --lora_r.")
    train_group.add_argument("--quantize-on-complete", action="store_true", help="[Train] Automatically quantize after training.")
    train_group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value. Set to 0 to disable.")
    ### MODIFICATION START ###
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01, help="[HRM] Weight for the ponder cost auxiliary loss.")
    train_group.add_argument("--override-scheduling", action="store_true", help="[Train] If resuming, ignore the scheduler state in the checkpoint and use the new LR args.")
    # <<< MODIFIED: Added num_workers argument >>>
    train_group.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading. Recommended: 2 or 4 for GPU training.")
    train_group.add_argument("--amp", action="store_true", help="[Train/Finetune/Chat] Enable Automatic Mixed Precision (AMP) for training/learning.") # MODIFICATION: Added AMP flag
    ### MODIFICATION END ###


    # --- Inference Arguments ---
    infer_group = parser.add_argument_group('Inference (Chat)')
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--enable-quantized-learning", action="store_true", help="[Chat] Enable LTM updates for quantized models. Requires --shadow-model-path.")
    infer_group.add_argument("--ltm-lora-path", type=str, default=None, help="[Chat] Optional: Path to save/load LTM updates as a separate delta file.")
    infer_group.add_argument("--device", type=str, default="cpu", choices=["cpu", "vulkan"], help="[Chat] Device for quantized inference.")
    ### MODIFICATION START ###
    infer_group.add_argument("--h-halt-thresh", type=float, default=0.9, help="[HRM] Probability threshold for early exiting the H-module loop during inference.")
    ### MODIFICATION END ###
    infer_group.add_argument("--static-ltm-lr", action="store_true", help="[Chat] Disable the cosine annealing schedule for LTM updates and use a fixed LR instead.")
    infer_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="[Chat] The number of updates in one cosine annealing cycle for LTM learning.")
    infer_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="[Chat] The minimum learning rate for the LTM cosine annealing schedule.")

    # --- Other Arguments ---
    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument("--qtype", type=str, default="INT4", choices=["INT4", "Q4_0", "Q8_0", "Q2_K"], help="Quantization type/format.")
    other_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))

    args = parser.parse_args()

    # --- Argument Validation ---
    # <<< REVERTED CHANGE: Removed model_path requirement for train mode >>>
    if args.mode == 'train' and not args.train and not args.resume_from_ckpt:
         parser.error("`--train` is required for train mode unless resuming with `--resume-from-ckpt`.")
    if args.mode == 'finetune' and not args.train:
         parser.error("`--train` is required for finetune mode.")
    # <<< REVERTED CHANGE: Added back model_path requirement checks for other modes >>>
    if args.mode == 'finetune' and not args.model_path:
         parser.error("`--model-path` (base model) is required for finetune mode.")
    if args.mode == 'merge-lora' and not args.model_path:
         parser.error("`--model-path` (base model) is required for merge-lora mode.")
    if args.mode == 'merge-lora' and not args.lora_adapter_path:
         parser.error("`--lora-adapter-path` is required for merge-lora mode.")
    if args.mode == 'quantize' and not args.model_path:
        # Quantize might be called internally without model_path, so this check is relaxed
        # Check only if it's the primary mode being run
        # We handle the model/tokenizer loading inside the quantize function anyway
        pass
    if args.mode == 'chat' and not args.model_path:
         parser.error("`--model-path` is required for chat mode.")
    if args.enable_quantized_learning and not args.shadow_model_path:
         parser.error("--enable-quantized-learning requires --shadow-model-path to be set.")

    set_threads(args.threads)
    pt_device = pick_device()
    print(f"Using PyTorch device: {pt_device}")

    # --- MODIFICATION START: Add warning if AMP is requested but not available ---
    if args.amp and not _HAS_AMP:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: --amp was specified, but torch.cuda.amp is not available.      !!!")
        print("!!!          AMP will be DISABLED. This may be because CUDA is not          !!!")
        print("!!!          installed or your PyTorch build does not include it.           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        args.amp = False # Force disable
    elif args.amp and pt_device == torch.device('cpu'):
        print("Warning: AMP (--amp) is enabled but running on CPU. AMP will have no effect.")
    # --- MODIFICATION END ---

    # <<< REVERTED TOKENIZER LOADING LOGIC >>>
    tokenizer = None
    if args.mode == "train":
        # If resuming, tokenizer path might not be needed if it's saved with the model/checkpoint
        # But we prioritize the command line argument if provided
        tokenizer_load_path = args.tokenizer_path # Default to CLI arg
        # Check if resuming and if tokenizer exists near checkpoint (optional robustness)
        if args.resume_from_ckpt:
            # If resuming, always prioritize --tokenizer-path if given
            if args.tokenizer_path:
                print(f"INFO: Resuming and using specified tokenizer path: {args.tokenizer_path}")
                tokenizer_load_path = args.tokenizer_path
            else:
                # If resuming and no --tokenizer-path, try finding near checkpoint
                ckpt_dir = os.path.dirname(args.resume_from_ckpt)
                potential_tokenizer_path = os.path.join(ckpt_dir, 'tokenizer_config.json')
                if os.path.exists(potential_tokenizer_path):
                    print(f"INFO: Resuming and found tokenizer config near checkpoint. Loading from: {ckpt_dir}")
                    tokenizer_load_path = ckpt_dir
                else:
                    # If not found near checkpoint and not specified, raise error
                    parser.error("Resuming training requires --tokenizer-path, or tokenizer files next to the checkpoint.")
        elif not args.tokenizer_path:
            parser.error("--tokenizer-path is required when starting training from scratch.")
        else: # Starting new training, use specified path
            print(f"Loading tokenizer '{args.tokenizer_path}' for new model training...")
            tokenizer_load_path = args.tokenizer_path

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
            print(f"Successfully loaded tokenizer from '{tokenizer_load_path}'")
        except Exception as e:
            print(f"Error loading tokenizer from '{tokenizer_load_path}': {e}")
            sys.exit(1)

    elif args.model_path: # Handles finetune, merge, quantize (if model_path given), chat
        print(f"Loading tokenizer from model directory: {args.model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            print(f"Successfully loaded tokenizer from '{args.model_path}'")
        except Exception as e:
            # Added more specific error for tokenizer loading failure
            print(f"Error loading tokenizer from model directory '{args.model_path}'. Ensure tokenizer files (tokenizer.json, config.json, etc.) exist there.")
            print(f"Details: {e}")
            sys.exit(1)
    elif args.mode == 'quantize' and not args.model_path:
        # Allow quantize to proceed if called internally (model/tokenizer passed directly)
        print("INFO: Quantize mode called without --model-path. Assuming model and tokenizer are passed directly.")
        tokenizer = None # Will be passed in the function call
    else:
        # This condition should ideally not be reached due to earlier validation, but acts as a safeguard.
        parser.error(f"Cannot determine tokenizer path. --model-path is required for mode '{args.mode}' unless it's 'train'.")
    # <<< END REVERTED TOKENIZER LOADING LOGIC >>>


    # --- Tokenizer Pad Token Handling ---
    if tokenizer and tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer missing pad token, setting pad_token = eos_token ({tokenizer.eos_token})")
        else:
            print("Warning: Tokenizer missing both pad and eos tokens. Adding a '[PAD]' token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Only resize embeddings if *starting* training, not resuming
            if args.mode == "train" and not args.resume_from_ckpt:
                # We need to know the vocab size *before* creating the model
                # This ensures the Embedding layer has the right size initially
                args.vocab_size = len(tokenizer)
                print(f"Updated args.vocab_size to {args.vocab_size} due to added pad token.")


    # --- Set Vocab Size if Training New Model ---
    # Moved this logic into the train function where it's needed before model init
    # if args.mode == "train" and not args.resume_from_ckpt:
    #     if not hasattr(args, 'vocab_size'):
    #         if tokenizer is None:
    #             parser.error("Tokenizer could not be loaded, cannot determine vocab_size for new model.")
    #         args.vocab_size = len(tokenizer)


    # --- Auto Max Length Scanning ---
    if args.auto_max_length:
        train_file_path = args.train
        # Added check if resuming and train path not given
        if not train_file_path and args.resume_from_ckpt:
            print("INFO: --auto-max-length used with --resume-from-ckpt. Trying to load train path from config...")
            try:
                # Load only config to check for train path if saved previously
                ckpt = torch.load(args.resume_from_ckpt, map_location='cpu', weights_only=False) # Need config
                train_file_path = ckpt.get('config', {}).get('train_data_path') # Assumes you save it
            except Exception as e:
                print(f"Warning: Could not load train path from checkpoint config: {e}")

        if not train_file_path or not os.path.exists(train_file_path):
            parser.error("--auto-max-length requires a valid --train file path (either directly or potentially via resumed config).")

        print("INFO: --auto-max-length enabled. Scanning dataset to find the maximum sequence length...")
        max_found_length = 0

        with open(train_file_path, 'r', encoding='utf-8') as f:
            # Helper to get text consistently
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
                        text = get_text_from_obj(obj, args.kayla)
                        # Use encode without special tokens for length check, add 1 for EOS later
                        if tokenizer: # Check if tokenizer loaded
                            length = len(tokenizer.encode(text, add_special_tokens=False)) + 1
                            if length > max_found_length: max_found_length = length
                        else:
                            print("Warning: Cannot scan length, tokenizer not available.")
                            break # Stop scanning if no tokenizer
            except json.JSONDecodeError:
                f.seek(0)
                for line in tqdm(f, desc="Scanning JSONL"):
                    try:
                        obj = json.loads(line)
                        text = get_text_from_obj(obj, args.kayla)
                        if tokenizer:
                            length = len(tokenizer.encode(text, add_special_tokens=False)) + 1
                            if length > max_found_length: max_found_length = length
                        else:
                            print("Warning: Cannot scan length, tokenizer not available.")
                            break # Stop scanning if no tokenizer
                    except Exception: # Catch JSON errors or others
                        continue
            except Exception as e:
                print(f"Error during dataset scan for max_length: {e}")


        if max_found_length > 0:
            # Add a small buffer and round up to a multiple of 8 for efficiency
            target_max_length = (max_found_length + 16 + 7) & -8
            print(f"✅ Auto-scan complete. Found max length ~{max_found_length}. Setting max_length to {target_max_length}.")
            args.max_length = target_max_length
            # Update tokenizer only if necessary
            if tokenizer and hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < target_max_length:
                tokenizer.model_max_length = target_max_length
                print(f"Updated tokenizer.model_max_length to {target_max_length}")

        else:
            print("⚠️ WARNING: Auto-scan did not find any valid entries or failed. Using default max_length.")


    # --- Execute Selected Mode ---
    if args.mode == "train":
        if tokenizer is None: # Ensure tokenizer is loaded for train
             print("Error: Tokenizer failed to load, cannot start training.")
             sys.exit(1)
        train(args, pt_device, tokenizer)
    elif args.mode == "finetune":
        if tokenizer is None: # Ensure tokenizer is loaded
             print("Error: Tokenizer failed to load, cannot start finetuning.")
             sys.exit(1)
        finetune(args, pt_device, tokenizer)
    elif args.mode == "merge-lora":
        # Tokenizer might be optional here if only merging weights, but needed for saving later
        if tokenizer is None and args.model_path:
            print("Warning: Loading tokenizer from base model path for saving.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            except Exception as e:
                print(f"Error loading tokenizer for merge output: {e}")
                # Decide if this is fatal or just prevents tokenizer saving
        elif tokenizer is None:
            print("Warning: Tokenizer not loaded, cannot save tokenizer with merged model.")

        merge_lora(args, pt_device, tokenizer) # Pass potentially None tokenizer
    elif args.mode == "quantize":
        # Quantize handles internal loading if tokenizer is None here
        quantize(args, pt_device, tokenizer=tokenizer)
    elif args.mode == "chat":
        # Ensure tokenizer was loaded for chat mode
        if tokenizer is None:
             print("Error: Tokenizer failed to load, cannot start chat.")
             sys.exit(1)
        chat(args, pt_device, tokenizer)


if __name__ == "__main__":
    main()
