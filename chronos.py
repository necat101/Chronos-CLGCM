import os
import sys
import json
import argparse
import time
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm
import traceback # Added for better error reporting

# <<< MODIFIED: Set Tokenizers Parallelism Environment Variable >>>
# Set this early, before tokenizers might be implicitly loaded by other imports
# Setting to "true" forces parallelism despite potential fork issues (use with caution)
# Setting to "false" explicitly disables parallelism in worker processes (safer, suppresses warning)
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set to false for safety

import torch
import torch.nn as nn
import torch.nn.functional as F
# <<< MODIFIED: Import IterableDataset >>>
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.serialization import safe_globals # May not be needed if not using legacy save/load

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
_torch_amp_available = False # Track if torch.amp or torch.cuda.amp was found

# --- Try importing autocast and GradScaler first (unconditionally) ---
try:
    # Prefer torch.amp (newer PyTorch versions)
    from torch.amp import GradScaler, autocast
    print("INFO: torch.amp (autocast/GradScaler) found.")
    _torch_amp_available = True
except ImportError:
    try:
        # Fallback for older PyTorch versions (requires CUDA context, but import might work)
        from torch.cuda.amp import GradScaler, autocast
        print("INFO: torch.cuda.amp (autocast/GradScaler) found.")
        _torch_amp_available = True
    except ImportError:
        print("Warning: torch.amp and torch.cuda.amp not found.")
        _torch_amp_available = False
        # --- Define dummy autocast if import failed ---
        import contextlib
        @contextlib.contextmanager
        def autocast(device_type, enabled=True, dtype=None): # Match signature
            # print("Warning: Using dummy autocast context manager.") # Optional: uncomment for debugging
            yield
        # GradScaler is only needed if _HAS_AMP is True later, so no dummy needed now

# --- Now, determine optimizer and final AMP status based on CUDA ---
if torch.cuda.is_available():
    # Attempt to use bitsandbytes 8-bit AdamW if available
    try:
        import bitsandbytes as bnb
        ADAM_OPTIMIZER = bnb.optim.AdamW8bit
        _HAS_BNB = True
        print("INFO: CUDA detected and bitsandbytes found. Using bitsandbytes 8-bit AdamW.")
    except ImportError:
        print("Warning: bitsandbytes not found.")
        print("INFO: Falling back to standard torch.optim.AdamW optimizer.")
        ADAM_OPTIMIZER = torch.optim.AdamW
        _HAS_BNB = False

    # Check if the actual AMP components were successfully imported earlier
    if _torch_amp_available:
        _HAS_AMP = True
        print("INFO: AMP support is enabled (CUDA available).")
    else:
        _HAS_AMP = False
        print("Warning: AMP support is disabled (torch.amp/torch.cuda.amp import failed).")

else: # No CUDA detected
    print("Warning: CUDA not detected. Using CPU training.")
    print("INFO: Falling back to standard torch.optim.AdamW optimizer (bitsandbytes requires CUDA).")
    ADAM_OPTIMIZER = torch.optim.AdamW
    _HAS_BNB = False
    _HAS_AMP = False # AMP not usable on CPU
    if _torch_amp_available:
        # This case means torch.amp was importable but we are on CPU
        print("INFO: AMP components found but disabled (running on CPU).")
    # If _torch_amp_available is False, the warning about dummy autocast will show if used


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
    print("!!!  - On Linux/macOS: Run bash setup.sh                                      !!!")
    print("!!!                                                                         !!!")
    print("!!! If you have already run the setup, you may need to activate the         !!!")
    print("!!! virtual environment first:                                              !!!")
    print("!!!  - On Windows:   .\\.venv\\Scripts\\Activate                              !!!")
    print("!!!  - On Linux/macOS: source .venv/bin/activate                              !!!")
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

# <<< START: New Iterable Dataset for Pre-Chunked JSONL Data >>>
class IterableChunkedJSONLDataset(IterableDataset):
    """
    An IterableDataset for loading pre-tokenized, chunked, masked, and padded
    data from a JSONL file line by line. Reduces RAM usage compared to loading
    the entire dataset into memory.

    Expects each line to be a JSON object containing 'input_ids', 'labels',
    and 'attention_mask' as lists of integers, all of the *same* pre-defined length.
    """
    def __init__(self, path: str, max_length: int):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path
        self.max_length = max_length # Used for validation

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        print(f"[Worker {worker_id}/{num_workers}] Opening dataset file: {self.path}")
        skipped_count = 0
        processed_count = 0

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    # Distribute lines across workers
                    if line_num % num_workers != worker_id:
                        continue

                    line = line.strip()
                    if not line: continue

                    try:
                        obj = json.loads(line)
                        # --- Basic Validation ---
                        if not all(k in obj for k in ["input_ids", "labels", "attention_mask"]):
                            # Print less frequently in multi-worker scenarios to avoid spam
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Missing required keys.")
                            skipped_count += 1
                            continue
                        if not all(isinstance(obj[k], list) for k in ["input_ids", "labels", "attention_mask"]):
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Required keys are not lists.")
                            skipped_count += 1
                            continue

                        # --- Length Validation ---
                        seq_len = len(obj["input_ids"])
                        if seq_len != self.max_length:
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Expected length {self.max_length}, found {seq_len}.")
                            skipped_count += 1
                            continue
                        elif len(obj["labels"]) != seq_len or len(obj["attention_mask"]) != seq_len:
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Length mismatch between input_ids, labels, and attention_mask.")
                            skipped_count += 1
                            continue

                        # Convert lists to tensors just before yielding
                        processed_count += 1
                        yield {
                            "input_ids": torch.tensor(obj["input_ids"], dtype=torch.long),
                            "labels": torch.tensor(obj["labels"], dtype=torch.long),
                            "attention_mask": torch.tensor(obj["attention_mask"], dtype=torch.long)
                        }

                    except json.JSONDecodeError:
                        if skipped_count % (100 * num_workers) == 0:
                            print(f"[Worker {worker_id}] Warning: Skipping invalid JSON on line ~{line_num+1}: {line[:100]}...")
                        skipped_count += 1
                        continue
                    except Exception as e:
                        if skipped_count % (100 * num_workers) == 0:
                            print(f"[Worker {worker_id}] Warning: Error processing line ~{line_num+1}: {e}. Line: {line[:100]}...")
                        skipped_count += 1
                        continue
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR during dataset iteration: {e}")
            traceback.print_exc() # Print full traceback if file reading fails etc.
            raise e # Re-raise the exception

        print(f"[Worker {worker_id}] Finished iterating. Processed: {processed_count}, Skipped: {skipped_count}")


def create_dataloader_for_chunked(path, max_length, batch_size, num_workers=0):
    """
    Creates a DataLoader specifically for the pre-chunked JSONL dataset using
    an IterableDataset to save RAM. Padding is assumed handled by the chunking script.
    """
    # Use the IterableDataset
    dataset = IterableChunkedJSONLDataset(path, max_length=max_length)

    def collate_fn_simple(batch):
        # Batch items are dictionaries with tensors of the same length
        if not batch: return None

        # Check if items are already dictionaries (expected from IterableDataset)
        if not isinstance(batch[0], dict):
            print(f"Warning: Unexpected item type in collate_fn_simple: {type(batch[0])}. Expected dict.")
            # Attempt to handle if it's a list/tuple, otherwise raise error
            if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 3: # Assuming order: ids, labels, mask
                input_ids_batch = torch.stack([item[0] for item in batch])
                labels_batch = torch.stack([item[1] for item in batch])
                attention_mask_batch = torch.stack([item[2] for item in batch])
            else:
                raise TypeError(f"Collate function received unexpected data structure: {type(batch[0])}")
        else:
            input_ids_batch = torch.stack([item['input_ids'] for item in batch])
            labels_batch = torch.stack([item['labels'] for item in batch])
            attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    pin_memory = torch.cuda.is_available() and num_workers > 0
    # persistent_workers is generally recommended with num_workers > 0
    # It avoids worker startup overhead for each epoch.
    persistent_workers = num_workers > 0

    # <<< MODIFIED: Removed shuffle=True, as it's not applicable here >>>
    # IterableDatasets handle shuffling differently (often requiring buffering)
    # or rely on the inherent order/distribution for large datasets.
    # Simple line-by-line iteration per worker is used here.
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_simple,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers)
# <<< END: Iterable Dataset/Loader for Pre-Chunked JSONL Data >>>

# <<< START: Modified Map-Style Dataset/Loader for Consolidated PT Tensors >>>
class PTChunkedDataset(Dataset):
    """
    A map-style Dataset for loading pre-tokenized, chunked, masked, and padded
    data directly from individual chunk entries listed in a manifest.jsonl,
    where multiple chunks are consolidated into single .pt files.
    Reduces RAM usage during startup compared to loading all text.

    Expects a directory containing consolidated .pt files (each a list of dicts)
    and a manifest.jsonl. Each line in manifest.jsonl should contain
    'file_path' (relative) and 'index_in_file' (int).
    Each chunk dict within the .pt files should have 'input_ids', 'labels',
    and 'attention_mask' as torch.tensors of the *same* pre-defined length.
    Implements caching to reduce redundant file reads.
    """
    def __init__(self, directory_path: str, max_length: int):
        super().__init__()
        self.directory_path = directory_path
        self.max_length = max_length # For potential validation
        # <<< MODIFIED: Store (filepath, index) tuples >>>
        self.chunk_pointers = []
        # <<< MODIFIED: Add caching attributes >>>
        self.last_loaded_path = None
        self.last_loaded_data = None # This will hold the list loaded from a .pt file

        manifest_file = os.path.join(directory_path, "manifest.jsonl")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        print(f"Loading chunk pointers from manifest: {manifest_file}")
        try:
            with open(manifest_file, "r", encoding="utf-8") as f_manifest:
                for line_num, line in enumerate(f_manifest): # Added line_num for better warnings
                    line = line.strip()
                    if not line: continue
                    try:
                        entry = json.loads(line)
                        relative_path = entry.get("file_path")
                        # <<< MODIFIED: Get index_in_file >>>
                        index_in_file = entry.get("index_in_file")

                        # <<< MODIFIED: Validate entry >>>
                        if relative_path and isinstance(relative_path, str) and \
                           index_in_file is not None and isinstance(index_in_file, int):
                            full_path = os.path.join(self.directory_path, relative_path)
                            self.chunk_pointers.append((full_path, index_in_file))
                        else:
                            print(f"Warning: Manifest line ~{line_num+1} missing or invalid 'file_path' (str) or 'index_in_file' (int): {line}")
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in manifest line ~{line_num+1}: {line[:100]}...")
            if not self.chunk_pointers:
                raise ValueError(f"No valid chunk pointers found in manifest: {manifest_file}")
            print(f"Found {len(self.chunk_pointers)} total logical chunks.")
        except Exception as e:
            print(f"Error reading manifest file {manifest_file}: {e}")
            raise e

    def __len__(self):
        # <<< MODIFIED: Length is the total number of pointers >>>
        return len(self.chunk_pointers)

    def __getitem__(self, idx):
        # <<< MODIFIED: Retrieve path and index >>>
        try:
            chunk_path, index_in_file = self.chunk_pointers[idx]
        except IndexError:
            # This shouldn't happen with standard DataLoader usage but is a safeguard
            print(f"Error: Index {idx} out of bounds for chunk pointers (len: {len(self.chunk_pointers)}).")
            return None # Indicate failure

        try:
            # <<< MODIFIED: Implement Caching >>>
            if chunk_path == self.last_loaded_path:
                # Cache Hit: Use the already loaded list
                if self.last_loaded_data is None:
                    # Should not happen if last_loaded_path is set, but handle defensively
                    print(f"Warning: Cache inconsistency for file {chunk_path}. Reloading.")
                    self.last_loaded_data = torch.load(chunk_path, map_location='cpu')
                    if not isinstance(self.last_loaded_data, list):
                         raise TypeError(f"Loaded data from {chunk_path} is not a list.")
                # Retrieve the specific chunk dictionary from the cached list
                data = self.last_loaded_data[index_in_file]

            else:
                # Cache Miss: Load the new consolidated file
                # print(f"Cache miss. Loading file: {chunk_path}") # Optional: for debugging
                loaded_list = torch.load(chunk_path, map_location='cpu') # Load to CPU initially
                if not isinstance(loaded_list, list):
                    raise TypeError(f"Loaded data from {chunk_path} is not a list.")

                # Update cache
                self.last_loaded_path = chunk_path
                self.last_loaded_data = loaded_list

                # Retrieve the specific chunk dictionary
                data = self.last_loaded_data[index_in_file]

            # --- Validation (Optional but recommended) ---
            if not isinstance(data, dict):
                 raise TypeError(f"Chunk at index {index_in_file} in {chunk_path} is not a dictionary.")
            if not all(k in data for k in ["input_ids", "labels", "attention_mask"]):
                print(f"Warning: Chunk dict at index {index_in_file} in {chunk_path} missing required keys. Skipping.")
                return None # Indicate failure
            if not all(isinstance(data[k], torch.Tensor) for k in ["input_ids", "labels", "attention_mask"]):
                 print(f"Warning: Data in chunk dict at index {index_in_file} in {chunk_path} are not tensors. Skipping.")
                 return None
            if data["input_ids"].shape[0] != self.max_length:
                 print(f"Warning: Chunk tensor 'input_ids' at index {index_in_file} in {chunk_path} has unexpected length {data['input_ids'].shape[0]}. Expected {self.max_length}. Skipping.")
                 return None
            # --- End Validation ---

            return data

        except FileNotFoundError:
             print(f"Error: Consolidated chunk file not found: {chunk_path}")
             self.last_loaded_path = None # Invalidate cache if file not found
             self.last_loaded_data = None
             return None
        except IndexError:
             print(f"Error: index_in_file {index_in_file} out of bounds for loaded list from {chunk_path} (len: {len(self.last_loaded_data) if self.last_loaded_data else 'N/A'}). Check manifest/chunking script.")
             # Consider invalidating cache here too if the file structure seems wrong
             # self.last_loaded_path = None
             # self.last_loaded_data = None
             return None
        except TypeError as e:
             print(f"Error: Type error processing chunk at index {index_in_file} in {chunk_path}: {e}")
             return None
        except Exception as e:
             # Catch other potential errors during loading or processing
             print(f"Error loading or processing chunk from {chunk_path} at index {index_in_file}: {e}")
             # Optionally invalidate cache on unexpected errors
             # self.last_loaded_path = None
             # self.last_loaded_data = None
             return None


def create_dataloader_pt_chunked(directory_path, max_length, batch_size, num_workers=0):
    """
    Creates a DataLoader for the pre-chunked consolidated .pt dataset using PTChunkedDataset.
    Handles shuffling and batching of pre-loaded tensors. Caching is handled within the Dataset.
    """
    dataset = PTChunkedDataset(directory_path, max_length=max_length) # Uses the MODIFIED dataset

    def collate_fn_pt(batch):
        # Filter out None items potentially returned by dataset __getitem__ on error
        batch = [item for item in batch if item is not None]
        if not batch: return None # Return None if batch becomes empty after filtering

        # Items are dictionaries with tensors of the same length
        try:
            input_ids_batch = torch.stack([item['input_ids'] for item in batch])
            labels_batch = torch.stack([item['labels'] for item in batch])
            attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])
        except Exception as e:
            print(f"Error during collate_fn_pt: {e}. One of the items might be malformed.")
            # Decide how to handle this - skip batch or raise error?
            # Returning None might be safer if errors are expected, but hides issues.
            # Raising the error stops training but makes the problem explicit.
            # Let's return None for now to avoid crashing training on rare errors.
            return None


        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    pin_memory = torch.cuda.is_available() and num_workers > 0
    persistent_workers = num_workers > 0

    # Map-style dataset allows shuffling
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_pt, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers)

# <<< END: Modified Map-Style Dataset/Loader for Consolidated PT Tensors >>>

# <<< START: Original Dataset/Loader (Renamed) - NO CHANGES NEEDED HERE >>>
class OriginalJSONLDataset(Dataset):
    """
    Handles both .jsonl (one JSON object per line) and .json (a list of objects) files.
    Also supports standard and "Kayla" instruction formats.
    Tokenizes data on the fly. Loads everything into RAM.
    """
    def __init__(self, path: str, tokenizer, max_length: int, kayla_mode: bool = False):
        super().__init__() # Add this if not present in your original code
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
                output_text = f"### Response:\n{obj.get('output', '')}" # No trailing newlines for response

                # Tokenize the different parts
                prompt_context_tokens = self.tokenizer.encode(prompt_context_text, add_special_tokens=True) # Add special tokens ONLY here
                thought_tokens = self.tokenizer.encode(thought_text, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)

                # Combine into final input_ids and labels
                input_ids = prompt_context_tokens + thought_tokens + output_tokens + [self.tokenizer.eos_token_id]
                # Labels: Mask prompt context, keep thought and output
                labels = ([-100] * len(prompt_context_tokens)) + thought_tokens + output_tokens + [self.tokenizer.eos_token_id]
            else: # Standard format
                prompt = f"### Instruction:\n{obj.get('instruction', '')}\n\n### Response:\n"
                completion = obj.get('output', '') or obj.get('response', '')

                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True) # Add special tokens ONLY here
                completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

                input_ids = prompt_tokens + completion_tokens + [self.tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_tokens)) + completion_tokens + [self.tokenizer.eos_token_id]

            # Truncate if necessary AFTER combining all parts
            if len(input_ids) > self.max_length:
                # Truncate from the right, ensuring EOS is kept if possible
                input_ids = input_ids[:self.max_length-1] + [self.tokenizer.eos_token_id]
                labels = labels[:self.max_length-1] + [self.tokenizer.eos_token_id]
            # Padding is handled by the collate function

            return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Added more robust error logging
            obj_repr = str(obj)
            print(f"Warning: Skipping invalid data entry: {obj_repr[:150] + ('...' if len(obj_repr) > 150 else '')}. Error: {e}")
            return None


    def _load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        print(f"Loading and tokenizing dataset from {path}...")
        if self.kayla_mode:
            print("INFO: Kayla-style instruction tuning is ENABLED.")

        skipped_count = 0
        line_num = 0
        with open(path, "r", encoding="utf-8") as f:
            # Try loading as a single JSON list first
            try:
                f.seek(0) # Ensure reading from start
                data = json.load(f)
                if isinstance(data, list):
                    print("Detected JSON file (list of objects). Processing...")
                    for obj in tqdm(data, desc="Tokenizing samples"):
                        processed = self._process_object(obj)
                        if processed:
                            self.samples.append(processed)
                        else:
                            skipped_count += 1
                    if skipped_count > 0: print(f"Skipped {skipped_count} invalid samples during JSON loading.")
                    return # Successfully loaded JSON list
                else:
                    print("Warning: JSON file does not contain a list. Attempting JSONL parsing.")
            except json.JSONDecodeError:
                # This is expected for a JSONL file, so pass through
                pass
            except Exception as e: # Catch other potential errors during JSON load
                print(f"Warning: Error loading as JSON list: {e}. Attempting JSONL parsing.")

            # If not a JSON list or loading failed, try JSONL
            print("Attempting JSONL file (one object per line). Processing...")
            f.seek(0) # Reset file pointer
            skipped_count = 0 # Reset skipped count for JSONL
            for line in tqdm(f, desc="Tokenizing samples (JSONL)"):
                line_num += 1
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    processed = self._process_object(obj)
                    if processed:
                        self.samples.append(processed)
                    else:
                        skipped_count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {line[:100]}...")
                    skipped_count += 1
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}. Line: {line[:100]}...")
                    skipped_count += 1
                    continue

        if skipped_count > 0: print(f"Skipped {skipped_count} invalid samples during JSONL loading.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_dataloader_original(path, tokenizer, max_length, batch_size, pad_token_id, kayla_mode=False, num_workers=0):
    """Creates a DataLoader for training or fine-tuning from original format, handling tokenization and padding."""
    dataset = OriginalJSONLDataset(path, tokenizer, max_length, kayla_mode=kayla_mode) # Use original dataset class
    if len(dataset) == 0:
        raise ValueError(f"Dataset loaded from {path} is empty or invalid after processing.")

    def collate_fn_original(batch):
        # Filter out None items potentially returned by dataset __getitem__ if _process_object failed
        batch = [item for item in batch if item is not None]
        if not batch: return None # Return None if batch becomes empty

        # Find max length *in this batch* for dynamic padding
        max_len_batch = max(len(item['input_ids']) for item in batch)
        # Ensure max_len_batch doesn't exceed the model's max_length capability
        # Note: dataset already truncated items longer than max_length
        # max_len_batch = min(max_len_batch, max_length) # Should not be necessary if dataset truncates

        input_ids_batch = torch.full((len(batch), max_len_batch), pad_token_id, dtype=torch.long)
        labels_batch = torch.full((len(batch), max_len_batch), -100, dtype=torch.long) # Use -100 for padding labels
        attention_mask_batch = torch.zeros((len(batch), max_len_batch), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            # Copy sequence data
            input_ids_batch[i, :seq_len] = item['input_ids']
            labels_batch[i, :seq_len] = item['labels']
            attention_mask_batch[i, :seq_len] = 1 # Set attention mask to 1 for non-pad tokens

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    pin_memory = torch.cuda.is_available() and num_workers > 0
    persistent_workers = num_workers > 0
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_original, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
# <<< END: Original Dataset/Loader (Renamed) >>>


# --- Quantization & Model Serialization ---
# ... (Quantization code remains unchanged) ...
def get_q_block_size(qtype: str) -> int:
    """Returns the block size for a given quantization type."""
    if qtype in ["INT4", "Q4_0", "Q8_0"]:
        return 32
    elif qtype == "Q2_K":
        return 256
    else:
        raise ValueError(f"Unknown quantization type: {qtype}")

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

    # Save the model config directly into the npz file
    config_to_save = dict(model.config)
    quantized_tensors['_config'] = np.array(config_to_save, dtype=object) # Use dtype=object for dict

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

    # Also save the tokenizer to make the directory self-contained
    try:
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer files saved to {output_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer files to {output_dir}. Error: {e}")

class QuantizedLinear:
    """A wrapper for a quantized linear layer that uses the C++ kernel for inference."""
    def __init__(self, name: str, q_data: dict):
        self.name = name
        weight_data_key = f'{name}.weight'
        bias_data_key = f'{name}.bias'

        if weight_data_key not in q_data:
            raise KeyError(f"Weight data '{weight_data_key}' not found in quantized file.")

        weight_meta = q_data[weight_data_key].item() # .item() needed for numpy object arrays
        if 'quantized' not in weight_meta:
            raise ValueError(f"Weight '{weight_data_key}' is not quantized (missing 'quantized' key).")

        self.quantized_w = weight_meta['quantized']
        self.qtype = str(weight_meta['qtype'])
        self.original_shape = weight_meta['original_shape']
        self.M, self.K = self.original_shape

        if bias_data_key in q_data:
            bias_meta = q_data[bias_data_key].item() # .item() needed
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

        # Ensure input K matches the quantized weight K (which is the original K + padding)
        # The C++ kernel expects the input K to match the *padded* K of the weight.
        padded_k = self.K
        if self.K % get_q_block_size(self.qtype) != 0:
            padded_k += get_q_block_size(self.qtype) - (self.K % get_q_block_size(self.qtype))

        if x_np.shape[-1] != padded_k:
            # Pad input if needed to match the kernel's expectation
            pad_k = padded_k - x_np.shape[-1]
            if pad_k > 0:
                x_np = np.pad(x_np, ((0, 0), (0, pad_k)), 'constant')
            elif pad_k < 0: # Input is larger than expected padded K? Should not happen.
                print(f"Warning: Input dimension ({x_np.shape[-1]}) > Expected padded K ({padded_k}) for layer {self.name}. Truncating input.")
                x_np = x_np[..., :padded_k]


        y_np = chronos_matmul.matmul_quantized(x_np, self.quantized_w, self.M, self.qtype, device)

        # Output shape should match original input dimensions + output features M
        if original_ndim > 2:
            output_shape = list(original_shape[:-1]) + [self.M]
            y_np = y_np.reshape(output_shape)
        elif original_ndim == 1:
            y_np = y_np.reshape(-1) # Reshape back to 1D


        if y_np.shape[-1] != self.M:
            # This should ideally not happen if matmul_quantized handles padding correctly,
            # but keep as a safeguard. Truncate output to expected feature dim M.
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
    # SOURCE_ID definitions
    SRC_UNKNOWN = 0
    SRC_USER_INTERACTION = 1
    SRC_TRAINING_DATA = 2

    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, val_dim) * 0.02)
        self.register_buffer("_mom_vals", torch.zeros_like(self.vals.data))
        self.lr, self.momentum, self.weight_decay = lr, momentum, wd

        # Buffers are not parameters; they are part of the model's state
        # but are not updated by the optimizer during training.
        self.register_buffer("timestamps", torch.zeros(n_slots, dtype=torch.float32))
        self.register_buffer("sources", torch.full((n_slots,), self.SRC_UNKNOWN, dtype=torch.long))

        # Buffer for accumulating deltas if not updating in-place
        self.register_buffer("ltm_deltas", torch.zeros_like(self.vals.data))
        self.accumulate_deltas = False


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
        num_valid_slots_per_query = sim.isfinite().sum(dim=-1) # Shape: [Batch] or [Batch, SeqLen] etc.
        num_valid_slots = num_valid_slots_per_query.min().item() # Minimum valid slots across all queries
        effective_topk = min(topk, int(num_valid_slots))

        if effective_topk <= 0:
            # Handle case where no slots match the filter for at least one query
            # print("Warning: No LTM slots matched the current filter criteria for at least one query.")
            # Return tensors filled with zeros/dummy values matching expected shape
            query_shape = list(queries.shape)
            # Shape: [..., TopK, ValDim] (handles arbitrary leading dims)
            vals_shape = query_shape[:-1] + [topk, self.vals.shape[-1]]
            # Shape: [..., TopK]
            idx_shape = query_shape[:-1] + [topk]
            return torch.zeros(vals_shape, device=queries.device, dtype=self.vals.dtype), \
                   torch.full(idx_shape, -1, device=queries.device, dtype=torch.long) # Use -1 for invalid index


        _, idx = torch.topk(sim, k=effective_topk, dim=-1) # Shape: [..., effective_topk]

        # Pad results if effective_topk < topk
        if effective_topk < topk:
            pad_size = topk - effective_topk
            # Pad indices with -1
            idx_pad_shape = list(idx.shape[:-1]) + [pad_size]
            idx_pad = torch.full(idx_pad_shape, -1, device=idx.device, dtype=idx.dtype)
            idx = torch.cat([idx, idx_pad], dim=-1) # Shape: [..., topk]

            # Pad retrieved values with zeros
            # Need to handle potential -1 indices introduced by filtering before indexing self.vals
            valid_idx_mask = idx[..., :effective_topk] >= 0
            vals_retrieved = torch.zeros(list(idx.shape[:-1]) + [effective_topk, self.vals.shape[-1]], device=self.vals.device, dtype=self.vals.dtype)
            # Only index self.vals where the mask is true
            if valid_idx_mask.any():
                actual_indices = idx[..., :effective_topk][valid_idx_mask]
                vals_retrieved[valid_idx_mask] = self.vals[actual_indices]

            vals_pad_shape = list(vals_retrieved.shape[:-2]) + [pad_size, vals_retrieved.shape[-1]]
            vals_pad = torch.zeros(vals_pad_shape, device=vals_retrieved.device, dtype=vals_retrieved.dtype)
            vals_ret = torch.cat([vals_retrieved, vals_pad], dim=-2) # Concatenate along the topk dimension
            return vals_ret, idx # Shape: [..., topk, ValDim], [..., topk]
        else:
            # If effective_topk == topk, we might still have filtered out slots, resulting in -inf
            # Clamp indices just in case topk returns out-of-bounds due to all -inf
            valid_idx_mask = idx >= 0
            ret_vals = torch.zeros(list(idx.shape) + [self.vals.shape[-1]], device=self.vals.device, dtype=self.vals.dtype)
            if valid_idx_mask.any():
                actual_indices = idx[valid_idx_mask].clamp(min=0, max=self.vals.shape[0]-1) # Clamp only valid ones
                ret_vals[valid_idx_mask] = self.vals[actual_indices]
            return ret_vals, idx


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
            if grads_tensor.shape[:-1] != topk_idx.shape: # Check batch and topk dims only
                print(f"Warning: grads_tensor shape {grads_tensor.shape[:-1]} mismatch with topk_idx shape {topk_idx.shape}. Skipping LTM update.")
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
            # Only apply update where counts > 0 (i.e., where gradients were actually accumulated)
            final_update[nonzero_mask] = update_delta[nonzero_mask]

            # --- UPDATE METADATA ---
            current_time = time.time()
            # Only update metadata for slots that actually received an update
            self.timestamps.data[nonzero_mask] = current_time
            self.sources.data[nonzero_mask] = source
            # --- END METADATA UPDATE ---

            if self.accumulate_deltas:
                # Add the final computed update to both the deltas buffer and the actual values
                self.ltm_deltas.data.add_(final_update)
                self.vals.data.add_(final_update) # Also apply immediately if accumulating
            else:
                self.vals.data.add_(final_update)


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
            # Allow max_length to be None initially, it might be set later
            if key not in self.config and key != 'max_length':
                raise ValueError(f"Missing required configuration key: '{key}'")
            # Handle case where max_length might be None in config, use a default if needed for pos_emb
            if key == 'max_length' and not self.config.get('max_length'):
                print("Warning: max_length not found in config during model init. Using default 1024 for pos_emb.")
                self.config['max_length'] = 1024 # Set a default if missing


        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.context_dim)
        # Use the potentially defaulted max_length for pos_emb
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

        # Add the halting projection layer
        self.h_halt_proj = nn.Linear(self.config.h_hidden, 1)

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

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, labels: Optional[torch.LongTensor] = None, min_timestamp: float = 0.0, source_filter: Optional[int] = None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device

        tok_embs = self.tok_emb(input_ids)
        # Handle potential sequence length exceeding max_length for position embeddings
        pos_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_indices = pos_indices % self.config.max_length # Use modulo for positions beyond max_length
        pos_embs = self.pos_emb(pos_indices)

        x = tok_embs + pos_embs

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

            # Pass filters to retrieve_topk
            topk_vals, topk_idx = self.ltm.retrieve_topk(
                query,
                topk=self.config.ltm_topk,
                min_timestamp=min_timestamp,
                source_filter=source_filter
            )

            # NOTE: This UserWarning is expected and necessary.
            # We must retain the grad on a non-leaf tensor (the retrieved values)
            # to calculate the "surprise" gradient for the LTM update.
            if self.training or torch.is_grad_enabled():
                # Only retain grad if the retrieved values are not just zeros (e.g., from failed filter)
                # and if the tensor actually requires grad (might be detached if loaded)
                if topk_vals.requires_grad:
                    # Check if tensor has grad_fn before calling retain_grad
                    if topk_vals.grad_fn is not None:
                        topk_vals.retain_grad()
                    # else: # Optional: print warning if it doesn't have grad_fn
                    #     print(f"Warning: LTM topk_vals at step {t} does not have grad_fn, cannot retain grad.")


            all_topk_vals.append(topk_vals)
            all_topk_idx.append(topk_idx)

            ltm_summary_flat = topk_vals.view(B, -1)

            mac_input = torch.cat([token_emb, p_read, ltm_summary_flat], dim=-1)
            enc = F.gelu(self.in_proj(mac_input))

            # Adaptive HRM Loop
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
                if not self.training and (halt_prob.mean() > halt_thresh): # Check mean halt prob across batch
                    break

            # After the loop, calculate the final output and ponder cost using ACT logic
            # Ensure step_outputs is not empty before stacking
            if not step_outputs:
                # This might happen if max_h_steps is 0 or if something went wrong
                print(f"Warning: No HRM steps executed for token {t}. Using initial encoding.")
                final_enc = enc
                ponder_cost = torch.tensor(0.0, device=device) # No ponder cost if no steps
            else:
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
                weights_normalized = weights / total_prob_sum.unsqueeze(0)
                remainder_normalized = remainder / total_prob_sum


                # Weighted average of step outputs + remainder contribution from last step
                final_enc = (weights_normalized.unsqueeze(-1) * step_outputs_t).sum(dim=0) + \
                            remainder_normalized.unsqueeze(-1) * step_outputs_t[-1]


                # Ponder cost: number of steps executed + probability of not halting (remainder)
                ponder_cost = num_steps_taken + remainder # Shape: [B]

            all_ponder_costs.append(ponder_cost)
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

            # Average the ponder cost across the sequence length and batch
            if all_ponder_costs:
                # Stack costs: Shape [T, B] -> Mean over T first, then mean over B
                # Ensure ponder_cost items are tensors before stacking
                valid_ponder_costs = [pc if isinstance(pc, torch.Tensor) else torch.tensor(pc, device=device) for pc in all_ponder_costs]
                if valid_ponder_costs: # Check if list is not empty
                    stacked_costs = torch.stack(valid_ponder_costs, dim=1) # Stack along sequence dim -> [B, T]
                    ponder_cost_out = stacked_costs.mean() # Mean over all elements
                else:
                    ponder_cost_out = torch.tensor(0.0, device=device)

            else: # Handle edge case T=0 or no valid HRM steps
                ponder_cost_out = torch.tensor(0.0, device=device)

        # Ensure returned tensors are correctly shaped even if loop didn't run
        seq_topk_vals = torch.stack(all_topk_vals, dim=1) if all_topk_vals else torch.empty(B, 0, self.config.ltm_topk, self.config.ltm_val_dim, device=device)
        seq_topk_idx = torch.stack(all_topk_idx, dim=1) if all_topk_idx else torch.empty(B, 0, self.config.ltm_topk, dtype=torch.long, device=device)


        return {"loss": loss, "logits": logits, "topk_vals": seq_topk_vals, "topk_idx": seq_topk_idx, "ponder_cost": ponder_cost_out}


class QuantizedChronos:
    # ... (QuantizedChronos remains unchanged) ...
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
            ltm_state = {}
            expected_ltm_keys = ['ltm.keys', 'ltm.vals', 'ltm.timestamps', 'ltm.sources']
            missing_ltm_keys = []
            for k in expected_ltm_keys:
                if k in q_data and 'raw' in q_data[k].item():
                    ltm_state[k.split('.', 1)[1]] = torch.from_numpy(q_data[k].item()['raw']) # Remove 'ltm.' prefix
                else:
                    missing_ltm_keys.append(k)

            if missing_ltm_keys:
                raise KeyError(f"Missing expected LTM parameters in quantized file: {missing_ltm_keys}")
            self.ltm.load_state_dict(ltm_state, strict=False) # strict=False to ignore _mom_vals etc.

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
                    # Check if weight exists for this linear layer before creating QuantizedLinear
                    weight_key = f"{layer_base_name}.weight"
                    if weight_key not in q_data:
                        raise KeyError(f"Weight key '{weight_key}' not found for layer '{layer_base_name}'")

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


    def __call__(self, input_ids: torch.LongTensor, h_state: torch.Tensor, l_state: torch.Tensor, device: str = "cpu", min_timestamp: float = 0.0, source_filter: Optional[int] = None):
        B, T = input_ids.shape
        # Use T-1 for single-token generation, but allow for longer sequences during initial prompt processing.
        current_pos_start = T - 1 if T > 1 else 0

        # Process the entire input_ids sequence token-by-token (or just the last token if T > 1)
        for t in range(current_pos_start, T):
            # Ensure inputs to embeddings are LongTensors on CPU
            current_token_ids = input_ids[:, t].cpu().long()
            # Calculate position, handle wrapping
            pos_id_val = t % self.config.max_length
            pos_ids = torch.tensor([pos_id_val], dtype=torch.long).cpu()

            token_emb = self.tok_emb(current_token_ids) + self.pos_emb(pos_ids)

            p_read = self.persistent.unsqueeze(0).expand(B, -1)
            query = self.qproj(token_emb, device=device) # Query projection happens on target device

            # Pass filters to retrieve_topk
            topk_vals, _ = self.ltm.retrieve_topk(
                query,
                topk=self.config.ltm_topk,
                min_timestamp=min_timestamp,
                source_filter=source_filter
            )

            # Ensure topk_vals is on CPU for concatenation if needed, though subsequent ops use target device
            ltm_summary_flat = topk_vals.view(B, -1).cpu()


            mac_input = torch.cat([token_emb.cpu(), p_read.cpu(), ltm_summary_flat], dim=-1)
            enc = F.gelu(self.in_proj(mac_input, device=device)) # Input projection on target device

            # Ensure RNN states are on CPU before passing to QuantizedGRUCell
            h_state = h_state.cpu()
            l_state = l_state.cpu()

            # Adaptive HRM Loop for Inference
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

        # Final operations
        final_embedding = self.out_norm(enc.cpu()).to(enc.dtype) # Norm on CPU, ensure correct dtype
        logits = self.lm_head(final_embedding, device=device) # LM head on target device

        # Return states on CPU as expected by the chat loop for quantized models
        return {"logits": logits.unsqueeze(1), "h_state": h_state.cpu(), "l_state": l_state.cpu()}


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

    config_dict = q_data['_config'].item() # Load config dict from npz
    # Ensure config gets 'model_type' if missing, useful for HF compatibility downstream
    if 'model_type' not in config_dict:
        config_dict['model_type'] = 'chronos'

    config = AttrDict(config_dict) # Convert to AttrDict

    return QuantizedChronos(config, q_data), config # Return both object and AttrDict config


def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model and its config from a directory."""
    weights_path = os.path.join(model_path, MODEL_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{MODEL_WEIGHTS_NAME}' not found in '{model_path}'")

    # Load checkpoint safely, allowing pickles only if necessary (e.g., for optimizer state)
    try:
        # Try weights_only=True first for security if config allows
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        # Verify config is present
        if 'config' not in checkpoint:
            print("INFO: Config not found in weights_only load. Retrying with weights_only=False.")
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False) # Allow pickles if needed
    except RuntimeError as e: # Catch errors potentially related to weights_only=True
        print(f"Warning: Failed to load checkpoint with weights_only=True ({e}). Retrying with weights_only=False (allowing pickles).")
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        except Exception as inner_e:
            raise RuntimeError(f"Failed to load checkpoint even allowing pickles: {inner_e}")
    except Exception as e: # Catch other loading errors
        raise RuntimeError(f"Failed to load checkpoint: {e}")


    # Config must be present now
    if 'config' not in checkpoint:
        raise ValueError("Model config not found in checkpoint. The model file is likely corrupted or from an old version.")

    config_dict = checkpoint['config'] # Config is likely a dict
    # Ensure model_type is present for HuggingFace compatibility
    if 'model_type' not in config_dict:
        config_dict['model_type'] = 'chronos'

    # Ensure vocab_size is present before creating model
    if 'vocab_size' not in config_dict:
        raise ValueError("Cannot initialize model: 'vocab_size' missing from checkpoint config.")

    config = AttrDict(config_dict) # Convert to AttrDict for model init


    model = ChronosCore(config).to(device) # Pass AttrDict config to model

    # Load state dict, be flexible with missing/extra keys if needed
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"Warning: Non-strict state dict loading due to mismatch: {e}. Trying strict=False.")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    return model, config # Return model and AttrDict config


def train(args, device, tokenizer):
    print("Running in TRAIN mode...")
    config = vars(args) # Start with CLI args
    # Ensure train data path is saved in config for potential resume with auto-max-length
    config['train_data_path'] = args.train
    config['model_type'] = 'chronos' # Ensure model_type is set
    # <<< MODIFIED: Save dataset type flags in config >>>
    config['pre_chunked_dataset'] = args.pre_chunked_dataset
    config['pre_pt_dataset'] = args.pre_pt_dataset

    # --- Determine vocab_size ---
    current_vocab_size = None
    if tokenizer:
        current_vocab_size = len(tokenizer)
        # If starting fresh AND not loading from a model_path, ensure config gets the vocab_size
        if not args.resume_from_ckpt and not args.model_path and 'vocab_size' not in config:
            config['vocab_size'] = current_vocab_size
        # Special case: If vocab_size was set in args due to adding pad token, use that
        elif 'vocab_size' in config and config['vocab_size'] != current_vocab_size and not args.resume_from_ckpt and not args.model_path:
            print(f"INFO: Using vocab_size {config['vocab_size']} from args (likely due to added pad token).")
        # For resume/load, vocab_size from checkpoint takes precedence later.
    elif not args.resume_from_ckpt and not args.model_path:
        # Should not happen if tokenizer loading in main is correct
        raise RuntimeError("Tokenizer not loaded, cannot determine vocab_size for new model.")

    model = None # Initialize model variable
    optimizer = None # Initialize optimizer variable
    start_epoch = 0
    model_config = None # Initialize model_config
    scaler = None # Initialize scaler
    scheduler = None # Initialize scheduler
    use_amp = args.amp and _HAS_AMP # Determine AMP usage early

    # --- Dataloader creation moved inside resume/load checks ---
    dataloader = None
    dataloader_len = 0 # Length is unknown for iterable dataset


    # --- Handle starting from an existing model directory ---
    if args.model_path and not args.resume_from_ckpt:
        print(f"INFO: Starting training using initial weights from model directory: {args.model_path}")
        try:
            # Load the model and its config. This should be an inference checkpoint.
            model, model_config = load_full_model_with_config(args.model_path, device)

            # Check vocab size consistency
            if current_vocab_size is not None and model_config.vocab_size != current_vocab_size:
                print(f"Warning: Loaded model vocab_size ({model_config.vocab_size}) differs from tokenizer ({current_vocab_size}). Using model's value.")
            elif 'vocab_size' not in model_config and current_vocab_size:
                print(f"Warning: 'vocab_size' missing from loaded model config. Using tokenizer's value ({current_vocab_size}).")
                model_config.vocab_size = current_vocab_size # Patch config
                # Re-initialize parts affected by vocab_size if necessary (tok_emb, lm_head)
                model.tok_emb = nn.Embedding(model_config.vocab_size, model_config.context_dim).to(device)
                model.lm_head = nn.Linear(model_config.context_dim, model_config.vocab_size, bias=False).to(device)
                model.tok_emb.weight = model.lm_head.weight # Re-tie weights
            elif 'vocab_size' not in model_config:
                raise ValueError("Cannot determine vocab_size: Not found in loaded model config and tokenizer not available.")

            # <<< Create Dataloader AFTER model is loaded/configured >>>
            try:
                # Use max_length from the loaded model config if not overridden by args
                max_len_for_loader = args.max_length if args.max_length is not None else model_config.max_length
                if max_len_for_loader is None: raise ValueError("max_length not found in args or loaded config.")

                # <<< MODIFIED: Conditional Dataloader Creation >>>
                if args.pre_pt_dataset:
                    print("INFO: Loading pre-chunked .pt tensors (map-style).")
                    dataloader = create_dataloader_pt_chunked(
                        args.train, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
                    )
                    dataloader_len = len(dataloader) # Map-style dataset has length
                    print(f"INFO: DataLoader created with {dataloader_len} batches.")
                elif args.pre_chunked_dataset:
                    print("INFO: Loading pre-chunked JSONL dataset (iterable).")
                    dataloader = create_dataloader_for_chunked(
                        args.train, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
                    )
                    # Estimate length for scheduler
                    try:
                        with open(args.train, 'r') as f:
                            estimated_lines = sum(1 for _ in f)
                            dataloader_len = estimated_lines // args.batch_size # Rough estimate
                            print(f"INFO: Estimated DataLoader length (for scheduler): {dataloader_len} batches.")
                    except:
                            print("Warning: Could not estimate dataset length for scheduler. Using placeholder T_max=100000.")
                            dataloader_len = 100000 # Placeholder large number
                else:
                    print("INFO: Loading and tokenizing dataset on the fly (map-style).")
                    dataloader = create_dataloader_original(
                        args.train, tokenizer, max_len_for_loader, args.batch_size,
                        tokenizer.pad_token_id, kayla_mode=args.kayla, num_workers=args.num_workers
                    )
                    if dataloader is None: raise ValueError("DataLoader creation failed.")
                    dataloader_len = len(dataloader) # Map-style dataset has length
                    print(f"INFO: DataLoader created with {dataloader_len} batches.")

            except Exception as e:
                print(f"ERROR creating DataLoader: {e}"); traceback.print_exc(); sys.exit(1)


            # --- Initialize optimizer, scaler, scheduler FRESH ---
            print("INFO: Initializing optimizer, scheduler, and scaler from scratch.")
            optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
            if use_amp:
                scaler = GradScaler()
                print("INFO: Automatic Mixed Precision (AMP) ENABLED for training.")

            num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0
            if not args.disable_lr_schedule and num_update_steps > 0:
                print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
                scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
            # start_epoch remains 0

        except FileNotFoundError:
            print(f"ERROR: --model-path specified ({args.model_path}), but it does not seem to contain a valid model directory.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load model from --model-path ({args.model_path}): {e}")
            traceback.print_exc()
            sys.exit(1)

    # <<< Resume logic >>>
    elif args.resume_from_ckpt:
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
            # Patch vocab_size if missing or different
            if 'vocab_size' not in model_config:
                if current_vocab_size:
                    print(f"Warning: 'vocab_size' not found in checkpoint config. Setting from loaded tokenizer ({current_vocab_size}).")
                    model_config['vocab_size'] = current_vocab_size
                else:
                    raise ValueError("Cannot determine vocab_size: Not found in checkpoint and tokenizer not loaded.")
            elif current_vocab_size is not None and model_config.vocab_size != current_vocab_size:
                print(f"Warning: Checkpoint vocab_size ({model_config.vocab_size}) differs from loaded tokenizer ({current_vocab_size}). Using checkpoint value.")

            # Ensure model_type is present for HuggingFace compatibility
            if 'model_type' not in model_config:
                model_config['model_type'] = 'chronos'

            print("INFO: Re-initializing model architecture from checkpoint config.")
            model = ChronosCore(model_config).to(device) # Create model AFTER potentially fixing vocab_size
        else:
            print("Warning: Config not found in checkpoint. Using current CLI args for model architecture.")
            cli_config = config # Use the initial config from vars(args)
            if 'vocab_size' not in cli_config and current_vocab_size:
                cli_config['vocab_size'] = current_vocab_size
            elif 'vocab_size' not in cli_config:
                raise ValueError("Cannot determine vocab_size: Not found in checkpoint or CLI args, and tokenizer not loaded.")
            model_config = AttrDict(cli_config) # Fallback, might cause issues if arch changed
            model = ChronosCore(model_config).to(device)


        # <<< Create Dataloader AFTER model config is determined >>>
        # Determine train path from config if not given via CLI
        train_path_for_loader = args.train if args.train else model_config.get('train_data_path')
        if not train_path_for_loader:
            raise ValueError("Training data path not found in CLI args or checkpoint config during resume.")
        try:
            # Use max_length from args if provided, else from checkpoint config
            max_len_for_loader = args.max_length if args.max_length is not None else model_config.get('max_length')
            if max_len_for_loader is None: raise ValueError("max_length not found in args or checkpoint config during resume.")

            # <<< MODIFIED: Determine dataset type based on flags or saved config >>>
            use_pre_pt = args.pre_pt_dataset or model_config.get('pre_pt_dataset', False)
            use_pre_jsonl = args.pre_chunked_dataset or model_config.get('pre_chunked_dataset', False)

            if use_pre_pt:
                 print("INFO: Loading pre-chunked .pt tensors (map-style, resuming).")
                 dataloader = create_dataloader_pt_chunked(
                     train_path_for_loader, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
                 )
                 dataloader_len = len(dataloader)
                 print(f"INFO: DataLoader created with {dataloader_len} batches.")
            elif use_pre_jsonl:
                print("INFO: Loading pre-chunked JSONL dataset (iterable, resuming).")
                dataloader = create_dataloader_for_chunked(
                    train_path_for_loader, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
                )
                # Estimate length for scheduler
                try:
                    with open(train_path_for_loader, 'r') as f:
                        estimated_lines = sum(1 for _ in f)
                        dataloader_len = estimated_lines // args.batch_size # Rough estimate
                        print(f"INFO: Estimated DataLoader length (for scheduler): {dataloader_len} batches.")
                except:
                        print("Warning: Could not estimate dataset length for scheduler. Using placeholder T_max=100000.")
                        dataloader_len = 100000 # Placeholder
            else:
                print("INFO: Loading and tokenizing dataset on the fly (map-style, resuming).")
                # Determine kayla mode based on flag or potentially from config
                use_kayla = args.kayla or model_config.get('kayla', False)
                dataloader = create_dataloader_original(
                    train_path_for_loader, tokenizer, max_len_for_loader, args.batch_size,
                    tokenizer.pad_token_id, kayla_mode=use_kayla, num_workers=args.num_workers
                )
                if dataloader is None: raise ValueError("DataLoader creation failed.")
                dataloader_len = len(dataloader)
                print(f"INFO: DataLoader created with {dataloader_len} batches.")

        except Exception as e:
            print(f"ERROR creating DataLoader during resume: {e}"); traceback.print_exc(); sys.exit(1)


        # --- Optimizer Initialization/Loading Logic ---
        initial_lr_for_optim = args.starting_lr if args.override_scheduling else model_config.get('starting_lr', args.starting_lr)
        optimizer = ADAM_OPTIMIZER(model.parameters(), lr=initial_lr_for_optim)

        # Load model state dict with flexibility
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Non-strict model state dict loading: {e}. Trying strict=False.")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # --- Conditional Optimizer State Loading ---
        if not args.override_scheduling:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Starting optimizer from scratch.")
                optimizer = ADAM_OPTIMIZER(model.parameters(), lr=initial_lr_for_optim) # Re-init with correct LR
        else:
            print("INFO: --override-scheduling detected. Skipping loading optimizer state.")

        start_epoch = checkpoint.get('completed_epoch', 0) # Use get for safety

        # --- Initialize AMP GradScaler ---
        if use_amp:
            scaler = GradScaler()
            print("INFO: Automatic Mixed Precision (AMP) ENABLED for training.")
            # Resume GradScaler state
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and not args.override_scheduling:
                print("Resuming GradScaler state.")
                try:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except Exception as e:
                    print(f"Warning: Failed to load scaler state: {e}. Continuing with a fresh scaler.")
            elif not args.override_scheduling:
                # Only warn if not overriding and state is missing
                if 'scaler_state_dict' not in checkpoint or checkpoint['scaler_state_dict'] is None:
                    print("Warning: Scaler state not found in checkpoint. Initializing a fresh scaler.")
            elif args.override_scheduling:
                print("INFO: --override-scheduling set. Initializing a fresh scaler.")


        # --- Initialize Scheduler (AFTER optimizer is potentially re-initialized) ---
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0

        if not args.disable_lr_schedule and num_update_steps > 0:
            # Use current args.starting_lr and args.min_lr when initializing scheduler
            print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)

            # --- Scheduler Resuming Logic ---
            checkpoint_has_scheduler = 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None

            if checkpoint_has_scheduler and not args.override_scheduling:
                # Load old state but WARN if LR args have changed without override flag
                old_lr = model_config.get('starting_lr')
                old_min_lr = model_config.get('min_lr')
                lr_mismatch = (old_lr is not None and not np.isclose(old_lr, args.starting_lr))
                min_lr_mismatch = (old_min_lr is not None and not np.isclose(old_min_lr, args.min_lr))

                if lr_mismatch or min_lr_mismatch:
                    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!! WARNING: New LR flags detected but --override-scheduling was not set.             !!!")
                    print(f"!!!   Your new LR ({args.starting_lr}) / Min LR ({args.min_lr}) WILL BE IGNORED.                  !!!")
                    print(f"!!!   Loading old schedule state (LR: {old_lr}, Min LR: {old_min_lr}).                      !!!")
                    print("!!!   To use your new LR flags, add --override-scheduling to your command.            !!!")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

                print("Resuming learning rate scheduler state.")
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    # Crucially, update optimizer's LR to match the resumed scheduler's state
                    # This handles cases where the saved optimizer state had a different LR
                    for i, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = scheduler.get_last_lr()[i]

                except Exception as e:
                    print(f"Warning: Failed to load scheduler state: {e}. Continuing with potentially incorrect LR.")

            elif args.override_scheduling or not checkpoint_has_scheduler:
                if args.override_scheduling and checkpoint_has_scheduler:
                    print("INFO: --override-scheduling detected. Ignoring checkpoint's scheduler state and using new LR args.")
                elif not checkpoint_has_scheduler:
                    print("Warning: No scheduler state found in checkpoint. Initializing new schedule based on new LR args.")

                # Set the step count to where it should be for the resumed epoch
                steps_per_epoch = dataloader_len // args.accumulation_steps if dataloader_len > 0 else 0
                # last_epoch should be the number of steps *completed*
                scheduler.last_epoch = max(-1, start_epoch * steps_per_epoch -1) # -1 if starting epoch 0
                print(f"INFO: Setting scheduler last_epoch to {scheduler.last_epoch} based on resumed epoch {start_epoch}.")


        print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}.")


    # <<< Starting completely fresh >>>
    else:
        print("INFO: Starting training from scratch (no --resume-from-ckpt or --model-path provided).")
        # Need to ensure vocab_size is in the config used to create the model
        if 'vocab_size' not in config:
            if current_vocab_size:
                config['vocab_size'] = current_vocab_size
            else:
                raise ValueError("Cannot determine vocab_size for new model.")
        # Ensure max_length is set in config
        if 'max_length' not in config or config['max_length'] is None:
            if args.max_length:
                config['max_length'] = args.max_length
            else:
                raise ValueError("max_length not determined for new model (use --max_length or --auto-max-length).")

        model = ChronosCore(config).to(device)
        optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
        model_config = AttrDict(config) # Use the potentially updated CLI args config

        # <<< Create Dataloader AFTER model is created >>>
        try:
            # Use max_length from args
            max_len_for_loader = args.max_length
            if max_len_for_loader is None: raise ValueError("max_length not set for dataloader.")

            # <<< MODIFIED: Conditional Dataloader Creation >>>
            if args.pre_pt_dataset:
                print("INFO: Loading pre-chunked .pt tensors (map-style).")
                dataloader = create_dataloader_pt_chunked(
                    args.train, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
                )
                dataloader_len = len(dataloader)
                print(f"INFO: DataLoader created with {dataloader_len} batches.")
            elif args.pre_chunked_dataset:
                print("INFO: Loading pre-chunked JSONL dataset (iterable).")
                dataloader = create_dataloader_for_chunked(
                    args.train, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
                )
                # Estimate length for scheduler
                try:
                    with open(args.train, 'r') as f:
                        estimated_lines = sum(1 for _ in f)
                        dataloader_len = estimated_lines // args.batch_size # Rough estimate
                        print(f"INFO: Estimated DataLoader length (for scheduler): {dataloader_len} batches.")
                except:
                        print("Warning: Could not estimate dataset length for scheduler. Using placeholder T_max=100000.")
                        dataloader_len = 100000 # Placeholder
            else:
                print("INFO: Loading and tokenizing dataset on the fly (map-style).")
                dataloader = create_dataloader_original(
                    args.train, tokenizer, max_len_for_loader, args.batch_size,
                    tokenizer.pad_token_id, kayla_mode=args.kayla, num_workers=args.num_workers
                )
                if dataloader is None: raise ValueError("DataLoader creation failed.")
                dataloader_len = len(dataloader)
                print(f"INFO: DataLoader created with {dataloader_len} batches.")
        except Exception as e:
            print(f"ERROR creating DataLoader: {e}"); traceback.print_exc(); sys.exit(1)


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

    global_step = 0 # Track global steps for iterable datasets and scheduler

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1} / {args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        total_loss = 0.0
        total_ponder_cost = 0.0
        # Flag to track if backward was called in the current accumulation cycle
        backward_called_in_cycle = False
        steps_in_epoch = 0 # Track steps for averaging

        for i, batch in enumerate(pbar):
            # Handle potential None batch from collate_fn if it was empty
            if batch is None:
                print(f"Warning: Skipping empty batch at step {i}.")
                continue

            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            # --- AMP autocast context ---
            # <<< FIXED: Added device_type >>>
            with autocast(device_type=device.type, enabled=use_amp):
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
                        print(f"\nWarning: Ponder cost is NaN/Inf at step {i+1}. Using only CrossEntropy loss for this step.")
                    combined_loss = cross_entropy_loss
                elif not ce_valid: # CE loss is invalid, skip backward
                    if i % args.accumulation_steps == 0:
                        print(f"\nWarning: CrossEntropy loss is NaN/Inf at step {i+1}. Skipping backward pass for this step.")
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
                steps_in_epoch += 1 # Count step only if loss was valid and backward called


            # --- Optimizer Step (End of Accumulation Cycle) ---
            if (i + 1) % args.accumulation_steps == 0:
                # Only proceed if backward was called at least once in this cycle
                if backward_called_in_cycle:
                    # --- LTM Update (Before Optimizer Step) ---
                    ltm_grads = None
                    # Ensure topk_vals exists and requires grad before accessing .grad
                    if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad and outputs["topk_vals"].grad_fn is not None:
                        # Check grad existence, backward() might not populate it if detached earlier
                        if outputs["topk_vals"].grad is not None:
                            ltm_grads = outputs["topk_vals"].grad
                        # else: # Optional warning
                        #     print(f"\nWarning: LTM topk_vals.grad is None at step {i+1}, skipping LTM update.")

                    if ltm_grads is not None:
                        # Make a copy before potentially modifying in-place with unscaling
                        ltm_grads_copy = ltm_grads.detach().clone()
                        if use_amp:
                            # Manually unscale LTM grads *if* the scaler is currently scaled
                            current_scale = scaler.get_scale()
                            if current_scale != 1.0: # Check if scaling is active
                                if scaler._enabled and scaler._scale is not None:
                                    assert current_scale > 0.0 # Should always be true if scale != 1.0
                                    ltm_grads_copy = ltm_grads_copy / current_scale # Unscale the copy
                                else: # If scaler somehow disabled or scale is None, don't unscale
                                    print(f"\nWarning: Scaler state inconsistent at step {i+1}, cannot unscale LTM grads.")

                        # Use the LTM LR specified in args for updates during training
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
                    global_step += 1 # Increment global step after optimizer step

                else: # Backward was not called in this cycle (all losses were NaN/Inf)
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss(es) in accumulation cycle.")
                    # Still need to zero gradients that might exist from previous cycles if resuming
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False # Reset flag

            # --- Update Progress Bar ---
            # Use steps_in_epoch for averaging loss/ponder
            avg_loss = total_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            avg_ponder = total_ponder_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "lr": f"{current_lr:.2e}"
            })

        # --- End of Epoch ---
        ckpt_path = os.path.join(args.out_dir, f"chronos_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} complete. Saving training checkpoint to {ckpt_path}")

        # Ensure config saved reflects current state (including potential patches)
        config_to_save = dict(model.config) # Get config directly from model
        config_to_save['starting_lr'] = args.starting_lr # Use current CLI args for these
        config_to_save['min_lr'] = args.min_lr
        config_to_save['disable_lr_schedule'] = args.disable_lr_schedule
        config_to_save['train_data_path'] = args.train # Save train path used
        # <<< MODIFIED: Save dataset type flags >>>
        config_to_save['pre_chunked_dataset'] = args.pre_chunked_dataset
        config_to_save['pre_pt_dataset'] = args.pre_pt_dataset
        config_to_save['kayla'] = args.kayla # Save kayla mode used
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

    # Ensure final config is saved correctly
    final_config_to_save = dict(model.config)
    final_config_to_save['starting_lr'] = args.starting_lr
    final_config_to_save['min_lr'] = args.min_lr
    final_config_to_save['disable_lr_schedule'] = args.disable_lr_schedule
    final_config_to_save['train_data_path'] = args.train
    # <<< MODIFIED: Save dataset type flags >>>
    final_config_to_save['pre_chunked_dataset'] = args.pre_chunked_dataset
    final_config_to_save['pre_pt_dataset'] = args.pre_pt_dataset
    final_config_to_save['kayla'] = args.kayla
    if 'vocab_size' not in final_config_to_save:
        print(f"CRITICAL WARNING: vocab_size missing from model config before saving final model!")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': final_config_to_save
    }, final_save_path)

    try:
        tokenizer.save_pretrained(args.out_dir)
        print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer files on completion. Error: {e}")


    if args.quantize_on_complete:
        print("\n--- Training Complete: Starting On-the-Fly Quantization ---")
        # Quantize to a new directory for clarity, e.g., './my_model-INT4'
        quantize_out_dir = args.out_dir.rstrip('/\\') + f"-{args.qtype}"
        quantize(args, device, model, tokenizer, quantize_out_dir)


# <<< FINETUNE Function >>>
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
            target_modules = ["qproj", "in_proj", "h_to_context", "l_to_out", "h_halt_proj", "W_ir", "W_hr", "W_iz", "W_hz", "W_in", "W_hn"] # Include GRU weights
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
        # Target more layers, including internal GRU weights if needed
        target_modules=["qproj", "in_proj", "h_to_context", "l_to_out", "h_halt_proj", "W_ir", "W_hr", "W_iz", "W_hz", "W_in", "W_hn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"], # Ensure LTM can still be updated directly
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # <<< Use conditional dataloader logic >>>
    dataloader_len = 0 # Initialize, length might be unknown
    try:
        # Use max_length from the loaded base model config
        if 'max_length' not in model_config: raise ValueError("Base model config missing max_length for dataloader.")
        max_len_for_loader = model_config.max_length

        # <<< MODIFIED: Conditional Dataloader Creation >>>
        if args.pre_pt_dataset:
            print("INFO: Loading pre-chunked .pt tensors for finetuning (map-style).")
            dataloader = create_dataloader_pt_chunked(
                args.train, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
            )
            dataloader_len = len(dataloader)
            print(f"INFO: DataLoader created with {dataloader_len} batches.")
        elif args.pre_chunked_dataset:
            print("INFO: Loading pre-chunked JSONL dataset for finetuning (iterable).")
            dataloader = create_dataloader_for_chunked(
                args.train, max_length=max_len_for_loader, batch_size=args.batch_size, num_workers=args.num_workers
            )
            # Estimate length for scheduler
            try:
                with open(args.train, 'r') as f:
                    estimated_lines = sum(1 for _ in f)
                    dataloader_len = estimated_lines // args.batch_size # Rough estimate
                    print(f"INFO: Estimated DataLoader length (for scheduler): {dataloader_len} batches.")
            except:
                    print("Warning: Could not estimate dataset length for scheduler. Using placeholder T_max=100000.")
                    dataloader_len = 100000 # Placeholder
        else:
            print("INFO: Loading and tokenizing dataset on the fly for finetuning (map-style).")
            dataloader = create_dataloader_original(
                args.train, tokenizer, max_len_for_loader, args.batch_size,
                tokenizer.pad_token_id, kayla_mode=args.kayla, num_workers=args.num_workers
            )
            if dataloader is None: raise ValueError("DataLoader creation failed.")
            dataloader_len = len(dataloader)
            print(f"INFO: DataLoader created with {dataloader_len} batches.")

    except Exception as e:
        print(f"ERROR creating DataLoader for finetuning: {e}"); traceback.print_exc(); sys.exit(1)


    optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr) # Only trainable params will have grads
    os.makedirs(args.out_dir, exist_ok=True)

    scaler = None
    use_amp = args.amp and _HAS_AMP
    if use_amp:
        scaler = GradScaler()
        print("INFO: Automatic Mixed Precision (AMP) ENABLED for fine-tuning.")

    scheduler = None
    if not args.disable_lr_schedule:
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0
        if num_update_steps > 0:
            print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED for finetuning. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
        else:
            print("Warning: Cannot enable LR schedule, dataset might be too small or empty.")


    optimizer.zero_grad(set_to_none=True)
    global_step = 0 # Track global steps for scheduler

    for epoch in range(args.epochs):
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}")
        total_loss = 0.0
        total_ponder_cost = 0.0
        backward_called_in_cycle = False
        steps_in_epoch = 0

        for i, batch in enumerate(pbar):
            if batch is None: continue # Skip empty batches

            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            # <<< FIXED: Added device_type >>>
            with autocast(device_type=device.type, enabled=use_amp):
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
                        print(f"\nWarning: Ponder cost is NaN/Inf at step {i+1}. Using only CrossEntropy loss.")
                    combined_loss = cross_entropy_loss
                elif not ce_valid:
                    if i % args.accumulation_steps == 0:
                        print(f"\nWarning: CrossEntropy loss is NaN/Inf at step {i+1}. Skipping backward pass.")
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
                steps_in_epoch += 1


            if (i + 1) % args.accumulation_steps == 0:
                if backward_called_in_cycle:
                    # LTM Update (Needs careful handling with PEFT)
                    ltm_grads = None
                    if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad and outputs["topk_vals"].grad_fn is not None:
                        if outputs["topk_vals"].grad is not None:
                            ltm_grads = outputs["topk_vals"].grad

                    if ltm_grads is not None:
                        # Access the base model's LTM module directly
                        base_ltm = model.base_model.model.ltm # Deeper nesting for PeftModel
                        ltm_grads_copy = ltm_grads.detach().clone() # Use a copy

                        if use_amp:
                            current_scale = scaler.get_scale()
                            if current_scale != 1.0 and scaler._enabled and scaler._scale is not None:
                                assert current_scale > 0.0
                                ltm_grads_copy = ltm_grads_copy / current_scale
                            elif current_scale != 1.0:
                                print(f"\nWarning: Scaler state inconsistent at step {i+1}, cannot unscale LTM grads.")

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
                    global_step += 1
                else:
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss(es) in accumulation cycle.")
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False

            # Use steps_in_epoch for averaging
            avg_loss = total_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            avg_ponder = total_ponder_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "lr": f"{current_lr:.2e}"
            })

    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    # Note: Only the adapter (+ saved modules like LTM) is saved here.
    # Save tokenizer too for completeness
    try:
        tokenizer.save_pretrained(args.out_dir)
        print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer with adapter. Error: {e}")


# <<< MERGE LORA Function >>>
# ... (merge_lora remains unchanged) ...
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

    # Copy tokenizer files from the original base model directory (or specified tokenizer path)
    tokenizer_source_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    if tokenizer_source_path:
        try:
            # Reload tokenizer if necessary, or use the one passed in
            if tokenizer is None:
                print(f"Loading tokenizer from {tokenizer_source_path} to save with merged model...")
                tokenizer_to_save = AutoTokenizer.from_pretrained(tokenizer_source_path, trust_remote_code=True)
            else:
                tokenizer_to_save = tokenizer

            tokenizer_to_save.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
        except Exception as e:
            print(f"Warning: Could not save tokenizer files from {tokenizer_source_path}: {e}")
    else:
        print("Warning: No tokenizer source path found, cannot save tokenizer with merged model.")


    print("Merge complete.")

# <<< QUANTIZE Function >>>
# ... (quantize remains unchanged) ...
def quantize(args, device, model=None, tokenizer=None, out_dir=None):
    print(f"Running in QUANTIZE mode with {args.qtype} precision...")

    # Allow passing in an already-loaded model (e.g., from train --quantize-on-complete)
    if model is None or tokenizer is None:
        if not args.model_path:
            raise ValueError("--model-path is required for quantize mode when model/tokenizer not provided.")
        print(f"Loading full-precision model from {args.model_path}...")
        # Tokenizer is loaded from the same directory as the model
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            model, _ = load_full_model_with_config(args.model_path, device)
        except Exception as e:
            print(f"Error loading model or tokenizer from {args.model_path}: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Ensure model is on CPU before quantization
    if model.device != torch.device('cpu'):
        print("Moving model to CPU for quantization...")
        model.cpu()

    # Determine output directory
    if out_dir is None:
        if not args.out_dir:
            # Default to creating a new dir next to the source, e.g., './my_model-INT4'
            source_dir = args.model_path if args.model_path else "./chronos_model"
            out_dir = source_dir.rstrip('/\\') + f"-{args.qtype}"
        else:
            out_dir = args.out_dir

    export_and_quantize_model(out_dir, model, tokenizer, qtype=args.qtype)


# <<< CHAT Function >>>
# ... (chat remains unchanged) ...
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
                # Load shadow model onto the main device (e.g., CPU if no CUDA)
                shadow_model, shadow_config = load_full_model_with_config(args.shadow_model_path, device)
                # Basic config check
                if shadow_config.context_dim != config.context_dim or shadow_config.ltm_slots != config.ltm_slots:
                    print("Warning: Shadow model config differs significantly from quantized config. Learning might be unstable.")
            except Exception as e:
                print(f"Error loading shadow model from {args.shadow_model_path}: {e}")
                traceback.print_exc()
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
            traceback.print_exc()
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
        print(f"         - Max LR: {args.ltm_lr:.2e}, Min LR: {args.ltm_schedule_min_lr:.2e}, Cycle Steps: {args.ltm_schedule_steps}")
        # Schedulers need an optimizer, so we create a dummy one for the LTM LR.
        dummy_param = nn.Parameter(torch.tensor(0.0)) # Needs to be Parameter
        # Use the main LTM LR as the MAX LR for the schedule
        ltm_optimizer = torch.optim.SGD([dummy_param], lr=args.ltm_lr)
        ltm_scheduler = CosineAnnealingLR(
            ltm_optimizer,
            T_max=args.ltm_schedule_steps,
            eta_min=args.ltm_schedule_min_lr
        )

    # Initialize AMP scaler and dummy optimizer for chat learning
    scaler = None
    dummy_optimizer = None
    # Enable AMP for learning if requested AND possible (CUDA available AND (full model OR quantized learning enabled))
    # <<< MODIFIED: Changed device check >>>
    use_amp = args.amp and _HAS_AMP and (not is_quantized or args.enable_quantized_learning) and (device.type == 'cuda')

    if use_amp:
        scaler = GradScaler()
        # Create a dummy optimizer for the scaler to track state (NaNs/Infs)
        dummy_param_amp = nn.Parameter(torch.tensor(0.0)).to(device) # Needs to be Parameter and on device
        dummy_optimizer = torch.optim.SGD([dummy_param_amp], lr=1.0) # Dummy optimizer for AMP scaler
        print("INFO: Automatic Mixed Precision (AMP) ENABLED for online learning.")

    print("\nWelcome to Chronos Chat. Type 'exit' or 'quit' to end.")
    print("Use '/filter time=-<seconds>' or '/filter source=<id>' to constrain memory.")
    print("Example: /filter time=-3600  (memories from the last hour)")
    print("Use '/filter reset' to clear memory filters.")
    if _HAS_KEYBOARD:
        print("Press Ctrl+X to stop generation at any time.")
    print("="*50)

    try:
        min_ts_filter = 0.0
        source_id_filter = None
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ["exit", "quit"]:
                break

            # Simple command parser for filtering
            if prompt.startswith('/filter'):
                parts = prompt.split()
                try:
                    if len(parts) == 1 or parts[1] == 'reset':
                        min_ts_filter = 0.0
                        source_id_filter = None
                        print("[INFO: Memory filters have been reset.]")
                        continue

                    for part in parts[1:]:
                        if '=' not in part: raise ValueError(f"Invalid filter format: {part}")
                        key, value = part.split('=', 1)
                        if key == 'time':
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
                        else:
                            print(f"[ERROR: Unknown filter key: {key}]")
                except Exception as e: # Catch broader errors like split issues
                    print(f"[ERROR: Invalid filter format. Use 'time=-<seconds>' or 'source=<id>'. Details: {e}]")
                continue

            # Kayla format assumes ### wrappers
            prompt_format = f"### Instruction:\n{prompt}\n\n### Response:\n"

            # Always use the main device for initial tokenization
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
                        # State is handled internally by the full model's forward pass


                    logits = outputs["logits"].to(device) # Ensure logits are on main device for sampling
                    next_token_logits = logits[:, -1, :]

                    # Simple argmax sampling (can be replaced with more sophisticated sampling)
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)


                    if next_token_id.item() == tokenizer.eos_token_id:
                        break

                    response_ids.append(next_token_id.item())
                    # Decode token safely, handling potential errors
                    try:
                        decoded_token = tokenizer.decode([next_token_id.item()])
                    except Exception as e:
                        print(f"[Decode Error: {e}]", end="")
                        decoded_token = "" # Continue with empty token


                    # Stop generation if a special token like ### is encountered
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
                    # <<< FIXED: Added device_type >>>
                    with autocast(device_type=target_device.type, enabled=use_amp):
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
                        if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad and outputs["topk_vals"].grad_fn is not None:
                            if outputs["topk_vals"].grad is not None:
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
                        update_model.zero_grad(set_to_none=True)

                        # Copy the updated LTM weights back to the live quantized model
                        if is_quantized:
                            model.ltm.load_state_dict(update_model.ltm.state_dict())

                    else: # Combined loss was None (e.g., NaN)
                        # Still need to handle AMP scaler state if used
                        if use_amp:
                            # Even if backward wasn't called, step/update cycle might be needed
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
                    try: # Handle potential EOFError in some environments
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
                        else:
                            print("Invalid input.")
                    except EOFError:
                        print("\nEOF detected. Assuming 'no' for saving.")
                        break
            else: # Need to re-quantize the shadow model
                output_dir = args.model_path # Overwrite the existing quantized model dir
                while True:
                    try: # Handle potential EOFError
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
                                traceback.print_exc()
                            break
                        elif response in ["n", "no"]:
                            print("Changes will be discarded. Exiting.")
                            break
                        else:
                            print("Invalid input.")
                    except EOFError:
                        print("\nEOF detected. Assuming 'no' for re-quantizing.")
                        break

        elif ltm_has_been_updated:
            print("\nLTM was updated, but saving is disabled (e.g., quantized mode without --enable-quantized-learning). Changes will be lost.")


def main():
    parser = argparse.ArgumentParser(description="Chronos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora"], help="Operation mode.")

    # --- Data and Path Arguments (Universal) ---
    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, default=None, help="[Train/Finetune] Path to training JSON/JSONL file, or directory for pre-chunked .pt tensors.")
    path_group.add_argument("--model-path", type=str, default=None, help="Path to the model directory (required for all modes except 'train' unless resuming or starting from scratch).")
    path_group.add_argument("--out-dir", type=str, default="./chronos_model", help="[Train/Finetune/Merge/Quantize] Directory to save the new model/adapter.")
    path_group.add_argument("--lora-adapter-path", type=str, default=None, help="[Merge/Finetune] Path to the LoRA adapter directory.")
    path_group.add_argument("--tokenizer-path", type=str, default=None, help="Path or HF name of the tokenizer (used if not loading from model-path, defaults to microsoft/phi-2).") # Allow None default
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="[Train] Path to a specific training checkpoint .pt file to resume from.")
    path_group.add_argument("--shadow-model-path", type=str, default=None, help="[Chat] Path to the original full-precision model dir, required for online learning with a quantized model.")

    # <<< MODIFIED: Added --pre_pt_dataset flag >>>
    data_fmt_group = parser.add_mutually_exclusive_group()
    data_fmt_group.add_argument("--pre_chunked_dataset", action="store_true", help="[Train/Finetune] If set, assumes --train points to a pre-tokenized/chunked/padded JSONL (IterableDataset). Requires --max_length.")
    data_fmt_group.add_argument("--pre_pt_dataset", action="store_true", help="[Train/Finetune] If set, assumes --train points to a directory with pre-chunked .pt tensor files and manifest.jsonl (Map-Style Dataset). Requires --max_length.")


    # --- Model Architecture Arguments (for Training) ---
    arch_group = parser.add_argument_group('Architecture (for --mode train, used if not resuming/loading)')
    arch_group.add_argument("--context_dim", type=int, default=2560) # Default for Phi-2
    arch_group.add_argument("--persistent_dim", type=int, default=256) # Adjusted
    arch_group.add_argument("--ltm_slots", type=int, default=2048)
    arch_group.add_argument("--ltm_key_dim", type=int, default=256) # Adjusted
    arch_group.add_argument("--ltm_val_dim", type=int, default=256) # Adjusted
    arch_group.add_argument("--h_hidden", type=int, default=2560) # Match context_dim
    arch_group.add_argument("--l_hidden", type=int, default=2560) # Match context_dim
    arch_group.add_argument("--max_h_steps", type=int, default=10, help="[HRM] Maximum number of high-level refinement steps.")
    arch_group.add_argument("--max_l_steps", type=int, default=10, help="[HRM] Maximum number of low-level iterations before forcing completion.")
    arch_group.add_argument("--l_conv_atol", type=float, default=1e-5, help="[HRM] Absolute tolerance for checking L-module state convergence.")
    arch_group.add_argument("--ltm_topk", type=int, default=4, help="Number of LTM slots to retrieve per token.")
    arch_group.add_argument("--max_length", type=int, default=None, help="Max sequence length. Required for --pre_chunked_dataset or --pre_pt_dataset, otherwise default 1024 or auto-detected.")
    arch_group.add_argument("--auto-max-length", action="store_true", help="Automatically scan the dataset to find the longest sequence and set it as max_length (ignored if using pre-chunked formats).")

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training and Finetuning')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--accumulation-steps", type=int, default=1, help="Simulates a larger batch size.")
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine annealing.")
    train_group.add_argument("--disable-lr-schedule", action="store_true", help="Use a fixed LR instead of cosine annealing.")
    train_group.add_argument("--ltm_lr", type=float, default=1e-2, help="[Static] LR for LTM updates, or [Scheduled] MAX LR for the LTM cosine schedule.")
    train_group.add_argument("--kayla", action="store_true", help="Enable Kayla-style instruction tuning (with thought-process). Ignored if using pre-chunked formats.")
    train_group.add_argument("--lora_r", type=int, default=8, help="[Finetune] LoRA rank.")
    train_group.add_argument("--lora_alpha", type=int, default=16, help="[Finetune] LoRA alpha.")
    train_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="[Finetune] Target percentage of params to train (e.g., 1.5 for 1.5%). Overrides --lora_r.")
    train_group.add_argument("--quantize-on-complete", action="store_true", help="[Train] Automatically quantize after training.")
    train_group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value. Set to 0 to disable.")
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01, help="[HRM] Weight for the ponder cost auxiliary loss.")
    train_group.add_argument("--override-scheduling", action="store_true", help="[Train] If resuming, ignore the scheduler state in the checkpoint and use the new LR args.")
    train_group.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading. Recommended: 2 or 4 for GPU training.")
    train_group.add_argument("--amp", action="store_true", help="[Train/Finetune/Chat] Enable Automatic Mixed Precision (AMP) for training/learning.")


    # --- Inference Arguments ---
    infer_group = parser.add_argument_group('Inference (Chat)')
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--enable-quantized-learning", action="store_true", help="[Chat] Enable LTM updates for quantized models. Requires --shadow-model-path.")
    infer_group.add_argument("--ltm-lora-path", type=str, default=None, help="[Chat] Optional: Path to save/load LTM updates as a separate delta file.")
    infer_group.add_argument("--device", type=str, default="cpu", choices=["cpu", "vulkan"], help="[Chat] Device for quantized inference.")
    infer_group.add_argument("--h-halt-thresh", type=float, default=0.9, help="[HRM] Probability threshold for early exiting the H-module loop during inference.")
    infer_group.add_argument("--static-ltm-lr", action="store_true", help="[Chat] Disable the cosine annealing schedule for LTM updates and use a fixed LR instead.")
    infer_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="[Chat] The number of updates in one cosine annealing cycle for LTM learning.")
    infer_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="[Chat] The minimum learning rate for the LTM cosine annealing schedule.")

    # --- Other Arguments ---
    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument("--qtype", type=str, default="INT4", choices=["INT4", "Q4_0", "Q8_0", "Q2_K"], help="Quantization type/format.")
    other_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.mode == 'train' and not args.train and not args.resume_from_ckpt:
        parser.error("`--train` is required for train mode unless resuming with `--resume-from-ckpt`.")
    if args.mode == 'finetune' and not args.train:
        parser.error("`--train` is required for finetune mode.")
    if args.mode == 'finetune' and not args.model_path:
        parser.error("`--model-path` (base model) is required for finetune mode.")
    if args.mode == 'merge-lora' and not args.model_path:
        parser.error("`--model-path` (base model) is required for merge-lora mode.")
    if args.mode == 'merge-lora' and not args.lora_adapter_path:
        parser.error("`--lora-adapter-path` is required for merge-lora mode.")
    if args.mode == 'quantize' and not args.model_path and not args.quantize_on_complete:
        # Quantize might be called internally, check if model_path wasn't passed via CLI specifically
        is_standalone_quantize = True
        for i, arg in enumerate(sys.argv):
            if arg == 'train' and '--quantize-on-complete' in sys.argv:
                is_standalone_quantize = False
                break
        if is_standalone_quantize and not args.model_path:
            parser.error("`--model-path` is required for standalone quantize mode.")
    if args.mode == 'chat' and not args.model_path:
        parser.error("`--model-path` is required for chat mode.")
    if args.enable_quantized_learning and not args.shadow_model_path:
        parser.error("--enable-quantized-learning requires --shadow-model-path to be set.")
    # <<< MODIFIED: Validation for new dataset types >>>
    if (args.pre_chunked_dataset or args.pre_pt_dataset) and not args.max_length:
         parser.error("--max_length must be specified when using --pre_chunked_dataset or --pre_pt_dataset.")
    if (args.pre_chunked_dataset or args.pre_pt_dataset) and args.auto_max_length:
        print("Warning: --auto-max-length is ignored when using pre-chunked dataset formats.")
    if (args.pre_chunked_dataset or args.pre_pt_dataset) and args.kayla:
         print("Warning: --kayla flag is ignored when using pre-chunked dataset formats.")
    if not args.pre_chunked_dataset and not args.pre_pt_dataset and not args.max_length and not args.auto_max_length:
        if args.mode in ['train', 'finetune']:
            print("INFO: Neither --max_length nor --auto-max-length specified. Will attempt auto-detection or use default 1024.")
            args.auto_max_length = True # Default to auto if nothing else specified
    elif not args.pre_chunked_dataset and not args.pre_pt_dataset and args.max_length:
        args.auto_max_length = False # Explicit length overrides auto


    set_threads(args.threads)
    pt_device = pick_device()
    print(f"Using PyTorch device: {pt_device}")

    # --- AMP Availability Check ---
    if args.amp and not _HAS_AMP:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: --amp was specified, but torch amp support is not available.   !!!")
        print("!!!   AMP will be DISABLED. Check CUDA and PyTorch install.               !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        args.amp = False # Force disable
    elif args.amp and pt_device == torch.device('cpu'):
        print("Warning: AMP (--amp) is enabled but running on CPU. AMP will have no effect.")
        # Don't disable args.amp, just warn

    # --- Tokenizer Loading ---
    tokenizer = None
    tokenizer_load_path = None
    default_tokenizer = "microsoft/phi-2"

    # 1. Prioritize --tokenizer-path if given
    if args.tokenizer_path:
        print(f"Attempting to load tokenizer from specified path: '{args.tokenizer_path}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            tokenizer_load_path = args.tokenizer_path
            print(f"Successfully loaded tokenizer from '{args.tokenizer_path}'")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer from '{args.tokenizer_path}'. Error: {e}")
            # Fall through to try model_path or default

    # 2. If no tokenizer yet, try --model-path (if applicable for the mode)
    if tokenizer is None and args.model_path and args.mode != 'train': # Only try model_path if not fresh train
        print(f"Attempting to load tokenizer from model directory: {args.model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            tokenizer_load_path = args.model_path
            print(f"Successfully loaded tokenizer from '{args.model_path}'")
        except Exception as e:
            print(f"Warning: Could not load tokenizer from model directory '{args.model_path}'. Will try default. Error: {e}")
            # Fall through to try default

    # 3. If still no tokenizer, try the default (important for fresh train)
    if tokenizer is None:
        print(f"Attempting to load default tokenizer: '{default_tokenizer}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(default_tokenizer, trust_remote_code=True)
            tokenizer_load_path = default_tokenizer
            print(f"Successfully loaded tokenizer from default '{default_tokenizer}'")
        except Exception as e:
            # If default fails, it's critical for modes needing it
            if args.mode in ["train", "finetune", "chat"] or (args.mode == "quantize" and not args.quantize_on_complete) or args.mode == "merge-lora":
                print(f"ERROR: Failed to load tokenizer from all potential sources (specified: '{args.tokenizer_path}', model: '{args.model_path}', default: '{default_tokenizer}'). Cannot continue.")
                print(f"Details: {e}")
                sys.exit(1)
            else:
                print(f"Warning: Tokenizer loading failed from all paths. Relying on it being passed directly later if needed.")
                tokenizer = None


    # --- Tokenizer Pad Token Handling ---
    if tokenizer and tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer missing pad token, setting pad_token = eos_token ({tokenizer.pad_token})")
        else:
            print("Warning: Tokenizer missing both pad and eos tokens. Adding a '[PAD]' token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # If adding token, vocab size changes. Model needs to know ONLY IF TRAINING FROM SCRATCH.
            if args.mode == "train" and not args.resume_from_ckpt and not args.model_path:
                # Update args directly as config isn't finalized yet
                args.vocab_size = len(tokenizer) # Set vocab_size based on modified tokenizer
                print(f"Updated args.vocab_size to {args.vocab_size} due to added pad token.")


    # --- Auto Max Length Scanning (Only if not pre-chunked and requested/defaulted) ---
    # <<< MODIFIED: Skip scanning if using pre-chunked formats >>>
    if args.auto_max_length and not args.pre_chunked_dataset and not args.pre_pt_dataset and args.mode in ['train', 'finetune']:
        train_file_path = args.train
        if not train_file_path and args.resume_from_ckpt:
            # Try loading from checkpoint config if resuming and path not given
            print("INFO: --auto-max-length used with --resume-from-ckpt. Trying to load train path from config...")
            try:
                ckpt = torch.load(args.resume_from_ckpt, map_location='cpu', weights_only=False) # Need config
                ckpt_conf = ckpt.get('config', {})
                train_file_path = ckpt_conf.get('train_data_path')
            except Exception as e: print(f"Warning: Could not load train path from checkpoint config: {e}")

        if not train_file_path or not os.path.exists(train_file_path):
            parser.error("--auto-max-length requires a valid --train file path.")
        if tokenizer is None:
            parser.error("--auto-max-length requires the tokenizer to be loaded first.")

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
                except: return ""

            try: # Try JSON list
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    for obj in tqdm(data, desc="Scanning JSON"):
                        text = get_text_from_obj(obj, args.kayla)
                        if tokenizer:
                            length = len(tokenizer.encode(text, add_special_tokens=True)) + 1 # Include special tokens + EOS
                            if length > max_found_length: max_found_length = length
                        else: break # Stop if no tokenizer
            except: # Try JSONL
                f.seek(0)
                line_num_scan = 0
                for line in tqdm(f, desc="Scanning JSONL"):
                    line_num_scan += 1
                    try:
                        obj = json.loads(line)
                        text = get_text_from_obj(obj, args.kayla)
                        if tokenizer:
                            length = len(tokenizer.encode(text, add_special_tokens=True)) + 1
                            if length > max_found_length: max_found_length = length
                        else: break
                    except Exception as scan_e:
                        print(f"Warning: Skipping line {line_num_scan} during scan: {scan_e}")
                        continue

        if max_found_length > 0:
            # Add a small buffer and round up to a multiple of 8 for efficiency
            target_max_length = (max_found_length + 16 + 7) & -8
            print(f"✅ Auto-scan complete. Found max length ~{max_found_length}. Setting max_length to {target_max_length}.")
            args.max_length = target_max_length # Update args for model creation/dataloader
            # Update tokenizer only if necessary
            if tokenizer and hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < target_max_length:
                tokenizer.model_max_length = target_max_length
                print(f"Updated tokenizer.model_max_length to {target_max_length}")
        else:
            print("⚠️ WARNING: Auto-scan did not find valid entries or failed. Using default max_length (1024).")
            args.max_length = args.max_length or 1024 # Set to default if scan failed and no explicit value
    elif not args.max_length and args.mode in ['train', 'finetune'] and not args.pre_chunked_dataset and not args.pre_pt_dataset:
        # Case where auto not enabled, not pre-chunked, and no length given
        print("Warning: No --max_length specified and --auto-max-length disabled. Using default 1024.")
        args.max_length = 1024


    # --- Execute Selected Mode ---
    if args.mode == "train":
        if tokenizer is None and not args.pre_pt_dataset: # Need tokenizer unless loading PT files
            print("Error: Tokenizer failed to load, cannot start training.")
            sys.exit(1)
        train(args, pt_device, tokenizer)
    elif args.mode == "finetune":
        if tokenizer is None and not args.pre_pt_dataset: # Need tokenizer unless loading PT files
            print("Error: Tokenizer failed to load, cannot start finetuning.")
            sys.exit(1)
        finetune(args, pt_device, tokenizer)
    elif args.mode == "merge-lora":
        # Tokenizer might be optional for merge itself, but needed for saving output dir
        merge_lora(args, pt_device, tokenizer) # Pass potentially None tokenizer
    elif args.mode == "quantize":
        # Quantize handles internal loading if tokenizer is None here
        quantize(args, pt_device, tokenizer=tokenizer)
    elif args.mode == "chat":
        if tokenizer is None:
            print("Error: Tokenizer failed to load, cannot start chat.")
            sys.exit(1)
        chat(args, pt_device, tokenizer)


if __name__ == "__main__":
    main()
