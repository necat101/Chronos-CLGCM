"""Unsupported legacy monolith.

Use ``hierarchos_cli.py`` and the modular ``hierarchos`` package. In
particular, this historical file's scalar-RWKV ``.npz`` quantization path does
not implement the active coherent-v9 recurrence and is deliberately disabled.
"""

if __name__ == "__main__":
    import sys as _startup_sys

    print(
        "ERROR: hierarchos.py is an unsupported historical monolith and must "
        "not be used for training or inference."
    )
    print(
        "       Run the maintained modular entry point instead: "
        "python hierarchos_cli.py <mode> ..."
    )
    raise SystemExit(2)

import os
import sys
import json
import argparse
import time
import subprocess
import functools
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm
import traceback # Added for better error reporting
import signal # <<< MODIFIED: Import signal >>>
from torch.compiler import cudagraph_mark_step_begin # <<< ADD THIS LINE
import math # For ceil in dataset chunking (though that script is separate)

_LEGACY_QUANTIZATION_UNSUPPORTED = (
    "Legacy hierarchos.py quantization is unsupported: its scalar-RWKV .npz "
    "format cannot preserve the active coherent-v9 recurrence/logit contract. "
    "Use hierarchos_cli.py with a full-precision hierarchos.pt checkpoint."
)

# <<< MODIFIED: Set Tokenizers Parallelism Environment Variable >>>
# Set this early, before tokenizers might be implicitly loaded by other imports
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_msvc_environment():
    # Only run this environment setup logic if on Windows
    if os.name != 'nt':
        return

    CACHE_FILE = 'vcvars_path.cache.txt'
    vcvars_path = None

    # 1. Try to load the path from the cache file
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cached_path = f.read().strip()
            if os.path.exists(cached_path):
                print(f"INFO: Found cached vcvars64.bat path: {cached_path}")
                vcvars_path = cached_path
            else:
                print(f"INFO: Cached path '{cached_path}' no longer exists. Will try auto-detect.")
        except Exception as e:
            print(f"WARNING: Could not read cache file. Will try auto-detect. Error: {e}")

    # 2. If no valid path from cache, try auto-detection
    if vcvars_path is None:
        print("INFO: No cached vcvars path. Attempting auto-detection...")
        # Check standard paths for vswhere.exe
        vswhere_path_progfiles_x86 = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        vswhere_path_progfiles = os.path.expandvars(r"%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe")
        
        vswhere_path = None
        if os.path.exists(vswhere_path_progfiles_x86):
            vswhere_path = vswhere_path_progfiles_x86
        elif os.path.exists(vswhere_path_progfiles):
            vswhere_path = vswhere_path_progfiles
            
        if vswhere_path:
            try:
                # <<< MODIFICATION: Use vswhere to find vcvars64.bat directly >>>
                print("INFO: vswhere.exe found. Querying directly for vcvars64.bat...")
                cmd = [
                    vswhere_path,
                    "-latest",
                    "-products", "*",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find", r"VC\Auxiliary\Build\vcvars64.bat",
                    "-nologo"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
                found_path = result.stdout.strip()

                if found_path and os.path.exists(found_path):
                    vcvars_path = found_path
                    print(f"✅ INFO: vswhere.exe successfully found vcvars64.bat at: {vcvars_path}")
                else:
                    print(f"INFO: vswhere.exe ran but did not return a valid path.")
                    print(f"   (stdout: {result.stdout})")

            except Exception as e:
                print(f"WARNING: 'vswhere.exe' direct find failed. Will fall back to manual prompt. Error: {e}")
                if hasattr(e, 'stdout'): print(f"   (stdout: {e.stdout})")
                if hasattr(e, 'stderr'): print(f"   (stderr: {e.stderr})")
        else:
            print(f"INFO: 'vswhere.exe' not found in standard locations. Will fall back to manual prompt.")

    # 3. If auto-detection failed, fall back to user prompt
    if vcvars_path is None:
        print("---" * 20)
        print("Microsoft Visual C++ (MSVC) 64-bit Compiler Setup (Manual Fallback)")
        print("---" * 20)
        print("Could not automatically find your 64-bit C++ compiler.")
        print("To use 'torch.compile' (e.g., with --gradient-checkpointing), this script")
        print("needs to find your 'vcvars64.bat' file.")
        
        try:
            user_path = input("Enter the full path to vcvars64.bat (or press Enter to skip): ").strip()

            if user_path and os.path.exists(user_path):
                vcvars_path = user_path
                print("INFO: Path validated.")
            elif user_path:
                print(f"WARNING: Path '{user_path}' not found or is invalid.")
                print("         Skipping environment setup. torch.compile may fail.")
            else:
                print("INFO: Skipping 64-bit environment setup.")
        
        except EOFError:
             print("\nINFO: No input received. Skipping 64-bit environment setup.")
        
        print("---" * 20)

    # 4. If we have a valid path, run it and set the environment
    if vcvars_path:
        try:
            with open(CACHE_FILE, 'w') as f:
                f.write(vcvars_path)
            print(f"INFO: 64-bit compiler path cached to '{CACHE_FILE}' for future runs.")
        except Exception as e:
            print(f"WARNING: Could not write cache file. You may be asked again. Error: {e}")

        print(f"INFO: Loading 64-bit MSVC environment from '{vcvars_path}'...")
        try:
            python_exe = sys.executable
            vcvars_dir = os.path.dirname(vcvars_path)
            vcvarsall_path = os.path.join(vcvars_dir, "vcvarsall.bat")

            if not os.path.exists(vcvarsall_path):
                print(f"❌ ERROR: 'vcvarsall.bat' not found in the same directory as 'vcvars64.bat'.")
                raise FileNotFoundError("vcvarsall.bat not found")

            cmd = f'"{vcvarsall_path}" x64 >NUL 2>&1 && echo ^"---ENV-JSON-START---^" && "{python_exe}" -c "import json, os; print(json.dumps(dict(os.environ)))"'
            
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,
                check=True,
                startupinfo=startupinfo,
                encoding='utf-8',
                errors='ignore',
                cwd=vcvars_dir
            )
            
            json_start = result.stdout.find("---ENV-JSON-START---")
            if json_start != -1:
                raw_output_after_marker = result.stdout[json_start + len("---ENV-JSON-START---"):]
                json_obj_start = raw_output_after_marker.find('{')
                json_obj_end = raw_output_after_marker.rfind('}')
                
                if json_obj_start != -1 and json_obj_end != -1 and json_obj_end > json_obj_start:
                    json_str = raw_output_after_marker[json_obj_start : json_obj_end + 1]
                    try:
                        env_vars = json.loads(json_str)
                        new_path = env_vars.get('PATH')
                        new_lib = env_vars.get('LIB')
                        new_include = env_vars.get('INCLUDE')
                        
                        if new_path is not None: os.environ['PATH'] = new_path
                        if new_lib is not None: os.environ['LIB'] = new_lib
                        else: os.environ['LIB'] = ""
                        if new_include is not None: os.environ['INCLUDE'] = new_include
                        else: os.environ['INCLUDE'] = ""
                        
                        print("INFO: 64-bit MSVC environment loaded successfully.")
                    except json.JSONDecodeError as json_e:
                        print(f"ERROR: Failed to parse the environment JSON. {json_e}")
                else:
                    print(f"WARNING: Could not find valid JSON object markers.")
            else:
                print(f"WARNING: Could not find JSON marker in vcvarsall.bat output.")

        except Exception as e:
            print(f"ERROR: An unexpected error occurred while trying to run vcvarsall.bat.")
            print(f"   Error: {e}")
            traceback.print_exc(limit=2)
        print("---" * 20)


# --- Fix for Python Build Environment (Windows) ---
# 1. Fix spaces in include paths (monkey-patch sysconfig).
# 2. Fix missing python313.lib (add libs to LIB env var).
import sysconfig
import ctypes

def _fix_python_build_environment_windows():
    if os.name != 'nt': return
    
    try:
        _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        _GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        _GetShortPathNameW.restype = ctypes.c_uint32

        def get_short_path(path):
            if not path or not os.path.exists(path): return path
            buf_size = _GetShortPathNameW(path, None, 0)
            if buf_size == 0: return path
            buf = ctypes.create_unicode_buffer(buf_size)
            _GetShortPathNameW(path, buf, buf_size)
            return buf.value

        # --- 1. Patch sysconfig for include paths ---
        _original_get_path = sysconfig.get_path
        _original_get_paths = sysconfig.get_paths

        def _patched_get_path(name, scheme=sysconfig.get_default_scheme(), vars=None, expand=True):
            path = _original_get_path(name, scheme, vars, expand)
            if name in ('include', 'platinclude') and path:
                return get_short_path(path)
            return path

        def _patched_get_paths(scheme=sysconfig.get_default_scheme(), vars=None, expand=True):
            paths = _original_get_paths(scheme, vars, expand)
            for key in ('include', 'platinclude'):
                if key in paths and paths[key]:
                    paths[key] = get_short_path(paths[key])
            return paths

        sysconfig.get_path = _patched_get_path
        sysconfig.get_paths = _patched_get_paths
        print("INFO: Patched sysconfig to use short paths for includes.")

        # --- 2. Add 'libs' to LIB environment variable ---
        # Windows Store Python often has libs in sys.exec_prefix (Program Files), 
        # but the compiler might not find them, or spaces might be an issue.
        exec_prefix_short = get_short_path(sys.exec_prefix)
        libs_path = os.path.join(exec_prefix_short, 'libs')
        
        if os.path.exists(libs_path):
            current_lib = os.environ.get('LIB', '')
            os.environ['LIB'] = f"{libs_path};{current_lib}"
            print(f"INFO: Added '{libs_path}' to LIB environment variable.")
        else:
            print(f"WARNING: Could not find libs path at '{libs_path}'")

    except Exception as e:
        print(f"WARNING: Failed to setup Python build environment: {e}")

# Call it immediately
_fix_python_build_environment_windows()

import torch
import torch.nn as nn
import torch.nn.functional as F
# <<< MODIFIED: Import IterableDataset and Dataset >>>
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
# Handle safe_globals compatibility (PyTorch 2.4+)
try:
    from torch.serialization import safe_globals
except ImportError:
    # Fallback for older PyTorch versions or if not exposed
    import contextlib
    @contextlib.contextmanager
    def safe_globals(allowlist):
        yield

# Import signal for graceful shutdown
import signal
import traceback
from torch.compiler import cudagraph_mark_step_begin
from torch.utils.checkpoint import checkpoint



# <<< NEW: Import Hugging Face datasets library >>>
try:
    from datasets import load_dataset
    _HAS_HF_DATASETS = True
except ImportError:
    print("Warning: 'datasets' library not found. Loading datasets from Hugging Face Hub or local files (e.g., CSV) using --hf_dataset will be unavailable.")
    print("         Please install it: pip install datasets")
    _HAS_HF_DATASETS = False


# --- Optional Imports ---
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    _HAS_PEFT = True
except ImportError:
    print("Warning: 'peft' library not found. LoRA fine-tuning and merging will be unavailable.")
    _HAS_PEFT = False

# --- Inductor Workarounds for Windows CPU ---
if os.name == 'nt' and not torch.cuda.is_available():
    print("INFO: Windows CPU detected. Applying Inductor config workarounds...")
    import torch._inductor.config
    torch._inductor.config.cpp_wrapper = False # Fixes AssertionError: buf72 in codegen and potential performance issues

# --- DirectML Import (AMD GPU Acceleration on Windows) ---
_HAS_DIRECTML = False
try:
    import torch_directml
    _HAS_DIRECTML = True
    # Don't print "found" yet to keep logs clean unless we actually use it
except ImportError:
    _HAS_DIRECTML = False

# <<< MODIFICATION: Early check for DML flag in sys.argv >>>
# We need to know this before setting up global optimizers/compilers
_REQUESTED_DML = False
if "--device" in sys.argv:
    try:
        _dev_idx = sys.argv.index("--device")
        if _dev_idx + 1 < len(sys.argv) and sys.argv[_dev_idx+1].lower() in ['dml', 'directml']:
            _REQUESTED_DML = True
    except ValueError:
        pass

# --- Call MSVC Setup ONLY if NOT using DirectML and CUDA is available ---
# We avoid this on DirectML to prevent the "zombie" compiler search.
if not _REQUESTED_DML and os.name == 'nt':
    setup_msvc_environment()


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
        # Fallback for older PyTorch versions
        from torch.cuda.amp import GradScaler, autocast
        print("INFO: torch.cuda.amp (autocast/GradScaler) found.")
        _torch_amp_available = True
    except ImportError:
        print("Warning: torch.amp and torch.cuda.amp not found.")
        _torch_amp_available = False
        import contextlib
        @contextlib.contextmanager
        def autocast(device_type, enabled=True, dtype=None):
            yield

# --- Now, determine optimizer and final AMP status based on device ---
# <<< MODIFICATION: Only enter this block if DML was explicitly requested >>>
if _HAS_DIRECTML and _REQUESTED_DML:
    # DirectML explicitly requested
    print("INFO: DirectML requested and found. Using standard torch.optim.AdamW optimizer.")
    ADAM_OPTIMIZER = torch.optim.AdamW
    _HAS_BNB = False
    
    # DirectML has limited AMP support - disable for stability as requested
    _HAS_AMP = False
    print("INFO: AMP support is DISABLED for DirectML (stability fix).")
        
elif torch.cuda.is_available():
    # CUDA detected - attempt to use bitsandbytes
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
    
    if _torch_amp_available:
        _HAS_AMP = True
        print("INFO: AMP support is enabled (CUDA available).")
    else:
        _HAS_AMP = False
        print("Warning: AMP support is disabled (torch.amp/torch.cuda.amp import failed).")
        
else:
    # CPU fallback (Default behavior if DML not requested and no CUDA)
    print("INFO: Using CPU training (CUDA not found or not selected).")
    ADAM_OPTIMIZER = torch.optim.AdamW
    _HAS_BNB = False
    _HAS_AMP = False
    if _torch_amp_available:
        print("INFO: AMP components found but disabled (running on CPU).")

# --- END Optimizer Selection ---



# --- C++ Kernel Import ---
_HAS_KERNEL = False
_HAS_VULKAN = False

# Disable custom kernels if DirectML is present to avoid incompatibility
# We allow kernel import even if DirectML is present, to support mixed environments (e.g. CUDA + DML installed).
# Runtime checks will prevent using kernels on DML devices if incompatible.
# We allow kernel import even if DirectML is present, to support mixed environments (e.g. CUDA + DML installed).
# Runtime checks will prevent using kernels on DML devices if incompatible.
# (Logic moved to main() and QuantizedHierarchos)
try:
    import hierarchos_matmul
    _HAS_KERNEL = True
    print("Successfully imported C++ quantization kernel.")
    if hasattr(hierarchos_matmul, "VULKAN_SUPPORT") and hierarchos_matmul.VULKAN_SUPPORT:
        print("INFO: Vulkan support is enabled in the compiled kernel.")
        _HAS_VULKAN = True
    else:
        print("INFO: Vulkan support is disabled in the compiled kernel.")
        _HAS_VULKAN = False
except ImportError:
    print("Warning: 'hierarchos_matmul' not found. Quantization disabled.")



# --- CONSTANTS ---
MODEL_WEIGHTS_NAME = "hierarchos.pt"
QUANTIZED_MODEL_WEIGHTS_NAME_TPL = "hierarchos-{qtype}.npz" # Template for the name


# --- HELPER CLASS: AttrDict ---
class AttrDict(dict):
    """A dictionary that allows for attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# --- Device Utilities ---
def pick_device(args=None):
    """Picks the best available device for PyTorch training."""
    # Check CLI args first if provided
    if args and hasattr(args, 'device') and args.device:
        if args.device in ['dml', 'directml']:
            if _HAS_DIRECTML: 
                print("INFO: User explicitly requested DirectML.")
                return torch_directml.device()
            else: 
                print("Warning: DirectML requested but torch-directml not available. Falling back to auto-detect.")
        elif args.device == 'cuda':
            if torch.cuda.is_available(): 
                return torch.device("cuda")
            else: 
                print("Warning: CUDA requested but not available. Falling back to auto-detect.")
        elif args.device == 'cpu':
            return torch.device("cpu")
    
    # Auto-detection priority: CUDA -> CPU (DML is now OPT-IN only)
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Note: We do NOT return torch_directml.device() here automatically anymore.
    
    return torch.device("cpu")

def set_threads(n: int):
    try:
        torch.set_num_threads(n)
        os.environ['OMP_NUM_THREADS'] = str(n)
    except Exception as e:
        print(f"Warning: Could not set thread count. {e}")

def is_directml_device(device):
    """Check if a device is DirectML."""
    if _HAS_DIRECTML:
        return isinstance(device, type(torch_directml.device()))
    return False

def get_device_type(device):
    """Get normalized device type string for compatibility checks."""
    if is_directml_device(device):
        return 'dml'
    elif isinstance(device, torch.device):
        return device.type
    return str(device)


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

    # DirectML doesn't support pin_memory in the same way as CUDA
    # Only enable pin_memory for actual CUDA devices
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

    # DirectML doesn't support pin_memory in the same way as CUDA
    pin_memory = torch.cuda.is_available() and num_workers > 0
    persistent_workers = num_workers > 0

    # Map-style dataset allows shuffling
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_pt, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers)

# <<< END: Modified Map-Style Dataset/Loader for Consolidated PT Tensors >>>


# <<< START: Utility function for processing text based on args (shared by JSONL and HF dataset) >>>
def process_text_sample(tokenizer, text_dict: dict, max_length: int, kayla_mode: bool = False,
                         text_column: Optional[str] = None,
                         prompt_column: Optional[str] = None, completion_column: Optional[str] = None):
    """
    Processes a dictionary containing text into tokenized input_ids and labels
    based on the specified mode (Kayla, instruction tuning, or text completion).
    Returns a dictionary {"input_ids": tensor, "labels": tensor} or None on error.
    """
    input_ids = []
    labels = []

    try:
        if text_column:
            # --- Text Completion Mode (e.g., pre-training) ---
            text = text_dict.get(text_column, "")
            if not isinstance(text, str): text = str(text) # Attempt conversion
            if not text: return None # Skip empty text

            input_ids = tokenizer.encode(text, add_special_tokens=True) + [tokenizer.eos_token_id]
            labels = list(input_ids) # Predict every token

        elif prompt_column and completion_column:
            # --- Instruction Tuning Mode (Standard or Kayla) ---
            if kayla_mode:
                # --- Kayla Format ---
                instruction = text_dict.get(prompt_column, "") # Use prompt_column for instruction
                completion = text_dict.get(completion_column, "") # Use completion_column for output
                # Assumes 'feelings' and 'thought-process' are also present in text_dict if needed
                feelings = text_dict.get('feelings', '')
                thought = text_dict.get('thought-process', '')

                if not isinstance(instruction, str): instruction = str(instruction)
                if not isinstance(completion, str): completion = str(completion)
                if not isinstance(feelings, str): feelings = str(feelings)
                if not isinstance(thought, str): thought = str(thought)


                instruction_text = f"### Instruction:\n{instruction}\n\n"
                feelings_text = f"### Feelings:\n{feelings}\n\n" if feelings else ""
                prompt_context_text = instruction_text + feelings_text
                thought_text = f"### Thought Process:\n{thought}\n\n"
                output_text = f"### Response:\n{completion}"

                prompt_context_tokens = tokenizer.encode(prompt_context_text, add_special_tokens=True)
                thought_tokens = tokenizer.encode(thought_text, add_special_tokens=False)
                output_tokens = tokenizer.encode(output_text, add_special_tokens=False)

                input_ids = prompt_context_tokens + thought_tokens + output_tokens + [tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_context_tokens)) + thought_tokens + output_tokens + [tokenizer.eos_token_id]

            else:
                # --- Standard Format (Modified for Alpaca) ---
                instruction_text = text_dict.get(prompt_column, "")
                # Check for an 'input' column, which is common in Alpaca
                input_text = text_dict.get('input', "") 
                completion_text = text_dict.get(completion_column, "")

                if not isinstance(instruction_text, str): instruction_text = str(instruction_text)
                if not isinstance(input_text, str): input_text = str(input_text)
                if not isinstance(completion_text, str): completion_text = str(completion_text)

                # Combine instruction and input if input exists
                if input_text:
                    prompt_formatted = f"### Instruction:\n{instruction_text}\n\n### Input:\n{input_text}\n\n### Response:\n"
                else:
                    prompt_formatted = f"### Instruction:\n{instruction_text}\n\n### Response:\n"

                prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=True)
                completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

                input_ids = prompt_tokens + completion_tokens + [tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_tokens)) + completion_tokens + [tokenizer.eos_token_id]

        else:
            # Invalid configuration or missing required columns
            # Try a default 'text' column if nothing else specified
            text = text_dict.get('text', "")
            if isinstance(text, str) and text:
                print("Warning: No text/prompt/completion columns specified. Falling back to 'text' column for text completion.")
                input_ids = tokenizer.encode(text, add_special_tokens=True) + [tokenizer.eos_token_id]
                labels = list(input_ids)
            else:
                 print(f"Warning: Skipping data entry due to missing or invalid text columns: {list(text_dict.keys())}")
                 return None # Skip this sample

        # --- Truncation ---
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length-1] + [tokenizer.eos_token_id]
            labels = labels[:max_length-1] + [tokenizer.eos_token_id]

        # Padding is handled by the collate function
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

    except Exception as e:
        obj_repr = str(text_dict)
        print(f"Warning: Skipping invalid data entry: {obj_repr[:150] + ('...' if len(obj_repr) > 150 else '')}. Error: {e}")
        traceback.print_exc(limit=1)
        return None

# <<< END: Utility function for processing text >>>


# <<< START: Original Dataset/Loader (Renamed) - Modified to use process_text_sample >>>
class OriginalJSONLDataset(Dataset):
    """
    Handles both .jsonl (one JSON object per line) and .json (a list of objects) files.
    Also supports standard and "Kayla" instruction formats using standard keys like
    'instruction', 'output', 'Instruction', 'thought-process', 'feelings'.
    Tokenizes data on the fly. Loads everything into RAM.
    """
    def __init__(self, path: str, tokenizer, max_length: int, kayla_mode: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.kayla_mode = kayla_mode
        self._load(path)

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
                        # <<< MODIFIED: Use shared processing function >>>
                        processed = process_text_sample(
                            self.tokenizer, obj, self.max_length, self.kayla_mode,
                            prompt_column='instruction', # Default keys for JSONL
                            completion_column='output'
                        )
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
                    # <<< MODIFIED: Use shared processing function >>>
                    processed = process_text_sample(
                        self.tokenizer, obj, self.max_length, self.kayla_mode,
                        prompt_column='instruction', # Default keys for JSONL
                        completion_column='output'
                    )
                    if processed:
                        self.samples.append(processed)
                    else:
                        skipped_count += 1
                except json.JSONDecodeError:
                    # Reduce verbosity
                    if skipped_count % 1000 == 0:
                        print(f"\nWarning: Skipping invalid JSON on line ~{line_num}: {line[:100]}...")
                    skipped_count += 1
                    continue
                except Exception as e:
                    if skipped_count % 1000 == 0:
                        print(f"\nWarning: Error processing line ~{line_num}: {e}. Line: {line[:100]}...")
                    skipped_count += 1
                    continue

        if skipped_count > 0: print(f"Skipped {skipped_count} invalid samples during JSONL loading.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# <<< NEW: Hugging Face Map-Style Dataset Class >>>
class HuggingFaceMapStyleDataset(Dataset):
    """
    Wraps a Hugging Face dataset (already loaded) for use with hierarchos.
    Handles tokenization and formatting on the fly based on specified columns.
    """
    def __init__(self, hf_dataset, tokenizer, max_length: int, kayla_mode: bool = False,
                 text_column: Optional[str] = None,
                 prompt_column: Optional[str] = None, completion_column: Optional[str] = None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kayla_mode = kayla_mode
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.completion_column = completion_column

        # --- Validate column existence ---
        ds_columns = hf_dataset.column_names
        if self.text_column and self.text_column not in ds_columns:
            raise ValueError(f"Specified text_column '{self.text_column}' not found in dataset columns: {ds_columns}")
        if self.prompt_column and self.prompt_column not in ds_columns:
            raise ValueError(f"Specified prompt_column '{self.prompt_column}' not found in dataset columns: {ds_columns}")
        if self.completion_column and self.completion_column not in ds_columns:
             raise ValueError(f"Specified completion_column '{self.completion_column}' not found in dataset columns: {ds_columns}")
        if not self.text_column and not (self.prompt_column and self.completion_column):
            if 'text' in ds_columns:
                print(f"Warning: No specific columns provided, defaulting to using 'text' column for text completion.")
                self.text_column = 'text' # Default fallback
            else:
                raise ValueError(f"Must specify either --text_column OR (--prompt_column AND --completion_column). Available columns: {ds_columns}")


    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Fetch the raw data entry from the Hugging Face dataset
        raw_item = self.hf_dataset[idx]
        if not isinstance(raw_item, dict):
            print(f"Warning: Unexpected data type from HF dataset at index {idx}: {type(raw_item)}. Skipping.")
            return None # Should return None to be filtered by collate_fn

        # Process the text using the shared utility function
        processed = process_text_sample(
            self.tokenizer, raw_item, self.max_length, self.kayla_mode,
            self.text_column, self.prompt_column, self.completion_column
        )
        return processed # process_text_sample returns None on error


# <<< START: NEW TOP-LEVEL COLLATE FUNCTION >>>
def _collate_fn_dynamic_padding(batch, pad_token_id: int):
    """
    Top-level collate function for map-style datasets, handling dynamic padding.
    MUST be top-level for multiprocessing (num_workers > 0) to work on Windows/spawn.
    """
    # Filter out None items potentially returned by dataset __getitem__ if processing failed
    batch = [item for item in batch if item is not None]
    if not batch: return None # Return None if batch becomes empty

    # Find max length *in this batch* for dynamic padding
    max_len_batch = max(len(item['input_ids']) for item in batch)

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
# <<< END: NEW TOP-LEVEL COLLATE FUNCTION >>>


# <<< Modified original dataloader creator function to handle ANY map-style dataset >>>
def create_map_style_dataloader(
    dataset: Dataset, # Accepts OriginalJSONLDataset or HuggingFaceMapStyleDataset
    batch_size: int,
    pad_token_id: int,
    num_workers: int = 0,
    shuffle: bool = True
):
    """
    Creates a DataLoader for map-style datasets (like OriginalJSONLDataset or HuggingFaceMapStyleDataset),
    handling dynamic padding.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset provided to create_map_style_dataloader is empty or invalid.")

    # <<< FIX: Use functools.partial to create a collate_fn instance >>>
    # This passes the pad_token_id to our new top-level function.
    # This is necessary for multiprocessing (num_workers > 0) to work.
    collate_fn_with_padding = functools.partial(
        _collate_fn_dynamic_padding,
        pad_token_id=pad_token_id
    )

    # DirectML doesn't support pin_memory in the same way as CUDA
    pin_memory = torch.cuda.is_available() and num_workers > 0
    persistent_workers = num_workers > 0
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_padding, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

# <<< END: Dataloader modifications >>>


# --- Quantization & Model Serialization ---
# ... (Quantization code remains unchanged - assuming hierarchos_matmul is available) ...
def get_q_block_size(qtype: str) -> int:
    """Returns the block size for a given quantization type."""
    if qtype in ["INT4", "Q4_0", "Q8_0"]:
        return 32
    elif qtype == "Q2_K":
        return 256
    else:
        raise ValueError(f"Unknown quantization type: {qtype}")

def export_and_quantize_model(output_dir: str, model: nn.Module, tokenizer, qtype: str):
    """Refuse the obsolete monolithic scalar-RWKV exporter."""
    raise RuntimeError(_LEGACY_QUANTIZATION_UNSUPPORTED)
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
        if tensor.ndim == 2 and "emb" not in name and "ltm" not in name and "timestamps" not in name and "sources" not in name and "ln" not in name:
            float_tensor_np = tensor.cpu().float().numpy()
            M, K = float_tensor_np.shape

            pad_cols = 0
            if K % Q_BLOCK_SIZE != 0:
                pad_cols = Q_BLOCK_SIZE - (K % Q_BLOCK_SIZE)
                float_tensor_np = np.pad(float_tensor_np, ((0, 0), (0, pad_cols)), 'constant')

            quantized_data = hierarchos_matmul.quantize(float_tensor_np, qtype)
            quantized_tensors[name] = {
                "quantized": quantized_data,
                "qtype": qtype,
                "original_shape": [M, K],
            }
        # Keep 1D tensors and explicitly excluded layers as raw numpy arrays
        # <<< FIX: Added "ln" to the check below so LayerNorms are saved raw >>>
        elif tensor.ndim >= 1 and ("emb" in name or "norm" in name or "bias" in name or "persistent" in name or "ltm." in name or "ln" in name):
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


        y_np = hierarchos_matmul.matmul_quantized(x_np, self.quantized_w, self.M, self.qtype, device)

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

class QuantizedRWKVCell:
    def __init__(self, n_embd, name_prefix, q_data):
        self.n_embd = n_embd
        self.key = QuantizedLinear(f'{name_prefix}.key', q_data)
        self.value = QuantizedLinear(f'{name_prefix}.value', q_data)
        self.receptance = QuantizedLinear(f'{name_prefix}.receptance', q_data)
        self.output = QuantizedLinear(f'{name_prefix}.output', q_data)
        self.key_cm = QuantizedLinear(f'{name_prefix}.key_cm', q_data)
        self.receptance_cm = QuantizedLinear(f'{name_prefix}.receptance_cm', q_data)
        self.value_cm = QuantizedLinear(f'{name_prefix}.value_cm', q_data)

        def load_raw(name):
            return torch.from_numpy(q_data[f'{name_prefix}.{name}'].item()['raw'])

        self.time_decay = load_raw('time_decay')
        self.time_first = load_raw('time_first')
        self.time_mix_k = load_raw('time_mix_k')
        self.time_mix_v = load_raw('time_mix_v')
        self.time_mix_r = load_raw('time_mix_r')
        self.time_mix_k_cm = load_raw('time_mix_k_cm')
        self.time_mix_r_cm = load_raw('time_mix_r_cm')

        # <<< FIX: Load LayerNorm Weights >>>
        self.ln1_w = load_raw('ln1.weight')
        self.ln1_b = load_raw('ln1.bias')
        self.ln2_w = load_raw('ln2.weight')
        self.ln2_b = load_raw('ln2.bias')

    def __call__(self, x, state, device="cpu"):
        # Move raw tensors to device if needed
        for p in [self.time_decay, self.time_first, self.time_mix_k, self.time_mix_v,
                  self.time_mix_r, self.time_mix_k_cm, self.time_mix_r_cm,
                  self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b]: # <<< Added LNs to list
            if p.device.type != device:
                p.data = p.data.to(device) # Use .data to modify in place safely

        # Capture input for next Time Mixing state (Token Shift)
        x_in = x 

        sx, aa, bb, pp, sx_cm = state.unbind(dim=2)

        # --- Time mixing ---
        # <<< FIX: Apply LayerNorm 1 >>>
        x_norm = F.layer_norm(x, (self.n_embd,), weight=self.ln1_w, bias=self.ln1_b)

        xk = x_norm * self.time_mix_k + sx * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + sx * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + sx * (1 - self.time_mix_r)

        r = torch.sigmoid(self.receptance(xr, device))
        k = self.key(xk, device)
        k = torch.clamp(k, max=60)
        v = self.value(xv, device)

        ww = self.time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = (e1 * aa + e2 * v) / (e1 * bb + e2 + 1e-8)
        
        # Time Mixing Output / Channel Mixing Input
        x = x + self.output(r * wkv, device)
        
        # Capture input for next Channel Mixing state
        x_tm = x 

        ww = pp + self.time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        aa = e1 * aa + e2 * v
        bb = e1 * bb + e2
        pp = p

        # --- Channel mixing ---
        # <<< FIX: Apply LayerNorm 2 >>>
        x_norm2 = F.layer_norm(x, (self.n_embd,), weight=self.ln2_w, bias=self.ln2_b)

        xk = x_norm2 * self.time_mix_k_cm + sx_cm * (1 - self.time_mix_k_cm)
        xr = x_norm2 * self.time_mix_r_cm + sx_cm * (1 - self.time_mix_r_cm)
        
        r = torch.sigmoid(self.receptance_cm(xr, device))
        k = torch.square(torch.relu(self.key_cm(xk, device)))
        x = x + r * self.value_cm(k, device)

        # Update state: [x_in, aa, bb, pp, x_tm]
        # Slot 0 is input to TM, Slot 4 is input to CM
        new_state = torch.stack([x_in, aa, bb, pp, x_tm], dim=2)
        return x, new_state

class RWKVCell(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        decay_speed = torch.arange(0, n_embd) / n_embd
        self.time_decay = nn.Parameter(-5 + 4 * decay_speed) 
        self.time_first = nn.Parameter(torch.ones(n_embd) * 0.5)
        
        curve = torch.arange(0, n_embd) / n_embd
        curve = torch.pow(curve, 0.5) 

        self.time_mix_k = nn.Parameter(curve.view(1, 1, n_embd))
        self.time_mix_v = nn.Parameter(curve.view(1, 1, n_embd) + 0.1 * torch.randn(1, 1, n_embd)) 
        self.time_mix_r = nn.Parameter(0.5 * curve.view(1, 1, n_embd)) 

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.time_mix_k_cm = nn.Parameter(torch.ones(1, 1, n_embd) * 0.05)
        self.time_mix_r_cm = nn.Parameter(torch.ones(1, 1, n_embd) * 0.05)

        self.key_cm = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.receptance_cm = nn.Linear(n_embd, n_embd, bias=False)
        self.value_cm = nn.Linear(n_embd * 4, n_embd, bias=False)

    def forward(self, x, state, timestep=None):
        # Handle torch.compile artifacts
        if x.dim() == 3 and x.shape[0] == 1: x = x.squeeze(0)
        if state.dim() == 4 and state.shape[0] == 1: state = state.squeeze(0)

        # <<< FIX: Conditional truncated BPTT instead of unconditional detachment >>>
        # Only detach at designated truncation boundaries to enable proper temporal learning
        # Detachment frequency is controlled by detach_every_n_steps (default: None means no detachment)
        # This allows gradients to flow through multiple timesteps while preventing infinite accumulation
        detach_every_n_steps = getattr(self, 'detach_every_n_steps', None)
        if self.training and detach_every_n_steps is not None and timestep is not None:
            if timestep > 0 and timestep % detach_every_n_steps == 0:
                state = state.detach()

        # --- FIX: Explicit Residual Structure x = x + Block(LN(x)) ---
        # We save the incoming 'x' (residual) and use normalized 'x_norm' for mixing
        x_resid_tm = x 
        x_norm = self.ln1(x)

        # Capture input for state update (Token Shift uses the normalized input in std RWKV)
        # Note: Some variants use raw x, standard uses x_norm. Keeping consistent with v4/5/6 logic.
        x_in = x_norm 

        tm_k = self.time_mix_k.view(-1)
        tm_v = self.time_mix_v.view(-1)
        tm_r = self.time_mix_r.view(-1)
        tm_k_cm = self.time_mix_k_cm.view(-1)
        tm_r_cm = self.time_mix_r_cm.view(-1)

        # Unbind state. 
        # Slot 0 (sx) is the previous timestep's input
        sx, aa, bb, pp, sx_cm = state.unbind(dim=-1)

        # --- Time mixing ---
        xk = x_norm * tm_k + sx * (1 - tm_k)
        xv = x_norm * tm_v + sx * (1 - tm_v)
        xr = x_norm * tm_r + sx * (1 - tm_r)

        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        
        # --- FIX APPLIED HERE ---
        # Increased clamp max from 30 to 60 (Standard RWKV) to improve dynamic range
        k = torch.clamp(k, max=60) 
        
        v = self.value(xv)

        # WKV Calculation (Float32 Stability)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            k_f, v_f, pp_f, aa_f, bb_f = k.float(), v.float(), pp.float(), aa.float(), bb.float()
            time_first_f = self.time_first.float()
            time_decay_f = self.time_decay.float()

            ww = time_first_f + k_f
            # Clamp ww and pp_f to prevent exp overflow/underflow issues
            ww = torch.clamp(ww, max=30.0)
            pp_f = torch.clamp(pp_f, max=30.0)
            
            p = torch.maximum(pp_f, ww)
            e1 = torch.exp(pp_f - p)
            e2 = torch.exp(ww - p)
            
            wkv = (e1 * aa_f + e2 * v_f) / (e1 * bb_f + e2 + 1e-8)
            wkv = wkv.to(dtype=x.dtype)

            # Update State
            ww = pp_f + time_decay_f
            # Clamp ww again
            ww = torch.clamp(ww, max=30.0)
            
            p = torch.maximum(ww, k_f)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k_f - p)
            aa = (e1 * aa_f + e2 * v_f).to(dtype=x.dtype)
            bb = (e1 * bb_f + e2).to(dtype=x.dtype)
            pp = p.to(dtype=x.dtype)

        # Residual Connection 1 (Time Mixing)
        x = x_resid_tm + self.output(r * wkv)
        
        # --- Channel mixing ---
        x_resid_cm = x
        x_norm2 = self.ln2(x) # Pre-LN for channel mix

        # Use previous x_norm2 (stored in state slot 4) for mixing
        # Slot 4 is sx_cm
        xk = x_norm2 * tm_k_cm + sx_cm * (1 - tm_k_cm)
        xr = x_norm2 * tm_r_cm + sx_cm * (1 - tm_r_cm)
        
        r = torch.sigmoid(self.receptance_cm(xr))
        k = torch.square(torch.relu(self.key_cm(xk)))
        
        # Residual Connection 2 (Channel Mixing)
        x = x_resid_cm + r * self.value_cm(k)

        # Save x_norm (Slot 0) and x_norm2 (Slot 4) for next step
        new_state = torch.stack([x_in, aa, bb, pp, x_norm2], dim=-1)
        
        # <<< FIX: Clamp state values to ensure numerical stability >>>
        new_state = torch.clamp(new_state, min=-50.0, max=50.0)

        return x, new_state
class LTMModule(nn.Module):
    """
    Titans-style Neural Memory Module with Dual-Store Architecture.
    
    MERGED VERSION (V3.2 - Time-Invariant Decay): 
    1. Retains 'Fast State' (Titans) for test-time updates.
    2. Scales forgetting rate based on token count to fix training/inference mismatch.
    """
    # SOURCE_ID definitions
    SRC_UNKNOWN = 0
    SRC_USER_INTERACTION = 1
    SRC_TRAINING_DATA = 2
    SRC_CORRECTION = 3 

    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4, forget_rate=0.01, reference_chunk_len=128):
        super().__init__()
        
        # --- Slow Weights (Long-Term Consolidation) ---
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        
        # Vals represent the consolidated content. 
        vals_init = torch.empty(n_slots, val_dim)
        nn.init.orthogonal_(vals_init)
        self.vals = nn.Parameter(vals_init * 0.02)
        
        # --- Fast State (Associative Working Memory - Titans Feature) ---
        self.register_buffer("fast_vals", torch.zeros(n_slots, val_dim))
        
        # --- Metadata & Optimizer Stats ---
        self.register_buffer("_mom_vals", torch.zeros_like(self.vals.data))
        self.lr, self.momentum, self.weight_decay = lr, momentum, wd
        
        # Base forget rate per REFERENCE chunk size
        self.forget_rate = forget_rate 
        self.reference_chunk_len = reference_chunk_len

        # Buffers for tracking history context
        self.register_buffer("timestamps", torch.zeros(n_slots, dtype=torch.float32))
        self.register_buffer("sources", torch.full((n_slots,), self.SRC_UNKNOWN, dtype=torch.long))

        # Buffer for neg_inf to avoid creation in hot loop
        self.register_buffer("neg_inf", torch.tensor(-float('inf')), persistent=False)

        # <<< FIX: Pre-allocate buffers for update calculations >>>
        self.register_buffer("update_counts", torch.zeros(n_slots, dtype=torch.float32), persistent=False)
        self.register_buffer("update_slots", torch.zeros(n_slots, val_dim, dtype=torch.float32), persistent=False)

        # <<< NEW: Online Learning Accumulator >>>
        self.register_buffer("ltm_deltas", torch.zeros(n_slots, val_dim), persistent=False)
        self.accumulate_deltas = False

    def reset_working_memory(self):
        """Zeros out the Fast State (Working Memory) and associated momentum buffers."""
        # ZLUDA Fix: Avoid in-place fill_ and direct device creation
        device = self.fast_vals.device
        
        # Create zeros on CPU and move to device to avoid ZLUDA cuRAND/creation issues
        zeros_vals = torch.zeros(self.fast_vals.shape, dtype=self.fast_vals.dtype, device='cpu').to(device)
        self.fast_vals.copy_(zeros_vals)
        self._mom_vals.copy_(zeros_vals) # Same shape/dtype
        
        zeros_ts = torch.zeros(self.timestamps.shape, dtype=self.timestamps.dtype, device='cpu').to(device)
        self.timestamps.copy_(zeros_ts)
        
        sources_init = torch.full(self.sources.shape, self.SRC_UNKNOWN, dtype=self.sources.dtype, device='cpu').to(device)
        self.sources.copy_(sources_init)

    def get_effective_memory(self):
        """Returns the combined memory (Slow + Fast)."""
        return self.vals + self.fast_vals

    def retrieve_topk(self, queries: torch.Tensor, topk: int = 4, min_timestamp: float = 0.0, source_filter: Optional[int] = None, fast_vals: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        scale_factor = self.keys.shape[1] ** -0.5
        sim = (queries @ self.keys.t()) * scale_factor

        if min_timestamp > 0.0 or source_filter is not None:
            with torch.no_grad():
                valid_mask = torch.ones(self.keys.size(0), dtype=torch.bool, device=self.keys.device)
                if min_timestamp > 0.0:
                    valid_mask &= (self.timestamps >= min_timestamp)
                if source_filter is not None:
                    valid_mask &= (self.sources == source_filter)

                sim = torch.nan_to_num(sim, nan=-torch.inf, posinf=torch.finfo(sim.dtype).max, neginf=-torch.inf)
                # FIX: Use torch.where instead of in-place indexing to avoid DirectML scatter error
                # Mask out invalid slots (similarity -> -inf)
                # Use pre-registered neg_inf to avoid device sync/creation overhead
                if not hasattr(self, 'neg_inf'):
                     self.register_buffer("neg_inf", torch.tensor(-float('inf'), device=self.vals.device))
                
                sim = torch.where(valid_mask, sim, self.neg_inf)

        num_valid_slots_per_query = sim.isfinite().sum(dim=-1)
        num_valid_slots = num_valid_slots_per_query.min().item()
        effective_topk = min(topk, int(num_valid_slots))

        if effective_topk <= 0:
            query_shape = list(queries.shape)
            vals_shape = query_shape[:-1] + [topk, self.vals.shape[-1]]
            idx_shape = query_shape[:-1] + [topk]
            
            return (torch.zeros(vals_shape, device=queries.device, dtype=self.vals.dtype), 
                    torch.full(idx_shape, -1, device=queries.device, dtype=torch.long), 
                    torch.zeros(idx_shape, device=queries.device, dtype=torch.float32))

        # Get top-k indices and values (sim scores)
        # FIX: Detach sim before topk to avoid 'scatter' in backward pass of topk
        # We will re-gather the values using our safe matmul approach so gradients still flow.
        sim_detached = sim.detach()
        _, idx = torch.topk(sim_detached, k=effective_topk, dim=-1)
        
        # Now gather sim_topk using safe matmul
        # idx: [B, T, K]
        # sim: [B, T, Slots]
        
        # Reuse the one-hot logic we need for vals anyway
        # We can do it once for both sim and vals if we structure it right.
        
        # Use provided fast_vals if available (for differentiable training), else use buffer
        current_fast_vals = fast_vals if fast_vals is not None else self.fast_vals
        effective_memory = self.vals + current_fast_vals
        
        # Clamp indices to valid range (just in case)
        effective_memory_size = self.vals.shape[0] # Slots
        idx_clamped = idx.clamp(min=0, max=effective_memory_size-1)
        
        batch_shape = idx_clamped.shape
        flat_idx = idx_clamped.view(-1)
        
        # Manual one-hot: [..., K, Slots]
        num_classes = effective_memory_size
        
        # FIX: Revert to manual broadcasting for DirectML safety (F.one_hot crashes)
        # Create range tensor with shape [1, ..., 1, Slots] matching idx rank
        range_tensor = torch.arange(num_classes, device=flat_idx.device)
        # Reshape range_tensor to [1] * idx.ndim + [-1]
        view_shape = [1] * idx.ndim + [-1]
        range_tensor = range_tensor.view(*view_shape)
        
        one_hot = (idx_clamped.unsqueeze(-1) == range_tensor).to(dtype=sim.dtype) # [..., K, Slots]
        
        # Gather sim_topk
        # sim: [..., Slots] -> [..., 1, Slots]
        sim_expanded = sim.unsqueeze(-2)
        sim_topk = (one_hot * sim_expanded).sum(dim=-1) # [..., K]
        
        # Gather vals
        # effective_memory: [Slots, ValDim]
        # [..., K, Slots] @ [Slots, ValDim] -> [..., K, ValDim]
        gathered_vals = torch.matmul(one_hot, effective_memory)
        
        # Compute attention weights from similarity scores
        attn_weights = F.softmax(sim_topk, dim=-1)  # [..., effective_topk]
        
        if torch.isnan(attn_weights).any():
            print("WARNING: NaN detected in LTM attention weights!")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Weighted values: [..., effective_topk, val_dim]
        # attn_weights: [..., effective_topk] -> [..., effective_topk, 1]
        # gathered_vals: [..., effective_topk, val_dim]
        weighted_vals = gathered_vals * attn_weights.unsqueeze(-1) # Element-wise multiplication, NO SUM
        
        # Ensure contiguous
        weighted_vals = weighted_vals.contiguous()
        
        # Gather timestamps (No gradients needed here, but use matmul for consistency/safety)
        # timestamps: [Slots] -> [Slots, 1]
        # [..., K, Slots] @ [Slots, 1] -> [..., K, 1]
        flat_ts = torch.matmul(one_hot, self.timestamps.unsqueeze(1)).squeeze(-1)
        ts_retrieved = flat_ts # [..., K]

        # Handle padding if needed
        if effective_topk < topk:
            pad_size = topk - effective_topk
            batch_shape_list = list(batch_shape[:-1])
            
            # Pad indices
            idx_pad = torch.full(batch_shape_list + [pad_size], -1, device=idx.device, dtype=idx.dtype)
            idx_ret = torch.cat([idx, idx_pad], dim=-1)
            
            # Pad weighted values
            vals_pad = torch.zeros(batch_shape_list + [pad_size, weighted_vals.shape[-1]], 
                                   device=weighted_vals.device, dtype=weighted_vals.dtype)
            vals_ret = torch.cat([weighted_vals, vals_pad], dim=-2)
            
            # Pad timestamps
            ts_pad = torch.zeros(batch_shape_list + [pad_size], 
                               device=ts_retrieved.device, dtype=ts_retrieved.dtype)
            ts_ret = torch.cat([ts_retrieved, ts_pad], dim=-1)
            
            return vals_ret, idx_ret, ts_ret
        else:
            return weighted_vals, idx, ts_retrieved

    def inner_update(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor, current_lr: float, timestamp: float, 
                    source: int = SRC_USER_INTERACTION, tokens_covered: int = None,
                    fast_vals: Optional[torch.Tensor] = None, mom_vals: Optional[torch.Tensor] = None,
                    inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Fast Weight Associative Update (Titans Style).
        If inplace=True, modifies fast_vals and mom_vals in-place (if they are provided or self buffers).
        Returns the new state (fast_vals, mom_vals).
        """
        # If no state provided, use buffers (but treat as starting point)
        curr_fast = fast_vals if fast_vals is not None else self.fast_vals
        curr_mom = mom_vals if mom_vals is not None else self._mom_vals
        
        if grads_tensor is None: 
            return curr_fast, curr_mom

        device = self.vals.device

        # Filter out invalid indices
        valid_mask = topk_idx >= 0
        if not valid_mask.any(): 
            return curr_fast, curr_mom

        idx_flat = topk_idx[valid_mask].view(-1)
        if grads_tensor.shape[:-1] != topk_idx.shape: 
            print(f"Warning: grads_tensor shape {grads_tensor.shape[:-1]} mismatch with topk_idx shape {topk_idx.shape}. Skipping LTM update.")
            return curr_fast, curr_mom
        
        grads_flat = grads_tensor[valid_mask].view(-1, self.vals.size(1))
        grads_flat = grads_flat.contiguous()

        # Aggregate gradients using Matmul (safer for DirectML)
        # Manual one-hot: [N, Slots]
        num_classes = self.vals.size(0)
        
        # FIX: Revert to manual broadcasting
        range_tensor = torch.arange(num_classes, device=device).unsqueeze(0)
        one_hot = (idx_flat.unsqueeze(1) == range_tensor).to(dtype=grads_flat.dtype) # [N, Slots]
        
        # Counts: [Slots]
        # Sum over N (dim 0 of one_hot)
        # <<< FIX: Use pre-allocated buffer >>>
        torch.sum(one_hot, dim=0, out=self.update_counts)
        counts = self.update_counts
        
        # Slot grads: [Slots, ValDim]
        # <<< FIX: Don't use pre-allocated buffer with out= - breaks autograd >>>
        slot_grads = torch.matmul(one_hot.t(), grads_flat.to(device))
        
        # FIX: Avoid in-place indexing for normalization
        # slot_grads[nonzero_mask] /= counts[nonzero_mask].unsqueeze(-1)
        
        # Safe division:
        div = counts.unsqueeze(-1)
        # Avoid division by zero by replacing 0 with 1
        div_safe = torch.where(div > 0, div, torch.ones_like(div))
        slot_grads = slot_grads / div_safe
        
        nonzero_mask = counts > 0

        # --- TITANS UPDATE LOGIC (FUNCTIONAL OR IN-PLACE) ---
        # 1. Momentum
        # new_mom = curr_mom * momentum + slot_grads
        if inplace:
            # In-place momentum update: mom = mom * momentum + grads
            curr_mom.mul_(self.momentum).add_(slot_grads)
            # Clamp Momentum
            curr_mom.clamp_(min=-50.0, max=50.0)
            new_mom = curr_mom
        else:
            new_mom = curr_mom * self.momentum + slot_grads
            # Clamp Momentum
            new_mom = torch.clamp(new_mom, min=-50.0, max=50.0)
        
        # 2. Compute Update
        # update_delta = (new_mom + self.weight_decay * curr_fast)
        # update_step = update_delta * (-current_lr)
        
        # 3. Apply Scaled Decay
        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        
        decay_scaler = tokens_covered / float(self.reference_chunk_len)
        retention_rate = (1.0 - self.forget_rate) ** decay_scaler
        
        if inplace:
            # Let's stick to the logic:
            # 1. Compute update_step (dense)
            # update_step = new_mom.add(curr_fast, alpha=self.weight_decay).mul_(-current_lr)
            
            # We can reuse `self.update_slots` for `update_step`.
            # `new_mom` depends on `slot_grads`. We updated `new_mom` (curr_mom) in-place.
            # So `slot_grads` data is already consumed into `curr_mom`.
            # So we can reuse `self.update_slots` for `update_step`.
            
            self.update_slots.copy_(new_mom)
            self.update_slots.add_(curr_fast, alpha=self.weight_decay)
            self.update_slots.mul_(-current_lr)
            
            # 2. Decay curr_fast in-place
            curr_fast.mul_(retention_rate)
            
            # 3. Apply update_step masked
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(curr_fast)
            self.update_slots.masked_fill_(~nonzero_mask_expanded, 0.0)
            
            curr_fast.add_(self.update_slots)
            
            # Clamp
            curr_fast.clamp_(min=-20.0, max=20.0)
            
            new_fast = curr_fast

        else:
            # Original functional logic
            update_delta = (new_mom + self.weight_decay * curr_fast)
            update_step = update_delta * (-current_lr)

            # Apply decay to current state
            decayed_fast = curr_fast * retention_rate

            # 4. Apply update
            update_mask = torch.zeros_like(decayed_fast)
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(update_mask)
            update_mask = torch.where(nonzero_mask_expanded, update_step, update_mask)
            
            new_fast = decayed_fast + update_mask

        # Clamp fast_vals
        if not inplace: # Already clamped if inplace
            new_fast = torch.clamp(new_fast, min=-20.0, max=20.0)

        # --- UPDATE METADATA (Side Effect - still in-place for now as it's not differentiable) ---
        # Timestamps and sources are not part of the gradient graph usually
        with torch.no_grad():
            self.timestamps.data[nonzero_mask] = timestamp
            self.sources.data[nonzero_mask] = source
            
        return new_fast, new_mom

    def update_memory_hebbian(self, topk_idx: torch.LongTensor, keys: torch.Tensor, vals: torch.Tensor, current_lr: float, timestamp: float, 
                              source: int = SRC_USER_INTERACTION, tokens_covered: int = None, fast_vals: Optional[torch.Tensor] = None,
                              mom_vals: Optional[torch.Tensor] = None, inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Hebbian-style associative update, BUT now unified with Titans Momentum logic.
        We treat the 'vals' (which we want to store) as a target, and the 'gradient' is (current - target).
        Actually, simpler: We just treat 'vals' as the negative gradient direction we want to move towards.
        
        Titans Update: M_new = M_old - lr * (Momentum + wd * M_old)
        Momentum_new = Momentum_old * momentum + Gradient
        
        Here, we want to store 'vals'. So we can pretend the gradient is `-vals`.
        Then Momentum accumulates `-vals`.
        And M updates by `-lr * (-vals)` = `+lr * vals`.
        
        This unifies the dynamics: Inference now has momentum!
        """
        # We delegate to inner_update, passing -vals as the gradient.
        # This means "minimize the difference between memory and vals" (roughly).
        # Actually, inner_update does: new_mom = curr_mom * momentum + slot_grads
        # And update = -lr * new_mom
        # So if we want update to be +lr * vals, we need new_mom to be -vals.
        # So we pass -vals as the gradient.
        
        # Invert vals to act as a negative gradient (driving memory TOWARDS vals)
        pseudo_grads = -vals
        
        return self.inner_update(
            topk_idx=topk_idx,
            grads_tensor=pseudo_grads,
            current_lr=current_lr,
            timestamp=timestamp,
            source=source,
            tokens_covered=tokens_covered,
            fast_vals=fast_vals,
            mom_vals=mom_vals,
            inplace=inplace
        )




class DirectMLAdamW(torch.optim.Optimizer):
    """
    A DirectML-optimized AdamW implementation that avoids 'lerp' and 'foreach' operations
    which are known to cause CPU fallbacks and performance issues on DirectML.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DirectMLAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('DirectMLAdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Weight decay
                if weight_decay != 0:
                    # p.data.mul_(1 - lr * weight_decay) # Standard AdamW decay
                    # Use add for safety: p = p - p * lr * wd
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Decay the first and second moment running average coefficient
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # Avoid lerp: exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                step_size = lr / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class WorkerLoop:
    def __init__(self, config, l_rnn, l_input_proj, context_drift_proj, l_to_out):
        self.config = config
        self.l_rnn = l_rnn
        self.l_input_proj = l_input_proj
        self.context_drift_proj = context_drift_proj
        self.l_to_out = l_to_out
        self.max_l_steps = config.max_l_steps
        self.l_conv_atol = config.l_conv_atol
        self.commitment_threshold = config.commitment_threshold

    def __call__(self, enc: torch.Tensor, static_context: torch.Tensor, l_state: torch.Tensor, initial_drift: torch.Tensor, timestep: Optional[int] = None):
        if enc.dim() == 3 and enc.shape[0] == 1: enc = enc.squeeze(0)
        if static_context.dim() == 3 and static_context.shape[0] == 1: static_context = static_context.squeeze(0)
        if l_state.dim() == 4 and l_state.shape[0] == 1: l_state = l_state.squeeze(0)
        if initial_drift.dim() == 3 and initial_drift.shape[0] == 1: initial_drift = initial_drift.squeeze(0)

        # --- FIX: Use calculated initial drift ---
        current_drift = initial_drift
        
        drift_costs = [] 
        current_enc = enc

        # --- FIX: Shadow state for exploration (pondering) ---
        shadow_l_state = l_state.clone()
        
        # --- FIX: Initialize dynamic_context ---
        dynamic_context = static_context + current_drift

        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = self.l_input_proj(l_input_vec)
        # <<< FIX: Clamp l_input for stability >>>
        l_input = torch.clamp(l_input, min=-50.0, max=50.0)
        
        check_idx = [0, 1, 2, 4]

        if not self.l_rnn.training:
            prev_shadow = shadow_l_state.clone()
            for step_idx in range(self.max_l_steps):
                # Use negative timestep for pondering
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1))
                # Safety Clamp
                shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out))
                current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = self.l_input_proj(l_input_vec)
                
                drift_converged = torch.mean(torch.abs(drift_delta)) < self.l_conv_atol
                state_converged = torch.allclose(shadow_l_state[..., check_idx], prev_shadow[..., check_idx], atol=self.l_conv_atol)
                if drift_converged or state_converged: break
                prev_shadow = shadow_l_state.clone()
        else:
            for step_idx in range(self.max_l_steps):
                # Use negative timestep for pondering
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1))
                # Safety Clamp
                shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out))
                current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                drift_sq = torch.sum(current_drift ** 2, dim=-1).mean()
                hinge_cost = torch.relu(drift_sq - self.commitment_threshold)
                # Safety clamp
                hinge_cost = torch.clamp(hinge_cost, max=100.0)
                drift_costs.append(hinge_cost)
                
                # Convergence check
                if torch.mean(torch.abs(drift_delta)) < self.l_conv_atol:
                   break
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = self.l_input_proj(l_input_vec)

        # <<< FIX: Use original l_state (not shadow) for the actual state update >>>
        ts = timestep if timestep is not None else 0
        
        # Recalculate l_input with final drift
        dynamic_context = static_context + current_drift
        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = self.l_input_proj(l_input_vec)
        
        final_l_out, next_l_state = self.l_rnn(l_input, l_state, timestep=ts)
        
        # <<< FIX: Clamp next_l_state >>>
        next_l_state = torch.clamp(next_l_state, min=-50.0, max=50.0)
        
        final_enc = current_enc + self.l_to_out(final_l_out)
        commitment_cost = torch.tensor(0.0, device=enc.device)
        if len(drift_costs) > 0:
            commitment_cost = torch.stack(drift_costs).mean()

        return final_enc, next_l_state, commitment_cost, current_drift

class HierarchosCore(nn.Module):
    """
    FINAL MERGED VERSION (V3):
    - Base: V1 Architecture (Keeps LERP to fix Jitter).
    - Fix: V2 Hinge Loss (Fixes Posterior Collapse/Laziness).
    - Fix: V2 Smart Indices (RWKV Stability).
    - Fix: V1 Shadow State (Prevents memory corruption during pondering).
    - Fix: V3.1 Symmetric LERP (Matches Training to Inference Context Dynamics).
    - Fix: AMP Stability (Force float32 loss)
    - Fix: Restored prepare_inputs_for_generation & QuantizedHierarchos
    - Fix: Encoder Clamping (Prevent Gelu spikes)
    """
    def reset_memory(self):
        """
        Resets the short-term 'fast' associative memory. 
        Must be called between independent training batches.
        """
        if hasattr(self, 'ltm'):
            self.ltm.reset_working_memory()

    def __init__(self, config: dict):
        super().__init__()
        self.config = AttrDict(config)
        required_keys = ['vocab_size', 'context_dim', 'max_length', 'persistent_dim', 
                         'ltm_slots', 'ltm_key_dim', 'ltm_val_dim', 'ltm_lr', 
                         'ltm_topk', 'h_hidden', 'l_hidden', 'max_h_steps', 
                         'max_l_steps', 'l_conv_atol']
        for key in required_keys:
            if key not in self.config and key != 'max_length':
                raise ValueError(f"Missing required configuration key: '{key}'")
            if key == 'max_length' and not self.config.get('max_length'):
                self.config['max_length'] = 1024

        if 'gradient_checkpointing' not in self.config:
            self.config['gradient_checkpointing'] = False
        
        if 'h_stride' not in self.config:
            self.config['h_stride'] = 4 

        if 'commitment_threshold' not in self.config:
            self.config['commitment_threshold'] = 0.05 

        # CPU Optimization: Flush Denormals to Zero (Global Fix)
        if not torch.cuda.is_available():
             torch.set_flush_denormal(True)

        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.context_dim)
        
        self.persistent = nn.Parameter(torch.randn(self.config.persistent_dim) * 0.02)

        # Learnable LTM Gate
        self.ltm_gate_logit = nn.Parameter(torch.tensor(-2.0))

        self.ltm = LTMModule(
            n_slots=self.config.ltm_slots,
            key_dim=self.config.ltm_key_dim,
            val_dim=self.config.ltm_val_dim,
            lr=self.config.ltm_lr,
            forget_rate=getattr(self.config, 'ltm_forget_rate', 0.01) 
        )

        # Sinusoidal Encoding for Timestamps
        half_dim = self.config.ltm_val_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('time_freqs', emb)

        # <<< FIX: Context-Aware LTM Query >>>
        # We concatenate token embedding and context for the query projection
        # This allows the model to retrieve memories relevant to the *situation*, not just the *word*.
        self.qproj = nn.Linear(self.config.context_dim * 2, self.config.ltm_key_dim, bias=False)
        self.val_proj = nn.Linear(self.config.context_dim, self.config.ltm_val_dim, bias=False) # <<< NEW: Project to val dim
        in_dim = self.config.context_dim + self.config.persistent_dim + (self.config.ltm_val_dim * self.config.ltm_topk)
        self.in_proj = nn.Linear(in_dim, self.config.context_dim)
        
        # Project the Worker's state (hidden) back to the Manager's dimensionality
        self.l_feedback_proj = nn.Linear(self.config.l_hidden, self.config.h_hidden, bias=False)
        # Initialize with small weights to introduce feedback gradually
        nn.init.normal_(self.l_feedback_proj.weight, mean=0.0, std=0.01)

        # RWKV cells
        self.h_rnn = RWKVCell(self.config.h_hidden)
        self.h_to_context = nn.Linear(self.config.h_hidden, self.config.context_dim)

        self.l_input_proj = nn.Linear(self.config.context_dim * 2, self.config.l_hidden)
        self.l_rnn = RWKVCell(self.config.l_hidden)
        self.l_to_out = nn.Linear(self.config.l_hidden, self.config.context_dim)

        # Configure truncated BPTT for RWKV cells
        # Detach state every N steps during training to prevent unbounded gradient accumulation
        # while still allowing temporal learning across multiple steps
        detach_freq = getattr(self.config, 'detach_every_n_steps', 32)  # Default: 32 timesteps
        self.h_rnn.detach_every_n_steps = detach_freq
        self.l_rnn.detach_every_n_steps = detach_freq

        # Context Drift Projection
        self.context_drift_proj = nn.Linear(self.config.l_hidden, self.config.context_dim, bias=False)
        nn.init.normal_(self.context_drift_proj.weight, mean=0.0, std=0.01)

        # Worker Loop Module (for JIT)
        self.worker_loop_module = WorkerLoop(
            config=self.config,
            l_rnn=self.l_rnn,
            l_input_proj=self.l_input_proj,
            context_drift_proj=self.context_drift_proj,
            l_to_out=self.l_to_out
        )

        self.h_halt_proj = nn.Linear(self.config.h_hidden, 1)
        self.out_norm = nn.LayerNorm(self.config.context_dim)
        self.lm_head = nn.Linear(self.config.context_dim, self.config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        # Detect device type early for compilation/JIT logic
        config_device = self.config.get('device', 'cpu')
        if isinstance(config_device, torch.device):
            device_type = config_device.type
        else:
            device_type = str(config_device)
            if ':' in device_type: device_type = device_type.split(':')[0]

        # Compilation is now handled explicitly via model.compile() to avoid state_dict mismatch
        if self.config.get('compile', False):
             print("INFO: Model initialized in Eager Mode. Call model.compile() to enable JIT.")
        else:
            print("INFO: torch.compile is DISABLED (Eager Mode).")

        # Keep a reference to the eager method for fallback (if needed)
        # self._worker_loop_eager = self._worker_loop.__wrapped__ if hasattr(self._worker_loop, "__wrapped__") else self._worker_loop




    def _worker_loop(self, enc, static_context, l_state, initial_drift, timestep=None):
        # Delegate to the module (which might be JIT scripted)
        return self.worker_loop_module(enc, static_context, l_state, initial_drift, timestep)

    def compile(self):
        """Applies torch.compile to the worker loop if enabled in config."""
        if not self.config.get('compile', False):
            return

        # Detect device type early for compilation/JIT logic
        config_device = self.config.get('device', 'cpu')
        if isinstance(config_device, torch.device):
            device_type = config_device.type
        else:
            device_type = str(config_device)
            if ':' in device_type: device_type = device_type.split(':')[0]

        # Check for DirectML (doesn't support torch.compile)
        if device_type == 'dml' or is_directml_device(torch.device(device_type) if device_type != 'unknown' else None):
            print("INFO: DirectML detected - torch.compile is not supported.")
            print("      Automatically disabling compilation. Training will use eager mode.")
            self.config['compile'] = False
            return

        # Check for Windows CPU + Compile (Known Hang Issue)
        # Allow override with --force-compile
        if os.name == 'nt' and device_type == 'cpu' and not self.config.get('force_compile', False):
            print("WARNING: torch.compile on Windows CPU is known to hang with complex RNN loops.")
            print("         Disabling compilation for the Worker Loop to ensure stability.")
            print("         Use --force-compile to override this safety check.")
            self.config['compile'] = False
            return

        try:
            if hasattr(torch, "compile"):
                print("INFO: --compile flag set. Applying torch.compile to Worker (L-RNN) loop...")
                
                # Dynamic shapes can cause instability/hangs on Windows CPU with Inductor
                use_dynamic = True
                # Robust check for Windows CPU
                if os.name == 'nt' and (device_type == 'cpu' or device_type == 'unknown'):
                    print(f"INFO: Device type detected as '{device_type}'. Keeping dynamic shapes ENABLED (User Override).")
                    use_dynamic = True

                compile_options = {"triton.cudagraphs": False} if device_type == 'cuda' else {}
                
                # Compile the WorkerLoop module's forward method
                self.worker_loop_module = torch.compile(
                    self.worker_loop_module,
                    dynamic=use_dynamic,
                    fullgraph=False,   
                    options=compile_options
                )
                print("INFO: WorkerLoop compiled successfully.")
        except Exception as e:
            print(f"WARNING: torch.compile failed ({e}) -- falling back to eager mode")

    def compile(self):
        """Applies torch.compile to the worker loop if enabled in config."""
        if not self.config.get('compile', False):
            return

        # Detect device type early for compilation/JIT logic
        config_device = self.config.get('device', 'cpu')
        if isinstance(config_device, torch.device):
            device_type = config_device.type
        else:
            device_type = str(config_device)
            if ':' in device_type: device_type = device_type.split(':')[0]

        # Check for DirectML (doesn't support torch.compile)
        if device_type == 'dml' or is_directml_device(torch.device(device_type) if device_type != 'unknown' else None):
            print("INFO: DirectML detected - torch.compile is not supported.")
            print("      Automatically disabling compilation. Training will use eager mode.")
            self.config['compile'] = False
            return

        # Check for Windows CPU + Compile (Known Hang Issue)
        # Allow override with --force-compile
        if os.name == 'nt' and device_type == 'cpu' and not self.config.get('force_compile', False):
            print("WARNING: torch.compile on Windows CPU is known to hang with complex RNN loops.")
            print("         Disabling compilation for the Worker Loop to ensure stability.")
            print("         Use --force-compile to override this safety check.")
            self.config['compile'] = False
            return

        try:
            if hasattr(torch, "compile"):
                print("INFO: --compile flag set. Applying torch.compile to Worker (L-RNN) loop...")
                
                # Dynamic shapes can cause instability/hangs on Windows CPU with Inductor
                use_dynamic = True
                # Robust check for Windows CPU
                if os.name == 'nt' and (device_type == 'cpu' or device_type == 'unknown'):
                    print(f"INFO: Device type detected as '{device_type}'. Keeping dynamic shapes ENABLED (User Override).")
                    use_dynamic = True

                compile_options = {"triton.cudagraphs": False} if device_type == 'cuda' else {}
                
                # Compile the WorkerLoop module's forward method
                self.worker_loop_module = torch.compile(
                    self.worker_loop_module,
                    dynamic=use_dynamic,
                    fullgraph=False,   
                    options=compile_options
                )
                print("INFO: WorkerLoop compiled successfully.")
        except Exception as e:
            print(f"WARNING: torch.compile failed ({e}) -- falling back to eager mode")


    def forward(self, input_ids, attention_mask=None, labels=None, 
                h_state=None, l_state=None, 
                prev_context=None, target_context=None, # State Persistence
                drift_state=None, # <<< NEW: Persisted Drift State
                ltm_memory_state=None, # <<< NEW: Persisted LTM State (fast_vals, mom_vals)
                global_pos_offset=0, # <<< NEW: For correct Stride/Lerp in inference
                min_timestamp=0.0, source_filter=None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device

        # Reverted to standard embedding for better memory usage
        x = self.tok_emb(input_ids)

        # ==================================================================
        # 1. STATE INITIALIZATION (With Context Recovery)
        # ==================================================================
        if h_state is None:
            h_state = torch.zeros(B, self.config.h_hidden, 5, device=device)
            h_state[:, :, 3] = -1e30   
            
            # Initialize Contexts
            prev_context = torch.zeros(B, self.config.context_dim, device=device)
            target_context = torch.zeros(B, self.config.context_dim, device=device)
        else:
            # FIX: Ensure state is on the correct device (it comes from CPU in training loop)
            h_state = h_state.to(device)
            
            # RECOVERY: If RNN state exists but context states are None (e.g., fresh start from load),
            # derive valid contexts from the hidden state to prevent "teleportation shock".
            if prev_context is None:
                restored_ctx = self.h_to_context(h_state[:, :, 0])
                prev_context = restored_ctx
            else:
                prev_context = prev_context.to(device)
                
            if target_context is None:
                # If we don't have a target, assume the current state IS the target
                restored_ctx = self.h_to_context(h_state[:, :, 0])
                target_context = restored_ctx
            else:
                target_context = target_context.to(device)

        if l_state is None:
            l_state = torch.zeros(B, self.config.l_hidden, 5, device=device)
            l_state[:, :, 3] = -1e30
        else:
            l_state = l_state.to(device)

        # Drift state is now derived from l_state, so we don't need to persist it explicitly
        # but we keep the argument for backward compatibility (it will be ignored/overwritten)


        # Initialize LTM State
        # If None, we start from the module's buffers (but treat them as tensors for the graph)
        # FIX: Now we handle BOTH fast_vals and mom_vals
        if ltm_memory_state is None:
            curr_fast_vals = self.ltm.fast_vals
            curr_mom_vals = self.ltm._mom_vals
        else:
            # ltm_memory_state is now a tuple (fast, mom)
            curr_fast_vals, curr_mom_vals = ltm_memory_state
            # Ensure they are on the correct device
            curr_fast_vals = curr_fast_vals.to(device)
            curr_mom_vals = curr_mom_vals.to(device)


        final_embs = []
        ponder_costs = []
        commitment_costs = []

        stride = self.config.h_stride

        # ==================================================================
        # 2. MAIN TIME LOOP (Training)
        # ==================================================================
        all_topk_vals = []
        all_topk_idx = []
        
        for t in range(T):
            token_x = x[:, t]
            
            # Calculate absolute position for Stride/Lerp
            abs_t = global_pos_offset + t
            
            # --- LTM Retrieval ---
            p = self.persistent.unsqueeze(0).expand(B, -1)
            
            # <<< FIX: Context-Aware Query >>>
            # Concatenate token_x (embedding) and prev_context
            # prev_context is [B, dim], token_x is [B, dim]
            q_in = torch.cat([token_x, prev_context], dim=-1)
            q = torch.clamp(self.qproj(q_in), min=-10, max=10)
            
            # Pass current fast_vals to retrieval
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(q, self.config.ltm_topk, min_timestamp, source_filter, fast_vals=curr_fast_vals)
            
            all_topk_vals.append(topk_vals)
            all_topk_idx.append(topk_idx)
            
            args = topk_ts.unsqueeze(-1) * self.time_freqs.unsqueeze(0).unsqueeze(0)
            pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.config.ltm_val_dim % 2 == 1: pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
            topk_vals = topk_vals + pe
            # FIX: Clamp gate input for DirectML stability
            gate_input = torch.clamp(self.ltm_gate_logit, min=-50.0, max=50.0)
            gate = torch.sigmoid(gate_input)
            gated_vals = topk_vals * gate
            mac_in = torch.cat([token_x, p, gated_vals.view(B, -1)], dim=-1)
            
            enc = F.gelu(self.in_proj(mac_in))
            
            # <<< STABILITY FIX from V2: Clamp encoder input >>>
            # This prevents spikes in embeddings from destabilizing the RWKV cells
            enc = torch.clamp(enc, min=-30.0, max=30.0)

            # ==================================================================
            # 3. HIERARCHICAL MANAGER (Continuous Watch, Strided Plan)
            # ==================================================================
            
            # <<< FIX: Apply Worker Feedback >>>
            # Project the Worker's previous state to the Manager's space
            # l_state shape: [B, l_hidden, 5]. Slot 0 is the hidden state.
            # FIX: Always apply feedback (even at t=0, using initial l_state) to match inference logic
            # and ensure continuity across chunks.
            l_feedback = self.l_feedback_proj(l_state[:, :, 0].to(device))
            enc_with_feedback = enc + l_feedback
            
            # A. REALITY STEP (Continuous)
            # Pass timestep for proper truncated BPTT
            h_out_real, h_state = self.h_rnn(enc_with_feedback, h_state, timestep=t)
            
            # Safety Clamp for H-Out (prevent NaN propagation)
            # FIX: Increased range to ±100.0 for better expressiveness on CPU
            h_out_real = torch.clamp(h_out_real, min=-100.0, max=100.0)
            
            if torch.isnan(h_out_real).any() or torch.isinf(h_out_real).any():
                print(f"WARNING: NaN/Inf detected in h_out_real at step {t}")
            
            step_ponder_cost = torch.tensor(0.0, device=device)
            
            # B. PLANNING STEP (Strided)
            # FIX: Use absolute position for stride check to ensure consistency across chunks/inference
            if abs_t % stride == 0:
                # The old target becomes the new starting point for interpolation
                prev_context = target_context

                # Pondering on Shadow State
                h_step_outputs = [h_out_real]
                halt_logit = self.h_halt_proj(h_out_real).squeeze(-1)
                h_halt_probs = [torch.sigmoid(halt_logit)]
                
                shadow_h_state = h_state.clone()
                current_enc_h = enc_with_feedback  # FIX: Use feedback-enhanced encoding for consistency 

                for step_idx in range(self.config.max_h_steps - 1):
                    if not self.training and h_halt_probs[-1].mean() > getattr(self.config, 'h_halt_thresh', 0.9): break
                    # Pass timestep for pondering steps (use negative to indicate pondering, won't trigger detachment)
                    h_out_ponder, shadow_h_state = self.h_rnn(current_enc_h, shadow_h_state, timestep=-(step_idx+1))
                    halt_logit = self.h_halt_proj(h_out_ponder).squeeze(-1)
                    h_step_outputs.append(h_out_ponder)
                    h_halt_probs.append(torch.sigmoid(halt_logit))

                h_stack = torch.stack(h_step_outputs, dim=0)
                halt_stack = torch.stack(h_halt_probs, dim=0)
                remain = 1.0 - halt_stack
                remain_shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
                # FIX: Removed duplicate remain_shifted computation that was breaking ACT weights
                
                # FIX: Revert to torch.cumprod for efficiency/compile support
                cum_remain = torch.cumprod(remain_shifted, dim=0)
                # cum_remain_list = []
                # curr_val = torch.ones_like(remain_shifted[0])
                # for i in range(remain_shifted.size(0)):
                #     curr_val = curr_val * remain_shifted[i]
                #     cum_remain_list.append(curr_val)
                # cum_remain = torch.stack(cum_remain_list, dim=0)
                
                weights = halt_stack * cum_remain
                remainder = cum_remain[-1] * (1.0 - halt_stack[-1])
                total = weights.sum(dim=0) + remainder + 1e-8
                weights = weights / total.unsqueeze(0)
                remainder = remainder / total
                final_h_out = (weights.unsqueeze(-1) * h_stack).sum(dim=0) + remainder.unsqueeze(-1) * h_stack[-1]
                
                # --- Stability Marker: Check for NaNs/Infs in Manager Output ---
                if torch.isnan(final_h_out).any() or torch.isinf(final_h_out).any():
                    print(f"WARNING: Manager output (final_h_out) contains NaNs/Infs at step {abs_t}. Clamping for stability.")
                    final_h_out = torch.nan_to_num(final_h_out, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Update Target Context
                target_context = self.h_to_context(final_h_out)
                # <<< FIX: Clamp target_context for stability >>>
                target_context = torch.clamp(target_context, min=-50.0, max=50.0)
                
                step_ponder_cost = cum_remain.sum(dim=0).mean()
                ponder_costs.append(step_ponder_cost)
            
            # C. LERP (Interpolation) - Matches Inference exactly
            step_in_stride = abs_t % stride
            alpha = step_in_stride / float(stride)
            # <<< RESTORED: torch.lerp >>>
            # Formula: start + weight * (end - start)
            sliding_context = torch.lerp(prev_context, target_context, alpha)
            # sliding_context = prev_context + alpha * (target_context - prev_context)

            # ==================================================================
            # 4. WORKER STEP (Receives Sliding Context)
            # ==================================================================
            
            # REMOVED: Pre-calculation of drift. Now handled inside _worker_loop.
            # dynamic_context = sliding_context + drift 

            # B. PONDER STEP (Worker Loop)
            # ------------------------------------------------------------------
            # The Worker iterates to find the best 'drift' (context adjustment)
            # to satisfy the Manager's goal.
            
            # --- FIX: Calculate Initial Drift from Previous State (Matching Inference) ---
            if self.context_drift_proj is not None:
                # l_state shape: [B, l_hidden, 5]. Slot 0 is the hidden state.
                prev_worker_h = l_state[:, :, 0].to(device) # <<< FIX: Ensure same device
                initial_drift = torch.tanh(self.context_drift_proj(prev_worker_h))
                initial_drift = torch.clamp(initial_drift, min=-5.0, max=5.0)
            else:
                initial_drift = torch.zeros(B, self.config.context_dim, device=device)

            # --- FIX: Explicit Detachment for L-RNN (Prevent Recompilation) ---
            # We handle detachment here instead of inside _worker_loop to avoid passing 
            # 'timestep' as an argument, which causes torch.compile to recompile for every step.
            detach_freq = getattr(self.config, 'detach_every_n_steps', 32)
            if self.training and detach_freq is not None and t > 0 and t % detach_freq == 0:
                 l_state = l_state.detach()

            if self.config.gradient_checkpointing and self.training:
                # Pass None for timestep to avoid specialization
                # Checkpointing the module call
                enc, l_state, cc, final_drift = checkpoint(self.worker_loop_module, enc, sliding_context, l_state, initial_drift, None, use_reentrant=False)
            else:
                # Use the module directly (it might be compiled)
                try:
                    # Pass timestep=None to avoid recompilation loop
                    enc, l_state, cc, final_drift = self.worker_loop_module(enc, sliding_context, l_state, initial_drift, timestep=None)
                    
                    # Robustness: If compiled loop produces NaNs or Infs, treat as failure and fallback
                    # Note: We can't easily fallback if the module itself is compiled in-place.
                    # But we can check for NaNs.
                    if self.config.get('compile', False) and (
                        torch.isnan(enc).any() or torch.isinf(enc).any() or 
                        torch.isnan(l_state).any() or torch.isinf(l_state).any() or
                        torch.isnan(final_drift).any() or torch.isinf(final_drift).any() or
                        torch.isnan(cc).any() or torch.isinf(cc).any()
                    ):
                         # If we detect NaNs, we might want to raise error or warn.
                         # Fallback is harder with compiled module unless we keep an eager copy.
                         # For now, just warn.
                         # print("WARNING: NaNs/Infs detected in compiled worker loop output")
                         pass
                except Exception as e:
                     raise e
            
            # <<< REVISION: Unconditional detachment REMOVED >>>
            # We now rely on the explicit detachment logic above.
            
            # Safety Clamp for Drift State
            # drift_state = torch.clamp(drift_state, min=-5.0, max=5.0) # No longer needed as final_drift is clamped inside loop


            final_embs.append(enc)
            commitment_costs.append(cc)

            # ==================================================================
            # 5. MEMORY UPDATE (Differentiable Hebbian)
            # ==================================================================
            # We update the memory using the current encoding as the "value" to associate
            # with the retrieved keys (topk_idx).
            # This allows the model to "remember" this step for future retrieval.
            if self.training:
                # We use the Hebbian update which is differentiable
                # Project enc to val_dim
                val_to_store = self.val_proj(enc)
                
                # <<< FIX: Clamp val_to_store to prevent memory saturation/instability >>>
                val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                
                # Expand to [B, topk, dim] to match topk_idx
                val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    topk_idx, None, val_expanded, # val is the value to store
                    current_lr=self.config.ltm_lr,
                    timestamp=float(abs_t), # <<< FIX: Use absolute position as timestamp
                    tokens_covered=1,
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals
                )
            else:
                # <<< FIX: Enable LTM Updates during Inference (Hebbian) >>>
                # Project enc to val_dim
                val_to_store = self.val_proj(enc)
                val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                
                # Update memory in-place (or return new state)
                # FIX: Use inplace=True during inference to persist memory in the module buffers
                do_inplace = not self.training
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    topk_idx, None, val_expanded,
                    current_lr=self.config.ltm_lr,
                    timestamp=0.0,
                    tokens_covered=1, # <<< FIX: Single token step
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals,
                    inplace=do_inplace
                )

        # ==================================================================
        # 5. FINAL OUTPUTS
        # ==================================================================
        final = self.out_norm(torch.stack(final_embs, dim=1))
        logits = self.lm_head(final)
        
        # Safety Clamp for Logits to prevent NaN loss
        # 30.0 is usually enough for softmax (exp(30) is huge)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
             print("WARNING: NaN/Inf detected in logits. Replacing with zeros and clamping...")
             logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        loss = None
        ponder_cost_out = None
        commitment_cost_out = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Check for valid labels to prevent NaN loss on masked chunks
            valid_mask = shift_labels != -100
            if not valid_mask.any():
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # ==============================================================
                # AMP STABILITY FIX: Force Float32 for Loss Calculation
                # ==============================================================
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size).float(), 
                    shift_labels.view(-1)
                )
            
            if ponder_costs: ponder_cost_out = torch.stack(ponder_costs).mean()
            if commitment_costs: commitment_cost_out = torch.stack(commitment_costs).mean()

        return {
            "loss": loss, 
            "logits": logits, 
            "ponder_cost": ponder_cost_out, 
            "commitment_cost": commitment_cost_out,
            "topk_vals": torch.stack(all_topk_vals, dim=1) if all_topk_vals else None, 
            "raw_topk_vals": all_topk_vals, # <<< FIX: Return raw list for gradient capture
            "topk_idx": torch.stack(all_topk_idx, dim=1) if all_topk_idx else None,
            "h_state": h_state,
            "l_state": l_state,
            # RETURN CONTEXT STATES FOR TBPTT
            "prev_context": prev_context,
            "target_context": target_context,
            "drift_state": final_drift, # Return new drift state for analysis/verification
            "ltm_memory_state": (curr_fast_vals, curr_mom_vals), # FIX: Return tuple (fast, mom)
        }

    # ==================================================================
    # RESTORED METHOD: Needed for Hugging Face Generate
    # ==================================================================
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        if "attention_mask" in kwargs:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        return model_inputs

    def update_memory(self, topk_idx: torch.LongTensor, grads: torch.Tensor, timestamp: float, lr: float = 1e-3):
        """
        Updates the LTM memory using gradients (Titans style).
        Requires external gradient computation (e.g. via Shadow Model).
        """
        # <<< FIX: Persist the updated state (Optimized In-Place) >>>
        new_fast, new_mom = self.ltm.inner_update(topk_idx, grads, current_lr=lr, timestamp=timestamp, source=LTMModule.SRC_USER_INTERACTION, inplace=True)
        # self.ltm.fast_vals.copy_(new_fast)
        # self.ltm._mom_vals.copy_(new_mom)

    def update_memory_hebbian(self, topk_idx: torch.LongTensor, vals: torch.Tensor, timestamp: float, lr: float = 1e-3, tokens_covered: int = 1):
        """
        Updates the LTM memory using Hebbian rule (Fallback for Inference).
        """
        # <<< FIX: Persist the updated state (Optimized In-Place) >>>
        new_fast, new_mom = self.ltm.update_memory_hebbian(topk_idx, None, vals, current_lr=lr, timestamp=timestamp, source=LTMModule.SRC_USER_INTERACTION, tokens_covered=tokens_covered, inplace=True)
        # self.ltm.fast_vals.copy_(new_fast) # No longer needed with inplace=True

# ==================================================================
# <<< RESTORED CLASS: QuantizedHierarchos for Inference >>>
# ==================================================================
class QuantizedHierarchos:
    """Obsolete monolithic quantized model; construction is disabled."""
    def __init__(self, config: dict, q_data: dict):
        raise RuntimeError(_LEGACY_QUANTIZATION_UNSUPPORTED)
        if not _HAS_KERNEL:
            raise ImportError("Cannot initialize QuantizedHierarchos: C++ kernel not found.")
        
        # Check for DirectML usage in config or environment (if possible to detect here)
        # We can't easily check 'device' here as it's passed in __call__, but we can warn/error if we know.
        # However, the user might load it on CPU.
        # But if they try to use it with DirectML later, it will fail.
        # We'll add a check in __call__ or just rely on the user not passing 'dml' device.
        # Actually, let's check if the global device is DirectML if we can, but this class is device-agnostic until call.
        pass # Logic handled in __call__ or via explicit device checks elsewhere.
        
        self.config = AttrDict(config)
        
        if 'h_stride' not in self.config:
            self.config['h_stride'] = 4 

        if 'l_conv_atol' not in self.config:
            self.config['l_conv_atol'] = 1e-4

        self.qtype = None

        # Load raw (non-quantized) parameters first
        try:
            self.tok_emb = nn.Embedding.from_pretrained(torch.from_numpy(q_data['tok_emb.weight'].item()['raw']))
            
            self.persistent = torch.from_numpy(q_data['persistent'].item()['raw'])
            self.out_norm = nn.LayerNorm(self.config.context_dim)
            self.out_norm.load_state_dict({
                'weight': torch.from_numpy(q_data['out_norm.weight'].item()['raw']),
                'bias': torch.from_numpy(q_data['out_norm.bias'].item()['raw'])
            })
            
            # Initialize LTM with new Dual-Store structure
            self.ltm = LTMModule(n_slots=self.config.ltm_slots,
                                 key_dim=self.config.ltm_key_dim,
                                 val_dim=self.config.ltm_val_dim)
                                 
            # LTM state loading - Adjusted for Dual Store
            ltm_state = {}
            for k in ['ltm.keys', 'ltm.vals', 'ltm.timestamps', 'ltm.sources']:
                if k in q_data:
                    # Strip prefix 'ltm.'
                    key_name = k.split('.', 1)[1]
                    ltm_state[key_name] = torch.from_numpy(q_data[k].item()['raw'])
            
            self.ltm.load_state_dict(ltm_state, strict=False)
            
            # Setup Time Frequencies for Inference
            half_dim = self.config.ltm_val_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            self.time_freqs = emb 

        except Exception as e:
            raise RuntimeError(f"Error loading raw parameters: {e}")

        # Load LTM Gate Logit (with fallback for backward compatibility)
        if 'ltm_gate_logit' in q_data:
            self.ltm_gate_logit = torch.from_numpy(q_data['ltm_gate_logit'].item()['raw'])
        else:
            print("Warning: 'ltm_gate_logit' not found in quantized weights. Initializing to 0.0 (open gate).")
            self.ltm_gate_logit = torch.tensor(0.0) # Scalar 0.0

        # ---------- Quantized RWKV layers ----------
        expected_quantized = [
            'qproj', 'in_proj', 'h_rnn', 'h_to_context',
            'l_input_proj', 'l_rnn', 'l_to_out', 'lm_head', 'h_halt_proj',
            'context_drift_proj', 'l_feedback_proj', 'val_proj' # <<< Added val_proj
        ]
        quantized_layers = {}
        for layer_name in expected_quantized:
            if layer_name in ['h_rnn', 'l_rnn']:
                hidden = self.config.h_hidden if layer_name == 'h_rnn' else self.config.l_hidden
                quantized_layers[layer_name] = QuantizedRWKVCell(hidden, layer_name, q_data)
                if self.qtype is None:
                    self.qtype = quantized_layers[layer_name].key.qtype
            else:
                if f'{layer_name}.weight' in q_data:
                    quantized_layers[layer_name] = QuantizedLinear(layer_name, q_data)
                    if self.qtype is None:
                        self.qtype = quantized_layers[layer_name].qtype
                elif layer_name == 'context_drift_proj':
                    print("Warning: 'context_drift_proj' not found in quantized weights. Initializing random fallback.")
                    quantized_layers[layer_name] = None 

        self.qproj          = quantized_layers['qproj']
        self.in_proj        = quantized_layers['in_proj']
        self.h_rnn          = quantized_layers['h_rnn']
        self.h_to_context   = quantized_layers['h_to_context']
        self.l_input_proj   = quantized_layers['l_input_proj']
        self.l_rnn          = quantized_layers['l_rnn']
        self.l_to_out       = quantized_layers['l_to_out']
        self.lm_head        = quantized_layers['lm_head']
        self.h_halt_proj    = quantized_layers['h_halt_proj']
        self.context_drift_proj = quantized_layers.get('context_drift_proj')
        self.l_feedback_proj = quantized_layers.get('l_feedback_proj') # <<< Added l_feedback_proj
        self.val_proj = quantized_layers.get('val_proj') # <<< Added val_proj
        
        # <<< FIX: Backward Compatibility for Context-Aware Query >>>
        self.use_context_aware_query = False
        if self.qproj.K == self.config.context_dim * 2:
            self.use_context_aware_query = True
            print("INFO: Context-Aware LTM Query detected and enabled.")
        elif self.qproj.K == self.config.context_dim:
            self.use_context_aware_query = False
            print("Warning: Quantized model uses old qproj dimension (token only). Context-aware query disabled.")
        else:
             print(f"Warning: Unexpected qproj dimension {self.qproj.K}. Expected {self.config.context_dim} or {self.config.context_dim * 2}.")

        print(f"Initialized QuantizedHierarchos ({self.qtype}) with RWKV recurrence.")

    def __call__(self, input_ids: torch.LongTensor, 
                 h_state: torch.Tensor, l_state: torch.Tensor, 
                 prev_context: torch.Tensor, target_context: torch.Tensor,
                 global_pos_offset: int = 0,
                 device: str = "cpu", min_timestamp: float = 0.0, source_filter: int = None):
        
        B, T = input_ids.shape
        
        # --- Speed Optimization: Prefill vs Generation ---
        start_t = 0
        if T == 1:
            start_t = 0 
        else:
            start_t = 0
            
        # Initialize return vars
        final_h_state = h_state
        final_l_state = l_state
        
        # Mutable context copies
        # We ensure they are on the correct device (Vulkan/CPU compat)
        # Quantized model does NOT support DirectML (C++ kernel incompatibility).
        if is_directml_device(device) or (isinstance(device, str) and device in ['dml', 'directml']):
             raise ValueError("QuantizedHierarchos does not support DirectML. Please use the full-precision HierarchosCore model.")

        curr_prev_context = prev_context.to(device if device == 'vulkan' else 'cpu')
        curr_target_context = target_context.to(device if device == 'vulkan' else 'cpu')
        
        # Helper for device check
        is_dml = is_directml_device(device) or (isinstance(device, str) and device in ['dml', 'directml', 'privateuseone'])
        
        logits = None
        stride = self.config.h_stride

        all_topk_vals = []
        all_topk_idx = []

        for t in range(start_t, T):
            # Absolute position for Stride/LERP calculations
            abs_t = global_pos_offset + t
            
            token_ids = input_ids[:, t].cpu().long()
            token_emb = self.tok_emb(token_ids) 
            p_read = self.persistent.unsqueeze(0).expand(B, -1)

            # <<< FIX: Context-Aware Query (with Backward Compatibility) >>>
            if self.use_context_aware_query:
                # Concatenate token_emb and curr_prev_context
                # Ensure both are on the same device (Quantized model handles device placement)
                q_in = torch.cat([token_emb.to(device), curr_prev_context.to(device)], dim=-1)
                query = self.qproj(q_in, device=device)
            else:
                # Old behavior: Token only
                query = self.qproj(token_emb.to(device), device=device)
            
            query = torch.clamp(query, min=-10.0, max=10.0)
            
            # --- LTM Retrieval ---
            # <<< FIX: Weighted Retrieval Parity >>>
            # We now expect retrieve_topk to return weighted values directly if we updated LTMModule correctly.
            # Let's check LTMModule.retrieve_topk again.
            # Yes, I updated LTMModule.retrieve_topk to return weighted values.
            # So `topk_vals` here is ALREADY weighted.
            
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(query, topk=self.config.ltm_topk, 
                                                           min_timestamp=min_timestamp, 
                                                           source_filter=source_filter)
            
            all_topk_vals.append(topk_vals)
            all_topk_idx.append(topk_idx)
            
            # Time Encoding
            args = topk_ts.unsqueeze(-1) * self.time_freqs.unsqueeze(0).unsqueeze(0)
            pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.config.ltm_val_dim % 2 == 1:
                 pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
            
            topk_vals = topk_vals + pe
            
            # <<< FIX: Apply LTM Gating (Match HierarchosCore) >>>
            # Ensure gate is on the correct device
            gate_input = torch.clamp(self.ltm_gate_logit.to(device), min=-50.0, max=50.0)
            gate = torch.sigmoid(gate_input)
            topk_vals = topk_vals * gate
            
            ltm_summary = topk_vals.view(B, -1).cpu()

            mac_input = torch.cat([token_emb.cpu(), p_read.cpu(), ltm_summary], dim=-1)
            enc = F.gelu(self.in_proj(mac_input, device=device))

            # ==================================================================
            # 1. HIERARCHICAL MANAGER (Continuous State, Strided Goal)
            # ==================================================================
            
            # <<< FIX: Apply Worker Feedback >>>
            if self.l_feedback_proj is not None:
                l_feedback = self.l_feedback_proj(final_l_state[:, :, 0].to(device), device=device)
                # Clamp feedback to prevent explosion
                l_feedback = torch.clamp(l_feedback, min=-10.0, max=10.0)
                enc_with_feedback = enc + l_feedback
            else:
                enc_with_feedback = enc
            
            # Clamp encoder input to Manager
            enc_with_feedback = torch.clamp(enc_with_feedback, min=-50.0, max=50.0)

            # A. Continuous Update (The V2 Fix)
            h_out_real, new_h_state = self.h_rnn(enc_with_feedback, final_h_state, device=device)
            
            # <<< FIX: Clamp h_out_real to match HierarchosCore (Coherence Fix) >>>
            h_out_real = torch.clamp(h_out_real, min=-100.0, max=100.0)
            
            final_h_state = new_h_state

            # B. Strided Goal Update
            if abs_t % stride == 0:
                curr_prev_context = curr_target_context
                
                # <<< FIX: Manager Pondering (Match Training Logic) >>>
                # Use shadow state to ponder and refine the target context
                h_step_outputs = [h_out_real]
                halt_logit = self.h_halt_proj(h_out_real, device=device).squeeze(-1)
                h_halt_probs = [torch.sigmoid(halt_logit)]
                
                shadow_h_state = new_h_state.clone()
                current_enc_h = enc_with_feedback
                
                for step_idx in range(self.config.max_h_steps - 1):
                    # Early exit for inference
                    if h_halt_probs[-1].mean() > getattr(self.config, 'h_halt_thresh', 0.9): 
                        break
                        
                    # Run H-RNN on shadow state
                    h_out_ponder, shadow_h_state = self.h_rnn(current_enc_h, shadow_h_state, device=device)
                    
                    halt_logit = self.h_halt_proj(h_out_ponder, device=device).squeeze(-1)
                    h_step_outputs.append(h_out_ponder)
                    h_halt_probs.append(torch.sigmoid(halt_logit))

                # Compute Weighted Average (ACT)
                h_stack = torch.stack(h_step_outputs, dim=0)
                halt_stack = torch.stack(h_halt_probs, dim=0)
                
                remain = 1.0 - halt_stack
                remain_shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
                cum_remain = torch.cumprod(remain_shifted, dim=0)
                weights = halt_stack * cum_remain
                remainder = cum_remain[-1] * (1.0 - halt_stack[-1])
                
                total = weights.sum(dim=0) + remainder + 1e-8
                weights = weights / total.unsqueeze(0)
                remainder = remainder / total
                
                final_h_out = (weights.unsqueeze(-1) * h_stack).sum(dim=0) + remainder.unsqueeze(-1) * h_stack[-1]

                # Use the pondered output to set the new target
                curr_target_context = self.h_to_context(final_h_out, device=device)
            
            # C. LERP (Interpolation)
            step_in_stride = abs_t % stride
            alpha = step_in_stride / float(stride)
            # <<< DIRECTML / VULKAN FIX: Explicit Lerp >>>
            # Even though Quantized model is primarily for Vulkan/CPU, explicit math 
            # is safer and often faster than the torch.lerp kernel dispatch overhead.
            diff = curr_target_context - curr_prev_context
            static_context = curr_prev_context + diff * alpha
            if device == 'vulkan':
                 static_context = static_context.to(device)

            # ==================================================================
            # 2. WORKER STEP (Corrected: Iterative Drift/Pondering)
            # ==================================================================
            # Logic now matches training _worker_loop exactly.
            # The model recursively updates the drift to refine the context.
            
            # Initialize drift for this token step (starts at 0)
            # --- FIX: Initialize drift from previous worker state ---
            if self.context_drift_proj is not None:
                prev_worker_h = final_l_state[:, :, 0].to(device)
                current_drift = torch.tanh(self.context_drift_proj(prev_worker_h))
                current_drift = torch.clamp(current_drift, min=-5.0, max=5.0)
            else:
                current_drift = torch.zeros_like(static_context)
            
            # Loop for max_l_steps (Pondering)
            # Loop for max_l_steps (Pondering)
            # <<< FIX: Use Shadow State for Pondering (Match Training Logic) >>>
            shadow_l_state = final_l_state.clone()
            
            for _ in range(self.config.max_l_steps):
                # 1. Calculate Dynamic Context
                dynamic_context = static_context + current_drift
                
                # 2. Prepare Input: [Encoder, Context]
                l_input_raw = torch.cat([enc.cpu(), dynamic_context.cpu()], dim=-1)
                l_input = self.l_input_proj(l_input_raw, device=device)

                # 3. Run Worker RNN Cell on SHADOW state
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, device=device)
                
                # Safety Clamp for Shadow State (Match Training Logic)
                shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
                
                # 4. Calculate Drift Delta (The "Steering")
                if self.context_drift_proj is not None:
                    drift_delta = torch.tanh(self.context_drift_proj(l_out, device=device))
                    current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                    
                    # Convergence Check (Match Training Logic)
                    if torch.mean(torch.abs(drift_delta)) < self.config.l_conv_atol:
                        break
                else:
                    break
            
            # Use the FINAL drift/context to generate the FINAL input
            dynamic_context = static_context + current_drift
            l_input_raw = torch.cat([enc.cpu(), dynamic_context.cpu()], dim=-1)
            l_input = self.l_input_proj(l_input_raw, device=device)
            
            # Update the actual state once
            l_out, final_l_state = self.l_rnn(l_input, final_l_state, device=device)
            
            # Clamp final state (Match Training Logic)
            final_l_state = torch.clamp(final_l_state, min=-50.0, max=50.0)

            # Final projection to output dimension
            enc = enc + self.l_to_out(l_out, device=device)

            final_embedding = self.out_norm(enc.cpu())
            logits = self.lm_head(final_embedding, device=device)

            # ==================================================================
            # 3. MEMORY UPDATE (Inference)
            # ==================================================================
            # Enable LTM Updates during Inference (Hebbian)
            if self.val_proj is not None:
                 # val_proj might be quantized or not.
                 val_to_store = self.val_proj(enc.to(device), device=device) if hasattr(self.val_proj, 'qtype') else self.val_proj(enc.to(device))
                 val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                 val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                 # Update memory using Hebbian rule (inplace)
                 self.update_memory_hebbian(topk_idx, val_expanded, timestamp=float(abs_t), lr=self.config.ltm_lr, tokens_covered=1)

        return {
            "logits": logits.unsqueeze(1) if logits is not None else None,
            "h_state": final_h_state.cpu(),
            "l_state": final_l_state.cpu(),
            "prev_context": curr_prev_context.cpu(),
            "target_context": curr_target_context.cpu(),
            "drift_state": current_drift.cpu(), # <<< FIX: Return drift state
            "topk_vals": torch.stack(all_topk_vals, dim=1).cpu() if all_topk_vals else None,
            "topk_idx": torch.stack(all_topk_idx, dim=1).cpu() if all_topk_idx else None
        }

    def update_memory(self, topk_idx: torch.LongTensor, grads: torch.Tensor, timestamp: float, lr: float = 1e-3):
        """
        Updates the LTM memory using gradients (Titans style).
        Requires external gradient computation (e.g. via Shadow Model).
        """
        # Persist the updated state (Optimized In-Place)
        self.ltm.inner_update(topk_idx, grads, current_lr=lr, timestamp=timestamp, source=LTMModule.SRC_USER_INTERACTION, inplace=True)

    def update_memory_hebbian(self, topk_idx: torch.LongTensor, vals: torch.Tensor, timestamp: float, lr: float = 1e-3, tokens_covered: int = 1):
        """
        Updates the LTM memory using Hebbian rule (Fallback for Inference).
        """
        # Persist the updated state (Optimized In-Place)
        self.ltm.update_memory_hebbian(topk_idx, None, vals, current_lr=lr, timestamp=timestamp, source=LTMModule.SRC_USER_INTERACTION, tokens_covered=tokens_covered, inplace=True)

def load_quantized(model_path: str, device=None):
    """Refuse obsolete monolithic ``.npz`` inference."""
    raise RuntimeError(_LEGACY_QUANTIZATION_UNSUPPORTED)
    # <<< FIX: DirectML Fallback >>>
    if device and is_directml_device(device):
        print("INFO: DirectML detected. Quantized models are incompatible with DirectML.")
        print("      Falling back to full-precision model loading.")
        return load_full_model_with_config(model_path, device)

    if not _HAS_KERNEL:
        raise ImportError("Cannot load quantized model: C++ kernel not found.")

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
        config_dict['model_type'] = 'hierarchos'

    config = AttrDict(config_dict) # Convert to AttrDict

    return Quantizedhierarchos(config, q_data), config # Return both object and AttrDict config


def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model and its config from a directory."""
    weights_path = os.path.join(model_path, MODEL_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{MODEL_WEIGHTS_NAME}' not found in '{model_path}'")

    # Load checkpoint safely, allowing pickles only if necessary (e.g., for optimizer state)
    try:
        # OPTION 2 FIX: Whitelist AttrDict using safe_globals
        # This allows AttrDict to be loaded even with weights_only=True
        with torch.serialization.safe_globals([AttrDict]):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
            
        # Verify config is present
        if 'config' not in checkpoint:
            print("INFO: Config not found in weights_only load. Retrying with weights_only=False.")
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False) # Allow pickles if needed

    except Exception as e: # Changed to 'Exception' to catch _pickle.UnpicklingError
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
        config_dict['model_type'] = 'hierarchos'

    # Ensure vocab_size is present before creating model
    if 'vocab_size' not in config_dict:
        raise ValueError("Cannot initialize model: 'vocab_size' missing from checkpoint config.")

    config = AttrDict(config_dict) # Convert to AttrDict for model init


    model = HierarchosCore(config).to(device) # Pass AttrDict config to model

    # <<< FIX: Handle qproj shape mismatch (Old -> New Architecture) >>>
    # We do this BEFORE loading to avoid "size mismatch" errors/warnings
    state_dict = checkpoint['model_state_dict']
    if 'qproj.weight' in state_dict:
        old_w = state_dict['qproj.weight']
        new_w = model.qproj.weight
        if old_w.shape != new_w.shape:
            print(f"INFO: Adapting qproj.weight from {old_w.shape} to {new_w.shape} for backward compatibility.")
            if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] * 2 == new_w.shape[1]:
                # Case: Output dim same, Input dim doubled (Context-Aware update)
                # Initialize new weight with noise
                adapted_w = torch.randn_like(new_w) * 0.02
                # Copy old weights to first half (Token part)
                adapted_w[:, :old_w.shape[1]] = old_w
                # Update state dict
                state_dict['qproj.weight'] = adapted_w
            else:
                print("Warning: qproj shape mismatch is not a simple doubling. Skipping adaptation (will be re-initialized).")
                del state_dict['qproj.weight'] # Remove to avoid error, let it re-init

    # Load state dict, be flexible with missing/extra keys if needed
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Non-strict state dict loading due to mismatch: {e}. Trying strict=False.")
        model.load_state_dict(state_dict, strict=False)


    return model, config # Return model and AttrDict config


def train(args, device, tokenizer, dataloader, dataloader_len):
    print("Running in TRAIN mode...")
    config = vars(args) # Start with CLI args

    # Explicitly check for DirectML and kill compile flags
    if is_directml_device(device):
        print("INFO: DirectML detected. Forcing --compile to False, disabling AMP, and disabling CUDAGraphs.")
        args.compile = False
        config['compile'] = False
        args.amp = False # Force disable AMP for DirectML stability

    if args.force_compile and not args.compile:
        print("INFO: --force-compile set but --compile missing. Auto-enabling --compile.")
        args.compile = True
        config['compile'] = True

    if args.debug_anomaly:
        print("INFO: Enabling torch.autograd.set_detect_anomaly(True). Expect slower training.")
        torch.autograd.set_detect_anomaly(True)

    # CPU Optimization: Flush Denormals to Zero (Fixes extreme slowness/hangs on CPU)
    if not torch.cuda.is_available():
        print("INFO: CPU Training detected. Enabling torch.set_flush_denormal(True) to prevent performance hangs.")
        torch.set_flush_denormal(True)
        # Optimize threading for CPU
        if hasattr(os, 'cpu_count'):
            num_threads = os.cpu_count()
            # On hyperthreaded systems, physical cores might be better, but let's start with logical
            # or maybe 3/4 of logical to leave room for OS?
            # PyTorch default is usually all cores.
            # Let's set it explicitly to ensure it's not restricted.
            print(f"INFO: Setting intra-op threads to {num_threads}")
            set_threads(num_threads)

    # ==========================================================
    # --- 1. ROBUST CONFIGURATION & RESUME LOGIC (From Old Version) ---
    # ==========================================================
    
    # Ensure train data path or HF dataset name is saved in config
    if args.hf_dataset:
        config['hf_dataset'] = args.hf_dataset
        config['hf_dataset_config'] = args.hf_dataset_config
        config['hf_dataset_split'] = args.hf_dataset_split
    else:
        config['train_data_path'] = args.train
    config['model_type'] = 'hierarchos' # Ensure model_type is set
    
    # Save dataset type flags in config
    config['pre_chunked_dataset'] = args.pre_chunked_dataset
    config['pre_pt_dataset'] = args.pre_pt_dataset
    config['is_hf_dataset'] = bool(args.hf_dataset) # Save flag for HF dataset usage
    
    # Add gradient checkpointing flag to initial config
    config['gradient_checkpointing'] = args.gradient_checkpointing

    # --- Determine vocab_size (already handled during tokenizer load) ---
    current_vocab_size = len(tokenizer) if tokenizer else None

    model = None 
    optimizer = None 
    start_epoch = 0
    model_config = None 
    scaler = None 
    scheduler = None 
    use_amp = args.amp and _HAS_AMP

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
                model_config.vocab_size = current_vocab_size 
                # Re-initialize parts affected by vocab_size if necessary (tok_emb, lm_head)
                model.tok_emb = nn.Embedding(model_config.vocab_size, model_config.context_dim).to(device)
                model.lm_head = nn.Linear(model_config.context_dim, model_config.vocab_size, bias=False).to(device)
                model.tok_emb.weight = model.lm_head.weight 
            elif 'vocab_size' not in model_config:
                raise ValueError("Cannot determine vocab_size: Not found in loaded model config and tokenizer not available.")

            # Ensure max_length from CLI is used if provided, otherwise from loaded config
            if args.max_length and args.max_length != model_config.max_length:
                 print(f"INFO: Overriding loaded model max_length ({model_config.max_length}) with CLI value ({args.max_length}).")
                 model_config.max_length = args.max_length
            elif 'max_length' not in model_config and args.max_length:
                 print(f"INFO: max_length missing from loaded config. Using CLI value ({args.max_length}).")
                 model_config.max_length = args.max_length
            elif 'max_length' not in model_config:
                 print(f"Warning: max_length missing from loaded config and CLI. Using default 1024.")
                 model_config.max_length = 1024

            # Ensure gradient_checkpointing flag from CLI is used
            if args.gradient_checkpointing != model_config.get('gradient_checkpointing', False):
                 print(f"INFO: Overriding loaded model gradient_checkpointing ({model_config.get('gradient_checkpointing', False)}) with CLI value ({args.gradient_checkpointing}).")
                 model_config.gradient_checkpointing = args.gradient_checkpointing
            elif 'gradient_checkpointing' not in model_config:
                 model_config.gradient_checkpointing = args.gradient_checkpointing

            # Update the model's config in case it was modified
            model.config = model_config


            # Initialize optimizer, scaler, scheduler FRESH
            print("INFO: Initializing optimizer, scheduler, and scaler from scratch.")
            
            # Use DirectMLAdamW if on DirectML to avoid CPU fallback
            if is_directml_device(device):
                print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
                optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
            else:
                optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
            if use_amp:
                scaler = GradScaler()
                print("INFO: Automatic Mixed Precision (AMP) ENABLED for training.")

            num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0
            if not args.disable_lr_schedule and num_update_steps > 0:
                print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
                scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
            # <<< FIX: Compile AFTER initialization >>>
            model.compile()
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
            # DirectML workaround: Load to CPU first, then move to device
            # DirectML device objects don't work with map_location parameter
            if is_directml_device(device):
                print("INFO: DirectML detected - loading checkpoint to CPU first, then moving to DirectML...")
                checkpoint = torch.load(args.resume_from_ckpt, map_location='cpu', weights_only=False)
            else:
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

            # Ensure max_length consistency, prioritize CLI arg
            if args.max_length and args.max_length != model_config.max_length:
                 print(f"INFO: Overriding checkpoint max_length ({model_config.max_length}) with CLI value ({args.max_length}).")
                 model_config.max_length = args.max_length
            elif 'max_length' not in model_config and args.max_length:
                 print(f"INFO: max_length missing from checkpoint config. Using CLI value ({args.max_length}).")
                 model_config.max_length = args.max_length
            elif 'max_length' not in model_config:
                 print(f"Warning: max_length missing from checkpoint config and CLI. Using default 1024.")
                 model_config.max_length = 1024

            # Ensure gradient_checkpointing consistency, prioritize CLI arg
            if args.gradient_checkpointing != model_config.get('gradient_checkpointing', False):
                 print(f"INFO: Overriding checkpoint gradient_checkpointing ({model_config.get('gradient_checkpointing', False)}) with CLI value ({args.gradient_checkpointing}).")
                 model_config.gradient_checkpointing = args.gradient_checkpointing
            elif 'gradient_checkpointing' not in model_config:
                 model_config.gradient_checkpointing = args.gradient_checkpointing # Add if missing

            # Ensure model_type is present for HuggingFace compatibility
            if 'model_type' not in model_config:
                model_config['model_type'] = 'hierarchos'

            print("INFO: Re-initializing model architecture from checkpoint config.")
            # Ensure compile flag is set from args, overriding checkpoint
            model_config['compile'] = args.compile
            # DEBUG: Show exact config being used
            print(f"DEBUG: Creating model with h_hidden={model_config.h_hidden}, l_hidden={model_config.l_hidden}, context_dim={model_config.context_dim}")
            model = HierarchosCore(model_config).to(device) 
        else:
            print("Warning: Config not found in checkpoint. Using current CLI args for model architecture.")
            cli_config = config # Use the initial config from vars(args)
            if 'vocab_size' not in cli_config and current_vocab_size:
                cli_config['vocab_size'] = current_vocab_size
            elif 'vocab_size' not in cli_config:
                raise ValueError("Cannot determine vocab_size: Not found in checkpoint or CLI args, and tokenizer not loaded.")
            # Ensure max_length is set in cli_config
            if 'max_length' not in cli_config and args.max_length:
                cli_config['max_length'] = args.max_length
            elif 'max_length' not in cli_config:
                cli_config['max_length'] = 1024 # Default
            # Ensure gradient_checkpointing is set in cli_config
            cli_config['gradient_checkpointing'] = args.gradient_checkpointing

            model_config = AttrDict(cli_config) # Fallback, might cause issues if arch changed
            model = HierarchosCore(model_config).to(device)


        # --- Optimizer Initialization/Loading Logic ---
        initial_lr_for_optim = args.starting_lr if args.override_scheduling else model_config.get('starting_lr', args.starting_lr)
        
        # Use DirectMLAdamW if on DirectML to avoid CPU fallback
        if is_directml_device(device):
            print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
            optimizer = DirectMLAdamW(model.parameters(), lr=initial_lr_for_optim)
        else:
            optimizer = ADAM_OPTIMIZER(model.parameters(), lr=initial_lr_for_optim)

        # <<< FIX: Handle qproj shape mismatch (Old -> New Architecture) >>>
        # We do this BEFORE loading to avoid "size mismatch" errors/warnings
        state_dict = checkpoint['model_state_dict']
        if 'qproj.weight' in state_dict:
            old_w = state_dict['qproj.weight']
            new_w = model.qproj.weight
            if old_w.shape != new_w.shape:
                print(f"INFO: Adapting qproj.weight from {old_w.shape} to {new_w.shape} for backward compatibility.")
                if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] * 2 == new_w.shape[1]:
                    # Case: Output dim same, Input dim doubled (Context-Aware update)
                    # Initialize new weight with noise
                    adapted_w = torch.randn_like(new_w) * 0.02
                    # Copy old weights to first half (Token part)
                    adapted_w[:, :old_w.shape[1]] = old_w
                    # Update state dict
                    state_dict['qproj.weight'] = adapted_w
                else:
                    print("Warning: qproj shape mismatch is not a simple doubling. Skipping adaptation (will be re-initialized).")
                    del state_dict['qproj.weight'] # Remove to avoid error, let it re-init

        # Load model state dict with flexibility
        try:
            # <<< FIX: Pre-handle transient LTM buffers to avoid warnings >>>
            for k in ["ltm.neg_inf", "ltm.update_counts", "ltm.update_slots", "ltm.ltm_deltas"]:
                if k not in state_dict and k in model.state_dict():
                    # Initialize missing buffers if model expects them (e.g. recent version change)
                    state_dict[k] = model.state_dict()[k]
                elif k in state_dict and k not in model.state_dict():
                    # Ignore unexpected buffers from old checkpoints
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Non-strict model state dict loading: {e}. Trying strict=False.")
            model.load_state_dict(state_dict, strict=False)

        # --- Conditional Optimizer State Loading ---
        if not args.override_scheduling:
            try:
                # <<< FIX: Adapt Optimizer State for qproj >>>
                if 'optimizer_state_dict' in checkpoint:
                    optim_state_dict = checkpoint['optimizer_state_dict']
                    if 'state' in optim_state_dict and 'param_groups' in optim_state_dict:
                        try:
                            # Find qproj parameter index in the model
                            qproj_param = model.qproj.weight
                            # Note: This assumes model.parameters() order matches the optimizer's param_groups[0]
                            # which is true for the simple optimizer init used here.
                            param_idx = list(model.parameters()).index(qproj_param)
                            
                            # Get the ID from the saved groups (assuming group 0)
                            if len(optim_state_dict['param_groups']) > 0 and len(optim_state_dict['param_groups'][0]['params']) > param_idx:
                                saved_id = optim_state_dict['param_groups'][0]['params'][param_idx]
                                
                                if saved_id in optim_state_dict['state']:
                                    param_state = optim_state_dict['state'][saved_id]
                                    # Check if adaptation is needed (compare exp_avg shape to current param shape)
                                    if 'exp_avg' in param_state and param_state['exp_avg'].shape != qproj_param.shape:
                                        print(f"INFO: Adapting optimizer state for qproj (ID {saved_id}) from {param_state['exp_avg'].shape} to {qproj_param.shape}")
                                        
                                        # Adapt exp_avg
                                        old_exp = param_state['exp_avg']
                                        new_exp = torch.zeros_like(qproj_param)
                                        if old_exp.shape[0] == new_exp.shape[0] and old_exp.shape[1] * 2 == new_exp.shape[1]:
                                             new_exp[:, :old_exp.shape[1]] = old_exp
                                             param_state['exp_avg'] = new_exp
                                        
                                        # Adapt exp_avg_sq
                                        if 'exp_avg_sq' in param_state:
                                            old_sq = param_state['exp_avg_sq']
                                            new_sq = torch.zeros_like(qproj_param)
                                            if old_sq.shape[0] == new_sq.shape[0] and old_sq.shape[1] * 2 == new_sq.shape[1]:
                                                 new_sq[:, :old_sq.shape[1]] = old_sq
                                                 param_state['exp_avg_sq'] = new_sq
                                        
                                        # If step is tensor, move to cpu? No, usually it's a scalar or tensor on device.
                                        # We don't need to change step.
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not adapt optimizer state for qproj: {e}")

                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Starting optimizer from scratch.")
                # Use DirectMLAdamW if on DirectML to avoid CPU fallback
                if is_directml_device(device):
                    print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer (fallback).")
                    optimizer = DirectMLAdamW(model.parameters(), lr=initial_lr_for_optim)
                else:
                    optimizer = ADAM_OPTIMIZER(model.parameters(), lr=initial_lr_for_optim) # Re-init with correct LR
        else:
            print("INFO: --override-scheduling detected. Skipping loading optimizer state.")

        start_epoch = checkpoint.get('completed_epoch', 0) 
        start_step = checkpoint.get('mid_epoch_step', 0)

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
                    print("!!! WARNING: New LR flags detected but --override-scheduling was not set.              !!!")
                    print(f"!!!  Your new LR ({args.starting_lr}) / Min LR ({args.min_lr}) WILL BE IGNORED.                        !!!")
                    print(f"!!!  Loading old schedule state (LR: {old_lr}, Min LR: {old_min_lr}).                           !!!")
                    print("!!!  To use your new LR flags, add --override-scheduling to your command.              !!!")
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
                print(f"INFO: Setting scheduler last_epoch to {scheduler.last_epoch} based on resumed epoch {start_epoch}.")


        if start_step > 0:
            print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}, step {start_step}.")
        else:
            print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}.")

        # <<< FIX: Compile AFTER loading state dict >>>
        model.compile()


    # <<< Starting completely fresh >>>
    else:
        # Ensure compile flag is set from args
        config['compile'] = args.compile
        
        print("INFO: Starting training from scratch (no --resume-from-ckpt or --model-path provided).")
        if 'vocab_size' not in config:
            if current_vocab_size:
                config['vocab_size'] = current_vocab_size
            else:
                raise ValueError("Cannot determine vocab_size for new model.")
        if 'max_length' not in config or config['max_length'] is None:
            if args.max_length:
                config['max_length'] = args.max_length
            else:
                raise ValueError("max_length not determined for new model (use --max_length or --auto-max-length).")

        model = HierarchosCore(config).to(device)
        
        # Use DirectMLAdamW if on DirectML to avoid CPU fallback
        if is_directml_device(device):
            print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
            optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
        else:
            optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
            
        model_config = AttrDict(config) 

        if use_amp:
            scaler = GradScaler()
            print("INFO: Automatic Mixed Precision (AMP) ENABLED for training.")
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs if dataloader_len > 0 else 0
        if not args.disable_lr_schedule and num_update_steps > 0:
            print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)

    # <<< FIX: Compile AFTER initialization >>>
    model.compile()

    # --- CRITICAL FIX: Correctly placed logic to update LTM reference chunk size ---
    training_chunk_size = getattr(args, 'training_chunk_size', 128)
    print(f"INFO: Training with Temporal Chunk Size: {training_chunk_size}")

    if hasattr(model, 'ltm'):
        if not hasattr(model.ltm, 'reference_chunk_len'):
            model.ltm.reference_chunk_len = training_chunk_size
    
        if model.ltm.reference_chunk_len != training_chunk_size:
            print(f"INFO: Updating LTM reference chunk length from {model.ltm.reference_chunk_len} to {training_chunk_size} to match training config.")
            model.ltm.reference_chunk_len = training_chunk_size
    
    # --- End of Fix ---

    # --- Initialize Step counters ---
    if not args.resume_from_ckpt:
        start_epoch = 0
        start_step = 0

    # ==========================================================
    # --- 2. TRAINING LOOP ---
    # ==========================================================
    model.train()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Zero gradients *before* starting the loop, especially important when resuming
    optimizer.zero_grad(set_to_none=True)

    # Zero gradients *before* starting the loop, especially important when resuming
    optimizer.zero_grad(set_to_none=True)

    global_step = 0 # Track global steps for iterable datasets and scheduler

    # Initialize states to None outside the loop. 
    running_h_state = None
    running_l_state = None
    running_prev_context = None
    running_target_context = None
    running_drift_state = None

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1} / {args.epochs} ---")
        
        # Reset states at epoch start
        running_h_state = None
        running_l_state = None
        running_prev_context = None
        running_target_context = None
        running_drift_state = None
        running_ltm_state = None # <<< NEW: Persist LTM state across chunks
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        total_loss = 0.0
        total_ponder_cost = 0.0
        total_commitment_cost = 0.0 # Track commitment cost for display
        
        # Flag to track if backward was called in the current accumulation cycle
        backward_called_in_cycle = False
        steps_in_epoch = 0 # Track steps for averaging

        for i, batch in enumerate(pbar):
            # Mid-Epoch Resumption: Skip steps already processed
            if epoch == start_epoch and i < start_step:
                if i == start_step - 1:
                    print(f"INFO: Resuming from mid-epoch step {start_step}...")
                continue

            # --- CRITICAL FIX: STATE RESET PER BATCH (Conditional) ---
            # If persist_state is True, we carry over states from the previous batch.
            # This is useful for training on long sequences split into chunks.
            if not getattr(args, 'persist_state', False):
                running_h_state = None
                running_l_state = None
                running_prev_context = None
                running_target_context = None
                running_drift_state = None
                running_ltm_state = None
                
                model.reset_memory()
            else:
                # If persisting, we detach states to prevent backprop through batches (TBPTT limited to batch)
                # But we keep the values.
                if running_h_state is not None: running_h_state = running_h_state.detach()
                if running_l_state is not None: running_l_state = running_l_state.detach()
                if running_prev_context is not None: running_prev_context = running_prev_context.detach()
                if running_target_context is not None: running_target_context = running_target_context.detach()
                if running_drift_state is not None: running_drift_state = running_drift_state.detach()
                if running_ltm_state is not None: 
                    running_ltm_state = (running_ltm_state[0].detach(), running_ltm_state[1].detach())


            # --- CUDAGraphs Protection ---
            if device.type == 'cuda' and not is_directml_device(device):
                try:
                    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()
                except:
                    pass

            if batch is None:
                print(f"Warning: Skipping empty batch at step {i}.")
                continue

            # Load full sequence
            full_input_ids = batch["input_ids"].to(device)
            full_attention_mask = batch["attention_mask"].to(device)
            full_labels = batch["labels"].to(device)
            
            B, T = full_input_ids.shape

           # ==========================================================
            # --- 4. TITANS CHUNKING LOOP ---
            # ==========================================================
            chunk_size = training_chunk_size
            if chunk_size <= 0 or chunk_size > T: chunk_size = T
            
            num_chunks = math.ceil(T / chunk_size)
            
            # --- FIX (From V2): Track Sequence Poisoning ---
            sequence_valid = True 

            for chunk_idx in range(num_chunks):
                
                # --- FIX (From V2): Stop processing if previous chunk was NaN ---
                if not sequence_valid: 
                    break 

                start_t = chunk_idx * chunk_size
                end_t = min((chunk_idx + 1) * chunk_size, T)
                
                # Slice the tensors
                input_ids = full_input_ids[:, start_t:end_t]
                attention_mask = full_attention_mask[:, start_t:end_t]
                labels = full_labels[:, start_t:end_t]

                # --- AMP autocast context ---
                # DirectML devices need to use 'cpu' as device_type for autocast
                autocast_device_type = 'cpu' if is_directml_device(device) else device.type
                with autocast(device_type=autocast_device_type, enabled=use_amp):
                    
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels,
                        h_state=running_h_state, 
                        l_state=running_l_state,
                        prev_context=running_prev_context,
                        target_context=running_target_context,
                        drift_state=running_drift_state,
                        ltm_memory_state=running_ltm_state,
                        global_pos_offset=start_t # <<< FIX: Pass chunk start position
                    )

                    if outputs.get("raw_topk_vals") is not None:
                        for t in outputs["raw_topk_vals"]:
                            if t.requires_grad: t.retain_grad()
                    
                    cross_entropy_loss = outputs["loss"]
                    ponder_cost = outputs["ponder_cost"]
                    commitment_cost = outputs["commitment_cost"]

                    # --- Extract States & DETACH (TBPTT) ---
                    # Added clamping from V1/V2 for stability
                    if outputs.get("h_state") is not None:
                        running_h_state = outputs["h_state"].detach()
                        running_h_state = torch.clamp(running_h_state, min=-50.0, max=50.0)
                    else: running_h_state = None

                    if outputs.get("l_state") is not None:
                        running_l_state = outputs["l_state"].detach()
                        running_l_state = torch.clamp(running_l_state, min=-50.0, max=50.0)
                    else: running_l_state = None

                    if outputs.get("prev_context") is not None:
                        running_prev_context = outputs["prev_context"].detach()
                        running_prev_context = torch.clamp(running_prev_context, min=-50.0, max=50.0)
                    else: running_prev_context = None

                    if outputs.get("target_context") is not None:
                        running_target_context = outputs["target_context"].detach()
                        running_target_context = torch.clamp(running_target_context, min=-50.0, max=50.0)
                    else: running_target_context = None

                    if outputs.get("drift_state") is not None:
                        running_drift_state = outputs["drift_state"].detach()
                        running_drift_state = torch.clamp(running_drift_state, min=-5.0, max=5.0)
                    else: running_drift_state = None

                    if outputs.get("ltm_memory_state") is not None:
                        # FIX: ltm_memory_state is now a tuple (fast, mom)
                        fast, mom = outputs["ltm_memory_state"]
                        running_ltm_state = (fast.detach(), mom.detach())
                    else: running_ltm_state = None

                    # --- Robust Loss Calculation ---
                    combined_loss = None
                    ce_valid = cross_entropy_loss is not None and not torch.isnan(cross_entropy_loss) and not torch.isinf(cross_entropy_loss)
                    pc_valid = ponder_cost is not None and not torch.isnan(ponder_cost) and not torch.isinf(ponder_cost)
                    cc_valid = commitment_cost is not None and not torch.isnan(commitment_cost) and not torch.isinf(commitment_cost)

                    # --- FIX (From V2): Aggressive NaN Handling ---
                    if not ce_valid:
                        print(f"\nWarning: CrossEntropy loss is NaN/Inf at step {i+1} chunk {chunk_idx}. Marking sequence as poisoned.")
                        sequence_valid = False
                        # Reset states so we don't accidentally carry garbage if we were to continue (though we break)
                        running_h_state = None
                        running_l_state = None
                        running_drift_state = None
                        running_ltm_state = None
                        continue 

                    loss_accum = 0.0
                    loss_accum = loss_accum + cross_entropy_loss
                    if pc_valid:
                        loss_accum = loss_accum + (args.ponder_loss_weight * ponder_cost)
                    if cc_valid:
                        loss_accum = loss_accum + (args.commitment_loss_weight * commitment_cost)

                    combined_loss = loss_accum

                # --- Backward Pass (Per Chunk) ---
                if combined_loss is not None:
                    loss_to_backward = combined_loss / args.accumulation_steps

                    if use_amp:
                        scaler.scale(loss_to_backward).backward()
                    else:
                        loss_to_backward.backward()

                    backward_called_in_cycle = True 

                    # ==========================================================
                    # --- 5. IMMEDIATE LTM UPDATE (Safe & Clipped) ---
                    # ==========================================================
                    ltm_grads = None
                    
                    if outputs.get("raw_topk_vals") is not None:
                        grads_list = [t.grad for t in outputs["raw_topk_vals"]]
                        if any(g is not None for g in grads_list):
                             # Replace None with Zeros for stability
                             cleaned_grads = []
                             for i, g in enumerate(grads_list):
                                 if g is None:
                                     cleaned_grads.append(torch.zeros_like(outputs["raw_topk_vals"][i]))
                                 else:
                                     cleaned_grads.append(g)
                             ltm_grads = torch.stack(cleaned_grads, dim=1)

                    if ltm_grads is not None:
                        ltm_grads_copy = ltm_grads.detach().clone()
                        
                        valid_update = True

                        if use_amp:
                            current_scale = scaler.get_scale()
                            if current_scale > 1e-6:
                                ltm_grads_copy = ltm_grads_copy / current_scale
                            else:
                                valid_update = False 

                        if valid_update and torch.isfinite(ltm_grads_copy).all():
                            if args.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_([ltm_grads_copy], args.grad_clip)
                                current_chunk_len = end_t - start_t

                            # <<< FIX: Capture and persist LTM updates (Optimized In-Place) >>>
                            new_fast, new_mom = model.ltm.inner_update(
                                outputs["topk_idx"], 
                                ltm_grads_copy, 
                                current_lr=args.ltm_lr, 
                                source=LTMModule.SRC_TRAINING_DATA,
                                timestamp=float(end_t), # Use end_t as deterministic timestamp
                                tokens_covered=end_t - start_t, # <<< FIX: Pass actual token count for decay scaling
                                inplace=True
                            )
                            # Update running state for next chunk with the Gradient-Updated memory
                            running_ltm_state = (new_fast.detach(), new_mom.detach())
                        else:
                            pass

                    # --- Accumulate Stats ---
                    chunk_ratio = (end_t - start_t) / T
                    if ce_valid: total_loss += cross_entropy_loss.item() * chunk_ratio
                    if pc_valid: total_ponder_cost += ponder_cost.item() * chunk_ratio
                    if cc_valid: total_commitment_cost += commitment_cost.item() * chunk_ratio

            # ==========================================================
            # --- End Chunk Loop ---
            # ==========================================================
            
            steps_in_epoch += 1 # Count step only after processing full sequence

            # --- Optimizer Step (End of Accumulation Cycle) ---
            if (i + 1) % args.accumulation_steps == 0:
                if backward_called_in_cycle:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()

                    if scheduler:
                        scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False
                    global_step += 1 

                    # Periodic Checkpointing (Progress Protection)
                    if args.save_steps > 0 and (i + 1) % args.save_steps == 0:
                        print(f"\n[Step {i+1}] Periodic Checkpoint: Saving to {args.out_dir}...")
                        torch.save({
                            'completed_epoch': epoch, # Not yet finished
                            'mid_epoch_step': i + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'config': dict(model.config),
                        }, os.path.join(args.out_dir, f"hierarchos_epoch_{epoch+1}_step_{i+1}.pt"))

                else:
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss(es) in accumulation cycle.")
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False 

            # --- Update Progress Bar ---
            avg_loss = total_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            avg_ponder = total_ponder_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            avg_commit = total_commitment_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "commit": f"{avg_commit:.2e}", 
                "lr": f"{current_lr:.2e}"
            })

        # --- End of Epoch ---
        ckpt_path = os.path.join(args.out_dir, f"hierarchos_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} complete. Saving training checkpoint to {ckpt_path}")

        # Ensure config saved reflects current state (including potential patches)
        config_to_save = dict(model.config) # Get config directly from model
        config_to_save['starting_lr'] = args.starting_lr # Use current CLI args for these
        config_to_save['min_lr'] = args.min_lr
        config_to_save['disable_lr_schedule'] = args.disable_lr_schedule
        # Save data source info
        if args.hf_dataset:
            config_to_save['hf_dataset'] = args.hf_dataset
            config_to_save['hf_dataset_config'] = args.hf_dataset_config
            config_to_save['hf_dataset_split'] = args.hf_dataset_split
        else:
            config_to_save['train_data_path'] = args.train # Save train path used
        # <<< MODIFIED: Save dataset type flags >>>
        config_to_save['pre_chunked_dataset'] = args.pre_chunked_dataset
        config_to_save['pre_pt_dataset'] = args.pre_pt_dataset
        config_to_save['is_hf_dataset'] = bool(args.hf_dataset)
        config_to_save['kayla'] = args.kayla # Save kayla mode used
        # <<< NEW: Save gradient checkpointing flag >>>
        config_to_save['gradient_checkpointing'] = args.gradient_checkpointing
        # Ensure vocab_size is saved (should be in model.config by now)
        if 'vocab_size' not in config_to_save:
            print(f"CRITICAL WARNING: vocab_size missing from model config before saving epoch {epoch+1} checkpoint!")
        # <<< Ensure max_length is saved >>>
        if 'max_length' not in config_to_save:
             print(f"CRITICAL WARNING: max_length missing from model config before saving epoch {epoch+1} checkpoint!")


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
    # Save data source info
    if args.hf_dataset:
        final_config_to_save['hf_dataset'] = args.hf_dataset
        final_config_to_save['hf_dataset_config'] = args.hf_dataset_config
        final_config_to_save['hf_dataset_split'] = args.hf_dataset_split
    else:
        final_config_to_save['train_data_path'] = args.train
    # <<< MODIFIED: Save dataset type flags >>>
    final_config_to_save['pre_chunked_dataset'] = args.pre_chunked_dataset
    final_config_to_save['pre_pt_dataset'] = args.pre_pt_dataset
    final_config_to_save['is_hf_dataset'] = bool(args.hf_dataset)
    final_config_to_save['kayla'] = args.kayla
    # <<< NEW: Save gradient checkpointing flag >>>
    final_config_to_save['gradient_checkpointing'] = args.gradient_checkpointing
    if 'vocab_size' not in final_config_to_save:
        print(f"CRITICAL WARNING: vocab_size missing from model config before saving final model!")
    if 'max_length' not in final_config_to_save:
        print(f"CRITICAL WARNING: max_length missing from model config before saving final model!")


    torch.save({
        'model_state_dict': model.state_dict(),
        'config': final_config_to_save
    }, final_save_path)

    try:
        if tokenizer:
            tokenizer.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer files on completion. Error: {e}")

    # <<< THIS WAS RESTORED >>>
    if args.quantize_on_complete:
        print("\n--- Training Complete: Starting On-the-Fly Quantization ---")
        # Quantize to a new directory for clarity, e.g., './my_model-INT4'
        quantize_out_dir = args.out_dir.rstrip('/\\') + f"-{args.qtype}"
        quantize(args, device, model, tokenizer, quantize_out_dir)


# <<< FINETUNE Function (modified to accept dataloader) >>>
# <<< FINETUNE Function (modified to accept dataloader and use Commitment Loss) >>>
def finetune(args, device, tokenizer, dataloader, dataloader_len): # <<< Pass dataloader in
    if not _HAS_PEFT: raise ImportError("Please install 'peft' for fine-tuning.")
    print("Running in FINETUNE mode with LoRA...")

    # Load the base model and its config from the specified directory
    model, model_config = load_full_model_with_config(args.model_path, device)

    # <<< Ensure max_length from CLI is used if provided >>>
    if args.max_length and args.max_length != model_config.max_length:
         print(f"INFO: Overriding loaded model max_length ({model_config.max_length}) with CLI value ({args.max_length}) for finetuning.")
         model_config.max_length = args.max_length
         model.pos_emb = nn.Embedding(model_config.max_length, model_config.context_dim).to(device)
    elif 'max_length' not in model_config:
         print(f"Warning: max_length missing from loaded config. Using default 1024 for finetuning.")
         model_config.max_length = 1024 # Or use args.max_length if provided
         model.pos_emb = nn.Embedding(model_config.max_length, model_config.context_dim).to(device)

    # <<< Ensure gradient_checkpointing flag from CLI is used >>>
    if args.gradient_checkpointing != model_config.get('gradient_checkpointing', False):
         print(f"INFO: Overriding loaded model gradient_checkpointing ({model_config.get('gradient_checkpointing', False)}) with CLI value ({args.gradient_checkpointing}) for finetuning.")
         model_config.gradient_checkpointing = args.gradient_checkpointing
    elif 'gradient_checkpointing' not in model_config:
         model_config.gradient_checkpointing = args.gradient_checkpointing # Add if missing

    # <<< NEW: Ensure h_stride flag from CLI is used for the Decoupled Manager >>>
    # This is critical so the H-RNN knows how often to tick during the forward pass.
    if args.h_stride != model_config.get('h_stride', 4):
         print(f"INFO: Overriding loaded model h_stride ({model_config.get('h_stride', 4)}) with CLI value ({args.h_stride}) for finetuning.")
         model_config.h_stride = args.h_stride
    elif 'h_stride' not in model_config:
         model_config.h_stride = args.h_stride

    # Update the model's config with all overrides
    model.config = model_config

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
        target_modules=[
            # RWKV time-mixing
            "key", "value", "receptance", "output",
            # RWKV channel-mixing
            "key_cm", "receptance_cm", "value_cm",
            # Hierarchos-specific layers
            "qproj", "in_proj", "h_to_context",
            "l_input_proj", "l_to_out", "h_halt_proj",
            "context_drift_proj", "l_feedback_proj"  # <<< ADD THIS
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"],  # LTM still updated directly
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Use DirectMLAdamW if on DirectML to avoid CPU fallback
    # Enable DirectMLAdamW for DirectML devices to avoid lerp CPU fallbacks
    if is_directml_device(device):
        print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
        optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
    else:
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
        total_commitment_cost = 0.0 # Track commitment cost
        
        backward_called_in_cycle = False
        steps_in_epoch = 0

        for i, batch in enumerate(pbar):
            if batch is None: continue # Skip empty batches

            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            # <<< FIXED: Added device_type and DirectML support >>>
            # DirectML devices need to use 'cpu' as device_type for autocast
            autocast_device_type = 'cpu' if is_directml_device(device) else device.type
            with autocast(device_type=autocast_device_type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                cross_entropy_loss = outputs["loss"]
                ponder_cost = outputs["ponder_cost"]
                commitment_cost = outputs["commitment_cost"]

                combined_loss = None
                ce_valid = cross_entropy_loss is not None and not torch.isnan(cross_entropy_loss) and not torch.isinf(cross_entropy_loss)
                pc_valid = ponder_cost is not None and not torch.isnan(ponder_cost) and not torch.isinf(ponder_cost)
                cc_valid = commitment_cost is not None and not torch.isnan(commitment_cost) and not torch.isinf(commitment_cost)

                loss_accum = 0.0

                if ce_valid:
                    loss_accum = loss_accum + cross_entropy_loss
                elif i % args.accumulation_steps == 0:
                    print(f"\nWarning: CrossEntropy loss is NaN/Inf at step {i+1}. Skipping backward pass.")

                if pc_valid:
                    loss_accum = loss_accum + (args.ponder_loss_weight * ponder_cost)
                
                # <<< NEW: Commitment Loss >>>
                if cc_valid:
                    loss_accum = loss_accum + (args.commitment_loss_weight * commitment_cost)

                if ce_valid:
                    combined_loss = loss_accum
                else:
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
                if cc_valid: total_commitment_cost += commitment_cost.item()
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

                        # --- CRITICAL FIX: ROBUST AMP UNSCALING AND NAN CHECK ---
                        valid_update = True

                        if use_amp:
                            current_scale = scaler.get_scale()
                            if current_scale > 1e-6 and scaler._enabled and scaler._scale is not None:
                                ltm_grads_copy = ltm_grads_copy / current_scale
                            else:
                                valid_update = False
                                # print(f"\nWarning: Scaler state inconsistent at step {i+1}, skipping LTM update.")

                        # Check for NaNs/Infs before applying
                        if valid_update and torch.isfinite(ltm_grads_copy).all():
                            # Optional: clip LTM grads here too if desired, though they are usually smaller in finetuning
                            if args.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_([ltm_grads_copy], args.grad_clip)

                            base_ltm.inner_update(
                                outputs["topk_idx"], 
                                ltm_grads_copy, 
                                current_lr=args.ltm_lr, 
                                source=LTMModule.SRC_TRAINING_DATA
                            )
                        else:
                            # print("Warning: Skipped LTM update in finetune due to NaN/Inf.")
                            pass

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
            avg_commit = total_commitment_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "commit": f"{avg_commit:.2e}",
                "lr": f"{current_lr:.2e}"
            })

    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    # Note: Only the adapter (+ saved modules like LTM) is saved here.
    # Save tokenizer too for completeness
    try:
        if tokenizer:
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
    raise RuntimeError(_LEGACY_QUANTIZATION_UNSUPPORTED)
    if not _HAS_KERNEL:
        print("ERROR: Cannot quantize model - C++ kernel not found or failed to import.")
        return

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
            source_dir = args.model_path if args.model_path else "./hierarchos_model"
            out_dir = source_dir.rstrip('/\\') + f"-{args.qtype}"
        else:
            out_dir = args.out_dir

    export_and_quantize_model(out_dir, model, tokenizer, qtype=args.qtype)


# <<< MODIFIED: Signal handling setup >>>
_interrupt_flag = False
_original_sigint_handler = None

def _handle_interrupt(sig, frame):
    """Sets the interrupt flag when SIGINT (Ctrl+C) is received."""
    global _interrupt_flag
    if not _interrupt_flag: # Prevent multiple prints if Ctrl+C is held
        print("\n[Interrupt received. Finishing current generation... Press Ctrl+C again to force exit.]", flush=True)
        _interrupt_flag = True
    else:
        # If interrupted again, restore original handler and exit
        print("\n[Forcing exit...]", flush=True)
        if _original_sigint_handler:
            signal.signal(signal.SIGINT, _original_sigint_handler)
        sys.exit(1)

# <<< CHAT Function (Modified) >>>
def is_positive_feedback(text: str) -> bool:
    """Checks if the user input looks like positive validation."""
    text = text.lower().strip()
    positive_triggers = {
        "good", "great", "correct", "yes", "nice", "cool", "perfect", 
        "thanks", "thx", "+", "right", "accurate"
    }
    # Check exact match or starts with (e.g., "Good job")
    if text in positive_triggers:
        return True
    first_word = text.split(' ')[0] if ' ' in text else text
    # Remove punctuation from first word for check (e.g. "Good," -> "good")
    first_word = ''.join(c for c in first_word if c.isalnum())
    return first_word in positive_triggers or text.startswith("/learn")

def is_correction_or_instruction(text: str) -> bool:
    """
    Checks if the user input looks like a correction or new instruction.
    Triggers on negative feedback words followed by content, or explicit instructional phrasing.
    """
    text_lower = text.lower().strip()
    
    # 1. explicit correction indicators
    correction_triggers = ["no", "wrong", "incorrect", "actually", "false", "not true"]
    for trigger in correction_triggers:
        # Check if the input starts with a trigger
        if text_lower.startswith(trigger):
            # It's a correction if it's long enough to contain info (e.g., "No, the answer is 5")
            # "No" by itself isn't useful training data.
            return len(text.split()) > 3 

    # 2. Implicit instruction / statement of fact
    # If the user types a decent amount of text without it being positive feedback, 
    # we assume they are teaching the model something new.
    # Threshold: arbitrary, say > 5 words to avoid noise like "hello there"
    word_count = len(text.split())
    if word_count > 5 and not is_positive_feedback(text):
        return True
        
    return False
def chat(args, device, tokenizer):
    print("Running in CHAT mode...")

    # =================================================================
    # 1. SETUP & SIGNAL HANDLING (From v1)
    # =================================================================
    global _interrupt_flag, _original_sigint_handler
    _interrupt_flag = False # Reset flag at start
    _original_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _handle_interrupt)
    except ValueError as e: # Handle potential issues registering handler (e.g., non-main thread)
        print(f"Warning: Could not set SIGINT handler: {e}. Ctrl+C interrupt may not work gracefully.")
        _original_sigint_handler = None # Ensure we don't try to restore a non-existent handler

    model = None
    shadow_model = None
    config = None
    is_quantized = False
    inference_device = "cpu" # Default for quantized models
    ltm_has_been_updated = False # Flag to track if we need to save
    
    # <<< NEW: Buffer for Anti-Echo-Chamber Logic >>>
    # We store the PREVIOUS turn here. We only train on it if the CURRENT turn is positive.
    pending_training_data = None 

    # =================================================================
    # 2. MODEL LOADING (From v1)
    # =================================================================
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
        # <<< FIX: Pass device to allow DirectML fallback >>>
        # We allow calling load_quantized even if _HAS_KERNEL is False, 
        # because it might fallback to full precision on DirectML.
        try:
            model, config = load_quantized(args.model_path, device=device)
        except ImportError as e:
            print(f"ERROR: {e}")
            return

        if isinstance(model, QuantizedHierarchos):
            is_quantized = True
            print(f"Loaded quantized model with {model.qtype} weights.")
        else:
            print("INFO: Loaded full-precision model (fallback active).")
            is_quantized = False
            inference_device = device

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
                # Ensure shadow model max_length matches quantized model
                if shadow_config.max_length != config.max_length:
                      print(f"Warning: Shadow model max_length ({shadow_config.max_length}) differs from quantized ({config.max_length}). Syncing shadow.")
                      shadow_config.max_length = config.max_length
                      shadow_model.config.max_length = config.max_length
                      shadow_model.pos_emb = nn.Embedding(config.max_length, shadow_model.config.context_dim).to(device)

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

    # If using DirectML, we must use the full model (HierarchosCore) because
    # QuantizedHierarchos depends on the C++ kernel which is CUDA/CPU only (or Vulkan via shader).
    # If the user loaded a quantized model but wants DML, we might have a problem unless
    # we de-quantize or if the kernel supports DML (it doesn't).
    # For now, we assume the user knows what they are doing.
    # But if is_quantized is True and device is DML, we should warn.


    # =================================================================
    # 3. LTM & OPTIMIZER SETUP (From v1)
    # =================================================================
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
        print(f"          - Max LR: {args.ltm_lr:.2e}, Min LR: {args.ltm_schedule_min_lr:.2e}, Cycle Steps: {args.ltm_schedule_steps}")
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
    # DirectML and CUDA can use AMP, but not CPU
    use_amp = args.amp and _HAS_AMP and (not is_quantized or args.enable_quantized_learning) and (device.type == 'cuda' or is_directml_device(device))

    if use_amp:
        scaler = GradScaler()
        # Create a dummy optimizer for the scaler to track state (NaNs/Infs)
        dummy_param_amp = nn.Parameter(torch.tensor(0.0)).to(device) # Needs to be Parameter and on device
        dummy_optimizer = torch.optim.SGD([dummy_param_amp], lr=1.0) # Dummy optimizer for AMP scaler
        print("INFO: Automatic Mixed Precision (AMP) ENABLED for online learning.")

    # --- LOCAL HELPER FOR LTM UPDATE (From v1) ---
    def perform_ltm_update(input_ids_tensor, label_ids_tensor, source_id, penalty=False):
        nonlocal ltm_has_been_updated
        
        update_model = shadow_model if is_quantized else model
        target_device = device

        update_model.train()
        with torch.enable_grad():
            # Reconstruct sequence
            full_sequence = torch.cat([input_ids_tensor, label_ids_tensor], dim=0).unsqueeze(0)
            # Create labels: mask input part with -100, keep label part
            labels = torch.cat([torch.full_like(input_ids_tensor, -100), label_ids_tensor], dim=0).unsqueeze(0)
            
            if full_sequence.shape[1] > config.max_length:
                 full_sequence = full_sequence[:, -config.max_length:]
                 labels = labels[:, -config.max_length:]

            # Zero grads
            if use_amp: dummy_optimizer.zero_grad(set_to_none=True)
            update_model.zero_grad(set_to_none=True)

            # Forward Pass
            combined_loss = None
            ltm_grads = None
            
            with autocast(device_type=target_device.type, enabled=use_amp):
                # <<< FIX: Pass labels=None to bypass internal CrossEntropy computation >>>
                # We need raw logits to calculate Unlikelihood loss manually if penalty=True.
                outputs = update_model(input_ids=full_sequence, labels=None)
                logits = outputs["logits"]
                
                # <<< CRITICAL FIX: Retain gradients for the topk_vals tensor >>>
                # <<< CRITICAL FIX: Retain gradients for the topk_vals tensor >>>
                if outputs.get("raw_topk_vals") is not None:
                    for t in outputs["raw_topk_vals"]:
                         if t.requires_grad: t.retain_grad()

                # --- LOSS CALCULATION START ---
                # 1. Shift logits and labels for autoregression
                # Logits: [B, T, V], Labels: [B, T]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # 2. Flatten
                flat_logits = shift_logits.view(-1, config.vocab_size)
                flat_labels = shift_labels.view(-1)

                # 3. Filter out ignored labels (-100)
                valid_mask = flat_labels != -100
                active_logits = flat_logits[valid_mask]
                active_labels = flat_labels[valid_mask]

                if penalty:
                    # === Unlikelihood Training (for Negative Feedback) ===
                    # Goal: Minimize the probability of the tokens that generated bad output.
                    # L = -log(1 - P(wrong_token))
                    # This creates a bounded gradient that pushes P(wrong) down without exploding.
                    probs = F.softmax(active_logits, dim=-1)
                    
                    # Gather the probabilities of the actual tokens generated (the "wrong" ones)
                    target_probs = torch.gather(probs, 1, active_labels.unsqueeze(1)).squeeze(1)
                    
                    # Clamp for numerical stability (prevent log(0))
                    target_probs = torch.clamp(target_probs, min=1e-7, max=1.0 - 1e-7)
                    
                    loss = -torch.log(1.0 - target_probs).mean()
                else:
                    # === Standard Cross Entropy (for Positive Feedback) ===
                    loss = F.cross_entropy(active_logits, active_labels)
                
                combined_loss = loss
                # --- LOSS CALCULATION END ---

            if combined_loss is not None:
                if use_amp:
                    scaler.scale(combined_loss).backward()
                else:
                    combined_loss.backward()
                
                # Safely get LTM gradients
                # Safely get LTM gradients
                ltm_grads = None
                if outputs.get("raw_topk_vals") is not None:
                    grads_list = [t.grad for t in outputs["raw_topk_vals"]]
                    if any(g is not None for g in grads_list):
                         # Replace None with Zeros for stability
                         cleaned_grads = []
                         for i, g in enumerate(grads_list):
                             if g is None:
                                 cleaned_grads.append(torch.zeros_like(outputs["raw_topk_vals"][i]))
                             else:
                                 cleaned_grads.append(g)
                         ltm_grads = torch.stack(cleaned_grads, dim=1)
                    else:
                        print("DEBUG: All topk_vals.grad are None!")
                        pass

                if ltm_grads is not None:
                    ltm_grads_copy = ltm_grads.detach().clone()
                    
                    # <<< REMOVED: penalty inversion (ltm_grads_copy * -1.0) >>>
                    # Because we used Unlikelihood Loss, the gradients are already
                    # pointing in the correct direction (descent on probability).

                    # Handle LR
                    current_ltm_lr = args.ltm_lr
                    if ltm_scheduler:
                        current_ltm_lr = ltm_scheduler.get_last_lr()[0]
                        ltm_scheduler.step()

                    # Handle AMP Unscale
                    if use_amp:
                        current_scale = scaler.get_scale()
                        if current_scale != 1.0:
                            ltm_grads_copy = ltm_grads_copy / current_scale
                    # --- UPDATE LTM ---
                    # <<< FIX: Capture and persist LTM updates (Optimized In-Place) >>>
                    new_fast, new_mom = update_model.ltm.inner_update(
                        outputs["topk_idx"], 
                        ltm_grads_copy, 
                        current_lr=current_ltm_lr, 
                        timestamp=0.0, # <<< FIX: Add required timestamp
                        source=source_id,
                        tokens_covered=full_sequence.shape[1],
                        inplace=True
                    )
                    # Persist updates to model state
                    # update_model.ltm.fast_vals.copy_(new_fast)
                    # update_model.ltm._mom_vals.copy_(new_mom)
                    ltm_has_been_updated = True
                    
                    
                    # Clean up AMP scaler state
                    if use_amp:
                        scaler.unscale_(dummy_optimizer)
                        scaler.step(dummy_optimizer)
                        scaler.update()
                    
                    update_model.zero_grad(set_to_none=True)
                    
                    # Sync back to quantized model if needed
                    if is_quantized:
                        model.ltm.load_state_dict(update_model.ltm.state_dict())
                    
                    if penalty:
                        print(f" Done. (Unlikelihood | Loss: {loss.item():.3f})")
                    else:
                        print(f" Done. (Reinforced | Loss: {loss.item():.3f})")
                else:
                    print(" (No LTM gradients generated - Model might be frozen or torch.compile is interfering)")
            else:
                print(" (Loss invalid, skipped)")
        
        update_model.eval()

    print("\nWelcome to hierarchos Chat. Type 'exit' or 'quit' to end.")
    print("Commands:")
    print("  /filter time=-<seconds> | /filter source=<id>  : Constrain memory retrieval")
    print("  /settings | /topk <int> | /topp <float>        : View/Change sampling")
    print("  /reset                                         : Clear RNN & Hierarchical states")
    print("  /reset_ltm                                     : Clear LTM memory (fast_vals)")
    print("  /status                                        : Show model state info")
    print("Press Ctrl+C to stop generation at any time.")
    print("="*50)

    try:
        min_ts_filter = 0.0
        source_id_filter = None
        
        # =================================================================
        # 4. STATE INITIALIZATION (Fixed from v2)
        # =================================================================
        # Initialize recurrent states OUTSIDE the loop to fix stuttering/amnesia
        rnn_device = "cpu" if is_quantized else device
        
        # 5-slot state for RWKV cell (assuming specific architecture)
        h_state = torch.zeros(1, config.h_hidden, 5, device=rnn_device) 
        h_state[:, :, 3] = -1e30 # PP init (Critical for stability)
        l_state = torch.zeros(1, config.l_hidden, 5, device=rnn_device) 
        l_state[:, :, 3] = -1e30 # PP init
        
        # Context States (Persisted for Lerp)
        prev_context = torch.zeros(1, config.context_dim, device=rnn_device)
        target_context = torch.zeros(1, config.context_dim, device=rnn_device)
        drift_state = torch.zeros(1, config.context_dim, device=rnn_device)
        
        # Track total token count for correct Stride/Lerp calculation across turns
        total_tokens_generated = 0
        
        while True:
            # <<< MODIFIED: Reset interrupt flag before getting input >>>
            _interrupt_flag = False
            try:
                prompt = input(">>> ")
            except EOFError: # Handle case where input stream ends unexpectedly
                print("\n[EOF detected. Exiting chat.]")
                break

            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit"]:
                break

            # Simple command parser for filtering
            if prompt.startswith('/filter'):
                # ... (Existing filter logic omitted for brevity, assume mostly unchanged) ...
                print("[Filter logic skipped for brevity in update]") 
                continue

            # <<< NEW: Sampling Parameter Tuning >>>
            if prompt.startswith('/temp'):
                try:
                    val = float(prompt.split()[1])
                    args.temperature = max(0.0, val)
                    print(f"Set temperature to {args.temperature}")
                except (IndexError, ValueError):
                    print("Usage: /temp <float>")
                continue

            if prompt.startswith('/topk'):
                try:
                    val = int(prompt.split()[1])
                    args.top_k = max(0, val)
                    print(f"Set top_k to {args.top_k}")
                except (IndexError, ValueError):
                    print("Usage: /topk <int>")
                continue

            if prompt.startswith('/topp'):
                try:
                    val = float(prompt.split()[1])
                    args.top_p = max(0.0, min(1.0, val))
                    print(f"Set top_p to {args.top_p}")
                except (IndexError, ValueError):
                    print("Usage: /topp <float>")
                continue

            if prompt.startswith('/settings'):
                print(f"Current Settings:\n  Temperature: {args.temperature}\n  Top-K: {args.top_k}\n  Top-P: {args.top_p}")
                continue

            if prompt.startswith('/reset_ltm'):
                print("Resetting LTM memory (clearing fast_vals, momentum, timestamps, and sources)...")
                model.ltm.reset_working_memory()
                if is_quantized and shadow_model:
                    shadow_model.ltm.reset_working_memory()
                ltm_has_been_updated = True # Trigger save prompt on exit if they want to persist the reset
                print("LTM Reset complete.")
                continue

            if prompt.startswith('/reset'):
                print("Resetting all RNN and Hierarchical states...")
                h_state.zero_()
                h_state[:, :, 3] = -1e30 # PP init
                l_state.zero_()
                l_state[:, :, 3] = -1e30 # PP init
                prev_context.zero_()
                target_context.zero_()
                drift_state.zero_()
                total_tokens_generated = 0
                print("State Reset complete. Model is now fresh.")
                continue

            if prompt.startswith('/status'):
                print(f"Model Status:")
                print(f"  Total Tokens Generated: {total_tokens_generated}")
                print(f"  Device: {device}")
                print(f"  Quantized: {is_quantized}")
                print(f"  LTM Learning: {'ACTIVE' if learning_enabled else 'OFF'}")
                continue
            
            # =================================================================
            # A. CHECK FOR FEEDBACK & PERFORM UPDATES (Anti-Echo-Chamber)
            # =================================================================
            learning_enabled = not is_quantized or args.enable_quantized_learning
            
            if learning_enabled:
                # 1. Positive Feedback: Reinforce the PREVIOUS turn
                if is_positive_feedback(prompt) and pending_training_data is not None:
                    print("[Positive feedback. Reinforcing previous memory...]", end="", flush=True)
                    perform_ltm_update(
                        pending_training_data['prompt_ids'][0], 
                        pending_training_data['response_ids'],
                        LTMModule.SRC_USER_INTERACTION,
                        penalty=False
                    )
                    pending_training_data = None 
                    continue # Skip generation

                # 2. Negative Feedback: Penalize the PREVIOUS turn
                elif prompt.strip().lower() in ["no", "n", "bad", "wrong", "bad bot"]:
                    if pending_training_data is not None:
                        print("[Negative feedback. Minimizing probability of previous output...]", end="", flush=True)
                        perform_ltm_update(
                            pending_training_data['prompt_ids'][0], 
                            pending_training_data['response_ids'],
                            LTMModule.SRC_USER_INTERACTION,
                            penalty=True  # <--- Triggers Unlikelihood Loss
                        )
                
                # <<< REMOVED: Passive and Correction LTM updates >>>
                # These were redundant with the model's internal Hebbian updates 
                # and caused instability due to high loss on self-prediction tasks.

            # Manual learn commands
            if prompt.strip() == "/learn" and pending_training_data:
                 print("[Manual learn command. Reinforcing previous...]", end="", flush=True)
                 perform_ltm_update(
                    pending_training_data['prompt_ids'][0], 
                    pending_training_data['response_ids'],
                    LTMModule.SRC_USER_INTERACTION,
                    penalty=False
                 )
                 continue 
            elif prompt.strip() == "/learn":
                 print("[Nothing pending to learn]")
                 continue


            # =================================================================
            # B. GENERATION LOGIC (Incremental: Prefill + Single Token Steps)
            # =================================================================
            prompt_format = f"### Instruction:\n{prompt}\n\n### Response:\n"
            if tokenizer is None:
                print("Error: Tokenizer not loaded. Cannot proceed.")
                break
            prompt_ids = tokenizer.encode(prompt_format, return_tensors="pt").to(device)

            print("\nhierarchos: ", end="", flush=True)
            response_ids = []
            
            # 1. PREFILL PASS
            with torch.no_grad():
                model_input_ids = prompt_ids.cpu() if is_quantized else prompt_ids.to(device)
                if is_quantized:
                    outputs = model(
                        input_ids=model_input_ids, 
                        h_state=h_state.cpu(), 
                        l_state=l_state.cpu(), 
                        prev_context=prev_context.cpu(),
                        target_context=target_context.cpu(),
                        drift_state=drift_state.cpu(), # <<< RESTORED
                        global_pos_offset=total_tokens_generated,
                        device=inference_device, 
                        min_timestamp=min_ts_filter, 
                        source_filter=source_id_filter
                    )
                    h_state = outputs['h_state']
                    l_state = outputs['l_state']
                    prev_context = outputs['prev_context']
                    target_context = outputs['target_context']
                    drift_state = outputs['drift_state']
                else:
                    outputs = model(
                        model_input_ids.to(device), 
                        h_state=h_state, 
                        l_state=l_state, 
                        prev_context=prev_context, 
                        target_context=target_context, 
                        drift_state=drift_state, # <<< RESTORED
                        global_pos_offset=total_tokens_generated,
                        min_timestamp=min_ts_filter, 
                        source_filter=source_id_filter
                    )
                    if outputs.get('h_state') is not None: h_state = outputs['h_state']
                    if outputs.get('l_state') is not None: l_state = outputs['l_state']
                    if outputs.get('drift_state') is not None: drift_state = outputs['drift_state']
                    if outputs.get('prev_context') is not None: prev_context = outputs['prev_context']
                    if outputs.get('target_context') is not None: target_context = outputs['target_context']

                logits = outputs["logits"].to(device)
                next_token_logits = logits[:, -1, :]
                
                # Sample first token
                if args.temperature > 0:
                    next_token_logits = next_token_logits / args.temperature
                    if args.top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                    if args.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > args.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                total_tokens_generated += prompt_ids.shape[1]
                
                if next_token_id.item() != tokenizer.eos_token_id:
                    response_ids.append(next_token_id.item())
                    decoded_token = tokenizer.decode([next_token_id.item()])
                    print(decoded_token, end="", flush=True)
                    current_ids = next_token_id # Start incremental loop with this token
                else:
                    current_ids = None

            # 2. INCREMENTAL GENERATION LOOP
            if current_ids is not None:
                with torch.no_grad():
                    for i in range(args.max_new_tokens - 1):
                        if _interrupt_flag:
                            _interrupt_flag = False
                            print("\n[Generation interrupted by user.]", end="", flush=True)
                            break

                        model_input_ids = current_ids.cpu() if is_quantized else current_ids.to(device)

                        if is_quantized:
                            outputs = model(
                                input_ids=model_input_ids, 
                                h_state=h_state.cpu(), 
                                l_state=l_state.cpu(), 
                                prev_context=prev_context.cpu(),
                                target_context=target_context.cpu(),
                                drift_state=drift_state.cpu(), # <<< RESTORED
                                global_pos_offset=total_tokens_generated,
                                device=inference_device, 
                                min_timestamp=min_ts_filter, 
                                source_filter=source_id_filter
                            )
                            h_state = outputs['h_state']
                            l_state = outputs['l_state']
                            prev_context = outputs['prev_context']
                            target_context = outputs['target_context']
                            drift_state = outputs['drift_state']
                        else:
                            outputs = model(
                                model_input_ids.to(device), 
                                h_state=h_state, 
                                l_state=l_state, 
                                prev_context=prev_context, 
                                target_context=target_context, 
                                drift_state=drift_state, # <<< RESTORED
                                global_pos_offset=total_tokens_generated,
                                min_timestamp=min_ts_filter, 
                                source_filter=source_id_filter
                            )
                            if outputs.get('h_state') is not None: h_state = outputs['h_state']
                            if outputs.get('l_state') is not None: l_state = outputs['l_state']
                            if outputs.get('drift_state') is not None: drift_state = outputs['drift_state']
                            if outputs.get('prev_context') is not None: prev_context = outputs['prev_context']
                            if outputs.get('target_context') is not None: target_context = outputs['target_context']

                        logits = outputs["logits"].to(device)
                        next_token_logits = logits[:, -1, :]

                        if args.temperature > 0:
                            next_token_logits = next_token_logits / args.temperature
                            if args.top_k > 0:
                                v, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
                                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                            if args.top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > args.top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                next_token_logits[indices_to_remove] = -float('Inf')
                            
                            probs = F.softmax(next_token_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                        if next_token_id.item() == tokenizer.eos_token_id:
                            break

                        response_ids.append(next_token_id.item())
                        try:
                            decoded_token = tokenizer.decode([next_token_id.item()])
                        except Exception as e:
                            print(f"[Decode Error: {e}]", end="")
                            decoded_token = ""

                        if "###" in decoded_token and len(decoded_token) <= 5:
                            break

                        print(decoded_token, end="", flush=True)
                        current_ids = next_token_id
                        total_tokens_generated += 1

            print("\n")

            # =================================================================
            # C. BUFFER DATA FOR NEXT TURN
            # =================================================================
            if len(response_ids) > 0:
                 pending_training_data = {
                    'prompt_ids': prompt_ids,
                    'response_ids': torch.tensor(response_ids, device=device)
                 }

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected. Exiting chat.]")

    finally:
        # <<< MODIFIED: Restore original signal handler >>>
        if _original_sigint_handler:
            signal.signal(signal.SIGINT, _original_sigint_handler)

        # --- SAVE ON EXIT LOGIC (From v1) ---
        updatable_model = shadow_model if is_quantized and args.enable_quantized_learning else model
        # Check if updatable_model is valid before proceeding
        can_update = updatable_model is not None and (not is_quantized or args.enable_quantized_learning)

        if can_update and args.ltm_lora_path and hasattr(updatable_model.ltm, 'accumulate_deltas') and updatable_model.ltm.accumulate_deltas:
            # Save accumulated deltas if they exist
            if torch.any(updatable_model.ltm.ltm_deltas != 0):
                print(f"\nSaving LTM memory deltas to {args.ltm_lora_path}...")
                try:
                    torch.save(updatable_model.ltm.ltm_deltas.cpu(), args.ltm_lora_path)
                    print("Deltas saved.")
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
                                print("Save complete.")
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
            else:
                # Re-quantize prompt
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
                                print("Re-quantization complete.")
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
             print("\n[Warning] LTM was updated, but no valid save configuration was found. Changes lost.")
def main():
    global _HAS_KERNEL
    parser = argparse.ArgumentParser(description="hierarchos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora"], help="Operation mode.")

    # --- Data and Path Arguments (Universal) ---
    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, nargs='?', default=None, const=True, help="[Train/Finetune] Path to local training data: JSON/JSONL file, or directory for pre-chunked .pt tensors. If used without a path, requires --hf_dataset.")
    path_group.add_argument("--hf_dataset", type=str, default=None, help="[Train/Finetune] Name or path to a Hugging Face dataset (e.g., 'wikitext', 'c4', 'path/to/my_csv/').")
    path_group.add_argument("--hf_dataset_config", type=str, default=None, help="[Train/Finetune] Optional configuration name for the HF dataset (e.g., 'wikitext-103-raw-v1' for wikitext).")
    path_group.add_argument("--hf_dataset_split", type=str, default="train", help="[Train/Finetune] Dataset split to use (e.g., 'train', 'validation', 'train[:10%%]').")
    path_group.add_argument("--text_column", type=str, default=None, help="[Train/Finetune] Column name for text completion data in HF dataset (mutually exclusive with prompt/completion). Defaults to 'text' if available.")
    path_group.add_argument("--prompt_column", type=str, default=None, help="[Train/Finetune] Column name for prompt/instruction in HF dataset.")
    path_group.add_argument("--completion_column", type=str, default=None, help="[Train/Finetune] Column name for completion/response in HF dataset.")
    
    path_group.add_argument("--model-path", type=str, default=None, help="Path to the model directory (required for all modes except 'train' unless resuming or starting from scratch).")
    path_group.add_argument("--out-dir", type=str, default="./hierarchos_model", help="[Train/Finetune/Merge/Quantize] Directory to save the new model/adapter.")
    path_group.add_argument("--lora-adapter-path", type=str, default=None, help="[Merge/Finetune] Path to the LoRA adapter directory.")
    path_group.add_argument("--tokenizer-path", type=str, default=None, help="Path or HF name of the tokenizer (used if not loading from model-path, defaults to openai-community/gpt2).") 
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="[Train] Path to a specific training checkpoint .pt file to resume from.")
    path_group.add_argument("--shadow-model-path", type=str, default=None, help="[Chat] Path to the original full-precision model dir, required for online learning with a quantized model.")

    data_fmt_group = parser.add_mutually_exclusive_group()
    data_fmt_group.add_argument("--pre_chunked_dataset", action="store_true", help="[Train/Finetune] If set, assumes --train points to a pre-tokenized/chunked/padded JSONL (IterableDataset). Requires --max_length.")
    data_fmt_group.add_argument("--pre_pt_dataset", action="store_true", help="[Train/Finetune] If set, assumes --train points to a directory with pre-chunked .pt tensor files and manifest.jsonl (Map-Style Dataset). Requires --max_length.")

    # --- Model Architecture Arguments (for Training) ---
    arch_group = parser.add_argument_group('Architecture (for --mode train, used if not resuming/loading)')
    arch_group.add_argument("--context_dim", type=int, default=768) 
    arch_group.add_argument("--persistent_dim", type=int, default=128)
    arch_group.add_argument("--ltm_slots", type=int, default=1024)
    arch_group.add_argument("--ltm_key_dim", type=int, default=128)
    arch_group.add_argument("--ltm_val_dim", type=int, default=128)
    
    # <<< MODIFIED: Defaults set to None to allow auto-sync logic >>>
    arch_group.add_argument("--h_hidden", type=int, default=None, help="[HRM] H-RNN hidden size. Defaults to context_dim if not set.")
    arch_group.add_argument("--l_hidden", type=int, default=None, help="[HRM] L-RNN hidden size. Defaults to context_dim if not set.")
    # <<< NEW ARGUMENT HERE >>>
    arch_group.add_argument("--h_stride", type=int, default=4, help="[HRM] Stride for the Manager (H-RNN). Updates h_state every N tokens.")
    arch_group.add_argument("--max_h_steps", type=int, default=5, help="[HRM] Maximum number of high-level refinement steps.")
    arch_group.add_argument("--max_l_steps", type=int, default=5, help="[HRM] Maximum number of low-level iterations before forcing completion.")
    arch_group.add_argument("--l_conv_atol", type=float, default=1e-4, help="[HRM] Absolute tolerance for checking L-module state convergence.")
    arch_group.add_argument("--ltm_topk", type=int, default=4, help="Number of LTM slots to retrieve per token.")
    arch_group.add_argument("--max_length", type=int, default=1024, help="Max sequence length. Required if using --pre_chunked_dataset, --pre_pt_dataset. Defaults to 1024 if not loading a model config or using --auto-max-length.")
    arch_group.add_argument("--auto-max-length", action="store_true", help="Automatically scan the dataset (--train or --hf_dataset) to find the longest sequence and set it as max_length.")

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training and Finetuning')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--accumulation-steps", type=int, default=1, help="Simulates a larger batch size.")
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine annealing.")
    train_group.add_argument("--disable-lr-schedule", action="store_true", help="Use a fixed LR instead of cosine annealing.")
    train_group.add_argument("--ltm_lr", type=float, default=1e-2, help="[Static] LR for LTM updates, or [Scheduled] MAX LR for the LTM cosine schedule.")
    train_group.add_argument("--kayla", action="store_true", help="Enable Kayla-style instruction tuning (with thought-process). Ignored if using pre-chunked formats or --text_column.")
    train_group.add_argument("--lora_r", type=int, default=8, help="[Finetune] LoRA rank.")
    train_group.add_argument("--lora_alpha", type=int, default=16, help="[Finetune] LoRA alpha.")
    train_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="[Finetune] Target percentage of params to train (e.g., 1.5 for 1.5%%). Overrides --lora_r.")
    train_group.add_argument("--quantize-on-complete", action="store_true", help="[Train] Automatically quantize after training.")
    train_group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value. Set to 0 to disable.")
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01, help="[HRM] Weight for the ponder cost auxiliary loss.")
    train_group.add_argument("--commitment-loss-weight", type=float, default=0.5, help="[HRM] Weight for the commitment auxiliary loss to prevent posterior collapse.")
    # <<< MERGED FROM V2: Commitment Threshold >>>
    train_group.add_argument("--commitment-threshold", type=float, default=0.05, help="[HRM] Hinge loss threshold (free bit budget) for drift penalty. Drift^2 below this is not penalized.")
    train_group.add_argument("--training-chunk-size", type=int, default=128, help="[Titans] Chunk size for temporal segmentation during training.")
    train_group.add_argument("--save-steps", type=int, default=0, help="Save a checkpoint every N steps (0 to disable).")
    
    train_group.add_argument("--override-scheduling", action="store_true", help="[Train] If resuming, ignore the scheduler state in the checkpoint and use the new LR args.")
    train_group.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading. Recommended: 2 or 4 for GPU training.")
    train_group.add_argument("--amp", action="store_true", help="[Train/Finetune/Chat] Enable Automatic Mixed Precision (AMP) for training/learning.")
    train_group.add_argument("--gradient-checkpointing", action="store_true", help="[Train/Finetune] Enable gradient checkpointing to save memory during training.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (experimental).")
    parser.add_argument("--force-compile", action="store_true", help="Force torch.compile even on Windows CPU (risky).")
    parser.add_argument("--debug-anomaly", action="store_true", help="Enable torch.autograd.set_detect_anomaly(True) for debugging NaNs.")

    # --- Inference Arguments ---
    infer_group = parser.add_argument_group('Inference (Chat)')
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--enable-quantized-learning", action="store_true", help="[Chat] Enable LTM updates for quantized models. Requires --shadow-model-path.")
    infer_group.add_argument("--ltm-lora-path", type=str, default=None, help="[Chat] Optional: Path to save/load LTM updates as a separate delta file.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "dml"], help="Force specific device (cuda, cpu, dml)")
    infer_group.add_argument("--h-halt-thresh", type=float, default=0.9, help="[HRM] Probability threshold for early exiting the H-module loop during inference.")
    infer_group.add_argument("--static-ltm-lr", action="store_true", help="[Chat] Disable the cosine annealing schedule for LTM updates and use a fixed LR instead.")
    infer_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="[Chat] The number of updates in one cosine annealing cycle for LTM learning.")
    infer_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="[Chat] The minimum learning rate for the LTM cosine annealing schedule.")

    # --- Other Arguments ---
    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument("--qtype", type=str, default="INT4", choices=["INT4", "Q4_0", "Q8_0", "Q2_K"], help="Quantization type/format.")
    other_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))

    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P (Nucleus) sampling")
    
    args = parser.parse_args()

    # <<< MODIFIED: Auto-Sync Dimensions (Only in Train Mode) >>>
    # If user did not specify hidden sizes, lock them to context_dim to prevent mismatch errors.
    if args.mode == 'train' and not args.resume_from_ckpt:
        if args.h_hidden is None:
            print(f"INFO: --h_hidden not specified. Defaulting to context_dim ({args.context_dim}).")
            args.h_hidden = args.context_dim
        if args.l_hidden is None:
            print(f"INFO: --l_hidden not specified. Defaulting to context_dim ({args.context_dim}).")
            args.l_hidden = args.context_dim
    # --------------------------------------

    # --- Argument Validation ---
    if args.mode in ['train', 'finetune']:
        if not args.hf_dataset and args.train is None and not (args.mode == 'train' and args.resume_from_ckpt):
            parser.error("Either `--train path/to/local/data` or `--hf_dataset name/or/path` must be specified for train/finetune mode (unless resuming train from ckpt).")
        if args.hf_dataset and args.train is not None and args.train is not True:
             parser.error("Cannot specify both `--train path/to/local/data` and `--hf_dataset` simultaneously.")
        if args.train is True and not args.hf_dataset:
            parser.error("The `--train` flag was used without a path, but no `--hf_dataset` was provided.")

        if args.hf_dataset and args.pre_chunked_dataset:
            parser.error("--hf_dataset cannot be used with --pre_chunked_dataset.")
        if args.hf_dataset and args.pre_pt_dataset:
            parser.error("--hf_dataset cannot be used with --pre_pt_dataset.")
        if args.hf_dataset and not _HAS_HF_DATASETS:
            parser.error("--hf_dataset specified, but the 'datasets' library is not installed. Please run: pip install datasets")
        if args.hf_dataset:
            if args.text_column and (args.prompt_column or args.completion_column):
                parser.error("--text_column is mutually exclusive with --prompt_column and --completion_column.")
            if (args.prompt_column and not args.completion_column) or (not args.prompt_column and args.completion_column):
                parser.error("Both --prompt_column and --completion_column must be specified together for instruction tuning.")

    # --- Existing validation ---
    if args.mode == 'finetune' and not args.model_path:
        parser.error("`--model-path` (base model) is required for finetune mode.")
    if args.mode == 'merge-lora' and not args.model_path:
        parser.error("`--model-path` (base model) is required for merge-lora mode.")
    if args.mode == 'merge-lora' and not args.lora_adapter_path:
        parser.error("`--lora-adapter-path` is required for merge-lora mode.")
    if args.mode == 'quantize' and not args.model_path and not args.quantize_on_complete:
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
    if (args.pre_chunked_dataset or args.pre_pt_dataset) and not args.max_length:
        parser.error("--max_length must be specified when using --pre_chunked_dataset or --pre_pt_dataset.")
    if (args.pre_chunked_dataset or args.pre_pt_dataset) and args.auto_max_length:
        print("Warning: --auto-max-length is ignored when using pre-chunked dataset formats.")
        args.auto_max_length = False 
    if (args.pre_chunked_dataset or args.pre_pt_dataset) and args.kayla:
        print("Warning: --kayla flag is ignored when using pre-chunked dataset formats.")

    set_threads(args.threads)
    pt_device = pick_device(args)
    print(f"Using PyTorch device: {pt_device}")

    # <<< DIRECTML CONSTRAINT ENFORCEMENT >>>
    if is_directml_device(pt_device):
        print("INFO: DirectML device selected. Enforcing stability constraints...")
        
        # 1. Disable torch.compile (not supported on DML)
        if args.compile:
            print("Warning: --compile is not supported on DirectML. Forcing --compile=False.")
            args.compile = False
        if args.force_compile:
            print("Warning: --force-compile ignored on DirectML.")
            args.force_compile = False
            
        # 2. Disable Custom Kernels (incompatible with DML tensors)
        if _HAS_KERNEL:
            print("INFO: Disabling custom C++ kernels (hierarchos_matmul) for this run to prevent crashes.")
            # We modify the global flag so QuantizedHierarchos sees it
            _HAS_KERNEL = False
            
        # 3. Disable AMP (if not already handled)
        if args.amp:
            print("Warning: AMP is unstable on DirectML. Forcing --amp=False.")
            args.amp = False

    if pt_device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    # --- AMP Availability Check ---
    if args.amp and not _HAS_AMP:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: --amp was specified, but torch amp support is not available.       !!!")
        print("!!!  AMP will be DISABLED. Check CUDA and PyTorch install.                  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        args.amp = False 
    elif args.amp and pt_device == torch.device('cpu'):
        print("Warning: AMP (--amp) is enabled but running on CPU. AMP will have no effect.")

    # --- Tokenizer Loading ---
    tokenizer = None
    tokenizer_load_path = None
    default_tokenizer = "openai-community/gpt2" 

    # 1. Prioritize --tokenizer-path
    if args.tokenizer_path:
        print(f"Attempting to load tokenizer from specified path: '{args.tokenizer_path}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            tokenizer_load_path = args.tokenizer_path
            print(f"Successfully loaded tokenizer from '{args.tokenizer_path}'")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer from '{args.tokenizer_path}'. Error: {e}")

    # 2. Try --model-path
    if tokenizer is None and args.model_path and not (args.mode == 'train' and not args.resume_from_ckpt and not args.model_path):
        print(f"Attempting to load tokenizer from model directory: {args.model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            tokenizer_load_path = args.model_path
            print(f"Successfully loaded tokenizer from '{args.model_path}'")
        except Exception as e:
            print(f"Warning: Could not load tokenizer from model directory '{args.model_path}'. Will try default. Error: {e}")

    # 3. Try default
    if tokenizer is None:
        print(f"Attempting to load default tokenizer: '{default_tokenizer}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(default_tokenizer, trust_remote_code=True)
            tokenizer_load_path = default_tokenizer
            print(f"Successfully loaded tokenizer from default '{default_tokenizer}'")
        except Exception as e:
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
            print(f"Set pad_token to eos_token ('{tokenizer.pad_token}')")
        else:
            print("Warning: Tokenizer missing both pad and eos tokens. Adding a '[PAD]' token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if args.mode == "train" and not args.resume_from_ckpt and not args.model_path:
                args.vocab_size = len(tokenizer) 
                print(f"Updated args.vocab_size to {args.vocab_size} due to added pad token.")

    # --- Temporary HF Dataset Loading (for auto-max-length) ---
    hf_raw_dataset_for_scan = None
    if args.auto_max_length and args.hf_dataset and args.mode in ['train', 'finetune']:
        if tokenizer is None:
             parser.error("--auto-max-length requires the tokenizer to be loaded first.")
        if not _HAS_HF_DATASETS:
             parser.error("--auto-max-length requested for HF dataset, but 'datasets' library is not installed.")
        print(f"Loading Hugging Face dataset ({args.hf_dataset}) temporarily for length scan...")
        try:
            hf_raw_dataset_for_scan = load_dataset(args.hf_dataset, name=args.hf_dataset_config, split=args.hf_dataset_split)
            print("Dataset loaded for scan.")
        except Exception as e:
            print(f"ERROR: Failed to load HF dataset '{args.hf_dataset}' for length scan: {e}")
            sys.exit(1)

    # --- Auto Max Length Scanning ---
    if args.auto_max_length and args.mode in ['train', 'finetune'] and not args.pre_chunked_dataset and not args.pre_pt_dataset:
        if tokenizer is None:
            parser.error("--auto-max-length requires the tokenizer to be loaded first.")

        max_found_length = 0
        skipped_scan_count = 0
        scan_desc = "Scanning Dataset"

        if args.hf_dataset:
            if hf_raw_dataset_for_scan is None:
                 print("ERROR: HF dataset required for scan was not loaded.")
                 sys.exit(1)
            print("INFO: --auto-max-length enabled. Scanning HF dataset...")
            scan_desc = f"Scanning HF: {args.hf_dataset}"
            for sample in tqdm(hf_raw_dataset_for_scan, desc=scan_desc):
                processed = process_text_sample(
                    tokenizer, sample, 999999, args.kayla, 
                    args.text_column, args.prompt_column, args.completion_column
                )
                if processed:
                    length = len(processed['input_ids'])
                    if length > max_found_length: max_found_length = length
                else: skipped_scan_count += 1

        else: 
            train_file_path = args.train 
            if not train_file_path and args.resume_from_ckpt:
                print("INFO: --auto-max-length used with --resume-from-ckpt. Trying to load train path from config...")
                try:
                    ckpt = torch.load(args.resume_from_ckpt, map_location='cpu', weights_only=False)
                    ckpt_conf = ckpt.get('config', {})
                    train_file_path = ckpt_conf.get('train_data_path')
                except Exception as e: print(f"Warning: Could not load train path from checkpoint config: {e}")

            if not train_file_path or not os.path.exists(train_file_path):
                parser.error("--auto-max-length requires a valid --train file path (and cannot be used with --hf_dataset).")

            print(f"INFO: --auto-max-length enabled. Scanning local file: {train_file_path}...")
            scan_desc = f"Scanning Local: {os.path.basename(train_file_path)}"
            with open(train_file_path, 'r', encoding='utf-8') as f:
                try: # Try JSON list
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        for obj in tqdm(data, desc=scan_desc):
                            processed = process_text_sample(
                                tokenizer, obj, 999999, args.kayla,
                                prompt_column='instruction', completion_column='output' 
                            )
                            if processed:
                                length = len(processed['input_ids'])
                                if length > max_found_length: max_found_length = length
                            else: skipped_scan_count += 1
                except: # Try JSONL
                    f.seek(0)
                    line_num_scan = 0
                    for line in tqdm(f, desc=scan_desc):
                        line_num_scan += 1
                        try:
                            obj = json.loads(line)
                            processed = process_text_sample(
                                tokenizer, obj, 999999, args.kayla,
                                prompt_column='instruction', completion_column='output'
                            )
                            if processed:
                                length = len(processed['input_ids'])
                                if length > max_found_length: max_found_length = length
                            else: skipped_scan_count += 1
                        except Exception as scan_e:
                            if skipped_scan_count % 1000 == 0:
                                print(f"\nWarning: Skipping line ~{line_num_scan} during scan: {scan_e}")
                            skipped_scan_count += 1
                            continue

        if skipped_scan_count > 0:
            print(f"Warning: Skipped {skipped_scan_count} invalid entries during length scan.")

        if max_found_length > 0:
            target_max_length = (max_found_length + 16 + 7) & -8
            print(f"Auto-scan complete. Found max length ~{max_found_length}. Setting max_length to {target_max_length}.")
            args.max_length = target_max_length 
            if tokenizer and hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < target_max_length:
                tokenizer.model_max_length = target_max_length
                print(f"Updated tokenizer.model_max_length to {target_max_length}")
        else:
            print(f"WARNING: Auto-scan did not find valid entries or failed. Using default max_length ({args.max_length}).")

    elif not args.max_length and args.mode in ['train', 'finetune'] and not args.pre_chunked_dataset and not args.pre_pt_dataset:
        print(f"Warning: No --max_length specified. Using default {args.max_length}.")

    # --- Dataloader Creation ---
    dataloader = None
    dataloader_len = 0 
    if args.mode in ["train", "finetune"]:
        if tokenizer is None and not args.pre_pt_dataset and not args.hf_dataset: 
            print("Error: Tokenizer failed to load, cannot create dataloader.")
            sys.exit(1)
        if args.max_length is None: 
            if not args.resume_from_ckpt and not args.model_path:
                print("Error: max_length could not be determined. Please specify --max_length or use --auto-max-length.")
                sys.exit(1)

        try:
            if args.hf_dataset:
                hf_raw_dataset = hf_raw_dataset_for_scan
                if hf_raw_dataset is None:
                     print(f"Loading Hugging Face dataset: {args.hf_dataset} (Config: {args.hf_dataset_config}, Split: {args.hf_dataset_split})")
                     hf_raw_dataset = load_dataset(args.hf_dataset, name=args.hf_dataset_config, split=args.hf_dataset_split)

                print(f"Dataset loaded: {hf_raw_dataset}")
                if tokenizer is None:
                     print("Error: Tokenizer is required for Hugging Face dataset processing but failed to load.")
                     sys.exit(1)
                hf_dataset = HuggingFaceMapStyleDataset(
                    hf_raw_dataset, tokenizer, args.max_length, args.kayla,
                    args.text_column, args.prompt_column, args.completion_column
                )
                dataloader = create_map_style_dataloader(
                    hf_dataset, args.batch_size, tokenizer.pad_token_id, args.num_workers, shuffle=True
                )
                dataloader_len = len(dataloader)
                print(f"INFO: HF DataLoader created with {dataloader_len} batches.")

            elif args.pre_pt_dataset:
                print("INFO: Loading pre-chunked .pt tensors (map-style).")
                dataloader = create_dataloader_pt_chunked(
                    args.train, max_length=args.max_length, batch_size=args.batch_size, num_workers=args.num_workers
                )
                dataloader_len = len(dataloader)
                print(f"INFO: DataLoader created with {dataloader_len} batches.")
            elif args.pre_chunked_dataset:
                print("INFO: Loading pre-chunked JSONL dataset (iterable).")
                dataloader = create_dataloader_for_chunked(
                    args.train, max_length=args.max_length, batch_size=args.batch_size, num_workers=args.num_workers
                )
                try:
                    with open(args.train, 'r') as f:
                        estimated_lines = sum(1 for _ in f)
                    dataloader_len = estimated_lines // args.batch_size 
                    print(f"INFO: Estimated DataLoader length (for scheduler): {dataloader_len} batches.")
                except:
                    print("Warning: Could not estimate dataset length for scheduler. Using placeholder T_max=100000.")
                    dataloader_len = 100000 
            else: 
                if not args.train or not isinstance(args.train, str):
                    parser.error("A local dataset path via --train is required when not using --hf_dataset or pre-chunked formats.")
                print("INFO: Loading and tokenizing JSON/JSONL dataset on the fly (map-style).")
                if tokenizer is None:
                     print("Error: Tokenizer is required for JSON/JSONL processing but failed to load.")
                     sys.exit(1)
                original_dataset = OriginalJSONLDataset(
                    args.train, tokenizer, args.max_length, kayla_mode=args.kayla
                )
                dataloader = create_map_style_dataloader(
                    original_dataset, args.batch_size, tokenizer.pad_token_id, args.num_workers, shuffle=True
                )
                dataloader_len = len(dataloader)
                print(f"INFO: DataLoader created with {dataloader_len} batches.")

        except Exception as e:
            print(f"ERROR creating DataLoader: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Execute Selected Mode ---
    if args.mode == "train":
        train(args, pt_device, tokenizer, dataloader, dataloader_len) 
    elif args.mode == "finetune":
        finetune(args, pt_device, tokenizer, dataloader, dataloader_len) 
    elif args.mode == "merge-lora":
        merge_lora(args, pt_device, tokenizer)
    elif args.mode == "quantize":
        quantize(args, pt_device, tokenizer=tokenizer)
    elif args.mode == "chat":
        if tokenizer is None:
            print("Error: Tokenizer failed to load, cannot start chat.")
            sys.exit(1)
        chat(args, pt_device, tokenizer)

