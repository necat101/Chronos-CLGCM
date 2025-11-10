-----

# Hierarchos v0.8.6 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

Due to Amazon's "Chronos" forcasting models (still based on transformers BTW) I've decided to rename the project to "Hierarchos" from this point forward. this should prevent any naming confusion that may occur. all functional code is still the same

-----

### 🚀 **Major Update in v0.8.5: Reworked Build System & Vulkan Auto-Setup**

> This version completely overhauls the C++ kernel build process for simplicity and cross-platform reliability, especially for the optional Vulkan backend.
>
> 1.  **CPU-First Default:** ⚙️ The build scripts (`setup.bat`, `setup.sh`) now compile the **CPU-optimized kernel by default**. This is the recommended path for most users, as Hierarchos's architecture is highly sequential and benefits from fast single-core performance.
> 2.  **Opt-in Vulkan Build:** 🔥 To enable the Vulkan backend for GPU-accelerated quantized inference, simply pass the `--vulkan` flag (e.g., `bash setup.sh --vulkan` or `setup.bat --vulkan`).
> 3.  **Automatic Dependency Installation (Linux):** 🐧 The `setup.sh` script will now automatically detect if Vulkan build tools (`glslang-tools`, `libvulkan-dev`) are missing and, after prompting for `sudo`, will install them via `apt`.
> 4.  **Cross-Platform Shader Compiler Fix:** 🛠️ The CMake build system is now smart enough to find the correct shader compiler on both Windows (`glslc.exe`) and Linux (`glslangValidator`), resolving all previous cross-platform compilation errors.

### 🚀 **Major Update in v0.8.0: `torch.compile` Integration for Massive Speedups**

> This version introduces **`torch.compile`**, leveraging the PyTorch 2.0+ JIT compiler to dramatically accelerate training on modern NVIDIA GPUs.
>
> 1.  **Experimental, Automatic Speedup:** 🔥 Hierarchos now **automatically** uses `torch.compile` on its core reasoning loop (`_adaptive_hrm_step`) if a compatible PyTorch version (2.0+) is detected. This fuses the model's complex operations into highly optimized kernels.
> 2.  **Pay a One-Time "Warmup" Cost:** ⏳ The **first few steps or the first epoch of training will be significantly slower** than usual (e.g., 30-60+ seconds per step). This is expected—it's the one-time cost of the JIT compiler analyzing and optimizing the model.
> 3.  **Massively Faster Training:** ⚡ After the initial compilation, subsequent training steps and epochs will be **dramatically faster** (often 1.5x-3x+), especially on powerful GPUs like the A100.
> 4.  **Intelligent Configuration:** ⚙️ The implementation is pre-configured to handle the most common pitfalls:
>       * **`dynamic=True`**: Handles variable sequence lengths from dynamic batching, preventing a recompilation on every step.
>       * **`options={"triton.cudagraphs": False}`**: CUDAGraphs is disabled to prevent notorious deadlocks with the `DataLoader`'s multiprocessing workers on Linux systems.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Hierarchos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Hierarchos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Hierarchos is built on two revolutionary, brain-inspired pillars:

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are now structured with timestamps and source metadata, allowing for sophisticated, context-aware queries.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth through **iterative convergence**. This enables it to solve complex, multi-step algorithmic problems where massive LLMs fail, though the depth of reasoning directly impacts computational cost during training.

## Features ✨

  * 🔥 **PyTorch 2.0+ Compiled Training**: **Automatically uses `torch.compile`** on the core HRM loop for massive speedups (1.5x-3x+) on modern NVIDIA GPUs after an initial "warmup" compilation.
  * 🌐 **Hugging Face `datasets` Integration**: Load datasets directly from the HF Hub or local paths in various formats (CSV, Parquet, JSON, etc.) using `--hf_dataset`.
  * 💾 **Optimized Consolidated Chunk Loading**: Dramatically reduces RAM usage and speeds up training startup for large datasets using pre-processed, consolidated `.pt` tensor files and a manifest (`--pre_pt_dataset`). Includes file caching for efficiency.
  * 📜 **Iterable Dataset Support**: Option to load pre-chunked JSONL datasets line-by-line (`--pre_chunked_dataset`) for minimal memory overhead during training.
  * ✂️ **Dataset Consolidation Script (`dataset_chunk_create.py`)**: Enhanced tool to prepare large datasets, chunking them into **consolidated `.pt` files** and creating a `manifest.jsonl` for efficient loading. Handles tokenization, anchoring, padding, and masking.
  * 📉 **Gradient Checkpointing**: Significantly reduces VRAM usage during training/fine-tuning (`--gradient-checkpointing`), enabling larger models or batches on memory-constrained hardware by trading compute for memory.
  * 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  * 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  * 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a Cosine Annealing LR schedule by default for more stable knowledge consolidation.
  * ⚡ **Accelerated Training with AMP**: Supports Automatic Mixed Precision (`--amp`) for faster training and reduced memory usage on compatible NVIDIA GPUs.
  * 🛡️ **Stable Training**: Built-in gradient clipping (`--grad-clip`) to prevent model instability and ensure smoother convergence.
  * 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config for easy sharing and use.
  * 💾 **Automatic Re-quantization**: After a learning session, Hierarchos can automatically re-quantize a model to persist the new knowledge (`--enable-quantized-learning` in `chat`). *(Requires compiled kernel)*
  * 🌱 **Enhanced Model Expansion**: Includes `expand_model.py` script to transplant weights from smaller models to larger ones, now supporting changes in `max_length` and automatic length detection from datasets.
  * ✨ **Flexible Training Initiation**: Supports starting training runs using weights from existing model directories (inference or expanded models via `--model-path` in `train` mode), not just resuming full training checkpoints (`--resume-from-ckpt`).
  * ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). *(Requires compiled kernel)*
  * 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX/NEON) or on GPUs via Vulkan for broad hardware compatibility. *(Requires compiled kernel)*
  * 🔧 **Comprehensive Tooling**: Includes a single script (`hierarchos.py`) for training, LoRA fine-tuning, merging, quantization, and interactive chat, plus the model expansion and dataset chunking scripts.

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  * Python 3.8+
  * **PyTorch 2.0+ (Required for `torch.compile` speedups)**
  * **For Hugging Face Datasets:** `pip install datasets`
  * **Optional (Quantization/Vulkan):**
  * A C++ compiler (e.g., MSVC on Windows, GCC on Linux)
  * CMake (must be available in your system's `PATH`)
  * Vulkan-compatible GPU and installed drivers (for Vulkan inference)
  * **Vulkan SDK** (if compiling with `--vulkan` on Windows. On Linux, the script will attempt to auto-install `glslang-tools` and `libvulkan-dev` via `apt`.)
  * **Optional (AMP Training/Gradient Checkpointing):** NVIDIA GPU with CUDA support (Compute Capability 7.0+ recommended) and a PyTorch build with CUDA enabled.
  * **Optional (Kernel Build Dependencies):** `pip install pybind11 cmake`

### Installation

-----

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/necat101/Hierarchos.git
    cd Hierarchos
    ```

2.  **Run the Setup Script:**
    This script automatically checks for core dependencies (like Python, CMake, and a C++ compiler), installs all required Python packages from `requirements_kernel.txt`, and builds the C++ kernel.

    **Default CPU Build (Recommended):**
    This is the standard build for most users, optimized for CPU performance.

    ```bash
    # On Windows
    setup.bat

    # On Linux/macOS
    bash setup.sh
    ```

    **Vulkan Build (Optional):**
    To enable GPU-accelerated quantized inference with Vulkan, pass the `--vulkan` flag.

    ```bash
    # On Windows (Requires Vulkan SDK to be installed)
    setup.bat --vulkan

    # On Linux/macOS (Will prompt to auto-install Vulkan tools via apt)
    bash setup.sh --vulkan
    ```

    Running the script creates the `hierarchos_matmul.*` kernel file in your project root. If you don't run the script, quantization and Vulkan inference will be disabled, but the core model training and inference will still function in pure Python.

-----

## 📚 User Guide: Comprehensive Workflows

This guide covers common scenarios from data preparation to inference.

### Workflow 1: Training a New Model

Choose **one** data source option:

**(A) Local JSON/JSONL File (Fits in RAM):**

```bash
python hierarchos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" `# Or your preferred tokenizer` \
    --out-dir "./my_Hierarchos_model" \
    --epochs 3 \
    --batch_size 4 \
    --accumulation-steps 2 `# Effective batch size = 8` \
    --auto-max-length `# Automatically determines max sequence length` \
    --context_dim 768 `# Example architecture` \
    --h_hidden 768 \
    --l_hidden 768 \
    --max_h_steps 5 \
    --max_l_steps 5 \
    --amp `# Enable Mixed Precision for speed` \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(B) Hugging Face Dataset (Text Completion):**

```bash
python hierarchos.py train \
    --hf_dataset "wikitext" \
    --hf_dataset_config "wikitext-2-raw-v1" \
    --hf_dataset_split "train" \
    --text_column "text" `# Column containing the text` \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_wikitext_model" \
    --epochs 1 \
    --batch_size 2 \
    --accumulation-steps 4 \
    --auto-max-length \
    --amp \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(C) Hugging Face Dataset (Instruction/Kayla Format):**

```bash
python hierarchos.py train \
    --hf_dataset "databricks/databricks-dolly-15k" \
    --prompt_column "Instruction" \
    --completion_column "output" \
    # --kayla # Add if your HF data structure matches Kayla format (instruction, output, thought-process, feelings) \
    # --text_column "context" # Example: Map 'context' field if needed for your format \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_dolly_model" \
    --epochs 2 \
    --batch_size 1 \
    --accumulation-steps 8 \
    --auto-max-length \
    --amp \
    --gradient-checkpointing # Add if VRAM is limited
```

**(D) Pre-Chunked Local Dataset (Very Large Dataset):**

  * **Step 1: Create Chunks**
    ```bash
    python dataset_chunk_create.py \
        --dataset "path/to/very_large_data.jsonl" \
        --tokenizer-path "openai-community/gpt2" \
        --output-dir "./very_large_data_chunked" \
        --overlap 512 \
        --chunks-per-file 1000
    # Note the MAX_SEQ_LENGTH printed by the script (e.g., 3153)
    ```
  * **Step 2: Train using Chunks**
    ```bash
    python hierarchos.py train \
        --pre_pt_dataset `# Enable loading via manifest` \
        --train "./very_large_data_chunked" `# Directory with .pt files & manifest` \
        --max_length 3153 `# MUST match chunker output` \
        --tokenizer-path "openai-community/gpt2" `# Still needed for model init` \
        --out-dir "./my_large_model" \
        --epochs 1 \
        --batch_size 1 \
        --accumulation-steps 8 \
        --amp \
        --gradient-checkpointing # Add this if VRAM is limited
    ```

-----

💡 **Accelerating Training with AMP:** Use `--amp` for faster training and lower VRAM usage on NVIDIA GPUs.
💾 **Training on Low Memory:** Use `--gradient-checkpointing` to significantly reduce VRAM usage at the cost of some extra computation.

## ⚠️ **HRM Convergence & Training Speed:** Higher `--max_h_steps` and `--max_l_steps` allow deeper reasoning but **significantly increase training time** per batch due to the iterative HRM process. Adjust based on your task and compute resources.

### Workflow 2: Fine-Tuning with LoRA

Adapt a pre-trained model using new data (any supported format).

```bash
python hierarchos.py finetune \
    --model-path "./my_Hierarchos_model" `# Path to your trained base model` \
    --hf_dataset "squad" `# Example: Use SQuAD for QA fine-tuning` \
    --prompt_column "question" \
    --completion_column "answers" `# Might need custom processing depending on format` \
    --text_column "context" `# Use context as part of the prompt` \
    --out-dir "./my_squad_lora" \
    --epochs 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --amp \
    --gradient-checkpointing `# Use if fine-tuning large models on limited VRAM`
```

### Workflow 3: Merging LoRA Adapter

Combine the base model and the LoRA adapter into a new, standalone model.

```bash
python hierarchos.py merge-lora \
    --model-path "./my_Hierarchos_model" \
    --lora-adapter-path "./my_squad_lora" \
    --out-dir "./my_model_merged_squad"
```

### Workflow 4: Quantizing a Model *(Requires Compiled Kernel)*

Convert a full-precision model to a quantized format for faster, lower-resource inference.

```bash
python hierarchos.py quantize \
    --model-path "./my_model_merged_squad" \
    --out-dir "./my_model_merged_squad-Q4_0" \
    --qtype Q4_0 `# Choose format: INT4, Q4_0, Q8_0, Q2_K`
```

### Workflow 5: Running Chat Inference

Interact with your trained or fine-tuned model.

**Full Precision:**

```bash
python hierarchos.py chat --model-path "./my_model_merged_squad"
```

**Quantized *(Requires Compiled Kernel)*:**

```bash
python hierarchos.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --device cpu `# Use "vulkan" if you built with --vulkan`
```

**Chat with Online Learning (Quantized Example - Requires Compiled Kernel):**

```bash
python hierarchos.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged_squad" `# Path to original full-precision model` \
    --amp `# Optional: Speed up the learning step on CUDA` \
    # --ltm-lora-path "./my_chat_ltm_updates.pt" # Optional: Save LTM updates separately
```

### Workflow 6: Resuming Interrupted Training

Continue a `train` run from a saved checkpoint (`.pt` file).

```bash
python hierarchos.py train \
    # Dataset args might be loaded from checkpoint, specify only if needed \
    --out-dir "./my_large_model" \
    --resume-from-ckpt "./my_large_model/Hierarchos_epoch_1.pt" \
    --epochs 3 `# Total desired epochs` \
    --amp \
    --gradient-checkpointing # Ensure flag is consistent with the resumed run if needed
```

  * Use `--override-scheduling` with `--starting-lr`/`--min-lr` to change the learning rate schedule upon resuming.

### Workflow 7: Expanding a Model *(Requires `expand_model.py`)*

Create a larger model architecture initialized with weights from a smaller trained one.

```bash
python expand_model.py \
    --old-model-path "./my_Hierarchos_model/Hierarchos.pt" `# Trained smaller model .pt file` \
    --output-path "./expanded_model/Hierarchos.pt" `# Path for the new, expanded .pt file` \
    --context_dim 1024 `# New larger dimension` \
    --h_hidden 1024 \
    --l_hidden 1024
    # Note: expand_model.py takes specific architecture args to change.
    # Other config values are copied from the old model's checkpoint.
```

### Workflow 8: Continuing Training (After Expanding or from Inference Checkpoint)

Start a *new* training session using only the *weights* from an existing model directory (not resuming optimizer/scheduler state).

```bash
python hierarchos.py train \
    --hf_dataset "new_dataset_for_larger_model" \
    --text_column "text" \
    --model-path "./expanded_model" `# Load weights from expanded/previous model directory` \
    --tokenizer-path "./expanded_model" `# Use its tokenizer (assuming it was copied)` \
    --out-dir "./expanded_model_trained" \
    --epochs 2 \
    --starting-lr 5e-5 `# Start with a potentially smaller LR` \
    --amp \
    --gradient-checkpointing # Add if VRAM is limited
```

-----

## ⚙️ Command-Line Reference

### `hierarchos.py` Arguments

| Argument | Mode(s) | Description | Default |
| :--- | :--- | :--- | :--- |
| **Paths & Data** | | | |
| `--train` | `train`, `finetune` | Path to **local** data: JSON/JSONL file, or directory for `--pre_pt_dataset`. Use flag without path if using `--hf_dataset`. Mutually Exclusive with `--hf_dataset` path. | `None` |
| `--hf_dataset` | `train`, `finetune` | Name or path to a Hugging Face dataset (e.g., 'wikitext', 'c4', 'path/to/my\_csv/'). Mutually Exclusive with `--train` path. | `None` |
| `--hf_dataset_config` | `train`, `finetune` | Optional configuration name for the HF dataset (e.g., 'wikitext-103-raw-v1'). | `None` |
| `--hf_dataset_split` | `train`, `finetune` | Dataset split to use (e.g., 'train', 'validation', 'train[:10%]'). | `train` |
| `--text_column` | `train`, `finetune` | Column name for text completion data in HF dataset (mutually exclusive with prompt/completion). Defaults to 'text' if available. | `None` |
| `--prompt_column` | `train`, `finetune` | Column name for prompt/instruction in HF dataset. Use with `--completion_column`. | `None` |
| `--completion_column` | `train`, `finetune` | Column name for completion/response in HF dataset. Use with `--prompt_column`. | `None` |
| `--pre_chunked_dataset` | `train`, `finetune` | Load pre-chunked **JSONL** dataset iteratively (requires `--max_length`). Mutually Exclusive with `--pre_pt_dataset` & `--hf_dataset`. | `False` |
| `--pre_pt_dataset` | `train`, `finetune` | Load pre-chunked **consolidated `.pt` tensor** dataset from directory specified in `--train` (requires `--max_length`). Mutually Exclusive with `--pre_chunked_dataset` & `--hf_dataset`. | `False` |
| `--model-path` | `train`, `finetune`, `merge`, `quantize`, `chat` | Path to model directory. **[Train]**: Loads weights only (starts fresh training). **[Other]**: Loads for the specified mode. | `None` |
| `--out-dir` | `train`, `finetune`, `merge`, `quantize` | Directory to save new models, checkpoints, or adapters. | `./Hierarchos_model` |
| `--tokenizer-path` | `train`, `finetune`, `merge`, `quantize` | Path or HF name of tokenizer (if not loading from model-path). | `openai-community/gpt2` |
| `--resume-from-ckpt` | `train` | Path to `.pt` checkpoint to **resume full training state** (optimizer, etc.). | `None` |
| `--shadow-model-path` | `chat` | Path to full-precision model dir for online learning with quantized model. | `None` |
| `--lora-adapter-path` | `merge`, `finetune` | Path to the trained LoRA adapter directory. | `None` |
| **Training/Fine-Tuning** | | | |
| `--epochs` | `train`, `finetune` | Number of training epochs. | `3` |
| `--batch_size` | `train`, `finetune` | Number of samples per forward pass. | `4` |
| `--accumulation-steps` | `train`, `finetune` | Number of steps to accumulate gradients over (simulates larger batch size). | `1` |
| `--gradient-checkpointing` | `train`, `finetune` | **Enable gradient checkpointing to save VRAM (trades compute for memory).** | `False` |
| `--grad-clip` | `train`, `finetune` | Gradient clipping value. Prevents gradient explosion (0 to disable). | `1.0` |
| `--ponder-loss-weight` | `train`, `finetune` | Weight for the Ponder Cost auxiliary loss. | `0.01` |
| `--override-scheduling` | `train` | **[If resuming]** Ignore checkpoint's schedule state and use new LR args. | `False` |
| `--starting-lr` | `train`, `finetune` | Max Learning Rate for the schedule, or fixed LR if schedule disabled. | `1e-4` |
| `--min-lr` | `train`, `finetune` | Minimum Learning Rate for cosine annealing schedule. | `1e-6` |
| `--disable-lr-schedule` | `train`, `finetune` | Use a fixed Learning Rate (`--starting-lr`) instead of cosine annealing. | `False` |
| `--ltm_lr` | `train`, `finetune`, `chat` | Learning Rate for LTM "surprise" updates (or max LR for LTM schedule in chat). | `0.01` |
| `--amp` | `train`, `finetune`, `chat` | **Enable Automatic Mixed Precision (requires CUDA).** | `False` |
| `--num_workers` | `train`, `finetune` | Number of CPU workers for data loading (and HF dataset mapping if applicable). | `0` |
| `--lora_r` | `finetune` | LoRA rank 'r'. | `8` |
| `--lora_alpha` | `finetune` | LoRA alpha scaling factor. | `16` |
| `--finetune-unlock-percent` | `finetune` | Target % of params to train (approx.). Overrides `--lora_r` if set. | `None` |
| `--kayla` | `train`, `finetune` | Enable Kayla-style instruction tuning format (with thought-process). **Ignored if using pre-chunked formats or --text\_column.** | `False` |
| **Quantization/Inference** | | | |
| `--qtype` | `quantize`, `train` | Quantization format (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). Used by `quantize` or `--quantize-on-complete`. **Requires compiled kernel.** | `INT4` |
| `--quantize-on-complete` | `train` | Automatically run quantization after training finishes. **Requires compiled kernel.** | `False` |
| `--device` | `chat` | Device for *quantized* inference (`cpu`, `vulkan`). **Requires kernel compiled with `--vulkan` flag.** | `cpu` |
| `--h-halt-thresh` | `chat` | Probability threshold for early exiting the HRM reasoning loop during inference. | `0.9` |
| `--max-new-tokens` | `chat` | Maximum number of tokens to generate in chat mode. | `512` |
| `--enable-quantized-learning` | `chat` | Enable LTM updates for quantized models (requires `--shadow-model-path` and **compiled kernel**). | `False` |
| `--ltm-lora-path` | `chat` | Optional: Path to save/load LTM updates as a separate delta file in chat mode. | `None` |
| `--static-ltm-lr` | `chat` | Disable cosine annealing for chat LTM updates, use fixed `--ltm_lr`. | `False` |
| `--ltm-schedule-steps` | `chat` | Number of chat updates per LTM LR cosine cycle. | `100` |
| `--ltm-schedule-min-lr` | `chat` | Minimum LR for chat LTM cosine schedule. | `1e-5` |
| **Architecture (Train)** | | *(Used only if starting train from scratch)* | |
| `--context_dim` | `train` | Core embedding dimension. | `768` |
| `--persistent_dim` | `train` | Dimension of the fixed Persistent Memory. | `128` |
| `--ltm_slots` | `train` | Number of slots in the Long-Term Memory. | `1024` |
| `--ltm_key_dim` | `train` | Dimension of LTM keys. | `128` |
| `--ltm_val_dim` | `train` | Dimension of LTM values. | `128` |
| `--h_hidden` | `train` | Hidden size of the High-Level (CEO) RNN. | `768` |
| `--l_hidden` | `train` | Hidden size of the Low-Level (Worker) RNN. | `768` |
| `--max_h_steps` | `train` | **Maximum** number of reasoning steps H-module can take. **Impacts training speed.** | `5` |
| `--max_l_steps` | `train` | **Maximum** number of iterations for L-module convergence per H-step. **Impacts training speed.** | `5` |
| `--l_conv_atol` | `train` | Absolute tolerance for checking L-module state convergence. | `1e-4` |
| `--ltm_topk` | `train` | Number of LTM slots to retrieve per token. | `2` |
| `--max_length` | `train`, `finetune` | Maximum sequence length. **Required if using pre-chunked formats.** Set via scan (`--auto-max-length`), manually, or loaded from config. | `1024` |
| `--auto-max-length` | `train`, `finetune` | Automatically scan dataset (`--train` or `--hf_dataset`) to set `max_length`. **Ignored if using pre-chunked formats.** | `False` |
| **Other** | | | |
| `--threads` | `All` | Number of CPU threads for PyTorch/OpenMP. | `CPU_Count/2` |

### `dataset_chunk_create.py` Arguments ✂️

*(No changes)*

| Argument | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `--dataset` | Path to the input **JSONL** dataset file (Kayla format recommended). | Yes | |
| `--tokenizer-path` | Path or Hugging Face name of the tokenizer to use for chunking. | No | `openai-community/gpt2` |
| `--output-dir` | Directory to save the output **consolidated** `.pt` chunk files and `manifest.jsonl`. | No | `train_Hierarchos_chunked_tensors` |
| `--overlap` | Number of tokens to overlap between consecutive chunks. | No | `1024` |
| `--chunks-per-file` | Number of individual chunks to **consolidate** into a single `.pt` file. | No | `1000` |

### `expand_model.py` Arguments 🌱

| Argument | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `--old-model-path` | Path to the trained smaller model ***.pt checkpoint file***. | Yes | |
| `--output-path` | Path to save the new, expanded ***.pt model file***. | Yes | |
| `--context_dim` | ***Required:*** New context dimension. | Yes | |
| `--h_hidden` | ***Required:*** New H-RNN hidden size. | Yes | |
| `--l_hidden` | ***Required:*** New L-RNN hidden size. | Yes | |
| *Other Arch Args* | *Optional:* Add other architectural args like `--ltm_slots`, `--max_length`, etc., if changing them. | No | *(Uses old model's value)* |

-----

### Real world training performance (ROG ALLY Z1 Extreme CPU on Hierarchos v0.8.0 alpha):
<img width="1113" height="626" alt="WindowsTerminal_GXV9uDzz0R" src="https://github.com/user-attachments/assets/ade557e2-b577-4df9-a688-06bbf0ec49b2" />
<img width="1824" height="647" alt="chrome_wX62mwSBIH" src="https://github.com/user-attachments/assets/8b864948-4320-41cd-b619-665309fd1f48" />



## Roadmap

  * [ ] Develop a user-friendly GUI wrapper for easier interaction.
  * [ ] Extend the architecture to support multi-modal inputs (images, audio).
  * [ ] Implement the entire training loop in Vulkan/CUDA for end-to-end GPU acceleration.

## License

The source code of Hierarchos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See `LICENSE.md` for full details.

## Support This Project

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Hierarchos in my spare time while working full-time at a grocery store.

**[https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)**

## Acknowledgements

  * This architecture is inspired by the concepts in Google's **Titans** and Sapient Intelligence's **HRM** papers.
  * The quantization kernel design is heavily influenced by the groundbreaking work in **llama.cpp**.
  * **pybind11** for seamless C++/Python integration.
  * **Hugging Face `datasets`** library for broad data compatibility.
  * **PyTorch Team** for `torch.compile` and gradient checkpointing functionality.

## Changelog
### v0.8.6 (alpha)

  * **Improved alpaca dataset handling**:
      * "Instruction" and "input" are now merged during training as needed to follow the alpaca dataset format!

### v0.8.5 (alpha)

  * **Reworked Kernel Build System**:
      * `setup.bat` and `setup.sh` now build the **CPU-only** kernel by default, which is recommended for most users due to the model's sequential nature.
      * Added a `--vulkan` flag to both scripts to optionally compile the Vulkan backend for GPU-accelerated quantized inference.
  * **Added Linux Vulkan SDK Auto-Installer**:
      * `setup.sh --vulkan` will now detect if `glslangValidator` is missing and attempt to install `glslang-tools` and `libvulkan-dev` via `apt` (prompting for sudo).
  * **Fixed Cross-Platform Vulkan Build**:
      * Updated `CMakeLists.txt` to correctly find both `glslangValidator` (common on Linux) and `glslc` (common on Windows).
      * Made the `-V` compiler flag conditional, as it's required by `glslangValidator` but rejected by `glslc`, fixing the Windows build failure.
      * Fixed the Linux build failure by adding a command to create the `shaders` build directory before the compiler tries to write to it.

### v0.8.0 (alpha)

  * **Added Experimental `torch.compile` Support**:
      * The core reasoning loop (`_adaptive_hrm_step`) is now automatically compiled using `torch.compile` if PyTorch 2.0+ is detected.
      * This results in a **massive training speedup** (1.5x-3x+) on modern NVIDIA GPUs after an initial one-time "warmup" compilation on the first few steps.
      * The compiler is pre-configured with `dynamic=True` to handle variable batch lengths and prevent recompilation.
      * CUDAGraphs is explicitly disabled (`options={"triton.cudagraphs": False}`) to resolve `DataLoader` multiprocessing deadlocks on Linux systems.

### v0.7.5 (alpha)

  * **Added Gradient Checkpointing**:
      * Implemented gradient checkpointing (`torch.utils.checkpoint.checkpoint`) within the `HierarchosCore` model's forward pass, specifically targeting the Adaptive HRM loop (`_adaptive_hrm_step`).
      * Added the `--gradient-checkpointing` command-line flag for `train` and `finetune` modes to enable this feature.
      * When enabled, this significantly reduces VRAM usage by recomputing activations during the backward pass instead of storing them, allowing for larger models or batches on memory-constrained GPUs.
  * Updated `train` function to save the `gradient_checkpointing` state in model config/checkpoints.
  * **Updated Documentation**: Added comprehensive documentation for gradient checkpointing in README (Features, User Guide, Command-Line Reference, Changelog). Updated version number. Corrected `expand_model.py` usage/arguments. Restored previously removed documentation sections.

### v0.7.0 (alpha)

  * **Added Hugging Face `datasets` Support**:
      * Integrated `datasets` library to load data directly from the Hub or local paths (CSV, Parquet, JSON, Arrow, text, etc.).
      * Added new arguments: `--hf_dataset`, `--hf_dataset_config`, `--hf_dataset_split`, `--text_column`, `--prompt_column`, `--completion_column`.
      * `--train` and `--hf_dataset` are now mutually exclusive sources.
      * Updated `train`, `finetune`, and `main` functions to handle the new loading mechanism.
      * Added `HuggingFaceMapStyleDataset` class and refactored dataloader creation.
      * Added `datasets` to requirements files.
  * **Clarified HRM Training Cost**: Added explanation in README about the impact of `--max_h_steps` and `--max_l_steps` on training speed and compute requirements due to iterative convergence.
  * **Updated Documentation**: Modified User Guide examples and Command-Line Reference to include HF dataset usage and arguments. Corrected defaults and argument descriptions based on latest code.

### v0.6.2 (alpha)

  * **Migrated from keyboard to signal**: Now uses Python standard "signal" library for chat interruption.

### v0.6.1 (alpha)

  * **Optimized Pre-Chunked Tensor Loading (`--pre_pt_dataset`)**:
  * `dataset_chunk_create.py` now saves **consolidated `.pt` files**.
  * A `manifest.jsonl` file is created for mapping chunks.
  * `PTChunkedDataset` updated to use manifest and **caching**.
  * **Documentation**: Updated README for consolidated chunking.

### v0.6 (alpha)

  * **Added Dataset Pre-processing Script (`dataset_chunk_create.py`)**: Chunks large `.jsonl` datasets into `.pt` tensor files.
  * **Implemented Direct Tensor Dataset Loading (`--pre_pt_dataset`)**: Load from `.pt` files + manifest.
  * **Implemented Iterable Pre-Chunked JSONL Loading (`--pre_chunked_dataset`)**: Load large JSONL line-by-line.
  * **Updated Dataloader Logic**: Conditional loading based on flags.
  * **Refined Training State Saving**: Checkpoints save dataset type flags.
  * **Documentation**: Updated for new chunking workflow.

### v0.5.2 (alpha)

  * **Added Flexible Training Initiation**: `--model-path` in `train` mode loads weights only for a new session.
  * **Enhanced `expand_model.py` Script**: Added `max_length` expansion and auto-detection.
  * **Added Automatic Mixed Precision (AMP)**: `--amp` flag for `train`, `finetune`, `chat`.
  * **Documentation**: Updated for new features.

### v0.5.1 (alpha)

  * **Added `--override-scheduling` flag**: Force new LR schedule when resuming.
  * **Documentation**: Updated for `--override-scheduling`.

### v0.5 (alpha)

  * **Implemented Structured Long-Term Memory**: Added timestamps and source metadata.
  * **Implemented Adaptive Reasoning Depth (Ponder Time)**: Dynamic HRM steps.
  * **Added Ponder Cost**: Auxiliary loss for efficiency.
  * **Added Halting Threshold**: Inference control (`--h-halt-thresh`).

### v0.4 (alpha)

  * **Implemented Dynamic LTM Learning Rate**: Default Cosine Annealing schedule in chat.
  * **Added Static LR Fallback**: `--static-ltm-lr` flag for chat.
  * **Added Gradient Clipping**: `--grad-clip` for training stability.

-----

© 2025 Makhi Burroughs
