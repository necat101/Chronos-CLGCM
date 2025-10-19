-----
# Chronos v0.5.2 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### 📢 **Major Update in v0.5: Enhanced Reasoning & Structured Memory**

> This version introduces two major architectural upgrades that work in tandem to create a more powerful and efficient cognitive process.
>
> 1.  **Adaptive Reasoning Depth ("Ponder Time"):** Chronos can now **"think longer" for harder problems**. The fixed-step reasoning loop has been replaced with a dynamic mechanism where the model learns to decide when it's confident enough to stop reasoning. This makes the model both more powerful and more efficient, better emulating the variable effort of human thought.
>
> 2.  **Structured & Queryable Long-Term Memory:** The LTM is no longer just an associative memory. Each memory slot is now **augmented with metadata, including a timestamp and a source identifier**. This allows the model—and the user—to perform sophisticated, context-aware queries, such as retrieving only memories learned from user interaction within the last hour. This brings the "Chronos" name to life.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Chronos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Chronos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Chronos is built on two revolutionary, brain-inspired pillars:

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are now structured with timestamps and source metadata, allowing for sophisticated, context-aware queries.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth, enabling it to solve complex, multi-step algorithmic problems where massive LLMs fail.

## Features

  - 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  - 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  - 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a Cosine Annealing LR schedule by default for more stable knowledge consolidation.
  - ⚡ **Accelerated Training with AMP**: Supports Automatic Mixed Precision (`--amp`) for faster training and reduced memory usage on compatible NVIDIA GPUs.
  - 🛡️ **Stable Training**: Built-in gradient clipping to prevent model instability and ensure smoother convergence.
  - 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config for easy sharing and use.
  - 💾 **Automatic Re-quantization**: After a learning session, Chronos can automatically re-quantize a model to persist the new knowledge.
  - 🌱 **Enhanced Model Expansion**: Includes `expand_model.py` script to transplant weights from smaller models to larger ones, now supporting changes in `max_length` and automatic length detection from datasets.
  - ✨ **Flexible Training Initiation**: Supports starting training runs using weights from existing model directories (inference or expanded models), not just resuming full training checkpoints.
  - ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`).
  - 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX) or on GPUs via Vulkan for broad hardware compatibility.
  - 🔧 **Comprehensive Tooling**: Includes a single script (`chronos.py`) for training, LoRA fine-tuning, merging, quantization, and interactive chat, plus the model expansion script.

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  - Python 3.8+
  - A C++ compiler (e.g., MSVC on Windows, GCC on Linux)
  - CMake (must be available in your system's `PATH`)
  - **For GPU Inference**: A Vulkan-compatible GPU and installed drivers.
  - **For GPU Training (AMP)**: An NVIDIA GPU with CUDA support (Compute Capability 7.0+ recommended for best performance) and a PyTorch build with CUDA enabled.
  - **For Developers (Re-compiling the Kernel)**: The official [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) must be installed to compile the kernel with Vulkan support.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/Chronos.git](https://github.com/your-username/Chronos.git)
    cd Chronos
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\Activate
    # On Linux/macOS
    source .venv/bin/activate
    ```

3.  **Run the setup script:** This will install Python dependencies and compile the C++ inference kernel.

    ```bash
    # On Windows
    setup.bat
    # On Linux/macOS
    bash setup.sh
    ```

    This will create a `chronos_matmul` library file in your project root. If this fails, you can try running `pip install .` from the project root.

-----

## 📚 User Guide: Step-by-Step Workflows

The `chronos.py` script is the main entry point for most operations. The `expand_model.py` script handles weight transplantation. All models are handled as directories.

### Basic Workflow (`chronos.py`)

1.  **Training (From Scratch):** Train a new model. This creates a new, self-contained model directory at `--out-dir`.

    ```bash
    python chronos.py train \
        --train "path/to/your_data.jsonl" \
        --tokenizer-path "microsoft/phi-2" \
        --out-dir "./my_chronos_model" \
        --epochs 5 \
        --batch_size 2 \
        --context_dim 512 \
        --max_h_steps 10 \
        --ponder-loss-weight 0.01 \
        --grad-clip 1.0 \
        --amp # <-- Optionally enable AMP
    ```

    -----

    💡 **Accelerating Training with AMP (NVIDIA GPUs):**
    If you are training on an NVIDIA GPU with CUDA support (Tensor Cores recommended, e.g., Volta, Turing, Ampere architecture or newer), you can enable **Automatic Mixed Precision (AMP)** using the `--amp` flag. AMP uses faster, lower-precision `float16` for many computations while maintaining model accuracy, significantly speeding up training and reducing VRAM usage. Make sure your PyTorch installation includes CUDA support.

    -----

2.  **Fine-Tuning (LoRA):** Adapt a pre-trained model using LoRA. This loads a full model directory and saves the adapter to a separate output directory.

    ```bash
    python chronos.py finetune \
        --model-path "./my_chronos_model" \
        --train "path/to/new_data.jsonl" \
        --out-dir "./my_lora_adapter" \
        --epochs 3 \
        --amp # <-- Optionally enable AMP
    ```

3.  **Merging a LoRA Adapter:** Merge the adapter back into the base model to create a new, standalone model directory.

    ```bash
    python chronos.py merge-lora \
        --model-path "./my_chronos_model" \
        --lora-adapter-path "./my_lora_adapter" \
        --out-dir "./my_model_merged"
    ```

4.  **Quantization:** Convert a full-precision model directory into a quantized model directory.

    ```bash
    python chronos.py quantize \
        --model-path "./my_model_merged" \
        --out-dir "./my_model_merged-INT4" \
        --qtype INT4
    ```

    > **Available `qtype`:** `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.

5.  **Inference (Chat Mode):** Run an interactive chat session.

    ```bash
    python chronos.py chat \
        --model-path "./my_model_merged-INT4"
    ```

### Resuming Incomplete Training (`chronos.py`)

If a training run was interrupted, you can resume it **exactly where it left off** using a training checkpoint (`chronos_epoch_*.pt` file) which contains the model weights, optimizer state, scheduler state, and epoch number.

```bash
python chronos.py train \
    --train "path/to/your_data.jsonl" `# Optional if path saved in ckpt` \
    --out-dir "./my_chronos_model" \
    --resume-from-ckpt "./my_chronos_model/chronos_epoch_3.pt" \
    --epochs 10 `# Set total desired epochs` \
    --amp # <-- Remember AMP if resuming an AMP session
````

#### Resuming Training with Modified Learning Rate

If you need to resume training from a checkpoint but want to **change the learning rate schedule** (e.g., start with a different maximum LR or use a different minimum LR), use the `--override-scheduling` flag along with your new `--starting-lr` and `--min-lr` values.

```bash
python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --out-dir "./my_chronos_model" \
    --epochs 10 \
    --resume-from-ckpt "./my_chronos_model/chronos_epoch_5.pt" \
    --starting-lr 1e-5 \
    --min-lr 1e-7 \
    --override-scheduling \
    --amp
```

> **Warning:** Without `--override-scheduling`, any new LR flags will be **ignored** when resuming, and the schedule from the checkpoint will be used.

### Expanding a Trained Model (`expand_model.py`) 🌱

Create a larger model by **transplanting the weights** from a smaller, trained Chronos model directory. This initializes the larger model with the learned knowledge.

1.  **Train a smaller base model** using `chronos.py`.

2.  **Use `expand_model.py`** to create the larger model directory, optionally changing dimensions or `max_length`.

    **Example:** Expanding dimensions and `max_length` (automatically detected):

    ```bash
    python expand_model.py \
        --old-model-path "./small_model_dir" \
        --output-dir "./large_model_dir" \
        --context_dim 1024 \
        --h_hidden 1024 \
        --l_hidden 1024 \
        --auto-max-length \
        --dataset-for-length "./larger_dataset.jsonl"
    ```

### Continuing Training from an Expanded or Inference Model (`chronos.py`) ✨

After expanding a model with `expand_model.py`, or if you simply want to **continue training using the weights from any existing Chronos model directory** (even a final inference model), you can use the `train` mode with the `--model-path` argument **instead of** `--resume-from-ckpt`.

This will load *only the model weights* from the specified directory and start a **completely new training session** (fresh optimizer, scheduler starting from epoch 1). This is useful for adapting a pre-trained or expanded model to a new dataset or continuing its training with different hyperparameters.

```bash
# Example: Continue training the expanded model from the previous step
python chronos.py train \
    --train "./larger_dataset.jsonl" \
    --model-path "./large_model_dir" `# Load weights from here` \
    --tokenizer-path "./large_model_dir" `# Load tokenizer from here too` \
    --out-dir "./large_model_continued_training" \
    --epochs 5 \
    --starting-lr 5e-5 `# Use a potentially smaller LR` \
    --min-lr 1e-6 \
    --amp # <-- Optionally enable AMP
```

> **Key Difference:**
>
>   * `--resume-from-ckpt`: Loads **full training state** (weights, optimizer, scheduler, epoch) from a `.pt` file to continue an interrupted run.
>   * `--model-path` (in `train` mode): Loads **only model weights** from a directory to start a *new* training run, initializing optimizer/scheduler fresh.

### Chat Mode Features (`chronos.py`)

#### Querying Structured Memory

Use `/filter time=-<seconds>` or `/filter source=<id>` (0=Unknown, 1=User, 2=Training) to constrain memory retrieval. Use `/filter reset` to clear filters.

#### Enabling Online Learning in Chat

Requires the quantized model (`--model-path`) and the original full-precision model (`--shadow-model-path`).

```bash
python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged" \
    --amp # <-- Optionally enable AMP for faster online learning
```

-----

## ⚙️ Command-Line Reference

### `chronos.py` Arguments

| Argument                | Mode(s)                             | Description                                                                     | Default           |
| :---------------------- | :---------------------------------- | :------------------------------------------------------------------------------ | :---------------- |
| **Paths** |                                     |                                                                                 |                   |
| `--model-path`          | `train`, `finetune`, `merge`, `quantize`, `chat` | Path to model directory. **[Train]**: Loads weights only (starts fresh training). **[Other]**: Loads for the specified mode. | `None`            |
| `--train`               | `train`, `finetune`                 | Path to the training `.json` or `.jsonl` file.                                  | `None`            |
| `--out-dir`             | `train`, `finetune`, `merge`, `quantize` | Directory to save new models, checkpoints, or adapters.                         | `./chronos_model` |
| `--tokenizer-path`      | `train`                             | Path or HF name of the tokenizer for a new model (or if overriding model's).      | `microsoft/phi-2` |
| `--resume-from-ckpt`    | `train`                             | Path to `.pt` checkpoint to **resume full training state** (optimizer, etc.).   | `None`            |
| `--shadow-model-path`   | `chat`                              | Path to full-precision model dir for online learning with quantized model.      | `None`            |
| `--lora-adapter-path`   | `merge`, `finetune`                 | Path to the trained LoRA adapter directory.                                     | `None`            |
| **Training/Fine-Tuning**|                                     |                                                                                 |                   |
| `--epochs`              | `train`, `finetune`                 | Number of training epochs.                                                      | `3`               |
| `--batch_size`          | `train`, `finetune`                 | Number of samples per forward pass.                                             | `4`               |
| `--accumulation-steps`  | `train`, `finetune`                 | Number of steps to accumulate gradients over (simulates larger batch size).     | `1`               |
| `--grad-clip`           | `train`, `finetune`                 | Gradient clipping value. Prevents gradient explosion (0 to disable).            | `1.0`             |
| `--ponder-loss-weight`  | `train`, `finetune`                 | Weight for the Ponder Cost auxiliary loss.                                      | `0.01`            |
| `--override-scheduling` | `train`                             | **[If resuming]** Ignore checkpoint's schedule state and use new LR args.         | `False`           |
| `--starting-lr`         | `train`, `finetune`                 | Max Learning Rate for the schedule, or fixed LR if schedule disabled.           | `1e-4`            |
| `--min-lr`              | `train`, `finetune`                 | Minimum Learning Rate for cosine annealing schedule.                            | `1e-6`            |
| `--disable-lr-schedule` | `train`, `finetune`                 | Use a fixed Learning Rate (`--starting-lr`) instead of cosine annealing.        | `False`           |
| `--ltm_lr`              | `train`, `finetune`, `chat`         | Learning Rate for LTM "surprise" updates (or max LR for LTM schedule in chat).  | `0.01`            |
| `--amp`                 | `train`, `finetune`, `chat`         | **Enable Automatic Mixed Precision (requires CUDA).** | `False`           |
| `--num_workers`         | `train`, `finetune`                 | Number of CPU workers for data loading.                                         | `0`               |
| `--lora_r`              | `finetune`                          | LoRA rank 'r'.                                                                  | `8`               |
| `--lora_alpha`          | `finetune`                          | LoRA alpha scaling factor.                                                      | `16`              |
| `--finetune-unlock-percent` | `finetune`                      | Target % of params to train (approx.). Overrides `--lora_r` if set.             | `None`            |
| `--kayla`               | `train`, `finetune`                 | Enable Kayla-style instruction tuning format (with thought-process).            | `False`           |
| **Quantization/Inference**|                                     |                                                                                 |                   |
| `--qtype`               | `quantize`, `train`                 | Quantization format (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). Used by `quantize` or `--quantize-on-complete`. | `INT4`            |
| `--quantize-on-complete`| `train`                             | Automatically run quantization after training finishes.                         | `False`           |
| `--device`              | `chat`                              | Device for *quantized* inference (`cpu`, `vulkan`).                             | `cpu`             |
| `--h-halt-thresh`       | `chat`                              | Probability threshold for early exiting the HRM reasoning loop during inference.  | `0.9`             |
| `--max-new-tokens`      | `chat`                              | Maximum number of tokens to generate in chat mode.                              | `512`             |
| `--enable-quantized-learning`| `chat`                         | Enable LTM updates for quantized models (requires `--shadow-model-path`).       | `False`           |
| `--ltm-lora-path`       | `chat`                              | Optional: Path to save/load LTM updates as a separate delta file in chat mode.  | `None`            |
| `--static-ltm-lr`       | `chat`                              | Disable cosine annealing for chat LTM updates, use fixed `--ltm_lr`.            | `False`           |
| `--ltm-schedule-steps`  | `chat`                              | Number of chat updates per LTM LR cosine cycle.                                 | `100`             |
| `--ltm-schedule-min-lr` | `chat`                              | Minimum LR for chat LTM cosine schedule.                                        | `1e-5`            |
| **Architecture (Train)**|                                     | *(Used only if starting train from scratch)* |                   |
| `--context_dim`         | `train`                             | Core embedding dimension.                                                       | `512`             |
| `--persistent_dim`      | `train`                             | Dimension of the fixed Persistent Memory.                                       | `128`             |
| `--ltm_slots`           | `train`                             | Number of slots in the Long-Term Memory.                                        | `2048`            |
| `--ltm_key_dim`         | `train`                             | Dimension of LTM keys.                                                          | `128`             |
| `--ltm_val_dim`         | `train`                             | Dimension of LTM values.                                                        | `128`             |
| `--h_hidden`            | `train`                             | Hidden size of the High-Level (CEO) RNN.                                        | `512`             |
| `--l_hidden`            | `train`                             | Hidden size of the Low-Level (Worker) RNN.                                      | `512`             |
| `--max_h_steps`         | `train`                             | Maximum number of reasoning steps the H-module can take per token.              | `10`              |
| `--max_l_steps`         | `train`                             | Maximum number of iterations for L-module convergence per H-step.               | `10`              |
| `--l_conv_atol`         | `train`                             | Absolute tolerance for checking L-module state convergence.                     | `1e-5`            |
| `--ltm_topk`            | `train`                             | Number of LTM slots to retrieve per token.                                      | `4`               |
| `--max_length`          | `train`                             | Maximum sequence length for positional embeddings.                              | `1024`            |
| `--auto-max-length`     | `train`                             | Automatically scan `--train` dataset to set `max_length`.                       | `False`           |
| **Other** |                                     |                                                                                 |                   |
| `--threads`             | `All`                               | Number of CPU threads for PyTorch/OpenMP.                                       | `CPU_Count/2`     |

### `expand_model.py` Arguments

| Argument               | Description                                                                          | Required | Default |
| :--------------------- | :----------------------------------------------------------------------------------- | :------- | :------ |
| `--old-model-path`     | Path to the trained smaller model *directory*.                                       | Yes      |         |
| `--output-dir`         | Path to save the new, expanded model *directory*.                                    | Yes      |         |
| `--context_dim`        | *Optional:* New context dimension.                                                   | No       |         |
| `--persistent_dim`     | *Optional:* New persistent memory dimension.                                         | No       |         |
| `--ltm_slots`          | *Optional:* New number of LTM slots.                                                 | No       |         |
| `--ltm_key_dim`        | *Optional:* New LTM key dimension.                                                   | No       |         |
| `--ltm_val_dim`        | *Optional:* New LTM value dimension.                                                 | No       |         |
| `--h_hidden`           | *Optional:* New H-RNN hidden size.                                                   | No       |         |
| `--l_hidden`           | *Optional:* New L-RNN hidden size.                                                   | No       |         |
| `--new-max-length`     | *Optional:* Manually specify the new maximum sequence length.                        | No       |         |
| `--auto-max-length`    | *Optional:* Automatically determine `new-max-length` by scanning a dataset.          | No       | `False` |
| `--dataset-for-length` | Path to dataset (.jsonl/.json) required if using `--auto-max-length`.                | If above |         |
| `--kayla`              | Use Kayla formatting when scanning dataset with `--auto-max-length`.                 | No       | `False` |

-----

## Roadmap

  - [ ] Develop a user-friendly GUI wrapper for easier interaction.
  - [ ] Extend the architecture to support multi-modal inputs (images, audio).
  - [ ] Implement the entire training loop in Vulkan/CUDA for end-to-end GPU acceleration.

## License

The source code of Chronos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See `LICENSE.md` for full details.

## Support This Project

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Chronos in my spare time while working full-time at a grocery store.

**[https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)**

## Acknowledgements

  - This architecture is inspired by the concepts in Google's **Titans** and Sapient Intelligence's **HRM** papers.
  - The quantization kernel design is heavily influenced by the groundbreaking work in **llama.cpp**.
  - **pybind11** for seamless C++/Python integration.

## Changelog

### v0.5.2 (alpha)

  - **Added Flexible Training Initiation**: The `train` mode now supports a `--model-path` argument to load *only weights* from an existing model directory (e.g., an expanded model or inference checkpoint) and start a *new* training session (fresh optimizer/scheduler). This is distinct from `--resume-from-ckpt` which loads the full training state.
  - **Enhanced `expand_model.py` Script**:
      - Added ability to expand `max_length` by correctly handling positional embeddings.
      - Added `--new-max-length` flag for manual specification.
      - Added `--auto-max-length` and `--dataset-for-length` flags to automatically detect required `max_length` from a dataset.
      - Automatically copies tokenizer files to the expanded model directory.
  - **Added Automatic Mixed Precision (AMP)**: Implemented `--amp` flag for `train`, `finetune`, and `chat` (online learning) modes to accelerate computation and reduce memory usage on NVIDIA CUDA GPUs.
  - **Documentation**: Updated README with detailed usage for the enhanced `expand_model.py`, AMP, and the new training initiation workflow using `--model-path`. Updated command reference table.

### v0.5.1 (alpha)

  - **Added `--override-scheduling` flag**: Allows users to force new learning rate schedule settings (`--starting-lr`, `--min-lr`) when resuming training from a checkpoint.
  - **Documentation**: Added usage instructions for `--override-scheduling`.

### v0.5 (alpha)

  - **Implemented Structured Long-Term Memory**: Memory slots are now augmented with timestamps and source metadata, enabling temporal and source-based filtering during chat.
  - **Implemented Adaptive Reasoning Depth (Ponder Time)**: The HRM's reasoning depth is now dynamic. The model learns to "think longer" for complex tokens and halt early on simple ones.
  - **Added Ponder Cost**: A new auxiliary loss (`--ponder-loss-weight`) trains the model to be computationally efficient.
  - **Added Halting Threshold**: A new inference flag (`--h-halt-thresh`) allows users to control the trade-off between speed and reasoning depth during chat.

### v0.4 (alpha)

  - **Implemented Dynamic LTM Learning Rate**: Online learning now defaults to a `CosineAnnealingLR` schedule for more stable knowledge consolidation.
  - **Added Static LR Fallback**: The `--static-ltm-lr` flag can be used in chat mode to revert to a fixed LTM learning rate.
  - **Added Gradient Clipping**: The `--grad-clip` argument was added to `train` and `finetune` modes to improve training stability.

-----

© 2025 Makhi Burroughs
