# Chronos v0.2 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

**License:** Custom (Free for non-commercial use) | **Python:** 3.8+

-----

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Chronos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Chronos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

### Core Concepts

Chronos is built on two revolutionary, brain-inspired pillars:

  * 🧠 **Titans Architecture (The Cognitive Substrate):** A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what* to remember based on the principle of "surprise," allowing it to consolidate new knowledge at inference time without catastrophic forgetting.

  * ⚙️ **Hierarchical Reasoning Model (The Cognitive Process):** A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth, enabling it to solve complex, multi-step algorithmic problems where massive LLMs fail.

By combining a system that learns what to remember with a system that learns how to reason, Chronos represents a tangible plan for the next generation of AI.

-----

## Features

  * **Hybrid Memory-Reasoning:** Deeply integrates a lifelong memory system with a hierarchical reasoning engine.
  * **Continual "Online" Learning:** Learns and updates its Long-Term Memory (LTM) at inference time based on new experiences.
  * **Automatic Re-quantization:** After a learning session with a quantized model, Chronos can automatically re-quantize the model to persist the new knowledge.
  * **High-Performance Inference:** Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`).
  * **CPU & GPU Support:** Runs fast quantized inference on standard CPUs (with AVX) or on GPUs via **Vulkan** for broad hardware compatibility.
  * **PEFT-Compliant Fine-Tuning:** Uses the Hugging Face `peft` library for efficient **LoRA** fine-tuning, with intuitive controls for trainable parameter percentage.
  * **Comprehensive Tooling:** Includes a single, powerful script for:
      * Training from scratch
      * Chain-of-Thought training via **Kayla Mode**
      * Efficient **LoRA** fine-tuning
      * Merging LoRA adapters
      * One-command quantization
      * Interactive chat with **Ctrl+X interrupt** and optional online learning

-----

## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  * Python 3.8+
  * A C++ compiler (e.g., MSVC on Windows, GCC on Linux)
  * CMake (must be available in your system's PATH)
  * **For GPU Inference:** A Vulkan-compatible GPU and installed drivers.
  * **For Developers (Re-compiling the Kernel):** To compile the kernel with Vulkan support, the official **[Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)** must be installed. This provides the `glslc` compiler needed to build the GPU shaders. **End-users running the pre-compiled kernel do not need the SDK.**

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/necat101/Chronos.git
    cd chronos
    ```

2.  **Create a virtual environment (recommended):**

    ```sh
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\Activate
    # On Linux/macOS
    source .venv/bin/activate
    ```

3.  **Run the setup script:** This will install Python dependencies and compile the C++ kernel.

    ```sh
    # On Windows
    setup.bat
    # On Linux/macOS
    bash setup.sh
    ```

    This will create a `chronos_matmul` library file in your project root, which `chronos.py` depends on.

-----

## User Guide

The `chronos.py` script is the main entry point for all operations.

### 1\. Training

Train a new model from scratch on a JSON or JSONL dataset.

```sh
python chronos.py train --train "path/to/your_data.jsonl" --epochs 10 --batch_size 1 --accumulation-steps 8 --starting-lr 1e-4 --min-lr 1e-6 --out-dir "./my_model"
```
Note: The tokenizer is not locked to `phi-2`. You can specify any compatible model from the Hugging Face Hub with the `--tokenizer` argument in your command.

#### Kayla Mode Training (Chain-of-Thought and feelings)

To train the model to produce a "thought process" and "feelings" before its final response, use **Kayla Mode**. This requires a dataset with four fields: `Instruction`, `thought-process`, 'feelings' and `output`.

**Example `kayla_data.jsonl` entry:**

```json
{"Instruction": "what is 3 + 3", "thought-process": "the human is asking me to solve a literal mathematical equation, 3 + 3 = 6", "feelings":"{Happy:950,00000.0}", "output": "6" }

```

Enable Kayla Mode by simply adding the `--kayla` flag to the training command:

```sh
python chronos.py train --train "path/to/kayla_data.jsonl" --kayla --epochs 5 --out-dir "./my_kayla_model"
```

#### Resuming Training

If your training is interrupted, you can resume from the last saved epoch checkpoint:

```sh
python chronos.py train --train "path/to/your_data.jsonl" --resume-from-ckpt "./my_model/chronos_epoch_5.pt"
```

-----

### 2\. Fine-Tuning (LoRA)

Efficiently adapt a pre-trained model to a new dataset using Low-Rank Adaptation (LoRA).

```sh
python chronos.py finetune --base-ckpt "./my_model/chronos_final.pt" --train "path/to/new_data.jsonl" --epochs 3 --out-dir "./my_lora_adapter"
```

  * You can intuitively control the size of the LoRA adapter with `--finetune-unlock-percent 1.5` (for 1.5% trainable parameters).

-----

### 3\. Merging a LoRA Adapter

To create a new, standalone model from your base model and the LoRA adapter, use the `merge-lora` mode.

```sh
python chronos.py merge-lora --base-ckpt "./my_model/chronos_final.pt" --lora-adapter-path "./my_lora_adapter" --merged-out-path "./my_model_merged.pt"
```

-----

### 4\. Quantization

Convert your trained `.pt` model to a much smaller and faster `.npz` file. The model's architecture is now saved inside the checkpoint, so you no longer need to specify it manually\!

```sh
python chronos.py quantize --ckpt "./my_model/chronos_final.pt" --qtype INT4
```

Available quantization types (`--qtype`): `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.

-----

### 5\. Inference (Chat Mode)

Run an interactive chat session with your model. You can press **`Ctrl+X`** at any time to interrupt the model's generation.

#### Running the Quantized Model (Recommended)

Model configuration is loaded automatically from the `.npz` file.

  * **On CPU:**
    ```sh
    python chronos.py chat --load-quantized "./my_model/chronos_final-INT4.npz"
    ```
  * **On GPU (Vulkan):**
    ```sh
    python chronos.py chat --load-quantized "./my_model/chronos_final-INT4.npz" --device vulkan
    ```

#### Enabling Online Learning in Chat

To allow the model to learn from your conversation, enable online learning.

  * **Modify & Re-quantize (Recommended):** This method updates the model in memory and then asks if you want to **save the changes by re-quantizing the model** upon exit. This is the easiest way to make your model permanently smarter.

    ```sh
    python chronos.py chat --load-quantized "./my_model/chronos_final-INT4.npz" --enable-quantized-learning --ckpt "./my_model/chronos_final.pt"
    ```

    *Note: `--enable-quantized-learning` requires providing the original full-precision `--ckpt` to use as a "shadow model" for calculating gradients.*

  * **Save Updates Separately:** To save only the LTM updates without modifying the base model file, use `--ltm-lora-path`. This is useful for experimenting with different memory states.

    ```sh
    python chronos.py chat --load-quantized "./my_model/chronos_final-INT4.npz" --enable-quantized-learning --ckpt "./my_model/chronos_final.pt" --ltm-lora-path "./my_ltm_updates.pt"
    ```

    The updates will be saved to `my_ltm_updates.pt` when you quit the chat, and automatically loaded the next time you run the same command.

-----

## Command-Line Reference

### Main Modes

| Mode | Description |
| :--- | :--- |
| `train` | Train a new model from scratch. |
| `finetune` | Apply LoRA fine-tuning to an existing model. |
| `merge-lora` | Merge a LoRA adapter into a base model. |
| `quantize` | Convert a `.pt` model to a quantized `.npz` model. |
| `chat` | Run an interactive chat session. |

### Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| **Paths** | | |
| `--train` | Path to the training `.json` or `.jsonl` file. | `None` |
| `--out-dir`| Directory to save checkpoints and adapters. | `./chronos_checkpoints` |
| `--ckpt`| Path to a full-precision `.pt` model. | `None` |
| `--load-quantized` | Path to a quantized `.npz` model for chat. | `None` |
| **Training** | | |
| `--epochs` | Number of training epochs. | `3` |
| `--batch_size` | Number of samples per batch. | `4` |
| `--accumulation-steps` | Simulate a larger batch size by accumulating gradients. | `1` |
| `--starting-lr` | The maximum learning rate for the scheduler. | `1e-4` |
| `--min-lr` | The minimum learning rate for the scheduler. | `1e-6` |
| `--disable-lr-schedule` | Use a fixed learning rate instead of a scheduler. | `False` |
| `--kayla` | Enable Chain-of-Thought style training. | `False` |
| `--auto-max-length`| Automatically scan the dataset to set `max_length`. | `False` |
| **LoRA** | | |
| `--lora_r` | LoRA attention dimension (rank). | `8` |
| `--finetune-unlock-percent` | Target percentage of trainable parameters for LoRA. | `None` |
| **Quantization & Inference** | | |
| `--qtype`| Quantization format. Options: `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.| `INT4` |
| `--device` | Device for quantized inference. Options: `cpu`, `vulkan`. | `cpu` |
| `--enable-quantized-learning` | Enable LTM updates for quantized models during chat. | `False` |
| **Model Architecture (for new models)** | | |
| `--max_length` | Maximum sequence length. | `1024` |
| `--context_dim` | The main embedding dimension of the model. | `512` |
| `--h_hidden` | Hidden size of the high-level HRM RNN. | `512` |
| `--l_hidden` | Hidden size of the low-level HRM RNN. | `512` |

-----

## Roadmap

  * [ ] Develop a user-friendly GUI wrapper for easier interaction.
  * [ ] Extend the architecture to support multi-modal inputs (images, audio).
  * [ ] Implement the entire training and inference loop in Vulkan/CUDA for end-to-end GPU acceleration.

## License

The source code of Chronos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement.

See `LICENSE.md` for full details.

Please donate to my patreon to help with my funding situation, I have motor cortex damage and cant get a job at a datacenter and I currently work at a grocery store full time and work on this on the side: [https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)

## Acknowledgements

  * This architecture is inspired by the concepts presented in Google's Titans and Sapient Intelligence's HRM papers.
  * The quantization kernel design is heavily influenced by the groundbreaking work in `llama.cpp`.
  * [pybind11](https://github.com/pybind/pybind11) for seamless C++/Python integration.

-----

v0.2 alpha changelog:

implemented ctrl + x to interrupt model generation
fixed fine-tuning to be compliant with the existing PEFT codebase
fixed 'kayla' mode to be fully compliant
added automatic requantization of quantized models in online learning mode
implemented automatic hyperparameter detection at inference and quantization time to prevent the need for manual specification every time you want to inference!

© 2025 Makhi Burroughs
