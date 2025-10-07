# Chronos v0.3 (alpha): A Hybrid Memory-Reasoning Architecture
A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

License: Custom (Free for non-commercial use) | Python: 3.8+

# What's New in v0.3
Major Change: Directory-Based Models!
To permanently fix tokenizer mismatch errors and make models truly portable, Chronos now saves and loads models as self-contained directories. Each directory includes both the model weights (.pt or .npz) and the necessary tokenizer files.

This is a critical change that simplifies the workflow and improves reliability. You will now use the --model-path argument to point to these directories instead of the old file-based flags like --ckpt and --load-quantized.

# About The Project
The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Chronos challenges this paradigm by focusing on architectural intelligence.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Chronos is conceived not merely to process information, but to think, learn, and remember in a cohesive, integrated, and human-like manner.

Core Concepts
Chronos is built on two revolutionary, brain-inspired pillars:

🧠 Titans Architecture (The Cognitive Substrate): A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns what to remember based on the principle of "surprise," allowing it to consolidate new knowledge at inference time without catastrophic forgetting.

⚙️ Hierarchical Reasoning Model (The Cognitive Process): A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth, enabling it to solve complex, multi-step algorithmic problems where massive LLMs fail.

By combining a system that learns what to remember with a system that learns how to reason, Chronos represents a tangible plan for the next generation of AI.

Features
Self-Contained Model Packages: Models are saved as directories containing both weights and tokenizer, eliminating mismatch errors.

Hybrid Memory-Reasoning: Deeply integrates a lifelong memory system with a hierarchical reasoning engine.

Continual "Online" Learning: Learns and updates its Long-Term Memory (LTM) at inference time based on new experiences.

Automatic Re-quantization: After a learning session, Chronos can automatically re-quantize a model to persist the new knowledge.

High-Performance Inference: Utilizes a custom C++ kernel inspired by llama.cpp for state-of-the-art quantization (INT4, Q4_0, Q8_0, Q2_K).

CPU & GPU Support: Runs fast quantized inference on standard CPUs (with AVX) or on GPUs via Vulkan for broad hardware compatibility.

Comprehensive Tooling: Includes a single, powerful script for training, LoRA fine-tuning, merging, quantization, and interactive chat.

# Getting Started:
Follow these steps to get a local copy up and running.

Prerequisites
Python 3.8+

A C++ compiler (e.g., MSVC on Windows, GCC on Linux)

CMake (must be available in your system's PATH)

For GPU Inference: A Vulkan-compatible GPU and installed drivers.

For Developers (Re-compiling the Kernel): To compile the kernel with Vulkan support, the official Vulkan SDK must be installed. This provides the glslc compiler needed to build the GPU shaders.

Installation
Clone the repository:



git clone https://github.com/necat101/Chronos.git
cd chronos
Create a virtual environment (recommended):



python -m venv .venv
# On Windows
.\.venv\Scripts\Activate
# On Linux/macOS
source .venv/bin/activate
Run the setup script: This will install Python dependencies and compile the C++ kernel.



# On Windows
setup.bat
# On Linux/macOS
bash setup.sh
This will create a chronos_matmul library file in your project root.

User Guide
The chronos.py script is the main entry point for all operations. All models are now handled as directories.

1. Training
Train a new model from scratch. This will create a new model directory at --out-dir containing the weights and tokenizer.



python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "microsoft/phi-2" \
    --out-dir "./my_chronos_model" \
    --epochs 5 \
    --batch_size 2


Kayla Mode Training (Chain-of-Thought and feelings)
Enable Kayla Mode by adding the --kayla flag. The dataset needs Instruction, thought-process, feelings, and output fields.



python chronos.py train --train "path/to/kayla_data.jsonl" --kayla --out-dir "./my_kayla_model"


Resuming Training
If your training is interrupted, you can resume by pointing to the model directory containing the partial checkpoints.



python chronos.py train --train "path/to/your_data.jsonl" --resume-from-model-path "./my_chronos_model"


2. Fine-Tuning (LoRA)
Adapt a pre-trained model using LoRA. This loads a full model directory and saves the adapter to a separate output directory.



python chronos.py finetune \
    --model-path "./my_chronos_model" \
    --train "path/to/new_data.jsonl" \
    --out-dir "./my_lora_adapter" \
    --epochs 3

    
You can control the adapter size with --finetune-unlock-percent 1.5 (for 1.5% trainable parameters).

3. Merging a LoRA Adapter
Merge the adapter back into the base model to create a new, standalone model directory.



python chronos.py merge-lora \
    --model-path "./my_chronos_model" \
    --lora-adapter-path "./my_lora_adapter" \
    --out-dir "./my_model_merged"


4. Quantization
Convert a full-precision model directory into a quantized model directory.



python chronos.py quantize \
    --model-path "./my_model_merged" \
    --out-dir "./my_model_merged-INT4" \
    --qtype INT4

    
The output directory will contain the .npz file and a copy of the essential tokenizer files. Available qtype: INT4, Q4_0, Q8_0, Q2_K.

5. Inference (Chat Mode)
Run an interactive chat session by pointing to any model directory (quantized or full-precision). You can press Ctrl+X at any time to interrupt generation.

Running a Quantized Model (Recommended)
On CPU:



python chronos.py chat --model-path "./my_model_merged-INT4"
On GPU (Vulkan):



python chronos.py chat --model-path "./my_model_merged-INT4" --device vulkan


Enabling Online Learning in Chat
Allow the model to learn from your conversation.

Modify & Re-quantize (Recommended): This method updates the model in memory and then asks if you want to save the changes by re-quantizing upon exit. This makes your model permanently smarter.



python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged"

    
Note: Online learning requires the quantized model (--model-path) and the original full-precision model (--shadow-model-path).

Save Updates Separately (LoRA-style): To save only the LTM updates without modifying the base model, use --ltm-lora-path.



python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged" \
    --ltm-lora-path "./my_ltm_updates.pt"

    
The memory updates will be saved to my_ltm_updates.pt when you quit.

Command-Line Reference
Main Modes
Mode	Description
train	Train a new model from scratch.
finetune	Apply LoRA fine-tuning to an existing model.
merge-lora	Merge a LoRA adapter into a base model.
quantize	Convert a model directory to a quantized version.
chat	Run an interactive chat session.

Export to Sheets
Key Arguments
Argument	Description	Default
Paths		
--model-path	Path to the model directory (for finetune, merge, quantize, chat).	None
--train	Path to the training .json or .jsonl file.	None
--out-dir	Directory to save new models, checkpoints, or adapters.	./chronos_model
--tokenizer-path	[Train] Path or HF name of the tokenizer for a new model.	microsoft/phi-2
--shadow-model-path	[Chat] Path to the original full-precision model directory for online learning.	None
Training & Fine-Tuning		
--epochs	Number of training epochs.	3
--batch_size	Number of samples per batch.	4
--accumulation-steps	Accumulate gradients to simulate a larger batch size.	1
--starting-lr	The maximum learning rate for the scheduler.	1e-4
--kayla	Enable Chain-of-Thought style training.	False
--finetune-unlock-percent	[Finetune] Target percentage of trainable parameters for LoRA.	None
Quantization & Inference		
--qtype	Quantization format. Options: INT4, Q4_0, Q8_0, Q2_K.	INT4
--device	Device for quantized inference. Options: cpu, vulkan.	cpu
--enable-quantized-learning	Enable LTM updates for quantized models during chat.	False
Model Architecture (for new models)		
--max_length	Maximum sequence length.	1024
--context_dim	The main embedding dimension of the model.	512

Export to Sheets
Roadmap
[ ] Develop a user-friendly GUI wrapper for easier interaction.

[ ] Extend the architecture to support multi-modal inputs (images, audio).

[ ] Implement the entire training loop in Vulkan/CUDA for end-to-end GPU acceleration.

License
The source code of Chronos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See LICENSE.md for full details.

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Chronos in my spare time while working full-time at a grocery store: https://www.patreon.com/cw/MakhiBurroughs

Acknowledgements
This architecture is inspired by the concepts in Google's Titans and Sapient Intelligence's HRM papers.

The quantization kernel design is heavily influenced by the groundbreaking work in llama.cpp.

pybind11 for seamless C++/Python integration.

v0.3 alpha changelog:
Refactored to a directory-based model system to bundle weights and tokenizer files together.

Fixed critical tokenizer mismatch bug that caused gibberish output during inference.

Simplified command-line interface by unifying model loading under the --model-path argument.

Deprecated old file-based flags (--ckpt, --load-quantized, etc.) in favor of the new, more robust system.

© 2025 Makhi Burroughs
