import json
from transformers import AutoTokenizer
import sys
import os
from tqdm import tqdm
import torch
import argparse # <<< Import argparse

# --- 1. Define Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Chunk a JSONL dataset for Chronos training, saving chunks as .pt files.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Path to the input JSONL dataset file (e.g., train.jsonl).")
parser.add_argument("--tokenizer-path", type=str, default="openai-community/gpt2",
                    help="Path or Hugging Face name of the tokenizer to use.")
parser.add_argument("--overlap", type=int, default=1024,
                    help="Number of tokens to overlap between consecutive chunks.")
parser.add_argument("--output-dir", type=str, default="train_chronos_chunked_tensors",
                    help="Directory to save the output .pt chunk files and manifest.jsonl.")
# --- Internal constants (could be args later if needed) ---
RESERVED_CHUNK_SPACE = 2048 # Minimum size reserved for the chunkable part
ANCHOR_SAFETY_MARGIN = 16 # Extra tokens added to the longest thought process

# --- Parse Arguments ---
args = parser.parse_args()

# --- Use arguments instead of static parameters ---
OVERLAP_TOKENS = args.overlap
TOKENIZER_PATH = args.tokenizer_path
DATASET_FILE = args.dataset
OUTPUT_DIR = args.output_dir

# --- 2. Load Tokenizer ---
print(f"Loading tokenizer '{TOKENIZER_PATH}'...")
try:
    # Use a large model_max_length during loading to avoid truncation warnings
    # The actual sequence length used for chunking is determined dynamically later
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, model_max_length=999999, trust_remote_code=True)
except Exception as e:
    print(f"ERROR: Failed to load tokenizer '{TOKENIZER_PATH}'. {e}")
    sys.exit(1)

if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token ('{tokenizer.pad_token}')")
    else:
        print("Warning: Tokenizer missing both pad and eos tokens. Adding '[PAD]' token.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

PAD_TOKEN_ID = tokenizer.pad_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id

print("---" * 10)

# --- 3. Pass 1: Analyze Dataset ---
print(f"Starting Pass 1: Analyzing '{DATASET_FILE}' to find longest thought-process...")
max_thought_len_tokens = 0
total_samples_analyzed = 0
thought_start_wrapper = "### Thought Process:\n"
thought_end_wrapper = "\n\n"

try:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            total_samples_analyzed += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                if (line_num + 1) % 1000 == 0:
                    print(f"Skipping malformed JSON line ~{line_num+1} in Pass 1...")
                continue

            thought_content_str = sample.get('thought-process', '')
            if not isinstance(thought_content_str, str): thought_content_str = ''

            # Tokenize parts separately to accurately measure length
            thought_start_tokens = tokenizer.encode(thought_start_wrapper, add_special_tokens=False)
            thought_content_tokens = tokenizer.encode(thought_content_str, add_special_tokens=False)
            thought_end_tokens = tokenizer.encode(thought_end_wrapper, add_special_tokens=False)
            current_thought_len = len(thought_start_tokens) + len(thought_content_tokens) + len(thought_end_tokens)

            if current_thought_len > max_thought_len_tokens:
                max_thought_len_tokens = current_thought_len

except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_FILE}'")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during Pass 1: {e}")
    sys.exit(1)

print(f"Analysis Complete: Processed {total_samples_analyzed} samples.")
if max_thought_len_tokens == 0:
    print("Warning: No valid 'thought-process' data found. Setting minimum anchor budget.")
    # Calculate a minimum based just on wrappers + a small buffer
    max_thought_len_tokens = len(tokenizer.encode(thought_start_wrapper, add_special_tokens=False)) + \
                             len(tokenizer.encode(thought_end_wrapper, add_special_tokens=False)) + 5

print("---" * 10)

# --- 4. Dynamically Set Final Parameters ---
MAX_ANCHOR_BUDGET = max_thought_len_tokens + ANCHOR_SAFETY_MARGIN
MIN_CHUNKABLE_TOKENS = RESERVED_CHUNK_SPACE
# Final max sequence length = Anchor budget + Reserved space for chunk content + 1 for EOS token
MAX_SEQ_LENGTH = MAX_ANCHOR_BUDGET + MIN_CHUNKABLE_TOKENS + 1

# Validate overlap against the chunkable space
if OVERLAP_TOKENS >= MIN_CHUNKABLE_TOKENS:
      print(f"WARNING: Specified overlap ({OVERLAP_TOKENS}) is >= RESERVED_CHUNK_SPACE ({MIN_CHUNKABLE_TOKENS}).")
      # Reduce overlap automatically to prevent issues
      new_overlap = max(1, MIN_CHUNKABLE_TOKENS // 2)
      print(f"         Reducing overlap to {new_overlap} to ensure forward progress.")
      OVERLAP_TOKENS = new_overlap

print(f"Longest thought-process: {max_thought_len_tokens} tokens")
print(f"Anchor safety margin: {ANCHOR_SAFETY_MARGIN} tokens")
print(f"Effective Anchor Budget (MAX_ANCHOR_BUDGET): {MAX_ANCHOR_BUDGET} tokens")
print(f"Reserved chunk space (MIN_CHUNKABLE_TOKENS): {MIN_CHUNKABLE_TOKENS} tokens")
print(f"Overlap set to: {OVERLAP_TOKENS} tokens")
print(f"FINAL MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH} tokens")
print("---" * 10)

# --- 5. Pass 2: Process and Chunk the Dataset ---
print(f"Starting Pass 2: Chunking dataset '{DATASET_FILE}' to {MAX_SEQ_LENGTH} tokens...")
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
manifest_path = os.path.join(OUTPUT_DIR, "manifest.jsonl")

original_sample_count = 0
skipped_samples = 0
total_chunks_created = 0

# Define wrappers for easier access
inst_start_wrapper = "### Instruction:\n"
inst_end_wrapper = "\n\n"
feel_start_wrapper = "### Feelings:\n"
feel_end_wrapper = "\n\n"
output_start_wrapper = "### Response:\n"

try:
    # Open input dataset and output manifest file
    with open(DATASET_FILE, "r", encoding="utf-8") as f_in, \
         open(manifest_path, "w", encoding="utf-8") as f_manifest:
        # Wrap the file reading with tqdm for progress bar
        for sample_idx, line in enumerate(tqdm(f_in, desc="Chunking Samples", unit="sample")):
            original_sample_count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                if (sample_idx + 1) % 1000 == 0:
                    print(f"Skipping malformed JSON line ~{sample_idx+1} in Pass 2...")
                skipped_samples += 1
                continue

            # --- 6. Extract Content ---
            inst_content = sample.get("Instruction", "")
            output_content = sample.get("output", "")
            thought_content_str = sample.get('thought-process', '')
            feelings_content_str = sample.get('feelings', '')

            # Ensure content are strings
            if not isinstance(inst_content, str): inst_content = ""
            if not isinstance(output_content, str): output_content = ""
            if not isinstance(thought_content_str, str): thought_content_str = ""
            if not isinstance(feelings_content_str, str): feelings_content_str = ""

            # --- 7. Build the STATIC ANCHOR (Thought Process) ---
            thought_start_tokens = tokenizer.encode(thought_start_wrapper, add_special_tokens=False)
            thought_content_tokens = tokenizer.encode(thought_content_str, add_special_tokens=False)
            thought_end_tokens = tokenizer.encode(thought_end_wrapper, add_special_tokens=False)
            current_thought_len = len(thought_start_tokens) + len(thought_content_tokens) + len(thought_end_tokens)

            # Truncate thought content if it exceeds the calculated budget
            if current_thought_len > MAX_ANCHOR_BUDGET:
                available_for_thought_content = MAX_ANCHOR_BUDGET - len(thought_start_tokens) - len(thought_end_tokens)
                if available_for_thought_content <= 0:
                    # Skip if even wrappers don't fit
                    skipped_samples += 1
                    continue
                thought_content_tokens = thought_content_tokens[:available_for_thought_content]

            anchor_tokens = thought_start_tokens + thought_content_tokens + thought_end_tokens
            anchor_len = len(anchor_tokens)

            # --- 8. Build the CONTINUOUS CHUNKABLE STREAM (Instruction, Feelings, Output) ---
            inst_tokens = tokenizer.encode(inst_start_wrapper + inst_content + inst_end_wrapper, add_special_tokens=False)
            feelings_tokens = []
            if feelings_content_str: # Only include feelings section if content exists
                feelings_tokens = tokenizer.encode(feel_start_wrapper + feelings_content_str + feel_end_wrapper, add_special_tokens=False)
            output_tokens = tokenizer.encode(output_start_wrapper + output_content, add_special_tokens=False)

            chunkable_tokens = inst_tokens + feelings_tokens + output_tokens
            # Calculate length of the prompt part within the chunkable stream for later masking
            prompt_len_in_stream = len(inst_tokens) + len(feelings_tokens)

            # --- 9. Chunking Loop ---
            # Calculate how many tokens are available for the chunkable content in each chunk
            available_length_for_chunk = MAX_SEQ_LENGTH - anchor_len - 1 # -1 for EOS

            # Skip sample if anchor is too long, leaving no or insufficient space for chunking
            if available_length_for_chunk <= OVERLAP_TOKENS:
                skipped_samples += 1
                continue
            if available_length_for_chunk <= 0:
                 skipped_samples += 1
                 continue


            start_idx = 0
            chunk_id = 0
            while start_idx < len(chunkable_tokens):
                end_idx = start_idx + available_length_for_chunk
                chunk_content_tokens = chunkable_tokens[start_idx:end_idx]

                # Handle edge case: if chunkable_tokens is very short and fits entirely
                if not chunk_content_tokens and start_idx == 0 and len(chunkable_tokens) > 0:
                    chunk_content_tokens = chunkable_tokens
                    end_idx = len(chunkable_tokens)
                elif not chunk_content_tokens and start_idx > 0: # Avoid infinite loop if step size becomes 0
                    break

                # --- Construct final input_ids LIST ---
                input_ids_list = anchor_tokens + chunk_content_tokens + [EOS_TOKEN_ID]

                # --- Padding ---
                current_len = len(input_ids_list)
                padding_length = MAX_SEQ_LENGTH - current_len
                if padding_length < 0:
                    # This shouldn't happen with correct available_length_for_chunk calculation, but good safeguard
                    input_ids_list = input_ids_list[:MAX_SEQ_LENGTH]
                    chunk_content_len = MAX_SEQ_LENGTH - anchor_len - 1
                    chunk_content_tokens = chunk_content_tokens[:chunk_content_len] if chunk_content_len > 0 else []
                    padding_length = 0
                elif padding_length > 0:
                    input_ids_list.extend([PAD_TOKEN_ID] * padding_length)

                # --- Construct labels LIST with masking ---
                # Start by copying input_ids, then mask specific parts
                labels_list = list(input_ids_list) # Make a copy

                # Mask 1: The entire anchor (thought process)
                labels_list[:anchor_len] = [-100] * anchor_len

                # Mask 2: The prompt part (Instruction + Feelings) within the current chunk
                # Calculate start/end of prompt relative to the *chunk content*
                chunk_rel_prompt_start = max(0, prompt_len_in_stream - start_idx)
                chunk_rel_prompt_end = min(len(chunk_content_tokens), prompt_len_in_stream - start_idx)
                if chunk_rel_prompt_start < chunk_rel_prompt_end: # Check if any part of the prompt is in this chunk
                    # Calculate indices relative to the *full* labels list
                    mask_start_index = anchor_len + chunk_rel_prompt_start
                    mask_end_index = anchor_len + chunk_rel_prompt_end
                    labels_list[mask_start_index:mask_end_index] = [-100] * (mask_end_index - mask_start_index)

                # Mask 3: The overlapping part in subsequent chunks (excluding the first chunk)
                if chunk_id > 0:
                    overlap_mask_start = anchor_len
                    overlap_mask_end = min(anchor_len + OVERLAP_TOKENS, anchor_len + len(chunk_content_tokens))
                    if overlap_mask_start < overlap_mask_end:
                         labels_list[overlap_mask_start:overlap_mask_end] = [-100] * (overlap_mask_end - overlap_mask_start)

                # Mask 4: The EOS token and any padding tokens
                eos_and_padding_start_index = anchor_len + len(chunk_content_tokens)
                # Ensure index doesn't go out of bounds if list was truncated
                eos_and_padding_start_index = min(eos_and_padding_start_index, MAX_SEQ_LENGTH)
                labels_list[eos_and_padding_start_index:] = [-100] * (MAX_SEQ_LENGTH - eos_and_padding_start_index)


                # --- Construct attention mask LIST ---
                # Attention mask is 1 for all non-padding tokens (anchor + content + EOS)
                valid_token_count = anchor_len + len(chunk_content_tokens) + 1 # +1 for EOS
                valid_token_count = min(valid_token_count, MAX_SEQ_LENGTH) # Cap at max length
                attention_mask_list = [1] * valid_token_count + [0] * (MAX_SEQ_LENGTH - valid_token_count)

                # --- Convert lists to Tensors ---
                input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
                labels_tensor = torch.tensor(labels_list, dtype=torch.long)
                attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long)

                # --- Save chunk to .pt file ---
                chunk_filename = f"chunk_{total_chunks_created:07d}.pt" # Use 7 digits for potentially millions of chunks
                chunk_filepath = os.path.join(OUTPUT_DIR, chunk_filename)
                torch.save({
                    "input_ids": input_ids_tensor,
                    "labels": labels_tensor,
                    "attention_mask": attention_mask_tensor
                }, chunk_filepath)

                # --- Write relative path to manifest ---
                manifest_entry = {"file_path": chunk_filename} # Store only the relative filename
                f_manifest.write(json.dumps(manifest_entry) + "\n")

                total_chunks_created += 1

                # --- Update start index for the next chunk ---
                # Step forward by the chunk size minus the overlap
                step_size = available_length_for_chunk - OVERLAP_TOKENS
                start_idx += max(1, step_size) # Ensure progress even if step_size is 0 or negative due to rounding/small available_length
                chunk_id += 1

except Exception as e:
    print(f"\nAn error occurred during Pass 2 processing sample around index {original_sample_count-1}: {e}")
    import traceback
    traceback.print_exc()
    # Attempt to close the manifest file cleanly if open
    try:
        f_manifest.close()
    except: pass
    sys.exit(1)

# --- 10. Final Summary ---
print("---" * 10)
print(f"Original samples processed: {original_sample_count}")
print(f"Samples skipped due to errors/length: {skipped_samples}")
print(f"New tensor chunks created: {total_chunks_created}")
print(f"Chunked tensors saved to directory: '{OUTPUT_DIR}'")
print(f"Manifest file created at: '{manifest_path}'")
print("Chunking process finished.")
