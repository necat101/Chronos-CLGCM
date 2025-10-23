import json
from transformers import AutoTokenizer
import sys
import os
from tqdm import tqdm
import torch
import argparse
import math # <<< Import math for ceil

# --- 1. Define Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Chunk a JSONL dataset for Chronos training, saving chunks into consolidated .pt files.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Path to the input JSONL dataset file (e.g., train.jsonl).")
parser.add_argument("--tokenizer-path", type=str, default="openai-community/gpt2", # Changed default
                    help="Path or Hugging Face name of the tokenizer to use.")
parser.add_argument("--overlap", type=int, default=1024,
                    help="Number of tokens to overlap between consecutive chunks.")
parser.add_argument("--output-dir", type=str, default="train_chronos_chunked_tensors",
                    help="Directory to save the output consolidated .pt chunk files and manifest.jsonl.")
# <<< ADDED: chunks_per_file argument >>>
parser.add_argument("--chunks-per-file", type=int, default=1000,
                    help="Number of chunks to consolidate into a single .pt file.")
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
CHUNKS_PER_FILE = args.chunks_per_file # <<< Use new argument

# --- 2. Load Tokenizer ---
print(f"Loading tokenizer '{TOKENIZER_PATH}'...")
try:
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
        # Estimate total lines for tqdm
        try:
            total_lines_estimate = sum(1 for _ in open(DATASET_FILE, 'r', encoding='utf-8'))
        except:
            total_lines_estimate = None # Handle potential errors reading file size

        f.seek(0) # Reset after counting
        pbar_analyze = tqdm(f, total=total_lines_estimate, desc="Analyzing Samples", unit="sample")
        for line_num, line in enumerate(pbar_analyze):
            total_samples_analyzed += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                # Reduce verbosity, maybe print every 1000th error
                if (line_num + 1) % 1000 == 0:
                     pbar_analyze.set_postfix_str(f"Skipping malformed JSON line ~{line_num+1}...")
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
                pbar_analyze.set_postfix_str(f"New max thought: {max_thought_len_tokens} tokens")

except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_FILE}'")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during Pass 1: {e}")
    sys.exit(1)

print(f"Analysis Complete: Processed {total_samples_analyzed} samples.")
if max_thought_len_tokens == 0:
    print("Warning: No valid 'thought-process' data found. Setting minimum anchor budget.")
    max_thought_len_tokens = len(tokenizer.encode(thought_start_wrapper, add_special_tokens=False)) + \
                             len(tokenizer.encode(thought_end_wrapper, add_special_tokens=False)) + 5

print("---" * 10)

# --- 4. Dynamically Set Final Parameters ---
MAX_ANCHOR_BUDGET = max_thought_len_tokens + ANCHOR_SAFETY_MARGIN
MIN_CHUNKABLE_TOKENS = RESERVED_CHUNK_SPACE
MAX_SEQ_LENGTH = MAX_ANCHOR_BUDGET + MIN_CHUNKABLE_TOKENS + 1 # +1 for EOS

if OVERLAP_TOKENS >= MIN_CHUNKABLE_TOKENS:
      print(f"WARNING: Specified overlap ({OVERLAP_TOKENS}) is >= RESERVED_CHUNK_SPACE ({MIN_CHUNKABLE_TOKENS}).")
      new_overlap = max(1, MIN_CHUNKABLE_TOKENS // 2)
      print(f"         Reducing overlap to {new_overlap} to ensure forward progress.")
      OVERLAP_TOKENS = new_overlap

print(f"Longest thought-process: {max_thought_len_tokens} tokens")
print(f"Anchor safety margin: {ANCHOR_SAFETY_MARGIN} tokens")
print(f"Effective Anchor Budget (MAX_ANCHOR_BUDGET): {MAX_ANCHOR_BUDGET} tokens")
print(f"Reserved chunk space (MIN_CHUNKABLE_TOKENS): {MIN_CHUNKABLE_TOKENS} tokens")
print(f"Overlap set to: {OVERLAP_TOKENS} tokens")
print(f"FINAL MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH} tokens")
print(f"Chunks per consolidated file: {CHUNKS_PER_FILE}")
print("---" * 10)

# --- 5. Pass 2: Process and Chunk the Dataset ---
print(f"Starting Pass 2: Chunking dataset '{DATASET_FILE}' to {MAX_SEQ_LENGTH} tokens...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
manifest_path = os.path.join(OUTPUT_DIR, "manifest.jsonl")

original_sample_count = 0
skipped_samples = 0
total_chunks_created = 0

# <<< MODIFICATION: Variables for consolidation >>>
chunk_buffer = [] # Holds chunks before saving
current_file_index = 0
current_chunk_filename = f"consolidated_chunks_{current_file_index:07d}.pt"

# Define wrappers for easier access
inst_start_wrapper = "### Instruction:\n"
inst_end_wrapper = "\n\n"
feel_start_wrapper = "### Feelings:\n"
feel_end_wrapper = "\n\n"
output_start_wrapper = "### Response:\n"

# <<< MODIFICATION: Function to save buffer >>>
def save_chunk_buffer(buffer, filename, output_dir):
    if not buffer:
        return
    filepath = os.path.join(output_dir, filename)
    try:
        torch.save(buffer, filepath)
    except Exception as e:
        print(f"\nError saving buffer to {filepath}: {e}")
        # Decide how to handle this - skip file, retry, exit?
        # For now, print error and continue, data might be lost.
    buffer.clear()

try:
    with open(DATASET_FILE, "r", encoding="utf-8") as f_in, \
         open(manifest_path, "w", encoding="utf-8") as f_manifest:
        # Wrap the file reading with tqdm for progress bar
        # Use estimated total lines if available
        pbar_chunk = tqdm(f_in, total=total_lines_estimate, desc="Chunking Samples", unit="sample")
        for sample_idx, line in enumerate(pbar_chunk):
            original_sample_count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                if (sample_idx + 1) % 1000 == 0:
                    pbar_chunk.set_postfix_str(f"Skipping malformed JSON line ~{sample_idx+1}...")
                skipped_samples += 1
                continue

            # --- 6. Extract Content ---
            inst_content = sample.get("Instruction", "")
            output_content = sample.get("output", "")
            thought_content_str = sample.get('thought-process', '')
            feelings_content_str = sample.get('feelings', '')

            if not isinstance(inst_content, str): inst_content = ""
            if not isinstance(output_content, str): output_content = ""
            if not isinstance(thought_content_str, str): thought_content_str = ""
            if not isinstance(feelings_content_str, str): feelings_content_str = ""

            # --- 7. Build the STATIC ANCHOR (Thought Process) ---
            thought_start_tokens = tokenizer.encode(thought_start_wrapper, add_special_tokens=False)
            thought_content_tokens = tokenizer.encode(thought_content_str, add_special_tokens=False)
            thought_end_tokens = tokenizer.encode(thought_end_wrapper, add_special_tokens=False)
            current_thought_len = len(thought_start_tokens) + len(thought_content_tokens) + len(thought_end_tokens)

            if current_thought_len > MAX_ANCHOR_BUDGET:
                available_for_thought_content = MAX_ANCHOR_BUDGET - len(thought_start_tokens) - len(thought_end_tokens)
                if available_for_thought_content <= 0:
                    skipped_samples += 1
                    continue
                thought_content_tokens = thought_content_tokens[:available_for_thought_content]

            anchor_tokens = thought_start_tokens + thought_content_tokens + thought_end_tokens
            anchor_len = len(anchor_tokens)

            # --- 8. Build the CONTINUOUS CHUNKABLE STREAM ---
            inst_tokens = tokenizer.encode(inst_start_wrapper + inst_content + inst_end_wrapper, add_special_tokens=False)
            feelings_tokens = []
            if feelings_content_str:
                feelings_tokens = tokenizer.encode(feel_start_wrapper + feelings_content_str + feel_end_wrapper, add_special_tokens=False)
            output_tokens = tokenizer.encode(output_start_wrapper + output_content, add_special_tokens=False)

            chunkable_tokens = inst_tokens + feelings_tokens + output_tokens
            prompt_len_in_stream = len(inst_tokens) + len(feelings_tokens)

            # --- 9. Chunking Loop ---
            available_length_for_chunk = MAX_SEQ_LENGTH - anchor_len - 1 # -1 for EOS

            if available_length_for_chunk <= OVERLAP_TOKENS or available_length_for_chunk <= 0:
                 skipped_samples += 1
                 continue

            start_idx = 0
            chunk_id = 0
            while start_idx < len(chunkable_tokens) or (start_idx == 0 and not chunkable_tokens): # Ensure at least one chunk if empty
                end_idx = start_idx + available_length_for_chunk
                chunk_content_tokens = chunkable_tokens[start_idx:end_idx]

                # --- Construct final input_ids LIST ---
                input_ids_list = anchor_tokens + chunk_content_tokens + [EOS_TOKEN_ID]

                # --- Padding ---
                current_len = len(input_ids_list)
                padding_length = MAX_SEQ_LENGTH - current_len
                if padding_length < 0:
                    input_ids_list = input_ids_list[:MAX_SEQ_LENGTH]
                    chunk_content_len = MAX_SEQ_LENGTH - anchor_len - 1
                    chunk_content_tokens = chunk_content_tokens[:chunk_content_len] if chunk_content_len > 0 else []
                    padding_length = 0
                elif padding_length > 0:
                    input_ids_list.extend([PAD_TOKEN_ID] * padding_length)

                # --- Construct labels LIST with masking ---
                labels_list = list(input_ids_list)
                labels_list[:anchor_len] = [-100] * anchor_len # Mask anchor
                chunk_rel_prompt_start = max(0, prompt_len_in_stream - start_idx)
                chunk_rel_prompt_end = min(len(chunk_content_tokens), prompt_len_in_stream - start_idx)
                if chunk_rel_prompt_start < chunk_rel_prompt_end:
                    mask_start_index = anchor_len + chunk_rel_prompt_start
                    mask_end_index = anchor_len + chunk_rel_prompt_end
                    labels_list[mask_start_index:mask_end_index] = [-100] * (mask_end_index - mask_start_index)
                if chunk_id > 0: # Mask overlap
                    overlap_mask_start = anchor_len
                    overlap_mask_end = min(anchor_len + OVERLAP_TOKENS, anchor_len + len(chunk_content_tokens))
                    if overlap_mask_start < overlap_mask_end:
                         labels_list[overlap_mask_start:overlap_mask_end] = [-100] * (overlap_mask_end - overlap_mask_start)
                eos_and_padding_start_index = min(anchor_len + len(chunk_content_tokens), MAX_SEQ_LENGTH)
                labels_list[eos_and_padding_start_index:] = [-100] * (MAX_SEQ_LENGTH - eos_and_padding_start_index)

                # --- Construct attention mask LIST ---
                valid_token_count = min(anchor_len + len(chunk_content_tokens) + 1, MAX_SEQ_LENGTH)
                attention_mask_list = [1] * valid_token_count + [0] * (MAX_SEQ_LENGTH - valid_token_count)

                # --- Create chunk dictionary (tensors created later or just before save) ---
                chunk_data = {
                    "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long)
                }

                # <<< MODIFICATION: Add to buffer >>>
                chunk_buffer.append(chunk_data)
                index_in_current_file = len(chunk_buffer) - 1

                # <<< MODIFICATION: Write manifest entry >>>
                manifest_entry = {
                    "file_path": current_chunk_filename,
                    "index_in_file": index_in_current_file
                }
                f_manifest.write(json.dumps(manifest_entry) + "\n")
                total_chunks_created += 1

                # <<< MODIFICATION: Save buffer if full >>>
                if len(chunk_buffer) >= CHUNKS_PER_FILE:
                    pbar_chunk.set_postfix_str(f"Saving {current_chunk_filename}...")
                    save_chunk_buffer(chunk_buffer, current_chunk_filename, OUTPUT_DIR)
                    current_file_index += 1
                    current_chunk_filename = f"consolidated_chunks_{current_file_index:07d}.pt"

                # --- Update start index for the next chunk ---
                step_size = available_length_for_chunk - OVERLAP_TOKENS
                start_idx += max(1, step_size)
                chunk_id += 1

                # Break if no chunkable tokens were present in the first place
                if start_idx == 0 and not chunkable_tokens:
                    break


except Exception as e:
    print(f"\nAn error occurred during Pass 2 processing sample around index {original_sample_count-1}: {e}")
    import traceback
    traceback.print_exc()
finally:
    # <<< MODIFICATION: Save any remaining chunks in the buffer >>>
    if chunk_buffer:
        print(f"\nSaving remaining {len(chunk_buffer)} chunks to {current_chunk_filename}...")
        save_chunk_buffer(chunk_buffer, current_chunk_filename, OUTPUT_DIR)
    # Ensure files are closed if an exception occurred within the 'with' block
    try:
        if 'f_in' in locals() and not f_in.closed: f_in.close()
        if 'f_manifest' in locals() and not f_manifest.closed: f_manifest.close()
    except Exception as close_e:
        print(f"Error closing files: {close_e}")


# --- 10. Final Summary ---
print("---" * 10)
print(f"Original samples processed: {original_sample_count}")
print(f"Samples skipped due to errors/length: {skipped_samples}")
print(f"New tensor chunks created: {total_chunks_created}")
print(f"Consolidated into {current_file_index + 1} '.pt' files.") # +1 because index is 0-based
print(f"Chunked tensors saved to directory: '{OUTPUT_DIR}'")
print(f"Manifest file created at: '{manifest_path}'")
print("Chunking process finished.")
