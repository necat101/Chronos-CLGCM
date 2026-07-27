"""
LM-Evaluation-Harness Wrapper for Hierarchos models.

This module provides a wrapper class that makes HierarchosCore compatible with
EleutherAI's lm-evaluation-harness for standardized benchmark evaluation.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from ..inference.chat import (
    boundary_drift_seed,
    resolve_inference_prefill_chunk_size,
    uses_full_sample_inference_recurrence,
)

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False
    # Create dummy classes if lm-eval not installed
    class LM:
        pass
    class Instance:
        pass


class HierarchosLM(LM):
    """
    lm-evaluation-harness compatible wrapper for HierarchosCore models.
    
    Implements the three core methods required by lm-eval:
    - loglikelihood: Compute log-prob for context+continuation
    - loglikelihood_rolling: Compute perplexity for entire sequences
    - generate_until: Generate text until stop condition
    
    Usage:
        from hierarchos.evaluation import HierarchosLM
        
        lm = HierarchosLM(model, tokenizer, device, batch_size=4)
        results = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"])
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device: torch.device,
        batch_size: int = 1,
        max_length: Optional[int] = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: HierarchosCore model instance
            tokenizer: Tokenizer (HuggingFace compatible)
            device: torch.device to run inference on
            batch_size: Batch size for processing requests
            max_length: Optional max sequence length (uses model config if None)
        """
        if not _HAS_LM_EVAL:
            raise ImportError(
                "lm-evaluation-harness is not installed. "
                "Install it with: pip install lm-eval>=0.4.0"
            )
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._batch_size = batch_size
        self._max_length = max_length or getattr(model.config, 'max_length', 1024)
        self._prefill_chunk_size = resolve_inference_prefill_chunk_size(
            getattr(model, "config", None)
        )
        
        # Cache the eot token id
        if tokenizer.eos_token_id is not None:
            self._eot_token_id = tokenizer.eos_token_id
        else:
            self._eot_token_id = tokenizer.pad_token_id or 0
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def max_length(self) -> int:
        return self._max_length
    
    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id
    
    @property  
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, value: torch.device):
        self._device = value
    
    def tok_encode(self, string: str) -> List[int]:
        """Encode a string to token ids."""
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token ids to string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Encode jointly so GPT-style BPE boundaries match the concatenated text."""
        if not context:
            continuation_enc = self.tok_encode(continuation)
            if not continuation_enc:
                return [self.eot_token_id], []
            if continuation_enc[0] == self.eot_token_id:
                return continuation_enc[:1], continuation_enc[1:]
            return [self.eot_token_id], continuation_enc

        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]

        full_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        return context_enc, full_enc[len(context_enc):]

    def _truncate_scoring_pair(
        self,
        context_enc: List[int],
        continuation_enc: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Keep at least one conditioning token and the newest scoreable targets."""
        if not continuation_enc:
            return context_enc[-self.max_length:], []
        # A causal input needs context + continuation[:-1], so a one-token
        # context can score max_length continuation tokens.
        max_targets = max(0, self.max_length)
        continuation_enc = continuation_enc[-max_targets:] if max_targets else []
        context_budget = max(1, self.max_length - len(continuation_enc) + 1)
        context_enc = context_enc[-context_budget:] or [self.eot_token_id]
        return context_enc, continuation_enc
    
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the model and return logits.
        
        Args:
            input_ids: Input token ids [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        self.model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            chunk_size = self._prefill_chunk_size if self._prefill_chunk_size > 0 else input_ids.shape[1]
            chunk_size = max(1, int(chunk_size))
            h_state = None
            l_state = None
            prev_context = None
            target_context = None
            drift_state = None
            ltm_state = None
            logits_parts = []
            model_config = getattr(self.model, "config", None)
            exact_full_sample = uses_full_sample_inference_recurrence(model_config)

            for start in range(0, input_ids.shape[1], chunk_size):
                outputs = self.model(
                    input_ids=input_ids[:, start:start + chunk_size],
                    h_state=h_state,
                    l_state=l_state,
                    prev_context=prev_context,
                    target_context=target_context,
                    drift_state=boundary_drift_seed(
                        drift_state,
                        start,
                        self._prefill_chunk_size,
                        exact_full_sample=exact_full_sample,
                    ),
                    ltm_memory_state=ltm_state,
                    suppress_hebbian=True,
                    global_pos_offset=start,
                )
                logits_parts.append(outputs['logits'])
                h_state = outputs.get('h_state')
                l_state = outputs.get('l_state')
                prev_context = outputs.get('prev_context')
                target_context = outputs.get('target_context')
                drift_state = outputs.get('drift_state')
                ltm_state = outputs.get('ltm_memory_state')

            return torch.cat(logits_parts, dim=1)
    
    def loglikelihood(
        self, 
        requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuation given context.
        
        Each request contains (context, continuation) and we compute:
        - The log probability of the continuation given the context
        - Whether the continuation is the greedy choice
        
        Args:
            requests: List of Instance objects with args=(context, continuation)
            
        Returns:
            List of (log_prob, is_greedy) tuples
        """
        results = [None] * len(requests)
        
        # Process in batches
        for i in tqdm(range(0, len(requests), self.batch_size), 
                      desc="loglikelihood", disable=len(requests) < 10):
            batch_requests = requests[i:i + self.batch_size]
            
            encoded_batch = []
            for batch_offset, req in enumerate(batch_requests):
                context, continuation = req.args
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                context_enc, continuation_enc = self._truncate_scoring_pair(context_enc, continuation_enc)
                result_index = i + batch_offset
                if not continuation_enc:
                    results[result_index] = (0.0, True)
                    continue
                encoded_batch.append((result_index, context_enc, continuation_enc))

            if not encoded_batch:
                continue

            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            pad_id = self.eot_token_id if pad_id is None else int(pad_id)
            max_input_len = max(len(context_enc) + len(continuation_enc) - 1 for _, context_enc, continuation_enc in encoded_batch)
            input_ids = torch.full(
                (len(encoded_batch), max_input_len),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, (_, context_enc, continuation_enc) in enumerate(encoded_batch):
                full_enc = context_enc + continuation_enc[:-1]
                input_ids[row, :len(full_enc)] = torch.tensor(full_enc, dtype=torch.long, device=self.device)

            logits = self._model_call(input_ids)

            for row, (result_index, context_enc, continuation_enc) in enumerate(encoded_batch):
                cont_start = len(context_enc) - 1
                cont_len = len(continuation_enc)
                cont_logits = logits[row, cont_start:cont_start + cont_len, :]
                cont_targets = torch.tensor(continuation_enc, dtype=torch.long, device=self.device)
                log_probs = F.log_softmax(cont_logits.float(), dim=-1)
                target_log_probs = log_probs.gather(
                    dim=-1,
                    index=cont_targets.unsqueeze(-1)
                ).squeeze(-1)
                total_log_prob = target_log_probs.sum().item()
                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_targets).all().item()
                results[result_index] = (total_log_prob, is_greedy)

        return results
    
    def loglikelihood_rolling(
        self, 
        requests: List[Instance]
    ) -> List[float]:
        """
        Compute rolling log-likelihood (perplexity) for entire sequences.
        
        This is used for perplexity evaluation on datasets like WikiText.
        
        Args:
            requests: List of Instance objects with args=(sequence,)
            
        Returns:
            One full-document log probability per request.
        """
        results = []
        
        for i in tqdm(range(0, len(requests), self.batch_size),
                      desc="loglikelihood_rolling", disable=len(requests) < 10):
            batch_requests = requests[i:i + self.batch_size]
            
            for req in batch_requests:
                (sequence,) = req.args
                
                # Tokenize
                tokens = self.tok_encode(sequence)
                
                if len(tokens) == 0:
                    results.append(0.0)
                    continue

                total_log_prob = 0.0
                windows = get_rolling_token_windows(
                    token_list=tokens,
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )
                for context_enc, continuation_enc in map(make_disjoint_window, windows):
                    context_enc, continuation_enc = self._truncate_scoring_pair(context_enc, continuation_enc)
                    full_enc = context_enc + continuation_enc[:-1]
                    input_ids = torch.tensor([full_enc], dtype=torch.long, device=self.device)
                    logits = self._model_call(input_ids)
                    cont_start = len(context_enc) - 1
                    cont_len = len(continuation_enc)
                    pred_logits = logits[0, cont_start:cont_start + cont_len, :]
                    targets = torch.tensor(continuation_enc, dtype=torch.long, device=self.device)
                    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
                    total_log_prob += log_probs.gather(-1, targets.unsqueeze(-1)).sum().item()

                results.append(total_log_prob)
        
        return results
    
    def generate_until(
        self, 
        requests: List[Instance]
    ) -> List[str]:
        """
        Generate text until a stop condition is met.
        
        This is used for generative tasks like question answering.
        
        Args:
            requests: List of Instance objects with args=(context, gen_kwargs)
                      gen_kwargs may contain: until, max_gen_toks, temperature, etc.
            
        Returns:
            List of generated strings (continuation only, not including context)
        """
        results = []
        
        for req in tqdm(requests, desc="generate_until", disable=len(requests) < 10):
            context, gen_kwargs = req.args
            
            # Parse generation kwargs
            until = gen_kwargs.get("until", [self.tokenizer.eos_token or "</s>"])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 128)
            temperature = gen_kwargs.get("temperature", 0.0)  # 0 = greedy
            
            # Tokenize context
            context_enc = self.tok_encode(context) or [self.eot_token_id]
            
            # Truncate context if needed
            context_budget = max(1, self.max_length - max_gen_toks)
            if len(context_enc) > context_budget:
                context_enc = context_enc[-context_budget:]
            
            input_ids = torch.tensor([context_enc], dtype=torch.long, device=self.device)
            
            # Generate tokens autoregressively
            generated = []
            self.model.eval()
            
            # State management for Hierarchos
            h_state = None
            l_state = None
            prev_context = None
            target_context = None
            drift_state = None
            ltm_state = None
            model_config = getattr(self.model, "config", None)
            prefill_chunk_size = resolve_inference_prefill_chunk_size(model_config)
            exact_full_sample = uses_full_sample_inference_recurrence(model_config)
            total_tokens_seen = 0
            
            with torch.no_grad():
                # Prefill with context
                prefill_step = prefill_chunk_size if prefill_chunk_size > 0 else input_ids.shape[1]
                prefill_step = max(1, int(prefill_step))
                chunk_drift_state = None
                outputs = None
                for start in range(0, input_ids.shape[1], prefill_step):
                    end = min(start + prefill_step, input_ids.shape[1])
                    outputs = self.model(
                        input_ids=input_ids[:, start:end],
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=chunk_drift_state,
                        ltm_memory_state=ltm_state,
                        suppress_hebbian=True,
                        global_pos_offset=start,
                    )
                    h_state = outputs.get('h_state')
                    l_state = outputs.get('l_state')
                    prev_context = outputs.get('prev_context')
                    target_context = outputs.get('target_context')
                    drift_state = outputs.get('drift_state')
                    ltm_state = outputs.get('ltm_memory_state')
                    chunk_drift_state = boundary_drift_seed(
                        drift_state,
                        end,
                        prefill_chunk_size,
                        exact_full_sample=exact_full_sample,
                    )
                total_tokens_seen = len(context_enc)
                
                # Get last logits for next token prediction
                logits = outputs['logits'][0, -1, :]
                
                for _ in range(max_gen_toks):
                    # Sample or greedy
                    if temperature <= 0 or temperature < 1e-4:
                        next_token = logits.argmax(dim=-1)
                    else:
                        probs = F.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    generated.append(next_token.item())
                    
                    # Check stop conditions
                    gen_text = self.tok_decode(generated)
                    should_stop = False
                    for stop_str in until:
                        if stop_str in gen_text:
                            # Truncate at stop string
                            gen_text = gen_text.split(stop_str)[0]
                            should_stop = True
                            break
                    
                    if should_stop or next_token.item() == self._eot_token_id:
                        break
                    
                    # Next step
                    next_input = next_token.unsqueeze(0).unsqueeze(0)
                    generation_drift_state = boundary_drift_seed(
                        drift_state,
                        total_tokens_seen,
                        prefill_chunk_size,
                        exact_full_sample=exact_full_sample,
                    )
                    outputs = self.model(
                        input_ids=next_input,
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        # Epoch-13 TBPTT parity: drift is fed only at chunk boundaries.
                        drift_state=generation_drift_state,
                        ltm_memory_state=ltm_state,
                        suppress_hebbian=True,
                        global_pos_offset=total_tokens_seen
                    )
                    total_tokens_seen += 1
                    h_state = outputs.get('h_state')
                    l_state = outputs.get('l_state')
                    prev_context = outputs.get('prev_context')
                    target_context = outputs.get('target_context')
                    drift_state = outputs.get('drift_state')
                    ltm_state = outputs.get('ltm_memory_state')
                    logits = outputs['logits'][0, -1, :]
            
            # Decode final result
            gen_text = self.tok_decode(generated)
            
            # Truncate at first stop string
            for stop_str in until:
                if stop_str in gen_text:
                    gen_text = gen_text.split(stop_str)[0]
                    break
            
            results.append(gen_text)
        
        return results
