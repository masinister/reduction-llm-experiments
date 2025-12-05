"""Core LLM interaction layer.

This module provides a minimal, stateless wrapper around vLLM for generating
completions. It is responsible ONLY for:
  - LLM initialization and configuration
  - Single-shot generation with sampling parameters and structured outputs
  
It does NOT handle chunking, merging, or orchestration logic.
"""

from typing import Any, Dict, List, Optional
from vllm import LLM, SamplingParams
import logging
import json

from src import config
from src.debug_printer import DebugPrinter

logger = logging.getLogger(__name__)


class CoreBackend:
    """Minimal LLM client wrapper with dependency injection support.
    
    Designed for:
      - Easy testing via LLM/tokenizer injection
      - Stateless generation (no internal state between calls)
      - Clean separation from higher-level orchestration
    """

    def __init__(
        self,
        *,
        llm: Optional[LLM] = None,
        tokenizer: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the LLM backend.
        
        Args:
            llm: Pre-initialized vLLM instance for testing/injection. If None, 
                creates from config.
            tokenizer: Pre-initialized tokenizer for testing. If None, uses LLM's 
                tokenizer.
            model_config: Optional config overrides. If None, uses module config.
        """
        if llm is None:
            # Build from config with sensible defaults
            cfg = model_config or {}
            self.llm = LLM(
                cfg.get('model_id', getattr(config, "MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")),
                gpu_memory_utilization=cfg.get('gpu_memory_utilization', getattr(config, "GPU_MEMORY_UTILIZATION", None)),
                tensor_parallel_size=cfg.get('tensor_parallel_size', getattr(config, "TENSOR_PARALLEL_SIZE", 1)),
                max_model_len=cfg.get('max_model_len', getattr(config, "MAX_MODEL_LEN", 8192)),
                compilation_config={"cudagraph_mode": cfg.get('cudagraph_mode', getattr(config, "CUDAGRAPH_MODE", False))},
                dtype=cfg.get('dtype', getattr(config, "DTYPE", None)),
            )
        else:
            self.llm = llm

        # Tokenizer shortcut for convenience
        self.tokenizer = tokenizer if tokenizer is not None else self.llm.get_tokenizer()
        
        # Store max model len for budget calculations
        self.max_model_len = getattr(config, "MAX_MODEL_LEN", 8192)

    def generate_once(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        max_retries: int = 2
    ) -> str:
        """Generate a single completion with structured output support.
        
        Uses llm.generate() with plain text prompts. Structured outputs
        constrain generation to valid JSON matching the schema.
        
        Args:
            prompt: Plain text prompt
            sampling_params: vLLM sampling configuration (includes structured outputs)
            max_retries: Number of retry attempts on empty output
            
        Returns:
            Generated text (may be empty string if all retries fail)
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning("Failed to apply chat template: %s", e)
        
        # DEBUG: Pretty-print the prompt
        if getattr(config, "DEBUG", False):
            DebugPrinter.print_prompt(prompt, sampling_params)
        
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            try:
                results = self.llm.generate([prompt], sampling_params=sampling_params)
                
                if results and results[0].outputs:
                    text = results[0].outputs[0].text or ""
                    # Check for garbage output (mostly whitespace/newlines)
                    if text.strip() and len(text.strip()) / max(len(text), 1) > 0.1:
                        # DEBUG: Pretty-print the response
                        if getattr(config, "DEBUG", False):
                            DebugPrinter.print_response(text)
                        return text
                    
                logger.warning(
                    "generate_once: empty output on attempt %d/%d",
                    attempt, max_retries + 1
                )
                    
            except Exception as e:
                logger.exception(
                    "generate_once: LLM generation failed on attempt %d/%d: %s",
                    attempt, max_retries + 1, e
                )
        
        logger.error(
            "generate_once: failed to produce non-empty output after %d attempts",
            max_retries + 1
        )
        return ""

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs back to text."""
        return self.tokenizer.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenize(text))
