"""Context window budget management and token accounting.

This module handles the allocation of the model's context window across different
prompt components (system, summary, user content, output) and provides utilities
for truncating/compressing text to fit within budgets.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional
import logging

from src import config

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Token budget allocation as percentages of model's max context length.
    
    Attributes:
        system_pct: Percentage allocated for system prompt (e.g., 0.10 = 10%)
        summary_pct: Percentage allocated for running summary (e.g., 0.20 = 20%)
        prompt_pct: Percentage allocated for user prompt/chunk (e.g., 0.20 = 20%)
        output_pct: Percentage allocated for model output (e.g., 0.50 = 50%)
        safety_margin_tokens: Additional buffer to prevent context overflow
    """
    system_pct: float = None
    summary_pct: float = None
    prompt_pct: float = None
    output_pct: float = None
    safety_margin_tokens: int = None
    
    def __post_init__(self):
        """Load defaults from config if not specified."""
        if self.system_pct is None:
            self.system_pct = getattr(config, "CONTEXT_BUDGET_SYSTEM_PCT", 0.05)
        if self.summary_pct is None:
            self.summary_pct = getattr(config, "CONTEXT_BUDGET_SUMMARY_PCT", 0.10)
        if self.prompt_pct is None:
            self.prompt_pct = getattr(config, "CONTEXT_BUDGET_PROMPT_PCT", 0.35)
        if self.output_pct is None:
            self.output_pct = getattr(config, "CONTEXT_BUDGET_OUTPUT_PCT", 0.50)
        if self.safety_margin_tokens is None:
            self.safety_margin_tokens = getattr(config, "CONTEXT_BUDGET_SAFETY_MARGIN", 32)
        
        """Validate that percentages sum to approximately 1.0."""
        total = self.system_pct + self.summary_pct + self.prompt_pct + self.output_pct
        if not (0.95 <= total <= 1.05):  # Allow small floating point error
            logger.warning(
                "ContextBudget percentages sum to %.2f (expected ~1.0). "
                "This may cause unexpected behavior.",
                total
            )

    def allocate(self, model_max_len: int) -> dict:
        """Calculate token budgets for each component.
        
        Args:
            model_max_len: Maximum context length of the model
            
        Returns:
            Dictionary with keys: system_tokens, summary_tokens, prompt_tokens, 
            output_tokens, total_tokens
        """
        available = model_max_len - self.safety_margin_tokens
        
        if available <= 0:
            logger.error(
                "Model max_len (%d) is too small or safety_margin (%d) is too large",
                model_max_len, self.safety_margin_tokens
            )
            available = model_max_len  # Fallback, no safety margin
        
        allocation = {
            'system_tokens': int(available * self.system_pct),
            'summary_tokens': int(available * self.summary_pct),
            'prompt_tokens': int(available * self.prompt_pct),
            'output_tokens': int(available * self.output_pct),
        }
        allocation['total_tokens'] = sum(allocation.values())
        
        # Debug logging
        logger.info(
            "[BUDGET] Model max_len=%d, Safety margin=%d, Available=%d",
            model_max_len, self.safety_margin_tokens, available
        )
        logger.info(
            "[BUDGET] Allocation: System=%d (%.1f%%), Summary=%d (%.1f%%), Prompt=%d (%.1f%%), Output=%d (%.1f%%), Total=%d",
            allocation['system_tokens'], self.system_pct * 100,
            allocation['summary_tokens'], self.summary_pct * 100,
            allocation['prompt_tokens'], self.prompt_pct * 100,
            allocation['output_tokens'], self.output_pct * 100,
            allocation['total_tokens']
        )
        
        return allocation


def truncate_to_tokens(
    tokenizer: Any,
    text: str,
    max_tokens: int,
    prefer_tail: bool = True
) -> str:
    """Truncate text to fit within a token budget.
    
    Args:
        tokenizer: Tokenizer instance with encode/decode methods
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        prefer_tail: If True, keep the end of the text. If False, keep the beginning.
        
    Returns:
        Truncated text that fits within max_tokens
    """
    if not text:
        return ""
    
    token_ids = tokenizer.encode(text)
    
    if len(token_ids) <= max_tokens:
        return text
    
    if prefer_tail:
        # Keep the most recent content (end of text)
        truncated_ids = token_ids[-max_tokens:]
        prefix = "..."
    else:
        # Keep the beginning
        truncated_ids = token_ids[:max_tokens]
        prefix = ""
    
    truncated_text = tokenizer.decode(truncated_ids)
    return (prefix + truncated_text) if prefix else truncated_text


def compress_summary_via_model(
    generate_fn: Callable[[Optional[str], str], str],
    current_summary: str,
    new_content: str,
    max_tokens: int,
    system_prompt: Optional[str] = None
) -> str:
    """Compress a summary using an LLM to stay within token budget.
    
    This is a callback-based approach that allows dependency injection of the
    actual LLM generation function, making it testable.
    
    Args:
        generate_fn: Callable that takes (system, user) and returns generated text.
                     Should be deterministic (temperature=0.0).
        current_summary: The existing summary to update
        new_content: New content to incorporate into the summary
        max_tokens: Target token count for compressed summary
        system_prompt: Optional system prompt for the generation
        
    Returns:
        Compressed summary text
    """
    user_prompt = (
        f"CONCISE SUMMARY TASK:\n"
        f"Compress the following information into a {max_tokens}-token summary "
        f"that captures the essential information.\n\n"
        f"Previous summary:\n{current_summary}\n\n"
        f"New content:\n{new_content}\n\n"
        f"Instructions: Produce a concise summary (target: {max_tokens} tokens) "
        f"that preserves key facts and context needed for downstream processing."
    )
    
    try:
        compressed = generate_fn(system_prompt, user_prompt)
        return compressed if compressed else current_summary
    except Exception as e:
        logger.exception("compress_summary_via_model failed: %s", e)
        # Fallback: naive concatenation + truncation
        return current_summary + "\n" + new_content


def chunk_text_by_tokens(
    tokenizer: Any,
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int = 0
) -> List[str]:
    """Split text into token-count-based chunks with optional overlap.
    
    Args:
        tokenizer: Tokenizer instance with encode/decode methods
        text: Text to chunk
        chunk_size_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return [""]
    
    token_ids = tokenizer.encode(text)
    n = len(token_ids)
    
    if n <= chunk_size_tokens:
        return [text]
    
    chunks: List[str] = []
    start = 0
    
    while start < n:
        end = min(start + chunk_size_tokens, n)
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
        
        if end >= n:
            break
        
        # Next start overlaps by overlap_tokens
        start = max(0, end - overlap_tokens)
    
    return chunks
