"""Context window budget management and token accounting.

This module handles the allocation of the model's context window across different
prompt components (summary, user content, output) and provides utilities
for truncating text to fit within budgets.
"""

from dataclasses import dataclass
from typing import Any, List
import logging

from src import config

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Token budget allocation as percentages of model's max context length.
    
    Attributes:
        summary_pct: Percentage allocated for running summary (e.g., 0.15 = 15%)
        prompt_pct: Percentage allocated for user prompt/chunk (e.g., 0.35 = 35%)
        output_pct: Percentage allocated for model output (e.g., 0.50 = 50%)
        safety_margin_tokens: Additional buffer to prevent context overflow
    """
    summary_pct: float = None
    prompt_pct: float = None
    output_pct: float = None
    safety_margin_tokens: int = None
    
    def __post_init__(self):
        """Load defaults from config if not specified."""
        if self.summary_pct is None:
            self.summary_pct = getattr(config, "CONTEXT_BUDGET_SUMMARY_PCT", 0.15)
        if self.prompt_pct is None:
            self.prompt_pct = getattr(config, "CONTEXT_BUDGET_PROMPT_PCT", 0.35)
        if self.output_pct is None:
            self.output_pct = getattr(config, "CONTEXT_BUDGET_OUTPUT_PCT", 0.50)
        if self.safety_margin_tokens is None:
            self.safety_margin_tokens = getattr(config, "CONTEXT_BUDGET_SAFETY_MARGIN", 32)
        
        # Validate that percentages sum to approximately 1.0
        total = self.summary_pct + self.prompt_pct + self.output_pct
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
            Dictionary with keys: summary_tokens, prompt_tokens, output_tokens, total_tokens
        """
        available = model_max_len - self.safety_margin_tokens
        
        if available <= 0:
            logger.error(
                "Model max_len (%d) is too small or safety_margin (%d) is too large",
                model_max_len, self.safety_margin_tokens
            )
            available = model_max_len  # Fallback, no safety margin
        
        allocation = {
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
            "[BUDGET] Allocation: Summary=%d (%.1f%%), Prompt=%d (%.1f%%), Output=%d (%.1f%%), Total=%d",
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


def chunk_text_by_tokens(
    tokenizer: Any,
    text: str,
    max_chunk_tokens: int,
    overlap_tokens: int = 0
) -> List[str]:
    """Split text into N equal-length chunks if it exceeds the budget.
    
    If text fits within max_chunk_tokens, returns it as a single chunk.
    Otherwise, calculates the minimum N such that each chunk fits within
    the budget, then splits the text into N equal-sized chunks with overlap.
    
    Args:
        tokenizer: Tokenizer instance with encode/decode methods
        text: Text to chunk
        max_chunk_tokens: Maximum tokens per chunk (budget)
        overlap_tokens: Number of tokens to overlap between consecutive chunks
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If max_chunk_tokens <= 0 or overlap_tokens >= max_chunk_tokens
    """
    # Input validation
    if max_chunk_tokens <= 0:
        raise ValueError(f"max_chunk_tokens must be > 0, got {max_chunk_tokens}")
    if overlap_tokens < 0:
        overlap_tokens = 0
    if overlap_tokens >= max_chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < max_chunk_tokens ({max_chunk_tokens})"
        )
    
    if not text:
        return [""]
    
    token_ids = tokenizer.encode(text)
    total_tokens = len(token_ids)
    
    # If text fits in budget, return as single chunk
    if total_tokens <= max_chunk_tokens:
        return [text]
    
    # Calculate minimum N chunks needed
    # Each chunk (except last) contributes (chunk_size - overlap) new tokens
    # So: N * (chunk_size - overlap) + overlap >= total_tokens
    # We want chunk_size <= max_chunk_tokens
    # Solving: N >= (total_tokens - overlap) / (max_chunk_tokens - overlap)
    effective_chunk = max_chunk_tokens - overlap_tokens
    n_chunks = max(2, (total_tokens - overlap_tokens + effective_chunk - 1) // effective_chunk)  # Ceiling division
    
    # Calculate actual chunk size for equal splitting
    # total_tokens = n_chunks * chunk_size - (n_chunks - 1) * overlap
    # chunk_size = (total_tokens + (n_chunks - 1) * overlap) / n_chunks
    chunk_size = (total_tokens + (n_chunks - 1) * overlap_tokens) // n_chunks
    
    # Build chunks
    chunks: List[str] = []
    step = chunk_size - overlap_tokens
    
    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_size, total_tokens)
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
    
    return chunks
