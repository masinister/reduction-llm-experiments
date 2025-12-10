"""Chunking strategies for long context processing.

Two approaches for handling text that exceeds context limits:
- Sequential: Process chunks with memory context, LLM combines at end
- Hierarchical: Process chunks independently, LLM combines all at once
"""

from typing import Callable, Type, TypeVar
from pydantic import BaseModel

from src.core_backend import Backend
from src import config

T = TypeVar('T', bound=BaseModel)


def needs_chunking(backend: Backend, text: str, buffer: int = 1000) -> bool:
    """Check if text needs chunking based on token count.
    
    Args:
        backend: Backend instance for token counting
        text: Text to check
        buffer: Token buffer for prompt/output overhead
        
    Returns:
        True if text exceeds context limit minus buffer
    """
    return backend.count_tokens(text) > config.MAX_CONTEXT - buffer


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks by approximate token count.
    
    Args:
        text: Text to split
        chunk_size: Target tokens per chunk (chars / 4)
        overlap: Token overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return [""]
    
    # Convert to char counts (approx 4 chars per token)
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    
    if len(text) <= char_size:
        return [text]
    
    chunks = []
    step = max(1, char_size - char_overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i + char_size])
        if i + char_size >= len(text):
            break
    
    return chunks


def sequential_extract(
    backend: Backend,
    text: str,
    response_model: Type[T],
    extract_prompt: Callable[[str, str | None], str],
    combine_prompt: Callable[[list[str]], str],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> T:
    """Process chunks sequentially with memory, then LLM combines.
    
    Each chunk sees a summary of previous chunks. After all chunks are
    processed, an LLM call combines the partial results.
    
    Args:
        backend: Backend instance
        text: Input text
        response_model: Pydantic model for final output
        extract_prompt: (chunk, previous_summary | None) -> prompt
        combine_prompt: (list of partial outputs as JSON) -> prompt
        chunk_size: Tokens per chunk (defaults to config)
        overlap: Token overlap (defaults to config)
        
    Returns:
        Combined structured output
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    
    if not needs_chunking(backend, text):
        return backend.create(extract_prompt(text, None), response_model)
    
    chunks = chunk_text(text, chunk_size, overlap)
    partials: list[str] = []
    summary: str | None = None
    
    for chunk in chunks:
        result = backend.create(extract_prompt(chunk, summary), response_model)
        result_json = result.model_dump_json()
        partials.append(result_json)
        # Use the output as context for next chunk
        summary = result_json
    
    # LLM combines all partial results
    return backend.create(combine_prompt(partials), response_model)


def hierarchical_extract(
    backend: Backend,
    text: str,
    response_model: Type[T],
    extract_prompt: Callable[[str], str],
    combine_prompt: Callable[[list[str]], str],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> T:
    """Process chunks independently, then LLM combines all.
    
    Each chunk is processed without knowledge of others.
    All results are combined in a single LLM call at the end.
    
    Args:
        backend: Backend instance
        text: Input text
        response_model: Pydantic model for final output
        extract_prompt: (chunk) -> prompt
        combine_prompt: (list of partial outputs as JSON) -> prompt
        chunk_size: Tokens per chunk (defaults to config)
        overlap: Token overlap (defaults to config)
        
    Returns:
        Combined structured output
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    
    if not needs_chunking(backend, text):
        return backend.create(extract_prompt(text), response_model)
    
    chunks = chunk_text(text, chunk_size, overlap)
    partials: list[str] = []
    
    for chunk in chunks:
        result = backend.create(extract_prompt(chunk), response_model)
        partials.append(result.model_dump_json())
    
    # LLM combines all partial results
    return backend.create(combine_prompt(partials), response_model)
