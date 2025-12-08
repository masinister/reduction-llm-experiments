"""Lightweight processing strategies for long contexts.

Strategies build on CoreBackend's structured output interface using Pydantic
models. Each strategy handles a different approach to processing text that
may exceed the model's context window.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Optional, Type, TypeVar
from pydantic import BaseModel
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
import logging

from src.core_backend import CoreBackend
from src.utils import parse_structured_output
from src import config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
M = TypeVar('M', bound=BaseModel)


def chunk_text(tokenizer: Any, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into chunks by token count.
    
    Args:
        tokenizer: Tokenizer with encode/decode methods
        text: Text to chunk
        chunk_size: Max tokens per chunk
        overlap: Tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return [""]
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break
    
    return chunks


class Strategy(ABC, Generic[T]):
    """Base class for processing strategies."""
    
    def __init__(self, backend: CoreBackend, output_model: Type[T]):
        self.backend = backend
        self.output_model = output_model
    
    @abstractmethod
    def process(self, text: str, prompt_fn: Callable[[str], str], **kwargs) -> Optional[T]:
        """Process text and return structured output."""
        pass


class DirectStrategy(Strategy[T]):
    """Process text directly in a single call. No chunking."""
    
    def process(
        self,
        text: str,
        prompt_fn: Callable[[str], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[T]:
        prompt = prompt_fn(text)
        
        params = SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            repetition_penalty=getattr(config, 'REPETITION_PENALTY', 1.0),
            structured_outputs=StructuredOutputsParams(
                json=self.output_model.model_json_schema()
            ),
        )
        
        output = self.backend.generate_once(prompt, params)
        return parse_structured_output(output, self.output_model)


class SlidingWindowStrategy(Strategy[T], Generic[T, M]):
    """Process chunks with structured memory carried between windows.
    
    Each chunk produces output of type T. A memory model M accumulates
    state across chunks. Final output combines all chunk outputs.
    """
    
    def __init__(
        self,
        backend: CoreBackend,
        output_model: Type[T],
        memory_model: Type[M],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ):
        super().__init__(backend, output_model)
        self.memory_model = memory_model
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        self.overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    
    def process(
        self,
        text: str,
        prompt_fn: Callable[[str, Optional[M]], str],
        combine_fn: Callable[[List[T]], T],
        update_memory_fn: Callable[[Optional[M], T], M],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[T]:
        """Process text with sliding window and memory.
        
        Args:
            text: Input text
            prompt_fn: (chunk_text, memory) -> prompt string
            combine_fn: Combine all chunk outputs into final output
            update_memory_fn: (old_memory, chunk_output) -> new_memory
            max_tokens: Max output tokens per chunk
            temperature: Sampling temperature
            
        Returns:
            Combined structured output
        """
        chunks = chunk_text(self.backend.tokenizer, text, self.chunk_size, self.overlap)
        
        memory: Optional[M] = None
        outputs: List[T] = []
        
        params = SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            repetition_penalty=getattr(config, 'REPETITION_PENALTY', 1.0),
            structured_outputs=StructuredOutputsParams(
                json=self.output_model.model_json_schema()
            ),
        )
        
        for i, chunk in enumerate(chunks):
            prompt = prompt_fn(chunk, memory)
            
            raw = self.backend.generate_once(prompt, params)
            parsed = parse_structured_output(raw, self.output_model)
            
            if parsed:
                outputs.append(parsed)
                memory = update_memory_fn(memory, parsed)
            else:
                logger.warning("Chunk %d/%d: failed to parse output", i + 1, len(chunks))
        
        if not outputs:
            return None
        
        return combine_fn(outputs)


class MapReduceStrategy(Strategy[T]):
    """Map over chunks independently, then reduce to final output.
    
    Simpler than SlidingWindow - no memory between chunks.
    Good for tasks where chunks are independent.
    """
    
    def __init__(
        self,
        backend: CoreBackend,
        output_model: Type[T],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ):
        super().__init__(backend, output_model)
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        self.overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    
    def process(
        self,
        text: str,
        prompt_fn: Callable[[str], str],
        combine_fn: Callable[[List[T]], T],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[T]:
        """Map prompt over chunks, combine results.
        
        Args:
            text: Input text
            prompt_fn: (chunk_text) -> prompt string
            combine_fn: Combine all chunk outputs into final output
            max_tokens: Max output tokens per chunk
            temperature: Sampling temperature
            
        Returns:
            Combined structured output
        """
        chunks = chunk_text(self.backend.tokenizer, text, self.chunk_size, self.overlap)
        
        params = SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            structured_outputs=StructuredOutputsParams(
                json=self.output_model.model_json_schema()
            ),
        )
        
        outputs: List[T] = []
        for i, chunk in enumerate(chunks):
            prompt = prompt_fn(chunk)
            raw = self.backend.generate_once(prompt, params)
            parsed = parse_structured_output(raw, self.output_model)
            
            if parsed:
                outputs.append(parsed)
            else:
                logger.warning("Chunk %d/%d: failed to parse output", i + 1, len(chunks))
        
        if not outputs:
            return None
        
        return combine_fn(outputs)


class StreamingStrategy(Strategy[T]):
    """Process chunks and concatenate outputs directly.
    
    Simplest strategy for position-independent tasks like conversion.
    No combining logic - just concatenates a field from each output.
    """
    
    def __init__(
        self,
        backend: CoreBackend,
        output_model: Type[T],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ):
        super().__init__(backend, output_model)
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        self.overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    
    def process(
        self,
        text: str,
        prompt_fn: Callable[[str], str],
        extract_fn: Callable[[T], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        separator: str = "",
    ) -> str:
        """Process chunks and concatenate extracted field.
        
        Args:
            text: Input text
            prompt_fn: (chunk_text) -> prompt string
            extract_fn: Extract string from parsed output
            max_tokens: Max output tokens per chunk
            temperature: Sampling temperature
            separator: String to join outputs
            
        Returns:
            Concatenated string output
        """
        chunks = chunk_text(self.backend.tokenizer, text, self.chunk_size, self.overlap)
        
        params = SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            structured_outputs=StructuredOutputsParams(
                json=self.output_model.model_json_schema()
            ),
        )
        
        results: List[str] = []
        for i, chunk in enumerate(chunks):
            prompt = prompt_fn(chunk)
            raw = self.backend.generate_once(prompt, params)
            parsed = parse_structured_output(raw, self.output_model)
            
            if parsed:
                results.append(extract_fn(parsed))
            else:
                logger.warning("Chunk %d/%d: failed to parse, using original", i + 1, len(chunks))
                results.append(chunk)  # Fallback to original
        
        return separator.join(results)
