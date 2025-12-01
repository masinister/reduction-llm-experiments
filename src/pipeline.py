"""High-level pipeline orchestration for chunked text processing.

This module coordinates the entire flow:
  1. Chunk input text using token-aware chunking
  2. Process each chunk with LLM (structured outputs)
  3. Maintain running summary across chunks
  4. Incrementally merge chunk outputs
  5. Return final structured output
  
It uses dependency injection for all components (backend, budget, merger) to
enable testing and configurability.
"""

from typing import Any, Callable, Dict, List, Optional
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from pydantic import BaseModel, Field
import json
import logging

from src.core_backend import CoreBackend
from src.context_budget import ContextBudget, chunk_text_by_tokens, truncate_to_tokens
from src.merger import IncrementalMerger, HierarchicalMerger
from src.utils import parse_structured_output, find_json_block
from src import config

logger = logging.getLogger(__name__)


class SummaryResponse(BaseModel):
    """Pydantic model for summary updates."""
    summary: str = Field(..., description="A concise summary of the content processed so far.")


def _extract_json(raw_output: str) -> str:
    """Extract JSON from raw LLM output, with fallback to original or empty object."""
    if not raw_output or not raw_output.strip():
        return "{}"
    return find_json_block(raw_output) or raw_output


class Pipeline:
    """High-level orchestrator for chunked LLM processing with merging.
    
    This class coordinates:
      - CoreBackend for LLM calls
      - ContextBudget for token allocation
      - IncrementalMerger or HierarchicalMerger for output assembly
      - Running summary maintenance across chunks
    """

    def __init__(
        self,
        core_backend: CoreBackend,
        context_budget: Optional[ContextBudget] = None,
        merger_strategy: str = "incremental",
        merger_batch_size: int = 5,
        use_summarization: bool = True
    ):
        """Initialize the pipeline.
        
        Args:
            core_backend: CoreBackend instance for LLM interactions
            context_budget: ContextBudget for token allocation (None uses defaults)
            merger_strategy: "incremental" or "hierarchical"
            merger_batch_size: Number of items per merge batch
            use_summarization: Whether to maintain running summary across chunks
        """
        self.core = core_backend
        self.budget = context_budget or ContextBudget()
        self.use_summarization = use_summarization
        self.merger_batch_size = merger_batch_size
        self._current_schema: Optional[Dict[str, Any]] = None  # Store schema for merge operations
        
        # Create merger with injected merge function
        if merger_strategy == "incremental":
            self.merger = IncrementalMerger(
                merge_fn=self._merge_batch,
                batch_size=merger_batch_size
            )
        elif merger_strategy == "hierarchical":
            self.merger = HierarchicalMerger(
                merge_fn=self._merge_batch,
                batch_size=merger_batch_size
            )
        else:
            raise ValueError(f"Unknown merger_strategy: {merger_strategy}")
        
        self.merger_strategy = merger_strategy

    def _merge_batch(self, outputs_batch: List[str], running_summary: str) -> str:
        """Merge a batch of chunk outputs using the LLM.
        
        This is the callback injected into the merger. It constructs a merge
        prompt and calls the backend to produce a merged output.
        
        Args:
            outputs_batch: List of chunk outputs to merge
            running_summary: Current summary for context
            
        Returns:
            Merged output text (typically JSON)
        """
        if not outputs_batch:
            return ""
        
        # Build merge prompt - simpler is better for small models
        prompt = "Merge these JSON outputs into one. Combine all points into a single array.\n\n"
        
        for i, output in enumerate(outputs_batch):
            prompt += f"Output {i+1}:\n{output}\n\n"
        
        # Note: /no_think disables Qwen3's thinking mode which can cause infinite loops
        prompt += "Merged result (include ALL points from ALL outputs): /no_think"
        
        # Use structured output with the current schema to ensure valid JSON
        sampling_params = SamplingParams(
            max_tokens=getattr(config, "MAX_TOKENS"),
            temperature=0.0,
            top_p=1.0,
        )
        
        # Add structured outputs if schema is available
        if self._current_schema is not None:
            sampling_params.structured_outputs = StructuredOutputsParams(json=self._current_schema)
        
        merged = self.core.generate_once(prompt, sampling_params)
        return _extract_json(merged)

    def _update_summary(
        self,
        current_summary: str,
        chunk_output: str,
        max_summary_tokens: int
    ) -> str:
        """Update the running summary with new chunk output.
        
        Args:
            current_summary: Current summary text
            chunk_output: Latest chunk output to incorporate
            max_summary_tokens: Target token count for summary
            
        Returns:
            Updated summary text
        """
        current_summary_tokens = self.core.count_tokens(current_summary) if current_summary else 0

        # Note: /no_think disables Qwen3's thinking mode which can cause infinite loops
        # Use few-shot prompting to help smaller models understand the task
        if current_summary and current_summary.strip() and current_summary.strip() != '...':
            prompt = (
                "Summarize the combined information from the previous summary and new chunk.\n\n"
                f"Previous summary:\n{current_summary}\n\n"
                f"New chunk result:\n{chunk_output}\n\n"
                f"Write a concise summary (under {max_summary_tokens} tokens) combining the key information from both. /no_think"
            )
        else:
            prompt = (
                "Summarize the key information from this chunk result.\n\n"
                f"Chunk result:\n{chunk_output}\n\n"
                f"Write a concise summary (under {max_summary_tokens} tokens) of the main points. /no_think"
            )
        
        # Use structured output for summary
        so_summary = StructuredOutputsParams(json=SummaryResponse.model_json_schema())
        summary_sampling_params = SamplingParams(
            max_tokens=max_summary_tokens,
            temperature=0.0,
            structured_outputs=so_summary,
        )
        
        try:
            raw_summary = self.core.generate_once(prompt, summary_sampling_params)
            
            # Parse using utils
            parsed = parse_structured_output(raw_summary, SummaryResponse)
            if parsed:
                return parsed.summary
            
            # Fallback: use raw text if it's non-empty
            if raw_summary.strip():
                return raw_summary.strip()
                
        except Exception as e:
            logger.exception("Summary update failed: %s", e)
        
        # Final fallback: concatenate and truncate
        combined = (current_summary + "\n" + chunk_output).strip()
        return truncate_to_tokens(
            self.core.tokenizer,
            combined,
            max_summary_tokens,
            prefer_tail=True
        )

    def process(
        self,
        text: str,
        json_schema: Dict[str, Any],
        prompt_formatter: Callable[[Dict[str, Any]], str],
        reasoning_mode: bool = False,
        chunk_size_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> str:
        """Process text through chunking, per-chunk inference, and merging.
        
        Args:
            text: Input text to process
            json_schema: Pydantic model's JSON schema for structured outputs
            prompt_formatter: Function to format chunk into user prompt
            reasoning_mode: If True, wraps schema to include reasoning_content field
            chunk_size_tokens: Override default chunk size
            overlap_tokens: Override default overlap size
            
        Returns:
            Final merged structured output as JSON string
        """
        # Calculate token budgets
        budgets = self.budget.allocate(self.core.max_model_len)
        
        # The prompt_tokens budget is for the CHUNK TEXT only.
        # The full prompt to the model will be: chunk + summary + boilerplate
        # Total input = prompt_tokens + summary_tokens + ~100 (boilerplate)
        # This should fit within (model_max_len - output_tokens)
        chunk_size_tokens = chunk_size_tokens or budgets['prompt_tokens']
        # Use 20% overlap to avoid breaking mid-sentence
        overlap_tokens = overlap_tokens or max(64, chunk_size_tokens // 5)
        summary_max_tokens = budgets['summary_tokens']
        
        # Debug: Count input text tokens
        input_token_count = self.core.count_tokens(text)
        logger.info("\n" + "─"*80)
        logger.info("📄 INPUT PHASE")
        logger.info(
            "  Input text: %d tokens, %d chars",
            input_token_count, len(text)
        )
        
        # Prepare schema (wrap if reasoning mode)
        if reasoning_mode:
            final_schema = {
                "type": "object",
                "properties": {
                    "reasoning_content": {"type": "string"},
                    "output": json_schema,
                },
                "required": ["reasoning_content", "output"],
            }
        else:
            final_schema = json_schema
        
        # Store schema for merge operations
        self._current_schema = final_schema
        
        # Chunk the text
        chunks = chunk_text_by_tokens(
            self.core.tokenizer,
            text,
            chunk_size_tokens,
            overlap_tokens
        )
        
        logger.info("\n" + "─"*80)
        logger.info("✂️ CHUNKING PHASE")
        logger.info(
            "  Processing %d chunks (chunk_size=%d tokens [budget: %d], overlap=%d)",
            len(chunks), chunk_size_tokens, budgets['prompt_tokens'], overlap_tokens
        )
        
        # Debug: Log actual chunk token counts
        for idx, chunk in enumerate(chunks[:3]):  # Log first 3 chunks to avoid spam
            chunk_tokens = self.core.count_tokens(chunk)
            logger.info(
                "  Chunk %d/%d: %d tokens (%.1f%% of chunk budget)",
                idx + 1, len(chunks), chunk_tokens,
                (chunk_tokens / budgets['prompt_tokens'] * 100) if budgets['prompt_tokens'] > 0 else 0
            )
        if len(chunks) > 3:
            logger.info("  ... (%d more chunks)", len(chunks) - 3)
        
        # Prepare sampling params for chunk processing
        # Use temperature=0.0 for reliable extraction (override config)
        so = StructuredOutputsParams(json=final_schema)
        chunk_sampling_params = SamplingParams(
            max_tokens=budgets['output_tokens'],
            temperature=0.0,  # Deterministic for extraction tasks
            top_p=1.0,
            top_k=0,
            structured_outputs=so,
        )
        
        # Process chunks
        logger.info("\n" + "─"*80)
        logger.info("⚙️ PROCESSING CHUNKS")
        running_summary = ""
        chunk_outputs: List[str] = []
        
        for i, chunk in enumerate(chunks):
            # Build plain text prompt for this chunk
            chunk_context = {"text": chunk}
            prompt = prompt_formatter(chunk_context)
            
            # Prepend summary if available
            if running_summary:
                prompt = (
                    f"PREVIOUS SUMMARY:\n{running_summary}\n\n"
                    f"PROCESS CHUNK {i+1}/{len(chunks)}:\n\n"
                    f"{prompt}"
                )
            else:
                prompt = (
                    f"PROCESS CHUNK {i+1}/{len(chunks)}:\n\n"
                    f"{prompt}"
                )
            
            # Add instructions
            # Note: /no_think disables Qwen3's thinking mode which can cause infinite loops
            # with structured JSON outputs at low temperatures
            if len(chunks) > 1 and running_summary and running_summary.strip() not in ('', '...', '"..."'):
                prompt += (
                    "\n\nInstructions: Extract key information from this chunk. "
                    "Be specific and detailed - include actual facts, definitions, steps, and reasoning, not just section headers. "
                    "Use the previous summary for context if this chunk starts mid-sentence. "
                    "Output must be valid JSON matching the schema. /no_think"
                )
            else:
                prompt += (
                    "\n\nInstructions: Extract key information. "
                    "Be specific and detailed - include actual facts, definitions, steps, and reasoning, not just section headers. "
                    "Output must be valid JSON matching the schema. /no_think"
                )
            
            # Generate for this chunk
            chunk_output = self.core.generate_once(prompt, chunk_sampling_params)
            
            if not chunk_output.strip():
                logger.warning("Empty output for chunk %d/%d", i+1, len(chunks))
                chunk_output = "{}"
            else:
                chunk_output = _extract_json(chunk_output)
            
            chunk_outputs.append(chunk_output)
            
            # Debug: Log chunk output size
            output_tokens = self.core.count_tokens(chunk_output)
            logger.info(
                "  ✅ Chunk %d/%d complete: %d tokens output (budget: %d, usage: %.1f%%)",
                i+1, len(chunks), output_tokens, budgets['output_tokens'],
                (output_tokens / budgets['output_tokens'] * 100) if budgets['output_tokens'] > 0 else 0
            )
            
            # Add to merger (for incremental strategy)
            if self.merger_strategy == "incremental":
                self.merger.add_and_maybe_compact(chunk_output, running_summary)
            
            # Update running summary (skip for last chunk or if single chunk)
            if self.use_summarization and len(chunks) > 1 and i < len(chunks) - 1:
                old_summary_tokens = self.core.count_tokens(running_summary) if running_summary else 0
                running_summary = self._update_summary(
                    running_summary,
                    chunk_output,
                    summary_max_tokens
                )
                new_summary_tokens = self.core.count_tokens(running_summary)
                logger.info(
                    "  📝 Summary updated: %d → %d tokens (budget: %d, usage: %.1f%%)",
                    old_summary_tokens, new_summary_tokens, budgets['summary_tokens'],
                    (new_summary_tokens / budgets['summary_tokens'] * 100) if budgets['summary_tokens'] > 0 else 0
                )
        
        # Final merge
        if self.merger_strategy == "incremental":
            final_output = self.merger.finalize(running_summary)
        else:
            # Hierarchical strategy - merge all at once
            final_output = self.merger.merge_all(chunk_outputs, running_summary)
        
        final_output = _extract_json(final_output)
        
        # Debug: Final output stats
        final_output_tokens = self.core.count_tokens(final_output)
        logger.info("\n" + "─"*80)
        logger.info("✨ FINAL OUTPUT")
        logger.info(
            "  Final output: %d tokens, %d chars",
            final_output_tokens, len(final_output)
        )
        logger.info(
            "  Total efficiency: Input=%d tokens → Output=%d tokens (%.1f%% compression)",
            input_token_count, final_output_tokens,
            (1 - final_output_tokens / input_token_count) * 100 if input_token_count > 0 else 0
        )
        logger.info("─"*80 + "\n")
        
        return final_output
