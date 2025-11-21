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
from src.utils import parse_structured_output
from src import config

logger = logging.getLogger(__name__)


class SummaryResponse(BaseModel):
    """Pydantic model for summary updates."""
    summary: str = Field(..., description="A concise summary of the content processed so far.")


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
        
        # Build merge prompt
        system = getattr(config, "SYSTEM_PROMPT", "You are a helpful assistant.")
        
        merge_instructions = (
            "MERGE OPERATION:\n"
            "You are merging partial structured outputs from a large document processing task.\n"
        )
        
        if running_summary:
            merge_instructions += f"Global Context Summary:\n{running_summary}\n\n"
        
        merge_instructions += "Partial Outputs to Merge:\n"
        merge_instructions += "\n\n---\n\n".join(
            f"ITEM {i+1}:\n{output}" for i, output in enumerate(outputs_batch)
        )
        merge_instructions += (
            "\n\nInstructions: Combine these partial outputs into a single coherent JSON output "
            "that conforms to the schema. Resolve any duplicates or conflicts using the global "
            "summary as a guide. If information is missing, be conservative and do not hallucinate. "
            "Output ONLY valid JSON that conforms to the schema - no think blocks or additional text."
        )
        
        # Use structured output with the current schema to ensure valid JSON
        sampling_params = SamplingParams(
            max_tokens=getattr(config, "MAX_TOKENS"),
            temperature=0.0,
            top_p=1.0,
        )
        
        # Add structured outputs if schema is available
        if self._current_schema is not None:
            sampling_params.structured_outputs = StructuredOutputsParams(json=self._current_schema)
        
        merged = self.core.generate_once(system, merge_instructions, sampling_params)
        return merged

    def _update_summary(
        self,
        current_summary: str,
        chunk_output: str,
        system_prompt: str,
        max_summary_tokens: int
    ) -> str:
        """Update the running summary with new chunk output.
        
        Args:
            current_summary: Current summary text
            chunk_output: Latest chunk output to incorporate
            system_prompt: System prompt for generation
            max_summary_tokens: Target token count for summary
            
        Returns:
            Updated summary text
        """
        summary_prompt = (
            "CONCISE SUMMARY TASK:\n"
            "Given the previous summary and the just-processed chunk result, produce a short "
            f"(target: {max_summary_tokens} tokens) summary that captures the important "
            "information needed to reason across chunks.\n\n"
            f"Previous summary:\n{current_summary}\n\n"
            f"Chunk result:\n{chunk_output}\n\n"
            "Return JSON with format: {\"summary\": \"...\"}"
        )
        
        # Use structured output for summary
        so_summary = StructuredOutputsParams(json=SummaryResponse.model_json_schema())
        summary_sampling_params = SamplingParams(
            max_tokens=max_summary_tokens,
            temperature=0.0,
            structured_outputs=so_summary,
        )
        
        try:
            raw_summary = self.core.generate_once(
                system_prompt,
                summary_prompt,
                summary_sampling_params
            )
            
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
        system_prompt: Optional[str] = None,
        reasoning_mode: bool = False,
        chunk_size_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> str:
        """Process text through chunking, per-chunk inference, and merging.
        
        Args:
            text: Input text to process
            json_schema: Pydantic model's JSON schema for structured outputs
            prompt_formatter: Function to format chunk into user prompt
            system_prompt: Optional system prompt (uses config default if None)
            reasoning_mode: If True, wraps schema to include reasoning_content field
            chunk_size_tokens: Override default chunk size
            overlap_tokens: Override default overlap size
            
        Returns:
            Final merged structured output as JSON string
        """
        # Calculate token budgets
        budgets = self.budget.allocate(self.core.max_model_len)
        
        # Use budget-based defaults if not specified
        chunk_size_tokens = chunk_size_tokens or budgets['prompt_tokens']
        overlap_tokens = overlap_tokens or max(32, chunk_size_tokens // 8)
        summary_max_tokens = budgets['summary_tokens']
        
        # Debug: Count input text tokens
        input_token_count = self.core.count_tokens(text)
        logger.info(
            "[BUDGET] Input text: %d tokens, %d chars",
            input_token_count, len(text)
        )
        
        # Set system prompt
        if system_prompt is None:
            reasoning_flag = "on" if reasoning_mode else "off"
            system_prompt = f"Reasoning mode = {reasoning_flag}.\n{getattr(config, 'SYSTEM_PROMPT', '')}"
        
        # Debug: Count system prompt tokens
        system_token_count = self.core.count_tokens(system_prompt) if system_prompt else 0
        logger.info(
            "[BUDGET] System prompt: %d tokens (budget: %d, usage: %.1f%%)",
            system_token_count, budgets['system_tokens'],
            (system_token_count / budgets['system_tokens'] * 100) if budgets['system_tokens'] > 0 else 0
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
                "additionalProperties": False,
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
        
        logger.info(
            "[BUDGET] Processing %d chunks (chunk_size=%d tokens [budget: %d], overlap=%d)",
            len(chunks), chunk_size_tokens, budgets['prompt_tokens'], overlap_tokens
        )
        
        # Debug: Log actual chunk token counts
        for idx, chunk in enumerate(chunks[:3]):  # Log first 3 chunks to avoid spam
            chunk_tokens = self.core.count_tokens(chunk)
            logger.info(
                "[BUDGET] Chunk %d/%d: %d tokens (%.1f%% of chunk budget)",
                idx + 1, len(chunks), chunk_tokens,
                (chunk_tokens / budgets['prompt_tokens'] * 100) if budgets['prompt_tokens'] > 0 else 0
            )
        if len(chunks) > 3:
            logger.info("[BUDGET] ... (%d more chunks)", len(chunks) - 3)
        
        # Prepare sampling params for chunk processing
        so = StructuredOutputsParams(json=final_schema)
        chunk_sampling_params = SamplingParams(
            max_tokens=budgets['output_tokens'],
            temperature=getattr(config, "TEMPERATURE", 0.0),
            top_p=getattr(config, "TOP_P", 1.0),
            top_k=getattr(config, "TOP_K", 0),
            structured_outputs=so,
        )
        
        # Process chunks
        running_summary = ""
        chunk_outputs: List[str] = []
        
        for i, chunk in enumerate(chunks):
            # Build user prompt for this chunk
            chunk_context = {"text": chunk}
            chunk_user_prompt = prompt_formatter(chunk_context)
            
            # Prepend summary if available
            if running_summary:
                chunk_user_prompt = (
                    f"PREVIOUS SUMMARY:\n{running_summary}\n\n"
                    f"PROCESS CHUNK {i+1}/{len(chunks)}:\n\n"
                    f"{chunk_user_prompt}"
                )
            else:
                chunk_user_prompt = (
                    f"PROCESS CHUNK {i+1}/{len(chunks)}:\n\n"
                    f"{chunk_user_prompt}"
                )
            
            # Add instructions
            chunk_user_prompt += (
                "\n\nInstructions: Analyze this chunk and extract information relevant to "
                "the task. Output must conform to the requested JSON schema."
            )
            
            # Generate for this chunk
            chunk_output = self.core.generate_once(
                system_prompt,
                chunk_user_prompt,
                chunk_sampling_params
            )
            
            if not chunk_output.strip():
                logger.warning("Empty output for chunk %d/%d", i+1, len(chunks))
                chunk_output = "{}"  # Fallback to empty JSON
            
            chunk_outputs.append(chunk_output)
            
            # Debug: Log chunk output size
            output_tokens = self.core.count_tokens(chunk_output)
            logger.info(
                "[BUDGET] Chunk %d/%d output: %d tokens (budget: %d, usage: %.1f%%)",
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
                    system_prompt,
                    summary_max_tokens
                )
                new_summary_tokens = self.core.count_tokens(running_summary)
                logger.info(
                    "[BUDGET] Summary updated: %d -> %d tokens (budget: %d, usage: %.1f%%)",
                    old_summary_tokens, new_summary_tokens, budgets['summary_tokens'],
                    (new_summary_tokens / budgets['summary_tokens'] * 100) if budgets['summary_tokens'] > 0 else 0
                )
        
        # Final merge
        if self.merger_strategy == "incremental":
            final_output = self.merger.finalize(running_summary)
        else:
            # Hierarchical strategy - merge all at once
            final_output = self.merger.merge_all(chunk_outputs, running_summary)
        
        # Debug: Final output stats
        final_output_tokens = self.core.count_tokens(final_output)
        logger.info(
            "[BUDGET] Pipeline complete. Final output: %d tokens, %d chars",
            final_output_tokens, len(final_output)
        )
        logger.info(
            "[BUDGET] Total efficiency: Input=%d tokens -> Output=%d tokens (%.1f%% compression)",
            input_token_count, final_output_tokens,
            (1 - final_output_tokens / input_token_count) * 100 if input_token_count > 0 else 0
        )
        
        return final_output
