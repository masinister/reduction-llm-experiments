from typing import Any, Callable, Dict, Optional, List, Tuple
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pydantic import BaseModel, Field
from src import config
from src import utils
import json

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="A concise summary of the content processed so far.")

class Backend:
    def __init__(self) -> None:
        self.llm = LLM(
            config.MODEL_ID,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            max_model_len=config.MAX_MODEL_LEN,
            compilation_config={"cudagraph_mode": config.CUDAGRAPH_MODE},
            dtype=config.DTYPE,
        )
        # tokenizer shortcut
        self.tokenizer = self.llm.get_tokenizer()

    # ------------------------------
    # Helper: split into token-aware chunks with overlap
    # ------------------------------
    def _token_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split `text` into token-count-based chunks with `overlap` tokens between chunks.
        Returns list of chunk strings (decoded).
        """
        enc = self.tokenizer.encode(text)  # returns list of token ids
        n = len(enc)
        if n <= chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunk_ids = enc[start:end]
            chunks.append(self.tokenizer.decode(chunk_ids))
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    # ------------------------------
    # Helper: build chat prompt using system + user style (same as your existing apply_chat_template usage)
    # ------------------------------
    def _make_full_prompt(self, system_prompt: str, user_content: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return full_prompt

    # ------------------------------
    # Helper: Merge a list of outputs into one
    # ------------------------------
    def _merge_outputs(
        self,
        outputs: List[str],
        running_summary: str,
        system_prompt: str,
        sampling_params: SamplingParams
    ) -> str:
        assembly_user_text = (
            "MERGE OPERATION:\n"
            "You are merging partial structured outputs from a large document processing task.\n"
            f"Global Context Summary:\n{running_summary}\n\n"
            "Partial Outputs to Merge:\n" + "\n\n---\n\n".join(f"ITEM {i+1}:\n{c}" for i, c in enumerate(outputs))
            + "\n\nInstructions: Combine these partial outputs into a single coherent JSON output that conforms to the schema. "
            "Resolve any duplicates or conflicts using the global summary as a guide."
        )
        full_prompt = self._make_full_prompt(system_prompt, assembly_user_text)
        results = self.llm.generate([full_prompt], sampling_params=sampling_params)
        if results and results[0].outputs:
            return results[0].outputs[0].text
        return ""

    # ------------------------------
    # Helper: Hierarchical merge
    # ------------------------------
    def _hierarchical_merge(
        self,
        chunk_outputs: List[str],
        running_summary: str,
        system_prompt: str,
        sampling_params: SamplingParams,
        batch_size: int = 5
    ) -> str:
        # If single item, we are done
        if len(chunk_outputs) == 1:
            return chunk_outputs[0]
        
        # If fits in one batch, merge once
        if len(chunk_outputs) <= batch_size:
            return self._merge_outputs(chunk_outputs, running_summary, system_prompt, sampling_params)
        
        # Otherwise, split and recurse
        next_level_outputs = []
        for i in range(0, len(chunk_outputs), batch_size):
            batch = chunk_outputs[i : i + batch_size]
            # Merge this batch
            merged = self._merge_outputs(batch, running_summary, system_prompt, sampling_params)
            next_level_outputs.append(merged)
            
        # Recurse on the results
        return self._hierarchical_merge(next_level_outputs, running_summary, system_prompt, sampling_params, batch_size)


    def inference(
        self,
        context: Dict[str, Any],
        prompt_formatter: Callable[[Dict[str, Any]], str],
        json_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        reasoning_mode: bool = False,
        chunk_size_tokens: int = config.MAX_MODEL_LEN // 8,
        overlap_tokens: int = config.MAX_MODEL_LEN // 16,
        summary_max_tokens: int = config.MAX_MODEL_LEN // 8,
        assemble_final: bool = True,
        use_summarization: bool = True,
    ) -> str:
        """
        Process a prompt by chunking (if needed), iteratively summarizing, and stitching.
        - chunk_size_tokens: target tokens per chunk
        - overlap_tokens: token overlap between chunks
        - summary_max_tokens: how long the running summary should be
        - assemble_final: whether to do a final pass to assemble structured output
        - use_summarization: whether to maintain a running summary across chunks
        """

        if system_prompt is None:
            system_prompt = f"reasoning mode = {'on' if reasoning_mode else 'off'}.\n{config.SYSTEM_PROMPT}"

        # Build the raw user prompt from the provided formatter + context
        user_prompt = prompt_formatter(context)

        # STEP 1: chunk the user_prompt
        chunks = self._token_chunk_text(user_prompt, chunk_size_tokens, overlap_tokens)

        # Prepare structured output schema
        if reasoning_mode:
            final_schema = {
                "type": "object",
                "properties": {
                    "reasoning_content": {"type": "string"},
                    "output": json_schema
                },
                "required": ["reasoning_content", "output"],
                "additionalProperties": False
            }
        else:
            final_schema = json_schema
        so = StructuredOutputsParams(json=final_schema)

        # Sampling params for chunk-level processing (conservative max_tokens)
        chunk_sampling_params = SamplingParams(
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            structured_outputs=so,
        )

        # Sampling params for summary-updating (shorter)
        so_summary = StructuredOutputsParams(json=SummaryResponse.model_json_schema())
        summary_sampling_params = SamplingParams(
            max_tokens=summary_max_tokens,
            temperature=0.0,
            structured_outputs=so_summary,
        )

        # Running summary of previously-seen content (start empty)
        running_summary = ""

        # Store per-chunk outputs (for assembling later)
        chunk_outputs: List[str] = []
        # chunk_reasoning: List[str] = []

        # Iterate through chunks, processing each with the LLM
        for i, chunk in enumerate(chunks):
            # For each chunk we provide:
            # - the system prompt
            # - a short running_summary of previous chunks (if any)
            # - the current chunk
            chunk_user_text = (
                "PREVIOUS SUMMARY:\n" + running_summary + "\n\n" if running_summary else ""
            )
            chunk_user_text += f"PROCESS CHUNK {i+1}/{len(chunks)}:\n\n{chunk}\n\n"
            chunk_user_text += (
                "Instructions: Analyze this chunk and extract information relevant to the task. "
                "Output must conform to the requested JSON schema."
            )

            full_prompt = self._make_full_prompt(system_prompt, chunk_user_text)

            # Generate structured output for this chunk
            outputs = self.llm.generate([full_prompt], sampling_params=chunk_sampling_params)
            chunk_text = ""
            if outputs and outputs[0].outputs:
                chunk_text = outputs[0].outputs[0].text
            chunk_outputs.append(chunk_text)

            # Skip summarization if there's only one chunk (no need to summarize for next chunk or merge)
            if len(chunks) == 1:
                continue

            if not use_summarization:
                continue

            # --- Update running_summary by asking the model to compress running_summary + chunk_text ---
            summary_user_text = (
                "CONCISE SUMMARY TASK:\n"
                "Given the previous summary and the just-processed chunk result, produce a short (1-2 sentence) "
                "summary that captures the important information needed to reason across chunks.\n\n"
                f"Previous summary:\n{running_summary}\n\n"
                f"Chunk result:\n{chunk_text}\n\n"
                "Return JSON: {\"summary\": \"...\"}"
            )
            summary_prompt = self._make_full_prompt(system_prompt, summary_user_text)
            s_outputs = self.llm.generate([summary_prompt], sampling_params=summary_sampling_params)
            
            new_summary = None
            if s_outputs and s_outputs[0].outputs:
                raw_summary = s_outputs[0].outputs[0].text
                # Try to parse as JSON using Pydantic
                try:
                    parsed = SummaryResponse.model_validate_json(raw_summary)
                    new_summary = parsed.summary
                except Exception:
                    # If not JSON but has text, use it (model might have ignored JSON instruction)
                    if raw_summary.strip():
                        new_summary = raw_summary.strip()

            if new_summary:
                running_summary = new_summary
            else:
                # Fallback: if summarization failed, append chunk text but truncate to avoid explosion
                combined = (running_summary + "\n" + chunk_text).strip()
                limit_chars = summary_max_tokens * 4
                if len(combined) > limit_chars:
                    # Keep the end as it's more likely to be relevant for immediate next chunk context
                    combined = "..." + combined[-limit_chars:]
                running_summary = combined

        # Optionally assemble final output: give the model all chunk outputs + final summary,
        # and ask it to produce a single final structured JSON according to the original schema.
        if assemble_final:
            # Use hierarchical merge to handle potentially large number of chunks
            return self._hierarchical_merge(
                chunk_outputs,
                running_summary,
                system_prompt,
                chunk_sampling_params,
                batch_size=5
            )

        # If assemble_final disabled, just return concatenation of chunk_outputs as a JSON list
        return "[" + ",".join(chunk_outputs) + "]"
