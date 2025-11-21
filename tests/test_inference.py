from typing import Any, Dict, List
from pydantic import BaseModel
from src.core_backend import CoreBackend
from src.pipeline import Pipeline
from src.context_budget import ContextBudget
import json
import os

class Person(BaseModel):
    name: str
    age: int

class KeyPoints(BaseModel):
    points: List[str]

def prompt_formatter(context: Dict[str, Any]) -> str:
    return f"Extract the person's name and age from: {context['text']}"

def long_context_formatter(context: Dict[str, Any]) -> str:
    return f"Extract key points from the following text: {context['text']}"

def main() -> None:
    person_schema = Person.model_json_schema()
    context = {"text": "Bob is 42 years old."}

    # Initialize core backend and pipeline
    core = CoreBackend()
    pipeline = Pipeline(
        core_backend=core,
        context_budget=ContextBudget(),
        merger_strategy="hierarchical",
        merger_batch_size=5,
        use_summarization=True
    )
    
    try:
        print("\n--- Normal Inference ---\n")
        out = pipeline.process(
            text=context['text'],
            json_schema=person_schema,
            prompt_formatter=prompt_formatter
        )
        print(out)

        print("\n--- System Prompt Inference ---\n")
        out_sys = pipeline.process(
            text=context['text'],
            json_schema=person_schema,
            prompt_formatter=prompt_formatter,
            system_prompt="Bob is short for Robert -- Replace all occurrences of 'Bob' with 'Robert'."
        )
        print(out_sys)

        print("\n--- Reasoning Mode Inference ---\n")
        out_reason = pipeline.process(
            text=context['text'],
            json_schema=person_schema,
            prompt_formatter=prompt_formatter,
            system_prompt="Bob is short for Robert -- Replace all occurrences of 'Bob' with 'Robert'.",
            reasoning_mode=True
        )
        print(out_reason)
        
        try:
            data = json.loads(out_reason)
            if "reasoning_content" in data and "output" in data:
                print("Reasoning mode structure verified.")
            else:
                print("Reasoning mode structure FAILED.")
        except:
            print("Reasoning mode output is not valid JSON.")

        print("\n--- Long Context Chunking Inference ---\n")

        file_path = os.path.join(os.path.dirname(__file__), "example_reduction.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                long_text = fh.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"{file_path} not found. Place example_reduction.txt alongside this test.")

        key_points_schema = KeyPoints.model_json_schema()
        context_long = {"text": long_text}
        
        # Force chunking with small token limit
        out_long = pipeline.process(
            text=long_text,
            json_schema=key_points_schema,
            prompt_formatter=long_context_formatter,
            chunk_size_tokens=512,  # Small limit to force multiple chunks
            overlap_tokens=64
        )
        print(out_long)
        
        try:
            data_long = json.loads(out_long)
            if "points" in data_long and isinstance(data_long["points"], list):
                print("Long context chunking structure verified.")
                print(f"Extracted {len(data_long['points'])} points.")
            else:
                print("Long context chunking structure FAILED.")
        except:
            print("Long context chunking output is not valid JSON.")
    
    finally:
        # Clean up the vLLM engine to prevent crash on exit
        if hasattr(core, 'llm') and core.llm is not None:
            try:
                # Delete the LLM instance explicitly
                del core.llm
            except:
                pass

if __name__ == "__main__":
    main()
