from typing import Any, Dict, List
from pydantic import BaseModel
from src.backend import Backend
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

    backend = Backend()
    
    print("\n--- Normal Inference ---\n")
    out = backend.inference(context, prompt_formatter, person_schema)
    print(out)

    print("\n--- System Prompt Inference ---\n")
    out_sys = backend.inference(
        context, 
        prompt_formatter, 
        person_schema, 
        system_prompt="Bob is short for Robert -- Replace all occurrences of 'Bob' with 'Robert'."
    )
    print(out_sys)

    print("\n--- Reasoning Mode Inference ---\n")
    out_reason = backend.inference(
        context, 
        prompt_formatter, 
        person_schema, 
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
    out_long = backend.inference(
        context_long,
        long_context_formatter,
        key_points_schema,
        chunk_size_tokens=512, # Small limit to force multiple chunks
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

    print("\n--- Long Context No Assembly Inference ---\n")
    # Test assemble_final=False
    out_no_assembly = backend.inference(
        context_long,
        long_context_formatter,
        key_points_schema,
        chunk_size_tokens=512,
        overlap_tokens=64,
        assemble_final=False,
        use_summarization=False
    )
    print(out_no_assembly[:200] + "...") # Print start to verify it's a list
    
    try:
        data_no_assembly = json.loads(out_no_assembly)
        if isinstance(data_no_assembly, list) and len(data_no_assembly) > 0:
            print("No assembly mode verified. Returned list of chunks.")
            print(f"Number of chunks: {len(data_no_assembly)}")
        else:
            print("No assembly mode FAILED.")
    except Exception as e:
        print(f"No assembly mode output is not valid JSON: {e}")

if __name__ == "__main__":
    main()
