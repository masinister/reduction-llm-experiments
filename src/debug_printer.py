"""Debug output formatting utilities for pretty-printing prompts and responses.

This module provides consistent, visually distinct formatting for debug output
to make it easy to distinguish between prompts, responses, and processing logs.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


class DebugPrinter:
    """Utility class for formatted debug output."""
    
    @staticmethod
    def print_prompt(prompt: str, sampling_params: "SamplingParams") -> None:
        """Print a formatted prompt with sampling parameters and schema info.
        
        Args:
            prompt: The prompt text to display
            sampling_params: Sampling parameters used for generation
        """
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*25 + "🔵 PROMPT TO LLM" + " "*37 + "║")
        print("╚" + "="*78 + "╝")
        
        # Prompt content is what actually gets sent to the LLM
        print("\n┌─ Prompt Text " + "─"*64)
        for line in prompt.split('\n'):
            print(f"│ {line}")
        
        # Schema is part of the prompt (constrains LLM output)
        if sampling_params.structured_outputs:
            so = sampling_params.structured_outputs
            if hasattr(so, 'json') and so.json is not None:
                print("├─ Output Schema " + "─"*62)
                schema_str = json.dumps(so.json, indent=2)
                for line in schema_str.split('\n'):
                    print(f"│ {line}")
        print("└" + "─"*78)
        
        # Sampling parameters are config, not part of the prompt
        print(f"\n  ⚙ Sampling: temp={sampling_params.temperature or 0.0:.2f}, "
              f"top_p={sampling_params.top_p or 1.0:.2f}, "
              f"max_tokens={sampling_params.max_tokens or 0}")
        print()
    
    @staticmethod
    def print_response(text: str) -> None:
        """Print a formatted LLM response.
        
        Args:
            text: The generated text to display
        """
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*24 + "🟢 RESPONSE FROM LLM" + " "*34 + "║")
        print("╚" + "="*78 + "╝")
        
        print(f"\n┌─ Generated Text (length={len(text)}) " + "─"*(38-len(str(len(text)))))
        try:
            parsed = json.loads(text)
            formatted_json = json.dumps(parsed, indent=2)
            for line in formatted_json.split('\n'):
                print(f"│ {line}")
        except (json.JSONDecodeError, ValueError):
            for line in text.split('\n'):
                print(f"│ {line}")
        print("└" + "─"*78)
        print()
    
    @staticmethod
    def print_section_header(title: str, icon: str = "📋") -> None:
        """Print a section header for pipeline phases.
        
        Args:
            title: Section title text
            icon: Emoji/icon to display before title
        """
        print("\n" + "─"*80)
        print(f"{icon} {title}")
    
    @staticmethod
    def print_separator() -> None:
        """Print a simple separator line."""
        print("─"*80)
