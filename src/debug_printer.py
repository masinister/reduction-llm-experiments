"""Debug pretty-printer for LLM prompts and responses."""

import json
from typing import Any

from pydantic import BaseModel

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _box_line(char: str = "─", width: int = 70) -> str:
    """Create a horizontal line for box drawing."""
    return char * width


def _header(title: str, color: str, width: int = 70) -> str:
    """Create a styled header."""
    padding = (width - len(title) - 4) // 2
    return (
        f"\n{color}{BOLD}"
        f"╭{_box_line('─', width)}╮\n"
        f"│{' ' * padding} {title} {' ' * (width - padding - len(title) - 2)}│\n"
        f"╰{_box_line('─', width)}╯"
        f"{RESET}\n"
    )


def _format_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty JSON with colors."""
    if isinstance(data, BaseModel):
        data = data.model_dump()
    
    formatted = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    
    # Add subtle coloring to JSON structure
    lines = []
    for line in formatted.split("\n"):
        # Color keys (strings followed by colon)
        if '": ' in line or '":' in line:
            parts = line.split('": ', 1)
            if len(parts) == 2:
                key_part = parts[0] + '"'
                value_part = parts[1]
                line = f"{CYAN}{key_part}{RESET}: {_color_value(value_part)}"
        lines.append(line)
    
    return "\n".join(lines)


def _color_value(value: str) -> str:
    """Apply color to a JSON value based on its type."""
    stripped = value.rstrip(",")
    suffix = value[len(stripped):]
    
    if stripped.startswith('"'):
        return f"{GREEN}{stripped}{RESET}{suffix}"
    elif stripped in ("true", "false"):
        return f"{MAGENTA}{stripped}{RESET}{suffix}"
    elif stripped == "null":
        return f"{DIM}{stripped}{RESET}{suffix}"
    elif stripped.replace(".", "").replace("-", "").isdigit():
        return f"{YELLOW}{stripped}{RESET}{suffix}"
    else:
        return value


def _wrap_text(text: str, width: int = 70, indent: str = "  ") -> str:
    """Wrap text to specified width with indentation."""
    lines = []
    for paragraph in text.split("\n"):
        if len(paragraph) <= width:
            lines.append(f"{indent}{paragraph}")
        else:
            words = paragraph.split()
            current_line = indent
            for word in words:
                if len(current_line) + len(word) + 1 <= width + len(indent):
                    if current_line == indent:
                        current_line += word
                    else:
                        current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = indent + word
            if current_line.strip():
                lines.append(current_line)
    return "\n".join(lines)


def _count_tokens(text: str) -> int:
    """Approximate token count (chars / 4)."""
    return len(text) // 4


def _token_info(label: str, text: str) -> str:
    """Format token count info."""
    tokens = _count_tokens(text)
    chars = len(text)
    return f"{DIM}  [{label}: ~{tokens:,} tokens, {chars:,} chars]{RESET}"


def print_prompt(prompt: str) -> None:
    """Pretty-print a prompt being sent to the LLM."""
    print(_header("PROMPT", CYAN))
    print(_token_info("input", prompt))
    print(_wrap_text(prompt))
    print(f"\n{DIM}{_box_line('·')}{RESET}\n")


def print_response(response: BaseModel | dict | Any) -> None:
    """Pretty-print a response received from the LLM."""
    print(_header("RESPONSE", GREEN))
    
    # Get JSON string for token counting
    if isinstance(response, BaseModel):
        response_str = response.model_dump_json()
    else:
        response_str = json.dumps(response, default=str)
    
    print(_token_info("output", response_str))
    print(_format_json(response))
    print(f"\n{DIM}{_box_line('·')}{RESET}\n")


def print_error(error: Exception | str) -> None:
    """Pretty-print an error."""
    RED = "\033[91m"
    print(_header("ERROR", RED))
    print(f"  {RED}{error}{RESET}")
    print(f"\n{DIM}{_box_line('·')}{RESET}\n")
