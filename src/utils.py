"""Shared utility functions for JSON parsing and data validation.

This module provides defensive JSON extraction and parsing utilities used
throughout the codebase.
"""

from typing import Any, Dict, Optional, Type, TypeVar
import json
import logging
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def find_json_block(text: str) -> Optional[str]:
    """Extract the first valid JSON object from text.
    
    Uses brace-matching with depth tracking to find a complete JSON object.
    Falls back to None if no valid JSON is found.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        JSON string or None if not found
    """
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    # Validate by parsing
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return None
    return None


def safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from text.
    
    Attempts to find a JSON object in the text and parse it. Returns None
    if extraction or parsing fails.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed dictionary or None
    """
    j = find_json_block(text)
    if not j:
        return None
    try:
        return json.loads(j)
    except Exception as e:
        logger.warning("safe_json_load: JSON parse failed: %s", e)
        return None


def parse_structured_output(
    text: str,
    model: Type[T],
    fallback_extract: bool = True
) -> Optional[T]:
    """Parse and validate structured output using Pydantic model.
    
    Tries multiple strategies:
    1. Direct JSON parsing of entire text
    2. Extract first JSON block from text (if fallback_extract=True)
    3. Pydantic validation of parsed JSON
    
    Args:
        text: Text containing structured output
        model: Pydantic model class for validation
        fallback_extract: If True, try extracting JSON block on parse failure
        
    Returns:
        Validated Pydantic model instance or None on failure
    """
    if not text or not text.strip():
        logger.warning("parse_structured_output: empty text")
        return None
    
    # Try direct parse first
    try:
        data = json.loads(text)
        return model.model_validate(data)
    except json.JSONDecodeError:
        if not fallback_extract:
            return None
        # Try extraction fallback
        pass
    except ValidationError as e:
        logger.warning("parse_structured_output: validation failed: %s", e)
        return None
    
    # Fallback: extract JSON block
    if fallback_extract:
        extracted = safe_json_load(text)
        if extracted:
            try:
                return model.model_validate(extracted)
            except ValidationError as e:
                logger.warning("parse_structured_output: validation failed after extraction: %s", e)
    
    return None