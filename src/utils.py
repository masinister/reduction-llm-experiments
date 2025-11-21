from typing import Optional
import json

def _find_json_block(text: str) -> Optional[str]:
    """
    Try to find a top-level JSON object in text.
    Naive approach: find first '{' and the matching '}'.
    Falls back to None if parse fails.
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
                    # quick parse check
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return None
    return None

def _safe_json_load(text: str) -> Optional[dict]:
    """
    Extract & parse JSON block from text using _find_json_block. Return dict or None.
    """
    j = _find_json_block(text)
    if not j:
        return None
    try:
        return json.loads(j)
    except Exception:
        return None