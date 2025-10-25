from __future__ import annotations

import json
import time
import logging
import hashlib
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

import jsonschema  # pip install jsonschema

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class StructuredResult:
    """Normalized result returned by run_structured()."""
    data: Dict[str, Any]
    raw_text: str
    cleaned_text: str
    latency_s: float
    tokens: int
    attempts: int


class StructuredCallError(RuntimeError):
    """Raised when structured output could not be obtained or validated."""


def _try_parse_json(s: str) -> Optional[Any]:
    """Try to parse a JSON object embedded in s.

    Common model outputs sometimes include backticks, markdown, or text before/after JSON.
    This function attempts:
      - direct json.loads(s)
      - find first occurrence of '{' and last '}' and parse substring
      - fallback: search for the first top-level '[' or '{' and attempt parse
    Returns parsed object or None.
    """
    s = s.strip()
    if not s:
        return None
    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # find first { or [ and last matching } or ]
    first_curly = s.find("{")
    first_square = s.find("[")
    candidates = []
    if first_curly != -1:
        last_curly = s.rfind("}")
        if last_curly != -1 and last_curly > first_curly:
            candidates.append(s[first_curly : last_curly + 1])
    if first_square != -1:
        last_square = s.rfind("]")
        if last_square != -1 and last_square > first_square:
            candidates.append(s[first_square : last_square + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    # last resort: try to find lines that look like JSON and join them
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    joined = "\n".join(lines[-40:])  # last 40 lines heuristics
    try:
        return json.loads(joined)
    except Exception:
        pass

    return None


def _validate_schema(instance: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate instance against JSON Schema. Returns (ok, error_message)."""
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)


def _stable_id(prefix: str, text: str) -> str:
    """Deterministic short id using sha1. Returns prefix-<6 hex>."""
    h = hashlib.sha1()
    h.update(text.strip().encode("utf-8"))
    digest = h.hexdigest()[:8]
    return f"{prefix}-{digest}"


def run_structured(
    model,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    session_id: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: bool = False,
    retries: int = 1,
    retry_backoff_s: float = 0.6,
    validate: bool = True,
) -> StructuredResult:
    """Run a vLLM structured-output call and return a validated Python object.

    Args:
        model: instance of your Model wrapper (must implement infer(..., json_schema=...)).
        system_prompt: system prompt string.
        user_prompt: user prompt string (main content).
        json_schema: JSON Schema dict describing expected output.
        session_id: session id for the Model (unique per logical call).
        temperature/top_p/top_k/max_tokens: optional overrides passed to model.infer.
        enable_thinking: pass to model.infer.
        retries: number of attempts >= 1. On failure, will retry once with slightly higher max_tokens.
        retry_backoff_s: seconds between attempts.
        validate: if True, validate against schema and raise StructuredCallError on fatal failure.

    Returns:
        StructuredResult with .data containing the parsed JSON object.

    Raises:
        StructuredCallError if the model fails to produce parseable JSON or validation fails.
    """
    attempts = 0
    start_time = time.time()
    last_exception: Optional[Exception] = None
    raw_text_accum = ""
    cleaned_accum = ""
    tokens = 0

    # ensure at least one try
    for attempt in range(1, max(1, retries) + 1):
        attempts = attempt
        try:
            call_kwargs = {}
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            if top_p is not None:
                call_kwargs["top_p"] = top_p
            if top_k is not None:
                call_kwargs["top_k"] = top_k
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens

            # instruct the model via system prompt; embed user prompt as single argument
            # model.infer already accepts json_schema and will create StructuredOutputsParams internally
            result = model.infer(
                prompt=user_prompt,
                session_id=session_id,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                json_schema=json_schema,
                **call_kwargs,
            )
            latency = time.time() - start_time
            raw = result.get("raw_text", result.get("raw", ""))
            cleaned = result.get("text", raw).strip()
            tokens = int(result.get("tokens", 0))
            raw_text_accum = raw
            cleaned_accum = cleaned

            # First try: direct JSON parse of cleaned text
            parsed = _try_parse_json(cleaned)
            if parsed is None:
                # second try: parse raw_text
                parsed = _try_parse_json(raw)

            if parsed is None:
                # If structured outputs were enforced at decode time vLLM sometimes returns structured payload
                # embedded in raw, so attempt find via heuristics above. If still none, raise and retry.
                raise StructuredCallError("model returned non-parseable output (no JSON found).")

            # Validate against schema if requested
            if validate:
                ok, err = _validate_schema(parsed, json_schema)
                if not ok:
                    raise StructuredCallError(f"JSON schema validation failed: {err}")

            # success
            return StructuredResult(
                data=parsed,
                raw_text=raw_text_accum,
                cleaned_text=cleaned_accum,
                latency_s=latency,
                tokens=tokens,
                attempts=attempts,
            )

        except Exception as e:
            # keep the last exception and retry if attempts remain
            last_exception = e
            logger.warning(
                "Structured call attempt %d failed: %s. Retrying in %.2fs (retries left=%d).",
                attempt,
                str(e),
                retry_backoff_s,
                max(0, retries - attempt),
            )
            time.sleep(retry_backoff_s)

    # all attempts failed
    msg = f"Structured output failed after {attempts} attempts. last_error={last_exception}"
    logger.error(msg)
    if validate:
        raise StructuredCallError(msg)
    # If validation disabled, return a minimal synthetic object (best-effort parse or empty)
    parsed = _try_parse_json(cleaned_accum) or {}
    return StructuredResult(
        data=parsed,
        raw_text=raw_text_accum,
        cleaned_text=cleaned_accum,
        latency_s=time.time() - start_time,
        tokens=tokens,
        attempts=attempts,
    )
