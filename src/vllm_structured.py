from __future__ import annotations

import json
import time
import logging
import hashlib
import uuid
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
    num_input_tokens: Optional[int] = None
    num_output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    reasoning_text: Optional[str] = None
    reasoning_tokens: Optional[int] = None


class StructuredCallError(RuntimeError):
    """Raised when structured output could not be obtained or validated."""


def _strip_thinking_markers(s: str) -> str:
    """Strip thinking/reasoning prefixes and extract actual JSON content."""
    s = s.strip()

    markers = [
        "assistantfinal",
        "assistant",
        "final",
        "json",
        "output",
    ]

    for marker in markers:
        idx = s.lower().rfind(marker)
        if idx != -1:
            s = s[idx + len(marker):].strip()
            break

    first_curly = s.find("{")
    if first_curly >= 0:
        s = s[first_curly:]

    return s.strip()


def _normalize_json_like_text(s: str) -> str:
    """Normalize common unicode punctuation so json.loads has a better chance."""
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
    }
    for src, dest in replacements.items():
        s = s.replace(src, dest)
    return s


def _try_parse_json(s: str) -> Optional[Any]:
    """Parse JSON from string. With vLLM structured outputs, this should reliably succeed.
    
    Fallback extraction is kept minimal since response_format enforcement should
    guarantee valid JSON at decode time.
    """
    s = s.strip()
    if not s:
        return None
    
    # First, strip any thinking/reasoning markers
    s = _strip_thinking_markers(s)
    s = _normalize_json_like_text(s)
    
    try:
        return json.loads(s)
    except Exception:
        pass

    # Minimal fallback: extract content between first { and last }
    first_curly = s.find("{")
    last_curly = s.rfind("}")
    if first_curly != -1 and last_curly != -1 and last_curly > first_curly:
        try:
            candidate = _normalize_json_like_text(s[first_curly : last_curly + 1])
            return json.loads(candidate)
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
    """Deterministic short id using sha1. Returns prefix-<8 hex>."""
    h = hashlib.sha1()
    h.update(text.strip().encode("utf-8"))
    digest = h.hexdigest()[:8]
    return f"{prefix}-{digest}"


# Cache for schema serialization to avoid repeated JSON dumps
_SCHEMA_CACHE: Dict[str, str] = {}


def _get_cached_schema_text(schema: Dict[str, Any]) -> str:
    """Cache serialized schema by hash to avoid repeated JSON dumps."""
    schema_str = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    schema_hash = hashlib.sha1(schema_str.encode("utf-8")).hexdigest()[:12]
    
    if schema_hash not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[schema_hash] = json.dumps(schema, indent=2, ensure_ascii=False)
    
    return _SCHEMA_CACHE[schema_hash]


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
    repair_on_failure: bool = True,
) -> StructuredResult:
    """Run a vLLM structured-output call and return a validated Python object.

    Args:
        model: instance of your Model wrapper (must implement infer(..., response_format=...)).
        system_prompt: system prompt string.
        user_prompt: user prompt string (main content).
        json_schema: JSON Schema dict describing expected output.
        session_id: session id for the Model (unique per logical call).
        temperature/top_p/top_k/max_tokens: optional overrides passed to model.infer.
        enable_thinking: pass to model.infer.
        retries: number of attempts >= 1. Exponential backoff applied between retries.
        retry_backoff_s: initial backoff time in seconds (grows exponentially).
        validate: if True, validate against schema and raise StructuredCallError on fatal failure.
        repair_on_failure: if True, attempt automatic JSON repair on parse/validation failures.

    Returns:
        StructuredResult with .data containing the parsed JSON object, plus metadata
        (latency, tokens, num_input_tokens, num_output_tokens, finish_reason, attempts).

    Raises:
        StructuredCallError if the model fails to produce parseable JSON or validation fails
        after all repair attempts and retries are exhausted.
    """
    attempts = 0
    start_time = time.time()
    last_exception: Optional[Exception] = None
    raw_text_accum = ""
    cleaned_accum = ""
    reasoning_accum: Optional[str] = None
    reasoning_tokens: Optional[int] = None
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

            call_session_id = f"{session_id}-{uuid.uuid4().hex[:8]}-attempt{attempt}"
            # Pass schema via response_format for vLLM decode-time enforcement
            result = model.infer(
                prompt=user_prompt,
                session_id=call_session_id,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                response_format={"type": "json_object", "schema": json_schema},
                **call_kwargs,
            )
            raw = result.get("raw_text", result.get("raw", ""))
            reasoning_text = result.get("reasoning_content")
            if reasoning_text:
                reasoning_accum = reasoning_text
            reasoning_tokens = result.get("reasoning_tokens", reasoning_tokens)

            cleaned_source = result.get("content") or result.get("text") or ""
            if not cleaned_source and isinstance(reasoning_text, str):
                cleaned_source = reasoning_text
            cleaned = str(cleaned_source).strip()
            tokens = int(result.get("tokens", 0))
            raw_text_accum = raw
            cleaned_accum = cleaned
            
            # Capture vLLM-specific metadata if available
            num_input_tokens = result.get("num_input_tokens")
            num_output_tokens = result.get("num_output_tokens")
            finish_reason = result.get("finish_reason")

            # First try: direct JSON parse of cleaned text
            parsed = _try_parse_json(cleaned)
            if parsed is None and isinstance(reasoning_text, str):
                parsed = _try_parse_json(reasoning_text)
            if parsed is None:
                # second try: parse raw_text
                parsed = _try_parse_json(raw)
                
            # Log raw output when parsing fails for debugging
            if parsed is None:
                logger.warning(
                    "JSON parsing failed for session %s. Raw output (first 500 chars): %s",
                    call_session_id,
                    raw[:500] if raw else cleaned[:500]
                )

            repair_attempted = False
            if parsed is None and repair_on_failure:
                repair_attempted = True
                repaired = _attempt_json_repair(
                    model,
                    failed_output=cleaned or raw,
                    json_schema=json_schema,
                    session_id=session_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if repaired is not None:
                    parsed, repair_raw, repair_cleaned, repair_tokens = repaired
                    raw_text_accum = repair_raw
                    cleaned_accum = repair_cleaned
                    reasoning_accum = result.get("reasoning_content", reasoning_accum)
                    tokens += repair_tokens

            if parsed is None:
                # If structured outputs were enforced at decode time vLLM sometimes returns structured payload
                # embedded in raw, so attempt find via heuristics above. If still none, raise and retry.
                raise StructuredCallError("model returned non-parseable output (no JSON found).")

            # Validate against schema if requested
            if validate:
                ok, err = _validate_schema(parsed, json_schema)
                if not ok:
                    if repair_on_failure and not repair_attempted:
                        repair_attempted = True
                        source_text = cleaned if cleaned else json.dumps(parsed, ensure_ascii=False)
                        repaired = _attempt_json_repair(
                            model,
                            failed_output=source_text,
                            json_schema=json_schema,
                            session_id=session_id,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if repaired is not None:
                            parsed, repair_raw, repair_cleaned, repair_tokens = repaired
                            raw_text_accum = repair_raw
                            cleaned_accum = repair_cleaned
                            tokens += repair_tokens
                            ok, err = _validate_schema(parsed, json_schema)
                    if not ok:
                        raise StructuredCallError(f"JSON schema validation failed: {err}")

            # success
            latency = time.time() - start_time
            return StructuredResult(
                data=parsed,
                raw_text=raw_text_accum,
                cleaned_text=cleaned_accum,
                latency_s=latency,
                tokens=tokens,
                attempts=attempts,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                finish_reason=finish_reason,
                reasoning_text=reasoning_accum,
                reasoning_tokens=reasoning_tokens,
            )

        except Exception as e:
            # keep the last exception and retry if attempts remain
            last_exception = e
            backoff_time = retry_backoff_s * (1.5 ** (attempt - 1))
            logger.warning(
                "Structured call attempt %d failed: %s. Retrying in %.2fs (retries left=%d).",
                attempt,
                str(e),
                backoff_time,
                max(0, retries - attempt),
            )
            time.sleep(backoff_time)

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
        reasoning_text=reasoning_accum,
        reasoning_tokens=reasoning_tokens,
    )


def _attempt_json_repair(
    model,
    *,
    failed_output: str,
    json_schema: Dict[str, Any],
    session_id: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Optional[Tuple[Dict[str, Any], str, str, int]]:
    if not failed_output:
        return None

    repair_system_prompt = (
        "You are a JSON repair assistant. Convert the provided text into a valid JSON object "
        "that satisfies the schema. Respond with JSON only."
    )
    schema_text = _get_cached_schema_text(json_schema)
    
    # Extract schema shape hint
    schema_type = json_schema.get("type", "object")
    is_object = schema_type == "object"
    required_keys = json_schema.get("required", []) if is_object else []
    
    shape_hint = ""
    if is_object and required_keys:
        shape_hint = (
            f"\n\nIMPORTANT: The output MUST be a JSON object (not an array) with these required keys: "
            f"{', '.join(required_keys)}. If the failed output is an array of strings, treat them as "
            f"content for the 'reasons' field and construct the full object around them."
        )
    
    repair_user_prompt = (
        "The previous model output failed JSON schema validation. Rewrite it so that it becomes "
        "valid JSON matching the schema below. Keep the substantive content, but fix formatting "
        f"and missing fields as necessary. Output only the corrected JSON object.{shape_hint}\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Failed output:\n{failed_output}\n"
    )

    repair_kwargs = {}
    if temperature is not None:
        repair_kwargs["temperature"] = min(temperature, 0.3)
    else:
        repair_kwargs["temperature"] = 0.0
    if max_tokens is not None:
        repair_kwargs["max_tokens"] = max_tokens

    try:
        result = model.infer(
            prompt=repair_user_prompt,
            session_id=f"{session_id}-repair-{uuid.uuid4().hex[:8]}",
            system_prompt=repair_system_prompt,
            enable_thinking=False,
            response_format={"type": "json_object", "schema": json_schema},
            **repair_kwargs,
        )
    except Exception as exc:
        logger.warning("JSON repair attempt failed: %s", exc)
        return None

    repair_raw = result.get("raw_text", result.get("raw", ""))
    repair_cleaned = result.get("text", repair_raw).strip()
    repair_tokens = int(result.get("tokens", 0))

    parsed = _try_parse_json(repair_cleaned)
    if parsed is None:
        parsed = _try_parse_json(repair_raw)
    if parsed is None:
        logger.warning(
            "JSON repair attempt returned non-parseable output. Raw (first 500 chars): %s",
            repair_raw[:500] if repair_raw else repair_cleaned[:500]
        )
        return None

    ok, err = _validate_schema(parsed, json_schema)
    if not ok:
        logger.warning("JSON repair attempt failed schema validation: %s", err)
        return None

    return parsed, repair_raw, repair_cleaned, repair_tokens
