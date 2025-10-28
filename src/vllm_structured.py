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
    repair_on_failure: bool = True,
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
    schema_text = json.dumps(json_schema, indent=2, ensure_ascii=False)
    repair_user_prompt = (
        "The previous model output failed JSON schema validation. Rewrite it so that it becomes "
        "valid JSON matching the schema below. Keep the substantive content, but fix formatting "
        "and missing fields as necessary. Output only the corrected JSON object.\n\n"
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
            session_id=f"{session_id}-repair",
            system_prompt=repair_system_prompt,
            enable_thinking=False,
            json_schema=json_schema,
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
        logger.warning("JSON repair attempt returned non-parseable output.")
        return None

    ok, err = _validate_schema(parsed, json_schema)
    if not ok:
        logger.warning("JSON repair attempt failed schema validation: %s", err)
        return None

    return parsed, repair_raw, repair_cleaned, repair_tokens
