import json

from src.parse import deterministic_split, parse_with_model


class FakeModel:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def infer(self, **_: object) -> dict:
        return {"text": self.response_text, "tokens": 12}


def test_parse_with_model_success():
    payload = json.dumps({"steps": ["step one", "step two"]})
    model = FakeModel(payload)
    steps = parse_with_model(
        model,
        session_id="test-parse",
        candidate_blob="raw",
        system_prompt="system",
    )
    assert steps == ["step one", "step two"]


def test_parse_with_model_fallback():
    model = FakeModel("not json")
    blob = "1. first\n2. second"
    steps = parse_with_model(
        model,
        session_id="test-parse-fallback",
        candidate_blob=blob,
        system_prompt="system",
    )
    assert steps == ["first", "second"]


def test_deterministic_split_handles_blank_lines():
    blob = """1. first step\n\n- second step continues\n   continues line\n\nThird step free form"""
    steps = deterministic_split(blob)
    assert steps == [
        "first step",
        "second step continues continues line",
        "Third step free form",
    ]
