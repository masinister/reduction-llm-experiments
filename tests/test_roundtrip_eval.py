from unittest.mock import patch

from src.evaluator import evaluate_step
from src.vllm_structured import StructuredCallError


def test_evaluate_step_returns_fallback_on_failure():
    steps = ["candidate step"]
    context = {}
    model = object()

    with patch("src.evaluator.run_structured", side_effect=StructuredCallError("boom")):
        data, raw = evaluate_step(
            model,
            session_base="sess",
            context=context,
            steps=steps,
            index=0,
        )

    assert raw is None
    assert data["step_index"] == 0
    assert data["passes"] is False
    assert data["issues"][0]["title"] == "Model JSON parse failure"
