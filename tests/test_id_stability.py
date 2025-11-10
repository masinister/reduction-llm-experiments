from src.vllm_structured import _stable_id


def test_stable_id_repeatable():
    text = "Same issue text"
    first = _stable_id("issue", text)
    second = _stable_id("issue", text)
    assert first == second


def test_stable_id_changes_with_text():
    base = _stable_id("issue", "text a")
    other = _stable_id("issue", "text b")
    assert base != other
