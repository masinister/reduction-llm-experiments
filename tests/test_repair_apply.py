from src.repair import apply_edits, collect_touched_indices


def test_apply_edits_full_cycle():
    steps = ["step a", "step b", "step c"]
    edits = [
        {"op": "replace", "index": 1, "content": "updated step b"},
        {"op": "insert", "index": 3, "content": "step d"},
        {"op": "move", "index": 0, "to_index": 2},
        {"op": "delete", "index": 0},
    ]

    new_steps, applied, skipped = apply_edits(steps, edits)

    assert new_steps == ["step c", "updated step b", "step d"]
    assert any(item["op"] == "move" for item in applied)
    assert not skipped

    touched = collect_touched_indices(applied)
    assert touched == [0, 1, 2]


def test_apply_edits_invalid_entries_skipped():
    steps = ["only"]
    edits = [
        {"op": "insert", "index": -1, "content": "bad"},
        {"op": "replace", "index": 0, "content": "new"},
        {"op": "delete", "index": 2},
    ]
    new_steps, applied, skipped = apply_edits(steps, edits)
    assert new_steps == ["new"]
    assert len(applied) == 1
    assert len(skipped) == 2
