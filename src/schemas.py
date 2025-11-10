"""
JSON-schema-like Python dicts for vLLM StructuredOutputsParams and local validation.
Paste this file into src/ and import the schema variables where needed.
"""

# Canonical StepIssue schema (reused by other schemas)
STEP_ISSUE = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "category": {
            "type": "string",
            "enum": [
                "clarity",
                "precision",
                "relevance",
                "logic",
                "dependency",
                "notation",
                "ground-truth-leak",
                "soundness",
                "other",
            ],
        },
        # Optional link to a specific step (0-based) or null if global
        "step_index": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        # Optional metadata map (free-form) — allowed but keep minimal
        "meta": {"type": "object"},
    },
    "required": ["id", "title", "description", "severity", "category"],
    "additionalProperties": False,
}

# 1) Parse schema: converts raw blob -> ordered list of step strings
PARSE_SCHEMA = {
    "title": "ParsedSteps",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        # Optional short notes by parser (e.g., "ambiguous splitting at lines 5-6")
        "notes": {"type": "string"},
    },
    "required": ["steps"],
    "additionalProperties": False,
}

# 2) StepEvaluation schema
STEP_EVAL_SCHEMA = {
    "title": "StepEvaluation",
    "type": "object",
    "properties": {
        "step_index": {"type": "integer"},
        "step_text": {"type": "string"},
        "classification": {
            "type": "string",
            "enum": ["definition", "claim", "justification", "other"],
        },
        "passes": {"type": "boolean"},
        # model confidence 0..1
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasons": {"type": "array", "items": {"type": "string"}},
        # issues is an array of STEP_ISSUE objects
        "issues": {"type": "array", "items": STEP_ISSUE},
    },
    "required": ["step_index", "step_text", "classification", "passes", "reasons", "issues"],
    "additionalProperties": False,
}

# 3) GlobalEvaluation schema
GLOBAL_EVAL_SCHEMA = {
    "title": "GlobalEvaluation",
    "type": "object",
    "properties": {
        "relies_on_ground_truth": {"type": "boolean"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": STEP_ISSUE},
    },
    "required": ["relies_on_ground_truth", "reasons", "issues"],
    "additionalProperties": False,
}

# 4) GroundTruthComparison schema
GT_COMPARE_SCHEMA = {
    "title": "GroundTruthComparison",
    "type": "object",
    "properties": {
        "consistent": {"type": "boolean"},
        "differences": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "issues": {
            "type": "array",
            "items": STEP_ISSUE,
            "maxItems": 3,
        },
    },
    "required": ["consistent", "differences", "issues"],
    "additionalProperties": False,
}

# 5) RepairResponse schema
REPAIR_SCHEMA = {
    "title": "RepairResponse",
    "type": "object",
    "properties": {
        "todo": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "linked_issue_ids": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["id", "description", "linked_issue_ids", "priority"],
                "additionalProperties": False,
            },
        },
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["insert", "replace", "delete", "move"]},
                    "index": {"type": "integer"},
                    # content required for insert/replace; optional/ignored for delete/move
                    "content": {"type": "string"},
                    # to_index used only for move (oneOf integer or null)
                    "to_index": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                    # optionally link edits to issues they aim to resolve
                    "linked_issue_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["op", "index"],
                "additionalProperties": False,
            },
        },
        "resolved_issue_ids": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["todo", "edits", "resolved_issue_ids"],
    "additionalProperties": False,
}

# 6) Final summary schema (pipeline stop / report)
FINAL_SUMMARY_SCHEMA = {
    "title": "FinalSummary",
    "type": "object",
    "properties": {
        # 0..1 fraction of per-step passes (or use 0..100 if you prefer percentages)
        "per_step_pass_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "remaining_issues_count": {"type": "integer", "minimum": 0},
        "remaining_high_medium": {"type": "integer", "minimum": 0},
        "recommend_continue": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["per_step_pass_rate", "remaining_issues_count", "remaining_high_medium", "recommend_continue"],
    "additionalProperties": False,
}

# Exported list of schemas for convenience (optional)
ALL_SCHEMAS = {
    "parse": PARSE_SCHEMA,
    "step_eval": STEP_EVAL_SCHEMA,
    "global_eval": GLOBAL_EVAL_SCHEMA,
    "gt_compare": GT_COMPARE_SCHEMA,
    "repair": REPAIR_SCHEMA,
    "final_summary": FINAL_SUMMARY_SCHEMA,
}
