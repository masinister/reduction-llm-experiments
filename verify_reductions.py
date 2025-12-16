#!/usr/bin/env python3
"""
Interactive script to verify reductions from a JSONL file.
Displays each reduction in a human-readable format and allows marking as verified/unverified.
"""

import json
import os
import argparse
from pathlib import Path

from pydantic import BaseModel, Field

from src import config
from src.core_backend import Backend

# Configuration
JSONL_FILE = Path(__file__).parent / "data" / "processed" / "karp_reductions.jsonl"
PROGRESS_FILE = Path(__file__).parent / "data" / "processed" / "verification_progress.json"
REFINED_JSONL_FILE = Path(__file__).parent / "data" / "processed" / "karp_reductions_refined.jsonl"

def _get_reduction_dict(record: dict) -> dict:
    """Return the reduction object for a record.
    """
    nested = record.get("reduction")
    if not isinstance(nested, dict):
        raise ValueError("Record missing required object field 'reduction'")
    return nested


def _set_reduction_dict(record: dict, reduction_dict: dict) -> None:
    """Set the canonical nested reduction."""
    record["reduction"] = reduction_dict


def _default_input_path() -> Path:
    """Prefer refined output if present, otherwise fall back to original."""
    return REFINED_JSONL_FILE if REFINED_JSONL_FILE.exists() else JSONL_FILE


def is_failed_record(record: dict) -> bool:
    """Heuristic for records that represent failures from refinement runs."""
    return "error" in record


# ============================================================================
# Pydantic Models (match extraction schema)
# ============================================================================


class ProblemDefinition(BaseModel):
    """Definition of a computational problem."""

    name: str = Field(description="Name of the problem")
    input_format: str = Field(description="Description of what constitutes a valid input instance")
    yes_condition: str = Field(description="The condition that makes an instance a YES instance")


class Reduction(BaseModel):
    """Structured representation of a computational reduction proof."""

    source_problem: str = Field(description="The source problem being reduced from (e.g., '3-SAT')")
    target_problem: str = Field(description="The target problem being reduced to (e.g., 'CLIQUE')")
    source_definition: ProblemDefinition = Field(description="Formal definition of the source problem")
    target_definition: ProblemDefinition = Field(description="Formal definition of the target problem")
    reduction_steps: list[str] = Field(
        description="Step-by-step construction procedure transforming a source instance to a target instance"
    )
    forward_proof: str = Field(description="Proof that YES instance of source implies YES instance of target")
    backward_proof: str = Field(description="Proof that YES instance of target implies YES instance of source")
    key_insight: str = Field(description="The key idea or intuition that makes this reduction work")


class Critique(BaseModel):
    """Critique of a structured reduction representation."""

    summary: str = Field(description="One-paragraph summary of quality and key issues")
    major_issues: list[str] = Field(description="Critical correctness/rigor issues that must be fixed")
    minor_issues: list[str] = Field(description="Smaller clarity/formatting issues")
    suggested_edits: list[str] = Field(description="Actionable edits to improve correctness and clarity")
    confidence: float = Field(description="Confidence in critique from 0.0 to 1.0")


# ============================================================================
# Critique + refine prompts
# ============================================================================


_PLAIN_TEXT_RULES = """*** CRITICAL: ALL OUTPUT MUST BE PLAIN TEXT - NO LATEX ***
Convert all math notation to readable plain text:
- \\land, \\wedge -> "AND"
- \\lor, \\vee -> "OR"
- \\neg, \\lnot -> "NOT"
- \\in -> "in"
- \\subseteq -> "subset of"
- \\forall -> "for all"
- \\exists -> "there exists"
- \\implies, \\Rightarrow -> "implies" or "=>"
- \\iff, \\Leftrightarrow -> "if and only if"
- $x_i$ -> "x_i"
- \\textsc{Name} -> "NAME"
- \\Big, \\big, \\left, \\right -> remove entirely
- Remove all \\begin{...}, \\end{...}, \\item, etc.
- Avoid backslashes in the final text.
"""


def make_critique_prompt(record: dict) -> str:
    record_json = json.dumps(record, ensure_ascii=False, indent=2)
    return f"""You are an expert in NP-completeness and polynomial-time reductions.

Your task: critique the following structured reduction representation for rigor, correctness, and clarity.

Focus on:
1) Definitions: do input_format and yes_condition match standard definitions? Any ambiguity?
2) Construction: are reduction_steps complete, deterministic, and polynomial-time? Are parameters (like k) defined?
3) Correctness: are forward_proof and backward_proof logically sound and consistent with steps?
4) Consistency: do the problems/definitions/steps/proofs align (same objects, symbols, direction)?
5) Completeness: is any crucial argument missing (e.g., iff, polynomial bounds, gadget correctness)?
6) Style: plain text only; avoid LaTeX; steps should be atomic and unambiguous.

Do NOT rewrite the whole reduction here. Only critique and propose concrete fixes.

Return a JSON object matching this schema:
- summary: string
- major_issues: list of strings
- minor_issues: list of strings
- suggested_edits: list of strings
- confidence: number in [0, 1]

{_PLAIN_TEXT_RULES}

=== STRUCTURED REDUCTION (INPUT) ===
{record_json}
"""


def make_refine_prompt(record: dict, critique: Critique) -> str:
    record_json = json.dumps(record, ensure_ascii=False, indent=2)
    critique_json = critique.model_dump_json(indent=2)

    source_problem = record.get("source_problem", "")
    target_problem = record.get("target_problem", "")

    return f"""You are an expert editor of NP-completeness reductions.

You will improve the structured reduction representation using the critique.

Hard constraints:
- Keep source_problem EXACTLY as: "{source_problem}"
- Keep target_problem EXACTLY as: "{target_problem}"
- Keep problem definition names consistent with those.
- Output MUST be valid JSON for the specified schema.
- Output MUST be plain text (no LaTeX, no backslashes).
- Do not invent a completely different reduction. Only fix rigor/clarity/consistency issues.

Quality bar:
- reduction_steps: atomic, imperative, and complete enough to implement.
- forward_proof/backward_proof: logically valid, references the construction, and addresses any parameters.
- Mention polynomial-time / size bounds if they are relevant and missing.

Return ONLY the JSON for a Reduction object with fields:
- source_problem
- target_problem
- source_definition: {{name, input_format, yes_condition}}
- target_definition: {{name, input_format, yes_condition}}
- reduction_steps: list of strings
- forward_proof
- backward_proof
- key_insight

{_PLAIN_TEXT_RULES}

=== ORIGINAL STRUCTURED REDUCTION ===
{record_json}

=== CRITIQUE (USE THIS) ===
{critique_json}
"""


def _format_critique(critique: dict) -> str:
    lines: list[str] = []
    lines.append("-" * 40)
    lines.append("  LLM CRITIQUE")
    lines.append("-" * 40)
    lines.append(f"  Summary: {critique.get('summary', 'N/A')}")
    lines.append("")

    lines.append(f"  Confidence: {critique.get('confidence', 'N/A')}")
    lines.append("")

    major = critique.get("major_issues") or []
    minor = critique.get("minor_issues") or []
    edits = critique.get("suggested_edits") or []

    lines.append("  Major issues:")
    if major:
        for i, item in enumerate(major, 1):
            lines.append(f"    {i}. {item}")
    else:
        lines.append("    (none)")
    lines.append("")

    lines.append("  Minor issues:")
    if minor:
        for i, item in enumerate(minor, 1):
            lines.append(f"    {i}. {item}")
    else:
        lines.append("    (none)")
    lines.append("")

    lines.append("  Suggested edits:")
    if edits:
        for i, item in enumerate(edits, 1):
            lines.append(f"    {i}. {item}")
    else:
        lines.append("    (none)")
    lines.append("")

    return "\n".join(lines)


def load_reductions(filepath: Path) -> list[dict]:
    """Load all reductions from the JSONL file."""
    reductions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reductions.append(json.loads(line))
    return reductions


def load_progress(filepath: Path) -> dict:
    """Load verification progress from JSON file."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            progress = json.load(f)
    else:
        progress = {"verified": [], "rejected": [], "notes": {}}

    # Backward compatible defaults
    progress.setdefault("verified", [])
    progress.setdefault("rejected", [])
    progress.setdefault("notes", {})
    progress.setdefault("critiques", {})
    return progress


def save_progress(filepath: Path, progress: dict):
    """Save verification progress to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def save_reductions_jsonl(filepath: Path, reductions: list[dict]) -> None:
    """Save reductions as JSONL."""
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in reductions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_reduction(reduction: dict, index: int, total: int) -> str:
    """Format a single reduction for display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  REDUCTION {index + 1} of {total}")
    lines.append(f"  Entry Key: {reduction.get('entry_key', 'N/A')}")
    lines.append("=" * 80)
    lines.append("")

    # Failure records (e.g., produced by refine_reductions.py)
    if is_failed_record(reduction):
        lines.append("-" * 40)
        lines.append("  REFINEMENT FAILURE")
        lines.append("-" * 40)
        lines.append(f"  Error: {wrap_text(str(reduction.get('error', 'N/A')), width=76, indent='  ')}")
        lines.append("")
        return "\n".join(lines)
    
    red = _get_reduction_dict(reduction)

    # Source and target
    source = red.get("source_problem", "N/A")
    target = red.get("target_problem", "N/A")
    difficulty = reduction.get("difficulty", "N/A")
    lines.append(f"  {source}  →  {target}")
    lines.append(f"  Difficulty: {difficulty}")
    lines.append("")
    
    # Source definition
    lines.append("-" * 40)
    lines.append("  SOURCE PROBLEM DEFINITION")
    lines.append("-" * 40)
    source_def = red.get("source_definition", {})
    if isinstance(source_def, dict):
        lines.append(f"  Name: {source_def.get('name', 'N/A')}")
        lines.append(f"  Input: {wrap_text(source_def.get('input_format', 'N/A'), width=72, indent='         ')}")
        lines.append(f"  YES if: {wrap_text(source_def.get('yes_condition', 'N/A'), width=70, indent='          ')}")
    else:
        lines.append(f"  {source_def}")
    lines.append("")
    
    # Target definition
    lines.append("-" * 40)
    lines.append("  TARGET PROBLEM DEFINITION")
    lines.append("-" * 40)
    target_def = red.get("target_definition", {})
    if isinstance(target_def, dict):
        lines.append(f"  Name: {target_def.get('name', 'N/A')}")
        lines.append(f"  Input: {wrap_text(target_def.get('input_format', 'N/A'), width=72, indent='         ')}")
        lines.append(f"  YES if: {wrap_text(target_def.get('yes_condition', 'N/A'), width=70, indent='          ')}")
    else:
        lines.append(f"  {target_def}")
    lines.append("")
    
    # Reduction steps
    lines.append("-" * 40)
    lines.append("  REDUCTION STEPS")
    lines.append("-" * 40)
    steps = red.get("reduction_steps", [])
    for i, step in enumerate(steps, 1):
        # Wrap long lines
        wrapped = wrap_text(step, width=72, indent="      ")
        lines.append(f"  {i}. {wrapped}")
    lines.append("")
    
    # Key insight
    lines.append("-" * 40)
    lines.append("  KEY INSIGHT")
    lines.append("-" * 40)
    insight = red.get("key_insight", "N/A")
    lines.append(f"  {wrap_text(insight, width=76, indent='  ')}")
    lines.append("")
    
    # Forward proof
    lines.append("-" * 40)
    lines.append("  FORWARD PROOF (YES → YES)")
    lines.append("-" * 40)
    forward = red.get("forward_proof", "N/A")
    lines.append(f"  {wrap_text(forward, width=76, indent='  ')}")
    lines.append("")
    
    # Backward proof
    lines.append("-" * 40)
    lines.append("  BACKWARD PROOF (YES ← YES)")
    lines.append("-" * 40)
    backward = red.get("backward_proof", "N/A")
    lines.append(f"  {wrap_text(backward, width=76, indent='  ')}")
    lines.append("")
    
    return "\n".join(lines)


def wrap_text(text: str, width: int = 76, indent: str = "") -> str:
    """Simple text wrapper that preserves line breaks."""
    if not text:
        return ""
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return ("\n" + indent).join(lines)


def get_status_display(entry_key: str, progress: dict) -> str:
    """Get status display string for an entry."""
    if entry_key in progress["verified"]:
        return "[✓ VERIFIED]"
    elif entry_key in progress["rejected"]:
        return "[✗ REJECTED]"
    else:
        return "[  PENDING ]"


def show_summary(reductions: list[dict], progress: dict):
    """Show a summary of all reductions and their verification status."""
    print("\n" + "=" * 80)
    print("  VERIFICATION SUMMARY")
    print("=" * 80)
    
    verified = len(progress["verified"])
    rejected = len(progress["rejected"])
    pending = len(reductions) - verified - rejected
    failed = sum(1 for r in reductions if is_failed_record(r))
    
    print(f"\n  Total reductions: {len(reductions)}")
    print(f"  Verified: {verified}")
    print(f"  Rejected: {rejected}")
    print(f"  Pending:  {pending}")
    print(f"  Failed:   {failed}")
    print("")
    
    print("-" * 80)
    print(f"{'#':<4} {'Status':<14} {'Source':<25} {'Target':<25}")
    print("-" * 80)
    
    for i, r in enumerate(reductions):
        key = r.get("entry_key", "")
        status = "[! FAILED ]" if is_failed_record(r) else get_status_display(key, progress)
        red = _get_reduction_dict(r)
        source = red.get("source_problem", "N/A")[:24]
        target = red.get("target_problem", "N/A")[:24]
        print(f"{i+1:<4} {status:<14} {source:<25} {target:<25}")
    
    print("-" * 80)
    print("")


def main():
    print("\n" + "=" * 80)
    print("  REDUCTION VERIFICATION TOOL")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="Verify and optionally refine reductions.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_default_input_path()),
        help="Input JSONL path (defaults to refined output if it exists)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    
    # Load data
    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return
    
    reductions = load_reductions(input_path)
    progress = load_progress(PROGRESS_FILE)
    
    print(f"\nLoaded {len(reductions)} reductions from {input_path.name}")
    print(f"Progress file: {PROGRESS_FILE.name}")
    
    current_index = 0
    backend: Backend | None = None
    refined_dirty = False
    
    while True:
        print("\n" + "-" * 40)
        print("Commands:")
        print("  [n]ext / [p]rev  - Navigate reductions")
        print("  [g]oto <num>     - Go to specific reduction")
        print("  [v]erify         - Mark current as verified")
        print("  [r]eject         - Mark current as rejected")
        print("  [c]lear          - Clear verification status")
        print("  [o]te <text>     - Add a note to current")
        print("  [k]ritique       - LLM critique current structured reduction")
        print("  [x]refine        - LLM critique + apply edits to current record")
        print("  [w]rite          - Write refined JSONL to default output")
        print("  [s]ummary        - Show summary of all")
        print("  [f]ilter <type>  - Show only pending/verified/rejected/failed")
        print("  [q]uit           - Save and exit")
        print("-" * 40)
        
        cmd = input("\nCommand: ").strip().lower()
        
        if not cmd:
            continue
        
        if cmd == "q" or cmd == "quit":
            save_progress(PROGRESS_FILE, progress)
            if refined_dirty:
                save_reductions_jsonl(REFINED_JSONL_FILE, reductions)
                print(f"\nRefined reductions written to {REFINED_JSONL_FILE.name}")
            print(f"\nProgress saved to {PROGRESS_FILE.name}")
            break
        
        elif cmd == "n" or cmd == "next":
            if current_index < len(reductions) - 1:
                current_index += 1
            else:
                print("Already at last reduction.")
            reduction = reductions[current_index]
            key = reduction.get("entry_key", "")
            print(format_reduction(reduction, current_index, len(reductions)))
            print(f"Status: {get_status_display(key, progress)}")
            if key in progress["notes"]:
                print(f"Note: {progress['notes'][key]}")
            if key in progress.get("critiques", {}):
                print(_format_critique(progress["critiques"][key]))
        
        elif cmd == "p" or cmd == "prev":
            if current_index > 0:
                current_index -= 1
            else:
                print("Already at first reduction.")
            reduction = reductions[current_index]
            key = reduction.get("entry_key", "")
            print(format_reduction(reduction, current_index, len(reductions)))
            print(f"Status: {get_status_display(key, progress)}")
            if key in progress["notes"]:
                print(f"Note: {progress['notes'][key]}")
            if key in progress.get("critiques", {}):
                print(_format_critique(progress["critiques"][key]))
        
        elif cmd.startswith("g") or cmd.startswith("goto"):
            parts = cmd.split()
            if len(parts) >= 2:
                try:
                    num = int(parts[1]) - 1
                    if 0 <= num < len(reductions):
                        current_index = num
                        reduction = reductions[current_index]
                        key = reduction.get("entry_key", "")
                        print(format_reduction(reduction, current_index, len(reductions)))
                        print(f"Status: {get_status_display(key, progress)}")
                        if key in progress["notes"]:
                            print(f"Note: {progress['notes'][key]}")
                        if key in progress.get("critiques", {}):
                            print(_format_critique(progress["critiques"][key]))
                    else:
                        print(f"Invalid number. Enter 1-{len(reductions)}")
                except ValueError:
                    print("Invalid number.")
            else:
                print("Usage: goto <number>")

        elif cmd == "k" or cmd == "critique":
            record = reductions[current_index]
            key = record.get("entry_key", "")

            if backend is None:
                config.load()
                backend = Backend()

            try:
                critique = backend.create(make_critique_prompt(_get_reduction_dict(record)), Critique)
                progress.setdefault("critiques", {})
                progress["critiques"][key] = critique.model_dump()
                save_progress(PROGRESS_FILE, progress)
                print(_format_critique(progress["critiques"][key]))
            except Exception as e:
                print(f"Critique failed: {e}")

        elif cmd == "x" or cmd == "refine":
            record = reductions[current_index]
            key = record.get("entry_key", "")

            if backend is None:
                config.load()
                backend = Backend()

            try:
                reduction_in = _get_reduction_dict(record)
                critique = backend.create(make_critique_prompt(reduction_in), Critique)
                refined = backend.create(make_refine_prompt(reduction_in, critique), Reduction)

                _set_reduction_dict(record, refined.model_dump())
                record["previous_reduction_critique"] = critique.model_dump()
                reductions[current_index] = record
                refined_dirty = True

                progress.setdefault("critiques", {})
                progress["critiques"][key] = critique.model_dump()
                save_progress(PROGRESS_FILE, progress)

                print("\n[Refinement applied to current record]")
                print(format_reduction(record, current_index, len(reductions)))
                print(_format_critique(progress["critiques"][key]))
            except Exception as e:
                print(f"Refinement failed: {e}")

        elif cmd == "w" or cmd == "write":
            save_reductions_jsonl(REFINED_JSONL_FILE, reductions)
            refined_dirty = False
            print(f"Wrote refined reductions to {REFINED_JSONL_FILE}")
        
        elif cmd == "v" or cmd == "verify":
            key = reductions[current_index].get("entry_key", "")
            if key not in progress["verified"]:
                progress["verified"].append(key)
            if key in progress["rejected"]:
                progress["rejected"].remove(key)
            save_progress(PROGRESS_FILE, progress)
            print(f"Marked '{key}' as VERIFIED")
        
        elif cmd == "r" or cmd == "reject":
            key = reductions[current_index].get("entry_key", "")
            if key not in progress["rejected"]:
                progress["rejected"].append(key)
            if key in progress["verified"]:
                progress["verified"].remove(key)
            save_progress(PROGRESS_FILE, progress)
            print(f"Marked '{key}' as REJECTED")
        
        elif cmd == "c" or cmd == "clear":
            key = reductions[current_index].get("entry_key", "")
            if key in progress["verified"]:
                progress["verified"].remove(key)
            if key in progress["rejected"]:
                progress["rejected"].remove(key)
            save_progress(PROGRESS_FILE, progress)
            print(f"Cleared status for '{key}'")
        
        elif cmd.startswith("o") or cmd.startswith("note"):
            parts = cmd.split(maxsplit=1)
            if len(parts) >= 2:
                key = reductions[current_index].get("entry_key", "")
                progress["notes"][key] = parts[1]
                save_progress(PROGRESS_FILE, progress)
                print(f"Added note to '{key}'")
            else:
                print("Usage: note <text>")
        
        elif cmd == "s" or cmd == "summary":
            show_summary(reductions, progress)
        
        elif cmd.startswith("f") or cmd.startswith("filter"):
            parts = cmd.split()
            if len(parts) >= 2:
                filter_type = parts[1]
                print(f"\n--- Showing {filter_type} reductions ---\n")
                for i, r in enumerate(reductions):
                    key = r.get("entry_key", "")
                    red = _get_reduction_dict(r)
                    if filter_type == "pending" and key not in progress["verified"] and key not in progress["rejected"]:
                        print(f"{i+1}. {red.get('source_problem')} → {red.get('target_problem')}")
                    elif filter_type == "verified" and key in progress["verified"]:
                        print(f"{i+1}. {red.get('source_problem')} → {red.get('target_problem')}")
                    elif filter_type == "rejected" and key in progress["rejected"]:
                        print(f"{i+1}. {red.get('source_problem')} → {red.get('target_problem')}")
                    elif filter_type == "failed" and is_failed_record(r):
                        print(f"{i+1}. {r.get('entry_key', 'N/A')} (FAILED)")
            else:
                print("Usage: filter <pending|verified|rejected|failed>")
        
        else:
            # If just pressing enter or unknown command, show current
            reduction = reductions[current_index]
            key = reduction.get("entry_key", "")
            print(format_reduction(reduction, current_index, len(reductions)))
            print(f"Status: {get_status_display(key, progress)}")
            if key in progress["notes"]:
                print(f"Note: {progress['notes'][key]}")
            if key in progress.get("critiques", {}):
                print(_format_critique(progress["critiques"][key]))


if __name__ == "__main__":
    main()
