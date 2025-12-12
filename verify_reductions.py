#!/usr/bin/env python3
"""
Interactive script to verify reductions from a JSONL file.
Displays each reduction in a human-readable format and allows marking as verified/unverified.
"""

import json
import os
from pathlib import Path

# Configuration
JSONL_FILE = Path(__file__).parent / "data" / "processed" / "karp_reductions_oss.jsonl"
PROGRESS_FILE = Path(__file__).parent / "data" / "processed" / "verification_progress.json"


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
            return json.load(f)
    return {"verified": [], "rejected": [], "notes": {}}


def save_progress(filepath: Path, progress: dict):
    """Save verification progress to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def format_reduction(reduction: dict, index: int, total: int) -> str:
    """Format a single reduction for display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  REDUCTION {index + 1} of {total}")
    lines.append(f"  Entry Key: {reduction.get('entry_key', 'N/A')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Source and target
    source = reduction.get("source_problem", "N/A")
    target = reduction.get("target_problem", "N/A")
    difficulty = reduction.get("difficulty", "N/A")
    lines.append(f"  {source}  →  {target}")
    lines.append(f"  Difficulty: {difficulty}")
    lines.append("")
    
    # Source definition
    lines.append("-" * 40)
    lines.append("  SOURCE PROBLEM DEFINITION")
    lines.append("-" * 40)
    source_def = reduction.get("source_definition", {})
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
    target_def = reduction.get("target_definition", {})
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
    steps = reduction.get("reduction_steps", [])
    for i, step in enumerate(steps, 1):
        # Wrap long lines
        wrapped = wrap_text(step, width=72, indent="      ")
        lines.append(f"  {i}. {wrapped}")
    lines.append("")
    
    # Key insight
    lines.append("-" * 40)
    lines.append("  KEY INSIGHT")
    lines.append("-" * 40)
    insight = reduction.get("key_insight", "N/A")
    lines.append(f"  {wrap_text(insight, width=76, indent='  ')}")
    lines.append("")
    
    # Forward proof
    lines.append("-" * 40)
    lines.append("  FORWARD PROOF (YES → YES)")
    lines.append("-" * 40)
    forward = reduction.get("forward_proof", "N/A")
    lines.append(f"  {wrap_text(forward, width=76, indent='  ')}")
    lines.append("")
    
    # Backward proof
    lines.append("-" * 40)
    lines.append("  BACKWARD PROOF (YES ← YES)")
    lines.append("-" * 40)
    backward = reduction.get("backward_proof", "N/A")
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
    
    print(f"\n  Total reductions: {len(reductions)}")
    print(f"  Verified: {verified}")
    print(f"  Rejected: {rejected}")
    print(f"  Pending:  {pending}")
    print("")
    
    print("-" * 80)
    print(f"{'#':<4} {'Status':<14} {'Source':<25} {'Target':<25}")
    print("-" * 80)
    
    for i, r in enumerate(reductions):
        key = r.get("entry_key", "")
        status = get_status_display(key, progress)
        source = r.get("source_problem", "N/A")[:24]
        target = r.get("target_problem", "N/A")[:24]
        print(f"{i+1:<4} {status:<14} {source:<25} {target:<25}")
    
    print("-" * 80)
    print("")


def main():
    print("\n" + "=" * 80)
    print("  REDUCTION VERIFICATION TOOL")
    print("=" * 80)
    
    # Load data
    if not JSONL_FILE.exists():
        print(f"Error: Could not find {JSONL_FILE}")
        return
    
    reductions = load_reductions(JSONL_FILE)
    progress = load_progress(PROGRESS_FILE)
    
    print(f"\nLoaded {len(reductions)} reductions from {JSONL_FILE.name}")
    print(f"Progress file: {PROGRESS_FILE.name}")
    
    current_index = 0
    
    while True:
        print("\n" + "-" * 40)
        print("Commands:")
        print("  [n]ext / [p]rev  - Navigate reductions")
        print("  [g]oto <num>     - Go to specific reduction")
        print("  [v]erify         - Mark current as verified")
        print("  [r]eject         - Mark current as rejected")
        print("  [c]lear          - Clear verification status")
        print("  [o]te <text>     - Add a note to current")
        print("  [s]ummary        - Show summary of all")
        print("  [f]ilter <type>  - Show only pending/verified/rejected")
        print("  [q]uit           - Save and exit")
        print("-" * 40)
        
        cmd = input("\nCommand: ").strip().lower()
        
        if not cmd:
            continue
        
        if cmd == "q" or cmd == "quit":
            save_progress(PROGRESS_FILE, progress)
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
                    else:
                        print(f"Invalid number. Enter 1-{len(reductions)}")
                except ValueError:
                    print("Invalid number.")
            else:
                print("Usage: goto <number>")
        
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
                    if filter_type == "pending" and key not in progress["verified"] and key not in progress["rejected"]:
                        print(f"{i+1}. {r.get('source_problem')} → {r.get('target_problem')}")
                    elif filter_type == "verified" and key in progress["verified"]:
                        print(f"{i+1}. {r.get('source_problem')} → {r.get('target_problem')}")
                    elif filter_type == "rejected" and key in progress["rejected"]:
                        print(f"{i+1}. {r.get('source_problem')} → {r.get('target_problem')}")
            else:
                print("Usage: filter <pending|verified|rejected>")
        
        else:
            # If just pressing enter or unknown command, show current
            reduction = reductions[current_index]
            key = reduction.get("entry_key", "")
            print(format_reduction(reduction, current_index, len(reductions)))
            print(f"Status: {get_status_display(key, progress)}")
            if key in progress["notes"]:
                print(f"Note: {progress['notes'][key]}")


if __name__ == "__main__":
    main()
