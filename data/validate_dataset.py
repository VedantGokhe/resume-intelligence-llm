"""
data/validate_dataset.py
========================
Validates resume_dataset.json against the ResumeSchema before training.

Run AFTER pasting GPT-generated examples into resume_dataset.json:
    python data/validate_dataset.py

Checks:
  1. Valid JSON array
  2. Required keys: instruction, input, output
  3. Each output validates against ResumeSchema
  4. No empty instruction or input
  5. Output fields never contain data not found in input (basic hallucination check)
  6. Prints a summary report

Usage:
    python data/validate_dataset.py
    python data/validate_dataset.py --path data/resume_dataset.json --strict
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import ResumeSchema, parse_resume_output
from pydantic import ValidationError


# ── Colors for terminal output ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def check_hallucination(output: ResumeSchema, input_text: str) -> list[str]:
    """
    Basic heuristic: checks if scalar string fields from the output
    appear (case-insensitive) somewhere in the input text.
    Returns a list of suspicious fields.
    """
    suspicious = []
    text_lower = input_text.lower()

    def check_value(value: str, field_name: str) -> None:
        # Only flag if the value is longer than 3 chars and not in the input
        if value and len(value) > 3:
            if value.lower() not in text_lower:
                suspicious.append(f"{field_name}='{value}'")

    if output.name:
        # Check at least the last name appears in text
        parts = output.name.split()
        if parts and not any(p.lower() in text_lower for p in parts):
            suspicious.append(f"name='{output.name}'")

    if output.email and output.email not in text_lower:
        suspicious.append(f"email='{output.email}'")

    for skill in output.skills:
        if skill.lower() not in text_lower:
            suspicious.append(f"skill='{skill}'")

    return suspicious


def validate_dataset(path: Path, strict: bool = False) -> int:
    """
    Validate all examples. Returns number of errors found.
    """
    print(f"\n{BOLD}Validating: {path}{RESET}\n")

    # 1. Load raw JSON
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{RED}[FATAL] Invalid JSON: {e}{RESET}")
        return 1

    if not isinstance(data, list):
        print(f"{RED}[FATAL] Expected a JSON array, got {type(data).__name__}{RESET}")
        return 1

    print(f"  Total examples loaded: {len(data)}\n")

    errors   = 0
    warnings = 0

    for i, example in enumerate(data, start=1):
        prefix = f"  Example {i:>3}"

        # 2. Required keys
        missing_keys = [k for k in ("instruction", "input", "output") if k not in example]
        if missing_keys:
            print(f"{prefix} {RED}ERROR{RESET} — missing keys: {missing_keys}")
            errors += 1
            continue

        # 3. Non-empty instruction and input
        if not example["instruction"].strip():
            print(f"{prefix} {RED}ERROR{RESET} — 'instruction' is empty")
            errors += 1
            continue

        if not example["input"].strip():
            print(f"{prefix} {RED}ERROR{RESET} — 'input' is empty")
            errors += 1
            continue

        input_word_count = len(example["input"].split())
        if input_word_count < 20:
            print(f"{prefix} {YELLOW}WARN {RESET} — 'input' is very short ({input_word_count} words)")
            warnings += 1

        # 4. Schema validation
        try:
            parsed = parse_resume_output(example["output"])
        except ValidationError as e:
            print(f"{prefix} {RED}ERROR{RESET} — schema validation failed:")
            for err in e.errors():
                print(f"              {err['loc']} → {err['msg']}")
            errors += 1
            continue
        except Exception as e:
            print(f"{prefix} {RED}ERROR{RESET} — unexpected error: {e}")
            errors += 1
            continue

        # 5. Hallucination check
        suspicious = check_hallucination(parsed, example["input"])
        if suspicious:
            level = f"{RED}ERROR{RESET}" if strict else f"{YELLOW}WARN {RESET}"
            print(f"{prefix} {level} — possible hallucination: {suspicious}")
            if strict:
                errors += 1
            else:
                warnings += 1
            continue

        # 6. All good
        print(f"{prefix} {GREEN}OK   {RESET} — "
              f"name={repr(parsed.name)}, "
              f"skills={len(parsed.skills)}, "
              f"exp={len(parsed.experience)}, "
              f"edu={len(parsed.education)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Total  : {len(data)}")
    print(f"  {GREEN}Passed {RESET}: {len(data) - errors - warnings}")
    print(f"  {YELLOW}Warnings{RESET}: {warnings}")
    print(f"  {RED}Errors {RESET}: {errors}")
    print(f"{'='*60}\n")

    if errors == 0 and warnings == 0:
        print(f"{GREEN}{BOLD}All examples valid! Ready to split and train.{RESET}")
        print("  Next: python data/split_dataset.py")
    elif errors == 0:
        print(f"{YELLOW}Warnings found — review before training.{RESET}")
    else:
        print(f"{RED}{BOLD}{errors} error(s) found. Fix before training.{RESET}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate resume_dataset.json")
    parser.add_argument(
        "--path", type=Path,
        default=Path(__file__).parent / "resume_dataset.json",
        help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat hallucination warnings as errors"
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"{RED}File not found: {args.path}{RESET}")
        sys.exit(1)

    error_count = validate_dataset(args.path, strict=args.strict)
    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
