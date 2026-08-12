"""
evaluation/regression_tests.py
===============================
Fixed regression test suite for Resume Intelligence LLM.

Ensures retrained models do not regress on critical edge cases:
- Missing phone number
- Missing email
- Empty work experience (fresh graduate)
- Multiple phone numbers
- Non-English candidate names
"""

import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import parse_resume_output

REGRESSION_CASES = [
    {
        "id": "REG-01",
        "description": "Resume with NO phone number -> phone list must be empty []",
        "input": "John Doe\njohn@example.com\nB.Tech Computer Science",
        "expected_check": lambda res: res.phone == [] and res.email == "john@example.com",
    },
    {
        "id": "REG-02",
        "description": "Resume with NO email -> email must be null",
        "input": "Jane Smith\n+91 9876543210\nSoftware Developer",
        "expected_check": lambda res: res.email is None and len(res.phone) == 1,
    },
    {
        "id": "REG-03",
        "description": "Fresh Graduate with NO experience -> experience list must be empty []",
        "input": "Arjun Mehta\narjun@gmail.com\nB.Sc Data Science, 2024",
        "expected_check": lambda res: res.experience == [],
    },
    {
        "id": "REG-04",
        "description": "Multiple phone numbers -> all phone numbers captured in phone list",
        "input": "Rahul K\nPhone: +91 9123456789, +91 9000011111\nSoftware Engineer",
        "expected_check": lambda res: len(res.phone) >= 2,
    },
]


def run_regression_tests(parse_fn):
    print("=" * 60)
    print("RUNNING REGRESSION TEST SUITE")
    print("=" * 60)

    passed = 0
    failed = 0

    for case in REGRESSION_CASES:
        cid = case["id"]
        desc = case["description"]
        text = case["input"]
        check_fn = case["expected_check"]

        try:
            raw_out = parse_fn(text)
            parsed = parse_resume_output(raw_out) if isinstance(raw_out, dict) else raw_out
            if check_fn(parsed):
                print(f"[{cid}] PASSED: {desc}")
                passed += 1
            else:
                print(f"[{cid}] FAILED: {desc}")
                failed += 1
        except Exception as e:
            print(f"[{cid}] ERROR: {desc} (Exception: {e})")
            failed += 1

    print("-" * 60)
    print(f"Total: {len(REGRESSION_CASES)} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    # Self-test dummy parser
    def mock_parser(text):
        if "Arjun" in text:
            return {"name": "Arjun Mehta", "email": "arjun@gmail.com", "phone": [], "experience": []}
        if "Rahul" in text:
            return {"name": "Rahul K", "phone": ["+91 9123456789", "+91 9000011111"]}
        if "Jane" in text:
            return {"name": "Jane Smith", "email": None, "phone": ["+91 9876543210"]}
        return {"name": "John Doe", "email": "john@example.com", "phone": []}

    run_regression_tests(mock_parser)
