"""
inference/infer.py
==================
Command Line Interface (CLI) for Resume Intelligence LLM.

Usage:
    python inference/infer.py --input resume.txt
    python inference/infer.py --text "John Doe\njohn@gmail.com\nPython, SQL"
"""

import argparse
import json
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.model_loader import parse_resume


def main():
    parser = argparse.ArgumentParser(description="Parse a resume text file into structured JSON.")
    parser.add_argument("--input", "-i", type=str, help="Path to raw resume text file")
    parser.add_argument("--text", "-t", type=str, help="Direct resume text string")
    parser.add_argument("--adapter", type=str, default="./adapter_v1", help="Path to LoRA adapter")
    args = parser.parse_args()

    resume_text = ""
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            resume_text = f.read()
    elif args.text:
        resume_text = args.text
    else:
        print("Error: Please provide --input <file> or --text '<string>'", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing resume text ({len(resume_text)} characters)...", file=sys.stderr)
    result = parse_resume(resume_text, adapter_path=args.adapter)

    # Print JSON output to stdout
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
