"""
evaluation/evaluate.py
=======================
Fully Automated Evaluation Harness for Resume Intelligence LLM.

Takes model predictions file (or runs test set evaluation), computes 
metrics via metrics.py, prints formatted summary table, and generates 
structured report.json and report.md.

Usage:
    python evaluation/evaluate.py --pred-file examples/after_finetuning/after_finetuning_results.json --tag v1
    python evaluation/evaluate.py --pred-file examples/before_finetuning/before_finetuning_results.json --tag base
"""

import argparse
import json
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import evaluate_batch


def parse_raw_output(raw: str) -> dict:
    """Safely parses raw model output into a Python dict."""
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}

    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    if "<think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def generate_markdown_report(metrics: dict, tag: str) -> str:
    md = f"""# Resume Intelligence LLM — Evaluation Report ({tag.upper()})

## Executive Metric Summary

| Evaluation Metric | Score | Industry Target | Status |
|---|---|---|---|
| **JSON Syntax Validity** | {metrics.get('json_validity_rate', 0)}% | > 95.0% | {'✅ PASSED' if metrics.get('json_validity_rate', 0) >= 95 else '❌ FAILED'} |
| **Pydantic Schema Validity** | {metrics.get('schema_validity_rate', 0)}% | > 90.0% | {'✅ PASSED' if metrics.get('schema_validity_rate', 0) >= 90 else '❌ FAILED'} |
| **Name Exact Match Accuracy** | {metrics.get('name_exact_match_acc', 0)}% | > 85.0% | {'✅ PASSED' if metrics.get('name_exact_match_acc', 0) >= 85 else '❌ FAILED'} |
| **Email Exact Match Accuracy** | {metrics.get('email_exact_match_acc', 0)}% | > 85.0% | {'✅ PASSED' if metrics.get('email_exact_match_acc', 0) >= 85 else '❌ FAILED'} |
| **Skills Precision** | {metrics.get('skills_precision', 0)}% | > 70.0% | {'✅ PASSED' if metrics.get('skills_precision', 0) >= 70 else '❌ FAILED'} |
| **Skills Recall** | {metrics.get('skills_recall', 0)}% | > 70.0% | {'✅ PASSED' if metrics.get('skills_recall', 0) >= 70 else '❌ FAILED'} |
| **Skills F1-Score** | {metrics.get('skills_f1', 0)}% | > 75.0% | {'✅ PASSED' if metrics.get('skills_f1', 0) >= 75 else '❌ FAILED'} |
| **Avg Hallucination Rate** | {metrics.get('avg_hallucination_rate', 0)}% | < 5.0% | {'✅ PASSED' if metrics.get('avg_hallucination_rate', 0) <= 5.0 else '❌ FAILED'} |

**Total Test Samples Evaluated**: {metrics.get('total_samples', 0)}
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Automated Resume LLM Evaluator.")
    parser.add_argument("--pred-file", type=str, default="examples/after_finetuning/after_finetuning_results.json", help="Path to predictions JSON")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory to save report")
    parser.add_argument("--tag", type=str, default="v1", help="Report tag identifier (e.g. base, v1)")
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    if not pred_path.exists():
        print(f"Error: Predictions file {pred_path} does not exist.")
        sys.exit(1)

    print(f"Loading predictions from {pred_path}...")
    with open(pred_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    predictions = []
    targets = []
    inputs = []

    for item in samples:
        raw_key = "finetuned_model_output_raw" if "finetuned_model_output_raw" in item else "base_model_output_raw"
        raw_val = item.get(raw_key, item.get("output", {}))
        parsed = parse_raw_output(raw_val)
        predictions.append(parsed)
        targets.append(item.get("ground_truth", item.get("output", {})))
        inputs.append(item.get("raw_resume_text", item.get("input", "")))

    print(f"Computing automated metrics over {len(samples)} samples...")
    metrics = evaluate_batch(predictions, targets, inputs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = output_dir / f"report_{args.tag}.json"
    report_md_path = output_dir / f"report_{args.tag}.md"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md_content = generate_markdown_report(metrics, args.tag)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 65)
    print(f"AUTOMATED EVALUATION REPORT ({args.tag.upper()})")
    print("=" * 65)
    print(f"  JSON Validity Rate       : {metrics.get('json_validity_rate')}%")
    print(f"  Schema Validity Rate     : {metrics.get('schema_validity_rate')}%")
    print(f"  Name Exact Match Acc     : {metrics.get('name_exact_match_acc')}%")
    print(f"  Email Exact Match Acc    : {metrics.get('email_exact_match_acc')}%")
    print(f"  Skills Precision         : {metrics.get('skills_precision')}%")
    print(f"  Skills Recall            : {metrics.get('skills_recall')}%")
    print(f"  Skills F1-Score          : {metrics.get('skills_f1')}%")
    print(f"  Avg Hallucination Rate   : {metrics.get('avg_hallucination_rate')}%")
    print("=" * 65)
    print(f"Saved JSON Report : {report_json_path}")
    print(f"Saved MD Report   : {report_md_path}\n")


if __name__ == "__main__":
    main()
