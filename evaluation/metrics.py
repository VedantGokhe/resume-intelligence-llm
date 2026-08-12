"""
evaluation/metrics.py
=====================
Evaluation metrics for Resume Intelligence LLM.

Calculates:
1. JSON Validity Rate: Percentage of outputs that parse as valid JSON.
2. Schema Validity Rate: Percentage of outputs that conform to ResumeSchema.
3. Field Exact Match / Accuracy: Accuracy for scalar fields (name, email, etc.).
4. List Set Precision / Recall / F1: Metrics for skills, certifications, links, etc.
5. Hallucination Rate: Percentage of extracted values unsupported by source text.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.schema import ResumeSchema, parse_resume_output
from pydantic import ValidationError


def compute_json_validity(raw_outputs: List[str]) -> Tuple[float, List[bool]]:
    """Checks if raw output strings parse as JSON."""
    valid_flags = []
    for raw in raw_outputs:
        try:
            # Strip markdown code blocks if present
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            json.loads(cleaned)
            valid_flags.append(True)
        except Exception:
            valid_flags.append(False)

    rate = (sum(valid_flags) / len(valid_flags)) * 100 if valid_flags else 0.0
    return rate, valid_flags


def compute_schema_validity(raw_outputs: List[str]) -> Tuple[float, List[bool]]:
    """Checks if raw outputs validate against ResumeSchema."""
    valid_flags = []
    for raw in raw_outputs:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            parse_resume_output(data)
            valid_flags.append(True)
        except Exception:
            valid_flags.append(False)

    rate = (sum(valid_flags) / len(valid_flags)) * 100 if valid_flags else 0.0
    return rate, valid_flags


def compute_list_f1(pred_list: List[str], target_list: List[str]) -> Tuple[float, float, float]:
    """Computes Set Precision, Recall, and F1 score for list fields (e.g. skills)."""
    pred_set = set(item.lower().strip() for item in pred_list if item)
    target_set = set(item.lower().strip() for item in target_list if item)

    if not target_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not target_set or not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set.intersection(target_set))
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(target_set) if target_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_hallucination_rate(pred_schema: ResumeSchema, input_text: str) -> float:
    """
    Computes ratio of extracted strings not supported by input text.
    """
    text_lower = input_text.lower()
    total_items = 0
    hallucinated_items = 0

    # Check scalar fields
    if pred_schema.name:
        total_items += 1
        # check last name or first name presence
        parts = pred_schema.name.split()
        if not any(p.lower() in text_lower for p in parts):
            hallucinated_items += 1

    if pred_schema.email:
        total_items += 1
        if pred_schema.email.lower() not in text_lower:
            hallucinated_items += 1

    # Check skills
    for skill in pred_schema.skills:
        total_items += 1
        if skill.lower() not in text_lower:
            hallucinated_items += 1

    if total_items == 0:
        return 0.0

    return (hallucinated_items / total_items) * 100


def evaluate_batch(
    predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]], inputs: List[str]
) -> Dict[str, Any]:
    """
    Evaluates a batch of predictions against targets and inputs.
    """
    total = len(targets)
    if total == 0:
        return {}

    # JSON & Schema validity
    raw_str_preds = [
        json.dumps(p) if isinstance(p, dict) else str(p) for p in predictions
    ]
    json_valid_rate, _ = compute_json_validity(raw_str_preds)
    schema_valid_rate, schema_flags = compute_schema_validity(raw_str_preds)

    name_matches = 0
    email_matches = 0
    skill_precisions, skill_recalls, skill_f1s = [], [], []
    hallucination_rates = []

    for i in range(total):
        target_dict = targets[i]
        pred_dict = predictions[i] if isinstance(predictions[i], dict) else {}
        input_text = inputs[i]

        try:
            target_obj = parse_resume_output(target_dict)
        except Exception:
            target_obj = ResumeSchema()

        try:
            pred_obj = parse_resume_output(pred_dict)
        except Exception:
            pred_obj = ResumeSchema()

        # Scalar matches
        if (target_obj.name or "").lower().strip() == (pred_obj.name or "").lower().strip():
            name_matches += 1

        if (target_obj.email or "").lower().strip() == (pred_obj.email or "").lower().strip():
            email_matches += 1

        # Skills F1
        p, r, f1 = compute_list_f1(pred_obj.skills, target_obj.skills)
        skill_precisions.append(p)
        skill_recalls.append(r)
        skill_f1s.append(f1)

        # Hallucination rate
        h_rate = compute_hallucination_rate(pred_obj, input_text)
        hallucination_rates.append(h_rate)

    avg_skills_f1 = (sum(skill_f1s) / total) * 100
    avg_skills_precision = (sum(skill_precisions) / total) * 100
    avg_skills_recall = (sum(skill_recalls) / total) * 100
    avg_hallucination_rate = sum(hallucination_rates) / total

    return {
        "total_samples": total,
        "json_validity_rate": round(json_valid_rate, 2),
        "schema_validity_rate": round(schema_valid_rate, 2),
        "name_exact_match_acc": round((name_matches / total) * 100, 2),
        "email_exact_match_acc": round((email_matches / total) * 100, 2),
        "skills_precision": round(avg_skills_precision, 2),
        "skills_recall": round(avg_skills_recall, 2),
        "skills_f1": round(avg_skills_f1, 2),
        "avg_hallucination_rate": round(avg_hallucination_rate, 2),
    }
