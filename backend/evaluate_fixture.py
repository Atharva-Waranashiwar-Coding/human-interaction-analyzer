"""Reproducible, offline evaluation of independently reviewed annotation exports.

Usage:
    python backend/evaluate_fixture.py path/to/scopes.json

Input: {"scopes": [{"request": {...EvaluationRequest fields...},
                    "annotations": [...], "review_log": [...]}]}
No video, model inference, network access or database changes are involved.
"""
import argparse
import json
from pathlib import Path

from app.schemas import EvaluationRequest
from app.services.evaluation import evaluate
from app.services.error_analysis import error_report, compare_reports


def summarize(scopes):
    if not isinstance(scopes, list) or not scopes:
        raise ValueError("Expected a nonempty scopes array")
    results = []
    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    for index, scope in enumerate(scopes):
        if not isinstance(scope, dict) or not isinstance(scope.get("annotations"), list):
            raise ValueError(f"Scope {index} must include annotations as a list")
        request = EvaluationRequest.model_validate(scope["request"])
        result = evaluate(scope["annotations"], request, scope.get("review_log", []))
        if not result["available"]:
            raise ValueError(f"Scope {index} has no valid evaluation: {result['reason']}")
        results.append(result)
        for key in totals:
            totals[key] += result["summary"][key]
    tp, fp, fn = (totals[key] for key in ("true_positives", "false_positives", "false_negatives"))
    totals.update(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )
    return {"summary": totals, "scopes": results, "error_analysis": error_report(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="Path to a local JSON evaluation fixture")
    parser.add_argument("--output", type=Path, help="Optional path to save deterministic JSON results")
    parser.add_argument("--baseline", type=Path, help="Optional previous benchmark report for scoped regression comparison")
    args = parser.parse_args()
    payload = json.loads(args.fixture.read_text(encoding="utf-8"))
    report = summarize(payload["scopes"])
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        report["regression_comparison"] = compare_reports(report, baseline)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
