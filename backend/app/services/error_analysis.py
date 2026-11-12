"""Deterministic local recognition-error report from evaluated reviewed scopes."""
from collections import Counter, defaultdict


def error_report(scopes):
    by_scenario = defaultdict(lambda: Counter())
    by_behavior = defaultdict(lambda: Counter())
    cases = []
    for index, result in enumerate(scopes):
        if not result.get("available"):
            raise ValueError(f"Scope {index} is not evaluated")
        scenario = result.get("scenario", "unspecified")
        for key in ("true_positives", "false_positives", "false_negatives"):
            by_scenario[scenario][key] += result["summary"][key]
        for behavior, values in result["per_behavior"].items():
            for key in ("true_positives", "false_positives", "false_negatives"):
                by_behavior[behavior][key] += values[key]
        for category, field in (("false_positive", "false_positive_examples"),
                                ("missed_ground_truth", "missed_ground_truth_examples")):
            for row in result.get(field, []):
                cases.append(dict(scope_index=index, scenario=scenario, error=category, **row))
    def summarize(counter):
        tp, fp, fn = (counter[k] for k in ("true_positives", "false_positives", "false_negatives"))
        return dict(counter, precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return {
        "by_scenario": {k: summarize(v) for k, v in sorted(by_scenario.items())},
        "by_behavior": {k: summarize(v) for k, v in sorted(by_behavior.items())},
        "error_examples": sorted(cases, key=lambda c: (c["scope_index"], c["start_time"], c["error"])),
        "note": "These are reviewed-scoped results only, not full-video or unreviewed accuracy.",
    }
