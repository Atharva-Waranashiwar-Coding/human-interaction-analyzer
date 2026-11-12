"""Generate candidate higher-level workflows from completed action evidence.

This is image-region reasoning, not task-intent classification. Any inferred
activity needs review; moving cameras and ambiguous regions are rejected by
the upstream region_at() evidence check.
"""
import uuid

from .sequences import recognize_activities, VERSION as SEQUENCE_VERSION


def _candidate(label, events, regions, description):
    action_ids = list(dict.fromkeys(
        action_id for event in events for action_id in event["details"]["action_ids"]
    ))
    objects = {obj["id"]: obj for event in events for obj in event["objects"]}
    return {
        "id": uuid.uuid5(uuid.NAMESPACE_URL, "complex-v1:" + label + ":" + ":".join(action_ids)).hex,
        "start_time": min(event["start_time"] for event in events),
        "end_time": max(event["end_time"] for event in events),
        "behavior": label,
        "annotation_kind": "activity",
        "details": {
            "activity_type": label,
            "action_ids": action_ids,
            "region_ids": sorted(regions),
            "sequence_status": "complete",
            "missing_steps": [],
        },
        "participants": list(events[0]["participants"]),
        "objects": list(objects.values()),
        "source": "local_sequence",
        "description": description + " Image-space and task intent are unconfirmed.",
        "confidence": None,
        "review_status": "needs_review",
        "rule_version": SEQUENCE_VERSION,
    }


def complex_candidates(actions, frames=(), regions=(), activity_rows=None):
    """Recognize distinct-object region patterns, only from complete relocations.

    Source-to-outside regions can indicate unpacking-like movements; movement
    to distinct labeled regions can suggest sorting-like distribution.
    This cannot establish packing, unloading, stacking, or actual containment.
    """
    activities = (activity_rows if activity_rows is not None else
                  recognize_activities(actions, frames, regions))
    relocations = [a for a in activities if a["behavior"] == "object_relocation"
                   and a["details"]["sequence_status"] == "complete"
                   and len(a["objects"]) == 1]
    kind = {r["id"]: r.get("kind") for r in regions}
    result = []
    for person in {tuple(e["participants"]) for e in relocations}:
        group = sorted((e for e in relocations if tuple(e["participants"]) == person),
                       key=lambda e: e["start_time"])
        # Consider pairs with distinct objects and no overlapping action windows.
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if second["start_time"] < first["end_time"]:
                    continue
                if second["end_time"] - first["start_time"] > 60:
                    break
                if first["objects"][0]["id"] == second["objects"][0]["id"]:
                    continue
                a, b = first["details"], second["details"]
                source_a, source_b = a.get("source_region"), b.get("source_region")
                dest_a, dest_b = a.get("destination_region"), b.get("destination_region")
                if (source_a and source_a == source_b and kind.get(source_a) == "container"
                        and dest_a and dest_b and source_a not in {dest_a, dest_b}):
                    result.append(_candidate(
                        "possible repeated container-region removal", [first, second],
                        {source_a, dest_a, dest_b},
                        "Two distinct objects have complete relocation evidence from the same container image region to other configured regions."
                    ))
                if (dest_a and dest_b and dest_a != dest_b and source_a and source_a == source_b
                        and dest_a != source_a and dest_b != source_a):
                    result.append(_candidate(
                        "possible multi-region object distribution", [first, second],
                        {source_a, dest_a, dest_b},
                        "Two distinct objects moved from the same configured source region to different configured destination regions."
                    ))
    return result
