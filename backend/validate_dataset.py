"""Validate a private, human-reviewed video benchmark manifest.

The manifest contains metadata and relative references only. It does not copy or
commit source media, predictions, or labels. Ground truth must be independently
reviewed; this command cannot create it.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SCENARIOS = {"static", "moving", "occlusion", "crowded", "low_light", "partial_body"}
REQUIRED = {"clip_id", "scenario", "duration", "video_path", "annotations_path",
            "reviewer", "split", "ground_truth_complete"}
SPLITS = {"train", "validation", "test"}


def validate_manifest(payload, root=None, verify_files=False):
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Manifest needs a nonempty clips array")
    ids = set()
    stats = Counter()
    groups = {}
    for idx, clip in enumerate(clips):
        if not isinstance(clip, dict) or REQUIRED - set(clip):
            raise ValueError(f"Clip {idx} missing required metadata")
        cid = clip["clip_id"]
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError(f"Invalid or duplicate clip_id: {cid!r}")
        ids.add(cid)
        if clip["scenario"] not in SCENARIOS or clip["split"] not in SPLITS:
            raise ValueError(f"Invalid scenario or split for {cid}")
        if type(clip["duration"]) not in (int, float) or not 0 < clip["duration"] <= 3600:
            raise ValueError(f"Invalid duration for {cid}")
        if clip["ground_truth_complete"] is not True or not clip["reviewer"]:
            raise ValueError(f"Independent exhaustive review required for {cid}")
        group = clip.get("source_group", cid)
        if not isinstance(group, str) or not group:
            raise ValueError(f"Invalid source_group for {cid}")
        if group in groups and groups[group] != clip["split"]:
            raise ValueError(f"Source group leaks across splits: {group}")
        groups[group] = clip["split"]
        for field in ("video_path", "annotations_path"):
            value = clip[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing {field} for {cid}")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe {field} for {cid}: must be relative within private root")
            if verify_files:
                if root is None or not (root / path).resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe path for {cid}")
                if not (root / path).is_file():
                    raise ValueError(f"Missing file {field} for {cid}")
        stats[(clip["scenario"], clip["split"])] += 1
    return {"clips": len(clips), "by_scenario_split": [
        {"scenario": scenario, "split": split, "count": count}
        for (scenario, split), count in sorted(stats.items())],
        "source_groups": len(groups),
        "note": "Manifest structure validated, not the correctness or independence of annotations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_manifest(payload, args.manifest.parent, args.verify_files)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
