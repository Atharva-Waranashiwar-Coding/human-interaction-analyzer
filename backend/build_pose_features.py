"""Convert locally reviewed 17-keypoint pose sequences to model feature JSONL.

Input JSONL: label, split, group, poses (T x 17 x [x,y,confidence]).
Missing/low-confidence joints become zero-valued features with visibility masks.
Coordinates are relative to each frame's hip midpoint and scaled by robust
torso length. Input must already have fixed temporal sampling and window size.
Never assigns labels or source groups automatically.
"""
import argparse
import json
import math
from pathlib import Path


def convert(row):
    poses = row.get("poses")
    if not isinstance(poses, list) or len(poses) < 3:
        raise ValueError("Need at least three pose frames")
    features = []
    for frame in poses:
        if not isinstance(frame, list) or len(frame) != 17:
            raise ValueError("Each pose frame needs 17 joints")
        if any(not isinstance(joint, list) or len(joint) != 3 or
               any(type(value) not in (float, int) or not math.isfinite(value)
                   for value in joint) for joint in frame):
            raise ValueError("Pose joints must be finite [x,y,confidence]")
        left_hip, right_hip = frame[11], frame[12]
        left_shoulder, right_shoulder = frame[5], frame[6]
        anchors = (left_hip, right_hip, left_shoulder, right_shoulder)
        if any(j[2] < 0.6 for j in anchors):
            # Entire frame is missing: encode all joints as invisible.
            features.append([0.0] * 51)
            continue
        origin = ((left_hip[0] + right_hip[0])/2,
                  (left_hip[1] + right_hip[1])/2)
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0])/2,
                        (left_shoulder[1] + right_shoulder[1])/2)
        scale = max(math.dist(origin, shoulder_mid), 1e-4)
        frame_features = []
        for x, y, confidence in frame:
            visible = float(confidence >= .6)
            frame_features.extend([
                max(-8, min(8, (x-origin[0])/scale)) * visible,
                max(-8, min(8, (y-origin[1])/scale)) * visible,
                visible,
            ])
        features.append(frame_features)
    return {key: row[key] for key in ("label", "split", "group")} | {"features": features}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Never place output inside Git: private labels and derived features remain local.
    lines = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not lines:
        raise ValueError("No pose sequences supplied")
    rows = [convert(row) for row in lines]
    from temporal_experiment import load_rows
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"validation.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        load_rows(path)
    args.output.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
