"""Optional local pose-sequence baseline with a small temporal convolution model.

No application integration or automatic model downloads. Train only on
independently labeled features and group-separated splits. Torch is optional.
Input JSONL rows: {"features": [[float, ...], ...], "label": "walking",
                   "split": "train|validation|test", "group": "source-video-id"}
Each row is one fixed-length sequence. No face identity or raw frames required.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("No labeled sequences supplied")
    shape = None
    groups = {}
    for row in rows:
        required = ("features", "label", "split", "group")
        if any(k not in row for k in required):
            raise ValueError("Each row needs features, label, split and group")
        x = row["features"]
        if (not isinstance(x, list) or len(x) < 3 or
                any(not isinstance(frame, list) or not frame for frame in x)):
            raise ValueError("Features must be sequences of at least 3 nonempty frames")
        current_shape = (len(x), len(x[0]))
        if any(len(frame) != current_shape[1] or
               any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in frame) for frame in x):
            raise ValueError("Features must be finite and rectangular")
        if shape is None:
            shape = current_shape
        if shape != current_shape:
            raise ValueError("All sequence shapes must match")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid split")
        if not isinstance(row["label"], str) or not row["label"]:
            raise ValueError("Nonempty class label required")
        group = row["group"]
        if not isinstance(group, str) or not group:
            raise ValueError("Nonempty source group required")
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Source group leaks across splits")
        groups[group] = row["split"]
    splits = Counter(row["split"] for row in rows)
    if any(splits[s] == 0 for s in ("train", "validation", "test")):
        raise ValueError("Train, validation and test splits must all be present")
    labels = sorted({row["label"] for row in rows})
    train_labels = {row["label"] for row in rows if row["split"] == "train"}
    if len(train_labels) < 2 or set(labels) != train_labels:
        raise ValueError("Train set must cover all labels, at least two")
    return rows, shape, labels


def experiment(path, epochs=10, seed=17):
    import torch
    from torch import nn
    rows, (length, features), labels = load_rows(path)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    index = {label: i for i, label in enumerate(labels)}
    def tensors(split):
        data = [row for row in rows if row["split"] == split]
        x = torch.tensor([row["features"] for row in data], dtype=torch.float32)
        y = torch.tensor([index[row["label"]] for row in data], dtype=torch.long)
        return x.transpose(1, 2), y
    train_x, train_y = tensors("train")
    val_x, val_y = tensors("validation")
    test_x, test_y = tensors("test")
    # Training split statistics only: no validation/test leakage.
    mean = train_x.mean(dim=(0, 2), keepdim=True)
    std = train_x.std(dim=(0, 2), keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std
    network = nn.Sequential(
        nn.Conv1d(features, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, len(labels)),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    best = None
    best_loss = float("inf")
    for _ in range(epochs):
        network.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(network(train_x), train_y)
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.no_grad():
            validation_loss = nn.functional.cross_entropy(network(val_x), val_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best = {k: v.detach().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best)
    network.eval()
    with torch.no_grad():
        predicted = network(test_x).argmax(dim=1)
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for truth, guess in zip(test_y.tolist(), predicted.tolist()):
        confusion[labels[truth]][labels[guess]] += 1
    count = len(test_y)
    return {
        "labels": labels,
        "samples": dict(Counter(row["split"] for row in rows)),
        "sequence_length": length,
        "input_features": features,
        "epochs": epochs,
        "seed": seed,
        "validation_cross_entropy": best_loss,
        "test_accuracy": sum(int(a == b) for a, b in zip(test_y.tolist(), predicted.tolist())) / count,
        "test_confusion": confusion,
        "note": "Experimental held-out metric, not production accuracy or calibrated confidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200:
        parser.error("--epochs must be 1..200")
    report = experiment(args.jsonl, epochs=args.epochs, seed=args.seed)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
