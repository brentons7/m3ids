"""Test one trained run: reload its model, score val and test, and write the metrics.

    python3 run.py test results/<run>

The alarm threshold is picked on val (best F1) and applied unchanged to test (src/evaluate.py). Writes into the
run folder: metrics.json (evaluation added to the training facts already there) and the raw scores
(val_scores.npy, test_scores.npy). Re-testing a run overwrites its previous evaluation.
"""
import json
from pathlib import Path

import numpy as np

from src import evaluate, models
from src.train import load_splits

TRAINING_FACTS = ("n_params", "train_seconds")   # written by src/train.py, kept when re-testing


def test(run_dir: Path, device: str, processed_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    splits, meta = load_splits(processed_dir, config["dataset"], names=("val", "test"))
    model = models.load(run_dir, len(meta["features"]), device)

    print(f"Testing {run_dir.name}")
    val_scores = model.score(splits["val"])
    test_scores = model.score(splits["test"])
    metrics = evaluate.evaluate(val_scores, splits["val"].y, test_scores, splits["test"].y, splits["test"].attack)
    if meta.get("val_from_test"):
        # val + test together are the full published test set: score it too, for comparison with other papers.
        metrics["full_test"] = evaluate.standard_metrics(
            np.concatenate([val_scores, test_scores]), np.concatenate([splits["val"].y, splits["test"].y]),
            metrics["threshold"])

    path = run_dir / "metrics.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    metrics.update({k: old[k] for k in TRAINING_FACTS if k in old})
    path.write_text(json.dumps(metrics, indent=2))
    np.save(run_dir / "val_scores.npy", val_scores)
    np.save(run_dir / "test_scores.npy", test_scores)

    print(f"\nResults saved to {run_dir}")
    for k in ["roc_auc", "pr_auc", "f1", "precision", "recall", "fpr", "balanced_accuracy"]:
        print(f"  {k:18s} {metrics[k]:.4f}")
    print("  flagged as attack, per class:")
    for a, rate in metrics["flagged_rate_per_class"].items():
        print(f"    {a:26s} {rate:.4f}")
    return metrics
