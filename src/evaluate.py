"""Metrics: pick the detection threshold on val, then report on test.

The threshold turns scores into benign/attack decisions. Choosing it on val (never on
test) keeps the test numbers honest.

Attack is the positive class. Because attacks are ~96% of CICIoMT2024, attack-focused
metrics (precision, recall, PR-AUC) look high almost automatically, so we also report
fpr (share of benign traffic wrongly flagged) and balanced accuracy.
"""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def pick_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Score threshold with the best F1 on this data."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(val_scores, val_y, test_scores, test_y, test_attack) -> dict:
    threshold = pick_threshold(val_scores, val_y)
    pred = test_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(test_y, pred, labels=[0, 1]).ravel()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(test_y, test_scores)),
        "pr_auc": float(average_precision_score(test_y, test_scores)),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "balanced_accuracy": float((recall + (1 - fpr)) / 2),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # Share of each class flagged as attack: should be ~0 for Benign and ~1 for every attack.
        "flagged_rate_per_class": {
            a: float(pred[test_attack == a].mean()) for a in sorted(np.unique(test_attack))
        },
    }
