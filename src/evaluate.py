"""Pick the threshold on val, report metrics on test (attack = positive class)."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def pick_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Score threshold with the best F1 on this data."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def standard_metrics(scores, y, threshold: float) -> dict:
    """The metrics other papers report, at a given threshold; used to score the full published test set."""
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "fpr": float(fp / max(fp + tn, 1)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "n_rows": int(len(y)),
    }


def evaluate(val_scores, val_y, test_scores, test_y, test_attack) -> dict:
    threshold = pick_threshold(val_scores, val_y)
    pred = test_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(test_y, pred, labels=[0, 1]).ravel()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    val_pred = val_scores >= threshold
    val_fpr, val_recall = float(val_pred[val_y == 0].mean()), float(val_pred[val_y == 1].mean())
    return {
        "threshold": threshold,
        # What tuning selects on: hyperparameters are chosen by val, never by test.
        "val": {"fpr": val_fpr, "recall": val_recall, "balanced_accuracy": (val_recall + 1 - val_fpr) / 2},
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
