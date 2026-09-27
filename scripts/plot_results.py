"""Compare models across seeds for one batch of runs (selected by --tag).

    python scripts/plot_results.py --tag valfix

Writes to results/figures/<tag>/:
    summary_table.csv     mean and std per model, over seeds
    metrics_by_model.png  false alarms / missed attacks / balanced accuracy, one dot per seed
    errors_by_class.png   per traffic type: % of attacks missed, or % of benign falsely flagged
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_ORDER = ["mamba3", "mamba2", "lstm", "transformer"]
MODEL_NAMES = {"mamba3": "Mamba-3", "mamba2": "Mamba-2", "lstm": "LSTM", "transformer": "Transformer"}
COLORS = {"mamba3": "#2a78d6", "mamba2": "#eb6834", "lstm": "#1baf7a", "transformer": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load_runs(tag: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((ROOT / "results").glob(f"*_{tag}")):
        config = json.loads((run_dir / "config.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        rows.append({"model": config["model"], "seed": config["seed"], **metrics})
    if not rows:
        raise SystemExit(f"No runs tagged {tag!r} in results/")
    df = pd.DataFrame(rows)
    df["miss_rate"] = 1 - df["recall"]
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["roc_auc", "balanced_accuracy", "fpr", "recall", "n_params", "train_seconds", "infer_us_per_row"]
    g = df.groupby("model")[cols]
    table = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_seeds"))
    return table.reindex([m for m in MODEL_ORDER if m in table.index])


def plot_metrics(df: pd.DataFrame, out: Path) -> None:
    """Three panels, each its own x-axis; one row per model; small dots = seeds, bar = mean."""
    panels = [
        ("fpr", "False alarms on benign traffic", "% of benign rows flagged (lower is better)", 100),
        ("miss_rate", "Missed attacks", "% of attack rows not flagged (lower is better)", 100),
        ("balanced_accuracy", "Balanced accuracy", "higher is better", 1),
    ]
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, (col, title, xlabel, scale) in zip(axes, panels):
        for i, m in enumerate(models):
            vals = df.loc[df["model"] == m, col].to_numpy() * scale
            ax.scatter(vals, np.full(len(vals), i), s=36, color=COLORS[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.plot([vals.mean()] * 2, [i - 0.28, i + 0.28], color=INK, linewidth=2, zorder=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        lo, hi = ax.get_xlim()
        pad = (hi - lo) * 0.08
        ax.set_xlim(max(lo - pad, 0) if col != "balanced_accuracy" else lo - pad, hi + pad)
    axes[0].set_yticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    axes[0].set_ylim(len(models) - 0.5, -0.5)
    n_seeds = df.groupby("model").size().min()
    fig.suptitle(f"CICIoMT2024 test set: dots = individual seeds ({n_seeds} per model), black bar = mean",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_errors_by_class(df: pd.DataFrame, out: Path) -> None:
    """Heatmap: rows = traffic types, cols = models. Cell = error rate (mean over seeds).

    For attacks the error is 'missed' (1 - flagged); for Benign it's 'falsely flagged'.
    Same direction either way: darker = worse.
    """
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    flagged = pd.DataFrame([{"model": r["model"], **r["flagged_rate_per_class"]} for _, r in df.iterrows()])
    flagged = flagged.groupby("model").mean().T[models]
    err = 1 - flagged
    err.loc["Benign"] = flagged.loc["Benign"]
    err = err.loc[["Benign"] + err.drop(index="Benign").mean(axis=1).sort_values(ascending=False).index.tolist()]
    pct = err * 100

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    vmax = max(pct.to_numpy().max(), 1)
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(pct) + 1.4))
    ax.imshow(pct.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for (r, c), v in np.ndenumerate(pct.to_numpy()):
        dark = v > vmax * 0.5
        ax.text(c, r, f"{v:.1f}" if v >= 0.05 else "0", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if dark else (INK if v >= 0.05 else INK_2))
    ax.set_xticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    ax.xaxis.tick_top()
    labels = ["Benign  (% falsely flagged)"] + [f"{a}  (% missed)" for a in pct.index[1:]]
    ax.set_yticks(range(len(pct)), labels)
    ax.set_xticks(np.arange(-0.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pct)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.axhline(0.5, color=INK_2, linewidth=1)
    ax.tick_params(length=0, which="both")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("Error rate by traffic type (%)", x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(), "Mean over seeds, CICIoMT2024 test set. Darker = worse.",
             ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="compare runs whose folder ends in _<tag>")
    args = parser.parse_args()

    df = load_runs(args.tag)
    out_dir = ROOT / "results" / "figures" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table = summary_table(df)
    table.to_csv(out_dir / "summary_table.csv")
    plot_metrics(df, out_dir / "metrics_by_model.png")
    plot_errors_by_class(df, out_dir / "errors_by_class.png")

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table[["n_seeds", "balanced_accuracy_mean", "balanced_accuracy_std", "fpr_mean", "fpr_std",
                     "recall_mean", "recall_std", "roc_auc_mean", "train_seconds_mean"]].to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
