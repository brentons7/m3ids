"""Detection results for the paper's lineup: Transformer, Mamba-2 and Mamba-3 (= MIMO; SISO runs are left out).

    python3 -m src.plotting.results --tags untuned
    python3 -m src.plotting.results --tags final

Writes to results/figures/<name>/:
  ml_metrics          accuracy, precision, recall, F1, ROC-AUC on the full published test set
                      (comparable to other papers)
  metrics_by_model    false alarms, missed attacks and balanced accuracy on the held-out test split
  errors_by_class     % missed per attack type (and % of benign falsely flagged) on the held-out test split
  summary_table.csv   mean, std and seed count per model for every metric above
Bars are the mean over seeds; small dots are the individual seeds.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import (BLUES, COLORS, DATASET_NAMES, GRID, INK, INK_2, MAIN, SURFACE, lineup, load_runs, parse_args, save,
               style_rows)

ML_METRICS = [("full_accuracy", "Accuracy"), ("full_precision", "Precision"), ("full_recall", "Recall"),
              ("full_f1", "F1"), ("full_roc_auc", "ROC-AUC")]


def seed_dots(ax, values: pd.Series, at: float, horizontal: bool) -> None:
    """The individual seeds as small dots on top of a bar (skipped for a single seed)."""
    if len(values) < 2:
        return
    pos = np.full(len(values), at)
    x, y = (values, pos) if horizontal else (pos, values)
    ax.scatter(x, y, s=10, color=INK, alpha=0.55, linewidths=0, zorder=4)


def plot_ml_metrics(df: pd.DataFrame, models: list[str], names: dict[str, str], out_dir: Path) -> None:
    """Grouped bars: one group per standard metric, one bar per model."""
    cols = [c for c, _ in ML_METRICS]
    mean = df.groupby("model")[cols].mean()
    lo = np.floor((df[cols].to_numpy().min() - 0.0003) * 1000) / 1000   # every bar starts at the same, stated floor

    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j, m in enumerate(models):
        xs = np.arange(len(cols)) + (j - (len(models) - 1) / 2) * width
        seeds = df[df["model"] == m]
        ax.bar(xs, mean.loc[m] - lo, bottom=lo, width=width, color=COLORS[m], edgecolor=SURFACE, linewidth=0.8,
               label=names[m], zorder=3)
        for x, c in zip(xs, cols):
            seed_dots(ax, seeds[c], x, horizontal=False)
            # ROC-AUC gets a 5th decimal, otherwise 0.99996 would print as a perfect 1.0000
            ax.annotate(f"{mean.loc[m, c]:.5f}" if c == "full_roc_auc" else f"{mean.loc[m, c]:.4f}",
                        (x, max(mean.loc[m, c], seeds[c].max())), xytext=(0, 2), textcoords="offset points",
                        rotation=90, ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(cols)), [label for _, label in ML_METRICS])
    ax.set_ylim(lo, 1.0006)
    ax.set_ylabel(f"score (axis starts at {lo:.3f})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)

    n = df.groupby("model").size().max()
    seeds_note = f", mean of {n} seeds (dots = seeds)" if n > 1 else ""
    ax.set_title(f"{DATASET_NAMES[df['dataset'].iloc[0]]}, full published test set{seeds_note}", loc="left",
                 fontsize=10, fontweight="normal", color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(models), frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "ml_metrics")


def bar_panel(ax, df: pd.DataFrame, models: list[str], names: dict[str, str], col: str, scale: float, title: str,
              xlabel: str, fmt: str) -> None:
    """One horizontal bar per model (mean over seeds), seeds as dots, value printed at the end."""
    vals = df[col] * scale
    mean = vals.groupby(df["model"]).mean()
    top = vals.max()
    for i, m in enumerate(models):
        seeds = vals[df["model"] == m]
        ax.barh(i, mean[m], height=0.6, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        seed_dots(ax, seeds, i, horizontal=True)
        ax.text(seeds.max() + top * 0.03, i, fmt.format(mean[m]), va="center", ha="left", color=INK, fontsize=9)
    ax.set_xlim(0, top * 1.3)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    style_rows(ax, models, names)


def dot_panel(ax, df: pd.DataFrame, models: list[str], names: dict[str, str], col: str, title: str, xlabel: str,
              fmt: str) -> None:
    """For scores near 1, where a zero-based bar would hide the differences: big dot = mean, small = seeds."""
    mean = df.groupby("model")[col].mean()
    for i, m in enumerate(models):
        seed_dots(ax, df.loc[df["model"] == m, col], i, horizontal=True)
        ax.scatter(mean[m], i, s=70, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(fmt.format(mean[m]), (mean[m], i), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    lo, hi = df[col].min(), df[col].max()
    span = max(hi - lo, 0.002)
    ax.set_xlim(lo - span * 0.25, hi + span * 0.25)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))   # few, full-length labels: no overlap
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(fmt.replace("{:", "{x:")))
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    style_rows(ax, models, names)


def plot_error_rates(df: pd.DataFrame, models: list[str], names: dict[str, str], out_dir: Path) -> None:
    """False alarms, missed attacks and balanced accuracy on the held-out test split."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(models) + 1.9), sharey=True)
    bar_panel(axes[0], df, models, names, "fpr", 100, "False alarms on benign traffic",
              "% of benign rows flagged (lower is better)", "{:.2f}%")
    bar_panel(axes[1], df, models, names, "miss_rate", 100, "Missed attacks",
              "% of attack rows not flagged (lower is better)", "{:.3f}%")
    dot_panel(axes[2], df, models, names, "balanced_accuracy", "Balanced accuracy", "higher is better", "{:.4f}")
    fig.suptitle(f"{DATASET_NAMES[df['dataset'].iloc[0]]}, held-out test split", x=0.01, ha="left", fontsize=10,
                 color=INK_2)
    fig.tight_layout()
    save(fig, out_dir, "metrics_by_model")


def plot_errors_by_class(df: pd.DataFrame, models: list[str], names: dict[str, str], out_dir: Path) -> None:
    """Heatmap of error rate per traffic type and model (missed for attacks, falsely flagged for Benign)."""
    flagged = pd.DataFrame([{"model": r["model"], **r["flagged_rate_per_class"]} for _, r in df.iterrows()])
    flagged = flagged.groupby("model").mean().T[models]
    err = 1 - flagged
    err.loc["Benign"] = flagged.loc["Benign"]
    err = err.loc[["Benign"] + err.drop(index="Benign").mean(axis=1).sort_values(ascending=False).index.tolist()]
    pct = err * 100

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    vmax = max(pct.to_numpy().max(), 1)
    fig, ax = plt.subplots(figsize=(2.2 + 1.5 * len(models), 0.36 * len(pct) + 1.4))
    ax.imshow(pct.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for (r, c), v in np.ndenumerate(pct.to_numpy()):
        dark = v > vmax * 0.5
        ax.text(c, r, f"{v:.1f}" if v >= 0.05 else "0", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if dark else (INK if v >= 0.05 else INK_2))
    ax.set_xticks(range(len(models)), [names[m] for m in models])
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
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(),
             f"{DATASET_NAMES[df['dataset'].iloc[0]]}, held-out test split. Darker = worse.",
             ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    save(fig, out_dir, "errors_by_class")


def summary_table(df: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    cols = ["fpr", "miss_rate", "balanced_accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1",
            *[c for c, _ in ML_METRICS], "full_fpr", "n_params", "train_seconds"]
    g = df.groupby("model")[cols]
    table = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_seeds"))
    table.reindex(models).to_csv(out_dir / "summary_table.csv")
    print("  summary_table.csv")


def make_all(df: pd.DataFrame, models: list[str], names: dict[str, str], out_dir: Path) -> None:
    """The three detection figures and the summary table for one lineup."""
    print(f"Writing to {out_dir}")
    plot_ml_metrics(df, models, names, out_dir)
    plot_error_rates(df, models, names, out_dir)
    plot_errors_by_class(df, models, names, out_dir)
    summary_table(df, models, out_dir)


def main() -> None:
    args = parse_args(__doc__)
    df, models = lineup(load_runs(args.tags, args.dataset), MAIN)
    make_all(df, models, MAIN, args.out)


if __name__ == "__main__":
    main()
