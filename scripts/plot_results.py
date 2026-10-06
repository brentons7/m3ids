"""Plot runs to results/figures/<name>/<dataset>/.

    python3 scripts/plot_results.py --tags baseline                  # one tag
    python3 scripts/plot_results.py --tags baseline mimo --dataset ciciomt2024 --name mimo

Runs of the same model (e.g. several seeds) are averaged into one value per model; the figures
show that single value. The per-seed spread is kept in summary_table.csv.
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
# "variant" = model name, plus a suffix for architecture options that make it a different model (MIMO).
MODEL_ORDER = ["mamba3", "mamba3_mimo", "mamba2", "transformer"]
MODEL_NAMES = {"mamba3": "Mamba-3", "mamba3_mimo": "Mamba-3 MIMO", "mamba2": "Mamba-2", "transformer": "Transformer"}
# Categorical slots in fixed order (blue, orange, aqua, yellow); checked for colorblind separation.
COLORS = {"mamba3": "#2a78d6", "mamba2": "#eb6834", "transformer": "#1baf7a", "mamba3_mimo": "#eda100"}
DATASET_NAMES = {"ciciomt2024": "CICIoMT2024", "iomtind2026": "IoMT-IND2026", "xiomt": "X-IoMT",
                 "iomttrafficdata": "IoMT-TrafficData"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Runs whose power numbers are known to be wrong, with the reason; their power is left out of figures.
INVALID_POWER = {
    "20261005-205939_ciciomt2024_mamba3_mimo": "TileLang kernels compiled inside the power window",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load_runs(tags: list[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        for run_dir in sorted((ROOT / "results").glob(f"*_{tag}")):
            config = json.loads((run_dir / "config.json").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            variant = config["model"] + ("_mimo" if config["hparams"].get("is_mimo") else "")
            rows.append({"run_id": run_dir.name, "model": variant, "seed": config["seed"],
                         "dataset": config["dataset"], **metrics})
    if not rows:
        raise SystemExit(f"No runs tagged {tags} in results/")
    df = pd.DataFrame(rows)
    df["miss_rate"] = 1 - df["recall"]
    # Plain accuracy is dominated by the majority class; balanced_accuracy is the one to lead with.
    df["accuracy"] = df["confusion"].apply(lambda c: (c["tp"] + c["tn"]) / max(sum(c.values()), 1))
    for key in ["mean_power_mw", "max_power_mw", "energy_mj", "mean_gpu_util_pct", "peak_ram_mb", "mean_gpu_temp_c"]:
        df[key] = df["edge"].apply(lambda e, k=key: ((e or {}).get("power") or {}).get(k))
    df.loc[df["run_id"].isin(INVALID_POWER), ["mean_power_mw", "max_power_mw", "energy_mj", "mean_gpu_util_pct"]] = None
    df["latency_p50_us"] = df["edge"].apply(lambda e: (((e or {}).get("latency") or {}).get("batch1") or {}).get("p50_us"))
    df["latency_p99_us"] = df["edge"].apply(lambda e: (((e or {}).get("latency") or {}).get("batch1") or {}).get("p99_us"))
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "fpr", "balanced_accuracy", "accuracy",
            "n_params", "train_seconds", "infer_us_per_row", "mean_power_mw", "max_power_mw", "energy_mj",
            "mean_gpu_util_pct", "peak_ram_mb", "mean_gpu_temp_c", "latency_p50_us", "latency_p99_us"]
    g = df.groupby("model")[cols]
    table = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_seeds"))
    return table.reindex([m for m in MODEL_ORDER if m in table.index])


def style_rows(ax, models: list[str]) -> None:
    ax.set_yticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def bar_panel(ax, means: pd.Series, models: list[str], title: str, xlabel: str, fmt: str, missing: str = "n/a") -> None:
    """One horizontal bar per model with its value printed at the end; NaN shows `missing` instead."""
    top = np.nanmax(means.to_numpy(dtype=float)) if means.notna().any() else 1
    for i, m in enumerate(models):
        v = means.get(m, np.nan)
        if pd.isna(v):
            ax.text(0, i, f" {missing}", va="center", ha="left", color=INK_2, fontsize=8.5)
            continue
        ax.barh(i, v, height=0.6, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(v + top * 0.02, i, fmt.format(v), va="center", ha="left", color=INK, fontsize=9)
    ax.set_xlim(0, top * 1.25)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    style_rows(ax, models)


def plot_metrics(df: pd.DataFrame, out: Path) -> None:
    """False alarms, missed attacks and balanced accuracy: one value per model."""
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    means = df.groupby("model")[["fpr", "miss_rate", "balanced_accuracy"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(models) + 1.9), sharey=True)
    bar_panel(axes[0], means["fpr"] * 100, models, "False alarms on benign traffic",
              "% of benign rows flagged (lower is better)", "{:.2f}%")
    bar_panel(axes[1], means["miss_rate"] * 100, models, "Missed attacks",
              "% of attack rows not flagged (lower is better)", "{:.3f}%")

    # Balanced accuracy sits near 1 for everyone, so a zero-based bar would hide the differences: dot + label.
    ax = axes[2]
    ba = means["balanced_accuracy"]
    for i, m in enumerate(models):
        ax.scatter(ba[m], i, s=70, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{ba[m]:.4f}", (ba[m], i), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    span = max(ba.max() - ba.min(), 0.005)
    ax.set_xlim(ba.min() - span * 0.3, min(ba.max() + span * 0.3, 1.0005))
    ax.set_title("Balanced accuracy", loc="left")
    ax.set_xlabel("higher is better")
    style_rows(ax, models)

    name = DATASET_NAMES.get(df["dataset"].iloc[0], df["dataset"].iloc[0])
    fig.suptitle(f"{name} test set", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_errors_by_class(df: pd.DataFrame, out: Path) -> None:
    """Heatmap of error rate per traffic type and model (missed for attacks, falsely flagged for Benign)."""
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
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
    fig.text(0.02, 1 - 0.55 / fig.get_figheight(),
             f"{DATASET_NAMES.get(df['dataset'].iloc[0], df['dataset'].iloc[0])} test set. Darker = worse.",
             ha="left", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_cost(df: pd.DataFrame, out: Path) -> None:
    """What each model costs: size, training time, single-row latency and power draw on the device."""
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    means = df.groupby("model")[["n_params", "train_seconds", "latency_p50_us", "mean_power_mw"]].mean()
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.55 * len(models) + 1.9), sharey=True)
    bar_panel(axes[0], means["n_params"] / 1000, models, "Model size", "parameters, thousands", "{:.0f}k")
    bar_panel(axes[1], means["train_seconds"] / 60, models, "Training time", "minutes (3 epochs)", "{:.0f} min")
    bar_panel(axes[2], means["latency_p50_us"] / 1000, models, "Inference latency",
              "ms per call, one row (median)", "{:.1f} ms")
    bar_panel(axes[3], means["mean_power_mw"] / 1000, models, "Power while scoring",
              "watts, whole board (VDD_IN)", "{:.2f} W", missing="n/a (invalid)")

    name = DATASET_NAMES.get(df["dataset"].iloc[0], df["dataset"].iloc[0])
    fig.suptitle(f"{name}, Jetson Orin Nano", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True, help="include runs whose folder ends in _<tag>")
    parser.add_argument("--dataset", help="only this dataset (default: every dataset found)")
    parser.add_argument("--name", help="output folder under results/figures/ (default: the tags joined by '+')")
    args = parser.parse_args()

    runs = load_runs(args.tags)
    if args.dataset:
        runs = runs[runs["dataset"] == args.dataset]
        if runs.empty:
            raise SystemExit(f"No {args.dataset} runs tagged {args.tags}")
    root = ROOT / "results" / "figures" / (args.name or "+".join(args.tags))

    tables = []
    for dataset, df in runs.groupby("dataset"):  # every figure compares models on one dataset
        df = df.reset_index(drop=True)
        out_dir = root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        table = summary_table(df)
        table.to_csv(out_dir / "summary_table.csv")
        tables.append(table.assign(dataset=dataset).set_index("dataset", append=True).swaplevel())
        plot_metrics(df, out_dir / "metrics_by_model.png")
        plot_errors_by_class(df, out_dir / "errors_by_class.png")
        plot_cost(df, out_dir / "cost_by_model.png")

        print(f"\n=== {DATASET_NAMES.get(dataset, dataset)} ===")
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
            print(table[["n_seeds", "roc_auc_mean", "f1_mean", "balanced_accuracy_mean", "fpr_mean", "recall_mean",
                         "n_params_mean", "train_seconds_mean", "mean_power_mw_mean", "latency_p50_us_mean"]].to_string())
        print(f"Saved to {out_dir}")

    pd.concat(tables).to_csv(root / "summary_all_datasets.csv")
    print(f"\nAll datasets in one table: {root / 'summary_all_datasets.csv'}")


if __name__ == "__main__":
    main()
