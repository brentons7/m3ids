"""Inference cost on the Jetson Orin Nano, from each run's benchmark.json (written by python3 run.py --task benchmark).

    python3 -m src.plotting.hardware --tags final

Lineup: Transformer, Mamba-2, Mamba-3 (= MIMO). Writes results/figures/<name>/hardware.{png,pdf}: latency for one
window, throughput at a large batch, GPU utilization, peak GPU memory, board power and model size. Values are the
mean over the runs (seeds) of each model; cost depends only on the model's shape, not on its trained weights.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import COLORS, INK, INK_2, MAIN, ROOT, SURFACE, lineup, load_runs, parse_args, save, style_rows, tint

BIG = 512   # the large batch size shown next to batch 1


def load_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, batch size) from results/<run>/benchmark.json; runs without one are skipped."""
    rows = []
    for _, r in df.iterrows():
        path = ROOT / "results" / r["run_id"] / "benchmark.json"
        if not path.exists():
            continue
        b = json.loads(path.read_text())
        for bs, v in b["batch"].items():
            p = v.get("power") or {}
            rows.append({"model": r["model"], "batch": int(bs), "n_params": b["n_params"],
                         "idle_w": (b.get("idle_power_mw") or np.nan) / 1000,
                         "p50_ms": v["p50_us"] / 1000, "p99_ms": v["p99_us"] / 1000,
                         "rows_per_s": v["throughput_rows_per_s"],
                         "gpu_util_pct": p.get("mean_gpu_util_pct", np.nan),
                         "gpu_mem_mb": v.get("gpu_mem_peak_mb", np.nan),
                         "power_w": p.get("mean_power_mw", np.nan) / 1000})
    if not rows:
        raise SystemExit("No benchmark.json in these runs: run python3 run.py --task benchmark on the Jetson first")
    return pd.DataFrame(rows)


def hbars(ax, models: list[str], names: dict[str, str], small: pd.Series, big: pd.Series | None, fmt: str,
          title: str, xlabel: str, label_after: pd.Series | None = None) -> None:
    """Horizontal bars per model: `small` (batch 1, lighter) above `big` (large batch, full color) when both given."""
    pairs = [(small, True), (big, False)] if big is not None else [(small, False)]
    h = 0.36 if big is not None else 0.62
    top = np.nanmax(np.concatenate([s.reindex(models).to_numpy(dtype=float) for s, _ in pairs]))
    for i, m in enumerate(models):
        for k, (series, light) in enumerate(pairs):
            v = series.get(m, np.nan)
            y = i + ((k - 0.5) * h if big is not None else 0)
            if pd.isna(v):
                ax.text(0, y, " n/a", va="center", fontsize=8.5, color=INK_2)
                continue
            ax.barh(y, v, height=h, color=tint(COLORS[m]) if light else COLORS[m], edgecolor=SURFACE,
                    linewidth=0.8, zorder=3)
            end = max(v, label_after.get(m, v)) if label_after is not None else v   # clear the p99 tick
            ax.text(end + top * 0.03, y, fmt.format(v), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.32)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    style_rows(ax, models, names)


def plot_hardware(bench: pd.DataFrame, models: list[str], names: dict[str, str], out_dir: Path) -> None:
    mean = bench.groupby(["model", "batch"]).mean(numeric_only=True)
    at = lambda bs, col: mean.xs(bs, level="batch")[col]
    fig, axes = plt.subplots(2, 3, figsize=(13, 1.3 * len(models) + 3.2), sharey=True)
    ax = axes.ravel()

    p99 = at(1, "p99_ms")
    hbars(ax[0], models, names, at(1, "p50_ms"), None, "{:.2f}", "Latency, one window",
          "ms per call (median; tick = p99)", label_after=p99)
    for i, m in enumerate(models):
        ax[0].plot([p99[m]] * 2, [i - 0.31, i + 0.31], color=INK, linewidth=1.2, zorder=4)
    ax[0].set_xlim(0, max(ax[0].get_xlim()[1], np.nanmax(p99.reindex(models)) * 1.3))
    hbars(ax[1], models, names, at(BIG, "rows_per_s") / 1000, None, "{:.0f}k", f"Throughput, batch {BIG}",
          "thousand windows / s")
    hbars(ax[2], models, names, at(1, "gpu_util_pct"), at(BIG, "gpu_util_pct"), "{:.0f}%", "GPU utilization",
          "% busy (tegrastats GR3D)")
    ax[2].set_xlim(0, 118)
    ax[2].set_xticks([0, 25, 50, 75, 100])
    hbars(ax[3], models, names, at(1, "gpu_mem_mb"), at(BIG, "gpu_mem_mb"), "{:.1f}", "GPU memory, peak",
          "MB allocated (weights + activations)")
    hbars(ax[4], models, names, at(1, "power_w"), at(BIG, "power_w"), "{:.2f}", "Board power",
          "W, whole board (dashed = idle)")
    idle = bench["idle_w"].mean()
    if not np.isnan(idle):
        ax[4].axvline(idle, color=INK_2, linestyle=(0, (3, 2)), linewidth=1, zorder=4)
    hbars(ax[5], models, names, mean.groupby(level="model")["n_params"].first() / 1000, None, "{:.1f}k",
          "Model size", "parameters, thousands")

    handles = [matplotlib.patches.Patch(color=tint("#7a7a76"), label="batch 1"),
               matplotlib.patches.Patch(color="#7a7a76", label=f"batch {BIG}")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Inference on Jetson Orin Nano (MAXN_SUPER, clocks locked)", x=0.01, ha="left", fontsize=10,
                 color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5)
    save(fig, out_dir, "hardware")


def main() -> None:
    args = parse_args(__doc__)
    df, models = lineup(load_runs(args.tags, args.dataset), MAIN)
    print(f"Writing to {args.out}")
    plot_hardware(load_benchmarks(df), models, MAIN, args.out)


if __name__ == "__main__":
    main()
