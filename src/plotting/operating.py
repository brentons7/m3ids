"""Matched false-alarm comparison: is a model's lower miss rate real, or just where its threshold landed?

    python3 -m src.plotting.operating --tags final
    python3 -m src.plotting.operating --tags final ablation_siso --ablation --name operating_ablation

Uses the scores test.py already saved (val_scores.npy, test_scores.npy); no retraining.
Writes to results/figures/<name>/:
  det_curve           miss rate vs false-alarm rate on the held-out test split (log-log), mean over seeds,
                      band = min..max over seeds, dots = each seed's actual operating point (val-F1 threshold)
  operating_runs.csv  one row per run and target false-alarm rate
  operating_table.csv mean and std over seeds per model and target
and prints a LaTeX table body.

Two ways to fix the false-alarm rate (FAR), both reported:
  matched   threshold set on the held-out test benign scores so each model has exactly the target FAR there.
            Threshold-free comparison of ranking quality (like reading a ROC curve); not a deployable setting.
  val-cal.  threshold set on validation benign scores to the target FAR, then applied unchanged to test.
            What a deployment could actually do; the realized test FAR will drift from the target.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import argparse

from . import ABLATION, COLORS, FIGURES, GRID, INK, INK_2, MAIN, ROOT, lineup, load_runs, save, tint

TARGETS = [0.0025, 0.005, 0.01]                      # false-alarm rates compared in the table
FOCUS = ["ARP_Spoofing", "Recon-VulScan"]            # per-attack misses reported at each target
GRID_FAR = np.logspace(-4, np.log10(0.05), 200)      # 0.01% .. 5% for the curve


def labels(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Labels and attack names in the split's original row order (the order the scores were saved in)."""
    df = pd.read_parquet(ROOT / "data" / "processed" / dataset / f"{split}.parquet", columns=["label", "attack"])
    return df["label"].to_numpy().astype(np.int64), df["attack"].astype(str).to_numpy()


def threshold_for_far(benign_scores: np.ndarray, far: float) -> float:
    """Smallest threshold whose false-alarm rate (share of benign scores >= threshold) is at most `far`."""
    s = np.sort(benign_scores)
    k = int(np.ceil(len(s) * (1 - far)))             # benign scores at index >= k would be flagged
    if k >= len(s):
        return np.nextafter(s[-1], np.inf)
    t = s[k]
    # ties: flagging >= t would flag every copy of t, so step just above it if that exceeds the target
    if np.mean(s >= t) > far:
        t = np.nextafter(t, np.inf)
    return float(t)


def rates(scores, y, attack, t: float) -> dict:
    flagged = scores >= t
    out = {"far": float(flagged[y == 0].mean()), "miss": float((~flagged[y == 1]).mean())}
    for a in FOCUS:
        m = attack == a
        if m.any():
            out[f"miss_{a}"] = float((~flagged[m]).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", required=True, help="include runs whose folder ends in _<tag>")
    parser.add_argument("--name", help="output folder under results/figures/")
    parser.add_argument("--dataset", default="ciciomt2024")
    parser.add_argument("--ablation", action="store_true", help="Mamba-3 SISO vs MIMO instead of the main lineup")
    args = parser.parse_args()
    args.out = FIGURES / (args.name or "operating_" + "+".join(args.tags))
    names = ABLATION if args.ablation else MAIN
    df, models = lineup(load_runs(args.tags, args.dataset), names)

    yv, _ = labels(args.dataset, "val")
    yt, at = labels(args.dataset, "test")

    rows, curves, points = [], {m: [] for m in models}, {m: [] for m in models}
    for _, run in df.iterrows():
        run_dir = ROOT / "results" / run["run_id"]
        sv, st = np.load(run_dir / "val_scores.npy"), np.load(run_dir / "test_scores.npy")
        assert len(sv) == len(yv) and len(st) == len(yt), f"{run_dir.name}: scores don't match the processed data"
        for far in TARGETS:
            matched = rates(st, yt, at, threshold_for_far(st[yt == 0], far))
            valcal = rates(st, yt, at, threshold_for_far(sv[yv == 0], far))
            rows.append({"model": run["model"], "seed": run["seed"], "target_far": far,
                         **{f"matched_{k}": v for k, v in matched.items()},
                         **{f"valcal_{k}": v for k, v in valcal.items()}})
        # curve: miss rate at each FAR on the grid (test-calibrated), and the run's real operating point
        benign = np.sort(st[yt == 0])
        attack_scores = np.sort(st[yt == 1])
        thr = np.array([threshold_for_far(benign, f) for f in GRID_FAR])
        curves[run["model"]].append(np.searchsorted(attack_scores, thr, side="left") / len(attack_scores))
        points[run["model"]].append((run["fpr"], 1 - run["recall"]))

    runs = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "operating_runs.csv", index=False)
    value_cols = [c for c in runs.columns if c.startswith(("matched_", "valcal_"))]
    table = runs.groupby(["model", "target_far"])[value_cols].agg(["mean", "std"])
    table.to_csv(args.out / "operating_table.csv")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in models:
        c = np.array(curves[m]) * 100
        ax.fill_between(GRID_FAR * 100, c.min(0), c.max(0), color=tint(COLORS[m], 0.6), alpha=0.6, linewidth=0)
        ax.plot(GRID_FAR * 100, c.mean(0), color=COLORS[m], linewidth=1.8, label=names[m])
        p = np.array(points[m]) * 100
        ax.scatter(p[:, 0], p[:, 1], color=COLORS[m], edgecolor=INK, linewidths=0.6, s=22, zorder=5)
    for far in TARGETS:
        ax.axvline(far * 100, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.6, zorder=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3])
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.set_xlabel("false alarms (% of benign records)")
    ax.set_ylabel("missed attacks (% of attack records)")
    ax.grid(color=GRID, linewidth=0.6, which="major")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Held-out test split: lower-left is better; dots = val-F1 operating points", loc="left",
                 fontsize=8.5, fontweight="normal", color=INK_2)
    fig.tight_layout()
    save(fig, args.out, "det_curve")

    # ---- LaTeX table body ----
    def cell(m, far, col, scale=100, digits=3):
        mean, std = table.loc[(m, far), (col, "mean")] * scale, table.loc[(m, far), (col, "std")] * scale
        return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}" if not np.isnan(std) else f"{mean:.{digits}f}"

    print("\n% Matched false-alarm rate (threshold set on test benign): Model & FAR & Missed (%) & "
          + " & ".join(f"{a} missed (%)" for a in FOCUS) + " & Val-cal. FAR (%) & Val-cal. missed (%) \\\\")
    for far in TARGETS:
        for i, m in enumerate(models):
            lead = f"\\multirow{{{len(models)}}}{{*}}{{{far * 100:.2f}\\%}}" if i == 0 else ""
            cols = [cell(m, far, "matched_miss")] + [cell(m, far, f"matched_miss_{a}", digits=1) for a in FOCUS]
            cols += [cell(m, far, "valcal_far", digits=2), cell(m, far, "valcal_miss")]
            print(f"{lead} & {names[m]} & " + " & ".join(cols) + " \\\\")
        print("\\midrule")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
