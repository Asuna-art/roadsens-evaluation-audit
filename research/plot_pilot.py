"""Render descriptive fold-level RoadSens screening plot from analyzer output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ["full_numeric", "missing_only", "context_only", "imu_only", "physics_window", "physics_calibrated"]
LABELS = ["Full numeric", "Missing only", "Context only", "IMU only", "Physics window", "Prefix calibrated"]
POLICIES = [("primary", "Primary labels"), ("blank_normal", "Blank = Normal sensitivity")]
SPLITS = [("random", "Random rows", "#4477AA"), ("group", "Recording-heldout", "#CC6677")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "pilot_v2")
    ap.add_argument("--output", type=Path, default=ROOT / "figures" / "pilot_macro_f1.png")
    args = ap.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("validated_method_fold_count") != 120:
        raise SystemExit("Refusing to plot a partial or unvalidated pilot.")
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)
    x = np.arange(len(METHODS), dtype=float)
    width = 0.34
    for ax, (policy, title) in zip(axes, POLICIES):
        for si, (split, split_label, color) in enumerate(SPLITS):
            offset = (-0.5 if si == 0 else 0.5) * width
            means = [summary["metrics"][policy][split][m]["mean"]["macro_f1"] for m in METHODS]
            ax.bar(x + offset, means, width=width, color=color, alpha=0.72,
                   edgecolor="black", linewidth=0.45, label=f"{split_label} mean")
            for mi, method in enumerate(METHODS):
                fold_values = [f["macro_f1"] for f in summary["metrics"][policy][split][method]["folds"]]
                jitter = np.linspace(-0.085, 0.085, len(fold_values))
                ax.scatter(np.full(len(fold_values), x[mi] + offset) + jitter, fold_values,
                           color=color, edgecolor="white", linewidth=0.45, s=30, zorder=4,
                           label="Fold scores (descriptive)" if mi == 0 and si == 0 else None)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xticks(x, LABELS, rotation=15, ha="right")
    fig.suptitle("RoadSens fixed-seed pilot: row-random vs recording-heldout", fontsize=15, weight="bold")
    fig.text(0.5, 0.012,
             "Bars show five-fold means; circles show individual fixed-fold scores. Folds are descriptive; random and held-out test rows differ.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
