"""Render the frozen main-A summary figures and README tables without model fits."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
RUN = PAPER / "runs" / "main_A_v2"
CLASSES = ("Normal", "Pothole", "Bump")
POLICIES = ("primary", "blank_normal")
SPLITS = ("random42", "record42", "timeblock42")
MODELS = ("RF100", "ExtraTrees100", "HistGB", "LogisticRegression")
FEATURES = ("full_numeric", "context_only", "imu_only", "missing_only")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary_path = HERE / "main_A_component_bootstrap.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row_hashes = {p: sha(PAPER / "runs" / "main_A_preflight_v2" / "row_indices" / f"{p}.npz")
                  for p in POLICIES}
    summary["input_sha256"]["main_A_preflight_row_indices"] = row_hashes
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    analysis = json.loads((RUN / "analysis" / "main_A_analysis.json").read_text(encoding="utf-8"))
    # Full feature model/split contrast. Fold-mean values are points; no intervals here.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    colors = ["#3B6FB6", "#E07A35", "#3A9D73", "#A05AA1"]
    x = np.arange(3); width = .18
    for ax, p in zip(axes, POLICIES):
        for mi, model in enumerate(MODELS):
            vals = [next(j["fold_mean_macro_f1"] for j in analysis["jobs"]
                         if j["policy"] == p and j["split"] == s and
                         j["feature"] == "full_numeric" and j["model"] == model)
                    for s in SPLITS]
            ax.bar(x + (mi - 1.5) * width, vals, width, label=model, color=colors[mi])
        ax.set_title("Primary labels" if p == "primary" else "Blank labels treated as Normal")
        ax.set_xticks(x, ["Row-random", "Recording-heldout", "Time-block"])
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.25); ax.set_ylabel("Fold-mean macro-F1")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Full numeric features: split sensitivity across four models\nBars show fold-mean estimates; no uncertainty intervals")
    fig.tight_layout()
    fig.savefig(HERE / "main_A_full_numeric_foldmean.png", dpi=220, bbox_inches="tight")
    fig.savefig(HERE / "main_A_full_numeric_foldmean.svg", bbox_inches="tight")
    plt.close(fig)

    # Recording-heldout feature-family findings, preserving both label interpretations.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    feat_names = ("Full numeric", "Context", "IMU", "Missingness")
    for ax, p in zip(axes, POLICIES):
        for feat, label in zip(FEATURES, feat_names):
            vals = [next(j["fold_mean_macro_f1"] for j in analysis["jobs"]
                         if j["policy"] == p and j["split"] == "record42" and
                         j["feature"] == feat and j["model"] == model) for model in MODELS]
            ax.plot(np.arange(4), vals, marker="o", linewidth=1.8, label=label)
        ax.set_title("Primary labels" if p == "primary" else "Blank labels treated as Normal")
        ax.set_xticks(np.arange(4), ["RF100", "ExtraTrees", "HistGB", "Logistic"])
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.25); ax.set_ylabel("Fold-mean macro-F1")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Recording-heldout results by measured feature family\nFold-mean points only; feature associations are not causal effects")
    fig.tight_layout()
    fig.savefig(HERE / "main_A_recording_features_foldmean.png", dpi=220, bbox_inches="tight")
    fig.savefig(HERE / "main_A_recording_features_foldmean.svg", bbox_inches="tight")
    plt.close(fig)

    # Pooled OOF confusion matrices, normalized by true-class support.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    for pi, p in enumerate(POLICIES):
        for mi, model in enumerate(MODELS):
            cm = np.asarray(summary["point_metrics"][p][model]["record42/full_numeric"]["confusion"], dtype=float)
            cm /= cm.sum(axis=1, keepdims=True)
            ax = axes[pi, mi]
            im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            color="white" if cm[i,j] > .6 else "black", fontsize=9)
            ax.set_xticks(range(3), CLASSES, rotation=28, ha="right")
            ax.set_yticks(range(3), CLASSES)
            if pi == 0: ax.set_title(model, fontsize=10)
            if mi == 0: ax.set_ylabel(("Primary labels" if pi == 0 else "Blank=Normal") + "\nTrue class")
            else: ax.set_yticklabels([])
            if pi == 1: ax.set_xlabel("Predicted class")
            else: ax.set_xticklabels([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.84, pad=.02, label="Within-class row proportion")
    fig.suptitle("Full numeric recording-heldout OOF confusion\nDiagonal cells are class recalls; row-normalized pooled predictions")
    fig.savefig(HERE / "main_A_recording_full_confusion.png", dpi=220, bbox_inches="tight")
    fig.savefig(HERE / "main_A_recording_full_confusion.svg", bbox_inches="tight")
    plt.close(fig)

    # Compact report table including estimates and all 24 fixed-component intervals.
    lines = ["# Main A component bootstrap and figures", "",
        "Main A's compact OOF predictions were mapped back to the frozen source rows and labels, then resampled by the v4 content components. This is a fixed-model conditional bootstrap: it does not refit models or quantify training/CV uncertainty.", "",
        "The 2,000 shared draws (seed 20261016) resample five Normal-only and 91 mixed components within strata. The table reports pooled OOF macro-F1 contrasts only; no fold-level intervals or p-values are reported. Degenerate draws are counted without redrawing.", "",
        "| Policy | Model | Pooled contrast | Estimate | 95% percentile interval | Valid draws |", "|---|---|---|---:|---:|---:|"]
    labels = {"primary": "Primary", "blank_normal": "Blank=Normal"}
    contrast_labels = {"full_random_minus_record": "Full: random − recording",
        "full_random_minus_timeblock": "Full: random − time-block",
        "context_minus_full_record": "Context − full: recording"}
    for p in POLICIES:
        for model in MODELS:
            for contrast in contrast_labels:
                r = summary["contrasts"][p][model][contrast]
                lo, hi = r["percentile_ci95"]
                lines.append(f"| {labels[p]} | {model} | {contrast_labels[contrast]} | {r['pooled_oof_macro_f1_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['valid_replicates']}/{r['replicates']} |")
    lines += ["", "Fold-mean findings are shown separately in the first two figures as bar-height point estimates without uncertainty intervals. Across recording-heldout features, IMU is weaker than full numeric under both policies and missingness is policy-sensitive (higher under primary labels, lower under blank-as-Normal). Context-versus-full is model-dependent: for primary-label pooled OOF, the difference is positive for RF100 (+0.1853, 95% CI [+0.0812, +0.3154]), negative for LogisticRegression (−0.1297, [−0.2353, −0.0596]), and crosses zero for ExtraTrees100 and HistGB. Under blank-as-Normal, all four context-versus-full intervals cross zero. These feature results are predictive associations, not causal effects.", "",
        "The third figure gives pooled full-numeric recording-heldout confusion matrices normalized within each true class; diagonal entries are class recall. CSV recording IDs are operational source units, not verified trips, routes, vehicles, or roads. Component resampling does not remove potential training/test content overlap in historical splits and does not support causal or population-significance claims.", "",
        "Machine-readable results: `main_A_component_bootstrap.json`; compact confusion counts and draws: `component_confusions_and_draws.npz`; inputs and output hashes: `provenance.json`."]
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov_path = HERE / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["input_sha256"]["main_A_preflight_row_indices"] = row_hashes
    prov["outputs"]["main_A_component_bootstrap.json"] = sha(summary_path)
    prov["figure_script_sha256"] = sha(Path(__file__))
    prov["figures_sha256"] = {name: sha(HERE / name) for name in (
        "main_A_full_numeric_foldmean.png", "main_A_full_numeric_foldmean.svg",
        "main_A_recording_features_foldmean.png", "main_A_recording_features_foldmean.svg",
        "main_A_recording_full_confusion.png", "main_A_recording_full_confusion.svg", "README.md")}
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "figures_refreshed", "figure_hashes": prov["figures_sha256"]}, indent=2))

if __name__ == "__main__":
    render()
