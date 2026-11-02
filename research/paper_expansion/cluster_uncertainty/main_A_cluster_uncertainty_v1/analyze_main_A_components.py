"""Fixed-model OOF component bootstrap and scoped figures for completed main A."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
ROOT = PAPER.parent
RUN = PAPER / "runs" / "main_A_v2"
PREFLIGHT = PAPER / "runs" / "main_A_preflight_v2"
AUDIT = PAPER / "data_audit"
FREEZE = AUDIT / "content_component_freeze_v4.json"
MEMBERSHIP = AUDIT / "content_component_membership_v4.npz"
CLASSES = ("Normal", "Pothole", "Bump")
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
POLICIES = ("primary", "blank_normal")
SPLITS = ("random42", "record42", "timeblock42")
FEATURES = ("full_numeric", "context_only", "imu_only", "missing_only")
MODELS = ("RF100", "ExtraTrees100", "HistGB", "LogisticRegression")
BOOTSTRAPS = 2000
SEED = 20261016


def sha(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def macro_f1(cm):
    support = cm.sum(axis=1)
    if np.any(support == 0):
        return np.nan
    tp = np.diag(cm).astype(float)
    den = 2 * tp + cm.sum(axis=0) - tp + support - tp
    f = np.divide(2 * tp, den, out=np.zeros(3, dtype=float), where=den != 0)
    return float(f.mean())


def metrics(cm):
    support = cm.sum(axis=1)
    recalls = [float(cm[i, i] / support[i]) if support[i] else None for i in range(3)]
    return {"macro_f1": macro_f1(cm), "balanced_accuracy": float(np.mean(recalls)) if all(v is not None for v in recalls) else None,
            "class_support": {c: int(support[i]) for i, c in enumerate(CLASSES)},
            "class_recall": dict(zip(CLASSES, recalls)), "confusion": cm.astype(int).tolist()}


def confusion(y, pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)), 1)
    return cm


def build_component_rows():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("artifact_version") != 4:
        raise RuntimeError("expected frozen content component membership v4")
    with np.load(MEMBERSHIP, allow_pickle=False) as d:
        cache_ids = d["cache_row_id"].astype(np.int64)
        groups = d["session_key"].astype(str)
        source_rows = d["source_row"].astype(np.int64)
        comp = d["component_id"].astype(str)
        label = {p: d[f"label_{p}"].astype(str) for p in POLICIES}
        score = {p: d[f"scoreable_global_row_{p}"].astype(np.int64) for p in POLICIES}
    if not np.array_equal(cache_ids, np.arange(len(cache_ids))):
        raise RuntimeError("component cache IDs are not contiguous")
    component_ids = [o["component_id"] for o in freeze["components"]]
    component_strata = [o["stratum"] for o in freeze["components"]]
    ci = {c: i for i, c in enumerate(component_ids)}
    normal = np.flatnonzero(np.asarray(component_strata) == "normal_only")
    mixed = np.flatnonzero(np.asarray(component_strata) == "mixed")
    if len(component_ids) != 96 or len(normal) != 5 or len(mixed) != 91:
        raise RuntimeError("content component strata do not match frozen 5+91 design")
    row_data = {}
    for p in POLICIES:
        m = score[p]
        ids = m[m >= 0]
        if not np.array_equal(np.sort(ids), np.arange(len(ids))):
            raise RuntimeError(f"{p}: component scoreable row IDs are not a complete 0..N-1 map")
        n = len(ids)
        g = np.empty(n, dtype=object); sr = np.empty(n, dtype=np.int64)
        yy = np.empty(n, dtype=np.int8); cc = np.empty(n, dtype=np.int16)
        for cache_row in np.flatnonzero(m >= 0):
            row = int(m[cache_row])
            g[row] = groups[cache_row]
            sr[row] = source_rows[cache_row]
            yy[row] = CLASS_TO_ID[label[p][cache_row]]
            cc[row] = ci[comp[cache_row]]
        row_data[p] = {"group": g, "source_row": sr, "y": yy, "component": cc}
    return freeze, component_ids, component_strata, normal, mixed, row_data


def load_and_validate_predictions(row_data):
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((RUN / "analysis" / "main_A_analysis.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "phase_complete" or manifest.get("failure_count") != 0:
        raise RuntimeError("main_A_v2 manifest is not complete/failure-free")
    if analysis.get("status") != "complete_validated" or analysis.get("complete_model_folds") != 480:
        raise RuntimeError("main A independent analyzer did not validate 480 folds")
    if manifest.get("preflight_manifest_sha256") != sha(PREFLIGHT / "preflight_manifest.json"):
        raise RuntimeError("A run manifest does not match the frozen preflight manifest")
    if analysis.get("run_manifest_sha256") != sha(RUN / "run_manifest.json"):
        raise RuntimeError("A analyzer report does not match the live run manifest")
    table = {}
    hashes = {"run_manifest": sha(RUN / "run_manifest.json"),
              "analysis": sha(RUN / "analysis" / "main_A_analysis.json")}
    for p in POLICIES:
        idx = np.load(PREFLIGHT / "row_indices" / f"{p}.npz", allow_pickle=False)
        expected = row_data[p]
        if not np.array_equal(idx["group"].astype(str), expected["group"]):
            raise RuntimeError(f"{p}: frozen component recording map differs from A row map")
        if not np.array_equal(idx["source_row"], expected["source_row"]):
            raise RuntimeError(f"{p}: frozen component source-row map differs from A row map")
        if not np.array_equal(idx["y"], expected["y"]):
            raise RuntimeError(f"{p}: frozen content labels differ from A row map")
        n = len(idx["y"])
        for split in SPLITS:
            seen_method_rows = None
            for feat in ("full_numeric", "context_only"):
                for model in MODELS:
                    path = RUN / "predictions" / p / split / f"{feat}__{model}.npz"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    hashes[str(path.relative_to(ROOT))] = sha(path)
                    with np.load(path, allow_pickle=False) as pred:
                        rows = pred["test_row"].astype(np.int64)
                        yt = pred["y_true"].astype(np.int8)
                        yp = pred["y_pred"].astype(np.int8)
                    if len(rows) != n or len(np.unique(rows)) != n or not np.array_equal(np.sort(rows), np.arange(n)):
                        raise RuntimeError(f"{p}/{split}/{feat}/{model}: not a complete unique OOF row map")
                    if not np.array_equal(yt, expected["y"][rows]):
                        raise RuntimeError(f"{p}/{split}/{feat}/{model}: prediction labels differ from source rows")
                    if np.any(yp < 0) or np.any(yp > 2):
                        raise RuntimeError(f"{p}/{split}/{feat}/{model}: unexpected prediction class code")
                    ids = rows
                    if seen_method_rows is None:
                        seen_method_rows = np.sort(ids)
                    elif not np.array_equal(seen_method_rows, np.sort(ids)):
                        raise RuntimeError(f"{p}/{split}: feature/model OOF populations differ")
                    table[(p, split, feat, model)] = (rows, yt, yp)
    return table, hashes, analysis


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = HERE
    if (OUT / "main_A_component_bootstrap.json").exists() or (OUT / "provenance.json").exists():
        raise FileExistsError(f"Refusing to overwrite main A component bootstrap outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    freeze, component_ids, component_strata, normal_ci, mixed_ci, rows = build_component_rows()
    pred, pred_hashes, analysis = load_and_validate_predictions(rows)
    ncomp = len(component_ids)
    cms = np.zeros((2, 3, 2, 4, ncomp, 3, 3), dtype=np.int64)
    for pi, p in enumerate(POLICIES):
        for si, split in enumerate(SPLITS):
            for fi, feat in enumerate(("full_numeric", "context_only")):
                for mi, model in enumerate(MODELS):
                    test_rows, yt, yp = pred[(p, split, feat, model)]
                    np.add.at(cms[pi, si, fi, mi],
                              (rows[p]["component"][test_rows], yt, yp), 1)

    # Shared component weights across all 24 estimates; no redraw for degeneracy.
    rng = np.random.RandomState(SEED)
    draws = np.zeros((BOOTSTRAPS, ncomp), dtype=np.int16)
    for b in range(BOOTSTRAPS):
        for strata in (normal_ci, mixed_ci):
            sampled = rng.choice(strata, size=len(strata), replace=True)
            draws[b] += np.bincount(sampled, minlength=ncomp).astype(np.int16)
    contrast_defs = {
        "full_random_minus_record": (0, 0, 0, 1),
        "full_random_minus_timeblock": (0, 0, 0, 2),
        "context_minus_full_record": (1, 1, 0, 1),
    }
    point, ci_result, invalid = {}, {}, {}
    for pi, p in enumerate(POLICIES):
        point[p], ci_result[p], invalid[p] = {}, {}, {}
        for mi, model in enumerate(MODELS):
            point[p][model] = {}
            for split_i, split in enumerate(SPLITS):
                for fi, feat in enumerate(("full_numeric", "context_only")):
                    cm = cms[pi, split_i, fi, mi].sum(axis=0)
                    point[p][model][f"{split}/{feat}"] = metrics(cm)
            ci_result[p][model], invalid[p][model] = {}, {}
            for name, (fa, sa, fb, sb) in contrast_defs.items():
                vals = []
                degenerate = 0
                for w in draws:
                    cma = np.einsum("c,cij->ij", w, cms[pi, sa, fa, mi])
                    cmb = np.einsum("c,cij->ij", w, cms[pi, sb, fb, mi])
                    a, b = macro_f1(cma), macro_f1(cmb)
                    if np.isfinite(a) and np.isfinite(b):
                        vals.append(a - b)
                    else:
                        degenerate += 1
                vals = np.asarray(vals, dtype=np.float64)
                base_a = point[p][model][f"{SPLITS[sa]}/{('full_numeric','context_only')[fa]}"]["macro_f1"]
                base_b = point[p][model][f"{SPLITS[sb]}/{('full_numeric','context_only')[fb]}"]["macro_f1"]
                ci_result[p][model][name] = {
                    "pooled_oof_macro_f1_difference": float(base_a - base_b),
                    "percentile_ci95": [float(x) for x in np.percentile(vals, [2.5, 97.5])] if len(vals) else None,
                    "valid_replicates": int(len(vals)), "replicates": BOOTSTRAPS}
                invalid[p][model][name] = {"degenerate_replicates_not_redrawn": degenerate}

    # Provenance: freeze and all cohort/index/prediction sources are immutable inputs.
    inputs = {"content_component_freeze": sha(FREEZE), "content_component_membership": sha(MEMBERSHIP),
        "main_A_run_manifest": sha(RUN / "run_manifest.json"),
        "main_A_analysis": sha(RUN / "analysis" / "main_A_analysis.json"),
        "main_A_preflight_manifest": sha(PREFLIGHT / "preflight_manifest.json"),
        "source_cache": sha(PAPER.parent / "session_feature_cache_v1.joblib"), **pred_hashes}
    result = {"status": "complete_validated", "scope": "fixed-model pooled OOF component bootstrap",
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED, "component_count": ncomp,
            "normal_only_components": int(len(normal_ci)), "mixed_components": int(len(mixed_ci)),
            "shared_draws_across_all_methods_policies": True,
            "fixed_models_no_refit": True, "fold_level_intervals_or_p_values": False},
        "class_order": list(CLASSES), "populations": {
            p: {"rows": len(rows[p]["y"]), "recordings": int(len(np.unique(rows[p]["group"]))),
                "support": {c: int(np.sum(rows[p]["y"] == i)) for i, c in enumerate(CLASSES)}}
            for p in POLICIES},
        "point_metrics": point, "contrasts": ci_result, "degenerate_draws": invalid,
        "limitations": [
            "Intervals condition on the already-fitted models and their fixed OOF predictions; they do not include model-training or cross-validation uncertainty.",
            "CSV files are operational recording units. Components approximate content-linked source groups, not independently verified roads, trips, vehicles, or routes.",
            "The historical split estimands remain unchanged; this bootstrap does not remove possible train-test content overlap or leakage.",
            "Only pooled OOF macro-F1 contrasts receive intervals. No fold-mean intervals or p-values are reported."],
        "input_sha256": inputs}
    (OUT / "main_A_component_bootstrap.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez_compressed(OUT / "component_confusions_and_draws.npz", component_confusions=cms,
                        bootstrap_component_weights=draws, component_ids=np.asarray(component_ids),
                        component_strata=np.asarray(component_strata))
    prov = {"created_epoch": time.time(), "script_sha256": sha(Path(__file__)),
        "outputs": {"main_A_component_bootstrap.json": sha(OUT / "main_A_component_bootstrap.json"),
                    "component_confusions_and_draws.npz": sha(OUT / "component_confusions_and_draws.npz")},
        "input_sha256": inputs}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")

    # Full-feature split/model comparison: fold-mean points only; no bootstrap interval.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    colors = ["#3B6FB6", "#E07A35", "#3A9D73", "#A05AA1"]
    x = np.arange(3); width = 0.18
    for ax, p in zip(axes, POLICIES):
        for mi, model in enumerate(MODELS):
            vals = []
            for split in SPLITS:
                job = next(j for j in analysis["jobs"] if j["policy"] == p and j["split"] == split
                           and j["feature"] == "full_numeric" and j["model"] == model)
                vals.append(job["fold_mean_macro_f1"])
            ax.bar(x + (mi - 1.5) * width, vals, width, label=model, color=colors[mi])
        ax.set_title("Primary labels" if p == "primary" else "Blank labels treated as Normal")
        ax.set_xticks(x, ["Row-random", "Recording-heldout", "Time-block"])
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.25); ax.set_ylabel("Fold-mean macro-F1")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Full numeric features: split sensitivity across four models\nFold-mean points; no uncertainty bars")
    fig.tight_layout()
    fig.savefig(OUT / "main_A_full_numeric_foldmean.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "main_A_full_numeric_foldmean.svg", bbox_inches="tight")
    plt.close(fig)

    # Recording-heldout macro-F1 by feature family and model, showing weaker/contrary families.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    all_feats = ("full_numeric", "context_only", "imu_only", "missing_only")
    feat_names = ("Full numeric", "Context", "IMU", "Missingness")
    for ax, p in zip(axes, POLICIES):
        for fi, feat in enumerate(all_feats):
            vals = []
            for model in MODELS:
                job = next(j for j in analysis["jobs"] if j["policy"] == p and j["split"] == "record42"
                           and j["feature"] == feat and j["model"] == model)
                vals.append(job["fold_mean_macro_f1"])
            ax.plot(np.arange(4), vals, marker="o", linewidth=1.8, label=feat_names[fi])
        ax.set_title("Primary labels" if p == "primary" else "Blank labels treated as Normal")
        ax.set_xticks(np.arange(4), ["RF100", "ExtraTrees", "HistGB", "Logistic"])
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.25); ax.set_ylabel("Fold-mean macro-F1")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Recording-heldout results by measured feature family\nFold-mean points only; feature associations are not causal effects")
    fig.tight_layout()
    fig.savefig(OUT / "main_A_recording_features_foldmean.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "main_A_recording_features_foldmean.svg", bbox_inches="tight")
    plt.close(fig)

    # Full numeric, recording-heldout pooled OOF confusion panels with row normalization.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    for pi, p in enumerate(POLICIES):
        for mi, model in enumerate(MODELS):
            cm = cms[pi, 1, 0, mi].sum(axis=0).astype(float)
            cm = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[pi, mi]
            im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            color="white" if cm[i,j] > .6 else "black", fontsize=9)
            ax.set_xticks(range(3), CLASSES, rotation=28, ha="right")
            ax.set_yticks(range(3), CLASSES)
            if pi == 0:
                ax.set_title(model, fontsize=10)
            if mi == 0:
                ax.set_ylabel(("Primary labels" if pi == 0 else "Blank=Normal") + "\nTrue class")
            else:
                ax.set_yticklabels([])
            if pi == 1:
                ax.set_xlabel("Predicted class")
            else:
                ax.set_xticklabels([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.84, pad=.02, label="Within-class row proportion")
    fig.suptitle("Full numeric recording-heldout OOF confusion\nDiagonal cells are class recalls; row-normalized pooled predictions")
    fig.savefig(OUT / "main_A_recording_full_confusion.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "main_A_recording_full_confusion.svg", bbox_inches="tight")
    plt.close(fig)

    readme = ["# Main A component bootstrap and figures", "",
        "This directory analyzes completed main_A_v2 compact OOF predictions without fitting models. The component bootstrap reports 24 pooled-OOF macro-F1 contrasts: three prespecified contrasts for each of four models under each of the two label policies.", "",
        "The fixed-model percentile intervals resample the frozen 96 content components (five Normal-only and 91 mixed) 2,000 times with seed 20261016. Shared component weights are used across models, splits, features, and policies. Degenerate replicates are counted and not redrawn. No fold-level intervals or p-values are reported.", "",
        "The first figure plots full-numeric fold-mean macro-F1 by split and model; bars are point estimates, with no uncertainty intervals. The second shows recording-heldout results by feature family: IMU is weaker than full numeric under both policies and missingness is policy-sensitive (higher under primary labels, lower under blank-as-Normal). Context-versus-full is model-dependent: for primary-label pooled OOF, the difference is positive for RF100 (+0.1853, 95% CI [+0.0812, +0.3154]), negative for LogisticRegression (−0.1297, [−0.2353, −0.0596]), and crosses zero for ExtraTrees100 and HistGB. Under blank-as-Normal, all four context-versus-full intervals cross zero. These are predictive associations, not causal effects. The third provides row-normalized pooled confusion matrices for full-numeric recording-heldout OOF predictions; diagonals are class recall.", "",
        "These are recording-level/content-component sensitivity summaries, not validated road/route/vehicle independence, causal effects, population significance, deployment performance, or training uncertainty. Historical split procedures remain unchanged; content-component resampling does not repair any training/test content overlap.", "",
        "Machine-readable estimates: `main_A_component_bootstrap.json`. Compact confusion tensors and shared bootstrap weights: `component_confusions_and_draws.npz`. Input/output hashes: `provenance.json`."]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "contrasts": 24,
        "models": len(MODELS), "policies": len(POLICIES), "features": len(FEATURES),
        "bootstrap": BOOTSTRAPS, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    run()
