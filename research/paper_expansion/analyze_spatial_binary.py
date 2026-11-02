"""Independent fit-free audit and compact report for spatial_binary_v1."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
LABELS = ("Pothole", "Bump")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics(y, pred):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    return {"n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "support": [int(np.sum(y == i)) for i in range(2)],
        "recall": [float(x) for x in recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)],
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def analyze(run_dir: Path, preflight_dir: Path):
    import sklearn
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve() or sklearn.__version__ != "1.9.0":
        raise RuntimeError("analysis must use locked Miniconda Python / sklearn 1.9.0")
    sys.path.insert(0, str(HERE))
    from qa_spatial_binary import verify as verify_preflight
    qa = verify_preflight(preflight_dir)
    if qa["status"] != "passed":
        raise AssertionError("spatial preflight QA failed")
    mpath = run_dir / "run_manifest.json"
    mp = json.loads(mpath.read_text(encoding="utf-8"))
    if mp.get("status") != "phase_complete" or mp.get("fit_count_completed") != 60 or mp.get("failure_count", 0):
        raise AssertionError("run is not a successful complete 60-fit matrix")
    pf = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    for key in ("source_sha256", "cache_sha256", "cache_parent_runner_sha256", "immutable_runner_sha256",
                "protocol_sha256", "spatial_driver_sha256"):
        if mp.get(key) != pf.get(key):
            raise AssertionError(f"run/preflight provenance mismatch: {key}")
    if mp.get("preflight_sha256") != sha(preflight_dir / "preflight.json"):
        raise AssertionError("preflight file changed after run")
    metrics_path = run_dir / "metrics.json"
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    if stored.get("status") != "phase_complete" or len(stored.get("folds", [])) != 60:
        raise AssertionError("metrics are not a complete 60-fold record")
    pop = np.load(preflight_dir / "population_rows.npz", allow_pickle=True)
    y_codes = {"Pothole": 0, "Bump": 1}
    y_all = np.asarray([y_codes[str(x)] for x in pop["y"]], dtype=np.int8)
    folds_report = []
    combos = []
    for width in (100, 500):
        gid = f"grid{width}m"
        splits = np.load(preflight_dir / f"{gid}_splits.npz", allow_pickle=True)
        for feature in ("full_numeric", "context_only", "imu_only"):
            for model in ("RF100", "HistGB"):
                key = (gid, feature, model)
                fold_rows = [r for r in stored["folds"] if (r["grid"], r["feature"], r["model"]) == key]
                fold_rows.sort(key=lambda z: z["fold"])
                if [r["fold"] for r in fold_rows] != [1, 2, 3, 4, 5]:
                    raise AssertionError(f"missing/duplicate folds for {key}")
                pred_path = run_dir / "predictions" / gid / f"{feature}__{model}.npz"
                pred = np.load(pred_path, allow_pickle=True)
                expected_rows = np.concatenate([splits[f"fold{i}_test"] for i in range(5)]).astype(np.int64)
                rows, yt, yp = pred["test_row"].astype(np.int64), pred["y_true"].astype(np.int8), pred["y_pred"].astype(np.int8)
                if not np.array_equal(rows, expected_rows) or len(np.unique(rows)) != len(rows):
                    raise AssertionError(f"OOF prediction row mapping mismatch for {key}")
                if not np.array_equal(yt, y_all[rows]):
                    raise AssertionError(f"stored truth differs from frozen source labels for {key}")
                if len(rows) != len(y_all) or np.any((yp < 0) | (yp > 1)):
                    raise AssertionError(f"OOF coverage or prediction labels invalid for {key}")
                offset = 0
                recomputed = []
                for f, fr in enumerate(fold_rows):
                    te = splits[f"fold{f}_test"].astype(np.int64)
                    pred_f = yp[offset:offset + len(te)]
                    true_f = yt[offset:offset + len(te)]
                    offset += len(te)
                    met = metrics(true_f, pred_f)
                    for metric in ("macro_f1", "balanced_accuracy"):
                        if abs(met[metric] - fr[metric]) > 1e-12:
                            raise AssertionError(f"{key} fold {f+1}: {metric} mismatch")
                    if met["support"] != [fr["class_support"][c] for c in LABELS]:
                        raise AssertionError(f"{key} fold {f+1}: class support mismatch")
                    artifact = run_dir / fr["model_artifact"]
                    if sha(artifact) != fr["model_sha256"]:
                        raise AssertionError(f"model artifact SHA mismatch: {artifact.name}")
                    recomputed.append(met)
                pooled = metrics(yt, yp)
                combos.append({"grid": gid, "feature": feature, "model": model,
                    "fold_mean_macro_f1": float(np.mean([x["macro_f1"] for x in recomputed])),
                    "fold_mean_balanced_accuracy": float(np.mean([x["balanced_accuracy"] for x in recomputed])),
                    "pooled_oof": pooled, "fit_count": 5,
                    "mean_fit_seconds": float(np.mean([x["fit_wall_seconds"] for x in fold_rows])),
                    "warning_count": int(sum(x["warning_count"] for x in fold_rows))})
    summary = {"status": "complete_validated", "fit_performed_by_analyzer": False,
        "matrix": "60/60 folds independently verified", "preflight_qa_status": qa["status"],
        "source_population": {"n": int(len(y_all)), "labels": list(LABELS),
            "support": [int(np.sum(y_all == i)) for i in range(2)], "recordings": 96},
        "warning_count": int(sum(x["warning_count"] for x in stored["folds"])),
        "results": combos}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("preflight_dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    summary = analyze(a.run_dir.resolve(), a.preflight_dir.resolve())
    out = a.out or (a.run_dir / "independent_summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Spatial binary result summary", "",
          "Explicit Pothole/Bump rows with finite observed GPS only; this is a descriptive spatial-cell diagnostic, not a three-class, road/route, or deployment benchmark.", "",
          "| Grid | Features | Model | Fold-mean macro-F1 | Fold-mean balanced accuracy | Pooled OOF macro-F1 |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["results"]:
        md.append(f"| {r['grid']} | {r['feature']} | {r['model']} | {r['fold_mean_macro_f1']:.4f} | {r['fold_mean_balanced_accuracy']:.4f} | {r['pooled_oof']['macro_f1']:.4f} |")
    (a.run_dir / "independent_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "combination_count": len(summary["results"]),
                      "warning_count": summary["warning_count"], "summary": str(out)}, indent=2))

if __name__ == "__main__":
    main()
