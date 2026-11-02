"""Independent fit-free verification for a spatial-binary preflight directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
EXPECTED_PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
LABELS = ("Pothole", "Bump")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(directory: Path):
    import sklearn
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve() or sklearn.__version__ != "1.9.0":
        raise RuntimeError("QA must use the locked Miniconda Python and sklearn 1.9.0")
    pf_path = directory / "preflight.json"
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    if pf["fit_performed"] is not False or pf["status"] != "preflight_complete":
        raise AssertionError("input is not a successful fit-free preflight")
    expected_members = {"population_rows.npz", "grid100m_splits.npz", "grid500m_splits.npz"}
    if set(pf.get("membership_files_sha256", {})) != expected_members:
        raise AssertionError("preflight does not hash all immutable membership NPZ files")
    for name, digest in pf["membership_files_sha256"].items():
        if sha(directory / name) != digest:
            raise AssertionError(f"membership artifact SHA mismatch: {name}")
    p = np.load(directory / "population_rows.npz", allow_pickle=True)
    y = p["y"].astype(str)
    group = p["group"].astype(str)
    src = p["source_row"].astype(np.int64)
    xy = np.column_stack([p["x"], p["y_projected"]]).astype(np.float64)
    n = len(y)
    if n != 28478 or len(np.unique(np.column_stack([group, src]), axis=0)) != n:
        raise AssertionError("cohort size or source identity uniqueness failed")
    if set(np.unique(y)) != set(LABELS):
        raise AssertionError("cohort label support changed")
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(HERE))
    import runner
    report = {"status": "passed", "fit_performed": False,
              "cohort_n": n, "source_rows_unique": True, "grids": {}}
    for width in (100, 500):
        gid = f"grid{width}m"
        s = np.load(directory / f"{gid}_splits.npz", allow_pickle=True)
        cell_xy = s["cell_xy"].astype(np.int64)
        cell_id = s["cell_id"].astype(str)
        if not np.array_equal(cell_xy, np.floor(xy / width).astype(np.int64)):
            raise AssertionError(f"{gid}: cell coordinates mismatch")
        u, c = np.unique(cell_id, return_counts=True)
        if not np.array_equal(u, s["unique_cell_id"].astype(str)) or not np.array_equal(c, s["cell_frequency"]):
            raise AssertionError(f"{gid}: all-cell frequency inventory mismatch")
        all_test = []
        fold_checks = []
        tree_errors = []
        for fold in range(5):
            def rows(name): return s[f"fold{fold}_{name}"].astype(np.int64)
            tr0, removed, tr, fit, te = (rows(k) for k in
                ("candidate_train", "removed", "post_buffer_train", "fit", "test"))
            all_test.extend(te.tolist())
            if len(np.intersect1d(tr0, te)) or not np.array_equal(np.sort(np.concatenate([tr0, te])), np.arange(n)):
                raise AssertionError(f"{gid}/fold{fold+1}: candidate train/test are not a partition")
            if set(cell_id[tr0]) & set(cell_id[te]):
                raise AssertionError(f"{gid}/fold{fold+1}: a test cell occurs in candidate train")
            if len(np.intersect1d(removed, tr)) or not np.array_equal(np.sort(np.concatenate([removed, tr])), tr0):
                raise AssertionError(f"{gid}/fold{fold+1}: purge rows do not partition candidate train")
            if not set(fit).issubset(set(tr)) or len(fit) > 20000:
                raise AssertionError(f"{gid}/fold{fold+1}: capped fit rows are invalid")
            expected_fit = runner.cap_train_indices(tr, y, fold, 42, cap=20000)
            if not np.array_equal(expected_fit, fit):
                raise AssertionError(f"{gid}/fold{fold+1}: cap rows do not match frozen quota sampler")
            for name, ix in (("candidate_train", tr0), ("test", te), ("post_buffer_train", tr), ("fit", fit)):
                if any(int(np.sum(y[ix] == cls)) == 0 for cls in LABELS):
                    raise AssertionError(f"{gid}/fold{fold+1}: missing class in {name}")
            dte, _ = cKDTree(xy[te]).query(xy[tr0], k=1, eps=0)
            expected_removed = tr0[dte <= 100.0]
            expected_retained = tr0[dte > 100.0]
            if not np.array_equal(np.sort(expected_removed), np.sort(removed)) or not np.array_equal(np.sort(expected_retained), np.sort(tr)):
                raise AssertionError(f"{gid}/fold{fold+1}: exact 100m purge differs")
            dfit, _ = cKDTree(xy[te]).query(xy[fit], k=1, eps=0)
            dmin = float(np.min(dfit))
            if dmin <= 100.0:
                raise AssertionError(f"{gid}/fold{fold+1}: fit/test distance {dmin}m is <=100m")
            if abs(dmin - pf["grids"][gid]["folds"][fold]["nearest_test_distance_fit_min"]) > 1e-9:
                raise AssertionError(f"{gid}/fold{fold+1}: recorded nearest distance mismatch")
            fold_checks.append({"fold": fold + 1, "candidate_train_n": len(tr0),
                "purged_n": len(removed), "fit_n": len(fit), "test_n": len(te),
                "class_support": {side: {cls: int(np.sum(y[ix] == cls)) for cls in LABELS}
                    for side, ix in (("fit", fit), ("test", te))},
                "nearest_test_fit_m": dmin})
            tree_errors.append(dmin)
        if sorted(all_test) != list(range(n)):
            raise AssertionError(f"{gid}: test folds do not form exactly one OOF partition")
        report["grids"][gid] = {"cells": int(len(u)), "all_cell_frequencies_sum": int(c.sum()),
            "all_test_rows_held_out_once": True, "min_fit_test_distance_m": min(tree_errors),
            "folds": fold_checks}
    report["preflight_sha256"] = sha(pf_path)
    report["qa_script_sha256"] = sha(Path(__file__))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preflight_dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    result = verify(a.preflight_dir.resolve())
    if a.out:
        a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
