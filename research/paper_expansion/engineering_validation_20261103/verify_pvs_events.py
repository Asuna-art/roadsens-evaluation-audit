"""Independent read-only checks of the saved PVS annotation-event diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EV = Path(__file__).resolve().parent
ROOT = BASE / "external_pvs_v1" / "run_primary_v1"

def maxmatch(edges, nalarms):
    owner = [-1] * nalarms
    def aug(e, seen):
        for a in edges[e]:
            if a in seen: continue
            seen.add(a)
            if owner[a] < 0 or aug(owner[a], seen):
                owner[a] = e
                return True
        return False
    for e in range(len(edges)): aug(e, set())
    return sum(x >= 0 for x in owner)

def main():
    current = json.loads((EV / "pvs_event_results.json").read_text(encoding="utf-8"))
    first = json.loads((EV / "pvs_event_results_v1_first_run.json").read_text(encoding="utf-8"))
    schema = json.loads((EV / "pvs_schema_audit.json").read_text(encoding="utf-8"))
    features = np.load(BASE / "external_pvs_v1" / "features.npz", allow_pickle=False)
    y, trip, rows, times = (features[k] for k in ("y", "trip", "anchor_row", "anchor_time"))
    segs = {int(x["trip"]): x["positive_annotation_segments_gap_le_0.1s"] for x in schema["trips"]}
    trip_report = {}
    for tid in range(1, 10):
        labpath = BASE / "external_pvs_v1" / "data" / f"PVS_{tid}" / "dataset_labels.csv"
        lab = pd.read_csv(labpath, usecols=["no_speed_bump", "speed_bump_asphalt", "speed_bump_cobblestone"]).to_numpy()
        valid = np.isin(lab, [0, 1]).all(axis=1) & (lab.sum(axis=1) == 1)
        rawy = (lab[:, 1:].sum(axis=1) == 1).astype(int)
        fi = np.flatnonzero(trip == tid)
        assert np.array_equal(y[fi], rawy[rows[fi]])
        trip_report[str(tid)] = {"annotation_intervals": len(segs[tid]), "feature_anchor_label_exact_raw_match": True,
                                 "positive_alarm_clusters_by_outer_selector": {}}
    cluster_counts = {}
    for axis in ("vehicle", "scenario"):
        for held in range(1, 4):
            outer = f"{axis}_{held}"
            sel = json.loads((ROOT / "outer_folds" / outer / "selection.json").read_text(encoding="utf-8"))
            test_trip = ((trip - 1)//3 + 1 == held) if axis == "vehicle" else ((trip - 1)%3 + 1 == held)
            for selector in ("row_random", "leave_one_trip", "matched_factor"):
                candidate = sel["selectors"][selector]["selected_candidate_id"]
                p = ROOT / "fits" / f"{outer}__outer_refit__{candidate}" / "predictions.npz"
                with np.load(p, allow_pickle=False) as z:
                    ix = z["test_indices"] if "test_indices" in z.files else z["indices"]
                    pred = z["y_pred"].astype(int)
                ix = ix.reshape(-1); pred = pred.reshape(-1)
                for tid in sorted(set(trip[ix].tolist())):
                    selidx = np.flatnonzero(trip[ix] == tid)
                    o = selidx[np.argsort(rows[ix[selidx]])]
                    rr, tt, pp = rows[ix[o]], times[ix[o]], pred[o]
                    clusters=0; active=False; last=-1
                    for j in range(len(pp)):
                        if pp[j] == 1:
                            if not (active and pp[j-1] == 1 and rr[j]-rr[j-1] == 25 and tt[j]-tt[j-1] <= .251): clusters += 1
                            active=True
                        else: active=False
                    key=f"{axis}/{outer}/{selector}"
                    cluster_counts.setdefault(key,{})[str(tid)]=clusters
    # Independently recompute annotation interval count and zero-tolerance match counts from saved predictions.
    match_counts = {}
    current_rows = current["by_trip_domain_selector_tolerance"]
    for axis in ("vehicle", "scenario"):
      for held in range(1,4):
       outer=f"{axis}_{held}"; sel=json.loads((ROOT/"outer_folds"/outer/"selection.json").read_text())
       for selector in ("row_random","leave_one_trip","matched_factor"):
        candidate=sel["selectors"][selector]["selected_candidate_id"]
        with np.load(ROOT/"fits"/f"{outer}__outer_refit__{candidate}"/"predictions.npz",allow_pickle=False) as z:
         ix=z["test_indices"] if "test_indices" in z.files else z["indices"]; pp=z["y_pred"].astype(int)
        ix=ix.reshape(-1); pp=pp.reshape(-1)
        for tid in sorted(set(trip[ix].tolist())):
         ids=np.flatnonzero(trip[ix]==tid); ids=ids[np.argsort(rows[ix[ids]])]
         rr,tt,hh=rows[ix[ids]],times[ix[ids]],pp[ids]
         alarms=[]; active=[]
         for j,h in enumerate(hh):
          if h==1:
           if active and not(hh[j-1]==1 and rr[j]-rr[j-1]==25 and tt[j]-tt[j-1]<=.251): alarms.append(active);active=[]
           active.append(j)
          elif active: alarms.append(active);active=[]
         if active:alarms.append(active)
         events=[]
         for s in segs[int(tid)]:
          if np.any((rr>=s["start_row_zero_based"])&(rr<=s["end_row_zero_based"])):events.append(s)
         edges=[]
         for s in events:
          edges.append([a for a,c in enumerate(alarms) if np.any((tt[c]>=s["start_timestamp"])&(tt[c]<=s["end_timestamp"]))])
         cnt=maxmatch(edges,len(alarms)); key=f"{axis}/{outer}/{selector}/{tid}"
         match_counts[key]=cnt
    # Compare each old v1 field except integrity/source hashes, and new amended-only fields.
    oldmap={(r["axis"],r["outer_id"],r["selector"],r["trip"],r["tolerance_seconds"]):r for r in first["by_trip_domain_selector_tolerance"]}
    newmap={(r["axis"],r["outer_id"],r["selector"],r["trip"],r["tolerance_seconds"]):r for r in current_rows}
    changed=[]
    for k,o in oldmap.items():
      n=newmap[k]
      for field,value in o.items():
       if field not in ("source_artifact_sha256","integrity") and n.get(field)!=value: changed.append({"key":str(k),"field":field})
    total_intervals=sum(len(v) for v in segs.values())
    exposures={}
    for tid in range(1,10):
      rows_here=rows[trip==tid]
      l=pd.read_csv(BASE/"external_pvs_v1"/"data"/f"PVS_{tid}"/"dataset_labels.csv",usecols=["no_speed_bump","speed_bump_asphalt","speed_bump_cobblestone"]).to_numpy()
      valid=np.isin(l,[0,1]).all(axis=1)&(l.sum(axis=1)==1); non=valid&(l[:,0]==1)
      exposures[str(tid)]=int(non[rows_here.min():rows_here.max()+1].sum())*.01/3600
    report={"total_annotation_intervals":total_intervals,"annotation_intervals_by_trip":{str(k):len(v) for k,v in segs.items()},
      "positive_alarm_clusters":cluster_counts,"zero_tolerance_matched_counts":match_counts,
      "nonbump_exposure_hours_by_trip":exposures,"nonbump_exposure_hours_total":sum(exposures.values()),
      "historical_v1_fields_unchanged":not changed,"changed_historical_fields":changed,
      "all_raw_anchor_labels_exact":True}
    (EV/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(EV/"verification.json")
if __name__=="__main__": main()
