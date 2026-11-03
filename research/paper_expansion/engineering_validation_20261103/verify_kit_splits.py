#!/usr/bin/env python3
"""Independent checks of saved KIT split membership and spatial buffers only."""
import json, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

BASE=Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion\engineering_validation_20261103")
RUN=BASE/"run_kit_v1"; OLD=BASE/"run_kit_v1_preflight_incomplete"/"split_map.npz"
FEATURE=BASE/"features_kit_v1.npz"; RAW=BASE/"kit_audit"/"raw_arrays_verified.npz"
PROTOCOL=BASE/"kit_protocol_v1.json"; REPORT=BASE/"kit_split_verification.json"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def projected_window_coords(z,f,i):
 fid=int(f["file_id"][i]); p=f"f{fid:02d}"; lo=int(f["source_start"][i]); hi=int(f["source_end"][i])
 lat=z[p+"_lat"][lo:hi+1]; lon=z[p+"_lon"][lo:hi+1]
 good=np.isfinite(lat)&np.isfinite(lon)&(lat>=48)&(lat<=50)&(lon>=7)&(lon<=10)
 # Protocol purge rule: any invalid/out-of-region coordinate makes training window ineligible.
 if not np.all(good): return None
 return np.column_stack(((lon-8.4)*111320*np.cos(np.deg2rad(49)),(lat-49)*111320))

def main():
 pre=json.loads((RUN/"preflight.json").read_text(encoding="utf-8")); proto=json.loads(PROTOCOL.read_text(encoding="utf-8"))
 f=np.load(FEATURE); z0=np.load(RAW,allow_pickle=False); z={k:z0[k] for k in z0.files}
 cur=np.load(RUN/"split_map.npz"); old=np.load(OLD)
 assert set(cur.files)==set(old.files)
 equality={k:bool(np.array_equal(cur[k],old[k])) for k in cur.files}; assert all(equality.values())
 assert sha(FEATURE)==pre["features_sha256"] and sha(PROTOCOL)==pre["protocol_sha256"]
 ot=np.asarray(cur["outer_train"],int); oe=np.asarray(cur["outer_test"],int)
 assert len(set(ot)&set(oe))==0
 assert np.array_equal(np.sort(oe),np.flatnonzero(f["day_id"]==2))
 assert np.all(np.isin(f["day_id"][ot],[0,1]))
 assert np.all(np.isin(f["day_id"][oe],[2]))
 # Test anchors are the saved eligible outer-test rows; raw training windows use every
 # coordinate in their closed saved row interval for the full-window spatial purge.
 def buffer_min(indices, reference_indices):
  ref=np.asarray(reference_indices,dtype=int)
  ref_xy=np.column_stack(((f["X_context"][ref,1]-8.4)*111320*np.cos(np.deg2rad(49)),(f["X_context"][ref,0]-49)*111320))
  tree=cKDTree(ref_xy)
  mins=[]; invalid=0
  for ix in indices:
   pts=projected_window_coords(z,f,int(ix))
   if pts is None: invalid+=1; continue
   mins.append(float(np.min(tree.query(pts,k=1)[0])))
  return (min(mins) if mins else None),invalid
 outer_min,outer_invalid=buffer_min(ot,oe)
 assert outer_invalid==0 and outer_min>100.0, (outer_min,outer_invalid)
 # Inner random folds intentionally permit overlapping raw windows. Check only grouped
 # date folds for full-window spatial buffer and source-file/date separation.
 inner=[]
 for j in range(4):
  tr=np.asarray(cur[f"inner_{j}_train"],int); va=np.asarray(cur[f"inner_{j}_validation"],int); fit=np.asarray(cur[f"inner_{j}_fit"],int)
  assert set(fit).issubset(set(tr))
  assert len(np.unique(f["y"][fit],return_counts=True)[1])==2
  counts=np.bincount(f["y"][fit],minlength=2).tolist(); assert counts==[1613,17],(j,counts)
  grouped=j>=2
  if grouped:
   assert set(f["day_id"][tr]).isdisjoint(set(f["day_id"][va]))
   assert set(f["file_id"][tr]).isdisjoint(set(f["file_id"][va]))
   bmin,invalid=buffer_min(tr,va); assert invalid==0 and bmin>100.0,(j,bmin,invalid)
   expected_min=pre["inner_splits"][j]["buffer_min_m"]
   assert np.isclose(bmin,expected_min,rtol=0,atol=1e-6),(j,bmin,expected_min)
  else: bmin=None
  inner.append({"split":j,"selector":"date_buffered" if grouped else "row_random","train_rows":len(tr),"validation_rows":len(va),"fit_class_counts":counts,"train_validation_file_disjoint":bool(set(f["file_id"][tr]).isdisjoint(set(f["file_id"][va]))),"train_validation_date_disjoint":bool(set(f["day_id"][tr]).isdisjoint(set(f["day_id"][va]))),"min_full_window_to_test_anchor_m":bmin})
 # Each selector's two validation folds must partition outer_train exactly once.
 coverage={}
 for label,indices in (("row_random",[0,1]),("date_buffered",[2,3])):
  vals=np.concatenate([cur[f"inner_{j}_validation"] for j in indices])
  assert len(vals)==len(ot) and len(np.unique(vals))==len(ot) and set(vals)==set(ot)
  coverage[label]={"validation_rows_total":len(vals),"unique_rows":len(np.unique(vals)),"equals_outer_train":True}
 report={"status":"passed","scope":"saved split membership and spatial grouping/buffer audit only; model outcomes not read",
         "split_maps_equal_old_incomplete":equality,"outer_train_rows":len(ot),"outer_test_rows":len(oe),
         "outer_test_exactly_day2":True,"outer_train_only_days_0_1":True,
         "outer_train_min_full_window_to_test_anchor_m":outer_min,"outer_invalid_coordinate_windows":outer_invalid,
         "inner_common_fit_class_budget":[1613,17],"inner_splits":inner,"validation_coverage":coverage,
         "hashes":{"features_matches_preflight":True,"protocol_matches_preflight":True},
         "raw_window_overlap_policy":"row-random selector permits overlap by design; no overlap prohibition asserted",
         "previous_wrong_reference":True,
         "correction_note":"Previous report computed inner buffer distances against outer-test anchors. Corrected to each split's validation anchors; split arrays were not changed.",
         "protocol_archive_sha256":proto["archive_sha256"]}
 REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print(json.dumps({"status":"passed","report":str(REPORT),"outer_min_m":outer_min,"inner":inner},ensure_ascii=False))

if __name__=="__main__":main()
