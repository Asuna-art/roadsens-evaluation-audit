#!/usr/bin/env python3
"""Independent integrity checks for KIT features; does not import the builder."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion\engineering_validation_20261103")
RAW = BASE / "kit_audit" / "raw_arrays_verified.npz"
GROUP = BASE / "kit_audit" / "group_audit_verified.json"
PROTO = BASE / "kit_protocol_v1.json"
FEATURE = BASE / "features_kit_v1.npz"
MANIFEST = BASE / "features_kit_v1_manifest.json"
REPORT = BASE / "kit_feature_verification.json"
SELECTED_DATES = ["2016-05-13", "2016-06-20", "2016-07-05"]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def dates(a):
    a=a[np.isfinite(a)]
    return {datetime.fromtimestamp(float(v),timezone.utc).date().isoformat() for v in a}

def stats(a):
    return np.column_stack((a.mean(0),a.std(0,ddof=0),np.sqrt(np.mean(a*a,0)),np.ptp(a,axis=0),np.max(np.abs(a),axis=0))).reshape(-1)

def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8")); protocol=json.loads(PROTO.read_text(encoding="utf-8")); group=json.loads(GROUP.read_text(encoding="utf-8"))
    hash_checks={"raw_arrays_manifest":sha(RAW)==manifest["input_sha256"][RAW.name],
                 "group_audit_manifest":sha(GROUP)==manifest["input_sha256"][GROUP.name],
                 "protocol_archive_equals_group_audit":protocol["archive_sha256"]==group["archive_sha256"]}
    assert all(hash_checks.values()), hash_checks
    assert group["archive_sha256"]==protocol["archive_sha256"]
    z=np.load(RAW,allow_pickle=False); raw={k:z[k] for k in z.files}; f=np.load(FEATURE,allow_pickle=False)
    expected={"X_sensor","X_context","y","file_id","day_id","anchor_row","source_start","source_end","anchor_time","decision_time"}
    assert set(f.files)==expected, f.files
    n=len(f["y"]); assert n==manifest["rows"]==5226
    assert f["X_sensor"].shape==(n,30) and f["X_context"].shape==(n,3)
    dayfile={int(r["file_id"][1:]):r for r in group["files"]}
    cache={}
    for fid, rec in dayfile.items():
        p=rec["file_id"]
        gd=dates(raw[p+"_gps_timestamp"])
        cache[fid]=(p,next(iter(gd)) if len(gd)==1 else None)
    day_to_int={d:i for i,d in enumerate(SELECTED_DATES)}
    checks={"rows":n,"all_anchor_and_window_checks":True,"random_recomputation_samples":[]}
    bydayclass={}
    for i in range(n):
        fid=int(f["file_id"][i]); p,gd=cache[fid]
        assert gd is not None
        assert gd in day_to_int and int(f["day_id"][i])==day_to_int[gd]
        ts=raw[p+"_timestamp"]; ax=raw[p+"_axes6"]; lab=raw[p+"_label"]
        lat=raw[p+"_lat"]; lon=raw[p+"_lon"]; speed=raw[p+"_speed"]
        a=int(f["anchor_row"][i]); lo=int(f["source_start"][i]); hi=int(f["source_end"][i]); d=float(f["decision_time"][i])
        assert lo<=a==hi<len(ts)
        assert float(ts[a])<=d+1e-12 and (a+1==len(ts) or float(ts[a+1])>d)
        expected_lo=int(np.searchsorted(ts,d-1.0,side="left")); assert lo==expected_lo
        assert np.isclose(float(f["anchor_time"][i]),float(ts[a]),rtol=0,atol=1e-10)
        w=ax[lo:a+1]; assert len(w)>=50 and np.isfinite(w).all()
        assert np.all(ts[lo:a+1]>=d-1.0-1e-10) and np.all(ts[lo:a+1]<=d+1e-10)
        assert len(w)<2 or float(np.max(np.diff(ts[lo:a+1])))<=0.1+1e-12
        assert lab[a] in (1,3) and int(f["y"][i])==(0 if lab[a]==1 else 1)
        assert np.isfinite(lat[a]) and np.isfinite(lon[a]) and 48<=lat[a]<=50 and 7<=lon[a]<=10
        assert np.allclose(f["X_context"][i,:2],[lat[a],lon[a]],rtol=0,atol=1e-12)
        if np.isfinite(speed[a]): assert np.isclose(f["X_context"][i,2],speed[a],rtol=0,atol=1e-12)
        bydayclass.setdefault((gd,int(f["y"][i])),[]).append(i)
    for di,day in enumerate(SELECTED_DATES):
        for cls in (0,1):
            ids=bydayclass.get((day,cls),[]); assert len(ids)>=5, (day,cls,len(ids))
            samples=[ids[k] for k in np.linspace(0,len(ids)-1,5,dtype=int)]
            for i in samples:
                p="f%02d"%int(f["file_id"][i]); lo=int(f["source_start"][i]); hi=int(f["source_end"][i])
                calc=stats(raw[p+"_axes6"][lo:hi+1]); err=float(np.max(np.abs(calc-f["X_sensor"][i])))
                assert np.allclose(calc,f["X_sensor"][i],rtol=1e-11,atol=1e-11), (day,cls,i,err)
                checks["random_recomputation_samples"].append({"date":day,"class":cls,"feature_row":int(i),"max_abs_error":err})
    checks["date_class_counts"]={d:{str(c):len(bydayclass.get((d,c),[])) for c in (0,1)} for d in SELECTED_DATES}
    checks["sha256_checks"]=hash_checks
    checks["protocol_archive_sha256"]=protocol["archive_sha256"]
    checks["raw_archive_hash_note"]="verified audit archive hash equals frozen protocol archive hash"
    checks["scope"]="independent data integrity and sampled feature recomputation; no split, fit, or model"
    REPORT.write_text(json.dumps(checks,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","report":str(REPORT),"rows":n,"sampled_windows":len(checks["random_recomputation_samples"])},ensure_ascii=False))

if __name__=="__main__": main()
