#!/usr/bin/env python
"""Frozen KIT nested selection/estimation runner. Default mode is fit-free."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

BASE = Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion")
HERE = BASE / "engineering_validation_20261103"
PROTOCOL = HERE / "kit_protocol_v1.json"
PROTOCOL_SHA = "2782b0f43fd442a408f9acb8e6521cf7e64432abec45172a800c3983bb9d9105"
RAW = HERE / "kit_audit" / "raw_arrays_verified.npz"
FEATURES = HERE / "features_kit_v1.npz"
MANIFEST = HERE / "features_kit_v1_manifest.json"
OUT = HERE / "run_kit_v1"
DEV_DATES = (0, 1); FINAL_DATE = 2; CAP = 20000
CANDIDATES = ({"id":"rf_sensor","estimator":"rf","features":"sensor"},
 {"id":"rf_sensor_context","estimator":"rf","features":"sensor_context"},
 {"id":"hgb_sensor","estimator":"hgb","features":"sensor"},
 {"id":"hgb_sensor_context","estimator":"hgb","features":"sensor_context"})

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def writej(p,o):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf8'); os.replace(t,p)
def fail_existing(out):
 if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing to overwrite existing run files: {out}")
def load():
 if sha(PROTOCOL)!=PROTOCOL_SHA: raise ValueError('protocol SHA256 mismatch')
 with np.load(FEATURES,allow_pickle=False) as z: d={k:z[k] for k in z.files}
 required=('X_sensor','X_context','y','file_id','day_id','source_start','source_end')
 if any(k not in d for k in required): raise ValueError('features NPZ schema missing required fields')
 n=len(d['y'])
 if d['X_sensor'].shape!=(n,30) or d['X_context'].shape!=(n,3): raise ValueError('feature matrix dimensions mismatch')
 if np.any(d['source_end']<d['source_start']) or not set(np.unique(d['y'])).issubset({0,1}): raise ValueError('bad labels/source spans')
 with np.load(RAW,allow_pickle=False) as r: raw={k:r[k] for k in r.files}
 return d,raw
def projected(lat,lon): return np.column_stack(((lon-8.4)*111320*np.cos(np.deg2rad(49)),(lat-49)*111320))
def buffer_purge(train,test,d,raw):
 # One KDTree query per source file; window is rejected if any raw coordinate is invalid or within 100m.
 keep=[]; min_dist=float('inf')
 tree=cKDTree(projected(d['X_context'][test,0],d['X_context'][test,1]))
 distances={}
 for fid in np.unique(d['file_id'][train]):
  pre=f'f{int(fid):02d}_';lat=raw[pre+'lat'];lon=raw[pre+'lon']
  valid=np.isfinite(lat)&np.isfinite(lon)&(lat>=48)&(lat<=50)&(lon>=7)&(lon<=10)
  dist=np.full(len(lat),-np.inf);dist[valid]=tree.query(projected(lat[valid],lon[valid]),k=1)[0]
  distances[int(fid)]=dist
 for i in train:
  fid=int(d['file_id'][i]); a=int(d['source_start'][i]); b=int(d['source_end'][i])
  md=float(np.min(distances[fid][a:b+1]))
  if md>100:
   keep.append(int(i));min_dist=min(min_dist,md)
 return np.asarray(keep,dtype=np.int64),min_dist
def make_trees(test,d,raw):
 global trees; trees={}
 anchors={}
 for i in test:
  fid=int(d['file_id'][i]); anchors.setdefault(fid,[]).append(projected(np.array([d['X_context'][i,0]]),np.array([d['X_context'][i,1]]))[0])
 for fid,pts in anchors.items(): trees[fid]=cKDTree(np.asarray(pts))
def splits(d,raw):
 y=d['y'].astype(np.int64); day=d['day_id']; ix=np.arange(len(y)); final=ix[day==FINAL_DATE]; dev=ix[np.isin(day,DEV_DATES)]
 outer,od=buffer_purge(dev,final,d,raw)
 if set(np.unique(y[outer]))!={0,1} or set(np.unique(y[final]))!={0,1}: raise ValueError('outer pool lacks a class')
 result={'outer_train':outer,'outer_test':final}; folds=[]
 for pos,valday in enumerate(DEV_DATES):
  val=outer[day[outer]==valday]; tr=outer[day[outer]!=valday]
  tr,md=buffer_purge(tr,val,d,raw)
  if set(np.unique(y[tr]))!={0,1} or set(np.unique(y[val]))!={0,1}: raise ValueError(f'date split {valday} lacks class')
  folds.append((tr,val,md))
 # exact random stratified two-fold over outer-train population
 from sklearn.model_selection import StratifiedKFold
 rand=[]
 for a,b in StratifiedKFold(2,shuffle=True,random_state=42).split(outer,y[outer]): rand.append((outer[a],outer[b],float('nan')))
 pools=rand+folds; budget=[int(min(np.sum(y[t]==c) for t,v,_ in pools)) for c in (0,1)]
 return outer,final,pools,budget,od

def sample(train,y,budget,seed):
 rng=np.random.default_rng(seed)
 return np.sort(np.concatenate([rng.choice(train[y[train]==c],int(budget[c]),replace=False) for c in (0,1)]).astype(np.int64))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); ap.add_argument('--run',action='store_true'); a=ap.parse_args()
 d,raw=load(); out=OUT; fail_existing(out); outer,test,pools,budget,outer_md=splits(d,raw); y=d['y'].astype(np.int64)
 pre={'status':'passed','protocol_sha256':sha(PROTOCOL),'features_sha256':sha(FEATURES),'raw_sha256':sha(RAW),'manifest_sha256':sha(MANIFEST),'script_sha256':sha(Path(__file__)),'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'train_rows':len(outer),'test_rows':len(test),'common_inner_per_class_budget':budget,'outer_buffer_min_m':outer_md,'inner_splits':[]}
 fits=[]
 for pos,(tr,va,md) in enumerate(pools):
  fr=sample(tr,y,budget,42+pos); fits.append(fr); pre['inner_splits'].append({'position':pos,'scheme':'row_random_two_fold' if pos<2 else 'date_buffered_two_fold','train_indices':tr.tolist(),'fit_indices':fr.tolist(),'validation_indices':va.tolist(),'buffer_min_m':None if not np.isfinite(md) else md})
 out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out/'split_map.npz',outer_train=outer,outer_test=test,**{f'inner_{i}_train':x[0] for i,x in enumerate(pools)},**{f'inner_{i}_fit':fits[i] for i,x in enumerate(pools)},**{f'inner_{i}_validation':x[1] for i,x in enumerate(pools)})
 pre['split_map_sha256']=sha(out/'split_map.npz')
 pre['outer_train_class_counts']=np.bincount(y[outer],minlength=2).tolist()
 pre['outer_test_class_counts']=np.bincount(y[test],minlength=2).tolist()
 pre['development_rows_before_spatial_purge']=int(np.isin(d['day_id'],DEV_DATES).sum())
 assert outer_md>100 and all(md>100 for _,_,md in pools[2:])
 assert set(d['file_id'][outer]).isdisjoint(d['file_id'][test])
 writej(out/'preflight.json',pre); writej(out/'provenance.json',{'protocol_sha256':sha(PROTOCOL),'features_sha256':sha(FEATURES),'raw_sha256':sha(RAW),'script_sha256':sha(Path(__file__))})
 if a.run: raise NotImplementedError('fit phase requires independent lead review of split_map/preflight before implementation is enabled')
 print(json.dumps({'mode':'preflight','status':'passed','output':str(out)}))

if __name__=='__main__': main()
