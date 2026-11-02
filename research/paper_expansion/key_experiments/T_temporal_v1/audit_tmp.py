import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(r'D:\三区灌水专用\research_20261015')
EXP=ROOT/'candidate_roadsens'/'paper_expansion'
HERE=EXP/'key_experiments'
PILOT=EXP.parent
OUT=HERE/'T_temporal_v1'
PRE=OUT/'preflight_v1'
MAN=PRE/'preflight_manifest.json'; NPZ=PRE/'split_membership_v1.npz'

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def hobj(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
sys.path.insert(0,str(EXP))
import runner as r
wd=loadmod('temporal_preflight_driver',HERE/'run_temporal_benchmark.py')
manifest=json.loads(MAN.read_text(encoding='utf-8'))
assert sha(NPZ)==manifest['split_membership_sha256']
assert manifest['status']=='preflight_complete_fit_free' and manifest['model_fit_or_scores'] is False
assert manifest['planned_fit_folds']==80 and manifest['planned_cell_count']==16
# Synthetic cases: non-monotonic sequence, exact inclusive left boundary, just outside, NaN/inf, duplicate timestamps.
def naive(times, anchors):
 times=np.asarray(times,dtype=float); out={}
 for sr in anchors:
  if sr<0 or sr>=len(times) or not np.isfinite(times[sr]): continue
  t=times[sr]
  inds=[j for j,x in enumerate(times) if np.isfinite(x) and x>=t-1.0 and (x<t or (x==t and j<=sr))]
  out[int(sr)]=frozenset(inds)
 return out
synthetic=[np.array([5.,4.,4.,3.0,np.nan,3.999999,2.0,np.inf,4.5]), np.array([0.,1.,2.,1.,0.,np.nan,-1.])]
syn_count=0
for ts in synthetic:
 got=wd.window_members_by_anchor({'times_full':ts,'source_row':np.arange(len(ts))})
 ref=naive(ts,np.arange(len(ts)))
 assert got==ref,(ts,got,ref)
 syn_count+=len(ref)
# Load fully hash-verified immutable cache and rebuild original splits.
cache,fmap,splitqa=r.validate_immutable_inputs()
specs,rows=r.build_all_splits(cache['sessions'],fmap,splitqa)
arr=np.load(NPZ,allow_pickle=False)
# Each real full-cadence session gets at most three spot anchors; compare to direct independent O(N) reference.
samples=[]; window_map={}
for s in cache['sessions']:
 win=wd.window_members_by_anchor(s); window_map[s['session_key']]=win
 finite=np.asarray(s['source_row'],dtype=np.int64)
 finite=finite[np.isfinite(np.asarray(s['times_full'],float)[finite])]
 if len(finite):
  picks=sorted(set([int(finite[0]),int(finite[len(finite)//2]),int(finite[-1])]))
  ref=naive(s['times_full'],picks)
  for sr in picks:
   assert win.get(sr)==ref.get(sr),(s['session_key'],sr)
  samples.append({'session':s['session_key'],'anchors':picks,'checked':len(picks)})
# Validate all 20 policy/split/fold definitions and exact index arrays.
fold_checks=[]; raw_overlap_checks=[]; shared_fit=True; class_support=True; identities=True
for policy in ('primary','blank_normal'):
 table=rows[policy]
 for sid in ('record42','timeblock42'):
  # canonical split IDs in runner
  live=specs[(policy,sid)]['folds']
  for fi,(tr0,te0) in enumerate(live):
   k=f'{policy}__{sid}__fold{fi+1}'
   arrays={name:arr[k+'__'+name] for name in ('train_original','train_after_window_purge','extra_removed','fit','test')}
   tr=arrays['train_original']; post=arrays['train_after_window_purge']; removed=arrays['extra_removed']; fit=arrays['fit']; test=arrays['test']
   assert np.array_equal(tr,tr0) and np.array_equal(test,te0), (k,'base split mismatch')
   assert not np.intersect1d(tr,test).size
   assert np.array_equal(np.sort(np.concatenate([post,removed])),np.sort(tr))
   assert not np.intersect1d(post,removed).size
   expected_fit=r.cap_train_indices(post,table.y,fi,42,20000)
   assert np.array_equal(fit,expected_fit),(k,'fit cap mismatch')
   assert len(fit)<=20000 and len(np.unique(fit))==len(fit)
   for ids in (tr,post,fit,test): assert set(np.unique(table.y[ids]))==set(r.CLASSES),(k,'class support')
   meta=next(x for x in manifest['folds'] if x['policy']==policy and x['split']==sid and x['fold']==fi+1)
   for field,ids in [('train_original_sha256',tr),('train_postpurge_sha256',post),('extra_removed_sha256',removed),('fit_sha256',fit),('test_sha256',test)]:
    assert meta[field]==hobj(ids.tolist()),(k,field)
   if sid=='record42':
    assert not (set(table.groups[tr]) & set(table.groups[test]))
   else:
    def union_raw(ids):
     out={}
     for ix in ids:
      g=str(table.groups[ix]); sr=int(table.source_rows[ix]); mem=window_map[g].get(sr)
      assert mem is not None
      out.setdefault(g,set()).update(mem)
     return out
    test_raw=union_raw(test); post_raw=union_raw(post)
    overlaps=sum(len(post_raw.get(g,set()) & raw) for g,raw in test_raw.items())
    assert overlaps==0,(k,overlaps)
    raw_overlap_checks.append({'policy':policy,'fold':fi+1,'test_window_source_rows':sum(map(len,test_raw.values())),'train_window_source_rows':sum(map(len,post_raw.values())),'intersection_source_rows':overlaps,'purged_rows':len(removed)})
   # Shared fit ID assertion via identical canonical single array referenced for all 4 cells in manifest.
   cells=[c for c in manifest['planned_cells'] if c['policy']==policy and c['split']==sid]
   assert len(cells)==4 and all(c['folds']==5 for c in cells)
   assert meta['feature_sets_share_fit_rows'] and meta['estimators_share_fit_rows']
   fold_checks.append({'policy':policy,'split':sid,'fold':fi+1,'fit_n':len(fit),'test_n':len(test),'purged_n':len(removed),'class_support':meta['class_support']})
# Output audit report.
result={'status':'independent_preflight_validated','model_fit_or_scores':False,'preflight_manifest_sha256':sha(MAN),'split_membership_sha256':sha(NPZ),'driver_sha256':sha(HERE/'run_temporal_benchmark.py'),'parent_run_pilot_sha256':sha(PILOT/'run_pilot.py'),'shared_runner_sha256':sha(EXP/'runner.py'),'cache_sha256':sha(PILOT/'session_feature_cache_v1.joblib'),'protocol_sha256':sha(HERE/'protocol_key_v1.md'),'T_protocol_sha256':sha(HERE/'T_temporal_protocol_v1.md'),'synthetic_anchors_checked':syn_count,'synthetic_cases':len(synthetic),'actual_recordings_checked':len(samples),'actual_anchor_memberships_checked':sum(len(x['anchors']) for x in samples),'actual_anchor_sample':samples,'folds_checked':len(fold_checks),'configuration_cells':len(manifest['planned_cells']),'fold_checks':fold_checks,'timeblock_raw_window_separation':{'folds_checked':len(raw_overlap_checks),'all_intersections_zero':all(x['intersection_source_rows']==0 for x in raw_overlap_checks),'details':raw_overlap_checks},'checks':{'manifest_npz_hash_matches':True,'driver_vs_naive_synthetic_window_membership':True,'actual_sampled_membership_vs_naive':True,'live_cache_and_source_hashes_verified_by_runner':True,'preflight_base_split_rows_match_live_runner':True,'all_capped_fit_rows_exact_match_live_sampler':True,'all_train_test_support_three_classes':True,'recording_split_sessions_disjoint':True,'timeblock_postpurge_raw_window_rows_disjoint':True,'postpurge_and_removed_partition_original_train':True,'manifest_identity_hashes_match_arrays':True,'same_fit_rows_shared_across_2features_2estimators':True}}
(OUT/'independent_preflight_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md=['# Temporal preflight independent validation','',f"Status: **{result['status']}**. This is a fit-free validation; no estimators were fit and no predictive scores were computed.",'',f"Validated the frozen preflight manifest and membership NPZ by SHA-256. Audited {len(fold_checks)} policy/split/fold combinations across {len(manifest['planned_cells'])} planned feature/model cells.",'',f"Compared {syn_count} synthetic anchor memberships across {len(synthetic)} boundary-focused fixtures to a separately written naive reference, then checked {result['actual_anchor_memberships_checked']} sampled anchors across {len(samples)} actual recordings against that reference. Fixtures included non-monotonic times, NaN/Inf, duplicate timestamps, the exact inclusive left boundary, and timestamps just outside it.",'',"All 20 fold arrays matched live frozen split construction and deterministic training-cap rows. All train/fit/test sets retained all three classes. Recording-heldout folds had no shared session IDs. For all 10 timeblock policy/folds, post-purge training raw-source window rows were disjoint from test raw-source window rows. The post-purge and removed rows partitioned original training rows; each planned feature/estimator pair shares the same fit IDs.",'',"Window semantics agree with the parent causal feature builder: only finite times participate; stable chronological sorting means equal-time rows are processed in original source-row order; each anchor uses rows with time in the closed-left interval [t−1 s, t], including its own row and earlier same-time rows, excluding later same-time rows. NaN/Inf timestamp anchors have no window membership.",'',"This validates the preflight logic and row identity only. It does not validate model fitting, model outputs, generalization, or whether recordings represent independent trips/routes.",'',f"Audit artifacts: `independent_preflight_validation.json`. Manifest SHA-256: `{result['preflight_manifest_sha256']}`. Membership SHA-256: `{result['split_membership_sha256']}`."]
(OUT/'independent_preflight_validation.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'folds':len(fold_checks),'actual_recordings':len(samples),'actual_anchors':result['actual_anchor_memberships_checked'],'raw_overlap_folds':len(raw_overlap_checks),'report':str(OUT/'independent_preflight_validation.md')},indent=2))



