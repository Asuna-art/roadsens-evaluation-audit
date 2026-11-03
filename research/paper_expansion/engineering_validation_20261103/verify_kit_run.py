#!/usr/bin/env python
"""Independent, read-only audit of the frozen KIT run; never fits a model."""
from __future__ import annotations
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score)

BASE=Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion")
HERE=BASE/"engineering_validation_20261103"; RUN=HERE/"run_kit_v1"
FEATURES=HERE/"features_kit_v1.npz"; SPLIT=RUN/"split_map.npz"
RESULTS=RUN/"results.json"; RECORDS=RUN/"fit_records.json"; SELECT=RUN/"selection.json"
VERIFY=RUN/"kit_run_verification.json"
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def writej(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf8')
def metrics(y,p):
 labs=np.array([0,1]); cm=confusion_matrix(y,p,labels=labs); recalls=recall_score(y,p,labels=labs,average=None,zero_division=0)
 return {'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p)),
  'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_precision':float(precision_score(y,p,average='macro',zero_division=0)),
  'macro_recall':float(recall_score(y,p,average='macro',zero_division=0)),'class_labels':[0,1],
  'class_recalls':recalls.astype(float).tolist(),'confusion_matrix_rows_true_columns_pred':cm.astype(int).tolist()}
def same(a,b): return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
def main():
 start=datetime.now(timezone.utc).isoformat(); findings=[]
 with np.load(FEATURES,allow_pickle=False) as z: y=z['y'].astype(np.int64); data={k:z[k] for k in z.files}
 with np.load(SPLIT,allow_pickle=False) as z: sp={k:z[k].astype(np.int64) for k in z.files}
 result=json.loads(RESULTS.read_text(encoding='utf8')); records=json.loads(RECORDS.read_text(encoding='utf8'))['records']; selection=json.loads(SELECT.read_text(encoding='utf8'))
 assert len(records)==24 and result['fit_count']==24
 checks=[]; oof={i:{} for i in range(4)}; candidates=['rf_sensor','rf_sensor_context','hgb_sensor','hgb_sensor_context']
 for r in records:
  fid=r['fit_id']; pp=Path(r['prediction_path']);
  with np.load(pp,allow_pickle=False) as p: ix=p['test_indices']; yt=p['y_true']; yp=p['y_pred']; fit=p['train_indices_fit']
  exact_y=np.array_equal(yt,y[ix]); exact_fit=np.array_equal(fit,np.asarray(r['train_indices_fit'],dtype=np.int64)); exact_test=np.array_equal(ix,np.asarray(r['test_indices'],dtype=np.int64))
  calc=metrics(yt,yp); metric_ok=same(calc,r['metrics'])
  if fid.startswith('inner_'):
   _,si,cand=fid.split('_',2); si=int(si)
   exact_split=np.array_equal(fit,sp[f'inner_{si}_fit']) and np.array_equal(ix,sp[f'inner_{si}_validation'])
   oof[si][cand]=(ix,yp)
  else:
   parts=fid.split('_'); target='_'.join(parts[1:-2]); cand='_'.join(parts[-2:])
   # Candidate ids themselves contain underscores; parse known prefix.
   target=next((t for t in ('full_development','common_inner_budget') if fid.startswith('outer_'+t+'_')),None)
   cand=fid[len('outer_'+target+'_'):] if target else ''
   if target=='full_development': exact_split=np.array_equal(fit,sp['outer_train'])
   elif target=='common_inner_budget':
    budget=[1613,17]; rng=np.random.default_rng(46)
    expected=np.sort(np.concatenate([rng.choice(sp['outer_train'][y[sp['outer_train']]==c],budget[c],replace=False) for c in (0,1)]))
    exact_split=np.array_equal(fit,expected)
   else: exact_split=False
  passed=all((exact_y,exact_fit,exact_test,metric_ok,exact_split))
  checks.append({'fit_id':fid,'prediction_sha256':sha(pp),'y_true_exact':exact_y,'fit_indices_exact':exact_fit,'test_indices_exact':exact_test,'metrics_recomputed_exact':metric_ok,'split_membership_exact':exact_split,'passed':passed})
  if not passed: findings.append(f'fit verification failed: {fid}')
 # Recalculate pooled OOF selections and compare frozen selection/results.
 selected={}
 for name,inds in [('row_random_two_fold',(0,1)),('date_buffered_two_fold',(2,3))]:
  scores={}
  for cand in candidates:
   ix=np.concatenate([oof[i][cand][0] for i in inds]); pred=np.concatenate([oof[i][cand][1] for i in inds]); order=np.argsort(ix,kind='stable')
   scores[cand]=metrics(y[ix[order]],pred[order])['macro_f1']
  pick=max(candidates,key=lambda c:scores[c]); selected[name]={'scores':scores,'selected':pick}
  frozen=selection['selectors'][name]
  if frozen['selected_candidate_id']!=pick or not all(np.isclose(frozen['pooled_inner_oof_macro_f1'][k],scores[k],rtol=0,atol=1e-14) for k in candidates): findings.append(f'selection mismatch: {name}')
 # Selection must precede all outer prediction files; inner predictions precede selection.
 sm=SELECT.stat().st_mtime_ns
 outer_times=[Path(r['prediction_path']).stat().st_mtime_ns for r in records if r['phase']!='inner']
 inner_times=[Path(r['prediction_path']).stat().st_mtime_ns for r in records if r['phase']=='inner']
 selection_order=bool(max(inner_times)<sm<min(outer_times))
 if not selection_order: findings.append('selection/prediction modification-time ordering failed')
 # Check two saved models without fitting; matrix semantics from audited frozen driver.
 import importlib.util
 driver=HERE/'kit_fit_frozen.py'; spec=importlib.util.spec_from_file_location('kit_fit_frozen_audit',driver); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 hist_spec=importlib.util.spec_from_file_location('kit_hist_audit',BASE/'external_pvs_v1'/'run_nested_evaluation.py'); hist=importlib.util.module_from_spec(hist_spec); hist_spec.loader.exec_module(hist)
 reloads=[]
 for fid in ('inner_0_rf_sensor','outer_full_development_hgb_sensor'):
  r=next(x for x in records if x['fit_id']==fid); cand=r['candidate']; model=joblib.load(Path(r['model_path']))
  with np.load(Path(r['prediction_path']),allow_pickle=False) as p: ix=p['test_indices']; saved=p['y_pred']
  modeldata={k:v for k,v in data.items()}; modeldata['_sensor_indices']=np.arange(30); modeldata['_context_indices']=np.arange(3)
  actual=model.predict(hist.matrix_for(cand,ix,modeldata)).astype(np.int64)
  ok=np.array_equal(saved,actual); reloads.append({'fit_id':fid,'reload_prediction_exact':ok})
  if not ok: findings.append(f'reload mismatch: {fid}')
 # Hash result, all saved predictions, fit driver and verifier source. Does not alter run artifacts.
 pred_hashes={Path(r['prediction_path']).relative_to(RUN).as_posix():sha(Path(r['prediction_path'])) for r in records}
 report={'status':'passed' if not findings else 'failed','started_utc':start,'completed_utc':datetime.now(timezone.utc).isoformat(),
  'fit_count':len(records),'fit_checks':checks,'selection_recomputed':selected,'selection_before_outer_predictions_and_after_inner':selection_order,
  'reload_checks':reloads,'hashes':{'results_sha256':sha(RESULTS),'fit_records_sha256':sha(RECORDS),'selection_sha256':sha(SELECT),
  'prediction_sha256':pred_hashes,'fit_driver_sha256':sha(driver),'verifier_sha256':sha(Path(__file__))},'findings':findings}
 writej(VERIFY,report)
 print(json.dumps({'status':report['status'],'fits':len(records),'findings':findings,'report':str(VERIFY)},ensure_ascii=False))
 if findings: raise SystemExit(1)
if __name__=='__main__': main()
