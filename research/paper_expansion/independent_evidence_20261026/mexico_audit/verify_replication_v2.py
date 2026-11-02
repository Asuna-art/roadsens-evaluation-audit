"""Independent raw reconstruction and saved-model audit; no execution-code import or refitting.
Luna verification design, corrected against concrete saved schema by lead.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import collections,hashlib,json,pathlib,zipfile
import numpy as np,joblib
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support
H=pathlib.Path(__file__).resolve().parent;D=H/'mexico_run_v2_frozen';R=D/'run_v2'
A=H.parent.parent/'independent_evidence_20261025/source_audit/applsci-15-10203-s001.zip'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):assert np.allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def f1(y,p):return float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0))
p=load(D/'preflight.json');s=load(R/'results.json')
assert digest(A)==p['archive_sha256']=='ecdf5128dd61501c01c7c0baec582d768de0ac3363ea5104aed80749696f4337'
assert digest(H/'protocol_v2.md')==p['protocol_sha256']=='6282e30b23601ab7a96e0f046b9b0fa92bdbbc3e4d927ca8029153f44cdf2cc3'
assert digest(H/'run_replication_v2.py')==p['code_sha256']
assert digest(D/'preflight.json')==(D/'preflight.sha256').read_text().split()[0]==s['preflight_sha256']
spans=[];decisions=[];vectors=[];labels=[];groups=[]
with zipfile.ZipFile(A) as z:
 for t in range(1,5):
  name=f'MATLAB/urban_trip_{t}.json';b=z.read(name);assert hashlib.sha256(b).hexdigest()==p['member_sha256'][name];o=json.loads(b)
  a=np.array([o[k] for k in ('accel_x','accel_y','accel_z','vib_x','vib_y','vib_z')],float).T;assert a.shape==(9000,6) and np.isfinite(a).all()
  for start in range(0,9000,10):
   c=collections.Counter(o['label'][start:start+10]);largest=max(c.values());win=[k for k,v in c.items() if v==largest];reason=None;lab=None
   if 'sudden braking' in c:reason='contains_sudden_braking'
   elif len(win)>1:reason='literal_majority_tie'
   elif largest<6:reason='literal_majority_below_6'
   else:lab={'':0,'pothole':1,'speed bump':2}[win[0]]
   row={'window_id':None,'trip':t,'start':start,'stop_exclusive':start+10,'source_label_counts':dict(c),'included':reason is None,'exclude_reason':reason,'label':lab}
   if reason is None:
    row['window_id']=len(labels);spans.append({k:row[k] for k in ('window_id','trip','start','stop_exclusive','label')});labels.append(lab);groups.append(t)
    a3=a[start:start+10,:3];signals=[a3[:,i] for i in range(3)]+[np.sqrt((a3*a3).sum(1))];v=[]
    for x in signals:v.extend([np.average(x),np.sqrt(np.mean((x-x.mean())**2)),min(x),max(x),np.linalg.norm(x)/np.sqrt(10),*np.percentile(x,[25,50,75]),np.std(np.diff(x))])
    vectors.append(v)
   decisions.append(row)
assert spans==p['included_windows'] and decisions==p['all_windows']
y=np.array(labels);g=np.array(groups);X=np.array(p['features_values']);near(X,np.array(vectors));assert X.shape==(2504,36)
assert labels==p['labels'] and groups==p['trips'];assert collections.Counter(groups)=={1:598,2:601,3:631,4:674}
assert hashlib.sha256(np.ascontiguousarray(X,dtype='<f8').tobytes()).hexdigest()==p['feature_matrix_sha256']
index={'spans':spans,'windows':decisions,'labels':labels,'trips':groups,'outer_folds':p['outer_folds']}
assert hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['index_manifest_sha256']
expected={}
for outer in p['outer_folds']:
 oi=outer['outer_index'];dev=outer['dev'];test=outer['test'];assert not set(dev)&set(test) and set(dev)|set(test)==set(range(2504))
 assert set(g[test])==set(outer['test_trips']) and set(g[dev])==set(outer['development_trips']);assert set(outer['test_trips'])==({1,2} if oi==0 else {3,4})
 all_train=[f['train'] for sc in outer['schemes'] for f in sc['folds']];budget={str(c):min(int(sum(y[i]==c for i in ids)) for ids in all_train) for c in (0,1,2)};assert budget==outer['per_class_budget']
 for sc in outer['schemes']:
  covered=[]
  for fold in sc['folds']:
   tr=fold['train'];va=fold['valid'];assert not set(tr)&set(va) and set(tr)|set(va)==set(dev);covered+=va
   if sc['scheme_index']==1:assert set(g[tr]).isdisjoint(g[va]) and len(set(g[va]))==1
   actual=sorted(i for ids in fold['budget_train_by_class'].values() for i in ids)
   assert len(actual)==len(set(actual)) and set(actual)<=set(tr)
   assert {str(c):int(sum(y[actual]==c)) for c in (0,1,2)}==budget
   for m in ('rf','hgb'):expected[f'outer{oi}_{sc["name"]}_fold{fold["fold_index"]}_{m}']=(actual,va)
  assert len(covered)==len(set(covered)) and set(covered)==set(dev)
 assert set(outer['full_dev_budget'])<=set(dev) and {str(c):int(sum(y[outer['full_dev_budget']]==c)) for c in (0,1,2)}==budget
 for target,tr in (('full',dev),('commonbudget',outer['full_dev_budget'])):
  for m in ('rf','hgb'):expected[f'outer{oi}_{m}_{target}']=(sorted(tr),test)
models={q.stem:q for q in R.glob('*.joblib')};records={load(q)['key']:load(q) for q in R.glob('*_predictions.json')};assert set(models)==set(records)==set(expected) and len(models)==24
verified=[];oof=collections.defaultdict(list);outer_f1={}
for key,rec in records.items():
 tr,ev=expected[key];assert tr==rec['train_ids'] and ev==rec['eval_ids'];assert digest(models[key])==rec['model_sha256'];est=joblib.load(models[key]);imp=est.steps[0][1];clf=est.steps[-1][1];near(imp.statistics_,np.median(X[tr],axis=0));param=clf.get_params();assert param['class_weight']=='balanced' and param['random_state']==42
 if rec['candidate']=='rf':assert (param['n_estimators'],param['max_depth'],param['min_samples_leaf'],param['max_features'])==(100,20,5,'sqrt')
 else:assert (param['max_iter'],param['max_leaf_nodes'],param['learning_rate'],param['l2_regularization'],param['early_stopping'])==(100,31,.1,1,False)
 pred=est.predict(X[ev]);assert pred.tolist()==rec['pred'] and y[ev].tolist()==rec['y_true'];metric=rec['metrics'];near(metric['macro_f1'],f1(y[ev],pred));near(metric['accuracy'],accuracy_score(y[ev],pred));near(metric['balanced_accuracy'],balanced_accuracy_score(y[ev],pred));assert metric['confusion_matrix']==confusion_matrix(y[ev],pred,labels=[0,1,2]).tolist()
 pr,re,fs,_=precision_recall_fscore_support(y[ev],pred,labels=[0,1,2],zero_division=0)
 for c,n in enumerate(('no_event','pothole','speed_bump')):
  for k,val in (('precision',pr[c]),('recall',re[c]),('f1',fs[c])):near(metric['per_class'][n][k],val)
 if 'scheme' in rec:oof[rec['outer_index'],rec['scheme'],rec['candidate']].append((ev,rec['pred']))
 else:outer_f1[rec['outer_index'],rec['target'],rec['candidate']]=metric['macro_f1']
 verified.append({'key':key,'model_sha256':rec['model_sha256'],'n_train':len(tr),'n_evaluation':len(ev),'macro_f1':metric['macro_f1']})
assert len(s['all_inner_fit_records'])==16 and len(s['all_outer_candidate_records'])==8
for rec in s['all_inner_fit_records']+s['all_outer_candidate_records']:assert rec==records[rec['key']]
calculated={}
for oi in (0,1):
 sel=load(R/f'outer{oi}_selections.json')['selector_results'];calculated[str(oi)]={}
 for scheme in ('stratified_random_2fold','leave_one_whole_trip_out'):
  scores={}
  for m in ('rf','hgb'):
   pool=oof[oi,scheme,m];ix=sum((a for a,b in pool),[]);pp=sum((b for a,b in pool),[]);assert len(ix)==len(set(ix)) and set(ix)==set(p['outer_folds'][oi]['dev']);scores[m]=f1(y[ix],pp)
  assert scores==sel[scheme]['pooled_inner_macro_f1'];chosen='rf' if scores['rf']>=scores['hgb'] else 'hgb';assert chosen==sel[scheme]['chosen'];calculated[str(oi)][scheme]={}
  for target in ('full','commonbudget'):
   iv=scores[chosen];ov=outer_f1[oi,target,chosen];out={'selected_candidate':chosen,'inner_macro_f1':iv,'outer_macro_f1':ov,'signed_inner_minus_outer':iv-ov,'absolute_error':abs(iv-ov)};assert out==s['selected_results_by_route_and_selector'][str(oi)][scheme][target];calculated[str(oi)][scheme][target]=out
for target in ('full','commonbudget'):
 diffs=[]
 for oi in (0,1):
  tr=calculated[str(oi)]['leave_one_whole_trip_out'][target];rd=calculated[str(oi)]['stratified_random_2fold'][target];diffs.append({'outer_index':oi,'trip_minus_random_outer_f1':tr['outer_macro_f1']-rd['outer_macro_f1'],'trip_minus_random_absolute_error':tr['absolute_error']-rd['absolute_error']})
 assert diffs==s['paired_comparisons'][target]['route_differences']
 for k in ('trip_minus_random_outer_f1','trip_minus_random_absolute_error'):near(s['paired_comparisons'][target]['arithmetic_mean'][k],np.mean([v[k] for v in diffs]))
report={'status':'passed','verified_model_count':len(verified),'raw_windows_verified':len(decisions),'included_windows':len(spans),'source_sha256':digest(A),'results_sha256':digest(R/'results.json'),'preflight_sha256':digest(D/'preflight.json'),'checks':['independent raw labels and 3600 inclusion decisions','independent 36-feature reconstruction within1e-10','exact saved split partitions and route/trip isolation','per-class minima and actual fit indices','24 model hashes and all reloaded predictions','all training-only imputation medians','fixed candidate hyperparameters','all metrics, candidate selections and paired summaries'],'selected_results':calculated,'model_checks':verified}
(H/'independent_verification_v2.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k] for k in ('status','verified_model_count','raw_windows_verified','included_windows','results_sha256')},indent=2))
