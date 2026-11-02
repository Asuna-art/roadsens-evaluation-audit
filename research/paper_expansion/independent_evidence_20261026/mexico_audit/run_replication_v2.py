"""Luna implementation with lead-reviewed fixes. Default preflight; explicit --run fits."""
import argparse, collections, hashlib, json, os, pathlib, platform, sys, zipfile
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import joblib, sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
HERE=pathlib.Path(__file__).resolve().parent
ARCHIVE=HERE.parent.parent/'independent_evidence_20261025/source_audit/applsci-15-10203-s001.zip'
PROTOCOL=HERE/'protocol_v2.md'; OUT=HERE/'mexico_run_v2_frozen'; PREF=OUT/'preflight.json'
PROTOCOL_SHA='6282e30b23601ab7a96e0f046b9b0fa92bdbbc3e4d927ca8029153f44cdf2cc3'
SOURCE_SHA='ecdf5128dd61501c01c7c0baec582d768de0ac3363ea5104aed80749696f4337'
CLASSES=(0,1,2); NAMES={0:'no_event',1:'pothole',2:'speed_bump'}; MODELS=('rf','hgb')
LOCKED=pathlib.Path('C:/Users/Administrator/miniconda3/python.exe')
def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(p): return sha(pathlib.Path(p).read_bytes())
def dump(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def guard():
 if pathlib.Path(sys.executable).resolve()!=LOCKED.resolve():raise RuntimeError('locked interpreter required')
 if file_sha(PROTOCOL)!=PROTOCOL_SHA or file_sha(ARCHIVE)!=SOURCE_SHA:raise RuntimeError('source/protocol hash mismatch')
def features(seg):
 sig=np.column_stack((seg,np.linalg.norm(seg,axis=1))); out=[]
 for a in sig.T:out.extend((float(a.mean()),float(a.std()),float(a.min()),float(a.max()),float(np.sqrt(np.mean(a*a))),float(np.quantile(a,.25)),float(np.quantile(a,.5)),float(np.quantile(a,.75)),float(np.std(np.diff(a)))))
 return out
def model(name):
 e=RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=4) if name=='rf' else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,learning_rate=.1,l2_regularization=1,class_weight='balanced',early_stopping=False,random_state=42)
 return make_pipeline(SimpleImputer(strategy='median'),e)
def score(y,p):
 pr,re,fs,_=precision_recall_fscore_support(y,p,labels=CLASSES,zero_division=0)
 return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=CLASSES,average='macro',zero_division=0)),'per_class':{NAMES[c]:{'precision':float(pr[c]),'recall':float(re[c]),'f1':float(fs[c])} for c in CLASSES},'confusion_matrix':confusion_matrix(y,p,labels=CLASSES).tolist()}
def build_data():
 XX=[]; yy=[]; gg=[]; spans=[]; windows=[]; hashes={}
 with zipfile.ZipFile(ARCHIVE) as z:
  for t in range(1,5):
   name=f'MATLAB/urban_trip_{t}.json';raw=z.read(name);hashes[name]=sha(raw);obj=json.loads(raw)
   A=np.column_stack([obj[k] for k in ('accel_x','accel_y','accel_z','vib_x','vib_y','vib_z')]).astype(float);lab=np.asarray(obj['label'],object)
   if A.shape!=(9000,6) or len(lab)!=9000 or not np.isfinite(A).all():raise RuntimeError('source shape/nonfinite failure')
   for s in range(0,8991,10):
    w=list(map(str,lab[s:s+10]));counts=dict(collections.Counter(w));top=max(counts.values());winners=[k for k,v in counts.items() if v==top];reason=None;cls=None
    if 'sudden braking' in w:reason='contains_sudden_braking'
    elif len(winners)!=1:reason='literal_majority_tie'
    elif top<6:reason='literal_majority_below_6'
    else:
     cls={'':0,'pothole':1,'speed bump':2}.get(winners[0])
     if cls is None:reason='unsupported_label'
    wid=None
    if reason is None:
     wid=len(yy);XX.append(features(A[s:s+10,:3]));yy.append(cls);gg.append(t);spans.append({'window_id':wid,'trip':t,'start':s,'stop_exclusive':s+10,'label':cls})
    windows.append({'window_id':wid,'trip':t,'start':s,'stop_exclusive':s+10,'source_label_counts':counts,'included':reason is None,'exclude_reason':reason,'label':cls})
 return np.asarray(XX,float),np.asarray(yy,int),np.asarray(gg,int),spans,windows,hashes
def check_classes(y,ids):
 if set(y[ids])!=set(CLASSES):raise RuntimeError('class support failure')
def make_folds(X,y,g):
 result=[]
 for oi,(testtr,devtr) in enumerate((((1,2),(3,4)),((3,4),(1,2)))):
  dev=np.flatnonzero(np.isin(g,devtr));test=np.flatnonzero(np.isin(g,testtr));schemes=[]
  specs=(('stratified_random_2fold',StratifiedKFold(2,shuffle=True,random_state=42).split(X[dev],y[dev])),('leave_one_whole_trip_out',LeaveOneGroupOut().split(X[dev],y[dev],g[dev])))
  for si,(name,splits) in enumerate(specs):schemes.append({'scheme_index':si,'name':name,'folds':[{'fold_index':fi,'train':dev[tr].tolist(),'valid':dev[va].tolist()} for fi,(tr,va) in enumerate(splits)]})
  o={'outer_index':oi,'test_trips':list(testtr),'development_trips':list(devtr),'dev':dev.tolist(),'test':test.tolist(),'schemes':schemes}
  for ids in (o['dev'],o['test']):check_classes(y,ids)
  pools=[f['train'] for sc in schemes for f in sc['folds']]
  budget={str(c):int(min(np.sum(y[ids]==c) for ids in pools)) for c in CLASSES};o['per_class_budget']=budget
  if min(budget.values())<1:raise RuntimeError('empty budget')
  for sc in schemes:
   for f in sc['folds']:
    for k in ('train','valid'):check_classes(y,f[k])
    f['budget_train_by_class']={}
    for c in CLASSES:
     ids=sorted(i for i in f['train'] if y[i]==c);seed=260026+1000*oi+100*sc['scheme_index']+10*f['fold_index']+c
     f['budget_train_by_class'][str(c)]=sorted(np.random.default_rng(seed).choice(ids,budget[str(c)],replace=False).tolist())
  o['full_dev_budget_by_class']={}
  for c in CLASSES:
   ids=sorted(i for i in dev if y[i]==c);o['full_dev_budget_by_class'][str(c)]=sorted(np.random.default_rng(260026+1000*oi+900+c).choice(ids,budget[str(c)],replace=False).tolist())
  o['full_dev_budget']=sorted(i for ids in o['full_dev_budget_by_class'].values() for i in ids);result.append(o)
 return result
def preflight():
 guard();X,y,g,spans,windows,members=build_data();folds=make_folds(X,y,g)
 index={'spans':spans,'windows':windows,'labels':y.tolist(),'trips':g.tolist(),'outer_folds':folds}
 d={'protocol_sha256':file_sha(PROTOCOL),'code_sha256':file_sha(__file__),'archive_sha256':file_sha(ARCHIVE),'member_sha256':members,'index_manifest_sha256':sha(json.dumps(index,sort_keys=True,separators=(',',':')).encode()),'included_windows':spans,'all_windows':windows,'features_values':X.tolist(),'feature_matrix_sha256':sha(np.ascontiguousarray(X,dtype='<f8').tobytes()),'labels':y.tolist(),'trips':g.tolist(),'outer_folds':folds,'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__}}
 if OUT.exists():raise FileExistsError('refusing preflight overwrite')
 OUT.mkdir();dump(PREF,d);PREF.with_suffix('.sha256').write_text(file_sha(PREF)+'  preflight.json\n',encoding='ascii')
 print(json.dumps({'included_n':len(y),'trip_counts':{str(t):{NAMES[c]:int(np.sum((g==t)&(y==c))) for c in CLASSES} for t in range(1,5)},'outer_budgets':[o['per_class_budget'] for o in folds],'preflight_sha256':file_sha(PREF),'code_sha256':d['code_sha256']},indent=2))
def execute():
 guard();d=json.loads(PREF.read_text());expected=PREF.with_suffix('.sha256').read_text().split()[0]
 if expected!=file_sha(PREF) or d['code_sha256']!=file_sha(__file__) or d['protocol_sha256']!=file_sha(PROTOCOL) or d['archive_sha256']!=file_sha(ARCHIVE):raise RuntimeError('frozen hash mismatch')
 X,y,g,spans,windows,members=build_data()
 if members!=d['member_sha256'] or spans!=d['included_windows'] or windows!=d['all_windows'] or y.tolist()!=d['labels'] or g.tolist()!=d['trips'] or sha(np.ascontiguousarray(X,dtype='<f8').tobytes())!=d['feature_matrix_sha256']:raise RuntimeError('source reconstruction mismatch')
 run=OUT/'run_v2'
 if run.exists():raise FileExistsError('refusing run overwrite')
 run.mkdir();(run/'preflight.json').write_bytes(PREF.read_bytes());all_inner=[];all_outer=[];selected={};nfits=0
 def fit_save(key,tr,te,metadata):
  check_classes(y,tr);check_classes(y,te)
  try:
   est=model(metadata['candidate']);est.fit(X[tr],y[tr]);pred=est.predict(X[te]);mp=run/(key+'.joblib');joblib.dump(est,mp)
  except Exception as e:dump(run/'fit_failure.json',{'key':key,'error':repr(e),'train_ids':tr,'eval_ids':te});raise
  rec=dict(metadata,key=key,train_ids=tr,eval_ids=te,y_true=y[te].tolist(),pred=pred.tolist(),metrics=score(y[te],pred),model_sha256=file_sha(mp));dump(run/(key+'_predictions.json'),rec);print(key,flush=True);return rec
 for o in d['outer_folds']:
  oi=o['outer_index'];oof={sc['name']:{m:[] for m in MODELS} for sc in o['schemes']};inner_recs=[]
  for sc in o['schemes']:
   for f in sc['folds']:
    tr=sorted(i for ids in f['budget_train_by_class'].values() for i in ids);va=f['valid']
    for m in MODELS:
     key=f'outer{oi}_{sc["name"]}_fold{f["fold_index"]}_{m}';rec=fit_save(key,tr,va,{'candidate':m,'outer_index':oi,'scheme':sc['name'],'fold_index':f['fold_index']});nfits+=1;inner_recs.append(rec);oof[sc['name']][m].append((va,np.asarray(rec['pred'])))
  sels={}
  for sc in o['schemes']:
   nm=sc['name'];scores={}
   for m in MODELS:
    ids=np.concatenate([np.asarray(ix,int) for ix,p in oof[nm][m]])
    if len(np.unique(ids))!=len(ids) or set(ids)!=set(o['dev']):raise RuntimeError('OOF partition failure')
    scores[m]=float(f1_score(np.concatenate([y[ix] for ix,p in oof[nm][m]]),np.concatenate([p for ix,p in oof[nm][m]]),labels=CLASSES,average='macro',zero_division=0))
   sels[nm]={'pooled_inner_macro_f1':scores,'chosen':'rf' if scores['rf']>=scores['hgb'] else 'hgb'}
  dump(run/f'outer{oi}_selections.json',{'outer_index':oi,'selector_results':sels,'inner_records':inner_recs});all_inner.extend(inner_recs);outer_records={};selected[str(oi)]={}
  for target,tr in (('full',o['dev']),('commonbudget',o['full_dev_budget'])):
   for m in MODELS:
    key=f'outer{oi}_{m}_{target}';rec=fit_save(key,sorted(tr),o['test'],{'candidate':m,'outer_index':oi,'target':target});nfits+=1;all_outer.append(rec);outer_records[target,m]=rec
  for nm,sel in sels.items():
   m=sel['chosen'];iv=sel['pooled_inner_macro_f1'][m];selected[str(oi)][nm]={}
   for target in ('full','commonbudget'):
    ov=outer_records[target,m]['metrics']['macro_f1'];selected[str(oi)][nm][target]={'selected_candidate':m,'inner_macro_f1':iv,'outer_macro_f1':ov,'signed_inner_minus_outer':iv-ov,'absolute_error':abs(iv-ov)}
 comparisons={}
 for target in ('full','commonbudget'):
  differences=[]
  for oi in ('0','1'):
   tr=selected[oi]['leave_one_whole_trip_out'][target];rd=selected[oi]['stratified_random_2fold'][target];differences.append({'outer_index':int(oi),'trip_minus_random_outer_f1':tr['outer_macro_f1']-rd['outer_macro_f1'],'trip_minus_random_absolute_error':tr['absolute_error']-rd['absolute_error']})
  comparisons[target]={'route_differences':differences,'arithmetic_mean':{k:float(np.mean([r[k] for r in differences])) for k in ('trip_minus_random_outer_f1','trip_minus_random_absolute_error')}}
 if nfits!=24:raise RuntimeError('fit count mismatch')
 dump(run/'results.json',{'protocol_sha256':d['protocol_sha256'],'code_sha256':d['code_sha256'],'archive_sha256':d['archive_sha256'],'preflight_sha256':file_sha(PREF),'n_fits':nfits,'selected_results_by_route_and_selector':selected,'paired_comparisons':comparisons,'all_inner_fit_records':all_inner,'all_outer_candidate_records':all_outer});print('COMPLETED',nfits)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args();execute() if args.run else preflight()
