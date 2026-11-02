import sys,json,hashlib,warnings,joblib,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(r'D:\三区灌水专用\research_20261015'); PILOT=ROOT/'candidate_roadsens'; EXP=PILOT/'paper_expansion'; HERE=EXP/'key_experiments'; AX=PILOT/'runs'/'axis_window_v1'; OUT=HERE/'T_temporal_v1'
sys.path.insert(0,str(EXP)); import runner as r
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
cache,fmap,qa=r.validate_immutable_inputs(); pilot=r._load_modules()[0]
axis=loadmod('axis_builder_runtime_audit',PILOT/'run_axis_window_control.py')
sessions,names=axis.build_axis_features(cache['sessions'],pilot)
manifest=json.loads((AX/'manifest.json').read_text(encoding='utf-8'))
records=[]; all_imputation_match=True; params_match=True; versions=set()
for policy in ('primary','blank_normal'):
 ps=sessions if policy=='primary' else [{**s,'labels':s['labels_blank_normal']} for s in sessions]
 x,y,groups,owners=pilot._concat(ps,'axis_window')
 for fold in range(1,6):
  stem=f'{policy}_group.axis_window.fold{fold}'
  cp_path=AX/f'{stem}.checkpoint.json'; model_path=AX/f'{stem}.joblib'
  cp=json.loads(cp_path.read_text(encoding='utf-8'))
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',InconsistentVersionWarning)
   model=joblib.load(model_path)
  iv=[str(w.message) for w in caught if issubclass(w.category,InconsistentVersionWarning)]
  orig=None
  for w in caught:
   if issubclass(w.category,InconsistentVersionWarning): orig=getattr(w.message,'original_sklearn_version',None)
  versions.add(orig or 'not_reported')
  params=model.get_params(deep=False)
  locked=manifest['rf_config']
  p_ok=all(params.get(k)==v for k,v in locked.items())
  train_ids=np.asarray([z['row'] for z in cp['fit_train_rows']],dtype=np.int64)
  test_ids=np.asarray([z['row'] for z in cp['test_rows']],dtype=np.int64)
  _,_,kept,medians=pilot._impute_train_test(x[train_ids],x[test_ids])
  old_kept=np.asarray(cp['kept_feature_indices'],dtype=np.int64)
  old_med=np.asarray(cp['training_imputation_medians'],dtype=np.float64)
  k_match=np.array_equal(kept,old_kept)
  m_match=np.array_equal(medians,old_med,equal_nan=True)
  all_imputation_match &= k_match and m_match; params_match &= p_ok
  records.append({'policy':policy,'fold':fold,'artifact_sha256':sha(model_path),'checkpoint_sha256':sha(cp_path),'estimator_class':type(model).__name__,'serialized_sklearn_version':orig,'inconsistent_version_warning':bool(iv),'model_parameters':params,'parameters_match_manifest':p_ok,'fit_row_count':len(train_ids),'test_row_count':len(test_ids),'kept_features_match_recomputed':k_match,'median_array_match_recomputed_exactly':m_match,'kept_feature_count':len(kept),'median_count':len(medians)})
result={'status':'historical_axis_window_runtime_audit_complete','fit_free':True,'models_loaded_for_metadata_only':10,'no_models_fit':True,'current_sklearn_version':sklearn.__version__,'serialized_sklearn_versions':sorted(versions),'historical_models_all_from_current_runtime':versions=={sklearn.__version__},'all_ten_have_version_mismatch_warning':all(x['inconsistent_version_warning'] for x in records),'all_rf_parameters_match_frozen_manifest':params_match,'all_current_parent_imputation_medians_match_historical_checkpoints_exactly':all_imputation_match,'feature_schema_count':len(names),'cache_sha256':sha(r.CACHE),'axis_manifest_sha256':sha(AX/'manifest.json'),'run_axis_window_code_sha256':sha(PILOT/'run_axis_window_control.py'),'records':records}
(OUT/'runtime_audit_axis_window_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md=['# Historical axis-window model runtime audit','',f"Status: **{result['status']}**. This was a fit-free audit; the ten existing serialized estimators were loaded only to inspect metadata. No models were fit and no predictions or scores were recomputed.",'',f"All {len(records)} artifacts report serialized scikit-learn version(s) {', '.join(sorted(versions))}; current locked runtime is scikit-learn {sklearn.__version__}. Version mismatch warnings were captured without printing other artifact warnings. All ten model parameter sets match the frozen RF configuration: **{params_match}**.",'',f"The historical checkpoint kept-feature indices and training medians were recomputed from current cached axis-window features using the parent training-only imputer and each checkpoint's recorded fit rows. All ten kept-feature arrays and median arrays match exactly: **{all_imputation_match}**. This indicates the observed historical mismatch is attributable to estimator runtime provenance rather than a detected change in feature rows or imputation.",'',"The old axis-window RF outputs should therefore remain a historical comparator. They are not a same-runtime controlled reference for the temporal experiment under scikit-learn 1.9.0. Any re-run comparator must use a separately versioned output directory and preserve these artifacts unchanged.",'',f"Provenance: axis manifest SHA-256 `{result['axis_manifest_sha256']}`; axis builder SHA-256 `{result['run_axis_window_code_sha256']}`; cache SHA-256 `{result['cache_sha256']}`.",'',"Per-artifact version, estimator parameters, hashes, and exact imputation comparisons are in `runtime_audit_axis_window_v1.json`."]
(OUT/'runtime_audit_axis_window_v1.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','current_sklearn_version','serialized_sklearn_versions','all_ten_have_version_mismatch_warning','all_rf_parameters_match_frozen_manifest','all_current_parent_imputation_medians_match_historical_checkpoints_exactly')},indent=2))
