"""Independent raw-source checks before the external PVS model matrix."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SENSOR = [f'{kind}_{axis}_dashboard' for kind in ('acc','gyro') for axis in ('x','y','z')]
CONTEXT = ['latitude','longitude','speed']
LABEL = ['no_speed_bump','speed_bump_asphalt','speed_bump_cobblestone']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    manifest = json.loads((HERE/'feature_manifest.json').read_text(encoding='utf-8'))
    with np.load(HERE/'features.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    feature_names = manifest.get('sensor_feature_names')
    assert feature_names and len(feature_names)==30, 'Manifest must declare all 30 feature names'
    results=[]
    total_checked=0
    for trip in range(1,10):
        sensor_path=HERE/'data'/f'PVS_{trip}'/'dataset_gps_mpu_left.csv'
        label_path=sensor_path.with_name('dataset_labels.csv')
        df=pd.read_csv(sensor_path,usecols=['timestamp']+SENSOR+CONTEXT)
        labels=pd.read_csv(label_path,usecols=LABEL).to_numpy()
        assert len(df)==len(labels)
        assert np.isin(labels,[0,1]).all()
        label_valid=labels.sum(axis=1)==1
        raw=df[SENSOR].to_numpy(dtype=np.float64)
        ts=df['timestamp'].to_numpy(dtype=np.float64)
        assert np.isfinite(ts).all() and (np.diff(ts)>=0).all()
        anchors=np.arange(99,len(df),25,dtype=np.int64)
        invalid=(~np.isfinite(raw).all(axis=1)).astype(np.int64)
        prefix=np.concatenate(([0],np.cumsum(invalid)))
        anchors=anchors[((prefix[anchors+1]-prefix[anchors-99])==0)&label_valid[anchors]]
        ix=np.flatnonzero(data['trip']==trip)
        assert np.array_equal(data['source_end'][ix],anchors)
        assert np.array_equal(data['source_start'][ix],anchors-99)
        expected_y=(labels[anchors,0]==0).astype(np.int64)
        assert np.array_equal(data['y'][ix],expected_y)
        assert np.all(data['vehicle'][ix]==(trip-1)//3+1)
        assert np.all(data['scenario'][ix]==(trip-1)%3+1)
        assert np.array_equal(data['anchor_time'][ix],ts[anchors])
        assert np.allclose(data['X_context'][ix],df[CONTEXT].to_numpy()[anchors],rtol=1e-6,atol=1e-6,equal_nan=True)
        rng=np.random.default_rng(20261017+trip)
        checked=np.unique(np.concatenate(([0,len(ix)-1],rng.choice(len(ix),size=min(33,len(ix)),replace=False))))
        max_error=0.
        for j in checked:
            end=int(anchors[j]); w=raw[end-99:end+1]
            vals={}
            for k,col in enumerate(SENSOR):
                a=w[:,k]
                for stat,v in [('mean',a.mean()),('std',a.std(ddof=0)),('rms',np.sqrt(np.mean(a*a))),('range',np.ptp(a)),('absmax',np.max(np.abs(a)))]:
                    vals[f'{col}__{stat}']=v
                    vals[f'{col}_{stat}']=v
                    vals[f'{stat}_{col}']=v
                vals[f'{col}__std_pop']=a.std(ddof=0)
                vals[f'{col}__max_abs']=np.max(np.abs(a))
            expected=np.asarray([vals[n] for n in feature_names],dtype=np.float64)
            actual=data['X_sensor'][ix[j]]
            assert np.allclose(actual,expected,rtol=3e-6,atol=3e-6), (trip,int(end),np.max(np.abs(actual-expected)))
            max_error=max(max_error,float(np.max(np.abs(actual-expected))))
        total_checked+=len(checked)
        spans=ts[anchors]-ts[anchors-99]
        results.append({'trip':trip,'raw_rows':len(df),'windows':len(ix),'class_counts':{str(k):int((expected_y==k).sum()) for k in (0,1)},'checked_feature_windows':len(checked),'max_absolute_float32_error':max_error,'window_span_seconds':{'min':float(spans.min()),'median':float(np.median(spans)),'max':float(spans.max())},'sensor_sha256':sha(sensor_path),'label_sha256':sha(label_path)})
    assert len(data['y'])==sum(r['windows'] for r in results)
    assert np.isfinite(data['X_sensor']).all() and not np.isinf(data['X_context']).any()
    result={'status':'passed','validation':'independent raw-label/population/window-feature verification, no model fitting','features_sha256':sha(HERE/'features.npz'),'manifest_sha256':sha(HERE/'feature_manifest.json'),'total_windows':len(data['y']),'total_feature_windows_independently_recomputed':total_checked,'trips':results}
    (HERE/'feature_population_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Build the fit-free, predeclared disjoint-window sensitivity population.
    take=np.flatnonzero((data['source_end']-99)%100==0)
    target=HERE/'nonoverlap'; target.mkdir(exist_ok=True)
    subset={k:v[take] if v.ndim and len(v)==len(data['y']) else v for k,v in data.items()}
    np.savez_compressed(target/'features.npz',**subset)
    child=dict(manifest)
    child.pop('stride25',None)
    child['anchor_rows_zero_based']='99,199,299,...; label is indicator at final anchor'
    child['combined_rows']=len(take)
    child['combined_class_counts']={str(k):int((subset['y']==k).sum()) for k in (0,1)}
    child['parent_excluded_counts']=child.pop('excluded_counts',{})
    child['npz_bytes']=(target/'features.npz').stat().st_size
    child['created_by']=Path(__file__).name
    child.update({'stride':100,'window_rows':100,'parent_features_sha256':result['features_sha256'],'parent_manifest_sha256':result['manifest_sha256'],'sensitivity_protocol_sha256':sha(HERE/'sensitivity_protocol_v1.md'),'retained_primary_indices':take.tolist(),'npz_sha256':sha(target/'features.npz'),'population_windows':len(take)})
    child['trips']=[]
    for trip in range(1,10):
        ids=np.flatnonzero(subset['trip']==trip)
        ends=subset['source_end'][ids]; starts=subset['source_start'][ids]
        assert (starts[1:]>ends[:-1]).all()
        assert set(np.unique(subset['y'][ids]))=={0,1}
        parent=next(r for r in results if r['trip']==trip)
        child['trips'].append({**parent,'windows':len(ids),'retained_windows':len(ids),'class_counts':{str(k):int((subset['y'][ids]==k).sum()) for k in (0,1)}})
    (target/'feature_manifest.json').write_text(json.dumps(child,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','primary_windows':len(data['y']),'nonoverlap_windows':len(take),'raw_windows_recomputed':total_checked}),flush=True)

if __name__=='__main__':
    main()
