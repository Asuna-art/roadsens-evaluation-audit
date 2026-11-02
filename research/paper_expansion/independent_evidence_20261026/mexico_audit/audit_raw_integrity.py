"""Luna-authored raw-data preflight, executed by lead due agent write restriction."""
import collections, hashlib, json, pathlib, zipfile
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
P=HERE.parent.parent/'independent_evidence_20261025/source_audit/applsci-15-10203-s001.zip'
CH=('accel_x','accel_y','accel_z','vib_x','vib_y','vib_z')
MAP={'':'no_event','pothole':'pothole','speed bump':'speed_bump','sudden braking':'sudden_braking'}
def h(b):return hashlib.sha256(b).hexdigest()
def main():
 report={'archive_sha256':h(P.read_bytes()),'trip':{},'pairwise':[],'window_rule_50_stride50':{}};ds={}
 with zipfile.ZipFile(P) as z:
  for t in range(1,5):
   name=f'MATLAB/urban_trip_{t}.json';raw=z.read(name);o=json.loads(raw);X=np.column_stack([o[k] for k in CH]).astype(float);y=np.asarray(o['label'],object);ds[t]=(X,y)
   starts=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1];ends=np.r_[starts[1:],len(y)];rh=collections.defaultdict(collections.Counter)
   for s,e in zip(starts,ends):rh[str(y[s])][int(e-s)]+=1
   stats={k:{'min':float(X[:,i].min()),'max':float(X[:,i].max()),'mean':float(X[:,i].mean()),'std':float(X[:,i].std()),'zero_n':int((X[:,i]==0).sum()),'unique_n':len(np.unique(X[:,i]))} for i,k in enumerate(CH)}
   per={MAP[lab]:{'n':int((y==lab).sum()),'mean':X[y==lab].mean(0).tolist(),'std':X[y==lab].std(0).tolist()} for lab in sorted(set(y))}
   report['trip'][str(t)]={'member':name,'json_sha256':h(raw),'rows':len(y),'finite_all':bool(np.isfinite(X).all()),'channel_stats':stats,'label_counts':{MAP[k]:int(v) for k,v in collections.Counter(y).items()},'label_run_count':len(starts),'label_run_length_histograms':{MAP[k]:dict(sorted(v.items())) for k,v in rh.items()},'per_label_channel_stats':per,'accel_vibration_correlations':[float(np.corrcoef(X[:,i],X[:,i+3])[0,1]) for i in range(3)]}
   wins=[]
   for s in range(0,len(y)-49,50):
    w=list(y[s:s+50]);c=collections.Counter(v for v in w if v);top=max(c.values()) if c else 0;winners=[k for k,v in c.items() if v==top];dom=MAP[winners[0]] if len(winners)==1 and top>=3 else 'no_event';rc=collections.Counter(w);mt=max(rc.values());mw=[k for k,v in rc.items() if v==mt]
    wins.append({'dominant':dom,'brake':'sudden braking' in w,'strict':MAP[mw[0]] if len(mw)==1 and mt>=26 else None})
   count=lambda seq:dict(collections.Counter(v for v in seq if v is not None))
   report['window_rule_50_stride50'][str(t)]={'n':len(wins),'supplied_nonempty_min3':count(w['dominant'] for w in wins),'exclude_any_brake':count(w['dominant'] for w in wins if not w['brake']),'exclude_dominant_brake':count(w['dominant'] for w in wins if w['dominant']!='sudden_braking'),'v1_strict26_exclude_any_brake':count(w['strict'] for w in wins if not w['brake'])}
 for a in range(1,5):
  for b in range(a+1,5):
   A,B=ds[a][0],ds[b][0];fr=[]
   for k in (2,5,10,25):
    def hs(X):return {h(np.ascontiguousarray(X[i:i+k],dtype='<f8').tobytes()) for i in range(len(X)-k+1)}
    fr.append({'length_rows':k,'exact_matches':len(hs(A)&hs(B))})
   report['pairwise'].append({'trip_a':a,'trip_b':b,'exact_six_channel_rows':len(set(map(tuple,A))&set(map(tuple,B))),'fragments':fr})
 report['interpretation']='Descriptive preflight only. Two documentation-derived route groups. No statistic authenticates acquisition provenance. Fixed event-run durations warrant annotation clarification, not misconduct inference.'
 out=HERE/'mexico_raw_integrity_audit.json';out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report['window_rule_50_stride50'],indent=2))
if __name__=='__main__':main()
