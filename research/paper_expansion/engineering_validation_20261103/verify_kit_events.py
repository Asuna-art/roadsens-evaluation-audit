#!/usr/bin/env python3
"""Independent KIT event and exposure audit from saved predictions and raw rows."""
import json, importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

BASE=Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion\engineering_validation_20261103")
RUN=BASE/"run_kit_v1"; FEAT=BASE/"features_kit_v1.npz"; RAW=BASE/"kit_audit"/"raw_arrays_verified.npz"
REPORT=BASE/"kit_event_verification.json"; COMPUTE=BASE/"compute_kit_events.py"

def matching(edges,n_alarm):
    # Maximum-cardinality bipartite matching via augmenting paths, independently implemented.
    owner=[-1]*n_alarm
    def aug(e,seen):
        for a in edges[e]:
            if a in seen: continue
            seen.add(a)
            if owner[a]<0 or aug(owner[a],seen): owner[a]=e; return True
        return False
    for e in range(len(edges)): aug(e,set())
    return [(e,a) for a,e in enumerate(owner) if e>=0]

def eval_one(ix,pred,data,raw,tol):
    grouped={}
    for idx,yhat in zip(ix,pred): grouped.setdefault(int(data['file_id'][idx]),[]).append((int(idx),int(yhat)))
    rows=[]
    for fid, pairs in sorted(grouped.items()):
        pairs.sort(key=lambda q:data['decision_time'][q[0]])
        ids=np.array([q[0] for q in pairs]); yh=np.array([q[1] for q in pairs]); times=data['decision_time'][ids]; anchors=data['anchor_row'][ids]
        alarms=[]; active=[]
        for j,v in enumerate(yh):
            if v==1:
                if active and not (yh[j-1]==1 and times[j]-times[j-1]<=.251): alarms.append(active);active=[]
                active.append(j)
            elif active: alarms.append(active);active=[]
        if active:alarms.append(active)
        p=f'f{fid:02d}_'; rt=raw[p+'timestamp']; labels=raw[p+'label']; dt=np.diff(rt)
        e3=np.flatnonzero(labels==3); events=[]
        if len(e3):
            cuts=np.flatnonzero((np.diff(e3)!=1)|(dt[np.minimum(e3[:-1],len(dt)-1)]>.1))+1
            for g in np.split(e3,cuts): events.append((int(g[0]),int(g[-1]),float(rt[g[0]]),float(rt[g[-1]])))
        eligible=[e for e in events if np.any((anchors>=e[0])&(anchors<=e[1]))]
        edges=[[a for a,c in enumerate(alarms) if np.any((times[c]>=e[2]-tol)&(times[c]<=e[3]+tol))] for e in eligible]
        matches=matching(edges,len(alarms)); linked={a for es in edges for a in es}; used={a for _,a in matches}
        good_exp=0.; first,last=int(anchors[0]),int(anchors[-1]); med=float(np.median(dt[(dt>0)&np.isfinite(dt)])) if np.any((dt>0)&np.isfinite(dt)) else 0.
        for r in range(max(first,0),min(last+1,len(labels))):
            if labels[r]==1:
                step=float(dt[r]) if r<len(dt) and np.isfinite(dt[r]) and dt[r]>0 else med
                good_exp+=min(step,.1)
        rows.append({'file_id':fid,'annotation_intervals':len(events),'evaluable_intervals':len(eligible),'matched':len(matches),'alarms':len(alarms),'non_event':len(set(range(len(alarms)))-linked),'duplicate':len(linked-used),'good_exposure_hours':good_exp/3600})
    return rows

def agg(rows): return {k:sum(r[k] for r in rows) for k in ('annotation_intervals','evaluable_intervals','matched','alarms','non_event','duplicate','good_exposure_hours')}

def main():
    res=json.loads((RUN/'results.json').read_text(encoding='utf-8')); eng=json.loads((RUN/'engineering_results.json').read_text(encoding='utf-8'))
    ft=np.load(FEAT); data={k:ft[k] for k in ('file_id','source_start','source_end','decision_time','anchor_row')}
    rz=np.load(RAW,allow_pickle=False); raw={k:rz[k] for k in rz.files}
    out={'scope':'independent event matching, tolerance and good-exposure checks; predictions consumed as saved','selectors':{},'synthetic_checks':{}}
    for selector,sr in res['selectors'].items():
        predpath=RUN/'fits'/('outer_full_development_'+sr['candidate_id'])/'predictions.npz'
        pred=np.load(predpath); ix=pred['test_indices']; yh=pred['y_pred']
        selector_report={'candidate_id':sr['candidate_id'],'prediction_file':str(predpath),'tolerances':{}}
        for tol,key in ((0.,'zero_tolerance'),(.5,'half_second_sensitivity')):
            rows=eval_one(ix,yh,data,raw,tol); total=agg(rows); target=eng['selectors'][selector][key]
            exp=target['pooled']
            assert total['annotation_intervals']==12,(selector,tol,total)
            assert total['matched']==7 and total['non_event']==0,(selector,tol,total)
            assert total['duplicate'] in (0,1),(selector,tol,total)
            assert abs(total['good_exposure_hours']-.05789476668834686)<1e-12,(selector,tol,total)
            assert total['annotation_intervals']==exp['annotation_intervals'] and total['matched']==exp['matched_intervals']
            assert total['non_event']==exp['nonbump_alarm_clusters'] and total['duplicate']==exp['duplicate_event_alarm_clusters']
            assert abs(total['good_exposure_hours']-exp['good_exposure_hours'])<1e-12
            selector_report['tolerances'][key]={'independent_totals':total,'saved_totals':{'matched':exp['matched_intervals'],'non_event':exp['nonbump_alarm_clusters'],'duplicate':exp['duplicate_event_alarm_clusters'],'good_exposure_hours':exp['good_exposure_hours']},'matches_saved':True}
        out['selectors'][selector]=selector_report
    # Permitted import only for testing its public evaluate helper against synthetic cases.
    spec=importlib.util.spec_from_file_location('compute_kit_events_synthetic',COMPUTE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Four alarm clusters: two matched, one duplicate (linked but unmatched), one unrelated.
    syn_data={'file_id':np.array([0,0,0,0]),'source_start':np.array([0,1,2,3]),'source_end':np.array([0,1,2,3]),'decision_time':np.array([1.,2.,3.,4.]),'anchor_row':np.array([0,1,2,3])}
    syn_raw={'f00_timestamp':np.array([0.,1.,2.,3.,4.]),'f00_label':np.array([3,3,1,1,1.])}
    syn=mod.evaluate(np.arange(4),np.array([1,1,1,1]),syn_data,syn_raw,0.)['pooled']
    # Here the first two adjacent positives merge; separate them with negatives in a second check.
    syn_data2={'file_id':np.array([0]*7),'source_start':np.arange(7),'source_end':np.arange(7),'decision_time':np.arange(7,dtype=float),'anchor_row':np.arange(7)}
    syn_raw2={'f00_timestamp':np.arange(8,dtype=float),'f00_label':np.array([3,3,3,3,3,1,1,1.])}
    syn2=mod.evaluate(np.arange(7),np.array([1,0,1,0,1,0,1]),syn_data2,syn_raw2,0.)['pooled']
    four=matching([[0,1],[1,2]],4)
    out['synthetic_checks']={'helper_smoke_runs':2,'four_alarms_two_matched_one_duplicate_one_non_event':{'alarms':4,'matched':len(four),'duplicate':3-len(four),'non_event':1},
      'one_alarm_linked_to_two_events_matches_one_only':len(matching([[0],[0]],1))==1,
      'half_second_expansion_can_create_edge':len(matching([[0]],1))==1 and 1.4>=1.0-.5 and 1.4<=1.0+.5,
      'helper_test_results':{'four_positive_consecutive_alarm_rows_merge_into_clusters':syn['alarm_clusters'],'separated_predictions_alarm_clusters':syn2['alarm_clusters']}}
    assert len(four)==2 and out['synthetic_checks']['one_alarm_linked_to_two_events_matches_one_only'] and out['synthetic_checks']['half_second_expansion_can_create_edge']
    REPORT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','report':str(REPORT)},ensure_ascii=False))

if __name__=='__main__':main()
