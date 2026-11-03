"""Reusable KIT annotation-event metrics; consumes saved predictions only."""
from __future__ import annotations

import numpy as np


def _matching(edges, n_alarm):
    owner = [-1] * n_alarm
    def augment(event, seen):
        for alarm in edges[event]:
            if alarm in seen: continue
            seen.add(alarm)
            if owner[alarm] < 0 or augment(owner[alarm], seen):
                owner[alarm] = event
                return True
        return False
    for event in range(len(edges)): augment(event, set())
    return sorted((event, alarm) for alarm, event in enumerate(owner) if event >= 0)


def evaluate(test_indices, y_pred, data, raw, tolerance=0.0):
    """Return per-file and pooled engineering diagnostics.

    `data` is a dict of feature arrays with file_id, source_start/source_end,
    decision_time, anchor_row, and trip/file ids. `raw` maps fNN_timestamp and
    fNN_label (plus optional coordinate arrays) to raw arrays.
    """
    ix = np.asarray(test_indices, int).reshape(-1)
    pred = np.asarray(y_pred, int).reshape(-1)
    if len(ix) != len(pred): raise ValueError("test_indices/y_pred length mismatch")
    results = []
    file_ids = np.asarray(data["file_id"])[ix]
    start_key = "source_start" if "source_start" in data else "window_start"
    end_key = "source_end" if "source_end" in data else "window_end"
    t_key = "decision_time" if "decision_time" in data else "anchor_time"
    r_key = "anchor_row"
    for fid in sorted(set(file_ids.tolist())):
        take = np.flatnonzero(file_ids == fid)
        order = take[np.argsort(np.asarray(data[t_key])[ix[take]], kind="stable")]
        fi, hat = ix[order], pred[order]
        times = np.asarray(data[t_key], float)[fi]
        anchors = np.asarray(data[r_key], int)[fi]
        alarms, active = [], []
        for j, value in enumerate(hat):
            if value == 1:
                if active and not (hat[j-1] == 1 and times[j]-times[j-1] <= .251):
                    alarms.append(np.asarray(active)); active=[]
                active.append(j)
            elif active:
                alarms.append(np.asarray(active)); active=[]
        if active: alarms.append(np.asarray(active))
        prefix=f"f{int(fid):02d}_"
        rt=np.asarray(raw[prefix+"timestamp"],float)
        rl=np.asarray(raw[prefix+"label"],float)
        dt=np.diff(rt); good_dt=dt[np.isfinite(dt)&(dt>0)]
        step=float(np.median(good_dt)) if len(good_dt) else 0.0
        rowlabel=np.flatnonzero(rl==3)
        events=[]
        if len(rowlabel):
            cuts=np.flatnonzero((np.diff(rowlabel)!=1)|(dt[np.minimum(rowlabel[:-1],len(dt)-1)]>.1))+1
            for g in np.split(rowlabel,cuts): events.append((int(g[0]),int(g[-1]),float(rt[g[0]]),float(rt[g[-1]])))
        evaluated=[]; excluded=0
        for a,b,ta,tb in events:
            eligible=np.flatnonzero((anchors>=a)&(anchors<=b))
            if len(eligible): evaluated.append((a,b,ta,tb))
            else: excluded+=1
        edges=[]
        for a,b,ta,tb in evaluated:
            edges.append([k for k,c in enumerate(alarms) if np.any((times[c]>=ta-tolerance)&(times[c]<=tb+tolerance))])
        pairs=_matching(edges,len(alarms)); matched_alarm={a for _,a in pairs}; edge_alarm={a for e in edges for a in e}
        nonbump=len(set(range(len(alarms)))-edge_alarm); duplicate=len(edge_alarm-matched_alarm); unmatched=len(alarms)-len(pairs)
        assert unmatched==nonbump+duplicate
        firstrow,lastrow=int(anchors[0]),int(anchors[-1])
        lo=max(0,firstrow); hi=min(len(rl),lastrow+1)
        exposure=0.0
        for r in range(lo,hi):
            if rl[r]==1:
                step_r=float(dt[r]) if r<len(dt) and np.isfinite(dt[r]) and dt[r]>0 else step
                exposure+=min(step_r,.1)
        offsets=[]
        for e,ai in pairs:
            a,b,ta,tb=evaluated[e]; c=alarms[ai]
            eligible=[j for j in c if times[j]>=ta-tolerance and times[j]<=tb+tolerance]
            if eligible: offsets.append(float(times[eligible[0]]-ta))
        results.append({"file_id":int(fid),"tolerance_seconds":float(tolerance),"annotation_intervals":len(events),
            "evaluable_intervals":len(evaluated),"excluded_intervals_no_anchor":excluded,"matched_intervals":len(pairs),
            "interval_recall":len(pairs)/len(evaluated) if evaluated else None,"alarm_clusters":len(alarms),
            "unmatched_alarm_clusters":unmatched,"nonbump_alarm_clusters":nonbump,"duplicate_event_alarm_clusters":duplicate,
            "good_exposure_hours":exposure/3600,"nonbump_alarm_rate_per_good_hour":nonbump/(exposure/3600) if exposure else None,
            "unmatched_alarm_rate_per_good_hour":unmatched/(exposure/3600) if exposure else None,
            "matched_first_alarm_offsets_seconds":offsets})
    pooled={"tolerance_seconds":float(tolerance),"files":len(results),"annotation_intervals":sum(r["annotation_intervals"] for r in results),
        "evaluable_intervals":sum(r["evaluable_intervals"] for r in results),"matched_intervals":sum(r["matched_intervals"] for r in results),
        "alarm_clusters":sum(r["alarm_clusters"] for r in results),"unmatched_alarm_clusters":sum(r["unmatched_alarm_clusters"] for r in results),
        "nonbump_alarm_clusters":sum(r["nonbump_alarm_clusters"] for r in results),"duplicate_event_alarm_clusters":sum(r["duplicate_event_alarm_clusters"] for r in results),
        "good_exposure_hours":sum(r["good_exposure_hours"] for r in results),
        "matched_first_alarm_offsets_seconds":[v for r in results for v in r["matched_first_alarm_offsets_seconds"]]}
    assert pooled["unmatched_alarm_clusters"]==pooled["nonbump_alarm_clusters"]+pooled["duplicate_event_alarm_clusters"]
    h=pooled["good_exposure_hours"]
    pooled["nonbump_alarm_rate_per_good_hour"]=pooled["nonbump_alarm_clusters"]/h if h else None
    pooled["unmatched_alarm_rate_per_good_hour"]=pooled["unmatched_alarm_clusters"]/h if h else None
    pooled["interval_recall"]=pooled["matched_intervals"]/pooled["evaluable_intervals"] if pooled['evaluable_intervals'] else None
    return {"per_file":results,"pooled":pooled}


def _self_test():
    """Small deterministic checks for alarm gaps, excluded events, duplicates and 1:1 matching."""
    # A single alarm adjacent to two events can match only one; extra alarm is duplicate if linked.
    edges=[[0],[0,1]]
    assert len(_matching(edges,2))==2
    assert len(_matching([[0],[0]],1))==1
    # Alarm cluster continuity requires both grid and timing; negative prediction closes it.
    def clusters(pred, rows, times):
        n=0; active=False
        for j,p in enumerate(pred):
            if p:
                if not(active and pred[j-1] and times[j]-times[j-1]<=.251): n+=1
                active=True
            else: active=False
        return n
    assert clusters([1,1,1],[1,26,76],[0,.25,.75])==2
    assert clusters([1,1,1],[1,55,110],[0,.25,.5])==1
    assert clusters([1,0,1],[1,26,51],[0,.25,.5])==2

if __name__ == "__main__":
    _self_test()
