from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
OUT_JSON=ROOT/'data_feasibility_manifest_v2.json'
OUT_CSV=ROOT/'session_inventory_v2.csv'

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def scan(p, group):
 with p.open('r',encoding='utf-8-sig',newline='') as f:
  rd=csv.reader(f); header=next(rd); rows=0; finite=Counter(); blank=Counter(); labels=Counter(); durations=Counter(); unique=defaultdict(set); tmin=tmax=None; malformed=0; prev=None; spans=Counter()
  for row in rd:
   if not row or (len(row)==1 and not row[0]): continue
   rows+=1
   if len(row)!=len(header):
    malformed+=1; row=(row+['']*max(0,len(header)-len(row)))[:len(header)]
   d=dict(zip(header,row)); label=d.get('annotation_text','').strip(); dur=d.get('annotation_millisecond_press_duration','').strip()
   if label: labels[label]+=1
   if dur: durations[dur]+=1
   if label!=prev and label: spans[label]+=1
   prev=label
   for k,raw in d.items():
    s=raw.strip()
    if not s: blank[k]+=1; continue
    try: x=float(s)
    except ValueError: continue
    if math.isfinite(x):
     finite[k]+=1
     if k=='seconds_elapsed': tmin=x if tmin is None else min(tmin,x); tmax=x if tmax is None else max(tmax,x)
     if k in {'Temperature (°C)','Humidity (%)','Cloud Cover (%)','Wind Speed (km/h)','barometer_pressure','barometer_relativeAltitude','location_latitude','location_longitude','location_verticalAccuracy'}: unique[k].add(x)
  return {'session_file':p.name,'session_id':p.stem.split()[0],'group':group,'file_bytes':p.stat().st_size,'rows':rows,'header':header,'n_fields':len(header),'malformed_row_count':malformed,'seconds_elapsed_min':tmin,'seconds_elapsed_max':tmax,'annotation_text_counts':dict(labels),'annotation_nonempty_contiguous_runs':dict(spans),'annotation_press_duration_counts':dict(durations),'field_finite':dict(finite),'field_blank':dict(blank),'selected_field_unique_counts':{k:len(v) for k,v in unique.items()},'sha256':sha256(p)}

allrows=[]; summary_files=[]
for group,folder in [('normal',DATA/'inner'/'combined_normal'),('anomaly_recording_labeled',DATA/'inner'/'combined_anomalies')]:
 for p in sorted(folder.glob('*.csv'),key=lambda p:(int(p.stem) if p.stem.isdigit() else 10**9,p.name)):
  # The file named 99.csv is a four-row aggregate summary, not a sensor session.
  if group.startswith('anomaly') and p.name=='99.csv': summary_files.append({'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'sha256':sha256(p),'role':'aggregate summary table, excluded from session counts'}); continue
  allrows.append(scan(p,group))
fields=sorted({f for r in allrows for f in r['header']})
summary={}; schemas={}
for g in ['normal','anomaly_recording_labeled']:
 rr=[r for r in allrows if r['group']==g]; n=len(rr); total_rows=sum(r['rows'] for r in rr)
 summary[g]={'session_csvs':n,'rows':total_rows,'bytes':sum(r['file_bytes'] for r in rr),'field_count_distribution':dict(Counter(r['n_fields'] for r in rr)),'nonempty_annotation_rows':dict(sum((Counter(r['annotation_text_counts']) for r in rr),Counter())),'contiguous_labeled_runs':dict(sum((Counter(r['annotation_nonempty_contiguous_runs']) for r in rr),Counter())),'sessions_with_annotation_text':sum(bool(r['annotation_text_counts']) for r in rr),'sessions_with_annotation_columns':sum('annotation_text' in r['header'] for r in rr),'sessions_with_both_gps_fields':sum({'location_longitude','location_latitude'}.issubset(r['header']) for r in rr)}
 for field in fields:
  present=[r for r in rr if field in r['header']]; denom=sum(r['rows'] for r in present); finite=sum(r['field_finite'].get(field,0) for r in present)
  schemas.setdefault(field,{})[g]={'files_with_field':len(present),'file_presence_rate':len(present)/n if n else None,'rows_in_files_with_field':denom,'finite_values':finite,'finite_rate':finite/denom if denom else None}
# Isolated package does not include a separate per-session Annotation.csv; compare its all-anomalies exports.
iso=[]
for folder in [DATA/'inner'/'isolated_anomalies_1_86',DATA/'inner'/'isolated_anomalies_87_103']:
 for p in folder.rglob('*.csv'):
  if p.name.endswith('isolated_all_anomalies.csv') or p.name=='isolated_all_anomalies.csv': iso.append(scan(p,'isolated_anomaly_rows'))
iso_counts=Counter(); iso_durations=Counter(); iso_spans=Counter()
for r in iso:
 iso_counts.update(r['annotation_text_counts']); iso_durations.update(r['annotation_press_duration_counts']); iso_spans.update(r['annotation_nonempty_contiguous_runs'])
payload={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Schema/label inventory only. No models or score metrics. Combined sensor session CSVs counted; aggregate 99.csv excluded. Isolated *_isolated_all_anomalies.csv files separately inspected.','source':'Official Figshare RoadSens-4M Dataset v3 files already downloaded; source checksums in data/download_manifest.json.','combined_group_summaries':summary,'aggregate_csv_excluded':summary_files,'schema_by_field':schemas,'combined_sessions':allrows,'isolated_all_anomalies_exports':{'file_count':len(iso),'rows':sum(r['rows'] for r in iso),'annotation_nonempty_rows':dict(iso_counts),'nonempty_contiguous_labeled_runs':dict(iso_spans),'press_duration_values':dict(iso_durations),'individual_files':[{'file':r['session_file'],'session_id':r['session_id'],'rows':r['rows'],'label_counts':r['annotation_text_counts'],'press_duration_counts':r['annotation_press_duration_counts'],'sha256':r['sha256']} for r in iso]},'interpretation_notes':['annotation_text repeats across contiguous sensor rows in Combined event recordings, producing rowwise labeled intervals; it is not a whole-session label.','annotation_millisecond_press_duration occurs sparsely on event-start rows and does not encode the full rowwise annotation interval.','No standalone Annotation.csv with seconds start/end/text columns was found in the downloaded Combined/Isolated archives; isolated all-anomalies exports retain rowwise fields.','Normal labels are the package folder semantics and remain normal; no event labels were inferred for normal recordings.']}
for p in (OUT_JSON,OUT_CSV):
 if p.exists(): raise FileExistsError(p)
OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with OUT_CSV.open('x',encoding='utf-8-sig',newline='') as f:
 fields_csv=['session_id','session_file','group','file_bytes','rows','n_fields','seconds_elapsed_min','seconds_elapsed_max','malformed_row_count','annotation_text_counts','annotation_nonempty_contiguous_runs','annotation_press_duration_counts','sha256']; w=csv.DictWriter(f,fieldnames=fields_csv); w.writeheader()
 for r in allrows:
  z={k:r.get(k) for k in fields_csv}
  for k in ['annotation_text_counts','annotation_nonempty_contiguous_runs','annotation_press_duration_counts']: z[k]=json.dumps(z[k],ensure_ascii=False)
  w.writerow(z)
print(json.dumps({'groups':summary,'excluded_summary':summary_files,'isolated_export_files':len(iso),'fields':fields},ensure_ascii=True))
