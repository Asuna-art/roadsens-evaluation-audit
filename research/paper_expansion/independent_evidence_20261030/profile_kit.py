"""No-fit XML-level source profile, retaining recording boundaries rather than counting files as trips."""
import pathlib,zipfile,io,json,collections,datetime,re
import numpy as np
from lxml import etree
B=pathlib.Path(__file__).resolve().parent/'kit_acquisition';records=[];imu_times=[];schlag_times={}
with zipfile.ZipFile(B/'Daten2018.zip') as outer:
 for n in outer.namelist():
  if not n.endswith('.xlsx'):continue
  with zipfile.ZipFile(io.BytesIO(outer.read(n))) as inner:
   sheet=next(p for p in inner.namelist() if p.startswith('xl/worksheets/sheet') and p.endswith('.xml'));times=[];events=collections.Counter();materials=collections.Counter();gps=[];rows=0
   xml=inner.read(sheet);rows=xml.count(b'<row ')-1;selected={k:[] for k in ('A','T','U','AA','AB')}
   for match in re.finditer(rb'<c\b[^>]*\br="(A|T|U|AA|AB)([0-9]+)"[^>]*>(.*?)</c>',xml):
    col,num,body=match.groups()
    if num==b'1':continue
    value=re.search(rb'<v>([^<]*)</v>',body)
    if value is not None:
     try:selected[col.decode()].append((int(num),float(value.group(1))))
     except ValueError:pass
   times=[v for i,v in selected['A']];materials.update(str(int(v)) for i,v in selected['AA']);events.update(str(int(v)) for i,v in selected['AB']);lons=dict(selected['U']);gps=[(v,lons[i]) for i,v in selected['T'] if i in lons]
   if '/IMU' in n:assert rows==len(times),(n,rows,len(times))
   a=np.asarray(times);d=np.diff(a);r={'file':n,'data_rows':rows,'timestamp_rows':len(times),'first_timestamp':float(a[0]) if len(a) else None,'last_timestamp':float(a[-1]) if len(a) else None,'decreasing_steps':int(np.sum(d<0)),'gaps_over_1s':int(np.sum(d>1)),'event_counts':dict(events),'material_counts':dict(materials),'GPS_rows':len(gps),'first_GPS':gps[0] if gps else None,'last_GPS':gps[-1] if gps else None};records.append(r)
   if '/IMU' in n:imu_times.append(a)
   else:schlag_times[n]=a
   print('profiled',len(records),n.split('/')[-1],flush=True)
joined=np.sort(np.concatenate(imu_times));overlap=[]
for n,a in schlag_times.items():
 ix=np.searchsorted(joined,a);hit=(ix<len(joined));hit[hit]=joined[ix[hit]]==a[hit];overlap.append({'file':n,'timestamp_rows':len(a),'exact_timestamp_matches_to_IMU_files':int(hit.sum())})
imu=sorted([r for r in records if '/IMU' in r['file']],key=lambda r:r['first_timestamp']);boundaries=[{'previous':a['file'],'next':b['file'],'gap_seconds':b['first_timestamp']-a['last_timestamp']} for a,b in zip(imu,imu[1:])]
report={'status':'source_profile_only_no_models','files':records,'adjacent_IMU_boundaries':boundaries,'Schlag_timestamp_overlap':overlap,'independent_acquisition_count':'not established; timestamps/GPS/source protocol must determine groups','event_semantics_primary_source':'https://doi.org/10.1155/2018/8647607','class_mapping':{'1':'good','3':'pothole','6':'speed bump'},'prohibition':'Do not count XLSX files or overlapping Schlag excerpts as independent drives.'}
(B/'source_profile.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'workbooks_profiled':len(records),'total_rows':sum(r['data_rows'] for r in records),'IMU_event_counts':dict(sum((collections.Counter(r['event_counts']) for r in imu),collections.Counter())),'Schlag_overlap':overlap},indent=2),flush=True)
