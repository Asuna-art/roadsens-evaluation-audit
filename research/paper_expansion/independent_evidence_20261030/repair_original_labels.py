"""Freeze an original-annotation cohort; audit exact restoration, without model fits.
Luna delegation supplied scope; lead implemented after agent filesystem access failed.
"""
import csv,io,zipfile,pathlib,json,hashlib,collections
import numpy as np
B=pathlib.Path(__file__).resolve().parent
Z=B.parent/'independent_evidence_20261026/indonesia_audit/github-main.zip'
OUT=B/'indonesia_original_label_v1'
if OUT.exists():raise FileExistsError('Never overwrite cohort')
def sha(b):return hashlib.sha256(b).hexdigest()
spec=[('motorcycle/samsung a30/M5-1 b samsung a30 labeled.csv','20200312_morning','samsung_handlebar'),('motorcycle/asus zen max/12M5-1 d zen max-5 classes.csv','20200312_morning','asus_pocket'),('motorcycle/asus zen max/12M0-2-p3 d zen max-5 classes.csv','20200312_afternoon','asus_pocket'),('motorcycle/asus zen max/12M0-2-p4 d zen max-5 classes.csv','20200312_afternoon','asus_pocket'),('motorcycle/samsung a30/M5-4 b samsung a30 labelled.csv','20200314_morning','samsung_handlebar')]
sources=[];sensor=[];labels=[];passes=[];streams=[];rowsource=[];rowindex=[];gps=[];times=[];elapsed=[];original={}
with zipfile.ZipFile(Z) as z:
 names=z.namelist()
 for si,(suffix,group,phone) in enumerate(spec):
  name=next(n for n in names if n.endswith('/'+suffix));raw=z.read(name);rows=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))));assert rows and all(len(r)==21 for r in rows)
  yy=[int(r[20]) for r in rows];assert set(yy)<=({0,1,2,3,4} if phone=='asus_pocket' else {0,1,2,3})
  A=np.asarray([[float(x) for x in r[:6]] for r in rows]);assert np.isfinite(A).all();original[suffix]=rows
  sources.append({'path':suffix,'sha256':sha(raw),'rows':len(rows),'pass_id':group,'phone_stream':phone,'labels':dict(collections.Counter(yy)),'first_timestamp':rows[0][19],'last_timestamp':rows[-1][19]})
  sensor.extend(A.tolist());labels.extend(yy);passes.extend([group]*len(rows));streams.extend([phone]*len(rows));rowsource.extend([si]*len(rows));rowindex.extend(range(len(rows)));gps.extend([[float(r[15]),float(r[16])] for r in rows]);times.extend(r[19] for r in rows);elapsed.extend(float(r[18]) for r in rows)
 # Restoration is an audited mapping, not permission to inherit labels for unmatched new acquisitions.
 restore=[]
 for suffix,group,phone in spec:
  if phone!='samsung_handlebar':continue
  rows=original[suffix];token='12M5-1 b samsung a30.csv' if 'M5-1 ' in suffix else 'M5-4 b ngebut gak ngerem samsung.csv'
  wide=next(n for n in names if 'motorcycle 6 class labelled/' in n and n.endswith('/'+token));wr=list(csv.reader(io.StringIO(z.read(wide).decode('utf-8-sig'))));bykey=collections.defaultdict(list)
  for i,r in enumerate(rows):bykey[r[18],r[19]].append((i,r))
  accepted=[];missing=changed=ambiguous=0
  for i,r in enumerate(wr):
   candidates=bykey.get((r[18],r[19]),[])
   if not candidates:missing+=1;continue
   matches=[(j,a) for j,a in candidates if a[:20]==r[:20]]
   if len(matches)>1:ambiguous+=1;continue
   if not matches:changed+=1;continue
   j,a=matches[0];accepted.append({'wide_source_row':i,'original_source_row':j,'restored_label':int(a[20]),'wide_column21':r[20],'wide_column22':r[21] if len(r)>21 else None})
  restore.append({'original':suffix,'wide':wide,'pass_id':group,'wide_sha256':sha(z.read(wide)),'accepted_exact20':len(accepted),'unmatched_key':missing,'different_sensor_fields':changed,'ambiguous_exact_matches':ambiguous,'label21_agreement':sum(a['restored_label']==int(a['wide_column21']) for a in accepted),'label22_agreement':sum(str(a['restored_label'])==a['wide_column22'] for a in accepted),'mapping':accepted})
OUT.mkdir()
np.savez_compressed(OUT/'original_cohort.npz',sensor6=np.asarray(sensor),label=np.asarray(labels,dtype=np.int8),pass_id=np.asarray(passes),phone_stream=np.asarray(streams),source_id=np.asarray(rowsource,dtype=np.int16),source_row=np.asarray(rowindex),latitude_longitude=np.asarray(gps),timestamp=np.asarray(times),elapsed=np.asarray(elapsed))
manifest={'status':'original_label_version_frozen_no_models','archive_sha256':sha(Z.read_bytes()),'annotation_source':'literal field21 of original motorcycle/ files only; physical annotation accuracy is not independently re-adjudicated','source_url':'https://github.com/s-budidarma/dataset-smarphone-sensor-for-road-surface-monitoring','sources':sources,'rows':len(labels),'unique_passes':len(set(passes)),'dates':2,'independent_roads':'not established','phone_streams_not_independent_acquisitions':True,'new_acquisitions_created_by_repair':0,'npz_sha256':sha((OUT/'original_cohort.npz').read_bytes()),'primary_task_class_map':{'0':'source flat road','1':'source pothole','2':'source speed bump','3':'source bumpy road','4':'source human movement in Asus only'},'restoration_summaries':[{k:v for k,v in a.items() if k!='mapping'} for a in restore]}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');(OUT/'restoration_exact_matches.json').write_text(json.dumps(restore,indent=2)+'\n',encoding='utf-8')
(OUT/'README.md').write_text('''# 原版标签修复队列（2026-10-30）

固定原始motorcycle/目录五个CSV的第21列标签，不混合六分类导出版本。三个经过组、两个日期；同一经过的两部手机保持分流，不作为独立采集；p3/p4保持来源索引并共享经过ID。原始时间、GPS、六轴信号、标签及来源行号均保留。未切窗、未训练模型、未使用预测伪造真值。

扩展三星文件按时间键连接，仅当前20列完全相同时恢复原版标签；不匹配行不继承标签。映射可逐行复查。该修复建立标签版本的一致性，不代表原版标注已获物理现场复核，也不增加采集数。扩展文件未匹配部分及其他新经过的标签来源仍待澄清。原始数据仅本地保留，不进入稿件分发包。
''',encoding='utf-8')
print(json.dumps({k:manifest[k] for k in ('status','rows','unique_passes','dates','npz_sha256','restoration_summaries')},indent=2))
