"""Independently compare all displayed external tables with checked results."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def display(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
C={'rf_sensor':'RF100 IMU','rf_sensor_context':'RF100 IMU+context','hgb_sensor':'HGB IMU','hgb_sensor_context':'HGB IMU+context'}
S={'row_random':'Random','leave_one_trip':'Trip','matched_factor':'Matched'}
A={'vehicle':'Vehicle','scenario':'Scenario'}

def parse(path):
    out={};current=None
    for line in path.read_text(encoding='utf-8').splitlines():
        m=re.match(r'\*\*(Table S?\d+)\.',line)
        if m:current=m.group(1);out[current]={'columns':None,'rows':[]}
        if current and line.startswith('|'):
            cells=[x.strip() for x in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?',x) for x in cells):continue
            if out[current]['columns'] is None:out[current]['columns']=cells
            else:out[current]['rows'].append(cells)
    return out

def main():
    p=load(HERE/'checked_summary.json');c=load(HERE/'nonoverlap/checked_summary.json');pop=load(HERE/'feature_population_validation.json')
    actual={**parse(HERE/'tables_main.md'),**parse(HERE/'tables_supplement.md')};expected={}
    for number,obj in [('Table 8',p),('Table S13',c)]:
        expected[number]=[[A[r['axis']],C[r['candidate_id']]]+[r[k] for k in ('mean_inner_random_macro_f1','mean_inner_trip_macro_f1','mean_inner_matched_macro_f1','mean_outer_macro_f1','mean_outer_balanced_accuracy')] for r in obj['candidate_axis_mean_rows']]
    for number,obj in [('Table 9',p),('Table S14',c)]:
        expected[number]=[[A[r['axis']],S[r['scheme']]]+[r[k] for k in ('mean_inner_macro_f1','mean_outer_macro_f1','mae','bias_outer_minus_inner')] for r in obj['axis_selector_rows']]
    vehicles={1:'Volkswagen Saveiro / Driver 1',2:'Fiat Bravo / Driver 2',3:'Fiat Palio / Driver 3'}
    expected['Table S8']=[[f"PVS {r['trip']}",vehicles[(r['trip']-1)//3+1],(r['trip']-1)%3+1,r['raw_rows'],r['windows'],r['class_counts']['1']] for r in pop['trips']]
    for axis,cn,sn,mn in [('vehicle','Table S9','Table S11','Table S15'),('scenario','Table S10','Table S12','Table S16')]:
        rows=[r for r in p['candidate_domain_rows'] if r['axis']==axis]
        expected[cn]=[[r['outer_id'],C[r['candidate_id']]]+[r[k] for k in ('inner_random_macro_f1','inner_trip_macro_f1','inner_matched_macro_f1','outer_macro_f1','outer_balanced_accuracy')] for r in rows]
        expected[mn]=[[r['outer_id'],C[r['candidate_id']],*r['outer_class_recalls'],*r['outer_confusion_matrix'][0],*r['outer_confusion_matrix'][1]] for r in rows]
        expected[sn]=[[r['outer_id'],S[r['scheme']],C[r['chosen_candidate_id']]]+[r[k] for k in ('inner_macro_f1','outer_macro_f1','absolute_error','oracle_regret_macro_f1')] for r in p['selector_domain_rows'] if r['axis']==axis]
    assert set(actual)==set(expected),(set(actual),set(expected))
    counts={}
    for table,rows in expected.items():
        formatted=[[display(v) for v in row] for row in rows]
        assert actual[table]['rows']==formatted,table
        counts[table]={'rows':len(rows),'columns':len(rows[0]),'cells':sum(map(len,rows))}
    proof={'status':'passed','tables':counts,'table_count':len(counts),'cells_checked':sum(v['cells'] for v in counts.values()),'sources':{name:sha(HERE/name) for name in ('checked_summary.json','nonoverlap/checked_summary.json','feature_population_validation.json','tables_main.md','tables_supplement.md','table_numeric_manifest.json')}}
    (HERE/'table_validation_v1.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','tables':len(counts),'cells':proof['cells_checked']}),flush=True)

if __name__=='__main__':main()
