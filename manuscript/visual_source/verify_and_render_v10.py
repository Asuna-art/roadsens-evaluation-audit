"""Read-only evidence checks and QA render. Does not fit or resample models."""
from pathlib import Path
import csv, json, hashlib, math, re
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT.parent.parent
QA=ROOT/'visual_source'/'qa'
QA.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_files={
 'Primary':BASE/'external_pvs_v1/checked_summary.json',
 'Disjoint windows':BASE/'external_pvs_v1/nonoverlap/checked_summary.json',
 'Common budget (post hoc)':BASE/'independent_evidence_20261025/budget_nonoverlap_v1/budget_matched_axis_summary.csv',
 'Mexico':BASE/'independent_evidence_20261026/mexico_audit/mexico_run_v2_frozen/run_v2/results.json',
}
plot_data=json.loads((ROOT/'figures/pvs_endpoints_v10.json').read_text())['rows']
locks={}
if all(p.exists() for p in source_files.values()):
    for label,path in source_files.items():
        locks[label]={'source':str(path.relative_to(BASE)),'sha256':sha(path)}
    for plotted in plot_data:
        group=plotted['group']; axis='vehicle' if plotted['domain']=='Vehicle/driver' else 'scenario'
        if group=='Common budget (post hoc)':
            source=list(csv.DictReader(source_files[group].open(encoding='utf-8')))
            mae='mean_mae'
        else:
            source=json.loads(source_files[group].read_text())['axis_selector_rows']; mae='mae'
        for label,scheme in [('random','row_random'),('matched','matched_factor')]:
            row=next(r for r in source if r['axis']==axis and r['scheme']==scheme)
            for plotted_key,source_key in [('outer_macro_f1','mean_outer_macro_f1'),('estimation_mae',mae)]:
                assert f"{float(row[source_key]):.4f}"==f"{plotted[plotted_key+'_'+label]:.4f}",(group,axis,label,plotted_key)
    primary=json.loads(source_files['Primary'].read_text())['primary_criterion']
    assert primary['supported_both_axes'] is False
    locks['checks']={'all_24_plotted_values_match_frozen_summaries_to_4_decimals':True,
                    'primary_threshold_unchanged':0.02,'supported_both_axes':False}
    (ROOT/'visual_source/evidence_lock_v10.json').write_text(json.dumps(locks,indent=2),encoding='utf-8')

report={}
for name in ['main','supplementary','graphical_abstract']:
    pdf=ROOT/(name+'.pdf'); doc=fitz.open(pdf)
    entries=[]; out_spans=[]; small=[]
    for i,p in enumerate(doc):
        text=p.get_text()
        if any(k in text for k in ['Graphical abstract','Figure 1:','Figure 2:','Figure 3:']) or name=='graphical_abstract':
            p.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(QA/f'{name}_page_{i+1:02}.png')
            entries.append(i+1)
        for b in p.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for span in line['spans']:
                    r=fitz.Rect(span['bbox'])
                    if r.x0 < -.5 or r.y0 < -.5 or r.x1 > p.rect.width+.5 or r.y1 > p.rect.height+.5:
                        out_spans.append({'page':i+1,'text':span['text'],'bbox':list(r)})
        assert '??' not in text, (name,i+1,'unresolved cross reference')
    # Whole-document thumbnails expose blank pages, overfull tables and broken floats.
    for start in range(0,len(doc),12):
        sheet=Image.new('RGB',(4*270,3*375),'#e8e8e8'); draw=ImageDraw.Draw(sheet)
        for j in range(start,min(start+12,len(doc))):
            pix=doc[j].get_pixmap(matrix=fitz.Matrix(.5,.5),alpha=False)
            im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            im.thumbnail((260,345))
            x=(j-start)%4*270; y=(j-start)//4*375
            sheet.paste(im,(x+(270-im.width)//2,y+20)); draw.text((x+10,y+3),str(j+1),fill='black')
        sheet.save(QA/f'{name}_contact_{start//12+1}.png')
    report[name]={'pages':len(doc),'pdf_sha256':sha(pdf),'rendered_figure_pages':entries,
                  'out_of_page_text_spans':out_spans,
                  'source_sha256':sha(ROOT/('main_body.tex' if name=='main' else 'supplementary_body.tex' if name=='supplementary' else name+'.tex'))}
    assert not out_spans,(name,out_spans)
    if name=='graphical_abstract': assert len(doc)==1,'Graphical abstract must be one page'
for name in ['graphical_abstract_v10','pvs_evaluation_workflow_v10','pvs_endpoints_v10']:
    p=fitz.open(ROOT/'figures'/(name+'.pdf'))[0]
    sizes=[s['size'] for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
    assert min(sizes)>=7.99,(name,min(sizes))
    report[name]={'width_mm':round(p.rect.width/72*25.4,2),'height_mm':round(p.rect.height/72*25.4,2),'min_text_pt':round(min(sizes),2)}
(ROOT/'visual_source/final_compile_qa_v10.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
