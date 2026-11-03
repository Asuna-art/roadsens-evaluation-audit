from pathlib import Path
import json, hashlib, re
import pymupdf as fitz
base=Path(__file__).resolve().parent.parent
vs=base/'visual_source'; qa=vs/'qa'; qa.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(n): return json.loads((vs/n).read_text(encoding='utf-8'))
result={'version':14,'checks':{},'manual_visual_inspection':'pending'}
# Compiler receipts: successful Tectonic exits and PDFs exist.
for key,receipt,pdf in [('main','main_build_v14.json',base/'main.pdf'),('standalone','pvs_evaluation_workflow_build_v14.json',base/'pvs_evaluation_workflow.pdf')]:
 d=load(receipt); result['checks'][key+'_compile']={'exit_code':d.get('exitCode'),'pdf_exists':pdf.exists(),'receipt_pdf_exists':d.get('pdfExists'),'passed':d.get('exitCode')==0 and pdf.exists() and d.get('pdfExists') is True}
# Figure vector/font/spec checks
fig=base/'figures'/'pvs_evaluation_workflow_v14.pdf'; doc=fitz.open(fig); page=doc[0]
fonts=sorted({f[3] for p in doc for f in p.get_fonts(full=True)})
spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
minpt=min((s['size'] for s in spans),default=0)
images=sum(len(p.get_images(full=True)) for p in doc)
size_mm=[round(page.rect.width*25.4/72,2),round(page.rect.height*25.4/72,2)]
result['checks']['figure_vector']={'fonts':fonts,'only_times_new_roman_ps_regular_bold':bool(fonts) and all(x.split('+')[-1] in ('TimesNewRomanPSMT','TimesNewRomanPS-BoldMT','TimesNewRomanPS','TimesNewRomanPS-Bold') for x in fonts),'embedded_raster_count':images,'minimum_font_pt':round(minpt,3),'minimum_font_pass':minpt>=8.99,'size_mm':size_mm,'size_pass':abs(size_mm[0]-178)<0.5 and abs(size_mm[1]-104)<0.5}
text=page.get_text()
result['checks']['new_figure_labels']={x:(x in text) for x in ['Select 1 of 4 candidates','Save inner macro-F1']}
# Locate Figure 1 on main PDF and render
main=fitz.open(base/'main.pdf'); cap='PVS acquisition grouping and nested evaluation.'; match=[]
result['checks']['main_page_count']=len(main)
for i,p in enumerate(main):
 t=p.get_text()
 if cap in t: match.append(i)
if match:
 idx=match[0]; result['checks']['figure1_page']=idx+1
 main[idx].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(qa/'main_figure1_page.png')
 page.get_pixmap(dpi=96,alpha=False).save(qa/'pvs_workflow_final_size.png')
# Main PDF text checks: no ?? or page-boundary overflow based on extracted span boxes.
issues=[]
for i,p in enumerate(main):
 if '??' in p.get_text(): issues.append({'page':i+1,'issue':'??'})
 for b in p.get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   for s in l['spans']:
    x0,y0,x1,y1=s['bbox']
    if x0 < -0.5 or y0 < -0.5 or x1 > p.rect.width+0.5 or y1 > p.rect.height+0.5:
     issues.append({'page':i+1,'issue':'out_of_page_text','text':s['text'][:80]})
result['checks']['main_pdf_text_bounds']={'issues':issues,'passed':not issues}
# Caption wording and tabularx identity
body=(base/'main_body.tex').read_text(encoding='utf-8-sig')
result['checks']['caption']={'contains_S_VD_definition':'S denotes scenario and V/D a vehicle/driver pair' in body,'contains_pooled_oof_wording':'by pooled inner out-of-fold macro-F1 on the six development acquisitions' in body,'passed':all(x in body for x in ['S denotes scenario and V/D a vehicle/driver pair','by pooled inner out-of-fold macro-F1 on the six development acquisitions'])}
def tabs(p): return re.findall(r'\\begin\{tabularx\}.*?\\end\{tabularx\}',p.read_text(encoding='utf-8-sig'),re.S)
v13=base.parent/'tex_v13'
tabcheck={}
for name in ['main_body.tex','supplementary_body.tex']:
 a=tabs(v13/name); b=tabs(base/name); tabcheck[name]={'v13_count':len(a),'v14_count':len(b),'identical':a==b}
result['checks']['tabularx']=tabcheck
# supplementary source/PDF unchanged
s13=v13/'supplementary_body.tex'; s14=base/'supplementary_body.tex'; p13=v13/'supplementary.pdf'; p14=base/'supplementary.pdf'
result['checks']['supplementary_unchanged']={'body_sha256_equal':sha(s13)==sha(s14),'pdf_sha256_equal':sha(p13)==sha(p14),'body_sha256':sha(s14),'pdf_sha256':sha(p14)}
assert result['checks']['main_compile']['passed']
assert result['checks']['standalone_compile']['passed']
assert result['checks']['figure_vector']['only_times_new_roman_ps_regular_bold']
assert result['checks']['figure_vector']['embedded_raster_count']==0
assert result['checks']['figure_vector']['minimum_font_pass']
assert result['checks']['figure_vector']['size_pass']
assert all(result['checks']['new_figure_labels'].values())
assert len(match)==1
assert all(x in main[match[0]].get_text() for x in ['Select 1 of 4 candidates','Save inner macro-F1'])
assert result['checks']['main_pdf_text_bounds']['passed']
assert result['checks']['caption']['passed']
assert all(v['identical'] for v in tabcheck.values())
assert result['checks']['supplementary_unchanged']['body_sha256_equal']
assert result['checks']['supplementary_unchanged']['pdf_sha256_equal']
(vs/'verification_v14.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
