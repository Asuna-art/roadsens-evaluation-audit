from pathlib import Path
import re,json,hashlib
import pymupdf as fitz

ROOT=Path(__file__).resolve().parent.parent
QA=ROOT/'visual_source/qa'; QA.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rx=re.compile(r'\\begin\{tabularx\}.*?\\end\{tabularx\}',re.S)
report={'version':12,'artwork':'Code-native vector architecture; no generative illustration',
        'semantic_review':['Three separate source tasks; no cross-dataset model transfer.',
         'PVS: nine acquisition units; six development and three outer-test acquisitions per fold.',
         'RF/HGB with IMU or IMU+context; maximum inner OOF F1 selects the candidate.',
         'Outer classification and absolute estimation error are separate endpoints.',
         'Disjoint-window, context and budget diagnostics reuse the same acquisitions.',
         'Original two-axis criterion remains failed.'],
        'render_review_status':'pending'}
for name in ['main_body.tex','supplementary_body.tex']:
    a=rx.findall((ROOT.parent/'tex_v10'/name).read_text(encoding='utf-8'))
    b=rx.findall((ROOT/name).read_text(encoding='utf-8'))
    assert a==b,(name,'changed original table')
    report[name]={'unchanged_tabularx_blocks':len(a),'source_sha256':sha(ROOT/name)}
for name in ['main','supplementary','framework_architecture']:
    doc=fitz.open(ROOT/(name+'.pdf')); outside=[]; pages=[]
    for i,p in enumerate(doc):
        t=p.get_text()
        assert '??' not in t,(name,'unresolved reference',i+1)
        if name=='framework_architecture' or (name=='main' and ('Framework overview' in t or i==0)) or (name=='supplementary' and i==0):
            p.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(QA/f'{name}_page_{i+1:02}.png')
            pages.append(i+1)
        for b in p.get_text('dict')['blocks']:
            for l in b.get('lines',[]):
                for s in l['spans']:
                    r=fitz.Rect(s['bbox'])
                    if r.x0 < -.5 or r.y0 < -.5 or r.x1 > p.rect.width+.5 or r.y1 > p.rect.height+.5:
                        outside.append({'page':i+1,'text':s['text']})
    assert not outside,(name,outside)
    if name=='framework_architecture': assert len(doc)==1
    report[name]={'pages':len(doc),'pdf_sha256':sha(ROOT/(name+'.pdf')),'out_of_page_text_spans':outside,'review_pages':pages}
art=fitz.open(ROOT/'figures/framework_architecture_v12.pdf')[0]
sizes=[s['size'] for b in art.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
assert min(sizes)>=7.99
report['vector_asset']={'width_mm':round(art.rect.width/72*25.4,2),'height_mm':round(art.rect.height/72*25.4,2),
                        'minimum_font_pt':round(min(sizes),2),'embedded_raster_images':len(art.get_images())}
assert not art.get_images()
(ROOT/'visual_source/verification_v12.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
