"""Move explanatory architecture notes to the caption, preserving all tables."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'main_body.tex'
s=p.read_text(encoding='utf-8')
a=s.index(r'{\small Architecture of the evaluation audit.')
b=s.index(r'}\par\medskip',a)
caption=('Architecture of the evaluation audit. RoadSens, PVS and Mexico are separate tasks; '
 'no cross-dataset model transfer is performed. Source hashes, content checks, frozen split '
 'memberships and train-only preprocessing define the audit. In PVS, V/D denotes a '
 'vehicle/driver pair; each outer fold holds out three acquisitions and develops on six, '
 'under either V/D or scenario holdout. Outer F1 and estimation MAE are three-domain means. '
 'The disjoint-window, context and common-budget diagnostics reuse the same nine acquisitions. '
 'The declared requirement of an MAE reduction of at least 0.02 on both axes remains unmet. '
 'Waveforms and matrices are schematic.')
p.write_text(s[:a]+r'{\small '+caption+s[b:],encoding='utf-8')

q=ROOT.parent/'tex_v12/visual_source/verify_framework_v12.py'
code=q.read_text(encoding='utf-8').replace('v12','v13').replace("'version':12","'version':13")
extra="""
font_names=sorted({f[3] for f in art.get_fonts()})
assert font_names and all('TimesNewRoman' in f for f in font_names),font_names
report['actual_pdf_fonts']=font_names
report['drawing_geometry']=json.loads((ROOT/'visual_source/drawing_qa_v13.json').read_text(encoding='utf-8'))
assert report['drawing_geometry']['fork_symmetry']
"""
at=code.index("(ROOT/'visual_source/verification_v13.json')")
code=code[:at]+extra+code[at:]
(ROOT/'visual_source/verify_framework_v13.py').write_text(code,encoding='utf-8')
print('Caption updated; v13 verification script prepared.')
