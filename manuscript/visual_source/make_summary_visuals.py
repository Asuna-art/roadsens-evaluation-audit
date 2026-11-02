"""Editorial diagrams for v10. No model fitting or resampling.

Run with the locked research interpreter from any directory.
All numerical labels retain the four-decimal precision of manuscript v9.
"""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'figures'
OUT.mkdir(exist_ok=True)
INK, MUTED, GRAY, RULE, TEAL, RISK = '#171717', '#505653', '#666866', '#D8DCD9', '#1F6F5F', '#B5473C'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': RULE,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .6})


def canvas(height_mm):
    fig = plt.figure(figsize=(178/25.4, height_mm/25.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    return fig, ax


def txt(ax, x, y, s, size=8.5, color=INK, weight='normal', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   va=kw.pop('va', 'top'), ha=kw.pop('ha', 'left'), **kw)


def rule(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color=RULE, lw=.7, clip_on=False)


def arrow(ax, start, end, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9,
                 lw=.8, color=MUTED, shrinkA=2, shrinkB=2, **kw))


def box(ax, x, y, w, h, label, accent=False, size=8.5):
    ax.add_patch(Rectangle((x,y), w,h, facecolor='white',
                           edgecolor=TEAL if accent else RULE, lw=1 if accent else .8))
    txt(ax,x+w/2,y+h/2,label,size=size,color=TEAL if accent else INK,
        ha='center',va='center',linespacing=1.35)


def export(fig, name):
    # Keep the exact physical canvas; bbox_inches='tight' changes insertion size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            bad.append(t.get_text())
    if bad: raise RuntimeError(f'Text outside figure: {bad}')
    for fmt in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{fmt}', dpi=300, facecolor='white')
    with Image.open(OUT/f'{name}.png') as im:
        im.convert('L').save(OUT/f'{name}_grayscale.png')
    plt.close(fig)


def graphical_abstract():
    fig, ax = canvas(118)
    txt(ax,.035,.965,'Model selection and performance estimation can diverge',11.3,weight='bold')
    txt(ax,.035,.905,'Road-sensing evaluation audit  |  Three sources, different grouping evidence',8.5,color=MUTED)
    # Counts are explicitly unlike units; no area/count chart implies equivalence.
    for x, title, sub in [(.035,'RoadSens: 101 CSV sources','Filename groups; not verified roads'),
                          (.365,'PVS: 9 acquisitions','Vehicle and driver are confounded'),
                          (.695,'Mexico: 2 route families','Documentation-derived groups')]:
        txt(ax,x,.825,title,9.0,weight='bold')
        txt(ax,x,.779,sub,8.0,color=MUTED)
    rule(ax,.035,.965,.727)

    txt(ax,.035,.691,'1  Audit the evaluation unit',9.5,weight='bold')
    txt(ax,.035,.642,'RoadSens primary policy, RF100',8.4,color=MUTED)
    # Paired dots at a full 0--1 F1 scale, visibly different evaluation targets.
    p=fig.add_axes([.180,.385,.255,.185])
    p.set(xlim=(0,1.025),ylim=(-.35,1.35),xticks=[0,.5,1],yticks=[1,0],
          yticklabels=['Random rows','Recording\nholdout'])
    p.tick_params(axis='both',length=0,labelsize=8.2,pad=4)
    p.xaxis.set_ticks_position('bottom')
    for s in ('top','left','right'): p.spines[s].set_visible(False)
    p.plot([.9945],[1],marker='o',ms=5,color=GRAY)
    p.plot([.4289],[0],marker='s',ms=5,color=TEAL)
    p.text(.9945,1.18,'0.9945',fontsize=9,ha='right',va='bottom',color=GRAY)
    p.text(.4289,.18,'0.4289',fontsize=9,ha='center',va='bottom',color=TEAL)
    p.set_xlabel('Macro-F1 (higher is better)',fontsize=8.3,labelpad=4)
    txt(ax,.035,.301,'Different prediction targets;',8.3,color=MUTED)
    txt(ax,.035,.264,'the score gap alone is not causal evidence.',8.3,color=MUTED)

    txt(ax,.515,.691,'2  Separate the endpoints',9.5,weight='bold')
    txt(ax,.515,.642,'PVS primary grid: random to matched selection',8.2,color=MUTED)
    txt(ax,.515,.579,'Selection: scenario outer macro-F1',8.6)
    txt(ax,.515,.532,'0.7640  →  0.8359',12,color=TEAL,weight='bold')
    txt(ax,.515,.458,'Estimation: mean absolute error (lower is better)',8.2)
    txt(ax,.515,.414,'Vehicle/driver   0.0530  →  0.0445',9)
    txt(ax,.515,.367,'Scenario           0.1696  →  0.1357',9)
    txt(ax,.515,.301,'Required: MAE reduction ≥ 0.02 on both axes.',8.2)
    txt(ax,.515,.264,'Declared two-axis criterion: NOT MET',8.8,color=RISK,weight='bold')

    rule(ax,.035,.965,.217)
    txt(ax,.035,.185,'3  Retain the contrary evidence',9.5,weight='bold')
    txt(ax,.035,.135,'Disjoint windows and post hoc common-budget diagnostics: MAE worsens on both PVS axes.',8.2)
    txt(ax,.035,.094,'Mexico released subset: both selectors choose the same RF; selection advantage is zero.',8.2)
    txt(ax,.035,.043,'Bounded evidence, not a universal validation improvement. Same PVS acquisitions; no population intervals.',8.0,color=MUTED)
    export(fig,'graphical_abstract_v10')


def workflow():
    fig,ax=canvas(104)
    txt(ax,.035,.965,'Evaluate the selected system and its estimate separately',10.7,weight='bold')
    txt(ax,.035,.905,'PVS: binary speed-bump task; fixed candidates; nested evaluation within nine acquisitions',8.4,color=MUTED)
    txt(ax,.035,.820,'Acquisition structure',9.0,weight='bold')
    # All nine cells are acquisition units, not nine independent vehicles.
    x0,y0,w,h=.100,.472,.197,.225
    for j in range(3):
        txt(ax,x0+(j+.5)*w/3,.736,f'S{j+1}',8.2,ha='center')
    for i in range(3):
        txt(ax,.079,y0+(2-i+.5)*h/3,f'V{i+1}/D{i+1}',8.0,ha='right',va='center')
        for j in range(3):
            ax.add_patch(Rectangle((x0+j*w/3,y0+(2-i)*h/3),w/3-.006,h/3-.010,
                         facecolor='#F1F4F2',edgecolor=GRAY,lw=.6))
    txt(ax,.035,.420,'S: scenario; V/D: vehicle/driver pair',8.0,color=MUTED)
    txt(ax,.035,.376,'Hold out one row OR one column.',8.2)
    txt(ax,.035,.334,'Repeat over 3 domains per axis.',8.2)

    arrow(ax,(.305,.676),(.382,.737))
    arrow(ax,(.305,.565),(.382,.505))
    box(ax,.385,.665,.273,.147,'Development acquisitions\nRandom / trip / matched CV')
    box(ax,.385,.448,.273,.123,'Outer-domain acquisitions\nExcluded from inner selection')
    arrow(ax,(.661,.739),(.708,.739))
    box(ax,.713,.665,.254,.147,'Select one of four candidates\nSave its inner F1 estimate',accent=True)
    arrow(ax,(.840,.662),(.840,.574))
    box(ax,.713,.448,.254,.123,'Refit on development data\nPredict the outer domain')
    arrow(ax,(.661,.509),(.708,.509))

    # Two explicitly distinct endpoints; labels define aggregation in caption.
    arrow(ax,(.840,.444),(.840,.296))
    ax.plot([.800,.800,.518],[.444,.369,.369],color=MUTED,lw=.8)
    arrow(ax,(.518,.369),(.518,.296))
    box(ax,.385,.135,.273,.157,'Selection endpoint\nOuter macro-F1\nHigher is better',accent=True)
    box(ax,.713,.135,.254,.157,'Estimation endpoint\n|inner F1 − outer F1|\nLower is better',accent=True)
    txt(ax,.035,.074,'Mean absolute estimation error is averaged over the three outer domains of an axis.',8.2)
    txt(ax,.035,.032,'Both axes reuse the same nine acquisitions; windows, candidates and fits do not add independent samples.',8.0,color=MUTED)
    export(fig,'pvs_evaluation_workflow_v10')


def specification():
    base=HERE.parents[1]
    # Content hashes document the exact manuscript used for numerical labels.
    manuscript=base/'tex_v9'/'main_body.tex'
    spec_file=HERE/'visual_spec_v10.json'
    # Standalone release bundles keep the authority hash, not an entire old draft.
    authority_hash=(hashlib.sha256(manuscript.read_bytes()).hexdigest() if manuscript.exists()
                    else json.loads(spec_file.read_text(encoding='utf-8'))['source_manuscript_sha256'])
    obj={
      'source_manuscript':'manuscript/tex_v9/main_body.tex (historical authority)',
      'source_manuscript_sha256':authority_hash,
      'claim':'Selecting a stronger model and estimating its performance accurately are separate endpoints.',
      'visual_forms':{'graphical_abstract':'Evidence overview with a small full-scale dot chart and exact values',
                      'workflow':'Acquisition structure and nested evaluation diagram',
                      'pvs_endpoints':'Paired dots with exact four-decimal values, separate metric axes'},
      'dimensions_mm':{'graphical_abstract':[178,118],'workflow':[178,104],'pvs_endpoints':[178,120]},
      'metric_units':'Macro-F1 and absolute macro-F1 estimation error; unit interval',
      'aggregation':'RoadSens: five-fold means. PVS: three outer-domain means per axis.',
      'uncertainty':'Descriptive summaries; do not manufacture population intervals from three domains. Same nine PVS acquisitions.',
      'baseline':'Random-window selector; matched-domain selector is focal, not claimed universally superior.',
      'criterion':'At least 0.02 MAE reduction versus random selection on BOTH primary axes; remains failed.',
      'scope':'RoadSens source filenames are not verified roads; PVS V/D confounded; Mexico two documentation-derived route families.',
      'references':['GPT-6 Astra governing grammar (local skill core)',
                    'OAI-VIS-052 MLE-bench evaluation workflow',
                    'OAI-VIS-001 sparse direct labeling',
                    'OAI-VIS-002 compact comparison hierarchy'],
      'outputs':['PDF','SVG','300 dpi PNG','editable Python','TeX wrappers'],
      'science_changes':'None: no new models, splits, thresholds, intervals or labels.'}
    (HERE/'visual_spec_v10.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    specification()
    graphical_abstract()
    workflow()
    print('Wrote graphical abstract and evaluation workflow (PDF/SVG/PNG/grayscale).')
