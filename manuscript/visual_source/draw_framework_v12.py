"""Editable vector architecture of the existing evaluation audit.
Glyphs show data structures, not empirical signal values or new results.
No model fitting, new split generation, raster artwork, or external images.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
INK='#182321'; GRAY='#596460'; RULE='#CCD4D0'; TEAL='#1F6F5F'; TINT='#E4EFEB'; RISK='#B5473C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.0,'pdf.fonttype':42,
                      'svg.fonttype':'none','text.color':INK})
fig=plt.figure(figsize=(178/25.4,140/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')

def text(x,y,s,size=8.0,bold=False,color=INK,ha='left',va='top',**kw):
    return ax.text(x,y,s,fontsize=size,weight='bold' if bold else 'normal',
                   color=color,ha=ha,va=va,linespacing=1.22,**kw)

def line(points,color=RULE,lw=.8,ls='-'):
    xs,ys=zip(*points); ax.plot(xs,ys,color=color,lw=lw,ls=ls,solid_capstyle='round')

def arrow(a,b,color=GRAY,style='-|>',ls='-',lw=.85):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=7,
                               lw=lw,color=color,linestyle=ls,shrinkA=1,shrinkB=1))

def rect(x,y,w,h,accent=False,fill=False,dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.003,rounding_size=0.005',
        fc=TINT if fill else 'white',ec=TEAL if accent else RULE,lw=.85,
        linestyle=(0,(3,2)) if dashed else '-'))

def box(x,y,w,h,s,accent=False,fill=False,size=8.0):
    rect(x,y,w,h,accent,fill)
    text(x+w/2,y+h/2,s,size=size,color=TEAL if accent else INK,ha='center',va='center')

def waveform(x,y,w,h,color=TEAL):
    t=np.linspace(0,1,160)
    signal=.16*np.sin(15*np.pi*t)+.42*np.exp(-((t-.54)/.07)**2)*np.sin(65*np.pi*t)
    line([(x,y+h/2),(x+w,y+h/2)],color=RULE,lw=.5)
    ax.plot(x+w*t,y+h*(.5+signal),color=color,lw=.75)

def files(x,y,w,h,n=3):
    for k in range(n-1,-1,-1):
        dx,dy=.009*k,.008*k
        ax.add_patch(Rectangle((x+dx,y+dy),w,h,fc='white',ec=GRAY,lw=.6))
    waveform(x+.007,y+.01,w-.014,h-.02)

def matrix(x,y,nr,nc,w,h,held=None):
    for r in range(nr):
        for c in range(nc):
            focal=held=='all' or (held is not None and (r==held if isinstance(held,int) else c==held[0]))
            ax.add_patch(Rectangle((x+c*w/nc,y+(nr-r-1)*h/nr),w/nc-.003,h/nr-.003,
                         fc=TEAL if focal else '#E7ECE9',ec='white',lw=.3))

def vector(x,y,w,h,n=6):
    for i in range(n):
        ax.add_patch(Rectangle((x,y+i*h/n),w,h/n-.003,fc=TINT if i%2 else '#F1F4F2',ec=GRAY,lw=.4))

def tree(x,y,w,h):
    nodes=[(.5,1),(.2,.55),(.8,.55),(0,0),(.4,0),(.7,0),(1,0)]
    for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
        line([(x+w*nodes[a][0],y+h*nodes[a][1]),(x+w*nodes[b][0],y+h*nodes[b][1])],GRAY,.6)
    for u,v in nodes: ax.plot(x+w*u,y+h*v,'o',ms=1.7,mfc='white',mec=GRAY,mew=.6)

text(.025,.978,'Road-sensing evaluation framework',11,bold=True)
text(.025,.936,'Source audit',8.5,bold=True,color=GRAY)
text(.225,.936,'Representations',8.5,bold=True,color=GRAY)
text(.455,.936,'Evaluation design',8.5,bold=True,color=GRAY)
text(.685,.936,'Fixed learners',8.5,bold=True,color=GRAY)
text(.864,.936,'Readouts',8.5,bold=True,color=GRAY)
line([(.025,.915),(.975,.915)],lw=.7)

# RoadSens branch: audit -> label/feature variants -> evaluation units -> learners.
text(.025,.887,'RoadSens',9,bold=True)
files(.035,.785,.080,.060)
text(.025,.769,'101 CSV sources',8)
text(.025,.733,'Hashes + content\n2 label policies',8,color=GRAY)
arrow((.142,.811),(.215,.811))
rect(.222,.772,.184,.088)
matrix(.235,.793,4,4,.043,.046)
text(.288,.848,'Full numeric\nIMU / context\nTemporal',8)
arrow((.410,.811),(.442,.811))
for i in range(4): matrix(.451,.786+i*.018,1,7,.061,.011,held=(i%7,))
text(.523,.855,'Random rows\nPurged time\nRecording\nContent / spatial',8)
arrow((.646,.811),(.675,.811))
tree(.690,.812,.082,.041)
text(.730,.787,'RF · ET\nHGB · LR',8,ha='center')
arrow((.790,.811),(.840,.811))
text(.850,.855,'Macro-F1\nBalanced acc.\nConditional\nintervals',8)
line([(.025,.685),(.975,.685)],lw=.7)

# PVS branch: explicit acquisition structure, input representation and nested selection.
text(.025,.659,'PVS',9,bold=True)
text(.025,.626,'9 acquisitions',8)
matrix(.040,.514,3,3,.087,.075,held=0)
text(.025,.491,'3 V/D pairs\n× 3 scenarios',8)
arrow((.139,.550),(.170,.550))
rect(.177,.507,.117,.082)
waveform(.185,.523,.100,.050)
for x in [.191,.217,.243]:
    ax.add_patch(Rectangle((x,.520),.030,.055,fc='none',ec=TEAL,lw=.55))
text(.178,.490,'100-row windows\nStride 25 / 100',8)
arrow((.300,.550),(.326,.550))
vector(.336,.519,.032,.066)
vector(.376,.541,.018,.044,n=3)
text(.328,.490,'30 IMU\n+ 3 context',8)
arrow((.400,.550),(.431,.550))

# The outlined region is the selected-system pipeline in one outer fold.
rect(.436,.309,.382,.336,accent=True,dashed=True)
text(.448,.632,'Nested selection',9,bold=True,color=TEAL)
text(.448,.603,'Outer axis: vehicle/driver OR scenario',8)
text(.448,.575,'Development: 6',8,color=GRAY)
matrix(.448,.536,1,6,.141,.020)
text(.692,.575,'Test: 3',8,color=GRAY)
matrix(.692,.536,1,3,.108,.020,held='all')
# Data enter the fixed outer partition, then development alone drives fitting.
arrow((.432,.550),(.444,.550))
arrow((.514,.531),(.514,.511))
box(.445,.387,.149,.120,'Train-only impute\nInner CV\nRandom / trip /\nmatched domain',accent=True,size=8)
arrow((.600,.455),(.619,.455))
for x,y,label in [(.626,.460,'RF\nIMU'),(.716,.460,'HGB\nIMU'),
                   (.626,.398,'RF\n+context'),(.716,.398,'HGB\n+context')]:
    box(x,y,.077,.053,label,size=8)
arrow((.699,.394),(.699,.357))
box(.448,.320,.354,.034,'Select max inner OOF F1 → refit',accent=True,size=8)

# A frozen outer test is used only after candidate selection/refitting.
line([(.803,.546),(.837,.546),(.837,.505)],GRAY,.8)
line([(.804,.337),(.837,.337),(.837,.505)],TEAL,.9)
ax.plot(.837,.505,'o',ms=2.7,mfc='white',mec=TEAL,mew=.8)
arrow((.840,.505),(.852,.505),color=TEAL)
text(.857,.620,'Two endpoints',8.5,bold=True)
box(.857,.500,.118,.073,'Outer F1 ↑',accent=True,fill=True,size=8.5)
box(.857,.385,.118,.082,'MAE ↓\n|inner − outer|',accent=True,fill=True,size=8)
line([(.837,.505),(.844,.505),(.844,.426)],TEAL,.8)
arrow((.844,.426),(.852,.426),color=TEAL)
text(.858,.366,'Mean over\n3 outer domains',8,color=GRAY)
text(.025,.392,'Frozen split membership',8,color=GRAY)
line([(.190,.285),(.802,.285)],color=GRAY,lw=.7,ls='--')
text(.190,.268,'Sensitivity checks on the same acquisitions',8,bold=True,color=GRAY)
text(.190,.241,'Disjoint windows · context blocks · common budget',8,color=GRAY)
text(.848,.300,'Both axes require\nMAE reduction\n≥ 0.02',8)
text(.858,.214,'Not met',8,bold=True,color=RISK)
line([(.025,.192),(.975,.192)],lw=.7)

# Bounded additional-source branch. No model transfer between source lanes.
text(.025,.172,'Mexico',9,bold=True)
line([(.035,.098),(.060,.125),(.088,.104),(.120,.133)],GRAY,1.0)
line([(.035,.081),(.060,.108),(.088,.087),(.120,.116)],TEAL,1.0)
text(.025,.066,'2 route families',8)
arrow((.143,.112),(.215,.112))
box(.222,.067,.185,.085,'10-position windows\n36 acceleration\nsummaries',size=8)
arrow((.411,.112),(.442,.112))
box(.449,.067,.190,.085,'Route-family holdout\nInner CV: random /\nwhole trip',size=8)
arrow((.644,.112),(.678,.112))
box(.685,.079,.108,.063,'RF / HGB',size=8)
arrow((.798,.112),(.844,.112))
text(.850,.145,'Same RF selected\nSelection\ncontrast = 0',8)
text(.025,.024,'Separate source tasks; no cross-dataset model transfer. V/D = vehicle/driver pair.',8,color=GRAY)

# Exact canvas size retained for direct insertion into the manuscript.
fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[]
for t in fig.findobj(matplotlib.text.Text):
    if not t.get_visible() or not t.get_text(): continue
    b=t.get_window_extent(renderer)
    if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5:
        outside.append(t.get_text())
if outside: raise RuntimeError(f'Clipped labels: {outside}')
for fmt in ('pdf','svg','png'):
    fig.savefig(OUT/f'framework_architecture_v12.{fmt}',dpi=300,facecolor='white')
with Image.open(OUT/'framework_architecture_v12.png') as im:
    im.convert('L').save(OUT/'framework_architecture_v12_grayscale.png')
spec={'claim':'Architecture separates source tasks, fixed validation/selection, and classification versus estimation endpoints.',
      'dimensions_mm':[178,140],'minimum_font_pt':8,'format':'Editable vector PDF/SVG from Python',
      'scope':'Existing evaluation audit, not a new classifier or universally validated evaluation algorithm',
      'sources':['tex_v10/main_body.tex sections 3.1–3.13, 4.11–4.12'],
      'data_glyphs':'Schematic waveforms, windows, matrices and model trees; no empirical scores encoded',
      'preserved':'9 PVS acquisitions, 6 development/3 test per outer fold, 4 fixed candidates, 3 domains/axis; primary criterion failed',
      'qa':{'outside_labels':outside}}
(ROOT/'visual_source/architecture_spec_v12.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print('Framework architecture exported (PDF/SVG/PNG/grayscale).')
