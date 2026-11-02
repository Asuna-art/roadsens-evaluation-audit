"""Times typography and balanced connector geometry for the audit architecture.

All data structures are schematic; no experiments or frozen splits are changed.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
INK='#182321'; STROKE='#58635F'; RULE='#CCD4D0'; TEAL='#1F6F5F'; TINT='#E4EFEB'; RISK='#A74336'
WIDTH,HEIGHT=178,140
FONT='Times New Roman'
font_paths={w:font_manager.findfont(font_manager.FontProperties(family=FONT,weight=w),
                                  fallback_to_default=False) for w in ('normal','bold')}
plt.rcParams.update({'font.family':FONT,'font.size':8.5,'pdf.fonttype':42,
                     'ps.fonttype':42,'svg.fonttype':'none','text.color':INK})
spec={'version':13,'dimensions_mm':[WIDTH,HEIGHT],'font':FONT,'resolved_fonts':font_paths,
      'minimum_font_pt':8.0,'ordinary_font_pt':8.5,
      'claim':'The existing audit separates source tasks, nested selection, outer refitting, and two evaluation endpoints.',
      'changes':['Times regular/bold; no font fallback.',
                 'One PVS entry connector; equal output-box heights and centered symmetric fork.',
                 'Aligned stage centers and consistent arrow clearances.',
                 'Explanatory gray prose moved to the manuscript caption.'],
      'evidence':'All counts, model families, feature variants, endpoints and failure are retained from v12.',
      'format':'Native vector PDF/SVG, editable Python, 300 dpi PNG'}
(ROOT/'visual_source/architecture_spec_v13.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')

fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
text_records=[]; label_containers=[]; arrow_records=[]

def text(x,y,s,size=8.5,bold=False,color=INK,ha='left',va='top',container=None):
    t=ax.text(x,y,s,fontsize=size,weight='bold' if bold else 'normal',fontfamily=FONT,
              color=color,ha=ha,va=va,linespacing=1.18)
    text_records.append(t)
    if container is not None: label_containers.append((t,container))
    return t

def line(points,color=RULE,lw=.7,ls='-'):
    xs,ys=zip(*points); ax.plot(xs,ys,color=color,lw=lw,ls=ls,solid_capstyle='round')

def arrow(a,b,color=STROKE,clearance=2.0,lw=.8):
    # Physical point clearances, independent of final raster resolution.
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,
                     lw=lw,color=color,shrinkA=clearance,shrinkB=clearance))
    arrow_records.append({'from':list(a),'to':list(b),'clearance_pt':clearance})

def rect(x,y,w,h,accent=False,fill=False,dashed=False,lw=.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.005',
        fc=TINT if fill else 'white',ec=TEAL if accent else RULE,lw=lw,
        linestyle=(0,(3,2)) if dashed else '-'))

def box(x,y,w,h,s,accent=False,fill=False,size=8.3):
    rect(x,y,w,h,accent,fill)
    return text(x+w/2,y+h/2,s,size=size,ha='center',va='center',
                color=TEAL if fill else INK,container=(x,y,w,h))

def waveform(x,y,w,h,color=TEAL):
    t=np.linspace(0,1,160)
    signal=.16*np.sin(15*np.pi*t)+.42*np.exp(-((t-.54)/.07)**2)*np.sin(65*np.pi*t)
    line([(x,y+h/2),(x+w,y+h/2)],color=RULE,lw=.45)
    ax.plot(x+w*t,y+h*(.5+signal),color=color,lw=.75)

def files(x,y,w,h,n=3):
    for k in range(n-1,-1,-1):
        ax.add_patch(Rectangle((x+.009*k,y+.008*k),w,h,fc='white',ec=STROKE,lw=.6))
    waveform(x+.007,y+.010,w-.014,h-.020)

def matrix(x,y,nr,nc,w,h,held=None):
    for r in range(nr):
        for c in range(nc):
            focal=held=='all' or (held is not None and (r==held if isinstance(held,int) else c==held[0]))
            ax.add_patch(Rectangle((x+c*w/nc,y+(nr-r-1)*h/nr),w/nc-.003,h/nr-.003,
                         fc=TEAL if focal else '#E7ECE9',ec='white',lw=.3))

def vector(x,y,w,h,n=6):
    for i in range(n):
        ax.add_patch(Rectangle((x,y+i*h/n),w,h/n-.003,
                     fc=TINT if i%2 else '#F1F4F2',ec=STROKE,lw=.4))

def tree(x,y,w,h):
    nodes=[(.5,1),(.2,.55),(.8,.55),(0,0),(.4,0),(.7,0),(1,0)]
    for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
        line([(x+w*nodes[a][0],y+h*nodes[a][1]),(x+w*nodes[b][0],y+h*nodes[b][1])],STROKE,.6)
    for u,v in nodes: ax.plot(x+w*u,y+h*v,'o',ms=1.7,mfc='white',mec=STROKE,mew=.6)

# Typography: title, stage headings, body. Rules have explicit breathing room.
text(.025,.977,'Road-sensing evaluation framework',11.8,bold=True)
for x,label in [(.025,'Source audit'),(.225,'Representations'),(.455,'Evaluation design'),
                 (.685,'Fixed learners'),(.852,'Readouts')]:
    text(x,.927,label,8.8,bold=True)
line([(.025,.891),(.978,.891)])

# RoadSens: one shared horizontal centerline.
Y=.788
text(.025,.866,'RoadSens',9.6,bold=True)
files(.035,.748,.080,.060)
text(.025,.733,'101 CSV sources\n2 label policies')
arrow((.133,Y),(.222,Y))
rect(.222,.739,.184,.098)
matrix(.235,.762,4,4,.043,.052)
text(.288,Y,'Full numeric\nIMU / context\nTemporal',va='center',container=(.281,.739,.125,.098))
arrow((.406,Y),(.451,Y))
for i in range(4):matrix(.451,.747+i*.024,1,7,.061,.014,held=(i%7,))
text(.523,Y,'Random rows\nPurged time\nRecording\nContent / spatial',va='center')
arrow((.644,Y),(.691,Y))
tree(.691,.761,.082,.054)
text(.732,.750,'RF · ET\nHGB · LR',ha='center')
arrow((.773,Y),(.852,Y))
text(.852,Y,'Macro-F1\nBalanced acc.\nConditional\nintervals',va='center')
line([(.025,.665),(.978,.665)])

# PVS: acquisition matrix, windowing and vector labels share horizontal baselines.
Y=.533
text(.025,.640,'PVS',9.6,bold=True)
text(.025,.609,'9 acquisitions')
matrix(.040,.492,3,3,.087,.082,held=0)
text(.025,.476,'3 V/D pairs\n× 3 scenarios')
arrow((.127,Y),(.177,Y))
rect(.177,.488,.117,.090)
waveform(.185,.503,.100,.060)
for x in [.191,.217,.243]:
    ax.add_patch(Rectangle((x,.501),.030,.064,fc='none',ec=TEAL,lw=.55))
text(.177,.476,'100-row windows\nStride 25 / 100')
arrow((.294,Y),(.336,Y))
vector(.336,.490,.032,.086)
vector(.376,.523,.018,.053,n=3)
text(.328,.476,'30 IMU\n+3 context')

# A single entry connector replaces the former duplicated arrowheads.
rect(.425,.270,.388,.372,accent=True,dashed=True)
text(.440,.627,'Nested selection',9.5,bold=True,color=TEAL)
text(.799,.627,'V/D or scenario',8.0,ha='right')
text(.512,.560,'Development: 6',ha='center',va='bottom')
text(.754,.560,'Test: 3',ha='center',va='bottom')
matrix(.440,.521,1,6,.144,.024)
matrix(.718,.521,1,3,.072,.024,held='all')
arrow((.394,Y),(.440,Y))

box(.440,.3675,.153,.122,'Train-only impute\nInner CV\nRandom / trip /\nmatched domain',accent=True)
arrow((.512,.521),(.512,.4895))

# The candidate set is a panel with uniform cell dimensions and padding.
rect(.618,.352,.183,.153,lw=.6)
for x,y,label in [(.626,.434,'RF\nIMU'),(.716,.434,'HGB\nIMU'),
                   (.626,.359,'RF\n+context'),(.716,.359,'HGB\n+context')]:
    box(x,y,.077,.062,label)
arrow((.593,.4285),(.618,.4285))
box(.440,.283,.360,.044,'Select max inner OOF F1 → refit',accent=True)
arrow((.7095,.352),(.7095,.327))

# Test data and the refitted system meet at one junction. Both outputs are centered.
JOIN_X,JOIN_Y,SPINE_X=.827,.4725,.839
line([(.790,.533),(JOIN_X,.533),(JOIN_X,JOIN_Y)],STROKE,.8)
line([(.800,.305),(JOIN_X,.305),(JOIN_X,JOIN_Y)],TEAL,.85)
ax.plot(JOIN_X,JOIN_Y,'o',ms=2.7,mfc='white',mec=TEAL,mew=.8)
line([(JOIN_X,JOIN_Y),(SPINE_X,JOIN_Y)],TEAL,.8)
line([(SPINE_X,.533),(SPINE_X,.412)],TEAL,.8)
text(.852,.621,'Two endpoints',8.8,bold=True)
box(.852,.488,.126,.090,'Outer F1 ↑',accent=True,fill=True,size=8.7)
box(.852,.367,.126,.090,'MAE ↓\n|inner − outer|',accent=True,fill=True)
arrow((SPINE_X,.533),(.852,.533),color=TEAL,clearance=1.5)
arrow((SPINE_X,.412),(.852,.412),color=TEAL,clearance=1.5)
text(.915,.329,'Criterion not met',8.3,bold=True,color=RISK,ha='center')

# Remove prose-like gray annotations; conditions and aggregation move to the caption.
text(.177,.247,'Sensitivity checks',8.5,bold=True)
text(.177,.212,'Disjoint windows · context blocks · common budget',8.3)
line([(.025,.172),(.978,.172)])

# Mexico: all processing modules share identical top/bottom edges and centerline.
Y=.0905
text(.025,.148,'Mexico',9.6,bold=True)
line([(.035,.078),(.060,.105),(.088,.084),(.120,.113)],STROKE,.9)
line([(.035,.061),(.060,.088),(.088,.067),(.120,.096)],TEAL,.9)
text(.025,.041,'2 route families')
arrow((.133,Y),(.222,Y))
box(.222,.048,.185,.085,'10-position windows\n36 acceleration\nsummaries',size=8.5)
arrow((.407,Y),(.449,Y))
box(.449,.048,.190,.085,'Route-family holdout\nInner CV: random /\nwhole trip',size=8.5)
arrow((.639,Y),(.685,Y))
box(.685,.048,.108,.085,'RF / HGB',size=8.5)
arrow((.793,Y),(.852,Y))
text(.852,Y,'Same RF selected\nContrast: 0',va='center')

# Audit true font resolution, canvas clipping and padding against rendered extents.
fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[]; padding=[]
for t in text_records:
    b=t.get_window_extent(renderer)
    if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5:
        outside.append(t.get_text())
for t,(x,y,w,h) in label_containers:
    b=t.get_window_extent(renderer)
    lo=ax.transData.transform((x,y)); hi=ax.transData.transform((x+w,y+h))
    # Canvas renderer DPI is normally 100; convert actual margins to points.
    p=[(b.x0-lo[0])*72/fig.dpi,(hi[0]-b.x1)*72/fig.dpi,
       (b.y0-lo[1])*72/fig.dpi,(hi[1]-b.y1)*72/fig.dpi]
    padding.append({'label':t.get_text(),'left_right_bottom_top_pt':[round(v,2) for v in p]})
if outside:raise RuntimeError(f'Clipped labels: {outside}')
bad=[p for p in padding if min(p['left_right_bottom_top_pt'])<3.0]
if bad:raise RuntimeError(f'Insufficient label padding: {bad}')
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'framework_architecture_v13.{ext}',dpi=300,facecolor='white')
with Image.open(OUT/'framework_architecture_v13.png') as im:
    im.convert('L').save(OUT/'framework_architecture_v13_grayscale.png')
qa={'font_paths':font_paths,'canvas_clipping':outside,'label_padding':padding,
    'arrows':arrow_records,'output_box_heights_equal':True,
    'output_port_centers':{'outer_f1':.533,'estimation_mae':.412},
    'junction_y':JOIN_Y,'fork_symmetry':round(.533-JOIN_Y,4)==round(JOIN_Y-.412,4),
    'min_text_pt':min(t.get_fontsize() for t in text_records)}
(ROOT/'visual_source/drawing_qa_v13.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps({'fonts':font_paths,'clipped_labels':outside,'minimum_box_padding_pt':
      min(min(p['left_right_bottom_top_pt']) for p in padding),'fork_symmetry':qa['fork_symmetry']},indent=2))
