from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import pymupdf as fitz
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
(ROOT/'editorial/qa').mkdir(parents=True,exist_ok=True)
FONT='Times New Roman'
font_manager.findfont(FONT,fallback_to_default=False)
plt.rcParams.update({'font.family':FONT,'font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
INK='#172323'; BLUE='#416A9B'; TEAL='#26796B'; ORANGE='#C98034'; GREY='#69716F'
W,H=178,110
fig=plt.figure(figsize=(160/25.4,H/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
texts=[]
def text(x,y,s,size=8.5,bold=False,color=INK,ha='center'):
    t=ax.text(x,y,s,fontsize=size,fontfamily=FONT,weight='bold' if bold else 'normal',color=color,
              ha=ha,va='center',linespacing=1.15);texts.append(t);return t
def box(x,y,w,h,fill,edge='none',rounding=2,lw=.7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',fc=fill,ec=edge,lw=lw))
def line(points,color=INK,lw=.8):ax.plot(*zip(*points),color=color,lw=lw,solid_capstyle='round')
def arrow(points,color=INK,lw=.9):
    if len(points)>2:line(points[:-1],color,lw)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=7,
                               color=color,lw=lw,shrinkA=1,shrinkB=1))
def grid(x,y,rows,cols,w,h,selected=None):
    for r in range(rows):
        for c in range(cols):
            focal=selected=='all' or r==selected
            ax.add_patch(Rectangle((x+c*w/cols,y+r*h/rows),w/cols-.7,h/rows-.7,
                                   fc=TEAL if focal else 'white',ec=BLUE,lw=.5))
def wave(x,y,w,h):
    t=np.linspace(0,1,180);v=.19*np.sin(16*np.pi*t)+.48*np.exp(-((t-.57)/.07)**2)*np.sin(65*np.pi*t)
    ax.plot(x+t*w,y+h/2+v*h,color=BLUE,lw=.8)

text(89,5.5,'Model selection and performance estimation can diverge',11.3,True)
box(3,14,39,78,'#E7EFF9');box(47,14,66,78,'#FFF1DF');box(118,14,57,78,'#E4F3F0')
text(22.5,20,'Acquisition evidence',9.1,True)
text(80,20,'Nested model selection',9.3,True)
text(146.5,20,'Two evaluation outcomes',9.1,True)

# Separate source tasks; only the PVS acquisition group enters the nested diagram.
for k in [2,1,0]:
    ax.add_patch(Rectangle((7+k*1.2,28-k*1.2),11,9,fc='white',ec=BLUE,lw=.6))
wave(8,29,9,6)
text(29,29,'RoadSens',8.7,True);text(29,35,'101 CSVs',8.3)
line([(7,42),(38,42)],BLUE,.45)
grid(7,48,3,3,13,15,0)
text(30,50,'PVS',9,True);text(30,57,'9 acquisitions',8.1)
text(22,67,'3 V/D pairs × 3 scenarios',8.0)
line([(7,73),(38,73)],BLUE,.45)
line([(7,83),(11,78),(15,81),(19,76)],BLUE,.9)
line([(7,86),(11,81),(15,84),(19,79)],TEAL,.9)
text(29,79,'Mexico',8.7,True);text(29,85,'2 route families',8.0)
arrow([(42,55),(47,55)],BLUE)

text(68,28,'Develop: 6',8.2,True);text(100,28,'Test: 3',8.2,True)
grid(52,32,1,6,32,5)
grid(92,32,1,3,17,5,'all')
arrow([(68,38),(68,42)])
text(83,40,'Compare selectors',8.1)
for x,w,s in [(51,18,'Random'),(72,15,'Trip'),(90,19,'Matched')]:
    box(x,44,w,10,'white',BLUE,1)
    text(x+w/2,49,s,8.2)
    line([(x+w/2,54),(x+w/2,58)],BLUE,.65)
line([(60,58),(99.5,58)],BLUE,.65)
arrow([(80,58),(80,62)],BLUE)
box(51,63,58,14,'white',ORANGE,1)
grid(54,66,2,2,8,8)
text(84,67,'Same four candidates',8.5,True)
text(84,73,'RF / HGB × IMU ± context',8.0)
arrow([(80,77),(80,81)],ORANGE)
box(51,82,58,8,'#FCE1BB',ORANGE,1)
text(80,86,'Save inner F1; refit on 6',8.1)
arrow([(109,86),(115.5,86),(115.5,58),(120,58)],TEAL)
line([(109,34),(115.5,34),(115.5,58)],TEAL,.8)
ax.plot(115.5,58,'o',ms=2.3,mfc='white',mec=TEAL,mew=.8)

# Paired dots show descriptive three-domain scenario means, with explicit axis ranges.
text(146.5,28,'PVS · scenario holdout',8.2,True)
text(146.5,34,'Disjoint windows',8.1)
ax.plot(127,40,'o',color=GREY,ms=3.3);text(130,40,'Random',8.0,ha='left')
ax.plot(151,40,'s',color=TEAL,ms=3.3);text(154,40,'Matched',8.0,ha='left')
def pair(y,title,lo,hi,a,b):
    text(146.5,y,title,8.8,True)
    x=lambda v:126+(v-lo)/(hi-lo)*43
    yy=y+10
    line([(126,yy+5),(169,yy+5)],GREY,.45)
    for xx,v in [(126,lo),(169,hi)]:
        line([(xx,yy+4.4),(xx,yy+5.6)],GREY,.45)
        text(xx,yy+9,f'{v:.2f}',8)
    line([(x(a),yy),(x(b),yy)],GREY,1)
    ax.plot(x(a),yy,'o',ms=4,color=GREY)
    ax.plot(x(b),yy,'s',ms=4,color=TEAL)
    text(x(a),yy+2.2,f'{a:.4f}',8.2,ha='right')
    text(x(b),yy-3.1,f'{b:.4f}',8.2,color=TEAL,ha='left')
pair(47,'Outer macro-F1 ↑',.75,.85,.7904,.8168)
pair(70,'Estimation MAE ↓',.08,.15,.1069,.1290)

box(3,96,172,11,'#E8EDF7',rounding=2)
text(89,99.8,'Better selected models ≠ more accurate performance estimates',10.0,True)
text(89,104.4,'Original two-axis MAE criterion not met · same nine PVS acquisitions',8.0)

fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for t in texts:
    b=t.get_window_extent(renderer)
    if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1):outside.append(t.get_text())
assert not outside,outside
for ext in ['pdf','svg']:
    fig.savefig(ROOT/f'figures/graphical_abstract_v17.{ext}',facecolor='white')
plt.close(fig)
p=fitz.open(ROOT/'figures/graphical_abstract_v17.pdf')[0]
p.get_pixmap(dpi=350,alpha=False).save(ROOT/'figures/graphical_abstract_v17.png')
p.get_pixmap(dpi=96,alpha=False).save(ROOT/'editorial/qa/final_size.png')
Image.open(ROOT/'figures/graphical_abstract_v17.png').convert('L').save(ROOT/'editorial/qa/grayscale.png')
fonts=sorted({f[3] for f in p.get_fonts()});assert all('TimesNewRoman' in f for f in fonts)
assert not p.get_images()
spec={'claim':'Selected-model performance and estimation accuracy can diverge',
 'size_mm':[160,H],'font':FONT,'actual_fonts':fonts,'minimum_font_pt':8,
 'style_references':'User supplied LoRA, RLHF/DPO and comparison-diagram examples',
 'source':'ijpe_v15/figures/pvs_endpoints_ijpe.json; main_body and article_text',
 'values':{'scenario_disjoint':{'random_outer_F1':.7904,'matched_outer_F1':.8168,'random_MAE':.1069,'matched_MAE':.1290}},
 'aggregation':'Three outer-domain means per axis; no population intervals',
 'semantics':'Separate source tasks; no cross-dataset transfer. V/D = vehicle/driver pair. Waveforms and matrix highlights are schematic.',
 'geometry_outside_text':outside,'render_review':'pending'}
(ROOT/'editorial/graphical_abstract_spec_v17.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print(json.dumps({'fonts':fonts,'outside_text':outside},indent=2))
