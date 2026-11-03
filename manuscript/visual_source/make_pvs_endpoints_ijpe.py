from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib import font_manager
from PIL import Image
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'figures'; OUT.mkdir(parents=True,exist_ok=True)
font_manager.findfont('Times New Roman', fallback_to_default=False)
rows=[('Primary','Vehicle/driver',.8679,.8679,.0530,.0445),('Primary','Scenario',.7640,.8359,.1696,.1357),('Disjoint windows','Vehicle/driver',.8380,.8522,.0489,.0535),('Disjoint windows','Scenario',.7904,.8168,.1069,.1290),('Common budget (post hoc)','Vehicle/driver',.8468,.8526,.0140,.0267),('Common budget (post hoc)','Scenario',.7083,.6951,.1627,.1724)]
cols=['group','domain','outer_macro_f1_random','outer_macro_f1_matched','estimation_mae_random','estimation_mae_matched']
data={'description':'Common-budget rows use the common-budget outer target, not the historical full target.','columns':cols,'rows':[dict(zip(cols,r)) for r in rows]}
(OUT/'pvs_endpoints_ijpe.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(160/25.4,110/25.4),gridspec_kw={'width_ratios':[1,1],'wspace':.5})
fig.patch.set_facecolor('white'); ys=[8,6.6,4.7,3.3,1.4,0]; colors={'Random':'#666866','Matched':'#1F6F5F'}
for ax in axs: ax.set_facecolor('white')
for j,(ax,title,xlim,k1,k2,xlabel) in enumerate([(axs[0],'Outer macro-F1 (higher is better)',(.65,.96),2,3,'Outer macro-F1 (narrow range)'),(axs[1],'Estimation MAE (lower is better)',(0,.20),4,5,'Estimation MAE')]):
 for yi,row in zip(ys,rows):
  a,b=row[k1],row[k2]; ax.plot([a,b],[yi,yi],color='#B8BCB9',lw=.8,zorder=1)
  ax.scatter(a,yi,s=31,marker='o',color=colors['Random'],zorder=3); ax.scatter(b,yi,s=27,marker='s',facecolors=colors['Matched'] if a!=b else 'none',edgecolors=colors['Matched'],linewidths=1.1,zorder=4)
  ax.annotate(f'{a:.4f}',(a,yi),xytext=(0,-5),textcoords='offset points',ha='center',va='top',fontsize=8.3,color=colors['Random'],clip_on=False)
  ax.annotate(f'{b:.4f}',(b,yi),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=8.3,color=colors['Matched'],clip_on=False)
 ax.set_xlim(*xlim); ax.set_ylim(-.8,9.2); ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in rows] if j==0 else []); ax.set_xlabel(xlabel); ax.set_title(title,pad=8)
 ax.grid(axis='x',color='#E7E9E7',lw=.6); ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y',length=0,pad=4); ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for yy,label in [(8.85,'Primary'),(5.55,'Disjoint windows'),(2.25,'Common budget (post hoc)')]:
 axs[0].text(.035,yy,label,transform=blended_transform_factory(fig.transFigure,axs[0].transData),ha='left',va='center',fontsize=8.5,weight='bold',clip_on=False)
handles=[Line2D([],[],marker='o',linestyle='None',color=colors['Random'],label='Random-window selector'),Line2D([],[],marker='s',linestyle='None',color=colors['Matched'],label='Matched selector')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.58,.965),ncol=2,frameon=False,fontsize=8.5,handletextpad=.5,columnspacing=1.3)
fig.text(.035,.995,'Model selection and estimation accuracy are distinct endpoints',ha='left',va='top',fontsize=10.5)
fig.subplots_adjust(left=.18,right=.97,top=.85,bottom=.22)
for ext in ('pdf','svg','png'): fig.savefig(OUT/f'pvs_endpoints_ijpe.{ext}',dpi=300,facecolor='white')
Image.open(OUT/'pvs_endpoints_ijpe.png').convert('L').save(OUT/'pvs_endpoints_ijpe_grayscale.png')
fig.canvas.draw(); renderer=fig.canvas.get_renderer(); W,H=fig.canvas.get_width_height(); issues=[]
for t in fig.findobj(match=matplotlib.text.Text):
 if not t.get_visible() or not t.get_text(): continue
 b=t.get_window_extent(renderer)
 if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H: issues.append((t.get_text(),tuple(round(v,1) for v in [b.x0,b.y0,b.x1,b.y1])))
print('bbox canvas',W,H,'issues',issues)
if issues: raise RuntimeError('Figure text bounds failed')
