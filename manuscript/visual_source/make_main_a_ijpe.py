"""Build the RoadSens point plot from the shipped, unchanged TeX data table."""
from pathlib import Path
import re
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'tables/main_02.tex'
FIGDIR = HERE.parent / "figures"
font_manager.findfont("Times New Roman", fallback_to_default=False)
text = SOURCE.read_text(encoding="utf-8")
match = re.search(r"\\midrule(.*?)\\bottomrule", text, re.S)
if not match:
    raise RuntimeError("Could not find the full-numeric data rows")
rows = []
for line in match.group(1).splitlines():
    if '&' not in line: continue
    cells = [x.strip().rstrip('\\').strip() for x in line.split('&')]
    rows.append(cells)
if len(rows) != 8:
    raise RuntimeError(f"Expected 8 Table 1 data rows; found {len(rows)}")

models = ["RF100", "Extra Trees", "HistGB", "Logistic regression"]
policies = ["Primary", "Blank-as-normal"]
metric_cols = [2, 3, 4]
values = {}
for row in rows:
    policy, model = row[:2]
    values[(policy, model)] = [float(row[i]) for i in metric_cols]
for policy in policies:
    for model in models:
        if (policy, model) not in values:
            raise RuntimeError(f"Missing Table 1 row: {policy}, {model}")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
fig, axs = plt.subplots(1, 2, figsize=(160 / 25.4, 89 / 25.4), sharey=True)
colors = ["#D9DEE2", "#8D999F", "#1F6F5F"]
markers = ["o", "^", "s"]
labels = ["Random rows", "Purged time blocks", "Heldout recordings"]
offsets = [0.20, 0.0, -0.20]
ypos = list(range(len(models)-1, -1, -1))
for ax, policy, title in zip(axs, policies, ("(a) Primary labels", "(b) Blank-as-normal sensitivity")):
    for model, y in zip(models, ypos):
        for j, (color, marker, offset) in enumerate(zip(colors, markers, offsets)):
            ax.scatter(values[(policy, model)][j], y+offset, s=34, marker=marker,
                       color=color, edgecolor="#263238", linewidth=0.35, zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlabel("Macro-F1")
    ax.set_title(title, pad=8)
    ax.set_yticks(ypos, models)
    ax.set_ylim(-0.48, len(models)-0.52)
    ax.grid(axis="x", color="#D9DEE2", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#68757C")
handles = [Line2D([0], [0], marker=m, linestyle="", markersize=6.5,
                  markerfacecolor=c, markeredgecolor="#263238", markeredgewidth=.35,
                  label=l) for c, m, l in zip(colors, markers, labels)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.01),
           ncol=3, frameon=False, handletextpad=.4, columnspacing=1.5)
fig.subplots_adjust(left=.205, right=.985, top=.84, bottom=.27, wspace=.14)
for path in (FIGDIR / "main_a_ijpe.pdf", FIGDIR / "main_a_ijpe.png", FIGDIR / "main_a_ijpe.svg"):
    fig.savefig(path, dpi=400 if path.suffix == ".png" else None,
                facecolor="white", metadata={"Title": "RoadSens-4M evaluation scale point estimates"})
plt.close(fig)

# A grayscale proof checks that marker shape and lightness remain distinguishable.
from PIL import Image
im = Image.open(FIGDIR / "main_a_ijpe.png").convert("L")
im.save(FIGDIR / "main_a_ijpe_grayscale.png")
(FIGDIR/'main_a_ijpe_values.json').write_text(json.dumps({'source':'tables/main_02.tex', 'rows':rows},indent=2),encoding='utf-8')
print(FIGDIR / "main_a_ijpe.pdf")
