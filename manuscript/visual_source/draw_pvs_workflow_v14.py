"""Editable PVS workflow: Times typography, centered ports, orthogonal routes.

This schematic preserves the v10 protocol; it does not run any experiments.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures'
WIDTH, HEIGHT = 178, 104
FONT = 'Times New Roman'
INK, TEAL, STROKE, RULE = '#182321', '#1F6F5F', '#58635F', '#CCD4D0'
FONT_PATHS = {w: font_manager.findfont(font_manager.FontProperties(family=FONT, weight=w),
              fallback_to_default=False) for w in ('normal', 'bold')}
plt.rcParams.update({'font.family': FONT, 'font.size': 9.0, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
spec = {
    'version': 14, 'claim': 'The same selected candidate supplies outer performance and an inner-versus-outer estimation error.',
    'visual_form': 'Acquisition matrix plus nested evaluation workflow',
    'metric': ['outer macro-F1', 'absolute inner minus outer macro-F1'],
    'unit': 'unitless', 'direction': ['higher', 'lower'],
    'focal_series': 'Two separate evaluation endpoints',
    'required_baselines': 'Random-window, whole-trip and matched-domain inner selectors; same four candidates',
    'category_order': ['Development', 'Selection', 'Outer test', 'Refitting and prediction', 'Selection endpoint', 'Estimation endpoint'],
    'dimensions_mm': [WIDTH, HEIGHT], 'font': FONT, 'resolved_fonts': FONT_PATHS,
    'uncertainty': 'No numerical results or intervals in this schematic',
    'source': 'tex_v13/main_body.tex Figure 1 caption; tex_v10/visual_source/make_summary_visuals.py workflow()',
    'immutable_counts': {'acquisitions': 9, 'development': 6, 'test': 3, 'candidates': 4, 'outer_domains_per_axis': 3},
    'annotation_plan': 'No gray explanatory prose; definitions, repeated-use caveat and aggregation remain in caption',
    'references': ['OAI-VIS-005', 'OAI-VIS-052', 'OAI-VIS-106'],
    'formats': ['PDF', 'SVG', '300-dpi PNG', 'grayscale PNG', 'editable Python', 'standalone TeX'],
}
(ROOT/'visual_source/pvs_workflow_spec_v14.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')

fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis('off')
texts, containers, routes = [], [], []

def label(x, y, value, size=9.0, bold=False, ha='center', color=INK):
    t = ax.text(x, y, value, fontsize=size, fontfamily=FONT, weight='bold' if bold else 'normal',
                va='center', ha=ha, color=color, linespacing=1.22)
    texts.append(t)
    return t

def box(x, y, heading, body, accent=False, h=20):
    w = 48
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec=TEAL if accent else RULE, lw=.85))
    a = label(x+w/2, y+h/2-2.8, heading, 9.3, True, color=TEAL if accent else INK)
    b = label(x+w/2, y+h/2+2.8, body, 9.0, color=TEAL if accent else INK)
    containers.extend([(a, (x, y, w, h)), (b, (x, y, w, h))])

def route(points, arrow=True, color=STROKE, start_clear=2, end_clear=2, name=''):
    # All bends are orthogonal; clearance is expressed in physical points.
    assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(points, points[1:])), points
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, lw=.85, solid_capstyle='butt', solid_joinstyle='miter')
        start_clear = 0
    a, b = points[-2:]
    if arrow:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7,
                     color=color, lw=.85, shrinkA=start_clear, shrinkB=end_clear))
    else:
        ax.plot(*zip(*points), color=color, lw=.85, solid_capstyle='butt')
    routes.append({'name': name, 'points_mm': points, 'arrow': arrow})

label(4, 5.2, 'PVS acquisition grouping and nested evaluation', 11.2, True, ha='left')
label(29, 18.8, 'Nine acquisitions', 9.7, True)

# The neutral matrix represents both possible axes: no simultaneous row/column holdout.
for c in range(3):
    label(25.5+12*c, 27.1, f'S{c+1}', 9.0)
for r in range(3):
    label(16.8, 36.5+10*r, f'V{r+1}/D{r+1}', 9.0, ha='right')
    for c in range(3):
        ax.add_patch(Rectangle((20+12*c, 32+10*r), 11, 9,
                              fc='#F1F4F2', ec=STROKE, lw=.65))
label(29, 72.5, 'Hold out one row\nOR one column', 9.0)

# Equal modules on two aligned columns; each has identical text insets.
box(65, 16, 'Develop: 6 acquisitions', 'Random / trip / matched CV')
box(121, 16, 'Select 1 of 4 candidates', 'Save inner macro-F1', accent=True)
box(65, 46, 'Test: 3 acquisitions', 'Excluded from inner selection')
box(121, 46, 'Refit on development data', 'Predict outer acquisitions')
box(65, 81, 'Selection endpoint', 'Outer macro-F1 ↑', accent=True, h=19)
box(121, 81, 'Estimation endpoint', '|Inner F1 − outer F1| ↓', accent=True, h=19)

# Acquisition split. One trunk and a vertical branch replace two diagonal arrows.
route([(55, 46.5), (59.5, 46.5)], arrow=False, name='acquisition_split_trunk')
route([(59.5, 46.5), (59.5, 26), (65, 26)], name='development_acquisitions')
route([(59.5, 46.5), (59.5, 56), (65, 56)], name='held_out_acquisitions')
route([(113, 26), (121, 26)], name='inner_selection')
route([(145, 36), (145, 46)], name='selected_candidate_refit')
route([(113, 56), (121, 56)], name='test_data_for_prediction')

# Outer result is shared; the saved inner estimate has a separate path to error.
route([(145, 66), (145, 73.5)], arrow=False, name='outer_result_trunk')
route([(145, 73.5), (89, 73.5), (89, 81)], name='outer_performance')
route([(145, 73.5), (145, 81)], name='outer_result_for_error', start_clear=0)
route([(169, 26), (174, 26), (174, 90.5), (169, 90.5)], color=TEAL,
      name='saved_inner_estimate_for_error')

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
canvas = fig.bbox
padding = []
for t, bounds in containers:
    x, y, w, h = bounds
    p1 = ax.transData.transform((x, y))
    p2 = ax.transData.transform((x+w, y+h))
    x0, x1 = sorted([p1[0], p2[0]])
    y0, y1 = sorted([p1[1], p2[1]])
    b = t.get_window_extent(renderer)
    pad = min(b.x0-x0, x1-b.x1, b.y0-y0, y1-b.y1)*72/fig.dpi
    padding.append(pad)
    assert pad >= 3.0, (t.get_text(), pad)
for i, t in enumerate(texts):
    b = t.get_window_extent(renderer)
    assert canvas.contains(b.x0, b.y0) and canvas.contains(b.x1, b.y1), t.get_text()
    for other in texts[i+1:]:
        assert not b.overlaps(other.get_window_extent(renderer)), (t.get_text(), other.get_text())

for suffix in ['pdf', 'svg']:
    fig.savefig(OUT/f'pvs_evaluation_workflow_v14.{suffix}', facecolor='white')
plt.close(fig)

# Preview the exported PDF rather than a separate plotting-backend rendering.
doc = fitz.open(OUT/'pvs_evaluation_workflow_v14.pdf')
page = doc[0]
page.get_pixmap(dpi=300, alpha=False).save(OUT/'pvs_evaluation_workflow_v14.png')
with Image.open(OUT/'pvs_evaluation_workflow_v14.png') as im:
    im.convert('L').save(OUT/'pvs_evaluation_workflow_v14_grayscale.png')
font_names = sorted({f[3] for f in page.get_fonts()})
assert font_names and all('TimesNewRoman' in f for f in font_names), font_names
assert not page.get_images()
sizes = [s['size'] for b in page.get_text('dict')['blocks'] for l in b.get('lines', [])
         for s in l['spans'] if s['text'].strip()]
assert min(sizes) >= 8.99
qa = {'version': 14, 'font_names': font_names, 'font_paths': FONT_PATHS,
      'minimum_font_pt': min(sizes), 'minimum_box_padding_pt': min(padding),
      'text_overlap_count': 0, 'out_of_canvas_text_count': 0,
      'embedded_raster_images': 0, 'all_routes_orthogonal': True,
      'module_width_mm': 48, 'module_heading_body_gap_mm': 5.6,
      'routes': routes, 'review_status': 'pending final-size visual inspection',
      'generated_at_utc': datetime.now(timezone.utc).isoformat()}
(ROOT/'visual_source/pvs_workflow_drawing_qa_v14.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in qa.items() if k not in ('routes', 'font_paths')}, indent=2))
