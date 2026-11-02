"""Apply v10 figure additions once; verify frozen numerical tables unchanged."""
from pathlib import Path
import re,json,hashlib

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'tex_v9'
def sha(b): return hashlib.sha256(b).hexdigest()
main=(OLD/'main_body.tex').read_text(encoding='utf-8')
supp=(OLD/'supplementary_body.tex').read_text(encoding='utf-8')
main=main.replace('Working manuscript, version 9','Working manuscript, version 10',1)
ga=r'''
\clearpage
\noindent\textbf{Graphical abstract}\par\medskip
\noindent\includegraphics[width=\linewidth]{figures/graphical_abstract_v10.pdf}\par
\smallskip
{\small Visual summary of the existing audit. RoadSens values are five-fold means; PVS values are means across three outer domains per axis. The same nine PVS acquisitions underlie both axes and all window/budget variants. No population intervals or new independent acquisitions are implied.}\par\medskip
'''
needle=r'\section*{1. Introduction}'
assert main.count(needle)==1
main=main.replace(needle,ga+'\n'+needle,1)
workflow=r'''
Figure~\ref{fig:pvs-workflow} separates the selected system's classification performance from the accuracy of its inner performance estimate. The acquisition grid describes grouping structure, not independent replications of the two endpoints.
\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/pvs_evaluation_workflow_v10.pdf}
\caption{PVS acquisition grouping and nested evaluation. Each cell is one acquisition; a vehicle/driver-heldout fold excludes a row, while a scenario-heldout fold excludes a column. Each selector chooses among the same four candidates on the six development acquisitions. Its saved inner macro-F1 and the same chosen candidate's outer macro-F1 define an absolute estimation error, averaged over three outer domains per axis. Candidate refitting and preprocessing use development data only. Vehicle and driver effects are inseparable, and both axes reuse the same nine acquisitions.}
\label{fig:pvs-workflow}
\end{figure}
'''
needle=r'\subsection*{3.12. Post hoc GPS and speed block ablation in PVS}'
assert main.count(needle)==1
main=main.replace(needle,workflow+'\n'+needle,1)
results=r'''
\begin{minipage}{\linewidth}
Figure~\ref{fig:pvs-endpoints} places selection and estimation endpoints on separate axes. It retains both the disjoint-window sensitivity and the post hoc common-budget outer target; these are analyses of the existing acquisitions rather than new independent tests. In the primary grid, the vehicle/driver MAE reduction is 0.0085 and the scenario reduction is 0.0339 after rounding, so the declared requirement of at least 0.02 on both axes is not met.
\par\medskip
\centering
\includegraphics[width=\linewidth]{figures/pvs_endpoints_v10.pdf}
\captionof{figure}{Selected-system outer performance and estimation error in PVS. Gray circles denote random-window selection; teal squares denote matched-domain selection. Connecting segments pair two selector summaries within one analysis and held-out axis, not individual independent acquisitions. Labels reproduce descriptive three-domain means to four decimals. The primary and disjoint-window rows use full-development outer refits; the post hoc common-budget rows use equal-budget outer refits and must not be pooled with them. The left axis is restricted to 0.65--0.96; the right axis starts at zero. No population confidence intervals are asserted for three domains drawn from the same nine acquisitions. Leave-one-trip selector results and the historical full-budget diagnostic target remain in Tables 9 and 11.}
\label{fig:pvs-endpoints}
\end{minipage}\par\medskip
'''
needle=r'\subsection*{4.12. Additional-source released-subset replication}'
assert main.count(needle)==1
main=main.replace(needle,results+'\n'+needle,1)
supp=supp.replace('Working version 9,','Working version 10,',1)

tables=re.compile(r'\\begin\{tabularx\}.*?\\end\{tabularx\}',re.S)
checks={}
for name,new in [('main_body.tex',main),('supplementary_body.tex',supp)]:
    old=(OLD/name).read_text(encoding='utf-8')
    a,b=tables.findall(old),tables.findall(new)
    assert a==b,f'Numeric table changed: {name}'
    checks[name]={'tabularx_count':len(a),'all_tabularx_blocks_identical':True,
                  'v9_file_sha256':sha((OLD/name).read_bytes()),
                  'v9_normalized_text_sha256':sha(old.encode())}
    (ROOT/name).write_text(new,encoding='utf-8')
    checks[name]['v10_file_sha256']=sha((ROOT/name).read_bytes())

(ROOT/'graphical_abstract.tex').write_text(r'''% !TEX program = tectonic
\documentclass[border=0pt]{standalone}
\usepackage{graphicx}
\begin{document}
\includegraphics[width=178mm,height=118mm]{figures/graphical_abstract_v10.pdf}
\end{document}
''',encoding='utf-8')

checks['note']='Figures/captions/forward references only; no fits, frozen runs, label policies or decision thresholds changed.'
(ROOT/'visual_source'/'integration_integrity_v10.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
