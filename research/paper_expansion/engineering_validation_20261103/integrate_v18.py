"""Integrate verified public-source and engineering evidence into an isolated TeX draft."""
from pathlib import Path
import json, hashlib

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent/'manuscript/ijpe_evidence_v18'
KIT=json.loads((HERE/'run_kit_v1/results.json').read_text(encoding='utf8'))
EVENT=json.loads((HERE/'run_kit_v1/engineering_results.json').read_text(encoding='utf8'))
PVS=json.loads((HERE/'pvs_event_results.json').read_text(encoding='utf8'))

def read(p):return p.read_text(encoding='utf8')
def write(p,s):p.write_text(s,encoding='utf8')
def once(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old))
    return text.replace(old,new)

def table(caption,label,columns,header,rows,note):
    return ('\\begin{table}[p]\n\\centering\n\\begin{singlespace}\n\\small\n'
            f'\\caption{{{caption}}}\\label{{{label}}}\n'
            f'\\begin{{tabularx}}{{\\linewidth}}{{{columns}}}\n\\toprule\n{header} \\\\\n\\midrule\n'
            +'\n'.join(rows)+'\n\\bottomrule\n\\end{tabularx}\n\\par\\smallskip\n'
            +'\\raggedright '+note+'\n\\end{singlespace}\n\\end{table}\n')

def main():
    a=read(PAPER/'article_text.tex')
    assert 'KIT: a final date' not in a,'Refusing repeated manuscript integration'
    a=once(a,'The present study concerns classification of the released road-event labels as a screening step; its outcomes are class-based scores, rather than measurements of roughness, distress severity or structural condition.',
           'The present study evaluates classification of released road-event labels and the associated event-screening workload. The measurement targets are labelled surface events; roughness, distress severity and structural condition require their own reference measurements.')
    a=once(a,'Each dataset is analysed as a separate task.',
           'A further independent-source test uses public Karlsruhe vehicle-inertial data \\citep{kitpaper,kitdata}, with a complete GPS-date group reserved for final testing and spatial buffering of repeated locations. Each dataset is analysed as a separate task.')
    a=once(a,'The resulting evidence informs how a pavement-screening study can report model choice and the expected reliability of its score.',
           'Annotation-interval detection and non-event alarm counts connect these score comparisons to screening workload. The resulting evidence informs how a pavement-screening study can report model choice, the reliability of its score and its event-level operating behaviour.')
    methods=r'''\subsection{Annotation-interval screening diagnostics}
We additionally evaluate the frozen PVS selected systems as event-screening outputs. The nine source label sequences contain 60 contiguous speed-bump intervals, with 11, seven and two per scenario for each vehicle/driver pair. These are labelled passages over features. A positive run ends at a non-positive row or a timestamp gap greater than 0.1 s. An interval is evaluable when it contains a saved outer-test decision anchor. Consecutive positive decisions on the 0.25-s grid form one alarm cluster; a negative decision or an excluded grid position separates clusters. Maximum one-to-one matching associates alarm clusters with annotated intervals using decision timestamps. The primary tolerance is zero; a 0.5-s expansion is a sensitivity check.

Alarm clusters outside every eligible interval are non-event alarms. Unmatched clusters associated with an interval are redundant notifications. Non-event rates use explicit non-bump exposure between the first and last assessed anchor, at the released 100-Hz cadence. Results pool counts across the three outer domains separately for each axis, covering every acquisition once per selector. Protocol definitions were fixed before computing these post hoc event outcomes. A subsequent reporting clarification separates redundant notifications from non-event alarms while retaining the initial total unmatched counts. Supplementary Section S16 supplies per-acquisition values and both tolerances.

\subsection{Karlsruhe independent-source date holdout}
The Karlsruhe release contains 53 IMU workbooks and 13 additional event excerpts \citep{kitdata}. The original acquisition system used an LSM9DS1 inertial sensor in a BMW 116d \citep{kitpaper}. The 53 IMU workbooks comprise 1,764,985 timestamped rows. Decoded GPS timestamps identify 15 workbooks in the 13 May 2016 group, 29 in the 20 June group and four in the 5 July group; five workbooks have no recoverable GPS date. The sensor's numeric epoch disagrees with the GPS calendar, so within-file numeric time defines durations and GPS time defines date groups. The excerpts and undated workbooks are excluded from this task. These file counts describe the release structure; repeated road features remain shared acquisition content.

The binary target uses the documented event codes 1 (good) and 3 (pothole). Other codes remain outside the labelled task, including code 0 (unknown) and unresolved code 7. Trailing one-second windows are evaluated every 0.25 s, using at least 50 raw rows, finite inertial axes and no timestamp gap above 0.1 s. The final source row at or before the decision provides its label and context. Thirty inertial statistics and the optional coordinate/speed block follow the PVS feature definitions. This yields 5,226 eligible windows.

The May and June GPS-date groups supply development data. The July group is fixed as the final test before model fitting. A development window is removed when any raw coordinate in its span is invalid or lies within 100 m of an eligible final-test anchor. Random-window two-fold selection is compared with two-fold GPS-date selection, applying the same additional spatial buffer against each date-validation population. The four existing candidates and their settings remain fixed. Every inner fit uses the common minimum available per-class counts across all four training pools; exact subsamples and split memberships are saved before fitting. Validation populations retain their observed class distribution.

Each selector's pooled OOF macro-F1 determines its candidate, with the existing tie order. Saved choices precede final-test prediction. Primary outer refits use all spatially eligible development windows. A separate outer target retains the common inner training budget. We report the selected candidate's final macro-F1 and the absolute difference from its saved inner estimate for each target. The engineering diagnostic applies the same interval/alarm matching to retained good/pothole decisions. Its exposure denominator is explicit good-labelled raw time, integrating positive timestamp increments capped at 0.1 s. A single final GPS-date group supplies this independent-source test; Supplementary Section S17 gives source checks, memberships and all candidate outcomes.

'''
    a=once(a,r'\subsection{Additional-source tasks}',methods+r'\subsection{Additional-source tasks}')
    a=once(a,'Protocol versions, preflight revisions, model reload checks and numeric reconstruction records are retained in the supplement.',
           'The Karlsruhe protocol, final date, buffer and candidate pool were fixed after source-quality inspection and before any Karlsruhe predictive fitting. Its inner training counts are matched across selectors. Protocol versions, preflight revisions, model reload checks and numeric reconstruction records are retained in the supplement.')
    pvs_results=r'''\subsection{PVS annotation-interval detection and alarm burden}
All 60 positive annotation intervals contain an assessed decision anchor. Under vehicle/driver holdout, the three selectors choose systems that detect 59/60 intervals, with 71 non-bump alarm clusters over 2.9522 non-bump hours (24.05 per hour) and 14 redundant event-associated clusters. Under scenario holdout, random-window and whole-trip selection detect 34/60 intervals, with 22 non-bump clusters (7.45 per hour) and 14 redundant clusters. Matched selection detects 52/60 intervals, with 51 non-bump clusters (17.28 per hour) and 18 redundant clusters (Table~\ref{tab:engineering}). Interval recall therefore rises from 56.7\% to 86.7\% alongside a higher non-bump alarm burden. The 0.5-s matching sensitivity preserves all pooled hit counts. It reclassifies five vehicle/driver alarms and one scenario alarm per selector as event-associated, preserving the scenario ranking in non-event burden. These diagnostics reuse the existing nine acquisitions and saved hard decisions.

'''
    a=once(a,r'\subsection{Common-budget diagnostic}',pvs_results+r'\subsection{Common-budget diagnostic}')
    kit_results=r'''\subsection{KIT: a final date with buffered spatial separation}
The final test contains 917 windows (835 good and 82 pothole) from the four July workbooks. Spatial purging removes 20 of the 4,309 development windows, leaving 4,123 good and 166 pothole windows. The closest retained full training window is 922.43 m from a test anchor; both date-validation separations exceed 2,432 m. Every inner candidate fit uses 1,613 good and 17 pothole windows.

Random-window selection chooses HGB with context, with inner macro-F1 0.9598. GPS-date selection chooses inertial-only HGB, with inner macro-F1 0.5783. Full-development final scores are 0.6782 and 0.7855, giving absolute estimation errors of 0.2817 and 0.2072, respectively (Table~\ref{tab:kit}). With common-budget outer refits, the corresponding final scores are 0.6699 and 0.7595, and errors are 0.2899 and 0.1812. Thus, the date selector chooses a stronger system and produces a closer estimate on this final date under both refit targets.

The full-development systems classify 20/82 and 35/82 pothole anchors correctly, respectively, with no good anchors classified as potholes. Both detect seven of the 12 evaluable pothole annotation intervals. The date-selected system adds one redundant event-associated cluster (Table~\ref{tab:engineering}). Both have zero observed non-event clusters during 3.47 minutes of assessed good-labelled exposure. That exposure is concentrated in one good-labelled workbook, and unknown-labelled travel is outside the task. The observed zero therefore describes this finite labelled exposure. The 0.5-s sensitivity gives the same interval hits and alarm categorisation. Higher final macro-F1 here reflects more correct decisions within already detected intervals rather than additional interval detections.

'''
    a=once(a,r'\section{Discussion}',kit_results+r'\section{Discussion}')
    a=once(a,'Distinguishing them from potholes is relevant to screening, but a class score alone does not measure the severity or maintenance priority of a pavement defect.',
           'Distinguishing them from potholes is relevant to screening, while severity and maintenance priority require additional reference information. The event diagnostics add a further distinction: PVS scenario recall improves at a higher alarm cost, whereas the Karlsruhe window-score improvement leaves the number of detected intervals unchanged. A screening comparison should therefore pair classification scores with missed intervals, non-event alarms, redundant notifications and their assessed exposure.')
    a=once(a,'Selector differences are thus contingent on candidate rankings and the available domains.',
           'The independent Karlsruhe date test favours the date selector for both selected-model score and estimation error, including under matched inner/outer training counts. Its unchanged interval hit count adds a third outcome alongside model ranking and score estimation. Selector differences are thus contingent on candidate rankings, the available domains and the engineering endpoint.')
    a=once(a,'These grouping counts, rather than row or fit counts, set the scale of the empirical evidence.',
           'Karlsruhe adds a separately collected source, three recoverable GPS-date groups and one final test date after spatial purging. Its single vehicle and sparse, selectively labelled test exposure define the transfer target. These grouping counts set the scale of the empirical evidence; windows, fitted models and repeated annotation intervals supply dependent observations within it.')
    a=once(a,'For an operational screening service, reference-confirmed events and false alarms per travelled distance would further connect classification scores to inspection workload.',
           'The present interval-based diagnostics quantify labelled-event coverage and notification workload. A field service additionally requires independently adjudicated physical events, complete normal-travel exposure and reliable travelled-distance records to assess maintenance inspection workload.')
    a=once(a,'The Mexico/Puebla subset gives identical selected models and small errors for both selectors.',
           'The Mexico/Puebla subset gives identical selected models and small errors for both selectors. The independent Karlsruhe final date favours buffered-date selection for classification and estimation, while both selected systems detect seven of 12 pothole intervals. PVS scenario detection improves alongside a higher non-bump alarm rate.')
    a=once(a,'Source data are available from the cited RoadSens, PVS, STRIDE and Mexico/Puebla providers.',
           'Source data are available from the cited RoadSens, PVS, STRIDE, Mexico/Puebla and Karlsruhe providers. The Karlsruhe archive is identified by DOI 10.5281/zenodo.1461243 and SHA-256 \\path{f7f7cc6b017977d2019f1eaaf9e90d6ad6132506965bafba03c12e6b053845d3}.')
    write(PAPER/'article_text.tex',a)
    write(PAPER/'abstract.tex', '''Inertial road-surface classifiers need validation at the scale of their intended deployment. We audit source structure and evaluate model selection and performance estimation as separate endpoints. The analyses use RoadSens, nine PVS acquisitions, two Mexico/Puebla route families and a date-held-out test of public Karlsruhe inertial data. RoadSens full-feature random-forest macro-F1 falls from 0.9945 under random-row validation to 0.4289 under recording holdout. PVS matched validation improves selected-model scenario performance but fails the declared two-axis estimation-error criterion. Its scenario annotation-interval recall rises from 34/60 to 52/60, accompanied by more non-bump alarms. On the untouched Karlsruhe date, buffered-date selection raises macro-F1 from 0.6782 to 0.7855 and reduces absolute estimation error from 0.2817 to 0.2072; both selected systems detect 7/12 annotated pothole intervals. These results distinguish candidate ranking, score estimation and event-screening performance, and support reporting them with explicit acquisition groups, training budgets and exposure denominators.\n''')

    rows=[]
    for axis,title in [('vehicle','PVS vehicle/driver'),('scenario','PVS scenario')]:
        rows.append(r'\multicolumn{6}{l}{\textit{'+title+r'}} \\')
        for selector,name in [('row_random','Random-window'),('matched_factor','Matched')]:
            r=next(x for x in PVS['pooled_by_axis_selector_tolerance'] if x['axis']==axis and x['selector']==selector and x['tolerance_seconds']==0)
            rows.append(f"{name} & {r['matched_intervals']}/{r['evaluated_intervals']} & {r['nonbump_alarm_clusters']} & {r['rate_nonbump_alarms_per_nonbump_hour']:.2f} & {r['duplicate_event_alarm_clusters']} & {r['nonbump_exposure_hours']:.4f} \\\\")
    rows.append(r'\midrule\multicolumn{6}{l}{\textit{Karlsruhe final GPS-date group}} \\')
    for selector,name in [('row_random_two_fold','Random-window'),('date_buffered_two_fold','Buffered-date')]:
        r=EVENT['selectors'][selector]['zero_tolerance']['pooled']
        rows.append(f"{name} & {r['matched_intervals']}/{r['evaluable_intervals']} & {r['nonbump_alarm_clusters']} & {r['nonbump_alarm_rate_per_good_hour']:.2f} & {r['duplicate_event_alarm_clusters']} & {r['good_exposure_hours']:.4f} \\\\")
    write(PAPER/'tables/main_09.tex',table('Annotation-interval coverage and alarm burden of selected full-development systems. Zero matching tolerance.','tab:engineering','XYYYYY',
        r'Selector & \makecell[r]{Hits /\\intervals} & \makecell[r]{Non-event\\clusters} & \makecell[r]{Clusters per\\negative h} & \makecell[r]{Redundant\\clusters} & \makecell[r]{Negative\\exposure (h)}',rows,
        'Higher interval coverage and lower alarm rates are desirable. Negative exposure means explicit non-bump time in PVS and explicit good-labelled time in Karlsruhe. PVS whole-trip results equal random-window results. Each PVS axis reuses the same nine acquisitions; Karlsruhe has one final date and only 3.47 minutes of assessed good exposure. Intervals are labelled passages, with repeated physical features possible.'))
    rows=[]
    for target,title in [('full_development','Full-development outer refit'),('common_inner_budget','Common-budget outer refit')]:
        rows.append(r'\multicolumn{5}{l}{\textit{'+title+r'}} \\')
        for selector,name in [('row_random_two_fold','Random-window'),('date_buffered_two_fold','Buffered-date')]:
            z=KIT['selectors'][selector]['targets'][target]
            candidate='HGB + context' if z['candidate_id']=='hgb_sensor_context' else 'HGB inertial'
            rows.append(f"{name} & {candidate} & {z['inner_macro_f1']:.4f} & {z['outer_test_metrics']['macro_f1']:.4f} & {z['absolute_error']:.4f} \\\\")
    write(PAPER/'tables/main_10.tex',table('Karlsruhe selection and estimation on the final GPS-date group.','tab:kit','XXYYY',r'Selector & Selected candidate & Inner F1 & Final F1 & Absolute error',rows,
        'The final group is 5 July 2016, decoded from GPS time. Higher final macro-F1 and lower absolute estimation error are desirable. Full refits use 4,289 development windows; common-budget refits use 1,613 good and 17 pothole windows. Both selectors use these same per-class counts for each inner fit. A single held-out date supplies each final score.'))
    b=read(PAPER/'main_body.tex');b=once(b,r'\input{tables/main_08.tex}',r'\input{tables/main_08.tex}'+'\n'+r'\input{tables/main_09.tex}'+'\n'+r'\input{tables/main_10.tex}');write(PAPER/'main_body.tex',b)

    refs=r'''\bibitem[Masino et~al.(2018)]{kitpaper}
Masino, J., Thumm, J., Levasseur, G., Frey, M., Gauterin, F., Mikut, R. and Reischl, M., 2018. Characterization of road condition with data mining based on measured kinematic vehicle parameters. \emph{Journal of Advanced Transportation}, 2018, 8647607. \url{https://doi.org/10.1155/2018/8647607}.

\bibitem[Masino(2018)]{kitdata}
Masino, J., 2018. \emph{Characterization of road condition with data mining based on measured kinematic vehicle parameters, data} [dataset]. Zenodo. \url{https://doi.org/10.5281/zenodo.1461243}.

'''
    for name in ['bibliography.tex','bibliography_supp.tex']:
        q=read(PAPER/name)
        # The supplement may share the main bibliography through an input.
        if r'\end{thebibliography}' in q:q=once(q,r'\end{thebibliography}',refs+r'\end{thebibliography}')
        write(PAPER/name,q)
    supp=r'''\clearpage
\section*{S16. PVS annotation-interval engineering diagnostic}
The protocol \path{PVS-ANNOTATION-EVENT-20261103-v1} was fixed after the original classification analysis and before event outcomes were computed. It reuses frozen outer predictions. The original protocol SHA-256 is \path{2e56a31b564da2d1cb2ba669c23b4c753daf86fb613d2813ab21c63796ad6d05}. The 60 contiguous positive source-label intervals are passages in the existing nine acquisitions. Every interval contains an assessed decision anchor. Negative exposure totals 2.9522222 h per axis and selector. A 100-row trailing window supplies a hard decision every 25 rows; a non-adjacent grid position or a negative decision separates alarm clusters.

An eligible alarm/event edge requires a positive decision timestamp in the annotated interval, expanded by the stated tolerance. Deterministic maximum-cardinality one-to-one matching counts hits. An alarm with no eligible edge is a non-event alarm; an unmatched alarm with an edge is a redundant event-associated notification. The initial calculation retained all unmatched alarms as a notification-burden endpoint. After inspecting that output, reporting amendment A1 separated its two components without changing predictions, thresholds, boundaries, matching, exposure or original counts. Both unmatched totals and the clarification are retained in the analysis records. Offsets are relative to annotation starts and describe annotation timing, with no physical-impact latency interpretation.

\input{tables/supp_new_pvs_events.tex}
\clearpage
\section*{S17. Independent-source Karlsruhe validation}
The original archive \path{Daten2018.zip} is 546,803,578 bytes; its official MD5 is \path{e9a4eb2da043c1eb3dbf57134cf8796b}, with all archive member CRC checks passing. The retained 53 IMU workbooks contain 1,764,985 timestamped rows. The 13 additional event excerpts are excluded because some contain samples already present in the IMU tables. Header checks identify acceleration in C:E, gyroscope in F:H, GPS timestamp in S, coordinates in T:U, speed in V, material in AA and event code in AB. Six-axis features are recomputed from these measured channels. The source event labels used are 1 (good) and 3 (pothole), following \citet{kitpaper}; other codes remain excluded.

GPS timestamps establish three usable date groups. Five workbooks without a recoverable GPS calendar are excluded before feature construction. Monotone sensor time supports within-file duration, while its differing epoch prevents a sensor-epoch calendar interpretation. The date/file mapping is retained with full arrays in the local audit. A file is a release container, and the date groups do not identify unique roads or new vehicles.

The frozen protocol is \path{KIT-PUBLIC-INDEPENDENT-SOURCE-20261103-v1}, SHA-256 \path{2782b0f43fd442a408f9acb8e6521cf7e64432abec45172a800c3983bb9d9105}. Schema, date/class counts and geographic overlap were inspected before this protocol; all model outcomes followed its freeze. The final date is excluded from candidate selection. Training windows are spatially purged using every raw coordinate in their span against eligible held-out anchors, followed by fixed common per-class inner training counts. Exact split and fit arrays are retained. The two date validation pools partition the same development windows as the random folds. Their full-window minimum separations are both 2,432.56 m. The final test has 917 windows and a retained-training separation of at least 922.43 m.

The initial preflight encountered an implementation exception after writing its index map. That incomplete fit-free output is retained separately. The corrected preflight produces the identical index arrays, verified element by element before predictive execution. Subsequent independent checks confirm labels, sample features, full-window buffers, class budgets, candidate choice, all saved metrics, model reload predictions and selection-file precedence over final predictions. Twenty-four fit-folds implement the four fixed candidates across the two selectors and two outer refit targets; they introduce no additional candidate architectures.

\input{tables/supp_new_kit.tex}

The final engineering population comprises 12 evaluable pothole annotation intervals and 0.0578948 h of assessed good-labelled travel. Unknown and other event classes are outside the binary task, including their travel exposure. Alarm decisions use the 0.25-s decision grid; raw row differences vary with sensor cadence. A good-labelled raw row contributes its positive time increment capped at 0.1 s, with the final interval row using the median positive increment. Non-event alarms and redundant event-associated notifications are reported separately. No travelled-distance denominator or independently video-adjudicated impact time is available in this analysis. Both tolerance settings and all per-file counts are supplied below.
\input{tables/supp_new_kit_events.tex}

'''
    s=read(PAPER/'supplementary_body.tex')
    s=once(s,r'\input{bibliography_supp.tex}',r'\input{bibliography_supp.tex}') # Check exact insertion anchor.
    # Insert before the supplement's final bibliography, outside its single-space wrapper.
    end='\\clearpage\n\\begin{singlespace}\\small\n\\input{bibliography_supp.tex}'
    s=once(s,end,supp+end);write(PAPER/'supplementary_body.tex',s)
    rows=[]
    for x in PVS['by_trip_domain_selector_tolerance']:
        if x['tolerance_seconds']!=0:continue
        name={'row_random':'Random','leave_one_trip':'Trip','matched_factor':'Matched'}[x['selector']]
        rows.append(f"{x['axis'].capitalize()} & {x['trip']} & {name} & {x['matched_intervals']}/{x['evaluated_intervals']} & {x['nonbump_alarm_clusters']} & {x['duplicate_event_alarm_clusters']} & {x['nonbump_exposure_hours']:.4f} \\\\")
    # A longtable preserves per-acquisition denominators without shrinking type.
    q='\\begin{singlespace}\\small\n\\begin{longtable}{llrrrrr}\n\\caption{PVS per-acquisition engineering counts at zero tolerance.}\\\\\n\\toprule\nAxis & Trip & Selector & Hits/intervals & Non-event & Redundant & Negative h \\\\\n\\midrule\\endfirsthead\n\\toprule Axis & Trip & Selector & Hits/intervals & Non-event & Redundant & Negative h \\\\\n\\midrule\\endhead\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{longtable}\n\\end{singlespace}\n'
    write(PAPER/'tables/supp_new_pvs_events.tex',q)
    rows=[]
    for day,good,pos,role in [('13 May',2511,149,'Development'),('20 June',1613,36,'Development'),('5 July',835,82,'Final test')]:
        rows.append(f'{day} & {role} & {good:,} & {pos} \\\\')
    rows.extend([r'\midrule',r'Purged development & Full outer refit & 4,123 & 166 \\',r'Each actual inner fit & Shared budget & 1,613 & 17 \\'])
    q=table('Karlsruhe window populations before and after final-test spatial purging.','tab:s-kit-pop','XXYY',r'GPS date / pool & Role & Good & Pothole',rows,'All dates are in 2016. Windows are one second long with 0.25-s decision spacing. Other labels and undated workbooks are excluded.')
    rows=[]
    for cand in ['rf_sensor','rf_sensor_context','hgb_sensor','hgb_sensor_context']:
        p=HERE/'run_kit_v1/fits'/f'outer_full_development_{cand}'/'fit.json';m=json.loads(read(p))['metrics']
        sel=json.loads(read(HERE/'run_kit_v1/selection.json'))['selectors']
        rows.append(f"{cand.replace('_', r'\_')} & {sel['row_random_two_fold']['pooled_inner_oof_macro_f1'][cand]:.4f} & {sel['date_buffered_two_fold']['pooled_inner_oof_macro_f1'][cand]:.4f} & {m['macro_f1']:.4f} & {m['class_recalls'][1]:.4f} \\\\")
    q+=table('Karlsruhe fixed-candidate estimates and final full-development scores.','tab:s-kit-candidates','XYYYY',r'Candidate & Random inner F1 & Date inner F1 & Final F1 & Pothole recall',rows,'Candidate selection uses the corresponding inner score only. All final candidate outcomes are retained for audit; outer outcomes do not determine selector choices. Sensor denotes the 30 inertial statistics.')
    write(PAPER/'tables/supp_new_kit.tex',q)
    rows=[]
    for selector,name in [('row_random_two_fold','Random'),('date_buffered_two_fold','Date')]:
        for key,tol in [('zero_tolerance',0),('half_second_sensitivity',.5)]:
            for r in EVENT['selectors'][selector][key]['per_file']:
                rows.append(f"{name} & {tol:.1f} & {r['file_id']} & {r['matched_intervals']}/{r['evaluable_intervals']} & {r['nonbump_alarm_clusters']} & {r['duplicate_event_alarm_clusters']} & {r['good_exposure_hours']:.5f} \\\\")
    write(PAPER/'tables/supp_new_kit_events.tex',table('Karlsruhe final-test per-file interval and notification counts.','tab:s-kit-events','XYYYYYY',r'Selector & Tolerance (s) & File ID & Hits/intervals & Non-event & Redundant & Good h',rows,'Zero intervals are shown as 0/0, a count rather than a defined recall. File IDs are preserved archive-container identities, not independent roads. Assessed good exposure is concentrated in file 49.'))
    cover=read(PAPER/'cover_letter.tex')
    cover=once(cover,'A two-route Mexico/Puebla task provides a contrasting result: the selectors choose identical systems and have small estimation errors.',
        'A two-route Mexico/Puebla task provides a contrasting result: the selectors choose identical systems and have small estimation errors. An independent public Karlsruhe source adds a final GPS-date test with spatially buffered development data and matched inner training counts. On that test, date selection improves classification and estimation, while both chosen systems detect seven of 12 annotated pothole intervals. Event and alarm diagnostics connect the evaluation results to screening workload.')
    cover=once(cover,'The article relates acquisition grouping and label semantics to the distinction between pavement distress, intentional road features and non-event observations.',
        'The article relates acquisition grouping and label semantics to the distinction between pavement distress, intentional road features and non-event observations, and quantifies the resulting interval coverage and notification burden.')
    write(PAPER/'cover_letter.tex',cover)
    report={'paper_directory':str(PAPER),'new_main_tables':['main_09.tex','main_10.tex'],'new_supplement_sections':['S16','S17'],
        'abstract_words':len(read(PAPER/'abstract.tex').split()),'inputs':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'pvs_event_results.json',HERE/'run_kit_v1/results.json',HERE/'run_kit_v1/engineering_results.json']},
        'original_failure_preserved':'The declared requirement of at least 0.02 on both axes is therefore not met.' in a}
    write(HERE/'manuscript_integration.json',json.dumps(report,indent=2))

if __name__=='__main__':main()
