# Road-sensing evaluation audit

Research artifacts for **Classifier selection, score estimation and event coverage in inertial pavement screening**.

## Current evidence

The focused v19 main article foregrounds classifier selection, score-estimation accuracy and event coverage. Five core experiment tables remain in the main text; five supporting tables are collected in Supplementary Section S18. Every numerical table block is retained, and no model, split or experimental result was changed for this revision. The author-inclusive main PDF is 18 pages and the anonymous version is 17 pages; the complete supplement is 43 pages.

The author-review manuscript is v19 (3 November 2026): [compiled main article](manuscript/main.pdf), [compiled supplement](manuscript/supplementary.pdf), and editable TeX in `manuscript/`. Anonymous review versions and separate author information are also supplied.

- RoadSens is a version-specific audit of 101 source CSVs. Source files are not assumed to be independent roads.
- PVS remains **nine acquisitions**, with inseparable vehicle/driver identities. Its original two-axis estimation-error criterion **failed** and remains reported. Windows, seeds, annotation intervals and fits do not increase this count.
- Mexico/Puebla remains **two documentation-derived route families**; both selectors choose the same RF.
- Karlsruhe (KIT) is now a **completed independent-source predictive test**. A verified public archive yields three recoverable GPS-date groups. The first two supply development data; 5 July 2016 is the frozen final test. Full-window spatial purging removes training observations within 100 m of eligible test anchors; the actual retained minimum is 922.43 m. The release uses one vehicle. Fifty-three IMU workbooks are not fifty-three independent drives or roads.
- Indonesia's expanded label-version conflict remains unresolved and is excluded from reliable predictive validation.

## New independent-source and engineering outcomes

The Karlsruhe final test has 917 windows (835 good, 82 pothole). Inner fit budgets are matched at 1,613 good and 17 pothole windows across validation schemes and fixed candidates. Buffered-date selection chooses inertial HGB; random-window selection chooses HGB with context. Full-development final macro-F1 is 0.7855 versus 0.6782, with absolute estimation errors 0.2072 versus 0.2817. Under the common-budget outer target, scores are 0.7595 versus 0.6699 and errors 0.1812 versus 0.2899. This is one final date, not a replicated multi-date mean.

Both selected systems detect **7/12 annotated pothole intervals**. The five intervals in workbook 50 are missed; hits occur in workbooks 51/52, which share a geographic neighbourhood. Zero non-event alarms are observed over only **3.47 minutes of explicit good-labelled exposure**, concentrated in one workbook. Unknown-labelled travel is outside the binary task. The event counts do not identify unique physical potholes.

The PVS scenario diagnostic changes from 34/60 to 52/60 detected annotation intervals while non-bump alarms rise from 7.45 to 17.28 per assessed non-bump hour. Event-associated duplicate notifications are reported separately. The frozen original predictions and nine acquisitions are reused. The post-hoc reporting amendment and original unmatched-notification results are preserved.

These findings separate selected-model classification, score-estimation accuracy and event-screening behaviour. They do not establish a universal improvement rule or a field-ready detector.

## Inspecting and reproducing evidence

- [New protocols and exact executed source](research/paper_expansion/engineering_validation_20261103/)
- [Public compact outcomes, provenance and independent checks](results/engineering_validation_20261103/)
- [Public evidence manifest](results/engineering_validation_20261103/PUBLIC_EVIDENCE_RELEASE.json)
- [New-source replay notes](research/paper_expansion/engineering_validation_20261103/REPLAY_NOTES.txt)

The original environment is Windows, Python 3.13.12, scikit-learn 1.9.0, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0 and joblib 1.5.3. The research interpreter is `C:\Users\Administrator\miniconda3\python.exe`. Historical sources retain their actual executed bytes, absolute workspace paths and hash guards. No tested one-command portable replay is claimed. Obtain source inputs separately, verify fingerprints, preserve exact memberships and never overwrite frozen runs or reconstruct historical split maps from seeds alone.

Public JSON views retain outcome counts and source fingerprints while omitting full inputs, arrays, models and coordinates. Each `originalauthoritySHA256` identifies the complete local JSON authority before those omissions. Public views are for result inspection and are not substitutes for hash-guarded execution authorities.

Compile `main.tex` and `supplementary.tex` from `manuscript/` with Tectonic and Times New Roman available. Actually compiled author-inclusive and anonymous PDFs are included. Figure source and vector artwork are preserved. Historical numerical values and contrary findings remain; new source/event tables extend them.

## Source data and reuse

- [RoadSens v3](https://doi.org/10.6084/m9.figshare.30341143.v3)
- [PVS](https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets), source CC BY-NC-ND 4.0
- [Mexico/Puebla source article](https://doi.org/10.3390/app151810203) and [later source documentation](https://doi.org/10.3390/vehicles8010004)
- [Karlsruhe data](https://doi.org/10.5281/zenodo.1461243), source CC BY 4.0; [original methods](https://doi.org/10.1155/2018/8647607)
- [Indonesia source](https://github.com/s-budidarma/dataset-smarphone-sensor-for-road-surface-monitoring)

Raw third-party signals, coordinates, video, download archives, full labels/predictions and fitted models are excluded. No new license is granted to source data, and no permissive code license is assigned on the authors' behalf. Funding, disclosure and contribution statements reflect author-supplied information. Final author consent and live journal submission checks remain with the authors; no journal submission is performed by this release.

## 中文说明

v18 增加了真正来自另一公开来源的 Karlsruhe 验证，按 GPS 日期留出完整最终测试组，并对重复地点做空间剔除；同时补充了事件漏检、非事件报警、重复通知及其暴露时长。PVS 仍是九次采集，原双轴失败保留。KIT 只使用单车、一个最终测试日期，事件为标注经过而非已确认独立物理坑洞；零误报只对应 3.47 分钟明确良好路面。原始数据、模型和完整数组不公开。当前稿定位为有工程指标的评估审计，不能据此声称检测算法已成熟。
