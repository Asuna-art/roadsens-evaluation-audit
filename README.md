# RoadSens evaluation audit

Research artifacts for **Model selection and performance estimation can diverge: a RoadSens-4M audit and a nine-acquisition PVS case study**.

## Scientific status

- Working manuscript v14 (3 November 2026), with source TeX and actually compiled PDFs: [main](manuscript/main.pdf), [supplement](manuscript/supplementary.pdf).
- PVS has **nine acquisitions**. The original two-axis estimation-error criterion **failed** and remains reported. Fits, seeds and windows are not additional independent acquisitions.
- Mexico/Puebla is a publisher-released subset with **two documentation-derived route families**. Both selectors chose the same RF. Route provenance/population mapping remains limited.
- Indonesia original-label repair and KIT acquisition are **data preparation only**, not completed predictive replications. No unvalidated extra acquisition count is claimed.
- This repository is an empirical evaluation audit, not a validated universal improvement algorithm. Author-contribution, funding and competing-interest declarations in the working manuscript remain to be completed before journal submission.

## What is included

`research/` preserves the historical Python module topology and protocols. `results/` contains compact score tables, selections, source hashes and explicitly labelled public split/result views. `manuscript/` includes TeX, figures and the compiled PDFs. See `SOURCE_INVENTORY.json` for copied-file fingerprints and `RELEASE_AUDIT.json` for release checks.

## Evaluation architecture and visual explanations

The main article includes an editable vector architecture of source auditing, feature representations, outer/inner validation, fixed candidates, selection/refitting, and distinct classification-performance and estimation-error endpoints. RoadSens, PVS and Mexico remain separate tasks. The PVS workflow and endpoint plots retain the failed primary criterion and contrary sensitivity outcomes. All 39 original numerical table blocks are unchanged. The architecture uses embedded Times New Roman regular/bold, centered symmetric output connectors, and reduced in-figure explanatory text; details remain in the caption. Figure 1 uses the same Times typography, equal-width modules, orthogonal connectors, and a distinct saved-inner-estimate path to the estimation endpoint. Its longer explanatory notes remain in the caption.

- [Standalone framework architecture](manuscript/framework_architecture.pdf)
- [Editable architecture SVG](manuscript/figures/framework_architecture_v13.svg)
- [Architecture source and compilation guide](manuscript/ARCHITECTURE_GUIDE_v13.txt)
- [PVS grouping and nested-evaluation diagram](manuscript/pvs_evaluation_workflow.pdf)
- [Editable PVS workflow SVG](manuscript/figures/pvs_evaluation_workflow_v14.svg)
- [PVS workflow source and compilation guide](manuscript/PVS_FIGURE_GUIDE_v14.txt)
- Diagram source: `manuscript/visual_source/draw_framework_v13.py`.
- PVS endpoint plotting source and checked aggregate values: `manuscript/visual_source/` and `manuscript/figures/pvs_endpoints_v10.json`.

## Reproduction scope

The exact historical environment was Windows, Python3.13.12, scikit-learn1.9.0, NumPy2.5.2, pandas3.0.5, SciPy1.18.0 and joblib1.5.3. Read `research/paper_expansion/runtime_lock_v1.md` and the relevant protocol before running anything. Historical scripts retain their original bytes, fixed interpreter paths and hash guards. This publication does **not** claim a tested one-command portable rerun. Do not regenerate frozen split maps from seeds or overwrite existing runs.

Obtain source datasets from their authors and verify the recorded hashes. Local caches, raw feature matrices and full execution authorities are not published here. Public views omit direct signal/label arrays and coordinates; their original-authority hash is recorded. They are not substitutes for absent inputs in hash-guarded historical replay commands. Results can be inspected directly in the manuscript and compact tables without fitting models.

To compile the manuscript, first run `tectonic framework_architecture.tex`, then `tectonic main.tex` and `tectonic supplementary.tex` from `manuscript/` with the documented fonts available. These are compilation instructions; a new remote-machine build has not been tested.

## Source data and reuse

- RoadSens: https://doi.org/10.6084/m9.figshare.30341143.v3
- PVS: https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets (source CC BY-NC-ND4.0)
- Mexico/Puebla: https://doi.org/10.3390/app151810203 and https://doi.org/10.3390/vehicles8010004
- Indonesia: https://github.com/s-budidarma/dataset-smarphone-sensor-for-road-surface-monitoring
- KIT: https://doi.org/10.5281/zenodo.1461243

Raw third-party data, fitted models, feature caches, credentials, video and download archives are excluded. No new license is granted to third-party source data. No permissive code license is assigned on behalf of the authors by this upload; contact the repository owner about reuse.

## 中文说明

本仓库发布关键代码、冻结方案、汇总结果与可编辑稿件。原双轴失败、小样本与来源限制均保留。新增数据准备不算已完成验证。公开视图省略原始信号、直接标签/预测数组和坐标，不替代原始冻结实验输入；完整复跑需要合法取得来源数据及对应输入。
