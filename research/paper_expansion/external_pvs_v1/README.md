# PVS independent-source extension — complete and independently checked

Primary: 43,204 windows (682 positive); disjoint-window sensitivity: 10,804 windows (170 positive). Nine acquisitions, three vehicle/driver pairs crossed with three scenarios. The task is no-speed-bump versus speed-bump, not normal/pothole/bump. Vehicle and driver are confounded.

Both grids completed 264 fits, 528 total. All models/predictions/splits/caps/preprocessing/metrics were independently checked. The pre-PVS 1,710-fit suite remains unchanged, giving 2,238 computational fits overall; this count is not an independent scientific sample size.

## Read first

- 独立数据复现报告.md: Chinese result interpretation, both criterion outcomes and selected systems' test scores.
- checked_summary.json and nonoverlap/checked_summary.json: independently recomputed canonical scores and selection/error endpoints.
- protocol_v1.md and sensitivity_protocol_v1.md: fit-free design and disjoint-window control.
- external_execution_inventory_v1.json: artifact hashes and unchanged prior inventory references.
- feature_population_validation.json: independent source-label/window checks and 315 recomputed feature windows.
- acquisition_and_schema_report.md: all 18 source archive/CSV integrity and schema checks.
- pvs_reference_evidence.md: primary-source label and prior-work boundaries.
- review_gate_v1.json: prefit group/class/raw-overlap gates. Its original validator hash is retained; validator_review_amendment.md describes later verifier-only implementation fixes.
- prefit_integration_amendment.md: storage-interface guard correction before any fit, with unchanged float32 model inputs.
- nonoverlap/feature_manifest_corrected_view.json: corrected descriptive view of the sensitivity manifest. The hash-bound original remains unchanged; several copied combined-population/anchor/byte-count descriptions referred to its parent. Actual stride, retained identities, NPZ hash and 10,804-window population were correct and independently checked throughout. No model input or result changed.

## Reproduce locally

Use C:/Users/Administrator/miniconda3/python.exe (Python 3.13.12, sklearn 1.9.0, NumPy 2.5.2). Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 4 before importing numeric libraries.

1. fetch_pvs.py obtains only the nine left combined CSVs and nine label CSVs from the author-linked public Kaggle release. No videos are required. The complete kaggle_inventory.json retains source responses, file identities and hashes; raw source files are not in the paper bundle.
2. build_features.py constructs the 100-row/25-stride population. verify_feature_population.py independently checks source labels/population and 315 windows, then creates nonoverlap/features.npz and its hash-bound parent manifest. These generation scripts must not be rerun into frozen canonical artifact paths; use an isolated reproduction checkout or output copies.
3. run_nested_evaluation.py defaults to fit-free preflight. It accepts --input-npz, --feature-manifest, --preflight-dir and, for a fresh run, --output-dir --run. Existing model outputs cannot be overwritten. Run each grid in a separate directory.
4. independent_validate.py accepts --input-npz, --feature-manifest, --run-dir, --protocol protocol_v1.md and --output. It does not import the experiment driver and must finish before results are used.
5. write_result_tables.py --primary checked_summary.json --nonoverlap nonoverlap/checked_summary.json creates Tables 8–9 and S8–S16 with numeric-cell provenance. update_manuscript_v3.py integrates the checked results into new v3 sources; v2 is retained. finalize_delivery_v3.py compiles/renders/checks and packages both PDFs and editable TeX.

Do not tune on outer scores, equate windows/seeds/fits with independent acquisition units, or call the release a full external three-class test. PVS source licence is CC BY-NC-ND 4.0. The manuscript bundle excludes raw PVS data, derived feature matrices and fitted models; obtain inputs from the official author-linked source. source_feasibility.md is a historical fit-free access investigation, superseded on completed acquisition by acquisition_and_schema_report.md.
