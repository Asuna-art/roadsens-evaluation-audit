# RoadSens paper-scale expansion: frozen analysis plan v1

Frozen 2026-10-16 after the exploratory 330 model-folds and before any new-model expansion scores. This is a disclosed post-pilot extension, not a preregistration before discovery, not an untouched test set. The original pilot, failures and thresholds remain immutable. All specified contrasts are reported, including null or contrary results. The goal is an empirical evaluation paper, not invention of a classifier or a guarantee of journal acceptance.

## Questions and populations

Q1: Is the random-row / recording-heldout contrast model-dependent? Q2: Which measured feature families carry predictive information, under each label interpretation? Q3: How sensitive are findings to training budget and forest capacity? Q4: Does the contrast persist without the five Normal-only recordings? Q5: What can a genuinely separate accessible dataset verify?

RoadSens population stays the 101 complete Combined v3 CSVs, excluding summary99 and retaining source rows 0,10,20,...; all original row histories and labels remain unchanged. Primary: directory-defined Normal plus explicit Pothole/Bump; mixed blanks excluded. Sensitivity: mixed blanks=Normal, an unverified interpretation. Operational recording means one CSV, not an independently verified drive, route, vehicle or device.

For three-class analyses keep Normal/Pothole/Bump order. Random and recording splits are exactly the existing seed42/43/44 split rules. Timeblock is exactly the existing source-span quintiles and inclusive 1-second embargo (seed42 model/sampling only), with every raw row assigned before label filtering. All methods within a policy/split/seed share the same scoreable population, folds and fit-row IDs. Training cap20,000 and class-proportional sampling remain original, RandomState(seed*1000+zero_based_fold). Only the expressly named uncapped budget analyses use all eligible training rows. No test-driven feature or parameter selection.

## A. Main cross-model table: 480 model-folds

Seed42, both label policies, random / recording / timeblock, five folds, four feature sets (full_numeric42, missing_only42, context_only8, imu_only12), four models:

- RF100: original100 trees, depth20, min_leaf5, sqrt, balanced, seed42, n_jobs4.
- ExtraTrees100: 100 trees, depth20, min_leaf5, sqrt, balanced, bootstrap=False, seed42, n_jobs4.
- HistGB: learning_rate0.1, max_iter100, max_leaf_nodes31, min_samples_leaf20, l2_regularization1, class_weight=balanced, early_stopping=False, random_state42; other installed sklearn defaults recorded. No internal random validation or tuning.
- Logistic regression: training-median imputation, training StandardScaler, C1, solver=lbfgs, max_iter2000, class_weight=balanced, tol1e-4, random_state42. Record convergence warnings and iterations, do not raise iteration limit after seeing scores.

All models use the original training-only median convention, dropping columns with no finite training value; fallback one zero feature if no columns remain. Missing-only uses finite masks of all42 fields, not the availability of target/time/IDs. As in the parent runner, missing-only drops fields whose underlying numeric value was never observed among fit rows; it does not keep such fields merely because their all-zero mask is finite. GPS-mask follows the same rule for its underlying fields. For full_with_mask, apply the ordinary finite-training-column rule to the concatenated84 columns (an all-zero mask is a retained constant even if its all-missing numeric column is dropped); this configuration is explicitly separate from missing-only. Preserve original string labels during model fitting; encode labels only for storage so RF reproduction does not change sorted class order. Model/BLAS thread limits4. Estimator failures/timeouts are retained and not silently replaced. An individual fit may be bounded at120s if necessary; mark failures rather than invent scores or change parameters.

Re-fit RF100 with compact artifacts and compare its seed42 predictions against the original corresponding run. This checks the new runner before interpreting new models; a mismatch must be investigated, not averaged away. The main480 includes120 refits of already examined RF diagnostic models, not480 novel independent confirmations.

## B. Fixed-seed robustness: 240 model-folds

All three non-RF models, full_numeric and context_only, random / recording, both policies, five folds, seeds43 and44. Their full model, split and fit-sampling seeds change together. No choosing models based on table A. Original RF43/44 is reused as historical stability evidence; not counted as new fits. Timeblock seed42 is descriptive, not randomized independence.

## C. Feature-family ablations: 240 model-folds

RF100 seed42, both policies, all three split types, five folds, eight additional fixed sets:

1. gps_values: location_latitude + location_longitude (2).
2. weather_only: the exact four existing weather columns (4).
3. elevation_only: GIS_Elevation (1).
4. sensor_only: full_numeric minus all eight context columns (34).
5. imu_plus_gps: existing12 IMU axes plus latitude/longitude (14).
6. imu_plus_weather: IMU12 plus four weather fields (16).
7. gps_mask: original finite masks of latitude/longitude only (2).
8. full_with_mask:42 numeric fields concatenated with their42 original finite masks (84).

Names and lists must be materialized and hashed before fitting. gps_values with imputation is not pure geographic-value information: its imputed constants may also expose missingness. Contrasts are predictive associations, not interventions or causal attribution. The observed source schema and missingness are never repaired with fabricated coordinates.

## D. Capacity and budget sensitivity: 90 new model-folds

full_numeric only, seed42, both policies, allthree splits, fivefolds. RF100 capped20k is the reference from A. Add RF100 uncapped; RF300 capped20k; RF300 uncapped. RF300 uses300 trees, depth25, min_leaf1, sqrt, balanced, seed42, fourthreads. This is a paper-inspired capacity setting, not a replication of the source paper's unidentified eight files/37 features or98.23% accuracy. RF100/300 changes multiple capacity parameters and cannot isolate the causal effect of tree count. Training labels/preprocessing/splits unchanged. Report all four capacity×budget conditions.

## E. Known-anomaly two-class check: 60 model-folds

Use only explicit Pothole/Bump rows from the mixed recordings with finite latitude and longitude; no inferred Normal labels and no Normal-only directory. Report removals and class support before fitting. This changes the task to Bump/Pothole classification and cannot be compared as the same three-class benchmark. RF100 seed42, four original feature sets full/missing/context/imu, random / recording / timeblock, fivefolds. Rebuild random StratifiedKFold on this two-class population; recording map stays original seed42 map restricted to these records; timeblock raw-source assignment unchanged. Alltrain/test folds need both classes or stop this block without reseeding. Class metrics use only the two explicit labels. This checks whether patterns require the Normal-directory contrast, not whether all missingness confounding is eliminated.

## F. Data audit, uncertainty and simple diagnostics

Audit all101 live source hashes, original schema, finite rates, within-record constants, class composition, time spans/cadence, exact retained-row duplicates and GPS grid overlap. Distinguish duplicate numerical vectors from confirmed duplicated physical events; report denominators. GPS-only overlap cannot locate the Normal-only files with missing coordinates.

Report both fold-mean macroF1 and pooled out-of-fold macroF1, balanced accuracy, per-class recall/confusion, source-level performance and class supports. OOF comparisons on the same scoreable rows can use paired prediction contrasts, even though old fold-number random-vs-group differences were not paired same-example tests. Do not combine the two definitions.

For main seed42 OOF contrasts (random−recording, random−timeblock forfull; context−full underrecording; sensor−full whereavailable), use2,000 percentile bootstrap replicates with fixedseed20261016, resampling originalCSV clusters separately within the five Normal-only and96 mixed-source strata. Use one draw perrecord for allmethods in a contrast, accumulating perrecord confusion matrices; preserve test cohort strata and never resample individual rows as independent. This estimates conditional prediction/cohort sensitivity with fixed trained models, not full training/CV uncertainty or an independently sampled vehicle/road population. Onlyfive Normal-source clusters is a serious limitation; report record counts and descriptive intervals, no claims of population significance from seeds or dozens of folds. Infeasible/degenerate replicates are counted and disclosed, not redrawn until favorable.

Optional zero-fit diagnostics, if implemented, are fixed prior-majority prediction and same-record nearest-training-timestamp label lookup with global training-majority fallback for unseen records; ties use earliest source row, all onlyfold-fit labels. The latter explicitly consumes recording ID/time to measure temporal label continuity, is excluded from ordinary deployable model rankings, and is not a new algorithm. It must use the identical capped fit rows, never testlabels. Results cannot be used to tune block sizes.

Wall times characterize this run only; concurrent workloads and preprocessing differences prevent treating them as controlled hardware benchmark results. No new mathematical novelty claims follow from a bootstrap or a feature ablation.

## G. Separate-source evidence

Search accessible raw smartphone road datasets with explicit target semantics and independently verifiable recording/vehicle IDs. Dataset availability, licensing, mapping and acquisition independence must be audited before any models. Freeze a dataset-specific extension after inspecting metadata but before its model scores. If the data has no confirmed Normal labels or shares acquisitions withRoadSens, report that limitation and choose a compatible task; no label guessing or silent transfer of normal definitions. Independent within-dataset replication of the split audit is not trained-on-RoadSens external-transfer validation. Do not claim successful cross-device/vehicle transfer without actually performing it on aligned measurements and labels.

## Execution and artifacts

Run in new paper_expansion directories; never overwrite exploratory evidence. Before fitting freeze source/cache/parent/code/protocol/config/split hashes and exact fit/test row identities. Keep split membership once, compressed per-modelOOF predictions, model/preprocessing artifacts, warnings/failures, metrics and a machine-readable completion matrix. Resume only exact matching manifests and validated complete checkpoints. A changed implementation gets a new version and explicit reason.

Planned RoadSens new fits: A480+B240+C240+D90+E60=1,110, excluding optional zero-fit diagnostics and separately frozen external-source experiments. A large fit count is not an independent sample count or a contribution by itself. Final report must state completed/missing blocks, results that weaken the hypothesis, and remaining manuscript limits.
