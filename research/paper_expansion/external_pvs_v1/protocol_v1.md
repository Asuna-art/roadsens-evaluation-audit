# PVS independent-source replication and deployment-matched validation

Frozen design: 16 October 2026, before any PVS model fit or score. This is a prospective extension of a previously explored RoadSens audit, not preregistration of the whole study. Source/schema preflight may reject infeasible inputs; any changes must be documented before fitting. Null and contrary results are retained.

## Source and task

Use all nine Passive Vehicular Sensors (PVS) acquisitions from Menegazzo and von Wangenheim's official repository-linked release. Three vehicle/driver pairs cross three scenarios: trips 1–3 pair 1, trips 4–6 pair 2, trips 7–9 pair 3; scenarios cycle 1,2,3. Vehicle and driver are confounded. Scenario is the author's acquisition descriptor, not independently sampled geography. This is a different research team, hardware and collection from RoadSens; it does not align the datasets' labels or establish a RoadSens-to-PVS transfer experiment.

Primary task: author-provided `no_speed_bump` versus either `speed_bump_asphalt` or `speed_bump_cobblestone`. Require exactly one of these three indicators at each retained anchor. No-speed-bump is NOT a verified normal road, and no pothole category is created. Do not substitute road-surface or roughness classes if this binary task fails without a recorded fit-free amendment. Retain all nine acquisitions if all have both binary labels and synchronized inputs; otherwise report feasibility failure before choosing an alternative.

Use only the left-network cabin/dashboard accelerometer and gyroscope (six axes); do not count left/right networks or multiple mounts as independent acquisitions. Verify column meanings against official documentation/code. Prefer the combined GPS/MPU left file plus corresponding labels; validate row identity, timestamps and length before joining, never silently truncate or nearest-match ambiguous labels. Preserve original files and download URLs/hashes. Derived data are for local analysis; the manuscript bundle does not redistribute the raw release.

## Deterministic observations and candidates

From each acquisition's chronological sensor rows, form trailing windows of 100 consecutive rows with anchors 99,124,149,... (25-row stride). At nominal 100 Hz this is approximately one second. Report observed timestamp spans rather than silently assert uniform one-second timing. No window crosses an acquisition boundary; label is the author's label at the final anchor, not a whole-window event label. Retain only finite six-axis windows and unambiguous anchor labels. Declare all exclusions and class supports. No balancing the test population or selecting events by model performance.

Sensor representation: per-axis mean, population standard deviation, RMS, range and maximum absolute magnitude (30 columns). Context representation: same 30 columns plus anchor latitude, longitude and speed (3); missing context is handled by training-only median imputation. Timestamps, row numbers, labels, trip/vehicle/scenario identifiers and target-derived columns are never predictors. All feature construction is per-window; no global normalization. Require each feature finite somewhere in every training partition, otherwise halt before fit rather than silently drop it.

Four fixed candidates, in tie order: RF100/sensor, RF100/sensor+context, HGB/sensor, HGB/sensor+context. RF: 100 trees, max_depth=20, min_samples_leaf=5, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=4. HGB: max_iter=100, max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1, class_weight=balanced, early_stopping=False, random_state=42. Training-only median imputation, no scaling. Locked Python/sklearn runtime is recorded. No hyperparameter tuning beyond selecting these four predeclared candidates.

## Outer assessment and three validation selectors

Evaluate two axes separately: leave-one-vehicle/driver-pair-out (three outer folds), and leave-one-author-scenario-out (three outer folds). Each outer fold holds three complete acquisitions absent from training. The remaining six acquisitions form a development population. Outer memberships are fixed from metadata, not searched by score. Test outcomes are not used for model selection.

On each development population and each candidate compare:

1. Stratified two-fold random-window validation, shuffle=True, seed=42. This intentionally measures within-acquisition interpolation and may share raw source rows between training and validation windows. Measure and report this sharing; do not present it as new-acquisition validation.
2. Leave-one-development-trip-out validation (six folds). No raw source rows are shared across its partitions, but another acquisition from the same vehicle/driver and scenario can remain in training.
3. Deployment-matched validation: leave-one-remaining-vehicle/driver-pair-out when the outer axis is vehicle; leave-one-remaining-scenario-out when the outer axis is scenario (two inner folds). Entire source acquisitions stay together and no raw windows cross the split.

Each selector chooses the candidate with highest pooled inner out-of-fold macro-F1; exact ties use the declared order. Fold means are supplementary, never counted as additional replications. Save all inner predictions and freeze selection records before any outer predictions. Refit all four candidates on the full development population once, for prespecified selection comparisons and descriptive oracle-regret diagnostics. The oracle is not a proposed deployable selector. Three selectors reuse these same four outer fits.

Maximum training budget 20,000 windows per fit, proportional class-stratified sampling without replacement with stable saved per-fit RNG and exact indices. Test rows are never capped. Save capped and uncapped counts; inner training sizes differ across schemes (especially six-fold trip validation), and any score differences are not attributable to split semantics alone. Two-fold random and matched validation use similar, not guaranteed identical, training fractions. Total: 6 outer folds × 4 candidates × (2 random + 6 trip + 2 matched + 1 outer refit) = 264 fits. No seeds or repeated windows are independent acquisitions.

## Endpoints and interpretation

Replication endpoint: for each axis and fixed candidate, compare mean inner random score with mean outer score, and report all domain-level values. Macro-F1 and balanced accuracy are required, with class recalls and confusion matrices. A positive random-minus-outer difference supports optimism for that configuration; absence/reversal is retained. No significance claims from only three outer groups per axis.

Primary evaluation-improvement endpoint: for each selector and outer domain, compare its selected candidate's inner pooled macro-F1 estimate with that same candidate's outer macro-F1. Report signed error and absolute error, and mean absolute error separately for the vehicle and scenario axes. Call the two-axis improvement criterion supported only if matched validation reduces mean absolute error by at least 0.02 macro-F1 versus random on BOTH axes. Otherwise call it partial support or unsupported. This threshold is a declared practical criterion, not a statistical significance level or known optimal choice.

Secondary endpoints: outer macro-F1 of the three selected systems, candidate-level estimation errors, and regret against the predeclared four-candidate outer oracle. Lower estimation error need not improve classifier accuracy; report these as distinct outcomes. Also report trip validation even if it performs better than the matched scheme. A conservative underestimate is not automatically better. The outer test task is group transfer within PVS, not a full three-class independent external test for RoadSens or transfer of RoadSens-trained weights.

## Provenance and gates

Before fitting: save feature/source/protocol/code hashes, exact outer and inner maps, class supports and training-cap indices. Check group disjointness and raw-row overlap (zero for grouped splits). Models, preprocessing, predictions and selected identities are saved. Independently reload all 264 fitted models to reproduce predictions and recompute metrics, and verify every outer window is held out once per axis for each candidate. Existing RoadSens inventories remain immutable. Failed attempts are preserved and excluded from canonical counts. Numerical tables in any updated TeX/PDF must match the checked result files.
