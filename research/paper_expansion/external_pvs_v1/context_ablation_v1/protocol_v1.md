# PVS GPS and speed block ablation — frozen follow-up diagnostic

Declared 17 October 2026 before fitting any of the 48 added models. This is a follow-up motivated by inspected v3 PVS outcomes, not an untouched confirmatory experiment. The parent study's two-axis estimation-error criterion failed and remains failed; this diagnostic cannot replace that endpoint or amend earlier selections.

## Fixed observations, partitions and budgets

Use the existing primary 100-row/25-stride population (43,204 windows, 682 speed-bump anchors) and its existing disjoint 100-row/100-stride subset (10,804 windows, 170 speed-bump anchors). Nine original acquisitions, three confounded vehicle/driver pairs crossed with three author-defined scenarios. No-speed-bump is not a general normal-road class. Source values, labels, eligibility, windows and preprocessing inputs are unchanged.

For every grid and each of the six saved outer domains (three vehicle/driver and three scenario), load exact `train_indices_uncapped`, `train_indices_fit` and `test_indices` from the parent's saved outer-refit predictions. Require equality across the parent's four candidates and consistency with their hash-bound input provenance and saved fold plan. Do not regenerate membership, seeds or proportional caps. Test windows remain uncapped; all original finite-feature and binary-support gates apply.

## Four feature blocks and two fixed classifiers

The stored `X_sensor` contains the same 30 six-axis trailing-window statistics. The stored `X_context` has exactly latitude, longitude and speed in that order. Compare:

- IMU: the original 30 features, reusing existing outer fitted models/predictions.
- IMU+speed: the 30 features plus context column 2, 31 predictors, new fits.
- IMU+GPS: the 30 features plus context columns 0 and 1, 32 predictors, new fits.
- IMU+GPS+speed: the original 33 predictors, reusing existing outer fitted models/predictions.

No timestamp, source identifier or target-derived predictor is added. Cast each model input to float32, and fit median imputation only on exact saved capped training rows, keep declared columns, no scaling. RF100: 100 trees, depth20, min_leaf5, max_features=sqrt, balanced classes, seed42, n_jobs4. HGB: max_iter100, max_leaf_nodes31, learning_rate0.1, L2=1, balanced classes, early_stopping=False, seed42. These are the parent's classifier settings. No tuning, new seeds, feature renormalization, threshold calibration or new candidate selection is performed.

Fit count: 2 grids × 6 saved outer folds × 2 classifiers × 2 new blocks = 48 new fits. The two baseline blocks reuse 48 previously verified outer fits; the merged diagnostic contains 96 domain-level model predictions, not 96 new fits. The unchanged prior canonical suite is 2,238 fits; adding 48 gives 2,286 total computational fits, including 576 PVS fits. Fits, windows, classifiers, grids and outer axes do not multiply the nine acquisition units.

## Endpoints and claims

Report all 96 domain macro-F1, balanced-accuracy, class-recall and confusion results; summarize macro-F1/BA by the unweighted mean of three outer domains per axis, retaining fixed configuration identities. For every grid/axis/classifier report all five feature contrasts: speed−IMU; GPS−IMU; all−IMU; all−speed (GPS conditional on speed); all−GPS (speed conditional on GPS). Also show their domain-level values and vehicle-axis minus scenario-axis mean gains. Axis differences are descriptive differences of means, not a test on matched independent domains.

The explanatory question is whether the previously observed contextual ranking reversal is more consistent with the GPS block, speed block, or their combination under these fixed pipelines. Retain every sign, magnitude and counterexample. Do not invent a pass threshold, select the most favorable grid/model/domain, or reinterpret a failed earlier primary endpoint as successful. Conditional block comparisons describe retrained predictive systems; added dimensions/interactions, correlated signals and acquisition differences prevent identifying GPS or speed as a sole physical cause. Removing a context block is not an intervention on road condition, and source GPS preprocessing is not audited here as an online causal pipeline.

These outer outcomes were already inspected to motivate this follow-up, so no new generalization guarantee, statistical significance, universal GPS-removal remedy, new validation algorithm or new independent external confirmation follows. Newly fitted block variants are diagnostics, not candidates retroactively inserted into the parent's nested selectors. Final paper must disclose this follow-up status and preserve the v3 parent results unchanged.

## Validation and delivery

Before fitting, bind source inputs, parent metadata/prediction/model hashes, exact partitions, this protocol and reviewed code in a fit-free review gate. Use a new exclusive output directory and refuse model overwrites. Save each new fitted pipeline, predictions, exact training/test identities, metrics and parent references. Independently reload all 48 new and 48 reused models to reproduce predictions, training medians, feature widths, classes and parameters; recompute all metrics/contrasts and check source/parent hashes. Failed attempts are retained, never silently overwritten.

Update new v4 manuscript and supplementary source/PDF files; retain v3. Supply editable TeX and successfully compiled PDF with all figure/body dependencies, checked numeric tables and a portable project ZIP. No raw PVS data or features are redistributed in the manuscript bundle, and no public submission or external message is authorized.
