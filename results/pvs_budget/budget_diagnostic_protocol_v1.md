# PVS training-budget diagnostic, version 1

Design fixed on 2026-10-25 after the original PVS results were inspected. This is an explicitly post hoc diagnostic, not an independent confirmation or a replacement for the failed original two-axis endpoint. No new candidate, decision threshold or source acquisition is introduced. Original artifacts remain immutable.

## Question and population

Does the difference between random-window, leave-one-trip and deployment-matched validation persist when every fit being compared uses exactly the same number of examples from each class? Use only the frozen disjoint-window population (10,804 windows, 170 speed-bump anchors, nine acquisitions), chosen in advance to avoid direct raw-row overlap as an additional explanation. There is no search over strides, seeds, budgets or candidate sets.

The existing nonoverlap/run_v1/preflight.json is the sole split authority. Load its exact outer memberships, inner test arrays and historical train_indices_fit arrays. Do not reconstruct splits from seeds or call a split generator. Existing capped fit arrays are the allowable training pools. Keep all historical validation/test observations and labels untouched. Hash the feature NPZ, original preflight, original summary, this protocol and code before any fit.

## Exact budget intervention

For each outer domain separately, find the minimum available count for class 0 and for class 1 across its ten inner training pools (two random, six trip, two matched) AND its historical outer-refit training pool. These two minima define the common class-count budget for every inner fit and one shared outer refit in that domain. Reject a domain if either class has fewer than two observations or any declared feature has no finite value in selected training rows. Do not revise the budget after observing scores.

Sample each class without replacement from the sorted historical fit pool with one fixed seed 20261025 plus 100 times the zero-based original outer position plus inner position (random 0-1, trip 2-7, matched 8-9, outer 10). Save the exact sorted selected indices before fitting. Every candidate within a split uses the same selected rows. There are no repeats. Budgets can differ between outer domains, so interpretation is paired within each domain.

Use exactly the original four candidates and original preprocessing/model settings from run_nested_evaluation.py: RF100 and HGB with sensor or sensor+context features, balanced class weighting, random_state 42, training-only median imputation. Float32 model input; same locked research interpreter C:/Users/Administrator/miniconda3/python.exe. Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 4 before numerical imports. No feature tuning, augmentation, early stopping or model expansion.

## Outcomes and order

Fit the ten inner folds for all four candidates, pool inner OOF predictions separately by validation method, and select maximum pooled macro-F1 using the original candidate order for ties. Save and hash all selections before fitting/predicting the budget-matched outer models. Fit four outer candidates once per domain on the exact common-budget outer training subset, shared by all three selectors. Planned new fits: 6 domains x 4 candidates x 11 fits = 264. Save all fitted models, fit/test indices, true labels and predictions. All outer domains and candidates must be retained, including unfavorable results.

Primary diagnostic estimand: selected-system estimation MAE and selected-system outer macro-F1 for the common-budget outer fits. Report each of three domains separately and the descriptive mean per axis, signed error defined as outer minus inner, absolute error, class counts and selected identities. For each axis report random MAE minus matched MAE. Compare these descriptively with the original nonoverlap outputs, without calling a sign change a successful new method.

Secondary diagnostic estimand: apply the newly frozen selections/inner estimates to the historical full-development outer candidate predictions, reused from saved files. This preserves the original deployment refit target but intentionally retains the inner-to-full-refit sample-size gap. Distinguish this outcome from the equal-budget target. No new full-development outer fit is needed. Report candidate-level errors too, to separate selection changes from score changes.

The original practical threshold (at least 0.02 MAE reduction on both axes) remains an archival original-study criterion. Do not use this post hoc experiment to turn its failure into support. This intervention changes training composition, class prevalence and effective training size together; it can diagnose sensitivity to a common budget but cannot identify a causal fraction of error attributable solely to size, establish independent-road generalization, or validate a revised algorithm.

## Verification and stopping

Before run: save exact selected arrays and per-fold class/overlap/metadata checks; require disjoint fit/test indices, zero intersecting source-row intervals in the disjoint population, and exact original grouped memberships. Verify class support and identical class counts across all 11 fits per domain. Review this preflight before fitting.

After run: a separate verifier must reload every saved model, reproduce predictions, recompute fixed-label binary confusion matrices/macro-F1, OOF coverage, tie-breaking, all domain and axis outcomes, and confirm selections were frozen before outer results. Hash-bound historical inputs must remain unchanged. Output a completion inventory. A failed/partial run is retained and explicitly labeled; no silent reruns into an existing output directory.

No confidence intervals or population-significance claims will be made from three outer domains per axis. New independent acquisitions remain required for confirmatory validation of any future proposal.
