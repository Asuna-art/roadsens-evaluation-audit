# Fit-free sensitivity amendment: disjoint raw windows

Declared 16 October 2026 before any PVS model fit or score. The primary protocol remains unchanged (264 fits). This addition tests whether its within/across-acquisition contrast depends entirely on overlapping input windows.

From the primary feature population retain anchors 99,199,299,... in each original acquisition (`(source_end - 99) % 100 == 0`). These are the same 100-row trailing windows and anchor labels, now on a 100-row grid. No alternate feature construction, relabelling, sensor location or test balancing is allowed. Save a separate features NPZ/manifest with parent hashes and exact retained primary indices. All raw source windows within an acquisition are now pairwise disjoint, and random validation must have zero raw-row intersections as an explicit fit-free gate.

Repeat the identical six outer folds, three selectors, four candidates and 264-fit matrix on this subset, in a separate immutable output directory. Independently verify all 264 predictions/models as for the primary grid. Main-grid selection records and results are not modified. Both grids thus total 528 planned PVS fits, but still only nine acquisition units; grids are sensitivity analyses of shared observations, not independent replications.

Report the same optimism and estimation-error endpoints on both grids, retaining every configuration. The primary two-axis 0.02 criterion belongs to the predeclared 25-row grid; report the 100-row result separately as a sensitivity, never substitute whichever passes. The smaller, nonoverlapping population changes training and test sample counts and cannot identify overlap as the sole causal mechanism. No additional seeds, feature sets or thresholds may be selected from the results.
