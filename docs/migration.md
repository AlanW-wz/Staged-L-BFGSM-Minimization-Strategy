# Repository reorganization

The previous tree at the base commit recorded in `provenance/migration.json`
described three superseded experiment labels and imported an absent `src` package.
The revised tree exposes two independent, runnable studies matching manuscript
Sections 3.1.1 and 3.1.2.

| Previous content | Current disposition |
|---|---|
| `legacy/lbfgsm_algorithm.py` | Computational solver retained in `experiments/parameter_study/solver.py`; demo removed |
| Old function collections and Levy-variant script | Required seven objective/gradient pairs extracted into `experiments/parameter_study/objectives.py` |
| `reference/experiment_B.csv` | Preserved byte-for-byte as `data/parameter_study/results.csv` |
| Selected parameter validation outputs | Preserved under `provenance/parameter_study/` |
| Quasi-Newton-family comparison and old eleven-function convergence experiment | Removed from active tree because these experiments were removed from the revised manuscript |
| Legacy one-off reports, cached figure generators and duplicate staged solvers | Removed; current execution and plotting use the documented portable entry points |
| Missing `src` imports in root entry points | Replaced by imports of the included `experiments` and `analysis` packages |
| Duplicate round-one/round-two quench runners and solvers | One runner and solver, with an explicit suite selector |
| Current three-panel plotting script | Ported to `analysis/quench_plot.py`; absolute four-panel dependency and external skill dependency removed |
| Four-panel endpoint-distance/refinement experiments | Not imported because the selected manuscript figure does not use them |
| Old package/source-selection manifests | Replaced by current data checksums, extraction provenance and an explicit migration record |

`provenance/migration.json` records 94 removed paths and their pre-removal hashes.
Some measurements and computational bodies were relocated, not discarded; the
table above identifies these cases. All prior files remain recoverable from Git
history. `provenance/data_manifest.json` identifies each imported archival file,
its original relative location and its unchanged hash.

The source directory `quench_study` and manuscript/SI files outside this repository
were read only. No large experiment was automatically rerun during reorganization.

Current checks include all archived table/figure aggregates, 23 unit tests, a
28-run parameter smoke and a 120-run quench smoke. Test reports under `results/`
are local generated outputs; CI runs the same commands on Windows and Linux.
