# Staged L-BFGSM minimization: numerical experiments

Code and archived analytical-function measurements for the revised manuscript
*Efficient Equilibration of Multicomponent PEDOT:PSS Systems Driven by a Staged
L-BFGSM Minimization Strategy*.

[中文说明](README_zh.md) · [Methods and provenance](docs/reproducibility.md) · [Migration record](docs/migration.md)

## The two experiments

| Study | Revised manuscript | Design | Implementation |
|---|---|---|---|
| Cautious-update parameters | Section 3.1.1, Table 2 | 7 functions × 4 dimensions × 4 settings = 112 runs | `experiments/parameter_study/` |
| Local-quench switching comparison | Section 3.1.2, Figure 1 | 10 functions × 10 dimensions × 30 paired starts × 6 methods = 18,000 runs per dataset | `experiments/local_quench/` |

The parameter study and quench study have different functions, budgets, gradient
criteria and timing measurements. Their implementations are kept separate.
The quench study uses threshold switching; it does not test the molecular
workflow's five-step branch comparison. GROMACS interfaces, molecular inputs,
MD trajectories and SAXS data are not supplied by this repository.

## Install

Use Python 3.12. The current checks use Python 3.12.14, NumPy 2.3.5,
SciPy 1.16.3 and Matplotlib 3.11.2.

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original parameter study used Python 3.12.7, NumPy 1.26.4 and SciPy 1.13.1.
Use `requirements-parameter-archive.txt` in a **separate environment** when
investigating that historical numerical behavior. Timing values and potentially
optimization trajectories depend on the runtime, hardware and thread settings.

## Verify and rebuild the archived results

These commands validate existing measurements and draw the selected figure;
they do not launch the full optimization studies.

```sh
python verify.py
python -m unittest discover -s tests -v
python -m experiments.parameter_study.summary --out results/parameter_summary.csv
python -m analysis.rebuild --out results/rebuilt_data
python replot.py --out results/quench_figure
```

`verify.py` checks file hashes, ten-function coverage, paired initial conditions,
stopping flags, the six observations at the supplementary start, 12 parameter-table
values and 60 Figure 1 aggregate cells. `analysis.rebuild` applies the documented
start selection to the paired measurements and reconstructs the manuscript CSV.
The figure command exports PDF, PNG, SVG, plot-data CSVs, a caption and a manifest.
Labels use method names; the caption does not list seed numbers.
All generated files go into ignored `results/` subdirectories. Choose a fresh
output directory when rebuilding or plotting; these commands refuse to replace
existing outputs.

## Run new experiments

Inspect the plan before executing a full workload:

```sh
python reproduce.py --study parameters --plan-only
python reproduce.py --study quench --plan-only
```

Small installation checks (not manuscript measurements):

```sh
python reproduce.py --study parameters --smoke --out results/parameter_smoke
python reproduce.py --study quench --smoke --out results/quench_smoke --workers 2
```

The parameter smoke has 28 runs at dimension 10; the quench smoke has 120 runs
at dimensions 16 and 32. Full commands are:

```sh
python reproduce.py --study parameters --out results/parameter_full
python reproduce.py --study quench --out results/quench_full --workers 4
```

The parameter runner accepts `--dims 10000` to select a reported dimension.
The quench runner additionally accepts `--functions`, `--dims`, `--seeds` and
`--methods`; use `python -m experiments.local_quench.run --help` for all options.
For example, the documented six-method supplementary start can be rerun with:

```sh
python -m experiments.local_quench.run --functions block8_corrugated_rosenbrock --dims 1000 --seeds 2026110731 --out results/supplementary_rerun --workers 1
```

Full studies can take substantial time. The quench runner checkpoints every
outcome with its accepted-iterate trace in SQLite. Repeating the identical
command resumes incomplete work; changes to sources, software or settings
require a new output directory. The parameter runner saves results after each
run but requires a new directory rather than resuming.

## Dataset selection and interpretation

Figure 1 compares six methods on ten functions. It is an exploratory comparison.
After inspection of the residuals, one starting point for the eight-coordinate
corrugated Rosenbrock blocks at dimension 1,000 was changed for all six methods.
A diagnostic rerun reproduced the original high residual. Both sets of six
observations are available for comparison.

| File under `data/local_quench/` | Meaning |
|---|---|
| `paired_starts.csv` | All 18,000 measurements at the original paired starts for these ten functions |
| `supplementary_runs.csv` | Six methods at the prescribed replacement start |
| `excluded_original_rows.csv` | The six original measurements replaced in the figure |
| `manuscript.csv` | The selected 18,000-run figure dataset |
| `figure1_plot_data.csv` | Archived means and ranges used to verify replotting |

Every plotted point averages all 300 outcomes, including budget-limited runs.
Shading is the observed minimum and maximum of 30 ten-function batch means,
not a confidence interval. A small gradient establishes stationarity rather
than a global minimum. Quench time is process CPU time; parameter-study time is
elapsed wall-clock time. The two historical memory statistics are preserved in
`data/parameter_study/reported_memory_statistics.csv`, with their incomplete
per-iteration provenance stated explicitly; the summary command does not invent
reconstructed values for them.

## Repository layout

```text
experiments/parameter_study/  Seven-function study, gradients, solver and runner
experiments/local_quench/    Paired quench runner, solver, ten-function benchmark definitions
analysis/                   Dataset reconstruction, validation and three-panel plotting
data/                       Archived measured CSVs and figure aggregates
provenance/                 Original metadata, checksums and migration decisions
docs/                       Reproduction details and removal/relocation record
tests/                      Numerical, checkpoint, archive and plot checks
results/                    Generated outputs (ignored by Git)
```

Legacy experiment IDs are retained inside archival CSVs and seed generation to
preserve compatibility. New reports use descriptive method and function names.
Retired quasi-Newton-family and eleven-function experiments, obsolete report
generators and duplicate implementations have been removed from the active tree;
their prior state remains recoverable from Git history.
