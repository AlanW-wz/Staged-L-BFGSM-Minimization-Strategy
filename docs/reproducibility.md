# Methods and provenance

## Parameter study

The seven objectives are Dixon–Price, Levy, scaled Levy, shifted Levy, Rastrigin,
Schwefel and the trigonometric chain. The last is an oscillatory strongly convex
control. The exact equations, gradients and base seeds are in
`experiments/parameter_study/objectives.py` and `config.py`.

The four dimensions are 10,000, 50,000, 100,000 and 500,000, with iteration budgets
300, 150, 100 and 80. The Euclidean gradient threshold is 1e-8. The four cautious
settings are (0,0,0), (0.9,1,1), (0.25,1,1) and (0.25,1,10); memory is eight.
The first disables the dynamic filter while retaining numerical safeguards.
The historical SciPy strong-Wolfe search, Armijo fallback and terminal flag
semantics are preserved. In particular, an accepted endpoint at the iteration
cap is not silently reclassified by a new stopping convention.

The extracted objective/gradient bodies and solver computational statements
were compared against the previous repository sources. Their original paths
and hashes are recorded in `experiments/parameter_study/source_provenance.json`;
the referenced originals are recoverable at the base commit in `provenance/migration.json`.

`data/parameter_study/results.csv` is the original 112-row table. Its runtime,
mean log-gradient and convergence fractions reproduce all 12 rounded values in
Table 2. `reported_memory_statistics.csv` is explicitly an archival report of
the remaining two columns; a complete iteration-level derivation is unavailable.
Selected historical reruns and environment metadata are retained under
`provenance/parameter_study/`.

## Local quench study

The six methods are standard L-BFGS, NCG–L-BFGSM, SD–L-BFGSM, SD–NCG,
SD–NCG–L-BFGSM and SD–standard L-BFGS. Function and display-name registries are
in `experiments/local_quench/problems.py` and `names.py`.
The final suite contains:

- Styblinski–Tang sum;
- two-coordinate three-hump camel, Bohachevsky-2 and Bohachevsky-3 blocks;
- eight-coordinate Griewank blocks;
- mixed/scaled quadratic–cosine variant and mixed/scaled Rastrigin;
- corrugated open Rosenbrock chain and eight-coordinate corrugated Rosenbrock blocks;
- heterogeneous periodic double-well chain.

The benchmark kernels in `experiments/local_quench/benchmarks.py` are drawn from
the author's AQN benchmark collection. The solver is checked against its archived
SHA-256. Each new run fingerprints the source files, settings and selected tasks.

Each method shares the same start from `default_rng(SeedSequence([seed, fid, dimension]))`.
Stable integer `fid` values and archival method keys are therefore not renumbered
when names are displayed. The stopping conditions are gradient infinity norm at
most 1e-6, 2,000 accepted steps or 20,000 joint objective/gradient evaluations.
All line-search trials count; unsuccessful final states remain in the data.
One numerical thread is used per worker, and each solve is measured with
`time.process_time()`; wall time is recorded separately.

After inspection of SD–NCG–L-BFGSM's residual, a supplementary start was used for
all six methods on the eight-coordinate corrugated Rosenbrock blocks at dimension
1,000. The original high residual was reproduced. The selection rationale, actual
seeds and batch assignment are recorded in `provenance/quench/replacement_plan.json`.
Use `paired_starts.csv` for the original-start comparison and `manuscript.csv`
for Figure 1. Both cover the same ten functions. The archived start audit refers
to the source CSV hashes listed as `source_sha256` in the data manifest.

Every point in the three-panel figure averages 300 terminal observations.
The highlighted range comprises the minimum and maximum of 30 ten-function batch
means. It is not an uncertainty estimate or confidence interval. Stored `fstar`
fields are inherited metadata; they do not certify a global optimum for each
transformed objective and are not used by the stationarity criterion.

## What is and is not rerun

The repository verification command checks measurements and aggregate calculations;
it does not repeat every high-dimensional optimization. Unit tests and the
documented smoke runs check gradients, seed streams, numerical update behavior,
evaluation accounting, output protection, checkpoint resume and figure aggregation.
Full replay commands are available separately. Original large SQLite traces and
unused endpoint-distance reconstructions are not bundled; new quench runs generate
their own SQLite outcome/trace checkpoints. Historical audit JSON files are retained
as records, not presented as newly performed trace audits.

No molecular simulation interface is reconstructed from analytical-function code.
The five-step molecular look-ahead rule and physical results require their own
simulation inputs and validation.
