"""Run the archived seven-function parameter study in a new output directory.

Examples:
  python -m experiments.parameter_study.run --plan-only
  python -m experiments.parameter_study.run --smoke --out results/parameter_smoke
  python -m experiments.parameter_study.run --dims 10000 --out results/parameter_10000

Full runs use one shared start per function/dimension and four parameter settings.
The historical solver is retained, including its finite-budget stopping behavior.
Use the separate local-quench runner for the newer threshold-switch experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from .config import (
    BUDGETS, DIMENSIONS, GRADIENT_TOLERANCE, MEMORY, PROBLEMS, ROOT, SETTINGS,
    setting_label,
)


def plan(smoke=False, dimensions=None):
    dimensions = tuple(dimensions or ((10,) if smoke else DIMENSIONS))
    if smoke and dimensions != (10,):
        raise ValueError("Smoke mode uses dimension 10 only")
    if not smoke and (not dimensions or len(set(dimensions)) != len(dimensions)
                      or any(n not in DIMENSIONS for n in dimensions)):
        raise ValueError(f"Dimensions must be a unique subset of {DIMENSIONS}")
    return {
        "study": "cautious-update parameter study",
        "mode": "smoke" if smoke else "formal",
        "target_runs": len(PROBLEMS) * len(dimensions) * len(SETTINGS),
        "dimensions": list(dimensions),
        "iteration_budgets": {str(n): 30 if smoke else BUDGETS[n] for n in dimensions},
        "gradient_norm": "Euclidean",
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "memory": MEMORY,
        "functions": PROBLEMS,
        "parameter_settings": SETTINGS,
        "initialization": "default_rng(base_seed + dimension).uniform(-3, 3, dimension)",
        "timing": "time.perf_counter elapsed wall-clock seconds",
        "line_search": "historical SciPy strong-Wolfe search with Armijo fallback",
        "limitations": [
            "An endpoint reached on the iteration cap retains the historical convergence flag.",
            "Mem. use and Pair acc. cannot be reconstructed from these archived outputs.",
            "This experiment does not test search-stage switching.",
        ],
    }


def output_path(path):
    """Resolve generated output beneath this repository's results directory."""
    path = Path(path)
    out = (path if path.is_absolute() else ROOT / path).resolve()
    results = ROOT / "results"
    if results.resolve() != results or out == results or results not in out.parents:
        raise ValueError("--out must be inside this repository's results/ folder")
    return out


def prepare_output(path):
    out = output_path(path)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Choose a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run(output, smoke=False, dimensions=None):
    # A missing SciPy dependency must not silently substitute a different search.
    import numpy as np
    import scipy
    from .objectives import initial_point, problem
    from .solver import lbfgsm, strong_wolfe_line_search

    design = plan(smoke, dimensions)
    out = prepare_output(output)
    numeric_dir = Path(__file__).resolve().parent
    design.update({
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
        "source_sha256": {
            name: hashlib.sha256((numeric_dir / name).read_bytes()).hexdigest()
            for name in ("config.py", "objectives.py", "solver.py", "run.py")
        },
    })
    atomic_json(out / "environment.json", design)
    rows, histories = [], {}
    for name, metadata in PROBLEMS.items():
        f, gradient = problem(name)
        histories[name] = {}
        for dimension in design["dimensions"]:
            start = initial_point(name, dimension)
            histories[name][str(dimension)] = {}
            budget = design["iteration_budgets"][str(dimension)]
            for method, (c0, c1, c2) in SETTINGS.items():
                begin = time.perf_counter()
                result = lbfgsm(
                    f, gradient, start, m=MEMORY, c0=c0, c1=c1, c2=c2,
                    tol=GRADIENT_TOLERANCE, max_iter=budget,
                    line_search_fn=strong_wolfe_line_search,
                    collect_grad_norm_history=True, collect_time_history=True,
                )
                elapsed = time.perf_counter() - begin
                row = dict(
                    problem=name, function_name=metadata["name"], method=method,
                    parameter_setting=setting_label(method), dimension=dimension,
                    converged=bool(result.converged), iterations=result.n_iter,
                    f_x=float(result.f), grad_norm=float(result.grad_norm),
                    elapsed_seconds=elapsed, x_norm=float(np.linalg.norm(result.x)),
                    seed=metadata["base_seed"] + dimension,
                    x0_sha256=hashlib.sha256(start.tobytes()).hexdigest(),
                )
                if not all(np.isfinite(row[key]) for key in
                           ("f_x", "grad_norm", "elapsed_seconds", "x_norm")):
                    raise FloatingPointError(f"Non-finite result for {name}/{dimension}/{method}")
                rows.append(row)
                histories[name][str(dimension)][method] = {
                    "grad": result.grad_norm_history, "time": result.time_history,
                }
                temporary = out / "results.csv.tmp"
                with temporary.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(out / "results.csv")
                atomic_json(out / "histories.json", histories)
                print(f"{len(rows)}/{design['target_runs']} {metadata['name']} "
                      f"n={dimension} {setting_label(method)}: G={result.grad_norm:.6g}", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="28 short runs, not a manuscript result")
    parser.add_argument("--plan-only", action="store_true", help="Print the design without importing numerical libraries or running")
    parser.add_argument("--dims", type=int, nargs="+", help="Subset of reported dimensions")
    parser.add_argument("--out", type=Path, help="New or empty directory under this repository's results/")
    args = parser.parse_args(argv)
    if args.smoke and args.dims:
        parser.error("--dims cannot be combined with --smoke")
    try:
        design = plan(args.smoke, args.dims)
        if args.plan_only:
            print(json.dumps(design, indent=2))
            return
        if args.out is None:
            parser.error("--out is required unless --plan-only is used")
        run(args.out, args.smoke, args.dims)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
