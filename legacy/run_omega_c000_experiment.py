# -*- coding: utf-8 -*-
"""Run omega-parameter comparison with (c0,c1,c2)=(0,0,0).

The setting is used as a pure L-BFGS-type baseline: the cautious-update
threshold is effectively removed, so curvature pairs are accepted without the
selective omega screening used by LBFGSM.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DESIGN = next(p.parent for p in ROOT.glob("*/hybrid_methods.py"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DESIGN) not in sys.path:
    sys.path.insert(0, str(DESIGN))

from config import GRAD_TOL, MAX_ITER  # noqa: E402
from hybrid_methods import MethodResult, run_method_on_problem  # noqa: E402
from test_functions import get_problem, make_x0  # noqa: E402


PROBLEMS = [
    "discus",
    "ellipsoidal",
    "rastrigin",
    "schwefel",
    "griewank",
    "bent_cigar",
    "levy",
    "dixon_price",
]

METHODS = ["lbfgsm_wolfe_c000", "sd_lbfgsm_c000"]
OUT = ROOT / "实验报告输出" / "omega_c000"
OUT.mkdir(parents=True, exist_ok=True)
RUN_DIMS = [10000, 50000, 100000, 500000]


def run_lbfgsm_c000_method(method: str, problem: str, f, grad_f, x0, max_iter: int, tol: float):
    from hybrid_methods import run_sd_lbfgsm
    from lbfgsm_algorithm import lbfgsm, strong_wolfe_line_search

    grad_hist: list[float] = []
    time_hist: list[float] = []
    import time

    start = time.perf_counter()
    if method == "lbfgsm_wolfe_c000":
        res = lbfgsm(
            f,
            grad_f,
            x0,
            m=8,
            c0=0.0,
            c1=0.0,
            c2=0.0,
            tol=tol,
            max_iter=max_iter,
            line_search_fn=strong_wolfe_line_search,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - start
        return MethodResult(
            method,
            problem,
            x0.size,
            bool(res.converged),
            int(res.n_iter),
            float(res.f),
            float(res.grad_norm),
            float(elapsed),
            float(np.linalg.norm(res.x)),
            None,
            None,
            list(res.grad_norm_history),
            list(res.time_history),
        )

    conv, iters, x, fx, gn, sw = run_sd_lbfgsm(
        f,
        grad_f,
        x0,
        max_iter,
        tol,
        c0=0.0,
        c1=0.0,
        c2=0.0,
        grad_hist=grad_hist,
        time_hist=time_hist,
    )
    elapsed = time.perf_counter() - start
    return MethodResult(
        method,
        problem,
        x0.size,
        conv,
        iters,
        fx,
        gn,
        float(elapsed),
        float(np.linalg.norm(x)),
        sw,
        None,
        list(grad_hist),
        list(time_hist),
    )


def main() -> None:
    rows: list[MethodResult] = []
    summary_path = OUT / "omega_c000_results.json"
    csv_path = OUT / "omega_c000_summary.csv"
    if summary_path.exists():
        rows = [MethodResult(**row) for row in json.loads(summary_path.read_text(encoding="utf-8"))]
    done = {(r.problem, int(r.dimension), r.method) for r in rows}
    for problem in PROBLEMS:
        _, f, grad_f, _ = get_problem(problem)
        for dim in RUN_DIMS:
            x0 = make_x0(dim, problem, seed_offset=0)
            for method in METHODS:
                if (problem, dim, method) in done:
                    print(f"[skip] {problem:12s} n={dim:<6d} {method:18s}", flush=True)
                    continue
                result = run_lbfgsm_c000_method(method, problem, f, grad_f, x0, MAX_ITER[dim], GRAD_TOL)
                rows.append(result)
                done.add((problem, dim, method))
                print(
                    f"{problem:12s} n={dim:<6d} {method:18s} "
                    f"iter={result.iterations:<4d} cpu={result.elapsed_seconds:.4f} "
                    f"grad={result.grad_norm:.3e}",
                    flush=True,
                )
                summary_path.write_text(
                    json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=[
                            "problem",
                            "dimension",
                            "method",
                            "converged",
                            "iterations",
                            "grad_norm",
                            "elapsed_seconds",
                        ],
                    )
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(
                            {
                                "problem": r.problem,
                                "dimension": r.dimension,
                                "method": r.method,
                                "converged": r.converged,
                                "iterations": r.iterations,
                                "grad_norm": r.grad_norm,
                                "elapsed_seconds": r.elapsed_seconds,
                            }
                        )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
