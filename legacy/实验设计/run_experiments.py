# -*- coding: utf-8 -*-
"""
Run the full 实验设计 benchmark suite.

Usage (from repo root):
  python 实验设计/run_experiments.py
  python 实验设计/run_experiments.py --problems ackley --dims 500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    DIMENSIONS,
    GRAD_TOL,
    MAX_ITER,
    METHODS,
    PROBLEMS,
    RAW_DIR,
    RESULTS_DIR,
    SOLUTIONS_DIR,
)
from hybrid_methods import MethodResult, run_method_on_problem  # noqa: E402
from test_functions import get_problem, make_x0  # noqa: E402


def _result_dict(r: MethodResult) -> Dict:
    d = asdict(r)
    return d


def _load_completed(path: Path) -> Set[Tuple[str, str, int]]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    done: Set[Tuple[str, str, int]] = set()
    for row in data:
        done.add((row["problem"], row["method"], int(row["dimension"])))
    return done


def run_suite(
    problems: List[str],
    dims: List[int],
    methods: List[str],
    resume: bool = True,
) -> List[MethodResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "summary.json"
    all_results: List[MethodResult] = []
    done = _load_completed(summary_path) if resume else set()
    if summary_path.exists() and resume:
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            all_results.append(MethodResult(**row))

    total = len(problems) * len(dims) * len(methods)
    count = len(done)

    for prob_key in problems:
        display_name, f, grad_f, _ = get_problem(prob_key)
        prob_dir = RAW_DIR / prob_key
        prob_dir.mkdir(parents=True, exist_ok=True)

        for dim in dims:
            x0 = make_x0(dim, prob_key, seed_offset=0)
            x0_path = prob_dir / f"x0_dim_{dim}.npy"
            np.save(x0_path, x0)

            dim_results: List[Dict] = []
            solutions: Dict[str, np.ndarray] = {}
            histories: Dict[str, List[float]] = {}
            time_histories: Dict[str, List[float]] = {}

            for method in methods:
                key = (prob_key, method, dim)
                if key in done:
                    print(f"[skip] {prob_key} n={dim} {method}")
                    existing = next(
                        r for r in all_results
                        if r.problem == prob_key and r.method == method and r.dimension == dim
                    )
                    dim_results.append(_result_dict(existing))
                    sol_path = SOLUTIONS_DIR / f"{prob_key}_dim{dim}.npz"
                    if sol_path.exists():
                        loaded = np.load(sol_path)
                        if method in loaded:
                            solutions[method] = loaded[method]
                    continue

                count += 1
                t0 = time.perf_counter()
                print(f"[{count}/{total}] {display_name} n={dim} {method} ...", flush=True)
                result, x_final = run_method_on_problem(
                    method,
                    prob_key,
                    f,
                    grad_f,
                    x0,
                    MAX_ITER[dim],
                    GRAD_TOL,
                )
                elapsed_total = time.perf_counter() - t0
                result.elapsed_seconds = float(elapsed_total)
                all_results.append(result)
                dim_results.append(_result_dict(result))
                solutions[method] = x_final
                histories[method] = result.grad_history
                time_histories[method] = result.time_history
                done.add(key)

                summary_path.write_text(
                    json.dumps([_result_dict(r) for r in all_results], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            dim_dir = prob_dir / f"dim_{dim}"
            dim_dir.mkdir(parents=True, exist_ok=True)
            (dim_dir / "results.json").write_text(
                json.dumps(dim_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if solutions:
                np.savez_compressed(SOLUTIONS_DIR / f"{prob_key}_dim{dim}.npz", **solutions)
            if histories:
                (dim_dir / "grad_histories.json").write_text(
                    json.dumps(histories, ensure_ascii=False),
                    encoding="utf-8",
                )
            if time_histories:
                (dim_dir / "time_histories.json").write_text(
                    json.dumps(time_histories, ensure_ascii=False),
                    encoding="utf-8",
                )

    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "problem",
                "method",
                "dimension",
                "converged",
                "iterations",
                "f_x",
                "grad_norm",
                "elapsed_seconds",
                "x_norm",
                "switch_sd_to_mid_iter",
                "switch_mid_to_final_iter",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r.problem,
                    r.method,
                    r.dimension,
                    r.converged,
                    r.iterations,
                    r.f_x,
                    r.grad_norm,
                    r.elapsed_seconds,
                    r.x_norm,
                    r.switch_sd_to_mid_iter,
                    r.switch_mid_to_final_iter,
                ]
            )

    print(f"Saved {len(all_results)} runs to {RESULTS_DIR}")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 实验设计 benchmark suite")
    parser.add_argument("--problems", nargs="*", default=PROBLEMS)
    parser.add_argument("--dims", nargs="*", type=int, default=DIMENSIONS)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_suite(args.problems, args.dims, args.methods, resume=not args.no_resume)


if __name__ == "__main__":
    main()
