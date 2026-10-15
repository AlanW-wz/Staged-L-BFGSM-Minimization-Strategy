# -*- coding: utf-8 -*-
"""Run omega-parameter comparison on Levy-like nonconvex test functions."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
DESIGN = next(p.parent for p in ROOT.glob("*/hybrid_methods.py"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DESIGN) not in sys.path:
    sys.path.insert(0, str(DESIGN))

from config import GRAD_TOL, MAX_ITER  # noqa: E402
from hybrid_methods import MethodResult, run_method_on_problem, run_sd_lbfgsm  # noqa: E402
from lbfgsm_algorithm import lbfgsm, strong_wolfe_line_search  # noqa: E402


Vector = np.ndarray
Objective = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]

RUN_DIMS = [10000, 50000, 100000, 500000]
METHODS = [
    "lbfgsm_wolfe_c000",
    "lbfgsm_wolfe",
    "lbfgsm_wolfe_c025",
    "lbfgsm_wolfe_c025_c10",
    "sd_lbfgsm_c000",
    "sd_lbfgsm",
    "sd_lbfgsm_c025",
    "sd_lbfgsm_c025_c10",
]
OUT = ROOT / "实验报告输出" / "omega_levy_like"
OUT.mkdir(parents=True, exist_ok=True)


def levy_scaled(x: Vector) -> float:
    z = 0.75 * x
    w = 1.0 + (z - 1.0) / 4.0
    first = math.sin(math.pi * w[0]) ** 2
    u = w[:-1]
    middle = np.sum((u - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * u + 1.0) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return float(first + middle + last)


def grad_levy_scaled(x: Vector) -> Vector:
    z = 0.75 * x
    w = 1.0 + (z - 1.0) / 4.0
    gw = np.zeros_like(x)
    gw[0] += math.pi * math.sin(2.0 * math.pi * w[0])
    u = w[:-1]
    arg = math.pi * u + 1.0
    gw[:-1] += 2.0 * (u - 1.0) * (1.0 + 10.0 * np.sin(arg) ** 2)
    gw[:-1] += (u - 1.0) ** 2 * 10.0 * math.pi * np.sin(2.0 * arg)
    u_last = w[-1]
    last_arg = 2.0 * math.pi * u_last
    gw[-1] += 2.0 * (u_last - 1.0) * (1.0 + math.sin(last_arg) ** 2)
    gw[-1] += (u_last - 1.0) ** 2 * 2.0 * math.pi * math.sin(2.0 * last_arg)
    return 0.75 * gw / 4.0


def levy_shifted(x: Vector) -> float:
    return levy_scaled(x - 0.5)


def grad_levy_shifted(x: Vector) -> Vector:
    return grad_levy_scaled(x - 0.5)


def cosine_modulated_quadratic(x: Vector) -> float:
    return float(0.5 * np.sum(x * x) + 0.15 * np.sum(1.0 - np.cos(3.0 * x)))


def grad_cosine_modulated_quadratic(x: Vector) -> Vector:
    return x + 0.45 * np.sin(3.0 * x)


def sine_coupled_quadratic(x: Vector) -> float:
    z = x[:-1] + x[1:]
    return float(0.5 * np.sum(x * x) + 0.08 * np.sum(np.sin(z) ** 2))


def grad_sine_coupled_quadratic(x: Vector) -> Vector:
    g = x.copy()
    z = x[:-1] + x[1:]
    q = 0.16 * np.sin(z) * np.cos(z)
    g[:-1] += q
    g[1:] += q
    return g


def trig_chain(x: Vector) -> float:
    d = x[1:] - x[:-1]
    return float(0.5 * np.sum(x * x) + 0.05 * np.sum(1.0 - np.cos(2.0 * d)))


def grad_trig_chain(x: Vector) -> Vector:
    g = x.copy()
    d = x[1:] - x[:-1]
    q = 0.10 * np.sin(2.0 * d)
    g[:-1] -= q
    g[1:] += q
    return g


PROBLEMS: dict[str, tuple[str, Objective, Gradient, int]] = {
    "levy_scaled": ("Scaled Levy", levy_scaled, grad_levy_scaled, 3100),
    "levy_shifted": ("Shifted Levy", levy_shifted, grad_levy_shifted, 3200),
    "cosine_modulated_quadratic": (
        "Cosine-modulated quadratic",
        cosine_modulated_quadratic,
        grad_cosine_modulated_quadratic,
        3300,
    ),
    "sine_coupled_quadratic": (
        "Sine-coupled quadratic",
        sine_coupled_quadratic,
        grad_sine_coupled_quadratic,
        3400,
    ),
    "trig_chain": ("Trigonometric chain", trig_chain, grad_trig_chain, 3500),
}


def x0(dim: int, seed: int) -> Vector:
    rng = np.random.default_rng(seed + dim)
    return rng.uniform(-3.0, 3.0, size=dim).astype(float)


def run_c000(method: str, problem: str, f, grad_f, start: Vector, max_iter: int) -> MethodResult:
    grad_hist: list[float] = []
    time_hist: list[float] = []
    t0 = time.perf_counter()
    if method == "lbfgsm_wolfe_c000":
        res = lbfgsm(
            f,
            grad_f,
            start,
            m=8,
            c0=0.0,
            c1=0.0,
            c2=0.0,
            tol=GRAD_TOL,
            max_iter=max_iter,
            line_search_fn=strong_wolfe_line_search,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - t0
        return MethodResult(
            method,
            problem,
            start.size,
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
        start,
        max_iter,
        GRAD_TOL,
        c0=0.0,
        c1=0.0,
        c2=0.0,
        grad_hist=grad_hist,
        time_hist=time_hist,
    )
    elapsed = time.perf_counter() - t0
    return MethodResult(
        method,
        problem,
        start.size,
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


def run_one(method: str, problem: str, f, grad_f, start: Vector, max_iter: int) -> MethodResult:
    if method.endswith("c000"):
        return run_c000(method, problem, f, grad_f, start, max_iter)
    result, _ = run_method_on_problem(method, problem, f, grad_f, start, max_iter, GRAD_TOL)
    return result


def main() -> None:
    json_path = OUT / "omega_levy_like_results.json"
    csv_path = OUT / "omega_levy_like_summary.csv"
    rows: list[MethodResult] = []
    if json_path.exists():
        rows = [MethodResult(**row) for row in json.loads(json_path.read_text(encoding="utf-8"))]
    done = {(r.problem, int(r.dimension), r.method) for r in rows}
    total = len(PROBLEMS) * len(RUN_DIMS) * len(METHODS)
    count = 0
    for key, (display, f, grad_f, seed) in PROBLEMS.items():
        for dim in RUN_DIMS:
            start = x0(dim, seed)
            for method in METHODS:
                count += 1
                if (key, dim, method) in done:
                    print(f"[skip] {display:28s} n={dim:<6d} {method}")
                    continue
                print(f"[{count}/{total}] {display:28s} n={dim:<6d} {method}", flush=True)
                result = run_one(method, key, f, grad_f, start, MAX_ITER[dim])
                rows.append(result)
                done.add((key, dim, method))
                json_path.write_text(
                    json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
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
                            "f_x",
                            "grad_norm",
                            "elapsed_seconds",
                            "x_norm",
                        ],
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {
                                "problem": row.problem,
                                "dimension": row.dimension,
                                "method": row.method,
                                "converged": row.converged,
                                "iterations": row.iterations,
                                "f_x": row.f_x,
                                "grad_norm": row.grad_norm,
                                "elapsed_seconds": row.elapsed_seconds,
                                "x_norm": row.x_norm,
                            }
                        )
                print(
                    f"      iter={result.iterations:<4d} cpu={result.elapsed_seconds:.4f} "
                    f"grad={result.grad_norm:.3e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
