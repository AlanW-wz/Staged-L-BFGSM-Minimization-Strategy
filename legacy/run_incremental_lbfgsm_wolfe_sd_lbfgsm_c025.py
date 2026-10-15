# -*- coding: utf-8 -*-
"""Run LBFGSM-Wolfe and SD-LBFGSM with c0=0.25,c1=1,c2=1."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT / "实验设计"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DESIGN) not in sys.path:
    sys.path.insert(0, str(DESIGN))

from config import DIMENSIONS, GRAD_TOL, MAX_ITER, RAW_DIR, RESULTS_DIR, SOLUTIONS_DIR  # noqa: E402
from hybrid_methods import MethodResult, run_method_on_problem  # noqa: E402
from test_functions import get_problem, make_x0  # noqa: E402

METHODS = ["lbfgsm_wolfe_c025", "sd_lbfgsm_c025"]
PROBLEMS = ["discus", "ellipsoidal", "rastrigin", "schwefel", "griewank", "bent_cigar", "levy", "dixon_price"]


def load_summary(path: Path) -> list[MethodResult]:
    if not path.exists():
        return []
    return [MethodResult(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def write_summary_json(path: Path, rows: list[MethodResult]) -> None:
    path.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[MethodResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
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
        for r in rows:
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


def merge_json_dict(path: Path, key: str, value) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def merge_results_json(path: Path, result: MethodResult) -> None:
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = [row for row in rows if row.get("method") != result.method]
    rows.append(asdict(result))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_solution(path: Path, method: str, x_final: np.ndarray) -> None:
    payload = {}
    if path.exists():
        existing = np.load(path)
        payload = {name: existing[name] for name in existing.files}
    payload[method] = x_final
    np.savez_compressed(path, **payload)


def main() -> None:
    summary_json = RESULTS_DIR / "summary.json"
    summary_csv = RESULTS_DIR / "summary.csv"
    all_results = load_summary(summary_json)
    done = {(r.problem, r.method, int(r.dimension)) for r in all_results}

    total = len(PROBLEMS) * len(DIMENSIONS) * len(METHODS)
    count = 0
    for prob_key in PROBLEMS:
        display_name, f, grad_f, _ = get_problem(prob_key)
        for dim in DIMENSIONS:
            x0 = make_x0(dim, prob_key, seed_offset=0)
            for method in METHODS:
                count += 1
                if (prob_key, method, dim) in done:
                    print(f"[skip] {display_name} n={dim} {method}")
                    continue
                print(f"[{count}/{total}] {display_name} n={dim} {method} ...", flush=True)
                result, x_final = run_method_on_problem(method, prob_key, f, grad_f, x0, MAX_ITER[dim], GRAD_TOL)
                all_results.append(result)
                done.add((prob_key, method, dim))

                dim_dir = RAW_DIR / prob_key / f"dim_{dim}"
                dim_dir.mkdir(parents=True, exist_ok=True)
                merge_results_json(dim_dir / "results.json", result)
                merge_json_dict(dim_dir / "grad_histories.json", method, result.grad_history)
                merge_json_dict(dim_dir / "time_histories.json", method, result.time_history)
                merge_solution(SOLUTIONS_DIR / f"{prob_key}_dim{dim}.npz", method, x_final)
                write_summary_json(summary_json, all_results)
                write_summary_csv(summary_csv, all_results)

    write_summary_json(summary_json, all_results)
    write_summary_csv(summary_csv, all_results)
    print(f"Saved merged results: {summary_csv}")


if __name__ == "__main__":
    main()
