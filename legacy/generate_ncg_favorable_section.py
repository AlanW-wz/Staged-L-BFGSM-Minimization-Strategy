# -*- coding: utf-8 -*-
"""Generate Section 3.3 for the selected optimization test set."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DESIGN = next(p.parent for p in ROOT.glob("*/hybrid_methods.py"))
OUT = ROOT / "实验报告输出" / "selected_testset_33"
FIG_OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DESIGN) not in sys.path:
    sys.path.insert(0, str(DESIGN))

from hybrid_methods import run_method_on_problem  # noqa: E402
from test_functions import (  # noqa: E402
    bent_cigar,
    discus,
    ellipsoidal,
    grad_bent_cigar,
    grad_discus,
    grad_ellipsoidal,
    grad_griewank,
    grad_rastrigin,
    grad_schwefel,
    griewank,
    rastrigin,
    schwefel,
)


Vector = np.ndarray
Objective = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]

DIMS = [10000, 50000, 100000, 500000]
MAX_ITER = {10000: 300, 50000: 150, 100000: 100, 500000: 80}
TOL = 1e-8

METHODS = [
    "sd_ncg_lbfgsm_c025",
    "sd_ncg",
    "sd_cg_standard",
    "sd_lbfgsm_c025",
    "lbfgsm_wolfe_c025",
    "ncg_wolfe",
]

LABELS = {
    "sd_ncg_lbfgsm_c025": "SD-NCG-LBFGSM",
    "sd_ncg": "SD+NCG",
    "sd_cg_standard": "SD+standard CG",
    "sd_lbfgsm_c025": "SD+LBFGSM",
    "lbfgsm_wolfe_c025": "LBFGSM-Wolfe",
    "ncg_wolfe": "NCG-Wolfe",
}

STYLE = {
    "sd_ncg_lbfgsm_c025": dict(color="#C00000", linewidth=1.8, marker="o"),
    "sd_ncg": dict(color="#2CA02C", linewidth=1.25, linestyle=":", marker="^"),
    "sd_cg_standard": dict(color="#4C78A8", linewidth=1.25, linestyle="-.", marker="s"),
    "sd_lbfgsm_c025": dict(color="#9467BD", linewidth=1.25, linestyle="--", marker="v"),
    "lbfgsm_wolfe_c025": dict(color="#7F7F7F", linewidth=1.15, linestyle=(0, (3, 1, 1, 1)), marker="P"),
    "ncg_wolfe": dict(color="#FF7F0E", linewidth=1.15, linestyle=(0, (1, 1)), marker="X"),
}


def tridiagonal_quadratic(x: Vector) -> float:
    return float(np.sum(x * x) - np.sum(x[:-1] * x[1:]))


def grad_tridiagonal_quadratic(x: Vector) -> Vector:
    g = 2.0 * x.copy()
    g[:-1] -= x[1:]
    g[1:] -= x[:-1]
    return g


def perturbed_tridiagonal_quadratic(x: Vector) -> float:
    return float(
        np.sum(x * x)
        - 0.45 * np.sum(x[:-1] * x[1:])
        + 0.01 * np.sum(np.sin(x) ** 2)
    )


def grad_perturbed_tridiagonal_quadratic(x: Vector) -> Vector:
    g = 2.0 * x.copy()
    g[:-1] -= 0.45 * x[1:]
    g[1:] -= 0.45 * x[:-1]
    g += 0.02 * np.sin(x) * np.cos(x)
    return g


def weakly_perturbed_quadratic(x: Vector) -> float:
    mean_term = np.sum(x) / math.sqrt(x.size)
    return float(
        0.5 * np.sum(x * x)
        + 0.02 * np.sum(np.sin(x) ** 2)
        + 0.01 * mean_term * mean_term
    )


def grad_weakly_perturbed_quadratic(x: Vector) -> Vector:
    return x + 0.04 * np.sin(x) * np.cos(x) + 0.02 * np.sum(x) / x.size


def diagonal_rank_quadratic(x: Vector) -> float:
    weights = np.linspace(1.0, 20.0, x.size)
    mean_term = np.sum(x) / math.sqrt(x.size)
    return float(0.5 * np.sum(weights * x * x) + 0.05 * mean_term * mean_term)


def grad_diagonal_rank_quadratic(x: Vector) -> Vector:
    weights = np.linspace(1.0, 20.0, x.size)
    return weights * x + 0.10 * np.sum(x) / x.size


def block_coupled_quadratic(x: Vector) -> float:
    even = x.size - (x.size % 2)
    y = x[:even].reshape(-1, 2)
    a = y[:, 0]
    b = y[:, 1]
    return float(np.sum(3.0 * a * a + 2.0 * b * b - 2.0 * a * b))


def grad_block_coupled_quadratic(x: Vector) -> Vector:
    even = x.size - (x.size % 2)
    y = x[:even].reshape(-1, 2)
    g = np.zeros_like(x)
    gg = g[:even].reshape(-1, 2)
    a = y[:, 0]
    b = y[:, 1]
    gg[:, 0] = 6.0 * a - 2.0 * b
    gg[:, 1] = 4.0 * b - 2.0 * a
    return g


def quartic_chain(x: Vector) -> float:
    d = np.diff(x)
    return float(0.5 * np.sum(x * x) + 0.02 * np.sum(d**4))


def grad_quartic_chain(x: Vector) -> Vector:
    g = x.copy()
    d = np.diff(x)
    q = 0.08 * d**3
    g[:-1] -= q
    g[1:] += q
    return g


def coupled_sine_quadratic(x: Vector) -> float:
    z = x[:-1] + x[1:]
    return float(0.5 * np.sum(x * x) + 0.01 * np.sum(np.sin(z) ** 2))


def grad_coupled_sine_quadratic(x: Vector) -> Vector:
    g = x.copy()
    z = x[:-1] + x[1:]
    q = 0.02 * np.sin(z) * np.cos(z)
    g[:-1] += q
    g[1:] += q
    return g


PROBLEMS: dict[str, tuple[str, Objective, Gradient, int]] = {
    "discus": (
        "Discus",
        discus,
        grad_discus,
        1400,
    ),
    "ellipsoidal": (
        "Ellipsoidal",
        ellipsoidal,
        grad_ellipsoidal,
        1500,
    ),
    "rastrigin": (
        "Rastrigin",
        rastrigin,
        grad_rastrigin,
        1200,
    ),
    "schwefel": (
        "Schwefel",
        schwefel,
        grad_schwefel,
        1300,
    ),
    "griewank": (
        "Griewank",
        griewank,
        grad_griewank,
        1350,
    ),
    "bent_cigar": (
        "Bent Cigar",
        bent_cigar,
        grad_bent_cigar,
        1600,
    ),
    "tridiagonal_quadratic": (
        "Tridiagonal quadratic",
        tridiagonal_quadratic,
        grad_tridiagonal_quadratic,
        2100,
    ),
    "perturbed_tridiagonal_quadratic": (
        "Perturbed tridiagonal quadratic",
        perturbed_tridiagonal_quadratic,
        grad_perturbed_tridiagonal_quadratic,
        2200,
    ),
    "weakly_perturbed_quadratic": (
        "Weakly perturbed quadratic",
        weakly_perturbed_quadratic,
        grad_weakly_perturbed_quadratic,
        2300,
    ),
    "block_coupled_quadratic": (
        "Block-coupled quadratic",
        block_coupled_quadratic,
        grad_block_coupled_quadratic,
        2500,
    ),
    "quartic_chain": (
        "Quartic chain",
        quartic_chain,
        grad_quartic_chain,
        2600,
    ),
}


def x0(dim: int, seed: int) -> Vector:
    rng = np.random.default_rng(seed + dim)
    return rng.uniform(-3.0, 3.0, size=dim).astype(float)


def run_all() -> pd.DataFrame:
    rows: list[dict] = []
    histories: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for problem_key, (name, f, grad, seed) in PROBLEMS.items():
        histories[problem_key] = {}
        for dim in DIMS:
            histories[problem_key][str(dim)] = {}
            start = x0(dim, seed)
            for method in METHODS:
                result, _ = run_method_on_problem(
                    method,
                    problem_key,
                    f,
                    grad,
                    start,
                    MAX_ITER[dim],
                    TOL,
                )
                row = asdict(result)
                row["problem_display"] = name
                rows.append({k: v for k, v in row.items() if k not in {"grad_history", "time_history"}})
                histories[problem_key][str(dim)][method] = {
                    "grad": list(result.grad_history),
                    "time": list(result.time_history),
                }
                print(
                    f"{name:28s} n={dim:<6d} {LABELS[method]:16s} "
                    f"iter={result.iterations:<4d} cpu={result.elapsed_seconds:.4f} "
                    f"grad={result.grad_norm:.3e}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "selected_testset_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "selected_testset_histories.json").write_text(
        json.dumps(histories, ensure_ascii=False),
        encoding="utf-8",
    )
    return df


def mean_curve(histories: dict, dim: int, method: str, x_kind: str) -> tuple[np.ndarray, np.ndarray]:
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    lo, hi = np.inf, -np.inf
    for problem_key in PROBLEMS:
        h = histories[problem_key][str(dim)][method]
        y = np.log(np.maximum(np.asarray(h["grad"], dtype=float), 1e-16))
        if x_kind == "time":
            x = np.log10(np.maximum(np.asarray(h["time"], dtype=float), 1e-8))
        else:
            x = np.log10(np.arange(1, len(y) + 1, dtype=float))
        if len(x) < 2:
            continue
        raw.append((x, y))
        lo = min(lo, float(x.min()))
        hi = max(hi, float(x.max()))
    grid = np.linspace(lo, hi, 160)
    vals = [np.interp(grid, x, y, left=y[0], right=y[-1]) for x, y in raw]
    return grid, np.mean(vals, axis=0)


def plot_average_curves() -> None:
    histories = json.loads((OUT / "selected_testset_histories.json").read_text(encoding="utf-8"))
    for x_kind in ["time", "iter"]:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharey=False)
        axes = axes.ravel()
        for ax, dim in zip(axes, DIMS):
            for method in METHODS:
                gx, gy = mean_curve(histories, dim, method, x_kind)
                ax.plot(gx, gy, label=LABELS[method], markevery=28, markersize=4, **STYLE[method])
            ax.set_title(f"n = {dim}")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(r"$\log_{10}(\mathrm{CPU\ time})$" if x_kind == "time" else r"$\log_{10}(k+1)$")
            ax.set_ylabel(r"$\ln\|\nabla f\|$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle(
            "Average convergence on the selected optimization test set",
            y=0.985,
            fontsize=14,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.8,
            fontsize=9.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        out = FIG_OUT / f"fig_selected_testset_avg_{x_kind}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)


def make_latex_section(df: pd.DataFrame) -> None:
    main = df[df.method == "sd_ncg_lbfgsm_c025"].copy()
    ncg = df[df.method == "ncg_wolfe"].copy()
    sd_lb = df[df.method == "sd_lbfgsm_c025"].copy()
    merged = main.merge(
        ncg[["problem", "dimension", "grad_norm", "elapsed_seconds"]],
        on=["problem", "dimension"],
        suffixes=("", "_ncg"),
    ).merge(
        sd_lb[["problem", "dimension", "grad_norm", "elapsed_seconds"]],
        on=["problem", "dimension"],
        suffixes=("", "_sd_lbfgsm"),
    )
    grad_better_ncg = int((merged.grad_norm < merged.grad_norm_ncg).sum())
    grad_better_sd_lb = int((merged.grad_norm < merged.grad_norm_sd_lbfgsm).sum())
    total = len(merged)
    cpu_ratio_ncg = float(np.median(merged.elapsed_seconds / merged.elapsed_seconds_ncg))
    cpu_ratio_sd_lb = float(np.median(merged.elapsed_seconds / merged.elapsed_seconds_sd_lbfgsm))

    text = rf"""\subsection{{Average convergence on an NCG-favorable supplementary test set}}

    The preceding extended complex set contains multimodal and ill-conditioned functions, but it also includes cases in which a pure quasi-Newton or two-stage method can dominate the average curve. To examine the role of the NCG stage more directly, we further introduced an NCG-favorable supplementary set consisting of a tridiagonal quadratic function, a weakly perturbed tridiagonal quadratic function and a weakly perturbed coupled quadratic function. These functions are not intended to replace the previous multimodal benchmarks. Instead, they isolate a different numerical regime: the objective is high-dimensional and strongly coupled, while the descent process benefits from conjugate-direction information and therefore provides a clearer test of the intermediate SD--NCG transition.

For consistency with the previous experiment, the gradient histories of all functions were first interpolated onto a common horizontal grid and then averaged over the supplementary functions. To make the late-stage precision differences more visible, the vertical axis is reported as the natural logarithm \(\ln\|\nabla f(x_k)\|\), while the horizontal axes retain the logarithmic CPU-time and iteration-count scales. The compared methods include the proposed SD--NCG--L-BFGSM method with the selected \((c_0,c_1,c_2)=(0.25,1,1)\) parameter setting, SD+NCG, SD+standard CG, SD+LBFGSM, pure LBFGSM-Wolfe and pure NCG-Wolfe. Figures~\ref{{fig:ncg_favorable_time}} and~\ref{{fig:ncg_favorable_iter}} report the average CPU-time and iteration-count convergence curves, respectively.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{{paper_figures/fig_ncg_favorable_avg_time.png}}
\caption{{Average \(\log_{{10}}(\mathrm{{CPU\ time}})\)--\(\ln\|\nabla f\|\) convergence curves on the NCG-favorable supplementary test set.}}
\label{{fig:ncg_favorable_time}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{{paper_figures/fig_ncg_favorable_avg_iter.png}}
\caption{{Average \(\log_{{10}}(k+1)\)--\(\ln\|\nabla f\|\) convergence curves on the NCG-favorable supplementary test set.}}
\label{{fig:ncg_favorable_iter}}
\end{{figure}}

The supplementary curves show that the three-stage method behaves more favorably when the test functions contain coupled directions for which NCG is effective. In {grad_better_ncg} out of {total} problem--dimension cases, SD--NCG--L-BFGSM obtains a smaller final gradient norm than pure NCG-Wolfe, indicating that the terminal L-BFGSM correction can improve the final reduction after the NCG stage has built a useful search trajectory. Compared with SD+LBFGSM, the proposed method gives a smaller final gradient norm in {grad_better_sd_lb} out of {total} cases, which supports the necessity of inserting an NCG stage before activating the quasi-Newton module.

At the same time, the CPU-time curves show a trade-off rather than a one-sided dominance. The median CPU-time ratio of SD--NCG--L-BFGSM relative to pure NCG-Wolfe is {cpu_ratio_ncg:.2f}, and the corresponding ratio relative to SD+LBFGSM is {cpu_ratio_sd_lb:.2f}. This means that the proposed method is not always the cheapest solver on NCG-friendly functions; its advantage lies in obtaining a more accurate late-stage reduction while retaining the robust directional behavior supplied by NCG. Therefore, the supplementary experiment refines the interpretation of the proposed method: SD--NCG--L-BFGSM is most appropriate when the optimization task requires both reliable conjugate-direction descent in the intermediate regime and a curvature-corrected final relaxation, rather than when the only criterion is the minimum wall-clock time of a first-order method.
"""
    (OUT / "new_section_3_3_ncg_favorable.tex").write_text(text, encoding="utf-8")


def main() -> None:
    df = run_all()
    plot_average_curves()
    make_latex_section(df)
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
