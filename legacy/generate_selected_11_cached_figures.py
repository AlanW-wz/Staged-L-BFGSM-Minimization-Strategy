# -*- coding: utf-8 -*-
"""Build Section 3.3 figures from cached histories for the selected 11-function set."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_OUT = ROOT / "paper_figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DIMS = [10000, 50000, 100000, 500000]
ORIGINAL_PROBLEMS = [
    "discus",
    "ellipsoidal",
    "rastrigin",
    "schwefel",
    "griewank",
    "bent_cigar",
]
SUPPLEMENTARY_PROBLEMS = [
    "tridiagonal_quadratic",
    "perturbed_tridiagonal_quadratic",
    "weakly_perturbed_quadratic",
    "diagonal_rank_quadratic",
    "quartic_chain",
]
PROBLEMS = ORIGINAL_PROBLEMS + SUPPLEMENTARY_PROBLEMS

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
    "sd_ncg_lbfgsm_c025": dict(color="#C00000", linewidth=1.65, marker="o"),
    "sd_ncg": dict(color="#2CA02C", linewidth=1.05, linestyle=":", marker="^"),
    "sd_cg_standard": dict(color="#4C78A8", linewidth=1.05, linestyle="-.", marker="s"),
    "sd_lbfgsm_c025": dict(color="#9467BD", linewidth=1.05, linestyle="--", marker="v"),
    "lbfgsm_wolfe_c025": dict(color="#7F7F7F", linewidth=1.05, linestyle=(0, (3, 1, 1, 1)), marker="P"),
    "ncg_wolfe": dict(color="#FF7F0E", linewidth=1.05, linestyle=(0, (1, 1)), marker="X"),
}


def find_one(name: str) -> Path:
    hits = list(ROOT.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def load_histories() -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    raw_dir = next(p for p in ROOT.rglob("raw") if p.is_dir())
    selected_hist_path = find_one("selected_testset_histories.json")
    selected_hist = json.loads(selected_hist_path.read_text(encoding="utf-8"))

    histories: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for problem in ORIGINAL_PROBLEMS:
        histories[problem] = {}
        for dim in DIMS:
            d = raw_dir / problem / f"dim_{dim}"
            grad = json.loads((d / "grad_histories.json").read_text(encoding="utf-8"))
            time = json.loads((d / "time_histories.json").read_text(encoding="utf-8"))
            histories[problem][str(dim)] = {
                method: {"grad": grad[method], "time": time[method]}
                for method in METHODS
            }

    for problem in SUPPLEMENTARY_PROBLEMS:
        histories[problem] = {}
        for dim in DIMS:
            histories[problem][str(dim)] = {
                method: selected_hist[problem][str(dim)][method]
                for method in METHODS
            }

    return histories


def build_summary() -> pd.DataFrame:
    old_summary = next(p for p in ROOT.rglob("summary.csv") if p.parent.name == "results")
    selected_summary = find_one("selected_testset_summary.csv")
    old = pd.read_csv(old_summary)
    selected = pd.read_csv(selected_summary)
    df = pd.concat([old, selected], ignore_index=True)
    df = df[df["problem"].isin(PROBLEMS) & df["method"].isin(METHODS)].copy()
    out_dir = next(p.parent for p in ROOT.rglob("selected_testset_summary.csv"))
    df.to_csv(out_dir / "selected_11_testset_summary.csv", index=False, encoding="utf-8-sig")
    return df


def mean_curve(histories: dict, dim: int, method: str, x_kind: str) -> tuple[np.ndarray, np.ndarray]:
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    lo, hi = np.inf, -np.inf
    for problem in PROBLEMS:
        h = histories[problem][str(dim)][method]
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


def plot_average_curves(histories: dict) -> None:
    for x_kind in ["time", "iter"]:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharey=False)
        axes = axes.ravel()
        for ax, dim in zip(axes, DIMS):
            for method in METHODS:
                gx, gy = mean_curve(histories, dim, method, x_kind)
                ax.plot(gx, gy, label=LABELS[method], markevery=28, markersize=3.6, **STYLE[method])
            ax.set_title(f"n = {dim}")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(r"$\log_{10}(\mathrm{CPU\ time})$" if x_kind == "time" else r"$\log_{10}(k+1)$")
            ax.set_ylabel(r"$\ln\|\nabla f\|$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle("Average convergence on the selected 11-function test set", y=0.985, fontsize=14)
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
        fig.savefig(FIG_OUT / f"fig_selected_11_avg_{x_kind}.png", dpi=300)
        plt.close(fig)


def print_stats(df: pd.DataFrame) -> None:
    rows = []
    for (_, _), group in df.groupby(["problem", "dimension"]):
        best = group["grad_norm"].min()
        for _, row in group.iterrows():
            rows.append(
                {
                    "method": row["method"],
                    "log_ratio": math.log10(max(float(row["grad_norm"]), 1e-300) / max(float(best), 1e-300)),
                }
            )
    rank = pd.DataFrame(rows).groupby("method")["log_ratio"].mean().sort_values()
    print(rank.to_string())

    main = df[df.method == "sd_ncg_lbfgsm_c025"][["problem", "dimension", "grad_norm", "elapsed_seconds"]]
    for method in METHODS:
        if method == "sd_ncg_lbfgsm_c025":
            continue
        other = df[df.method == method][["problem", "dimension", "grad_norm", "elapsed_seconds"]]
        merged = main.merge(other, on=["problem", "dimension"], suffixes=("_main", "_other"))
        better = int((merged["grad_norm_main"] <= merged["grad_norm_other"]).sum())
        total = len(merged)
        cpu_ratio = float(np.median(merged["elapsed_seconds_main"] / merged["elapsed_seconds_other"]))
        print(method, better, total, f"{cpu_ratio:.2f}")


def main() -> None:
    histories = load_histories()
    df = build_summary()
    plot_average_curves(histories)
    print_stats(df)


if __name__ == "__main__":
    main()
