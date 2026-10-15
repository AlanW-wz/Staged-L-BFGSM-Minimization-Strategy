# -*- coding: utf-8 -*-
"""Replot manuscript Figures 1--3 using the original numerical data.

The calculations and averaging rules match the source scripts used for the
current manuscript figures. Only the visual presentation and export settings
are changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "AFM_figures_1_3_redesign"
RAW = ROOT / "实验设计" / "results" / "raw"
QN_CSV = ROOT / "实验报告输出" / "quasi_newton_family_results.csv"
SELECTED_HISTORIES = (
    ROOT / "实验报告输出" / "selected_testset_33" / "selected_testset_histories.json"
)

OUT.mkdir(parents=True, exist_ok=True)

DIMS = [10000, 50000, 100000, 500000]
QN_DIMS = [10, 50, 100, 500]
QN_METHODS = ["DFP", "SR1", "BFGS", "LBFGSM"]

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

# Restrained, colorblind-aware palette with redundant line-style encoding.
QN_STYLE = {
    "DFP": dict(color="#71864D", marker="D", linestyle=(0, (5, 2))),
    "SR1": dict(color="#7A7A7A", marker="v", linestyle=(0, (1.2, 1.5))),
    "BFGS": dict(color="#557B99", marker="s", linestyle=(0, (4, 1.5, 1, 1.5))),
    "LBFGSM": dict(color="#C51B32", marker="o", linestyle="-"),
}

CURVE_STYLE = {
    "sd_ncg_lbfgsm_c025": dict(color="#C51B32", linestyle="-", marker="o", zorder=6),
    "sd_ncg": dict(color="#58897D", linestyle=(0, (1.2, 1.5)), marker="^", zorder=4),
    "sd_cg_standard": dict(color="#5C7E9A", linestyle=(0, (5, 2, 1, 2)), marker="s", zorder=3),
    "sd_lbfgsm_c025": dict(color="#806E91", linestyle=(0, (5, 2)), marker="v", zorder=5),
    "lbfgsm_wolfe_c025": dict(color="#777777", linestyle=(0, (3, 1, 1, 1)), marker="P", zorder=2),
    "ncg_wolfe": dict(color="#A87A4F", linestyle=(0, (1, 1)), marker="X", zorder=1),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "mathtext.tt": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#202020",
            "axes.labelcolor": "#202020",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.3,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axes(ax: plt.Axes, horizontal_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if horizontal_grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    else:
        ax.grid(False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    common = dict(bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", **common)
    fig.savefig(OUT / f"{stem}.svg", **common)
    fig.savefig(OUT / f"{stem}.png", dpi=600, **common)
    fig.savefig(
        OUT / f"{stem}.tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )


def load_histories() -> tuple[dict, list[Path]]:
    selected = json.loads(SELECTED_HISTORIES.read_text(encoding="utf-8"))
    used_files = [SELECTED_HISTORIES]
    histories: dict = {}

    for problem in ORIGINAL_PROBLEMS:
        histories[problem] = {}
        for dim in DIMS:
            base = RAW / problem / f"dim_{dim}"
            grad_path = base / "grad_histories.json"
            time_path = base / "time_histories.json"
            grad = json.loads(grad_path.read_text(encoding="utf-8"))
            timing = json.loads(time_path.read_text(encoding="utf-8"))
            used_files.extend([grad_path, time_path])
            histories[problem][str(dim)] = {
                method: {"grad": grad[method], "time": timing[method]}
                for method in METHODS
            }

    for problem in SUPPLEMENTARY_PROBLEMS:
        histories[problem] = {}
        for dim in DIMS:
            histories[problem][str(dim)] = {
                method: selected[problem][str(dim)][method] for method in METHODS
            }

    return histories, used_files


def mean_curve(histories: dict, dim: int, method: str, x_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Match the interpolation-and-mean rule in the original plotting script."""
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    lo, hi = np.inf, -np.inf
    for problem in PROBLEMS:
        history = histories[problem][str(dim)][method]
        y = np.log(np.maximum(np.asarray(history["grad"], dtype=float), 1e-16))
        if x_kind == "time":
            x = np.log10(np.maximum(np.asarray(history["time"], dtype=float), 1e-8))
        else:
            x = np.log10(np.arange(1, len(y) + 1, dtype=float))
        if len(x) < 2:
            continue
        raw.append((x, y))
        lo = min(lo, float(x.min()))
        hi = max(hi, float(x.max()))

    grid = np.linspace(lo, hi, 160)
    interpolated = [
        np.interp(grid, x, y, left=y[0], right=y[-1]) for x, y in raw
    ]
    return grid, np.mean(interpolated, axis=0)


def plot_figure_1() -> pd.DataFrame:
    qn = pd.read_csv(QN_CSV)
    agg = (
        qn.groupby(["dimension", "method"], as_index=False)
        .agg(mean_cpu=("elapsed_seconds", "mean"), mean_grad=("grad_norm", "mean"))
    )

    fig, ax = plt.subplots(figsize=(7.08, 3.55))
    dimension_sizes = {10: 18, 50: 28, 100: 40, 500: 54}
    label_offsets = {
        "DFP": (-8, 10, "right"),
        "SR1": (-8, 10, "right"),
        "BFGS": (8, 0, "left"),
        "LBFGSM": (-10, 11, "right"),
    }

    for method in QN_METHODS:
        sub = agg[agg["method"] == method].sort_values("dimension")
        x = np.log10(np.maximum(sub["mean_cpu"].to_numpy(dtype=float), 1e-12))
        y = np.log10(np.maximum(sub["mean_grad"].to_numpy(dtype=float), 1e-16))
        style = QN_STYLE[method]
        line, = ax.plot(
            x,
            y,
            label=method,
            linewidth=1.05 if method != "LBFGSM" else 1.70,
            color=style["color"],
            linestyle=style["linestyle"],
            zorder=3 if method != "LBFGSM" else 5,
        )
        if method == "LBFGSM":
            line.set_path_effects(
                [pe.Stroke(linewidth=4.6, foreground="#F6D9DE", alpha=0.9), pe.Normal()]
            )
        ax.scatter(
            x,
            y,
            s=[dimension_sizes[int(dim)] for dim in sub["dimension"]],
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.6,
            zorder=6 if method == "LBFGSM" else 4,
        )
        dx, dy, align = label_offsets[method]
        ax.annotate(
            method,
            xy=(x[-1], y[-1]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=7.2,
            fontweight="bold" if method == "LBFGSM" else "normal",
            color=style["color"],
        )

    clean_axes(ax)
    ax.set_xlabel(r"$\log_{10}$(mean CPU time / s)")
    ax.set_ylabel(r"$\log_{10}$(mean final $\|\nabla f\|$)")
    dimension_handles = [
        Line2D(
            [0],
            [0],
            linestyle="",
            marker="o",
            markersize=np.sqrt(dimension_sizes[dim]),
            markerfacecolor="#777777",
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=f"{dim}",
        )
        for dim in QN_DIMS
    ]
    ax.legend(
        handles=dimension_handles,
        title=r"dimension $n$",
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
        title_fontsize=7.2,
    )
    ax.margins(x=0.065, y=0.10)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.17, top=0.98)
    save_figure(fig, "Figure_1_AFM")
    plt.close(fig)
    return agg


def transition_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    """Return the start, steepest-descent transition, and end indices."""
    slope = np.gradient(y, x)
    transition = int(np.nanargmin(slope))
    return sorted(set([0, transition, len(x) - 1]))


def spread_label_positions(
    values: dict[str, float], lower: float, upper: float, gap: float = 1.28
) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    placed: dict[str, float] = {}
    cursor = lower
    for method in ordered:
        position = max(values[method], cursor)
        placed[method] = position
        cursor = position + gap

    overflow = max(placed.values()) - upper
    if overflow > 0:
        placed = {method: value - overflow for method, value in placed.items()}
    underflow = lower - min(placed.values())
    if underflow > 0:
        placed = {method: value + underflow for method, value in placed.items()}
    return placed


def add_end_labels(
    ax: plt.Axes,
    entries: dict[str, tuple[float, float]],
    y_lower: float,
    y_upper: float,
) -> None:
    data_x_min = min(value[0] for value in entries.values())
    data_x_max = max(value[0] for value in entries.values())
    current_min = min(ax.get_xlim()[0], data_x_min)
    span = max(data_x_max - current_min, 1e-6)
    original_ticks = ax.get_xticks()
    label_x = data_x_max + 0.055 * span
    ax.set_xlim(current_min, data_x_max + 0.43 * span)
    ax.set_xticks(original_ticks[(original_ticks >= current_min) & (original_ticks <= data_x_max)])

    label_y = spread_label_positions(
        {method: value[1] for method, value in entries.items()},
        y_lower + 0.38,
        y_upper - 0.38,
    )
    for method in METHODS:
        x_end, y_end = entries[method]
        style = CURVE_STYLE[method]
        main = method == "sd_ncg_lbfgsm_c025"
        ax.plot(
            [x_end, label_x - 0.012 * span],
            [y_end, label_y[method]],
            color=style["color"],
            linewidth=0.55 if not main else 0.8,
            alpha=0.82,
            clip_on=False,
        )
        ax.text(
            label_x,
            label_y[method],
            LABELS[method],
            fontsize=6.25,
            fontweight="bold" if main else "normal",
            color=style["color"] if main else "#303030",
            va="center",
            ha="left",
            clip_on=False,
        )


def plot_average_figure(histories: dict, x_kind: str, number: int) -> pd.DataFrame:
    curves: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    data_rows: list[pd.DataFrame] = []
    all_y: list[np.ndarray] = []

    for dim in DIMS:
        for method in METHODS:
            x, y = mean_curve(histories, dim, method, x_kind)
            curves[(dim, method)] = (x, y)
            all_y.append(y)
            data_rows.append(
                pd.DataFrame(
                    {
                        "dimension": dim,
                        "method": method,
                        "grid_index": np.arange(len(x)),
                        "x": x,
                        "mean_ln_gradient_norm": y,
                    }
                )
            )

    y_min = min(float(y.min()) for y in all_y)
    y_max = max(float(y.max()) for y in all_y)
    y_pad = 0.035 * (y_max - y_min)

    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.48), sharey=True)
    axes = axes.ravel()
    panel_labels = ["a", "b", "c", "d"]

    for panel_index, (ax, dim, panel) in enumerate(zip(axes, DIMS, panel_labels)):
        end_entries: dict[str, tuple[float, float]] = {}
        for method in METHODS:
            x, y = curves[(dim, method)]
            style = CURVE_STYLE[method]
            main = method == "sd_ncg_lbfgsm_c025"
            line, = ax.plot(
                x,
                y,
                linewidth=1.70 if main else 0.92,
                color=style["color"],
                linestyle=style["linestyle"],
                zorder=style["zorder"],
            )
            if main:
                line.set_path_effects(
                    [pe.Stroke(linewidth=4.5, foreground="#F6D9DE", alpha=0.88), pe.Normal()]
                )
            marks = transition_indices(x, y)
            ax.scatter(
                x[marks],
                y[marks],
                s=17 if main else 10,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=style["zorder"] + 1,
            )
            end_entries[method] = (float(x[-1]), float(y[-1]))
        clean_axes(ax)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if panel_index in (1, 3):
            add_end_labels(ax, end_entries, y_min - y_pad, y_max + y_pad)
        ax.text(
            -0.13,
            1.025,
            f"({panel})",
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.text(
            0.02,
            1.02,
            rf"$n={dim // 1000}\,\mathrm{{k}}$",
            transform=ax.transAxes,
            fontsize=7.2,
            va="bottom",
            ha="left",
            color="#303030",
            clip_on=False,
        )

    fig.supxlabel(
        r"$\log_{10}$(CPU time / s)" if x_kind == "time" else r"$\log_{10}(k+1)$",
        x=0.53,
        y=0.025,
        fontsize=8.5,
    )
    fig.supylabel(r"mean $\ln\|\nabla f\|$", x=0.012, fontsize=8.5)

    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.115, top=0.96, wspace=0.20, hspace=0.31)
    save_figure(fig, f"Figure_{number}_AFM")
    plt.close(fig)

    return pd.concat(data_rows, ignore_index=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_audit(used_files: list[Path], qn_agg: pd.DataFrame) -> None:
    audit = {
        "statement": (
            "Figures were regenerated from the original CSV/JSON histories. "
            "No observations were removed, replaced, smoothed, or rescaled beyond "
            "the logarithms and interpolation already used in the original scripts."
        ),
        "figure_1_aggregation": "Arithmetic mean within each dimension-method group.",
        "figures_2_3_aggregation": (
            "For every dimension and method, ln(gradient norm) was linearly interpolated "
            "onto a 160-point common log10 time/iteration grid and then averaged over "
            "the same eleven functions as the original figure script."
        ),
        "problems": PROBLEMS,
        "dimensions_figures_2_3": DIMS,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(set([QN_CSV] + used_files))
        },
        "figure_1_aggregated_values": qn_agg.to_dict(orient="records"),
    }
    (OUT / "data_provenance.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure_style()
    histories, used_files = load_histories()
    qn_agg = plot_figure_1()
    figure_2_data = plot_average_figure(histories, "time", 2)
    figure_3_data = plot_average_figure(histories, "iter", 3)
    qn_agg.to_csv(OUT / "Figure_1_plot_data.csv", index=False)
    figure_2_data.to_csv(OUT / "Figure_2_plot_data.csv", index=False)
    figure_3_data.to_csv(OUT / "Figure_3_plot_data.csv", index=False)
    write_audit(used_files, qn_agg)
    print(f"AFM-style figures written to: {OUT}")


if __name__ == "__main__":
    main()
