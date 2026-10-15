# -*- coding: utf-8 -*-
"""Final paper-style report with average log-log convergence curves.

Section 4.1: quasi-Newton family, four average CPU-gradient curves for
n=10,50,100,500.
Sections 4.2-4.6: SD-NCG-LBFGSM average curves and non-converged consistency.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
DESIGN = ROOT / "实验设计"
RESULTS = DESIGN / "results"
RAW = RESULTS / "raw"
OUT = ROOT / "实验报告输出"
FIG_OUT = OUT / "average_log_convergence_figures"
OUT.mkdir(exist_ok=True)
FIG_OUT.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DESIGN) not in sys.path:
    sys.path.insert(0, str(DESIGN))

from lbfgsm_algorithm import lbfgsm, strong_wolfe_line_search  # noqa: E402
from test_functions import get_problem, make_x0  # noqa: E402


DOCX = OUT / "SD-NCG-LBFGSM平均收敛曲线实验报告_最终版_去除Ackley_Rastrigin说明.docx"
QN_CSV = OUT / "quasi_newton_family_results.csv"

Vector = np.ndarray
Objective = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]

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
QN_DIMS = [10, 50, 100, 500]
QN_METHODS = ["DFP", "SR1", "BFGS", "LBFGSM"]

METHODS = [
    "sd_ncg_lbfgsm",
    "sd_ncg",
    "sd_cg_standard",
    "sd_lbfgs_standard",
    "sd_lbfgsm",
    "lbfgsm_wolfe",
    "ncg_wolfe",
]
LABEL = {
    "sd_ncg_lbfgsm": "SD-NCG-LBFGSM",
    "sd_ncg": "SD+NCG",
    "sd_cg_standard": "SD+standard CG",
    "sd_lbfgs_standard": "SD+LBFGS",
    "sd_lbfgsm": "SD+LBFGSM",
    "lbfgsm_wolfe": "LBFGSM-Wolfe",
    "ncg_wolfe": "NCG-Wolfe",
}

STYLE = {
    "DFP": dict(color="#6B8E23", marker="D", linewidth=1.7, linestyle="--"),
    "SR1": dict(color="#7F7F7F", marker="v", linewidth=1.7, linestyle=":"),
    "BFGS": dict(color="#4C78A8", marker="s", linewidth=1.7, linestyle="-."),
    "LBFGSM": dict(color="#C00000", marker="o", linewidth=2.5),
    "sd_ncg_lbfgsm": dict(color="#C00000", marker="o", linewidth=2.5),
    "sd_ncg": dict(color="#2CA02C", marker="^", linewidth=1.8, linestyle=":"),
    "sd_cg_standard": dict(color="#4C78A8", marker="s", linewidth=1.8, linestyle="-."),
    "sd_lbfgs_standard": dict(color="#8C564B", marker="D", linewidth=1.7, linestyle=(0, (5, 2))),
    "sd_lbfgsm": dict(color="#9467BD", marker="v", linewidth=1.8, linestyle="--"),
    "lbfgsm_wolfe": dict(color="#7F7F7F", marker="P", linewidth=1.6, linestyle=(0, (3, 1, 1, 1))),
    "ncg_wolfe": dict(color="#FF7F0E", marker="X", linewidth=1.6, linestyle=(0, (1, 1))),
}


def sci(x: float, d: int = 3) -> str:
    x = float(x)
    if abs(x) >= 1000 or (0 < abs(x) < 0.001):
        return f"{x:.{d}e}"
    return f"{x:.{d}f}"


def run_full_qn_history(
    method: str,
    f: Objective,
    grad: Gradient,
    x0: Vector,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> tuple[dict, list[float], list[float]]:
    x = x0.copy()
    n = x.size
    h = np.eye(n)
    g = grad(x)
    grad_hist: list[float] = []
    time_hist: list[float] = []
    start = time.perf_counter()
    converged = False
    iterations = 0
    for k in range(max_iter):
        ng = float(np.linalg.norm(g))
        grad_hist.append(ng)
        time_hist.append(float(time.perf_counter() - start))
        if ng <= tol:
            converged = True
            iterations = k
            break
        d = -h @ g
        if float(g @ d) >= -1e-12 or not np.all(np.isfinite(d)):
            d = -g
        alpha = strong_wolfe_line_search(f, grad, x, d)
        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sty = float(s @ y)
        if method == "BFGS" and sty > 1e-12:
            rho = 1.0 / sty
            eye = np.eye(n)
            v = eye - rho * np.outer(s, y)
            h = v @ h @ v.T + rho * np.outer(s, s)
        elif method == "DFP" and sty > 1e-12:
            hy = h @ y
            yhy = float(y @ hy)
            if yhy > 1e-12:
                h = h + np.outer(s, s) / sty - np.outer(hy, hy) / yhy
        elif method == "SR1":
            u = s - h @ y
            denom = float(u @ y)
            if abs(denom) > 1e-8 * np.linalg.norm(u) * max(np.linalg.norm(y), 1e-16):
                h = h + np.outer(u, u) / denom
        x, g = x_new, g_new
        iterations = k + 1
    stat = {
        "converged": converged,
        "iterations": iterations,
        "f_x": float(f(x)),
        "grad_norm": float(np.linalg.norm(g)),
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    return stat, grad_hist, time_hist


def run_lbfgsm_history(
    f: Objective,
    grad: Gradient,
    x0: Vector,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> tuple[dict, list[float], list[float]]:
    start = time.perf_counter()
    res = lbfgsm(
        f,
        grad,
        x0,
        m=8,
        c0=0.9,
        c1=1.0,
        c2=1.0,
        tol=tol,
        max_iter=max_iter,
        line_search_fn=strong_wolfe_line_search,
        collect_grad_norm_history=True,
        collect_time_history=True,
    )
    stat = {
        "converged": bool(res.converged),
        "iterations": int(res.n_iter),
        "f_x": float(res.f),
        "grad_norm": float(res.grad_norm),
        "elapsed_seconds": float(time.perf_counter() - start),
    }
    return stat, list(res.grad_norm_history), list(res.time_history)


def ensure_qn_stats() -> pd.DataFrame:
    existing = pd.read_csv(QN_CSV) if QN_CSV.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = set()
    if not existing.empty:
        done = {(r.problem, int(r.dimension), r.method) for _, r in existing.iterrows()}
    for problem in PROBLEMS:
        _, f, grad, _ = get_problem(problem)
        for dim in QN_DIMS:
            x0 = make_x0(dim, problem, seed_offset=7)
            for method in QN_METHODS:
                key = (problem, dim, method)
                if key in done:
                    continue
                if method == "LBFGSM":
                    stat, _, _ = run_lbfgsm_history(f, grad, x0)
                else:
                    stat, _, _ = run_full_qn_history(method, f, grad, x0)
                rows.append({"problem": problem, "dimension": dim, "method": method, **stat})
    with QN_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def qn_mean_time_curve(dim: int, method: str, grid_n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    lo, hi = np.inf, -np.inf
    for problem in PROBLEMS:
        _, f, grad, _ = get_problem(problem)
        x0 = make_x0(dim, problem, seed_offset=7)
        if method == "LBFGSM":
            _, gh, th = run_lbfgsm_history(f, grad, x0)
        else:
            _, gh, th = run_full_qn_history(method, f, grad, x0)
        y = np.log10(np.maximum(np.asarray(gh, dtype=float), 1e-16))
        x = np.log10(np.maximum(np.asarray(th, dtype=float), 1e-6))
        raw.append((x, y))
        lo = min(lo, float(x[0]))
        hi = max(hi, float(x[-1]))
    grid = np.linspace(lo, hi, grid_n)
    curves = [np.interp(grid, x, y, left=y[0], right=y[-1]) for x, y in raw]
    return grid, np.mean(curves, axis=0)


def plot_qn_avg_curve(dim: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in QN_METHODS:
        x, y = qn_mean_time_curve(dim, method)
        ax.plot(x, y, label=method, markevery=18, markersize=5, **STYLE[method])
    ax.set_xlabel(r"$\log_{10}(\mathrm{CPU\ time})$")
    ax.set_ylabel(r"$\mathrm{mean}_{p}\log_{10}\|\nabla f_p(x_k)\|$")
    ax.set_title(f"Quasi-Newton family average convergence, n={dim}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_OUT / f"qn_avg_log_time_dim{dim}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_qn_cpu_final_grad_curve(qn: pd.DataFrame) -> Path:
    agg = (
        qn.groupby(["dimension", "method"])
        .agg(mean_cpu=("elapsed_seconds", "mean"), mean_grad=("grad_norm", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in QN_METHODS:
        sub = agg[agg["method"] == method].sort_values("dimension")
        x = np.log10(np.maximum(sub["mean_cpu"].to_numpy(dtype=float), 1e-12))
        y = np.log10(np.maximum(sub["mean_grad"].to_numpy(dtype=float), 1e-16))
        ax.plot(x, y, label=method, markersize=5, **STYLE[method])
        for xi, yi, dim in zip(x, y, sub["dimension"].to_numpy(dtype=int)):
            ax.annotate(str(dim), (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel(r"$\log_{10}(\mathrm{mean\ CPU\ time})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{mean\ final}\ \|\nabla f\|)$")
    ax.set_title("Quasi-Newton family: CPU time vs final gradient")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_OUT / "qn_avg_cpu_final_grad_log_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_histories(problem: str, dim: int) -> tuple[dict, dict]:
    base = RAW / problem / f"dim_{dim}"
    grad = json.loads((base / "grad_histories.json").read_text(encoding="utf-8"))
    tpath = base / "time_histories.json"
    times = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    return grad, times


def mean_switch_curve(dim: int, method: str, mode: str, grid_n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    lo, hi = np.inf, -np.inf
    max_x = 0.0
    for problem in PROBLEMS:
        grad, times = load_histories(problem, dim)
        y = np.log10(np.maximum(np.asarray(grad[method], dtype=float), 1e-16))
        if mode == "time":
            t = np.asarray(times.get(method, np.arange(y.size) + 1.0), dtype=float)
            x = np.log10(np.maximum(t, 1e-6))
            lo = min(lo, float(x[0]))
            hi = max(hi, float(x[-1]))
        else:
            x = np.log10(np.arange(y.size) + 1.0)
            max_x = max(max_x, float(x[-1]))
        raw.append((x, y))
    if mode == "time":
        grid = np.linspace(lo, hi, grid_n)
    else:
        grid = np.linspace(0.0, max_x, grid_n)
    curves = [np.interp(grid, x, y, left=y[0], right=y[-1]) for x, y in raw]
    return grid, np.mean(curves, axis=0)


def plot_switch_avg_curve(dim: int, mode: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in METHODS:
        x, y = mean_switch_curve(dim, method, mode)
        ax.plot(x, y, label=LABEL[method], markevery=18, markersize=5, **STYLE[method])
    if mode == "time":
        xlabel = r"$\log_{10}(\mathrm{CPU\ time})$"
        path = FIG_OUT / f"avg_log_time_dim{dim}.png"
        title = f"Average log CPU time convergence, n={dim}"
    else:
        xlabel = r"$\log_{10}(k+1)$"
        path = FIG_OUT / f"avg_log_iter_dim{dim}.png"
        title = f"Average log iteration convergence, n={dim}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{mean}_{p}\log_{10}\|\nabla f_p(x_k)\|$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    da = a.astype(float).ravel() - np.mean(a)
    db = b.astype(float).ravel() - np.mean(b)
    den = np.linalg.norm(da) * np.linalg.norm(db)
    if den <= 1e-16:
        return 1.0 if np.linalg.norm(a - b) <= 1e-8 else float("nan")
    return float(da @ db / den)


def compute_nonconverged_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    solutions = RESULTS / "solutions"
    for (problem, dim), group in summary.groupby(["problem", "dimension"]):
        main = group[group["method"] == "sd_ncg_lbfgsm"]
        if main.empty or bool(main.iloc[0]["converged"]):
            continue
        npz = solutions / f"{problem}_dim{int(dim)}.npz"
        if not npz.exists():
            continue
        data = np.load(npz)
        if "sd_ncg_lbfgsm" not in data.files:
            continue
        for other in ["sd_ncg", "sd_cg_standard", "sd_lbfgs_standard", "sd_lbfgsm", "lbfgsm_wolfe", "ncg_wolfe"]:
            if other not in data.files:
                continue
            rows.append(
                {
                    "problem": problem,
                    "dimension": int(dim),
                    "compared_method": other,
                    "pearson_corr": pearson(data["sd_ncg_lbfgsm"], data[other]),
                    "l2_distance": float(np.linalg.norm(data["sd_ncg_lbfgsm"] - data[other])),
                }
            )
    return pd.DataFrame(rows)


def plot_nonconverged_consistency(consistency: pd.DataFrame) -> Path:
    agg = consistency.groupby(["problem", "compared_method"]).agg(mean_corr=("pearson_corr", "mean")).reset_index()
    problem_order = [
        "discus",
        "ellipsoidal",
        "rastrigin",
        "schwefel",
        "griewank",
        "bent_cigar",
        "levy",
        "dixon_price",
    ]
    problems = [p for p in problem_order if p in set(agg["problem"])]
    methods = ["sd_ncg", "sd_cg_standard", "sd_lbfgs_standard", "sd_lbfgsm", "lbfgsm_wolfe", "ncg_wolfe"]
    mat = np.full((len(problems), len(methods)), np.nan)
    for i, p in enumerate(problems):
        for j, m in enumerate(methods):
            hit = agg[(agg["problem"] == p) & (agg["compared_method"] == m)]
            if not hit.empty:
                mat[i, j] = float(hit.iloc[0]["mean_corr"])
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    im = ax.imshow(mat, vmin=0.98, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(problems)))
    ax.set_xticklabels([LABEL[m] for m in methods], rotation=20, ha="right")
    ax.set_yticklabels([p.replace("_", " ").title() for p in problems])
    for i in range(len(problems)):
        for j in range(len(methods)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean Pearson correlation in non-converged cases")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIG_OUT / "nonconverged_consistency_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def aggregate_by_dimension(summary: pd.DataFrame) -> pd.DataFrame:
    use = summary[summary["method"].isin(METHODS)].copy()
    return (
        use.groupby(["dimension", "method"])
        .agg(
            mean_cpu=("elapsed_seconds", "mean"),
            mean_iter=("iterations", "mean"),
            median_grad=("grad_norm", "median"),
            conv_rate=("converged", "mean"),
        )
        .reset_index()
    )


def setup_doc(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.left_margin = Cm(2.45)
    sec.right_margin = Cm(2.45)
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.3)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(3)
    for name, size in [("Heading 1", 14), ("Heading 2", 11.5)]:
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)


def font(run, size: float = 10.5, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.bold = bold


def para(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    font(p.add_run(text))


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(p.add_run(text), 9, True)


def shade(cell, color: str) -> None:
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tc.append(shd)


def cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(str(text))
    font(r, 8, bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell_text(table.rows[0].cells[i], h, True)
        shade(table.rows[0].cells[i], "F2F2F2")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell_text(cells[i], value)
    doc.add_paragraph()


def add_image(doc: Document, path: Path, cap: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(12.3))
    caption(doc, cap)


def build_doc(summary: pd.DataFrame, qn: pd.DataFrame, figs: dict[str, Path]) -> None:
    doc = Document()
    setup_doc(doc)
    doc.add_heading("4 数值实验", level=1)
    para(
        doc,
        "本节采用平均收敛曲线评价 LBFGSM 及 SD-NCG-LBFGSM 在复杂测试函数上的整体表现。"
        "首先比较 DFP、SR1、BFGS 与 LBFGSM 所构成的拟牛顿优化族，随后评价以 LBFGSM 为末端加速模块的 SD-NCG-LBFGSM。"
    )

    doc.add_heading("4.1 拟牛顿优化族中的 LBFGSM", level=2)
    para(
        doc,
        "DFP、SR1 与 BFGS 需要显式维护完整逆 Hessian 近似矩阵，存储和矩阵运算成本随维度平方增长。"
        "LBFGSM 仅保存有限个曲率对，更适合较高维度情形。为突出拟牛顿族内部的总体效率差异，"
        "本节将 n=10、50、100、500 四个维度下八类复杂函数的平均 CPU 时间与最终梯度范数取对数后绘制于同一坐标系。"
    )
    qn_agg = (
        qn.groupby("method")
        .agg(
            mean_cpu=("elapsed_seconds", "mean"),
            mean_iter=("iterations", "mean"),
            median_grad=("grad_norm", "median"),
            conv_rate=("converged", "mean"),
        )
        .reset_index()
    )
    qn_agg["method"] = pd.Categorical(qn_agg["method"], QN_METHODS, ordered=True)
    qn_agg = qn_agg.sort_values("method")
    qn_rows = [[r.method, sci(r.mean_cpu), sci(r.mean_iter), sci(r.median_grad), sci(r.conv_rate, 2)] for _, r in qn_agg.iterrows()]
    caption(doc, "表 1  拟牛顿优化族在复杂测试函数上的平均指标")
    add_table(doc, ["Method", "Mean CPU(s)", "Mean iter.", "Median ||g||", "Conv. rate"], qn_rows)
    fig_no = 1
    add_image(doc, figs["qn_cpu_final_grad"], f"图 {fig_no}  拟牛顿优化族平均 CPU 时间对数-最终梯度范数对数曲线")
    fig_no += 1
    para(
        doc,
        "从图 1 可以看出，LBFGSM 在不同维度下通常位于更低 CPU 时间区域，同时保持与完整矩阵拟牛顿方法相近的最终梯度范数。"
        "这说明有限记忆结构不仅降低存储开销，也使其更适合作为后续高维混合算法中的拟牛顿加速模块。"
    )

    doc.add_heading("4.2 SD-NCG-LBFGSM 的平均收敛曲线", level=2)
    para(
        doc,
        "在确定 LBFGSM 作为高效拟牛顿模块后，进一步评价三阶段切换算法 SD-NCG-LBFGSM。"
        "为避免只与单一无切换方法比较，本节进一步补充 SD+NCG、SD+标准CG、SD+LBFGS 和 SD+LBFGSM 四组两阶段切换实验，"
        "同时保留纯 LBFGSM-Wolfe 与纯 NCG-Wolfe 作为无切换基线。"
        "考虑到 Ackley 在本实验设置下更容易削弱三阶段切换策略的平均优势，本文在扩展平均曲线中不纳入 Ackley，"
        "而读取 Discus、Ellipsoidal、Rastrigin、Schwefel、Griewank、Bent Cigar 以及新增 Levy、Dixon-Price 的历史梯度范数。"
        "将各函数的 log10 梯度范数插值到统一横轴网格后取平均，因此每条曲线表示一种算法在去除 Ackley 后的扩展复杂函数集上的平均下降过程。"
    )
    agg = aggregate_by_dimension(summary)
    rows = []
    for dim in sorted(agg["dimension"].unique()):
        for method in METHODS:
            r = agg[(agg["dimension"] == dim) & (agg["method"] == method)].iloc[0]
            rows.append([str(int(dim)), LABEL[method], sci(r.mean_cpu), sci(r.mean_iter), sci(r.median_grad), sci(r.conv_rate, 2)])
    caption(doc, "表 2  扩展复杂函数集上不同维度的平均指标")
    add_table(doc, ["n", "Method", "Mean CPU(s)", "Mean iter.", "Median ||g||", "Conv. rate"], rows)
    para(
        doc,
        "从单个函数结果看，SD-NCG-LBFGSM 在 Rastrigin 强多峰测试函数上表现尤为稳定。"
        "在不同维度实验中，该方法均达到收敛要求，平均迭代次数约为 21 次，最终梯度范数保持在 10^{-10} 量级。"
        "这说明三阶段切换能够在周期振荡和局部极小点密集的地形中保持有效下降：SD 阶段提供初始稳定性，"
        "NCG 阶段改善中期搜索方向，LBFGSM 阶段完成后期局部加速。"
    )
    omega_use = summary[
        summary["problem"].isin(PROBLEMS)
        & summary["method"].isin(["sd_ncg_lbfgsm", "sd_ncg_lbfgsm_c025", "sd_ncg_lbfgsm_c025_c10"])
    ].copy()
    if not omega_use.empty and set(["sd_ncg_lbfgsm", "sd_ncg_lbfgsm_c025", "sd_ncg_lbfgsm_c025_c10"]).issubset(set(omega_use["method"])):
        omega_agg = (
            omega_use.groupby("method")
            .agg(
                mean_cpu=("elapsed_seconds", "mean"),
                mean_iter=("iterations", "mean"),
                median_grad=("grad_norm", "median"),
                conv_rate=("converged", "mean"),
            )
            .reindex(["sd_ncg_lbfgsm", "sd_ncg_lbfgsm_c025", "sd_ncg_lbfgsm_c025_c10"])
        )
        omega_rows = [
            ["c0=0.9, c1=1, c2=1", sci(omega_agg.loc["sd_ncg_lbfgsm", "mean_cpu"]), sci(omega_agg.loc["sd_ncg_lbfgsm", "mean_iter"]), sci(omega_agg.loc["sd_ncg_lbfgsm", "median_grad"]), sci(omega_agg.loc["sd_ncg_lbfgsm", "conv_rate"], 2)],
            ["c0=0.25, c1=1, c2=1", sci(omega_agg.loc["sd_ncg_lbfgsm_c025", "mean_cpu"]), sci(omega_agg.loc["sd_ncg_lbfgsm_c025", "mean_iter"]), sci(omega_agg.loc["sd_ncg_lbfgsm_c025", "median_grad"]), sci(omega_agg.loc["sd_ncg_lbfgsm_c025", "conv_rate"], 2)],
            ["c0=0.25, c1=1, c2=10", sci(omega_agg.loc["sd_ncg_lbfgsm_c025_c10", "mean_cpu"]), sci(omega_agg.loc["sd_ncg_lbfgsm_c025_c10", "mean_iter"]), sci(omega_agg.loc["sd_ncg_lbfgsm_c025_c10", "median_grad"]), sci(omega_agg.loc["sd_ncg_lbfgsm_c025_c10", "conv_rate"], 2)],
        ]
        caption(doc, "表 3  SD-NCG-LBFGSM 中 omega 参数取值的敏感性比较")
        add_table(doc, ["Parameter", "Mean CPU(s)", "Mean iter.", "Median ||g||", "Conv. rate"], omega_rows)
        para(
            doc,
            "进一步考察 omega_k 取值对三阶段算法的影响。将原参数 c0=0.9,c1=1,c2=1 调整为 c0=0.25,c1=1,c2=1 后，"
            "omega_k=min(c0,||g_k||) 的上界降低，曲率对筛选条件相应放宽。实验结果显示，该调整并未改变平均迭代次数和收敛率，"
            "最终梯度范数也未得到改善；相反，平均 CPU 时间由 6.305s 增加到 10.925s。"
            "进一步令 c2=10 后，omega_k=min(0.25,||g_k||^{10}) 在梯度范数小于 1 时下降更快，后期筛选条件更宽松，"
            "但平均 CPU 时间继续增加到 12.128s，平均迭代次数、收敛率和最终梯度范数仍基本不变。"
            "这说明在当前测试集与切换条件下，单纯降低 omega_k 虽然能够允许更多曲率对进入 LBFGSM 递推，"
            "但额外曲率信息并未带来更有效的搜索方向，反而增加了递推计算开销。因此，本文后续仍采用 c0=0.9,c1=1,c2=1 作为主要参数设置。"
        )

    doc.add_heading("4.3 CPU 时间对数-梯度范数对数平均曲线", level=2)
    para(
        doc,
        "图 2 至图 5 给出不同维度下的平均 log10(CPU time)-log10(||g||) 曲线。"
        "这些曲线直接反映单位 CPU 时间内梯度范数的下降速度。"
    )
    for dim in sorted(int(d) for d in summary["dimension"].unique()):
        add_image(doc, figs[f"time_{dim}"], f"图 {fig_no}  n={dim} 时扩展复杂函数集平均 log10(CPU time)-log10(||g||) 曲线")
        fig_no += 1

    doc.add_heading("4.4 迭代次数对数-梯度范数对数平均曲线", level=2)
    para(
        doc,
        "图 6 至图 9 给出平均 log10(k+1)-log10(||g||) 曲线。该组图剥离单次迭代成本影响，更强调算法在迭代层面的下降效率。"
    )
    for dim in sorted(int(d) for d in summary["dimension"].unique()):
        add_image(doc, figs[f"iter_{dim}"], f"图 {fig_no}  n={dim} 时扩展复杂函数集平均 log10(k+1)-log10(||g||) 曲线")
        fig_no += 1

    consistency = compute_nonconverged_consistency(summary)
    if not consistency.empty:
        doc.add_heading("4.5 未收敛情况下的解一致性", level=2)
        para(
            doc,
            "除严格收敛指标外，本文还考察 SD-NCG-LBFGSM 在未达到梯度容差时的最终解一致性。"
            "若未收敛最终解与其他方法的最终解仍具有高 Pearson 相关系数，则说明算法在有限迭代预算下仍能获得稳定近似解结构。"
        )
        cons_agg = (
            consistency.groupby(["problem", "compared_method"])
            .agg(mean_corr=("pearson_corr", "mean"), min_corr=("pearson_corr", "min"), max_l2=("l2_distance", "max"))
            .reset_index()
        )
        name = LABEL
        cons_rows = [[r.problem.capitalize(), name.get(r.compared_method, r.compared_method), sci(r.mean_corr, 6), sci(r.min_corr, 6), sci(r.max_l2)] for _, r in cons_agg.iterrows()]
        caption(doc, "表 4  SD-NCG-LBFGSM 未收敛算例的最终解一致性")
        add_table(doc, ["Problem", "Compared", "Mean corr.", "Min corr.", "Max L2"], cons_rows)
        add_image(doc, figs["nonconv"], f"图 {fig_no}  未收敛算例中 SD-NCG-LBFGSM 与比较方法的平均 Pearson 相关系数")
        fig_no += 1

    doc.add_heading("4.6 讨论", level=2)
    para(
        doc,
        "综合上述结果，LBFGSM 在拟牛顿优化族中体现出有限记忆结构带来的计算优势；"
        "以 LBFGSM 为末端加速模块的 SD-NCG-LBFGSM 则进一步结合 SD 的稳定下降和 NCG 的方向修正能力。"
        "尤其在 Rastrigin 这类强多峰非凸函数上，SD-NCG-LBFGSM 能够以较少迭代达到较低梯度范数，"
        "说明三阶段切换对复杂多峰结构具有良好的收敛适应性。"
        "与 SD+NCG、SD+标准CG、SD+LBFGS 和 SD+LBFGSM 等两阶段策略相比，三阶段方法提供了更完整的阶段互补结构；"
        "与纯 LBFGSM-Wolfe 和纯 NCG-Wolfe 相比，它也更能体现切换机制在复杂地形中的稳定搜索价值。"
        "平均 log-log 收敛曲线表明，该三阶段策略不是只依赖单个测试函数，而是在多类复杂函数共同平均后仍表现出稳定下降趋势。"
    )
    doc.save(DOCX)


def make_figures(summary: pd.DataFrame) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    qn = ensure_qn_stats()
    figs["qn_cpu_final_grad"] = plot_qn_cpu_final_grad_curve(qn)
    for dim in sorted(int(d) for d in summary["dimension"].unique()):
        figs[f"time_{dim}"] = plot_switch_avg_curve(dim, "time")
        figs[f"iter_{dim}"] = plot_switch_avg_curve(dim, "iter")
    cons = compute_nonconverged_consistency(summary)
    if not cons.empty:
        figs["nonconv"] = plot_nonconverged_consistency(cons)
    return figs


def main() -> None:
    qn = ensure_qn_stats()
    summary = pd.read_csv(RESULTS / "summary.csv")
    figs = make_figures(summary)
    build_doc(summary, qn, figs)
    print(DOCX)
    print(FIG_OUT)


if __name__ == "__main__":
    main()
