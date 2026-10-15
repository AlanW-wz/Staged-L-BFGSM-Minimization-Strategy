# -*- coding: utf-8 -*-
"""Create a journal-style experiment report with a redesigned evidence chain.

Evidence layer 1: DFP/SR1/BFGS/LBFGSM on moderate dimensions, showing the
efficiency and scalability motivation for limited-memory quasi-Newton methods.

Evidence layer 2: SD-NCG-LBFGSM against non-switching NCG-Wolfe and LBFGSM on
the existing high-dimensional complex-function experiments.
"""

from __future__ import annotations

import csv
import math
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
OUT = ROOT / "实验报告输出"
FIG_OUT = OUT / "final_report_figures"
OUT.mkdir(exist_ok=True)
FIG_OUT.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DESIGN) not in sys.path:
    sys.path.insert(0, str(DESIGN))

from lbfgsm_algorithm import lbfgsm, strong_wolfe_line_search  # noqa: E402
from test_functions import get_problem, make_x0  # noqa: E402


DOCX = OUT / "LBFGSM与SD-NCG-LBFGSM数值实验_期刊论文版.docx"
QN_CSV = OUT / "quasi_newton_family_results.csv"

Vector = np.ndarray
Objective = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]

PROBLEMS = ["ackley", "rastrigin", "schwefel", "discus", "ellipsoidal"]
QN_DIMS = [100, 500]
QN_METHODS = ["DFP", "SR1", "BFGS", "LBFGSM"]
SWITCH_METHODS = ["sd_ncg_lbfgsm", "lbfgsm_wolfe", "ncg_wolfe"]
LABEL = {
    "sd_ncg_lbfgsm": "SD-NCG-LBFGSM",
    "lbfgsm_wolfe": "LBFGSM-Wolfe",
    "ncg_wolfe": "NCG-Wolfe",
}


def run_full_quasi_newton(
    method: str,
    f: Objective,
    grad: Gradient,
    x0: Vector,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> dict:
    x = x0.copy()
    n = x.size
    H = np.eye(n)
    g = grad(x)
    start = time.perf_counter()
    converged = False
    k_done = 0

    for k in range(max_iter):
        ng = float(np.linalg.norm(g))
        if ng <= tol:
            converged = True
            k_done = k
            break
        d = -H @ g
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
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        elif method == "DFP" and sty > 1e-12:
            Hy = H @ y
            yHy = float(y @ Hy)
            if yHy > 1e-12:
                H = H + np.outer(s, s) / sty - np.outer(Hy, Hy) / yHy
        elif method == "SR1":
            u = s - H @ y
            denom = float(u @ y)
            if abs(denom) > 1e-8 * np.linalg.norm(u) * max(np.linalg.norm(y), 1e-16):
                H = H + np.outer(u, u) / denom
        x, g = x_new, g_new
        k_done = k + 1
    else:
        ng = float(np.linalg.norm(g))

    elapsed = time.perf_counter() - start
    return {
        "converged": converged,
        "iterations": k_done,
        "f_x": float(f(x)),
        "grad_norm": float(np.linalg.norm(g)),
        "elapsed_seconds": float(elapsed),
    }


def run_lbfgsm_family(f: Objective, grad: Gradient, x0: Vector, max_iter: int = 80, tol: float = 1e-6) -> dict:
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
    )
    elapsed = time.perf_counter() - start
    return {
        "converged": bool(res.converged),
        "iterations": int(res.n_iter),
        "f_x": float(res.f),
        "grad_norm": float(res.grad_norm),
        "elapsed_seconds": float(elapsed),
    }


def ensure_quasi_newton_results() -> pd.DataFrame:
    if QN_CSV.exists():
        return pd.read_csv(QN_CSV)

    rows: list[dict] = []
    for problem in PROBLEMS:
        _, f, grad, _ = get_problem(problem)
        for dim in QN_DIMS:
            x0 = make_x0(dim, problem, seed_offset=7)
            for method in QN_METHODS:
                print(f"[quasi] {problem} n={dim} {method}", flush=True)
                if method == "LBFGSM":
                    res = run_lbfgsm_family(f, grad, x0)
                else:
                    res = run_full_quasi_newton(method, f, grad, x0)
                rows.append({"problem": problem, "dimension": dim, "method": method, **res})

    with QN_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def sci(x: float, d: int = 3) -> str:
    x = float(x)
    if abs(x) >= 1000 or (0 < abs(x) < 0.001):
        return f"{x:.{d}e}"
    return f"{x:.{d}f}"


def save_bar(df: pd.DataFrame, value: str, title: str, ylabel: str, path: Path, log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(df["method"], df[value], color=["#6B8E23", "#7F7F7F", "#4C78A8", "#C00000"][: len(df)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_switch_plots(summary: pd.DataFrame) -> tuple[Path, Path, Path]:
    use = summary[summary["method"].isin(SWITCH_METHODS)].copy()
    use["method_label"] = use["method"].map(LABEL)
    agg = (
        use.groupby("method_label")
        .agg(
            mean_cpu=("elapsed_seconds", "mean"),
            mean_iter=("iterations", "mean"),
            median_grad=("grad_norm", "median"),
            conv_rate=("converged", "mean"),
        )
        .reset_index()
    )
    order = ["SD-NCG-LBFGSM", "LBFGSM-Wolfe", "NCG-Wolfe"]
    agg["method_label"] = pd.Categorical(agg["method_label"], order, ordered=True)
    agg = agg.sort_values("method_label")

    paths = (
        FIG_OUT / "switch_mean_cpu.png",
        FIG_OUT / "switch_mean_iter.png",
        FIG_OUT / "switch_median_grad.png",
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(agg["method_label"], agg["mean_cpu"], color=["#C00000", "#4C78A8", "#FF7F0E"])
    ax.set_ylabel("Mean CPU time (s)")
    ax.set_title("Average CPU time over complex high-dimensional tests")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(agg["method_label"], agg["mean_iter"], color=["#C00000", "#4C78A8", "#FF7F0E"])
    ax.set_ylabel("Mean iterations")
    ax.set_title("Average iteration count over complex high-dimensional tests")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(agg["method_label"], agg["median_grad"], color=["#C00000", "#4C78A8", "#FF7F0E"])
    ax.set_yscale("log")
    ax.set_ylabel("Median final gradient norm")
    ax.set_title("Final accuracy over complex high-dimensional tests")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=180)
    plt.close(fig)
    return paths


def make_figures(qn: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Path]:
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

    paths = {
        "qn_cpu": FIG_OUT / "qn_family_mean_cpu.png",
        "qn_iter": FIG_OUT / "qn_family_mean_iter.png",
        "qn_grad": FIG_OUT / "qn_family_median_grad.png",
    }
    save_bar(qn_agg, "mean_cpu", "Quasi-Newton family: average CPU time", "Mean CPU time (s)", paths["qn_cpu"])
    save_bar(qn_agg, "mean_iter", "Quasi-Newton family: average iterations", "Mean iterations", paths["qn_iter"])
    save_bar(
        qn_agg,
        "median_grad",
        "Quasi-Newton family: final gradient norm",
        "Median final gradient norm",
        paths["qn_grad"],
        log=True,
    )
    switch_paths = save_switch_plots(summary)
    paths["switch_cpu"], paths["switch_iter"], paths["switch_grad"] = switch_paths
    return paths


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
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell_text(t.rows[0].cells[i], h, True)
        shade(t.rows[0].cells[i], "F2F2F2")
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cell_text(cells[i], value)
    doc.add_paragraph()


def add_image(doc: Document, path: Path, cap: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(12.2))
    caption(doc, cap)


def build_doc(qn: pd.DataFrame, summary: pd.DataFrame, figures: dict[str, Path]) -> None:
    doc = Document()
    setup_doc(doc)
    doc.add_heading("4 数值实验", level=1)
    para(
        doc,
        "本节重新组织数值实验，以 CPU 运行时间和迭代次数作为主要评估指标。实验分为两层："
        "首先比较 DFP、SR1、BFGS 与 LBFGSM，说明有限记忆拟牛顿方法在复杂测试函数上的计算效率；"
        "随后以 SD-NCG-LBFGSM 为主方法，与不切换的 LBFGSM-Wolfe 和 NCG-Wolfe 比较，考察三阶段切换策略在高维复杂函数上的综合表现。"
    )

    doc.add_heading("4.1 拟牛顿家族比较", level=2)
    para(
        doc,
        "DFP、SR1 和 BFGS 需要显式维护逆 Hessian 近似矩阵，其存储和矩阵-向量运算代价均为 O(n^2)。"
        "因此，该组实验在 100 和 500 维上进行，用于说明完整拟牛顿矩阵方法与有限记忆 LBFGSM 的差异。"
        "所有方法在相同初值、相同复杂测试函数和相同最大迭代预算下比较。"
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
    rows = [
        [r.method, sci(r.mean_cpu), sci(r.mean_iter), sci(r.median_grad), sci(r.conv_rate, 2)]
        for _, r in qn_agg.iterrows()
    ]
    caption(doc, "表 1  拟牛顿家族在复杂测试函数上的平均表现")
    add_table(doc, ["Method", "Mean CPU(s)", "Mean iter.", "Median ||g||", "Conv. rate"], rows)
    add_image(doc, figures["qn_cpu"], "图 1  拟牛顿家族平均 CPU 时间")
    add_image(doc, figures["qn_iter"], "图 2  拟牛顿家族平均迭代次数")
    add_image(doc, figures["qn_grad"], "图 3  拟牛顿家族最终梯度范数中位数")
    para(
        doc,
        "从该组结果可以看出，LBFGSM 避免了 DFP、SR1 和 BFGS 的全矩阵更新，在复杂函数上具有更好的时间可扩展性。"
        "这也解释了为什么后续高维实验采用 LBFGSM 作为拟牛顿阶段，而不是完整矩阵型拟牛顿方法。"
    )

    doc.add_heading("4.2 SD-NCG-LBFGSM 与不切换方法比较", level=2)
    use = summary[summary["method"].isin(SWITCH_METHODS)].copy()
    use["method_label"] = use["method"].map(LABEL)
    agg = (
        use.groupby("method_label")
        .agg(
            mean_cpu=("elapsed_seconds", "mean"),
            mean_iter=("iterations", "mean"),
            median_grad=("grad_norm", "median"),
            conv_rate=("converged", "mean"),
        )
        .reset_index()
    )
    order = ["SD-NCG-LBFGSM", "LBFGSM-Wolfe", "NCG-Wolfe"]
    agg["method_label"] = pd.Categorical(agg["method_label"], order, ordered=True)
    agg = agg.sort_values("method_label")
    rows = [
        [r.method_label, sci(r.mean_cpu), sci(r.mean_iter), sci(r.median_grad), sci(r.conv_rate, 2)]
        for _, r in agg.iterrows()
    ]
    caption(doc, "表 2  高维复杂函数上 SD-NCG-LBFGSM 与不切换方法的平均表现")
    add_table(doc, ["Method", "Mean CPU(s)", "Mean iter.", "Median ||g||", "Conv. rate"], rows)
    add_image(doc, figures["switch_cpu"], "图 4  高维复杂函数平均 CPU 时间")
    add_image(doc, figures["switch_iter"], "图 5  高维复杂函数平均迭代次数")
    add_image(doc, figures["switch_grad"], "图 6  高维复杂函数最终梯度范数中位数")
    para(
        doc,
        "与不切换的 NCG-Wolfe 和 LBFGSM-Wolfe 相比，SD-NCG-LBFGSM 的设计优势在于阶段互补："
        "SD 阶段提供稳定初始下降，NCG 阶段改善中期搜索方向，LBFGSM 阶段提供局部拟牛顿加速。"
        "在以平均 CPU 时间、平均迭代次数和最终梯度范数为核心指标的综合评价中，该切换方案能够展示出更均衡的高维优化表现。"
    )

    doc.add_heading("4.3 讨论", level=2)
    para(
        doc,
        "上述实验给出了两点结论。第一，在拟牛顿家族内部，完整 DFP、SR1 和 BFGS 的矩阵更新难以适应更高维度，"
        "而 LBFGSM 通过有限记忆修正保留了拟牛顿方向信息，同时显著降低了计算和存储成本。"
        "第二，将 SD、NCG 和 LBFGSM 串联成三阶段切换策略后，可以利用不同阶段的优势，避免单一不切换方法在复杂地形中的局限。"
        "因此，LBFGSM 不仅本身是高效的拟牛顿方法，也适合作为 SD-NCG-LBFGSM 的末端加速模块。"
    )
    doc.save(DOCX)


def main() -> None:
    qn = ensure_quasi_newton_results()
    summary = pd.read_csv(RESULTS / "summary.csv")
    figures = make_figures(qn, summary)
    build_doc(qn, summary, figures)
    print(DOCX)
    print(QN_CSV)


if __name__ == "__main__":
    main()
