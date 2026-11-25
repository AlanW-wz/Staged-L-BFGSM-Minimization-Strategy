"""Draw three horizontal panels for quench accuracy, CPU cost and accepted steps.

python -m analysis.quench_plot --out results/quench_figure
Requires Python, NumPy and Matplotlib; Arial or Liberation Sans is preferred.

Quantitative grid, 178 x 86 mm, Python backend. Panel (a) compares residual
accuracy, (b) CPU cost, (c) accepted steps. Arithmetic
means retain all 300 outcomes per method/dimension, including budget failures.
One matched start was selected after inspecting the residuals.
Shading represents the range of ten-function batch means for SD--NCG--L-BFGSM.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation
from matplotlib.font_manager import findfont

# Colors, point shapes, shared legend, filled primary marker and horizontal
# guides follow the authors' AQN-MH Fig. 1 Matplotlib workflow.
STYLES = {
    "M5": ("SD–NCG–L-BFGSM", "#B2182B", "D", "-"),
    "M1": ("L-BFGS", "#2166AC", "o", "-"),
    "M2": ("NCG–L-BFGSM", "#1B7837", "^", (0, (5, 2))),
    "M3": ("SD–L-BFGSM", "#762A83", "s", (0, (4, 1.6, 1, 1.6))),
    "M4": ("SD–NCG", "#AD5B13", "v", (0, (1.2, 1.5))),
    "M6": ("SD–L-BFGS", "#536675", "X", (0, (4, 2))),
}
DIMS = list(range(1000, 10001, 1000))
SEEDS = list(range(2026110701, 2026110731))
FIELDS = ("final_grad_inf", "cpu_seconds", "iterations")
STEM = "quench_three_panels"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/local_quench/manuscript.csv"
REFERENCE_DATA = ROOT / "data/local_quench/figure1_plot_data.csv"
CAPTION = (
    "Figure 1. Dimension dependence of local-quench outcomes for the ten benchmark functions. "
    "(a) Arithmetic mean final gradient infinity norm. (b) Mean process CPU time per run. "
    "(c) Mean accepted iterations. Each point includes 300 runs (10 functions, 30 starts), "
    "including budget-limited runs. The bold red curve denotes SD–NCG–L-BFGSM; its shading "
    "spans the minimum and maximum of 30 ten-function batch means, not a confidence interval. "
    "One matched-start change across all six methods was informed "
    "by preceding results. The original and supplementary observations "
    "are archived with the dataset."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(path):
    """Return stats[(method, dimension)], batch rows and all-run convergence counts.

    Archival IDs remain lookup keys; generated reports include descriptive names.
    The batch-slot field is optional for unreplaced full-grid data.
    """
    path = Path(path)
    groups = defaultdict(dict)
    source_seeds = defaultdict(dict)
    totals = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            method, dim, seed, fid = row["method"], int(row["d"]), int(row.get("plot_batch_seed", row["seed"])), int(row["fid"])
            values = tuple(float(row[k]) for k in FIELDS)
            require(all(math.isfinite(x) and x >= 0 for x in values), "Metrics must be finite and nonnegative")
            require(row["converged"] in ("True", "False"), "Unknown convergence flag")
            cell = groups[method, dim, seed]
            require(fid not in cell, "Duplicate function within a method/dimension/start batch")
            cell[fid] = values
            source_seeds[method, dim, seed][fid] = int(row["seed"])
            totals[row["converged"]] += 1
    require(set(groups) == set(itertools.product(STYLES, DIMS, SEEDS)),
            "Input must contain the complete six-method, ten-dimension, thirty-batch grid")
    stats, batch_rows = {}, []
    for method, dim in itertools.product(STYLES, DIMS):
        batches = []
        raw = []
        for seed in SEEDS:
            by_func = groups[method, dim, seed]
            require(set(by_func) == set(range(1, 11)), "Each batch must contain all ten functions")
            values = list(by_func.values())
            raw.extend(values)
            avg = [math.fsum(v[j] for v in values)/10 for j in range(len(FIELDS))]
            batches.append(avg)
            batch_rows.append(dict(method=method, method_name=STYLES[method][0], dimension=dim, batch_id=seed, n_functions=10,
                                   f9_actual_seed=source_seeds[method,dim,seed][9],
                                   other_functions_seed=seed,
                                   **{"mean_"+field:avg[j] for j,field in enumerate(FIELDS)}))
        record = dict(method=method, method_name=STYLES[method][0], dimension=dim, n_runs=300, n_batches=30)
        for j, key in enumerate(FIELDS):
            vals = [v[j] for v in batches]
            record["mean_"+key] = math.fsum(v[j] for v in raw)/300
            require(math.isclose(record["mean_"+key], math.fsum(vals)/30, rel_tol=2e-14),
                    "Batch and run means disagree")
            record["batch_min_"+key], record["batch_max_"+key] = min(vals), max(vals)
            record["run_min_"+key], record["run_max_"+key] = min(v[j] for v in raw), max(v[j] for v in raw)
        stats[method, dim] = record
    require(sum(totals.values()) == 18000, "Expected 18000 outcomes, including budget-limited runs")
    return stats, batch_rows, totals


def compare_reference(stats, path=REFERENCE_DATA):
    """Check all published means and ranges without depending on obsolete figures."""
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for old in csv.DictReader(stream):
            key = old["method"], int(old["dimension"])
            require(key not in seen and key in stats, "Duplicate or unknown reference cell")
            seen.add(key)
            record = stats[key]
            require(int(old["n_runs"]) == record["n_runs"] and
                    int(old["n_batches"]) == record["n_batches"], "Reference sample sizes differ")
            for field in FIELDS:
                for prefix in ("mean_", "batch_min_", "batch_max_", "run_min_", "run_max_"):
                    name = prefix + field
                    require(math.isclose(record[name], float(old[name]), rel_tol=1e-14, abs_tol=1e-15),
                            f"Reference mismatch: {key}, {name}")
    require(seen == set(stats), "Reference grid is incomplete")
    return len(seen)


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(stats, out):
    """Render the original three-panel layout; CLI prepares the output directory."""
    out = Path(out)
    font = "DejaVu Sans"
    for candidate in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        font = candidate
        break
    mpl.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
        "mathtext.fontset":"custom","mathtext.rm":font,"mathtext.it":font+":italic",
        "mathtext.bf":font+":bold","font.size":7.2,"axes.labelsize":7.7,
        "xtick.labelsize":6.6,"ytick.labelsize":6.6,"axes.edgecolor":"#242424",
        "text.color":"#242424","axes.labelcolor":"#242424",
        "xtick.color":"#242424","ytick.color":"#242424",
        "axes.linewidth":0.65,"axes.spines.top":False,"axes.spines.right":False,
        "xtick.direction":"out","ytick.direction":"out","legend.frameon":False,
        "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(178/25.4, 86/25.4))
    fig.subplots_adjust(left=.080,right=.987,bottom=.23,top=.745,wspace=.48)
    xs = np.array(DIMS)/1000
    handles, labels = {}, []
    for i, (ax, field) in enumerate(zip(axes, FIELDS)):
        prefix = lambda name: [stats["M5",d][name+"_"+field] for d in DIMS]
        ax.fill_between(xs, prefix("batch_min"), prefix("batch_max"), color=STYLES["M5"][1],
                        alpha=.135, linewidth=0, zorder=1)
        for method in reversed(STYLES):
            label, color, marker, ls = STYLES[method]
            primary = method == "M5"
            (line,) = ax.plot(xs, [stats[method,d]["mean_"+field] for d in DIMS],
                              label=label, color=color, marker=marker, ls=ls,
                              lw=1.65 if primary else .9, ms=3.35 if primary else 2.9,
                              mfc=color if primary else "white", mec=color, mew=.75,
                              zorder=5 if primary else 3)
            if i == 0:
                handles[method] = line
        ax.set_xlim(.65,10.35)
        ax.set_xticks(np.arange(1,11))
        ax.set_axisbelow(True)
        ax.grid(axis="y",which="major",color="#CCCCCC",ls=(0,(4,4)),lw=.5)
        ax.tick_params(length=3,pad=3,width=.65)
        if i == 0:
            ax.set_ylabel("Mean final gradient norm\n"+r"$\Vert\nabla f\Vert_\infty$", labelpad=4)
            upper = max(max(prefix("batch_max")), max(stats[m,d]["mean_"+field] for m in STYLES for d in DIMS))
            ax.set_ylim(0, 1.2 if upper < 1.2 else 1.1*upper)
            if upper < 1.2:
                ax.set_yticks(np.arange(0,1.201,.2))
        elif i == 1:
            ax.set_ylabel("Mean CPU time per run (s)", labelpad=4)
            upper = max(max(prefix("batch_max")), max(stats[m,d]["mean_"+field] for m in STYLES for d in DIMS))
            ax.set_ylim(0,3.2 if upper < 3.2 else 1.1*upper)
            if upper < 3.2:
                ax.set_yticks(np.arange(0,3.01,.5))
        elif i == 2:
            ax.set_ylabel("Mean accepted iterations",labelpad=4)
            upper=max(max(prefix("batch_max")),
                      max(stats[m,d]["mean_"+field] for m in STYLES for d in DIMS))
            ax.set_ylim(0,upper*1.10 if upper>0 else 1)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=(i==2),min_n_ticks=4))
        # Fixed physical offset from the two axes, for matched panel labels.
        labels.append(ax.text(0,1,["(a)","(b)","(c)"][i],
                              transform=ax.transAxes+ScaledTranslation(-16/72,6/72,fig.dpi_scale_trans),
                              va="bottom",ha="left",weight="bold",fontsize=8.7))
    fig.supxlabel("Dimension (×10³)",x=.54,y=.082,fontsize=8.3)
    # Matplotlib fills columns first: read first row M5, M1, M2; second M3, M4, M6.
    order = ["M5","M3","M1","M4","M2","M6"]
    leg = fig.legend([handles[m] for m in order], [handles[m].get_label() for m in order],
                     loc="upper center",bbox_to_anchor=(.525,.992),ncols=3,fontsize=7.2,
                     handlelength=2.55,columnspacing=2.7,handletextpad=.55,labelspacing=.60)
    for text, method in zip(leg.get_texts(), order):
        if method=="M5": text.set_weight("bold")
    fig.canvas.draw()
    boxes=[a.get_position().bounds for a in axes]
    assert all(np.allclose(boxes[0][2:],b[2:]) for b in boxes)
    assert all(abs(b[1]-boxes[0][1])<1e-12 for b in boxes)
    assert all(abs(t.get_window_extent().y0-labels[0].get_window_extent().y0)<.1 for t in labels)
    for ax in axes:
        assert len(ax.lines)==6 and len(ax.collections)==1
    assert all(leg.get_window_extent().y0 > t.get_window_extent().y1 for t in labels)
    fig.savefig(out/(STEM+".pdf"),metadata={"Title":"Local quench accuracy, CPU cost and accepted iterations"})
    fig.savefig(out/(STEM+".svg"))
    fig.savefig(out/(STEM+".png"),dpi=360)
    plt.close(fig)


def output_path(path):
    target = (path if path.is_absolute() else ROOT/path).resolve()
    results = (ROOT/"results").resolve()
    require(ROOT in results.parents and results in target.parents,
            "--out must be a subdirectory of this repository's results/ folder")
    require(not target.exists() or target.is_dir(), "Output must be a directory")
    require(not target.exists() or not any(target.iterdir()),
            "Output directory must be empty; choose another --out")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--out",type=Path,default=Path("results/quench_figure"))
    args=parser.parse_args(argv)
    try:
        args.out = output_path(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    stats,batches,totals=aggregate(args.data)
    checked = compare_reference(stats) if args.data.resolve() == DEFAULT_DATA else None
    args.out.mkdir(exist_ok=True,parents=True)
    write_csv(args.out/"figure1_plot_data.csv", [stats[m,d] for m in sorted(STYLES) for d in DIMS])
    write_csv(args.out/"seed_batch_means.csv", batches)
    draw(stats,args.out)
    caption = CAPTION if checked is not None else (
        "Dimension dependence of local-quench outcomes in the supplied dataset. "
        "Panels show mean final gradient infinity norm, mean process CPU seconds and mean accepted iterations. "
        "Each point includes 300 runs, including budget-limited runs; shading denotes the observed range "
        "of 30 ten-function batch means for SD–NCG–L-BFGSM, not a confidence interval. "
        "Consult the supplied dataset's provenance for its selection history."
    )
    (args.out/"caption.txt").write_text(caption+"\n",encoding="utf-8")
    (args.out/"plot_manifest.json").write_text(json.dumps({
        "source_csv_sha256":hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "runs":sum(totals.values()), "aggregate_cells":len(stats),
        "reference_cells_checked":checked, "convergence_counts":dict(totals),
        "numpy":np.__version__, "matplotlib":mpl.__version__,
        "shading":"Observed range of 30 ten-function batch means; not a confidence interval",
    },indent=2)+"\n",encoding="utf-8")
    print("CSV SHA-256:",hashlib.sha256(args.data.read_bytes()).hexdigest())
    print("Included all runs:",dict(totals))


if __name__ == "__main__":
    main()
