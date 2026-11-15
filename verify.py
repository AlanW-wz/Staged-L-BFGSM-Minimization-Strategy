"""Check archived measurements, reconstruction and revised manuscript aggregates."""
import argparse
import ast
import json
import platform
from pathlib import Path
from analysis.archives import ROOT, require, verify_archives
from experiments.parameter_study.summary import summarize


def verify():
    report = {"python": platform.python_version(), "archives": verify_archives()}
    summary = summarize()
    expected = {
        "lbfgsm_wolfe_c000": (1.135, -6.242, .750),
        "lbfgsm_wolfe": (1., -5.398, .714),
        "lbfgsm_wolfe_c025": (1.488, -6.622, .821),
        "lbfgsm_wolfe_c025_c10": (1.922, -6.536, .821),
    }
    for row in summary:
        values = tuple(row[k] for k in ("wall_time_ratio", "mean_log10_G", "converged_fraction"))
        require(all(abs(x-y) <= .0005 for x,y in zip(values, expected[row["method"]])),
                f"Parameter-table aggregate mismatch: {row['method']}")
    report["parameter_study"] = {"runs": 112, "verified_reported_values": 12,
                                  "summary": summary,
                                  "memory_statistics": "Historical reported values only; not independently reconstructed."}
    from analysis.quench_plot import aggregate, compare_reference
    stats, _, totals = aggregate(ROOT / "data/local_quench/manuscript.csv")
    compare_reference(stats)
    report["quench_figure"] = {"dimension_method_cells": len(stats), "all_run_outcomes": dict(totals),
                               "aggregation": "All outcomes; observed range of 30 ten-function batch means."}
    sources = [*ROOT.glob("*.py")]
    for folder in ("experiments", "analysis", "tests"):
        sources.extend((ROOT / folder).rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    report["python_files_parsed"] = len(sources)
    report["scope"] = "Archive and aggregation verification; not a full optimization or molecular rerun."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional new JSON under repository results/")
    args = parser.parse_args(argv)
    if args.out:
        args.out = args.out.resolve()
        if not args.out.is_relative_to(ROOT / "results") or args.out.exists():
            parser.error("--out must be a new file under this repository's results/")
    report = verify()
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
