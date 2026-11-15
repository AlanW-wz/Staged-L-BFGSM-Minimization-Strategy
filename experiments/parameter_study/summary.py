"""Validate and summarize the 112 archived cautious-update runs.

Only runtime, final-gradient and convergence columns have recoverable raw data.
The reported memory-use and pair-acceptance columns are not reconstructed here.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from .config import DIMENSIONS, PROBLEMS, ROOT, SETTINGS, setting_label
from .run import output_path


def summarize(path=None):
    path = Path(path) if path is not None else ROOT / "data/parameter_study/results.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = set(itertools.product(PROBLEMS, DIMENSIONS, SETTINGS))
    groups = {key: [] for key in SETTINGS}
    actual = set()
    for row in rows:
        key = row["problem"], int(row["dimension"]), row["method"]
        if key in actual:
            raise ValueError(f"Duplicated parameter-study run: {key}")
        actual.add(key)
        if key not in expected:
            raise ValueError(f"Unexpected parameter-study run: {key}")
        g, seconds = float(row["grad_norm"]), float(row["elapsed_seconds"])
        if not math.isfinite(g) or g < 0 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid gradient/timing: {key}")
        flag = row["converged"].lower()
        if flag not in ("true", "false"):
            raise ValueError(f"Invalid convergence flag: {key}")
        groups[row["method"]].append((g, seconds, flag == "true"))
    if actual != expected:
        raise ValueError(f"Expected all 112 parameter-study rows, got {len(actual)}")
    base = math.fsum(row[1] for row in groups["lbfgsm_wolfe"]) / 28
    if base <= 0:
        raise ValueError("Reference setting must have positive mean wall time")
    summary = []
    for method, values in groups.items():
        mean_time = math.fsum(v[1] for v in values) / len(values)
        summary.append({
            "method": method, "parameter_setting": setting_label(method),
            "runs": len(values), "mean_wall_seconds": mean_time,
            "wall_time_ratio": mean_time / base,
            "mean_log10_G": math.fsum(math.log10(max(v[0], 1e-16)) for v in values) / len(values),
            "converged_fraction": sum(v[2] for v in values) / len(values),
        })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/parameter_study/results.csv")
    parser.add_argument("--out", type=Path, help="New CSV under this repository's results/; omitted to print JSON")
    args = parser.parse_args(argv)
    rows = summarize(args.data)
    if args.out:
        try:
            out = output_path(args.out)
        except ValueError as error:
            parser.error(str(error))
        if out.exists():
            parser.error("Refusing to overwrite an existing output file")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
