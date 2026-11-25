"""Reconstruct the Figure 1 dataset from measurements of the ten quench functions."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/local_quench"
PROVENANCE = ROOT / "provenance/quench"
KEY = ("fid", "d", "seed", "method")
METHODS = tuple(f"M{i}" for i in range(1, 7))
DIMS = tuple(range(1000, 10001, 1000))
SEEDS = tuple(range(2026110701, 2026110731))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"Empty CSV: {path}")
    return rows


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return int(row["fid"]), int(row["d"]), int(row["seed"]), row["method"]


def indexed(rows):
    result = {key(row): row for row in rows}
    require(len(result) == len(rows), "Duplicate experiment keys")
    return result


def equivalent(left, right):
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except (ValueError, TypeError):
        return False


def compare_rows(actual, expected):
    a, b = indexed(actual), indexed(expected)
    require(a.keys() == b.keys(), "Reconstructed dataset has different experiment keys")
    for k in a:
        require(a[k].keys() == b[k].keys(), f"Different columns at {k}")
        require(all(equivalent(a[k][field], b[k][field]) for field in a[k]),
                f"Changed observation or provenance at {k}")


def reconstruct():
    """Apply the documented paired-start selection to the ten-function grid."""
    original = read_csv(DATA / "paired_starts.csv")
    plan = json.loads((PROVENANCE / "replacement_plan.json").read_text(encoding="utf-8"))
    encoded = json.dumps(plan, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    plan_hash = hashlib.sha256(encoded).hexdigest()
    omitted, plotted = [], []
    for row in original:
        if key(row)[:3] == (plan["function_id"], plan["dimension"], plan["old_seed"]):
            omitted.append(row)
        else:
            plotted.append({**row, "plot_batch_seed": row["seed"]})
    supplemental = read_csv(DATA / "supplementary_runs.csv")
    expected = set(itertools.product([plan["function_id"]], [plan["dimension"]],
                                    [plan["new_seed"]], METHODS))
    require(set(indexed(supplemental)) == expected and len(omitted) == 6,
            "The matched-start replacement must include all six methods")
    plotted.extend({**row, "source_manifest_fingerprint": plan_hash,
                    "plot_batch_seed": str(plan["old_seed"])} for row in supplemental)
    plotted.sort(key=key)
    return original, plotted, omitted


def validate_grid(rows, suite, selected=False):
    """Check complete coverage, paired starts, finite endpoints and stopping flags."""
    index = indexed(rows)
    expected = set(itertools.product(range(1, 11), DIMS, SEEDS, METHODS))
    if selected:
        expected.difference_update((9, 1000, SEEDS[0], m) for m in METHODS)
        expected.update((9, 1000, SEEDS[-1] + 1, m) for m in METHODS)
    require(index.keys() == expected, f"Incomplete {suite} experiment grid")
    pairs = defaultdict(list)
    successes = dict.fromkeys(METHODS, 0)
    from experiments.local_quench.problems import NAMES
    for k, row in index.items():
        require(row["name"] == NAMES[k[0]], f"Function name mismatch: {k}")
        inf, l2 = float(row["final_grad_inf"]), float(row["final_grad_l2"])
        seconds, steps, calls = float(row["cpu_seconds"]), int(row["iterations"]), int(row["evaluations"])
        require(all(math.isfinite(v) and v >= 0 for v in (inf, l2, seconds)), f"Nonfinite measurement: {k}")
        require(inf <= l2 * (1 + 1e-12) and l2 <= math.sqrt(k[1]) * inf * (1 + 1e-12), f"Norm mismatch: {k}")
        require(0 <= steps <= 2000 and steps + 1 <= calls <= 20000, f"Budget mismatch: {k}")
        require(row["converged"] in ("True", "False"), f"Invalid convergence flag: {k}")
        success = row["converged"] == "True"
        require(success == (inf <= 1e-6) == (row["reason"] == "converged"), f"Stopping mismatch: {k}")
        require(float(row["final_f"]) <= float(row["initial_f"]) + 1e-12 * max(1, abs(float(row["initial_f"]))), f"Objective increased: {k}")
        successes[k[3]] += success
        pairs[k[:3]].append((k[3], row["x0_sha256"], row["initial_f"],
                            row["initial_grad_l2"], row["initial_grad_inf"]))
    require(all({p[0] for p in values} == set(METHODS) and
                len({p[1:] for p in values}) == 1 for values in pairs.values()), "Unpaired initial conditions")
    return {"runs": len(rows), "paired_starts": len(pairs), "successes": successes}


def summary_rows(datasets):
    from experiments.local_quench.names import METHOD_NAMES
    output = []
    for dataset, rows in datasets.items():
        for method, name in METHOD_NAMES.items():
            group = [r for r in rows if r["method"] == method]
            count = sum(r["converged"] == "True" for r in group)
            output.append({"dataset": dataset, "method_name": name, "runs": len(group),
                           "successes": count, "success_fraction": count / len(group),
                           **{f"mean_{field}": math.fsum(float(r[field]) for r in group) / len(group)
                              for field in ("final_grad_inf", "cpu_seconds", "iterations")}})
    return output


def verify_archives():
    manifest = json.loads((ROOT / "provenance/data_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        require(path.is_relative_to(ROOT), "Manifest path leaves repository")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"Archive hash mismatch: {item['path']}")
    original, selected, omitted = reconstruct()
    compare_rows(selected, read_csv(DATA / "manuscript.csv"))
    compare_rows(omitted, read_csv(DATA / "excluded_original_rows.csv"))
    datasets = {"paired_starts": original, "manuscript": selected}
    return {"files_verified": len(manifest["files"]),
            "datasets": {name: validate_grid(rows, name, name == "manuscript") for name, rows in datasets.items()},
            "unchanged_after_matched_start_replacement": 17994,
            "interpretation": "Outcome-informed sensitivity analysis; min/max batch range is not a confidence interval."}
