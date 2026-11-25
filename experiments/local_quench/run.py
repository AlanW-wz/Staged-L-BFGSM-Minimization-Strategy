"""Run paired local quenches with atomic outcome/trace checkpoints.

Use ``python -m experiments.local_quench.run --plan-only`` to inspect the
formal grid, or add ``--smoke --out results/quench_smoke`` for a small check.
Fresh CPU measurements are not expected to match archived measurements.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Apply before workers or the main process import any numeric library.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _variable in THREAD_VARS:
    os.environ[_variable] = "1"

import numpy as np

from . import problems, solvers
from .names import METHOD_LABELS, METHOD_ALIASES, FUNCTION_LABELS, method_id


SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[1]
NUMERICAL_FILES = tuple(sorted(p.relative_to(SOURCE_DIR).as_posix()
                              for p in SOURCE_DIR.rglob("*.py")))
METHODS = tuple(f"M{i}" for i in range(1, 7))
FIDS = tuple(range(1, 11))
DIMENSIONS = tuple(range(1000, 10001, 1000))
SEEDS = tuple(range(2026110701, 2026110731))
FORMAL_SETTINGS = dict(max_iterations=2000, max_evaluations=20000, tol=1e-6)
SOLVER_SETTINGS = FORMAL_SETTINGS.copy()
SMOKE = False
KEY = ("fid", "d", "seed", "method")
CSV_FIELDS = ("fid", "name", "d", "seed", "method", "final_grad_l2", "final_grad_inf",
              "cpu_seconds", "wall_seconds", "converged", "reason", "evaluations", "iterations",
              "initial_grad_l2", "initial_grad_inf", "initial_f", "final_f", "fstar", "x0_sha256",
              "function_name", "method_name", "suite")
SCHEMA = 2
DB_NAME = "quench_results.sqlite3"


def configure(dims=None, seeds=None, methods=None, smoke=False, functions=None):
    """Set the workload; workers receive settings explicitly on submission."""
    global SMOKE, DIMENSIONS, SEEDS, METHODS, SOLVER_SETTINGS, FIDS
    SMOKE = bool(smoke)
    aliases = {str(fid): fid for fid in problems.IDS}
    aliases.update({name.lower(): fid for fid, name in problems.NAMES.items()})
    aliases.update({name.lower(): fid for fid, name in FUNCTION_LABELS.items()})
    try:
        FIDS = tuple(aliases[str(value).lower()] for value in
                     (functions if functions is not None else problems.IDS))
    except KeyError as exc:
        raise ValueError(f"Unknown function {exc.args[0]!r} in the quench study") from exc
    DIMENSIONS = tuple(dims if dims is not None else ((16, 32) if smoke else range(1000, 10001, 1000)))
    SEEDS = tuple(seeds if seeds is not None else ((914276,) if smoke else range(2026110701, 2026110731)))
    METHODS = tuple(method_id(m) for m in (methods if methods is not None else METHOD_LABELS))
    if not DIMENSIONS or any(d < 8 or d % 8 for d in DIMENSIONS):
        raise ValueError("Dimensions must be positive multiples of eight, at least 8")
    if not SEEDS or any(s < 0 for s in SEEDS):
        raise ValueError("Seeds must be nonnegative integers")
    for name, values in (("functions", FIDS), ("dimensions", DIMENSIONS), ("seeds", SEEDS), ("methods", METHODS)):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} are not allowed")
    if not METHODS or not FIDS:
        raise ValueError("Select at least one method and one function")
    SOLVER_SETTINGS = (dict(max_iterations=30, max_evaluations=500, tol=1e-6)
                       if smoke else FORMAL_SETTINGS.copy())


def output_path(path):
    """Confine all generated checkpoints to a subdirectory of repository results/."""
    target = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    results = (REPO_ROOT / "results").resolve()
    if REPO_ROOT not in results.parents or target == results or results not in target.parents:
        raise ValueError("--out must be a subdirectory of this repository's results/ folder")
    return target


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


def task_order(limit):
    """Each 600-job wave covers every function/dimension/method once at a shared seed."""
    tasks = []
    for seed in SEEDS:
        rng = random.Random(seed)
        fids, dimensions, methods = list(FIDS), list(DIMENSIONS), list(METHODS)
        for values in (fids, dimensions, methods):
            rng.shuffle(values)
        for offset in range(len(methods)):
            diagonals = list(range(len(dimensions)))
            rng.shuffle(diagonals)
            for diagonal in diagonals:
                positions = list(range(len(fids)))
                rng.shuffle(positions)
                for i in positions:
                    # Every group of 10 spans all functions and dimensions.
                    task = (fids[i], dimensions[(i + diagonal) % len(dimensions)], seed,
                            methods[(offset + i + 3 * diagonal) % len(methods)])
                    tasks.append(task)
                    if limit and len(tasks) >= limit:
                        return tasks
    return tasks


def make_manifest(tasks, limit):
    formal_grid = (DIMENSIONS == tuple(range(1000, 10001, 1000))
                   and SEEDS == tuple(range(2026110701, 2026110731))
                   and METHODS == tuple(METHOD_LABELS) and FIDS == problems.IDS and not limit)
    manifest = dict(schema=SCHEMA, design="single-quench/stratified-waves-v1",
                    suite="local_quench", mode="smoke" if SMOKE else ("formal" if formal_grid else "subset"), limit=limit,
                    target_runs=len(tasks), functions={fid: problems.NAMES[fid] for fid in FIDS},
                    function_ids=FIDS, dimensions=DIMENSIONS,
                    function_labels={fid: FUNCTION_LABELS[fid] for fid in FIDS},
                    methods=METHODS, method_labels={m: METHOD_LABELS[m] for m in METHODS},
                    seeds=SEEDS, solver=SOLVER_SETTINGS,
                    initialization="numpy.default_rng(SeedSequence([seed, fid, d]))",
                    workload_sha256=hashlib.sha256(encoded(tasks)).hexdigest(),
                    numerical_threads={var: os.environ[var] for var in THREAD_VARS},
                    python=sys.version, numpy=np.__version__, platform=platform.platform(),
                    machine=platform.machine(),
                    source_sha256={name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
                                   for name in NUMERICAL_FILES},
                    provenance="New run of the portable local-quench package.")
    manifest["fingerprint"] = hashlib.sha256(encoded(manifest)).hexdigest()
    return manifest


@contextmanager
def exclusive_writer(folder):
    """OS releases the byte-range lock even if the runner is killed."""
    with (folder / "writer.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another writer holds {folder / 'writer.lock'}") from exc
        yield


def open_checkpoint(folder, manifest):
    db = sqlite3.connect(folder / DB_NAME, timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS runs(
                fid INTEGER NOT NULL,d INTEGER NOT NULL,seed INTEGER NOT NULL,method TEXT NOT NULL,
                result_json TEXT NOT NULL,trace_npz BLOB NOT NULL,
                PRIMARY KEY(fid,d,seed,method))""")
            existing = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if existing is None:
                if db.execute("SELECT 1 FROM runs LIMIT 1").fetchone():
                    raise RuntimeError("Checkpoint has results without a workload fingerprint")
                db.executemany("INSERT INTO metadata VALUES(?,?)",
                               [("fingerprint", manifest["fingerprint"]),
                                ("manifest", encoded(manifest).decode())])
            elif existing[0] != manifest["fingerprint"]:
                raise RuntimeError("Source, software, settings, or task selection changed. "
                                   "Use another --out directory or restore the original experiment.")
        atomic_json(folder / "manifest.json", manifest)
        return db
    except BaseException:
        db.close()
        raise


def run_one(task, solver_settings=None):
    """Return one metrics dictionary and compressed trace bytes without file I/O."""
    fid, dimension, seed, method = task
    objective = problems.problem(fid, dimension)
    x0 = problems.initial_point(fid, dimension, seed)
    x0_hash = hashlib.sha256(x0.tobytes()).hexdigest()
    begin_wall, begin_cpu = time.perf_counter(), time.process_time()
    result, trace = solvers.solve(objective.value_grad, x0, method,
                                 **(FORMAL_SETTINGS if solver_settings is None else solver_settings))
    cpu_seconds = time.process_time() - begin_cpu
    wall_seconds = time.perf_counter() - begin_wall
    result.update(fid=fid, name=problems.NAMES[fid], d=dimension, seed=seed, method=method,
                  x0_sha256=x0_hash, fstar=float(objective.fstar), cpu_seconds=cpu_seconds,
                  wall_seconds=wall_seconds, suite="local_quench",
                  function_name=FUNCTION_LABELS[fid], method_name=METHOD_LABELS[method])
    if set(CSV_FIELDS).difference(result):
        raise ValueError(f"Solver result lacks {set(CSV_FIELDS).difference(result)}")
    if (len(trace["evals"]) != result["iterations"] + 1
            or len({len(values) for values in trace.values()}) != 1):
        raise ValueError("Trace length does not match the number of accepted iterations")
    trace_file = io.BytesIO()
    np.savez_compressed(trace_file, **trace)
    return result, trace_file.getvalue()


class Counts:
    def __init__(self, tasks):
        self.target = Counter((f, d, m) for f, d, _, m in tasks)
        self.completed = Counter()
        self.solve_wall = Counter()
        self.cpu = Counter()
        self.successes = Counter()
        self.reasons = Counter()
        self.initial_hashes = {}
        self.done = set()

    def add(self, row):
        key = tuple(row[name] for name in KEY)
        fid, dim, seed, method = key
        pair = (fid, dim, seed)
        expected = self.initial_hashes.setdefault(pair, row["x0_sha256"])
        if expected != row["x0_sha256"]:
            raise ValueError(f"Paired starts differ at {pair}")
        if key in self.done:
            raise ValueError(f"Duplicate key {key}")
        self.done.add(key)
        stratum = fid, dim, method
        self.completed[stratum] += 1
        self.solve_wall[stratum] += row["wall_seconds"]
        self.cpu[method] += row["cpu_seconds"]
        self.successes[method] += int(row["converged"])
        self.reasons[row["reason"]] += 1


def start_csv(folder, db, counts):
    path = folder / "quench_results.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for (serialized,) in db.execute("SELECT result_json FROM runs ORDER BY rowid"):
            row = json.loads(serialized)
            counts.add(row)
            writer.writerow(row)
    os.replace(temporary, path)
    return path.open("a", newline="", encoding="utf-8")


def update_progress(folder, counts, workers, start, current_start, status):
    elapsed = max(0., time.perf_counter() - start)
    completed = len(counts.done)
    target = sum(counts.target.values())
    observed_strata = len(counts.completed)
    session_n = completed - current_start
    eta = None
    if completed == target:
        eta, basis = 0., "complete"
    elif observed_strata == len(counts.target):
        # Mean individual elapsed solve time in each stratum, scaled to observed
        # throughput so runner overhead and achieved concurrency are reflected.
        remaining = sum((n - counts.completed[s]) * counts.solve_wall[s] / counts.completed[s]
                        for s, n in counts.target.items())
        all_mean = sum(counts.solve_wall.values()) / max(1, completed)
        parallelism = min(workers, all_mean * session_n / elapsed) if elapsed and session_n else 0
        eta = remaining / parallelism if parallelism > 0 else None
        basis = "stratum means from completed runs and observed session throughput; point estimate"
    else:
        eta = (target - completed) * elapsed / session_n if session_n else None
        basis = "early extrapolation from completed jobs / session wall time; incomplete strata"
    snapshot = dict(updated_utc=utc_now(), status=status, target=target, completed=completed,
                    remaining=target - completed, converged=sum(counts.successes.values()),
                    observed_strata=observed_strata, planned_strata=len(counts.target),
                    workers=workers, session_elapsed_seconds=elapsed, session_completed=session_n,
                    eta_seconds=eta, eta_basis=basis)
    atomic_json(folder / "progress.json", snapshot)
    method_totals = Counter()
    for (_, _, method), n in counts.completed.items():
        method_totals[method] += n
    summary = dict(**snapshot, reasons=dict(counts.reasons),
                   by_method={m: dict(completed=method_totals[m], converged=counts.successes[m],
                                      total_cpu_seconds=counts.cpu[m]) for m in METHODS})
    atomic_json(folder / "summary.json", summary)
    return snapshot


def log_event(stream, event, **fields):
    stream.write(json.dumps(dict(utc=utc_now(), event=event, **fields), ensure_ascii=False) + "\n")
    stream.flush()


def execute(folder, workers, limit):
    folder = output_path(Path(folder))
    tasks = task_order(limit)
    manifest = make_manifest(tasks, limit)
    if folder.exists() and any(folder.iterdir()) and not (folder / DB_NAME).is_file():
        raise ValueError("Output contains unrelated files and no matching checkpoint; choose another --out")
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive_writer(folder):
        db = open_checkpoint(folder, manifest)
        try:
            counts = Counts(tasks)
            with start_csv(folder, db, counts) as csv_stream, (folder / "events.jsonl").open(
                    "a", encoding="utf-8") as events:
                todo = [task for task in tasks if task not in counts.done]
                if len(todo) + len(counts.done) != len(tasks):
                    raise RuntimeError("Checkpoint has a result outside the frozen workload")
                initial = len(counts.done)
                start = time.perf_counter()
                log_event(events, "start", fingerprint=manifest["fingerprint"], workers=workers,
                          resumed=initial, remaining=len(todo))
                update_progress(folder, counts, workers, start, initial, "running" if todo else "complete")
                if not todo:
                    return
                csv_out = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
                last_snapshot = last_log = start
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
                pending = {}
                remaining = iter(todo)

                def refill():
                    while len(pending) < 2 * workers:
                        task = next(remaining, None)
                        if task is None:
                            break
                        pending[pool.submit(run_one, task, SOLVER_SETTINGS)] = task

                try:
                    refill()
                    while pending:
                        finished, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                        for future in finished:
                            task = pending.pop(future)
                            try:
                                row, trace = future.result()
                                if tuple(row[k] for k in KEY) != task:
                                    raise ValueError("Worker result does not match dispatched task")
                                counts.add(row)
                                with db:
                                    db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                                               (*task, encoded(row).decode(), trace))
                                csv_out.writerow(row)
                                csv_stream.flush()
                            except BaseException:
                                log_event(events, "task_error", task=task, traceback=traceback.format_exc())
                                raise
                        refill()
                        now = time.perf_counter()
                        if now - last_snapshot >= 5:
                            snapshot = update_progress(folder, counts, workers, start, initial, "running")
                            last_snapshot = now
                            if now - last_log >= 30:
                                log_event(events, "progress", **snapshot)
                                last_log = now
                except BaseException:
                    update_progress(folder, counts, workers, start, initial, "interrupted")
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
                snapshot = update_progress(folder, counts, workers, start, initial, "complete")
                log_event(events, "complete", **snapshot)
        finally:
            db.close()


def read_progress(folder):
    # Read directly so the committed count includes jobs since the last refresh.
    path = folder / DB_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)
    try:
        total = db.execute("SELECT count(*) FROM runs").fetchone()[0]
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    finally:
        db.close()
    snapshot_path = folder / "progress.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
    snapshot.update(committed=total, target=manifest["target_runs"], mode=manifest["mode"])
    print(json.dumps(snapshot, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Checkpoint directory under repository results/")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true", help="All functions/methods at dimensions 16/32, one diagnostic seed, 30 iterations")
    parser.add_argument("--dims", type=int, nargs="+", help="Custom dimensions (multiples of eight); reported as a subset")
    parser.add_argument("--seeds", type=int, nargs="+", help="Custom nonnegative seeds")
    parser.add_argument("--functions", nargs="+", help="Function registry names or stable numeric IDs (for example block8_corrugated_rosenbrock or 9)")
    parser.add_argument("--methods", nargs="+", help="Methods: " + ", ".join(METHOD_ALIASES) + "; archival M IDs also accepted")
    parser.add_argument("--limit", type=int, help="Truncate the selected workload; use a separate output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--progress-only", action="store_true")
    mode.add_argument("--plan-only", action="store_true", help="Print workload and source fingerprint; do not run or write")
    args = parser.parse_args(argv)
    if args.out is None and not args.plan_only:
        parser.error("--out is required for execution or progress inspection")
    try:
        configure(args.dims, args.seeds, args.methods, args.smoke, args.functions)
        folder = output_path(args.out) if args.out else None
    except ValueError as exc:
        parser.error(str(exc))
    if args.progress_only:
        read_progress(folder)
        return
    if args.workers < 1 or (sys.platform == "win32" and args.workers > 61):
        parser.error("invalid number of workers")
    count = len(FIDS) * len(DIMENSIONS) * len(SEEDS) * len(METHODS)
    if args.limit is not None and not 1 <= args.limit < count:
        parser.error(f"--limit must be between 1 and {count-1}; omit for the complete selected grid")
    if args.plan_only:
        print(json.dumps(make_manifest(task_order(args.limit), args.limit), indent=2))
    else:
        execute(folder, args.workers, args.limit)


if __name__ == "__main__":
    main()
