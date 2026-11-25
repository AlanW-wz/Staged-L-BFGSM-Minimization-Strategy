"""Numerical and checkpoint regression checks for the portable local-quench runner."""
import csv
from contextlib import closing
import hashlib
import io
import itertools
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from experiments.local_quench import run, problems, solvers
import numpy as np


class QuenchNumerics(unittest.TestCase):
    def tearDown(self):
        run.configure()

    def test_archived_solver_preserved(self):
        expected = {
            "solvers.py": "5ad9553df89d6526d471be6fbaec2ee1ac6114d36de899a410d1071d390a1d5d",
        }
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((run.SOURCE_DIR / filename).read_bytes()).hexdigest(), digest)

    def test_all_gradients_and_seed_streams(self):
        rng = np.random.default_rng(19371)
        for fid in problems.IDS:
            with self.subTest(function=problems.NAMES[fid]):
                obj = problems.problem(fid, 16)
                ref = np.random.default_rng(np.random.SeedSequence([713, fid, 16]))
                np.testing.assert_array_equal(
                    problems.initial_point(fid, 16, 713), ref.uniform(*problems.RANGES[fid], 16))
                x = rng.uniform(-0.8, 0.8, 16)
                direction = rng.normal(size=16)
                direction /= np.linalg.norm(direction)
                _, grad = obj.value_grad(x)
                h = 1e-6
                numeric = (obj.value_grad(x+h*direction)[0] - obj.value_grad(x-h*direction)[0])/(2*h)
                analytic = float(grad @ direction)
                self.assertLess(abs(numeric-analytic)/max(1., abs(analytic)), 3e-6)

    def test_all_methods_reach_stationarity_on_quadratic(self):
        for method in solvers.METHODS:
            with self.subTest(method=method):
                result, trace = solvers.solve(lambda x: (float(x @ x)/2, x.copy()), np.ones(8), method)
                self.assertTrue(result["converged"])
                self.assertEqual(result["iterations"], 1)
                self.assertEqual(result["evaluations"], 2)
                self.assertEqual(result["final_grad_inf"], 0)
                self.assertEqual(len(trace["f"]), 2)

    def test_failed_trial_counts_against_budget_without_changing_endpoint(self):
        calls = []
        def quartic(x):
            calls.append(x.copy())
            return float(np.sum(x**4)), 4*x**3
        result, trace = solvers.solve(quartic, np.full(8, 10.), "M5", max_evaluations=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["reason"], "evaluation_limit")
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["initial_f"], result["final_f"])
        self.assertEqual(len(trace["f"]), 1)

    def test_smoke_grid_includes_every_function_and_method(self):
        run.configure(smoke=True)
        tasks = run.task_order(None)
        expected = set(itertools.product(range(1, 11), (16, 32), (914276,), solvers.METHODS))
        self.assertEqual(len(tasks), 120)
        self.assertEqual(set(tasks), expected)
        manifest = run.make_manifest(tasks, None)
        self.assertEqual(manifest["mode"], "smoke")
        self.assertEqual(manifest["solver"]["max_iterations"], 30)
        self.assertIn("run.py", manifest["source_sha256"])
        self.assertIn("benchmarks.py", manifest["source_sha256"])
        planned = subprocess.run(
            [sys.executable, "-B", "reproduce.py", "--study", "quench", "--plan-only"],
            cwd=run.REPO_ROOT, capture_output=True, text=True, check=True, timeout=30)
        root_manifest = json.loads(planned.stdout)
        self.assertEqual(root_manifest["target_runs"], 18000)
        self.assertEqual(root_manifest["mode"], "formal")

    def test_function_selection_matches_archived_name(self):
        run.configure([16], [713], ["sd-ncg-lbfgsm"], functions=["block_bohachevsky3"])
        self.assertEqual(run.task_order(None), [(7, 16, 713, "M5")])
        self.assertEqual(run.make_manifest(run.task_order(None), None)["functions"], {7: "block_bohachevsky3"})
        row, trace_bytes = run.run_one((7, 16, 713, "M5"), {"max_iterations": 0})
        self.assertEqual(row["function_name"], "Two-coordinate Bohachevsky-3 blocks")
        self.assertEqual(row["method_name"], "SD--NCG--L-BFGSM")
        with np.load(io.BytesIO(trace_bytes), allow_pickle=False) as trace:
            self.assertEqual(len(trace["f"]), 1)


class QuenchCheckpoint(unittest.TestCase):
    def setUp(self):
        results = run.REPO_ROOT / "results"
        results.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="quench_test_", dir=results)
        self.folder = Path(self.tmp.name)
        self.assertEqual(self.folder.resolve().parent, results.resolve())

    def tearDown(self):
        run.configure()
        self.tmp.cleanup()

    def test_resume_preserves_paired_records_and_rejects_changed_functions(self):
        run.configure([16], [713], ["lbfgs", "sd-lbfgs"], smoke=True, functions=[1])
        run.execute(self.folder, workers=1, limit=None)
        csv_path = self.folder / "quench_results.csv"
        before = csv_path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(before.decode())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r["x0_sha256"] for r in rows}), 1)
        with closing(sqlite3.connect(self.folder / run.DB_NAME)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM runs").fetchone()[0], 2)
            for serialized, payload in db.execute("SELECT result_json,trace_npz FROM runs"):
                row = json.loads(serialized)
                with np.load(io.BytesIO(payload), allow_pickle=False) as trace:
                    self.assertEqual(len(trace["f"]), row["iterations"]+1)
                    self.assertEqual(float(trace["grad_inf"][-1]), row["final_grad_inf"])
        run.execute(self.folder, workers=1, limit=None)
        self.assertEqual(csv_path.read_bytes(), before)
        run.configure([16], [713], ["lbfgs", "sd-lbfgs"], smoke=True, functions=[2])
        with self.assertRaisesRegex(RuntimeError, "Source, software, settings"):
            run.execute(self.folder, workers=1, limit=None)
        self.assertEqual(csv_path.read_bytes(), before)

    def test_output_guard_and_unrelated_files(self):
        for target in [run.REPO_ROOT, run.REPO_ROOT / "data/test", run.REPO_ROOT.parent / "outside"]:
            with self.assertRaises(ValueError):
                run.output_path(target)
        (self.folder / "do_not_overwrite.txt").write_text("original", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrelated files"):
            run.execute(self.folder, workers=1, limit=None)
        self.assertEqual((self.folder / "do_not_overwrite.txt").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
