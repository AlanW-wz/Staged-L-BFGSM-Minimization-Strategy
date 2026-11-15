"""Numerical and archive checks for the parameter experiment."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.parameter_study.config import PROBLEMS, ROOT, SETTINGS
from experiments.parameter_study.objectives import initial_point, problem
from experiments.parameter_study.run import plan, prepare_output
from experiments.parameter_study.solver import lbfgsm, strong_wolfe_line_search
from experiments.parameter_study.summary import main as summary_main, summarize


class ParameterStudyTests(unittest.TestCase):
    def test_analytical_gradients(self):
        for name in PROBLEMS:
            for dimension in (1, 10):
                with self.subTest(function=name, dimension=dimension):
                    f, gradient = problem(name)
                    x = initial_point(name, dimension)
                    direction = np.random.default_rng(42).normal(size=dimension)
                    direction /= np.linalg.norm(direction)
                    h = 1e-5
                    numeric = (f(x + h * direction) - f(x - h * direction)) / (2 * h)
                    analytic = float(gradient(x) @ direction)
                    self.assertLess(abs(numeric - analytic) / max(1, abs(analytic)), 1e-5)

    def test_full_and_smoke_designs(self):
        self.assertEqual(plan()["target_runs"], 112)
        self.assertEqual(plan(smoke=True)["target_runs"], 28)
        self.assertEqual(plan(dimensions=[10000])["target_runs"], 28)
        with self.assertRaises(ValueError):
            plan(dimensions=[999])
        with self.assertRaises(ValueError):
            plan(dimensions=[10000, 10000])

    def test_paired_starts_and_finite_smoke_results(self):
        # The historical helper falls back if SciPy is absent. Require the
        # dependency here so this test actually exercises its Wolfe path.
        import scipy.optimize

        for name in PROBLEMS:
            f, gradient = problem(name)
            x = initial_point(name, 10)
            np.testing.assert_array_equal(x, initial_point(name, 10))
            original = x.copy()
            for c0, c1, c2 in SETTINGS.values():
                result = lbfgsm(f, gradient, x, m=8, c0=c0, c1=c1, c2=c2,
                                max_iter=8, tol=1e-8,
                                line_search_fn=strong_wolfe_line_search)
                self.assertTrue(np.isfinite(result.grad_norm))
                self.assertLessEqual(result.n_iter, 8)
                np.testing.assert_array_equal(x, original)

    def test_archived_table_aggregates(self):
        rows = summarize()
        expected = [(1.135, -6.242, .750), (1.000, -5.398, .714),
                    (1.488, -6.622, .821), (1.922, -6.536, .821)]
        for row, target in zip(rows, expected):
            actual = [row["wall_time_ratio"], row["mean_log10_G"], row["converged_fraction"]]
            np.testing.assert_allclose(actual, target, rtol=0, atol=.0005)

    def test_output_protection(self):
        invalid = (ROOT, ROOT / "results", ROOT / "data/new-results",
                   ROOT / "experiments/new-results", ROOT / "tests/new-results",
                   ROOT.parent / "external-parameter-results",
                   Path("results/../outside-results"))
        for target in invalid:
            with self.subTest(path=target), self.assertRaises(ValueError):
                prepare_output(target)
            with self.subTest(summary_path=target), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                summary_main(["--out", str(target)])

    def test_output_creation_and_no_overwrite(self):
        results = ROOT / "results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="parameter_test_", dir=results) as temporary:
            folder = Path(temporary)
            relative = folder.relative_to(ROOT) / "new-run"
            created = prepare_output(relative)
            self.assertEqual(created, folder / "new-run")
            saved = created / "results.csv"
            saved.write_text("existing result", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                prepare_output(relative)
            self.assertEqual(saved.read_text(encoding="utf-8"), "existing result")

            target = folder.relative_to(ROOT) / "summary.csv"
            with redirect_stdout(io.StringIO()):
                summary_main(["--out", str(target)])
            summary_file = ROOT / target
            original = summary_file.read_bytes()
            self.assertIn(b"mean_log10_G", original)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                summary_main(["--out", str(target)])
            self.assertEqual(summary_file.read_bytes(), original)

    def test_output_symlink_escape_is_refused(self):
        results = ROOT / "results"
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="parameter_link_test_", dir=results) as temporary:
            link = Path(temporary) / "escaped"
            try:
                link.symlink_to(ROOT / "data", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlinks are unavailable: {error}")
            try:
                with self.assertRaises(ValueError):
                    prepare_output(link / "new-run")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    summary_main(["--out", str(link / "new-summary.csv")])
            finally:
                link.unlink()


if __name__ == "__main__":
    unittest.main()
