"""Check archived three-panel quantities, input validation and portable exports."""
import csv
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from analysis import quench_plot


class QuenchPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats, cls.batches, cls.totals = quench_plot.aggregate(quench_plot.DEFAULT_DATA)

    def setUp(self):
        results = quench_plot.ROOT / "results"
        results.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="plot_test_", dir=results)
        self.folder = Path(self.tmp.name)
        self.assertEqual(self.folder.resolve().parent, results.resolve())

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_archived_aggregate_cells_are_reproduced(self):
        self.assertEqual(sum(self.totals.values()), 18000)
        self.assertEqual(len(self.stats), 60)
        self.assertEqual(len(self.batches), 1800)
        self.assertEqual(quench_plot.compare_reference(self.stats), 60)
        for record in self.stats.values():
            self.assertEqual(record["n_runs"], 300)
            self.assertEqual(record["n_batches"], 30)

    def test_bad_data_do_not_silently_disappear(self):
        with quench_plot.DEFAULT_DATA.open(encoding="utf-8-sig", newline="") as stream:
            original = next(csv.DictReader(stream))
        path = self.folder / "invalid.csv"
        for variant, message in (([original, original], "Duplicate"),
                                 ([{**original, "final_grad_inf": "nan"}], "finite"),
                                 ([original], "complete")):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(original))
                writer.writeheader()
                writer.writerows(variant)
            with self.assertRaisesRegex(ValueError, message):
                quench_plot.aggregate(path)

    def test_caption_uses_method_names_without_seed_ids(self):
        self.assertIn("SD–NCG–L-BFGSM", quench_plot.CAPTION)
        self.assertIn("informed by preceding results", quench_plot.CAPTION)
        self.assertIsNone(re.search(r"\b[MF]\d+\b|202611\d+", quench_plot.CAPTION))

    def test_protected_and_nonempty_output_directories_are_refused(self):
        for target in (quench_plot.ROOT, quench_plot.ROOT / "data/local_quench", quench_plot.ROOT.parent / "external"):
            with self.assertRaises(ValueError):
                quench_plot.output_path(target)
        (self.folder / "existing.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            quench_plot.output_path(self.folder)
        self.assertEqual((self.folder / "existing.txt").read_text(), "keep")

    def test_main_writes_publication_exports_and_matching_aggregates(self):
        quench_plot.main(["--out", str(self.folder)])
        for extension in ("pdf", "png", "svg"):
            self.assertGreater((self.folder / f"{quench_plot.STEM}.{extension}").stat().st_size, 10000)
        manifest = json.loads((self.folder / "plot_manifest.json").read_text())
        self.assertEqual(manifest["reference_cells_checked"], 60)
        self.assertEqual(manifest["runs"], 18000)
        with (self.folder / "figure1_plot_data.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        self.assertIn("method_name", rows[0])
        self.assertEqual((self.folder / "caption.txt").read_text(encoding="utf-8").strip(), quench_plot.CAPTION)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            quench_plot.main(["--out", str(self.folder)])


if __name__ == "__main__":
    unittest.main()
