"""Integrity checks for the exact archived data underlying the revised figures."""
import unittest
from analysis.archives import compare_rows, read_csv, reconstruct, validate_grid, verify_archives, DATA


class ArchivesTest(unittest.TestCase):
    def test_full_reconstruction_and_hashes(self):
        report = verify_archives()
        self.assertEqual(report["datasets"]["manuscript"]["successes"]["M5"], 2596)
        self.assertEqual(report["datasets"]["round2"]["successes"]["M5"], 2595)

    def test_missing_and_duplicate_rows_rejected(self):
        rows = read_csv(DATA / "round1.csv")
        with self.assertRaises(ValueError):
            validate_grid(rows[:-1], "round1")
        with self.assertRaises(ValueError):
            validate_grid(rows + rows[:1], "round1")

    def test_changed_observation_rejected(self):
        round2, _, _ = reconstruct()
        modified = [{**r} for r in round2]
        modified[0]["final_grad_inf"] = "12345"
        with self.assertRaises(ValueError):
            compare_rows(modified, round2)


if __name__ == "__main__":
    unittest.main()
