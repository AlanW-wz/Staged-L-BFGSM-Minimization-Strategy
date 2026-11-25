"""Rebuild the Figure 1 table and summarize the paired-start comparison."""
import argparse
import json
from pathlib import Path
from .archives import ROOT, reconstruct, summary_rows, verify_archives, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/rebuilt_data")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out == ROOT / "results" or not out.is_relative_to(ROOT / "results"):
        parser.error("--out must be a subdirectory of this repository's results/")
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a new or empty output directory")
    report = verify_archives()
    original, selected, omitted = reconstruct()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manuscript.csv", selected)
    write_csv(out / "excluded_original_rows.csv", omitted)
    write_csv(out / "method_summary.csv", summary_rows({"paired_starts": original,
                                                        "manuscript": selected}))
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
