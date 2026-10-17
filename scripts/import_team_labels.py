"""Import expanded teammate labels into a PRIVATE local staging directory.

Does not transmit data, alter the workbook, assign categories, start training,
change detector verdicts, or export descriptions into terminal output.
"""

import argparse
import json
import os
from pathlib import Path
from app.team_labels import read_workbook, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet", default="labels")
    args = parser.parse_args()
    os.umask(0o077)
    baseline = json.loads(args.baseline_audit.read_text())["videos"]
    records = read_workbook(args.workbook, baseline, args.sheet)
    summary = summarize(records)
    if args.output_dir.exists():
        parser.error(
            "Output directory already exists; choose a new directory to preserve prior imports"
        )
    args.output_dir.mkdir(parents=True, mode=0o700)
    with (args.output_dir / "records.jsonl").open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "summary.json").open("x") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
