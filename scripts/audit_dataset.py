"""Read only: labels and container metadata. Does not decode or export any video frames."""

import argparse
import json
import re
from pathlib import Path
from openpyxl import load_workbook
from app.video import probe


def parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).lower().strip()
    if ":" in value:
        result = 0.0
        for part in value.split(":"):
            result = result * 60 + float(part)
        return result
    minutes = re.search(r"(\d+)\s*м", value)
    seconds = re.search(r"(\d+)\s*с", value)
    if minutes or seconds:
        return float(
            (int(minutes[1]) * 60 if minutes else 0)
            + (int(seconds[1]) if seconds else 0)
        )
    raise ValueError(f"Unknown timecode format: {value!r}")


def audit(videos: Path, labels: Path):
    paths = {p.stem: p for p in videos.iterdir() if p.suffix.lower() == ".mov"}
    # Never normalize Cyrillic Х to Latin X: exact names prevent silent collisions.
    book = load_workbook(labels, read_only=True, data_only=True)
    rows = list(book.active.values)
    book.close()
    records = []
    for name, timecode, label, *_ in rows[1:]:
        if name is None:
            continue
        stem = Path(str(name).strip()).stem
        path = paths.get(stem)
        if path is None:
            raise ValueError(f"No exact filename match: {name!r}")
        raw = str(label).strip().lower()
        if raw not in ("нарушение", "без нарушения"):
            raise ValueError(f"Unknown label: {label!r}")
        timestamp = parse_time(timecode)
        metadata = probe(path)
        if timestamp is not None and not 0 <= timestamp <= metadata["duration_sec"]:
            raise ValueError(f"Timecode outside duration: {name!r}")
        records.append(
            dict(
                video_id=path.name,
                binary_label=int(raw == "нарушение"),
                timecode_sec=timestamp,
                raw_timecode=timecode,
                size_bytes=path.stat().st_size,
                **metadata,
            )
        )
    if len({r["video_id"] for r in records}) != len(records):
        raise ValueError("Duplicate label rows")
    return dict(
        summary=dict(
            video_count=len(paths),
            label_count=len(records),
            positives=sum(r["binary_label"] for r in records),
            negatives=sum(not r["binary_label"] for r in records),
            bytes=sum(p.stat().st_size for p in paths.values()),
            duration_min_sec=min(r["duration_sec"] for r in records),
            duration_max_sec=max(r["duration_sec"] for r in records),
            fps_min=min(r["fps"] for r in records),
            unlabeled=sorted(
                p.name
                for k, p in paths.items()
                if p.name not in {r["video_id"] for r in records}
            ),
        ),
        videos=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.videos, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
