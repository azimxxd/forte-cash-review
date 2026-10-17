"""Evaluate the real local model on PRIVATE development videos.

Keep all descriptions, evidence and raw results inside the service host. Only
numeric metrics, file IDs and error types are printed. Ground-truth labels and
reference timecodes are NEVER supplied to the analysis function/model.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4
from pydantic import ValidationError

from app.config import Settings
from app.model import QwenAdapter
from app.pipeline import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--input-mode", choices=["images", "video"], default="video")
    p.add_argument("--discovery-style", choices=["narrative", "checklist"], default="narrative")
    p.add_argument("--video-id", action="append", default=[])
    p.add_argument("--limit-per-label", type=int, default=0)
    a = p.parse_args()
    os.umask(0o077)
    a.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    labels = json.loads(a.audit.read_text())["videos"]
    if a.video_id:
        selected = set(a.video_id)
        labels = [r for r in labels if r["video_id"] in selected]
        if len(labels) != len(selected):
            raise ValueError("Requested video not found")
    elif a.limit_per_label:
        labels = [
            r
            for label in (1, 0)
            for r in sorted(
                [x for x in labels if x["binary_label"] == label],
                key=lambda x: x["duration_sec"],
            )[: a.limit_per_label]
        ]
    s = Settings()
    s.qwen_input_mode = a.input_mode
    s.qwen_discovery_style = a.discovery_style
    s.calibration_profile_path = str(a.profile) if a.profile else ""
    s.overview_fps = 2 if a.input_mode == "video" else 1
    s.max_overview_frames = 240 if a.input_mode == "video" else 145
    model = QwenAdapter(s)
    print(
        json.dumps(
            {
                "stage": "loading",
                "video_count": len(labels),
                "input_mode": a.input_mode,
                    "discovery_style": a.discovery_style,
                "guidance_enabled": bool(a.profile),
            }
        ),
        flush=True,
    )
    model.load()
    results = []
    for reference in labels:
        video_id = reference["video_id"]
        source = (a.dataset_dir / video_id).resolve()
        if not source.is_file() or source.parent != a.dataset_dir.resolve():
            raise ValueError("Invalid input path")
        job_id = str(uuid4())
        folder = a.output_dir / job_id
        folder.mkdir(mode=0o700)
        (folder / "source.mov").symlink_to(source)
        created = time.time()
        record = {
            "video_id": video_id,
            "job_id": job_id,
            "actual_binary_label": reference["binary_label"],
        }
        print(
            json.dumps(
                {
                    "stage": "analyzing",
                    "video_id": video_id,
                    "duration_sec": reference["duration_sec"],
                }
            ),
            flush=True,
        )
        try:
            result, extended = analyze(
                job_id,
                folder,
                s,
                model,
                created,
                created + s.job_timeout_sec,
                lambda name: None,
            )
            (folder / "result.json").write_text(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            (folder / "extended.json").write_text(
                json.dumps(extended, ensure_ascii=False) + "\n"
            )
            record.update(
                status="done",
                predicted_binary_label=int(result["verdict"] == "violation"),
                event_count=len(result["violations"]),
                candidate_count=extended["metrics"]["candidate_count"],
                candidate_categories=[candidate["category"] for candidate in extended.get("candidates", [])],
                total_sec=extended["metrics"]["total_sec"],
            )
            if (
                reference["binary_label"]
                and result["violations"]
                and reference["timecode_sec"] is not None
            ):
                record["nearest_reference_error_sec"] = min(
                    abs(v["timecode_sec"] - reference["timecode_sec"])
                    for v in result["violations"]
                )
        except Exception as exc:
            # Full diagnostics remain private; never print the exception message.
            import traceback

            (folder / "failure.json").write_text(json.dumps({
                "error_type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
                "raw_output": getattr(model, "last_output", ""),
            }, ensure_ascii=False) + "\n")
            record.update(
                status="failed",
                error_type=type(exc).__name__,
                total_sec=time.time() - created,
            )
            if isinstance(exc, ValidationError):
                record["validation_errors"] = [
                    {"type": err["type"], "loc": list(err["loc"])}
                    for err in exc.errors(include_input=False, include_url=False)
                ]
        record["inference_calls"] = list(model.inference_metrics)
        results.append(record)
        (a.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "evaluation_kind": "development_only_not_independent_holdout",
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
    counts = Counter()
    for r in results:
        if r["status"] != "done":
            counts["failed"] += 1
            continue
        actual, predicted = r["actual_binary_label"], r["predicted_binary_label"]
        counts[
            "tp"
            if actual and predicted
            else "fn"
            if actual
            else "fp"
            if predicted
            else "tn"
        ] += 1
    summary = {
        "evaluation_kind": "development_only_not_independent_holdout",
        "input_mode": a.input_mode,
        "discovery_style": a.discovery_style,
        "guidance_enabled": bool(a.profile),
        "video_count": len(results),
        **dict(counts),
    }
    (a.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps({"stage": "failed", "error_type": type(exc).__name__}),
            flush=True,
        )
        raise SystemExit(1)
