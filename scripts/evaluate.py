"""Binary metrics from audit + benchmark; event metrics require real future annotations."""

import argparse
import json
from pathlib import Path


def evaluate(audit, benchmark, annotations=None):
    labels = {r["video_id"]: r for r in audit["videos"]}
    tp = tn = fp = fn = failures = 0
    errors = []
    category_counts = {}
    for result in benchmark["results"]:
        label = labels.get(result["video_id"])
        if label is None:
            raise ValueError("Benchmark video missing from ground truth")
        if result["status"] != "done":
            failures += 1
            continue
        positive = result["verdict"] == "violation"
        actual = bool(label["binary_label"])
        tp += int(positive and actual)
        tn += int(not positive and not actual)
        fp += int(positive and not actual)
        fn += int(not positive and actual)
        if actual and positive and label["timecode_sec"] is not None:
            errors.append(
                min(
                    abs(v["timecode_sec"] - label["timecode_sec"])
                    for v in result["violations"]
                )
            )
        if annotations is not None:
            expected = [x for x in annotations if x["video_id"] == result["video_id"]]
            matched = set()
            for prediction in result["violations"]:
                category = prediction["type"]
                counts = category_counts.setdefault(
                    category, {"tp": 0, "fp": 0, "fn": 0}
                )
                eligible = [
                    (i, event)
                    for i, event in enumerate(expected)
                    if i not in matched
                    and event["category"] == category
                    and event["start_sec"] - 1
                    <= prediction["timecode_sec"]
                    <= event["end_sec"] + 1
                ]
                if eligible:
                    chosen = min(
                        eligible,
                        key=lambda pair: abs(
                            pair[1]["start_sec"] - prediction["timecode_sec"]
                        ),
                    )[0]
                    matched.add(chosen)
                    counts["tp"] += 1
                else:
                    counts["fp"] += 1
            for i, event in enumerate(expected):
                if i not in matched:
                    category_counts.setdefault(
                        event["category"], {"tp": 0, "fp": 0, "fn": 0}
                    )["fn"] += 1
    total = len(benchmark["results"])
    return {
        "video": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "failed_or_rejected": failures,
            "failure_rate": failures / total if total else None,
            "recall_completed_only": tp / (tp + fn) if tp + fn else None,
            "precision_completed_only": tp / (tp + fp) if tp + fp else None,
        },
        "timecode_nearest_label_mae_sec": sum(errors) / len(errors) if errors else None,
        "timecode_note": "Binary sheet provides one reference point; not full event intervals. Failures reported separately.",
        "categories": category_counts if annotations is not None else None,
        "evidence_relevance": None,
        "evidence_note": "Requires independent human review; image presence is not relevance.",
        "latency": benchmark["summary"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    annotations = (
        [
            json.loads(line)
            for line in a.annotations.read_text().splitlines()
            if line.strip()
        ]
        if a.annotations
        else None
    )
    result = evaluate(
        json.loads(a.audit.read_text()),
        json.loads(a.benchmark.read_text()),
        annotations,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
