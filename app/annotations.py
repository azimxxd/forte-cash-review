"""Validated future event annotations. No category labels are inferred from binary labels."""

import json
import math
from pathlib import Path
from pydantic import Field, model_validator
from .schemas import Category, StrictModel


class Annotation(StrictModel):
    video_id: str
    category: Category
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    observed_action: str = Field(min_length=8)
    rule_id: str

    @model_validator(mode="after")
    def check(self):
        if Path(self.video_id).name != self.video_id or self.video_id in (".", ".."):
            raise ValueError("video_id must be an exact filename, not a path")
        if self.end_sec < self.start_sec or self.rule_id != self.category.value:
            raise ValueError("Invalid interval or rule")
        return self


def load_annotations(path: Path, manifest: list[dict]):
    videos = {row["video_id"]: row for row in manifest}
    if len(videos) != len(manifest):
        raise ValueError("Duplicate videos in manifest")
    scenes = {}
    for row in manifest:
        if not row.get("scene_id") or row.get("split") not in (
            "train",
            "validation",
            "test",
        ):
            raise ValueError("Explicit scene_id and split required for every video")
        if row.get("origin") not in ("open_dataset", "team_owned", "jury"):
            raise ValueError("Unknown data origin")
        if row["origin"] == "jury" and row["split"] != "test":
            raise ValueError("Jury data can only belong to test")
        if not math.isfinite(row["duration_sec"]) or row["duration_sec"] <= 0:
            raise ValueError("Invalid manifest duration")
        old = scenes.setdefault(row["scene_id"], row["split"])
        if old != row["split"]:
            raise ValueError("Related scenes leak across splits")
    annotations = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        event = Annotation.model_validate_json(line)
        if event.video_id not in videos:
            raise ValueError(f"Line {number}: unknown video_id")
        if event.end_sec > videos[event.video_id]["duration_sec"]:
            raise ValueError(f"Line {number}: timestamp outside video")
        annotations.append(event)
    return annotations
