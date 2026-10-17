"""General visual guidance only; no filenames, answer keys or expected timestamps."""

import re
from pydantic import Field, model_validator
from .schemas import Category, StrictModel


class GuidancePattern(StrictModel):
    category: Category
    observable_pattern: str = Field(min_length=8, max_length=650)
    uncertainty_cautions: str = Field(min_length=8, max_length=650)

    @model_validator(mode="after")
    def no_answer_keys(self):
        text = self.observable_pattern + " " + self.uncertainty_cautions
        if re.search(
            r"(?:[XХ]-\d|\.mov\b|\b\d{1,2}:\d{2}\b|video_id|timecode_sec|binary_label)",
            text,
            re.I,
        ):
            raise ValueError("Guidance must not contain per-video answer keys")
        return self


class CalibrationProfile(StrictModel):
    version: str = "visual-guidance-v1"
    patterns: list[GuidancePattern] = Field(min_length=1, max_length=21)
