from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(str, Enum):
    leave_workplace_during_service = "leave_workplace_during_service"
    cash_counting_without_client = "cash_counting_without_client"
    cash_counting_without_controller = "cash_counting_without_controller"
    previous_cash_left_unsecured = "previous_cash_left_unsecured"
    mixing_cash_between_clients = "mixing_cash_between_clients"
    unregistered_cash_transfer_between_workers = (
        "unregistered_cash_transfer_between_workers"
    )
    personal_items_or_cash_in_cash_area = "personal_items_or_cash_in_cash_area"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Event(StrictModel):
    category: Category
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    observed_action: str = Field(min_length=8, max_length=1600)
    rule_id: str
    uncertainty_reason: str = Field(min_length=1, max_length=1600)
    applicability_basis: str = Field(min_length=8, max_length=1600)
    evidence_timestamps: list[float] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_sec < self.start_sec:
            raise ValueError("Reversed event interval")
        if self.rule_id != self.category.value:
            raise ValueError("Rule/category mismatch")
        if any(
            not self.start_sec <= t <= self.end_sec for t in self.evidence_timestamps
        ):
            raise ValueError("Evidence outside event interval")
        return self


class ModelReport(StrictModel):
    events: list[Event] = Field(max_length=20)
    limitations: list[str] = Field(max_length=20)
    review_notes: list[str] = Field(max_length=20)


class OverviewReport(StrictModel):
    """Visual discovery before rule applicability is assessed in a separate pass."""

    scene_summary: str = Field(min_length=8, max_length=1600)
    observations: list[str] = Field(min_length=1, max_length=12)
    candidates: list[Event] = Field(max_length=20)
    limitations: list[str] = Field(max_length=20)
    review_notes: list[str] = Field(max_length=20)


class EvidenceFrame(StrictModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str


class Violation(StrictModel):
    type: Category
    timecode_sec: float = Field(ge=0)
    description: str
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class DoneResult(StrictModel):
    job_id: str
    status: str = "done"
    verdict: str
    violations: list[Violation]

    @model_validator(mode="after")
    def consistent(self):
        expected = "violation" if self.violations else "no_violation"
        if self.verdict != expected or self.status != "done":
            raise ValueError("Inconsistent verdict")
        return self
