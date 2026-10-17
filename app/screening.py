"""Experimental visual screening, separate from final category applicability."""

from typing import Literal
from pydantic import Field, model_validator

from .schemas import Category, Event, ModelReport, OverviewReport, StrictModel


class VisualCheck(StrictModel):
    category: Category
    visual_trigger_seen: bool
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, ge=0)
    observed_action: str = Field(min_length=8, max_length=800)
    uncertainty_reason: str = Field(min_length=1, max_length=800)
    evidence_frame_indices: list[int] = Field(max_length=3)

    @model_validator(mode="after")
    def check_evidence(self):
        if self.visual_trigger_seen:
            if (
                self.start_frame_index is None
                or self.end_frame_index is None
                or not self.evidence_frame_indices
            ):
                raise ValueError("A visible trigger requires an interval and evidence")
            if self.end_frame_index < self.start_frame_index or any(
                not self.start_frame_index <= i <= self.end_frame_index
                for i in self.evidence_frame_indices
            ):
                raise ValueError("Invalid screening interval or evidence")
        elif (
            self.start_frame_index is not None
            or self.end_frame_index is not None
            or self.evidence_frame_indices
        ):
            raise ValueError("An absent trigger cannot cite an event interval or evidence")
        return self


class ScreeningReport(StrictModel):
    scene_summary: str = Field(min_length=8, max_length=1600)
    observations: list[str] = Field(min_length=1, max_length=12)
    category_checks: list[VisualCheck] = Field(min_length=7, max_length=7)
    limitations: list[str] = Field(max_length=20)
    review_notes: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def complete_categories(self):
        if {check.category for check in self.category_checks} != set(Category):
            raise ValueError("Screening must consider each category exactly once")
        return self

    def overview(self, frames):
        # A candidate still passes the independent, full applicability check.
        return OverviewReport(
            scene_summary=self.scene_summary,
            observations=self.observations,
            candidates=[
                Event(
                    category=check.category,
                    **frame_interval(
                        check.start_frame_index, check.end_frame_index,
                        check.evidence_frame_indices, frames,
                    ),
                    observed_action=check.observed_action,
                    rule_id=check.category.value,
                    uncertainty_reason=check.uncertainty_reason,
                    applicability_basis="Визуальный кандидат: условия рабочей категории ещё необходимо проверить.",
                )
                for check in self.category_checks
                if check.visual_trigger_seen
            ],
            limitations=self.limitations,
            review_notes=self.review_notes,
        )


def frame_interval(start, end, evidence, frames):
    if not 0 <= start <= end < len(frames) or any(
        not start <= i <= end for i in evidence
    ):
        raise ValueError("Model cited an invalid frame index")
    return dict(
        start_sec=frames[start].timestamp,
        end_sec=frames[end].timestamp,
        evidence_timestamps=[frames[i].timestamp for i in evidence],
    )


class IndexedEvent(StrictModel):
    category: Category
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(ge=0)
    observed_action: str = Field(min_length=8, max_length=1600)
    rule_id: str
    uncertainty_reason: str = Field(min_length=1, max_length=1600)
    applicability_basis: str = Field(min_length=8, max_length=1600)
    evidence_frame_indices: list[int] = Field(min_length=1, max_length=3)

    def event(self, frames):
        return Event(
            category=self.category,
            rule_id=self.rule_id,
            observed_action=self.observed_action,
            uncertainty_reason=self.uncertainty_reason,
            applicability_basis=self.applicability_basis,
            **frame_interval(
                self.start_frame_index, self.end_frame_index,
                self.evidence_frame_indices, frames,
            ),
        )


class VerificationReport(StrictModel):
    events: list[IndexedEvent] = Field(max_length=20)
    limitations: list[str] = Field(max_length=20)
    review_notes: list[str] = Field(max_length=20)
    candidate_decision: Literal["confirmed", "rejected", "inconclusive"]
    decision_basis: str = Field(min_length=12, max_length=1200)

    @model_validator(mode="after")
    def consistent_decision(self):
        if bool(self.events) != (self.candidate_decision == "confirmed"):
            raise ValueError("Candidate decision and events disagree")
        return self

    def final_report(self, frames):
        notes = list(self.review_notes)
        if self.candidate_decision != "confirmed":
            notes = ["Проверка кандидата: " + self.decision_basis] + notes[:19]
        return ModelReport(
            events=[event.event(frames) for event in self.events],
            limitations=self.limitations,
            review_notes=notes,
        )
