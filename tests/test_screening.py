import pytest
from types import SimpleNamespace
from pydantic import ValidationError

from app.screening import ScreeningReport, VerificationReport
from app.schemas import Category

FRAMES = [SimpleNamespace(timestamp=t) for t in [0.0, 1.033333, 2.066667]]


def report():
    return {
        "scene_summary": "Синтетическая сцена без реальных банковских данных",
        "observations": ["Одноцветные тестовые изображения"],
        "category_checks": [
            {
                "category": category.value,
                "visual_trigger_seen": False,
                "start_frame_index": None,
                "end_frame_index": None,
                "observed_action": "Видимый триггер в тесте отсутствует",
                "uncertainty_reason": "Синтетический тест",
                "evidence_frame_indices": [],
            }
            for category in Category
        ],
        "limitations": [],
        "review_notes": [],
    }


def test_screening_covers_each_category_without_forcing_positive():
    parsed = ScreeningReport.model_validate(report())
    assert parsed.overview(FRAMES).candidates == []
    duplicate = report()
    duplicate["category_checks"][-1] = duplicate["category_checks"][0]
    with pytest.raises(ValidationError):
        ScreeningReport.model_validate(duplicate)


def test_visible_trigger_is_only_a_candidate_with_real_interval():
    data = report()
    check = data["category_checks"][0]
    check.update(visual_trigger_seen=True, start_frame_index=1, end_frame_index=2, evidence_frame_indices=[1])
    parsed = ScreeningReport.model_validate(data)
    candidate = parsed.overview(FRAMES).candidates[0]
    assert candidate.category == Category.leave_workplace_during_service
    assert candidate.evidence_timestamps == [1.033333]
    assert candidate.start_sec == 1.033333 and candidate.end_sec == 2.066667
    assert "ещё необходимо проверить" in candidate.applicability_basis
    check["evidence_frame_indices"] = [3]
    with pytest.raises(ValidationError):
        ScreeningReport.model_validate(data)
    check["evidence_frame_indices"] = []
    with pytest.raises(ValidationError):
        ScreeningReport.model_validate(data)


def test_negative_check_cannot_smuggle_event_evidence():
    data = report()
    data["category_checks"][0]["evidence_frame_indices"] = [1]
    with pytest.raises(ValidationError):
        ScreeningReport.model_validate(data)


def test_rejected_candidate_cannot_disappear_without_explanation():
    empty = dict(events=[], limitations=[], review_notes=[])
    with pytest.raises(ValidationError):
        VerificationReport(**empty)
    with pytest.raises(ValidationError):
        VerificationReport(**empty, candidate_decision="confirmed", decision_basis="Кандидат в синтетическом тесте")
    parsed = VerificationReport(**empty, candidate_decision="inconclusive", decision_basis="Условия категории в синтетическом примере не видны.")
    assert parsed.final_report(FRAMES).events == []
    assert parsed.final_report(FRAMES).review_notes == ["Проверка кандидата: " + parsed.decision_basis]


def test_confirmed_event_resolves_indices_to_actual_pts_and_rejects_missing_frame():
    candidate = dict(
        category="leave_workplace_during_service", rule_id="leave_workplace_during_service",
        observed_action="Синтетический пример для проверки формата",
        uncertainty_reason="Тестовое ограничение", applicability_basis="Синтетические условия",
        start_frame_index=1, end_frame_index=2, evidence_frame_indices=[2],
    )
    parsed = VerificationReport(events=[candidate], limitations=[], review_notes=[],
        candidate_decision="confirmed", decision_basis="Подтверждён только синтетический пример.")
    event = parsed.final_report(FRAMES).events[0]
    assert event.evidence_timestamps == [2.066667]
    assert event.start_sec == 1.033333 and event.end_sec == 2.066667
    candidate["end_frame_index"] = 3
    invalid = VerificationReport(events=[candidate], limitations=[], review_notes=[],
        candidate_decision="confirmed", decision_basis="Синтетический пример с неверным индексом.")
    with pytest.raises(ValueError, match="invalid frame index"):
        invalid.final_report(FRAMES)
