import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import pytest
from pydantic import ValidationError
from app.annotations import load_annotations
from app.pipeline import analyze, merge_events, validate_report
from app.schemas import Event, ModelReport, OverviewReport
from app.store import Store, QueueFull
from app.video import extract, probe, sample_times


def event(**changes):
    values = dict(
        category="leave_workplace_during_service",
        start_sec=0.0,
        end_sec=1.5,
        observed_action="Тестовое наблюдаемое действие",
        rule_id="leave_workplace_during_service",
        uncertainty_reason="Тестовые ограничения",
        applicability_basis="Тестовый видимый контекст",
        evidence_timestamps=[0.0],
    )
    return Event(**(values | changes))


class TestOnlyModel:
    __test__ = False

    def __init__(self, positive=True):
        self.positive = positive
        self.calls = 0

    def reset_memory(self):
        pass

    def memory(self):
        return {"test_only_model": True}

    def infer(self, frames, context, deadline):
        self.calls += 1
        events = (
            [event(evidence_timestamps=[frames[0].timestamp])] if self.positive else []
        )
        if context.get("phase") in ("discover", "audit"):
            return OverviewReport(
                scene_summary="Синтетические цветные кадры для теста",
                observations=["Меняется цвет тестового изображения"],
                candidates=events,
                limitations=[],
                review_notes=[],
            )
        return ModelReport(events=events, limitations=[], review_notes=[])


def test_original_frames_and_complete_pipeline(settings, synthetic_mov):
    folder = settings.data_dir / "case"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    model = TestOnlyModel()
    stages = []
    result, extended = analyze(
        "case", folder, settings, model, time.time(), time.time() + 30, stages.append
    )
    assert result["verdict"] == "violation" and model.calls == 2
    assert extended["events"][0]["start_sec"] == 0
    assert "overview" in stages and "detail_1" in stages
    frame = result["violations"][0]["evidence_frames"][0]
    image = Image.open(io.BytesIO(base64.b64decode(frame["image_base64"])))
    assert image.format == "JPEG"
    assert abs(image.getpixel((100, 100))[2] - 220) < 12
    assert frame["timestamp_sec"] == 0
    assert (folder / "preview.mp4").stat().st_size > 0
    assert probe(folder / "preview.mp4")["duration_sec"] == pytest.approx(2, abs=0.1)


def test_successful_empty_result(settings, synthetic_mov):
    folder = settings.data_dir / "empty"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    model = TestOnlyModel(False)
    result, extended = analyze(
        "empty",
        folder,
        settings,
        model,
        time.time(),
        time.time() + 30,
        lambda s: None,
    )
    assert result["verdict"] == "no_violation" and result["violations"] == []
    assert model.calls == 2 and extended["metrics"]["negative_audit_performed"] is True
    assert extended["scene_summary"] and len(extended["analysis_samples"]) >= 1


def test_model_failure_is_not_negative(settings, synthetic_mov):
    class Broken(TestOnlyModel):
        def infer(self, *args):
            raise ValueError("malformed model output")

    folder = settings.data_dir / "broken"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    with pytest.raises(ValueError):
        analyze(
            "broken",
            folder,
            settings,
            Broken(),
            time.time(),
            time.time() + 30,
            lambda s: None,
        )
    with pytest.raises(TimeoutError):
        analyze(
            "broken",
            folder,
            settings,
            TestOnlyModel(),
            time.time(),
            time.time() - 1,
            lambda s: None,
        )


def test_time_bounds_and_evidence(synthetic_mov):
    frames = extract(synthetic_mov, [0, 1], 131072)
    with pytest.raises(ValueError):
        validate_report(
            ModelReport(events=[event(end_sec=3)], limitations=[], review_notes=[]),
            2,
            frames,
        )
    with pytest.raises(ValueError):
        validate_report(
            ModelReport(
                events=[event(evidence_timestamps=[0.5])],
                limitations=[],
                review_notes=[],
            ),
            2,
            frames,
        )
    with pytest.raises(ValidationError):
        event(category="no_violation")
    with pytest.raises(ValidationError):
        event(start_sec=float("nan"))
    with pytest.raises(ValidationError):
        event(end_sec=-1)
    with pytest.raises(ValidationError):
        event(rule_id="made_up")
    with pytest.raises(ValidationError):
        event(evidence_timestamps=[2])


def test_overview_covers_long_video():
    times = sample_times(145, 1, 60)
    assert len(times) == 60 and times[0] == 0 and times[-1] > 144


def test_merge_preserves_distinct_operations():
    merged = merge_events(
        [
            event(end_sec=1),
            event(start_sec=0.5, end_sec=1.5, evidence_timestamps=[1]),
            event(start_sec=2, end_sec=3, evidence_timestamps=[2]),
        ]
    )
    assert len(merged) == 2 and merged[0].end_sec == 1.5


def test_queue_atomic_capacity_and_restart(tmp_path):
    store = Store(tmp_path)

    def reserve(n):
        try:
            store.reserve(str(n), "x.mov", time.time(), time.time() + 30, 3)
            return True
        except QueueFull:
            return False

    with ThreadPoolExecutor(max_workers=10) as pool:
        accepted = list(pool.map(reserve, range(10)))
    assert sum(accepted) == 3
    store.recover()
    assert all(
        store.get(str(i))["status"] == "failed" for i, ok in enumerate(accepted) if ok
    )


def test_expired_job_cannot_become_success(tmp_path):
    store = Store(tmp_path)
    store.reserve("a", "x.mov", time.time() - 10, time.time() - 1, 1)
    store.expire()
    store.complete("a", {"verdict": "no_violation"}, {})
    assert store.get("a")["status"] == "failed" and store.get("a")["result"] is None


def test_annotation_scene_leakage_and_no_invented_labels(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text("")
    manifest = [
        dict(
            video_id="X-01.mov",
            scene_id="scene1",
            split="train",
            duration_sec=10,
            origin="open_dataset",
        ),
        dict(
            video_id="Х-01.mov",
            scene_id="scene1",
            split="validation",
            duration_sec=10,
            origin="open_dataset",
        ),
    ]
    with pytest.raises(ValueError, match="leak"):
        load_annotations(path, manifest)
    manifest[1]["split"] = "train"
    assert load_annotations(path, manifest) == []
    manifest[0]["origin"] = "jury"
    with pytest.raises(ValueError, match="Jury"):
        load_annotations(path, manifest)


def test_empty_overview_can_discover_candidate_in_second_pass(settings, synthetic_mov):
    class MissedFirstPass(TestOnlyModel):
        def infer(self, frames, context, deadline):
            self.positive = context.get("phase") != "discover"
            return super().infer(frames, context, deadline)

    folder = settings.data_dir / "recovery"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    model = MissedFirstPass()
    result, extended = analyze(
        "recovery",
        folder,
        settings,
        model,
        time.time(),
        time.time() + 30,
        lambda s: None,
    )
    assert result["verdict"] == "violation" and model.calls == 3
    assert extended["metrics"]["negative_audit_performed"] is True
    assert extended["metrics"]["candidate_count"] == 1


def test_failed_negative_audit_never_returns_no_violation(settings, synthetic_mov):
    class AuditFails(TestOnlyModel):
        def infer(self, frames, context, deadline):
            if context.get("phase") == "audit":
                raise ValueError("audit failed")
            self.positive = False
            return super().infer(frames, context, deadline)

    folder = settings.data_dir / "audit-error"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    with pytest.raises(ValueError, match="audit failed"):
        analyze(
            "audit-error",
            folder,
            settings,
            AuditFails(),
            time.time(),
            time.time() + 30,
            lambda s: None,
        )


def test_visible_candidate_is_not_automatically_a_violation(settings, synthetic_mov):
    class RejectedCandidate(TestOnlyModel):
        def infer(self, frames, context, deadline):
            if context.get("phase") == "verify":
                self.calls += 1
                return ModelReport(
                    events=[],
                    limitations=[],
                    review_notes=[
                        "Условия категории не установлены по видимым действиям."
                    ],
                )
            return super().infer(frames, context, deadline)

    folder = settings.data_dir / "rejected"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    result, extended = analyze(
        "rejected",
        folder,
        settings,
        RejectedCandidate(),
        time.time(),
        time.time() + 30,
        lambda s: None,
    )
    assert (
        result["verdict"] == "no_violation"
        and extended["metrics"]["candidate_count"] == 1
    )
    assert extended["review_notes"]


def test_multiple_separate_candidates_keep_global_context(settings, synthetic_mov):
    class Multiple(TestOnlyModel):
        def infer(self, frames, context, deadline):
            self.calls += 1
            if context["phase"] == "discover":
                late = frames[-1].timestamp
                return OverviewReport(
                    scene_summary="Только синтетическая тестовая последовательность",
                    observations=["Синтетические цветные кадры"],
                    candidates=[
                        event(end_sec=0.4),
                        event(start_sec=1.8, end_sec=1.99, evidence_timestamps=[late]),
                    ],
                    limitations=[],
                    review_notes=[],
                )
            assert frames[0].timestamp == 0 and frames[-1].timestamp > 1.9
            return ModelReport(
                events=[Event.model_validate(context["candidate"])],
                limitations=[],
                review_notes=[],
            )

    folder = settings.data_dir / "multiple"
    folder.mkdir(parents=True)
    (folder / "source.mov").write_bytes(synthetic_mov.read_bytes())
    model = Multiple()
    result, extended = analyze(
        "multiple",
        folder,
        settings,
        model,
        time.time(),
        time.time() + 30,
        lambda s: None,
    )
    assert len(result["violations"]) == 2 and model.calls == 3
