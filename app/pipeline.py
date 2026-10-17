import base64
import time
import resource
import sys
from pathlib import Path
from .schemas import ModelReport, OverviewReport, DoneResult, Violation, EvidenceFrame
from .rules import RULE_BY_ID, violation_description
from .video import probe, extract, sample_times, make_preview

LIMITATIONS = [
    "Результат — признаки для проверки человеком, а не юридическое заключение.",
    "Связь рабочих категорий с конкретными пунктами нормативов не проверена.",
    "Редкая выборка может пропустить короткие движения; ракурс и перекрытия ограничивают наблюдение.",
    "Регистрация операций, личная принадлежность денег и внутренние процедуры могут быть не видны.",
    "Неизвестные типы событий могут не распознаться; качество и скорость требуют оценки на GPU.",
]


def validate_report(report: ModelReport | OverviewReport, duration: float, frames):
    known = [f.timestamp for f in frames]
    for event in (
        report.candidates if isinstance(report, OverviewReport) else report.events
    ):
        if event.end_sec > duration or event.rule_id not in RULE_BY_ID:
            raise ValueError("Event outside duration or unknown rule")
        if any(
            min(abs(t - k) for k in known) > 0.001 for t in event.evidence_timestamps
        ):
            raise ValueError("Model cited a frame it was not shown")
    return report


def context_frames(frames, count):
    if len(frames) <= count:
        return frames
    return [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]


def merge_events(events):
    # Only overlapping same-category events. Adjacent operations remain separate.
    merged = []
    for event in sorted(events, key=lambda e: (e.category.value, e.start_sec)):
        if (
            merged
            and merged[-1].category == event.category
            and event.start_sec <= merged[-1].end_sec
        ):
            previous = merged[-1]
            previous.end_sec = max(previous.end_sec, event.end_sec)
            previous.evidence_timestamps = sorted(
                set(previous.evidence_timestamps + event.evidence_timestamps)
            )[:3]
            previous.observed_action = (
                previous.observed_action + " " + event.observed_action
            )[:1600]
            previous.uncertainty_reason = (
                previous.uncertainty_reason + " " + event.uncertainty_reason
            )[:1600]
        else:
            merged.append(event.model_copy(deep=True))
    return sorted(merged, key=lambda e: e.start_sec)


def analyze(
    job_id, folder: Path, settings, model, created: float, deadline: float, stage
):
    started = time.perf_counter()
    metrics = {"queue_wait_sec": max(0, time.time() - created), "stages": {}}
    model.reset_memory()

    def step(name, fn):
        if time.time() >= deadline:
            raise TimeoutError()
        stage(name)
        tick = time.perf_counter()
        value = fn()
        metrics["stages"][name] = time.perf_counter() - tick
        if time.time() >= deadline:
            raise TimeoutError()
        return value

    path = folder / "source.mov"
    metadata = step("metadata", lambda: probe(path))
    duration = metadata["duration_sec"]
    frames = step(
        "overview_frames",
        lambda: extract(
            path,
            sample_times(duration, settings.overview_fps, settings.max_overview_frames),
            settings.detail_frame_max_pixels
            if settings.qwen_input_mode == "video"
            else settings.frame_max_pixels,
        ),
    )
    all_frames = {f.timestamp: f for f in frames}
    overview = step(
        "overview",
        lambda: validate_report(
            model.infer(
                frames,
                {
                    "phase": "discover",
                    "instruction": "Опиши видимую последовательность всей операции, затем выдели кандидатов для подробной проверки.",
                    "duration_sec": duration,
                    "source_fps": metadata["fps"],
                },
                deadline,
            ),
            duration,
            frames,
        ),
    )
    candidates = merge_events(overview.candidates)
    global_frames = context_frames(frames, settings.context_frame_count)
    audit = None
    if not candidates:
        # An empty first pass cannot bypass the second look. Use fewer, larger
        # images across the WHOLE recording, retaining the initial global summary.
        audit_frames = step(
            "audit_frames",
            lambda: extract(
                path,
                sample_times(duration, 2, settings.audit_frame_count),
                settings.detail_frame_max_pixels,
            ),
        )
        all_frames.update({f.timestamp: f for f in audit_frames})
        audit_input = sorted(
            {f.timestamp: f for f in global_frames + audit_frames}.values(),
            key=lambda f: f.timestamp,
        )
        audit = step(
            "audit",
            lambda: validate_report(
                model.infer(
                    audit_input,
                    {
                        "phase": "audit",
                        "instruction": "Независимая повторная визуальная проверка всей записи. Первый обзор мог пропустить действия. Подробные кадры имеют приоритет. Не добавляй события без видимого основания.",
                        "duration_sec": duration,
                        "source_fps": metadata["fps"],
                        "initial_scene_summary": overview.scene_summary,
                        "initial_observations": overview.observations,
                        "detail_timestamps": [f.timestamp for f in audit_frames],
                    },
                    deadline,
                ),
                duration,
                audit_input,
            ),
        )
        candidates = merge_events(audit.candidates)
    if len(candidates) > settings.max_candidates:
        raise ValueError("Too many candidates for the configured analysis budget")
    final_events = []
    notes = list(overview.review_notes)
    limitations = list(overview.limitations)
    if audit:
        notes.extend(audit.review_notes)
        limitations.extend(audit.limitations)
    candidate_count = len(candidates)
    for index, event in enumerate(candidates):
        lo, hi = max(0, event.start_sec - 2), min(duration - 0.05, event.end_sec + 2)
        count = min(settings.max_detail_frames, max(2, int((hi - lo) * 3) + 1))
        targets = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
        detail = step(
            f"detail_frames_{index + 1}",
            lambda: extract(path, targets, settings.detail_frame_max_pixels),
        )
        all_frames.update({f.timestamp: f for f in detail})
        # Re-send the global sequence as context, plus denser local samples.
        combined = sorted(
            {f.timestamp: f for f in global_frames + detail}.values(),
            key=lambda f: f.timestamp,
        )
        context = {
            "phase": "verify",
            "instruction": "Уточни только этого кандидата. Используй глобальную последовательность как контекст. Если он не подтверждается, events=[].",
            "candidate": event.model_dump(mode="json"),
            "scene_summary": overview.scene_summary,
            "observations": overview.observations,
            "duration_sec": duration,
            "source_fps": metadata["fps"],
            "detail_timestamps": [f.timestamp for f in detail],
        }
        report = step(
            f"detail_{index + 1}",
            lambda: validate_report(
                model.infer(combined, context, deadline), duration, combined
            ),
        )
        for refined in report.events:
            if (
                refined.category != event.category
                or refined.end_sec < lo
                or refined.start_sec > hi
            ):
                raise ValueError("Refinement returned an unrelated event")
        final_events.extend(report.events)
        notes.extend(report.review_notes)
        limitations.extend(report.limitations)
    events = merge_events(final_events)
    violations = []
    for event in events:
        evidence = []
        for timestamp in event.evidence_timestamps:
            nearest = min(all_frames, key=lambda t: abs(t - timestamp))
            frame = all_frames[nearest]
            if (
                not event.start_sec - 0.001 <= nearest <= event.end_sec + 0.001
                or nearest > duration
            ):
                raise ValueError("Invalid evidence timestamp")
            evidence.append(
                EvidenceFrame(
                    timestamp_sec=nearest,
                    image_base64=base64.b64encode(frame.jpeg).decode(),
                )
            )
        description = violation_description(event)
        violations.append(
            Violation(
                type=RULE_BY_ID[event.category.value]["name"],
                timecode_sec=evidence[0].timestamp_sec,
                description=description,
                evidence_frames=evidence,
            )
        )
    # Preview must also stay within the job deadline. A preview failure does not invent a result.
    step(
        "preview",
        lambda: make_preview(path, folder / "preview.mp4", deadline - time.time()),
    )
    result = DoneResult(
        job_id=job_id,
        verdict="violation" if violations else "no_violation",
        violations=violations,
    )
    metrics.update(model.memory())
    metrics["process_peak_rss_bytes"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    metrics.update(
        processing_sec=time.perf_counter() - started,
        total_sec=time.time() - created,
        overview_frame_count=len(frames),
        unique_frame_count=len(all_frames),
        overview_first_timestamp_sec=frames[0].timestamp,
        overview_last_timestamp_sec=frames[-1].timestamp,
        candidate_count=candidate_count,
        negative_audit_performed=audit is not None,
        pipeline_version=(
            "category-screening-v7"
            if settings.qwen_discovery_style == "checklist"
            else "temporal-guidance-v3"
            if settings.qwen_input_mode == "video" or settings.calibration_profile_path
            else "visual-discovery-v2"
        ),
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        adapter_path_configured=bool(settings.lora_path),
        discovery_style=settings.qwen_discovery_style,
    )
    extended = dict(
        candidates=[candidate.model_dump(mode="json") for candidate in candidates],
        events=[e.model_dump(mode="json") for e in events],
        metadata=metadata,
        metrics=metrics,
        scene_summary=(audit.scene_summary if audit else overview.scene_summary),
        observations=list(
            dict.fromkeys(overview.observations + (audit.observations if audit else []))
        ),
        analysis_samples=[
            dict(
                timestamp_sec=f.timestamp,
                image_base64=base64.b64encode(f.jpeg).decode(),
            )
            for f in context_frames(frames, min(3, len(frames)))
        ],
        limitations=list(dict.fromkeys(LIMITATIONS + limitations)),
        review_notes=list(dict.fromkeys(notes)),
        rules=[RULE_BY_ID[e.rule_id] for e in events],
    )
    return result.model_dump(mode="json"), extended
