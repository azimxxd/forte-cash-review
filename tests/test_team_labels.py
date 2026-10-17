"""Synthetic text only. No private teammate workbook content in tests."""

import pytest
from app.team_labels import import_rows, point_seconds, summarize

BASELINE = [
    dict(video_id="X-1.mov", binary_label=1, timecode_sec=18.0, duration_sec=25.6),
    dict(video_id="Х-1.mov", binary_label=0, timecode_sec=None, duration_sec=25.6),
]


def test_preserves_teammate_description_without_inventing_labels():
    text = "Авторская тестовая аннотация для проверки импорта."
    rows = [("X-1", "18 секунда", text), ("Х-1", None, "без нарушения")]
    records = import_rows(rows, BASELINE, "test.xlsx", "test-hash", "labels")
    assert records[0]["team_annotation"] == text
    assert records[0]["reference_timecode_sec"] == 18
    assert records[0]["category"] is None and records[0]["end_sec"] is None
    assert records[0]["start_sec"] is None and records[0]["scene_id"] is None
    assert records[0]["observed_action"] is None
    assert records[0]["provenance"]["annotation_cell"] == "C2"
    assert records[0]["ready_for_event_training"] is False
    assert records[1]["video_id"] == "Х-1.mov"
    assert summarize(records)["positive_with_description"] == 1
    assert summarize(records)["ready_for_event_training_count"] == 0


@pytest.mark.parametrize(
    "value,expected",
    [
        (18, 18),
        ("18 секунда", 18),
        ("1минута 8 секунда", 68),
        ("01:08", 68),
        (None, None),
    ],
)
def test_reference_time_parsing(value, expected):
    assert point_seconds(value) == expected


@pytest.mark.parametrize("value", ["18-22 сек", float("nan"), -1, "неизвестно"])
def test_rejects_unsupported_or_invalid_time(value):
    with pytest.raises(ValueError):
        point_seconds(value)


def test_no_silent_unicode_name_normalization_or_duplicate_rows():
    with pytest.raises(ValueError, match="duplicate_video_id"):
        import_rows(
            [("X-1", 18, "тест"), ("X-1", 18, "тест")],
            BASELINE,
            "test",
            "hash",
            "labels",
        )
    with pytest.raises(ValueError, match="no_exact_filename_match"):
        import_rows([("Х-2", 18, "тест")], BASELINE, "test", "hash", "labels")


def test_rejects_label_conflict_and_out_of_range_time():
    with pytest.raises(ValueError, match="binary_label_conflict"):
        import_rows([("X-1", 18, "без нарушения")], BASELINE, "test", "hash", "labels")
    with pytest.raises(ValueError, match="timecode_outside_video"):
        import_rows(
            [("X-1", 50, "Тестовая аннотация")], BASELINE, "test", "hash", "labels"
        )


def test_point_time_does_not_become_zero_length_interval():
    record = import_rows(
        [("X-1", 18, "Тестовая аннотация")], BASELINE, "test", "hash", "labels"
    )[0]
    assert record["reference_timecode_sec"] == 18
    assert record["start_sec"] is None and record["end_sec"] is None
