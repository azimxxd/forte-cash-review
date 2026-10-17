import pytest
from pydantic import ValidationError
from app.calibration import CalibrationProfile


def test_guidance_contains_patterns_without_per_video_answers():
    profile = CalibrationProfile(
        patterns=[
            dict(
                category="leave_workplace_during_service",
                observable_pattern="Рабочее место покидается в продолжающейся последовательности обслуживания.",
                uncertainty_cautions="Нужно отличить уход из рабочей зоны от выхода за границу кадра.",
            )
        ]
    )
    assert profile.patterns[0].category.value == "leave_workplace_during_service"


@pytest.mark.parametrize(
    "key", ["X-26.mov", "Х-26", "01:18", "video_id", "binary_label"]
)
def test_rejects_answer_key_in_runtime_guidance(key):
    with pytest.raises(ValidationError):
        CalibrationProfile(
            patterns=[
                dict(
                    category="leave_workplace_during_service",
                    observable_pattern="Специфичный ответ для " + key,
                    uncertainty_cautions="Тестовые ограничения для импорта.",
                )
            ]
        )
