"""Optional CPU tests of the REAL Qwen processor; public tokenizer files only, no model weights.
Offline: skips if the manually prepared processor-only snapshot is absent.
"""

import io
import os
from pathlib import Path
import pytest
from PIL import Image
from app.model import QwenAdapter
from app.video import Frame

SNAPSHOT = Path(__file__).resolve().parents[1] / "models/processor-smoke"
pytestmark = pytest.mark.skipif(
    not (SNAPSHOT / "tokenizer.json").exists(),
    reason="Optional public processor snapshot not prepared",
)


@pytest.fixture
def adapter(settings, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    from transformers import AutoProcessor

    adapter = QwenAdapter(settings)
    adapter.processor = AutoProcessor.from_pretrained(
        str(SNAPSHOT), local_files_only=True, trust_remote_code=False
    )
    return adapter


def synthetic_frame(timestamp=0):
    output = io.BytesIO()
    Image.new("RGB", (640, 360), "navy").save(output, format="JPEG")
    return Frame(timestamp, output.getvalue())


def test_real_processor_timestamped_images(adapter):
    inputs = adapter.prepare_inputs(
        [synthetic_frame(), synthetic_frame(1.5)], "Проверка только подготовки входа"
    )
    assert inputs["image_grid_thw"].shape[0] == 2
    assert inputs["input_ids"].shape[-1] > 0
    text = adapter.processor.tokenizer.decode(inputs["input_ids"][0])
    assert "t=1.500000" in text
    assert (
        inputs["image_grid_thw"].prod(dim=1).sum().item() // 4
        <= adapter.settings.max_visual_tokens
    )


def test_dense_refinement_stays_within_visual_budget(adapter):
    inputs = adapter.prepare_inputs(
        [synthetic_frame(i) for i in range(169)],
        "Синтетический тест максимального числа кадров",
    )
    actual = (
        int(inputs["image_grid_thw"].prod(dim=1).sum())
        // adapter.processor.image_processor.merge_size**2
    )
    assert inputs["image_grid_thw"].shape[0] == 169
    assert actual <= adapter.settings.max_visual_tokens
    assert inputs["input_ids"].shape[-1] <= adapter.settings.max_input_tokens


def test_detail_images_receive_more_tokens_without_losing_global_context(adapter):
    adapter.settings.max_visual_tokens = 12000
    frames = [synthetic_frame(i) for i in range(48)]
    inputs = adapter.prepare_inputs(
        frames,
        {"phase": "verify", "detail_timestamps": [f.timestamp for f in frames[24:]]},
    )
    tokens = (
        inputs["image_grid_thw"].prod(dim=1)
        // adapter.processor.image_processor.merge_size**2
    )
    assert len(tokens) == 48
    assert tokens[24:].float().mean() > tokens[:24].float().mean()
    assert int(tokens.sum()) <= 12000
    assert adapter._prepared_metrics["image_count"] == 48
