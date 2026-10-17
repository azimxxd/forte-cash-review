from fractions import Fraction
import av
import pytest
from PIL import Image
from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        api_key="test-api-key-0000000000000000",
        ui_password="test-ui-password-00000000000",
        data_dir=tmp_path / "data",
        frontend_dir=tmp_path / "dist",
    )


@pytest.fixture
def synthetic_mov(tmp_path):
    # Explicit test-only generated source; no user dataset is decoded by tests.
    path = tmp_path / "synthetic.mov"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"
        for i in range(60):
            picture = Image.new("RGB", (320, 240), (i * 4, 40, 220 - i * 3))
            frame = av.VideoFrame.from_image(picture)
            frame.pts = i
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
