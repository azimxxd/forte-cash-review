"""Local FFmpeg decoding through PyAV. Never makes network requests."""

import io
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
import av
from PIL import Image


@dataclass
class Frame:
    timestamp: float
    jpeg: bytes

    def image(self):
        return Image.open(io.BytesIO(self.jpeg)).convert("RGB")


def probe(path: Path):
    with av.open(
        str(path),
        format="mov",
        options={"enable_drefs": "0", "protocol_whitelist": "file,pipe"},
    ) as container:
        if not container.streams.video:
            raise ValueError("В файле нет видеопотока")
        stream = container.streams.video[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else (container.duration or 0) / av.time_base
        )
        fps = float(stream.average_rate or 0)
        if not math.isfinite(duration) or not 1 <= duration <= 145.5:
            raise ValueError("Поддерживаются видео длительностью от 1 до 145 секунд")
        if not 0 < stream.width * stream.height <= 3840 * 2160:
            raise ValueError("Неподдерживаемое разрешение видео")
        return dict(
            duration_sec=duration,
            fps=fps,
            width=stream.width,
            height=stream.height,
            codec=stream.codec_context.name,
        )


def sample_times(duration: float, fps: float, cap: int):
    # Cover the entire timeline, including the final part, even when the cap binds.
    count = min(cap, max(2, math.ceil(duration * fps)))
    end = max(0, duration - 0.05)
    return [i * end / (count - 1) for i in range(count)]


def extract(path: Path, targets: list[float], max_pixels: int) -> list[Frame]:
    if not targets:
        return []
    targets = sorted(set(targets))
    if targets[0] < 0 or not all(math.isfinite(t) for t in targets):
        raise ValueError("Invalid extraction timestamp")
    result = []
    index = 0
    with av.open(
        str(path),
        format="mov",
        options={"enable_drefs": "0", "protocol_whitelist": "file,pipe"},
    ) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        last = None

        def encode(frame, timestamp):
            picture = frame.to_image().convert("RGB")
            scale = min(1.0, math.sqrt(max_pixels / (picture.width * picture.height)))
            if scale < 1:
                picture = picture.resize(
                    (
                        max(1, int(picture.width * scale)),
                        max(1, int(picture.height * scale)),
                    )
                )
            output = io.BytesIO()
            picture.save(output, format="JPEG", quality=90)
            return Frame(round(max(0, timestamp), 6), output.getvalue())

        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame lacks presentation timestamp")
            timestamp = float(frame.pts * stream.time_base) - origin
            last = (frame, timestamp)
            if index < len(targets) and timestamp + 1e-6 >= targets[index]:
                result.append(encode(frame, timestamp))
                while index < len(targets) and targets[index] <= timestamp + 1e-6:
                    index += 1
            if index == len(targets):
                break
        if index < len(targets) and last:
            # Only permit the final target within one frame of the real end.
            tolerance = 1 / float(stream.average_rate or 30) + 0.01
            if targets[-1] - last[1] > tolerance:
                raise ValueError("Video ended before requested timestamps")
            if not result or result[-1].timestamp != round(last[1], 6):
                result.append(encode(*last))
        if not result:
            raise ValueError("No frames decoded")
    return result


def make_preview(source: Path, destination: Path, timeout: float):
    import imageio_ffmpeg

    # Original retained. Preview is only a browser-compatible derivative, never evidence.
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "mov",
            "-enable_drefs",
            "0",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        capture_output=True,
        timeout=max(0.1, timeout),
    )
