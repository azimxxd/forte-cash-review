"""Integration smoke test using ONLY a generated color video and the real model.
Does not read any dataset files. Model output/evidence is not printed.
"""
import argparse
import io
import json
import time
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import av
from PIL import Image
from dotenv import dotenv_values


def request(url, key, body=None, content_type=None):
    headers = {"X-API-Key": key}
    if content_type:
        headers["Content-Type"] = content_type
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=20) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--env-file", type=Path, default=Path(".env.runpod"))
    parser.add_argument("--output", type=Path, default=Path("reports/gpu-smoke.json"))
    args = parser.parse_args()
    key = dotenv_values(args.env_file)["CASH_API_KEY"]
    code, ready = request(args.url + "/health/ready", key)
    if code != 200 or not ready.get("ready"):
        raise RuntimeError("Real model is not ready: " + json.dumps(ready))
    with TemporaryDirectory(prefix="forte-synthetic-smoke-") as root:
        path = Path(root) / "synthetic-color.mov"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
            for index in range(90):
                image = Image.new("RGB", (320, 240), (20, 40 + index // 3, 100))
                frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = index, Fraction(1, 30)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        boundary = "forte-synthetic-smoke-boundary"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic-color.mov"\r\n'
            "Content-Type: video/quicktime\r\n\r\n"
        ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        started = time.monotonic()
        code, job = request(args.url + "/api/v1/analyze", key, body, "multipart/form-data; boundary=" + boundary)
        if code != 202:
            raise RuntimeError("Upload failed: " + json.dumps(job, ensure_ascii=False))
        job_id = job["job_id"]
        while time.monotonic() - started < 305:
            code, result = request(args.url + "/api/v1/jobs/" + job_id, key)
            if code != 200:
                raise RuntimeError("Polling failed")
            if result["status"] != "processing":
                break
            time.sleep(1)
        else:
            raise TimeoutError("Smoke test exceeded 305 seconds")
    summary = {
        "test": "synthetic_color_video_real_qwen",
        "job_id": job_id,
        "status": result["status"],
        "wall_time_sec": round(time.monotonic() - started, 3),
        "verdict": result.get("verdict"),
        "event_count": len(result.get("violations", [])),
        "error": result.get("error"),
        "note": "Infrastructure check only; not a measure of bank-video detection quality or long-video latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result["status"] != "done":
        raise RuntimeError("Real inference smoke failed")


if __name__ == "__main__":
    main()
