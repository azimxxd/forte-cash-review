"""Submit local videos to the real API; measure total latency including queue and upload.
No model substitutes. Default: loopback only. Run after GPU startup, never against jury training data.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse
import httpx
from app.video import probe


async def run(args):
    host = urlparse(args.url).hostname
    if host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        raise ValueError(
            "Remote upload disabled. Obtain organizer permission before --allow-remote."
        )
    key = os.environ.get("CASH_API_KEY")
    if not key:
        raise ValueError(
            "Set CASH_API_KEY in the environment; never pass it on the command line"
        )
    paths = sorted(args.videos.glob("*.mov"))
    if args.longest:
        paths = sorted(paths, key=lambda p: probe(p)["duration_sec"], reverse=True)[
            : args.longest
        ]
    if not paths:
        raise ValueError("No .mov files")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.url, headers={"X-API-Key": key}, timeout=30
    ) as client:
        ready = await client.get("/health/ready")
        if ready.status_code != 200:
            raise RuntimeError("Real model is not ready")

        async def one(path, repeat):
            async with semaphore:
                started = time.perf_counter()
                record = {"video_id": path.name, "repeat": repeat}
                try:
                    async with asyncio.timeout(args.timeout):
                        with path.open("rb") as f:
                            response = await client.post(
                                "/api/v1/analyze",
                                files={"file": (path.name, f, "video/quicktime")},
                            )
                        if response.status_code != 202:
                            return {
                                **record,
                                "status": "rejected",
                                "http_status": response.status_code,
                                "total_sec": time.perf_counter() - started,
                            }
                        job_id = response.json()["job_id"]
                        while True:
                            response = await client.get("/api/v1/jobs/" + job_id)
                            response.raise_for_status()
                            result = response.json()
                            if result["status"] != "processing":
                                break
                            await asyncio.sleep(0.5)
                        record.update(result)
                        # Do not duplicate sensitive evidence images into benchmark reports.
                        for event in record.get("violations", []):
                            for frame in event["evidence_frames"]:
                                frame.pop("image_base64", None)
                except TimeoutError:
                    record.update(status="failed", error={"code": "benchmark_timeout"})
                except Exception as e:
                    record.update(status="failed", error={"code": type(e).__name__})
                record["total_sec"] = time.perf_counter() - started
                return record

        records = await asyncio.gather(
            *(one(path, i) for i in range(args.repeat) for path in paths)
        )
    times = sorted(r["total_sec"] for r in records)
    summary = {
        "requests": len(records),
        "concurrency": args.concurrency,
        "done": sum(r["status"] == "done" for r in records),
        "failed": sum(r["status"] == "failed" for r in records),
        "rejected": sum(r["status"] == "rejected" for r in records),
        "over_300_sec": sum(r["total_sec"] > 300 for r in records),
        "latency_p50_sec": statistics.median(times),
        "latency_p95_sec": times[min(len(times) - 1, int(len(times) * 0.95))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"summary": summary, "results": records}, ensure_ascii=False, indent=2
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    p.add_argument("--repeat", type=int, choices=range(1, 11), default=1)
    p.add_argument("--longest", type=int, default=0)
    p.add_argument("--timeout", type=float, default=305)
    p.add_argument("--allow-remote", action="store_true")
    asyncio.run(run(p.parse_args()))
