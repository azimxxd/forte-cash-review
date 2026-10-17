"""Explicit test-only subprocesses: verify hard deadline enforcement without downloading weights."""

import multiprocessing as mp
import threading
import time
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import create_app
from app.pipeline import analyze
from app.store import Store
from app.worker import Worker
from test_core import TestOnlyModel


def hanging_child(connection):
    connection.send(("ready", None))
    connection.recv()
    time.sleep(60)


class HangingTestWorker(Worker):
    def __init__(self, *args):
        super().__init__(*args)
        self.spawns = 0
        self.pids = []

    def spawn(self):
        self.ready = False
        self.state = "loading"
        self.spawns += 1
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=hanging_child, args=(child,), daemon=True)
        self.process.start()
        self.pids.append(self.process.pid)
        child.close()


def wait_until(predicate, timeout=5):
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.02)


def test_hard_timeout_reloads_worker_and_never_returns_negative(settings):
    store = Store(settings.data_dir)
    settings.model_enabled = True
    worker = HangingTestWorker(settings, store)
    worker.start()
    try:
        wait_until(lambda: worker.ready)
        job_id = str(uuid4())
        store.reserve(job_id, "synthetic.mov", time.time(), time.time() + 0.5, 1)
        store.stage(job_id, "queued")
        wait_until(lambda: store.get(job_id)["status"] == "failed")
        wait_until(lambda: worker.spawns == 2)
        result = store.get(job_id)
        assert result["error"]["code"] == "timeout" and result["result"] is None
        assert worker.pids[0] != worker.pids[1]
    finally:
        worker.close()


class PipelineTestWorker:
    """TEST ONLY: executes real local decoding/pipeline with a deterministic test model."""

    def __init__(self, settings, store):
        self.settings = settings
        self.store = store
        self.ready = True
        self.state = "test_only"
        self.stopping = threading.Event()

    def start(self):
        self.store.recover()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stopping.wait(0.02):
            job = self.store.next_job()
            if not job:
                continue
            try:
                result, extended = analyze(
                    job["id"],
                    self.settings.data_dir / job["id"],
                    self.settings,
                    TestOnlyModel(),
                    job["created"],
                    job["deadline"],
                    lambda s: self.store.stage(job["id"], s),
                )
                self.store.complete(job["id"], result, extended)
            except Exception:
                self.store.fail(
                    job["id"],
                    "test_pipeline_error",
                    "Тестовый сбой декодирования или анализа",
                )

    def close(self):
        self.stopping.set()
        self.thread.join(5)


def test_http_to_pipeline_to_evidence_and_protected_range_video(
    settings, synthetic_mov
):
    with TestClient(create_app(settings, worker_factory=PipelineTestWorker)) as client:
        headers = {"X-API-Key": settings.api_key}
        response = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("x.mov", synthetic_mov.read_bytes())},
        )
        job_id = response.json()["job_id"]
        store = client.app.state.store
        wait_until(lambda: store.get(job_id)["status"] != "processing")
        result = client.get("/api/v1/jobs/" + job_id, headers=headers).json()
        assert result["status"] == "done" and result["verdict"] == "violation"
        assert set(result) == {"job_id", "status", "verdict", "violations"}
        assert len(result["violations"][0]["evidence_frames"]) >= 1
        video = "/ui/api/jobs/" + job_id + "/video"
        assert client.get(video).status_code == 401
        response = client.get(
            video,
            auth=(settings.ui_user, settings.ui_password),
            headers={"Range": "bytes=0-31"},
        )
        assert response.status_code == 206 and len(response.content) == 32
        assert response.headers["cache-control"] == "no-store"
        bad = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("bad.mov", b"not a movie")},
        ).json()["job_id"]
        wait_until(lambda: store.get(bad)["status"] == "failed")
        result = client.get("/api/v1/jobs/" + bad, headers=headers).json()
        assert "verdict" not in result
