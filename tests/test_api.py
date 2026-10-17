import time
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from app.main import create_app


class ReadyTestWorker:
    """TEST ONLY: queue remains pending to test HTTP contracts independently of GPU."""

    def __init__(self, settings, store):
        self.ready = True
        self.state = "test_only"
        self.store = store

    def start(self):
        self.store.recover()

    def close(self):
        pass


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, worker_factory=ReadyTestWorker)) as c:
        yield c


def test_api_key_required_even_with_ui_password(client, settings):
    path = "/api/v1/jobs/" + str(uuid4())
    assert client.get(path).status_code == 401
    assert (
        client.get(path, auth=(settings.ui_user, settings.ui_password)).status_code
        == 401
    )
    assert (
        client.post("/api/v1/analyze", files={"file": ("a.mov", b"test")}).status_code
        == 401
    )
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get(path, headers={"X-API-Key": settings.api_key}).status_code == 404


def test_upload_async_contract_and_queue(client, settings, synthetic_mov):
    headers = {"X-API-Key": settings.api_key}
    response = client.post(
        "/api/v1/analyze",
        headers=headers,
        files={"file": ("Х-test.mov", synthetic_mov.read_bytes(), "video/quicktime")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get("/api/v1/jobs/" + job_id, headers=headers)
    assert status.json() == {"job_id": job_id, "status": "processing"}
    for _ in range(2):
        assert (
            client.post(
                "/api/v1/analyze", headers=headers, files={"file": ("a.mov", b"a")}
            ).status_code
            == 202
        )
    assert (
        client.post(
            "/api/v1/analyze", headers=headers, files={"file": ("a.mov", b"a")}
        ).status_code
        == 429
    )


def test_failed_job_has_no_verdict(client, settings):
    store = client.app.state.store
    job_id = str(uuid4())
    store.reserve(job_id, "bad.mov", time.time(), time.time() + 20, 3)
    store.fail(job_id, "decode_error", "Ошибка декодирования")
    data = client.get(
        "/api/v1/jobs/" + job_id, headers={"X-API-Key": settings.api_key}
    ).json()
    assert (
        data["status"] == "failed"
        and "verdict" not in data
        and "violations" not in data
    )


def test_ui_protects_video_rules_and_results(client, settings):
    auth = (settings.ui_user, settings.ui_password)
    for path in ["/", "/ui/api/rules", "/ui/api/jobs/" + str(uuid4()) + "/video"]:
        assert client.get(path).status_code == 401
    assert client.get("/ui/api/rules", auth=auth).status_code == 200
    assert (
        client.post(
            "/ui/api/analyze",
            auth=auth,
            headers={"Origin": "https://evil.example"},
            files={"file": ("a.mov", b"a")},
        ).status_code
        == 403
    )
    assert client.get("/runtime/jobs.sqlite3", auth=auth).status_code == 404
    assert client.get("/ui/api/jobs/../../etc/passwd", auth=auth).status_code == 404


def test_unavailable_model_never_fakes_result(settings, synthetic_mov):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 503
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": settings.api_key},
            files={"file": ("a.mov", synthetic_mov.read_bytes())},
        )
        assert response.status_code == 503
        assert "verdict" not in response.json()


def test_size_type_empty_and_invalid_id(client, settings):
    headers = {"X-API-Key": settings.api_key}
    assert (
        client.post(
            "/api/v1/analyze", headers=headers, files={"file": ("a.txt", b"a")}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/v1/analyze", headers=headers, files={"file": ("a.mov", b"")}
        ).status_code
        == 400
    )
    assert client.get("/api/v1/jobs/not-a-uuid", headers=headers).status_code == 404
    assert (
        client.post(
            "/api/v1/analyze",
            headers=headers | {"Content-Length": str(1024**3)},
            content=b"x",
        ).status_code
        == 413
    )


def test_slow_streaming_upload_is_bounded(settings):
    import asyncio
    import httpx

    settings.job_timeout_sec = 0.05
    app = create_app(settings, worker_factory=ReadyTestWorker)

    async def run():
        async def slow_body():
            yield b"--boundary\r\n"
            await asyncio.sleep(1)
            yield b"--boundary--\r\n"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/v1/analyze",
                headers={
                    "X-API-Key": settings.api_key,
                    "Content-Type": "multipart/form-data; boundary=boundary",
                },
                content=slow_body(),
            )
            assert response.status_code == 408

    asyncio.run(run())
