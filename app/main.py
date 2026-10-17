import base64
import binascii
import asyncio
import fcntl
import os
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.concurrency import run_in_threadpool
from .config import Settings
from .rules import RULES
from .store import Store, QueueFull
from .worker import Worker


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise HTTPException(413, "Файл превышает допустимый размер")
            return message

        headers = dict(scope.get("headers", []))
        try:
            too_large = int(headers.get(b"content-length", b"0")) > self.limit
        except ValueError:
            too_large = True
        if too_large:
            return await JSONResponse(
                {"detail": "Файл превышает допустимый размер"}, status_code=413
            )(scope, receive, send)
        await self.app(scope, bounded_receive, send)


def create_app(settings=None, worker_factory=Worker):
    settings = settings or Settings()
    os.umask(0o077)
    settings.data_dir = settings.data_dir.resolve()
    store = Store(settings.data_dir)
    worker = worker_factory(settings, store)

    @asynccontextmanager
    async def lifespan(app):
        lock = (settings.data_dir / "worker.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(
                "Only one server process may use this data directory. Use --workers 1."
            )
        worker.start()
        try:
            yield
        finally:
            await run_in_threadpool(worker.close)
            lock.close()

    app = FastAPI(
        title="Forte Cash Review",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store, app.state.worker = store, worker
    app.add_middleware(
        BodyLimit, limit=(settings.max_upload_mb * 1024 * 1024) + 1024 * 1024
    )
    basic = HTTPBasic(auto_error=False)

    def api_auth(x_api_key: str = Header(default="")):
        if not secrets.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Неверный API-ключ")

    def ui_auth(credentials: HTTPBasicCredentials | None = Depends(basic)):
        if credentials is None or not (
            secrets.compare_digest(
                credentials.username.encode(), settings.ui_user.encode()
            )
            and secrets.compare_digest(
                credentials.password.encode(), settings.ui_password.encode()
            )
        ):
            raise HTTPException(
                401,
                "Требуется доступ проверяющего",
                headers={
                    "WWW-Authenticate": 'Basic realm="Forte Review", charset="UTF-8"'
                },
            )

    def ui_write(request: Request, _: None = Depends(ui_auth)):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Запрос с другого сайта запрещён")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Запрос с другого сайта запрещён")

    @app.middleware("http")
    async def headers(request, call_next):
        request.state.started = time.time()
        # Authenticate before multipart parsing, so unauthorized uploads are not spooled.
        if request.url.path.startswith("/api/v1/"):
            try:
                api_auth(request.headers.get("x-api-key", ""))
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        elif not request.url.path.startswith("/health/"):
            try:
                scheme, token = request.headers.get("authorization", "").split(" ", 1)
                if scheme.lower() != "basic":
                    raise ValueError()
                username, password = (
                    base64.b64decode(token, validate=True).decode("utf-8").split(":", 1)
                )
                ui_auth(HTTPBasicCredentials(username=username, password=password))
            except (ValueError, UnicodeDecodeError, binascii.Error, HTTPException):
                return JSONResponse(
                    {"detail": "Требуется доступ проверяющего"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="Forte Review", charset="UTF-8"'
                    },
                )
        if request.method == "POST" and request.url.path in (
            "/api/v1/analyze",
            "/ui/api/analyze",
        ):
            try:
                async with asyncio.timeout(settings.job_timeout_sec):
                    response = await call_next(request)
            except TimeoutError:
                response = JSONResponse(
                    {"detail": "Истекло время загрузки"}, status_code=408
                )
        else:
            response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'",
            }
        )
        return response

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        return JSONResponse(
            {"ready": worker.ready, "model_state": worker.state},
            status_code=200 if worker.ready else 503,
        )

    def job_or_404(job_id):
        try:
            if str(UUID(job_id)) != job_id:
                raise ValueError()
        except ValueError:
            raise HTTPException(404, "Задание не найдено")
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Задание не найдено")
        return job

    async def upload(request, file):
        job_id = None
        folder = None
        try:
            if not worker.ready:
                raise HTTPException(
                    503, "Модель не готова. Проверьте локальные веса и GPU."
                )
            if Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(415, "Выберите файл .mov")
            job_id = str(uuid4())
            created = request.state.started
            try:
                store.reserve(
                    job_id,
                    Path(file.filename).name,
                    created,
                    created + settings.job_timeout_sec,
                    settings.queue_capacity,
                )
            except QueueFull:
                raise HTTPException(
                    429,
                    "Очередь заполнена. Повторите позже.",
                    headers={"Retry-After": "15"},
                )
            folder = settings.data_dir / job_id
            folder.mkdir(mode=0o700)
            size = 0
            with (folder / "source.mov").open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "Файл слишком большой")
                    if time.time() >= created + settings.job_timeout_sec:
                        raise HTTPException(408, "Истекло время загрузки")
                    await run_in_threadpool(target.write, chunk)
            if not size:
                raise HTTPException(400, "Пустой файл")
            store.stage(job_id, "queued")
            return {"job_id": job_id}
        except BaseException:
            if job_id and store.get(job_id):
                store.fail(job_id, "upload_failed", "Не удалось загрузить файл")
            if folder:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await file.close()

    @app.post("/api/v1/analyze", dependencies=[Depends(api_auth)], status_code=202)
    async def api_upload(request: Request, file: UploadFile = File(...)):
        return await upload(request, file)

    @app.post("/ui/api/analyze", dependencies=[Depends(ui_write)], status_code=202)
    async def ui_upload(request: Request, file: UploadFile = File(...)):
        return await upload(request, file)

    def status(job):
        if job["status"] == "done":
            return job["result"]
        result = {"job_id": job["id"], "status": job["status"]}
        if job["error"]:
            result["error"] = job["error"]
        return result

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(api_auth)])
    def api_job(job_id: str):
        return status(job_or_404(job_id))

    @app.get("/ui/api/jobs/{job_id}", dependencies=[Depends(ui_auth)])
    def ui_job(job_id: str):
        job = job_or_404(job_id)
        return {
            **status(job),
            "stage": job["stage"],
            "filename": job["filename"],
            "created_at": job["created"],
            "extended": job["extended"],
            "video_url": f"/ui/api/jobs/{job_id}/video"
            if job["status"] == "done"
            else None,
        }

    @app.get("/ui/api/jobs/{job_id}/video", dependencies=[Depends(ui_auth)])
    def video(job_id: str):
        job = job_or_404(job_id)
        path = settings.data_dir / job_id / "preview.mp4"
        if job["status"] != "done" or not path.is_file():
            raise HTTPException(404, "Видео ещё не готово")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/ui/api/rules", dependencies=[Depends(ui_auth)])
    def rules():
        return RULES

    @app.get("/ui/api/health", dependencies=[Depends(ui_auth)])
    def ui_health():
        return {
            "ready": worker.ready,
            "model_state": worker.state,
            "max_upload_mb": settings.max_upload_mb,
            "model": "Qwen3-VL-4B-Instruct",
            "queue_capacity": settings.queue_capacity,
        }

    @app.get("/{path:path}", dependencies=[Depends(ui_auth)])
    def frontend(path: str):
        if path.startswith(("api/", "ui/", "health/")):
            raise HTTPException(404)
        root = settings.frontend_dir.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise HTTPException(404)
        if path and target.is_file():
            return FileResponse(target)
        if not path and (root / "index.html").exists():
            return FileResponse(root / "index.html")
        raise HTTPException(404, "Соберите frontend: npm ci && npm run build")

    return app
