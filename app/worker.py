"""One persistent GPU process, supervised with a hard end-to-end deadline."""

import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from .model import QwenAdapter
from .pipeline import analyze

log = logging.getLogger(__name__)


def child_main(connection, settings):
    try:
        model = QwenAdapter(settings)
        model.load()
        connection.send(("ready", None))
    except Exception as exc:
        log.error("Model startup failed (%s)", type(exc).__name__)
        connection.send(("load_failed", type(exc).__name__))
        return
    while True:
        task = connection.recv()
        if task is None:
            return
        try:
            job_id, created, deadline = task
            result = analyze(
                job_id,
                settings.data_dir / job_id,
                settings,
                model,
                created,
                deadline,
                lambda name: connection.send(("stage", name)),
            )
            connection.send(("done", result))
        except Exception as exc:
            # Never log video-derived model responses or local frame contents.
            log.error("Analysis failed (%s)", type(exc).__name__)
            connection.send(
                (
                    "failed",
                    "timeout" if isinstance(exc, TimeoutError) else "analysis_error",
                )
            )


class Worker:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.ready = False
        self.state = "disabled" if not settings.model_enabled else "loading"
        self.active = None
        self.process = None
        self.connection = None
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        self.store.recover()
        if self.settings.model_enabled:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def spawn(self):
        self.ready = False
        self.state = "loading"
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(
            target=child_main, args=(child, self.settings), daemon=True
        )
        self.process.start()
        child.close()

    def kill(self):
        self.ready = False
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2)
            self.process.close()
            self.process = None
        if self.connection:
            self.connection.close()
            self.connection = None

    def run(self):
        self.spawn()
        try:
            while not self.stopping.wait(0.1):
                self.store.expire()
                if self.active and time.time() >= self.active["deadline"]:
                    self.store.fail(
                        self.active["id"],
                        "timeout",
                        "Истекло время обработки. Результат не получен.",
                    )
                    self.active = None
                    self.kill()
                    self.spawn()
                    continue
                if self.connection and self.connection.poll():
                    try:
                        kind, payload = self.connection.recv()
                    except EOFError:
                        kind, payload = "crashed", None
                    if kind == "ready":
                        self.ready, self.state = True, "ready"
                    elif kind in ("load_failed", "crashed"):
                        if self.active:
                            self.store.fail(
                                self.active["id"],
                                "worker_crashed",
                                "Процесс анализа завершился с ошибкой.",
                            )
                        self.active = None
                        self.kill()
                        self.state = (
                            "load_failed" if kind == "load_failed" else "worker_crashed"
                        )
                        # Require restart after a load/crash error; no busy retry loop.
                        return
                    elif self.active:
                        job_id = self.active["id"]
                        if kind == "stage":
                            self.store.stage(job_id, payload)
                        elif kind == "done":
                            if time.time() < self.active["deadline"]:
                                self.store.complete(job_id, *payload)
                            else:
                                self.store.fail(
                                    job_id, "timeout", "Истекло время обработки."
                                )
                            self.active = None
                        elif kind == "failed":
                            self.store.fail(
                                job_id,
                                payload,
                                "Анализ не завершён. Проверьте видео и журнал сервиса.",
                            )
                            self.active = None
                if self.ready and not self.active:
                    job = self.store.next_job()
                    if job:
                        self.active = job
                        self.store.stage(job["id"], "starting")
                        self.connection.send(
                            (job["id"], job["created"], job["deadline"])
                        )
        except Exception:
            log.exception("Supervisor failure")
            self.ready, self.state = False, "worker_crashed"
            if self.active:
                self.store.fail(
                    self.active["id"], "worker_crashed", "Процесс анализа недоступен."
                )
        finally:
            self.kill()

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(6)
        if self.active:
            self.store.fail(
                self.active["id"],
                "service_stopped",
                "Сервис остановлен во время обработки.",
            )
        self.kill()
