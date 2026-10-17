import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class QueueFull(Exception):
    pass


class Store:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                created REAL NOT NULL, deadline REAL NOT NULL, filename TEXT NOT NULL,
                result TEXT, extended TEXT, error TEXT)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, job_id, filename, created, deadline, capacity):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM jobs WHERE status='processing'"
            ).fetchone()[0]
            if count >= capacity:
                raise QueueFull()
            db.execute(
                "INSERT INTO jobs(id,status,stage,created,deadline,filename) VALUES (?,'processing','uploading',?,?,?)",
                (job_id, created, deadline, filename),
            )

    def stage(self, job_id, stage):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET stage=? WHERE id=? AND status='processing'",
                (stage, job_id),
            )

    def fail(self, job_id, code, message):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage='failed',error=? WHERE id=? AND status='processing'",
                (
                    json.dumps({"code": code, "message": message}, ensure_ascii=False),
                    job_id,
                ),
            )

    def complete(self, job_id, result, extended):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='done',stage='done',result=?,extended=? WHERE id=? AND status='processing'",
                (
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(extended, ensure_ascii=False),
                    job_id,
                ),
            )

    def recover(self):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage='failed',error=? WHERE status='processing'",
                (
                    json.dumps(
                        {
                            "code": "service_restarted",
                            "message": "Сервис перезапущен. Загрузите видео повторно.",
                        }
                    ),
                ),
            )

    def next_job(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE status='processing' AND stage='queued' ORDER BY created LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def expire(self):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage='failed',error=? WHERE status='processing' AND deadline<?",
                (
                    json.dumps(
                        {
                            "code": "timeout",
                            "message": "Превышено время обработки, включая ожидание в очереди.",
                        }
                    ),
                    time.time(),
                ),
            )

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        for key in ("result", "extended", "error"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result
