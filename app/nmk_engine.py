"""Engine adapter: runs the two-pass pipeline (pipeline.py) inside the forte-cash-review service.

Selected with CASH_ENGINE=nmk. The worker process calls NmkModel.load() once and then
analyze() per job; analyze() returns the same (result, extended) pair as app.pipeline.analyze,
with the result in the jury API format (type, timecode_sec, description, evidence_frames).
"""
import base64
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

from PIL import Image

from .schemas import DoneResult, EvidenceFrame, Violation
from .video import extract, make_preview

# Which focused check holds the zoomed view and the observation for each type.
CHECK_FOR_TYPE = {
    "SAFE_LEFT_OPEN": "safe", "SAFE_KEY_LEFT_IN_LOCK": "key", "COMPUTER_NOT_SECURED": "screen",
    "CASH_NOT_SECURED_BEFORE_LEAVING": "cash",
    "DOOR_LEFT_OPEN": "door", "PERSONAL_ITEMS_ON_WORK_DESK": "personal", "CASH_EXPOSED_DURING_COFFEE": "coffee",
    "SEALED_BAG_ACCEPTED_WITHOUT_DOCUMENT": "bag", "SEALED_BAG_CASH_NOT_COUNTED": "bag_open",
    "CASH_COUNTING_WITHOUT_DOCUMENT_CHECK": "doc_check", "CASH_COUNTED_WITHOUT_CLIENT_PRESENT": "no_client",
    "PREVIOUS_CLIENT_CASH_NOT_SECURED": "next_client", "CLIENTS_CASH_MIXED": "next_client",
    "NEXT_CLIENT_STARTED_BEFORE_PREVIOUS_COMPLETED": "next_client",
}
# Only full-sentence observations; raw item lists ("what_on_desk", "desk_items") read as noise.
OBSERVATION_KEYS = ("what_happens", "what_on_screen", "what_near_lock", "difference")
# Short act names for one-line citations; the full title, quote and link are in the rule card.
ACT_SHORT = {"NBRK231": "Правил кассовых операций (пост. НБРК № 231)",
             "NBRK14": "Правил охраны помещений банков (пост. НБРК № 14)",
             "NBRK48": "Требований к ИБ банков (пост. НБРК № 48)"}


class NmkModel:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.processor = None

    def load(self):
        sys.path.insert(0, str(self.settings.nmk_dir))
        import infer_video
        self.model, self.processor = infer_video.load_model(str(self.settings.nmk_model_path),
                                                            self.settings.nmk_precision)

    def reset_memory(self):
        import torch
        torch.cuda.reset_peak_memory_stats()

    def memory(self):
        import torch
        return {"gpu_peak_bytes": torch.cuda.max_memory_allocated()}


def _budget_end(settings, started, deadline, margin=25.0, floor=25.0):
    """Deadline for this job's analysis so that every queued video can still meet its own deadline.

    Jobs run one at a time in submission order. Queued job k (1-based) finishes after this job and
    the k-1 jobs before it, so with an equal share s per job we need started + (k+1)*s <= deadline_k."""
    end = deadline - margin
    try:
        db = sqlite3.connect(settings.data_dir / "jobs.sqlite3", timeout=5)
        rows = db.execute("SELECT deadline FROM jobs WHERE status='processing' AND stage='queued' "
                          "ORDER BY created").fetchall()
        db.close()
    except sqlite3.Error:
        rows = []
    for k, (d,) in enumerate(rows, start=1):
        end = min(end, started + (d - margin - started) / (k + 1))
    return max(end, started + floor)


def _jpeg(img, max_side=960):
    scale = min(1.0, max_side / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _evidence(path, t, duration, boxes):
    """Full frame at t, plus crops (zoomed object, action area) from the same frame."""
    t = max(0.0, min(t, max(0.0, duration - 0.05)))
    frames = extract(path, [t], 4_000_000)
    img = Image.open(io.BytesIO(frames[0].jpeg))
    ts = round(frames[0].timestamp, 2)
    out = [EvidenceFrame(timestamp_sec=ts, image_base64=_jpeg(img))]
    for box in boxes:
        if box and len(out) < 3:
            x0, y0, x1, y1 = [int(v) for v in box]
            if x1 - x0 > 8 and y1 - y0 > 8:
                crop = img.crop((x0, y0, x1, y1))
                crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
                out.append(EvidenceFrame(timestamp_sec=ts, image_base64=_jpeg(crop)))
    return out


def _observation(check):
    ans = (check or {}).get("ans") or {}
    for k in OBSERVATION_KEYS:
        v = ans.get(k)
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        if isinstance(v, str) and len(v.strip()) > 5:
            return v.strip().rstrip(".")
    return None


def _cite(info):
    src = info["sources"][0]
    return f"{src['section']} {ACT_SHORT.get(src['act'], src['act'])}"


def _description(vio, info, acts, check):
    """Two short sentences: what happened, and why it may break the rule (with a short citation)."""
    return f"{vio['description'].rstrip('.')}. {info['why'].rstrip('.')} — {_cite(info)}."


UNCERTAINTY = "Вывод только по видео — подсказка для проверки человеком."


def _load_rules(settings):
    return json.loads((Path(settings.nmk_dir) / "violation_rules.json").read_text(encoding="utf-8"))


def ui_rules(settings):
    """Our violation types in the rule format the forte-cash-review UI renders (/ui/api/rules)."""
    rules = _load_rules(settings)
    out = []
    for code, info in rules["types"].items():
        first = info["sources"][0]
        act = rules["_acts"][first["act"]]
        out.append({
            "id": code,
            "name": info["name"],
            "description": info["why"],
            "applicability_conditions": "",
            "exceptions": "",
            "observable_signs": "",
            "normative_source": {"title": act["title"], "url": act["url"], "section": first["section"]},
            "source_verification_status": "verified",
            "research_leads": [
                {"title": rules["_acts"][s["act"]]["title"], "url": rules["_acts"][s["act"]]["url"],
                 "note": f"{s['section']}: «{s['quote']}»"
                         + ("" if s["link"] == "direct" else " Связь с нарушением выведена из текста нормы.")}
                for s in info["sources"]
            ],
        })
    return out


def analyze(job_id, folder: Path, settings, model, created: float, deadline: float, stage):
    sys.path.insert(0, str(settings.nmk_dir))
    import importlib
    import pipeline
    importlib.reload(pipeline)

    started = time.time()
    path = folder / "source.mov"
    rules = _load_rules(settings)
    stages = {}
    # Keep room for evidence frames; when other videos are waiting, stay short so they all fit.
    budget_end = _budget_end(settings, started, deadline)
    # Re-evaluated at every stage: videos uploaded while this one runs shrink its share.
    pipeline.set_deadline(lambda: _budget_end(settings, started, deadline))
    model.reset_memory()

    stage("starting")
    pipeline.set_progress(stage)
    tick = time.time()
    result, debug = pipeline.analyze(model.model, model.processor, str(path),
                                     {"single_violation": True, "recall_mode": True})
    pipeline.set_deadline(None)
    pipeline.set_progress(None)
    stages["analysis"] = round(time.time() - tick, 1)

    stage("evidence")
    tick = time.time()
    duration = float((debug or {}).get("dur") or 0)
    checks = (debug or {}).get("checks") or {}
    action_box = (debug or {}).get("box")
    evidence_level = {x["type"]: x.get("_ev", 1) for x in (debug or {}).get("all_violations", [])}
    violations, codes, events = [], [], []
    for vio in result["violations"]:
        code = vio["type"] if vio["type"] in rules["types"] else "OTHER_VISIBLE_VIOLATION"
        info = rules["types"][code]
        check = checks.get(CHECK_FOR_TYPE.get(code, ""))
        t = float(vio["timecode_sec"])
        frames = _evidence(path, t, duration or 1e9, [(check or {}).get("zoom_box"), action_box])
        violations.append(Violation(type=info["name"], timecode_sec=round(t, 1),
                                    description=_description(vio, info, rules["_acts"], check),
                                    evidence_frames=frames))
        codes.append(code)
        obs = _observation(check)
        uncertain = UNCERTAINTY + (" Второй просмотр фрагмента признак не подтвердил."
                                   if evidence_level.get(vio["type"]) == 0 else "")
        events.append({
            "category": code, "rule_id": code,
            # a thin marker at the moment itself
            "start_sec": round(max(0.0, t - 0.3), 2), "end_sec": round(min(duration or t + 0.3, t + 0.3), 2),
            "observed_action": obs.rstrip(".") + "." if obs else vio["description"],
            "applicability_basis": f"{info['why'].rstrip('.')} — {_cite(info)}.",
            "uncertainty_reason": uncertain,
            "evidence_timestamps": [f.timestamp_sec for f in frames[:1]],
        })
    stages["evidence"] = round(time.time() - tick, 1)

    # Playable preview for the UI player; never fails the job.
    stage("preview")
    tick = time.time()
    try:
        make_preview(path, folder / "preview.mp4", max(5.0, deadline - time.time() - 3))
    except Exception:  # noqa: BLE001
        pass
    stages["preview"] = round(time.time() - tick, 1)

    done = DoneResult(job_id=job_id, verdict="violation" if violations else "no_violation",
                      violations=violations)
    p1 = (debug or {}).get("pass1") or {}
    extended = {
        "engine": "nmk-two-pass",
        "events": events,
        "metadata": {"duration_sec": round(duration, 2)},
        "scene_summary": p1.get("observations") or "",
        "observations": [p1["observations"]] if p1.get("observations") else [],
        # Compact trace of what each stage saw, for explaining any result (ignored by the UI).
        "diagnostics": {
            "pass1": {k: v for k, v in p1.items() if k != "observations"},
            "leave": (debug or {}).get("leave"),
            "checks": {k: {"t": c.get("t"), "ans": c.get("ans"), "second": c.get("second"), "weak": c.get("_weak")}
                       for k, c in checks.items() if isinstance(c, dict) and "ans" in c},
            "windows": [{"start": w.get("start"), "end": w.get("end"),
                         "seen": (w.get("ans") or {}).get("what_happens"),
                         "true": [k for k, v in (w.get("ans") or {}).items() if v is True]}
                        for w in (debug or {}).get("pass2", [])],
        },
        "review_notes": [],
        "rules": [r for r in ui_rules(settings) if r["id"] in codes],
        "codes": codes,
        "all_candidates": [{"type": x["type"], "timecode_sec": x["timecode_sec"], "evidence": x.get("_ev")}
                           for x in (debug or {}).get("all_violations", [])],
        "observations": p1.get("observations"),
        "error": (debug or {}).get("error"),
        "metrics": {"processing_sec": round(time.time() - started, 1), "queue_wait_sec": round(started - created, 1),
                    "total_sec": round(time.time() - created, 1), "stages": stages,
                    "pipeline_version": "nmk-two-pass", **model.memory()},
        "limitations": [
            "Результат — признаки для проверки человеком, а не юридическое или дисциплинарное заключение.",
            "Мелкие объекты (ключ в замке, отдельные купюры) при низком разрешении видео могут быть не распознаны.",
            "Нарушения, требующие невидимых данных (проводки, сверки, регистрации в системе), не оцениваются.",
        ],
    }
    return done.model_dump(mode="json"), extended
