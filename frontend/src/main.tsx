import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  Check,
  CircleAlert,
  Clock3,
  FileVideo,
  Film,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  ScanLine,
  Upload,
  X,
} from "lucide-react";
import "@fontsource-variable/open-sans";
import "./style.css";

type Rule = {
  id: string;
  name: string;
  description: string;
  applicability_conditions: string;
  exceptions: string;
  observable_signs: string;
  normative_source: null | { title: string; url: string; section?: string };
  source_verification_status: string;
  research_leads: { title: string; url: string; note: string }[];
};
type Event = {
  category: string;
  start_sec: number;
  end_sec: number;
  observed_action: string;
  rule_id: string;
  uncertainty_reason: string;
  applicability_basis: string;
};
type Violation = {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: { timestamp_sec: number; image_base64: string }[];
};
type Job = {
  job_id: string;
  status: "processing" | "done" | "failed";
  stage: string;
  filename: string;
  created_at: number;
  verdict?: string;
  violations?: Violation[];
  error?: { message: string };
  video_url?: string;
  extended?: {
    events: Event[];
    limitations: string[];
    review_notes: string[];
    scene_summary?: string;
    observations?: string[];
    analysis_samples?: { timestamp_sec: number; image_base64: string }[];
    metadata: { duration_sec: number };
    metrics: {
      total_sec: number;
      processing_sec: number;
      stages: Record<string, number>;
      overview_frame_count?: number;
      unique_frame_count?: number;
      negative_audit_performed?: boolean;
      pipeline_version?: string;
      inference_calls?: {
        phase: string;
        image_count: number;
        first_timestamp_sec: number;
        last_timestamp_sec: number;
      }[];
    };
  };
};
type Health = {
  ready: boolean;
  model_state: string;
  max_upload_mb: number;
  model: string;
};
const time = (v: number) =>
  `${Math.floor(v / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(v % 60)
    .toString()
    .padStart(2, "0")}.${Math.floor((v % 1) * 10)}`;
const stages: Record<string, string> = {
  uploading: "Загрузка файла",
  queued: "Ожидание в очереди",
  starting: "Подготовка анализа",
  metadata: "Проверка видео",
  overview_frames: "Выбор кадров всей записи",
  overview: "Обзор всей последовательности",
  audit_frames: "Выбор кадров для повторной проверки",
  audit: "Повторная проверка всей записи",
  leave_check: "Проверка рабочего места после ухода",
  focused_checks: "Проверка отдельных признаков",
  evidence: "Подготовка кадров-доказательств",
  preview: "Подготовка плеера",
  done: "Проверка завершена",
  failed: "Проверка не завершена",
};
const stageName = (s: string) =>
  stages[s] ||
  (s.startsWith("detail_frames")
    ? "Выбор кадров фрагмента"
    : s.startsWith("detail")
      ? "Уточнение возможного события"
      : s);
type Marker = { at: number; label: string };

/** Video with the site's own controls: one timeline to scrub, with red violation markers on it. */
function Player({
  src,
  duration,
  markers,
  selected,
  videoRef,
  onMarker,
  onError,
}: {
  src?: string;
  duration: number;
  markers: Marker[];
  selected: number;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  onMarker: (index: number) => void;
  onError: () => void;
}) {
  const [playing, setPlaying] = useState(false),
    [current, setCurrent] = useState(0),
    [length, setLength] = useState(duration),
    [hover, setHover] = useState<number | null>(null);
  const bar = useRef<HTMLDivElement>(null),
    dragging = useRef(false);
  const total = length || duration || 1;

  function toggle() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => undefined);
    else v.pause();
  }
  function timeAt(clientX: number) {
    const r = bar.current!.getBoundingClientRect();
    return Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * total;
  }
  function jump(t: number) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.min(Math.max(0, t), total);
    setCurrent(v.currentTime);
  }
  function onKey(e: React.KeyboardEvent) {
    if (e.key === " " || e.key === "k") {
      e.preventDefault();
      toggle();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      jump(current - 5);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      jump(current + 5);
    }
  }
  return (
    <div className="player" tabIndex={0} onKeyDown={onKey}>
      <video
        ref={videoRef}
        preload="metadata"
        src={src}
        onClick={toggle}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => !dragging.current && setCurrent(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setLength(e.currentTarget.duration || duration)}
        onError={onError}
      />
      <div className="controls">
        <button className="ctrl-btn" onClick={toggle} aria-label={playing ? "Пауза" : "Воспроизвести"}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <span className="ctrl-time">
          {time(current)} / {time(total)}
        </span>
        <div
          ref={bar}
          className="scrub"
          role="slider"
          aria-label="Позиция в видео"
          aria-valuemin={0}
          aria-valuemax={total}
          aria-valuenow={current}
          onPointerDown={(e) => {
            dragging.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            jump(timeAt(e.clientX));
          }}
          onPointerMove={(e) => {
            setHover(timeAt(e.clientX));
            if (dragging.current) jump(timeAt(e.clientX));
          }}
          onPointerUp={() => (dragging.current = false)}
          onPointerLeave={() => setHover(null)}
        >
          <div className="scrub-track">
            <div className="scrub-fill" style={{ width: `${(current / total) * 100}%` }} />
          </div>
          {markers.map((m, i) => (
            <button
              key={i}
              className={`scrub-marker ${selected === i ? "selected" : ""}`}
              style={{ left: `${(m.at / total) * 100}%` }}
              title={`${time(m.at)} — ${m.label}`}
              aria-label={`Перейти к нарушению ${i + 1}: ${m.label}`}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={() => onMarker(i)}
            />
          ))}
          <div className="scrub-head" style={{ left: `${(current / total) * 100}%` }} />
          {hover !== null && (
            <span className="scrub-hover" style={{ left: `${(hover / total) * 100}%` }}>
              {time(hover)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...options, credentials: "same-origin" });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Ошибка запроса (${response.status})`,
    );
  }
  return response.json();
}
function App() {
  const [rules, setRules] = useState<Rule[]>([]),
    [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null),
    [job, setJob] = useState<Job | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => {
    const requested = new URLSearchParams(window.location.search).get("job");
    return requested && /^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(requested)
      ? requested
      : localStorage.getItem("cash:last-job");
  });
  const [error, setError] = useState(""),
    [uploading, setUploading] = useState(false),
    [drag, setDrag] = useState(false),
    [selected, setSelected] = useState(0),
    [now, setNow] = useState(Date.now()),
    [pollError, setPollError] = useState("");
  const video = useRef<HTMLVideoElement>(null),
    input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    api<Rule[]>("/ui/api/rules")
      .then(setRules)
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    let live = true;
    const update = () =>
      api<Health>("/ui/api/health")
        .then((h) => {
          if (live) setHealth(h);
        })
        .catch(() => {
          if (live) setHealth(null);
        });
    void update();
    const id = setInterval(update, 10000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, []);
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false,
      timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api<Job>(`/ui/api/jobs/${jobId}`);
        if (cancelled) return;
        setJob(next);
        setPollError("");
        if (next.status === "processing") timer = setTimeout(poll, 1500);
      } catch (e) {
        if (cancelled) return;
        setPollError((e as Error).message);
        timer = setTimeout(poll, 5000);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId]);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const busy = uploading || job?.status === "processing";
  function choose(candidate?: File) {
    if (!candidate || busy) return;
    setError("");
    if (!candidate.name.toLowerCase().endsWith(".mov")) {
      setError("Выберите видео в формате .mov.");
      return;
    }
    if (candidate.size > (health?.max_upload_mb || 100) * 1024 * 1024) {
      setError("Файл превышает допустимый размер.");
      return;
    }
    setFile(candidate);
  }
  async function submit() {
    if (!file || busy) return;
    setUploading(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await api<{ job_id: string }>("/ui/api/analyze", {
        method: "POST",
        body,
      });
      setJob(null);
      setSelected(0);
      setJobId(result.job_id);
      localStorage.setItem("cash:last-job", result.job_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }
  function reset() {
    setJob(null);
    setJobId(null);
    setFile(null);
    setError("");
    setPollError("");
    localStorage.removeItem("cash:last-job");
  }
  function seek(seconds: number, index: number) {
    setSelected(index);
    if (video.current) {
      video.current.currentTime = seconds;
      video.current.focus();
    }
  }
  const events = job?.extended?.events || [],
    current = events[selected],
    currentRule = rules.find((r) => r.id === current?.rule_id);
  const finished = job?.status === "done";
  const hasViolation = job?.verdict
    ? job.verdict === "violation"
    : events.length > 0;
  const modelNote = health?.ready
    ? ""
    : health?.model_state === "loading"
      ? "Модель загружается, запуск станет доступен через минуту."
      : "Модель недоступна, запуск станет доступен, когда сервер будет готов.";
  return (
    <div className="ft-app">
      <header className="site-header">
        <div className="header-inner">
          <a href="/" className="brand">
            <span className="brand-name">Проверка кассовых видео</span>
            <span className="brand-product">Datathon 26</span>
          </a>
          {(finished || job?.status === "failed") && (
            <button className="secondary header-new" onClick={reset}>
              <Plus size={17} />
              Новая проверка
            </button>
          )}
        </div>
      </header>
      <main className="workspace">
        {error && (
          <div className="error" role="alert">
            <CircleAlert size={18} />
            {error}
            <button aria-label="Закрыть ошибку" onClick={() => setError("")}>
              <X size={16} />
            </button>
          </div>
        )}
        {pollError && (
          <div className="error" role="alert">
            Не удалось обновить статус: {pollError}. Повторяем запрос.
          </div>
        )}
        {!job && (
          <section
            className={`upload ${drag ? "drag" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              choose(e.dataTransfer.files[0]);
            }}
          >
            <div className="upload-icon">
              {file ? <FileVideo size={30} /> : <Upload size={30} />}
            </div>
            <h1>{file ? file.name : "Загрузите видео с кассы"}</h1>
            <p>
              {file
                ? `${(file.size / 1024 / 1024).toFixed(1)} МБ · готово к проверке`
                : `Перетащите файл сюда · MOV до ${health?.max_upload_mb || 100} МБ`}
            </p>
            <div className="upload-actions">
              <button
                className="secondary"
                disabled={busy}
                onClick={() => input.current?.click()}
              >
                {file ? "Другой файл" : "Выбрать видео"}
              </button>
              {file && (
                <button
                  className="primary"
                  disabled={!health?.ready || !!busy}
                  onClick={submit}
                >
                  {uploading ? (
                    <LoaderCircle className="spin" size={17} />
                  ) : (
                    <ScanLine size={17} />
                  )}
                  {uploading ? "Загрузка…" : "Проверить"}
                </button>
              )}
            </div>
            <input
              ref={input}
              type="file"
              accept=".mov,video/quicktime"
              hidden
              onChange={(e) => choose(e.target.files?.[0])}
            />
            {modelNote && <span className="model-note">{modelNote}</span>}
          </section>
        )}
        {job?.status === "processing" && (
          <section className="panel progress-panel" aria-live="polite">
            <LoaderCircle className="spin" size={25} />
            <div>
              <h2>{stageName(job.stage)}</h2>
              <p>
                {job.filename} · прошло{" "}
                {Math.max(0, Math.floor(now / 1000 - job.created_at))} сек
              </p>
            </div>
            <span className="pill">В обработке</span>
          </section>
        )}
        {job?.status === "failed" && (
          <section className="panel failed-panel" role="alert">
            <CircleAlert size={25} />
            <div>
              <h2>Проверка не завершена</h2>
              <p>
                {job.error?.message || "Не удалось обработать видео."} Вердикт
                не сформирован.
              </p>
            </div>
            <button className="secondary" onClick={reset}>
              Повторить
            </button>
          </section>
        )}
        {finished && (
          <>
            <section
              className={`verdict reveal-now ${hasViolation ? "attention" : "clear"}`}
            >
              <div className="verdict-icon">
                {hasViolation ? <CircleAlert size={26} /> : <Check size={26} />}
              </div>
              <div>
                <span className="eyebrow">Проверка завершена</span>
                <h2>
                  {hasViolation
                    ? "Обнаружены признаки возможных нарушений"
                    : "Нарушение не обнаружено"}
                </h2>
                <p>
                  {hasViolation
                    ? `${events.length} ${plural(events.length, "момент требует", "момента требуют", "моментов требуют")} вашей оценки — таймкоды, кадры и правила ниже.`
                    : "Видимых признаков нарушений по известным категориям нет. Это подсказка для проверяющего, а не окончательное заключение."}
                </p>
              </div>
              <span className="verdict-time">
                <Clock3 size={16} />
                {job.extended?.metrics.total_sec.toFixed(1)} сек
              </span>
            </section>
            <div className="results-grid">
              <section className="panel player-panel">
                <div className="panel-heading">
                  <div>
                    <Film size={18} />
                    <h2>{job.filename}</h2>
                  </div>
                  <span className="muted">
                    {time(job.extended?.metadata.duration_sec || 0)}
                  </span>
                </div>
                <Player
                  src={job.video_url}
                  duration={job.extended?.metadata.duration_sec || 0}
                  videoRef={video}
                  selected={selected}
                  markers={events.map((event, i) => ({
                    at: job.violations?.[i]?.timecode_sec ?? event.start_sec,
                    label:
                      job.violations?.[i]?.type ||
                      rules.find((r) => r.id === event.category)?.name ||
                      event.category,
                  }))}
                  onMarker={(i) =>
                    seek(job.violations?.[i]?.timecode_sec ?? events[i].start_sec, i)
                  }
                  onError={() =>
                    setError(
                      "Не удалось воспроизвести видео. Проверьте доступ и обновите страницу.",
                    )
                  }
                />
                <div className="timeline-label">
                  <span>Нарушения отмечены красным — нажмите, чтобы перейти</span>
                  <span>{events.length}</span>
                </div>
              </section>
              <section className="panel event-list">
                <div className="panel-heading">
                  <h2>Моменты для проверки</h2>
                  <span className="count">{events.length}</span>
                </div>
                {events.length ? (
                  events.map((event, i) => {
                    const violation = job.violations?.[i];
                    const thumb = violation?.evidence_frames[0];
                    const at = violation?.timecode_sec ?? event.start_sec;
                    const rule = rules.find((r) => r.id === event.category);
                    return (
                      <button
                        key={i}
                        className={`event-item ${selected === i ? "selected" : ""}`}
                        onClick={() => seek(at, i)}
                      >
                        {thumb && (
                          <span className="event-thumb">
                            <img
                              src={`data:image/jpeg;base64,${thumb.image_base64}`}
                              alt=""
                            />
                          </span>
                        )}
                        <span className="event-body">
                          <span className="event-meta">
                            <span className="event-index">
                              {String(i + 1).padStart(2, "0")}
                            </span>
                            <span className="event-time">
                              <Play size={12} />
                              {time(at)}
                            </span>
                          </span>
                          <h3>
                            {violation?.type || rule?.name || event.category}
                          </h3>
                          <p>
                            {violation?.description || event.observed_action}
                          </p>
                          {rule && (
                            <span className="event-rule">
                              <BookOpen size={13} />
                              {sourceLabel(rule)}
                            </span>
                          )}
                        </span>
                      </button>
                    );
                  })
                ) : (
                  <p className="no-events">
                    Событий по рабочим категориям не найдено. Ознакомьтесь с
                    ограничениями анализа ниже.
                  </p>
                )}
              </section>
            </div>
            {current && (
              <section className="panel evidence-panel">
                <div className="panel-heading">
                  <div>
                    <span className="step-num">
                      {String(selected + 1).padStart(2, "0")}
                    </span>
                    <h2>
                      {job.violations?.[selected]?.type ||
                        currentRule?.name ||
                        "Основание для проверки"}
                    </h2>
                  </div>
                  <span className="event-time">
                    <Clock3 size={14} />
                    Таймкод{" "}
                    {time(
                      job.violations?.[selected]?.timecode_sec ??
                        current.start_sec,
                    )}
                  </span>
                </div>
                <div className="evidence-grid">
                  {job.violations?.[selected]?.evidence_frames.map((frame) => (
                    <button
                      key={frame.timestamp_sec}
                      className="evidence"
                      onClick={() => seek(frame.timestamp_sec, selected)}
                    >
                      <img
                        src={`data:image/jpeg;base64,${frame.image_base64}`}
                        alt={`Наблюдаемое событие, ${time(frame.timestamp_sec)}`}
                      />
                      <span>
                        <Play size={12} />
                        {time(frame.timestamp_sec)}
                      </span>
                    </button>
                  ))}
                </div>
                <div className="explanation">
                  <div>
                    <h3>Что видно на видео</h3>
                    <p>{current.observed_action}</p>
                    <h3>Почему это может быть нарушением</h3>
                    <p>{current.applicability_basis}</p>
                    {current.uncertainty_reason && (
                      <>
                        <h3>Что нельзя утверждать по видео</h3>
                        <p>{current.uncertainty_reason}</p>
                      </>
                    )}
                  </div>
                  {currentRule && <Source rule={currentRule} />}
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
const stripLead = (title: string) =>
  title.replace(/\s*—\s*отправная точка исследования\s*$/i, "");
function sourceLabel(rule: Rule) {
  if (rule.normative_source)
    return [rule.normative_source.title, rule.normative_source.section]
      .filter(Boolean)
      .join(", ");
  const lead = rule.research_leads[0];
  return lead
    ? `${stripLead(lead.title)} · пункт уточняется`
    : "Источник уточняется";
}
function plural(n: number, one: string, few: string, many: string) {
  const m10 = n % 10,
    m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
function Source({ rule }: { rule: Rule }) {
  const verified =
    rule.source_verification_status === "verified" && rule.normative_source;
  const lead = rule.research_leads[0];
  return (
    <aside className="source">
      <span className="source-label">
        <BookOpen size={15} /> Правило и источник
      </span>
      <strong className="source-rule">{rule.name}</strong>
      <p className="source-desc">{rule.description}</p>
      {rule.normative_source ? (
        <p className="source-doc">
          <a href={rule.normative_source.url} target="_blank" rel="noreferrer">
            {rule.normative_source.title}
          </a>
          {rule.normative_source.section && (
            <span>{rule.normative_source.section}</span>
          )}
        </p>
      ) : lead ? (
        <p className="source-doc">
          <a href={lead.url} target="_blank" rel="noreferrer">
            {stripLead(lead.title)} ↗
          </a>
        </p>
      ) : null}
      {!verified && (
        <span className="source-status">
          {rule.normative_source || lead
            ? "Пункт документа уточняется"
            : "Источник уточняется"}
        </span>
      )}
    </aside>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
