import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  FileVideo,
  Film,
  ListChecks,
  LoaderCircle,
  LockKeyhole,
  Play,
  Plus,
  ScanLine,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
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
  const [tab, setTab] = useState<"review" | "rules">("review");
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
  return (
    <div className="shell">
      <aside className="sidebar">
        <a href="/" className="brand">
          <span className="brand-mark">
            <ScanLine size={24} />
          </span>
          касса<span className="brand-dot">.</span>
        </a>
        <div className="workspace">
          <span className="workspace-avatar">F</span>
          <div>
            ForteBank<span>Datathon 26 · MVP</span>
          </div>
        </div>
        <span className="nav-label">РАБОЧЕЕ ПРОСТРАНСТВО</span>
        <nav>
          <button
            className={tab === "review" ? "active" : ""}
            onClick={() => setTab("review")}
          >
            <Film size={18} />
            Проверка видео
            <ChevronRight size={15} />
          </button>
          <button
            className={tab === "rules" ? "active" : ""}
            onClick={() => setTab("rules")}
          >
            <ListChecks size={18} />
            Каталог правил<span className="count">7</span>
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="local-icon">
            <LockKeyhole size={18} />
          </div>
          <strong>Внутри вашего контура</strong>
          <p>Видео обрабатывается локально. Внешние AI API не используются.</p>
          <span className="version">ПРОТОТИП / V0.1</span>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <span>
            Рабочее пространство <ChevronRight size={14} />{" "}
            <strong>
              {tab === "review" ? "Проверка видео" : "Каталог правил"}
            </strong>
          </span>
          <span className="private">
            <LockKeyhole size={13} /> Защищённый доступ
          </span>
        </header>
        <main>
          <div className="heading">
            <div>
              <div className="eyebrow">ВИДЕОАНАЛИТИКА КАССОВЫХ ОПЕРАЦИЙ</div>
              <h1>
                {tab === "review"
                  ? "Внимание к каждой операции."
                  : "Правила и границы выводов."}
              </h1>
              <p>
                {tab === "review"
                  ? "Найдите моменты, которые требуют проверки человеком."
                  : "Рабочая таксономия команды. Нормативное обоснование ещё проверяется."}
              </p>
            </div>
            {finished && tab === "review" && (
              <button className="secondary" onClick={reset}>
                <Plus size={17} />
                Новая проверка
              </button>
            )}
          </div>
          <div className="notice">
            <ShieldCheck size={19} />
            <span>
              Помощник проверяющего. Выводы основаны на видимых действиях и
              требуют вашей оценки.
            </span>
            <span className="notice-tag">ЧЕЛОВЕК ПРИНИМАЕТ РЕШЕНИЕ</span>
          </div>
          {tab === "rules" ? (
            <div className="rules-list">
              {rules.map((rule, i) => (
                <article className="rule-card" key={rule.id}>
                  <span className="step-num">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div>
                    <h2>{rule.name}</h2>
                    <p>{rule.description}</p>
                    <dl>
                      <dt>Условия применимости</dt>
                      <dd>{rule.applicability_conditions}</dd>
                      <dt>Исключения</dt>
                      <dd>{rule.exceptions}</dd>
                      <dt>Наблюдаемые признаки</dt>
                      <dd>{rule.observable_signs}</dd>
                    </dl>
                    <Source rule={rule} />
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <>
              {error && (
                <div className="error" role="alert">
                  <CircleAlert size={18} />
                  {error}
                  <button
                    aria-label="Закрыть ошибку"
                    onClick={() => setError("")}
                  >
                    <X size={16} />
                  </button>
                </div>
              )}
              {pollError && (
                <div className="error" role="alert">
                  Не удалось обновить статус: {pollError}. Повторяем запрос;
                  результат пока неизвестен.
                </div>
              )}
              {!finished && (
                <div className="upload-grid">
                  <section className="panel upload-panel">
                    <div className="panel-heading">
                      <div>
                        <span className="step-num">01</span>
                        <h2>Загрузите запись</h2>
                      </div>
                      <span className="muted">Одна касса · одно видео</span>
                    </div>
                    <div
                      className={`dropzone ${drag ? "drag" : ""}`}
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
                      <h3>{file ? file.name : "Перетащите видео сюда"}</h3>
                      <p>
                        {file
                          ? `${(file.size / 1024 / 1024).toFixed(1)} МБ · готово к загрузке`
                          : "или выберите файл на вашем устройстве"}
                      </p>
                      <button
                        className="secondary"
                        disabled={busy}
                        onClick={() => input.current?.click()}
                      >
                        {file ? "Выбрать другой файл" : "Выбрать видео"}
                        <ArrowUpRight size={16} />
                      </button>
                      <input
                        ref={input}
                        type="file"
                        accept=".mov,video/quicktime"
                        hidden
                        onChange={(e) => choose(e.target.files?.[0])}
                      />
                      <span className="file-hint">
                        MOV · до {health?.max_upload_mb || 100} МБ · до 145
                        секунд
                      </span>
                    </div>
                    <div className="upload-footer">
                      <span>
                        <LockKeyhole size={15} /> Доступ только проверяющим
                      </span>
                      <button
                        className="primary"
                        disabled={!file || !health?.ready || !!busy}
                        onClick={submit}
                      >
                        {uploading ? (
                          <LoaderCircle className="spin" size={17} />
                        ) : (
                          <ScanLine size={17} />
                        )}{" "}
                        {uploading ? "Загрузка…" : "Начать проверку"}
                      </button>
                    </div>
                  </section>
                  <section className="panel process-panel">
                    <div className="panel-heading">
                      <div>
                        <span className="step-num">02</span>
                        <h2>Проверка записи</h2>
                      </div>
                    </div>
                    <div
                      className={`model-status ${health?.ready ? "ready" : ""}`}
                    >
                      <span className="status-dot" />
                      <div>
                        <strong>
                          {health?.ready
                            ? "Модель готова"
                            : health?.model_state === "loading"
                              ? "Модель загружается"
                              : health?.model_state === "disabled"
                                ? "Модель не запущена"
                                : "Модель недоступна"}
                        </strong>
                        <span>Qwen3-VL-4B-Instruct · локально</span>
                      </div>
                    </div>
                    {!health?.ready && (
                      <p className="setup-note">
                        Для анализа нужны локальные веса и доступный GPU.
                        Результат появится только после настоящего анализа.
                      </p>
                    )}
                    <ol className="steps">
                      {[
                        "Обзор всей последовательности",
                        "Уточнение возможных событий",
                        "Подбор кадров и объяснение",
                      ].map((s, i) => (
                        <li key={s}>
                          <span>{i + 1}</span>
                          <div>
                            <strong>{s}</strong>
                            <p>
                              {
                                [
                                  "Сохраняем контекст операции",
                                  "Проверяем подозрительные фрагменты",
                                  "Показываем, что требует внимания",
                                ][i]
                              }
                            </p>
                          </div>
                        </li>
                      ))}
                    </ol>
                    <div className="processing-foot">
                      <Clock3 size={16} />
                      <span>
                        Лимит — 5 минут, включая очередь.
                        <br />
                        Скорость модели ещё не измерена.
                      </span>
                    </div>
                  </section>
                </div>
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
                      {job.error?.message || "Не удалось обработать видео."}{" "}
                      Вердикт не сформирован.
                    </p>
                  </div>
                  <button className="secondary" onClick={reset}>
                    Повторить
                  </button>
                </section>
              )}
              {!job && (
                <section className="empty-result">
                  <span className="empty-result-icon">
                    <Film size={22} />
                  </span>
                  <div>
                    <h2>Здесь появится результат проверки</h2>
                    <p>
                      Видео, таймкоды, события и кадры для самостоятельной
                      оценки.
                    </p>
                  </div>
                  <div className="empty-tags">
                    <span>СОБЫТИЯ</span>
                    <span>КАДРЫ</span>
                    <span>ИСТОЧНИКИ</span>
                  </div>
                </section>
              )}
              {finished && (
                <>
                  <section
                    className={`verdict ${events.length ? "attention" : "clear"}`}
                  >
                    <div className="verdict-icon">
                      {events.length ? (
                        <CircleAlert size={26} />
                      ) : (
                        <Check size={26} />
                      )}
                    </div>
                    <div>
                      <span className="eyebrow">ПРОВЕРКА ЗАВЕРШЕНА</span>
                      <h2>
                        {events.length
                          ? "Обнаружены признаки возможных нарушений"
                          : "Признаки по рабочим категориям не найдены"}
                      </h2>
                      <p>
                        {events.length
                          ? `${events.length} событий требуют вашей оценки`
                          : "Достаточно обоснованных признаков перечисленных нарушений не найдено. Это не гарантия их отсутствия."}
                      </p>
                    </div>
                    <span className="verdict-time">
                      <Clock3 size={16} />
                      {job.extended?.metrics.total_sec.toFixed(1)} сек
                    </span>
                  </section>
                  <section className="panel analysis-coverage">
                    <h2>Что проанализировано</h2>
                    <p className="coverage-counts">
                      Кадров в обзоре:{" "}
                      <strong>
                        {job.extended?.metrics.overview_frame_count ?? "—"}
                      </strong>
                      {job.extended?.metrics.unique_frame_count !==
                        undefined && (
                        <>
                          {" "}
                          · Всего разных кадров:{" "}
                          <strong>
                            {job.extended.metrics.unique_frame_count}
                          </strong>
                        </>
                      )}
                      {job.extended?.metrics.inference_calls && (
                        <>
                          {" "}
                          · Вызовов Qwen:{" "}
                          <strong>
                            {job.extended.metrics.inference_calls.length}
                          </strong>
                        </>
                      )}
                    </p>
                    {job.extended?.scene_summary ? (
                      <>
                        <h3>Описание видимого по версии модели</h3>
                        <p>{job.extended.scene_summary}</p>
                        <ul>
                          {job.extended.observations?.map((note, index) => (
                            <li key={index}>{note}</li>
                          ))}
                        </ul>
                        {job.extended.metrics.negative_audit_performed && (
                          <p className="coverage-audit">
                            <Check size={15} />
                            После пустого обзора выполнена повторная проверка
                            всей записи.
                          </p>
                        )}
                        <details>
                          <summary>Кадры, переданные в обзор</summary>
                          <p>
                            Это примеры входных кадров. Они сами по себе не
                            доказывают нарушение.
                          </p>
                          <div className="evidence-grid">
                            {job.extended.analysis_samples?.map((frame) => (
                              <button
                                key={frame.timestamp_sec}
                                className="evidence"
                                onClick={() =>
                                  seek(frame.timestamp_sec, selected)
                                }
                              >
                                <img
                                  src={`data:image/jpeg;base64,${frame.image_base64}`}
                                  alt={`Входной кадр ${time(frame.timestamp_sec)}`}
                                />
                                <span>{time(frame.timestamp_sec)}</span>
                              </button>
                            ))}
                          </div>
                        </details>
                      </>
                    ) : (
                      <p>
                        Результат получен предыдущей версией анализа. Для
                        повторной проверки с новым алгоритмом загрузите запись
                        снова.
                      </p>
                    )}
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
                      <video
                        ref={video}
                        controls
                        preload="metadata"
                        src={job.video_url}
                        onError={() =>
                          setError(
                            "Не удалось воспроизвести видео. Проверьте доступ и обновите страницу.",
                          )
                        }
                      />
                      <div className="timeline-label">
                        <span>События на записи</span>
                        <span>{events.length} отметок</span>
                      </div>
                      <div className="timeline">
                        {events.map((event, i) => (
                          <button
                            key={i}
                            title={`${time(event.start_sec)} — ${rules.find((r) => r.id === event.category)?.name}`}
                            aria-label={`Перейти к событию ${i + 1}`}
                            onClick={() => seek(event.start_sec, i)}
                            style={{
                              left: `${(event.start_sec / (job.extended?.metadata.duration_sec || 1)) * 100}%`,
                              width: `${Math.max(1, ((event.end_sec - event.start_sec) / (job.extended?.metadata.duration_sec || 1)) * 100)}%`,
                            }}
                          />
                        ))}
                      </div>
                      <div className="timeline-label">
                        <span>00:00</span>
                        <span>
                          {time(job.extended?.metadata.duration_sec || 0)}
                        </span>
                      </div>
                    </section>
                    <section className="panel event-list">
                      <div className="panel-heading">
                        <h2>Моменты для проверки</h2>
                        <span className="count">{events.length}</span>
                      </div>
                      {events.length ? (
                        events.map((event, i) => (
                          <button
                            key={i}
                            className={`event-item ${selected === i ? "selected" : ""}`}
                            onClick={() => seek(event.start_sec, i)}
                          >
                            <div>
                              <span className="event-index">
                                {String(i + 1).padStart(2, "0")}
                              </span>
                              <span className="event-time">
                                <Play size={12} />
                                {time(event.start_sec)} — {time(event.end_sec)}
                              </span>
                            </div>
                            <h3>
                              {rules.find((r) => r.id === event.category)
                                ?.name || event.category}
                            </h3>
                            <p>{event.observed_action}</p>
                          </button>
                        ))
                      ) : (
                        <p className="no-events">
                          Событий по рабочим категориям не найдено. Ознакомьтесь
                          с ограничениями анализа ниже.
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
                          <h2>Основание для проверки</h2>
                        </div>
                        <span className="muted">Кадры из исходного видео</span>
                      </div>
                      <div className="evidence-grid">
                        {job.violations?.[selected]?.evidence_frames.map(
                          (frame) => (
                            <button
                              key={frame.timestamp_sec}
                              className="evidence"
                              onClick={() =>
                                seek(frame.timestamp_sec, selected)
                              }
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
                          ),
                        )}
                      </div>
                      <div className="explanation">
                        <div>
                          <h3>Наблюдаемое действие</h3>
                          <p>{current.observed_action}</p>
                          <h3>Почему требует проверки</h3>
                          <p>{current.applicability_basis}</p>
                        </div>
                        <div>
                          <h3>Неопределённость</h3>
                          <p>{current.uncertainty_reason}</p>
                          {currentRule && <Source rule={currentRule} />}
                        </div>
                      </div>
                    </section>
                  )}
                  <details className="panel limitations" open>
                    <summary>Ограничения и дополнительные наблюдения</summary>
                    <ul>
                      {job.extended?.limitations.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                    {!!job.extended?.review_notes.length && (
                      <>
                        <h3>Наблюдения, не включённые в вердикт</h3>
                        <ul>
                          {job.extended.review_notes.map((s) => (
                            <li key={s}>{s}</li>
                          ))}
                        </ul>
                      </>
                    )}
                    <details>
                      <summary>Время этапов</summary>
                      <dl>
                        {Object.entries(job.extended?.metrics.stages || {}).map(
                          ([s, n]) => (
                            <React.Fragment key={s}>
                              <dt>{stageName(s)}</dt>
                              <dd>{n.toFixed(2)} сек</dd>
                            </React.Fragment>
                          ),
                        )}
                      </dl>
                    </details>
                  </details>
                </>
              )}
            </>
          )}
          <footer>
            <span>
              касса.{" "}
              <span className="muted">Помощник проверки видеозаписей</span>
            </span>
            <span>
              Datathon 26 <span className="footer-dot">•</span> ForteBank
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
function Source({ rule }: { rule: Rule }) {
  return (
    <div className="source">
      <span className="source-status">
        {rule.source_verification_status === "verified"
          ? "Источник проверен"
          : "Нормативная связь не проверена"}
      </span>
      {rule.normative_source ? (
        <p>
          <a href={rule.normative_source.url} target="_blank" rel="noreferrer">
            {rule.normative_source.title}
          </a>{" "}
          {rule.normative_source.section}
        </p>
      ) : (
        <p>Подтверждённый источник для этой категории пока не указан.</p>
      )}
      {rule.research_leads.map((lead) => (
        <details key={lead.url}>
          <summary>Отправная точка исследования</summary>
          <a href={lead.url} target="_blank" rel="noreferrer">
            {lead.title} ↗
          </a>
          <p>{lead.note}</p>
        </details>
      ))}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
