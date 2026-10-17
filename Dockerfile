FROM node:22-bookworm-slim AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /service
RUN groupadd --gid 10001 cash && useradd --uid 10001 --gid cash --create-home cash
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .
COPY --from=frontend /ui/dist ./frontend/dist
COPY scripts ./scripts
RUN mkdir /service/runtime /service/models && chown -R cash:cash /service/runtime /service/models
ENV CASH_DATA_DIR=/service/runtime CASH_MODEL_PATH=/service/models/Qwen3-VL-4B-Instruct
USER cash
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

# Build base for local API/UI diagnostics; GPU target includes local inference dependencies.
FROM base AS gpu
USER root
RUN pip install --no-cache-dir '.[gpu]'
USER cash
