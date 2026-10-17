#!/bin/bash
# Start the service with the nmk two-pass engine (Qwen3-VL-8B) on 0.0.0.0:8000.
# Secrets (CASH_API_KEY, CASH_UI_PASSWORD) come from .env; paths can be overridden by env vars.
cd "$(dirname "$0")/../.."
export CASH_ENGINE=nmk CASH_MODEL_ENABLED=true CASH_UI_OPEN=${CASH_UI_OPEN:-true} \
       CASH_NMK_DIR=${CASH_NMK_DIR:-$PWD/nmk} \
       CASH_NMK_MODEL_PATH=${CASH_NMK_MODEL_PATH:-/root/models/Qwen3-VL-8B-Instruct} \
       CASH_NMK_PRECISION=${CASH_NMK_PRECISION:-bf16} \
       CASH_GPU_MIN_FREE_MB=${CASH_GPU_MIN_FREE_MB:-0} CASH_QUEUE_CAPACITY=${CASH_QUEUE_CAPACITY:-10} \
       CASH_JOB_TIMEOUT_SEC=${CASH_JOB_TIMEOUT_SEC:-290} CASH_MAX_UPLOAD_MB=${CASH_MAX_UPLOAD_MB:-200} \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
exec ${PYTHON:-python} -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
