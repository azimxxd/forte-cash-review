#!/usr/bin/env bash
# Run ON the RunPod host from the project folder (e.g. /opt/forte-cash-review).
# Starts the service and a public HTTPS tunnel, each in its own tmux session.
# Idempotent: sessions that already run are left alone. Prints the public URL.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE=forte-cash-review
TUNNEL=forte-cash-tunnel
CLOUDFLARED_VERSION=2024.8.2

if ! tmux has-session -t "$SERVICE" 2>/dev/null; then
  tmux new-session -d -s "$SERVICE" -c "$PWD" \
    'exec env OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log > service.log 2>&1'
fi

echo "Waiting for the model to load (up to 10 min)…"
for _ in $(seq 1 120); do
  curl -fsS http://127.0.0.1:8000/health/ready >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS http://127.0.0.1:8000/health/ready || {
  echo "Service is not ready; see service.log" >&2
  exit 1
}
echo

if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /usr/local/bin/cloudflared \
    "https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
  chmod +x /usr/local/bin/cloudflared
fi

if ! tmux has-session -t "$TUNNEL" 2>/dev/null; then
  : > tunnel.log
  tmux new-session -d -s "$TUNNEL" -c "$PWD" \
    'exec cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 > tunnel.log 2>&1'
fi

for _ in $(seq 1 30); do
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' tunnel.log | head -1 || true)
  [ -n "$url" ] && break
  sleep 2
done
if [ -z "${url:-}" ]; then
  echo "Tunnel URL not found; see tunnel.log" >&2
  exit 1
fi
echo "Public URL: $url"
echo "The URL changes whenever the tunnel restarts — resend it to the jury if so."
