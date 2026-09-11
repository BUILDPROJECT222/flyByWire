#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -d ui/node_modules || ! -f data/malecns/visual-circuit.npz || ! -f data/malecns/full-spiking.npz || ! -f data/malecns/full-brain.json || ! -f data/malecns/full-positions.bin || ! -f data/malecns/full-graded.npz || ! -f data/malecns/full-retina.json ]]; then
  echo 'Dependencies or circuit data are missing. Follow README.md setup first.' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import socket
for port in (8766,3000):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try: sock.bind(('127.0.0.1',port))
        except OSError: raise SystemExit(f'Port {port} is occupied. Stop the existing lab before starting another.')
PY
mkdir -p .lab-logs
.venv/bin/python -m uvicorn lab.server:app --host 127.0.0.1 --port 8766 --no-access-log >.lab-logs/worker.log 2>&1 &
worker_pid=$!
(cd ui && exec npm run dev -- --host 127.0.0.1 --port 3000 --strictPort) >.lab-logs/ui.log 2>&1 &
ui_pid=$!
cleanup() {
  kill "$worker_pid" "$ui_pid" 2>/dev/null || true
  wait "$worker_pid" "$ui_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 0' INT TERM
echo 'Fly / Flight: http://localhost:3000/'
echo 'Brain observation + explicit supervised flight controls. Ctrl-C requests landing before shutdown.'
echo 'Logs: .lab-logs/worker.log and .lab-logs/ui.log'
while kill -0 "$worker_pid" 2>/dev/null && kill -0 "$ui_pid" 2>/dev/null; do sleep 1; done
echo 'A service stopped. See .lab-logs/ for details.' >&2
exit 1
