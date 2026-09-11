#!/bin/bash
# Container entrypoint: worker + UI on loopback, Caddy on $PORT. Exit if any stops so the host restarts us.
set -euo pipefail
cd /app
python -m uvicorn lab.server:app --host 127.0.0.1 --port 8766 --no-access-log &
(cd ui && exec node node_modules/vinext/dist/cli.js start --hostname 127.0.0.1 --port 3000) &
caddy run --config deploy/Caddyfile --adapter caddyfile &
wait -n
echo 'A service stopped; exiting.' >&2
exit 1
