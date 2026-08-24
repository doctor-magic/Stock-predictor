#!/bin/bash
# run_local.sh — run the full Stock Predictor UI on the Mac, against a local
# copy of the server's data. No GCP, no network exposure, no cost.
#
#   ./run_local.sh          sync fresh data, then serve on http://localhost:8010
#   ./run_local.sh --no-sync   serve immediately with whatever data is local
#
# The server keeps collecting; this is a read-only local view of it.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" != "--no-sync" ]; then
  echo "→ syncing data from server…"
  ./pull_caches_local.sh
fi

echo "→ starting local API on http://localhost:8010"
echo "  (login with the pair from api_data.env; Ctrl-C to stop)"
exec ./venv/bin/python3 -m uvicorn api:app --host 127.0.0.1 --port 8010
