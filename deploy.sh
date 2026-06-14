#!/usr/bin/env bash
#
# deploy.sh — Operational Resilience Pipeline (Phase 1, ROADMAP.md)
#
# Design Enforcements:
#   • set -euo pipefail — Any error aborts the entire pipeline immediately.
#   • Unit Tests Gate — Local tests must pass before touching the server.
#   • Safe by Default — Always builds + deploys frontend unless explicit opt-out.
#   • Zero-Stub Verification — Hard network + disk checks kill deployment blindness.
#   • Hang Prevention — Layered network timeouts + BatchMode isolate all components.
#
# Health target: /api/health — purpose-built, cheap even if ENABLE_AUTH=false.
# Under production auth it returns 401 (proves the full stack + auth middleware).

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
SSH_KEY="$HOME/.ssh/gcp_stock_rsa"
SERVER="elimaoz99@35.239.74.178"
REMOTE_DIR="/home/elimaoz99/stock_predictor"
PROJECT_DIR="$HOME/Desktop/Stock-predictor"
BACKEND_FILES=("api.py" "scanners.py" "db.py")
SERVICE="stock-app.service"

BACKEND_ONLY=false
if [[ "${1:-}" == "--backend-only" ]]; then
  BACKEND_ONLY=true
fi

# ── Loud failure trap — name the stage that broke (no rollback, by design) ────
CURRENT_STAGE="init"
trap 'echo "❌ DEPLOY ABORTED at stage: ${CURRENT_STAGE} (Exit Code: $?)" >&2' ERR

stage() {
  CURRENT_STAGE="$1"
  echo "▶ Starting stage: [ $1 ]"
}

# Hang-proof SSH helper (network vector via ConnectTimeout + auth vector via BatchMode)
ssh_cmd() {
  ssh -o ConnectTimeout=4 -o BatchMode=yes -i "$SSH_KEY" "$SERVER" "$@"
}

cd "$PROJECT_DIR"

# ── Stage 0 — Blocking unittest gate ─────────────────────────────────────────
stage "unittest gate"
python3 -m unittest test_scanners
echo "  ✓ Local tests passed. Gate opened."

# ── Stage 1 — Frontend build + local hash extraction (source of truth) ───────
LOCAL_ASSET=""
if [[ "$BACKEND_ONLY" == false ]]; then
  stage "frontend build"
  ( cd frontend && npm run build )

  LOCAL_ASSET=$(grep -oE 'assets/index-[A-Za-z0-9_-]+\.js' frontend/dist/index.html | head -1)
  if [[ -z "$LOCAL_ASSET" ]]; then
    echo "  ❌ Error: Failed to extract Vite production asset hash from index.html" >&2
    exit 1
  fi
  echo "  ✓ Frontend built. Target entry asset: $LOCAL_ASSET"
fi

# ── Stage 2 — Unified, atomic SCP upload ─────────────────────────────────────
stage "scp backend"
# The three core files travel together — api.py imports scanners + db at module load.
scp -i "$SSH_KEY" "${BACKEND_FILES[@]}" "$SERVER:$REMOTE_DIR/"
echo "  ✓ Backend files copied."

if [[ "$BACKEND_ONLY" == false ]]; then
  stage "scp frontend"
  scp -r -i "$SSH_KEY" frontend/dist/ "$SERVER:$REMOTE_DIR/"
  # Sync into the directory FastAPI actually serves from (full copy — Vite hashes filenames).
  ssh_cmd "cp -r $REMOTE_DIR/dist/. $REMOTE_DIR/frontend/dist/"
  echo "  ✓ Frontend static assets synchronized."
fi

# ── Stage 3 — Restart ────────────────────────────────────────────────────────
stage "server restart"
ssh_cmd "sudo systemctl restart $SERVICE"
echo "  ✓ Restart signal acknowledged. Entering cold-start recovery window..."

# ── Stage 4 — Remote verification (zero-blindness leaves) ────────────────────
stage "backend health check"
MAX_RETRIES=5
RETRY_INTERVAL=3
ATTEMPT=1
HTTP_CODE="000"

while [ "$ATTEMPT" -le "$MAX_RETRIES" ]; do
  # Time-boxed curl catches a frozen/crashing uvicorn, not just a dead port.
  HTTP_CODE=$(ssh_cmd "curl --connect-timeout 2 --max-time 4 -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/health") || HTTP_CODE="000"

  # 200 (open) or 401 (alive behind global Basic Auth) both prove the stack is up.
  if [[ "$HTTP_CODE" == "200" || "$HTTP_CODE" == "401" ]]; then
    echo "  ✓ Backend live and responding (HTTP $HTTP_CODE) on attempt $ATTEMPT."
    break
  fi

  echo "  [Attempt $ATTEMPT/$MAX_RETRIES] Backend not ready (HTTP $HTTP_CODE). Retrying in ${RETRY_INTERVAL}s..."
  sleep "$RETRY_INTERVAL"
  ATTEMPT=$((ATTEMPT + 1))
done

if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "401" ]]; then
  echo "  ❌ Error: Backend failed health check (last HTTP $HTTP_CODE) after $MAX_RETRIES attempts." >&2
  exit 1
fi

if [[ "$BACKEND_ONLY" == false ]]; then
  stage "frontend disk check"
  # Verify the exact hashed asset we built locally physically landed in the active dir.
  if ! ssh_cmd "test -f $REMOTE_DIR/frontend/dist/$LOCAL_ASSET"; then
    echo "  ❌ Error: Frontend consistency check failed — $LOCAL_ASSET missing from active dir." >&2
    exit 1
  fi
  echo "  ✓ Frontend asset consistency verified on remote disk."
fi

echo "========================================================================="
if [[ "$BACKEND_ONLY" == true ]]; then
  echo "✅ DEPLOYMENT SUCCESSFUL: gates passed, backend live (frontend skipped)."
else
  echo "✅ DEPLOYMENT SUCCESSFUL: gates passed, backend live, assets verified on disk."
fi
echo "========================================================================="
