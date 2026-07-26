#!/bin/bash
# warm_volume_cache.sh — keep the Volume Leaders backend cache warm (force=true).
#
# Runs from the */25 cron. The OLD cron called the API with NO credentials, so
# after the May-2026 Basic Auth hardening it silently returned 401 every 25 min
# for weeks (its output went to /dev/null, so nobody saw it). This sends the
# first BASIC_AUTH_USERS pair (mirrors live_tracker._basic_auth_header) and LOGS
# the HTTP code, so the next failure (401/429/500) is visible, not buried.
set -uo pipefail

cd /home/elimaoz99/stock_predictor || exit 1
ENVF=api_data.env
TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Market-hours gate (added Jul 5 2026, pre-downgrade): the warm forces the FULL
# volume-leaders pipeline (Yahoo screener + yfinance batches + on-demand ML) —
# running it 24/7 incl. weekends/holidays wastes the shared-core e2-medium and
# burns Yahoo IP budget for data that cannot change. Warm only 09:00-16:59 ET
# on NYSE trading days (market_calendar is the same guard the crons use).
GATE=$(venv/bin/python3 - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
from market_calendar import is_us_market_session
now = datetime.now(ZoneInfo("America/New_York"))
print("RUN" if (is_us_market_session(now.date()) and 9 <= now.hour < 17) else "SKIP")
PY
)
if [ "$GATE" != "RUN" ]; then
  echo "[$TS] market closed/off-hours — warm skipped"
  exit 0
fi

# First "user:pass" pair from BASIC_AUTH_USERS (strip surrounding quotes + spaces).
RAW=$(grep -E '^BASIC_AUTH_USERS=' "$ENVF" 2>/dev/null | head -1 | cut -d= -f2-)
RAW="${RAW%\"}"; RAW="${RAW#\"}"
RAW="${RAW%\'}"; RAW="${RAW#\'}"
CREDS="${RAW%%,*}"
CREDS="$(echo "$CREDS" | xargs)"

if [ -z "$CREDS" ] || [[ "$CREDS" != *:* ]]; then
  echo "[$TS] ERROR: no valid BASIC_AUTH_USERS pair in $ENVF" >&2
  exit 1
fi

CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
  --user "$CREDS" "http://localhost:8000/api/volume-leaders?force=true")

echo "[$TS] volume-leaders warm -> HTTP $CODE"
[ "$CODE" = "200" ] || { echo "[$TS] WARN: non-200 from warm call" >&2; exit 1; }

# Staggered warm for gainers + reversion-leaders (added Jul 26 2026): neither
# had ANY cron coverage — setup_log showed reversion_hunter=0 rows ever (48
# days) and gainers stopped Jul 21. Both depend on a browser hitting the tab,
# which starves the setup_log sample (incl. the sitting's thin BREAKOUT
# CONFIRMED bucket) independent of how rare the underlying verdict is.
# Alternate per run (not every-run for both) to avoid doubling e2-medium load
# and Yahoo call volume on top of the existing volume-leaders warm.
STATE_FILE=warm_alt_state
STATE=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
if [ "$STATE" = "0" ]; then
  ALT_EP=gainers
  NEXT=1
else
  ALT_EP=reversion-leaders
  NEXT=0
fi
echo "$NEXT" > "$STATE_FILE"

ALT_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
  --user "$CREDS" "http://localhost:8000/api/${ALT_EP}?force=true")
echo "[$TS] ${ALT_EP} warm -> HTTP $ALT_CODE"
[ "$ALT_CODE" = "200" ] || echo "[$TS] WARN: non-200 from ${ALT_EP} warm call" >&2
