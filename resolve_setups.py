#!/usr/bin/env python3
"""Nightly setup_log resolver — bounded, throttled forward-return resolution.

Decoupled from /api/setup-stats (which is now read-only) to keep the yfinance
burst off the web-request path. Runs as a server cron at 03:00 UTC (US market
deep-closed, after the 20:05/20:30 UTC collection crons).

  cron: 0 3 * * * cd /home/elimaoz99/stock_predictor && \
        venv/bin/python3 -u resolve_setups.py >> resolve_setups.log 2>&1

max_rows=50 → drains any backlog over a few nights without bursting Yahoo's
IP rate-limit. See db.setup_resolve() for the per-row throttle + logging.
"""
import sys
from datetime import datetime, timezone

import db

if __name__ == "__main__":
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"[resolve_setups] start {datetime.now(timezone.utc).isoformat()} max_rows={max_rows}",
          flush=True)
    db.setup_resolve(max_rows=max_rows)
    print(f"[resolve_setups] done {datetime.now(timezone.utc).isoformat()}", flush=True)
