#!/bin/bash
# pull_caches_local.sh — on-demand, ONE-WAY server→Mac sync of the display/cache
# DBs so the UI can run locally against fresh data.
#
# Deliberately NOT part of pull_backup.py: that system backs up only the
# unrebuildable research DBs (~350KB/day, 365-day rotation). These caches are
# rebuildable and ~20MB — mixing them in would break its size assumptions.
#
# Uses python3's sqlite3.backup() and NOT the sqlite3 CLI: the server does not
# have the CLI installed. .backup() is the online-backup API, so it produces a
# consistent copy even if a cron is mid-write (WAL + busy timeout apply).
#
# ⚠️ NEVER copy the other way. tracker.db pushed Mac→server destroyed server
# data once already (Jun 14 2026). This script has no push path on purpose.
set -euo pipefail

KEY=~/.ssh/gcp_stock_rsa
SERVER=elimaoz99@35.239.74.178
RDIR=/home/elimaoz99/stock_predictor
LDIR="$HOME/Desktop/Stock-predictor"

ssh -o ConnectTimeout=15 -o BatchMode=yes -i "$KEY" "$SERVER" "
  set -e
  rm -rf /tmp/local_sync && mkdir /tmp/local_sync
  cd $RDIR
  venv/bin/python3 - <<'PY'
import sqlite3, os
DBS = ['scanner_cache.db','intraday_cache.db','r1_price_cache.db',
       'setup_log.db','tracker.db','falling_knife_log.db','positions.db']
for f in DBS:
    if not os.path.exists(f):
        print('  skip (missing):', f); continue
    src = sqlite3.connect(f'file:{f}?mode=ro', uri=True, timeout=30)
    dst = sqlite3.connect(os.path.join('/tmp/local_sync', f))
    with dst:
        src.backup(dst)
    dst.close(); src.close()
PY
"
scp -q -o ConnectTimeout=15 -o BatchMode=yes -i "$KEY" "$SERVER:/tmp/local_sync/*.db" "$LDIR/"
ssh -o ConnectTimeout=15 -o BatchMode=yes -i "$KEY" "$SERVER" "rm -rf /tmp/local_sync"

echo "synced $(date '+%Y-%m-%d %H:%M'):"
for f in scanner_cache intraday_cache r1_price_cache setup_log tracker falling_knife_log positions; do
  [ -f "$LDIR/$f.db" ] && ls -lh "$LDIR/$f.db" | awk '{printf "  %-24s %s\n", $9, $5}'
done
