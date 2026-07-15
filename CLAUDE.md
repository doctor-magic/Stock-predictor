# Stock Predictor Pro — CLAUDE.md

## ✅ RESTORED Jul 5 2026 — the six features lost in the Jun 7 refactor
History: the Jun 7 refactor was built from a stale api.py, silently dropping May-2026 features that were deployed but never committed (discovered Jul 3, absent from all git history AND the live server). **All restored Jul 5 2026 from the specs below, with enhanced instrumentation (spec v2):** HOD Gap/ATR gate, RVOL slope gate, Power Hour whale alert, merger-pinned filter, FALLING KNIFE logging call, wedge compression thresholds (0.40/0.45/0.50 — now guarded by a synthetic-wedge regression test in test_scanners.py). Also restored: market_id whitelist + top_n/min_confidence clamps (pydantic Field). The frontend lit up automatically — field names were kept identical.

**Instrumentation added with the restoration (why it exists — the old gates left ZERO measurable rows):**
- `setup_log.blocked_reasons` (JSON array) — ALL gates evaluated per row, no short-circuit ("HOD"/"RVOL"/"BETA")
- `setup_log.market_state` (tailwind/headwind/mixed) + `setup_log.vix_state` (VIX state machine) at signal time
- VL logging coverage EXPANDED (never narrow): also logs rows whose setup fired or was gate-blocked, even at verdict HOLD
- Early-close aware windows: `market_calendar.session_close_hour()` (13:00 on Nov 27 / Dec 24) — Power Hour = last hour before ACTUAL close; HOD window = 10:00→actual close
- **DEVELOPING display breaker (pre-registered safety rule Jul 3):** at N≥20 resolved gainers/DEVELOPING with mean ret_5d < −5% → display demotes to WATCH (`db.developing_breaker_tripped()`, 1h cache); the LOG keeps the true verdict
- Gate thresholds (HOD 0.35 etc.) are STARTING POINTS — calibrate only at the next N≥50 instrumented rows; beta gate decision still waits for tracker N≥50

## Quick Reference
```bash
# Restart service
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 "sudo systemctl restart stock-app.service"

# Deploy (CANONICAL — verified pipeline: unittest gate → scp → restart → health(200|401,retry) → frontend disk-hash check)
./deploy.sh                 # full: backend + frontend
./deploy.sh --backend-only  # backend only (skips frontend build/copy)

# ── Manual fallback (only if deploy.sh is unavailable) ──
# Deploy backend (always deploy all 3 together — api.py imports from scanners + db)
scp -i ~/.ssh/gcp_stock_rsa \
  ~/Desktop/Stock-predictor/api.py \
  ~/Desktop/Stock-predictor/scanners.py \
  ~/Desktop/Stock-predictor/db.py \
  elimaoz99@35.239.74.178:/home/elimaoz99/stock_predictor/ && \
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 "sudo systemctl restart stock-app.service && sleep 3 && systemctl is-active stock-app.service"

# Deploy frontend (always copy FULL dist — Vite hashes filenames)
cd ~/Desktop/Stock-predictor/frontend && npm run build && \
scp -r -i ~/.ssh/gcp_stock_rsa dist/ elimaoz99@35.239.74.178:/home/elimaoz99/stock_predictor/ && \
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 \
  "cp -r /home/elimaoz99/stock_predictor/dist/. /home/elimaoz99/stock_predictor/frontend/dist/ && sudo systemctl restart stock-app.service"

# Check logs (live)
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 "sudo journalctl -u stock-app.service -f"

# Force fresh scan
GET /api/volume-leaders?force=true   # or reversion-leaders / gainers

# Run pre_scan manually (server)
cd /home/elimaoz99/stock_predictor && nohup venv/bin/python3 -u pre_scan.py >> pre_scan.log 2>&1 &
```

**Key endpoints:** `/api/volume-leaders` · `/api/reversion-leaders` · `/api/gainers` · `/api/setup-stats` · `/api/falling-knife-stats` · `/api/wedge-scan`

---

## Current Active Configuration
| Parameter | Value | File |
|-----------|-------|------|
| CONFIDENCE_THRESHOLD | 0.70 | core_logic.py |
| SCAN_CONFIDENCE_THRESHOLD | 0.57 | api.py |
| PREMIUM_SCAN_THRESHOLD | 0.65 | api.py |
| _BETA_HIGH_THRESHOLD | 1.5 | api.py |
| HOD gap threshold (`_HOD_GAP_MAX`) | 0.35 | api.py |
| RVOL slope threshold | 0.95 × mean(T-1,T-2), slot guard 270s | api.py |
| Power Hour: pct_from_low | < 2.0% at ET_hour == session_close_hour−1 | api.py |
| DEVELOPING breaker | N≥20 AND mean ret_5d < −5% → display WATCH | db.py |
| MODEL_VERSION | "2026-05_ema_dist_regime" | live_tracker.py |
| Volume Leaders TTL | 1800s | api.py |
| Reversion Hunter TTL | 900s | api.py |
| Macro strip TTL | 300s | api.py |
| FRED dashboard TTL | 21600s | api.py |
| Lev sentiment TTL (`_LEV_SENTIMENT_TTL`) | 300s | scanners.py |
| Sector heatmap TTL (`_SECTOR_HEATMAP_TTL`) | 300s | scanners.py |

---

## Databases & State Files
| File | Location | Purpose | Writer | Reader |
|------|----------|---------|--------|--------|
| `scanner_cache.db` | server + local | Scan results cache (sp500/nasdaq100) | `db.py` | `api.py` |
| `intraday_cache.db` | server | 5m bars for time-of-day RVOL | `fetch_intraday.py` | `scanners.get_tod_rvol_cached()` |
| `setup_log.db` | server | Scanner signal outcome tracking (VL + Rev + Gainers) | `db.setup_log_event()` via api.py | `/api/setup-stats` → `db.get_setup_breakdown()` |
| `falling_knife_log.db` | server | Falling Knife signal outcome tracking | `db.fk_log_event()` via reversion endpoint, 13:00–15:00 ET (call site restored Jul 5 2026 — was lost Jun 7; pipeline verified end-to-end Jul 10 incl. a root-owned-file write bug fixed by chown) | `/api/falling-knife-stats` → `db.get_fk_stats()` |
| `tracker.db` | **server** (was local until Jun 14 2026) | Daily BUY signal log + outcome resolver | `live_tracker.py` cron (20:05 server/IL time Mon–Fri) | `live_tracker.py --report` |
| `fred_cache.json` | server | FRED dashboard disk cache (survives restarts) | `api.py` | `api.py` (startup) |
| `wedge_cache.json` | server | Wedge scan results from pre_scan.py | `pre_scan.py` | `/api/wedge-scan` |
| `macro_state.json` | server | VIX state machine persistence | `api.py` | `api.py` |

---

## Environment Variables
| Variable | Used By | Source File |
|----------|---------|-------------|
| FRED_API_KEY | `api.py` (macro dashboard) | `api_data.env` |
| TELEGRAM_BOT_TOKEN | `api.py`, `pre_scan.py`, `live_tracker.py` | `api_data.env` |
| TELEGRAM_CHAT_ID | `api.py`, `pre_scan.py`, `live_tracker.py` | `api_data.env` |
| BASIC_AUTH_USERS | `api.py` (Basic Auth on endpoints), `live_tracker.py` (sends auth header to `/api/scan`) | `api_data.env` |
| ENABLE_AUTH | `api.py` (enforces auth), `live_tracker.py` (fail-fast warn if true but BASIC_AUTH_USERS unset) | `api_data.env` (default `"true"`) |

`api_data.env` lives in `/home/elimaoz99/stock_predictor/` on server and `~/Desktop/Stock-predictor/` locally. `api.py` self-loads it at startup. `live_tracker.py` (server cron) loads **api_data.env first, then .env** via `os.environ.setdefault()` (Jun 14 2026 fix — loading `.env` first missed `BASIC_AUTH_USERS` and caused a silent 9-day 401 outage). Other local scripts load via `python-dotenv` or manual parse.

---

## Debugging & Inspection

### setup_log.db (scanner signal outcomes)
```bash
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178
cd /home/elimaoz99/stock_predictor

# Recent signals (last 20)
sqlite3 setup_log.db "SELECT source, symbol, date, verdict, ml_confidence, beta_blocked, resolved FROM setup_log ORDER BY log_ts DESC LIMIT 20;"

# Breakdown by verdict + outcome (resolved only)
sqlite3 setup_log.db "SELECT source, verdict, beta_blocked, COUNT(*) as n, ROUND(AVG(ret_5d),2) as mean_5d FROM setup_log WHERE resolved=1 GROUP BY source, verdict, beta_blocked ORDER BY mean_5d DESC;"

# Unresolved (waiting for close data)
sqlite3 setup_log.db "SELECT symbol, date, verdict FROM setup_log WHERE resolved=0;"
```
→ API shortcut: `GET /api/setup-stats`

### falling_knife_log.db (FK signal outcomes)
```bash
# Recent FK events (last 20)
sqlite3 falling_knife_log.db "SELECT symbol, date, price, change_pct, rsi, ph_return, resolved FROM fk_events ORDER BY date DESC LIMIT 20;"

# Mean next-day return (resolved only)
sqlite3 falling_knife_log.db "SELECT COUNT(*) as n, ROUND(AVG(ph_return),2) as mean_ph FROM fk_events WHERE resolved=1;"
```
→ API shortcut: `GET /api/falling-knife-stats`

### Health check after deploy
```bash
# Verify service is active
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 "systemctl is-active stock-app.service"

# Check for ImportError on startup (most common failure after partial deploy)
ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 "sudo journalctl -u stock-app.service -n 30 --no-pager | grep -i 'error\|import\|started'"
```
**Most common failure mode:** deploying `api.py` alone (without `scanners.py` + `db.py`) → `ImportError` on service start → site down. Always deploy all 3.

### Post-Deploy Checklist
- [ ] `systemctl is-active stock-app.service` returns `active`
- [ ] No `ImportError` / `ModuleNotFoundError` in last 30 log lines (especially `scanners` or `db`)
- [ ] `GET /api/health` returns `{"status": "ok"}`
- [ ] `GET /api/volume-leaders` returns 200
- [ ] Force-refresh works: `GET /api/volume-leaders?force=true`
- [ ] If frontend deployed: open stock-predictor.online → verify no JS errors in console + tabs load
- [ ] If scanner signals were sent during testing: verify logging pipeline survived deploy
  ```bash
  sqlite3 setup_log.db "SELECT source, symbol, date, verdict, resolved FROM setup_log ORDER BY log_ts DESC LIMIT 5;"
  ```

---

## Infrastructure
| Property | Value |
|----------|-------|
| Live site | stock-predictor.online |
| GitHub | doctor-magic/Stock-predictor (branch: main) |
| SSH | `ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178` |
| Active dir | `/home/elimaoz99/stock_predictor/` (**NOT** `stock_app/`) |
| Service | `stock-app.service` (systemd, uvicorn port 8000) |
| Sudo | passwordless for `systemctl restart stock-app.service` + `chown` + `chattr` (verified `sudo -n -l` Jun 21 2026; used Jul 10 to fix the root-owned FK db) |
| Static IP | `35.239.74.178` (`stock-app-ip`) — survives Stop/Start |
| Machine | **e2-medium** (2 shared vCPU, 4GB RAM) — ~₪60/month, downgraded Jul 5 2026 and canary-PROVEN (full-session peak 320MB on a heavy red day, Jul 7). Do NOT use e2-small (yfinance spikes to ~1.5GB). |
| GCP credits | expired July 10 2026 — VM now billed; downgrade landed 5 days before the deadline |

### ✅ Downgrade EXECUTED Jul 5 2026 — canary PROVEN by Jul 10 (durable facts kept below)
e2-standard-2 → e2-medium via Console Stop/Edit/Start. Service auto-started (first reboot in 75 days), health 200, crontab + DBs + static IP intact. Day-1 and day-2 canary held: zero OOM, NRestarts=0, full-session `memory.peak` = 320MB on a heavy red day.
- **Swap deliberately NOT created** — user challenged the mandate; live-load data (320MB peak vs 4GB) proved it unnecessary. Do not add one "just in case"; decide from `memory.peak`, not folklore.
- Rollback if ever needed: Console resize back (now paid, no longer free-canary).
- Monitor cmd: `cat /sys/fs/cgroup/system.slice/stock-app.service/memory.peak` + `free -m`.
- The */25 warm cron was gated to 09:00–16:59 ET trading days BEFORE the downgrade (keep gated).
- Expect ~2× slower full scans (shared-core) — scan semaphore + 300s cooldown already protect.
- Cloud Run is NOT an option: SQLite local state + cron jobs + sklearn cold-start make it unsuitable.
- Still open in Console: disable VM Manager (₪3.14/mo) + Network Intelligence Center (₪3.16/mo).

## Stack
FastAPI (`api.py`) + React (`frontend/src/App.jsx`, built with Vite → `frontend/dist/`)

## Architecture (updated Jun 7 2026)
- `api.py` — **1387 lines** — FastAPI endpoints + macro/VIX logic only. Thin routing layer. Imports from `scanners` and `db`.
- `scanners.py` — **520 lines (new Jun 7)** — All scanner helpers: `compute_verdict`, `compute_momentum`, `gainers_verdict`, `detect_falling_wedge`, `classify_regime`, `get_tod_rvol_cached`, `get_intraday_signals`, `get_market_context`, `get_overhead_supply`, `get_vaccel` + their module-level caches. No imports from api.py.
- `db.py` — **316 lines** — SQLite logic: scan cache (original) + FK log functions + setup log functions. `fk_db_init`/`setup_db_init` run at module load. WAL mode on the two high-write logs ONLY (`setup_log.db`, `falling_knife_log.db`); the read-mostly cache DBs (`scanner_cache.db`, `intraday_cache.db`) do NOT use WAL.
- `core_logic.py` — ML model (HistGradientBoostingClassifier, 20 features), CONFIDENCE_THRESHOLD=0.70
- `models.py` — Pydantic models
- `pre_scan.py` — overnight cron (05:00 server/IL time): wedge scan + Telegram alert (in git since Jul 3 2026 — 342 lines, pulled from server; repo == server)
- `fetch_intraday.py` — cron 20:30 server/IL time: downloads 1m bars → resamples to 5m → `intraday_cache.db`
- `watchdog.py` — cron 09:30 server/IL time (added Jul 12 2026): read-only daily health digest → Telegram (pre_scan channel). Alert-only by hard rule — never restarts/fixes/writes.
- `live_tracker.py` — daily BUY signal logger + outcome resolver. **Runs as a server cron (20:05 server/IL time Mon–Fri) since Jun 14 2026** (was a local Mac script). Writes `/home/elimaoz99/stock_predictor/tracker.db`. Calls `/api/scan` with Basic Auth — any auth/endpoint change in api.py must update it too. Usage: `live_tracker.py --log | --report`

**Where to add new code:**
- New scanner gates, verdict logic, intraday analysis, or momentum/mean-reversion logic → `scanners.py`
- New SQLite tables, logging functions, or persistence logic → `db.py`
- New API endpoints, macro/VIX logic, or routing → `api.py`
- New cron scripts or local analysis tools → standalone files (e.g. `pre_scan.py`, `live_tracker.py` pattern)

**Rule:** Never move scanner-related logic back into `api.py`. The refactor boundary is hard.

## 8 Tabs
Single predict | Scanner (with ALMOST BUY) | Daily report | FRED dashboard | Macro score | Volume Leaders | Wedge Scan | Reversion Hunter
> Leumi Options tab removed Jun 18 2026 (friends finished using the calculator). Was a standalone frontend-only `LeumiOptionsView` in App.jsx — no backend/DB/cron. Removal touched App.jsx only (tab button + render conditional + component block + `Calculator` lucide import).

---

## Reversion Hunter — Feature Spec (added May 26 2026)

### Core Engine
- Yahoo screener: `day_losers` (same API pattern as `most_actives`)
- Filter: marketCap ≥ $500M, volume ≥ 2M shares, last trade < 24h, change_pct ≤ -5%
- On-demand ML via `get_prediction(sym, light_mode=True)`, ThreadPoolExecutor(5)
- Cache: `_reversion_cache`, TTL = 900s (15 min), `?force=true` to bypass
- Endpoint: `GET /api/reversion-leaders`
- Backend: `api.py` (endpoint + globals `_reversion_cache`, `_REVERSION_TTL`); helpers in `scanners.py`; logging via `db.setup_log_event()`

### Verdict Tiers
- **DEEP BUY**: ML verdict == BUY AND RSI < 35 AND vwap_gap_pct < -2%
- **POTENTIAL BOUNCE**: 2/3 of above conditions met
- **OVERSOLD**: RSI < 35 only (no ML BUY)
- **WATCH**: default

### API Response Fields
`symbol`, `price`, `change_pct`, `volume`, `rsi`, `vwap`, `vwap_gap_pct`, `ml_signal`, `ml_confidence`, `regime`, `reversion_verdict`, `rvol`, `rvol_quality`, `rvol_alert`

**Gotcha — field names differ from Volume Leaders:**
- `ml_signal` (NOT `ml_verdict`)
- `reversion_verdict` (NOT `verdict`)
- `ml_confidence` is **0–100 scale** (backend does `round(conf * 100, 1)`) — do NOT multiply by 100 in frontend

### RVOL Alert (added Jun 2 2026)
- `rvol_alert = True` when `rvol > 5.0` (extreme intraday volume surge)
- Frontend: red animate-ping dot in Symbol cell, same CSS pattern as Volume Leaders Power Hour alert
- Tooltip: `🚨 RVOL חריג — Nx נפח גבוה במיוחד`
- Threshold 5.0 chosen based on SOC 9.3x incident — catches genuine anomalies without noise

### Entry Rule
Signal = awareness only. Enter ONLY on VWAP bounce confirmation, not immediately on signal.
VWAP is the target — not the entry.

### Volume Leaders — Full Feature List

### Core Engine
- Yahoo screener: top 50 most-active → filtered by mktcap >$200M, vol>0, last trade <24h, avgVol3M×price ≥$2M
- Intraday: VWAP (resets 9:30 ET), time-of-day RVOL (median-based, SQLite cache), ORB Breakout, LIQUID SURGE, VWAP Bounce
- Merger-pinned filter: price range <5% over 10d + vol_ratio≥2.0 → skip (restored Jul 5 2026)
- ML verdict: requires vol_ratio≥1.0 AND price>SMA50 AND confidence≥threshold
- Regime: ADX-14 (Wilder's) × ATR-14 percentile → 9 regimes (ranging/weak_trend/strong_trend × lo/med/hi_vol)
- Score 0–10: Signal(4) + RVOL(2) + RSI(1) + VWAP(1) + day%(1) + Setup(1) + Wedge(0.5)
- SPY/QQQ market context bar (tailwind/headwind/mixed), 2min cache
- Earnings badge: ⚠️ red ≤7d, 📅 yellow ≤14d

### Momentum Gates (added May 22 2026; restored Jul 5 2026 with blocked_reasons instrumentation)
Two filters that suppress BUY setup signals when momentum is exhausted.
Model DNA: **momentum chaser** — gates answer "is fuel still alive?" not "where is the wall?"

**Gate 1 — HOD Gap / ATR-14**
- Active: `is_live=True AND 10 <= ET_hour < 16` (post-ORB, regular session only)
- `hod = regularMarketDayHigh` from Yahoo screener
- `atr14_val` from `_atr_daily_cache` (computed once/day via pandas ewm(alpha=1/14)); resets at midnight ET
- `hod_gap_ratio = (hod - price) / atr14_val`
- Threshold: **0.35** — starting point, calibrate after ~50 resolved trades
- Fires → `setup = None`, `setup_blocked_by = "HOD"`

**Gate 2 — RVOL Slope**
- `_rvol_history`: global `dict[str, deque(maxlen=3)]`, newest-first
- Slot guard: `_SLOT_SEC = 270` — `appendleft` only if last entry >4.5min ago; else update `[0]` in-place (prevents F5 spam)
- Needs 3 readings (~15 min) to activate
- Fires: `RVOL_now < mean(T-1, T-2) * 0.95` → `rvol_trend = "down"` → setup suppressed
- Fires → `setup = None`, `setup_blocked_by = "RVOL"`

**API response fields:** `setup_blocked_by`, `hod_gap_ratio`, `rvol_trend`
**Frontend:** RVOL cell gets inline ▲/▼/→ arrow; SETUP cell shows `— ⊘` with native `title` tooltip in Hebrew

**Calibration (IONQ May 21 2026):**
Entry at 10:43 ET ($58.04, HOD $61, ATR ~$4): ratio 0.74 → blocked (would have missed +$30)
Entry at 13:25 ET ($59.18, RVOL ▼): ratio 0.455 + RVOL down → blocked (avoided -$52)
Net: $0 with gates vs -$22 without.

### Beta Gate (added May 23 2026)
Suppresses ML BUY on high-beta stocks — model is Mean-Reversion on institutional stocks, fails on momentum-driven names.

**OOS validation (backtest_month.py, May 23 2026):**
- Large-Cap (beta ~0.7-1.2): 3/4 BUY signals correct, avg return +11.7%
- High-Beta (beta >1.5): 0/5 BUY signals correct, avg return -10.8%

**Implementation in `get_volume_leaders()`:**
- `_BETA_HIGH_THRESHOLD = 1.5` constant in api.py (near TTL constants)
- SPY downloaded once per call: `yf.download("SPY", period="6mo")` → `spy_returns = pct_change()`
- Per-symbol: `beta = Cov(stock_returns, spy_returns) / Var(spy_returns)` over aligned 6mo window, requires ≥60 bars
- If `beta > 1.5` AND `verdict == "BUY"` → `verdict = "HIGH-BETA"`, `beta_blocked = True`
- All computation inside `try/except: beta = None` — per-symbol failure does not affect others
- SPY download inside `try/except: spy_returns = None` — gate disabled gracefully if SPY unavailable
- `import pandas as pd` + `import numpy as np` declared locally inside `get_volume_leaders()`

**API response additions:** `beta` (float|null), `beta_blocked` (bool)
**Frontend:** SETUP cell shows `β ⊘` in purple (`text-purple-400`) with Hebrew tooltip when `row.beta_blocked === true`
- Consistent with HOD/RVOL `— ⊘` pattern
- Threshold 1.5 is empirical — calibrate via tracker.db after ~50 resolved trades

### Power Hour Whale Alert (added May 27 2026; restored Jul 5 2026 — window now session_close_hour−1, early-close aware)
Concept: institutional players accumulate beaten-down stocks at daily lows during 15:00–16:00 ET.
Validated by: QBTS (May 26), WOLF/PDD/NVTS/QCOM all surged in final 15–20 min on May 27.

**Implementation in `get_volume_leaders()` (`api.py`); helpers in `scanners.py`:**
- `_rvol_history: dict` — global in `api.py`, symbol → `deque(maxlen=3)` of `(rvol_val, timestamp)`, newest-first
- `_SLOT_SEC = 270` — slot guard: `appendleft` only if `now - deque[0].ts >= 270`; else update `[0]` in-place (prevents F5 spam)
- `pct_from_low = (price - regularMarketDayLow) / price * 100` — zero extra API calls (`regularMarketDayLow` already in Yahoo screener quote)
- Time gate: `is_live AND ET_hour == 15` (15:00–15:59 ET only)
- `reversion_alert = time_gate AND pct_from_low < 2.0 AND rvol_trend != "down"`
- RVOL trend needs 3 readings (~15 min); before that `rvol_trend=None` → alert still fires on `pct_from_low` alone
- Imports added at module level: `from collections import deque as _deque` + `import statistics as _stats`

**API response additions:** `pct_from_low` (float|null), `reversion_alert` (bool)

**Frontend (`App.jsx` — VolumeLeadersView):**
- Red `animate-ping` dot (Tailwind) placed inside `flex items-center gap-1.5 whitespace-nowrap` div, after TV badge
- `{row.reversion_alert && (<span className="relative flex h-3 w-3 flex-shrink-0" title="...">...)}`
- Tooltip: `🚨 Power Hour — X.X% מהתחתית, נפח עולה`
- No extra setInterval needed — existing `REFRESH_SECS=300` auto-refresh covers power hour

### Leveraged Sentiment Strip (added Jul 7 2026 — OBSERVATIONAL ONLY, not a gate)
Leveraged-ETF flow as market sentiment: **dollar-volume** ratio short/long — `SOXS:SOXL` (semis) + `SQQQ:TQQQ` (NDX). Born from the ETF-tab discussion: the most-active ETF list is ~90% leveraged mirrors of stocks already scanned, so no tab — just the ratio as context.
- `scanners.get_lev_sentiment()` — one batched yf.download of the 4 fixed tickers, cache `_LEV_SENTIMENT_TTL=300s`, try/except → stale-or-None (must never break a scan). Pure math in `_compute_lev_ratios()` (unit-tested).
- **Dollar volume, NOT share volume** — the pairs trade at wildly different unit prices (Jul 7: SOXL $160.51 vs SOXS $4.91 — share ratio 9:1 "fear" was actually dollar ratio 0.28 dip-buying).
- Rides inside `get_market_context()` as `lev: {semis, qqq}` — no new endpoint, no cron.
- Logged RAW per signal: `setup_log.lev_sent_semis` / `lev_sent_qqq` (idempotent migrations in db.py) alongside market_state/vix_state.
- Frontend: neutral gray `⚖` spans in the VL market-context bar. **NO Fear/Greed labels yet** — baseline is ~0.3 not 1.0 (longs carry more AUM); calibrate labels from the logged distribution after ~2 weeks. Do NOT add display thresholds before that.
- Pre-registered N≥50 question + bucket re-registration rule live in the spec memory (lev-sentiment-spec). Promotion to any gate/filter requires that test to pass — until then display+collect only.

### Sector Heatmap (added Jul 15 2026 — DISPLAY ONLY, not a gate, not logged)
11 SPDR sector ETFs (XLK/XLF/XLE/XLV/XLY/XLP/XLI/XLB/XLU/XLRE/XLC) day %-change as market context ("where the wind blows" — complements the SPY/QQQ tailwind bar + lev strip; NOT a trading list).
- `scanners.get_sector_heatmap()` — one batched `yf.download(period="5d", interval="1d")`, cache `_SECTOR_HEATMAP_TTL=300s`, whole fetch try/except → stale-or-None (a sector failure must never break a scan). Pure math in `_compute_sector_changes()` (unit-tested). During the session the last daily bar is live → change tracks intraday.
- Rides inside `get_market_context()` as `sectors: {XLK: pct, …}` — no new endpoint, no cron.
- **NO setup_log columns, NO gate (spec Jul 13 2026).** If sector data is ever wanted as a logged covariate, that is a new-covariate decision belonging in a milestone bundle — never mid-collection.
- Frontend: CSS-grid strip under the VL market-context bar, sorted by day change; color buckets at ±0.5% / ±1.5%.

---

## Critical Rules (do not revert)

### Model
- CONFIDENCE_THRESHOLD=0.70 in core_logic.py (not 0.65)
- SCAN_CONFIDENCE_THRESHOLD=0.57 (light mode offset)
- SELL→HOLD everywhere — SELL class is broken (positive fwd return in OOS)
- 3-class model (BUY/SELL/HOLD) — do NOT convert to binary
- Features: ema9_dist/ema21_dist/ema50_dist = (Close-EMA)/EMA (normalized, not raw dollars)
- MODEL_VERSION = "2026-05_ema_dist_regime" in live_tracker.py — bump on any material change
- PREMIUM_SCAN_THRESHOLD=0.65 (not 0.57 or 0.70) for 9-stock premium universe

---

### Trading Entry Rules (updated May 27 2026)
- Volume Leaders BUY signal = watchlist alert, NOT immediate entry. Enter on VWAP pullback + bounce.
- Reversion Hunter signal = same. VWAP is the TARGET, not the entry price.
- Entering near HOD on a BUY signal = bad R:R (confirmed: AAL $14.79 vs HOD $14.84 = $0.05 upside vs $0.21 risk)
- On headwind days (SPY+QQQ both below VWAP): Reversion Hunter requires confirmed VWAP bounce — green 5m candle closes above VWAP after retest. "+0.2% above VWAP" at entry is NOT confirmation. (confirmed: PDD May 27, -$20.44)
- One active trade at a time on headwind days. Two simultaneous positions split attention at the critical exit/entry moment. (confirmed: IREN alert missed while managing PDD exit, May 27)
- Do NOT average down. First entry going wrong = exit signal, not add-more signal. (confirmed: CRCL -$79, NVTS -$86 both from averaging down, May 26)

---

### Architecture (Jun 7 2026 — do not revert)
- **Any change to api.py auth or endpoints must also update live_tracker.py** — it runs as a cron on the server (20:05 UTC) and calls `/api/scan` with Basic Auth from `api_data.env`. Silent failures only appear in `tracker_cron.log`. (Jun 14 2026: auth hardening broke the tracker for 9 days before discovery)
- **Deploy api.py + scanners.py + db.py together** — api.py imports both; deploying api.py alone causes ImportError on startup
- **Scanner helpers live in scanners.py** — `classify_regime`, `detect_falling_wedge`, `compute_verdict`, `get_intraday_signals`, `get_market_context` etc. Do NOT move back into api.py.
- **DB logic lives in db.py** — `fk_log_event`, `setup_log_event`, `setup_resolve`, `get_fk_stats`, `get_setup_breakdown`. api.py calls them via `_db.*`.
- **No imports from api.py in scanners.py or db.py** — would create circular imports.
- **All SQLite connections: timeout=30, WAL mode** — setup_log.db and falling_knife_log.db have `PRAGMA journal_mode=WAL` set in their `*_db_init()`. Do not revert to timeout=3 or rollback mode.

### Setup Logging Coverage (Jun 7 2026 — do not narrow)
- **Volume Leaders**: logs all verdicts EXCEPT HOLD and N/A — includes HIGH-BETA, OVEREXTENDED, VOL BREAKOUT
- **Reversion Hunter**: logs DEEP BUY and POTENTIAL BOUNCE
- **Gainers**: logs all verdicts EXCEPT WATCH — includes BREAKOUT CONFIRMED, DEVELOPING, FADE RISK, OVERHEAD WALL
- Narrowing the logged set causes selection bias in `/api/setup-stats` — you'd only measure BUY outcomes and never see if gates blocked winners

### setup_log is FORWARD-ONLY (hard rule, Jul 10 2026 — do not violate)
- **Never backfill feature values into rows older than the feature's deploy date** — no matter how plausible the historical reconstruction. A feature not captured live at signal time does not exist for that row.
- New columns: idempotent `ALTER TABLE ADD COLUMN` with NULL default (the dist_from_sma50 pattern) is the ONLY sanctioned migration. Analyses filter `WHERE <col> IS NOT NULL`.
- Why: the whole pre-registration program (shadow query, lev_sent, sitting addenda) rests on live-captured features; retro-filling silently reintroduces look-ahead risk and mixes measurement regimes.

---

### Volume Leaders
- hist download: period="6mo" (NOT 3mo — needed for 100-bar ATR percentile)
- RVOL uses MEDIAN not mean in `scanners.get_tod_rvol_cached()` (robust to earnings volume spikes)
- `scanners.classify_regime()` uses Wilder's smoothing (alpha=1/N), NOT pandas .ewm() — do not replace
- `scanners.classify_regime()` requires `np.asarray(..., dtype=float).ravel()` on all inputs (yfinance 2.x MultiIndex guard)
- Regime is observational only — no BUY filter until ≥50 resolved signals per regime in tracker.db
- HOD gate (restored Jul 5): threshold `_HOD_GAP_MAX`=0.35, window 10:00→`session_close_hour()` ET, live session only, `_atr_daily_cache` — never inline per-request
- RVOL slope (restored Jul 5): slot guard 270s, deque maxlen=3 — do not remove guard
- Beta gate: `_BETA_HIGH_THRESHOLD = 1.5`, SPY downloaded once per call, beta computed from 6mo hist already available. Do NOT remove try/except isolation. Do NOT make threshold a query param yet (premature — calibrate first)
- Power Hour whale alert (restored Jul 5): `_rvol_history` deque slot guard = 270s — do not remove (prevents F5 spam corrupting slope). `pct_from_low < 2.0` threshold — do not loosen above 3% without evidence. Alert fires on `rvol_trend=None` (warming up) — only suppressed on `"down"`. Time gate: `ET_hour == session_close_hour()−1` (15 on full days, 12 on half-days) — do not extend earlier (pre-power-hour has different dynamics).

---

### Frontend
- Never hardcode http://localhost:8000 — use relative /api/... URLs
- Never remove Google Analytics G-5KHC440K09 from frontend/index.html
- Deploy: always copy FULL dist/ (Vite hashes filenames — index.html alone breaks JS/CSS)
- Do not put JSX inside module-level object literals (Vite/Rolldown parse error)
- TradingView links on all 5 symbol tables (Scanner BUY, ALMOST BUY, Volume Leaders, Wedge Scan, Reversion Hunter): URL = `https://www.tradingview.com/chart/?symbol=${symbol}` — no exchange prefix needed
- Symbol cell wrapper for Yahoo + TV links: use `flex items-center gap-1.5 whitespace-nowrap` — NOT `inline-flex` (inline-flex inside `<td>` renders as block, stacks children vertically)

---

### Security (May 2026 — do not revert)
- task_id format enforcement, scan rate limit + semaphore
- market_id whitelist (`_MARKET_ID_WHITELIST` in api.py → 400) + top_n/min_confidence clamps (pydantic `Field` in models.py → 422) — restored Jul 5 2026
- Basic Auth fails CLOSED (Jul 3 2026, `4667b20`): ENABLE_AUTH=true with empty BASIC_AUTH_USERS → 503 on every request + startup stderr warning. An env-load failure must never silently open the API.

---

### FRED API
- Monthly series: NO frequency/aggregation_method params
- Daily (DGS10, DGS2): frequency=m&aggregation_method=avg — flagged via `"daily": True` in FRED_INDICATOR_META
- NEVER use aggregation_method=eop
- **Fetch must be sequential with `time.sleep(0.5)` between each series** — do NOT use ThreadPoolExecutor (Jun 1 2026)
  - Root cause: FRED rate-limits at ~2 concurrent requests → HTTP 429 → nulls cached for 6 hours silently
- **Disk cache:** `fred_cache.json` in `/home/elimaoz99/stock_predictor/` — **do not delete**
  - Loaded into `_macro_dash_cache` at module startup via `_load_fred_disk_cache()`
  - Saved after every successful full fetch via `_save_fred_disk_cache(data)`
  - Survives service restarts — prevents burst of 11 FRED calls on first request after restart
- **Stale-cache fallback:** if `valid < 4` indicators returned by fetch, serve existing cache instead of overwriting with nulls

---

## Crons (server)
⚠️ **Times are SERVER-LOCAL = Asia/Jerusalem, NOT UTC** (old labels here said "UTC" — wrong; crond uses the system TZ. Proof, Jul 12 2026: the `0 3 * * *` resolver line logs `start ...T00:00:01+00:00` = 03:00 IL). ET-sensitive behavior is guarded inside the scripts (`market_calendar`), not by cron times.
- 03:00 daily → `resolve_setups.py` → setup_log outcome resolver, 50 rows/night (deliberately NOT holiday-guarded — resolves on real historical bars)
- 05:00 daily → `pre_scan.py` → wedge scan → Telegram
- 09:30 daily → `watchdog.py` → read-only health digest → Telegram (`# WATCHDOG-DAILY`; alert-only, never fixes; added Jul 12 2026)
- 14:45 Mon–Fri → `fetch_raw_messages.py`
- 14:50 Mon–Fri → `fetch_clal_48h.py`
- 15:00 Mon–Fri → `generate_report.py` → Telegram
- 20:05 Mon–Fri → `live_tracker.py --log --no-telegram` → tracker.db
- 20:30 Mon–Fri → `fetch_intraday.py` → `intraday_cache.db`
- */25 always → `warm_volume_cache.sh` (internally gated to 09:00–16:59 ET trading days)
- (legacy stock_app: 12:00/18:00 `fetch_24h.py`, 12:05/18:05 `tg_scraper.py`)

## Local Scripts (Mac, ~/Desktop/Stock-predictor/)
- `backtest_month.py` — backtests ML (thresholds: 0.70/0.30). Two universe groups: TICKERS_LARGECAP (40) + TICKERS_HIGHBETA (20). FEATURES use normalized EMA: ema9_dist/ema21_dist/ema50_dist — do NOT revert to raw dollar values.
- `live_tracker.py` — source lives on Mac but **executes as a server cron since Jun 14 2026** (see Architecture). signals table has `beta REAL` column (migration auto-runs on next `--log`). `_batch_regimes()` downloads 6mo (NOT 3mo — required for 100-bar ATR window) + SPY in one batch call. Beta computed per-symbol via `Cov/Var` with `join="inner"` alignment. Telegram shows `⚠β2.3` when beta > 1.5.
- `swing_backtest.py` — walk-forward OOS (--filtered = 9-stock premium)
- `live_tracker.py --log | --report [--no-telegram]`
- `orb_backtest.py` — ORB intraday backtest
- `.env` — API keys (also at ~/Desktop/daily_reports/.env)

## Git State (Jun 7 2026) — all committed, latest commit: `9fe3557`
Everything is in sync: local `~/Desktop/Stock-predictor/` = server `/home/elimaoz99/stock_predictor/` = GitHub main.

**Committed Jun 7 2026 (this session):**
- `scanners.py` — new file, 520 lines, all scanner helpers extracted from api.py
- `db.py` — expanded to 316 lines: FK log + setup log functions + WAL mode
- `api.py` — reduced to 1387 lines; imports from scanners + db
- Setup outcome logger: `setup_log.db`, `/api/setup-stats`, broader logging coverage
- SQLite WAL mode + timeout=30 on all connections
- CLAUDE.md: Quick Reference, Config, Databases, Env Vars sections

**Previously committed (all in main):**
- May–Jun 2026: Wedge Scan tab, SWING/Score columns, SPY/QQQ context, Earnings Calendar, Regime Classification, Premium Scan, Momentum Gates (HOD+RVOL), Beta Gate, Reversion Hunter (Tab 9), TradingView TV links, Power Hour Whale Alert, FRED disk cache, Reversion Hunter RVOL alert, Wedge Scan Touches column, Falling Knife Logger

## Pending actions
- **THE SITTING — Jul 24 2026 or later** (user on vacation Jul 19–23; moved from ~Jul 21). One sitting closes: locked beta-gate query decision (N≥50 REACHED Jul 12: 52 resolved-with-beta — **NO PEEKING before the sitting**), restoration H1/H2 (watch: only 22 resolved BREAKOUT CONFIRMED as of Jul 12 vs min n=15/bucket — thin buckets ⇒ defer, don't loosen), lev_sent bucket re-registration + label calibration (semis PRIMARY — retail agentic-bot confounder on the qqq pair), scanner-health threshold locking, payoff-ratio + CLUSTER-bootstrap addenda (resample trading DAYS, B=10k, seed=42, interpretive only). **CLOSED CONFIRMATORY FAMILY: only the shadow query, H1, H2, and lev buckets may drive code change** — everything else descriptive. Full spec in the handoff Jul 10/12 entries + memory.
- **Code freeze until the sitting** — no new filters/gates/params; collection only.
- **Instrumented-gate calibration:** blocked_reasons/market_state/vix_state collection started Jul 6 2026 — calibrate HOD 0.35 etc. at the next N≥50 resolved instrumented rows (separate clock from the sitting)
- **DEVELOPING breaker:** pre-registered rule armed (N≥20 AND mean<−5% → display demotion); n=11/mean −8.71% as of Jul 3
- **After ~50 resolved signals per regime:** run per-regime precision analysis → Phase 2 regime filter
- **Parked post-sitting candidates (pre-register before use):** overnight-gap covariate (log-only), MAE/MFE outcome columns at resolution
- **Step 3 of refactor (future):** move `get_volume_leaders`, `get_reversion_leaders`, `get_gainers` to `scanners.py` — completes the architecture split
- **Phase 2 infra:** service still runs as User=root (all log DBs elimaoz99-owned since Jul 10); extend NYSE_HOLIDAYS for 2027; v_accel UX
