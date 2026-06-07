# Stock Predictor Pro — CLAUDE.md

## Infrastructure
- **Live site:** stock-predictor.online
- **GitHub:** doctor-magic/Stock-predictor (branch: main)
- **Server:** GCP Ubuntu VM — `elimaoz99@35.239.74.178`
- **SSH:** `ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178`
- **Active dir:** `/home/elimaoz99/stock_predictor/` (NOT stock_app/)
- **Service:** `stock-app.service` (systemd, uvicorn on port 8000)
- **Restart:** `ssh -i ~/.ssh/gcp_stock_rsa elimaoz99@35.239.74.178 "sudo systemctl restart stock-app.service"`
- **Sudo:** passwordless ONLY for that systemctl restart — nothing else
- **Static IP:** `35.239.74.178` is a named reserved static IP (`stock-app-ip`) — will NOT change on Stop/Start
- **Current machine:** e2-standard-2 (2 vCPU, 8GB RAM) — ~₪174/month
- **GCP credits expire: July 10 2026**

### Downgrade plan (execute July 10 2026)
1. GCP Console → Compute Engine → VM → **Stop**
2. Edit → Machine Type → **e2-medium** (4GB RAM) → Save
3. **Start**
4. Verify: `sudo systemctl status stock-app.service` + open site
- Target cost: ~₪60/month (vs ₪174 now)
- Do NOT use e2-small (2GB) — yfinance batch 50+ tickers × 6mo spikes to ~1.5GB peak
- Also disable: VM Manager (₪3.14/mo) + Network Intelligence Center (₪3.16/mo)
- Cloud Run is NOT an option: SQLite local state + cron jobs + sklearn cold-start make it unsuitable

## Stack
FastAPI (`api.py`) + React (`frontend/src/App.jsx`, built with Vite → `frontend/dist/`)

## Deploy
**Backend:**
```
scp -i ~/.ssh/gcp_stock_rsa <local_api.py> elimaoz99@35.239.74.178:/home/elimaoz99/stock_predictor/api.py
# then restart
```
**Frontend (always copy FULL dist):**
```
npm run build   # in frontend/
scp -r -i ~/.ssh/gcp_stock_rsa dist/ elimaoz99@35.239.74.178:/home/elimaoz99/stock_predictor/
ssh ... "cp -r /home/elimaoz99/stock_predictor/dist/. /home/elimaoz99/stock_predictor/frontend/dist/"
# then restart
```

## Architecture
- `api.py` — FastAPI endpoints, Volume Leaders engine, intraday signals, ML verdict
- `core_logic.py` — ML model (HistGradientBoostingClassifier, 20 features), CONFIDENCE_THRESHOLD=0.70
- `db.py` — SQLite scan cache
- `models.py` — Pydantic models
- `pre_scan.py` — overnight cron (5:00 UTC): wedge scan + Telegram alert (server copy is authoritative — 334 lines)
- `fetch_intraday.py` — cron 20:30 UTC: downloads 1m bars → resamples to 5m → `intraday_cache.db`
- `live_tracker.py` — daily BUY signal logger + outcome resolver. Usage: `python3 live_tracker.py --log | --report`

## 9 Tabs
Single predict | Scanner (with ALMOST BUY) | Daily report | FRED dashboard | Macro score | Leumi Options | Volume Leaders | Wedge Scan | Reversion Hunter

---

## Reversion Hunter — Feature Spec (added May 26 2026)

### Core Engine
- Yahoo screener: `day_losers` (same API pattern as `most_actives`)
- Filter: marketCap ≥ $500M, volume ≥ 2M shares, last trade < 24h, change_pct ≤ -5%
- On-demand ML via `get_prediction(sym, light_mode=True)`, ThreadPoolExecutor(5)
- Cache: `_reversion_cache`, TTL = 900s (15 min), `?force=true` to bypass
- Endpoint: `GET /api/reversion-leaders`
- Backend file: `api.py` (globals `_reversion_cache`, `_REVERSION_TTL`; helper `_clean_rev()`)

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
- Merger-pinned filter: price range <5% over 10d + vol_ratio≥2.0 → skip
- ML verdict: requires vol_ratio≥1.0 AND price>SMA50 AND confidence≥threshold
- Regime: ADX-14 (Wilder's) × ATR-14 percentile → 9 regimes (ranging/weak_trend/strong_trend × lo/med/hi_vol)
- Score 0–10: Signal(4) + RVOL(2) + RSI(1) + VWAP(1) + day%(1) + Setup(1) + Wedge(0.5)
- SPY/QQQ market context bar (tailwind/headwind/mixed), 2min cache
- Earnings badge: ⚠️ red ≤7d, 📅 yellow ≤14d

### Momentum Gates (added May 22 2026)
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

### Power Hour Whale Alert (added May 27 2026)
Concept: institutional players accumulate beaten-down stocks at daily lows during 15:00–16:00 ET.
Validated by: QBTS (May 26), WOLF/PDD/NVTS/QCOM all surged in final 15–20 min on May 27.

**Implementation in `get_volume_leaders()` (`api.py`):**
- `_rvol_history: dict` — global, symbol → `deque(maxlen=3)` of `(rvol_val, timestamp)`, newest-first
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

### Trading Entry Rules (updated May 27 2026)
- Volume Leaders BUY signal = watchlist alert, NOT immediate entry. Enter on VWAP pullback + bounce.
- Reversion Hunter signal = same. VWAP is the TARGET, not the entry price.
- Entering near HOD on a BUY signal = bad R:R (confirmed: AAL $14.79 vs HOD $14.84 = $0.05 upside vs $0.21 risk)
- On headwind days (SPY+QQQ both below VWAP): Reversion Hunter requires confirmed VWAP bounce — green 5m candle closes above VWAP after retest. "+0.2% above VWAP" at entry is NOT confirmation. (confirmed: PDD May 27, -$20.44)
- One active trade at a time on headwind days. Two simultaneous positions split attention at the critical exit/entry moment. (confirmed: IREN alert missed while managing PDD exit, May 27)
- Do NOT average down. First entry going wrong = exit signal, not add-more signal. (confirmed: CRCL -$79, NVTS -$86 both from averaging down, May 26)

### Volume Leaders
- hist download: period="6mo" (NOT 3mo — needed for 100-bar ATR percentile)
- RVOL uses MEDIAN not mean in `_get_tod_rvol_cached()` (robust to earnings volume spikes)
- `_classify_regime()` uses Wilder's smoothing (alpha=1/N), NOT pandas .ewm() — do not replace
- `_classify_regime()` requires `np.asarray(..., dtype=float).ravel()` on all inputs (yfinance 2.x MultiIndex guard)
- Regime is observational only — no BUY filter until ≥50 resolved signals per regime in tracker.db
- HOD gate: threshold 0.35, window 10:00–16:00 ET only, `_atr_daily_cache` — never inline per-request
- RVOL slope: slot guard 270s, deque maxlen=3 — do not remove guard
- Beta gate: `_BETA_HIGH_THRESHOLD = 1.5`, SPY downloaded once per call, beta computed from 6mo hist already available. Do NOT remove try/except isolation. Do NOT make threshold a query param yet (premature — calibrate first)
- Power Hour whale alert: `_rvol_history` deque slot guard = 270s — do not remove (prevents F5 spam corrupting slope). `pct_from_low < 2.0` threshold — do not loosen above 3% without evidence. Alert fires on `rvol_trend=None` (warming up) — only suppressed on `"down"`. Time gate strictly `ET_hour == 15` — do not extend to 14:xx (pre-power-hour has different dynamics).

### Frontend
- Never hardcode http://localhost:8000 — use relative /api/... URLs
- Never remove Google Analytics G-5KHC440K09 from frontend/index.html
- Deploy: always copy FULL dist/ (Vite hashes filenames — index.html alone breaks JS/CSS)
- Do not put JSX inside module-level object literals (Vite/Rolldown parse error)
- TradingView links on all 5 symbol tables (Scanner BUY, ALMOST BUY, Volume Leaders, Wedge Scan, Reversion Hunter): URL = `https://www.tradingview.com/chart/?symbol=${symbol}` — no exchange prefix needed
- Symbol cell wrapper for Yahoo + TV links: use `flex items-center gap-1.5 whitespace-nowrap` — NOT `inline-flex` (inline-flex inside `<td>` renders as block, stacks children vertically)

### Security (May 2026 — do not revert)
- market_id whitelist, top_n clamp, min_confidence clamp, task_id format enforcement, scan semaphore

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
- 05:00 UTC daily → `pre_scan.py` → wedge scan → Telegram
- 14:45 UTC → `fetch_raw_messages.py`
- 14:50 UTC → `fetch_clal_48h.py`
- 15:00 UTC → `generate_report.py` → Telegram
- 20:30 UTC Mon-Fri → `fetch_intraday.py` → `intraday_cache.db`

## Local Scripts (Mac, ~/Desktop/Stock-predictor/)
- `backtest_month.py` — backtests ML (thresholds: 0.70/0.30). Two universe groups: TICKERS_LARGECAP (40) + TICKERS_HIGHBETA (20). FEATURES use normalized EMA: ema9_dist/ema21_dist/ema50_dist — do NOT revert to raw dollar values.
- `live_tracker.py` — signals table has `beta REAL` column (migration auto-runs on next `--log`). `_batch_regimes()` downloads 6mo (NOT 3mo — required for 100-bar ATR window) + SPY in one batch call. Beta computed per-symbol via `Cov/Var` with `join="inner"` alignment. Telegram shows `⚠β2.3` when beta > 1.5.
- `swing_backtest.py` — walk-forward OOS (--filtered = 9-stock premium)
- `live_tracker.py --log | --report [--no-telegram]`
- `orb_backtest.py` — ORB intraday backtest
- `.env` — API keys (also at ~/Desktop/daily_reports/.env)

## Git State (May 28 2026)
All features deployed to server but NOT committed to GitHub:
- Wedge Scan tab, SWING column, Score column, SPY/QQQ context
- Earnings Calendar badge, Regime Classification
- Premium Scan mode, model_version tracking
- Merger-pinned filter, high-risk badge (pre_scan.py)
- Momentum Gates: HOD/ATR + RVOL slope (May 22 2026)
- Beta Gate: rolling beta vs SPY, HIGH-BETA verdict, β ⊘ UI indicator (May 23 2026)
- backtest_month.py: EMA normalization fix + two-group universe (May 23 2026)
- live_tracker.py: beta column in DB, _batch_regimes() 6mo+SPY batch, ⚠β Telegram tag (May 24 2026)
- Reversion Hunter tab (Tab 9): /api/reversion-leaders endpoint, ReversionView component (May 26 2026)
- TradingView TV links on all 5 symbol tables (May 26 2026)
- **Power Hour Whale Alert: `_rvol_history` deque + `pct_from_low` + `reversion_alert` in api.py; 🚨 animate-ping badge in VolumeLeadersView (May 27 2026)**
- **FRED disk cache + sequential fetch fix (Jun 1 2026):** `_load_fred_disk_cache()` / `_save_fred_disk_cache()` added to api.py; `ThreadPoolExecutor` removed from `get_macro_dashboard()`; `"daily": True` on DGS10/DGS2 in FRED_INDICATOR_META; `fred_cache.json` on server
- **Reversion Hunter RVOL alert (Jun 2 2026):** `rvol_alert = bool((rvol or 0) > 5.0)` added to reversion endpoint; red animate-ping dot in ReversionView Symbol cell when `rvol_alert=true`
- **Wedge Scan Touches column (Jun 1 2026):** `upper_touches`/`lower_touches` added to pre_scan.py return + scan; Touches column in WedgeScanView (green ≥5, yellow =4, gray =3)
- **Reversion Hunter Price column (Jun 1 2026):** `price` already in API, Price column added to ReversionView table

## Pending actions
- **July 10 2026:** VM downgrade e2-standard-2 → e2-medium (see Infrastructure section)
- **After ~50 resolved signals with beta IS NOT NULL:** run per-beta SQL analysis in tracker.db:
  `SELECT CASE WHEN beta > 1.5 THEN 'high' ELSE 'normal' END, AVG(hit), COUNT(*) FROM signals WHERE beta IS NOT NULL GROUP BY 1`
- **After ~50 resolved signals per regime:** run per-regime precision analysis → Phase 2 regime filter
