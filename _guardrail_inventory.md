# Guardrail Inventory — safety net for the CLAUDE.md trim

Built 2026-08-03 at stage 0, by a **full read** of CLAUDE.md (497 lines), not by grep.
Baseline restore point: CLAUDE.md as committed in `007f471` (tree at HEAD `ffdeead`).

**Purpose.** Every `IRON` line below must survive every trim step. `check_guardrails.py`
verifies each quote still appears verbatim in CLAUDE.md (or, after stage 4, in the skill
that legitimately owns it). A missing quote fails the step — the step rolls back.

**This file is NOT loaded into context.** It is tracked in git and read only by the checker
and by a human reviewing a trim.

## Classification

- `IRON` — irreversible failure, silent experiment corruption, or fragile ops that cannot be
  guessed from the code. **Stays in CLAUDE.md. Never moves to a skill** (a skill loads only
  when called; these must be present even for an innocent question).
- `WORKFLOW` — task-specific procedure. May move to a skill with a one-line pointer.
- `UNSURE` — normative-sounding prose that may be description in imperative clothing.
  Goes to the stage-3 report for classification. **Not deleted in stages 1–2.**

## Format

`GR-nnn | CLASS | source section | verbatim quote`

Quotes are exact contiguous substrings. No pipe characters inside quotes.

---

## IRON — research validity / pre-registration

GR-001 | IRON | Restored features → instrumentation | VL logging coverage EXPANDED (never narrow)
GR-002 | IRON | Restored features → instrumentation | at N≥20 resolved gainers/DEVELOPING with mean ret_5d < −5% → display demotes to WATCH
GR-003 | IRON | Restored features → instrumentation | are STARTING POINTS — calibrate only at the next N≥50 instrumented rows
GR-004 | IRON | Reversion Hunter → funnel diagnostics | Log-only — NO DB writes, NO verdict changes
GR-005 | IRON | Reversion Hunter → p(BUY) | Diagnostics only — never a gate.
GR-006 | IRON | Momentum Gates → Gate 1 | Threshold: **0.35** — starting point, calibrate after ~50 resolved trades
GR-007 | IRON | Beta Gate → frontend | Threshold 1.5 is empirical — calibrate via tracker.db after ~50 resolved trades
GR-008 | IRON | Lev Sentiment Strip | Do NOT add display thresholds before that.
GR-009 | IRON | Lev Sentiment Strip | Promotion to any gate/filter requires that test to pass — until then display+collect only.
GR-010 | IRON | Sector Heatmap | a new-covariate decision belonging in a milestone bundle — never mid-collection
GR-011 | IRON | Setup Logging Coverage | Narrowing the logged set causes selection bias
GR-012 | IRON | setup_log FORWARD-ONLY | Never backfill feature values into rows older than the feature's deploy date
GR-013 | IRON | setup_log FORWARD-ONLY | A feature not captured live at signal time does not exist for that row.
GR-014 | IRON | setup_log FORWARD-ONLY | is the ONLY sanctioned migration
GR-015 | IRON | setup_log FORWARD-ONLY | retro-filling silently reintroduces look-ahead risk and mixes measurement regimes
GR-016 | IRON | Volume Leaders rules | Regime is observational only — no BUY filter until ≥50 resolved signals per regime in tracker.db
GR-017 | IRON | Crons → resolver | the forward-only rule (signal-time features) is untouched
GR-018 | IRON | VM downgrade | The */25 warm cron was gated to 09:00–16:59 ET trading days BEFORE the downgrade (keep gated).

## IRON — rules currently trapped inside "Pending actions" (see WARNING below)

GR-019 | IRON | Pending actions | the closed-confirmatory-family rule still stands
GR-020 | IRON | Pending actions | Amendments are additive write-once files — never edit `lev_spec_frozen.json`.
GR-021 | IRON | Pending actions | Changing the budget changes sample composition
GR-022 | IRON | Pending actions | Do NOT loosen bucket definitions.
GR-023 | IRON | Pending actions | EXPLORATORY-ONLY, do not act

## IRON — deploy / ops that cannot be guessed

GR-024 | IRON | Quick Reference | always deploy all 3 together — api.py imports from scanners + db
GR-025 | IRON | Health check | Always deploy all 3.
GR-026 | IRON | Architecture rules | Deploy api.py + scanners.py + db.py together
GR-027 | IRON | Architecture rules | Any change to api.py auth or endpoints must also update live_tracker.py
GR-028 | IRON | Architecture rules | No imports from api.py in scanners.py or db.py
GR-029 | IRON | Architecture rules | All SQLite connections: timeout=30, WAL mode
GR-030 | IRON | Architecture rules | Do not revert to timeout=3 or rollback mode.
GR-031 | IRON | Architecture → where to add code | Never move scanner-related logic back into `api.py`. The refactor boundary is hard.
GR-032 | IRON | Infrastructure | Active dir
GR-033 | IRON | Infrastructure | Do NOT use e2-small (yfinance spikes to ~1.5GB).
GR-034 | IRON | Crons | Times are SERVER-LOCAL = Asia/Jerusalem, NOT UTC
GR-035 | IRON | Crons → watchdog | alert-only, never fixes
GR-036 | IRON | Architecture → watchdog | Alert-only by hard rule — never restarts/fixes/writes.

## IRON — model constants

GR-037 | IRON | Critical Rules → Model | CONFIDENCE_THRESHOLD=0.70 in core_logic.py (not 0.65)
GR-038 | IRON | Critical Rules → Model | SELL→HOLD everywhere — SELL class is broken (positive fwd return in OOS)
GR-039 | IRON | Critical Rules → Model | 3-class model (BUY/SELL/HOLD) — do NOT convert to binary
GR-040 | IRON | Critical Rules → Model | Features: ema9_dist/ema21_dist/ema50_dist = (Close-EMA)/EMA (normalized, not raw dollars)
GR-041 | IRON | Critical Rules → Model | MODEL_VERSION = "2026-05_ema_dist_regime" in live_tracker.py — bump on any material change
GR-042 | IRON | Critical Rules → Model | PREMIUM_SCAN_THRESHOLD=0.65 (not 0.57 or 0.70) for 9-stock premium universe
GR-043 | IRON | Local Scripts | do NOT revert to raw dollar values.

## IRON — scanner internals that look replaceable and are not

GR-044 | IRON | Volume Leaders rules | hist download: period="6mo" (NOT 3mo — needed for 100-bar ATR percentile)
GR-045 | IRON | Volume Leaders rules | RVOL uses MEDIAN not mean
GR-046 | IRON | Volume Leaders rules | uses Wilder's smoothing (alpha=1/N), NOT pandas .ewm() — do not replace
GR-047 | IRON | Volume Leaders rules | requires `np.asarray(..., dtype=float).ravel()` on all inputs (yfinance 2.x MultiIndex guard)
GR-048 | IRON | Volume Leaders rules | never inline per-request
GR-049 | IRON | Volume Leaders rules | slot guard 270s, deque maxlen=3 — do not remove guard
GR-050 | IRON | Volume Leaders rules | Do NOT remove try/except isolation.
GR-051 | IRON | Volume Leaders rules | Do NOT make threshold a query param yet
GR-052 | IRON | Volume Leaders rules | do not loosen above 3% without evidence
GR-053 | IRON | Volume Leaders rules | do not extend earlier (pre-power-hour has different dynamics)
GR-054 | IRON | Local Scripts | downloads 6mo (NOT 3mo — required for 100-bar ATR window)
GR-055 | IRON | Screener guard | do NOT persist it to disk

## IRON — security / auth

GR-056 | IRON | Security | An env-load failure must never silently open the API.
GR-057 | IRON | Login gate | Credentials in **sessionStorage only**
GR-058 | IRON | Login gate | never localStorage, never in a URL or log line
GR-059 | IRON | Login gate | All three are load-bearing:
GR-060 | IRON | Login gate | Do NOT "restore" the WWW-Authenticate header
GR-061 | IRON | Login gate | Auth still fails CLOSED

## IRON — frontend

GR-062 | IRON | Frontend | Never hardcode http://localhost:8000 — use relative /api/... URLs
GR-063 | IRON | Frontend | Never remove Google Analytics G-5KHC440K09 from frontend/index.html
GR-064 | IRON | Frontend | always copy FULL dist/ (Vite hashes filenames — index.html alone breaks JS/CSS)
GR-065 | IRON | Frontend | Do not put JSX inside module-level object literals (Vite/Rolldown parse error)
GR-066 | IRON | Frontend | NOT `inline-flex` (inline-flex inside `<td>` renders as block, stacks children vertically)
GR-067 | IRON | Reversion Hunter → gotcha | do NOT multiply by 100 in frontend

## IRON — FRED

GR-068 | IRON | FRED API | Monthly series: NO frequency/aggregation_method params
GR-069 | IRON | FRED API | NEVER use aggregation_method=eop
GR-070 | IRON | FRED API | do NOT use ThreadPoolExecutor
GR-071 | IRON | FRED API | **do not delete**
GR-072 | IRON | FRED API | serve existing cache instead of overwriting with nulls

## IRON — trading discipline (money is irreversible)

GR-073 | IRON | Reversion Hunter → entry rule | Signal = awareness only. Enter ONLY on VWAP bounce confirmation, not immediately on signal.
GR-074 | IRON | Trading Entry Rules | Volume Leaders BUY signal = watchlist alert, NOT immediate entry.
GR-075 | IRON | Trading Entry Rules | is NOT confirmation
GR-076 | IRON | Trading Entry Rules | One active trade at a time on headwind days.
GR-077 | IRON | Trading Entry Rules | Do NOT average down. First entry going wrong = exit signal, not add-more signal.

---

## WORKFLOW — may move to a skill in stage 4 (with a pointer left in CLAUDE.md)

- `Debugging & Inspection` — setup_log.db + falling_knife_log.db query recipes, health check
- `Post-Deploy Checklist` — the 7-item list
- `Quick Reference` command block — EXCEPT GR-024 and GR-064, which are IRON and stay

## UNSURE — normative-sounding prose, decide in stage 3, do NOT delete before then

- VM downgrade: "Swap deliberately NOT created ... Do not add one 'just in case'; decide from `memory.peak`, not folklore."
- VM downgrade: "Cloud Run is NOT an option: SQLite local state + cron jobs + sklearn cold-start make it unsuitable."
- Lev strip: "try/except → stale-or-None (must never break a scan)"
- Sector heatmap: "whole fetch try/except → stale-or-None (a sector failure must never break a scan)"
- Screener guard: "The hollow leg never re-baselines, so one bad response cannot lower the bar for the next"
- Screener guard: "No same-session cache → the thin payload passes through (thin truth beats nothing)"
- Sector heatmap: "staged in repo — deploy AFTER Jul 24" (looks OBSOLETE — SOXX shipped Jul 24; verify before deleting)

---

## ⚠ WARNING for stage 2 — Pending actions is NOT pure state

GR-019 … GR-023 are IRON rules physically located inside the `## Pending actions` section.
Moving that section wholesale to `PENDING.md` would move five guardrails out of always-loaded
context — the exact failure this inventory exists to prevent.

**Stage 2 must therefore be two commits, in this order:**
1. Extract GR-019…GR-023 into a rules section that stays in CLAUDE.md. Checker green.
2. Only then move the remaining (genuinely state) items to `PENDING.md`. Checker green again.

Doing it in one commit is forbidden: the checker would pass only by accident of ordering.
