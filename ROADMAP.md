# ROADMAP — Operational Resilience

**Status:** PLANNED **Date:** 2026-06-14
**Rule:** Items migrate to `CLAUDE.md` (current reality) only when shipped AND verified in production. This file shrinks as phases land; it is a roadmap, not an archive.
**Why these decisions (the reasoning):** lives in the design session, not here. This file captures decisions + acceptance criteria only — by design, to avoid becoming a drift surface.

**Root principle driving all phases — kill Silent Degradation:**
> A script must ASSERT its preconditions explicitly (is it a trading day? has the market closed? did the API return 200?) rather than infer health from absence-of-exception. Every failure we traced — the 9-day 401, 0-signals ambiguity, holiday-empty data, winter-DST early run — is the same disease: *the run completed, but the result was degenerate, silently.*

---

## 🟢 Phase 0 — Safety net `[DONE]`
**Done when:** `test_scanners.py` runs 46 local `unittest` tests in <1ms, covering every edge case, epsilon boundary, and precedence trap of `compute_verdict`, `compute_momentum`, and `gainers_verdict`.
**Shipped:** 2026-06-14. Pure functions only; zero dependencies.

---

## 🟡 Phase 1 — `deploy.sh`
**Done when:** a single command on the Mac performs, as one atomic chain:
1. local `unittest` run as a **blocking gate** (abort deploy on any failure)
2. unified `scp` of `api.py` + `scanners.py` + `db.py` **together** (never api.py alone → ImportError)
3. `systemctl restart stock-app.service`
4. backend liveness: `curl localhost:8000/api/...` accepting **200 OR 401** as alive (auth-aware), with **sleep + retry** for the ~3-5s sklearn cold-start
5. frontend disk check: extract the Vite hash from the **local** freshly-built `dist/index.html`, then `test -f` the matching `frontend/dist/assets/index-[HASH].js` on the server (the ACTIVE dir) — guards against the May 10 partial-copy 404.

---

## 🟡 Phase 2 — Honest scripts
**Done when:** `live_tracker.py` stops swallowing exceptions in `fetch_scan` (current `except Exception: continue` → exit 0 even on 401). Specifically:
- distinguishes **API failure** (401 / network / HTTPError → `exit 1`) from **0 legitimate signals** (`exit 0`)
- explicit trading-day check (e.g. "does SPY have a bar dated today?") to recognize market holidays as a legitimate "ran, market closed" state → `exit 0` + valid heartbeat ping
- never infers health from empty data.

---

## 🟡 Phase 3 — Cron monitoring
**Done when:** the two core server crons (`pre_scan.py`, `live_tracker.py`) are:
- wrapped in a **shell-level `||` gate** that fires an immediate Telegram alert ONLY on `exit != 0` (catches "ran-and-failed")
- wired to an **external dead-man's switch** (e.g. Healthchecks.io) configured to the trading schedule (Mon–Fri, ~20 min grace) — and which **still receives a success ping on market holidays** (so it does not false-alarm on holiday Mondays).

---

## 🟡 Phase 4 — Uptime + CRON_TZ
**Done when:**
- a continuous **24/7 external uptime monitor** hits the public Live URL (`stock-predictor.online`) to catch host death OUTSIDE the trading schedule (e.g. the OOM-kills expected after the July e2-medium downgrade) — this is host-liveness, separate from job-liveness.
- all market-dependent server crons use `CRON_TZ=America/New_York` (scheduled at 16:05 ET, not fixed 20:05 UTC) — neutralizing the DST drift that would otherwise run `live_tracker` ~55 min before the close every winter.

---

## 🟡 Phase 5 — Group B extraction
**Done when:** `hod_gate`, `beta_gate`, and `rvol_slope` are extracted from `api.py`'s inline `get_volume_leaders()` into **pure functions** in `scanners.py`, stabilized under unit tests in `test_scanners.py` (decision pure; the stateful deque/slot-guard and ATR/beta computation stay in api.py), and the same pure pattern is applied to the report pipeline.

---

### Notes
- **FRED is the exemplar, not a TODO.** Its sequential fetch + `fred_cache.json` disk cache + `valid < 4 → serve stale` guard is already the "honest, degradation-aware" pattern the other crons should copy.
- **Risk order is deliberate.** Phases 1–2 come first: they prevent recurrence of the 9-day 401 silent failure and the May 10 frontend outage — the two incidents that actually happened.
