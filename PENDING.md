# Pending — open research questions and tasks

Not auto-loaded. Load with `@PENDING.md` when working on the research.
Rules live in `CLAUDE.md` → *Critical Rules → Research discipline*; this file holds only what is open.

## The rule for this file

**Every item declares a trigger. An item with neither of these gets swept.**

- `TRIGGER:` — a condition plus **where to check it**. Never expires. This is how a
  pre-registration waits: it is data that opens it, not the calendar.
- `REVIEW:` — a date. If it passes with no movement, the item is re-decided or dropped.

Why not "older than 30 days → archive": most items here are count-triggered
pre-registrations that are *supposed* to sit for months. A calendar sweep would delete
exactly the commitments the whole program rests on — the same category error as the
drought criterion deferred in July as mis-specified. The calendar sweeps vague ideas,
never triggers.

**Do not restate live counts here.** A trigger names the condition and the command that
answers it. A number copied into prose is stale the day after.

Closing an item: move it to `## Archive` with a one-line outcome. Append-only — never delete.

---

## Open — count-triggered (pre-registrations; these do not expire)

- **Stage B outcome test — lev buckets.**
  Boundary frozen at semis 0.37 (`lev_spec_frozen.json`); eligibility floored at 2026-07-27
  by `lev_spec_amendment_1.json`.
  `TRIGGER:` N ≥ 50 eligible resolved rows — check with `lev_sitting.py --unblind`, which
  refuses below 50 *before* selecting any outcome column, so running it stays blind.

- **Instrumented-gate calibration.**
  `blocked_reasons` / `market_state` / `vix_state` collection started Jul 6 2026. Calibrate
  HOD 0.35 and friends only at the trigger — separate clock from the sitting.
  `TRIGGER:` N ≥ 50 resolved rows carrying the instrumentation columns — query `setup_log.db`
  for resolved rows `WHERE blocked_reasons IS NOT NULL`.

- **DEVELOPING display breaker.**
  Pre-registered rule already armed in code (`db.developing_breaker_tripped()`): display
  demotes to WATCH when it fires. Nothing to do but let it accrue.
  `TRIGGER:` N ≥ 20 resolved gainers/DEVELOPING rows — the code evaluates this itself.

- **Per-regime precision → Phase 2 regime filter.**
  `TRIGGER:` ≥ 50 resolved signals **per regime** in `tracker.db` (filter `WHERE regime IS NOT NULL`).

- **H1/H2 re-run.**
  Deferred at the sitting for thin buckets. Must re-run under the post-Jul-26 sample
  composition only.
  `TRIGGER:` resolved BREAKOUT CONFIRMED reaches n ≥ 15 **per bucket**.

- **Question B — walk-forward model retraining.**
  `TRIGGER:` N ≥ 200 resolved `tracker.db` signals with a consistent `model_version`
  (`live_tracker.py --report` gives the resolved count).

## Open — date-reviewed

- **Prediction-budget question.** Funnel now shows in-session coverage is not the binding
  constraint; the falling-knife downgrade ate the first ML survivor. Decide nothing before
  the evidence is in, and pre-register before changing coverage.
  `REVIEW:` 2026-08-10 — read the `fk_downgrade` counter across the Aug 3–7 week
  (`journalctl -u stock-app.service | grep reversion-funnel`).

- **Third off-VM copy of `setup_log.db`.** The cohort now lives in two domains (VM + Mac).
  GCS with versioning is the right target; it is a cost + setup decision, not a copy.
  `REVIEW:` 2026-08-17.

- **CLAUDE.md trim, stages 3–4.** Stage 3 = classify every section
  (`KEEP_GOTCHA / SKILL / DERIVED_DELETE / NARRATIVE_DELETE / STATE_DELETE`), report only.
  Stage 4 = execute approved cuts + extract the deploy/debug skills.
  `REVIEW:` 2026-08-17. Stopping mid-way is a legitimate outcome.

- **Repo visibility.** Public, verified 2026-08-02. No open-source intent; methodology and
  thresholds are published. Flip to private after checking history for secrets.
  `REVIEW:` 2026-08-17.

- **Lev display labels** around semis 0.37 — unlocked at the sitting, display-only frontend
  change. `REVIEW:` 2026-09-01.

- **NYSE_HOLIDAYS for 2027** must be extended in `market_calendar.py` before year end.
  `REVIEW:` 2026-12-01.

- **Housekeeping — server `frontend/dist/assets`** holds ~83 old JS bundles / 58MB;
  `deploy.sh` copies the full dist and never prunes. Harmless at current disk use but
  unbounded. Also still open in GCP Console: disable VM Manager + Network Intelligence Center.
  `REVIEW:` 2026-09-01.

- **Phase 2 infra.** Service still runs as `User=root` (all log DBs are elimaoz99-owned since
  Jul 10). v_accel UX. `REVIEW:` 2026-10-01.

- **Step 3 of the refactor.** Move `get_volume_leaders`, `get_reversion_leaders`, `get_gainers`
  into `scanners.py` to complete the architecture split. `REVIEW:` 2026-10-01.

## Parked — needs a pre-registered question before it may be acted on

These are observations, not tasks. Acting on one without pre-registering it first violates
*Research discipline* in `CLAUDE.md`.

- BREAKOUT CONFIRMED showed a negative median net ret_5d across the resolved rows available
  at the sitting. Exploratory only.
- Overnight-gap covariate (log-only).
- MAE/MFE outcome columns captured at resolution.

## Archive

- **THE SITTING — closed Jul 24 2026.** Beta-gate status quo (no extension to `/api/scan`);
  H1/H2 deferred on thin buckets; lev boundary frozen at semis 0.37; scanner-health criterion
  deferred as mis-specified, watchdog check #6 shipped instead. Code freeze lifted.
- **Stage B sample reset — locked blind Jul 31 2026.** Jul-26 warm-cron fix is a sample
  composition break; eligibility floored at 2026-07-27 via `lev_spec_amendment_1.json`.
- **Track-1 basis gap — closed Jul 30 2026** (`e3cceff`): both resolvers capture forward-SPY
  at resolution, so the dual basis is live for rows resolved from that date on.
