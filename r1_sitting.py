#!/usr/bin/env python3
"""r1_sitting.py — R1: does the actionable BUY tier beat its baselines?

The confirmatory test the program has never run: tracker BUY signals at
confidence >= 0.70 are the only thing the UI presents as actionable, and their
value has never been measured against a baseline that costs nothing to follow.

Same two-clock discipline as lev_sitting.py (freeze blind, unblind once):

  STAGE A (blind — covariates only, safe to run any time):
      python3 r1_sitting.py                 # preview: cohort counts + universe size
      python3 r1_sitting.py --freeze        # THE SITTING: writes r1_spec_frozen.json
    Reads ONLY (date_logged, sym, confidence, model_version) from tracker.db and
    the constituent lists from core_logic. NEVER an outcome column. Freezes the
    universe SNAPSHOT and the baseline draw seed, so the random arm cannot be
    re-rolled until it looks weak.

  PRICE CACHE (blind — prices are not outcomes of ours, but fetch before unblind
  so the one-shot run cannot die halfway on a Yahoo rate limit):
      python3 r1_sitting.py --fetch-cache

  STAGE B (unblind — guarded, one shot):
      python3 r1_sitting.py --unblind
    Refuses unless (1) r1_spec_frozen.json exists AND was written by --freeze,
    (2) eligible-row COUNT >= MIN_N_ROWS, (3) distinct-trading-day COUNT >=
    MIN_N_DAYS. Both counts run BEFORE any outcome column is selected.

  SELF-TEST (synthetic DBs, verifies the pipeline recovers a planted effect):
      python3 r1_sitting.py --selftest

WHY A DAY GATE AND NOT JUST A ROW GATE (pre-registered, do not remove):
the inference is a day-cluster bootstrap, so the effective sample size is the
number of distinct trading days, not the number of rows. 24 rows across 9 days
is a 9-cluster sample wearing a 24-row costume. Both gates must pass.

SANCTIONED RETRO-COMPUTATION (declared up front, printed at every unblind):
`outcomes.spy_fwd_ret` was added Jul 30 2026 and is NULL on the entire existing
>=0.70 cohort, and the B2 arm has no 10-day outcome stored at all. This script
therefore computes forward returns for the SPY and baseline legs at analysis
time. That is explicitly allowed: they are OUTCOME columns, not signal-time
features, and CLAUDE.md's forward-only rule carves them out. Nothing is written
back to any database. The model arm's own return is never recomputed — it is
read from `outcomes.fwd_ret` exactly as the tracker resolved it.

Pre-registered constants below are FROZEN — do not edit after --freeze has run.
"""
import argparse
import json
import os
import random
import sqlite3
import statistics
import sys
import tempfile
from datetime import date, datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Pre-registered constants (locked at the R1 sitting) ────────────────────
TRACKER_DB_DEFAULT = os.path.join(_HERE, "tracker.db")
SETUP_DB_DEFAULT   = os.path.join(_HERE, "setup_log.db")
SPEC_PATH          = os.path.join(_HERE, "r1_spec_frozen.json")
AMEND_PATH         = os.path.join(_HERE, "r1_spec_amendment_1.json")
PRICE_CACHE_PATH   = os.path.join(_HERE, "r1_price_cache.db")

CONF_FLOOR     = 0.70      # the actionable tier — what the UI calls a BUY
MODEL_VERSION  = "2026-05_ema_dist_regime"   # consistent-version rows only
FORWARD_DAYS   = 10        # trading days — MUST match live_tracker.FORWARD_DAYS
HIT_THRESHOLD  = 0.03      # MUST match live_tracker.HIT_THRESHOLD
COMMISSION_PCT = 0.16      # round trip, 0.08% per side
# The two gates are sized to bind at roughly the SAME time, on purpose. The
# inference is day-clustered, so 25 days is the statistically meaningful bar;
# at the cohort's observed ~2.5 rows per signal day, 25 days yields ~62 rows.
# A row gate of 80 would push the unblind 3-4 months past the day gate while
# adding little, and would make the day gate decorative. Neither gate should be
# decoration. Set BEFORE the sitting, with no outcome ever read — moving it
# after the freeze would be indistinguishable from moving the goalposts.
MIN_N_ROWS     = 60
MIN_N_DAYS     = 25        # cluster-bootstrap effective N — see docstring
BOOT_B         = 10_000
BOOT_SEED      = 42
B0_DRAWS       = 20        # random symbols drawn per signal date
B2_VOL_RATIO   = 2.0       # the "high RVOL, no ML opinion" screener baseline

# Pre-registered SECONDARY STRATUM (decided at the sitting, never a pass rule).
# The tracker pools the whole BUY tier, so the 0.57-0.70 band accrues far faster
# than the actionable tier. Reporting it alongside costs nothing and wastes no
# data; letting it into the pass rule would answer a question about ALMOST BUY
# while wearing the product claim's name. It is reported, never decisive.
SECONDARY_STRATUM_FLOOR = 0.57

# PRIMARY endpoint — one number, one CI, one pass rule.
#   median over cohort rows of  alpha_net = (fwd_ret - spy_ret) * 100 - COMMISSION_PCT
#   PASS iff the lower bound of its 95% day-cluster bootstrap CI is > 0.
# Everything else in the output is SECONDARY and descriptive: it may not be
# used to declare success, and may only seed a new pre-registered question.
PRIMARY_RULE = ("PASS iff lower bound of 95% day-cluster CI on median alpha_net > 0; "
                "arms B0/B2 and precision are SECONDARY, descriptive only")


# ── Trading-day arithmetic — identical to live_tracker.resolve_outcomes ────
def _holidays():
    # ALL covered years. Changed Aug 5 2026, AFTER the freeze, deliberately:
    # this function's contract is "identical to live_tracker.resolve_outcomes",
    # and that now uses the full table, so staying on 2026 would have made R1's
    # exit dates disagree with the very resolver that produced the outcomes it
    # reads. R1 unblinds around Dec 2026 and its 10-day windows run into 2027,
    # where an unlisted holiday is silently counted as a trading day.
    # This touches no frozen value — not the cohort, gates, rule, seed or
    # universe — and reads no outcome column. r1_spec_frozen.json is unchanged.
    from market_calendar import NYSE_HOLIDAYS
    return sorted(NYSE_HOLIDAYS)


def exit_date_for(entry_day: str) -> str:
    """+FORWARD_DAYS trading days, roll forward — the tracker's own convention."""
    import numpy as np
    return str(np.busday_offset(entry_day, FORWARD_DAYS, roll="forward",
                                holidays=_holidays()))


# ── Stage A: cohort + universe snapshot, covariates only ───────────────────
def frozen_universe():
    """Constituent snapshot taken AT THE SITTING and stored in the spec.

    The lists load live from Wikipedia, so an unfrozen universe would drift
    between the sitting and the unblind and quietly change the random arm.
    """
    from core_logic import load_sp500, load_nasdaq100
    syms = set(load_sp500().values()) | set(load_nasdaq100().values())
    return sorted(s for s in syms if s and s.isascii())


def cohort_counts(tracker_db, sample_start=None):
    """Row + distinct-day counts for the eligible cohort. No outcome column."""
    con = sqlite3.connect(f"file:{tracker_db}?mode=ro", uri=True)
    where = eligible_where(sample_start)
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM signals s JOIN outcomes o ON o.signal_id = s.id "
        f"WHERE {where}").fetchone()[0]
    n_days = con.execute(
        f"SELECT COUNT(DISTINCT s.date_logged) FROM signals s "
        f"JOIN outcomes o ON o.signal_id = s.id WHERE {where}").fetchone()[0]
    n_open = con.execute(
        f"SELECT COUNT(*) FROM signals s WHERE s.confidence >= {CONF_FLOOR} "
        f"AND s.model_version = '{MODEL_VERSION}' AND s.resolved = 0").fetchone()[0]
    con.close()
    return {"n_rows": n_rows, "n_days": n_days, "n_unresolved_in_tier": n_open}


ELIGIBLE_WHERE = (
    f"s.confidence >= {CONF_FLOOR} AND s.model_version = '{MODEL_VERSION}' "
    f"AND o.fwd_ret IS NOT NULL AND s.entry_price > 0"
)


def eligible_where(sample_start=None):
    """ELIGIBLE_WHERE, optionally floored at a pre-registered sample start date."""
    if not sample_start:
        return ELIGIBLE_WHERE
    return f"{ELIGIBLE_WHERE} AND s.date_logged >= '{sample_start}'"


def stage_a(tracker_db, freeze):
    counts = cohort_counts(tracker_db)
    print(f"[stage A] cohort conf>={CONF_FLOOR} version={MODEL_VERSION}")
    print(f"[stage A]   resolved rows : {counts['n_rows']:>4}  (gate {MIN_N_ROWS})")
    print(f"[stage A]   distinct days : {counts['n_days']:>4}  (gate {MIN_N_DAYS})")
    print(f"[stage A]   still open    : {counts['n_unresolved_in_tier']:>4}")
    if not freeze:
        print("[stage A] PREVIEW ONLY — nothing written. Use --freeze at the sitting.")
        return
    if os.path.exists(SPEC_PATH):
        sys.exit(f"REFUSING: {SPEC_PATH} already exists — the spec is frozen once. "
                 "Delete it manually only if the sitting itself is being redone.")
    universe = frozen_universe()
    if len(universe) < 400:
        sys.exit(f"REFUSING to freeze: universe snapshot only {len(universe)} symbols — "
                 "a Wikipedia fetch failed and PRESET_STOCKS fell back. Retry.")
    spec = {
        "question": "R1 — do conf>=0.70 tracker BUY signals beat their baselines "
                    "net of cost on a 10-trading-day horizon?",
        "frozen": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "cohort": {"conf_floor": CONF_FLOOR, "model_version": MODEL_VERSION,
                   "forward_days": FORWARD_DAYS, "entry_basis": "signal price (entry_price)",
                   "exit_basis": f"close at +{FORWARD_DAYS} trading days, roll forward"},
        "gates": {"min_n_rows": MIN_N_ROWS, "min_n_days": MIN_N_DAYS},
        "commission_pct": COMMISSION_PCT,
        "hit_threshold": HIT_THRESHOLD,
        "primary_endpoint": "median alpha_net = (fwd_ret - spy_ret)*100 - commission",
        "primary_rule": PRIMARY_RULE,
        "secondary_stratum": {
            "conf_floor": SECONDARY_STRATUM_FLOOR,
            "role": "reported alongside the primary, NEVER part of the pass rule",
            "why": "the tracker pools the full BUY tier; the 0.57-0.70 band accrues "
                   "faster but is not what the UI calls actionable"},
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "resample": "trading_days"},
        "arms": {
            "MODEL": "cohort rows, return read from outcomes.fwd_ret as resolved",
            "B1_SPY": "same windows — implicit in alpha; alpha_net > 0 IS beating SPY",
            "B0_RANDOM": f"{B0_DRAWS} symbols/date from the frozen universe, long-only, "
                         "same entry date and horizon, per-date seeded draw",
            "B2_RVOL_NO_ML": f"setup_log volume_leaders rows, vol_ratio >= {B2_VOL_RATIO}, "
                             f"ml_signal != BUY, forward return recomputed at "
                             f"{FORWARD_DAYS}td so the horizon matches",
        },
        "universe_snapshot": universe,
        "universe_size": len(universe),
        "cohort_at_freeze": counts,
    }
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[stage A] FROZEN -> {SPEC_PATH}  (universe {len(universe)} symbols)")


# ── Amendment (additive, write-once — never edit the frozen spec) ──────────
def write_amendment(sample_start, reason):
    """Reads NO database column, so it cannot be outcome-informed."""
    if os.path.exists(AMEND_PATH):
        sys.exit(f"REFUSING: {AMEND_PATH} already exists — an amendment is written once.")
    datetime.strptime(sample_start, "%Y-%m-%d")
    payload = {"amendment": 1, "amends": os.path.basename(SPEC_PATH),
               "sample_start": sample_start, "reason": reason,
               "written_at": datetime.now(timezone.utc).isoformat(),
               "gates_unchanged": True, "primary_rule_unchanged": True}
    with open(AMEND_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


def load_amendment():
    if not os.path.exists(AMEND_PATH):
        return None
    with open(AMEND_PATH) as f:
        return json.load(f)


# ── Price cache — blind, idempotent, re-runnable ───────────────────────────
def _cache_con():
    con = sqlite3.connect(PRICE_CACHE_PATH, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS closes "
                "(symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))")
    con.commit()
    return con


def fetch_cache(symbols, start, end, chunk=40):
    """Batch-download daily closes into the local cache. Safe to re-run."""
    import yfinance as yf
    import time
    con = _cache_con()
    have = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM closes")}
    todo = sorted(set(symbols) - have)
    print(f"[cache] {len(have)} cached, {len(todo)} to fetch, {start} -> {end}")
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        try:
            df = yf.download(batch, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False, group_by="ticker")
        except Exception as e:
            print(f"[cache] batch {i // chunk} failed: {e}")
            continue
        for sym in batch:
            try:
                ser = df[sym]["Close"] if len(batch) > 1 else df["Close"]
                for ts, val in ser.dropna().items():
                    con.execute("INSERT OR REPLACE INTO closes VALUES (?,?,?)",
                                (sym, str(ts.date()), float(val)))
            except Exception:
                continue
        con.commit()
        print(f"[cache]   {min(i + chunk, len(todo))}/{len(todo)}")
        time.sleep(1.0)
    con.close()


def _close_on_or_after(con, sym, day):
    r = con.execute("SELECT close FROM closes WHERE symbol=? AND date>=? "
                    "ORDER BY date LIMIT 1", (sym, day)).fetchone()
    return r[0] if r else None


def window_return(con, sym, entry_day):
    """Fraction return over the frozen horizon, from cached closes. None if absent."""
    p0 = _close_on_or_after(con, sym, entry_day)
    p1 = _close_on_or_after(con, sym, exit_date_for(entry_day))
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return p1 / p0 - 1


# ── Stage B ────────────────────────────────────────────────────────────────
def _ci95(vals):
    vals = sorted(vals)
    if not vals:
        return None
    return [round(vals[int(0.025 * len(vals))], 3),
            round(vals[int(0.975 * len(vals))], 3)]


def _median_by_day(rows):
    """rows: (day, value). -> {day: median}"""
    by = {}
    for d, v in rows:
        by.setdefault(d, []).append(v)
    return {d: statistics.median(v) for d, v in by.items()}


def stage_b(tracker_db, setup_db):
    if not os.path.exists(SPEC_PATH):
        sys.exit("REFUSING --unblind: no r1_spec_frozen.json. Run --freeze first.")
    with open(SPEC_PATH) as f:
        spec = json.load(f)
    if not spec.get("frozen"):
        sys.exit("REFUSING --unblind: spec exists but was not written by --freeze.")

    amend = load_amendment()
    sample_start = amend.get("sample_start") if amend else None
    if sample_start:
        print(f"[stage B] amendment {amend['amendment']} in force: rows >= {sample_start}"
              f" ({amend['reason']})")

    # THE GUARD — both counts, before any outcome column is selected.
    counts = cohort_counts(tracker_db, sample_start)
    gates = spec["gates"]
    if counts["n_rows"] < gates["min_n_rows"] or counts["n_days"] < gates["min_n_days"]:
        sys.exit(
            f"NotEnoughData: rows = {counts['n_rows']} (need {gates['min_n_rows']}), "
            f"days = {counts['n_days']} (need {gates['min_n_days']}). "
            f"{counts['n_unresolved_in_tier']} signals still open in the tier. "
            "No outcome columns were read. Come back when the clock matures.")

    print(f"[stage B] rows={counts['n_rows']} days={counts['n_days']} — UNBLINDING "
          f"(one-shot, spec frozen {spec['frozen_at']})")
    print("[stage B] SANCTIONED RETRO-COMPUTATION: SPY and baseline forward returns "
          "are computed here from cached closes (outcome columns, not features). "
          "The model arm's return is read as the tracker resolved it. Nothing is written back.")

    con = sqlite3.connect(f"file:{tracker_db}?mode=ro", uri=True)
    cohort = con.execute(
        f"SELECT s.date_logged, s.sym, o.fwd_ret FROM signals s "
        f"JOIN outcomes o ON o.signal_id = s.id WHERE {eligible_where(sample_start)}"
    ).fetchall()
    con.close()

    pc = _cache_con()
    missing = []
    model_rows, spy_by_day = [], {}
    for day, sym, fwd in cohort:
        spy = window_return(pc, "SPY", day)
        if spy is None:
            missing.append((day, "SPY"))
            continue
        spy_by_day[day] = spy
        model_rows.append((day, (fwd - spy) * 100 - COMMISSION_PCT))
    if missing:
        sys.exit(f"REFUSING to report: SPY closes missing for {len(missing)} rows. "
                 "Run --fetch-cache first — a partial benchmark is not a benchmark.")

    # ── PRIMARY ──
    alphas = [v for _, v in model_rows]
    days = sorted({d for d, _ in model_rows})
    point = round(statistics.median(alphas), 3)
    rows_by_day = {d: [v for dd, v in model_rows if dd == d] for d in days}
    rng = random.Random(spec["bootstrap"]["seed"])
    reps = []
    for _ in range(spec["bootstrap"]["B"]):
        sample = []
        for d in rng.choices(days, k=len(days)):
            sample.extend(rows_by_day[d])
        reps.append(statistics.median(sample))
    ci = _ci95(reps)
    passed = ci is not None and ci[0] > 0
    print(json.dumps({"PRIMARY": {
        "median_alpha_net_pct": point, "ci95": ci, "n_rows": len(alphas),
        "n_days": len(days), "rule": spec["primary_rule"],
        "RESULT": "PASS" if passed else "FAIL"}}, indent=2))

    # ── SECONDARY — descriptive only, may not declare success ──
    sec = {"note": "SECONDARY — descriptive; may only seed a new pre-registered question"}
    raw = [(d, f * 100 - COMMISSION_PCT) for (d, _, f) in cohort]
    sec["model_median_raw_net_pct"] = round(statistics.median([v for _, v in raw]), 3)
    sec["model_precision_pct"] = round(
        100 * sum(1 for (_, _, f) in cohort if f >= HIT_THRESHOLD) / len(cohort), 1)
    sec["spy_up_rate_pct"] = round(
        100 * sum(1 for d in days if spy_by_day[d] > 0) / len(days), 1)

    # B0 — random long-only draw from the frozen universe, per-date seeded.
    universe = spec["universe_snapshot"]
    b0 = []
    for d in days:
        drng = random.Random(f"{spec['bootstrap']['seed']}|{d}")
        picks = drng.sample(universe, min(B0_DRAWS, len(universe)))
        vals = [window_return(pc, s, d) for s in picks]
        vals = [(v - spy_by_day[d]) * 100 - COMMISSION_PCT for v in vals if v is not None]
        if vals:
            b0.append((d, statistics.median(vals)))
    if b0:
        model_day_med = _median_by_day(model_rows)
        paired = [model_day_med[d] - v for d, v in b0 if d in model_day_med]
        sec["B0_random"] = {
            "n_days": len(b0),
            "median_alpha_net_pct": round(statistics.median([v for _, v in b0]), 3),
            "median_paired_edge_vs_model_pct": round(statistics.median(paired), 3)}

    # Pre-registered secondary stratum — the full BUY tier, reported not decisive.
    scon = sqlite3.connect(f"file:{tracker_db}?mode=ro", uri=True)
    strat_where = eligible_where(sample_start).replace(
        f"s.confidence >= {CONF_FLOOR}", f"s.confidence >= {SECONDARY_STRATUM_FLOOR}")
    strat = scon.execute(
        f"SELECT s.date_logged, o.fwd_ret FROM signals s "
        f"JOIN outcomes o ON o.signal_id = s.id WHERE {strat_where}").fetchall()
    scon.close()
    svals = [(d, (f - s) * 100 - COMMISSION_PCT)
             for d, f in strat
             for s in [spy_by_day.get(d) or window_return(pc, "SPY", d)] if s is not None]
    if svals:
        sec[f"stratum_conf_ge_{SECONDARY_STRATUM_FLOOR}"] = {
            "n_rows": len(svals), "n_days": len({d for d, _ in svals}),
            "median_alpha_net_pct": round(statistics.median([v for _, v in svals]), 3),
            "role": "NOT part of the pass rule"}

    # B2 — high-RVOL screener rows with no ML opinion, horizon rebuilt to match.
    if os.path.exists(setup_db):
        scon = sqlite3.connect(f"file:{setup_db}?mode=ro", uri=True)
        b2_src = scon.execute(
            "SELECT date, symbol FROM setup_log WHERE source='volume_leaders' "
            f"AND vol_ratio >= {B2_VOL_RATIO} AND (ml_signal IS NULL OR ml_signal <> 'BUY')"
        ).fetchall()
        scon.close()
        b2 = []
        for d, sym in b2_src:
            r = window_return(pc, sym, d)
            s = spy_by_day.get(d) or window_return(pc, "SPY", d)
            if r is not None and s is not None:
                b2.append((d, (r - s) * 100 - COMMISSION_PCT))
        if b2:
            sec["B2_rvol_no_ml"] = {
                "n_rows": len(b2), "n_days": len({d for d, _ in b2}),
                "median_alpha_net_pct": round(statistics.median([v for _, v in b2]), 3),
                "model_minus_B2_pct": round(point - statistics.median([v for _, v in b2]), 3)}
    pc.close()
    print(json.dumps({"SECONDARY": sec}, indent=2))
    print("[stage B] One shot spent. Promotion of anything here to a gate requires "
          "the PRIMARY to have passed AND a recorded sitting decision.")


# ── Self-test: synthetic DBs, known planted effect ─────────────────────────
def selftest():
    tmp = tempfile.mkdtemp()
    tdb = os.path.join(tmp, "tracker.db")
    con = sqlite3.connect(tdb)
    con.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, sym TEXT, date_logged TEXT,"
                " entry_price REAL, confidence REAL, resolved INTEGER, model_version TEXT)")
    con.execute("CREATE TABLE outcomes (id INTEGER PRIMARY KEY, signal_id INTEGER,"
                " fwd_ret REAL, hit INTEGER, spy_fwd_ret REAL)")
    # 30 trading days x 3 rows: model +2.0%, SPY +0.5% -> alpha 1.5% - 0.16% = 1.34%
    days = []
    d = date(2026, 3, 2)
    while len(days) < 30:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d = date.fromordinal(d.toordinal() + 1)
    sid = 0
    for day in days:
        for j in range(3):
            sid += 1
            con.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?)",
                        (sid, f"SYM{j}", day, 100.0, 0.75, 1, MODEL_VERSION))
            con.execute("INSERT INTO outcomes (signal_id, fwd_ret, hit, spy_fwd_ret)"
                        " VALUES (?,?,?,?)", (sid, 0.02, 0, None))
    # a sub-threshold row and a wrong-version row must both be excluded
    con.execute("INSERT INTO signals VALUES (999,'LOW',?,100.0,0.60,1,?)",
                (days[0], MODEL_VERSION))
    con.execute("INSERT INTO outcomes (signal_id, fwd_ret, hit, spy_fwd_ret) VALUES (999,9.9,1,NULL)")
    con.execute("INSERT INTO signals VALUES (998,'OLDV',?,100.0,0.90,1,'old_model')", (days[0],))
    con.execute("INSERT INTO outcomes (signal_id, fwd_ret, hit, spy_fwd_ret) VALUES (998,9.9,1,NULL)")
    con.commit()
    con.close()

    counts = cohort_counts(tdb)
    assert counts["n_rows"] == 90, counts
    assert counts["n_days"] == 30, counts
    print(f"[selftest] eligibility ok: {counts['n_rows']} rows / {counts['n_days']} days "
          "(sub-threshold + wrong-version rows excluded)")

    # gate must refuse below EITHER threshold, without touching outcomes
    thin = os.path.join(tmp, "thin.db")
    con = sqlite3.connect(thin)
    con.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, sym TEXT, date_logged TEXT,"
                " entry_price REAL, confidence REAL, resolved INTEGER, model_version TEXT)")
    con.execute("CREATE TABLE outcomes (id INTEGER PRIMARY KEY, signal_id INTEGER,"
                " fwd_ret REAL, hit INTEGER, spy_fwd_ret REAL)")
    for i in range(MIN_N_ROWS + 10):          # rows PASS, days FAIL
        con.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?)",
                    (i + 1, "X", days[i % 5], 100.0, 0.80, 1, MODEL_VERSION))
        con.execute("INSERT INTO outcomes (signal_id, fwd_ret, hit, spy_fwd_ret)"
                    " VALUES (?,?,?,?)", (i + 1, 0.05, 1, None))
    con.commit()
    con.close()
    tc = cohort_counts(thin)
    assert tc["n_rows"] >= MIN_N_ROWS and tc["n_days"] < MIN_N_DAYS, tc
    print(f"[selftest] day-gate ok: {tc['n_rows']} rows across only {tc['n_days']} days "
          "would REFUSE — a 5-cluster sample cannot pass on row count alone")

    # bootstrap recovers a planted alpha carried by NOISY rows — identical values
    # would make any CI degenerate and prove nothing about the resampling.
    nrng = random.Random(11)
    rows = [(day, (0.02 - 0.005) * 100 - COMMISSION_PCT + nrng.uniform(-1.5, 1.5))
            for day in days for _ in range(3)]
    med = statistics.median([v for _, v in rows])
    assert 0.8 < med < 1.9, med
    rng = random.Random(BOOT_SEED)
    by_day = {d: [v for dd, v in rows if dd == d] for d in days}
    reps = []
    for _ in range(2000):
        s = []
        for d in rng.choices(days, k=len(days)):
            s.extend(by_day[d])
        reps.append(statistics.median(s))
    ci = _ci95(reps)
    assert ci[0] < med < ci[1], (ci, med)          # a real interval, not a point
    assert ci[0] > 0, ci                            # and it clears zero -> PASS
    print(f"[selftest] planted alpha recovered: median={med:.2f}% ci95={ci} -> PASS")

    # a null effect must NOT pass — the rule has to be able to fail
    null_rows = [(day, nrng.uniform(-1.5, 1.5)) for day in days for _ in range(3)]
    nby = {d: [v for dd, v in null_rows if dd == d] for d in days}
    nrep = []
    for _ in range(2000):
        s = []
        for d in rng.choices(days, k=len(days)):
            s.extend(nby[d])
        nrep.append(statistics.median(s))
    nci = _ci95(nrep)
    assert nci[0] <= 0 <= nci[1], nci
    print(f"[selftest] null effect correctly FAILS: ci95={nci} straddles zero")
    print("[selftest] all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker-db", default=TRACKER_DB_DEFAULT)
    ap.add_argument("--setup-db", default=SETUP_DB_DEFAULT)
    ap.add_argument("--freeze", action="store_true", help="stage A: write r1_spec_frozen.json")
    ap.add_argument("--unblind", action="store_true", help="stage B: guarded confirmatory run")
    ap.add_argument("--fetch-cache", action="store_true", help="pre-download closes (blind)")
    ap.add_argument("--selftest", action="store_true", help="run pipeline on synthetic data")
    ap.add_argument("--amend-sample-start", metavar="YYYY-MM-DD")
    ap.add_argument("--amend-reason", default="")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.amend_sample_start:
        if not a.amend_reason:
            sys.exit("--amend-reason is required: an unexplained amendment is not an amendment.")
        return write_amendment(a.amend_sample_start, a.amend_reason)
    if a.fetch_cache:
        if not os.path.exists(SPEC_PATH):
            sys.exit("Freeze the spec first — the cache follows the frozen universe.")
        with open(SPEC_PATH) as f:
            spec = json.load(f)
        con = sqlite3.connect(f"file:{a.tracker_db}?mode=ro", uri=True)
        first = con.execute("SELECT MIN(date_logged) FROM signals").fetchone()[0]
        con.close()
        syms = set(spec["universe_snapshot"]) | {"SPY"}
        if os.path.exists(a.setup_db):
            scon = sqlite3.connect(f"file:{a.setup_db}?mode=ro", uri=True)
            syms |= {r[0] for r in scon.execute(
                "SELECT DISTINCT symbol FROM setup_log WHERE source='volume_leaders' "
                f"AND vol_ratio >= {B2_VOL_RATIO}")}
            scon.close()
        return fetch_cache(sorted(syms), first, str(date.today()))
    if a.unblind:
        return stage_b(a.tracker_db, a.setup_db)
    return stage_a(a.tracker_db, a.freeze)


if __name__ == "__main__":
    main()
