#!/usr/bin/env python3
"""lev_sitting.py — the pre-registered lev_sent bucket pipeline, blinded by design.

Two clocks, two stages (spec locked Jul 14 2026, THE SITTING is Jul 24+):

  STAGE A (blind — covariates only, safe to run any time):
      python3 lev_sitting.py                # preview: prints the would-be boundary
      python3 lev_sitting.py --freeze       # THE SITTING: writes lev_spec_frozen.json
    Reads ONLY (date, lev_sent_semis, lev_sent_qqq) — never any outcome column.
    Per-day median -> median of day-medians -> the frozen bucket boundary.

  STAGE B (unblind — runs the confirmatory query, guarded):
      python3 lev_sitting.py --unblind
    Refuses unless (1) lev_spec_frozen.json exists AND was written by --freeze,
    (2) eligible-row COUNT >= 50. The count runs BEFORE any ret_* column is
    selected; below threshold the script exits without ever building the
    outcome query. At N>=50: dual-basis medians net of 0.16% commission,
    hard binary split (<= boundary -> LOW), day-cluster bootstrap B=10k seed=42,
    95% percentile CI on the median difference. Interpretive only — promotion
    to any gate additionally requires the pre-registered test to pass.

  SELF-TEST (synthetic DB, verifies the pipeline math end-to-end):
      python3 lev_sitting.py --selftest

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
from datetime import datetime, timezone

# ── Pre-registered constants (locked Jul 14 2026) ──────────────────────────
DB_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_log.db")
SPEC_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lev_spec_frozen.json")
MIN_N           = 50          # resolved BREAKOUT CONFIRMED rows with lev + outcomes
COMMISSION_PCT  = 0.16        # round trip, 0.08% per side
BOOT_B          = 10_000
BOOT_SEED       = 42
PRIMARY         = "semis"     # qqq is secondary (retail-bot confounder on the pair)
TIE_RULE        = "value <= boundary -> LOW bucket"


# ── Stage A: covariate-only boundary ────────────────────────────────────────
def compute_boundary(con):
    """Day-median of lev_sent values -> median of day-medians. Covariates ONLY."""
    rows = con.execute(
        "SELECT date, lev_sent_semis, lev_sent_qqq FROM setup_log "
        "WHERE lev_sent_semis IS NOT NULL"
    ).fetchall()
    by_day = {}
    for d, semis, qqq in rows:
        by_day.setdefault(d, {"semis": [], "qqq": []})
        by_day[d]["semis"].append(semis)
        if qqq is not None:
            by_day[d]["qqq"].append(qqq)
    if not by_day:
        sys.exit("No lev_sent rows found — nothing to compute a boundary from.")
    day_medians = {
        "semis": {d: round(statistics.median(v["semis"]), 4) for d, v in sorted(by_day.items())},
        "qqq":   {d: round(statistics.median(v["qqq"]), 4)
                  for d, v in sorted(by_day.items()) if v["qqq"]},
    }
    boundary = {k: round(statistics.median(list(m.values())), 4)
                for k, m in day_medians.items() if m}
    return {
        "n_days": len(by_day),
        "n_rows": len(rows),
        "day_medians": day_medians,
        "boundary": boundary,
        "primary": PRIMARY,
        "tie_rule": TIE_RULE,
    }


def stage_a(db_path, freeze):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    spec = compute_boundary(con)
    con.close()
    print(f"[stage A] days={spec['n_days']} rows={spec['n_rows']}")
    for k in ("semis", "qqq"):
        if k in spec["boundary"]:
            tag = " (PRIMARY)" if k == PRIMARY else ""
            print(f"[stage A] boundary {k}{tag}: {spec['boundary'][k]}")
    if not freeze:
        print("[stage A] PREVIEW ONLY — nothing written. Use --freeze at the sitting.")
        return
    if os.path.exists(SPEC_PATH):
        sys.exit(f"REFUSING: {SPEC_PATH} already exists — the boundary is frozen once. "
                 "Delete it manually only if the sitting itself is being redone.")
    spec.update({
        "frozen": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "min_n": MIN_N,
        "commission_pct": COMMISSION_PCT,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "resample": "trading_days"},
    })
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[stage A] ❄ FROZEN -> {SPEC_PATH}")


# ── Stage B: guarded confirmatory query ─────────────────────────────────────
ELIGIBLE_WHERE = (
    "source='gainers' AND verdict='BREAKOUT CONFIRMED' AND resolved=1 "
    "AND lev_sent_semis IS NOT NULL AND ret_5d IS NOT NULL "
    "AND close_5d IS NOT NULL AND price > 0"
)


def median_diff(rows, boundary):
    """rows: (day, lev, ret_close, ret_signal). Returns per-basis {low,high,diff} medians
    net of commission, or None if either bucket is empty."""
    out = {}
    for basis, idx in (("close_basis", 2), ("signal_basis", 3)):
        low  = [r[idx] - COMMISSION_PCT for r in rows if r[1] <= boundary]
        high = [r[idx] - COMMISSION_PCT for r in rows if r[1] >  boundary]
        if not low or not high:
            return None
        out[basis] = {
            "n_low": len(low), "n_high": len(high),
            "median_low": round(statistics.median(low), 3),
            "median_high": round(statistics.median(high), 3),
            "diff_high_minus_low": round(statistics.median(high) - statistics.median(low), 3),
        }
    return out


def stage_b(db_path):
    if not os.path.exists(SPEC_PATH):
        sys.exit("REFUSING --unblind: no lev_spec_frozen.json. Run --freeze at the sitting first.")
    with open(SPEC_PATH) as f:
        spec = json.load(f)
    if not spec.get("frozen"):
        sys.exit("REFUSING --unblind: spec file exists but was not written by --freeze.")
    boundary = spec["boundary"][PRIMARY]

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # The guard: COUNT first. No outcome column is selected before this passes.
    n = con.execute(f"SELECT COUNT(*) FROM setup_log WHERE {ELIGIBLE_WHERE}").fetchone()[0]
    if n < spec.get("min_n", MIN_N):
        con.close()
        sys.exit(f"NotEnoughData: N = {n}, requires minimum {spec.get('min_n', MIN_N)}. "
                 "No outcome columns were read. Come back when the clock matures.")

    print(f"[stage B] N = {n} >= {spec.get('min_n', MIN_N)} — UNBLINDING (one-shot, "
          f"boundary {PRIMARY}={boundary} from {spec['frozen_at']})")
    rows = [
        (d, lev, ret5, round((c5 / price - 1) * 100, 2))
        for d, lev, ret5, c5, price in con.execute(
            f"SELECT date, lev_sent_semis, ret_5d, close_5d, price "
            f"FROM setup_log WHERE {ELIGIBLE_WHERE}"
        )
    ]
    con.close()

    point = median_diff(rows, boundary)
    if point is None:
        sys.exit("Degenerate split: one bucket is empty at the frozen boundary. "
                 "Record this outcome — it is itself a result.")
    print(json.dumps({"point_estimate": point}, indent=2))

    # Day-cluster bootstrap on the median difference.
    days = sorted({r[0] for r in rows})
    rows_by_day = {d: [r for r in rows if r[0] == d] for d in days}
    rng = random.Random(spec["bootstrap"]["seed"])
    B = spec["bootstrap"]["B"]
    diffs = {"close_basis": [], "signal_basis": []}
    degenerate = 0
    for _ in range(B):
        sample = []
        for d in rng.choices(days, k=len(days)):
            sample.extend(rows_by_day[d])
        md = median_diff(sample, boundary)
        if md is None:
            degenerate += 1
            continue
        for basis in diffs:
            diffs[basis].append(md[basis]["diff_high_minus_low"])
    ci = {}
    for basis, vals in diffs.items():
        vals.sort()
        ci[basis] = {
            "ci95": [round(vals[int(0.025 * len(vals))], 3),
                     round(vals[int(0.975 * len(vals))], 3)],
            "replicates_used": len(vals),
        }
    print(json.dumps({"bootstrap": ci, "degenerate_replicates": degenerate,
                      "n_days": len(days)}, indent=2))
    print("[stage B] Interpretive only — gate promotion requires the pre-registered "
          "test to pass AND a recorded sitting decision.")


# ── Self-test: synthetic DB, known planted effect ───────────────────────────
def selftest():
    tmp = os.path.join(tempfile.mkdtemp(), "synthetic_setup_log.db")
    con = sqlite3.connect(tmp)
    con.execute("""CREATE TABLE setup_log (
        id INTEGER PRIMARY KEY, source TEXT, verdict TEXT, date TEXT, price REAL,
        lev_sent_semis REAL, lev_sent_qqq REAL, resolved INTEGER,
        close_5d REAL, ret_5d REAL)""")
    rng = random.Random(7)
    # 30 synthetic days: lev ramps 0.20 -> 0.49; true day-median boundary = 0.345.
    # Planted effect: high-lev days return ~+2%, low-lev days ~-1%.
    for i in range(30):
        day = f"2026-06-{i + 1:02d}"
        lev = round(0.20 + i * 0.01, 2)
        for j in range(3):  # 3 rows/day -> 90 rows, 60 eligible breakout rows below
            ret = (2.0 if lev > 0.345 else -1.0) + rng.uniform(-0.3, 0.3)
            price = 100.0
            con.execute(
                "INSERT INTO setup_log (source, verdict, date, price, lev_sent_semis,"
                " lev_sent_qqq, resolved, close_5d, ret_5d) VALUES (?,?,?,?,?,?,?,?,?)",
                ("gainers", "BREAKOUT CONFIRMED" if j < 2 else "DEVELOPING", day, price,
                 lev, lev + 0.1, 1, price * (1 + ret / 100), round(ret, 2)))
    con.commit()

    spec = compute_boundary(con)
    assert spec["n_days"] == 30 and spec["n_rows"] == 90, spec
    assert abs(spec["boundary"]["semis"] - 0.345) < 1e-9, spec["boundary"]
    print(f"[selftest] stage A ok: boundary={spec['boundary']['semis']} (expected 0.345)")

    n = con.execute(f"SELECT COUNT(*) FROM setup_log WHERE {ELIGIBLE_WHERE}").fetchone()[0]
    assert n == 60, n
    rows = [(d, lev, ret5, round((c5 / p - 1) * 100, 2)) for d, lev, ret5, c5, p in
            con.execute(f"SELECT date, lev_sent_semis, ret_5d, close_5d, price "
                        f"FROM setup_log WHERE {ELIGIBLE_WHERE}")]
    point = median_diff(rows, spec["boundary"]["semis"])
    diff = point["close_basis"]["diff_high_minus_low"]
    assert 2.4 < diff < 3.6, point   # planted ~3.0
    assert point["close_basis"]["n_low"] == 30 and point["close_basis"]["n_high"] == 30
    print(f"[selftest] stage B ok: N={n}, planted diff recovered = {diff} (expected ~3.0)")

    # Guard check: with only 20 eligible rows the count gate must refuse.
    con.execute("DELETE FROM setup_log WHERE id NOT IN "
                "(SELECT id FROM setup_log WHERE verdict='BREAKOUT CONFIRMED' LIMIT 20)")
    con.commit()
    n2 = con.execute(f"SELECT COUNT(*) FROM setup_log WHERE {ELIGIBLE_WHERE}").fetchone()[0]
    assert n2 == 20 and n2 < MIN_N
    print(f"[selftest] N-guard ok: N={n2} < {MIN_N} would refuse to unblind")
    con.close()
    print("[selftest] ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="path to setup_log.db")
    ap.add_argument("--freeze", action="store_true", help="stage A: write lev_spec_frozen.json (THE SITTING)")
    ap.add_argument("--unblind", action="store_true", help="stage B: guarded confirmatory run")
    ap.add_argument("--selftest", action="store_true", help="run pipeline on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.unblind:
        stage_b(args.db)
    else:
        stage_a(args.db, freeze=args.freeze)
