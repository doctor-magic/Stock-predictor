"""
Unit tests for the pure decision functions in scanners.py (Group A).

Philosophy
----------
These tests cover the THREE pure, isolated verdict functions that carry the
system's decision logic — the cheap, fast, dependency-free safety net for a
solo, soon-to-be-RAM-constrained project:

    - compute_verdict   (Volume Leaders 10-day ML outlook + momentum overlays)
    - compute_momentum  (Volume Leaders momentum label)
    - gainers_verdict   (Gainers tab verdict)

Each function is a *cascading precedence ladder*: the first matching branch wins.
The highest-value tests here are the "precedence traps" — cases where an early
rung of the ladder overrides a strong signal on a later rung. Locking these
guards the structural order: if a condition line is ever moved up/down by
accident, these tests fire immediately.

Scenarios are weighted by REAL production frequency, pulled from setup_log.db
(Jun 8-12 2026, 53 events): OVERHEAD WALL x19, OVEREXTENDED x18, FADE RISK x11,
DEVELOPING x2, VOL BREAKOUT x2, HIGH-BETA x1.

NOT covered here (Group B — deliberately deferred):
    The HOD gate, Beta gate and RVOL-slope gate are still inline in api.py's
    get_volume_leaders(), not yet extracted into pure functions. They require a
    small, separate extraction before they can be unit-tested. See the Group A/B
    split discussed in the handoff.

Stdlib `unittest` only — zero new dependencies (intentional; we are cutting RAM,
not adding pytest).

Run:
    cd ~/Desktop/Stock-predictor && python3 -m unittest test_scanners -v
"""

import unittest

import scanners


class TestComputeVerdict(unittest.TestCase):
    """compute_verdict precedence ladder:
        1. OVEREXTENDED  (current vs open >= +3%)
        2. VOL BREAKOUT  (vol_ratio>=3.0 AND rsi is not None AND rsi<65)
        3. BUY           (ml=BUY AND conf>=threshold AND vol_ratio>=1.0 AND above_sma50)
                         threshold = 0.65 if vol_ratio>=2.0 else 0.70
        4. HOLD          (ml=SELL  -> Long-only Expert suppression)
        5. N/A           (ml in {None, "N/A"})
        6. HOLD          (default)
    open_price is left None unless a test specifically exercises OVEREXTENDED,
    so step 1 never preempts the branch under test.
    """

    # --- happy path + thresholds ------------------------------------------
    def test_buy_full_valid(self):
        self.assertEqual(
            scanners.compute_verdict("BUY", 0.72, vol_ratio=1.5, above_sma50=True),
            "BUY",
        )

    def test_buy_threshold_070_inclusive(self):
        # vol_ratio<2.0 -> threshold 0.70; 0.70 >= 0.70 is inclusive
        self.assertEqual(scanners.compute_verdict("BUY", 0.70, vol_ratio=1.0), "BUY")

    def test_below_normal_threshold_holds(self):
        self.assertEqual(scanners.compute_verdict("BUY", 0.69, vol_ratio=1.0), "HOLD")

    def test_highvol_threshold_065(self):
        # vol_ratio>=2.0 lowers threshold to 0.65
        self.assertEqual(scanners.compute_verdict("BUY", 0.66, vol_ratio=2.0), "BUY")

    def test_highvol_below_065_holds(self):
        self.assertEqual(scanners.compute_verdict("BUY", 0.64, vol_ratio=2.0), "HOLD")

    # --- PLTR case studies (the documented bad-signal guards) -------------
    def test_pltr_below_sma50_holds(self):
        # Strong BUY confidence but price below SMA50 -> must NOT be BUY
        self.assertEqual(
            scanners.compute_verdict("BUY", 0.75, vol_ratio=2.0, above_sma50=False),
            "HOLD",
        )

    def test_pltr_low_volume_holds(self):
        # 0.7x volume = no market confirmation -> must NOT be BUY
        self.assertEqual(scanners.compute_verdict("BUY", 0.72, vol_ratio=0.7), "HOLD")

    # --- SELL suppression + N/A -------------------------------------------
    def test_sell_suppressed_to_hold(self):
        self.assertEqual(scanners.compute_verdict("SELL", 0.80, vol_ratio=1.2), "HOLD")

    def test_none_signal_is_na(self):
        self.assertEqual(scanners.compute_verdict(None, 0.50, vol_ratio=1.0), "N/A")

    def test_na_signal_is_na(self):
        self.assertEqual(scanners.compute_verdict("N/A", 0.50, vol_ratio=1.0), "N/A")

    # --- OVEREXTENDED + the +3% epsilon boundary (18/21 of real output) ---
    def test_overextended_basic(self):
        self.assertEqual(
            scanners.compute_verdict("BUY", 0.80, open_price=100, current_price=104),
            "OVEREXTENDED",
        )

    def test_overextended_boundary_exactly_3pct(self):
        # +3.00% is inclusive (>= 0.03)
        self.assertEqual(
            scanners.compute_verdict("BUY", 0.80, open_price=100, current_price=103),
            "OVEREXTENDED",
        )

    def test_just_below_3pct_falls_through_to_buy(self):
        # +2.99% must NOT be OVEREXTENDED -> evaluates as a normal BUY
        self.assertEqual(
            scanners.compute_verdict(
                "BUY", 0.80, vol_ratio=1.5, open_price=100, current_price=102.99
            ),
            "BUY",
        )

    # --- VOL BREAKOUT strict boundaries -----------------------------------
    def test_vol_breakout_boundary_vol3_rsi64(self):
        self.assertEqual(
            scanners.compute_verdict("HOLD", 0.0, vol_ratio=3.0, rsi=64),
            "VOL BREAKOUT",
        )

    def test_rsi65_exact_not_vol_breakout(self):
        # rsi<65 is strict; 65 does not qualify
        self.assertEqual(
            scanners.compute_verdict("HOLD", 0.0, vol_ratio=3.0, rsi=65), "HOLD"
        )

    # --- defensive nulls / corrupt feed (the nan/_clean failure class) ----
    def test_none_confidence_buy_holds(self):
        # (ml_confidence or 0) guard -> never crashes, never false BUY
        self.assertEqual(
            scanners.compute_verdict("BUY", None, vol_ratio=1.5), "HOLD"
        )

    def test_zero_open_price_no_zerodivision(self):
        # halted/new stock: open_price=0 must not raise; OVEREXTENDED skipped
        self.assertEqual(
            scanners.compute_verdict(
                "BUY", 0.80, vol_ratio=1.5, open_price=0, current_price=104
            ),
            "BUY",
        )

    # --- precedence TRAPS (early rung overrides a stronger later signal) --
    def test_trap_overextended_preempts_buy(self):
        # Perfect BUY (0.90, high vol) but +4% from open -> OVEREXTENDED wins
        self.assertEqual(
            scanners.compute_verdict(
                "BUY", 0.90, vol_ratio=2.0, open_price=100, current_price=104
            ),
            "OVEREXTENDED",
        )

    def test_trap_vol_breakout_preempts_sell(self):
        # SELL would suppress to HOLD, but VOL BREAKOUT (step 2) fires first
        self.assertEqual(
            scanners.compute_verdict("SELL", 0.50, vol_ratio=3.5, rsi=50),
            "VOL BREAKOUT",
        )

    def test_trap_rsi_none_disables_vol_breakout(self):
        # vol_ratio 3.5 but rsi=None -> step 2 skipped -> evaluates as BUY
        self.assertEqual(
            scanners.compute_verdict("BUY", 0.70, vol_ratio=3.5, rsi=None),
            "BUY",
        )


class TestComputeMomentum(unittest.TestCase):
    """compute_momentum precedence ladder:
        1. OVEREXTENDED  (rsi>75)
        2. WATCH         (rsi<35 AND vol_ratio>=2.0)
        3. SURGING       (vol_ratio>=3.0 AND ret_5d>1.0)
        4. SELLING OFF   (ret_5d<-4.0 AND vol_ratio>=2.0)
        5. NEUTRAL       (default)
    NOTE: OVEREXTENDED here is RSI-based (>75) — a DIFFERENT trigger from the
    price-vs-open OVEREXTENDED in compute_verdict. Do not conflate them.
    """

    def test_overextended_rsi(self):
        self.assertEqual(scanners.compute_momentum(80, 1.0, 0), "OVEREXTENDED")

    def test_trap_overextended_preempts_surging(self):
        # rsi>75 wins even though vol+ret would otherwise be SURGING
        self.assertEqual(scanners.compute_momentum(80, 3.5, 5.0), "OVEREXTENDED")

    def test_watch(self):
        self.assertEqual(scanners.compute_momentum(30, 2.5, 0), "WATCH")

    def test_watch_boundary_vol2_inclusive(self):
        self.assertEqual(scanners.compute_momentum(30, 2.0, 0), "WATCH")

    def test_low_rsi_weak_volume_is_neutral(self):
        # rsi<35 but vol_ratio<2.0 -> not WATCH -> NEUTRAL
        self.assertEqual(scanners.compute_momentum(30, 1.5, 0), "NEUTRAL")

    def test_surging(self):
        self.assertEqual(scanners.compute_momentum(50, 3.5, 2.0), "SURGING")

    def test_surging_boundary_vol3_ret_just_above_1(self):
        self.assertEqual(scanners.compute_momentum(50, 3.0, 1.01), "SURGING")

    def test_ret_exactly_1_not_surging(self):
        # ret_5d>1.0 is strict; 1.0 does not surge
        self.assertEqual(scanners.compute_momentum(50, 3.5, 1.0), "NEUTRAL")

    def test_selling_off(self):
        self.assertEqual(scanners.compute_momentum(50, 2.5, -5.0), "SELLING OFF")

    def test_rsi_boundary_75_not_overextended(self):
        self.assertEqual(scanners.compute_momentum(75, 1.0, 0), "NEUTRAL")

    def test_rsi_none_neutral(self):
        self.assertEqual(scanners.compute_momentum(None, 1.0, 0), "NEUTRAL")

    def test_ret_none_neutral(self):
        # (ret_5d or 0) guard -> 0 -> neither SURGING nor SELLING OFF
        self.assertEqual(scanners.compute_momentum(50, 3.5, None), "NEUTRAL")


class TestGainersVerdict(unittest.TestCase):
    """gainers_verdict precedence ladder:
        0. above_vwap = (vwap_gap_pct is not None) AND (vwap_gap_pct > 0)
        1. OVERHEAD WALL        (overhead_blocked)
        2. FADE RISK            (v_accel<1.0)
        3. BREAKOUT CONFIRMED   (v_accel>=1.5 AND above_vwap)
        4. DEVELOPING           (v_accel>=1.0)
        5. WATCH                (default)
    Two v_accel boundaries (1.0 and 1.5); above_vwap modulates the 1.5 one.
    """

    # OVERHEAD WALL — the #1 production verdict (x19)
    def test_overhead_wall_basic(self):
        self.assertEqual(scanners.gainers_verdict(2.0, True, 1.0), "OVERHEAD WALL")

    def test_trap_overhead_wall_preempts_breakout(self):
        # Perfect breakout inputs, but overhead_blocked (step 1) wins
        self.assertEqual(scanners.gainers_verdict(2.0, True, 5.0), "OVERHEAD WALL")

    def test_overhead_wall_with_none_vaccel(self):
        # step 1 ignores v_accel entirely
        self.assertEqual(scanners.gainers_verdict(None, True, 2.0), "OVERHEAD WALL")

    # FADE RISK — #3 production verdict (x11)
    def test_fade_risk_basic(self):
        self.assertEqual(scanners.gainers_verdict(0.8, False, 1.0), "FADE RISK")

    def test_fade_risk_boundary_099(self):
        self.assertEqual(scanners.gainers_verdict(0.99, False, 1.0), "FADE RISK")

    def test_vaccel_1_flips_to_developing(self):
        # v_accel<1.0 is strict; exactly 1.0 -> DEVELOPING, not FADE RISK
        self.assertEqual(scanners.gainers_verdict(1.0, False, 1.0), "DEVELOPING")

    # BREAKOUT CONFIRMED — never fired in production (gates suppressed it)
    def test_breakout_confirmed(self):
        self.assertEqual(scanners.gainers_verdict(1.6, False, 2.0), "BREAKOUT CONFIRMED")

    def test_breakout_boundary_15_inclusive(self):
        self.assertEqual(scanners.gainers_verdict(1.5, False, 0.5), "BREAKOUT CONFIRMED")

    # above_vwap traps — same v_accel, different verdict
    def test_trap_15_at_vwap_is_developing(self):
        # vwap_gap_pct == 0.0 -> above_vwap False -> DEVELOPING, not BREAKOUT
        self.assertEqual(scanners.gainers_verdict(1.5, False, 0.0), "DEVELOPING")

    def test_trap_16_below_vwap_is_developing(self):
        self.assertEqual(scanners.gainers_verdict(1.6, False, -1.0), "DEVELOPING")

    def test_trap_149_above_vwap_is_developing(self):
        # 1.49 just under the 1.5 breakout cut -> DEVELOPING despite above_vwap
        self.assertEqual(scanners.gainers_verdict(1.49, False, 2.0), "DEVELOPING")

    def test_vwap_none_not_breakout(self):
        # vwap_gap_pct None -> above_vwap False -> DEVELOPING
        self.assertEqual(scanners.gainers_verdict(2.0, False, None), "DEVELOPING")

    # WATCH default
    def test_none_vaccel_is_watch(self):
        self.assertEqual(scanners.gainers_verdict(None, False, 2.0), "WATCH")

    def test_all_none_default_watch(self):
        self.assertEqual(scanners.gainers_verdict(None, False, None), "WATCH")


class TestLevSentiment(unittest.TestCase):
    """Leveraged-ETF sentiment ratios (added Jul 7 2026, observational only).
    Ratio = short/long DOLLAR volume; None on any missing/zero leg — a sentiment
    failure must never break a scan."""

    def test_ratio_math_dollar_volume(self):
        # SOXS $10 x 60M = $600M vs SOXL $20 x 15M = $300M -> 2.0 (share ratio would be 4.0)
        out = scanners._compute_lev_ratios({
            "SOXS": 10.0 * 60e6, "SOXL": 20.0 * 15e6,
            "SQQQ": 30.0 * 10e6, "TQQQ": 75.0 * 8e6,
        })
        self.assertEqual(out["semis"], 2.0)
        self.assertEqual(out["qqq"], 0.5)

    def test_zero_long_leg_gives_none(self):
        out = scanners._compute_lev_ratios({"SOXS": 1e6, "SOXL": 0, "SQQQ": 1e6, "TQQQ": 1e6})
        self.assertIsNone(out["semis"])
        self.assertEqual(out["qqq"], 1.0)

    def test_missing_leg_gives_none_per_pair_only(self):
        out = scanners._compute_lev_ratios({"SOXS": None, "SOXL": 5e6, "SQQQ": 2e6, "TQQQ": 4e6})
        self.assertIsNone(out["semis"])
        self.assertEqual(out["qqq"], 0.5)

    def test_fetch_failure_returns_stale_or_none(self):
        # yfinance blowing up must fall back to the cached value (None when cold)
        saved = dict(scanners._lev_sentiment_cache)
        try:
            scanners._lev_sentiment_cache["ts"] = 0
            scanners._lev_sentiment_cache["data"] = None
            import builtins
            real_import = builtins.__import__
            def _boom(name, *a, **k):
                if name == "yfinance":
                    raise RuntimeError("network down")
                return real_import(name, *a, **k)
            builtins.__import__ = _boom
            try:
                self.assertIsNone(scanners.get_lev_sentiment())
            finally:
                builtins.__import__ = real_import
        finally:
            scanners._lev_sentiment_cache.update(saved)


class TestMarketCalendar(unittest.TestCase):
    """market_calendar: holiday table + half-day close hours (added Jul 5 2026)."""

    def test_full_day_close_hour(self):
        from datetime import date
        from market_calendar import session_close_hour
        self.assertEqual(session_close_hour(date(2026, 7, 6)), 16)   # regular Monday

    def test_half_day_close_hours(self):
        from datetime import date
        from market_calendar import session_close_hour
        self.assertEqual(session_close_hour(date(2026, 11, 27)), 13)  # post-Thanksgiving
        self.assertEqual(session_close_hour(date(2026, 12, 24)), 13)  # Christmas Eve

    def test_half_days_are_trading_days(self):
        # Half-days are OPEN sessions — the guard must NOT block them
        from datetime import date
        from market_calendar import is_us_market_session
        self.assertTrue(is_us_market_session(date(2026, 11, 27)))
        self.assertTrue(is_us_market_session(date(2026, 12, 24)))

    def test_holiday_and_weekend_closed(self):
        from datetime import date
        from market_calendar import is_us_market_session
        self.assertFalse(is_us_market_session(date(2026, 7, 3)))   # Independence Day (obs.)
        self.assertFalse(is_us_market_session(date(2026, 7, 5)))   # Sunday

    def test_pre_open_logging_guard(self):
        # Jul-7-2026 QNT incident: pre-open (00:22 ET) must NOT be loggable —
        # screeners still serve yesterday's quotes under today's ET date.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from market_calendar import has_session_opened
        ET = ZoneInfo("America/New_York")
        self.assertFalse(has_session_opened(datetime(2026, 7, 7, 0, 22, tzinfo=ET)))   # pre-open
        self.assertFalse(has_session_opened(datetime(2026, 7, 7, 9, 29, tzinfo=ET)))   # 1 min early
        self.assertTrue(has_session_opened(datetime(2026, 7, 7, 9, 30, tzinfo=ET)))    # open
        self.assertTrue(has_session_opened(datetime(2026, 7, 7, 15, 59, tzinfo=ET)))   # live session
        self.assertTrue(has_session_opened(datetime(2026, 7, 7, 21, 0, tzinfo=ET)))    # post-close, same ET day — quotes ARE today's
        self.assertFalse(has_session_opened(datetime(2026, 7, 5, 12, 0, tzinfo=ET)))   # Sunday noon


def _has_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def _synthetic_wedge(compression_target: float):
    """30-bar falling wedge with an exact target compression.

    Upper line: 100 - 0.5*i (peaks touch it every 5 bars).
    Lower line starts at (100 - width_start) and converges so that
    width_end = width_start * (1 - compression_target).
    Closes sit mid-channel (no breakout); volume declines.
    """
    n = 30
    width_start = 12.0
    width_end = width_start * (1.0 - compression_target)
    upper = [100.0 - 0.5 * i for i in range(n)]
    lower_start = upper[0] - width_start
    lower_end = upper[-1] - width_end
    lower_slope = (lower_end - lower_start) / (n - 1)
    lower = [lower_start + lower_slope * i for i in range(n)]
    highs = [upper[i] if i % 5 == 0 else upper[i] - 2.0 for i in range(n)]
    lows = [lower[i] if i % 5 == 2 else lower[i] + 2.0 for i in range(n)]
    closes = [(upper[i] + lower[i]) / 2 for i in range(n)]
    volumes = [1000.0 - 20.0 * i for i in range(n)]
    return highs, lows, closes, volumes


@unittest.skipUnless(_has_scipy(), "scipy not installed")
class TestWedgeCompressionThresholds(unittest.TestCase):
    """Regression tripwire for the Jun 7 2026 refactor bug: thresholds silently
    reverted from 0.40/0.45/0.50 to the pre-May-15 0.15/0.20/0.25. A ~0.30
    compression wedge must NOT be detected; a ~0.65 one must be."""

    def test_weak_compression_rejected(self):
        h, l, c, v = _synthetic_wedge(0.30)
        self.assertEqual(scanners.detect_falling_wedge(h, l, c, v), {},
                         "0.30 compression detected — thresholds regressed below 0.40?")

    def test_strong_compression_detected(self):
        h, l, c, v = _synthetic_wedge(0.65)
        result = scanners.detect_falling_wedge(h, l, c, v)
        self.assertTrue(result.get("detected"),
                        "0.65 compression NOT detected — detector broken or thresholds too high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
