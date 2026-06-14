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


if __name__ == "__main__":
    unittest.main(verbosity=2)
