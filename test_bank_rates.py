"""
Unit tests for the pure functions in bank_rates.py.

Philosophy
----------
Same spirit as test_scanners.py: cover the pure, dependency-free maths that
carries the meaning, and lock the properties that would silently invert the
feature's conclusion if someone "tidied" the code later.

The three that matter most, in order:

1. THE SIGN ON IMPACT. The paper's whole result is that a rate rise compresses
   the margin first and lifts it later. That depends on an easy-to-miss timing
   convention (the shock is unanticipated, so it lands in the lag-1 difference
   slot). Standard regression timing flips the impact quarter positive and the
   feature would then say the opposite of the research it cites. Locked here
   against the paper's own Chart 5 value.

2. YTD DE-CUMULATION. FDIC reports year-to-date annualised ratios. Reading them
   as standalone quarters understates every Q2-Q4 change and quietly biases the
   difference terms toward zero — the exact terms the paper cares about.

3. UNIT TRANSFER. The paper's printed coefficients are ambiguous (Table A says
   per cent, the text's magnitudes imply basis points). The reconciliation is
   pinned to the four economic magnitudes the paper states in prose, so a future
   edit to the scaling constant fails loudly.

No network. Every test runs on synthetic or literal data.

Run:
    cd ~/Desktop/Stock-predictor && python3 -m unittest test_bank_rates -v
"""

import unittest

import bank_rates as br


class TestDecumulateYTD(unittest.TestCase):
    """FDIC year-to-date annualised -> standalone-quarter annualised."""

    def test_q1_passes_through(self):
        # Q1 YTD is already a single quarter; nothing to de-cumulate.
        out = br._decumulate_ytd({"20260331": 2.90})
        self.assertAlmostEqual(out["2026-01"], 2.90, places=6)

    def test_quarter_end_month_maps_to_fred_quarter_start(self):
        # FDIC stamps the quarter END (Mar/Jun/Sep/Dec); FRED stamps the quarter
        # START (Jan/Apr/Jul/Oct). Getting this wrong yields an empty join and
        # a silently blank feature — which is exactly how the first draft failed.
        out = br._decumulate_ytd({
            "20250331": 3.0, "20250630": 3.0, "20250930": 3.0, "20251231": 3.0,
        })
        self.assertEqual(sorted(out), ["2025-01", "2025-04", "2025-07", "2025-10"])

    def test_flat_ytd_implies_flat_quarters(self):
        # A constant YTD ratio means every quarter was identical.
        out = br._decumulate_ytd({
            "20250331": 3.0, "20250630": 3.0, "20250930": 3.0, "20251231": 3.0,
        })
        for q in out:
            self.assertAlmostEqual(out[q], 3.0, places=6)

    def test_rising_ytd_understates_the_true_quarter(self):
        # The bug this guards: YTD 3.00 -> 3.20 looks like +0.20, but the
        # standalone Q2 is 3.40, so the true quarterly change is +0.40 — double.
        out = br._decumulate_ytd({"20250331": 3.00, "20250630": 3.20})
        self.assertAlmostEqual(out["2025-04"], 3.40, places=6)
        naive_delta = 3.20 - 3.00
        true_delta = out["2025-04"] - out["2025-01"]
        self.assertAlmostEqual(true_delta, 2 * naive_delta, places=6)

    def test_q3_and_q4_use_the_right_predecessor(self):
        # Q3 standalone = YTD_3*3 - YTD_2*2 ; Q4 = YTD_4*4 - YTD_3*3
        ytd = {"20250331": 3.0, "20250630": 3.2, "20250930": 3.3, "20251231": 3.1}
        out = br._decumulate_ytd(ytd)
        self.assertAlmostEqual(out["2025-07"], 3.3 * 3 - 3.2 * 2, places=6)
        self.assertAlmostEqual(out["2025-10"], 3.1 * 4 - 3.3 * 3, places=6)

    def test_missing_predecessor_is_dropped_not_guessed(self):
        # A reporting hole must not become a fabricated quarter.
        out = br._decumulate_ytd({"20250331": 3.0, "20250930": 3.3})
        self.assertIn("2025-01", out)
        self.assertNotIn("2025-07", out)

    def test_none_and_off_cycle_dates_ignored(self):
        out = br._decumulate_ytd({"20250331": None, "20250215": 9.9, "20250630": 3.2})
        self.assertEqual(out, {})


class TestOLS(unittest.TestCase):
    """The stdlib least-squares solver."""

    def test_recovers_exact_coefficients(self):
        # y = 2*x1 - 3*x2 + 5, noiseless -> exact recovery, R2 == 1.
        y, X = [], []
        for x1 in range(1, 7):
            for x2 in range(1, 7):
                y.append(2.0 * x1 - 3.0 * x2 + 5.0)
                X.append([float(x1), float(x2), 1.0])
        betas, ses, ts, r2, n = br._ols(y, X)
        self.assertAlmostEqual(betas[0], 2.0, places=6)
        self.assertAlmostEqual(betas[1], -3.0, places=6)
        self.assertAlmostEqual(betas[2], 5.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)
        self.assertEqual(n, 36)

    def test_singular_design_returns_none(self):
        # Perfectly collinear columns must not raise or emit garbage.
        y = [1.0, 2.0, 3.0, 4.0]
        X = [[1.0, 2.0, 1.0], [2.0, 4.0, 1.0], [3.0, 6.0, 1.0], [4.0, 8.0, 1.0]]
        self.assertIsNone(br._ols(y, X))

    def test_underdetermined_returns_none(self):
        self.assertIsNone(br._ols([1.0, 2.0], [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]))

    def test_standard_errors_are_positive_and_finite(self):
        y = [1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8]
        X = [[float(i), 1.0] for i in range(1, 9)]
        betas, ses, ts, r2, n = br._ols(y, X)
        for se in ses:
            self.assertGreater(se, 0.0)
            self.assertEqual(se, se)          # not NaN
        self.assertGreater(abs(ts[0]), 2.0)   # a clean trend must be significant


class TestBuildDesign(unittest.TestCase):
    """Design-matrix assembly and its gap handling."""

    def _curve(self, quarters):
        return ({q: 1.0 + i * 0.1 for i, q in enumerate(quarters)},
                {q: 2.0 - i * 0.05 for i, q in enumerate(quarters)})

    def test_drops_first_two_quarters(self):
        qs = ["2020-01", "2020-04", "2020-07", "2020-10"]
        dep = {q: 3.0 for q in qs}
        r3m, slope = self._curve(qs)
        y, X, used = br.build_design(dep, r3m, slope)
        self.assertEqual(used, ["2020-07", "2020-10"])   # two lags consumed
        self.assertEqual(len(y), 2)

    def test_column_order_matches_TERMS(self):
        qs = ["2020-01", "2020-04", "2020-07"]
        dep = {"2020-01": 3.0, "2020-04": 3.1, "2020-07": 3.2}
        r3m = {"2020-01": 1.0, "2020-04": 1.5, "2020-07": 2.0}
        slope = {"2020-01": 2.0, "2020-04": 1.8, "2020-07": 1.5}
        y, X, used = br.build_design(dep, r3m, slope)
        row = X[0]
        self.assertEqual(len(row), len(br.TERMS))
        self.assertAlmostEqual(row[0], 3.1)              # ar1  = dep at t-1
        self.assertAlmostEqual(row[1], 2.0)              # r3m level at t
        self.assertAlmostEqual(row[2], 1.5 - 1.0)        # d_r3m over t-2 -> t-1
        self.assertAlmostEqual(row[3], 1.5)              # slope level at t
        self.assertAlmostEqual(row[4], 1.8 - 2.0)        # d_slope over t-2 -> t-1
        self.assertAlmostEqual(row[5], 1.0)              # const

    def test_reporting_gap_is_skipped(self):
        # A missing quarter must not become a fake one-quarter difference that
        # is really a two-quarter move.
        qs = ["2020-01", "2020-04", "2021-01", "2021-04", "2021-07"]
        dep = {q: 3.0 for q in qs}
        r3m, slope = self._curve(qs)
        y, X, used = br.build_design(dep, r3m, slope)
        self.assertNotIn("2021-01", used)
        self.assertNotIn("2021-04", used)
        self.assertIn("2021-07", used)

    def test_start_filter_trims_the_sample(self):
        qs = ["2019-01", "2019-04", "2019-07", "2019-10", "2020-01", "2020-04"]
        dep = {q: 3.0 for q in qs}
        r3m, slope = self._curve(qs)
        _, _, used = br.build_design(dep, r3m, slope, start="2020-01")
        self.assertEqual(used, ["2020-01", "2020-04"])

    def test_months_between(self):
        self.assertEqual(br._months_between("2020-01", "2020-04"), 3)
        self.assertEqual(br._months_between("2020-10", "2021-01"), 3)
        self.assertEqual(br._months_between("2020-01", "2021-01"), 12)


class TestPaperUnitTransfer(unittest.TestCase):
    """The paper's printed coefficients -> annualised percentage points.

    Table A says rates are in per cent; the coefficient interpretations only
    reconcile if they entered in basis points. We anchor on the four economic
    magnitudes the paper states in prose. If someone edits the scaling constant,
    these fail rather than silently rescaling every number in the feature.
    """

    def test_nim_short_rate_matches_stated_magnitude(self):
        # p21: +100bp short rate -> +0.035pp per quarter, 9.2% of mean 0.374.
        quarterly = br.PAPER["nim"]["r3m"] / 4.0
        self.assertAlmostEqual(quarterly, 0.035, places=4)
        self.assertAlmostEqual(quarterly / br.PAPER["nim"]["mean_dep_q"], 0.092,
                               places=2)

    def test_nim_slope_matches_stated_magnitude(self):
        # p21: +100bp slope -> around 8% of the mean flow.
        quarterly = br.PAPER["nim"]["slope"] / 4.0
        self.assertAlmostEqual(quarterly / br.PAPER["nim"]["mean_dep_q"], 0.080,
                               places=2)

    def test_operating_profit_matches_stated_magnitudes(self):
        # p31: +100bp short rate -> +0.04pp/quarter = 14.4% of mean 0.267;
        #      +100bp slope      -> around 18% of the mean.
        mean = br.PAPER["roa"]["mean_dep_q"]
        self.assertAlmostEqual(br.PAPER["roa"]["r3m"] / 4.0, 0.039, places=3)
        self.assertAlmostEqual((br.PAPER["roa"]["r3m"] / 4.0) / mean, 0.144,
                               places=2)
        self.assertAlmostEqual((br.PAPER["roa"]["slope"] / 4.0) / mean, 0.180,
                               places=2)

    def test_banking_book_and_trading_book_have_opposite_signs(self):
        # The paper's hedging evidence: level and slope move NII and trading
        # income in opposite directions. If this ever agrees in sign, the
        # hedging story in the UI is wrong.
        self.assertGreater(br.PAPER["nim"]["slope"], 0)
        self.assertLess(br.PAPER["trading"]["slope"], 0)
        self.assertGreater(br.PAPER["nim"]["r3m"], 0)
        self.assertLess(br.PAPER["trading"]["r_ib"], 0)

    def test_levels_positive_differences_negative(self):
        # The core qualitative claim of the paper, for both dependent variables.
        for kind in ("nim", "roa"):
            self.assertGreater(br.PAPER[kind]["r3m"], 0, kind)
            self.assertGreater(br.PAPER[kind]["slope"], 0, kind)
            self.assertLess(br.PAPER[kind]["d_r3m_l1"], 0, kind)
            self.assertLess(br.PAPER[kind]["d_slope_l1"], 0, kind)


class TestImpulseResponse(unittest.TestCase):
    """The scenario engine — where the paper's result can silently invert."""

    def setUp(self):
        self.paper = br._paper_coefs("nim")

    def test_reproduces_the_papers_chart5_impact(self):
        # The paper's own experiment (their Section 9.1): +100bp to the 3m rate,
        # the 10y rises only ~27bp so the curve flattens by ~73bp, shock decays
        # at 0.66. Their Chart 5 shows an impact of about -0.024pp on the
        # quarterly margin. This is the single strongest validation available
        # that both the unit transfer and the timing convention are right.
        irf = br.impulse_response(self.paper, d_r3m_bp=100, d_slope_bp=-73,
                                  horizon=12, persistence=0.66)
        impact_quarterly = irf[0]["effect_pp"] / 4.0
        self.assertAlmostEqual(impact_quarterly, -0.024, places=2)

    def test_impact_quarter_is_negative_under_paper_timing(self):
        # THE headline property. A rate rise must compress the margin first.
        irf = br.impulse_response(self.paper, d_r3m_bp=100, horizon=8,
                                  persistence=0.66)
        self.assertLess(irf[0]["effect_pp"], 0.0)

    def test_long_run_turns_positive(self):
        # ...and must recover once repricing works through.
        irf = br.impulse_response(self.paper, d_r3m_bp=100, horizon=12,
                                  persistence=1.0)
        self.assertLess(irf[0]["effect_pp"], 0.0)
        self.assertGreater(irf[-1]["effect_pp"], 0.0)
        self.assertGreater(irf[-1]["cum_pp"], 0.0)

    def test_anticipated_timing_flips_the_impact_sign(self):
        # Documents exactly what the timing switch does, so the default is a
        # deliberate choice rather than an accident.
        kw = dict(d_r3m_bp=100, horizon=8, persistence=0.66)
        unanticipated = br.impulse_response(self.paper, timing="unanticipated", **kw)
        anticipated = br.impulse_response(self.paper, timing="anticipated", **kw)
        self.assertLess(unanticipated[0]["effect_pp"], 0.0)
        self.assertGreater(anticipated[0]["effect_pp"], 0.0)

    def test_steepening_alone_is_positive_in_the_long_run(self):
        # A pure steepening with no short-rate move: the slope level term
        # dominates once the one-off difference term has passed.
        irf = br.impulse_response(self.paper, d_slope_bp=100, horizon=12,
                                  persistence=1.0)
        self.assertGreater(irf[-1]["effect_pp"], 0.0)
        self.assertGreater(irf[-1]["cum_pp"], 0.0)

    def test_zero_shock_is_a_flat_zero_path(self):
        irf = br.impulse_response(self.paper, horizon=6)
        for row in irf:
            self.assertAlmostEqual(row["effect_pp"], 0.0, places=12)
            self.assertAlmostEqual(row["cum_pp"], 0.0, places=12)

    def test_linear_in_shock_size(self):
        # The model is linear, so doubling the shock doubles the response.
        a = br.impulse_response(self.paper, d_slope_bp=50, horizon=6, persistence=0.9)
        b = br.impulse_response(self.paper, d_slope_bp=100, horizon=6, persistence=0.9)
        for ra, rb in zip(a, b):
            self.assertAlmostEqual(2 * ra["effect_pp"], rb["effect_pp"], places=10)

    def test_cumulative_is_the_running_sum(self):
        irf = br.impulse_response(self.paper, d_r3m_bp=100, d_slope_bp=-50,
                                  horizon=10, persistence=0.7)
        running = 0.0
        for row in irf:
            running += row["effect_pp"]
            self.assertAlmostEqual(row["cum_pp"], running, places=10)

    def test_decaying_shock_dies_out(self):
        irf = br.impulse_response(self.paper, d_r3m_bp=100, horizon=24,
                                  persistence=0.5)
        self.assertLess(abs(irf[-1]["effect_pp"]), 1e-4)

    def test_separate_slope_persistence_is_honoured(self):
        # A flattening that unwinds slowly must not give the same path as one
        # that snaps back with the short rate.
        fast = br.impulse_response(self.paper, d_r3m_bp=100, d_slope_bp=-73,
                                   horizon=12, persistence=0.66,
                                   slope_persistence=0.66)
        slow = br.impulse_response(self.paper, d_r3m_bp=100, d_slope_bp=-73,
                                   horizon=12, persistence=0.66,
                                   slope_persistence=0.95)
        self.assertNotAlmostEqual(fast[6]["cum_pp"], slow[6]["cum_pp"], places=4)
        # A slope that stays flat longer is worse for the margin.
        self.assertLess(slow[6]["cum_pp"], fast[6]["cum_pp"])

    def test_accepts_plain_float_coefficients(self):
        # estimate() returns {"b":..} dicts, PAPER holds bare floats; both must work.
        flat = {"ar1": 0.5, "r3m": 0.1, "slope": 0.1,
                "d_r3m_l1": -0.2, "d_slope_l1": -0.1}
        irf = br.impulse_response(flat, d_r3m_bp=100, horizon=4)
        self.assertEqual(len(irf), 4)
        self.assertLess(irf[0]["effect_pp"], 0.0)


class TestEstimate(unittest.TestCase):
    """End-to-end estimation on synthetic data with known coefficients."""

    def _synthetic(self, n=80, ar=0.5, b_r3m=0.10, b_slope=0.20,
                   b_dr3m=-0.15, b_dslope=-0.05):
        quarters, r3m, slope = [], {}, {}
        for i in range(n):
            y, m = 2000 + i // 4, (i % 4) * 3 + 1
            q = "%d-%02d" % (y, m)
            quarters.append(q)
            # Deterministic but non-collinear rate paths.
            r3m[q] = 3.0 + 1.5 * ((i * 7) % 11) / 11.0
            slope[q] = 1.0 + 1.2 * ((i * 5) % 13) / 13.0
        dep = {quarters[0]: 3.0, quarters[1]: 3.0}
        for i in range(2, n):
            q, p1, p2 = quarters[i], quarters[i - 1], quarters[i - 2]
            dep[q] = (ar * dep[p1] + b_r3m * r3m[q] + b_dr3m * (r3m[p1] - r3m[p2])
                      + b_slope * slope[q] + b_dslope * (slope[p1] - slope[p2]) + 0.4)
        return dep, r3m, slope

    def test_recovers_known_coefficients(self):
        dep, r3m, slope = self._synthetic()
        est = br.estimate(dep, r3m, slope)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est["ar1"]["b"], 0.50, places=5)
        self.assertAlmostEqual(est["r3m"]["b"], 0.10, places=5)
        self.assertAlmostEqual(est["slope"]["b"], 0.20, places=5)
        self.assertAlmostEqual(est["d_r3m_l1"]["b"], -0.15, places=5)
        self.assertAlmostEqual(est["d_slope_l1"]["b"], -0.05, places=5)
        self.assertAlmostEqual(est["_meta"]["r2"], 1.0, places=6)

    def test_long_run_multiplier(self):
        # LR = b / (1 - ar): 0.20 / 0.5 = 0.40 for the slope.
        dep, r3m, slope = self._synthetic()
        est = br.estimate(dep, r3m, slope)
        self.assertAlmostEqual(est["_meta"]["lr_slope"], 0.40, places=4)
        self.assertAlmostEqual(est["_meta"]["lr_r3m"], 0.20, places=4)

    def test_too_few_observations_returns_none(self):
        dep, r3m, slope = self._synthetic(n=20)
        self.assertIsNone(br.estimate(dep, r3m, slope))

    def test_meta_reports_the_actual_window(self):
        dep, r3m, slope = self._synthetic()
        est = br.estimate(dep, r3m, slope)
        self.assertEqual(est["_meta"]["first_q"], "2000-07")
        self.assertGreaterEqual(est["_meta"]["n"], br._MIN_OBS)


class TestDollars(unittest.TestCase):
    def test_converts_annualised_pp_to_quarterly_dollars(self):
        # 0.10pp annualised on $1bn of assets = $1m/yr = $250k/quarter.
        # FDIC ASSET is in $ thousands, so $1bn -> 1_000_000.
        self.assertAlmostEqual(br.dollars_per_quarter(0.10, 1_000_000),
                               250_000.0, places=2)

    def test_none_assets_returns_none(self):
        self.assertIsNone(br.dollars_per_quarter(0.10, None))

    def test_sign_is_preserved(self):
        self.assertLess(br.dollars_per_quarter(-0.10, 1_000_000), 0)


class TestResilienceMetrics(unittest.TestCase):
    """Layers 2-3: funding fragility and mark-to-market capital erosion.

    The property that must never drift: a LOSS is NEGATIVE (fair value below
    carrying cost), and a missing field yields None, never zero — a bank that
    does not report a field has no metric, not a perfect score.
    """

    ROW = {
        "DEP": 1_000_000.0,        # $1bn of deposits (FDIC units: $ thousands)
        "DEPUNA": 450_000.0,       # 45% uninsured
        "DEPNIDOM": 300_000.0,     # 30% non-interest-bearing
        "SCHA": 200_000.0,         # HTM amortized cost
        "SCHF": 180_000.0,         # HTM fair value  -> -20,000 unrealized
        "SCAF": 95_000.0,          # AFS fair value
        "SCAA": 100_000.0,         # AFS amortized   -> -5,000 unrealized
        "RBCT1": 100_000.0,        # Tier 1 capital
        "RBCT1CER": 12.5,
    }

    def test_ratios(self):
        m = br.resilience_metrics(self.ROW)
        self.assertAlmostEqual(m["uninsured_pct"], 45.0, places=6)
        self.assertAlmostEqual(m["nib_pct"], 30.0, places=6)
        self.assertAlmostEqual(m["cet1_ratio"], 12.5, places=6)

    def test_loss_is_negative(self):
        m = br.resilience_metrics(self.ROW)
        self.assertAlmostEqual(m["htm_unreal_thousands"], -20_000.0, places=6)
        self.assertAlmostEqual(m["afs_unreal_thousands"], -5_000.0, places=6)
        self.assertAlmostEqual(m["mtm_total_thousands"], -25_000.0, places=6)
        # -25,000 / 100,000 Tier 1 = -25% — marking to market erodes capital.
        self.assertAlmostEqual(m["mtm_over_t1_pct"], -25.0, places=6)

    def test_gain_is_positive(self):
        row = dict(self.ROW, SCHF=210_000.0, SCAF=100_000.0, SCAA=95_000.0)
        m = br.resilience_metrics(row)
        self.assertGreater(m["htm_unreal_thousands"], 0)
        self.assertGreater(m["mtm_over_t1_pct"], 0)

    def test_missing_field_is_none_not_zero(self):
        row = dict(self.ROW)
        del row["DEPUNA"]
        m = br.resilience_metrics(row)
        self.assertIsNone(m["uninsured_pct"])
        # ...and the other metrics are unaffected.
        self.assertAlmostEqual(m["nib_pct"], 30.0, places=6)

    def test_partial_securities_book_still_totals(self):
        # A bank with no HTM book (GS-like) still gets an AFS-only MTM figure.
        row = dict(self.ROW)
        row["SCHA"] = row["SCHF"] = None
        m = br.resilience_metrics(row)
        self.assertIsNone(m["htm_unreal_thousands"])
        self.assertAlmostEqual(m["mtm_total_thousands"], -5_000.0, places=6)
        self.assertAlmostEqual(m["mtm_over_t1_pct"], -5.0, places=6)

    def test_no_securities_data_at_all(self):
        row = {"DEP": 1_000_000.0, "DEPUNA": 100_000.0}
        m = br.resilience_metrics(row)
        self.assertIsNone(m["mtm_total_thousands"])
        self.assertIsNone(m["mtm_over_t1_pct"])

    def test_zero_deposits_no_division(self):
        row = dict(self.ROW, DEP=0.0)
        m = br.resilience_metrics(row)
        self.assertIsNone(m["uninsured_pct"])
        self.assertIsNone(m["nib_pct"])

    def test_zero_tier1_no_division(self):
        row = dict(self.ROW, RBCT1=0.0)
        m = br.resilience_metrics(row)
        self.assertIsNone(m["mtm_over_t1_pct"])

    def test_empty_row(self):
        m = br.resilience_metrics({})
        for v in m.values():
            self.assertIsNone(v)


class TestUniverseAndGuardrails(unittest.TestCase):
    def test_certs_are_unique(self):
        certs = [c for c, _ in br.BANK_CERTS.values()]
        self.assertEqual(len(certs), len(set(certs)))

    def test_slope_definition_is_the_papers(self):
        # The paper defines SLOPE = R10y - R3m (their Section 5.2). The app's
        # macro score uses 10y-2y elsewhere; both must stay available so the
        # two screens can be reconciled rather than silently disagree.
        self.assertEqual(br.FRED_SERIES["r3m"], "DGS3MO")
        self.assertEqual(br.FRED_SERIES["r10y"], "DGS10")
        self.assertEqual(br.FRED_SERIES["r2y"], "DGS2")

    def test_caveats_are_present(self):
        # These are shown in the UI. An empty list means the disclaimers were
        # dropped, which is a correctness problem, not a cosmetic one.
        self.assertGreaterEqual(len(br.PAPER_CAVEATS), 4)
        joined = " ".join(br.PAPER_CAVEATS).lower()
        self.assertIn("holding company", joined)
        self.assertIn("equity return", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
