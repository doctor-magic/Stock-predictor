"""
bank_rates.py — Yield curve → bank profitability.

OBSERVATIONAL / DISPLAY ONLY. Not a gate, not logged to setup_log.db, not a
covariate for any scanner verdict. Follows the Sector Heatmap precedent
(CLAUDE.md, Jul 15 2026): this module may never influence a BUY/HOLD decision.
Promotion to any gate is a milestone-bundle decision, never a mid-collection tweak.

What it does
------------
Estimates how the level and the slope of the Treasury curve move US banks'
net interest margin (NIM) and return on assets (ROA), using the specification
from:

    Alessandri, P and Nelson, B (2012), "Simple banking: profitability and the
    yield curve", Bank of England Working Paper No. 452.

The paper's specification (their equation 4, parametrised as on their page 20):

    y_t = a*y_{t-1} + b0*R3m_t + b1*dR3m_{t-1} + g0*SLOPE_t + g1*dSLOPE_{t-1} + c

Levels terms (R3m, SLOPE) carry the LONG-run effect; difference terms
(dR3m, dSLOPE) carry the SHORT-run repricing friction. The paper's finding —
which this module re-tests on US data — is that levels enter positively and
differences negatively: a rate rise compresses the margin on impact, then
lifts it once the loan book reprices.

Two coefficient sets are served side by side:
    "paper"  — the UK 1992-2009 estimates, transferred as-is
    "us"     — re-estimated per bank on FDIC data, 1995-present

Data sources
------------
FRED  (needs FRED_API_KEY)  DGS3MO, DGS2, DGS10 — quarterly averages for the
                           model, plus the latest DAILY observation of the same
                           three series for display (see fetch_curve_live).
FDIC  (no key required)     api.fdic.gov/banks/financials — NIMY, ROA, ROE, ASSET.

FRED's own aggregate bank series (USNIM/USROA/USROE) are DISCONTINUED — they
stop at 2020-07-01. Do not reintroduce them; the FDIC API is the live source.

Caveats that must stay visible in the UI (see PAPER_CAVEATS)
------------------------------------------------------------
1. FDIC reports the insured bank subsidiary, not the listed holding company.
   JPMorgan Chase Bank NA is not JPM the stock.
2. This models an interest MARGIN, not an equity return. The step from margin
   to share price is not made here and must not be made silently elsewhere.
3. FDIC NIMY/ROA are year-to-date annualised. We de-cumulate to standalone
   quarters (see _decumulate_ytd) — exact only if average assets are stable
   within the year, so single-quarter estimates carry extra noise.

Resilience layers (added same day, after review)
------------------------------------------------
The NIM model answers "who benefits from a steeper curve". It does not answer
"who survives getting there". Two cross-sectional layers close that gap, both
computed from the SAME FDIC fetch (no extra calls):

Layer 2 — funding fragility (Drechsler, Savov & Schnabl's deposit-franchise
    lens, cross-sectional proxies only — we do NOT fit their nonlinear beta):
      uninsured/total deposits (DEPUNA/DEP)  — flight-prone funding
      non-interest-bearing/total (DEPNIDOM/DEP) — cheap sticky franchise

Layer 3 — capital at risk (the Jiang et al. 2023 / SVB mark-to-market lens):
      HTM unrealized  = SCHF − SCHA   (fair value − amortized cost)
      AFS unrealized  = SCAF − SCAA
      mtm_over_t1_pct = (HTM + AFS) / RBCT1
    Validation anchor: ZION 2022Q4 computes to −21.8% of Tier 1 — reproducing
    the known regional-bank stress ranking of that quarter. JPM same quarter:
    −17.4%, recovered to −7.0% by 2026Q1.

These are DESCRIPTIVE DISPLAY metrics — percentile shading in the UI, never a
score, never a threshold, never a gate. See RESILIENCE_CAVEATS.

Stdlib only — no numpy, no pandas, no statsmodels (this box is RAM-constrained;
see the VM downgrade note in CLAUDE.md). The OLS is a small Gauss-Jordan solve.

Run the estimator standalone:
    cd ~/Desktop/Stock-predictor && python3 bank_rates.py
"""

import json as _json
import math
import os
import time
import urllib.request as _req

# ── Paper coefficients ───────────────────────────────────────────────────────
#
# UNITS NOTE (important, and genuinely ambiguous in the source).
# The paper's Table A reports rates "in per cent" (R3m mean 6.064), but its
# coefficient interpretations only reconcile if rates entered the regressions in
# BASIS POINTS. The text states the economic magnitude three separate times and
# it is internally consistent across both dependent variables:
#
#   NIM,     Table D(3): +100bp short rate -> +0.035pp/quarter = 9.2% of mean 0.374  (p21)
#   NIM,     Table D(3): +100bp slope      -> ~8% of mean                            (p21)
#   OpProf,  Table K(5): +100bp short rate -> +0.04pp/quarter = 14.4% of mean 0.267  (p31)
#   OpProf,  Table K(5): +100bp slope      -> ~18% of mean                           (p31)
#
# So we anchor on the ECONOMIC statement, not on the ambiguous table units:
# printed coefficient x 100 = percentage points of QUARTERLY margin per 100bp.
# We then x4 to put everything on the ANNUALISED basis that FDIC NIMY uses,
# so paper and US coefficients are directly comparable.
_Q_COEF_TO_ANNUAL_PP = 100.0 * 4.0

PAPER = {
    "source": "Alessandri & Nelson (2012), BoE Working Paper No. 452",
    "sample": "UK banking groups, 1992 Q1 – 2009 Q3",
    "nim": {
        # Table D, model (3) — major UK banks, fixed effects, quarterly NIM/TA
        "table": "Table D model (3), major UK banks (MUK), fixed effects",
        "ar1": 0.35533,
        "r3m": 0.00035 * _Q_COEF_TO_ANNUAL_PP,          # +0.14 pp annualised per 100bp
        "d_r3m_l0": 0.00015 * _Q_COEF_TO_ANNUAL_PP,     # not significant (t=0.57)
        "d_r3m_l1": -0.00055 * _Q_COEF_TO_ANNUAL_PP,    # -0.22, t=-2.00
        "slope": 0.00030 * _Q_COEF_TO_ANNUAL_PP,        # +0.12, t=3.06
        "d_slope_l0": -0.00013 * _Q_COEF_TO_ANNUAL_PP,  # not significant (t=-1.00)
        "d_slope_l1": -0.00025 * _Q_COEF_TO_ANNUAL_PP,  # not significant (t=-1.40)
        "mean_dep_q": 0.374,                            # Table A, quarterly NIM/TA
    },
    "roa": {
        # Table K, column (5) — major UK banks, System GMM, operating profit/TA
        "table": "Table K column (5), major UK banks (MUK), System GMM",
        "ar1": -0.07610,
        "ar2": 0.00899,
        "r3m": 0.00039 * _Q_COEF_TO_ANNUAL_PP,
        "d_r3m_l0": 0.00022 * _Q_COEF_TO_ANNUAL_PP,
        "d_r3m_l1": -0.00062 * _Q_COEF_TO_ANNUAL_PP,    # t=-2.59
        "slope": 0.00048 * _Q_COEF_TO_ANNUAL_PP,        # t=2.71
        "d_slope_l0": -0.00034 * _Q_COEF_TO_ANNUAL_PP,
        "d_slope_l1": -0.00030 * _Q_COEF_TO_ANNUAL_PP,
        "mean_dep_q": 0.267,                            # Table A, quarterly OpProf/TA
    },
    "trading": {
        # Table J, column (3) — the hedging offset. Level and slope enter the
        # TRADING book with the OPPOSITE sign to the banking book, which is the
        # paper's evidence that banks hedge rate risk through derivatives.
        "table": "Table J column (3), major UK banks, static fixed effects",
        "r_ib": -0.00031 * _Q_COEF_TO_ANNUAL_PP,        # t=-3.01
        "slope": -0.00033 * _Q_COEF_TO_ANNUAL_PP,       # t=-2.66
        "mean_dep_q": 0.029,
    },
}

PAPER_CAVEATS = [
    "The paper estimates UK banking groups, 1992–2009. Applying it to US banks "
    "in 2026 is an out-of-sample transfer, not a validated forecast.",
    "FDIC data covers the insured bank subsidiary, not the listed holding "
    "company — JPMorgan Chase Bank NA is not JPM the stock.",
    "This models an interest margin, not an equity return. Margin does not "
    "translate mechanically into share price.",
    "FDIC NIMY/ROA are year-to-date annualised and are de-cumulated here to "
    "standalone quarters; that step adds noise to single-quarter estimates.",
    "Per-bank OLS with a lagged dependent variable is biased downward in the AR "
    "term (Nickell bias, ~1/T). With T≈120 the bias is small, and the paper "
    "itself uses per-bank OLS for exactly this cross-section (their Section 6.4).",
]

RESILIENCE_CAVEATS = [
    "שיעור הפיקדונות הלא־מבוטחים ברמת חברת הבת כולל יתרות תפעוליות ובין־חברתיות "
    "— אצל בנק גדול עם עסקי משמורת וסיטונאות המספר אינו סיכון בריחה בסגנון SVB.",
    "הפסדי AFS כבר יושבים בהון החשבונאי (וברגולטורי רק אצל בנקי advanced-approaches); "
    "הפסדי HTM אינם מוכרים בשום מקום. הסכום MTM/T1 הוא ניסוי מחשבתי בנוסח "
    "Jiang et al (2023), לא מספר GAAP.",
    "מודל ה־NIM הליניארי סופג את ה־deposit beta הממוצע של כל בנק דרך המקדם שלו, "
    "אבל לא את אי־הליניאריות (Drechsler-Savov-Schnabl): ברמות ריבית גבוהות לחץ "
    "עלות המימון גדול ממה שהמקדם ההיסטורי מרמז.",
    "המדדים תיאוריים — צביעה לפי שליש חתכי בין 17 הבנקים, לא סף ולא ציון.",
]

# ── Universe ─────────────────────────────────────────────────────────────────
# ticker -> FDIC CERT of the main insured subsidiary. All verified active
# against api.fdic.gov/banks/institutions on 2026-08-20.
BANK_CERTS = {
    "JPM":  (628,   "JPMorgan Chase Bank NA"),
    "BAC":  (3510,  "Bank of America NA"),
    "WFC":  (3511,  "Wells Fargo Bank NA"),
    "C":    (7213,  "Citibank NA"),
    "USB":  (6548,  "U.S. Bank NA"),
    "PNC":  (6384,  "PNC Bank NA"),
    "TFC":  (9846,  "Truist Bank"),
    "COF":  (4297,  "Capital One NA"),
    "FITB": (6672,  "Fifth Third Bank NA"),
    "KEY":  (17534, "KeyBank NA"),
    "RF":   (12368, "Regions Bank"),
    "CFG":  (57957, "Citizens Bank NA"),
    "MTB":  (588,   "Manufacturers and Traders Trust"),
    "HBAN": (6560,  "The Huntington National Bank"),
    "ZION": (2270,  "Zions Bancorporation NA"),
    "GS":   (33124, "Goldman Sachs Bank USA"),
    "MS":   (32992, "Morgan Stanley Bank NA"),
}

FRED_SERIES = {"r3m": "DGS3MO", "r2y": "DGS2", "r10y": "DGS10"}

_ESTIMATION_START = "1995-01-01"
_MIN_OBS = 40                    # below this an estimate is not reported
_SUBSAMPLE_START = "2010-01"     # post-crisis cut, mirrors the paper's Table L

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bank_rates_cache.json")
_CACHE_TTL = 6 * 3600            # FDIC publishes quarterly; 6h is generous
# Bump when the payload shape changes — a stale disk cache from an older shape
# must be refetched, not served for 6 hours with fields silently missing.
_SCHEMA = 3                      # 3 = curve.live (daily quote) added
_cache = {"ts": 0, "data": None}


# ── Pure maths (unit-tested) ─────────────────────────────────────────────────

def _decumulate_ytd(ytd_by_repdte):
    """FDIC year-to-date annualised ratios -> standalone-quarter annualised.

    For quarter n of a year, YTD_n annualised = (sum of n quarters) / avg assets * 4/n,
    so the standalone quarter is  YTD_n * n - YTD_{n-1} * (n-1).

    Exact only if average assets are stable within the year. Q1 needs no
    de-cumulation. A quarter whose predecessor is missing is dropped rather
    than guessed.

    Input:  {"20260331": 2.91, "20260630": 2.94, ...}
    Output: {"2026-01": 2.91, "2026-04": ..., ...}   (FRED quarter-START keys)
    """
    out = {}
    for repdte, val in ytd_by_repdte.items():
        if val is None:
            continue
        year, month = repdte[:4], int(repdte[4:6])
        if month not in (3, 6, 9, 12):
            continue
        qn = month // 3
        # FDIC stamps the quarter END month; FRED stamps the quarter START month.
        key = "%s-%02d" % (year, month - 2)
        if qn == 1:
            out[key] = val
            continue
        prev = ytd_by_repdte.get("%s%02d%s" % (year, month - 3,
                                               "31" if month - 3 in (3, 12) else "30"))
        if prev is None:
            continue
        out[key] = val * qn - prev * (qn - 1)
    return out


def _ols(y, X):
    """Least squares with HC0-free classical standard errors.

    Returns (betas, ses, tstats, r2, n) or None if the design is singular.
    X includes its own intercept column; no column is added here.
    """
    n, k = len(y), len(X[0])
    if n <= k:
        return None

    xtx = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]

    # Solve and invert in one Gauss-Jordan pass: [XtX | I | Xty]
    aug = [xtx[r][:] + [1.0 if c == r else 0.0 for c in range(k)] + [xty[r]]
           for r in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(k):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [v - f * aug[col][j] for j, v in enumerate(aug[r])]

    betas = [aug[i][2 * k] for i in range(k)]
    xtx_inv = [[aug[i][k + j] for j in range(k)] for i in range(k)]

    resid = [y[i] - sum(betas[a] * X[i][a] for a in range(k)) for i in range(n)]
    ss_res = sum(e * e for e in resid)
    ybar = sum(y) / n
    ss_tot = sum((v - ybar) ** 2 for v in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sigma2 = ss_res / (n - k)
    ses, ts = [], []
    for i in range(k):
        var = sigma2 * xtx_inv[i][i]
        se = math.sqrt(var) if var > 0 else float("nan")
        ses.append(se)
        ts.append(betas[i] / se if se and se == se and se > 0 else float("nan"))

    return betas, ses, ts, r2, n


# Column order of the design matrix, and therefore of every coefficient list.
TERMS = ["ar1", "r3m", "d_r3m_l1", "slope", "d_slope_l1", "const"]


def build_design(dep_by_q, r3m_by_q, slope_by_q, start=None):
    """Assemble (y, X, quarters) for the paper's specification.

    Only quarters with a clean 3-month gap to both predecessors are used, so a
    reporting hole never silently becomes a fake one-quarter difference.
    """
    quarters = sorted(set(dep_by_q) & set(r3m_by_q) & set(slope_by_q))
    y, X, used = [], [], []
    for i, q in enumerate(quarters):
        if i < 2:
            continue
        p1, p2 = quarters[i - 1], quarters[i - 2]
        if _months_between(p1, q) != 3 or _months_between(p2, p1) != 3:
            continue
        if start and q < start:
            continue
        y.append(dep_by_q[q])
        X.append([
            dep_by_q[p1],                          # ar1
            r3m_by_q[q],                           # r3m level
            r3m_by_q[p1] - r3m_by_q[p2],           # d_r3m lag 1
            slope_by_q[q],                         # slope level
            slope_by_q[p1] - slope_by_q[p2],       # d_slope lag 1
            1.0,                                   # const
        ])
        used.append(q)
    return y, X, used


def _months_between(a, b):
    return (int(b[:4]) * 12 + int(b[5:7])) - (int(a[:4]) * 12 + int(a[5:7]))


def estimate(dep_by_q, r3m_by_q, slope_by_q, start=None):
    """Run the specification and return a coefficient dict, or None."""
    y, X, used = build_design(dep_by_q, r3m_by_q, slope_by_q, start=start)
    if len(y) < _MIN_OBS:
        return None
    fit = _ols(y, X)
    if fit is None:
        return None
    betas, ses, ts, r2, n = fit
    coefs = {t: {"b": betas[i], "se": ses[i], "t": ts[i]}
             for i, t in enumerate(TERMS)}
    coefs["_meta"] = {
        "n": n, "r2": r2,
        "first_q": used[0] if used else None,
        "last_q": used[-1] if used else None,
    }
    # Long-run multipliers: b / (1 - ar1). This is the steady-state effect the
    # paper's NIM equation implies once repricing has worked through.
    ar1 = betas[0]
    if abs(1.0 - ar1) > 1e-6:
        coefs["_meta"]["lr_r3m"] = betas[1] / (1.0 - ar1)
        coefs["_meta"]["lr_slope"] = betas[3] / (1.0 - ar1)
    return coefs


def impulse_response(coefs, d_r3m_bp=0.0, d_slope_bp=0.0, horizon=12,
                     persistence=1.0, slope_persistence=None,
                     timing="unanticipated"):
    """Path of the dependent variable after a curve shock.

    d_r3m_bp / d_slope_bp are the size of the shock in basis points.
    persistence is the quarterly AR decay of the shock itself (1.0 = permanent;
    the paper's VAR, their Chart 4, decays fast — around 0.66 for the 3m rate).

    Returns one dict per quarter with the deviation from baseline and the
    running cumulative, both in percentage points of the ANNUALISED dependent
    variable.

    TIMING — this is the subtle part, and it decides the sign on impact.

    "unanticipated" (default, and the paper's own convention, their p32):
        "We assume that the initial shock is unanticipated, and its period one
        effect is captured by the coefficient on Dr3m_{t-1}."
        So in the impact quarter the level has already jumped, and the quarter's
        rate CHANGE lands in the lag-1 difference slot. Because the paper's
        lag-1 difference coefficient (-0.00055) is larger in magnitude than its
        level coefficient (+0.00035), the impact quarter comes out NEGATIVE —
        the repricing friction that is the whole point of the paper. Calibrated
        to their Chart 5, this reproduces an impact of about -0.024pp on the
        quarterly margin, matching the chart.

    "anticipated":
        Standard regression timing, level and contemporaneous difference fire
        together at t=0. The margin then expands immediately. Offered so the
        assumption is visible and switchable rather than buried.

    Do not "fix" the unanticipated branch to standard timing: it would silently
    flip the sign of the headline result.
    """
    ar1 = coefs["ar1"]["b"] if isinstance(coefs["ar1"], dict) else coefs["ar1"]

    def _b(name):
        v = coefs.get(name)
        if v is None:
            return 0.0
        return v["b"] if isinstance(v, dict) else v

    r3m_pp, slope_pp = d_r3m_bp / 100.0, d_slope_bp / 100.0

    # Level path: full shock in the impact quarter, decaying from the next one.
    # The short rate and the slope get their own decay: in the paper's VAR
    # (their Chart 4) the 3m rate falls back faster than the 10y, so a shock
    # that flattens the curve does not unwind at the same speed it arrived.
    sp = persistence if slope_persistence is None else slope_persistence

    def _path(size, decay):
        return [size * (decay ** max(0, t - 1)) for t in range(horizon + 2)]

    lvl_r3m = _path(r3m_pp, persistence)
    lvl_slope = _path(slope_pp, sp)
    # First differences, with a zero pre-shock quarter so the jump itself is d[0].
    d_r3m = [lvl_r3m[0]] + [lvl_r3m[t] - lvl_r3m[t - 1] for t in range(1, len(lvl_r3m))]
    d_slope = [lvl_slope[0]] + [lvl_slope[t] - lvl_slope[t - 1] for t in range(1, len(lvl_slope))]

    # Under the paper's convention the current quarter's change occupies the
    # lag-1 slot; under standard timing it occupies the contemporaneous slot.
    if timing == "unanticipated":
        def slots(t):
            return (d_r3m[t - 1] if t >= 1 else 0.0, d_r3m[t],
                    d_slope[t - 1] if t >= 1 else 0.0, d_slope[t])
    else:
        def slots(t):
            return (d_r3m[t], d_r3m[t - 1] if t >= 1 else 0.0,
                    d_slope[t], d_slope[t - 1] if t >= 1 else 0.0)

    out, prev, cum = [], 0.0, 0.0
    for t in range(horizon):
        dr_l0, dr_l1, ds_l0, ds_l1 = slots(t)
        dev = (ar1 * prev
               + _b("r3m") * lvl_r3m[t]
               + _b("d_r3m_l0") * dr_l0
               + _b("d_r3m_l1") * dr_l1
               + _b("slope") * lvl_slope[t]
               + _b("d_slope_l0") * ds_l0
               + _b("d_slope_l1") * ds_l1)
        cum += dev
        out.append({"q": t, "effect_pp": dev, "cum_pp": cum})
        prev = dev
    return out


def resilience_metrics(row):
    """Layers 2-3 of the resilience matrix, from one latest-quarter FDIC row.

    Sign convention: unrealized amounts are fair value minus carrying cost, so
    a LOSS is NEGATIVE, and mtm_over_t1_pct negative means marking the whole
    securities book to market would erode that share of Tier 1 capital.

    AFS and HTM are kept separate in the output because they are not the same
    kind of number: AFS losses already sit in accounting equity via AOCI, while
    HTM losses are recognised nowhere — the combined figure is a Jiang et al.
    (2023) style thought experiment, not a GAAP quantity.

    Missing fields degrade to None, never to zero — a bank that does not report
    a field has no metric, not a perfect score.
    """
    def g(k):
        v = row.get(k)
        return float(v) if v is not None else None

    dep, una, nib = g("DEP"), g("DEPUNA"), g("DEPNIDOM")
    scha, schf = g("SCHA"), g("SCHF")
    scaf, scaa = g("SCAF"), g("SCAA")
    t1 = g("RBCT1")

    htm = (schf - scha) if (schf is not None and scha is not None) else None
    afs = (scaf - scaa) if (scaf is not None and scaa is not None) else None
    mtm = None
    if htm is not None or afs is not None:
        mtm = (htm or 0.0) + (afs or 0.0)

    return {
        "uninsured_pct": (una / dep * 100.0) if (una is not None and dep) else None,
        "nib_pct": (nib / dep * 100.0) if (nib is not None and dep) else None,
        "htm_unreal_thousands": htm,
        "afs_unreal_thousands": afs,
        "mtm_total_thousands": mtm,
        "mtm_over_t1_pct": (mtm / t1 * 100.0) if (mtm is not None and t1) else None,
        "cet1_ratio": g("RBCT1CER"),
    }


def dollars_per_quarter(effect_pp, assets_thousands):
    """Translate a margin impulse into quarterly dollars.

    effect_pp is annualised percentage points; FDIC ASSET is in $ thousands.
    """
    if assets_thousands is None:
        return None
    return (effect_pp / 100.0) * (assets_thousands * 1000.0) / 4.0


# ── I/O (every fetch is try/except -> stale-or-None; must never raise) ────────

def _fred_quarterly(series_id, api_key, start=_ESTIMATION_START):
    url = ("https://api.stlouisfed.org/fred/series/observations"
           "?series_id=%s&frequency=q&aggregation_method=avg"
           "&api_key=%s&file_type=json&observation_start=%s"
           % (series_id, api_key, start))
    with _req.urlopen(url, timeout=20) as r:
        obs = _json.loads(r.read())["observations"]
    return {o["date"][:7]: float(o["value"]) for o in obs if o["value"] != "."}


def _fred_latest(series_id, api_key, lookback=10):
    """Most recent non-missing DAILY observation. Returns (date, value) or None.

    FRED stamps holidays and non-trading days with "." — hence the lookback
    window rather than limit=1.
    """
    url = ("https://api.stlouisfed.org/fred/series/observations"
           "?series_id=%s&api_key=%s&file_type=json"
           "&sort_order=desc&limit=%d"
           % (series_id, api_key, lookback))
    with _req.urlopen(url, timeout=20) as r:
        obs = _json.loads(r.read())["observations"]
    for o in obs:
        if o["value"] != ".":
            return o["date"], float(o["value"])
    return None


def fetch_curve_live(api_key):
    """Latest daily Treasury quote. Returns dict or None. DISPLAY ONLY.

    This is NOT a model input and must never become one. The Alessandri &
    Nelson specification is estimated on quarterly data, so every coefficient,
    impulse response and scenario in this module reads the quarterly averages
    from fetch_curve. This block exists only so the UI can show the curve as it
    stands today next to the quarter it actually modelled — the two differ by
    the length of the open quarter, which was the Aug 21 2026 confusion.

    Sequential with sleeps, same FRED rate-limit rule as fetch_curve.
    """
    try:
        r3m = _fred_latest(FRED_SERIES["r3m"], api_key)
        time.sleep(0.5)
        r10y = _fred_latest(FRED_SERIES["r10y"], api_key)
        time.sleep(0.5)
        r2y = _fred_latest(FRED_SERIES["r2y"], api_key)
    except Exception:
        return None
    if not r3m or not r10y:
        return None
    return {
        # Series can settle on different days; the UI stamps the 10y date and
        # each value carries its own so a lagging series is never hidden.
        "as_of": max(d for d, _ in (r3m, r10y, r2y or r10y)),
        "r3m": r3m[1],
        "r3m_as_of": r3m[0],
        "r2y": r2y[1] if r2y else None,
        "r2y_as_of": r2y[0] if r2y else None,
        "r10y": r10y[1],
        "r10y_as_of": r10y[0],
        "slope": r10y[1] - r3m[1],
        "slope_2y": (r10y[1] - r2y[1]) if r2y else None,
    }


def fetch_curve(api_key):
    """Quarterly average Treasury curve. Returns dict or None.

    Sequential with a sleep between series, per the FRED rule in CLAUDE.md:
    concurrent FRED requests rate-limit at ~2 and return 429, which then gets
    cached as nulls for hours (the Jun 1 2026 incident). Three calls, so this
    costs one second on a fetch that happens at most every six hours.
    """
    try:
        r3m = _fred_quarterly(FRED_SERIES["r3m"], api_key)
        time.sleep(0.5)
        r10y = _fred_quarterly(FRED_SERIES["r10y"], api_key)
        time.sleep(0.5)
        r2y = _fred_quarterly(FRED_SERIES["r2y"], api_key)
    except Exception:
        return None
    quarters = sorted(set(r3m) & set(r10y))
    if not quarters:
        return None
    # SLOPE = 10y - 3m, the paper's own definition (their Section 5.2).
    # The app's macro score uses 10y-2y elsewhere; both are carried so the two
    # screens can be reconciled instead of silently disagreeing.
    return {
        "r3m": r3m,
        "r2y": r2y,
        "r10y": r10y,
        "slope": {q: r10y[q] - r3m[q] for q in quarters},
        "slope_2y": {q: r10y[q] - r2y[q] for q in quarters if q in r2y},
        "quarters": quarters,
    }


def fetch_bank_panel(cert):
    """FDIC quarterly financials for one CERT. Returns dict or None.

    One request carries both the estimation series (NIMY/ROA history) and the
    latest-quarter resilience fields — the extra columns are free, a second
    call per bank would not be.
    """
    url = ("https://api.fdic.gov/banks/financials?filters=CERT:%d"
           "&fields=REPDTE,NIMY,ROA,ROE,ASSET,"
           "DEP,DEPUNA,DEPNIDOM,SCHA,SCHF,SCAF,SCAA,RBCT1,RBCT1CER"
           "&sort_by=REPDTE&sort_order=ASC&limit=250&format=json" % cert)
    try:
        with _req.urlopen(url, timeout=30) as r:
            rows = _json.loads(r.read())["data"]
    except Exception:
        return None

    nim_ytd, roa_ytd, assets = {}, {}, {}
    latest_rd, latest_row = "", {}
    for row in rows:
        d = row.get("data") or {}
        rd = d.get("REPDTE")
        if not rd:
            continue
        if rd > latest_rd:
            latest_rd, latest_row = rd, d
        if d.get("NIMY") is not None:
            nim_ytd[rd] = float(d["NIMY"])
        if d.get("ROA") is not None:
            roa_ytd[rd] = float(d["ROA"])
        if d.get("ASSET") is not None:
            assets[rd] = float(d["ASSET"])

    if not nim_ytd:
        return None
    last = max(assets) if assets else None
    return {
        "nim": _decumulate_ytd(nim_ytd),
        "roa": _decumulate_ytd(roa_ytd),
        "nim_latest_ytd": nim_ytd[max(nim_ytd)],
        "roa_latest_ytd": roa_ytd[max(roa_ytd)] if roa_ytd else None,
        "assets_thousands": assets.get(last),
        "as_of": last,
        "resilience": resilience_metrics(latest_row),
    }


# ── Assembly ─────────────────────────────────────────────────────────────────

def _load_disk_cache():
    try:
        with open(_CACHE_PATH) as f:
            d = _json.load(f)
        if d and d.get("schema") == _SCHEMA:
            _cache["data"] = d
            # Treat a disk cache as an hour from expiry so a restart refreshes soon.
            _cache["ts"] = time.time() - _CACHE_TTL + 3600
    except Exception:
        pass


def _save_disk_cache(data):
    try:
        with open(_CACHE_PATH, "w") as f:
            _json.dump(data, f)
    except Exception:
        pass


_load_disk_cache()


def get_bank_rates(api_key, force=False):
    """Full payload for the Yield Curve → Banks tab.

    Never raises: on a total failure it returns the last good cache, or a
    payload with error set and banks empty.
    """
    now = time.time()
    if not force and _cache["data"] and now - _cache["ts"] < _CACHE_TTL:
        return _cache["data"]

    curve = fetch_curve(api_key) if api_key else None
    if curve is None:
        if _cache["data"]:
            return dict(_cache["data"], stale=True)
        return {"error": "curve_unavailable", "banks": [], "curve": None}

    # Display-only daily quote. A failure here degrades to no live row in the
    # UI; it must never block the quarterly payload the model depends on.
    live = fetch_curve_live(api_key) if api_key else None

    qs = curve["quarters"]
    latest = qs[-1]
    prev = qs[-2] if len(qs) > 1 else latest

    banks = []
    for ticker, (cert, legal_name) in BANK_CERTS.items():
        panel = fetch_bank_panel(cert)
        if panel is None:
            banks.append({"ticker": ticker, "cert": cert, "name": legal_name,
                          "error": "fdic_unavailable"})
            continue

        entry = {
            "ticker": ticker,
            "cert": cert,
            "name": legal_name,
            "assets_thousands": panel["assets_thousands"],
            "as_of": panel["as_of"],
            "nim_latest_ytd": panel["nim_latest_ytd"],
            "roa_latest_ytd": panel["roa_latest_ytd"],
            "resilience": panel["resilience"],
            "nim_full": estimate(panel["nim"], curve["r3m"], curve["slope"]),
            "nim_post2010": estimate(panel["nim"], curve["r3m"], curve["slope"],
                                     start=_SUBSAMPLE_START),
            "roa_full": estimate(panel["roa"], curve["r3m"], curve["slope"]),
            # Realised history for the "did it actually hold" panel.
            "nim_history": [{"q": q, "nim": panel["nim"][q]}
                            for q in sorted(panel["nim"]) if q >= "2000-01"],
        }
        banks.append(entry)
        time.sleep(0.15)   # be polite to the FDIC API

    data = {
        "schema": _SCHEMA,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resilience_caveats": RESILIENCE_CAVEATS,
        "curve": {
            "latest_q": latest,
            "live": live,
            "r3m": curve["r3m"].get(latest),
            "r2y": curve["r2y"].get(latest),
            "r10y": curve["r10y"].get(latest),
            "slope": curve["slope"].get(latest),
            "slope_2y": curve["slope_2y"].get(latest),
            "d_slope_q": (curve["slope"].get(latest, 0.0)
                          - curve["slope"].get(prev, 0.0)),
            "d_r3m_q": (curve["r3m"].get(latest, 0.0)
                        - curve["r3m"].get(prev, 0.0)),
            "history": [{"q": q,
                         "r3m": curve["r3m"].get(q),
                         "r10y": curve["r10y"].get(q),
                         "slope": curve["slope"].get(q)}
                        for q in qs if q >= "2000-01"],
        },
        "paper": PAPER,
        "caveats": PAPER_CAVEATS,
        "banks": banks,
        "terms": TERMS,
    }

    ok = sum(1 for b in banks if b.get("nim_full"))
    if ok < 3 and _cache["data"]:
        return dict(_cache["data"], stale=True)

    _cache["ts"] = now
    _cache["data"] = data
    _save_disk_cache(data)
    return data


def _paper_coefs(kind="nim"):
    """Paper coefficients shaped like an estimate() result, for the scenario engine."""
    p = PAPER[kind]
    return {k: {"b": p[k], "se": None, "t": None}
            for k in ("ar1", "r3m", "d_r3m_l0", "d_r3m_l1",
                      "slope", "d_slope_l0", "d_slope_l1") if k in p}


if __name__ == "__main__":
    key = ""
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("FRED_API_KEY="):
                key = line.split("=", 1)[1].strip()
    key = key or os.getenv("FRED_API_KEY", "")

    d = get_bank_rates(key, force=True)
    c = d.get("curve") or {}
    print("curve %s:  R3m=%.2f  R10y=%.2f  SLOPE=%+.2f  (dSLOPE/q %+.2f)"
          % (c.get("latest_q"), c.get("r3m") or 0, c.get("r10y") or 0,
             c.get("slope") or 0, c.get("d_slope_q") or 0))
    print()
    print("%-6s %5s %7s %9s %9s %9s %9s %7s" %
          ("bank", "N", "AR", "R3m", "dR3m-1", "SLOPE", "dSLP-1", "R2"))
    for b in d.get("banks", []):
        e = b.get("nim_full")
        if not e:
            print("%-6s  %s" % (b["ticker"], b.get("error", "insufficient")))
            continue
        m = e["_meta"]
        print("%-6s %5d %7.3f %9.4f %9.4f %9.4f %9.4f %7.3f"
              % (b["ticker"], m["n"], e["ar1"]["b"], e["r3m"]["b"],
                 e["d_r3m_l1"]["b"], e["slope"]["b"], e["d_slope_l1"]["b"],
                 m["r2"]))
