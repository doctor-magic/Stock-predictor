"""
Walk-Forward Swing Model Backtest
Universe: 25 liquid stocks across S&P 500 sectors
Method: expanding window, quarterly steps (~63 trading days)
Label: close +3% in 10 trading days → BUY
Reports: OOS precision, avg forward return by signal, confidence filter impact
"""

import sys, os, time, warnings, argparse
sys.path.insert(0, os.path.expanduser("~/Desktop/Stock-predictor"))
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

from core_logic import build_features, build_labels, FEATURES, FORWARD_DAYS, THRESHOLD

_parser = argparse.ArgumentParser()
_parser.add_argument("--filtered", action="store_true", help="Use PREMIUM_UNIVERSE (14 regime-resilient stocks)")
_args = _parser.parse_args()

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",   # mega-cap tech
    "JPM", "GS", "BAC",                          # finance
    "JNJ", "UNH", "LLY",                         # health
    "XOM", "CVX",                                 # energy
    "COST", "WMT",                                # consumer
    "CAT", "HON",                                 # industrials
    "PLTR", "TSLA", "AMD",                        # high-vol momentum
    "AMT", "PG", "KO",                            # defensive
    "AMZN", "CRM",                                # cloud/e-comm
]

# Elite 9: filtered by per-symbol OOS precision >= 46% AND avg fwd ret >= +2.5%
# (swing_backtest May 2026, 14-stock PREMIUM run).
# Removed: LLY(28.4% prec), META(40.5%,+0.67%), PLTR(43.1%,+0.91%),
#          AMZN(41.7%,+1.77%), MSFT(10 signals only).
PREMIUM_UNIVERSE = [
    "GOOGL", "NVDA",                             # mega-cap tech
    "GS", "BAC",                                 # finance
    "XOM",                                       # energy
    "CAT",                                       # industrials
    "TSLA", "AMD",                               # high-vol momentum
    "AAPL",                                      # mega-cap
]

TRAIN_MIN_DAYS = 252   # ~1 year minimum before first test window
STEP_DAYS      = 63    # quarterly steps (~3 months)
N_STEPS        = 4     # 4 OOS quarters = ~1 year of testing
CONF_THRESHOLD = 0.65  # mirrors live model

HGBT_PARAMS = dict(
    max_iter=200, max_leaf_nodes=31, min_samples_leaf=50,
    l2_regularization=0.1, learning_rate=0.01,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, random_state=42,
)


def fetch_raw(sym: str) -> pd.DataFrame:
    raw = yf.download(sym, period="3y", progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close", "Volume"]].copy() if not raw.empty else pd.DataFrame()


def walk_forward(sym: str, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Return (oos_records, is_records) for one symbol.
    IS window = last STEP_DAYS of the training set — same length as OOS quarter,
    closest in time, allowing apples-to-apples IS/OOS degradation comparison.
    """
    oos_records = []
    is_records  = []
    n = len(df)

    for step in range(N_STEPS):
        train_end  = TRAIN_MIN_DAYS + step * STEP_DAYS
        test_start = train_end
        test_end   = min(test_start + STEP_DAYS, n)

        if test_end - test_start < 5:
            break

        train_df = df.iloc[:train_end]
        test_df  = df.iloc[test_start:test_end]
        is_df    = df.iloc[max(0, train_end - STEP_DAYS):train_end]  # last quarter of IS

        if len(train_df) < 100:
            continue

        clf = HistGradientBoostingClassifier(**HGBT_PARAMS)
        clf.fit(train_df[FEATURES], train_df["label"])
        classes = list(clf.classes_)

        def _records_for(window_df, store):
            preds  = clf.predict(window_df[FEATURES])
            probas = clf.predict_proba(window_df[FEATURES])
            for i, (idx, row) in enumerate(window_df.iterrows()):
                pred = preds[i]
                conf = float(probas[i][classes.index(pred)]) if pred in classes else 0.0
                store.append({
                    "sym":        sym,
                    "date":       idx,
                    "step":       step + 1,
                    "predicted":  pred,
                    "confidence": conf,
                    "actual":     row["label"],
                    "fwd_ret":    row.get("fwd_ret"),
                })

        _records_for(test_df, oos_records)
        if len(is_df) >= 5:
            _records_for(is_df, is_records)

    return oos_records, is_records


_active_universe = PREMIUM_UNIVERSE if _args.filtered else UNIVERSE
_universe_label  = f"PREMIUM ({len(_active_universe)} stocks)" if _args.filtered else f"FULL ({len(_active_universe)} stocks)"

all_oos = []
all_is  = []

for sym in _active_universe:
    try:
        raw = fetch_raw(sym)
        if raw.empty:
            print(f"  {sym}: no data")
            continue

        df = build_features(raw)
        df = build_labels(df)
        # attach actual forward return value for avg-return analysis
        df["fwd_ret"] = raw["Close"].shift(-FORWARD_DAYS) / raw["Close"] - 1
        df = df.dropna(subset=FEATURES + ["label"])

        if len(df) < TRAIN_MIN_DAYS + STEP_DAYS:
            print(f"  {sym}: only {len(df)} rows — skip")
            continue

        oos_recs, is_recs = walk_forward(sym, df)
        all_oos.extend(oos_recs)
        all_is.extend(is_recs)
        buy_recs = [r for r in oos_recs if r["predicted"] == "BUY"]
        print(f"  {sym}: {len(oos_recs)} OOS days  /  {len(buy_recs)} BUY signals")
        time.sleep(0.3)

    except Exception as exc:
        print(f"  {sym}: ERROR — {exc}")


# ── Analysis ─────────────────────────────────────────────────────────────────
oos = pd.DataFrame(all_oos)
oos["correct"] = oos["predicted"] == oos["actual"]

is_ = pd.DataFrame(all_is)
is_["correct"] = is_["predicted"] == is_["actual"]

print(f"\n\n{'═'*64}")
print(f"  WALK-FORWARD RESULTS  [{_universe_label}]  {N_STEPS} OOS quarters each")
print(f"{'═'*64}")
print(f"  Total OOS predictions: {len(oos)}")

print(f"\n  {'Signal':<6} {'N':>5} {'Precision':>10} {'Avg Fwd Ret':>12} {'Hit Rate':>10}")
print(f"  {'-'*48}")
for sig in ["BUY", "HOLD", "SELL"]:
    sub = oos[oos["predicted"] == sig]
    if sub.empty:
        continue
    prec    = sub["correct"].mean() * 100
    avg_ret = sub["fwd_ret"].dropna().mean() * 100
    hit     = (sub["fwd_ret"].dropna() >= THRESHOLD).mean() * 100
    print(f"  {sig:<6} {len(sub):>5} {prec:>9.1f}% {avg_ret:>+11.2f}% {hit:>9.1f}%")

# High-confidence BUY only
for thresh in [0.60, 0.65, 0.70, 0.75]:
    hc = oos[(oos["predicted"] == "BUY") & (oos["confidence"] >= thresh)]
    if hc.empty:
        continue
    prec    = hc["correct"].mean() * 100
    avg_ret = hc["fwd_ret"].dropna().mean() * 100
    hit     = (hc["fwd_ret"].dropna() >= THRESHOLD).mean() * 100
    print(f"  BUY≥{int(thresh*100)}%  {len(hc):>5} {prec:>9.1f}% {avg_ret:>+11.2f}% {hit:>9.1f}%")

# Per-step OOS drift
print(f"\n  OOS Quarter breakdown (BUY signals only):")
print(f"  {'Quarter':<10} {'N BUY':>7} {'Precision':>10} {'Avg Ret':>9}")
print(f"  {'-'*40}")
oos_buy = oos[oos["predicted"] == "BUY"]
for step in sorted(oos["step"].unique()):
    sub = oos_buy[oos_buy["step"] == step]
    if sub.empty:
        continue
    prec    = sub["correct"].mean() * 100
    avg_ret = sub["fwd_ret"].dropna().mean() * 100
    print(f"  Q{step:<9} {len(sub):>7} {prec:>9.1f}% {avg_ret:>+8.2f}%")

# ── IS vs OOS degradation ─────────────────────────────────────────────────────
# IS window = last STEP_DAYS of each training set (same length as OOS quarter).
# Deg% = (OOS - IS) / IS * 100. Negative = degradation.
# Rule of thumb: Deg% > -30% → ✓ stable  |  -30% to -50% → ⚠  |  < -50% → ✗ suspect overfitting/regime shift
print(f"\n  IS vs OOS Degradation (BUY≥{int(CONF_THRESHOLD*100)}%  |  IS = last quarter of training set):")
print(f"  {'':10} {'IS Prec':>8} {'OOS Prec':>9} {'PrecDeg':>8}  {'IS Ret':>7} {'OOS Ret':>8} {'RetDeg':>7}")
print(f"  {'-'*65}")

is_buy = is_[(is_["predicted"] == "BUY") & (is_["confidence"] >= CONF_THRESHOLD)]
hc_oos = oos[(oos["predicted"] == "BUY") & (oos["confidence"] >= CONF_THRESHOLD)]
hc_oos = hc_oos.copy()
hc_oos["correct"] = hc_oos["predicted"] == hc_oos["actual"]

def _deg(oos_val, is_val):
    return (oos_val - is_val) / abs(is_val) * 100 if is_val != 0 else 0.0

def _flag(deg):
    return "✓" if deg >= -30 else ("⚠" if deg >= -50 else "✗")

rows = []
for step in sorted(oos["step"].unique()):
    o = hc_oos[hc_oos["step"] == step]
    s = is_buy[is_buy["step"] == step]
    if o.empty or s.empty:
        continue
    o_prec = o["correct"].mean() * 100
    s_prec = s["correct"].mean() * 100
    o_ret  = o["fwd_ret"].dropna().mean() * 100
    s_ret  = s["fwd_ret"].dropna().mean() * 100
    pd_    = _deg(o_prec, s_prec)
    rd_    = _deg(o_ret,  s_ret)
    rows.append((step, s_prec, o_prec, pd_, s_ret, o_ret, rd_))
    print(f"  Q{step:<9} {s_prec:>7.1f}% {o_prec:>8.1f}% {pd_:>+7.1f}%  {s_ret:>+6.2f}% {o_ret:>+7.2f}% {rd_:>+6.1f}%  {_flag(pd_)}")

if rows:
    print(f"  {'-'*65}")
    o_prec = hc_oos["correct"].mean() * 100
    s_prec = is_buy["correct"].mean() * 100
    o_ret  = hc_oos["fwd_ret"].dropna().mean() * 100
    s_ret  = is_buy["fwd_ret"].dropna().mean() * 100
    pd_    = _deg(o_prec, s_prec)
    rd_    = _deg(o_ret,  s_ret)
    print(f"  {'OVERALL':<10} {s_prec:>7.1f}% {o_prec:>8.1f}% {pd_:>+7.1f}%  {s_ret:>+6.2f}% {o_ret:>+7.2f}% {rd_:>+6.1f}%  {_flag(pd_)}")

# ── OOS Consistency ──────────────────────────────────────────────────────────
# SD of per-quarter OOS precision (population SD, ddof=0).
# Measures internal stability: how much does the model's edge fluctuate across
# market regimes? Low SD = regime-robust. High SD = regime-sensitive.
# Thresholds (precision points): <5pp STABLE | 5-10pp MODERATE | >10pp UNSTABLE
print(f"\n  OOS Consistency (BUY≥{int(CONF_THRESHOLD*100)}%):")
q_precs = []
q_rets  = []
for step in sorted(oos["step"].unique()):
    sub = hc_oos[hc_oos["step"] == step]
    if len(sub) >= 5:
        q_precs.append(sub["correct"].mean() * 100)
        q_rets.append(sub["fwd_ret"].dropna().mean() * 100)

if len(q_precs) >= 2:
    prec_s  = pd.Series(q_precs)
    ret_s   = pd.Series(q_rets)
    sd_prec = prec_s.std(ddof=0)
    sd_ret  = ret_s.std(ddof=0)
    cv_prec = sd_prec / prec_s.mean() * 100
    stability = "STABLE" if sd_prec < 5 else ("MODERATE" if sd_prec < 10 else "UNSTABLE")
    vals_str  = "  ".join(f"Q{i+1}:{p:.1f}%" for i, p in enumerate(q_precs))
    print(f"  {vals_str}")
    print(f"  Mean OOS Prec : {prec_s.mean():.1f}%")
    print(f"  SD (Prec)     : {sd_prec:.1f}pp   ← regime sensitivity (lower = better)")
    print(f"  CV (Prec)     : {cv_prec:.1f}%    ← % of mean explained by variance")
    print(f"  SD (Avg Ret)  : {sd_ret:.2f}pp")
    print(f"  Verdict       : {stability}  (thresholds: <5pp=STABLE, 5-10pp=MODERATE, >10pp=UNSTABLE)")

# ── Per-symbol regime resilience ─────────────────────────────────────────────
# For each stock: compute OOS precision per quarter, then mean / worst-quarter / SD.
# "Regime-resilient" = worst quarter still >= RESILIENT_FLOOR.
# These stocks maintain edge even when the broader model degrades (like Q3 2026).
RESILIENT_FLOOR = 45.0   # min acceptable precision in worst quarter (pp)
print(f"\n  Per-symbol OOS by quarter (BUY≥{int(CONF_THRESHOLD*100)}%, min {int(RESILIENT_FLOOR)}% in worst quarter):")
steps = sorted(oos["step"].unique())
hdr   = f"  {'Sym':<6} {'Mean':>6} {'SD':>5} {'Worst':>6}  " + "  ".join(f"Q{s}" for s in steps)
print(hdr)
print(f"  {'-'*60}")

sym_resilience = []
for sym in sorted(oos["sym"].unique()):
    sym_q = []
    for step in steps:
        sub = hc_oos[(hc_oos["sym"] == sym) & (hc_oos["step"] == step)]
        sym_q.append(sub["correct"].mean() * 100 if len(sub) >= 3 else float("nan"))
    valid = [p for p in sym_q if not pd.isna(p)]
    if len(valid) < 2:
        continue
    s   = pd.Series(valid)
    mean_p, sd_p, min_p = s.mean(), s.std(ddof=0), s.min()
    sym_resilience.append((sym, mean_p, sd_p, min_p, sym_q))

# Sort: resilient first (by worst quarter desc), then by mean
resilient = [(s, m, sd, mn, q) for s, m, sd, mn, q in sym_resilience if mn >= RESILIENT_FLOOR]
others    = [(s, m, sd, mn, q) for s, m, sd, mn, q in sym_resilience if mn < RESILIENT_FLOOR]
resilient.sort(key=lambda x: (-x[3], -x[1]))
others.sort(key=lambda x: -x[1])

def _sym_row(sym, mean_p, sd_p, min_p, sym_q, star):
    q_str = "  ".join(f"{p:>4.0f}%" if not pd.isna(p) else "   —" for p in sym_q)
    return f"  {sym:<6} {mean_p:>5.1f}% {sd_p:>4.1f}pp {min_p:>5.1f}%  {q_str}  {star}"

if resilient:
    print(f"  ★ Regime-resilient (worst quarter ≥ {int(RESILIENT_FLOOR)}%):")
    for row in resilient:
        print(_sym_row(*row, "★"))
if others:
    print(f"  ○ Others:")
    for row in others:
        print(_sym_row(*row, ""))

# Top symbols by BUY precision (summary, unchanged)
print(f"\n  Per-symbol BUY performance (OOS, all quarters pooled):")
print(f"  {'Sym':<6} {'N BUY':>6} {'Prec':>7} {'Avg Ret':>9}")
print(f"  {'-'*32}")
sym_stats = []
for sym, grp in oos_buy.groupby("sym"):
    if len(grp) < 3:
        continue
    sym_stats.append((sym, len(grp), grp["correct"].mean()*100, grp["fwd_ret"].dropna().mean()*100))
sym_stats.sort(key=lambda x: -x[2])
for sym, n, prec, ret in sym_stats:
    print(f"  {sym:<6} {n:>6} {prec:>6.1f}% {ret:>+8.2f}%")
