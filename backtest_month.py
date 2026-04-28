"""
Backtest — updated model with ATR normalization, low52w_dist, asymmetric precision, VIX-adaptive chaos filter.
Windows:
  1. May 2025 - July 2025
  2. Jan 2026 - Apr 2026 (includes April Black Swan)
"""
import warnings
warnings.filterwarnings("ignore")

import os, json, urllib.request as _req
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# ── Config (matches updated core_logic.py) ───────────────────────────────────
PERIODS = [
    (date(2025,  5, 30), date(2025,  6, 16), "May 2025 (calm bull)"),
    (date(2026,  1, 31), date(2026,  2, 14), "Jan 2026 (normal)"),
    (date(2026,  3, 26), date(2026,  4,  9), "Mar 2026 (Black Swan)"),
]

CONFIDENCE_THRESHOLD = 0.65
MIN_PRECISION_BUY    = 0.48
MIN_PRECISION_SELL   = 0.52
THRESHOLD            = 0.03
FORWARD_DAYS         = 10

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",
    "rsi", "macd_gap", "bb_pos",
    "vol_ratio", "ret_3d_atr", "ret_5d_atr", "ret_10d_atr",
    "sma200_dist",
    "low52w_dist",
    "atr_pct",
    "vix", "dgs10", "t10y2y",
    "rel_strength_spy",
    "cmf",
    "rel_strength_sector",
]

INTERACTION_GROUPS = [
    [0, 1, 2, 3, 11, 12, 17, 19],
    [4, 5, 6, 18],
    [7, 8, 9, 10, 13, 18],
    [4, 5, 6, 14, 15, 16],
    [8, 9, 10, 13, 14, 15, 16, 17, 19],
    [4, 6, 12],
    [4, 17, 19],
]

SECTOR_ETF_MAP = {
    "Technology": "XLK", "Financial Services": "XLF",
    "Energy": "XLE", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Healthcare": "XLV",
    "Industrials": "XLI", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC",
}
_sector_cache: dict = {}

def get_sector_etf(ticker: str) -> str:
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    try:
        sector = yf.Ticker(ticker).info.get("sector", "")
        etf = SECTOR_ETF_MAP.get(sector, "")
        col = f"sect_{etf}" if etf else "spy_close"
    except Exception:
        col = "spy_close"
    _sector_cache[ticker] = col
    return col

TICKERS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "META", "GOOGL", "AMZN", "SPY",
    "PLTR", "HOOD", "HWM", "EL", "ALB",
    "JPM", "GS", "BAC", "XOM", "CVX", "PFE", "JNJ", "WMT",
    "NFLX", "AMD", "INTC", "CRM", "ORCL", "UBER", "ABNB",
]

# ── FRED key ──────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
FRED_KEY = ""
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            if line.strip().startswith("FRED_API_KEY="):
                FRED_KEY = line.strip().split("=", 1)[1]

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def fetch_fred(series_id, col, end_date):
    if not FRED_KEY:
        return pd.DataFrame(columns=[col])
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&limit=2000&sort_order=desc"
           f"&api_key={FRED_KEY}&file_type=json")
    with _req.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    obs = pd.DataFrame(data["observations"])
    obs = obs[obs["value"] != "."].copy()
    obs.index = pd.to_datetime(obs["date"])
    obs[col] = pd.to_numeric(obs["value"], errors="coerce")
    return obs[[col]][obs.index <= pd.Timestamp(end_date)]

def build_features(df, macro, sector_col="spy_close"):
    df = df.copy()
    df["ema9"]        = df["Close"].ewm(span=9,  adjust=False).mean()
    df["ema21"]       = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"]       = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_cross"]   = (df["ema9"] > df["ema21"]).astype(int)
    sma200            = df["Close"].rolling(200).mean()
    df["sma200_dist"] = (df["Close"] - sma200) / sma200
    df["rsi"]         = compute_rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_gap"]    = macd - macd.ewm(span=9, adjust=False).mean()
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    bb_range = (sma20 + 2*std20 - (sma20 - 2*std20)).replace(0, np.nan)
    df["bb_pos"]    = (df["Close"] - (sma20 - 2*std20)) / bb_range
    vol_mean        = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["vol_ratio"] = df["Volume"] / vol_mean

    high_low_diff = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_diff
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    df["cmf"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)

    # ATR(14)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_pct"] = atr14 / df["Close"].replace(0, np.nan)

    low52 = df["Close"].rolling(252).min()
    df["low52w_dist"] = (df["Close"] - low52) / low52.replace(0, np.nan)

    # ATR-normalized returns
    atr_d = df["atr_pct"].replace(0, np.nan)
    ret_3d = df["Close"].pct_change(3)
    ret_5d = df["Close"].pct_change(5)
    ret_10d = df["Close"].pct_change(10)
    
    df["ret_3d_atr"]  = ret_3d  / atr_d
    df["ret_5d_atr"]  = ret_5d  / atr_d
    df["ret_10d_atr"] = ret_10d / atr_d

    # Macro join
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index = idx
    df = df.join(macro, how="left")
    macro_cols = [c for c in ["vix", "dgs10", "t10y2y", "spy_close"] + [f"sect_{e}" for e in SECTOR_ETF_MAP.values()] if c in df.columns]
    df[macro_cols] = df[macro_cols].ffill()

    df["spy_ret_5d"] = df["spy_close"].pct_change(5)
    df["rel_strength_spy"] = ret_5d - df["spy_ret_5d"]
    sect_close = df[sector_col] if sector_col in df.columns else df["spy_close"]
    df["rel_strength_sector"] = ret_5d - sect_close.pct_change(5)
    return df

def build_labels(df):
    df  = df.copy()
    fwd = df["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1
    df["label"] = "HOLD"
    df.loc[fwd >=  THRESHOLD, "label"] = "BUY"
    df.loc[fwd <= -THRESHOLD, "label"] = "SELL"
    df.loc[df.index[-FORWARD_DAYS:], "label"] = np.nan
    return df

def train_predict(df):
    clean = df.dropna(subset=FEATURES + ["label"])
    if len(clean) < 300:
        return None, None, None
    split = int(len(clean) * 0.8)
    train, test = clean.iloc[:split], clean.iloc[split:]
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=0.1, learning_rate=0.01,
        interaction_cst=INTERACTION_GROUPS,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42,
    )
    clf.fit(train[FEATURES], train["label"])
    rpt = classification_report(test["label"], clf.predict(test[FEATURES]),
                                output_dict=True, zero_division=0)

    MIN_SUPPORT = 5
    def _prec(label):
        info = rpt.get(label, {})
        if info.get("support", 0) < MIN_SUPPORT:
            return 0.0
        return min(info.get("precision", 0.0), 0.90)

    proba    = clf.predict_proba(df[FEATURES].ffill().iloc[[-1]])[0]
    pred_idx = np.argmax(proba)
    signal   = clf.classes_[pred_idx]
    conf     = float(proba[pred_idx])
    if conf < CONFIDENCE_THRESHOLD:
        signal = "HOLD"
    precision = _prec("SELL") if signal == "SELL" else _prec("BUY")
    return signal, conf, precision

# ── Run one period ────────────────────────────────────────────────────────────
def run_period(backtest_date, check_date, label):
    print(f"\n{'='*66}")
    print(f"  {label}  |  train up to {backtest_date}  |  check {check_date}")
    print(f"{'='*66}")

    start = backtest_date - timedelta(days=365*5 + 30)

    # Macro
    vix_raw = yf.download("^VIX", start=str(start), end=str(backtest_date),
                           progress=False, auto_adjust=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    macro = vix_raw[["Close"]].rename(columns={"Close": "vix"})
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    
    spy_raw = yf.download("SPY", start=str(start), end=str(backtest_date),
                           progress=False, auto_adjust=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_close = spy_raw[["Close"]].rename(columns={"Close": "spy_close"})
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
    macro = macro.join(spy_close, how="left")

    try:
        etf_tickers = list(SECTOR_ETF_MAP.values())
        raw_sect = yf.download(etf_tickers, start=str(start), end=str(backtest_date),
                               progress=False, auto_adjust=False)
        sect_close = raw_sect["Close"].copy() if isinstance(raw_sect.columns, pd.MultiIndex) else raw_sect[["Close"]]
        sect_close = sect_close.rename(columns={etf: f"sect_{etf}" for etf in etf_tickers})
        sect_close.index = pd.to_datetime(sect_close.index).tz_localize(None)
        macro = macro.join(sect_close, how="left")
    except Exception as e:
        print(f"  Sector ETF fetch error: {e}")
    try:
        macro = macro.join(fetch_fred("DGS10",  "dgs10",  backtest_date), how="left")
        macro = macro.join(fetch_fred("T10Y2Y", "t10y2y", backtest_date), how="left")
        macro = macro.ffill()
    except Exception as e:
        print(f"  FRED error: {e}")
        macro["dgs10"] = macro["t10y2y"] = np.nan

    print(f"\n{'TICKER':<8} {'SIGNAL':<6} {'CONF':>6} {'PREC':>6}  {'THEN':>8} {'AFTER':>8} {'RET%':>7}  RESULT")
    print("-" * 66)

    results = []
    for ticker in TICKERS:
        try:
            raw = yf.download(ticker, start=str(start), end=str(backtest_date),
                              progress=False, auto_adjust=False)
            if raw.empty or len(raw) < 300:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open", "High", "Low", "Close", "Volume"]]

            sector_col = get_sector_etf(ticker)
            df = build_features(raw, macro, sector_col=sector_col)

            avg_atr = df["atr_pct"].dropna().tail(30).mean()
            current_vix = df["vix"].dropna().iloc[-1] if "vix" in df.columns and not df["vix"].dropna().empty else 20
            chaos_threshold = 0.05 if current_vix < 15 else (0.07 if current_vix < 25 else 0.10)
            if avg_atr > chaos_threshold:
                continue  # chaos filter

            df = build_labels(df)
            signal, conf, prec = train_predict(df)

            min_prec = MIN_PRECISION_SELL if signal == "SELL" else MIN_PRECISION_BUY
            if signal is None or signal == "HOLD" or prec < min_prec:
                continue

            price_then = float(raw["Close"].iloc[-1])
            future = yf.download(ticker,
                                  start=str(check_date),
                                  end=str(check_date + timedelta(days=7)),
                                  progress=False, auto_adjust=False)
            if future.empty:
                continue
            if isinstance(future.columns, pd.MultiIndex):
                future.columns = future.columns.get_level_values(0)
            price_after = float(future["Close"].iloc[0])
            ret = (price_after - price_then) / price_then

            correct = (signal == "BUY"  and ret >=  THRESHOLD) or \
                      (signal == "SELL" and ret <= -THRESHOLD)
            tag = "HIT" if correct else "MISS"
            print(f"{ticker:<8} {signal:<6} {conf:>6.1%} {prec:>6.1%}  "
                  f"{price_then:>8.2f} {price_after:>8.2f} {ret*100:>+7.1f}%  {tag}")
            results.append({"ticker": ticker, "signal": signal,
                             "ret_pct": round(ret*100, 2), "correct": correct,
                             "precision": prec})
        except Exception as ex:
            print(f"{ticker:<8} ERROR: {ex}")

    print("-" * 66)
    if results:
        df_r  = pd.DataFrame(results)
        n_ok  = int(df_r["correct"].sum())
        n     = len(df_r)
        avg_p = df_r["precision"].mean()
        print(f"Signals: {n}  |  Hit rate: {n_ok}/{n} = {n_ok/n:.0%}  |  Avg model precision: {avg_p:.1%}")
        for sig in ["BUY", "SELL"]:
            sub = df_r[df_r["signal"] == sig]
            if not sub.empty:
                print(f"  {sig}: {int(sub['correct'].sum())}/{len(sub)}  avg ret {sub['ret_pct'].mean():+.1f}%")
        return n_ok, n, df_r
    else:
        print(f"No signals passed MIN_PRECISION filter (BUY={MIN_PRECISION_BUY}, SELL={MIN_PRECISION_SELL}).")
        return 0, 0, pd.DataFrame()

# ── Main ──────────────────────────────────────────────────────────────────────
all_results = []
total_ok = total_n = 0
for bdate, cdate, lbl in PERIODS:
    ok, n, df_r = run_period(bdate, cdate, lbl)
    total_ok += ok
    total_n  += n
    if not df_r.empty:
        df_r["period"] = lbl
        all_results.append(df_r)

print(f"\n{'='*66}")
print(f"COMBINED ACROSS ALL PERIODS")
if total_n:
    combined = pd.concat(all_results) if all_results else pd.DataFrame()
    print(f"Total signals: {total_n}  |  Hit rate: {total_ok}/{total_n} = {total_ok/total_n:.0%}")
    print(f"Avg model precision: {combined['precision'].mean():.1%}")

    # Recall check: did AAPL/NVDA appear at all?
    print("\nRecall check (AAPL / NVDA):")
    for t in ["AAPL", "NVDA"]:
        rows = combined[combined["ticker"] == t]
        if rows.empty:
            print(f"  {t}: no signal emitted (filtered or HOLD)")
        else:
            for _, r in rows.iterrows():
                print(f"  {t}: {r['signal']} in '{r['period']}' -> {r['ret_pct']:+.1f}%  {'HIT' if r['correct'] else 'MISS'}")
else:
    print("No signals at all.")
