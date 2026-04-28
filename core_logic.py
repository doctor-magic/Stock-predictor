import os
import json
import urllib.request
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from cachetools import cached, TTLCache
import requests
from io import StringIO
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
cache = TTLCache(maxsize=100, ttl=900)
_options_cache: TTLCache = TTLCache(maxsize=50, ttl=900)  # 15-min TTL for option chains
_macro_cache: TTLCache = TTLCache(maxsize=1, ttl=3600)    # 1h TTL for macro timeseries

PERIOD = "5y"
FORWARD_DAYS = 10
THRESHOLD = 0.03
CONFIDENCE_THRESHOLD  = 0.65
MIN_PRECISION_BUY     = 0.48  # BUY precision harder to achieve — lower floor
MIN_PRECISION_SELL    = 0.52  # SELL precision easier in volatile markets
MIN_PRECISION         = MIN_PRECISION_SELL  # default used in backtest imports

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",              # 0-3  trend
    "rsi", "macd_gap", "bb_pos",                         # 4-6  momentum
    "vol_ratio", "ret_3d_atr", "ret_5d_atr", "ret_10d_atr",  # 7-10 ATR-normalized returns+volume
    "sma200_dist",                                        # 11   long-term trend
    "low52w_dist",                                        # 12   distance from 52-week low (bounce detector)
    "atr_pct",                                            # 13   volatility regime
    "vix", "dgs10", "t10y2y",                            # 14-16 macro regime
    "rel_strength_spy",                                   # 17   relative strength to SPY
    "cmf",                                                # 18   Chaikin Money Flow
    "rel_strength_sector",                                # 19   relative strength to own sector ETF
]

INTERACTION_GROUPS = [
    [0, 1, 2, 3, 11, 12, 17, 19],       # trend + distance + relative strengths (SPY + sector)
    [4, 5, 6, 18],                       # momentum + cmf
    [7, 8, 9, 10, 13, 18],              # returns+volume+atr + cmf
    [4, 5, 6, 14, 15, 16],              # momentum × macro
    [8, 9, 10, 13, 14, 15, 16, 17, 19], # returns_atr × macro + relative strengths
    [4, 6, 12],                          # RSI + bb_pos + low52w_dist (oversold bounce detector)
    [4, 17, 19],                         # momentum + sector vs market divergence
]

OPTION_FEATURES = {"pc_ratio", "iv_skew", "volume_shock"}

SECTOR_ETF_MAP = {
    "Technology": "XLK", "Financial Services": "XLF",
    "Energy": "XLE", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Healthcare": "XLV",
    "Industrials": "XLI", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC",
}
_sector_cache: dict = {}  # ticker → macro column name, e.g. "sect_XLK"

PRESET_STOCKS = {
    "us": {
        "Apple": "AAPL",
        "NVIDIA": "NVDA",
        "Tesla": "TSLA",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Meta": "META",
        "S&P 500 ETF": "SPY",
    },
    "tase": {
        "בנק הפועלים": "POLI.TA", "בנק לאומי": "LUMI.TA", "בנק מזרחי": "MZTF.TA",
        "בנק דיסקונט": "DSCT.TA", "הבנק הבינלאומי": "FTIN.TA",
        "הראל": "HARL.TA", "הפניקס": "PHOE.TA", "כלל ביטוח": "CALI.TA",
        "מגדל": "MGDL.TA", "מנורה מבטחים": "MNRT.TA", "צ'ק פוינט": "CHKP.TA",
        "נייס": "NICE.TA", "סייברארק": "CYBR.TA", "סאפיינס": "SPNS.TA",
        "טאואר": "TSEM.TA", "אלביט": "ESLT.TA", "טבע": "TEVA.TA",
        "כיל": "ICL.TA", "אזריאלי": "AZRG.TA", "אלוני חץ": "ALHE.TA",
        "מליסרון": "MLSR.TA", "ביג": "BIG.TA", "שיכון ובינוי": "SKBN.TA",
        "בזק": "BEZQ.TA", "פרטנר": "PTNR.TA", "סלקום": "SCEL.TA",
        "דלק קבוצה": "DLEKG.TA", "פז נפט": "PZOL.TA", "בתי זיקוק": "ORL.TA",
        "אנליט": "ENLT.TA", "שטראוס": "STRS.TA", "שופרסל": "SAE.TA",
        "רמי לוי": "RMLI.TA", "מיבנה": "MVNE.TA", "אפריקה ישראל": "AFPR.TA",
    }
}

@cached(cache)
def load_sp500():
    try:
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
        df   = pd.read_html(StringIO(html))[0]
        return dict(zip(df["Security"], df["Symbol"].str.replace(".", "-", regex=False)))
    except Exception as e:
        print("SP500 Load Error:", e)
        return PRESET_STOCKS["us"]

@cached(cache)
def load_nasdaq100():
    try:
        url  = "https://en.wikipedia.org/wiki/Nasdaq-100"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
        for t in pd.read_html(StringIO(html)):
            if "Ticker" in t.columns or "Symbol" in t.columns:
                col      = "Ticker" if "Ticker" in t.columns else "Symbol"
                name_col = "Company" if "Company" in t.columns else t.columns[0]
                return dict(zip(t[name_col], t[col].str.replace(".", "-", regex=False)))
    except Exception as e:
        print("NASDAQ Load Error:", e)
    return PRESET_STOCKS["us"]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def fetch_options_features(ticker: str, spot: float) -> dict:
    """Fetch ATM option metrics with 15-min cache. spot comes from existing OHLCV data."""
    defaults = {"pc_ratio": None, "iv_skew": None, "volume_shock": None}
    if ticker in _options_cache:
        return _options_cache[ticker]
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return defaults

        # Nearest expiry >= 14 days out to avoid gamma/pin distortion
        today = datetime.date.today()
        target = next(
            (e for e in expirations if (datetime.date.fromisoformat(e) - today).days >= 14),
            expirations[0]
        )
        chain = tk.option_chain(target)
        calls, puts = chain.calls.copy(), chain.puts.copy()

        # PC Ratio: weighted over 3 strikes closest to ATM
        n = 3
        call_atm = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[:n]]
        put_atm  = puts.iloc[(puts["strike"]  - spot).abs().argsort().iloc[:n]]
        call_oi  = call_atm["openInterest"].fillna(0).sum()
        put_oi   = put_atm["openInterest"].fillna(0).sum()
        if call_oi < 100 or put_oi < 100:  # illiquid market — data not trustworthy
            return defaults
        pc_ratio = put_oi / call_oi

        # IV Skew: 5% OTM strikes, bid>0 validates the IV isn't garbage
        liquid_puts  = puts[(puts["bid"]  > 0) & (puts["ask"]  > 0)]
        liquid_calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
        if liquid_puts.empty or liquid_calls.empty:
            iv_skew = None
        else:
            p_row  = liquid_puts.iloc[(liquid_puts["strike"]   - spot * 0.95).abs().argsort().iloc[0]]
            c_row  = liquid_calls.iloc[(liquid_calls["strike"] - spot * 1.05).abs().argsort().iloc[0]]
            p_iv, c_iv = p_row["impliedVolatility"], c_row["impliedVolatility"]
            iv_skew = (round(float(p_iv) - float(c_iv), 4)
                       if pd.notna(p_iv) and pd.notna(c_iv) else None)

        # Volume Shock: option turnover ratio vs open interest
        total_vol = float(calls["volume"].fillna(0).sum() + puts["volume"].fillna(0).sum())
        total_oi  = float(calls["openInterest"].fillna(0).sum() + puts["openInterest"].fillna(0).sum())
        vol_shock = round(total_vol / total_oi, 4) if total_oi > 0 else None

        result = {
            "pc_ratio":     round(float(pc_ratio), 4) if pc_ratio is not None else None,
            "iv_skew":      iv_skew,
            "volume_shock": vol_shock,
        }
    except Exception as e:
        print(f"Options fetch error [{ticker}]: {e}")
        result = defaults

    _options_cache[ticker] = result
    return result

def fetch_macro_timeseries() -> pd.DataFrame:
    """Return a tz-naive daily DataFrame with columns: vix, dgs10, t10y2y, spy_close.
    Cached for 1 hour — shared across all per-stock model fits in a scan."""
    if "macro" in _macro_cache:
        return _macro_cache["macro"]

    frames: list[pd.DataFrame] = []

    # VIX via yfinance
    try:
        raw = yf.download("^VIX", period=PERIOD, progress=False, auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        vix = raw[["Close"]].rename(columns={"Close": "vix"})
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        frames.append(vix)
    except Exception as e:
        print(f"VIX fetch error: {e}")

    # SPY via yfinance
    try:
        raw_spy = yf.download("SPY", period=PERIOD, progress=False, auto_adjust=False)
        if isinstance(raw_spy.columns, pd.MultiIndex):
            raw_spy.columns = raw_spy.columns.get_level_values(0)
        spy_close = raw_spy[["Close"]].rename(columns={"Close": "spy_close"})
        spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
        frames.append(spy_close)
    except Exception as e:
        print(f"SPY fetch error: {e}")

    # Sector ETFs (XLK, XLF, XLE, etc.) — one batch call, cached with macro
    try:
        etf_tickers = list(SECTOR_ETF_MAP.values())
        raw_sect = yf.download(etf_tickers, period=PERIOD, progress=False, auto_adjust=False)
        sect_close = raw_sect["Close"].copy() if isinstance(raw_sect.columns, pd.MultiIndex) else raw_sect[["Close"]]
        sect_close = sect_close.rename(columns={etf: f"sect_{etf}" for etf in etf_tickers})
        sect_close.index = pd.to_datetime(sect_close.index).tz_localize(None)
        frames.append(sect_close)
    except Exception as e:
        print(f"Sector ETF fetch error: {e}")

    # DGS10 + T10Y2Y via FRED
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        for series_id, col in [("DGS10", "dgs10"), ("T10Y2Y", "t10y2y")]:
            try:
                url = (
                    "https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&limit=2000&sort_order=desc"
                    f"&api_key={fred_key}&file_type=json"
                )
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = json.loads(r.read())
                obs = pd.DataFrame(data["observations"])
                obs = obs[obs["value"] != "."].copy()
                obs.index = pd.to_datetime(obs["date"])
                obs[col] = pd.to_numeric(obs["value"], errors="coerce")
                frames.append(obs[[col]])
            except Exception as e:
                print(f"FRED {series_id} error: {e}")

    if not frames:
        return pd.DataFrame(columns=["vix", "dgs10", "t10y2y", "spy_close"])

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.join(f, how="outer")
    macro = macro.sort_index().ffill().bfill()

    sect_cols = [f"sect_{etf}" for etf in SECTOR_ETF_MAP.values()]
    for col in ["vix", "dgs10", "t10y2y", "spy_close"] + sect_cols:
        if col not in macro.columns:
            macro[col] = np.nan

    _macro_cache["macro"] = macro
    return macro


def get_sector_etf(ticker: str) -> str:
    """Return macro column name for ticker's sector ETF. Falls back to spy_close."""
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


def build_features(df: pd.DataFrame, sector_col: str = "spy_close") -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_cross"] = (df["ema9"] > df["ema21"]).astype(int)
    sma200 = df["Close"].rolling(200).mean()
    df["sma200_dist"] = (df["Close"] - sma200) / sma200
    df["rsi"] = compute_rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_gap"] = macd - macd.ewm(span=9, adjust=False).mean()
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    lower = sma20 - 2 * std20
    upper = sma20 + 2 * std20
    bb_range = (upper - lower).replace(0, np.nan)
    df["bb_pos"] = (df["Close"] - lower) / bb_range
    vol_mean = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["vol_ratio"] = df["Volume"] / vol_mean

    # Chaikin Money Flow (CMF) 20-period
    high_low_diff = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_diff
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    df["cmf"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)

    # ATR(14): True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_pct"] = atr14 / df["Close"].replace(0, np.nan)

    # 52-week low distance: how far above the annual low the stock is sitting
    low52 = df["Close"].rolling(252).min()
    df["low52w_dist"] = (df["Close"] - low52) / low52.replace(0, np.nan)

    # ATR-normalized returns: express moves in units of daily ATR
    ret_3d  = df["Close"].pct_change(3)
    ret_5d  = df["Close"].pct_change(5)
    ret_10d = df["Close"].pct_change(10)
    atr_denom = df["atr_pct"].replace(0, np.nan)
    df["ret_3d_atr"]  = ret_3d  / atr_denom
    df["ret_5d_atr"]  = ret_5d  / atr_denom
    df["ret_10d_atr"] = ret_10d / atr_denom

    # Macro regime features (VIX, 10Y yield, yield curve) + SPY for relative strength
    try:
        macro = fetch_macro_timeseries()
        idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df.index = idx
        macro_cols = [c for c in macro.columns if c in ["vix", "dgs10", "t10y2y", "spy_close"] or c.startswith("sect_")]
        df = df.join(macro[macro_cols], how="left")
        df[macro_cols] = df[macro_cols].ffill()
        
        df["spy_ret_5d"] = df["spy_close"].pct_change(5)
        df["rel_strength_spy"] = ret_5d - df["spy_ret_5d"]
        sect_close = df[sector_col] if sector_col in df.columns else df["spy_close"]
        df["rel_strength_sector"] = ret_5d - sect_close.pct_change(5)
    except Exception as e:
        print(f"Macro join error: {e}")
        df["vix"] = np.nan
        df["dgs10"] = np.nan
        df["t10y2y"] = np.nan
        df["rel_strength_spy"] = np.nan
        df["rel_strength_sector"] = np.nan

    return df

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fwd_ret = df["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1
    df["label"] = "HOLD"
    df.loc[fwd_ret >= THRESHOLD, "label"] = "BUY"
    df.loc[fwd_ret <= -THRESHOLD, "label"] = "SELL"
    df.loc[df.index[-FORWARD_DAYS:], "label"] = np.nan
    return df

@cached(cache)
def fetch_stock_data(ticker: str, period=PERIOD) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close", "Volume"]].copy()

def train_and_evaluate(df: pd.DataFrame, light_mode=False):
    from sklearn.metrics import classification_report
    clean = df.dropna(subset=FEATURES + ["label"])
    split = int(len(clean) * 0.8)
    train, test = clean.iloc[:split], clean.iloc[split:]
    X_train, y_train = train[FEATURES], train["label"]
    X_test, y_test = test[FEATURES], test["label"]

    clf = HistGradientBoostingClassifier(
        max_iter=100 if light_mode else 300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.1,
        learning_rate=0.01,
        interaction_cst=INTERACTION_GROUPS,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    MIN_SUPPORT = 5  # fewer predictions → precision is noise
    def _prec(label):
        info = rpt.get(label, {})
        if info.get("support", 0) < MIN_SUPPORT:
            return 0.0
        return min(info.get("precision", 0.0), 0.90)  # cap at 90% — 100% = tiny sample

    return clf, _prec("BUY"), _prec("SELL")

def get_prediction(ticker: str, light_mode=False):
    df_raw = fetch_stock_data(ticker)
    if df_raw.empty:
        return None
    sector_col = get_sector_etf(ticker)
    df = build_features(df_raw, sector_col=sector_col)

    avg_atr = df["atr_pct"].dropna().tail(30).mean()
    current_vix = df["vix"].dropna().iloc[-1] if "vix" in df.columns and not df["vix"].dropna().empty else 20
    chaos_threshold = 0.05 if current_vix < 15 else (0.07 if current_vix < 25 else 0.10)
    if avg_atr > chaos_threshold:
        return {
            "symbol": ticker, "signal": "EXCLUDED", "confidence": 0.0,
            "options_filtered": False, "precision_score": 0.0,
            "last_price": float(df_raw["Close"].iloc[-1]),
            "last_date": str(df_raw.index[-1].date()),
            "rows_trained": 0, "importance": {}, "options_context": {},
            "excluded_reason": f"avg_atr_30d > {chaos_threshold:.0%} at VIX={current_vix:.0f} (high-volatility / meme stock)",
        }

    df = build_labels(df)
    clf, buy_prec, sell_prec = train_and_evaluate(df, light_mode=light_mode)
    latest = df[FEATURES].ffill().iloc[[-1]]

    proba = clf.predict_proba(latest)[0]
    classes = clf.classes_
    pred_idx = np.argmax(proba)
    raw_signal = classes[pred_idx]
    confidence = proba[pred_idx]

    final_signal = raw_signal if confidence >= CONFIDENCE_THRESHOLD else "HOLD"
    
    # Hard filter: prevent 'Falling Knife' buys if institutional money is flowing out
    if final_signal == "BUY" and df["cmf"].iloc[-1] < 0:
        final_signal = "HOLD"

    precision = sell_prec if final_signal == "SELL" else buy_prec

    # Fetch options before filter so they can adjust confidence
    spot = float(df["Close"].iloc[-1])
    options_ctx = fetch_options_features(ticker, spot)

    # Options post-prediction filter: bearish options reduce BUY confidence; bullish reduce SELL
    options_filtered = False
    if final_signal != "HOLD":
        pc = options_ctx.get("pc_ratio")
        skew = options_ctx.get("iv_skew")
        adj = confidence
        if final_signal == "BUY":
            if pc is not None and pc > 1.2:    # more puts than calls — hedging pressure
                adj *= 0.85
            if skew is not None and skew > 0.05:  # fear premium on downside
                adj *= 0.90
        elif final_signal == "SELL":
            if pc is not None and pc < 0.8:    # call-heavy — market positioning bullish
                adj *= 0.85
            if skew is not None and skew < -0.02:  # call IV > put IV — bullish skew
                adj *= 0.90
        if adj < confidence:
            options_filtered = True
            confidence = adj
            if confidence < CONFIDENCE_THRESHOLD:
                final_signal = "HOLD"

    try:
        importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).to_dict()
    except AttributeError:
        # HGBDT + CalibratedClassifierCV don't expose feature_importances_ — use permutation importance
        try:
            from sklearn.inspection import permutation_importance as perm_imp
            clean = df.dropna(subset=FEATURES + ["label"])
            eval_size = min(200, len(clean) // 5)
            X_eval = clean[FEATURES].iloc[-eval_size:]
            y_eval = clean["label"].iloc[-eval_size:]
            perm = perm_imp(clf, X_eval, y_eval, n_repeats=3, scoring="neg_log_loss", random_state=42)
            importances = {
                FEATURES[i]: float(perm.importances_mean[i])
                for i in range(len(FEATURES))
                if perm.importances_mean[i] > 0
            }
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            importances = {}

    return {
        "symbol": ticker,
        "signal": final_signal,
        "confidence": float(confidence),
        "options_filtered": options_filtered,
        "precision_score": float(precision),
        "last_price": spot,
        "last_date": str(df.index[-1].date()),
        "rows_trained": len(df.dropna(subset=FEATURES + ["label"])),
        "importance": {k: float(v) for k, v in list(importances.items())[:5]},
        "options_context": options_ctx,
    }

def _apply_options_filter(result: dict):
    """Fetch options for a scan result and apply the same confidence adjustment
    as get_prediction(). Returns None if the signal downgrades to HOLD."""
    signal = result["signal"]
    if signal == "HOLD":
        return result
    try:
        opts = fetch_options_features(result["symbol"], result["last_price"])
        pc   = opts.get("pc_ratio")
        skew = opts.get("iv_skew")
        conf = result["confidence"]
        if signal == "BUY":
            if pc   is not None and pc   > 1.2:  conf *= 0.85
            if skew is not None and skew > 0.05: conf *= 0.90
        elif signal == "SELL":
            if pc   is not None and pc   < 0.8:   conf *= 0.85
            if skew is not None and skew < -0.02:  conf *= 0.90
        if conf < result["confidence"]:
            if conf < CONFIDENCE_THRESHOLD:
                return None  # downgraded to HOLD — drop from scanner results
            result = {**result, "confidence": conf, "options_filtered": True}
    except Exception as e:
        print(f"Options filter error [{result['symbol']}]: {e}")
    return result


def _train_single(name, sym, raw_data, multi):
    """Train a single model — designed to run in a thread."""
    try:
        raw = raw_data[sym].copy() if multi else raw_data.copy()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(how="all")[["Open", "High", "Low", "Close", "Volume"]]
        if raw.empty:
            return None

        sector_col = get_sector_etf(sym)
        df = build_features(raw, sector_col=sector_col)

        # Chaos filter: skip high-volatility meme/speculative stocks.
        # avg ATR >5% over last 30 sessions means the stock is too noisy for
        # a 10-day swing model — signals would be random noise regardless of precision.
        avg_atr = df["atr_pct"].dropna().tail(30).mean()
        current_vix = df["vix"].dropna().iloc[-1] if "vix" in df.columns and not df["vix"].dropna().empty else 20
        chaos_threshold = 0.05 if current_vix < 15 else (0.07 if current_vix < 25 else 0.10)
        if avg_atr > chaos_threshold:
            return None

        df = build_labels(df)
        clf, buy_prec, sell_prec = train_and_evaluate(df, light_mode=True)

        latest = df[FEATURES].ffill().iloc[[-1]]
        proba = clf.predict_proba(latest)[0]
        confident_signal = clf.classes_[np.argmax(proba)]
        confidence = np.max(proba)
        final_signal = confident_signal if confidence >= CONFIDENCE_THRESHOLD else "HOLD"

        # Hard filter: prevent 'Falling Knife' buys if institutional money is flowing out
        if final_signal == "BUY" and df["cmf"].iloc[-1] < 0:
            final_signal = "HOLD"

        precision = sell_prec if final_signal == "SELL" else buy_prec
        min_prec = MIN_PRECISION_SELL if final_signal == "SELL" else MIN_PRECISION_BUY
        if precision < min_prec and final_signal != "HOLD":
            return None
        return {
            "symbol": sym,
            "symbol_name": name,
            "signal": final_signal,
            "confidence": float(confidence),
            "precision": float(precision),
            "last_price": float(raw["Close"].iloc[-1])
        }
    except Exception as e:
        print(f"Failed scanning {sym}: {e}")
        return None


def run_market_scan(market_id: str, progress_callback=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if market_id == "sp500":
        stocks = load_sp500()
    elif market_id == "nasdaq100":
        stocks = load_nasdaq100()
    else:
        stocks = PRESET_STOCKS.get(market_id, PRESET_STOCKS["us"])

    results = []
    items = list(stocks.items())
    all_tickers = [sym for _, sym in items]
    total = len(all_tickers)
    BATCH = 50

    # Phase 1: Batch download
    all_data = {}
    for batch_start in range(0, total, BATCH):
        batch_end = min(batch_start + BATCH, total)
        batch_syms = all_tickers[batch_start:batch_end]
        if progress_callback:
            progress_callback(batch_start, total,
                              f"Downloading batch {batch_start // BATCH + 1} "
                              f"({batch_start + 1}-{batch_end} of {total})...")

        chunk = yf.download(batch_syms, period=PERIOD, progress=False,
                            group_by="ticker", auto_adjust=False)
        for sym in batch_syms:
            try:
                if len(batch_syms) > 1:
                    all_data[sym] = chunk[sym].copy()
                else:
                    all_data[sym] = chunk.copy()
            except Exception:
                pass

    # Phase 2: Parallel model training
    if progress_callback:
        progress_callback(0, total, "Training models...")

    done_count = 0

    def on_done(sym):
        nonlocal done_count
        done_count += 1
        if progress_callback:
            progress_callback(done_count, total, f"Trained {done_count}/{total}: {sym}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for name, sym in items:
            if sym not in all_data:
                on_done(sym)
                continue
            single_df = all_data[sym]
            fut = pool.submit(_train_single, name, sym, {sym: single_df}, True)
            futures[fut] = sym

        for fut in as_completed(futures):
            sym = futures[fut]
            on_done(sym)
            res = fut.result()
            if res:
                results.append(res)

    # Apply options filter only to the small set of BUY/SELL results
    if results:
        with ThreadPoolExecutor(max_workers=min(len(results), 8)) as pool:
            filtered = list(pool.map(_apply_options_filter, results))
        results = [r for r in filtered if r is not None]

    return sorted(results, key=lambda x: x["confidence"], reverse=True)
