import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score
from cachetools import cached, TTLCache
import requests
from io import StringIO
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
cache = TTLCache(maxsize=100, ttl=900)
_options_cache: TTLCache = TTLCache(maxsize=50, ttl=900)  # 15-min TTL for option chains

PERIOD = "5y"
FORWARD_DAYS = 10
THRESHOLD = 0.03
CONFIDENCE_THRESHOLD = 0.68
MIN_PRECISION = 0.36

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",
    "rsi", "macd_gap", "bb_pos",
    "vol_ratio", "ret_3d", "ret_5d", "ret_10d",
    "sma200_dist"
]

INTERACTION_GROUPS = [
    [0, 1, 2, 3, 11],  # trend: ema9, ema21, ema50, ema_cross, sma200_dist
    [4, 5, 6],          # momentum: rsi, macd_gap, bb_pos
    [7, 8, 9, 10],      # returns+volume: vol_ratio, ret_3d, ret_5d, ret_10d
]

OPTION_FEATURES = {"pc_ratio", "iv_skew", "volume_shock"}

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
        pc_ratio = (put_oi / call_oi) if call_oi > 0 else None

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

def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["ret_3d"] = df["Close"].pct_change(3)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
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
    buy_precision = rpt.get("BUY", {}).get("precision", 0.0)
    return clf, buy_precision

def get_prediction(ticker: str, light_mode=False):
    df_raw = fetch_stock_data(ticker)
    if df_raw.empty:
        return None
    df = build_features(df_raw)
    df = build_labels(df)
    clf, precision = train_and_evaluate(df, light_mode=light_mode)
    latest = df[FEATURES].ffill().iloc[[-1]]

    proba = clf.predict_proba(latest)[0]
    classes = clf.classes_
    pred_idx = np.argmax(proba)
    raw_signal = classes[pred_idx]
    confidence = proba[pred_idx]

    final_signal = raw_signal if confidence >= CONFIDENCE_THRESHOLD else "HOLD"
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

    # spot from OHLCV close — reliable fallback vs fast_info which fails after-hours
    spot = float(df["Close"].iloc[-1])
    options_ctx = fetch_options_features(ticker, spot)

    return {
        "symbol": ticker,
        "signal": final_signal,
        "confidence": float(confidence),
        "precision_score": float(precision),
        "last_price": spot,
        "last_date": str(df.index[-1].date()),
        "rows_trained": len(df.dropna(subset=FEATURES + ["label"])),
        "importance": {k: float(v) for k, v in list(importances.items())[:5]},
        "options_context": options_ctx,
    }

def _train_single(name, sym, raw_data, multi):
    """Train a single model — designed to run in a thread."""
    try:
        raw = raw_data[sym].copy() if multi else raw_data.copy()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(how="all")[["Open", "High", "Low", "Close", "Volume"]]
        if raw.empty:
            return None

        df = build_features(raw)
        df = build_labels(df)
        clf, precision = train_and_evaluate(df, light_mode=True)

        latest = df[FEATURES].ffill().iloc[[-1]]
        proba = clf.predict_proba(latest)[0]
        confident_signal = clf.classes_[np.argmax(proba)]
        confidence = np.max(proba)
        final_signal = confident_signal if confidence >= CONFIDENCE_THRESHOLD else "HOLD"

        if precision < MIN_PRECISION and final_signal != "HOLD":
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

    return sorted(results, key=lambda x: x["confidence"], reverse=True)
