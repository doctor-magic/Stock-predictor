import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from cachetools import cached, TTLCache
import warnings

# Ignore minor warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Cache for 15 minutes to avoid hitting Yahoo Finance too often
cache = TTLCache(maxsize=100, ttl=900)

PERIOD = "5y"
FORWARD_DAYS = 5
THRESHOLD = 0.025 # 2.5% move expected
CONFIDENCE_THRESHOLD = 0.65

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",
    "rsi", "macd_gap", "bb_pos",
    "vol_ratio", "ret_3d", "ret_5d", "ret_10d",
    "sma200_dist" # New competitive feature
]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Moving Averages
    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_cross"] = (df["ema9"] > df["ema21"]).astype(int)
    
    # Distance from Long-term SMA (Macro trend context)
    sma200 = df["Close"].rolling(200).mean()
    df["sma200_dist"] = (df["Close"] - sma200) / sma200
    
    # RSI
    df["rsi"] = compute_rsi(df["Close"])
    
    # MACD Gap
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_gap"] = macd - macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands Position
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    lower = sma20 - 2 * std20
    upper = sma20 + 2 * std20
    # Avoid div/0
    bb_range = (upper - lower).replace(0, np.nan)
    df["bb_pos"] = (df["Close"] - lower) / bb_range
    
    # Volume Relative (True liquidity indicator)
    vol_mean = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["vol_ratio"] = df["Volume"] / vol_mean
    
    # Short term Momentum
    df["ret_3d"] = df["Close"].pct_change(3)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
    
    return df

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Calculate future returns (shift backwards aligns future with row T)
    fwd_ret = df["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1
    
    df["label"] = "HOLD"
    df.loc[fwd_ret >= THRESHOLD, "label"] = "BUY"
    df.loc[fwd_ret <= -THRESHOLD, "label"] = "SELL"
    
    # VERY IMPORTANT: Prevent Look-Ahead Bias. 
    # Nullify labels for the last FORWARD_DAYS rows.
    df.loc[df.index[-FORWARD_DAYS:], "label"] = np.nan
    return df

@cached(cache)
def fetch_stock_data(ticker: str, period=PERIOD) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()
    
    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close", "Volume"]].copy()

def train_and_evaluate(df: pd.DataFrame):
    # Drop rows without features or known future label
    clean = df.dropna(subset=FEATURES + ["label"])
    
    # STRICT chronological split to avoid Look-Ahead Bias
    split = int(len(clean) * 0.8)
    train, test = clean.iloc[:split], clean.iloc[split:]
    
    X_train, y_train = train[FEATURES], train["label"]
    X_test, y_test = test[FEATURES], test["label"]
    
    # Restrict capacity to prevent overfitting the noise
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,            # Restricted depth is critical for financial data
        min_samples_leaf=5,     # Prevent nodes with single samples
        random_state=42,
        n_jobs=-1,
        class_weight="balanced" # Handle class imbalance
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # We care about Precision (When model says BUY/SELL, how often is it right?)
    # Macro avg precision handles all 3 classes impartially
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    
    return clf, precision

def get_prediction(ticker: str):
    df_raw = fetch_stock_data(ticker)
    if df_raw.empty:
        return None
        
    df = build_features(df_raw)
    df = build_labels(df)
    
    clf, precision = train_and_evaluate(df)
    
    # Predict today (Ensure we don't have NaNs in features)
    latest = df[FEATURES].ffill().iloc[[-1]]
    
    proba = clf.predict_proba(latest)[0]
    classes = clf.classes_
    
    pred_idx = np.argmax(proba)
    raw_signal = classes[pred_idx]
    confidence = proba[pred_idx]
    
    # Apply threshold filtering (If not confident enough -> HOLD)
    final_signal = raw_signal if confidence >= CONFIDENCE_THRESHOLD else "HOLD"
    
    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).to_dict()
    
    return {
        "symbol": ticker,
        "signal": final_signal,
        "confidence": float(confidence),
        "precision_score": float(precision),
        "last_price": float(df["Close"].iloc[-1]),
        "last_date": str(df.index[-1].date()),
        "rows_trained": len(df.dropna(subset=FEATURES + ["label"])),
        "importance": {k: float(v) for k, v in list(importances.items())[:5]} # Top 5
    }
