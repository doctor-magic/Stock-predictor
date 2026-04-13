import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

TICKER = "AZRG.TA"
PERIOD = "5y"
FORWARD_DAYS = 5       # how many days ahead to predict
THRESHOLD = 0.02       # 2% move = BUY or SELL, else HOLD
N_ESTIMATORS = 200
TRAIN_RATIO = 0.8

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",
    "rsi", "macd_gap",
    "bb_pos",
    "vol_ratio",
    "ret_3d", "ret_5d", "ret_10d",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    pass  # Task 3


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    pass  # Task 4


def train_and_evaluate(df: pd.DataFrame) -> tuple[RandomForestClassifier, float]:
    pass  # Task 5


def predict_today(clf: RandomForestClassifier, df: pd.DataFrame) -> None:
    pass  # Task 6


def main() -> None:
    print(f"Downloading {TICKER} data ({PERIOD})...")
    raw = yf.download(TICKER, period=PERIOD, progress=False)
    raw.columns = raw.columns.get_level_values(0)   # flatten MultiIndex if present
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"  {len(df)} rows loaded, last date: {df.index[-1].date()}\n")

    df = add_features(df)
    df = add_labels(df)
    clf, accuracy = train_and_evaluate(df)
    predict_today(clf, df)


if __name__ == "__main__":
    main()
