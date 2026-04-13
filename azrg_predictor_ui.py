import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── constants ────────────────────────────────────────────────
PERIOD       = "5y"
FORWARD_DAYS = 5
THRESHOLD    = 0.02
N_ESTIMATORS = 200
TRAIN_RATIO  = 0.8

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",
    "rsi", "macd_gap", "bb_pos",
    "vol_ratio", "ret_3d", "ret_5d", "ret_10d",
]

PRESET_STOCKS = {
    # ── TASE ──────────────────────────────────────────
    "אזריאלי   | Azrieli Group":        "AZRG.TA",
    "אלביט     | Elbit Systems":        "ESLT.TA",
    "טבע       | Teva":                 "TEVA.TA",
    "נייס      | NICE Systems":         "NICE.TA",
    "סייברארק  | CyberArk":             "CYBR.TA",
    "בנק הפועלים | Bank Hapoalim":      "POLI.TA",
    "בנק לאומי  | Bank Leumi":          "LUMI.TA",
    "בנק מזרחי  | Mizrahi Tefahot":     "MZTF.TA",
    "בזק       | Bezeq":               "BEZQ.TA",
    "פרטנר     | Partner":             "PTNR.TA",
    "ICL Group":                        "ICL.TA",
    "צ'ק פוינט  | Check Point":         "CHKP.TA",
    "טאואר     | Tower Semiconductor": "TSEM.TA",
    "סאפיינס   | Sapiens":             "SPNS.TA",
    # ── US ────────────────────────────────────────────
    "Apple":                            "AAPL",
    "NVIDIA":                           "NVDA",
    "Tesla":                            "TSLA",
    "Microsoft":                        "MSFT",
    "Amazon":                           "AMZN",
    "Google":                           "GOOGL",
    "Meta":                             "META",
    "S&P 500 ETF (SPY)":                "SPY",
    # ── Custom ────────────────────────────────────────
    "Custom ticker...":                 "__custom__",
}

# ── helpers ──────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df):
    df = df.copy()
    df["ema9"]      = df["Close"].ewm(span=9,  adjust=False).mean()
    df["ema21"]     = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"]     = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_cross"] = (df["ema9"] > df["ema21"]).astype(int)
    df["rsi"]       = compute_rsi(df["Close"])
    ema12           = df["Close"].ewm(span=12, adjust=False).mean()
    ema26           = df["Close"].ewm(span=26, adjust=False).mean()
    macd            = ema12 - ema26
    df["macd_gap"]  = macd - macd.ewm(span=9, adjust=False).mean()
    sma20           = df["Close"].rolling(20).mean()
    std20           = df["Close"].rolling(20).std()
    df["bb_pos"]    = (df["Close"] - (sma20 - 2*std20)) / (4*std20)
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["ret_3d"]    = df["Close"].pct_change(3)
    df["ret_5d"]    = df["Close"].pct_change(5)
    df["ret_10d"]   = df["Close"].pct_change(10)
    return df

def build_labels(df):
    df       = df.copy()
    fwd      = df["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1
    df["label"] = "HOLD"
    df.loc[fwd >  THRESHOLD, "label"] = "BUY"
    df.loc[fwd < -THRESHOLD, "label"] = "SELL"
    df.loc[df.index[-FORWARD_DAYS:], "label"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def run_prediction(ticker):
    raw = yf.download(ticker, period=PERIOD, progress=False)
    if raw.empty:
        return None
    raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = build_features(df)
    df = build_labels(df)

    clean = df.dropna(subset=FEATURES + ["label"])
    split = int(len(clean) * TRAIN_RATIO)
    train, test = clean.iloc[:split], clean.iloc[split:]

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
    clf.fit(train[FEATURES], train["label"])

    accuracy = accuracy_score(test["label"], clf.predict(test[FEATURES]))

    latest     = df[FEATURES].dropna().iloc[[-1]]
    proba      = clf.predict_proba(latest)[0]
    pred_class = clf.classes_[np.argmax(proba)]
    confidence = np.max(proba)

    importance = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)

    last_price = float(df["Close"].iloc[-1])
    last_date  = df.index[-1].date()

    return {
        "signal":     pred_class,
        "confidence": confidence,
        "accuracy":   accuracy,
        "importance": importance,
        "rows":       len(df),
        "last_price": last_price,
        "last_date":  last_date,
    }

# ── build searchable list ─────────────────────────────────────
# Each entry: display label → ticker symbol
# We add a plain-English + Hebrew alias string so typing either works
STOCK_OPTIONS = {k: v for k, v in PRESET_STOCKS.items() if v != "__custom__"}

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Swing Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Swing Predictor")
st.caption("Random Forest · 3-5 day outlook · No cloud calls · Free data")

tab1, tab2 = st.tabs(["🔍 מניה בודדת", "📊 סורק מניות"])

# ── TAB 1: single stock ──────────────────────────────────────
with tab1:
    st.markdown("**חפש מניה** — הקלד שם בעברית, אנגלית, או סימול (לדוג׳ ESLT.TA)")

    choice = st.selectbox(
        label="בחר מניה",
        options=list(STOCK_OPTIONS.keys()),
        index=0,
        placeholder="התחל להקליד שם מניה...",
        label_visibility="collapsed",
    )

    st.caption("מניה שלא ברשימה? הקלד את הסימול ישירות:")
    custom_ticker = st.text_input("סימול מותאם אישית (לדוג׳ MSFT או BEZQ.TA)", "").strip().upper()
    ticker = custom_ticker if custom_ticker else STOCK_OPTIONS[choice]

    if st.button("Run Prediction", type="primary", disabled=not ticker):
        with st.spinner(f"Downloading {ticker} data and training model..."):
            result = run_prediction(ticker)

        if result is None:
            st.error(f"Could not download data for **{ticker}**. Check the ticker symbol and try again.")
        else:
            st.divider()
            signal    = result["signal"]
            color_map = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
            color     = color_map.get(signal, "gray")

            c1, c2, c3 = st.columns(3)
            c1.metric("Signal",     signal)
            c2.metric("Confidence", f"{result['confidence']:.0%}")
            c3.metric("Backtest Accuracy", f"{result['accuracy']:.1%}")

            st.markdown(f"### :{color}[{signal}]")
            st.caption(f"Last price: {result['last_price']:,.2f}  |  Date: {result['last_date']}  |  {result['rows']} trading days of history")

            st.subheader("Feature importance")
            st.bar_chart(result["importance"])

            st.info(
                "Backtest accuracy above ~55% suggests the model has found a pattern. "
                "Below 40% means the signal is weak — use alongside other analysis."
            )

# ── TAB 2: scanner ───────────────────────────────────────────
with tab2:
    st.markdown("סרוק את כל המניות ברשימה ומיין לפי Confidence")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        filter_signal = st.selectbox("סנן לפי Signal", ["הכל", "BUY", "SELL", "HOLD"])
    with col_b:
        top_n = st.number_input("הצג Top N", min_value=1, max_value=50, value=10)

    if st.button("סרוק את כל המניות", type="primary"):
        rows = []
        progress = st.progress(0, text="מוריד נתונים ומאמן מודלים...")
        tickers_list = list(STOCK_OPTIONS.items())

        for i, (name, sym) in enumerate(tickers_list):
            progress.progress((i + 1) / len(tickers_list), text=f"מעבד {sym}...")
            res = run_prediction(sym)
            if res:
                rows.append({
                    "מניה":       name,
                    "סימול":      sym,
                    "Signal":     res["signal"],
                    "Confidence": round(res["confidence"] * 100, 1),
                    "Accuracy":   round(res["accuracy"] * 100, 1),
                    "מחיר":       round(res["last_price"], 2),
                })

        progress.empty()

        df_results = pd.DataFrame(rows)

        # filter
        if filter_signal != "הכל":
            df_results = df_results[df_results["Signal"] == filter_signal]

        # sort by confidence, take top N
        df_results = df_results.sort_values("Confidence", ascending=False).head(top_n).reset_index(drop=True)
        df_results.index += 1  # start from 1

        if df_results.empty:
            st.warning("לא נמצאו מניות עם הפילטר שנבחר.")
        else:
            def color_signal(val):
                colors = {"BUY": "background-color: #d4edda; color: #155724",
                          "SELL": "background-color: #f8d7da; color: #721c24",
                          "HOLD": "background-color: #fff3cd; color: #856404"}
                return colors.get(val, "")

            styled = df_results.style.map(color_signal, subset=["Signal"])
            st.dataframe(styled, use_container_width=True)
            st.caption(f"מציג {len(df_results)} מניות מתוך {len(STOCK_OPTIONS)} · ממוין לפי Confidence")
