import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data(show_spinner=False)
def load_sp500():
    """Fetch S&P 500 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return dict(zip(df["Security"], df["Symbol"].str.replace(".", "-", regex=False)))

@st.cache_data(show_spinner=False)
def load_nasdaq100():
    """Fetch NASDAQ-100 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # find the table with Symbol column
    for t in tables:
        if "Ticker" in t.columns or "Symbol" in t.columns:
            col = "Ticker" if "Ticker" in t.columns else "Symbol"
            name_col = "Company" if "Company" in t.columns else t.columns[0]
            return dict(zip(t[name_col], t[col].str.replace(".", "-", regex=False)))
    return NASDAQ100  # fallback to hardcoded

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

# ── NASDAQ-100 tickers ────────────────────────────────────────
NASDAQ100 = {
    "Apple":                  "AAPL",
    "Microsoft":              "MSFT",
    "NVIDIA":                 "NVDA",
    "Amazon":                 "AMZN",
    "Meta":                   "META",
    "Broadcom":               "AVGO",
    "Tesla":                  "TSLA",
    "Costco":                 "COST",
    "Alphabet (GOOGL)":       "GOOGL",
    "Alphabet (GOOG)":        "GOOG",
    "Netflix":                "NFLX",
    "T-Mobile":               "TMUS",
    "ASML":                   "ASML",
    "Cisco":                  "CSCO",
    "Adobe":                  "ADBE",
    "AMD":                    "AMD",
    "PepsiCo":                "PEP",
    "Linde":                  "LIN",
    "Qualcomm":               "QCOM",
    "Intuit":                 "INTU",
    "Texas Instruments":      "TXN",
    "Intuitive Surgical":     "ISRG",
    "Comcast":                "CMCSA",
    "Booking Holdings":       "BKNG",
    "Amgen":                  "AMGN",
    "Honeywell":              "HON",
    "Vertex Pharma":          "VRTX",
    "Palo Alto Networks":     "PANW",
    "ADP":                    "ADP",
    "Starbucks":              "SBUX",
    "Gilead Sciences":        "GILD",
    "Micron":                 "MU",
    "Analog Devices":         "ADI",
    "Intel":                  "INTC",
    "Regeneron":              "REGN",
    "Lam Research":           "LRCX",
    "Mondelez":               "MDLZ",
    "KLA Corp":               "KLAC",
    "Synopsys":               "SNPS",
    "Cadence Design":         "CDNS",
    "MercadoLibre":           "MELI",
    "Airbnb":                 "ABNB",
    "Fortinet":               "FTNT",
    "Cintas":                 "CTAS",
    "CSX":                    "CSX",
    "O'Reilly Auto":          "ORLY",
    "Marvell Tech":           "MRVL",
    "Paccar":                 "PCAR",
    "Workday":                "WDAY",
    "Constellation Energy":   "CEG",
    "NXP Semi":               "NXPI",
    "Copart":                 "CPRT",
    "Keurig Dr Pepper":       "KDP",
    "Ross Stores":            "ROST",
    "Dexcom":                 "DXCM",
    "Diamondback Energy":     "FANG",
    "Trade Desk":             "TTD",
    "Paychex":                "PAYX",
    "GE HealthCare":          "GEHC",
    "AstraZeneca":            "AZN",
    "IDEXX Labs":             "IDXX",
    "Old Dominion Freight":   "ODFL",
    "Fastenal":               "FAST",
    "Cognizant":              "CTSH",
    "Verisk Analytics":       "VRSK",
    "Microchip Tech":         "MCHP",
    "Biogen":                 "BIIB",
    "ON Semi":                "ON",
    "Datadog":                "DDOG",
    "Atlassian":              "TEAM",
    "Exelon":                 "EXC",
    "Xcel Energy":            "XEL",
    "Monster Beverage":       "MNST",
    "ANSYS":                  "ANSS",
    "Charter Comm":           "CHTR",
    "Illumina":               "ILMN",
    "Zscaler":                "ZS",
    "PayPal":                 "PYPL",
    "PDD Holdings":           "PDD",
    "CrowdStrike":            "CRWD",
    "Splunk":                 "SPLK",
    "Rivian":                 "RIVN",
    "Lucid Group":            "LCID",
    "Dollar Tree":            "DLTR",
    "Warner Bros":            "WBD",
    "Zoom":                   "ZM",
    "MongoDB":                "MDB",
    "Snowflake":              "SNOW",
    "Cloudflare":             "NET",
    "Uber":                   "UBER",
    "Arm Holdings":           "ARM",
    "Lululemon":              "LULU",
    "Moderna":                "MRNA",
    "Mercado Libre":          "MELI",
    "Super Micro":            "SMCI",
    "DoorDash":               "DASH",
    "CoStar Group":           "CSGP",
    "Electronic Arts":        "EA",
    "Kraft Heinz":            "KHC",
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

# ── split stocks by market ───────────────────────────────────
TASE_STOCKS = {k: v for k, v in PRESET_STOCKS.items() if v.endswith(".TA")}
US_STOCKS   = {k: v for k, v in PRESET_STOCKS.items() if not v.endswith(".TA") and v != "__custom__"}
STOCK_OPTIONS = {k: v for k, v in PRESET_STOCKS.items() if v != "__custom__"}

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Swing Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Swing Predictor")
st.caption("Random Forest · 3-5 day outlook · No cloud calls · Free data")

tab1, tab2 = st.tabs(["🔍 מניה בודדת", "📊 סורק מניות"])

# ── TAB 1: single stock ──────────────────────────────────────
with tab1:
    market = st.radio(
        "שוק",
        ["🇮🇱 ישראל (TASE)", "🇺🇸 ארה״ב (US)", "🌍 הכל"],
        horizontal=True,
    )

    if market == "🇮🇱 ישראל (TASE)":
        filtered = TASE_STOCKS
    elif market == "🇺🇸 ארה״ב (US)":
        filtered = US_STOCKS
    else:
        filtered = STOCK_OPTIONS

    choice = st.selectbox(
        label="בחר מניה",
        options=list(filtered.keys()),
        index=0,
        placeholder="התחל להקליד שם מניה...",
        label_visibility="collapsed",
    )

    st.caption("מניה שלא ברשימה? הקלד את הסימול ישירות:")
    custom_ticker = st.text_input("סימול מותאם אישית (לדוג׳ MSFT או BEZQ.TA)", "").strip().upper()
    ticker = custom_ticker if custom_ticker else filtered[choice]

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
    st.markdown("סרוק מניות ומיין לפי Confidence")

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
    with col_a:
        scan_market = st.selectbox("שוק", ["🌍 הכל", "🇮🇱 ישראל (TASE)", "🇺🇸 ארה״ב (US)", "📈 NASDAQ-100", "📊 S&P 500"])
    with col_b:
        filter_signal = st.selectbox("סנן לפי Signal", ["הכל", "BUY", "SELL", "HOLD"])
    with col_c:
        top_n = st.number_input("הצג Top N", min_value=1, max_value=100, value=10)

    if scan_market == "🇮🇱 ישראל (TASE)":
        scan_list = TASE_STOCKS
    elif scan_market == "🇺🇸 ארה״ב (US)":
        scan_list = US_STOCKS
    elif scan_market == "📈 NASDAQ-100":
        scan_list = load_nasdaq100()
        st.warning("סריקת NASDAQ-100 תיקח כ-8-10 דקות. הראשונה בלבד — הבאות מהירות יותר.")
    elif scan_market == "📊 S&P 500":
        scan_list = load_sp500()
        st.warning("סריקת S&P 500 תיקח כ-40-45 דקות. הראשונה בלבד — הבאות מהירות יותר.")
    else:
        scan_list = STOCK_OPTIONS

    if st.button("סרוק מניות", type="primary"):
        rows = []
        progress = st.progress(0, text="מוריד נתונים ומאמן מודלים...")
        tickers_list = list(scan_list.items())

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
            st.caption(f"מציג {len(df_results)} מניות מתוך {len(scan_list)} · ממוין לפי Confidence")
