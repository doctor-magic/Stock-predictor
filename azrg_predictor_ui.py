import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import requests
from io import StringIO

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

@st.cache_data(show_spinner=False)
def load_sp500():
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=HEADERS, timeout=15).text
    df   = pd.read_html(StringIO(html))[0]
    return dict(zip(df["Security"], df["Symbol"].str.replace(".", "-", regex=False)))

@st.cache_data(show_spinner=False)
def load_nasdaq100():
    url  = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = requests.get(url, headers=HEADERS, timeout=15).text
    for t in pd.read_html(StringIO(html)):
        if "Ticker" in t.columns or "Symbol" in t.columns:
            col      = "Ticker" if "Ticker" in t.columns else "Symbol"
            name_col = "Company" if "Company" in t.columns else t.columns[0]
            return dict(zip(t[name_col], t[col].str.replace(".", "-", regex=False)))
    return NASDAQ100

# ── constants ────────────────────────────────────────────────
PERIOD       = "10y"
FORWARD_DAYS = 5
THRESHOLD    = 0.03
N_ESTIMATORS = 300
TRAIN_RATIO  = 0.8

FEATURES = [
    "ema9", "ema21", "ema50", "ema_cross",
    "rsi", "macd_gap", "bb_pos",
    "vol_ratio", "ret_3d", "ret_5d", "ret_10d",
    "atr_ratio", "obv_ratio",
]

PRESET_STOCKS = {
    # ── TA-35 ─────────────────────────────────────────
    # בנקים
    "בנק הפועלים | Bank Hapoalim":      "POLI.TA",
    "בנק לאומי   | Bank Leumi":         "LUMI.TA",
    "בנק מזרחי   | Mizrahi Tefahot":    "MZTF.TA",
    "בנק דיסקונט | Discount Bank":      "DSCT.TA",
    "הבנק הבינלאומי | First Intl Bank": "FTIN.TA",
    # ביטוח ופיננסים
    "הראל        | Harel Insurance":    "HARL.TA",
    "הפניקס      | Phoenix Holdings":   "PHOE.TA",
    "כלל ביטוח   | Clal Insurance":     "CALI.TA",
    "מגדל        | Migdal Insurance":   "MGDL.TA",
    "מנורה מבטחים | Menora Mivtachim": "MNRT.TA",
    # טכנולוגיה
    "צ'ק פוינט   | Check Point":        "CHKP.TA",
    "נייס        | NICE Systems":       "NICE.TA",
    "סייברארק    | CyberArk":           "CYBR.TA",
    "סאפיינס     | Sapiens":            "SPNS.TA",
    "טאואר       | Tower Semiconductor":"TSEM.TA",
    "אלביט       | Elbit Systems":      "ESLT.TA",
    # פרמה ותעשייה
    "טבע         | Teva":               "TEVA.TA",
    "כיל         | ICL Group":          "ICL.TA",
    # נדל"ן
    "אזריאלי     | Azrieli Group":      "AZRG.TA",
    "אלוני חץ    | Alony Hetz":         "ALHE.TA",
    "מליסרון     | Melisron":           "MLSR.TA",
    "ביג         | Big Centers":        "BIG.TA",
    "שיכון ובינוי | Shikun & Binui":   "SKBN.TA",
    # תקשורת
    "בזק         | Bezeq":             "BEZQ.TA",
    "פרטנר       | Partner":           "PTNR.TA",
    "סלקום       | Cellcom":           "SCEL.TA",
    # אנרגיה
    "דלק קבוצה   | Delek Group":       "DLEKG.TA",
    "פז נפט      | Paz Oil":           "PZOL.TA",
    "בתי זיקוק   | Bazan (ORL)":       "ORL.TA",
    "אנליט       | Enlight Energy":    "ENLT.TA",
    # מזון וקמעונאות
    "שטראוס      | Strauss Group":     "STRS.TA",
    "שופרסל      | Shufersal":         "SAE.TA",
    "רמי לוי     | Rami Levy":         "RMLI.TA",
    # אחר
    "מיבנה        | Mivne Real Estate": "MVNE.TA",
    "אפריקה ישראל | Africa Israel":    "AFPR.TA",
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
    # ATR (normalized by price)
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    atr        = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_ratio"] = atr.rolling(14).mean() / df["Close"]
    # OBV (normalized vs rolling mean)
    df["obv"]       = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    obv_mean        = df["obv"].rolling(20).mean().replace(0, np.nan)
    df["obv_ratio"] = df["obv"] / obv_mean
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

    le = LabelEncoder().fit(["BUY", "HOLD", "SELL"])
    clf = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(train[FEATURES], le.transform(train["label"]))

    accuracy = accuracy_score(test["label"], le.inverse_transform(clf.predict(test[FEATURES])))

    latest     = df[FEATURES].dropna().iloc[[-1]]
    proba      = clf.predict_proba(latest)[0]
    pred_class = le.inverse_transform([np.argmax(proba)])[0]
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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600&display=swap');

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0c29 40%, #302b63 80%, #24243e 100%);
        color: #e8e8f0;
        font-family: 'Fira Sans', -apple-system, sans-serif;
    }

    /* Pulse animation for live indicators */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 4px #22C55E; opacity: 1; }
        50%       { box-shadow: 0 0 14px #22C55E, 0 0 28px rgba(34,197,94,0.3); opacity: 0.8; }
    }
    @keyframes signal-fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Title */
    h1 {
        font-size: 2.6rem !important; font-weight: 800 !important;
        background: linear-gradient(90deg, #00d2ff, #a200ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-family: 'Fira Code', monospace !important;
        letter-spacing: -0.02em;
    }
    h2, h3 { font-family: 'Fira Code', monospace !important; color: #e8e8f0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem; font-weight: 600; color: #888;
        transition: color 150ms ease-out;
        font-family: 'Fira Sans', sans-serif;
    }
    .stTabs [aria-selected="true"] { color: #00d2ff !important; border-bottom: 2px solid #00d2ff; }
    .stTabs [data-baseweb="tab"]:hover { color: #ccc !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 20px;
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        transition: border-color 200ms ease-out, box-shadow 200ms ease-out;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(0,210,255,0.3);
        box-shadow: 0 0 20px rgba(0,210,255,0.07);
    }
    [data-testid="metric-container"] label {
        color: #888 !important; font-size: 0.78rem !important;
        text-transform: uppercase; letter-spacing: 0.08em;
        font-family: 'Fira Sans', sans-serif !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 700 !important;
        color: #fff !important; font-family: 'Fira Code', monospace !important;
    }

    /* Signal badge */
    .signal-badge {
        display: inline-block; padding: 0.55rem 2rem;
        border-radius: 50px; font-family: 'Fira Code', monospace;
        font-weight: 700; font-size: 1.3rem; letter-spacing: 0.12em;
        text-transform: uppercase; animation: signal-fade-in 300ms ease-out;
    }
    .signal-buy  { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.35);  box-shadow: 0 0 24px rgba(34,197,94,0.15); }
    .signal-sell { background: rgba(220,38,38,0.12);  color: #f87171; border: 1px solid rgba(220,38,38,0.35);  box-shadow: 0 0 24px rgba(220,38,38,0.15); }
    .signal-hold { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.35); box-shadow: 0 0 24px rgba(251,191,36,0.12); }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #00d2ff, #a200ff) !important;
        border: none !important; color: white !important;
        font-weight: 700 !important; border-radius: 10px !important;
        padding: 0.6rem 2.5rem !important; font-size: 1rem !important;
        font-family: 'Fira Sans', sans-serif !important;
        transition: opacity 150ms ease-out, transform 150ms ease-out, box-shadow 150ms ease-out !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9; transform: scale(1.02);
        box-shadow: 0 4px 24px rgba(0,210,255,0.28) !important;
    }
    .stButton > button[kind="primary"]:active { transform: scale(0.98) !important; }

    /* Inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput input {
        background: rgba(255,255,255,0.07) !important; color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 10px !important;
        font-family: 'Fira Sans', sans-serif !important;
        transition: border-color 150ms ease-out !important;
    }
    .stSelectbox svg { fill: #fff !important; }

    /* Radio */
    .stRadio label { color: #ccc !important; font-size: 0.95rem !important; font-family: 'Fira Sans', sans-serif !important; }

    /* General text */
    p, label, .stMarkdown { color: #ccc !important; font-family: 'Fira Sans', sans-serif; }

    /* Dropdown */
    [data-baseweb="popover"] { background: #1a1730 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
    [data-baseweb="option"] { background: #1a1730 !important; color: #ddd !important; }
    [data-baseweb="option"]:hover { background: #302b63 !important; }

    /* Dataframe */
    .stDataFrame { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08) !important; }

    /* Progress bar */
    .stProgress > div > div > div { background: linear-gradient(90deg, #00d2ff, #a200ff) !important; border-radius: 4px !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Caption */
    .stCaption { color: #666 !important; font-size: 0.82rem !important; }

    /* Alert boxes */
    .stAlert { border-radius: 10px !important; }

    /* Expander headers */
    .streamlit-expanderHeader { color: #e0e0e0 !important; font-weight: 600 !important; font-family: 'Fira Sans', sans-serif !important; }
    .streamlit-expanderHeader:hover { color: #00d2ff !important; }
    [data-testid="stExpander"] details summary { color: #e0e0e0 !important; font-weight: 600 !important; }
    [data-testid="stExpander"] details summary:hover { color: #00d2ff !important; }
    [data-testid="stExpander"] details summary p { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Swing Predictor")
st.caption("XGBoost · 3-5 day outlook · 10y data · Free data")

st.markdown("""
<details style="
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 18px;
    font-family: Fira Sans, sans-serif;
">
<summary style="
    color: #a0aec0;
    font-size: 0.88rem;
    cursor: pointer;
    user-select: none;
    list-style: none;
">
    &#9432;&nbsp; איך המודל עובד? (לחץ להרחבה)
</summary>
<div style="margin-top: 10px; color: #8a9bb5; font-size: 0.85rem; line-height: 1.7; direction: rtl; text-align: right;">
    <b style="color:#c0cfe0;">מה המודל עושה?</b><br>
    המודל מנתח אינדיקטורים טכניים (RSI, MACD, EMA, ATR, OBV ועוד) ומנסה לחזות
    האם המניה תעלה או תרד <b>ב-5 הימים הקרובים</b> בלבד. אין כאן ניתוח פונדמנטלי.
    <br><br>
    <b style="color:#c0cfe0;">למה מניה שעלתה מקבלת SELL?</b><br>
    עלייה חדה מביאה RSI גבוה, מחיר מעל פס בולינגר, ומומנטום קצר-טווח שמיצה את עצמו —
    דפוסים שהמודל לומד לקשר לתיקון של 3%+ בשבוע הקרוב. <b>SELL ≠ מניה גרועה</b>,
    אלא "תיקון טכני צפוי בטווח הקצר".
    <br><br>
    <b style="color:#c0cfe0;">Confidence vs Accuracy</b><br>
    <b>Confidence</b> — כמה המודל בטוח בחיזוי הנוכחי.<br>
    <b>Accuracy</b> — כמה % מהחיזויים ההיסטוריים של המניה הזו היו נכונים.
    מומלץ להסתמך רק על תוצאות עם Accuracy &gt; 55%.
    <br><br>
    <span style="color:#6b7a99; font-size:0.78rem;">
    &#9888; אין בכלי זה ייעוץ השקעות. המידע מיועד למטרות לימודיות בלבד.
    </span>
</div>
</details>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["🔍 מניה בודדת", "📊 סורק מניות", "📰 המלצות יומיות"])

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
            signal     = result["signal"]
            badge_cls  = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(signal, "signal-hold")
            badge_icon = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}.get(signal, "●")

            st.markdown(
                f'<div style="text-align:center; padding: 1.2rem 0 0.8rem;">'
                f'<span class="signal-badge {badge_cls}">{badge_icon} {signal}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence",       f"{result['confidence']:.0%}")
            c2.metric("Backtest Accuracy", f"{result['accuracy']:.1%}")
            c3.metric("Last Price",        f"{result['last_price']:,.2f}")
            c4.metric("Trading Days",      f"{result['rows']:,}")

            st.markdown(f'<p style="font-size:0.85rem; color:#888;"><a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" style="color:#4a9eff; text-decoration:none; font-weight:bold;">{ticker}</a> · {result["last_date"]}</p>', unsafe_allow_html=True)

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

    import os, json
    cache_file = os.path.join(os.path.dirname(__file__), f"scan_cache_{scan_market.split()[0].strip()}.json")

    def show_results(df_all):
        df_view = df_all.copy()
        # Filter and sort while values are still numeric
        if filter_signal != "הכל":
            df_view = df_view[df_view["Signal"] == filter_signal]
        df_view = df_view.sort_values("Confidence", ascending=False).head(top_n).reset_index(drop=True)
        # Format for display only after sorting
        df_view["Confidence"] = df_view["Confidence"].apply(lambda x: f"{x:.1f}%")
        df_view["Accuracy"]   = df_view["Accuracy"].apply(lambda x: f"{x:.1f}%")
        df_view["מחיר"]       = df_view["מחיר"].apply(lambda x: f"{x:,.2f}")
        df_view["סימול"]      = df_view["סימול"].apply(lambda s: f"https://finance.yahoo.com/quote/{s}")
        df_view.index += 1
        if df_view.empty:
            st.warning("לא נמצאו מניות עם הפילטר שנבחר.")
        else:
            SIG_STYLE = {
                "BUY":  "background:#0d3320;color:#4ade80;font-weight:bold",
                "SELL": "background:#3b0d0d;color:#f87171;font-weight:bold",
                "HOLD": "background:#3b2d00;color:#fbbf24;font-weight:bold",
            }
            TD = 'style="padding:8px 14px;border-bottom:1px solid #2a2a3e;'
            rows_html = ""
            for i, row in df_view.iterrows():
                conf_num = float(row["Confidence"].replace("%",""))
                conf_col = "#4ade80" if conf_num >= 70 else "#fbbf24" if conf_num >= 55 else "#f87171"
                sym_url  = row["סימול"]
                sym_name = sym_url.split("/quote/")[-1]
                rows_html += (
                    f'<tr>' +
                    f'<td {TD}color:#666;">{i}</td>' +
                    f'<td {TD}color:#e0e0e0;">{row["מניה"]}</td>' +
                    f'<td {TD}">' +
                    f'<a href="{sym_url}" target="_blank" ' +
                    f'style="color:#38bdf8;text-decoration:underline;font-weight:700;font-family:monospace;font-size:0.9rem;">{sym_name}</a></td>' +
                    f'<td {TD}{SIG_STYLE.get(row["Signal"],"")};">{row["Signal"]}</td>' +
                    f'<td {TD}color:{conf_col};font-weight:bold;">{row["Confidence"]}</td>' +
                    f'<td {TD}color:#e0e0e0;">{row["Accuracy"]}</td>' +
                    f'<td {TD}color:#e0e0e0;text-align:right;">{row["מחיר"]}</td>' +
                    f'</tr>'
                )
            th = 'style="padding:10px 14px;background:#0f0c29;color:#aaa;font-size:0.82rem;font-weight:600;text-align:left;border-bottom:2px solid #2a2a3e;"'
            html_table = (
                '<div style="overflow-x:auto;border-radius:14px;border:1px solid rgba(255,255,255,0.08);">' +
                '<table style="width:100%;border-collapse:collapse;background:#1a1a2e;font-family:Fira Sans,sans-serif;font-size:0.9rem;">' +
                f'<thead><tr><th {th}>#</th><th {th}>מניה</th><th {th}>סימול</th>' +
                f'<th {th}>Signal</th><th {th}>Confidence</th><th {th}>Accuracy</th><th {th}>מחיר</th></tr></thead>' +
                f'<tbody>{rows_html}</tbody></table></div>'
            )
            st.markdown(html_table, unsafe_allow_html=True)
            updated = json.load(open(cache_file, encoding="utf-8"))["updated"] if os.path.exists(cache_file) else "לא ידוע"
            st.caption(f"מציג {len(df_view)} מניות מתוך {len(df_all)} · ממוין לפי Confidence · עודכן: {updated}")

    # show cached results if exist
    if os.path.exists(cache_file):
        cached = json.load(open(cache_file, encoding="utf-8"))
        df_cached = pd.DataFrame(cached["rows"])
        st.success(f"תוצאות שמורות מ-{cached['updated']} — לחץ 'עדכן סריקה' לרענון")
        show_results(df_cached)

    btn_label = "עדכן סריקה" if os.path.exists(cache_file) else "סרוק מניות"
    if st.button(btn_label, type="primary"):
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
        from datetime import date
        payload = {"updated": str(date.today()), "rows": rows}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        st.success(f"סריקה הושלמה — {len(rows)} מניות נסרקו. דחוף ל-GitHub כדי לשתף עם החברים.")
        show_results(pd.DataFrame(rows))

# ── TAB 3: daily recommendations ─────────────────────────────
with tab3:
    import glob, re

    rec_dir   = os.path.dirname(os.path.abspath(__file__))
    rec_files = sorted(
        glob.glob(os.path.join(rec_dir, "stock_recommendations_*.txt")),
        reverse=True,
    )

    if not rec_files:
        st.warning("לא נמצאו קבצי המלצות. הפעל את ה-pipeline להוספת המלצות.")
    else:
        MONTHS_HE = {
            "01": "ינואר", "02": "פברואר", "03": "מרץ",    "04": "אפריל",
            "05": "מאי",   "06": "יוני",   "07": "יולי",   "08": "אוגוסט",
            "09": "ספטמבר","10": "אוקטובר","11": "נובמבר", "12": "דצמבר",
        }

        def fname_to_label(path):
            m = re.search(r"stock_recommendations_(\d{2})_(\d{2})_(\d{4})", path)
            if m:
                d, mo, y = m.groups()
                return f"{d} ב{MONTHS_HE.get(mo, mo)} {y}"
            return os.path.basename(path)

        options        = {fname_to_label(f): f for f in rec_files}
        selected_label = st.selectbox("תאריך", list(options.keys()), index=0)
        selected_file  = options[selected_label]

        with open(selected_file, encoding="utf-8") as fh:
            content = fh.read()

        # ── Parse sections separated by --- Title ---
        lines    = content.splitlines()
        sections = []
        cur_title, cur_body = None, []

        for line in lines:
            stripped = line.strip()
            if re.match(r"^---.*---$", stripped) and len(stripped) > 6:
                if cur_title is not None:
                    sections.append((cur_title, "\n".join(cur_body).strip()))
                cur_title = stripped.strip("-").strip()
                cur_body  = []
            elif re.match(r"^## ", stripped):
                if cur_title is not None:
                    sections.append((cur_title, "\n".join(cur_body).strip()))
                cur_title = stripped.lstrip("#").strip()
                cur_body  = []
            elif stripped == "---" or stripped.startswith("==="):
                continue
            elif re.match(r"^# ", stripped):
                continue
            else:
                cur_body.append(line)

        if cur_title is not None:
            sections.append((cur_title, "\n".join(cur_body).strip()))

        # ── Find overall date header
        header_text = next(
            (l.strip().lstrip("#").strip() for l in lines
             if l.strip() and not l.strip().startswith("=") and not l.strip().startswith("---")),
            ""
        )
        if header_text:
            st.markdown(
                f'<p style="text-align:right; color:#aaa; font-size:0.9rem; margin-bottom:1rem;">'
                f'{header_text}</p>',
                unsafe_allow_html=True,
            )

        if not sections:
            st.text(content)
        else:
            for title, body in sections:
                html_body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_body = html_body.replace("\n", "<br>")
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);'
                    f'border-radius:12px;padding:16px 22px;margin-bottom:14px;">'
                    f'<p style="color:#00d2ff;font-weight:700;font-size:1rem;margin:0 0 10px 0;'
                    f'direction:rtl;text-align:right;font-family:Fira Sans,sans-serif;">{title}</p>'
                    f'<div style="direction:rtl;text-align:right;line-height:1.9;'
                    f'color:#d0d0d0;font-size:0.95rem;">{html_body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
