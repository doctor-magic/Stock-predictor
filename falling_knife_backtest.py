#!/usr/bin/env python3
"""
falling_knife_backtest.py
Hypothesis: stocks down ≥8% by 14:30 ET bounce in Power Hour (15:00 → close)?
Compares FALLING KNIFE days vs all other days for the same tickers.
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats

TICKERS = [
    # Recent Reversion Hunter appearances
    'CELC', 'ABVX', 'MSTR', 'OSCR', 'CBOE', 'MIAX', 'BRZE', 'CVNA',
    'NU',   'PATH', 'INTU', 'SHAK', 'ZS',   'XMTR', 'HUBS', 'IMNM',
    'TTD',  'FIG',  'RVMD', 'SARO', 'KVYO',
    # Volume Leaders regulars
    'NVDA', 'INTC', 'SOFI', 'SMCI', 'RGTI', 'QBTS', 'IREN', 'PLTR',
    'MRVL', 'HPE',  'NOK',  'SMR',  'PLUG', 'ONDS', 'BB',   'BBAI',
    'AAL',  'ACHR', 'OPEN', 'F',    'KEEL',
    # Large-cap common day_losers
    'TSLA', 'AMD',  'COIN', 'HOOD', 'RIVN', 'NFLX', 'META',
    'AMZN', 'GOOG', 'AAPL', 'MSFT', 'ARKK',
]

DROP_THRESH  = -8.0   # % drop by 14:30 ET to classify as FALLING KNIFE
DETECT_HOUR  = '14:30'
PH_START     = '15:00'
CLOSE_START  = '15:55'
CLOSE_END    = '16:01'
MIN_EVENTS   = 10     # minimum events for meaningful statistics


def _rsi14(s: pd.Series) -> float:
    """RSI-14 on a price series, returns last value."""
    if len(s) < 15:
        return float('nan')
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - 100 / (1 + rs))


def main():
    print(f"Downloading 5-min bars (60 days) for {len(TICKERS)} tickers…")
    raw = yf.download(
        TICKERS, period='60d', interval='5m',
        progress=True, auto_adjust=True, group_by='ticker',
    )

    events  = []   # FALLING KNIFE days
    control = []   # all other normal days

    for ticker in TICKERS:
        try:
            if ticker not in raw.columns.get_level_values(0):
                continue
            closes = raw[ticker]['Close'].dropna()
            if closes.empty:
                continue
            closes.index = closes.index.tz_convert('America/New_York')

            for date, day in closes.groupby(closes.index.date):
                # --- open price ---
                open_bars = day.between_time('09:30', '09:36')
                if open_bars.empty:
                    continue
                open_price = float(open_bars.iloc[0])
                if open_price <= 0:
                    continue

                # --- price at detection time (14:30 ET) ---
                detect_bars = day.between_time('14:25', '14:35')
                if detect_bars.empty:
                    continue
                price_detect = float(detect_bars.iloc[-1])

                # --- price at Power Hour start (15:00 ET) ---
                ph_bars = day.between_time('15:00', '15:06')
                if ph_bars.empty:
                    continue
                price_ph = float(ph_bars.iloc[0])

                # --- close price (last bar ≥15:55) ---
                close_bars = day.between_time(CLOSE_START, CLOSE_END)
                if close_bars.empty:
                    close_bars = day.between_time('15:50', CLOSE_END)
                    if close_bars.empty:
                        continue
                price_close = float(close_bars.iloc[-1])

                # --- RSI at detection time (intraday bars up to 14:30) ---
                hist = day[day.index.time <= pd.Timestamp(DETECT_HOUR).time()]
                rsi_val = _rsi14(hist)

                # --- metrics ---
                change_pct = (price_detect / open_price - 1) * 100
                ph_return  = (price_close  / price_ph   - 1) * 100

                row = dict(
                    ticker=ticker, date=str(date),
                    change_pct=round(change_pct, 2),
                    rsi=round(rsi_val, 1) if not np.isnan(rsi_val) else None,
                    price_detect=round(price_detect, 2),
                    price_ph=round(price_ph, 2),
                    price_close=round(price_close, 2),
                    ph_return=round(ph_return, 3),
                )

                if change_pct <= DROP_THRESH:
                    events.append(row)
                else:
                    control.append(row)

        except Exception as e:
            print(f"  {ticker}: {e}")

    # ── Results ──────────────────────────────────────────────────────────────
    df_fk  = pd.DataFrame(events)
    df_ctl = pd.DataFrame(control)

    sep = '=' * 55
    print(f"\n{sep}")
    print(f"FALLING KNIFE events (drop ≤ {DROP_THRESH}% by 14:30 ET): {len(df_fk)}")
    print(f"Control days (normal):                                   {len(df_ctl)}")

    if len(df_fk) < MIN_EVENTS:
        print(f"\n⚠ Only {len(df_fk)} events — need ≥{MIN_EVENTS} for statistics.")
        print("Suggestion: lower DROP_THRESH or add more tickers.")
        return

    fk_ph    = df_fk['ph_return'].dropna()
    ctl_ph   = df_ctl['ph_return'].dropna()

    mean_fk  = fk_ph.mean()
    med_fk   = fk_ph.median()
    win_fk   = (fk_ph > 0).mean() * 100
    t1, p1   = stats.ttest_1samp(fk_ph, 0)

    mean_ctl = ctl_ph.mean()
    win_ctl  = (ctl_ph > 0).mean() * 100
    t2, p2   = stats.ttest_ind(fk_ph, ctl_ph)

    print(f"\n{'─'*55}")
    print(f"FALLING KNIFE → Power Hour return:")
    print(f"  N={len(fk_ph):<5}  Mean={mean_fk:+.3f}%  Median={med_fk:+.3f}%")
    print(f"  Win rate: {win_fk:.1f}%")
    print(f"  vs zero:  t={t1:.2f}  p={p1:.4f}")

    print(f"\nControl → Power Hour return:")
    print(f"  N={len(ctl_ph):<5}  Mean={mean_ctl:+.3f}%")
    print(f"  Win rate: {win_ctl:.1f}%")
    print(f"  FK vs Control: t={t2:.2f}  p={p2:.4f}")

    print(f"\n{'─'*55}")
    if p1 < 0.05 and mean_fk > 0:
        print("✅ STATISTICALLY SIGNIFICANT positive bounce in Power Hour")
    elif p1 < 0.10 and mean_fk > 0:
        print("⚠ MARGINAL significance — collect more data")
    elif mean_fk > 0:
        print(f"➡ Positive trend ({mean_fk:+.3f}%) but NOT significant (p={p1:.3f})")
    else:
        print(f"❌ No bounce — mean={mean_fk:+.3f}%, p={p1:.3f}")

    # ── Top events ───────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Top 15 FALLING KNIFE Power Hour bounces:")
    cols = ['ticker', 'date', 'change_pct', 'rsi', 'ph_return']
    top = df_fk[cols].sort_values('ph_return', ascending=False).head(15)
    print(top.to_string(index=False))

    print(f"\nBottom 10 (worst):")
    bot = df_fk[cols].sort_values('ph_return').head(10)
    print(bot.to_string(index=False))

    # ── Save results ─────────────────────────────────────────────────────────
    out = '/Users/elim/Desktop/Stock-predictor/falling_knife_results.csv'
    df_fk.sort_values('ph_return', ascending=False).to_csv(out, index=False)
    print(f"\nSaved full results → {out}")


if __name__ == '__main__':
    main()
