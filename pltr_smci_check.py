import yfinance as yf
from datetime import date

symbols = ['PLTR', 'SMCI']

for sym in symbols:
    tk = yf.Ticker(sym)
    df = tk.history(period='1d', interval='1m')
    df = df[df.index.date == date.today()]
    if df.empty:
        print(f"{sym}: no data")
        continue

    open_px  = float(df['Open'].iloc[0])
    close_px = float(df['Close'].iloc[-1])
    high_px  = float(df['High'].max())
    low_px   = float(df['Low'].min())
    high_time = df['High'].idxmax().strftime('%H:%M')
    low_time  = df['Low'].idxmin().strftime('%H:%M')
    pct_oc   = (close_px - open_px) / open_px * 100

    # 9:45 candle
    t15_candidates = [i for i, ts in enumerate(df.index) if ts.hour == 9 and ts.minute == 45]
    t15_px = float(df['Close'].iloc[t15_candidates[0]]) if t15_candidates else None

    print("")
    print("=" * 46)
    print("  " + sym)
    print("=" * 46)
    print(f"  פתיחה  (9:30):         ${open_px:.2f}")
    if t15_px:
        pct_to_t15 = (t15_px - open_px) / open_px * 100
        pct_t15_close = (close_px - t15_px) / t15_px * 100
        print(f"  9:45   (BUY signal):   ${t15_px:.2f}   ({pct_to_t15:+.2f}% from open)")
        print(f"  כניסה 9:45 -> סגירה:  {pct_t15_close:+.2f}%")
    print(f"  שיא יומי:              ${high_px:.2f}   ({high_time})")
    print(f"  שפל יומי:              ${low_px:.2f}   ({low_time})")
    print(f"  סגירה:                 ${close_px:.2f}")
    print(f"  פתיחה -> סגירה:        {pct_oc:+.2f}%")

    print("")
    print("  Timeline:")
    for hour in [10, 11, 12, 13, 14, 15, 16]:
        rows_h = [i for i, ts in enumerate(df.index) if ts.hour == hour and ts.minute == 0]
        if rows_h:
            px = float(df['Close'].iloc[rows_h[0]])
            chg = (px - open_px) / open_px * 100
            sign = "+" if chg >= 0 else ""
            bar = "#" * min(int(abs(chg) * 2), 30)
            print(f"    {hour}:00  ${px:>7.2f}  {sign}{chg:.2f}%  {bar}")
