import yfinance as yf
import json
import urllib.request
import pandas as pd
from datetime import date

# Get Volume Leaders
with urllib.request.urlopen('http://localhost:8000/api/volume-leaders') as resp:
    data = json.loads(resp.read())

results = data['results']
vl_map = {r['symbol']: r for r in results}
symbols = list(vl_map.keys())

# Fetch today's 5m intraday
raw = yf.download(symbols, period='1d', interval='5m', group_by='ticker', progress=False, auto_adjust=True)

today = date.today()
rows = []

for sym in symbols:
    r = vl_map[sym]
    try:
        df = raw[sym] if len(symbols) > 1 else raw
        df = df.dropna(subset=['Close'])
        df = df[df.index.date == today]
        if df.empty:
            continue
        open_px  = float(df['Open'].iloc[0])
        close_px = float(df['Close'].iloc[-1])
        high_px  = float(df['High'].max())
        low_px   = float(df['Low'].min())
        pct      = (close_px - open_px) / open_px * 100
        ml       = r.get('ml_signal') or 'N/A'
        conf     = r.get('ml_confidence')
        conf_s   = f"{conf:.1f}%" if conf else '--'
        mom      = r.get('momentum') or '--'
        rows.append((sym, ml, conf_s, mom, open_px, close_px, high_px, low_px, pct))
    except Exception as e:
        pass

# Sort: BUY first, then by pct desc
def sort_key(row):
    order = {'BUY': 0, 'SELL': 1, 'HOLD': 2, 'N/A': 3}
    return (order.get(row[1], 9), -row[8])
rows.sort(key=sort_key)

print(f"\n{'Symbol':<7} {'ML':<5} {'Conf':<7} {'Momentum':<16} {'Open':>7} {'Close':>7} {'High':>7} {'Low':>7} {'Change':>8}")
print('-' * 82)
for sym, ml, conf_s, mom, op, cl, hi, lo, pct in rows:
    marker = ' +' if pct > 0 else (' -' if pct < 0 else '')
    print(f"{sym:<7} {ml:<5} {conf_s:<7} {mom:<16} {op:>7.2f} {cl:>7.2f} {hi:>7.2f} {lo:>7.2f} {pct:>+7.2f}%{marker}")

# Summary
buy_rows = [r for r in rows if r[1] == 'BUY']
surge    = [r for r in rows if r[3] == 'SURGING']
all_pos  = [r for r in rows if r[8] > 0]
all_neg  = [r for r in rows if r[8] < 0]

print("\n-- Summary --")
print(f"BUY signals:     {len(buy_rows)} stocks  {[r[0] for r in buy_rows]}")
print(f"SURGING signals: {len(surge)} stocks  {[r[0] for r in surge]}")
print(f"Positive today:  {len(all_pos)}/{len(rows)}")
print(f"Negative today:  {len(all_neg)}/{len(rows)}")
if rows:
    avg = sum(r[8] for r in rows) / len(rows)
    print(f"Avg change all:  {avg:+.2f}%")
if buy_rows:
    avg_b = sum(r[8] for r in buy_rows) / len(buy_rows)
    print(f"Avg change BUY:  {avg_b:+.2f}%")
if surge:
    avg_s = sum(r[8] for r in surge) / len(surge)
    print(f"Avg SURGING chg: {avg_s:+.2f}%")
