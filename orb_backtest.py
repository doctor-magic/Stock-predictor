"""
ORB (Opening Range Breakout) Backtest
Symbols: PLTR, NVDA, TSLA, META, AMD
Interval: 5m | Period: 60 days
ORB window: 9:30-10:00 (first 6 candles)
Label: close at 15:55+ > ORB High?
Filter: RVOL of ORB window >= 1.5x (vs 10-day avg)
"""

import time
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SYMBOLS       = ['PLTR', 'NVDA', 'TSLA', 'META', 'AMD']
ORB_CANDLES   = 6      # 5-min × 6 = 9:30–9:55 inclusive (covers 9:30 open)
RVOL_MIN      = 1.5
RVOL_WINDOW   = 10     # rolling days for baseline avg
VWAP_DIST_MAX = 1.5    # % above VWAP at breakout — above this = chasing


def _to_ny(idx):
    if idx.tz is None:
        return idx.tz_localize('UTC').tz_convert('America/New_York')
    return idx.tz_convert('America/New_York')


def analyze(sym: str) -> pd.DataFrame | None:
    raw = yf.download(sym, period='60d', interval='5m', progress=False, auto_adjust=True)
    if raw.empty:
        print(f"{sym}: no data from yfinance")
        return None

    # Flatten MultiIndex columns (yfinance >= 0.2 returns (field, ticker) tuples)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = _to_ny(raw.index)

    trading_days = sorted(set(raw.index.date))
    rows = []

    for day in trading_days:
        day_df = raw[raw.index.date == day]
        day_df = day_df.between_time('09:30', '16:00')
        if len(day_df) < ORB_CANDLES + 2:
            continue

        orb      = day_df.iloc[:ORB_CANDLES]
        orb_high = float(orb['High'].max())
        orb_low  = float(orb['Low'].min())
        orb_vol  = float(orb['Volume'].sum())

        # VWAP at ORB end (cumulative volume-weighted price)
        cum_vol   = float(day_df['Volume'].cumsum().iloc[ORB_CANDLES - 1])
        cum_pv    = float((day_df['Close'] * day_df['Volume']).cumsum().iloc[ORB_CANDLES - 1])
        vwap_orb  = cum_pv / cum_vol if cum_vol else None

        # Breakout candle: first candle AFTER ORB that closes above ORB High
        post_orb  = day_df.iloc[ORB_CANDLES:]
        bo_mask   = post_orb['Close'] > orb_high
        bo_time   = str(post_orb.index[bo_mask.values][0].time()) if bo_mask.any() else None
        bo_price  = float(post_orb['Close'][bo_mask].iloc[0]) if bo_mask.any() else None

        # Close price: last candle at or after 15:50
        close_cands = day_df.between_time('15:50', '16:00')
        if close_cands.empty:
            continue
        close_px  = float(close_cands['Close'].iloc[-1])

        # Distance of breakout entry from VWAP (positive = above VWAP = bullish)
        vwap_dist_pct = ((bo_price - vwap_orb) / vwap_orb * 100) if (bo_price and vwap_orb) else None

        # Return from breakout entry to close
        entry_to_close = ((close_px - bo_price) / bo_price * 100) if bo_price else None

        # MAE: worst drawdown from entry to close (min low after breakout candle)
        if bo_mask.any():
            bo_idx   = post_orb.index[bo_mask.values][0]
            after_bo = day_df[day_df.index >= bo_idx]
            mae_low  = float(after_bo['Low'].min())
            mae_pct  = (mae_low - bo_price) / bo_price * 100 if bo_price else None
        else:
            mae_pct = None

        rows.append({
            'date':           day,
            'orb_high':       orb_high,
            'orb_low':        orb_low,
            'orb_vol':        orb_vol,
            'vwap_orb':       vwap_orb,
            'broke_out':      bo_mask.any(),
            'bo_time':        bo_time,
            'bo_price':       bo_price,
            'vwap_dist_pct':  vwap_dist_pct,
            'close':          close_px,
            'entry_to_close': entry_to_close,
            'mae_pct':        mae_pct,
        })

    if not rows:
        print(f"{sym}: no complete trading days")
        return None

    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    # RVOL: today's ORB vol vs rolling 10-day avg of past days
    df['avg_orb_vol_10d'] = df['orb_vol'].shift(1).rolling(RVOL_WINDOW).mean()
    df['rvol']            = df['orb_vol'] / df['avg_orb_vol_10d']
    df = df.dropna(subset=['rvol']).reset_index(drop=True)

    return df


def _fmt(val, fmt, suffix='', fallback='   -  '):
    return f"{val:{fmt}}{suffix}" if pd.notna(val) else fallback


def print_report(sym: str, df: pd.DataFrame):
    all_bo    = df[df['broke_out']]
    rvol_days = df[df['rvol'] >= RVOL_MIN]
    rvol_bo   = rvol_days[rvol_days['broke_out']]

    # VWAP filter: only breakout days where distance from VWAP < threshold
    vwap_bo   = rvol_bo[rvol_bo['vwap_dist_pct'].notna() &
                         (rvol_bo['vwap_dist_pct'] < VWAP_DIST_MAX)]

    wr_all    = len(all_bo)  / len(df)        * 100 if len(df)        else 0
    wr_rvol   = len(rvol_bo) / len(rvol_days) * 100 if len(rvol_days) else 0
    # WR = fraction of VWAP-filtered trades where close > entry (entry_to_close > 0)
    wr_vwap   = (vwap_bo['entry_to_close'] > 0).sum() / len(vwap_bo) * 100 if len(vwap_bo) else float('nan')

    avg_ret_all  = all_bo['entry_to_close'].mean()  if not all_bo.empty  else float('nan')
    avg_ret_rvol = rvol_bo['entry_to_close'].mean() if not rvol_bo.empty else float('nan')
    avg_ret_vwap = vwap_bo['entry_to_close'].mean() if not vwap_bo.empty else float('nan')

    print(f"\n{'═'*62}")
    print(f"  {sym}")
    print(f"{'═'*62}")
    print(f"  Days in sample:                    {len(df)}")
    print(f"  ORB breakout rate (all):           {wr_all:.1f}%   avg ret {avg_ret_all:+.2f}%")
    print(f"  High-RVOL days (≥{RVOL_MIN}x):          {len(rvol_days)}  /  breakouts {len(rvol_bo)}")
    print(f"  ORB breakout rate (RVOL):          {wr_rvol:.1f}%   avg ret {avg_ret_rvol:+.2f}%")
    print(f"  + VWAP filter (<{VWAP_DIST_MAX}%):          {len(vwap_bo)} trades")
    if not vwap_bo.empty:
        print(f"  ORB breakout rate (RVOL+VWAP):     {wr_vwap:.1f}%   avg ret {avg_ret_vwap:+.2f}%")
    else:
        print(f"  ORB breakout rate (RVOL+VWAP):     no trades passed filter")

    # Detail table — RVOL days, flag which ones passed VWAP filter
    if not rvol_days.empty:
        print(f"\n  {'Date':<12} {'RVOL':>5} {'ORB Hi':>8} {'BO Time':<8} {'BO Px':>8}"
              f" {'VWAP D':>7} {'MAE':>7} {'->Close':>8} {'F':>2}")
        print(f"  {'-'*76}")
        for _, r in rvol_days.iterrows():
            status = 'V' if r['broke_out'] else 'X'
            vd_val = r['vwap_dist_pct']
            passed = pd.notna(vd_val) and vd_val < VWAP_DIST_MAX and r['broke_out']
            flag   = '*' if passed else ' '
            bt     = r['bo_time'][:5] if isinstance(r['bo_time'], str) else '  -  '
            bp     = _fmt(r['bo_price'],   '.2f', suffix='', fallback='   -  ')
            bp     = f"${bp}" if pd.notna(r['bo_price']) else '   -  '
            vd     = _fmt(vd_val,           '+.1f', '%')
            mae    = _fmt(r['mae_pct'],     '+.1f', '%')
            ec     = _fmt(r['entry_to_close'], '+.2f', '%')
            print(f"  {str(r['date']):<12} {r['rvol']:>4.1f}x {r['orb_high']:>8.2f} {bt:<8}"
                  f" {bp:>8} {vd:>7} {mae:>7} {ec:>8} {status}{flag}")
    print(f"  (* = passed RVOL + VWAP filter)")


summary_rows = []

for sym in SYMBOLS:
    df = analyze(sym)
    if df is None:
        continue
    print_report(sym, df)
    time.sleep(1)

    rvol_days = df[df['rvol'] >= RVOL_MIN]
    rvol_bo   = rvol_days[rvol_days['broke_out']]
    vwap_bo   = rvol_bo[rvol_bo['vwap_dist_pct'].notna() &
                         (rvol_bo['vwap_dist_pct'] < VWAP_DIST_MAX)]
    summary_rows.append({
        'sym':       sym,
        'n':         len(df),
        'wr_all':    len(df[df['broke_out']]) / len(df) * 100 if len(df) else 0,
        'rvol_n':    len(rvol_days),
        'wr_rvol':   len(rvol_bo) / len(rvol_days) * 100 if len(rvol_days) else 0,
        'ret_rvol':  rvol_bo['entry_to_close'].mean() if not rvol_bo.empty else float('nan'),
        'vwap_n':    len(vwap_bo),
        'ret_vwap':  vwap_bo['entry_to_close'].mean() if not vwap_bo.empty else float('nan'),
    })

print(f"\n\n{'═'*72}")
print("  SUMMARY")
print(f"{'═'*72}")
print(f"  {'Sym':<6} {'WR-All':>7} {'RVOL N':>7} {'WR-RVOL':>8} {'Ret-RVOL':>9}"
      f" {'VWAP N':>7} {'Ret-VWAP':>9}")
print(f"  {'-'*64}")
for r in summary_rows:
    rv = f"{r['ret_vwap']:>+8.2f}%" if pd.notna(r['ret_vwap']) else "      n/a"
    print(f"  {r['sym']:<6} {r['wr_all']:>6.1f}% {r['rvol_n']:>7} {r['wr_rvol']:>7.1f}%"
          f" {r['ret_rvol']:>+8.2f}% {r['vwap_n']:>7} {rv}")
