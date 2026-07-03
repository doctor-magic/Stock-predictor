#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_data.env")
if os.path.exists(_env_path):
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import core_logic, db
import urllib.request as _req
import json as _json
from datetime import datetime


MARKETS = ["sp500", "nasdaq100", "tase"]


def _send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = _json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = _req.Request(url, data=payload, headers={"Content-Type": "application/json"})
        _req.urlopen(req, timeout=5)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def _detect_falling_wedge(h_all, l_all, c_all, v_all) -> dict:
    """Detect a falling wedge on daily OHLCV arrays.
    Tries windows 30 → 45 → 60 days; returns first valid match.
    """
    try:
        from scipy.signal import find_peaks
        import numpy as np

        h_all = np.array(h_all, dtype=float).flatten()
        l_all = np.array(l_all, dtype=float).flatten()
        c_all = np.array(c_all, dtype=float).flatten()
        v_all = np.array(v_all, dtype=float).flatten()
        total = len(c_all)

        min_compression = {30: 0.40, 45: 0.45, 60: 0.50}

        for lookback in (30, 45, 60):
            n = min(lookback, total)
            if n < 25:
                continue

            h = h_all[-n:]; l = l_all[-n:]
            c = c_all[-n:]; v = v_all[-n:]
            x = np.arange(n, dtype=float)

            price_range = h.max() - l.min()
            if price_range <= 0:
                continue

            prominence = price_range * 0.03
            peaks,   _ = find_peaks( h, distance=3, prominence=prominence)
            troughs, _ = find_peaks(-l, distance=3, prominence=prominence)

            if len(peaks) < 3 or len(troughs) < 3:
                continue

            us, ui = np.polyfit(peaks,   h[peaks],  1)
            ls, li = np.polyfit(troughs, l[troughs], 1)

            if us >= -0.001 or ls >= -0.001:
                continue
            if us >= ls:
                continue

            uf = us * peaks   + ui
            lf = ls * troughs + li
            ut = int(np.sum(np.abs(h[peaks]   - uf) / np.abs(uf) < 0.04))
            lt = int(np.sum(np.abs(l[troughs] - lf) / np.abs(lf) < 0.04))
            if ut < 3 or lt < 3:
                continue

            ws = ui - li
            we = (us * (n-1) + ui) - (ls * (n-1) + li)
            if ws <= 0 or we < 0:
                continue

            compression = 1.0 - (we / ws)
            if compression < min_compression[lookback]:
                continue

            half = n // 2
            vol_declining = float(v[half:].mean()) < float(v[:half].mean()) * 0.85

            upper_now = float(us * (n-1) + ui)
            breakout  = float(c[-1]) > upper_now

            fresh_breakout = False
            if breakout:
                ul = us * x + ui
                for i in range(max(0, n - 3), n):
                    prev_below = (i == 0) or (c[i-1] <= ul[i-1])
                    if c[i] > ul[i] and prev_below:
                        fresh_breakout = True
                        break

            return {
                "detected":       True,
                "breakout":       breakout,
                "fresh_breakout": fresh_breakout,
                "compression":    round(float(compression), 2),
                "vol_declining":  vol_declining,
                "lookback_used":  n,
                "upper_touches":  ut,
                "lower_touches":  lt,
            }

        return {}
    except Exception:
        return {}


def _get_earnings_dates(symbols: list) -> dict:
    """Fetch next earnings date for each symbol. Returns {symbol: 'YYYY-MM-DD' or None}."""
    import yfinance as yf
    from datetime import date
    today = date.today().isoformat()
    dates = {}
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
            if cal is None:
                continue
            # calendar is a dict: {'Earnings Date': [Timestamp, ...], ...}
            ed = cal.get("Earnings Date") or cal.get("earningsDate")
            if ed is None:
                continue
            if hasattr(ed, "tolist"):
                ed = ed.tolist()
            if not isinstance(ed, list):
                ed = [ed]
            for d in ed:
                s = str(d)[:10]
                if s >= today:
                    dates[sym] = s
                    break
        except Exception:
            pass
    return dates


def _scan_wedge_patterns(symbols: list) -> list:
    """Download 4mo daily OHLCV for all symbols and detect wedge patterns.
    Returns list of dicts sorted: fresh breakouts first, then forming by compression.
    """
    import yfinance as yf

    results = []
    chunk_size = 100
    print(f"  Wedge scan: {len(symbols)} symbols in {-(-len(symbols)//chunk_size)} chunks...", flush=True)

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            df = yf.download(
                chunk, period="4mo", interval="1d",
                group_by="ticker", progress=False, auto_adjust=True,
                threads=True,
            )
            for sym in chunk:
                try:
                    if len(chunk) == 1:
                        h = df["High"].dropna().values.flatten()
                        l = df["Low"].dropna().values.flatten()
                        c = df["Close"].dropna().values.flatten()
                        v = df["Volume"].dropna().values.flatten()
                    else:
                        h = df[sym]["High"].dropna().values.flatten()
                        l = df[sym]["Low"].dropna().values.flatten()
                        c = df[sym]["Close"].dropna().values.flatten()
                        v = df[sym]["Volume"].dropna().values.flatten()

                    if len(c) < 25:
                        continue

                    w = _detect_falling_wedge(h, l, c, v)
                    if w.get("detected"):
                        # FORMING patterns without declining volume are weak — skip
                        if not w["breakout"] and not w["vol_declining"]:
                            continue
                        ret_30d = round((float(c[-1]) / float(c[-30]) - 1) * 100, 1) if len(c) >= 30 else None
                        ret_4m  = round((float(c[-1]) / float(c[0])   - 1) * 100, 1) if len(c) >= 2  else None
                        high_risk = ret_4m is not None and ret_4m < -30
                        results.append({
                            "symbol":        sym,
                            "close":         round(float(c[-1]), 2),
                            "breakout":      w["breakout"],
                            "fresh_breakout": w["fresh_breakout"],
                            "compression":   w["compression"],
                            "vol_declining": w["vol_declining"],
                            "lookback":      w["lookback_used"],
                            "ret_30d":       ret_30d,
                            "ret_4m":        ret_4m,
                            "high_risk":     high_risk,
                            "upper_touches": w.get("upper_touches", 0),
                            "lower_touches": w.get("lower_touches", 0),
                        })
                except Exception:
                    pass
        except Exception as e:
            print(f"  Wedge chunk {i//chunk_size+1} error: {e}")

    # Sort: fresh breakouts → breakouts → forming (by compression desc)
    def _rank(r):
        if r["fresh_breakout"]: return (0, -r["compression"])
        if r["breakout"]:       return (1, -r["compression"])
        return                         (2, -r["compression"])

    results.sort(key=_rank)

    # Enrich with earnings dates
    if results:
        syms = [r["symbol"] for r in results]
        print(f"  Fetching earnings dates for {len(syms)} symbols...", flush=True)
        earn = _get_earnings_dates(syms)
        for r in results:
            r["earnings_date"] = earn.get(r["symbol"])

    return results


def _format_wedge_telegram(wedge_results: list, date_str: str) -> str:
    fresh    = [r for r in wedge_results if r["fresh_breakout"]]
    broken   = [r for r in wedge_results if r["breakout"] and not r["fresh_breakout"]]
    forming  = [r for r in wedge_results if not r["breakout"]]

    lines = [f"📐 Wedge Pattern Scan — {date_str}"]

    if fresh:
        lines.append(f"\n▲ FRESH BREAKOUT ({len(fresh)}):")
        for r in fresh[:8]:
            vd = " vol↓" if r["vol_declining"] else ""
            lines.append(f"  {r['symbol']} ${r['close']}  comp={int(r['compression']*100)}%{vd}")

    if broken:
        lines.append(f"\n↑ BROKEN ({len(broken)}):")
        for r in broken[:5]:
            vd = " vol↓" if r["vol_declining"] else ""
            lines.append(f"  {r['symbol']} ${r['close']}  comp={int(r['compression']*100)}%{vd}")

    if forming:
        lines.append(f"\n◇ FORMING — Watchlist ({len(forming)}):")
        for r in forming[:15]:
            vd = " vol↓" if r["vol_declining"] else ""
            lines.append(f"  {r['symbol']} ${r['close']}  {r['lookback']}d  comp={int(r['compression']*100)}%{vd}")

    if not fresh and not broken and not forming:
        lines.append("אין פטרנים פעילים היום.")

    return "\n".join(lines)


def main():
    db.init_db()
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting pre-scan...")

    summary_lines = []
    any_failure = False
    all_us_symbols = set()

    for market_id in MARKETS:
        try:
            print(f"  Scanning {market_id}...", flush=True)
            results = core_logic.run_market_scan(market_id)
            db.save_scan_results(market_id, results)
            count = len(results)
            print(f"  Done: {market_id} — {count} results saved")

            # Collect US symbols for wedge scan (skip TASE)
            if market_id != "tase":
                for r in results:
                    if r.get("symbol"):
                        all_us_symbols.add(r["symbol"])

            if count == 0:
                any_failure = True
                summary_lines.append(f"  ⚠️ {market_id.upper()}: 0 תוצאות")
            else:
                full_buys   = [r for r in results if r.get("signal") == "BUY" and r.get("confidence", 0) >= 0.70]
                almost_buys = [r for r in results if r.get("signal") == "BUY" and r.get("confidence", 0) < 0.70]
                parts = []
                if full_buys:
                    syms = " ".join(r["symbol"] for r in full_buys[:5])
                    parts.append(f"BUY:{len(full_buys)} ({syms})")
                else:
                    parts.append("BUY:0")
                if almost_buys:
                    syms = " ".join(r["symbol"] for r in almost_buys[:3])
                    parts.append(f"⚠ Almost:{len(almost_buys)} ({syms})")
                summary_lines.append(f"  ✅ {market_id.upper()}: " + " | ".join(parts))

        except Exception as e:
            any_failure = True
            print(f"  ERROR {market_id}: {e}")
            summary_lines.append(f"  ❌ {market_id.upper()}: שגיאה — {e}")

    end_time = datetime.now()
    duration = int((end_time - start_time).total_seconds() / 60)
    print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Pre-scan complete.")

    status = "⚠️ סריקה הושלמה עם כשלים" if any_failure else "✅ סריקה הושלמה בהצלחה"
    msg = f"[Stock Predictor] {status}\n" + "\n".join(summary_lines) + f"\n  ⏱ {duration} דקות"
    _send_telegram(msg)

    # Wedge pattern scan — separate Telegram message
    if all_us_symbols:
        try:
            print(f"  Starting wedge scan on {len(all_us_symbols)} US symbols...", flush=True)
            wedge_start = datetime.now()
            wedge_results = _scan_wedge_patterns(sorted(all_us_symbols))
            wedge_dur = int((datetime.now() - wedge_start).total_seconds())
            print(f"  Wedge scan done in {wedge_dur}s — {len(wedge_results)} patterns found")
            date_str = start_time.strftime("%Y-%m-%d")
            wedge_msg = _format_wedge_telegram(wedge_results, date_str)
            _send_telegram(wedge_msg)

            wedge_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wedge_cache.json")
            with open(wedge_cache_path, "w") as _wf:
                _json.dump({"scan_date": date_str, "scan_ts": wedge_start.timestamp(), "results": wedge_results}, _wf)
        except Exception as e:
            print(f"  Wedge scan ERROR: {e}")
            _send_telegram(f"[Stock Predictor] ⚠️ Wedge scan failed: {e}")


if __name__ == "__main__":
    main()
