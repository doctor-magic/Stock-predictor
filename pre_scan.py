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


def main():
    db.init_db()
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting pre-scan...")

    summary_lines = []
    any_failure = False

    for market_id in MARKETS:
        try:
            print(f"  Scanning {market_id}...", flush=True)
            results = core_logic.run_market_scan(market_id)
            db.save_scan_results(market_id, results)
            count = len(results)
            print(f"  Done: {market_id} — {count} results saved")

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


if __name__ == "__main__":
    main()
