#!/usr/bin/env python3
"""Ops watchdog — read-only daily health digest over Telegram.

Checks (each isolated; a check crash reports red, never kills the run):
  1. API service answering on :8000
  2. Resolver cron completed within the last 26h + last run counters
  3. Yesterday's setup_log collection (trading days only) + lev_sent fill
  4. Disk usage
  5. Available memory
  6. Cron output freshness — the three daily crons nothing else watches
     (tracker.db / intraday_cache.db / wedge_cache.json). Silent-cron-death
     guard; recreates the detector for the Jun 6-14 live_tracker outage.

Sends ONE message every day (green heartbeat or red alert) on the existing
pre_scan Telegram channel. Alert-only by design: this script never fixes,
restarts, or writes anything.
"""
import sys, os, re, json, sqlite3, shutil
import urllib.request, urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

_env_path = os.path.join(BASE, "api_data.env")
if os.path.exists(_env_path):
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

checks = []  # (ok, label, detail)


def add(ok, label, detail=""):
    checks.append((bool(ok), label, detail))


# --- 1. service alive -------------------------------------------------------
try:
    req = urllib.request.Request("http://localhost:8000/", method="GET")
    try:
        urllib.request.urlopen(req, timeout=10)
        add(True, "api", "HTTP up")
    except urllib.error.HTTPError as e:
        # 401/404 etc. still prove the service is answering
        add(True, "api", f"HTTP up ({e.code})")
except Exception as e:
    add(False, "api", f"NOT ANSWERING: {e}")

# --- 2. resolver ------------------------------------------------------------
try:
    with open(os.path.join(BASE, "resolve_setups.log")) as f:
        tail = f.readlines()[-12:]
    done_ts = None
    counters = ""
    for line in tail:
        m = re.search(r"\[resolve_setups\] done (\S+)", line)
        if m:
            done_ts = datetime.fromisoformat(m.group(1))
        m = re.search(r"candidates=\d+ attempted=\d+ resolved=\d+ stale=\d+ failed=\d+", line)
        if m:
            counters = m.group(0)
    if done_ts is None:
        add(False, "resolver", "no 'done' line found in log tail")
    else:
        age_h = (datetime.now(ZoneInfo("UTC")) - done_ts).total_seconds() / 3600
        failed = int(re.search(r"failed=(\d+)", counters).group(1)) if counters else 0
        fresh = age_h <= 26
        add(fresh and failed == 0, "resolver",
            f"last run {age_h:.0f}h ago, {counters or 'no counters'}")
except Exception as e:
    add(False, "resolver", f"check error: {e}")

# --- 3. yesterday's collection (trading days only) ---------------------------
try:
    import market_calendar
    yday = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()
    try:
        was_session = market_calendar.is_us_market_session(yday)
    except TypeError:
        was_session = market_calendar.is_us_market_session(yday.isoformat())
    if not was_session:
        add(True, "collection", f"{yday} was not a session — skip")
    else:
        con = sqlite3.connect(os.path.join(BASE, "setup_log.db"), timeout=30)
        n, lev = con.execute(
            "SELECT COUNT(*), SUM(lev_sent_semis IS NOT NULL) FROM setup_log WHERE date=?",
            (yday.isoformat(),)).fetchone()
        con.close()
        lev = lev or 0
        add(n > 0 and lev == n, "collection", f"{yday}: rows={n} lev_fill={lev}/{n}")
except Exception as e:
    add(False, "collection", f"check error: {e}")

# --- 4. disk -----------------------------------------------------------------
try:
    du = shutil.disk_usage("/")
    pct = du.used / du.total * 100
    add(pct < 90, "disk", f"{pct:.0f}% used")
except Exception as e:
    add(False, "disk", f"check error: {e}")

# --- 5. memory ---------------------------------------------------------------
try:
    with open("/proc/meminfo") as f:
        info = f.read()
    avail_mb = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) // 1024
    add(avail_mb > 150, "memory", f"{avail_mb}MB available")
except Exception as e:
    add(False, "memory", f"check error: {e}")

# --- 6. cron output freshness (silent-cron-death guard) ----------------------
# The three daily crons nothing else watches. Each output's newest date must be
# >= the last completed US session — market_calendar handles weekends/holidays,
# so 0% market/calendar false-positives. scanner_cache.db is deliberately
# EXCLUDED: multiple writers at variable cadence make its freshness ambiguous.
try:
    import market_calendar
    et_yday = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()
    last_session = None
    probe = et_yday
    for _ in range(10):
        try:
            sess = market_calendar.is_us_market_session(probe)
        except TypeError:
            sess = market_calendar.is_us_market_session(probe.isoformat())
        if sess:
            last_session = probe.isoformat()
            break
        probe -= timedelta(days=1)

    # tracker.db — live_tracker 20:05 Mon-Fri (the Jun 6-14 outage detector)
    try:
        con = sqlite3.connect(os.path.join(BASE, "tracker.db"), timeout=30)
        mx = con.execute("SELECT MAX(substr(date_logged,1,10)) FROM signals").fetchone()[0]
        con.close()
        add(last_session is not None and mx is not None and mx >= last_session,
            "tracker", f"newest={mx} (need >= last session {last_session})")
    except Exception as e:
        add(False, "tracker", f"check error: {e}")

    # intraday_cache.db — fetch_intraday 20:30 Mon-Fri
    try:
        con = sqlite3.connect(os.path.join(BASE, "intraday_cache.db"), timeout=30)
        mx = con.execute("SELECT MAX(date) FROM intraday_bars").fetchone()[0]
        con.close()
        add(last_session is not None and mx is not None and mx >= last_session,
            "intraday", f"newest bar={mx} (need >= last session {last_session})")
    except Exception as e:
        add(False, "intraday", f"check error: {e}")

    # wedge_cache.json — pre_scan 05:00 DAILY; age ceiling matches resolver's 26h
    try:
        import time as _time
        with open(os.path.join(BASE, "wedge_cache.json")) as f:
            wc = json.load(f)
        age_h = (_time.time() - float(wc.get("scan_ts", 0))) / 3600
        add(age_h <= 26, "wedge", f"scan_date={wc.get('scan_date')} age={age_h:.0f}h")
    except Exception as e:
        add(False, "wedge", f"check error: {e}")
except Exception as e:
    add(False, "cron-freshness", f"check error: {e}")

# --- report ------------------------------------------------------------------
all_ok = all(ok for ok, _, _ in checks)
today = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%d/%m %H:%M")
if all_ok:
    lines = [f"\U0001F7E2 WATCHDOG {today} — all green"]
    lines += [f"✓ {label}: {detail}" for ok, label, detail in checks]
else:
    lines = [f"\U0001F534 WATCHDOG ALERT {today}"]
    lines += [("✓" if ok else "✗") + f" {label}: {detail}"
              for ok, label, detail in checks]
msg = "\n".join(lines)
print(msg)

token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")
if token and chat_id:
    try:
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15)
        print("[watchdog] telegram sent")
    except Exception as e:
        print(f"[watchdog] telegram send failed: {e}")
else:
    print("[watchdog] no telegram creds — printed only")
