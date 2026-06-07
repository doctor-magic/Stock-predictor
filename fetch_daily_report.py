import os
import sys
import asyncio
import traceback
import subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import anthropic
import httpx
from datetime import datetime, timezone
from telethon import TelegramClient
from dotenv import load_dotenv
from pathlib import Path

base_path = Path.home() / "Desktop" / "daily_reports"
load_dotenv(dotenv_path=base_path / '.env')

API_ID     = int(os.getenv("TG_API_ID"))
API_HASH   = os.getenv("TG_API_HASH")
PHONE      = os.getenv("TG_PHONE")
CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")

SESSION_FILE = str(Path.home() / "tg_session" / "session_v3")
TARGETS      = ["capitalflow_co_il", "claltelegram", "altshuler_shaham", "msh_telegram2023"]

now           = datetime.now(timezone.utc)
search_cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

claude = anthropic.Anthropic(api_key=CLAUDE_KEY)


def send_telegram_alert(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing — cannot send alert.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        httpx.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")


def build_report(messages: list[dict]) -> str:
    numbered = ""
    for i, m in enumerate(messages, 1):
        numbered += f"\n[{i}] {m['date']} [{m['channel']}]  {m['text'][:1200]}\n"

    prompt = (
        "אתה כתב כלכלי בכיר של TheMarker.\n\n"
        "קיבלת הודעות טלגרם מערוצים פיננסיים ישראליים מהיום.\n"
        "כתוב סקירה יומית בעברית בפורמט הבא בדיוק — אל תשנה שמות כותרות:\n\n"
        "# סקירה יומית - [תאריך היום]\n\n"
        "## סיכום כללי\n"
        "2-3 משפטים. מספרים ספציפיים בלבד (%, שקלים, נקודות מדד). ללא הכללות.\n\n"
        "## אותות חיוביים (BUY)\n"
        "רשימה ממוספרת. לכל חברה: שם + הקשר ספציפי (% שינוי / אירוע) + סיבה (3-5 מילים) + רמת אמינות (★).\n\n"
        "## אותות שליליים (SELL)\n"
        "רשימה ממוספרת, אותו פורמט.\n\n"
        "## ניטרלי / מעקב\n"
        "חברות ללא כיוון ברור.\n\n"
        "## שורה תחתונה\n"
        "משפט אחד. ספציפי. ללא קלישאות.\n\n"
        "---\n"
        "כללי סגנון — חובה:\n"
        "1. משפטים קצרים — מקסימום 20 מילה.\n"
        "2. מספרים ספציפיים: לא 'עלייה חדה' אלא '+3.2%'; לא 'ירידה משמעותית' אלא '-4.1%'.\n"
        "3. אסור להשתמש: 'מצד אחד... מצד שני', 'תופעה המעידה', 'יום המאופיין', "
        "'ביטחון יחסי', 'צמיחה מרשימה', 'רקע מורכב', 'אי-ודאות כלכלית'.\n"
        "4. הסגנון: TheMarker — לא ChatGPT. קצר, ישיר, מבוסס עובדות.\n"
        "5. מונחים: 'סנטימנט חיובי' (לא 'מגמה חיובית'), 'מתיחות גיאופוליטית' "
        "(לא 'חיכוך גיאופוליטי'), 'סקטור ההייטק' (לא 'הטכנולוגיה הגבוהה').\n\n"
        "כתוב רק את הסקירה, בלי הקדמות.\n\n"
        f"ההודעות:\n{numbered}"
    )

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        error_msg = f"❌ fetch_daily_report.py — Claude API error:\n{type(e).__name__}: {e}"
        print(error_msg)
        send_telegram_alert(error_msg)
        raise

    return response.content[0].text.strip()


async def main():
    today_str   = datetime.now().strftime("%d_%m_%Y")
    output_file = base_path / f"stock_recommendations_{today_str}.txt"

    print(f"Fetching messages since {search_cutoff.strftime('%Y-%m-%d %H:%M')} UTC...\n")

    all_messages = []

    async with TelegramClient(SESSION_FILE, API_ID, API_HASH) as client:
        await client.start(phone=PHONE)

        for target in TARGETS:
            try:
                entity = await client.get_entity(target)
                count  = 0
                async for msg in client.iter_messages(entity, limit=200):
                    if msg.date < search_cutoff:
                        break
                    if not msg.text or len(msg.text) < 15:
                        continue
                    all_messages.append({
                        "channel": target,
                        "date":    msg.date.strftime("%Y-%m-%d %H:%M"),
                        "text":    msg.text,
                    })
                    count += 1
                print(f"  {target}: {count} messages")
            except Exception as e:
                print(f"  Error fetching {target}: {e}")

    print(f"\nTotal: {len(all_messages)} messages. Building report with Claude...\n")

    if not all_messages:
        print("No messages found for today.")
        return

    report = build_report(all_messages)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved to: {output_file}")

    ssh_key = str(Path.home() / ".ssh" / "gcp_stock_rsa")
    remote  = "elimaoz99@35.239.74.178:/home/elimaoz99/stock_predictor/"
    result  = subprocess.run(
        ["scp", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", str(output_file), remote],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Uploaded to server: {remote}")
        send_telegram_alert(f"✅ סקירה יומית עלתה לאתר: {output_file.name}")
    else:
        err = result.stderr.strip()
        print(f"SCP failed: {err}")
        send_telegram_alert(f"❌ fetch_daily_report.py — SCP נכשל:\n{err}")

    print("=" * 60)
    print(report)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        err = traceback.format_exc()
        print(err)
        send_telegram_alert(f"❌ fetch_daily_report.py נכשל:\n{err[-800:]}")
