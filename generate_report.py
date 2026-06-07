import os
import sys
import csv
import anthropic
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

base_path = Path(__file__).parent.absolute()
load_dotenv(dotenv_path=Path("/home/elimaoz99/stock_app/.env"))

CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_KEY)

today_str = datetime.now().strftime("%d_%m_%Y")
csv_file = base_path / f"raw_messages_{datetime.now().strftime('%d-%m-%Y')}.csv"

if not csv_file.exists():
    print(f"CSV not found: {csv_file}")
    sys.exit(1)

rows = []
with open(csv_file, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"Loaded {len(rows)} messages from CSV")

messages_text = ""
for i, row in enumerate(rows, 1):
    ch  = row.get("ערוץ", "")
    dt  = row.get("תאריך ושעה", "")
    txt = row.get("הודעה", "").strip().replace("\n", " ")[:2000]
    messages_text += f"[{i}] {dt} | {ch}\n{txt}\n\n"

SYSTEM_PROMPT = (
    "אתה כתב כלכלי בכיר של TheMarker.\n\n"
    "כללי סגנון — חובה:\n"
    "1. משפטים קצרים — מקסימום 20 מילה.\n"
    "2. מספרים ספציפיים: לא 'עלייה חדה' אלא '+3.2%'; לא 'ירידה משמעותית' אלא '-4.1%'.\n"
    "3. אסור: 'מצד אחד... מצד שני', 'תופעה המעידה', 'יום המאופיין', 'ביטחון יחסי', "
    "'אמירה חיובית', 'צמיחה מרשימה', 'רקע מורכב', 'אי-ודאות כלכלית'.\n"
    "4. הסגנון: TheMarker — לא ChatGPT. קצר, ישיר, מבוסס עובדות.\n"
    "5. אל תמציא נתונים שלא מופיעים בהודעות.\n"
    "6. כתוב רק את הסקירה, בלי הקדמות.\n"
)

user_content = (
    f"קיבלת {len(rows)} הודעות מערוצי טלגרם פיננסיים מ-24 השעות האחרונות.\n"
    "כתוב סקירה יומית בעברית בפורמט הבא בדייק — אל תשנה שמות כותרות:\n\n"
    f"# סקירת שוק ההון — {datetime.now().strftime('%d/%m/%Y')}\n\n"
    "## סיכום כללי\n"
    "2-3 משפטים. מספרים ספציפיים בלבד (%, שקלים/דולרים, נקודות מדד). ללא הכללות.\n\n"
    "## שוק ישראלי — אותות חיוביים\n"
    "רשימה ממוספרת. לכל מניה: שם + הקשר ספציפי (% שינוי / אירוע) + סיבה קצרה (3-5 מילים).\n\n"
    "## שוק ישראלי — אותות שליליים\n"
    "רשימה ממוספרת, אותו פורמט.\n\n"
    "## שוק האמריקאי\n"
    "S&P 500, נאסד\"ק, מניות מובילות — % שינוי ואירועים ספציפיים מהמקורות. "
    "אם אין מידע על ארהב בהודעות — כתוב 'לא נמצא מידע בערוצים'.\n\n"
    "## נושאים חמים ומגמות\n"
    "נושאים שחוזרים בערוצים שונים — מאקרו, סקטורים, אירועים גיאופוליטיים.\n\n"
    "## המלצות ברמת אמינות גבוהה\n"
    "רק מניות/נכסים עם נתונים קשים: דוחות, ידיעות עם מקור. לא שמועות.\n\n"
    "## שורה תחתונה\n"
    "משפט אחד. ספציפי. ללא קלישאות.\n\n"
    "---\n"
    f"הודעות:\n{messages_text[:40000]}"
)

print("Sending to Claude...")
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4000,
    system=[{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }],
    messages=[{"role": "user", "content": user_content}],
)

report = response.content[0].text.strip()
output_file = base_path / f"stock_recommendations_{today_str}_Evening.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Report saved: {output_file}")
print("\n--- PREVIEW ---")
print(report[:500])
