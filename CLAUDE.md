# Stock Predictor – CLAUDE.md

## File
`C:\Users\elima\OneDrive\Desktop\azrg_predictor_ui.py` (~660 lines)

## Architecture
- Line ~262: CSS block (dark theme, ends with `</style>`)
- Line ~390: `st.tabs` — tab1=מניה בודדת, tab2=סורק, tab3=המלצות יומיות
- Line ~480: `show_results()` — custom HTML table with Yahoo Finance links
- Line ~568: TAB 3 — daily recommendations (custom HTML sections, no st.expander)

## Critical rules
- **Hebrew strings cannot be matched by Edit tool** — use a Python patch script instead:
  open file UTF-8, `str.replace()` with unicode escapes (`\u05de\u05d7\u05d9\u05e8` etc.), write back
- Syntax check: `py -c "import py_compile; py_compile.compile('azrg_predictor_ui.py', doraise=True)"`
- Deploy: `cd "C:\Users\elima\OneDrive\Desktop" && git add azrg_predictor_ui.py && git commit -m "msg" && git push`

## Stack
Streamlit + pandas + yfinance | GitHub: `doctor-magic/Stock-predictor` | branch: `main`
