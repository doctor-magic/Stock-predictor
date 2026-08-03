#!/usr/bin/env python3
"""Verify every IRON guardrail still exists verbatim after a CLAUDE.md trim.

Reads _guardrail_inventory.md, extracts every line classified IRON, and requires
each quote to appear as a contiguous substring of CLAUDE.md — or of any skill file
under .claude/skills/ once stage 4 has moved WORKFLOW content there.

Exit 0 = all present. Exit 1 = at least one guardrail vanished; the trim step that
caused it must be rolled back.

    python3 check_guardrails.py            # check
    python3 check_guardrails.py --verbose  # list every guardrail as it is checked
"""
import re
import sys
from pathlib import Path

# A guardrail row, not prose that merely mentions an id ("GR-019 ... are IRON rules").
ROW = re.compile(r"^GR-\d{3}\s*\|")

ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "_guardrail_inventory.md"
CLAUDE_MD = ROOT / "CLAUDE.md"
SKILLS_DIR = ROOT / ".claude" / "skills"


def load_iron() -> list[tuple[str, str, str]]:
    """Return [(id, section, quote)] for every IRON row in the inventory."""
    items = []
    for raw in INVENTORY.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not ROW.match(line):
            continue
        parts = [p.strip() for p in line.split("|", 3)]
        if len(parts) != 4:
            print(f"MALFORMED inventory row (needs 4 pipe-separated fields): {line}")
            sys.exit(2)
        gid, cls, section, quote = parts
        if cls == "IRON":
            items.append((gid, section, quote))
    return items


def haystacks() -> list[tuple[str, str]]:
    """CLAUDE.md plus any skill files — a guardrail may legitimately live in either."""
    out = [(str(CLAUDE_MD.relative_to(ROOT)), CLAUDE_MD.read_text(encoding="utf-8"))]
    if SKILLS_DIR.is_dir():
        for p in sorted(SKILLS_DIR.rglob("*.md")):
            out.append((str(p.relative_to(ROOT)), p.read_text(encoding="utf-8")))
    return out


def main() -> int:
    verbose = "--verbose" in sys.argv
    iron = load_iron()
    if not iron:
        print("FAIL: inventory contains no IRON rows — refusing to pass vacuously.")
        return 2

    sources = haystacks()
    missing = []
    for gid, section, quote in iron:
        found_in = next((name for name, text in sources if quote in text), None)
        if found_in is None:
            missing.append((gid, section, quote))
        elif verbose:
            print(f"  ok  {gid}  [{found_in}]  {quote[:70]}")

    print(f"\nchecked {len(iron)} IRON guardrails against {len(sources)} file(s)")
    if missing:
        print(f"\nFAIL — {len(missing)} guardrail(s) no longer present:\n")
        for gid, section, quote in missing:
            print(f"  {gid}  (was in: {section})")
            print(f"      {quote}\n")
        print("Roll back the trim step that removed them.")
        return 1

    print("PASS — every IRON guardrail is still present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
