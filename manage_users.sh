#!/usr/bin/env bash
# manage_users.sh — change a password or add/remove a login, safely.
#
# Editing api_data.env by hand is how you get locked out: one stray comma and
# BASIC_AUTH_USERS parses to nothing, which (by design) makes the API answer 503
# on every request. This script never lets you see or type the password on the
# command line, keeps a timestamped backup, refuses to write a file it cannot
# parse back, and restarts the service for you.
#
#   ./manage_users.sh list
#   ./manage_users.sh passwd <user>
#   ./manage_users.sh add <user>
#   ./manage_users.sh remove <user>
#
# Run it ON THE SERVER, from /home/elimaoz99/stock_predictor.
set -euo pipefail
cd "$(dirname "$0")"
ENVF="api_data.env"
SERVICE="stock-app.service"

[[ -f "$ENVF" ]] || { echo "ERROR: $ENVF not found in $(pwd)" >&2; exit 1; }
[[ -w "$ENVF" ]] || { echo "ERROR: $ENVF is not writable by $(whoami)" >&2; exit 1; }

usage() { sed -n '3,17p' "$0" | sed 's/^# \?//'; exit 1; }
[[ $# -ge 1 ]] || usage
ACTION="$1"; USER_ARG="${2:-}"

# The first pair is special: live_tracker.py and warm_volume_cache.sh both
# authenticate as it. Reordering it silently changes which account the crons
# run as, so this script always preserves position.
current_users() { grep -E '^BASIC_AUTH_USERS=' "$ENVF" | head -1 | cut -d= -f2-; }

case "$ACTION" in
  list)
    echo "Logins configured (passwords hidden):"
    i=1
    IFS=',' read -ra PAIRS <<< "$(current_users)"
    for p in "${PAIRS[@]}"; do
      name="${p%%:*}"
      if [[ $i -eq 1 ]]; then
        echo "  $i. $name   <- crons authenticate as this one; keep it first"
      else
        echo "  $i. $name"
      fi
      i=$((i+1))
    done
    exit 0
    ;;
  passwd|add|remove) [[ -n "$USER_ARG" ]] || usage ;;
  *) usage ;;
esac

if [[ "$USER_ARG" =~ [,:] ]]; then
  echo "ERROR: a username cannot contain ',' or ':' — they separate the fields." >&2
  exit 1
fi

PW=""
if [[ "$ACTION" != "remove" ]]; then
  # -s: never echoed. Read into the environment, never onto the command line —
  # argv is visible to every user on the box via ps.
  read -rsp "New password for '$USER_ARG': " PW; echo
  read -rsp "Repeat it: " PW2; echo
  [[ "$PW" == "$PW2" ]] || { echo "ERROR: passwords do not match — nothing changed." >&2; exit 1; }
  [[ ${#PW} -ge 8 ]] || { echo "ERROR: use at least 8 characters — nothing changed." >&2; exit 1; }
  if [[ "$PW" =~ [,:] ]]; then
    echo "ERROR: the password cannot contain ',' or ':' — they separate the fields." >&2
    exit 1
  fi
  # ASCII ONLY. HTTP Basic Auth carries credentials as base64 of raw bytes with
  # no agreed encoding, so a non-ASCII character (Hebrew, accents, emoji) is
  # encoded one way by the browser and decoded another way by Starlette — the
  # password then never matches and the account is locked out, taking
  # live_tracker.py and warm_volume_cache.sh down with it. Learned the hard way
  # on Aug 17 2026: a 5-Hebrew-character password produced exactly that.
  if LC_ALL=C grep -qP '[^\x20-\x7E]' <<< "$PW" 2>/dev/null \
     || [[ "$(LC_ALL=C printf '%s' "$PW" | wc -c)" -ne "${#PW}" ]]; then
    echo "ERROR: use ASCII only — English letters, digits and symbols." >&2
    echo "       Hebrew or other non-ASCII characters cannot travel through" >&2
    echo "       HTTP Basic Auth and would lock the account out." >&2
    echo "       Nothing changed." >&2
    exit 1
  fi
fi

BACKUP="${ENVF}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
cp -p "$ENVF" "$BACKUP"

# Python does the rewrite: it reads the password from stdin (not argv — argv is
# visible to every user on the box via ps) and rebuilds the line field by field,
# so an unrelated key in api_data.env is never touched.
#
# The helper goes to a temp FILE rather than `python3 - <<PY`: a heredoc IS
# stdin, so with that form the program itself lands where the password should
# be and every password silently becomes empty. (Caught in rehearsal — the
# resulting line was `elim:,dani:`, which would have locked the account out.)
PYHELPER="$(mktemp)"
trap 'rm -f "$PYHELPER"' EXIT
cat > "$PYHELPER" <<'PY'
import os, sys

action, target, path = os.environ["ACTION"], os.environ["TARGET"], os.environ["ENVF"]
password = sys.stdin.read()
if action != "remove" and not password:
    sys.exit("refusing to write an EMPTY password — that would lock the account out")
if action != "remove" and not all(32 <= ord(c) < 127 for c in password):
    sys.exit("refusing to write a non-ASCII password — HTTP Basic Auth cannot "
             "carry it and the account would be locked out")

lines = open(path).read().splitlines()
idx = next((i for i, l in enumerate(lines) if l.startswith("BASIC_AUTH_USERS=")), None)
if idx is None:
    sys.exit("BASIC_AUTH_USERS= line not found")

raw = lines[idx].split("=", 1)[1].strip().strip('"').strip("'")
pairs = [p for p in (x.strip() for x in raw.split(",")) if ":" in p]
names = [p.split(":", 1)[0] for p in pairs]

if action == "add":
    if target in names:
        sys.exit(f"'{target}' already exists — use passwd to change its password")
    pairs.append(f"{target}:{password}")          # appended LAST: the crons' first pair is untouched
elif action == "passwd":
    if target not in names:
        sys.exit(f"'{target}' not found — use add to create it")
    pairs[names.index(target)] = f"{target}:{password}"   # in place, position preserved
elif action == "remove":
    if target not in names:
        sys.exit(f"'{target}' not found")
    if len(pairs) == 1:
        sys.exit("refusing to remove the last login — that locks everyone out (API would 503)")
    if names.index(target) == 0:
        sys.exit("refusing to remove the FIRST login — live_tracker.py and "
                 "warm_volume_cache.sh authenticate as it. Promote another one first.")
    pairs.pop(names.index(target))

lines[idx] = "BASIC_AUTH_USERS=" + ",".join(pairs)

# Parse the result back before writing: if this cannot round-trip, the API would
# have 503'd on every request after the restart.
check = [p for p in lines[idx].split("=", 1)[1].split(",") if ":" in p]
if len(check) != len(pairs) or not all(p.split(":", 1)[0] for p in check):
    sys.exit("refusing to write — the rebuilt line does not parse back")

open(path, "w").write("\n".join(lines) + "\n")
print(f"OK: {action} '{target}' — {len(pairs)} login(s) configured")
PY

if ! printf '%s' "$PW" | ACTION="$ACTION" TARGET="$USER_ARG" ENVF="$ENVF" python3 "$PYHELPER"
then
  echo "FAILED — restoring $BACKUP, nothing changed." >&2
  cp -p "$BACKUP" "$ENVF"
  exit 1
fi

echo "Restarting $SERVICE (the file is parsed once, at startup)…"
sudo systemctl restart "$SERVICE"
sleep 3
if [[ "$(systemctl is-active "$SERVICE")" == "active" ]]; then
  echo "Service is active. Backup kept at $BACKUP"
  echo
  echo "NOTE: browsers hold credentials in sessionStorage — anyone whose password"
  echo "      changed gets a 401 on their next call and is returned to the login screen."
else
  echo "SERVICE DID NOT COME BACK. Restore with:" >&2
  echo "  cp $BACKUP $ENVF && sudo systemctl restart $SERVICE" >&2
  exit 1
fi
