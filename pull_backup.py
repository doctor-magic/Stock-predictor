#!/usr/bin/env python3
"""pull_backup.py — Mac-side PULL backup of the unrebuildable research DBs.

Runs on the Mac, on a launchd timer. Asks the server for a consistent snapshot,
copies it down, proves it byte-for-byte, and posts a heartbeat back so the
server-side watchdog can alert when this stops running.

  python3 pull_backup.py            # the real run
  python3 pull_backup.py --dry-run  # snapshot + verify, no rotation, no heartbeat
  python3 pull_backup.py --selftest # offline checks of the pure logic

WHY PULL AND NOT PUSH — this is the point of the design, not a detail:
a push-to-cloud backup needs write credentials ON the production VM, whose
service still runs as User=root. Any compromise of the app would then reach the
backups it is supposed to survive. Pulling inverts that: the server holds no
credential and cannot touch its own backups. The SSH key lives here, on the
machine that is not exposed to the internet.

WHAT THIS TRADES AWAY: an always-on VM runs its cron every night; a laptop runs
it when it is awake. launchd re-fires a missed calendar job on wake, and the
heartbeat closes the rest of the gap — the server-side watchdog reports the
heartbeat's age daily over Telegram, so "the backup quietly stopped weeks ago"
becomes visible without anyone remembering to look.

Verification is SHA-256 on both ends, compared after the copy. A mismatch is a
hard failure: no heartbeat is written, and the run exits non-zero. A backup that
was not proven is treated as a backup that did not happen.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import hashlib
from datetime import datetime, timezone

SSH_KEY    = os.path.expanduser("~/.ssh/gcp_stock_rsa")
SERVER     = "elimaoz99@35.239.74.178"
REMOTE_DIR = "/home/elimaoz99/stock_predictor"
# NOT under ~/Desktop, ~/Documents or ~/Downloads. Those are TCC-protected on
# macOS, and a launchd agent is denied there with a bare "Operation not
# permitted" — the backup would simply stop, silently, which is precisely the
# failure this whole design exists to prevent. The alternative was granting Full
# Disk Access to the python3 binary, which hands that access to every script it
# ever runs; relocating is the smaller blast radius.
DEST_ROOT  = os.path.expanduser("~/stock-predictor-backups")
HEARTBEAT  = "mac_backup_heartbeat.json"
KEEP_DIRS  = 365          # ~350KB/day — a year of dailies is about a quarter GB


def log(msg):
    print(f"[pull] {msg}", flush=True)


def ssh(cmd, timeout=180):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                        "-i", SSH_KEY, SERVER, cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_workdir(stdout):
    """backup_dbs.py --keep prints 'temp kept at /tmp/dbbackup_xxxx'."""
    m = re.search(r"temp kept at (\S+)", stdout)
    if not m:
        raise RuntimeError("could not find the snapshot workdir in the server output")
    return m.group(1)


def parse_remote_hashes(stdout):
    """`sha256sum path` lines -> {basename: digest}."""
    out = {}
    for line in stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[os.path.basename(parts[1])] = parts[0]
    return out


def dirs_to_prune(existing, keep):
    """Dated backup dirs beyond the newest `keep`. Pure — the selftest pins it."""
    dated = sorted(d for d in existing if re.fullmatch(r"\d{8}", d))
    return dated[:-keep] if len(dated) > keep else []


def heartbeat_payload(entries, ok):
    return {"ts": datetime.now(timezone.utc).isoformat(), "ok": ok,
            "source": "mac-pull", "host": os.uname().nodename,
            "files": entries}


def pull(dry_run=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest = os.path.join(DEST_ROOT, stamp)
    os.makedirs(dest, exist_ok=True)

    log("asking the server for a consistent snapshot…")
    out = ssh(f"cd {REMOTE_DIR} && venv/bin/python3 backup_dbs.py --snapshot --keep")
    work = parse_workdir(out)
    remote = parse_remote_hashes(ssh(f"sha256sum {work}/*.db"))
    if not remote:
        raise RuntimeError("server produced no snapshot files")

    try:
        r = subprocess.run(["scp", "-q", "-i", SSH_KEY,
                            f"{SERVER}:{work}/*.db", dest + "/"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"scp failed: {r.stderr.strip()[:300]}")
    finally:
        # Always clear the server's temp dir, even if the copy failed.
        try:
            ssh(f"rm -rf {work}")
        except Exception as e:
            log(f"WARNING: could not clean server temp {work}: {e}")

    entries, bad = [], []
    for name, rhash in sorted(remote.items()):
        local_path = os.path.join(dest, name)
        if not os.path.exists(local_path):
            bad.append(f"{name}: never arrived")
            continue
        lhash = sha256_of(local_path)
        if lhash != rhash:
            bad.append(f"{name}: sha256 mismatch")
            continue
        entries.append({"file": name, "sha256": lhash,
                        "bytes": os.path.getsize(local_path)})
        log(f"{name}: {os.path.getsize(local_path):,}B verified {lhash[:16]}…")

    if bad:
        for b in bad:
            log(f"  ! {b}")
        raise RuntimeError(f"{len(bad)} file(s) failed verification — no heartbeat written")

    if dry_run:
        log(f"dry run — {len(entries)} file(s) verified into {dest}; "
            "no rotation, no heartbeat")
        return

    # Rotation, then heartbeat — the heartbeat is the LAST thing, so it only
    # exists when everything before it worked.
    pruned = dirs_to_prune(os.listdir(DEST_ROOT), KEEP_DIRS)
    for d in pruned:
        shutil.rmtree(os.path.join(DEST_ROOT, d), ignore_errors=True)
    if pruned:
        log(f"pruned {len(pruned)} dir(s) older than the newest {KEEP_DIRS}")

    payload = heartbeat_payload(entries, ok=True)
    tmp = os.path.join(dest, HEARTBEAT)
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    r = subprocess.run(["scp", "-q", "-i", SSH_KEY, tmp,
                        f"{SERVER}:{REMOTE_DIR}/{HEARTBEAT}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"heartbeat upload failed: {r.stderr.strip()[:300]}")
    log(f"ok — {len(entries)} file(s) in {dest}, heartbeat posted to the server")


def selftest():
    assert parse_workdir("[backup] temp kept at /tmp/dbbackup_ab12\n") == "/tmp/dbbackup_ab12"
    try:
        parse_workdir("[backup] ok — 3 database(s) snapshotted")
        raise AssertionError("missing workdir should have raised")
    except RuntimeError:
        pass
    print("[selftest] workdir parsing ok, and a missing workdir raises")

    h = parse_remote_hashes("abc123  /tmp/x/setup_log_20260805.db\n"
                            "def456  /tmp/x/tracker_20260805.db\n")
    assert h == {"setup_log_20260805.db": "abc123", "tracker_20260805.db": "def456"}, h
    print(f"[selftest] remote hash parsing ok: {len(h)} files")

    # Rotation must be date-ordered and must ignore anything that is not a
    # dated dir — a stray folder here should never cost a real backup.
    dirs = [f"2026{m:02d}{d:02d}" for m in (1, 2) for d in range(1, 11)] + ["notes", ".DS_Store"]
    pruned = dirs_to_prune(dirs, 5)
    assert len(pruned) == 15 and "notes" not in pruned and ".DS_Store" not in pruned
    assert pruned[-1] < sorted(d for d in dirs if d.isdigit())[-5], pruned
    assert dirs_to_prune(["20260101"], 5) == []
    print(f"[selftest] rotation keeps the newest 5, prunes {len(pruned)}, ignores non-dated")

    p = heartbeat_payload([{"file": "a.db", "sha256": "x", "bytes": 1}], ok=True)
    assert p["ok"] is True and p["source"] == "mac-pull" and p["files"]
    datetime.fromisoformat(p["ts"])
    print("[selftest] heartbeat payload well-formed")
    print("[selftest] all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        pull(dry_run=a.dry_run)
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
