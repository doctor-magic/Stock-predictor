#!/usr/bin/env python3
"""pull_backup.py — Mac-side PULL backup of the unrebuildable research DBs.

Runs on the Mac, on a launchd timer. Asks the server for a consistent snapshot,
copies it down, proves it byte-for-byte, and posts a heartbeat back so the
server-side watchdog can alert when this stops running.

  python3 pull_backup.py            # the real run
  python3 pull_backup.py --dry-run  # snapshot + verify, keeps nothing, no heartbeat
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

THE WAKE RACE (Aug 6-8 2026, three silent nights): the same launchd behaviour
that makes a laptop backup viable — re-firing a missed calendar job on wake —
fires it INTO a Mac whose Wi-Fi has not associated yet. Three runs died on
"Network is unreachable" / "Connection reset", one attempt each, and the
server-side watchdog only crosses its 72h threshold on the fourth day. Hence
the two rules below: connection-shaped failures are RETRIED, and the dated
directory is created only once files are verified, so a failed run leaves
nothing that could be mistaken for a backup.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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
ATTEMPTS   = 5            # ~4 min of patience, against a Wi-Fi link coming up
RETRY_WAIT = 60


def log(msg):
    print(f"[pull] {msg}", flush=True)


class NetworkError(RuntimeError):
    """A failure a later attempt could plausibly fix. Kept distinct from a
    verification failure, which must NEVER be retried into looking fine."""


# Matched against stderr, not against ssh's exit code: 255 is both "I could not
# connect" and "your remote command exited 255", and only the first is worth
# waiting on.
_TRANSIENT = re.compile(
    r"network is unreachable|connection reset|connection refused|"
    r"connection closed|no route to host|connection timed out|"
    r"operation timed out|temporary failure in name resolution|"
    r"could not resolve hostname|broken pipe|host is down",
    re.IGNORECASE)


def is_transient(text):
    """Pure — the selftest pins it against the three stderrs actually seen."""
    return bool(_TRANSIENT.search(text or ""))


def _run(argv, what, timeout):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise NetworkError(f"{what} timed out after {timeout}s")
    if r.returncode != 0:
        err = r.stderr.strip()[:300]
        msg = f"{what} failed ({r.returncode}): {err}"
        raise NetworkError(msg) if is_transient(err) else RuntimeError(msg)
    return r.stdout


def ssh(cmd, timeout=180):
    return _run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                 "-i", SSH_KEY, SERVER, cmd], "ssh", timeout)


def scp(src, dst, timeout=300):
    return _run(["scp", "-q", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                 "-i", SSH_KEY, src, dst], "scp", timeout)


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


def promote(staging, dest, names):
    """Move verified files into the dated dir — which is created HERE and
    nowhere earlier. A run that dies before this point must leave no trace:
    an empty 20260806/ sitting between real backups reads to every future
    reader as a backup that happened. Three of those are how the Aug 6-8
    outage stayed invisible."""
    os.makedirs(dest, exist_ok=True)
    for n in names:
        shutil.move(os.path.join(staging, n), os.path.join(dest, n))
    return dest


def pull(dry_run=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest = os.path.join(DEST_ROOT, stamp)
    os.makedirs(DEST_ROOT, exist_ok=True)
    # Files land here first and are proven here. Not a dated name, so rotation
    # ignores it; removed in the finally, so a crash cannot leave it behind.
    staging = tempfile.mkdtemp(prefix=".incoming-", dir=DEST_ROOT)

    try:
        log("asking the server for a consistent snapshot…")
        # Sweep snapshots orphaned by an earlier broken run before making a new
        # one. The per-run `finally` below cannot be relied on for this: it
        # cleans up over the very SSH connection that just died (Aug 7 and 8
        # each leaked ~370KB this way). +60min never touches a live run's dir.
        out = ssh(f"find /tmp -maxdepth 1 -name 'dbbackup_*' -type d -mmin +60 "
                  f"-exec rm -rf {{}} + 2>/dev/null; "
                  f"cd {REMOTE_DIR} && venv/bin/python3 backup_dbs.py --snapshot --keep")
        work = parse_workdir(out)
        remote = parse_remote_hashes(ssh(f"sha256sum {work}/*.db"))
        if not remote:
            raise RuntimeError("server produced no snapshot files")

        try:
            scp(f"{SERVER}:{work}/*.db", staging + "/")
        finally:
            # Always clear the server's temp dir, even if the copy failed.
            try:
                ssh(f"rm -rf {work}")
            except Exception as e:
                log(f"WARNING: could not clean server temp {work}: {e}")

        entries, bad = [], []
        for name, rhash in sorted(remote.items()):
            local_path = os.path.join(staging, name)
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
            log(f"dry run — {len(entries)} file(s) verified; nothing kept, "
                "no rotation, no heartbeat")
            return

        promote(staging, dest, [e["file"] for e in entries])

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
        scp(tmp, f"{SERVER}:{REMOTE_DIR}/{HEARTBEAT}", timeout=60)
        log(f"ok — {len(entries)} file(s) in {dest}, heartbeat posted to the server")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def pull_with_retry(dry_run=False, attempts=ATTEMPTS, wait=RETRY_WAIT, sleeper=time.sleep):
    """Retry ONLY connection-shaped failures. A verification failure is a real
    answer and propagates on the first attempt — retrying that would be the
    program arguing with its own evidence."""
    for i in range(1, attempts + 1):
        try:
            return pull(dry_run=dry_run)
        except NetworkError as e:
            if i == attempts:
                spent = (attempts - 1) * wait
                span = f"{spent // 60} min" if spent >= 60 else f"{spent}s"
                raise RuntimeError(
                    f"{e} — gave up after {attempts} attempts over ~{span}")
            log(f"attempt {i}/{attempts}: {e} — retrying in {wait}s")
            sleeper(wait)


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

    # The three stderrs that actually killed Aug 6, 7 and 8 must classify as
    # retryable; a verification failure must never be swept into that bucket.
    for s in ("ssh: connect to host 35.239.74.178 port 22: Network is unreachable",
              "Connection reset by 35.239.74.178 port 22",
              "ssh: connect to host x port 22: Operation timed out",
              "Could not resolve hostname x: nodename nor servname provided"):
        assert is_transient(s), s
    for s in ("Permission denied (publickey).",
              "1 file(s) failed verification — no heartbeat written",
              "sha256 mismatch", ""):
        assert not is_transient(s), s
    print("[selftest] transient classification: the 3 observed wake-race errors "
          "retry, auth + verification failures do not")

    # Retry: transient recovers, hard fails immediately, exhaustion is an error.
    calls = {"n": 0}
    def flaky(dry_run=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise NetworkError("Network is unreachable")
        return "ok"
    _real_pull = globals()["pull"]
    try:
        globals()["pull"] = flaky
        assert pull_with_retry(attempts=5, wait=0, sleeper=lambda s: None) == "ok"
        assert calls["n"] == 3, calls

        globals()["pull"] = lambda dry_run=False: (_ for _ in ()).throw(
            NetworkError("Connection reset"))
        try:
            pull_with_retry(attempts=3, wait=0, sleeper=lambda s: None)
            raise AssertionError("exhausted retries should have raised")
        except RuntimeError as e:
            assert "gave up after 3 attempts" in str(e), e
            assert not isinstance(e, NetworkError)

        hard = {"n": 0}
        def verification_failure(dry_run=False):
            hard["n"] += 1
            raise RuntimeError("1 file(s) failed verification")
        globals()["pull"] = verification_failure
        try:
            pull_with_retry(attempts=5, wait=0, sleeper=lambda s: None)
            raise AssertionError("a hard failure should have raised")
        except RuntimeError:
            pass
        assert hard["n"] == 1, f"verification failure was retried {hard['n']}x"
    finally:
        globals()["pull"] = _real_pull
    print("[selftest] retry: recovers on attempt 3, gives up after N, "
          "never retries a verification failure")

    # The invariant behind the empty 20260806/: no dated dir before promotion.
    with tempfile.TemporaryDirectory() as root:
        staging = os.path.join(root, ".incoming-x")
        os.makedirs(staging)
        for n in ("setup_log_20260809.db", "tracker_20260809.db"):
            with open(os.path.join(staging, n), "w") as f:
                f.write("x")
        dest = os.path.join(root, "20260809")
        assert not os.path.exists(dest), "dated dir must not exist before promote"
        promote(staging, dest, ["setup_log_20260809.db", "tracker_20260809.db"])
        assert sorted(os.listdir(dest)) == ["setup_log_20260809.db",
                                            "tracker_20260809.db"]
        assert os.listdir(staging) == []
    print("[selftest] promote creates the dated dir only once files are verified")
    print("[selftest] all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS,
                    help=f"tries before giving up (default {ATTEMPTS})")
    ap.add_argument("--retry-wait", type=int, default=RETRY_WAIT,
                    help=f"seconds between tries (default {RETRY_WAIT})")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        pull_with_retry(dry_run=a.dry_run, attempts=a.attempts, wait=a.retry_wait)
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
