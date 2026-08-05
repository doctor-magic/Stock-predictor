#!/usr/bin/env python3
"""backup_dbs.py — third-copy snapshot of the irreplaceable research DBs.

setup_log.db, tracker.db and falling_knife_log.db are the only artifacts in this
project that cannot be rebuilt: every row is a live signal-time capture, and the
forward-only rule means a lost row is lost permanently, not recomputable.

  python3 backup_dbs.py --snapshot        # snapshot + hash only, no upload
  python3 backup_dbs.py --bucket gs://X   # snapshot + hash + upload + VERIFY
  python3 backup_dbs.py --selftest        # verify snapshot + hash logic offline

DESIGN CONSTRAINTS (deliberate, do not relax):
  • VACUUM INTO, never a file copy. A plain `cp` of a WAL database can capture a
    torn page set — the -wal sidecar holds committed pages the main file lacks.
    VACUUM INTO takes a read transaction and writes a self-contained, already
    compacted database. WAL readers do not block writers, so the API keeps
    serving throughout. The service is NEVER stopped and live -wal/-shm files
    are NEVER touched.
  • AN UNVERIFIED UPLOAD IS A FAILURE, NOT A WARNING. This script exits non-zero
    if the remote hash is missing or does not match. The failure mode being
    designed against is a backup that looks healthy for months and is not there
    when it matters — a silenced warning is exactly how that happens.
  • Hash comparison is base64-to-base64. GCS reports md5Hash as base64 of the
    RAW 16-byte digest; a hex digest string will never equal it, and comparing
    the two formats silently "fails to verify" forever. The selftest pins this.
  • The gcloud path is resolved IN CODE, not via a crontab PATH= line. Editing
    the crontab's PATH changes the environment of every other job already
    working; this script's dependency is this script's problem.
  • Tooling is `gcloud storage`, not `gsutil` — gsutil is on the way out and
    mixing the two across a project is how commands rot.

BUCKET REQUIREMENTS (set once, in the Console):
  • Object Versioning ON — the point is surviving a bad write or a deletion,
    not just a disk death. Without versioning a corrupted snapshot overwrites
    the good one on the next run.
  • Lifecycle on daysSinceNoncurrentTime (NOT age, which counts from creation)
    at 365 days. These files are ~300KB; retention costs cents, and a corruption
    found two months late is exactly the case the backup exists for.
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT_PATH = os.path.join(_HERE, "backup_state.json")

# Every one of these is live-captured and unrebuildable.
DATABASES = ["setup_log.db", "tracker.db", "falling_knife_log.db"]

# Well above any plausible size for these files. Composite uploads are what
# make GCS omit md5Hash; a missing hash is a hard failure below, so this is the
# belt to that suspenders.
COMPOSITE_THRESHOLD = "1G"


def log(msg):
    print(f"[backup] {msg}", flush=True)


def gcloud_bin():
    """Resolve gcloud once, loudly. Cron has a minimal PATH and snap installs
    land in /snap/bin, which is the usual reason a nightly backup silently stops."""
    found = shutil.which("gcloud") or ("/snap/bin/gcloud"
                                       if os.path.exists("/snap/bin/gcloud") else None)
    if not found:
        raise RuntimeError("gcloud not found on PATH or at /snap/bin/gcloud — "
                           "cannot upload. Refusing to report a backup.")
    return found


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def md5_b64(path, chunk=1 << 20):
    """base64 of the RAW md5 digest — the exact form GCS reports as md5Hash."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


def verify_hashes(local_b64, remote_b64, label=""):
    """Hard gate. Returns True or raises — it never returns False, because a
    caller that can keep going on a failed verification is the bug."""
    if not remote_b64:
        raise RuntimeError(
            f"{label}: GCS returned no md5Hash — cannot verify. This happens on "
            "composite uploads; the object may be fine but it is NOT proven, and "
            "an unproven backup is treated as a failed one.")
    if local_b64 != remote_b64:
        raise RuntimeError(f"{label}: hash mismatch — local {local_b64} != remote {remote_b64}")
    return True


def snapshot(src_path, dest_path):
    """Consistent, compacted copy of a live WAL database. Read-only on the source."""
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    con = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=30)
    try:
        con.execute("VACUUM INTO ?", (dest_path,))
    finally:
        con.close()
    # A snapshot that cannot be opened and counted is not a snapshot.
    ver = sqlite3.connect(f"file:{dest_path}?mode=ro", uri=True)
    try:
        if ver.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"integrity_check failed on {dest_path}")
        tables = [r[0] for r in ver.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: ver.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0] for t in tables}
    finally:
        ver.close()


def remote_md5(uri):
    """The object's own md5Hash, straight from GCS metadata. None if absent."""
    r = subprocess.run([gcloud_bin(), "storage", "objects", "describe", uri,
                        "--format=value(md5Hash)"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"objects describe failed: {r.stderr.strip()[:300]}")
    return (r.stdout or "").strip() or None


def upload(local_path, bucket, remote_name):
    """Upload, then prove it. Raises on anything less than a hash match."""
    uri = f"{bucket.rstrip('/')}/{remote_name}"
    log(f"uploading -> {uri}")
    r = subprocess.run([gcloud_bin(), "storage", "cp",
                        f"--parallel-composite-upload-threshold={COMPOSITE_THRESHOLD}",
                        local_path, uri], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcloud storage cp failed: {r.stderr.strip()[:300]}")
    verify_hashes(md5_b64(local_path), remote_md5(uri), label=remote_name)
    return uri


def write_receipt(entries, bucket, ok):
    """Local receipt so the watchdog can judge backup health without cloud creds."""
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "ok": ok,
               "bucket": bucket, "databases": entries}
    tmp = RECEIPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, RECEIPT_PATH)


def run(bucket=None, keep=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    workdir = tempfile.mkdtemp(prefix="dbbackup_")
    os.chmod(workdir, 0o700)
    entries, failures = [], []
    try:
        for name in DATABASES:
            src = os.path.join(_HERE, name)
            base = name.replace(".db", "")
            dest = os.path.join(workdir, f"{base}_{stamp}.db")
            try:
                counts = snapshot(src, dest)
                entry = {"db": name, "sha256": sha256_of(dest),
                         "md5_b64": md5_b64(dest),
                         "bytes": os.path.getsize(dest), "rows": counts}
                log(f"{name}: {entry['bytes']:,}B sha256={entry['sha256'][:16]}… rows={counts}")
                if bucket:
                    entry["uri"] = upload(dest, bucket, f"{base}/{base}_{stamp}.db")
                    entry["verified"] = True
                    log(f"{name}: uploaded and VERIFIED")
                entries.append(entry)
            except Exception as e:
                failures.append(f"{name}: {e}")
                log(f"FAILED {name}: {e}")
        if not bucket:
            log("no --bucket given: snapshot+hash only, nothing uploaded")
    finally:
        if keep:
            log(f"temp kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    ok = not failures
    if bucket or ok:
        write_receipt(entries, bucket, ok)
    if failures:
        for f in failures:
            log(f"  ! {f}")
        log(f"{len(failures)} of {len(DATABASES)} FAILED")
        sys.exit(1)
    log(f"ok — {len(entries)} database(s) snapshotted"
        + (f", uploaded and verified to {bucket}" if bucket else ""))


def selftest():
    tmp = tempfile.mkdtemp()

    # 1. snapshot consistency under a live open writer — the case a cp gets wrong
    src = os.path.join(tmp, "t.db")
    con = sqlite3.connect(src)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE rows (id INTEGER PRIMARY KEY, v TEXT)")
    con.executemany("INSERT INTO rows (v) VALUES (?)", [(f"r{i}",) for i in range(500)])
    con.commit()
    other = sqlite3.connect(src)
    other.execute("INSERT INTO rows (v) VALUES ('uncommitted')")

    dest = os.path.join(tmp, "snap.db")
    counts = snapshot(src, dest)
    assert counts["rows"] == 500, counts
    print(f"[selftest] snapshot consistent under a live open writer: {counts}")

    d1 = sha256_of(dest)
    dest2 = os.path.join(tmp, "snap2.db")
    snapshot(src, dest2)
    assert sha256_of(dest2) == d1, "VACUUM INTO not byte-stable across runs"
    print(f"[selftest] hash stable across runs: {d1[:16]}…")
    other.rollback(); other.close(); con.close()

    # 2. THE BUG THIS FILE WAS REWRITTEN FOR: base64-of-raw-digest, not hex.
    #    A hex digest compared against a GCS md5Hash never matches, and the old
    #    code degraded that into a warning — a permanently unverified backup.
    probe = os.path.join(tmp, "probe.bin")
    with open(probe, "wb") as f:
        f.write(b"stock-predictor")
    b64 = md5_b64(probe)
    assert base64.b64decode(b64).hex() == hashlib.md5(b"stock-predictor").hexdigest()
    assert b64 != hashlib.md5(b"stock-predictor").hexdigest()
    print(f"[selftest] md5 is base64-of-raw-digest as GCS reports it: {b64}")

    # 3. the verification gate must RAISE, never return False
    assert verify_hashes(b64, b64) is True
    for bad, why in ((None, "missing remote hash"),
                     ("", "empty remote hash"),
                     ("ZZZZZZZZZZZZZZZZZZZZZZ==", "mismatched hash")):
        try:
            verify_hashes(b64, bad, label="probe")
            raise AssertionError(f"{why} should have raised")
        except RuntimeError:
            pass
    print("[selftest] verification raises on missing, empty and mismatched remote hash")

    # 4. a missing source must raise rather than produce an empty backup
    try:
        snapshot(os.path.join(tmp, "nope.db"), os.path.join(tmp, "x.db"))
        raise AssertionError("missing source should have raised")
    except FileNotFoundError:
        print("[selftest] missing source raises rather than writing an empty backup")

    print("[selftest] all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", help="gs://bucket/prefix — omit for snapshot-only")
    ap.add_argument("--snapshot", action="store_true", help="explicit snapshot-only run")
    ap.add_argument("--keep", action="store_true", help="keep temp dir (debugging)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(bucket=None if a.snapshot else a.bucket, keep=a.keep)


if __name__ == "__main__":
    main()
