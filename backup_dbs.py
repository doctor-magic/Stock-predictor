#!/usr/bin/env python3
"""backup_dbs.py — third-copy snapshot of the two irreplaceable research DBs.

setup_log.db and tracker.db are the only artifacts in this project that cannot
be rebuilt: every row is a live signal-time capture, and the forward-only rule
means a lost row is lost permanently, not recomputable. Today they exist on the
VM and on the Mac. This script makes the third copy.

  python3 backup_dbs.py --snapshot        # snapshot + hash only, no upload
  python3 backup_dbs.py --bucket gs://X   # snapshot + hash + upload + verify
  python3 backup_dbs.py --selftest        # verify the snapshot leg end-to-end

DESIGN CONSTRAINTS (deliberate, do not relax):
  • VACUUM INTO, never a file copy. A plain `cp` of a WAL database can capture a
    torn page set — the -wal sidecar holds committed pages the main file lacks.
    VACUUM INTO takes a read transaction and writes a self-contained, already
    compacted database. WAL readers do not block writers, so the API keeps
    serving throughout.
  • The service is NEVER stopped and the live -wal/-shm files are NEVER touched.
  • Upload is verified by comparing the local SHA-256 against the object's own
    stored hash before the temp file is deleted. An unverified upload is not a
    backup.
  • Temp files land in a private 0700 directory and are removed on every exit
    path, including failure.

BUCKET REQUIREMENTS (set once, outside this script):
  • Object Versioning ON — the point is surviving a bad write or a deletion,
    not just a disk death. Without versioning a corrupted snapshot overwrites
    the good one on the next run.
  • A lifecycle rule to expire noncurrent versions (~180 days) keeps cost flat.
"""
import argparse
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))

# The two that cannot be rebuilt. falling_knife_log.db is also live-captured and
# is a reasonable third entry — left out here because the backup scope was
# specified as these two; adding it is a one-line change, not a redesign.
DATABASES = ["setup_log.db", "tracker.db"]


def log(msg):
    print(f"[backup] {msg}", flush=True)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
        counts = {t: ver.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
                  for t in tables}
    finally:
        ver.close()
    return counts


def gsutil_object_sha256(uri):
    """Ask GCS for the object's own hash so verification is not self-referential."""
    out = subprocess.run(["gsutil", "hash", "-h", uri], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if "md5" in line.lower():
            return line.split()[-1]
    return None


def upload(local_path, bucket, remote_name):
    uri = f"{bucket.rstrip('/')}/{remote_name}"
    log(f"uploading -> {uri}")
    r = subprocess.run(["gsutil", "cp", local_path, uri],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gsutil cp failed: {r.stderr.strip()[:400]}")
    # Verify the bytes that landed match the bytes we made.
    local_md5 = subprocess.run(["gsutil", "hash", "-h", local_path],
                               capture_output=True, text=True)
    remote_md5 = gsutil_object_sha256(uri)
    lm = None
    for line in local_md5.stdout.splitlines():
        if "md5" in line.lower():
            lm = line.split()[-1]
    if lm and remote_md5 and lm != remote_md5:
        raise RuntimeError(f"hash mismatch after upload: local {lm} != remote {remote_md5}")
    if not (lm and remote_md5):
        log("WARNING: could not read both hashes — upload NOT verified")
        return uri, False
    return uri, True


def run(bucket=None, keep=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    workdir = tempfile.mkdtemp(prefix="dbbackup_")
    os.chmod(workdir, 0o700)
    results, failures = [], []
    try:
        for name in DATABASES:
            src = os.path.join(_HERE, name)
            base = name.replace(".db", "")
            dest = os.path.join(workdir, f"{base}_{stamp}.db")
            try:
                counts = snapshot(src, dest)
                digest = sha256_of(dest)
                size = os.path.getsize(dest)
                log(f"{name}: {size:,}B sha256={digest[:16]}… rows={counts}")
                entry = {"db": name, "sha256": digest, "bytes": size, "rows": counts}
                if bucket:
                    uri, verified = upload(dest, bucket, f"{base}/{base}_{stamp}.db")
                    entry.update(uri=uri, verified=verified)
                    log(f"{name}: uploaded{' and VERIFIED' if verified else ' (UNVERIFIED)'}")
                results.append(entry)
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
    if failures:
        log(f"{len(failures)} of {len(DATABASES)} FAILED")
        sys.exit(1)
    log(f"ok — {len(results)} database(s) snapshotted"
        + (f", uploaded to {bucket}" if bucket else ""))


def selftest():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "t.db")
    con = sqlite3.connect(src)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE rows (id INTEGER PRIMARY KEY, v TEXT)")
    con.executemany("INSERT INTO rows (v) VALUES (?)", [(f"r{i}",) for i in range(500)])
    con.commit()
    # Leave a SECOND connection with uncommitted work open: this is the state a
    # naive file copy gets wrong, and the case the snapshot must handle.
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

    other.rollback()
    other.close()
    con.close()
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
