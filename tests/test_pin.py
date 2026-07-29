#!/usr/bin/env python3
"""runtime.pin is the single definition of the runtime; keep it honest."""

import hashlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = os.path.join(REPO, "runtime.pin")
PATCH_DIR = os.path.join(REPO, "patches")

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def main():
    with open(PIN) as fh:
        lines = fh.read().splitlines()

    url = [l for l in lines if l.startswith("url=")]
    commit = [l for l in lines if l.startswith("commit=")]
    check(len(url) == 1, "exactly one url= line")
    check(len(commit) == 1, "exactly one commit= line")
    if commit:
        sha = commit[0].split("=", 1)[1]
        check(re.fullmatch(r"[0-9a-f]{40}", sha) is not None,
              f"commit is a full 40-hex sha, got {sha!r}")
    if url:
        check(url[0].split("=", 1)[1].startswith("https://"),
              "url is https")

    listed = {}
    for line in lines:
        if line.startswith(("#", "url=", "commit=")) or not line.strip():
            continue
        parts = line.split()
        check(len(parts) == 2, f"patch line has file and sha256: {line!r}")
        if len(parts) == 2:
            listed[parts[0]] = parts[1]

    on_disk = set()
    if os.path.isdir(PATCH_DIR):
        on_disk = {f for f in os.listdir(PATCH_DIR) if not f.startswith(".")}

    check(set(listed) == on_disk,
          f"patches/ and runtime.pin agree: pin={sorted(listed)} "
          f"disk={sorted(on_disk)}")

    for name, want in listed.items():
        path = os.path.join(PATCH_DIR, name)
        if os.path.isfile(path):
            got = hashlib.sha256(open(path, "rb").read()).hexdigest()
            check(got == want, f"{name} sha256 matches pin")

    for msg in failures:
        print(f"FAIL: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
