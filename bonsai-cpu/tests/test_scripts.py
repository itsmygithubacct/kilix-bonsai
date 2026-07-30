#!/usr/bin/env python3
"""The shell scripts parse, and keep the promises the README makes about
where things land."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def main():
    scripts = [os.path.join(REPO, "scripts", s)
               for s in ("get-runtime.sh", "build-runtime.sh")]
    scripts.append(os.path.join(REPO, "install.sh"))

    for path in scripts:
        name = os.path.basename(path)
        check(os.access(path, os.X_OK), f"{name} is executable")
        rc = subprocess.call(["bash", "-n", path])
        check(rc == 0, f"{name} parses (bash -n)")
        text = open(path).read()
        check("set -euo pipefail" in text, f"{name} fails fast")
        check("BASH_VERSINFO" in text, f"{name} refuses bash 4.3 and older")

    # Both runtime scripts honour the same override, or builds and checkouts
    # would land in different places.
    for name in ("get-runtime.sh", "build-runtime.sh"):
        text = open(os.path.join(REPO, "scripts", name)).read()
        check("BONSAI_CPU_RUNTIME_DIR" in text,
              f"{name} honours BONSAI_CPU_RUNTIME_DIR")

    # Nothing under version control may hard-code a home directory.
    needle = "/" + "home" + "/"          # not a literal, or this file trips it
    tracked = []
    for dp, dns, fns in os.walk(REPO):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__", ".work")]
        tracked += [os.path.join(dp, f) for f in fns
                    if not f.endswith(".pyc")]
    for path in tracked:
        try:
            text = open(path, errors="ignore").read()
        except OSError:
            continue
        check(needle not in text,
              f"{os.path.relpath(path, REPO)} avoids absolute home paths")

    for msg in failures:
        print(f"FAIL: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
