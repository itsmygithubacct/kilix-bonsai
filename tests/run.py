#!/usr/bin/env python3
"""Run every test suite, one subprocess each, so a crash in one is a failure
in one rather than a mystery in all of them."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    wanted = set(sys.argv[1:])
    suites = sorted(
        f for f in os.listdir(HERE)
        if f.startswith("test_") and f.endswith(".py"))
    if wanted:
        suites = [f for f in suites
                  if f in wanted or f[len("test_"):-len(".py")] in wanted]
        if not suites:
            print(f"no suite matches {sorted(wanted)}", file=sys.stderr)
            return 2
    failures = 0
    for suite in suites:
        rc = subprocess.call([sys.executable, os.path.join(HERE, suite)])
        status = "ok" if rc == 0 else f"FAIL ({rc})"
        print(f"{suite}: {status}")
        failures += rc != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
