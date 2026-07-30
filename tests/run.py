#!/usr/bin/env python3
"""Run every suite in a fresh subprocess.

Same shape as the runner the other Kilix repositories use: one process per
file, so a suite that leaves curses or an import in a bad state cannot affect
the next one, and a crash is attributed to the file that caused it.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    suites = [
        (name, os.path.join(HERE, name))
        for name in sorted(os.listdir(HERE))
        if name.startswith("test_") and name.endswith(".py")
    ]
    cpu_runner = os.path.join(ROOT, "bonsai-cpu", "tests", "run.py")
    if os.path.isfile(cpu_runner):
        suites.append(("bonsai-cpu", cpu_runner))
    if sys.argv[1:]:
        suites = [
            suite for suite in suites
            if any(argument in suite[0] for argument in sys.argv[1:])
        ]
    failed = []
    for name, path in suites:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, env=env,
        )
        if result.returncode == 0:
            print(f"PASS  {name}")
        else:
            failed.append(name)
            print(f"FAIL  {name}")
            for line in (result.stdout + result.stderr).splitlines():
                print(f"  {line}")
    print(f"{len(suites) - len(failed)}/{len(suites)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
