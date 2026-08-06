"""Running a model folder's own scripts, from the CLI or from the TUI.

The scripts are the implementation; nothing here reimplements a download or a
dependency check. That matters more than it looks: `pull.sh` and
`install-deps.sh` have to work on a machine with no Python front end at all —
over SSH, from a provisioning script, from a Makefile — so they are the real
interface and this module is one of two callers.

The TUI hands the terminal *back* while a script runs rather than capturing its
output into a widget. A 4.5 GB download wants curl's own progress meter, and a
dependency install may ask for a sudo password; both are things a curses pane
does badly and a plain terminal does perfectly.
"""
from __future__ import annotations

import curses
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Sequence

from .catalog import Model, Variant


def pull_argv(model: Model, variant: Variant | None = None, *,
              force: bool = False, dry_run: bool = False,
              source: str | None = None) -> list[str]:
    argv = [model.script("pull.sh")]
    if variant is not None and not variant.default:
        argv += ["--variant", variant.id]
    if source:
        argv += ["--from", source]
    if force:
        argv.append("--force")
    if dry_run:
        argv.append("--dry-run")
    return argv


def deps_argv(model: Model, *, apt: bool = False) -> list[str]:
    argv = [model.script("install-deps.sh")]
    if apt:
        argv.append("--apt")
    return argv


def run(argv: Sequence[str], *, cwd: str | None = None) -> int:
    """Run one script with the terminal it needs, and return its exit status."""
    try:
        return subprocess.call(list(argv), cwd=cwd)
    except OSError as error:
        print(f"kilix-bonsai: could not run {argv[0]}: {error}",
              file=sys.stderr)
        return 127


@contextmanager
def terminal(state: Any, heading: str):
    """Hand the terminal back for the duration of a long, noisy operation.

    `state.stdscr` is None under `--screenshot` and in the tests, where there is
    no curses screen to suspend; the body still runs, which is what makes these
    paths testable without a pty.
    """
    stdscr = getattr(state, "stdscr", None)
    if stdscr is not None:
        curses.def_prog_mode()
        curses.endwin()
    try:
        print(f"\n\033[1m{heading}\033[0m\n")
        yield
    finally:
        if stdscr is not None:
            print("press Enter to return to Kilix Bonsai… ", end="", flush=True)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            stdscr.refresh()
            curses.doupdate()


def run_detached_from_curses(state: Any, argv: Sequence[str],
                             heading: str) -> int:
    """Suspend curses, run a script on the real terminal, then resume."""
    return run_steps_detached_from_curses(state, [argv], heading)


def run_steps_detached_from_curses(state: Any,
                                   argvs: Sequence[Sequence[str]],
                                   heading: str) -> int:
    """Run several commands under one terminal handover, stopping at failure.

    One handover, not one per command: a confirmed offer that needs two steps
    (install the build tools, then build) must not bounce through
    \"press Enter to return\" in the middle, and a failed first step must not
    run the second — that would bury the failure under the follow-on error.
    """
    status = 127
    with terminal(state, heading):
        for argv in argvs:
            print(f"\n$ {' '.join(argv)}\n")
            status = run(argv)
            if status != 0:
                print(f"\n{os.path.basename(argv[0])} exited {status}.")
                break
        else:
            print("\ndone.")
    return status
