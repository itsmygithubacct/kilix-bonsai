"""The panel chrome, borrowed from `kilix-tui-utils` when it is installed.

This repository deliberately copied a small `screen.py` rather than importing
the shared core, on the grounds that a model store should not need a second
checkout to draw a list. A shared *theme* is a much better reason to depend
than a shared event loop was — thirteen tools inventing their own panel
vocabulary is exactly the duplication that core exists to prevent — but the
original reason has not gone away either.

So the dependency is optional, the same bargain `kilix_tui.theme` already
strikes with the Kilix SDK: import it when it is reachable, and fall back to a
plain layout with the identical API when it is not. A machine with the
utilities installed gets the full chrome; a bare checkout over `ssh` still
runs, and every screen still renders to plain text for the tests.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_CORE: Any | None = None


def _core() -> Any | None:
    """Return the `kilix_tui` package, or None when it is not reachable."""
    global _CORE
    if _CORE is not None:
        return _CORE or None
    import importlib

    source_home = os.environ.get("GPU_TERMINAL_SOURCE_HOME") or os.path.join(
        os.path.expanduser("~"), "gpu_terminal")
    candidates = [
        os.environ.get("KILIX_TUI_UTILS_HOME", ""),
        os.path.join(os.path.abspath(os.path.expanduser(source_home)),
                     "kilix-tui-utils"),
    ]
    for home in candidates:
        source = os.path.join(home, "src") if home else ""
        if source and os.path.isdir(os.path.join(source, "kilix_tui")):
            if source not in sys.path:
                sys.path.insert(0, source)
            break
    try:
        chrome = importlib.import_module("kilix_tui.chrome")
        importlib.import_module("kilix_tui.panel")
        _CORE = chrome
    except ImportError:
        _CORE = False
        return None
    return _CORE


def available() -> bool:
    return _core() is not None


class _PlainPage:
    """The fallback: a title, a rule, and the whole surface as the well.

    Same API as the shared `Page`, so a screen written against one works
    against the other without asking which it got.
    """

    def __init__(self, title: str, sections=(), **_: Any) -> None:
        self.title = title
        self.sections = list(sections)
        self._height, self._width = 24, 80

    def measure(self, surface: Any) -> None:
        try:
            self._height, self._width = surface.getmaxyx()
        except Exception:                                  # pragma: no cover
            self._height, self._width = 24, 80

    @property
    def spined(self) -> bool:
        return False

    def content_box(self) -> tuple[int, int, int, int]:
        return 2, 0, max(0, self._height - 4), self._width

    def render(self, surface: Any, active_section: int = 0, *,
               footer: str = "", status: str = "") -> None:
        from . import screen
        self.measure(surface)
        screen.write(surface, 0, 0, self.title)
        if status:
            screen.write(surface, 0, max(0, self._width - len(status) - 1),
                         status)
        screen.write(surface, 1, 0, "─" * max(0, self._width - 1))
        if footer:
            screen.write(surface, self._height - 1, 0, footer)


def page(title: str, sections=(), **kwargs: Any) -> Any:
    """Return a shared `Page` when the core is installed, else the fallback."""
    core = _core()
    if core is None:
        return _PlainPage(title, sections, **kwargs)
    return core.Page(title, sections, **kwargs)
