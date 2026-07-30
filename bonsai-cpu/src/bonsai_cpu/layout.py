"""Portable copy of the canonical Kilix text shell for CPU chat."""
from __future__ import annotations

import curses

from . import screen

SECTIONS = ("Chats", "Conversation", "Models", "Settings", "Help")
_ATTRS: dict[str, int] | None = None
_PAIR_BASE = 16


def attr(role: str) -> int:
    """Resolve the shared blue/red/white/grey Tango roles."""
    global _ATTRS
    if _ATTRS is None:
        try:
            if not curses.has_colors():
                raise RuntimeError("no colours")
            try:
                curses.use_default_colors()
                background = -1
            except Exception:
                background = curses.COLOR_BLACK
            pairs = (
                ("title", curses.COLOR_WHITE, -1),
                ("accent", curses.COLOR_BLUE, -1),
                ("alert", curses.COLOR_RED, -1),
                ("muted", curses.COLOR_WHITE, -1),
                ("selected", curses.COLOR_WHITE, curses.COLOR_BLUE),
            )
            resolved: dict[str, int] = {}
            for index, (name, foreground, pair_background) in enumerate(pairs):
                curses.init_pair(
                    _PAIR_BASE + index,
                    foreground,
                    pair_background if pair_background != -1 else background,
                )
                resolved[name] = curses.color_pair(_PAIR_BASE + index)
            resolved["title"] |= curses.A_BOLD
            resolved["muted"] |= curses.A_DIM
            resolved["selected"] |= curses.A_BOLD
            _ATTRS = resolved
        except Exception:
            _ATTRS = {
                "title": 101,
                "accent": 102,
                "alert": 103,
                "muted": 104,
                "selected": 105,
            }
    return _ATTRS.get(role, 0)


def shell(surface, *, breadcrumb: str, active: int, status: str,
          footer: str) -> tuple[int, int, int]:
    """Draw the shared header/navigation/footer and return the content well.

    The return value is ``(top, bottom, width)`` with ``bottom`` exclusive.
    The layout is fixed to the same rows as the VirtualBox manager.
    """
    height, width = surface.getmaxyx()
    if height <= 0 or width <= 0:
        return 0, 0, max(1, width)
    left = 1 if width > 2 else 0
    inner = max(0, width - (2 if width > 2 else 1))
    screen.write(surface, 0, left, "KILIX TUI", attr("title"))
    strap = "Bonsai · CPU Chat"
    if width - len(strap) - 1 > left + len("KILIX TUI"):
        screen.write(surface, 0, width - len(strap) - 1, strap,
                     attr("muted"))

    column = left
    for index, name in enumerate(SECTIONS):
        item = f"{'▶' if index == active else ' '}{index + 1} {name} "
        if column + len(item) >= width:
            break
        screen.write(
            surface, 1, column, item,
            attr("selected" if index == active else "muted"),
        )
        column += len(item)
    screen.write(surface, 2, 0, "─" * max(0, width - 1),
                 attr("muted"))

    summary = breadcrumb
    if status:
        summary = f"{summary} · {status}" if summary else status
    screen.write(surface, 3, left, summary[:inner],
                 attr("alert" if status else "muted"))
    if height > 1:
        screen.write(surface, height - 1, left, footer[:inner],
                     attr("muted"))
    return min(height - 1, 4), max(0, height - 1), width
