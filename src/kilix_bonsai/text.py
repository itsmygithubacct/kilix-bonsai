"""The canonical Kilix text shell, kept portable for the Bonsai tools.

Bonsai intentionally runs without requiring a second checkout, so this is a
small local copy of ``kilix_tui.shell``.  Its contract is the same: identity
and application on row zero, numbered navigation on row one, one divider,
status on row three, content below, and a quiet footer.
"""
from __future__ import annotations

import curses
from dataclasses import dataclass
from typing import Iterable

from . import screen

_ATTRS: dict[str, int] | None = None
_PAIR_BASE = 16


def _clip(value: str, width: int) -> str:
    return value[:max(0, width)]


def attr(role: str) -> int:
    """Resolve the same Tango text roles as the shared Kilix shell."""
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
                ("danger", curses.COLOR_WHITE, curses.COLOR_RED),
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
            resolved["danger"] |= curses.A_BOLD
            _ATTRS = resolved
        except Exception:
            _ATTRS = {
                "title": 101,
                "accent": 102,
                "alert": 103,
                "muted": 104,
                "selected": 105,
                "danger": 106,
            }
    return _ATTRS.get(role, 0)


@dataclass
class Page:
    """Measured text-mode shell shared by the launcher and task tools."""

    area: str
    breadcrumb: str
    sections: tuple[str, ...] = ()
    active: int = 0
    _size: tuple[int, int] = (24, 80)

    def measure(self, surface) -> None:
        self._size = surface.getmaxyx()

    def content_box(self) -> tuple[int, int, int, int]:
        height, width = self._size
        top = 4
        bottom = max(top, height - 1)
        left = 1 if width > 2 else 0
        return top, left, max(0, bottom - top), \
            max(0, width - (2 if width > 2 else 1))

    def render(self, surface, selected: int = 0, *, footer: str = "",
               status: str = "") -> None:
        del selected
        self.measure(surface)
        height, width = self._size
        if height <= 0 or width <= 0:
            return

        left = 1 if width > 2 else 0
        inner = max(0, width - (2 if width > 2 else 1))
        screen.write(surface, 0, left, _clip("KILIX TUI", inner),
                     attr("title"))
        strap = f"Bonsai · {self.area.title()}"
        if width - len(strap) - 1 > left + len("KILIX TUI"):
            screen.write(surface, 0, width - len(strap) - 1, strap,
                         attr("muted"))

        column = left
        labels = self.sections or ("Overview",)
        for index, label in enumerate(labels):
            item = f"{'▶' if index == self.active else ' '}{index + 1} {label} "
            if column + len(item) >= width:
                break
            screen.write(
                surface, 1, column, item,
                attr("selected" if index == self.active else "muted"),
            )
            column += len(item)

        screen.write(surface, 2, 0, "─" * max(0, width - 1),
                     attr("muted"))
        summary = self.breadcrumb
        if status:
            summary = f"{summary} · {status}" if summary else status
        screen.write(surface, 3, left, _clip(summary, inner),
                     attr("alert" if status else "muted"))

        if height >= 2:
            screen.write(surface, height - 1, left,
                         _clip(footer, inner), attr("muted"))


def page(area: str, breadcrumb: str, sections: Iterable[str] = (),
         active: int = 0) -> Page:
    return Page(area, breadcrumb, tuple(sections), active)
