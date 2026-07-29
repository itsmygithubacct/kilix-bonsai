"""Portable text layout for the Kilix Bonsai interfaces.

The graphical path uses the same Tango desktop as ``kilix-tui``.  This is its
small-terminal floor: one quiet header, a breadcrumb, an optional section bar,
one content well, and a footer.  It deliberately contains no panel segments,
fictional node numbers, or decorative terminal art.
"""
from __future__ import annotations

import curses
from dataclasses import dataclass
from typing import Iterable

from . import screen


def _clip(value: str, width: int) -> str:
    return value[:max(0, width)]


@dataclass
class Page:
    """Measured text-mode shell shared by the launcher and task tools."""

    area: str
    breadcrumb: str
    sections: tuple[str, ...] = ()
    active: int = 0
    _size: tuple[int, int] = (24, 80)
    _extra_status: bool = False

    def measure(self, surface) -> None:
        self._size = surface.getmaxyx()

    def content_box(self) -> tuple[int, int, int, int]:
        height, width = self._size
        top = 4 if self.sections and height >= 9 else 3
        top += int(self._extra_status)
        bottom = max(top, height - 1)
        return top, 1 if width > 4 else 0, max(1, bottom - top), \
            max(1, width - (3 if width > 4 else 1))

    def render(self, surface, selected: int = 0, *, footer: str = "",
               status: str = "") -> None:
        del selected
        self.measure(surface)
        height, width = self._size
        if height <= 0 or width <= 0:
            return

        screen.write(surface, 0, 1 if width > 2 else 0,
                     _clip("KILIX TUI", width - 2), curses.A_BOLD)
        strap = f"BONSAI // {self.area.upper()}"
        if width >= len(strap) + 14:
            screen.write(surface, 0, width - len(strap) - 1, strap,
                         curses.A_DIM)
        if height > 1:
            screen.write(surface, 1, 0, "─" * max(0, width - 1))

        crumb_row = 2
        screen.write(surface, crumb_row, 1 if width > 2 else 0,
                     _clip(self.breadcrumb.upper(), width - 2),
                     curses.A_BOLD)
        status_fits = width >= len(status) + len(self.breadcrumb) + 5
        self._extra_status = bool(
            status and not status_fits
            and (4 if self.sections and height >= 9 else 3) < height - 1)
        if status and status_fits:
            screen.write(surface, crumb_row, width - len(status) - 1,
                         _clip(status, width - 2), curses.A_DIM)

        if self.sections and height >= 9:
            parts = []
            for index, label in enumerate(self.sections):
                parts.append(f"[{label}]" if index == self.active else label)
            screen.write(surface, 3, 1, _clip("  ".join(parts), width - 2),
                         curses.A_DIM)
        if self._extra_status:
            row = 4 if self.sections and height >= 9 else 3
            screen.write(surface, row, 1, _clip(status, width - 2),
                         curses.A_DIM)

        if height >= 2:
            screen.write(surface, height - 1, 1 if width > 2 else 0,
                         _clip(footer, width - 2), curses.A_DIM)


def page(area: str, breadcrumb: str, sections: Iterable[str] = (),
         active: int = 0) -> Page:
    return Page(area, breadcrumb, tuple(sections), active)
