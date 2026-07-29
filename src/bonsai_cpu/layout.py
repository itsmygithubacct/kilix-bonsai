"""Portable Kilix-style shell for the CPU chat's text fallback."""
from __future__ import annotations

import curses

from . import screen

SECTIONS = ("Chats", "Conversation", "Models", "Settings", "Help")


def shell(surface, *, breadcrumb: str, active: int, status: str,
          footer: str) -> tuple[int, int, int]:
    """Draw the shared header/navigation/footer and return the content well.

    The return value is ``(top, bottom, width)`` with ``bottom`` exclusive.
    Navigation folds away before content does on a very short terminal.
    """
    height, width = surface.getmaxyx()
    if height <= 0 or width <= 0:
        return 0, 0, max(1, width)
    screen.write(surface, 0, 1 if width > 2 else 0, "KILIX TUI",
                 curses.A_BOLD)
    strap = "BONSAI // CPU CHAT"
    if width >= len(strap) + 14:
        screen.write(surface, 0, width - len(strap) - 1, strap,
                     curses.A_DIM)
    if height > 1:
        screen.write(surface, 1, 0, "─" * max(0, width - 1))

    nav = height >= 9
    crumb_row = 3 if nav else 2
    if nav:
        parts = [f"[{name}]" if index == active else name
                 for index, name in enumerate(SECTIONS)]
        screen.write(surface, 2, 1, "  ".join(parts), curses.A_DIM)
    status_row = crumb_row
    separate_status = bool(
        status and width < len(status) + len(breadcrumb) + 5
        and crumb_row + 1 < height - 1)
    if crumb_row < height - 1:
        screen.write(surface, crumb_row, 1, breadcrumb.upper(),
                     curses.A_BOLD)
        if status and not separate_status:
            screen.write(surface, crumb_row, width - len(status) - 1,
                         status, curses.A_DIM)
        elif separate_status:
            status_row += 1
            screen.write(surface, status_row, 1, status, curses.A_DIM)
    if height > 1:
        screen.write(surface, height - 1, 1 if width > 2 else 0, footer,
                     curses.A_DIM)
    return min(height - 1, status_row + 1), max(0, height - 1), width
