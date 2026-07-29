"""The curses event loop, and the headless surface the tests render into.

This is deliberately the same shape and the same keymap as `kilix_tui`, the
core the other Kilix terminal utilities share — a user who knows one of those
tools can drive this one without reading anything. It is a small copy rather
than an import because this repository ships a *model store*, and making a
multi-gigabyte download depend on a second checkout being present to draw a
list would be the wrong trade. If these two ever need to diverge, that is the
signal to move this file into the shared core instead.

`render(surface, state)` is all a screen has to supply. Rendering into
`TextSurface` instead of a terminal is what makes every screen here assertable
as plain text, with no pty and no display.
"""
from __future__ import annotations

import curses
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

QUIT = frozenset({ord("q"), ord("Q"), 27})           # q, Esc
HELP = frozenset({ord("?")})
REFRESH = frozenset({ord("r"), ord("R"), ord("l") - 96})
UP = frozenset({ord("k"), curses.KEY_UP})
DOWN = frozenset({ord("j"), curses.KEY_DOWN})
BACK = frozenset({ord("h"), curses.KEY_LEFT, 27})
SELECT = frozenset({ord("\n"), ord("\r"), ord(" "), curses.KEY_ENTER,
                    ord("l"), curses.KEY_RIGHT})
YES = frozenset({ord("y"), ord("Y")})


def is_quit(key: int) -> bool:
    return key in QUIT


def direction(key: int) -> int:
    """Return -1 for up, 1 for down, 0 otherwise."""
    if key in UP:
        return -1
    if key in DOWN:
        return 1
    return 0


class Surface(Protocol):
    """The subset of a curses window a screen is allowed to use."""

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None: ...
    def getmaxyx(self) -> tuple[int, int]: ...


@dataclass
class TextSurface:
    """A capture target used by `--screenshot` and by the tests."""

    height: int = 24
    width: int = 80
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lines:
            self.lines = [" " * self.width for _ in range(self.height)]

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        del attr
        if not (0 <= y < self.height) or x < 0:
            return
        row = self.lines[y]
        text = text[: max(0, self.width - x)]
        self.lines[y] = (row[:x] + text + row[x + len(text):])[: self.width]

    def getmaxyx(self) -> tuple[int, int]:
        return self.height, self.width

    def __str__(self) -> str:
        return "\n".join(line.rstrip() for line in self.lines).rstrip("\n")


def render_to_text(render: Callable[[Any, Any], None], state: Any, *,
                   height: int = 24, width: int = 80) -> str:
    """Render one frame headlessly and return it as text."""
    surface = TextSurface(height=height, width=width)
    render(surface, state)
    return str(surface)


def write(surface: Surface, y: int, x: int, text: str) -> None:
    """Draw one clipped line. Every screen goes through this.

    curses raises when a write reaches the last cell of the last row, and a
    pane can be any size, so clipping is not optional and should not be each
    screen's problem.
    """
    height, width = surface.getmaxyx()
    if not (0 <= y < height) or x >= width:
        return
    try:
        surface.addstr(y, x, text[: max(0, width - x - 1)])
    except curses.error:                                 # pragma: no cover
        pass


def run(render: Callable[[Any, Any], None], state: Any, *,
        handle: Callable[[int, Any], bool], tick_ms: int | None = None) -> int:
    """Run until `handle` returns False.

    `tick_ms` makes the loop wake on its own, which is what lets a screen
    repaint while a background thread is still producing — tokens arriving from
    a model, or a transcript growing chunk by chunk.

    `KILIX_TUI_HEADLESS=1` prints one frame and exits, which is how the other
    Kilix tools are smoke-tested and how these are too.
    """
    if os.environ.get("KILIX_TUI_HEADLESS") == "1":
        print(render_to_text(render, state))
        return 0

    def _loop(stdscr: Any) -> int:
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
        if tick_ms:
            stdscr.timeout(tick_ms)
        state.stdscr = stdscr
        while True:
            stdscr.erase()
            curses.curs_set(0)
            render(stdscr, state)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (-1, curses.KEY_RESIZE):
                continue
            if not handle(key, state):
                return 0

    return curses.wrapper(_loop)


def screenshot_argv(argv: list[str]) -> str | None:
    """Return the path for `--screenshot PATH`, or None."""
    if "--screenshot" in argv:
        index = argv.index("--screenshot")
        if index + 1 < len(argv):
            return argv[index + 1]
    return None
