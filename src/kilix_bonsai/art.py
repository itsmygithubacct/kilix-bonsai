"""The Kilix branding art, drawn with half-blocks and 256-colour pairs.

Two pixel rows per character cell, using `▀` with the upper pixel as the
foreground and the lower as the background. That trick is what lets a 64x64
sprite fit in 64x32 cells and still look like pixel art rather than a mosaic.

The sprite is authored at its largest useful size and *scaled down* to whatever
a pane can spare, by whole-number subsampling so the pixels stay square and
crisp. A launcher that only showed its art on a maximised window would show it
almost never.

Deliberately not the Kitty graphics protocol. Graphics would be sharper, but
this has to survive `ssh`, `tmux`, a plain terminal, and — most importantly for
this repository — rendering to plain text in a test. The asset itself is
generated from a PNG by `scripts/make-art.py`; nothing here needs Pillow.

The sprite is a flame kitten playing with a bonsai, which is the Kilix mascot
and the thing this tool manages, in one image.
"""
from __future__ import annotations

import curses
import json
import os
from functools import lru_cache

UPPER_HALF = "▀"

# Curses wants a pair per (fg, bg) combination and the number of pairs is
# finite, so pairs are allocated on demand and reused. 37 colours in the sprite
# means far fewer distinct vertical pairs than the theoretical maximum.
_PAIRS: dict[tuple[int, int], int] = {}
_NEXT_PAIR = 32          # leave the low pairs to whatever else the app uses


@lru_cache(maxsize=1)
def sprite() -> dict:
    """Load the pixel asset, or return an empty sprite when it is absent."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "kitten.json")
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"width": 0, "height": 0, "rows": []}
    return document


_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)
# The 16 system colours are terminal-configurable; these are the xterm
# defaults, used only to decide light from dark.
_SYSTEM = ((0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
           (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
           (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
           (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255))


def index_rgb(index: int) -> tuple[int, int, int]:
    """Return the approximate RGB of an xterm-256 index."""
    if index < 16:
        return _SYSTEM[index]
    if index >= 232:
        value = 8 + 10 * (index - 232)
        return value, value, value
    offset = index - 16
    return (_CUBE_LEVELS[offset // 36],
            _CUBE_LEVELS[(offset // 6) % 6],
            _CUBE_LEVELS[offset % 6])


def is_ink(index: int, threshold: int = 48) -> bool:
    """True when a palette index is light enough to read as drawn.

    Luminance, not index: the art's background is a near-black that lands in
    the grey ramp at index 232-ish, so any test on the index itself calls the
    darkest part of the image bright.
    """
    red, green, blue = index_rgb(index)
    return (299 * red + 587 * green + 114 * blue) // 1000 > threshold


def size(factor: int = 1) -> tuple[int, int]:
    """Return the sprite's size in character cells at a scale factor."""
    document = sprite()
    width = document["width"] // factor
    return width, ((document["height"] // factor) + 1) // 2


def fit(max_width: int, max_height: int) -> int:
    """Return the smallest whole-number reduction that fits the given box.

    Whole numbers only: a fractional scale on pixel art produces uneven pixel
    widths, which reads as a rendering fault rather than as a smaller sprite.
    """
    document = sprite()
    if not document.get("rows"):
        return 0
    for factor in (1, 2, 3, 4):
        width, cells = size(factor)
        if width <= max_width and cells <= max_height:
            return factor
    return 0


def _pair(foreground: int, background: int) -> int:
    global _NEXT_PAIR
    key = (foreground, background)
    if key in _PAIRS:
        return _PAIRS[key]
    if _NEXT_PAIR >= min(curses.COLOR_PAIRS, 256):
        return 0
    try:
        curses.init_pair(_NEXT_PAIR, foreground, background)
    except curses.error:
        return 0
    _PAIRS[key] = _NEXT_PAIR
    _NEXT_PAIR += 1
    return _PAIRS[key]


def usable() -> bool:
    """True when this terminal can show the sprite in colour."""
    try:
        return curses.has_colors() and curses.COLORS >= 256
    except curses.error:
        return False


def draw(surface, top: int, left: int, *, max_height: int | None = None,
         colour: bool = True) -> int:
    """Draw the sprite and return how many rows it used.

    Falls back to a monochrome silhouette when the terminal has no 256-colour
    support, and draws nothing at all when there is no room — a launcher that
    refused to start in an 80x24 pane because its decoration did not fit would
    be a bad trade.
    """
    document = sprite()
    rows = document.get("rows") or []
    if not rows:
        return 0
    height, width = surface.getmaxyx()
    room_height = max(0, height - top - 1)
    if max_height is not None:
        room_height = min(room_height, max_height)
    room_width = max(0, width - left - 1)
    factor = fit(room_width, room_height)
    if factor == 0:
        return 0
    span, cells = size(factor)
    for cell in range(cells):
        upper = rows[min(cell * 2 * factor, len(rows) - 1)]
        lower_index = min((cell * 2 + 1) * factor, len(rows) - 1)
        lower = rows[lower_index]
        for column in range(span):
            source = min(column * factor, len(upper) - 1)
            top_colour, bottom_colour = upper[source], lower[source]
            try:
                if colour:
                    attribute = curses.color_pair(
                        _pair(top_colour, bottom_colour))
                    surface.addstr(top + cell, left + column, UPPER_HALF,
                                   attribute)
                else:
                    # No 256-colour: show shape, not colour.
                    lit = is_ink(top_colour) or is_ink(bottom_colour)
                    surface.addstr(top + cell, left + column,
                                   UPPER_HALF if lit else " ")
            except curses.error:
                pass                       # last cell of the last row
    return cells


def as_text(factor: int = 1) -> str:
    """Render the sprite as plain text, for tests and `--screenshot`.

    Ink and space only. This exists so the launcher's layout can be asserted
    without a terminal, not to be pretty.
    """
    document = sprite()
    rows = document.get("rows") or []
    if not rows:
        return ""
    span, cells = size(factor)
    lines = []
    for cell in range(cells):
        upper = rows[min(cell * 2 * factor, len(rows) - 1)]
        lower = rows[min((cell * 2 + 1) * factor, len(rows) - 1)]
        line = "".join(
            UPPER_HALF if is_ink(upper[min(x * factor, len(upper) - 1)])
            or is_ink(lower[min(x * factor, len(lower) - 1)]) else " "
            for x in range(span))
        lines.append(line.rstrip())
    return "\n".join(lines)
