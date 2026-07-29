#!/usr/bin/env python3
"""Turn a generated PNG into the compact pixel asset the launcher renders.

The launcher draws with half-block characters and 256-colour curses pairs
rather than the Kitty graphics protocol. That is a deliberate trade: graphics
would be prettier, but it would not survive `ssh`, `tmux`, or a screenshot
test, and this repository's whole test approach is rendering screens to plain
text. Half-blocks give two pixel rows per character cell, which at 40x20 cells
is a 40x40 sprite — enough for the art to read, small enough to commit.

Requires Pillow, and is run by hand when the art changes. The asset it writes
is what ships; the launcher never needs Pillow.

    python3 scripts/make-art.py source.png src/kilix_bonsai/assets/kitten.json [size]
"""
from __future__ import annotations

import json
import sys

try:
    from PIL import Image
except ImportError:                                        # pragma: no cover
    sys.exit("this script needs Pillow: pip install pillow")

# 64x64 fills 64 columns by 32 rows of half-blocks. The renderer scales down
# to whatever a pane can spare, so this is the ceiling rather than the size it
# is always shown at — author big, let the terminal decide.
WIDTH = 64
HEIGHT = 64


def xterm256(rgb: tuple[int, int, int]) -> int:
    """Map an RGB triple to the nearest xterm-256 index.

    The 6x6x6 cube plus the grey ramp, picking whichever is closer — a plain
    cube quantisation turns the dark background into a muddy near-black rather
    than the true black the art was drawn on.
    """
    red, green, blue = rgb

    def cube(value: int) -> int:
        for index, edge in enumerate((0, 95, 135, 175, 215, 255)):
            if value <= edge:
                return index
        return 5

    levels = (0, 95, 135, 175, 215, 255)
    ci = [cube(c) for c in (red, green, blue)]
    cube_rgb = tuple(levels[i] for i in ci)
    cube_index = 16 + 36 * ci[0] + 6 * ci[1] + ci[2]

    grey = round((red + green + blue) / 3)
    step = max(0, min(23, round((grey - 8) / 10)))
    grey_value = 8 + 10 * step
    grey_index = 232 + step

    def distance(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    if distance(rgb, (grey_value,) * 3) < distance(rgb, cube_rgb):
        return grey_index
    return cube_index


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        sys.exit(__doc__.strip().splitlines()[-1].strip())
    source, target = argv[0], argv[1]
    global WIDTH, HEIGHT
    if len(argv) == 3:
        WIDTH = HEIGHT = int(argv[2])
    image = Image.open(source).convert("RGB")
    # NEAREST, not LANCZOS: the source is already pixel art, and a smooth
    # filter turns crisp edges into a halo of in-between colours that the
    # 256-colour quantisation then scatters.
    image = image.resize((WIDTH, HEIGHT), Image.NEAREST)
    rows = [[xterm256(image.getpixel((x, y))) for x in range(WIDTH)]
            for y in range(HEIGHT)]
    palette = sorted({index for row in rows for index in row})
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({
            "width": WIDTH, "height": HEIGHT,
            "colors": len(palette),
            "rows": rows,
        }, handle)
        handle.write("\n")
    print(f"{target}: {WIDTH}x{HEIGHT}, {len(palette)} colours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
