"""Kilix-style pixel application shell for the Bonsai task interfaces.

``kilix-tui`` supplies the rasterizer, Kitty transport, drawing facade and
Tango palette.  This module supplies the application-shaped part it does not:
a reusable header/sidebar/content/footer layout and a raw-terminal event loop
for screens that update while workers stream text or produce media.

The dependency is optional.  ``run`` returns ``None`` when the session should
use its portable text renderer instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import importlib
import os
import re
import select
import shutil
import signal
import struct
import sys
import termios
import time
import tty
from typing import Any, Callable, Iterable

from . import desktop

Box = tuple[int, int, int, int]

_SEQUENCES = {
    b"\x1b[A": 259, b"\x1bOA": 259,
    b"\x1b[B": 258, b"\x1bOB": 258,
    b"\x1b[C": 261, b"\x1bOC": 261,
    b"\x1b[D": 260, b"\x1bOD": 260,
    b"\x1b[5~": 339, b"\x1b[6~": 338,
    b"\x1b[H": 262, b"\x1b[F": 360,
    b"\x1b[3~": 330,
}
_MOUSE = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")


def shared() -> tuple[Any, Any] | None:
    """Return the shared graphics and Tango modules when installed."""
    if desktop.shared() is None:
        return None
    try:
        return (importlib.import_module("kilix_desk.graphics"),
                importlib.import_module("kilix_desk.tango"))
    except ImportError:
        return None


def _wrap(value: str, columns: int) -> list[str]:
    """Word-wrap text while preserving deliberate line breaks."""
    if columns <= 0:
        return []
    lines: list[str] = []
    for paragraph in (value or "").split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            while len(word) > columns:
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:columns])
                word = word[columns:]
            if not current:
                current = word
            elif len(current) + len(word) + 1 <= columns:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


@dataclass
class Body:
    """Convenience drawing API handed to a screen-specific renderer."""

    draw: Any
    box: Box
    scale: float
    tango: Any
    graphics: Any
    hits: list[tuple[Any, Box]]

    def font(self, size: float, *, bold: bool = False) -> Any:
        return self.graphics.font_for(size * self.scale, bold=bold)

    def text(self, box: Box, value: str, *, size: float = 12,
             color: Any | None = None, bold: bool = False,
             align: str = "left", valign: str = "center") -> None:
        self.draw.text(box, value, self.font(size, bold=bold),
                       color or self.tango.SILVER,
                       align=align, valign=valign)

    def card(self, box: Box, *, selected: bool = False,
             danger: bool = False) -> None:
        if selected:
            color = self.tango.RED_DEEP if danger else self.tango.BLUE
            self.draw.rounded(box, max(4, int(8 * self.scale)), color)
            marker = self.tango.RED_BRIGHT if danger \
                else self.tango.BLUE_BRIGHT
            self.draw.fill((box[0] + 3, box[1] + 3,
                            box[0] + 3 + max(3, int(4 * self.scale)),
                            box[3] - 3), marker)
        else:
            self.draw.rounded(box, max(4, int(8 * self.scale)),
                              self.tango.ROW_ALT)

    def wrapped(self, box: Box, value: str, *, size: float = 12,
                color: Any | None = None, bold: bool = False,
                max_lines: int | None = None) -> list[str]:
        font = self.font(size, bold=bold)
        columns = max(1, (box[2] - box[0]) // max(1, 8 * font.scale))
        lines = _wrap(value, columns)
        if max_lines is not None:
            lines = lines[:max_lines]
        line_h = font.height + max(2, int(4 * self.scale))
        for index, line in enumerate(lines):
            top = box[1] + index * line_h
            if top + font.height > box[3]:
                break
            self.draw.text((box[0], top, box[2], top + font.height),
                           line, font, color or self.tango.SILVER,
                           valign="top")
        return lines


class Renderer:
    """Base class matching the visual grammar of ``kilix-tui``."""

    area = "BONSAI"

    def __init__(self, *, canvas_factory: Any | None = None) -> None:
        modules = shared()
        if modules is None:
            raise RuntimeError("kilix-tui-utils is not installed")
        self.graphics, self.tango = modules
        if canvas_factory is None:
            raster, library = self.graphics._soft_raster_backend()
            self._canvas = lambda width, height: raster.Canvas(
                width, height, library=library)
        else:
            self._canvas = canvas_factory
        self.hits: list[tuple[Any, Box]] = []

    def navigation(self, state: Any) -> Iterable[tuple[str, Any]]:
        raise NotImplementedError

    def active_navigation(self, state: Any) -> int:
        return 0

    def breadcrumb(self, state: Any) -> str:
        return self.area

    def footer(self, state: Any) -> str:
        return "Ctrl-Q quit"

    def status(self, state: Any) -> str:
        return str(getattr(state, "status", "") or "")

    def snapshot(self, state: Any) -> Any:
        """Cheap render identity; concrete screens add their live content."""
        return (
            getattr(state, "status", ""),
            getattr(state, "view", ""),
            getattr(state, "show_help", False),
            getattr(state, "selected", 0),
        )

    def body(self, body: Body, state: Any) -> None:
        raise NotImplementedError

    def activate(self, action: Any, state: Any,
                 handle: Callable[[int, Any], bool]) -> bool:
        if action is None:
            return True
        if callable(action):
            result = action(state)
            return True if result is None else bool(result)
        return handle(int(action), state)

    def render(self, state: Any, columns: int, rows: int,
               pixel_size: tuple[int, int], *,
               clock: str | None = None) -> Any:
        width, height = pixel_size
        canvas = self._canvas(width, height)
        self.hits = []
        try:
            draw = self.graphics.Draw(canvas)
            scale = max(0.72, min(1.6, width / 1150.0,
                                  height / 640.0))
            self._background(draw, width, height)
            header_h = max(40, int(58 * scale))
            footer_h = max(18, int(24 * scale))
            self._header(draw, width, header_h, scale,
                         clock or time.strftime("%H:%M"))
            margin = max(8, int(14 * scale))
            sidebar_w = int(210 * scale) if width >= 560 else 0
            top = header_h + margin
            bottom = height - footer_h - margin // 2
            if sidebar_w:
                self._sidebar(draw, (margin, top, margin + sidebar_w, bottom),
                              state, scale)
                left = margin + sidebar_w + margin
            else:
                left = margin
            content = (left, top, width - margin, bottom)
            inner = self._content(draw, content, state, scale)
            self.body(Body(draw, inner, scale, self.tango, self.graphics,
                           self.hits), state)
            self._footer(draw, width, height, footer_h, state, scale)
            rgb = canvas.rgb_bytes()
        finally:
            canvas.close()
        return self.graphics.GraphicsFrame(
            rgb=rgb, width=width, height=height,
            columns=max(1, columns), rows=max(1, rows),
            content_key=f"kilix-bonsai-{self.area.lower()}")

    def _background(self, draw: Any, width: int, height: int) -> None:
        steps = 24
        band = max(1, height // steps + 1)
        for index in range(steps + 1):
            ratio = index / steps
            color = tuple(
                int(self.tango.BG_TOP[channel] * (1 - ratio)
                    + self.tango.BG_BOTTOM[channel] * ratio)
                for channel in range(3))
            y = index * band
            draw.fill((0, y, width, min(height, y + band)), color)

    def _header(self, draw: Any, width: int, height: int, scale: float,
                clock: str) -> None:
        draw.fill((0, 0, width, height), self.tango.HEADER)
        draw.fill((0, height - max(2, int(3 * scale)), width, height),
                  self.tango.BLUE)
        pad = max(10, int(18 * scale))
        title = self.graphics.font_for(44 * scale, bold=True)
        small = self.graphics.font_for(11 * scale)
        draw.text((pad, 0, width // 2, height - 4), "KILIX TUI", title,
                  self.tango.WHITE)
        strap = f"BONSAI · {self.area.upper()}"
        draw.text((pad + title.width("KILIX TUI") + pad, 0,
                   width - pad - small.width(clock) - pad, height - 4),
                  strap, small, self.tango.GREY)
        clock_font = self.graphics.font_for(26 * scale, bold=True)
        draw.text((width // 2, 0, width - pad, height - 4), clock,
                  clock_font, self.tango.SILVER, align="right")

    def _sidebar(self, draw: Any, box: Box, state: Any,
                 scale: float) -> None:
        items = list(self.navigation(state))
        if not items:
            return
        left, top, right, bottom = box
        active = max(0, min(self.active_navigation(state), len(items) - 1))
        label = self.graphics.font_for(24 * scale, bold=True)
        index_font = self.graphics.font_for(10 * scale)
        row_h = max(28, min(int(52 * scale),
                            (bottom - top) // max(1, len(items))))
        for index, (name, action) in enumerate(items):
            row_top = top + index * row_h
            row = (left, row_top + 2, right, row_top + row_h - 4)
            self.hits.append((action, row))
            selected = index == active
            if selected:
                draw.rounded(row, max(4, int(9 * scale)), self.tango.BLUE)
                draw.fill((left, row[1] + 3,
                           left + max(3, int(4 * scale)), row[3] - 3),
                          self.tango.BLUE_BRIGHT)
            pad = max(10, int(16 * scale))
            draw.text((row[0] + pad, row[1], row[2] - pad, row[3]),
                      name, label,
                      self.tango.WHITE if selected else self.tango.SILVER)
            draw.text((row[0] + pad, row[1], row[2] - pad, row[3]),
                      str(index + 1), index_font,
                      self.tango.WHITE if selected else self.tango.GREY_DARK,
                      align="right")

    def _content(self, draw: Any, box: Box, state: Any,
                 scale: float) -> Box:
        draw.panel(box, radius=max(6, int(14 * scale)))
        left, top, right, bottom = box
        pad = max(10, int(20 * scale))
        head = self.graphics.font_for(12 * scale, bold=True)
        draw.text((left + pad, top + pad // 2, right - pad,
                   top + pad // 2 + head.height + 4),
                  self.breadcrumb(state).upper(), head,
                  self.tango.BLUE_BRIGHT)
        status = self.status(state)
        if status:
            draw.text((left + pad, top + pad // 2, right - pad,
                       top + pad // 2 + head.height + 4),
                      status, self.graphics.font_for(10 * scale),
                      self.tango.GREY, align="right")
        rule = top + pad // 2 + head.height + max(6, int(8 * scale))
        draw.hline(left + pad, right - pad, rule, self.tango.CARD_EDGE,
                   max(1, int(scale)))
        return (left + pad, rule + max(6, int(10 * scale)),
                right - pad, bottom - pad)

    def _footer(self, draw: Any, width: int, height: int, footer_h: int,
                state: Any, scale: float) -> None:
        top = height - footer_h
        draw.fill((0, top, width, height), self.tango.HEADER)
        font = self.graphics.font_for(10 * scale)
        pad = max(10, int(18 * scale))
        draw.text((pad, top, width - pad, height), self.footer(state), font,
                  self.tango.GREY)
        draw.text((pad, top, width - pad, height), "KILIX BONSAI", font,
                  self.tango.GREY_DARK, align="right")


class TerminalSession:
    def __init__(self, title: str) -> None:
        self.title = title
        self.fd = sys.stdin.fileno()
        self._attributes: list[Any] | None = None

    def __enter__(self) -> "TerminalSession":
        self._attributes = termios.tcgetattr(self.fd)
        attributes = list(self._attributes)
        attributes[0] &= ~termios.IXON
        termios.tcsetattr(self.fd, termios.TCSANOW, attributes)
        tty.setcbreak(self.fd)
        sys.stdout.write(
            "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H"
            "\x1b[?1002h\x1b[?1006h\x1b[?1016h"
            f"\x1b]2;{self.title}\x07")
        sys.stdout.flush()
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            if self._attributes is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN,
                                  self._attributes)
        finally:
            sys.stdout.write(
                "\x1b[?1016l\x1b[?1006l\x1b[?1002l"
                "\x1b[0m\x1b[?25h\x1b[?1049l\x1b]2;\x07")
            sys.stdout.flush()


class Application:
    """Raw input, complete frames and worker-friendly timed redraws."""

    def __init__(self, renderer: Renderer, state: Any,
                 handle: Callable[[int, Any], bool], tick_ms: int) -> None:
        self.renderer = renderer
        self.state = state
        self.handle = handle
        self.tick = max(0.03, tick_ms / 1000)
        self.running = True
        self.display: Any | None = None
        self.clear = True
        self.cells = (100, 30)
        self.render_px = (1000, 600)
        self.raw_px = self.render_px
        self.pending = b""

    def _stop(self, *_args: object) -> None:
        self.running = False

    def _resize(self, *_args: object) -> None:
        self.clear = True

    def _draw(self) -> None:
        graphics = self.renderer.graphics
        size = shutil.get_terminal_size((100, 30))
        pixels = graphics.terminal_pixel_size(
            sys.stdout.fileno(), size.columns, size.lines)
        self.cells = (size.columns, size.lines)
        self.render_px = pixels
        try:
            packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ,
                                 b"\0" * 8)
            _rows, _columns, raw_w, raw_h = struct.unpack("HHHH", packed)
            self.raw_px = (raw_w, raw_h) if raw_w and raw_h else pixels
        except (OSError, struct.error):
            self.raw_px = pixels
        frame = self.renderer.render(self.state, size.columns, size.lines,
                                     pixels)
        if self.clear:
            sys.stdout.write("\x1b[2J\x1b[H")
            self.display.invalidate()
        self.display.present(frame, force_full=self.clear)
        sys.stdout.flush()
        self.clear = False

    def _point(self, x: int, y: int) -> tuple[int, int]:
        columns, rows = self.cells
        render_w, render_h = self.render_px
        raw_w, raw_h = self.raw_px
        if x > columns + 1 or y > rows + 1:
            return (int(x * render_w / max(1, raw_w)),
                    int(y * render_h / max(1, raw_h)))
        return (int((x - 0.5) * render_w / max(1, columns)),
                int((y - 0.5) * render_h / max(1, rows)))

    def _click(self, x: int, y: int) -> None:
        px, py = self._point(x, y)
        for action, (left, top, right, bottom) in reversed(
                self.renderer.hits):
            if left <= px <= right and top <= py <= bottom:
                self.running = self.renderer.activate(
                    action, self.state, self.handle) and self.running
                return

    def _key(self, key: int) -> None:
        self.running = self.handle(key, self.state) and self.running

    def _bytes(self, incoming: bytes) -> None:
        data = self.pending + incoming
        self.pending = b""
        while data and self.running:
            if data.startswith(b"\x1b[<") and not _MOUSE.match(data):
                if len(data) < 24:
                    self.pending = data
                    return
            mouse = _MOUSE.match(data)
            if mouse:
                button = int(mouse.group(1))
                if button in (64, 65):
                    self._key(259 if button == 64 else 258)
                elif (mouse.group(4) == b"M" and button & 3 == 0
                      and not button & 32):
                    self._click(int(mouse.group(2)), int(mouse.group(3)))
                data = data[mouse.end():]
                continue
            match = next(((sequence, key)
                          for sequence, key in _SEQUENCES.items()
                          if data.startswith(sequence)), None)
            if match:
                sequence, key = match
                self._key(key)
                data = data[len(sequence):]
                continue
            if data.startswith(b"\x1b") and len(data) == 1:
                self._key(27)
                return
            if data.startswith(b"\x1b") and data[1:2] in (b"[", b"O"):
                end = 2
                while end < len(data) and not (
                        data[end:end + 1].isalpha()
                        or data[end:end + 1] == b"~"):
                    end += 1
                data = data[min(len(data), end + 1):]
                continue
            first = data[0]
            if first < 128:
                data = data[1:]
                self._key(10 if first in (10, 13) else first)
                continue
            length = (2 if first & 0xE0 == 0xC0 else
                      3 if first & 0xF0 == 0xE0 else
                      4 if first & 0xF8 == 0xF0 else 1)
            if len(data) < length:
                self.pending = data
                return
            try:
                value = data[:length].decode("utf-8")
            except UnicodeDecodeError:
                data = data[1:]
                continue
            data = data[length:]
            for character in value:
                self._key(ord(character))

    def run(self) -> int:
        previous: dict[int, Any] = {}
        for sig, callback in ((signal.SIGHUP, self._stop),
                              (signal.SIGINT, self._stop),
                              (signal.SIGTERM, self._stop),
                              (signal.SIGWINCH, self._resize)):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, callback)
        try:
            with TerminalSession(f"Kilix Bonsai {self.renderer.area.title()}"):
                self.display = self.renderer.graphics.KittyDisplay(
                    sys.stdout, max_fps=min(20.0, 1.0 / self.tick))
                previous_frame: Any = None
                try:
                    while (self.running
                           and not getattr(self.state, "finished", False)):
                        identity = (time.strftime("%H:%M"),
                                    self.renderer.snapshot(self.state))
                        if self.clear or identity != previous_frame:
                            self._draw()
                            previous_frame = identity
                        deadline = self.display.next_deadline
                        timeout = self.tick
                        if deadline is not None:
                            timeout = min(timeout, max(
                                0.0, deadline - time.monotonic()))
                        try:
                            readable, _, _ = select.select(
                                [sys.stdin], [], [], timeout)
                        except InterruptedError:
                            readable = []
                        if readable:
                            try:
                                data = os.read(sys.stdin.fileno(), 512)
                            except OSError:
                                data = b""
                            if data:
                                self._bytes(data)
                        result = self.display.flush()
                        if getattr(result, "emitted", False):
                            sys.stdout.flush()
                finally:
                    self.display.close()
                    self.display = None
            return 0
        finally:
            for sig, callback in previous.items():
                signal.signal(sig, callback)


def run(argv: list[str], renderer: Renderer | Callable[[], Renderer],
        state: Any,
        handle: Callable[[int, Any], bool], *, tick_ms: int,
        command: str) -> int | None:
    """Run explicitly requested pixels, otherwise use the canonical text TUI."""
    if ("--text" in argv or "--screenshot" in argv
            or os.environ.get("KILIX_TUI_HEADLESS") == "1"):
        return None
    forced = "--graphics" in argv
    if not forced:
        return None
    modules = shared()
    if modules is None:
        if forced:
            print(f"{command}: graphics unavailable: "
                  "kilix-tui-utils is not installed", file=sys.stderr)
            return 1
        return None
    graphics, _tango = modules
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        if forced:
            print(f"{command}: graphics unavailable: standard input and "
                  "output must both be terminals", file=sys.stderr)
            return 1
        return None
    if not forced and os.environ.get("KILIX_TUI_GRAPHICS") == "0":
        return None
    if not forced and not graphics.kitty_graphics_likely():
        return None
    ready, detail = graphics.available()
    if not ready:
        if forced:
            print(f"{command}: graphics unavailable: {detail}",
                  file=sys.stderr)
            return 1
        return None
    try:
        selected = renderer() if callable(renderer) else renderer
        return Application(selected, state, handle, tick_ms).run()
    except (graphics.GraphicsUnavailable, termios.error, RuntimeError) as error:
        if forced:
            print(f"{command}: graphics unavailable: {error}",
                  file=sys.stderr)
            return 1
        return None
