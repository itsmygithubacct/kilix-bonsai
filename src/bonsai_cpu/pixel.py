"""Pixel chat matching the Tango desktop in ``kilix-tui``.

The model runtime remains entirely independent.  When the optional shared
Kilix raster stack is present, this module draws complete RGB frames and
presents them through Kitty graphics; otherwise :mod:`bonsai_cpu.chat_tui`
uses its portable text renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import termios
import time
from typing import Any, Callable

from . import models, widgets

Box = tuple[int, int, int, int]
SECTIONS = ("Chats", "Conversation", "Models", "Settings", "Help")
HELP_ROWS = (
    ("Enter", "send"),
    ("Esc", "stop generating, or back"),
    ("Ctrl-O", "switch model"),
    ("Ctrl-T", "toggle thinking (27B)"),
    ("Ctrl-F", "fold or reveal thinking"),
    ("Ctrl-P", "system prompt and sampling"),
    ("Ctrl-N", "new conversation"),
    ("Ctrl-R", "rename conversation"),
    ("PgUp/PgDn", "scroll transcript"),
    ("Ctrl-Q", "quit"),
)


def shared() -> tuple[Any, Any] | None:
    """Find the optional shared graphics package without requiring install."""
    source_home = os.environ.get("GPU_TERMINAL_SOURCE_HOME") or os.path.join(
        os.path.expanduser("~"), "gpu_terminal")
    repository = Path(__file__).resolve().parents[2]
    homes = (
        os.environ.get("KILIX_TUI_UTILS_HOME", ""),
        os.path.join(os.path.abspath(os.path.expanduser(source_home)),
                     "kilix-tui-utils"),
        str(repository.parent / "kilix-tui-utils"),
    )
    for home in homes:
        source = os.path.join(home, "src") if home else ""
        if source and os.path.isdir(os.path.join(source, "kilix_desk")):
            if source not in sys.path:
                sys.path.insert(0, source)
            break
    try:
        return (importlib.import_module("kilix_desk.graphics"),
                importlib.import_module("kilix_desk.tango"))
    except ImportError:
        return None


def _wrap(value: str, columns: int) -> list[str]:
    return widgets.wrap(value, max(1, columns))


@dataclass
class Body:
    draw: Any
    box: Box
    scale: float
    graphics: Any
    tango: Any
    hits: list[tuple[Any, Box]]

    def font(self, size: float, *, bold: bool = False) -> Any:
        return self.graphics.font_for(size * self.scale, bold=bold)

    def text(self, box: Box, value: str, *, size: float = 12,
             bold: bool = False, color: Any | None = None,
             align: str = "left", valign: str = "center") -> None:
        self.draw.text(box, value, self.font(size, bold=bold),
                       color or self.tango.SILVER,
                       align=align, valign=valign)

    def card(self, box: Box, *, selected: bool = False,
             danger: bool = False) -> None:
        color = (self.tango.RED_DEEP if danger else self.tango.BLUE) \
            if selected else self.tango.ROW_ALT
        self.draw.rounded(box, max(4, int(8 * self.scale)), color)
        if selected:
            accent = self.tango.RED_BRIGHT if danger \
                else self.tango.BLUE_BRIGHT
            self.draw.fill((box[0] + 3, box[1] + 3,
                            box[0] + 3 + max(3, int(4 * self.scale)),
                            box[3] - 3), accent)

    def wrapped(self, box: Box, value: str, *, size: float = 12,
                color: Any | None = None, max_lines: int | None = None) -> None:
        font = self.font(size)
        columns = max(1, (box[2] - box[0]) // max(1, 8 * font.scale))
        lines = _wrap(value, columns)
        if max_lines is not None:
            lines = lines[:max_lines]
        line_h = font.height + max(3, int(5 * self.scale))
        for index, line in enumerate(lines):
            y = box[1] + index * line_h
            if y + font.height > box[3]:
                break
            self.text((box[0], y, box[2], y + line_h), line, size=size,
                      color=color)


class ChatRenderer:
    """All CPU chat views inside one Kilix desktop frame."""

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

    @staticmethod
    def active(state: Any) -> int:
        return {"list": 0, "chat": 1, "picker": 2,
                "params": 3, "help": 4}.get(state.view, 1)

    @staticmethod
    def breadcrumb(state: Any) -> str:
        if state.error:
            return "Chat  /  Error"
        if state.view == "list":
            return "Chats  /  Saved conversations"
        if state.view == "picker":
            return "Models  /  Switch model"
        if state.view == "params":
            return "Settings  /  Conversation"
        if state.view == "help":
            return "Help  /  Keyboard"
        if state.convo is None:
            return "Conversation"
        return f"Conversation  /  {state.spec()['title']}"

    @staticmethod
    def footer(state: Any) -> str:
        if state.error:
            return "r retry  ·  q quit  ·  any other key back"
        if state.view == "list":
            return "Enter open  ·  n new  ·  d delete  ·  Ctrl-Q quit"
        if state.view == "picker":
            return "Enter switch  ·  Esc back"
        if state.view == "params":
            return ("Enter commit  ·  Esc cancel" if state.param_editor
                    else "Enter edit  ·  Esc back")
        if state.view == "help":
            return "any key returns"
        if state.streaming:
            return "generating  ·  Esc stop  ·  Ctrl-Q quit"
        return ("Enter send  ·  Ctrl-O model  ·  Ctrl-T think  ·  "
                "Ctrl-P settings  ·  Ctrl-F fold  ·  ? help")

    @staticmethod
    def status(state: Any) -> str:
        if state.loading:
            return state.loading
        if state.status:
            return state.status
        if state.view == "chat" and state.convo is not None:
            # Import here to avoid a module cycle during chat_tui startup.
            from .chat_tui import status_line
            return status_line(state)
        return ""

    @staticmethod
    def snapshot(state: Any) -> Any:
        conversation = state.convo
        messages = () if conversation is None else tuple(
            (message.get("role"), message.get("content"),
             message.get("reasoning"), message.get("reasoning_tokens"),
             tuple(sorted((message.get("stats") or {}).items())))
            for message in conversation.messages)
        conversations = tuple(
            (item.path, item.name, item.model_id, item.updated,
             len(item.messages)) for item in state.conversations)
        return (
            state.view, state.error, state.loading, state.status,
            state.streaming, state.show_thinking, state.selected,
            state.pending_delete, state.param_index,
            state.editor.text, state.editor.cursor,
            None if state.renaming is None else (
                state.renaming.text, state.renaming.cursor),
            None if state.param_editor is None else (
                state.param_editor.text, state.param_editor.cursor),
            state.scroll.offset, state.scroll.following,
            tuple(sorted((state.progress or {}).items())),
            tuple(sorted((state.stats or {}).items())),
            conversations,
            None if conversation is None else (
                conversation.path, conversation.name, conversation.model_id,
                conversation.think,
                tuple(sorted(conversation.params.items()))),
            messages,
        )

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
            body = Body(draw, inner, scale, self.graphics, self.tango,
                        self.hits)
            self._body(body, state)
            if state.loading:
                self._loading(body, state.loading)
            self._footer(draw, width, height, footer_h, state, scale)
            rgb = canvas.rgb_bytes()
        finally:
            canvas.close()
        return self.graphics.GraphicsFrame(
            rgb=rgb, width=width, height=height,
            columns=max(1, columns), rows=max(1, rows),
            content_key="bonsai-cpu-chat")

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
        draw.text((pad + title.width("KILIX TUI") + pad, 0,
                   width - pad - small.width(clock) - pad, height - 4),
                  "BONSAI  //  CPU CHAT", small, self.tango.GREY)
        clock_font = self.graphics.font_for(26 * scale, bold=True)
        draw.text((width // 2, 0, width - pad, height - 4), clock,
                  clock_font, self.tango.SILVER, align="right")

    def _sidebar(self, draw: Any, box: Box, state: Any,
                 scale: float) -> None:
        left, top, right, bottom = box
        active = self.active(state)
        label = self.graphics.font_for(24 * scale, bold=True)
        number = self.graphics.font_for(10 * scale)
        row_h = max(28, min(int(52 * scale),
                            (bottom - top) // len(SECTIONS)))
        actions = ("list", "chat", "picker", "params", "help")
        for index, name in enumerate(SECTIONS):
            y = top + index * row_h
            row = (left, y + 2, right, y + row_h - 4)
            self.hits.append((("view", actions[index]), row))
            selected = index == active
            if selected:
                draw.rounded(row, max(4, int(9 * scale)), self.tango.BLUE)
                draw.fill((left, row[1] + 3,
                           left + max(3, int(4 * scale)), row[3] - 3),
                          self.tango.BLUE_BRIGHT)
            pad = max(10, int(16 * scale))
            draw.text((left + pad, row[1], right - pad, row[3]), name,
                      label, self.tango.WHITE if selected
                      else self.tango.SILVER)
            draw.text((left + pad, row[1], right - pad, row[3]),
                      str(index + 1), number,
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
                  self.tango.RED_BRIGHT if state.error
                  else self.tango.BLUE_BRIGHT)
        status = self.status(state)
        if status:
            draw.text((left + pad, top + pad // 2, right - pad,
                       top + pad // 2 + head.height + 4), status,
                      self.graphics.font_for(10 * scale), self.tango.GREY,
                      align="right")
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
        draw.text((pad, top, width - pad, height), "BONSAI CPU", font,
                  self.tango.GREY_DARK, align="right")

    def _body(self, body: Body, state: Any) -> None:
        if state.error:
            self._error(body, state.error)
        elif state.view == "list":
            self._list(body, state)
        elif state.view == "picker":
            self._picker(body, state)
        elif state.view == "params":
            self._params(body, state)
        elif state.view == "help":
            self._help(body)
        else:
            self._chat(body, state)

    def _list(self, body: Body, state: Any) -> None:
        left, top, right, bottom = body.box
        rows = [(None, "New conversation", "", "")]
        for conversation in state.conversations:
            spec = models.MODELS.get(conversation.model_id, {})
            rows.append((conversation,
                         conversation.name or "Untitled",
                         spec.get("title", conversation.model_id),
                         f"{len(conversation.messages) // 2} turns  ·  "
                         f"{conversation.updated[:10]}"))
        row_h = max(38, int(52 * body.scale))
        capacity = max(1, (bottom - top) // row_h)
        first = max(0, min(state.selected - capacity + 1,
                           len(rows) - capacity))
        for line, (_conversation, name, model, detail) in enumerate(
                rows[first:first + capacity]):
            index = first + line
            y = top + line * row_h
            row = (left, y + 2, right, y + row_h - 3)
            selected = index == state.selected
            body.card(row, selected=selected)
            body.hits.append((("select", index), row))
            pad = max(10, int(14 * body.scale))
            body.text((left + pad, row[1], right - pad, row[3]), name,
                      size=14, bold=selected,
                      color=body.tango.WHITE if selected
                      else body.tango.SILVER)
            hint = "NEW" if index == 0 else f"{model}  ·  {detail}"
            body.text((left + pad, row[1], right - pad, row[3]), hint,
                      size=10, color=body.tango.WHITE if selected
                      else body.tango.GREY, align="right")

    def _picker(self, body: Body, state: Any) -> None:
        left, top, right, bottom = body.box
        ids = sorted(models.MODELS)
        row_h = max(42, int(58 * body.scale))
        for index, model_id in enumerate(ids):
            y = top + index * row_h
            if y + row_h > bottom:
                break
            spec = models.MODELS[model_id]
            row = (left, y + 2, right, y + row_h - 3)
            selected = index == state.selected
            body.card(row, selected=selected)
            body.hits.append((("select", index), row))
            pad = max(10, int(14 * body.scale))
            current = state.convo is not None \
                and model_id == state.convo.model_id
            body.text((left + pad, row[1], right - pad, row[3]),
                      spec["title"], size=16, bold=selected,
                      color=body.tango.WHITE if selected
                      else body.tango.SILVER)
            hint = "CURRENT" if current else spec.get("speed_note", "")
            body.text((left + pad, row[1], right - pad, row[3]), hint,
                      size=10, color=body.tango.WHITE if selected
                      else body.tango.GREY, align="right")

    def _params(self, body: Body, state: Any) -> None:
        from .chat_tui import PARAM_FIELDS
        left, top, right, bottom = body.box
        row_h = max(38, int(52 * body.scale))
        for index, field in enumerate(PARAM_FIELDS):
            y = top + index * row_h
            if y + row_h > bottom:
                break
            row = (left, y + 2, right, y + row_h - 3)
            selected = index == state.param_index
            body.card(row, selected=selected)
            body.hits.append((("select", index), row))
            pad = max(10, int(14 * body.scale))
            split = left + max(130, int((right - left) * 0.25))
            body.text((left + pad, row[1], split, row[3]), field.upper(),
                      size=10, bold=True,
                      color=body.tango.WHITE if selected
                      else body.tango.GREY)
            if state.param_editor is not None and selected:
                value = state.param_editor.view(80)[0]
            else:
                value = str(state.convo.params[field])
            body.text((split, row[1], right - pad, row[3]), value,
                      size=12, color=body.tango.WHITE if selected
                      else body.tango.SILVER)

    def _help(self, body: Body) -> None:
        left, top, right, bottom = body.box
        row_h = max(25, int(36 * body.scale))
        for index, (key, action) in enumerate(HELP_ROWS):
            y = top + index * row_h
            if y + row_h > bottom:
                break
            if index % 2:
                body.card((left, y + 1, right, y + row_h - 1))
            split = left + max(120, int((right - left) * 0.24))
            body.text((left + 8, y, split, y + row_h), key, size=11,
                      bold=True, color=body.tango.BLUE_BRIGHT)
            body.text((split, y, right - 8, y + row_h), action, size=11)

    def _chat(self, body: Body, state: Any) -> None:
        left, top, right, bottom = body.box
        gap = max(8, int(12 * body.scale))
        input_h = max(42, int(54 * body.scale))
        input_box = (left, bottom - input_h, right, bottom)
        transcript = (left, top, right, input_box[1] - gap)
        font = body.font(12)
        line_h = font.height + max(3, int(5 * body.scale))
        columns = max(8, (right - left - 24)
                      // max(1, 8 * font.scale))
        lines: list[tuple[str, str]] = []
        total = len(state.convo.messages)
        for index, message in enumerate(state.convo.messages):
            if message["role"] == "user":
                lines.append(("role", "YOU"))
            else:
                lines.append(("role", state.spec()["title"].upper()))
                reasoning = message.get("reasoning") or ""
                if reasoning:
                    tokens = message.get("reasoning_tokens",
                                         len(reasoning.split()))
                    live = state.streaming and index == total - 1
                    label = (f"THINKING  {tokens} TOKENS"
                             if live else f"THOUGHT FOR {tokens} TOKENS")
                    lines.append(("thinking", label))
                    if state.show_thinking:
                        lines.extend(("thinking", line)
                                     for line in _wrap(reasoning, columns))
            content = message.get("content", "")
            if content:
                lines.extend(("body", line)
                             for line in _wrap(content, columns))
            elif (message["role"] == "assistant" and state.streaming
                  and index == total - 1):
                lines.append(("body", "..."))
            lines.append(("gap", ""))
        capacity = max(1, (transcript[3] - transcript[1]) // line_h)
        first = state.scroll.clamp(len(lines), capacity)
        if not lines:
            body.text(transcript, "Type a message below to begin.",
                      size=12, color=body.tango.GREY, align="center")
        else:
            y = transcript[1]
            for kind, line in lines[first:first + capacity]:
                if kind == "role":
                    body.text((left + 4, y, right - 4, y + line_h), line,
                              size=10, bold=True,
                              color=body.tango.BLUE_BRIGHT)
                elif kind == "thinking":
                    body.text((left + 12, y, right - 4, y + line_h), line,
                              size=10, color=body.tango.GREY)
                elif kind == "body":
                    body.text((left + 12, y, right - 4, y + line_h), line,
                              size=12, color=body.tango.SILVER)
                y += line_h

        body.card(input_box, selected=not state.busy())
        pad = max(12, int(16 * body.scale))
        editor = state.renaming if state.renaming is not None else state.editor
        prefix = "NAME" if state.renaming is not None else ">"
        input_font = body.font(14)
        columns = max(1, (right - left - pad * 2 - 50)
                      // max(1, 8 * input_font.scale))
        visible, _cursor = editor.view(columns)
        body.text((left + pad, input_box[1], left + pad + 52, bottom),
                  prefix, size=12, bold=True,
                  color=body.tango.BLUE_BRIGHT)
        placeholder = ("new name" if state.renaming is not None
                       else ("model is loading" if state.loading
                             else "message"))
        body.text((left + pad + 52, input_box[1], right - pad, bottom),
                  visible or placeholder, size=14,
                  color=body.tango.WHITE if visible else body.tango.SILVER)

    def _error(self, body: Body, message: str) -> None:
        left, top, right, bottom = body.box
        body.card(body.box, selected=True, danger=True)
        pad = max(12, int(18 * body.scale))
        body.text((left + pad, top + pad, right - pad, top + pad + 34),
                  "SERVER ERROR", size=18, bold=True,
                  color=body.tango.RED_BRIGHT, valign="top")
        body.wrapped((left + pad, top + pad + 42, right - pad,
                      bottom - pad), message, size=12)

    def _loading(self, body: Body, message: str) -> None:
        left, top, right, bottom = body.box
        width = min(right - left, max(300, int((right - left) * 0.58)))
        height = max(70, int(90 * body.scale))
        x = left + (right - left - width) // 2
        y = top + (bottom - top - height) // 2
        card = (x, y, x + width, y + height)
        body.draw.panel(card, radius=max(7, int(12 * body.scale)),
                        edge=body.tango.BLUE)
        body.text((x + 16, y, x + width - 16, y + height), message,
                  size=12, bold=True, color=body.tango.WHITE,
                  align="center")


def _action(action: Any, state: Any,
            handle: Callable[[int, Any], bool]) -> bool:
    if isinstance(action, tuple) and action[0] == "view":
        target = action[1]
        if target == "list":
            state.refresh_listing()
            state.view = "list"
        elif target == "chat":
            if state.convo is not None:
                state.view = "chat"
        elif target == "picker" and state.convo is not None:
            state.selected = sorted(models.MODELS).index(state.convo.model_id)
            state.view = "picker"
        elif target == "params" and state.convo is not None:
            state.param_index = 0
            state.view = "params"
        elif target == "help":
            state.view = "help"
        return True
    if isinstance(action, tuple) and action[0] == "select":
        index = int(action[1])
        if state.view in ("list", "picker"):
            if state.selected == index:
                return handle(10, state)
            state.selected = index
        elif state.view == "params":
            if state.param_index == index:
                return handle(10, state)
            state.param_index = index
        return True
    return True


def run(state: Any, handle: Callable[[int, Any], bool], *,
        mode: bool | None = None, tick_ms: int = 100) -> int | None:
    """Run the graphical chat, or return ``None`` for the text fallback."""
    if mode is False or os.environ.get("KILIX_TUI_HEADLESS") == "1":
        return None
    forced = mode is True
    modules = shared()
    if modules is None:
        if forced:
            print("bonsai-cpu: graphics unavailable: "
                  "kilix-tui-utils is not installed", file=sys.stderr)
            return 1
        return None
    graphics, _tango = modules
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        if forced:
            print("bonsai-cpu: graphics unavailable: standard input and "
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
            print(f"bonsai-cpu: graphics unavailable: {detail}",
                  file=sys.stderr)
            return 1
        return None

    try:
        renderer = ChatRenderer()
    except (graphics.GraphicsUnavailable, RuntimeError) as error:
        if forced:
            print(f"bonsai-cpu: graphics unavailable: {error}",
                  file=sys.stderr)
            return 1
        return None

    def loop(stdscr: Any) -> int:
        try:
            attributes = termios.tcgetattr(sys.stdin.fileno())
            updated = list(attributes)
            updated[0] &= ~termios.IXON
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, updated)
        except (OSError, ValueError, termios.error):
            pass
        try:
            import curses
            curses.curs_set(0)
            stdscr.keypad(True)
            stdscr.timeout(tick_ms)
            try:
                curses.mousemask(curses.ALL_MOUSE_EVENTS)
            except curses.error:
                pass
            stdscr.erase()
            stdscr.refresh()
            display = graphics.KittyDisplay(
                sys.stdout, max_fps=min(20.0, 1000.0 / tick_ms))
            clear = True
            previous_frame: Any = None
            try:
                while not getattr(state, "finished", False):
                    rows, columns = stdscr.getmaxyx()
                    identity = (time.strftime("%H:%M"),
                                renderer.snapshot(state), rows, columns)
                    if clear or identity != previous_frame:
                        pixels = graphics.terminal_pixel_size(
                            sys.stdout.fileno(), columns, rows)
                        frame = renderer.render(
                            state, columns, rows, pixels)
                        if clear:
                            sys.stdout.write("\x1b[2J\x1b[H")
                            display.invalidate()
                        display.present(frame, force_full=clear)
                        sys.stdout.flush()
                        clear = False
                        previous_frame = identity
                    key = stdscr.getch()
                    result = display.flush()
                    if getattr(result, "emitted", False):
                        sys.stdout.flush()
                    if key == curses.KEY_RESIZE:
                        clear = True
                        continue
                    if key == -1:
                        continue
                    if key == curses.KEY_MOUSE:
                        try:
                            _id, x, y, _z, buttons = curses.getmouse()
                        except curses.error:
                            continue
                        if buttons & (curses.BUTTON1_CLICKED
                                      | curses.BUTTON1_PRESSED):
                            px = int((x + 0.5) * pixels[0] / max(1, columns))
                            py = int((y + 0.5) * pixels[1] / max(1, rows))
                            for action, (x1, y1, x2, y2) in reversed(
                                    renderer.hits):
                                if x1 <= px <= x2 and y1 <= py <= y2:
                                    if not _action(action, state, handle):
                                        return 0
                                    break
                        continue
                    if not handle(key, state):
                        return 0
            finally:
                display.close()
            return 0
        finally:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN,
                                  attributes)
            except (UnboundLocalError, OSError, ValueError, termios.error):
                pass

    try:
        import curses
        return curses.wrapper(loop)
    except (graphics.GraphicsUnavailable, termios.error, RuntimeError,
            curses.error) as error:
        if forced:
            print(f"bonsai-cpu: graphics unavailable: {error}",
                  file=sys.stderr)
            return 1
        return None
