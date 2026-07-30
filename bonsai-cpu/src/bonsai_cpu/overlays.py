"""The screens around the transcript: list, picker, params, help — and the
two banners (loading, error) that paint over whatever is beneath them.

Every renderer here is a plain function over (surface, state) and clips
through screen.write, so all of them hold together at 20x8 the same way the
chat screen does.
"""
from __future__ import annotations

import curses

from . import layout
from . import models
from . import screen
from . import widgets

BOLD = curses.A_BOLD
DIM = curses.A_DIM


def render_list(surface, state) -> None:
    top, bottom, width = layout.shell(
        surface, breadcrumb="Chats / Saved conversations", active=0,
        status=state.loading or state.status,
        footer="Enter open · n new · d delete · Ctrl-Q quit")
    rows = [("[ new conversation ]", "", "")]
    for conversation in state.conversations:
        spec = models.MODELS.get(conversation.model_id, {})
        rows.append((conversation.name or "untitled",
                     spec.get("title", conversation.model_id),
                     f"{len(conversation.messages) // 2} turns · "
                     f"{conversation.updated[:10]}"))
    visible = max(1, bottom - top)
    first = max(0, min(state.selected - visible + 1, len(rows) - visible))
    for row, (name, model, detail) in enumerate(rows[first:first + visible]):
        index = first + row
        marker = "▸ " if index == state.selected else "  "
        attr = BOLD if index == state.selected else 0
        line = f"{marker}{name}"
        screen.write(surface, top + row, 0, line, attr)
        tail = f"{model}  {detail}".strip()
        if tail and width - len(line) - 3 > len(tail):
            screen.write(surface, top + row, width - len(tail) - 1, tail, DIM)


def render_picker(surface, state) -> None:
    top, bottom, width = layout.shell(
        surface, breadcrumb="Models / Switch model", active=2,
        status=state.loading or state.status,
        footer="Enter switch · Esc back — switching restarts the server")
    ids = sorted(models.MODELS)
    for row, model_id in enumerate(ids):
        if top + row >= bottom:
            break
        spec = models.MODELS[model_id]
        marker = "▸ " if row == state.selected else "  "
        current = " (current)" if model_id == state.convo.model_id else ""
        attr = BOLD if row == state.selected else 0
        screen.write(surface, top + row, 0,
                     f"{marker}{spec['title']}{current}", attr)
        note = spec.get("speed_note", "")
        if note and width > len(spec["title"]) + len(note) + 14:
            screen.write(surface, top + row, width - len(note) - 1, note, DIM)


def render_params(surface, state) -> None:
    hint = ("Enter commit · Esc cancel" if state.param_editor is not None
            else "Enter edit · Esc back")
    top, bottom, width = layout.shell(
        surface, breadcrumb="Settings / Conversation", active=3,
        status=state.loading or state.status, footer=hint)
    from .chat_tui import PARAM_FIELDS
    for row, field in enumerate(PARAM_FIELDS):
        if top + row >= bottom:
            break
        editing = (state.param_editor is not None
                   and row == state.param_index)
        marker = "▸ " if row == state.param_index else "  "
        if editing:
            visible, _ = state.param_editor.view(max(1, width - 16))
            value = visible + "▏"
        else:
            value = str(state.convo.params[field])
        label = f"{marker}{field:<12}"
        screen.write(surface, top + row, 0, label,
                     BOLD if row == state.param_index else 0)
        screen.write(surface, top + row, len(label),
                     value[: max(0, width - len(label) - 1)])


HELP_ROWS = (
    ("Enter", "send"),
    ("Esc", "stop generating, or back"),
    ("Ctrl-O", "switch model"),
    ("Ctrl-T", "toggle thinking (27B)"),
    ("Ctrl-F", "fold or reveal the thinking"),
    ("Ctrl-P", "system prompt and sampling"),
    ("Ctrl-N", "new conversation"),
    ("Ctrl-R", "rename conversation"),
    ("PgUp/PgDn", "scroll the transcript"),
    ("Ctrl-Q", "quit"),
)


def render_help(surface, state) -> None:
    top, bottom, _width = layout.shell(
        surface, breadcrumb="Help / Keyboard", active=4,
        status="", footer="any key returns")
    for row, (key, action) in enumerate(HELP_ROWS):
        if top + row >= bottom:
            break
        screen.write(surface, top + row, 1, f"{key:>10}", BOLD)
        screen.write(surface, top + row, 13, action)


def render_loading(surface, state) -> None:
    height, width = surface.getmaxyx()
    row = height // 2
    message = state.loading
    screen.write(surface, row, 0, " " * (width - 1))
    screen.write(surface, row, max(0, (width - len(message)) // 2),
                 message, BOLD)


def render_error(surface, state) -> None:
    row, bottom, width = layout.shell(
        surface, breadcrumb="Chat / Server error", active=1,
        status="", footer="r retry · q quit · any other key back")
    for line in widgets.wrap(state.error, max(1, width - 1)):
        if row >= bottom:
            break
        screen.write(surface, row, 1, line)
        row += 1
