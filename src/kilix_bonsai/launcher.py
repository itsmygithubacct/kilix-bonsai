"""The portable launch screen: pick a model and open its interface.

Which UI a model gets is not a property of this file. Each MODEL.json declares
a `runtime.kind` — chat, image, speech-to-text — and that is what selects the
tool, so a new model arrives with its own answer rather than needing a case
added here.

This canonical text renderer is the default in every terminal. The optional
graphical launcher supplied by :mod:`kilix_bonsai.desktop` is available only
when explicitly requested.
"""
from __future__ import annotations

import os
import shutil
import sys

from . import screen, store, text
from .catalog import Model

# Interface per declared runtime kind, and what to say when one is missing.
TOOLS = {
    "chat": ("kilix-bonsai-chat", "Chat"),
    "image": ("kilix-bonsai-image", "Generate images"),
    "speech-to-text": ("kilix-bonsai-speech", "Transcribe speech"),
}

def tool_entry(kind: str) -> str | None:
    """Return the tool's main.py inside this checkout, or None."""
    command = (TOOLS.get(kind) or (None, None))[0]
    if command is None:
        return None
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    candidate = os.path.join(root, "tools", command, "main.py")
    return candidate if os.path.isfile(candidate) else None


def tool_argv(model: Model) -> list[str] | None:
    """Return the command that opens this model's interface.

    An installed command wins over the checkout, the same order everything
    else in this stack resolves in, so a machine with the tools on PATH runs
    those rather than whatever tree happens to be nearby.
    """
    kind = model.runtime.get("kind", "")
    command = (TOOLS.get(kind) or (None, None))[0]
    if command is None:
        return None
    if installed := shutil.which(command):
        return [installed, model.id]
    entry = tool_entry(kind)
    return [sys.executable, entry, model.id] if entry else None


def launchable(model: Model) -> tuple[bool, str]:
    """Return whether this model can be opened, and why not when it cannot."""
    kind = model.runtime.get("kind", "")
    if kind not in TOOLS:
        return False, "no interface for this model"
    if tool_argv(model) is None:
        return False, f"{TOOLS[kind][0]} is not installed"
    if store.state(model).state != store.PRESENT:
        return False, "not downloaded yet"
    return True, TOOLS[kind][1]


def render(surface, state) -> None:
    """Draw the no-pixels launch screen."""
    sections = ("Chat", "Speech", "Images", "Models")
    kinds = {"chat": 0, "speech-to-text": 1, "image": 2}
    selected = state.models[state.selected] if state.models else None
    active = kinds.get(selected.runtime.get("kind", ""), 3) \
        if selected else 3
    page = text.page("DESKTOP", "Open a model", sections, active)
    page.render(surface,
                footer="↑/↓ move · Enter open · Tab store · q quit",
                status=state.message[:40])
    top, left, well_height, well_width = page.content_box()
    height, width = surface.getmaxyx()
    list_width = max(1, well_width - 1)

    row = top
    screen.write(surface, row, left, "Choose what to open:")
    row += 1
    for index, model in enumerate(state.models):
        if row >= min(height - 2, top + well_height):
            break
        ok, detail = launchable(model)
        selected_row = index == state.selected
        marker = "▶" if selected_row else " "
        kind = model.runtime.get("kind", "—")
        name = f"{marker} {model.title}"
        hint = f"{kind} · {detail}"
        row_attr = text.attr("selected") if selected_row else 0
        screen.write(
            surface, row, left, name.ljust(list_width)[:list_width], row_attr)
        if len(name) + len(hint) + 3 < list_width:
            screen.write(surface, row, left + list_width - len(hint) - 1,
                         hint, row_attr)
        row += 1
