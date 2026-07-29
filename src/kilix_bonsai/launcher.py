"""The launch screen: pick a model, open the interface that suits it.

Which UI a model gets is not a property of this file. Each MODEL.json declares
a `runtime.kind` — chat, image, speech-to-text — and that is what selects the
tool, so a new model arrives with its own answer rather than needing a case
added here.

The art is the point of the layout. A launcher with five rows in an eighty
column pane is mostly empty space, and empty space in a branded tool is a
missed opportunity — so the flame kitten and its bonsai sit beside the list,
and the list moves aside on panes too narrow to hold both.
"""
from __future__ import annotations

import os
import shutil
import sys

from . import art, chrome, screen, store
from .catalog import Model

# Interface per declared runtime kind, and what to say when one is missing.
TOOLS = {
    "chat": ("kilix-bonsai-chat", "Chat"),
    "image": ("kilix-bonsai-image", "Generate images"),
    "speech-to-text": ("kilix-bonsai-speech", "Transcribe speech"),
}

# The list needs this much to stay readable; whatever is left over is offered
# to the art, which picks a scale that fits or declines to draw. So the
# threshold is about the *list*, not the sprite's authored size.
LIST_MIN_WIDTH = 46


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
    """Draw the launch screen. `state` is the model-store TUI's State.

    The chrome comes from the shared utilities when they are installed and
    degrades to a plain title and rule when they are not, so the same code
    draws both and the tests keep asserting plain text.
    """
    page = chrome.page("KILIX BONSAI",
                       [m.title for m in state.models], node="BONSAI 001")
    page.measure(surface)
    page.render(surface, state.selected,
                footer="↑/↓ move · Enter open · Tab store · q quit",
                status=state.message[:40])
    top, left, well_height, well_width = page.content_box()
    height, width = surface.getmaxyx()
    spare = well_width - LIST_MIN_WIDTH - 2
    factor = art.fit(max(0, spare), max(0, well_height - 1)) if spare > 0 else 0
    art_width = art.size(factor)[0] if factor else 0
    list_width = (well_width - art_width - 3) if factor else (well_width - 1)

    if factor:
        drawn = art.draw(surface, top, width - art_width - 1,
                         max_height=well_height, colour=art.usable())
        if drawn and top + drawn < height - 2:
            screen.write(surface, top + drawn, width - art_width - 1,
                         "kilix".center(max(1, art_width - 1)))

    row = top
    screen.write(surface, row, left, "Choose what to open:")
    row += 2
    for index, model in enumerate(state.models):
        if row >= height - 3:
            break
        ok, detail = launchable(model)
        marker = ">" if index == state.selected else " "
        kind = model.runtime.get("kind", "—")
        name = f"{marker} {model.title}"
        screen.write(surface, row, left, name[:list_width])
        screen.write(surface, row + 1, left,
                     f"    {kind:<15.15} {detail}"[:list_width])
        row += 2


