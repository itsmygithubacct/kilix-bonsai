"""kilix-bonsai-enlarge — bonsai-enlarge: enlarge a thumbnail, or a detail of an image.

With no arguments it opens the composer, a TUI in the Kilix text shell
(``--graphics`` for the pixel desktop). With arguments it is the command line:

    kilix-bonsai-enlarge emoji.png --output big.png
    kilix-bonsai-enlarge photo.jpg --crop 120,40,64,64 --scale 8 --output detail.png
    kilix-bonsai-enlarge icon.png --strength 0.35 --baseline lanczos.png --output big.png

The source is cut (``--crop``), laid on ``--background`` where it is
transparent, resampled with Lanczos to the output size, and then partly
redrawn by Bonsai Image 4B: ``--strength`` is how far (0.5 by default; lower
keeps more of the source). ``--baseline`` also writes the plain Lanczos result.
Each output gets ``OUTPUT.json`` recording what produced it.

Weights install on first use, after their licence is shown, exactly as for
kilix-bonsai-image. Generation is local, on an NVIDIA GPU with 4.5 GiB free.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import catalog, pixel, provision, screen, store, text, widgets  # noqa: E402
from kilix_bonsai.runtime import enlarge as plan  # noqa: E402

CLI = os.path.join(ROOT, "tools", "bonsai-image", "main.py")
TITLE = "Kilix Bonsai Enlarge"
FIELDS = ("input", "crop", "size", "strength", "steps", "seed", "prompt", "output")
RESULTS = Path(os.path.expanduser("~")) / "kilix-bonsai-images" / "enlarge"
HINTS = {
    "input": "PNG or JPEG path (required)",
    "crop": "whole image",
    "size": "auto: long side 512",
    "seed": "random",
    "prompt": "generic: sharp, clean, high resolution",
}


def _variant(argv: list[str]) -> str:
    for index, argument in enumerate(argv):
        if argument == "--variant" and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith("--variant="):
            return argument.split("=", 1)[1]
    return "ternary"


def ensure_weights(model: catalog.Model, variant: str) -> int:
    """Install the chosen weight set on first use; 0 when present."""
    chosen = model.variant(f"{variant}-gemlite")
    if store.state(model, chosen).state == store.PRESENT:
        return 0
    status = provision.ensure_downloaded(model, chosen)
    if store.state(model, chosen).state == store.PRESENT:
        return 0
    print(f"kilix-bonsai-enlarge: {chosen.title} weights are not installed",
          file=sys.stderr)
    return status if status not in (None, 0) else 1


class State:
    def __init__(self, model: catalog.Model, results: Path = RESULTS) -> None:
        self.model = model
        self.results = results
        self.editor = widgets.Editor()
        self.field = 0
        self.request = plan.Request(output=str(plan.next_output(results)))
        self.entries = plan.history(results)
        self.selected = max(0, len(self.entries) - 1)
        self.status = "choose an input image, then Enter"
        self.running = False
        self.started = 0.0
        self.show_help = False
        self.view = "compose"

    def value(self, name: str) -> str:
        request = self.request
        return {
            "input": request.input, "crop": request.crop, "size": request.size,
            "strength": f"{request.strength:g}", "steps": str(request.steps),
            "seed": "" if request.seed is None else str(request.seed),
            "prompt": request.prompt, "output": request.output,
        }[name]

    def shown(self, name: str, room: int = 0) -> str:
        """The field as displayed; a long path keeps its end, where the name is."""
        value = self.value(name) or HINTS.get(name, "")
        if room and len(value) > room and name in ("input", "output"):
            return "…" + value[-(room - 1):]
        return value

    def load_field(self) -> None:
        self.editor.set(self.value(FIELDS[self.field]))

    def commit_field(self) -> None:
        name = FIELDS[self.field]
        value = self.editor.text.strip()
        request = self.request
        try:
            if name == "strength":
                request.strength = float(value) if value else plan.DEFAULT_STRENGTH
            elif name == "steps":
                request.steps = int(value) if value else plan.DEFAULT_STEPS
            elif name == "seed":
                request.seed = int(value) if value else None
            else:
                setattr(request, name, value)
        except ValueError:
            self.status = f"{name} must be a number"

    def enlarge(self) -> None:
        if self.running:
            return
        problem = self.request.validate()
        if problem:
            self.status = problem
            return
        argv = [sys.executable, *self.request.argv(CLI)]
        self.running = True
        self.started = time.monotonic()
        self.status = "loading the model…"

        def work() -> None:
            try:
                done = subprocess.run(argv, capture_output=True, text=True,
                                      timeout=1800)
                lines = [line for line in (done.stdout + done.stderr).splitlines()
                         if line.strip()]
                verdict = next((line for line in reversed(lines)
                                if line.startswith(("result:", "Error:"))),
                               lines[-1] if lines else "no output")
                ok = done.returncode == 0 and verdict.startswith("result:")
            except (OSError, subprocess.SubprocessError) as error:
                ok, verdict = False, str(error)
            seconds = time.monotonic() - self.started
            self.entries = plan.history(self.results)
            self.selected = max(0, len(self.entries) - 1)
            if ok:
                self.status = f"enlarged in {seconds:.0f}s: {self.request.output}"
                self.request.output = str(plan.next_output(self.results))
                if FIELDS[self.field] == "output":
                    self.load_field()
            else:
                self.status = verdict.removeprefix("Error:").strip()[:200]
            self.running = False
        threading.Thread(target=work, daemon=True).start()

    def reuse(self) -> None:
        if not self.entries:
            return
        _, data = self.entries[self.selected]
        crop = data.get("crop") or []
        self.request = plan.Request(
            input=data.get("source", ""),
            crop=",".join(map(str, crop)) if len(crop) == 4 else "",
            size=f"{data.get('width')}x{data.get('height')}",
            strength=float(data.get("strength", plan.DEFAULT_STRENGTH)),
            steps=int(data.get("steps", plan.DEFAULT_STEPS)),
            seed=data.get("seed"), prompt=data.get("prompt", ""),
            output=str(plan.next_output(self.results)),
            variant=data.get("variant", "ternary"))
        self.load_field()
        self.status = f"loaded the settings of {self.entries[self.selected][0].name}"

    def open_selected(self) -> None:
        if not self.entries:
            return
        path = str(self.entries[self.selected][0])
        import shutil
        for viewer in ("feh", "xdg-open", "eog", "firefox-esr"):
            if shutil.which(viewer):
                subprocess.Popen([viewer, path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
                self.status = f"opened in {viewer}"
                return
        self.status = "no image viewer found"


HELP = [
    "Tab      next field            Enter    enlarge",
    "r        results view          u        reuse selected settings",
    "o        open selected result  ?        this help",
    "Strength 0.35-0.5 keeps the picture; 0.75 redraws it.",
    "Ctrl-Q   quit",
]


def _footer(state: State) -> str:
    if state.running:
        return f"enlarging {time.monotonic() - state.started:.0f}s · Ctrl-Q quit"
    if state.view == "results":
        return "↑/↓ move · u reuse · o open · r compose"
    return "Tab field · Enter enlarge · r results · ? help"


class PixelRenderer(pixel.Renderer):
    area = "ENLARGE"

    def navigation(self, state: State):
        return (("Compose", "compose"), ("Results", "results"), ("Help", ord("?")))

    def active_navigation(self, state: State) -> int:
        if state.show_help:
            return 2
        return 1 if state.view == "results" else 0

    def breadcrumb(self, state: State) -> str:
        view = "Results" if state.view == "results" else "Compose"
        return f"Enlarge  /  {view}  /  {state.model.title}"

    def footer(self, state: State) -> str:
        return _footer(state)

    def snapshot(self, state: State):
        return (state.status, state.view, state.show_help, state.running,
                int(time.monotonic() - state.started) if state.running else 0,
                state.field, state.selected, state.editor.text, state.editor.cursor,
                tuple(state.value(name) for name in FIELDS),
                tuple(str(path) for path, _ in state.entries))

    def activate(self, action, state: State, handle) -> bool:
        if action in ("compose", "results"):
            state.view = action
            state.show_help = False
            return True
        if isinstance(action, tuple) and action[0] == "field":
            state.commit_field()
            state.field = action[1]
            state.load_field()
            return True
        if isinstance(action, tuple) and action[0] == "result":
            state.selected = action[1]
            return True
        return super().activate(action, state, handle)

    def body(self, body: pixel.Body, state: State) -> None:
        left, top, right, bottom = body.box
        if state.show_help:
            rows = [(line, None) for line in HELP]
        elif state.view == "results":
            rows = [(f"{path.name}  ·  strength {data.get('strength')}  ·  "
                     f"{data.get('width')}x{data.get('height')}", ("result", i))
                    for i, (path, data) in enumerate(state.entries)] or \
                [("Nothing enlarged yet. Choose Compose to begin.", None)]
        else:
            rows = []
            for index, name in enumerate(FIELDS):
                shown = state.editor.text if index == state.field else state.shown(
                    name, max(8, int((right - left) / max(1, 8 * body.scale)) - 14))
                rows.append((f"{name.upper():<10} {shown}", ("field", index)))
        status = textwrap.wrap(state.status, max(20, int(
            (right - left) / max(1, 7 * body.scale))))[:3]
        row_h = max(30, int(40 * body.scale))
        for index, (line, action) in enumerate(rows):
            y = top + index * row_h
            if y + row_h > bottom - row_h * len(status):
                break
            box = (left, y + 2, right, y + row_h - 3)
            selected = ((action == ("field", state.field) and state.view == "compose")
                        or action == ("result", state.selected))
            body.card(box, selected=selected)
            if action:
                body.hits.append((action, box))
            body.text((left + 14, box[1], right - 14, box[3]), line, size=12,
                      color=body.tango.WHITE if selected else body.tango.SILVER)
        for index, line in enumerate(status):
            y = bottom - row_h * (len(status) - index)
            body.text((left, y, right, y + row_h), line, size=10,
                      color=body.tango.WHITE)
        if state.running:
            body.draw.fill((left, bottom - max(3, int(4 * body.scale)), right, bottom),
                           body.tango.BLUE_BRIGHT)


def render(surface, state: State) -> None:
    page = text.page("ENLARGE",
                     ("Results" if state.view == "results" else "Compose")
                     + f" / {state.model.title}",
                     ("Compose", "Results", "Help"),
                     2 if state.show_help else (1 if state.view == "results" else 0))
    page.render(surface, footer=_footer(state), status=state.status[:38])
    top, left, well, width = page.content_box()
    if state.show_help:
        for index, line in enumerate(HELP[:well]):
            screen.write(surface, top + index, left + 1, line[:width - 1])
        return
    if state.view == "results":
        if not state.entries:
            screen.write(surface, top, left, "  nothing enlarged yet")
            return
        visible = max(1, well - 1)
        start = max(0, min(state.selected - visible // 2,
                           max(0, len(state.entries) - visible)))
        for row, (path, data) in enumerate(state.entries[start:start + visible]):
            index = start + row
            selected = index == state.selected
            line = (f"{'▶' if selected else ' '} {path.name:<20.20} "
                    f"{data.get('width')}x{data.get('height'):<5} "
                    f"s{data.get('strength')} {data.get('prompt', '')}")
            screen.write(surface, top + row, left, line[:width].ljust(width),
                         text.attr("selected") if selected else 0)
        return
    row = top
    for index, name in enumerate(FIELDS):
        if row >= top + well:
            return
        selected = index == state.field
        shown = state.editor.view(max(1, width - 14))[0] if selected \
            else state.shown(name, max(2, width - 14))
        screen.write(surface, row, left,
                     f"{'▶' if selected else ' '} {name:<9} {shown}"[:width].ljust(width),
                     text.attr("selected") if selected else 0)
        row += 1
    row += 1
    lines = textwrap.wrap(state.status, max(1, width - 4))
    if state.running:
        lines = [f"enlarging… {time.monotonic() - state.started:.0f}s elapsed"] + lines
    for offset, line in enumerate(lines):
        if row + offset >= top + well:
            break
        screen.write(surface, row + offset, left + 2, line)


def handle(key: int, state: State) -> bool:
    if state.show_help:
        state.show_help = False
        return True
    if key == 17:                                    # Ctrl-Q
        return False
    if state.view == "results":
        step = screen.direction(key)
        if step:
            state.selected = max(0, min(state.selected + step, len(state.entries) - 1))
        elif key in (ord("r"), 27):
            state.view = "compose"
        elif key in (ord("u"), ord("U")):
            state.reuse()
            state.view = "compose"
        elif key in (ord("o"), ord("O")):
            state.open_selected()
        return True
    if key == ord("\t"):
        state.commit_field()
        state.field = (state.field + 1) % len(FIELDS)
        state.load_field()
        return True
    if key in widgets.SUBMIT:
        state.commit_field()
        state.enlarge()
        return True
    if key == ord("?") and not state.editor.text:
        state.show_help = True
        return True
    if key == ord("r") and not state.editor.text:
        state.view = "results"
        return True
    state.editor.handle(key)
    return True


def interface(argv: list[str]) -> int:
    model = catalog.find("bonsai-image-4b")
    state = State(model)
    state.load_field()
    if path := screen.screenshot_argv(argv):
        with open(path, "w", encoding="utf-8") as target:
            target.write(screen.render_to_text(render, state) + "\n")
        return 0
    status = ensure_weights(model, "ternary")
    if status:
        return status
    provision.ensure_deps(model)
    status = pixel.run(argv, PixelRenderer, state, handle, tick_ms=200,
                       command="kilix-bonsai-enlarge")
    return (screen.run(render, state, handle=handle, tick_ms=200)
            if status is None else status)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or all(a.startswith("--") and a in ("--graphics", "--screenshot")
                       or (i > 0 and argv[i - 1] == "--screenshot")
                       for i, a in enumerate(argv)):
        return interface(argv)
    if argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, CLI, "enlarge", "--help"])
    variant = _variant(argv)
    if variant not in ("ternary", "binary"):
        print("kilix-bonsai-enlarge: --variant must be ternary or binary",
              file=sys.stderr)
        return 2
    status = ensure_weights(catalog.find("bonsai-image-4b"), variant)
    if status:
        return status
    os.execv(sys.executable, [sys.executable, CLI, "enlarge", *argv])
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main())
