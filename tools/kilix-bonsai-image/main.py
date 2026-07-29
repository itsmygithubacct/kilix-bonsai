"""kilix-bonsai-image — generate images with the 1.58-bit diffusion model.

Built around iteration rather than one-shot generation. A prompt is rarely
right first time, so the seed, the size, and the reference image stay on screen
and stay set between runs; every result is written with a sidecar recording
exactly what produced it; and `u` re-loads any past generation's settings so
"that one, but warmer light" is two keystrokes rather than retyping.

Reference images are first class — `i` picks one, and the gallery can feed its
own output back in, which is how an edit chain works.

Generation runs on a worker thread; the UI keeps painting and the elapsed time
keeps moving, because a minute of frozen terminal reads as a crash.
"""
from __future__ import annotations

import curses
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from kilix_bonsai import catalog, chrome, screen, widgets       # noqa: E402
from kilix_bonsai.runtime import image as backend                # noqa: E402

TITLE = "Kilix Bonsai Image"

FIELDS = ("prompt", "reference", "size", "seed", "steps")


class State:
    def __init__(self, model: catalog.Model) -> None:
        self.model = model
        self.editor = widgets.Editor()
        self.field = 0
        self.request = backend.Request(prompt="")
        self.preset = 0
        self.gallery = backend.Gallery(
            os.path.join(os.path.expanduser("~"), "kilix-bonsai-images"))
        self.gallery.load()
        self.selected = max(0, len(self.gallery.entries) - 1)
        self.status = "probing backends…"
        self.backends: dict[str, tuple[bool, str]] = {}
        self.running = False
        self.started = 0.0
        self.stdscr = None
        self.show_help = False
        self.view = "compose"

    def probe(self) -> None:
        """Ask which backends work here, off the UI thread.

        Both are probed rather than assumed: a host whose GPU cannot execute
        the kernels may still have a remote that can, and the reverse is just
        as common.
        """
        def work() -> None:
            self.backends = backend.probe_backends()
            usable = [name for name, (ok, _) in self.backends.items() if ok]
            if usable:
                self.request.backend = (backend.LOCAL
                                        if backend.LOCAL in usable
                                        else usable[0])
                self.status = f"ready · {self.request.backend}"
            else:
                self.status = "no usable image backend"
        threading.Thread(target=work, daemon=True).start()

    @property
    def ready(self) -> bool:
        return any(ok for ok, _ in self.backends.values())

    def load_field(self) -> None:
        """Put the selected field's current value into the editor."""
        name = FIELDS[self.field]
        value = {
            "prompt": self.request.prompt,
            "reference": self.request.input_image or "",
            "size": self.request.size,
            "seed": "" if self.request.seed is None else str(self.request.seed),
            "steps": "" if self.request.steps is None
                     else str(self.request.steps),
        }[name]
        self.editor.set(value)

    def commit_field(self) -> None:
        name = FIELDS[self.field]
        text = self.editor.text.strip()
        if name == "prompt":
            self.request.prompt = text
        elif name == "reference":
            self.request.input_image = text or None
        elif name == "size":
            width, _, height = text.partition("x")
            try:
                self.request.width = int(width)
                self.request.height = int(height)
            except ValueError:
                self.status = "size must look like 512x512"
        elif name == "seed":
            self.request.seed = int(text) if text.lstrip("-").isdigit() else None
        elif name == "steps":
            self.request.steps = int(text) if text.isdigit() else None

    def cycle_preset(self) -> None:
        self.preset = (self.preset + 1) % len(backend.PRESETS)
        _, width, height = backend.PRESETS[self.preset]
        self.request.width, self.request.height = width, height
        if FIELDS[self.field] == "size":
            self.load_field()

    def cycle_backend(self) -> None:
        order = [backend.LOCAL, backend.REMOTE]
        current = order.index(self.request.backend) \
            if self.request.backend in order else 0
        self.request.backend = order[(current + 1) % len(order)]
        ok, why = self.backends.get(self.request.backend, (False, "unprobed"))
        self.status = f"{self.request.backend}: {'ready' if ok else why}"

    def generate(self) -> None:
        if self.running:
            return
        problem = self.request.validate()
        if problem:
            self.status = problem
            return
        ok, why = self.backends.get(self.request.backend, (False, "unprobed"))
        if not ok:
            self.status = f"{self.request.backend} is not usable: {why}"
            return
        self.running = True
        self.started = time.monotonic()
        target = self.gallery.next_path(len(self.gallery.entries) + 1)
        request = backend.Request(**vars(self.request))

        def work() -> None:
            result = backend.generate(request, target)
            self.gallery.record(result)
            self.selected = len(self.gallery.entries) - 1
            self.running = False
            self.status = (f"generated in {result.seconds:.0f}s"
                           if result.ok else result.detail)
        threading.Thread(target=work, daemon=True).start()

    def reuse(self) -> None:
        """Load a past generation's settings back into the composer."""
        if not self.gallery.entries:
            return
        entry = self.gallery.entries[self.selected]
        self.request = backend.Request(**vars(entry.request))
        self.load_field()
        self.status = f"loaded the settings from {os.path.basename(entry.path)}"

    def use_as_reference(self) -> None:
        if not self.gallery.entries:
            return
        entry = self.gallery.entries[self.selected]
        self.request.input_image = entry.path
        self.status = f"reference: {os.path.basename(entry.path)}"
        if FIELDS[self.field] == "reference":
            self.load_field()

    def open_selected(self) -> None:
        if not self.gallery.entries:
            return
        path = self.gallery.entries[self.selected].path
        for viewer in ("feh", "xdg-open", "eog"):
            import shutil as _shutil
            if _shutil.which(viewer):
                import subprocess
                subprocess.Popen([viewer, path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
                self.status = f"opened in {viewer}"
                return
        self.status = "no image viewer found"


HELP = [
    "Tab      next field            Enter    generate",
    "Ctrl-P   cycle size preset     Ctrl-B   local / remote backend",
    "g        gallery view          u        reuse selected settings",
    "r        selected as reference o        open selected in a viewer",
    "?        this help             Ctrl-Q   quit",
]


def render_compose(surface, state: State, top: int, left: int,
                   well: int, width: int) -> None:
    height = top + well
    values = {
        "prompt": state.request.prompt or "(required)",
        "reference": state.request.input_image or "(none)",
        "size": state.request.size,
        "seed": "random" if state.request.seed is None
                else str(state.request.seed),
        "steps": "default" if state.request.steps is None
                 else str(state.request.steps),
    }
    row = top
    for index, name in enumerate(FIELDS):
        marker = ">" if index == state.field else " "
        if index == state.field:
            visible, _ = state.editor.view(max(1, width - 16))
            shown = visible
        else:
            shown = values[name]
        screen.write(surface, row, left, f"{marker} {name:<10} {shown}")
        row += 1
    row += 1
    preset = backend.PRESETS[state.preset][0]
    screen.write(surface, row, left,
                 f"  backend    {state.request.backend}    preset  {preset}")
    row += 2
    if state.gallery.entries:
        screen.write(surface, row, left,
                     f"  {len(state.gallery.entries)} generations · "
                     "g for the gallery")
    if state.running:
        elapsed = time.monotonic() - state.started
        screen.write(surface, min(row + 2, height - 1), left,
                     f"  generating… {elapsed:.0f}s elapsed")


def render_gallery(surface, state: State, top: int, left: int,
                   well: int, width: int) -> None:
    if not state.gallery.entries:
        screen.write(surface, top, left, "  nothing generated yet")
        return
    visible = max(1, well - 2)
    start = max(0, min(state.selected - visible // 2,
                       max(0, len(state.gallery.entries) - visible)))
    row = top
    for index, entry in enumerate(
            state.gallery.entries[start:start + visible]):
        position = start + index
        marker = ">" if position == state.selected else " "
        screen.write(surface, row, left,
                     f"{marker} {os.path.basename(entry.path):<22.22} "
                     f"{entry.request.size:>9}  "
                     f"{entry.request.prompt[:width - 40]}")
        row += 1
    entry = state.gallery.entries[state.selected]
    screen.write(surface, top + well - 1, left,
                 f"  seed {entry.request.seed} · steps {entry.request.steps} · "
                 f"reference {os.path.basename(entry.request.input_image or '') or 'none'}")


def render(surface, state: State) -> None:
    footer = ("↑/↓ move · u reuse · r as reference · o open · g compose"
              if state.view == "gallery" else
              "Tab field · Enter generate · Ctrl-P preset · Ctrl-B backend"
              " · g gallery · ? help")
    page = chrome.page(TITLE, [state.model.title], node="IMAGE 001")
    page.render(surface, 0, footer=footer, status=state.status[:38])
    top, left, well, well_width = page.content_box()
    if state.show_help:
        for index, line in enumerate(HELP):
            screen.write(surface, top + index, left + 1, line)
        return
    if state.view == "gallery":
        render_gallery(surface, state, top, left, well, well_width)
    else:
        render_compose(surface, state, top, left, well, well_width)


def handle(key: int, state: State) -> bool:
    if state.show_help:
        state.show_help = False
        return True
    if key == 17:                                    # Ctrl-Q
        return False
    if state.view == "gallery":
        step = screen.direction(key)
        if step:
            state.selected = max(0, min(state.selected + step,
                                        len(state.gallery.entries) - 1))
        elif key in (ord("g"), 27):
            state.view = "compose"
        elif key in (ord("u"), ord("U")):
            state.reuse()
            state.view = "compose"
        elif key in (ord("r"), ord("R")):
            state.use_as_reference()
        elif key in (ord("o"), ord("O")):
            state.open_selected()
        return True
    if key == ord("\t"):
        state.commit_field()
        state.field = (state.field + 1) % len(FIELDS)
        state.load_field()
        return True
    if key == 16:                                    # Ctrl-P
        state.cycle_preset()
        return True
    if key == 2:                                     # Ctrl-B
        state.cycle_backend()
        return True
    if key in widgets.SUBMIT:
        state.commit_field()
        state.generate()
        return True
    if key == ord("?") and not state.editor.text:
        state.show_help = True
        return True
    if key == ord("g") and not state.editor.text:
        state.view = "gallery"
        return True
    state.editor.handle(key)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wanted = next((a for a in argv if not a.startswith("-")), None)
    try:
        model = catalog.find(wanted) if wanted \
            else next(m for m in catalog.load()
                      if m.runtime.get("kind") == "image")
    except (catalog.CatalogError, StopIteration) as error:
        print(f"kilix-bonsai-image: {error}", file=sys.stderr)
        return 2
    state = State(model)
    state.load_field()
    if path := screen.screenshot_argv(argv):
        with open(path, "w", encoding="utf-8") as target:
            target.write(screen.render_to_text(render, state) + "\n")
        return 0
    state.probe()
    return screen.run(render, state, handle=handle, tick_ms=200)


if __name__ == "__main__":
    raise SystemExit(main())
