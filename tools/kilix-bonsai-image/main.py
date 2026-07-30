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

from kilix_bonsai import catalog, pixel, screen, text, widgets  # noqa: E402
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


class PixelRenderer(pixel.Renderer):
    area = "IMAGES"

    def navigation(self, state: State):
        return (
            ("Compose", "compose"),
            ("Gallery", "gallery"),
            ("Backend", 2),              # Ctrl-B
            ("Help", ord("?")),
        )

    def active_navigation(self, state: State) -> int:
        if state.show_help:
            return 3
        return 1 if state.view == "gallery" else 0

    def breadcrumb(self, state: State) -> str:
        view = "Gallery" if state.view == "gallery" else "Compose"
        return f"Images  /  {view}  /  {state.model.title}"

    def footer(self, state: State) -> str:
        if state.running:
            return (f"generating {time.monotonic() - state.started:.0f}s  ·  "
                    "Ctrl-Q quit")
        if state.view == "gallery":
            return ("Up/Down select  ·  u reuse  ·  r reference  ·  "
                    "o open  ·  g compose")
        return ("Tab next field  ·  Enter generate  ·  Ctrl-P preset  ·  "
                "Ctrl-B backend  ·  g gallery  ·  ? help")

    def snapshot(self, state: State):
        request = state.request
        return (
            state.model.id, state.status, state.view, state.show_help,
            state.running,
            int(time.monotonic() - state.started) if state.running else 0,
            state.field, state.preset, state.selected,
            state.editor.text, state.editor.cursor,
            request.prompt, request.input_image, request.width,
            request.height, request.seed, request.steps, request.backend,
            tuple(sorted(state.backends.items())),
            tuple((entry.path, entry.request.prompt, entry.request.size)
                  for entry in state.gallery.entries),
        )

    def activate(self, action, state: State, handle) -> bool:
        if action == "compose":
            state.view = "compose"
            state.show_help = False
            return True
        if action == "gallery":
            state.view = "gallery"
            state.show_help = False
            return True
        if isinstance(action, tuple) and action[0] == "field":
            state.commit_field()
            state.field = action[1]
            state.load_field()
            return True
        if isinstance(action, tuple) and action[0] == "gallery":
            state.selected = action[1]
            return True
        return super().activate(action, state, handle)

    def body(self, body: pixel.Body, state: State) -> None:
        if state.show_help:
            self._help(body)
        elif state.view == "gallery":
            self._gallery(body, state)
        else:
            self._compose(body, state)

    def _compose(self, body: pixel.Body, state: State) -> None:
        left, top, right, bottom = body.box
        gap = max(6, int(9 * body.scale))
        summary_h = max(48, int(64 * body.scale))
        fields_bottom = bottom - summary_h - gap
        row_h = max(34, (fields_bottom - top)
                    // max(1, len(FIELDS)))
        values = {
            "prompt": state.request.prompt or "Describe an image",
            "reference": state.request.input_image or "None",
            "size": state.request.size,
            "seed": "Random" if state.request.seed is None
                    else str(state.request.seed),
            "steps": "Default" if state.request.steps is None
                     else str(state.request.steps),
        }
        for index, name in enumerate(FIELDS):
            y = top + index * row_h
            row = (left, y + 2, right, min(fields_bottom, y + row_h - 3))
            selected = index == state.field
            body.card(row, selected=selected)
            body.hits.append((("field", index), row))
            pad = max(10, int(14 * body.scale))
            split = left + max(110, int((right - left) * 0.24))
            body.text((left + pad, row[1], split, row[3]), name.upper(),
                      size=10, bold=True,
                      color=(body.tango.WHITE if selected
                             else body.tango.GREY))
            if selected:
                font = body.font(14)
                columns = max(1, (right - split - pad)
                              // max(1, 8 * font.scale))
                editing = state.editor.view(columns)[0]
                value = editing or values[name]
            else:
                editing = ""
                value = values[name]
            body.text((split, row[1], right - pad, row[3]), value,
                      size=14, color=(body.tango.WHITE if editing
                                      else body.tango.SILVER))

        summary = (left, bottom - summary_h, right, bottom)
        mid = left + (right - left - gap) // 2
        backend_box = (left, summary[1], mid, bottom)
        preset_box = (mid + gap, summary[1], right, bottom)
        body.card(backend_box)
        body.card(preset_box)
        pad = max(10, int(14 * body.scale))
        body.text((left + pad, summary[1] + 4, mid - pad,
                   summary[1] + 24), "BACKEND", size=10, bold=True,
                  color=body.tango.GREY, valign="top")
        body.text((left + pad, summary[1] + 20, mid - pad, bottom - 3),
                  state.request.backend, size=16, bold=True,
                  color=body.tango.WHITE)
        preset = backend.PRESETS[state.preset][0]
        body.text((mid + gap + pad, summary[1] + 4, right - pad,
                   summary[1] + 24), "PRESET", size=10, bold=True,
                  color=body.tango.GREY, valign="top")
        body.text((mid + gap + pad, summary[1] + 20, right - pad,
                   bottom - 3), preset, size=16, bold=True,
                  color=body.tango.WHITE)
        if state.running:
            body.draw.fill((left, bottom - max(3, int(4 * body.scale)),
                            right, bottom), body.tango.BLUE_BRIGHT)

    def _gallery(self, body: pixel.Body, state: State) -> None:
        left, top, right, bottom = body.box
        if not state.gallery.entries:
            body.text(body.box,
                      "Nothing generated yet. Choose Compose to begin.",
                      size=12, color=body.tango.GREY, align="center")
            return
        row_h = max(38, int(52 * body.scale))
        capacity = max(1, (bottom - top) // row_h)
        start = max(0, min(state.selected - capacity + 1,
                           len(state.gallery.entries) - capacity))
        for line, entry in enumerate(
                state.gallery.entries[start:start + capacity]):
            index = start + line
            y = top + line * row_h
            row = (left, y + 2, right, y + row_h - 3)
            selected = index == state.selected
            body.card(row, selected=selected)
            body.hits.append((("gallery", index), row))
            pad = max(10, int(14 * body.scale))
            body.text((left + pad, row[1], right - pad, row[3]),
                      os.path.basename(entry.path), size=14,
                      bold=selected,
                      color=(body.tango.WHITE if selected
                             else body.tango.SILVER))
            hint = f"{entry.request.size}  ·  seed {entry.request.seed}"
            body.text((left + pad, row[1], right - pad, row[3]), hint,
                      size=10, color=(body.tango.WHITE if selected
                                      else body.tango.GREY), align="right")

    @staticmethod
    def _help(body: pixel.Body) -> None:
        left, top, right, bottom = body.box
        row_h = max(30, int(42 * body.scale))
        for index, line in enumerate(HELP):
            y = top + index * row_h
            if y + row_h > bottom:
                break
            body.card((left, y, right, y + row_h))
            body.text((left + 14, y, right - 14, y + row_h), line,
                      size=12)


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
        selected = index == state.field
        marker = "▶" if selected else " "
        if selected:
            visible, _ = state.editor.view(max(1, width - 16))
            shown = visible
        else:
            shown = values[name]
        screen.write(
            surface,
            row,
            left,
            f"{marker} {name:<10} {shown}".ljust(width),
            text.attr("selected") if selected else 0,
        )
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
        selected = position == state.selected
        marker = "▶" if selected else " "
        line = (f"{marker} {os.path.basename(entry.path):<22.22} "
                f"{entry.request.size:>9}  "
                f"{entry.request.prompt[:width - 40]}")
        screen.write(
            surface,
            row,
            left,
            line.ljust(width),
            text.attr("selected") if selected else 0,
        )
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
    page = text.page("IMAGES",
                     ("Gallery" if state.view == "gallery" else "Compose")
                     + f" / {state.model.title}",
                     ("Compose", "Gallery", "Backend", "Help"),
                     3 if state.show_help else
                     (1 if state.view == "gallery" else 0))
    page.render(surface, footer=footer, status=state.status[:38])
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
    positional: list[str] = []
    skip = False
    for argument in argv:
        if skip:
            skip = False
        elif argument == "--screenshot":
            skip = True
        elif not argument.startswith("-"):
            positional.append(argument)
    wanted = positional[0] if positional else None
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
    status = pixel.run(
        argv, PixelRenderer, state, handle, tick_ms=200,
        command="kilix-bonsai-image")
    return (screen.run(render, state, handle=handle, tick_ms=200)
            if status is None else status)


if __name__ == "__main__":
    raise SystemExit(main())
