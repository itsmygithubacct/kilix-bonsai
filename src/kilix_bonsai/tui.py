"""kilix-bonsai — the terminal UI over the BitNet model set.

The one behaviour that shaped this file: **opening it before anything is
downloaded is the normal case, not an error.** A model store whose UI greets a
new machine with "no such directory" would be a worse tool than the shell
command it replaced, so an empty store opens the setup screen instead of the
list, every action that needs weights offers to fetch them, and the size of
what is about to be downloaded is on screen before the confirmation, never
after it.

Downloads and dependency installs are the model folder's own scripts, run on
the real terminal with curses suspended (see `provision`). This file never
fetches a byte itself.
"""
from __future__ import annotations

import os
import sys

from . import launcher, provision, screen, store
from .catalog import Model, Variant, load

TITLE = "Kilix Bonsai"

# Two vocabularies for the same three states: the table has one column left
# after the numbers, the detail screen has the whole width.
_SHORT = {store.PRESENT: "ready", store.PARTIAL: "partial",
          store.MISSING: "missing"}
_LONG = {store.PRESENT: "downloaded and ready",
         store.PARTIAL: "partly downloaded",
         store.MISSING: "not downloaded"}


class State:
    """Everything the screens draw from, and the only thing they mutate."""

    def __init__(self, models: list[Model] | None = None) -> None:
        self.models = models if models is not None else load()
        self.selected = 0
        self.variant_index = 0
        self.message = ""
        self.stdscr = None
        self.pending: dict | None = None
        self.states: dict[tuple[str, str], store.VariantState] = {}
        self.refresh()
        # Three entry states, cheapest question first. Nothing downloaded is
        # a first run and opens on setup. Something runnable opens on the
        # launcher, because "open a model" is what someone came to do. Weights
        # present but no interface for them falls back to the store list.
        if not store.any_present(self.models):
            self.screen = "setup"
        elif any(launcher.launchable(model)[0] for model in self.models):
            self.screen = "launch"
        else:
            self.screen = "list"

    # -- data ---------------------------------------------------------------

    def refresh(self) -> None:
        """Re-stat every variant. Cheap enough to run after any action."""
        self.states = {
            (model.id, variant.id): store.variant_state(model, variant)
            for model in self.models for variant in model.variants
        }

    def state_of(self, model: Model, variant: Variant) -> store.VariantState:
        return self.states[(model.id, variant.id)]

    @property
    def model(self) -> Model:
        return self.models[self.selected]

    @property
    def variant(self) -> Variant:
        variants = self.model.variants
        return variants[min(self.variant_index, len(variants) - 1)]

    @property
    def total_bytes(self) -> int:
        return sum(model.default_variant.bytes for model in self.models)

    @property
    def ready_count(self) -> int:
        return sum(1 for model in self.models
                   if any(self.state_of(model, v).state == store.PRESENT
                          for v in model.variants))

    # -- actions ------------------------------------------------------------

    def ask(self, heading: str, detail: str, argv: list[str]) -> None:
        """Put an action behind a confirmation showing what it will cost."""
        self.pending = {"heading": heading, "detail": detail, "argv": argv}
        self.screen = "confirm"

    def confirm(self) -> None:
        pending = self.pending
        self.pending = None
        self.screen = "detail"
        if pending is None:
            return
        status = provision.run_detached_from_curses(
            self, pending["argv"], pending["heading"])
        self.refresh()
        self.message = (pending["heading"] + (" — done" if status == 0
                                              else f" — exited {status}"))

    def download(self, model: Model, variant: Variant) -> None:
        state = self.state_of(model, variant)
        remaining = variant.bytes - state.present_bytes
        detail = (f"{len(variant.files)} files, "
                  f"{store.human_bytes(remaining)} still to fetch\n"
                  f"from {variant.files[0].repo}\n"
                  f"into {state.directory}")
        self.ask(f"Download {model.title} · {variant.title}", detail,
                 provision.pull_argv(model, variant))

    def launch(self, model: Model) -> None:
        """Hand the terminal to a model's interface, and take it back after.

        Deliberately not a background process: these are full-screen terminal
        programs, and the launcher is the thing that was in the way.
        """
        ok, detail = launcher.launchable(model)
        if not ok:
            self.message = f"{model.title}: {detail}"
            return
        argv = launcher.tool_argv(model)
        provision.run_detached_from_curses(
            self, argv, f"{detail} — {model.title}")
        self.refresh()
        self.message = f"closed {model.title}"

    def install_deps(self, model: Model) -> None:
        packages = ", ".join(model.deps.get("apt") or ()) or "none"
        self.ask(f"Install dependencies for {model.title}",
                 f"checks: {packages}\n"
                 "system packages are only listed unless you pass --apt;\n"
                 "python packages go in this model's own virtualenv",
                 provision.deps_argv(model))


# -- rendering --------------------------------------------------------------


def _header(surface, state: State, subtitle: str) -> None:
    _, width = surface.getmaxyx()
    ready = f"{state.ready_count}/{len(state.models)} ready"
    screen.write(surface, 0, 0, f"{TITLE} — {subtitle}")
    screen.write(surface, 0, max(0, width - len(ready) - 1), ready)
    screen.write(surface, 1, 0, "─" * max(0, width - 1))


def _footer(surface, state: State, keys: str) -> None:
    height, _ = surface.getmaxyx()
    screen.write(surface, height - 2, 0, state.message)
    screen.write(surface, height - 1, 0, keys)


def render_setup(surface, state: State) -> None:
    height, _ = surface.getmaxyx()
    _header(surface, state, "first run")
    screen.write(surface, 2, 0,
                 "No model weights are on this machine yet.")
    screen.write(surface, 3, 0,
                 "Pick one below and press Enter to download it, or press d to")
    screen.write(surface, 4, 0,
                 "install its dependencies first. Nothing is fetched until you")
    screen.write(surface, 5, 0, "confirm the size.")
    row = 7
    for index, model in enumerate(state.models):
        if row >= height - 3:
            break
        marker = ">" if index == state.selected else " "
        variant = model.default_variant
        screen.write(surface, row, 0,
                     f"{marker} {model.title:<22.22} {model.task:<15.15} "
                     f"{store.human_bytes(variant.bytes):>7}  {model.summary}")
        row += 1
    total = store.human_bytes(state.total_bytes)
    screen.write(surface, min(row + 1, height - 3), 0,
                 f"  all {len(state.models)} models: {total}")
    _footer(surface, state,
            "↑/↓ move · Enter download · d dependencies · Tab list · q quit")


def render_list(surface, state: State) -> None:
    height, _ = surface.getmaxyx()
    _header(surface, state, "BitNet models")
    screen.write(surface, 2, 0,
                 f"  {'MODEL':<22.22} {'TASK':<15.15} {'QUANTIZATION':<17.17} "
                 f"{'SIZE':>6}  STATE")
    row = 3
    visible = max(1, height - 6)
    start = max(0, min(state.selected - visible // 2,
                       max(0, len(state.models) - visible)))
    for index, model in enumerate(state.models[start:start + visible]):
        position = start + index
        marker = ">" if position == state.selected else " "
        variant = model.default_variant
        disk = state.state_of(model, variant)
        label = _SHORT[disk.state]
        if disk.state == store.PARTIAL:
            label = (f"{store.human_bytes(disk.present_bytes)}"
                     f"/{store.human_bytes(variant.bytes)}")
        screen.write(surface, row, 0,
                     f"{marker} {model.title:<22.22} {model.task:<15.15} "
                     f"{model.quantization:<17.17} "
                     f"{store.human_bytes(variant.bytes):>6}  {label}")
        row += 1
    _footer(surface, state,
            "↑/↓ move · Enter details · o open · d download · i deps · q quit")


def render_detail(surface, state: State) -> None:
    height, width = surface.getmaxyx()
    model = state.model
    _header(surface, state, model.title)
    lines = [
        model.summary,
        "",
        f"task         {model.task}",
        f"parameters   {model.parameters}",
        f"quantization {model.quantization}",
        f"license      {model.license}",
        f"upstream     {model.upstream}",
        f"store        {model.store}",
    ]
    if model.shared_with:
        lines.append(f"shared with  {model.shared_with}")
    deps = store.deps_state(model)
    lines.append("dependencies " + ("installed" if deps else "not installed"))
    row = 2
    for line in lines:
        if row >= height - 4:
            break
        if not line:
            row += 1
            continue
        # Long values (an upstream URL, a store path) wrap rather than vanish.
        while line and row < height - 4:
            screen.write(surface, row, 0, line[: width - 1])
            line = line[width - 1:]
            row += 1
    row = min(row + 1, height - 4)
    screen.write(surface, row, 0, "VARIANTS")
    row += 1
    for index, variant in enumerate(model.variants):
        if row >= height - 2:
            break
        marker = ">" if index == min(state.variant_index,
                                     len(model.variants) - 1) else " "
        disk = state.state_of(model, variant)
        default = " (default)" if variant.default else ""
        screen.write(surface, row, 0,
                     f"{marker} {variant.title + default:<40.40} "
                     f"{store.human_bytes(variant.bytes):>7}  "
                     f"{_LONG[disk.state]}")
        row += 1
    _footer(surface, state,
            "↑/↓ variant · Enter download · i deps · v verify · h back · q quit")


def render_confirm(surface, state: State) -> None:
    pending = state.pending or {}
    _header(surface, state, "confirm")
    screen.write(surface, 3, 2, pending.get("heading", ""))
    row = 5
    for line in str(pending.get("detail", "")).splitlines():
        screen.write(surface, row, 2, line)
        row += 1
    screen.write(surface, row + 1, 2,
                 "The download runs in this terminal and can be interrupted;")
    screen.write(surface, row + 2, 2,
                 "it resumes from where it stopped when you run it again.")
    _footer(surface, state, "y confirm · any other key cancel")


SCREENS = {
    "launch": launcher.render,
    "setup": render_setup,
    "list": render_list,
    "detail": render_detail,
    "confirm": render_confirm,
}


def render(surface, state: State) -> None:
    SCREENS[state.screen](surface, state)


# -- input ------------------------------------------------------------------


def handle(key: int, state: State) -> bool:
    if state.screen == "confirm":
        if key in screen.YES:
            state.confirm()
        else:
            state.pending = None
            state.screen = "detail"
        return True

    if screen.is_quit(key) and state.screen != "detail":
        return False

    if state.screen == "detail":
        if key in screen.BACK or screen.is_quit(key):
            state.screen = "list"
            return True
        model = state.model
        step = screen.direction(key)
        if step:
            state.variant_index = max(
                0, min(state.variant_index + step, len(model.variants) - 1))
        elif key in screen.SELECT or key in (ord("d"), ord("D")):
            state.download(model, state.variant)
        elif key in (ord("i"), ord("I")):
            state.install_deps(model)
        elif key in (ord("v"), ord("V")):
            _verify(state, model, state.variant)
        elif key in screen.REFRESH:
            state.refresh()
        return True

    step = screen.direction(key)
    if step:
        state.selected = max(0, min(state.selected + step,
                                    len(state.models) - 1))
    elif state.screen == "launch":
        if key in screen.SELECT:
            state.launch(state.model)
        elif key == ord("\t"):
            state.screen = "list"
        elif key in (ord("d"), ord("D")):
            state.download(state.model, state.model.default_variant)
        elif key in screen.REFRESH:
            state.refresh()
    elif key == ord("\t"):
        state.screen = "list" if state.screen == "setup" else "setup"
    elif key in screen.SELECT:
        if state.screen == "setup":
            state.download(state.model, state.model.default_variant)
        else:
            state.variant_index = 0
            state.screen = "detail"
    elif key in (ord("d"), ord("D")):
        state.download(state.model, state.model.default_variant)
    elif key in (ord("o"), ord("O")):
        state.launch(state.model)
    elif key in (ord("i"), ord("I")):
        state.install_deps(state.model)
    elif key in screen.REFRESH:
        state.refresh()
    return True


def _verify(state: State, model: Model, variant: Variant) -> None:
    """Hash a variant on the real terminal — far too slow to do behind a frame."""
    ok = False
    with provision.terminal(state, f"Verify {model.title} · {variant.title}"):
        ok = store.verify(
            model, variant,
            report=lambda path, good, detail: print(
                f"  {'ok  ' if good else 'FAIL'} {path} — {detail}"))
        print("\nall files verified." if ok else "\nverification failed.")
    state.refresh()
    state.message = (f"{model.id} · {variant.id}: "
                     + ("verified" if ok else "verification FAILED"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    state = State()
    if path := screen.screenshot_argv(argv):
        with open(path, "w", encoding="utf-8") as target:
            target.write(screen.render_to_text(render, state) + "\n")
        return 0
    return screen.run(render, state, handle=handle)
