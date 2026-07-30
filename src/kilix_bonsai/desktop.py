"""Adapt Kilix Bonsai's launcher and store to the shared pixel desktop.

`kilix-tui` already owns the Tango-themed rasterizer and Kitty transport, so
this module supplies its section and entry state rather than drawing another
approximation. The Models section drills into every model and exposes the same
variants, dependency install, verification, and launch actions as the
self-contained text TUI.

The dependency remains optional and pixel mode is explicit. `run()` returns
``None`` unless ``--graphics`` requested it; the command otherwise runs the
repository's canonical text TUI.
"""
from __future__ import annotations

import os
import shutil
import sys
import termios
from typing import Any

from . import catalog, launcher, provision, store

# One section per task, plus the store. The shared renderer draws whatever
# section names it is given, so this is the whole navigation model.
SECTIONS = ("Chat", "Speech", "Images", "Models")

_DESK: Any | None = None


def shared() -> Any | None:
    """Return the `kilix_desk` package, or None when it is not installed."""
    global _DESK
    if _DESK is not None:
        return _DESK or None
    import importlib

    source_home = os.environ.get("GPU_TERMINAL_SOURCE_HOME") or os.path.join(
        os.path.expanduser("~"), "gpu_terminal")
    for home in (os.environ.get("KILIX_TUI_UTILS_HOME", ""),
                 os.path.join(os.path.abspath(os.path.expanduser(source_home)),
                              "kilix-tui-utils")):
        source = os.path.join(home, "src") if home else ""
        if source and os.path.isdir(os.path.join(source, "kilix_desk")):
            if source not in sys.path:
                sys.path.insert(0, source)
            break
    try:
        _DESK = importlib.import_module("kilix_desk.desk")
    except ImportError:
        _DESK = False
        return None
    return _DESK


def available() -> bool:
    return shared() is not None


def _kind_for(section: str) -> str:
    return {"Chat": "chat", "Speech": "speech-to-text",
            "Images": "image"}[section]


def _section_for(kind: str) -> str | None:
    return {"chat": "Chat", "speech-to-text": "Speech",
            "image": "Images"}.get(kind)


def _command_argv() -> tuple[str, ...]:
    """Return this checkout's CLI, preferring its installed command."""
    if installed := shutil.which("kilix-bonsai"):
        return (installed,)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    entry = os.path.join(root, "tools", "kilix-bonsai", "main.py")
    return (sys.executable, entry)


def _configure_sections(desk: Any, graphics: Any | None = None) -> None:
    """Keep both shared renderers on this adapter's navigation table.

    ``kilix_desk.graphics`` imports ``SECTIONS`` by value. Updating only
    ``desk.SECTIONS`` therefore works when graphics is imported afterwards but
    silently leaves six stale desktop sections when it was imported first.
    Synchronising an already-loaded graphics module makes either import order
    safe.
    """
    desk.SECTIONS = SECTIONS
    loaded = graphics or sys.modules.get("kilix_desk.graphics")
    if loaded is not None:
        loaded.SECTIONS = SECTIONS


def build_state() -> Any:
    """Return a shared-desktop State whose entries are this repo's models."""
    desk = shared()
    if desk is None:
        raise RuntimeError("kilix_desk is not installed")

    _configure_sections(desk)
    Entry = desk.Entry
    models = catalog.load()
    by_submenu = {f"model:{model.id}": model for model in models}
    total_bytes = sum(model.default_variant.bytes for model in models)

    class BonsaiState(desk.State):
        """Shared navigation and confirmation around Bonsai model actions."""

        def __init__(self) -> None:
            super().__init__()
            self.models = models
            if not store.any_present(self.models):
                self.section = SECTIONS.index("Models")
                return
            for model in self.models:
                if launcher.launchable(model)[0]:
                    section = _section_for(model.runtime.get("kind", ""))
                    if section is not None:
                        self.section = SECTIONS.index(section)
                    return
            self.section = SECTIONS.index("Models")

        def entries(self) -> list[Any]:
            section = SECTIONS[self.section % len(SECTIONS)]
            if section == "Models":
                return self._model_entries()
            wanted = _kind_for(section)
            out = []
            for model in self.models:
                if model.runtime.get("kind") != wanted:
                    continue
                ok, detail = launcher.launchable(model)
                argv = launcher.tool_argv(model) if ok else None
                out.append(Entry(model.title,
                                 tuple(argv) if argv else None,
                                 hint=detail if ok else "",
                                 reason="" if ok else detail))
            return out

        def _model_entries(self) -> list[Any]:
            model = by_submenu.get(self.submenu or "")
            if model is not None:
                return self._model_actions(model)

            out = []
            for item in self.models:
                states = [store.variant_state(item, variant)
                          for variant in item.variants]
                ready = sum(state.state == store.PRESENT for state in states)
                if ready:
                    status = f"{ready}/{len(states)} variants ready"
                else:
                    status = ("download "
                              f"{store.human_bytes(item.default_variant.bytes)}")
                out.append(Entry(
                    item.title, None, submenu=f"model:{item.id}",
                    hint=f"{status}  ▸"))
            out.append(Entry(
                f"All {len(self.models)} default variants", None,
                reason="combined size of each model's default variant",
                hint=store.human_bytes(total_bytes)))
            return out

        def _model_actions(self, model: Any) -> list[Any]:
            out = []
            ok, detail = launcher.launchable(model)
            argv = launcher.tool_argv(model) if ok else None
            if ok:
                out.append(Entry(
                    "Open interface", tuple(argv) if argv else None,
                    hint=detail))

            for variant in model.variants:
                disk = store.variant_state(model, variant)
                if disk.state == store.PRESENT:
                    verify = (*_command_argv(), "verify", model.id,
                              "--variant", variant.id, "--wait")
                    out.append(Entry(
                        f"Verify · {variant.title}", verify,
                        hint=f"{store.human_bytes(variant.bytes)} · ready"))
                    continue
                remaining = max(0, variant.bytes - disk.present_bytes)
                label = (f"Download · {variant.title} · "
                         f"{store.human_bytes(remaining)}")
                hint = ("resume" if disk.state == store.PARTIAL
                        else "download")
                out.append(Entry(
                    label, tuple(provision.pull_argv(model, variant)),
                    hint=hint, confirm=True))

            deps = store.deps_state(model)
            out.append(Entry(
                "Dependencies",
                None if deps else tuple(provision.deps_argv(model)),
                hint="installed" if deps else "install",
                reason="already installed" if deps else "",
                confirm=not deps))

            if not ok:
                out.append(Entry("Open interface", None, reason=detail,
                                 hint=detail))

            out.extend((
                Entry(f"About · {model.task} · {model.parameters}", None,
                      reason=model.summary, hint=model.quantization),
                Entry(f"Store · {model.store}", None, reason=model.store,
                      hint=model.license),
            ))
            return out

        def breadcrumb(self) -> str:
            section = SECTIONS[self.section % len(SECTIONS)]
            model = by_submenu.get(self.submenu or "")
            if section == "Models" and model is not None:
                return f"Models ▸ {model.title}"
            return section

    return BonsaiState()


def run(argv: list[str]) -> int | None:
    """Run the explicitly requested pixel desktop.

    None means the canonical text TUI should be used.  Pixel rendering remains
    available through ``--graphics`` but is never selected automatically.
    """
    if "--text" in argv or "--screenshot" in argv:
        return None
    forced = "--graphics" in argv
    if not forced:
        return None
    desk = shared()
    if desk is None:
        if forced:
            print("kilix-bonsai: graphics unavailable: "
                  "kilix-tui-utils is not installed",
                  file=sys.stderr)
            return 1
        return None
    state = build_state()
    try:
        from kilix_desk import graphics, gui
    except ImportError as error:
        if forced:
            print(f"kilix-bonsai: graphics unavailable: {error}",
                  file=sys.stderr)
            return 1
        return None
    _configure_sections(desk, graphics)

    problem = ""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        problem = "standard input and output must both be terminals"
    elif not forced and os.environ.get("KILIX_TUI_GRAPHICS") == "0":
        return None
    elif not forced and not graphics.kitty_graphics_likely():
        return None
    else:
        ready, detail = graphics.available()
        if not ready:
            problem = detail

    if problem:
        if forced:
            print(f"kilix-bonsai: graphics unavailable: {problem}",
                  file=sys.stderr)
            return 1
        return None

    try:
        return gui.run(state)
    except (graphics.GraphicsUnavailable, termios.error) as error:
        if forced:
            print(f"kilix-bonsai: graphics unavailable: {error}",
                  file=sys.stderr)
            return 1
        return None
