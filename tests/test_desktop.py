"""The optional pixel launcher preserves the model store's full contract.

The shared renderer is deliberately optional, but when it is present its
copied section table, terminal gate, and model drill-down all need coverage.
These are exactly the seams that a non-TTY smoke test otherwise skips.
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import tempfile
import termios
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import desktop, store  # noqa: E402


def load_entry():
    path = os.path.join(ROOT, "tools", "kilix-bonsai", "main.py")
    spec = importlib.util.spec_from_file_location("kilix_bonsai_entry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StubCanvas:
    def __init__(self, width, height):
        self.width, self.height = width, height

    def fill_rect(self, *args, **kwargs):
        pass

    def fill_circle(self, *args, **kwargs):
        pass

    def text(self, *args, **kwargs):
        pass

    def text_shadow(self, *args, **kwargs):
        pass

    def rgb_bytes(self):
        return b"\0" * (self.width * self.height * 3)

    def close(self):
        pass


class SectionConfigurationTest(unittest.TestCase):
    def test_an_already_imported_renderer_gets_the_bonsai_sections(self):
        desk = SimpleNamespace(SECTIONS=("old",))
        graphics = SimpleNamespace(SECTIONS=("stale",))
        desktop._configure_sections(desk, graphics)
        self.assertEqual(desk.SECTIONS, desktop.SECTIONS)
        self.assertEqual(graphics.SECTIONS, desktop.SECTIONS)

    @unittest.skipUnless(desktop.available(),
                         "kilix-tui-utils is not alongside this checkout")
    def test_real_renderer_uses_four_sections_and_draws_entries(self):
        desk = desktop.shared()
        from kilix_desk import graphics

        # Recreate the failing import order: graphics has already copied the
        # stock desktop's table before Bonsai constructs its State.
        stock = ("Home", "Programs", "Machine",
                 "System", "Session", "Power")
        desk.SECTIONS = stock
        graphics.SECTIONS = stock

        state = desktop.build_state()
        renderer = graphics.DesktopRenderer(
            canvas_factory=lambda width, height: StubCanvas(width, height))
        state.section = desktop.SECTIONS.index("Chat")
        renderer.render(state, 100, 30, (960, 560), clock="12:00")

        sections = [index for kind, index, _box in renderer.hits
                    if kind == "section"]
        entries = [index for kind, index, _box in renderer.hits
                   if kind == "entry"]
        self.assertEqual(desk.SECTIONS, desktop.SECTIONS)
        self.assertEqual(graphics.SECTIONS, desktop.SECTIONS)
        self.assertEqual(sections, list(range(len(desktop.SECTIONS))))
        self.assertTrue(entries)


@unittest.skipUnless(desktop.available(),
                     "kilix-tui-utils is not alongside this checkout")
class ModelStoreStateTest(unittest.TestCase):
    def test_an_empty_machine_opens_on_models(self):
        with mock.patch.object(desktop.store, "any_present",
                               return_value=False):
            state = desktop.build_state()
        self.assertEqual(desktop.SECTIONS[state.section], "Models")

    def test_every_model_and_combined_default_size_are_visible(self):
        state = desktop.build_state()
        state.section = desktop.SECTIONS.index("Models")
        entries = state.entries()
        models = state.models
        self.assertEqual(
            [entry.label for entry in entries if entry.submenu],
            [model.title for model in models])
        self.assertIn(store.human_bytes(sum(
            model.default_variant.bytes for model in models)),
            entries[-1].hint)

    def test_each_model_drills_into_every_variant_and_action(self):
        state = desktop.build_state()
        state.section = desktop.SECTIONS.index("Models")
        roots = [entry for entry in state.entries() if entry.submenu]

        for root, model in zip(roots, state.models):
            state.submenu = root.submenu
            actions = state.entries()
            labels = [entry.label for entry in actions]
            self.assertTrue(any(label.startswith("Open") for label in labels),
                            model.id)
            self.assertIn("Dependencies", labels, model.id)
            for variant in model.variants:
                matching = [entry for entry in actions
                            if variant.title in entry.label]
                self.assertEqual(len(matching), 1, (model.id, variant.id))
                action = matching[0]
                self.assertTrue(
                    action.label.startswith(("Download", "Verify")),
                    action.label)
                if action.label.startswith("Download"):
                    self.assertTrue(action.confirm, action.label)
                    self.assertIn(store.human_bytes(max(
                        0, variant.bytes - store.variant_state(
                            model, variant).present_bytes)), action.label)
                else:
                    self.assertIsNotNone(action.argv)
                    self.assertIn("verify", action.argv)
                    self.assertIn("--wait", action.argv)

    def test_missing_variants_and_dependencies_always_confirm(self):
        missing = SimpleNamespace(state=store.MISSING, present_bytes=0)
        with mock.patch.object(desktop.store, "variant_state",
                               return_value=missing), \
                mock.patch.object(desktop.store, "deps_state",
                                  return_value=None), \
                mock.patch.object(desktop.launcher, "launchable",
                                  return_value=(False, "not downloaded yet")):
            state = desktop.build_state()
            state.section = desktop.SECTIONS.index("Models")
            root = next(entry for entry in state.entries() if entry.submenu)
            state.submenu = root.submenu
            actions = state.entries()

        model = state.models[0]
        downloads = [entry for entry in actions
                     if entry.label.startswith("Download")]
        self.assertEqual(len(downloads), len(model.variants))
        self.assertTrue(all(entry.confirm for entry in downloads))
        deps = next(entry for entry in actions
                    if entry.label == "Dependencies")
        self.assertTrue(deps.confirm)

    def test_model_breadcrumb_names_the_drill_down(self):
        state = desktop.build_state()
        state.section = desktop.SECTIONS.index("Models")
        root = next(entry for entry in state.entries() if entry.submenu)
        state.submenu = root.submenu
        self.assertIn(root.label, state.breadcrumb())


class ModeSelectionTest(unittest.TestCase):
    def test_screenshot_bypasses_graphics_and_writes_a_frame(self):
        entry = load_entry()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "frame.txt")
            with mock.patch.object(
                    entry.desktop, "shared",
                    side_effect=AssertionError("graphics path was consulted")):
                status = entry.main(["--screenshot", path])
            self.assertEqual(status, 0)
            with open(path, encoding="utf-8") as handle:
                frame = handle.read()
        self.assertIn("KILIX BONSAI", frame)

    def test_forced_graphics_without_the_shared_package_is_a_clean_error(self):
        error = io.StringIO()
        with mock.patch.object(desktop, "shared", return_value=None), \
                redirect_stderr(error):
            status = desktop.run(["--graphics"])
        self.assertEqual(status, 1)
        self.assertIn("graphics unavailable", error.getvalue())
        self.assertIn("kilix-tui-utils", error.getvalue())

    @unittest.skipUnless(desktop.available(),
                         "kilix-tui-utils is not alongside this checkout")
    def test_forced_graphics_without_a_tty_is_a_clean_error(self):
        stream = SimpleNamespace(isatty=lambda: False)
        error = io.StringIO()
        with mock.patch.object(desktop.sys, "stdin", stream), \
                mock.patch.object(desktop.sys, "stdout", stream), \
                redirect_stderr(error):
            status = desktop.run(["--graphics"])
        self.assertEqual(status, 1)
        self.assertIn("must both be terminals", error.getvalue())

    @unittest.skipUnless(desktop.available(),
                         "kilix-tui-utils is not alongside this checkout")
    def test_terminal_setup_failure_is_reported_without_a_traceback(self):
        from kilix_desk import graphics, gui

        stream = SimpleNamespace(isatty=lambda: True)
        error = io.StringIO()
        failure = termios.error(25, "Inappropriate ioctl for device")
        with mock.patch.object(desktop.sys, "stdin", stream), \
                mock.patch.object(desktop.sys, "stdout", stream), \
                mock.patch.object(graphics, "available",
                                  return_value=(True, "")), \
                mock.patch.object(gui, "run", side_effect=failure), \
                redirect_stderr(error):
            status = desktop.run(["--graphics"])
        self.assertEqual(status, 1)
        self.assertIn("graphics unavailable", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_text_and_graphics_flags_are_mutually_exclusive(self):
        entry = load_entry()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.build_parser().parse_args(["--graphics", "--text"])


if __name__ == "__main__":
    unittest.main()
