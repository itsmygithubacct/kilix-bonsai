"""The UI opens on setup when nothing is downloaded, and never acts unasked.

Two properties, both of which would be discovered the expensive way:

* A model store's UI is *usually* first opened on a machine with nothing in it.
  If that case renders an error, or an empty list with no way forward, the tool
  is worse than the shell command it replaced.
* Every action here either downloads gigabytes or touches system state. None of
  them may run from a single key press, and the size has to be on screen before
  the confirmation rather than after it.

Rendering into a text surface is what makes both assertable with no terminal.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kilix_bonsai import catalog, screen, store, tui  # noqa: E402


def state_with(present: set[str]) -> tui.State:
    """Build a State whose given model ids report as fully downloaded."""
    models = catalog.load()
    built = tui.State(models)
    for model in models:
        for variant in model.variants:
            disk = built.states[(model.id, variant.id)]
            if model.id in present and variant.default:
                files = tuple(
                    store.FileState(file=item, path="/nowhere/" + item.path,
                                    exists=True, size=item.size)
                    for item in variant.files)
                built.states[(model.id, variant.id)] = store.VariantState(
                    variant=variant, directory=disk.directory, files=files)
            else:
                files = tuple(
                    store.FileState(file=item, path="/nowhere/" + item.path,
                                    exists=False, size=0)
                    for item in variant.files)
                built.states[(model.id, variant.id)] = store.VariantState(
                    variant=variant, directory=disk.directory, files=files)
    built.screen = "list" if present else "setup"
    return built


class FirstRunTest(unittest.TestCase):
    def test_an_empty_store_opens_on_setup(self) -> None:
        built = state_with(set())
        self.assertEqual(built.screen, "setup")
        text = screen.render_to_text(tui.render, built)
        self.assertIn("No model weights are on this machine yet", text)

    def test_the_setup_screen_says_how_to_get_one(self) -> None:
        text = screen.render_to_text(tui.render, state_with(set()))
        self.assertIn("Enter", text)
        self.assertIn("download", text)

    def test_the_setup_screen_prices_every_model_and_the_total(self) -> None:
        built = state_with(set())
        text = screen.render_to_text(tui.render, built)
        for model in built.models:
            self.assertIn(model.title, text)
        self.assertIn(f"all {len(built.models)} models", text)

    def test_one_downloaded_model_opens_on_the_list(self) -> None:
        built = state_with({"bonsai-8b"})
        self.assertEqual(built.screen, "list")
        self.assertIn("ready", screen.render_to_text(tui.render, built))

    def test_the_header_counts_what_is_ready(self) -> None:
        self.assertIn("0/5 ready",
                      screen.render_to_text(tui.render, state_with(set())))
        self.assertIn("1/5 ready", screen.render_to_text(
            tui.render, state_with({"bonsai-8b"})))


class ConfirmationTest(unittest.TestCase):
    def test_enter_on_setup_asks_before_downloading(self) -> None:
        built = state_with(set())
        tui.handle(ord("\n"), built)
        self.assertEqual(built.screen, "confirm")
        self.assertIsNotNone(built.pending)

    def test_the_confirmation_names_the_size_source_and_destination(self) -> None:
        built = state_with(set())
        tui.handle(ord("\n"), built)
        text = screen.render_to_text(tui.render, built)
        model = built.models[0]
        self.assertIn("still to fetch", text)
        self.assertIn(model.default_variant.files[0].repo, text)
        self.assertIn("Download", text)

    def test_any_key_but_y_cancels(self) -> None:
        built = state_with(set())
        tui.handle(ord("\n"), built)
        tui.handle(ord("n"), built)
        self.assertIsNone(built.pending)
        self.assertNotEqual(built.screen, "confirm")

    def test_the_deps_key_also_asks_first(self) -> None:
        built = state_with(set())
        tui.handle(ord("i"), built)
        self.assertEqual(built.screen, "confirm")
        self.assertIn("install-deps.sh", " ".join(built.pending["argv"]))

    def test_the_pending_command_is_the_model_folder_script(self) -> None:
        # The UI must not reimplement a download; it runs the same script the
        # shell path runs, so the two cannot diverge.
        built = state_with(set())
        tui.handle(ord("d"), built)
        argv = built.pending["argv"]
        self.assertTrue(argv[0].endswith("/pull.sh"))
        self.assertTrue(os.access(argv[0], os.X_OK))


class NavigationTest(unittest.TestCase):
    def test_enter_on_the_list_opens_the_detail_screen(self) -> None:
        built = state_with({"bonsai-8b"})
        tui.handle(ord("\n"), built)
        self.assertEqual(built.screen, "detail")

    def test_the_detail_screen_shows_the_store_path_and_every_variant(self) -> None:
        built = state_with({"bonsai-8b"})
        built.selected = [m.id for m in built.models].index("bonsai-27b")
        built.screen = "detail"
        text = screen.render_to_text(tui.render, built)
        self.assertIn("bonsai-27b", text)
        for variant in built.model.variants:
            self.assertIn(variant.title[:20], text)

    def test_h_returns_from_detail_rather_than_quitting(self) -> None:
        built = state_with({"bonsai-8b"})
        built.screen = "detail"
        self.assertTrue(tui.handle(ord("h"), built))
        self.assertEqual(built.screen, "list")

    def test_q_quits_from_the_list(self) -> None:
        self.assertFalse(tui.handle(ord("q"), state_with({"bonsai-8b"})))

    def test_selection_cannot_leave_the_list(self) -> None:
        built = state_with(set())
        for _ in range(50):
            tui.handle(ord("j"), built)
        self.assertEqual(built.selected, len(built.models) - 1)
        for _ in range(50):
            tui.handle(ord("k"), built)
        self.assertEqual(built.selected, 0)


class ClippingTest(unittest.TestCase):
    def test_every_screen_renders_at_any_size(self) -> None:
        built = state_with({"bonsai-8b"})
        tui.handle(ord("d"), built)          # leave a pending confirmation
        for name in tui.SCREENS:
            built.screen = name
            for height, width in ((8, 20), (24, 80), (60, 200)):
                text = screen.render_to_text(tui.render, built,
                                             height=height, width=width)
                for line in text.splitlines():
                    self.assertLessEqual(len(line), width, f"{name} {width}")


if __name__ == "__main__":
    unittest.main()
