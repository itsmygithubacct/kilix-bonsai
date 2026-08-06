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


class CpuRuntimeBuildTest(unittest.TestCase):
    """Chat's CPU fallback needs one build; it is offered, never assumed."""

    def _chat_state(self) -> tui.State:
        built = state_with({"bonsai-8b"})
        built.selected = [m.id for m in built.models].index("bonsai-8b")
        return built

    def test_launch_consults_the_build_offer(self) -> None:
        import inspect
        self.assertIn("offer_cpu_build", inspect.getsource(tui.State.launch))

    def _offer(self, built: tui.State, *, missing: tuple[str, ...] = (),
               apt: bool = True) -> bool:
        saved = (tui.chat.cpu_fallback_pending_build, tui.chat.cpu_runtime,
                 tui.chat.cpu_build_missing_packages, tui.shutil.which)
        tui.chat.cpu_fallback_pending_build = lambda: True
        tui.chat.cpu_runtime = lambda: "/usr/bin/bonsai-cpu"
        tui.chat.cpu_build_missing_packages = lambda: missing
        tui.shutil.which = lambda name: (
            "/usr/bin/apt-get" if apt and name == "apt-get" else None)
        try:
            return built.offer_cpu_build(built.model)
        finally:
            (tui.chat.cpu_fallback_pending_build, tui.chat.cpu_runtime,
             tui.chat.cpu_build_missing_packages, tui.shutil.which) = saved

    def test_a_pending_build_is_offered_behind_the_standard_confirm(self) -> None:
        built = self._chat_state()
        self.assertTrue(self._offer(built))
        self.assertEqual(built.screen, "confirm")
        self.assertEqual(built.pending["argv"], ["/usr/bin/bonsai-cpu", "build"])
        # Accepting is a launch attempt, so success must open chat.
        self.assertIs(built.pending["then_launch"], built.model)
        text = screen.render_to_text(tui.render, built)
        # The invariant: what it costs is on screen before the confirmation.
        self.assertIn("compiles", text)
        self.assertIn("git clone", text)
        self.assertIn("y confirm", text)
        # And no download-only promise on a compile.
        self.assertNotIn("resumes from where it stopped", text)

    def test_missing_build_tools_are_priced_into_the_offer(self) -> None:
        # The fresh-box shape: build-essential provisioned, cmake absent. The
        # offer must install the named packages first, in the same confirm,
        # rather than run a build that refuses after the clone.
        built = self._chat_state()
        self.assertTrue(self._offer(built, missing=("cmake",)))
        self.assertEqual(built.screen, "confirm")
        self.assertEqual(
            built.pending["steps"],
            [["sudo", "apt-get", "install", "-y", "--", "cmake"],
             ["/usr/bin/bonsai-cpu", "build"]])
        self.assertIs(built.pending["then_launch"], built.model)
        text = screen.render_to_text(tui.render, built)
        self.assertIn("cmake", text)
        self.assertIn("sudo apt-get install", text)
        self.assertIn("y confirm", text)

    def test_no_apt_means_a_refusal_with_the_words(self) -> None:
        # Not a Debian machine and the tools are missing: nothing honest to
        # run, so the launch is blocked with the remedy named instead of a
        # confirm screen whose yes would fail.
        built = self._chat_state()
        self.assertTrue(self._offer(built, missing=("cmake",), apt=False))
        self.assertIsNone(built.pending)
        self.assertNotEqual(built.screen, "confirm")
        self.assertIn("cmake", built.message)
        self.assertIn("package manager", built.message)

    def test_confirm_runs_the_steps_in_order_and_stops_on_failure(self) -> None:
        ran: list[list[str]] = []
        saved = tui.provision.run

        def fake_run(argv, cwd=None):
            ran.append(list(argv))
            return 1 if argv[0] == "falls-over" else 0

        tui.provision.run = fake_run
        try:
            built = self._chat_state()
            built.pending = {"heading": "x", "detail": "", "argv": ["a"],
                             "steps": [["falls-over"], ["never-runs"]],
                             "then_launch": None}
            built.screen = "confirm"
            built.confirm()
        finally:
            tui.provision.run = saved
        self.assertEqual(ran, [["falls-over"]])
        self.assertIn("exited 1", built.message)

    def test_a_successful_confirmed_build_launches_chat(self) -> None:
        launched: list[str] = []
        saved_run = tui.provision.run
        tui.provision.run = lambda argv, cwd=None: 0
        try:
            built = self._chat_state()
            model = built.model
            built.launch = lambda m: launched.append(m.id)  # type: ignore
            built.pending = {"heading": "x", "detail": "", "argv": ["ok"],
                             "steps": None, "then_launch": model}
            built.screen = "confirm"
            built.confirm()
        finally:
            tui.provision.run = saved_run
        self.assertEqual(launched, [model.id])

    def test_no_offer_when_nothing_is_pending(self) -> None:
        built = self._chat_state()
        saved = tui.chat.cpu_fallback_pending_build
        tui.chat.cpu_fallback_pending_build = lambda: False
        try:
            self.assertFalse(built.offer_cpu_build(built.model))
        finally:
            tui.chat.cpu_fallback_pending_build = saved
        self.assertIsNone(built.pending)

    def test_the_detail_screen_reports_the_cpu_runtime_honestly(self) -> None:
        built = self._chat_state()
        built.screen = "detail"
        saved = tui.chat.cpu_runtime_built
        try:
            tui.chat.cpu_runtime_built = lambda: False
            text = screen.render_to_text(tui.render, built)
            self.assertIn("cpu runtime", text)
            self.assertIn("not built", text)
            self.assertIn("bonsai-cpu build", text)
            tui.chat.cpu_runtime_built = lambda: True
            text = screen.render_to_text(tui.render, built)
            self.assertIn("cpu runtime  built", text)
        finally:
            tui.chat.cpu_runtime_built = saved


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
    def test_every_store_screen_uses_the_canonical_kilix_shell(self) -> None:
        built = state_with({"bonsai-8b"})
        built.pending = {
            "heading": "Download Bonsai",
            "detail": "1 file",
            "argv": ["pull.sh"],
        }
        for name in tui.SCREENS:
            with self.subTest(screen=name):
                built.screen = name
                frame = screen.render_to_text(
                    tui.render, built, height=24, width=100)
                lines = frame.splitlines()
                self.assertIn("KILIX TUI", lines[0])
                self.assertIn("▶", lines[1])
                self.assertTrue(lines[2].startswith("─"))
                self.assertNotIn(" // ", frame)

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
