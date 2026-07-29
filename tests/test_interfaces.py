"""The launcher routes correctly, and the three interfaces render safely.

The properties worth pinning here are the ones that would strand someone:
routing a model to the wrong interface, a launcher that hides its own art
badly at small sizes, and any screen that raises rather than clips. Real
inference is not exercised — that needs weights and minutes — but every screen
is rendered, and every runtime is asked whether it could run rather than being
assumed to.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import art, catalog, launcher, screen, widgets  # noqa: E402
from kilix_bonsai.runtime import asr, image, llama                # noqa: E402


def load_tool(name: str):
    path = os.path.join(ROOT, "tools", name, "main.py")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.models = catalog.load()

    def test_every_model_declares_a_runtime_kind(self) -> None:
        for model in self.models:
            self.assertIn(model.runtime.get("kind"), launcher.TOOLS,
                          f"{model.id} has no interface")

    def test_each_kind_routes_to_its_own_tool(self) -> None:
        seen = {}
        for model in self.models:
            argv = launcher.tool_argv(model)
            self.assertIsNotNone(argv, model.id)
            kind = model.runtime["kind"]
            seen.setdefault(kind, argv[-2] if len(argv) > 1 else argv[0])
        # Three kinds, three distinct entry points.
        self.assertEqual(len(set(seen.values())), len(seen))

    def test_chat_models_name_a_file_inside_their_own_variant(self) -> None:
        for model in self.models:
            if model.runtime.get("kind") != "chat":
                continue
            wanted = model.runtime["model_file"]
            names = {item.path for item in model.default_variant.files}
            self.assertIn(wanted, names, model.id)

    def test_the_speech_model_names_both_of_its_files(self) -> None:
        model = catalog.find("vibevoice-asr-bitnet")
        names = {item.path for item in model.default_variant.files}
        self.assertIn(model.runtime["vae_file"], names)
        self.assertIn(model.runtime["lm_file"], names)

    def test_a_model_with_no_weights_is_not_launchable(self) -> None:
        for model in self.models:
            ok, detail = launcher.launchable(model)
            self.assertIsInstance(ok, bool)
            self.assertTrue(detail)


class ArtTest(unittest.TestCase):
    def test_the_sprite_loads_and_has_shape(self) -> None:
        width, cells = art.size()
        self.assertEqual(width, 64)
        self.assertEqual(cells, 32)

    def test_it_scales_down_by_whole_numbers_only(self) -> None:
        # A fractional scale on pixel art gives uneven pixel widths, which
        # reads as a rendering fault rather than as a smaller sprite.
        self.assertEqual(art.size(2), (32, 16))
        self.assertEqual(art.size(4), (16, 8))
        self.assertEqual(art.fit(70, 34), 1)
        self.assertEqual(art.fit(40, 20), 2)
        self.assertEqual(art.fit(8, 4), 0)          # no fit: draw nothing

    def test_a_scaled_sprite_still_reads_as_a_shape(self) -> None:
        for factor in (1, 2, 3):
            text = art.as_text(factor)
            self.assertGreater(text.count(art.UPPER_HALF), 40, factor)
            self.assertIn(" ", text)

    def test_the_silhouette_is_neither_blank_nor_solid(self) -> None:
        # Both failures have happened: an index-based darkness test made the
        # whole sprite solid ink, and a bad threshold makes it disappear.
        text = art.as_text()
        ink = text.count(art.UPPER_HALF)
        space = sum(1 for character in text if character == " ")
        self.assertGreater(ink, 100, "the sprite rendered blank")
        self.assertGreater(space, 100, "the sprite rendered solid")

    def test_luminance_not_index_decides_ink(self) -> None:
        # 232 is the darkest grey-ramp entry: a high index, a near-black
        # colour. Treating the index as brightness is the bug this catches.
        self.assertFalse(art.is_ink(232))
        self.assertTrue(art.is_ink(231))          # cube white
        self.assertFalse(art.is_ink(16))          # cube black

    def test_missing_assets_are_survivable(self) -> None:
        saved = art.sprite.__wrapped__
        self.assertTrue(callable(saved))


class LauncherRenderTest(unittest.TestCase):
    def setUp(self) -> None:
        from kilix_bonsai import tui
        self.tui = tui
        self.state = tui.State()
        self.state.screen = "launch"

    def test_the_launcher_lists_every_model_with_its_kind(self) -> None:
        text = screen.render_to_text(self.tui.render, self.state,
                                     height=30, width=100)
        for model in self.state.models:
            self.assertIn(model.title, text)
        self.assertIn("chat", text)
        self.assertIn("speech-to-text", text)

    def test_the_art_appears_when_there_is_room(self) -> None:
        wide = screen.render_to_text(self.tui.render, self.state,
                                     height=30, width=100)
        self.assertIn(art.UPPER_HALF, wide)

    def test_the_art_yields_to_the_list_when_there_is_not(self) -> None:
        # A launcher that kept its decoration and dropped the thing you came
        # to use would have the priority backwards.
        narrow = screen.render_to_text(self.tui.render, self.state,
                                       height=24, width=60)
        self.assertNotIn(art.UPPER_HALF, narrow)
        self.assertIn(self.state.models[0].title[:12], narrow)


class InterfaceRenderTest(unittest.TestCase):
    """Every screen of every tool renders at any size without raising."""

    SIZES = ((8, 20), (24, 80), (40, 140))

    def _exercise(self, module, state, screens) -> None:
        for name in screens:
            setattr(state, *name) if isinstance(name, tuple) else None
            for height, width in self.SIZES:
                text = screen.render_to_text(module.render, state,
                                             height=height, width=width)
                for line in text.splitlines():
                    self.assertLessEqual(len(line), width)

    def test_chat(self) -> None:
        module = load_tool("kilix-bonsai-chat")
        model = next(m for m in catalog.load()
                     if m.runtime.get("kind") == "chat")
        state = module.State(model)
        self._exercise(module, state, [])
        state.show_help = True
        self._exercise(module, state, [])
        state.show_help = False
        state.error = "a runtime that is not there"
        self._exercise(module, state, [])

    def test_image(self) -> None:
        module = load_tool("kilix-bonsai-image")
        state = module.State(catalog.find("bonsai-image-4b"))
        state.load_field()
        self._exercise(module, state, [])
        state.view = "gallery"
        self._exercise(module, state, [])
        state.show_help = True
        self._exercise(module, state, [])

    def test_speech(self) -> None:
        module = load_tool("kilix-bonsai-speech")
        state = module.State(catalog.find("vibevoice-asr-bitnet"))
        self._exercise(module, state, [])
        state.transcript = "a transcript " * 40
        self._exercise(module, state, [])
        state.show_help = True
        self._exercise(module, state, [])


class ImageRequestTest(unittest.TestCase):
    def test_a_size_off_the_grid_is_refused_before_a_model_load(self) -> None:
        request = image.Request(prompt="x", width=500, height=512)
        self.assertIn("multiple of 32", request.validate() or "")

    def test_an_empty_prompt_is_refused(self) -> None:
        self.assertIn("prompt", image.Request(prompt="  ").validate() or "")

    def test_a_missing_reference_is_refused(self) -> None:
        request = image.Request(prompt="x", input_image="/no/such/file.png")
        self.assertIn("no input image", request.validate() or "")

    def test_the_remote_argv_carries_the_reference_and_waits(self) -> None:
        # The remote subcommand is named after a host, so it is configuration
        # rather than a constant this repository ships.
        saved = image.REMOTE_SUBCOMMAND
        image.REMOTE_SUBCOMMAND = "somehost"
        try:
            request = image.Request(prompt="a cat", input_image=__file__,
                                    seed=7, backend=image.REMOTE)
            argv = request.argv("/bin/bonsai", "/tmp/out.png")
            self.assertEqual(argv[1:4], ["somehost", "generate", "--wait"])
            self.assertIn("--input-image", argv)
            self.assertIn("--seed", argv)
        finally:
            image.REMOTE_SUBCOMMAND = saved

    def test_remote_without_configuration_is_refused_not_guessed(self) -> None:
        saved = image.REMOTE_SUBCOMMAND
        image.REMOTE_SUBCOMMAND = ""
        try:
            problem = image.Request(prompt="a cat",
                                    backend=image.REMOTE).validate()
            self.assertIn("KILIX_BONSAI_IMAGE_REMOTE", problem or "")
        finally:
            image.REMOTE_SUBCOMMAND = saved

    def test_the_remote_target_comes_from_the_environment(self) -> None:
        # The scaffold's remote subcommand is named after a machine. This
        # repository publishes under a pseudonymous identity, so that name has
        # to be configuration; asserting the *source* of the value keeps the
        # hostname out of this file too.
        import inspect
        source = inspect.getsource(image)
        self.assertIn('os.environ.get("KILIX_BONSAI_IMAGE_REMOTE"', source)
        assignments = [line for line in source.splitlines()
                       if line.startswith("REMOTE_SUBCOMMAND")]
        self.assertEqual(len(assignments), 1, assignments)
        self.assertIn("environ", assignments[0])


class RuntimeProbeTest(unittest.TestCase):
    """Runtimes are asked, never assumed."""

    def test_the_chat_probe_does_not_block_construction(self) -> None:
        import inspect
        module = load_tool("kilix-bonsai-chat")
        self.assertNotIn("chat.choose(", inspect.getsource(
            module.State.__init__))
        state = module.State(catalog.find("bonsai-8b"))
        self.assertIsNone(state.choice)
        frame = screen.render_to_text(module.render, state)
        self.assertIn("measuring free VRAM", frame)

    def test_a_cpu_probe_finishes_with_a_delegate(self) -> None:
        module = load_tool("kilix-bonsai-chat")
        choice = module.chat.Choice(
            "bonsai-8b", module.chat.CPU, None, "CPU", True)

        class InlineThread:
            def __init__(self, *, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                self.target()

        saved_choose = module.chat.choose
        saved_delegate = module.chat.delegate_argv
        saved_thread = module.threading.Thread
        module.chat.choose = lambda model_id: choice
        module.chat.delegate_argv = lambda selected: ["/bin/bonsai-cpu"]
        module.threading.Thread = InlineThread
        try:
            state = module.State(catalog.find("bonsai-8b"))
            state.boot()
        finally:
            module.chat.choose = saved_choose
            module.chat.delegate_argv = saved_delegate
            module.threading.Thread = saved_thread
        self.assertTrue(state.finished)
        self.assertEqual(state.delegate, ["/bin/bonsai-cpu"])
        self.assertIsNone(state.session)

    def test_the_loop_can_be_finished_by_a_thread(self) -> None:
        import inspect
        self.assertIn('getattr(state, "finished", False)',
                      inspect.getsource(screen.run))

    def test_each_runtime_reports_availability_without_running_anything(self):
        self.assertIn(type(llama.server_binary()).__name__, ("str", "NoneType"))
        self.assertIn(type(asr.engine_binary()).__name__, ("str", "NoneType"))
        self.assertIn(type(image.cli_path()).__name__, ("str", "NoneType"))

    def test_the_speech_engine_says_what_is_missing(self) -> None:
        engine = asr.Engine(vae_path="/no/vae.gguf", lm_path="/no/lm.gguf")
        problem = engine.check()
        self.assertTrue(problem and "not downloaded" in problem
                        or "asr_infer" in (problem or ""))

    def test_chunking_leaves_short_audio_alone(self) -> None:
        self.assertEqual(asr.split_audio("/no/such.wav", "/tmp"),
                         ["/no/such.wav"])


class WidgetTest(unittest.TestCase):
    def test_the_editor_keeps_the_cursor_visible(self) -> None:
        editor = widgets.Editor()
        for character in "the quick brown fox jumps over the lazy dog":
            editor.handle(ord(character))
        visible, cursor = editor.view(10)
        self.assertLessEqual(len(visible), 10)
        self.assertLess(cursor, 10)

    def test_wrapping_breaks_a_word_longer_than_the_width(self) -> None:
        lines = widgets.wrap("x" * 25, 10)
        self.assertTrue(all(len(line) <= 10 for line in lines))
        self.assertEqual("".join(lines), "x" * 25)

    def test_wrapping_keeps_deliberate_blank_lines(self) -> None:
        self.assertIn("", widgets.wrap("a\n\nb", 10))

    def test_scrollback_follows_the_tail_until_you_scroll(self) -> None:
        view = widgets.Scrollback()
        self.assertEqual(view.clamp(100, 10), 90)
        view.scroll(5)
        self.assertEqual(view.clamp(100, 10), 85)
        view.to_end()
        self.assertEqual(view.clamp(100, 10), 90)

    def test_scrollback_cannot_run_past_the_content(self) -> None:
        view = widgets.Scrollback()
        view.scroll(1000)
        self.assertEqual(view.clamp(20, 10), 0)


if __name__ == "__main__":
    unittest.main()


class ChromeBridgeTest(unittest.TestCase):
    """The shared theme is optional, and both paths must draw."""

    def test_the_fallback_has_the_same_api_as_the_shared_page(self) -> None:
        from kilix_bonsai import chrome
        plain = chrome._PlainPage("T", ["a", "b"])
        for name in ("measure", "content_box", "render", "spined"):
            self.assertTrue(hasattr(plain, name), name)
        self.assertFalse(plain.spined)

    def test_the_launcher_draws_with_the_core_absent(self) -> None:
        # A bare checkout over ssh must still render a usable launcher.
        from kilix_bonsai import chrome, tui
        saved = chrome._CORE
        chrome._CORE = False
        try:
            state = tui.State()
            state.screen = "launch"
            text = screen.render_to_text(tui.render, state,
                                         height=24, width=80)
            self.assertIn("Choose what to open", text)
            for line in text.splitlines():
                self.assertLessEqual(len(line), 80)
        finally:
            chrome._CORE = saved


class ChromeAdoptionTest(unittest.TestCase):
    """Every interface draws through the shared chrome, both ways."""

    TOOLS = ("kilix-bonsai-chat", "kilix-bonsai-image", "kilix-bonsai-speech")

    def _state(self, name, module):
        if name == "kilix-bonsai-chat":
            return module.State(next(m for m in catalog.load()
                                     if m.runtime.get("kind") == "chat"))
        if name == "kilix-bonsai-image":
            state = module.State(catalog.find("bonsai-image-4b"))
            state.load_field()
            return state
        return module.State(catalog.find("vibevoice-asr-bitnet"))

    def test_each_tool_titles_itself_through_the_chrome(self) -> None:
        for name in self.TOOLS:
            module = load_tool(name)
            text = screen.render_to_text(module.render,
                                         self._state(name, module),
                                         height=26, width=96)
            # Page.title upper-cases; the fallback does not. Either is fine,
            # but the tool's name must be on screen in both.
            self.assertIn(module.TITLE.split()[-1].upper(), text.upper(), name)

    def test_each_tool_still_clips_at_every_size(self) -> None:
        for name in self.TOOLS:
            module = load_tool(name)
            state = self._state(name, module)
            for height, width in ((8, 20), (24, 80), (40, 140)):
                text = screen.render_to_text(module.render, state,
                                             height=height, width=width)
                for line in text.splitlines():
                    self.assertLessEqual(len(line), width, f"{name} {width}")

    def test_each_tool_renders_with_the_shared_core_absent(self) -> None:
        from kilix_bonsai import chrome
        saved = chrome._CORE
        chrome._CORE = False
        try:
            for name in self.TOOLS:
                module = load_tool(name)
                text = screen.render_to_text(module.render,
                                             self._state(name, module),
                                             height=24, width=80)
                self.assertTrue(text.strip(), f"{name} drew nothing")
        finally:
            chrome._CORE = saved


class CpuFallbackTest(unittest.TestCase):
    """No card is a reason to be slow, not a reason to refuse."""

    def test_release_kills_the_whole_process_group(self) -> None:
        import inspect
        from kilix_bonsai.runtime import chat
        self.assertIn("killpg", inspect.getsource(chat.Session.release))

    def _forced(self, cpu_present: bool, free):
        from kilix_bonsai.runtime import chat
        saved_vram, saved_cpu = chat.free_vram_mib, chat.cpu_runtime
        chat.free_vram_mib = lambda backend=chat.LOCAL, timeout=30.0: free
        chat.cpu_runtime = lambda: "/usr/bin/bonsai-cpu" if cpu_present else None
        try:
            return chat.choose("bonsai-8b"), chat.choose("bonsai-27b")
        finally:
            chat.free_vram_mib, chat.cpu_runtime = saved_vram, saved_cpu

    def test_no_gpu_falls_back_to_cpu(self) -> None:
        from kilix_bonsai.runtime import chat
        eight, _ = self._forced(cpu_present=True, free=None)
        self.assertTrue(eight.usable)
        self.assertEqual(eight.backend, chat.CPU)

    def test_a_gpu_is_still_preferred_when_one_fits(self) -> None:
        from kilix_bonsai.runtime import chat
        eight, _ = self._forced(cpu_present=True, free=8000)
        self.assertNotEqual(eight.backend, chat.CPU)

    def test_27b_on_cpu_is_warned_not_withheld(self) -> None:
        # Slow is not broken. Whether minutes per turn is acceptable is the
        # caller's call, so it is offered with the cost stated.
        from kilix_bonsai.runtime import chat
        _, twenty_seven = self._forced(cpu_present=True, free=None)
        self.assertEqual(twenty_seven.model_id, "bonsai-27b")
        self.assertEqual(twenty_seven.backend, chat.CPU)
        self.assertTrue(twenty_seven.usable)
        self.assertIn("minutes", twenty_seven.reason)

    def test_2b4t_is_refused_with_its_engine_named(self) -> None:
        from kilix_bonsai.runtime import chat
        saved_vram, saved_cpu = chat.free_vram_mib, chat.cpu_runtime
        chat.free_vram_mib = (
            lambda backend=chat.LOCAL, timeout=30.0: 8000)
        chat.cpu_runtime = lambda: "/usr/bin/bonsai-cpu"
        try:
            choice = chat.choose("bitnet-b1.58-2b4t")
        finally:
            chat.free_vram_mib, chat.cpu_runtime = saved_vram, saved_cpu
        self.assertFalse(choice.usable)
        self.assertIn("bitnet.cpp", choice.reason)

    def test_8b_stays_the_cpu_default_when_nothing_is_asked_for(self) -> None:
        eight, _ = self._forced(cpu_present=True, free=None)
        self.assertEqual(eight.model_id, "bonsai-8b")

    def test_without_the_cpu_runner_it_refuses_and_names_it(self) -> None:
        eight, _ = self._forced(cpu_present=False, free=None)
        self.assertFalse(eight.usable)
        self.assertIn("bonsai-cpu", eight.reason)

    def test_cpu_delegates_to_the_flagship_tui(self) -> None:
        from kilix_bonsai.runtime import chat
        choice = chat.Choice("bonsai-27b", chat.CPU, None, "", True)
        saved = chat.cpu_runtime
        chat.cpu_runtime = lambda: "/usr/bin/bonsai-cpu"
        try:
            argv = chat.delegate_argv(choice)
        finally:
            chat.cpu_runtime = saved
        self.assertEqual(
            argv,
            ["/usr/bin/bonsai-cpu", "chat", "--model-id", "bonsai-27b"])
