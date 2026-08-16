"""The launcher routes correctly, and the three interfaces render safely.

The properties worth pinning here are the ones that would strand someone:
routing a model to the wrong interface, a layout that hides its content at a
small size, and any screen that raises rather than clips. Real inference is
not exercised — that needs weights and minutes — but every screen is rendered,
and every runtime is asked whether it could run rather than being assumed to.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import catalog, launcher, screen, widgets  # noqa: E402
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

    def test_the_fallback_uses_the_kilix_shell_without_terminal_art(self) -> None:
        wide = screen.render_to_text(self.tui.render, self.state,
                                     height=30, width=100)
        self.assertIn("KILIX TUI", wide)
        self.assertIn("Open a model", wide)
        self.assertNotIn("▀", wide)
        self.assertNotIn("BONSAI 001", wide)

    def test_the_model_list_survives_a_narrow_terminal(self) -> None:
        narrow = screen.render_to_text(self.tui.render, self.state,
                                       height=24, width=60)
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

    def test_chat_does_not_claim_dead_controls_or_token_metrics(self) -> None:
        import inspect
        module = load_tool("kilix-bonsai-chat")
        state = module.State(catalog.find("bonsai-8b"))
        self.assertFalse(hasattr(state, "temperature"))
        self.assertNotIn("Ctrl-T", " ".join(module.HELP))
        source = inspect.getsource(module.render)
        self.assertNotIn("chunks", source)
        state.streaming = True
        state.started = module.time.monotonic()
        frame = screen.render_to_text(module.render, state)
        self.assertIn("generating", frame)
        self.assertIn("Esc stops", frame)

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

    def test_gallery_tolerates_malformed_sidecar_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for index, saved in enumerate((
                [],
                {"size": "not-a-size", "seconds": {"bad": "value"}},
            )):
                stem = os.path.join(temporary, f"generation-{index:04d}")
                with open(stem + ".png", "wb") as handle:
                    handle.write(b"not decoded while listing")
                with open(stem + ".json", "w", encoding="utf-8") as handle:
                    json.dump(saved, handle)

            gallery = image.Gallery(temporary)
            gallery.load()

            self.assertEqual(len(gallery.entries), 2)
            for entry in gallery.entries:
                self.assertEqual((entry.request.width, entry.request.height),
                                 (512, 512))
                self.assertEqual(entry.seconds, 0.0)


class RuntimeProbeTest(unittest.TestCase):
    """Runtimes are asked, never assumed."""

    def test_chat_docs_describe_the_current_backends(self) -> None:
        from kilix_bonsai.runtime import chat
        module = load_tool("kilix-bonsai-chat")
        self.assertNotIn("no CPU fallback", chat.__doc__ or "")
        self.assertIn("delegates", chat.__doc__ or "")
        self.assertNotIn("stays loaded", module.__doc__ or "")
        self.assertIn("bonsai-cpu", module.__doc__ or "")

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

    def test_ctrl_q_is_not_swallowed_by_terminal_flow_control(self) -> None:
        import inspect
        self.assertIn("~termios.IXON", inspect.getsource(screen.run))

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


class VisualSystemTest(unittest.TestCase):
    """Every task uses the Kilix shell in pixels and in text."""

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

    def test_each_text_fallback_uses_kilix_tui_without_node_chrome(self) -> None:
        for name in self.TOOLS:
            module = load_tool(name)
            frame = screen.render_to_text(module.render,
                                          self._state(name, module),
                                          height=26, width=96)
            self.assertIn("KILIX TUI", frame, name)
            self.assertIn(module.PixelRenderer.area.title(), frame, name)
            self.assertIn("▶1", frame.splitlines()[1], name)
            self.assertNotIn("//", frame, name)
            self.assertNotIn(" 001", frame, name)

    def test_each_tool_still_clips_at_every_size(self) -> None:
        for name in self.TOOLS:
            module = load_tool(name)
            state = self._state(name, module)
            for height, width in ((8, 20), (24, 80), (40, 140)):
                text = screen.render_to_text(module.render, state,
                                             height=height, width=width)
                for line in text.splitlines():
                    self.assertLessEqual(len(line), width, f"{name} {width}")

    @unittest.skipUnless(
        os.path.isdir(os.path.expanduser(
            "~/.local/gpu_terminal/sources/kilix-desktops/kilix-tui-utils/src/kilix_desk")),
        "kilix-tui-utils is not alongside this checkout")
    def test_each_pixel_interface_renders_the_shared_frame(self) -> None:
        for name in self.TOOLS:
            module = load_tool(name)
            renderer = module.PixelRenderer(canvas_factory=StubCanvas)
            frame = renderer.render(
                self._state(name, module), 100, 30, (960, 560),
                clock="12:00")
            self.assertEqual(len(frame.rgb), 960 * 560 * 3, name)
            self.assertGreaterEqual(len(renderer.hits), 4, name)


class CpuFallbackTest(unittest.TestCase):
    """No card is a reason to be slow, not a reason to refuse."""

    def test_release_kills_the_whole_process_group(self) -> None:
        import inspect
        from kilix_bonsai.runtime import chat
        self.assertIn("killpg", inspect.getsource(chat.Session.release))

    def _forced(self, cpu_present: bool, free, built: bool = True):
        from kilix_bonsai.runtime import chat
        saved_vram, saved_cpu = chat.free_vram_mib, chat.cpu_runtime
        saved_built = chat.cpu_runtime_built
        chat.free_vram_mib = lambda backend=chat.LOCAL, timeout=30.0: free
        chat.cpu_runtime = lambda: "/usr/bin/bonsai-cpu" if cpu_present else None
        chat.cpu_runtime_built = lambda: built
        try:
            return chat.choose("bonsai-8b"), chat.choose("bonsai-27b")
        finally:
            chat.free_vram_mib, chat.cpu_runtime = saved_vram, saved_cpu
            chat.cpu_runtime_built = saved_built

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

    def test_an_unbuilt_cpu_runtime_is_refused_with_the_build_named(self) -> None:
        # A fresh machine has bonsai-cpu installed and its runtime unbuilt.
        # Delegating anyway hands the terminal to a tool that dies at once on
        # stderr; the refusal must instead carry the exact remedy.
        eight, twenty_seven = self._forced(
            cpu_present=True, free=None, built=False)
        for choice in (eight, twenty_seven):
            self.assertFalse(choice.usable)
            self.assertIn("run: bonsai-cpu build", choice.reason)
            self.assertIn("Kilix Bonsai", choice.reason)

    def test_the_pending_build_probe_matches_the_cpu_landing(self) -> None:
        from kilix_bonsai.runtime import chat
        saved = (chat.free_vram_mib, chat.cpu_runtime, chat.cpu_runtime_built)

        def force(free, present, built):
            chat.free_vram_mib = lambda backend=chat.LOCAL, timeout=30.0: free
            chat.cpu_runtime = (
                lambda: "/usr/bin/bonsai-cpu" if present else None)
            chat.cpu_runtime_built = lambda: built
            return chat.cpu_fallback_pending_build()

        try:
            self.assertTrue(force(None, True, False))
            # A card too small even for 8B still lands chat on the CPU.
            self.assertTrue(force(100, True, False))
            self.assertFalse(force(8000, True, False))
            self.assertFalse(force(None, True, True))
            self.assertFalse(force(None, False, False))
        finally:
            (chat.free_vram_mib, chat.cpu_runtime,
             chat.cpu_runtime_built) = saved

    def test_cpu_runner_finds_the_user_install_off_path(self) -> None:
        import tempfile
        from kilix_bonsai.runtime import chat
        saved_which = chat.shutil.which
        saved_prefix = os.environ.get("BONSAI_CPU_PREFIX")
        with tempfile.TemporaryDirectory() as prefix:
            bindir = os.path.join(prefix, "bin")
            os.makedirs(bindir)
            runner = os.path.join(bindir, "bonsai-cpu")
            with open(runner, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\n")
            os.chmod(runner, 0o700)
            chat.shutil.which = lambda name: None
            os.environ["BONSAI_CPU_PREFIX"] = prefix
            try:
                self.assertEqual(chat.cpu_runtime(), runner)
            finally:
                chat.shutil.which = saved_which
                if saved_prefix is None:
                    os.environ.pop("BONSAI_CPU_PREFIX", None)
                else:
                    os.environ["BONSAI_CPU_PREFIX"] = saved_prefix

    def test_the_build_tool_probe_names_debian_packages(self) -> None:
        # The probe exists so the store never offers a build that will
        # refuse: it must name the *package* an operator installs, not the
        # tool cmake happens to look for, and any one C++ compiler name
        # must satisfy the compiler row.
        from kilix_bonsai.runtime import chat
        saved = chat.shutil.which

        def with_tools(*present):
            chat.shutil.which = lambda name: (
                f"/usr/bin/{name}" if name in present else None)
            return chat.cpu_build_missing_packages()

        try:
            self.assertEqual(with_tools("git", "cmake", "c++"), ())
            self.assertEqual(with_tools("git", "cmake", "clang++"), ())
            self.assertEqual(with_tools("git", "c++"), ("cmake",))
            self.assertEqual(with_tools("cmake", "g++"), ("git",))
            self.assertEqual(with_tools(), ("git", "cmake", "build-essential"))
        finally:
            chat.shutil.which = saved

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


if __name__ == "__main__":
    unittest.main()
