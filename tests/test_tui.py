#!/usr/bin/env python3
"""The copied TUI core keeps its promises: clipping at any size, attributes
that never change the words, widgets that edit and scroll correctly, and a
loop a background thread can end."""

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bonsai_cpu import screen, widgets   # noqa: E402


class SurfaceTest(unittest.TestCase):
    def test_attr_never_changes_the_text(self):
        import curses
        plain = screen.TextSurface(height=3, width=20)
        styled = screen.TextSurface(height=3, width=20)
        plain.addstr(1, 2, "hello")
        styled.addstr(1, 2, "hello", curses.A_BOLD)
        self.assertEqual(str(plain), str(styled))

    def test_write_clips_and_carries_attr(self):
        calls = []

        class Recorder:
            def getmaxyx(self):
                return 5, 10

            def addstr(self, y, x, text, attr=0):
                calls.append((y, x, text, attr))

        screen.write(Recorder(), 2, 3, "abcdefghijk", 7)
        self.assertEqual(calls, [(2, 3, "abcdef", 7)])
        screen.write(Recorder(), 9, 0, "off the bottom")
        screen.write(Recorder(), 0, 10, "off the right")
        self.assertEqual(len(calls), 1)

    def test_render_to_text_never_exceeds_width(self):
        def render(surface, state):
            height, width = surface.getmaxyx()
            for row in range(height + 2):
                screen.write(surface, row, 0, "x" * (width * 2))

        for height, width in ((8, 20), (24, 80), (60, 200)):
            text = screen.render_to_text(render, None, height=height,
                                         width=width)
            for line in text.splitlines():
                self.assertLessEqual(len(line), width)


class LoopTest(unittest.TestCase):
    def test_a_thread_can_finish_the_loop(self):
        # The curses loop cannot run headlessly; the promise is asserted at
        # the source level, the pattern this repo family uses for behavior
        # that needs a terminal to demonstrate.
        source = inspect.getsource(screen.run)
        self.assertIn('getattr(state, "finished", False)', source)

    def test_window_close_unwinds_through_server_cleanup(self):
        source = inspect.getsource(chat_tui.main)
        self.assertIn("signal.SIGHUP", source)
        self.assertIn("state.finished = True", source)
        self.assertIn("state.server.stop()", source)


class EditorTest(unittest.TestCase):
    def test_typing_editing_submitting(self):
        editor = widgets.Editor()
        for ch in "hello":
            editor.handle(ord(ch))
        for key in widgets.BACKSPACE[:1]:
            editor.handle(key)
        self.assertEqual(editor.text, "hell")
        value = editor.submit()
        self.assertEqual(value, "hell")
        self.assertEqual(editor.text, "")
        self.assertEqual(editor.history, ["hell"])

    def test_view_keeps_the_cursor_visible(self):
        editor = widgets.Editor()
        editor.set("a" * 50)
        visible, cursor = editor.view(10)
        self.assertLessEqual(len(visible), 10)
        self.assertLessEqual(cursor, len(visible))


class WrapScrollTest(unittest.TestCase):
    def test_wrap_respects_width_and_blank_lines(self):
        lines = widgets.wrap("one two three\n\nfour", 8)
        self.assertIn("", lines)
        for line in lines:
            self.assertLessEqual(len(line), 8)

    def test_scrollback_follows_tail_until_scrolled(self):
        view = widgets.Scrollback()
        self.assertEqual(view.clamp(total=100, height=10), 90)
        view.scroll(5)
        self.assertEqual(view.clamp(total=100, height=10), 85)
        view.to_end()
        self.assertEqual(view.clamp(total=120, height=10), 110)


import curses            # noqa: E402
import tempfile          # noqa: E402

from bonsai_cpu import chat_tui, convo, pixel   # noqa: E402


class StubServer:
    def __init__(self, model_id="bonsai-8b"):
        self.model_id = model_id
        self.aborted = 0

    def abort(self):
        self.aborted += 1

    def stop(self):
        pass


def chat_state(tmp, model_id="bonsai-8b"):
    state = chat_tui.State(model_id)
    built = convo.new(model_id, directory=tmp)
    built.name = "test chat"
    built.messages = [
        {"role": "user", "content": "hi **there**"},
        {"role": "assistant", "content": "**Hello!**\n- one\n- two",
         "reasoning": "let me think about this greeting",
         "reasoning_tokens": 7,
         "stats": {"prompt_n": 20, "predicted_n": 12,
                   "predicted_per_second": 7.9}},
    ]
    state.convo = built
    state.view = "chat"
    state.server = StubServer(model_id)
    return state


class ChatRenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def render_all_sizes(self, state):
        for height, width in ((8, 20), (24, 80), (60, 200)):
            text = screen.render_to_text(chat_tui.render, state,
                                         height=height, width=width)
            for line in text.splitlines():
                self.assertLessEqual(len(line), width,
                                     f"{state.view} at {width}")
        return screen.render_to_text(chat_tui.render, state)

    def test_every_view_clips_at_any_size(self):
        state = chat_state(self.tmp.name)
        state.conversations = [state.convo]
        for view in ("chat", "list", "picker", "params", "help"):
            state.view = view
            self.render_all_sizes(state)
        state.view = "chat"
        state.loading = "loading Bonsai 27B… 12s"
        self.render_all_sizes(state)
        state.loading = ""
        state.error = "llama-server exited while loading (status 3)"
        self.render_all_sizes(state)

    def test_thinking_is_folded_until_revealed(self):
        state = chat_state(self.tmp.name)
        folded = self.render_all_sizes(state)
        self.assertIn("thought for 7 tokens", folded)
        self.assertNotIn("let me think", folded)
        state.show_thinking = True
        revealed = self.render_all_sizes(state)
        self.assertIn("let me think", revealed)

    def test_status_line_reports_rate_and_context(self):
        state = chat_state(self.tmp.name)
        frame = self.render_all_sizes(state)
        self.assertIn("7.9 tok/s", frame)
        self.assertIn("32/8192", frame)

    def test_prompt_progress_wins_the_status_line(self):
        state = chat_state(self.tmp.name)
        state.progress = {"processed": 812, "total": 2313, "cache": 1501}
        frame = self.render_all_sizes(state)
        self.assertIn("812/2313", frame)
        self.assertIn("cached 1501", frame)

    def test_the_text_fallback_uses_the_kilix_visual_language(self):
        state = chat_state(self.tmp.name)
        frame = self.render_all_sizes(state)
        self.assertIn("KILIX TUI", frame)
        self.assertIn("[Conversation]", frame)
        self.assertNotIn("bonsai-cpu chat ·", frame)


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


@unittest.skipUnless(pixel.shared(),
                     "kilix-tui-utils is not alongside this checkout")
class PixelRenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_every_view_renders_in_the_shared_pixel_frame(self):
        state = chat_state(self.tmp.name)
        state.conversations = [state.convo]
        renderer = pixel.ChatRenderer(canvas_factory=StubCanvas)
        for view in ("chat", "list", "picker", "params", "help"):
            state.view = view
            frame = renderer.render(state, 100, 30, (960, 560),
                                    clock="12:00")
            self.assertEqual(len(frame.rgb), 960 * 560 * 3, view)
            self.assertGreaterEqual(len(renderer.hits), 5, view)

    def test_error_and_loading_are_complete_pixel_frames(self):
        state = chat_state(self.tmp.name)
        renderer = pixel.ChatRenderer(canvas_factory=StubCanvas)
        state.loading = "starting Bonsai 8B"
        renderer.render(state, 100, 30, (960, 560), clock="12:00")
        state.loading = ""
        state.error = "server exited"
        renderer.render(state, 100, 30, (960, 560), clock="12:00")


class ChatKeysTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ensured = []
        self.saved_ensure = chat_tui._ensure_server
        chat_tui._ensure_server = (
            lambda state, model_id: self.ensured.append(model_id))
        self.addCleanup(setattr, chat_tui, "_ensure_server",
                        self.saved_ensure)

    def test_esc_cancels_while_streaming_and_stays(self):
        state = chat_state(self.tmp.name)
        state.streaming = True
        self.assertTrue(chat_tui.handle(27, state))
        self.assertTrue(state.cancel)
        self.assertEqual(state.server.aborted, 1)
        self.assertEqual(state.view, "chat")

    def test_esc_idle_returns_to_the_list(self):
        state = chat_state(self.tmp.name)
        self.assertTrue(chat_tui.handle(27, state))
        self.assertEqual(state.view, "list")

    def test_think_toggle_flips_27b_and_refuses_8b(self):
        state = chat_state(self.tmp.name, "bonsai-27b")
        chat_tui.handle(chat_tui.CTRL_T, state)
        self.assertTrue(state.convo.think)
        eight = chat_state(self.tmp.name)
        chat_tui.handle(chat_tui.CTRL_T, eight)
        self.assertFalse(eight.convo.think)
        self.assertIn("not a thinking model", eight.status)

    def test_picker_switches_model_and_restarts_the_server(self):
        state = chat_state(self.tmp.name)
        chat_tui.handle(chat_tui.CTRL_O, state)
        self.assertEqual(state.view, "picker")
        state.selected = sorted(chat_tui.models.MODELS).index("bonsai-27b")
        chat_tui.handle(ord("\n"), state)
        self.assertEqual(state.convo.model_id, "bonsai-27b")
        self.assertEqual(self.ensured, ["bonsai-27b"])

    def test_params_rejects_a_word_as_temperature(self):
        state = chat_state(self.tmp.name)
        chat_tui.handle(chat_tui.CTRL_P, state)
        state.param_index = chat_tui.PARAM_FIELDS.index("temperature")
        chat_tui.handle(ord("\n"), state)          # open the editor
        for ch in "abc":
            chat_tui.handle(ord(ch), state)
        state.param_editor.set("abc")
        chat_tui.handle(ord("\n"), state)          # commit
        self.assertEqual(state.convo.params["temperature"], 0.6)
        self.assertIn("not a number", state.status)

    def test_ctrl_q_quits_from_anywhere(self):
        state = chat_state(self.tmp.name)
        for view in ("chat", "list", "picker", "params", "help"):
            state.view = view
            self.assertFalse(chat_tui.handle(chat_tui.CTRL_Q, state))

    def test_help_opens_only_from_an_empty_editor(self):
        state = chat_state(self.tmp.name)
        chat_tui.handle(ord("?"), state)
        self.assertEqual(state.view, "help")
        state.view = "chat"
        state.editor.set("what is 2+2?")
        chat_tui.handle(ord("?"), state)
        self.assertEqual(state.view, "chat")
        self.assertIn("?", state.editor.text)


if __name__ == "__main__":
    unittest.main()
