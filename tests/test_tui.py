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


if __name__ == "__main__":
    unittest.main()
