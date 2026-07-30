#!/usr/bin/env python3
"""Styling promises: bold lands exactly where the markers were, code keeps
its columns, lists hang, wrapping loses nothing."""

import curses
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bonsai_cpu import richtext   # noqa: E402


def spans_with(lines, attr):
    return [text for line in lines for a, text in line if a & attr]


class StyleTest(unittest.TestCase):
    def test_bold_is_exactly_the_marked_text(self):
        lines = richtext.style("a **bold** word", 40)
        self.assertEqual(spans_with(lines, curses.A_BOLD), ["bold"])
        self.assertEqual(richtext.plain(lines), "a bold word")

    def test_code_is_dim_and_hard_wrapped(self):
        code = "```\n    indented_call(with_a_very_long_argument_name)\n```"
        lines = richtext.style(code, 20)
        for line in lines:
            for attr, text in line:
                self.assertTrue(attr & curses.A_DIM)
                self.assertLessEqual(len(text), 20)
        self.assertTrue(richtext.plain(lines).startswith("    indented_call"))

    def test_list_items_hang(self):
        lines = richtext.style("- alpha beta gamma delta epsilon", 14)
        text_lines = richtext.plain(lines).split("\n")
        self.assertTrue(text_lines[0].startswith("- alpha"))
        for continuation in text_lines[1:]:
            self.assertTrue(continuation.startswith("  "))
            self.assertFalse(continuation.startswith("- "))

    def test_bold_survives_a_wrap_boundary(self):
        lines = richtext.style("x **one two three four five six**", 12)
        bold_text = " ".join(spans_with(lines, curses.A_BOLD))
        for word in ("one", "six"):
            self.assertIn(word, bold_text)

    def test_wrapping_loses_no_characters(self):
        text = "the quick brown fox jumps over the lazy dog again and again"
        lines = richtext.style(text, 17)
        rebuilt = " ".join(richtext.plain(lines).split())
        self.assertEqual(rebuilt, text)
        for line in richtext.plain(lines).split("\n"):
            self.assertLessEqual(len(line), 17)

    def test_headings_are_bold(self):
        lines = richtext.style("## Results", 30)
        self.assertEqual(spans_with(lines, curses.A_BOLD), ["Results"])

    def test_long_unbreakable_word_is_split_not_dropped(self):
        url = "https://example.invalid/a/very/long/path/that/keeps/going"
        lines = richtext.style(url, 16)
        self.assertEqual(richtext.plain(lines).replace("\n", ""), url)

    def test_fold_line_states_the_count(self):
        self.assertIn("214", richtext.fold_line(214, streaming=False))
        self.assertIn("thinking", richtext.fold_line(41, streaming=True))


if __name__ == "__main__":
    unittest.main()
