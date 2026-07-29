"""Markdown-lite for a transcript pane: spans, not a parser.

A model writes prose with a little structure — bold runs, bullet lists,
fenced code, the odd heading. Rendering that faithfully needs exactly one
idea: a line is a list of (attr, text) spans, wrapped so that a bold run
split across lines stays bold and a code block keeps its indentation. This
module is curses-free apart from the attribute constants, so every promise
it makes is assertable headlessly.
"""
from __future__ import annotations

import curses
import re

BOLD = curses.A_BOLD
DIM = curses.A_DIM

Span = tuple  # (attr: int, text: str)

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_BULLET_RE = re.compile(r"([-*]|\d+[.)])\s+")


def _inline(text: str, base: int = 0) -> list[Span]:
    """Apply **bold** markers, stripping them from the text."""
    spans: list[Span] = []
    position = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > position:
            spans.append((base, text[position:match.start()]))
        spans.append((base | BOLD, match.group(1)))
        position = match.end()
    if position < len(text):
        spans.append((base, text[position:]))
    return spans


def _words(spans: list[Span]) -> list[list[Span]]:
    """Split spans into words; a word may cross an attribute boundary."""
    words: list[list[Span]] = []
    current: list[Span] = []
    for attr, text in spans:
        pieces = text.split(" ")
        for index, piece in enumerate(pieces):
            if index and current:
                words.append(current)
                current = []
            if piece:
                current.append((attr, piece))
    if current:
        words.append(current)
    return words


def _split_word(word: list[Span], width: int) -> tuple[list[Span], list[Span]]:
    """Cut one word at width characters, preserving attributes."""
    head: list[Span] = []
    tail: list[Span] = []
    remaining = width
    for attr, text in word:
        if remaining <= 0:
            tail.append((attr, text))
        elif len(text) <= remaining:
            head.append((attr, text))
            remaining -= len(text)
        else:
            head.append((attr, text[:remaining]))
            tail.append((attr, text[remaining:]))
            remaining = 0
    return head, tail


def _fill(words: list[list[Span]], width: int) -> list[list[Span]]:
    """Greedy word-fill into span lines of at most width characters."""
    if width <= 0:
        return []
    lines: list[list[Span]] = []
    current: list[Span] = []
    used = 0
    for word in words:
        length = sum(len(text) for _, text in word)
        while length > width:                    # a URL, or a code blob
            if current:
                lines.append(current)
                current, used = [], 0
            head, word = _split_word(word, width)
            lines.append(head)
            length = sum(len(text) for _, text in word)
        needed = length + (1 if used else 0)
        if used and used + needed > width:
            lines.append(current)
            current, used = [], 0
            needed = length
        if used:
            current.append((0, " "))
        current.extend(word)
        used += needed
    if current:
        lines.append(current)
    return lines or [[]]


def _indented(lines: list[list[Span]], first: str,
              rest: str) -> list[list[Span]]:
    out = []
    for index, line in enumerate(lines):
        prefix = first if index == 0 else rest
        out.append(([(0, prefix)] if prefix else []) + line)
    return out


def style(text: str, width: int) -> list[list[Span]]:
    """Turn model output into wrapped span lines for a pane of `width`."""
    if width <= 0:
        return []
    lines: list[list[Span]] = []
    in_code = False
    for raw in text.split("\n"):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            # Hard chunks, not word wrap: code means what its columns say.
            if not raw:
                lines.append([(DIM, "")])
            for start in range(0, len(raw), width):
                lines.append([(DIM, raw[start:start + width])])
            continue
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if not stripped:
            lines.append([])
            continue
        pad = " " * indent
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            filled = _fill(_words([(BOLD, heading)]), max(1, width - indent))
            lines.extend(_indented(filled, pad, pad))
            continue
        bullet = _BULLET_RE.match(stripped)
        if bullet:
            marker = bullet.group(1)
            hang = " " * (indent + len(marker) + 1)
            body = _words(_inline(stripped[bullet.end():]))
            filled = _fill(body, max(1, width - len(hang)))
            lines.extend(_indented(filled, pad + marker + " ", hang))
            continue
        filled = _fill(_words(_inline(stripped)), max(1, width - indent))
        lines.extend(_indented(filled, pad, pad))
    return lines


def plain(lines: list[list[Span]]) -> str:
    """The text of span lines, styling discarded — what the tests read."""
    return "\n".join("".join(text for _, text in line) for line in lines)


def fold_line(tokens: int, streaming: bool) -> str:
    if streaming:
        return f"▸ thinking… {tokens} tokens"
    return f"▸ thought for {tokens} tokens"
