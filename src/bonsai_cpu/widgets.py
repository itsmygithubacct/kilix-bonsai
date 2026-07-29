"""The two widgets a chat interface needs: a line editor and a scrollback.

A copy of the shared Kilix widget set, for the same reason screen.py is a
copy: this repository stands alone. One set of editing keys, and one place
where the awkward parts — wrapping, clamping a scroll offset to content that
changed under it, a cursor that must stay visible — are got right.
"""
from __future__ import annotations

import curses
from dataclasses import dataclass, field

BACKSPACE = (curses.KEY_BACKSPACE, 127, 8)
DELETE = (curses.KEY_DC,)
LEFT = (curses.KEY_LEFT,)
RIGHT = (curses.KEY_RIGHT,)
HOME = (curses.KEY_HOME, 1)                       # Ctrl-A
END = (curses.KEY_END, 5)                         # Ctrl-E
KILL_LINE = (11,)                                 # Ctrl-K
KILL_WORD = (23,)                                 # Ctrl-W
SUBMIT = (ord("\n"), ord("\r"), curses.KEY_ENTER)


@dataclass
class Editor:
    """A single-line editor with the readline keys people already know."""

    text: str = ""
    cursor: int = 0
    history: list[str] = field(default_factory=list)
    history_index: int = -1

    def handle(self, key: int) -> bool:
        """Consume a key. Returns True when the key was the editor's."""
        if key in BACKSPACE:
            if self.cursor:
                self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
                self.cursor -= 1
            return True
        if key in DELETE:
            self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
            return True
        if key in LEFT:
            self.cursor = max(0, self.cursor - 1)
            return True
        if key in RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
            return True
        if key in HOME:
            self.cursor = 0
            return True
        if key in END:
            self.cursor = len(self.text)
            return True
        if key in KILL_LINE:
            self.text = self.text[:self.cursor]
            return True
        if key in KILL_WORD:
            head = self.text[:self.cursor].rstrip()
            cut = head.rfind(" ") + 1
            self.text = self.text[:cut] + self.text[self.cursor:]
            self.cursor = cut
            return True
        if key == curses.KEY_UP and self.history:
            self.history_index = min(len(self.history) - 1,
                                     self.history_index + 1)
            self.text = self.history[-1 - self.history_index]
            self.cursor = len(self.text)
            return True
        if key == curses.KEY_DOWN and self.history:
            self.history_index = max(-1, self.history_index - 1)
            self.text = ("" if self.history_index < 0
                         else self.history[-1 - self.history_index])
            self.cursor = len(self.text)
            return True
        if 32 <= key < 127 or key > 255:
            try:
                self.text = (self.text[:self.cursor] + chr(key)
                             + self.text[self.cursor:])
            except ValueError:
                return False
            self.cursor += 1
            return True
        return False

    def submit(self) -> str:
        """Take the current text, remember it, and clear."""
        value = self.text
        if value.strip():
            self.history.append(value)
        self.text, self.cursor, self.history_index = "", 0, -1
        return value

    def set(self, value: str) -> None:
        self.text, self.cursor = value, len(value)

    def view(self, width: int) -> tuple[str, int]:
        """Return the visible slice and where the cursor sits inside it.

        Scrolls horizontally so the cursor is always on screen — without this
        a long prompt silently loses its own end.
        """
        if width <= 1:
            return "", 0
        start = max(0, self.cursor - width + 1)
        return self.text[start:start + width], self.cursor - start


def wrap(text: str, width: int) -> list[str]:
    """Wrap on word boundaries, preserving deliberate blank lines."""
    if width <= 0:
        return []
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            while len(word) > width:                 # a URL, or a code blob
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:width])
                word = word[width:]
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


@dataclass
class Scrollback:
    """A view over wrapped lines that follows the tail until you scroll up."""

    offset: int = 0          # lines scrolled up from the bottom
    following: bool = True

    def clamp(self, total: int, height: int) -> int:
        """Return the first visible line index for this content and height."""
        if height <= 0:
            return 0
        if self.following:
            self.offset = 0
        limit = max(0, total - height)
        self.offset = max(0, min(self.offset, limit))
        return limit - self.offset

    def scroll(self, delta: int) -> None:
        self.offset = max(0, self.offset + delta)
        # Scrolling up detaches from the tail; scrolling back to the bottom
        # re-attaches, which is what every chat client does and what people
        # expect without being told.
        self.following = self.offset == 0

    def to_end(self) -> None:
        self.offset, self.following = 0, True
