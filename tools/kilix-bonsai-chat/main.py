"""kilix-bonsai-chat — route a chat model to the backend that can run it.

On a GPU, generation runs in a worker thread and the UI redraws while one
vendor process streams the turn; Esc abandons it and the process releases the
card. On CPU, the probe ends this smaller interface and hands the terminal to
``bonsai-cpu chat``, whose resident server and persistent conversations are the
flagship experience.

MODEL.json declares which interface and engine a model needs. An engine still
needs an adapter here: unsupported BitNet checkpoints are refused by name
rather than replaced with a different model.
"""
from __future__ import annotations

import curses
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from kilix_bonsai import catalog, pixel, screen, store, text, widgets  # noqa: E402
from kilix_bonsai.runtime import chat                      # noqa: E402

TITLE = "Kilix Bonsai Chat"


class State:
    def __init__(self, model: catalog.Model) -> None:
        self.model = model
        runtime = model.runtime
        self.messages: list[chat.Turn] = []
        self.system = runtime.get("system", "You are a helpful assistant.")
        self.editor = widgets.Editor()
        self.scroll = widgets.Scrollback()
        self.status = "measuring free VRAM…"
        self.streaming = False
        self.cancel = False
        self.error = ""
        self.stdscr = None
        self.show_help = False
        self.started = 0.0
        self.finished = False
        self.delegate: list[str] | None = None
        self.choice: chat.Choice | None = None
        self.session: chat.Session | None = None

    # -- lifecycle ----------------------------------------------------------

    def boot(self) -> None:
        """Resolve a backend on a thread. No model loads until a turn is sent.

        Measuring a remote card may take up to 30 seconds. The first frame must
        not wait for that probe, and nothing is claimed here: the card is shared
        with image generation, so opening the UI must not take it.
        """
        def work() -> None:
            choice = chat.choose(self.model.id)
            if not choice.usable:
                self.choice = choice
                self.error = choice.reason
                self.status = "unavailable"
                return
            if choice.backend == chat.CPU:
                try:
                    self.delegate = chat.delegate_argv(choice)
                except chat.ChatError as error:
                    self.choice = choice
                    self.error = str(error)
                    self.status = "unavailable"
                    return
                self.choice = choice
                self.status = f"opening bonsai-cpu · {choice.reason}"
                self.finished = True
                return
            if choice.model_id != self.model.id:
                self.model = catalog.find(choice.model_id)
            runtime = self.model.runtime
            self.session = chat.Session(
                choice, context=int(runtime.get("context") or 4096))
            self.choice = choice
            self.status = f"ready · {choice.backend} · {choice.reason}"

        threading.Thread(target=work, daemon=True).start()

    def _progress(self, message: str) -> None:
        self.status = message

    def shutdown(self) -> None:
        self.cancel = True
        if self.session is not None:
            self.session.release()

    @property
    def ready(self) -> bool:
        return (self.choice is not None and self.choice.usable
                and self.session is not None and not self.error)

    # -- conversation -------------------------------------------------------

    def send(self, text: str) -> None:
        if self.choice is None:
            self.status = "still measuring free VRAM…"
            return
        if not text.strip() or self.streaming or not self.ready:
            return
        self.messages.append(chat.Turn("user", text.strip()))
        self.messages.append(chat.Turn("assistant", ""))
        self.streaming = True
        self.cancel = False
        self.started = time.monotonic()
        self.scroll.to_end()
        threading.Thread(target=self._generate, daemon=True).start()

    def _generate(self) -> None:
        """Run one turn, claiming the GPU only for its duration."""
        assert self.session is not None
        try:
            for piece in self.session.stream(
                    self.messages[:-1], system=self.system,
                    should_stop=lambda: self.cancel):
                self.messages[-1].content += piece
        except chat.ChatError as error:
            self.messages[-1].content += f"\n[{error}]"
        finally:
            self.session.release()          # give the card back immediately
            self.streaming = False
            if self.cancel:
                self.messages[-1].content += " […stopped]"

    def stop_generation(self) -> None:
        if self.streaming:
            self.cancel = True

    def reset(self) -> None:
        if not self.streaming:
            self.messages.clear()
            self.status = "ready"

    def save(self) -> str:
        directory = os.path.join(os.path.expanduser("~"), "kilix-bonsai-chats")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.model.id}-{int(time.time())}.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"# {self.model.title}\n\n")
            for message in self.messages:
                handle.write(f"**{message.role}**\n\n{message.content}\n\n")
        return path


# -- rendering --------------------------------------------------------------

HELP = [
    "Enter      send            Esc      stop generating",
    "↑/↓        history         PgUp/Dn  scroll transcript",
    "Ctrl-R     new conversation         Ctrl-S  save to markdown",
    "?          this help                 Ctrl-Q  quit",
]


def transcript_lines(state: State, width: int) -> list[str]:
    lines: list[str] = []
    for message in state.messages:
        who = "you" if message.role == "user" else state.model.title
        lines.append(f"{who}:")
        body = message.content or ("…" if state.streaming else "")
        lines.extend("  " + line for line in widgets.wrap(body, width - 2))
        lines.append("")
    return lines


class PixelRenderer(pixel.Renderer):
    """Streaming chat in the same desktop frame as ``kilix-tui``."""

    area = "CHAT"

    def navigation(self, state: State):
        return (
            ("Chat", None),
            ("New", 18),                  # Ctrl-R
            ("Save", 19),                 # Ctrl-S
            ("Help", ord("?")),
        )

    def active_navigation(self, state: State) -> int:
        return 3 if state.show_help else 0

    def breadcrumb(self, state: State) -> str:
        return f"Chat  /  {state.model.title}"

    def footer(self, state: State) -> str:
        if state.streaming:
            elapsed = time.monotonic() - state.started
            return f"generating {elapsed:.0f}s  ·  Esc stop  ·  Ctrl-Q quit"
        if not state.ready:
            return "waiting for a backend  ·  Ctrl-Q quit"
        return ("Enter send  ·  PgUp/PgDn scroll  ·  Ctrl-R new  ·  "
                "Ctrl-S save  ·  ? help")

    def snapshot(self, state: State):
        choice = state.choice
        return (
            state.model.id, state.status, state.error, state.show_help,
            state.streaming,
            int(time.monotonic() - state.started) if state.streaming else 0,
            state.finished, state.ready,
            None if choice is None else (
                choice.model_id, choice.backend, choice.reason,
                choice.usable),
            tuple((message.role, message.content)
                  for message in state.messages),
            state.editor.text, state.editor.cursor,
            state.scroll.offset, state.scroll.following,
        )

    def body(self, body: pixel.Body, state: State) -> None:
        left, top, right, bottom = body.box
        gap = max(8, int(12 * body.scale))
        if state.error:
            body.card(body.box, selected=True, danger=True)
            pad = max(12, int(18 * body.scale))
            body.text((left + pad, top + pad, right - pad,
                       top + pad + body.font(18, bold=True).height),
                      "BACKEND UNAVAILABLE", size=18, bold=True,
                      color=body.tango.RED_BRIGHT, valign="top")
            body.wrapped((left + pad, top + pad + 34, right - pad,
                          bottom - pad), state.error, size=12)
            return
        if state.show_help:
            self._help(body, HELP)
            return

        input_h = max(42, int(54 * body.scale))
        input_box = (left, bottom - input_h, right, bottom)
        transcript = (left, top, right, input_box[1] - gap)
        font = body.font(12)
        line_h = font.height + max(3, int(5 * body.scale))
        columns = max(8, (right - left - 24) // max(1, 8 * font.scale))
        lines: list[tuple[str, str]] = []
        for message in state.messages:
            role = "YOU" if message.role == "user" \
                else state.model.title.upper()
            lines.append(("role", role))
            value = message.content or ("..." if state.streaming else "")
            lines.extend(("body", line)
                         for line in pixel._wrap(value, columns))
            lines.append(("gap", ""))

        capacity = max(1, (transcript[3] - transcript[1]) // line_h)
        first = state.scroll.clamp(len(lines), capacity)
        if not lines:
            body.text(transcript,
                      "Type a message below. The model loads only when needed.",
                      size=12, color=body.tango.GREY, align="center")
        else:
            y = transcript[1]
            for kind, line in lines[first:first + capacity]:
                if kind == "role":
                    body.text((left + 4, y, right - 4, y + line_h), line,
                              size=10, bold=True,
                              color=body.tango.BLUE_BRIGHT)
                elif kind == "body":
                    body.text((left + 12, y, right - 4, y + line_h), line,
                              size=12, color=body.tango.SILVER)
                y += line_h

        body.card(input_box, selected=state.ready and not state.streaming)
        pad = max(12, int(16 * body.scale))
        input_font = body.font(14)
        columns = max(1, (right - left - pad * 2 - 18)
                      // max(1, 8 * input_font.scale))
        visible, cursor = state.editor.view(columns)
        body.text((left + pad, input_box[1], left + pad + 18, bottom),
                  ">", size=14, bold=True, color=body.tango.BLUE_BRIGHT)
        body.text((left + pad + 18, input_box[1], right - pad, bottom),
                  visible or ("waiting for backend..."
                              if state.choice is None else "message"),
                  size=14, color=(body.tango.WHITE if visible
                                  else body.tango.SILVER))
        if visible and state.ready and not state.streaming:
            cell = 8 * input_font.scale
            x = min(right - pad - 2, left + pad + 18 + cursor * cell)
            body.draw.fill((x, input_box[1] + input_h // 2
                            + input_font.height // 2,
                            x + max(2, input_font.scale), bottom - 8),
                           body.tango.BLUE_BRIGHT)

    @staticmethod
    def _help(body: pixel.Body, rows: list[str]) -> None:
        left, top, right, bottom = body.box
        row_h = max(30, int(42 * body.scale))
        for index, line in enumerate(rows):
            y = top + index * row_h
            if y + row_h > bottom:
                break
            body.card((left, y, right, y + row_h))
            body.text((left + 14, y, right - 14, y + row_h), line,
                      size=12)


def render(surface, state: State) -> None:
    height, width = surface.getmaxyx()
    if state.streaming:
        elapsed = time.monotonic() - state.started
        foot = f"generating… {elapsed:.0f}s · Esc stops"
    elif not state.ready:
        foot = state.status
    else:
        foot = "Enter send · Ctrl-R new · Ctrl-S save · ? help · Ctrl-Q quit"
    page = text.page("CHAT", state.model.title,
                     ("Chat", "New", "Save", "Help"),
                     3 if state.show_help else 0)
    page.render(surface, footer=foot, status=state.status[:38])
    top, left, well, well_width = page.content_box()

    if state.error:
        for index, line in enumerate(widgets.wrap(state.error, well_width - 2)):
            screen.write(surface, top + index, left + 1, line)
        return

    if state.show_help:
        for index, line in enumerate(HELP):
            screen.write(surface, top + index, left + 1, line)
        return

    body_height = max(1, well - 2)
    lines = transcript_lines(state, well_width - 1)
    first = state.scroll.clamp(len(lines), body_height)
    for index, line in enumerate(lines[first:first + body_height]):
        screen.write(surface, top + index, left, line)

    visible, cursor = state.editor.view(max(1, well_width - 3))
    screen.write(surface, top + body_height, left, "> " + visible)

    if state.stdscr is not None and state.ready and not state.streaming:
        try:
            state.stdscr.move(top + body_height,
                              min(width - 1, left + 2 + cursor))
            curses.curs_set(1)
        except curses.error:
            pass


def handle(key: int, state: State) -> bool:
    if state.show_help:
        state.show_help = False
        return True
    if key == 17:                                   # Ctrl-Q
        return False
    if state.error:
        return not screen.is_quit(key)
    if key == 27:                                   # Esc
        state.stop_generation()
        return True
    if key == ord("?") and not state.editor.text:
        state.show_help = True
        return True
    if key == 18:                                   # Ctrl-R
        state.reset()
        return True
    if key == 19:                                   # Ctrl-S
        if state.messages:
            state.status = f"saved {os.path.basename(state.save())}"
        return True
    if key == curses.KEY_PPAGE:
        state.scroll.scroll(5)
        return True
    if key == curses.KEY_NPAGE:
        state.scroll.scroll(-5)
        return True
    if key in widgets.SUBMIT:
        if state.choice is None:
            state.status = "still measuring free VRAM…"
            return True
        state.send(state.editor.submit())
        return True
    state.editor.handle(key)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    chat_models = [m for m in catalog.load()
                   if m.runtime.get("kind") == "chat"]
    positional: list[str] = []
    skip = False
    for argument in argv:
        if skip:
            skip = False
        elif argument == "--screenshot":
            skip = True
        elif not argument.startswith("-"):
            positional.append(argument)
    wanted = positional[0] if positional else None
    try:
        model = catalog.find(wanted) if wanted else None
    except catalog.CatalogError as error:
        print(f"kilix-bonsai-chat: {error}", file=sys.stderr)
        return 2
    if model is None:
        ready = [m for m in chat_models
                 if store.state(m).state == store.PRESENT]
        if not ready:
            print("kilix-bonsai-chat: no chat model is downloaded. "
                  "Run `kilix bonsai` and download one.", file=sys.stderr)
            return 1
        model = ready[0]
    state = State(model)
    if path := screen.screenshot_argv(argv):
        with open(path, "w", encoding="utf-8") as target:
            target.write(screen.render_to_text(render, state) + "\n")
        return 0
    state.boot()
    try:
        status = pixel.run(
            argv, PixelRenderer, state, handle, tick_ms=100,
            command="kilix-bonsai-chat")
        if status is None:
            status = screen.run(render, state, handle=handle, tick_ms=100)
    finally:
        state.shutdown()
    if (state.delegate is not None
            and os.environ.get("KILIX_TUI_HEADLESS") != "1"):
        try:
            os.execvp(state.delegate[0], state.delegate)
        except OSError as error:
            print(f"kilix-bonsai-chat: cannot open bonsai-cpu: {error}",
                  file=sys.stderr)
            return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
