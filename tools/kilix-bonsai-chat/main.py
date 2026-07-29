"""kilix-bonsai-chat — talk to a local 1-bit model.

The generation runs on a worker thread and the UI keeps redrawing while tokens
arrive, which is the whole reason this is worth using over `llama-cli`: the
answer appears as it is written, Esc abandons it mid-sentence, and the model
stays loaded between turns.

Everything the model server needs is derived from the model's own MODEL.json —
which file to load, how much context, what system prompt to start with — so a
new chat model is a new folder, not a change here.
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

from kilix_bonsai import catalog, screen, store, widgets   # noqa: E402
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
        self.status = "starting the model server…"
        self.streaming = False
        self.cancel = False
        self.error = ""
        self.stdscr = None
        self.temperature = 0.7
        self.top_p = 0.9
        self.show_help = False
        self.started = 0.0
        self.tokens = 0
        # The GPU is picked from what is measured free right now, not from a
        # default, and the choice explains itself so the header can say which
        # card and why this model rather than the other.
        self.choice = chat.choose(model.id)
        self.session = chat.Session(self.choice,
                                    context=int(runtime.get("context") or 4096))

    # -- lifecycle ----------------------------------------------------------

    def boot(self) -> None:
        """Resolve a backend. No model is loaded until a turn is sent.

        Deliberately nothing is claimed here: the card is shared with image
        generation, so opening the UI must not take it.
        """
        if not self.choice.usable:
            self.error = self.choice.reason
            self.status = "unavailable"
            return
        if self.choice.model_id != self.model.id:
            self.model = catalog.find(self.choice.model_id)
        self.status = f"ready · {self.choice.backend} · {self.choice.reason}"

    def _progress(self, message: str) -> None:
        self.status = message

    def shutdown(self) -> None:
        self.cancel = True
        self.session.release()

    @property
    def ready(self) -> bool:
        return self.choice.usable and not self.error

    # -- conversation -------------------------------------------------------

    def send(self, text: str) -> None:
        if not text.strip() or self.streaming or not self.ready:
            return
        self.messages.append(chat.Turn("user", text.strip()))
        self.messages.append(chat.Turn("assistant", ""))
        self.streaming = True
        self.cancel = False
        self.tokens = 0
        self.started = time.monotonic()
        self.scroll.to_end()
        threading.Thread(target=self._generate, daemon=True).start()

    def _generate(self) -> None:
        """Run one turn, claiming the GPU only for its duration."""
        try:
            for piece in self.session.stream(
                    self.messages[:-1], system=self.system,
                    should_stop=lambda: self.cancel):
                self.messages[-1].content += piece
                self.tokens += 1
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
    "Ctrl-T     temperature     ?        this help      Ctrl-Q  quit",
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


def render(surface, state: State) -> None:
    height, width = surface.getmaxyx()
    right = f"{state.status}"
    screen.write(surface, 0, 0, f"{TITLE} — {state.model.title}")
    screen.write(surface, 0, max(0, width - len(right) - 1), right)
    screen.write(surface, 1, 0, "─" * max(0, width - 1))

    if state.error:
        for index, line in enumerate(widgets.wrap(state.error, width - 2)):
            screen.write(surface, 3 + index, 1, line)
        screen.write(surface, height - 1, 0, "q quit")
        return

    if state.show_help:
        for index, line in enumerate(HELP):
            screen.write(surface, 3 + index, 2, line)
        screen.write(surface, height - 1, 0, "any key returns")
        return

    body_height = max(1, height - 5)
    lines = transcript_lines(state, width - 1)
    first = state.scroll.clamp(len(lines), body_height)
    for index, line in enumerate(lines[first:first + body_height]):
        screen.write(surface, 2 + index, 0, line)

    screen.write(surface, height - 3, 0, "─" * max(0, width - 1))
    visible, cursor = state.editor.view(max(1, width - 3))
    screen.write(surface, height - 2, 0, "> " + visible)

    if state.streaming:
        elapsed = time.monotonic() - state.started
        rate = state.tokens / elapsed if elapsed > 0 else 0.0
        footer = (f"generating… {state.tokens} chunks, {rate:.0f}/s · "
                  "Esc stops")
    elif not state.ready:
        footer = state.status
    else:
        footer = (f"Enter send · Ctrl-R new · Ctrl-S save · ? help · "
                  f"Ctrl-Q quit · temp {state.temperature:.1f}")
    screen.write(surface, height - 1, 0, footer)
    if state.stdscr is not None and state.ready and not state.streaming:
        try:
            state.stdscr.move(height - 2, min(width - 1, 2 + cursor))
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
    if key == 20:                                   # Ctrl-T
        state.temperature = round(
            0.1 if state.temperature >= 1.2 else state.temperature + 0.2, 1)
        return True
    if key == curses.KEY_PPAGE:
        state.scroll.scroll(5)
        return True
    if key == curses.KEY_NPAGE:
        state.scroll.scroll(-5)
        return True
    if key in widgets.SUBMIT:
        state.send(state.editor.submit())
        return True
    state.editor.handle(key)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    chat_models = [m for m in catalog.load()
                   if m.runtime.get("kind") == "chat"]
    wanted = next((a for a in argv if not a.startswith("-")), None)
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
        return screen.run(render, state, handle=handle, tick_ms=100)
    finally:
        state.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
