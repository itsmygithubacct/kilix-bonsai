"""The chat interface: streaming, switching, resuming.

One curses loop, one worker thread per turn, one server owned for as long
as a conversation is open. The loop repaints ten times a second whether or
not a key arrives, which is all the synchronisation the worker needs: it
appends to the transcript and the next frame shows it.

Nothing here blocks the screen. The server starts on a thread with its
progress narrated; the probe-then-freeze failure mode of tools that measure
the world in their constructors is avoided by never measuring anything
before the first frame.
"""
from __future__ import annotations

import curses
import os
import signal
import threading

from . import convo as convo_mod
from . import layout
from . import models
from . import overlays
from . import pixel
from . import richtext
from . import screen
from . import server as server_mod
from . import widgets

TICK_MS = 100

CTRL_F, CTRL_N, CTRL_O, CTRL_P, CTRL_Q, CTRL_R, CTRL_T = 6, 14, 15, 16, 17, 18, 20


class State:
    """Everything the render and the workers share."""

    def __init__(self, model_id: str, think: bool = False,
                 threads: int | None = None, ctx: int | None = None):
        self.view = "list"
        self.closing = False
        self.loading = ""
        self.error = ""
        self.status = ""
        self.streaming = False
        self.cancel = False
        self.finished = False
        self.show_thinking = False
        self.initial_model = model_id
        self.initial_think = think
        self.threads = threads
        self.ctx = ctx
        self.server: server_mod.ChatServer | None = None
        self.convo: convo_mod.Conversation | None = None
        self.conversations: list[convo_mod.Conversation] = []
        self.selected = 0
        self.editor = widgets.Editor()
        self.scroll = widgets.Scrollback()
        self.renaming: widgets.Editor | None = None
        self.pending_delete: int | None = None
        self.param_index = 0
        self.param_editor: widgets.Editor | None = None
        self.stats: dict = {}
        self.progress: dict | None = None

    # -- derived ------------------------------------------------------------

    def spec(self) -> dict:
        model_id = self.convo.model_id if self.convo else self.initial_model
        return models.MODELS[model_id]

    def context_limit(self) -> int:
        return self.ctx or self.spec()["ctx"]

    def busy(self) -> bool:
        return bool(self.loading) or self.streaming

    def refresh_listing(self) -> None:
        self.conversations = convo_mod.listing()
        self.selected = min(self.selected, len(self.conversations))


def _ensure_server(state: State, model_id: str) -> None:
    """Make the resident server match the model, on a thread."""
    if os.environ.get("KILIX_TUI_HEADLESS") == "1":
        # A one-frame smoke run must not spawn a multi-gigabyte child it
        # will never get the chance to stop.
        return
    current = state.server
    if current is not None and current.model_id == model_id:
        return
    title = models.MODELS[model_id]["title"]
    state.loading = f"starting {title}…"

    def work():
        try:
            started = server_mod.swap(
                state.server, model_id, context=state.ctx,
                threads=state.threads,
                on_progress=lambda message: setattr(state, "loading", message))
            if state.closing:
                # The UI quit while the model was still loading; the server
                # belongs to nobody now and must not outlive this thread.
                started.stop()
            else:
                state.server = started
        except server_mod.ServerError as error:
            state.server = None
            state.error = str(error)
        finally:
            state.loading = ""

    threading.Thread(target=work, daemon=True).start()


def open_conversation(state: State, conversation) -> None:
    state.convo = conversation
    state.view = "chat"
    state.editor = widgets.Editor()
    state.scroll.to_end()
    state.stats = {}
    state.progress = None
    _ensure_server(state, conversation.model_id)


def new_conversation(state: State, model_id: str | None = None) -> None:
    built = convo_mod.new(model_id or state.initial_model)
    spec = models.MODELS[built.model_id]
    built.think = state.initial_think and bool(spec.get("thinking"))
    open_conversation(state, built)


def _wire_messages(state: State) -> list[dict]:
    """What the server sees: system prompt plus role/content only —
    reasoning and stats are ours, not context. The trailing assistant
    placeholder send() just appended is the slot for the answer, not part
    of the question, so it stays off the wire."""
    messages = [{"role": "system", "content": state.convo.params["system"]}]
    for message in state.convo.messages[:-1]:
        messages.append({"role": message["role"],
                         "content": message["content"]})
    return messages


def _generate(state: State) -> None:
    assistant = state.convo.messages[-1]
    spec = state.spec()
    params = state.convo.params
    thinking = state.convo.think if spec.get("thinking") else None
    try:
        events = state.server.stream_chat(
            _wire_messages(state),
            temperature=params["temperature"], top_p=params["top_p"],
            top_k=params["top_k"], max_tokens=params["n_predict"],
            enable_thinking=thinking,
            should_stop=lambda: state.cancel)
        for event in events:
            if event.kind == "reasoning":
                assistant["reasoning"] += event.text
                assistant["reasoning_tokens"] = (
                    assistant.get("reasoning_tokens", 0) + 1)
            elif event.kind == "content":
                assistant["content"] += event.text
            elif event.kind == "progress":
                state.progress = event.data
            elif event.kind == "stats":
                state.stats = event.data
                state.progress = None
    except server_mod.ServerError as error:
        state.error = str(error)
    finally:
        state.streaming = False
        state.progress = None
        if state.stats:
            assistant["stats"] = {
                key: state.stats[key] for key in
                ("prompt_n", "predicted_n", "predicted_per_second")
                if key in state.stats}
        if not state.convo.name:
            first = next((m["content"] for m in state.convo.messages
                          if m["role"] == "user"), "")
            state.convo.name = convo_mod.default_name(first)
        try:
            convo_mod.save(state.convo)
        except convo_mod.ConvoError as error:
            state.status = str(error)


def send(state: State, text: str) -> None:
    text = text.strip()
    if not text or state.busy() or state.server is None:
        return
    # The user message and the empty assistant reply land together, so the
    # very next frame shows both the question and where the answer will go.
    state.convo.messages.append({"role": "user", "content": text})
    state.convo.messages.append(
        {"role": "assistant", "content": "", "reasoning": ""})
    state.streaming = True
    state.cancel = False
    state.scroll.to_end()
    threading.Thread(target=_generate, args=(state,), daemon=True).start()


def stop_generation(state: State) -> None:
    state.cancel = True
    if state.server is not None:
        state.server.abort()


# -- rendering ---------------------------------------------------------------

BOLD = curses.A_BOLD
DIM = curses.A_DIM
REVERSE = curses.A_REVERSE


def transcript_lines(state: State, width: int) -> list:
    lines: list = []
    total = len(state.convo.messages)
    for index, message in enumerate(state.convo.messages):
        if message["role"] == "user":
            lines.append([(BOLD, "you")])
            lines.extend([[(0, text)] for text in
                          widgets.wrap(message["content"], width)])
        else:
            lines.append([(BOLD, state.spec()["title"].lower())])
            reasoning = message.get("reasoning") or ""
            if reasoning:
                live = state.streaming and index == total - 1
                tokens = message.get("reasoning_tokens",
                                     len(reasoning.split()))
                lines.append([(DIM, richtext.fold_line(tokens, live))])
                if state.show_thinking:
                    lines.extend([[(DIM, text)] for text in
                                  widgets.wrap(reasoning, width)])
            if message["content"]:
                lines.extend(richtext.style(message["content"], width))
            elif state.streaming and index == total - 1 and not reasoning:
                lines.append([(DIM, "…")])
        lines.append([])
    return lines


def status_line(state: State) -> str:
    if state.progress:
        done = state.progress.get("processed", 0)
        total = state.progress.get("total", 0)
        cached = state.progress.get("cache", 0)
        return f"processing prompt… {done}/{total} (cached {cached})"
    stats = state.stats or {}
    if not stats and state.convo and state.convo.messages:
        for message in reversed(state.convo.messages):
            if message.get("stats"):
                stats = message["stats"]
                break
    if not stats:
        return ""
    rate = stats.get("predicted_per_second")
    used = stats.get("prompt_n", 0) + stats.get("predicted_n", 0)
    limit = state.context_limit()
    filled = min(10, round(10 * used / limit)) if limit else 0
    meter = "#" * filled + "-" * (10 - filled)
    rate_text = f"{rate:.1f} tok/s · " if rate else ""
    return f"{rate_text}ctx [{meter}] {used}/{limit}"


def render_chat(surface, state: State) -> None:
    spec = state.spec()
    think = ""
    if spec.get("thinking"):
        think = " · think " + ("ON" if state.convo.think else "off")
    name = state.convo.name or "new conversation"
    if state.loading:
        footer = state.loading
    elif state.streaming:
        footer = "generating… Esc stops · Ctrl-Q quit"
    else:
        footer = ("Enter send · ^O model · ^T think · ^P settings · "
                  "^F fold · ? help")
    status = state.status or status_line(state)
    pane_top, pane_bottom, width = layout.shell(
        surface, breadcrumb=f"Conversation / {spec['title']}{think} / {name}",
        active=1, status=status, footer=footer)
    pane_bottom -= 1
    pane_height = max(1, pane_bottom - pane_top + 1)
    lines = transcript_lines(state, max(1, width - 1))
    first = state.scroll.clamp(len(lines), pane_height)
    for row, line in enumerate(lines[first:first + pane_height]):
        x = 0
        for attr, text in line:
            screen.write(surface, pane_top + row, x, text, attr)
            x += len(text)

    if state.renaming is not None:
        screen.write(surface, pane_bottom, 1,
                     "name: " + state.renaming.view(width - 8)[0])
    else:
        visible, cursor = state.editor.view(max(1, width - 3))
        screen.write(surface, pane_bottom, 1, "> " + visible)
        if cursor < len(visible):
            screen.write(surface, pane_bottom, 3 + cursor, visible[cursor],
                         REVERSE)
        else:
            screen.write(surface, pane_bottom, 3 + cursor, " ", REVERSE)


def render(surface, state: State) -> None:
    if state.error:
        # The error owns the frame: painting it over a live view would leave
        # the two interleaved, and a broken server deserves the whole screen.
        overlays.render_error(surface, state)
        return
    if state.view == "list":
        overlays.render_list(surface, state)
    elif state.view == "picker":
        overlays.render_picker(surface, state)
    elif state.view == "params":
        overlays.render_params(surface, state)
    elif state.view == "help":
        overlays.render_help(surface, state)
    else:
        render_chat(surface, state)
    if state.loading:
        overlays.render_loading(surface, state)


# -- keys --------------------------------------------------------------------

def _handle_list(key: int, state: State) -> bool:
    rows = len(state.conversations) + 1          # row 0 is [ new ]
    if state.pending_delete is not None:
        target = state.pending_delete
        state.pending_delete = None
        if key in screen.YES and 0 <= target < len(state.conversations):
            victim = state.conversations[target]
            try:
                os.remove(victim.path)
            except OSError as error:
                state.status = f"could not delete: {error}"
            state.refresh_listing()
        else:
            state.status = ""
        return True
    move = screen.direction(key)
    if move:
        state.selected = max(0, min(rows - 1, state.selected + move))
        return True
    if key in screen.SELECT:
        if state.selected == 0:
            new_conversation(state)
        else:
            open_conversation(state,
                              state.conversations[state.selected - 1])
        return True
    if key == ord("n"):
        new_conversation(state)
        return True
    if key == ord("d") and state.selected > 0:
        victim = state.conversations[state.selected - 1]
        state.pending_delete = state.selected - 1
        state.status = f"delete '{victim.name or 'untitled'}'? y confirms"
        return True
    if screen.is_quit(key):
        return False
    return True


def _handle_picker(key: int, state: State) -> bool:
    ids = sorted(models.MODELS)
    move = screen.direction(key)
    if move:
        state.selected = max(0, min(len(ids) - 1, state.selected + move))
        return True
    if key in screen.SELECT:
        chosen = ids[state.selected]
        state.view = "chat"
        if chosen != state.convo.model_id:
            state.convo.model_id = chosen
            if not models.MODELS[chosen].get("thinking"):
                state.convo.think = False
            convo_mod.save(state.convo)
            _ensure_server(state, chosen)
        return True
    if key == 27:
        state.view = "chat"
        return True
    return True


PARAM_FIELDS = ("system", "temperature", "top_p", "top_k", "n_predict")


def _commit_param(state: State, text: str) -> None:
    field = PARAM_FIELDS[state.param_index]
    old = state.convo.params[field]
    if field == "system":
        state.convo.params[field] = text
    else:
        try:
            value = float(text) if field in ("temperature", "top_p") \
                else int(text)
        except ValueError:
            state.status = f"{field}: '{text}' is not a number, keeping {old}"
            return
        state.convo.params[field] = value
    convo_mod.save(state.convo)


def _handle_params(key: int, state: State) -> bool:
    if state.param_editor is not None:
        if key in widgets.SUBMIT:
            _commit_param(state, state.param_editor.text)
            state.param_editor = None
        elif key == 27:
            state.param_editor = None
        else:
            state.param_editor.handle(key)
        return True
    move = screen.direction(key)
    if move:
        state.param_index = max(0, min(len(PARAM_FIELDS) - 1,
                                       state.param_index + move))
        return True
    if key in screen.SELECT:
        editor = widgets.Editor()
        editor.set(str(state.convo.params[PARAM_FIELDS[state.param_index]]))
        state.param_editor = editor
        return True
    if key == 27:
        state.view = "chat"
        return True
    return True


def _handle_chat(key: int, state: State) -> bool:
    if state.renaming is not None:
        if key in widgets.SUBMIT:
            name = state.renaming.text.strip()
            if name:
                state.convo.name = name
                convo_mod.save(state.convo)
            state.renaming = None
        elif key == 27:
            state.renaming = None
        else:
            state.renaming.handle(key)
        return True
    if key == 27:
        if state.streaming:
            stop_generation(state)
        else:
            state.refresh_listing()
            state.view = "list"
        return True
    if key == CTRL_O and not state.busy():
        state.selected = sorted(models.MODELS).index(state.convo.model_id)
        state.view = "picker"
        return True
    if key == CTRL_T:
        if state.spec().get("thinking"):
            state.convo.think = not state.convo.think
            convo_mod.save(state.convo)
        else:
            state.status = f"{state.spec()['title']} is not a thinking model"
        return True
    if key == CTRL_P and not state.streaming:
        state.param_index = 0
        state.view = "params"
        return True
    if key == CTRL_F:
        state.show_thinking = not state.show_thinking
        return True
    if key == CTRL_N and not state.busy():
        new_conversation(state, state.convo.model_id)
        return True
    if key == CTRL_R:
        editor = widgets.Editor()
        editor.set(state.convo.name)
        state.renaming = editor
        return True
    if key == curses.KEY_PPAGE:
        state.scroll.scroll(5)
        return True
    if key == curses.KEY_NPAGE:
        state.scroll.scroll(-5)
        return True
    if key == ord("?") and not state.editor.text:
        state.view = "help"
        return True
    if key in widgets.SUBMIT:
        if state.busy():
            # Typing ahead while the model loads is welcome; sending has to
            # wait, and silence would read as a swallowed message.
            state.status = state.loading or "still generating — Esc stops it"
        else:
            send(state, state.editor.submit())
        return True
    state.editor.handle(key)
    return True


def handle(key: int, state: State) -> bool:
    state.status = "" if key != -1 else state.status
    if key == CTRL_Q:
        return False
    if state.error:
        if screen.is_quit(key):
            return False
        if key in (ord("r"), ord("R")):
            state.error = ""
            if state.convo is not None and state.server is None:
                _ensure_server(state, state.convo.model_id)
        else:
            state.error = ""
        return True
    if state.loading and state.view != "chat":
        return True                       # nothing to do but wait or Ctrl-Q
    if state.view == "help":
        state.view = "chat" if state.convo else "list"
        return True
    if state.view == "list":
        return _handle_list(key, state)
    if state.view == "picker":
        return _handle_picker(key, state)
    if state.view == "params":
        return _handle_params(key, state)
    return _handle_chat(key, state)


def main(model_id: str = "bonsai-8b", think: bool = False,
         threads: int | None = None, ctx: int | None = None,
         graphics: bool | None = None) -> int:
    state = State(model_id, think=think, threads=threads, ctx=ctx)
    state.refresh_listing()
    if not state.conversations:
        # An empty store means the list screen would be one row; go straight
        # to a new conversation, which is why the person is here.
        new_conversation(state)
    previous_signals = {}

    def close_session(_signal, _frame) -> None:
        # Closing a Kilix window sends SIGHUP/SIGTERM.  Let the UI loop
        # unwind so its resident llama-server is stopped in the normal
        # finally block instead of becoming an orphan.
        state.finished = True

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        previous_signals[signum] = signal.getsignal(signum)
        signal.signal(signum, close_session)
    try:
        status = pixel.run(state, handle, mode=graphics, tick_ms=TICK_MS)
        return (screen.run(render, state, handle=handle, tick_ms=TICK_MS)
                if status is None else status)
    finally:
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
        state.closing = True
        if state.streaming:
            stop_generation(state)
        if state.server is not None:
            state.server.stop()
        if state.convo is not None and state.convo.messages:
            try:
                convo_mod.save(state.convo)
            except convo_mod.ConvoError:
                pass
