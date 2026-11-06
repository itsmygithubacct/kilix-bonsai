"""kilix-bonsai-chat-gui — a window onto the same chat backends as the TUI.

The curses program stays the terminal fallback. This process has no
controlling terminal: it does not import curses, speak the Kitty graphics
protocol, or hand itself to ``bonsai-cpu chat``. A GPU or remote turn uses
``Session.stream`` and releases the card when the turn ends. A CPU turn
calls bonsai-cpu's resident HTTP server, which streams text without taking
the terminal. Opening the window only chooses a backend. The model loads
on the first send.
"""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "bonsai-cpu", "src"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from kilix_bonsai import catalog, store                   # noqa: E402
from kilix_bonsai.runtime import chat                      # noqa: E402

# Gtk.Application is a process-wide singleton that asks the host session for
# a native window. This program is meant to run on a private display, so it
# builds a Gtk.Window and never registers with that session.
WINDOW_TITLE = "LLM Chat"
EMPTY_HINT = (
    "Type a message below. The model loads only when a message is sent.")


class Resident:
    """The CPU server, published only while this window still owns it.

    ``swap`` blocks for the whole model load. Shutdown can race that load,
    and a multi-gigabyte child must not survive the window that started it.
    """

    def __init__(self) -> None:
        self.server = None
        self.lock = threading.Lock()
        self.closing = False

    def current(self):
        with self.lock:
            return self.server

    def adopt(self, server) -> bool:
        """Keep ``server``, or stop it when the window is already gone."""
        with self.lock:
            if self.closing:
                server.stop()
                return False
            self.server = server
            return True

    def drop_unless(self, model_id: str, keep: bool) -> None:
        """Stop a server that the next choice will not use.

        Called from the probe thread. Stopping a 27B is not instant, and the
        window should keep painting while it happens.
        """
        with self.lock:
            server = self.server
            if server is None:
                return
            if keep and server.model_id == model_id and not self.closing:
                return
            self.server = None
        server.stop()

    def take(self):
        with self.lock:
            self.closing = True
            server, self.server = self.server, None
            return server


def chat_models() -> list[catalog.Model]:
    """Every chat model, using the same kind filter as the curses TUI."""
    return [model for model in catalog.load()
            if model.runtime.get("kind") == "chat"]


def present_chat_models() -> list[catalog.Model]:
    """Chat models whose weights are on disk. Nothing is downloaded."""
    return [model for model in chat_models()
            if store.state(model).state == store.PRESENT]


def iter_cpu_text(model_id: str, wire_messages: list[dict], *,
                  context: int | None, should_stop, on_status,
                  resident: Resident):
    """Yield assistant text from bonsai-cpu's HTTP server.

    ``ChatServer.stream_chat`` speaks HTTP. It does not open a terminal, and
    this function does not call the curses chat command. The server stays
    up between turns; the first send is what loads the weights.
    """
    from bonsai_cpu import convo as cpu_convo
    from bonsai_cpu import models as cpu_models
    from bonsai_cpu import server as cpu_server

    if model_id not in cpu_models.MODELS:
        raise chat.ChatError(
            f"nothing here drives {model_id} without taking the terminal")
    server = resident.current()
    if server is None or server.model_id != model_id:
        previous = server if server is not None and not resident.closing \
            else None
        started = cpu_server.swap(
            previous, model_id, context=context, on_progress=on_status)
        if should_stop is not None and should_stop():
            started.stop()
            return
        if not resident.adopt(started):
            return
        server = started
    params = cpu_convo.default_params()
    spec = cpu_models.MODELS.get(model_id, {})
    # The CPU chat defaults to answering directly. Thinking stays off unless
    # a later control asks for it; sending the flag only when the checkpoint
    # has a thinking switch matches the server's own contract.
    thinking = False if spec.get("thinking") else None
    events = server.stream_chat(
        wire_messages,
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        max_tokens=params["n_predict"],
        enable_thinking=thinking,
        should_stop=should_stop)
    for event in events:
        if event.kind == "reasoning" and event.text:
            if on_status is not None:
                on_status("thinking…")
            continue
        if event.kind == "progress" and on_status is not None:
            on_status("reading the prompt…")
            continue
        if event.kind == "content" and event.text:
            yield event.text


class Controller:
    """Transcript, backend choice, and one in-flight turn. No widgets."""

    def __init__(self, model: catalog.Model | None,
                 models: list[catalog.Model] | None = None, *,
                 schedule=None) -> None:
        self.model = model
        self.models = list(models or ())
        self.schedule = schedule or (lambda fn: fn())
        self.thread_factory = threading.Thread
        self.iter_cpu_text = iter_cpu_text
        self.on_change = lambda: None
        self.resident = Resident()
        self.messages: list[chat.Turn] = []
        self.system = "You are a helpful assistant."
        if model is not None:
            self.system = model.runtime.get("system", self.system)
        self.status = "measuring free VRAM…" if model is not None else ""
        self.error = ""
        self.reason = ""
        self.choice: chat.Choice | None = None
        self.session: chat.Session | None = None
        self.streaming = False
        self.cancel = False
        self.closing = False
        self.started_session = False
        self._boot_generation = 0

    @property
    def heading(self) -> str:
        if self.model is None:
            return "No chat model"
        return self.model.title

    def can_send(self) -> bool:
        return (self.choice is not None and self.choice.usable
                and not self.error and not self.streaming)

    def transcript_text(self) -> str:
        if not self.messages:
            return self.error or EMPTY_HINT
        blocks = []
        last = len(self.messages) - 1
        for index, message in enumerate(self.messages):
            who = "you" if message.role == "user" else self.heading
            body = message.content
            if not body and self.streaming and index == last:
                body = "…"
            blocks.append(f"{who}:\n{body}")
        return "\n\n".join(blocks)

    def append_pieces(self, pieces) -> None:
        """Append streamed pieces onto the open assistant turn."""
        if not self.messages:
            self.messages.append(chat.Turn("assistant", ""))
        for piece in pieces:
            self.messages[-1].content += piece

    def boot(self) -> None:
        """Resolve a backend off the UI thread. Do not load a model."""
        if self.model is None or self.closing:
            return
        self._boot_generation += 1
        generation = self._boot_generation
        model_id = self.model.id

        def work() -> None:
            try:
                choice = chat.choose(model_id)
            except Exception as error:
                choice = chat.Choice(
                    model_id, chat.LOCAL, None, str(error), False)
            self.resident.drop_unless(
                choice.model_id,
                choice.usable and choice.backend == chat.CPU)

            def apply() -> None:
                if generation != self._boot_generation or self.closing:
                    return
                self.apply_choice(choice)

            self.schedule(apply)

        self._spawn(work)

    def apply_choice(self, choice: chat.Choice) -> None:
        """Record a probe result. An unusable choice opens no session."""
        self.choice = choice
        self.reason = choice.reason
        if (choice.model_id and self.model is not None
                and choice.model_id != self.model.id):
            try:
                self.model = catalog.find(choice.model_id)
            except catalog.CatalogError:
                pass
            else:
                self.system = self.model.runtime.get(
                    "system", "You are a helpful assistant.")
        if not choice.usable:
            self.session = None
            self.error = choice.reason
            self.status = choice.reason or "unavailable"
            self.on_change()
            return
        self.error = ""
        if choice.backend == chat.CPU:
            # The CPU chat program is curses. Keep this window, and wait
            # until send() to start the HTTP server.
            self.session = None
            self.status = f"ready · cpu · {choice.reason}"
            self.on_change()
            return
        context = 4096
        if self.model is not None:
            context = int(self.model.runtime.get("context") or 4096)
        self.session = chat.Session(choice, context=context)
        self.status = f"ready · {choice.backend} · {choice.reason}"
        self.on_change()

    def select_model(self, model_id: str) -> None:
        if self.streaming or self.closing or not model_id:
            return
        found = next(
            (item for item in self.models if item.id == model_id), None)
        if found is None or (
                self.model is not None and found.id == self.model.id):
            return
        self.model = found
        self.system = found.runtime.get(
            "system", "You are a helpful assistant.")
        self.choice = None
        self.session = None
        self.error = ""
        self.reason = ""
        self.status = "measuring free VRAM…"
        self.on_change()
        self.boot()

    def send(self, text: str) -> bool:
        if self.choice is None:
            self.status = "still measuring free VRAM…"
            self.on_change()
            return False
        cleaned = text.strip()
        if not cleaned or self.streaming or not self.choice.usable or self.error:
            return False
        self.messages.append(chat.Turn("user", cleaned))
        self.messages.append(chat.Turn("assistant", ""))
        self.streaming = True
        self.cancel = False
        self.status = "generating…"
        self.on_change()
        self._spawn(self._generate)
        return True

    def stop(self) -> None:
        if not self.streaming:
            return
        self.cancel = True
        server = self.resident.current()
        if server is not None:
            try:
                server.abort()
            except Exception:
                pass

    def reset(self) -> None:
        if self.streaming or self.closing:
            return
        self.messages.clear()
        if self.choice is not None and self.choice.usable:
            self.status = (
                f"ready · {self.choice.backend} · {self.choice.reason}")
        self.on_change()

    def shutdown(self) -> None:
        self.closing = True
        self.cancel = True
        if self.session is not None:
            try:
                self.session.release()
            except Exception:
                pass
        server = self.resident.take()
        if server is not None:
            try:
                server.abort()
            except Exception:
                pass
            server.stop()

    def wire_messages(self) -> list[dict]:
        """System prompt plus the turns the server should see.

        The empty assistant placeholder is where the answer will land. It is
        not part of the request.
        """
        messages = []
        system = (self.system or "").strip()
        if system:
            messages.append({"role": "system", "content": system})
        for turn in self.messages[:-1]:
            if turn.content.strip():
                messages.append(
                    {"role": turn.role, "content": turn.content})
        return messages

    def _spawn(self, target) -> None:
        self.thread_factory(target=target, daemon=True).start()

    def _post_status(self, message: str) -> None:
        def apply(message: str = message) -> None:
            if self.closing or self.status == message:
                return
            self.status = message
            self.on_change()

        self.schedule(apply)

    def _append(self, piece: str) -> None:
        if not piece:
            return

        def apply(piece: str = piece) -> None:
            if self.closing:
                return
            self.append_pieces([piece])
            if self.streaming:
                self.status = "generating…"
            self.on_change()

        self.schedule(apply)

    def _stream_gpu(self) -> None:
        if self.session is None:
            raise chat.ChatError("no chat session is open")
        self.started_session = True
        for piece in self.session.stream(
                self.messages[:-1], system=self.system,
                should_stop=lambda: self.cancel or self.closing):
            self._append(piece)

    def _stream_cpu(self) -> None:
        context = None
        if self.model is not None:
            raw = self.model.runtime.get("context")
            context = int(raw) if raw else None
        for piece in self.iter_cpu_text(
                self.choice.model_id, self.wire_messages(),
                context=context,
                should_stop=lambda: self.cancel or self.closing,
                on_status=self._post_status,
                resident=self.resident):
            self._append(piece)

    def _generate(self) -> None:
        error_text = ""
        try:
            if self.choice is None or not self.choice.usable:
                return
            if self.choice.backend == chat.CPU:
                self._stream_cpu()
            else:
                self._stream_gpu()
        except Exception as error:
            error_text = f"\n[{error}]"
        finally:
            if self.session is not None and (
                    self.choice is None
                    or self.choice.backend != chat.CPU):
                try:
                    self.session.release()
                except Exception as error:
                    if not error_text:
                        error_text = f"\n[{error}]"
        stopped = self.cancel

        def finish() -> None:
            if not self.messages:
                self.streaming = False
                return
            if error_text:
                self.messages[-1].content += error_text
            if stopped:
                self.messages[-1].content += " […stopped]"
            self.streaming = False
            if (not self.error and self.choice is not None
                    and self.choice.usable and not self.closing):
                self.status = (
                    f"ready · {self.choice.backend} · {self.choice.reason}")
            if not self.closing:
                self.on_change()

        self.schedule(finish)


def load_gtk():
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GLib, Gtk
    return Gtk, Gdk, GLib


def build_window(controller: Controller, gtk, gdk):
    """Lay out the chat window. ``gtk`` is the real toolkit or a test double."""
    window = gtk.Window()
    window.set_title(WINDOW_TITLE)
    window.set_default_size(760, 560)

    outer = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
    outer.set_border_width(8)
    window.add(outer)

    header = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2)
    outer.pack_start(header, False, False, 0)

    title = gtk.Label()
    title.set_halign(gtk.Align.START)
    title.set_xalign(0)
    header.pack_start(title, False, False, 0)

    combo = None
    guard = {"on": False}
    if len(controller.models) > 1:
        combo = gtk.ComboBoxText()
        for model in controller.models:
            combo.append(model.id, model.title)
        if controller.model is not None:
            guard["on"] = True
            combo.set_active_id(controller.model.id)
            guard["on"] = False

        def on_model(widget) -> None:
            if guard["on"]:
                return
            controller.select_model(widget.get_active_id())

        combo.connect("changed", on_model)
        header.pack_start(combo, False, False, 0)

    status = gtk.Label()
    status.set_halign(gtk.Align.START)
    status.set_xalign(0)
    status.set_line_wrap(True)
    header.pack_start(status, False, False, 0)

    scrolled = gtk.ScrolledWindow()
    scrolled.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
    scrolled.set_vexpand(True)
    transcript = gtk.TextView()
    transcript.set_editable(False)
    transcript.set_wrap_mode(gtk.WrapMode.WORD_CHAR)
    transcript.set_cursor_visible(False)
    transcript.set_left_margin(8)
    transcript.set_right_margin(8)
    transcript.set_top_margin(8)
    transcript.set_bottom_margin(8)
    scrolled.add(transcript)
    outer.pack_start(scrolled, True, True, 0)

    row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
    entry = gtk.Entry()
    entry.set_hexpand(True)
    entry.set_placeholder_text("Message")
    send = gtk.Button(label="Send")
    stop = gtk.Button(label="Stop")
    new = gtk.Button(label="New")
    row.pack_start(entry, True, True, 0)
    row.pack_start(send, False, False, 0)
    row.pack_start(stop, False, False, 0)
    row.pack_start(new, False, False, 0)
    outer.pack_start(row, False, False, 0)

    def refresh(*_args) -> None:
        title.set_text(controller.heading)
        status.set_text(controller.status)
        buffer = transcript.get_buffer()
        buffer.set_text(controller.transcript_text())
        transcript.scroll_to_iter(
            buffer.get_end_iter(), 0.0, True, 0.0, 1.0)
        busy = controller.streaming
        send.set_sensitive(controller.can_send())
        entry.set_sensitive(not busy)
        stop.set_sensitive(busy)
        new.set_sensitive(bool(controller.messages) and not busy)
        if combo is not None and controller.model is not None:
            if combo.get_active_id() != controller.model.id:
                guard["on"] = True
                combo.set_active_id(controller.model.id)
                guard["on"] = False
            combo.set_sensitive(not busy)

    def on_send(*_args) -> None:
        if controller.send(entry.get_text()):
            entry.set_text("")
        refresh()

    def on_key(_widget, event):
        if event.keyval == gdk.KEY_Escape:
            controller.stop()
            refresh()
            return True
        return False

    send.connect("clicked", on_send)
    entry.connect("activate", on_send)
    stop.connect("clicked", lambda *_: (controller.stop(), refresh()))
    new.connect("clicked", lambda *_: (controller.reset(), refresh()))
    window.connect("key-press-event", on_key)
    controller.on_change = refresh
    refresh()
    return window


def resolve_startup(argv: list[str]) -> tuple[
        catalog.Model | None, list[catalog.Model], str]:
    """Pick a chat model without loading it.

    No argument uses the first downloaded chat model. An explicit id is
    accepted even before its weights are present, matching the TUI, so the
    window can say why a later send failed instead of exiting.
    """
    present = present_chat_models()
    wanted = next((arg for arg in argv if not arg.startswith("-")), None)
    if wanted:
        try:
            model = catalog.find(wanted)
        except catalog.CatalogError as error:
            return None, present, str(error)
        if model.runtime.get("kind") != "chat":
            return None, present, f"{model.id} is not a chat model"
        models = list(present)
        if all(item.id != model.id for item in models):
            models.insert(0, model)
        return model, models, ""
    if present:
        return present[0], present, ""
    if chat_models():
        return None, [], (
            "no chat model is downloaded. "
            "Run `kilix bonsai` and download one.")
    return None, [], "no chat model is declared in this checkout"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    model, models, error = resolve_startup(argv)
    try:
        gtk, gdk, glib = load_gtk()
    except Exception as error_gtk:
        print(f"kilix-bonsai-chat-gui: GTK is not available: {error_gtk}",
              file=sys.stderr)
        return 1
    opened = gtk.init_check([])
    if not opened[0]:
        print("kilix-bonsai-chat-gui: could not open a display",
              file=sys.stderr)
        return 1

    controller = Controller(model, models)

    def schedule(fn) -> None:
        def wrapper() -> bool:
            try:
                fn()
            except Exception as failure:
                print(f"kilix-bonsai-chat-gui: {failure}", file=sys.stderr)
            return False

        glib.idle_add(wrapper)

    controller.schedule = schedule
    if error:
        controller.error = error
        controller.reason = error
        controller.status = error
    window = build_window(controller, gtk, gdk)

    def on_destroy(*_args) -> None:
        controller.shutdown()
        gtk.main_quit()

    window.connect("destroy", on_destroy)
    window.show_all()
    if model is not None and not error:
        glib.idle_add(lambda: controller.boot() or False)
    gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
