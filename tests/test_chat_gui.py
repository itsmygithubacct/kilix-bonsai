"""The desktop chat window streams tokens without a terminal."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import catalog  # noqa: E402
from kilix_bonsai.runtime import chat  # noqa: E402


def load_gui():
    path = os.path.join(ROOT, "tools", "kilix-bonsai-chat-gui", "main.py")
    spec = importlib.util.spec_from_file_location(
        "kilix_bonsai_chat_gui", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUI = load_gui()


class InlineThread:
    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class FakeBuffer:
    def __init__(self) -> None:
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text

    def get_end_iter(self):
        return object()


class FakeWidget:
    def __init__(self, kind: str, **kwargs) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.children = []
        self.handlers = {}
        self.label = kwargs.get("label", "")
        self.text = ""
        self.title = ""
        self.sensitive = True
        self.ids = []
        self.active = None
        self.buffer = FakeBuffer()

    def connect(self, signal, callback):
        self.handlers.setdefault(signal, []).append(callback)
        return callback

    def pack_start(self, child, expand, fill, padding) -> None:
        self.children.append(child)

    def add(self, child) -> None:
        self.children.append(child)

    def set_title(self, title: str) -> None:
        self.title = title

    def get_title(self) -> str:
        return self.title

    def set_text(self, text: str) -> None:
        self.text = text

    def get_text(self) -> str:
        return self.text

    def get_buffer(self):
        return self.buffer

    def append(self, id_, text) -> None:
        self.ids.append((id_, text))

    def set_active_id(self, id_) -> None:
        self.active = id_

    def get_active_id(self):
        return self.active

    def set_sensitive(self, value: bool) -> None:
        self.sensitive = bool(value)

    def scroll_to_iter(self, *_args) -> None:
        return None

    def __getattr__(self, name):
        if name.startswith("set_") or name.startswith("get_"):
            def method(*_args, **_kwargs):
                return None
            return method
        raise AttributeError(name)


class _Enum:
    def __init__(self, **names) -> None:
        for key, value in names.items():
            setattr(self, key, value)


class FakeGtk:
    Orientation = _Enum(VERTICAL="vertical", HORIZONTAL="horizontal")
    Align = _Enum(START="start")
    PolicyType = _Enum(AUTOMATIC="automatic")
    WrapMode = _Enum(WORD_CHAR="word-char")

    def Window(self):
        return FakeWidget("window")

    def Box(self, orientation=None, spacing=0):
        return FakeWidget("box", orientation=orientation, spacing=spacing)

    def Label(self, label=""):
        return FakeWidget("label", label=label)

    def ScrolledWindow(self):
        return FakeWidget("scrolled")

    def TextView(self):
        return FakeWidget("text")

    def Entry(self):
        return FakeWidget("entry")

    def Button(self, label=""):
        return FakeWidget("button", label=label)

    def ComboBoxText(self):
        return FakeWidget("combo")


def walk(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from walk(child)


def gtk_display_available() -> bool:
    try:
        gtk, _gdk, _glib = GUI.load_gtk()
    except Exception:
        return False
    try:
        opened = gtk.init_check([])
    except Exception:
        return False
    return bool(opened[0])


class StreamHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = catalog.find("bonsai-8b")
        self.controller = GUI.Controller(
            self.model, [self.model], schedule=lambda fn: fn())
        self.controller.thread_factory = InlineThread

    def test_append_pieces_updates_the_transcript(self) -> None:
        self.controller.messages.append(chat.Turn("assistant", ""))
        self.controller.append_pieces(["Hello ", "there"])
        text = self.controller.transcript_text()
        self.assertIn("Hello there", text)
        self.assertIn(self.model.title, text)

    def test_fake_session_stream_updates_the_transcript_and_releases(self) -> None:
        class FakeSession:
            def __init__(self) -> None:
                self.released = False
                self.process = None

            def stream(self, history, system="", tokens=512,
                       should_stop=None):
                self.history = history
                self.system = system
                yield "Hello "
                yield "there"

            def release(self) -> None:
                self.released = True

        session = FakeSession()
        self.controller.choice = chat.Choice(
            "bonsai-8b", chat.LOCAL, 9000, "9000 MiB free", True)
        self.controller.session = session
        self.assertTrue(self.controller.send("hello"))
        self.assertIn("Hello there", self.controller.transcript_text())
        self.assertIn("you:\nhello", self.controller.transcript_text())
        self.assertTrue(session.released)
        self.assertTrue(self.controller.started_session)
        self.assertFalse(self.controller.streaming)

    def test_unusable_choice_records_its_reason_and_skips_the_session(self) -> None:
        reason = ("nothing here drives bitnet-b1.58-2b4t — its engine is "
                  "bitnet.cpp, which no interface in this repository speaks yet")
        self.controller.apply_choice(chat.Choice(
            "bitnet-b1.58-2b4t", chat.LOCAL, None, reason, False))
        self.assertEqual(self.controller.reason, reason)
        self.assertEqual(self.controller.error, reason)
        self.assertIsNone(self.controller.session)
        self.assertIn(reason, self.controller.transcript_text())
        self.assertFalse(self.controller.send("hello"))
        self.assertEqual(self.controller.messages, [])
        self.assertFalse(self.controller.started_session)

    def test_boot_of_an_unusable_choice_does_not_start_a_session(self) -> None:
        reason = "refused before any session"
        choice = chat.Choice(
            "bitnet-b1.58-2b4t", chat.LOCAL, None, reason, False)
        saved = GUI.chat.choose
        GUI.chat.choose = lambda *_args, **_kwargs: choice
        try:
            self.controller.boot()
        finally:
            GUI.chat.choose = saved
        self.assertEqual(self.controller.reason, reason)
        self.assertIsNone(self.controller.session)
        self.assertFalse(self.controller.started_session)

    def test_cpu_choice_stays_in_process_until_send(self) -> None:
        calls = []

        def fake_iter(model_id, messages, **kwargs):
            calls.append((model_id, messages))
            yield "cpu "
            yield "text"

        self.controller.iter_cpu_text = fake_iter
        self.controller.apply_choice(chat.Choice(
            "bonsai-8b", chat.CPU, None,
            "no GPU available — running on CPU", True))
        self.assertIsNone(self.controller.session)
        self.assertEqual(calls, [])
        self.assertIn("cpu", self.controller.status)
        self.assertTrue(self.controller.send("hello"))
        self.assertEqual(calls[0][0], "bonsai-8b")
        self.assertIn("cpu text", self.controller.transcript_text())
        self.assertIsNone(self.controller.session)
        self.assertFalse(self.controller.started_session)
        roles = [item["role"] for item in calls[0][1]]
        self.assertEqual(roles[0], "system")
        self.assertIn("user", roles)
        self.assertNotIn("assistant", roles)

    def test_construction_does_not_choose_or_load(self) -> None:
        source = inspect.getsource(GUI.Controller.__init__)
        self.assertNotIn("chat.choose", source)
        self.assertNotIn("iter_cpu_text(", source)
        self.assertIsNone(self.controller.choice)
        self.assertIsNone(self.controller.session)
        tree = ast.parse(inspect.getsource(GUI))
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    called.add(func.attr)
                elif isinstance(func, ast.Name):
                    called.add(func.id)
        self.assertNotIn("curses", imported)
        self.assertNotIn("delegate_argv", called)
        self.assertFalse(called & {"execvp", "execv", "execl", "execve"})
        self.assertNotIn("Application", called)

    def test_chat_models_use_the_tui_kind_filter(self) -> None:
        models = GUI.chat_models()
        self.assertIn("bonsai-8b", {model.id for model in models})
        self.assertNotIn("bonsai-image-4b", {model.id for model in models})
        self.assertTrue(all(
            model.runtime.get("kind") == "chat" for model in models))


class WindowTest(unittest.TestCase):
    def test_stubbed_window_has_the_desktop_controls(self) -> None:
        model = catalog.find("bonsai-8b")
        other = catalog.find("bonsai-27b")
        controller = GUI.Controller(
            model, [model, other], schedule=lambda fn: fn())
        controller.thread_factory = InlineThread

        class FakeSession:
            def __init__(self) -> None:
                self.released = False
                self.process = None

            def stream(self, history, system="", tokens=512,
                       should_stop=None):
                yield "Hello "
                yield "there"

            def release(self) -> None:
                self.released = True

        session = FakeSession()
        controller.choice = chat.Choice(
            "bonsai-8b", chat.LOCAL, 9000, "ready", True)
        controller.session = session
        window = GUI.build_window(controller, FakeGtk(), SimpleNamespace(
            KEY_Escape=65307))
        widgets = list(walk(window))
        labels = {
            widget.label for widget in widgets
            if widget.kind == "button"}
        self.assertEqual(window.get_title(), "LLM Chat")
        self.assertEqual(labels, {"Send", "Stop", "New"})
        entry = next(widget for widget in widgets if widget.kind == "entry")
        transcript = next(
            widget for widget in widgets if widget.kind == "text")
        send = next(
            widget for widget in widgets
            if widget.kind == "button" and widget.label == "Send")
        entry.set_text("hello")
        send.handlers["clicked"][0](send)
        self.assertIn("Hello there", transcript.get_buffer().text)
        self.assertEqual(entry.get_text(), "")
        self.assertTrue(session.released)

        controller.streaming = True
        event = SimpleNamespace(keyval=65307)
        handled = window.handlers["key-press-event"][0](window, event)
        self.assertTrue(handled)
        self.assertTrue(controller.cancel)
        ignored = window.handlers["key-press-event"][0](
            window, SimpleNamespace(keyval=ord("a")))
        self.assertFalse(ignored)
        combo = next(widget for widget in widgets if widget.kind == "combo")
        self.assertEqual(
            [item[0] for item in combo.ids],
            ["bonsai-8b", "bonsai-27b"])

    @unittest.skipUnless(
        gtk_display_available(), "Gtk cannot open a display")
    def test_real_window_title_without_entering_the_main_loop(self) -> None:
        model = catalog.find("bonsai-8b")
        controller = GUI.Controller(
            model, [model], schedule=lambda fn: fn())
        gtk, gdk, _glib = GUI.load_gtk()
        window = GUI.build_window(controller, gtk, gdk)
        self.assertEqual(window.get_title(), "LLM Chat")
        window.destroy()


if __name__ == "__main__":
    unittest.main()
