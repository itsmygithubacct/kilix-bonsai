#!/usr/bin/env python3
"""ChatServer's promises: the request it sends, the events it yields, the
cancel that is instant, and a process group that is actually gone."""

import http.server
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bonsai_cpu import models, server   # noqa: E402


FRAMES = [
    '{"prompt_progress": {"total": 10, "cache": 3, "processed": 10}}',
    '{"choices":[{"delta":{"reasoning_content":"hmm "}}]}',
    '{"choices":[{"delta":{"reasoning_content":"ok"}}]}',
    '{"choices":[{"delta":{"content":"Hello"}}],'
    '"timings":{"predicted_per_second":8.0}}',
    '{"choices":[{"delta":{"content":" world"}}]}',
    '{"timings":{"prompt_n":10,"predicted_n":2,"predicted_per_second":7.9}}',
]


class FakeEndpoint(http.server.BaseHTTPRequestHandler):
    """A scripted /v1/chat/completions that records what it was asked."""

    bodies: list[dict] = []
    hang_after: int | None = None       # frames to send before blocking

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).bodies.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for index, frame in enumerate(FRAMES):
                if self.hang_after is not None and index >= self.hang_after:
                    time.sleep(8)
                self.wfile.write(f"data: {frame}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def endpoint_server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEndpoint)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def client(port):
    return server.ChatServer(model_id="bonsai-8b", model_path="unused",
                             context=1024, threads=1, port=port)


class StreamTest(unittest.TestCase):
    def setUp(self):
        FakeEndpoint.bodies = []
        FakeEndpoint.hang_after = None
        self.httpd = endpoint_server()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def collect(self, **kwargs):
        chat = client(self.httpd.server_address[1])
        return list(chat.stream_chat(
            [{"role": "user", "content": "hi"}], temperature=0.6,
            top_p=0.95, top_k=20, max_tokens=64, **kwargs))

    def test_events_arrive_in_kind_order(self):
        kinds = [e.kind for e in self.collect()]
        self.assertEqual(kinds, ["progress", "reasoning", "reasoning",
                                 "content", "stats", "content", "stats"])

    def test_text_and_stats_carry_through(self):
        events = self.collect()
        text = "".join(e.text for e in events if e.kind == "content")
        self.assertEqual(text, "Hello world")
        final = [e for e in events if e.kind == "stats"][-1]
        self.assertEqual(final.data["predicted_per_second"], 7.9)

    def test_thinking_kwarg_sent_only_when_set(self):
        self.collect(enable_thinking=False)
        self.collect()
        with_kwarg, without = FakeEndpoint.bodies
        self.assertIs(
            with_kwarg["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotIn("chat_template_kwargs", without)

    def test_sampling_round_trips(self):
        self.collect()
        body = FakeEndpoint.bodies[0]
        self.assertEqual(
            (body["temperature"], body["top_p"], body["top_k"],
             body["max_tokens"], body["timings_per_token"],
             body["return_progress"]),
            (0.6, 0.95, 20, 64, True, True))

    def test_should_stop_ends_the_stream(self):
        seen = []

        def stop():
            return any(e.kind == "content" for e in seen)

        chat = client(self.httpd.server_address[1])
        for event in chat.stream_chat([{"role": "user", "content": "hi"}],
                                      temperature=0.6, top_p=0.95, top_k=20,
                                      max_tokens=64, should_stop=stop):
            seen.append(event)
        text = "".join(e.text for e in seen if e.kind == "content")
        self.assertEqual(text, "Hello")

    def test_abort_unblocks_a_stuck_reader(self):
        FakeEndpoint.hang_after = 4          # two reasoning + one content out
        chat = client(self.httpd.server_address[1])
        seen = []

        def read():
            for event in chat.stream_chat([{"role": "user", "content": "hi"}],
                                          temperature=0.6, top_p=0.95,
                                          top_k=20, max_tokens=64):
                seen.append(event)

        reader = threading.Thread(target=read)
        reader.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(e.kind == "content" for e in seen):
                break
            time.sleep(0.02)
        chat.abort()
        reader.join(timeout=5)
        self.assertFalse(reader.is_alive(),
                         "abort() must unblock a blocked SSE read")


FAKE_SERVER = """#!/usr/bin/env python3
import http.server, os, subprocess, sys, time
if os.environ.get("BONSAI_CPU_FAKE_CRASH") == "1":
    sys.stderr.write("boom: refused to load\\n")
    sys.exit(3)
port = int(sys.argv[sys.argv.index("--port") + 1])
child = subprocess.Popen(["sleep", "60"])
with open(os.environ["BONSAI_CPU_FAKE_PIDS"], "w") as fh:
    fh.write(f"{os.getpid()} {child.pid}")
time.sleep(0.4)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        bin_dir = os.path.join(self.tmp.name, "bin")
        os.makedirs(bin_dir)
        fake = os.path.join(bin_dir, "llama-server")
        with open(fake, "w") as fh:
            fh.write(FAKE_SERVER)
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        self.saved_bin = models.RUNTIME_BIN
        models.RUNTIME_BIN = bin_dir
        self.addCleanup(setattr, models, "RUNTIME_BIN", self.saved_bin)
        self.model = os.path.join(self.tmp.name, "model.gguf")
        with open(self.model, "w") as fh:
            fh.write("weights")
        self.pids_path = os.path.join(self.tmp.name, "pids")
        os.environ["BONSAI_CPU_FAKE_PIDS"] = self.pids_path
        self.addCleanup(os.environ.pop, "BONSAI_CPU_FAKE_PIDS", None)
        os.environ.pop("BONSAI_CPU_FAKE_CRASH", None)

    def chat_server(self):
        return server.ChatServer(
            model_id="bonsai-8b", model_path=self.model, context=512,
            threads=1, log_path=os.path.join(self.tmp.name, "log.txt"))

    def test_start_polls_health_and_stop_kills_the_group(self):
        progress = []
        chat = self.chat_server()
        chat.start(on_progress=progress.append)
        self.assertTrue(progress, "loading must report progress")
        with open(self.pids_path) as fh:
            parent, child = (int(p) for p in fh.read().split())
        self.assertTrue(alive(parent) and alive(child))
        chat.stop()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and (alive(parent) or alive(child)):
            time.sleep(0.05)
        self.assertFalse(alive(parent), "server must be dead after stop()")
        self.assertFalse(alive(child),
                         "the server's own children must die with it")

    def test_a_crash_surfaces_the_log_tail(self):
        os.environ["BONSAI_CPU_FAKE_CRASH"] = "1"
        chat = self.chat_server()
        with self.assertRaises(server.ServerError) as caught:
            chat.start()
        message = str(caught.exception)
        self.assertIn("boom", message)
        self.assertIn(chat.log_path, message)


if __name__ == "__main__":
    unittest.main()
