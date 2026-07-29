"""A managed llama-server retained as a runtime probe and reference client.

The current chat tool does not use this path. GPU chat needs the vendor
launchers, while CPU chat delegates to ``bonsai-cpu``, which owns its pinned
upstream server and richer interface. This module remains useful to callers
that explicitly want the repository's separately built server: it binds to
loopback, health-polls startup, streams completions, and kills its child group
on exit.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .. import paths

# A 3.8 GB checkpoint is slow to map on a cold page cache, and the failure a
# user actually hits is "I gave up before it finished", so the ceiling is
# generous and the wait reports progress rather than blocking silently.
LOAD_TIMEOUT = 600.0
HEALTH_INTERVAL = 0.25


class RuntimeError_(RuntimeError):
    """A runtime that could not be found, started, or reached."""


def server_binary() -> str | None:
    """Return the llama-server to use, or None when nothing is built."""
    candidate = os.path.join(paths.runtime_dir(), "build", "bin",
                             "llama-server")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("llama-server")


def _free_port() -> int:
    """Ask the kernel for a port, then hand it straight to the server.

    Racy in principle. In practice the alternative — a fixed port — collides
    with a second instance of this UI, which is the case that actually occurs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Server:
    """One llama-server process, owned for the lifetime of one chat session."""

    model_path: str
    context: int = 4096
    threads: int = 0
    port: int = 0
    process: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, on_progress: Callable[[str], None] | None = None) -> None:
        binary = server_binary()
        if binary is None:
            raise RuntimeError_(
                "no llama-server is available. Build it with "
                "scripts/build-runtime.sh, or put one on PATH.")
        if not os.path.isfile(self.model_path):
            raise RuntimeError_(f"no model file at {self.model_path}")
        self.port = self.port or _free_port()
        argv = [binary, "-m", self.model_path, "--host", "127.0.0.1",
                "--port", str(self.port), "-c", str(self.context)]
        if self.threads:
            argv += ["-t", str(self.threads)]
        # Its own process group, so a Ctrl-C in the UI's terminal does not
        # reach the server and leave the UI talking to a corpse.
        self.process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", start_new_session=True)
        self._await_health(on_progress)

    def _await_health(self, on_progress: Callable[[str], None] | None) -> None:
        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError_(
                    "llama-server exited while loading the model "
                    f"(status {self.process.returncode}). "
                    "Run it by hand to see why.")
            try:
                with urllib.request.urlopen(
                        f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            if on_progress is not None:
                waited = LOAD_TIMEOUT - (deadline - time.monotonic())
                on_progress(f"loading the model… {waited:.0f}s")
            time.sleep(HEALTH_INTERVAL)
        self.stop()
        raise RuntimeError_(
            f"llama-server did not become ready within {LOAD_TIMEOUT:.0f}s")

    def stop(self) -> None:
        """Terminate the server, and make sure it is actually gone.

        A model server holding gigabytes resident is exactly the process you do
        not want surviving the UI that started it, so this escalates rather
        than trusting a polite signal.
        """
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()

    # -- inference ----------------------------------------------------------

    def stream_chat(self, messages: list[Message], *, temperature: float = 0.7,
                    top_p: float = 0.9, max_tokens: int = 1024,
                    should_stop: Callable[[], bool] | None = None
                    ) -> Iterator[str]:
        """Yield text fragments as the model produces them.

        `should_stop` is polled between fragments; returning True closes the
        connection, which is how the UI abandons a generation. llama-server
        drops the slot when the client disappears, so an abandoned turn stops
        costing compute rather than running to completion unseen.
        """
        payload = json.dumps({
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as error:
            raise RuntimeError_(
                f"the model server refused the request: {error}") from error
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError_(
                f"lost the model server: {error}") from error
        try:
            for raw in response:
                if should_stop is not None and should_stop():
                    return
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", ()):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
        finally:
            response.close()

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
