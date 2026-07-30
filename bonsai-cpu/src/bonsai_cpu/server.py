"""A llama-server of our own: spawned, health-checked, killed as a group.

The chat interface talks to an HTTP server rather than scraping a CLI, and
that choice is what makes the interface worth using: tokens arrive as they
are produced, the model's reasoning arrives on its own channel, a turn can
be abandoned by closing the connection, and the server holds the KV cache so
the second turn does not pay for the first again.

The cost is a process to own, and this module owns it completely: always our
own child on a kernel-chosen port — never someone else's server, because
model switching needs restart authority and kill-on-exit is only safe on a
child of ours. Its output goes to a log file, not a pipe: a pipe nobody
drains fills at 64 KB and wedges the server mid-session, a failure that
would appear as "chat stopped answering" hours from its cause.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import models

# A cold page cache maps gigabytes slowly, and the failure a user actually
# hits is "I gave up before it finished" — so the ceiling is generous and
# the wait reports progress rather than blocking silently.
LOAD_TIMEOUT = 600.0
HEALTH_INTERVAL = 0.25
LOG_TAIL = 15


class ServerError(RuntimeError):
    """A server that could not be started, reached, or kept."""


@dataclass
class StreamEvent:
    """One thing the server said while generating.

    kind is "reasoning" or "content" for text, "progress" for prompt
    processing (data = the server's prompt_progress object), and "stats"
    for timings (data = the server's timings object).
    """

    kind: str
    text: str = ""
    data: dict | None = None


def _free_port() -> int:
    """Ask the kernel for a port, then hand it straight to the server.

    Racy in principle. In practice the alternative — a fixed port — collides
    with `bonsai-cpu serve` or a second chat, which is the case that occurs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def default_log_path() -> str:
    return os.path.join(models.GPU_TERMINAL_HOME, "bonsai-cpu", "logs",
                        "llama-server.log")


@dataclass
class ChatServer:
    """One llama-server process, owned for the lifetime of one chat."""

    model_id: str
    model_path: str
    context: int
    threads: int
    port: int = 0
    process: subprocess.Popen | None = None
    log_path: str = ""
    _response: object = field(default=None, repr=False)
    _aborted: bool = field(default=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _log_tail(self) -> str:
        try:
            with open(self.log_path, errors="replace") as fh:
                lines = fh.readlines()[-LOG_TAIL:]
            return "".join(lines).rstrip()
        except OSError:
            return "(no log)"

    def start(self, on_progress: Callable[[str], None] | None = None) -> None:
        try:
            binary = models.binary("llama-server")
        except models.ModelError as error:
            raise ServerError(str(error)) from error
        if not os.path.isfile(self.model_path):
            raise ServerError(f"no model file at {self.model_path}")
        self.port = self.port or _free_port()
        self.log_path = self.log_path or default_log_path()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        argv = [binary, "-m", self.model_path, "--host", "127.0.0.1",
                "--port", str(self.port), "-c", str(self.context),
                "-t", str(self.threads)]
        # Its own process group, so a Ctrl-C in the UI's terminal does not
        # reach the server and leave the UI talking to a corpse.
        with open(self.log_path, "ab") as log:
            self.process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        self._await_health(on_progress)

    def _await_health(self, on_progress: Callable[[str], None] | None) -> None:
        title = models.MODELS.get(self.model_id, {}).get("title", self.model_id)
        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                status = self.process.returncode
                raise ServerError(
                    f"llama-server exited while loading (status {status}).\n"
                    f"log tail ({self.log_path}):\n{self._log_tail()}")
            try:
                with urllib.request.urlopen(
                        f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            if on_progress is not None:
                waited = LOAD_TIMEOUT - (deadline - time.monotonic())
                on_progress(f"loading {title}… {waited:.0f}s")
            time.sleep(HEALTH_INTERVAL)
        self.stop()
        raise ServerError(
            f"llama-server did not become ready within {LOAD_TIMEOUT:.0f}s")

    def stop(self) -> None:
        """Terminate the server, and make sure it is actually gone.

        A process holding gigabytes resident is exactly the one you do not
        want surviving the UI that started it, so this escalates rather than
        trusting a polite signal — and it signals the whole group, because
        the runtime may have children of its own.
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

    def abort(self) -> None:
        """Close the in-flight response from another thread.

        This is what makes Esc instant at one token per second: the reader
        is almost always blocked inside a socket read, and closing the
        response is the only thing that unblocks it now rather than a token
        from now. llama-server frees the slot when the client vanishes.
        """
        response, self._response = self._response, None
        self._aborted = True
        if response is not None:
            try:
                response.close()
            except OSError:
                pass

    def stream_chat(self, messages: list[dict], *, temperature: float,
                    top_p: float, top_k: int, max_tokens: int,
                    enable_thinking: bool | None = None,
                    should_stop: Callable[[], bool] | None = None
                    ) -> Iterator[StreamEvent]:
        """Yield StreamEvents as the model produces them.

        `enable_thinking` is sent only when it is not None: the 8B has no
        thinking to switch, and sending template kwargs it does not use
        would be noise. Verified against the pinned server: the kwarg is
        honoured per-request, reasoning arrives as delta.reasoning_content,
        and timings/prompt_progress ride along in-band.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "stream": True,
            "timings_per_token": True,
            "return_progress": True,
        }
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking}
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        self._aborted = False
        try:
            response = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as error:
            raise ServerError(
                f"the model server refused the request: {error}") from error
        except (urllib.error.URLError, OSError) as error:
            raise ServerError(f"lost the model server: {error}") from error
        self._response = response
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
                progress = chunk.get("prompt_progress")
                if progress:
                    yield StreamEvent("progress", data=progress)
                for choice in chunk.get("choices", ()):
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield StreamEvent("reasoning", text=reasoning)
                    piece = delta.get("content")
                    if piece:
                        yield StreamEvent("content", text=piece)
                timings = chunk.get("timings")
                if timings:
                    yield StreamEvent("stats", data=timings)
        except (ValueError, OSError) as error:
            # A close() from abort() lands here as a read on a dead socket;
            # that is a cancel, not a failure.
            if self._aborted:
                return
            raise ServerError(f"lost the model server: {error}") from error
        finally:
            self._response = None
            try:
                response.close()
            except OSError:
                pass

    def __enter__(self) -> "ChatServer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


def swap(old: ChatServer | None, model_id: str, *,
         context: int | None = None, threads: int | None = None,
         on_progress: Callable[[str], None] | None = None) -> ChatServer:
    """Stop one server, start another, sequentially.

    Sequential on purpose: one resident model at a time keeps peak memory at
    one model's worth and the page cache honest. The progress callback
    narrates both halves, because the stop of a 27B is not instant either.
    """
    if old is not None:
        if on_progress is not None:
            title = models.MODELS.get(old.model_id, {}).get(
                "title", old.model_id)
            on_progress(f"stopping {title}…")
        old.stop()
    spec = models.MODELS[model_id]
    try:
        path = models.resolve_runnable(model_id)
    except models.ModelError as error:
        raise ServerError(str(error)) from error
    server = ChatServer(
        model_id=model_id, model_path=path,
        context=context or spec["ctx"],
        threads=threads or models.physical_cores())
    server.start(on_progress)
    return server
