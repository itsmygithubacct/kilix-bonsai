"""kilix-bonsai-speech — transcribe speech with the BitNet ASR model.

Two ways in: hold a recording, or point it at a WAV. Both land in the same
place, because the useful thing about a transcript is what you do with it
afterwards — `y` copies it, `w` writes it out.

Long audio transcribes in chunks and the text grows on screen as each one
lands. At roughly twice real time on a CPU, a five-minute recording is ten
minutes of work, and a UI that showed nothing for ten minutes would be
indistinguishable from a hung one.

This is the same model and the same files Kilix dictation reads. Nothing here
downloads a second copy, and a transcription started here does not interfere
with the voice daemon — they only share the weights on disk.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from kilix_bonsai import catalog, chrome, screen, widgets       # noqa: E402
from kilix_bonsai.runtime import asr                             # noqa: E402

TITLE = "Kilix Bonsai Speech"


class State:
    def __init__(self, model: catalog.Model) -> None:
        self.model = model
        runtime = model.runtime
        store_dir = model.default_variant.directory(model.store)
        self.engine = asr.Engine(
            vae_path=os.path.join(store_dir, runtime.get("vae_file", "")),
            lm_path=os.path.join(store_dir, runtime.get("lm_file", "")))
        self.editor = widgets.Editor()
        self.scroll = widgets.Scrollback()
        self.transcript = ""
        self.partial = ""
        self.status = self.engine.check() or "ready"
        self.working = False
        self.recording: asr.Recording | None = None
        self.stdscr = None
        self.show_help = False
        self.field = "file"          # or "context"
        self.history: list[tuple[str, str]] = []
        self.workdir = os.path.join(os.path.expanduser("~"),
                                    "kilix-bonsai-speech")

    @property
    def ready(self) -> bool:
        return self.engine.check() is None

    # -- recording ----------------------------------------------------------

    def toggle_record(self) -> None:
        if self.working:
            return
        if self.recording is not None:
            path = self.recording.stop()
            self.recording = None
            self.status = f"recorded {asr.audio_seconds(path):.1f}s"
            self.transcribe(path)
            return
        os.makedirs(self.workdir, exist_ok=True)
        path = os.path.join(self.workdir, f"recording-{int(time.time())}.wav")
        self.recording = asr.Recording(path=path)
        try:
            self.recording.start()
            self.status = "recording — press r again to stop"
        except asr.AsrError as error:
            self.recording = None
            self.status = str(error)

    # -- transcription ------------------------------------------------------

    def transcribe(self, path: str) -> None:
        if self.working:
            return
        if not os.path.isfile(path):
            self.status = f"no audio file at {path}"
            return
        self.working = True
        self.partial = ""
        seconds = asr.audio_seconds(path)
        self.status = f"transcribing {seconds:.1f}s of audio…"
        started = time.monotonic()

        def work() -> None:
            try:
                text = self.engine.transcribe(
                    path, on_partial=self._on_partial)
                elapsed = time.monotonic() - started
                self.transcript = text
                self.history.append((os.path.basename(path), text))
                self.status = (f"done in {elapsed:.0f}s"
                               + (f" (RTF {elapsed / seconds:.2f})"
                                  if seconds else ""))
            except asr.AsrError as error:
                self.status = str(error)
            finally:
                self.partial = ""
                self.working = False
        threading.Thread(target=work, daemon=True).start()

    def _on_partial(self, text: str) -> None:
        self.partial = text
        self.scroll.to_end()

    # -- output -------------------------------------------------------------

    def copy(self) -> None:
        text = self.transcript
        if not text:
            return
        for argv in (["wl-copy"], ["xclip", "-selection", "clipboard"],
                     ["xsel", "--clipboard", "--input"]):
            import shutil
            if shutil.which(argv[0]):
                try:
                    subprocess.run(argv, input=text, text=True, timeout=5)
                    self.status = f"copied with {argv[0]}"
                    return
                except (OSError, subprocess.SubprocessError):
                    continue
        self.status = "no clipboard tool (wl-copy, xclip, or xsel)"

    def write(self) -> None:
        if not self.transcript:
            return
        os.makedirs(self.workdir, exist_ok=True)
        path = os.path.join(self.workdir, f"transcript-{int(time.time())}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.transcript + "\n")
        self.status = f"wrote {os.path.basename(path)}"


HELP = [
    "r        record / stop and transcribe",
    "Enter    transcribe the WAV path in the field",
    "Tab      switch between the path and hotword fields",
    "y        copy the transcript      w  write it to a file",
    "Ctrl-G   greedy / sampled         ?  this help    Ctrl-Q  quit",
]


def render(surface, state: State) -> None:
    height, width = surface.getmaxyx()
    foot = ("r record · Enter transcribe · y copy · w write · ? help "
            "· Ctrl-Q quit")
    if state.working:
        foot = "transcribing… " + foot
    page = chrome.page(TITLE, [state.model.title], node="SPEECH 001")
    page.render(surface, 0, footer=foot, status=state.status[:38])
    top, left, well, well_width = page.content_box()

    if state.show_help:
        for index, line in enumerate(HELP):
            screen.write(surface, top + index, left + 1, line)
        return

    if not state.ready:
        for index, line in enumerate(widgets.wrap(
                state.engine.check() or "", well_width - 2)):
            screen.write(surface, top + index, left + 1, line)
        screen.write(surface, top + 3, left + 1,
                     "Download it from the model store: kilix bonsai")
        return

    label = "wav path" if state.field == "file" else "hotwords"
    visible, cursor = state.editor.view(max(1, width - 14))
    screen.write(surface, top, left, f"> {label:<10} {visible}")
    mode = "greedy" if state.engine.greedy else "sampled"
    screen.write(surface, top + 1, left,
                 f"  decoding   {mode}, {state.engine.threads} threads")
    if state.recording is not None:
        screen.write(surface, top + 2, left,
                     f"  ● recording {state.recording.seconds:.1f}s")

    body = state.partial or state.transcript
    lines = widgets.wrap(body, max(1, well_width - 2)) if body else \
        ["(nothing transcribed yet — press r to record, or type a path)"]
    body_height = max(1, well - 4)
    first = state.scroll.clamp(len(lines), body_height)
    for index, line in enumerate(lines[first:first + body_height]):
        screen.write(surface, top + 4 + index, left + 1, line)


def handle(key: int, state: State) -> bool:
    if state.show_help:
        state.show_help = False
        return True
    if key == 17:                                    # Ctrl-Q
        if state.recording is not None:
            state.recording.stop()
        return False
    if not state.ready:
        return True
    if key == 7:                                     # Ctrl-G
        state.engine.greedy = not state.engine.greedy
        return True
    if key == ord("\t"):
        state.field = "context" if state.field == "file" else "file"
        state.editor.set(state.engine.context_words
                         if state.field == "context" else "")
        return True
    if key in widgets.SUBMIT:
        if state.field == "context":
            state.engine.context_words = state.editor.text
            state.status = "hotwords set"
        else:
            state.transcribe(os.path.expanduser(state.editor.text.strip()))
        return True
    if not state.editor.text:
        if key in (ord("r"), ord("R")):
            state.toggle_record()
            return True
        if key in (ord("y"), ord("Y")):
            state.copy()
            return True
        if key in (ord("w"), ord("W")):
            state.write()
            return True
        if key == ord("?"):
            state.show_help = True
            return True
    state.editor.handle(key)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    positional = [a for a in argv if not a.startswith("-")]
    wanted = positional[0] if positional else None
    try:
        model = catalog.find(wanted) if wanted \
            else next(m for m in catalog.load()
                      if m.runtime.get("kind") == "speech-to-text")
    except (catalog.CatalogError, StopIteration) as error:
        print(f"kilix-bonsai-speech: {error}", file=sys.stderr)
        return 2
    state = State(model)
    if len(positional) > 1:
        state.editor.set(positional[1])
    if path := screen.screenshot_argv(argv):
        with open(path, "w", encoding="utf-8") as target:
            target.write(screen.render_to_text(render, state) + "\n")
        return 0
    return screen.run(render, state, handle=handle, tick_ms=200)


if __name__ == "__main__":
    raise SystemExit(main())
