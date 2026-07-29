"""Speech recognition, and recording for it to transcribe.

The obvious design here is the one that does not work. VibeASR ships
`asr_stream_server` — one resident process, models loaded once, audio paths in
and transcriptions out — which is exactly the shape a UI wants. In the build
available here it aborts inside `vae_encode_impl` on *every* clip, including
ones its one-shot sibling transcribes without complaint. So this drives
`asr_infer` instead and pays a model load (about 1.5s) per utterance. If the
server is ever fixed, `engine_binary("asr_stream_server")` is where to start.

Long audio is chunked, for feedback rather than for correctness: the one-shot
engine handles any length tested, but at roughly 1.9x real time on this CPU a
five-minute recording is ten minutes of a UI showing nothing. Chunks turn that
into a transcript that grows every half minute.

Recording stays optional. A machine with no capture device can still transcribe
files, and that path is worth keeping usable rather than gating the tool on a
microphone.
"""
from __future__ import annotations

import array
import os
import shutil
import signal
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Callable

from .. import paths

# 24 kHz mono 16-bit is what the model was trained on, and the engine resamples
# anything else, so recording at the native rate avoids a needless conversion.
SAMPLE_RATE = 24000

# Large enough to amortise the per-invocation model load, small enough that a
# long recording produces visible progress.
CHUNK_SECONDS = 30.0

# How far either side of a chunk boundary to hunt for a quiet moment. Cutting
# mid-word costs accuracy on both sides; a syllable of slack usually finds a
# gap between words.
SEEK_SECONDS = 0.6


class AsrError(RuntimeError):
    """A speech runtime that could not be found or run."""


def engine_binary(name: str = "asr_infer") -> str | None:
    """Return a VibeASR binary, or None when it is not built."""
    for candidate in (
            os.path.join(paths.runtime_dir(), "build", "bin", name),
            os.path.join(os.path.expanduser("~"), "VibeASR.cpp", "build",
                         "bin", name)):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which(name)


def recorder() -> list[str] | None:
    """Return an argv template for recording 24 kHz mono WAV, or None."""
    if shutil.which("pw-record"):
        return ["pw-record", "--rate", str(SAMPLE_RATE), "--channels", "1",
                "--format", "s16"]
    if shutil.which("parecord"):
        return ["parecord", "--rate", str(SAMPLE_RATE), "--channels", "1",
                "--format=s16le", "--file-format=wav"]
    if shutil.which("arecord"):
        return ["arecord", "-q", "-r", str(SAMPLE_RATE), "-c", "1", "-f",
                "S16_LE", "-t", "wav"]
    return None


def audio_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate()
                                               or SAMPLE_RATE)
    except (OSError, wave.Error):
        return 0.0


def _quietest_split(frames: bytes, channels: int, width: int, around: int,
                    slack: int) -> int:
    """Return a frame index near `around` where the audio is quietest.

    Only 16-bit PCM is examined; for any other width the exact boundary is
    used, since guessing at an unknown sample format would be worse than an
    occasional mid-word cut.
    """
    if width != 2 or slack <= 0:
        return around
    low = max(0, around - slack)
    high = min(len(frames) // (channels * width), around + slack)
    if high - low < 2:
        return around
    samples = array.array("h")
    samples.frombytes(frames[low * channels * width: high * channels * width])
    window = max(1, int(0.02 * SAMPLE_RATE))
    best, best_energy = around, None
    for start in range(0, max(1, len(samples) - window), window):
        chunk = samples[start:start + window]
        energy = sum(abs(value) for value in chunk) / len(chunk)
        if best_energy is None or energy < best_energy:
            best_energy, best = energy, low + (start + window // 2) // channels
    return best


def split_audio(path: str, directory: str,
                seconds: float = CHUNK_SECONDS) -> list[str]:
    """Split a WAV at quiet boundaries; short files pass through untouched."""
    total = audio_seconds(path)
    if total <= seconds or total == 0.0:
        return [path]
    with wave.open(path, "rb") as source:
        params = source.getparams()
        frames = source.readframes(params.nframes)
    per_chunk = int(seconds * params.framerate)
    slack = int(SEEK_SECONDS * params.framerate)
    stride = params.nchannels * params.sampwidth
    chunks: list[str] = []
    start = index = 0
    while start < params.nframes:
        boundary = min(start + per_chunk, params.nframes)
        if boundary < params.nframes:
            boundary = max(start + per_chunk // 2,
                           _quietest_split(frames, params.nchannels,
                                           params.sampwidth, boundary, slack))
        target = os.path.join(directory, f"chunk-{index:03d}.wav")
        with wave.open(target, "wb") as sink:
            sink.setparams(params)
            sink.writeframes(frames[start * stride: boundary * stride])
        chunks.append(target)
        start, index = boundary, index + 1
    return chunks


@dataclass
class Engine:
    """One-shot transcription through `asr_infer`."""

    vae_path: str
    lm_path: str
    threads: int = 4
    greedy: bool = True
    temperature: float = 0.7
    context_words: str = ""

    def check(self) -> str | None:
        """Return a human-readable problem, or None when this can run."""
        if engine_binary() is None:
            return ("no asr_infer binary was found — build VibeASR.cpp or put "
                    "it on PATH")
        for label, path in (("VAE encoder", self.vae_path),
                            ("LM decoder", self.lm_path)):
            if not os.path.isfile(path):
                return f"the {label} is not downloaded ({path})"
        return None

    def argv(self, audio_path: str) -> list[str]:
        binary = engine_binary()
        argv = [str(binary), "--vae-model", self.vae_path,
                "--lm-model", self.lm_path, "--audio", audio_path,
                "-t", str(self.threads)]
        if self.greedy:
            argv.append("--greedy")
        else:
            argv += ["--temperature", f"{self.temperature:.2f}"]
        if self.context_words.strip():
            argv += ["--context", self.context_words.strip()]
        return argv

    def transcribe_one(self, audio_path: str, timeout: float = 900.0) -> str:
        """Transcribe one clip. Raises AsrError with the runtime's own words."""
        problem = self.check()
        if problem:
            raise AsrError(problem)
        if not os.path.isfile(audio_path):
            raise AsrError(f"no audio file at {audio_path}")
        try:
            done = subprocess.run(self.argv(audio_path), capture_output=True,
                                  text=True, errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise AsrError(f"gave up after {timeout:.0f}s") from error
        except OSError as error:
            raise AsrError(str(error)) from error
        if done.returncode != 0:
            detail = next((line for line in reversed(
                (done.stdout + done.stderr).splitlines()) if line.strip()),
                f"exited {done.returncode}")
            raise AsrError(detail.strip()[:200])
        return self._text_of(done.stdout)

    @staticmethod
    def _text_of(output: str) -> str:
        """Pull the transcription out of the engine's report.

        It prints timing lines and then the text. Everything up to and
        including the statistics line is discarded rather than pattern-matched
        word by word, so a future extra metric does not end up in a transcript.
        """
        lines = output.splitlines()
        cut = 0
        for index, line in enumerate(lines):
            if "RTF" in line or line.strip().startswith(("VAE:", "Audio:")):
                cut = index + 1
        return "\n".join(lines[cut:]).strip()

    def transcribe(self, audio_path: str,
                   on_partial: Callable[[str], None] | None = None,
                   seconds: float = CHUNK_SECONDS) -> str:
        """Transcribe audio of any length, reporting the transcript as it grows."""
        if audio_seconds(audio_path) <= seconds:
            text = self.transcribe_one(audio_path)
            if on_partial is not None:
                on_partial(text)
            return text
        with tempfile.TemporaryDirectory(prefix="kilix-bonsai-asr-") as work:
            pieces: list[str] = []
            for chunk in split_audio(audio_path, work, seconds):
                pieces.append(self.transcribe_one(chunk))
                if on_partial is not None:
                    on_partial(" ".join(p for p in pieces if p).strip())
        return " ".join(piece for piece in pieces if piece).strip()


@dataclass
class Recording:
    """A capture in progress, stoppable from the UI thread."""

    path: str
    process: subprocess.Popen | None = None
    started: float = 0.0

    def start(self) -> None:
        template = recorder()
        if template is None:
            raise AsrError(
                "no recorder found: install pipewire-utils, pulseaudio-utils, "
                "or alsa-utils, or transcribe an existing WAV instead.")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.process = subprocess.Popen(
            [*template, self.path], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        self.started = time.monotonic()

    @property
    def active(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started if self.started else 0.0

    def stop(self) -> str:
        """Stop capture and return the path, once the file is actually there."""
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except (OSError, ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        # The recorder writes the WAV header on exit, so a file that is not
        # there yet is a race rather than a failure.
        for _ in range(20):
            if os.path.isfile(self.path) and os.path.getsize(self.path) > 44:
                break
            time.sleep(0.05)
        return self.path
