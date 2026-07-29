"""Image generation, driven through the image scaffold's own CLI.

Nothing here reimplements a diffusion pipeline. The scaffold that owns these
weights already has a working command with a local path and a remote one, and
it is the copy that is actually tested against them — so this drives that
command and confines itself to the parts a UI is better at: composing a
request, remembering what was asked, and showing the result.

Which backend to use is a fact about the machine, not a preference. A card that
cannot execute the kernels fails at the first launch, minutes into a model
load, so the backend is chosen from what `doctor` reports rather than from a
default that is wrong half the time.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

LOCAL = "local"
REMOTE = "remote"

# The image scaffold reaches a GPU host through a subcommand named after that
# host, which is per-machine configuration rather than something this published
# repository should carry. Set KILIX_BONSAI_IMAGE_REMOTE to whatever the local
# scaffold calls it; remote generation is simply unavailable until it is set,
# which is the correct answer on a machine with no second GPU anyway.
REMOTE_SUBCOMMAND = os.environ.get("KILIX_BONSAI_IMAGE_REMOTE", "")

# The scaffold requires multiples of 32 and 512x512 is its fast preview.
SIZE_STEP = 32
PRESETS = [
    ("square fast", 512, 512),
    ("square", 1024, 1024),
    ("landscape fast", 624, 416),
    ("landscape", 1248, 832),
    ("portrait fast", 416, 624),
    ("portrait", 832, 1248),
    ("wide", 1408, 704),
]


class ImageError(RuntimeError):
    """An image backend that could not be found or run."""


def cli_path() -> str | None:
    """Return the image scaffold's CLI, or None when it is not installed."""
    override = os.environ.get("KILIX_BONSAI_IMAGE_CLI")
    if override:
        return override if os.access(override, os.X_OK) else None
    for candidate in (
            os.path.join(os.path.expanduser("~"), "bonsai_image_generation",
                         "bonsai"),
            os.path.join(os.environ.get("GPU_TERMINAL_SOURCE_HOME")
                         or os.path.expanduser("~/.local/gpu_terminal/sources"),
                         "bonsai_image_generation", "bonsai")):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("bonsai")


def probe_backends(timeout: float = 120.0) -> dict[str, tuple[bool, str]]:
    """Ask the CLI which backends actually work here.

    Returns {backend: (usable, one-line reason)}. Both are probed because the
    answer differs per machine and neither is a safe assumption: this host
    cannot run the kernels locally but has a remote that can, and a laptop with
    no remote configured is the exact mirror image.
    """
    cli = cli_path()
    if cli is None:
        return {LOCAL: (False, "the image CLI was not found"),
                REMOTE: (False, "the image CLI was not found")}
    results = {}
    probes = [(LOCAL, [cli, "doctor"])]
    if REMOTE_SUBCOMMAND:
        probes.append((REMOTE, [cli, REMOTE_SUBCOMMAND, "doctor"]))
    else:
        results[REMOTE] = (False, "KILIX_BONSAI_IMAGE_REMOTE is not set")
    for backend, argv in probes:
        try:
            done = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout)
        except (OSError, subprocess.SubprocessError) as error:
            results[backend] = (False, str(error))
            continue
        output = (done.stdout + done.stderr).strip().splitlines()
        verdict = next((line for line in reversed(output)
                        if line.lower().startswith("result:")), "")
        usable = done.returncode == 0 and "ready" in verdict.lower()
        reason = verdict or (output[-1] if output else "no output")
        results[backend] = (usable, reason.strip())
    return results


@dataclass
class Request:
    """One generation, as the UI composed it."""

    prompt: str
    width: int = 512
    height: int = 512
    seed: int | None = None
    steps: int | None = None
    input_image: str | None = None
    backend: str = REMOTE

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    def validate(self) -> str | None:
        """Return a human-readable problem, or None when it is runnable."""
        if not self.prompt.strip():
            return "a prompt is required"
        for name, value in (("width", self.width), ("height", self.height)):
            if value % SIZE_STEP:
                return f"{name} must be a multiple of {SIZE_STEP}"
            if not 128 <= value <= 2048:
                return f"{name} must be between 128 and 2048"
        if self.input_image and not os.path.isfile(self.input_image):
            return f"no input image at {self.input_image}"
        if self.backend == REMOTE and not REMOTE_SUBCOMMAND:
            return ("remote generation needs KILIX_BONSAI_IMAGE_REMOTE set to "
                    "the image scaffold's remote subcommand")
        return None

    def argv(self, cli: str, output: str) -> list[str]:
        argv = [cli]
        argv += [REMOTE_SUBCOMMAND, "generate", "--wait"] \
            if self.backend == REMOTE and REMOTE_SUBCOMMAND else ["generate"]
        argv += ["-p", self.prompt, "--size", self.size, "--output", output]
        if self.seed is not None:
            argv += ["--seed", str(self.seed)]
        if self.steps is not None:
            argv += ["--steps", str(self.steps)]
        if self.input_image:
            argv += ["--input-image", self.input_image]
        return argv


@dataclass
class Result:
    request: Request
    path: str
    ok: bool
    detail: str
    seconds: float


@dataclass
class Gallery:
    """Where generations land, and what was asked for each one.

    The sidecar JSON is the point: an image whose prompt and seed were not
    recorded cannot be iterated on, and iterating is what this UI is for.
    """

    directory: str
    entries: list[Result] = field(default_factory=list)

    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def next_path(self, index: int) -> str:
        return os.path.join(self.directory, f"generation-{index:04d}.png")

    def record(self, result: Result) -> None:
        self.entries.append(result)
        if not result.ok:
            return
        sidecar = os.path.splitext(result.path)[0] + ".json"
        try:
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump({
                    "prompt": result.request.prompt,
                    "size": result.request.size,
                    "seed": result.request.seed,
                    "steps": result.request.steps,
                    "input_image": result.request.input_image,
                    "backend": result.request.backend,
                    "seconds": round(result.seconds, 1),
                }, handle, indent=2)
        except OSError:
            pass                      # a missing sidecar must not lose the PNG

    def load(self) -> None:
        """Re-read a previous session's generations from their sidecars."""
        try:
            names = sorted(os.listdir(self.directory))
        except OSError:
            return
        for name in names:
            if not name.endswith(".png"):
                continue
            path = os.path.join(self.directory, name)
            sidecar = os.path.splitext(path)[0] + ".json"
            try:
                with open(sidecar, encoding="utf-8") as handle:
                    saved = json.load(handle)
            except (OSError, json.JSONDecodeError):
                saved = {}
            width, _, height = str(saved.get("size", "512x512")).partition("x")
            self.entries.append(Result(
                request=Request(
                    prompt=saved.get("prompt", "(prompt not recorded)"),
                    width=int(width or 512), height=int(height or 512),
                    seed=saved.get("seed"), steps=saved.get("steps"),
                    input_image=saved.get("input_image"),
                    backend=saved.get("backend", REMOTE)),
                path=path, ok=True, detail="from a previous session",
                seconds=float(saved.get("seconds") or 0.0)))


def generate(request: Request, output: str, *,
             timeout: float = 900.0) -> Result:
    """Run one generation and return where it landed."""
    cli = cli_path()
    started = time.monotonic()
    if cli is None:
        return Result(request, output, False,
                      "the image CLI was not found", 0.0)
    problem = request.validate()
    if problem:
        return Result(request, output, False, problem, 0.0)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    try:
        done = subprocess.run(request.argv(cli, output), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(request, output, False,
                      f"gave up after {timeout:.0f}s",
                      time.monotonic() - started)
    except OSError as error:
        return Result(request, output, False, str(error),
                      time.monotonic() - started)
    seconds = time.monotonic() - started
    if done.returncode != 0 or not os.path.isfile(output):
        lines = [line for line in (done.stdout + done.stderr).splitlines()
                 if line.strip()]
        detail = next((line for line in reversed(lines)
                       if "ERR" in line or "rror" in line),
                      lines[-1] if lines else "generation failed")
        return Result(request, output, False, detail.strip()[:200], seconds)
    return Result(request, output, True, "ok", seconds)
