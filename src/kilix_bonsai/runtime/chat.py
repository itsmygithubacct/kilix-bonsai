"""Chat against the regular Bonsai checkpoints, on whichever GPU is free.

These are not stock llama.cpp models — `Q1_0` is not a ggml tensor type — so
this drives the vendor's pinned llama.cpp build, which understands it. That
build is CUDA-only with no CPU fallback, so a host without a working driver
cannot serve chat at all and has to borrow one, exactly as image generation
already does.

Two policies live here, both deliberate:

**The GPU is claimed on demand and released.** A resident 27B holds most of an
8 GB card, and the image model wants that same card. Holding it between turns
would mean the two tools quietly exclude each other, so the process is started
for a session and stopped when the session ends.

**8B is the default; 27B is earned.** Free VRAM is measured before choosing,
not assumed, because the failure when it is wrong is an out-of-memory abort
minutes into a load rather than a refusal.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass

# Measured, not guessed: the vendor's own notes put the regular 27B process at
# about 4.17 GiB by itself at the default 4096-token context. The headroom
# covers the KV cache growing over a long conversation and anything else
# already resident on the card.
VRAM_27B_MIB = 4270
VRAM_8B_MIB = 1600
HEADROOM_MIB = 1300

# The vendor launcher's own preset, from the model card. Repeated here so a
# dry run can be compared against it rather than trusted.
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

LOCAL = "local"
REMOTE = "remote"

# The remote is named after a machine, which is configuration rather than
# something a published repository should carry.
REMOTE_HOST = os.environ.get("KILIX_BONSAI_CHAT_REMOTE", "")


class ChatError(RuntimeError):
    """A chat backend that could not be found, reached, or fitted."""


# Verified by dry-run against the real launchers rather than inferred from
# their names, which mislead: `bonsai-cli` is the *deterministic integer*
# engine ("launch the Bonsai-8B deterministic integer inference engine"),
# while `bonsai-27b-cli` is the regular one. Both are thin wrappers over one
# CLI, and what selects the regular path is `--engine prismml.cpp` with the
# pinned binary directory — a flag, not a separate program. So regular 8B is
# reachable by the same route pointed at the 8B GGUF.
REGULAR_ENGINE = "prismml.cpp"


def launcher(model_id: str) -> str | None:
    """Return the vendor launcher for a model, or None when absent.

    Only the 27B wrapper selects the regular engine on its own. For 8B the
    wrapper would give the integer engine instead, so a regular 8B session has
    to pass the engine flag explicitly.
    """
    names = {"bonsai-8b": "bonsai-cli", "bonsai-27b": "bonsai-27b-cli"}
    name = names.get(model_id)
    if name is None:
        return None
    override = os.environ.get("KILIX_BONSAI_CHAT_LAUNCHER_DIR")
    if override:
        candidate = os.path.join(override, name)
        return candidate if os.access(candidate, os.X_OK) else None
    return shutil.which(name)


def free_vram_mib(backend: str = LOCAL, timeout: float = 30.0) -> int | None:
    """Return free VRAM in MiB, or None when no GPU can be queried.

    None and 0 mean different things and are kept apart: None is "there is no
    GPU here to ask", 0 would be "there is one and it is full".
    """
    query = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    argv = query if backend == LOCAL else (
        ["ssh", REMOTE_HOST, " ".join(query)] if REMOTE_HOST else None)
    if argv is None:
        return None
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    first = (done.stdout or "").strip().splitlines()
    try:
        return int(first[0].strip())
    except (IndexError, ValueError):
        return None


def needs_mib(model_id: str) -> int:
    return (VRAM_27B_MIB if model_id == "bonsai-27b" else VRAM_8B_MIB) \
        + HEADROOM_MIB


def fits(model_id: str, free_mib: int | None) -> bool:
    return free_mib is not None and free_mib >= needs_mib(model_id)


@dataclass
class Turn:
    """One message. The vendor CLI is one-shot, so history is re-rendered
    into a single prompt per turn rather than held by a server."""

    role: str
    content: str


@dataclass
class Choice:
    """Which model to open, on which backend, and why."""

    model_id: str
    backend: str
    free_mib: int | None
    reason: str
    usable: bool


def choose(preferred: str | None = None,
           available: tuple[str, ...] = ("bonsai-8b", "bonsai-27b")) -> Choice:
    """Pick a model and a backend from what is actually free right now.

    8B is the default. 27B is used only when it was explicitly asked for and
    the measured free VRAM covers it — an unasked-for upgrade to a model four
    times the size is not a favour.
    """
    for backend in (LOCAL, REMOTE):
        if backend == REMOTE and not REMOTE_HOST:
            continue
        free = free_vram_mib(backend)
        if free is None:
            continue
        if preferred and preferred in available and fits(preferred, free):
            return Choice(preferred, backend, free,
                          f"{free} MiB free covers {preferred}", True)
        if preferred == "bonsai-27b" and not fits("bonsai-27b", free):
            # Say so rather than silently substituting: someone who asked for
            # 27B wants to know it did not fit.
            if fits("bonsai-8b", free):
                return Choice("bonsai-8b", backend, free,
                              f"27B needs {needs_mib('bonsai-27b')} MiB, only "
                              f"{free} MiB free — falling back to 8B", True)
        if fits("bonsai-8b", free):
            return Choice("bonsai-8b", backend, free,
                          f"{free} MiB free on the {backend} GPU", True)
    return Choice(preferred or "bonsai-8b", LOCAL, None,
                  "no GPU with enough free memory was found; the vendor "
                  "runtime is CUDA-only and has no CPU path", False)


@dataclass
class Session:
    """One vendor-launcher process, held only while a session is open.

    Released rather than kept warm: the card is shared with image generation,
    and a model that holds it between conversations makes the other tool fail
    for reasons its user cannot see.
    """

    choice: Choice
    context: int = 4096
    process: subprocess.Popen | None = None

    def argv(self, prompt: str, tokens: int = 256) -> list[str]:
        # A remote launcher must be resolved on the machine that will run it.
        # Probing this filesystem for it would fail on exactly the hosts that
        # need the remote — the ones with no usable GPU and no runtime.
        binary = (os.environ.get("KILIX_BONSAI_CHAT_LAUNCHER_DIR", "").rstrip("/")
                  + "/" + {"bonsai-8b": "bonsai-cli",
                           "bonsai-27b": "bonsai-27b-cli"}[self.choice.model_id]
                  ) if self.choice.backend == REMOTE else launcher(
                      self.choice.model_id)
        if self.choice.backend == REMOTE and not os.environ.get(
                "KILIX_BONSAI_CHAT_LAUNCHER_DIR"):
            raise ChatError(
                "remote chat needs KILIX_BONSAI_CHAT_LAUNCHER_DIR set to the "
                "directory holding the launchers on the remote host")
        if binary is None:
            raise ChatError(
                f"no vendor launcher for {self.choice.model_id}; install the "
                "pinned runtime, or set KILIX_BONSAI_CHAT_LAUNCHER_DIR")
        # The launcher auto-fits its context from the model's 262144-token
        # maximum, which asks for a 16 GB KV cache and dies on an 8 GB card.
        # Pinning it is not optional; the vendor's own default of 4096 is
        # chosen for exactly this hardware.
        env = [f"BONSAI_27B_CTX_SIZE={self.context}",
               f"BONSAI_CONTEXT_SIZE={self.context}"]
        if self.choice.model_id != "bonsai-8b":
            return_local = ["env", *env, binary, prompt, "-n", str(tokens)]
            local = return_local
        else:
            # 8B has no regular wrapper — `bonsai-cli` hardcodes the integer
            # engine's flags and exits 1 when handed --engine. So the shared
            # CLI is invoked directly with the same arguments the 27B wrapper
            # builds for itself, pointed at the 8B weights.
            home = os.environ.get("KILIX_BONSAI_NOTARY_HOME", "")
            binaries = os.environ.get("KILIX_BONSAI_PRISM_BIN", "")
            gguf = os.environ.get("KILIX_BONSAI_8B_GGUF", "")
            if not (home and binaries and gguf):
                raise ChatError(
                    "regular 8B needs KILIX_BONSAI_NOTARY_HOME, "
                    "KILIX_BONSAI_PRISM_BIN and KILIX_BONSAI_8B_GGUF set — it "
                    "has no wrapper of its own, unlike 27B")
            # Absolute PYTHONPATH: the vendor wrapper gets away with a
            # relative "src" because it cd's first, and this does not.
            engine_root = os.path.join(os.path.dirname(binary), "bonsai")
            local = ["env", f"BONSAI_NOTARY_HOME={home}",
                     f"PYTHONPATH={os.path.join(engine_root, 'src')}",
                     f"LD_LIBRARY_PATH={binaries}", *env,
                     os.path.join(engine_root, ".venv", "bin", "python"),
                     "-m", "trinote.cli.run_bonsai_cli",
                     "--engine", REGULAR_ENGINE, "--no-receipt",
                     "--gguf", gguf, "--bin-dir", binaries,
                     "--n-gpu-layers", "99", "--flash-attn",
                     "-n", str(tokens), "-p", prompt]
        if self.choice.backend == LOCAL:
            return local
        if not REMOTE_HOST:
            raise ChatError("KILIX_BONSAI_CHAT_REMOTE is not set")
        # shlex.quote, not list2cmdline: the latter applies Windows quoting
        # rules, which leaves a multi-word prompt to be word-split by the
        # remote shell into arguments the launcher rejects.
        return ["ssh", REMOTE_HOST,
                " ".join(shlex.quote(part) for part in local)]

    def render(self, history: list[Turn], system: str = "") -> str:
        """Flatten a conversation into one prompt.

        The launcher takes a prompt and exits; there is no server holding
        state. Re-sending the transcript each turn is what makes the
        conversation continuous, and is also why long ones get slower.
        """
        parts = [system.strip()] if system.strip() else []
        for turn in history:
            if not turn.content.strip():
                continue
            speaker = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{speaker}: {turn.content.strip()}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def stream(self, history: list[Turn], system: str = "", tokens: int = 512,
               should_stop=None):
        """Run one turn, yielding output as it arrives, then release the GPU.

        The card is claimed for exactly this call. `should_stop` returning True
        kills the process, which is the only way to abandon a turn when the
        backend is a one-shot command rather than a cancellable request.
        """
        argv = self.argv(self.render(history, system), tokens)
        try:
            self.process = subprocess.Popen(
                # stderr is merged, not discarded: the launcher reports why it
                # refused there, and swallowing it turns every failure into a
                # silent empty answer.
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1,
                start_new_session=True)
        except OSError as error:
            raise ChatError(str(error)) from error
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if should_stop is not None and should_stop():
                    return
                yield line
        finally:
            self.release()

    def dry_run(self, prompt: str) -> list[str]:
        """Return the command without running it.

        The vendor launcher honours BONSAI_DRYRUN, so what it would run can be
        compared against what we think it will run.
        """
        return self.argv(prompt)

    def release(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
