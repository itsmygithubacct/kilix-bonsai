"""What the models are, where they live, and which binaries run them.

Extracted from the CLI so the chat interface and the command line cannot
drift apart about a digest, a context size, or a refusal message. Errors are
`ModelError` with the exit code the CLI should use; the caller formats them.
"""

import os

GPU_TERMINAL_HOME = os.environ.get(
    "GPU_TERMINAL_HOME", os.path.expanduser("~/.local/gpu_terminal"))
RUNTIME_DIR = os.environ.get(
    "BONSAI_CPU_RUNTIME_DIR",
    os.path.join(GPU_TERMINAL_HOME, "bonsai-cpu", "runtime"))
RUNTIME_SRC = os.path.join(RUNTIME_DIR, "llama.cpp")
RUNTIME_BIN = os.path.join(RUNTIME_SRC, "build", "bin")

STORE_ROOT = os.environ.get(
    "KILIX_BONSAI_MODELS_DIR",
    os.path.join(GPU_TERMINAL_HOME, "kilix-bonsai", "models"))

# What this repository knows how to run. Sizes and digests are the ones
# kilix-bonsai pins; disagreeing with the store about what a model is would
# be worse than not checking.
MODELS = {
    "bonsai-8b": {
        "title": "Bonsai 8B",
        "file": "Bonsai-8B-Q1_0.gguf",
        "size": 1158654496,
        "sha256": "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54",
        "env": "KILIX_BONSAI_BONSAI_8B_DIR",
        "ctx": 8192,
        "supported": True,
        "thinking": False,
        "speed_note": "~8 tok/s on a 4-core laptop",
    },
    "bonsai-27b": {
        "title": "Bonsai 27B",
        "file": "Bonsai-27B-Q1_0.gguf",
        "size": 3803452480,
        "sha256": "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
        "env": "KILIX_BONSAI_BONSAI_27B_DIR",
        "ctx": 4096,
        "supported": True,
        "thinking": True,
        # A thinking model at ~1 tok/s would spend minutes reasoning before
        # its first visible word; answers directly unless asked to think.
        "extra": ["--reasoning", "off"],
        "speed_note": "~1-2 tok/s, thinking model",
    },
}

# Qwen3's recommended sampling, which is what these checkpoints are.
SAMPLING = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]
SAMPLING_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


class ModelError(Exception):
    """A model or runtime that is not usable, with the CLI exit code."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


def physical_cores() -> int:
    """Physical cores where /proc/cpuinfo exposes topology (x86), else the
    logical count. The hyperthreading penalty this avoids was measured on
    x86; on machines without physical id/core id fields, os.cpu_count() is
    the best honest default."""
    seen = set()
    try:
        phys = core = None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        seen.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            seen.add((phys, core))
    except OSError:
        pass
    return len(seen) or os.cpu_count() or 1


def model_dir(model_id: str) -> str:
    spec = MODELS[model_id]
    override = os.environ.get(spec["env"])
    if override:
        return override
    return os.path.join(STORE_ROOT, model_id)


def resolve(model_id: str, override: str | None = None) -> str:
    """Resolve to a file on disk. Informational — no runnability gate, so
    `path` and `verify` work on any model the store may hold."""
    if override:
        return override
    spec = MODELS[model_id]
    path = os.path.join(model_dir(model_id), spec["file"])
    if not os.path.isfile(path):
        raise ModelError(
            f"{spec['title']} is not downloaded ({path} missing).\n"
            f"bonsai-cpu does not download weights — that is the store's "
            f"job:  kilix-bonsai pull {model_id}")
    actual = os.path.getsize(path)
    if actual != spec["size"]:
        raise ModelError(
            f"{path} is {actual} bytes, expected {spec['size']} — "
            f"an interrupted download? `kilix-bonsai pull {model_id}` "
            f"resumes it, `bonsai-cpu verify` checks the digest")
    return path


def resolve_runnable(model_id: str, override: str | None = None) -> str:
    """resolve() plus the gate for commands that execute the model. The
    gate covers explicit paths too: an unverified graph does not become
    verified by naming its file explicitly."""
    spec = MODELS[model_id]
    if not spec["supported"]:
        raise ModelError(spec["unsupported_reason"])
    return resolve(model_id, override)


def binary(name: str) -> str:
    path = os.path.join(RUNTIME_BIN, name)
    if not os.access(path, os.X_OK):
        raise ModelError(
            f"runtime not built ({path} missing) — run: bonsai-cpu build")
    return path


# What `bonsai-cpu build` runs, and the Debian package that provides each
# tool. The compiler row accepts any of the three names cmake would find; the
# package column is what a refusal should tell a Debian operator to install.
# Plebian-OS provisions build-essential but not cmake, so on a fresh box the
# missing one is almost always cmake — the probe reports whatever is absent
# rather than assuming.
BUILD_TOOLS = (
    (("git",), "git"),
    (("cmake",), "cmake"),
    (("c++", "g++", "clang++"), "build-essential"),
)


def missing_build_packages() -> tuple[str, ...]:
    """Debian packages for the build tools this machine is missing.

    Checked *before* the fetch, not discovered by the build: without this a
    doomed `bonsai-cpu build` clones a few hundred megabytes of llama.cpp and
    then dies on one line of stderr, which reads as \"the build silently did
    nothing\" from a TUI.
    """
    import shutil
    return tuple(
        package for names, package in BUILD_TOOLS
        if not any(shutil.which(name) for name in names))
