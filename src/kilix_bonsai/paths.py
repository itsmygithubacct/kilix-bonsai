"""Where weights live, resolved the same way the rest of the stack resolves it.

Every Kilix component derives its writable state from `GPU_TERMINAL_HOME`, and
two of the model stores here are *shared* with components that already own
them: the image weights belong to the image scaffold's data directory, and the
speech weights belong to the voice catalog Kilix dictation reads. Getting those
two paths wrong would mean a second multi-gigabyte copy of a model the machine
already has, so the expansion rules live in one place and are asserted by the
tests rather than repeated per model.

Nothing here creates a directory. Resolution has to be safe to call from a
render loop.
"""
from __future__ import annotations

import os

# The store roots a MODEL.json may name. A model that wants somewhere else has
# to add it here, which is the point: the set of places this repo writes
# multi-gigabyte files should be short enough to read.
_VARIABLES = ("KILIX_BONSAI_MODELS_DIR", "GPU_TERMINAL_HOME",
              "KILIX_STORAGE_HOME", "KILIX_DATA_HOME")


class PathError(ValueError):
    """A store path a MODEL.json asked for that cannot be resolved."""


def _expand(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value))


def gpu_terminal_home() -> str:
    """Return the root of the writable state shared by the whole stack."""
    return _expand(os.environ.get("GPU_TERMINAL_HOME")
                   or os.path.join(os.path.expanduser("~"), ".local",
                                   "gpu_terminal"))


def storage_home() -> str:
    """Return the root of Kilix-owned writable files."""
    return _expand(os.environ.get("KILIX_STORAGE_HOME")
                   or os.path.join(gpu_terminal_home(), "kilix"))


def data_home() -> str:
    """Return the Kilix data directory, durable across sessions.

    This is the parent of the voice catalog, so it has to agree with
    kilix-voice's own resolution exactly — same variable, same default.
    """
    return _expand(os.environ.get("KILIX_DATA_HOME")
                   or os.path.join(storage_home(), "data"))


def models_dir() -> str:
    """Return the default root for models this repo owns outright."""
    return _expand(os.environ.get("KILIX_BONSAI_MODELS_DIR")
                   or os.path.join(gpu_terminal_home(), "kilix-bonsai",
                                   "models"))


def runtime_dir() -> str:
    """Return where built inference runtimes live.

    Out of the source tree for the same reason the weights are: a compiled
    binary is per-machine, and a repository published under a pseudonymous
    identity should not accumulate build output.
    """
    return _expand(os.environ.get("KILIX_BONSAI_RUNTIME_DIR")
                   or os.path.join(gpu_terminal_home(), "kilix-bonsai",
                                   "runtime"))


def venv_dir(model_id: str) -> str:
    """Return the virtualenv `install-deps.sh` builds for one model.

    Deliberately not under the model's store: two of those stores belong to
    other components, and dropping a Python environment into the directory
    kilix-voice enumerates as its speech catalog would be rude at best.
    """
    return os.path.join(gpu_terminal_home(), "kilix-bonsai", "venv", model_id)


def voice_model_dir(catalog_id: str) -> str:
    """Return the directory kilix-voice reads for one speech catalog id.

    kilix-voice builds this from `KILIX_DATA_HOME/voice/models/<id>`; writing
    the speech weights anywhere else would leave dictation still looking at an
    empty directory.
    """
    return os.path.join(data_home(), "voice", "models", catalog_id)


def variables() -> dict[str, str]:
    """Return the store roots a MODEL.json may reference, resolved."""
    return {
        "KILIX_BONSAI_MODELS_DIR": models_dir(),
        "GPU_TERMINAL_HOME": gpu_terminal_home(),
        "KILIX_STORAGE_HOME": storage_home(),
        "KILIX_DATA_HOME": data_home(),
    }


def resolve(template: str, override: str | None = None) -> str:
    """Expand a `store.default` template, or return `override` if it is set.

    `override` is the value of the model's own environment variable, which wins
    outright: a machine with the weights already on a second disk should not
    have to reproduce the directory layout to say so.
    """
    if override:
        return _expand(override)
    resolved = template
    for name, value in variables().items():
        resolved = resolved.replace("$" + name, value)
    if "$" in resolved:
        unknown = resolved[resolved.index("$"):].split("/", 1)[0]
        raise PathError(
            f"unknown store variable {unknown} in {template!r}: MODEL.json may "
            f"only use {', '.join('$' + name for name in _VARIABLES)}")
    return _expand(resolved)
