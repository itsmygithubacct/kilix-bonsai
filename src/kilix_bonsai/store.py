"""What is actually on disk, and how sure we are about it.

Two questions, deliberately answered by two functions with very different
costs. `state()` stats files and is safe to call from a render loop; `verify()`
hashes gigabytes and is not. A TUI that hashed a 3.8 GB checkpoint every frame
would be unusable, and one that claimed a truncated download was fine because
the path existed would be worse — so the cheap answer names its own limits and
`PRESENT` never means "hash-checked".
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable

from .catalog import File, Model, Variant

MISSING = "missing"
PARTIAL = "partial"
PRESENT = "present"

DEPS_STAMP = ".kilix-bonsai-deps.json"

# Read in chunks rather than whole: these files are larger than the memory of
# the smallest machine in the stack.
_CHUNK = 1024 * 1024


def human_bytes(count: float) -> str:
    """Format a byte count the way every other Kilix tool formats one."""
    step = 1024.0
    for unit in ("B", "K", "M", "G", "T"):
        if abs(count) < step or unit == "T":
            if unit == "B":
                return f"{int(count)}{unit}"
            return f"{count:.1f}{unit}"
        count /= step
    return f"{count:.1f}T"


@dataclass(frozen=True)
class FileState:
    file: File
    path: str
    exists: bool
    size: int

    @property
    def complete(self) -> bool:
        """True when the file is present at exactly its published size.

        Size equality is not integrity — it is the strongest claim a stat can
        support, and it does catch the failure that actually happens, which is
        an interrupted download.
        """
        return self.exists and self.size == self.file.size


@dataclass(frozen=True)
class VariantState:
    variant: Variant
    directory: str
    files: tuple[FileState, ...]

    @property
    def state(self) -> str:
        if all(entry.complete for entry in self.files):
            return PRESENT
        if any(entry.exists for entry in self.files):
            return PARTIAL
        return MISSING

    @property
    def present_bytes(self) -> int:
        return sum(min(entry.size, entry.file.size) for entry in self.files
                   if entry.exists)

    @property
    def missing(self) -> tuple[FileState, ...]:
        return tuple(entry for entry in self.files if not entry.complete)


def variant_state(model: Model, variant: Variant) -> VariantState:
    directory = variant.directory(model.store)
    entries = []
    for item in variant.files:
        path = os.path.join(directory, item.path)
        try:
            size = os.path.getsize(path)
            exists = True
        except OSError:
            size = 0
            exists = False
        entries.append(FileState(file=item, path=path, exists=exists,
                                 size=size))
    return VariantState(variant=variant, directory=directory,
                        files=tuple(entries))


def state(model: Model, variant: Variant | None = None) -> VariantState:
    """Return the on-disk state of one variant, defaulting to the default one."""
    return variant_state(model, variant or model.default_variant)


def any_present(models: Iterable[Model]) -> bool:
    """True when at least one model has a complete variant on disk.

    This is what decides whether the TUI opens on its normal list or on the
    first-run setup screen, so it asks about *every* variant: a machine that
    downloaded only the binary image weights has models, and should not be told
    it has none.
    """
    for model in models:
        for variant in model.variants:
            if variant_state(model, variant).state == PRESENT:
                return True
    return False


def deps_state(model: Model) -> dict | None:
    """Return what `install-deps.sh` recorded for this model, or None.

    The stamp is written by the script rather than inferred here, because only
    the script knows whether the packages it found were the ones it needed.
    """
    try:
        with open(os.path.join(model.store, DEPS_STAMP), encoding="utf-8") as f:
            document = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def sha256(path: str, progress: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            if progress is not None:
                progress(len(chunk))
    return digest.hexdigest()


def verify(model: Model, variant: Variant,
           report: Callable[[str, bool, str], None] | None = None) -> bool:
    """Hash every file of a variant against its published digest.

    Slow and deliberate: this is the answer `state()` cannot give. Files with
    no published digest — small text files HuggingFace does not store in LFS —
    are checked for size only, and `report` is told which check ran so a caller
    never presents a size check as a hash check.
    """
    ok = True
    for entry in variant_state(model, variant).files:
        if not entry.exists:
            ok = False
            if report:
                report(entry.file.path, False, "not downloaded")
            continue
        if entry.file.sha256:
            actual = sha256(entry.path)
            good = actual == entry.file.sha256
            detail = "sha256 ok" if good else f"sha256 mismatch ({actual[:16]}…)"
        else:
            good = entry.size == entry.file.size
            detail = "size ok, no published digest" if good else "wrong size"
        ok = ok and good
        if report:
            report(entry.file.path, good, detail)
    return ok
