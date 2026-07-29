"""The model registry, read from the `MODEL.json` in each model folder.

There is deliberately no catalog table in this file. A model folder is
self-describing — its `MODEL.json` names the upstream repository, the pinned
revision, every file with its size and sha256, where the weights belong, and
what the runtime needs — and both shell scripts in that folder read the same
document. So `pull.sh`, `install-deps.sh`, the CLI, and the TUI cannot disagree
about what a model is, and adding a model is adding a folder.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import paths


class CatalogError(ValueError):
    """A MODEL.json that is missing or does not describe a model."""


def models_root() -> str:
    """Return this checkout's `models/` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "models")


@dataclass(frozen=True)
class File:
    """One file to fetch: its path inside the store and what it must hash to."""

    path: str
    size: int
    sha256: str | None
    repo: str
    revision: str

    @property
    def url(self) -> str:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        return f"{endpoint}/{self.repo}/resolve/{self.revision}/{self.path}"


@dataclass(frozen=True)
class Variant:
    """One downloadable form of a model — a quantization, or a set to convert."""

    id: str
    title: str
    default: bool
    subdir: str
    files: tuple[File, ...]
    bytes: int

    def directory(self, store: str) -> str:
        return os.path.join(store, self.subdir) if self.subdir else store


@dataclass(frozen=True)
class Model:
    id: str
    title: str
    summary: str
    task: str
    parameters: str
    quantization: str
    license: str
    upstream: str
    deps: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    variants: tuple[Variant, ...] = ()
    store_env: str = ""
    store_default: str = ""
    shared_with: str | None = None
    folder: str = ""

    @property
    def store(self) -> str:
        """Return where this model's weights belong on this machine."""
        return paths.resolve(self.store_default,
                             os.environ.get(self.store_env))

    @property
    def default_variant(self) -> Variant:
        for variant in self.variants:
            if variant.default:
                return variant
        return self.variants[0]

    def variant(self, variant_id: str | None) -> Variant:
        if not variant_id:
            return self.default_variant
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        known = ", ".join(v.id for v in self.variants)
        raise CatalogError(
            f"{self.id} has no variant {variant_id!r}; it has {known}")

    def script(self, name: str) -> str:
        return os.path.join(self.folder, name)


def _variant(raw: dict) -> Variant:
    files: list[File] = []
    for source in raw.get("sources", ()):
        repo = source["repo"]
        revision = source["revision"]
        for entry in source.get("files", ()):
            files.append(File(path=entry["path"], size=int(entry["size"]),
                              sha256=entry.get("sha256"), repo=repo,
                              revision=revision))
    if not files:
        raise CatalogError(f"variant {raw.get('id')!r} lists no files")
    return Variant(
        id=raw["id"], title=raw.get("title", raw["id"]),
        default=bool(raw.get("default")), subdir=raw.get("subdir", ""),
        files=tuple(files),
        bytes=int(raw.get("bytes") or sum(f.size for f in files)))


def load_model(folder: str) -> Model:
    """Read one model folder's MODEL.json."""
    document = os.path.join(folder, "MODEL.json")
    try:
        with open(document, encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as error:
        raise CatalogError(f"{document} could not be read: {error}") from error
    except json.JSONDecodeError as error:
        raise CatalogError(f"{document} is not valid JSON: {error}") from error
    store = raw.get("store") or {}
    if not store.get("default"):
        raise CatalogError(f"{document} does not say where its weights belong")
    variants = tuple(_variant(entry) for entry in raw.get("variants", ()))
    if not variants:
        raise CatalogError(f"{document} lists no variants")
    if sum(1 for variant in variants if variant.default) > 1:
        raise CatalogError(f"{document} marks more than one variant default")
    return Model(
        id=raw["id"], title=raw["title"], summary=raw.get("summary", ""),
        task=raw.get("task", ""), parameters=raw.get("parameters", ""),
        quantization=raw.get("quantization", ""),
        license=raw.get("license", ""), upstream=raw.get("upstream", ""),
        deps=raw.get("deps") or {}, runtime=raw.get("runtime") or {},
        variants=variants,
        store_env=store.get("env", ""), store_default=store["default"],
        shared_with=store.get("shared_with"), folder=folder)


def load(root: str | None = None) -> list[Model]:
    """Read every model folder, ordered by download size, smallest first."""
    root = root or models_root()
    models = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if os.path.isfile(os.path.join(folder, "MODEL.json")):
            models.append(load_model(folder))
    models.sort(key=lambda model: model.default_variant.bytes)
    return models


def find(model_id: str, root: str | None = None) -> Model:
    for model in load(root):
        if model.id == model_id:
            return model
    known = ", ".join(model.id for model in load(root))
    raise CatalogError(f"no model {model_id!r}; this checkout has {known}")
