"""Conversations that survive the terminal.

One JSON document per conversation. The filename is a stable id minted at
creation and never changes; the human-readable name lives inside the file,
so a rename is an edit, not a move, and nothing external ever dangles.
Saves are atomic (tmp + os.replace) because a conversation is worth more
than the turn that crashed while writing it. Reading creates nothing — a
listing of an empty store must be safe from a render loop.
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field

from . import models

NAME_LIMIT = 48

DEFAULT_SYSTEM = "You are Bonsai, a concise and careful assistant."


class ConvoError(Exception):
    """A conversation file that could not be read or written."""


def chats_dir() -> str:
    return os.environ.get(
        "BONSAI_CPU_CHATS_DIR",
        os.path.join(models.GPU_TERMINAL_HOME, "bonsai-cpu", "chats"))


def default_params() -> dict:
    return {"system": DEFAULT_SYSTEM,
            **models.SAMPLING_DEFAULTS,
            "n_predict": 1024}


def default_name(first_message: str) -> str:
    name = " ".join(first_message.split())
    return name[:NAME_LIMIT] if name else "untitled"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass
class Conversation:
    path: str
    name: str = ""
    model_id: str = "bonsai-8b"
    think: bool = False
    params: dict = field(default_factory=default_params)
    messages: list = field(default_factory=list)
    created: str = ""
    updated: str = ""
    extra: dict = field(default_factory=dict)   # unknown keys, kept verbatim

    def as_document(self) -> dict:
        document = {
            "version": 1,
            "name": self.name,
            "created": self.created,
            "updated": self.updated,
            "model_id": self.model_id,
            "think": self.think,
            "params": self.params,
            "messages": self.messages,
        }
        # Unknown keys ride along so an older build cannot silently strip
        # what a newer one wrote.
        for key, value in self.extra.items():
            document.setdefault(key, value)
        return document


def new(model_id: str, *, directory: str | None = None) -> Conversation:
    directory = directory or chats_dir()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(directory, f"c-{stamp}.json")
    suffix = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"c-{stamp}-{suffix}.json")
        suffix += 1
    return Conversation(path=path, model_id=model_id,
                        created=_now(), updated=_now())


def save(convo: Conversation) -> None:
    convo.updated = _now()
    directory = os.path.dirname(convo.path)
    os.makedirs(directory, exist_ok=True)
    scratch = convo.path + ".tmp"
    try:
        with open(scratch, "w") as fh:
            json.dump(convo.as_document(), fh, indent=1)
            fh.write("\n")
        os.replace(scratch, convo.path)
    except OSError as error:
        raise ConvoError(f"could not save {convo.path}: {error}") from error


_KNOWN = {"version", "name", "created", "updated", "model_id", "think",
          "params", "messages"}


def load(path: str) -> Conversation:
    try:
        with open(path) as fh:
            document = json.load(fh)
    except (OSError, json.JSONDecodeError) as error:
        raise ConvoError(f"could not read {path}: {error}") from error
    if not isinstance(document, dict) or not isinstance(
            document.get("messages", None), list):
        raise ConvoError(f"{path} is not a conversation")
    params = default_params()
    params.update(document.get("params") or {})
    return Conversation(
        path=path,
        name=str(document.get("name", "")),
        model_id=str(document.get("model_id", "bonsai-8b")),
        think=bool(document.get("think", False)),
        params=params,
        messages=document["messages"],
        created=str(document.get("created", "")),
        updated=str(document.get("updated", "")),
        extra={key: value for key, value in document.items()
               if key not in _KNOWN})


def listing(directory: str | None = None) -> list[Conversation]:
    """Every readable conversation, newest first. Corrupt files are skipped,
    not fatal: one bad write must not hide every good conversation."""
    directory = directory or chats_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    found = []
    for name in sorted(names):
        if not (name.startswith("c-") and name.endswith(".json")):
            continue
        try:
            found.append(load(os.path.join(directory, name)))
        except ConvoError:
            continue
    found.sort(key=lambda convo: convo.updated, reverse=True)
    return found
