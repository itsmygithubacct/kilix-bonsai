# kilix-bonsai

Every BitNet model this stack can run, in one place: one folder per model, each
self-describing, each with its own dependency and download scripts, and one
terminal UI over all of them.

```sh
kilix bonsai                       # or: kilix-bonsai
```

Opening it on a machine with nothing downloaded is the normal case. The UI
starts on a setup screen, shows what each model costs, and offers to fetch it —
it never opens on an error about a directory that was always going to be empty.

## The models

| Model | Task | Quantization | Download |
|---|---|---|---|
| **Bonsai 8B** | text | `Q1_0`, 1-bit | 1.16 GB |
| **BitNet b1.58 2B4T** | text | `I2_S`, 1.58-bit ternary | 1.20 GB |
| **VibeVoice ASR BitNet** | speech-to-text | `I2_S` + `Q6_K` decoder, `I8_S` tokenizer | 1.71 GB |
| **Bonsai 27B** | text | `Q1_0`, 1-bit | 3.80 GB |
| **Bonsai Image 4B** | image | ternary `int2` / binary `int1` | 4.55 GB |

Each has a `README.md` in its folder covering what runs it and what it needs.
Several have larger optional variants — a vision projector for 27B, the binary
image weights, the unquantized sources for re-conversion — listed there and
selectable in the UI.

## Two models are shared, not owned

Three of the five stores are this repository's own. Two are not, on purpose:

- **VibeVoice ASR BitNet is the dictation model.** `kilix-voice` resolves
  speech models by catalog id, and `vibevoice-asr-bitnet` resolves to the exact
  directory this repository writes to. Downloading it here is what makes it
  available to dictation; having it for dictation makes it **ready** here.
- **Bonsai Image 4B lands in the image scaffold's data directory**, in the
  subdirectory names that scaffold already looks for.

In both cases the alternative — a private copy per component — would mean
carrying between 1.6 and 4.5 GB twice for no benefit. The paths are derived
from the same environment variables the owning components use, and the tests
assert that rather than trusting a comment to stay true.

## Commands

Everything the UI does is also a subcommand, because provisioning a machine, a
Makefile, and an SSH session all want the non-interactive path:

```sh
kilix-bonsai                       # the TUI
kilix-bonsai list                  # one line per model
kilix-bonsai status [MODEL]        # what is on disk, per variant
kilix-bonsai doctor                # downloader, free space, store permissions
kilix-bonsai deps MODEL [--apt]    # dependencies
kilix-bonsai pull MODEL [--variant V] [--from DIR] [--force] [--dry-run]
kilix-bonsai verify MODEL [--variant V]
kilix-bonsai path MODEL [--variant V]
```

`kilix` reaches the same surface: `kilix bonsai`, `kilix bonsai list`,
`kilix bonsai pull …`, and so on. It is also in the Kilix 95 Start menu under
Programs ▸ **BitNet Models**.

## A model is a folder

```
models/bonsai-27b/
├── MODEL.json          what the model is: upstream repo, pinned commit,
│                       every file with its size and sha256, where the
│                       weights belong, what the runtime needs
├── README.md           what runs it, and what it costs
├── install-deps.sh     dependencies
└── pull.sh             the download
```

`MODEL.json` is the single source of truth and every caller reads it — the two
scripts, the CLI, and the UI. So they cannot disagree about what a model is,
and **adding a model is adding a folder**: no table to update, no code to
touch. The scripts themselves are one shared implementation in `models/_shared/`
that each folder's wrapper names itself to; what differs between models is the
JSON, not the logic.

Digests and sizes in `MODEL.json` were read from the upstream API rather than
transcribed. For Bonsai 8B the recorded digest independently matches the one in
the integer engine's identity record.

## What the download does

`pull.sh` is not a `curl` loop:

- **Resumable** — an interrupted 4.5 GB transfer continues, it does not restart.
- **Verified** — every file with a published digest is sha256-checked after it
  lands. A mismatch fails; it does not warn.
- **Atomic per file** — downloads land on `.part` and are renamed only after
  they verify, so an interrupted run never leaves a file that looks complete.
- **Idempotent** — a file already present at the right size and digest costs a
  stat and a hash, so re-running after a network drop is cheap.
- **Adoptable** — `--from DIR` takes files from a copy already on the machine,
  hard-linking rather than copying when it can, and still checking every digest.

## What the dependency install does — and does not

`install-deps.sh` **checks and reports** system packages; it does not install
them unless you pass `--apt`. That is deliberate: installing distro packages
needs root, and release images pin an apt snapshot for reproducibility, so a
script that quietly ran `apt-get install` from a model UI could drift a machine
off its pinned closure. Python packages go into a virtualenv this repository
owns, one per model — never the system interpreter, and never inside a store
another component owns.

## Tests

```sh
python3 tests/run.py               # every suite, one subprocess each
python3 tests/run.py catalog       # one
```

The suites assert the properties that would be expensive to discover later: the
two shared store paths resolve to what their owning components read, every
`MODEL.json` is complete and internally consistent, `pull.sh` and the UI agree
on what is present, the UI opens on setup when the store is empty, no action
that costs bytes or touches the system runs without a confirmation, and every
screen clips cleanly down to 20×8.

## Install

```sh
./install.sh                       # kilix-bonsai into ~/.local/bin
KILIX_BONSAI_PREFIX=/usr/local ./install.sh
```

The command is a launcher that runs the tool from this checkout, so updating is
a `git pull` rather than a reinstall.

## Versioning

`0.1.0`. This is not yet one of the components the coordinated stack release
pins; when it becomes one, its version moves in lockstep with the rest and its
tag is created only by the release procedure.
