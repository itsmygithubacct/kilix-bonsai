# kilix-bonsai

Every BitNet model this stack can run, in one place: one folder per model, each
self-describing, each with its own dependency and download scripts, one
terminal UI over all of them, and the bundled `bonsai-cpu` runtime for direct
CPU inference.

**Status:** ships with Kilix in Plebian-OS 0.2.3 as an install-on-first-use
component. Kilix pins this repository by commit in
`scripts/install-kilix-bonsai.sh`; the first `kilix bonsai` run installs it,
and the Kilix content catalog's **Model Store** entry (`kilix-model-store`, a
system entry) opens the same command. Model weights are never installed until
you pull them.

## What it does

```sh
kilix bonsai                       # the launcher, the store, and the interfaces
```

Opening it on a machine with nothing downloaded is the normal case. The TUI
starts on setup, shows what each model costs, and offers to fetch it. It does
not open on an error about a directory that was always going to be empty. Once
something is downloaded the first runnable task is selected instead, because
by then "open a model" is what you came to do.

The no-argument launcher uses the same canonical text shell as the rest of the
Kilix utilities: `KILIX TUI`, numbered navigation, one divider, a compact
status row, application content, and a footer. `--graphics` explicitly selects
the optional Tango pixel desktop and reports a clean error when the terminal
cannot provide it. `--screenshot` always captures the text rendering.

### Interfaces and CPU runtime

| Command | For | What it does |
|---|---|---|
| `kilix-bonsai` | all of them | The launcher and the model store |
| `kilix-bonsai-chat` | the 1-bit text models | On a GPU, streams one vendor process per turn; on CPU, opens bonsai-cpu's resident, persistent chat |
| `kilix-bonsai-image` | Bonsai Image 4B | Prompt, reference image, seed, size and steps; a gallery that records what produced each result and can feed one back in |
| `kilix-bonsai-speech` | VibeVoice ASR BitNet | Record or point at a WAV; transcript grows chunk by chunk; copy or write it out |
| `bonsai-cpu` | Bonsai 8B and 27B | Pinned upstream llama.cpp, CPU-only build, resident chat, one-shot inference, serving, and benchmarks |

The launcher and all three task interfaces use that text shell by default.
Chat stays in it while it streams, as do image generation and transcription.
The optional pixel rendering remains available with `--graphics`; every text
screen renders to plain text for tests.

Which interface a model gets is declared by that model's `MODEL.json`
(`runtime.kind`), not decided here — so a new model arrives with its own
answer rather than needing the launcher changed.

### The models

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

Two of the stores are shared with other components: the VibeVoice store is
the one Kilix Voice dictation reads, and Bonsai Image 4B lives in the
receipt-backed Kilix Content store. See
[docs/models-and-downloads.md](docs/models-and-downloads.md).

## Quick start

### Install

```sh
./install.sh                       # all five commands into ~/.local/bin
KILIX_BONSAI_PREFIX=/usr/local ./install.sh
./install.sh --uninstall           # remove this checkout's unchanged launchers
```

The commands are launchers that run the tools from this checkout, including
`bonsai-cpu/bin/bonsai-cpu`, so updating is a `git pull` rather than a
reinstall. Uninstall removes only launchers whose contents still exactly match
what this checkout generated; a locally edited or replaced command is retained
and reported instead of being deleted.

On a Kilix desktop, `kilix bonsai` installs the pinned commit on first use
and accepts every subcommand below (`kilix bonsai list`, `kilix bonsai pull
MODEL`, ...). It is also in the Kilix 95 Start menu under Programs ▸ **BitNet
Models**.

## Usage

### Commands

Everything the UI does is also a subcommand, because provisioning a machine, a
Makefile, and an SSH session all want the non-interactive path:

```sh
kilix-bonsai                       # the TUI
kilix-bonsai --graphics            # require the shared pixel desktop
kilix-bonsai --text                # explicitly select the default text TUI
kilix-bonsai --screenshot FRAME    # write one text frame and exit
kilix-bonsai list                  # one line per model
kilix-bonsai status [MODEL]        # what is on disk, per variant
kilix-bonsai doctor                # downloader, free space, store permissions
kilix-bonsai deps MODEL [--apt]    # dependencies
kilix-bonsai pull MODEL [--variant V] [--from DIR] [--force] [--dry-run]
kilix-bonsai verify MODEL [--variant V]
kilix-bonsai path MODEL [--variant V]
```

## How it works

Each model is a folder under `models/` whose `MODEL.json` records the
upstream repository and commit, every file with its size and sha256, where
the weights belong and which interface runs them (`runtime.kind`). The
launcher, the CLI and the shared `models/_shared/pull.sh` and
`install-deps.sh` scripts all read that file, so adding a model is adding a
folder. Downloads are resumable, verified, atomic per file and idempotent.

```
kilix-bonsai (launcher, store, CLI)
   |-- reads models/<id>/MODEL.json
   |-- pull.sh / install-deps.sh  (shared implementation in models/_shared/)
   `-- opens the interface MODEL.json names:
         kilix-bonsai-chat   -> GPU vendor launchers, or bonsai-cpu (llama.cpp, CPU)
         kilix-bonsai-image  -> local image runtime (isolated venv) or a remote
         kilix-bonsai-speech -> VibeASR asr_infer
```

Nothing here reimplements inference. Chat probes a local GPU, then a remote
one, then hands over to `bonsai-cpu chat`. Environment variables, the runtime
choices and what was learned running them are in
[docs/configuration-and-runtimes.md](docs/configuration-and-runtimes.md).

## Tests

```sh
python3 tests/run.py               # every suite, one subprocess each
python3 tests/run.py catalog       # one
```

The suites assert the properties that would be expensive to discover later: the
two shared store paths resolve to what their owning components read, every
`MODEL.json` is complete and internally consistent, `pull.sh` and the UI agree
on what is present, both launchers expose every store action, the pixel
renderer keeps the Bonsai section table regardless of import order, the UI
opens on setup when the store is empty, no action that costs bytes or touches
the system runs without a confirmation, and every screen clips cleanly down to
20×8.

## Documentation

- [docs/models-and-downloads.md](docs/models-and-downloads.md) — shared stores, the model folder layout, what `pull.sh` and `install-deps.sh` do
- [docs/configuration-and-runtimes.md](docs/configuration-and-runtimes.md) — environment variables, GPU/CPU/remote runtimes, recorded findings
- [bonsai-cpu/README.md](bonsai-cpu/README.md) — the bundled CPU runtime on pinned upstream llama.cpp
- [models/bonsai-8b/README.md](models/bonsai-8b/README.md), [models/bonsai-27b/README.md](models/bonsai-27b/README.md), [models/bitnet-b1.58-2b4t/README.md](models/bitnet-b1.58-2b4t/README.md), [models/vibevoice-asr-bitnet/README.md](models/vibevoice-asr-bitnet/README.md), [models/bonsai-image-4b/README.md](models/bonsai-image-4b/README.md) — one guide per model

## Related repositories

- [kilix](https://github.com/itsmygithubacct/kilix#readme) — pins and installs this repository behind `kilix bonsai`
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme) — submodule at `third_party/kilix-content`; the licence-receipt store for Bonsai Image and the `kilix-model-store` catalog entry
- [kilix-voice](https://github.com/itsmygithubacct/kilix-voice#readme) — dictation reads the VibeVoice store this repository fills
- [kilix-tui-utils](https://github.com/itsmygithubacct/kilix-tui-utils#readme) — provides the optional pixel desktop for `--graphics`; the text shell here is a small local copy of its `kilix_tui.shell`
- [kilix-95](https://github.com/itsmygithubacct/kilix-95#readme) — Start menu entry Programs ▸ BitNet Models

## Part of Plebian-OS

kilix-bonsai is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
Kilix installs it at a pinned commit the first time `kilix bonsai` or the
Model Store runs. The map of every repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## Versioning

`0.2.0`. Kilix Bonsai is in the coordinated Plebian-OS 0.2.0 version set and is
selected by an exact commit in Kilix. Tags and release artifacts are still
created only by the release procedure.

## Licence

GPL-3.0. See [LICENSE](LICENSE). The bundled `bonsai-cpu/` component is MIT;
see [bonsai-cpu/LICENSE](bonsai-cpu/LICENSE).
