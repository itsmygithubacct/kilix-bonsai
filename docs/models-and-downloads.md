# Models, stores and downloads

Reference for [kilix-bonsai](../README.md).

## Two models are shared, not owned

Three of the five stores are this repository's own. Two are not, on purpose:

- **VibeVoice ASR BitNet is the dictation model.** `kilix-voice` resolves
  speech models by catalog id, and `vibevoice-asr-bitnet` resolves to the exact
  directory this repository writes to. Downloading it here is what makes it
  available to dictation; having it for dictation makes it **ready** here.
- **Bonsai Image 4B uses the receipt-backed Kilix Content store** shared with
  `kilix wizard`. The image pull command uses that installer, and the guarded
  runtime verifies its exact catalog population and current licence coverage.

The shared paths avoid a separate model copy per component. Image runtime
dependencies are installed separately with `kilix bonsai deps bonsai-image-4b`.
The image composer offers a ternary/binary variant field and uses the variant
selected in the wizard when only that variant is installed. See the
[Bonsai Image guide](../models/bonsai-image-4b/README.md) for the frozen runtime,
legacy model import, and measured preview limits.

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
