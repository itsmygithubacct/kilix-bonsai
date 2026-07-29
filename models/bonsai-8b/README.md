# Bonsai 8B

A 1-bit (`Q1_0`) Qwen3 text model, 8.19B parameters, 1.16 GB on disk.

This is the checkpoint the deterministic integer work was built around: with an
all-integer front end, the same prompt and seed produce the *same bytes* on any
machine, which is a property floating-point runtimes structurally cannot offer.
That front end is not part of this repository — what is here is the weight set,
pinned and verified.

## Get it

```sh
./install-deps.sh          # reports missing system packages, builds the venv
./pull.sh                  # 1.16 GB, resumable, sha256-verified
kilix-bonsai verify bonsai-8b
```

Or from the TUI: `kilix bonsai`, select Bonsai 8B, Enter.

## Where it lands

`$KILIX_BONSAI_MODELS_DIR/bonsai-8b`, by default
`~/.local/gpu_terminal/kilix-bonsai/models/bonsai-8b`. Set
`KILIX_BONSAI_BONSAI_8B_DIR` to put it somewhere else — a second disk, or a
copy the machine already has.

## Running it

> **This does not run under llama.cpp.** `Q1_0` is not a ggml tensor type, so
> a stock `llama-server` aborts while reading tensor info. These checkpoints
> are run by a deterministic integer engine that first *imports* the GGUF into
> a safetensors artifact and then executes it with its own kernel — a
> published, open project. A vendor CUDA runtime is an optional accelerator,
> not a requirement; the CPU path is the canonical one.

The GGUF loads in that fork's runtime; this repository does not build
one, and deliberately does not pretend to pick one for you. The Python packages
`install-deps.sh` puts in the model's virtualenv (`numpy`, `safetensors`,
`ecdsa`) are what a deterministic integer front end and its receipt signing
need on top of a runtime, not what the GGUF needs to load.

## Provenance

Everything in `MODEL.json` — the repository, the pinned commit, and the sha256
of each file — was read from the upstream API, not transcribed. The digest of
`Bonsai-8B-Q1_0.gguf` matches the one recorded independently in the integer
engine's identity record, which is the closest thing to a second source
available for this artifact.

Upstream: <https://huggingface.co/prism-ml/Bonsai-8B-gguf> · Apache-2.0

An MLX 1-bit conversion of the same model exists upstream for Apple Silicon.
It is not carried here, because nothing in this stack runs MLX.
