# Bonsai 27B

A 1-bit (`Q1_0`) Qwen3.5 text model, 3.80 GB on disk — the largest weight set
carried here, and still smaller than most 7B models at 4-bit.

## Get it

```sh
./install-deps.sh
./pull.sh                                # 3.80 GB, resumable, sha256-verified
./pull.sh --variant q1_0-mmproj          # 4.43 GB, adds the vision projector
kilix-bonsai verify bonsai-27b
```

Or from the TUI: `kilix bonsai`, select Bonsai 27B, Enter.

## Variants

| Variant | Size | What it adds |
|---|---|---|
| `q1_0` *(default)* | 3.80 GB | The 1-bit text model |
| `q1_0-mmproj` | 4.43 GB | The same weights plus `Bonsai-27B-mmproj-Q8_0.gguf`, the multimodal projector a runtime needs to accept images |

The projector is a separate variant rather than an always-on extra because 629
MB is a real cost on a machine that only wants text, and because a runtime
without multimodal support cannot use it at all.

## Where it lands

`$KILIX_BONSAI_MODELS_DIR/bonsai-27b`, by default
`~/.local/gpu_terminal/kilix-bonsai/models/bonsai-27b`. Override with
`KILIX_BONSAI_BONSAI_27B_DIR`.

## Running it

> **Stock llama.cpp cannot load this.** `Q1_0` is not a ggml tensor
> type, so a stock `llama-server` aborts while reading tensor info. It needs
> the vendor's own llama.cpp fork.

It loads on CPU. A CUDA runtime is much faster, but 27B at 1-bit plus a KV
cache does not comfortably share an 8 GiB card with a second float process —
budget for one at a time rather than discovering it under load.

Upstream: <https://huggingface.co/prism-ml/Bonsai-27B-gguf> · Apache-2.0

The upstream repository also publishes F16 (53.8 GB), bf16, and Q4_1
conversions. None are carried here: this repository is for the 1-bit family,
and a 53.8 GB download does not belong behind a one-key confirmation.
