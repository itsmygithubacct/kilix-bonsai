# BitNet b1.58 2B4T

The reference 1.58-bit model the whole family is named after: 2B parameters
trained on 4T tokens, 1.19 GB as an `I2_S` GGUF.

It earns its place here by being the smallest and the most widely supported.
When a runtime, a build, or a machine is in question, this is the model to try
first — a failure here is a problem with the setup, not with a 3.8 GB
checkpoint.

## Get it

```sh
./install-deps.sh
./pull.sh                                  # 1.20 GB, ready to run
./pull.sh --variant safetensors            # 1.19 GB, BF16, for re-converting
kilix-bonsai verify bitnet-b1.58-2b4t
```

Or from the TUI: `kilix bonsai`, select BitNet b1.58 2B4T, Enter.

## Where it lands

`$KILIX_BONSAI_MODELS_DIR/bitnet-b1.58-2b4t`, by default
`~/.local/gpu_terminal/kilix-bonsai/models/bitnet-b1.58-2b4t`. Override with
`KILIX_BONSAI_BITNET_2B4T_DIR`.

## A note on the two upstream repositories

The GGUF and the tokenizer live in *different* upstream repositories — the
weights in `microsoft/bitnet-b1.58-2B-4T-gguf`, the tokenizer and config in
`microsoft/bitnet-b1.58-2B-4T`. The default variant pulls from both and lands
them in one directory, because a GGUF without its tokenizer is not a usable
model and making that someone's second manual step is how it gets missed.
Each repository is pinned at its own commit.

## Running it

Built to be run by `bitnet.cpp`, which compiles its ternary kernels with clang —
hence `clang` and `cmake` in this model's dependency list. It also loads in
recent llama.cpp builds with I2_S support.

Upstream: <https://huggingface.co/microsoft/bitnet-b1.58-2B-4T> · MIT
