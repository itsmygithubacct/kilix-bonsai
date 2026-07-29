# VibeVoice ASR BitNet

Speech recognition compressed to run in real time on a CPU: 4.62 GB of FP16
weights down to 1.58 GB through heterogeneous quantization — `I8_S` for the VAE
tokenizer, `I2_S` with a `Q6_K` embedding for the LM decoder.

Real-time (RTF < 1) from three threads on commodity x86 or ARM, no GPU. English,
Chinese, French, Italian, Korean, Portuguese, Vietnamese and more.

## This is the dictation model

Kilix dictation and this repository are two front ends over **one copy** of
these weights. `kilix-voice` resolves its speech models from a catalog id, and
the id `vibevoice-asr-bitnet` resolves to exactly the directory `pull.sh` writes
to here:

```
$KILIX_DATA_HOME/voice/models/vibevoice-asr-bitnet
```

by default `~/.local/gpu_terminal/kilix/data/voice/models/vibevoice-asr-bitnet`.

So downloading it here is what makes it selectable as a dictation engine, and a
machine that already has it for dictation shows it as **ready** here with
nothing to fetch. Downloading it to this repository's own model root instead
would leave dictation looking at an empty directory and cost a second 1.6 GB —
which is why the path is derived from the same environment variables
`kilix-voice` uses, and asserted by a test rather than left to a comment.

## Get it

```sh
./install-deps.sh
./pull.sh                                  # 1.71 GB, the ready-to-use GGUFs
./pull.sh --variant safetensors            # 11.27 GB, only for re-converting
kilix-bonsai verify vibevoice-asr-bitnet
```

Or from the TUI: `kilix bonsai`, select VibeVoice ASR BitNet, Enter.

Already have a copy on this machine? Adopt it instead of downloading again —
each file is still checked against its published sha256, and files that match
are hard-linked rather than copied:

```sh
./pull.sh --from /path/to/an/existing/VibeVoice-ASR-BitNet
```

## Variants

| Variant | Size | What it is |
|---|---|---|
| `gguf` *(default)* | 1.71 GB | The two quantized GGUFs plus tokenizer and config — ready to run |
| `safetensors` | 11.27 GB | The original FP16 shards, needed only to re-run the quantization yourself |

## Running it

The GGUFs are consumed by a ggml-based ASR runtime with the fused BitNet
operators; `install-deps.sh` checks for the C toolchain and cmake that building
one needs, and does not build it for you.

Upstream: <https://huggingface.co/microsoft/VibeVoice-ASR-BitNet> · MIT
