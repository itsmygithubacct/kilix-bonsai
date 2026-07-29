# bonsai-cpu

Run the 1-bit Bonsai models on CPU. No GPU, no cloud, no fork — upstream
llama.cpp at a pinned commit, built CPU-only, driven by one small command
that knows what these checkpoints are.

```sh
./install.sh                 # bonsai-cpu into ~/.local/bin
bonsai-cpu build             # fetch the pinned runtime and compile it
bonsai-cpu run "Name three trees."
```

## Why this exists

The Bonsai checkpoints ship as GGUF `Q1_0` — a 1-bit format (1.125 bits per
weight: one sign bit each, an fp16 scale per 128) that mainline llama.cpp
only learned to run in April 2026, and that most installed builds therefore
still abort on. The vendor's own runtimes accelerate CUDA, Metal and Apple
Silicon; the CPU path is upstream's, and it is good: the AVX2 sign-select
kernel makes an 8B model answer at reading speed on a laptop from 2016.

This repository pins a llama.cpp commit that runs these models well, builds
it CPU-only, and wraps it with defaults measured for them — nothing more.
The interesting engineering lives upstream, where it belongs; what is here
is the part that has to be exactly right: which commit, which flags, which
thread count, which model file, verified against which digest.

Measured on an i7-6820HQ (4 cores, dual-channel DDR4-2133), Bonsai-8B:

| threads | prompt t/s | generation t/s |
|--:|--:|--:|
| 2 | 5.7 | 5.0 |
| **4 (default)** | **10.5** | **8.1** |
| 8 (hyperthreads) | 9.3 | 6.5 |

Generation at 1.125 bits/weight is memory-bandwidth-bound, which is why
hyperthreading *loses* throughput: the default is physical cores, counted,
not `nproc`.

## The models

| Model | Status |
|---|---|
| **Bonsai 8B** — Qwen3 dense, 1.16 GB | runs |
| **Bonsai 27B** — Qwen3.5 hybrid-attention, 3.8 GB | phase 2: graph not yet verified under the pinned runtime |

The commands that *execute* a model (`run`, `chat`, `serve`, `bench`)
refuse the 27B until its graph is verified — `--model` included, because an
unverified graph does not become verified by naming its file explicitly.
The informational commands (`path`, `verify`) work on any model the store
holds.

Weights are resolved from the kilix-bonsai store
(`KILIX_BONSAI_MODELS_DIR`, per-model `KILIX_BONSAI_BONSAI_8B_DIR`, default
`~/.local/gpu_terminal/kilix-bonsai/models/`), or passed explicitly with
`--model`. **This repository never downloads weights** — the store owns
acquisition, resumability and verification; `kilix-bonsai pull bonsai-8b`
is the answer to a missing model, and every refusal here says so.

## Commands

```sh
bonsai-cpu run [-n N] [--greedy] PROMPT   # answer once and exit
bonsai-cpu chat                           # interactive, model's own template
bonsai-cpu serve [--port 8188]            # OpenAI-compatible llama-server
bonsai-cpu bench [-- FLAGS]               # llama-bench; its flags after --
                                          #   e.g. bench -- -p 512 -n 128
bonsai-cpu verify                         # sha256 against the pinned digest
bonsai-cpu path                           # where the model resolved to
bonsai-cpu doctor                         # what is present, what is missing
bonsai-cpu build                          # get + build the pinned runtime
```

Sampling defaults are Qwen3's recommended (temp 0.6, top-p 0.95, top-k 20);
`--greedy` for reproducible spot checks. `run` uses the runtime's
`--single-turn` mode deliberately: the interactive UI treats a prompt as
turn one of a conversation and waits for a second, which is wrong for
scripts.

## The runtime pin

`runtime.pin` is the single definition of the runtime: upstream URL, exact
commit, and the sha256 of any patch applied on top (none at present — the
machinery exists because a repacked-GEMM prompt-processing kernel is the
known next gap, and it will arrive as a patch here before it is an upstream
PR). `scripts/get-runtime.sh` obeys that file and nothing else; the
checkout and build land under `~/.local/gpu_terminal/bonsai-cpu/runtime`
(override: `BONSAI_CPU_RUNTIME_DIR`), outside this repository, because a
compiled binary is per-machine.

A local llama.cpp clone can seed the checkout without re-downloading:
`BONSAI_CPU_LLAMA_MIRROR=/path/to/llama.cpp bonsai-cpu build`.

The shell scripts require bash 4.4+ and say so up front — bash ≤ 4.3
mishandles empty arrays under `set -u`, and a version check that fails
loudly beats a clone that dies halfway with an "unbound variable".

## Tests

```sh
python3 tests/run.py          # every suite
python3 tests/run.py cli      # one
```

The suites cover what would be expensive to discover later: the pin file
and `patches/` cannot disagree, resolution order (per-model env var over
store root over default; `--model` over all), refusals that name the
command that fixes them, a truncated download diagnosed as such, and no
absolute home path anywhere under version control.

## Versioning

`0.1.0`. Not yet part of any coordinated stack release; local repository
for now.
