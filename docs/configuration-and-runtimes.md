# Configuration and runtimes

Reference for [kilix-bonsai](../README.md).

## Configuration

Machine facts are environment variables rather than constants, because this
repository is published and a hostname or a home directory baked into it would
ship with it.

| Variable | For |
|---|---|
| `KILIX_BONSAI_CHAT_REMOTE` | Host to borrow a GPU from when this one has none |
| `KILIX_BONSAI_CHAT_LAUNCHER_DIR` | Directory holding the vendor launchers on that host |
| `KILIX_BONSAI_NOTARY_HOME` | Vendor runtime's state root (regular 8B) |
| `KILIX_BONSAI_PRISM_BIN` | Pinned vendor binary directory (regular 8B) |
| `KILIX_BONSAI_8B_GGUF` | Path to the 8B weights on the running host |
| `BONSAI_CPU_PREFIX` | Bundled CPU tool's install prefix when it is not on PATH (default `~/.local`) |
| `KILIX_BONSAI_IMAGE_REMOTE` | Image scaffold's remote subcommand |
| `KILIX_TUI_UTILS_HOME` | Where to find the shared pixel renderer, if not alongside |

Only the first two are needed for 27B; 8B needs the middle three as well,
because it has no wrapper of its own and the shared CLI has to be invoked
directly. Each is reported by name when it is missing, so nothing has to be
guessed from a stack trace.

## Runtimes

Nothing here reimplements inference. On a GPU, chat drives the pinned vendor
launchers, one process per turn. On CPU it delegates to the bundled
`bonsai-cpu/` component, whose pinned upstream `llama-server` remains resident
behind the full chat TUI.
Speech drives VibeASR's `asr_infer`. Images default to the bundled local CUDA
CLI; install its isolated dependencies with
`models/bonsai-image-4b/install-deps.sh`. Remote execution requires explicit
selection and an external CLI. Missing local dependencies are reported in the pane.

Three findings worth recording, all discovered by running the things rather than
reading about them:

- **`asr_stream_server` is unusable in the build available here.** It aborts
  inside `vae_encode_impl` on every clip, including ones `asr_infer`
  transcribes without complaint. The resident-server design would be better;
  paying a model load per utterance is the cost of using the binary that works.
- **The first stock llama.cpp tested here could not load the Bonsai GGUFs.**
  `Q1_0` was not a ggml tensor type at that revision, so the original working
  path used the vendor engine; a separate deterministic integer engine can
  still import the GGUF and produce byte-identical receipts. Upstream
  llama.cpp learned `Q1_0` in April 2026, and `bonsai-cpu/` now pins and owns
  that direct CPU path. The BitNet `I2_S` checkpoint is a separate case again
  and wants `bitnet.cpp`. The standalone `scripts/build-runtime.sh` is
  therefore a reference build, not a chat backend selected automatically.
- **The image model cannot run on every host that can hold it.** Compute
  capability 7.0 is required and older cards fail at the first kernel launch,
  well after a long model load — so the backend is chosen from what `doctor`
  reports, and a remote is a first-class option rather than a fallback.

Chat probes in that order too: a local card, then a remote one, then **CPU**.
The CPU choice is a delegation, not another per-turn backend in this TUI:
curses exits and `bonsai-cpu chat` takes over with its resident server, saved
conversations, model picker, and reasoning controls. 8B stays the default;
27B is available when explicitly requested, with its minutes-per-response cost
stated rather than hidden. BitNet 2B4T is refused instead of substituted:
its declared engine is `bitnet.cpp`, and no chat adapter here drives it yet.

`kilix` reaches the same surface: `kilix bonsai`, `kilix bonsai list`,
`kilix bonsai pull …`, and so on. It is also in the Kilix 95 Start menu under
Programs ▸ **BitNet Models**.
