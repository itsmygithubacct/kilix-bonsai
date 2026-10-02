# Bonsai Image 4B

Local text-to-image with ternary and binary Gemlite transformers, a shared HQQ
4-bit text encoder, and the Flux2 VAE. Both variants are offered by the Kilix
model wizard. The sizing recommendation depends on available RAM and GPU memory.

## Setup

```sh
kilix wizard                              # choices first, grouped terms last
kilix bonsai deps bonsai-image-4b          # install the frozen runtime separately
kilix-bonsai-image bonsai-image-4b         # open the image composer
```

Model acquisition uses the selected Kilix Content catalog and current Licence
receipts. `kilix bonsai pull bonsai-image-4b --variant binary-gemlite` uses the
same receipt-backed installer; `--from DIR` imports catalog-matching supplied
files after consent. No model files are acquired by the dependency installer.

Models live in `$KILIX_CONTENT_ROOT/assets/bonsai-image-4b-{ternary,binary}-gemlite`.
Through `kilix bonsai`, that root is bound to the host's actual installation
root. Standalone tools default to `$KILIX_DATA_HOME/desktop-apps`. Legacy image
scaffold directories do not satisfy runtime admission; import them with
`kilix models install MODEL --from DIR` after reviewing the current terms.

## Runtime

`install-deps.sh` requires uv 0.12.5 and git. It installs managed Python 3.12.8
and the complete frozen CUDA dependency graph, including the exact Prism GPU
backend source revision. `scripts/install-image-runtime.sh --offline` requires
that pinned Content sources, Python, and dependencies are already cached; it
never fetches Git sources. `--check` runs the guarded default ternary doctor.

The bundled runtime requires an NVIDIA GPU with compute capability 7.0 or newer.
It checks current receipts and every catalog member before creating sealed
read-only model copies. Inference uses those copies with model network access
turned off. Missing, changed, extra, or unowned files refuse startup.

Select `ternary` or `binary` in the composer's variant field. When only binary
is installed, the composer selects it initially. The CLI also offers
`tools/bonsai-image/main.py doctor --variant binary` and
`generate --variant binary -p TEXT --output FILE.png`.

Start with the 512x512, four-step preview. On cards below 8 GiB, the VAE loads
on CPU in float32 to avoid a temporary GPU allocation peak. Larger cards use
the upstream GPU VAE path. The measured sizing profiles cover only the CPU VAE
512x512 path on GPU 0; other modes and larger images remain unmeasured. Available
memory from other applications can prevent a recommendation even when a card's
nominal capacity is sufficient. Resource estimates are not release qualification.

Local execution is the default. Remote execution requires explicit selection
and a configured external CLI (`KILIX_BONSAI_IMAGE_CLI` and
`KILIX_BONSAI_IMAGE_REMOTE`). Reference images require an external runtime with
that capability; the bundled pipeline supports text-to-image.

Upstream: <https://huggingface.co/collections/prism-ml/bonsai-image>.
Apple Silicon MLX conversions are not carried here.
