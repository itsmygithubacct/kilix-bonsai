# Bonsai Image 4B

A 4B diffusion image model in two quantizations: ternary (1.58-bit) at 4.55 GB
and binary (1-bit) at 4.09 GB. Ternary is the default — the quality difference
is much larger than the half-gigabyte.

Both share the same HQQ 4-bit text encoder and the same VAE; only the
transformer differs, which is why the two variants are nearly the same size
despite one being 2-bit and the other 1-bit.

## Get it

```sh
./install-deps.sh
./pull.sh                                  # ternary, 4.55 GB
./pull.sh --variant binary-gemlite         # binary,  4.09 GB
kilix-bonsai verify bonsai-image-4b
```

Or from the TUI: `kilix bonsai`, select Bonsai Image 4B, Enter.

## Where it lands

**Not** under this repository's own model root. These weights go to
`$GPU_TERMINAL_HOME/bonsai_image_generation`, the data directory the image
generation scaffold already uses as `BONSAI_MODELS_DIR`, in the subdirectories
it already looks for (`bonsai-image-4B-ternary-gemlite`,
`bonsai-image-4B-binary-gemlite`).

That is the whole point of the shared path: a machine that has already set up
image generation shows these as **ready** here with nothing to download, and a
machine that downloads them here can generate images without a second copy.
Override with `KILIX_BONSAI_BONSAI_IMAGE_4B_DIR` if the weights live elsewhere.

## Running it

Run `./install-deps.sh` to install the pinned local CUDA pipeline in an
isolated Python 3.11 environment. This requires `uv` and `git`; it does not
download model weights. `./install-deps.sh --check` reports missing dependencies,
weights, or CUDA support. Existing ternary weights are reused in place.

Local is the default and an unavailable local backend never selects remote.
Remote requires an explicit backend selection and a configured external CLI
(`KILIX_BONSAI_IMAGE_CLI` and `KILIX_BONSAI_IMAGE_REMOTE`).

The bundled runtime uses the ternary model and supports text-to-image.
Reference images require an external runtime with that capability. Start with
the 512x512 fast preset; larger images require more GPU memory. When other GPU
jobs leave too little memory, generation reports why and can be retried later.

Upstream: <https://huggingface.co/collections/prism-ml/bonsai-image>

MLX conversions of both variants exist upstream for Apple Silicon and are not
carried here.
