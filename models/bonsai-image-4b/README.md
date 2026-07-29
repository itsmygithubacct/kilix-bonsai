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

The CUDA stack these weights need — gemlite kernels, HQQ, a matching Triton and
torch — is large and version-sensitive, so `install-deps.sh` deliberately does
**not** pin it a second time. Resolve it from the image scaffold's own lock,
which is the copy that is actually tested against these weights.

It needs an NVIDIA GPU of compute capability 7.0 or newer. Older cards can hold
and verify the weights but cannot execute the kernels — the failure is at the
first kernel launch, well after a long model load, so check the card first.

Upstream: <https://huggingface.co/collections/prism-ml/bonsai-image>

MLX conversions of both variants exist upstream for Apple Silicon and are not
carried here.
