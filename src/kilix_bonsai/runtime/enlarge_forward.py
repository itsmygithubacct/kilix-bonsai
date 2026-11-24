"""Klein 4B image-to-image forward for bonsai-enlarge.

The pinned Prism forward (``backend_gpu.diffusion_klein``) is text-to-image
only; its docstring says it strips upstream img2img. This adds it back in the
flow-matching form, reusing that module's prompt encoder, shift and scheduler
so the schedule is the one the text-to-image path uses:

    x0      = bn_normalize(patchify(vae.encode(image).mode()))
    x_start = sigma[start] * noise + (1 - sigma[start]) * x0
    then steps start .. steps-1 of the ordinary denoising loop.

The VAE encode follows ``Flux2Pipeline._encode_vae_image`` (argmax latents,
2x2 patchify, batch-norm normalize). The decode mirrors ``diffusion_forward``.
Only imported inside the image runtime's environment: it needs torch.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from diffusers import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
from backend_gpu import diffusion_klein


def _param(module):
    return next(module.parameters())


def _bn(vae, like):
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like.device, torch.float32)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).float()
                     + vae.config.batch_norm_eps).to(like.device)
    return mean, std


@torch.no_grad()
def encode_image(vae, image: Image.Image) -> torch.Tensor:
    """RGB PIL image -> normalized, patchified latents (1, 128, H/16, W/16), float32."""
    weight = _param(vae)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    pixels = pixels.to(device=weight.device, dtype=weight.dtype)
    encoded = vae.encode(pixels)
    dist = getattr(encoded, "latent_dist", None)
    latents = dist.mode() if dist is not None else encoded.latents
    latents = Flux2Pipeline._patchify_latents(latents.float())
    mean, std = _bn(vae, latents)
    return (latents - mean) / std


@torch.no_grad()
def decode_latents(vae, latents_4d: torch.Tensor) -> Image.Image:
    """Inverse of encode_image: (1, 128, H/16, W/16) -> RGB PIL image."""
    weight = _param(vae)
    latents = latents_4d.to(weight.device, torch.float32)
    mean, std = _bn(vae, latents)
    latents = Flux2Pipeline._unpatchify_latents(latents * std + mean)
    image = vae.decode(latents.to(weight.dtype), return_dict=False)[0]
    img = image[0].clamp(-1.0, 1.0).float()
    img = ((img + 1.0) * 127.5).clamp(0.0, 255.0).round().to(torch.uint8)
    return Image.fromarray(img.permute(1, 2, 0).cpu().numpy(), mode="RGB")


@torch.no_grad()
def img2img_forward(pipe, init_image: Image.Image, prompt: str, *,
                    num_steps: int, start: int, seed: int,
                    max_sequence_length: int = 512,
                    guidance: float = diffusion_klein.DEFAULT_GUIDANCE) -> Image.Image:
    """Denoise ``init_image`` from step ``start`` of ``num_steps``.

    ``pipe`` is a prewarmed ``backend_gpu.pipeline_gpu.GpuPipeline``.
    ``start`` 0 is pure text-to-image from noise, except that the noise is not
    discarded: sigma[0] is 1.0, so the image contributes nothing.
    """
    transformer, vae = pipe._transformer, pipe._vae
    width, height = init_image.size
    if width % 32 or height % 32:
        raise ValueError("image sides must be multiples of 32")
    if not 0 <= start < num_steps:
        raise ValueError("start must leave at least one step to run")
    device = _param(transformer).device
    dtype = getattr(transformer, "_inference_dtype", torch.float16)

    prompt_embeds = diffusion_klein._encode_klein_qwen3_prompt(
        text_encoder=pipe._text_encoder, tokenizer=pipe._tokenizer,
        prompt=prompt, max_sequence_length=max_sequence_length)
    text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(device)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

    image_4d = encode_image(vae, init_image)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_4d = torch.randn(tuple(image_4d.shape), generator=generator,
                           dtype=torch.float32)
    latent_ids = Flux2Pipeline._prepare_latent_ids(noise_4d).to(device)
    image_seq = Flux2Pipeline._pack_latents(image_4d.cpu())
    noise_seq = Flux2Pipeline._pack_latents(noise_4d)

    scheduler = pipe._scheduler or diffusion_klein._build_default_scheduler()
    mu = diffusion_klein._mflux_empirical_mu(image_seq_len=image_seq.shape[1],
                                             num_steps=num_steps)
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if getattr(scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    timesteps, _ = retrieve_timesteps(scheduler, num_steps, device,
                                      sigmas=sigmas, mu=mu)
    sigma = float(scheduler.sigmas[start])
    latents = (sigma * noise_seq + (1.0 - sigma) * image_seq).to(device=device, dtype=dtype)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(start)

    guidance_t = torch.full([1], guidance, device=device, dtype=torch.float32)
    for t in timesteps[start:]:
        noise_pred = transformer(
            hidden_states=latents, timestep=t.expand(1).to(dtype) / 1000,
            guidance=guidance_t, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(dtype)

    latents_4d = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids)
    return decode_latents(vae, latents_4d)
