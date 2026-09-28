"""Offline local generation with the pinned Prism GPU pipeline."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import secrets

from kilix_bonsai import catalog


def weights() -> Path:
    model = catalog.find('bonsai-image-4b')
    return Path(model.store) / 'bonsai-image-4B-ternary-gemlite'


def check() -> None:
    for module in ('torch', 'gemlite', 'hqq', 'transformers', 'diffusers', 'backend_gpu.pipeline_gpu'):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError('image dependencies missing; run the Bonsai Image install-deps.sh: ' + str(error)) from error
    root = weights()
    for name in ('transformer-gemlite-int2/state_dict.pt', 'text_encoder-hqq-4bit/qmodel.pt',
                 'text_encoder-hqq-4bit/tokenizer/tokenizer.json', 'vae/diffusion_pytorch_model.safetensors'):
        if not (root / name).is_file():
            raise RuntimeError(f'missing image weights: {root / name}; download the ternary image model in Bonsai')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('local images require an NVIDIA GPU and a working CUDA driver')
    if torch.cuda.get_device_capability() < (7, 0):
        raise RuntimeError('local images require NVIDIA compute capability 7.0 or newer')


def generate(args: argparse.Namespace) -> None:
    if args.input_image:
        raise ValueError('reference images are not supported by the installed local pipeline; clear the reference field')
    try:
        width, height = map(int, args.size.lower().split('x'))
    except ValueError as error:
        raise ValueError('size must look like 512x512') from error
    if any(n < 128 or n > 2048 or n % 32 for n in (width, height)):
        raise ValueError('image dimensions must be multiples of 32 between 128 and 2048')
    if not args.prompt.strip() or args.steps < 1:
        raise ValueError('a prompt and at least one step are required')
    check()
    import torch
    free, total = torch.cuda.mem_get_info()
    if free < 4500 * 1024**2:
        raise RuntimeError(f'GPU busy: {free // 1024**2} MiB free; local images need at least 4500 MiB. Retry after other GPU jobs finish.')
    from backend_gpu.pipeline_gpu import GpuPipeline
    root = weights()
    pipe = GpuPipeline(backend='bonsai-ternary-gemlite',
                       binary_transformer_path=root / 'transformer-gemlite-int1',
                       ternary_transformer_path=root / 'transformer-gemlite-int2',
                       text_encoder_path=root / 'text_encoder-hqq-4bit',
                       tokenizer_path=str(root / 'text_encoder-hqq-4bit/tokenizer'),
                       vae_path=root / 'vae')
    print('Loading local image model…', flush=True)
    pipe.prewarm()
    if total < 8 * 1024**3:
        # Keep VAE decoding off small GPUs. The pinned pipeline transfers the
        # final latents to the VAE device, so this needs no kernel patching.
        pipe._vae.to('cpu')
        torch.cuda.empty_cache()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    print(f'Generating locally: {width}x{height}, seed {seed}…', flush=True)
    data = pipe.generate_png(args.prompt, seed=seed, steps=args.steps, width=width, height=height)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f'result: generated {output}', flush=True)


def main(argv: list[str] | None = None) -> int:
    # The model store owns downloads; inference must never fetch another model.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    run = commands.add_parser('generate')
    run.add_argument('-p', '--prompt', required=True)
    run.add_argument('--size', default='512x512')
    run.add_argument('--output', required=True)
    run.add_argument('--seed', type=int)
    run.add_argument('--steps', type=int, default=4)
    run.add_argument('--input-image')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            check()
            print('result: ready: local CUDA image runtime')
        else:
            generate(args)
        return 0
    except Exception as error:
        print(f'Error: {error}', flush=True)
        return 1
