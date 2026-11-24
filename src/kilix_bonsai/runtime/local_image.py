"""Offline local generation with the pinned Prism GPU pipeline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import secrets
import resource
import time

from kilix_bonsai.paths import data_home
from .installed_image import InstalledModel


def weights() -> Path:
    return Path(os.environ.get('KILIX_CONTENT_ROOT') or Path(data_home())/'desktop-apps')


@contextmanager
def model_view(variant='ternary'):
    if variant not in ('ternary','binary'):
        raise ValueError('unknown image variant')
    model = InstalledModel('bonsai-image-4b-'+variant+'-gemlite',weights(),
                           maximum_bytes=6*1024**3,provider='kilix-bonsai',
                           consumer_schema='kilix.bonsai.runtime')
    deadline = time.monotonic()+180
    def check_deadline():
        if time.monotonic() >= deadline:
            raise RuntimeError('installed image verification exceeded its deadline')
    with model, model.view(check_deadline) as root:
        yield root


def check(variant='ternary') -> None:
    # Consent and catalog-bound bytes precede dependency imports or CUDA access.
    with model_view(variant):
        pass
    for module in ('torch', 'gemlite', 'hqq', 'transformers', 'diffusers', 'backend_gpu.pipeline_gpu'):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError('image dependencies missing; run the Bonsai Image install-deps.sh: ' + str(error)) from error
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('local images require an NVIDIA GPU and a working CUDA driver')
    if torch.cuda.get_device_capability() < (7, 0):
        raise RuntimeError('local images require NVIDIA compute capability 7.0 or newer')


def require_gpu_memory(torch) -> int:
    """Refuse before any model loads when other jobs hold the GPU; return its total."""
    free, total = torch.cuda.mem_get_info()
    if free < 4500 * 1024**2:
        raise RuntimeError(f'GPU busy: {free // 1024**2} MiB free; local images need at least 4500 MiB. Retry after other GPU jobs finish.')
    return total


def enlarge(args: argparse.Namespace) -> None:
    """bonsai-enlarge: image-to-image enlargement of a thumbnail or a detail."""
    from . import enlarge as plan
    from PIL import Image
    source = Path(args.input)
    if not source.is_file():
        raise ValueError(f'no input image at {source}')
    crop = plan.parse_crop(args.crop) if args.crop else None
    background = plan.parse_colour(args.background)
    with Image.open(source) as opened:
        opened.load()
        image = opened.copy()
    box = plan.check_crop(crop, image.size)
    size = plan.target_size(box[2:], scale=args.scale,
                            size=plan.parse_size(args.size) if args.size else None)
    start = plan.schedule_start(args.steps, args.strength)
    prompt = args.prompt or plan.DEFAULT_PROMPT
    if not prompt.strip():
        raise ValueError('a prompt is required')
    variant = getattr(args, 'variant', 'ternary')
    baseline = plan.prepare(image, box, size, background)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_path = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline.save(baseline_path, format='PNG')
    check(variant)
    import torch
    total = require_gpu_memory(torch)
    from backend_gpu import pipeline_gpu
    from . import enlarge_forward
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    started = time.monotonic()
    with model_view(variant) as root:
        pipe = load_pipeline(root, total, variant, pipeline_gpu, torch)
        print(f'Enlarging locally: {box[2]}x{box[3]} -> {size[0]}x{size[1]}, '
              f'{args.steps - start} of {args.steps} steps, seed {seed}…', flush=True)
        result = enlarge_forward.img2img_forward(
            pipe, baseline, prompt, num_steps=args.steps, start=start, seed=seed)
    result.save(output, format='PNG')
    plan.Record(source=str(source), crop=list(box), width=size[0], height=size[1],
                prompt=prompt, strength=args.strength, steps=args.steps,
                steps_run=args.steps - start, seed=seed, variant=variant,
                background=args.background.lstrip('#'),
                baseline=str(baseline_path) if baseline_path else None,
                seconds=round(time.monotonic() - started, 1)).write(output)
    print(f'result: enlarged {output}', flush=True)


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
    variant = getattr(args,'variant','ternary')
    check(variant)
    import torch
    total = require_gpu_memory(torch)
    from backend_gpu import pipeline_gpu
    with model_view(variant) as root:
        data = generate_verified(args,root,width,height,total,variant,pipeline_gpu,torch)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f'result: generated {output}', flush=True)


def load_pipeline(root,total,variant,pipeline_gpu,torch):
    """Prewarm the pinned GPU pipeline from a verified model view."""
    pipe = pipeline_gpu.GpuPipeline(backend='bonsai-'+variant+'-gemlite',
                       device='cuda:0',
                       binary_transformer_path=root / 'transformer-gemlite-int1',
                       ternary_transformer_path=root / 'transformer-gemlite-int2',
                       text_encoder_path=root / 'text_encoder-hqq-4bit',
                       tokenizer_path=str(root / 'text_encoder-hqq-4bit/tokenizer'),
                       vae_path=root / 'vae')
    print('Loading local image model…', flush=True)
    small_gpu = total < 8*1024**3
    original_loader = pipeline_gpu._load_vae if small_gpu else None
    if small_gpu:
        # Load directly on CPU; moving after prewarm still incurs a CUDA peak.
        def cpu_vae(path, *, device=None):
            from diffusers import AutoencoderKLFlux2
            return AutoencoderKLFlux2.from_pretrained(str(path),torch_dtype=torch.float32,
                                                      local_files_only=True).to('cpu').eval()
        pipeline_gpu._load_vae = cpu_vae
    try:
        pipe.prewarm()
    finally:
        if small_gpu:
            pipeline_gpu._load_vae = original_loader
    if small_gpu:
        # The pinned GPU pipeline supplies bf16 latents even on CPU. Convert
        # them at the decoder boundary: bf16 convolutions can take minutes on
        # CPUs without native bf16 instructions, while float32 uses fast kernels.
        decode = pipe._vae.decode

        def decode_cpu(latents, *args, **kwargs):
            return decode(latents.float(), *args, **kwargs)

        pipe._vae.decode = decode_cpu
        torch.cuda.empty_cache()
    return pipe


def generate_verified(args,root,width,height,total,variant,pipeline_gpu,torch):
    pipe = load_pipeline(root,total,variant,pipeline_gpu,torch)
    cold_allocated = torch.cuda.max_memory_allocated()
    cold_reserved = torch.cuda.max_memory_reserved()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    print(f'Generating locally: {width}x{height}, seed {seed}…', flush=True)
    data = pipe.generate_png(prompt=args.prompt, seed=seed, steps=args.steps, width=width, height=height)
    if getattr(args,'measurement',None):
        observation = {'schema':'kilix.bonsai.memory-observation/v1-development',
            'qualification_eligible':False,'variant':variant,'width':width,'height':height,
            'steps':args.steps,'seed':seed,'vae_device':'cpu' if small_gpu else 'cuda',
            'ram_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'cold_cuda_allocated_bytes':cold_allocated,'cold_cuda_reserved_bytes':cold_reserved,
            'generation_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'generation_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'method':'Per-process RSS and Torch allocator peaks; external CUDA allocations are not included.'}
        Path(args.measurement).write_text(json.dumps(observation,sort_keys=True,indent=2)+'\n')
    return data


def main(argv: list[str] | None = None) -> int:
    # The model store owns downloads; inference must never fetch another model.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor')
    doctor.add_argument('--variant',choices=('ternary','binary'),default='ternary')
    run = commands.add_parser('generate')
    run.add_argument('-p', '--prompt', required=True)
    run.add_argument('--size', default='512x512')
    run.add_argument('--output', required=True)
    run.add_argument('--seed', type=int)
    run.add_argument('--steps', type=int, default=4)
    run.add_argument('--input-image')
    run.add_argument('--variant',choices=('ternary','binary'),default='ternary')
    run.add_argument('--measurement',help='write development process/allocator memory observations')
    big = commands.add_parser('enlarge', help='enlarge a thumbnail or a detail (bonsai-enlarge)')
    big.add_argument('input', help='source image (PNG, JPEG, ...)')
    big.add_argument('--output', required=True)
    big.add_argument('--crop', help='detail to enlarge, X,Y,W,H in source pixels')
    big.add_argument('--scale', type=float, help='output = region x SCALE, snapped to 32')
    big.add_argument('--size', help='exact output WxH, multiples of 32')
    big.add_argument('-p', '--prompt', help='what the picture is; a generic sharpening prompt by default')
    big.add_argument('--strength', type=float, default=0.5,
                     help='0..1: how far the model may redraw (default 0.5)')
    big.add_argument('--steps', type=int, default=8, help='schedule length (default 8)')
    big.add_argument('--seed', type=int)
    big.add_argument('--background', default='ffffff', help='colour under transparency')
    big.add_argument('--baseline', help='also write the plain Lanczos enlargement here')
    big.add_argument('--variant', choices=('ternary', 'binary'), default='ternary')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            check(args.variant)
            print('result: ready: local CUDA image runtime')
        elif args.command == 'enlarge':
            enlarge(args)
        else:
            generate(args)
        return 0
    except Exception as error:
        print(f'Error: {error}', flush=True)
        return 1
