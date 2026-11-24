"""bonsai-enlarge: enlarge a thumbnail, or a detail of an image, with Bonsai Image.

The installed Bonsai pipeline draws from text alone. Enlarging needs the
picture as well, so this does image-to-image in the way flow-matching models
allow it (SDEdit):

1. cut the detail out of the source, if one is asked for, and resample it to
   the output size with Lanczos, which is the plain enlargement;
2. encode that with the model's own VAE;
3. move it part of the way back towards noise, as far as ``strength`` says;
4. run only the remaining denoising steps, guided by the prompt.

A low strength keeps the picture and redraws its edges and texture; a high
strength lets the model repaint it. This module holds the parts that need no
GPU: sizes, crops, the schedule and the record written beside each result.
The forward itself is in ``enlarge_forward``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

SIZE_STEP = 32
MIN_SIDE = 128
MAX_SIDE = 2048
# 512 is the fast preview the image runtime measured on 6 GiB cards; anything
# larger is allowed but unmeasured.
DEFAULT_LONG_SIDE = 512
DEFAULT_STRENGTH = 0.5
DEFAULT_STEPS = 8
DEFAULT_PROMPT = ("a sharp, clean, high resolution version of this picture, "
                  "crisp edges, smooth flat colours, no blur, no pixelation")
SIDECAR_SCHEMA = "kilix.bonsai.enlarge/v1"


class EnlargeError(ValueError):
    """A request that cannot be run as given."""


def _snap(value: float) -> int:
    """Round to the nearest multiple of SIZE_STEP inside the allowed range."""
    snapped = int(round(value / SIZE_STEP)) * SIZE_STEP
    return max(MIN_SIDE, min(MAX_SIDE, snapped))


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError as error:
        raise EnlargeError("size must look like 512x512") from error
    for name, value in (("width", width), ("height", height)):
        if value % SIZE_STEP or not MIN_SIDE <= value <= MAX_SIDE:
            raise EnlargeError(f"{name} must be a multiple of {SIZE_STEP} "
                               f"between {MIN_SIDE} and {MAX_SIDE}")
    return width, height


def parse_crop(text: str) -> tuple[int, int, int, int]:
    """``X,Y,W,H`` in source pixels."""
    try:
        x, y, w, h = (int(part) for part in text.split(","))
    except ValueError as error:
        raise EnlargeError("crop must look like X,Y,W,H") from error
    if x < 0 or y < 0 or w < 1 or h < 1:
        raise EnlargeError("crop needs X,Y >= 0 and W,H >= 1")
    return x, y, w, h


def check_crop(crop: tuple[int, int, int, int] | None,
               source: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the box to use; the whole source when no crop is given."""
    width, height = source
    if crop is None:
        return 0, 0, width, height
    x, y, w, h = crop
    if x + w > width or y + h > height:
        raise EnlargeError(f"crop {x},{y},{w},{h} lies outside the "
                           f"{width}x{height} source")
    return crop


def target_size(region: tuple[int, int], *, scale: float | None = None,
                size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Output size for a region of ``region`` pixels.

    ``size`` is taken exactly. ``scale`` multiplies the region. With neither,
    the long side becomes DEFAULT_LONG_SIDE. The aspect ratio is kept and both
    sides are snapped to the multiple of 32 the transformer needs.
    """
    if size is not None and scale is not None:
        raise EnlargeError("give a size or a scale, not both")
    if size is not None:
        return size
    w, h = region
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            raise EnlargeError("scale must be a positive number")
        return _snap(w * scale), _snap(h * scale)
    factor = DEFAULT_LONG_SIDE / max(w, h)
    return _snap(w * factor), _snap(h * factor)


def schedule_start(steps: int, strength: float) -> int:
    """Index of the first denoising step that runs.

    Same rule as the diffusers image-to-image pipelines: of ``steps`` steps,
    ``round(steps * strength)`` run, and at least one always does.
    """
    if steps < 1:
        raise EnlargeError("steps must be at least 1")
    if not 0.0 < strength <= 1.0:
        raise EnlargeError("strength must be above 0 and at most 1")
    run = max(1, min(steps, int(round(steps * strength))))
    return steps - run


def flatten(image, background: tuple[int, int, int] = (255, 255, 255)):
    """RGB copy of a PIL image, with any transparency laid on ``background``.

    Emoji and symbols usually come with an alpha channel; the VAE takes RGB.
    """
    from PIL import Image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and
                                        "transparency" in image.info):
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(base, rgba).convert("RGB")
    return image.convert("RGB")


def prepare(image, crop: tuple[int, int, int, int] | None,
            size: tuple[int, int],
            background: tuple[int, int, int] = (255, 255, 255)):
    """Cut, flatten and Lanczos-resample: the plain enlargement."""
    from PIL import Image
    x, y, w, h = check_crop(crop, image.size)
    region = flatten(image, background).crop((x, y, x + w, y + h))
    return region.resize(size, Image.Resampling.LANCZOS)


def parse_colour(text: str) -> tuple[int, int, int]:
    value = text.lstrip("#")
    if len(value) != 6:
        raise EnlargeError("background must look like ffffff")
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as error:
        raise EnlargeError("background must look like ffffff") from error


@dataclass
class Record:
    """What produced one enlargement, written as ``OUTPUT.json``."""

    source: str
    crop: list[int]
    width: int
    height: int
    prompt: str
    strength: float
    steps: int
    steps_run: int
    seed: int
    variant: str
    background: str
    baseline: str | None = None
    seconds: float = 0.0
    schema: str = field(default=SIDECAR_SCHEMA)

    def write(self, output: Path) -> Path:
        path = output.with_suffix(output.suffix + ".json")
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path
