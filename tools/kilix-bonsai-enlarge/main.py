"""kilix-bonsai-enlarge — enlarge a thumbnail, or a detail of an image, with Bonsai Image 4B.

    kilix-bonsai-enlarge emoji.png --output big.png
    kilix-bonsai-enlarge photo.jpg --crop 120,40,64,64 --scale 8 --output detail.png
    kilix-bonsai-enlarge icon.png --strength 0.35 --baseline lanczos.png --output big.png

The source is cut (``--crop``), laid on ``--background`` where it is
transparent, resampled with Lanczos to the output size, and then partly
redrawn by the model: ``--strength`` is how far (0.5 by default; lower keeps
more of the source). ``--baseline`` also writes the plain Lanczos result for
comparison. Each output gets ``OUTPUT.json`` recording what produced it.

Weights install on first use, after their licence is shown, exactly as for
kilix-bonsai-image. Generation is local, on an NVIDIA GPU with 4.5 GiB free.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import catalog, provision, store  # noqa: E402

CLI = os.path.join(ROOT, "tools", "bonsai-image", "main.py")


def _variant(argv: list[str]) -> str:
    for index, argument in enumerate(argv):
        if argument == "--variant" and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith("--variant="):
            return argument.split("=", 1)[1]
    return "ternary"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        if argv:
            os.execv(sys.executable, [sys.executable, CLI, "enlarge", "--help"])
        return 0 if argv else 2
    variant = _variant(argv)
    if variant not in ("ternary", "binary"):
        print("kilix-bonsai-enlarge: --variant must be ternary or binary",
              file=sys.stderr)
        return 2
    model = catalog.find("bonsai-image-4b")
    chosen = model.variant(f"{variant}-gemlite")
    if store.state(model, chosen).state != store.PRESENT:
        status = provision.ensure_downloaded(model, chosen)
        if store.state(model, chosen).state != store.PRESENT:
            print(f"kilix-bonsai-enlarge: {chosen.title} weights are not installed",
                  file=sys.stderr)
            return status if status not in (None, 0) else 1
    os.execv(sys.executable, [sys.executable, CLI, "enlarge", *argv])
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main())
