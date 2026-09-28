#!/usr/bin/env python3
"""Run the image CLI in its isolated dependency environment."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from kilix_bonsai.paths import venv_dir

python = Path(venv_dir('bonsai-image-4b')) / 'bin/python'
if Path(sys.prefix) != python.parent.parent:
    if not python.exists():
        print(f"result: unavailable: install dependencies with {ROOT / 'models/bonsai-image-4b/install-deps.sh'}")
        sys.exit(1)
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
from kilix_bonsai.runtime.local_image import main
sys.exit(main())
