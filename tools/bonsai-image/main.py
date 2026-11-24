#!/usr/bin/env python3
"""Run the image CLI in its isolated dependency environment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from kilix_bonsai.paths import venv_dir

python = Path(venv_dir('bonsai-image-4b')) / 'bin/python'
if Path(sys.prefix) != python.parent.parent:
    if not python.exists():
        flag = os.environ.get('KILIX_MODEL_AUTO_INSTALL')
        interactive = (flag is None or flag.strip().casefold() not in {'0', 'no', 'false', 'off'})
        try:
            interactive = interactive and sys.stdin.isatty() and sys.stdout.isatty()
        except (AttributeError, OSError, ValueError):
            interactive = False
        script = ROOT / 'models/bonsai-image-4b/install-deps.sh'
        if interactive and script.is_file():
            print('Installing Bonsai image dependencies...', flush=True)
            subprocess.call([str(script)])
        if not python.exists():
            print(f"result: unavailable: install dependencies with {script}")
            sys.exit(1)
    os.execv(str(python), [str(python), '-I', '-B', str(Path(__file__).resolve()), *sys.argv[1:]])
from kilix_bonsai.runtime.local_image import main
sys.exit(main())
