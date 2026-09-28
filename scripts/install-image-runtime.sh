#!/usr/bin/env bash
# Install the local image runtime separately from model weights.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
case "${1:-}" in -h|--help) echo "usage: install-deps.sh [--check]"; exit 0 ;; esac
if [ "${1:-}" = --check ]; then
  exec "$repo/tools/bonsai-image/main.py" doctor
fi
[ "$#" = 0 ] || { echo 'usage: install-deps.sh [--check]' >&2; exit 2; }
[ "$(id -u)" != 0 ] || { echo 'Run as the desktop user, not root.' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv, then run this script again (Python 3.11 is managed by uv).' >&2; exit 1; }
command -v git >/dev/null || { echo 'Install git, then run this script again.' >&2; exit 1; }
mapfile -t locations < <(PYTHONPATH="$repo/src" python3 - <<'PY'
from kilix_bonsai.paths import runtime_dir, venv_dir
print(runtime_dir())
print(venv_dir('bonsai-image-4b'))
PY
)
runtime="${locations[0]}/image-studio"
venv="${locations[1]}"
revision=31b02171634c16b5da0eec6aea075e7489d5fb39
if [ ! -d "$runtime/.git" ]; then
  [ ! -e "$runtime" ] || { echo "Refusing to replace $runtime" >&2; exit 1; }
  mkdir -p -- "$(dirname -- "$runtime")"
  git clone https://github.com/PrismML-Eng/image-studio.git "$runtime"
fi
[ -z "$(git -C "$runtime" status --porcelain)" ] || { echo 'Runtime checkout has local changes.' >&2; exit 1; }
git -C "$runtime" fetch origin "$revision"
git -C "$runtime" checkout --detach "$revision"
[ -x "$venv/bin/python" ] || uv venv --python 3.11 "$venv"
uv pip install --python "$venv/bin/python" 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$venv/bin/python" \
  'torch==2.6.0' 'triton==3.2.0' 'gemlite==0.4.7' 'hqq==0.2.8.post1' \
  'diffusers==0.38.0' 'transformers==5.8.1' 'accelerate==1.13.0' \
  'setuptools==80.10.2' 'pillow>=10.4' "$runtime/backend_gpu"
echo 'Local image dependencies installed. Download the ternary weights in Bonsai if needed.'
