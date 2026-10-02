#!/usr/bin/env bash
# Install the complete frozen image dependency graph; never acquire models.
set -euo pipefail
umask 077
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
case "${1:-}" in -h|--help) echo "usage: install-deps.sh [--check|--offline]"; exit 0 ;; esac
if [ "${1:-}" = --check ]; then
  exec "$repo/tools/bonsai-image/main.py" doctor
fi
offline=()
if [ "${1:-}" = --offline ]; then offline=(--offline); shift; fi
[ "$#" = 0 ] || { echo 'usage: install-deps.sh [--check|--offline]' >&2; exit 2; }
[ "$(id -u)" != 0 ] || { echo 'Run as the desktop user, not root.' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv 0.12.5, then run this script again.' >&2; exit 1; }
command -v git >/dev/null || { echo 'git is required.' >&2; exit 1; }
[[ "$(uv --version)" =~ ^uv\ 0\.12\.5($|[[:space:]]) ]] || { echo 'This runtime requires uv 0.12.5.' >&2; exit 1; }
if [ "${#offline[@]}" != 0 ]; then
  selected="$(git -C "$repo" ls-tree HEAD third_party/kilix-content | awk '{print $3}')"
  actual="$(git -C "$repo/third_party/kilix-content" rev-parse HEAD 2>/dev/null || true)"
  if [ -z "$selected" ] || [ "$actual" != "$selected" ]; then
    echo 'Offline pinned Content source is unavailable.' >&2; exit 1
  fi
else
  git -C "$repo" submodule update --init third_party/kilix-content
fi
python3 -I -B "$repo/third_party/kilix-content/tools/vendored_kilix_license.py" --check
venv="$(PYTHONPATH="$repo/src" python3 -B -c 'from kilix_bonsai.paths import venv_dir; print(venv_dir("bonsai-image-4b"))')"
UV_PROJECT_ENVIRONMENT="$venv" uv sync --directory "$repo/runtime/image" \
  --frozen --no-config --no-dev --no-editable --no-install-project \
  --python 3.12.8 --managed-python "${offline[@]}"
echo 'Frozen local image dependencies installed. Models require kilix wizard.'
