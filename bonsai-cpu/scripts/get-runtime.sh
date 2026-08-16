#!/usr/bin/env bash
# Fetch the pinned inference runtime and apply this repository's patches.
#
# The checkout lands outside the source tree because it is per-machine build
# input, not part of this repository. Everything about *what* is fetched lives
# in runtime.pin; this script only obeys it.
set -euo pipefail

# bash ≤ 4.3 mishandles empty arrays under `set -u`; refuse rather than
# fail strangely partway through.
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'bonsai-cpu: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GPU_TERMINAL_HOME="${GPU_TERMINAL_HOME:-$HOME/.local/gpu_terminal}"
RUNTIME_DIR="${BONSAI_CPU_RUNTIME_DIR:-$GPU_TERMINAL_HOME/bonsai-cpu/runtime}"
SRC_DIR="$RUNTIME_DIR/llama.cpp"
PIN="$REPO_DIR/runtime.pin"

die() { printf 'bonsai-cpu: %s\n' "$*" >&2; exit 1; }
log() { printf 'bonsai-cpu: %s\n' "$*" >&2; }

[ -f "$PIN" ] || die "missing $PIN"
url="$(sed -n 's/^url=//p' "$PIN")"
commit="$(sed -n 's/^commit=//p' "$PIN")"
if [ -z "$url" ] || [ -z "$commit" ]; then
    die "runtime.pin lacks url= or commit="
fi

command -v git >/dev/null 2>&1 || die "git is required"

umask 077
mkdir -p -- "$RUNTIME_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
    # A local mirror avoids re-downloading when one is already on the machine.
    ref_args=()
    if [ -n "${BONSAI_CPU_LLAMA_MIRROR:-}" ] && [ -d "$BONSAI_CPU_LLAMA_MIRROR/.git" ]; then
        ref_args=(--reference-if-able "$BONSAI_CPU_LLAMA_MIRROR" --dissociate)
    fi
    log "cloning $url"
    git clone "${ref_args[@]}" -- "$url" "$SRC_DIR"
fi

cd "$SRC_DIR"
git fetch -q origin "$commit" 2>/dev/null || git fetch -q origin
git rev-parse --verify -q "$commit^{commit}" >/dev/null \
    || die "pinned commit $commit not present after fetch"
git checkout -q --detach "$commit"
# A pristine tree, so patch application is deterministic.
git reset -q --hard "$commit"
git clean -qfd

applied=0
while read -r patch sum; do
    case "$patch" in ''|'#'*) continue ;; esac
    file="$REPO_DIR/patches/$patch"
    [ -f "$file" ] || die "runtime.pin names missing patch $patch"
    echo "$sum  $file" | sha256sum -c --quiet - \
        || die "patch $patch does not match the sha256 in runtime.pin"
    git apply --index -- "$file" || die "patch $patch failed to apply"
    applied=$((applied + 1))
done < <(grep -v '^url=\|^commit=' "$PIN")

log "runtime source at $SRC_DIR ($commit, $applied patch(es))"
