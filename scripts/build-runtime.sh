#!/usr/bin/env bash
# Build the repository's standalone/reference llama-server.
#
# The current chat interface does not use this build: GPU chat drives the
# vendor launchers, and CPU chat delegates to bonsai-cpu's pinned runtime.
# Keep this explicit builder for callers of src/kilix_bonsai/runtime/llama.py
# and for runtime experiments; it is not silently selected as a model adapter.
#
# The build lands outside the source tree, next to the weights, because a
# compiled binary is per-machine and this repository is published.
set -euo pipefail

GPU_TERMINAL_HOME="${GPU_TERMINAL_HOME:-$HOME/.local/gpu_terminal}"
RUNTIME_DIR="${KILIX_BONSAI_RUNTIME_DIR:-$GPU_TERMINAL_HOME/kilix-bonsai/runtime}"
# The llama.cpp tree to build. VibeASR.cpp vendors one, which is why a machine
# set up for the speech model usually already has the source for this.
LLAMA_SOURCE="${KILIX_BONSAI_LLAMA_SOURCE:-$HOME/VibeASR.cpp/3rdparty/llama.cpp}"

die() { printf 'kilix-bonsai: %s\n' "$*" >&2; exit 1; }
log() { printf 'kilix-bonsai: %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
usage: build-runtime.sh [--check]

  --check   report whether the runtime is built; build nothing

Environment:
  KILIX_BONSAI_LLAMA_SOURCE   llama.cpp checkout to build
                              (default: ~/VibeASR.cpp/3rdparty/llama.cpp)
  KILIX_BONSAI_RUNTIME_DIR    where the build lands
EOF
}

check_only=0
case "${1:-}" in
  '') ;;
  --check) check_only=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

server="$RUNTIME_DIR/build/bin/llama-server"

if [ "$check_only" = 1 ]; then
  if [ -x "$server" ]; then
    log "built: $server"
    exit 0
  fi
  log "not built (run this script without --check)"
  exit 1
fi

[ -x "$server" ] && { log "already built: $server"; exit 0; }

command -v cmake >/dev/null 2>&1 || die "cmake is required"
[ -d "$LLAMA_SOURCE" ] \
  || die "no llama.cpp source at $LLAMA_SOURCE — set KILIX_BONSAI_LLAMA_SOURCE"

umask 077
mkdir -p -- "$RUNTIME_DIR"
log "configuring $LLAMA_SOURCE"
cmake -S "$LLAMA_SOURCE" -B "$RUNTIME_DIR/build" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON >/dev/null \
  || die "cmake configure failed"

log "building llama-server (this takes a few minutes)"
cmake --build "$RUNTIME_DIR/build" --target llama-server \
  -j "$(nproc 2>/dev/null || echo 4)" >/dev/null \
  || die "build failed"

[ -x "$server" ] || die "the build did not produce $server"
log "built: $server"
