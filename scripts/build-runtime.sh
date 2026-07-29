#!/usr/bin/env bash
# Build the pinned runtime, CPU only.
#
# `llama-cli`, `llama-server`, `llama-bench`: the one-shot path, the API the
# chat interfaces drive, and the instrument the numbers in the README come
# from. The build lands next to the checkout, outside this repository,
# because a compiled binary is per-machine.
set -euo pipefail

GPU_TERMINAL_HOME="${GPU_TERMINAL_HOME:-$HOME/.local/gpu_terminal}"
RUNTIME_DIR="${BONSAI_CPU_RUNTIME_DIR:-$GPU_TERMINAL_HOME/bonsai-cpu/runtime}"
SRC_DIR="$RUNTIME_DIR/llama.cpp"

die() { printf 'bonsai-cpu: %s\n' "$*" >&2; exit 1; }
log() { printf 'bonsai-cpu: %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
usage: build-runtime.sh [--check]

  --check   report whether the runtime is built; build nothing

Environment:
  BONSAI_CPU_RUNTIME_DIR   where the checkout and build live
                           (default: ~/.local/gpu_terminal/bonsai-cpu/runtime)
  BONSAI_CPU_BUILD_JOBS    parallel compile jobs (default: nproc)
EOF
}

check_only=0
case "${1:-}" in
  '') ;;
  --check) check_only=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

cli="$SRC_DIR/build/bin/llama-cli"

if [ "$check_only" = 1 ]; then
  if [ -x "$cli" ]; then
    log "built: $SRC_DIR/build/bin"
    exit 0
  fi
  log "not built (run scripts/get-runtime.sh, then this script)"
  exit 1
fi

command -v cmake >/dev/null 2>&1 || die "cmake is required"
[ -d "$SRC_DIR" ] || die "no runtime source — run scripts/get-runtime.sh first"

jobs="${BONSAI_CPU_BUILD_JOBS:-$(nproc)}"
cd "$SRC_DIR"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
    >/dev/null
cmake --build build -j "$jobs" --target llama-cli llama-server llama-bench
log "built: $SRC_DIR/build/bin"
