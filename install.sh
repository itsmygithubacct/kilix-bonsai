#!/usr/bin/env bash
# Put `bonsai-cpu` on PATH as a launcher that runs the tool from this
# checkout, so updating is a `git pull` rather than a reinstall.
set -euo pipefail

PREFIX="${BONSAI_CPU_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p -- "$BIN_DIR"
cat > "$BIN_DIR/bonsai-cpu" <<EOF
#!/bin/sh
exec "$REPO_DIR/bin/bonsai-cpu" "\$@"
EOF
chmod 755 "$BIN_DIR/bonsai-cpu"
printf 'bonsai-cpu: installed %s\n' "$BIN_DIR/bonsai-cpu" >&2

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) printf 'bonsai-cpu: note: %s is not on PATH\n' "$BIN_DIR" >&2 ;;
esac
