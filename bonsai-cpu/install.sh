#!/usr/bin/env bash
# Put `bonsai-cpu` on PATH as a launcher that runs the tool from this
# checkout, so updating is a `git pull` rather than a reinstall.
set -euo pipefail

# Same floor as the scripts this installs a launcher for.
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] \
    || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -le 3 ]; }; then
  printf 'bonsai-cpu: bash 4.4+ required (running %s)\n' "${BASH_VERSION:-unknown}" >&2
  exit 1
fi

PREFIX="${BONSAI_CPU_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

shell_quote() {
    printf "'"
    printf '%s' "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

mkdir -p -- "$BIN_DIR"
quoted_target="$(shell_quote "$REPO_DIR/bin/bonsai-cpu")"
cat > "$BIN_DIR/bonsai-cpu" <<EOF
#!/bin/sh
exec $quoted_target "\$@"
EOF
chmod 755 "$BIN_DIR/bonsai-cpu"
printf 'bonsai-cpu: installed %s\n' "$BIN_DIR/bonsai-cpu" >&2

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) printf 'bonsai-cpu: note: %s is not on PATH\n' "$BIN_DIR" >&2 ;;
esac
