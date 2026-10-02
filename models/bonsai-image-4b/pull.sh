#!/usr/bin/env bash
# Acquisition and consent belong to the same catalog used by the wizard.
set -euo pipefail
here="$(cd -- "$(dirname -- "$0")" && pwd)"
original=("$@")
variant=ternary-gemlite
source_args=()
dry_run=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) echo 'usage: pull.sh [--variant ternary-gemlite|binary-gemlite] [--from DIR] [--dry-run]'; exit 0 ;;
    --variant) [ "$#" -ge 2 ] || exit 2; variant="$2"; shift 2 ;;
    --from) [ "$#" -ge 2 ] || exit 2; source_args=(--from "$2"); shift 2 ;;
    --force) shift ;; # Content verifies and reuses valid installed bytes.
    --dry-run) dry_run=1; shift ;;
    *) echo 'usage: pull.sh [--variant ternary-gemlite|binary-gemlite] [--from DIR] [--dry-run]' >&2; exit 2 ;;
  esac
done
case "$variant" in
  ternary-gemlite|binary-gemlite) ;;
  *) echo "bonsai-image-4b has no variant '$variant'" >&2; exit 2 ;;
esac
if [ "$dry_run" = 1 ]; then
  exec "$here/../_shared/pull.sh" --model-dir "$here" "${original[@]}"
fi
command -v kilix >/dev/null || {
  echo 'Install Kilix, then use kilix wizard to accept terms and install this model.' >&2; exit 3;
}
exec kilix models install "bonsai-image-4b-$variant" "${source_args[@]}"
