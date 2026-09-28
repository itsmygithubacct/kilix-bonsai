#!/usr/bin/env sh
here="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$here/../../scripts/install-image-runtime.sh" "$@"
