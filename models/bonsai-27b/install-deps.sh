#!/usr/bin/env sh
# install-deps.sh for bonsai-27b. The implementation is shared by every model folder — what
# differs between models is MODEL.json, not this logic — so this is a wrapper
# that names its own folder and hands over.
here="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$here/../_shared/install-deps.sh" --model-dir "$here" "$@"
