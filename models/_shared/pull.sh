#!/usr/bin/env bash
# The download every model folder's pull.sh runs.
#
# One implementation, five callers. What a model *is* — its upstream
# repository, the pinned revision, every file with its size and sha256, and
# where the weights belong — lives in that model's MODEL.json and is read here
# through `kilix-bonsai plan`, so this script never parses JSON, never resolves
# a store path, and cannot drift from what the TUI shows.
#
# Properties worth stating, because they are the reason this is not a curl
# loop:
#   * Resumable. Interrupted transfers continue with `curl -C -` rather than
#     restarting a 3.8 GB download from zero.
#   * Verified. Every file with a published digest is sha256-checked after it
#     lands, and a mismatch is a failure, not a warning.
#   * Idempotent. A file already present at the right size and digest is
#     skipped, so re-running after a network drop costs a stat and a hash.
#   * Atomic per file. Downloads land on `.part` and are renamed only after
#     they verify, so an interrupted run never leaves a file that looks
#     complete.
set -euo pipefail

MODEL_DIR=""
VARIANT=""
FORCE=0
DRY_RUN=0
NO_VERIFY=0
FROM=""

usage() {
  cat <<'EOF'
usage: pull.sh [--variant ID] [--from DIR] [--force] [--dry-run] [--no-verify]

  --variant ID   download a non-default variant (see MODEL.json)
  --from DIR     take files from a copy already on this machine instead of
                 fetching them; each one still has to match its published
                 sha256, and anything DIR does not have is downloaded normally
  --force        re-download files that are already present and verified
  --dry-run      print what would be fetched, touch nothing
  --no-verify    skip sha256 checking (not recommended; the digests are the
                 only thing standing between a truncated proxy response and a
                 model that loads to garbage)
EOF
}

while (($#)); do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --variant) VARIANT="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-verify) NO_VERIFY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'pull.sh: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$MODEL_DIR" ] || { echo "pull.sh: --model-dir is required" >&2; exit 2; }
MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd)"
REPO="$(cd -- "$MODEL_DIR/../.." && pwd)"
CLI="$REPO/tools/kilix-bonsai/main.py"
MODEL_ID="$(basename -- "$MODEL_DIR")"

die() { printf 'kilix-bonsai: %s\n' "$*" >&2; exit 1; }
log() { printf 'kilix-bonsai: %s\n' "$*" >&2; }

command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v sha256sum >/dev/null 2>&1 || [ "$NO_VERIFY" = 1 ] \
  || die "sha256sum is required (or pass --no-verify)"

downloader=""
if command -v curl >/dev/null 2>&1; then downloader=curl
elif command -v wget >/dev/null 2>&1; then downloader=wget
else die "neither curl nor wget is installed"; fi

plan_args=(python3 "$CLI" plan "$MODEL_ID")
[ -z "$VARIANT" ] || plan_args+=(--variant "$VARIANT")
plan="$("${plan_args[@]}")" || die "could not read $MODEL_DIR/MODEL.json"

field() { printf '%s\n' "$plan" | awk -v k="$1" -F'\t' '$1==k{print $2; exit}'; }

title="$(field TITLE)"
variant_id="$(field VARIANT)"
directory="$(field DIR)"
total="$(field BYTES)"

# Files that are already present at the right size and digest are not fetched
# again, so this reports what the run will actually do rather than the size of
# the whole variant.
outstanding=0
count=0
while IFS=$'\t' read -r kind path size sha url; do
  [ "$kind" = FILE ] || continue
  count=$((count + 1))
  target="$directory/$path"
  if [ "$FORCE" = 0 ] && [ -f "$target" ] \
     && [ "$(stat -c %s -- "$target" 2>/dev/null || echo 0)" = "$size" ]; then
    continue
  fi
  outstanding=$((outstanding + size))
done <<<"$plan"

human() {
  awk -v b="$1" 'BEGIN{
    split("B K M G T", u, " ");
    i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf (i == 1 ? "%d%s\n" : "%.1f%s\n"), b, u[i]
  }'
}

log "$title — variant $variant_id"
log "into $directory"
log "$count files, $(human "$total") total, $(human "$outstanding") to fetch"

if [ "$DRY_RUN" = 1 ]; then
  while IFS=$'\t' read -r kind path size sha url; do
    [ "$kind" = FILE ] || continue
    printf '%12s  %s\n' "$(human "$size")" "$path"
  done <<<"$plan"
  exit 0
fi

[ "$outstanding" = 0 ] && [ "$FORCE" = 0 ] && { log "already complete"; exit 0; }

# 0700: these stores hold multi-gigabyte files under the user's own state
# directory, and two of them sit inside directories other components own.
umask 077
mkdir -p -- "$directory" || die "could not create $directory"

fetch() {
  local url="$1" target="$2"
  case "$downloader" in
    curl) curl -fL --retry 3 --retry-delay 2 -C - --progress-bar \
               ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
               -o "$target" -- "$url" ;;
    wget) wget -c --tries=3 --show-progress -q \
               ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} \
               -O "$target" -- "$url" ;;
  esac
}

failed=0
index=0
while IFS=$'\t' read -r kind path size sha url; do
  [ "$kind" = FILE ] || continue
  index=$((index + 1))
  target="$directory/$path"
  part="$target.part"

  if [ "$FORCE" = 0 ] && [ -f "$target" ] \
     && [ "$(stat -c %s -- "$target" 2>/dev/null || echo 0)" = "$size" ]; then
    if [ "$NO_VERIFY" = 1 ] || [ "$sha" = "-" ]; then
      continue
    fi
    if [ "$(sha256sum -- "$target" | cut -d' ' -f1)" = "$sha" ]; then
      continue
    fi
    log "$path: digest mismatch on an existing file, re-fetching"
    rm -f -- "$target"
  fi

  mkdir -p -- "$(dirname -- "$target")"

  # --from: a machine that already has these weights somewhere else should not
  # re-download them. The copy is verified exactly as a download would be, and
  # a mismatch falls through to fetching rather than failing — the local copy
  # being wrong is not a reason to refuse the model.
  if [ -n "$FROM" ] && [ -f "$FROM/$path" ]; then
    local_size="$(stat -c %s -- "$FROM/$path" 2>/dev/null || echo 0)"
    if [ "$local_size" = "$size" ] && { [ "$NO_VERIFY" = 1 ] || [ "$sha" = "-" ] \
         || [ "$(sha256sum -- "$FROM/$path" | cut -d' ' -f1)" = "$sha" ]; }; then
      printf '[%d/%d] %s (%s) — from %s\n' \
        "$index" "$count" "$path" "$(human "$size")" "$FROM" >&2
      # Hard link when the two paths share a filesystem, so a 4 GB adoption
      # costs no extra disk; copy when they do not.
      ln -f -- "$FROM/$path" "$target" 2>/dev/null \
        || cp -f -- "$FROM/$path" "$target"
      continue
    fi
    log "$path: the copy in $FROM does not match, downloading instead"
  fi

  printf '[%d/%d] %s (%s)\n' "$index" "$count" "$path" "$(human "$size")" >&2
  # A previous run's .part is resumed; a --force run starts clean.
  [ "$FORCE" = 0 ] || rm -f -- "$part"

  # A .part at or past the published size cannot be resumed from — it is
  # already wrong. This happens for real: `curl -C -` sends a Range, an
  # upstream CDN ignores it and replies with the whole body, and curl appends,
  # leaving a file LARGER than the target. Without this guard every retry
  # appends another copy and the download can never succeed.
  part_size="$(stat -c %s -- "$part" 2>/dev/null || echo 0)"
  if [ "$part_size" -ge "$size" ]; then
    [ "$part_size" = 0 ] || log "$path: discarding an unusable partial file"
    rm -f -- "$part"
  fi

  if ! fetch "$url" "$part"; then
    log "$path: download failed"
    failed=$((failed + 1))
    continue
  fi

  actual="$(stat -c %s -- "$part" 2>/dev/null || echo 0)"
  if [ "$actual" != "$size" ]; then
    if [ "$actual" -gt "$size" ]; then
      # Overshot: resuming would append yet again, so the partial is dropped.
      log "$path: got $actual bytes for a $size byte file — discarding it"
      rm -f -- "$part"
    else
      log "$path: expected $size bytes, got $actual — .part kept to resume"
    fi
    failed=$((failed + 1))
    continue
  fi
  if [ "$NO_VERIFY" = 0 ] && [ "$sha" != "-" ]; then
    if [ "$(sha256sum -- "$part" | cut -d' ' -f1)" != "$sha" ]; then
      log "$path: sha256 mismatch — upstream content changed, or the transfer"
      log "$path: was corrupted. Discarding it rather than installing it."
      # Not kept: a complete-but-wrong file is not a resumable prefix, and
      # keeping it would make the next run resume from its end.
      rm -f -- "$part"
      failed=$((failed + 1))
      continue
    fi
  fi
  mv -f -- "$part" "$target"
done <<<"$plan"

if [ "$failed" -gt 0 ]; then
  die "$failed file(s) did not complete — re-run to resume"
fi

log "$title · $variant_id is ready in $directory"
