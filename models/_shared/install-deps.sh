#!/usr/bin/env bash
# The dependency install every model folder's install-deps.sh runs.
#
# Two kinds of dependency, handled differently on purpose:
#
#   System packages are CHECKED and REPORTED, not installed, unless you pass
#   --apt. Installing distro packages needs root, and Plebian-OS release images
#   pin an apt snapshot for reproducibility — a script that silently ran
#   `sudo apt-get install` from a model UI could drift a machine off its pinned
#   closure. Printing the exact command and letting you run it is the honest
#   default; --apt is there for when you have decided.
#
#   Python packages go into a virtualenv this repository owns, one per model,
#   never into the system interpreter and never into the directory another
#   component owns.
#
# On success a stamp is written into the model's store so the TUI can say
# "dependencies installed" without shelling out on every frame.
set -euo pipefail

MODEL_DIR=""
APT=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
usage: install-deps.sh [--apt] [--check]

  --apt     install the missing system packages too (runs sudo apt-get)
  --check   report what is missing and exit non-zero if anything is; changes
            nothing
EOF
}

while (($#)); do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --apt) APT=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'install-deps.sh: unknown option: %s\n' "$1" >&2
       usage >&2; exit 2 ;;
  esac
done

[ -n "$MODEL_DIR" ] || {
  echo "install-deps.sh: --model-dir is required" >&2; exit 2; }
MODEL_DIR="$(cd -- "$MODEL_DIR" && pwd)"
REPO="$(cd -- "$MODEL_DIR/../.." && pwd)"
CLI="$REPO/tools/kilix-bonsai/main.py"
MODEL_ID="$(basename -- "$MODEL_DIR")"

die() { printf 'kilix-bonsai: %s\n' "$*" >&2; exit 1; }
log() { printf 'kilix-bonsai: %s\n' "$*" >&2; }

command -v python3 >/dev/null 2>&1 || die "python3 is required"
[ "$(id -u)" -ne 0 ] || die "run this as the desktop user, not root"

plan="$(python3 "$CLI" plan "$MODEL_ID")" \
  || die "could not read $MODEL_DIR/MODEL.json"
field() { printf '%s\n' "$plan" | awk -v k="$1" -F'\t' '$1==k{print $2; exit}'; }
column() { printf '%s\n' "$plan" | awk -v k="$1" -F'\t' '$1==k{print $2}'; }

title="$(field TITLE)"
store="$(field STORE)"
venv="$(field VENV)"
mapfile -t apt_packages < <(column APT)
mapfile -t pip_packages < <(column PIP)

log "$title — dependencies"

# ---------------------------------------------------------------- system ----
missing=()
if [ "${#apt_packages[@]}" -gt 0 ]; then
  if command -v dpkg-query >/dev/null 2>&1; then
    for package in "${apt_packages[@]}"; do
      if dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
           | grep -q "ok installed"; then
        printf '  ok      %s\n' "$package"
      else
        printf '  MISSING %s\n' "$package"
        missing+=("$package")
      fi
    done
  else
    # Not Debian. Naming the packages is still useful; claiming to know their
    # names on this distro would not be.
    log "not a dpkg system — install the equivalents of:" \
        "${apt_packages[*]}"
  fi
fi

if [ "${#missing[@]}" -gt 0 ]; then
  if [ "$CHECK_ONLY" = 1 ]; then
    die "${#missing[@]} system package(s) missing"
  elif [ "$APT" = 1 ]; then
    command -v apt-get >/dev/null 2>&1 || die "apt-get is not available"
    log "installing: ${missing[*]}"
    sudo apt-get install -y -- "${missing[@]}" \
      || die "apt-get failed; install the packages above and re-run"
  else
    log "run this, then re-run install-deps.sh:"
    printf '\n    sudo apt-get install %s\n\n' "${missing[*]}"
  fi
fi

# ---------------------------------------------------------------- python ----
if [ "${#pip_packages[@]}" -gt 0 ]; then
  if [ "$CHECK_ONLY" = 1 ]; then
    [ -x "$venv/bin/python" ] || die "no virtualenv at $venv"
  else
    umask 077
    if [ ! -x "$venv/bin/python" ]; then
      log "creating $venv"
      python3 -m venv "$venv" \
        || die "could not create $venv — install python3-venv and re-run"
    fi
    log "installing: ${pip_packages[*]}"
    "$venv/bin/python" -m pip install --quiet --upgrade pip \
      || log "could not upgrade pip in the virtualenv; continuing"
    "$venv/bin/python" -m pip install --quiet -- "${pip_packages[@]}" \
      || die "pip install failed"
  fi
fi

[ "$CHECK_ONLY" = 0 ] || { log "all dependencies present"; exit 0; }

# ----------------------------------------------------------------- stamp ----
umask 077
mkdir -p -- "$store"
stamp="$store/.kilix-bonsai-deps.json"
python3 - "$stamp" "$MODEL_ID" "$venv" "${#missing[@]}" \
         "${apt_packages[*]-}" "${pip_packages[*]-}" <<'PY'
import json
import sys

path, model_id, venv, missing, apt_packages, pip_packages = sys.argv[1:7]
document = {
    "model": model_id,
    "venv": venv if pip_packages.strip() else None,
    "apt": apt_packages.split(),
    "pip": pip_packages.split(),
    # Recorded rather than asserted: with the default (no --apt) a system
    # package can still be missing, and a stamp that claimed otherwise would
    # be worse than no stamp.
    "system_packages_missing": int(missing),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2)
    handle.write("\n")
PY

if [ "${#missing[@]}" -gt 0 ] && [ "$APT" = 0 ]; then
  log "python side ready; ${#missing[@]} system package(s) still missing"
  exit 1
fi
log "$title dependencies are ready"
