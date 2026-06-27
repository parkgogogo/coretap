#!/usr/bin/env bash
set -euo pipefail

CORETAP_REPO_URL="${CORETAP_REPO_URL:-https://github.com/parkgogogo/coretap.git}"
CORETAP_REF="${CORETAP_REF:-main}"
CORETAP_HOME="${CORETAP_HOME:-$HOME/Library/Application Support/Coretap}"
IDB_COMPANION_VERSION="${IDB_COMPANION_VERSION:-1.1.8}"
IDB_COMPANION_SHA256="${IDB_COMPANION_SHA256:-3b72cc6a9a5b1a22a188205a84090d3a294347a846180efd755cf1a3c848e3e7}"

SKIP_MODEL=0
SKIP_OCR=0
SKIP_SIMULATOR=0
SKIP_DEVICE=0
SKIP_NODE_SMOKE=0
NO_BREW_INSTALL=0
NO_WARM=0
PRINT_HELP=0
CORETAP_SOURCE_DIR=""

log() {
  printf '[coretap-install] %s\n' "$*"
}

warn() {
  printf '[coretap-install] warning: %s\n' "$*" >&2
}

fail() {
  printf '[coretap-install] error: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

usage() {
  cat <<'EOF'
Coretap installer

Usage:
  install.sh [options]

Options:
  --skip-model          Install the CLI but do not download/check/warm MAI-UI.
  --skip-ocr            Do not install/check Tesseract OCR.
  --skip-simulator      Do not install/check Simulator tap support.
  --skip-device         Do not install/check pymobiledevice3.
  --skip-node-smoke     Do not run the Node test-kit smoke check from a checkout.
  --no-brew-install     Never install missing packages with Homebrew.
  --no-warm             Install/check the model but skip model warm.
  --ref <git-ref>       Git ref to install when not running from a checkout.
  --repo <git-url>      Git repository URL to install when not running locally.
  -h, --help            Show this help.

Environment:
  CORETAP_HOME          Defaults to ~/Library/Application Support/Coretap.
  CORETAP_REF           Defaults to main.
  CORETAP_REPO_URL      Defaults to https://github.com/parkgogogo/coretap.git.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-model)
      SKIP_MODEL=1
      ;;
    --skip-ocr)
      SKIP_OCR=1
      ;;
    --skip-simulator)
      SKIP_SIMULATOR=1
      ;;
    --skip-device)
      SKIP_DEVICE=1
      ;;
    --skip-node-smoke)
      SKIP_NODE_SMOKE=1
      ;;
    --no-brew-install)
      NO_BREW_INSTALL=1
      ;;
    --no-warm)
      NO_WARM=1
      ;;
    --ref)
      shift
      [ "$#" -gt 0 ] || fail "--ref requires a value"
      CORETAP_REF="$1"
      ;;
    --repo)
      shift
      [ "$#" -gt 0 ] || fail "--repo requires a value"
      CORETAP_REPO_URL="$1"
      ;;
    -h|--help)
      PRINT_HELP=1
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
  shift
done

if [ "$PRINT_HELP" -eq 1 ]; then
  usage
  exit 0
fi

if [ "$(uname -s)" != "Darwin" ]; then
  fail "Coretap's local iOS automation installer currently supports macOS only"
fi

if [ "$(uname -m)" != "arm64" ]; then
  fail "Coretap's built-in MLX model pack requires Apple Silicon arm64"
fi

ensure_brew_package() {
  package="$1"
  binary="$2"
  if have "$binary"; then
    return 0
  fi
  if [ "$NO_BREW_INSTALL" -eq 1 ]; then
    return 1
  fi
  if ! have brew; then
    return 1
  fi
  log "Installing $package with Homebrew"
  brew install "$package"
}

ensure_brew_formula() {
  formula="$1"
  if [ "$NO_BREW_INSTALL" -eq 1 ]; then
    return 1
  fi
  if ! have brew; then
    return 1
  fi
  if brew list --formula "$formula" >/dev/null 2>&1; then
    return 0
  fi
  log "Installing $formula with Homebrew"
  brew install "$formula"
}

tesseract_has_lang() {
  lang="$1"
  have tesseract || return 1
  tesseract --list-langs 2>/dev/null | awk 'NR > 1 { print $0 }' | grep -qx "$lang"
}

ensure_uv() {
  if have uv; then
    return 0
  fi
  if ensure_brew_package uv uv; then
    return 0
  fi
  fail "uv is required. Install it first or rerun with Homebrew available: brew install uv"
}

tool_bin_dir() {
  uv tool dir --bin 2>/dev/null || printf '%s/.local/bin\n' "$HOME"
}

path_contains() {
  case ":$PATH:" in
    *":$1:"*) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_coretap_bin() {
  uv_bin="$(tool_bin_dir)"
  coretap_bin="$uv_bin/coretap"
  if [ ! -x "$coretap_bin" ]; then
    coretap_bin="$(command -v coretap || true)"
  fi
  [ -n "$coretap_bin" ] || fail "coretap executable was not found after install"
  printf '%s\n' "$coretap_bin"
}

install_coretap_cli() {
  script_path="${BASH_SOURCE[0]}"
  script_dir="$(cd "$(dirname "$script_path")" >/dev/null 2>&1 && pwd -P || true)"
  source_spec="git+$CORETAP_REPO_URL@$CORETAP_REF"

  if [ -n "$script_dir" ] && [ -f "$script_dir/pyproject.toml" ] && grep -q 'name = "coretap"' "$script_dir/pyproject.toml"; then
    source_spec="$script_dir"
    CORETAP_SOURCE_DIR="$script_dir"
  fi

  log "Installing Coretap CLI from $source_spec"
  if [ -n "$CORETAP_SOURCE_DIR" ]; then
    uv tool install --force --editable "$source_spec"
  else
    uv tool install --force "$source_spec"
  fi
}

stop_existing_daemon() {
  coretap_bin="$1"
  log "Stopping any existing coretap daemon"
  if "$coretap_bin" --daemon off --format json daemon stop >/dev/null 2>&1; then
    log "Stopped existing coretap daemon"
  else
    log "No running coretap daemon detected"
  fi
}

install_device_tools() {
  if [ "$SKIP_DEVICE" -eq 1 ]; then
    log "Skipping pymobiledevice3 install"
    return 0
  fi

  log "Installing pymobiledevice3"
  uv tool install --force "pymobiledevice3"
}

install_ocr_tools() {
  if [ "$SKIP_OCR" -eq 1 ]; then
    log "Skipping OCR install"
    return 0
  fi

  if have tesseract; then
    log "Tesseract already installed"
  elif ! ensure_brew_package tesseract tesseract; then
    warn "Tesseract is not installed. Text assertions will fail until you install it, e.g. brew install tesseract tesseract-lang"
    return 0
  fi

  if tesseract_has_lang eng && tesseract_has_lang chi_sim; then
    log "Tesseract English and Simplified Chinese language data available"
    return 0
  fi

  if ensure_brew_formula tesseract-lang && tesseract_has_lang eng && tesseract_has_lang chi_sim; then
    log "Tesseract language data installed"
    return 0
  fi

  warn "Tesseract default OCR language data is missing. Install it with: brew install tesseract-lang"
}

install_idb_companion() {
  if [ "$SKIP_SIMULATOR" -eq 1 ]; then
    log "Skipping Simulator tap support"
    return 0
  fi

  dest_root="$CORETAP_HOME/tools/idb-companion/$IDB_COMPANION_VERSION"
  companion="$dest_root/idb-companion.universal/bin/idb_companion"
  if [ -x "$companion" ]; then
    log "idb_companion already installed at $companion"
    return 0
  fi

  mkdir -p "$dest_root"
  tmp_dir="$(mktemp -d)"
  archive="$tmp_dir/idb-companion.universal.tar.gz"
  url="https://github.com/facebook/idb/releases/download/v$IDB_COMPANION_VERSION/idb-companion.universal.tar.gz"

  log "Downloading idb_companion $IDB_COMPANION_VERSION"
  curl -fL "$url" -o "$archive"
  actual_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
  if [ "$actual_sha" != "$IDB_COMPANION_SHA256" ]; then
    rm -rf "$tmp_dir"
    fail "idb_companion sha256 mismatch: expected $IDB_COMPANION_SHA256, got $actual_sha"
  fi

  tar -xzf "$archive" -C "$dest_root"
  rm -rf "$tmp_dir"
  if have xattr; then
    xattr -dr com.apple.quarantine "$dest_root/idb-companion.universal" >/dev/null 2>&1 || true
  fi
  [ -x "$companion" ] || fail "idb_companion install did not produce $companion"
  log "Installed idb_companion at $companion"

  log "Prewarming fb-idb client through uvx"
  uvx --from fb-idb idb --help >/dev/null
}

run_coretap_setup() {
  coretap_bin="$1"
  tmp_doctor=""
  log "Running coretap setup"
  "$coretap_bin" --daemon off --format json setup >/dev/null

  if [ "$SKIP_MODEL" -eq 0 ]; then
    log "Installing built-in MAI-UI model pack"
    "$coretap_bin" --daemon off --format json model install
    log "Checking model pack"
    "$coretap_bin" --daemon off --format json model check --deep
    if [ "$NO_WARM" -eq 0 ]; then
      log "Warming model pack"
      "$coretap_bin" --daemon off --format json model warm
    fi
  else
    log "Skipping model install"
  fi

  if [ "$SKIP_OCR" -eq 0 ]; then
    if ! "$coretap_bin" --daemon off --format json ocr check; then
      warn "OCR check failed. Install Tesseract if you need assert text / wait text."
    fi
  fi

  tmp_doctor="$(mktemp)"
  if "$coretap_bin" --daemon off --format json doctor >"$tmp_doctor"; then
    cat "$tmp_doctor"
    if have python3 && python3 - "$tmp_doctor" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
sys.exit(0 if data.get("result", {}).get("ready") is True else 1)
PY
    then
      log "Coretap doctor passed"
    else
      warn "Coretap doctor completed but reported not ready. Review the JSON output above."
    fi
  else
    cat "$tmp_doctor"
    warn "Coretap doctor did not pass. Review the JSON output above."
  fi
  rm -f "$tmp_doctor"
}

run_node_smoke() {
  coretap_bin="$1"
  if [ "$SKIP_NODE_SMOKE" -eq 1 ]; then
    log "Skipping Node test-kit smoke check"
    return 0
  fi
  if [ -z "$CORETAP_SOURCE_DIR" ] || [ ! -f "$CORETAP_SOURCE_DIR/packages/node/smoke.js" ]; then
    log "Skipping Node test-kit smoke check outside a source checkout"
    return 0
  fi
  if ! have node; then
    warn "Node.js is not installed. Skipping Node test-kit smoke check."
    return 0
  fi

  log "Running Node test-kit smoke check"
  CORETAP_BIN="$coretap_bin" node "$CORETAP_SOURCE_DIR/packages/node/smoke.js"
}

main() {
  log "Starting Coretap install"
  ensure_uv

  uv_bin="$(tool_bin_dir)"
  if ! path_contains "$uv_bin"; then
    warn "$uv_bin is not on PATH. Add it to PATH so agent processes can find coretap and pymobiledevice3."
  fi

  install_coretap_cli
  coretap_bin="$(resolve_coretap_bin)"
  stop_existing_daemon "$coretap_bin"
  install_device_tools
  install_ocr_tools
  install_idb_companion
  run_coretap_setup "$coretap_bin"
  run_node_smoke "$coretap_bin"

  log "Done"
}

main
