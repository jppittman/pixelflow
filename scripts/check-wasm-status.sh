#!/usr/bin/env bash
# Report every crate even when an earlier check fails. The workflow owns
# the non-blocking policy; this script must still return the real outcome.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

failed=0
check() {
  local label=$1
  shift
  echo "::group::$label"
  if cargo check --target wasm32-unknown-unknown "$@"; then
    echo "PASS: $label"
    echo "::endgroup::"
    return
  fi
  echo "::warning::FAIL: $label does not build for wasm32-unknown-unknown"
  echo "::endgroup::"
  failed=1
}

check "pixelflow-search" -p pixelflow-search
check "pixelflow-core" -p pixelflow-core
DISPLAY_DRIVER=web check "pixelflow-runtime web driver" -p pixelflow-runtime --no-default-features --features display_web

exit "$failed"
