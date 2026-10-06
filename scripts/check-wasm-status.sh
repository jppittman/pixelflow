#!/usr/bin/env bash
#
# Which crates build for wasm32-unknown-unknown, checked against a baseline.
#
# Every crate is reported even when an earlier one fails, so the log says what
# the state of wasm32 support actually is rather than where it first tripped.
# The verdict comes from comparing that state to
# scripts/wasm_status_baseline.txt: a new failure and a fixed crate both exit
# nonzero, because both mean the file no longer describes the tree. See that
# file for why the listed crates fail.
#
# The workflow owns whether a nonzero exit blocks the merge; this script owns
# reporting the real outcome. Those were the same thing in the inline version
# this replaces, and it got them both wrong: its last command was an `echo`,
# so three failed builds exited 0.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

baseline_file="scripts/wasm_status_baseline.txt"

# Comments and blank lines out; the rest are labels, which contain spaces.
expected_failures=()
while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*(#.*)?$ ]] && continue
  expected_failures+=("$line")
done < "$baseline_file"

in_baseline() {
  local needle=$1 entry
  for entry in ${expected_failures+"${expected_failures[@]}"}; do
    [[ "$entry" == "$needle" ]] && return 0
  done
  return 1
}

actual_failures=()
check() {
  local label=$1
  shift
  echo "::group::$label"
  if cargo check --target wasm32-unknown-unknown "$@"; then
    echo "PASS: $label"
  else
    echo "FAIL: $label does not build for wasm32-unknown-unknown"
    actual_failures+=("$label")
  fi
  echo "::endgroup::"
}

check "pixelflow-search" -p pixelflow-search
check "pixelflow-core" -p pixelflow-core
DISPLAY_DRIVER=web check "pixelflow-runtime web driver" -p pixelflow-runtime --no-default-features --features display_web

status=0
for label in ${actual_failures+"${actual_failures[@]}"}; do
  in_baseline "$label" && continue
  echo "::error::$label stopped building for wasm32-unknown-unknown. If this is intended, add it to $baseline_file."
  status=1
done
for label in ${expected_failures+"${expected_failures[@]}"}; do
  printf '%s\n' ${actual_failures+"${actual_failures[@]}"} | grep -qxF "$label" && continue
  echo "::error::$label now builds for wasm32-unknown-unknown. Remove it from $baseline_file."
  status=1
done

if [[ "$status" -eq 0 ]]; then
  echo "wasm32 buildability matches $baseline_file (${#actual_failures[@]} known failure(s))."
fi
exit "$status"
