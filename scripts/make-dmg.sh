#!/usr/bin/env bash
#
# Packages CoreTerm.app into a distributable DMG: a staging folder with the
# app and an /Applications symlink (the standard drag-to-install layout),
# compressed via hdiutil (built into macOS -- no extra tool to install).
#
# Usage: make-dmg.sh <path-to-CoreTerm.app> <output-dmg-path>
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <path-to-CoreTerm.app> <output-dmg-path>" >&2
  exit 2
fi

APP_PATH="$1"
DMG_PATH="$2"

if [[ ! -d "$APP_PATH" ]]; then
  echo "::error::app bundle not found at $APP_PATH" >&2
  exit 1
fi

staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT

cp -R "$APP_PATH" "$staging_dir/"
ln -s /Applications "$staging_dir/Applications"

rm -f "$DMG_PATH"
hdiutil create \
  -volname "CoreTerm" \
  -srcfolder "$staging_dir" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

echo "Created $DMG_PATH"
