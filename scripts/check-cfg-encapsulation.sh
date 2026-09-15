#!/usr/bin/env bash
#
# Platform `#[cfg(...)]` is encapsulation, not sprinkle: it must gate a whole
# file/mod, or be a single-line dispatch item in mod.rs (e.g.
# `#[cfg(aarch64)] pub use native::Foo as PlatformFoo;`). Anything else --
# scattered on a fn/struct/impl/field inside an otherwise platform-agnostic
# file -- means the platform split is `grep -r cfg` instead of a file
# boundary. See check_cfg_encapsulation.py's module docstring for the exact
# rule and why it's baselined against pre-existing violations.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 scripts/check_cfg_encapsulation.py
