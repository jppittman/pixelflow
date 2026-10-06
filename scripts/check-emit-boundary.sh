#!/usr/bin/env bash
#
# The emit boundary: pixelflow-codegen's emitter (`emit/`, the register
# allocator included) names nothing upstream of it -- no optimizer, no
# lowering, no scoping, no driver -- and `program/` names nothing downstream.
# See docs/plans/2026-09-12-emit-should-just-emit.md, and
# check_emit_boundary.py's module docstring for the exact rules and for what a
# text scan cannot see. There is no baseline: zero hits is the bar.
#
# The scanner's own self-test runs first, so a scanner that has stopped
# flagging fails the job by itself.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 scripts/check_emit_boundary.py --self-test
python3 scripts/check_emit_boundary.py
