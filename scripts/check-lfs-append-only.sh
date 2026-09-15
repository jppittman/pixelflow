#!/usr/bin/env bash
#
# An append-only LFS-tracked file may not shrink.
#
# WHAT WENT WRONG, twice, in one week. `docs/results/journal.jsonl` is matched
# by `*.jsonl filter=lfs` in .gitattributes, so a working tree without git-lfs
# installed holds the ~130-byte *pointer text* where the file should be:
#
#     version https://git-lfs.github.com/spec/v1
#     oid sha256:859447a1...
#     size 70578
#
# An agent or a human appending a journal record to that text and committing it
# does not append to the journal -- git-lfs on the push side cleans
# pointer-plus-one-line into a fresh 217-byte LFS object, and the 70,578 bytes
# of accumulated provenance are gone. The commit looks perfect: a two-line diff
# on a pointer, "docs(results): journal the run", green on every job we ran.
#
# PRs #1207 and #1215 both did exactly this. A reviewer caught it on each, by
# reading the pointer's `size` field by eye. That is not a gate -- it is
# somebody happening to look -- and the failure is silent by construction,
# because the only visible artifact is a number inside a text file nobody
# reads. So: a check.
#
# WHAT THIS CHECKS. For each path listed in APPEND_ONLY below, if the file is
# an LFS pointer on both sides, its declared `size` may not decrease between
# the base commit and the head commit. Growth is fine, an unchanged file is
# fine, and a file that is not an LFS pointer on either side is skipped -- this
# is a guard on the pointer arithmetic, not a content review.
#
# WHY SIZE AND NOT CONTENT. Reading the content needs the LFS objects, which
# needs an `lfs pull` of a file that only grows -- cost on every CI run, to
# check a property the pointer already states. The pointer's own `size` field
# is the fact that was wrong in both incidents, and it is free to read.
#
# LIMITS, stated rather than papered over: this catches truncation, not a
# rewrite that replaces records while keeping the byte count at or above the
# base. That would need the objects. Truncation is the failure that actually
# happened, twice, and it is the one a missing git-lfs produces.

set -euo pipefail

# Paths the repository treats as append-only. Add a path here when it acquires
# that contract -- and say where the contract is written down.
#
#   docs/results/journal.jsonl
#     "append-only", docs/plans/2026-08-05-egraph-nnue-research-workflow.md:295
APPEND_ONLY=(
  "docs/results/journal.jsonl"
)

BASE_COMMIT="${1:?usage: check-lfs-append-only.sh <base-commit> <head-commit>}"
HEAD_COMMIT="${2:?usage: check-lfs-append-only.sh <base-commit> <head-commit>}"

# Echoes the `size` line of an LFS pointer, or nothing when the path is absent
# at that commit or is not a pointer. A pointer is identified by its first
# line, which the LFS spec fixes.
pointer_size() {
  local commit="$1" path="$2" blob
  blob=$(git show "$commit:$path" 2>/dev/null) || return 0
  case "$blob" in
    "version https://git-lfs.github.com/spec/v1"*) ;;
    *) return 0 ;;
  esac
  printf '%s\n' "$blob" | sed -n 's/^size \([0-9][0-9]*\)$/\1/p' | head -1
}

failed=0
checked=0

for path in "${APPEND_ONLY[@]}"; do
  base_size=$(pointer_size "$BASE_COMMIT" "$path")
  head_size=$(pointer_size "$HEAD_COMMIT" "$path")

  # Not a pointer on one side or the other: nothing to compare. This covers a
  # newly added file, a deleted one, and a path that is not LFS-tracked.
  if [ -z "$base_size" ] || [ -z "$head_size" ]; then
    continue
  fi

  checked=$((checked + 1))

  if [ "$head_size" -lt "$base_size" ]; then
    lost=$((base_size - head_size))
    echo "error: $path shrank from $base_size to $head_size bytes (-$lost)" >&2
    echo "" >&2
    echo "  $path is append-only. An LFS pointer that shrinks is almost always" >&2
    echo "  a checkout without git-lfs: the pointer text was edited as if it" >&2
    echo "  were the file, and the real contents were replaced rather than" >&2
    echo "  appended to." >&2
    echo "" >&2
    echo "  Fix: install git-lfs, 'git lfs pull' the path, redo the edit on the" >&2
    echo "  real contents. To keep a merge moving without the objects, take the" >&2
    echo "  base side of the file unchanged and append the record separately." >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

echo "OK: $checked append-only LFS path(s) did not shrink between $BASE_COMMIT and $HEAD_COMMIT"
