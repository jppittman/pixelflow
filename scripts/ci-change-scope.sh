#!/usr/bin/env bash
# Classify a CL by what it touches, so presubmit can skip the build-and-test
# jobs on a change that cannot affect them.
#
# A CL is "docs only" when every changed path is under docs/ or is a Markdown
# file anywhere in the tree (README.md, CLAUDE.md, .claude/agents/*.md). Those
# files are read by no build, no test and no check script that gates a merge --
# except `check-doc-paths.sh`, which reads docs and so is exactly the job a
# docs-only CL most needs.
#
# The six metadata jobs (CL metadata, driver-cfg coverage, bin declarations,
# platform-cfg encapsulation, provenance-journal scope, doc source paths) still
# run on every CL because they are seconds each and none of them needs a build.
#
# An empty diff, an unreadable range, or any path outside the pattern is NOT
# docs-only: the conservative answer is to run everything.
#
# Usage: ci-change-scope.sh <base-commit> <head-commit>
# Prints `docs_only=true|false`, and appends the same line to $GITHUB_OUTPUT
# when that is set.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <base-commit> <head-commit>" >&2
  exit 2
fi

base_commit=$1
head_commit=$2

emit() {
  echo "docs_only=$1"
  if [[ -n ${GITHUB_OUTPUT:-} ]]; then
    echo "docs_only=$1" >>"$GITHUB_OUTPUT"
  fi
}

# Three dots: the diff against the merge base, which is what the PR shows and
# what CI tests. Two dots would also count everything main gained meanwhile.
if ! changed=$(git diff --name-only "${base_commit}...${head_commit}" 2>/dev/null); then
  echo "cannot diff ${base_commit}...${head_commit}; running everything" >&2
  emit false
  exit 0
fi

if [[ -z $changed ]]; then
  echo "empty diff; running everything" >&2
  emit false
  exit 0
fi

docs_only=true
while IFS= read -r path; do
  case "$path" in
    docs/*) ;;
    *.md) ;;
    *)
      docs_only=false
      echo "not docs-only: $path" >&2
      break
      ;;
  esac
done <<<"$changed"

emit "$docs_only"
