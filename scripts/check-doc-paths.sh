#!/usr/bin/env bash
#
# A plan or design may not name a source file the tree does not have.
#
# `docs/plans/2026-09-06-kernel-with-a-lattice.md` §4 asked for this check by
# name, after every stage of that programme deleted files that prose elsewhere
# still described, and every stage found them by grepping rather than by a job
# failing:
#
#     "A gap in CI is a check to write: a job that fails when a doc names a
#      path or a type the tree does not have would have caught all of it,
#      cheaply, every time."
#
# This is the path half. Types are deliberately not checked: a backticked
# PascalCase word is as often prose, a rustc error code, or a table cell as it
# is an identifier, and a check that cries wolf is worse than no check.
# A backticked `foo/bar.rs` is unambiguous, so that is what this reads.
#
# SCOPE -- plans, designs, superpowers/ and the loose docs at the root of
# docs/, per docs/README.md's own classification. Those are the documents a
# reader *acts on*, so a dead path in one sends somebody to a file that is not
# there. `docs/results/` and `docs/bugs/` are dated evidence: a July audit
# naming a file deleted in August is a correct record of July, and flagging it
# would be wrong.
#
# THREE WAYS TO BE FINE, the first two of which say something true to a reader:
#
#   1. The document carries a supersession banner in its first 18 lines
#      (ARCHIVED / Superseded / Retracted / Historical / Obsolete / Withdrawn).
#      A historical document is *expected* to name things that are gone; the
#      banner is what tells the reader not to go looking for them.
#
#   2. The mention declares its own death in the surrounding paragraph --
#      "deleted",
#      "removed", "retired", "gone", "no longer", "never existed". A live plan
#      saying "`ir_bridge.rs` is deleted by this change" is correct prose, not
#      rot, and `docs/designs/opkind-numbering-is-private.md` is the worked
#      example in the tree.
#
#   3. The pair is recorded in the baseline (see below).
#
# So the banner is load-bearing rather than decorative: adding one is how a
# stale document is made honest, which is the outcome this check exists to buy.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Every .rs path in the tree, as full paths and as bare basenames. Docs cite
# both ("`pixelflow-search/src/egraph/cost.rs`" and "`cost.rs`"), and a
# basename hit is enough to prove the file exists somewhere.
existing_paths="$(mktemp)"
existing_names="$(mktemp)"
find . -name '*.rs' -not -path './target/*' | sed 's|^\./||' | sort -u >"$existing_paths"
sed 's|.*/||' "$existing_paths" | sort -u >"$existing_names"

# Prose that makes naming a dead file correct rather than stale. Matched over
# the mention's whole paragraph rather than its line, because prose wraps: a
# banner routinely lists four dead paths and puts "were deleted" on the next
# line, and line-scoped matching would flag the very sentence that fixes it.
declares_death='delet|remov|retir|\bgone\b|no longer|never exist|used to |replaced by|absorbed|renamed|supersed|dropped|split into|moved to|collapsed into|folded into|is not in the tree'
banner='ARCHIVED|Supersed|supersed|Retracted|retracted|Historical|historical|Obsolete|obsolete|Withdrawn|withdrawn'

# THE BASELINE, and why one is needed at all.
#
# A path-existence check cannot tell "this file was deleted" from "this plan
# proposes to create this file" -- both are absent from the tree, and only the
# author's intent separates them. `2026-09-01-guide-candidate-context.md` is the
# worked example: it designs `egraph/cell.rs` and `training/rule_conditioned.rs`,
# neither of which exists, and it is correct to name them.
#
# So the check ships baselined, the way #1237 baselines platform-cfg
# encapsulation: today's remaining pairs are recorded, and only NEW ones fail.
# Each baseline line is `<doc>\t<path>`; blank lines and `#` comments ignored.
# Deleting a line once the doc is fixed is the intended direction of travel.
baseline_file="scripts/doc-paths-baseline.txt"
baseline="$(mktemp)"
trap 'rm -f "$existing_paths" "$existing_names" "$baseline"' EXIT
if [[ -f "$baseline_file" ]]; then
  # `|| true`: a baseline that is all comments (or empty) is the healthy end
  # state, and grep exits 1 on no matches, which `pipefail` would turn into a
  # spurious hard failure of the whole check.
  { grep -vE '^[[:space:]]*(#|$)' "$baseline_file" || true; } | sort -u >"$baseline"
else
  : >"$baseline"
fi

fail=0
checked=0
docs=0
baselined=0

while IFS= read -r doc; do
  docs=$((docs + 1))

  # Way 1: a supersession banner up top exempts the whole document.
  if head -n 18 "$doc" | grep -qE "$banner"; then
    continue
  fi

  # Backticked `*.rs` paths, one per line, with the line they appeared on.
  while IFS=: read -r lineno line; do
    for path in $(printf '%s\n' "$line" | grep -oE '`[A-Za-z0-9_][A-Za-z0-9_./-]*\.rs`' | tr -d '`'); do
      checked=$((checked + 1))
      base="${path##*/}"
      grep -qxF "$path" "$existing_paths" && continue
      grep -qxF "$base" "$existing_names" && continue

      # Way 2: the mention, or the prose immediately around it, says the file
      # is gone. A five-line window is one wrapped paragraph in this corpus.
      if sed -n "$((lineno > 2 ? lineno - 2 : 1)),$((lineno + 2))p" "$doc" \
        | grep -qiE "$declares_death"; then
        continue
      fi

      # Way 3: recorded in the baseline as known and accepted.
      if grep -qxF "$(printf '%s\t%s' "$doc" "$path")" "$baseline"; then
        baselined=$((baselined + 1))
        continue
      fi

      echo "FAIL: $doc:$lineno names \`$path\`, which is not in the tree" >&2
      echo "      $(printf '%s\n' "$line" | sed 's/^[[:space:]]*//' | cut -c1-100)" >&2
      echo "      Fix one of three ways:" >&2
      echo "        - update the path, if the file moved;" >&2
      echo "        - say it is gone on that line (\"deleted\", \"renamed to X\");" >&2
      echo "        - add a supersession banner up top, if the whole doc is history." >&2
      echo "" >&2
      fail=1
    done
  done < <(grep -nE '`[A-Za-z0-9_][A-Za-z0-9_./-]*\.rs`' "$doc" || true)
done < <(
  # Plans, designs, and the superpowers plans (which are plans, in their own
  # directory) -- plus the loose docs at the root of docs/, which are a mix of
  # design notes and analyses that a reader acts on the same way.
  #
  # `docs/results/` and `docs/bugs/` stay out, deliberately: they are dated
  # evidence, and a July audit naming a file deleted in August is a correct
  # record of July rather than a defect.
  {
    find docs/plans docs/designs docs/superpowers -name '*.md'
    find docs -maxdepth 1 -name '*.md' ! -name 'README.md'
  } | sort
)

if [[ "$fail" -eq 0 ]]; then
  echo "OK: $checked source paths cited across $docs plans, designs and notes resolve"
  echo "    ($baselined accepted via $baseline_file)"
fi

exit "$fail"
