# Intent lives with the module

**Status:** `Plan of record` (JP, 2026-10-15: one `intent.md` per module
directory; fold the crate agent personas into it).
**Verified against:** `8c9a2d08315d3b82476dcb56ca22ba9281c56603`.

## 0. The decision

Every module directory carries an `intent.md`: what the module *is*, in the
denotation's words; what it alone decides; what it never does or knows; its
contract; what enforces each of those; and where the code is behind it. One
file, beside the `mod.rs` whose contract it explains, or beside `Cargo.toml`
for a crate whose root is `src/lib.rs`.

The intent this captures already exists and is already enforced, in pieces:
CLAUDE.md's constraints, `docs/process/STYLE.md`, the eleven `.claude/agents/*.md`
personas, the desloppify rules and skills, the ontology, and forty plans.
What none of them gives is one place, per module, that a reader or a review
rule can open and find the whole of what that module promises. "No register
allocation outside the register allocator" is a plan, a rule, an ontology
entry and a module doc today; it should be one line in
`pixelflow-codegen/src/emit/regalloc/intent.md`, and the rule should cite it.

This follows Anthropic's AI-native SDLC play "capture intent" in its
outline: a human-readable, machine-actionable intent file, accepted before
design starts, kept beside what it describes. The play's page
(`academy.claude.com/courses/ai-native-sdlc-playbook/capture-intent`) was
not reachable from the environment when this was written; the format below
is ours and is adjusted to the play's when it is read.

## 1. The file

```
# <module path>

**Is.** One sentence, in the words of docs/DENOTATION.md: what this module is.

## Owns
What this module alone decides or holds: a resource, a choice, a representation.

## Never
What it does not do and does not know. Each line is a statement a review can
test against the code ("no function outside `regalloc/` names a register").

## Contract
Where the contract is (`mod.rs`, or the traits), and its public entries, one
line each, so drift between the list and the outline is visible.

## Enforced by
For each line above: the type that makes the wrong thing unrepresentable, the
desloppify rule, the CI script, or the test. A line with nothing here is debt.

## Behind
Where the code is behind the intent today, dated, with the plan that closes it.

## Decided in
The plans and rulings.
```

Short: under sixty lines. Prose explains; the headings are fixed so a tool can
read the sections. A module whose `mod.rs` doc already states the contract
does not restate it; the Contract section points and lists.

## 2. Where

| module root | intent |
|---|---|
| `<crate>/src/lib.rs` or `main.rs` | `<crate>/intent.md` |
| `<dir>/mod.rs` | `<dir>/intent.md` |

Every crate in the workspace and every `mod.rs` directory under `src/` gets
one; `tests/support` and `tests/common` directories do not. A flat file that
offers crate items with no `mod.rs` above it (`pixelflow-ir/src/*.rs`,
`pixelflow-compiler/src/*.rs`) is a module with no contract yet, as the
behavioral-contracts skill says; its intent lives in the crate's file until
it becomes a directory.

## 3. The personas fold in

`.claude/agents/<crate>.md` for `actor-scheduler`, `core-term`,
`pixelflow-core`, `pixelflow-graphics`, `pixelflow-ml` and
`pixelflow-runtime` are crate intent written as a role: "Crate Purpose",
"What Lives Here", "Invariants You Must Maintain", "Anti-Patterns". Each
folds into its crate's `intent.md` (purpose → **Is**, invariants and
anti-patterns → **Never**, what lives here → **Contract**) and is deleted;
what is stale in them (`origin: [f32; 2]`, `backend/`) is dropped rather
than carried. The role personas (`algebraist`, `language-mechanic`,
`numerics`, `refactor`, `reviewer`) are not module intent and stay.

## 4. Machine-actionable

Two hooks, both small:

1. **desloppify reads the intent** (built). A rule's `"context"` takes
   `"module_intent"`: each call also sees the nearest `intent.md` up the
   reviewed file's directory tree within its crate (`review/module.rs`, beside
   `root_outline`), under the same "for context only" framing.
   The module-shape rules (`mod-rs-is-a-contract`,
   `interface-lives-in-mod-rs`, `behavior-through-the-trait`,
   `registers-come-from-the-allocator`, `phases-in-order`) take it, so a
   finding is judged against what the module says of itself, not against a
   workspace-wide skill.
2. **CI checks presence and paths.** `scripts/check-intent.sh`: every crate
   and every `src/**/mod.rs` directory has an `intent.md`; every backticked
   `.rs` path in one resolves (the doc-paths check extended to them); every
   `## Contract` entry names an item the module root declares.

## 5. What shrinks

CLAUDE.md's "Critical Constraints" keep the workspace-wide rules (terminal
logic, SIMD, the control plane, denote before you build, floating point) and
point at the module intent for the module-specific ones. The desloppify
skills `pixelflow-architecture` and `style-guide` stay as the distillations
they are (46 and 44 lines against CLAUDE.md's 584 and STYLE.md's 277): a
rule prompt that carried the whole of either would pay for text the rule
never asks about, which is the opposite of what a skill is for. `ontology`
is a symlink because the rules want the entries themselves. The ontology's
**Lives** entries point at the module's intent beside the plan that decided
it.

## 6. Order

1. This note and two exemplars, by hand:
   `pixelflow-codegen/src/emit/regalloc/intent.md` and `pixelflow-ir/intent.md`.
2. The remaining 45 modules, one writer each against the template and the
   exemplars, one reviewer each, landed per crate.
3. The personas folded and deleted with their crate's PR.
4. The two hooks, and the CLAUDE.md and skill shrink, as one PR after.
