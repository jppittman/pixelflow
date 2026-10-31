# A call is about a target

**Status:** denotation, 2026-10-31. Forward pass of
`.claude/skills/denotational-design` from root R1 of a desloppify review of
`desloppify/src/review` (20 findings, 5 things, 2 roots; the brief is quoted
in pass 1). Not built yet; the implementation is obliged to this page.

Passes 1 and 2 were written from the brief before the code was reread. Pass
4 is where the code comes back in.

---

## Pass 1 — Abbott: the problem in English

A rule-driven reviewer asks questions about code. Every question is asked
about *somewhere*: this file, or this crate. Where a question is asked
decides two things — what the model is shown, and which findings later
count as being about the same place.

A Rust project is a tree of files. Some directories hold a package
manifest; the files under a manifest, down to but not into any nested
manifest, are that manifest's crate. Every file is in at most one crate.
Within a crate, files are grouped into modules: a module is a directory of
files together with the one file that declares it, its root. The root is
spelled in one of two layouts — a `mod` file inside the directory, or a file
of the directory's name beside it — and a crate's entry file is the root of
its top directory. A module may also carry an intent document, a short
page saying what the module is and never does. The root and the intent are
never past the crate boundary: a module belongs to the crate its files
belong to.

A rule names a scope: it is judged on a file, or on a crate. Given a file
and a scope, there is one target — the file itself, or the crate that
contains it. A target covers member files: one for a file target, all of
the crate's for a crate target. For a file target the reviewer may also
show context that is not under review — the module root's outline and the
intent document — so that the model knows what the file is part of.

To ask a question, the reviewer renders the target: each member file
appears once, under a header naming it, with its own lines and their
original one-based numbers, so a line the model cites maps back to a real
line of a real file. The context, when shown, is marked as context. A
budget bounds how much is rendered; what the budget leaves out is said, and
the file it came from is named. Rendering is a function of the target and
the budget and nothing else, so two rules that look at the same target the
same way are shown byte-identical text, and a journal keyed on the text
replays.

Findings come back against a path and a line. A finding is about the
module of the target it was found in, and findings that converge on a
module are what the next stage names. So resolving a file to its module
must give one answer, the same answer every time, from the file tree and
the manifests alone.

What can go wrong: a file under no manifest, which belongs to no crate; a
directory with both layouts of root, which is a conflict and never a guess;
a rendering that leaves a file out without saying so; a cited line that
names no shown line; and a finding whose module disagrees with the context
that was shown for it.

### The harvest

| Noun | Category | Candidate |
|---|---|---|
| project | the tree | the set of files with their manifests |
| crate, crate boundary (nearest manifest) | place | type |
| module, module directory, module root, intent document | place; root and intent are parts of a module | type with parts |
| layout (mod-file, sibling, crate-entry) | the ways a root is spelled | sort |
| file | place; a member | type |
| scope (file, crate) | kind of place a rule is judged on | sort |
| target | a scope applied to a file: the place a question is about | type |
| member file; context (root outline, intent) | what a target shows; context is not under review | part of a target |
| snippet, outline, line, line number, header, omission notice | shown text: renderings of member files | parts of a rendering |
| budget | bounds a rendering | value |
| rendering | the text a model is shown | type |
| unresolved | a file in no crate | a value of resolution, not an error |

| Verb | Relates |
|---|---|
| resolve | file → crate, module, root, intent |
| select | scope × file → target |
| cover | target → member files |
| render | target × budget → rendering |
| cite | rendering × (path, line) → a shown line |
| cluster | findings → by module of target |

| Invariant | |
|---|---|
| I1 | every file resolves to exactly one module in exactly one crate, or is unresolved |
| I2 | a module root is in the file's directory or is its directory's sibling, never past the crate boundary |
| I3 | both layouts present is a conflict, never a guess |
| I4 | resolution depends only on the tree and the manifests |
| I5 | a rendering shows each member file once, under one header, with its original lines and absolute numbers |
| I6 | what a rendering omits for budget is said, naming the file |
| I7 | the same target rendered the same way is byte-identical text |
| I8 | a finding's module is the module of the target it was found in |

Nouns always said together: "root and intent" (both parts of a module);
"crate and module" (a located file has both). A noun only said with another's
name in it: "crate boundary" (a crate's), "module root" (a module's) — parts.

---

## Pass 2 — Elliott: the denotation

### Meanings

```
Tree      = (Files : FinSet Path, manifests : FinSet Path)
Crate     = a manifest m; members(m) = { f | nearest manifest above f = m }
Module    = (dir, root : Path + None, intent : Path + None, members : FinSet Path)
Located   = Crate × Module                      what a file resolves to
Resolved  = Located + Unresolved                I1
Layout    = ModFile | Sibling | CrateEntry      how root is spelled
Conflict  = ModFile × Sibling                   I3: both exist
Scope     = File | Crate
Target    = File(f, Located) | Crate(Crate)     select(scope, f)
Rendering = List (Header × List (LineNo × Text)) × List Omitted
```

`μ(resolve) : Tree → Path → Resolved + Conflict`. It is a *function of the
tree* (I4): it reads no clock, no cwd, no cache.

`μ(select) : Scope → Path → Resolved → Target`. A file scope's target is the
file with its location; a crate scope's target is the location's crate.

`μ(cover) : Target → FinSet Path`. `cover(File(f, _)) = {f}`;
`cover(Crate(c)) = members(c)`. Context — the root's outline and the intent —
is *not* in `cover`; it rides on a file target as what is shown but not
reviewed.

`μ(render) : Target → Budget → Rendering`, a function, which is I7 for free:
same inputs, same bytes. No other input exists.

`μ(cite) : Rendering → (Path, LineNo) → Text + NotShown`. The line a model
names is looked up in what was rendered, not in the file on disk.

`μ(cluster) : Finding → Module = module(target(finding))`, I8.

### Laws

- **L1 (one place).** `resolve` is total on `Files` and single-valued:
  `∀ f ∈ Files. ∃! r. resolve(f) = r`.
- **L2 (containment).** `resolve(f) = Located(c, m) ⟹ f ∈ members(m) ∧
  members(m) ⊆ members(c) ∧ root(m) ∈ dir(f) ∪ {sibling(dir(f))}`.
- **L3 (no guess).** `ModFile(d) ∈ Files ∧ Sibling(d) ∈ Files ⟹ resolve(f)
  = Conflict` for every `f` under `d`.
- **L4 (purity).** `resolve` is a function `Tree → Path → _`; two calls on
  the same tree agree. (Equivalent: it may be memoised, and nothing observes
  the difference.)
- **L5 (round trip).** `(h, (n, t)) ∈ render(T, b) ⟹ lines(path(h))[n] = t`.
  Each path appears in exactly one header; each `n` under it at most once.
- **L6 (nothing silent).** `f ∈ cover(T) ∧ f ∉ headers(render(T, b)) ⟹ f ∈
  omitted(render(T, b))`.
- **L7 (determinism).** `render` is a function; stated as a law so a test
  pins it: `render(T, b) = render(T, b)` across two constructions of `T`
  from the same tree.
- **L8 (clustering agrees with context).** For a finding `φ` from target
  `T`: `module(φ) = module(T)`, and the context shown for `T` was
  `root(module(T))`, `intent(module(T))`.

### Type class morphisms

`Target` has an equality, and it is the equality of its meaning: two
targets are the same iff they cover the same files with the same context.
That equality is the planner's key, so "two rules share a call" is
`target₁ = target₂ ∧ seen₁ = seen₂`, with no side map for crates.

`Rendering` is a monoid under concatenation of headed sections, and
`render(Crate(c))` is the fold of `render` over `members(c)` in path
order — a crate rendering is *made of* file renderings, so a crate target
cannot have a second way of writing a header.

### What is not a concept

- **A bare path as the identity of a call.** A path is the name of a file;
  a target is the place a question is about. A call holds a target.
- **"Crate-wide" as a flag or a side path.** It is `Scope::Crate`, a value
  of the same sort as `Scope::File`, selected once.
- **A pre-numbered string.** Lines carry their numbers as data in a
  `Rendering`; the gutter is written by `render` and nowhere else.
- **The current directory.** A file under no manifest is `Unresolved`, a
  value, not a fallback to wherever the process happens to be.
- **A root outline walk apart from resolution.** The root is a field of
  the module `resolve` returned.
- **A journal key kept stable by rendering a file twice.** L7 keeps keys
  stable; duplication is a workaround for a renderer that was not a
  function.

### The opinion in the types

- `Target` is an enum with two variants; `Scope` is an enum with two
  variants; "which is it" is a `match` at the one place a call is built.
- `Located` holds the crate *and* the module, so a finding's module is a
  field read and cannot disagree with the context (L8 by construction).
- `Conflict` is a variant of resolution's result, not a log line.
- `Unresolved` is a variant, not `PathBuf::new()` or `"."`.
- `Rendering` holds `(Header, Vec<(LineNo, Text)>)`, so a cited line is a
  lookup that can fail by name (L5), and the only `format!` of a gutter is
  inside `render`.
- `Budget` is a newtype; a rendering under a budget records what it omitted.

### Consequences to closure

If a call holds a target, then: the planner's key is `(Seen, Surroundings,
Target)` and there is one `Call` constructor. Then the crate group needs no
side map and is not "built only after the last file is read". Then
`Call.path`, `Call.snippet`, `Call.root`, `Call.intent` collapse to
`Call.target` and `Call.shown: Rendering`. Then the header, the gutter and
the omission notice are written once, in `render`, so `shape`, `omitted`
and `under_headers` are one function or gone. Then `Source.text`'s
"numbered, as a snippet is" convention has nothing to describe. Then
`containing`'s crate-wide special case is the `Crate` arm of `cover`. Then
the 2018 sibling is a `Layout` value, not a comment. Then a `Reply.path`
that is "set when the code shown came from several files" is `Option` by
the target's kind: present for `Crate`, absent for `File`, and checked
against `cover(T)`. Then the duplicate root outline kept "so journal keys
stay stable" is deleted, because L7 keeps them stable. Then a finding's
module is `target.module()`, so `identify`'s clustering and `synthesize`'s
outlines read the same field and cannot disagree. Nothing else falls out.

### Exceptions refused

- "A crate-wide call gets no root or intent whatever its context" — not an
  exception: a crate target *has* no single module, so it has no context
  of that shape. It is the `Crate` arm of `cover`, and the `Behind` line in
  `review/intent.md` that records it as a known gap is deleted.
- "The root with symptoms but not the most is outlined twice" — not a
  budget quirk to pin by golden: a rendering shows each member once (L5).
  The golden changes on purpose.

---

## Pass 3 — Pike: the shape of the code

No trait, no generic, no new module. One file, `review/target.rs`, replaces
`review/module.rs`; it is the only file that touches the file tree for
context, and the only one that writes a header or a gutter.

```rust
/// Where a file is: its crate and its module. Resolved once, from the
/// tree, and read everywhere else.
pub(super) struct Located {
    pub(super) krate: PathBuf,      // the directory holding the manifest
    pub(super) module: Module,
}

pub(super) struct Module {
    pub(super) dir: PathBuf,
    pub(super) root: Option<PathBuf>,   // mod.rs | lib.rs | main.rs | sibling
    pub(super) intent: Option<PathBuf>,
}

pub(super) enum Resolved {
    Located(Located),
    Unresolved,                     // under no manifest
}

pub(super) enum Target {
    File { path: PathBuf, at: Located },
    Crate { krate: PathBuf, members: Vec<PathBuf> },
}

/// The text a model is shown for a target: headed sections, each line
/// with its absolute number, and what was left out.
pub(super) struct Rendering {
    pub(super) sections: Vec<Section>,
    pub(super) omitted: Vec<PathBuf>,
}
pub(super) struct Section { path: PathBuf, lines: Vec<(u64, String)>, context: bool }

pub(super) fn resolve(path: &Path) -> Result<Resolved>;   // Err: Conflict
pub(super) fn target(scope: Scope, path: &Path, at: Resolved) -> Target;
pub(super) fn render(t: &Target, context: Surroundings, budget: Budget) -> Result<Rendering>;
impl Rendering { fn to_prompt(&self) -> String; fn cite(&self, path, line) -> Option<&str> }
impl Target { fn module(&self) -> Option<&Module>; fn covers(&self, path) -> bool }
```

`Call` becomes `Call { rules, target, shown: Rendering }`. The planner's
key becomes `(Seen, Surroundings, Target)`; `Target` derives `Eq + Ord` on
its fields, which is the equality of its meaning. The crate map and the
second `Call` literal go. `identify::clusters` keys on
`call.target.module()`; `synthesize::outlines` calls `render` on
`Target::File` per module with the symptom files, and the one `Rendering`
type carries the omission notice. `Reply.path` is read through
`target.covers(path)`, so a model that cites a file the call did not show
fails its reply by name, like a bad finding number does.

Names: `Target`, `Located`, `Module`, `Rendering`, `resolve`, `render`,
`cite`, `covers` — the nouns of pass 1. Not `TargetResolver`, not
`PromptFormatter`.

### The laws as tests (`desloppify/tests/target.rs`)

| Law | Test name |
|---|---|
| L1 | `every_file_in_the_tree_resolves_to_one_module_in_one_crate_or_is_unresolved` |
| L2 | `a_module_root_is_in_the_files_directory_or_its_sibling_and_inside_the_crate` |
| L3 | `a_directory_with_both_root_layouts_is_a_conflict_not_a_guess` |
| L4 | `resolving_the_same_file_twice_gives_the_same_location` |
| L5 | `every_line_a_rendering_shows_is_that_line_of_that_file` |
| L6 | `a_file_a_rendering_leaves_out_is_named_as_omitted` |
| L7 | `two_targets_built_from_the_same_tree_render_byte_identical_text` |
| L8 | `a_findings_module_is_the_module_of_the_target_it_was_found_in` |

L4 and L7 were kept by the golden transcript and a comment; they become
tests that say what they pin. The golden still holds every prompt, and it
changes once, on purpose: the duplicated root outline (I5) goes.

---

## Pass 4 — Holding the code to it

The code reread after passes 1–3 (`module.rs`, `plan.rs`, `mod.rs`,
`reply.rs`, `run.rs`, and the callers in `identify.rs`/`synthesize.rs`).

| Denotation | In the code today | Change |
|---|---|---|
| `resolve : Path → Resolved` | four walks: `root_outline` (own dir + sibling), `intent` (up to `Cargo.toml`), `crate_of` (up to `Cargo.toml`, cwd fallback), `containing` (up to a dir with a root file, `is_dir` special case) — `module.rs:42,66,90,100` | one walk returning `Located`; the four are deleted |
| `Unresolved` | `PathBuf::new()` from `crate_of`/`containing` — "the current directory if there is none" | a variant |
| `Conflict` (L3) | `root_files(dir).into_iter().next().or(sibling)` — mod-file silently wins | `Err(Conflict)` |
| `Target` | `Call.path`, "the file, or for a crate-wide rule the crate's directory" by comment — `mod.rs:46`; `containing`'s `is_dir` branch recovers which | enum; comment and branch deleted |
| `Scope::Crate` as a value | `Seen::Parts(_, Group::Crate)` matched mid-loop to divert into a side map `crates`, flushed after the loop — `plan.rs:34,58,85` | the `Crate` arm of `target`; one `Call` literal |
| `render` writes the header once | `== path ==` in `shape`, `omitted`, `under_headers` — `module.rs:118,164`; `plan.rs:96` | one function |
| `render` writes the gutter once | `format!("{:>5} \| {line}")` in `intent` and in `snippet::outline` — `module.rs:73` | `Rendering::to_prompt` |
| `Rendering` holds numbered lines | `Source.text: String`, "Numbered, as a snippet is" — `mod.rs:61`; `Snippet.numbered: String` | `Vec<(u64, String)>`; the convention has nothing to describe |
| L5 cite | `Reply.path.unwrap_or_else(\|\| call.path)` — `run.rs:214`; no check that `line` was shown | `target.covers` + `shown.cite`; a bad cite fails the reply |
| L6 omission | `omitted(path)` pushed by `shape` only; `under_headers` has no budget | one `omitted` list in `Rendering` |
| L7 determinism | kept by the golden and by `files.dedup()` "only neighbours: journals key on this exact prompt" — `module.rs:147` | a function; the `dedup` and the comment go, the golden moves once |
| L8 | `identify` clusters by `module::containing(path)`; `synthesize` outlines by `shape(component, …)` — two walks that can disagree on a nested dir without a root | both read `target.module()` |
| context not under review | `Call.root`, `Call.intent: Option<Source>` — two fields filled by two walks under a `match context` each — `plan.rs:66–75` | `Section.context: bool` in the one `Rendering` |

What the code has that the denotation does not: `root_files`' ordering
comment ("path order, not the order the names are listed in"), the
`is_dir` test in `containing`, the `crates` side map, `under_headers`,
`Source`, and the `Behind` line in `review/intent.md` about crate-wide
calls having no context. All deleted.

Net: `module.rs` (≈175 lines) and `plan.rs`'s crate path (≈30) go;
`target.rs` is one walk, one render, two small enums — smaller than what it
replaces. One golden change, eight law tests.

### Signs

- One sentence — *a call is about a target* — explains D2, D3 and D4 (ten
  findings across `module.rs`, `plan.rs`, `mod.rs`, `reply.rs`, `run.rs`).
- The fix is mostly deletion.
- Three special cases become ordinary: the 2018 sibling is a `Layout`; the
  crate group is an arm; the twice-outlined root is a `Vec` with one entry.
- Two questions disappear: "which walk?" (there is one) and "is `Reply.path`
  set?" (it is the target's kind).

Nothing above needs an exception.
