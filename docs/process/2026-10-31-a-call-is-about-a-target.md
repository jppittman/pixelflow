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
Crate     = a manifest m; members(m) = { f | crate_of(f) = m }
Module    = (dir, root : Path + None, intent : Path + None, members : FinSet Path)
Located   = Crate × Module                      what a file resolves to
Resolved  = Located + Unresolved                I1
Layout    = ModFile | Sibling | CrateEntry      how root is spelled
Conflict  = ModFile × Sibling                   I3: both exist
Scope     = File | Crate
Target    = File(f, Located) | Crate(Crate)     select(scope, f)
Section   = Path × (LineNo ⇀ Text) × Context    one shown file
Rendering = List Section × FinSet Path          sections, omitted
```

`μ(resolve) : Tree → Path → Resolved + Conflict`.
`μ(select) : Scope → Path → Resolved → Target`.
`μ(cover) : Target → FinSet Path`.
`μ(render) : Target → Budget → Rendering`.
`μ(cite) : Rendering → (Path × LineNo) ⇀ Text`.
`μ(cluster) : Finding → Module = module(target(finding))`.

### What is true, before any law is written

The eight invariants of pass 1 are not eight facts. They are consequences
of six pieces of structure the problem already has, and each piece of
structure names the type that carries it. A law that follows from the
structure needs a test; a law the type makes unrepresentable needs none.

**S1. The ancestors of a path form a chain.** The file tree under the
prefix order is a rooted forest, so for any file `f` the set
`anc(f) = {f, parent(f), parent(parent(f)), …}` is totally ordered. Hence
"the nearest ancestor satisfying `P`" is `max {a ∈ anc(f) | P(a)}` — a
maximum over a chain — and is therefore **unique or absent, never
ambiguous**. Every walk the problem needs is this one operation with a
different `P`: the crate is `nearest(is_manifest)`, the module root is
`nearest(has_root)`, the intent is `nearest(has_intent)`. I1's "exactly
one" is not a rule to enforce; it is what a maximum over a chain *is*.

  *Carried by:* one function `nearest : (Path → Bool) → Path → Option<Path>`
  over one chain. Four walks (`crate_of`, `containing`, `root_outline`,
  `intent`) are four spellings of it, and three of them can disagree only
  because they are not it.

**S2. Containment is an interval on the chain.** I2 — "a module root is
never past the crate boundary" — says `module_of(f)` lies in the interval
`[crate_of(f), f]` of the chain. The module walk is `nearest(has_root)`
*on the chain truncated at the manifest*, not on the whole chain. A root
found above the manifest is not a root the problem recognises.

  *Carried by:* `resolve` computes the chain once, cuts it at the first
  manifest, and finds root and intent on the cut chain. Past-the-boundary is
  unrepresentable; L2 needs no runtime check. (`intent` today walks up and
  stops at `Cargo.toml`; `root_outline` looks only at the file's own
  directory and its sibling; `containing` walks all the way up with no
  boundary at all. Three different truncations of the same chain.)

**S3. `crate_of` and `module_of` are quotient maps, and one refines the
other.** A function `g : Files → X` partitions its domain into fibres
`g⁻¹(x)`. `crate_of` partitions files into crates; `module_of` partitions
them into modules; and because `module_of(f)` is found on the chain cut
at `crate_of(f)`, every module fibre lies inside one crate fibre: the
module partition **refines** the crate partition. So

```
Files ──module_of──▶ Modules ──crate_of_module──▶ Crates
  └────────────────── crate_of ─────────────────────┘      commutes
```

I8 — a finding's module equals its target's module — is the statement that
clustering uses *this* quotient map and no other. Today `identify`
clusters by `containing(path)` and `synthesize` outlines by the thing's
`component`, two maps that agree on most inputs and not on a nested
directory without a root file.

  *Carried by:* `Located = Crate × Module` is the pair of fibre
  representatives, computed once; `cluster` is the projection
  `Located → Module`, a field read. Two call sites cannot read two maps.

**S4. Root layout is a cardinality split, not a choice.** For a directory
`d` the root candidates are `R(d) = {d/mod.rs, sibling(d)} ∩ Files` (and at
a crate root, the entry file). "The root of `d`" is a *function* only if
`|R(d)| ≤ 1`. The problem's own word for `|R(d)| = 2` is *conflict*: the
relation is not functional there, and the answer is to say so. Any
expression that picks one — `first().or(sibling)` — is a non-injective map
from `{0, 1, 2}` candidates onto `{none, one}` and discards exactly the
case the practitioner needs to see.

  *Carried by:* the sum `Option<Layout> + Conflict`, i.e.
  `Result<Option<Layout>, Conflict>`: three outcomes for three
  cardinalities. I3 is the type's shape, not a guard. This is the one law
  the current code actively violates.

**S5. A rendering is a restriction, not a reindexing.** A file is a total
function `lines(f) : [1..n] → Text`. What a model is shown of it is a
**sub-function** — an outline elides bodies, a budget elides tails — so a
`Section` is a partial function `LineNo ⇀ Text` whose graph is a subset of
`lines(f)`'s graph. "Absolute line numbers" means the domain is the
original index set, never a renumbering from 1; "each line a model cites
maps back" (L5) means `cite(r, (p, n)) = lines(p)(n)` whenever `(p, n) ∈
dom(r)`. And `cite` has the same shape as the numbering check the module
already does for findings, things and diagnoses: **a model's reference is a
position in what it was shown**, and validity is membership in the domain.
L5 is the finding-number law applied to lines.

  *Carried by:* `Section.lines : Vec<(u64, String)>` rather than a
  `String` with a `first_line` offset (today's `Snippet`, which encodes a
  reindexing and so cannot be looked up); `cite` is a `get`; a reply's
  `(path, line)` is resolved through it exactly as a `FindingIx` is
  resolved through `Numbering`, and fails its reply by name when absent.

**S6. Rendering is a monoid homomorphism from a set of files.** `Rendering`
under concatenation of sections is a monoid (the free monoid on `Section`,
paired with set union on `omitted`). Rendering a crate target is the fold
of a one-file renderer over the members in path order:

```
render(Crate(c), b) = ⊕_{f ∈ sort(members(c))} render_file(f, b_f)
```

Two things fall out. First, *each member appears once* (I5) because
`members(c)` is a **set** — a list built as `roots ++ ranked` can hold the
root twice, and `dedup()` on neighbours is an apology for that list not
being a set. Second, I6 — nothing omitted silently — is the partition
equation `shown(r) ⊎ omitted(r) = cover(T)`: a disjoint union that equals
the cover. And I7, byte-identical text for the same target, is simply that
`render` is a function of `(Tree, Target, Budget)` and nothing else; it
needs no `dedup` and no journal-stabilising comment to stay true.

  *Carried by:* `Rendering` constructed only by `render`, which partitions
  `cover(T)` into `sections` and `omitted`; the header and gutter are
  written by `Rendering::to_prompt` and nowhere else; `render` takes a
  `&Tree` snapshot and touches no `std::fs`, so purity (L4, L7) is visible
  in the signature.

### Laws

Each law names the structure it follows from and how it is held.

| Law | Statement | From | Held by |
|---|---|---|---|
| L1 one place | `∀ f ∈ Files. ∃! r. resolve(f) = r` | S1 | the type: `nearest` over a chain |
| L2 containment | `crate_of(f) ≤ module_of(f) ≤ f` on the chain | S2 | the type: the chain is cut at the manifest |
| L3 no guess | `\|R(d)\| = 2 ⟹ resolve = Conflict` | S4 | the type: `Result<Option<Layout>, Conflict>` |
| L4 purity | `resolve` is a function of `Tree` | S1, S6 | signature `fn(&Tree, …)`; test |
| L5 round trip | `(p, n) ∈ dom(r) ⟹ cite(r, p, n) = lines(p)(n)` | S5 | `Section` is a restriction; test |
| L6 nothing silent | `shown(r) ⊎ omitted(r) = cover(T)` | S6 | `render` partitions; test |
| L7 determinism | `render(T, b)` is a function | S6 | signature; test replaces the golden comment |
| L8 cluster = context | `module(φ) = module(target(φ))` | S3 | `Located` is read, never recomputed |
| L9 section stable | `section_f(render(Crate c)) = section_f(render(File f))` | S6 | one `render_file`; test |

Four laws are discharged by types and need no runtime guard; five are
pinned by one test each (pass 3). None is kept by a comment.

### Type class morphisms

`Target` has an equality, and it is the equality of its meaning: two
targets are equal iff they cover the same files with the same context.
Since `cover` and context are functions of the target's fields, derived
structural `Eq`/`Ord` *is* that equality. It is the planner's key, so "two
rules share a call" is `target₁ = target₂ ∧ seen₁ = seen₂`, with no side
map for crates.

`Rendering` is a monoid and `render` on a crate target is the monoid
homomorphism from the sorted member list that extends `render_file` (S6).
A crate rendering is *made of* file renderings, so a crate target cannot
have a second way of writing a header.

`cover` is monotone: `cover(File(f, at)) = {f} ⊆ members(at.krate) =
cover(Crate(at.krate))`. Targets are partially ordered by coverage and the
crate target is the join of its file targets. `render` is monotone over
the same order — a functor between the two posets — which is a law of its
own:

- **L9 (a file's section is the same section everywhere).**
  `section_f(render(Crate(c), b)) = section_f(render(File(f, at), b_f))`
  for every `f ∈ members(c)` not omitted. A file rendered alone and the
  same file inside its crate are byte-identical, so a journal key or a
  prompt cache for the one is a hit for the other. Pinned by a test
  (`a_file_rendered_inside_its_crate_is_the_file_rendered_alone`). Today
  `under_headers` and `shape` cannot satisfy it: they format independently.

This is why a crate-wide call is the same sort of thing as a file call,
one step up, and not a side path. The code will spell the functor as a
`for` loop over a `BTreeSet` calling `render_file`; nothing in it says
"functor", and nothing needs to.

### What is not a concept

- **A bare path as the identity of a call.** A path is the name of a file;
  a target is the place a question is about. A call holds a target.
- **"Crate-wide" as a flag or a side path.** It is `Scope::Crate`, a value
  of the same sort as `Scope::File`, selected once (S3's order).
- **A pre-numbered string.** Lines carry their numbers as data (S5); the
  gutter is written by `to_prompt` and nowhere else.
- **The current directory.** A file under no manifest is `Unresolved`, a
  value, not a fallback to wherever the process happens to be.
- **A root outline walk apart from resolution.** The root is a field of
  the module `resolve` returned (S1).
- **A journal key kept stable by rendering a file twice.** L7 keeps keys
  stable; duplication is a workaround for a renderer that was not a
  function (S6).
- **A second walk with its own boundary.** There is one chain and one cut
  (S2).

### The opinion in the types

| Structure | Type that carries it | What becomes unrepresentable |
|---|---|---|
| S1 one chain | `fn nearest(pred, path) -> Option<Path>`; `resolve` calls it thrice on one chain | two walks disagreeing |
| S2 interval | the chain is cut at the manifest before root/intent are sought | a root past the crate boundary |
| S3 refinement | `Located { krate, module }`, one value per file; `cluster` is a projection | clustering and context reading different maps |
| S4 cardinality | `Result<Option<Layout>, Conflict>` | silently preferring `mod.rs` |
| S5 restriction | `Section { path, lines: Vec<(u64, String)>, context: bool }`; `cite` is a lookup | a cited line that was never shown |
| S6 homomorphism | `Rendering` built only by `render` over a `BTreeSet<Path>` | a file shown twice; an omission unsaid |
| purity | `resolve`/`render` take `&Tree`; `Tree::scan` is the one `std::fs` reader | a result that depends on cwd or on when it ran |
| `Unresolved` | a variant of `Resolved` | `PathBuf::new()` meaning "no crate" |
| `Target` | an enum of two variants, `Eq + Ord` derived | a call whose place is a comment |

Prioritised by whether the wrong value is *silently* representable: S4
(wrong root, plausible prompt), S3 (wrong cluster, plausible review) and
S5 (wrong line, plausible finding) are the three the types must take.
Purity and the interval are cheap to carry and are carried; the rest are
one test each.

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
| L9 | `a_file_rendered_inside_its_crate_is_the_file_rendered_alone` |

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
replaces. One golden change, nine law tests.

### Signs

- One sentence — *a call is about a target* — explains D2, D3 and D4 (ten
  findings across `module.rs`, `plan.rs`, `mod.rs`, `reply.rs`, `run.rs`).
- The fix is mostly deletion.
- Three special cases become ordinary: the 2018 sibling is a `Layout`; the
  crate group is an arm; the twice-outlined root is a `Vec` with one entry.
- Two questions disappear: "which walk?" (there is one) and "is `Reply.path`
  set?" (it is the target's kind).

Nothing above needs an exception.
