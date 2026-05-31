# Function Namespace Audit

_Generated audit of every function in the workspace, characterizing function
names whose prefix is really a **namespace** (a type or a verb-domain) baked
into the identifier instead of being expressed by the module/type system._

- Full machine-readable dump: [`docs/function-audit.tsv`](./function-audit.tsv)
  (6,486 rows: `crate · file · line · name · visibility · context · context_type · is_test`).
- Coordinate decomposition: [`docs/function-coordinates.tsv`](./function-coordinates.tsv)
  (544 production functions with ≥2 namespace coordinates: `verb · coordinate_tokens · qualifier_tokens`).
- Extraction is whitespace/brace-aware and separates `#[cfg(test)]` / `#[test]`
  code from production code.

## 1. Inventory

| Metric | Count |
|---|---|
| Total functions | 6,486 |
| Production | 4,327 |
| Test / bench | 2,159 |
| Methods (`impl`) | 3,194 |
| Trait items | 344 |
| Free functions | 789 |

Production functions by crate:

| Crate | Fns | | Crate | Fns |
|---|--:|---|---|--:|
| pixelflow-core | 1,499 | | pixelflow-runtime | 254 |
| pixelflow-ir | 867 | | pixelflow-pipeline | 238 |
| pixelflow-search | 586 | | pixelflow-compiler | 168 |
| core-term | 317 | | actor-scheduler | 96 |
| pixelflow-graphics | 275 | | (others) | <30 |

Visibility (production): `priv` 2,914 · `pub` 1,300 · `pub(crate)` 72 · `pub(restricted)` 41.

## 2. The model: `name = verb + coordinates + qualifiers`

A function name answers two different questions, and only one of them belongs in
the identifier:

- **verb** — *what does it do?* (`compile`, `emit`, `read`, `handle`). This is the
  semantic core and the only thing that should survive in the bare name.
- **coordinates** — *which one? / for what?* (`arena`, `dag`, `jet`, `scanline`,
  `control`, `u32`, `x86`). These are **namespace coordinates**: they position the
  function in a space. That positioning is the job of the **module path** or the
  **type system** (an associated type, a generic parameter, an enum the function
  matches on). When they pile up in the identifier instead, it means a namespace
  or associated type is *missing* — `compile_arena_dag_jet` is really
  `emit::compile` over `ExprArena`, output-shape and numeric-mode selected by type.
- **qualifiers** — *which variant of the action?* (`with_ctx`, `hoisted`,
  `parallel`). These genuinely modify the verb and may justify distinct entry
  points or a parameter/builder. They are **not** namespaces.

> The earlier "leading-prefix" framing was too weak. A coordinate is a coordinate
> wherever it sits in the name, and *whether or not a type carrying it exists
> today* — its presence in the name is precisely the evidence that the type or
> module ought to exist.

Worked example — the `compile_arena_dag*` family in `pixelflow-ir`, every member
of which takes `arena: &ExprArena, root: ExprId`:

| name | verb | coordinates | qualifiers | should be |
|---|---|---|---|---|
| `compile_arena` | compile | arena | — | `compile` (arena is the param type) |
| `compile_arena_dag` | compile | arena, dag | — | `compile` → `CompileResult` |
| `compile_arena_dag_scanline` | compile | arena, dag, scanline | — | output shape via return type |
| `compile_arena_dag_scanline_hoisted` | compile | arena, dag, scanline | hoisted | strategy = real qualifier |
| `compile_arena_dag_with_ctx` | compile | arena, dag | with_ctx | ctx = parameter |

`arena`, `dag`, `scanline` say *which compile / for what*; `hoisted` and `with_ctx`
say *how*. Strip the coordinates into the type system and the family collapses to
one or two honest functions.

### The latent axes (where coordinates should go)

Counting how many production functions carry each kind of coordinate token
(see `docs/function-coordinates.tsv` for the 544 functions with ≥2 coordinates):

| coordinate axis | fns | structural home |
|---|--:|---|
| numeric repr (`jet`/`field`/`f32`/`u32`/`masked`/`raw`) | 194 | generic / associated type |
| AST node kind (`unary`/`binary`/`ternary`/`nary`/`call`/`ident`/`literal`) | 131 | `match` on `OpKind` (one fn) |
| IR / representation (`arena`/`dag`/`scanline`/`tree`/`graph`/`schedule`) | 129 | module path or return type |
| actor lane (`control`/`management`/`data`) | 47 | per-lane type / trait method |
| architecture (`x86`/`arm`/`neon`/`avx`/`sse`) | 10 | `#[cfg(target_arch)]` module |
| platform (`macos`/`linux`/`x11`/`cocoa`/`wasm`) | 3 | platform module |

Each axis is a concrete refactor: e.g. `read_u32`/`read_f32` → `read::<T>()`;
`push_unary`/`push_binary`/`push_ternary`/`push_nary` → `push(node)` matching on
arity; `handle_control`/`handle_data`/`handle_management` → a `Lane`-parametrized
handler; `set_fp_fast_mode_x86`/`_arm` → `set_fp_fast_mode` in an arch `#[cfg]` module.

### Distribution (production functions)

| coordinate tokens in name | functions |
|--:|--:|
| 0 (already clean) | 2,088 |
| 1 | 1,695 |
| 2 | 481 |
| 3 | 53 |
| 4+ | 10 |

**544 production functions carry ≥2 coordinate tokens** — the bulk-reorg target.

### Sub-patterns of a single coordinate (the leading-prefix cases)

The ≥4-prefix groups from §3 are the special case where the single coordinate is
also the *leading* token; they still resolve via the same three homes:

1. **Type-stutter** — coordinate is the first arg's type:
   `sh2_multiply(a: &Sh2, b: &Sh2)` → `Sh2::multiply`;
   `field_sin(s: NativeSimd)` → method on the SIMD wrapper;
   `emit_addps(code: &mut Vec<u8>, …)` → method on `Assembler(Vec<u8>)`.
2. **Verb redundant with its file** — every fn in one file shares the verb, so the
   module already is the namespace: `encode_*` (19, `aarch64.rs`), `optimize_*`
   (8, `optimize.rs`), `find_*`, `patch_*`.
3. **Homeless domain** — a sub-domain across files with no home module yet
   (`objc_*`, `msg_*`, `wide_*`/`deep_*`, `acc_*`).

Test code uses prefixes deliberately as a grouping convention (`csi_*`, `sgr_*`,
`esc_*`, `utf8_*`, `kubelet_*`, `adversarial_*`); these are descriptive sentences,
**intentional and out of scope** — left as-is.

## 3. Findings — production free-function namespace groups (≥4)

29 prefix groups cover **334 free functions**. Curated by remedy:

### A. Type-namespace → fold into the type (highest value)

| n | pub | crate | prefix | location | first-arg / type | remedy |
|--:|--:|---|---|---|---|---|
| 152 | 87 | pixelflow-ir | `emit_*` | backend/emit/{x86_64,aarch64,mod}.rs | `code: &mut Vec<u8>` | newtype `Assembler(Vec<u8>)`; `emit_addps(code,…)` → `asm.addps(…)` |
| 15 | 12 | pixelflow-ir | `compile_*` | backend/emit/mod.rs | arena/workspace | assoc fns on `CompileWorkspace` |
| 8 | 0 | pixelflow-core | `field_*` | core/src/lib.rs | `NativeSimd` | private methods on the SIMD wrapper |
| 4 | 4 | pixelflow-core | `sh2_*` | combinators/spherical.rs | `&Sh2` / `&Sh2Field` | `impl Sh2 { fn multiply… }` |
| 7 | 7 | pixelflow-runtime | `msg_*` | platform/macos/objc.rs | objc send | `objc::msg` submodule |
| 6 | 6 | pixelflow-runtime | `objc_*` | platform/macos/* | objc runtime | `objc` submodule, drop prefix |

> `emit_*` is the single biggest item: **87 `pub` flat functions** form one
> giant un-namespaced public assembler API across two architectures — exactly
> the surface the project's "minimal public API" rule warns against.

### B. Verb-namespace, single file → drop the prefix (mechanical)

| n | crate | prefix | file |
|--:|---|---|---|
| 19 | pixelflow-ir | `encode_*` | backend/emit/aarch64.rs |
| 8 | pixelflow-compiler | `optimize_*` | optimize.rs |
| 6 | pixelflow-pipeline | `backward_*` | training/unified_backward.rs |
| 5 | pixelflow-pipeline | `parse_*` | training/factored.rs |
| 4 | pixelflow-ir | `patch_*` | backend/emit/aarch64.rs |
| 4 | pixelflow-pipeline | `generate_*` | training/self_play.rs |
| 4 | pixelflow-compiler | `find_*` | codegen/emitter.rs |
| 4 | pixelflow-runtime | `handle_*` | platform/linux/events.rs |

### C. Verb/domain-namespace, multi-file → carve a submodule or make methods

| n | crate | prefix | files | note |
|--:|---|---|--:|---|
| 16 | pixelflow-pipeline | `read_*` | 2 | serde-like; pair with `write_*` in a `codec` module |
| 13 | pixelflow-pipeline | `write_*` | 2 | ″ |
| 8 | core-term | `handle_*` | 2 | dispatch on input/ansi handlers — methods on the handler structs |
| 5 | core-term | `process_*` | 4 | spread across emulator — candidate trait method |
| 5/5 | pixelflow-pipeline | `wide_*` / `deep_*` | 2 | network-shape domains, no home module |
| 4 | core-term | `create_*` | 3 | factory verbs — assoc `::new`/builder methods |
| 4 | pixelflow-search | `extract_*` / `const_*` | 2 | e-graph extraction submodule |

## 4. Method stutter (names that repeat their own `impl` type)

Only ~30 genuine cases (after excluding trait-mandated names like `From::from`,
`Div::div`). The type already namespaces them, so the prefix is redundant:

| impl | redundant methods | fix |
|---|---|---|
| `CostModel` | `cost`, `cost_by_kind`, `cost_by_name`, `node_cost`, `depth_cost` | drop `cost`/`_cost` |
| `EdgeAccumulator` | `add_edge`, `remove_edge`, `*_with_pe` | `add`, `remove` |
| `DepsAnalysis` | `node_deps`, `compute_class_deps`, `leaf_deps_and_children` | drop `deps` |
| `ExprNnue` | `compute_expr_embed`, `forward_expr_only`, `backprop_expr_proj` | drop `expr` |
| `Screen` | `enter_alt_screen`, `exit_alt_screen` | `enter_alt`, `exit_alt` |
| `ScanlineJitManifold` | `eval_scanline` | `eval` |
| `Sh2Field`/`Field` | `from_sh2`, `from_field` | covered by `From` |

This is small and low-risk — good warm-up candidates.

## 5. Proposed bulk reorganization plan

The work is organized by **latent axis** (from §2), because each axis is a single
structural decision applied to many functions at once. Each phase is independently
shippable and test-gated (`cargo test --workspace` after each). Ordered by
value-to-risk.

**Phase 0 — guard rails.** Commit the dumps + this report. Add an xtask/clippy lint
that rejects new function names carrying ≥2 coordinate tokens (re-uses
`scripts/function_audit.py`), so names don't re-accumulate namespaces while we clean up.

**Phase 1 — collapse the `compile_*` / `emit_*` IR families (biggest win).**
`pixelflow-ir/src/backend/emit/`. Two moves:
- Introduce `Assembler(Vec<u8>)`; fold the 152 `emit_*` free fns (87 `pub`) into
  methods, collapsing the flat assembler API into one type.
- Collapse the `compile_arena_dag_scanline*` family: drop `arena` (it's the param
  type), express output shape (`dag`/`scanline`) via the return type, keep only
  real qualifiers (`hoisted`, `with_ctx`). Target: `emit::compile`.

**Phase 2 — AST-node-kind axis → enum dispatch (131 fns).** `push_unary` /
`push_binary` / `push_ternary` / `push_nary`, `fold_unary`/`fold_binary`/…,
`parse_*` node variants. Replace the per-variant functions with one function that
matches on `OpKind`/arity. Concentrated in `pixelflow-ir`, `pixelflow-compiler`,
`pixelflow-pipeline`.

**Phase 3 — numeric-repr axis → generic / associated type (194 fns).**
`read_u32`/`read_f32`/`write_f32` → `read::<T>()`/`write(v: T)` via a small codec
trait; `field_*`/`jet_*` SIMD ops → methods on the wrapper; `select_raw`/`add_masked`
mode tokens → keep private (these honor the "no public raw_*" rule — verify none leak).

**Phase 4 — actor-lane axis → type/trait (47 fns).** `handle_control` /
`handle_data` / `handle_management` and `drain_control_and_management` → a
`Lane`-parametrized handler or per-lane methods, in `actor-scheduler` + the
`pixelflow-runtime`/`core-term` consumers.

**Phase 5 — architecture / platform axis → `#[cfg]` modules (13 fns).**
`set_fp_fast_mode_x86`/`_arm`, `objc_*`/`msg_*` → `platform::macos::objc`. Small,
mechanical, and aligns with the existing backend module layout.

**Phase 6 — single-coordinate leftovers (the §3 prefix groups).** Mechanical
prefix-strip where the file is already the module (`encode_*`, `optimize_*`,
`patch_*`, `find_*`), type-stutter folds (`sh2_*`→`impl Sh2`), and homeless-domain
submodules (`wide_*`/`deep_*`/`acc_*`), plus the method-stutter table in §4.

**Phase 7 — re-audit.** Re-run `scripts/function_audit.py`; confirm the ≥2-coordinate
count drops from 544 and no new flat `pub` namespaces appeared.

### Execution notes
- The biggest lever is the type system, not renaming: most coordinates disappear by
  introducing one generic, enum-match, or associated type that serves dozens of fns.
- The API-surface changes (Phases 1, 3) touch `pub` items and directly serve the
  repo's "minimal public API" rule — review these carefully.
- Re-run the committed extractor at any time to regenerate both TSVs and diff
  coordinate counts before/after each phase.
