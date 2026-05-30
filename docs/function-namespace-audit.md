# Function Namespace Audit

_Generated audit of every function in the workspace, characterizing function
names whose prefix is really a **namespace** (a type or a verb-domain) baked
into the identifier instead of being expressed by the module/type system._

- Full machine-readable dump: [`docs/function-audit.tsv`](./function-audit.tsv)
  (6,486 rows: `crate · file · line · name · visibility · context · context_type · is_test`).
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

## 2. What "a namespace in the name" means here

A function name carries a namespace when a leading token duplicates something
the type system could express:

1. **Type-namespace (stutter)** — the prefix is a type that is *also the type of
   the first/main argument*. The receiver should carry the namespace.
   - `sh2_multiply(a: &Sh2, b: &Sh2)` → `Sh2::multiply` / `a.multiply(b)`
   - `field_sin(s: NativeSimd) -> NativeSimd` → method on the SIMD wrapper
   - `emit_addps(code: &mut Vec<u8>, …)` → method on an `Assembler(Vec<u8>)`
2. **Verb-namespace** — the prefix is a verb shared by every fn in *one file*.
   The file (module) already is the namespace, so the prefix is pure redundancy.
   - `encode_*` (19, all in `aarch64.rs`), `optimize_*` (8, `optimize.rs`),
     `find_*` (4, `emitter.rs`), `patch_*` (4, `aarch64.rs`).
3. **Domain-namespace** — the prefix names a sub-domain spread across files that
   has no home module yet (`objc_*`, `msg_*`, `wide_*`/`deep_*`, `acc_*`).

Test code uses prefixes deliberately as a grouping convention (`csi_*`, `sgr_*`,
`esc_*`, `utf8_*`, `kubelet_*`, `adversarial_*`, `round_trip_*`); these are
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

Ordered by value-to-risk. Each step is independently shippable and test-gated
(`cargo test --workspace` after each).

**Phase 0 — guard rails.** Commit the dump + this report. Add a CI/clippy check
(or xtask lint) that flags new `pub fn <verb>_*` free functions, so the surface
doesn't regress while we clean up.

**Phase 1 — assembler API (biggest win).** Introduce `Assembler(Vec<u8>)` (or
extend the existing code-buffer type) in `pixelflow-ir/src/backend/emit/`.
Convert the 152 `emit_*` free fns to methods, collapsing 87 `pub` symbols into
one type. This is the dominant public-API-surface reduction. Do it per-arch
(x86_64, then aarch64) to keep diffs reviewable.

**Phase 2 — single-file verb prefixes (mechanical).** Strip redundant prefixes
where the file is already the module: `encode_*`, `optimize_*`, `patch_*`,
`find_*`, `generate_*`, `backward_*`, `parse_*`. Pure rename; rust-analyzer /
`cargo fix`-style find-and-replace.

**Phase 3 — type-stutter folds.** `sh2_*`→`impl Sh2`, `field_*`→SIMD-wrapper
methods, `compile_*`→`CompileWorkspace` assoc fns, the method-stutter table in §4.

**Phase 4 — homeless domains → submodules.** `objc`/`msg` → `platform::macos::objc`;
`read_*`/`write_*` → a `corpus::codec` module; `wide_*`/`deep_*`/`acc_*` into
named network-shape modules; core-term `handle_*`/`process_*` onto their handler
structs.

**Phase 5 — re-audit.** Re-run the extractor; confirm group counts dropped and no
new flat `pub` namespaces appeared.

### Execution notes
- Renames are the safe 80%; keep them prefix-strip-only and let the type/module
  carry the namespace.
- The API-surface changes (Phase 1, 3) are the ones worth careful review — they
  touch `pub` items and directly serve the repo's "minimal public API" rule.
- Re-run `python3` extractor (committed alongside) to regenerate the TSV at any
  time and diff the namespace groups.
