---
name: ontology
description: What each thing in this project IS (kernel, kernel!, uniform, binding time, binder, fold, monoid, range, mask, If, arm, Var, Ref, identity, lattice, manifold, collapse, lane, pipeline P, stage vs tier, saturation, budget, extraction, cost, variance, demand, scope, placement, root, park, carry, slot, label, operand, instruction, register, value, temp, block, loop, phase, guard, priority lane, cell, glyph, font program), along with what it is not, what follows from it, and where it lives in the code. Read it before designing, reviewing or cleaning up code that touches one of these terms. Read it whenever a special case appears, because the deeper model is usually here.
---

# Ontology

Each entry says what a thing **is**, what it **is not**, what **follows**
from that, and where it **lives** in the code, with the plan that decided it.
"Follows" means the consequences code must carry all the way, not just the
first one. When code treats a term as something other than its entry says,
either the code is wrong or the entry is. In both cases, say so. A special
case beside one of these terms usually marks a place where the definition's
consequences were not followed.

How to read the entries:

- **Latest decision wins.** When documents disagree, the latest decision is
  the entry. Earlier meanings go under **Is not**, with the date that
  retired them, so a reader who meets the old wording in a plan, an agent
  file or a comment knows it is stale.
- **Lives** names the type or module and the document that decided it. A
  "today … contradicts this entry" note marks code or prose known to be
  behind the decision.
- **Homonyms** are marked. One word in this project often names several
  unrelated things, and each sense is listed separately: tier, region, slot,
  shape, root, guard, arm, block, phase, frame, lane, park, label, fold,
  uniform, schedule. Before you reason from a word, ask which sense it is.

The entries are in one file per part. Read the part a term belongs to, not
the whole ontology; a term's homonyms may live in another part, so check the
index for the word first.

## Index

### Terms about the other terms — [meta.md](meta.md)

- Exception (special case, carve-out, escape hatch)
- A convention written in a comment
- Control plane and data plane
- Fold and dispatch (style). Homonym of the IR's Fold

### The language and its IR — [language.md](language.md)

- Kernel
- kernel! (the language)
- Entry, helper, record, Args
- Binding time (a name bound later)
- Structural parameter
- Uniform (the parameter)
- Uniform (adjective: lane-, batch-, row-, frame-uniform). Homonym
- Kernel-typed parameter
- Application (contramap, `at`)
- Coordinate (X, Y)
- Var
- Param
- Binder
- Fold (Reduce)
- Monoid (SEQ)
- Range (trip count)
- Unrolling
- Mask
- If
- Constant (Const) and its two domains
- Dwrt (derivative)
- Ref, unit and link (composition)
- Identity
- Table, Buffer, Gather, Broadcast
- Library (vs primitive)
- Transcendental expansion
- Totality
- ExprArena (rooted term)
- OpKind numbering
- Retired: Field, the combinator tier, the integral

### The lattice and evaluation — [lattice.md](lattice.md)

- Lattice
- Shape (extent). Homonym
- Manifold (BoundManifold, bind, bake)
- CompiledKernel
- Collapse
- Write and Seq (effects)
- Lane, batch, width (SIMD). Homonym of the actor lane
- pack (strip-mine)
- Execution rule (stripe)
- DiscreteManifold (tabulation)
- Origin
- Index range, band (and Union)
- Derived range and domain split (lowering 1)
- Region. Homonym
- Scene (channel kernels, pack, colour)
- Pixel
- Pull-based rendering and the fixed observer

### The compiler and search — [compiler.md](compiler.md)

- P (the pipeline)
- Stage (open value, closed program)
- Tier. Homonym
- Program, template, instance
- Pass, legalize, Optimize
- Legality
- Decline
- Rule set and vocabulary
- E-graph (e-node, e-class, IR trait)
- Hash-consing
- Saturation
- Budget (and class cap)
- Extraction
- Tie-break
- Cost model (latency prior, fold price)
- Reranker
- Variance
- Demand
- Widening and narrowing
- Compile cache key (canonical)
- Numeric contract (precision versus range)
- External oracle and the gate
- Guide (candidate, growth)
- Provenance and the hindsight label. Homonym of codegen's Label
- Corpus, fence and held-out
- Cost label, sentinel, journal entry, registration

### Codegen — [codegen.md](codegen.md)

- Phase. Homonym
- Lowering
- Schedule (Def). Homonym
- Scope and the nest
- Placement (Span, Point). Homonym
- Root. Homonym
- Hoist (LICM)
- Park. Homonym of the actor park
- Carry
- Back edge (head reconciliation)
- Value
- Class
- Pointer (Context)
- Constant pool (constants in codegen)
- Register
- Temp
- Spill, reload, remat
- Slot. Homonym
- Frame. Homonym
- Arm ownership. Homonym of Arm
- Guard. Homonym
- Mispredict bound
- Loop
- Block. Homonym
- Label. Homonym of the hindsight label and the cost label
- Operand
- Instruction
- Branch
- Assembly program
- Assembler
- Emitter
- Scaffold (retired)
- ISA tier (vector width)

### The runtime and actors — [runtime.md](runtime.md)

- Priority lane (doorbell). Homonym of the SIMD lane
- Actor and troupe
- Driver (PlatformOps)

### The terminal — [terminal.md](terminal.md)

- Host versus program
- Cell and the cell grid
- Frame (display)
- Font program and id tree
- Glyph
- Piece, band, box
- Coverage and the closed form
- Contour
- Geometry on the host
- Run
- Atlas
- Zoom level
