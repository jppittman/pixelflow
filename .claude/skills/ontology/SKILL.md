---
name: ontology
description: What each thing in this project IS — label, operand, instruction, register, value, block, loop, phase, mask, If — with what it is not, what follows from it, and where it lives in the code. Read before designing, reviewing or cleaning up code that touches one of these terms, and whenever a special case appears: the deeper model is usually here.
---

# Ontology

Each entry says what a thing **is**, what it **is not**, what **follows**
from that — the consequences code must carry all the way, not just the first
one — and where it **lives** in the code, with the plan that decided it. When
code treats a term as something other than its entry says, the code is
wrong or the entry is; either way, say so. A special case beside one of these
is usually a place the definition's consequences were not followed.

## Label

- **Is:** the name of an address — an abstraction over a position in a
  program, nothing more. Things *have* labels: a block has one, the constant
  pool (a data section) has one. A label is minted with the thing that lives
  at its address, so it always names something and no two things share one.
- **Is not:** a string; an instruction; a field on an instruction; a sum of
  the things that can be labelled (`Label::ConstPool | Head(..) | Join(..)`
  inverts the dependency — the assembler would know codegen's vocabulary, and
  every new labelled thing would edit the label type); derived from the id of
  what it names (two sibling folds carry one `ValueId`).
- **Follows:** an operand can be a label, so a branch is an ordinary
  instruction whose argument is a label. The assembler maps labels to
  addresses and owns nothing else about them. Because the label is born with
  its block, the code that creates a block holds its label — no map from
  sites to labels, and no label bound twice or bound nowhere. A loop header is
  its block's label. A label is a name, so it is `Copy`; it has no public
  constructor, so it cannot dangle; it is 64-bit, like every id.
- **Lives:** `emit::Label` — today a 31-byte string built by `format!`, which
  contradicts this entry. History: R0 of
  `docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md` made a label an
  item and its reference an operand; "A label should be keyed by the node it
  names" in the same plan removed the site-to-label map by keying on the
  node — minting the label with the block removes it without the label
  knowing about nodes.

## Assembly program

- **Is:** one per kernel: a value made of sections, each a sequence of
  items — instructions, and label bindings — with the constant pool as a data
  section. It is the namespace of its labels, and it mints them.
- **Is not:** a buffer of bytes; split per scope and spliced together (one
  program, one namespace — a scope is the allocator's concept, not the
  assembler's).
- **Follows:** whatever builds the program — instruction selection, then the
  allocator inserting its own code — builds a value; nothing writes bytes.
- **Lives:** to be built. Today `emit_scope` assembles each scope into its
  own `Assembly` and the parent splices the bytes in (`emit/mod.rs`), which
  contradicts this entry.

## Assembler

- **Is:** a function from an assembly program to binary: lay the sections
  out, map each label to an address, encode each instruction given the
  addresses of the labels it names. Here, a very small in-memory one: no
  object files, no relocations or symbols beyond labels.
- **Is not:** stateful; a builder; a code buffer other code writes bytes
  into; aware of registers, values, scopes or the IR.
- **Follows:** it stands alone — its module imports nothing from the rest of
  the crate, and everything above it depends on it, never the reverse.
- **Lives:** `emit::Assembly` (a push/bind/finish builder with a public `code`
  field the driver writes into) and `AsmProgram` in `emit/mod.rs` — two front
  ends where the program is the value and `assemble` the function; today's
  builder contradicts this entry.

## Operand

- **Is:** an argument of an instruction: a register (a value with an access —
  read, written, both — and a constraint), a label, an immediate, an address
  (with register operands inside), or a frame slot.
- **Is not:** split into special methods by kind (no `target()` beside the
  operands).
- **Follows:** each phase reads the operand kinds it owns — the allocator
  binds registers and frame slots, the assembler resolves labels, the encoder
  writes all of them. A block's successors are the label operands of its last
  instruction.
- **Lives:** to be built (`docs/plans/2026-10-08-selection-is-a-phase.md`).

## Instruction

- **Is:** an operation of the machine applied to operands: a value, chosen by
  instruction selection, encoded last.
- **Is not:** bytes written into a buffer by the code that chose it.
- **Follows:** every register an instruction touches is one of its operands —
  including the ones it clobbers and the flags — so nothing about it is hidden
  from the allocator.

## Register

- **Is:** a finite resource of a machine, in a class (vector, general purpose,
  mask, flags). A resource, not a name.
- **Is not:** `Copy`; constructible anywhere; reserved by convention; chosen
  by an encoder.
- **Follows:** registers are tokens minted once, by the backend's register-file
  declaration, and owned by the allocator, which binds values to them by move
  and lends them to encoders by borrow. Two live values in one register, or a
  register chosen outside the allocator, are compile errors. The ABI is
  initial ownership: the entry block's parameters own the argument registers.
  There is no allocation outside the allocator, and no exception.
- **Lives:** today `Reg`/`Gpr`/`PtrReg`/`KReg(pub u8)`, `Copy` — contradicts
  this entry. Escape hatches: `docs/plans/2026-09-01-register-allocation-escape-hatches.md`.

## Value

- **Is:** a name for something computed: one definition, its reads, a class.
- **Is not:** a register; a resource.
- **Follows:** names copy, resources move — a value may be `Copy`, a register
  may not.

## Temp

- **Is:** nothing. A "temp" is a value with a short life, or a register some
  stage needed and could not ask the allocator for.
- **Follows:** where code reserves temps (`temps_for`, `REDUCE_TEMPS`), a
  phase is missing — the stage that would have made them values runs after
  the one that needed them (see Phase).

## Block

- **Is:** a label, its parameters, and a sequence of instructions ending in
  one whose label operands are the block's successors.
- **Follows:** a value live into a block from more than one predecessor is a
  block parameter; that is what a phi is.

## Loop

- **Is:** a block whose last instruction may branch back to its own label. A
  fold that survives optimization is a loop.
- **Is not:** an instruction; a scaffold of verbs around a body.
- **Follows:** the header is the block's label. The binder and accumulator are
  block parameters. The trip test, the step and the accumulate are ordinary
  instructions. Nothing about a loop needs reserved registers.
- **Lives:** decided in `docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md`
  (R1: "the accumulate is an ordinary `Binary` def … the allocator learns
  nothing about folds"); today emitted by verbs in `emit/mod.rs`'s `Reduce`
  arm, which contradicts it.

## Phase

- **Is:** a function from one representation to the next that owns the
  decisions about its output.
- **Is not:** told in advance what a later phase will need; deciding what an
  earlier phase owns.
- **Follows:** a count, reservation or size handed to an earlier stage by a
  later one's needs means a phase is missing between them. Codegen's phases:
  IR → instruction selection (instructions over values) → register allocation
  (and frame layout, and its own spill code) → assembly (labels, sections) →
  encoding.

## Mask

- **Is:** a bit pattern: all-ones for true, all-zeros for false.
- **Is not:** a number. `1.0` is not true.
- **Lives:** CLAUDE.md, "Floating point at the edges".

## If

- **Is:** dispatch — `if m then a else b`, two cases.
- **Is not:** a blend; a fold.
- **Follows:** a uniform mask takes an arm (a jump); the bitwise blend is the
  path for a mask that varies by lane. The arms are blocks.
- **Lives:** CLAUDE.md, "`If` contains an if".
