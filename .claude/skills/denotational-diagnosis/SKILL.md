---
name: denotational-diagnosis
description: Go from symptoms to the missing model of a thing, then let the code fall out. Use when code needs cleaning up, a review or desloppify run turns up many findings in one module, a design keeps growing special cases or escape hatches, or someone asks why a piece of the codebase is a mess.
---

# Denotational diagnosis

Findings are symptoms. Many of them in one place are usually one wrong model,
not many bugs. Fix the model and the symptoms disappear; fix the symptoms one
at a time and the model grows new ones.

This is CLAUDE.md's "denote before you build" run backwards, on code that
already exists: recover what the thing *means*, then hold the code to it.

## The steps

1. **Collect symptoms. Fix none yet.** desloppify findings; registers, ids or
   resources chosen outside the component that owns them; special cases
   defended in a comment ("none of this goes through the schedule because…");
   identities formatted into strings; an earlier stage told in advance what a
   later one will need; invariants kept by convention.

2. **Name the thing** in its domain's words, as a practitioner would: "an
   assembler", "a register allocator", "instruction selection", "a terminal
   emulator". Not what the code calls it.

3. **Denote it without looking at the code.** What is an X? What are its
   parts, and what do practitioners call them? What does it do, how does it
   work, how does it behave — inputs, outputs, guarantees, what is true of
   every correct one? Write this before rereading the code: the code's model
   is the suspect, and reading it first borrows it.

   > An assembler: a program is a sequence of items — instructions, labels,
   > directives — in sections. An instruction's arguments are operands, and an
   > operand may be a register, an immediate, an address or a label. Registers
   > come in classes; the flags are one of them.

4. **Translate the shape.** For each part of the description: where is it in
   the code — a type, a function, a convention in a comment, or nowhere? What
   does the code have that the description does not?

   > Labels: a 31-byte string, not a key. Instructions: bytes written inline by
   > ~236 functions, not values. Flags: nowhere. The constant pool: a data
   > section with no name.

5. **Diagnose.** State the missing or wrong concept as what the thing is —
   "a label is an operand", "instruction selection is a phase". It should
   explain many symptoms at once, across files. If it explains one, look
   deeper.

6. **Follow the consequences to closure.** This is the step that gets skipped,
   and skipping it is how a right plan becomes wrong code. Ask "and then
   what?" until nothing else falls out.

   > If a label is keyed by the node it names, a loop header is a position.
   > Then everything around it — the trip test, the step, the accumulate — is
   > an ordinary def. Then the loop verbs, the reserved temps and the alias
   > check have nothing left to do. The 2026-09-10 plan named the first step;
   > the code stopped there and kept the old model around the new type.

7. **Refuse exceptions.** "No allocation outside the allocator, except x16" is
   not a rule with an exception; it is a symptom the diagnosis has not
   explained yet. Each refused exception forces a deeper model — here: the
   allocator owns its own spill code, so it allocates the address register
   too.

8. **Subtract.** The change should delete more than it adds. What the new
   model makes unrepresentable goes; what it makes ordinary stops being
   special.

9. **Write the denotation down, then build.** It goes in `docs/plans/` before
   code, and the implementation is obliged to it. After it lands, reread the
   plan against the code: did it carry every consequence, or stop at the
   first?

## Signs the diagnosis is right

- One sentence explains symptoms in several files.
- The fix is mostly deletion.
- Special cases become ordinary ones: a loop's combine becomes a `Binary` def.
- Questions disappear instead of being answered: there is no "which temp?"
  once every register an instruction touches is an operand.

## Signs it is not

- The fix adds a mechanism beside the old one.
- A comment still explains why something need not go through the general path.
- The model needs an exception.

## Tools

- `desloppify`'s diagnose step runs steps 2–6 for each module whose findings
  converge; its description call never sees the code (step 3).
- Its `trait-first` (bypassed general path), `invariant-in-comment` (identity
  in a string) and `phases-in-order` (predictions, late decisions) rules are
  tuned to the symptoms above.
