---
name: denotational-design
description: Design a thing by saying what it means before building it, in three passes — Abbott (the problem in plain English; its nouns, verbs and categories), Elliott (a meaning for each noun, a law for each verb, the opinion in the types), Pike (code in which the algebra is invisible and the result looks obvious). Use forward, before writing anything new, and backward whenever symptoms converge — a desloppify review's roots, a simplify run that keeps finding the same recurring parameters or name family, five functions that pass the same loose pieces around, a design that keeps growing special cases, or someone asking why a module is a mess. Replaces denotational-diagnosis.
---

# Denotational design

Design is choosing the denotation; the implementation is then obliged to it
(CLAUDE.md, "Denote before you build"). This skill is that rule as a
procedure, in three passes named for the people whose method each one is.
The passes run in order and pull in opposite directions, and that is the
point: each refuses something the one before it would have let through.

| Pass | Voice | In | Out | Refuses |
|---|---|---|---|---|
| 1 | Russell Abbott | the problem | a glossary | code, and the code's own names |
| 2 | Conal Elliott | the glossary | a denotation | a type without a meaning, an operation without a law |
| 3 | Rob Pike | the denotation | code | a mechanism the laws did not force |

There are two ways in. **Forward:** something new is to be built; start at
pass 1 with the task. **Backward:** code exists and symptoms converge on it;
start at pass 0, then run passes 1–3 on the *problem*, never on the code.

## Pass 0 — Backward entry: symptoms

Findings are symptoms. Many of them in one place are usually one wrong
model, not many bugs. Fix the model and the symptoms disappear; fix the
symptoms one at a time and the model grows new ones.

1. **Collect symptoms. Fix none yet.** desloppify findings and roots; a
   simplify run's recurring cleanups; the same three parameters travelling
   through five functions; a name family (`s3_upload`, `s3_copy`,
   `s3_delete`); a resource chosen outside the component that owns it; a
   special case defended in a comment; an identity formatted into a string;
   an earlier stage told what a later one will need; an invariant kept by
   convention.

2. **Name the thing** in its domain's words, as a practitioner would: "an
   object store", "a register allocator", "instruction selection", "a
   terminal emulator". Not what the code calls it.

3. **Do not reread the code before pass 2.** The code's model is the suspect,
   and reading it first borrows it. Pass 1 describes the *problem* the code
   is solving, as if the code did not exist. desloppify's `identify` stage
   already does this: the describing call is never shown the code.

Then run passes 1–3. Pass 4 is where the code comes back in.

## Pass 1 — Abbott: the problem in English

Abbott's method (*Program Design by Informal English Descriptions*, 1983):
describe the problem in excruciating, plain, complete English, then read
the description as data.

Write it as a practitioner would explain it to a careful newcomer — what it
is, what it is part of, what comes in, what goes out, what must always be
true, what can go wrong. Long is fine. No code, no identifiers, no
"the struct".

Then harvest, in a table:

| From the prose | Becomes a candidate |
|---|---|
| a noun | type |
| a category of nouns ("a bucket, a key and a version are all parts of an address") | sort, or an algebra over a type |
| a verb with a subject and an object | operation, with its signature |
| "always", "never", "must", "exactly one", "the same as" | invariant |
| two nouns that are always said together | one type, not two parameters |
| a noun that is only ever said with another noun's name in it | a part, not a thing |

The output is a glossary: every noun, what category it is in, every verb
with the nouns it relates, every invariant in one sentence. Nothing in it
mentions the code.

> *An object store holds objects. An object is addressed by a key, and a key
> is unique within a bucket. Uploading puts an object at a key; copying
> makes the object at one key also present at another; deleting makes a
> key hold nothing. After a copy the two objects are equal.*
>
> Nouns: object, key, bucket, address (a bucket and a key, always together).
> Verbs: upload(address, object), copy(address, address), delete(address).
> Invariants: an address names at most one object; copy then read at the
> target equals read at the source.

The table is not the design. It is the input to pass 2, which may merge,
split or reject any row. A noun that is only a convenience of the prose is
not a type.

## Pass 2 — Elliott: the denotation

Elliott's method (*Denotational Design*): for each type, say what a value
*means* as a mathematical object, `μ : T → M`; for each operation, say what
it does to the meaning; make every operation a homomorphism. The meaning is
chosen so that the obvious instances are forced.

1. **A meaning for each noun.** State `μ` in one line. A kernel is a
   function `R² → Real`; an address is a pair; an object store is a partial
   function from addresses to objects. If you cannot state it without naming
   the implementation, it is not a concept yet — go back to pass 1 or drop
   it. `docs/DENOTATION.md` is the model of what this looks like done.

2. **A law for each verb.** Write each operation as an equation on
   meanings: `μ(copy(s, a, b)) = μ(s)[b ↦ μ(s)(a)]`. An operation with no
   law has no reason to exist in this type.

3. **Type class morphisms.** If the type is a `Num`, a monoid, a functor,
   its meaning must be one too, and the instance is then forced:
   `μ(a + b) = μ(a) + μ(b)`. This is what makes "is this operation right?"
   stop being a question.

4. **Say what is not a concept.** A memory read, a lane, a batch, a
   "bucket name" apart from its key. Each one the denotation leaves out is
   one the code may not reintroduce. `docs/DENOTATION.md` keeps a list.

5. **Put the opinion in the types.** A law the compiler can enforce should
   be one. Resources move, names copy: a register, a slot, a handle is a
   token that is not `Copy`, minted once by its owner. Two nouns always said
   together are one struct. A value with a documented range is an enum or a
   newtype. Prioritise by whether a wrong value would be *silently*
   representable; one that panics on the next line can wait.

6. **Follow the consequences to closure.** Ask "and then what?" until
   nothing else falls out. If an address is a value, every function that
   took a bucket and a key takes an address, and the five that were kept
   apart by their prefixes become methods of one type. The step that gets
   skipped is this one, and skipping it is how a right design becomes
   wrong code.

7. **Refuse exceptions.** "No allocation outside the allocator, except x16"
   is not a rule with an exception; it is a consequence not yet followed.
   Each refused exception forces a deeper meaning.

8. **Write it down before building.** A denotation page in `docs/<area>/`
   dated like its neighbours, and the module's `intent.md` updated to cite
   it: what the thing is, its parts, its laws, what it is not. The
   implementation is obliged to this page; where they disagree later, the
   page wins or is changed on purpose.

## Pass 3 — Pike: the code

Pike's sensibility (*Go Proverbs*, *Notes on Programming in C*): clear is
better than clever; a little copying is better than a little dependency;
the bigger the interface, the weaker the abstraction. The teammate who
reads the code should find it obviously right without needing to know an
algebra chose it.

1. **The algebra is invisible.** The denotation lives in the page and the
   `intent.md`; the code has a struct, some methods and a test. No
   `Homomorphism` trait, no `Semantics` module, no comment saying "this is
   a monoid". If the code needs the comment, the shape is wrong.

2. **Subtract.** The change should delete more than it adds. What the
   meaning makes unrepresentable goes; what it makes ordinary stops being
   special. Reach for a new abstraction only after subtraction has failed.

3. **Concrete until something varies.** No trait with one impl, no generic
   with one instantiation, no `Box<dyn>` on a hot path. When the second
   implementation arrives, extract the trait around the first — that is the
   style guide's "trait first", and it is a pass-3 move, not a pass-2 one.

4. **The laws are the tests.** Each equation from pass 2 is one test, named
   as the law reads: `copy_then_read_at_the_target_equals_read_at_the_source`.
   A law with no test is a law the next change will break.

5. **Plain names.** The nouns from pass 1, as the practitioner says them.
   `Address`, not `BucketKeyPair`; `copy`, not `s3_copy_object_between_keys`.

## The tie-break

Elliott's will, Pike's hands. Elliott decides what structure *exists* —
whether this is a monoid, a functor, a quotient map — and that decision is
final; Pike decides what it is *called* and how it is spelled, and that is
final too. A pass-3 simplification that breaks a law is rejected; a pass-2
abstraction the laws did not require is rejected; but a structure the laws
*do* require is built, and the hands do not get to leave it out because
naming it would sound clever. The functor is a `for` loop over a `BTreeSet`
calling `render_file`, and nothing in the code says "functor". When they
disagree, ask which side the disagreement is on — meaning, or spelling —
and that side's voice wins.

In Rust the hands are dtolnay's rather than Pike's, because Rust has
traits where Go has none, and a trait is where an Elliott instance lands.
serde is the worked example: a Church-encoded traversal that nobody using
it has had to call one, and `anyhow` hides an error algebra behind `?`.
Pike's rules below still hold; the exemplar of applying them to Rust is
that code.

## Pass 4 — Backward only: hold the code to it

Only now reread the code. For each part of the denotation: where is it — a
type, a function, a convention in a comment, or nowhere? What does the code
have that the denotation does not? Each gap is one change, and the changes
are mostly deletions.

Signs the model is right: one sentence explains symptoms in several files;
the fix is mostly deletion; special cases become ordinary; questions
disappear instead of being answered.

Signs it is not: the fix adds a mechanism beside the old one; a comment
still explains why something skips the general path; the model needs an
exception.

## Tools

- `desloppify`'s pipeline is passes 0–2 in code: `identify` names each
  converging module's thing and describes it without the code; `synthesize`
  diagnoses, roots, and writes for each root a **brief** — the pass-1
  harvest already drafted from the evidence. A review whose roots carry
  briefs is an instruction to run this skill, forward, from those briefs.
- Its `function-shapes` (recurring arguments, name families, missing
  types), `trait-first` (bypassed general path), `invariant-in-comment`
  (identity in a string, overloaded type) and `phases-in-order` rules are
  tuned to the pass-0 symptoms.
- `/simplify` finds the same recurring shapes in a diff. When its cleanups
  keep touching one type's parameters or one name family, stop simplifying
  and run this skill on that thing.
