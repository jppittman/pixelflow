//! How selection writes a function, one block at a time.
//!
//! A leaf module: the constructors of [`Def`], [`Early`], [`Tie`] and
//! [`Pending`] are private to it, so a definition can only be made by a
//! [`Builder`], and a [`Function`] only leaves one through
//! [`Builder::finish`], which asserts the invariants its doc names.

use super::{
    Block, Class, ClassId, Constant, Constants, Entry, Function, IsaBackend, Label, Labels, Loop,
    Operand, Pushed, Scope, Selected, Spill, Target, Value, ValueName, operands,
};
use crate::error::CompileError;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// The right to define one fresh value of class `C`.
///
/// Affine: only [`Builder::def`] mints it, and placing it in an instruction's
/// write field consumes it. A second definition of a value is therefore
/// unrepresentable. Its name is readable so that selection can return it.
#[must_use = "a value whose definition is dropped is read but never written"]
pub(super) struct Def<C: Class> {
    value: Value<C>,
}

impl<C: Class> Def<C> {
    pub(super) fn value(&self) -> Value<C> {
        self.value
    }
}

/// A definition the instruction writes before it has finished reading: its
/// register is none of this instruction's reads' registers, not even one
/// whose last use is here. Example: `vgatherdps`'s destination.
#[must_use = "a value whose definition is dropped is read but never written"]
pub(super) struct Early<C: Class> {
    value: Value<C>,
}

impl<C: Class> Early<C> {
    pub(super) fn value(&self) -> Value<C> {
        self.value
    }
}

/// A read that the instruction overwrites in place, and the right to name
/// what it leaves there: `vfmadd231ps`'s addend, NEON `BSL`'s mask, `imul`'s
/// destination, a gather's mask.
#[must_use = "a value whose definition is dropped is read but never written"]
pub(super) struct Tie<C: Class> {
    read: Value<C>,
    write: Value<C>,
}

impl<C: Class> Tie<C> {
    pub(super) fn read(&self) -> Value<C> {
        self.read
    }

    pub(super) fn write(&self) -> Value<C> {
        self.write
    }
}

/// A block minted but not yet entered. Affine; [`Builder::finish`] panics on
/// one never entered.
#[must_use = "a block that is never entered is bound nowhere"]
pub(super) struct Pending {
    label: Label,
    params: Vec<ValueName>,
    scope: Scope,
}

impl Pending {
    pub(super) fn label(&self) -> Label {
        self.label
    }

    /// The parameters, as the allocator names them.
    pub(super) fn params(&self) -> &[ValueName] {
        &self.params
    }
}

impl Entry {
    fn of([ctx, out, pitch]: [ValueName; 3]) -> Self {
        Self {
            ctx: ctx.typed(),
            out: out.typed(),
            pitch: pitch.typed(),
        }
    }
}

/// A block's successors: the `Target` operands of its last instruction.
fn targets<I>(block: &Block<Pushed<I>>) -> impl Iterator<Item = &Target> {
    block
        .insts
        .last()
        .into_iter()
        .flat_map(|pushed| &pushed.operands)
        .filter_map(|operand| match operand {
            Operand::Target(target) => Some(target),
            Operand::Reg { .. } | Operand::Frame(_) => None,
        })
}

/// Whether the block's last instruction branches: nothing follows it.
fn ended<I>(block: &Block<Pushed<I>>) -> bool {
    targets(block).next().is_some()
}

/// How selection writes a function, one block at a time.
pub(super) struct Builder<B: IsaBackend> {
    labels: Labels,
    /// One entry per value minted: its class, and whether its definition has
    /// been pushed. A value's id is its index.
    classes: Vec<ClassId>,
    defined: Vec<bool>,
    /// In layout order. The last is open.
    blocks: Vec<Block<Pushed<B::Inst<Selected>>>>,
    loops: Vec<Loop>,
    constants: Constants<B::Constant>,
    interned: BTreeMap<B::Constant, Constant>,
    /// Blocks minted and not yet entered.
    pending: usize,
}

impl<B: IsaBackend> Builder<B> {
    /// A function whose entry block is open, and its three parameters.
    pub(super) fn new() -> (Self, Entry) {
        let mut labels = Labels::default();
        let constants = Constants {
            label: labels.mint(),
            entries: Vec::new(),
        };
        let head = labels.mint();
        let mut builder = Self {
            labels,
            classes: Vec::new(),
            defined: Vec::new(),
            blocks: Vec::new(),
            loops: Vec::new(),
            constants,
            interned: BTreeMap::new(),
            pending: 0,
        };
        let params =
            [ClassId::Pointer, ClassId::Pointer, ClassId::Integer].map(|c| builder.mint(c));
        builder.defined.fill(true);
        builder.open(head, params.to_vec(), Scope::Body);
        (builder, Entry::of(params))
    }

    fn mint(&mut self, class: ClassId) -> ValueName {
        let id = self.defined.len() as u64;
        self.classes.push(class);
        self.defined.push(false);
        ValueName { id, class }
    }

    fn fresh<C: Class>(&mut self) -> Value<C> {
        self.mint(C::ID).typed()
    }

    fn open(&mut self, label: Label, params: Vec<ValueName>, scope: Scope) {
        self.blocks.push(Block {
            label,
            params,
            insts: Vec::new(),
            scope,
        });
    }

    pub(super) fn def<C: Class>(&mut self) -> Def<C> {
        Def {
            value: self.fresh(),
        }
    }

    #[expect(dead_code, reason = "live from B8")]
    pub(super) fn early<C: Class>(&mut self) -> Early<C> {
        Early {
            value: self.fresh(),
        }
    }

    pub(super) fn tie<C: Class>(&mut self, read: Value<C>) -> Tie<C> {
        Tie {
            read,
            write: self.fresh(),
        }
    }

    /// Append `inst` to the open block. An instruction with a `Target` ends
    /// it.
    ///
    /// # Panics
    /// - the open block has ended;
    /// - a write not minted by this builder, or already defined;
    /// - a read of a value not yet defined, which includes a read of a value
    ///   this same instruction defines;
    /// - an [`Operand::Frame`]: only the instructions the allocator inserts
    ///   carry a slot.
    pub(super) fn push(&mut self, inst: B::Inst<Selected>) {
        let open = self.blocks.len() - 1;
        assert!(
            !ended(&self.blocks[open]),
            "{:?} has ended: nothing follows a branch in its block",
            self.blocks[open].label
        );
        // The one walk of this instruction: the block keeps the list beside it,
        // for the allocator.
        let operands = operands::<B>(&inst);
        for operand in &operands {
            match operand {
                Operand::Reg { value, access } if access.reads() => self.assert_defined(*value),
                Operand::Reg { .. } => {}
                Operand::Target(target) => target.args.iter().for_each(|&a| self.assert_defined(a)),
                Operand::Frame(slot) => panic!(
                    "selection named {slot:?}: only the instructions the allocator inserts carry a slot"
                ),
            }
        }
        for operand in &operands {
            match operand {
                Operand::Reg { value, access } if !access.reads() => self.define(*value),
                Operand::Reg { .. } | Operand::Frame(_) | Operand::Target(_) => {}
            }
        }
        self.blocks[open].insts.push(Pushed { inst, operands });
    }

    fn assert_defined(&self, value: ValueName) {
        assert!(
            self.defined.get(value.id as usize) == Some(&true),
            "{value:?} is read before it is defined"
        );
    }

    fn define(&mut self, value: ValueName) {
        let Some(defined) = self.defined.get_mut(value.id as usize) else {
            panic!("{value:?} was not minted by this builder")
        };
        assert!(!*defined, "{value:?} is defined twice");
        *defined = true;
    }

    /// A block to be entered later. Its label is minted now, with it.
    pub(super) fn block(&mut self, params: &[ClassId], scope: Scope) -> Pending {
        self.pending += 1;
        Pending {
            params: params.iter().map(|&class| self.mint(class)).collect(),
            label: self.labels.mint(),
            scope,
        }
    }

    /// Open `block`. It is consumed, so it cannot be entered twice.
    ///
    /// # Panics
    /// If the open block has not ended: only the last block of a function has
    /// no successor.
    pub(super) fn enter(&mut self, block: Pending) {
        let open = self.blocks.len() - 1;
        assert!(
            ended(&self.blocks[open]),
            "{:?} ends without a branch, and {:?} follows it",
            self.blocks[open].label,
            block.label
        );
        self.pending -= 1;
        for param in &block.params {
            self.defined[param.id as usize] = true;
        }
        self.open(block.label, block.params, block.scope);
    }

    /// The pool entry for `k`, deduplicated.
    ///
    /// # Errors
    /// [`CompileError::BudgetExceeded`] past the backend's reach.
    pub(super) fn constant(&mut self, k: B::Constant) -> Result<Constant, CompileError> {
        if let Some(&entry) = self.interned.get(&k) {
            return Ok(entry);
        }
        let index = self.constants.entries.len() as u64;
        if index >= B::POOL_REACH {
            return Err(CompileError::BudgetExceeded(
                "the constant pool outgrew what one instruction reaches",
            ));
        }
        let entry = Constant {
            label: self.labels.mint(),
        };
        self.constants.entries.push((entry.label, k));
        self.interned.insert(k, entry);
        Ok(entry)
    }

    /// Record the loop whose head is `head`, nested in `parent`, returning its
    /// index.
    pub(super) fn open_loop(&mut self, head: Label, parent: Option<usize>, trips: u64) -> usize {
        self.loops.push(Loop {
            head,
            parent,
            trips,
        });
        self.loops.len() - 1
    }

    /// The finished function: its open block is the exit.
    ///
    /// # Panics
    /// If any invariant of [`Function`] does not hold, a block was minted and
    /// never entered, or a value was minted and never defined (a dropped
    /// [`Def`], [`Early`] or [`Tie`]). Each is a selection bug, never a fact
    /// about a kernel.
    pub(super) fn finish(self) -> Function<B> {
        assert_eq!(self.pending, 0, "a block was minted and never entered");
        if let Some(id) = self.defined.iter().position(|&defined| !defined) {
            panic!("value {id} was minted and its definition dropped: nothing writes it");
        }
        let exit = self.blocks.len() - 1;
        assert!(
            !ended(&self.blocks[exit]),
            "{:?} ends in a branch, so the function has no exit",
            self.blocks[exit].label
        );
        let at: BTreeMap<Label, usize> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| (block.label, i))
            .collect();
        let block_of = |label: Label| match at.get(&label) {
            Some(&i) => i,
            None => panic!("{label:?} is branched to and bound by no block"),
        };

        let mut latches: Vec<Option<usize>> = alloc::vec![None; self.loops.len()];
        for (i, block) in self.blocks.iter().enumerate() {
            let from = block.label;
            let targets: Vec<&Target> = targets(block).collect();
            assert!(targets.len() <= 2, "{from:?} has {} targets", targets.len());
            for &target in &targets {
                let to = block_of(target.label);
                let params = &self.blocks[to].params;
                assert!(
                    params.len() == target.args.len()
                        && params
                            .iter()
                            .zip(&target.args)
                            .all(|(p, a)| p.class == a.class),
                    "{from:?} passes {:?} to the parameters of {:?}",
                    target.args,
                    target.label
                );
                if to > i {
                    continue;
                }
                let Some(l) = self.loops.iter().position(|l| l.head == target.label) else {
                    panic!(
                        "{from:?} branches back to {:?}, which heads no loop",
                        target.label
                    )
                };
                assert!(
                    latches[l].replace(i).is_none(),
                    "{:?} has two backward branches",
                    target.label
                );
            }
            if let [_, next] = targets[..] {
                assert!(
                    self.blocks
                        .get(i + 1)
                        .is_some_and(|b| b.label == next.label)
                        && next.args.is_empty(),
                    "{from:?}'s second target is not the following block, or passes it arguments"
                );
            }
        }

        // A loop is the run of blocks from its head to its one backward branch.
        let extents: Vec<(usize, usize)> = self
            .loops
            .iter()
            .zip(&latches)
            .map(|(l, latch)| {
                let Some(latch) = *latch else {
                    panic!("{:?} heads a loop with no backward branch", l.head)
                };
                (block_of(l.head), latch)
            })
            .collect();
        for (i, block) in self.blocks.iter().enumerate() {
            let from = block.label;
            for target in targets(block) {
                let to = block_of(target.label);
                if to <= i + 1 {
                    continue;
                }
                assert_eq!(
                    self.blocks[i].scope, self.blocks[to].scope,
                    "the branch from {from:?} to {:?} leaves its scope",
                    target.label
                );
                let crosses = extents.iter().any(|&(head, latch)| {
                    (head..=latch).contains(&i) != (head..=latch).contains(&to)
                });
                assert!(
                    !crosses,
                    "the branch from {from:?} to {:?} enters or leaves a loop",
                    target.label
                );
            }
        }

        let &[ctx, out, pitch] = self.blocks[0].params.as_slice() else {
            unreachable!("the entry block is opened with three parameters")
        };
        Function {
            entry: Entry::of([ctx, out, pitch]),
            blocks: self.blocks,
            loops: self.loops,
            constants: self.constants,
            labels: self.labels,
            classes: self.classes,
            scheduled: Vec::new(),
        }
    }
}

/// The rights the allocator's own verbs get: to define a fresh value and to
/// insert an instruction. `def` mints only [`Spill`] classes, so a copy, a
/// spill or a reload cannot write the flags.
pub(super) struct Spiller<'a, B: IsaBackend> {
    /// The next value id: the allocator continues the function's numbering.
    next: &'a mut u64,
    insts: &'a mut Vec<Pushed<B::Inst<Selected>>>,
}

impl<'a, B: IsaBackend> Spiller<'a, B> {
    pub(super) fn new(next: &'a mut u64, insts: &'a mut Vec<Pushed<B::Inst<Selected>>>) -> Self {
        Self { next, insts }
    }

    pub(in crate::emit) fn def<C: Spill>(&mut self) -> Def<C> {
        let id = *self.next;
        *self.next += 1;
        Def {
            value: ValueName { id, class: C::ID }.typed(),
        }
    }

    /// Append `inst` with its operand list, which is computed here and
    /// nowhere else.
    pub(in crate::emit) fn push(&mut self, inst: B::Inst<Selected>) {
        let operands = operands::<B>(&inst);
        self.insts.push(Pushed { inst, operands });
    }
}
