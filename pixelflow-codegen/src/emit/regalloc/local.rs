//! The local allocator: the simplest correct one.
//!
//! Every value lives in a frame slot. It is stored when it is defined and
//! reloaded into a register at every read, so no value is in a register
//! across an instruction except a `Flags` value, which has no slot and is held
//! from its definition to its read. What the later allocators add (residency,
//! carried values, rematerialization) are optimizations of this one's output,
//! and each is checked against it.
//!
//! The slots are the allocation. Liveness is intervals in layout order: a value
//! lives from its definition to its last read, a value defined before a loop
//! and read inside it lives to the loop's latch, and a block parameter lives
//! from the first branch that passes it an argument. Two values with disjoint
//! intervals share a slot. The frame is laid out first, so by the time any
//! instruction is bound the frame is final and a slot is a borrowed offset.
//!
//! Block parameters live in their slots, and a branch's arguments are stored
//! to them just before it.

use super::resource::{Frame, In, InOut, Leases, Lent, Out, Pool, SlotLease, SlotName};
use crate::emit::asm::{AsmProgram, Item, Label, Labels};
use crate::emit::build::{self, Spiller};
use crate::emit::{
    Access, Block, Bound, CONST_POOL_ALIGN, Class, ClassId, Constants, File, FrameSlot, Function,
    Integer, IsaBackend, Loop, Operand, Opmask, Pointer, Pushed, Rebind, Selected, Target, Value,
    ValueName, Vector,
};
use crate::error::CompileError;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

/// Why an instruction is there. `EmitTraffic` counts them per scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) enum Origin {
    Selected,
    Spill,
    Reload,
}

/// An allocated instruction, and why it is there.
pub(in crate::emit) struct Emitted<'m, B: IsaBackend> {
    pub(in crate::emit) inst: B::Inst<Bound<'m, B>>,
    pub(in crate::emit) origin: Origin,
}

/// An allocated kernel: every register a borrowed token, every slot a
/// borrowed slot, every block argument a placed move. A block has no
/// parameters any more: they are slots.
pub(in crate::emit) struct Allocated<'m, B: IsaBackend> {
    pub(in crate::emit) blocks: Vec<Block<Emitted<'m, B>>>,
    pub(in crate::emit) loops: Vec<Loop>,
    pub(in crate::emit) constants: Constants<B::Constant>,
    labels: Labels,
    /// The position after the last instruction, where the data section's
    /// padding begins.
    pub(in crate::emit) text_end: Label,
    pub(in crate::emit) frame_bytes: u64,
    /// Frame slots minted.
    pub(in crate::emit) slots: u64,
    /// Values read inside a loop that does not contain their definition.
    pub(in crate::emit) hoisted: u64,
    pub(in crate::emit) scheduled: Vec<u64>,
}

impl<'m, B: IsaBackend> Allocated<'m, B> {
    /// The kernel as a program for the assembler: its code, then its pool.
    /// The program owns the label mint from here on.
    pub(in crate::emit) fn program(&mut self) -> AsmProgram<&Emitted<'m, B>> {
        let mut text = Vec::new();
        for block in &self.blocks {
            text.push(Item::Bind(block.label));
            text.extend(block.insts.iter().map(Item::Inst));
        }
        text.push(Item::Bind(self.text_end));
        let mut data = Vec::new();
        if !self.constants.entries.is_empty() {
            data.push(Item::Align(CONST_POOL_ALIGN as u64));
        }
        data.push(Item::Bind(self.constants.label));
        for &(label, constant) in &self.constants.entries {
            data.push(Item::Bind(label));
            data.push(Item::Bytes(B::constant_bytes(constant)));
        }
        AsmProgram {
            text,
            data,
            labels: core::mem::take(&mut self.labels),
        }
    }
}

/// What the allocator knows of one value before it places anything.
struct Life {
    class: ClassId,
    /// Where it is defined: an instruction's position, or for a block
    /// parameter the block's first.
    def: usize,
    /// The first position its slot is needed: `def`, or earlier for a
    /// parameter that a branch fills.
    from: usize,
    /// The last position its slot is needed.
    last: usize,
    read: bool,
    /// Read inside a loop that does not contain its definition.
    hoisted: bool,
    slot: Option<SlotName>,
}

/// Every value's [`Life`], by id.
fn lives<B: IsaBackend>(function: &Function<B>) -> Vec<Life> {
    let mut lives: Vec<Life> = function
        .classes
        .iter()
        .map(|&class| Life {
            class,
            def: 0,
            from: 0,
            last: 0,
            read: false,
            hoisted: false,
            slot: None,
        })
        .collect();

    let mut starts = Vec::with_capacity(function.blocks.len());
    let mut position = 0;
    for block in &function.blocks {
        starts.push(position);
        position += block.insts.len();
    }
    let at: BTreeMap<Label, usize> = function
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| (block.label, i))
        .collect();
    for (block, &start) in function.blocks.iter().zip(&starts) {
        for param in &block.params {
            let life = &mut lives[param.id as usize];
            (life.def, life.from, life.last) = (start, start, start);
        }
    }

    // A loop is the run of positions from its head to its one backward branch.
    let mut extents: Vec<(usize, usize)> = function.loops.iter().map(|_| (0, 0)).collect();
    let mut position = 0;
    for (i, block) in function.blocks.iter().enumerate() {
        position += block.insts.len();
        let targets = block.insts.last().into_iter().flat_map(|p| &p.operands);
        for operand in targets {
            let Operand::Target(target) = operand else {
                continue;
            };
            if at[&target.label] > i {
                continue;
            }
            let l = function
                .loops
                .iter()
                .position(|l| l.head == target.label)
                .expect("a backward branch targets a loop's head");
            extents[l] = (starts[at[&target.label]], position - 1);
        }
    }

    let mut position = 0;
    for block in &function.blocks {
        for pushed in &block.insts {
            for operand in &pushed.operands {
                match operand {
                    Operand::Reg { value, access } if access.reads() => {
                        read(&mut lives, &extents, *value, position);
                    }
                    Operand::Reg { value, .. } => {
                        let life = &mut lives[value.id as usize];
                        (life.def, life.from, life.last) = (position, position, position);
                    }
                    Operand::Target(target) => {
                        for &arg in &target.args {
                            read(&mut lives, &extents, arg, position);
                        }
                        let params = &function.blocks[at[&target.label]].params;
                        for param in params {
                            let life = &mut lives[param.id as usize];
                            life.from = life.from.min(position);
                            life.last = life.last.max(position);
                        }
                    }
                    Operand::Frame(_) => unreachable!("selection names no frame slot"),
                }
            }
            position += 1;
        }
    }
    lives
}

/// Value `v` is read at `position`.
fn read(lives: &mut [Life], extents: &[(usize, usize)], v: ValueName, position: usize) {
    let life = &mut lives[v.id as usize];
    life.read = true;
    life.last = life.last.max(position);
    for &(head, latch) in extents {
        if (head..=latch).contains(&position) && life.def < head {
            life.last = life.last.max(latch);
            life.hoisted = true;
        }
    }
}

/// Give every value that is read, and can be stored, a slot; lay the frame
/// out for them.
fn layout(lives: &mut [Life], frame: &mut Frame) -> Result<(), CompileError> {
    let mut stored: Vec<usize> = (0..lives.len())
        .filter(|&id| lives[id].read && lives[id].class != ClassId::Flags)
        .collect();
    stored.sort_by_key(|&id| (lives[id].from, id));

    // The peak number of narrow values live at once, by a difference array
    // over positions.
    let end = lives.iter().map(|life| life.last).max().unwrap_or(0) + 2;
    let mut delta = alloc::vec![0i64; end];
    for &id in &stored {
        let life = &lives[id];
        if life.class != ClassId::Vector {
            delta[life.from] += 1;
            delta[life.last + 1] -= 1;
        }
    }
    let peak = delta
        .iter()
        .scan(0, |live, d| {
            *live += d;
            Some(*live)
        })
        .max()
        .unwrap_or(0);
    frame.reserve_narrow(peak as u64)?;

    let mut leases: BTreeMap<usize, SlotLease> = BTreeMap::new();
    let mut active: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for id in stored {
        while let Some(&Reverse((last, done))) = active.peek() {
            if last >= lives[id].from {
                break;
            }
            active.pop();
            frame.release(leases.remove(&done).expect("an active value holds a slot"));
        }
        let lease = frame.lease(lives[id].class)?;
        lives[id].slot = Some(lease.name());
        leases.insert(id, lease);
        active.push(Reverse((lives[id].last, id)));
    }
    Ok(())
}

/// Allocate `function`: bind every value to leases of `pool` and slots of
/// `frame`, inserting the stores and reloads that make the binding hold.
///
/// # Errors
/// [`CompileError::BudgetExceeded`] when the narrow region passes 4,096 slots
/// or the frame passes `MAX_FRAME`.
///
/// # Panics
/// A selection bug, never a fact about a kernel: an instruction holding more
/// registers of a file than the file has, a `Flags` value read after another
/// flags write, or a flags value read in another block.
pub(in crate::emit) fn allocate<'m, B: IsaBackend>(
    function: Function<B>,
    pool: &'m Pool<B>,
    frame: &'m mut Frame,
) -> Result<Allocated<'m, B>, CompileError> {
    let mut lives = lives(&function);
    layout(&mut lives, frame)?;
    let frame: &'m Frame = frame;

    let leases = Leases::new(pool);
    let free = [
        leases.vector.into_iter().map(Lent::Vector).collect(),
        leases.general.into_iter().map(Lent::General).collect(),
        leases.opmask.into_iter().map(Lent::Opmask).collect(),
        leases.flags.into_iter().map(Lent::Flags).collect(),
    ];
    // The ABI's registers are the entry parameters' to begin with.
    let entry = [
        function.entry.ctx.name(),
        function.entry.out.name(),
        function.entry.pitch.name(),
    ];
    let held = BTreeMap::from([
        (entry[0].id, Lent::General(leases.entry.ctx)),
        (entry[1].id, Lent::General(leases.entry.out)),
        (entry[2].id, Lent::General(leases.entry.pitch)),
    ]);

    let mut labels = function.labels;
    let text_end = labels.mint();
    let params = function
        .blocks
        .iter()
        .map(|block| (block.label, block.params.clone()))
        .collect();
    let mut scan = Scan {
        frame,
        lives: &lives,
        params,
        free,
        held,
        next: function.classes.len() as u64,
        out: Vec::new(),
        at: function.blocks[0].label,
        position: 0,
    };

    let mut blocks = Vec::with_capacity(function.blocks.len());
    for block in function.blocks {
        scan.at = block.label;
        for pushed in block.insts {
            scan.instruction(pushed, &entry);
            scan.position += 1;
        }
        blocks.push(Block {
            label: block.label,
            params: Vec::new(),
            insts: core::mem::take(&mut scan.out),
            scope: block.scope,
        });
    }

    Ok(Allocated {
        blocks,
        loops: function.loops,
        constants: function.constants,
        labels,
        text_end,
        frame_bytes: frame.bytes(),
        slots: frame.minted(),
        hoisted: lives.iter().filter(|life| life.hoisted).count() as u64,
        scheduled: function.scheduled,
    })
}

/// The scan's state: which registers are free, which values are in one, and
/// the instructions emitted so far in the open block.
struct Scan<'a, 'm, B: IsaBackend> {
    frame: &'m Frame,
    lives: &'a [Life],
    /// The parameters of every block, which a branch's arguments are stored to.
    params: BTreeMap<Label, Vec<ValueName>>,
    /// The leases nothing holds, by file.
    free: [Vec<Lent<'m, B>>; 4],
    /// The values in a register, by id.
    held: BTreeMap<u64, Lent<'m, B>>,
    /// The next value id the allocator mints.
    next: u64,
    out: Vec<Emitted<'m, B>>,
    /// Where the scan is, for what it says when a function cannot be allocated.
    at: Label,
    position: usize,
}

impl<'m, B: IsaBackend> Scan<'_, 'm, B> {
    /// One selected instruction: its branch arguments stored, its reads
    /// reloaded, itself, its writes stored.
    fn instruction(&mut self, pushed: Pushed<B::Inst<Selected>>, entry: &[ValueName; 3]) {
        for operand in &pushed.operands {
            if let Operand::Target(target) = operand {
                self.pass(target);
            }
        }
        for operand in &pushed.operands {
            let Operand::Reg { value, access } = operand else {
                continue;
            };
            if access.reads() && !self.held.contains_key(&value.id) {
                self.reload_into(*value);
            }
        }
        let mut written: Vec<ValueName> = pushed
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Reg { value, access } if !access.reads() => Some(*value),
                Operand::Reg { .. } | Operand::Target(_) | Operand::Frame(_) => None,
            })
            .collect();
        if self.position == 0 {
            // The entry block's first instruction makes the frame, so the
            // ABI's registers are stored just after it.
            written.extend(entry);
        }
        self.place(pushed, Origin::Selected);
        for value in written {
            if let Some(slot) = self.lives.get(value.id as usize).and_then(|l| l.slot) {
                self.store(value, slot);
            }
        }
        self.release();
    }

    /// Store each of `target`'s arguments to its parameter's slot.
    fn pass(&mut self, target: &Target) {
        let params = self.params[&target.label].clone();
        for (arg, param) in target.args.iter().zip(&params) {
            let Some(home) = self.lives[param.id as usize].slot else {
                continue;
            };
            let from = self.lives[arg.id as usize]
                .slot
                .expect("an argument is read, so it has a slot");
            let moved = self.reload(arg.class, from);
            self.store(moved, home);
            let lent = self.held.remove(&moved.id).expect("a reload is held");
            self.free[lent.file() as usize].push(lent);
        }
    }

    /// Make `value` resident: reload it from its slot, into the register the
    /// instruction will read it from.
    fn reload_into(&mut self, value: ValueName) {
        assert_ne!(
            value.class,
            ClassId::Flags,
            "{value:?} is read at instruction {} of {:?}, which does not hold the flags",
            self.position,
            self.at
        );
        let slot = self.lives[value.id as usize]
            .slot
            .expect("a value that is read has a slot");
        let reloaded = self.reload(value.class, slot);
        let lent = self.held.remove(&reloaded.id).expect("a reload is held");
        self.held.insert(value.id, lent);
    }

    /// Insert the load of `slot` into a fresh value of `class`, held.
    fn reload(&mut self, class: ClassId, slot: SlotName) -> ValueName {
        let frame = self.frame;
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let slot = frame.slot(slot);
        let fresh = match class {
            ClassId::Vector => B::reload::<Vector>(&mut spiller, slot).name(),
            ClassId::Pointer => B::reload::<Pointer>(&mut spiller, slot).name(),
            ClassId::Integer => B::reload::<Integer>(&mut spiller, slot).name(),
            ClassId::Opmask => B::reload::<Opmask>(&mut spiller, slot).name(),
            ClassId::Flags => unreachable!("the flags are not stored"),
        };
        for pushed in insts {
            self.place(pushed, Origin::Reload);
        }
        fresh
    }

    /// Insert the store of `value`, which is held, to `slot`.
    fn store(&mut self, value: ValueName, slot: SlotName) {
        let frame = self.frame;
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let slot = frame.slot(slot);
        match value.class {
            ClassId::Vector => B::spill::<Vector>(&mut spiller, value.typed(), slot),
            ClassId::Pointer => B::spill::<Pointer>(&mut spiller, value.typed(), slot),
            ClassId::Integer => B::spill::<Integer>(&mut spiller, value.typed(), slot),
            ClassId::Opmask => B::spill::<Opmask>(&mut spiller, value.typed(), slot),
            ClassId::Flags => unreachable!("the flags are not stored"),
        }
        for pushed in insts {
            self.place(pushed, Origin::Spill);
        }
    }

    /// The free lease of the lowest number in `class`'s file.
    fn take(&mut self, class: ClassId) -> Lent<'m, B> {
        let bank = &mut self.free[class.file() as usize];
        let lowest = bank
            .iter()
            .enumerate()
            .min_by_key(|(_, lent)| lent.number())
            .map(|(i, _)| i);
        let Some(i) = lowest else {
            panic!(
                "instruction {} of {:?} holds more {:?} registers than the file has",
                self.position,
                self.at,
                class.file()
            )
        };
        bank.swap_remove(i)
    }

    /// Bind `pushed`: its writes take free leases, its reads are already held,
    /// and a tied write is left in its read's register.
    fn place(&mut self, pushed: Pushed<B::Inst<Selected>>, origin: Origin) {
        let Pushed { inst, operands } = pushed;
        for operand in &operands {
            if let Operand::Reg {
                value,
                access: Access::Write | Access::Early,
            } = operand
            {
                let lent = self.take(value.class);
                assert!(
                    self.held.insert(value.id, lent).is_none(),
                    "{value:?} is defined while in a register"
                );
            }
        }
        let size = self.frame.bytes();
        let inst = B::walk::<Bound<'m, B>>(
            &inst,
            &mut Binding {
                frame: self.frame,
                held: &self.held,
                size,
            },
        );
        for operand in &operands {
            let Operand::Reg {
                value,
                access: Access::Tied { read },
            } = operand
            else {
                continue;
            };
            let Operand::Reg { value: read, .. } = &operands[*read] else {
                unreachable!("a tie names a register operand")
            };
            let lent = self.held.remove(&read.id).expect("a tied read is held");
            self.held.insert(value.id, lent);
        }
        self.out.push(Emitted { inst, origin });
    }

    /// Return the leases the instruction is done with. A flags value that is
    /// read later keeps its lease.
    fn release(&mut self) {
        for (id, lent) in core::mem::take(&mut self.held) {
            let pending = self
                .lives
                .get(id as usize)
                .is_some_and(|life| life.class == ClassId::Flags && life.last > self.position);
            match pending {
                true => {
                    self.held.insert(id, lent);
                }
                false => self.free[lent.file() as usize].push(lent),
            }
        }
    }
}

/// How an instruction is bound: each operand becomes the token of the lease
/// its value holds.
struct Binding<'a, 'm, B: IsaBackend> {
    frame: &'m Frame,
    held: &'a BTreeMap<u64, Lent<'m, B>>,
    size: u64,
}

impl<'a, 'm, B: IsaBackend> Binding<'a, 'm, B> {
    fn lent(&self, value: ValueName) -> &'a Lent<'m, B> {
        self.held
            .get(&value.id)
            .unwrap_or_else(|| panic!("{value:?} is bound while in no register"))
    }
}

impl<'m, B: IsaBackend> Rebind<Bound<'m, B>> for Binding<'_, 'm, B> {
    fn read<C: Class>(&mut self, v: Value<C>) -> In<'m, B, C> {
        C::File::lease(self.lent(v.name())).read()
    }
    fn write<C: Class>(&mut self, d: &build::Def<C>) -> Out<'m, B, C> {
        C::File::lease(self.lent(d.value().name())).write()
    }
    fn early<C: Class>(&mut self, d: &build::Early<C>) -> Out<'m, B, C> {
        C::File::lease(self.lent(d.value().name())).write()
    }
    fn tie<C: Class>(&mut self, t: &build::Tie<C>) -> InOut<'m, B, C> {
        C::File::lease(self.lent(t.read().name())).tie()
    }
    fn slot(&mut self, s: SlotName) -> &'m FrameSlot {
        self.frame.slot(s)
    }
    fn target(&mut self, t: &Target) -> Label {
        t.label
    }
    fn frame_size(&mut self) -> u64 {
        self.size
    }
}
