//! The local allocator: one pass over the instructions in layout order.
//!
//! A value stays in its register until something needs the register, and a
//! value that is read while in none is reloaded into a fresh one that stays
//! until it is needed in turn. What the scan gives up a register for is
//! priced by [`EvictionRank`]: the value that costs no store, then the one
//! read farthest out.
//!
//! **A spilled value is stored right after its definition**, never at the
//! eviction point, which a guard can skip. The scan only learns that a value
//! must be stored when it first evicts it, so the store is inserted
//! retroactively: it is kept beside the definition it follows and placed
//! there when the blocks are bound.
//!
//! **Loop heads are flushed.** A value live into a loop head is stored and
//! dropped from its register there, so inside the loop its home is its slot
//! and the reloads are split values that are dead before the latch. Block
//! parameters are slot-homed too: a branch stores its arguments to them.
//!
//! The slots are the second register file. Liveness is intervals in layout
//! order: a value lives from its definition to its last read, a value defined
//! before a loop and read inside it lives to the loop's latch, and a block
//! parameter lives from the first branch that passes it an argument. The
//! narrow slots are laid out before the scan, from the peak number of narrow
//! values live at once; a vector slot is leased for one value's interval when
//! the value is first stored.
//!
//! Registers are bound after the scan: it records which register every
//! operand of every instruction held, and the frame is final by then, so a
//! slot is a borrowed offset.

use super::policy::{EvictionRank, ReadHere, Store};
use super::resource::{Frame, In, InOut, Leases, Lent, Out, Pool, SlotLease, SlotName};
use crate::emit::asm::{AsmProgram, Item, Label, Labels};
use crate::emit::build::{self, Spiller};
use crate::emit::{
    Access, Block, Bound, CONST_POOL_ALIGN, Class, ClassId, Constants, File, FileId, FrameSlot,
    Function, Integer, IsaBackend, Loop, Operand, Opmask, Pointer, Pushed, Rebind, Selected,
    Target, Value, ValueName, Vector,
};
use crate::error::CompileError;
use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::ops::Range;

/// Why an instruction is there. `EmitTraffic` counts them per scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) enum Origin {
    Selected,
    /// A selected constant the backend can recompute rather than store
    /// ([`IsaBackend::rematerializable`]): `EmitTraffic`'s remats.
    Remat,
    Spill,
    Reload,
    /// A register copy ([`IsaBackend::copy`]): an operand the instruction
    /// overwrites in place, which is read again later.
    Copy,
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
    constants: Constants<B::Constant>,
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
    /// Where its positions are among the scan's, which are sorted by value.
    reads: Range<usize>,
    /// Read inside a loop that does not contain its definition.
    hoisted: bool,
    slot: Option<SlotName>,
    /// Whether the slot holds the value from here on: it has been stored, or
    /// it is a parameter, which a branch fills.
    stored: bool,
}

/// Every value's [`Life`], and the positions that read it.
struct Liveness {
    lives: Vec<Life>,
    /// `(value, position)` for every read, in layout order.
    reads: Vec<(u64, usize)>,
    /// Each loop's first and last position: from its head to its one backward
    /// branch.
    extents: Vec<(usize, usize)>,
}

impl Liveness {
    fn of<B: IsaBackend>(function: &Function<B>) -> Self {
        let lives = function
            .classes
            .iter()
            .map(|&class| Life {
                class,
                def: 0,
                from: 0,
                last: 0,
                read: false,
                reads: 0..0,
                hoisted: false,
                slot: None,
                stored: false,
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
        let mut live = Self {
            lives,
            reads: Vec::new(),
            extents: function.loops.iter().map(|_| (0, 0)).collect(),
        };
        for (block, &start) in function.blocks.iter().zip(&starts) {
            for param in &block.params {
                let life = &mut live.lives[param.id as usize];
                (life.def, life.from, life.last) = (start, start, start);
            }
        }

        // A loop is the run of positions from its head to its one backward
        // branch.
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
                live.extents[l] = (starts[at[&target.label]], position - 1);
            }
        }

        let mut position = 0;
        for block in &function.blocks {
            for pushed in &block.insts {
                for operand in &pushed.operands {
                    match operand {
                        Operand::Reg { value, access } if access.reads() => {
                            live.read(*value, position);
                        }
                        Operand::Reg { value, .. } => {
                            let life = &mut live.lives[value.id as usize];
                            (life.def, life.from, life.last) = (position, position, position);
                        }
                        Operand::Target(target) => {
                            for &arg in &target.args {
                                live.read(arg, position);
                            }
                            let params = &function.blocks[at[&target.label]].params;
                            for param in params {
                                let life = &mut live.lives[param.id as usize];
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
        live
    }

    /// Value `v` is read at `position`.
    fn read(&mut self, v: ValueName, position: usize) {
        self.reads.push((v.id, position));
        let life = &mut self.lives[v.id as usize];
        life.read = true;
        life.last = life.last.max(position);
        for &(head, latch) in &self.extents {
            if (head..=latch).contains(&position) && life.def < head {
                life.last = life.last.max(latch);
                life.hoisted = true;
            }
        }
    }
}

/// Give every narrow value that is read a slot; lay the narrow region out for
/// them. A vector value's slot is leased when it is first stored.
fn layout(lives: &mut [Life], frame: &mut Frame) -> Result<(), CompileError> {
    let mut narrow: Vec<usize> = (0..lives.len())
        .filter(|&id| {
            lives[id].read
                && matches!(
                    lives[id].class,
                    ClassId::Pointer | ClassId::Integer | ClassId::Opmask
                )
        })
        .collect();
    narrow.sort_by_key(|&id| (lives[id].from, id));

    // The peak number of narrow values live at once, by a difference array
    // over positions.
    let end = lives.iter().map(|life| life.last).max().unwrap_or(0) + 2;
    let mut delta = alloc::vec![0i64; end];
    for &id in &narrow {
        delta[lives[id].from] += 1;
        delta[lives[id].last + 1] -= 1;
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
    for id in narrow {
        while let Some(&Reverse((last, done))) = active.peek() {
            if last >= lives[id].from {
                break;
            }
            active.pop();
            frame.release(leases.remove(&done).expect("an active value holds a slot"));
        }
        let lease = frame.lease_narrow();
        lives[id].slot = Some(lease.name());
        leases.insert(id, lease);
        active.push(Reverse((lives[id].last, id)));
    }
    Ok(())
}

/// Allocate `function`: bind every value to leases of `pool` and slots of
/// `frame`, inserting the stores, reloads and copies that make the binding
/// hold.
///
/// # Errors
/// [`CompileError::BudgetExceeded`] when the narrow region passes 4,096 slots
/// or the frame passes `MAX_FRAME`.
///
/// # Panics
/// A selection bug, never a fact about a kernel: an instruction holding more
/// registers of a file than the file has, a `Flags` value read after another
/// flags write or held to the end of its block, a branch argument that is a
/// parameter of one of the branch's targets (a move the allocator does not
/// order), a forward branch past a block (a join), or an entry block whose
/// first instruction does not make the frame.
pub(in crate::emit) fn allocate<'m, B: IsaBackend>(
    function: Function<B>,
    pool: &'m Pool<B>,
    frame: &'m mut Frame,
) -> Result<Allocated<'m, B>, CompileError> {
    let Liveness {
        mut lives,
        mut reads,
        ..
    } = Liveness::of(&function);
    reads.sort_unstable();
    let mut start = 0;
    while let Some(&(value, _)) = reads.get(start) {
        let len = reads[start..].partition_point(|&(v, _)| v == value);
        lives[value as usize].reads = start..start + len;
        start += len;
    }
    let reads = reads.into_iter().map(|(_, position)| position).collect();
    layout(&mut lives, frame)?;
    let hoisted = lives.iter().filter(|life| life.hoisted).count() as u64;

    let leases = Leases::new(pool);
    let mut free: [Vec<Lent<B>>; 4] = [
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
    let mut held = BTreeMap::from([
        (entry[0].id, Lent::General(leases.entry.ctx)),
        (entry[1].id, Lent::General(leases.entry.out)),
        (entry[2].id, Lent::General(leases.entry.pitch)),
    ]);
    // One nothing reads is dead on arrival, which the scan would find only
    // after the first instruction.
    for (_, lent) in held.extract_if(.., |&id, _| lives[id as usize].reads.is_empty()) {
        free[lent.file() as usize].push(lent);
    }
    // The ABI registers are stored, if they ever are, right after the entry
    // block's first instruction, which makes the frame.
    let mut defined: Vec<Option<(usize, usize)>> = alloc::vec![None; lives.len()];
    for value in entry {
        defined[value.id as usize] = Some((0, 0));
    }

    let mut labels = function.labels;
    let text_end = labels.mint();
    let blocks_at = function
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| (block.label, (i, block.params.clone())))
        .collect();
    let heads: BTreeSet<Label> = function.loops.iter().map(|l| l.head).collect();
    let mut scan = Scan {
        frame: &mut *frame,
        reads,
        lives,
        blocks_at,
        free,
        held,
        pinned: Vec::new(),
        next: function.classes.len() as u64,
        defined,
        records: Vec::new(),
        late: BTreeMap::new(),
        numbers: Vec::new(),
        at: function.blocks[0].label,
        block: 0,
        position: 0,
    };

    let mut shapes = Vec::with_capacity(function.blocks.len());
    for block in function.blocks {
        scan.at = block.label;
        scan.records.push(Vec::new());
        if heads.contains(&block.label) {
            scan.flush()?;
        }
        for pushed in block.insts {
            scan.instruction(pushed)?;
            scan.position += 1;
        }
        let flags = scan
            .held
            .iter()
            .find_map(|(id, lent)| (lent.file() == FileId::Flags).then_some(id));
        if let Some(flags) = flags {
            panic!(
                "value {flags} is a flags value live at the end of {:?}, and no flags value is live at a label",
                block.label
            );
        }
        shapes.push((block.label, block.scope));
        scan.block += 1;
    }
    assert!(
        scan.held.is_empty(),
        "values are held after the last instruction: {:?}",
        scan.held.keys()
    );
    let Scan {
        free,
        records,
        late,
        numbers,
        ..
    } = scan;

    let slots = frame.minted();
    let frame: &'m Frame = frame;
    let mut bank: [BTreeMap<u8, Lent<'m, B>>; 4] = Default::default();
    for lent in free.into_iter().flatten() {
        bank[lent.file() as usize].insert(lent.number(), lent);
    }
    let mut binding = Binding {
        frame,
        bank,
        numbers: &numbers,
        next: 0,
        framed: false,
    };
    let mut late = late;
    let mut blocks = Vec::with_capacity(records.len());
    for (b, (block, (label, scope))) in records.into_iter().zip(shapes).enumerate() {
        let mut insts = Vec::with_capacity(block.len());
        for (k, record) in block.into_iter().enumerate() {
            insts.push(binding.record(&record));
            assert!(
                (b, k) != (0, 0) || binding.framed,
                "the entry block's first instruction does not make the frame, so the ABI registers have nowhere to be stored"
            );
            for stored in late.remove(&(b, k)).unwrap_or_default() {
                insts.push(binding.record(&stored));
            }
        }
        blocks.push(Block {
            label,
            params: Vec::new(),
            insts,
            scope,
        });
    }

    Ok(Allocated {
        blocks,
        loops: function.loops,
        constants: function.constants,
        labels,
        text_end,
        frame_bytes: frame.bytes(),
        slots,
        hoisted,
        scheduled: function.scheduled,
    })
}

/// An instruction the scan has placed: what to bind, why it is there, and
/// where in the scan's numbers its operands' registers begin.
struct Record<B: IsaBackend> {
    pushed: Pushed<B::Inst<Selected>>,
    origin: Origin,
    regs: usize,
}

/// The scan's state: which registers are free, which values are in one, and
/// the instructions placed so far.
struct Scan<'f, 'm, B: IsaBackend> {
    frame: &'f mut Frame,
    lives: Vec<Life>,
    /// The position of every read, by value and then by position.
    reads: Vec<usize>,
    /// Each block's index and parameters, by label: what a branch's arguments
    /// are stored to.
    blocks_at: BTreeMap<Label, (usize, Vec<ValueName>)>,
    /// The leases nothing holds, by file.
    free: [Vec<Lent<'m, B>>; 4],
    /// The values in a register, by id.
    held: BTreeMap<u64, Lent<'m, B>>,
    /// The values the instruction being placed reads or defines: the ones
    /// eviction leaves alone.
    pinned: Vec<u64>,
    /// The next value id the allocator mints.
    next: u64,
    /// Where each value is defined: a block, and an index into its records.
    defined: Vec<Option<(usize, usize)>>,
    /// The instructions placed so far, by block.
    records: Vec<Vec<Record<B>>>,
    /// The stores of evicted values, by the block and index of the definition
    /// each follows.
    late: BTreeMap<(usize, usize), Vec<Record<B>>>,
    /// The register number of every operand of every record, in operand order;
    /// zero where the operand is not a register.
    numbers: Vec<u8>,
    /// Where the scan is, for what it says when a function cannot be allocated.
    at: Label,
    block: usize,
    position: usize,
}

/// `$body` with `$class` standing for the spillable class `$id` names.
macro_rules! of_class {
    ($id:expr, $class:ident => $body:expr) => {
        match $id {
            ClassId::Vector => {
                type $class = Vector;
                $body
            }
            ClassId::Pointer => {
                type $class = Pointer;
                $body
            }
            ClassId::Integer => {
                type $class = Integer;
                $body
            }
            ClassId::Opmask => {
                type $class = Opmask;
                $body
            }
            ClassId::Flags => unreachable!("the flags are not stored"),
        }
    };
}

impl<'m, B: IsaBackend> Scan<'_, 'm, B> {
    /// One selected instruction: its reads in registers, its branch arguments
    /// stored, itself, and the registers it is done with returned.
    fn instruction(&mut self, pushed: Pushed<B::Inst<Selected>>) -> Result<(), CompileError> {
        let targets: Vec<&Target> = pushed
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Target(target) => Some(target),
                Operand::Reg { .. } | Operand::Frame(_) => None,
            })
            .collect();
        self.refuse_overlapping_moves(&targets);
        for target in &targets {
            assert!(
                self.blocks_at[&target.label].0 <= self.block + 1,
                "{:?} branches to {:?}, past a block: a forward join, which arrives with B9",
                self.at,
                target.label
            );
        }
        let reads: Vec<ValueName> = pushed
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Reg { value, access } if access.reads() => Some(*value),
                Operand::Reg { .. } | Operand::Target(_) | Operand::Frame(_) => None,
            })
            .chain(
                targets
                    .iter()
                    .flat_map(|target| target.args.iter().copied()),
            )
            .collect();
        self.pinned.clear();
        self.pinned.extend(reads.iter().map(|read| read.id));
        for read in reads {
            if !self.held.contains_key(&read.id) {
                self.reload_into(read)?;
            }
        }
        for target in targets {
            self.pass(target)?;
        }
        self.emit(pushed, Origin::Selected)?;
        self.release_dead();
        Ok(())
    }

    /// Arguments are moved one at a time, through the parameters' slots, which
    /// is a parallel move only if no argument is a parameter an earlier store
    /// overwrote. Ordering one that is (a swap needs a cycle broken through a
    /// fresh value) is the loops commit's; until then it is refused.
    fn refuse_overlapping_moves(&self, targets: &[&Target]) {
        for target in targets {
            for arg in &target.args {
                if let Some(other) = targets
                    .iter()
                    .find(|other| self.blocks_at[&other.label].1.contains(arg))
                {
                    panic!(
                        "{arg:?} is passed to {:?} and is a parameter of {:?}, whose slot an earlier move of the same branch overwrites",
                        target.label, other.label
                    );
                }
            }
        }
    }

    /// Store each of `target`'s arguments, which are in registers, to its
    /// parameter's slot. A parameter nothing reads has none.
    fn pass(&mut self, target: &Target) -> Result<(), CompileError> {
        let params = self.blocks_at[&target.label].1.clone();
        for (&arg, param) in target.args.iter().zip(&params) {
            if !self.lives[param.id as usize].read {
                continue;
            }
            let slot = self.slot(*param)?;
            self.lives[param.id as usize].stored = true;
            for pushed in self.spill_of(arg, slot) {
                self.emit(pushed, Origin::Spill)?;
            }
        }
        Ok(())
    }

    /// Make `value` resident: reload it from its slot, into the register the
    /// instruction will read it from. It stays there until something needs
    /// the register.
    fn reload_into(&mut self, value: ValueName) -> Result<(), CompileError> {
        assert_ne!(
            value.class,
            ClassId::Flags,
            "{value:?} is read at instruction {} of {:?}, which does not hold the flags",
            self.position,
            self.at
        );
        let life = &self.lives[value.id as usize];
        let slot = life
            .slot
            .filter(|_| life.stored)
            .unwrap_or_else(|| panic!("{value:?} is read while in no register and in no slot"));
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let slot = self.frame.slot(slot);
        let fresh = of_class!(value.class, C => B::reload::<C>(&mut spiller, slot).name());
        for pushed in insts {
            self.emit(pushed, Origin::Reload)?;
        }
        let lent = self.held.remove(&fresh.id).expect("a reload is held");
        self.held.insert(value.id, lent);
        Ok(())
    }

    /// The instructions that store `value`, which is in a register, to `slot`.
    fn spill_of(&mut self, value: ValueName, slot: SlotName) -> Vec<Pushed<B::Inst<Selected>>> {
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let slot = self.frame.slot(slot);
        of_class!(value.class, C => B::spill::<C>(&mut spiller, value.typed(), slot));
        insts
    }

    /// `value`'s slot, leasing a vector value's now.
    fn slot(&mut self, value: ValueName) -> Result<SlotName, CompileError> {
        let life = &mut self.lives[value.id as usize];
        if let Some(slot) = life.slot {
            return Ok(slot);
        }
        assert_eq!(
            life.class,
            ClassId::Vector,
            "{value:?} is stored, and the narrow region was laid out for the values that are read"
        );
        assert!(
            (life.from..=life.last).contains(&self.position),
            "{value:?} is stored at {}, outside its span {}..={}",
            self.position,
            life.from,
            life.last
        );
        let slot = self.frame.lease_vector(life.from..=life.last)?;
        life.slot = Some(slot);
        Ok(slot)
    }

    /// Store `value`, which is in its defining register, right after its
    /// definition, unless that is done.
    fn store_at_definition(&mut self, value: u64) -> Result<(), CompileError> {
        let life = &self.lives[value as usize];
        if life.stored {
            return Ok(());
        }
        let name = ValueName {
            id: value,
            class: life.class,
        };
        let slot = self.slot(name)?;
        self.lives[value as usize].stored = true;
        let at = self.defined[value as usize].expect("a value in a register was defined");
        let number = self.held[&value].number();
        for pushed in self.spill_of(name, slot) {
            // The register `value` is in is the only one such a store may
            // name: the point it follows is behind the scan.
            let regs = self.numbers.len();
            for operand in &pushed.operands {
                self.numbers.push(match operand {
                    Operand::Reg { value: read, .. } => {
                        assert_eq!(
                            read.id, value,
                            "the store of {value} after its definition needs a register at a point the scan has left"
                        );
                        number
                    }
                    Operand::Target(_) | Operand::Frame(_) => 0,
                });
            }
            self.late.entry(at).or_default().push(Record {
                pushed,
                origin: Origin::Spill,
                regs,
            });
        }
        Ok(())
    }

    /// Drop every value from its register, storing the ones not yet stored:
    /// what a loop head expects of the state it is entered in.
    fn flush(&mut self) -> Result<(), CompileError> {
        for id in self.held.keys().copied().collect::<Vec<_>>() {
            self.store_at_definition(id)?;
            let lent = self.held.remove(&id).expect("the key was just read");
            self.free[lent.file() as usize].push(lent);
        }
        Ok(())
    }

    /// The first position after this one that reads `value`.
    fn next_read(&self, value: u64) -> Option<usize> {
        let own = &self.reads[self.lives.get(value as usize)?.reads.clone()];
        own.get(own.partition_point(|&p| p <= self.position))
            .copied()
    }

    /// A free lease of `class`'s file, the lowest-numbered; when there is none,
    /// the one the cheapest value gives up.
    fn acquire(&mut self, class: ClassId) -> Result<Lent<'m, B>, CompileError> {
        let file = class.file();
        if self.free[file as usize].is_empty() {
            self.evict(file)?;
        }
        let bank = &mut self.free[file as usize];
        let lowest = bank
            .iter()
            .enumerate()
            .min_by_key(|(_, lent)| lent.number())
            .map(|(i, _)| i)
            .expect("eviction frees a register or panics");
        Ok(bank.swap_remove(lowest))
    }

    /// Free a register of `file`: the one whose value costs the least to give
    /// up, which is stored first if the slot does not hold it.
    ///
    /// # Panics
    /// When every register of the file holds a value the instruction needs, or
    /// the file is the flags, which cannot be stored.
    fn evict(&mut self, file: FileId) -> Result<(), CompileError> {
        let rank = |id: u64| {
            let store = match self.lives[id as usize].stored {
                true => Store::NotNeeded,
                false => Store::Needed,
            };
            let distance = self.next_read(id).map(|next| next - self.position);
            (EvictionRank::new(ReadHere::No, store, distance), id)
        };
        let victim = self
            .held
            .iter()
            .filter(|(id, lent)| {
                lent.file() == file
                    && file != FileId::Flags
                    && (**id as usize) < self.lives.len()
                    && !self.pinned.contains(id)
            })
            .map(|(&id, _)| rank(id))
            .min();
        let Some((_, victim)) = victim else {
            panic!(
                "instruction {} of {:?} holds more {file:?} registers than the file has",
                self.position, self.at
            )
        };
        self.store_at_definition(victim)?;
        let lent = self.held.remove(&victim).expect("the victim is held");
        self.free[file as usize].push(lent);
        Ok(())
    }

    /// Place `pushed`, whose reads are in registers. A write takes a free
    /// lease, or one an eviction frees; a plain write may take the lease of a
    /// read that is dead after a selected instruction; a tied write takes its
    /// read's, after a copy if the read is read again.
    fn emit(
        &mut self,
        pushed: Pushed<B::Inst<Selected>>,
        origin: Origin,
    ) -> Result<(), CompileError> {
        let first = self.numbers.len();
        self.numbers.resize(first + pushed.operands.len(), 0);
        let operands = &pushed.operands;
        for (k, operand) in operands.iter().enumerate() {
            if let Operand::Reg {
                value,
                access: Access::Read,
            } = operand
            {
                self.numbers[first + k] = self.held[&value.id].number();
            }
        }
        for (k, operand) in operands.iter().enumerate() {
            let Operand::Reg {
                value,
                access: Access::Tied { read },
            } = operand
            else {
                continue;
            };
            let Operand::Reg { value: tied, .. } = operands[*read] else {
                unreachable!("a tie names a register operand")
            };
            let lent = match self.next_read(tied.id) {
                None => self.held.remove(&tied.id).expect("a tied read is held"),
                Some(_) => self.copy_of(tied)?,
            };
            self.numbers[first + read] = lent.number();
            self.numbers[first + k] = lent.number();
            self.pinned.push(value.id);
            self.held.insert(value.id, lent);
        }
        for (k, operand) in operands.iter().enumerate() {
            if let Operand::Reg {
                value,
                access: Access::Early,
            } = operand
            {
                self.write(*value, first + k)?;
            }
        }
        if origin == Origin::Selected {
            for operand in operands {
                let Operand::Reg {
                    value,
                    access: Access::Read,
                } = operand
                else {
                    continue;
                };
                if self.next_read(value.id).is_some() {
                    continue;
                }
                if let Some(lent) = self.held.remove(&value.id) {
                    self.free[lent.file() as usize].push(lent);
                }
            }
        }
        for (k, operand) in operands.iter().enumerate() {
            if let Operand::Reg {
                value,
                access: Access::Write,
            } = operand
            {
                self.write(*value, first + k)?;
            }
        }
        if origin == Origin::Selected {
            let at = (self.block, self.records[self.block].len());
            for operand in operands {
                let Operand::Reg { value, access } = operand else {
                    continue;
                };
                if !access.reads() {
                    self.defined[value.id as usize] = Some(at);
                }
            }
        }
        let origin = match origin {
            Origin::Selected if B::rematerializable(&pushed.inst) => Origin::Remat,
            other => other,
        };
        self.records[self.block].push(Record {
            pushed,
            origin,
            regs: first,
        });
        Ok(())
    }

    /// Define `value` in a register, recording its number at `at`.
    fn write(&mut self, value: ValueName, at: usize) -> Result<(), CompileError> {
        let lent = self.acquire(value.class)?;
        self.numbers[at] = lent.number();
        self.pinned.push(value.id);
        assert!(
            self.held.insert(value.id, lent).is_none(),
            "{value:?} is defined while in a register"
        );
        Ok(())
    }

    /// A copy of `value` in a fresh register: the lease it holds.
    fn copy_of(&mut self, value: ValueName) -> Result<Lent<'m, B>, CompileError> {
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let copy = of_class!(value.class, C => B::copy::<C>(&mut spiller, value.typed()).name());
        for pushed in insts {
            self.emit(pushed, Origin::Copy)?;
        }
        Ok(self.held.remove(&copy.id).expect("a copy is held"))
    }

    /// Return the leases of the values that nothing reads again. Only a value
    /// the instruction read or defined can have just become one: the pinned.
    fn release_dead(&mut self) {
        for &id in &self.pinned {
            if self.next_read(id).is_some() {
                continue;
            }
            if let Some(lent) = self.held.remove(&id) {
                self.free[lent.file() as usize].push(lent);
            }
        }
        debug_assert!(
            self.held.keys().all(|&id| self.next_read(id).is_some()),
            "a value nothing reads again is held past the instruction that last touched it"
        );
    }
}

/// How the records are bound once the scan is over: each operand becomes the
/// token of the lease the scan recorded for it.
struct Binding<'a, 'm, B: IsaBackend> {
    frame: &'m Frame,
    /// Every lease, by file and number.
    bank: [BTreeMap<u8, Lent<'m, B>>; 4],
    /// The register number of every operand of every record.
    numbers: &'a [u8],
    /// The operand being bound.
    next: usize,
    /// Whether the instruction asked for the frame's size.
    framed: bool,
}

impl<'m, B: IsaBackend> Binding<'_, 'm, B> {
    /// `record`, bound.
    fn record(&mut self, record: &Record<B>) -> Emitted<'m, B> {
        self.next = record.regs;
        self.framed = false;
        Emitted {
            inst: B::walk::<Bound<'m, B>>(&record.pushed.inst, self),
            origin: record.origin,
        }
    }

    /// The lease of the next operand, a register of `C`'s file.
    fn lent<C: Class>(&mut self) -> &Lent<'m, B> {
        let number = self.numbers[self.next];
        self.next += 1;
        self.bank[C::File::ID as usize]
            .get(&number)
            .unwrap_or_else(|| panic!("register {number} is in no lease of {:?}", C::File::ID))
    }
}

impl<'m, B: IsaBackend> Rebind<Bound<'m, B>> for Binding<'_, 'm, B> {
    fn read<C: Class>(&mut self, _: Value<C>) -> In<'m, B, C> {
        C::File::lease(self.lent::<C>()).read()
    }
    fn write<C: Class>(&mut self, _: &build::Def<C>) -> Out<'m, B, C> {
        C::File::lease(self.lent::<C>()).write()
    }
    fn early<C: Class>(&mut self, _: &build::Early<C>) -> Out<'m, B, C> {
        C::File::lease(self.lent::<C>()).write()
    }
    fn tie<C: Class>(&mut self, _: &build::Tie<C>) -> InOut<'m, B, C> {
        let write = self.numbers[self.next + 1];
        let lent = self.lent::<C>();
        assert_eq!(
            lent.number(),
            write,
            "a tie's write is in its read's register"
        );
        let tie = C::File::lease(lent).tie();
        self.next += 1;
        tie
    }
    fn slot(&mut self, s: SlotName) -> &'m FrameSlot {
        self.next += 1;
        self.frame.slot(s)
    }
    fn target(&mut self, t: &Target) -> Label {
        self.next += 1;
        t.label
    }
    fn frame_size(&mut self) -> u64 {
        self.framed = true;
        self.frame.bytes()
    }
}
