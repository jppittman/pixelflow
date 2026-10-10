//! The local allocator: one pass over the instructions in layout order.
//!
//! A value stays in its register until something needs the register, and a
//! value that is read while in none is reloaded into a fresh one that stays
//! until it is needed in turn. What the scan gives up a register for is
//! priced by [`EvictionRank`]: the value that is defined again rather than
//! stored, then the one that costs no store, then the one read farthest out.
//!
//! **A constant is defined where it is read.** An instruction the backend can
//! recompute ([`IsaBackend::rematerializable`]: a pool load, a zero, an
//! all-ones) is not placed where it is selected. It is held until a read finds
//! its value in no register, and is placed again before that read, as often as
//! that happens. It has no slot and no store; one nothing reads is never placed.
//!
//! **A spilled value is stored right after its definition**, never at the
//! eviction point, which a guard can skip. The scan only learns that a value
//! must be stored when it first evicts it, so the store is inserted
//! retroactively: it is kept beside the definition it follows and placed
//! there when the blocks are bound. A definition that runs every trip of a
//! loop which is over by then has its store at the loop's exit instead, once.
//!
//! **A loop carries its hottest values.** The values live into a loop's head
//! that its reads save the most reloads on, and the loop's own parameters, keep
//! a register from before the head to the latch ([`plan_carries`], priced as
//! legacy's carries were and bounded by what a file has beyond a reserve). The
//! preheader makes them resident, nothing evicts them inside, and the backward
//! branch moves each parameter's next value into the parameter's register: a
//! parallel move, with the value written there in the first place wherever the
//! parameter is dead by then. Every other value live into the head is stored and
//! dropped there, so inside the loop its home is its slot and the reloads are
//! split values that are dead before the latch. A parameter that is not carried
//! is slot-homed too: a branch stores its arguments to it.
//!
//! **A join takes what the paths agree on.** A forward branch leaves its state
//! for the block it reaches, and the block starts in their intersection
//! ([`Scan::arrive`]): a value is in a register when it is in the same one on
//! every path that defines it. A value an arm defines exists on the path
//! through the arm alone, so the join takes its register from that path; one
//! some path evicted, or placed again, is read from its slot after the join,
//! and a reload made inside an arm dies at the arm's end. The slots stay
//! valid on every path because each is written right after its definition. A
//! join's parameter ([`Scan::join`]) is in a register too: the first branch to
//! reach it gives it one, as a loop's entering branch does, and each later
//! branch moves its argument there.
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

use super::policy::{
    self, Budget, CARRY_RESERVE, Candidate, EvictionRank, GENERAL_CARRY_RESERVE, ReadHere, Store,
};
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
    /// A definition the backend can recompute, placed again where a read needs
    /// its value in a register ([`IsaBackend::rematerializable`]):
    /// `EmitTraffic`'s remats when it reads the constant pool
    /// ([`IsaBackend::reads_pool`]).
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
    /// The latch of the loop that carries it: its register is its own, and
    /// nothing evicts it, up to that position.
    carried_until: Option<usize>,
}

/// What the allocator knows of one loop before it places anything.
struct LoopFacts {
    /// Its first position and its last, the one backward branch.
    head: usize,
    latch: usize,
    /// The block that follows the latch's.
    exit: usize,
    /// How many times its body runs per call.
    trips: u64,
    parent: Option<usize>,
    /// The head's parameters, and what the backward branch passes to them.
    params: Vec<ValueName>,
    latch_args: Vec<ValueName>,
}

impl LoopFacts {
    fn takes(&self, param: u64) -> bool {
        self.params.iter().any(|p| p.id == param)
    }
}

/// Every value's [`Life`], and the positions that read it.
struct Liveness {
    lives: Vec<Life>,
    /// `(value, position)` for every read, in layout order.
    reads: Vec<(u64, usize)>,
    /// Each loop, from its head to its one backward branch.
    loops: Vec<LoopFacts>,
    /// How many instructions there are.
    positions: usize,
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
                carried_until: None,
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
            loops: function
                .loops
                .iter()
                .map(|l| LoopFacts {
                    head: 0,
                    latch: 0,
                    exit: 0,
                    trips: l.trips,
                    parent: l.parent,
                    params: Vec::new(),
                    latch_args: Vec::new(),
                })
                .collect(),
            positions: 0,
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
                let facts = &mut live.loops[l];
                facts.head = starts[at[&target.label]];
                facts.latch = position - 1;
                facts.exit = i + 1;
                facts.params = function.blocks[at[&target.label]].params.clone();
                facts.latch_args = target.args.clone();
            }
        }
        live.positions = position;

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

    /// Group the reads by value, and give each value the range of the returned
    /// positions that are its own.
    fn index_reads(&mut self) -> Vec<usize> {
        self.reads.sort_unstable();
        let mut start = 0;
        while let Some(&(value, _)) = self.reads.get(start) {
            let len = self.reads[start..].partition_point(|&(v, _)| v == value);
            self.lives[value as usize].reads = start..start + len;
            start += len;
        }
        self.reads.iter().map(|&(_, position)| position).collect()
    }

    /// Value `v` is read at `position`.
    fn read(&mut self, v: ValueName, position: usize) {
        self.reads.push((v.id, position));
        let life = &mut self.lives[v.id as usize];
        life.read = true;
        life.last = life.last.max(position);
        for facts in &self.loops {
            if (facts.head..=facts.latch).contains(&position) && life.def < facts.head {
                life.last = life.last.max(facts.latch);
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

/// The values each loop carries in registers, from before its head to its
/// latch: the ones whose reads inside it save the most reloads, while no loop
/// has more of a file's registers carried than the file's budget.
///
/// A candidate is a value live into a loop's head, or one of its parameters,
/// and it is priced by what its reads inside the loop would cost: one reload
/// per read per time the read runs, and for a parameter one store per trip.
/// A value is a candidate of the outermost loop that it is live into, so a
/// carry is not counted once for the loop and again for each loop nested in it.
fn plan_carries(live: &Liveness, reads: &[usize], budget: Budget) -> Vec<Vec<u64>> {
    let loops = &live.loops;

    // What one read costs, by position: the trips of the innermost loop it is
    // in. A nest is listed outer loop first.
    let mut trips_at = alloc::vec![1usize; live.positions];
    for facts in loops {
        let trips = usize::try_from(facts.trips).unwrap_or(usize::MAX);
        trips_at[facts.head..=facts.latch].fill(trips);
    }

    // A carry across a loop is live across every loop inside it.
    let mut inside: Vec<Vec<usize>> = (0..loops.len()).map(|l| alloc::vec![l]).collect();
    for (j, facts) in loops.iter().enumerate() {
        let mut outer = facts.parent;
        while let Some(o) = outer {
            inside[o].push(j);
            outer = loops[o].parent;
        }
    }

    let mut candidates = Vec::new();
    for (id, life) in live.lives.iter().enumerate() {
        let file = life.class.file();
        if !life.read || budget[file as usize] == 0 {
            continue;
        }
        let own = &reads[life.reads.clone()];
        let mut chosen: Vec<usize> = Vec::new();
        for (l, facts) in loops.iter().enumerate() {
            // A loop's parameters are defined at its head.
            let param = life.def == facts.head && facts.takes(id as u64);
            if life.def >= facts.head && !param {
                continue;
            }
            let (from, to) = (
                own.partition_point(|&p| p < facts.head),
                own.partition_point(|&p| p <= facts.latch),
            );
            if from == to {
                continue;
            }
            let covered = chosen
                .iter()
                .any(|&o| (loops[o].head..=loops[o].latch).contains(&facts.head));
            if covered {
                continue;
            }
            let reloads: usize = own[from..to].iter().map(|&p| trips_at[p]).sum();
            let latch_stores = match param {
                true => usize::try_from(facts.trips).unwrap_or(usize::MAX),
                false => 0,
            };
            chosen.push(l);
            candidates.push(Candidate {
                weight: reloads + latch_stores,
                class: file,
                live_across: inside[l].clone(),
                root: (id as u64, l),
            });
        }
    }

    let mut plan = alloc::vec![Vec::new(); loops.len()];
    for (value, l) in policy::carried(candidates, loops.len(), budget) {
        plan[l].push(value);
    }
    plan
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
/// flags write or held to the end of its block, a carried value that is not in
/// its register at its loop's latch, a loop parameter read past the latch, a
/// backward branch whose moves form a register cycle (no producer before a loop
/// passes one parameter to another), a latch move into a register that a value
/// other than the loop's parameters and the moved arguments holds, a value
/// dropped at a join that is in no slot, a block no scanned branch reaches, or
/// an entry block whose first instruction does not make the frame.
pub(in crate::emit) fn allocate<'m, B: IsaBackend>(
    function: Function<B>,
    pool: &'m Pool<B>,
    frame: &'m mut Frame,
) -> Result<Allocated<'m, B>, CompileError> {
    let mut live = Liveness::of(&function);
    let reads = live.index_reads();
    layout(&mut live.lives, frame)?;
    let hoisted = live.lives.iter().filter(|life| life.hoisted).count() as u64;
    let budget = [
        B::FILE
            .members(FileId::Vector)
            .len()
            .saturating_sub(CARRY_RESERVE),
        B::FILE
            .members(FileId::General)
            .len()
            .saturating_sub(GENERAL_CARRY_RESERVE),
        0,
        0,
    ];
    let plan = plan_carries(&live, &reads, budget);
    let Liveness { lives, loops, .. } = live;

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
    let loop_at = function
        .loops
        .iter()
        .enumerate()
        .map(|(i, l)| (l.head, i))
        .collect();
    let mut scan = Scan {
        frame: &mut *frame,
        reads,
        lives,
        blocks_at,
        loops,
        plan,
        loop_at,
        active: Vec::new(),
        want: BTreeMap::new(),
        edges: BTreeMap::new(),
        joined: BTreeMap::new(),
        free,
        held,
        pinned: Vec::new(),
        next: function.classes.len() as u64,
        defined,
        insts: Vec::new(),
        remats: BTreeMap::new(),
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
        scan.head(block.label)?;
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
        scan.held.is_empty() && scan.active.is_empty() && scan.edges.is_empty(),
        "values are held after the last instruction: {:?}, loops are open: {}, and branches to {:?} were never arrived at",
        scan.held.keys(),
        scan.active.len(),
        scan.edges.keys()
    );
    let Scan {
        free,
        insts,
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
        insts: &insts,
        bank,
        numbers: &numbers,
        next: 0,
        framed: false,
    };
    let mut late = late;
    let mut blocks = Vec::with_capacity(records.len());
    for (b, (block, (label, scope))) in records.into_iter().zip(shapes).enumerate() {
        let mut insts = Vec::with_capacity(block.len());
        for stored in late.remove(&StorePoint::Entry(b)).unwrap_or_default() {
            insts.push(binding.record(&stored));
        }
        for (k, record) in block.into_iter().enumerate() {
            insts.push(binding.record(&record));
            assert!(
                (b, k) != (0, 0) || binding.framed,
                "the entry block's first instruction does not make the frame, so the ABI registers have nowhere to be stored"
            );
            for stored in late.remove(&StorePoint::After(b, k)).unwrap_or_default() {
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

/// An instruction the scan has placed: which of the scan's instructions to
/// bind, why it is there, and where in the scan's numbers its operands'
/// registers begin. Placing one instruction twice (a rematerialization) is two
/// records of one instruction.
struct Record {
    inst: usize,
    origin: Origin,
    regs: usize,
}

/// Where an evicted value's store goes in the blocks.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum StorePoint {
    /// Right after the record at this block and index.
    After(usize, usize),
    /// At the start of this block.
    Entry(usize),
}

/// A loop the scan is inside: where its carried values live.
struct Active {
    index: usize,
    /// Each carried parameter, and the number of the register that is its home
    /// from the loop's entry to its latch.
    homes: Vec<(u64, u8)>,
    /// Each carried value defined before the loop, and its register.
    carried: Vec<(u64, u8)>,
}

/// The register each value in one holds, by file and number.
type Residents = BTreeMap<u64, (FileId, u8)>;

/// What a forward branch leaves for the block it reaches.
struct Edge {
    /// The position of the branch.
    from: usize,
    residents: Residents,
}

/// One register-to-register move of a branch's parallel move.
struct Move {
    /// A value of the moved class, to type the copy.
    value: ValueName,
    from: u8,
    to: u8,
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
    loops: Vec<LoopFacts>,
    /// The values each loop carries (see [`plan_carries`]), and each loop's
    /// index by its head.
    plan: Vec<Vec<u64>>,
    loop_at: BTreeMap<Label, usize>,
    /// The loops the scan is inside, innermost last.
    active: Vec<Active>,
    /// A value a backward branch passes, by the parameter it is passed to: it
    /// is written into that parameter's register when that is free of any
    /// other use, which makes the branch's move for it nothing.
    want: BTreeMap<u64, u64>,
    /// What each forward branch left for the block it reaches, by block, until
    /// the scan gets there.
    edges: BTreeMap<Label, Vec<Edge>>,
    /// A forward join's parameter, once a branch has given it a register: its
    /// block and the register's number.
    joined: BTreeMap<u64, (usize, u8)>,
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
    /// Every instruction the scan has been given or has inserted, in the order
    /// it met them; a record names one by its index.
    insts: Vec<B::Inst<Selected>>,
    /// The selected instructions that are placed only where a read needs the
    /// value, by the value each defines: an index into `insts`.
    remats: BTreeMap<u64, usize>,
    /// The instructions placed so far, by block.
    records: Vec<Vec<Record>>,
    /// The stores of evicted values, by where each goes.
    late: BTreeMap<StorePoint, Vec<Record>>,
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
    /// One selected instruction: its reads in registers, the loops it enters
    /// carrying theirs, its branch arguments passed, itself, and the registers
    /// it is done with returned.
    fn instruction(&mut self, pushed: Pushed<B::Inst<Selected>>) -> Result<(), CompileError> {
        if B::rematerializable(&pushed.inst) {
            self.defer(pushed);
            return Ok(());
        }
        let targets: Vec<&Target> = pushed
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Target(target) => Some(target),
                Operand::Reg { .. } | Operand::Frame(_) => None,
            })
            .collect();
        let args: Vec<ValueName> = targets
            .iter()
            .flat_map(|target| target.args.iter().copied())
            .collect();
        let reads: Vec<ValueName> = pushed
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Reg { value, access } if access.reads() => Some(*value),
                Operand::Reg { .. } | Operand::Target(_) | Operand::Frame(_) => None,
            })
            .chain(args.iter().copied())
            .collect();
        self.pinned.clear();
        self.pinned.extend(reads.iter().map(|read| read.id));
        for read in reads {
            if !self.held.contains_key(&read.id) {
                self.reload_into(read)?;
            }
        }
        for target in &targets {
            if let Some(l) = self.entered(target) {
                self.enter(l)?;
            }
        }
        for target in &targets {
            self.pass(target, &args)?;
        }
        let forward: Vec<Label> = targets
            .iter()
            .map(|target| target.label)
            .filter(|label| self.blocks_at[label].0 > self.block)
            .collect();
        self.emit(pushed, Origin::Selected)?;
        self.release_dead();
        for target in forward {
            self.leave(target);
        }
        Ok(())
    }

    /// Hold `pushed`, a definition the backend can recompute, until a read
    /// needs its value in a register ([`Self::reload_into`]). One nothing reads
    /// is never placed.
    ///
    /// # Panics
    /// When the instruction does more than define one value, or defines flags,
    /// which a placement elsewhere could clobber.
    fn defer(&mut self, pushed: Pushed<B::Inst<Selected>>) {
        let [
            Operand::Reg {
                value,
                access: Access::Write,
            },
        ] = pushed.operands[..]
        else {
            panic!("a rematerializable instruction defines one value and reads none")
        };
        assert_ne!(
            value.class,
            ClassId::Flags,
            "{value:?} is rematerializable, and a rematerialization would clobber the flags"
        );
        let inst = self.keep(pushed.inst);
        self.remats.insert(value.id, inst);
    }

    /// The loop `target` is the way into: a forward branch to its head.
    fn entered(&self, target: &Target) -> Option<usize> {
        let forward = self.blocks_at[&target.label].0 > self.block;
        self.loop_at.get(&target.label).copied().filter(|_| forward)
    }

    /// Whether the scan is inside the region `value` is carried across, so its
    /// register is its own.
    fn kept(&self, value: u64) -> bool {
        let carried = self.lives.get(value as usize).and_then(|l| l.carried_until);
        carried.is_some_and(|latch| self.position <= latch)
    }

    /// Whether `value` is wanted after the instruction being placed: it is
    /// read again, or its loop has not reached its latch.
    fn live_after(&self, value: u64) -> bool {
        let carried = self.lives.get(value as usize).and_then(|l| l.carried_until);
        self.next_read(value).is_some() || carried.is_some_and(|latch| self.position < latch)
    }

    /// Enter loop `l`: its carried values are in registers before its head and
    /// are kept there to its latch. A parameter's register is chosen by the
    /// branch that fills it ([`Self::home`]).
    fn enter(&mut self, l: usize) -> Result<(), CompileError> {
        let latch = self.loops[l].latch;
        for &value in &self.plan[l] {
            self.lives[value as usize].carried_until = Some(latch);
        }
        for value in self.plan[l].clone() {
            let life = &self.lives[value as usize];
            let name = ValueName {
                id: value,
                class: life.class,
            };
            if !self.loops[l].takes(value) && !self.held.contains_key(&value) {
                self.reload_into(name)?;
            }
        }
        self.active.push(Active {
            index: l,
            homes: Vec::new(),
            carried: Vec::new(),
        });
        Ok(())
    }

    /// The start of block `label`, in the state every forward branch to it
    /// agrees on ([`Self::arrive`]). A loop's head is entered in one state
    /// however it is reached: the carried values in their registers and
    /// everything else in its slot, which `flush` makes true.
    fn head(&mut self, label: Label) -> Result<(), CompileError> {
        self.arrive(label);
        let Some(&l) = self.loop_at.get(&label) else {
            return Ok(());
        };
        self.flush()?;
        let facts = &self.loops[l];
        let active = self.active.last_mut().expect("a head is entered");
        assert_eq!(
            active.index, l,
            "{label:?} is entered from outside its loop"
        );
        for &value in self.plan[l].iter().filter(|&&v| !facts.takes(v)) {
            let lent = self.held.get(&value);
            let lent = lent.expect("a value carried across a loop is in a register at its head");
            active.carried.push((value, lent.number()));
        }
        for (&arg, param) in facts.latch_args.iter().zip(&facts.params) {
            let homed = active.homes.iter().any(|&(p, _)| p == param.id);
            if homed && !facts.takes(arg.id) {
                self.want.entry(arg.id).or_insert(param.id);
            }
        }
        Ok(())
    }

    /// Leave the state as it is for `label`, a block after this one, which
    /// takes it up when the scan gets there.
    fn leave(&mut self, label: Label) {
        let residents: Residents = self
            .held
            .iter()
            .map(|(&id, lent)| {
                assert!(
                    (id as usize) < self.lives.len() && lent.file() != FileId::Flags,
                    "value {id} is held at a branch to {label:?}, and no flags value or temporary is live at a label"
                );
                (id, (lent.file(), lent.number()))
            })
            .collect();
        let from = self.position;
        self.edges
            .entry(label)
            .or_default()
            .push(Edge { from, residents });
    }

    /// Enter the block at `label` in the state every forward branch to it
    /// leaves agreeing: a value is in a register when it is in the same one
    /// on every path that defines it, and nothing else is. A value some path
    /// has evicted is read from its slot after the join, and a register a path
    /// filled with a reload that the others did not make is free again: a
    /// reload made inside an arm dies at the arm's end.
    ///
    /// # Panics
    /// When a branch to the block was never scanned, or a value that would
    /// be dropped is in no slot, or a carried value is not in a register on
    /// every path.
    fn arrive(&mut self, label: Label) {
        if self.block == 0 {
            return;
        }
        let edges = self
            .edges
            .remove(&label)
            .unwrap_or_else(|| panic!("{label:?} is reached by no branch the scan has seen"));
        // The block after the branch, entered by it alone, is in its state.
        if let [Edge { from, .. }] = edges[..]
            && from + 1 == self.position
        {
            return;
        }
        let start = self.position;
        let ids: BTreeSet<u64> = edges
            .iter()
            .flat_map(|e| e.residents.keys().copied())
            .collect();
        let mut agreed = Residents::new();
        for id in ids {
            let life = &self.lives[id as usize];
            let carried = life.carried_until.is_some_and(|latch| start <= latch);
            if !carried && !self.read_from(id, start) {
                continue;
            }
            let mut register = None;
            let mut kept = true;
            for edge in &edges {
                match (edge.residents.get(&id), register) {
                    (Some(&at), None) => register = Some(at),
                    (Some(&at), Some(first)) => kept &= at == first,
                    // On a path the definition comes after, the value does not
                    // exist, and takes its register from the paths where it
                    // does. On any other it was evicted, or placed again there.
                    (None, _) => {
                        kept &= life.def > edge.from && !self.remats.contains_key(&id);
                    }
                }
            }
            let register = register.expect("a candidate is in a register on some path");
            match kept {
                true => {
                    agreed.insert(id, register);
                }
                false => {
                    assert!(
                        !carried && (life.stored || self.remats.contains_key(&id)),
                        "value {id} is not in the same register on every path to {label:?}, and is in no slot"
                    );
                }
            }
        }
        self.restore(&agreed);
    }

    /// Whether `value` is read at `position` or after.
    fn read_from(&self, value: u64, position: usize) -> bool {
        let own = &self.reads[self.lives[value as usize].reads.clone()];
        own.partition_point(|&p| p < position) < own.len()
    }

    /// Put the leases where `residents` says: each value in its register,
    /// every other register free.
    fn restore(&mut self, residents: &Residents) {
        let held = core::mem::take(&mut self.held).into_values();
        let free = self.free.iter_mut().flat_map(|bank| bank.drain(..));
        let mut leases: BTreeMap<(FileId, u8), Lent<'m, B>> = held
            .chain(free)
            .map(|lent| ((lent.file(), lent.number()), lent))
            .collect();
        for (&id, at) in residents {
            let lent = leases.remove(at);
            let lent = lent.unwrap_or_else(|| panic!("value {id} is in a register no lease has"));
            self.held.insert(id, lent);
        }
        for lent in leases.into_values() {
            self.free[lent.file() as usize].push(lent);
        }
    }

    /// Pass `target`'s arguments to its parameters. A parameter the loop
    /// carries is in a register: the entering branch gives it one
    /// ([`Self::home`]), and a backward branch moves its argument there. Any
    /// other is stored to its slot. A join's parameter is in a register too
    /// ([`Self::join`]). Every argument is in a register before the first one
    /// is passed, so the moves are a parallel move however they overlap.
    fn pass(&mut self, target: &Target, args: &[ValueName]) -> Result<(), CompileError> {
        let params = self.blocks_at[&target.label].1.clone();
        let back = self.blocks_at[&target.label].0 <= self.block;
        let carrying = self.loop_at.get(&target.label).copied();
        let mut moves = Vec::new();
        for (&arg, param) in target.args.iter().zip(&params) {
            if arg.id == param.id || !self.lives[param.id as usize].read {
                continue;
            }
            let Some(l) = carrying else {
                self.join(*param, arg, args, self.blocks_at[&target.label].0)?;
                continue;
            };
            let carried = self.plan[l].contains(&param.id);
            match (carried, back) {
                (true, false) => {
                    let lent = self.home(arg, args)?;
                    let active = self.active.last_mut().expect("a loop is being entered");
                    active.homes.push((param.id, lent.number()));
                    self.held.insert(param.id, lent);
                }
                (true, true) => moves.push((arg, *param)),
                (false, _) => {
                    let slot = self.slot(*param)?;
                    self.lives[param.id as usize].stored = true;
                    for pushed in self.spill_of(arg, slot) {
                        self.emit(pushed, Origin::Spill)?;
                    }
                }
            }
        }
        if let Some(l) = carrying.filter(|_| back) {
            self.latch(l, &moves);
        }
        Ok(())
    }

    /// A register for a parameter that `arg` is passed to: the lease of the
    /// argument when nothing else wants that, and a copy's otherwise.
    fn home(&mut self, arg: ValueName, args: &[ValueName]) -> Result<Lent<'m, B>, CompileError> {
        let once = args.iter().filter(|other| other.id == arg.id).count() == 1;
        match once && !self.live_after(arg.id) {
            true => Ok(self.held.remove(&arg.id).expect("an argument is held")),
            false => self.copy_of(arg),
        }
    }

    /// Pass `arg` to `param`, a parameter of the forward join at block `join`.
    /// The first branch to reach the join gives `param` its register
    /// ([`Self::home`]); every later one puts its argument in that register,
    /// after storing whatever else holds it there.
    fn join(
        &mut self,
        param: ValueName,
        arg: ValueName,
        args: &[ValueName],
        join: usize,
    ) -> Result<(), CompileError> {
        let Some(&(_, to)) = self.joined.get(&param.id) else {
            let lent = self.home(arg, args)?;
            self.joined.insert(param.id, (join, lent.number()));
            self.held.insert(param.id, lent);
            return Ok(());
        };
        assert!(
            !self.held.contains_key(&param.id),
            "{param:?} is in a register before the branch that passes it its argument"
        );
        let file = arg.class.file();
        let from = self.held[&arg.id].number();
        if from != to {
            let lent = self.vacate(file, to)?;
            self.held.insert(param.id, lent);
            self.move_register(&Move {
                value: arg,
                from,
                to,
            });
            return Ok(());
        }
        // The argument is in the parameter's register already, and is read
        // again: it goes to its slot, and the parameter has the register.
        if self.live_after(arg.id) {
            self.store_at_definition(arg.id)?;
        }
        let lent = self.held.remove(&arg.id).expect("an argument is held");
        self.held.insert(param.id, lent);
        Ok(())
    }

    /// The lease of register `number` of `file`, which the value in it, if
    /// any, gives up: it is stored first when the slot does not hold it.
    ///
    /// # Panics
    /// When the value is one the instruction being placed needs, or a carried
    /// one inside its loop.
    fn vacate(&mut self, file: FileId, number: u8) -> Result<Lent<'m, B>, CompileError> {
        let holder = self
            .held
            .iter()
            .find(|(_, lent)| lent.file() == file && lent.number() == number)
            .map(|(&id, _)| id);
        if let Some(holder) = holder {
            assert!(
                !self.pinned.contains(&holder) && !self.kept(holder),
                "instruction {} of {:?} needs register {number} of {file:?}, which value {holder} must keep",
                self.position,
                self.at
            );
            self.store_at_definition(holder)?;
            let lent = self.held.remove(&holder).expect("the holder is held");
            self.free[file as usize].push(lent);
        }
        let bank = &mut self.free[file as usize];
        let at = bank.iter().position(|lent| lent.number() == number);
        Ok(bank.swap_remove(at.expect("a register no value holds is free")))
    }

    /// The backward branch of loop `l`: move each carried parameter's argument
    /// into the parameter's register, and leave the loop.
    ///
    /// # Panics
    /// When a carried value is not in the register it had at the head, a
    /// parameter is read past the latch, a move's destination holds a value
    /// other than the loop's own parameters and the arguments being moved, or
    /// the moves form a register cycle.
    fn latch(&mut self, l: usize, moves: &[(ValueName, ValueName)]) {
        let active = self.active.pop().expect("a latch is inside its loop");
        assert_eq!(active.index, l, "{:?} closes a loop it is not in", self.at);
        for &(value, number) in &active.carried {
            let at = self.held.get(&value).map(|lent| lent.number());
            assert_eq!(
                at,
                Some(number),
                "value {value} is carried across the loop headed {:?}, and is not in its register at the latch",
                self.loops[l].head
            );
        }
        for param in &self.loops[l].params {
            assert!(
                self.next_read(param.id).is_none(),
                "{param:?} is read after the latch of its loop, where the next trip's value has replaced it"
            );
        }
        let mut pending = Vec::new();
        for &(arg, param) in moves {
            let (_, to) = *active
                .homes
                .iter()
                .find(|&&(p, _)| p == param.id)
                .expect("a carried parameter has a home");
            let from = self.held[&arg.id].number();
            if from == to {
                continue;
            }
            // Only the loop's own dead parameters, and the arguments being
            // moved, may be in a register a move writes.
            let holder = self
                .held
                .iter()
                .find(|(_, lent)| lent.file() == arg.class.file() && lent.number() == to);
            if let Some((&holder, _)) = holder {
                let homed = active.homes.iter().any(|&(p, _)| p == holder);
                let moved = moves.iter().any(|&(a, _)| a.id == holder);
                assert!(
                    homed || moved,
                    "the latch of the loop headed {:?} moves {arg:?} into register {to}, which value {holder} holds",
                    self.loops[l].head
                );
            }
            pending.push(Move {
                value: arg,
                from,
                to,
            });
        }
        self.sequence(pending);
        self.want
            .retain(|_, param| !active.homes.iter().any(|&(p, _)| p == *param));
        let expiring = active.carried.iter().chain(&active.homes);
        self.pinned.extend(expiring.map(|&(value, _)| value));
    }

    /// Emit `pending`, a parallel move, in an order no move overwrites what
    /// another reads.
    ///
    /// # Panics
    /// When the moves form a register cycle: no selected function passes one
    /// loop parameter to another, so breaking one through a fresh value lands
    /// with its first producer.
    fn sequence(&mut self, mut pending: Vec<Move>) {
        while !pending.is_empty() {
            let free = pending.iter().position(|m| {
                !pending.iter().any(|other| {
                    other.value.class.file() == m.value.class.file() && other.from == m.to
                })
            });
            let free = free.unwrap_or_else(|| {
                panic!(
                    "{:?}: a backward branch's moves form a register cycle; no selected function passes a loop parameter to another",
                    self.at
                )
            });
            let m = pending.remove(free);
            self.move_register(&m);
        }
    }

    /// Copy register `m.from` to register `m.to`, which no value uses.
    fn move_register(&mut self, m: &Move) {
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        of_class!(m.value.class, C => B::copy::<C>(&mut spiller, m.value.typed()).name());
        for pushed in insts {
            let regs = self.numbers.len();
            for operand in &pushed.operands {
                self.numbers.push(match operand {
                    Operand::Reg { access, .. } if access.reads() => m.from,
                    Operand::Reg { .. } => m.to,
                    Operand::Target(_) | Operand::Frame(_) => 0,
                });
            }
            let inst = self.keep(pushed.inst);
            self.records[self.block].push(Record {
                inst,
                origin: Origin::Copy,
                regs,
            });
        }
    }

    /// Make `value` resident: define it again if it is rematerializable, and
    /// otherwise reload it from its slot, into the register the instruction
    /// will read it from. It stays there until something needs the register.
    fn reload_into(&mut self, value: ValueName) -> Result<(), CompileError> {
        assert_ne!(
            value.class,
            ClassId::Flags,
            "{value:?} is read at instruction {} of {:?}, which does not hold the flags",
            self.position,
            self.at
        );
        if let Some(&inst) = self.remats.get(&value.id) {
            let define = [Operand::Reg {
                value,
                access: Access::Write,
            }];
            return self.place(inst, &define, Origin::Remat);
        }
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

    /// Where `value`'s store goes: right after its definition, or, when it is
    /// defined in a loop that is over by now, at the loop's exit, where it
    /// runs once and not every trip. Either dominates every read the slot is
    /// for, and the value has stayed in its register since the definition.
    fn store_point(&self, value: u64) -> StorePoint {
        let def = self.lives[value as usize].def;
        // The outermost, because the nest is listed outer loop first.
        let over = self
            .loops
            .iter()
            .find(|l| (l.head..=l.latch).contains(&def) && l.latch < self.position);
        if let Some(l) = over {
            return StorePoint::Entry(l.exit);
        }
        // A join's parameter is defined where the join begins.
        if let Some(&(block, _)) = self.joined.get(&value) {
            return StorePoint::Entry(block);
        }
        let (block, index) =
            self.defined[value as usize].expect("a value in a register was defined");
        StorePoint::After(block, index)
    }

    /// Store `value`, which is in its defining register, right after its
    /// definition, unless that is done, or it can be defined again instead.
    fn store_at_definition(&mut self, value: u64) -> Result<(), CompileError> {
        let life = &self.lives[value as usize];
        if life.stored || self.remats.contains_key(&value) {
            return Ok(());
        }
        let name = ValueName {
            id: value,
            class: life.class,
        };
        let slot = self.slot(name)?;
        self.lives[value as usize].stored = true;
        let at = self.store_point(value);
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
            let inst = self.keep(pushed.inst);
            self.late.entry(at).or_default().push(Record {
                inst,
                origin: Origin::Spill,
                regs,
            });
        }
        Ok(())
    }

    /// Drop every value from its register but the carried ones, storing the
    /// ones not yet stored: what a loop head expects of the state it is
    /// entered in.
    fn flush(&mut self) -> Result<(), CompileError> {
        let dropped: Vec<u64> = self
            .held
            .keys()
            .copied()
            .filter(|&id| !self.kept(id))
            .collect();
        for id in dropped {
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

    /// Whether `value` is read after the instruction being placed, by the
    /// code below it or by the next trip of a loop it is live across: a loop
    /// inside the value's own, which the value outlives.
    fn read_again(&self, value: u64) -> bool {
        let life = &self.lives[value as usize];
        let own = &self.reads[life.reads.clone()];
        let around = |l: &LoopFacts| {
            life.def < l.head
                && (l.head..=l.latch).contains(&self.position)
                && own.iter().any(|p| (l.head..=l.latch).contains(p))
        };
        self.next_read(value).is_some() || self.loops.iter().any(around)
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
            let store = match (
                self.remats.contains_key(&id),
                self.lives[id as usize].stored,
            ) {
                (true, _) => Store::Never,
                (false, true) => Store::NotNeeded,
                (false, false) => Store::Needed,
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
                    && !self.kept(**id)
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
        let inst = self.keep(pushed.inst);
        self.place(inst, &pushed.operands, origin)
    }

    /// Place instruction `inst`, whose operands are `operands`, as
    /// [`Self::emit`] does.
    fn place(
        &mut self,
        inst: usize,
        operands: &[Operand],
        origin: Origin,
    ) -> Result<(), CompileError> {
        let first = self.numbers.len();
        self.numbers.resize(first + operands.len(), 0);
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
            let lent = match self.live_after(tied.id) {
                false => self.held.remove(&tied.id).expect("a tied read is held"),
                true => self.copy_of(tied)?,
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
                if self.live_after(value.id) {
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
                self.write_plain(*value, first + k)?;
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
        self.records[self.block].push(Record {
            inst,
            origin,
            regs: first,
        });
        Ok(())
    }

    /// `inst`, now one of the scan's.
    fn keep(&mut self, inst: B::Inst<Selected>) -> usize {
        self.insts.push(inst);
        self.insts.len() - 1
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

    /// Define `value` in a register as [`Self::write`] does, except that a value
    /// a backward branch passes to a carried parameter that is read no more
    /// takes the parameter's register, so the branch has nothing to move. The
    /// value is then kept there to the latch, as the parameter was.
    fn write_plain(&mut self, value: ValueName, at: usize) -> Result<(), CompileError> {
        let home = self
            .want
            .get(&value.id)
            .copied()
            .filter(|&param| !self.read_again(param))
            .and_then(|param| Some((param, self.held.remove(&param)?)));
        let Some((param, lent)) = home else {
            return self.write(value, at);
        };
        self.lives[value.id as usize].carried_until = self.lives[param as usize].carried_until;
        self.numbers[at] = lent.number();
        self.pinned.push(value.id);
        self.held.insert(value.id, lent);
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
            if self.live_after(id) {
                continue;
            }
            if let Some(lent) = self.held.remove(&id) {
                self.free[lent.file() as usize].push(lent);
            }
        }
        debug_assert!(
            self.held.keys().all(|&id| self.live_after(id)),
            "a value nothing reads again is held past the instruction that last touched it"
        );
    }
}

/// How the records are bound once the scan is over: each operand becomes the
/// token of the lease the scan recorded for it.
struct Binding<'a, 'm, B: IsaBackend> {
    frame: &'m Frame,
    /// Every instruction the scan placed, which the records name.
    insts: &'a [B::Inst<Selected>],
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
    fn record(&mut self, record: &Record) -> Emitted<'m, B> {
        self.next = record.regs;
        self.framed = false;
        Emitted {
            inst: B::walk::<Bound<'m, B>>(&self.insts[record.inst], self),
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
