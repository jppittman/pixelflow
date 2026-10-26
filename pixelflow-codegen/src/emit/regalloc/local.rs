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
//! a register from before the head to the latch ([`plan_carries`], priced by
//! the reloads they save and bounded by what a file has beyond a reserve). The
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
//! valid on every path because each is written right after its definition.
//!
//! The slots are the second register file. Liveness is intervals in layout
//! order: a value lives from its definition to its last read, a value defined
//! before a loop and read inside it lives to the loop's latch, and a block
//! parameter lives from the first branch that passes it an argument. The
//! narrow slots are laid out before the scan, from the peak number of narrow
//! values live at once; a vector slot is leased for one value's interval when
//! the value is first stored.
//!
//! **Each instruction is bound as it is placed.** The scan holds every lease,
//! so an operand's token is borrowed from the lease its value holds at that
//! point; a read whose register a write of the same instruction takes is bound
//! through the write's. A slot is a descriptor fixed at mint, and the frame's
//! size reaches `Enter` and `Ret` through a handle the scan fills when it
//! ends. There is no register-number table.

use super::policy::{
    self, Budget, CARRY_RESERVE, Candidate, EvictionRank, GENERAL_CARRY_RESERVE, Store,
};
use super::resource::{Frame, FrameBytes, In, InOut, Leases, Lent, Out, Pool, SlotLease, SlotName};
use super::{Allocated, Emitted, Origin, RegisterAllocator};
use crate::emit::asm::Label;
use crate::emit::build::{self, Spiller};
use crate::emit::{
    Access, Block, Bound, Class, ClassId, File, FileId, FrameSlot, Function, IsaBackend, Operand,
    Pushed, Rebind, Selected, Target, Value, ValueName,
};
use crate::error::CompileError;
use crate::pass::Pass;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::ops::Range;

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
    /// The selected instruction that defines it again, for a value that is
    /// placed only where a read needs it ([`IsaBackend::rematerializable`]): an
    /// index into the scan's instructions.
    remat: Option<usize>,
    /// Whether its definition can be placed again, known before the scan: such
    /// a value is never stored, so it needs no slot.
    recompute: Recompute,
    /// The general registers lent to nothing right after its definition, by
    /// number: the allocator's record of that point, which a store placed
    /// there behind the scan ([`Scan::store_at_definition`]) draws on.
    free_general: u64,
}

/// Whether a value's definition can be placed again, and what that reads.
#[derive(Clone, Copy)]
enum Recompute {
    No,
    /// It reads no register: a zero, an all-ones, an immediate, an x86
    /// RIP-relative pool load.
    Leaf,
    /// It reads one value: an aarch64 pool load reads the pool's base. A read
    /// of the value is a read of that too, because placing the definition
    /// again reads it.
    Reads(ValueName),
}

/// What the allocator knows of one loop before it places anything.
struct LoopFacts {
    /// Its first position and its last, the one backward branch.
    head: usize,
    latch: usize,
    /// The block that follows the latch's.
    exit: usize,
    /// How many times its body runs per call, an estimate's price for a fold
    /// whose range is not known.
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
                remat: None,
                recompute: Recompute::No,
                free_general: 0,
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
                    trips: l.trips.weight(),
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
                if B::rematerializable(&pushed.inst)
                    && let [Operand::Reg { value, .. }, reads @ ..] = &pushed.operands[..]
                {
                    let life = &mut live.lives[value.id as usize];
                    let mut reads = reads.iter().filter_map(|operand| match operand {
                        Operand::Reg { value, .. } => Some(*value),
                        Operand::Target(_) | Operand::Frame(_) => None,
                    });
                    life.recompute = reads.next().map_or(Recompute::Leaf, Recompute::Reads);
                    assert!(
                        reads.next().is_none(),
                        "a rematerializable instruction reads at most one value"
                    );
                }
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
        if let Recompute::Reads(base) = life.recompute {
            self.read(base, position);
        }
    }
}

/// Give every narrow value that is read a slot; lay the narrow region out for
/// them. A vector value's slot is leased when it is first stored.
fn layout(lives: &mut [Life], frame: &mut Frame) -> Result<(), CompileError> {
    let mut narrow: Vec<usize> = (0..lives.len())
        .filter(|&id| {
            lives[id].read
                && matches!(lives[id].recompute, Recompute::No)
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
            // The sum saturates. A read weighs its nest's trips, `u64::MAX` for
            // a nest priced past `u64`, so adding two such reads would overflow;
            // a value read that deep weighs the most a weight can and compiles.
            let reloads = own[from..to]
                .iter()
                .map(|&p| trips_at[p])
                .fold(0, usize::saturating_add);
            let latch_stores = match param {
                true => usize::try_from(facts.trips).unwrap_or(usize::MAX),
                false => 0,
            };
            chosen.push(l);
            candidates.push(Candidate {
                weight: reloads.saturating_add(latch_stores),
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

/// The allocator: one scan in layout order, binding every value to leases of
/// `pool` and slots of `frame`.
pub(in crate::emit) struct LinearScan<'m, B: IsaBackend> {
    pub(in crate::emit) pool: &'m Pool<B>,
    pub(in crate::emit) frame: &'m mut Frame,
    /// Where the frame's size goes when the scan ends: what the `Enter` and
    /// `Ret` it bound read.
    pub(in crate::emit) frame_bytes: &'m FrameBytes,
}

impl<'m, B: IsaBackend> Pass for LinearScan<'m, B> {
    type Input = Function<B>;
    type Output = Allocated<'m, B>;

    fn run(self, function: Function<B>) -> Result<Allocated<'m, B>, CompileError> {
        self.allocate(function)
    }
}

impl<'m, B: IsaBackend> RegisterAllocator<'m, B> for LinearScan<'m, B> {
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
    /// passes one parameter to another), a value dropped at a join that is in no
    /// slot, a block no scanned branch reaches, a loop's exit reached by anything
    /// but its latch, or an entry block whose first instruction does not make the
    /// frame.
    fn allocate(self, function: Function<B>) -> Result<Allocated<'m, B>, CompileError> {
        let Self {
            pool,
            frame,
            frame_bytes,
        } = self;
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
        let Liveness {
            mut lives, loops, ..
        } = live;

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
        let mut held = Held(Vec::new());
        let abi = [leases.entry.ctx, leases.entry.out, leases.entry.pitch];
        for (value, lease) in entry.iter().zip(abi) {
            held.insert(value.id, Lent::General(lease));
        }
        // One nothing reads is dead on arrival, which the scan would find only
        // after the first instruction.
        for value in entry {
            if lives[value.id as usize].reads.is_empty() {
                let lent = held.remove(value.id).expect("an entry value is held");
                free[lent.file() as usize].push(lent);
            }
        }
        // The ABI registers are stored, if they ever are, right after the entry
        // block's first instruction, which makes the frame.
        let mut defined: Vec<Option<(usize, usize)>> = alloc::vec![None; lives.len()];
        let free_general = general_mask(&free[FileId::General as usize]);
        for value in entry {
            defined[value.id as usize] = Some((0, 0));
            lives[value.id as usize].free_general = free_general;
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
            pool,
            frame: &mut *frame,
            frame_bytes,
            reads,
            lives,
            blocks_at,
            loops,
            plan,
            loop_at,
            active: Vec::new(),
            want: BTreeMap::new(),
            edges: BTreeMap::new(),
            free,
            held,
            pinned: Vec::new(),
            next: function.classes.len() as u64,
            defined,
            free_general_at_entry: alloc::vec![0; function.blocks.len()],
            deferred: Vec::new(),
            records: Vec::new(),
            late: BTreeMap::new(),
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
            scan.held.0.is_empty() && scan.active.is_empty() && scan.edges.is_empty(),
            "values are held after the last instruction: {:?}, loops are open (by index): {:?}, and branches to {:?} were never arrived at",
            scan.held.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            scan.active.iter().map(|a| a.index).collect::<Vec<_>>(),
            scan.edges.keys()
        );
        let Scan {
            records, mut late, ..
        } = scan;

        let slots = frame.minted();
        frame_bytes.set(frame.bytes());
        let mut blocks = Vec::with_capacity(records.len());
        for (b, (block, (label, scope))) in records.into_iter().zip(shapes).enumerate() {
            let mut insts = late.remove(&StorePoint::Entry(b)).unwrap_or_default();
            for (k, emitted) in block.into_iter().enumerate() {
                insts.push(emitted);
                insts.extend(late.remove(&StorePoint::After(b, k)).unwrap_or_default());
            }
            blocks.push(Block {
                label,
                params: Vec::new(),
                insts,
                scope,
            });
        }
        assert!(
            late.is_empty(),
            "stores were placed at points no block has: {:?}",
            late.keys()
        );

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
}

/// Where an evicted value's store goes in the blocks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StorePoint {
    /// Right after the record at this block and index.
    After(usize, usize),
    /// At the start of this block.
    Entry(usize),
}

/// A loop the scan is inside: where its carried values live.
struct Active {
    index: usize,
    /// Each carried parameter: its lease, taken at the loop's entry, is its
    /// home to the latch.
    homes: Vec<u64>,
    /// Each carried value defined before the loop, and its register.
    carried: Vec<(u64, u8)>,
}

/// The register each value in one holds, by file and number, in id order.
type Residents = Vec<(u64, (FileId, u8))>;

/// The values in a register and their leases, in id order: a file's registers
/// are few, so a search of them is short, and nothing is allocated per value.
struct Held<'m, B: IsaBackend>(Vec<(u64, Lent<'m, B>)>);

/// The registers of `leases`, by number, as a mask.
fn general_mask<B: IsaBackend>(leases: &[Lent<'_, B>]) -> u64 {
    leases
        .iter()
        .fold(0, |mask, lent| mask | 1 << lent.number())
}

impl<'m, B: IsaBackend> Held<'m, B> {
    fn at(&self, id: u64) -> Result<usize, usize> {
        self.0.binary_search_by_key(&id, |&(held, _)| held)
    }

    fn get(&self, id: u64) -> Option<&Lent<'m, B>> {
        Some(&self.0[self.at(id).ok()?].1)
    }

    fn number(&self, id: u64) -> u8 {
        self.get(id).expect("a value is held").number()
    }

    /// A lease is a register, so a value already held cannot take another:
    /// the one it had would be dropped, and its register lost.
    fn insert(&mut self, id: u64, lent: Lent<'m, B>) {
        let Err(at) = self.at(id) else {
            panic!("value {id} is already in a register");
        };
        self.0.insert(at, (id, lent));
    }

    fn remove(&mut self, id: u64) -> Option<Lent<'m, B>> {
        Some(self.0.remove(self.at(id).ok()?).1)
    }

    fn iter(&self) -> impl Iterator<Item = (u64, &Lent<'m, B>)> {
        self.0.iter().map(|(id, lent)| (*id, lent))
    }
}

/// What a forward branch leaves for the block it reaches.
struct Edge {
    /// The position of the branch.
    from: usize,
    residents: Residents,
}

/// One register-to-register move of a branch's parallel move: the argument
/// into the parameter's lease.
struct Move {
    arg: ValueName,
    param: ValueName,
}

/// The scan's state: which registers are free, which values are in one, and
/// the instructions placed so far.
struct Scan<'f, 'm, B: IsaBackend> {
    pool: &'m Pool<B>,
    frame: &'f mut Frame,
    frame_bytes: &'m FrameBytes,
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
    /// The leases nothing holds, by file.
    free: [Vec<Lent<'m, B>>; 4],
    /// The values in a register, by id.
    held: Held<'m, B>,
    /// The values the instruction being placed reads or defines: the ones
    /// eviction leaves alone.
    pinned: Vec<u64>,
    /// The next value id the allocator mints.
    next: u64,
    /// Where each value is defined: a block, and an index into its records.
    defined: Vec<Option<(usize, usize)>>,
    /// By block, the general registers lent to nothing when the scan arrived:
    /// the record a store placed at the block's entry draws on.
    free_general_at_entry: Vec<u64>,
    /// The definitions the backend can recompute, held back from where they
    /// were selected ([`Self::defer`]) and placed where a read needs them; a
    /// [`Life::remat`] names one by index. One is out while it is being placed.
    deferred: Vec<Option<B::Inst<Selected>>>,
    /// The instructions placed so far, bound, by block.
    records: Vec<Vec<Emitted<'m, B>>>,
    /// The stores of evicted values, bound, by where each goes.
    late: BTreeMap<StorePoint, Vec<Emitted<'m, B>>>,
    /// Where the scan is, for what it says when a function cannot be allocated.
    at: Label,
    block: usize,
    position: usize,
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
        let reads = || {
            let operands = pushed.operands.iter();
            let reads = operands.filter_map(|operand| match operand {
                Operand::Reg { value, access } if access.reads() => Some(*value),
                Operand::Reg { .. } | Operand::Target(_) | Operand::Frame(_) => None,
            });
            reads.chain(args.iter().copied())
        };
        self.pinned.clear();
        for read in reads() {
            self.pin(read.id);
        }
        for read in reads() {
            if self.held.get(read.id).is_none() {
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

    /// Leave `value` alone while the instruction being placed is, and what
    /// placing its definition again would read: [`Liveness`] counts that read
    /// here, so it is held to this position even when it is not needed.
    fn pin(&mut self, value: u64) {
        self.pinned.push(value);
        if let Recompute::Reads(base) = self.lives[value as usize].recompute {
            self.pin(base.id);
        }
    }

    /// Hold `pushed`, a definition the backend can recompute, until a read
    /// needs its value in a register ([`Self::reload_into`]). One nothing reads
    /// is never placed.
    ///
    /// # Panics
    /// When the instruction does more than define one value first and read
    /// others, or defines flags, which a placement elsewhere could clobber.
    fn defer(&mut self, pushed: Pushed<B::Inst<Selected>>) {
        let [
            Operand::Reg {
                value,
                access: Access::Write,
            },
            reads @ ..,
        ] = &pushed.operands[..]
        else {
            panic!("a rematerializable instruction defines one value, first, and reads others")
        };
        assert!(
            reads.iter().all(|operand| matches!(
                operand,
                Operand::Reg {
                    access: Access::Read,
                    ..
                }
            )),
            "a rematerializable instruction defines one value and otherwise only reads"
        );
        let value = *value;
        assert_ne!(
            value.class,
            ClassId::Flags,
            "{value:?} is rematerializable, and a rematerialization would clobber the flags"
        );
        self.lives[value.id as usize].remat = Some(self.deferred.len());
        self.deferred.push(Some(pushed.inst));
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
            if !self.loops[l].takes(value) && self.held.get(value).is_none() {
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
        self.free_general_at_entry[self.block] = general_mask(&self.free[FileId::General as usize]);
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
            let lent = self.held.get(value);
            let lent = lent.expect("a value carried across a loop is in a register at its head");
            active.carried.push((value, lent.number()));
        }
        for (&arg, param) in facts.latch_args.iter().zip(&facts.params) {
            if active.homes.contains(&param.id) && !facts.takes(arg.id) {
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
            .map(|(id, lent)| {
                assert!(
                    (id as usize) < self.lives.len() && lent.file() != FileId::Flags,
                    "value {id} is held at a branch to {label:?}, and no flags value, and no value the allocator made, is live at a label"
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
    /// every path, or the block is a loop's exit and anything but the latch
    /// reaches it.
    fn arrive(&mut self, label: Label) {
        if self.block == 0 {
            return;
        }
        let edges = self
            .edges
            .remove(&label)
            .unwrap_or_else(|| panic!("{label:?} is reached by no branch the scan has seen"));
        // A loop-defined value is stored once at the exit ([`Self::store_point`]),
        // which is sound while the latch is the only way there.
        for l in self.loops.iter().filter(|l| l.exit == self.block) {
            assert!(
                matches!(edges[..], [Edge { from, .. }] if from == l.latch),
                "{label:?} is the exit of the loop latched at {}, and is reached from {:?}: a loop's exit has one predecessor, its latch",
                l.latch,
                edges.iter().map(|e| e.from).collect::<Vec<_>>()
            );
        }
        // The block after the branch, entered by it alone, is in its state.
        if let [Edge { from, .. }] = edges[..]
            && from + 1 == self.position
        {
            return;
        }
        let start = self.position;
        let mut ids: Vec<u64> = edges
            .iter()
            .flat_map(|e| e.residents.iter().map(|&(id, _)| id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
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
                let at = edge.residents.binary_search_by_key(&id, |&(held, _)| held);
                match (at.ok().map(|i| edge.residents[i].1), register) {
                    (Some(at), None) => register = Some(at),
                    (Some(at), Some(first)) => kept &= at == first,
                    // On a path the definition comes after, the value does not
                    // exist, and takes its register from the paths where it
                    // does. On any other it was evicted, or placed again there.
                    (None, _) => {
                        kept &= life.def > edge.from && life.remat.is_none();
                    }
                }
            }
            let register = register.expect("a candidate is in a register on some path");
            match kept {
                true => {
                    agreed.push((id, register));
                }
                false => {
                    assert!(
                        !carried && (life.stored || life.remat.is_some()),
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
        let held = self.held.0.drain(..).map(|(_, lent)| lent);
        let free = self.free.iter_mut().flat_map(|bank| bank.drain(..));
        let mut leases: Vec<Lent<'m, B>> = held.chain(free).collect();
        for &(id, (file, number)) in residents {
            let at = leases
                .iter()
                .position(|lent| lent.file() == file && lent.number() == number);
            let at = at.unwrap_or_else(|| panic!("value {id} is in a register no lease has"));
            let lent = leases.swap_remove(at);
            self.held.insert(id, lent);
        }
        for lent in leases {
            self.free[lent.file() as usize].push(lent);
        }
    }

    /// Pass `target`'s arguments to its parameters. A parameter the loop
    /// carries is in a register: the entering branch gives it one
    /// ([`Self::home`]), and a backward branch moves its argument there. Any
    /// other is stored to its slot. Every argument is in a register before the
    /// first one is passed, so the moves are a parallel move however they
    /// overlap.
    fn pass(&mut self, target: &Target, args: &[ValueName]) -> Result<(), CompileError> {
        let params = self.blocks_at[&target.label].1.clone();
        let back = self.blocks_at[&target.label].0 <= self.block;
        let carrying = self.loop_at.get(&target.label).copied();
        let mut moves = Vec::new();
        for (&arg, param) in target.args.iter().zip(&params) {
            if arg.id == param.id || !self.lives[param.id as usize].read {
                continue;
            }
            let l = carrying.expect("only a loop's head takes parameters");
            let carried = self.plan[l].contains(&param.id);
            match (carried, back) {
                (true, false) => {
                    let lent = self.home(arg, args)?;
                    let active = self.active.last_mut().expect("a loop is being entered");
                    active.homes.push(param.id);
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
            true => Ok(self.held.remove(arg.id).expect("an argument is held")),
            false => self.copy_of(arg),
        }
    }

    /// The backward branch of loop `l`: move each carried parameter's argument
    /// into the parameter's register, and leave the loop.
    ///
    /// # Panics
    /// When a carried value is not in the register it had at the head, a
    /// parameter is read past the latch, or the moves form a register cycle.
    fn latch(&mut self, l: usize, moves: &[(ValueName, ValueName)]) {
        let active = self.active.pop().expect("a latch is inside its loop");
        assert_eq!(active.index, l, "{:?} closes a loop it is not in", self.at);
        for &(value, number) in &active.carried {
            let at = self.held.get(value).map(|lent| lent.number());
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
            assert!(
                active.homes.contains(&param.id),
                "a carried parameter has a home"
            );
            // The parameter's register is its own to the latch, so the move is
            // into its lease. A parameter holding none gave its register to the
            // argument when that was defined ([`Self::write_plain`]), and its
            // move is nothing.
            if self.held.get(param.id).is_some() {
                pending.push(Move { arg, param });
            }
        }
        self.sequence(pending);
        self.want.retain(|_, param| !active.homes.contains(param));
        let carried = active.carried.iter().map(|&(value, _)| value);
        self.pinned
            .extend(carried.chain(active.homes.iter().copied()));
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
            // A move may go when no other reads the register it writes: every
            // value is in a lease of its own, so that is the parameter itself
            // being passed on.
            let free = pending
                .iter()
                .position(|m| !pending.iter().any(|other| other.arg.id == m.param.id));
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

    /// Copy `m.arg` into the lease of `m.param`, which nothing reads again.
    fn move_register(&mut self, m: &Move) {
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let copy = B::copy(&mut spiller, m.arg);
        for pushed in insts {
            let inst = self.bind(&pushed.inst, &[(copy.id, m.param.id)]);
            self.records[self.block].push(Emitted {
                inst,
                origin: Origin::Copy,
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
        let life = &self.lives[value.id as usize];
        if let Some(at) = life.remat {
            let recompute = life.recompute;
            let mut operands = alloc::vec![Operand::Reg {
                value,
                access: Access::Write,
            }];
            if let Recompute::Reads(base) = recompute {
                // The definition reads, so what it reads is placed first, and
                // kept while it is.
                self.pin(base.id);
                if self.held.get(base.id).is_none() {
                    self.reload_into(base)?;
                }
                operands.push(Operand::Reg {
                    value: base,
                    access: Access::Read,
                });
            }
            let inst = self.deferred[at]
                .take()
                .expect("a definition is placed again one read at a time");
            let placed = self.place(&inst, &operands, Origin::Remat);
            self.deferred[at] = Some(inst);
            return placed;
        }
        let life = &self.lives[value.id as usize];
        let slot = life
            .slot
            .filter(|_| life.stored)
            .unwrap_or_else(|| panic!("{value:?} is read while in no register and in no slot"));
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let slot = self.frame.slot(slot);
        let fresh = B::reload(&mut spiller, value.class, slot);
        for pushed in insts {
            self.emit(pushed, Origin::Reload)?;
        }
        let lent = self.held.remove(fresh.id).expect("a reload is held");
        self.held.insert(value.id, lent);
        Ok(())
    }

    /// The instructions that store `value`, which is in a register, to `slot`.
    fn spill_of(&mut self, value: ValueName, slot: SlotName) -> Vec<Pushed<B::Inst<Selected>>> {
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let slot = self.frame.slot(slot);
        B::spill(&mut spiller, value, slot);
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
        let (block, index) =
            self.defined[value as usize].expect("a value in a register was defined");
        StorePoint::After(block, index)
    }

    /// Store `value`, which is in its defining register, right after its
    /// definition, unless that is done, or it can be defined again instead.
    fn store_at_definition(&mut self, value: u64) -> Result<(), CompileError> {
        let life = &self.lives[value as usize];
        if life.stored || life.remat.is_some() {
            return Ok(());
        }
        let name = ValueName {
            id: value,
            class: life.class,
        };
        let slot = self.slot(name)?;
        self.lives[value as usize].stored = true;
        let at = self.store_point(value);
        let insts = self.spill_of(name, slot);
        // The point the store follows is behind the scan, where no lease can be
        // taken. The store reads `value` from the register it has held since
        // its definition, and the address of a slot too deep for it to encode
        // from a general register the allocator's record of that point says
        // was lent to nothing: the pool's one past-tense lease, which the
        // address value holds for the binding alone.
        let mut address = None;
        for operand in insts.iter().flat_map(|pushed| &pushed.operands) {
            match operand {
                Operand::Reg { value: read, .. } if read.id == value => {}
                Operand::Reg {
                    value: pointer,
                    access,
                } if pointer.id as usize >= self.lives.len() => {
                    assert_eq!(
                        pointer.class,
                        ClassId::Pointer,
                        "{pointer:?} is not a slot's address"
                    );
                    if !access.reads() {
                        let number = self.general_free_at(value, at)?;
                        self.held.insert(pointer.id, self.pool.past(number));
                        assert!(
                            address.replace(pointer.id).is_none(),
                            "a slot's address is one value"
                        );
                    }
                }
                Operand::Reg { value: read, .. } => panic!(
                    "the store of {value} after its definition reads {read:?}, a register at a point the scan has left"
                ),
                Operand::Target(_) | Operand::Frame(_) => {}
            }
        }
        for pushed in &insts {
            let inst = self.bind(&pushed.inst, &[]);
            self.late.entry(at).or_default().push(Emitted {
                inst,
                origin: Origin::Spill,
            });
        }
        if let Some(address) = address {
            assert!(
                self.held.remove(address).is_some(),
                "the address value held the past-tense lease for the binding"
            );
        }
        Ok(())
    }

    /// A general register lent to nothing at `at`, where `value`'s store goes,
    /// from the allocator's record of that point ([`Life::free_general`],
    /// [`Self::free_general_at_entry`]): the lowest-numbered, as
    /// [`Self::acquire`] would choose now.
    ///
    /// # Errors
    /// [`CompileError::BudgetExceeded`] when every general register was lent
    /// there.
    fn general_free_at(&self, value: u64, at: StorePoint) -> Result<u8, CompileError> {
        let free = match at {
            StorePoint::After(..) => self.lives[value as usize].free_general,
            StorePoint::Entry(block) => self.free_general_at_entry[block],
        };
        B::FILE
            .members(FileId::General)
            .iter()
            .copied()
            .find(|&number| free & 1 << number != 0)
            .ok_or(CompileError::BudgetExceeded(
                "a vector slot past the reach of one store, defined where every general register was lent",
            ))
    }

    /// Drop every value from its register but the carried ones, storing the
    /// ones not yet stored: what a loop head expects of the state it is
    /// entered in.
    fn flush(&mut self) -> Result<(), CompileError> {
        let dropped: Vec<u64> = self
            .held
            .iter()
            .map(|(id, _)| id)
            .filter(|&id| !self.kept(id))
            .collect();
        for id in dropped {
            self.store_at_definition(id)?;
            let lent = self.held.remove(id).expect("the key was just read");
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
        let lent = bank.swap_remove(lowest);
        Ok(lent)
    }

    /// Free a register of `file`: the one whose value costs the least to give
    /// up, which is stored first if the slot does not hold it.
    ///
    /// # Panics
    /// When every register of the file holds a value the instruction needs, or
    /// the file is the flags, which cannot be stored.
    fn evict(&mut self, file: FileId) -> Result<(), CompileError> {
        let rank = |id: u64| {
            let life = &self.lives[id as usize];
            let store = match (life.remat.is_some(), life.stored) {
                (true, _) => Store::Never,
                (false, true) => Store::NotNeeded,
                (false, false) => Store::Needed,
            };
            let distance = self.next_read(id).map(|next| next - self.position);
            (EvictionRank::new(store, distance), id)
        };
        let victim = self
            .held
            .iter()
            .filter(|&(id, lent)| {
                lent.file() == file
                    && file != FileId::Flags
                    && (id as usize) < self.lives.len()
                    && !self.pinned.contains(&id)
                    && !self.kept(id)
            })
            .map(|(id, _)| rank(id))
            .min();
        let Some((_, victim)) = victim else {
            panic!(
                "instruction {} of {:?} holds more {file:?} registers than the file has",
                self.position, self.at
            )
        };
        self.store_at_definition(victim)?;
        let lent = self.held.remove(victim).expect("the victim is held");
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
        self.place(&pushed.inst, &pushed.operands, origin)
    }

    /// Place instruction `inst`, whose operands are `operands`, as
    /// [`Self::emit`] does, and bind it: its reads in the state it reads them
    /// in, its writes in the state it leaves.
    fn place(
        &mut self,
        inst: &B::Inst<Selected>,
        operands: &[Operand],
        origin: Origin,
    ) -> Result<(), CompileError> {
        // Values bound through another's lease: a read whose register a write
        // of this instruction takes.
        let mut via: Vec<(u64, u64)> = Vec::new();
        for operand in operands {
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
                false => {
                    via.push((tied.id, value.id));
                    self.held.remove(tied.id).expect("a tied read is held")
                }
                true => self.copy_of(tied)?,
            };
            self.pinned.push(value.id);
            self.held.insert(value.id, lent);
        }
        for operand in operands {
            if let Operand::Reg {
                value,
                access: Access::Early,
            } = operand
            {
                self.write(*value)?;
            }
        }
        // The reads nothing reads again (a value the allocator made is read
        // once, by the instruction after the one that defines it) keep their
        // leases until the instruction is bound: a plain write may take one
        // ([`Self::write_plain`]), and the rest are returned after.
        let mut dying: Vec<u64> = Vec::new();
        for operand in operands {
            let Operand::Reg {
                value,
                access: Access::Read,
            } = operand
            else {
                continue;
            };
            let minted = value.id as usize >= self.lives.len();
            let dies = (origin == Origin::Selected || minted) && !self.live_after(value.id);
            if dies && self.held.get(value.id).is_some() && !dying.contains(&value.id) {
                dying.push(value.id);
            }
        }
        for operand in operands {
            if let Operand::Reg {
                value,
                access: Access::Write,
            } = operand
            {
                self.write_plain(*value, &mut dying, &mut via)?;
            }
        }
        let mut binder = self.binder(&via);
        let bound = B::walk(inst, &mut binder);
        assert!(
            (self.block, self.records[self.block].len()) != (0, 0) || binder.framed,
            "the entry block's first instruction does not make the frame, so the ABI registers have nowhere to be stored"
        );
        for id in dying {
            let lent = self.held.remove(id);
            let lent = lent.expect("a dying read is held until its instruction is bound");
            self.free[lent.file() as usize].push(lent);
        }
        if origin == Origin::Selected {
            let at = (self.block, self.records[self.block].len());
            let free_general = general_mask(&self.free[FileId::General as usize]);
            for operand in operands {
                let Operand::Reg { value, access } = operand else {
                    continue;
                };
                if !access.reads() {
                    self.defined[value.id as usize] = Some(at);
                    self.lives[value.id as usize].free_general = free_general;
                }
            }
        }
        self.records[self.block].push(Emitted {
            inst: bound,
            origin,
        });
        Ok(())
    }

    /// `inst`, bound: each operand's token comes from the lease its value
    /// holds, or, for a value in `via`, from the lease of the value named for
    /// it ([`Binder`]).
    fn bind(&self, inst: &B::Inst<Selected>, via: &[(u64, u64)]) -> B::Inst<Bound<'m, B>> {
        B::walk(inst, &mut self.binder(via))
    }

    fn binder<'a>(&'a self, via: &'a [(u64, u64)]) -> Binder<'a, 'm, B> {
        Binder {
            held: &self.held,
            via,
            frame: &*self.frame,
            frame_bytes: self.frame_bytes,
            framed: false,
        }
    }

    /// Define `value` in a free lease, or one an eviction frees.
    fn write(&mut self, value: ValueName) -> Result<(), CompileError> {
        let lent = self.acquire(value.class)?;
        self.pinned.push(value.id);
        self.held.insert(value.id, lent);
        Ok(())
    }

    /// Define `value` as [`Self::write`] does, with two exceptions. A value a
    /// backward branch passes to a carried parameter that is read no more takes
    /// the parameter's register, so the branch has nothing to move, and is kept
    /// there to the latch as the parameter was. Otherwise the lease is the
    /// lowest-numbered of the free list's and the `dying` reads' of its file,
    /// which is the one [`Self::acquire`] finds once the dying reads are
    /// returned; a dying read's goes to `via`, since the read is bound through
    /// it.
    fn write_plain(
        &mut self,
        value: ValueName,
        dying: &mut Vec<u64>,
        via: &mut Vec<(u64, u64)>,
    ) -> Result<(), CompileError> {
        let home = self
            .want
            .get(&value.id)
            .copied()
            .filter(|&param| !self.read_again(param))
            .and_then(|param| Some((param, self.held.remove(param)?)));
        if let Some((param, lent)) = home {
            self.lives[value.id as usize].carried_until = self.lives[param as usize].carried_until;
            // The parameter may be a read of this instruction, `i + 1` into
            // `i`'s register: the read is bound through the value that took it.
            via.push((param, value.id));
            self.pinned.push(value.id);
            self.held.insert(value.id, lent);
            return Ok(());
        }
        let file = value.class.file();
        let dead = dying
            .iter()
            .enumerate()
            .filter(|&(_, &id)| self.held.get(id).is_some_and(|lent| lent.file() == file))
            .min_by_key(|&(_, &id)| self.held.number(id))
            .map(|(at, &id)| (at, id));
        let free = self.free[file as usize].iter().map(Lent::number).min();
        let lent = match dead {
            Some((at, id)) if free.is_none_or(|free| self.held.number(id) < free) => {
                dying.swap_remove(at);
                via.push((id, value.id));
                self.held.remove(id).expect("a dying read is held")
            }
            _ => self.acquire(value.class)?,
        };
        self.pinned.push(value.id);
        self.held.insert(value.id, lent);
        Ok(())
    }

    /// A copy of `value` in a fresh register: the lease it holds.
    fn copy_of(&mut self, value: ValueName) -> Result<Lent<'m, B>, CompileError> {
        let mut insts = Vec::new();
        let mut spiller = Spiller::<B>::new(&mut self.next, &mut insts);
        let copy = B::copy(&mut spiller, value);
        for pushed in insts {
            self.emit(pushed, Origin::Copy)?;
        }
        Ok(self.held.remove(copy.id).expect("a copy is held"))
    }

    /// Return the leases of the values that nothing reads again. Only a value
    /// the instruction read or defined can have just become one: the pinned.
    fn release_dead(&mut self) {
        for &id in &self.pinned {
            if self.live_after(id) {
                continue;
            }
            if let Some(lent) = self.held.remove(id) {
                self.free[lent.file() as usize].push(lent);
            }
        }
        debug_assert!(
            self.held.iter().all(|(id, _)| self.live_after(id)),
            "a value nothing reads again is held past the instruction that last touched it"
        );
    }
}

/// How the scan binds an instruction as it places it: each operand becomes a
/// token borrowed from the lease its value holds, the information the scan has
/// in hand at that point.
struct Binder<'a, 'm, B: IsaBackend> {
    held: &'a Held<'m, B>,
    /// Values bound through another value's lease: a read whose register a
    /// write of the same instruction took, and a latch move's copy, which
    /// writes the parameter's.
    via: &'a [(u64, u64)],
    frame: &'a Frame,
    frame_bytes: &'m FrameBytes,
    /// Whether the instruction asked for the frame's size.
    framed: bool,
}

impl<'a, 'm, B: IsaBackend> Binder<'a, 'm, B> {
    /// The lease value `id` is bound through.
    fn lent(&self, id: u64) -> &'a Lent<'m, B> {
        let through = || {
            self.via
                .iter()
                .find(|&&(value, _)| value == id)
                .map(|&(_, w)| w)
        };
        self.held
            .get(id)
            .or_else(|| self.held.get(through()?))
            .unwrap_or_else(|| panic!("value {id} is in no register"))
    }
}

impl<'m, B: IsaBackend> Rebind<Bound<'m, B>> for Binder<'_, 'm, B> {
    fn read<C: Class>(&mut self, v: Value<C>) -> In<'m, B, C> {
        C::File::lease(self.lent(v.id)).read()
    }
    fn write<C: Class>(&mut self, d: &build::Def<C>) -> Out<'m, B, C> {
        C::File::lease(self.lent(d.value().id)).write()
    }
    fn early<C: Class>(&mut self, d: &build::Early<C>) -> Out<'m, B, C> {
        C::File::lease(self.lent(d.value().id)).write()
    }
    fn tie<C: Class>(&mut self, t: &build::Tie<C>) -> InOut<'m, B, C> {
        C::File::lease(self.lent(t.write().id)).tie()
    }
    fn slot(&mut self, s: SlotName) -> FrameSlot {
        self.frame.slot(s)
    }
    fn target(&mut self, t: &Target) -> Label {
        t.label
    }
    fn frame_size(&mut self) -> &'m FrameBytes {
        self.framed = true;
        self.frame_bytes
    }
}
