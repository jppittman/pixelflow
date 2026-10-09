//! The machine's resources: registers and frame slots, as tokens.
//!
//! A leaf module: a [`Reg`] is minted only by [`Pool::mint`], a [`Lease`] only
//! by [`Leases::new`], a [`FrameSlot`] only by [`Frame`], and an [`In`],
//! [`Out`] or [`InOut`] only from a lease, so a register chosen anywhere but
//! the allocator does not compile. Values and labels are names and copy;
//! these are resources and move.

use crate::emit::{
    Class, ClassId, File, FileId, FlagsFile, GeneralFile, IsaBackend, OpmaskFile, VectorFile,
};
use crate::error::CompileError;
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

/// The largest frame a kernel may lay out. [`Frame::lease`] refuses a vector
/// slot past it, and the nest's layout refuses the whole frame (spills, fold
/// roots and parks) past it, so every slot offset is below it by construction.
pub(in crate::emit) const MAX_FRAME: u32 = 2 * 1024 * 1024;

/// Why a frame was refused for passing [`MAX_FRAME`].
pub(in crate::emit) const FRAME_OVERFLOW: &str = "spill frame overflow: exceeds 2MB stack limit";

/// A physical register of file `F` on backend `B`. A resource, not a name.
///
/// Branded by backend, so an AVX-512 token cannot reach an AVX2 encoder, whose
/// VEX prefix would drop bit 4 and encode `zmm20` as `ymm12`.
pub(in crate::emit) struct Reg<B: IsaBackend, F: File> {
    number: u8,
    _brand: PhantomData<(fn() -> B, F)>,
}

impl<B: IsaBackend, F: File> Reg<B, F> {
    /// The number the encoding's register fields take (`ModRM`, `VEX.vvvv`,
    /// `Rd`/`Rn`/`Rm`): a width the ISA dictates.
    pub(in crate::emit) fn number(&self) -> u8 {
        self.number
    }
}

/// Every member of `B::FILE`, minted. One per compile.
pub(in crate::emit) struct Pool<B: IsaBackend> {
    vector: Box<[Reg<B, VectorFile>]>,
    general: Box<[Reg<B, GeneralFile>]>,
    opmask: Box<[Reg<B, OpmaskFile>]>,
    flags: Box<[Reg<B, FlagsFile>]>,
}

impl<B: IsaBackend> Pool<B> {
    /// The only constructor of a [`Reg`].
    pub(in crate::emit) fn mint() -> Self {
        fn file<B: IsaBackend, F: File>() -> Box<[Reg<B, F>]> {
            B::FILE
                .members(F::ID)
                .iter()
                .map(|&number| Reg {
                    number,
                    _brand: PhantomData,
                })
                .collect()
        }
        Self {
            vector: file(),
            general: file(),
            opmask: file(),
            flags: file(),
        }
    }
}

/// The allocator's exclusive right to one register for one allocation.
///
/// Not `Copy`, not `Clone`. [`Leases::new`] makes exactly one per member of a
/// pool, and only the allocator can call it. During allocation a free list
/// owns a lease, binding a value moves the lease into the value's state, and
/// eviction and death move it back. So two live values in one register is
/// unrepresentable in the scan's state.
pub(in crate::emit) struct Lease<'m, B: IsaBackend, F: File>(&'m Reg<B, F>);

impl<'m, B: IsaBackend, F: File> Lease<'m, B, F> {
    fn number(&self) -> u8 {
        self.0.number
    }

    /// The register as an instruction's read.
    pub(super) fn read<C: Class<File = F>>(&self) -> In<'m, B, C> {
        In(self.0, PhantomData)
    }

    /// The register as an instruction's write or early write.
    pub(super) fn write<C: Class<File = F>>(&self) -> Out<'m, B, C> {
        Out(self.0, PhantomData)
    }

    /// The register as a read the instruction overwrites in place.
    pub(super) fn tie<C: Class<File = F>>(&self) -> InOut<'m, B, C> {
        InOut(self.0, PhantomData)
    }
}

/// A lease whose file is a run-time fact: what the scan, which meets values of
/// every class, holds. [`File::lease`] gets the typed lease back where an
/// instruction field has a class.
pub(in crate::emit) enum Lent<'m, B: IsaBackend> {
    Vector(Lease<'m, B, VectorFile>),
    General(Lease<'m, B, GeneralFile>),
    Opmask(Lease<'m, B, OpmaskFile>),
    Flags(Lease<'m, B, FlagsFile>),
}

impl<B: IsaBackend> Lent<'_, B> {
    pub(super) fn file(&self) -> FileId {
        match self {
            Lent::Vector(_) => FileId::Vector,
            Lent::General(_) => FileId::General,
            Lent::Opmask(_) => FileId::Opmask,
            Lent::Flags(_) => FileId::Flags,
        }
    }

    /// The register's hardware number, the scan's tie-break: lowest first.
    pub(super) fn number(&self) -> u8 {
        match self {
            Lent::Vector(lease) => lease.number(),
            Lent::General(lease) => lease.number(),
            Lent::Opmask(lease) => lease.number(),
            Lent::Flags(lease) => lease.number(),
        }
    }
}

impl<B: IsaBackend> core::fmt::Debug for Lent<'_, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} register {}", self.file(), self.number())
    }
}

/// The registers the ABI hands the entry block, already bound: initial
/// ownership, not a reservation, so once a parameter is dead or spilled its
/// register is free.
pub(super) struct EntryLeases<'m, B: IsaBackend> {
    pub(super) ctx: Lease<'m, B, GeneralFile>,
    pub(super) out: Lease<'m, B, GeneralFile>,
    pub(super) pitch: Lease<'m, B, GeneralFile>,
}

/// One allocation's leases. The entry's three are handed out already bound,
/// so the allocator never looks a register up by number.
pub(super) struct Leases<'m, B: IsaBackend> {
    pub(super) vector: Vec<Lease<'m, B, VectorFile>>,
    pub(super) general: Vec<Lease<'m, B, GeneralFile>>,
    pub(super) opmask: Vec<Lease<'m, B, OpmaskFile>>,
    pub(super) flags: Vec<Lease<'m, B, FlagsFile>>,
    pub(super) entry: EntryLeases<'m, B>,
}

impl<'m, B: IsaBackend> Leases<'m, B> {
    /// `pub(super)`: `regalloc` only. Selection, the backends and the
    /// assembler cannot obtain a lease.
    pub(super) fn new(pool: &'m Pool<B>) -> Self {
        fn all<'m, B: IsaBackend, F: File>(regs: &'m [Reg<B, F>]) -> Vec<Lease<'m, B, F>> {
            regs.iter().map(Lease).collect()
        }
        fn take<'m, B: IsaBackend>(
            general: &mut Vec<Lease<'m, B, GeneralFile>>,
            number: u8,
        ) -> Lease<'m, B, GeneralFile> {
            let at = general
                .iter()
                .position(|lease| lease.number() == number)
                .expect("RegisterFile::new proves each entry register is a general member");
            general.remove(at)
        }
        let mut general = all(&pool.general);
        let abi = B::FILE.entry();
        let entry = EntryLeases {
            ctx: take(&mut general, abi.ctx),
            out: take(&mut general, abi.out),
            pitch: take(&mut general, abi.pitch),
        };
        Self {
            vector: all(&pool.vector),
            general,
            opmask: all(&pool.opmask),
            flags: all(&pool.flags),
            entry,
        }
    }
}

/// A register as one instruction's read, one write or one in-place update: a
/// shared borrow of the pool's token, copied out of the lease its value held
/// at that instruction. Three types, so that `walk` cannot put a read where a
/// write belongs. Each is made only from a [`Lease`], and none is `Copy`:
/// each is produced for exactly one operand.
pub(in crate::emit) struct In<'m, B: IsaBackend, C: Class>(&'m Reg<B, C::File>, PhantomData<C>);
/// See [`In`].
pub(in crate::emit) struct Out<'m, B: IsaBackend, C: Class>(&'m Reg<B, C::File>, PhantomData<C>);
/// See [`In`].
pub(in crate::emit) struct InOut<'m, B: IsaBackend, C: Class>(&'m Reg<B, C::File>, PhantomData<C>);

macro_rules! deref_to_reg {
    ($($operand:ident),*) => {$(
        impl<'m, B: IsaBackend, C: Class> Deref for $operand<'m, B, C> {
            type Target = Reg<B, C::File>;
            fn deref(&self) -> &Reg<B, C::File> {
                self.0
            }
        }
    )*};
}
deref_to_reg!(In, Out, InOut);

/// A frame slot's name: `Copy`, carried by selected-stage instructions the way
/// [`ValueName`](crate::emit::ValueName) is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) struct SlotName(u64);

/// A frame slot: at `offset` from the stack pointer. A resource like
/// [`Reg`]: not `Copy`, not `Clone`, minted only by [`Frame`], at an offset
/// fixed for its life.
pub(in crate::emit) struct FrameSlot {
    name: SlotName,
    offset: u64,
}

impl FrameSlot {
    /// Bytes from the stack pointer. The control plane is 64-bit: each encoder
    /// narrows this to its displacement field, and the backend's `spill` and
    /// `reload` turn an offset that does not fit into address arithmetic.
    pub(in crate::emit) fn offset(&self) -> u64 {
        self.offset
    }

    pub(in crate::emit) fn name(&self) -> SlotName {
        self.name
    }
}

/// The allocator's exclusive right to one slot. Affine, with no lifetime: the
/// offset is fixed at mint, so nothing borrows the frame while the scan grows
/// it.
pub(super) struct SlotLease {
    name: SlotName,
}

impl SlotLease {
    pub(super) fn name(&self) -> SlotName {
        self.name
    }
}

/// Bytes of a narrow slot: a `Pointer`, an `Integer` or an `Opmask`.
const NARROW_BYTES: u64 = 8;
/// The most narrow slots: an `ldr x, [sp, #imm12·8]` on aarch64 reaches
/// `imm12` 0 through 4095, which is 4096 slots, and one bound serves every
/// target.
const MAX_NARROW_SLOTS: u64 = 4096;
/// The stack pointer's alignment, on both ABIs.
const SP_ALIGN: u64 = 16;

const fn align_up(n: u64, to: u64) -> u64 {
    n.next_multiple_of(to)
}

/// One function's frame, and the stack pointer with it.
///
/// - **Layout.** The narrow region (8 bytes a slot) occupies `[0, 8·N)`, where
///   `N` is the peak number of simultaneously live narrow values, measured by
///   the allocator before the scan. The vector region starts at
///   `align_up(8·N, max(16, vector_bytes))` and grows as the scan leases
///   slots, lowest free first.
/// - **Size.** The vector region's high-water mark, or the narrow region
///   alone when no vector was spilled, aligned to the stack pointer's 16.
/// - **The stack pointer is the frame's.** Its only readers are frame-slot
///   operands and the backend's slot address, and its only writers are the
///   function's `Enter` and `Ret`. It is in no class.
pub(in crate::emit) struct Frame {
    vector_bytes: u64,
    /// The narrow region's slots first, then the vector region's, in mint
    /// order: a name is an index.
    slots: Vec<FrameSlot>,
    narrow: u64,
    vector_start: u64,
    free_narrow: BTreeSet<SlotName>,
    free_vector: BTreeSet<SlotName>,
}

impl Frame {
    pub(in crate::emit) fn empty(vector_bytes: u64) -> Self {
        Self {
            vector_bytes,
            slots: Vec::new(),
            narrow: 0,
            vector_start: 0,
            free_narrow: BTreeSet::new(),
            free_vector: BTreeSet::new(),
        }
    }

    /// Lay out the narrow region for `peak` simultaneously live narrow values.
    /// Once, before any lease.
    ///
    /// # Errors
    /// [`CompileError::BudgetExceeded`] past [`MAX_NARROW_SLOTS`].
    pub(super) fn reserve_narrow(&mut self, peak: u64) -> Result<(), CompileError> {
        assert!(
            self.slots.is_empty(),
            "the narrow region is laid out once, before any slot is leased"
        );
        if peak > MAX_NARROW_SLOTS {
            return Err(CompileError::BudgetExceeded(
                "more live pointers and integers than the narrow frame region addresses",
            ));
        }
        self.narrow = peak;
        self.vector_start = align_up(peak * NARROW_BYTES, SP_ALIGN.max(self.vector_bytes));
        for offset in (0..peak).map(|i| i * NARROW_BYTES) {
            self.mint(offset);
        }
        self.free_narrow = self.slots.iter().map(FrameSlot::name).collect();
        Ok(())
    }

    fn mint(&mut self, offset: u64) -> SlotName {
        let name = SlotName(self.slots.len() as u64);
        self.slots.push(FrameSlot { name, offset });
        name
    }

    /// The lowest free slot that holds a `class`, minting one past the vector
    /// region's high-water mark when none is free.
    ///
    /// # Errors
    /// [`CompileError::BudgetExceeded`] when the frame would pass `MAX_FRAME`.
    ///
    /// # Panics
    /// `Flags` cannot be stored, and a narrow slot beyond what
    /// [`reserve_narrow`](Self::reserve_narrow) laid out is an allocator bug:
    /// the peak is measured before the scan.
    pub(super) fn lease(&mut self, class: ClassId) -> Result<SlotLease, CompileError> {
        let name = match class {
            ClassId::Flags => panic!("the flags cannot be stored in a frame slot"),
            ClassId::Pointer | ClassId::Integer | ClassId::Opmask => self
                .free_narrow
                .pop_first()
                .expect("more narrow values live than the peak the frame was laid out for"),
            ClassId::Vector => match self.free_vector.pop_first() {
                Some(name) => name,
                None => {
                    let offset = self.vector_end();
                    if offset + self.vector_bytes > u64::from(MAX_FRAME) {
                        return Err(CompileError::BudgetExceeded(FRAME_OVERFLOW));
                    }
                    self.mint(offset)
                }
            },
        };
        Ok(SlotLease { name })
    }

    /// Where the next vector slot would start.
    fn vector_end(&self) -> u64 {
        let vectors = self.slots.len() as u64 - self.narrow;
        self.vector_start + vectors * self.vector_bytes
    }

    pub(super) fn release(&mut self, slot: SlotLease) {
        let free = if slot.name.0 < self.narrow {
            &mut self.free_narrow
        } else {
            &mut self.free_vector
        };
        free.insert(slot.name);
    }

    /// How many slots the frame has minted: the narrow region's, then the
    /// vector region's.
    pub(super) fn minted(&self) -> u64 {
        self.slots.len() as u64
    }

    pub(in crate::emit) fn slot(&self, name: SlotName) -> &FrameSlot {
        &self.slots[name.0 as usize]
    }

    /// The frame's size in bytes: what `Enter` subtracts from the stack
    /// pointer.
    pub(in crate::emit) fn bytes(&self) -> u64 {
        let vectors = self.slots.len() as u64 - self.narrow;
        if vectors == 0 {
            return align_up(self.narrow * NARROW_BYTES, SP_ALIGN);
        }
        self.vector_end()
    }
}
