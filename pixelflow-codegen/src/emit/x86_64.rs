//! x86-64 leaf encoders: what the AVX2 and AVX-512 tiers share below the
//! vector width.
//!
//! What is here is the *architecture's*: [`Gp`], the general-register
//! instructions the loop nest and the store's address arithmetic are made
//! of (branches, the pointer class's loads and stores and the frame among
//! them), the memory-operand tail (`Mem`, `Disp`) every vector encoder's
//! ModRM/SIB is built from, the vector operations both tiers name ([`Alu`] and
//! its kin, whose bytes are each tier's), and the constant pool with its
//! anchor. Nothing here names a vector width: the `ymm`/`zmm` encodings live
//! in `avx2.rs` and `avx512.rs`, each with its own `LegacyBackend` driver, and
//! the 128-bit tier that used to sit in this file is gone
//! (docs/plans/2026-09-22-the-isa-is-decided-at-startup.md §7).

use super::asm::Encoding;
use super::{
    AsmInsn, Assembly, Binding, EncodedInst, Flags, Gpr, Integer, Label, LabelRef, Loc, Physical,
    Placed, Pointer, PtrReg, Rebind, Reg, Selected, Stage, WritePlan, regalloc,
};
use crate::error::CompileError;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// =============================================================================
// Constants
// =============================================================================

/// The register that holds the constant pool's address for the whole kernel:
/// `r8`, SysV's fifth integer argument, which the collapse ABI does not pass,
/// and caller-saved, so nothing preserves it. The x86 counterpart of
/// aarch64's `X17`. A pointer register because that is what it holds; each
/// tier's register file (`avx2::driver::AVX2_FILE`, `avx512::driver::AVX512_FILE`)
/// keeps its GPR roles clear of it the way they stay clear of the three
/// arguments.
const POOL_BASE: PtrReg = PtrReg(8);

/// A kernel's constants, deduplicated, laid out after its `ret`.
///
/// One per emitted function, shared by every scope of the nest: a compile
/// emits every scope through one backend, so the offsets an outer scope baked
/// in stay valid as inner scopes append. Entry `k` is at `[POOL_BASE + 4k]`;
/// [`anchor`] materializes `POOL_BASE` once, after the frame, and
/// [`ConstPool::finish`] writes the entries after the return, binding the
/// label the anchor names.
///
/// Four bytes per constant, not a register's worth: every tier loads a
/// constant with `vbroadcastss` from a 32-bit source, so the splat is the
/// instruction's business and the pool holds the scalar. This replaced two
/// idioms that were each two instructions per read: a `jmp` over sixteen
/// bytes of inline data followed by a RIP-relative load on the 128-bit tier,
/// and a store to the red zone followed by a broadcast from it on the wide
/// ones — a store-forward on the critical path of every constant read.
#[derive(Default)]
pub(super) struct ConstPool {
    /// The entries, in pool order.
    entries: Vec<u32>,
    /// Each entry's position in `entries`, by its bits.
    index: BTreeMap<u32, u64>,
}

impl ConstPool {
    /// The memory operand of the constant with these bits: `[POOL_BASE +
    /// offset]`, entering it into the pool on first use.
    ///
    /// Always a `disp32`, never a `disp8`: EVEX scales a `disp8` by the
    /// operand's tuple size, VEX does not, and one form for both is worth
    /// three bytes per load.
    ///
    /// # Errors
    ///
    /// [`CompileError::BudgetExceeded`] when the entry lies past a `disp32`
    /// ([`block_element`]): a pool that cannot be addressed never grows.
    pub(super) fn operand(&mut self, bits: u32) -> Result<Mem<Physical, Imm32>, CompileError> {
        if let Some(&position) = self.index.get(&bits) {
            return block_element(POOL_BASE, position);
        }
        let position = self.entries.len() as u64;
        let element = block_element(POOL_BASE, position)?;
        self.entries.push(bits);
        self.index.insert(bits, position);
        Ok(element)
    }

    /// Append the pool after the return and bind `pool` where it lands.
    pub(super) fn finish(&self, asm: &mut Assembly, pool: Label) {
        asm.pool(
            pool,
            self.entries
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect(),
        );
    }
}

/// Every x86 tier's anchor: `POOL_BASE = &pool`, once, after the frame.
pub(super) fn anchor(asm: &mut Assembly, pool: Label) {
    asm.push(Gp::LeaRip {
        dst: POOL_BASE,
        to: pool,
    });
}

// =============================================================================
// The pointer class: an address between a general register and memory
// =============================================================================

/// Bytes per pointer in the context array.
pub(super) const PTR_BYTES: i32 = 8;

/// The GPRs a broadcast load runs through: the buffer's address, wherever
/// the allocator keeps that pointer value, and the one index, this
/// instruction's `RegisterFile::gpr_scratch` reservation. Each tier's
/// `emit_broadcast_load` truncates lane 0 into the index and reads the
/// element once, `vbroadcastss [base + index*4]`, into every lane.
#[derive(Clone, Copy)]
pub(super) struct BroadcastGprs {
    /// The buffer base pointer.
    pub(super) base: PtrReg,
    /// Receives the truncated index.
    pub(super) index: Gpr,
}

#[cfg(test)]
mod label_tests {
    use super::Gp;
    use crate::emit::{Assembly, EncodedInst};

    fn ret() -> EncodedInst {
        EncodedInst::from_slice(&[0xC3])
    }

    /// The one thing a label does that a fixup token could not: name a
    /// position that does not exist yet.
    #[test]
    fn a_forward_branch_names_a_position_bound_later() {
        let mut asm = Assembly::default();
        let end = asm.mint();
        asm.push(Gp::Jmp { to: end });
        asm.push(ret());
        asm.bind(end);
        let code = asm.finish();

        // `jmp rel32` is five bytes; `ret` is one; the label lands at 6. The
        // displacement is measured from the end of the branch, so it is 1.
        assert_eq!(code.len(), 6);
        assert_eq!(code[0], 0xE9);
        assert_eq!(i32::from_le_bytes([code[1], code[2], code[3], code[4]]), 1);
    }

    /// A back edge — the shape a loop is made of, and the reason the
    /// resolution pass is separate from the layout pass.
    #[test]
    fn a_back_edge_resolves_to_a_negative_displacement() {
        let mut asm = Assembly::default();
        let top = asm.mint();
        asm.bind(top);
        asm.push(ret());
        asm.push(Gp::Jmp { to: top });
        let code = asm.finish();

        // `ret` at 0, `jmp` at 1..6. Target 0, origin 6, so the displacement
        // is -6 — and getting this sign backwards is the classic way a loop
        // becomes an infinite one.
        assert_eq!(code.len(), 6);
        assert_eq!(i32::from_le_bytes([code[2], code[3], code[4], code[5]]), -6);
    }

    /// A label may name a position no instruction occupies — the end of the
    /// program. That is why binding a label is a step of its own rather than
    /// a field on an instruction: there is nothing here to hang it on.
    #[test]
    fn a_label_can_end_the_program() {
        let mut asm = Assembly::default();
        let end = asm.mint();
        asm.push(Gp::Jmp { to: end });
        asm.bind(end);
        let code = asm.finish();
        assert_eq!(code.len(), 5);
        assert_eq!(i32::from_le_bytes([code[1], code[2], code[3], code[4]]), 0);
    }

    /// And two labels may name the same position, for the same reason.
    #[test]
    fn two_labels_can_share_a_position() {
        let mut asm = Assembly::default();
        let (a, b) = (asm.mint(), asm.mint());
        asm.push_branch(|next| Gp::je(a, next));
        asm.push(Gp::Jmp { to: b });
        asm.bind(a);
        asm.bind(b);
        let code = asm.finish();
        // `je` is 6 bytes, `jmp` 5, both landing at 11.
        assert_eq!(code.len(), 11);
        assert_eq!(i32::from_le_bytes([code[2], code[3], code[4], code[5]]), 5);
        assert_eq!(i32::from_le_bytes([code[7], code[8], code[9], code[10]]), 0);
    }

    /// Offsets are relative to the program, not the buffer, so a program can
    /// be assembled after bytes that are already there.
    #[test]
    fn a_program_is_position_independent() {
        let program = |prefix: &[u8]| {
            let mut asm = Assembly::default();
            asm.run.extend_from_slice(prefix);
            let end = asm.mint();
            asm.push(Gp::Jmp { to: end });
            asm.bind(end);
            asm.finish()
        };
        let offset = program(&[0x90; 7]);
        assert_eq!(&program(&[])[..], &offset[7..]);
    }

    #[test]
    #[should_panic(expected = "never bound")]
    fn an_unbound_label_is_a_bug_and_not_a_jump_to_itself() {
        let mut asm = Assembly::default();
        let target = asm.mint();
        asm.push(Gp::Jmp { to: target });
        let code = asm.finish();
        unreachable!("assembled {} bytes around an unbound label", code.len());
    }

    #[test]
    #[should_panic(expected = "bound twice")]
    fn a_label_bound_twice_is_a_bug() {
        let mut asm = Assembly::default();
        let twice = asm.mint();
        asm.bind(twice);
        asm.bind(twice);
        let code = asm.finish();
        unreachable!("assembled {} bytes around a label bound twice", code.len());
    }
}

// =============================================================================
// General-purpose registers
// =============================================================================

// The x86-64 general register file.
//
// A distinct type from [`Reg`], which names the *vector* file. They are
// different register files that happen to be numbered the same way, so
// `Gpr(10)` is `r10` and `Reg(10)` is `xmm10`, and nothing can silently pass
// one where the other belongs. Before this existed the general file had no
// type at all: it appeared as bare `u8` in a few encoder signatures and as
// raw opcode bytes everywhere else.
//
// A SIMD language barely touches these — loop counters, pointers, and the
// scalar half of a broadcast — which is why the vocabulary below is nine
// instructions rather than an assembler.

/// The general registers the emitted kernels name. Which is *for* what is
/// the register file's to say (`avx2::driver::AVX2_FILE`,
/// `avx512::driver::AVX512_FILE`), not a constant's.
pub(super) mod gpr {
    use super::Gpr;

    /// Scratch / `movmskps` destination.
    pub(in crate::emit) const RAX: Gpr = Gpr(0);
    /// Scratch; SysV's 4th integer argument, which the kernel ABI does not use.
    pub(in crate::emit) const RCX: Gpr = Gpr(1);
    /// 3rd integer argument: the pitch.
    pub(in crate::emit) const RDX: Gpr = Gpr(2);
    /// 2nd integer argument: the output plane.
    pub(in crate::emit) const RSI: Gpr = Gpr(6);
    /// 1st integer argument: the context pointer — the array of bound buffer
    /// bases, then the uniform and origin blocks. Read-only for the whole
    /// kernel.
    pub(in crate::emit) const RDI: Gpr = Gpr(7);
    pub(in crate::emit) const R9: Gpr = Gpr(9);
    pub(in crate::emit) const R10: Gpr = Gpr(10);
    pub(in crate::emit) const R11: Gpr = Gpr(11);
}

/// SysV argument and scratch pointer registers.
pub(super) mod ptr {
    use super::PtrReg;

    /// Stack pointer register (`rsp`).
    pub(in crate::emit) const RSP: PtrReg = PtrReg(4);
}

/// An x86-64 general-register instruction.
///
/// Generic over what its operands are ([`Stage`]); a field is declared by the
/// class of what it holds, so `lea`'s base is a `Pointer` and its index an
/// `Integer`, and `imul` cannot be handed an address. An instruction that
/// writes `EFLAGS` says so with a `flags` field, `()` at [`Physical`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Gp<S: Stage> {
    /// `mov dst, src`
    Mov {
        dst: S::Write<Pointer>,
        src: S::Read<Pointer>,
    },
    /// `mov dst, src`, an integer between registers: the same bytes as
    /// [`Gp::Mov`], and not the same instruction, because an address and an
    /// integer are different classes.
    MovInt {
        dst: S::Write<Integer>,
        src: S::Read<Integer>,
    },
    /// `mov r32, imm32`: zero-extended into the 64-bit register.
    MovImm32 { dst: S::Write<Integer>, imm: u32 },
    /// `movabs dst, imm64`
    Movabs { dst: S::Write<Integer>, imm: u64 },
    /// `imul dst, src`: the two-operand 64-bit multiply.
    Imul {
        dst: S::Tie<Integer>,
        src: S::Read<Integer>,
        flags: S::Write<Flags>,
    },
    /// `add dst, src`
    Add {
        dst: S::Tie<Integer>,
        src: S::Read<Integer>,
        flags: S::Write<Flags>,
    },
    /// `lea dst, [base + index*4]`: the element address of a plane of `f32`s.
    Lea4 {
        dst: S::Write<Pointer>,
        base: S::Read<Pointer>,
        index: S::Read<Integer>,
    },
    /// `mov dst, [base + disp32]`: a pointer from the context array or a
    /// frame slot.
    MovLoad {
        dst: S::Write<Pointer>,
        src: Mem<S, Imm32>,
    },
    /// `mov [base + disp32], src`: [`Gp::MovLoad`]'s mirror, a pointer to a
    /// frame slot.
    MovStore {
        dst: Mem<S, Imm32>,
        src: S::Read<Pointer>,
    },
    /// `mov [rsp + slot], src`: the allocator's spill of an address.
    SpillPtr {
        src: S::Read<Pointer>,
        slot: S::Slot,
    },
    /// `mov dst, [rsp + slot]`: the allocator's reload of an address.
    ReloadPtr {
        dst: S::Write<Pointer>,
        slot: S::Slot,
    },
    /// `mov [rsp + slot], src`: the allocator's spill of an integer.
    SpillInt {
        src: S::Read<Integer>,
        slot: S::Slot,
    },
    /// `mov dst, [rsp + slot]`: the allocator's reload of an integer.
    ReloadInt {
        dst: S::Write<Integer>,
        slot: S::Slot,
    },
    /// `test src32, src32`: ZF iff the low half is zero.
    Test {
        flags: S::Write<Flags>,
        src: S::Read<Integer>,
    },
    /// `cmp src8, imm8`: ZF iff the low byte is `imm`. Unlike `cmp src32,
    /// imm8` (sign-extending `83 /7`), it compares the raw byte, which is what
    /// an 8-lane all-true check (`movmskps`'s `0xFF`) needs: the extension
    /// would compare against `0xFFFFFFFF`, which a zero-extended mask can
    /// never equal.
    CmpByte {
        flags: S::Write<Flags>,
        src: S::Read<Integer>,
        imm: u8,
    },
    /// `jcc rel32`. `next` is the position laid out right after it, where the
    /// branch falls through to: it encodes to nothing.
    Jcc {
        cond: Cond,
        flags: S::Read<Flags>,
        taken: S::Target,
        next: S::Target,
    },
    /// `jmp rel32`.
    Jmp { to: S::Target },
    /// Go to the position laid out right after this instruction: encodes to
    /// nothing, and is what a transfer to the next block is.
    Fallthrough { to: S::Target },
    /// `lea dst, [rip + to]`: a position's address, in one instruction.
    LeaRip {
        dst: S::Write<Pointer>,
        to: S::Target,
    },
    /// `sub rsp, imm32`: the function's frame.
    Enter {
        size: S::FrameSize,
        flags: S::Write<Flags>,
    },
    /// `add rsp, imm32; vzeroupper; ret`: the function's one exit.
    ///
    /// The caller is Rust built for baseline x86-64, so its floating point is
    /// legacy SSE, and Intel cores charge legacy-SSE code for vector registers
    /// whose upper halves a VEX or EVEX instruction left dirty — a state
    /// transition on older cores, a false dependency and a merge per
    /// instruction on Skylake and later — until something clears them, which
    /// nothing in a Rust caller does. So the kernel clears them on the way
    /// out. Nothing is lost: the collapse ABI returns nothing in a vector
    /// register, and every result is already stored.
    ///
    /// Measured on an AVX-512 Xeon, with a 1×1 kernel whose Rust caller runs
    /// a 256-term scalar sum after each call: the call cost 190 ns over the
    /// sum on the AVX-512 tier and 88 ns on AVX2 without this, and 32 and 15
    /// ns with it. The charge is not the call's: the same sum, run after one
    /// call and never calling again, took 482 ns rather than 266.
    Ret {
        size: S::FrameSize,
        flags: S::Write<Flags>,
    },
}

/// The stack pointer, as the operand of the frame's `sub` and `add`.
const RSP: Gpr = ptr::RSP.as_gpr();

impl Gp<Physical> {
    /// `je taken` — ZF set.
    #[must_use]
    pub(super) const fn je(taken: Label, next: Label) -> Self {
        Gp::Jcc {
            cond: Cond::E,
            flags: (),
            taken,
            next,
        }
    }

    /// `jb taken` — CF set; unsigned `<`.
    #[must_use]
    pub(super) const fn jb(taken: Label, next: Label) -> Self {
        Gp::Jcc {
            cond: Cond::B,
            flags: (),
            taken,
            next,
        }
    }
}

impl Gp<Selected> {
    /// Rebuild at stage `T`, visiting each operand once, in field order.
    pub(super) fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Gp<T> {
        match self {
            Gp::Mov { dst, src } => Gp::Mov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Gp::MovInt { dst, src } => Gp::MovInt {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Gp::MovImm32 { dst, imm } => Gp::MovImm32 {
                dst: f.write(dst),
                imm: *imm,
            },
            Gp::Movabs { dst, imm } => Gp::Movabs {
                dst: f.write(dst),
                imm: *imm,
            },
            Gp::Imul { dst, src, flags } => Gp::Imul {
                dst: f.tie(dst),
                src: f.read(*src),
                flags: f.write(flags),
            },
            Gp::Add { dst, src, flags } => Gp::Add {
                dst: f.tie(dst),
                src: f.read(*src),
                flags: f.write(flags),
            },
            Gp::Lea4 { dst, base, index } => Gp::Lea4 {
                dst: f.write(dst),
                base: f.read(*base),
                index: f.read(*index),
            },
            Gp::MovLoad { dst, src } => Gp::MovLoad {
                dst: f.write(dst),
                src: src.walk(f),
            },
            Gp::MovStore { dst, src } => Gp::MovStore {
                dst: dst.walk(f),
                src: f.read(*src),
            },
            Gp::SpillPtr { src, slot } => Gp::SpillPtr {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Gp::ReloadPtr { dst, slot } => Gp::ReloadPtr {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Gp::SpillInt { src, slot } => Gp::SpillInt {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Gp::ReloadInt { dst, slot } => Gp::ReloadInt {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
            Gp::Test { flags, src } => Gp::Test {
                flags: f.write(flags),
                src: f.read(*src),
            },
            Gp::CmpByte { flags, src, imm } => Gp::CmpByte {
                flags: f.write(flags),
                src: f.read(*src),
                imm: *imm,
            },
            Gp::Jcc {
                cond,
                flags,
                taken,
                next,
            } => Gp::Jcc {
                cond: *cond,
                flags: f.read(*flags),
                taken: f.target(taken),
                next: f.target(next),
            },
            Gp::Jmp { to } => Gp::Jmp { to: f.target(to) },
            Gp::Fallthrough { to } => Gp::Fallthrough { to: f.target(to) },
            // A position's address is a data label, which `walk` would
            // report as a branch: selection reads its constants RIP-relative
            // and never selects this.
            Gp::LeaRip { .. } => unreachable!("selection never selects `lea dst, [rip + to]`"),
            Gp::Enter { size: _, flags } => Gp::Enter {
                size: f.frame_size(),
                flags: f.write(flags),
            },
            Gp::Ret { size: _, flags } => Gp::Ret {
                size: f.frame_size(),
                flags: f.write(flags),
            },
        }
    }
}

impl<S: Placed> Gp<S> {
    pub(super) fn encode(&self) -> EncodedInst {
        let mut inst = EncodedInst::new();
        match self {
            Gp::Mov { dst, src } => {
                inst.extend(&rr(0x89, S::write::<Pointer>(dst), S::read::<Pointer>(src)));
            }
            Gp::MovInt { dst, src } => {
                inst.extend(&rr(0x89, S::write::<Integer>(dst), S::read::<Integer>(src)));
            }
            Gp::MovImm32 { dst, imm } => {
                let dst = S::write::<Integer>(dst);
                if dst >= 8 {
                    inst.push(0x41);
                }
                inst.push(0xB8 | (dst & 7));
                inst.extend(&imm.to_le_bytes());
            }
            Gp::Movabs { dst, imm } => {
                let dst = S::write::<Integer>(dst);
                inst.push(0x48 | ((dst >> 3) & 1));
                inst.push(0xB8 | (dst & 7));
                inst.extend(&imm.to_le_bytes());
            }
            Gp::Imul { dst, src, .. } => {
                let (dst, src) = (S::tie::<Integer>(dst), S::read::<Integer>(src));
                inst.extend(&[rex_w(dst, src), 0x0F, 0xAF, modrm_rr(dst, src)]);
            }
            Gp::Add { dst, src, .. } => {
                inst.extend(&rr(0x01, S::tie::<Integer>(dst), S::read::<Integer>(src)));
            }
            // `rbp`/`r13` have no `mod = 00` form as a SIB base (that
            // encoding means "no base"), so those two take `mod = 01` with a
            // zero `disp8`.
            Gp::Lea4 { dst, base, index } => {
                let (dst, base, index) = (
                    S::write::<Pointer>(dst),
                    S::read::<Pointer>(base),
                    S::read::<Integer>(index),
                );
                debug_assert!(index & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
                let disp8_form = base & 7 == RM_RIP_AT_MOD0;
                inst.push(
                    0x48 | (((dst >> 3) & 1) << 2) | (((index >> 3) & 1) << 1) | ((base >> 3) & 1),
                );
                inst.push(0x8D);
                inst.push(if disp8_form { 0x40 } else { 0x00 } | ((dst & 7) << 3) | RM_SIB);
                inst.push((0b10 << 6) | ((index & 7) << 3) | (base & 7));
                if disp8_form {
                    inst.push(0);
                }
            }
            Gp::MovLoad { dst, src } => {
                let (dst, base) = (S::write::<Pointer>(dst), S::read::<Pointer>(&src.base));
                inst.push(rex_w(dst, base));
                inst.push(0x8B);
                mem_operand_into(&mut inst, dst, base, src.disp);
            }
            Gp::MovStore { dst, src } => {
                let (src, base) = (S::read::<Pointer>(src), S::read::<Pointer>(&dst.base));
                inst.push(rex_w(src, base));
                inst.push(0x89);
                mem_operand_into(&mut inst, src, base, dst.disp);
            }
            Gp::SpillPtr { src, slot } => {
                spill_into(&mut inst, S::read::<Pointer>(src), Imm32(S::slot(slot)));
            }
            Gp::SpillInt { src, slot } => {
                spill_into(&mut inst, S::read::<Integer>(src), Imm32(S::slot(slot)));
            }
            Gp::ReloadPtr { dst, slot } => {
                reload_into(&mut inst, S::write::<Pointer>(dst), Imm32(S::slot(slot)));
            }
            Gp::ReloadInt { dst, slot } => {
                reload_into(&mut inst, S::write::<Integer>(dst), Imm32(S::slot(slot)));
            }
            Gp::Test { src, .. } => {
                let src = S::read::<Integer>(src);
                if src >= 8 {
                    inst.push(0x45);
                }
                inst.push(0x85);
                inst.push(modrm_rr(src, src));
            }
            Gp::CmpByte { src, imm, .. } => {
                match S::read::<Integer>(src) {
                    // The accumulator has a short form.
                    0 => inst.push(0x3C),
                    n => {
                        // Without a REX prefix 4-7 name `ah`..`bh`, not `spl`..`dil`.
                        if n >= 4 {
                            inst.push(0x40 | ((n >> 3) & 1));
                        }
                        inst.push(0x80);
                        inst.push(modrm_rr(7, n));
                    }
                }
                inst.push(*imm);
            }
            Gp::Jcc { cond, .. } => inst.extend(&[0x0F, 0x80 | *cond as u8, 0, 0, 0, 0]),
            Gp::Jmp { .. } => inst.extend(&[0xE9, 0, 0, 0, 0]),
            Gp::Fallthrough { .. } => {}
            Gp::LeaRip { dst, .. } => {
                let dst = S::write::<Pointer>(dst);
                inst.push(0x48 | (((dst >> 3) & 1) << 2));
                inst.push(0x8D);
                inst.push(((dst & 7) << 3) | RM_RIP_AT_MOD0);
                inst.extend(&[0, 0, 0, 0]);
            }
            Gp::Enter { size, .. } => {
                inst.extend(&[rex_w(0, RSP.0), 0x81, modrm_rr(5, RSP.0)]);
                inst.extend(&S::frame_size(size).to_le_bytes());
            }
            Gp::Ret { size, .. } => {
                inst.extend(&[rex_w(0, RSP.0), 0x81, modrm_rr(0, RSP.0)]);
                inst.extend(&S::frame_size(size).to_le_bytes());
                // `vzeroupper`: `VEX.128.0F.WIG 77`, zeroing bits 128 and up of
                // vector registers 0–15, the sixteen a legacy-SSE instruction
                // can name. The same three bytes on every tier.
                inst.extend(&[0xC5, 0xF8, 0x77]);
                inst.push(0xC3);
            }
        }
        inst
    }

    /// The instruction's bytes, and the label fields in them.
    pub(super) fn assemble(&self, out: &mut Encoding<'_>) {
        out.bytes(self.encode().as_bytes());
        match self {
            Gp::Jmp { to } => out.field(JMP_DISP, S::target(to), patch_rel32),
            Gp::Jcc { taken, next, .. } => {
                out.field(JCC_DISP, S::target(taken), patch_rel32);
                out.falls_through(S::target(next));
            }
            Gp::Fallthrough { to } => out.falls_through(S::target(to)),
            Gp::LeaRip { to, .. } => out.field(LEA_RIP_DISP, S::target(to), patch_rel32),
            _ => {}
        }
    }
}

impl AsmInsn for Gp<Physical> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.encode().emit_into(code);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        let (at, label) = match self {
            Gp::Jmp { to } => (JMP_DISP, to),
            Gp::Jcc { taken, .. } => (JCC_DISP, taken),
            Gp::LeaRip { to, .. } => (LEA_RIP_DISP, to),
            _ => return None,
        };
        Some(LabelRef {
            at,
            label,
            patch: patch_rel32,
        })
    }
}

/// A sign-extended 8-bit immediate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Imm8(pub(super) i8);

/// A 32-bit immediate: a displacement's width in the `mod = 10` form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Imm32(pub(super) i32);

/// `REX.W` plus the extension bits for a two-register form.
///
/// `R` extends the ModRM.reg field (the source here), `B` extends ModRM.rm
/// (the destination).
#[inline(always)]
const fn rex_w(reg: u8, rm: u8) -> u8 {
    0x48 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1)
}

/// ModRM for the register-direct form: `mod = 11`.
#[inline(always)]
const fn modrm_rr(reg: u8, rm: u8) -> u8 {
    0xC0 | ((reg & 7) << 3) | (rm & 7)
}

/// `REX.W opcode /r` with both operands in registers.
#[inline(always)]
const fn rr(opcode: u8, dst: u8, src: u8) -> [u8; 3] {
    [rex_w(src, dst), opcode, modrm_rr(src, dst)]
}

/// `mov [rsp + slot], src`.
fn spill_into(inst: &mut EncodedInst, src: u8, slot: Imm32) {
    inst.push(rex_w(src, RSP.0));
    inst.push(0x89);
    mem_operand_into(inst, src, RSP.0, slot);
}

/// `mov dst, [rsp + slot]`.
fn reload_into(inst: &mut EncodedInst, dst: u8, slot: Imm32) {
    inst.push(rex_w(dst, RSP.0));
    inst.push(0x8B);
    mem_operand_into(inst, dst, RSP.0, slot);
}

/// The 4-bit condition an x86 `jcc` tests: the conditions the emitter
/// branches on.
///
/// `0F 8x rel32` is one instruction whose low opcode nibble *is* this value, so
/// the assembler encodes it by casting rather than by dispatching to one
/// hand-written mnemonic per condition. Named by the ISA's own mnemonics, with
/// their aliases, because that is what a reader checks against the manual.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Cond {
    /// `jb` / `jc` / `jnae` — CF set; unsigned `<`.
    B = 0x2,
    /// `je` / `jz` — ZF set.
    E = 0x4,
}

/// Bytes from a `jmp`'s start to its displacement field: one opcode byte.
const JMP_DISP: usize = 1;
/// Bytes from a `jcc`'s start to its displacement field: `0F` plus the
/// condition byte.
const JCC_DISP: usize = 2;
/// Bytes from a `lea`'s start to its displacement field: REX, opcode, ModRM.
const LEA_RIP_DISP: usize = 3;

/// Write a `rel32` at `pos` so the instruction it belongs to reaches `target`.
///
/// `rel32` is measured from the *end* of the instruction, which is the end of
/// the displacement field itself.
pub(super) fn patch_rel32(code: &mut [u8], pos: usize, target: usize) {
    let rel = (target as i64) - (pos as i64 + 4);
    let rel = i32::try_from(rel).expect("an x86 rel32 spans \u{00b1}2 GiB");
    code[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
}

// =============================================================================
// The vector operations both tiers encode
// =============================================================================

// What an operation *is* is the architecture's; the bytes that say it are the
// tier's, so `avx2.rs` and `avx512.rs` each give these a `vex()` or `evex()`.

/// A three-operand arithmetic or bitwise instruction: `op dst, a, b`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Alu {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    And,
    /// `!a & b`
    AndNot,
    Or,
    Xor,
    /// `vpaddd`: the integer-domain add.
    IAdd,
}

/// A one-source instruction: `op dst, src`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Lanewise {
    Sqrt,
    Rsqrt,
    Recip,
    /// `vcvttps2dq`
    ToInt,
    /// `vcvtdq2ps`
    FromInt,
    /// `vpmovzxbd`: bytes widened to dword lanes.
    WidenBytes,
}

/// The predicate of a `vcmpps`: its imm8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Pred {
    Eq = 0,
    Lt = 1,
    Le = 2,
    Ne = 4,
    Ge = 5,
    /// `>`: the unordered-safe "not less-or-equal".
    Nle = 6,
}

/// The rounding mode of a `vroundps` or `vrndscaleps`: its imm8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Rounding {
    Nearest = 0,
    Floor = 1,
    Ceil = 2,
}

/// The direction of an integer shift by an immediate: the `/digit` of its
/// opcode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    Left = 6,
    Right = 2,
}

/// A tier's `cvttss2si`, the one instruction the store's address arithmetic
/// needs a vector encoding for: lane 0 of a register, or the first word of a
/// slot, truncated into a general register.
pub(super) trait Truncate: AsmInsn {
    /// `cvttss2si dst, src`
    fn from_xmm(dst: Gpr, src: Reg) -> Self;
    /// `cvttss2si dst, [src]`
    fn from_slot(dst: Gpr, src: Mem<Physical, Imm32>) -> Self;
}

// =============================================================================
// The store's address arithmetic
// =============================================================================

/// A slot in the allocated spill frame. Kernels are leaves with no base
/// pointer, so a slot *is* `rsp + offset`, on every x86 tier.
pub(super) const fn frame_slot(offset: u32) -> Mem<Physical, Imm32> {
    Mem {
        base: ptr::RSP,
        disp: Imm32(offset as i32),
    }
}

/// Bytes one `f32` occupies: the element pitch of a uniform block.
pub(super) const F32_BYTES: u64 = 4;

/// The `offset`-th `f32` of the block at `base`, `[base + 4*offset]`, as a
/// `disp32` operand — the one place a uniform's or a pool entry's 64-bit
/// position meets the width x86 gives a displacement, shared by the VEX and
/// EVEX tiers the way [`mem_operand_into`] is.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when `4 * offset` does not fit a signed
/// 32-bit displacement: an offset past the encoding is refused, never
/// wrapped into an address that reads some other argument.
pub(super) fn block_element(
    base: PtrReg,
    offset: u64,
) -> Result<Mem<Physical, Imm32>, CompileError> {
    Ok(Mem {
        base,
        disp: displacement(offset, F32_BYTES)?,
    })
}

/// The `disp32` that reaches element `index` of an array of `size`-byte
/// elements.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when it does not fit a signed 32-bit
/// displacement.
pub(super) fn displacement(index: u64, size: u64) -> Result<Imm32, CompileError> {
    index
        .checked_mul(size)
        .and_then(|bytes| i32::try_from(bytes).ok())
        .map(Imm32)
        .ok_or(CompileError::BudgetExceeded(
            "block offset past x86's disp32",
        ))
}

/// `dst = trunc(index)` as a 64-bit integer, wherever a fold keeps its
/// binder: a broadcast, so lane 0 of a register or the first word of a
/// slot is the index. Shared by the x86 tiers, which differ only in the
/// *vector* encoding this reads through — the GPR half is the
/// architecture's.
fn index_into<I: Truncate>(code: &mut Vec<u8>, dst: Gpr, at: Binding) {
    match at {
        Binding::Loc(Loc::Reg(r)) => I::from_xmm(dst, r).emit_into(code),
        Binding::Loc(Loc::Slot(slot)) => {
            I::from_slot(dst, frame_slot(slot.offset())).emit_into(code)
        }
        Binding::Loc(Loc::Ptr(_)) => unreachable!("a fold's binder is a vector"),
        // `emit_scope` hands a rematerialized binder over as its slot, so the
        // only caller, `write_address`, never holds a constant here.
        Binding::Remat(bits) => unreachable!(
            "a fold's binder is read from a register or a slot, never rematerialized ({bits:#x})"
        ),
    }
}

/// The store's address: `out + 4 · (row · pitch + col)`, in `scratch[0]`,
/// leaving `scratch[1]` free. The row and column indices are converted
/// through `I`, the tier's own `cvttss2si`.
pub(super) fn write_address<I: Truncate>(
    code: &mut Vec<u8>,
    file: &regalloc::RegisterFile,
    write: &WritePlan,
) -> PtrReg {
    let row = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(0));
    let col = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(1));
    let out = file.gpr_out.expect("x86's store needs the output pointer");
    let pitch = file.gpr_pitch.expect("x86's store needs the pitch");
    index_into::<I>(code, row, write.row);
    Gp::Imul {
        dst: row,
        src: pitch,
        flags: (),
    }
    .emit_into(code);
    index_into::<I>(code, col, write.col);
    Gp::Add {
        dst: row,
        src: col,
        flags: (),
    }
    .emit_into(code);
    let address = PtrReg(row.0);
    Gp::Lea4 {
        dst: address,
        base: PtrReg(out.0),
        index: row,
    }
    .emit_into(code);
    address
}

// =============================================================================
// Memory operands
// =============================================================================

/// The displacement half of a [`Mem`] — and, because x86 spells the
/// displacement's *width* in the ModRM `mod` field, the addressing mode.
///
/// `mod = 00 / 01 / 10` are three modes rather than three spellings of one:
/// they cost a different number of bytes, and `mod = 00` is not "a
/// displacement of zero" (see [`NoDisp`]). So the width is picked by the
/// operand's TYPE — never by the caller reaching for a differently-named
/// function, which is where that choice used to live.
pub(super) trait Disp: Copy {
    /// The ModRM `mod` field this displacement implies.
    const MOD: u8;
    /// Append the displacement bytes into an `EncodedInst`.
    fn emit_inst(self, inst: &mut EncodedInst);
}

/// No displacement — the `mod = 00` form, `[base]`.
///
/// A mode of its own, not `Imm8(0)`: it is a byte shorter, and it is not
/// available for every base. `mod = 00` with `rm = 101` (rbp/r13) means
/// RIP-relative, a different address entirely, so those two registers have no
/// bare `[base]` form and must spell it `Imm8(0)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct NoDisp;

impl Disp for NoDisp {
    const MOD: u8 = 0x00;
    #[inline(always)]
    fn emit_inst(self, _inst: &mut EncodedInst) {}
}

impl Disp for Imm8 {
    const MOD: u8 = 0x40;
    #[inline(always)]
    fn emit_inst(self, inst: &mut EncodedInst) {
        inst.push(self.0 as u8);
    }
}

impl Disp for Imm32 {
    const MOD: u8 = 0x80;
    #[inline(always)]
    fn emit_inst(self, inst: &mut EncodedInst) {
        inst.extend(&self.0.to_le_bytes());
    }
}

/// An address spelled `[base + disp]`.
///
/// The base being a [`PtrReg`] is the point: `rsp` is a value here. It used
/// to be the `_rsp` and `_base` suffixes of five separate functions that all
/// encoded the same `movups`, where nothing could check it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Mem<S: Stage, D> {
    /// The register the displacement is measured from.
    pub(super) base: S::Read<Pointer>,
    /// The displacement — and, through its type, the mode (see [`Disp`]).
    pub(super) disp: D,
}

impl<D: Copy> Mem<Selected, D> {
    /// Rebuild at stage `T`: the base is the one operand.
    pub(super) fn walk<T: Stage>(&self, f: &mut impl Rebind<T>) -> Mem<T, D> {
        Mem {
            base: f.read(self.base),
            disp: self.disp,
        }
    }
}

/// ModRM `rm` meaning "a SIB byte follows" — also the low three bits of
/// `rsp`/`r12`, which is why exactly those two bases always take one.
const RM_SIB: u8 = 0b100;

/// ModRM `rm` that means RIP-relative when `mod = 00` — also the low three
/// bits of `rbp`/`r13`.
const RM_RIP_AT_MOD0: u8 = 0b101;

/// SIB naming the base register alone: scale 1, index `100` (none).
const SIB_BASE_ONLY: u8 = 0x24;

/// Write the ModRM/SIB tail for `[base + index*4]` into an `EncodedInst`:
/// `mod = 00`, a SIB with scale 4 and no displacement — the element of a
/// plane of `f32`s, addressed in one instruction. The tail is the
/// architecture's, not the prefix's, so the VEX and EVEX tiers share it the
/// way they share [`mem_operand_into`]. Registers are hardware numbers.
pub(super) fn scaled4_operand_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    debug_assert!(index & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
    sib4_tail_into(inst, reg, base, index);
}

/// [`scaled4_operand_into`] with a *vector* index — the VSIB a gather
/// addresses through, `[base + ymm*4]`. Same bytes; the one rule that does
/// not carry over is the GPR one, because SIB index `100` means "no index"
/// only when the index is a general register: as a vector number it is
/// `ymm4`/`ymm12`, which a gather may perfectly well be indexed by. The
/// prefix's X bit carries the index's high bit either way.
pub(super) fn vsib4_operand_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    sib4_tail_into(inst, reg, base, index);
}

/// The ModRM/SIB bytes both scaled-index forms share.
fn sib4_tail_into(inst: &mut EncodedInst, reg: u8, base: u8, index: u8) {
    assert!(
        base & 7 != RM_RIP_AT_MOD0,
        "[r{base} + index*4] has no mod=00 form: rbp/r13 as a SIB base means no base"
    );
    inst.push(((reg & 7) << 3) | RM_SIB);
    inst.push((0b10 << 6) | ((index & 7) << 3) | (base & 7));
}

/// Write the ModRM/SIB/disp tail of `[base + disp]` into an `EncodedInst`.
pub(super) fn mem_operand_into<D: Disp>(inst: &mut EncodedInst, reg: u8, base: u8, disp: D) {
    let rm = base & 7;
    debug_assert!(
        D::MOD != NoDisp::MOD || rm != RM_RIP_AT_MOD0,
        "[rbp]/[r13] has no mod=00 form: that encoding is RIP-relative"
    );
    inst.push(D::MOD | ((reg & 7) << 3) | rm);
    if rm == RM_SIB {
        inst.push(SIB_BASE_ONLY);
    }
    disp.emit_inst(inst);
}

/// The ModRM and `disp32` of `[rip + label]`: the displacement is a label
/// field, the last four bytes of the instruction.
pub(super) fn rip_operand_into(inst: &mut EncodedInst, reg: u8) {
    inst.push(((reg & 7) << 3) | RM_RIP_AT_MOD0);
    inst.extend(&[0, 0, 0, 0]);
}

#[cfg(test)]
mod gpr_tests {
    use super::gpr::*;
    use super::*;
    use crate::emit::AsmProgram;

    fn asm(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut c = Vec::new();
        f(&mut c);
        c
    }

    fn one(inst: EncodedInst) -> Vec<u8> {
        asm(|c| AsmProgram::from([inst]).assemble(c))
    }

    fn gp(inst: Gp<Physical>) -> Vec<u8> {
        asm(|c| AsmProgram::from([inst]).assemble(c))
    }

    fn mov(dst: u8, src: u8) -> Vec<u8> {
        gp(Gp::Mov {
            dst: PtrReg(dst),
            src: PtrReg(src),
        })
    }

    /// A vehicle for the memory-operand tail: `movups [addr], xmm` (`0F 11
    /// /r`), the simplest legacy instruction that takes a [`Mem`]. Test-only
    /// — no tier emits it — so what is under test is [`mem_operand_into`]
    /// and the REX bits, not a 128-bit store.
    fn movups_store<D: Disp>(src: Reg, addr: Mem<Physical, D>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rex = 0x40 | (u8::from(src.0 >= 8) << 2) | u8::from(addr.base.0 >= 8);
        if rex != 0x40 {
            inst.push(rex);
        }
        inst.push(0x0F);
        inst.push(0x11);
        mem_operand_into(&mut inst, src.0, addr.base.0, addr.disp);
        inst
    }

    /// Each encoding checked against the Intel SDM's form for that mnemonic.
    #[test]
    fn encodings_match_the_manual() {
        // REX.W 89 /r — MOV r/m64, r64
        assert_eq!(mov(10, 1), [0x49, 0x89, 0xCA]);
        // REX.W 01 /r — ADD r/m64, r64
        assert_eq!(
            gp(Gp::Add {
                dst: RSI,
                src: Gpr(8),
                flags: ()
            }),
            [0x4C, 0x01, 0xC6]
        );
        // REX.W 81 /5 id — SUB r/m64, imm32
        assert_eq!(
            gp(Gp::Enter {
                size: 16,
                flags: ()
            }),
            [0x48, 0x81, 0xEC, 16, 0, 0, 0]
        );
        // REX.W 81 /0 id — ADD r/m64, imm32; VEX.128.0F.WIG 77 — VZEROUPPER;
        // C3 — RET
        assert_eq!(
            gp(Gp::Ret {
                size: 16,
                flags: ()
            }),
            [0x48, 0x81, 0xC4, 16, 0, 0, 0, 0xC5, 0xF8, 0x77, 0xC3]
        );
        // 85 /r — TEST r/m32, r32
        assert_eq!(
            gp(Gp::Test {
                flags: (),
                src: RAX
            }),
            [0x85, 0xC0]
        );
        assert_eq!(
            gp(Gp::Test { flags: (), src: R9 }),
            [0x45, 0x85, 0xC9],
            "REX.R and REX.B"
        );
        // 3C ib — CMP AL, imm8; 80 /7 ib — CMP r/m8, imm8 (checked against
        // `objdump -M intel`, binutils 2.42).
        let cmp = |src| {
            gp(Gp::CmpByte {
                flags: (),
                src,
                imm: 0xFF,
            })
        };
        assert_eq!(cmp(RAX), [0x3C, 0xFF], "the accumulator's short form");
        assert_eq!(cmp(RCX), [0x80, 0xF9, 0xFF]);
        assert_eq!(cmp(RSI), [0x40, 0x80, 0xFE, 0xFF], "REX names sil, not dh");
        assert_eq!(cmp(R9), [0x41, 0x80, 0xF9, 0xFF], "REX.B");
        // REX.W B8+rd io — MOV r64, imm64
        assert_eq!(
            gp(Gp::Movabs {
                dst: RAX,
                imm: 0x3F80_0000_0000_0000
            }),
            [0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0x80, 0x3F]
        );
        assert_eq!(gp(Gp::Movabs { dst: R9, imm: 1 })[..2], [0x49, 0xB9]);
        // B8+rd id — MOV r32, imm32
        assert_eq!(gp(Gp::MovImm32 { dst: RCX, imm: 7 }), [0xB9, 7, 0, 0, 0]);
        assert_eq!(
            gp(Gp::MovImm32 { dst: R9, imm: 7 }),
            [0x41, 0xB9, 7, 0, 0, 0]
        );
        // REX.W 0F AF /r — IMUL r64, r/m64
        assert_eq!(
            gp(Gp::Imul {
                dst: RAX,
                src: RDX,
                flags: ()
            }),
            [0x48, 0x0F, 0xAF, 0xC2]
        );
        // REX.W 8D /r — LEA r64, [base + index*4]
        let lea = |dst, base, index| {
            gp(Gp::Lea4 {
                dst: PtrReg(dst),
                base: PtrReg(base),
                index: Gpr(index),
            })
        };
        assert_eq!(lea(0, 6, 0), [0x48, 0x8D, 0x04, 0x86]);
        assert_eq!(
            lea(1, 5, 1),
            [0x48, 0x8D, 0x4C, 0x8D, 0x00],
            "rbp as a base takes the disp8 form"
        );
        assert_eq!(lea(9, 6, 10)[0], 0x4E, "REX.R and REX.X");
    }

    #[test]
    fn asm_program_declarative_array() {
        let mut buff = Vec::new();
        let add = Gp::Add {
            dst: R9,
            src: R10,
            flags: (),
        };
        AsmProgram::from([add]).assemble(&mut buff);
        assert_eq!(buff, [0x4D, 0x01, 0xD1]);

        let mut seq = Vec::new();
        AsmProgram::from([add, Gp::Ret { size: 0, flags: () }]).assemble(&mut seq);
        assert_eq!(
            seq,
            [
                0x4D, 0x01, 0xD1, // add r9, r10
                0x48, 0x81, 0xC4, 0, 0, 0, 0, // add rsp, 0
                0xC5, 0xF8, 0x77, // vzeroupper
                0xC3, // ret
            ]
        );
    }

    /// REX.R extends the source, REX.B the destination; a register above r7
    /// on either side must set its own bit and no other.
    #[test]
    fn rex_extends_each_operand_independently() {
        assert_eq!(mov(1, 2)[0], 0x48, "neither extended");
        assert_eq!(mov(9, 2)[0], 0x49, "destination extended → B");
        assert_eq!(mov(1, 9)[0], 0x4C, "source extended → R");
        assert_eq!(mov(9, 10)[0], 0x4D, "both extended");
    }

    /// A branch's displacement is measured from the *next* instruction, and a
    /// backward one is negative. Asserted through the assembler, because that
    /// is who writes it: the mnemonics emit a zero placeholder and report
    /// nothing.
    #[test]
    fn branches_patch_relative_to_the_next_instruction() {
        use crate::emit::Assembly;

        let mut asm = Assembly::default();
        let end = asm.mint();
        asm.push(Gp::Jmp { to: end });
        // Eleven bytes of padding, so the label lands at 16.
        asm.push(EncodedInst::from_slice(&[0x90; 11]));
        asm.bind(end);
        let c = asm.finish();
        assert_eq!(c.len(), 16);
        assert_eq!(c[0], 0xE9, "E9 + rel32");
        assert_eq!(
            &c[1..5],
            &(16i32 - 5).to_le_bytes(),
            "rel is from the next insn"
        );

        let mut asm = Assembly::default();
        let top = asm.mint();
        asm.bind(top);
        asm.push_branch(|next| Gp::jb(top, next));
        let c = asm.finish();
        assert_eq!(c[..2], [0x0F, 0x82]);
        assert_eq!(&c[2..6], &(-6i32).to_le_bytes(), "a back edge is negative");
    }

    /// The `Context` def's own instruction: `mov r9, [rdi + 16]` (`REX.WR 8B
    /// /r`) for context slot 2.
    #[test]
    fn a_context_pointer_is_read_by_one_mov() {
        let mut code = Vec::new();
        AsmProgram::from([Gp::MovLoad {
            dst: PtrReg(9),
            src: Mem {
                base: PtrReg(7),
                disp: Imm32(2 * PTR_BYTES),
            },
        }])
        .assemble(&mut code);
        assert_eq!(code, [0x4C, 0x8B, 0x8F, 0x10, 0, 0, 0]);
    }

    /// The pointer class's loads and stores reach every GPR and address the
    /// frame: `mov r9, [rdi + 96]` (REX.R for the high destination), `mov
    /// r10, [rsp + 32]` (REX.R, and `rsp`'s SIB), `mov [rsp + 32], r11`.
    #[test]
    fn pointer_loads_and_stores_encode_high_registers_and_the_frame() {
        let mut code = Vec::new();
        let frame = |offset| Mem {
            base: ptr::RSP,
            disp: Imm32(offset),
        };
        AsmProgram::from([
            Gp::MovLoad {
                dst: PtrReg(9),
                src: Mem {
                    base: PtrReg(7),
                    disp: Imm32(96),
                },
            },
            Gp::MovLoad {
                dst: PtrReg(10),
                src: frame(32),
            },
            Gp::MovStore {
                dst: frame(32),
                src: PtrReg(11),
            },
        ])
        .assemble(&mut code);
        assert_eq!(
            code,
            vec![
                0x4C, 0x8B, 0x8F, 96, 0, 0, 0, // mov r9, [rdi + 96]
                0x4C, 0x8B, 0x94, 0x24, 32, 0, 0, 0, // mov r10, [rsp + 32]
                0x4C, 0x89, 0x9C, 0x24, 32, 0, 0, 0, // mov [rsp + 32], r11
            ]
        );
    }

    /// REX extends each side of a memory operand independently: R for the
    /// register, B for the base, both, or neither.
    #[test]
    fn a_memory_operands_rex_bits_follow_its_two_registers() {
        let store = |reg, base| {
            one(movups_store(
                reg,
                Mem {
                    base: PtrReg(base),
                    disp: NoDisp,
                },
            ))
        };
        assert_eq!(store(Reg(3), 3), [0x0F, 0x11, ((3 & 7) << 3) | 3]);
        assert_eq!(store(Reg(9), 3), [0x44, 0x0F, 0x11, ((9 & 7) << 3) | 3]);
        assert_eq!(
            store(Reg(2), 11),
            [0x41, 0x0F, 0x11, ((2 & 7) << 3) | (11 & 7)]
        );
        assert_eq!(
            store(Reg(11), 14),
            [0x45, 0x0F, 0x11, ((11 & 7) << 3) | (14 & 7)]
        );
    }

    /// One instruction, three bases: `[rsp+8]`, `[rax+8]` and `[r10+8]`
    /// differ only in ModRM.rm (plus the SIB `rsp` implies and the REX.B `r10`
    /// does). That is why the base is an operand and not a name suffix.
    #[test]
    fn the_base_register_is_an_operand() {
        let store = |base| {
            one(movups_store(
                Reg(1),
                Mem {
                    base,
                    disp: Imm8(8),
                },
            ))
        };
        // 0F 11 /r, mod=01: rsp takes a SIB byte, rax and r10 do not.
        assert_eq!(store(PtrReg(4)), [0x0F, 0x11, 0x4C, 0x24, 0x08]);
        assert_eq!(store(PtrReg(0)), [0x0F, 0x11, 0x48, 0x08]);
        assert_eq!(store(PtrReg(10)), [0x41, 0x0F, 0x11, 0x4A, 0x08]);
    }

    /// The displacement's *type* picks the encoding, so the same address at the
    /// same offset is a 5-byte or an 8-byte instruction depending on which
    /// operand the caller built — and `NoDisp` is a third, shorter mode, not
    /// `Imm8(0)`.
    #[test]
    fn the_displacement_type_picks_the_encoding() {
        let d8 = one(movups_store(
            Reg(0),
            Mem {
                base: PtrReg(4),
                disp: Imm8(16),
            },
        ));
        let d32 = one(movups_store(
            Reg(0),
            Mem {
                base: PtrReg(4),
                disp: Imm32(16),
            },
        ));
        assert_eq!(d8, [0x0F, 0x11, 0x44, 0x24, 0x10], "mod=01, disp8");
        assert_eq!(
            d32,
            [0x0F, 0x11, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00],
            "mod=10, disp32"
        );

        let bare = one(movups_store(
            Reg(0),
            Mem {
                base: PtrReg(6),
                disp: NoDisp,
            },
        ));
        let zero = one(movups_store(
            Reg(0),
            Mem {
                base: PtrReg(6),
                disp: Imm8(0),
            },
        ));
        assert_eq!(bare, [0x0F, 0x11, 0x06], "mod=00 is its own mode");
        assert_eq!(zero, [0x0F, 0x11, 0x46, 0x00], "and a byte longer than it");
    }

    /// The frame slot every x86 tier spills through: `[rsp + disp32]`.
    #[test]
    fn a_frame_slot_is_rsp_plus_a_disp32() {
        let slot = frame_slot(64);
        assert_eq!(slot.base, ptr::RSP);
        assert_eq!(slot.disp, Imm32(64));
        assert_eq!(
            one(movups_store(Reg(9), slot)),
            [0x44, 0x0F, 0x11, 0x8C, 0x24, 64, 0, 0, 0]
        );
    }

    /// A constant is entered once, at the next four-byte slot, and read again
    /// from where it first went. Where the slots run out is
    /// `an_offset_past_the_displacement_is_refused_on_every_backend`'s: this
    /// pool's operand is the same `block_element`.
    #[test]
    fn a_constant_is_entered_once_at_four_bytes_a_slot() {
        let (a, b) = (1.0f32.to_bits(), 2.0f32.to_bits());
        let mut pool = ConstPool::default();
        let mut disp = |bits| pool.operand(bits).expect("two entries fit a disp32").disp;
        let (first, second, again) = (disp(a), disp(b), disp(a));
        assert_eq!((first, second, again), (Imm32(0), Imm32(4), Imm32(0)));
        assert_eq!(pool.entries, [a, b], "the repeat is not entered again");
    }

    /// `Gpr` and `Reg` name different files; the same index is a different
    /// register in each, which is why they are different types.
    #[test]
    fn the_two_register_files_are_not_interchangeable() {
        // r10 and xmm10 share an index and nothing else.
        assert_eq!(R10.0, Reg(10).0);
        // `Mov` takes registers of the general file; passing Reg(10) would
        // not compile. Encoding r10 as the destination sets REX.B, which a
        // vector encoder never emits here.
        assert_eq!(mov(R10.0, RAX.0)[0] & 1, 1);
    }
}
