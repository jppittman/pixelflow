//! x86-64 AVX-512 (EVEX) JIT encoder — 512-bit, 16-lane `zmm` kernels.
//!
//! The widest of the x86-64 tiers (`crate::isa`), above the AVX2 VEX
//! encoders (`avx2.rs`, 256-bit). It targets the full `zmm0..zmm31` register
//! file via EVEX, so it can also use the extended registers (`zmm16..31`)
//! that VEX cannot reach; the general-register half of every kernel and the
//! vector instructions both tiers have are `x86_64.rs`'s, shared with AVX2.
//! What is this file's is the EVEX prefix, the instructions only AVX-512 has
//! ([`Extra`]), and how it selects.
//!
//! Scope: arithmetic, FMA, sqrt/recip/rsqrt, min/max, bitwise, comparisons,
//! `If`, constant broadcast, the integer bit-manipulation atoms
//! (`IAdd`/`BitAnd`/`BitOr`/`TruncToInt`/`IntToFloat`), and shifts by an
//! immediate — so the exp/log lowering reaches this backend intact.
//! A comparison is `vcmpps` into a mask register. [`Avx512`] keeps it in `k`
//! ([`Lane`]) and widens it (`vpmovm2d`) only where an instruction has no mask
//! form. Note `vpmovm2d` is AVX-512**DQ**, not F: an
//! F-only part would fault on any kernel containing a comparison.
//!
//! Transcendentals are expanded by `legalize` before any backend sees them;
//! ops with no rule here are refused up front rather than mis-emitted.
//!
//! Spills use a real stack frame (a `zmm` is 64 bytes — far past the 128-byte
//! red zone).

use super::build::{Builder, Spiller};
use super::regalloc::resource::FrameSlot;
use super::x86_64;
use super::x86_64::{
    Direction, Evex, EvexPrefix, Lanewise, Mem, NoDisp, Pred, RegField, Sign, Vex, X86, emit,
};
use super::{
    Bound, ClassId, Edges, EncodedInst, Flags, Integer, LaneOp, Opmask, Pointer, Rebind, Selected,
    Stage, Store, Test, Value, ValueName, Vector,
};
use crate::error::CompileError;
use crate::program::IfArm;
use crate::program::MaskOp;

/// `vpternlogd`'s truth table for `A ? B : C` per bit, with `A` the
/// destination (the mask), `B` the first source and `C` the second.
const TERNLOG_SELECT: u8 = 0xCA;

type Inst<S> = x86_64::Inst<S, Avx512>;
type Op<S> = x86_64::Op<S, Avx512>;

/// The instructions only AVX-512 has: EVEX forms with no VEX twin, and the
/// VEX-encoded mask-register forms. A destination that is also a source says
/// so with a `Tie`.
pub(super) enum Extra<S: Stage> {
    /// `vpternlogd dst, if_true, if_false, 0xCA`: `dst = dst ? if_true :
    /// if_false` per bit, `dst` holding an all-ones/all-zeros mask going in.
    Blend {
        dst: S::Tie<Vector>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
    },
    /// `vpinsrq dst, dst, src, 1`: eight bytes into the high half of the low
    /// 128 bits.
    InsertHigh {
        dst: S::Tie<Vector>,
        src: S::Read<Integer>,
    },
    /// `vcmpps dst, a, b, pred`: a bit where it holds, clear where it does not.
    CmpK {
        pred: Pred,
        dst: S::Write<Opmask>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpmovm2d dst, k`: an all-ones lane where the bit is set, all-zero
    /// where it is not. AVX-512**DQ**, not F: an F-only part would fault.
    Movm2d {
        dst: S::Write<Vector>,
        k: S::Read<Opmask>,
    },
    /// `vptestmd dst, a, b`: a bit where the lanes' AND is nonzero.
    Ptestm {
        dst: S::Write<Opmask>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `kortestw k, k`: ZF iff no bit is set, CF iff all sixteen are.
    KorTest {
        flags: S::Write<Flags>,
        k: S::Read<Opmask>,
    },
    /// `kmovw dst, src`: the low sixteen bits of a general register.
    KFromGpr {
        dst: S::Write<Opmask>,
        src: S::Read<Integer>,
    },
    /// `vgatherdps dst{mask}, [base + index*4]`: one `f32` per lane whose
    /// `mask` bit is set. The instruction clears the bits as it completes
    /// lanes, and `#UD`s if `dst` and `index` are one register.
    Gather {
        dst: S::Early<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Vector>,
        mask: S::Tie<Opmask>,
    },
    /// `vmovups [dst]{mask}, src`: the lanes whose bit is set, the rest of
    /// memory left untouched.
    StoreMasked {
        dst: Mem<S, NoDisp>,
        src: S::Read<Vector>,
        mask: S::Read<Opmask>,
    },
    /// `vblendmps dst{k}, if_false, if_true`: `if_true` where the bit is set,
    /// `if_false` where it is not.
    BlendK {
        dst: S::Write<Vector>,
        k: S::Read<Opmask>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
    },
    /// `kandw`/`korw dst, a, b`
    KLogic {
        op: KOp,
        dst: S::Write<Opmask>,
        a: S::Read<Opmask>,
        b: S::Read<Opmask>,
    },
    /// `kxnorw dst, k0, k0`: all sixteen bits set. `k0` is never written, so
    /// the instruction waits on nothing, where `kxnorw k, k, k` would wait on
    /// the last gather still clearing `k`. `k0` is part of the idiom's
    /// encoding, not a value: plan H8 asks for a source nothing writes.
    KOnes { dst: S::Write<Opmask> },
    /// `kmovw dst, src`
    KMov {
        dst: S::Write<Opmask>,
        src: S::Read<Opmask>,
    },
    /// `kmovw [rsp + slot], src`: the allocator's spill.
    SpillMask { src: S::Read<Opmask>, slot: S::Slot },
    /// `kmovw dst, [rsp + slot]`: the allocator's reload.
    ReloadMask {
        dst: S::Write<Opmask>,
        slot: S::Slot,
    },
}

/// The bitwise operations on mask registers that [`Extra::KLogic`] spells.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum KOp {
    And = 0x41,
    Or = 0x45,
}

impl From<MaskOp> for KOp {
    /// The mask-register form of a bitwise op on masks
    /// ([`BinaryOp::on_masks`](crate::program::BinaryOp::on_masks)).
    fn from(op: MaskOp) -> Self {
        match op {
            MaskOp::And => Self::And,
            MaskOp::Or => Self::Or,
        }
    }
}

/// `kxnorw`'s opcode.
const KXNOR: u8 = 0x46;

/// The two `k0` fields of `kxnorw dst, k0, k0`: all-ones however `k0` was
/// left, because nothing writes it. A field the encoding fixes, not a
/// register: `k0` is in no file, and reaches the word as the zero this is.
#[derive(Copy, Clone)]
struct K0;

impl RegField for K0 {
    fn field(self) -> u8 {
        0
    }
}

/// `vpternlogd`'s truth table for all-ones.
const TERNLOG_ONES: u8 = 0xFF;

/// The bytes `0..8` and `8..16`, little end first: what two `movabs` carry
/// in for `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: [u64; 2] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];

// =============================================================================
// Selection
// =============================================================================

/// The AVX-512 machine, as the selection pipeline sees it.
pub(in crate::emit) enum Avx512 {}

/// One IR lane value on this machine. A comparison, and the `kandw`/`korw` of
/// two of them, is a bit per lane in the opmask file; everything else is in the
/// vector file. `ScheduledOp::is_mask` asks the same question before selection,
/// through the same `BinaryOp::on_masks`, and a test holds the two equal.
/// Where an instruction reads the other file's lane it is first widened
/// (`vpmovm2d`), or, when it has a form for the lane where it is (`vblendmps`,
/// `kortestw`), it reads it there.
#[derive(Copy, Clone)]
pub(super) enum Lane {
    Vector(Value<Vector>),
    Opmask(Value<Opmask>),
}

impl Lane {
    /// This lane as a mask, if it is held in the opmask file.
    fn opmask(self) -> Option<Value<Opmask>> {
        match self {
            Lane::Vector(_) => None,
            Lane::Opmask(k) => Some(k),
        }
    }

    /// For an instruction with no `k` form: this lane in the vector file, all
    /// ones where its bit is set.
    fn vector(self, b: &mut Builder<Avx512>) -> Value<Vector> {
        match self {
            Lane::Vector(vector) => vector,
            Lane::Opmask(k) => emit(b, |dst| Inst::Extra(Extra::Movm2d { dst, k })),
        }
    }
}

impl X86 for Avx512 {
    type Prefix = EvexPrefix;
    type Extra<S: Stage> = Extra<S>;
    type Held = Lane;

    // All thirty-two `zmm`. `k0` is not an opmask member: an EVEX `aaa` of 0
    // means "no mask".
    const VECTORS: &'static [u8] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ];
    const OPMASKS: &'static [u8] = &[1, 2, 3, 4, 5, 6, 7];
    const VECTOR_BYTES: u64 = 64;

    /// `vpternlogd dst, dst, dst, 0xFF`
    fn ones(dst: impl RegField) -> EncodedInst {
        Evex::m0f3a_66(0x25).imm(TERNLOG_ONES).rrr(dst, dst, dst)
    }

    /// All-ones shifted by the amount that leaves the sign bit, or all that is
    /// below it: two instructions and no read of the pool, where a loop would
    /// read it every trip.
    fn sign_mask(b: &mut Builder<Self>, sign: Sign) -> Result<Value<Vector>, CompileError> {
        let (direction, amount) = match sign {
            Sign::Negate => (Direction::Left, 31),
            Sign::Magnitude => (Direction::Right, 1),
        };
        let ones = emit(b, |dst| Inst::Ones { dst });
        Ok(emit(b, |dst| Inst::Shift {
            direction,
            dst,
            src: ones,
            amount,
        }))
    }

    fn rematerializable_extra(extra: &Extra<Selected>) -> bool {
        matches!(extra, Extra::KOnes { .. })
    }

    fn walk_extra<To: Stage>(extra: &Extra<Selected>, f: &mut impl Rebind<To>) -> Extra<To> {
        match extra {
            Extra::Blend {
                dst,
                if_true,
                if_false,
            } => Extra::Blend {
                dst: f.tie(dst),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Extra::InsertHigh { dst, src } => Extra::InsertHigh {
                dst: f.tie(dst),
                src: f.read(*src),
            },
            Extra::CmpK { pred, dst, a, b } => Extra::CmpK {
                pred: *pred,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Extra::Movm2d { dst, k } => Extra::Movm2d {
                dst: f.write(dst),
                k: f.read(*k),
            },
            Extra::Ptestm { dst, a, b } => Extra::Ptestm {
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Extra::KorTest { flags, k } => Extra::KorTest {
                flags: f.write(flags),
                k: f.read(*k),
            },
            Extra::KFromGpr { dst, src } => Extra::KFromGpr {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Extra::Gather {
                dst,
                base,
                index,
                mask,
            } => Extra::Gather {
                dst: f.early(dst),
                base: f.read(*base),
                index: f.read(*index),
                mask: f.tie(mask),
            },
            Extra::StoreMasked { dst, src, mask } => Extra::StoreMasked {
                dst: dst.walk(f),
                src: f.read(*src),
                mask: f.read(*mask),
            },
            Extra::BlendK {
                dst,
                k,
                if_true,
                if_false,
            } => Extra::BlendK {
                dst: f.write(dst),
                k: f.read(*k),
                if_true: f.read(*if_true),
                if_false: f.read(*if_false),
            },
            Extra::KLogic { op, dst, a, b } => Extra::KLogic {
                op: *op,
                dst: f.write(dst),
                a: f.read(*a),
                b: f.read(*b),
            },
            Extra::KOnes { dst } => Extra::KOnes { dst: f.write(dst) },
            Extra::KMov { dst, src } => Extra::KMov {
                dst: f.write(dst),
                src: f.read(*src),
            },
            Extra::SpillMask { src, slot } => Extra::SpillMask {
                src: f.read(*src),
                slot: f.slot(*slot),
            },
            Extra::ReloadMask { dst, slot } => Extra::ReloadMask {
                dst: f.write(dst),
                slot: f.slot(*slot),
            },
        }
    }

    fn encode_extra(extra: &Extra<Bound<'_, Self>>) -> EncodedInst {
        match extra {
            Extra::Blend {
                dst,
                if_true,
                if_false,
            } => Evex::m0f3a_66(0x25)
                .imm(TERNLOG_SELECT)
                .rrr(dst, if_true, if_false),
            Extra::InsertHigh { dst, src } => {
                Evex::m0f3a_66(0x22).w1().xmm().imm(1).rrr(dst, dst, src)
            }
            Extra::CmpK { pred, dst, a, b } => Evex::m0f(0xC2).imm(*pred as u8).rrr(dst, a, b),
            Extra::Movm2d { dst, k } => Evex::m0f38_f3(0x38).rr(dst, k),
            Extra::Ptestm { dst, a, b } => Evex::m0f38_66(0x27).rrr(dst, a, b),
            // Both operands the one mask register.
            Extra::KorTest { k, .. } => Vex::m0f(0x98).xmm().rr(k, k),
            Extra::KFromGpr { dst, src } => Vex::m0f(0x92).xmm().rr(dst, src),
            Extra::Gather {
                dst,
                base,
                index,
                mask,
            } => {
                debug_assert!(
                    dst.number() != index.number(),
                    "vgatherdps: dst and index must differ"
                );
                // `base` is never `rbp`/`r13` (`GENERAL` excludes them), so
                // the SIB's no-base encoding is unreachable.
                Evex::m0f38_66(0x92)
                    .masked(mask)
                    .vsib_scaled4(dst, base, index)
            }
            Extra::StoreMasked { dst, src, mask: k } => {
                Evex::m0f(0x11).masked(k).rm(src, &dst.base, dst.disp)
            }
            // `vvvv` is the lanes the mask leaves, and `rm` the ones it takes.
            Extra::BlendK {
                dst,
                k,
                if_true,
                if_false,
            } => Evex::m0f38_66(0x65).masked(k).rrr(dst, if_false, if_true),
            Extra::KLogic { op, dst, a, b } => Vex::m0f(*op as u8).rrr(dst, a, b),
            Extra::KOnes { dst } => Vex::m0f(KXNOR).rrr(dst, K0, K0),
            Extra::KMov { dst, src } => Vex::m0f(0x90).xmm().rr(dst, src),
            Extra::SpillMask { src, slot } => Vex::m0f(0x91).xmm().frame(src, *slot),
            Extra::ReloadMask { dst, slot } => Vex::m0f(0x90).xmm().frame(dst, *slot),
        }
    }

    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Lane, CompileError> {
        Ok(match op {
            LaneOp::Const(value) => Lane::Vector(x86_64::constant(b, value)?),
            // The bytes `0..16` in through two general registers, widened to
            // dwords, converted.
            LaneOp::Lanes => {
                let [low, high] = IOTA_BYTES;
                let low = x86_64::define(b, |dst| x86_64::Gp::Movabs { dst, imm: low });
                let packed = emit(b, |dst| Inst::Movq { dst, src: low });
                let high = x86_64::define(b, |dst| x86_64::Gp::Movabs { dst, imm: high });
                let joined = b.tie(packed);
                let halves = joined.write();
                b.push(Op::Vector(Inst::Extra(Extra::InsertHigh {
                    dst: joined,
                    src: high,
                })));
                Lane::Vector(x86_64::iota(b, halves))
            }
            LaneOp::Unary(op, src) => {
                let src = src.vector(b);
                Lane::Vector(x86_64::unary(b, op, src)?)
            }
            LaneOp::Binary(op, a, c) => match op.on_masks(a.opmask(), c.opmask()) {
                Some((op, x, y)) => Lane::Opmask(emit(b, |dst| {
                    Inst::Extra(Extra::KLogic {
                        op: op.into(),
                        dst,
                        a: x,
                        b: y,
                    })
                })),
                None => {
                    let (a, c) = (a.vector(b), c.vector(b));
                    Lane::Vector(x86_64::arithmetic(b, op, (a, c)))
                }
            },
            LaneOp::Compare(op, a, c) => {
                let (a, c) = (a.vector(b), c.vector(b));
                let pred = op.into();
                Lane::Opmask(emit(b, |dst| {
                    Inst::Extra(Extra::CmpK { pred, dst, a, b: c })
                }))
            }
            LaneOp::MulAdd(x, y, addend) => {
                let (x, y, addend) = (x.vector(b), y.vector(b), addend.vector(b));
                Lane::Vector(x86_64::mul_add(b, (x, y), addend))
            }
            // `cond ? if_true : if_false`: `vblendmps` under a mask register,
            // else `vpternlogd` bit by bit.
            LaneOp::Blend {
                cond,
                if_true,
                if_false,
            } => {
                let (if_true, if_false) = (if_true.vector(b), if_false.vector(b));
                Lane::Vector(match cond {
                    Lane::Opmask(k) => emit(b, |dst| {
                        Inst::Extra(Extra::BlendK {
                            dst,
                            k,
                            if_true,
                            if_false,
                        })
                    }),
                    Lane::Vector(mask) => {
                        let dst = b.tie(mask);
                        let blended = dst.write();
                        b.push(Op::Vector(Inst::Extra(Extra::Blend {
                            dst,
                            if_true,
                            if_false,
                        })));
                        blended
                    }
                })
            }
            LaneOp::Shift(op, src, amount) => {
                let src = src.vector(b);
                Lane::Vector(x86_64::shift(b, op, src, amount))
            }
            LaneOp::Uniform { base, element } => Lane::Vector(x86_64::uniform(b, base, element)?),
            // `vgatherdps` under an all-ones mask: the indices are floats, so
            // they are truncated first, and the instruction clears the mask
            // as it completes lanes.
            LaneOp::Gather { base, index } => {
                let index = index.vector(b);
                let indices = emit(b, |dst| Inst::Unary {
                    op: Lanewise::ToInt,
                    dst,
                    src: index,
                });
                let ones = emit(b, |dst| Inst::Extra(Extra::KOnes { dst }));
                let (dst, mask) = (b.early(), b.tie(ones));
                let gathered = dst.value();
                b.push(Op::Vector(Inst::Extra(Extra::Gather {
                    dst,
                    base,
                    index: indices,
                    mask,
                })));
                Lane::Vector(gathered)
            }
            LaneOp::Broadcast { base, index } => {
                let index = index.vector(b);
                Lane::Vector(x86_64::broadcast(b, base, index))
            }
        })
    }

    fn lane_name(lane: Lane) -> ValueName {
        match lane {
            Lane::Vector(vector) => vector.name(),
            Lane::Opmask(k) => k.name(),
        }
    }

    fn param_lane(param: ValueName) -> Lane {
        Lane::Vector(param.typed())
    }

    /// A full batch is one `vmovups`; a remainder is the same store under a
    /// writemask of its lanes.
    fn store(b: &mut Builder<Self>, store: Store<Self>) {
        let Store {
            out,
            pitch,
            row,
            col,
            value,
            lanes,
        } = store;
        let (row, col, value) = (row.vector(b), col.vector(b), value.vector(b));
        let base = x86_64::element_address(b, out, pitch, (row, col));
        let dst = Mem { base, disp: NoDisp };
        if u64::from(lanes) == Self::VECTOR_BYTES / 4 {
            b.push(Op::Vector(Inst::StoreBatch { dst, src: value }));
            return;
        }
        let bits = x86_64::define(b, |dst| x86_64::Gp::MovImm32 {
            dst,
            imm: (1u32 << lanes) - 1,
        });
        let mask = emit(b, |dst| Inst::Extra(Extra::KFromGpr { dst, src: bits }));
        b.push(Op::Vector(Inst::Extra(Extra::StoreMasked {
            dst,
            src: value,
            mask,
        })));
    }

    /// `kortestw` of the mask (`vptestmd` first when it is a vector), then
    /// `je` (ZF: no lane is set) or `jb` (CF: all sixteen are).
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges) {
        let k = match test.cond {
            Lane::Opmask(k) => k,
            Lane::Vector(mask) => emit(b, |dst| {
                Inst::Extra(Extra::Ptestm {
                    dst,
                    a: mask,
                    b: mask,
                })
            }),
        };
        let flags = b.def();
        let tested = flags.value();
        b.push(Op::Vector(Inst::Extra(Extra::KorTest { flags, k })));
        let cond = match test.dead {
            IfArm::True => x86_64::Cond::E,
            IfArm::False => x86_64::Cond::B,
        };
        x86_64::branch(b, cond, tested, edges);
    }

    fn copy_other(b: &mut Spiller<'_, Self>, src: ValueName) -> ValueName {
        match src.class {
            ClassId::Opmask => b.define::<Opmask>(|dst| {
                Op::Vector(Inst::Extra(Extra::KMov {
                    dst,
                    src: src.typed(),
                }))
            }),
            class => unreachable!("{class:?} is not copied"),
        }
    }

    fn spill_other(b: &mut Spiller<'_, Self>, src: ValueName, slot: FrameSlot) {
        match src.class {
            ClassId::Opmask => b.push(Op::Vector(Inst::Extra(Extra::SpillMask {
                src: src.typed(),
                slot: slot.name(),
            }))),
            class => unreachable!("{class:?} is not spilled"),
        }
    }

    fn reload_other(b: &mut Spiller<'_, Self>, class: ClassId, slot: FrameSlot) -> ValueName {
        match class {
            ClassId::Opmask => b.define::<Opmask>(|dst| {
                Op::Vector(Inst::Extra(Extra::ReloadMask {
                    dst,
                    slot: slot.name(),
                }))
            }),
            class => unreachable!("{class:?} is not reloaded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::IsaBackend;
    use crate::program::{BinaryOp, CompareOp, ScheduledOp, ShiftOp, UnaryOp, ValueId};

    /// Whether `selected` lands in the opmask file exactly when `scheduled`
    /// is a mask, value 0 of the schedule being the one mask operand.
    fn is_where_is_mask_says(
        b: &mut Builder<Avx512>,
        selected: LaneOp<Avx512>,
        scheduled: &ScheduledOp,
    ) -> bool {
        let lane = <Avx512 as X86>::lane(b, selected).expect("the op selects");
        let in_opmask = matches!(lane, Lane::Opmask(_));
        in_opmask == scheduled.is_mask(|value| value == ValueId(0))
    }

    /// Lowering refuses a fold's bound that is a mask (`ScheduledOp::is_mask`),
    /// which names no count, by asking before selection which values this tier
    /// will keep in the opmask file. The two answer one question: this
    /// selects every op that reads a lane over a mask and a number in every
    /// operand position (unary, shift, compare, binary, mul-add, the select,
    /// gather and broadcast), and asks the file each result lands in. A
    /// constant, the lane iota and a uniform read no lane, so they have no
    /// operand to make a mask.
    #[test]
    fn a_value_is_an_opmask_exactly_when_is_mask_says_it_is_a_mask() {
        let (mut b, entry) = Builder::<Avx512>::new();
        <Avx512 as IsaBackend>::enter(&mut b);
        let number = <Avx512 as X86>::lane(&mut b, LaneOp::Const(1.0)).expect("a constant selects");
        let compared = LaneOp::Compare(CompareOp::Lt, number, number);
        let mask = <Avx512 as X86>::lane(&mut b, compared).expect("a comparison selects");
        assert!(matches!(mask, Lane::Opmask(_)));

        // The schedule's view of the same two values.
        let operands = [(mask, ValueId(0)), (number, ValueId(1))];
        let mut every = |selected: LaneOp<Avx512>, scheduled: ScheduledOp| {
            assert!(
                is_where_is_mask_says(&mut b, selected, &scheduled),
                "{scheduled:?}"
            );
        };
        for &(x, x_id) in &operands {
            for op in UnaryOp::ALL {
                every(LaneOp::Unary(*op, x), ScheduledOp::Unary(*op, x_id));
            }
            for op in ShiftOp::ALL {
                every(
                    LaneOp::Shift(*op, x, 3),
                    ScheduledOp::ShiftImm(*op, x_id, 3),
                );
            }
            // The base is a pointer, the context's; the schedule names it by an
            // id no lane can be.
            let base = ValueId(2);
            every(
                LaneOp::Gather {
                    base: entry.ctx,
                    index: x,
                },
                ScheduledOp::Gather(x_id, base),
            );
            every(
                LaneOp::Broadcast {
                    base: entry.ctx,
                    index: x,
                },
                ScheduledOp::Broadcast(x_id, base),
            );
            for &(y, y_id) in &operands {
                for op in CompareOp::ALL {
                    every(
                        LaneOp::Compare(*op, x, y),
                        ScheduledOp::Compare(*op, x_id, y_id),
                    );
                }
                for op in BinaryOp::ALL {
                    every(
                        LaneOp::Binary(*op, x, y),
                        ScheduledOp::Binary(*op, x_id, y_id),
                    );
                }
                for &(z, z_id) in &operands {
                    every(
                        LaneOp::MulAdd(x, y, z),
                        ScheduledOp::MulAdd(x_id, y_id, z_id),
                    );
                    every(
                        LaneOp::Blend {
                            cond: x,
                            if_true: y,
                            if_false: z,
                        },
                        ScheduledOp::If(x_id, y_id, z_id),
                    );
                }
            }
        }

        // The agreement is not vacuous: a comparison and a bitwise op on two
        // masks are masks, and the same op on a number is not.
        let bitwise = |op| ScheduledOp::Binary(op, ValueId(0), ValueId(0));
        let on_a_number = |op| ScheduledOp::Binary(op, ValueId(1), ValueId(0));
        let is_mask = |scheduled: ScheduledOp| scheduled.is_mask(|value| value == ValueId(0));
        assert!(is_mask(ScheduledOp::Compare(
            CompareOp::Lt,
            ValueId(1),
            ValueId(1)
        )));
        for op in [BinaryOp::BitAnd, BinaryOp::BitOr] {
            assert!(is_mask(bitwise(op)), "{op:?} on two masks is a mask");
            assert!(!is_mask(on_a_number(op)), "{op:?} on a number is a number");
        }
    }
}
