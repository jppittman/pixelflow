//! What a backend's register files are, and the one way to declare them.
//!
//! A leaf module: [`RegisterFile`]'s fields are private to it and its one
//! constructor checks, so a declaration that contradicts itself does not
//! exist for `Pool::mint` or `Leases::new` to trust.
#![expect(dead_code, reason = "live from B4")]

use super::FileId;

/// The allocatable members of each file, by hardware number.
#[derive(Copy, Clone, Debug)]
pub(super) struct Members {
    pub(super) vector: &'static [u8],
    pub(super) general: &'static [u8],
    pub(super) opmask: &'static [u8],
    pub(super) flags: &'static [u8],
}

/// Where the ABI puts the collapse's three arguments.
#[derive(Copy, Clone, Debug)]
pub(super) struct EntryRegisters {
    pub(super) ctx: u8,
    pub(super) out: u8,
    pub(super) pitch: u8,
}

/// What a backend's register files *are*: the allocatable members of each, by
/// hardware number, and where the ABI puts the three arguments. Numbers only;
/// nothing here is a register until `Pool::mint`. It says nothing about what an
/// instruction needs: selection runs first, so the allocator reads that off
/// the function.
///
/// A register outside every list belongs to the platform or the caller:
/// callee-saved registers, the stack pointer (the frame's), `x30`, and Apple's
/// `x18`. None of them is a scratch reservation.
///
/// The calling convention is SysV on x86-64 and AAPCS64 on aarch64, because
/// `executable.rs` builds only for Linux and macOS.
#[derive(Copy, Clone, Debug)]
pub(super) struct RegisterFile {
    members: Members,
    /// The members of `general` the three arguments arrive in. This is initial
    /// ownership, not a reservation: once a parameter is dead or spilled its
    /// register is free.
    entry: EntryRegisters,
    /// Bytes per `Vector` register and per vector frame slot: 16, 32 or 64.
    vector_bytes: u64,
}

const fn contains(members: &[u8], number: u8) -> bool {
    let mut i = 0;
    while i < members.len() {
        if members[i] == number {
            return true;
        }
        i += 1;
    }
    false
}

const fn distinct(members: &[u8]) -> bool {
    let mut i = 0;
    while i < members.len() {
        let (_, rest) = members.split_at(i + 1);
        if contains(rest, members[i]) {
            return false;
        }
        i += 1;
    }
    true
}

impl RegisterFile {
    /// The only way to make one, refusing a self-contradictory declaration at
    /// compile time (it is `const`, and a backend declares `FILE` with it): a
    /// file that names a register twice, two flags registers, an entry
    /// register outside `general` or shared by two arguments, a vector
    /// narrower than 16 bytes or not a power of two.
    pub(super) const fn new(members: Members, entry: EntryRegisters, vector_bytes: u64) -> Self {
        assert!(
            distinct(members.vector)
                && distinct(members.general)
                && distinct(members.opmask)
                && distinct(members.flags),
            "a register file names a member twice"
        );
        assert!(members.flags.len() <= 1, "there is one flags register");
        let EntryRegisters { ctx, out, pitch } = entry;
        assert!(
            contains(members.general, ctx)
                && contains(members.general, out)
                && contains(members.general, pitch),
            "an entry argument arrives in a register outside the general file"
        );
        assert!(
            ctx != out && ctx != pitch && out != pitch,
            "two entry arguments arrive in one register"
        );
        assert!(
            vector_bytes >= 16 && vector_bytes.is_power_of_two(),
            "a vector is a power of two bytes, at least 16"
        );
        Self {
            members,
            entry,
            vector_bytes,
        }
    }

    /// The numbers of the members of `file`.
    pub(super) fn members(&self, file: FileId) -> &'static [u8] {
        match file {
            FileId::Vector => self.members.vector,
            FileId::General => self.members.general,
            FileId::Opmask => self.members.opmask,
            FileId::Flags => self.members.flags,
        }
    }

    pub(super) fn entry(&self) -> EntryRegisters {
        self.entry
    }

    pub(super) fn vector_bytes(&self) -> u64 {
        self.vector_bytes
    }
}
