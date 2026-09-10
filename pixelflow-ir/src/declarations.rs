//! Buffer, uniform, and variable-slot declarations used by expression graphs.

pub use crate::arena::{
    BufferDecl, BufferId, BufferIdentity, COORD_AXES, UniformDecl, UniformId, UniformIdentity,
};
pub(crate) use crate::arena::{REDUCE_BINDER_BASE, REDUCE_BINDERS, RETIRED_COORD_AXES};
