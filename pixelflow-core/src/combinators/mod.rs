//! # Combinators Module
//!
//! Control flow and structural combinators.

pub mod at;
pub mod binding;
pub mod block;
pub mod computed;
pub mod context; // Flat context tuple prototype
pub mod fix;
pub mod map;
pub mod pack;
pub mod project;
pub mod select;
pub mod spherical;
pub mod texture;
pub mod with_gradient;

pub use at::{At, AtArray};
pub use binding::{
    B0, B1, Empty, GAdd, GDiv, GMul, GSub, Get, Graph, Let, Lift, N0, N1, N2, N3, N4, N5, N6, N7,
    Root, UInt, UTerm, Var,
};
pub use block::Block;
pub use computed::{Computed, ManifoldBind};
pub use context::{
    A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, ContextFree, CtxVar,
    WithContext,
}; // Array-based context
pub use fix::{Fix, RecDomain, RecFix, Recurse};
pub use map::{ClosureMap, Map};
pub use pack::Pack;
pub use project::Project;
pub use select::Select;
pub use spherical::{
    SH_NORM, Sh1, Sh2, Sh3, ShCoeffs, ShProject, ShReconstruct, SphericalHarmonic, ZonalHarmonic,
};
pub use texture::Texture;
pub use with_gradient::{WithGradient, WithGradient2D, WithGradient3D};
