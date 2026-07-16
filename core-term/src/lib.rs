#![allow(clippy::type_complexity)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::too_many_arguments)]
//! Core-term library crate.
//!
//! This exposes the internal modules for testing and library usage.

pub mod ansi;
pub mod color;
pub mod config;
pub mod glyph;
pub mod io;
pub mod keys;
pub mod messages;
pub mod surface;
pub mod term;
pub mod terminal_app;
