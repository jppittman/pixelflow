//! Test that kernel! macro compiles with 5 parameters using WithContext.
//!
//! This is the key test - the old nested Let approach failed with >4 params.

use pixelflow_core::{Field, Manifold};
use pixelflow_compiler::kernel;

type Field4 = (Field, Field, Field, Field);










