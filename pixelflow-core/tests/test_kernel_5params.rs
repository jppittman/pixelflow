//! Test that kernel! macro compiles with 5 parameters using WithContext.
//!
//! This is the key test - the old nested Let approach failed with >4 params.

use pixelflow_compiler::kernel;
