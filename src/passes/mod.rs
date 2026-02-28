//! # Optimization Passes Module
//!
//! This module implements the optimization pipeline for the `bcc` C compiler,
//! supporting `-O0` through `-O2` optimization levels. Each pass is independently
//! testable and operates on the SSA-form IR defined in `crate::ir`.
//!
//! ## Pass Architecture
//!
//! All optimization passes implement the `FunctionPass` trait, which defines:
//! - A `name()` method returning a human-readable pass name
//! - A `run_on_function()` method that transforms the IR in place and returns
//!   whether any changes were made (used for fixed-point iteration at -O2)

use crate::ir::Function;

// Submodule declarations.
pub mod mem2reg;

/// Trait for optimization passes that operate on individual functions.
///
/// Each pass receives a mutable reference to a `Function` and transforms
/// its IR in place. The `run_on_function` method returns `true` if any
/// changes were made, enabling fixed-point iteration at `-O2`.
pub trait FunctionPass {
    /// Returns a human-readable name for this pass (used in diagnostics and debugging).
    fn name(&self) -> &str;

    /// Runs this optimization pass on the given function.
    ///
    /// # Returns
    /// `true` if the IR was modified by this pass, `false` if no changes were made.
    fn run_on_function(&mut self, function: &mut Function) -> bool;
}

// Re-export the mem2reg pass type.
pub use mem2reg::Mem2RegPass;
