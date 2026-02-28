//! # Intermediate Representation (IR) Module
//!
//! This module defines the target-independent SSA-form intermediate representation
//! for the `bcc` C compiler. The IR bridges the typed AST from semantic analysis
//! to architecture-specific code generation.
//!
//! ## Architecture
//!
//! The IR is structured as:
//! - **Module**: Contains all functions and global variables for a translation unit
//! - **Function**: Contains a control flow graph of basic blocks
//! - **BasicBlock**: Contains a sequence of instructions and a terminator
//! - **Instruction**: SSA operations (arithmetic, memory, control flow, etc.)
//!
//! ## Data Flow
//!
//! ```text
//! Typed AST (from sema) → IrBuilder → Module (SSA IR)
//!                                        ↓
//!                                   Optimization Passes
//!                                        ↓
//!                                   Code Generation
//! ```
//!
//! ## SSA Form
//!
//! The IR uses Static Single Assignment form where every value is defined
//! exactly once. Phi nodes at control flow join points merge values from
//! different predecessor blocks.

pub mod types;
pub mod instructions;

// Re-export key types for convenient access by downstream modules.
pub use types::{IrType, StructLayout};
pub use instructions::{
    Instruction, Value, Constant, CompareOp, FloatCompareOp,
    CastOp, Callee, LocatedInstruction, BlockId,
};
