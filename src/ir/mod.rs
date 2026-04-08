//! # Intermediate Representation (IR) Module
//!
//! This module defines the target-independent SSA-form intermediate representation
//! for the `bcc` C compiler. The IR bridges the typed AST from semantic analysis
//! to architecture-specific code generation.
//!
//! ## Architecture
//!
//! The IR is structured as a hierarchy:
//! - **[`Module`]**: Top-level container holding all functions and global variables
//!   for a single C translation unit.
//! - **[`Function`]**: A single function definition or declaration, containing a
//!   control flow graph of basic blocks with parameters and a return type.
//! - **[`BasicBlock`]**: A straight-line sequence of [`Instruction`]s ending with
//!   a [`Terminator`] that directs control flow to successor blocks.
//! - **[`Instruction`]**: An SSA operation (arithmetic, memory, comparison, call,
//!   phi node, cast, etc.) that consumes [`Value`] operands and optionally
//!   produces a result [`Value`].
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
//! The [`IrBuilder`] translates the fully type-checked AST into an IR [`Module`].
//! The optimization pass pipeline (`src/passes/`) transforms the module in-place,
//! and the code generation backends (`src/codegen/`) lower the optimized IR to
//! architecture-specific machine code.
//!
//! ## SSA Form
//!
//! The IR uses Static Single Assignment form where every [`Value`] is defined
//! exactly once. [`PhiNode`]s at control flow join points merge values from
//! different predecessor blocks. SSA construction is performed by
//! [`construct_ssa`], and SSA destruction (for register allocation) by
//! [`destruct_ssa`].
//!
//! ## Submodules
//!
//! | Submodule        | Contents                                                   |
//! |------------------|------------------------------------------------------------|
//! | [`types`]        | IR type system ([`IrType`], [`StructLayout`])              |
//! | [`instructions`] | Instruction set, values, constants, comparisons, casts     |
//! | [`builder`]      | AST-to-IR translation engine ([`IrBuilder`])               |
//! | [`cfg`]          | Control flow graph, dominance, terminators, loops          |
//! | [`ssa`]          | SSA construction and destruction algorithms                |

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// IR type system: [`IrType`] enum and [`StructLayout`] for target-parametric
/// size and alignment computation across x86-64, i686, AArch64, and RISC-V 64.
pub mod types;

/// IR instruction set: [`Instruction`] enum covering arithmetic, bitwise,
/// comparison, memory, function call, phi, cast, and miscellaneous operations.
/// Also defines [`Value`] (SSA reference), [`BlockId`] (block identifier),
/// [`Constant`] (compile-time values), comparison/cast operation enums, and
/// [`LocatedInstruction`] (instruction with source location for DWARF).
pub mod instructions;

/// IR builder: [`IrBuilder`] translating the typed AST into SSA-form IR.
/// Produces [`Module`] containing [`Function`]s and [`GlobalVariable`]s.
pub mod builder;

/// Control flow graph: [`BasicBlock`], [`ControlFlowGraph`], [`DominanceTree`],
/// [`Terminator`], [`PhiNode`], and [`Loop`] structures. Includes dominance
/// computation (Cooper-Harvey-Kennedy algorithm) and natural loop detection.
pub mod cfg;

/// SSA construction and destruction: [`construct_ssa`] (iterated dominance
/// frontier phi-node placement and variable renaming) and [`destruct_ssa`]
/// (phi elimination via copy insertion for code generation).
pub mod ssa;

// ---------------------------------------------------------------------------
// Public re-exports — the IR API surface
// ---------------------------------------------------------------------------
// All downstream consumers (passes, codegen, debug, driver) import from
// `crate::ir` rather than reaching into submodules directly.

// Type system
pub use types::{IrType, StructLayout};

// Instruction set and SSA value model
pub use instructions::{
    BlockId, Callee, CastOp, CompareOp, Constant, FloatCompareOp, Instruction, LocatedInstruction,
    Value,
};

// Builder and top-level IR containers
pub use builder::{Function, GlobalVariable, IrBuilder, Module};

// Control flow graph, dominance, terminators, phi nodes, and loops
pub use cfg::{BasicBlock, ControlFlowGraph, DominanceTree, Loop, PhiNode, Terminator};

// SSA construction and destruction entry points
pub use ssa::{construct_ssa, destruct_ssa};
