//! Common types and utilities shared across all compiler phases.
//!
//! The `common` module forms the foundation layer of the `bcc` compiler. It has
//! zero internal dependencies (imports only from `std`) and is consumed by every
//! other module in the compiler: `driver`, `frontend`, `sema`, `ir`, `passes`,
//! `codegen`, `linker`, and `debug`.
//!
//! # Submodules
//!
//! | Submodule       | Purpose |
//! |-----------------|---------|
//! | [`diagnostics`] | GCC-compatible error/warning/note reporting to stderr |
//! | [`source_map`]  | Source file registry and byte-offset-to-line/column mapping |
//! | [`intern`]      | String interning for identifiers and literals |
//! | [`arena`]       | Chunk-based arena allocator for AST and IR nodes |
//! | [`numeric`]     | Arbitrary-width integer type for compile-time constant evaluation |
//!
//! # Re-Exports
//!
//! The most frequently used types from each submodule are re-exported at the
//! `common` module level for convenient access. Consumers can write:
//!
//! ```ignore
//! use crate::common::{SourceLocation, DiagnosticEmitter, Interner, Arena, BigInt};
//! ```
//!
//! instead of fully qualifying each submodule path.
//!
//! # Design Principles
//!
//! - **Zero external dependencies** — Uses only the Rust standard library (`std`).
//! - **No `unsafe` code** — This module file contains purely declarations and
//!   re-exports. Individual submodules (notably `arena`) document their `unsafe`
//!   usage per the project's safety comment policy.
//! - **Foundation layer** — No circular dependencies; `common` never imports
//!   from any other `bcc` module.
//! - **Performance-critical** — String interning and arena allocation are
//!   essential for meeting the SQLite compilation constraint (<60 seconds,
//!   <2 GB RSS for ~230K LOC at `-O0`).

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// GCC-compatible diagnostic reporting: error, warning, and note messages
/// formatted as `file:line:col: severity: message` on stderr.
pub mod diagnostics;

/// Source file registry and position tracking: maps byte offsets to
/// file/line/column positions, tracks macro expansion chains, and handles
/// `#line` directive overrides.
pub mod source_map;

/// String interning for identifiers and string literals: deduplicates
/// strings at lexing time and returns compact [`InternId`] handles for
/// O(1) equality comparison throughout the compiler pipeline.
pub mod intern;

/// Chunk-based arena allocator providing O(1) bump allocation and batch
/// deallocation for AST nodes, IR nodes, and interned string storage.
pub mod arena;

/// Arbitrary-width integer representation (`BigInt`) for compile-time
/// constant evaluation in preprocessor `#if` directives and the constant
/// folding optimization pass.
pub mod numeric;

// ---------------------------------------------------------------------------
// Public type re-exports
// ---------------------------------------------------------------------------

// From diagnostics: the central error/warning/note emitter and its data types.
pub use diagnostics::{Diagnostic, DiagnosticEmitter, Severity};

// From source_map: position tracking types embedded in every token and AST node.
pub use source_map::{FileId, SourceLocation, SourceMap, SourceSpan};

// From intern: string deduplication handles used by lexer, parser, and sema.
pub use intern::{InternId, Interner};

// From arena: the arena allocator used by parser and IR builder.
pub use arena::Arena;

// From numeric: compile-time constant evaluation integer type.
pub use numeric::BigInt;
