// Common module — shared types and utilities for all compiler phases.
// Declares submodules for arena, diagnostics, intern, numeric, and source_map.

pub mod arena;
pub mod diagnostics;
pub mod intern;
pub mod numeric;
pub mod source_map;

pub use diagnostics::{DiagnosticEmitter, Diagnostic, Severity};
pub use intern::{InternId, Interner};
pub use numeric::BigInt;
pub use source_map::{FileId, SourceLocation, SourceMap, SourceSpan};
