// Common module — shared types and utilities for all compiler phases.
// Declares submodules for arena, numeric, source_map, and intern.
// Other submodules (diagnostics) will be declared by their respective
// assigned agents.

pub mod arena;
pub mod intern;
pub mod numeric;
pub mod source_map;

pub use intern::{InternId, Interner};
pub use numeric::BigInt;
pub use source_map::{FileId, SourceLocation, SourceMap, SourceSpan};
