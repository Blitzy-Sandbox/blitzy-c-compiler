// Common module — shared types and utilities for all compiler phases.
// This stub declares only the arena, numeric, and source_map submodules for now.
// Other submodules (diagnostics, intern) will be declared by their respective
// assigned agents.

pub mod arena;
pub mod numeric;
pub mod source_map;

pub use numeric::BigInt;
pub use source_map::{FileId, SourceLocation, SourceMap, SourceSpan};
