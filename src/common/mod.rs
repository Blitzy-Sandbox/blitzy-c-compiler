// Common module — shared types and utilities for all compiler phases.
// This stub declares only the arena and numeric submodules for now. Other
// submodules (diagnostics, source_map, intern) will be declared by their
// respective assigned agents.

pub mod arena;
pub mod numeric;

pub use numeric::BigInt;
