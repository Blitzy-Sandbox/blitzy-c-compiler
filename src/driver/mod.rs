//! Driver module: CLI argument parsing, target configuration, and compilation pipeline orchestration.
//!
//! This module coordinates the entire compilation process:
//! 1. `cli` — Parses GCC-compatible command-line flags
//! 2. `target` — Resolves target triple to architecture-specific configuration
//! 3. `pipeline` — Sequences all compiler phases: preprocessor → lexer → parser → sema → IR → optimizer → codegen → linker
//!
//! Entry point: `parse_args()` → `resolve_target()` → `pipeline::run()`

pub mod cli;
pub mod target;

// Re-export key types from cli for convenient access by main.rs and other consumers.
pub use cli::{
    derive_output_path, parse_args, parse_args_from, CliArgs, MacroDefinition, OptLevel,
};

// Re-export key types from target for convenient access by main.rs and other consumers.
pub use target::{
    detect_host, parse_target, resolve_target, AbiVariant, Architecture, ElfClass, Endianness,
    TargetConfig,
};
