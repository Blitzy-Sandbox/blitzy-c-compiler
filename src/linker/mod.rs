//! Integrated ELF Linker
//!
//! This module implements a complete ELF linker that reads relocatable objects
//! and `ar` static archives for system CRT and library linkage, without
//! reliance on external `ld`, `lld`, or `gold`.
//!
//! Stub module declaration for section validation. The full implementation
//! will be created by the linker/mod.rs agent.

pub mod archive;
pub mod elf;
pub mod sections;
pub mod symbols;
