//! # Code Generation Module
//!
//! This module implements native machine code generation for four architectures:
//! x86-64, i686, AArch64, and RISC-V 64. Each backend provides an integrated
//! assembler that directly encodes machine instructions without reliance on
//! external `as` or `gas`.
//!
//! ## Submodules
//!
//! - `regalloc` — Shared linear scan register allocator

pub mod regalloc;
