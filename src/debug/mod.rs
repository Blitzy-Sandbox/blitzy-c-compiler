// Debug information generation module for bcc compiler.
//
// This module provides DWARF v4 debug information generation across
// all four target architectures. It is conditionally invoked when
// the `-g` flag is specified.
//
// Submodules:
// - `dwarf`: Core DWARF v4 structures, LEB128 encoding, abbreviation
//            tables, string tables, and section builder.
// - `info`: .debug_info section DIE generation (pending).
// - `line_program`: .debug_line section line number program (pending).
// - `frame`: .debug_frame section CFI generation (pending).

pub mod dwarf;
pub mod frame;
pub mod info;
pub mod line_program;
