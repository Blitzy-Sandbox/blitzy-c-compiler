//! # Relocation Processing for Four Architectures
//!
//! This module applies ELF relocations for all four target architectures:
//! x86-64, i686, AArch64, and RISC-V 64. After section merging assigns final
//! virtual addresses, this module patches machine code bytes in the output
//! sections by computing and applying relocation fixups.
//!
//! ## Relocation Variables
//!
//! The standard ELF relocation formulas use these variables:
//! - **S**: Symbol value (final virtual address of the referenced symbol)
//! - **A**: Addend (explicit addend from RELA entry)
//! - **P**: Place (address where the relocation is applied)
//! - **G**: GOT entry offset for the symbol
//! - **GOT**: Base address of the Global Offset Table
//! - **L**: PLT entry address for the symbol
//! - **B**: Base address of the shared object (for PIC)
//!
//! ## Zero External Dependencies
//!
//! This module uses only the Rust standard library and sibling linker modules.
//! No external crates are imported.

use std::fmt;

use super::symbols::ResolvedSymbol;
use crate::codegen::Architecture;

// ===========================================================================
// x86-64 Relocation Type Constants (ELF AMD64 ABI)
// ===========================================================================

/// No relocation.
pub const R_X86_64_NONE: u32 = 0;
/// S + A — absolute 64-bit address.
pub const R_X86_64_64: u32 = 1;
/// S + A - P — PC-relative 32-bit signed.
pub const R_X86_64_PC32: u32 = 2;
/// G + A — 32-bit GOT entry offset.
pub const R_X86_64_GOT32: u32 = 3;
/// L + A - P — 32-bit PLT-relative.
pub const R_X86_64_PLT32: u32 = 4;
/// S — GOT slot (absolute, for dynamic linking).
pub const R_X86_64_GLOB_DAT: u32 = 6;
/// S — PLT/GOT slot (absolute, for dynamic linking).
pub const R_X86_64_JUMP_SLOT: u32 = 7;
/// B + A — base-relative (for PIC/shared objects).
pub const R_X86_64_RELATIVE: u32 = 8;
/// G + GOT + A - P — GOT PC-relative 32-bit.
pub const R_X86_64_GOTPCREL: u32 = 9;
/// S + A — absolute 32-bit (zero-extended), overflow-checked.
pub const R_X86_64_32: u32 = 10;
/// S + A — absolute 32-bit (sign-extended), overflow-checked.
pub const R_X86_64_32S: u32 = 11;
/// S + A — absolute 16-bit.
pub const R_X86_64_16: u32 = 12;
/// S + A - P — PC-relative 16-bit.
pub const R_X86_64_PC16: u32 = 13;
/// S + A - P — PC-relative 64-bit.
pub const R_X86_64_PC64: u32 = 24;

// ===========================================================================
// i686 Relocation Type Constants (ELF i386 ABI)
// ===========================================================================

/// No relocation.
pub const R_386_NONE: u32 = 0;
/// S + A — absolute 32-bit.
pub const R_386_32: u32 = 1;
/// S + A - P — PC-relative 32-bit.
pub const R_386_PC32: u32 = 2;
/// G + A — 32-bit GOT entry offset.
pub const R_386_GOT32: u32 = 3;
/// L + A - P — 32-bit PLT-relative.
pub const R_386_PLT32: u32 = 4;
/// S — GOT slot (absolute).
pub const R_386_GLOB_DAT: u32 = 6;
/// S — jump slot (PLT).
pub const R_386_JMP_SLOT: u32 = 7;
/// B + A — base-relative.
pub const R_386_RELATIVE: u32 = 8;
/// S + A - GOT — offset from GOT base.
pub const R_386_GOTOFF: u32 = 9;
/// GOT + A - P — PC-relative GOT address.
pub const R_386_GOTPC: u32 = 10;

// ===========================================================================
// AArch64 Relocation Type Constants (ELF ARM64 ABI)
// ===========================================================================

/// No relocation.
pub const R_AARCH64_NONE: u32 = 0;
/// S + A — absolute 64-bit data.
pub const R_AARCH64_ABS64: u32 = 257;
/// S + A — absolute 32-bit data, overflow-checked.
pub const R_AARCH64_ABS32: u32 = 258;
/// S + A — absolute 16-bit data, overflow-checked.
pub const R_AARCH64_ABS16: u32 = 259;
/// S + A - P — PC-relative 64-bit data.
pub const R_AARCH64_PREL64: u32 = 260;
/// S + A - P — PC-relative 32-bit data, overflow-checked.
pub const R_AARCH64_PREL32: u32 = 261;
/// Page(S+A) - Page(P) — 21-bit page-relative for ADRP instruction.
pub const R_AARCH64_ADR_PREL_PG_HI21: u32 = 275;
/// (S + A) & 0xFFF — 12-bit page offset for ADD immediate (no check).
pub const R_AARCH64_ADD_ABS_LO12_NC: u32 = 277;
/// S + A - P — 26-bit branch offset for B instruction.
pub const R_AARCH64_JUMP26: u32 = 282;
/// S + A - P — 26-bit branch offset for BL instruction.
pub const R_AARCH64_CALL26: u32 = 283;
/// ((S+A) & 0xFFF) >> 3 — 12-bit scaled offset for 64-bit LD/ST (no check).
pub const R_AARCH64_LDST64_ABS_LO12_NC: u32 = 286;
/// S — GOT slot (absolute).
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
/// S — jump slot (PLT).
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
/// B + A — base-relative (PIC).
pub const R_AARCH64_RELATIVE: u32 = 1027;

// ===========================================================================
// RISC-V 64 Relocation Type Constants (ELF RISC-V ABI)
// ===========================================================================

/// No relocation.
pub const R_RISCV_NONE: u32 = 0;
/// S + A — absolute 32-bit.
pub const R_RISCV_32: u32 = 1;
/// S + A — absolute 64-bit.
pub const R_RISCV_64: u32 = 2;
/// B + A — base-relative.
pub const R_RISCV_RELATIVE: u32 = 3;
/// S — jump slot (PLT).
pub const R_RISCV_JUMP_SLOT: u32 = 5;
/// S + A - P — B-type branch (12-bit signed, scattered encoding).
pub const R_RISCV_BRANCH: u32 = 16;
/// S + A - P — J-type jump (20-bit signed, scattered encoding).
pub const R_RISCV_JAL: u32 = 17;
/// S + A - P — AUIPC+JALR pair (32-bit split hi20/lo12).
pub const R_RISCV_CALL: u32 = 18;
/// S + A - P — upper 20 bits for AUIPC (PC-relative).
pub const R_RISCV_PCREL_HI20: u32 = 23;
/// Lower 12 bits, I-type format (complements PCREL_HI20).
pub const R_RISCV_PCREL_LO12_I: u32 = 24;
/// Lower 12 bits, S-type format (complements PCREL_HI20).
pub const R_RISCV_PCREL_LO12_S: u32 = 25;
/// S + A — upper 20 bits for LUI (absolute).
pub const R_RISCV_HI20: u32 = 26;
/// S + A — lower 12 bits, I-type format (absolute).
pub const R_RISCV_LO12_I: u32 = 27;
/// S + A — lower 12 bits, S-type format (absolute).
pub const R_RISCV_LO12_S: u32 = 28;
/// Linker relaxation hint — no patch, skip.
pub const R_RISCV_RELAX: u32 = 51;

// ===========================================================================
// RelocationEntry — input to the relocation processor
// ===========================================================================

/// A single relocation entry describing one fixup to apply.
///
/// Corresponds to ELF `Elf64_Rela` / `Elf32_Rela` structures, abstracted
/// to a target-independent representation.
#[derive(Debug, Clone)]
pub struct RelocationEntry {
    /// Byte offset within the section where the fixup is applied.
    pub offset: u64,
    /// Architecture-specific relocation type constant (e.g., `R_X86_64_PC32`).
    pub reloc_type: u32,
    /// Index into the symbol table identifying the referenced symbol.
    pub symbol_index: u32,
    /// Explicit addend value (from RELA relocations).
    pub addend: i64,
    /// Index of the section this relocation patches (into `section_data`).
    pub section_index: usize,
}

// ===========================================================================
// RelocationContext — state needed during relocation application
// ===========================================================================

/// Provides all context needed to compute relocation fixup values.
///
/// Created by the linker after section merging and symbol resolution, then
/// passed to [`apply_relocations`] to patch the output section data.
pub struct RelocationContext<'a> {
    /// Resolved symbol table with final virtual addresses.
    /// Indexed by `RelocationEntry::symbol_index`.
    pub symbols: &'a [ResolvedSymbol],
    /// Base virtual address for each merged output section.
    /// Indexed by `RelocationEntry::section_index`.
    pub section_addresses: &'a [u64],
    /// Base address of the Global Offset Table (.got section).
    pub got_address: u64,
    /// Base address of the Procedure Linkage Table (.plt section).
    pub plt_address: u64,
    /// Target architecture determining which relocation handlers to use.
    pub arch: Architecture,
    /// Whether position-independent code is being generated (`-fPIC`).
    pub is_pic: bool,
}

// ===========================================================================
// RelocationError — error type for relocation failures
// ===========================================================================

/// Errors that can occur during relocation processing.
///
/// Each variant carries enough context for the linker to produce a
/// GCC-compatible diagnostic message identifying the specific failure.
#[derive(Debug)]
pub enum RelocationError {
    /// The relocation type is not supported for the given architecture.
    /// Contains `(reloc_type, architecture_name)`.
    UnsupportedType(u32, &'static str),
    /// The symbol index is out of bounds in the resolved symbol table.
    /// Contains the invalid symbol index.
    SymbolNotFound(u32),
    /// The computed relocation value overflows the target bit width.
    /// Contains the relocation type, computed value, and maximum bits.
    OverflowError {
        reloc_type: u32,
        value: i64,
        max_bits: u32,
    },
    /// The section index is out of bounds.
    SectionOutOfBounds(usize),
    /// The byte offset within a section is out of bounds for the write size.
    OffsetOutOfBounds {
        section: usize,
        offset: u64,
        size: usize,
    },
}

impl fmt::Display for RelocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelocationError::UnsupportedType(rtype, arch) => {
                write!(
                    f,
                    "unsupported relocation type {} for architecture {}",
                    rtype, arch
                )
            }
            RelocationError::SymbolNotFound(idx) => {
                write!(f, "symbol index {} not found in resolved symbol table", idx)
            }
            RelocationError::OverflowError {
                reloc_type,
                value,
                max_bits,
            } => {
                write!(
                    f,
                    "relocation overflow: type {} value {:#x} exceeds {}-bit range",
                    reloc_type, value, max_bits
                )
            }
            RelocationError::SectionOutOfBounds(idx) => {
                write!(f, "section index {} out of bounds", idx)
            }
            RelocationError::OffsetOutOfBounds {
                section,
                offset,
                size,
            } => {
                write!(
                    f,
                    "offset {:#x} + {} bytes exceeds section {} data length",
                    offset, size, section
                )
            }
        }
    }
}

impl std::error::Error for RelocationError {}

// ===========================================================================
// Public API — overflow checking
// ===========================================================================

/// Check whether a computed relocation value fits in the specified bit width.
///
/// For signed relocations, verifies `-2^(bits-1) <= value < 2^(bits-1)`.
/// For unsigned relocations, verifies `0 <= value < 2^bits`.
///
/// Returns `Ok(())` if the value fits, or `Err(RelocationError::OverflowError)`
/// with the relocation type, value, and bit width if it does not.
pub fn check_overflow(
    value: i64,
    bits: u32,
    signed: bool,
    reloc_type: u32,
) -> Result<(), RelocationError> {
    if bits >= 64 {
        return Ok(());
    }
    if signed {
        let min = -(1i64 << (bits - 1));
        let max = (1i64 << (bits - 1)) - 1;
        if value < min || value > max {
            return Err(RelocationError::OverflowError {
                reloc_type,
                value,
                max_bits: bits,
            });
        }
    } else {
        // For unsigned, the value (interpreted as i64) must be in [0, 2^bits).
        if value < 0 || value >= (1i64 << bits) {
            return Err(RelocationError::OverflowError {
                reloc_type,
                value,
                max_bits: bits,
            });
        }
    }
    Ok(())
}

// ===========================================================================
// Public API — apply all relocations
// ===========================================================================

/// Apply all relocations to the output section data.
///
/// This is the primary entry point for relocation processing. It is called
/// after section merging has assigned final addresses to all sections and
/// symbols. For each relocation entry, it computes the fixup value using
/// the architecture-specific formula and patches the machine code bytes
/// in the corresponding output section buffer.
///
/// # Arguments
///
/// * `section_data` — Mutable slice of section byte vectors (one per merged
///   output section). Each `Vec<u8>` is the raw machine code/data for that
///   section, indexed by `RelocationEntry::section_index`.
/// * `relocations` — All relocation entries to apply.
/// * `ctx` — The relocation context providing symbol addresses, section
///   addresses, GOT/PLT addresses, and the target architecture.
///
/// # Errors
///
/// Returns the first error encountered. Processing stops on the first
/// failure to ensure corrupted output is never produced silently.
pub fn apply_relocations(
    section_data: &mut [Vec<u8>],
    relocations: &[RelocationEntry],
    ctx: &RelocationContext,
) -> Result<(), RelocationError> {
    for reloc in relocations {
        apply_single_relocation(section_data, reloc, ctx)?;
    }
    Ok(())
}

// ===========================================================================
// Internal — dispatch to architecture-specific handler
// ===========================================================================

/// Dispatch a single relocation to the correct architecture handler.
fn apply_single_relocation(
    section_data: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    ctx: &RelocationContext,
) -> Result<(), RelocationError> {
    match ctx.arch {
        Architecture::X86_64 => apply_x86_64_relocation(section_data, reloc, ctx),
        Architecture::I686 => apply_i686_relocation(section_data, reloc, ctx),
        Architecture::Aarch64 => apply_aarch64_relocation(section_data, reloc, ctx),
        Architecture::Riscv64 => apply_riscv64_relocation(section_data, reloc, ctx),
    }
}

// ===========================================================================
// Internal — resolve symbol value
// ===========================================================================

/// Look up the final virtual address for a symbol by its index.
///
/// Returns the `address` field of the resolved symbol, which is the 'S'
/// variable used in relocation formulas.
fn symbol_value(symbol_index: u32, ctx: &RelocationContext) -> Result<u64, RelocationError> {
    let idx = symbol_index as usize;
    if idx >= ctx.symbols.len() {
        return Err(RelocationError::SymbolNotFound(symbol_index));
    }
    Ok(ctx.symbols[idx].address)
}

/// Compute the Place address (P) for a relocation: the virtual address
/// where the fixup is applied.
fn place_address(reloc: &RelocationEntry, ctx: &RelocationContext) -> Result<u64, RelocationError> {
    if reloc.section_index >= ctx.section_addresses.len() {
        return Err(RelocationError::SectionOutOfBounds(reloc.section_index));
    }
    Ok(ctx.section_addresses[reloc.section_index] + reloc.offset)
}

// ===========================================================================
// x86-64 relocation handler
// ===========================================================================

/// Apply a single x86-64 relocation.
fn apply_x86_64_relocation(
    section_data: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    ctx: &RelocationContext,
) -> Result<(), RelocationError> {
    let s = symbol_value(reloc.symbol_index, ctx)?;
    let a = reloc.addend;
    let p = place_address(reloc, ctx)?;

    match reloc.reloc_type {
        R_X86_64_NONE => { /* No-op */ }

        R_X86_64_64 => {
            // S + A (absolute 64-bit)
            let value = (s as i64).wrapping_add(a) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        R_X86_64_PC32 | R_X86_64_PLT32 => {
            // S + A - P (PC-relative 32-bit, signed)
            let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
            check_overflow(value, 32, true, reloc.reloc_type)?;
            write_i32_le(section_data, reloc, value as i32)?;
        }

        R_X86_64_GOT32 => {
            // G + A (GOT entry offset, 32-bit)
            let value = (ctx.got_address as i64).wrapping_add(a);
            write_u32_le(section_data, reloc, value as u32)?;
        }

        R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => {
            // S (absolute symbol address, 64-bit, for dynamic linking)
            write_u64_le(section_data, reloc, s)?;
        }

        R_X86_64_RELATIVE => {
            // B + A (base-relative; B = 0 for static executables)
            let base: u64 = 0;
            let value = (base as i64).wrapping_add(a) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        R_X86_64_GOTPCREL => {
            // G + GOT + A - P (GOT PC-relative, 32-bit signed)
            let value = (ctx.got_address as i64).wrapping_add(a).wrapping_sub(p as i64);
            check_overflow(value, 32, true, reloc.reloc_type)?;
            write_i32_le(section_data, reloc, value as i32)?;
        }

        R_X86_64_32 => {
            // S + A (truncated to unsigned 32-bit, overflow-checked)
            let value = (s as i64).wrapping_add(a);
            check_overflow(value, 32, false, reloc.reloc_type)?;
            write_u32_le(section_data, reloc, value as u32)?;
        }

        R_X86_64_32S => {
            // S + A (sign-extended 32-bit, overflow-checked)
            let value = (s as i64).wrapping_add(a);
            check_overflow(value, 32, true, reloc.reloc_type)?;
            write_u32_le(section_data, reloc, value as u32)?;
        }

        R_X86_64_16 => {
            // S + A (16-bit absolute)
            let value = (s as i64).wrapping_add(a);
            check_overflow(value, 16, false, reloc.reloc_type)?;
            write_u16_le(section_data, reloc, value as u16)?;
        }

        R_X86_64_PC16 => {
            // S + A - P (PC-relative 16-bit, signed)
            let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
            check_overflow(value, 16, true, reloc.reloc_type)?;
            write_u16_le(section_data, reloc, value as u16)?;
        }

        R_X86_64_PC64 => {
            // S + A - P (PC-relative 64-bit)
            let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        _ => {
            return Err(RelocationError::UnsupportedType(
                reloc.reloc_type,
                "x86-64",
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// i686 relocation handler
// ===========================================================================

/// Apply a single i686 (32-bit x86) relocation.
fn apply_i686_relocation(
    section_data: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    ctx: &RelocationContext,
) -> Result<(), RelocationError> {
    let s = symbol_value(reloc.symbol_index, ctx)?;
    let a = reloc.addend;
    let p = place_address(reloc, ctx)?;

    match reloc.reloc_type {
        R_386_NONE => { /* No-op */ }

        R_386_32 => {
            // S + A (absolute 32-bit)
            let value = (s as i64).wrapping_add(a) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        R_386_PC32 | R_386_PLT32 => {
            // S + A - P (PC-relative 32-bit)
            let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        R_386_GOT32 => {
            // G + A (GOT entry offset, 32-bit)
            let value = (ctx.got_address as i64).wrapping_add(a) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        R_386_GLOB_DAT | R_386_JMP_SLOT => {
            // S (absolute symbol address, 32-bit)
            write_u32_le(section_data, reloc, s as u32)?;
        }

        R_386_RELATIVE => {
            // B + A (base-relative; B = 0 for static executables)
            let base: u64 = 0;
            let value = (base as i64).wrapping_add(a) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        R_386_GOTOFF => {
            // S + A - GOT (offset from GOT base)
            let value =
                (s as i64).wrapping_add(a).wrapping_sub(ctx.got_address as i64) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        R_386_GOTPC => {
            // GOT + A - P (PC-relative GOT address)
            let value =
                (ctx.got_address as i64).wrapping_add(a).wrapping_sub(p as i64) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        _ => {
            return Err(RelocationError::UnsupportedType(
                reloc.reloc_type,
                "i686",
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// AArch64 relocation handler
// ===========================================================================

/// Apply a single AArch64 (ARM 64-bit) relocation.
fn apply_aarch64_relocation(
    section_data: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    ctx: &RelocationContext,
) -> Result<(), RelocationError> {
    let s = symbol_value(reloc.symbol_index, ctx)?;
    let a = reloc.addend;
    let p = place_address(reloc, ctx)?;

    match reloc.reloc_type {
        R_AARCH64_NONE => { /* No-op */ }

        R_AARCH64_ABS64 => {
            // S + A (absolute 64-bit data)
            let value = (s as i64).wrapping_add(a) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        R_AARCH64_ABS32 => {
            // S + A (absolute 32-bit data, overflow-checked)
            let value = (s as i64).wrapping_add(a);
            check_overflow(value, 32, true, reloc.reloc_type)?;
            write_u32_le(section_data, reloc, value as u32)?;
        }

        R_AARCH64_ABS16 => {
            // S + A (absolute 16-bit data, overflow-checked)
            let value = (s as i64).wrapping_add(a);
            check_overflow(value, 16, true, reloc.reloc_type)?;
            write_u16_le(section_data, reloc, value as u16)?;
        }

        R_AARCH64_PREL64 => {
            // S + A - P (PC-relative 64-bit data)
            let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        R_AARCH64_PREL32 => {
            // S + A - P (PC-relative 32-bit data, overflow-checked)
            let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
            check_overflow(value, 32, true, reloc.reloc_type)?;
            write_i32_le(section_data, reloc, value as i32)?;
        }

        R_AARCH64_CALL26 | R_AARCH64_JUMP26 => {
            // S + A - P, then >> 2 (26-bit branch offset in 4-byte aligned insn)
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
            // Branch offsets must be 4-byte aligned (bit 0,1 = 0)
            check_overflow(offset >> 2, 26, true, reloc.reloc_type)?;
            let imm26 = ((offset >> 2) as u32) & 0x03FF_FFFF;
            patch_instruction_bits(
                section_data,
                reloc.section_index,
                reloc.offset,
                imm26,
                0x03FF_FFFF,
            )?;
        }

        R_AARCH64_ADR_PREL_PG_HI21 => {
            // Page(S+A) - Page(P), then >> 12 (21-bit page offset for ADRP)
            let sa = (s as i64).wrapping_add(a);
            let page_sa = sa & !0xFFF;
            let page_p = (p as i64) & !0xFFF;
            let page_off = page_sa.wrapping_sub(page_p) >> 12;
            check_overflow(page_off, 21, true, reloc.reloc_type)?;
            patch_adrp_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                page_off as u32,
            )?;
        }

        R_AARCH64_ADD_ABS_LO12_NC => {
            // (S + A) & 0xFFF (12-bit page offset, no overflow check)
            let value = ((s as i64).wrapping_add(a) & 0xFFF) as u32;
            patch_add_immediate(section_data, reloc.section_index, reloc.offset, value)?;
        }

        R_AARCH64_LDST64_ABS_LO12_NC => {
            // ((S + A) & 0xFFF) >> 3 — scaled 64-bit load/store offset
            let byte_off = ((s as i64).wrapping_add(a) & 0xFFF) as u32;
            let scaled = byte_off >> 3;
            // Patch the imm12 field at bits [21:10]
            let mask = 0xFFF << 10;
            let encoded = (scaled & 0xFFF) << 10;
            patch_instruction_bits(
                section_data,
                reloc.section_index,
                reloc.offset,
                encoded,
                mask,
            )?;
        }

        R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT => {
            // S (absolute symbol address, 64-bit)
            write_u64_le(section_data, reloc, s)?;
        }

        R_AARCH64_RELATIVE => {
            // B + A (base-relative; B = 0 for static)
            let base: u64 = 0;
            let value = (base as i64).wrapping_add(a) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        _ => {
            return Err(RelocationError::UnsupportedType(
                reloc.reloc_type,
                "aarch64",
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// RISC-V 64 relocation handler
// ===========================================================================

/// Apply a single RISC-V 64-bit relocation.
fn apply_riscv64_relocation(
    section_data: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    ctx: &RelocationContext,
) -> Result<(), RelocationError> {
    let s = symbol_value(reloc.symbol_index, ctx)?;
    let a = reloc.addend;
    let p = place_address(reloc, ctx)?;

    match reloc.reloc_type {
        R_RISCV_NONE => { /* No-op */ }

        R_RISCV_64 => {
            // S + A (absolute 64-bit)
            let value = (s as i64).wrapping_add(a) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        R_RISCV_32 => {
            // S + A (absolute 32-bit)
            let value = (s as i64).wrapping_add(a) as u32;
            write_u32_le(section_data, reloc, value)?;
        }

        R_RISCV_RELATIVE => {
            // B + A (base-relative; B = 0 for static)
            let base: u64 = 0;
            let value = (base as i64).wrapping_add(a) as u64;
            write_u64_le(section_data, reloc, value)?;
        }

        R_RISCV_JUMP_SLOT => {
            // S (absolute symbol address, 64-bit)
            write_u64_le(section_data, reloc, s)?;
        }

        R_RISCV_BRANCH => {
            // S + A - P (B-type immediate, 13-bit signed, bit 0 always 0)
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as i32;
            check_overflow(offset as i64, 13, true, reloc.reloc_type)?;
            patch_riscv_b_immediate(section_data, reloc.section_index, reloc.offset, offset)?;
        }

        R_RISCV_JAL => {
            // S + A - P (J-type immediate, 21-bit signed, bit 0 always 0)
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as i32;
            check_overflow(offset as i64, 21, true, reloc.reloc_type)?;
            patch_riscv_j_immediate(section_data, reloc.section_index, reloc.offset, offset)?;
        }

        R_RISCV_CALL => {
            // S + A - P split across AUIPC (hi20) + JALR (lo12)
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as i32;
            // Add 0x800 to round the upper 20 bits correctly when the
            // lower 12 bits are sign-extended by JALR.
            let hi = ((offset.wrapping_add(0x800)) >> 12) & 0xFFFFF;
            let lo = offset & 0xFFF;
            // Patch AUIPC at reloc.offset (U-type)
            patch_riscv_u_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                hi as u32,
            )?;
            // Patch JALR at reloc.offset + 4 (I-type)
            patch_riscv_i_immediate(
                section_data,
                reloc.section_index,
                reloc.offset + 4,
                lo as u32,
            )?;
        }

        R_RISCV_PCREL_HI20 => {
            // S + A - P (upper 20 bits for AUIPC, PC-relative)
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as i32;
            let hi = ((offset.wrapping_add(0x800)) >> 12) & 0xFFFFF;
            patch_riscv_u_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                hi as u32,
            )?;
        }

        R_RISCV_PCREL_LO12_I => {
            // Lower 12 bits complementing a PCREL_HI20, I-type format.
            // In our simplified model, the symbol S and addend A describe
            // the same target as the paired HI20 relocation.
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as i32;
            let lo = offset & 0xFFF;
            patch_riscv_i_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                lo as u32,
            )?;
        }

        R_RISCV_PCREL_LO12_S => {
            // Lower 12 bits complementing a PCREL_HI20, S-type format.
            let offset = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as i32;
            let lo = offset & 0xFFF;
            patch_riscv_s_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                lo as u32,
            )?;
        }

        R_RISCV_HI20 => {
            // S + A (upper 20 bits for LUI, absolute)
            let value = (s as i64).wrapping_add(a) as i32;
            let hi = ((value.wrapping_add(0x800)) >> 12) & 0xFFFFF;
            patch_riscv_u_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                hi as u32,
            )?;
        }

        R_RISCV_LO12_I => {
            // S + A (lower 12 bits, I-type, absolute)
            let value = (s as i64).wrapping_add(a) as i32;
            let lo = value & 0xFFF;
            patch_riscv_i_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                lo as u32,
            )?;
        }

        R_RISCV_LO12_S => {
            // S + A (lower 12 bits, S-type, absolute)
            let value = (s as i64).wrapping_add(a) as i32;
            let lo = value & 0xFFF;
            patch_riscv_s_immediate(
                section_data,
                reloc.section_index,
                reloc.offset,
                lo as u32,
            )?;
        }

        R_RISCV_RELAX => {
            // Linker relaxation hint — we do not implement relaxation,
            // so this is silently skipped.
        }

        _ => {
            return Err(RelocationError::UnsupportedType(
                reloc.reloc_type,
                "riscv64",
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// Byte-level write helpers (safe, little-endian)
// ===========================================================================

/// Validate that the section and offset are in bounds for a write of `size` bytes.
fn validate_bounds(
    sections: &[Vec<u8>],
    section_index: usize,
    offset: u64,
    size: usize,
) -> Result<(), RelocationError> {
    if section_index >= sections.len() {
        return Err(RelocationError::SectionOutOfBounds(section_index));
    }
    let end = offset as usize + size;
    if end > sections[section_index].len() {
        return Err(RelocationError::OffsetOutOfBounds {
            section: section_index,
            offset,
            size,
        });
    }
    Ok(())
}

/// Write a 64-bit little-endian value at the relocation's section and offset.
fn write_u64_le(
    sections: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    value: u64,
) -> Result<(), RelocationError> {
    validate_bounds(sections, reloc.section_index, reloc.offset, 8)?;
    let off = reloc.offset as usize;
    sections[reloc.section_index][off..off + 8].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Write a 32-bit unsigned little-endian value at the relocation's position.
fn write_u32_le(
    sections: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    value: u32,
) -> Result<(), RelocationError> {
    validate_bounds(sections, reloc.section_index, reloc.offset, 4)?;
    let off = reloc.offset as usize;
    sections[reloc.section_index][off..off + 4].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Write a 32-bit signed little-endian value at the relocation's position.
fn write_i32_le(
    sections: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    value: i32,
) -> Result<(), RelocationError> {
    validate_bounds(sections, reloc.section_index, reloc.offset, 4)?;
    let off = reloc.offset as usize;
    sections[reloc.section_index][off..off + 4].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Write a 16-bit little-endian value at the relocation's position.
fn write_u16_le(
    sections: &mut [Vec<u8>],
    reloc: &RelocationEntry,
    value: u16,
) -> Result<(), RelocationError> {
    validate_bounds(sections, reloc.section_index, reloc.offset, 2)?;
    let off = reloc.offset as usize;
    sections[reloc.section_index][off..off + 2].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

// ===========================================================================
// Instruction bit-field patching (AArch64 and RISC-V)
// ===========================================================================

/// Read a 32-bit little-endian instruction word, clear the bits indicated
/// by `mask`, OR in `value & mask`, and write it back.
///
/// This preserves all non-immediate bits (opcode, register fields) while
/// patching only the immediate field.
fn patch_instruction_bits(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: u32,
    mask: u32,
) -> Result<(), RelocationError> {
    validate_bounds(sections, section_index, offset, 4)?;
    let off = offset as usize;
    let insn_bytes = &sections[section_index][off..off + 4];
    let insn = u32::from_le_bytes([insn_bytes[0], insn_bytes[1], insn_bytes[2], insn_bytes[3]]);
    let patched = (insn & !mask) | (value & mask);
    sections[section_index][off..off + 4].copy_from_slice(&patched.to_le_bytes());
    Ok(())
}

// ---------------------------------------------------------------------------
// AArch64 instruction patching helpers
// ---------------------------------------------------------------------------

/// Patch the ADRP instruction immediate (21-bit page offset).
///
/// ADRP format: `[1 | immlo(2) | 10000 | immhi(19) | Rd(5)]`
/// - `immlo` = value[1:0] placed at bits [30:29]
/// - `immhi` = value[20:2] placed at bits [23:5]
fn patch_adrp_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: u32,
) -> Result<(), RelocationError> {
    let immlo = value & 0x3;
    let immhi = (value >> 2) & 0x7_FFFF;
    let encoded = (immlo << 29) | (immhi << 5);
    let mask = (0x3u32 << 29) | (0x7_FFFFu32 << 5);
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

/// Patch the ADD immediate instruction (12-bit immediate).
///
/// ADD format: `[sf | 0 | 0 | 100010 | sh | imm12(12) | Rn(5) | Rd(5)]`
/// - `imm12` placed at bits [21:10]
fn patch_add_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: u32,
) -> Result<(), RelocationError> {
    let encoded = (value & 0xFFF) << 10;
    let mask = 0xFFFu32 << 10;
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

// ---------------------------------------------------------------------------
// RISC-V instruction patching helpers
// ---------------------------------------------------------------------------

/// Patch a RISC-V B-type instruction immediate (branch offset).
///
/// B-type format bits:
/// - inst[31]    = imm[12]
/// - inst[30:25] = imm[10:5]
/// - inst[11:8]  = imm[4:1]
/// - inst[7]     = imm[11]
///
/// The offset is a signed byte count; bit 0 is not encoded (always 0).
fn patch_riscv_b_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: i32,
) -> Result<(), RelocationError> {
    let v = value as u32;
    let imm12 = (v >> 12) & 1;
    let imm10_5 = (v >> 5) & 0x3F;
    let imm4_1 = (v >> 1) & 0xF;
    let imm11 = (v >> 11) & 1;
    let encoded = (imm12 << 31) | (imm10_5 << 25) | (imm4_1 << 8) | (imm11 << 7);
    let mask = (1u32 << 31) | (0x3Fu32 << 25) | (0xFu32 << 8) | (1u32 << 7);
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

/// Patch a RISC-V J-type instruction immediate (JAL offset).
///
/// J-type format bits:
/// - inst[31]    = imm[20]
/// - inst[30:21] = imm[10:1]
/// - inst[20]    = imm[11]
/// - inst[19:12] = imm[19:12]
///
/// The offset is a signed byte count; bit 0 is not encoded (always 0).
fn patch_riscv_j_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: i32,
) -> Result<(), RelocationError> {
    let v = value as u32;
    let imm20 = (v >> 20) & 1;
    let imm10_1 = (v >> 1) & 0x3FF;
    let imm11 = (v >> 11) & 1;
    let imm19_12 = (v >> 12) & 0xFF;
    let encoded = (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12);
    let mask = (1u32 << 31) | (0x3FFu32 << 21) | (1u32 << 20) | (0xFFu32 << 12);
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

/// Patch a RISC-V U-type instruction immediate (upper 20 bits).
///
/// U-type format: inst[31:12] = imm[31:12]
fn patch_riscv_u_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: u32,
) -> Result<(), RelocationError> {
    let encoded = (value & 0xF_FFFF) << 12;
    let mask: u32 = 0xFFFF_F000;
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

/// Patch a RISC-V I-type instruction immediate (lower 12 bits).
///
/// I-type format: inst[31:20] = imm[11:0]
fn patch_riscv_i_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: u32,
) -> Result<(), RelocationError> {
    let encoded = (value & 0xFFF) << 20;
    let mask: u32 = 0xFFF0_0000;
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

/// Patch a RISC-V S-type instruction immediate (lower 12 bits, split).
///
/// S-type format:
/// - inst[31:25] = imm[11:5]
/// - inst[11:7]  = imm[4:0]
fn patch_riscv_s_immediate(
    sections: &mut [Vec<u8>],
    section_index: usize,
    offset: u64,
    value: u32,
) -> Result<(), RelocationError> {
    let hi = (value >> 5) & 0x7F;
    let lo = value & 0x1F;
    let encoded = (hi << 25) | (lo << 7);
    let mask = (0x7Fu32 << 25) | (0x1Fu32 << 7);
    patch_instruction_bits(sections, section_index, offset, encoded, mask)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal RelocationContext for a given architecture.
    fn make_ctx<'a>(
        arch: Architecture,
        symbols: &'a [ResolvedSymbol],
        section_addrs: &'a [u64],
    ) -> RelocationContext<'a> {
        RelocationContext {
            symbols,
            section_addresses: section_addrs,
            got_address: 0x3000,
            plt_address: 0x2000,
            arch,
            is_pic: false,
        }
    }

    /// Helper: create a ResolvedSymbol with just a name and address.
    fn sym(name: &str, address: u64) -> ResolvedSymbol {
        use crate::linker::symbols::{SymbolBinding, SymbolType, SymbolVisibility};
        ResolvedSymbol {
            name: name.to_string(),
            address,
            size: 0,
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::Function,
            visibility: SymbolVisibility::Default,
            output_section_index: 0,
        }
    }

    // -----------------------------------------------------------------------
    // x86-64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_pc32() {
        // S = 0x1000, A = -4, P = 0x2010
        // value = S + A - P = 0x1000 + (-4) - 0x2010 = -0x1014
        let symbols = vec![sym("target", 0x1000)];
        let section_addrs = vec![0x2000u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);

        let mut sections = vec![vec![0u8; 64]];
        let reloc = RelocationEntry {
            offset: 0x10,
            reloc_type: R_X86_64_PC32,
            symbol_index: 0,
            addend: -4,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = i32::from_le_bytes([
            sections[0][0x10],
            sections[0][0x11],
            sections[0][0x12],
            sections[0][0x13],
        ]);
        assert_eq!(result, -0x1014);
    }

    #[test]
    fn test_x86_64_abs64() {
        let symbols = vec![sym("data", 0xDEAD_BEEF_CAFE_0000)];
        let section_addrs = vec![0x4000u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);

        let mut sections = vec![vec![0u8; 32]];
        let reloc = RelocationEntry {
            offset: 8,
            reloc_type: R_X86_64_64,
            symbol_index: 0,
            addend: 0x100,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = u64::from_le_bytes([
            sections[0][8],
            sections[0][9],
            sections[0][10],
            sections[0][11],
            sections[0][12],
            sections[0][13],
            sections[0][14],
            sections[0][15],
        ]);
        assert_eq!(result, 0xDEAD_BEEF_CAFE_0100);
    }

    #[test]
    fn test_x86_64_none_skipped() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0xFFu8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_X86_64_NONE,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        // Data must be unchanged
        assert_eq!(sections[0], vec![0xFFu8; 8]);
    }

    #[test]
    fn test_x86_64_plt32() {
        // PLT32 uses same formula as PC32: S + A - P
        let symbols = vec![sym("puts", 0x5000)];
        let section_addrs = vec![0x1000u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 16]];
        let reloc = RelocationEntry {
            offset: 4,
            reloc_type: R_X86_64_PLT32,
            symbol_index: 0,
            addend: -4,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        let result = i32::from_le_bytes([
            sections[0][4],
            sections[0][5],
            sections[0][6],
            sections[0][7],
        ]);
        // 0x5000 + (-4) - 0x1004 = 0x3FF8
        assert_eq!(result, 0x3FF8);
    }

    #[test]
    fn test_x86_64_pc64() {
        let symbols = vec![sym("far", 0x1_0000_0000)];
        let section_addrs = vec![0x1000u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 16]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_X86_64_PC64,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        let result = u64::from_le_bytes(sections[0][0..8].try_into().unwrap());
        // 0x1_0000_0000 + 0 - 0x1000 = 0xFFFF_F000
        assert_eq!(result, 0xFFFF_F000);
    }

    // -----------------------------------------------------------------------
    // i686 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i686_abs32() {
        let symbols = vec![sym("func", 0x0804_8100)];
        let section_addrs = vec![0x0804_8000u64];
        let ctx = make_ctx(Architecture::I686, &symbols, &section_addrs);

        let mut sections = vec![vec![0u8; 16]];
        let reloc = RelocationEntry {
            offset: 4,
            reloc_type: R_386_32,
            symbol_index: 0,
            addend: 0x10,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = u32::from_le_bytes([
            sections[0][4],
            sections[0][5],
            sections[0][6],
            sections[0][7],
        ]);
        assert_eq!(result, 0x0804_8110);
    }

    #[test]
    fn test_i686_pc32() {
        let symbols = vec![sym("target", 0x1000)];
        let section_addrs = vec![0x2000u64];
        let ctx = make_ctx(Architecture::I686, &symbols, &section_addrs);

        let mut sections = vec![vec![0u8; 16]];
        let reloc = RelocationEntry {
            offset: 4,
            reloc_type: R_386_PC32,
            symbol_index: 0,
            addend: -4,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = i32::from_le_bytes([
            sections[0][4],
            sections[0][5],
            sections[0][6],
            sections[0][7],
        ]);
        // 0x1000 - 4 - 0x2004 = -0x1008
        assert_eq!(result, -0x1008);
    }

    #[test]
    fn test_i686_none_skipped() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::I686, &symbols, &section_addrs);
        let mut sections = vec![vec![0xAAu8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_386_NONE,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        assert_eq!(sections[0], vec![0xAAu8; 8]);
    }

    // -----------------------------------------------------------------------
    // AArch64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_aarch64_call26() {
        // S = 0x40000, A = 0, P = 0x10000
        // offset = 0x30000, imm26 = 0x30000 >> 2 = 0xC000
        let symbols = vec![sym("func", 0x40000)];
        let section_addrs = vec![0x10000u64];
        let ctx = make_ctx(Architecture::Aarch64, &symbols, &section_addrs);

        // Start with a BL instruction skeleton: opcode = 0x94000000
        let mut sections = vec![vec![0u8; 8]];
        let bl_insn: u32 = 0x9400_0000;
        sections[0][0..4].copy_from_slice(&bl_insn.to_le_bytes());

        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_CALL26,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = u32::from_le_bytes([
            sections[0][0],
            sections[0][1],
            sections[0][2],
            sections[0][3],
        ]);
        // imm26 = 0xC000, merged with BL opcode
        assert_eq!(result & 0x03FF_FFFF, 0xC000);
        assert_eq!(result & 0xFC00_0000, 0x9400_0000);
    }

    #[test]
    fn test_aarch64_adrp() {
        // S = 0x401000, A = 0, P = 0x400000
        // page_s = 0x401000, page_p = 0x400000
        // page_off = (0x401000 - 0x400000) >> 12 = 1
        let symbols = vec![sym("data", 0x401000)];
        let section_addrs = vec![0x400000u64];
        let ctx = make_ctx(Architecture::Aarch64, &symbols, &section_addrs);

        // ADRP x0 skeleton: 0x90000000
        let mut sections = vec![vec![0u8; 8]];
        let adrp: u32 = 0x9000_0000;
        sections[0][0..4].copy_from_slice(&adrp.to_le_bytes());

        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_ADR_PREL_PG_HI21,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = u32::from_le_bytes([
            sections[0][0],
            sections[0][1],
            sections[0][2],
            sections[0][3],
        ]);
        // page_off = 1 → immlo = 1 & 3 = 1, immhi = (1>>2) & 0x7FFFF = 0
        let immlo = (result >> 29) & 0x3;
        let immhi = (result >> 5) & 0x7_FFFF;
        assert_eq!(immlo, 1);
        assert_eq!(immhi, 0);
    }

    #[test]
    fn test_aarch64_add_lo12() {
        // (S + A) & 0xFFF = 0x401234 & 0xFFF = 0x234
        let symbols = vec![sym("data", 0x401234)];
        let section_addrs = vec![0x400000u64];
        let ctx = make_ctx(Architecture::Aarch64, &symbols, &section_addrs);

        // ADD x0, x0, #0 skeleton: 0x91000000
        let mut sections = vec![vec![0u8; 8]];
        let add: u32 = 0x9100_0000;
        sections[0][0..4].copy_from_slice(&add.to_le_bytes());

        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_ADD_ABS_LO12_NC,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = u32::from_le_bytes(sections[0][0..4].try_into().unwrap());
        let imm12 = (result >> 10) & 0xFFF;
        assert_eq!(imm12, 0x234);
    }

    #[test]
    fn test_aarch64_abs64() {
        let symbols = vec![sym("global", 0xABCD_EF01_2345_6789)];
        let section_addrs = vec![0x1000u64];
        let ctx = make_ctx(Architecture::Aarch64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 16]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_ABS64,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        let result = u64::from_le_bytes(sections[0][0..8].try_into().unwrap());
        assert_eq!(result, 0xABCD_EF01_2345_6789);
    }

    #[test]
    fn test_aarch64_none_skipped() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::Aarch64, &symbols, &section_addrs);
        let mut sections = vec![vec![0xBBu8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_NONE,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        assert_eq!(sections[0], vec![0xBBu8; 8]);
    }

    // -----------------------------------------------------------------------
    // RISC-V tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_riscv_call() {
        // S = 0x10100, A = 0, P = 0x10000
        // offset = 0x100
        // hi = (0x100 + 0x800) >> 12 = 0
        // lo = 0x100 & 0xFFF = 0x100
        let symbols = vec![sym("func", 0x10100)];
        let section_addrs = vec![0x10000u64];
        let ctx = make_ctx(Architecture::Riscv64, &symbols, &section_addrs);

        // AUIPC x1 (0x00000097) + JALR x1, x1 (0x000080E7)
        let mut sections = vec![vec![0u8; 16]];
        let auipc: u32 = 0x0000_0097;
        let jalr: u32 = 0x0000_80E7;
        sections[0][0..4].copy_from_slice(&auipc.to_le_bytes());
        sections[0][4..8].copy_from_slice(&jalr.to_le_bytes());

        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_RISCV_CALL,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        // Check AUIPC: upper 20 bits should be 0 (hi = 0)
        let auipc_result = u32::from_le_bytes(sections[0][0..4].try_into().unwrap());
        assert_eq!(auipc_result & 0xFFFF_F000, 0);

        // Check JALR: imm[11:0] should be 0x100 at bits [31:20]
        let jalr_result = u32::from_le_bytes(sections[0][4..8].try_into().unwrap());
        let imm_i = jalr_result >> 20;
        assert_eq!(imm_i & 0xFFF, 0x100);
    }

    #[test]
    fn test_riscv_branch() {
        // S = 0x10010, A = 0, P = 0x10000
        // offset = 0x10 (16 bytes forward)
        let symbols = vec![sym("label", 0x10010)];
        let section_addrs = vec![0x10000u64];
        let ctx = make_ctx(Architecture::Riscv64, &symbols, &section_addrs);

        // BEQ x0, x0 skeleton: 0x00000063
        let mut sections = vec![vec![0u8; 8]];
        let beq: u32 = 0x0000_0063;
        sections[0][0..4].copy_from_slice(&beq.to_le_bytes());

        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_RISCV_BRANCH,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        let result = u32::from_le_bytes(sections[0][0..4].try_into().unwrap());
        // offset = 0x10 = 0b10000
        // imm[4:1] = 0x10 >> 1 = 8
        let imm4_1 = (result >> 8) & 0xF;
        assert_eq!(imm4_1, 8);
    }

    #[test]
    fn test_riscv_abs64() {
        let symbols = vec![sym("val", 0x1234_5678_9ABC_DEF0)];
        let section_addrs = vec![0x1000u64];
        let ctx = make_ctx(Architecture::Riscv64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 16]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_RISCV_64,
            symbol_index: 0,
            addend: 0x10,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        let result = u64::from_le_bytes(sections[0][0..8].try_into().unwrap());
        assert_eq!(result, 0x1234_5678_9ABC_DF00);
    }

    #[test]
    fn test_riscv_none_skipped() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::Riscv64, &symbols, &section_addrs);
        let mut sections = vec![vec![0xCCu8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_RISCV_NONE,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        assert_eq!(sections[0], vec![0xCCu8; 8]);
    }

    #[test]
    fn test_riscv_relax_skipped() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::Riscv64, &symbols, &section_addrs);
        let mut sections = vec![vec![0xDDu8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_RISCV_RELAX,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();
        assert_eq!(sections[0], vec![0xDDu8; 8]);
    }

    // -----------------------------------------------------------------------
    // Overflow detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_overflow_signed_in_range() {
        assert!(check_overflow(127, 8, true, 0).is_ok());
        assert!(check_overflow(-128, 8, true, 0).is_ok());
        assert!(check_overflow(0, 8, true, 0).is_ok());
    }

    #[test]
    fn test_overflow_signed_out_of_range() {
        assert!(check_overflow(128, 8, true, 0).is_err());
        assert!(check_overflow(-129, 8, true, 0).is_err());
    }

    #[test]
    fn test_overflow_unsigned_in_range() {
        assert!(check_overflow(255, 8, false, 0).is_ok());
        assert!(check_overflow(0, 8, false, 0).is_ok());
    }

    #[test]
    fn test_overflow_unsigned_out_of_range() {
        assert!(check_overflow(256, 8, false, 0).is_err());
        assert!(check_overflow(-1, 8, false, 0).is_err());
    }

    #[test]
    fn test_overflow_32bit_pc_relative() {
        let big_val: i64 = 3_000_000_000;
        assert!(check_overflow(big_val, 32, true, R_X86_64_PC32).is_err());
    }

    #[test]
    fn test_overflow_64bit_always_ok() {
        // 64-bit overflow check always succeeds (no truncation possible)
        assert!(check_overflow(i64::MAX, 64, true, 0).is_ok());
        assert!(check_overflow(i64::MIN, 64, true, 0).is_ok());
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unsupported_relocation_type() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: 999,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        let err = apply_relocations(&mut sections, &[reloc], &ctx).unwrap_err();
        match err {
            RelocationError::UnsupportedType(999, "x86-64") => {}
            _ => panic!("expected UnsupportedType, got {:?}", err),
        }
    }

    #[test]
    fn test_symbol_not_found() {
        let symbols: Vec<ResolvedSymbol> = vec![];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_X86_64_64,
            symbol_index: 5,
            addend: 0,
            section_index: 0,
        };
        let err = apply_relocations(&mut sections, &[reloc], &ctx).unwrap_err();
        match err {
            RelocationError::SymbolNotFound(5) => {}
            _ => panic!("expected SymbolNotFound, got {:?}", err),
        }
    }

    #[test]
    fn test_section_out_of_bounds() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 8]];
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_X86_64_64,
            symbol_index: 0,
            addend: 0,
            section_index: 5,
        };
        let err = apply_relocations(&mut sections, &[reloc], &ctx).unwrap_err();
        match err {
            RelocationError::SectionOutOfBounds(5) => {}
            _ => panic!("expected SectionOutOfBounds, got {:?}", err),
        }
    }

    #[test]
    fn test_offset_out_of_bounds() {
        let symbols = vec![sym("x", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections = vec![vec![0u8; 4]]; // only 4 bytes
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_X86_64_64, // needs 8 bytes
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        };
        let err = apply_relocations(&mut sections, &[reloc], &ctx).unwrap_err();
        match err {
            RelocationError::OffsetOutOfBounds { section: 0, offset: 0, size: 8 } => {}
            _ => panic!("expected OffsetOutOfBounds, got {:?}", err),
        }
    }

    // -----------------------------------------------------------------------
    // Byte order test
    // -----------------------------------------------------------------------

    #[test]
    fn test_little_endian_write() {
        let symbols = vec![sym("val", 0)];
        let section_addrs = vec![0u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);

        let mut sections = vec![vec![0u8; 16]];
        // Write 0x0102_0304 via R_X86_64_32
        let reloc = RelocationEntry {
            offset: 0,
            reloc_type: R_X86_64_32,
            symbol_index: 0,
            addend: 0x0102_0304,
            section_index: 0,
        };
        apply_relocations(&mut sections, &[reloc], &ctx).unwrap();

        // Little-endian: least significant byte first
        assert_eq!(sections[0][0], 0x04);
        assert_eq!(sections[0][1], 0x03);
        assert_eq!(sections[0][2], 0x02);
        assert_eq!(sections[0][3], 0x01);
    }

    // -----------------------------------------------------------------------
    // Relocation constant value tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_constant_values() {
        assert_eq!(R_X86_64_NONE, 0);
        assert_eq!(R_X86_64_64, 1);
        assert_eq!(R_X86_64_PC32, 2);
        assert_eq!(R_X86_64_GOT32, 3);
        assert_eq!(R_X86_64_PLT32, 4);
        assert_eq!(R_X86_64_GLOB_DAT, 6);
        assert_eq!(R_X86_64_JUMP_SLOT, 7);
        assert_eq!(R_X86_64_RELATIVE, 8);
        assert_eq!(R_X86_64_GOTPCREL, 9);
        assert_eq!(R_X86_64_32, 10);
        assert_eq!(R_X86_64_32S, 11);
        assert_eq!(R_X86_64_16, 12);
        assert_eq!(R_X86_64_PC16, 13);
        assert_eq!(R_X86_64_PC64, 24);
    }

    #[test]
    fn test_i686_constant_values() {
        assert_eq!(R_386_NONE, 0);
        assert_eq!(R_386_32, 1);
        assert_eq!(R_386_PC32, 2);
        assert_eq!(R_386_GOT32, 3);
        assert_eq!(R_386_PLT32, 4);
        assert_eq!(R_386_GLOB_DAT, 6);
        assert_eq!(R_386_JMP_SLOT, 7);
        assert_eq!(R_386_RELATIVE, 8);
        assert_eq!(R_386_GOTOFF, 9);
        assert_eq!(R_386_GOTPC, 10);
    }

    #[test]
    fn test_aarch64_constant_values() {
        assert_eq!(R_AARCH64_NONE, 0);
        assert_eq!(R_AARCH64_ABS64, 257);
        assert_eq!(R_AARCH64_ABS32, 258);
        assert_eq!(R_AARCH64_ABS16, 259);
        assert_eq!(R_AARCH64_PREL64, 260);
        assert_eq!(R_AARCH64_PREL32, 261);
        assert_eq!(R_AARCH64_ADR_PREL_PG_HI21, 275);
        assert_eq!(R_AARCH64_ADD_ABS_LO12_NC, 277);
        assert_eq!(R_AARCH64_JUMP26, 282);
        assert_eq!(R_AARCH64_CALL26, 283);
        assert_eq!(R_AARCH64_LDST64_ABS_LO12_NC, 286);
        assert_eq!(R_AARCH64_GLOB_DAT, 1025);
        assert_eq!(R_AARCH64_JUMP_SLOT, 1026);
        assert_eq!(R_AARCH64_RELATIVE, 1027);
    }

    #[test]
    fn test_riscv_constant_values() {
        assert_eq!(R_RISCV_NONE, 0);
        assert_eq!(R_RISCV_32, 1);
        assert_eq!(R_RISCV_64, 2);
        assert_eq!(R_RISCV_RELATIVE, 3);
        assert_eq!(R_RISCV_JUMP_SLOT, 5);
        assert_eq!(R_RISCV_BRANCH, 16);
        assert_eq!(R_RISCV_JAL, 17);
        assert_eq!(R_RISCV_CALL, 18);
        assert_eq!(R_RISCV_PCREL_HI20, 23);
        assert_eq!(R_RISCV_PCREL_LO12_I, 24);
        assert_eq!(R_RISCV_PCREL_LO12_S, 25);
        assert_eq!(R_RISCV_HI20, 26);
        assert_eq!(R_RISCV_LO12_I, 27);
        assert_eq!(R_RISCV_LO12_S, 28);
        assert_eq!(R_RISCV_RELAX, 51);
    }

    // -----------------------------------------------------------------------
    // Display / Error trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relocation_error_display() {
        let err = RelocationError::UnsupportedType(42, "x86-64");
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported relocation type 42"));
        assert!(msg.contains("x86-64"));

        let err = RelocationError::OverflowError {
            reloc_type: 2,
            value: 0x1_0000_0000,
            max_bits: 32,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overflow"));
        assert!(msg.contains("32-bit"));
    }

    #[test]
    fn test_relocation_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(
            RelocationError::SymbolNotFound(0),
        );
        let _ = format!("{}", err);
    }

    // -----------------------------------------------------------------------
    // Multiple relocations in sequence
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_relocations() {
        let symbols = vec![sym("a", 0x1000), sym("b", 0x2000)];
        let section_addrs = vec![0x3000u64];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);

        let mut sections = vec![vec![0u8; 32]];
        let relocs = vec![
            RelocationEntry {
                offset: 0,
                reloc_type: R_X86_64_64,
                symbol_index: 0,
                addend: 0,
                section_index: 0,
            },
            RelocationEntry {
                offset: 8,
                reloc_type: R_X86_64_64,
                symbol_index: 1,
                addend: 0,
                section_index: 0,
            },
        ];
        apply_relocations(&mut sections, &relocs, &ctx).unwrap();

        let val_a = u64::from_le_bytes(sections[0][0..8].try_into().unwrap());
        let val_b = u64::from_le_bytes(sections[0][8..16].try_into().unwrap());
        assert_eq!(val_a, 0x1000);
        assert_eq!(val_b, 0x2000);
    }

    #[test]
    fn test_empty_relocations() {
        let symbols: Vec<ResolvedSymbol> = vec![];
        let section_addrs: Vec<u64> = vec![];
        let ctx = make_ctx(Architecture::X86_64, &symbols, &section_addrs);
        let mut sections: Vec<Vec<u8>> = vec![];
        // No relocations => no-op, no error
        apply_relocations(&mut sections, &[], &ctx).unwrap();
    }
}
