//! # Default Linker Script Behavior
//!
//! This module implements the default linker script behavior for section
//! ordering, entry point selection, and memory layout. It provides the
//! implicit rules that `ld` uses when no explicit linker script is given.
//!
//! ## Responsibilities
//! - Define default base addresses per architecture
//! - Specify section-to-segment mapping rules
//! - Select entry point (`_start` by default)
//! - Compute page-aligned segment layout
//! - Generate `PT_INTERP` for dynamically linked executables

use crate::driver::target::{Architecture, TargetConfig};

// ============================================================================
// Default Base Addresses
// ============================================================================

/// Default virtual base address for x86-64 executables.
/// Matches GNU ld default for x86-64 Linux.
pub const DEFAULT_BASE_X86_64: u64 = 0x400000;

/// Default virtual base address for i686 executables.
/// Matches GNU ld default for i386 Linux.
pub const DEFAULT_BASE_I686: u64 = 0x08048000;

/// Default virtual base address for AArch64 executables.
/// Matches GNU ld default for AArch64 Linux.
pub const DEFAULT_BASE_AARCH64: u64 = 0x400000;

/// Default virtual base address for RISC-V 64 executables.
pub const DEFAULT_BASE_RISCV64: u64 = 0x10000;

/// Default page size for memory alignment.
pub const PAGE_SIZE: u64 = 0x1000;

/// Default entry point symbol name.
pub const DEFAULT_ENTRY_POINT: &str = "_start";

/// Dynamic linker path for x86-64 Linux.
pub const INTERP_X86_64: &str = "/lib64/ld-linux-x86-64.so.2";

/// Dynamic linker path for i686 Linux.
pub const INTERP_I686: &str = "/lib/ld-linux.so.2";

/// Dynamic linker path for AArch64 Linux.
pub const INTERP_AARCH64: &str = "/lib/ld-linux-aarch64.so.1";

/// Dynamic linker path for RISC-V 64 Linux.
pub const INTERP_RISCV64: &str = "/lib/ld-linux-riscv64-lp64d.so.1";

// ============================================================================
// Public Functions
// ============================================================================

/// Get the default base virtual address for the given target architecture.
///
/// This is the starting virtual address where the first PT_LOAD segment
/// begins. The actual section addresses are computed by adding offsets
/// from the ELF header and program header sizes.
///
/// # Arguments
/// * `target` — Target configuration specifying the architecture.
///
/// # Returns
/// The default base address for the target architecture.
pub fn default_base_address(target: &TargetConfig) -> u64 {
    match target.arch {
        Architecture::X86_64 => DEFAULT_BASE_X86_64,
        Architecture::I686 => DEFAULT_BASE_I686,
        Architecture::Aarch64 => DEFAULT_BASE_AARCH64,
        Architecture::Riscv64 => DEFAULT_BASE_RISCV64,
    }
}

/// Get the dynamic linker (interpreter) path for the given target.
///
/// Used to populate the `PT_INTERP` program header in dynamically linked
/// executables.
///
/// # Arguments
/// * `target` — Target configuration specifying the architecture.
///
/// # Returns
/// The filesystem path to the dynamic linker for the target.
pub fn interpreter_path(target: &TargetConfig) -> &'static str {
    match target.arch {
        Architecture::X86_64 => INTERP_X86_64,
        Architecture::I686 => INTERP_I686,
        Architecture::Aarch64 => INTERP_AARCH64,
        Architecture::Riscv64 => INTERP_RISCV64,
    }
}

/// Standard section ordering for the output binary.
///
/// Sections are laid out in this order within their respective segments:
/// 1. Executable code (`.init`, `.plt`, `.text`, `.fini`)
/// 2. Read-only data (`.rodata`, `.eh_frame`, `.eh_frame_hdr`)
/// 3. Init/fini arrays (`.init_array`, `.fini_array`)
/// 4. Read-write data (`.data`, `.got`, `.got.plt`)
/// 5. Uninitialized data (`.bss`)
/// 6. Non-loadable metadata (`.comment`, `.shstrtab`, `.strtab`, `.symtab`)
pub fn default_section_order() -> Vec<&'static str> {
    vec![
        ".init",
        ".plt",
        ".text",
        ".fini",
        ".rodata",
        ".eh_frame",
        ".eh_frame_hdr",
        ".init_array",
        ".fini_array",
        ".data",
        ".got",
        ".got.plt",
        ".bss",
        ".comment",
    ]
}

/// Compute the aligned address for a section given its alignment requirement.
///
/// # Arguments
/// * `address` — Current address.
/// * `alignment` — Required alignment (must be a power of 2).
///
/// # Returns
/// The next address that satisfies the alignment constraint.
pub fn align_to(address: u64, alignment: u64) -> u64 {
    if alignment == 0 || alignment == 1 {
        return address;
    }
    let mask = alignment - 1;
    (address + mask) & !mask
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::target::TargetConfig;

    #[test]
    fn test_default_base_address_x86_64() {
        let target = TargetConfig::x86_64();
        assert_eq!(default_base_address(&target), DEFAULT_BASE_X86_64);
    }

    #[test]
    fn test_default_base_address_i686() {
        let target = TargetConfig::i686();
        assert_eq!(default_base_address(&target), DEFAULT_BASE_I686);
    }

    #[test]
    fn test_default_base_address_aarch64() {
        let target = TargetConfig::aarch64();
        assert_eq!(default_base_address(&target), DEFAULT_BASE_AARCH64);
    }

    #[test]
    fn test_default_base_address_riscv64() {
        let target = TargetConfig::riscv64();
        assert_eq!(default_base_address(&target), DEFAULT_BASE_RISCV64);
    }

    #[test]
    fn test_interpreter_path_x86_64() {
        let target = TargetConfig::x86_64();
        assert_eq!(interpreter_path(&target), INTERP_X86_64);
    }

    #[test]
    fn test_interpreter_path_i686() {
        let target = TargetConfig::i686();
        assert_eq!(interpreter_path(&target), INTERP_I686);
    }

    #[test]
    fn test_section_ordering() {
        let order = default_section_order();
        let text_pos = order.iter().position(|&s| s == ".text").unwrap();
        let data_pos = order.iter().position(|&s| s == ".data").unwrap();
        let bss_pos = order.iter().position(|&s| s == ".bss").unwrap();
        assert!(text_pos < data_pos);
        assert!(data_pos < bss_pos);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 4), 0);
        assert_eq!(align_to(1, 4), 4);
        assert_eq!(align_to(4, 4), 4);
        assert_eq!(align_to(5, 8), 8);
        assert_eq!(align_to(0x1001, 0x1000), 0x2000);
    }

    #[test]
    fn test_align_to_edge_cases() {
        assert_eq!(align_to(100, 0), 100);
        assert_eq!(align_to(100, 1), 100);
        assert_eq!(align_to(0, 0x1000), 0);
    }

    #[test]
    fn test_page_size() {
        assert_eq!(PAGE_SIZE, 4096);
    }
}
