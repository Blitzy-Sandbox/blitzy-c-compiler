//! Target triple parsing and configuration for the `bcc` compiler.
//!
//! This module parses `--target` command-line flag values (e.g., `x86_64-linux-gnu`,
//! `i686-linux-gnu`, `aarch64-linux-gnu`, `riscv64-linux-gnu`) and produces a
//! [`TargetConfig`] struct that carries all architecture-specific parameters needed
//! throughout the entire compilation pipeline — from type sizes in semantic analysis,
//! through code generation backend selection, to ELF format and relocation types in
//! the linker.
//!
//! # Supported Targets
//!
//! | Target Triple           | Arch    | Pointer | ELF   | ABI            |
//! |-------------------------|---------|---------|-------|----------------|
//! | `x86_64-linux-gnu`      | x86-64  | 8 bytes | ELF64 | System V AMD64 |
//! | `i686-linux-gnu`        | i686    | 4 bytes | ELF32 | cdecl/SysV i386|
//! | `aarch64-linux-gnu`     | AArch64 | 8 bytes | ELF64 | AAPCS64        |
//! | `riscv64-linux-gnu`     | RISC-V  | 8 bytes | ELF64 | LP64D          |
//!
//! # Usage
//!
//! ```ignore
//! use crate::driver::target::{resolve_target, TargetConfig};
//!
//! // From CLI --target flag:
//! let config = resolve_target(Some("x86_64-linux-gnu")).unwrap();
//! assert_eq!(config.pointer_size, 8);
//!
//! // Default to host architecture:
//! let host_config = resolve_target(None).unwrap();
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Architecture enum
// ---------------------------------------------------------------------------

/// Enumerates the four supported target processor architectures.
///
/// Each variant maps to a distinct code generation backend, integrated assembler,
/// ABI convention, and ELF relocation type set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// AMD/Intel 64-bit (x86-64 / AMD64)
    X86_64,
    /// Intel 32-bit (i686 / IA-32)
    I686,
    /// ARM 64-bit (AArch64 / ARMv8-A)
    Aarch64,
    /// RISC-V 64-bit (RV64GC)
    Riscv64,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::X86_64 => write!(f, "x86_64"),
            Architecture::I686 => write!(f, "i686"),
            Architecture::Aarch64 => write!(f, "aarch64"),
            Architecture::Riscv64 => write!(f, "riscv64"),
        }
    }
}

// ---------------------------------------------------------------------------
// ElfClass enum
// ---------------------------------------------------------------------------

/// Discriminates between 32-bit and 64-bit ELF object formats.
///
/// ELF32 is used for the i686 target; all other supported targets use ELF64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElfClass {
    /// 32-bit ELF (ELFCLASS32) — used by i686
    Elf32,
    /// 64-bit ELF (ELFCLASS64) — used by x86-64, AArch64, RISC-V 64
    Elf64,
}

impl fmt::Display for ElfClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElfClass::Elf32 => write!(f, "ELF32"),
            ElfClass::Elf64 => write!(f, "ELF64"),
        }
    }
}

// ---------------------------------------------------------------------------
// Endianness enum
// ---------------------------------------------------------------------------

/// Byte order for the target architecture.
///
/// All four supported Linux targets are little-endian. The `Big` variant is
/// included for completeness and forward compatibility but is not currently
/// selected by any supported target triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// Little-endian byte order (least significant byte first)
    Little,
    /// Big-endian byte order (most significant byte first)
    Big,
}

impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Endianness::Little => write!(f, "little-endian"),
            Endianness::Big => write!(f, "big-endian"),
        }
    }
}

// ---------------------------------------------------------------------------
// AbiVariant enum
// ---------------------------------------------------------------------------

/// Calling convention / ABI variant for the target architecture.
///
/// Each variant defines register argument passing conventions, stack frame
/// layout, caller/callee responsibilities, and return value conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbiVariant {
    /// System V AMD64 ABI: integer args in rdi, rsi, rdx, rcx, r8, r9;
    /// float/SSE args in xmm0-xmm7; caller cleans stack; 16-byte alignment.
    SystemVAmd64,
    /// System V i386 cdecl ABI: all arguments passed on stack (right-to-left
    /// push order); caller cleanup; return in eax/edx pair; 16-byte stack
    /// alignment at call site.
    SystemVi386Cdecl,
    /// AAPCS64 (Procedure Call Standard for AArch64): integer args in x0-x7;
    /// SIMD/FP args in v0-v7; x30 is the link register; SP must be 16-byte
    /// aligned; callee-saved x19-x28.
    Aapcs64,
    /// RISC-V LP64D ABI: integer args in a0-a7; float args in fa0-fa7;
    /// ra is the return address register; callee-saved s0-s11; SP 16-byte
    /// aligned.
    Lp64d,
}

impl fmt::Display for AbiVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbiVariant::SystemVAmd64 => write!(f, "System V AMD64"),
            AbiVariant::SystemVi386Cdecl => write!(f, "System V i386 cdecl"),
            AbiVariant::Aapcs64 => write!(f, "AAPCS64"),
            AbiVariant::Lp64d => write!(f, "LP64D"),
        }
    }
}

// ---------------------------------------------------------------------------
// TargetConfig struct
// ---------------------------------------------------------------------------

/// Comprehensive target configuration carrying all architecture-specific
/// parameters needed throughout the compilation pipeline.
///
/// A single `TargetConfig` instance is created at startup (via [`resolve_target`])
/// and propagated through every pipeline phase:
///
/// - **Semantic analysis** — uses `pointer_size`, `long_size`, type size methods
///   for C type layout computation.
/// - **Code generation** — uses `arch` to dispatch to the correct backend
///   (x86-64 / i686 / AArch64 / RISC-V 64).
/// - **Integrated linker** — uses `elf_class`, `elf_machine`, `endianness` for
///   ELF header generation and architecture-specific relocation processing.
/// - **Debug info** — uses architecture details for DWARF register mappings.
#[derive(Debug, Clone)]
pub struct TargetConfig {
    /// The target processor architecture.
    pub arch: Architecture,

    /// Full target triple string (e.g., `"x86_64-linux-gnu"`).
    pub triple: String,

    /// Pointer width in bytes (4 for i686, 8 for x86-64/AArch64/RISC-V 64).
    pub pointer_size: u32,

    /// Size of the C `long` type in bytes (4 for i686, 8 for others).
    pub long_size: u32,

    /// Size of the C `long double` type in bytes.
    ///
    /// - x86-64: 16 (x87 80-bit extended, padded to 16 bytes per SysV AMD64 ABI)
    /// - i686:   12 (x87 80-bit extended, padded to 12 bytes per SysV i386 ABI)
    /// - AArch64: 16 (IEEE 754 binary128 quad precision)
    /// - RISC-V 64: 16 (IEEE 754 binary128 quad precision)
    pub long_double_size: u32,

    /// ELF object format class (ELF32 for i686, ELF64 for all others).
    pub elf_class: ElfClass,

    /// Byte order (little-endian for all four supported targets).
    pub endianness: Endianness,

    /// ABI variant governing calling conventions and stack frame layout.
    pub abi: AbiVariant,

    /// Maximum natural alignment for the target, in bytes.
    ///
    /// This determines the alignment of `max_align_t` in `<stddef.h>`.
    /// All four targets require 16-byte maximum alignment (SSE / NEON / V-ext).
    pub max_alignment: u32,

    /// Required stack alignment in bytes at function call boundaries.
    ///
    /// All four supported ABIs mandate 16-byte stack alignment.
    pub stack_alignment: u32,

    /// ELF `e_machine` header field value.
    ///
    /// - `EM_X86_64`  = 62
    /// - `EM_386`     = 3
    /// - `EM_AARCH64` = 183
    /// - `EM_RISCV`   = 243
    pub elf_machine: u16,

    /// ELF `e_ident[EI_OSABI]` value. `ELFOSABI_NONE` (0) for Linux/GNU.
    pub elf_osabi: u8,

    /// Number of general-purpose registers in the architecture's register file.
    ///
    /// Used by the register allocator to determine available physical registers.
    pub gpr_count: u32,

    /// Number of floating-point / SIMD registers.
    pub fpr_count: u32,

    /// Default search paths for CRT startup objects (crt1.o, crti.o, crtn.o).
    pub crt_search_paths: Vec<String>,

    /// Default search paths for system libraries (libc, libm, etc.).
    pub lib_search_paths: Vec<String>,

    /// Whether retpoline generation is enabled (`-mretpoline`).
    /// Defaults to `false`; set by the pipeline from CLI arguments.
    pub retpoline: bool,

    /// Whether Intel CET control-flow protection is enabled (`-fcf-protection`).
    /// Defaults to `false`; set by the pipeline from CLI arguments.
    pub cf_protection: bool,

    /// Whether position-independent code generation is enabled (`-fPIC`).
    /// Defaults to `false`; set by the pipeline from CLI arguments.
    pub pic: bool,

    /// Whether the 128-byte red zone should be omitted (`-mno-red-zone`).
    /// Relevant only for x86-64; defaults to `false`.
    pub no_red_zone: bool,

    /// Whether each function should be emitted into `.text.<funcname>` (`-ffunction-sections`).
    pub function_sections: bool,

    /// Whether each global variable should be emitted into `.data.<varname>` (`-fdata-sections`).
    pub data_sections: bool,
}

// ---------------------------------------------------------------------------
// TargetConfig — per-architecture constructors
// ---------------------------------------------------------------------------

impl TargetConfig {
    /// Creates a `TargetConfig` for the **x86-64** (AMD64) architecture.
    ///
    /// System V AMD64 ABI, ELF64, 8-byte pointers, 16 GPRs (rax–r15),
    /// 16 SSE registers (xmm0–xmm15).
    pub fn x86_64() -> Self {
        TargetConfig {
            arch: Architecture::X86_64,
            triple: "x86_64-linux-gnu".to_string(),
            pointer_size: 8,
            long_size: 8,
            long_double_size: 16, // x87 80-bit, padded to 16 bytes
            elf_class: ElfClass::Elf64,
            endianness: Endianness::Little,
            abi: AbiVariant::SystemVAmd64,
            max_alignment: 16,
            stack_alignment: 16,
            elf_machine: 62, // EM_X86_64
            elf_osabi: 0,    // ELFOSABI_NONE
            gpr_count: 16,   // rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8–r15
            fpr_count: 16,   // xmm0–xmm15
            crt_search_paths: vec![
                "/usr/lib/x86_64-linux-gnu".to_string(),
                "/usr/lib64".to_string(),
                "/usr/lib".to_string(),
            ],
            lib_search_paths: vec![
                "/usr/lib/x86_64-linux-gnu".to_string(),
                "/usr/lib64".to_string(),
                "/usr/lib".to_string(),
            ],
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        }
    }

    /// Creates a `TargetConfig` for the **i686** (IA-32) architecture.
    ///
    /// System V i386 cdecl ABI, ELF32, 4-byte pointers, 8 GPRs (eax–edi),
    /// 8 SSE registers (xmm0–xmm7) or x87 FP stack.
    pub fn i686() -> Self {
        TargetConfig {
            arch: Architecture::I686,
            triple: "i686-linux-gnu".to_string(),
            pointer_size: 4,
            long_size: 4,
            long_double_size: 12, // x87 80-bit, padded to 12 bytes on i386
            elf_class: ElfClass::Elf32,
            endianness: Endianness::Little,
            abi: AbiVariant::SystemVi386Cdecl,
            max_alignment: 16, // SSE requires 16-byte alignment
            stack_alignment: 16,
            elf_machine: 3, // EM_386
            elf_osabi: 0,
            gpr_count: 8, // eax, ecx, edx, ebx, esp, ebp, esi, edi
            fpr_count: 8, // xmm0–xmm7 (or x87 st(0)–st(7))
            crt_search_paths: vec![
                "/usr/lib/i386-linux-gnu".to_string(),
                "/usr/i686-linux-gnu/lib".to_string(),
                "/usr/lib32".to_string(),
                "/usr/lib".to_string(),
            ],
            lib_search_paths: vec![
                "/usr/lib/i386-linux-gnu".to_string(),
                "/usr/i686-linux-gnu/lib".to_string(),
                "/usr/lib32".to_string(),
                "/usr/lib".to_string(),
            ],
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        }
    }

    /// Creates a `TargetConfig` for the **AArch64** (ARMv8-A 64-bit) architecture.
    ///
    /// AAPCS64 ABI, ELF64, 8-byte pointers, 31 GPRs (x0–x30),
    /// 32 SIMD/FP registers (v0–v31).
    pub fn aarch64() -> Self {
        TargetConfig {
            arch: Architecture::Aarch64,
            triple: "aarch64-linux-gnu".to_string(),
            pointer_size: 8,
            long_size: 8,
            long_double_size: 16, // IEEE 754 binary128 quad precision
            elf_class: ElfClass::Elf64,
            endianness: Endianness::Little,
            abi: AbiVariant::Aapcs64,
            max_alignment: 16,
            stack_alignment: 16,
            elf_machine: 183, // EM_AARCH64
            elf_osabi: 0,
            gpr_count: 31, // x0–x30 (x31 is SP/ZR, not a GPR)
            fpr_count: 32, // v0–v31
            crt_search_paths: vec![
                "/usr/lib/aarch64-linux-gnu".to_string(),
                "/usr/aarch64-linux-gnu/lib".to_string(),
                "/usr/lib".to_string(),
            ],
            lib_search_paths: vec![
                "/usr/lib/aarch64-linux-gnu".to_string(),
                "/usr/aarch64-linux-gnu/lib".to_string(),
                "/usr/lib".to_string(),
            ],
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        }
    }

    /// Creates a `TargetConfig` for the **RISC-V 64-bit** (RV64GC) architecture.
    ///
    /// LP64D ABI, ELF64, 8-byte pointers, 32 GPRs (x0–x31),
    /// 32 FP registers (f0–f31).
    pub fn riscv64() -> Self {
        TargetConfig {
            arch: Architecture::Riscv64,
            triple: "riscv64-linux-gnu".to_string(),
            pointer_size: 8,
            long_size: 8,
            long_double_size: 16, // IEEE 754 binary128 quad precision
            elf_class: ElfClass::Elf64,
            endianness: Endianness::Little,
            abi: AbiVariant::Lp64d,
            max_alignment: 16,
            stack_alignment: 16,
            elf_machine: 243, // EM_RISCV
            elf_osabi: 0,
            gpr_count: 32, // x0–x31
            fpr_count: 32, // f0–f31
            crt_search_paths: vec![
                "/usr/lib/riscv64-linux-gnu".to_string(),
                "/usr/riscv64-linux-gnu/lib".to_string(),
                "/usr/lib".to_string(),
            ],
            lib_search_paths: vec![
                "/usr/lib/riscv64-linux-gnu".to_string(),
                "/usr/riscv64-linux-gnu/lib".to_string(),
                "/usr/lib".to_string(),
            ],
            retpoline: false,
            cf_protection: false,
            pic: false,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
        }
    }
}

// ---------------------------------------------------------------------------
// TargetConfig — C type size query methods
// ---------------------------------------------------------------------------

impl TargetConfig {
    /// Size of the C `char` type in bytes (always 1 on all Linux targets).
    #[inline]
    pub fn char_size(&self) -> u32 {
        1
    }

    /// Size of the C `short` type in bytes (always 2 on all Linux targets).
    #[inline]
    pub fn short_size(&self) -> u32 {
        2
    }

    /// Size of the C `int` type in bytes (always 4 on all Linux targets).
    #[inline]
    pub fn int_size(&self) -> u32 {
        4
    }

    /// Size of the C `long` type in bytes.
    ///
    /// Returns 4 on i686 (ILP32) and 8 on x86-64, AArch64, and RISC-V 64 (LP64).
    #[inline]
    pub fn long_size(&self) -> u32 {
        self.long_size
    }

    /// Size of the C `long long` type in bytes (always 8 on all Linux targets).
    #[inline]
    pub fn long_long_size(&self) -> u32 {
        8
    }

    /// Size of a C pointer in bytes.
    ///
    /// Returns 4 on i686 and 8 on all 64-bit targets.
    #[inline]
    pub fn pointer_size(&self) -> u32 {
        self.pointer_size
    }

    /// Size of the C `float` type in bytes (always 4, IEEE 754 binary32).
    #[inline]
    pub fn float_size(&self) -> u32 {
        4
    }

    /// Size of the C `double` type in bytes (always 8, IEEE 754 binary64).
    #[inline]
    pub fn double_size(&self) -> u32 {
        8
    }

    /// Size of the C `long double` type in bytes (target-specific).
    ///
    /// - x86-64: 16 bytes (80-bit x87 padded to 16)
    /// - i686: 12 bytes (80-bit x87 padded to 12)
    /// - AArch64: 16 bytes (IEEE 754 binary128)
    /// - RISC-V 64: 16 bytes (IEEE 754 binary128)
    #[inline]
    pub fn long_double_size(&self) -> u32 {
        self.long_double_size
    }

    /// Size of `size_t` in bytes (same as pointer size per C standard).
    #[inline]
    pub fn size_t_size(&self) -> u32 {
        self.pointer_size
    }

    /// Size of `ptrdiff_t` in bytes (same as pointer size per C standard).
    #[inline]
    pub fn ptrdiff_t_size(&self) -> u32 {
        self.pointer_size
    }

    /// Computes the natural alignment for a type of the given `size` in bytes.
    ///
    /// The alignment is `min(size, max_alignment)`, following the target's
    /// maximum alignment constraint. For example, on a target with
    /// `max_alignment = 16`, a 4-byte `int` aligns to 4 bytes, while a
    /// 32-byte struct aligns to at most 16 bytes.
    #[inline]
    pub fn alignment_of(&self, size: u32) -> u32 {
        std::cmp::min(size, self.max_alignment)
    }

    /// Returns `true` if this target uses 64-bit pointers.
    #[inline]
    pub fn is_64bit(&self) -> bool {
        self.pointer_size == 8
    }

    /// Returns `true` if this target uses 32-bit pointers.
    #[inline]
    pub fn is_32bit(&self) -> bool {
        self.pointer_size == 4
    }

    /// Returns `true` if retpoline generation is enabled (`-mretpoline`).
    ///
    /// Retpoline replaces indirect branch instructions with speculative-safe
    /// thunk sequences to mitigate Spectre variant 2 attacks. Currently
    /// supported only on the x86-64 backend.
    #[inline]
    pub fn retpoline_enabled(&self) -> bool {
        self.retpoline
    }

    /// Returns `true` if Intel CET control-flow protection is enabled
    /// (`-fcf-protection`).
    ///
    /// When enabled, the x86-64 backend inserts `endbr64` instructions at
    /// all indirect branch targets for forward-edge CFI compatibility.
    #[inline]
    pub fn cf_protection_enabled(&self) -> bool {
        self.cf_protection
    }

    /// Returns `true` if position-independent code generation is enabled
    /// (`-fPIC`).
    ///
    /// When enabled, the code generator emits position-independent sequences
    /// using GOT-relative addressing and PLT stubs for all four architectures.
    #[inline]
    pub fn pic_enabled(&self) -> bool {
        self.pic
    }
}

// ---------------------------------------------------------------------------
// Target triple parsing
// ---------------------------------------------------------------------------

/// Parses a target triple string and returns the corresponding [`TargetConfig`].
///
/// The parser extracts the architecture component (the first `-`-delimited field)
/// and matches it against supported architectures. The remaining triple components
/// (vendor, OS, environment) are accepted but not independently validated, as the
/// compiler only targets Linux/GNU across all four architectures.
///
/// # Supported Triples
///
/// | Input                          | Architecture |
/// |--------------------------------|-------------|
/// | `x86_64-linux-gnu`             | x86-64      |
/// | `x86_64-unknown-linux-gnu`     | x86-64      |
/// | `i686-linux-gnu`               | i686        |
/// | `i686-unknown-linux-gnu`       | i686        |
/// | `i386-linux-gnu`               | i686 (alias)|
/// | `aarch64-linux-gnu`            | AArch64     |
/// | `aarch64-unknown-linux-gnu`    | AArch64     |
/// | `arm64-linux-gnu`              | AArch64 (alias) |
/// | `riscv64-linux-gnu`            | RISC-V 64   |
/// | `riscv64-unknown-linux-gnu`    | RISC-V 64   |
/// | `riscv64gc-unknown-linux-gnu`  | RISC-V 64   |
///
/// # Errors
///
/// Returns `Err(String)` with a descriptive message if:
/// - The triple string is empty.
/// - The architecture component does not match any supported target.
///
/// # Examples
///
/// ```ignore
/// let config = parse_target("x86_64-linux-gnu").unwrap();
/// assert_eq!(config.arch, Architecture::X86_64);
/// assert_eq!(config.pointer_size, 8);
///
/// let err = parse_target("mips-linux-gnu");
/// assert!(err.is_err());
/// ```
pub fn parse_target(triple: &str) -> Result<TargetConfig, String> {
    if triple.is_empty() {
        return Err("invalid target triple: empty string".to_string());
    }

    // Extract the architecture component: everything before the first '-'.
    // For triples like "x86_64-linux-gnu" or "riscv64gc-unknown-linux-gnu",
    // the first field uniquely identifies the architecture.
    let arch_str = triple.split('-').next().unwrap_or("");

    match arch_str {
        "x86_64" => {
            let mut config = TargetConfig::x86_64();
            config.triple = triple.to_string();
            Ok(config)
        }
        "i686" | "i386" => {
            let mut config = TargetConfig::i686();
            config.triple = triple.to_string();
            Ok(config)
        }
        "aarch64" | "arm64" => {
            let mut config = TargetConfig::aarch64();
            config.triple = triple.to_string();
            Ok(config)
        }
        "riscv64" | "riscv64gc" => {
            let mut config = TargetConfig::riscv64();
            config.triple = triple.to_string();
            Ok(config)
        }
        _ => Err(format!(
            "unsupported target architecture '{}'. Supported: x86_64, i686, aarch64, riscv64",
            arch_str
        )),
    }
}

// ---------------------------------------------------------------------------
// Host architecture detection
// ---------------------------------------------------------------------------

/// Detects the host machine's architecture at compile time and returns
/// a default [`TargetConfig`] for native compilation.
///
/// This function uses Rust's `#[cfg(target_arch = "...")]` conditional
/// compilation attributes to select the appropriate target configuration.
/// When the host architecture is not one of the four supported targets,
/// it falls back to x86-64 as a reasonable default.
///
/// # Returns
///
/// A `TargetConfig` matching the host architecture.
pub fn detect_host() -> TargetConfig {
    #[cfg(target_arch = "x86_64")]
    {
        return TargetConfig::x86_64();
    }

    #[cfg(target_arch = "x86")]
    {
        return TargetConfig::i686();
    }

    #[cfg(target_arch = "aarch64")]
    {
        return TargetConfig::aarch64();
    }

    #[cfg(target_arch = "riscv64")]
    {
        return TargetConfig::riscv64();
    }

    // Fallback for architectures that are not among the four supported targets.
    // Default to x86-64 as the most common compilation host.
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    {
        return TargetConfig::x86_64();
    }
}

// ---------------------------------------------------------------------------
// Combined target resolution
// ---------------------------------------------------------------------------

/// Resolves the compilation target by combining CLI input with host detection.
///
/// If `target_option` is `Some(triple)`, parses the given triple string.
/// If `target_option` is `None`, falls back to the host architecture via
/// [`detect_host`].
///
/// This is the primary entry point called from `main.rs` after CLI parsing.
///
/// # Errors
///
/// Returns `Err(String)` if a target triple is provided but cannot be parsed.
///
/// # Examples
///
/// ```ignore
/// // Explicit target selection:
/// let config = resolve_target(Some("aarch64-linux-gnu")).unwrap();
/// assert_eq!(config.arch, Architecture::Aarch64);
///
/// // Default to host:
/// let config = resolve_target(None).unwrap();
/// ```
pub fn resolve_target(target_option: Option<&str>) -> Result<TargetConfig, String> {
    match target_option {
        Some(triple) => parse_target(triple),
        None => Ok(detect_host()),
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Target triple parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_x86_64_linux_gnu() {
        let config = parse_target("x86_64-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::X86_64);
        assert_eq!(config.triple, "x86_64-linux-gnu");
        assert_eq!(config.pointer_size, 8);
        assert_eq!(config.long_size, 8);
        assert_eq!(config.elf_class, ElfClass::Elf64);
        assert_eq!(config.abi, AbiVariant::SystemVAmd64);
        assert_eq!(config.elf_machine, 62);
    }

    #[test]
    fn test_parse_x86_64_unknown_linux_gnu() {
        let config = parse_target("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::X86_64);
        assert_eq!(config.triple, "x86_64-unknown-linux-gnu");
        assert_eq!(config.pointer_size, 8);
    }

    #[test]
    fn test_parse_i686_linux_gnu() {
        let config = parse_target("i686-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::I686);
        assert_eq!(config.triple, "i686-linux-gnu");
        assert_eq!(config.pointer_size, 4);
        assert_eq!(config.long_size, 4);
        assert_eq!(config.elf_class, ElfClass::Elf32);
        assert_eq!(config.abi, AbiVariant::SystemVi386Cdecl);
        assert_eq!(config.elf_machine, 3);
    }

    #[test]
    fn test_parse_i386_alias() {
        let config = parse_target("i386-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::I686);
        assert_eq!(config.triple, "i386-linux-gnu");
        assert_eq!(config.pointer_size, 4);
    }

    #[test]
    fn test_parse_aarch64_linux_gnu() {
        let config = parse_target("aarch64-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::Aarch64);
        assert_eq!(config.triple, "aarch64-linux-gnu");
        assert_eq!(config.pointer_size, 8);
        assert_eq!(config.long_size, 8);
        assert_eq!(config.elf_class, ElfClass::Elf64);
        assert_eq!(config.abi, AbiVariant::Aapcs64);
        assert_eq!(config.elf_machine, 183);
    }

    #[test]
    fn test_parse_arm64_alias() {
        let config = parse_target("arm64-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::Aarch64);
        assert_eq!(config.triple, "arm64-linux-gnu");
        assert_eq!(config.pointer_size, 8);
    }

    #[test]
    fn test_parse_riscv64_linux_gnu() {
        let config = parse_target("riscv64-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::Riscv64);
        assert_eq!(config.triple, "riscv64-linux-gnu");
        assert_eq!(config.pointer_size, 8);
        assert_eq!(config.long_size, 8);
        assert_eq!(config.elf_class, ElfClass::Elf64);
        assert_eq!(config.abi, AbiVariant::Lp64d);
        assert_eq!(config.elf_machine, 243);
    }

    #[test]
    fn test_parse_riscv64gc_unknown_linux_gnu() {
        let config = parse_target("riscv64gc-unknown-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::Riscv64);
        assert_eq!(config.triple, "riscv64gc-unknown-linux-gnu");
        assert_eq!(config.pointer_size, 8);
    }

    #[test]
    fn test_parse_unsupported_architecture() {
        let err = parse_target("mips-linux-gnu");
        assert!(err.is_err());
        let msg = err.unwrap_err();
        assert!(msg.contains("unsupported target architecture"));
        assert!(msg.contains("mips"));
    }

    #[test]
    fn test_parse_empty_string() {
        let err = parse_target("");
        assert!(err.is_err());
        let msg = err.unwrap_err();
        assert!(msg.contains("empty string"));
    }

    #[test]
    fn test_parse_unknown_triple() {
        let err = parse_target("sparc-sun-solaris");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("sparc"));
    }

    // -----------------------------------------------------------------------
    // TargetConfig field validation per architecture
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_config_details() {
        let cfg = TargetConfig::x86_64();
        assert_eq!(cfg.arch, Architecture::X86_64);
        assert_eq!(cfg.triple, "x86_64-linux-gnu");
        assert_eq!(cfg.pointer_size, 8);
        assert_eq!(cfg.long_size, 8);
        assert_eq!(cfg.long_double_size, 16);
        assert_eq!(cfg.elf_class, ElfClass::Elf64);
        assert_eq!(cfg.endianness, Endianness::Little);
        assert_eq!(cfg.abi, AbiVariant::SystemVAmd64);
        assert_eq!(cfg.max_alignment, 16);
        assert_eq!(cfg.stack_alignment, 16);
        assert_eq!(cfg.elf_machine, 62);
        assert_eq!(cfg.elf_osabi, 0);
        assert_eq!(cfg.gpr_count, 16);
        assert_eq!(cfg.fpr_count, 16);
        assert!(!cfg.crt_search_paths.is_empty());
        assert!(!cfg.lib_search_paths.is_empty());
    }

    #[test]
    fn test_i686_config_details() {
        let cfg = TargetConfig::i686();
        assert_eq!(cfg.arch, Architecture::I686);
        assert_eq!(cfg.triple, "i686-linux-gnu");
        assert_eq!(cfg.pointer_size, 4);
        assert_eq!(cfg.long_size, 4);
        assert_eq!(cfg.long_double_size, 12);
        assert_eq!(cfg.elf_class, ElfClass::Elf32);
        assert_eq!(cfg.endianness, Endianness::Little);
        assert_eq!(cfg.abi, AbiVariant::SystemVi386Cdecl);
        assert_eq!(cfg.max_alignment, 16);
        assert_eq!(cfg.stack_alignment, 16);
        assert_eq!(cfg.elf_machine, 3);
        assert_eq!(cfg.elf_osabi, 0);
        assert_eq!(cfg.gpr_count, 8);
        assert_eq!(cfg.fpr_count, 8);
        assert!(!cfg.crt_search_paths.is_empty());
        assert!(!cfg.lib_search_paths.is_empty());
    }

    #[test]
    fn test_aarch64_config_details() {
        let cfg = TargetConfig::aarch64();
        assert_eq!(cfg.arch, Architecture::Aarch64);
        assert_eq!(cfg.triple, "aarch64-linux-gnu");
        assert_eq!(cfg.pointer_size, 8);
        assert_eq!(cfg.long_size, 8);
        assert_eq!(cfg.long_double_size, 16);
        assert_eq!(cfg.elf_class, ElfClass::Elf64);
        assert_eq!(cfg.endianness, Endianness::Little);
        assert_eq!(cfg.abi, AbiVariant::Aapcs64);
        assert_eq!(cfg.max_alignment, 16);
        assert_eq!(cfg.stack_alignment, 16);
        assert_eq!(cfg.elf_machine, 183);
        assert_eq!(cfg.elf_osabi, 0);
        assert_eq!(cfg.gpr_count, 31);
        assert_eq!(cfg.fpr_count, 32);
        assert!(!cfg.crt_search_paths.is_empty());
        assert!(!cfg.lib_search_paths.is_empty());
    }

    #[test]
    fn test_riscv64_config_details() {
        let cfg = TargetConfig::riscv64();
        assert_eq!(cfg.arch, Architecture::Riscv64);
        assert_eq!(cfg.triple, "riscv64-linux-gnu");
        assert_eq!(cfg.pointer_size, 8);
        assert_eq!(cfg.long_size, 8);
        assert_eq!(cfg.long_double_size, 16);
        assert_eq!(cfg.elf_class, ElfClass::Elf64);
        assert_eq!(cfg.endianness, Endianness::Little);
        assert_eq!(cfg.abi, AbiVariant::Lp64d);
        assert_eq!(cfg.max_alignment, 16);
        assert_eq!(cfg.stack_alignment, 16);
        assert_eq!(cfg.elf_machine, 243);
        assert_eq!(cfg.elf_osabi, 0);
        assert_eq!(cfg.gpr_count, 32);
        assert_eq!(cfg.fpr_count, 32);
        assert!(!cfg.crt_search_paths.is_empty());
        assert!(!cfg.lib_search_paths.is_empty());
    }

    #[test]
    fn test_all_targets_little_endian() {
        assert_eq!(TargetConfig::x86_64().endianness, Endianness::Little);
        assert_eq!(TargetConfig::i686().endianness, Endianness::Little);
        assert_eq!(TargetConfig::aarch64().endianness, Endianness::Little);
        assert_eq!(TargetConfig::riscv64().endianness, Endianness::Little);
    }

    // -----------------------------------------------------------------------
    // C type size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_char_size_all_targets() {
        assert_eq!(TargetConfig::x86_64().char_size(), 1);
        assert_eq!(TargetConfig::i686().char_size(), 1);
        assert_eq!(TargetConfig::aarch64().char_size(), 1);
        assert_eq!(TargetConfig::riscv64().char_size(), 1);
    }

    #[test]
    fn test_short_size_all_targets() {
        assert_eq!(TargetConfig::x86_64().short_size(), 2);
        assert_eq!(TargetConfig::i686().short_size(), 2);
        assert_eq!(TargetConfig::aarch64().short_size(), 2);
        assert_eq!(TargetConfig::riscv64().short_size(), 2);
    }

    #[test]
    fn test_int_size_all_targets() {
        assert_eq!(TargetConfig::x86_64().int_size(), 4);
        assert_eq!(TargetConfig::i686().int_size(), 4);
        assert_eq!(TargetConfig::aarch64().int_size(), 4);
        assert_eq!(TargetConfig::riscv64().int_size(), 4);
    }

    #[test]
    fn test_long_size_varies() {
        assert_eq!(TargetConfig::x86_64().long_size(), 8);
        assert_eq!(TargetConfig::i686().long_size(), 4);
        assert_eq!(TargetConfig::aarch64().long_size(), 8);
        assert_eq!(TargetConfig::riscv64().long_size(), 8);
    }

    #[test]
    fn test_long_long_size_all_targets() {
        assert_eq!(TargetConfig::x86_64().long_long_size(), 8);
        assert_eq!(TargetConfig::i686().long_long_size(), 8);
        assert_eq!(TargetConfig::aarch64().long_long_size(), 8);
        assert_eq!(TargetConfig::riscv64().long_long_size(), 8);
    }

    #[test]
    fn test_pointer_size_varies() {
        assert_eq!(TargetConfig::x86_64().pointer_size(), 8);
        assert_eq!(TargetConfig::i686().pointer_size(), 4);
        assert_eq!(TargetConfig::aarch64().pointer_size(), 8);
        assert_eq!(TargetConfig::riscv64().pointer_size(), 8);
    }

    #[test]
    fn test_float_size_all_targets() {
        assert_eq!(TargetConfig::x86_64().float_size(), 4);
        assert_eq!(TargetConfig::i686().float_size(), 4);
        assert_eq!(TargetConfig::aarch64().float_size(), 4);
        assert_eq!(TargetConfig::riscv64().float_size(), 4);
    }

    #[test]
    fn test_double_size_all_targets() {
        assert_eq!(TargetConfig::x86_64().double_size(), 8);
        assert_eq!(TargetConfig::i686().double_size(), 8);
        assert_eq!(TargetConfig::aarch64().double_size(), 8);
        assert_eq!(TargetConfig::riscv64().double_size(), 8);
    }

    #[test]
    fn test_long_double_size_varies() {
        assert_eq!(TargetConfig::x86_64().long_double_size(), 16);
        assert_eq!(TargetConfig::i686().long_double_size(), 12);
        assert_eq!(TargetConfig::aarch64().long_double_size(), 16);
        assert_eq!(TargetConfig::riscv64().long_double_size(), 16);
    }

    #[test]
    fn test_size_t_equals_pointer_size() {
        assert_eq!(TargetConfig::x86_64().size_t_size(), 8);
        assert_eq!(TargetConfig::i686().size_t_size(), 4);
        assert_eq!(TargetConfig::aarch64().size_t_size(), 8);
        assert_eq!(TargetConfig::riscv64().size_t_size(), 8);
    }

    #[test]
    fn test_ptrdiff_t_equals_pointer_size() {
        assert_eq!(TargetConfig::x86_64().ptrdiff_t_size(), 8);
        assert_eq!(TargetConfig::i686().ptrdiff_t_size(), 4);
        assert_eq!(TargetConfig::aarch64().ptrdiff_t_size(), 8);
        assert_eq!(TargetConfig::riscv64().ptrdiff_t_size(), 8);
    }

    // -----------------------------------------------------------------------
    // Alignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignment_of_small_type() {
        let cfg = TargetConfig::x86_64();
        assert_eq!(cfg.alignment_of(1), 1); // char
        assert_eq!(cfg.alignment_of(2), 2); // short
        assert_eq!(cfg.alignment_of(4), 4); // int
        assert_eq!(cfg.alignment_of(8), 8); // long / pointer
    }

    #[test]
    fn test_alignment_of_large_type_capped() {
        let cfg = TargetConfig::x86_64();
        // Types larger than max_alignment are capped
        assert_eq!(cfg.alignment_of(32), 16);
        assert_eq!(cfg.alignment_of(64), 16);
        assert_eq!(cfg.alignment_of(16), 16);
    }

    #[test]
    fn test_alignment_of_zero() {
        let cfg = TargetConfig::x86_64();
        assert_eq!(cfg.alignment_of(0), 0);
    }

    // -----------------------------------------------------------------------
    // Bitness tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_64bit() {
        assert!(TargetConfig::x86_64().is_64bit());
        assert!(!TargetConfig::i686().is_64bit());
        assert!(TargetConfig::aarch64().is_64bit());
        assert!(TargetConfig::riscv64().is_64bit());
    }

    #[test]
    fn test_is_32bit() {
        assert!(!TargetConfig::x86_64().is_32bit());
        assert!(TargetConfig::i686().is_32bit());
        assert!(!TargetConfig::aarch64().is_32bit());
        assert!(!TargetConfig::riscv64().is_32bit());
    }

    #[test]
    fn test_is_64bit_and_32bit_mutually_exclusive() {
        for cfg in &[
            TargetConfig::x86_64(),
            TargetConfig::i686(),
            TargetConfig::aarch64(),
            TargetConfig::riscv64(),
        ] {
            assert_ne!(cfg.is_64bit(), cfg.is_32bit());
        }
    }

    // -----------------------------------------------------------------------
    // Host detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_host_returns_valid_config() {
        let host = detect_host();
        // Must be one of the four supported architectures
        assert!(matches!(
            host.arch,
            Architecture::X86_64
                | Architecture::I686
                | Architecture::Aarch64
                | Architecture::Riscv64
        ));
        // Basic sanity: pointer size must be either 4 or 8
        assert!(host.pointer_size == 4 || host.pointer_size == 8);
        // Triple must not be empty
        assert!(!host.triple.is_empty());
    }

    #[test]
    fn test_detect_host_on_x86_64() {
        // On our CI / build host (x86-64), detect_host() should return x86-64
        let host = detect_host();
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(host.arch, Architecture::X86_64);
            assert_eq!(host.pointer_size, 8);
        }
        // On other hosts, just verify it returns a valid config (covered above)
        let _ = host;
    }

    // -----------------------------------------------------------------------
    // resolve_target tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_target_with_explicit_triple() {
        let config = resolve_target(Some("x86_64-linux-gnu")).unwrap();
        assert_eq!(config.arch, Architecture::X86_64);
        assert_eq!(config.pointer_size, 8);
    }

    #[test]
    fn test_resolve_target_with_aarch64() {
        let config = resolve_target(Some("aarch64-linux-gnu")).unwrap();
        assert_eq!(config.arch, Architecture::Aarch64);
    }

    #[test]
    fn test_resolve_target_with_i686() {
        let config = resolve_target(Some("i686-linux-gnu")).unwrap();
        assert_eq!(config.arch, Architecture::I686);
    }

    #[test]
    fn test_resolve_target_with_riscv64() {
        let config = resolve_target(Some("riscv64-linux-gnu")).unwrap();
        assert_eq!(config.arch, Architecture::Riscv64);
    }

    #[test]
    fn test_resolve_target_none_defaults_to_host() {
        let config = resolve_target(None).unwrap();
        let host = detect_host();
        assert_eq!(config.arch, host.arch);
        assert_eq!(config.pointer_size, host.pointer_size);
    }

    #[test]
    fn test_resolve_target_unsupported_returns_error() {
        let result = resolve_target(Some("mips-linux-gnu"));
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_architecture_display() {
        assert_eq!(format!("{}", Architecture::X86_64), "x86_64");
        assert_eq!(format!("{}", Architecture::I686), "i686");
        assert_eq!(format!("{}", Architecture::Aarch64), "aarch64");
        assert_eq!(format!("{}", Architecture::Riscv64), "riscv64");
    }

    #[test]
    fn test_elf_class_display() {
        assert_eq!(format!("{}", ElfClass::Elf32), "ELF32");
        assert_eq!(format!("{}", ElfClass::Elf64), "ELF64");
    }

    #[test]
    fn test_endianness_display() {
        assert_eq!(format!("{}", Endianness::Little), "little-endian");
        assert_eq!(format!("{}", Endianness::Big), "big-endian");
    }

    #[test]
    fn test_abi_variant_display() {
        assert_eq!(format!("{}", AbiVariant::SystemVAmd64), "System V AMD64");
        assert_eq!(
            format!("{}", AbiVariant::SystemVi386Cdecl),
            "System V i386 cdecl"
        );
        assert_eq!(format!("{}", AbiVariant::Aapcs64), "AAPCS64");
        assert_eq!(format!("{}", AbiVariant::Lp64d), "LP64D");
    }

    // -----------------------------------------------------------------------
    // Clone / Copy / PartialEq trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_architecture_clone_copy() {
        let a = Architecture::X86_64;
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_target_config_clone() {
        let cfg = TargetConfig::x86_64();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.arch, cfg2.arch);
        assert_eq!(cfg.triple, cfg2.triple);
        assert_eq!(cfg.pointer_size, cfg2.pointer_size);
        assert_eq!(cfg.elf_machine, cfg2.elf_machine);
    }

    // -----------------------------------------------------------------------
    // CRT and library path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_crt_paths_contain_expected() {
        let cfg = TargetConfig::x86_64();
        assert!(cfg
            .crt_search_paths
            .iter()
            .any(|p| p.contains("x86_64-linux-gnu")));
    }

    #[test]
    fn test_i686_crt_paths_contain_expected() {
        let cfg = TargetConfig::i686();
        assert!(cfg
            .crt_search_paths
            .iter()
            .any(|p| p.contains("i386-linux-gnu") || p.contains("i686-linux-gnu")));
    }

    #[test]
    fn test_aarch64_crt_paths_contain_expected() {
        let cfg = TargetConfig::aarch64();
        assert!(cfg
            .crt_search_paths
            .iter()
            .any(|p| p.contains("aarch64-linux-gnu")));
    }

    #[test]
    fn test_riscv64_crt_paths_contain_expected() {
        let cfg = TargetConfig::riscv64();
        assert!(cfg
            .crt_search_paths
            .iter()
            .any(|p| p.contains("riscv64-linux-gnu")));
    }

    // -----------------------------------------------------------------------
    // ELF machine constant verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf_machine_constants() {
        assert_eq!(TargetConfig::x86_64().elf_machine, 62); // EM_X86_64
        assert_eq!(TargetConfig::i686().elf_machine, 3); // EM_386
        assert_eq!(TargetConfig::aarch64().elf_machine, 183); // EM_AARCH64
        assert_eq!(TargetConfig::riscv64().elf_machine, 243); // EM_RISCV
    }

    #[test]
    fn test_elf_osabi_all_zero() {
        assert_eq!(TargetConfig::x86_64().elf_osabi, 0);
        assert_eq!(TargetConfig::i686().elf_osabi, 0);
        assert_eq!(TargetConfig::aarch64().elf_osabi, 0);
        assert_eq!(TargetConfig::riscv64().elf_osabi, 0);
    }

    // -----------------------------------------------------------------------
    // Register count tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_counts() {
        let x86 = TargetConfig::x86_64();
        assert_eq!(x86.gpr_count, 16);
        assert_eq!(x86.fpr_count, 16);

        let i686 = TargetConfig::i686();
        assert_eq!(i686.gpr_count, 8);
        assert_eq!(i686.fpr_count, 8);

        let aarch64 = TargetConfig::aarch64();
        assert_eq!(aarch64.gpr_count, 31);
        assert_eq!(aarch64.fpr_count, 32);

        let riscv64 = TargetConfig::riscv64();
        assert_eq!(riscv64.gpr_count, 32);
        assert_eq!(riscv64.fpr_count, 32);
    }

    // -----------------------------------------------------------------------
    // Edge case parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_triple_with_extra_components() {
        // Triples with additional components should still parse by first field
        let config = parse_target("x86_64-pc-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::X86_64);
    }

    #[test]
    fn test_parse_riscv64_unknown_linux_gnu() {
        let config = parse_target("riscv64-unknown-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::Riscv64);
    }

    #[test]
    fn test_parse_i686_unknown_linux_gnu() {
        let config = parse_target("i686-unknown-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::I686);
    }

    #[test]
    fn test_parse_aarch64_unknown_linux_gnu() {
        let config = parse_target("aarch64-unknown-linux-gnu").unwrap();
        assert_eq!(config.arch, Architecture::Aarch64);
    }

    #[test]
    fn test_parse_single_word_architecture() {
        // Just the architecture name without any triple separators
        let config = parse_target("x86_64").unwrap();
        assert_eq!(config.arch, Architecture::X86_64);
        assert_eq!(config.triple, "x86_64");
    }

    #[test]
    fn test_parse_preserves_original_triple() {
        let config = parse_target("riscv64gc-unknown-linux-gnu").unwrap();
        assert_eq!(config.triple, "riscv64gc-unknown-linux-gnu");
        assert_eq!(config.arch, Architecture::Riscv64);
    }

    // -----------------------------------------------------------------------
    // Stack alignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_targets_16_byte_stack_alignment() {
        assert_eq!(TargetConfig::x86_64().stack_alignment, 16);
        assert_eq!(TargetConfig::i686().stack_alignment, 16);
        assert_eq!(TargetConfig::aarch64().stack_alignment, 16);
        assert_eq!(TargetConfig::riscv64().stack_alignment, 16);
    }

    #[test]
    fn test_all_targets_16_byte_max_alignment() {
        assert_eq!(TargetConfig::x86_64().max_alignment, 16);
        assert_eq!(TargetConfig::i686().max_alignment, 16);
        assert_eq!(TargetConfig::aarch64().max_alignment, 16);
        assert_eq!(TargetConfig::riscv64().max_alignment, 16);
    }
}
