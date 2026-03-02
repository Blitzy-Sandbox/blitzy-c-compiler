//! # Code Generation Module
//!
//! This module implements native machine code generation for four architectures:
//! x86-64, i686, AArch64, and RISC-V 64. Each backend provides an integrated
//! assembler that directly encodes machine instructions without reliance on
//! external `as` or `gas`.
//!
//! ## Architecture
//!
//! All backends implement the [`CodeGen`] trait, which defines:
//! - `fn generate(&self, module: &Module, target: &TargetConfig) -> Result<ObjectCode, CodeGenError>`
//! - `fn target_arch(&self) -> Architecture`
//!
//! The [`generate_code`] top-level function dispatches to the appropriate backend
//! based on the [`TargetConfig`] architecture field.
//!
//! ## Data Flow
//!
//! ```text
//! Optimized IR (from passes) → CodeGen::generate() → ObjectCode
//!                                                      │
//!                                                      ├── machine code bytes (Vec<u8>)
//!                                                      ├── relocation entries
//!                                                      ├── symbol definitions
//!                                                      └── section assignments
//! ```
//!
//! ## Backends
//!
//! - `x86_64`: System V AMD64 ABI, REX-prefix encoding, security hardening
//!   (retpoline, CET endbr64, stack probing)
//! - `i686`: cdecl ABI, 32-bit encoding, register pair arithmetic for 64-bit ops
//! - `aarch64`: AAPCS64 ABI, fixed-width 32-bit instruction encoding
//! - `riscv64`: LP64D ABI, variable-length encoding (32-bit base + 16-bit compressed)
//!
//! ## Zero External Dependencies
//!
//! This module and all its submodules use only the Rust standard library (`std`).
//! No external crates are imported, per project constraint.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Shared linear scan register allocator used by all four backends.
/// Provides physical register assignment, spill code insertion, and
/// live interval computation parameterised by the target register file.
pub mod regalloc;

/// x86-64 backend: System V AMD64 ABI, REX-prefix instruction encoding,
/// security hardening (retpoline, CET `endbr64`, stack probing).
pub mod x86_64;

/// i686 (32-bit x86) backend: cdecl ABI, legacy instruction encoding
/// without REX prefix, 64-bit arithmetic via register pairs.
pub mod i686;

/// AArch64 (ARM 64-bit) backend: AAPCS64 ABI, fixed-width 32-bit
/// instruction encoding with barrel shifter operands.
pub mod aarch64;

/// RISC-V 64-bit backend: LP64D ABI, RV64GC instruction set with
/// variable-length encoding (32-bit base, optional 16-bit compressed).
pub mod riscv64;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::driver::target::TargetConfig;
use crate::ir::Module;

// SourceLocation is used in MachineInstr.loc for debug info correlation.
// Imported from the common module re-export (originally from source_map).
#[allow(unused_imports)]
use crate::common::SourceLocation;

use std::fmt;

// ---------------------------------------------------------------------------
// Re-exports from driver::target — Architecture enum
// ---------------------------------------------------------------------------
// The Architecture enum is canonically defined in src/driver/target.rs.
// We re-export it here so that all codegen consumers (backends, linker,
// debug) can import it from crate::codegen directly.

pub use crate::driver::target::Architecture;

// ---------------------------------------------------------------------------
// Re-exports from regalloc — shared register allocation types
// ---------------------------------------------------------------------------

pub use regalloc::{AllocationResult, LiveInterval, PhysReg, RegClass, RegisterInfo};

// ---------------------------------------------------------------------------
// CodeGenError — error type for code generation failures
// ---------------------------------------------------------------------------

/// Errors that can occur during code generation.
///
/// Each variant carries a human-readable description suitable for inclusion
/// in GCC-compatible diagnostic messages emitted on stderr.
#[derive(Debug)]
pub enum CodeGenError {
    /// An IR instruction or construct is not supported by the target backend.
    /// Contains a description of the unsupported instruction and the target.
    UnsupportedInstruction(String),

    /// The register allocator could not find a valid assignment.
    /// Typically caused by excessive register pressure that spilling
    /// cannot resolve.
    RegisterAllocationFailed(String),

    /// A machine instruction could not be encoded into the target's
    /// binary format. For example, an immediate value that exceeds the
    /// field width, or an invalid operand combination.
    EncodingError(String),

    /// A violation of the target ABI was detected. This includes invalid
    /// calling convention usage, incorrect stack alignment, or unsupported
    /// argument passing configurations.
    AbiError(String),

    /// An internal compiler error that indicates a bug in the code generator
    /// rather than invalid user input. Should never occur in a correct
    /// implementation.
    InternalError(String),
}

impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedInstruction(msg) => {
                write!(f, "unsupported instruction: {}", msg)
            }
            Self::RegisterAllocationFailed(msg) => {
                write!(f, "register allocation failed: {}", msg)
            }
            Self::EncodingError(msg) => {
                write!(f, "encoding error: {}", msg)
            }
            Self::AbiError(msg) => {
                write!(f, "ABI error: {}", msg)
            }
            Self::InternalError(msg) => {
                write!(f, "internal error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CodeGenError {}

// ---------------------------------------------------------------------------
// SectionType, SectionFlags, Section — machine code section descriptors
// ---------------------------------------------------------------------------

/// The type of an ELF section produced by code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionType {
    /// `.text` — executable code
    Text,
    /// `.data` — initialised read-write data
    Data,
    /// `.rodata` — read-only data (string literals, constants)
    Rodata,
    /// `.bss` — uninitialised data (zero-filled at load time)
    Bss,
    /// Custom section identified by a numeric tag.
    Custom(u32),
}

/// Permission and allocation flags for a section.
///
/// These map directly to ELF section header flags:
/// - `SHF_WRITE` → `writable`
/// - `SHF_EXECINSTR` → `executable`
/// - `SHF_ALLOC` → `allocatable`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionFlags {
    /// Section is writable at runtime (`SHF_WRITE`).
    pub writable: bool,
    /// Section contains executable machine code (`SHF_EXECINSTR`).
    pub executable: bool,
    /// Section occupies memory at runtime (`SHF_ALLOC`).
    pub allocatable: bool,
}

impl SectionFlags {
    /// Default flags for a `.text` section: executable, allocatable,
    /// not writable.
    pub fn text() -> Self {
        Self {
            writable: false,
            executable: true,
            allocatable: true,
        }
    }

    /// Default flags for a `.data` section: writable, allocatable,
    /// not executable.
    pub fn data() -> Self {
        Self {
            writable: true,
            executable: false,
            allocatable: true,
        }
    }

    /// Default flags for a `.rodata` section: not writable, not executable,
    /// allocatable.
    pub fn rodata() -> Self {
        Self {
            writable: false,
            executable: false,
            allocatable: true,
        }
    }

    /// Default flags for a `.bss` section: writable, allocatable,
    /// not executable.
    pub fn bss() -> Self {
        Self {
            writable: true,
            executable: false,
            allocatable: true,
        }
    }
}

/// A section of machine code or data produced by code generation.
///
/// Sections are the primary output containers from the code generator.
/// The linker merges sections of the same name and type from multiple
/// compilation units into the final ELF binary.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section name (e.g. ".text", ".data", ".rodata", ".bss").
    pub name: String,
    /// Raw bytes of section content. For `.bss` sections this is empty
    /// (the linker allocates zero-filled space based on declared size).
    pub data: Vec<u8>,
    /// The semantic type of this section.
    pub section_type: SectionType,
    /// Required byte alignment for the start of this section. Must be a
    /// power of two (e.g. 1, 4, 8, 16).
    pub alignment: u32,
    /// ELF section header flags controlling permissions and allocation.
    pub flags: SectionFlags,
}

// ---------------------------------------------------------------------------
// SymbolBinding, SymbolType, SymbolVisibility, Symbol
// ---------------------------------------------------------------------------

/// ELF symbol binding: determines how the linker resolves this symbol
/// when multiple definitions exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBinding {
    /// Symbol is local to the compilation unit and invisible to the linker's
    /// cross-unit symbol resolution.
    Local,
    /// Symbol is visible across compilation units. The linker will produce
    /// an error if multiple global definitions conflict.
    Global,
    /// Like `Global`, but a strong definition from another unit takes
    /// precedence without error.
    Weak,
}

/// The kind of entity a symbol refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    /// Symbol refers to a function (executable code).
    Function,
    /// Symbol refers to a data object (variable, constant).
    Object,
    /// Symbol type is unspecified or irrelevant (e.g. external references).
    NoType,
    /// Symbol refers to a section itself (used internally by the linker).
    Section,
}

/// ELF symbol visibility: controls how the dynamic linker resolves this
/// symbol in shared library contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolVisibility {
    /// Symbol is visible according to its binding rules (default behaviour).
    Default,
    /// Symbol is not exported by the shared library; resolves only within
    /// the defining module.
    Hidden,
    /// Like `Default`, but the symbol cannot be preempted by another
    /// definition at runtime.
    Protected,
}

/// A symbol definition produced by code generation.
///
/// Symbols represent named entities (functions, global variables, section
/// markers) that the linker uses for relocation resolution and export.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// The symbol's name (e.g. "main", "_start", "global_var").
    pub name: String,
    /// Index of the section containing this symbol (into `ObjectCode.sections`).
    /// For defined symbols, this is the 0-based index into the `ObjectCode.sections`
    /// array. For undefined symbols, this value is ignored (use `is_definition` to
    /// distinguish defined from undefined symbols).
    pub section_index: usize,
    /// Byte offset of the symbol's definition within its section.
    pub offset: u64,
    /// Size of the symbol in bytes (0 if unknown or not applicable).
    pub size: u64,
    /// Symbol binding (local, global, or weak).
    pub binding: SymbolBinding,
    /// Symbol type (function, object, section, or no-type).
    pub symbol_type: SymbolType,
    /// Symbol visibility for dynamic linking.
    pub visibility: SymbolVisibility,
    /// Whether this symbol is a definition (`true`) or an undefined reference
    /// (`false`). This flag is independent of `section_index` and is the
    /// authoritative source for definedness — the linker uses it to distinguish
    /// defined symbols from external references, avoiding ambiguity when a
    /// defined symbol's section happens to be at index 0.
    pub is_definition: bool,
}

// ---------------------------------------------------------------------------
// RelocationType, Relocation — cross-architecture relocation entries
// ---------------------------------------------------------------------------

/// Relocation types for all four supported architectures.
///
/// Each variant corresponds to an ELF relocation type (`R_*` constant) that
/// instructs the linker how to patch a machine code or data reference to
/// account for the final symbol addresses.
///
/// Variant names deliberately mirror the ELF `R_*` constant names for
/// readability and grep-ability, hence the underscore-separated naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum RelocationType {
    // ----- x86-64 relocations (ELF64, EM_X86_64) -----
    /// `R_X86_64_64` — 64-bit absolute address.
    X86_64_64,
    /// `R_X86_64_PC32` — 32-bit PC-relative address.
    X86_64_PC32,
    /// `R_X86_64_PLT32` — 32-bit PLT-relative address for function calls.
    X86_64_PLT32,
    /// `R_X86_64_GOT64` — 64-bit GOT entry offset.
    X86_64_GOT64,
    /// `R_X86_64_GOTPCREL` — 32-bit PC-relative GOT entry address.
    X86_64_GOTPCREL,

    // ----- i686 relocations (ELF32, EM_386) -----
    /// `R_386_32` — 32-bit absolute address.
    I386_32,
    /// `R_386_PC32` — 32-bit PC-relative address.
    I386_PC32,
    /// `R_386_GOT32` — 32-bit GOT offset.
    I386_GOT32,
    /// `R_386_PLT32` — 32-bit PLT-relative address.
    I386_PLT32,

    // ----- AArch64 relocations (ELF64, EM_AARCH64) -----
    /// `R_AARCH64_ABS64` — 64-bit absolute address.
    Aarch64_ABS64,
    /// `R_AARCH64_ABS32` — 32-bit absolute address.
    Aarch64_ABS32,
    /// `R_AARCH64_CALL26` — 26-bit PC-relative call (BL instruction).
    Aarch64_CALL26,
    /// `R_AARCH64_JUMP26` — 26-bit PC-relative jump (B instruction).
    Aarch64_JUMP26,
    /// `R_AARCH64_ADR_PREL_PG_HI21` — ADRP page-relative high 21 bits.
    Aarch64_ADR_PREL_PG_HI21,
    /// `R_AARCH64_ADD_ABS_LO12_NC` — ADD low 12 bits, no overflow check.
    Aarch64_ADD_ABS_LO12_NC,

    // ----- RISC-V 64 relocations (ELF64, EM_RISCV) -----
    /// `R_RISCV_64` — 64-bit absolute address.
    Riscv_64,
    /// `R_RISCV_32` — 32-bit absolute address.
    Riscv_32,
    /// `R_RISCV_BRANCH` — 12-bit PC-relative branch offset.
    Riscv_Branch,
    /// `R_RISCV_JAL` — 20-bit PC-relative JAL offset.
    Riscv_Jal,
    /// `R_RISCV_CALL` — Paired relocation for AUIPC+JALR call sequence.
    Riscv_Call,
    /// `R_RISCV_PCREL_HI20` — High 20 bits of PC-relative address.
    Riscv_Pcrel_Hi20,
    /// `R_RISCV_PCREL_LO12_I` — Low 12 bits of PC-relative (I-format).
    Riscv_Pcrel_Lo12_I,
    /// `R_RISCV_HI20` — High 20 bits of absolute address (LUI).
    Riscv_Hi20,
    /// `R_RISCV_LO12_I` — Low 12 bits of absolute address (I-format).
    Riscv_Lo12_I,
}

/// A relocation entry produced by code generation.
///
/// Each relocation tells the linker: "at byte offset `offset` in section
/// `section_index`, patch the value using the address of `symbol` with
/// the computation specified by `reloc_type`, plus `addend`."
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Byte offset within the section where the relocation applies.
    pub offset: u64,
    /// Name of the symbol whose address is used to compute the patched value.
    pub symbol: String,
    /// The relocation computation to apply (architecture-specific).
    pub reloc_type: RelocationType,
    /// Constant addend for the relocation computation.
    pub addend: i64,
    /// Index of the section containing the relocation site
    /// (into `ObjectCode.sections`).
    pub section_index: usize,
}

// ---------------------------------------------------------------------------
// ObjectCode — the complete output of code generation
// ---------------------------------------------------------------------------

/// The output of code generation for a single compilation unit.
///
/// Contains machine code sections, symbol definitions, and relocation entries
/// ready for the integrated linker to merge with CRT objects and library
/// archives into a final ELF binary.
#[derive(Debug)]
pub struct ObjectCode {
    /// Sections of machine code and data (`.text`, `.data`, `.rodata`, `.bss`,
    /// and any custom sections like `.note.GNU-stack`).
    pub sections: Vec<Section>,
    /// Symbol definitions exported by this compilation unit (functions,
    /// global variables, section markers).
    pub symbols: Vec<Symbol>,
    /// Relocation entries that the linker must resolve against the final
    /// symbol addresses.
    pub relocations: Vec<Relocation>,
    /// The target architecture, used by the linker to validate that all
    /// input objects are for the same target and to select the correct
    /// relocation processing logic.
    pub target_arch: Architecture,
}

impl ObjectCode {
    /// Create a new, empty `ObjectCode` for the given target architecture.
    pub fn new(target_arch: Architecture) -> Self {
        Self {
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
            target_arch,
        }
    }

    /// Add a section and return its index in the sections list.
    pub fn add_section(&mut self, section: Section) -> usize {
        let index = self.sections.len();
        self.sections.push(section);
        index
    }

    /// Add a symbol definition.
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.symbols.push(symbol);
    }

    /// Add a relocation entry.
    pub fn add_relocation(&mut self, relocation: Relocation) {
        self.relocations.push(relocation);
    }
}

// ---------------------------------------------------------------------------
// MachineInstr, MachineOperand — pre-encoding instruction representation
// ---------------------------------------------------------------------------

/// A machine instruction operand.
///
/// Operands are the arguments to machine instructions after instruction
/// selection but before final encoding. The encoder translates these into
/// binary fields within the instruction encoding.
#[derive(Debug, Clone)]
pub enum MachineOperand {
    /// A physical register, assigned by the register allocator.
    Register(PhysReg),
    /// An immediate integer value encoded directly in the instruction.
    Immediate(i64),
    /// A memory reference with a base register and signed byte offset.
    /// Encodes as `[base + offset]` in the target's addressing mode.
    Memory {
        /// Base register for the memory access.
        base: PhysReg,
        /// Signed byte offset from the base register.
        offset: i32,
    },
    /// A reference to a named symbol (function call target, global variable).
    /// The encoder emits a relocation for the linker to resolve.
    Symbol(String),
    /// A basic block label used as a branch target. The encoder resolves
    /// the label to a PC-relative offset during final encoding.
    Label(u32),
}

/// A machine instruction before final binary encoding.
///
/// This is the intermediate representation between instruction selection
/// (which chooses architecture-specific opcodes) and the encoder (which
/// emits raw bytes). Each backend defines its own opcode numbering in
/// its `isel` module.
#[derive(Debug, Clone)]
pub struct MachineInstr {
    /// Architecture-specific opcode identifying the instruction.
    /// The meaning of this field is defined by each backend's isel module.
    pub opcode: u32,
    /// Ordered list of operands for this instruction.
    pub operands: Vec<MachineOperand>,
    /// Optional source location for DWARF debug info correlation.
    /// When present, the debug info generator maps this instruction's
    /// address to the corresponding source file, line, and column.
    pub loc: Option<SourceLocation>,
}

impl MachineInstr {
    /// Create a new machine instruction with the given opcode and no operands.
    pub fn new(opcode: u32) -> Self {
        Self {
            opcode,
            operands: Vec::new(),
            loc: None,
        }
    }

    /// Create a machine instruction with the given opcode and operands.
    pub fn with_operands(opcode: u32, operands: Vec<MachineOperand>) -> Self {
        Self {
            opcode,
            operands,
            loc: None,
        }
    }

    /// Attach a source location for debug information.
    pub fn with_loc(mut self, loc: SourceLocation) -> Self {
        self.loc = Some(loc);
        self
    }
}

// ---------------------------------------------------------------------------
// CodeGen trait — the core interface for all backends
// ---------------------------------------------------------------------------

/// Trait for architecture-specific code generation backends.
///
/// Each of the four backends (x86-64, i686, AArch64, RISC-V 64) implements
/// this trait. The [`generate`](CodeGen::generate) method accepts optimised
/// SSA-form IR and produces an [`ObjectCode`] containing machine code bytes,
/// relocations, and symbol definitions ready for the integrated linker.
///
/// The trait is object-safe so that [`generate_code`] can dispatch dynamically
/// via `Box<dyn CodeGen>` based on the target architecture.
pub trait CodeGen {
    /// Generate machine code for the given IR module targeting the specified
    /// architecture.
    ///
    /// # Arguments
    ///
    /// * `module` — The optimised IR module containing functions and globals
    ///   in SSA form.
    /// * `target` — Target configuration specifying architecture, ABI details,
    ///   pointer width, endianness, and CLI flag settings (e.g. `-fPIC`,
    ///   `-mretpoline`).
    ///
    /// # Returns
    ///
    /// An [`ObjectCode`] on success containing all sections, symbols, and
    /// relocations for this compilation unit. Returns a [`CodeGenError`] if
    /// code generation fails for any reason (unsupported construct, register
    /// pressure, encoding constraint violation, etc.).
    fn generate(
        &self,
        module: &Module,
        target: &TargetConfig,
    ) -> Result<ObjectCode, CodeGenError>;

    /// Returns the target architecture this backend generates code for.
    ///
    /// Used by the driver and linker to verify that the selected backend
    /// matches the intended target.
    fn target_arch(&self) -> Architecture;
}

// ---------------------------------------------------------------------------
// generate_code — top-level dispatch function
// ---------------------------------------------------------------------------

/// Generate machine code for the given IR module by dispatching to the
/// appropriate architecture backend based on the target configuration.
///
/// This is the main entry point called by the driver pipeline after
/// optimisation passes have been applied. It creates the correct backend
/// instance for the target architecture and delegates code generation.
///
/// # Arguments
///
/// * `module` — The optimised IR module (functions and globals in SSA form).
/// * `target` — Target configuration from CLI `--target` flag parsing.
///
/// # Returns
///
/// An [`ObjectCode`] on success, or a [`CodeGenError`] on failure.
///
/// # Example
///
/// ```ignore
/// use crate::codegen::generate_code;
/// use crate::driver::target::TargetConfig;
///
/// let target = TargetConfig::x86_64();
/// let object = generate_code(&ir_module, &target)?;
/// // object.sections, object.symbols, object.relocations ready for linker
/// ```
pub fn generate_code(
    module: &Module,
    target: &TargetConfig,
) -> Result<ObjectCode, CodeGenError> {
    // Select the architecture-specific backend based on the target config.
    let backend: Box<dyn CodeGen> = match target.arch {
        Architecture::X86_64 => Box::new(x86_64::X86_64CodeGen::new()),
        Architecture::I686 => Box::new(i686::I686CodeGen::new()),
        Architecture::Aarch64 => Box::new(aarch64::Aarch64CodeGen::new()),
        Architecture::Riscv64 => Box::new(riscv64::Riscv64CodeGen::new()),
    };

    backend.generate(module, target)
}

// ---------------------------------------------------------------------------
// Shared utility functions
// ---------------------------------------------------------------------------

/// Align `value` up to the next multiple of `alignment`.
///
/// `alignment` **must** be a power of two. If `value` is already aligned,
/// it is returned unchanged.
///
/// # Panics
///
/// Panics in debug mode if `alignment` is zero.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(align_to(0, 16), 0);
/// assert_eq!(align_to(1, 16), 16);
/// assert_eq!(align_to(16, 16), 16);
/// assert_eq!(align_to(17, 16), 32);
/// ```
#[inline]
pub fn align_to(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two, got {}",
        alignment
    );
    // Bitmask trick: (value + align-1) & ~(align-1).
    // Works because alignment is a power of two.
    (value + alignment - 1) & !(alignment - 1)
}

/// Encode a signed integer as a variable-length SLEB128 (Signed Little-Endian
/// Base 128) byte sequence.
///
/// SLEB128 is used in DWARF debug information and other compact binary formats.
/// Each output byte encodes 7 data bits and a continuation bit (bit 7).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(encode_sleb128(0), vec![0x00]);
/// assert_eq!(encode_sleb128(-1), vec![0x7F]);
/// assert_eq!(encode_sleb128(128), vec![0x80, 0x01]);
/// assert_eq!(encode_sleb128(-128), vec![0x80, 0x7F]);
/// ```
pub fn encode_sleb128(mut value: i64) -> Vec<u8> {
    let mut result = Vec::with_capacity(10);
    let mut more = true;

    while more {
        // Extract the low 7 bits.
        let mut byte = (value & 0x7F) as u8;
        // Arithmetic right shift to propagate the sign bit.
        value >>= 7;

        // Determine if more bytes are needed:
        // - If value is 0 and the sign bit of the current byte is 0, we're done.
        // - If value is -1 and the sign bit of the current byte is 1, we're done.
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            // Set the continuation bit.
            byte |= 0x80;
        }

        result.push(byte);
    }

    result
}

/// Encode an unsigned integer as a variable-length ULEB128 (Unsigned
/// Little-Endian Base 128) byte sequence.
///
/// ULEB128 is used in DWARF debug information for encoding unsigned values
/// such as abbreviation codes, attribute values, and form lengths.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(encode_uleb128(0), vec![0x00]);
/// assert_eq!(encode_uleb128(127), vec![0x7F]);
/// assert_eq!(encode_uleb128(128), vec![0x80, 0x01]);
/// assert_eq!(encode_uleb128(624485), vec![0xE5, 0x8E, 0x26]);
/// ```
pub fn encode_uleb128(mut value: u64) -> Vec<u8> {
    let mut result = Vec::with_capacity(10);

    loop {
        // Extract the low 7 bits.
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            // More bytes follow: set the continuation bit.
            byte |= 0x80;
        }

        result.push(byte);

        if value == 0 {
            break;
        }
    }

    result
}

/// Write a 16-bit unsigned integer in little-endian byte order to a buffer.
///
/// Appends exactly 2 bytes to `buf`.
#[inline]
pub fn write_le_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Write a 32-bit unsigned integer in little-endian byte order to a buffer.
///
/// Appends exactly 4 bytes to `buf`.
#[inline]
pub fn write_le_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Write a 64-bit unsigned integer in little-endian byte order to a buffer.
///
/// Appends exactly 8 bytes to `buf`.
#[inline]
pub fn write_le_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Architecture enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_architecture_variants_exist() {
        let x86 = Architecture::X86_64;
        let i686 = Architecture::I686;
        let arm = Architecture::Aarch64;
        let rv = Architecture::Riscv64;

        // All four variants are distinct.
        assert_ne!(x86, i686);
        assert_ne!(x86, arm);
        assert_ne!(x86, rv);
        assert_ne!(i686, arm);
        assert_ne!(i686, rv);
        assert_ne!(arm, rv);
    }

    #[test]
    fn test_architecture_copy_clone() {
        let arch = Architecture::X86_64;
        let copy = arch;
        let clone = arch.clone();
        assert_eq!(arch, copy);
        assert_eq!(arch, clone);
    }

    // -----------------------------------------------------------------------
    // ObjectCode construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_object_code_new() {
        let obj = ObjectCode::new(Architecture::X86_64);
        assert!(obj.sections.is_empty());
        assert!(obj.symbols.is_empty());
        assert!(obj.relocations.is_empty());
        assert_eq!(obj.target_arch, Architecture::X86_64);
    }

    #[test]
    fn test_object_code_add_section() {
        let mut obj = ObjectCode::new(Architecture::Aarch64);
        let section = Section {
            name: ".text".to_string(),
            data: vec![0x00, 0x01, 0x02],
            section_type: SectionType::Text,
            alignment: 16,
            flags: SectionFlags::text(),
        };
        let idx = obj.add_section(section);
        assert_eq!(idx, 0);
        assert_eq!(obj.sections.len(), 1);
        assert_eq!(obj.sections[0].name, ".text");
        assert_eq!(obj.sections[0].data.len(), 3);
    }

    #[test]
    fn test_object_code_add_symbol() {
        let mut obj = ObjectCode::new(Architecture::I686);
        obj.add_symbol(Symbol {
            name: "main".to_string(),
            section_index: 0,
            offset: 0,
            size: 42,
            binding: SymbolBinding::Global,
            symbol_type: SymbolType::Function,
            visibility: SymbolVisibility::Default,
            is_definition: true,
        });
        assert_eq!(obj.symbols.len(), 1);
        assert_eq!(obj.symbols[0].name, "main");
        assert_eq!(obj.symbols[0].size, 42);
    }

    #[test]
    fn test_object_code_add_relocation() {
        let mut obj = ObjectCode::new(Architecture::Riscv64);
        obj.add_relocation(Relocation {
            offset: 0x10,
            symbol: "printf".to_string(),
            reloc_type: RelocationType::Riscv_Call,
            addend: 0,
            section_index: 0,
        });
        assert_eq!(obj.relocations.len(), 1);
        assert_eq!(obj.relocations[0].symbol, "printf");
        assert_eq!(obj.relocations[0].reloc_type, RelocationType::Riscv_Call);
    }

    // -----------------------------------------------------------------------
    // Section flags convenience constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_section_flags_text() {
        let f = SectionFlags::text();
        assert!(!f.writable);
        assert!(f.executable);
        assert!(f.allocatable);
    }

    #[test]
    fn test_section_flags_data() {
        let f = SectionFlags::data();
        assert!(f.writable);
        assert!(!f.executable);
        assert!(f.allocatable);
    }

    #[test]
    fn test_section_flags_rodata() {
        let f = SectionFlags::rodata();
        assert!(!f.writable);
        assert!(!f.executable);
        assert!(f.allocatable);
    }

    #[test]
    fn test_section_flags_bss() {
        let f = SectionFlags::bss();
        assert!(f.writable);
        assert!(!f.executable);
        assert!(f.allocatable);
    }

    // -----------------------------------------------------------------------
    // SectionType equality
    // -----------------------------------------------------------------------

    #[test]
    fn test_section_type_equality() {
        assert_eq!(SectionType::Text, SectionType::Text);
        assert_ne!(SectionType::Text, SectionType::Data);
        assert_eq!(SectionType::Custom(1), SectionType::Custom(1));
        assert_ne!(SectionType::Custom(1), SectionType::Custom(2));
    }

    // -----------------------------------------------------------------------
    // Symbol enums
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbol_binding_variants() {
        assert_ne!(SymbolBinding::Local, SymbolBinding::Global);
        assert_ne!(SymbolBinding::Global, SymbolBinding::Weak);
        assert_ne!(SymbolBinding::Local, SymbolBinding::Weak);
    }

    #[test]
    fn test_symbol_type_variants() {
        assert_ne!(SymbolType::Function, SymbolType::Object);
        assert_ne!(SymbolType::Object, SymbolType::NoType);
        assert_ne!(SymbolType::NoType, SymbolType::Section);
    }

    #[test]
    fn test_symbol_visibility_variants() {
        assert_ne!(SymbolVisibility::Default, SymbolVisibility::Hidden);
        assert_ne!(SymbolVisibility::Hidden, SymbolVisibility::Protected);
    }

    // -----------------------------------------------------------------------
    // RelocationType coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_relocation_type_x86_64() {
        let types = [
            RelocationType::X86_64_64,
            RelocationType::X86_64_PC32,
            RelocationType::X86_64_PLT32,
            RelocationType::X86_64_GOT64,
            RelocationType::X86_64_GOTPCREL,
        ];
        // All distinct.
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_relocation_type_i386() {
        let types = [
            RelocationType::I386_32,
            RelocationType::I386_PC32,
            RelocationType::I386_GOT32,
            RelocationType::I386_PLT32,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_relocation_type_aarch64() {
        let types = [
            RelocationType::Aarch64_ABS64,
            RelocationType::Aarch64_ABS32,
            RelocationType::Aarch64_CALL26,
            RelocationType::Aarch64_JUMP26,
            RelocationType::Aarch64_ADR_PREL_PG_HI21,
            RelocationType::Aarch64_ADD_ABS_LO12_NC,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_relocation_type_riscv() {
        let types = [
            RelocationType::Riscv_64,
            RelocationType::Riscv_32,
            RelocationType::Riscv_Branch,
            RelocationType::Riscv_Jal,
            RelocationType::Riscv_Call,
            RelocationType::Riscv_Pcrel_Hi20,
            RelocationType::Riscv_Pcrel_Lo12_I,
            RelocationType::Riscv_Hi20,
            RelocationType::Riscv_Lo12_I,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Alignment utility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_to_zero() {
        assert_eq!(align_to(0, 16), 0);
    }

    #[test]
    fn test_align_to_one_byte() {
        assert_eq!(align_to(1, 16), 16);
    }

    #[test]
    fn test_align_to_already_aligned() {
        assert_eq!(align_to(16, 16), 16);
    }

    #[test]
    fn test_align_to_just_past() {
        assert_eq!(align_to(17, 16), 32);
    }

    #[test]
    fn test_align_to_alignment_1() {
        // Every value is already aligned to 1.
        assert_eq!(align_to(0, 1), 0);
        assert_eq!(align_to(1, 1), 1);
        assert_eq!(align_to(1000, 1), 1000);
    }

    #[test]
    fn test_align_to_various() {
        assert_eq!(align_to(0, 4), 0);
        assert_eq!(align_to(3, 4), 4);
        assert_eq!(align_to(4, 4), 4);
        assert_eq!(align_to(5, 4), 8);
        assert_eq!(align_to(7, 8), 8);
        assert_eq!(align_to(9, 8), 16);
        assert_eq!(align_to(4096, 4096), 4096);
        assert_eq!(align_to(4097, 4096), 8192);
    }

    // -----------------------------------------------------------------------
    // ULEB128 encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_uleb128_zero() {
        assert_eq!(encode_uleb128(0), vec![0x00]);
    }

    #[test]
    fn test_uleb128_small() {
        assert_eq!(encode_uleb128(1), vec![0x01]);
        assert_eq!(encode_uleb128(63), vec![0x3F]);
    }

    #[test]
    fn test_uleb128_127() {
        assert_eq!(encode_uleb128(127), vec![0x7F]);
    }

    #[test]
    fn test_uleb128_128() {
        assert_eq!(encode_uleb128(128), vec![0x80, 0x01]);
    }

    #[test]
    fn test_uleb128_multi_byte() {
        // 624485 = 0x98765 → ULEB128: 0xE5, 0x8E, 0x26
        assert_eq!(encode_uleb128(624485), vec![0xE5, 0x8E, 0x26]);
    }

    #[test]
    fn test_uleb128_large() {
        // 2^32 - 1 = 4294967295
        let encoded = encode_uleb128(0xFFFFFFFF);
        assert_eq!(encoded, vec![0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
    }

    #[test]
    fn test_uleb128_u64_max() {
        let encoded = encode_uleb128(u64::MAX);
        assert_eq!(encoded.len(), 10);
        // Last byte should not have the continuation bit set.
        assert_eq!(encoded[9] & 0x80, 0);
    }

    // -----------------------------------------------------------------------
    // SLEB128 encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sleb128_zero() {
        assert_eq!(encode_sleb128(0), vec![0x00]);
    }

    #[test]
    fn test_sleb128_positive_small() {
        assert_eq!(encode_sleb128(1), vec![0x01]);
        assert_eq!(encode_sleb128(63), vec![0x3F]);
    }

    #[test]
    fn test_sleb128_negative_one() {
        assert_eq!(encode_sleb128(-1), vec![0x7F]);
    }

    #[test]
    fn test_sleb128_negative_128() {
        assert_eq!(encode_sleb128(-128), vec![0x80, 0x7F]);
    }

    #[test]
    fn test_sleb128_positive_128() {
        assert_eq!(encode_sleb128(128), vec![0x80, 0x01]);
    }

    #[test]
    fn test_sleb128_negative_large() {
        // -123456 → SLEB128
        let encoded = encode_sleb128(-123456);
        // Verify round-trip by decoding.
        let decoded = decode_sleb128_for_test(&encoded);
        assert_eq!(decoded, -123456);
    }

    #[test]
    fn test_sleb128_positive_large() {
        let encoded = encode_sleb128(123456);
        let decoded = decode_sleb128_for_test(&encoded);
        assert_eq!(decoded, 123456);
    }

    /// Test helper: decode SLEB128 back to i64 for round-trip verification.
    fn decode_sleb128_for_test(bytes: &[u8]) -> i64 {
        let mut result: i64 = 0;
        let mut shift: u32 = 0;
        let mut last_byte: u8 = 0;

        for &byte in bytes {
            last_byte = byte;
            result |= ((byte & 0x7F) as i64) << shift;
            shift += 7;

            if byte & 0x80 == 0 {
                break;
            }
        }

        // Sign-extend if the sign bit of the last byte is set.
        if shift < 64 && (last_byte & 0x40) != 0 {
            result |= !0i64 << shift;
        }

        result
    }

    // -----------------------------------------------------------------------
    // Little-endian write tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_le_u16() {
        let mut buf = Vec::new();
        write_le_u16(&mut buf, 0x0102);
        assert_eq!(buf, vec![0x02, 0x01]);
    }

    #[test]
    fn test_write_le_u16_zero() {
        let mut buf = Vec::new();
        write_le_u16(&mut buf, 0);
        assert_eq!(buf, vec![0x00, 0x00]);
    }

    #[test]
    fn test_write_le_u16_max() {
        let mut buf = Vec::new();
        write_le_u16(&mut buf, 0xFFFF);
        assert_eq!(buf, vec![0xFF, 0xFF]);
    }

    #[test]
    fn test_write_le_u32() {
        let mut buf = Vec::new();
        write_le_u32(&mut buf, 0x01020304);
        assert_eq!(buf, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_write_le_u32_zero() {
        let mut buf = Vec::new();
        write_le_u32(&mut buf, 0);
        assert_eq!(buf, vec![0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_write_le_u64() {
        let mut buf = Vec::new();
        write_le_u64(&mut buf, 0x0102030405060708);
        assert_eq!(buf, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_write_le_u64_zero() {
        let mut buf = Vec::new();
        write_le_u64(&mut buf, 0);
        assert_eq!(buf, vec![0x00; 8]);
    }

    #[test]
    fn test_write_le_append() {
        // Verify that writes append rather than overwrite.
        let mut buf = Vec::new();
        write_le_u16(&mut buf, 0xAABB);
        write_le_u32(&mut buf, 0xCCDDEEFF);
        assert_eq!(buf.len(), 6);
        assert_eq!(buf, vec![0xBB, 0xAA, 0xFF, 0xEE, 0xDD, 0xCC]);
    }

    // -----------------------------------------------------------------------
    // CodeGenError display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_codegen_error_display_unsupported() {
        let err = CodeGenError::UnsupportedInstruction("vector add".to_string());
        assert_eq!(
            format!("{}", err),
            "unsupported instruction: vector add"
        );
    }

    #[test]
    fn test_codegen_error_display_regalloc() {
        let err = CodeGenError::RegisterAllocationFailed("too many live values".to_string());
        assert_eq!(
            format!("{}", err),
            "register allocation failed: too many live values"
        );
    }

    #[test]
    fn test_codegen_error_display_encoding() {
        let err = CodeGenError::EncodingError("immediate too large".to_string());
        assert_eq!(format!("{}", err), "encoding error: immediate too large");
    }

    #[test]
    fn test_codegen_error_display_abi() {
        let err = CodeGenError::AbiError("stack misaligned".to_string());
        assert_eq!(format!("{}", err), "ABI error: stack misaligned");
    }

    #[test]
    fn test_codegen_error_display_internal() {
        let err = CodeGenError::InternalError("unexpected state".to_string());
        assert_eq!(format!("{}", err), "internal error: unexpected state");
    }

    #[test]
    fn test_codegen_error_is_std_error() {
        // Verify CodeGenError implements std::error::Error.
        let err: Box<dyn std::error::Error> =
            Box::new(CodeGenError::InternalError("test".to_string()));
        // source() should return None (no underlying cause).
        assert!(err.source().is_none());
    }

    // -----------------------------------------------------------------------
    // MachineInstr and MachineOperand tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_machine_instr_new() {
        let instr = MachineInstr::new(42);
        assert_eq!(instr.opcode, 42);
        assert!(instr.operands.is_empty());
        assert!(instr.loc.is_none());
    }

    #[test]
    fn test_machine_instr_with_operands() {
        let operands = vec![
            MachineOperand::Register(PhysReg(0)),
            MachineOperand::Immediate(100),
        ];
        let instr = MachineInstr::with_operands(7, operands);
        assert_eq!(instr.opcode, 7);
        assert_eq!(instr.operands.len(), 2);
    }

    #[test]
    fn test_machine_operand_variants() {
        // Verify all operand variants can be constructed.
        let _reg = MachineOperand::Register(PhysReg(5));
        let _imm = MachineOperand::Immediate(-42);
        let _mem = MachineOperand::Memory {
            base: PhysReg(7),
            offset: -16,
        };
        let _sym = MachineOperand::Symbol("printf".to_string());
        let _lbl = MachineOperand::Label(3);
    }

    // -----------------------------------------------------------------------
    // CodeGen trait object safety
    // -----------------------------------------------------------------------

    #[test]
    fn test_codegen_trait_is_object_safe() {
        // This test verifies that `dyn CodeGen` compiles — the trait is
        // object-safe (no generic methods, no Self by value). We construct
        // a function pointer type that accepts `&dyn CodeGen` to prove it.
        fn _accepts_dyn_codegen(_cg: &dyn CodeGen) {}
        // If this compiles, the trait is object-safe. No runtime assertion
        // needed.
    }

    // -----------------------------------------------------------------------
    // Re-export accessibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_phys_reg_re_export() {
        let reg = PhysReg(10);
        assert_eq!(reg.0, 10);
    }

    #[test]
    fn test_reg_class_re_export() {
        let _int = RegClass::Integer;
        let _float = RegClass::Float;
        assert_ne!(RegClass::Integer, RegClass::Float);
    }
}
