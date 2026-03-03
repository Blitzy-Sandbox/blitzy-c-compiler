//! ELF Format Structures (ELF32/ELF64)
//!
//! This module provides complete reading and writing support for both ELF32 and
//! ELF64 structures. It is the foundational module used by nearly every other
//! linker submodule. It defines header structs, section headers, program headers,
//! symbol table entries, relocation entries, string table reading, and all ELF
//! format constants required by the ELF specification.
//!
//! # ELF Dual-Width Support
//!
//! All four supported targets are little-endian:
//! - ELF64: x86-64, AArch64, RISC-V 64
//! - ELF32: i686
//!
//! # Zero External Dependencies
//!
//! This module uses only the Rust standard library. No external ELF libraries
//! (like `goblin` or `object`) are used.

use crate::driver::target::{Architecture, ElfClass, TargetConfig};

// ===========================================================================
// ELF Constants
// ===========================================================================

/// ELF magic number: 0x7f 'E' 'L' 'F'
pub const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];

// ---------------------------------------------------------------------------
// ELF Class (e_ident[EI_CLASS])
// ---------------------------------------------------------------------------

/// Invalid ELF class.
pub const ELFCLASSNONE: u8 = 0;
/// 32-bit ELF objects.
pub const ELFCLASS32: u8 = 1;
/// 64-bit ELF objects.
pub const ELFCLASS64: u8 = 2;

// ---------------------------------------------------------------------------
// ELF Data encoding (e_ident[EI_DATA])
// ---------------------------------------------------------------------------

/// Little-endian data encoding.
pub const ELFDATA2LSB: u8 = 1;

// ---------------------------------------------------------------------------
// ELF Version (e_ident[EI_VERSION])
// ---------------------------------------------------------------------------

/// Current ELF version.
pub const EV_CURRENT: u8 = 1;

// ---------------------------------------------------------------------------
// ELF OS/ABI (e_ident[EI_OSABI])
// ---------------------------------------------------------------------------

/// UNIX System V ABI (used for Linux targets).
pub const ELFOSABI_NONE: u8 = 0;

// ---------------------------------------------------------------------------
// e_ident index constants
// ---------------------------------------------------------------------------

/// Start of the magic number in e_ident.
pub const EI_MAG0: usize = 0;
/// ELF class byte index in e_ident.
pub const EI_CLASS: usize = 4;
/// Data encoding byte index in e_ident.
pub const EI_DATA: usize = 5;
/// ELF version byte index in e_ident.
pub const EI_VERSION: usize = 6;
/// OS/ABI byte index in e_ident.
pub const EI_OSABI: usize = 7;
/// Total size of the e_ident array.
pub const EI_NIDENT: usize = 16;

// ---------------------------------------------------------------------------
// ELF Type (e_type)
// ---------------------------------------------------------------------------

/// No file type.
pub const ET_NONE: u16 = 0;
/// Relocatable object file.
pub const ET_REL: u16 = 1;
/// Executable file.
pub const ET_EXEC: u16 = 2;
/// Shared object file.
pub const ET_DYN: u16 = 3;
/// Core dump file.
pub const ET_CORE: u16 = 4;

// ---------------------------------------------------------------------------
// ELF Machine (e_machine)
// ---------------------------------------------------------------------------

/// Intel 80386 (i686).
pub const EM_386: u16 = 3;
/// AMD x86-64.
pub const EM_X86_64: u16 = 62;
/// ARM AARCH64 (64-bit ARM).
pub const EM_AARCH64: u16 = 183;
/// RISC-V.
pub const EM_RISCV: u16 = 243;

// ---------------------------------------------------------------------------
// ELF Header sizes
// ---------------------------------------------------------------------------

/// Size of an ELF64 header in bytes.
pub const ELF64_EHDR_SIZE: usize = 64;
/// Size of an ELF32 header in bytes.
pub const ELF32_EHDR_SIZE: usize = 52;

// ---------------------------------------------------------------------------
// Program header types (p_type)
// ---------------------------------------------------------------------------

/// Unused segment.
pub const PT_NULL: u32 = 0;
/// Loadable segment.
pub const PT_LOAD: u32 = 1;
/// Dynamic linking information.
pub const PT_DYNAMIC: u32 = 2;
/// Interpreter path.
pub const PT_INTERP: u32 = 3;
/// Auxiliary information.
pub const PT_NOTE: u32 = 4;
/// Program header table.
pub const PT_PHDR: u32 = 6;
/// GNU stack permissions.
pub const PT_GNU_STACK: u32 = 0x6474e551;
/// GNU read-only after relocation.
pub const PT_GNU_RELRO: u32 = 0x6474e552;

// ---------------------------------------------------------------------------
// Program header flags (p_flags)
// ---------------------------------------------------------------------------

/// Segment is executable.
pub const PF_X: u32 = 0x1;
/// Segment is writable.
pub const PF_W: u32 = 0x2;
/// Segment is readable.
pub const PF_R: u32 = 0x4;

// ---------------------------------------------------------------------------
// Section header types (sh_type)
// ---------------------------------------------------------------------------

/// Inactive section header.
pub const SHT_NULL: u32 = 0;
/// Program-defined data.
pub const SHT_PROGBITS: u32 = 1;
/// Symbol table.
pub const SHT_SYMTAB: u32 = 2;
/// String table.
pub const SHT_STRTAB: u32 = 3;
/// Relocations with explicit addends.
pub const SHT_RELA: u32 = 4;
/// Symbol hash table.
pub const SHT_HASH: u32 = 5;
/// Dynamic linking information.
pub const SHT_DYNAMIC: u32 = 6;
/// Notes.
pub const SHT_NOTE: u32 = 7;
/// Section occupies no space in file (BSS).
pub const SHT_NOBITS: u32 = 8;
/// Relocations without explicit addends.
pub const SHT_REL: u32 = 9;
/// Dynamic symbol table.
pub const SHT_DYNSYM: u32 = 11;

// ---------------------------------------------------------------------------
// Section header flags (sh_flags)
// ---------------------------------------------------------------------------

/// Section is writable.
pub const SHF_WRITE: u64 = 0x1;
/// Section occupies memory during execution.
pub const SHF_ALLOC: u64 = 0x2;
/// Section contains executable instructions.
pub const SHF_EXECINSTR: u64 = 0x4;

// ---------------------------------------------------------------------------
// Special section indices
// ---------------------------------------------------------------------------

/// Undefined section index.
pub const SHN_UNDEF: u16 = 0;
/// Absolute symbol (not relative to any section).
pub const SHN_ABS: u16 = 0xFFF1;
/// Common symbol (unallocated C external).
pub const SHN_COMMON: u16 = 0xFFF2;

// ---------------------------------------------------------------------------
// Symbol binding (upper 4 bits of st_info)
// ---------------------------------------------------------------------------

/// Local symbol.
pub const STB_LOCAL: u8 = 0;
/// Global symbol.
pub const STB_GLOBAL: u8 = 1;
/// Weak symbol.
pub const STB_WEAK: u8 = 2;

// ---------------------------------------------------------------------------
// Symbol type (lower 4 bits of st_info)
// ---------------------------------------------------------------------------

/// Symbol type not specified.
pub const STT_NOTYPE: u8 = 0;
/// Data object symbol.
pub const STT_OBJECT: u8 = 1;
/// Function entry point symbol.
pub const STT_FUNC: u8 = 2;
/// Section symbol.
pub const STT_SECTION: u8 = 3;
/// Source file name symbol.
pub const STT_FILE: u8 = 4;

// ---------------------------------------------------------------------------
// Symbol visibility (lower 2 bits of st_other)
// ---------------------------------------------------------------------------

/// Default visibility.
pub const STV_DEFAULT: u8 = 0;
/// Hidden visibility.
pub const STV_HIDDEN: u8 = 2;
/// Protected visibility.
pub const STV_PROTECTED: u8 = 3;

// ===========================================================================
// ElfError
// ===========================================================================

/// Errors that can occur during ELF parsing or writing.
#[derive(Debug)]
pub enum ElfError {
    /// Input data does not start with the ELF magic bytes.
    InvalidMagic,
    /// ELF class is not ELFCLASS32 or ELFCLASS64.
    UnsupportedClass(u8),
    /// ELF data encoding is not little-endian (ELFDATA2LSB).
    UnsupportedEndianness(u8),
    /// Input data is too short for the requested structure.
    TruncatedData(&'static str),
    /// Section index is out of range.
    InvalidSectionIndex(u32),
    /// String offset is out of range or points to invalid UTF-8.
    InvalidStringOffset(u32),
    /// Symbol index is out of range.
    InvalidSymbolIndex(u32),
    /// Wraps a `std::io::Error` from file I/O operations.
    IoError(std::io::Error),
}

impl std::fmt::Display for ElfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElfError::InvalidMagic => write!(f, "invalid ELF magic number"),
            ElfError::UnsupportedClass(c) => {
                write!(f, "unsupported ELF class: {}", c)
            }
            ElfError::UnsupportedEndianness(e) => {
                write!(
                    f,
                    "unsupported ELF data encoding: {} (only little-endian supported)",
                    e
                )
            }
            ElfError::TruncatedData(what) => {
                write!(f, "truncated ELF data while reading {}", what)
            }
            ElfError::InvalidSectionIndex(idx) => {
                write!(f, "invalid section index: {}", idx)
            }
            ElfError::InvalidStringOffset(off) => {
                write!(f, "invalid string table offset: {}", off)
            }
            ElfError::InvalidSymbolIndex(idx) => {
                write!(f, "invalid symbol index: {}", idx)
            }
            ElfError::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for ElfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ElfError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ElfError {
    fn from(e: std::io::Error) -> Self {
        ElfError::IoError(e)
    }
}

// ===========================================================================
// Byte Reading Helpers (Little-Endian)
// ===========================================================================

/// Read a little-endian `u16` from `data` at the given byte `offset`.
pub fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, ElfError> {
    if offset + 2 > data.len() {
        return Err(ElfError::TruncatedData("u16"));
    }
    Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

/// Read a little-endian `u32` from `data` at the given byte `offset`.
pub fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, ElfError> {
    if offset + 4 > data.len() {
        return Err(ElfError::TruncatedData("u32"));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian `u64` from `data` at the given byte `offset`.
pub fn read_u64_le(data: &[u8], offset: usize) -> Result<u64, ElfError> {
    if offset + 8 > data.len() {
        return Err(ElfError::TruncatedData("u64"));
    }
    Ok(u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Read a little-endian `i32` from `data` at the given byte `offset`.
pub fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, ElfError> {
    if offset + 4 > data.len() {
        return Err(ElfError::TruncatedData("i32"));
    }
    Ok(i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian `i64` from `data` at the given byte `offset`.
pub fn read_i64_le(data: &[u8], offset: usize) -> Result<i64, ElfError> {
    if offset + 8 > data.len() {
        return Err(ElfError::TruncatedData("i64"));
    }
    Ok(i64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

// ===========================================================================
// Byte Writing Helpers (Little-Endian)
// ===========================================================================

/// Append a little-endian `u16` to `buf`.
pub fn write_u16_le(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a little-endian `u32` to `buf`.
pub fn write_u32_le(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a little-endian `u64` to `buf`.
pub fn write_u64_le(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a little-endian `i32` to `buf`.
pub fn write_i32_le(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Append a little-endian `i64` to `buf`.
pub fn write_i64_le(buf: &mut Vec<u8>, value: i64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

// ===========================================================================
// Symbol table helpers
// ===========================================================================

/// Extract the symbol binding from the `st_info` byte (upper 4 bits).
#[inline]
pub fn elf_st_bind(info: u8) -> u8 {
    info >> 4
}

/// Extract the symbol type from the `st_info` byte (lower 4 bits).
#[inline]
pub fn elf_st_type(info: u8) -> u8 {
    info & 0xf
}

/// Construct an `st_info` byte from binding and type values.
#[inline]
pub fn elf_st_info(bind: u8, stype: u8) -> u8 {
    (bind << 4) | (stype & 0xf)
}

/// Extract the symbol visibility from the `st_other` byte (lower 2 bits).
#[inline]
pub fn elf_st_visibility(other: u8) -> u8 {
    other & 0x3
}

// ===========================================================================
// Relocation info helpers
// ===========================================================================

/// Extract the symbol index from a 64-bit relocation `r_info` field.
#[inline]
pub fn elf64_r_sym(info: u64) -> u32 {
    (info >> 32) as u32
}

/// Extract the relocation type from a 64-bit relocation `r_info` field.
#[inline]
pub fn elf64_r_type(info: u64) -> u32 {
    info as u32
}

/// Construct a 64-bit `r_info` field from symbol index and relocation type.
#[inline]
pub fn elf64_r_info(sym: u32, rtype: u32) -> u64 {
    ((sym as u64) << 32) | (rtype as u64)
}

/// Extract the symbol index from a 32-bit relocation `r_info` field.
#[inline]
pub fn elf32_r_sym(info: u32) -> u32 {
    info >> 8
}

/// Extract the relocation type from a 32-bit relocation `r_info` field.
#[inline]
pub fn elf32_r_type(info: u32) -> u32 {
    info & 0xff
}

/// Construct a 32-bit `r_info` field from symbol index and relocation type.
#[inline]
pub fn elf32_r_info(sym: u32, rtype: u32) -> u32 {
    (sym << 8) | (rtype & 0xff)
}

// ===========================================================================
// String table reading
// ===========================================================================

/// Read a null-terminated string from a string table at the given byte offset.
///
/// Returns `Err(ElfError::InvalidStringOffset)` if the offset is out of range
/// or the bytes are not valid UTF-8.
pub fn read_string(strtab: &[u8], offset: u32) -> Result<String, ElfError> {
    let start = offset as usize;
    if start >= strtab.len() {
        return Err(ElfError::InvalidStringOffset(offset));
    }
    let end = strtab[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|pos| start + pos)
        .unwrap_or(strtab.len());
    String::from_utf8(strtab[start..end].to_vec())
        .map_err(|_| ElfError::InvalidStringOffset(offset))
}

// ===========================================================================
// Target-to-ELF mapping helpers
// ===========================================================================

/// Return the ELF `e_machine` value for a given target configuration.
///
/// Maps each architecture to the appropriate ELF machine constant. Falls back
/// to `target.elf_machine` for any architecture not explicitly matched.
pub fn machine_for_target(target: &TargetConfig) -> u16 {
    match target.arch {
        Architecture::X86_64 => EM_X86_64,
        Architecture::I686 => EM_386,
        Architecture::Aarch64 => EM_AARCH64,
        Architecture::Riscv64 => EM_RISCV,
    }
}

/// Return the ELF class byte (ELFCLASS32 or ELFCLASS64) for a given target.
pub fn class_for_target(target: &TargetConfig) -> u8 {
    match target.elf_class {
        ElfClass::Elf32 => ELFCLASS32,
        ElfClass::Elf64 => ELFCLASS64,
    }
}

/// Returns `true` if the given ELF class is 64-bit.
#[inline]
pub fn is_64bit(class: u8) -> bool {
    class == ELFCLASS64
}

/// Return the ELF header size in bytes for the given class.
#[inline]
pub fn ehdr_size(class: u8) -> usize {
    if is_64bit(class) {
        ELF64_EHDR_SIZE
    } else {
        ELF32_EHDR_SIZE
    }
}

/// Return the program header entry size in bytes for the given class.
#[inline]
pub fn phdr_size(class: u8) -> usize {
    if is_64bit(class) {
        56
    } else {
        32
    }
}

/// Return the section header entry size in bytes for the given class.
#[inline]
pub fn shdr_size(class: u8) -> usize {
    if is_64bit(class) {
        64
    } else {
        40
    }
}

// ===========================================================================
// ELF64 Header (64 bytes)
// ===========================================================================

/// ELF64 file header, exactly 64 bytes when serialized.
#[derive(Debug, Clone)]
pub struct Elf64Ehdr {
    /// ELF identification bytes (magic, class, data, version, OS/ABI, padding).
    pub e_ident: [u8; 16],
    /// Object file type (ET_EXEC, ET_DYN, ET_REL, etc.).
    pub e_type: u16,
    /// Target architecture (EM_X86_64, EM_AARCH64, etc.).
    pub e_machine: u16,
    /// Object file version (EV_CURRENT).
    pub e_version: u32,
    /// Entry point virtual address.
    pub e_entry: u64,
    /// Program header table file offset.
    pub e_phoff: u64,
    /// Section header table file offset.
    pub e_shoff: u64,
    /// Processor-specific flags.
    pub e_flags: u32,
    /// ELF header size in bytes (64 for ELF64).
    pub e_ehsize: u16,
    /// Size of one program header entry (56 for ELF64).
    pub e_phentsize: u16,
    /// Number of program header entries.
    pub e_phnum: u16,
    /// Size of one section header entry (64 for ELF64).
    pub e_shentsize: u16,
    /// Number of section header entries.
    pub e_shnum: u16,
    /// Section header string table index.
    pub e_shstrndx: u16,
}

impl Elf64Ehdr {
    /// Deserialize an ELF64 header from raw bytes.
    ///
    /// The input must be at least 64 bytes. Validates magic, class, and data
    /// encoding before parsing the remaining fields.
    pub fn read(data: &[u8]) -> Result<Self, ElfError> {
        if data.len() < ELF64_EHDR_SIZE {
            return Err(ElfError::TruncatedData("ELF64 header"));
        }
        // Validate magic
        if data[0..4] != ELF_MAGIC {
            return Err(ElfError::InvalidMagic);
        }
        // Validate class
        if data[EI_CLASS] != ELFCLASS64 {
            return Err(ElfError::UnsupportedClass(data[EI_CLASS]));
        }
        // Validate data encoding (little-endian)
        if data[EI_DATA] != ELFDATA2LSB {
            return Err(ElfError::UnsupportedEndianness(data[EI_DATA]));
        }

        let mut e_ident = [0u8; 16];
        e_ident.copy_from_slice(&data[0..16]);

        Ok(Elf64Ehdr {
            e_ident,
            e_type: read_u16_le(data, 16)?,
            e_machine: read_u16_le(data, 18)?,
            e_version: read_u32_le(data, 20)?,
            e_entry: read_u64_le(data, 24)?,
            e_phoff: read_u64_le(data, 32)?,
            e_shoff: read_u64_le(data, 40)?,
            e_flags: read_u32_le(data, 48)?,
            e_ehsize: read_u16_le(data, 52)?,
            e_phentsize: read_u16_le(data, 54)?,
            e_phnum: read_u16_le(data, 56)?,
            e_shentsize: read_u16_le(data, 58)?,
            e_shnum: read_u16_le(data, 60)?,
            e_shstrndx: read_u16_le(data, 62)?,
        })
    }

    /// Serialize this ELF64 header to a 64-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ELF64_EHDR_SIZE);
        buf.extend_from_slice(&self.e_ident);
        write_u16_le(&mut buf, self.e_type);
        write_u16_le(&mut buf, self.e_machine);
        write_u32_le(&mut buf, self.e_version);
        write_u64_le(&mut buf, self.e_entry);
        write_u64_le(&mut buf, self.e_phoff);
        write_u64_le(&mut buf, self.e_shoff);
        write_u32_le(&mut buf, self.e_flags);
        write_u16_le(&mut buf, self.e_ehsize);
        write_u16_le(&mut buf, self.e_phentsize);
        write_u16_le(&mut buf, self.e_phnum);
        write_u16_le(&mut buf, self.e_shentsize);
        write_u16_le(&mut buf, self.e_shnum);
        write_u16_le(&mut buf, self.e_shstrndx);
        debug_assert_eq!(buf.len(), ELF64_EHDR_SIZE);
        buf
    }
}

// ===========================================================================
// ELF32 Header (52 bytes)
// ===========================================================================

/// ELF32 file header, exactly 52 bytes when serialized.
#[derive(Debug, Clone)]
pub struct Elf32Ehdr {
    pub e_ident: [u8; 16],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    /// Entry point address (32-bit).
    pub e_entry: u32,
    /// Program header table file offset (32-bit).
    pub e_phoff: u32,
    /// Section header table file offset (32-bit).
    pub e_shoff: u32,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

impl Elf32Ehdr {
    /// Deserialize an ELF32 header from raw bytes (at least 52 bytes).
    pub fn read(data: &[u8]) -> Result<Self, ElfError> {
        if data.len() < ELF32_EHDR_SIZE {
            return Err(ElfError::TruncatedData("ELF32 header"));
        }
        if data[0..4] != ELF_MAGIC {
            return Err(ElfError::InvalidMagic);
        }
        if data[EI_CLASS] != ELFCLASS32 {
            return Err(ElfError::UnsupportedClass(data[EI_CLASS]));
        }
        if data[EI_DATA] != ELFDATA2LSB {
            return Err(ElfError::UnsupportedEndianness(data[EI_DATA]));
        }

        let mut e_ident = [0u8; 16];
        e_ident.copy_from_slice(&data[0..16]);

        Ok(Elf32Ehdr {
            e_ident,
            e_type: read_u16_le(data, 16)?,
            e_machine: read_u16_le(data, 18)?,
            e_version: read_u32_le(data, 20)?,
            e_entry: read_u32_le(data, 24)?,
            e_phoff: read_u32_le(data, 28)?,
            e_shoff: read_u32_le(data, 32)?,
            e_flags: read_u32_le(data, 36)?,
            e_ehsize: read_u16_le(data, 40)?,
            e_phentsize: read_u16_le(data, 42)?,
            e_phnum: read_u16_le(data, 44)?,
            e_shentsize: read_u16_le(data, 46)?,
            e_shnum: read_u16_le(data, 48)?,
            e_shstrndx: read_u16_le(data, 50)?,
        })
    }

    /// Serialize this ELF32 header to a 52-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ELF32_EHDR_SIZE);
        buf.extend_from_slice(&self.e_ident);
        write_u16_le(&mut buf, self.e_type);
        write_u16_le(&mut buf, self.e_machine);
        write_u32_le(&mut buf, self.e_version);
        write_u32_le(&mut buf, self.e_entry);
        write_u32_le(&mut buf, self.e_phoff);
        write_u32_le(&mut buf, self.e_shoff);
        write_u32_le(&mut buf, self.e_flags);
        write_u16_le(&mut buf, self.e_ehsize);
        write_u16_le(&mut buf, self.e_phentsize);
        write_u16_le(&mut buf, self.e_phnum);
        write_u16_le(&mut buf, self.e_shentsize);
        write_u16_le(&mut buf, self.e_shnum);
        write_u16_le(&mut buf, self.e_shstrndx);
        debug_assert_eq!(buf.len(), ELF32_EHDR_SIZE);
        buf
    }
}

// ===========================================================================
// Section Headers
// ===========================================================================

/// ELF64 section header (64 bytes).
#[derive(Debug, Clone, Default)]
pub struct Elf64Shdr {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
}

impl Elf64Shdr {
    /// Read an ELF64 section header from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 64 > data.len() {
            return Err(ElfError::TruncatedData("ELF64 section header"));
        }
        Ok(Elf64Shdr {
            sh_name: read_u32_le(data, offset)?,
            sh_type: read_u32_le(data, offset + 4)?,
            sh_flags: read_u64_le(data, offset + 8)?,
            sh_addr: read_u64_le(data, offset + 16)?,
            sh_offset: read_u64_le(data, offset + 24)?,
            sh_size: read_u64_le(data, offset + 32)?,
            sh_link: read_u32_le(data, offset + 40)?,
            sh_info: read_u32_le(data, offset + 44)?,
            sh_addralign: read_u64_le(data, offset + 48)?,
            sh_entsize: read_u64_le(data, offset + 56)?,
        })
    }

    /// Serialize this section header to a 64-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u32_le(&mut buf, self.sh_name);
        write_u32_le(&mut buf, self.sh_type);
        write_u64_le(&mut buf, self.sh_flags);
        write_u64_le(&mut buf, self.sh_addr);
        write_u64_le(&mut buf, self.sh_offset);
        write_u64_le(&mut buf, self.sh_size);
        write_u32_le(&mut buf, self.sh_link);
        write_u32_le(&mut buf, self.sh_info);
        write_u64_le(&mut buf, self.sh_addralign);
        write_u64_le(&mut buf, self.sh_entsize);
        debug_assert_eq!(buf.len(), 64);
        buf
    }
}

/// ELF32 section header (40 bytes).
#[derive(Debug, Clone, Default)]
pub struct Elf32Shdr {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u32,
    pub sh_addr: u32,
    pub sh_offset: u32,
    pub sh_size: u32,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u32,
    pub sh_entsize: u32,
}

impl Elf32Shdr {
    /// Read an ELF32 section header from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 40 > data.len() {
            return Err(ElfError::TruncatedData("ELF32 section header"));
        }
        Ok(Elf32Shdr {
            sh_name: read_u32_le(data, offset)?,
            sh_type: read_u32_le(data, offset + 4)?,
            sh_flags: read_u32_le(data, offset + 8)?,
            sh_addr: read_u32_le(data, offset + 12)?,
            sh_offset: read_u32_le(data, offset + 16)?,
            sh_size: read_u32_le(data, offset + 20)?,
            sh_link: read_u32_le(data, offset + 24)?,
            sh_info: read_u32_le(data, offset + 28)?,
            sh_addralign: read_u32_le(data, offset + 32)?,
            sh_entsize: read_u32_le(data, offset + 36)?,
        })
    }

    /// Serialize this section header to a 40-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(40);
        write_u32_le(&mut buf, self.sh_name);
        write_u32_le(&mut buf, self.sh_type);
        write_u32_le(&mut buf, self.sh_flags);
        write_u32_le(&mut buf, self.sh_addr);
        write_u32_le(&mut buf, self.sh_offset);
        write_u32_le(&mut buf, self.sh_size);
        write_u32_le(&mut buf, self.sh_link);
        write_u32_le(&mut buf, self.sh_info);
        write_u32_le(&mut buf, self.sh_addralign);
        write_u32_le(&mut buf, self.sh_entsize);
        debug_assert_eq!(buf.len(), 40);
        buf
    }
}

// ===========================================================================
// Program Headers
// ===========================================================================

/// ELF64 program header (56 bytes).
///
/// Note: In ELF64, `p_flags` comes immediately after `p_type` (offset 4).
#[derive(Debug, Clone, Default)]
pub struct Elf64Phdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

impl Elf64Phdr {
    /// Read an ELF64 program header from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 56 > data.len() {
            return Err(ElfError::TruncatedData("ELF64 program header"));
        }
        Ok(Elf64Phdr {
            p_type: read_u32_le(data, offset)?,
            p_flags: read_u32_le(data, offset + 4)?,
            p_offset: read_u64_le(data, offset + 8)?,
            p_vaddr: read_u64_le(data, offset + 16)?,
            p_paddr: read_u64_le(data, offset + 24)?,
            p_filesz: read_u64_le(data, offset + 32)?,
            p_memsz: read_u64_le(data, offset + 40)?,
            p_align: read_u64_le(data, offset + 48)?,
        })
    }

    /// Serialize this program header to a 56-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(56);
        write_u32_le(&mut buf, self.p_type);
        write_u32_le(&mut buf, self.p_flags);
        write_u64_le(&mut buf, self.p_offset);
        write_u64_le(&mut buf, self.p_vaddr);
        write_u64_le(&mut buf, self.p_paddr);
        write_u64_le(&mut buf, self.p_filesz);
        write_u64_le(&mut buf, self.p_memsz);
        write_u64_le(&mut buf, self.p_align);
        debug_assert_eq!(buf.len(), 56);
        buf
    }
}

/// ELF32 program header (32 bytes).
///
/// IMPORTANT: In ELF32, `p_flags` is at byte offset 24 (after `p_memsz`),
/// unlike ELF64 where it is at offset 4. This difference is per the ELF spec.
#[derive(Debug, Clone, Default)]
pub struct Elf32Phdr {
    pub p_type: u32,
    pub p_offset: u32,
    pub p_vaddr: u32,
    pub p_paddr: u32,
    pub p_filesz: u32,
    pub p_memsz: u32,
    pub p_flags: u32,
    pub p_align: u32,
}

impl Elf32Phdr {
    /// Read an ELF32 program header from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 32 > data.len() {
            return Err(ElfError::TruncatedData("ELF32 program header"));
        }
        Ok(Elf32Phdr {
            p_type: read_u32_le(data, offset)?,
            p_offset: read_u32_le(data, offset + 4)?,
            p_vaddr: read_u32_le(data, offset + 8)?,
            p_paddr: read_u32_le(data, offset + 12)?,
            p_filesz: read_u32_le(data, offset + 16)?,
            p_memsz: read_u32_le(data, offset + 20)?,
            p_flags: read_u32_le(data, offset + 24)?,
            p_align: read_u32_le(data, offset + 28)?,
        })
    }

    /// Serialize this program header to a 32-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        write_u32_le(&mut buf, self.p_type);
        write_u32_le(&mut buf, self.p_offset);
        write_u32_le(&mut buf, self.p_vaddr);
        write_u32_le(&mut buf, self.p_paddr);
        write_u32_le(&mut buf, self.p_filesz);
        write_u32_le(&mut buf, self.p_memsz);
        write_u32_le(&mut buf, self.p_flags);
        write_u32_le(&mut buf, self.p_align);
        debug_assert_eq!(buf.len(), 32);
        buf
    }
}

// ===========================================================================
// Symbol Table Entries
// ===========================================================================

/// ELF64 symbol table entry (24 bytes).
///
/// Field order: st_name, st_info, st_other, st_shndx, st_value, st_size.
#[derive(Debug, Clone, Default)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

impl Elf64Sym {
    /// Read an ELF64 symbol table entry from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 24 > data.len() {
            return Err(ElfError::TruncatedData("ELF64 symbol entry"));
        }
        Ok(Elf64Sym {
            st_name: read_u32_le(data, offset)?,
            st_info: data[offset + 4],
            st_other: data[offset + 5],
            st_shndx: read_u16_le(data, offset + 6)?,
            st_value: read_u64_le(data, offset + 8)?,
            st_size: read_u64_le(data, offset + 16)?,
        })
    }

    /// Serialize this symbol entry to a 24-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        write_u32_le(&mut buf, self.st_name);
        buf.push(self.st_info);
        buf.push(self.st_other);
        write_u16_le(&mut buf, self.st_shndx);
        write_u64_le(&mut buf, self.st_value);
        write_u64_le(&mut buf, self.st_size);
        debug_assert_eq!(buf.len(), 24);
        buf
    }
}

/// ELF32 symbol table entry (16 bytes).
///
/// IMPORTANT: Field order differs from ELF64:
/// st_name, st_value, st_size, st_info, st_other, st_shndx.
#[derive(Debug, Clone, Default)]
pub struct Elf32Sym {
    pub st_name: u32,
    pub st_value: u32,
    pub st_size: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
}

impl Elf32Sym {
    /// Read an ELF32 symbol table entry from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 16 > data.len() {
            return Err(ElfError::TruncatedData("ELF32 symbol entry"));
        }
        Ok(Elf32Sym {
            st_name: read_u32_le(data, offset)?,
            st_value: read_u32_le(data, offset + 4)?,
            st_size: read_u32_le(data, offset + 8)?,
            st_info: data[offset + 12],
            st_other: data[offset + 13],
            st_shndx: read_u16_le(data, offset + 14)?,
        })
    }

    /// Serialize this symbol entry to a 16-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        write_u32_le(&mut buf, self.st_name);
        write_u32_le(&mut buf, self.st_value);
        write_u32_le(&mut buf, self.st_size);
        buf.push(self.st_info);
        buf.push(self.st_other);
        write_u16_le(&mut buf, self.st_shndx);
        debug_assert_eq!(buf.len(), 16);
        buf
    }
}

// ===========================================================================
// Relocation Table Entries
// ===========================================================================

/// ELF64 relocation entry with explicit addend (RELA, 24 bytes).
#[derive(Debug, Clone, Default)]
pub struct Elf64Rela {
    pub r_offset: u64,
    pub r_info: u64,
    pub r_addend: i64,
}

impl Elf64Rela {
    /// Read an ELF64 RELA entry from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 24 > data.len() {
            return Err(ElfError::TruncatedData("ELF64 RELA entry"));
        }
        Ok(Elf64Rela {
            r_offset: read_u64_le(data, offset)?,
            r_info: read_u64_le(data, offset + 8)?,
            r_addend: read_i64_le(data, offset + 16)?,
        })
    }

    /// Serialize this RELA entry to a 24-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        write_u64_le(&mut buf, self.r_offset);
        write_u64_le(&mut buf, self.r_info);
        write_i64_le(&mut buf, self.r_addend);
        debug_assert_eq!(buf.len(), 24);
        buf
    }
}

/// ELF32 relocation entry with explicit addend (RELA, 12 bytes).
#[derive(Debug, Clone, Default)]
pub struct Elf32Rela {
    pub r_offset: u32,
    pub r_info: u32,
    pub r_addend: i32,
}

impl Elf32Rela {
    /// Read an ELF32 RELA entry from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 12 > data.len() {
            return Err(ElfError::TruncatedData("ELF32 RELA entry"));
        }
        Ok(Elf32Rela {
            r_offset: read_u32_le(data, offset)?,
            r_info: read_u32_le(data, offset + 4)?,
            r_addend: read_i32_le(data, offset + 8)?,
        })
    }

    /// Serialize this RELA entry to a 12-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12);
        write_u32_le(&mut buf, self.r_offset);
        write_u32_le(&mut buf, self.r_info);
        write_i32_le(&mut buf, self.r_addend);
        debug_assert_eq!(buf.len(), 12);
        buf
    }
}

/// ELF32 relocation entry without addend (REL, 8 bytes).
#[derive(Debug, Clone, Default)]
pub struct Elf32Rel {
    pub r_offset: u32,
    pub r_info: u32,
}

impl Elf32Rel {
    /// Read an ELF32 REL entry from `data` at the given byte offset.
    pub fn read(data: &[u8], offset: usize) -> Result<Self, ElfError> {
        if offset + 8 > data.len() {
            return Err(ElfError::TruncatedData("ELF32 REL entry"));
        }
        Ok(Elf32Rel {
            r_offset: read_u32_le(data, offset)?,
            r_info: read_u32_le(data, offset + 4)?,
        })
    }

    /// Serialize this REL entry to an 8-byte little-endian byte vector.
    pub fn write(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8);
        write_u32_le(&mut buf, self.r_offset);
        write_u32_le(&mut buf, self.r_info);
        debug_assert_eq!(buf.len(), 8);
        buf
    }
}

// ===========================================================================
// ELF Object Reader (Parsed Types)
// ===========================================================================

/// A parsed section from an input ELF object.
#[derive(Debug, Clone)]
pub struct ParsedSection {
    /// Section name (resolved from .shstrtab).
    pub name: String,
    /// Section type (SHT_PROGBITS, SHT_NOBITS, etc.).
    pub section_type: u32,
    /// Section flags (SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR).
    pub flags: u64,
    /// Raw section data bytes (empty for SHT_NOBITS).
    pub data: Vec<u8>,
    /// Required alignment.
    pub alignment: u64,
    /// Link to associated section.
    pub link: u32,
    /// Additional section info.
    pub info: u32,
    /// Entry size for table sections.
    pub entry_size: u64,
}

/// A parsed symbol from an input ELF object.
#[derive(Debug, Clone)]
pub struct ParsedSymbol {
    /// Symbol name (resolved from .strtab).
    pub name: String,
    /// Symbol value (address or offset within section).
    pub value: u64,
    /// Symbol size.
    pub size: u64,
    /// Symbol binding (STB_LOCAL, STB_GLOBAL, STB_WEAK).
    pub binding: u8,
    /// Symbol type (STT_NOTYPE, STT_FUNC, STT_OBJECT, etc.).
    pub symbol_type: u8,
    /// Symbol visibility (STV_DEFAULT, STV_HIDDEN, STV_PROTECTED).
    pub visibility: u8,
    /// Section index this symbol is defined in.
    pub section_index: u16,
}

/// A parsed relocation from an input ELF object.
#[derive(Debug, Clone)]
pub struct ParsedRelocation {
    /// Offset within the target section where the fixup applies.
    pub offset: u64,
    /// Architecture-specific relocation type.
    pub reloc_type: u32,
    /// Index into the symbol table.
    pub symbol_index: u32,
    /// Explicit addend value.
    pub addend: i64,
    /// Index of the section being relocated (the target section).
    pub section_index: usize,
}

/// A fully parsed ELF object file (either 32-bit or 64-bit).
#[derive(Debug, Clone)]
pub struct ElfObject {
    /// ELF class (ELFCLASS32 or ELFCLASS64).
    pub class: u8,
    /// Target machine type (EM_X86_64, EM_386, etc.).
    pub machine: u16,
    /// Parsed sections.
    pub sections: Vec<ParsedSection>,
    /// Parsed symbols.
    pub symbols: Vec<ParsedSymbol>,
    /// Parsed relocations.
    pub relocations: Vec<ParsedRelocation>,
}

impl ElfObject {
    /// Parse an ELF object file from raw bytes.
    ///
    /// Supports both ELF32 and ELF64 formats. Reads the ELF header, section
    /// headers, section names, symbol tables, and relocation sections.
    pub fn parse(data: &[u8]) -> Result<Self, ElfError> {
        if data.len() < EI_NIDENT {
            return Err(ElfError::TruncatedData("ELF identification"));
        }
        if data[0..4] != ELF_MAGIC {
            return Err(ElfError::InvalidMagic);
        }

        let class = data[EI_CLASS];
        if data[EI_DATA] != ELFDATA2LSB {
            return Err(ElfError::UnsupportedEndianness(data[EI_DATA]));
        }

        match class {
            ELFCLASS64 => Self::parse_64(data),
            ELFCLASS32 => Self::parse_32(data),
            _ => Err(ElfError::UnsupportedClass(class)),
        }
    }

    /// Internal: parse a 64-bit ELF object.
    fn parse_64(data: &[u8]) -> Result<Self, ElfError> {
        let ehdr = Elf64Ehdr::read(data)?;
        let shnum = ehdr.e_shnum as usize;
        let shoff = ehdr.e_shoff as usize;
        let shentsize = ehdr.e_shentsize as usize;
        let shstrndx = ehdr.e_shstrndx as usize;

        // Read all section headers
        let mut shdrs = Vec::with_capacity(shnum);
        for i in 0..shnum {
            let offset = shoff + i * shentsize;
            shdrs.push(Elf64Shdr::read(data, offset)?);
        }

        // Read section name string table
        let shstrtab_data = if shstrndx < shnum {
            let sh = &shdrs[shstrndx];
            let start = sh.sh_offset as usize;
            let end = start + sh.sh_size as usize;
            if end > data.len() {
                return Err(ElfError::TruncatedData(".shstrtab"));
            }
            &data[start..end]
        } else {
            &[] as &[u8]
        };

        // Build parsed sections
        let mut sections = Vec::with_capacity(shnum);
        for sh in &shdrs {
            let name = if !shstrtab_data.is_empty() {
                read_string(shstrtab_data, sh.sh_name).unwrap_or_default()
            } else {
                String::new()
            };

            let section_data = if sh.sh_type != SHT_NOBITS && sh.sh_size > 0 {
                let start = sh.sh_offset as usize;
                let end = start + sh.sh_size as usize;
                if end > data.len() {
                    return Err(ElfError::TruncatedData("section data"));
                }
                data[start..end].to_vec()
            } else {
                Vec::new()
            };

            sections.push(ParsedSection {
                name,
                section_type: sh.sh_type,
                flags: sh.sh_flags,
                data: section_data,
                alignment: sh.sh_addralign,
                link: sh.sh_link,
                info: sh.sh_info,
                entry_size: sh.sh_entsize,
            });
        }

        // Parse symbol tables and relocations
        let mut symbols = Vec::new();
        let mut relocations = Vec::new();

        for (sec_idx, sh) in shdrs.iter().enumerate() {
            match sh.sh_type {
                SHT_SYMTAB | SHT_DYNSYM => {
                    let strtab_idx = sh.sh_link as usize;
                    let strtab_data = if strtab_idx < sections.len() {
                        &sections[strtab_idx].data
                    } else {
                        continue;
                    };
                    let sym_data = &sections[sec_idx].data;
                    let entry_size = if sh.sh_entsize > 0 {
                        sh.sh_entsize as usize
                    } else {
                        24
                    };
                    let count = sym_data.len() / entry_size;
                    for i in 0..count {
                        let sym = Elf64Sym::read(sym_data, i * entry_size)?;
                        let sym_name = read_string(strtab_data, sym.st_name).unwrap_or_default();
                        symbols.push(ParsedSymbol {
                            name: sym_name,
                            value: sym.st_value,
                            size: sym.st_size,
                            binding: elf_st_bind(sym.st_info),
                            symbol_type: elf_st_type(sym.st_info),
                            visibility: elf_st_visibility(sym.st_other),
                            section_index: sym.st_shndx,
                        });
                    }
                }
                SHT_RELA => {
                    let target_section = sh.sh_info as usize;
                    let rela_data = &sections[sec_idx].data;
                    let entry_size = if sh.sh_entsize > 0 {
                        sh.sh_entsize as usize
                    } else {
                        24
                    };
                    let count = rela_data.len() / entry_size;
                    for i in 0..count {
                        let rela = Elf64Rela::read(rela_data, i * entry_size)?;
                        relocations.push(ParsedRelocation {
                            offset: rela.r_offset,
                            reloc_type: elf64_r_type(rela.r_info),
                            symbol_index: elf64_r_sym(rela.r_info),
                            addend: rela.r_addend,
                            section_index: target_section,
                        });
                    }
                }
                SHT_REL => {
                    let target_section = sh.sh_info as usize;
                    let rel_data = &sections[sec_idx].data;
                    let entry_size = if sh.sh_entsize > 0 {
                        sh.sh_entsize as usize
                    } else {
                        16
                    };
                    let count = rel_data.len() / entry_size;
                    for i in 0..count {
                        if i * entry_size + 16 > rel_data.len() {
                            break;
                        }
                        let r_offset = read_u64_le(rel_data, i * entry_size)?;
                        let r_info = read_u64_le(rel_data, i * entry_size + 8)?;
                        relocations.push(ParsedRelocation {
                            offset: r_offset,
                            reloc_type: elf64_r_type(r_info),
                            symbol_index: elf64_r_sym(r_info),
                            addend: 0,
                            section_index: target_section,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(ElfObject {
            class: ELFCLASS64,
            machine: ehdr.e_machine,
            sections,
            symbols,
            relocations,
        })
    }

    /// Internal: parse a 32-bit ELF object.
    fn parse_32(data: &[u8]) -> Result<Self, ElfError> {
        let ehdr = Elf32Ehdr::read(data)?;
        let shnum = ehdr.e_shnum as usize;
        let shoff = ehdr.e_shoff as usize;
        let shentsize = ehdr.e_shentsize as usize;
        let shstrndx = ehdr.e_shstrndx as usize;

        // Read all section headers
        let mut shdrs = Vec::with_capacity(shnum);
        for i in 0..shnum {
            let offset = shoff + i * shentsize;
            shdrs.push(Elf32Shdr::read(data, offset)?);
        }

        // Read section name string table
        let shstrtab_data = if shstrndx < shnum {
            let sh = &shdrs[shstrndx];
            let start = sh.sh_offset as usize;
            let end = start + sh.sh_size as usize;
            if end > data.len() {
                return Err(ElfError::TruncatedData(".shstrtab"));
            }
            &data[start..end]
        } else {
            &[] as &[u8]
        };

        // Build parsed sections
        let mut sections = Vec::with_capacity(shnum);
        for sh in &shdrs {
            let name = if !shstrtab_data.is_empty() {
                read_string(shstrtab_data, sh.sh_name).unwrap_or_default()
            } else {
                String::new()
            };

            let section_data = if sh.sh_type != SHT_NOBITS && sh.sh_size > 0 {
                let start = sh.sh_offset as usize;
                let end = start + sh.sh_size as usize;
                if end > data.len() {
                    return Err(ElfError::TruncatedData("section data"));
                }
                data[start..end].to_vec()
            } else {
                Vec::new()
            };

            sections.push(ParsedSection {
                name,
                section_type: sh.sh_type,
                flags: sh.sh_flags as u64,
                data: section_data,
                alignment: sh.sh_addralign as u64,
                link: sh.sh_link,
                info: sh.sh_info,
                entry_size: sh.sh_entsize as u64,
            });
        }

        // Parse symbol tables and relocations
        let mut symbols = Vec::new();
        let mut relocations = Vec::new();

        for (sec_idx, sh) in shdrs.iter().enumerate() {
            match sh.sh_type {
                SHT_SYMTAB | SHT_DYNSYM => {
                    let strtab_idx = sh.sh_link as usize;
                    let strtab_data = if strtab_idx < sections.len() {
                        &sections[strtab_idx].data
                    } else {
                        continue;
                    };
                    let sym_data = &sections[sec_idx].data;
                    let entry_size = if sh.sh_entsize > 0 {
                        sh.sh_entsize as usize
                    } else {
                        16
                    };
                    let count = sym_data.len() / entry_size;
                    for i in 0..count {
                        let sym = Elf32Sym::read(sym_data, i * entry_size)?;
                        let sym_name = read_string(strtab_data, sym.st_name).unwrap_or_default();
                        symbols.push(ParsedSymbol {
                            name: sym_name,
                            value: sym.st_value as u64,
                            size: sym.st_size as u64,
                            binding: elf_st_bind(sym.st_info),
                            symbol_type: elf_st_type(sym.st_info),
                            visibility: elf_st_visibility(sym.st_other),
                            section_index: sym.st_shndx,
                        });
                    }
                }
                SHT_RELA => {
                    let target_section = sh.sh_info as usize;
                    let rela_data = &sections[sec_idx].data;
                    let entry_size = if sh.sh_entsize > 0 {
                        sh.sh_entsize as usize
                    } else {
                        12
                    };
                    let count = rela_data.len() / entry_size;
                    for i in 0..count {
                        let rela = Elf32Rela::read(rela_data, i * entry_size)?;
                        relocations.push(ParsedRelocation {
                            offset: rela.r_offset as u64,
                            reloc_type: elf32_r_type(rela.r_info),
                            symbol_index: elf32_r_sym(rela.r_info),
                            addend: rela.r_addend as i64,
                            section_index: target_section,
                        });
                    }
                }
                SHT_REL => {
                    let target_section = sh.sh_info as usize;
                    let rel_data = &sections[sec_idx].data;
                    let entry_size = if sh.sh_entsize > 0 {
                        sh.sh_entsize as usize
                    } else {
                        8
                    };
                    let count = rel_data.len() / entry_size;
                    for i in 0..count {
                        let rel = Elf32Rel::read(rel_data, i * entry_size)?;
                        relocations.push(ParsedRelocation {
                            offset: rel.r_offset as u64,
                            reloc_type: elf32_r_type(rel.r_info),
                            symbol_index: elf32_r_sym(rel.r_info),
                            addend: 0,
                            section_index: target_section,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(ElfObject {
            class: ELFCLASS32,
            machine: ehdr.e_machine,
            sections,
            symbols,
            relocations,
        })
    }
}

// ===========================================================================
// ELF Writer (Output Types)
// ===========================================================================

/// Header metadata for an output section.
#[derive(Debug, Clone, Default)]
pub struct OutputSectionHeader {
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
    pub sh_link: u32,
    pub sh_info: u32,
}

/// An output section for the ELF writer.
#[derive(Debug, Clone)]
pub struct OutputSection {
    /// Section name (written into .shstrtab).
    pub name: String,
    /// Section data bytes.
    pub data: Vec<u8>,
    /// Section header metadata.
    pub header: OutputSectionHeader,
    /// Optional explicit file offset from the linker's layout engine.
    /// When `Some(offset)`, the ELF writer places this section's data at
    /// exactly that file offset (padding with zeros as needed). When `None`,
    /// the writer auto-assigns offsets sequentially with alignment padding.
    pub file_offset: Option<u64>,
}

/// An output program header for the ELF writer.
#[derive(Debug, Clone, Default)]
pub struct OutputPhdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

/// ELF file writer that produces complete ELF32 or ELF64 binaries.
///
/// Usage:
/// 1. Create with `ElfWriter::new(is_64bit, machine)`
/// 2. Set type and entry point
/// 3. Add sections and program headers
/// 4. Call `write()` to produce the final byte output
pub struct ElfWriter {
    is_64bit: bool,
    machine: u16,
    elf_type: u16,
    entry: u64,
    flags: u32,
    sections: Vec<OutputSection>,
    program_headers: Vec<OutputPhdr>,
}

impl ElfWriter {
    /// Create a new ELF writer for the specified width and machine type.
    pub fn new(is_64bit: bool, machine: u16) -> Self {
        ElfWriter {
            is_64bit,
            machine,
            elf_type: ET_EXEC,
            entry: 0,
            flags: 0,
            sections: Vec::new(),
            program_headers: Vec::new(),
        }
    }

    /// Set the ELF file type (ET_EXEC, ET_DYN, ET_REL).
    pub fn set_type(&mut self, elf_type: u16) {
        self.elf_type = elf_type;
    }

    /// Set the entry point virtual address.
    pub fn set_entry(&mut self, entry: u64) {
        self.entry = entry;
    }

    /// Add an output section.
    pub fn add_section(&mut self, section: OutputSection) {
        self.sections.push(section);
    }

    /// Add a program header.
    pub fn add_program_header(&mut self, phdr: OutputPhdr) {
        self.program_headers.push(phdr);
    }

    /// Write the complete ELF binary to a byte vector.
    ///
    /// Layout:
    /// 1. ELF header
    /// 2. Program headers (immediately after ELF header)
    /// 3. Section data (with alignment padding)
    /// 4. Section header string table (.shstrtab)
    /// 5. Section headers
    pub fn write(&self) -> Vec<u8> {
        let ehdr_sz = if self.is_64bit {
            ELF64_EHDR_SIZE
        } else {
            ELF32_EHDR_SIZE
        };
        let phdr_sz = if self.is_64bit { 56usize } else { 32usize };
        let shdr_sz = if self.is_64bit { 64usize } else { 40usize };

        let phdr_count = self.program_headers.len();
        let phdr_total = phdr_count * phdr_sz;
        let phdr_off = if phdr_count > 0 { ehdr_sz } else { 0 };

        // Build section header string table
        let mut shstrtab = vec![0u8]; // null byte at offset 0
        let mut name_offsets: Vec<u32> = Vec::new();

        // Reserve the null section (index 0) name offset
        // Then section names
        for sec in &self.sections {
            let off = shstrtab.len() as u32;
            name_offsets.push(off);
            shstrtab.extend_from_slice(sec.name.as_bytes());
            shstrtab.push(0);
        }
        // .shstrtab itself
        let shstrtab_name_off = shstrtab.len() as u32;
        shstrtab.extend_from_slice(b".shstrtab\0");

        // Compute section data offsets.
        // When a section has an explicit `file_offset` (from the linker's
        // layout engine), use that offset. Otherwise, auto-assign offsets
        // sequentially with alignment padding. This ensures that section
        // header file offsets and program header file offsets are consistent.
        let mut current_offset = ehdr_sz + phdr_total;
        let mut section_offsets: Vec<usize> = Vec::new();
        for sec in &self.sections {
            if let Some(explicit_offset) = sec.file_offset {
                // Use the linker-assigned file offset. Advance current_offset
                // past this section so non-explicit sections placed after it
                // don't overlap.
                let off = explicit_offset as usize;
                section_offsets.push(off);
                if sec.header.sh_type != SHT_NOBITS {
                    let end = off + sec.data.len();
                    if end > current_offset {
                        current_offset = end;
                    }
                } else {
                    // NOBITS: no file content, but track file position
                    if off > current_offset {
                        current_offset = off;
                    }
                }
            } else {
                let align = if sec.header.sh_addralign > 1 {
                    sec.header.sh_addralign as usize
                } else {
                    1
                };
                // Align current offset
                let remainder = current_offset % align;
                if remainder != 0 {
                    current_offset += align - remainder;
                }
                section_offsets.push(current_offset);
                if sec.header.sh_type != SHT_NOBITS {
                    current_offset += sec.data.len();
                }
            }
        }

        // Align for shstrtab
        let shstrtab_offset = current_offset;
        current_offset += shstrtab.len();

        // Align section headers to 8-byte boundary
        let remainder = current_offset % 8;
        if remainder != 0 {
            current_offset += 8 - remainder;
        }
        let shdr_off = current_offset;

        // Total section count: null + sections + .shstrtab
        let total_sections = 1 + self.sections.len() + 1;
        let shstrndx = total_sections - 1; // .shstrtab is the last section

        // --- Build the output ---
        let total_size = shdr_off + total_sections * shdr_sz;
        let mut output = Vec::with_capacity(total_size);

        // 1. Write ELF header
        if self.is_64bit {
            self.write_elf64_header(
                &mut output,
                phdr_off as u64,
                shdr_off as u64,
                phdr_count as u16,
                total_sections as u16,
                shstrndx as u16,
            );
        } else {
            self.write_elf32_header(
                &mut output,
                phdr_off as u32,
                shdr_off as u32,
                phdr_count as u16,
                total_sections as u16,
                shstrndx as u16,
            );
        }

        // 2. Write program headers
        for phdr in &self.program_headers {
            if self.is_64bit {
                write_u32_le(&mut output, phdr.p_type);
                write_u32_le(&mut output, phdr.p_flags);
                write_u64_le(&mut output, phdr.p_offset);
                write_u64_le(&mut output, phdr.p_vaddr);
                write_u64_le(&mut output, phdr.p_paddr);
                write_u64_le(&mut output, phdr.p_filesz);
                write_u64_le(&mut output, phdr.p_memsz);
                write_u64_le(&mut output, phdr.p_align);
            } else {
                write_u32_le(&mut output, phdr.p_type);
                write_u32_le(&mut output, phdr.p_offset as u32);
                write_u32_le(&mut output, phdr.p_vaddr as u32);
                write_u32_le(&mut output, phdr.p_paddr as u32);
                write_u32_le(&mut output, phdr.p_filesz as u32);
                write_u32_le(&mut output, phdr.p_memsz as u32);
                write_u32_le(&mut output, phdr.p_flags);
                write_u32_le(&mut output, phdr.p_align as u32);
            }
        }

        // 3. Write section data with alignment padding
        for (i, sec) in self.sections.iter().enumerate() {
            let target_off = section_offsets[i];
            while output.len() < target_off {
                output.push(0);
            }
            if sec.header.sh_type != SHT_NOBITS {
                output.extend_from_slice(&sec.data);
            }
        }

        // 4. Write .shstrtab data
        while output.len() < shstrtab_offset {
            output.push(0);
        }
        output.extend_from_slice(&shstrtab);

        // Pad to section header alignment
        while output.len() < shdr_off {
            output.push(0);
        }

        // 5. Write section headers
        // First: null section header (all zeroes)
        if self.is_64bit {
            output.extend_from_slice(&Elf64Shdr::default().write());
        } else {
            output.extend_from_slice(&Elf32Shdr::default().write());
        }

        // Section headers for each output section
        for (i, sec) in self.sections.iter().enumerate() {
            let data_size = if sec.header.sh_type == SHT_NOBITS {
                sec.data.len() as u64
            } else {
                sec.data.len() as u64
            };
            if self.is_64bit {
                let shdr = Elf64Shdr {
                    sh_name: name_offsets[i],
                    sh_type: sec.header.sh_type,
                    sh_flags: sec.header.sh_flags,
                    sh_addr: sec.header.sh_addr,
                    sh_offset: section_offsets[i] as u64,
                    sh_size: data_size,
                    sh_link: sec.header.sh_link,
                    sh_info: sec.header.sh_info,
                    sh_addralign: sec.header.sh_addralign,
                    sh_entsize: sec.header.sh_entsize,
                };
                output.extend_from_slice(&shdr.write());
            } else {
                let shdr = Elf32Shdr {
                    sh_name: name_offsets[i],
                    sh_type: sec.header.sh_type,
                    sh_flags: sec.header.sh_flags as u32,
                    sh_addr: sec.header.sh_addr as u32,
                    sh_offset: section_offsets[i] as u32,
                    sh_size: data_size as u32,
                    sh_link: sec.header.sh_link,
                    sh_info: sec.header.sh_info,
                    sh_addralign: sec.header.sh_addralign as u32,
                    sh_entsize: sec.header.sh_entsize as u32,
                };
                output.extend_from_slice(&shdr.write());
            }
        }

        // .shstrtab section header
        if self.is_64bit {
            let shdr = Elf64Shdr {
                sh_name: shstrtab_name_off,
                sh_type: SHT_STRTAB,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: shstrtab_offset as u64,
                sh_size: shstrtab.len() as u64,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 1,
                sh_entsize: 0,
            };
            output.extend_from_slice(&shdr.write());
        } else {
            let shdr = Elf32Shdr {
                sh_name: shstrtab_name_off,
                sh_type: SHT_STRTAB,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: shstrtab_offset as u32,
                sh_size: shstrtab.len() as u32,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 1,
                sh_entsize: 0,
            };
            output.extend_from_slice(&shdr.write());
        }

        output
    }

    /// Internal: write an ELF64 header to the output buffer.
    fn write_elf64_header(
        &self,
        buf: &mut Vec<u8>,
        phoff: u64,
        shoff: u64,
        phnum: u16,
        shnum: u16,
        shstrndx: u16,
    ) {
        let mut ident = [0u8; EI_NIDENT];
        ident[EI_MAG0] = ELF_MAGIC[0];
        ident[EI_MAG0 + 1] = ELF_MAGIC[1];
        ident[EI_MAG0 + 2] = ELF_MAGIC[2];
        ident[EI_MAG0 + 3] = ELF_MAGIC[3];
        ident[EI_CLASS] = ELFCLASS64;
        ident[EI_DATA] = ELFDATA2LSB;
        ident[EI_VERSION] = EV_CURRENT;
        ident[EI_OSABI] = ELFOSABI_NONE;

        buf.extend_from_slice(&ident);
        write_u16_le(buf, self.elf_type);
        write_u16_le(buf, self.machine);
        write_u32_le(buf, EV_CURRENT as u32);
        write_u64_le(buf, self.entry);
        write_u64_le(buf, phoff);
        write_u64_le(buf, shoff);
        write_u32_le(buf, self.flags);
        write_u16_le(buf, ELF64_EHDR_SIZE as u16);
        write_u16_le(buf, 56u16); // phentsize
        write_u16_le(buf, phnum);
        write_u16_le(buf, 64u16); // shentsize
        write_u16_le(buf, shnum);
        write_u16_le(buf, shstrndx);
    }

    /// Internal: write an ELF32 header to the output buffer.
    fn write_elf32_header(
        &self,
        buf: &mut Vec<u8>,
        phoff: u32,
        shoff: u32,
        phnum: u16,
        shnum: u16,
        shstrndx: u16,
    ) {
        let mut ident = [0u8; EI_NIDENT];
        ident[EI_MAG0] = ELF_MAGIC[0];
        ident[EI_MAG0 + 1] = ELF_MAGIC[1];
        ident[EI_MAG0 + 2] = ELF_MAGIC[2];
        ident[EI_MAG0 + 3] = ELF_MAGIC[3];
        ident[EI_CLASS] = ELFCLASS32;
        ident[EI_DATA] = ELFDATA2LSB;
        ident[EI_VERSION] = EV_CURRENT;
        ident[EI_OSABI] = ELFOSABI_NONE;

        buf.extend_from_slice(&ident);
        write_u16_le(buf, self.elf_type);
        write_u16_le(buf, self.machine);
        write_u32_le(buf, EV_CURRENT as u32);
        write_u32_le(buf, self.entry as u32);
        write_u32_le(buf, phoff);
        write_u32_le(buf, shoff);
        write_u32_le(buf, self.flags);
        write_u16_le(buf, ELF32_EHDR_SIZE as u16);
        write_u16_le(buf, 32u16); // phentsize
        write_u16_le(buf, phnum);
        write_u16_le(buf, 40u16); // shentsize
        write_u16_le(buf, shnum);
        write_u16_le(buf, shstrndx);
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ELF Constants
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf_magic() {
        assert_eq!(ELF_MAGIC, [0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_elf_class_constants() {
        assert_eq!(ELFCLASSNONE, 0);
        assert_eq!(ELFCLASS32, 1);
        assert_eq!(ELFCLASS64, 2);
    }

    #[test]
    fn test_machine_type_constants() {
        assert_eq!(EM_386, 3);
        assert_eq!(EM_X86_64, 62);
        assert_eq!(EM_AARCH64, 183);
        assert_eq!(EM_RISCV, 243);
    }

    #[test]
    fn test_elf_type_constants() {
        assert_eq!(ET_NONE, 0);
        assert_eq!(ET_REL, 1);
        assert_eq!(ET_EXEC, 2);
        assert_eq!(ET_DYN, 3);
        assert_eq!(ET_CORE, 4);
    }

    #[test]
    fn test_program_header_type_constants() {
        assert_eq!(PT_NULL, 0);
        assert_eq!(PT_LOAD, 1);
        assert_eq!(PT_DYNAMIC, 2);
        assert_eq!(PT_INTERP, 3);
        assert_eq!(PT_NOTE, 4);
        assert_eq!(PT_PHDR, 6);
        assert_eq!(PT_GNU_STACK, 0x6474e551);
        assert_eq!(PT_GNU_RELRO, 0x6474e552);
    }

    #[test]
    fn test_program_header_flag_constants() {
        assert_eq!(PF_X, 0x1);
        assert_eq!(PF_W, 0x2);
        assert_eq!(PF_R, 0x4);
    }

    #[test]
    fn test_section_header_type_constants() {
        assert_eq!(SHT_NULL, 0);
        assert_eq!(SHT_PROGBITS, 1);
        assert_eq!(SHT_SYMTAB, 2);
        assert_eq!(SHT_STRTAB, 3);
        assert_eq!(SHT_RELA, 4);
        assert_eq!(SHT_HASH, 5);
        assert_eq!(SHT_DYNAMIC, 6);
        assert_eq!(SHT_NOTE, 7);
        assert_eq!(SHT_NOBITS, 8);
        assert_eq!(SHT_REL, 9);
        assert_eq!(SHT_DYNSYM, 11);
    }

    #[test]
    fn test_section_header_flag_constants() {
        assert_eq!(SHF_WRITE, 0x1);
        assert_eq!(SHF_ALLOC, 0x2);
        assert_eq!(SHF_EXECINSTR, 0x4);
    }

    #[test]
    fn test_special_section_indices() {
        assert_eq!(SHN_UNDEF, 0);
        assert_eq!(SHN_ABS, 0xFFF1);
        assert_eq!(SHN_COMMON, 0xFFF2);
    }

    #[test]
    fn test_symbol_binding_constants() {
        assert_eq!(STB_LOCAL, 0);
        assert_eq!(STB_GLOBAL, 1);
        assert_eq!(STB_WEAK, 2);
    }

    #[test]
    fn test_symbol_type_constants() {
        assert_eq!(STT_NOTYPE, 0);
        assert_eq!(STT_OBJECT, 1);
        assert_eq!(STT_FUNC, 2);
        assert_eq!(STT_SECTION, 3);
        assert_eq!(STT_FILE, 4);
    }

    #[test]
    fn test_symbol_visibility_constants() {
        assert_eq!(STV_DEFAULT, 0);
        assert_eq!(STV_HIDDEN, 2);
        assert_eq!(STV_PROTECTED, 3);
    }

    // -----------------------------------------------------------------------
    // Byte Read/Write Helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_u16_le() {
        let data = [0x34, 0x12];
        assert_eq!(read_u16_le(&data, 0).unwrap(), 0x1234);
    }

    #[test]
    fn test_read_u32_le() {
        let data = [0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_u32_le(&data, 0).unwrap(), 0x12345678);
    }

    #[test]
    fn test_read_u64_le() {
        let data = [0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_u64_le(&data, 0).unwrap(), 0x1234567890ABCDEF);
    }

    #[test]
    fn test_read_i32_le() {
        // -1 in little-endian
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(read_i32_le(&data, 0).unwrap(), -1);
    }

    #[test]
    fn test_read_i64_le() {
        // -1 in little-endian
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(read_i64_le(&data, 0).unwrap(), -1);
    }

    #[test]
    fn test_read_truncated() {
        let data = [0x00];
        assert!(read_u16_le(&data, 0).is_err());
        assert!(read_u32_le(&data, 0).is_err());
        assert!(read_u64_le(&data, 0).is_err());
    }

    #[test]
    fn test_write_u16_le() {
        let mut buf = Vec::new();
        write_u16_le(&mut buf, 0x1234);
        assert_eq!(buf, [0x34, 0x12]);
    }

    #[test]
    fn test_write_u32_le() {
        let mut buf = Vec::new();
        write_u32_le(&mut buf, 0x12345678);
        assert_eq!(buf, [0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_write_u64_le() {
        let mut buf = Vec::new();
        write_u64_le(&mut buf, 0x1234567890ABCDEF);
        assert_eq!(buf, [0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_write_i32_le() {
        let mut buf = Vec::new();
        write_i32_le(&mut buf, -1);
        assert_eq!(buf, [0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_write_i64_le() {
        let mut buf = Vec::new();
        write_i64_le(&mut buf, -1);
        assert_eq!(buf, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    // -----------------------------------------------------------------------
    // Symbol table helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf_st_info_roundtrip() {
        let info = elf_st_info(STB_GLOBAL, STT_FUNC);
        assert_eq!(elf_st_bind(info), STB_GLOBAL);
        assert_eq!(elf_st_type(info), STT_FUNC);
    }

    #[test]
    fn test_elf_st_visibility_helper() {
        assert_eq!(elf_st_visibility(0x02), STV_HIDDEN);
        assert_eq!(elf_st_visibility(0x00), STV_DEFAULT);
        assert_eq!(elf_st_visibility(0x03), STV_PROTECTED);
    }

    // -----------------------------------------------------------------------
    // Relocation info helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf64_r_info_roundtrip() {
        let info = elf64_r_info(42, 7);
        assert_eq!(elf64_r_sym(info), 42);
        assert_eq!(elf64_r_type(info), 7);
    }

    #[test]
    fn test_elf32_r_info_roundtrip() {
        let info = elf32_r_info(42, 7);
        assert_eq!(elf32_r_sym(info), 42);
        assert_eq!(elf32_r_type(info), 7);
    }

    // -----------------------------------------------------------------------
    // String table reading
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_string() {
        let strtab = b"\0hello\0world\0";
        assert_eq!(read_string(strtab, 0).unwrap(), "");
        assert_eq!(read_string(strtab, 1).unwrap(), "hello");
        assert_eq!(read_string(strtab, 7).unwrap(), "world");
    }

    #[test]
    fn test_read_string_invalid_offset() {
        let strtab = b"\0hello\0";
        assert!(read_string(strtab, 100).is_err());
    }

    // -----------------------------------------------------------------------
    // ELF64 Header serialization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf64_ehdr_roundtrip() {
        let mut ident = [0u8; 16];
        ident[0] = 0x7f;
        ident[1] = b'E';
        ident[2] = b'L';
        ident[3] = b'F';
        ident[EI_CLASS] = ELFCLASS64;
        ident[EI_DATA] = ELFDATA2LSB;
        ident[EI_VERSION] = EV_CURRENT;
        ident[EI_OSABI] = ELFOSABI_NONE;

        let hdr = Elf64Ehdr {
            e_ident: ident,
            e_type: ET_EXEC,
            e_machine: EM_X86_64,
            e_version: 1,
            e_entry: 0x401000,
            e_phoff: 64,
            e_shoff: 0x1000,
            e_flags: 0,
            e_ehsize: 64,
            e_phentsize: 56,
            e_phnum: 3,
            e_shentsize: 64,
            e_shnum: 10,
            e_shstrndx: 9,
        };

        let bytes = hdr.write();
        assert_eq!(bytes.len(), ELF64_EHDR_SIZE);

        let restored = Elf64Ehdr::read(&bytes).unwrap();
        assert_eq!(restored.e_type, ET_EXEC);
        assert_eq!(restored.e_machine, EM_X86_64);
        assert_eq!(restored.e_entry, 0x401000);
        assert_eq!(restored.e_phoff, 64);
        assert_eq!(restored.e_shoff, 0x1000);
        assert_eq!(restored.e_phnum, 3);
        assert_eq!(restored.e_shnum, 10);
        assert_eq!(restored.e_shstrndx, 9);
    }

    #[test]
    fn test_elf64_ehdr_size() {
        assert_eq!(ELF64_EHDR_SIZE, 64);
    }

    // -----------------------------------------------------------------------
    // ELF32 Header serialization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf32_ehdr_roundtrip() {
        let mut ident = [0u8; 16];
        ident[0] = 0x7f;
        ident[1] = b'E';
        ident[2] = b'L';
        ident[3] = b'F';
        ident[EI_CLASS] = ELFCLASS32;
        ident[EI_DATA] = ELFDATA2LSB;
        ident[EI_VERSION] = EV_CURRENT;

        let hdr = Elf32Ehdr {
            e_ident: ident,
            e_type: ET_EXEC,
            e_machine: EM_386,
            e_version: 1,
            e_entry: 0x08048000,
            e_phoff: 52,
            e_shoff: 0x800,
            e_flags: 0,
            e_ehsize: 52,
            e_phentsize: 32,
            e_phnum: 2,
            e_shentsize: 40,
            e_shnum: 5,
            e_shstrndx: 4,
        };

        let bytes = hdr.write();
        assert_eq!(bytes.len(), ELF32_EHDR_SIZE);

        let restored = Elf32Ehdr::read(&bytes).unwrap();
        assert_eq!(restored.e_type, ET_EXEC);
        assert_eq!(restored.e_machine, EM_386);
        assert_eq!(restored.e_entry, 0x08048000);
        assert_eq!(restored.e_phnum, 2);
        assert_eq!(restored.e_shnum, 5);
    }

    #[test]
    fn test_elf32_ehdr_size() {
        assert_eq!(ELF32_EHDR_SIZE, 52);
    }

    // -----------------------------------------------------------------------
    // Error on invalid magic
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_invalid_magic() {
        let data = [0x00u8; 64];
        assert!(Elf64Ehdr::read(&data).is_err());
        let data32 = [0x00u8; 52];
        assert!(Elf32Ehdr::read(&data32).is_err());
    }

    // -----------------------------------------------------------------------
    // Error on truncated input
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_truncated_elf64() {
        let data = [0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(Elf64Ehdr::read(&data).is_err());
    }

    #[test]
    fn test_error_truncated_elf32() {
        let data = [0x7f, b'E', b'L', b'F', 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(Elf32Ehdr::read(&data).is_err());
    }

    // -----------------------------------------------------------------------
    // Section header serialization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf64_shdr_roundtrip() {
        let shdr = Elf64Shdr {
            sh_name: 42,
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            sh_addr: 0x401000,
            sh_offset: 0x1000,
            sh_size: 0x200,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 16,
            sh_entsize: 0,
        };
        let bytes = shdr.write();
        assert_eq!(bytes.len(), 64);
        let restored = Elf64Shdr::read(&bytes, 0).unwrap();
        assert_eq!(restored.sh_name, 42);
        assert_eq!(restored.sh_type, SHT_PROGBITS);
        assert_eq!(restored.sh_flags, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(restored.sh_addr, 0x401000);
        assert_eq!(restored.sh_size, 0x200);
        assert_eq!(restored.sh_addralign, 16);
    }

    #[test]
    fn test_elf32_shdr_roundtrip() {
        let shdr = Elf32Shdr {
            sh_name: 10,
            sh_type: SHT_PROGBITS,
            sh_flags: (SHF_ALLOC | SHF_WRITE) as u32,
            sh_addr: 0x08049000,
            sh_offset: 0x1000,
            sh_size: 0x100,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 4,
            sh_entsize: 0,
        };
        let bytes = shdr.write();
        assert_eq!(bytes.len(), 40);
        let restored = Elf32Shdr::read(&bytes, 0).unwrap();
        assert_eq!(restored.sh_name, 10);
        assert_eq!(restored.sh_addr, 0x08049000);
    }

    // -----------------------------------------------------------------------
    // Program header serialization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf64_phdr_roundtrip() {
        let phdr = Elf64Phdr {
            p_type: PT_LOAD,
            p_flags: PF_R | PF_X,
            p_offset: 0x1000,
            p_vaddr: 0x401000,
            p_paddr: 0x401000,
            p_filesz: 0x200,
            p_memsz: 0x200,
            p_align: 0x1000,
        };
        let bytes = phdr.write();
        assert_eq!(bytes.len(), 56);
        let restored = Elf64Phdr::read(&bytes, 0).unwrap();
        assert_eq!(restored.p_type, PT_LOAD);
        assert_eq!(restored.p_flags, PF_R | PF_X);
        assert_eq!(restored.p_vaddr, 0x401000);
    }

    #[test]
    fn test_elf32_phdr_roundtrip() {
        let phdr = Elf32Phdr {
            p_type: PT_LOAD,
            p_offset: 0x1000,
            p_vaddr: 0x08048000,
            p_paddr: 0x08048000,
            p_filesz: 0x100,
            p_memsz: 0x100,
            p_flags: PF_R | PF_W,
            p_align: 0x1000,
        };
        let bytes = phdr.write();
        assert_eq!(bytes.len(), 32);
        let restored = Elf32Phdr::read(&bytes, 0).unwrap();
        assert_eq!(restored.p_type, PT_LOAD);
        assert_eq!(restored.p_flags, PF_R | PF_W);
        assert_eq!(restored.p_vaddr, 0x08048000);
    }

    #[test]
    fn test_phdr_flags_position_difference() {
        // ELF64: p_flags at offset 4 (right after p_type)
        let phdr64 = Elf64Phdr {
            p_type: PT_LOAD,
            p_flags: 0x05,
            ..Default::default()
        };
        let bytes64 = phdr64.write();
        // p_flags should be at bytes 4..8
        assert_eq!(read_u32_le(&bytes64, 4).unwrap(), 0x05);

        // ELF32: p_flags at offset 24 (after p_memsz)
        let phdr32 = Elf32Phdr {
            p_type: PT_LOAD,
            p_flags: 0x05,
            ..Default::default()
        };
        let bytes32 = phdr32.write();
        // p_flags should be at bytes 24..28
        assert_eq!(read_u32_le(&bytes32, 24).unwrap(), 0x05);
    }

    // -----------------------------------------------------------------------
    // Symbol table entry serialization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf64_sym_roundtrip() {
        let sym = Elf64Sym {
            st_name: 5,
            st_info: elf_st_info(STB_GLOBAL, STT_FUNC),
            st_other: 0,
            st_shndx: 1,
            st_value: 0x401000,
            st_size: 64,
        };
        let bytes = sym.write();
        assert_eq!(bytes.len(), 24);
        let restored = Elf64Sym::read(&bytes, 0).unwrap();
        assert_eq!(restored.st_name, 5);
        assert_eq!(elf_st_bind(restored.st_info), STB_GLOBAL);
        assert_eq!(elf_st_type(restored.st_info), STT_FUNC);
        assert_eq!(restored.st_value, 0x401000);
        assert_eq!(restored.st_size, 64);
    }

    #[test]
    fn test_elf32_sym_roundtrip() {
        let sym = Elf32Sym {
            st_name: 3,
            st_value: 0x08048100,
            st_size: 32,
            st_info: elf_st_info(STB_LOCAL, STT_OBJECT),
            st_other: 0,
            st_shndx: 2,
        };
        let bytes = sym.write();
        assert_eq!(bytes.len(), 16);
        let restored = Elf32Sym::read(&bytes, 0).unwrap();
        assert_eq!(restored.st_name, 3);
        assert_eq!(restored.st_value, 0x08048100);
        assert_eq!(elf_st_bind(restored.st_info), STB_LOCAL);
        assert_eq!(elf_st_type(restored.st_info), STT_OBJECT);
    }

    #[test]
    fn test_sym_field_ordering_difference() {
        // ELF64: st_name(4), st_info(1), st_other(1), st_shndx(2), st_value(8), st_size(8)
        // ELF32: st_name(4), st_value(4), st_size(4), st_info(1), st_other(1), st_shndx(2)
        let sym64 = Elf64Sym {
            st_name: 1,
            st_info: 0x12,
            st_other: 0,
            st_shndx: 3,
            st_value: 0x1000,
            st_size: 0x20,
        };
        let bytes64 = sym64.write();
        // st_info is at byte 4 in ELF64
        assert_eq!(bytes64[4], 0x12);

        let sym32 = Elf32Sym {
            st_name: 1,
            st_value: 0x1000,
            st_size: 0x20,
            st_info: 0x12,
            st_other: 0,
            st_shndx: 3,
        };
        let bytes32 = sym32.write();
        // st_info is at byte 12 in ELF32
        assert_eq!(bytes32[12], 0x12);
    }

    // -----------------------------------------------------------------------
    // Relocation entry serialization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf64_rela_roundtrip() {
        let rela = Elf64Rela {
            r_offset: 0x1000,
            r_info: elf64_r_info(5, 2),
            r_addend: -4,
        };
        let bytes = rela.write();
        assert_eq!(bytes.len(), 24);
        let restored = Elf64Rela::read(&bytes, 0).unwrap();
        assert_eq!(restored.r_offset, 0x1000);
        assert_eq!(elf64_r_sym(restored.r_info), 5);
        assert_eq!(elf64_r_type(restored.r_info), 2);
        assert_eq!(restored.r_addend, -4);
    }

    #[test]
    fn test_elf32_rela_roundtrip() {
        let rela = Elf32Rela {
            r_offset: 0x100,
            r_info: elf32_r_info(3, 1),
            r_addend: -8,
        };
        let bytes = rela.write();
        assert_eq!(bytes.len(), 12);
        let restored = Elf32Rela::read(&bytes, 0).unwrap();
        assert_eq!(restored.r_offset, 0x100);
        assert_eq!(elf32_r_sym(restored.r_info), 3);
        assert_eq!(elf32_r_type(restored.r_info), 1);
        assert_eq!(restored.r_addend, -8);
    }

    #[test]
    fn test_elf32_rel_roundtrip() {
        let rel = Elf32Rel {
            r_offset: 0x200,
            r_info: elf32_r_info(7, 2),
        };
        let bytes = rel.write();
        assert_eq!(bytes.len(), 8);
        let restored = Elf32Rel::read(&bytes, 0).unwrap();
        assert_eq!(restored.r_offset, 0x200);
        assert_eq!(elf32_r_sym(restored.r_info), 7);
        assert_eq!(elf32_r_type(restored.r_info), 2);
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_64bit() {
        assert!(is_64bit(ELFCLASS64));
        assert!(!is_64bit(ELFCLASS32));
        assert!(!is_64bit(ELFCLASSNONE));
    }

    #[test]
    fn test_ehdr_size_fn() {
        assert_eq!(ehdr_size(ELFCLASS64), 64);
        assert_eq!(ehdr_size(ELFCLASS32), 52);
    }

    #[test]
    fn test_phdr_size_fn() {
        assert_eq!(phdr_size(ELFCLASS64), 56);
        assert_eq!(phdr_size(ELFCLASS32), 32);
    }

    #[test]
    fn test_shdr_size_fn() {
        assert_eq!(shdr_size(ELFCLASS64), 64);
        assert_eq!(shdr_size(ELFCLASS32), 40);
    }

    // -----------------------------------------------------------------------
    // ELF Object parsing from constructed minimal header
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_minimal_elf64() {
        // Build a minimal valid ELF64 relocatable with a null section only
        let mut ident = [0u8; 16];
        ident[0..4].copy_from_slice(&ELF_MAGIC);
        ident[EI_CLASS] = ELFCLASS64;
        ident[EI_DATA] = ELFDATA2LSB;
        ident[EI_VERSION] = EV_CURRENT;

        let ehdr = Elf64Ehdr {
            e_ident: ident,
            e_type: ET_REL,
            e_machine: EM_X86_64,
            e_version: 1,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: 64, // section headers right after ehdr
            e_flags: 0,
            e_ehsize: 64,
            e_phentsize: 56,
            e_phnum: 0,
            e_shentsize: 64,
            e_shnum: 1, // just the null section header
            e_shstrndx: 0,
        };

        let mut data = ehdr.write();
        // Append one null section header (64 bytes of zeros)
        data.extend_from_slice(&[0u8; 64]);

        let obj = ElfObject::parse(&data).unwrap();
        assert_eq!(obj.class, ELFCLASS64);
        assert_eq!(obj.machine, EM_X86_64);
        assert_eq!(obj.sections.len(), 1);
        assert_eq!(obj.symbols.len(), 0);
        assert_eq!(obj.relocations.len(), 0);
    }

    #[test]
    fn test_parse_error_on_invalid_magic() {
        let data = vec![0x00u8; 128];
        assert!(ElfObject::parse(&data).is_err());
    }

    #[test]
    fn test_parse_error_on_truncated() {
        let data = vec![0x7f, b'E', b'L', b'F'];
        assert!(ElfObject::parse(&data).is_err());
    }

    // -----------------------------------------------------------------------
    // ELF Writer basic smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf_writer_elf64_basic() {
        let mut writer = ElfWriter::new(true, EM_X86_64);
        writer.set_type(ET_EXEC);
        writer.set_entry(0x401000);

        writer.add_section(OutputSection {
            name: ".text".to_string(),
            data: vec![0xCC; 16], // 16 bytes of INT3
            header: OutputSectionHeader {
                sh_type: SHT_PROGBITS,
                sh_flags: SHF_ALLOC | SHF_EXECINSTR,
                sh_addr: 0x401000,
                sh_addralign: 16,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
            },
            file_offset: None,
        });

        let output = writer.write();
        // Should start with ELF magic
        assert_eq!(&output[0..4], &ELF_MAGIC);
        // Class should be ELFCLASS64
        assert_eq!(output[EI_CLASS], ELFCLASS64);
        // Should be parseable
        let ehdr = Elf64Ehdr::read(&output).unwrap();
        assert_eq!(ehdr.e_type, ET_EXEC);
        assert_eq!(ehdr.e_machine, EM_X86_64);
        assert_eq!(ehdr.e_entry, 0x401000);
        // Should have sections: null + .text + .shstrtab = 3
        assert_eq!(ehdr.e_shnum, 3);
    }

    #[test]
    fn test_elf_writer_elf32_basic() {
        let mut writer = ElfWriter::new(false, EM_386);
        writer.set_type(ET_EXEC);
        writer.set_entry(0x08048000);

        writer.add_section(OutputSection {
            name: ".text".to_string(),
            data: vec![0x90; 8], // 8 bytes of NOP
            header: OutputSectionHeader {
                sh_type: SHT_PROGBITS,
                sh_flags: SHF_ALLOC | SHF_EXECINSTR,
                sh_addr: 0x08048000,
                sh_addralign: 4,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
            },
            file_offset: None,
        });

        let output = writer.write();
        assert_eq!(&output[0..4], &ELF_MAGIC);
        assert_eq!(output[EI_CLASS], ELFCLASS32);
        let ehdr = Elf32Ehdr::read(&output).unwrap();
        assert_eq!(ehdr.e_type, ET_EXEC);
        assert_eq!(ehdr.e_machine, EM_386);
        assert_eq!(ehdr.e_entry, 0x08048000);
        assert_eq!(ehdr.e_shnum, 3);
    }

    // -----------------------------------------------------------------------
    // ElfError Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_elf_error_display() {
        let err = ElfError::InvalidMagic;
        let msg = format!("{}", err);
        assert!(msg.contains("invalid ELF magic"));

        let err = ElfError::TruncatedData("header");
        let msg = format!("{}", err);
        assert!(msg.contains("truncated"));
        assert!(msg.contains("header"));
    }
}
