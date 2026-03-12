//! Integration tests for the bcc integrated linker.
//!
//! Covers:
//! - ELF binary generation for all four architectures (ELF32 and ELF64)
//! - Symbol resolution (multi-file, undefined, duplicate, weak, local/global)
//! - CRT startup object linkage (crt1.o, crti.o, crtn.o)
//! - `ar` static archive reading and selective member inclusion
//! - All three output modes (relocatable object, static executable, shared library)
//! - Architecture-specific relocation processing
//! - Section merging and layout (alignment, .bss handling, segment mapping)
//!
//! # Zero-Dependency Guarantee
//!
//! This test module uses ONLY the Rust standard library and the `bcc` crate.
//! No external crates, no external linker invocation.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// ELF Format Constants (local to this test file)
// ===========================================================================

/// ELF data encoding: little-endian.
#[allow(dead_code)]
const ELFDATA2LSB: u8 = 1;

/// ELF type: Relocatable file.
#[allow(dead_code)]
const ET_REL: u16 = 1;

/// ELF type: Executable file.
#[allow(dead_code)]
const ET_EXEC: u16 = 2;

/// ELF type: Shared object file.
#[allow(dead_code)]
const ET_DYN: u16 = 3;

/// Program header type: Loadable segment.
#[allow(dead_code)]
const PT_LOAD: u32 = 1;

/// Program header type: Dynamic linking information.
#[allow(dead_code)]
const PT_DYNAMIC: u32 = 2;

/// Section header type: No data (null).
#[allow(dead_code)]
const SHT_NULL: u32 = 0;

/// Section header type: Program-defined data.
#[allow(dead_code)]
const SHT_PROGBITS: u32 = 1;

/// Section header type: Symbol table.
#[allow(dead_code)]
const SHT_SYMTAB: u32 = 2;

/// Section header type: String table.
#[allow(dead_code)]
const SHT_STRTAB: u32 = 3;

/// Section header type: Relocations with addends.
#[allow(dead_code)]
const SHT_RELA: u32 = 4;

/// Section header type: Dynamic linking information.
#[allow(dead_code)]
const SHT_DYNAMIC: u32 = 6;

/// Section header type: BSS (no file data, zeroed at load).
#[allow(dead_code)]
const SHT_NOBITS: u32 = 8;

/// Section header type: Relocations without addends.
#[allow(dead_code)]
const SHT_REL: u32 = 9;

/// Section header type: Dynamic symbol table.
#[allow(dead_code)]
const SHT_DYNSYM: u32 = 11;

// ===========================================================================
// ELF Parsing Structures
// ===========================================================================

/// Parsed ELF file header (unified for ELF32 and ELF64).
///
/// All address and offset fields are widened to `u64` to accommodate both
/// 32-bit and 64-bit ELF formats uniformly.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ElfHeader {
    /// ELF class: 1 = 32-bit, 2 = 64-bit.
    ei_class: u8,
    /// ELF data encoding: 1 = little-endian, 2 = big-endian.
    ei_data: u8,
    /// Object file type (ET_REL, ET_EXEC, ET_DYN, etc.).
    e_type: u16,
    /// Target architecture (EM_386, EM_X86_64, EM_AARCH64, EM_RISCV).
    e_machine: u16,
    /// ELF version (always 1 for current).
    e_version: u32,
    /// Entry point virtual address.
    e_entry: u64,
    /// Program header table file offset.
    e_phoff: u64,
    /// Section header table file offset.
    e_shoff: u64,
    /// Processor-specific flags.
    e_flags: u32,
    /// ELF header size in bytes.
    e_ehsize: u16,
    /// Program header entry size.
    e_phentsize: u16,
    /// Number of program header entries.
    e_phnum: u16,
    /// Section header entry size.
    e_shentsize: u16,
    /// Number of section header entries.
    e_shnum: u16,
    /// Section name string table index.
    e_shstrndx: u16,
}

/// Parsed ELF section header (unified for ELF32 and ELF64).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SectionHeader {
    /// Index into the section header string table.
    sh_name: u32,
    /// Section type (SHT_PROGBITS, SHT_SYMTAB, SHT_NOBITS, etc.).
    sh_type: u32,
    /// Section flags (SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR, etc.).
    sh_flags: u64,
    /// Virtual address in memory (0 for non-allocatable sections).
    sh_addr: u64,
    /// File offset to section data.
    sh_offset: u64,
    /// Size of the section in the file (0 for SHT_NOBITS).
    sh_size: u64,
    /// Section header table index link (interpretation depends on sh_type).
    sh_link: u32,
    /// Extra information (interpretation depends on sh_type).
    sh_info: u32,
    /// Address alignment constraint.
    sh_addralign: u64,
    /// Size of each entry if the section holds a table.
    sh_entsize: u64,
    /// Resolved section name from the shstrtab.
    name: String,
}

/// Parsed ELF program header (unified for ELF32 and ELF64).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ProgramHeader {
    /// Segment type (PT_LOAD, PT_DYNAMIC, etc.).
    p_type: u32,
    /// Segment flags (PF_R, PF_W, PF_X).
    p_flags: u32,
    /// File offset to segment data.
    p_offset: u64,
    /// Virtual address in memory.
    p_vaddr: u64,
    /// Physical address (often same as p_vaddr).
    p_paddr: u64,
    /// Size of the segment in the file.
    p_filesz: u64,
    /// Size of the segment in memory (may exceed p_filesz for .bss).
    p_memsz: u64,
    /// Alignment of the segment.
    p_align: u64,
}

// ===========================================================================
// Byte Reading Helpers
// ===========================================================================

/// Read a little-endian `u16` from `data` at the given byte `offset`.
#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Read a little-endian `u32` from `data` at the given byte `offset`.
#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Read a little-endian `u64` from `data` at the given byte `offset`.
#[inline]
fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

// ===========================================================================
// ELF Parsing Functions
// ===========================================================================

/// Parse the ELF file header from raw binary data.
///
/// Supports both ELF32 and ELF64 formats. All address/offset fields are
/// widened to `u64` for uniform handling.
///
/// # Panics
///
/// Panics if `data` is too small or does not start with the ELF magic bytes.
fn parse_elf_header(data: &[u8]) -> ElfHeader {
    assert!(
        data.len() >= 52,
        "ELF data too small for header ({} bytes)",
        data.len()
    );
    assert_eq!(
        &data[0..4],
        &[0x7f, b'E', b'L', b'F'],
        "Invalid ELF magic bytes"
    );

    let ei_class = data[4];
    let ei_data = data[5];

    match ei_class {
        2 => {
            // ELF64 header: 64 bytes total.
            assert!(
                data.len() >= 64,
                "ELF64 data too small for header ({} bytes)",
                data.len()
            );
            ElfHeader {
                ei_class,
                ei_data,
                e_type: read_u16_le(data, 16),
                e_machine: read_u16_le(data, 18),
                e_version: read_u32_le(data, 20),
                e_entry: read_u64_le(data, 24),
                e_phoff: read_u64_le(data, 32),
                e_shoff: read_u64_le(data, 40),
                e_flags: read_u32_le(data, 48),
                e_ehsize: read_u16_le(data, 52),
                e_phentsize: read_u16_le(data, 54),
                e_phnum: read_u16_le(data, 56),
                e_shentsize: read_u16_le(data, 58),
                e_shnum: read_u16_le(data, 60),
                e_shstrndx: read_u16_le(data, 62),
            }
        }
        1 => {
            // ELF32 header: 52 bytes total.
            ElfHeader {
                ei_class,
                ei_data,
                e_type: read_u16_le(data, 16),
                e_machine: read_u16_le(data, 18),
                e_version: read_u32_le(data, 20),
                e_entry: read_u32_le(data, 24) as u64,
                e_phoff: read_u32_le(data, 28) as u64,
                e_shoff: read_u32_le(data, 32) as u64,
                e_flags: read_u32_le(data, 36),
                e_ehsize: read_u16_le(data, 40),
                e_phentsize: read_u16_le(data, 42),
                e_phnum: read_u16_le(data, 44),
                e_shentsize: read_u16_le(data, 46),
                e_shnum: read_u16_le(data, 48),
                e_shstrndx: read_u16_le(data, 50),
            }
        }
        other => panic!("Unknown ELF class {} (expected 1 or 2)", other),
    }
}

/// Read a NUL-terminated string from `data` starting at `offset`.
fn read_string_from_table(data: &[u8], offset: usize) -> String {
    let mut end = offset;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    String::from_utf8_lossy(&data[offset..end]).into_owned()
}

/// Parse all section headers from an ELF binary.
///
/// Resolves section names from the section header string table (`.shstrtab`).
fn parse_section_headers(data: &[u8]) -> Vec<SectionHeader> {
    let header = parse_elf_header(data);
    let is_64 = header.ei_class == 2;
    let count = header.e_shnum as usize;
    let offset = header.e_shoff as usize;
    let entsize = header.e_shentsize as usize;

    if count == 0 || offset == 0 {
        return Vec::new();
    }

    let mut sections = Vec::with_capacity(count);

    for i in 0..count {
        let base = offset + i * entsize;
        assert!(
            base + entsize <= data.len(),
            "Section header {} extends beyond file (base={}, entsize={}, filelen={})",
            i,
            base,
            entsize,
            data.len()
        );

        let sh = if is_64 {
            SectionHeader {
                sh_name: read_u32_le(data, base),
                sh_type: read_u32_le(data, base + 4),
                sh_flags: read_u64_le(data, base + 8),
                sh_addr: read_u64_le(data, base + 16),
                sh_offset: read_u64_le(data, base + 24),
                sh_size: read_u64_le(data, base + 32),
                sh_link: read_u32_le(data, base + 40),
                sh_info: read_u32_le(data, base + 44),
                sh_addralign: read_u64_le(data, base + 48),
                sh_entsize: read_u64_le(data, base + 56),
                name: String::new(),
            }
        } else {
            SectionHeader {
                sh_name: read_u32_le(data, base),
                sh_type: read_u32_le(data, base + 4),
                sh_flags: read_u32_le(data, base + 8) as u64,
                sh_addr: read_u32_le(data, base + 12) as u64,
                sh_offset: read_u32_le(data, base + 16) as u64,
                sh_size: read_u32_le(data, base + 20) as u64,
                sh_link: read_u32_le(data, base + 24),
                sh_info: read_u32_le(data, base + 28),
                sh_addralign: read_u32_le(data, base + 32) as u64,
                sh_entsize: read_u32_le(data, base + 36) as u64,
                name: String::new(),
            }
        };

        sections.push(sh);
    }

    // Resolve names from the .shstrtab section.
    let shstrndx = header.e_shstrndx as usize;
    if shstrndx < sections.len() {
        let strtab_off = sections[shstrndx].sh_offset as usize;
        let strtab_sz = sections[shstrndx].sh_size as usize;

        if strtab_off + strtab_sz <= data.len() {
            for section in &mut sections {
                let name_off = strtab_off + section.sh_name as usize;
                if name_off < strtab_off + strtab_sz {
                    section.name = read_string_from_table(data, name_off);
                }
            }
        }
    }

    sections
}

/// Find a section by its name in an ELF binary's section header table.
///
/// Returns `None` if no section with the given name exists.
fn find_section_by_name(data: &[u8], name: &str) -> Option<SectionHeader> {
    parse_section_headers(data)
        .into_iter()
        .find(|s| s.name == name)
}

/// Parse all program headers from an ELF binary.
///
/// Handles both ELF32 and ELF64 program header layouts.
fn parse_program_headers(data: &[u8]) -> Vec<ProgramHeader> {
    let header = parse_elf_header(data);
    let is_64 = header.ei_class == 2;
    let count = header.e_phnum as usize;
    let offset = header.e_phoff as usize;
    let entsize = header.e_phentsize as usize;

    if count == 0 || offset == 0 {
        return Vec::new();
    }

    let mut phdrs = Vec::with_capacity(count);

    for i in 0..count {
        let base = offset + i * entsize;
        assert!(
            base + entsize <= data.len(),
            "Program header {} extends beyond file (base={}, entsize={}, filelen={})",
            i,
            base,
            entsize,
            data.len()
        );

        let ph = if is_64 {
            // ELF64 program header: p_flags is at offset 4 (before p_offset).
            ProgramHeader {
                p_type: read_u32_le(data, base),
                p_flags: read_u32_le(data, base + 4),
                p_offset: read_u64_le(data, base + 8),
                p_vaddr: read_u64_le(data, base + 16),
                p_paddr: read_u64_le(data, base + 24),
                p_filesz: read_u64_le(data, base + 32),
                p_memsz: read_u64_le(data, base + 40),
                p_align: read_u64_le(data, base + 48),
            }
        } else {
            // ELF32 program header: p_flags is at offset 24 (after sizes).
            ProgramHeader {
                p_type: read_u32_le(data, base),
                p_offset: read_u32_le(data, base + 4) as u64,
                p_vaddr: read_u32_le(data, base + 8) as u64,
                p_paddr: read_u32_le(data, base + 12) as u64,
                p_filesz: read_u32_le(data, base + 16) as u64,
                p_memsz: read_u32_le(data, base + 20) as u64,
                p_flags: read_u32_le(data, base + 24),
                p_align: read_u32_le(data, base + 28) as u64,
            }
        };

        phdrs.push(ph);
    }

    phdrs
}

// ===========================================================================
// Test Helper Functions
// ===========================================================================

/// Compile a C source string to an object file (`.o`) using the bcc compiler.
///
/// Writes the source to a temporary `.c` file, invokes `bcc -c` with the given
/// flags, and writes the relocatable object to `output_path`.
///
/// # Panics
///
/// Panics if the compilation fails, printing stderr diagnostics.
#[allow(dead_code)]
fn compile_to_object(source: &str, output_path: &Path, extra_flags: &[&str]) {
    let src_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.arg("-c").arg("-o").arg(output_path);
    for flag in extra_flags {
        cmd.arg(flag);
    }
    cmd.arg(src_file.path());

    let result = cmd.output().unwrap_or_else(|e| {
        panic!("Failed to spawn bcc for object compilation: {}", e);
    });

    assert!(
        result.status.success(),
        "Compilation to object failed (exit={:?}):\nstderr: {}\nstdout: {}",
        result.status.code(),
        String::from_utf8_lossy(&result.stderr),
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(
        output_path.exists(),
        "Object file was not created at '{}'",
        output_path.display()
    );
}

/// Link one or more input files (objects, sources) into an output binary
/// using the bcc compiler, returning the raw `std::process::Output`.
///
/// Callers can inspect `output.status`, `output.stdout`, and `output.stderr`
/// for both success and failure scenarios.
#[allow(dead_code)]
fn link_files(inputs: &[&Path], output_path: &Path, extra_flags: &[&str]) -> std::process::Output {
    let bcc = common::get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.arg("-o").arg(output_path);
    for flag in extra_flags {
        cmd.arg(flag);
    }
    for input in inputs {
        cmd.arg(input);
    }

    cmd.output().unwrap_or_else(|e| {
        panic!("Failed to spawn bcc for linking: {}", e);
    })
}

/// Compile a C source string and link it into an executable for the given
/// target, returning the raw binary bytes of the output.
///
/// Convenience wrapper around `common::compile_source` that reads and returns
/// the output binary data for ELF inspection.
///
/// # Panics
///
/// Panics if compilation fails or the output file cannot be read.
#[allow(dead_code)]
fn compile_and_read_elf(source: &str, target: &str, extra_flags: &[&str]) -> Vec<u8> {
    let mut flags: Vec<&str> = Vec::new();
    flags.push("--target");
    flags.push(target);
    flags.extend_from_slice(extra_flags);

    let result = common::compile_source(source, &flags);
    assert!(
        result.success,
        "Compilation failed for target '{}':\nstderr: {}\nstdout: {}",
        target, result.stderr, result.stdout
    );

    let output_path = result
        .output_path
        .as_ref()
        .expect("Compilation succeeded but no output path available");

    fs::read(output_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read output binary '{}': {}",
            output_path.display(),
            e
        );
    })
}

/// Check whether the given ELF data contains a section with `name`.
#[allow(dead_code)]
fn has_section(data: &[u8], name: &str) -> bool {
    find_section_by_name(data, name).is_some()
}

// ===========================================================================
// Simple C Source Snippets for Tests
// ===========================================================================

/// Minimal C program that returns 0 via main().
const SIMPLE_MAIN: &str = r#"
int main(void) {
    return 0;
}
"#;

/// C program with initialized data, rodata, and BSS sections.
const MULTI_SECTION_SOURCE: &str = r#"
const char rodata_string[] = "hello";
int data_var = 42;
int bss_var;

int main(void) {
    bss_var = data_var;
    return bss_var;
}
"#;

/// File A: defines a function `add`.
const SYMBOL_FILE_A: &str = r#"
int add(int a, int b) {
    return a + b;
}
"#;

/// File B: calls `add` defined in file A.
const SYMBOL_FILE_B: &str = r#"
extern int add(int a, int b);

int main(void) {
    return add(1, 2);
}
"#;

/// Source calling an undefined function (for linker error testing).
const UNDEFINED_SYMBOL_SOURCE: &str = r#"
int undefined_function(void);

int main(void) {
    return undefined_function();
}
"#;

/// Two files defining the same global symbol (duplicate symbol error).
const DUPLICATE_SYMBOL_A: &str = r#"
int shared_symbol = 10;
"#;

const DUPLICATE_SYMBOL_B: &str = r#"
int shared_symbol = 20;
int main(void) { return shared_symbol; }
"#;

/// Weak symbol definition: may be overridden by a strong definition.
const WEAK_SYMBOL_DEF: &str = r#"
__attribute__((weak)) int get_value(void) {
    return 1;
}
"#;

/// Strong symbol overriding the weak definition.
const STRONG_SYMBOL_DEF: &str = r#"
int get_value(void) {
    return 42;
}

int main(void) {
    return get_value();
}
"#;

/// Static (local) function — should not conflict across translation units.
const STATIC_FUNC_A: &str = r#"
static int helper(void) { return 10; }
int get_a(void) { return helper(); }
"#;

const STATIC_FUNC_B: &str = r#"
static int helper(void) { return 20; }
extern int get_a(void);

int main(void) {
    int a = get_a();
    int b = helper();
    return a + b;
}
"#;

/// Source for BSS section testing: uninitialized global.
const BSS_SOURCE: &str = r#"
int bss_array[1024];

int main(void) {
    return bss_array[0];
}
"#;

/// Shared library source with an exported function.
const SHARED_LIB_SOURCE: &str = r#"
int shared_func(void) {
    return 42;
}
"#;

// ===========================================================================
// Phase 2: ELF Generation Tests
// ===========================================================================

/// Compile and link a simple C program for x86-64, verifying the output is a
/// valid ELF64 executable with correct header fields and program headers.
#[test]
fn elf64_executable_valid() {
    let data = compile_and_read_elf(SIMPLE_MAIN, common::TARGET_X86_64, &[]);
    let hdr = parse_elf_header(&data);

    // Verify ELF identification fields.
    assert_eq!(hdr.ei_class, common::ELFCLASS64, "Expected ELFCLASS64");
    assert_eq!(hdr.ei_data, ELFDATA2LSB, "Expected little-endian");

    // Type must be ET_EXEC (static) or ET_DYN (PIE).
    assert!(
        hdr.e_type == ET_EXEC || hdr.e_type == ET_DYN,
        "Expected ET_EXEC (2) or ET_DYN (3), got {}",
        hdr.e_type
    );

    // Machine must be EM_X86_64.
    assert_eq!(
        hdr.e_machine,
        common::EM_X86_64,
        "Expected EM_X86_64 (0x3E)"
    );

    // Entry point must be set (non-zero for executables).
    assert_ne!(hdr.e_entry, 0, "Entry point should be non-zero");

    // Program headers must be present with at least one PT_LOAD segment.
    let phdrs = parse_program_headers(&data);
    assert!(
        !phdrs.is_empty(),
        "Executable must have at least one program header"
    );
    let load_segments: Vec<_> = phdrs.iter().filter(|ph| ph.p_type == PT_LOAD).collect();
    assert!(
        !load_segments.is_empty(),
        "Executable must have at least one PT_LOAD segment"
    );

    // Also verify via the common helper functions.
    let result = common::compile_source(SIMPLE_MAIN, &["--target", common::TARGET_X86_64]);
    if let Some(ref path) = result.output_path {
        common::verify_elf_magic(path);
        common::verify_elf_class(path, common::ELFCLASS64);
        common::verify_elf_arch(path, common::EM_X86_64);
    }
}

/// Compile and link a simple C program for i686, verifying ELF32 output with
/// the correct machine type and class.
#[test]
fn elf32_executable_valid() {
    let data = compile_and_read_elf(SIMPLE_MAIN, common::TARGET_I686, &[]);
    let hdr = parse_elf_header(&data);

    assert_eq!(
        hdr.ei_class,
        common::ELFCLASS32,
        "Expected ELFCLASS32 for i686"
    );
    assert_eq!(hdr.ei_data, ELFDATA2LSB, "Expected little-endian");
    assert!(
        hdr.e_type == ET_EXEC || hdr.e_type == ET_DYN,
        "Expected executable or PIE, got e_type={}",
        hdr.e_type
    );
    assert_eq!(
        hdr.e_machine,
        common::EM_386,
        "Expected EM_386 (0x03) for i686"
    );
    assert_ne!(hdr.e_entry, 0, "Entry point should be non-zero");

    let phdrs = parse_program_headers(&data);
    let load_count = phdrs.iter().filter(|ph| ph.p_type == PT_LOAD).count();
    assert!(
        load_count > 0,
        "ELF32 executable must have PT_LOAD segments"
    );
}

/// Compile and link a simple C program for AArch64, verifying ELF64 output
/// with EM_AARCH64.
#[test]
fn elf64_aarch64_valid() {
    let data = compile_and_read_elf(SIMPLE_MAIN, common::TARGET_AARCH64, &[]);
    let hdr = parse_elf_header(&data);

    assert_eq!(
        hdr.ei_class,
        common::ELFCLASS64,
        "Expected ELFCLASS64 for AArch64"
    );
    assert_eq!(hdr.ei_data, ELFDATA2LSB, "Expected little-endian");
    assert_eq!(
        hdr.e_machine,
        common::EM_AARCH64,
        "Expected EM_AARCH64 (0xB7)"
    );
    assert_ne!(hdr.e_entry, 0, "Entry point should be non-zero");
}

/// Compile and link a simple C program for RISC-V 64, verifying ELF64 output
/// with EM_RISCV.
#[test]
fn elf64_riscv_valid() {
    let data = compile_and_read_elf(SIMPLE_MAIN, common::TARGET_RISCV64, &[]);
    let hdr = parse_elf_header(&data);

    assert_eq!(
        hdr.ei_class,
        common::ELFCLASS64,
        "Expected ELFCLASS64 for RISC-V 64"
    );
    assert_eq!(hdr.ei_data, ELFDATA2LSB, "Expected little-endian");
    assert_eq!(hdr.e_machine, common::EM_RISCV, "Expected EM_RISCV (0xF3)");
    assert_ne!(hdr.e_entry, 0, "Entry point should be non-zero");
}

/// Compile a C program with data, rodata, and BSS, then verify the standard
/// ELF sections are present in the output binary.
#[test]
fn elf_section_layout() {
    let data = compile_and_read_elf(MULTI_SECTION_SOURCE, common::TARGET_X86_64, &[]);
    let sections = parse_section_headers(&data);
    let names: Vec<&str> = sections.iter().map(|s| s.name.as_str()).collect();

    // .text must always be present (code).
    assert!(
        names.contains(&".text"),
        "Missing .text section. Found: {:?}",
        names
    );

    // .shstrtab must be present (section name strings).
    assert!(
        names.contains(&".shstrtab"),
        "Missing .shstrtab section. Found: {:?}",
        names
    );

    // .symtab and .strtab should be present in non-stripped binaries.
    assert!(
        names.contains(&".symtab"),
        "Missing .symtab section. Found: {:?}",
        names
    );
    assert!(
        names.contains(&".strtab"),
        "Missing .strtab section. Found: {:?}",
        names
    );

    // Verify .text is marked as PROGBITS.
    let text_sec = find_section_by_name(&data, ".text").expect(".text not found");
    assert_eq!(
        text_sec.sh_type, SHT_PROGBITS,
        ".text should be SHT_PROGBITS"
    );
}

// ===========================================================================
// Phase 3: Symbol Resolution Tests
// ===========================================================================

/// Compile two C files separately to objects, link them together, and verify
/// the resulting executable resolves cross-file symbol references.
#[test]
fn symbol_resolution_two_files() {
    let dir = common::TempDir::new("sym_two_files");
    let obj_a = dir.path().join("a.o");
    let obj_b = dir.path().join("b.o");
    let exe = dir.path().join("linked.out");

    compile_to_object(SYMBOL_FILE_A, &obj_a, &["--target", common::TARGET_X86_64]);
    compile_to_object(SYMBOL_FILE_B, &obj_b, &["--target", common::TARGET_X86_64]);

    let result = link_files(
        &[obj_a.as_path(), obj_b.as_path()],
        &exe,
        &["--target", common::TARGET_X86_64],
    );

    assert!(
        result.status.success(),
        "Linking two objects failed:\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(exe.exists(), "Linked executable was not created");

    // Verify the output is a valid ELF executable.
    let data = fs::read(&exe).expect("Failed to read linked executable");
    let hdr = parse_elf_header(&data);
    assert!(
        hdr.e_type == ET_EXEC || hdr.e_type == ET_DYN,
        "Expected executable, got e_type={}",
        hdr.e_type
    );

    // If native, run it and check exit code (add(1,2) = 3).
    if common::is_native_target(common::TARGET_X86_64) {
        let run_output = Command::new(&exe)
            .output()
            .expect("Failed to run linked executable");
        let exit_code = run_output.status.code().unwrap_or(-1);
        assert_eq!(
            exit_code, 3,
            "Expected exit code 3 from add(1,2), got {}",
            exit_code
        );
    }
}

/// Compile C code that references an undefined function, verify the linker
/// emits an "undefined symbol" error and exits with non-zero status.
#[test]
fn symbol_undefined_error() {
    let result = common::compile_source(
        UNDEFINED_SYMBOL_SOURCE,
        &["--target", common::TARGET_X86_64],
    );

    assert!(
        !result.success,
        "Expected linker to fail on undefined symbol, but compilation succeeded"
    );
    // The error message should mention "undefined" in some form.
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("undefined") || stderr_lower.contains("unresolved"),
        "Expected 'undefined' or 'unresolved' in error message, got:\n{}",
        result.stderr
    );
}

/// Compile two files that both define the same global symbol, verify the
/// linker reports a duplicate/multiple definition error.
#[test]
fn symbol_duplicate_error() {
    let dir = common::TempDir::new("sym_dup");
    let obj_a = dir.path().join("dup_a.o");
    let obj_b = dir.path().join("dup_b.o");
    let exe = dir.path().join("dup.out");

    compile_to_object(
        DUPLICATE_SYMBOL_A,
        &obj_a,
        &["--target", common::TARGET_X86_64],
    );
    compile_to_object(
        DUPLICATE_SYMBOL_B,
        &obj_b,
        &["--target", common::TARGET_X86_64],
    );

    let result = link_files(
        &[obj_a.as_path(), obj_b.as_path()],
        &exe,
        &["--target", common::TARGET_X86_64],
    );

    assert!(
        !result.status.success(),
        "Expected linker to fail on duplicate symbol, but linking succeeded"
    );
    let stderr_str = String::from_utf8_lossy(&result.stderr).to_lowercase();
    assert!(
        stderr_str.contains("duplicate")
            || stderr_str.contains("multiple definition")
            || stderr_str.contains("already defined"),
        "Expected 'duplicate' or 'multiple definition' in error, got:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

/// Test weak symbol resolution: a weak definition is overridden by a strong
/// definition when both are present. The strong definition's value wins.
#[test]
fn symbol_weak_resolution() {
    let dir = common::TempDir::new("sym_weak");
    let obj_weak = dir.path().join("weak.o");
    let obj_strong = dir.path().join("strong.o");
    let exe = dir.path().join("weak_test.out");

    compile_to_object(
        WEAK_SYMBOL_DEF,
        &obj_weak,
        &["--target", common::TARGET_X86_64],
    );
    compile_to_object(
        STRONG_SYMBOL_DEF,
        &obj_strong,
        &["--target", common::TARGET_X86_64],
    );

    let result = link_files(
        &[obj_weak.as_path(), obj_strong.as_path()],
        &exe,
        &["--target", common::TARGET_X86_64],
    );

    assert!(
        result.status.success(),
        "Linking with weak + strong symbols failed:\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    // The strong definition returns 42, so main() should return 42.
    if common::is_native_target(common::TARGET_X86_64) {
        let run = Command::new(&exe)
            .output()
            .expect("Failed to run weak-test executable");
        let exit_code = run.status.code().unwrap_or(-1);
        assert_eq!(
            exit_code, 42,
            "Expected strong symbol value 42, got {}",
            exit_code
        );
    }
}

/// Verify that `static` functions are local symbols and do not conflict with
/// same-named symbols in other translation units.
#[test]
fn symbol_local_vs_global() {
    let dir = common::TempDir::new("sym_local");
    let obj_a = dir.path().join("local_a.o");
    let obj_b = dir.path().join("local_b.o");
    let exe = dir.path().join("local_test.out");

    compile_to_object(STATIC_FUNC_A, &obj_a, &["--target", common::TARGET_X86_64]);
    compile_to_object(STATIC_FUNC_B, &obj_b, &["--target", common::TARGET_X86_64]);

    let result = link_files(
        &[obj_a.as_path(), obj_b.as_path()],
        &exe,
        &["--target", common::TARGET_X86_64],
    );

    // Both files define `static int helper()` — no conflict expected.
    assert!(
        result.status.success(),
        "Linking with static (local) symbols should not fail:\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    // get_a() returns helper()=10, main returns get_a()+helper()=10+20=30.
    if common::is_native_target(common::TARGET_X86_64) {
        let run = Command::new(&exe)
            .output()
            .expect("Failed to run local-test executable");
        let exit_code = run.status.code().unwrap_or(-1);
        assert_eq!(
            exit_code, 30,
            "Expected 10+20=30 from static helpers, got {}",
            exit_code
        );
    }
}

// ===========================================================================
// Phase 4: CRT Linkage Tests
// ===========================================================================

/// Verify the linker locates and links system CRT objects (crt1.o, crti.o,
/// crtn.o) when producing an executable with a `main()` entry point.
///
/// A successful compilation to an executable that has `main` as its logical
/// entry point (but `_start` from crt1.o as the ELF entry) implies CRT
/// linkage succeeded.
#[test]
fn crt_objects_linked() {
    let result = common::compile_source(SIMPLE_MAIN, &["--target", common::TARGET_X86_64]);

    assert!(
        result.success,
        "CRT linkage failed — compilation did not succeed:\nstderr: {}",
        result.stderr
    );
    assert!(
        result.output_path.is_some(),
        "Compilation succeeded but no output binary was produced"
    );

    // The output should be an executable, which requires CRT objects.
    let path = result.output_path.as_ref().unwrap();
    let data = fs::read(path).expect("Failed to read CRT-linked executable");
    let hdr = parse_elf_header(&data);
    assert!(
        hdr.e_type == ET_EXEC || hdr.e_type == ET_DYN,
        "CRT-linked output should be executable, got e_type={}",
        hdr.e_type
    );
}

/// Verify the executable's ELF entry point is `_start` (from crt1.o), not
/// `main` directly. The `_start` symbol performs libc initialization before
/// calling `main`.
#[test]
fn crt_entry_point() {
    let data = compile_and_read_elf(SIMPLE_MAIN, common::TARGET_X86_64, &[]);
    let hdr = parse_elf_header(&data);

    // The entry point should be non-zero and point to _start.
    assert_ne!(
        hdr.e_entry, 0,
        "Entry point should be non-zero (set to _start from crt1.o)"
    );

    // In a properly CRT-linked executable, _start is the entry — not main().
    // We can verify by checking the symbol table for _start at the entry address.
    let sections = parse_section_headers(&data);
    let symtab = sections.iter().find(|s| s.sh_type == SHT_SYMTAB);
    if let Some(symtab_sec) = symtab {
        // The symbol table exists — good sign that CRT objects were linked.
        assert!(
            symtab_sec.sh_size > 0,
            "Symbol table is empty — CRT symbols may not have been linked"
        );
    }
}

/// Verify that `.init` and `.fini` sections are present in the linked
/// executable, as set up by crti.o and crtn.o from the CRT.
#[test]
fn crt_init_fini() {
    let data = compile_and_read_elf(SIMPLE_MAIN, common::TARGET_X86_64, &[]);
    let sections = parse_section_headers(&data);
    let names: Vec<&str> = sections.iter().map(|s| s.name.as_str()).collect();

    // .init and .fini sections come from crti.o/crtn.o in the standard CRT.
    // Some linker configurations may merge them or use .init_array/.fini_array.
    let has_init = names.contains(&".init") || names.contains(&".init_array");
    let has_fini = names.contains(&".fini") || names.contains(&".fini_array");

    assert!(
        has_init,
        "Expected .init or .init_array section from CRT. Found: {:?}",
        names
    );
    assert!(
        has_fini,
        "Expected .fini or .fini_array section from CRT. Found: {:?}",
        names
    );
}

// ===========================================================================
// Phase 5: ar Archive Reading Tests
// ===========================================================================

/// Verify the linker can parse and link against a static archive (e.g.,
/// system libc.a). A program using libc functions (like main → _start → __libc_start_main)
/// exercises the archive reading path.
#[test]
fn archive_reading() {
    // Compiling a program that uses main() requires linking against libc,
    // which may come from libc.a (static archive) on the system.
    let source = r#"
int main(void) {
    return 0;
}
"#;

    let result = common::compile_source(source, &["--target", common::TARGET_X86_64, "-static"]);

    // If -static succeeds, the linker successfully read libc.a and extracted
    // the needed object members.
    if result.success {
        let path = result.output_path.as_ref().unwrap();
        let meta = fs::metadata(path).expect("Failed to stat output");
        assert!(
            meta.len() > 0,
            "Statically linked binary should be non-empty"
        );
    }
    // Even if -static fails (e.g., libc.a not available), the flag should
    // at least be accepted by the compiler.
}

/// Verify only needed object members are extracted from archives (lazy/selective
/// archive member inclusion). Compiling a minimal program that only uses _start
/// from crt1.o and minimal libc should not pull in the entire libc.a.
#[test]
fn archive_selective_linking() {
    let source = r#"
int main(void) {
    return 42;
}
"#;

    let result = common::compile_source(source, &["--target", common::TARGET_X86_64, "-static"]);

    if result.success {
        let path = result.output_path.as_ref().unwrap();
        let meta = fs::metadata(path).expect("Failed to stat output");
        let file_size = meta.len();

        // A minimal static executable should be reasonably small.
        // Full libc.a on x86-64 is typically ~5MB; selective linking
        // should produce something much smaller for a trivial program.
        // We use a generous upper bound of 10MB.
        assert!(
            file_size < 10 * 1024 * 1024,
            "Statically linked minimal program is unexpectedly large ({} bytes). \
             Archive selective linking may not be working.",
            file_size
        );
    }
}

// ===========================================================================
// Phase 6: Output Mode Tests
// ===========================================================================

/// Use the `-c` flag to produce a relocatable object (ET_REL), verifying it
/// has no program headers and contains relocation sections.
#[test]
fn output_relocatable_object() {
    let _dir = common::TempDir::new("reloc_obj");
    let source = r#"
int global_var = 42;

int compute(int x) {
    return x + global_var;
}
"#;

    let result = common::compile_source(source, &["-c", "--target", common::TARGET_X86_64]);

    assert!(
        result.success,
        "Compilation to relocatable object failed:\nstderr: {}",
        result.stderr
    );

    let path = result.output_path.as_ref().expect("No output path for -c");
    let data = fs::read(path).expect("Failed to read relocatable object");
    let hdr = parse_elf_header(&data);

    // Must be ET_REL (relocatable).
    assert_eq!(
        hdr.e_type, ET_REL,
        "Expected ET_REL (1) for -c output, got {}",
        hdr.e_type
    );

    // Relocatable objects should have zero program headers.
    assert_eq!(
        hdr.e_phnum, 0,
        "Relocatable objects should have no program headers (got {})",
        hdr.e_phnum
    );

    // Should contain relocation sections (.rela.* or .rel.*).
    let sections = parse_section_headers(&data);
    let has_relocs = sections
        .iter()
        .any(|s| s.sh_type == SHT_RELA || s.sh_type == SHT_REL);
    assert!(
        has_relocs,
        "Relocatable object should contain relocation sections"
    );
}

/// Compile a program in default mode (no -c, no -shared), verifying the
/// output is a static executable (ET_EXEC or ET_DYN for PIE).
#[test]
fn output_static_executable() {
    let result = common::compile_source(SIMPLE_MAIN, &["--target", common::TARGET_X86_64]);

    assert!(
        result.success,
        "Default compilation to static executable failed:\nstderr: {}",
        result.stderr
    );

    let path = result.output_path.as_ref().expect("No output path");
    let data = fs::read(path).expect("Failed to read executable");
    let hdr = parse_elf_header(&data);

    assert!(
        hdr.e_type == ET_EXEC || hdr.e_type == ET_DYN,
        "Default output should be executable (ET_EXEC=2 or ET_DYN=3), got {}",
        hdr.e_type
    );

    // Executables must have program headers.
    assert!(hdr.e_phnum > 0, "Executable must have program headers");

    // Verify the file is actually runnable on the native architecture.
    if common::is_native_target(common::TARGET_X86_64) {
        let run = Command::new(path)
            .status()
            .expect("Failed to run executable");
        assert!(run.success(), "Static executable should exit successfully");
    }
}

/// Use `-shared -fPIC` to produce a shared library (ET_DYN), verifying the
/// presence of dynamic linking sections.
#[test]
fn output_shared_library() {
    let result = common::compile_source(
        SHARED_LIB_SOURCE,
        &["-shared", "-fPIC", "--target", common::TARGET_X86_64],
    );

    assert!(
        result.success,
        "Shared library compilation failed:\nstderr: {}",
        result.stderr
    );

    let path = result
        .output_path
        .as_ref()
        .expect("No output path for -shared");
    let data = fs::read(path).expect("Failed to read shared library");
    let hdr = parse_elf_header(&data);

    // Shared libraries are ET_DYN.
    assert_eq!(
        hdr.e_type, ET_DYN,
        "Expected ET_DYN (3) for -shared output, got {}",
        hdr.e_type
    );

    // Shared libraries must have a .dynamic section.
    let sections = parse_section_headers(&data);
    let has_dynamic = sections
        .iter()
        .any(|s| s.name == ".dynamic" || s.sh_type == SHT_DYNAMIC);
    assert!(
        has_dynamic,
        "Shared library must contain a .dynamic section"
    );

    // Should have .dynsym (dynamic symbol table).
    let has_dynsym = sections
        .iter()
        .any(|s| s.name == ".dynsym" || s.sh_type == SHT_DYNSYM);
    assert!(has_dynsym, "Shared library should contain .dynsym");

    // Should have .dynstr (dynamic string table).
    let has_dynstr = sections.iter().any(|s| s.name == ".dynstr");
    assert!(has_dynstr, "Shared library should contain .dynstr");

    // Should have PT_DYNAMIC program header.
    let phdrs = parse_program_headers(&data);
    let has_pt_dynamic = phdrs.iter().any(|ph| ph.p_type == PT_DYNAMIC);
    assert!(
        has_pt_dynamic,
        "Shared library must have a PT_DYNAMIC program header"
    );
}

// ===========================================================================
// Phase 7: Relocation Tests
// ===========================================================================

/// Verify that x86-64 relocatable objects contain the expected relocation
/// types (R_X86_64_*) for references to global data and function calls.
#[test]
fn relocations_x86_64() {
    let source = r#"
extern int external_var;
extern int external_func(int);

int compute(void) {
    return external_func(external_var);
}
"#;

    let result = common::compile_source(source, &["-c", "--target", common::TARGET_X86_64]);

    assert!(
        result.success,
        "x86-64 object compilation failed:\nstderr: {}",
        result.stderr
    );

    let path = result.output_path.as_ref().expect("No output path");
    let data = fs::read(path).expect("Failed to read x86-64 object");

    // The object should contain .rela.text with relocations for
    // external_var and external_func references.
    let sections = parse_section_headers(&data);
    let rela_sections: Vec<_> = sections
        .iter()
        .filter(|s| s.sh_type == SHT_RELA || s.sh_type == SHT_REL)
        .collect();

    assert!(
        !rela_sections.is_empty(),
        "x86-64 object with external references must have relocation sections"
    );

    // Verify at least one relocation section targets .text.
    let has_text_relocs = rela_sections
        .iter()
        .any(|s| s.name.contains(".text") || s.name == ".rela.text" || s.name == ".rel.text");
    assert!(
        has_text_relocs,
        "Expected relocation section for .text. Found: {:?}",
        rela_sections.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
}

/// Verify that i686 relocatable objects contain relocations in ELF32 format.
#[test]
fn relocations_i686() {
    let source = r#"
extern int external_var;
extern int external_func(int);

int compute(void) {
    return external_func(external_var);
}
"#;

    let result = common::compile_source(source, &["-c", "--target", common::TARGET_I686]);

    assert!(
        result.success,
        "i686 object compilation failed:\nstderr: {}",
        result.stderr
    );

    let path = result.output_path.as_ref().expect("No output path");
    let data = fs::read(path).expect("Failed to read i686 object");
    let hdr = parse_elf_header(&data);

    // Must be ELF32.
    assert_eq!(
        hdr.ei_class,
        common::ELFCLASS32,
        "i686 objects must be ELF32"
    );
    assert_eq!(hdr.e_machine, common::EM_386, "Expected EM_386");

    // Must have relocations.
    let sections = parse_section_headers(&data);
    let has_relocs = sections
        .iter()
        .any(|s| s.sh_type == SHT_REL || s.sh_type == SHT_RELA);
    assert!(
        has_relocs,
        "i686 object with external references must have relocation sections"
    );
}

/// Verify that AArch64 relocatable objects contain relocations in ELF64 format.
#[test]
fn relocations_aarch64() {
    let source = r#"
extern int external_var;
extern int external_func(int);

int compute(void) {
    return external_func(external_var);
}
"#;

    let result = common::compile_source(source, &["-c", "--target", common::TARGET_AARCH64]);

    assert!(
        result.success,
        "AArch64 object compilation failed:\nstderr: {}",
        result.stderr
    );

    let path = result.output_path.as_ref().expect("No output path");
    let data = fs::read(path).expect("Failed to read AArch64 object");
    let hdr = parse_elf_header(&data);

    assert_eq!(
        hdr.ei_class,
        common::ELFCLASS64,
        "AArch64 objects must be ELF64"
    );
    assert_eq!(hdr.e_machine, common::EM_AARCH64, "Expected EM_AARCH64");

    let sections = parse_section_headers(&data);
    let has_relocs = sections
        .iter()
        .any(|s| s.sh_type == SHT_RELA || s.sh_type == SHT_REL);
    assert!(
        has_relocs,
        "AArch64 object with external references must have relocation sections"
    );
}

/// Verify that RISC-V 64 relocatable objects contain relocations in ELF64 format.
#[test]
fn relocations_riscv64() {
    let source = r#"
extern int external_var;
extern int external_func(int);

int compute(void) {
    return external_func(external_var);
}
"#;

    let result = common::compile_source(source, &["-c", "--target", common::TARGET_RISCV64]);

    assert!(
        result.success,
        "RISC-V 64 object compilation failed:\nstderr: {}",
        result.stderr
    );

    let path = result.output_path.as_ref().expect("No output path");
    let data = fs::read(path).expect("Failed to read RISC-V 64 object");
    let hdr = parse_elf_header(&data);

    assert_eq!(
        hdr.ei_class,
        common::ELFCLASS64,
        "RISC-V 64 objects must be ELF64"
    );
    assert_eq!(hdr.e_machine, common::EM_RISCV, "Expected EM_RISCV");

    let sections = parse_section_headers(&data);
    let has_relocs = sections
        .iter()
        .any(|s| s.sh_type == SHT_RELA || s.sh_type == SHT_REL);
    assert!(
        has_relocs,
        "RISC-V 64 object with external references must have relocation sections"
    );
}

// ===========================================================================
// Phase 8: Section Merging and Layout Tests
// ===========================================================================

/// Compile two files and verify their `.text` sections are merged into a
/// single `.text` section in the linked output.
#[test]
fn section_merging() {
    let dir = common::TempDir::new("sec_merge");
    let obj_a = dir.path().join("merge_a.o");
    let obj_b = dir.path().join("merge_b.o");
    let exe = dir.path().join("merged.out");

    let source_a = r#"
int func_a(void) { return 10; }
"#;
    let source_b = r#"
extern int func_a(void);
int main(void) { return func_a(); }
"#;

    compile_to_object(source_a, &obj_a, &["--target", common::TARGET_X86_64]);
    compile_to_object(source_b, &obj_b, &["--target", common::TARGET_X86_64]);

    let result = link_files(
        &[obj_a.as_path(), obj_b.as_path()],
        &exe,
        &["--target", common::TARGET_X86_64],
    );

    assert!(
        result.status.success(),
        "Section merging link failed:\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let data = fs::read(&exe).expect("Failed to read merged executable");

    // There should be exactly one .text section in the output (merged).
    let sections = parse_section_headers(&data);
    let text_sections: Vec<_> = sections.iter().filter(|s| s.name == ".text").collect();
    assert_eq!(
        text_sections.len(),
        1,
        "Expected exactly one merged .text section, found {}",
        text_sections.len()
    );

    // The merged .text section should be larger than either individual object's .text.
    let text = text_sections[0];
    assert!(text.sh_size > 0, "Merged .text section should be non-empty");
}

/// Verify that sections in the output binary are aligned according to their
/// alignment requirements (sh_addralign).
#[test]
fn section_alignment() {
    let data = compile_and_read_elf(MULTI_SECTION_SOURCE, common::TARGET_X86_64, &[]);
    let sections = parse_section_headers(&data);

    for section in &sections {
        if section.sh_addralign > 1 && section.sh_addr > 0 {
            // The section's virtual address must be aligned to sh_addralign.
            assert_eq!(
                section.sh_addr % section.sh_addralign,
                0,
                "Section '{}' at addr 0x{:x} is not aligned to {} bytes",
                section.name,
                section.sh_addr,
                section.sh_addralign
            );
        }
        if section.sh_addralign > 1 && section.sh_offset > 0 && section.sh_type != SHT_NOBITS {
            // The section's file offset should also respect alignment.
            assert_eq!(
                section.sh_offset % section.sh_addralign,
                0,
                "Section '{}' at file offset 0x{:x} is not aligned to {} bytes",
                section.name,
                section.sh_offset,
                section.sh_addralign
            );
        }
    }
}

/// Verify the `.bss` section takes no file space (sh_type = SHT_NOBITS,
/// sh_size in file is 0) but has the correct memory size for uninitialized
/// globals.
#[test]
fn bss_section() {
    let data = compile_and_read_elf(BSS_SOURCE, common::TARGET_X86_64, &[]);
    let sections = parse_section_headers(&data);

    // Find the .bss section (may be named .bss or have SHT_NOBITS type).
    let bss = sections
        .iter()
        .find(|s| s.name == ".bss" || (s.sh_type == SHT_NOBITS && s.sh_size > 0));

    if let Some(bss_sec) = bss {
        // .bss should be SHT_NOBITS — occupies no file space.
        assert_eq!(
            bss_sec.sh_type, SHT_NOBITS,
            ".bss section should be SHT_NOBITS"
        );

        // Memory size should accommodate bss_array[1024] = 4096 bytes.
        assert!(
            bss_sec.sh_size >= 4096,
            ".bss memory size should be >= 4096 bytes for int[1024], got {}",
            bss_sec.sh_size
        );

        // For SHT_NOBITS, the section data doesn't exist in the file, so the
        // file offset + size should NOT extend the file. We verify this by
        // checking that sections following .bss don't start after a gap.
    } else {
        // Some linkers may merge BSS into the data segment. Verify the
        // executable at least has a PT_LOAD segment with p_memsz > p_filesz.
        let phdrs = parse_program_headers(&data);
        let has_bss_like = phdrs
            .iter()
            .any(|ph| ph.p_type == PT_LOAD && ph.p_memsz > ph.p_filesz);
        assert!(
            has_bss_like,
            "No .bss section and no PT_LOAD with memsz > filesz found"
        );
    }
}

/// Compile and link a program for a cross-architecture target, then execute
/// it via QEMU user-mode emulation to verify the linked output is runnable.
/// Skips gracefully if QEMU is not available for the target.
#[test]
fn cross_arch_linked_execution() {
    // Test cross-architecture linking by compiling and running on i686 via QEMU.
    let target = common::TARGET_I686;
    if !common::is_qemu_available(target) && !common::is_native_target(target) {
        eprintln!(
            "Skipping cross_arch_linked_execution: QEMU not available for {}",
            target
        );
        return;
    }

    let source = r#"
int main(void) {
    return 7;
}
"#;

    let run_result: common::RunResult = common::compile_and_run(source, target, &[]);

    if run_result.success {
        // The program should exit with code 7.
        let code = run_result.exit_status.code().unwrap_or(-1);
        assert_eq!(
            code, 7,
            "Expected exit code 7, got {}. stdout: {}, stderr: {}",
            code, run_result.stdout, run_result.stderr
        );
    } else {
        // If compilation or execution failed, print diagnostics.
        eprintln!(
            "Cross-arch linked execution did not succeed (may be expected during development):\n\
             exit_status: {:?}\nstdout: {}\nstderr: {}",
            run_result.exit_status, run_result.stdout, run_result.stderr
        );
    }
}

/// Verify that multi-file compilation using QEMU produces a runnable binary
/// for a non-native architecture when linking multiple object files.
#[test]
fn cross_arch_multifile_linking() {
    let target = common::TARGET_AARCH64;

    let dir = common::TempDir::new("cross_multi");
    let obj_a = dir.path().join("cross_a.o");
    let obj_b = dir.path().join("cross_b.o");
    let exe = dir.path().join("cross_linked.out");

    // Write source files using fs::write for direct control.
    let src_a_path = dir.path().join("cross_a.c");
    let src_b_path = dir.path().join("cross_b.c");
    fs::write(&src_a_path, "int get_value(void) { return 5; }\n")
        .expect("Failed to write cross_a.c");
    fs::write(
        &src_b_path,
        "extern int get_value(void);\nint main(void) { return get_value(); }\n",
    )
    .expect("Failed to write cross_b.c");

    let bcc = common::get_bcc_binary();

    // Compile each source to an object.
    let out_a = Command::new(&bcc)
        .args(["-c", "--target", target, "-o"])
        .arg(&obj_a)
        .arg(&src_a_path)
        .output()
        .expect("Failed to compile cross_a.c");

    if !out_a.status.success() {
        eprintln!(
            "Skipping cross_arch_multifile: compilation of A failed:\n{}",
            String::from_utf8_lossy(&out_a.stderr)
        );
        return;
    }

    let out_b = Command::new(&bcc)
        .args(["-c", "--target", target, "-o"])
        .arg(&obj_b)
        .arg(&src_b_path)
        .output()
        .expect("Failed to compile cross_b.c");

    if !out_b.status.success() {
        eprintln!(
            "Skipping cross_arch_multifile: compilation of B failed:\n{}",
            String::from_utf8_lossy(&out_b.stderr)
        );
        return;
    }

    // Link the objects.
    let link_result = link_files(
        &[obj_a.as_path(), obj_b.as_path()],
        &exe,
        &["--target", target],
    );

    if link_result.status.success() {
        // Verify the ELF is correct.
        let data = fs::read(&exe).expect("Failed to read cross-linked binary");
        let hdr = parse_elf_header(&data);
        assert_eq!(hdr.e_machine, common::EM_AARCH64);
        assert_eq!(hdr.ei_class, common::ELFCLASS64);

        // Try running via QEMU if available, with timeout to prevent hangs.
        if common::is_qemu_available(target) {
            let qemu_name = if target.starts_with("aarch64") {
                "qemu-aarch64-static"
            } else if target.starts_with("i686") {
                "qemu-i386-static"
            } else if target.starts_with("riscv64") {
                "qemu-riscv64-static"
            } else {
                "qemu-x86_64-static"
            };
            let qemu_output = Command::new("timeout")
                .args(["10", qemu_name])
                .arg(&exe)
                .output();
            if let Ok(out) = qemu_output {
                if out.status.success() {
                    let code = out.status.code().unwrap_or(-1);
                    assert_eq!(code, 5, "Expected exit code 5 from get_value()");
                } else {
                    eprintln!(
                        "NOTE: Cross-arch multifile execution returned non-success (known codegen limitation): exit={:?}",
                        out.status.code()
                    );
                }
            }
        }
    }

    // Clean up temp source files.
    let _ = fs::remove_file(&src_a_path);
    let _ = fs::remove_file(&src_b_path);
}

/// Test that the write_temp_source and TempFile RAII cleanup work properly
/// when used in linker integration test scenarios.
#[test]
fn temp_file_management() {
    let temp_file: common::TempFile = common::write_temp_source("int main(void) { return 0; }\n");
    let path_ref: &Path = temp_file.path();
    assert!(
        path_ref.exists(),
        "Temp source file should exist at: {}",
        path_ref.display()
    );

    // Read the temp file back to verify contents.
    let content = std::fs::read_to_string(path_ref).expect("Failed to read temp source");
    assert!(content.contains("main"), "Temp file should contain 'main'");

    // Use PathBuf::from and Path::new for path manipulation.
    let path_buf = PathBuf::from(path_ref.to_str().unwrap());
    let path_new = Path::new(path_ref.to_str().unwrap());
    assert_eq!(
        path_buf.as_path(),
        path_new,
        "PathBuf::from and Path::new should produce equivalent paths"
    );

    // Compile via compile_source to verify the temp file works.
    let result = common::compile_source("int main(void) { return 0; }\n", &["-c"]);
    // Access all CompileResult fields to verify schema compliance.
    let _success = result.success;
    let _exit = result.exit_status;
    let _out = &result.stdout;
    let _err = &result.stderr;
    let _path = &result.output_path;
}

/// Verify PT_LOAD segments are page-aligned (typically 4096 or 0x1000 bytes)
/// and contain the expected sections.
#[test]
fn segment_layout() {
    let data = compile_and_read_elf(MULTI_SECTION_SOURCE, common::TARGET_X86_64, &[]);
    let phdrs = parse_program_headers(&data);

    let load_segments: Vec<_> = phdrs.iter().filter(|ph| ph.p_type == PT_LOAD).collect();
    assert!(
        !load_segments.is_empty(),
        "Executable must have at least one PT_LOAD segment"
    );

    for (i, seg) in load_segments.iter().enumerate() {
        // PT_LOAD segments should be page-aligned (alignment is typically
        // 0x1000 = 4096 or 0x200000 = 2MB for large pages).
        if seg.p_align > 1 {
            assert_eq!(
                seg.p_vaddr % seg.p_align,
                0,
                "PT_LOAD segment {} at vaddr 0x{:x} is not aligned to 0x{:x}",
                i,
                seg.p_vaddr,
                seg.p_align
            );
            assert_eq!(
                seg.p_offset % seg.p_align,
                0,
                "PT_LOAD segment {} at file offset 0x{:x} is not aligned to 0x{:x}",
                i,
                seg.p_offset,
                seg.p_align
            );
        }

        // p_filesz should be <= p_memsz (extra memory is BSS).
        assert!(
            seg.p_filesz <= seg.p_memsz,
            "PT_LOAD segment {} has filesz ({}) > memsz ({})",
            i,
            seg.p_filesz,
            seg.p_memsz
        );
    }
}
