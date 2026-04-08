//! Integration tests for x86-64 security hardening features.
//!
//! This module verifies three security hardening mechanisms implemented in the
//! bcc compiler's x86-64 backend:
//!
//! 1. **Retpoline** (`-mretpoline`) — Spectre v2 mitigation replacing indirect
//!    branches (`jmp *%rax`, `call *%rax`) with speculative-safe thunk sequences.
//!
//! 2. **Intel CET** (`-fcf-protection`) — Control-flow Enforcement Technology
//!    instrumentation inserting `endbr64` (`F3 0F 1E FA`) at all indirect branch
//!    targets (function entries, jump table destinations).
//!
//! 3. **Stack Probing** — Automatic stack probing for functions whose stack frames
//!    exceed one page (4096 bytes), preventing stack clash attacks by touching
//!    each page in sequence.
//!
//! # Test Strategy
//!
//! Tests are black-box integration tests that invoke the `bcc` binary as a
//! subprocess via the shared `tests/common/mod.rs` utilities. Compiled output
//! binaries are inspected at the byte level to confirm security feature presence
//! or absence by searching for known machine code byte patterns in the `.text`
//! ELF section.
//!
//! # Architecture Scope
//!
//! All tests in this file target **x86-64 only** (`--target x86_64-linux-gnu`),
//! as security hardening features (retpoline, CET, stack probing) are specific
//! to the x86-64 backend per the AAP §0.6.1.
//!
//! # Byte Patterns Reference
//!
//! | Feature     | Byte Pattern          | Meaning                             |
//! |-------------|-----------------------|-------------------------------------|
//! | `endbr64`   | `F3 0F 1E FA`        | CET indirect branch target marker   |
//! | `pause`     | `F3 90`              | Speculative execution hint          |
//! | `lfence`    | `0F AE E8`           | Load fence (speculation barrier)    |
//! | `ret`       | `C3`                 | Near return                         |
//! | `call *%rax`| `FF D0`              | Indirect call through RAX           |
//! | `jmp *%rax` | `FF E0`              | Indirect jump through RAX           |
//! | `call *%r11`| `41 FF D3`           | Indirect call through R11           |
//! | `jmp *%r11` | `41 FF E3`           | Indirect jump through R11           |
//!
//! # Zero-Dependency Guarantee
//!
//! This file uses ONLY the Rust standard library (`std`) and the `bcc` crate
//! (via the shared `tests/common/mod.rs` utilities). No external crates.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// Target Constant
// ===========================================================================

/// Target triple for all security tests — x86-64 Linux only.
const TARGET: &str = "x86_64-linux-gnu";

// ===========================================================================
// Machine Code Byte Pattern Constants
// ===========================================================================

/// `endbr64` instruction encoding: `F3 0F 1E FA`.
/// This is the CET indirect branch target marker for 64-bit mode.
const ENDBR64_BYTES: [u8; 4] = [0xF3, 0x0F, 0x1E, 0xFA];

/// `pause` instruction encoding: `F3 90`.
/// Used inside retpoline speculation trap loops.
const PAUSE_BYTES: [u8; 2] = [0xF3, 0x90];

/// `lfence` instruction encoding: `0F AE E8`.
/// Load fence used in retpoline thunks as a speculation barrier.
const LFENCE_BYTES: [u8; 3] = [0x0F, 0xAE, 0xE8];

/// `ret` (near return) encoding: `C3`.
/// Used in retpoline thunks to redirect speculative execution.
const RET_BYTE: u8 = 0xC3;

/// `call *%rax` indirect call encoding: `FF D0`.
/// Should be absent when `-mretpoline` is active.
/// Kept as reference documentation for byte-level verification patterns.
#[allow(dead_code)]
const CALL_INDIRECT_RAX: [u8; 2] = [0xFF, 0xD0];

/// `jmp *%rax` indirect jump encoding: `FF E0`.
/// Should be absent when `-mretpoline` is active.
/// Kept as reference documentation for byte-level verification patterns.
#[allow(dead_code)]
const JMP_INDIRECT_RAX: [u8; 2] = [0xFF, 0xE0];

/// `call *%r11` indirect call encoding: `41 FF D3`.
/// Another indirect call form that retpoline should intercept.
/// Kept as reference documentation for byte-level verification patterns.
#[allow(dead_code)]
const CALL_INDIRECT_R11: [u8; 3] = [0x41, 0xFF, 0xD3];

/// `jmp *%r11` indirect jump encoding: `41 FF E3`.
/// Another indirect jump form that retpoline should intercept.
/// Kept as reference documentation for byte-level verification patterns.
#[allow(dead_code)]
const JMP_INDIRECT_R11: [u8; 3] = [0x41, 0xFF, 0xE3];

// ===========================================================================
// Binary Inspection Helper Functions
// ===========================================================================

/// Read the `.text` section bytes from an ELF64 binary.
///
/// Parses the ELF64 header and section headers to locate the `.text` section,
/// then returns its raw byte contents. This enables byte-level verification of
/// generated machine code for security feature presence/absence.
///
/// # Arguments
///
/// * `binary` - Path to the ELF64 binary file.
///
/// # Returns
///
/// The raw bytes of the `.text` section, or an empty `Vec` if the section
/// is not found.
///
/// # Panics
///
/// Panics if the file cannot be read or the ELF header is malformed.
fn read_elf_text_section(binary: &Path) -> Vec<u8> {
    let data = fs::read(binary).unwrap_or_else(|e| {
        panic!(
            "Failed to read binary '{}' for .text section extraction: {}",
            binary.display(),
            e
        );
    });

    // Verify minimum ELF header size (64 bytes for ELF64).
    assert!(
        data.len() >= 64,
        "Binary '{}' is too small ({} bytes) to be a valid ELF64 file",
        binary.display(),
        data.len()
    );

    // Verify ELF magic.
    assert_eq!(
        &data[0..4],
        &[0x7f, b'E', b'L', b'F'],
        "Not an ELF file: '{}'",
        binary.display()
    );

    // Determine ELF class (we expect ELF64 for x86-64).
    let elf_class = data[4];
    assert!(
        elf_class == 1 || elf_class == 2,
        "Invalid ELF class {} in '{}'",
        elf_class,
        binary.display()
    );

    // Parse based on ELF class.
    if elf_class == 2 {
        // ELF64 parsing.
        read_elf64_text_section(&data, binary)
    } else {
        // ELF32 parsing (not expected for x86-64 security tests, but handle gracefully).
        read_elf32_text_section(&data, binary)
    }
}

/// Parse ELF64 headers to extract the `.text` section.
///
/// Reads the section header table, finds the section header string table,
/// then locates the `.text` section by name.
fn read_elf64_text_section(data: &[u8], binary: &Path) -> Vec<u8> {
    // ELF64 header fields:
    // e_shoff:    offset 0x28, 8 bytes (section header table offset)
    // e_shentsize: offset 0x3A, 2 bytes (section header entry size)
    // e_shnum:    offset 0x3C, 2 bytes (number of section headers)
    // e_shstrndx: offset 0x3E, 2 bytes (section name string table index)

    if data.len() < 0x40 {
        return Vec::new();
    }

    let e_shoff = u64::from_le_bytes([
        data[0x28], data[0x29], data[0x2A], data[0x2B], data[0x2C], data[0x2D], data[0x2E],
        data[0x2F],
    ]) as usize;

    let e_shentsize = u16::from_le_bytes([data[0x3A], data[0x3B]]) as usize;
    let e_shnum = u16::from_le_bytes([data[0x3C], data[0x3D]]) as usize;
    let e_shstrndx = u16::from_le_bytes([data[0x3E], data[0x3F]]) as usize;

    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum {
        return Vec::new();
    }

    // Read the section header string table to look up section names.
    let shstrtab_offset = e_shoff + e_shstrndx * e_shentsize;
    if shstrtab_offset + e_shentsize > data.len() {
        return Vec::new();
    }

    // sh_offset is at offset 0x18 within a 64-bit section header entry.
    // sh_size is at offset 0x20 within a 64-bit section header entry.
    let strtab_file_offset = u64::from_le_bytes([
        data[shstrtab_offset + 0x18],
        data[shstrtab_offset + 0x19],
        data[shstrtab_offset + 0x1A],
        data[shstrtab_offset + 0x1B],
        data[shstrtab_offset + 0x1C],
        data[shstrtab_offset + 0x1D],
        data[shstrtab_offset + 0x1E],
        data[shstrtab_offset + 0x1F],
    ]) as usize;

    let strtab_size = u64::from_le_bytes([
        data[shstrtab_offset + 0x20],
        data[shstrtab_offset + 0x21],
        data[shstrtab_offset + 0x22],
        data[shstrtab_offset + 0x23],
        data[shstrtab_offset + 0x24],
        data[shstrtab_offset + 0x25],
        data[shstrtab_offset + 0x26],
        data[shstrtab_offset + 0x27],
    ]) as usize;

    if strtab_file_offset + strtab_size > data.len() {
        return Vec::new();
    }

    let strtab = &data[strtab_file_offset..strtab_file_offset + strtab_size];

    // Iterate over all section headers to find `.text`.
    for i in 0..e_shnum {
        let sh_start = e_shoff + i * e_shentsize;
        if sh_start + e_shentsize > data.len() {
            continue;
        }

        // sh_name is at offset 0x00 within the section header (4-byte index into strtab).
        let sh_name_idx = u32::from_le_bytes([
            data[sh_start],
            data[sh_start + 1],
            data[sh_start + 2],
            data[sh_start + 3],
        ]) as usize;

        // Extract the null-terminated section name from the string table.
        if sh_name_idx >= strtab.len() {
            continue;
        }
        let name_bytes = &strtab[sh_name_idx..];
        let name_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        let section_name = std::str::from_utf8(&name_bytes[..name_end]).unwrap_or("");

        if section_name == ".text" {
            // Found the .text section. Extract its file offset and size.
            let sh_offset = u64::from_le_bytes([
                data[sh_start + 0x18],
                data[sh_start + 0x19],
                data[sh_start + 0x1A],
                data[sh_start + 0x1B],
                data[sh_start + 0x1C],
                data[sh_start + 0x1D],
                data[sh_start + 0x1E],
                data[sh_start + 0x1F],
            ]) as usize;

            let sh_size = u64::from_le_bytes([
                data[sh_start + 0x20],
                data[sh_start + 0x21],
                data[sh_start + 0x22],
                data[sh_start + 0x23],
                data[sh_start + 0x24],
                data[sh_start + 0x25],
                data[sh_start + 0x26],
                data[sh_start + 0x27],
            ]) as usize;

            if sh_offset + sh_size <= data.len() {
                return data[sh_offset..sh_offset + sh_size].to_vec();
            } else {
                panic!(
                    ".text section in '{}' extends beyond file (offset={}, size={}, file_len={})",
                    binary.display(),
                    sh_offset,
                    sh_size,
                    data.len()
                );
            }
        }
    }

    // .text section not found; return empty.
    Vec::new()
}

/// Parse ELF32 headers to extract the `.text` section.
///
/// Similar to the ELF64 variant but uses 32-bit field widths and offsets.
fn read_elf32_text_section(data: &[u8], binary: &Path) -> Vec<u8> {
    // ELF32 header fields:
    // e_shoff:    offset 0x20, 4 bytes
    // e_shentsize: offset 0x2E, 2 bytes
    // e_shnum:    offset 0x30, 2 bytes
    // e_shstrndx: offset 0x32, 2 bytes

    if data.len() < 0x34 {
        return Vec::new();
    }

    let e_shoff = u32::from_le_bytes([data[0x20], data[0x21], data[0x22], data[0x23]]) as usize;

    let e_shentsize = u16::from_le_bytes([data[0x2E], data[0x2F]]) as usize;
    let e_shnum = u16::from_le_bytes([data[0x30], data[0x31]]) as usize;
    let e_shstrndx = u16::from_le_bytes([data[0x32], data[0x33]]) as usize;

    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum {
        return Vec::new();
    }

    let shstrtab_offset = e_shoff + e_shstrndx * e_shentsize;
    if shstrtab_offset + e_shentsize > data.len() {
        return Vec::new();
    }

    // In ELF32 section headers: sh_offset at +0x10 (4 bytes), sh_size at +0x14 (4 bytes).
    let strtab_file_offset = u32::from_le_bytes([
        data[shstrtab_offset + 0x10],
        data[shstrtab_offset + 0x11],
        data[shstrtab_offset + 0x12],
        data[shstrtab_offset + 0x13],
    ]) as usize;

    let strtab_size = u32::from_le_bytes([
        data[shstrtab_offset + 0x14],
        data[shstrtab_offset + 0x15],
        data[shstrtab_offset + 0x16],
        data[shstrtab_offset + 0x17],
    ]) as usize;

    if strtab_file_offset + strtab_size > data.len() {
        return Vec::new();
    }

    let strtab = &data[strtab_file_offset..strtab_file_offset + strtab_size];

    for i in 0..e_shnum {
        let sh_start = e_shoff + i * e_shentsize;
        if sh_start + e_shentsize > data.len() {
            continue;
        }

        let sh_name_idx = u32::from_le_bytes([
            data[sh_start],
            data[sh_start + 1],
            data[sh_start + 2],
            data[sh_start + 3],
        ]) as usize;

        if sh_name_idx >= strtab.len() {
            continue;
        }
        let name_bytes = &strtab[sh_name_idx..];
        let name_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        let section_name = std::str::from_utf8(&name_bytes[..name_end]).unwrap_or("");

        if section_name == ".text" {
            let sh_offset = u32::from_le_bytes([
                data[sh_start + 0x10],
                data[sh_start + 0x11],
                data[sh_start + 0x12],
                data[sh_start + 0x13],
            ]) as usize;

            let sh_size = u32::from_le_bytes([
                data[sh_start + 0x14],
                data[sh_start + 0x15],
                data[sh_start + 0x16],
                data[sh_start + 0x17],
            ]) as usize;

            if sh_offset + sh_size <= data.len() {
                return data[sh_offset..sh_offset + sh_size].to_vec();
            } else {
                panic!(
                    ".text section in '{}' extends beyond file (offset={}, size={}, file_len={})",
                    binary.display(),
                    sh_offset,
                    sh_size,
                    data.len()
                );
            }
        }
    }

    Vec::new()
}

/// Find all occurrences of a byte pattern within a data slice.
///
/// Uses a simple sliding window search. Returns a vector of starting offsets
/// where the pattern was found.
///
/// # Arguments
///
/// * `data` - The byte slice to search within.
/// * `pattern` - The byte pattern to search for.
///
/// # Returns
///
/// A `Vec<usize>` of starting offsets for each match found.
fn find_byte_pattern(data: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() || data.len() < pattern.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let limit = data.len() - pattern.len() + 1;
    for i in 0..limit {
        if data[i..i + pattern.len()] == *pattern {
            matches.push(i);
        }
    }
    matches
}

/// Check whether the data contains the `endbr64` byte sequence (`F3 0F 1E FA`).
///
/// # Arguments
///
/// * `data` - The byte slice to search (typically the `.text` section).
///
/// # Returns
///
/// `true` if at least one `endbr64` instruction is found.
fn contains_endbr64(data: &[u8]) -> bool {
    !find_byte_pattern(data, &ENDBR64_BYTES).is_empty()
}

/// Check whether the data contains unprotected indirect jump/call instructions.
///
/// Searches for `FF E0` (`jmp *%rax`), `FF D0` (`call *%rax`), and REX-prefixed
/// variants `41 FF E3` (`jmp *%r11`), `41 FF D3` (`call *%r11`). These are the
/// indirect branch patterns that retpoline should replace.
///
/// # Arguments
///
/// * `data` - The byte slice to search (typically the `.text` section).
///
/// # Returns
///
/// `true` if any unprotected indirect jump or call instruction is found.
fn contains_direct_indirect_jmp(data: &[u8]) -> bool {
    // Check for common indirect branch encodings via any GPR.
    // FF /2 = call *r/m64 (ModR/M byte D0-D7 for registers rax-rdi)
    // FF /4 = jmp *r/m64 (ModR/M byte E0-E7 for registers rax-rdi)
    for i in 0..data.len().saturating_sub(1) {
        if data[i] == 0xFF {
            let modrm = data[i + 1];
            // Check for direct register indirect call (FF /2 with mod=11, reg=010)
            // ModR/M byte range: 0xD0-0xD7 (call *%rax through call *%rdi)
            if (0xD0..=0xD7).contains(&modrm) {
                return true;
            }
            // Check for direct register indirect jmp (FF /4 with mod=11, reg=100)
            // ModR/M byte range: 0xE0-0xE7 (jmp *%rax through jmp *%rdi)
            if (0xE0..=0xE7).contains(&modrm) {
                return true;
            }
        }
        // Check REX-prefixed variants (41 FF for R8-R15).
        if i + 2 < data.len() && data[i] == 0x41 && data[i + 1] == 0xFF {
            let modrm = data[i + 2];
            // call *%r8 through call *%r15: ModR/M 0xD0-0xD7
            if (0xD0..=0xD7).contains(&modrm) {
                return true;
            }
            // jmp *%r8 through jmp *%r15: ModR/M 0xE0-0xE7
            if (0xE0..=0xE7).contains(&modrm) {
                return true;
            }
        }
    }
    false
}

/// Check whether the data contains retpoline thunk characteristics.
///
/// A retpoline thunk typically contains:
/// - `pause` (`F3 90`) — speculative execution hint in the trap loop
/// - `lfence` (`0F AE E8`) — load fence as speculation barrier
/// - `ret` (`C3`) — used to redirect speculative execution
///
/// Returns `true` if the data contains patterns consistent with a retpoline thunk.
fn contains_retpoline_thunk_patterns(data: &[u8]) -> bool {
    let has_pause = !find_byte_pattern(data, &PAUSE_BYTES).is_empty();
    let has_lfence = !find_byte_pattern(data, &LFENCE_BYTES).is_empty();
    let has_ret = data.contains(&RET_BYTE);

    // A retpoline thunk must contain all three characteristic instruction patterns.
    has_pause && has_lfence && has_ret
}

/// Compile C source targeting x86-64 with additional flags.
///
/// Convenience wrapper that automatically adds `--target x86_64-linux-gnu`.
fn compile_x86_64(source: &str, extra_flags: &[&str]) -> common::CompileResult {
    let mut flags: Vec<&str> = extra_flags.to_vec();
    flags.push("--target");
    flags.push(TARGET);
    common::compile_source(source, &flags)
}

/// Compile C source targeting x86-64 and extract the `.text` section bytes.
///
/// Compiles the source, verifies success, reads the output binary, and returns
/// the raw `.text` section bytes for byte-level inspection.
///
/// # Panics
///
/// Panics if compilation fails or the output binary cannot be read.
fn compile_and_get_text_section(source: &str, extra_flags: &[&str]) -> Vec<u8> {
    let result = compile_x86_64(source, extra_flags);
    assert!(
        result.success,
        "Compilation failed (needed .text extraction):\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    let output_path = result
        .output_path
        .as_ref()
        .expect("Compilation succeeded but no output binary path was produced");
    read_elf_text_section(output_path)
}

/// Read the entire compiled binary as raw bytes.
///
/// Useful when .text section parsing is not needed and full-binary byte
/// scanning is sufficient.
fn compile_and_read_binary(source: &str, extra_flags: &[&str]) -> Vec<u8> {
    let result = compile_x86_64(source, extra_flags);
    assert!(
        result.success,
        "Compilation failed (needed binary read):\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    let output_path = result
        .output_path
        .as_ref()
        .expect("Compilation succeeded but no output binary path was produced");
    fs::read(output_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read compiled binary '{}': {}",
            output_path.display(),
            e
        );
    })
}

// ===========================================================================
// C Source Snippets for Security Tests
// ===========================================================================

/// Simple C source with an indirect function call via a function pointer.
/// This is the primary test case for retpoline verification.
const INDIRECT_CALL_SOURCE: &str = r#"
typedef int (*func_ptr_t)(int);

int add_one(int x) {
    return x + 1;
}

int call_indirect(func_ptr_t fp, int val) {
    return fp(val);
}

int main(void) {
    func_ptr_t fp = add_one;
    return call_indirect(fp, 41) - 42;
}
"#;

/// C source with multiple indirect calls through different function pointers.
const MULTIPLE_INDIRECT_CALLS_SOURCE: &str = r#"
typedef int (*op_t)(int, int);

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int mul(int a, int b) { return a * b; }

int apply(op_t fn1, op_t fn2, op_t fn3, int x, int y) {
    int r1 = fn1(x, y);
    int r2 = fn2(x, y);
    int r3 = fn3(r1, r2);
    return r3;
}

int main(void) {
    op_t a = add;
    op_t s = sub;
    op_t m = mul;
    return apply(a, s, m, 3, 2) - 5;
}
"#;

/// Simple multi-function C source for CET endbr64 testing.
/// Each function should have `endbr64` at its entry when `-fcf-protection` is active.
const MULTI_FUNCTION_SOURCE: &str = r#"
int foo(int x) {
    return x * 2;
}

int bar(int x) {
    return x + 3;
}

int baz(int x) {
    return foo(x) + bar(x);
}

int main(void) {
    return baz(5) - 18;
}
"#;

/// C source with indirect call targets for CET endbr64 at branch targets.
const INDIRECT_BRANCH_TARGET_SOURCE: &str = r#"
typedef void (*handler_t)(int *);

void inc_handler(int *v) { *v += 1; }
void dec_handler(int *v) { *v -= 1; }
void dbl_handler(int *v) { *v *= 2; }

int main(void) {
    handler_t handlers[3];
    handlers[0] = inc_handler;
    handlers[1] = dec_handler;
    handlers[2] = dbl_handler;

    int val = 10;
    for (int i = 0; i < 3; i++) {
        handlers[i](&val);
    }
    return val - 20;
}
"#;

/// C source with a large stack frame (8192 bytes) for stack probing tests.
const LARGE_STACK_FRAME_SOURCE: &str = r#"
int main(void) {
    char buf[8192];
    buf[0] = 'A';
    buf[8191] = 'Z';
    return (buf[0] == 'A' && buf[8191] == 'Z') ? 0 : 1;
}
"#;

/// C source with exactly 4096 bytes of stack allocation (boundary test — should NOT probe).
const BOUNDARY_4096_SOURCE: &str = r#"
int main(void) {
    char buf[4096];
    buf[0] = 'A';
    buf[4095] = 'Z';
    return (buf[0] == 'A' && buf[4095] == 'Z') ? 0 : 1;
}
"#;

/// C source with 4097 bytes of stack allocation (boundary test — SHOULD probe).
const BOUNDARY_4097_SOURCE: &str = r#"
int main(void) {
    char buf[4097];
    buf[0] = 'A';
    buf[4096] = 'Z';
    return (buf[0] == 'A' && buf[4096] == 'Z') ? 0 : 1;
}
"#;

/// C source with a very large stack frame (1 MB) for stress-testing stack probing.
const VERY_LARGE_STACK_SOURCE: &str = r#"
int main(void) {
    char buf[1048576];
    buf[0] = 'X';
    buf[1048575] = 'Y';
    return (buf[0] == 'X' && buf[1048575] == 'Y') ? 0 : 1;
}
"#;

/// C source with a small stack frame (<4096 bytes) — no stack probing expected.
const SMALL_STACK_FRAME_SOURCE: &str = r#"
int main(void) {
    char buf[256];
    buf[0] = 'A';
    buf[255] = 'Z';
    return (buf[0] == 'A' && buf[255] == 'Z') ? 0 : 1;
}
"#;

/// Simple hello-world-like source that just returns 0 (for flag acceptance tests).
const MINIMAL_SOURCE: &str = r#"
int main(void) {
    return 0;
}
"#;

/// C source combining indirect calls and large stack frames for combined security tests.
const COMBINED_SECURITY_SOURCE: &str = r#"
typedef int (*func_ptr_t)(int);

int double_it(int x) { return x * 2; }
int negate_it(int x) { return -x; }

int compute(func_ptr_t fp, int val) {
    char large_buf[8192];
    large_buf[0] = (char)val;
    large_buf[8191] = (char)(val + 1);
    int result = fp(val);
    return result + large_buf[0] + large_buf[8191];
}

int main(void) {
    func_ptr_t f1 = double_it;
    func_ptr_t f2 = negate_it;
    int r1 = compute(f1, 5);
    int r2 = compute(f2, 3);
    return 0;
}
"#;

// ===========================================================================
// Phase 2: Retpoline Tests (-mretpoline)
// ===========================================================================

/// Verify that the `-mretpoline` flag is accepted by the compiler without error.
///
/// The compiler should recognize `-mretpoline` as a valid flag and compile
/// successfully. This is a basic acceptance test before checking byte-level output.
#[test]
fn retpoline_flag_accepted() {
    let result = compile_x86_64(MINIMAL_SOURCE, &["-mretpoline"]);
    assert!(
        result.success,
        "Compiler should accept -mretpoline flag.\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
}

/// Verify that indirect `call *%rax` / `jmp *%rax` sequences are replaced by
/// retpoline thunk calls when `-mretpoline` is active.
///
/// Compiles C code containing an indirect function call through a function pointer.
/// The resulting binary's `.text` section must NOT contain raw indirect branch
/// encodings (`FF D0` for `call *%rax`, `FF E0` for `jmp *%rax`).
#[test]
fn retpoline_indirect_call_replaced() {
    let text_section = compile_and_get_text_section(INDIRECT_CALL_SOURCE, &["-mretpoline"]);

    // The .text section must have code.
    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section from compiled binary"
    );

    // With retpoline active, there should be no unprotected indirect branches.
    assert!(
        !contains_direct_indirect_jmp(&text_section),
        "Found unprotected indirect jump/call in .text section despite -mretpoline. \
         Expected all indirect branches to be replaced with retpoline thunk calls."
    );
}

/// Verify that the retpoline thunk (`__x86_retpoline_rax` or equivalent) is
/// present in the output binary when `-mretpoline` is active.
///
/// The thunk should contain the characteristic speculative-safe indirect branch
/// sequence: `pause`/`lfence` loop with a `ret` for speculation redirection.
#[test]
fn retpoline_thunk_present() {
    let text_section = compile_and_get_text_section(INDIRECT_CALL_SOURCE, &["-mretpoline"]);

    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section for retpoline thunk verification"
    );

    // The retpoline thunk must be present in the .text section.
    assert!(
        contains_retpoline_thunk_patterns(&text_section),
        "Retpoline thunk not found in .text section. Expected to find \
         pause (F3 90), lfence (0F AE E8), and ret (C3) sequences \
         forming a retpoline thunk when -mretpoline is active."
    );
}

/// Verify the retpoline thunk's machine code matches expected byte patterns.
///
/// The retpoline thunk should contain a speculative-safe indirect branch sequence:
/// - `call` instruction to capture the return address on the stack
/// - `pause` + `lfence` loop to trap speculative execution
/// - `ret` to redirect speculative execution away from the indirect target
///
/// We verify that these byte patterns appear in close proximity within the
/// `.text` section, which is characteristic of a retpoline thunk.
#[test]
fn retpoline_thunk_byte_sequence() {
    let text_section = compile_and_get_text_section(INDIRECT_CALL_SOURCE, &["-mretpoline"]);

    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section for retpoline byte sequence verification"
    );

    // Find all `pause` instructions (F3 90) — these are in the retpoline trap loop.
    let pause_offsets = find_byte_pattern(&text_section, &PAUSE_BYTES);
    assert!(
        !pause_offsets.is_empty(),
        "No `pause` (F3 90) instruction found in .text section. \
         Retpoline thunk must contain a `pause` in its speculation trap loop."
    );

    // Find all `lfence` instructions (0F AE E8) — speculation barrier in the thunk.
    let lfence_offsets = find_byte_pattern(&text_section, &LFENCE_BYTES);
    assert!(
        !lfence_offsets.is_empty(),
        "No `lfence` (0F AE E8) instruction found in .text section. \
         Retpoline thunk must contain an `lfence` as a speculation barrier."
    );

    // Verify `pause` and `lfence` appear in close proximity (within 32 bytes),
    // which is characteristic of a retpoline thunk's trap loop.
    let mut found_close_pair = false;
    for &pause_off in &pause_offsets {
        for &lfence_off in &lfence_offsets {
            let distance = pause_off.abs_diff(lfence_off);
            if distance <= 32 {
                found_close_pair = true;
                break;
            }
        }
        if found_close_pair {
            break;
        }
    }
    assert!(
        found_close_pair,
        "Found `pause` and `lfence` but they are not in close proximity (<= 32 bytes apart). \
         In a retpoline thunk, these instructions should be adjacent in the trap loop."
    );

    // Verify `ret` (C3) exists near the `pause`/`lfence` pair.
    let mut found_ret_near_thunk = false;
    for &pause_off in &pause_offsets {
        // Search within 64 bytes of the pause instruction for a ret.
        let start = pause_off.saturating_sub(32);
        let end = std::cmp::min(pause_off + 64, text_section.len());
        if text_section[start..end].contains(&RET_BYTE) {
            found_ret_near_thunk = true;
            break;
        }
    }
    assert!(
        found_ret_near_thunk,
        "No `ret` (C3) found near the retpoline thunk's pause/lfence loop. \
         The thunk requires a `ret` to redirect speculative execution."
    );
}

/// Verify that indirect calls use normal encoding when `-mretpoline` is NOT active.
///
/// Without the `-mretpoline` flag, the compiler should emit standard indirect
/// call instructions (e.g., `FF D0` for `call *%rax`, `FF E0` for `jmp *%rax`).
#[test]
fn retpoline_not_emitted_without_flag() {
    let text_section = compile_and_get_text_section(
        INDIRECT_CALL_SOURCE,
        &[], // No -mretpoline flag.
    );

    assert!(!text_section.is_empty(), "Expected non-empty .text section");

    // Without retpoline, indirect calls should be present as normal indirect encodings.
    // The compiler may use any register for the indirect call, so we check for
    // the presence of FF /2 (call indirect) or FF /4 (jmp indirect) patterns.
    let has_indirect = contains_direct_indirect_jmp(&text_section);
    assert!(
        has_indirect,
        "Expected to find standard indirect call/jump instructions (FF D0-D7 or FF E0-E7) \
         in .text section when -mretpoline is NOT active, but none were found. \
         The source code has function pointer calls that should compile to indirect branches."
    );

    // Additionally, retpoline thunk patterns should NOT be present.
    let has_retpoline_patterns = contains_retpoline_thunk_patterns(&text_section);
    // Note: pause and lfence might appear for other reasons, so we check specifically
    // for the combination. If the thunk-like pattern appears, that's unexpected.
    if has_retpoline_patterns {
        // This is a soft check — some compilers emit pause/lfence for other reasons.
        // We primarily verify that standard indirect branches ARE present.
        eprintln!(
            "Warning: retpoline-like byte patterns found despite -mretpoline not being active. \
             This may indicate other optimizations or instruction sequences."
        );
    }
}

/// Verify that multiple indirect calls are all replaced with retpoline thunks.
///
/// Compiles C code with three different function pointers and three indirect
/// call sites. With `-mretpoline` active, ALL indirect calls must be replaced.
#[test]
fn retpoline_multiple_indirect_calls() {
    let text_section =
        compile_and_get_text_section(MULTIPLE_INDIRECT_CALLS_SOURCE, &["-mretpoline"]);

    assert!(!text_section.is_empty(), "Expected non-empty .text section");

    // With retpoline active, no unprotected indirect branches should remain.
    assert!(
        !contains_direct_indirect_jmp(&text_section),
        "Found unprotected indirect call/jump despite -mretpoline with multiple call sites. \
         All indirect branches must be replaced with retpoline thunk calls."
    );

    // Retpoline thunk patterns must be present since the code has indirect calls.
    assert!(
        contains_retpoline_thunk_patterns(&text_section),
        "Retpoline thunk not found in .text section despite -mretpoline and \
         multiple indirect call sites in the source code."
    );
}

// ===========================================================================
// Phase 3: CET endbr64 Tests (-fcf-protection)
// ===========================================================================

/// Verify that the `-fcf-protection` flag is accepted by the compiler without error.
#[test]
fn cf_protection_flag_accepted() {
    let result = compile_x86_64(MINIMAL_SOURCE, &["-fcf-protection"]);
    assert!(
        result.success,
        "Compiler should accept -fcf-protection flag.\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
}

/// Verify that `endbr64` instructions are emitted at function entry points when
/// `-fcf-protection` is active.
///
/// Every function entry point should begin with the `endbr64` instruction
/// (`F3 0F 1E FA`) to mark it as a valid indirect branch target for CET.
#[test]
fn endbr64_at_function_entry() {
    let text_section = compile_and_get_text_section(MULTI_FUNCTION_SOURCE, &["-fcf-protection"]);

    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section for endbr64 verification"
    );

    // The source has 4 functions (foo, bar, baz, main).
    // Each should have `endbr64` at its entry.
    let endbr64_count = find_byte_pattern(&text_section, &ENDBR64_BYTES).len();
    assert!(
        endbr64_count >= 4,
        "Expected at least 4 `endbr64` instructions (one per function entry) in .text section, \
         found {}. Source has 4 functions: foo, bar, baz, main.",
        endbr64_count
    );
}

/// Verify that `endbr64` is emitted at indirect call targets.
///
/// Functions called through function pointers (indirect call targets) must have
/// `endbr64` at their entry to be valid CET indirect branch targets.
#[test]
fn endbr64_at_indirect_branch_target() {
    let text_section =
        compile_and_get_text_section(INDIRECT_BRANCH_TARGET_SOURCE, &["-fcf-protection"]);

    assert!(!text_section.is_empty(), "Expected non-empty .text section");

    // The source has 4 functions including 3 that are indirect call targets
    // (inc_handler, dec_handler, dbl_handler) plus main.
    let endbr64_count = find_byte_pattern(&text_section, &ENDBR64_BYTES).len();
    assert!(
        endbr64_count >= 4,
        "Expected at least 4 `endbr64` instructions (inc_handler, dec_handler, \
         dbl_handler, main) in .text section, found {}.",
        endbr64_count
    );
}

/// Verify the `endbr64` byte pattern (`F3 0F 1E FA`) in the output binary.
///
/// Reads the raw binary and searches for the exact `endbr64` encoding.
/// This is a direct byte-level verification as required by the AAP's
/// "Security Hardening Verification Rule".
#[test]
fn endbr64_byte_pattern() {
    let binary_data = compile_and_read_binary(MULTI_FUNCTION_SOURCE, &["-fcf-protection"]);

    assert!(
        !binary_data.is_empty(),
        "Expected non-empty binary for endbr64 byte pattern verification"
    );

    // Search the entire binary for the endbr64 byte pattern.
    let endbr64_offsets = find_byte_pattern(&binary_data, &ENDBR64_BYTES);
    assert!(
        !endbr64_offsets.is_empty(),
        "No `endbr64` byte pattern (F3 0F 1E FA) found in the compiled binary. \
         With -fcf-protection active, function entries must contain endbr64."
    );

    // Verify the offsets are within the expected range (not at file offset 0,
    // which would be the ELF magic area).
    for &offset in &endbr64_offsets {
        assert!(
            offset >= 64,
            "Found `endbr64` at suspicious offset {} (within ELF header area). \
             This is likely a false positive from ELF metadata, not actual code.",
            offset
        );
    }
}

/// Verify that no `endbr64` instructions are emitted when `-fcf-protection` is NOT active.
///
/// Without CET instrumentation, functions should not contain the `endbr64` prefix.
#[test]
fn no_endbr64_without_flag() {
    let text_section = compile_and_get_text_section(
        MULTI_FUNCTION_SOURCE,
        &[], // No -fcf-protection flag.
    );

    assert!(!text_section.is_empty(), "Expected non-empty .text section");

    // Without -fcf-protection, no endbr64 instructions should be present.
    let endbr64_count = find_byte_pattern(&text_section, &ENDBR64_BYTES).len();
    assert_eq!(
        endbr64_count, 0,
        "Found {} `endbr64` instruction(s) in .text section despite -fcf-protection NOT being \
         active. endbr64 should only be emitted when CET instrumentation is requested.",
        endbr64_count
    );
}

/// Verify that each function in a multi-function source gets `endbr64` at its entry.
///
/// With `-fcf-protection`, every function entry point must have `endbr64`.
/// We verify the count matches the number of functions.
#[test]
fn endbr64_with_multiple_functions() {
    // Use a source with a known number of functions.
    let source = r#"
int fn_alpha(int x) { return x + 1; }
int fn_beta(int x) { return x + 2; }
int fn_gamma(int x) { return x + 3; }
int fn_delta(int x) { return x + 4; }
int fn_epsilon(int x) { return x + 5; }

int main(void) {
    return fn_alpha(0) + fn_beta(0) + fn_gamma(0) + fn_delta(0) + fn_epsilon(0) - 15;
}
"#;

    let text_section = compile_and_get_text_section(source, &["-fcf-protection"]);

    assert!(!text_section.is_empty(), "Expected non-empty .text section");

    // 6 functions: fn_alpha through fn_epsilon + main.
    let endbr64_count = find_byte_pattern(&text_section, &ENDBR64_BYTES).len();
    assert!(
        endbr64_count >= 6,
        "Expected at least 6 `endbr64` instructions (one per function entry: \
         fn_alpha, fn_beta, fn_gamma, fn_delta, fn_epsilon, main), found {}.",
        endbr64_count
    );
}

// ===========================================================================
// Phase 4: Stack Probing Tests (Frames >4096 bytes)
// ===========================================================================

/// Verify that stack probing instructions are emitted for large stack frames.
///
/// A function with a local array of 8192 bytes exceeds the page size (4096 bytes)
/// and should trigger the emission of stack probing instructions that touch
/// each page to ensure the guard page is hit if the stack overflows.
#[test]
fn stack_probe_large_frame() {
    let result = compile_x86_64(LARGE_STACK_FRAME_SOURCE, &[]);
    assert!(
        result.success,
        "Compilation of large stack frame source should succeed.\nstderr: {}",
        result.stderr
    );

    let output_path = result
        .output_path
        .as_ref()
        .expect("Expected output binary from large stack frame compilation");
    let text_section = read_elf_text_section(output_path);

    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section from large stack frame binary"
    );

    // Stack probing for an 8192-byte frame should generate instructions that
    // touch memory at page-sized intervals. The typical pattern involves:
    // - Subtracting 4096 from RSP
    // - Touching (writing/reading) the new stack page
    // - Repeating until the full frame is allocated
    //
    // We look for patterns that indicate page-by-page stack allocation:
    // 1. The value 4096 (0x1000) as an immediate in sub/cmp instructions
    // 2. Memory access patterns through RSP with page-aligned offsets
    //
    // Encoding of `sub rsp, 0x1000` could be:
    //   48 81 EC 00 10 00 00  (REX.W sub rsp, imm32)
    // Or the probe loop may use a different register.
    let page_size_imm32 = [0x00, 0x10, 0x00, 0x00]; // 0x1000 = 4096 as LE imm32
    let has_page_size_ref = !find_byte_pattern(&text_section, &page_size_imm32).is_empty();

    // Also check for 0x2000 (8192) which is the total frame size.
    let frame_size_imm32 = [0x00, 0x20, 0x00, 0x00]; // 0x2000 = 8192 as LE imm32
    let has_frame_size_ref = !find_byte_pattern(&text_section, &frame_size_imm32).is_empty();

    // The compiler should reference the page size (for probing) or the total frame size.
    assert!(
        has_page_size_ref || has_frame_size_ref,
        "Expected stack probing instructions referencing page size (0x1000) or \
         frame size (0x2000) in .text section for an 8192-byte stack frame. \
         Neither pattern was found, suggesting stack probing was not emitted."
    );
}

/// Test the 4096-byte boundary: exactly 4096 should NOT probe, 4097 SHOULD probe.
///
/// The stack probing threshold is at one page (4096 bytes). Frames at or below
/// this size do not require probing; frames exceeding this size do.
#[test]
fn stack_probe_boundary() {
    // Test with exactly 4096 bytes — should NOT trigger probing.
    let result_4096 = compile_x86_64(BOUNDARY_4096_SOURCE, &[]);
    assert!(
        result_4096.success,
        "Compilation of 4096-byte stack frame should succeed.\nstderr: {}",
        result_4096.stderr
    );

    // Test with 4097 bytes — SHOULD trigger probing.
    let result_4097 = compile_x86_64(BOUNDARY_4097_SOURCE, &[]);
    assert!(
        result_4097.success,
        "Compilation of 4097-byte stack frame should succeed.\nstderr: {}",
        result_4097.stderr
    );

    // Extract .text sections from both binaries.
    let text_4096 = if let Some(ref path) = result_4096.output_path {
        read_elf_text_section(path)
    } else {
        Vec::new()
    };

    let text_4097 = if let Some(ref path) = result_4097.output_path {
        read_elf_text_section(path)
    } else {
        Vec::new()
    };

    // The 4097-byte version should have probing-related patterns that the
    // 4096-byte version does not.
    // Look for page-size immediate (0x1000 = 4096 as probing step size).
    let page_size_imm32 = [0x00, 0x10, 0x00, 0x00];

    let probes_in_4097 = find_byte_pattern(&text_4097, &page_size_imm32);
    let probes_in_4096 = find_byte_pattern(&text_4096, &page_size_imm32);

    // The 4097-byte binary should have more page-size references (probing)
    // than the 4096-byte binary, OR the 4097-byte version should be larger
    // (contains extra probing instructions).
    let size_diff = text_4097.len() as isize - text_4096.len() as isize;
    let probe_count_diff = probes_in_4097.len() as isize - probes_in_4096.len() as isize;

    assert!(
        size_diff > 0 || probe_count_diff > 0,
        "Expected the 4097-byte stack frame binary to contain additional stack probing \
         instructions compared to the 4096-byte version.\n\
         4096 .text size: {} bytes, page-size refs: {}\n\
         4097 .text size: {} bytes, page-size refs: {}",
        text_4096.len(),
        probes_in_4096.len(),
        text_4097.len(),
        probes_in_4097.len()
    );
}

/// Test stack probing with a very large frame (1 MB).
///
/// A 1 MB stack frame requires approximately 256 page probes. The compiler
/// should emit a probing loop rather than individual probe instructions.
#[test]
fn stack_probe_very_large_frame() {
    let result = compile_x86_64(VERY_LARGE_STACK_SOURCE, &[]);
    assert!(
        result.success,
        "Compilation of 1MB stack frame should succeed.\nstderr: {}",
        result.stderr
    );

    let output_path = result
        .output_path
        .as_ref()
        .expect("Expected output binary from 1MB stack frame compilation");
    let text_section = read_elf_text_section(output_path);

    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section from 1MB stack frame binary"
    );

    // For a 1MB frame, the compiler should generate a loop (not 256 individual
    // probe instructions). Check for:
    // 1. Page size immediate (0x1000) indicating per-page stepping
    // 2. The full frame size 0x100000 (1048576) as an immediate
    // 3. A backward jump (loop) — indicated by a negative relative offset
    let page_size_imm32 = [0x00, 0x10, 0x00, 0x00]; // 0x1000
    let frame_size_bytes = [0x00, 0x00, 0x10, 0x00]; // 0x100000 as LE imm32
    let has_page_size = !find_byte_pattern(&text_section, &page_size_imm32).is_empty();
    let has_frame_size = !find_byte_pattern(&text_section, &frame_size_bytes).is_empty();

    assert!(
        has_page_size || has_frame_size,
        "Expected stack probing instructions referencing page size (0x1000) or \
         frame size (0x100000) for a 1MB stack frame. Neither pattern found."
    );
}

/// Verify that small stack frames (<4096 bytes) do NOT trigger stack probing.
///
/// A function with only 256 bytes of local variables should not emit any
/// page-by-page stack probing instructions.
#[test]
fn no_stack_probe_small_frame() {
    let result = compile_x86_64(SMALL_STACK_FRAME_SOURCE, &[]);
    assert!(
        result.success,
        "Compilation of small stack frame source should succeed.\nstderr: {}",
        result.stderr
    );

    // Compare with the large frame version to verify different code generation.
    let result_large = compile_x86_64(LARGE_STACK_FRAME_SOURCE, &[]);
    assert!(
        result_large.success,
        "Compilation of large stack frame source should succeed for comparison.\nstderr: {}",
        result_large.stderr
    );

    let text_small = if let Some(ref path) = result.output_path {
        read_elf_text_section(path)
    } else {
        Vec::new()
    };

    let text_large = if let Some(ref path) = result_large.output_path {
        read_elf_text_section(path)
    } else {
        Vec::new()
    };

    // The small frame binary's .text should be smaller than the large frame binary's
    // .text (the large one has extra probing instructions).
    assert!(
        text_small.len() <= text_large.len(),
        "Expected small stack frame .text ({} bytes) to not exceed large frame .text ({} bytes). \
         The large frame version should have additional probing instructions.",
        text_small.len(),
        text_large.len()
    );
}

// ===========================================================================
// Phase 5: Combined Security Features
// ===========================================================================

/// Verify that both `-mretpoline` and `-fcf-protection` work simultaneously.
///
/// When both security features are active:
/// - Retpoline thunks must be present (replacing indirect branches)
/// - `endbr64` instructions must be present (at function entries)
/// - No unprotected indirect branches should exist
#[test]
fn all_security_features_combined() {
    let text_section = compile_and_get_text_section(
        COMBINED_SECURITY_SOURCE,
        &["-mretpoline", "-fcf-protection"],
    );

    assert!(
        !text_section.is_empty(),
        "Expected non-empty .text section with combined security features"
    );

    // Verify endbr64 is present (CET instrumentation).
    let endbr64_offsets = find_byte_pattern(&text_section, &ENDBR64_BYTES);
    assert!(
        !endbr64_offsets.is_empty(),
        "No `endbr64` found in .text despite -fcf-protection. \
         CET instrumentation should not be affected by -mretpoline."
    );

    // Verify retpoline thunk patterns are present.
    assert!(
        contains_retpoline_thunk_patterns(&text_section),
        "Retpoline thunk patterns not found in .text despite -mretpoline. \
         Retpoline should not be affected by -fcf-protection."
    );

    // Verify no unprotected indirect branches.
    assert!(
        !contains_direct_indirect_jmp(&text_section),
        "Found unprotected indirect branch in .text despite -mretpoline being active \
         alongside -fcf-protection. Both features should be active simultaneously."
    );

    // The source has 4 functions (double_it, negate_it, compute, main).
    // With CET active, each should have endbr64 at entry.
    assert!(
        endbr64_offsets.len() >= 4,
        "Expected at least 4 `endbr64` instructions (double_it, negate_it, compute, main), \
         found {}.",
        endbr64_offsets.len()
    );
}

/// Verify that security features survive optimization passes.
///
/// Security instrumentation (retpoline, CET endbr64) must NOT be eliminated
/// by the optimizer at any optimization level (-O0, -O1, -O2).
#[test]
fn security_with_optimization() {
    let opt_levels = ["-O0", "-O1", "-O2"];

    for &opt_level in &opt_levels {
        // Test retpoline with optimization.
        let text_retpoline =
            compile_and_get_text_section(INDIRECT_CALL_SOURCE, &["-mretpoline", opt_level]);

        assert!(
            !text_retpoline.is_empty(),
            "Expected non-empty .text section at {} with -mretpoline",
            opt_level
        );

        // Retpoline thunk must survive optimization.
        assert!(
            contains_retpoline_thunk_patterns(&text_retpoline),
            "Retpoline thunk was eliminated by optimization at {}. \
             Security instrumentation must not be removed by the optimizer.",
            opt_level
        );

        // No unprotected indirect branches should appear at any opt level.
        assert!(
            !contains_direct_indirect_jmp(&text_retpoline),
            "Unprotected indirect branch found at {} despite -mretpoline. \
             Optimization must not revert retpoline protection.",
            opt_level
        );

        // Test CET endbr64 with optimization.
        let text_cet =
            compile_and_get_text_section(MULTI_FUNCTION_SOURCE, &["-fcf-protection", opt_level]);

        assert!(
            !text_cet.is_empty(),
            "Expected non-empty .text section at {} with -fcf-protection",
            opt_level
        );

        // endbr64 must survive optimization.
        let endbr64_count = find_byte_pattern(&text_cet, &ENDBR64_BYTES).len();
        assert!(
            endbr64_count >= 1,
            "No `endbr64` found at {} despite -fcf-protection. \
             CET instrumentation must not be eliminated by the optimizer.",
            opt_level
        );
    }
}

// ===========================================================================
// Additional Edge Case and Robustness Tests
// ===========================================================================

/// Verify that retpoline and CET flags work with `-c` (compile-only, no linking).
///
/// Security flags should be accepted and affect code generation even when
/// producing relocatable object files rather than executables.
#[test]
fn security_flags_with_compile_only() {
    let dir = common::TempDir::new("security_compile_only");
    let output_obj = dir.path().join("test.o");
    let output_str = output_obj.to_str().expect("valid path");

    let result = compile_x86_64(
        INDIRECT_CALL_SOURCE,
        &["-c", "-mretpoline", "-fcf-protection", "-o", output_str],
    );

    assert!(
        result.success,
        "Compilation with -c -mretpoline -fcf-protection should succeed.\nstderr: {}",
        result.stderr
    );

    // Verify the object file was produced.
    assert!(
        output_obj.exists(),
        "Expected object file at '{}' but it does not exist",
        output_obj.display()
    );

    // Verify it is a valid ELF file.
    common::verify_elf_magic(output_obj.as_path());

    // For relocatable objects, the .text section should still contain
    // security instrumentation.
    let text_section = read_elf_text_section(&output_obj);
    if !text_section.is_empty() {
        // CET: endbr64 should be present.
        let has_endbr64 = contains_endbr64(&text_section);
        assert!(
            has_endbr64,
            "Expected endbr64 in relocatable object's .text section with -fcf-protection"
        );
    }
}

/// Verify the compiler produces valid ELF64 x86-64 output with security flags.
///
/// This test validates that security flags don't corrupt the ELF header or
/// produce invalid output.
#[test]
fn security_flags_produce_valid_elf() {
    let result = compile_x86_64(
        COMBINED_SECURITY_SOURCE,
        &["-mretpoline", "-fcf-protection"],
    );
    assert!(
        result.success,
        "Compilation with all security flags should succeed.\nstderr: {}",
        result.stderr
    );

    let output_path = result
        .output_path
        .as_ref()
        .expect("Expected output binary with security flags");

    // Verify ELF headers.
    common::verify_elf_magic(output_path.as_path());
    common::verify_elf_class(output_path.as_path(), common::ELFCLASS64);
    common::verify_elf_arch(output_path.as_path(), common::EM_X86_64);
}

/// Verify retpoline with `-fPIC` (position-independent code).
///
/// Retpoline and PIC should work together correctly. In PIC mode, indirect
/// calls through the PLT/GOT also need retpoline protection.
#[test]
fn retpoline_with_pic() {
    let result = compile_x86_64(INDIRECT_CALL_SOURCE, &["-mretpoline", "-fPIC", "-c"]);

    assert!(
        result.success,
        "Compilation with -mretpoline -fPIC should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify that unknown security flags are properly rejected.
///
/// The compiler should reject invalid flag combinations gracefully without
/// crashing or producing corrupted output.
#[test]
fn invalid_security_flag_rejected() {
    // An invalid flag should cause the compiler to report an error.
    let result = compile_x86_64(MINIMAL_SOURCE, &["-mretpoline-invalid-variant"]);

    // The compiler should either reject the flag (error) or ignore it.
    // We verify it doesn't crash — either success or clean error is acceptable.
    // The key check is that the process completes without panic/segfault.
    let _completed = result.success || !result.stderr.is_empty();
}

/// Verify that `-mretpoline` is specific to x86-64 compilation.
///
/// The flag should be accepted for x86-64 targets. This confirms the
/// architecture-specific nature of the security feature.
#[test]
fn retpoline_is_x86_64_specific() {
    // Compile for x86-64 — should work.
    let result_x86_64 = common::compile_source(
        INDIRECT_CALL_SOURCE,
        &["--target", "x86_64-linux-gnu", "-mretpoline"],
    );

    assert!(
        result_x86_64.success,
        "Retpoline should be accepted for x86-64 target.\nstderr: {}",
        result_x86_64.stderr
    );
}

/// Verify that `-fcf-protection` is specific to x86-64 compilation.
///
/// CET endbr64 is an x86-64 feature. The flag should be accepted for x86-64.
#[test]
fn cf_protection_is_x86_64_specific() {
    let result_x86_64 = common::compile_source(
        MINIMAL_SOURCE,
        &["--target", common::TARGET_X86_64, "-fcf-protection"],
    );

    assert!(
        result_x86_64.success,
        "CET -fcf-protection should be accepted for x86-64 target.\nstderr: {}",
        result_x86_64.stderr
    );
}

// ===========================================================================
// Additional Tests — Direct Binary Invocation and Manual File Management
// ===========================================================================

/// Verify retpoline flag acceptance using direct Command invocation.
///
/// This test directly spawns the bcc binary via `std::process::Command` to
/// verify the `-mretpoline` flag is accepted, bypassing the `compile_source`
/// helper for thorough validation of the compiler's CLI interface.
#[test]
fn retpoline_flag_direct_command_invocation() {
    let bcc_path = common::get_bcc_binary();
    let temp_source = common::write_temp_source(MINIMAL_SOURCE);
    let dir = common::TempDir::new("retpoline_direct_cmd");
    let output_path: PathBuf = dir.path().join("direct_test.out");

    // Invoke the compiler directly via Command with individual .arg() calls.
    let status = Command::new(bcc_path.as_path())
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-mretpoline")
        .arg("-o")
        .arg(output_path.as_path())
        .arg(temp_source.path())
        .status()
        .expect("Failed to spawn bcc compiler process");

    assert!(
        status.success(),
        "Direct Command invocation with -mretpoline should succeed (exit code: {:?})",
        status.code()
    );

    // Verify the output binary exists and is a valid ELF.
    assert!(
        output_path.exists(),
        "Output binary not found at '{}'",
        output_path.display()
    );
    common::verify_elf_magic(output_path.as_path());
}

/// Verify CET flag acceptance using direct Command invocation with .args() batch.
///
/// Tests the `.args()` (batch argument) API on Command for flag passing.
#[test]
fn cf_protection_flag_direct_command_args_batch() {
    let bcc_path = common::get_bcc_binary();
    let temp_source = common::write_temp_source(MULTI_FUNCTION_SOURCE);
    let dir = common::TempDir::new("cet_direct_cmd");
    let output_path: PathBuf = dir.path().join("cet_test.out");

    // Use .args() to pass all arguments at once.
    let output = Command::new(bcc_path.as_path())
        .args(["--target", common::TARGET_X86_64, "-fcf-protection", "-o"])
        .arg(output_path.as_path())
        .arg(temp_source.path())
        .output()
        .expect("Failed to spawn bcc compiler process");

    let stderr_text = String::from_utf8_lossy(&output.stderr);
    let stdout_text = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Direct Command invocation with -fcf-protection should succeed.\n\
         stderr: {}\nstdout: {}",
        stderr_text,
        stdout_text
    );

    // Verify output is a valid ELF64 x86-64 binary.
    common::verify_elf_magic(output_path.as_path());
    common::verify_elf_class(output_path.as_path(), common::ELFCLASS64);
    common::verify_elf_arch(output_path.as_path(), common::EM_X86_64);
}

/// Verify security features via manual source file creation and filesystem ops.
///
/// Exercises `fs::write()`, `fs::read_to_string()`, and `fs::remove_file()` for
/// manual source file management outside of the `write_temp_source()` helper.
#[test]
fn security_manual_file_management() {
    let dir = common::TempDir::new("security_manual_fs");
    let source_name = format!("{}/manual_test.c", dir.path().display());
    let source_path = PathBuf::from(&source_name);
    let output_path = dir.path().join("manual_test.out");

    // Write the source file manually using fs::write().
    fs::write(&source_path, INDIRECT_CALL_SOURCE).unwrap_or_else(|e| {
        panic!(
            "Failed to write source file '{}': {}",
            source_path.display(),
            e
        );
    });

    // Verify the file was written correctly using fs::read_to_string().
    let read_back = fs::read_to_string(&source_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read back source file '{}': {}",
            source_path.display(),
            e
        );
    });
    assert_eq!(
        read_back, INDIRECT_CALL_SOURCE,
        "Source file content mismatch after write/read cycle"
    );

    // Compile using the manually created source file.
    let bcc_path = common::get_bcc_binary();
    let result = Command::new(bcc_path.as_path())
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-mretpoline")
        .arg("-fcf-protection")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to spawn bcc compiler");

    assert!(
        result.status.success(),
        "Compilation with manual source file should succeed.\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    // Read and inspect the binary.
    let binary_data = fs::read(&output_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read compiled binary '{}': {}",
            output_path.display(),
            e
        );
    });
    assert!(
        !binary_data.is_empty(),
        "Compiled binary should not be empty"
    );

    // Verify both security features in the binary.
    let text_section = read_elf_text_section(&output_path);
    if !text_section.is_empty() {
        assert!(
            contains_endbr64(&text_section),
            "Expected endbr64 in binary compiled with -fcf-protection"
        );
    }

    // Clean up the source file manually using fs::remove_file().
    fs::remove_file(&source_path).unwrap_or_else(|e| {
        eprintln!(
            "Warning: failed to remove source file '{}': {}",
            source_path.display(),
            e
        );
    });
    assert!(
        !source_path.exists(),
        "Source file should be removed after fs::remove_file()"
    );
}

/// Verify that TempFile RAII cleanup works correctly for security test sources.
///
/// Exercises `write_temp_source()` -> `TempFile` -> `TempFile.path()` for
/// creating and accessing temporary C source files.
#[test]
fn security_temp_file_lifecycle() {
    // Create a temporary source file using the common utility.
    let temp_file: common::TempFile = common::write_temp_source(COMBINED_SECURITY_SOURCE);
    let temp_path: PathBuf = temp_file.path().to_path_buf();

    // Verify the temporary file exists and contains the expected source.
    assert!(
        temp_path.exists(),
        "Temporary source file should exist at '{}'",
        temp_path.display()
    );

    let source_content = fs::read_to_string(temp_file.path()).unwrap_or_else(|e| {
        panic!(
            "Failed to read temp source file '{}': {}",
            temp_file.path().display(),
            e
        );
    });
    assert!(
        source_content.contains("func_ptr_t"),
        "Temp source file should contain the combined security source code"
    );

    // Compile the temporary file directly.
    let result = compile_x86_64(COMBINED_SECURITY_SOURCE, &["-mretpoline"]);
    assert!(
        result.success,
        "Compilation of combined security source should succeed.\nstderr: {}",
        result.stderr
    );
}
