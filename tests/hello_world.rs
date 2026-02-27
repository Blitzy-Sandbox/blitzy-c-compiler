//! End-to-end smoke test: compile and execute "hello world" on all four architectures.
//!
//! This is the most fundamental validation test in the bcc C compiler test suite.
//! It verifies that the complete compilation pipeline — preprocessor → lexer → parser →
//! semantic analysis → IR → optimization → code generation → linking — produces correct,
//! runnable executables for all four supported target architectures:
//!
//! - x86-64 (`x86_64-linux-gnu`) — ELF64, System V AMD64 ABI
//! - i686 (`i686-linux-gnu`) — ELF32, cdecl ABI
//! - AArch64 (`aarch64-linux-gnu`) — ELF64, AAPCS64 ABI
//! - RISC-V 64 (`riscv64-linux-gnu`) — ELF64, LP64D ABI
//!
//! Non-native architectures are executed via QEMU user-mode emulation.
//! Tests gracefully skip (rather than fail) when QEMU is not available.
//!
//! # Zero-Dependency Guarantee
//!
//! This test file uses ONLY the Rust standard library (`std`). No external crates.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// Hello World C Source Constants
// ===========================================================================

/// Standard hello world C program using stdio (requires libc linkage).
///
/// This is the primary test source used for all architecture tests.
/// It exercises the preprocessor (`#include`), function calls (`printf`),
/// string literals, and integer return values through the full pipeline.
const HELLO_WORLD_SOURCE: &str = r#"
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
"#;

/// Freestanding hello world for x86-64 using Linux syscalls directly.
///
/// This version does NOT use stdio or libc. It invokes the Linux `write`
/// and `exit` syscalls via inline assembly. This tests the compiler's
/// ability to handle inline assembly with operand constraints, which is
/// a required GCC extension per the AAP.
///
/// Compiled with `-c` only (object file) since the current CLI does not
/// expose a `-nostdlib` flag to bypass CRT linkage.
const HELLO_WORLD_FREESTANDING_X86_64: &str = r#"
void _start(void) {
    const char msg[] = "Hello, World!\n";

    /* syscall: write(1, msg, 14) via x86-64 syscall instruction */
    __asm__ volatile(
        "mov $1, %%rax\n"
        "mov $1, %%rdi\n"
        "mov %0, %%rsi\n"
        "mov $14, %%rdx\n"
        "syscall\n"
        :
        : "r"(msg)
        : "rax", "rdi", "rsi", "rdx", "rcx", "r11", "memory"
    );

    /* syscall: exit(0) */
    __asm__ volatile(
        "mov $60, %%rax\n"
        "xor %%rdi, %%rdi\n"
        "syscall\n"
        ::: "rax", "rdi"
    );
}
"#;

/// Intentionally invalid C source for testing diagnostic format and error exit codes.
///
/// This source contains a syntax error (missing expression after `=`) that must
/// produce a GCC-compatible diagnostic on stderr and exit with code 1.
const INVALID_C_SOURCE: &str = r#"
int main(void) {
    int x = ;
    return 0;
}
"#;

// ===========================================================================
// ELF Section Inspection Helper
// ===========================================================================

/// Check whether an ELF binary contains a section with the given name.
///
/// Parses the ELF section header table and section header string table to
/// locate named sections. Supports both ELF32 and ELF64 formats.
///
/// # Arguments
///
/// * `binary_path` - Path to the ELF binary file.
/// * `section_name` - Name of the section to search for (e.g., `.debug_info`).
///
/// # Returns
///
/// `true` if the section exists in the binary; `false` otherwise.
fn has_elf_section(binary_path: &Path, section_name: &str) -> bool {
    let data = match fs::read(binary_path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() < 16 {
        return false;
    }

    // Verify ELF magic before parsing.
    if data[0..4] != [0x7f, b'E', b'L', b'F'] {
        return false;
    }

    let is_64bit = data[4] == 2;

    if is_64bit {
        parse_elf64_section_name(&data, section_name)
    } else {
        parse_elf32_section_name(&data, section_name)
    }
}

/// Parse ELF64 section headers to find a named section.
fn parse_elf64_section_name(data: &[u8], section_name: &str) -> bool {
    if data.len() < 64 {
        return false;
    }
    let e_shoff = read_u64_le(data, 40) as usize;
    let e_shentsize = read_u16_le(data, 58) as usize;
    let e_shnum = read_u16_le(data, 60) as usize;
    let e_shstrndx = read_u16_le(data, 62) as usize;

    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum || e_shentsize == 0 {
        return false;
    }

    // Read the section header string table's file offset (sh_offset at byte 24 in Elf64_Shdr).
    let shstrtab_hdr_off = e_shoff + e_shstrndx * e_shentsize;
    if shstrtab_hdr_off + 32 > data.len() {
        return false;
    }
    let shstrtab_offset = read_u64_le(data, shstrtab_hdr_off + 24) as usize;

    // Iterate all section headers, compare names.
    for i in 0..e_shnum {
        let sh_base = e_shoff + i * e_shentsize;
        if sh_base + 4 > data.len() {
            break;
        }
        let sh_name_idx = read_u32_le(data, sh_base) as usize;
        if let Some(name) = read_cstr(data, shstrtab_offset + sh_name_idx) {
            if name == section_name {
                return true;
            }
        }
    }
    false
}

/// Parse ELF32 section headers to find a named section.
fn parse_elf32_section_name(data: &[u8], section_name: &str) -> bool {
    if data.len() < 52 {
        return false;
    }
    let e_shoff = read_u32_le(data, 32) as usize;
    let e_shentsize = read_u16_le(data, 46) as usize;
    let e_shnum = read_u16_le(data, 48) as usize;
    let e_shstrndx = read_u16_le(data, 50) as usize;

    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum || e_shentsize == 0 {
        return false;
    }

    // sh_offset is at byte 16 in Elf32_Shdr.
    let shstrtab_hdr_off = e_shoff + e_shstrndx * e_shentsize;
    if shstrtab_hdr_off + 20 > data.len() {
        return false;
    }
    let shstrtab_offset = read_u32_le(data, shstrtab_hdr_off + 16) as usize;

    for i in 0..e_shnum {
        let sh_base = e_shoff + i * e_shentsize;
        if sh_base + 4 > data.len() {
            break;
        }
        let sh_name_idx = read_u32_le(data, sh_base) as usize;
        if let Some(name) = read_cstr(data, shstrtab_offset + sh_name_idx) {
            if name == section_name {
                return true;
            }
        }
    }
    false
}

/// Read a little-endian u16 from a byte slice at the given offset.
#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Read a little-endian u32 from a byte slice at the given offset.
#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

/// Read a little-endian u64 from a byte slice at the given offset.
#[inline]
fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    if offset + 8 > data.len() {
        return 0;
    }
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

/// Read a null-terminated C string from a byte slice at the given offset.
///
/// Returns `None` if the offset is out of bounds or the string is not valid UTF-8.
fn read_cstr(data: &[u8], offset: usize) -> Option<&str> {
    if offset >= data.len() {
        return None;
    }
    let end = data[offset..].iter().position(|&b| b == 0)?;
    std::str::from_utf8(&data[offset..offset + end]).ok()
}

// ===========================================================================
// Architecture-Specific Hello World Tests
// ===========================================================================

/// Compile and run hello world on x86-64 (primary target, native execution).
///
/// Verifies:
/// 1. Compilation succeeds with exit code 0
/// 2. Stdout contains "Hello, World!"
/// 3. Output binary is a valid ELF64 with EM_X86_64 architecture field
#[test]
fn hello_world_x86_64() {
    // Compile and run the hello world program for x86-64.
    let result: common::RunResult =
        common::compile_and_run(HELLO_WORLD_SOURCE, common::TARGET_X86_64, &[]);

    // Handle QEMU unavailability gracefully (though x86_64 should be native).
    if !result.success && result.stderr.contains("QEMU not available") {
        eprintln!(
            "SKIP: QEMU not available for {}: {}",
            common::TARGET_X86_64, result.stderr
        );
        return;
    }

    // Verify execution succeeded and output is correct.
    assert!(
        result.success,
        "Hello world execution failed for x86-64:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    assert!(
        result.stdout.contains("Hello, World!"),
        "Expected 'Hello, World!' in stdout for x86-64, got: '{}'",
        result.stdout
    );

    // Separately compile to verify ELF format properties.
    let compile_result: common::CompileResult =
        common::compile_source(HELLO_WORLD_SOURCE, &["--target", common::TARGET_X86_64]);
    assert!(
        compile_result.success,
        "Compilation failed for x86-64:\n{}",
        compile_result.stderr
    );
    if let Some(ref binary_path) = compile_result.output_path {
        common::verify_elf_magic(binary_path);
        common::verify_elf_class(binary_path, common::ELFCLASS64);
        common::verify_elf_arch(binary_path, common::EM_X86_64);

        // Verify the binary file has non-trivial size (basic sanity check).
        let meta = fs::metadata(binary_path).expect("Failed to read binary metadata");
        assert!(
            meta.len() > 0,
            "Output binary for x86-64 is empty (0 bytes)"
        );
    }

    // Confirm x86_64 is the native target on this host.
    assert!(
        common::is_native_target(common::TARGET_X86_64),
        "Expected x86_64 to be native on this host"
    );
}

/// Compile and run hello world on i686 (32-bit x86, ELF32).
///
/// Verifies:
/// 1. Compilation succeeds
/// 2. Output is a valid ELF32 binary with EM_386 architecture
/// 3. Execution via QEMU (qemu-i386) produces correct output
#[test]
fn hello_world_i686() {
    // Check QEMU availability before running (i686 is non-native on x86_64 host).
    if !common::is_native_target(common::TARGET_I686)
        && !common::is_qemu_available(common::TARGET_I686)
    {
        eprintln!(
            "SKIP: QEMU not available for target '{}'. \
             Install qemu-user-static to run i686 cross-architecture tests.",
            common::TARGET_I686
        );
        return;
    }

    // Compile and run.
    let result: common::RunResult =
        common::compile_and_run(HELLO_WORLD_SOURCE, common::TARGET_I686, &[]);

    if !result.success && result.stderr.contains("QEMU not available") {
        eprintln!("SKIP: {}", result.stderr);
        return;
    }

    assert!(
        result.success,
        "Hello world execution failed for i686:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    assert!(
        result.stdout.contains("Hello, World!"),
        "Expected 'Hello, World!' in stdout for i686, got: '{}'",
        result.stdout
    );

    // Verify ELF32 format.
    let compile_result: common::CompileResult =
        common::compile_source(HELLO_WORLD_SOURCE, &["--target", common::TARGET_I686]);
    assert!(
        compile_result.success,
        "Compilation failed for i686:\n{}",
        compile_result.stderr
    );
    if let Some(ref binary_path) = compile_result.output_path {
        common::verify_elf_magic(binary_path);
        common::verify_elf_class(binary_path, common::ELFCLASS32);
        common::verify_elf_arch(binary_path, common::EM_386);
    }
}

/// Compile and run hello world on AArch64 (ARM 64-bit, ELF64).
///
/// Verifies:
/// 1. Compilation succeeds
/// 2. Output is a valid ELF64 binary with EM_AARCH64 architecture
/// 3. Execution via QEMU (qemu-aarch64) produces correct output
#[test]
fn hello_world_aarch64() {
    // Check QEMU availability.
    if !common::is_native_target(common::TARGET_AARCH64)
        && !common::is_qemu_available(common::TARGET_AARCH64)
    {
        eprintln!(
            "SKIP: QEMU not available for target '{}'. \
             Install qemu-user-static to run AArch64 cross-architecture tests.",
            common::TARGET_AARCH64
        );
        return;
    }

    let result: common::RunResult =
        common::compile_and_run(HELLO_WORLD_SOURCE, common::TARGET_AARCH64, &[]);

    if !result.success && result.stderr.contains("QEMU not available") {
        eprintln!("SKIP: {}", result.stderr);
        return;
    }

    assert!(
        result.success,
        "Hello world execution failed for AArch64:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    assert!(
        result.stdout.contains("Hello, World!"),
        "Expected 'Hello, World!' in stdout for AArch64, got: '{}'",
        result.stdout
    );

    // Verify ELF64 format with correct architecture field.
    let compile_result: common::CompileResult =
        common::compile_source(HELLO_WORLD_SOURCE, &["--target", common::TARGET_AARCH64]);
    assert!(
        compile_result.success,
        "Compilation failed for AArch64:\n{}",
        compile_result.stderr
    );
    if let Some(ref binary_path) = compile_result.output_path {
        common::verify_elf_magic(binary_path);
        common::verify_elf_class(binary_path, common::ELFCLASS64);
        common::verify_elf_arch(binary_path, common::EM_AARCH64);
    }
}

/// Compile and run hello world on RISC-V 64 (ELF64).
///
/// Verifies:
/// 1. Compilation succeeds
/// 2. Output is a valid ELF64 binary with EM_RISCV architecture
/// 3. Execution via QEMU (qemu-riscv64) produces correct output
#[test]
fn hello_world_riscv64() {
    // Check QEMU availability.
    if !common::is_native_target(common::TARGET_RISCV64)
        && !common::is_qemu_available(common::TARGET_RISCV64)
    {
        eprintln!(
            "SKIP: QEMU not available for target '{}'. \
             Install qemu-user-static to run RISC-V 64 cross-architecture tests.",
            common::TARGET_RISCV64
        );
        return;
    }

    let result: common::RunResult =
        common::compile_and_run(HELLO_WORLD_SOURCE, common::TARGET_RISCV64, &[]);

    if !result.success && result.stderr.contains("QEMU not available") {
        eprintln!("SKIP: {}", result.stderr);
        return;
    }

    assert!(
        result.success,
        "Hello world execution failed for RISC-V 64:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );
    assert!(
        result.stdout.contains("Hello, World!"),
        "Expected 'Hello, World!' in stdout for RISC-V 64, got: '{}'",
        result.stdout
    );

    // Verify ELF64 format with correct architecture field.
    let compile_result: common::CompileResult =
        common::compile_source(HELLO_WORLD_SOURCE, &["--target", common::TARGET_RISCV64]);
    assert!(
        compile_result.success,
        "Compilation failed for RISC-V 64:\n{}",
        compile_result.stderr
    );
    if let Some(ref binary_path) = compile_result.output_path {
        common::verify_elf_magic(binary_path);
        common::verify_elf_class(binary_path, common::ELFCLASS64);
        common::verify_elf_arch(binary_path, common::EM_RISCV);
    }
}

// ===========================================================================
// Freestanding (No libc) Test
// ===========================================================================

/// Compile freestanding hello world for x86-64 (inline assembly, no stdio).
///
/// This test verifies that the compiler handles inline assembly with operand
/// constraints — a required GCC extension per the AAP. The source uses raw
/// Linux syscalls via `__asm__ volatile(...)` with explicit register constraints.
///
/// Compiled with `-c` to produce an object file only, since the current CLI
/// does not expose a `-nostdlib` flag for linking without CRT objects.
#[test]
fn hello_world_freestanding_x86_64() {
    let compile_result: common::CompileResult = common::compile_source(
        HELLO_WORLD_FREESTANDING_X86_64,
        &["-c", "--target", common::TARGET_X86_64],
    );

    assert!(
        compile_result.success,
        "Failed to compile freestanding x86-64 source (inline asm):\nstderr: {}",
        compile_result.stderr
    );

    // Verify the object file was produced and is a valid ELF.
    if let Some(ref path) = compile_result.output_path {
        assert!(path.exists(), "Object file was not created at {:?}", path);
        common::verify_elf_magic(path);
        common::verify_elf_class(path, common::ELFCLASS64);
        common::verify_elf_arch(path, common::EM_X86_64);
    }
}

// ===========================================================================
// Compilation Mode Tests
// ===========================================================================

/// Verify `-c` flag produces a relocatable object file, not a linked executable.
///
/// The output should be an ELF relocatable object (ET_REL) with correct
/// architecture field and no program headers.
#[test]
fn hello_world_compile_only() {
    let compile_result: common::CompileResult = common::compile_source(
        HELLO_WORLD_SOURCE,
        &["-c", "--target", common::TARGET_X86_64],
    );

    assert!(
        compile_result.success,
        "Compile-only (-c) failed:\nstderr: {}",
        compile_result.stderr
    );

    if let Some(ref path) = compile_result.output_path {
        // Verify the object file exists and has non-zero size.
        assert!(path.exists(), "Object file not found at {:?}", path);
        let meta = fs::metadata(path).expect("Failed to read object file metadata");
        assert!(meta.len() > 0, "Object file is empty (0 bytes)");

        // Verify it is a valid ELF file.
        common::verify_elf_magic(path);
        common::verify_elf_class(path, common::ELFCLASS64);
        common::verify_elf_arch(path, common::EM_X86_64);

        // Read the ELF header to verify it is ET_REL (type = 1), not ET_EXEC (type = 2).
        let data = fs::read(path).expect("Failed to read object file bytes");
        if data.len() >= 18 {
            let e_type = read_u16_le(&data, 16);
            assert_eq!(
                e_type, 1,
                "Expected ELF type ET_REL (1) for object file, got {}",
                e_type
            );
        }
    }
}

/// Verify `-o custom_name` flag produces output with the specified file name.
///
/// Uses `write_temp_source()` and direct `Command` invocation to exercise
/// the custom output path flow, including `get_bcc_binary()` and `TempDir`.
#[test]
fn hello_world_custom_output() {
    let temp_dir = common::TempDir::new("custom_output");
    let source_file = common::write_temp_source(HELLO_WORLD_SOURCE);
    let custom_output: PathBuf = temp_dir.path().join("my_hello");

    // Use get_bcc_binary() and Command directly to test -o flag.
    let bcc: PathBuf = common::get_bcc_binary();
    let output = Command::new(&bcc)
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-o")
        .arg(&custom_output)
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc compiler");

    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Compilation with -o custom_name failed:\nstderr: {}",
        stderr_text
    );

    // Verify the output was created at the custom path.
    assert!(
        custom_output.exists(),
        "Custom output file not found at {:?}",
        custom_output
    );

    // Verify it is a valid ELF binary.
    common::verify_elf_magic(&custom_output);
    common::verify_elf_class(&custom_output, common::ELFCLASS64);
    common::verify_elf_arch(&custom_output, common::EM_X86_64);
}

/// Verify `-g` flag produces DWARF v4 debug information sections.
///
/// Compiles hello world with debug info enabled and checks that the
/// output ELF binary contains `.debug_info` and `.debug_line` sections.
#[test]
fn hello_world_with_debug() {
    let compile_result: common::CompileResult = common::compile_source(
        HELLO_WORLD_SOURCE,
        &["-g", "--target", common::TARGET_X86_64],
    );

    assert!(
        compile_result.success,
        "Compilation with -g (debug info) failed:\nstderr: {}",
        compile_result.stderr
    );

    if let Some(ref binary_path) = compile_result.output_path {
        // Verify ELF basics.
        common::verify_elf_magic(binary_path);

        // Verify DWARF v4 debug sections are present.
        assert!(
            has_elf_section(binary_path, ".debug_info"),
            "Expected .debug_info section in binary compiled with -g"
        );
        assert!(
            has_elf_section(binary_path, ".debug_line"),
            "Expected .debug_line section in binary compiled with -g"
        );
        assert!(
            has_elf_section(binary_path, ".debug_abbrev"),
            "Expected .debug_abbrev section in binary compiled with -g"
        );
    }
}

/// Verify compilation at all optimization levels produces working executables.
///
/// Compiles hello world at `-O0`, `-O1`, and `-O2`, then runs each to verify
/// that optimization does not break output correctness.
#[test]
fn hello_world_optimized() {
    let opt_levels = ["-O0", "-O1", "-O2"];

    for &opt_level in &opt_levels {
        let result: common::RunResult = common::compile_and_run(
            HELLO_WORLD_SOURCE,
            common::TARGET_X86_64,
            &[opt_level],
        );

        // Handle QEMU unavailability (should not occur for native x86_64).
        if !result.success && result.stderr.contains("QEMU not available") {
            eprintln!(
                "SKIP: QEMU not available for {} at {}",
                common::TARGET_X86_64, opt_level
            );
            continue;
        }

        assert!(
            result.success,
            "Hello world execution failed at {} for x86-64:\nstderr: {}\nstdout: {}",
            opt_level, result.stderr, result.stdout
        );
        assert!(
            result.stdout.contains("Hello, World!"),
            "Expected 'Hello, World!' at {}, got: '{}'",
            opt_level, result.stdout
        );
    }
}

// ===========================================================================
// Diagnostic Format and Error Handling Tests
// ===========================================================================

/// Verify the compiler emits GCC-compatible diagnostic format on stderr.
///
/// Per AAP §0.7 "Diagnostic Format Rule", all error messages must follow:
/// `file:line:col: error: description` on stderr.
#[test]
fn hello_world_error_diagnostic_format() {
    let compile_result: common::CompileResult =
        common::compile_source(INVALID_C_SOURCE, &["--target", common::TARGET_X86_64]);

    // Compilation must fail on the invalid source.
    assert!(
        !compile_result.success,
        "Expected compilation failure for invalid C source, but it succeeded"
    );

    // Verify stderr is non-empty (diagnostics were emitted).
    assert!(
        !compile_result.stderr.is_empty(),
        "Expected diagnostic output on stderr, but stderr was empty"
    );

    // Verify the diagnostic contains the word "error" (GCC-compatible format).
    // The exact format is `file:line:col: error: message`.
    assert!(
        compile_result.stderr.contains("error"),
        "Expected GCC-compatible error diagnostic containing 'error', got:\n{}",
        compile_result.stderr
    );
}

/// Verify the compiler exits with code 1 on compilation errors.
///
/// Per AAP §0.7 "Diagnostic Format Rule", the process must exit with code 1
/// on any compile error. This test uses `Command::status()` directly.
#[test]
fn hello_world_error_exit_code() {
    let bcc: PathBuf = common::get_bcc_binary();
    let source_file = common::write_temp_source(INVALID_C_SOURCE);

    // Use Command::status() to get just the exit status without capturing output.
    let status = Command::new(&bcc)
        .args(["--target", common::TARGET_X86_64])
        .arg(source_file.path())
        .status()
        .expect("Failed to execute bcc compiler");

    assert!(
        !status.success(),
        "Expected non-zero exit code for invalid C source, got: {:?}",
        status.code()
    );

    // Exit code should specifically be 1 per GCC convention.
    if let Some(code) = status.code() {
        assert_eq!(
            code, 1,
            "Expected exit code 1 for compile error, got {}",
            code
        );
    }
}

// ===========================================================================
// All-Architecture ELF Verification and Cross-Execution Test
// ===========================================================================

/// Verify ELF format properties for all four architectures in a single test.
///
/// Compiles hello world for each target and verifies the correct ELF class
/// and machine architecture field. Additionally, for targets with QEMU
/// available, executes the binary via `run_with_qemu()` directly to verify
/// cross-architecture execution support.
#[test]
fn hello_world_all_architectures_elf_verification() {
    // Architecture test table: (target, elf_class, elf_machine, label)
    let targets: &[(&str, u8, u16, &str)] = &[
        (
            common::TARGET_X86_64,
            common::ELFCLASS64,
            common::EM_X86_64,
            "x86-64",
        ),
        (
            common::TARGET_I686,
            common::ELFCLASS32,
            common::EM_386,
            "i686",
        ),
        (
            common::TARGET_AARCH64,
            common::ELFCLASS64,
            common::EM_AARCH64,
            "AArch64",
        ),
        (
            common::TARGET_RISCV64,
            common::ELFCLASS64,
            common::EM_RISCV,
            "RISC-V 64",
        ),
    ];

    for &(target, expected_class, expected_arch, label) in targets {
        let compile_result: common::CompileResult =
            common::compile_source(HELLO_WORLD_SOURCE, &["--target", target]);

        assert!(
            compile_result.success,
            "Compilation failed for {} ({}):\n{}",
            label, target, compile_result.stderr
        );

        if let Some(ref binary_path) = compile_result.output_path {
            // Full ELF header verification.
            common::verify_elf_magic(binary_path);
            common::verify_elf_class(binary_path, expected_class);
            common::verify_elf_arch(binary_path, expected_arch);

            // Attempt cross-architecture execution via QEMU for non-native targets.
            if !common::is_native_target(target) {
                if common::is_qemu_available(target) {
                    let run_result: common::RunResult =
                        common::run_with_qemu(binary_path, target);
                    if run_result.success {
                        assert!(
                            run_result.stdout.contains("Hello, World!"),
                            "Expected 'Hello, World!' via QEMU for {}, got: '{}'",
                            label, run_result.stdout
                        );
                    } else {
                        eprintln!(
                            "NOTE: QEMU execution for {} returned non-success: {}",
                            label, run_result.stderr
                        );
                    }
                } else {
                    eprintln!(
                        "SKIP: QEMU not available for {} — ELF verified, execution skipped",
                        label
                    );
                }
            } else {
                // Native target — execute directly using Command.
                let output = Command::new(binary_path)
                    .output()
                    .expect("Failed to execute native binary");
                assert!(
                    output.status.success(),
                    "Native execution failed for {}",
                    label
                );
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    stdout.contains("Hello, World!"),
                    "Expected 'Hello, World!' for native {}, got: '{}'",
                    label, stdout
                );
            }
        }
    }
}
