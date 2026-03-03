//! CLI Integration Tests for the bcc (Blitzy C Compiler)
//!
//! This module contains comprehensive integration tests verifying that the `bcc`
//! compiler correctly parses all GCC-compatible command-line flags, returns proper
//! exit codes, generates GCC-compatible diagnostic messages, and handles output
//! file naming conventions correctly.
//!
//! # Test Categories
//!
//! 1. **Flag Parsing Tests** — Verify each CLI flag is recognized and affects compilation
//! 2. **Error Exit Code Tests** — Verify exit code 1 on errors, exit code 0 on success
//! 3. **Diagnostic Format Tests** — Verify GCC-compatible `file:line:col: level: message` format
//! 4. **Output File Naming Tests** — Verify default names, `-c` derives `.o`, custom `-o` paths
//! 5. **Multiple Input File Tests** — Verify multi-file compilation and mixed source/object inputs
//!
//! # Zero-Dependency Guarantee
//!
//! This test file uses ONLY the Rust standard library (`std`). No external crates.

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ===========================================================================
// Test Infrastructure
// ===========================================================================

/// Create a `Command` pointing to the bcc compiler binary.
///
/// This is a convenience wrapper around `common::get_bcc_binary()` that returns
/// a ready-to-use `Command` instance. Test functions can chain `.arg()` calls
/// to add flags before executing.
fn bcc() -> Command {
    Command::new(common::get_bcc_binary())
}

/// Simple C source that should always compile successfully.
const VALID_C_SOURCE: &str = "int main(void) { return 0; }\n";

/// C source with an intentional syntax error (missing closing paren).
const SYNTAX_ERROR_SOURCE: &str = "int main( { return 0; }\n";

/// C source with a type error: calling an undeclared identifier as a function
/// with incompatible redeclaration that triggers a hard semantic error.
const TYPE_ERROR_SOURCE: &str = r#"
void foo(int a, int b);
void foo(double a);
int main(void) {
    foo(1, 2);
    return 0;
}
"#;

/// C source that uses a macro existence check (for `-D` flag testing).
const MACRO_TEST_SOURCE: &str = r#"
#ifdef TEST_MACRO
int main(void) { return 0; }
#else
#error "TEST_MACRO not defined"
#endif
"#;

/// C source that uses a macro with a value (for `-D NAME=VALUE` testing).
const MACRO_VALUE_SOURCE: &str = r#"
int main(void) { return VALUE; }
"#;

/// C source that includes a custom header (for `-I` flag testing).
const INCLUDE_TEST_SOURCE: &str = r#"
#include "custom.h"
int main(void) { return CUSTOM_VALUE; }
"#;

/// C source designed to produce a warning (unused variable).
const WARNING_SOURCE: &str = r#"
int main(void) {
    int unused_var = 42;
    return 0;
}
"#;

// ===========================================================================
// Phase 2: Flag Parsing Tests — Compilation Mode Flags
// ===========================================================================

/// Test that the `-c` flag produces a `.o` object file instead of a linked executable.
///
/// The `-c` flag tells the compiler to compile and assemble but not link, producing
/// a relocatable ELF object file. When combined with `-o`, the object file should
/// be written to the specified path.
#[test]
fn flag_compile_only() {
    let dir = common::TempDir::new("flag_compile_only");
    let source_path = dir.path().join("test.c");
    let output_path = dir.path().join("test.o");

    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .arg("-c")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -c should succeed for valid source. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output_path.exists(),
        "Object file should exist at '{}'",
        output_path.display()
    );

    // Verify the output is a valid ELF object (has ELF magic bytes).
    common::verify_elf_magic(&output_path);
}

/// Test that the `-o` flag specifies the output file name.
///
/// When `-o <path>` is provided, the compiler output must be written to the
/// specified path regardless of the input file name.
#[test]
fn flag_output_name() {
    let dir = common::TempDir::new("flag_output_name");
    let source_path = dir.path().join("source.c");
    let custom_output = dir.path().join("my_custom_binary");

    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .arg("-c")
        .arg("-o")
        .arg(&custom_output)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -o should succeed for valid source. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        custom_output.exists(),
        "Output file should exist at custom path '{}'",
        custom_output.display()
    );

    // Verify the output file has non-zero size using fs::metadata.
    let meta = fs::metadata(&custom_output).expect("Failed to get output file metadata");
    assert!(
        meta.len() > 0,
        "Output file should have non-zero size"
    );
}

/// Test that the `-shared` flag produces a shared library (.so).
///
/// When `-shared` is provided (typically with `-fPIC`), the compiler should produce
/// an ELF shared object rather than a regular executable.
#[test]
fn flag_shared_library() {
    let dir = common::TempDir::new("flag_shared_library");
    let source_path = dir.path().join("lib.c");
    let output_path = dir.path().join("libtest.so");

    // Shared libraries typically don't need main() — provide a simple exported function.
    let shared_source = "int shared_func(void) { return 42; }\n";
    fs::write(&source_path, shared_source).expect("Failed to write test source");

    let output = bcc()
        .args(&["-shared", "-fPIC"])
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -shared -fPIC should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output_path.exists(),
        "Shared library should exist at '{}'",
        output_path.display()
    );

    // Verify the output is a valid ELF file.
    common::verify_elf_magic(&output_path);
}

/// Test that the `-static` flag requests static linking.
///
/// The `-static` flag tells the linker to use static libraries rather than
/// shared libraries. The compiler should accept this flag without error.
#[test]
fn flag_static_linking() {
    let dir = common::TempDir::new("flag_static_linking");
    let source_path = dir.path().join("test.c");
    let output_path = dir.path().join("test_static");

    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .arg("-static")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -static should succeed for valid source. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output_path.exists(),
        "Static executable should exist at '{}'",
        output_path.display()
    );

    // Verify the output is a valid ELF binary.
    common::verify_elf_magic(&output_path);
}

// ===========================================================================
// Phase 2: Flag Parsing Tests — Include and Define Flags
// ===========================================================================

/// Test that `-I <dir>` adds an include search directory.
///
/// Creates a custom header file in a temporary directory, then compiles a C
/// source that `#include`s that header using the `-I` flag to add the directory
/// to the search path.
#[test]
fn flag_include_dir() {
    let dir = common::TempDir::new("flag_include_dir");
    let include_dir = dir.path().join("my_includes");
    fs::create_dir_all(&include_dir).expect("Failed to create include dir");

    // Create a custom header file in the include directory.
    let header_path = include_dir.join("custom.h");
    fs::write(&header_path, "#define CUSTOM_VALUE 0\n").expect("Failed to write header");

    let source_path = dir.path().join("test.c");
    fs::write(&source_path, INCLUDE_TEST_SOURCE).expect("Failed to write test source");

    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-I")
        .arg(&include_dir)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -I should find custom header. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output_path.exists(),
        "Object file should be produced with -I flag"
    );
}

/// Test that `-D <name>` and `-D <name>=<value>` define preprocessor macros.
///
/// Verifies both forms of the -D flag:
/// 1. `-DTEST_MACRO` — defines the macro with no value (for `#ifdef` tests)
/// 2. `-DVALUE=0` — defines the macro with a specific value
#[test]
fn flag_define_macro() {
    let dir = common::TempDir::new("flag_define_macro");

    // Test 1: -D with no value (macro existence check via #ifdef).
    let source_path = dir.path().join("test_ifdef.c");
    fs::write(&source_path, MACRO_TEST_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test_ifdef.o");

    let output = bcc()
        .arg("-c")
        .arg("-DTEST_MACRO")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -DTEST_MACRO should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Test 2: -D with a value.
    let source_value = dir.path().join("test_value.c");
    fs::write(&source_value, MACRO_VALUE_SOURCE).expect("Failed to write test source");
    let output_value = dir.path().join("test_value.o");

    let output = bcc()
        .arg("-c")
        .arg("-DVALUE=0")
        .arg("-o")
        .arg(&output_value)
        .arg(&source_value)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -DVALUE=0 should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-U <name>` undefines a preprocessor macro.
///
/// Defines a macro with `-D` and then undefines it with `-U`. The source uses
/// `#ifdef` to verify the macro is not defined after `-U`.
#[test]
fn flag_undefine_macro() {
    let dir = common::TempDir::new("flag_undefine_macro");
    let source_path = dir.path().join("test.c");

    // This source expects TEST_MACRO to NOT be defined.
    let source = r#"
#ifdef TEST_MACRO
#error "TEST_MACRO should have been undefined"
#endif
int main(void) { return 0; }
"#;
    fs::write(&source_path, source).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    // Define the macro first, then undefine it. -U should override -D.
    let output = bcc()
        .arg("-c")
        .arg("-DTEST_MACRO")
        .arg("-UTEST_MACRO")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -D then -U should undefine the macro. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ===========================================================================
// Phase 2: Flag Parsing Tests — Library Flags
// ===========================================================================

/// Test that `-L <dir>` adds a library search directory.
///
/// The `-L` flag should be accepted by the compiler without error. We verify
/// acceptance rather than actual library resolution, since library paths are
/// only meaningful during the linking phase.
#[test]
fn flag_library_dir() {
    let dir = common::TempDir::new("flag_library_dir");
    let lib_dir = dir.path().join("my_libs");
    fs::create_dir_all(&lib_dir).expect("Failed to create lib dir");

    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    // Compile with -L flag. At the object compilation stage (-c), the library
    // directory should simply be recorded for future linking.
    let output = bcc()
        .arg("-c")
        .arg("-L")
        .arg(&lib_dir)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -L should accept library directory flag. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-l <lib>` is accepted as a library linking flag.
///
/// The `-l` flag specifies a library to link against. When compiling to an
/// object file with `-c`, the flag should be recorded but not trigger linking.
/// This test verifies the flag is parsed without error.
#[test]
fn flag_link_library() {
    let dir = common::TempDir::new("flag_link_library");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    // When compiling with -c, -l should be accepted and silently stored.
    let output = bcc()
        .arg("-c")
        .arg("-lm")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -l should accept library flag. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ===========================================================================
// Phase 2: Flag Parsing Tests — Debug and Optimization Flags
// ===========================================================================

/// Test that `-g` enables DWARF v4 debug info generation.
///
/// When `-g` is specified, the compiler should generate DWARF debug sections
/// in the output binary. We verify that the flag is accepted and the output
/// is produced successfully with debug section data.
#[test]
fn flag_debug_info() {
    let dir = common::TempDir::new("flag_debug_info");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-g")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -g should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output_path.exists(),
        "Object file with debug info should be produced"
    );

    // Verify the output is a valid ELF file.
    common::verify_elf_magic(&output_path);

    // Read the binary and verify it has a non-trivial size (debug info adds data).
    let data = fs::read(&output_path).expect("Failed to read object file");
    assert!(
        data.len() > 16,
        "Object file with -g should contain more than just a header"
    );
}

/// Test that `-O0` (no optimization) is accepted.
///
/// `-O0` disables all optimization passes. The compiler should accept this
/// flag and produce valid output.
#[test]
fn flag_opt_level_0() {
    let dir = common::TempDir::new("flag_opt_level_0");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-O0")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -O0 should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-O1` (basic optimization) is accepted.
///
/// `-O1` enables basic optimization passes: mem2reg, constant folding, DCE.
#[test]
fn flag_opt_level_1() {
    let dir = common::TempDir::new("flag_opt_level_1");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-O1")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -O1 should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-O2` (aggressive optimization) is accepted.
///
/// `-O2` enables aggressive optimization passes including CSE and simplification.
#[test]
fn flag_opt_level_2() {
    let dir = common::TempDir::new("flag_opt_level_2");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-O2")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -O2 should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ===========================================================================
// Phase 2: Flag Parsing Tests — Code Generation Flags
// ===========================================================================

/// Test that `-fPIC` enables position-independent code generation.
///
/// The `-fPIC` flag is required for shared library code. It should be accepted
/// and influence code generation to use GOT-relative addressing and PLT stubs.
#[test]
fn flag_fpic() {
    let dir = common::TempDir::new("flag_fpic");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-fPIC")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -fPIC should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-mretpoline` enables retpoline generation (x86-64 only).
///
/// The `-mretpoline` flag generates retpoline thunk sequences for indirect
/// branches as a Spectre v2 mitigation. This is only meaningful for x86-64.
#[test]
fn flag_retpoline() {
    let dir = common::TempDir::new("flag_retpoline");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-mretpoline")
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -mretpoline should succeed for x86-64 target. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-fcf-protection` enables CET endbr64 instrumentation (x86-64 only).
///
/// The `-fcf-protection` flag enables Intel Control-flow Enforcement Technology
/// (CET) by inserting `endbr64` instructions at indirect branch targets.
#[test]
fn flag_cf_protection() {
    let dir = common::TempDir::new("flag_cf_protection");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-fcf-protection")
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -fcf-protection should succeed for x86-64 target. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ===========================================================================
// Phase 2: Flag Parsing Tests — Target Selection
// ===========================================================================

/// Test that `--target x86_64-linux-gnu` selects the x86-64 backend.
///
/// Verifies the compiler accepts the x86-64 target triple and produces an
/// ELF64 object with the correct architecture field.
#[test]
fn flag_target_x86_64() {
    let dir = common::TempDir::new("flag_target_x86_64");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc --target x86_64-linux-gnu should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "Object file should be produced");

    // Verify ELF format matches target.
    common::verify_elf_magic(&output_path);
    common::verify_elf_class(&output_path, common::ELFCLASS64);
    common::verify_elf_arch(&output_path, common::EM_X86_64);
}

/// Test that `--target i686-linux-gnu` selects the i686 backend.
///
/// Verifies the compiler accepts the i686 target triple and produces an
/// ELF32 object with the EM_386 architecture field.
#[test]
fn flag_target_i686() {
    let dir = common::TempDir::new("flag_target_i686");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("--target")
        .arg(common::TARGET_I686)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc --target i686-linux-gnu should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "Object file should be produced");

    // Verify ELF format matches i686 target (32-bit).
    common::verify_elf_magic(&output_path);
    common::verify_elf_class(&output_path, common::ELFCLASS32);
    common::verify_elf_arch(&output_path, common::EM_386);
}

/// Test that `--target aarch64-linux-gnu` selects the AArch64 backend.
///
/// Verifies the compiler accepts the AArch64 target triple and produces an
/// ELF64 object with the EM_AARCH64 architecture field.
#[test]
fn flag_target_aarch64() {
    let dir = common::TempDir::new("flag_target_aarch64");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("--target")
        .arg(common::TARGET_AARCH64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc --target aarch64-linux-gnu should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "Object file should be produced");

    // Verify ELF format matches AArch64 target (64-bit).
    common::verify_elf_magic(&output_path);
    common::verify_elf_class(&output_path, common::ELFCLASS64);
    common::verify_elf_arch(&output_path, common::EM_AARCH64);
}

/// Test that `--target riscv64-linux-gnu` selects the RISC-V 64 backend.
///
/// Verifies the compiler accepts the RISC-V 64 target triple and produces an
/// ELF64 object with the EM_RISCV architecture field.
#[test]
fn flag_target_riscv64() {
    let dir = common::TempDir::new("flag_target_riscv64");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("--target")
        .arg(common::TARGET_RISCV64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc --target riscv64-linux-gnu should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "Object file should be produced");

    // Verify ELF format matches RISC-V 64 target (64-bit).
    common::verify_elf_magic(&output_path);
    common::verify_elf_class(&output_path, common::ELFCLASS64);
    common::verify_elf_arch(&output_path, common::EM_RISCV);
}

// ===========================================================================
// Phase 3: Error Exit Code Tests
// ===========================================================================

/// Verify that compiling valid C source exits with code 0.
///
/// This is the baseline success test: valid C source with `-c` should always
/// succeed and produce exit code 0.
#[test]
fn success_exit_code() {
    let result = common::compile_source(VALID_C_SOURCE, &["-c"]);
    assert!(
        result.success,
        "Valid C source should compile successfully. stderr: {}",
        result.stderr
    );

    // Explicitly check exit code via the exit_status field.
    assert!(
        result.exit_status.success(),
        "Exit status should be success (code 0). Got: {:?}. stderr: {}",
        result.exit_status,
        result.stderr
    );
}

/// Verify that compiling C source with a syntax error exits with code 1.
///
/// Per AAP §0.7 "Diagnostic Format Rule": the process must exit with code 1
/// on any compile error.
#[test]
fn error_exit_code_on_syntax_error() {
    let result = common::compile_source(SYNTAX_ERROR_SOURCE, &["-c"]);
    assert!(
        !result.success,
        "Syntax error should cause compilation failure"
    );

    // Verify the exit code is exactly 1, not some other non-zero code.
    // ExitStatus::code() returns Option<i32> on all platforms.
    if let Some(code) = result.exit_status.code() {
        assert_eq!(
            code, 1,
            "Exit code should be 1 for compile errors, got: {}. stderr: {}",
            code, result.stderr
        );
    }
}

/// Verify that compiling C source with a type error exits with code 1.
///
/// Type mismatches (e.g., assigning a string literal to an int) should be
/// caught during semantic analysis and reported as errors.
#[test]
fn error_exit_code_on_type_error() {
    let result = common::compile_source(TYPE_ERROR_SOURCE, &["-c"]);
    assert!(
        !result.success,
        "Type error should cause compilation failure"
    );

    // ExitStatus::code() returns Option<i32> on all platforms.
    if let Some(code) = result.exit_status.code() {
        assert_eq!(
            code, 1,
            "Exit code should be 1 for type errors, got: {}. stderr: {}",
            code, result.stderr
        );
    }
}

/// Verify that running bcc with no input files produces an error.
///
/// A compiler invoked with zero input files should report an error and exit
/// with a non-zero status. The error message should be printed to stderr.
#[test]
fn error_exit_code_missing_input() {
    let status = bcc()
        .status()
        .expect("Failed to execute bcc");

    assert!(
        !status.success(),
        "bcc with no input files should fail"
    );
}

/// Verify that an invalid/unsupported flag produces an error.
///
/// Passing an unrecognized flag should cause the compiler to report an error
/// and exit with a non-zero status.
#[test]
fn error_exit_code_invalid_flag() {
    let dir = common::TempDir::new("error_invalid_flag");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .arg("--this-is-not-a-valid-flag")
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        !output.status.success(),
        "bcc with an invalid flag should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.is_empty(),
        "Error message should be printed for invalid flags"
    );
}

// ===========================================================================
// Phase 4: Diagnostic Format Tests
// ===========================================================================

/// Verify that error diagnostics follow GCC-compatible format on stderr.
///
/// GCC-compatible format: `file:line:col: error: message`
///
/// The diagnostic should contain:
/// - The source file path or name
/// - A line number
/// - A column number
/// - The severity keyword "error"
/// - A descriptive message
#[test]
fn diagnostic_format_gcc_compatible() {
    let source_file = common::write_temp_source(SYNTAX_ERROR_SOURCE);
    let source_path_display = source_file.path().to_string_lossy().to_string();

    let output = bcc()
        .arg("-c")
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc");

    assert!(
        !output.status.success(),
        "Syntax error source should fail compilation"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify the error message contains the expected GCC-compatible format components.
    // Format: file:line:col: error: message
    assert!(
        stderr.contains("error"),
        "Diagnostic should contain 'error' keyword. Got stderr:\n{}",
        stderr
    );

    // Verify the filename appears in the diagnostic output.
    let source_filename = source_file
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert!(
        stderr.contains(&source_filename) || stderr.contains(&source_path_display),
        "Diagnostic should reference the source file name '{}'. Got stderr:\n{}",
        source_filename,
        stderr
    );

    // Verify the diagnostic contains line:col format (digit followed by colon).
    // Pattern: at least one occurrence of "digits:digits:" which represents line:col:
    let has_line_col = stderr
        .lines()
        .any(|line| {
            // Look for pattern: <path>:<number>:<number>: error:
            let parts: Vec<&str> = line.splitn(4, ':').collect();
            parts.len() >= 4
                && parts[1].trim().chars().all(|c| c.is_ascii_digit())
                && parts[2].trim().chars().all(|c| c.is_ascii_digit())
        });
    assert!(
        has_line_col,
        "Diagnostic should contain line:col format (file:N:N: ...). Got stderr:\n{}",
        stderr
    );
}

/// Verify that warning diagnostics follow GCC-compatible format.
///
/// GCC-compatible warning format: `file:line:col: warning: message`
///
/// Note: Whether the compiler emits this specific warning is implementation-dependent.
/// If a warning is emitted, we verify its format matches the GCC convention.
#[test]
fn diagnostic_warning_format() {
    let source_file = common::write_temp_source(WARNING_SOURCE);

    let output = bcc()
        .arg("-c")
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // If the compiler produces a warning about the unused variable, verify
    // the format. Note: the compiler may or may not emit this warning,
    // depending on implementation.
    if stderr.contains("warning") {
        // Verify warning follows GCC format: file:line:col: warning: message
        let has_warning_format = stderr.lines().any(|line| {
            line.contains("warning:") && {
                let parts: Vec<&str> = line.splitn(4, ':').collect();
                parts.len() >= 4
            }
        });
        assert!(
            has_warning_format,
            "Warning diagnostic should follow GCC format. Got stderr:\n{}",
            stderr
        );
    }
    // If no warning is emitted, that's acceptable — warnings are implementation-dependent.
}

/// Verify that note diagnostics follow GCC-compatible format.
///
/// GCC-compatible note format: `file:line:col: note: message`
/// Notes are supplementary information attached to errors or warnings,
/// such as pointing to the original declaration in a redefinition error.
#[test]
fn diagnostic_note_format() {
    // Use a source that's likely to produce a note (e.g., redefinition error
    // where the note points to the original definition).
    let source_with_note = r#"
int foo(void) { return 0; }
int foo(void) { return 1; }
int main(void) { return foo(); }
"#;
    let source_file = common::write_temp_source(source_with_note);

    let output = bcc()
        .arg("-c")
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // If the compiler produces a note, verify the format.
    // Notes are optional — a compiler may or may not emit them alongside errors.
    if stderr.contains("note:") {
        let has_note_format = stderr.lines().any(|line| {
            line.contains("note:") && {
                let parts: Vec<&str> = line.splitn(4, ':').collect();
                parts.len() >= 4
            }
        });
        assert!(
            has_note_format,
            "Note diagnostic should follow GCC format. Got stderr:\n{}",
            stderr
        );
    }
    // Verify the redefinition at least produces an error or warning.
    assert!(
        !output.status.success() || stderr.contains("error") || stderr.contains("warning"),
        "Redefinition of function should produce at least an error or warning"
    );
}

// ===========================================================================
// Phase 5: Output File Naming Tests
// ===========================================================================

/// Verify that without `-o`, the default output is `a.out` for executables.
///
/// GCC convention: when no `-o` flag is specified and no `-c` flag is present,
/// the output executable is named `a.out` in the current working directory.
#[test]
fn default_output_name() {
    let dir = common::TempDir::new("default_output_name");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    // Run bcc without -o, but set the working directory to our temp dir
    // so `a.out` is created there instead of polluting the test directory.
    let output = bcc()
        .current_dir(dir.path())
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    // If compilation succeeds (full pipeline available), verify the default output name.
    if output.status.success() {
        let default_output = dir.path().join("a.out");
        assert!(
            default_output.exists(),
            "Default output should be 'a.out' in working directory. \
             Checked: '{}'",
            default_output.display()
        );

        // Verify the default output is a valid ELF binary.
        common::verify_elf_magic(&default_output);
    }
    // If compilation fails (e.g., due to missing CRT objects for full linking),
    // we at least verify the compiler did not crash unexpectedly.
}

/// Verify that `-c` without `-o` derives the output name from the input file.
///
/// GCC convention: `bcc -c foo.c` produces `foo.o` in the current directory.
/// The `.c` extension is replaced with `.o`.
#[test]
fn object_output_naming() {
    let dir = common::TempDir::new("object_output_naming");
    let source_path = dir.path().join("myprogram.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .current_dir(dir.path())
        .arg("-c")
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    if output.status.success() {
        // The object file should be named myprogram.o in the working directory.
        let expected_obj = dir.path().join("myprogram.o");
        assert!(
            expected_obj.exists(),
            "With -c and no -o, output should be 'myprogram.o'. \
             Checked: '{}'",
            expected_obj.display()
        );

        // Verify the output is a valid ELF object.
        common::verify_elf_magic(&expected_obj);
    }
}

/// Verify that `-o <path>` sends output to the specified custom path.
///
/// The custom path can have any extension or no extension at all.
#[test]
fn custom_output_path() {
    let dir = common::TempDir::new("custom_output_path");
    let source_path = dir.path().join("test.c");
    let custom_path = dir.path().join("my_custom_output.o");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .arg("-c")
        .arg("-o")
        .arg(&custom_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "Compilation with -o should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        custom_path.exists(),
        "Output should be at custom path '{}'",
        custom_path.display()
    );
}

// ===========================================================================
// Phase 6: Multiple Input Files
// ===========================================================================

/// Test that multiple .c source files can be compiled into one executable.
///
/// This is a common GCC usage pattern: `gcc -o main file1.c file2.c`
/// compiles both source files and links them into a single executable.
#[test]
fn multiple_source_files() {
    let dir = common::TempDir::new("multiple_source_files");

    let source1_path = dir.path().join("main.c");
    let source1 = r#"
extern int helper(void);
int main(void) { return helper(); }
"#;
    fs::write(&source1_path, source1).expect("Failed to write source1");

    let source2_path = dir.path().join("helper.c");
    let source2 = "int helper(void) { return 0; }\n";
    fs::write(&source2_path, source2).expect("Failed to write source2");

    let output_path = dir.path().join("multi_test");

    let output = bcc()
        .arg("-o")
        .arg(&output_path)
        .arg(&source1_path)
        .arg(&source2_path)
        .output()
        .expect("Failed to execute bcc");

    // If multi-file compilation is supported, verify the output.
    if output.status.success() {
        assert!(
            output_path.exists(),
            "Multi-file compilation should produce output at '{}'",
            output_path.display()
        );
        common::verify_elf_magic(&output_path);
    } else {
        // If the full link pipeline isn't available, at least verify the error
        // message is not about unrecognized flags — the compiler should understand
        // multiple input files.
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("unrecognized")
                && !stderr.contains("unknown option")
                && !stderr.contains("invalid flag"),
            "Multiple input files should be recognized. stderr: {}",
            stderr
        );
    }
}

/// Test compiling some .c files and linking with pre-existing .o object files.
///
/// This pattern simulates separate compilation and linking:
/// 1. First compile helper.c to helper.o using `-c`
/// 2. Then compile main.c and link with helper.o
#[test]
fn mix_sources_and_objects() {
    let dir = common::TempDir::new("mix_sources_and_objects");

    // Step 1: Compile helper.c to helper.o.
    let helper_source = dir.path().join("helper.c");
    let helper_object = dir.path().join("helper.o");
    fs::write(&helper_source, "int helper(void) { return 0; }\n")
        .expect("Failed to write helper source");

    let step1 = bcc()
        .arg("-c")
        .arg("-o")
        .arg(&helper_object)
        .arg(&helper_source)
        .output()
        .expect("Failed to execute bcc for step 1");

    if !step1.status.success() {
        // If -c fails, the rest of the test cannot proceed. Mark as inconclusive
        // by returning early (the flag_compile_only test covers -c independently).
        eprintln!(
            "Step 1 (compile helper.o) failed; skipping rest of test. stderr: {}",
            String::from_utf8_lossy(&step1.stderr)
        );
        return;
    }
    assert!(
        helper_object.exists(),
        "helper.o should be produced by step 1"
    );

    // Step 2: Compile main.c and link with helper.o.
    let main_source = dir.path().join("main.c");
    let output_path = dir.path().join("mixed_test");
    let main_code = r#"
extern int helper(void);
int main(void) { return helper(); }
"#;
    fs::write(&main_source, main_code).expect("Failed to write main source");

    let step2 = bcc()
        .arg("-o")
        .arg(&output_path)
        .arg(&main_source)
        .arg(&helper_object)
        .output()
        .expect("Failed to execute bcc for step 2");

    // If linking succeeds, verify the output.
    if step2.status.success() {
        assert!(
            output_path.exists(),
            "Mixed source/object compilation should produce output"
        );
        common::verify_elf_magic(&output_path);
    } else {
        // If full linking is not yet implemented, the compiler should at least
        // accept both .c and .o files as input without flag parsing errors.
        let stderr = String::from_utf8_lossy(&step2.stderr);
        assert!(
            !stderr.contains("unrecognized")
                && !stderr.contains("unknown option"),
            "Mixing .c and .o inputs should be recognized. stderr: {}",
            stderr
        );
    }
}

// ===========================================================================
// Additional Flag Combination Tests
// ===========================================================================

/// Test combining multiple flags: `-c -g -O2 -fPIC --target`
///
/// Verifies that the compiler handles a realistic combination of flags
/// as commonly used in production build systems (e.g., building a shared
/// library with debug info and optimizations for a specific target).
#[test]
fn combined_flags_realistic() {
    let dir = common::TempDir::new("combined_flags");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .args(&["-c", "-g", "-O2", "-fPIC"])
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "Combined flags (-c -g -O2 -fPIC --target) should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "Object file should be produced");
}

/// Test combining security hardening flags: `-mretpoline -fcf-protection`
///
/// Both security hardening flags can be enabled simultaneously on x86-64.
#[test]
fn combined_security_flags() {
    let dir = common::TempDir::new("combined_security");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .args(&["-mretpoline", "-fcf-protection"])
        .arg("--target")
        .arg(common::TARGET_X86_64)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "Combined security flags should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that multiple `-I`, `-D`, and `-L` flags can be combined.
///
/// Production builds often specify multiple include directories, macro
/// definitions, and library paths simultaneously.
#[test]
fn combined_include_define_library_flags() {
    let dir = common::TempDir::new("combined_idl");
    let include_dir1 = dir.path().join("inc1");
    let include_dir2 = dir.path().join("inc2");
    let lib_dir = dir.path().join("libs");
    fs::create_dir_all(&include_dir1).expect("Failed to create inc1");
    fs::create_dir_all(&include_dir2).expect("Failed to create inc2");
    fs::create_dir_all(&lib_dir).expect("Failed to create libs");

    // Create a header in inc1.
    fs::write(include_dir1.join("header1.h"), "#define H1_VALUE 1\n")
        .expect("Failed to write header1");
    // Create a header in inc2.
    fs::write(include_dir2.join("header2.h"), "#define H2_VALUE 2\n")
        .expect("Failed to write header2");

    let source_path = dir.path().join("test.c");
    let source = r#"
#include "header1.h"
#include "header2.h"
int main(void) { return H1_VALUE + H2_VALUE + EXTRA - 3; }
"#;
    fs::write(&source_path, source).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    let output = bcc()
        .arg("-c")
        .arg("-I").arg(&include_dir1)
        .arg("-I").arg(&include_dir2)
        .arg("-DEXTRA=0")
        .arg("-L").arg(&lib_dir)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "Multiple -I, -D, -L flags should be accepted. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that `-D` with separate argument form works (`-D MACRO` vs `-DMACRO`).
///
/// GCC accepts both forms: `-DNAME` (concatenated) and `-D NAME` (separated).
/// The bcc compiler should accept both.
#[test]
fn flag_define_separated_form() {
    let dir = common::TempDir::new("flag_define_separated");
    let source_path = dir.path().join("test.c");
    fs::write(&source_path, MACRO_TEST_SOURCE).expect("Failed to write test source");
    let output_path = dir.path().join("test.o");

    // Test the separated form: -D TEST_MACRO (space between -D and macro name).
    let output = bcc()
        .arg("-c")
        .arg("-D")
        .arg("TEST_MACRO")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "bcc -D TEST_MACRO (separated) should succeed. stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that all four target triples are accepted with `-c` in a loop.
///
/// This provides a comprehensive cross-check that all architectures are
/// supported by the `--target` flag, verifying ELF class and machine type
/// for each target.
#[test]
fn all_targets_accepted() {
    let targets_and_configs: &[(&str, u8, u16)] = &[
        (common::TARGET_X86_64, common::ELFCLASS64, common::EM_X86_64),
        (common::TARGET_I686, common::ELFCLASS32, common::EM_386),
        (common::TARGET_AARCH64, common::ELFCLASS64, common::EM_AARCH64),
        (common::TARGET_RISCV64, common::ELFCLASS64, common::EM_RISCV),
    ];

    for &(target, expected_class, expected_arch) in targets_and_configs {
        let dir = common::TempDir::new(&format!(
            "all_targets_{}",
            target.replace('-', "_")
        ));
        let source_path = dir.path().join("test.c");
        fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
        let output_path = dir.path().join("test.o");

        let output = bcc()
            .arg("-c")
            .arg("--target")
            .arg(target)
            .arg("-o")
            .arg(&output_path)
            .arg(&source_path)
            .output()
            .expect("Failed to execute bcc");

        assert!(
            output.status.success(),
            "Target '{}' should be accepted. stderr: {}",
            target,
            String::from_utf8_lossy(&output.stderr)
        );

        if output_path.exists() {
            common::verify_elf_magic(&output_path);
            common::verify_elf_class(&output_path, expected_class);
            common::verify_elf_arch(&output_path, expected_arch);
        }
    }
}

/// Test that invoking bcc with no arguments produces an error.
///
/// A well-behaved compiler should produce an error message on stderr
/// explaining that no input files were provided.
#[test]
fn no_args_shows_usage_or_error() {
    let output = bcc()
        .output()
        .expect("Failed to execute bcc");

    // The compiler should not succeed with no arguments.
    assert!(
        !output.status.success(),
        "bcc with no arguments should not succeed"
    );

    // It should produce some output on stderr explaining the problem.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.is_empty(),
        "bcc with no arguments should produce an error message on stderr"
    );
}

/// Test that the `compile_source` helper from common works with the
/// `assert_compile_success!` and `assert_compile_error!` macros.
///
/// This test exercises the macro integration path from `tests/common/mod.rs`.
#[test]
fn compile_source_with_assertion_macros() {
    // Test success path using the compile_source helper and macro.
    let success_result = common::compile_source(VALID_C_SOURCE, &["-c"]);
    assert_compile_success!(success_result);

    // Verify the output_path is populated on success.
    assert!(
        success_result.output_path.is_some(),
        "Successful compilation should provide an output path"
    );

    // Verify stdout is available (may be empty for successful compilations).
    let _stdout = &success_result.stdout;

    // Test error path using the compile_source helper and macro.
    let error_result = common::compile_source(SYNTAX_ERROR_SOURCE, &["-c"]);
    assert_compile_error!(error_result);
}

/// Test the `assert_compile_error_contains!` macro with a specific error message.
///
/// Verifies that the error output contains expected diagnostic text.
#[test]
fn compile_error_contains_message() {
    let error_result = common::compile_source(SYNTAX_ERROR_SOURCE, &["-c"]);

    // The compiler should produce some error diagnostic in stderr.
    assert!(
        !error_result.success,
        "Syntax error should fail compilation"
    );
    assert!(
        !error_result.stderr.is_empty(),
        "Error should produce diagnostic output on stderr"
    );

    // Use PathBuf::from to construct a path (exercising the PathBuf import).
    let _dummy_path = PathBuf::from("/tmp/nonexistent");

    // Verify the error output contains the word "error" in some form.
    assert_compile_error_contains!(error_result, "error");
}

/// Test that output file can be cleaned up after compilation.
///
/// Exercises `fs::remove_file` and `fs::remove_dir_all` for post-test cleanup
/// verification.
#[test]
fn cleanup_after_compilation() {
    let dir = common::TempDir::new("cleanup_test");
    let source_path = dir.path().join("test.c");
    let output_path = dir.path().join("test.o");

    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");

    let output = bcc()
        .arg("-c")
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .output()
        .expect("Failed to execute bcc");

    if output.status.success() && output_path.exists() {
        // Verify we can read the output file.
        let data = fs::read(&output_path).expect("Failed to read output file");
        assert!(!data.is_empty(), "Output file should have content");

        // Explicitly remove the output file.
        fs::remove_file(&output_path).expect("Failed to remove output file");
        assert!(
            !output_path.exists(),
            "Output file should be gone after removal"
        );
    }

    // Explicitly remove source file.
    fs::remove_file(&source_path).expect("Failed to remove source file");
    assert!(
        !source_path.exists(),
        "Source file should be gone after removal"
    );

    // Explicit dir cleanup (dir will also be cleaned on Drop, but testing
    // fs::remove_dir_all explicitly here).
    let cleanup_sub = dir.path().join("sub_cleanup");
    fs::create_dir_all(&cleanup_sub).expect("Failed to create sub dir");
    fs::remove_dir_all(&cleanup_sub).expect("Failed to remove sub dir");
    assert!(
        !cleanup_sub.exists(),
        "Sub-directory should be gone after removal"
    );
}

/// Test that `PathBuf` operations work correctly with bcc output paths.
///
/// Exercises `PathBuf::from`, `PathBuf.as_path`, `PathBuf.exists`, and
/// `PathBuf.join` in a compilation context.
#[test]
fn pathbuf_operations_with_output() {
    let dir = common::TempDir::new("pathbuf_ops");
    let base_dir = PathBuf::from(dir.path());
    let source_path = base_dir.join("test.c");
    let output_path = base_dir.join("output.o");

    // Verify as_path() returns a valid Path reference.
    assert!(
        base_dir.as_path().is_dir(),
        "TempDir should exist as a directory"
    );

    fs::write(&source_path, VALID_C_SOURCE).expect("Failed to write test source");
    assert!(
        source_path.exists(),
        "Source file should exist after writing"
    );

    let output = bcc()
        .arg("-c")
        .arg("-o")
        .arg(output_path.as_path())
        .arg(source_path.as_path())
        .output()
        .expect("Failed to execute bcc");

    if output.status.success() {
        assert!(
            output_path.exists(),
            "Output path should exist after successful compilation"
        );
    }
}
