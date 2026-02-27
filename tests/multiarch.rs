//! Cross-Architecture Integration Tests via QEMU
//!
//! This module verifies that the `bcc` compiler can produce correct executables
//! for all four supported architectures:
//!
//! | Target Triple           | ELF Class | Machine Type   |
//! |-------------------------|-----------|----------------|
//! | `x86_64-linux-gnu`      | ELF64     | EM_X86_64      |
//! | `i686-linux-gnu`        | ELF32     | EM_386         |
//! | `aarch64-linux-gnu`     | ELF64     | EM_AARCH64     |
//! | `riscv64-linux-gnu`     | ELF64     | EM_RISCV       |
//!
//! Non-native architectures are tested via QEMU user-mode emulation. Tests that
//! require QEMU will gracefully skip (with a diagnostic message on stderr) when
//! the emulator is not installed, rather than failing.
//!
//! # Test Categories
//!
//! - **Cross-Compilation** — Compile and run programs on all four targets
//! - **ABI Compliance** — Verify calling conventions per architecture
//! - **ELF Format** — Validate magic, class, data encoding, architecture fields
//! - **Output Modes** — Static executable, relocatable object, shared library
//! - **Target Selection** — `--target` flag parsing, invalid targets, default target

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

// ===========================================================================
// Constants
// ===========================================================================

/// All four supported target triples for cross-compilation testing.
const TARGETS: &[&str] = &[
    "x86_64-linux-gnu",
    "i686-linux-gnu",
    "aarch64-linux-gnu",
    "riscv64-linux-gnu",
];

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Execute a compiled binary on its target architecture.
///
/// If the target matches the host architecture, the binary is executed directly.
/// Otherwise, QEMU user-mode emulation is used via `common::run_with_qemu`.
/// If QEMU is not available for the target, returns a failure result with a
/// descriptive skip message in the stderr component.
///
/// # Arguments
///
/// * `binary` - Path to the compiled ELF binary.
/// * `target` - Target triple the binary was compiled for.
///
/// # Returns
///
/// A tuple of `(ExitStatus, stdout, stderr)` from the execution.
fn run_on_target(binary: &Path, target: &str) -> (ExitStatus, String, String) {
    if common::is_native_target(target) {
        // Execute natively — no QEMU needed.
        let output = Command::new(binary)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to execute native binary '{}': {}",
                    binary.display(),
                    e
                )
            });
        (
            output.status,
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    } else {
        // Delegate to the common module's QEMU execution helper.
        let result = common::run_with_qemu(binary, target);
        (result.exit_status, result.stdout, result.stderr)
    }
}

/// Returns the expected ELF class for a given target triple.
///
/// i686 targets produce ELF32; all other supported targets produce ELF64.
fn expected_elf_class(target: &str) -> u8 {
    if target.starts_with("i686") || target.starts_with("i386") {
        common::ELFCLASS32
    } else {
        common::ELFCLASS64
    }
}

/// Returns the expected ELF machine architecture constant for a target triple.
fn expected_elf_arch(target: &str) -> u16 {
    if target.starts_with("x86_64") {
        common::EM_X86_64
    } else if target.starts_with("i686") || target.starts_with("i386") {
        common::EM_386
    } else if target.starts_with("aarch64") {
        common::EM_AARCH64
    } else if target.starts_with("riscv64") {
        common::EM_RISCV
    } else {
        panic!("Unknown target triple for ELF arch mapping: {}", target);
    }
}

/// Verify that an ELF binary uses little-endian (ELFDATA2LSB) data encoding.
///
/// The ELF data encoding byte is at offset 5 (`EI_DATA`).
/// Value `1` = `ELFDATA2LSB` (little-endian).
fn verify_elf_data_encoding_le(binary: &Path) {
    let data = fs::read(binary).unwrap_or_else(|e| {
        panic!(
            "Failed to read binary '{}' for data encoding check: {}",
            binary.display(),
            e
        );
    });
    assert!(
        data.len() >= 6,
        "Binary '{}' is too small ({} bytes) to contain EI_DATA field",
        binary.display(),
        data.len()
    );
    assert_eq!(
        data[5], 1,
        "Expected ELFDATA2LSB (1) for little-endian in '{}', got {}",
        binary.display(),
        data[5]
    );
}

/// Check whether we can execute binaries for the given target.
///
/// Returns `true` if the target is native to the host, or if QEMU user-mode
/// emulation is available for this target.
fn can_run_target(target: &str) -> bool {
    common::is_native_target(target) || common::is_qemu_available(target)
}

/// Verify that an ELF binary file has a reasonable size (non-zero, within limits).
///
/// Uses `fs::metadata()` to inspect file properties without reading the entire binary.
fn verify_binary_exists_and_nontrivial(path: &Path) {
    let meta = fs::metadata(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read metadata for '{}': {}",
            path.display(),
            e
        );
    });
    assert!(
        meta.len() > 0,
        "Binary '{}' has zero size",
        path.display()
    );
    // A minimal ELF header is at least 52 bytes (ELF32) or 64 bytes (ELF64).
    assert!(
        meta.len() >= 52,
        "Binary '{}' is too small ({} bytes) to be a valid ELF file",
        path.display(),
        meta.len()
    );
}

// ===========================================================================
// Phase 2: Cross-Compilation Tests
// ===========================================================================

/// Verify that the compiler can produce valid ELF binaries for all four architectures.
///
/// For each target, this test:
/// 1. Compiles a minimal C program with `--target`
/// 2. Verifies the output file exists and has a reasonable size
/// 3. Checks ELF magic bytes, class (32/64), data encoding, and architecture field
#[test]
fn cross_compile_all_targets() {
    let source = r#"
int main(void) {
    return 0;
}
"#;

    for &target in TARGETS {
        let result = common::compile_source(source, &["--target", target]);
        assert!(
            result.success,
            "Compilation failed for target '{}': stdout={}\nstderr={}",
            target, result.stdout, result.stderr
        );

        let output = result
            .output_path
            .as_ref()
            .unwrap_or_else(|| panic!("No output binary produced for target '{}'", target));

        // Verify file exists and has a reasonable size.
        verify_binary_exists_and_nontrivial(output);

        // Verify all ELF format properties.
        common::verify_elf_magic(output);
        common::verify_elf_class(output, expected_elf_class(target));
        common::verify_elf_arch(output, expected_elf_arch(target));
        verify_elf_data_encoding_le(output);
    }
}

/// Test integer arithmetic correctness across all architectures.
///
/// Exercises `int`, `long`, and `long long` arithmetic to catch width and
/// sign-extension issues between 32-bit and 64-bit targets. On i686, `long`
/// is 4 bytes; on all 64-bit targets, `long` is 8 bytes.
#[test]
fn cross_compile_integer_arithmetic() {
    let source = r#"
int main(void) {
    /* Basic int arithmetic */
    int a = 100;
    int b = 42;
    int sum = a + b;
    if (sum != 142) return 1;

    /* Subtraction */
    int diff = a - b;
    if (diff != 58) return 2;

    /* Multiplication */
    int prod = a * b;
    if (prod != 4200) return 3;

    /* Division */
    int quot = a / b;
    if (quot != 2) return 4;

    /* Modulus */
    int rem = a % b;
    if (rem != 16) return 5;

    /* long arithmetic (width varies by target) */
    long la = 1000000L;
    long lb = 999999L;
    long lsum = la + lb;
    if (lsum != 1999999L) return 6;

    /* long long arithmetic (always 8 bytes on all targets) */
    long long lla = 4294967296LL; /* 2^32 */
    long long llb = 1LL;
    long long llsum = lla + llb;
    if (llsum != 4294967297LL) return 7;

    /* Negative values */
    int neg = -42;
    if (neg + 42 != 0) return 8;

    return 0;
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_integer_arithmetic for '{}' \u{2014} QEMU not available",
                target
            );
            continue;
        }

        let result = common::compile_and_run(source, target, &[]);
        assert!(
            result.success,
            "Integer arithmetic test failed on '{}': exit={:?}\nstdout: {}\nstderr: {}",
            target,
            result.exit_status.code(),
            result.stdout,
            result.stderr
        );
    }
}

/// Verify that `sizeof(void*)` returns the correct pointer width on each architecture.
///
/// i686 should report 4-byte pointers; x86-64, AArch64, and RISC-V 64 should
/// report 8-byte pointers. The pointer size is encoded as the process exit code
/// for easy verification.
#[test]
fn cross_compile_pointer_sizes() {
    let source = r#"
int main(void) {
    return (int)(unsigned long)sizeof(void*);
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_pointer_sizes for '{}' -- QEMU not available",
                target
            );
            continue;
        }

        let compile_result = common::compile_source(source, &["--target", target]);
        assert!(
            compile_result.success,
            "Compilation failed for pointer sizes test on '{}': {}",
            target, compile_result.stderr
        );

        let binary = compile_result
            .output_path
            .as_ref()
            .unwrap_or_else(|| {
                panic!("No output binary for pointer sizes test on '{}'", target)
            });

        let (status, _stdout, stderr) = run_on_target(binary, target);
        let exit_code = status.code().unwrap_or(-1);

        let expected_ptr_size: i32 = if target.starts_with("i686") { 4 } else { 8 };

        assert_eq!(
            exit_code, expected_ptr_size,
            "sizeof(void*) mismatch for target '{}': expected {}, got {}\nstderr: {}",
            target, expected_ptr_size, exit_code, stderr
        );
    }
}

/// Test struct member alignment and padding across architectures.
///
/// Verifies that struct layout follows the target's alignment rules.
#[test]
fn cross_compile_struct_layout() {
    let source = r#"
struct Padded {
    char a;      /* offset 0 */
    int b;       /* offset 4 (after 3 bytes padding to align int) */
    char c;      /* offset 8 */
    /* total: 12 bytes (with 3 bytes trailing padding) */
};

struct Packed {
    char x;
    char y;
    char z;
    /* total: 3 bytes */
};

int main(void) {
    if (sizeof(struct Padded) != 12) return 1;
    if (sizeof(struct Packed) != 3) return 2;

    struct Padded p;
    p.a = 1;
    p.b = 42;
    p.c = 2;
    if (p.a != 1) return 3;
    if (p.b != 42) return 4;
    if (p.c != 2) return 5;

    return 0;
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_struct_layout for '{}' -- QEMU not available",
                target
            );
            continue;
        }

        let result = common::compile_and_run(source, target, &[]);
        assert!(
            result.success,
            "Struct layout test failed on '{}': exit={:?}\nstdout: {}\nstderr: {}",
            target,
            result.exit_status.code(),
            result.stdout,
            result.stderr
        );
    }
}

/// Test function calling conventions across all four architectures.
///
/// Exercises multiple argument counts and recursion to verify register-based
/// and stack-based argument passing, plus callee-saved register preservation.
#[test]
fn cross_compile_function_calls() {
    let source = r#"
int add_six(int a, int b, int c, int d, int e, int f) {
    return a + b + c + d + e + f;
}

int add_eight(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a + b + c + d + e + f + g + h;
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main(void) {
    if (add_six(1, 2, 3, 4, 5, 6) != 21) return 1;
    if (add_eight(1, 2, 3, 4, 5, 6, 7, 8) != 36) return 2;
    if (fibonacci(10) != 55) return 3;

    int r = add_six(-1, -2, -3, -4, -5, -6);
    if (r != -21) return 4;

    return 0;
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_function_calls for '{}' -- QEMU not available",
                target
            );
            continue;
        }

        let result = common::compile_and_run(source, target, &[]);
        assert!(
            result.success,
            "Function call test failed on '{}': exit={:?}\nstdout: {}\nstderr: {}",
            target,
            result.exit_status.code(),
            result.stdout,
            result.stderr
        );
    }
}

// ===========================================================================
// Phase 3: ABI Compliance Tests
// ===========================================================================

/// Verify System V AMD64 ABI compliance for x86-64.
///
/// SysV AMD64 ABI: first 6 integer args in rdi, rsi, rdx, rcx, r8, r9;
/// return value in rax; 7th+ args on stack; 16-byte stack alignment.
#[test]
fn abi_x86_64_sysv() {
    let target = common::TARGET_X86_64;

    let source = r#"
/* Args: rdi=1, rsi=2, rdx=3, rcx=4, r8=5, r9=6, stack=7, stack=8 */
int many_args(int a, int b, int c, int d, int e, int f, int g, int h) {
    if (a != 1) return 10;
    if (b != 2) return 11;
    if (c != 3) return 12;
    if (d != 4) return 13;
    if (e != 5) return 14;
    if (f != 6) return 15;
    if (g != 7) return 16;
    if (h != 8) return 17;
    return 0;
}

long long return_large(void) {
    return 0x123456789ABCDEF0LL;
}

int main(void) {
    int r = many_args(1, 2, 3, 4, 5, 6, 7, 8);
    if (r != 0) return r;

    long long v = return_large();
    if (v != 0x123456789ABCDEF0LL) return 20;

    return 0;
}
"#;

    if !can_run_target(target) {
        eprintln!("SKIP: abi_x86_64_sysv -- cannot run x86_64 binaries");
        return;
    }

    let result = common::compile_and_run(source, target, &[]);
    assert!(
        result.success,
        "x86_64 SysV ABI test failed: exit={:?}\nstderr: {}",
        result.exit_status.code(),
        result.stderr
    );
}

/// Verify cdecl ABI compliance for i686.
///
/// cdecl ABI: all args on stack (right-to-left push), caller cleanup,
/// eax return (eax:edx for 64-bit).
#[test]
fn abi_i686_cdecl() {
    let target = common::TARGET_I686;

    let source = r#"
int sum_args(int a, int b, int c, int d) {
    return a + b + c + d;
}

long long return_64bit(void) {
    return 0xDEADBEEFCAFEBABELL;
}

int main(void) {
    int s = sum_args(10, 20, 30, 40);
    if (s != 100) return 1;

    long long val = return_64bit();
    if (val != (long long)0xDEADBEEFCAFEBABELL) return 2;

    /* Sequential calls verify caller cleanup correctness */
    int s1 = sum_args(1, 1, 1, 1);
    int s2 = sum_args(2, 2, 2, 2);
    if (s1 + s2 != 12) return 3;

    return 0;
}
"#;

    if !can_run_target(target) {
        eprintln!("SKIP: abi_i686_cdecl -- cannot run i686 binaries");
        return;
    }

    let result = common::compile_and_run(source, target, &[]);
    assert!(
        result.success,
        "i686 cdecl ABI test failed: exit={:?}\nstderr: {}",
        result.exit_status.code(),
        result.stderr
    );
}

/// Verify AAPCS64 ABI compliance for AArch64.
///
/// AAPCS64: integer args in x0-x7, return in x0, SP 16-byte aligned.
#[test]
fn abi_aarch64_aapcs64() {
    let target = common::TARGET_AARCH64;

    let source = r#"
long compute(long a, long b, long c, long d, long e, long f, long g, long h) {
    return a + b + c + d + e + f + g + h;
}

long return_val(void) {
    return 0x0123456789ABCDEFL;
}

int main(void) {
    long s = compute(1, 2, 3, 4, 5, 6, 7, 8);
    if (s != 36) return 1;

    long v = return_val();
    if (v != 0x0123456789ABCDEFL) return 2;

    long big = compute(10, 20, 30, 40, 50, 60, 70, 80);
    if (big != 360) return 3;

    return 0;
}
"#;

    if !can_run_target(target) {
        eprintln!("SKIP: abi_aarch64_aapcs64 -- cannot run aarch64 binaries");
        return;
    }

    let result = common::compile_and_run(source, target, &[]);
    assert!(
        result.success,
        "AArch64 AAPCS64 ABI test failed: exit={:?}\nstderr: {}",
        result.exit_status.code(),
        result.stderr
    );
}

/// Verify LP64D ABI compliance for RISC-V 64.
///
/// LP64D: integer args in a0-a7, return in a0.
#[test]
fn abi_riscv64_lp64d() {
    let target = common::TARGET_RISCV64;

    let source = r#"
long compute(long a, long b, long c, long d, long e, long f, long g, long h) {
    return a + b + c + d + e + f + g + h;
}

long return_val(void) {
    return (long)0xFEDCBA9876543210UL;
}

int main(void) {
    long s = compute(1, 2, 3, 4, 5, 6, 7, 8);
    if (s != 36) return 1;

    long v = return_val();
    if (v != (long)0xFEDCBA9876543210UL) return 2;

    return 0;
}
"#;

    if !can_run_target(target) {
        eprintln!("SKIP: abi_riscv64_lp64d -- cannot run riscv64 binaries");
        return;
    }

    let result = common::compile_and_run(source, target, &[]);
    assert!(
        result.success,
        "RISC-V 64 LP64D ABI test failed: exit={:?}\nstderr: {}",
        result.exit_status.code(),
        result.stderr
    );
}

// ===========================================================================
// Phase 4: ELF Format Verification Tests
// ===========================================================================

/// Verify ELF64 little-endian format with EM_X86_64 architecture.
#[test]
fn elf64_x86_64() {
    let source = "int main(void) { return 0; }\n";
    let result = common::compile_source(source, &["--target", common::TARGET_X86_64]);
    assert!(
        result.success,
        "Failed to compile for x86_64: {}",
        result.stderr
    );

    let binary = result
        .output_path
        .as_ref()
        .expect("No output binary for x86_64 ELF test");
    common::verify_elf_magic(binary);
    common::verify_elf_class(binary, common::ELFCLASS64);
    common::verify_elf_arch(binary, common::EM_X86_64);
    verify_elf_data_encoding_le(binary);
}

/// Verify ELF32 little-endian format with EM_386 architecture.
#[test]
fn elf32_i686() {
    let source = "int main(void) { return 0; }\n";
    let result = common::compile_source(source, &["--target", common::TARGET_I686]);
    assert!(
        result.success,
        "Failed to compile for i686: {}",
        result.stderr
    );

    let binary = result
        .output_path
        .as_ref()
        .expect("No output binary for i686 ELF test");
    common::verify_elf_magic(binary);
    common::verify_elf_class(binary, common::ELFCLASS32);
    common::verify_elf_arch(binary, common::EM_386);
    verify_elf_data_encoding_le(binary);
}

/// Verify ELF64 little-endian format with EM_AARCH64 architecture.
#[test]
fn elf64_aarch64() {
    let source = "int main(void) { return 0; }\n";
    let result = common::compile_source(source, &["--target", common::TARGET_AARCH64]);
    assert!(
        result.success,
        "Failed to compile for aarch64: {}",
        result.stderr
    );

    let binary = result
        .output_path
        .as_ref()
        .expect("No output binary for aarch64 ELF test");
    common::verify_elf_magic(binary);
    common::verify_elf_class(binary, common::ELFCLASS64);
    common::verify_elf_arch(binary, common::EM_AARCH64);
    verify_elf_data_encoding_le(binary);
}

/// Verify ELF64 little-endian format with EM_RISCV architecture.
#[test]
fn elf64_riscv64() {
    let source = "int main(void) { return 0; }\n";
    let result = common::compile_source(source, &["--target", common::TARGET_RISCV64]);
    assert!(
        result.success,
        "Failed to compile for riscv64: {}",
        result.stderr
    );

    let binary = result
        .output_path
        .as_ref()
        .expect("No output binary for riscv64 ELF test");
    common::verify_elf_magic(binary);
    common::verify_elf_class(binary, common::ELFCLASS64);
    common::verify_elf_arch(binary, common::EM_RISCV);
    verify_elf_data_encoding_le(binary);
}

// ===========================================================================
// Phase 5: Output Mode Tests per Architecture
// ===========================================================================

/// Test static executable output mode for all architectures.
///
/// The default output mode should produce a runnable static executable
/// with correct ELF format properties and an ET_EXEC type.
#[test]
fn output_modes_static_executable() {
    let source = r#"
int main(void) {
    return 42;
}
"#;

    for &target in TARGETS {
        let dir = common::TempDir::new(
            &format!("static_exec_{}", target.split('-').next().unwrap_or("unknown")),
        );
        let mut output_path = PathBuf::from(dir.path());
        output_path.push("test_static");

        let result = common::compile_source(
            source,
            &["--target", target, "-o", output_path.to_str().unwrap()],
        );
        assert!(
            result.success,
            "Static executable compilation failed for '{}': {}",
            target, result.stderr
        );

        // Verify the output file exists and has reasonable size.
        verify_binary_exists_and_nontrivial(&output_path);

        // Verify ELF format properties.
        common::verify_elf_magic(&output_path);
        common::verify_elf_class(&output_path, expected_elf_class(target));
        common::verify_elf_arch(&output_path, expected_elf_arch(target));

        // If we can run on this target, verify the exit code.
        if can_run_target(target) {
            let (status, _stdout, stderr) = run_on_target(&output_path, target);
            assert_eq!(
                status.code().unwrap_or(-1),
                42,
                "Expected exit code 42 for static executable on '{}': stderr={}",
                target,
                stderr
            );
        }
    }
}

/// Test relocatable object output (`-c`) for all architectures.
///
/// The `-c` flag should produce a `.o` relocatable ELF object with `ET_REL`
/// type and correct ELF class and architecture.
#[test]
fn output_modes_relocatable_object() {
    let source = r#"
int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
"#;

    for &target in TARGETS {
        let dir = common::TempDir::new(
            &format!("reloc_obj_{}", target.split('-').next().unwrap_or("unknown")),
        );
        let output_path = dir.path().join("test.o");

        let result = common::compile_source(
            source,
            &["-c", "--target", target, "-o", output_path.to_str().unwrap()],
        );
        assert!(
            result.success,
            "Relocatable object compilation failed for '{}': {}",
            target, result.stderr
        );

        // Read the .o file and verify ELF header fields directly.
        let data = fs::read(&output_path).unwrap_or_else(|e| {
            panic!("Failed to read .o file for '{}': {}", target, e);
        });

        assert!(
            data.len() >= 20,
            ".o file for '{}' is too small ({} bytes)",
            target,
            data.len()
        );

        // ELF magic bytes at offset 0-3.
        assert_eq!(
            &data[0..4],
            &[0x7f, b'E', b'L', b'F'],
            "Expected ELF magic in .o for '{}'",
            target
        );

        // ELF class at offset 4.
        let exp_class = expected_elf_class(target);
        assert_eq!(
            data[4], exp_class,
            "ELF class mismatch in .o for '{}': expected {}, got {}",
            target, exp_class, data[4]
        );

        // e_type at offset 16-17 = ET_REL (1).
        let e_type = u16::from_le_bytes([data[16], data[17]]);
        assert_eq!(
            e_type, 1,
            "Expected ET_REL (1) for .o on '{}', got {}",
            target, e_type
        );

        // e_machine at offset 18-19.
        let e_machine = u16::from_le_bytes([data[18], data[19]]);
        let exp_arch = expected_elf_arch(target);
        assert_eq!(
            e_machine, exp_arch,
            "Architecture mismatch in .o for '{}': expected 0x{:04X}, got 0x{:04X}",
            target, exp_arch, e_machine
        );
    }
}

/// Test shared library output (`-shared -fPIC`) for all architectures.
///
/// The `-shared -fPIC` flags should produce a `.so` shared object with
/// `ET_DYN` type and correct ELF class and architecture.
#[test]
fn output_modes_shared_library() {
    let source = r#"
int shared_add(int a, int b) {
    return a + b;
}

int shared_multiply(int a, int b) {
    return a * b;
}
"#;

    for &target in TARGETS {
        let dir = common::TempDir::new(
            &format!("shared_lib_{}", target.split('-').next().unwrap_or("unknown")),
        );
        let output_path = dir.path().join("libtest.so");

        let result = common::compile_source(
            source,
            &[
                "-shared",
                "-fPIC",
                "--target",
                target,
                "-o",
                output_path.to_str().unwrap(),
            ],
        );
        assert!(
            result.success,
            "Shared library compilation failed for '{}': {}",
            target, result.stderr
        );

        // Read the .so file and verify ELF header fields directly.
        let data = fs::read(&output_path).unwrap_or_else(|e| {
            panic!("Failed to read .so file for '{}': {}", target, e);
        });

        assert!(
            data.len() >= 20,
            ".so file for '{}' is too small ({} bytes)",
            target,
            data.len()
        );

        // ELF magic bytes.
        assert_eq!(
            &data[0..4],
            &[0x7f, b'E', b'L', b'F'],
            "Expected ELF magic in .so for '{}'",
            target
        );

        // ELF class.
        let exp_class = expected_elf_class(target);
        assert_eq!(
            data[4], exp_class,
            "ELF class mismatch in .so for '{}': expected {}, got {}",
            target, exp_class, data[4]
        );

        // e_type at offset 16-17 = ET_DYN (3) for shared objects.
        let e_type = u16::from_le_bytes([data[16], data[17]]);
        assert_eq!(
            e_type, 3,
            "Expected ET_DYN (3) for .so on '{}', got {}",
            target, e_type
        );

        // e_machine at offset 18-19.
        let e_machine = u16::from_le_bytes([data[18], data[19]]);
        let exp_arch = expected_elf_arch(target);
        assert_eq!(
            e_machine, exp_arch,
            "Architecture mismatch in .so for '{}': expected 0x{:04X}, got 0x{:04X}",
            target, exp_arch, e_machine
        );
    }
}

// ===========================================================================
// Phase 7: Target Selection Tests
// ===========================================================================

/// Verify that `--target` flag is correctly parsed for all four target triples.
///
/// This test invokes the `bcc` binary directly with `--target` for each
/// supported triple and verifies the compiler accepts the target and produces
/// a valid output binary.
#[test]
fn target_flag_parsing() {
    let source = "int main(void) { return 0; }\n";
    let source_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();

    for &target in TARGETS {
        let dir = common::TempDir::new(
            &format!("target_parse_{}", target.split('-').next().unwrap_or("unknown")),
        );
        let output_path = dir.path().join("test_output");

        // Use Command::args() to pass multiple flags at once.
        let output = Command::new(&bcc)
            .args(&["--target", target, "-o"])
            .arg(&output_path)
            .arg(source_file.path())
            .output()
            .unwrap_or_else(|e| panic!("Failed to execute bcc: {}", e));

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "bcc rejected valid target '{}': stdout={}\nstderr={}",
            target, stdout, stderr
        );

        // Verify the output binary was produced.
        assert!(
            output_path.exists(),
            "No output binary produced for target '{}'",
            target
        );
    }
}

/// Verify that an unsupported target triple produces an error with exit code 1.
///
/// The compiler must exit with code 1 and emit a diagnostic on stderr when
/// given an unrecognized `--target` value. This verifies the GCC-compatible
/// diagnostic format rule.
#[test]
fn invalid_target_error() {
    let source = "int main(void) { return 0; }\n";
    let source_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();

    let invalid_targets = &[
        "mips-linux-gnu",
        "sparc-linux-gnu",
        "powerpc-linux-gnu",
        "wasm32-unknown-unknown",
        "totally-invalid-target",
    ];

    for &invalid_target in invalid_targets {
        // Use Command::status() for a quick pass/fail check first.
        let status = Command::new(&bcc)
            .args(&["--target", invalid_target])
            .arg(source_file.path())
            .status()
            .unwrap_or_else(|e| panic!("Failed to execute bcc: {}", e));

        assert!(
            !status.success(),
            "Expected error for unsupported target '{}', but compilation succeeded",
            invalid_target
        );

        // The compiler should exit with code 1 per the diagnostic format rule.
        let exit_code = status.code().unwrap_or(-1);
        assert_eq!(
            exit_code, 1,
            "Expected exit code 1 for invalid target '{}', got {}",
            invalid_target, exit_code
        );
    }
}

/// Verify that the default target matches the host architecture when `--target`
/// is omitted.
///
/// When no `--target` flag is provided, the compiler should default to the
/// host machine's architecture. This test compiles without `--target`, then
/// verifies the ELF output format matches the host and the binary is natively
/// runnable.
#[test]
fn default_target_is_host() {
    let source = r#"
int main(void) {
    return (int)(unsigned long)sizeof(void*);
}
"#;

    let source_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();
    let dir = common::TempDir::new("default_target");
    let output_path = dir.path().join("default_out");

    // Compile without --target flag.
    let output = Command::new(&bcc)
        .arg("-o")
        .arg(&output_path)
        .arg(source_file.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute bcc: {}", e));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Default target compilation failed: {}",
        stderr
    );

    // Verify ELF format matches the host architecture.
    common::verify_elf_magic(&output_path);

    let host_arch = std::env::consts::ARCH;
    match host_arch {
        "x86_64" => {
            common::verify_elf_class(&output_path, common::ELFCLASS64);
            common::verify_elf_arch(&output_path, common::EM_X86_64);
        }
        "x86" => {
            common::verify_elf_class(&output_path, common::ELFCLASS32);
            common::verify_elf_arch(&output_path, common::EM_386);
        }
        "aarch64" => {
            common::verify_elf_class(&output_path, common::ELFCLASS64);
            common::verify_elf_arch(&output_path, common::EM_AARCH64);
        }
        "riscv64" | "riscv64gc" => {
            common::verify_elf_class(&output_path, common::ELFCLASS64);
            common::verify_elf_arch(&output_path, common::EM_RISCV);
        }
        _ => {
            // Unknown host — just verify it is a valid ELF, no arch-specific checks.
            eprintln!(
                "Unknown host arch '{}': skipping architecture-specific ELF checks",
                host_arch
            );
        }
    }

    // The binary should be natively runnable since we compiled for the host.
    let run_result = Command::new(&output_path)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to execute default-target binary '{}': {}",
                output_path.display(),
                e
            )
        });

    // sizeof(void*) on the host Rust process should match the C binary's result.
    let expected_ptr_size = std::mem::size_of::<*const u8>() as i32;
    let actual_exit = run_result.status.code().unwrap_or(-1);
    assert_eq!(
        actual_exit, expected_ptr_size,
        "Default target pointer size mismatch: expected {}, got {}",
        expected_ptr_size, actual_exit
    );
}

// ===========================================================================
// Additional Cross-Architecture Tests
// ===========================================================================

/// Verify that global variables are correctly initialized and accessible
/// across all architectures.
///
/// Tests both zero-initialized (BSS) and explicitly initialized (DATA)
/// global variables.
#[test]
fn cross_compile_global_variables() {
    let source = r#"
int global_zero;                /* BSS: zero-initialized */
int global_init = 12345;        /* DATA: explicitly initialized */
static int static_var = 99;     /* Static storage */

int main(void) {
    if (global_zero != 0) return 1;
    if (global_init != 12345) return 2;
    if (static_var != 99) return 3;

    /* Modify and verify */
    global_zero = 42;
    if (global_zero != 42) return 4;

    global_init = global_init + 1;
    if (global_init != 12346) return 5;

    return 0;
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_global_variables for '{}' -- QEMU not available",
                target
            );
            continue;
        }

        let result = common::compile_and_run(source, target, &[]);
        assert!(
            result.success,
            "Global variables test failed on '{}': exit={:?}\nstdout: {}\nstderr: {}",
            target,
            result.exit_status.code(),
            result.stdout,
            result.stderr
        );
    }
}

/// Verify control flow constructs work correctly across all architectures.
///
/// Tests if/else, for loops, while loops, and switch/case to exercise
/// branch instruction encoding on each architecture.
#[test]
fn cross_compile_control_flow() {
    let source = r#"
int main(void) {
    /* if/else */
    int x = 10;
    if (x > 5) {
        x = x + 1;
    } else {
        return 1;
    }
    if (x != 11) return 2;

    /* for loop */
    int sum = 0;
    int i;
    for (i = 0; i < 10; i++) {
        sum = sum + i;
    }
    if (sum != 45) return 3;

    /* while loop */
    int count = 0;
    while (count < 5) {
        count = count + 1;
    }
    if (count != 5) return 4;

    /* switch/case */
    int val = 3;
    int result = 0;
    switch (val) {
        case 1: result = 10; break;
        case 2: result = 20; break;
        case 3: result = 30; break;
        default: result = -1; break;
    }
    if (result != 30) return 5;

    return 0;
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_control_flow for '{}' -- QEMU not available",
                target
            );
            continue;
        }

        let result = common::compile_and_run(source, target, &[]);
        assert!(
            result.success,
            "Control flow test failed on '{}': exit={:?}\nstdout: {}\nstderr: {}",
            target,
            result.exit_status.code(),
            result.stdout,
            result.stderr
        );
    }
}

/// Verify pointer arithmetic and dereferencing work correctly on all architectures.
///
/// Tests pointer-to-int operations, array indexing via pointers, and
/// pointer arithmetic which exercises target-specific pointer widths.
#[test]
fn cross_compile_pointers() {
    let source = r#"
int main(void) {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;

    /* Array access via pointer */
    int *p = arr;
    if (*p != 10) return 1;
    if (*(p + 1) != 20) return 2;
    if (*(p + 4) != 50) return 3;

    /* Pointer increment */
    p++;
    if (*p != 20) return 4;

    /* Pointer difference */
    int *end = &arr[4];
    int *start = &arr[0];
    long diff = end - start;
    if (diff != 4) return 5;

    return 0;
}
"#;

    for &target in TARGETS {
        if !can_run_target(target) {
            eprintln!(
                "SKIP: cross_compile_pointers for '{}' -- QEMU not available",
                target
            );
            continue;
        }

        let result = common::compile_and_run(source, target, &[]);
        assert!(
            result.success,
            "Pointer test failed on '{}': exit={:?}\nstdout: {}\nstderr: {}",
            target,
            result.exit_status.code(),
            result.stdout,
            result.stderr
        );
    }
}

/// Verify that compilation with `-O0` and `-O2` flags works for all targets.
///
/// This exercises the optimization pipeline integration with each backend.
#[test]
fn cross_compile_optimization_levels() {
    let source = r#"
int compute(int n) {
    int sum = 0;
    int i;
    for (i = 1; i <= n; i++) {
        sum = sum + i;
    }
    return sum;
}

int main(void) {
    /* n*(n+1)/2 for n=10 = 55 */
    if (compute(10) != 55) return 1;
    if (compute(0) != 0) return 2;
    if (compute(1) != 1) return 3;
    return 0;
}
"#;

    let opt_levels = &["-O0", "-O1", "-O2"];

    for &target in TARGETS {
        for &opt in opt_levels {
            if !can_run_target(target) {
                eprintln!(
                    "SKIP: cross_compile_optimization_levels for '{}' {} -- QEMU not available",
                    target, opt
                );
                continue;
            }

            let result = common::compile_and_run(source, target, &[opt]);
            assert!(
                result.success,
                "Optimization test failed on '{}' with {}: exit={:?}\nstdout: {}\nstderr: {}",
                target,
                opt,
                result.exit_status.code(),
                result.stdout,
                result.stderr
            );
        }
    }
}

/// Verify that the `--target` flag can be combined with other compilation flags
/// (e.g., `-O2`, `-g`) across all architectures.
#[test]
fn cross_compile_combined_flags() {
    let source = r#"
int main(void) {
    int x = 42;
    return x - 42;
}
"#;
    let bcc = common::get_bcc_binary();
    let source_file = common::write_temp_source(source);

    for &target in TARGETS {
        let dir = common::TempDir::new(
            &format!("combined_{}", target.split('-').next().unwrap_or("unknown")),
        );
        let out_path = dir.path().join("combined_out");

        // Combine --target, -O2, -g, and -o flags using Command::args().
        let output = Command::new(&bcc)
            .args(&[
                "--target", target,
                "-O2",
                "-g",
                "-o",
            ])
            .arg(&out_path)
            .arg(source_file.path())
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to execute bcc with combined flags for '{}': {}",
                    target, e
                )
            });

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "Combined flags compilation failed for '{}': {}",
            target, stderr
        );

        // Verify ELF output.
        let path_ref = Path::new(out_path.to_str().unwrap());
        verify_binary_exists_and_nontrivial(path_ref);
        common::verify_elf_magic(path_ref);
        common::verify_elf_class(path_ref, expected_elf_class(target));
        common::verify_elf_arch(path_ref, expected_elf_arch(target));
    }
}
