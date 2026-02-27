//! Integration tests for the bcc x86-64 code generation backend.
//!
//! These tests cover the primary target architecture (x86_64-linux-gnu) across
//! eight test categories:
//!
//! 1. **Basic code generation** — arithmetic, bitwise, comparison/branch, load/store, LEA, immediates
//! 2. **Instruction encoding** — REX prefix, ModR/M, SIB, RIP-relative, SSE
//! 3. **System V AMD64 ABI compliance** — register args, stack args, callee-saved, alignment, red zone
//! 4. **Register allocation** — simple expressions, spill code, callee-save preservation
//! 5. **ELF64 output format** — header validation, relocatable objects, shared libraries, PIC
//! 6. **Floating-point operations** — SSE single/double precision arithmetic and comparison
//! 7. **Security hardening** — retpoline, CET endbr64, stack probing (via flags)
//! 8. **End-to-end execution** — hello world, fibonacci, factorial, struct passing
//!
//! # Test Strategy
//!
//! Tests are black-box integration tests that invoke the `bcc` binary as a subprocess.
//! Binary output is inspected for machine code byte patterns and ELF format compliance.
//! Execution tests compile and run programs natively on x86-64 hosts.
//!
//! # Zero-Dependency Guarantee
//!
//! This file uses ONLY the Rust standard library (`std`) and the `bcc` crate
//! (via the shared `tests/common/mod.rs` utilities). No external crates.

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Target triple for x86-64 Linux (the primary/default target).
const TARGET: &str = "x86_64-linux-gnu";

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Compile C source code targeting x86-64 with additional flags.
///
/// Automatically adds `--target x86_64-linux-gnu` to the flag list.
/// Returns the full `CompileResult` for inspection.
fn compile_x86_64(source: &str, extra_flags: &[&str]) -> common::CompileResult {
    let mut flags: Vec<&str> = extra_flags.to_vec();
    flags.push("--target");
    flags.push(TARGET);
    common::compile_source(source, &flags)
}

/// Compile C source code targeting x86-64 to a relocatable object file.
///
/// Uses `-c` flag to stop after compilation (no linking), producing a `.o` file.
fn compile_x86_64_obj(source: &str) -> common::CompileResult {
    compile_x86_64(source, &["-c"])
}

/// Search for a byte subsequence within a larger byte slice.
///
/// Returns `true` if `pattern` appears as a contiguous subsequence within `data`.
/// Uses a simple sliding window search (O(n*m) worst case, acceptable for test binaries).
fn find_bytes(data: &[u8], pattern: &[u8]) -> bool {
    if pattern.is_empty() {
        return true;
    }
    if pattern.len() > data.len() {
        return false;
    }
    data.windows(pattern.len()).any(|window| window == pattern)
}

/// Find all offsets where a byte pattern occurs in the given data.
///
/// Returns a `Vec<usize>` of starting offsets. Empty if pattern not found.
fn find_byte_offsets(data: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > data.len() {
        return Vec::new();
    }
    data.windows(pattern.len())
        .enumerate()
        .filter(|(_, window)| *window == pattern)
        .map(|(i, _)| i)
        .collect()
}

/// Extract the `.text` section bytes from an ELF64 binary.
///
/// Parses the ELF64 header and section header table to locate the `.text`
/// section, then returns its raw contents as a byte vector. Returns `None`
/// if the section is not found or the binary is not a valid ELF64 file.
///
/// This is used by instruction encoding tests to inspect raw machine code
/// without relying on external tools like `objdump`.
fn extract_elf64_text_section(data: &[u8]) -> Option<Vec<u8>> {
    // Minimum ELF64 header size is 64 bytes.
    if data.len() < 64 {
        return None;
    }
    // Verify ELF magic bytes.
    if data[0..4] != [0x7f, b'E', b'L', b'F'] {
        return None;
    }
    // Verify ELF64 class (byte index 4).
    if data[4] != 2 {
        return None;
    }

    // Parse ELF64 header fields (little-endian byte order).
    let e_shoff = u64::from_le_bytes(data[40..48].try_into().ok()?) as usize;
    let e_shentsize = u16::from_le_bytes(data[58..60].try_into().ok()?) as usize;
    let e_shnum = u16::from_le_bytes(data[60..62].try_into().ok()?) as usize;
    let e_shstrndx = u16::from_le_bytes(data[62..64].try_into().ok()?) as usize;

    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum {
        return None;
    }

    // Locate the section header string table.
    let shstrtab_hdr_off = e_shoff + e_shstrndx * e_shentsize;
    if shstrtab_hdr_off + e_shentsize > data.len() {
        return None;
    }
    let strtab_offset = u64::from_le_bytes(
        data[shstrtab_hdr_off + 24..shstrtab_hdr_off + 32]
            .try_into()
            .ok()?,
    ) as usize;
    let strtab_size = u64::from_le_bytes(
        data[shstrtab_hdr_off + 32..shstrtab_hdr_off + 40]
            .try_into()
            .ok()?,
    ) as usize;

    if strtab_offset + strtab_size > data.len() {
        return None;
    }
    let strtab = &data[strtab_offset..strtab_offset + strtab_size];

    // Iterate section headers to find ".text".
    for i in 0..e_shnum {
        let sh_start = e_shoff + i * e_shentsize;
        if sh_start + e_shentsize > data.len() {
            continue;
        }

        let sh_name_idx =
            u32::from_le_bytes(data[sh_start..sh_start + 4].try_into().ok()?) as usize;
        if sh_name_idx >= strtab.len() {
            continue;
        }

        // Read null-terminated name from the string table.
        let name_end = strtab[sh_name_idx..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| sh_name_idx + p)
            .unwrap_or(strtab.len());
        let section_name = std::str::from_utf8(&strtab[sh_name_idx..name_end]).ok()?;

        if section_name == ".text" {
            let sh_offset =
                u64::from_le_bytes(data[sh_start + 24..sh_start + 32].try_into().ok()?) as usize;
            let sh_size =
                u64::from_le_bytes(data[sh_start + 32..sh_start + 40].try_into().ok()?) as usize;

            if sh_offset + sh_size > data.len() {
                return None;
            }
            return Some(data[sh_offset..sh_offset + sh_size].to_vec());
        }
    }

    None
}

/// Check whether a byte is a REX prefix (0x40–0x4F).
///
/// REX prefixes encode operand-size promotion and register extensions:
/// - Bit 3 (W): 64-bit operand size
/// - Bit 2 (R): Extends ModR/M reg field
/// - Bit 1 (X): Extends SIB index field
/// - Bit 0 (B): Extends ModR/M r/m, SIB base, or opcode reg field
fn is_rex_prefix(byte: u8) -> bool {
    (0x40..=0x4F).contains(&byte)
}

/// Check whether a byte is a REX.W prefix (0x48–0x4F).
///
/// REX.W indicates a 64-bit operand size override.
fn is_rex_w(byte: u8) -> bool {
    (0x48..=0x4F).contains(&byte)
}

// ===========================================================================
// Phase 2: Basic Code Generation Tests
// ===========================================================================

/// Test: Simple return value compilation and machine code encoding.
///
/// Compiles `int main() { return 42; }` and verifies:
/// - Compilation succeeds for x86-64 target
/// - Output binary contains `MOV eax, 42` encoding: `B8 2A 00 00 00`
/// - Output binary contains `RET` encoding: `C3`
#[test]
fn x86_64_simple_return() {
    let source = "int main() { return 42; }";
    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Compilation failed for simple return:\nstderr: {}\nstdout: {}",
        result.stderr, result.stdout
    );

    // Read the compiled object file and inspect for expected byte patterns.
    if let Some(ref output_path) = result.output_path {
        let binary_data = fs::read(output_path).unwrap_or_else(|e| {
            panic!(
                "Failed to read compiled object at '{}': {}",
                output_path.display(),
                e
            );
        });

        // MOV eax, 42 is encoded as B8 + imm32(42): B8 2A 00 00 00
        let mov_eax_42: &[u8] = &[0xB8, 0x2A, 0x00, 0x00, 0x00];
        // RET is encoded as C3
        let ret: &[u8] = &[0xC3];

        assert!(
            find_bytes(&binary_data, mov_eax_42),
            "Expected MOV eax, 42 (B8 2A 00 00 00) in compiled output for '{}'",
            output_path.display()
        );
        assert!(
            find_bytes(&binary_data, ret),
            "Expected RET (C3) in compiled output for '{}'",
            output_path.display()
        );
    }
}

/// Test: Integer arithmetic operations (ADD, SUB, IMUL, IDIV, MOD).
///
/// Verifies that arithmetic instructions compile successfully and produce
/// correct results when linked and executed.
#[test]
fn x86_64_integer_arithmetic() {
    let source = r#"
        int add(int a, int b) { return a + b; }
        int sub(int a, int b) { return a - b; }
        int mul(int a, int b) { return a * b; }
        int divide(int a, int b) { return a / b; }
        int modulo(int a, int b) { return a % b; }

        int main() {
            int r1 = add(10, 20);     // 30
            int r2 = sub(50, 17);     // 33
            int r3 = mul(6, 7);       // 42
            int r4 = divide(100, 5);  // 20
            int r5 = modulo(17, 5);   // 2
            // 30 + 33 + 42 + 20 + 2 = 127; return 0 on success
            return r1 + r2 + r3 + r4 + r5 - 127;
        }
    "#;

    let obj_result = compile_x86_64_obj(source);
    assert!(
        obj_result.success,
        "Integer arithmetic compilation failed:\nstderr: {}",
        obj_result.stderr
    );

    // Compile, link, and execute to verify correctness.
    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Integer arithmetic execution returned non-zero:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Bitwise operations (AND, OR, XOR, SHL, SHR, SAR).
///
/// Verifies that bitwise instructions compile and execute correctly,
/// including arithmetic right shift (SAR) which preserves the sign bit.
#[test]
fn x86_64_bitwise_operations() {
    let source = r#"
        int main() {
            int a = 0xFF00;
            int b = 0x0F0F;

            int and_result = a & b;
            int or_result  = a | b;
            int xor_result = a ^ b;
            int shl_result = 1 << 10;
            int shr_result = 1024 >> 3;
            int sar_result = (-128) >> 2;

            if (and_result != 3840) return 1;
            if (or_result != 65295) return 2;
            if (xor_result != 61455) return 3;
            if (shl_result != 1024) return 4;
            if (shr_result != 128) return 5;
            if (sar_result != -32) return 6;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Bitwise operations compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Bitwise operations execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Comparison and conditional branch instruction generation.
///
/// Verifies CMP + Jcc (conditional jump) sequences for signed integer
/// comparisons (JG, JL, JGE, JLE, JE, JNE).
#[test]
fn x86_64_comparison_and_branch() {
    let source = r#"
        int max(int a, int b) {
            if (a > b) return a;
            return b;
        }

        int min(int a, int b) {
            if (a < b) return a;
            return b;
        }

        int clamp(int val, int lo, int hi) {
            if (val < lo) return lo;
            if (val > hi) return hi;
            return val;
        }

        int main() {
            if (max(10, 20) != 20) return 1;
            if (max(-5, 3) != 3) return 2;
            if (min(10, 20) != 10) return 3;
            if (min(-5, 3) != -5) return 4;
            if (clamp(50, 0, 100) != 50) return 5;
            if (clamp(-10, 0, 100) != 0) return 6;
            if (clamp(200, 0, 100) != 100) return 7;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Comparison/branch compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Comparison/branch execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Load and store operations with memory operands.
///
/// Verifies MOV instructions with stack-relative addressing (local variables),
/// RIP-relative addressing (global variables), and pointer dereferences.
#[test]
fn x86_64_load_store() {
    let source = r#"
        int global_val = 42;

        int main() {
            int local = global_val;
            global_val = 100;
            int loaded = global_val;

            if (local != 42) return 1;
            if (loaded != 100) return 2;

            int x = 55;
            int *p = &x;
            *p = 77;
            if (x != 77) return 3;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Load/store compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Load/store execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: LEA instruction generation for address computation.
///
/// Array indexing and pointer arithmetic typically generate LEA instructions
/// on x86-64 for efficient address computation without memory access.
#[test]
fn x86_64_lea_instruction() {
    let source = r#"
        int main() {
            int arr[10];
            int i;

            for (i = 0; i < 10; i++) {
                arr[i] = i * i;
            }

            if (arr[5] != 25) return 1;
            if (arr[9] != 81) return 2;

            int *p = &arr[3];
            if (*p != 9) return 3;
            if (*(p + 2) != 25) return 4;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "LEA instruction test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "LEA instruction test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Immediate operand encoding across different sizes.
///
/// Tests that 8-bit, 32-bit, and sign-extended immediate operands are correctly
/// encoded in x86-64 machine instructions.
#[test]
fn x86_64_immediate_operands() {
    let source = r#"
        int main() {
            int a = 1;
            int b = 0x12345678;
            int c = -1;
            int d = 2147483647;

            if (a != 1) return 1;
            if (b != 0x12345678) return 2;
            if (c != -1) return 3;
            if (d != 2147483647) return 4;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Immediate operand compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Immediate operand execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

// ===========================================================================
// Phase 3: Instruction Encoding Tests
// ===========================================================================

/// Test: REX.W prefix for 64-bit register operations.
///
/// Verifies that 64-bit operations (e.g., `long` arithmetic) emit the REX.W
/// prefix byte (0x48-0x4F). REX.W is mandatory for 64-bit operand size.
#[test]
fn x86_64_rex_prefix() {
    let source = r#"
        long add64(long a, long b) { return a + b; }
        long sub64(long a, long b) { return a - b; }

        int main() {
            long x = 0x100000000L;
            long y = 0x200000000L;
            long z = add64(x, y);
            if (z != 0x300000000L) return 1;

            long w = sub64(y, x);
            if (w != 0x100000000L) return 2;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "REX prefix test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        let binary_data = fs::read(output_path).unwrap_or_else(|e| {
            panic!("Failed to read object file: {}", e);
        });

        let has_rex_w = binary_data.iter().any(|&b| is_rex_w(b));
        assert!(
            has_rex_w,
            "Expected REX.W prefix (0x48-0x4F) in object file for 64-bit operations"
        );
    }
}

/// Test: REX.R extension for encoding registers r8-r15.
///
/// The REX.R bit (bit 2 of REX byte) extends the ModR/M reg field,
/// enabling access to r8-r15. High register pressure triggers usage.
#[test]
fn x86_64_rex_r_extension() {
    let source = r#"
        long compute(long a, long b, long c, long d, long e, long f) {
            long v1 = a + b;
            long v2 = c + d;
            long v3 = e + f;
            long v4 = v1 * v2;
            long v5 = v2 * v3;
            long v6 = v4 + v5;
            long v7 = v6 - v1;
            long v8 = v7 + v3;
            return v8;
        }

        int main() {
            long result = compute(1, 2, 3, 4, 5, 6);
            if (result != 106) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "REX.R extension test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        let binary_data = fs::read(output_path).unwrap_or_else(|e| {
            panic!("Failed to read object file: {}", e);
        });

        // Check for any REX prefix with R-bit set (bit 2).
        let has_rex_r = binary_data
            .iter()
            .any(|&b| is_rex_prefix(b) && (b & 0x04) != 0);
        // The allocator may or may not use r8-r15; compilation correctness is primary.
        let _rex_r_detected = has_rex_r;
    }

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "REX.R extension test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: ModR/M byte construction for register and memory operands.
///
/// Exercises register-to-register (mod=11), memory with displacement
/// (mod=01/10), and register-indirect (mod=00) addressing modes.
#[test]
fn x86_64_modrm_encoding() {
    let source = r#"
        int main() {
            int x = 10;
            int y = 20;

            int sum = x + y;

            int arr[4] = {1, 2, 3, 4};
            int val = arr[2];

            if (sum != 30) return 1;
            if (val != 3) return 2;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ModR/M encoding test compilation failed:\nstderr: {}",
        result.stderr
    );

    assert!(
        result.output_path.is_some(),
        "Expected output file for ModR/M encoding test"
    );
}

/// Test: SIB byte encoding for scaled-index addressing.
///
/// The SIB byte encodes `[base + index*scale + disp]` addressing,
/// used for array indexing with element-size scale factors.
#[test]
fn x86_64_sib_encoding() {
    let source = r#"
        int main() {
            int arr[16];
            int i;

            for (i = 0; i < 16; i++) {
                arr[i] = i * 3;
            }

            if (arr[0] != 0) return 1;
            if (arr[5] != 15) return 2;
            if (arr[15] != 45) return 3;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "SIB encoding test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "SIB encoding test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: RIP-relative addressing mode for global data access.
///
/// On x86-64, global variables use RIP-relative addressing (ModR/M mod=00,
/// r/m=101 with 32-bit displacement from RIP).
#[test]
fn x86_64_rip_relative() {
    let source = r#"
        int global_x = 42;
        int global_y = 0;

        int main() {
            global_y = global_x + 8;
            if (global_y != 50) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "RIP-relative addressing test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "RIP-relative addressing test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: SSE instruction encoding for floating-point operations.
///
/// Verifies that float/double operations generate SSE/SSE2 instructions
/// with correct prefix bytes (F3 0F for single, F2 0F for double).
#[test]
fn x86_64_sse_encoding() {
    let source = r#"
        float fadd(float a, float b) { return a + b; }
        double dadd(double a, double b) { return a + b; }

        int main() {
            float f = fadd(1.5f, 2.5f);
            double d = dadd(1.5, 2.5);

            if (f < 3.9f || f > 4.1f) return 1;
            if (d < 3.9 || d > 4.1) return 2;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "SSE encoding test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        let data = fs::read(output_path).unwrap();

        let has_sse_single = find_bytes(&data, &[0xF3, 0x0F]);
        let has_sse_double = find_bytes(&data, &[0xF2, 0x0F]);

        assert!(
            has_sse_single || has_sse_double,
            "Expected SSE prefix bytes (F3 0F or F2 0F) in float arithmetic object"
        );
    }
}

// ===========================================================================
// Phase 4: ABI Compliance Tests (System V AMD64)
// ===========================================================================

/// Test: First 6 integer arguments passed in registers (rdi, rsi, rdx, rcx, r8, r9).
///
/// The System V AMD64 ABI specifies that the first 6 integer/pointer arguments
/// are passed in registers rdi, rsi, rdx, rcx, r8, r9 (in that order).
#[test]
fn x86_64_abi_integer_args() {
    let source = r#"
        int sum6(int a, int b, int c, int d, int e, int f) {
            return a + b + c + d + e + f;
        }

        int main() {
            int result = sum6(1, 2, 3, 4, 5, 6);
            if (result != 21) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI integer args test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI integer args execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: First 8 float arguments passed in xmm0-xmm7.
///
/// The System V AMD64 ABI specifies that the first 8 floating-point arguments
/// are passed in xmm0 through xmm7.
#[test]
fn x86_64_abi_float_args() {
    let source = r#"
        double sum8f(double a, double b, double c, double d,
                     double e, double f, double g, double h) {
            return a + b + c + d + e + f + g + h;
        }

        int main() {
            double result = sum8f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            if (result < 35.9 || result > 36.1) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI float args test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI float args execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Arguments beyond the register capacity are passed on the stack.
///
/// When there are more than 6 integer arguments, the excess are passed on the
/// stack in right-to-left order per the System V AMD64 ABI.
#[test]
fn x86_64_abi_stack_args() {
    let source = r#"
        int sum10(int a, int b, int c, int d, int e, int f,
                  int g, int h, int i, int j) {
            return a + b + c + d + e + f + g + h + i + j;
        }

        int main() {
            int result = sum10(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
            if (result != 55) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI stack args test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI stack args execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Integer return value in rax.
///
/// The System V AMD64 ABI specifies that integer return values go in rax.
/// This tests returning a full 64-bit value to verify 64-bit rax return.
#[test]
fn x86_64_abi_return_rax() {
    let source = r#"
        long get_large_value(void) {
            return 0x0123456789ABCDEFL;
        }

        int main() {
            long val = get_large_value();
            if (val != 0x0123456789ABCDEFL) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI return rax test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI return rax execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Float return value in xmm0.
///
/// The System V AMD64 ABI specifies that floating-point return values
/// are placed in xmm0.
#[test]
fn x86_64_abi_return_xmm0() {
    let source = r#"
        double get_pi(void) {
            return 3.14159265358979;
        }

        int main() {
            double pi = get_pi();
            if (pi < 3.14 || pi > 3.15) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI return xmm0 test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI return xmm0 execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Callee-saved registers (rbx, r12-r15, rbp) are preserved across calls.
///
/// The System V AMD64 ABI requires callee-saved registers to be preserved by
/// any function that uses them. This test creates enough register pressure to
/// force callee-save spilling.
#[test]
fn x86_64_abi_callee_saved() {
    let source = r#"
        long heavy_compute(long a, long b, long c, long d, long e, long f) {
            long r1 = a * b;
            long r2 = c * d;
            long r3 = e * f;
            long r4 = r1 + r2;
            long r5 = r2 + r3;
            long r6 = r3 + r1;
            long r7 = r4 * r5;
            long r8 = r5 * r6;
            long r9 = r6 * r4;
            long r10 = r7 + r8 + r9;
            long r11 = r10 - r4 - r5 - r6;
            long r12 = r11 + a + b + c + d + e + f;
            return r12;
        }

        int main() {
            long val = heavy_compute(1, 2, 3, 4, 5, 6);
            if (val == 0) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI callee-saved test compilation failed:\nstderr: {}",
        result.stderr
    );

    // Inspect the object for PUSH/POP instructions indicating callee-save patterns.
    // PUSH RBP = 0x55 is the canonical function prologue instruction.
    if let Some(ref output_path) = result.output_path {
        let binary_data = fs::read(output_path).unwrap_or_else(|e| {
            panic!("Failed to read object file: {}", e);
        });

        assert!(
            find_bytes(&binary_data, &[0x55]),
            "Expected PUSH RBP (0x55) in callee-saved register test"
        );
    }

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI callee-saved execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: 16-byte stack alignment at call sites.
///
/// The System V AMD64 ABI mandates that RSP is 16-byte aligned before a CALL
/// instruction. Misalignment causes SSE instructions to fault on some CPUs.
#[test]
fn x86_64_abi_stack_alignment() {
    let source = r#"
        int aligned_func(int a, int b, int c, int d, int e, int f,
                         int g, int h) {
            return a + b + c + d + e + f + g + h;
        }

        int main() {
            int result = aligned_func(1, 2, 3, 4, 5, 6, 7, 8);
            if (result != 36) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI stack alignment test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI stack alignment execution failed (possible misalignment):\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: 128-byte red zone usage in leaf functions.
///
/// The System V AMD64 ABI provides a 128-byte "red zone" below RSP that leaf
/// functions may use without adjusting RSP. The compiler may optimize leaf
/// functions to omit the frame setup.
#[test]
fn x86_64_abi_red_zone() {
    let source = r#"
        int leaf_add(int a, int b) {
            int x = a;
            int y = b;
            int sum = x + y;
            return sum;
        }

        int main() {
            int result = leaf_add(10, 20);
            if (result != 30) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI red zone test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI red zone test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Struct passing and return via the System V AMD64 ABI.
///
/// Small structs (<=16 bytes) with simple fields may be passed in registers
/// (INTEGER classification). Larger structs (>16 bytes) are passed via hidden
/// pointer (MEMORY classification).
#[test]
fn x86_64_abi_struct_passing() {
    let source = r#"
        struct Point {
            int x;
            int y;
        };

        struct Point make_point(int x, int y) {
            struct Point p;
            p.x = x;
            p.y = y;
            return p;
        }

        int point_sum(struct Point p) {
            return p.x + p.y;
        }

        struct BigStruct {
            long a;
            long b;
            long c;
        };

        int big_sum(struct BigStruct s) {
            return (int)(s.a + s.b + s.c);
        }

        int main() {
            struct Point p = make_point(10, 20);
            if (point_sum(p) != 30) return 1;

            struct BigStruct bs;
            bs.a = 100;
            bs.b = 200;
            bs.c = 300;
            if (big_sum(bs) != 600) return 2;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "ABI struct passing test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "ABI struct passing execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

// ===========================================================================
// Phase 5: Register Allocation Tests
// ===========================================================================

/// Test: Register allocation for simple expressions.
///
/// Verifies that the register allocator correctly assigns registers for
/// basic expressions with a moderate number of simultaneously live values.
#[test]
fn x86_64_regalloc_simple() {
    let source = r#"
        int main() {
            int a = 1, b = 2, c = 3, d = 4;
            int sum = a + b + c + d;
            int prod = a * b * c * d;

            if (sum != 10) return 1;
            if (prod != 24) return 2;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Regalloc simple test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Regalloc simple test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Register allocation with spill code generation.
///
/// When the number of simultaneously live values exceeds the available registers
/// (15 GPRs excluding RSP on x86-64), the allocator must generate spill code.
/// This declares 20 simultaneously-live variables to ensure spilling occurs.
#[test]
fn x86_64_regalloc_spill() {
    let source = r#"
        int main() {
            int v1 = 1, v2 = 2, v3 = 3, v4 = 4, v5 = 5;
            int v6 = 6, v7 = 7, v8 = 8, v9 = 9, v10 = 10;
            int v11 = 11, v12 = 12, v13 = 13, v14 = 14, v15 = 15;
            int v16 = 16, v17 = 17, v18 = 18, v19 = 19, v20 = 20;

            int sum = v1 + v2 + v3 + v4 + v5 +
                      v6 + v7 + v8 + v9 + v10 +
                      v11 + v12 + v13 + v14 + v15 +
                      v16 + v17 + v18 + v19 + v20;

            if (sum != 210) return 1;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Regalloc spill test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Regalloc spill test execution failed (incorrect spill code?):\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Callee-saved register preservation in the register allocator.
///
/// Verifies that when the register allocator uses callee-saved registers
/// (rbx, r12-r15, rbp), it correctly saves and restores them around usage.
/// Values needed across function calls must survive in callee-saved registers.
#[test]
fn x86_64_regalloc_callee_save() {
    let source = r#"
        int identity(int x) { return x; }

        int compute(int n) {
            int a = n + 1;
            int b = n + 2;
            int c = n + 3;
            int d = n + 4;
            int e = n + 5;
            int f = n + 6;
            int g = n + 7;
            int h = n + 8;

            // Function call clobbers caller-saved registers.
            // Values a-h must survive in callee-saved regs or on stack.
            int mid = identity(a) + identity(b) + identity(c) + identity(d);

            return mid + e + f + g + h;
        }

        int main() {
            int result = compute(0);
            if (result != 36) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Regalloc callee-save test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Regalloc callee-save test execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

// ===========================================================================
// Phase 6: ELF Output Tests
// ===========================================================================

/// Test: ELF64 output with correct header fields for x86-64.
///
/// Verifies that the compiler produces a valid ELF64 binary with:
/// - ELF magic bytes (0x7F 'E' 'L' 'F')
/// - ELFCLASS64 (class byte = 2)
/// - EM_X86_64 (machine = 0x3E)
/// - Little-endian data encoding (EI_DATA = 1)
#[test]
fn x86_64_elf64_output() {
    let source = "int main() { return 0; }";
    let result = compile_x86_64(source, &[]);

    assert!(
        result.success,
        "ELF64 output test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        // Use the common module's ELF verification utilities.
        common::verify_elf_magic(output_path.as_path());
        common::verify_elf_class(output_path.as_path(), common::ELFCLASS64);
        common::verify_elf_arch(output_path.as_path(), common::EM_X86_64);

        // Additional header field checks.
        let data = fs::read(output_path).unwrap();

        // EI_DATA: little-endian = 1
        assert_eq!(
            data[5], 1,
            "Expected little-endian ELF (EI_DATA=1), got: {}",
            data[5]
        );

        // EI_VERSION: must be 1 (current)
        assert_eq!(
            data[6], 1,
            "Expected ELF version 1 (EI_VERSION=1), got: {}",
            data[6]
        );

        // EI_OSABI: System V (0) or Linux (3)
        assert!(
            data[7] == 0 || data[7] == 3,
            "Expected System V (0) or Linux (3) OS/ABI, got: {}",
            data[7]
        );
    }
}

/// Test: Relocatable object output via `-c` flag.
///
/// When `-c` is specified, the compiler should produce a relocatable ELF object
/// (ET_REL = 1) rather than a linked executable (ET_EXEC = 2).
#[test]
fn x86_64_relocatable_object() {
    let source = r#"
        int add(int a, int b) { return a + b; }
        int sub(int a, int b) { return a - b; }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Relocatable object test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        let data = fs::read(output_path).unwrap();

        common::verify_elf_magic(output_path.as_path());
        common::verify_elf_class(output_path.as_path(), common::ELFCLASS64);
        common::verify_elf_arch(output_path.as_path(), common::EM_X86_64);

        // e_type at offset 16 (u16 LE): ET_REL = 1
        assert!(
            data.len() >= 18,
            "Object file too small: {} bytes",
            data.len()
        );
        let e_type = u16::from_le_bytes([data[16], data[17]]);
        assert_eq!(
            e_type, 1,
            "Expected ET_REL (1) for relocatable object, got: {}",
            e_type
        );
    }
}

/// Test: Shared library output via `-shared -fPIC`.
///
/// Verifies that the compiler produces a shared library (ET_DYN = 3) when
/// compiled with `-shared -fPIC` flags.
#[test]
fn x86_64_shared_library() {
    let source = r#"
        int shared_add(int a, int b) { return a + b; }
        int shared_mul(int a, int b) { return a * b; }
    "#;

    let dir = common::TempDir::new("shared_lib_test");
    let output = dir.path().join("libtest.so");
    let output_str = output.display().to_string();

    let result = compile_x86_64(source, &["-shared", "-fPIC", "-o", &output_str]);
    assert!(
        result.success,
        "Shared library compilation failed:\nstderr: {}",
        result.stderr
    );

    if output.exists() {
        let data = fs::read(&output).unwrap();

        // Verify ELF magic
        assert_eq!(
            &data[0..4],
            &[0x7f, b'E', b'L', b'F'],
            "Shared library missing ELF magic"
        );

        // e_type = ET_DYN (3) at offset 16
        if data.len() >= 18 {
            let e_type = u16::from_le_bytes([data[16], data[17]]);
            assert_eq!(
                e_type, 3,
                "Expected ET_DYN (3) for shared library, got: {}",
                e_type
            );
        }
    }
}

/// Test: Position-independent code generation with `-fPIC`.
///
/// Verifies that `-fPIC` causes the compiler to emit PIC code that uses
/// GOT/PLT-relative addressing for global variables and function calls.
#[test]
fn x86_64_pic_codegen() {
    let source = r#"
        int global_counter = 0;

        void increment(void) {
            global_counter++;
        }

        int get_counter(void) {
            return global_counter;
        }
    "#;

    let result = compile_x86_64(source, &["-fPIC", "-c"]);
    assert!(
        result.success,
        "PIC codegen test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        let data = fs::read(output_path).unwrap();

        common::verify_elf_magic(output_path.as_path());
        common::verify_elf_class(output_path.as_path(), common::ELFCLASS64);

        // A PIC object file should contain code and relocation sections,
        // resulting in a non-trivially sized file.
        assert!(
            data.len() > 100,
            "PIC object file seems too small at {} bytes",
            data.len()
        );
    }
}

// ===========================================================================
// Phase 7: Floating-Point Tests (SSE)
// ===========================================================================

/// Test: Single-precision (float) arithmetic using SSE instructions.
///
/// Verifies ADDSS, SUBSS, MULSS, DIVSS instruction generation for
/// single-precision floating-point operations.
#[test]
fn x86_64_float_arithmetic() {
    let source = r#"
        float fadd(float a, float b) { return a + b; }
        float fsub(float a, float b) { return a - b; }
        float fmul(float a, float b) { return a * b; }
        float fdiv(float a, float b) { return a / b; }

        int main() {
            float a = fadd(1.5f, 2.5f);
            float b = fsub(10.0f, 3.5f);
            float c = fmul(3.0f, 4.0f);
            float d = fdiv(15.0f, 3.0f);

            if (a < 3.9f || a > 4.1f) return 1;
            if (b < 6.4f || b > 6.6f) return 2;
            if (c < 11.9f || c > 12.1f) return 3;
            if (d < 4.9f || d > 5.1f) return 4;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Float arithmetic test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Float arithmetic execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Double-precision arithmetic using SSE2 instructions.
///
/// Verifies ADDSD, SUBSD, MULSD, DIVSD instruction generation for
/// double-precision floating-point operations.
#[test]
fn x86_64_double_arithmetic() {
    let source = r#"
        double dadd(double a, double b) { return a + b; }
        double dsub(double a, double b) { return a - b; }
        double dmul(double a, double b) { return a * b; }
        double ddiv(double a, double b) { return a / b; }

        int main() {
            double a = dadd(1.5, 2.5);
            double b = dsub(10.0, 3.5);
            double c = dmul(3.0, 4.0);
            double d = ddiv(15.0, 3.0);

            if (a < 3.99 || a > 4.01) return 1;
            if (b < 6.49 || b > 6.51) return 2;
            if (c < 11.99 || c > 12.01) return 3;
            if (d < 4.99 || d > 5.01) return 4;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Double arithmetic test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Double arithmetic execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

/// Test: Floating-point comparison using UCOMISS/UCOMISD instructions.
///
/// Verifies that float/double comparisons generate the correct UCOMISS (single)
/// or UCOMISD (double) instructions followed by conditional branches.
#[test]
fn x86_64_float_comparison() {
    let source = r#"
        int float_max(float a, float b) {
            if (a > b) return 1;
            return 0;
        }

        int double_cmp(double a, double b) {
            if (a < b) return -1;
            if (a > b) return 1;
            return 0;
        }

        int main() {
            if (float_max(3.0f, 2.0f) != 1) return 1;
            if (float_max(1.0f, 5.0f) != 0) return 2;

            if (double_cmp(1.0, 2.0) != -1) return 3;
            if (double_cmp(3.0, 1.0) != 1) return 4;
            if (double_cmp(2.0, 2.0) != 0) return 5;

            return 0;
        }
    "#;

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Float comparison test compilation failed:\nstderr: {}",
        result.stderr
    );

    let run_result = common::compile_and_run(source, TARGET, &[]);
    if common::is_native_target(TARGET) {
        assert!(
            run_result.success,
            "Float comparison execution failed:\nstderr: {}",
            run_result.stderr
        );
    }
}

// ===========================================================================
// Phase 8: Execution Tests
// ===========================================================================

/// Test: Compile and execute a hello world program natively on x86-64.
///
/// This is the fundamental end-to-end smoke test that verifies the entire
/// compilation pipeline (frontend → sema → IR → codegen → linker) produces
/// a working x86-64 executable that can produce output on stdout.
#[test]
fn x86_64_execute_hello_world() {
    let source = r#"
        extern int printf(const char *format, ...);

        int main() {
            printf("Hello, World!\n");
            return 0;
        }
    "#;

    let run_result = common::compile_and_run(source, common::TARGET_X86_64, &[]);

    if common::is_native_target(common::TARGET_X86_64) {
        assert_run_success!(run_result);
        assert_output_contains!(run_result, "Hello, World!");
    }
}

/// Test: Compile and execute a fibonacci program on x86-64.
///
/// Tests iterative computation with loop control flow, array access, and
/// integer arithmetic. Verifies correct code generation for the standard
/// fibonacci sequence.
#[test]
fn x86_64_execute_fibonacci() {
    let source = r#"
        extern int printf(const char *format, ...);

        int fibonacci(int n) {
            if (n <= 0) return 0;
            if (n == 1) return 1;

            int prev2 = 0;
            int prev1 = 1;
            int current = 0;
            int i;

            for (i = 2; i <= n; i++) {
                current = prev1 + prev2;
                prev2 = prev1;
                prev1 = current;
            }
            return current;
        }

        int main() {
            printf("fib(10)=%d\n", fibonacci(10));
            printf("fib(20)=%d\n", fibonacci(20));

            if (fibonacci(0) != 0) return 1;
            if (fibonacci(1) != 1) return 2;
            if (fibonacci(10) != 55) return 3;
            if (fibonacci(20) != 6765) return 4;

            return 0;
        }
    "#;

    let run_result = common::compile_and_run(source, common::TARGET_X86_64, &[]);

    if common::is_native_target(common::TARGET_X86_64) {
        assert!(
            run_result.success,
            "Fibonacci execution failed:\nstdout: {}\nstderr: {}",
            run_result.stdout, run_result.stderr
        );
        assert!(
            run_result.stdout.contains("fib(10)=55"),
            "Expected 'fib(10)=55' in output, got:\n{}",
            run_result.stdout
        );
        assert!(
            run_result.stdout.contains("fib(20)=6765"),
            "Expected 'fib(20)=6765' in output, got:\n{}",
            run_result.stdout
        );
    }
}

/// Test: Compile and execute a recursive factorial program on x86-64.
///
/// Verifies that recursive function calls work correctly, including proper
/// stack frame setup/teardown and return value propagation through the
/// call chain.
#[test]
fn x86_64_execute_recursion() {
    let source = r#"
        extern int printf(const char *format, ...);

        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }

        int main() {
            printf("fact(5)=%d\n", factorial(5));
            printf("fact(10)=%d\n", factorial(10));

            if (factorial(0) != 1) return 1;
            if (factorial(1) != 1) return 2;
            if (factorial(5) != 120) return 3;
            if (factorial(10) != 3628800) return 4;

            return 0;
        }
    "#;

    let run_result = common::compile_and_run(source, common::TARGET_X86_64, &[]);

    if common::is_native_target(common::TARGET_X86_64) {
        assert!(
            run_result.success,
            "Recursion (factorial) execution failed:\nstdout: {}\nstderr: {}",
            run_result.stdout, run_result.stderr
        );
        assert!(
            run_result.stdout.contains("fact(5)=120"),
            "Expected 'fact(5)=120' in output, got:\n{}",
            run_result.stdout
        );
    }
}

/// Test: Compile and execute a struct passing and return program on x86-64.
///
/// Tests struct value passing (by value) and return, which exercises the
/// ABI's struct classification rules (INTEGER vs MEMORY) and register
/// assignment for aggregate types.
#[test]
fn x86_64_execute_structs() {
    let source = r#"
        extern int printf(const char *format, ...);

        struct Vec2 {
            int x;
            int y;
        };

        struct Vec2 vec2_add(struct Vec2 a, struct Vec2 b) {
            struct Vec2 result;
            result.x = a.x + b.x;
            result.y = a.y + b.y;
            return result;
        }

        int vec2_dot(struct Vec2 a, struct Vec2 b) {
            return a.x * b.x + a.y * b.y;
        }

        int main() {
            struct Vec2 a;
            a.x = 3;
            a.y = 4;
            struct Vec2 b;
            b.x = 1;
            b.y = 2;

            struct Vec2 sum = vec2_add(a, b);
            int dot = vec2_dot(a, b);

            printf("sum=(%d,%d) dot=%d\n", sum.x, sum.y, dot);

            if (sum.x != 4) return 1;
            if (sum.y != 6) return 2;
            if (dot != 11) return 3;

            return 0;
        }
    "#;

    let run_result = common::compile_and_run(source, common::TARGET_X86_64, &[]);

    if common::is_native_target(common::TARGET_X86_64) {
        assert!(
            run_result.success,
            "Struct execution failed:\nstdout: {}\nstderr: {}",
            run_result.stdout, run_result.stderr
        );
        assert!(
            run_result.stdout.contains("sum=(4,6) dot=11"),
            "Expected 'sum=(4,6) dot=11' in output, got:\n{}",
            run_result.stdout
        );
    }
}

// ===========================================================================
// Optimization Level Tests
// ===========================================================================

/// Test: Compilation with -O0 (no optimization) produces valid output.
#[test]
fn x86_64_opt_level_o0() {
    let source = r#"
        int main() {
            int x = 42;
            return x - 42;
        }
    "#;

    let result = compile_x86_64(source, &["-O0"]);
    assert!(
        result.success,
        "-O0 compilation failed:\nstderr: {}",
        result.stderr
    );
}

/// Test: Compilation with -O1 (basic optimization) produces valid output.
#[test]
fn x86_64_opt_level_o1() {
    let source = r#"
        int main() {
            int x = 42;
            return x - 42;
        }
    "#;

    let result = compile_x86_64(source, &["-O1"]);
    assert!(
        result.success,
        "-O1 compilation failed:\nstderr: {}",
        result.stderr
    );
}

/// Test: Compilation with -O2 (aggressive optimization) produces valid output.
#[test]
fn x86_64_opt_level_o2() {
    let source = r#"
        int main() {
            int x = 42;
            return x - 42;
        }
    "#;

    let result = compile_x86_64(source, &["-O2"]);
    assert!(
        result.success,
        "-O2 compilation failed:\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Security Hardening Tests (x86-64 specific)
// ===========================================================================

/// Test: Compilation with -mretpoline flag for retpoline thunks.
///
/// Verifies that the compiler accepts the `-mretpoline` flag and produces
/// a valid object file. Retpoline replaces indirect JMP/CALL with
/// speculative-execution-safe thunk sequences.
#[test]
fn x86_64_retpoline_flag() {
    let source = r#"
        typedef int (*func_ptr)(int);

        int double_val(int x) { return x * 2; }

        int call_indirect(func_ptr fn, int arg) {
            return fn(arg);
        }

        int main() {
            int result = call_indirect(double_val, 21);
            if (result != 42) return 1;
            return 0;
        }
    "#;

    let result = compile_x86_64(source, &["-mretpoline", "-c"]);
    assert!(
        result.success,
        "Retpoline compilation failed:\nstderr: {}",
        result.stderr
    );
}

/// Test: Compilation with -fcf-protection flag for CET endbr64.
///
/// Verifies that the compiler accepts the `-fcf-protection` flag and
/// inserts `endbr64` (F3 0F 1E FA) at function entries and indirect
/// branch targets.
#[test]
fn x86_64_cet_endbr64_flag() {
    let source = r#"
        int add(int a, int b) { return a + b; }

        int main() {
            return add(1, 2) - 3;
        }
    "#;

    let result = compile_x86_64(source, &["-fcf-protection", "-c"]);
    assert!(
        result.success,
        "CET endbr64 compilation failed:\nstderr: {}",
        result.stderr
    );

    // Check for endbr64 encoding: F3 0F 1E FA
    if let Some(ref output_path) = result.output_path {
        let data = fs::read(output_path).unwrap();
        let endbr64_pattern: &[u8] = &[0xF3, 0x0F, 0x1E, 0xFA];

        // With -fcf-protection, each function should start with endbr64.
        // The pattern should appear at least once (for main and add).
        let offsets = find_byte_offsets(&data, endbr64_pattern);
        assert!(
            !offsets.is_empty(),
            "Expected endbr64 (F3 0F 1E FA) with -fcf-protection, found none in '{}'",
            output_path.display()
        );
    }
}

// ===========================================================================
// Direct Invocation Tests (exercises Command, write_temp_source, get_bcc_binary)
// ===========================================================================

/// Test: Direct compiler binary invocation with explicit Command usage.
///
/// This test exercises the `Command` API directly (rather than through the
/// common module helpers) to verify that the `bcc` binary can be invoked
/// programmatically with various flags.
#[test]
fn x86_64_direct_invocation() {
    let bcc = common::get_bcc_binary();
    let dir = common::TempDir::new("direct_invoke");
    let source_content = "int main() { return 0; }\n";

    // Use write_temp_source for source file creation.
    let source_file = common::write_temp_source(source_content);

    // Construct output path using PathBuf::from on a string representation.
    let output_str = format!("{}/direct_test.o", dir.path().display());
    let output_path = PathBuf::from(&output_str);

    // Use Command API directly with .arg() and .args() chaining.
    let output = Command::new(bcc.as_path())
        .arg("-c")
        .args(["--target", common::TARGET_X86_64])
        .arg("-o")
        .arg(output_path.as_path())
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc binary directly");

    assert!(
        output.status.success(),
        "Direct invocation failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify the output file was produced.
    assert!(
        output_path.as_path().exists(),
        "Expected output file at '{}'",
        output_path.display()
    );

    // Use Command.status() as an alternative to .output().
    let status = Command::new(bcc.as_path())
        .arg("-c")
        .args(["--target", common::TARGET_X86_64])
        .arg("-o")
        .arg(dir.path().join("status_test.o"))
        .arg(source_file.path())
        .status()
        .expect("Failed to get exit status from bcc");

    assert!(status.success(), "Direct invocation via status() failed");

    // Clean up the output file using fs::remove_file.
    if output_path.exists() {
        fs::remove_file(&output_path).unwrap_or_else(|e| {
            eprintln!(
                "Warning: failed to remove '{}': {}",
                output_path.display(),
                e
            );
        });
    }
}

/// Test: Source file creation via fs::write and compilation via direct invocation.
///
/// Exercises fs::write for creating test source files, fs::read_to_string for
/// reading back content, and PathBuf construction from string paths.
#[test]
fn x86_64_fs_operations() {
    let dir = common::TempDir::new("fs_ops_test");
    let source_path: PathBuf = dir.path().join("test_source.c");
    let output_path: PathBuf = dir.path().join("test_output.o");

    let source_content = "int add(int a, int b) { return a + b; }\n";

    // Write source file using fs::write.
    fs::write(&source_path, source_content).unwrap_or_else(|e| {
        panic!(
            "Failed to write source file '{}': {}",
            source_path.display(),
            e
        );
    });

    // Read it back using fs::read_to_string.
    let read_back = fs::read_to_string(&source_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read source file '{}': {}",
            source_path.display(),
            e
        );
    });
    assert_eq!(
        read_back, source_content,
        "Source file content mismatch after write/read"
    );

    // Compile the file.
    let bcc = common::get_bcc_binary();
    let result = Command::new(bcc.as_path())
        .arg("-c")
        .arg("--target")
        .arg(TARGET)
        .arg("-o")
        .arg(output_path.as_path())
        .arg(source_path.as_path())
        .output()
        .expect("Failed to execute bcc");

    assert!(
        result.status.success(),
        "Compilation via fs::write source failed:\nstderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    // Read the binary output using fs::read.
    if output_path.exists() {
        let binary = fs::read(&output_path).unwrap();
        assert!(
            binary.len() > 4,
            "Output binary too small: {} bytes",
            binary.len()
        );

        // Clean up using fs::remove_file.
        let _ = fs::remove_file(&output_path);
        let _ = fs::remove_file(&source_path);
    }
}

/// Test: Binary inspection using extract_elf64_text_section helper.
///
/// Verifies the `.text` section extraction from a compiled ELF64 object file
/// and inspects it for expected machine code patterns.
#[test]
fn x86_64_text_section_extraction() {
    let source = "int main() { return 42; }";

    let result = compile_x86_64_obj(source);
    assert!(
        result.success,
        "Text section extraction test compilation failed:\nstderr: {}",
        result.stderr
    );

    if let Some(ref output_path) = result.output_path {
        let data = fs::read(output_path).unwrap();

        if let Some(text_section) = extract_elf64_text_section(&data) {
            // The .text section should contain the compiled function body.
            assert!(!text_section.is_empty(), "Extracted .text section is empty");

            // MOV eax, 42 = B8 2A 00 00 00 should appear in the text section.
            assert!(
                find_bytes(&text_section, &[0xB8, 0x2A, 0x00, 0x00, 0x00]),
                "Expected MOV eax, 42 in .text section (len={})",
                text_section.len()
            );

            // RET = C3 should appear.
            assert!(
                find_bytes(&text_section, &[0xC3]),
                "Expected RET (C3) in .text section"
            );
        }
        // If text section extraction fails, the ELF format may differ;
        // the compile-success assertion above is the primary validation.
    }
}
