//! Shared test utilities for the bcc C compiler integration tests.
//!
//! This module provides reusable test infrastructure for all integration test files
//! in the `tests/` directory. It includes:
//!
//! - **RAII temporary file/directory management** (`TempDir`, `TempFile`) with automatic cleanup
//! - **Compiler invocation helpers** (`compile_source`, `compile_and_run`) for invoking the `bcc` binary
//! - **ELF binary verification** (`verify_elf_magic`, `verify_elf_class`, `verify_elf_arch`)
//! - **QEMU cross-architecture execution** (`run_with_qemu`, `is_qemu_available`, `is_native_target`)
//! - **Target architecture constants** for all four supported architectures
//! - **Assertion macros** for concise test assertions
//!
//! # Zero-Dependency Guarantee
//!
//! This module uses ONLY the Rust standard library (`std`). No external crates are imported.
//!
//! # Supported Architectures
//!
//! | Target Triple           | ELF Class | Architecture Constant |
//! |-------------------------|-----------|-----------------------|
//! | `x86_64-linux-gnu`      | ELF64     | `EM_X86_64` (0x3E)   |
//! | `i686-linux-gnu`        | ELF32     | `EM_386` (0x03)       |
//! | `aarch64-linux-gnu`     | ELF64     | `EM_AARCH64` (0xB7)  |
//! | `riscv64-linux-gnu`     | ELF64     | `EM_RISCV` (0xF3)    |

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Static atomic counter for generating unique temp directory/file names.
// Using SeqCst ordering ensures no naming collisions when `cargo test` runs
// multiple integration tests concurrently across threads.
// ---------------------------------------------------------------------------
static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ===========================================================================
// ELF Architecture Constants
// ===========================================================================

/// ELF machine type for Intel 80386 (i686 / IA-32).
#[allow(dead_code)]
pub const EM_386: u16 = 0x03;

/// ELF machine type for AMD x86-64 (x86_64).
#[allow(dead_code)]
pub const EM_X86_64: u16 = 0x3E;

/// ELF machine type for ARM AARCH64 (AArch64).
#[allow(dead_code)]
pub const EM_AARCH64: u16 = 0xB7;

/// ELF machine type for RISC-V.
#[allow(dead_code)]
pub const EM_RISCV: u16 = 0xF3;

// ===========================================================================
// ELF Class Constants
// ===========================================================================

/// ELF class identifier for 32-bit objects (i686).
#[allow(dead_code)]
pub const ELFCLASS32: u8 = 1;

/// ELF class identifier for 64-bit objects (x86-64, AArch64, RISC-V 64).
#[allow(dead_code)]
pub const ELFCLASS64: u8 = 2;

// ===========================================================================
// Target Triple Constants
// ===========================================================================

/// Target triple for x86-64 Linux (primary target).
#[allow(dead_code)]
pub const TARGET_X86_64: &str = "x86_64-linux-gnu";

/// Target triple for i686 (32-bit x86) Linux.
#[allow(dead_code)]
pub const TARGET_I686: &str = "i686-linux-gnu";

/// Target triple for AArch64 (ARM 64-bit) Linux.
#[allow(dead_code)]
pub const TARGET_AARCH64: &str = "aarch64-linux-gnu";

/// Target triple for RISC-V 64-bit Linux.
#[allow(dead_code)]
pub const TARGET_RISCV64: &str = "riscv64-linux-gnu";

/// All four supported target triples.
#[allow(dead_code)]
pub const ALL_TARGETS: &[&str] = &[TARGET_X86_64, TARGET_I686, TARGET_AARCH64, TARGET_RISCV64];

// ===========================================================================
// Result Types
// ===========================================================================

/// Result of invoking the `bcc` compiler on a C source file.
///
/// Contains the full compiler output (stdout, stderr), exit status, and the
/// path to the output binary (if compilation produced one).
#[derive(Debug)]
#[allow(dead_code)]
pub struct CompileResult {
    /// Whether the compilation succeeded (exit code 0).
    #[allow(dead_code)]
    pub success: bool,
    /// The raw exit status from the compiler process.
    #[allow(dead_code)]
    pub exit_status: ExitStatus,
    /// Captured standard output from the compiler.
    #[allow(dead_code)]
    pub stdout: String,
    /// Captured standard error from the compiler (diagnostics appear here).
    #[allow(dead_code)]
    pub stderr: String,
    /// Path to the output binary, if one was produced.
    /// `None` if compilation failed before producing output.
    #[allow(dead_code)]
    pub output_path: Option<PathBuf>,
}

/// Result of executing a compiled binary (natively or via QEMU).
///
/// Contains the full process output and exit status.
#[derive(Debug)]
#[allow(dead_code)]
pub struct RunResult {
    /// Whether the execution succeeded (exit code 0).
    #[allow(dead_code)]
    pub success: bool,
    /// The raw exit status from the executed process.
    #[allow(dead_code)]
    pub exit_status: ExitStatus,
    /// Captured standard output from the executed binary.
    #[allow(dead_code)]
    pub stdout: String,
    /// Captured standard error from the executed binary.
    #[allow(dead_code)]
    pub stderr: String,
}

// ===========================================================================
// TempDir — RAII Temporary Directory
// ===========================================================================

/// RAII wrapper for a temporary directory that is automatically removed on drop.
///
/// Each `TempDir` is created with a unique name under `std::env::temp_dir()`
/// using a combination of process ID and an atomic counter to prevent naming
/// collisions during parallel test execution.
///
/// # Example
/// ```ignore
/// let dir = TempDir::new("my_test");
/// let source_path = dir.path().join("test.c");
/// // directory is automatically removed when `dir` goes out of scope
/// ```
#[allow(dead_code)]
pub struct TempDir {
    #[allow(dead_code)]
    path: PathBuf,
}

#[allow(dead_code)]
impl TempDir {
    /// Create a new temporary directory with the given prefix.
    ///
    /// The directory is created under `std::env::temp_dir()` with the format:
    /// `bcc_test_{pid}_{counter}_{prefix}`
    ///
    /// # Panics
    ///
    /// Panics if the directory cannot be created (e.g., filesystem permissions).
    pub fn new(prefix: &str) -> Self {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir_name = format!("bcc_test_{}_{}_{}", pid, counter, prefix);
        let path = env::temp_dir().join(dir_name);
        fs::create_dir_all(&path).unwrap_or_else(|e| {
            panic!(
                "Failed to create temp directory '{}': {}",
                path.display(),
                e
            );
        });
        TempDir { path }
    }

    /// Returns a reference to the temporary directory path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best-effort cleanup: ignore errors (directory may already be removed,
        // or we may not have permissions in unusual test environments).
        let _ = fs::remove_dir_all(&self.path);
    }
}

// ===========================================================================
// TempFile — RAII Temporary File
// ===========================================================================

/// RAII wrapper for a temporary file that is automatically removed on drop.
///
/// Each `TempFile` is created with a unique name using an atomic counter
/// to prevent naming collisions during parallel test execution.
#[allow(dead_code)]
pub struct TempFile {
    #[allow(dead_code)]
    path: PathBuf,
}

#[allow(dead_code)]
impl TempFile {
    /// Returns a reference to the temporary file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        // Best-effort cleanup: ignore errors (file may already be removed).
        let _ = fs::remove_file(&self.path);
    }
}

// ===========================================================================
// Core Utility Functions
// ===========================================================================

/// Locate the `bcc` compiler binary from the Cargo build output directory.
///
/// This function uses the `CARGO_BIN_EXE_bcc` environment variable (set by Cargo
/// for integration tests since Rust 1.43+) to find the compiled `bcc` binary.
/// If that variable is not set, it falls back to searching `target/debug/bcc` and
/// `target/release/bcc` relative to `CARGO_MANIFEST_DIR`.
///
/// # Panics
///
/// Panics with a descriptive message if the binary cannot be found, which
/// indicates a build failure or misconfigured test environment.
#[allow(dead_code)]
pub fn get_bcc_binary() -> PathBuf {
    // Strategy 1: Use CARGO_BIN_EXE_bcc (available in integration tests since Rust 1.43+).
    // This is the preferred approach as Cargo sets it automatically.
    if let Ok(bin_path) = env::var("CARGO_BIN_EXE_bcc") {
        let path = PathBuf::from(&bin_path);
        if path.exists() {
            return path;
        }
    }

    // Strategy 2: Fall back to searching known build output directories.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let manifest_path = PathBuf::from(&manifest_dir);

    // Check debug build first (most common during development).
    let debug_path = manifest_path.join("target").join("debug").join("bcc");
    if debug_path.exists() {
        return debug_path;
    }

    // Check release build.
    let release_path = manifest_path.join("target").join("release").join("bcc");
    if release_path.exists() {
        return release_path;
    }

    panic!(
        "Could not find the 'bcc' binary. Ensure the project is built before running tests.\n\
         Searched:\n\
         - CARGO_BIN_EXE_bcc environment variable\n\
         - {}/target/debug/bcc\n\
         - {}/target/release/bcc",
        manifest_dir, manifest_dir
    );
}

/// Write C source code to a unique temporary `.c` file.
///
/// Creates a temporary file under `std::env::temp_dir()` with a unique name
/// (using process ID and atomic counter), writes the provided content, and
/// returns a `TempFile` handle that will automatically clean up on drop.
///
/// # Arguments
///
/// * `content` - The C source code to write to the temporary file.
///
/// # Returns
///
/// A `TempFile` RAII handle. The file will be deleted when this handle is dropped.
///
/// # Panics
///
/// Panics if the file cannot be created or the content cannot be written.
#[allow(dead_code)]
pub fn write_temp_source(content: &str) -> TempFile {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let file_name = format!("bcc_test_{}_{}.c", pid, counter);
    let path = env::temp_dir().join(file_name);

    let mut file = fs::File::create(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to create temp source file '{}': {}",
            path.display(),
            e
        );
    });
    file.write_all(content.as_bytes()).unwrap_or_else(|e| {
        panic!(
            "Failed to write to temp source file '{}': {}",
            path.display(),
            e
        );
    });

    TempFile { path }
}

/// Compile C source code using the `bcc` compiler binary.
///
/// Writes the source to a temporary `.c` file, invokes `bcc` with the provided
/// flags, and captures the compiler's output. If no `-o` flag is present in `flags`,
/// an output path is automatically generated in a temporary directory.
///
/// The temporary source file is cleaned up via RAII, but the output binary (if any)
/// is preserved so callers can inspect it (e.g., for ELF verification).
///
/// # Arguments
///
/// * `source` - The C source code to compile.
/// * `flags` - Additional compiler flags (e.g., `&["-O2", "--target", "x86_64-linux-gnu"]`).
///
/// # Returns
///
/// A `CompileResult` containing the exit status, captured stdout/stderr, and
/// the path to the output binary (if one was produced).
///
/// # Panics
///
/// Panics if the `bcc` binary cannot be found or the compiler process fails to spawn.
#[allow(dead_code)]
pub fn compile_source(source: &str, flags: &[&str]) -> CompileResult {
    let source_file = write_temp_source(source);
    let bcc = get_bcc_binary();

    // Determine whether the caller supplied an `-o` flag.
    let has_output_flag = flags.iter().any(|f| *f == "-o");

    // If no `-o` flag was provided, generate a temp output path.
    let temp_dir = if !has_output_flag {
        Some(TempDir::new("compile_output"))
    } else {
        None
    };

    let output_path = if let Some(ref dir) = temp_dir {
        Some(dir.path().join("a.out"))
    } else {
        // Scan flags to find the value after `-o`.
        let mut output = None;
        let mut found_o = false;
        for flag in flags {
            if found_o {
                output = Some(PathBuf::from(flag));
                break;
            }
            if *flag == "-o" {
                found_o = true;
            }
        }
        output
    };

    let mut cmd = Command::new(&bcc);
    cmd.args(flags);

    // Add the output flag if not already present.
    if let (false, Some(ref out_path)) = (has_output_flag, &output_path) {
        cmd.arg("-o").arg(out_path);
    }

    // Add the source file as the last argument.
    cmd.arg(source_file.path());

    let result: Output = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Failed to execute bcc compiler at '{}': {}",
            bcc.display(),
            e
        );
    });

    let success = result.status.success();
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&result.stderr).into_owned();

    // Determine the actual output path. If compilation failed, the output may not exist.
    let final_output_path = if success {
        output_path.filter(|p| p.exists())
    } else {
        None
    };

    // Prevent the temp_dir from being dropped (and cleaned up) if it contains
    // the output binary that the caller may want to inspect.
    if let Some(dir) = temp_dir {
        if final_output_path.is_some() {
            // Leak the TempDir so its contents persist for caller inspection.
            // The OS will clean up temp directories eventually, and individual
            // test runs are short-lived.
            std::mem::forget(dir);
        }
        // If final_output_path is None, dir drops normally and cleans up.
    }

    CompileResult {
        success,
        exit_status: result.status,
        stdout,
        stderr,
        output_path: final_output_path,
    }
}

/// Compile C source code and run the resulting binary.
///
/// Compiles the source for the specified target architecture, then executes the
/// binary either natively (if the target matches the host) or via QEMU user-mode
/// emulation (for cross-architecture targets).
///
/// # Arguments
///
/// * `source` - The C source code to compile.
/// * `target` - The target triple (e.g., `"x86_64-linux-gnu"`).
/// * `flags` - Additional compiler flags beyond `--target`.
///
/// # Returns
///
/// A `RunResult` with the execution output. If compilation fails, the `RunResult`
/// will have `success = false` and `stderr` will contain the compiler diagnostics.
#[allow(dead_code)]
pub fn compile_and_run(source: &str, target: &str, flags: &[&str]) -> RunResult {
    // Build the flag list: user flags + --target.
    let mut all_flags: Vec<&str> = flags.to_vec();
    all_flags.push("--target");
    all_flags.push(target);

    let compile_result = compile_source(source, &all_flags);

    // If compilation failed, return a RunResult reflecting the failure.
    if !compile_result.success {
        return RunResult {
            success: false,
            exit_status: compile_result.exit_status,
            stdout: compile_result.stdout,
            stderr: format!(
                "Compilation failed for target '{}':\n{}",
                target, compile_result.stderr
            ),
        };
    }

    let binary_path = match compile_result.output_path {
        Some(ref p) => p.clone(),
        None => {
            return RunResult {
                success: false,
                exit_status: compile_result.exit_status,
                stdout: String::new(),
                stderr: format!(
                    "Compilation succeeded but no output binary found for target '{}'",
                    target
                ),
            };
        }
    };

    // Determine execution strategy: native or QEMU.
    if is_native_target(target) {
        // Execute natively.
        let result = Command::new(&binary_path).output();
        match result {
            Ok(output) => RunResult {
                success: output.status.success(),
                exit_status: output.status,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            },
            Err(e) => RunResult {
                success: false,
                exit_status: compile_result.exit_status,
                stdout: String::new(),
                stderr: format!("Failed to execute binary '{}': {}", binary_path.display(), e),
            },
        }
    } else {
        // Execute via QEMU.
        run_with_qemu(&binary_path, target)
    }
}

/// Execute a compiled binary via QEMU user-mode emulation.
///
/// Maps the target triple to the appropriate QEMU binary name and invokes it.
/// If QEMU is not available for the target, returns a `RunResult` with
/// `success = false` and a descriptive skip message in `stderr`.
///
/// # Arguments
///
/// * `binary` - Path to the compiled ELF binary.
/// * `target` - The target triple (determines which QEMU emulator to use).
///
/// # Returns
///
/// A `RunResult` with the execution output.
#[allow(dead_code)]
pub fn run_with_qemu(binary: &Path, target: &str) -> RunResult {
    let qemu_name = qemu_binary_for_target(target);

    // Check if QEMU is available before attempting to run.
    if !is_qemu_available(target) {
        // Return a non-success result with a skip message rather than panicking.
        // This allows tests to gracefully skip when QEMU is not installed.
        //
        // We fabricate a dummy ExitStatus by running a known command.
        let dummy = Command::new("false").status().unwrap_or_else(|e| {
            panic!("Failed to run 'false' for dummy exit status: {}", e);
        });
        return RunResult {
            success: false,
            exit_status: dummy,
            stdout: String::new(),
            stderr: format!(
                "QEMU not available for target '{}' (tried '{}'). \
                 Install qemu-user-static to run cross-architecture tests.",
                target, qemu_name
            ),
        };
    }

    let result = Command::new(qemu_name).arg(binary).output();

    match result {
        Ok(output) => RunResult {
            success: output.status.success(),
            exit_status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        },
        Err(e) => {
            let dummy = Command::new("false").status().unwrap_or_else(|err| {
                panic!("Failed to run 'false' for dummy exit status: {}", err);
            });
            RunResult {
                success: false,
                exit_status: dummy,
                stdout: String::new(),
                stderr: format!(
                    "Failed to execute '{}' for binary '{}': {}",
                    qemu_name,
                    binary.display(),
                    e
                ),
            }
        }
    }
}

// ===========================================================================
// ELF Verification Utilities
// ===========================================================================

/// Verify that a binary file starts with the ELF magic bytes (`\x7fELF`).
///
/// # Arguments
///
/// * `binary` - Path to the binary file to check.
///
/// # Panics
///
/// Panics (via `assert_eq!`) if the first 4 bytes do not match the ELF magic,
/// or if the file cannot be read.
#[allow(dead_code)]
pub fn verify_elf_magic(binary: &Path) {
    let data = fs::read(binary).unwrap_or_else(|e| {
        panic!(
            "Failed to read binary '{}' for ELF magic verification: {}",
            binary.display(),
            e
        );
    });
    assert!(
        data.len() >= 4,
        "Binary '{}' is too small ({} bytes) to contain ELF magic",
        binary.display(),
        data.len()
    );
    let magic = &data[0..4];
    assert_eq!(
        magic,
        &[0x7f, b'E', b'L', b'F'],
        "Expected ELF magic bytes [7f 45 4c 46], got: {:02x?} in '{}'",
        magic,
        binary.display()
    );
}

/// Verify the ELF class (32-bit or 64-bit) of a binary file.
///
/// The ELF class is stored at byte offset 4 (`EI_CLASS`) in the ELF identification
/// header.
///
/// # Arguments
///
/// * `binary` - Path to the ELF binary file.
/// * `expected_class` - Expected ELF class: `ELFCLASS32` (1) or `ELFCLASS64` (2).
///
/// # Panics
///
/// Panics (via `assert_eq!`) if the ELF class does not match, or if the file
/// cannot be read.
#[allow(dead_code)]
pub fn verify_elf_class(binary: &Path, expected_class: u8) {
    let data = fs::read(binary).unwrap_or_else(|e| {
        panic!(
            "Failed to read binary '{}' for ELF class verification: {}",
            binary.display(),
            e
        );
    });
    assert!(
        data.len() >= 5,
        "Binary '{}' is too small ({} bytes) to contain ELF class field",
        binary.display(),
        data.len()
    );
    let actual_class = data[4];
    let class_name = match expected_class {
        1 => "ELFCLASS32",
        2 => "ELFCLASS64",
        _ => "UNKNOWN",
    };
    let actual_name = match actual_class {
        1 => "ELFCLASS32",
        2 => "ELFCLASS64",
        _ => "UNKNOWN",
    };
    assert_eq!(
        actual_class, expected_class,
        "ELF class mismatch in '{}': expected {} ({}), got {} ({})",
        binary.display(),
        expected_class,
        class_name,
        actual_class,
        actual_name
    );
}

/// Verify the ELF machine architecture (`e_machine`) of a binary file.
///
/// The `e_machine` field is a little-endian `u16` at byte offset 18 in both
/// ELF32 and ELF64 headers.
///
/// # Arguments
///
/// * `binary` - Path to the ELF binary file.
/// * `expected_arch` - Expected ELF machine type (e.g., `EM_X86_64`, `EM_386`).
///
/// # Panics
///
/// Panics (via `assert_eq!`) if the architecture does not match, or if the file
/// cannot be read.
#[allow(dead_code)]
pub fn verify_elf_arch(binary: &Path, expected_arch: u16) {
    let data = fs::read(binary).unwrap_or_else(|e| {
        panic!(
            "Failed to read binary '{}' for ELF architecture verification: {}",
            binary.display(),
            e
        );
    });
    assert!(
        data.len() >= 20,
        "Binary '{}' is too small ({} bytes) to contain ELF e_machine field",
        binary.display(),
        data.len()
    );
    // e_machine is at offset 18 in both ELF32 and ELF64 headers, little-endian u16.
    let actual_arch = u16::from_le_bytes([data[18], data[19]]);
    let arch_name = |v: u16| -> &str {
        match v {
            0x03 => "EM_386 (i686)",
            0x3E => "EM_X86_64 (x86-64)",
            0xB7 => "EM_AARCH64 (AArch64)",
            0xF3 => "EM_RISCV (RISC-V)",
            _ => "UNKNOWN",
        }
    };
    assert_eq!(
        actual_arch, expected_arch,
        "ELF architecture mismatch in '{}': expected 0x{:04X} ({}), got 0x{:04X} ({})",
        binary.display(),
        expected_arch,
        arch_name(expected_arch),
        actual_arch,
        arch_name(actual_arch)
    );
}

// ===========================================================================
// QEMU Availability Helpers
// ===========================================================================

/// Map a target triple to the corresponding QEMU user-mode binary name.
///
/// # Arguments
///
/// * `target` - A target triple string (e.g., `"x86_64-linux-gnu"`).
///
/// # Returns
///
/// The QEMU binary name as a static string (e.g., `"qemu-x86_64"`).
///
/// # Panics
///
/// Panics with a descriptive message for unknown or unsupported target triples.
#[allow(dead_code)]
pub fn qemu_binary_for_target(target: &str) -> &'static str {
    if target.starts_with("x86_64") {
        "qemu-x86_64"
    } else if target.starts_with("i686") || target.starts_with("i386") {
        "qemu-i386"
    } else if target.starts_with("aarch64") {
        "qemu-aarch64"
    } else if target.starts_with("riscv64") {
        "qemu-riscv64"
    } else {
        panic!(
            "Unknown target triple '{}': cannot determine QEMU binary. \
             Supported targets: x86_64-linux-gnu, i686-linux-gnu, \
             aarch64-linux-gnu, riscv64-linux-gnu",
            target
        );
    }
}

/// Check whether the QEMU user-mode emulator is available for a given target.
///
/// Attempts to run `<qemu-binary> --version` and checks whether the command
/// succeeds. Returns `false` gracefully if QEMU is not installed, rather than
/// panicking.
///
/// # Arguments
///
/// * `target` - A target triple string.
///
/// # Returns
///
/// `true` if the QEMU emulator is installed and responds to `--version`;
/// `false` otherwise.
#[allow(dead_code)]
pub fn is_qemu_available(target: &str) -> bool {
    let qemu_name = qemu_binary_for_target(target);
    Command::new(qemu_name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Check whether the given target triple matches the host architecture.
///
/// Uses `std::env::consts::ARCH` to determine the host architecture and
/// compares it against the target triple.
///
/// # Arguments
///
/// * `target` - A target triple string (e.g., `"x86_64-linux-gnu"`).
///
/// # Returns
///
/// `true` if the target matches the host architecture; `false` otherwise.
#[allow(dead_code)]
pub fn is_native_target(target: &str) -> bool {
    let host_arch = env::consts::ARCH;
    match host_arch {
        "x86_64" => target.starts_with("x86_64"),
        "x86" => target.starts_with("i686") || target.starts_with("i386"),
        "aarch64" => target.starts_with("aarch64"),
        "riscv64" | "riscv64gc" => target.starts_with("riscv64"),
        _ => false,
    }
}

// ===========================================================================
// Assertion Macros
// ===========================================================================

/// Assert that a compilation result represents a successful compilation.
///
/// # Usage
/// ```ignore
/// let result = common::compile_source("int main() { return 0; }", &[]);
/// assert_compile_success!(result);
/// ```
#[macro_export]
macro_rules! assert_compile_success {
    ($result:expr) => {
        assert!(
            $result.success,
            "Compilation failed:\nstderr: {}\nstdout: {}",
            $result.stderr, $result.stdout
        );
    };
}

/// Assert that a compilation result represents a compilation error.
///
/// # Usage
/// ```ignore
/// let result = common::compile_source("int main( { return 0; }", &[]);
/// assert_compile_error!(result);
/// ```
#[macro_export]
macro_rules! assert_compile_error {
    ($result:expr) => {
        assert!(
            !$result.success,
            "Expected compilation error but succeeded:\nstdout: {}",
            $result.stdout
        );
    };
}

/// Assert that a compilation error occurred and its stderr contains the expected text.
///
/// # Usage
/// ```ignore
/// let result = common::compile_source("int main( { return 0; }", &[]);
/// assert_compile_error_contains!(result, "expected ')'");
/// ```
#[macro_export]
macro_rules! assert_compile_error_contains {
    ($result:expr, $expected:expr) => {
        assert!(
            !$result.success,
            "Expected compilation error but succeeded"
        );
        assert!(
            $result.stderr.contains($expected),
            "Expected error message to contain '{}', got:\n{}",
            $expected,
            $result.stderr
        );
    };
}

/// Assert that a run result represents a successful execution (exit code 0).
///
/// # Usage
/// ```ignore
/// let result = common::compile_and_run(src, "x86_64-linux-gnu", &[]);
/// assert_run_success!(result);
/// ```
#[macro_export]
macro_rules! assert_run_success {
    ($result:expr) => {
        assert!(
            $result.success,
            "Execution failed:\nstderr: {}\nstdout: {}",
            $result.stderr, $result.stdout
        );
    };
}

/// Assert that a run result's stdout contains the expected text.
///
/// # Usage
/// ```ignore
/// let result = common::compile_and_run(src, "x86_64-linux-gnu", &[]);
/// assert_output_contains!(result, "Hello, World!");
/// ```
#[macro_export]
macro_rules! assert_output_contains {
    ($result:expr, $expected:expr) => {
        assert!(
            $result.stdout.contains($expected),
            "Expected output to contain '{}', got:\n{}",
            $expected,
            $result.stdout
        );
    };
}
