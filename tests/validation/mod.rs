//! Validation suite coordinator for real-world codebase testing.
//!
//! This module provides shared infrastructure for all validation test modules
//! (SQLite, Lua, zlib, Redis). It includes source fetching utilities,
//! compilation orchestration, test execution helpers, and result reporting.
//!
//! # Validation Autonomy
//!
//! All validation runs execute autonomously with no human intervention.
//! Source trees are fetched at validation time and are **never** committed
//! to the repository. Tests skip gracefully when the network is unavailable.
//!
//! # Cross-Architecture Support
//!
//! Handles all four target architectures:
//! - `x86_64-linux-gnu`  (ELF64, System V AMD64 ABI)
//! - `i686-linux-gnu`    (ELF32, cdecl ABI)
//! - `aarch64-linux-gnu` (ELF64, AAPCS64 ABI)
//! - `riscv64-linux-gnu` (ELF64, LP64D ABI)
//!
//! Non-native binaries are executed via QEMU user-mode emulation.

// Allow dead code on all public items — not every submodule uses every utility.
#![allow(dead_code)]
#![allow(unused_imports)]

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// =========================================================================
// Sub-module declarations for validation test suites
// =========================================================================

pub mod sqlite;
pub mod lua;
pub mod zlib;
pub mod redis;
pub mod linux;

// =========================================================================
// Target Architecture Constants
// =========================================================================

/// All four supported target architectures.
#[allow(dead_code)]
pub const ALL_TARGETS: &[&str] = &[
    "x86_64-linux-gnu",
    "i686-linux-gnu",
    "aarch64-linux-gnu",
    "riscv64-linux-gnu",
];

// =========================================================================
// Validation Result Types
// =========================================================================

/// Result of a validation run for a single target/architecture combination.
///
/// Captures compilation success, test suite results, timing information,
/// and error details for reporting in the validation summary.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ValidationResult {
    /// Human-readable name for the validation target (e.g., "SQLite", "Lua").
    pub target_name: String,
    /// Target architecture triple (e.g., "x86_64-linux-gnu").
    pub architecture: String,
    /// Whether compilation of all required source files succeeded.
    pub compilation_success: bool,
    /// Whether the test suite passed. `None` for compile-only targets (Redis).
    pub test_suite_success: Option<bool>,
    /// Total wall-clock time spent on compilation.
    pub compile_time: Duration,
    /// Number of source files that compiled successfully.
    pub files_compiled: usize,
    /// Number of source files that failed to compile.
    pub files_failed: usize,
    /// Collection of error messages encountered during validation.
    pub errors: Vec<String>,
}

/// Result of attempting to fetch a source archive from the internet.
///
/// Encodes the three possible outcomes: successful download, network
/// unavailability, and extraction failure.
#[derive(Debug)]
#[allow(dead_code)]
pub enum SourceFetchResult {
    /// Download succeeded; contains the path to the downloaded archive file.
    Success(PathBuf),
    /// Network is unavailable or download failed; contains error description.
    NetworkUnavailable(String),
    /// Archive download succeeded but extraction failed; contains error description.
    ExtractionFailed(String),
}

// =========================================================================
// Atomic Counter for Unique Directory Names
// =========================================================================

/// Global atomic counter for generating unique work directory names across
/// parallel test execution. Incremented via `fetch_add(Ordering::SeqCst)`
/// to ensure no naming collisions when `cargo test` runs multiple
/// validation tests concurrently.
static WORK_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

// =========================================================================
// Source Fetching Utilities
// =========================================================================

/// Download a source archive from `url` into `work_dir` with the given
/// `archive_name`.
///
/// Attempts `curl` first (more commonly available on Linux, follows
/// redirects via `-L`), falling back to `wget` if `curl` is unavailable
/// or fails. Sets a 300-second timeout on downloads to prevent tests
/// from hanging indefinitely on slow connections.
///
/// # Returns
///
/// - `SourceFetchResult::Success(path)` with the path to the downloaded
///   archive file on success.
/// - `SourceFetchResult::NetworkUnavailable(msg)` if both download
///   methods fail.
#[allow(dead_code)]
pub fn fetch_source_archive(url: &str, work_dir: &Path, archive_name: &str) -> SourceFetchResult {
    let output_path = work_dir.join(archive_name);
    let output_str = output_path.display().to_string();

    // Attempt download with curl first (preferred: follows redirects, silent
    // mode with fail-on-HTTP-error).
    let curl_result = Command::new("curl")
        .args(&[
            "-sSfL",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "-o",
            &output_str,
            url,
        ])
        .output();

    match curl_result {
        Ok(output) if output.status.success() && output_path.exists() => {
            return SourceFetchResult::Success(output_path);
        }
        _ => {
            // curl failed or is unavailable; try wget as fallback.
        }
    }

    // Fallback: attempt download with wget.
    let wget_result = Command::new("wget")
        .args(&["-q", "--timeout=30", "-O", &output_str, url])
        .output();

    match wget_result {
        Ok(output) if output.status.success() && output_path.exists() => {
            SourceFetchResult::Success(output_path)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            SourceFetchResult::NetworkUnavailable(format!(
                "Both curl and wget failed to download '{}'. wget stderr: {}",
                url, stderr
            ))
        }
        Err(e) => SourceFetchResult::NetworkUnavailable(format!(
            "Both curl and wget are unavailable or failed for '{}': {}",
            url, e
        )),
    }
}

/// Extract a `.tar.gz` (or `.tgz`) archive into the destination directory.
///
/// After extraction, locates the top-level directory created by `tar`
/// (usually named after the project, e.g., `lua-5.4.7/`). Returns the
/// path to that directory.
///
/// # Errors
///
/// Returns a descriptive error string if `tar` invocation fails or the
/// extracted directory cannot be found.
#[allow(dead_code)]
pub fn extract_tarball(archive: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    let archive_str = archive.display().to_string();
    let dest_str = dest_dir.display().to_string();

    // Collect directory entries BEFORE extraction to identify new directories.
    let pre_existing: std::collections::HashSet<PathBuf> = fs::read_dir(dest_dir)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();

    let output = Command::new("tar")
        .args(&["xzf", &archive_str, "-C", &dest_str])
        .output()
        .map_err(|e| format!("Failed to invoke tar: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("tar extraction failed: {}", stderr));
    }

    // Find the newly created top-level directory.
    if let Ok(entries) = fs::read_dir(dest_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && !pre_existing.contains(&path) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') {
                    return Ok(path);
                }
            }
        }
    }

    // Fallback: return any non-hidden directory (for cases where
    // pre-extraction directory listing failed).
    if let Ok(entries) = fs::read_dir(dest_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') {
                    return Ok(path);
                }
            }
        }
    }

    // If no subdirectory was found, assume files were extracted directly
    // into dest_dir.
    Ok(dest_dir.to_path_buf())
}

/// Extract a `.zip` archive into the destination directory.
///
/// Uses the system `unzip` command. After extraction, locates the
/// top-level directory created. Returns its path.
///
/// # Errors
///
/// Returns a descriptive error string if `unzip` invocation fails or
/// the extracted directory cannot be found.
#[allow(dead_code)]
pub fn extract_zip(archive: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    let archive_str = archive.display().to_string();
    let dest_str = dest_dir.display().to_string();

    // Collect directory entries BEFORE extraction.
    let pre_existing: std::collections::HashSet<PathBuf> = fs::read_dir(dest_dir)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();

    let output = Command::new("unzip")
        .args(&["-q", "-o", &archive_str, "-d", &dest_str])
        .output()
        .map_err(|e| format!("Failed to invoke unzip: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("unzip extraction failed: {}", stderr));
    }

    // Find the newly created top-level directory.
    if let Ok(entries) = fs::read_dir(dest_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && !pre_existing.contains(&path) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') {
                    return Ok(path);
                }
            }
        }
    }

    // Fallback: return any non-hidden directory.
    if let Ok(entries) = fs::read_dir(dest_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with('.') {
                    return Ok(path);
                }
            }
        }
    }

    // Files extracted directly into dest_dir.
    Ok(dest_dir.to_path_buf())
}

/// Check whether the network is available by attempting a lightweight
/// HTTP connection.
///
/// Tries to reach a well-known host with a short timeout. Used to
/// decide whether to skip validation tests that require downloading
/// source archives.
///
/// # Returns
///
/// `true` if the network appears available, `false` otherwise.
#[allow(dead_code)]
pub fn is_network_available() -> bool {
    // Try curl first with a 5-second connect timeout.
    let curl_ok = Command::new("curl")
        .args(&[
            "-sSf",
            "--connect-timeout",
            "5",
            "--max-time",
            "10",
            "-o",
            "/dev/null",
            "https://www.google.com/",
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if curl_ok {
        return true;
    }

    // Fallback: try wget with similar timeout.
    Command::new("wget")
        .args(&["-q", "--timeout=5", "--spider", "https://www.google.com/"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// =========================================================================
// Compilation Orchestration Utilities
// =========================================================================

/// Locate the `bcc` compiler binary built by cargo.
///
/// First attempts to use the `CARGO_BIN_EXE_bcc` environment variable
/// set by cargo during `cargo test`. Falls back to searching common
/// build output directories relative to `CARGO_MANIFEST_DIR`.
///
/// # Panics
///
/// Panics with a descriptive message if the binary cannot be found.
/// This is intentional: tests cannot proceed without the compiler.
#[allow(dead_code)]
pub fn get_bcc_binary() -> PathBuf {
    // Attempt 1: Use cargo's integration test binary path.
    if let Ok(path) = env::var("CARGO_BIN_EXE_bcc") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return p;
        }
    }

    // Attempt 2: Derive path from CARGO_MANIFEST_DIR.
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let base = PathBuf::from(&manifest_dir);

        // Check debug build first (most common during development).
        let debug_path = base.join("target").join("debug").join("bcc");
        if debug_path.exists() {
            return debug_path;
        }

        // Check release build.
        let release_path = base.join("target").join("release").join("bcc");
        if release_path.exists() {
            return release_path;
        }
    }

    // Attempt 3: Relative paths from current directory.
    let cwd_debug = PathBuf::from("target/debug/bcc");
    if cwd_debug.exists() {
        return cwd_debug;
    }

    let cwd_release = PathBuf::from("target/release/bcc");
    if cwd_release.exists() {
        return cwd_release;
    }

    panic!(
        "Could not locate the bcc binary. Ensure `cargo build` has been \
         run before executing validation tests. Checked: \
         CARGO_BIN_EXE_bcc, target/debug/bcc, target/release/bcc"
    );
}

/// Compile a single C source file to an object file using the `bcc` compiler.
///
/// Invokes `bcc --target {target} -c -o {output} {flags...} {source}`.
/// Captures stdout and stderr for the caller to inspect.
///
/// # Arguments
///
/// * `source`  — Path to the `.c` source file.
/// * `target`  — Target triple (e.g., `"x86_64-linux-gnu"`).
/// * `output`  — Path for the output `.o` object file.
/// * `flags`   — Additional compiler flags (e.g., `["-O0", "-DFOO"]`).
///
/// # Returns
///
/// `Ok(Output)` containing the process exit status, stdout, and stderr.
/// `Err(String)` if the `bcc` binary could not be invoked at all.
#[allow(dead_code)]
pub fn compile_c_file(
    source: &Path,
    target: &str,
    output: &Path,
    flags: &[&str],
) -> Result<Output, String> {
    let bcc = get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-c");
    cmd.arg("-o").arg(output);

    for flag in flags {
        cmd.arg(flag);
    }

    cmd.arg(source);

    cmd.output()
        .map_err(|e| format!("Failed to invoke bcc at '{}': {}", bcc.display(), e))
}

/// Compile multiple source files and link them into a single output binary.
///
/// Each source file is first compiled to a temporary `.o` file, then all
/// `.o` files are linked together into the final output. This two-step
/// process mirrors the standard `cc *.c -o output` workflow.
///
/// # Arguments
///
/// * `sources` — List of `.c` source file paths.
/// * `target`  — Target triple.
/// * `output`  — Path for the final linked output binary.
/// * `flags`   — Additional compiler/linker flags.
///
/// # Returns
///
/// `Ok(Output)` from the linking step on success.
/// `Err(String)` if compilation or linking fails.
#[allow(dead_code)]
pub fn compile_and_link(
    sources: &[PathBuf],
    target: &str,
    output: &Path,
    flags: &[&str],
) -> Result<Output, String> {
    let bcc = get_bcc_binary();
    let work_dir = output.parent().unwrap_or_else(|| Path::new("."));

    // Step 1: Compile each source to a .o file.
    let mut object_files: Vec<PathBuf> = Vec::new();
    for source in sources {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("obj");
        let obj_path = work_dir.join(format!("{}.o", stem));

        let compile_output = compile_c_file(source, target, &obj_path, flags)?;
        if !compile_output.status.success() {
            let stderr = String::from_utf8_lossy(&compile_output.stderr);
            return Err(format!(
                "Compilation failed for '{}': {}",
                source.display(),
                stderr
            ));
        }
        object_files.push(obj_path);
    }

    // Step 2: Link all .o files together.
    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-o").arg(output);

    for flag in flags {
        cmd.arg(flag);
    }

    for obj in &object_files {
        cmd.arg(obj);
    }

    cmd.output()
        .map_err(|e| format!("Failed to invoke bcc linker: {}", e))
}

/// Compile multiple source files individually, tolerating partial failures.
///
/// Each source file is compiled to a `.o` object file in `output_dir`.
/// Unlike `compile_and_link`, this function continues compiling remaining
/// files even when some fail, collecting successes and failures separately.
///
/// # Arguments
///
/// * `sources`    — List of `.c` source file paths.
/// * `target`     — Target triple.
/// * `output_dir` — Directory for output `.o` files.
/// * `flags`      — Additional compiler flags.
///
/// # Returns
///
/// A tuple of:
/// - `Vec<PathBuf>` — Paths to successfully compiled `.o` files.
/// - `Vec<(PathBuf, String)>` — Failed source files with error messages.
#[allow(dead_code)]
pub fn compile_multiple_files(
    sources: &[PathBuf],
    target: &str,
    output_dir: &Path,
    flags: &[&str],
) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
    let _ = fs::create_dir_all(output_dir);

    let mut successes: Vec<PathBuf> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    for source in sources {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("obj");
        let obj_path = output_dir.join(format!("{}.o", stem));

        match compile_c_file(source, target, &obj_path, flags) {
            Ok(output) if output.status.success() => {
                successes.push(obj_path);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                failures.push((source.clone(), stderr));
            }
            Err(e) => {
                failures.push((source.clone(), e));
            }
        }
    }

    (successes, failures)
}

// =========================================================================
// Execution Utilities
// =========================================================================

/// Execute a compiled binary, using QEMU for cross-architecture targets.
///
/// If the `target` matches the host architecture, the binary is run
/// directly. Otherwise, the appropriate QEMU user-mode emulator is
/// invoked to execute the binary.
///
/// # Arguments
///
/// * `binary` — Path to the compiled ELF executable.
/// * `target` — Target triple the binary was compiled for.
/// * `args`   — Command-line arguments to pass to the binary.
///
/// # Returns
///
/// `Ok(Output)` containing exit status, stdout, and stderr.
/// `Err(String)` if the binary or QEMU could not be invoked.
#[allow(dead_code)]
pub fn run_binary(binary: &Path, target: &str, args: &[&str]) -> Result<Output, String> {
    if is_native_target(target) {
        // Run directly on the host.
        Command::new(binary)
            .args(args)
            .output()
            .map_err(|e| format!("Failed to run '{}': {}", binary.display(), e))
    } else {
        // Run via QEMU user-mode emulation.
        let qemu = qemu_binary_for_target(target);

        if !is_qemu_available(target) {
            return Err(format!(
                "QEMU binary '{}' not available for target '{}'",
                qemu, target
            ));
        }

        Command::new(qemu)
            .arg(binary)
            .args(args)
            .output()
            .map_err(|e| {
                format!(
                    "Failed to run '{}' via QEMU '{}': {}",
                    binary.display(),
                    qemu,
                    e
                )
            })
    }
}

/// Check whether the QEMU user-mode emulator is available for a target.
///
/// Attempts to invoke `{qemu_binary} --version` and checks for a
/// successful exit code.
#[allow(dead_code)]
pub fn is_qemu_available(target: &str) -> bool {
    let qemu = qemu_binary_for_target(target);
    Command::new(qemu)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Map a target triple to the corresponding QEMU user-mode binary name.
///
/// # Panics
///
/// Panics on an unrecognized target triple.
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
        panic!("Unknown target triple for QEMU mapping: '{}'", target);
    }
}

/// Check whether a target triple matches the host architecture.
///
/// Compares `std::env::consts::ARCH` with the architecture component
/// of the target triple to determine if the target is native.
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

// =========================================================================
// Temporary Directory Management
// =========================================================================

/// Create a uniquely named validation work directory under the system
/// temp directory.
///
/// Directory name format: `bcc_validation_{name}_{pid}_{counter}`
/// where the atomic counter ensures uniqueness across concurrent test
/// threads.
///
/// # Panics
///
/// Panics if the directory cannot be created.
#[allow(dead_code)]
pub fn create_validation_work_dir(name: &str) -> PathBuf {
    let counter = WORK_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir_name = format!("bcc_validation_{}_{}_{}", name, pid, counter);
    let path = env::temp_dir().join(dir_name);

    fs::create_dir_all(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to create validation work directory '{}': {}",
            path.display(),
            e
        );
    });

    path
}

/// Remove a work directory and all its contents (best-effort cleanup).
///
/// Errors are silently ignored — this is a cleanup utility, and
/// failure to clean up temporary files should not cause test failures.
#[allow(dead_code)]
pub fn cleanup_work_dir(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// RAII wrapper around a validation work directory.
///
/// Creates the directory on construction and removes it (with all
/// contents) when dropped. Provides a `path()` accessor for the
/// directory path.
#[allow(dead_code)]
pub struct ValidationWorkDir {
    dir_path: PathBuf,
}

#[allow(dead_code)]
impl ValidationWorkDir {
    /// Create a new validation work directory with the given name prefix.
    pub fn new(name: &str) -> Self {
        let dir_path = create_validation_work_dir(name);
        ValidationWorkDir { dir_path }
    }

    /// Return a borrowed reference to the work directory path.
    pub fn path(&self) -> &Path {
        &self.dir_path
    }
}

impl Drop for ValidationWorkDir {
    fn drop(&mut self) {
        cleanup_work_dir(&self.dir_path);
    }
}

// =========================================================================
// Performance Measurement Utilities
// =========================================================================

/// Measure the wall-clock time to compile a single source file.
///
/// Records `Instant::now()` before invoking `compile_c_file` and
/// returns the elapsed `Duration` alongside the compilation success
/// status.
///
/// # Arguments
///
/// * `source` — Path to the `.c` source file.
/// * `target` — Target triple.
/// * `flags`  — Additional compiler flags.
///
/// # Returns
///
/// `(success: bool, elapsed: Duration)`.
#[allow(dead_code)]
pub fn measure_compilation_time(source: &Path, target: &str, flags: &[&str]) -> (bool, Duration) {
    let output_dir = env::temp_dir();
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("measure_obj");
    let obj_path = output_dir.join(format!("bcc_measure_{}.o", stem));

    let start = Instant::now();
    let success = match compile_c_file(source, target, &obj_path, flags) {
        Ok(output) => output.status.success(),
        Err(_) => false,
    };
    let elapsed = start.elapsed();

    // Clean up the temporary object file.
    let _ = fs::remove_file(&obj_path);

    (success, elapsed)
}

/// Measure the peak resident set size (RSS) of a command execution.
///
/// Executes the command and returns the output. For precise RSS
/// measurement, callers should use `measure_peak_rss_with_time()`
/// which wraps the invocation with `/usr/bin/time -v`.
///
/// # Arguments
///
/// * `cmd` — A mutable `Command` reference to execute.
///
/// # Returns
///
/// `Ok((Output, rss_bytes))` where `rss_bytes` is the peak RSS in
/// bytes. Returns 0 for RSS when direct measurement is not possible
/// through this interface.
/// `Err(String)` if command execution fails.
#[allow(dead_code)]
pub fn measure_peak_rss_of_command(cmd: &mut Command) -> Result<(Output, u64), String> {
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute command: {}", e))?;

    // Direct RSS measurement from a completed process is not possible
    // without /usr/bin/time wrapping. Return 0 as a sentinel value.
    // Callers needing precise RSS should use measure_peak_rss_with_time().
    Ok((output, 0))
}

/// Measure peak RSS using `/usr/bin/time -v` wrapper for a bcc
/// compilation command.
///
/// Wraps the entire compilation invocation with `/usr/bin/time -v` and
/// parses the "Maximum resident set size" from its stderr output.
///
/// # Arguments
///
/// * `bcc_path`   — Path to the bcc binary.
/// * `bcc_args`   — Arguments to pass to bcc.
///
/// # Returns
///
/// Peak RSS in bytes, or 0 if measurement fails.
#[allow(dead_code)]
pub fn measure_peak_rss_with_time(bcc_path: &Path, bcc_args: &[String]) -> u64 {
    let time_binary = PathBuf::from("/usr/bin/time");
    if !time_binary.exists() {
        return 0;
    }

    let mut cmd = Command::new(&time_binary);
    cmd.arg("-v");
    cmd.arg(bcc_path);
    for arg in bcc_args {
        cmd.arg(arg);
    }

    let output = match cmd.output() {
        Ok(o) => o,
        Err(_) => return 0,
    };

    // Parse "Maximum resident set size (kbytes): NNNN" from stderr.
    // /usr/bin/time -v writes its output to stderr.
    let stderr = String::from_utf8_lossy(&output.stderr);
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Maximum resident set size") {
            if let Some(colon_pos) = trimmed.rfind(':') {
                let num_str = trimmed[colon_pos + 1..].trim();
                if let Ok(kb) = num_str.parse::<u64>() {
                    // Convert from kilobytes to bytes.
                    return kb * 1024;
                }
            }
        }
    }

    0
}

// =========================================================================
// Result Reporting
// =========================================================================

/// Print a formatted validation summary table to stderr.
///
/// Uses `eprintln!` to avoid interfering with cargo test's stdout
/// capture. The table shows compilation status, test results, timing,
/// and file counts for each validation target/architecture combination.
///
/// # Arguments
///
/// * `results` — Slice of `ValidationResult` entries to summarize.
#[allow(dead_code)]
pub fn print_validation_summary(results: &[ValidationResult]) {
    eprintln!();
    eprintln!("=== Validation Results ===");
    eprintln!(
        "{:<12} | {:<12} | {:<8} | {:<6} | {:<10} | {}",
        "Target", "Arch", "Compile", "Tests", "Time", "Files"
    );
    eprintln!(
        "{:<12}-+-{:<12}-+-{:<8}-+-{:<6}-+-{:<10}-+-{}",
        "------------",
        "------------",
        "--------",
        "------",
        "----------",
        "----------"
    );

    for result in results {
        let compile_status = if result.compilation_success {
            "PASS"
        } else {
            "FAIL"
        };

        let test_status = match result.test_suite_success {
            Some(true) => "PASS".to_string(),
            Some(false) => "FAIL".to_string(),
            None => "N/A".to_string(),
        };

        let time_str = format!("{:.1}s", result.compile_time.as_secs_f64());

        let total_files = result.files_compiled + result.files_failed;
        let files_str = format!("{}/{}", result.files_compiled, total_files);

        eprintln!(
            "{:<12} | {:<12} | {:<8} | {:<6} | {:<10} | {}",
            result.target_name,
            result.architecture,
            compile_status,
            test_status,
            time_str,
            files_str
        );
    }

    // Print error summary if any errors occurred.
    let total_errors: usize = results.iter().map(|r| r.errors.len()).sum();
    if total_errors > 0 {
        eprintln!();
        eprintln!("--- Errors ({} total) ---", total_errors);
        for result in results {
            for error in &result.errors {
                eprintln!(
                    "  [{}:{}] {}",
                    result.target_name, result.architecture, error
                );
            }
        }
    }

    eprintln!();
}

// =========================================================================
// Network Skip Helper
// =========================================================================

/// Check if the network is available and skip the test if it is not.
///
/// Returns `true` if the test should be skipped (network unavailable),
/// `false` if the test should proceed.
///
/// When returning `true`, prints a SKIPPED message to stderr
/// identifying the test that was skipped.
#[allow(dead_code)]
pub fn skip_if_offline(test_name: &str) -> bool {
    if !is_network_available() {
        eprintln!("SKIPPED: {} (network unavailable)", test_name);
        return true;
    }
    false
}
