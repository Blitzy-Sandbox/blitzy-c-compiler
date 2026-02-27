//! Validation suite coordinator for real-world codebase testing.
//!
//! Provides shared infrastructure for all validation test modules
//! (SQLite, Lua, zlib, Redis).

#![allow(dead_code)]
#![allow(unused_imports)]


use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// Sub-module declarations for validation test suites.
// Each sub-module tests compilation and (optionally) execution
// of a real-world C codebase.
#[path = "validation/lua.rs"]
pub mod lua;
#[path = "validation/redis.rs"]
pub mod redis;
#[path = "validation/sqlite.rs"]
pub mod sqlite;
#[path = "validation/zlib.rs"]
pub mod zlib;

// =========================================================================
// Target Architecture Constants
// =========================================================================

/// All four supported target architectures.
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
#[derive(Debug)]
pub struct ValidationResult {
    pub target_name: String,
    pub architecture: String,
    pub compilation_success: bool,
    pub test_suite_success: Option<bool>,
    pub compile_time: Duration,
    pub files_compiled: usize,
    pub files_failed: usize,
    pub errors: Vec<String>,
}

/// Outcome of attempting to fetch a source archive.
#[derive(Debug)]
pub enum SourceFetchResult {
    Success(PathBuf),
    NetworkUnavailable(String),
    ExtractionFailed(String),
}

// =========================================================================
// Validation Work Directory (RAII)
// =========================================================================

static WORK_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// RAII wrapper around a temporary directory used for validation work.
/// Automatically cleaned up when dropped.
pub struct ValidationWorkDir {
    path: PathBuf,
}

impl ValidationWorkDir {
    /// Create a new unique work directory under the system temp directory.
    pub fn new(name: &str) -> Self {
        let counter = WORK_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = env::temp_dir().join(format!(
            "bcc_validation_{}_{}_{}",
            name, pid, counter
        ));
        let _ = fs::create_dir_all(&dir);
        Self { path: dir }
    }

    /// Return a reference to the work directory path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ValidationWorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

// =========================================================================
// Source Fetching Utilities
// =========================================================================

/// Fetch a source archive from `url` into `work_dir`, saving it as
/// `archive_name`.
pub fn fetch_source_archive(
    url: &str,
    work_dir: &Path,
    archive_name: &str,
) -> SourceFetchResult {
    let dest = work_dir.join(archive_name);

    // Try curl first.
    let curl_status = Command::new("curl")
        .args(&["-sSfL", "--connect-timeout", "30", "--max-time", "300", "-o"])
        .arg(&dest)
        .arg(url)
        .status();

    match curl_status {
        Ok(status) if status.success() => return SourceFetchResult::Success(dest),
        _ => {}
    }

    // Fallback to wget.
    let wget_status = Command::new("wget")
        .args(&["-q", "--timeout=30", "-O"])
        .arg(&dest)
        .arg(url)
        .status();

    match wget_status {
        Ok(status) if status.success() => SourceFetchResult::Success(dest),
        _ => SourceFetchResult::NetworkUnavailable(format!(
            "Failed to download {} with both curl and wget",
            url
        )),
    }
}

/// Extract a `.tar.gz` archive into the destination directory.
pub fn extract_tarball(archive: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    let status = Command::new("tar")
        .args(&["xzf"])
        .arg(archive)
        .arg("-C")
        .arg(dest_dir)
        .status()
        .map_err(|e| format!("Failed to invoke tar: {}", e))?;

    if !status.success() {
        return Err("tar extraction failed".to_string());
    }

    // Find the top-level directory that was extracted.
    if let Ok(entries) = fs::read_dir(dest_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() && entry.file_name() != "." && entry.file_name() != ".." {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    return Ok(entry.path());
                }
            }
        }
    }

    Err("Could not find extracted directory".to_string())
}

/// Extract a `.zip` archive into the destination directory.
pub fn extract_zip(archive: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    let status = Command::new("unzip")
        .args(&["-q", "-o"])
        .arg(archive)
        .arg("-d")
        .arg(dest_dir)
        .status()
        .map_err(|e| format!("Failed to invoke unzip: {}", e))?;

    if !status.success() {
        return Err("unzip extraction failed".to_string());
    }

    if let Ok(entries) = fs::read_dir(dest_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    return Ok(entry.path());
                }
            }
        }
    }

    Err("Could not find extracted directory".to_string())
}

/// Check whether the network is reachable (best-effort probe).
pub fn is_network_available() -> bool {
    Command::new("curl")
        .args(&["-sSf", "--connect-timeout", "5", "https://www.google.com/"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// =========================================================================
// Compiler Invocation Utilities
// =========================================================================

/// Locate the `bcc` compiler binary built by Cargo.
pub fn get_bcc_binary() -> PathBuf {
    // Try the CARGO_BIN_EXE_bcc env var first (set by cargo test).
    if let Ok(path) = env::var("CARGO_BIN_EXE_bcc") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return p;
        }
    }

    // Fallback: search relative to CARGO_MANIFEST_DIR.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());

    for profile in &["debug", "release"] {
        let candidate = PathBuf::from(&manifest_dir)
            .join("target")
            .join(profile)
            .join("bcc");
        if candidate.exists() {
            return candidate;
        }
    }

    // Last resort: hope it's on PATH.
    PathBuf::from("bcc")
}

/// Compile a single C source file using `bcc`.
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
        .map_err(|e| format!("Failed to execute bcc: {}", e))
}

/// Compile and link multiple source files into a final binary.
pub fn compile_and_link(
    sources: &[PathBuf],
    target: &str,
    output: &Path,
    flags: &[&str],
) -> Result<Output, String> {
    let bcc = get_bcc_binary();
    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-o").arg(output);
    for flag in flags {
        cmd.arg(flag);
    }
    for src in sources {
        cmd.arg(src);
    }
    cmd.output()
        .map_err(|e| format!("Failed to execute bcc: {}", e))
}

/// Compile multiple C files individually, returning successes and failures.
pub fn compile_multiple_files(
    sources: &[PathBuf],
    target: &str,
    output_dir: &Path,
    flags: &[&str],
) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    let _ = fs::create_dir_all(output_dir);

    for src in sources {
        let stem = src
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let obj = output_dir.join(format!("{}.o", stem));

        match compile_c_file(src, target, &obj, flags) {
            Ok(output) if output.status.success() => {
                successes.push(obj);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                failures.push((src.clone(), stderr));
            }
            Err(e) => {
                failures.push((src.clone(), e));
            }
        }
    }

    (successes, failures)
}

// =========================================================================
// Execution Utilities
// =========================================================================

/// Run a compiled binary, using QEMU for non-native architectures.
pub fn run_binary(
    binary: &Path,
    target: &str,
    args: &[&str],
) -> Result<Output, String> {
    if is_native_target(target) {
        Command::new(binary)
            .args(args)
            .output()
            .map_err(|e| format!("Failed to run binary: {}", e))
    } else {
        let qemu = qemu_binary_for_target(target);
        if !is_qemu_available(target) {
            return Err(format!("QEMU ({}) not available for {}", qemu, target));
        }
        Command::new(qemu)
            .arg(binary)
            .args(args)
            .output()
            .map_err(|e| format!("Failed to run via QEMU: {}", e))
    }
}

/// Check whether QEMU user-mode emulation is available for the given target.
pub fn is_qemu_available(target: &str) -> bool {
    let qemu = qemu_binary_for_target(target);
    Command::new(qemu)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Map a target triple to the corresponding QEMU binary name.
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
        panic!("Unknown target for QEMU: {}", target)
    }
}

/// Check whether the given target matches the host architecture.
pub fn is_native_target(target: &str) -> bool {
    let host_arch = env::consts::ARCH;
    match host_arch {
        "x86_64" => target.starts_with("x86_64"),
        "x86" => target.starts_with("i686") || target.starts_with("i386"),
        "aarch64" => target.starts_with("aarch64"),
        "riscv64" => target.starts_with("riscv64"),
        _ => false,
    }
}

// =========================================================================
// Temporary Directory Helpers
// =========================================================================

/// Create a validation work directory with a unique name.
pub fn create_validation_work_dir(name: &str) -> PathBuf {
    let counter = WORK_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = env::temp_dir().join(format!(
        "bcc_validation_{}_{}_{}",
        name, pid, counter
    ));
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Remove a validation work directory (best-effort).
pub fn cleanup_work_dir(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

// =========================================================================
// Performance Measurement Utilities
// =========================================================================

/// Time a single compilation and return `(success, duration)`.
pub fn measure_compilation_time(
    source: &Path,
    target: &str,
    flags: &[&str],
) -> (bool, Duration) {
    let bcc = get_bcc_binary();
    let out = env::temp_dir().join("bcc_timing_output.o");

    let start = Instant::now();
    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-c");
    cmd.arg("-o").arg(&out);
    for f in flags {
        cmd.arg(f);
    }
    cmd.arg(source);
    let success = cmd.status().map(|s| s.success()).unwrap_or(false);
    let elapsed = start.elapsed();

    let _ = fs::remove_file(&out);
    (success, elapsed)
}

/// Measure peak RSS of a command by wrapping it with `/usr/bin/time -v`.
pub fn measure_peak_rss_of_command(
    cmd: &mut Command,
) -> Result<(Output, u64), String> {
    // Use /usr/bin/time to capture peak RSS.
    let time_output = Command::new("/usr/bin/time")
        .arg("-v")
        .args(cmd.get_args())
        .output()
        .map_err(|e| format!("Failed to run /usr/bin/time: {}", e))?;

    let stderr = String::from_utf8_lossy(&time_output.stderr);
    let mut peak_rss_kb: u64 = 0;

    for line in stderr.lines() {
        if line.contains("Maximum resident set size") {
            if let Some(val) = line.split(':').last() {
                if let Ok(kb) = val.trim().parse::<u64>() {
                    peak_rss_kb = kb;
                }
            }
        }
    }

    Ok((time_output, peak_rss_kb * 1024))
}

// =========================================================================
// Result Reporting
// =========================================================================

/// Print a formatted summary of validation results.
pub fn print_validation_summary(results: &[ValidationResult]) {
    eprintln!("=== Validation Results ===");
    eprintln!(
        "{:<14} {:<10} {:<9} {:<7} {:<10} {:<8}",
        "Target", "Arch", "Compile", "Tests", "Time", "Files"
    );
    eprintln!("{}", "-".repeat(62));

    for r in results {
        let tests_str = match r.test_suite_success {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "N/A",
        };
        eprintln!(
            "{:<14} {:<10} {:<9} {:<7} {:<10.1}s {}/{}",
            r.target_name,
            r.architecture,
            if r.compilation_success { "PASS" } else { "FAIL" },
            tests_str,
            r.compile_time.as_secs_f64(),
            r.files_compiled,
            r.files_compiled + r.files_failed,
        );
    }
}

// =========================================================================
// Skip Helpers
// =========================================================================

/// Check network availability and skip the test if offline.
/// Returns `true` if the test should be skipped.
pub fn skip_if_offline(test_name: &str) -> bool {
    if !is_network_available() {
        eprintln!("SKIPPED: {} (network unavailable)", test_name);
        return true;
    }
    false
}
