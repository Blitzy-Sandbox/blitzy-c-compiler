//! Linux kernel source validation tests for the bcc compiler.
//!
//! This module verifies that `bcc` can compile individual C files from the
//! Linux kernel source tree (v6.12). It clones `torvalds/linux` at tag `v6.12`
//! with `--depth=1` and compiles each `.c` file in `lib/` and `kernel/`
//! individually via `bcc -c` with standard kernel include flags.
//!
//! # Gating
//!
//! All tests in this module are gated behind `#[cfg(feature = "linux_validation")]`
//! to avoid long clone/compile times during routine CI runs.
//!
//! # Validation Criteria
//!
//! - **ICE count**: Must be exactly 0 (no internal compiler errors).
//! - **`lib/` pass rate**: At least 80% of files must compile without errors.
//! - **`kernel/` pass rate**: At least 60% of files must compile without errors.
//!
//! # Include Flags
//!
//! Standard kernel include flags used for compilation:
//! ```text
//! -I include -I arch/x86/include -I include/uapi -I arch/x86/include/uapi
//! -I arch/x86/include/generated -I arch/x86/include/generated/uapi
//! -I include/generated/uapi
//! -D__KERNEL__ -D__x86_64__ -DCONFIG_X86_64 -DCONFIG_64BIT
//! -DKBUILD_MODNAME="bcc_test" -DKBUILD_BASENAME="bcc_test"
//! -DMODULE -ffreestanding -mno-red-zone -fno-strict-aliasing
//! -fno-common -fno-delete-null-pointer-checks -ffunction-sections
//! -fdata-sections
//! ```

#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use super::{get_bcc_binary, create_validation_work_dir, is_network_available};

// =========================================================================
// Constants
// =========================================================================

/// Git repository URL for the Linux kernel.
const LINUX_REPO_URL: &str = "https://github.com/torvalds/linux.git";

/// Git tag to clone.
const LINUX_TAG: &str = "v6.12";

/// Environment variable for caching the Linux source tree.
const LINUX_CACHE_DIR_ENV: &str = "BCC_LINUX_CACHE_DIR";

/// Default cache directory name under the validation work directory.
const DEFAULT_CACHE_SUBDIR: &str = "linux-v6.12";

// =========================================================================
// Result Types
// =========================================================================

/// Result of compiling a single C file.
#[derive(Debug, Clone)]
pub struct FileCompileResult {
    /// Path relative to the kernel source root (e.g., "lib/string.c").
    pub relative_path: String,
    /// Whether compilation succeeded (exit code 0).
    pub passed: bool,
    /// Whether an internal compiler error (ICE) occurred.
    pub is_ice: bool,
    /// Stderr output from the compiler.
    pub stderr: String,
    /// Compilation duration.
    pub duration: Duration,
    /// Categorized error type, if any.
    pub error_category: Option<String>,
}

/// Summary of compilation results for a subsystem.
#[derive(Debug)]
pub struct SubsystemResult {
    /// Name of the subsystem (e.g., "lib", "kernel").
    pub name: String,
    /// Total number of files attempted.
    pub total: usize,
    /// Number of files that compiled successfully.
    pub passed: usize,
    /// Number of files that produced ICEs.
    pub ice_count: usize,
    /// Pass rate as a percentage.
    pub pass_rate: f64,
    /// Per-file results.
    pub files: Vec<FileCompileResult>,
    /// Error category breakdown.
    pub error_categories: HashMap<String, usize>,
}

// =========================================================================
// Linux Source Management
// =========================================================================

/// Gets or clones the Linux kernel source tree.
///
/// Checks `BCC_LINUX_CACHE_DIR` for a pre-existing clone. If not set,
/// clones into the validation work directory. Uses `--depth=1` and
/// `--branch v6.12` for a shallow single-tag clone.
fn get_linux_source() -> Option<PathBuf> {
    // Check for user-provided cache directory.
    if let Ok(cache_dir) = std::env::var(LINUX_CACHE_DIR_ENV) {
        let path = PathBuf::from(&cache_dir);
        if path.join("Makefile").exists() {
            eprintln!("[linux] Using cached source at {}", path.display());
            return Some(path);
        }
    }

    // Check default cache location.
    let work_dir = create_validation_work_dir("linux");
    let source_dir = work_dir.join(DEFAULT_CACHE_SUBDIR);

    if source_dir.join("Makefile").exists() {
        eprintln!("[linux] Using cached source at {}", source_dir.display());
        return Some(source_dir);
    }

    // Clone from git.
    if !is_network_available() {
        eprintln!("[linux] Network unavailable, skipping clone");
        return None;
    }

    eprintln!("[linux] Cloning {} at tag {} (shallow)...", LINUX_REPO_URL, LINUX_TAG);
    let clone_result = Command::new("git")
        .args([
            "clone",
            "--depth=1",
            "--branch", LINUX_TAG,
            "--single-branch",
            LINUX_REPO_URL,
            &source_dir.to_string_lossy(),
        ])
        .output();

    match clone_result {
        Ok(output) if output.status.success() => {
            eprintln!("[linux] Clone complete: {}", source_dir.display());
            Some(source_dir)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("[linux] Clone failed: {}", stderr);
            None
        }
        Err(e) => {
            eprintln!("[linux] Git clone error: {}", e);
            None
        }
    }
}

/// Runs `make defconfig` for x86_64 in the Linux source tree to generate
/// configuration headers needed for compilation.
fn run_defconfig(source_dir: &Path) -> bool {
    // Check if generated headers already exist.
    if source_dir.join("include/generated/autoconf.h").exists() {
        eprintln!("[linux] Generated headers already exist, skipping defconfig");
        return true;
    }

    eprintln!("[linux] Running make ARCH=x86_64 defconfig...");
    let output = Command::new("make")
        .current_dir(source_dir)
        .args(["ARCH=x86_64", "defconfig"])
        .env("KCONFIG_ALLCONFIG", "1")
        .output();

    match output {
        Ok(o) if o.status.success() => {
            eprintln!("[linux] defconfig complete");
            true
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("[linux] defconfig failed: {}", stderr);
            // Try to continue anyway — some headers may have been generated.
            // Check for the critical generated header.
            source_dir.join("include/generated/autoconf.h").exists()
        }
        Err(e) => {
            eprintln!("[linux] make error: {}", e);
            false
        }
    }
}

// =========================================================================
// Kernel Include Flags
// =========================================================================

/// Returns the standard kernel include flags for x86_64 compilation.
fn kernel_include_flags(source_dir: &Path) -> Vec<String> {
    let src = source_dir.to_string_lossy();
    vec![
        format!("-I{}/include", src),
        format!("-I{}/arch/x86/include", src),
        format!("-I{}/include/uapi", src),
        format!("-I{}/arch/x86/include/uapi", src),
        format!("-I{}/arch/x86/include/generated", src),
        format!("-I{}/arch/x86/include/generated/uapi", src),
        format!("-I{}/include/generated/uapi", src),
        format!("-I{}/include/generated", src),
        // Additional include directories commonly needed.
        format!("-I{}/tools/include", src),
        "-D__KERNEL__".to_string(),
        "-D__x86_64__".to_string(),
        "-DCONFIG_X86_64".to_string(),
        "-DCONFIG_64BIT".to_string(),
        "-DKBUILD_MODNAME=\"bcc_test\"".to_string(),
        "-DKBUILD_BASENAME=\"bcc_test\"".to_string(),
        "-DMODULE".to_string(),
        "-ffreestanding".to_string(),
        "-mno-red-zone".to_string(),
        "-fno-strict-aliasing".to_string(),
        "-fno-common".to_string(),
        "-fno-delete-null-pointer-checks".to_string(),
        "-ffunction-sections".to_string(),
        "-fdata-sections".to_string(),
        "--target".to_string(),
        "x86_64-linux-gnu".to_string(),
    ]
}

// =========================================================================
// File Discovery
// =========================================================================

/// Recursively discovers all `.c` files in a given directory.
fn find_c_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() || !dir.is_dir() {
        return files;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip some subdirectories that require special handling or
                // are not meaningful for per-file compilation tests.
                let dirname = path.file_name().unwrap_or_default().to_string_lossy();
                if dirname == ".git" || dirname == "test" || dirname == "tests" {
                    continue;
                }
                files.extend(find_c_files(&path));
            } else if path.extension().map_or(false, |e| e == "c") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

// =========================================================================
// Compilation
// =========================================================================

/// Compiles a single kernel C file and returns the result.
fn compile_kernel_file(
    bcc: &Path,
    source_file: &Path,
    source_root: &Path,
    include_flags: &[String],
    output_dir: &Path,
) -> FileCompileResult {
    let relative = source_file
        .strip_prefix(source_root)
        .unwrap_or(source_file)
        .to_string_lossy()
        .to_string();

    // Create output path mirroring the source tree structure.
    let obj_name = source_file
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let obj_path = output_dir.join(format!("{}.o", obj_name));

    let start = Instant::now();

    let mut args: Vec<String> = vec!["-c".to_string()];
    args.extend(include_flags.iter().cloned());
    args.push("-o".to_string());
    args.push(obj_path.to_string_lossy().to_string());
    args.push(source_file.to_string_lossy().to_string());

    let output = Command::new(bcc)
        .args(&args)
        .output();

    let duration = start.elapsed();

    match output {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let passed = out.status.success();
            let is_ice = stderr.contains("internal compiler error")
                || stderr.contains("INTERNAL COMPILER ERROR")
                || stderr.contains("panicked at")
                || stderr.contains("thread '") && stderr.contains("panicked");

            let error_category = if passed {
                None
            } else {
                Some(categorize_error(&stderr))
            };

            FileCompileResult {
                relative_path: relative,
                passed,
                is_ice,
                stderr,
                duration,
                error_category,
            }
        }
        Err(e) => FileCompileResult {
            relative_path: relative,
            passed: false,
            is_ice: false,
            stderr: format!("Failed to execute bcc: {}", e),
            duration,
            error_category: Some("execution_error".to_string()),
        },
    }
}

// =========================================================================
// Error Categorization
// =========================================================================

/// Categorizes a compilation error based on the stderr output.
fn categorize_error(stderr: &str) -> String {
    if stderr.contains("internal compiler error") || stderr.contains("panicked at") {
        return "ice".to_string();
    }
    if stderr.contains("unexpected token in preprocessor expression") {
        return "preprocessor_expression".to_string();
    }
    if stderr.contains("unrecognized option") {
        return "unrecognized_flag".to_string();
    }
    if stderr.contains("undeclared identifier") || stderr.contains("use of undeclared") {
        return "undeclared_identifier".to_string();
    }
    if stderr.contains("implicit declaration of function") {
        return "implicit_declaration".to_string();
    }
    if stderr.contains("unknown type name") || stderr.contains("expected type") {
        return "unknown_type".to_string();
    }
    if stderr.contains("#include") && (stderr.contains("not found") || stderr.contains("No such file")) {
        return "missing_include".to_string();
    }
    if stderr.contains("expected") && stderr.contains("found") {
        return "parse_error".to_string();
    }
    if stderr.contains("incompatible type") || stderr.contains("type mismatch") {
        return "type_error".to_string();
    }
    if stderr.contains("#error") {
        return "directive_error".to_string();
    }
    if stderr.contains("redefinition of") || stderr.contains("conflicting types") {
        return "redefinition".to_string();
    }
    if stderr.contains("too many errors") {
        return "too_many_errors".to_string();
    }
    // Generic fallback — take first error line keyword.
    "other".to_string()
}

/// Compiles all C files in a subsystem directory and returns aggregated results.
fn compile_subsystem(
    bcc: &Path,
    source_root: &Path,
    subsystem: &str,
    include_flags: &[String],
    output_dir: &Path,
) -> SubsystemResult {
    let subsystem_dir = source_root.join(subsystem);
    let c_files = find_c_files(&subsystem_dir);

    eprintln!(
        "[linux] Compiling {} subsystem: {} .c files found",
        subsystem,
        c_files.len()
    );

    let sub_output_dir = output_dir.join(subsystem);
    let _ = fs::create_dir_all(&sub_output_dir);

    let mut files = Vec::new();
    let mut error_categories: HashMap<String, usize> = HashMap::new();

    for (idx, c_file) in c_files.iter().enumerate() {
        let result = compile_kernel_file(bcc, c_file, source_root, include_flags, &sub_output_dir);

        if let Some(ref cat) = result.error_category {
            *error_categories.entry(cat.clone()).or_insert(0) += 1;
        }

        // Print progress every 50 files.
        if (idx + 1) % 50 == 0 || idx + 1 == c_files.len() {
            let passed_so_far = files.iter().filter(|f: &&FileCompileResult| f.passed).count()
                + if result.passed { 1 } else { 0 };
            eprintln!(
                "[linux] {}: {}/{} compiled ({} passed so far)",
                subsystem,
                idx + 1,
                c_files.len(),
                passed_so_far
            );
        }

        files.push(result);
    }

    let total = files.len();
    let passed = files.iter().filter(|f| f.passed).count();
    let ice_count = files.iter().filter(|f| f.is_ice).count();
    let pass_rate = if total > 0 {
        (passed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    eprintln!(
        "[linux] {} result: {}/{} passed ({:.1}%), {} ICEs",
        subsystem, passed, total, pass_rate, ice_count
    );

    SubsystemResult {
        name: subsystem.to_string(),
        total,
        passed,
        ice_count,
        pass_rate,
        files,
        error_categories,
    }
}

// =========================================================================
// Report Generation
// =========================================================================

/// Generates a detailed report of compilation results.
fn generate_report(
    lib_result: &SubsystemResult,
    kernel_result: &SubsystemResult,
) -> String {
    let mut report = String::new();

    report.push_str("# Linux Kernel v6.12 Compilation Baseline Report\n\n");

    // Summary table.
    report.push_str("## Summary\n\n");
    report.push_str("| Subsystem | Total | Passed | Failed | ICEs | Pass Rate |\n");
    report.push_str("|-----------|-------|--------|--------|------|----------|\n");
    report.push_str(&format!(
        "| lib/      | {}    | {}     | {}     | {}   | {:.1}%    |\n",
        lib_result.total,
        lib_result.passed,
        lib_result.total - lib_result.passed,
        lib_result.ice_count,
        lib_result.pass_rate
    ));
    report.push_str(&format!(
        "| kernel/   | {}    | {}     | {}     | {}   | {:.1}%    |\n",
        kernel_result.total,
        kernel_result.passed,
        kernel_result.total - kernel_result.passed,
        kernel_result.ice_count,
        kernel_result.pass_rate
    ));

    // Error category breakdown.
    report.push_str("\n## Error Categories\n\n");
    for (name, result) in [("lib", lib_result), ("kernel", kernel_result)] {
        if !result.error_categories.is_empty() {
            report.push_str(&format!("### {} subsystem\n\n", name));
            report.push_str("| Category | Count |\n");
            report.push_str("|----------|-------|\n");
            let mut cats: Vec<_> = result.error_categories.iter().collect();
            cats.sort_by(|a, b| b.1.cmp(a.1));
            for (cat, count) in cats {
                report.push_str(&format!("| {} | {} |\n", cat, count));
            }
            report.push('\n');
        }
    }

    // Per-file failure details.
    report.push_str("\n## Failed Files\n\n");
    for (name, result) in [("lib", lib_result), ("kernel", kernel_result)] {
        let failures: Vec<_> = result.files.iter().filter(|f| !f.passed).collect();
        if !failures.is_empty() {
            report.push_str(&format!("### {} ({} failures)\n\n", name, failures.len()));
            for f in &failures {
                let cat = f.error_category.as_deref().unwrap_or("unknown");
                // Extract the first error line for brevity.
                let first_error = f
                    .stderr
                    .lines()
                    .find(|l| l.contains("error:"))
                    .unwrap_or("(no error line)")
                    .trim();
                // Truncate long lines.
                let truncated = if first_error.len() > 120 {
                    format!("{}...", &first_error[..120])
                } else {
                    first_error.to_string()
                };
                report.push_str(&format!(
                    "- `{}` [{}]: {}\n",
                    f.relative_path, cat, truncated
                ));
            }
            report.push('\n');
        }
    }

    // Merge gate evaluation.
    let total_ice = lib_result.ice_count + kernel_result.ice_count;
    report.push_str("\n## Merge Gate Evaluation\n\n");
    report.push_str(&format!(
        "- lib/ pass rate: {:.1}% (gate: ≥80%) — {}\n",
        lib_result.pass_rate,
        if lib_result.pass_rate >= 80.0 { "✅ PASS" } else { "❌ FAIL" }
    ));
    report.push_str(&format!(
        "- kernel/ pass rate: {:.1}% (gate: ≥60%) — {}\n",
        kernel_result.pass_rate,
        if kernel_result.pass_rate >= 60.0 { "✅ PASS" } else { "❌ FAIL" }
    ));
    report.push_str(&format!(
        "- ICE count: {} (gate: 0) — {}\n",
        total_ice,
        if total_ice == 0 { "✅ PASS" } else { "❌ FAIL" }
    ));

    report
}

// =========================================================================
// Tests
// =========================================================================

/// Core test runner that compiles the Linux kernel lib/ and kernel/ subsystems.
/// Returns (lib_result, kernel_result) or None if the source tree is unavailable.
#[cfg(feature = "linux_validation")]
fn run_linux_validation() -> Option<(SubsystemResult, SubsystemResult)> {
    let source_dir = get_linux_source()?;

    // Run defconfig to generate headers.
    if !run_defconfig(&source_dir) {
        eprintln!("[linux] Warning: defconfig failed, attempting compilation anyway");
    }

    let bcc = get_bcc_binary();
    let include_flags = kernel_include_flags(&source_dir);

    let output_dir = create_validation_work_dir("linux-output");
    let _ = fs::create_dir_all(&output_dir);

    // Compile lib/ subsystem.
    let lib_result = compile_subsystem(&bcc, &source_dir, "lib", &include_flags, &output_dir);

    // Compile kernel/ subsystem.
    let kernel_result = compile_subsystem(&bcc, &source_dir, "kernel", &include_flags, &output_dir);

    // Generate and save the report.
    let report = generate_report(&lib_result, &kernel_result);
    let report_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("BENCHMARKING.md");
    if let Err(e) = fs::write(&report_path, &report) {
        eprintln!("[linux] Warning: Failed to write BENCHMARKING.md: {}", e);
    } else {
        eprintln!("[linux] Report written to {}", report_path.display());
    }

    Some((lib_result, kernel_result))
}

#[cfg(feature = "linux_validation")]
#[test]
fn linux_kernel_lib_compilation() {
    let (lib_result, _kernel_result) = match run_linux_validation() {
        Some(r) => r,
        None => {
            eprintln!("[linux] Skipping: Linux source unavailable");
            return;
        }
    };

    // Verify zero ICEs.
    assert_eq!(
        lib_result.ice_count, 0,
        "Internal compiler errors detected in lib/: {} ICEs",
        lib_result.ice_count
    );

    // Verify ≥80% pass rate for lib/.
    assert!(
        lib_result.pass_rate >= 80.0,
        "lib/ pass rate {:.1}% is below the 80% gate ({}/{} passed)",
        lib_result.pass_rate,
        lib_result.passed,
        lib_result.total
    );
}

#[cfg(feature = "linux_validation")]
#[test]
fn linux_kernel_subsystem_compilation() {
    let (_lib_result, kernel_result) = match run_linux_validation() {
        Some(r) => r,
        None => {
            eprintln!("[linux] Skipping: Linux source unavailable");
            return;
        }
    };

    // Verify zero ICEs.
    assert_eq!(
        kernel_result.ice_count, 0,
        "Internal compiler errors detected in kernel/: {} ICEs",
        kernel_result.ice_count
    );

    // Verify ≥60% pass rate for kernel/.
    assert!(
        kernel_result.pass_rate >= 60.0,
        "kernel/ pass rate {:.1}% is below the 60% gate ({}/{} passed)",
        kernel_result.pass_rate,
        kernel_result.passed,
        kernel_result.total
    );
}

#[cfg(feature = "linux_validation")]
#[test]
fn linux_kernel_zero_ices() {
    let (lib_result, kernel_result) = match run_linux_validation() {
        Some(r) => r,
        None => {
            eprintln!("[linux] Skipping: Linux source unavailable");
            return;
        }
    };

    let total_ice = lib_result.ice_count + kernel_result.ice_count;

    if total_ice > 0 {
        // Report all ICE files.
        let mut ice_files = Vec::new();
        for f in lib_result.files.iter().chain(kernel_result.files.iter()) {
            if f.is_ice {
                ice_files.push(&f.relative_path);
            }
        }
        panic!(
            "Found {} ICEs in kernel compilation:\n{}",
            total_ice,
            ice_files
                .iter()
                .map(|p| format!("  - {}", p))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
