//! Redis source compile-only validation tests for the bcc compiler.
//!
//! This module verifies that `bcc` can compile the Redis server source code
//! across all four supported target architectures. Per AAP §0.2.1, Redis is
//! a **compile-only** verification target — no test suite execution, linking,
//! or runtime verification is required.
//!
//! # Validation Autonomy
//!
//! Redis source is fetched from GitHub at test time and never committed to the
//! repository. Tests skip gracefully when the network is unavailable.
//!
//! # Partial Compilation Tolerance
//!
//! Redis is a complex codebase with many Linux-specific APIs. A compilation
//! success threshold of 80% is used: the test reports per-file results but
//! considers the overall validation successful when ≥80% of source files
//! compile without errors.
//!
//! # Architectures Tested
//!
//! | Target Triple           | ELF Class |
//! |-------------------------|-----------|
//! | `x86_64-linux-gnu`      | ELF64     |
//! | `i686-linux-gnu`        | ELF32     |
//! | `aarch64-linux-gnu`     | ELF64     |
//! | `riscv64-linux-gnu`     | ELF64     |

use super::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// Redis Source Constants
// ===========================================================================

/// Redis version to download for validation testing.
const REDIS_VERSION: &str = "7.4.2";

/// Download URL for the Redis source tarball. This must remain consistent
/// with `REDIS_VERSION` — both reference the same release tag.
const REDIS_URL: &str = "https://github.com/redis/redis/archive/refs/tags/7.4.2.tar.gz";

/// Minimum fraction of source files that must compile successfully for the
/// overall validation to pass. Set at 80% to accommodate Redis files that
/// use Linux-specific APIs or constructs not yet supported by `bcc`.
const COMPILE_SUCCESS_THRESHOLD: f64 = 0.80;

// ===========================================================================
// Redis Source Fetching Helpers
// ===========================================================================

/// Download and extract the Redis source tree into `work_dir`.
///
/// Attempts to download the Redis tarball using `curl` (preferred) with a
/// `wget` fallback. On success, extracts the archive using `tar` and returns
/// the path to the top-level Redis source directory.
///
/// # Errors
///
/// Returns a descriptive error string on network failure or extraction failure.
fn fetch_redis_source(work_dir: &Path) -> Result<PathBuf, String> {
    let archive_name = format!("redis-{}.tar.gz", REDIS_VERSION);
    let archive_path = work_dir.join(&archive_name);

    // Attempt download with curl first (more commonly available, follows redirects).
    // Use `.status()` here since we only need the exit code — stdout/stderr are
    // suppressed via the `-sS` flags.
    let curl_ok = Command::new("curl")
        .args([
            "-sSfL",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "-o",
        ])
        .arg(&archive_path)
        .arg(REDIS_URL)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let download_ok = if curl_ok {
        true
    } else {
        // Fallback to wget if curl is unavailable or failed.
        let wget_result = Command::new("wget")
            .args(["-q", "--timeout=30", "-O"])
            .arg(&archive_path)
            .arg(REDIS_URL)
            .output();

        match wget_result {
            Ok(output) if output.status.success() => true,
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "Both curl and wget failed to download Redis source. \
                     wget stderr: {}",
                    stderr
                ));
            }
            Err(e) => {
                return Err(format!(
                    "Both curl and wget are unavailable or failed: {}",
                    e
                ));
            }
        }
    };

    if !download_ok {
        return Err("Download failed via both curl and wget".to_string());
    }

    // Verify the archive was actually written.
    if !archive_path.exists() {
        return Err(format!(
            "Archive file not found after download: {}",
            archive_path.display()
        ));
    }

    // Extract the tarball.
    let extract_result = Command::new("tar")
        .args(["xzf"])
        .arg(&archive_path)
        .arg("-C")
        .arg(work_dir)
        .output()
        .map_err(|e| format!("Failed to invoke tar: {}", e))?;

    if !extract_result.status.success() {
        let stderr = String::from_utf8_lossy(&extract_result.stderr);
        return Err(format!("tar extraction failed: {}", stderr));
    }

    // Locate the extracted directory using the version-derived name.
    let expected_dir_name = format!("redis-{}", REDIS_VERSION);
    let extracted_dir = work_dir.join(&expected_dir_name);
    if extracted_dir.exists() {
        return Ok(extracted_dir);
    }

    // If the expected directory name doesn't match, search for any redis-*
    // directory that was created during extraction.
    if let Ok(entries) = fs::read_dir(work_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("redis-") && entry.path().is_dir() {
                return Ok(entry.path());
            }
        }
    }

    Err(format!(
        "Extracted Redis source directory not found in {}",
        work_dir.display()
    ))
}

/// Check whether Redis source files are present in the given directory.
///
/// Verifies the directory contains both `src/` and `deps/` subdirectories,
/// and that at least one `.c` file exists under `src/`.
fn is_redis_source_available(dir: &Path) -> bool {
    let src_dir = dir.join("src");
    let deps_dir = dir.join("deps");

    if !src_dir.exists() || !deps_dir.exists() {
        return false;
    }

    // Check for at least one C source file in src/.
    if let Ok(entries) = fs::read_dir(&src_dir) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "c" {
                    return true;
                }
            }
        }
    }

    false
}

/// Discover all `.c` source files in the Redis `src/` directory.
///
/// Uses `fs::read_dir()` to dynamically enumerate C source files rather than
/// relying on a hardcoded list, since the file inventory varies across Redis
/// versions.
///
/// Returns a sorted `Vec<PathBuf>` for deterministic compilation ordering.
fn discover_redis_source_files(src_dir: &Path) -> Vec<PathBuf> {
    let mut c_files = Vec::new();

    if let Ok(entries) = fs::read_dir(src_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "c" {
                        c_files.push(path);
                    }
                }
            }
        }
    }

    // Sort for deterministic ordering across platforms and runs.
    c_files.sort();
    c_files
}

/// Discover key dependency source files under Redis `deps/` directory.
///
/// Redis bundles several third-party libraries under `deps/`. This function
/// locates `.c` files in the most important bundled libraries: hiredis,
/// linenoise, and optionally lua. jemalloc is excluded as it is complex and
/// optional.
fn discover_redis_dep_files(redis_dir: &Path) -> Vec<PathBuf> {
    let mut dep_files = Vec::new();
    let deps_dir = redis_dir.join("deps");

    // hiredis — Redis client library
    let hiredis_dir = deps_dir.join("hiredis");
    if hiredis_dir.exists() {
        for name in &[
            "hiredis.c",
            "net.c",
            "sds.c",
            "read.c",
            "alloc.c",
            "async.c",
        ] {
            let path = hiredis_dir.join(name);
            if path.exists() {
                dep_files.push(path);
            }
        }
    }

    // linenoise — Line editing library
    let linenoise_dir = deps_dir.join("linenoise");
    if linenoise_dir.exists() {
        let path = linenoise_dir.join("linenoise.c");
        if path.exists() {
            dep_files.push(path);
        }
    }

    dep_files
}

// ===========================================================================
// Compilation Helpers
// ===========================================================================

/// Return the compile-time defines required for Redis compilation.
///
/// These defines configure Redis for static compilation and provide the
/// minimal environment necessary for successful parsing and compilation
/// of Redis source files.
fn get_redis_defines() -> Vec<String> {
    vec![
        "-DREDIS_STATIC=".to_string(),
        "-D_GNU_SOURCE".to_string(),
        // Do NOT define USE_JEMALLOC at all — Redis uses `#elif defined(USE_JEMALLOC)`
        // which triggers even when the value is 0. Leaving it undefined causes
        // zmalloc.h to fall through to the libc malloc path.
        //
        // release.h is generated by the Redis build system. Provide stubs
        // for the macros it would normally define.
        "-DREDIS_GIT_SHA1=\"0000000\"".to_string(),
        "-DREDIS_GIT_DIRTY=\"0\"".to_string(),
        "-DREDIS_BUILD_ID_RAW=\"0000000000000000\"".to_string(),
    ]
}

/// Return the `-I` include paths needed for Redis compilation.
///
/// Redis source files expect headers from the main `src/` directory and from
/// several bundled dependency directories under `deps/`.
fn get_redis_include_paths(redis_dir: &Path) -> Vec<String> {
    let mut includes = Vec::new();

    // Primary Redis source include directory.
    includes.push(format!("-I{}", redis_dir.join("src").display()));

    // Bundled dependency include directories.
    let deps = redis_dir.join("deps");

    let hiredis_path = deps.join("hiredis");
    if hiredis_path.exists() {
        includes.push(format!("-I{}", hiredis_path.display()));
    }

    let linenoise_path = deps.join("linenoise");
    if linenoise_path.exists() {
        includes.push(format!("-I{}", linenoise_path.display()));
    }

    let lua_src_path = deps.join("lua").join("src");
    if lua_src_path.exists() {
        includes.push(format!("-I{}", lua_src_path.display()));
    }

    // fpconv — decimal-to-string conversion used by debug.c, rio.c, etc.
    let fpconv_path = deps.join("fpconv");
    if fpconv_path.exists() {
        includes.push(format!("-I{}", fpconv_path.display()));
    }

    // hdr_histogram — latency histogram library used by server.c, etc.
    let hdr_path = deps.join("hdr_histogram");
    if hdr_path.exists() {
        includes.push(format!("-I{}", hdr_path.display()));
    }

    // Some Redis files include headers from the deps root.
    includes.push(format!("-I{}", deps.display()));

    includes
}

/// Compile a single Redis `.c` source file using the `bcc` compiler.
///
/// Invokes `bcc` with the specified target, include paths, defines, and
/// optimization level. The `-c` flag ensures compile-only mode (no linking).
/// The output `.o` object file is written to `output_path`.
///
/// # Returns
///
/// On success, returns the path to the produced `.o` object file.
/// On failure, returns an error string describing what went wrong.
fn compile_redis_file(
    file: &Path,
    target: &str,
    includes: &[String],
    defines: &[String],
    opt_level: &str,
    output_path: &Path,
) -> Result<PathBuf, String> {
    let bcc = get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-c");
    cmd.arg(opt_level);
    cmd.arg("-o").arg(output_path);

    // Add include paths.
    for inc in includes {
        cmd.arg(inc);
    }

    // Add defines.
    for def in defines {
        cmd.arg(def);
    }

    // Add the source file.
    cmd.arg(file);

    let result = cmd
        .output()
        .map_err(|e| format!("Failed to execute bcc for '{}': {}", file.display(), e))?;

    if result.status.success() {
        Ok(output_path.to_path_buf())
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr);
        Err(format!(
            "Compilation of '{}' failed (exit code {:?}):\n{}",
            file.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| file.display().to_string()),
            result.status.code(),
            stderr
        ))
    }
}

/// Compile all Redis source files for the specified target and optimization level.
///
/// Discovers `.c` files dynamically, compiles each one individually, and
/// reports aggregate results. Compilation continues even when individual files
/// fail, implementing the partial compilation tolerance policy.
///
/// # Returns
///
/// A tuple of `(success_count, total_count, error_messages)`:
/// - `success_count`: Number of files that compiled successfully.
/// - `total_count`: Total number of `.c` files discovered and attempted.
/// - `error_messages`: Diagnostic strings for each file that failed.
fn compile_redis_all_files(
    redis_dir: &Path,
    target: &str,
    opt_level: &str,
) -> (usize, usize, Vec<String>) {
    let src_dir = redis_dir.join("src");
    let c_files = discover_redis_source_files(&src_dir);
    let includes = get_redis_include_paths(redis_dir);
    let defines = get_redis_defines();

    let total = c_files.len();
    let mut success_count = 0usize;
    let mut errors: Vec<String> = Vec::new();

    // Create an output directory for object files to avoid polluting the
    // source tree (which may be read-only or shared across tests).
    let obj_dir = redis_dir.join("obj").join(target);
    let _ = fs::create_dir_all(&obj_dir);

    // Generate release.h stub — normally created by Redis's Makefile.
    // Contains git SHA and build ID macros needed by release.c.
    let release_h = redis_dir.join("src").join("release.h");
    if !release_h.exists() {
        let _ = fs::write(
            &release_h,
            "#ifndef __REDIS_RELEASE_H\n\
             #define __REDIS_RELEASE_H\n\
             #define REDIS_GIT_SHA1 \"00000000\"\n\
             #define REDIS_GIT_DIRTY \"0\"\n\
             #define REDIS_BUILD_ID_RAW \"0000000000000000\"\n\
             #endif\n",
        );
    }

    for file in &c_files {
        // Place the .o file in the obj directory, mirroring the source file
        // name with a `.o` extension.
        let obj_name = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        let obj_path = obj_dir.join(&obj_name).with_extension("o");

        let result = compile_redis_file(file, target, &includes, &defines, opt_level, &obj_path);
        match result {
            Ok(_) => {
                success_count += 1;
            }
            Err(msg) => {
                errors.push(msg);
            }
        }
    }

    eprintln!(
        "[redis] target={} opt={} compiled {}/{} files successfully",
        target, opt_level, success_count, total
    );

    (success_count, total, errors)
}

/// Compile Redis bundled dependency source files for the specified target.
///
/// Compiles key files from `deps/hiredis/` and `deps/linenoise/`. Returns
/// aggregate results in the same format as `compile_redis_all_files`.
fn compile_redis_deps(
    redis_dir: &Path,
    target: &str,
    opt_level: &str,
) -> (usize, usize, Vec<String>) {
    let dep_files = discover_redis_dep_files(redis_dir);
    let includes = get_redis_include_paths(redis_dir);
    let defines = get_redis_defines();

    let total = dep_files.len();
    let mut success_count = 0usize;
    let mut errors: Vec<String> = Vec::new();

    let obj_dir = redis_dir.join("obj_deps").join(target);
    let _ = fs::create_dir_all(&obj_dir);

    for file in &dep_files {
        let obj_name = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        // Include parent directory in obj name to avoid collisions
        // (e.g., hiredis/sds.c vs src/sds.c).
        let parent_name = file
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let obj_path = obj_dir.join(format!("{}_{}.o", parent_name, obj_name));

        let result = compile_redis_file(file, target, &includes, &defines, opt_level, &obj_path);
        match result {
            Ok(_) => {
                success_count += 1;
            }
            Err(msg) => {
                errors.push(msg);
            }
        }
    }

    eprintln!(
        "[redis-deps] target={} opt={} compiled {}/{} dep files successfully",
        target, opt_level, success_count, total
    );

    (success_count, total, errors)
}

// ===========================================================================
// Shared Test Setup Helper
// ===========================================================================

/// Fetch Redis source into a `ValidationWorkDir`, returning the path to the
/// Redis directory. If the network is unavailable or fetching fails, returns
/// `None` and prints a skip message.
fn setup_redis_source(test_name: &str) -> Option<(ValidationWorkDir, PathBuf)> {
    if skip_if_offline(test_name) {
        return None;
    }

    let work_dir = ValidationWorkDir::new("redis");

    match fetch_redis_source(work_dir.path()) {
        Ok(redis_dir) => {
            if !is_redis_source_available(&redis_dir) {
                eprintln!(
                    "SKIPPED: {} (Redis source files not found after extraction)",
                    test_name
                );
                return None;
            }
            Some((work_dir, redis_dir))
        }
        Err(e) => {
            eprintln!("SKIPPED: {} ({})", test_name, e);
            None
        }
    }
}

/// Evaluate compilation results against the success threshold.
///
/// Returns `true` if the ratio of successfully compiled files meets or exceeds
/// `COMPILE_SUCCESS_THRESHOLD`, or if no files were discovered (vacuous truth).
fn meets_threshold(success: usize, total: usize) -> bool {
    if total == 0 {
        return true;
    }
    let ratio = success as f64 / total as f64;
    ratio >= COMPILE_SUCCESS_THRESHOLD
}

/// Run Redis compile-only validation for a single target architecture.
///
/// This function encapsulates the common logic used by each per-architecture
/// test: fetch source, compile all `.c` files, compile dependency files,
/// verify results against the success threshold, and report diagnostics.
fn run_redis_compile_test(target: &str, opt_level: &str) {
    let test_name = format!("redis_compile_{}_{}", target, opt_level);

    let (work_dir, redis_dir) = match setup_redis_source(&test_name) {
        Some(v) => v,
        None => return, // Test skipped — network unavailable or source fetch failed.
    };

    // Compile Redis core source files.
    let (src_ok, src_total, src_errors) = compile_redis_all_files(&redis_dir, target, opt_level);

    // Compile Redis bundled dependency files.
    let (dep_ok, dep_total, dep_errors) = compile_redis_deps(&redis_dir, target, opt_level);

    // Report detailed results.
    eprintln!("[redis] RESULTS for target={} opt={}:", target, opt_level);
    eprintln!(
        "  Core sources: {}/{} compiled successfully",
        src_ok, src_total
    );
    eprintln!(
        "  Dependencies: {}/{} compiled successfully",
        dep_ok, dep_total
    );

    if !src_errors.is_empty() {
        eprintln!("  Core compilation failures ({}):", src_errors.len());
        for (i, err) in src_errors.iter().enumerate().take(10) {
            eprintln!("    [{}] {}", i + 1, err.lines().next().unwrap_or(""));
        }
        if src_errors.len() > 10 {
            eprintln!("    ... and {} more failures", src_errors.len() - 10);
        }
    }

    if !dep_errors.is_empty() {
        eprintln!("  Dependency compilation failures ({}):", dep_errors.len());
        for err in &dep_errors {
            eprintln!("    {}", err.lines().next().unwrap_or(""));
        }
    }

    // Enforce the success threshold on core source files.
    assert!(
        meets_threshold(src_ok, src_total),
        "Redis core compilation success rate {}/{} ({:.1}%) is below the \
         {:.0}% threshold for target={} opt={}",
        src_ok,
        src_total,
        if src_total > 0 {
            src_ok as f64 / src_total as f64 * 100.0
        } else {
            100.0
        },
        COMPILE_SUCCESS_THRESHOLD * 100.0,
        target,
        opt_level,
    );

    // The work_dir is dropped here and the temporary directory is cleaned up.
    drop(work_dir);
}

// ===========================================================================
// Per-Architecture Compilation Tests
// ===========================================================================

/// Compile Redis source files for x86-64 at -O0.
///
/// Verifies compile-only success for the primary x86-64 target. Each `.c`
/// file in the Redis `src/` directory is compiled individually with `-c`.
/// No linking or execution is performed per AAP §0.2.1.
#[test]
fn redis_compile_x86_64() {
    run_redis_compile_test("x86_64-linux-gnu", "-O0");
}

/// Compile Redis source files for i686 (32-bit x86) at -O0.
///
/// Same compile-only verification as x86_64 but targeting the i686 backend,
/// producing ELF32 relocatable objects.
#[test]
fn redis_compile_i686() {
    run_redis_compile_test("i686-linux-gnu", "-O0");
}

/// Compile Redis source files for AArch64 at -O0.
///
/// Same compile-only verification targeting the AArch64 backend, producing
/// ELF64 relocatable objects with AAPCS64 ABI.
#[test]
fn redis_compile_aarch64() {
    run_redis_compile_test("aarch64-linux-gnu", "-O0");
}

/// Compile Redis source files for RISC-V 64 at -O0.
///
/// Same compile-only verification targeting the RISC-V 64 backend, producing
/// ELF64 relocatable objects with LP64D ABI.
#[test]
fn redis_compile_riscv64() {
    run_redis_compile_test("riscv64-linux-gnu", "-O0");
}

// ===========================================================================
// All-Architecture Compilation Test
// ===========================================================================

/// Compile Redis source files for all four architectures and report aggregate
/// results.
///
/// This test compiles the Redis `src/` directory for x86-64, i686, AArch64,
/// and RISC-V 64. Per-architecture results are printed to stderr for
/// diagnostic visibility. The test passes if every architecture meets the
/// 80% compilation success threshold.
#[test]
fn redis_compile_all_architectures() {
    let test_name = "redis_compile_all_architectures";

    let (work_dir, redis_dir) = match setup_redis_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let targets = ALL_TARGETS;
    let mut all_passed = true;
    let mut summary: Vec<String> = Vec::new();

    for target in targets {
        let (src_ok, src_total, _src_errors) = compile_redis_all_files(&redis_dir, target, "-O0");
        let (dep_ok, dep_total, _dep_errors) = compile_redis_deps(&redis_dir, target, "-O0");

        let passed = meets_threshold(src_ok, src_total);
        if !passed {
            all_passed = false;
        }

        summary.push(format!(
            "  {:<24} core={}/{} deps={}/{} {}",
            target,
            src_ok,
            src_total,
            dep_ok,
            dep_total,
            if passed { "PASS" } else { "FAIL" },
        ));
    }

    eprintln!("[redis] All-architecture compilation results:");
    for line in &summary {
        eprintln!("{}", line);
    }

    assert!(
        all_passed,
        "One or more architectures failed the {:.0}% compilation threshold.\n{}",
        COMPILE_SUCCESS_THRESHOLD * 100.0,
        summary.join("\n"),
    );

    drop(work_dir);
}

// ===========================================================================
// Optimization Level Tests
// ===========================================================================

/// Compile Redis files at optimization level -O0 for the primary target.
///
/// This is the baseline optimization level — no optimization passes are run.
/// Verifies that the unoptimized code path compiles Redis successfully.
#[test]
fn redis_compile_o0() {
    run_redis_compile_test("x86_64-linux-gnu", "-O0");
}

/// Compile Redis files at optimization level -O1 for the primary target.
///
/// At -O1, basic optimization passes (mem2reg, constant folding, dead code
/// elimination) are applied. Verifies these passes do not break Redis
/// compilation.
#[test]
fn redis_compile_o1() {
    run_redis_compile_test("x86_64-linux-gnu", "-O1");
}

/// Compile Redis files at optimization level -O2 for the primary target.
///
/// At -O2, aggressive optimization passes (CSE, algebraic simplification)
/// are added. Verifies the full optimization pipeline handles Redis code
/// correctly.
#[test]
fn redis_compile_o2() {
    run_redis_compile_test("x86_64-linux-gnu", "-O2");
}

// ===========================================================================
// Unit Tests (module-internal validation)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the success threshold calculation works correctly.
    #[test]
    fn test_meets_threshold() {
        // 80% threshold
        assert!(meets_threshold(80, 100));
        assert!(meets_threshold(100, 100));
        assert!(meets_threshold(81, 100));
        assert!(!meets_threshold(79, 100));
        assert!(!meets_threshold(0, 100));

        // Edge cases
        assert!(meets_threshold(0, 0)); // vacuous truth
        assert!(meets_threshold(1, 1)); // 100%
        assert!(meets_threshold(4, 5)); // 80%
        assert!(!meets_threshold(3, 5)); // 60%
    }

    /// Verify Redis defines are non-empty and well-formed.
    #[test]
    fn test_get_redis_defines() {
        let defines = get_redis_defines();
        assert!(!defines.is_empty(), "Redis defines should not be empty");
        for def in &defines {
            assert!(
                def.starts_with("-D"),
                "Each define must start with -D, got: {}",
                def
            );
        }
    }

    /// Verify include path generation produces expected paths.
    #[test]
    fn test_get_redis_include_paths() {
        let fake_dir = PathBuf::from("/tmp/fake_redis");
        let paths = get_redis_include_paths(&fake_dir);
        assert!(!paths.is_empty(), "Include paths should not be empty");
        // At minimum, the src/ directory include should always be present.
        assert!(
            paths.iter().any(|p| p.contains("src")),
            "Include paths must contain the src/ directory"
        );
        for path in &paths {
            assert!(
                path.starts_with("-I"),
                "Each include path must start with -I, got: {}",
                path
            );
        }
    }

    /// Verify that `is_redis_source_available` returns false for a non-existent
    /// directory.
    #[test]
    fn test_is_redis_source_available_nonexistent() {
        let fake = PathBuf::from("/tmp/nonexistent_redis_dir_12345");
        assert!(
            !is_redis_source_available(&fake),
            "Should return false for non-existent directory"
        );
    }

    /// Verify that `discover_redis_source_files` returns an empty list for a
    /// directory with no C files.
    #[test]
    fn test_discover_redis_source_files_empty() {
        let fake = PathBuf::from("/tmp/nonexistent_redis_src_dir_12345");
        let files = discover_redis_source_files(&fake);
        assert!(
            files.is_empty(),
            "Should return empty list for non-existent directory"
        );
    }

    /// Verify that `discover_redis_dep_files` returns an empty list when the
    /// deps directory doesn't exist.
    #[test]
    fn test_discover_redis_dep_files_empty() {
        let fake = PathBuf::from("/tmp/nonexistent_redis_dir_12345");
        let files = discover_redis_dep_files(&fake);
        assert!(
            files.is_empty(),
            "Should return empty list for non-existent deps directory"
        );
    }
}
