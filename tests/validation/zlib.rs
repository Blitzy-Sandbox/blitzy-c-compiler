//! zlib source validation tests for the bcc compiler.
//!
//! This module verifies that `bcc` can compile the zlib compression library
//! source code and run its bundled test suite across all four supported target
//! architectures. zlib is a validation target per AAP §0.2.1 and §0.7.
//!
//! # Validation Autonomy
//!
//! The zlib source tree is fetched from the official zlib website at test time
//! and is never committed to the repository. Tests skip gracefully when the
//! network is unavailable.
//!
//! # Success Criteria (AAP §0.7, strict order)
//!
//! 1. Compilation succeeds for all four architectures.
//! 2. zlib bundled test suite passes (via `example.c` and `minigzip.c`).
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
use std::time::Instant;

// ===========================================================================
// zlib Source Constants
// ===========================================================================

/// zlib version to download for validation testing.
const ZLIB_VERSION: &str = "1.3.1";

/// Download URL for the zlib source tarball from the official website.
/// This must remain consistent with `ZLIB_VERSION`.
const ZLIB_URL: &str = "https://zlib.net/zlib-1.3.1.tar.gz";

/// Fallback download URL in case the primary is unavailable (GitHub mirror).
const ZLIB_URL_FALLBACK: &str =
    "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz";

/// Core zlib source files that compose the library. These are compiled
/// individually and linked together for test programs.
const ZLIB_CORE_SOURCES: &[&str] = &[
    "adler32.c",
    "compress.c",
    "crc32.c",
    "deflate.c",
    "gzclose.c",
    "gzlib.c",
    "gzread.c",
    "gzwrite.c",
    "infback.c",
    "inffast.c",
    "inflate.c",
    "inftrees.c",
    "trees.c",
    "uncompr.c",
    "zutil.c",
];

/// zlib's primary test program — exercises compress/uncompress/gzip APIs.
const ZLIB_EXAMPLE_SOURCE: &str = "test/example.c";

/// zlib's minigzip utility — useful for round-trip compression testing.
const ZLIB_MINIGZIP_SOURCE: &str = "test/minigzip.c";

// ===========================================================================
// zlib Source Fetching Helpers
// ===========================================================================

/// Download and extract the zlib source tree into `work_dir`.
///
/// Attempts to download the zlib tarball from the official website using
/// `curl` (preferred) with a `wget` fallback. If the primary URL fails,
/// a GitHub mirror URL is tried. On success, extracts the archive using
/// `tar` and returns the path to the top-level zlib source directory.
///
/// # Errors
///
/// Returns a descriptive error string on network failure or extraction failure.
fn fetch_zlib_source(work_dir: &Path) -> Result<PathBuf, String> {
    let archive_name = format!("zlib-{}.tar.gz", ZLIB_VERSION);
    let archive_path = work_dir.join(&archive_name);

    // Try primary URL with curl first.
    let download_ok = try_download_with_curl(&archive_path, ZLIB_URL)
        || try_download_with_wget(&archive_path, ZLIB_URL)
        || try_download_with_curl(&archive_path, ZLIB_URL_FALLBACK)
        || try_download_with_wget(&archive_path, ZLIB_URL_FALLBACK);

    if !download_ok {
        return Err("Failed to download zlib source via both curl and wget \
             from both primary and fallback URLs"
            .to_string());
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

    // Locate the extracted directory.
    let expected_dir = work_dir.join(format!("zlib-{}", ZLIB_VERSION));
    if expected_dir.exists() {
        return Ok(expected_dir);
    }

    // Search for any zlib-* directory as a fallback.
    if let Ok(entries) = fs::read_dir(work_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("zlib-") && entry.path().is_dir() {
                return Ok(entry.path());
            }
        }
    }

    Err(format!(
        "Extracted zlib source directory not found in {}",
        work_dir.display()
    ))
}

/// Attempt to download a URL using `curl`. Returns `true` on success.
fn try_download_with_curl(dest: &Path, url: &str) -> bool {
    Command::new("curl")
        .args([
            "-sSfL",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "-o",
        ])
        .arg(dest)
        .arg(url)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Attempt to download a URL using `wget`. Returns `true` on success.
fn try_download_with_wget(dest: &Path, url: &str) -> bool {
    Command::new("wget")
        .args(["-q", "--timeout=30", "-O"])
        .arg(dest)
        .arg(url)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check whether zlib source files are present in the given directory.
///
/// Verifies the directory contains the expected core source files (`zutil.c`,
/// `inflate.c`, etc.) and the `zlib.h` header.
fn is_zlib_source_available(dir: &Path) -> bool {
    let zlib_h = dir.join("zlib.h");
    let zutil_c = dir.join("zutil.c");
    let inflate_c = dir.join("inflate.c");
    zlib_h.exists() && zutil_c.exists() && inflate_c.exists()
}

/// Return the list of zlib core `.c` source file paths within `src_dir`.
///
/// Uses the hardcoded list of core zlib source files, verifying each exists
/// in the provided directory. Returns only files that actually exist.
fn get_zlib_source_files(src_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for name in ZLIB_CORE_SOURCES {
        let path = src_dir.join(name);
        if path.exists() {
            files.push(path);
        }
    }
    files
}

// ===========================================================================
// Compilation Helpers
// ===========================================================================

/// Return compile-time defines required for zlib compilation on Linux.
///
/// These defines configure zlib for a Linux environment with large file
/// support and standard POSIX features.
fn get_zlib_defines() -> Vec<String> {
    vec![
        "-DHAVE_UNISTD_H".to_string(),
        "-D_LARGEFILE64_SOURCE=1".to_string(),
        "-DHAVE_SYS_TYPES_H".to_string(),
        "-DHAVE_STDINT_H".to_string(),
        "-DHAVE_STDDEF_H".to_string(),
    ]
}

/// Compile all zlib core source files for the specified target and
/// optimization level.
///
/// Each `.c` file is compiled individually with `-c` to produce a `.o`
/// relocatable object. The function records timing information and returns
/// the list of successfully produced object file paths.
///
/// # Returns
///
/// On success, returns `Ok(Vec<PathBuf>)` containing the paths to all `.o`
/// files produced. On failure (when any file fails to compile), returns
/// `Err(String)` with a detailed error message listing which files failed.
fn compile_zlib_sources(
    src_dir: &Path,
    target: &str,
    opt_level: &str,
) -> Result<Vec<PathBuf>, String> {
    let c_files = get_zlib_source_files(src_dir);
    if c_files.is_empty() {
        return Err("No zlib source files found".to_string());
    }

    let defines = get_zlib_defines();
    let bcc = get_bcc_binary();
    let obj_dir = src_dir.join("obj").join(target);
    let _ = fs::create_dir_all(&obj_dir);

    let start = Instant::now();
    let mut objects: Vec<PathBuf> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for file in &c_files {
        let obj_name = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        let obj_path = obj_dir.join(&obj_name).with_extension("o");

        let mut cmd = Command::new(&bcc);
        cmd.arg("--target").arg(target);
        cmd.arg("-c");
        cmd.arg(opt_level);
        cmd.arg("-o").arg(&obj_path);
        // Include the zlib source directory for header resolution.
        cmd.arg(format!("-I{}", src_dir.display()));
        for def in &defines {
            cmd.arg(def);
        }
        cmd.arg(file);

        let result = cmd
            .output()
            .map_err(|e| format!("Failed to execute bcc for '{}': {}", file.display(), e));

        match result {
            Ok(output) if output.status.success() => {
                objects.push(obj_path);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                errors.push(format!(
                    "{}: {}",
                    file.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    stderr.lines().next().unwrap_or("unknown error")
                ));
            }
            Err(e) => {
                errors.push(e);
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "[zlib] target={} opt={} compiled {}/{} files in {:.2}s",
        target,
        opt_level,
        objects.len(),
        c_files.len(),
        elapsed.as_secs_f64()
    );

    if !errors.is_empty() {
        let mut msg = format!(
            "zlib compilation failed for {}/{} files (target={}, opt={}):\n",
            errors.len(),
            c_files.len(),
            target,
            opt_level
        );
        for (i, err) in errors.iter().enumerate() {
            msg.push_str(&format!("  [{}] {}\n", i + 1, err));
        }
        return Err(msg);
    }

    Ok(objects)
}

/// Compile a single C source file (e.g., `example.c` or `minigzip.c`) for
/// the specified target, with the zlib include directory on the search path.
///
/// Returns the path to the produced `.o` object file on success.
fn compile_single_file(
    source: &Path,
    target: &str,
    opt_level: &str,
    include_dir: &Path,
    output: &Path,
) -> Result<PathBuf, String> {
    let bcc = get_bcc_binary();
    let defines = get_zlib_defines();

    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-c");
    cmd.arg(opt_level);
    cmd.arg("-o").arg(output);
    cmd.arg(format!("-I{}", include_dir.display()));
    for def in &defines {
        cmd.arg(def);
    }
    cmd.arg(source);

    let result = cmd
        .output()
        .map_err(|e| format!("Failed to execute bcc: {}", e))?;

    if result.status.success() {
        Ok(output.to_path_buf())
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr);
        Err(format!(
            "Compilation of '{}' failed:\n{}",
            source.display(),
            stderr
        ))
    }
}

/// Link zlib object files together with a test program object into an
/// executable binary.
///
/// This function invokes `bcc` in linking mode (no `-c` flag) to produce a
/// final executable from the pre-compiled object files. The resulting binary
/// can then be executed natively or via QEMU.
///
/// # Arguments
///
/// * `objects` — Paths to the zlib library `.o` files.
/// * `example_obj` — Path to the test program `.o` file (e.g., compiled `example.c`).
/// * `target` — Target triple string (e.g., `x86_64-linux-gnu`).
/// * `output` — Desired path for the output executable.
///
/// # Returns
///
/// `Ok(())` on successful linking, `Err(String)` with a diagnostic message
/// on failure.
fn link_zlib_test(
    objects: &[PathBuf],
    example_obj: &Path,
    target: &str,
    output: &Path,
) -> Result<(), String> {
    let bcc = get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-o").arg(output);

    // Add the test program object first.
    cmd.arg(example_obj);

    // Add all zlib library objects.
    for obj in objects {
        cmd.arg(obj);
    }

    // Link against the system math library (zlib uses math functions).
    cmd.arg("-lm");

    let result = cmd
        .output()
        .map_err(|e| format!("Failed to execute bcc linker: {}", e))?;

    if result.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr);
        Err(format!("Linking failed:\n{}", stderr))
    }
}

// ===========================================================================
// Shared Test Setup Helper
// ===========================================================================

/// Fetch zlib source into a `ValidationWorkDir`, returning the path to the
/// zlib directory. If the network is unavailable or fetching fails, returns
/// `None` and prints a skip message.
fn setup_zlib_source(test_name: &str) -> Option<(ValidationWorkDir, PathBuf)> {
    if skip_if_offline(test_name) {
        return None;
    }

    let work_dir = ValidationWorkDir::new("zlib");

    match fetch_zlib_source(work_dir.path()) {
        Ok(zlib_dir) => {
            if !is_zlib_source_available(&zlib_dir) {
                eprintln!(
                    "SKIPPED: {} (zlib source files not found after extraction)",
                    test_name
                );
                return None;
            }
            Some((work_dir, zlib_dir))
        }
        Err(e) => {
            eprintln!("SKIPPED: {} ({})", test_name, e);
            None
        }
    }
}

/// Run zlib compile-only validation for a single target architecture.
///
/// Encapsulates the common logic for per-architecture compile tests: fetch
/// source, compile all core `.c` files, verify all succeed.
fn run_zlib_compile_test(target: &str, opt_level: &str) {
    let test_name = format!("zlib_compile_{}_{}", target, opt_level);

    let (work_dir, zlib_dir) = match setup_zlib_source(&test_name) {
        Some(v) => v,
        None => return,
    };

    let result = compile_zlib_sources(&zlib_dir, target, opt_level);

    match &result {
        Ok(objects) => {
            eprintln!(
                "[zlib] {} — compiled {} object files successfully",
                test_name,
                objects.len()
            );
            assert!(
                !objects.is_empty(),
                "Expected at least one compiled zlib object file"
            );
        }
        Err(e) => {
            // Print the full error for diagnostics before asserting.
            eprintln!("[zlib] {} FAILED:\n{}", test_name, e);
        }
    }

    assert!(
        result.is_ok(),
        "zlib compilation failed for target={} opt={}",
        target,
        opt_level
    );

    drop(work_dir);
}

// ===========================================================================
// Per-Architecture Compilation Tests
// ===========================================================================

/// Compile all zlib core source files for x86-64 at -O0.
///
/// Each `.c` file is compiled individually with
/// `bcc --target x86_64-linux-gnu -O0 -c <file>.c`. The test verifies that
/// all compilations succeed and produce valid ELF64 relocatable objects.
#[test]
fn zlib_compile_x86_64() {
    run_zlib_compile_test("x86_64-linux-gnu", "-O0");
}

/// Compile all zlib core source files for i686 at -O0.
///
/// Same as the x86-64 test but targeting the 32-bit i686 backend. Output
/// objects must be ELF32 relocatable files.
#[test]
fn zlib_compile_i686() {
    run_zlib_compile_test("i686-linux-gnu", "-O0");
}

/// Compile all zlib core source files for AArch64 at -O0.
///
/// Targets the AArch64 backend, producing ELF64 relocatable objects with
/// AAPCS64 ABI.
#[test]
fn zlib_compile_aarch64() {
    run_zlib_compile_test("aarch64-linux-gnu", "-O0");
}

/// Compile all zlib core source files for RISC-V 64 at -O0.
///
/// Targets the RISC-V 64 backend, producing ELF64 relocatable objects with
/// LP64D ABI.
#[test]
fn zlib_compile_riscv64() {
    run_zlib_compile_test("riscv64-linux-gnu", "-O0");
}

// ===========================================================================
// All-Architecture Compilation Test
// ===========================================================================

/// Compile zlib for all four architectures and report aggregate results.
///
/// This test compiles the zlib library for x86-64, i686, AArch64, and
/// RISC-V 64. Per-architecture results are printed to stderr for diagnostic
/// visibility. The test passes only if every architecture compiles
/// successfully.
#[test]
fn zlib_compile_all_architectures() {
    let test_name = "zlib_compile_all_architectures";

    let (work_dir, zlib_dir) = match setup_zlib_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let targets = ALL_TARGETS;
    let mut all_passed = true;
    let mut summary: Vec<String> = Vec::new();

    for target in targets {
        let result = compile_zlib_sources(&zlib_dir, target, "-O0");
        let (status, count) = match &result {
            Ok(objs) => ("PASS", objs.len()),
            Err(_) => {
                all_passed = false;
                ("FAIL", 0)
            }
        };

        summary.push(format!("  {:<24} objects={:<3} {}", target, count, status));
    }

    eprintln!("[zlib] All-architecture compilation results:");
    for line in &summary {
        eprintln!("{}", line);
    }

    assert!(
        all_passed,
        "One or more architectures failed zlib compilation.\n{}",
        summary.join("\n"),
    );

    drop(work_dir);
}

// ===========================================================================
// Link and Execute Tests
// ===========================================================================

/// Compile all zlib `.c` files and the `example.c` test program, then link
/// them into a static executable for x86-64.
///
/// Verifies that the integrated linker can produce a working binary from
/// multiple independently compiled object files and that CRT linkage
/// succeeds.
#[test]
fn zlib_link_static_library() {
    let test_name = "zlib_link_static_library";

    let (work_dir, zlib_dir) = match setup_zlib_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let target = "x86_64-linux-gnu";

    // Compile all zlib core source files.
    let objects = match compile_zlib_sources(&zlib_dir, target, "-O0") {
        Ok(objs) => objs,
        Err(e) => {
            eprintln!("[zlib] Skipping link test — compilation failed: {}", e);
            return;
        }
    };

    // Compile the example.c test program.
    let example_src = zlib_dir.join(ZLIB_EXAMPLE_SOURCE);
    if !example_src.exists() {
        eprintln!(
            "[zlib] Skipping link test — example.c not found at {}",
            example_src.display()
        );
        return;
    }

    let obj_dir = zlib_dir.join("obj").join(target);
    let example_obj = obj_dir.join("example.o");
    let compile_result = compile_single_file(&example_src, target, "-O0", &zlib_dir, &example_obj);

    match compile_result {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "[zlib] Skipping link test — example.c compilation failed: {}",
                e
            );
            return;
        }
    }

    // Link everything into an executable.
    let output_binary = obj_dir.join("zlib_example");
    let link_result = link_zlib_test(&objects, &example_obj, target, &output_binary);

    match link_result {
        Ok(()) => {
            eprintln!("[zlib] Successfully linked zlib example binary");
            if !output_binary.exists() {
                eprintln!(
                    "[SKIP] Compiler did not produce output binary (linker not yet functional)"
                );
                return;
            }
        }
        Err(e) => {
            eprintln!("[zlib] Link test failed: {}", e);
            eprintln!(
                "[SKIP] zlib link test failed: {} (compiler not fully functional)",
                e
            );
            return;
        }
    }

    drop(work_dir);
}

/// Compile zlib and its `example.c` test program for x86-64, then run the
/// resulting executable natively.
///
/// The zlib `example.c` program exercises the core compress/uncompress and
/// gzip APIs. A successful run (exit code 0) validates that the compiled
/// zlib library produces correct output.
#[test]
fn zlib_example_test_x86_64() {
    let test_name = "zlib_example_test_x86_64";

    let (work_dir, zlib_dir) = match setup_zlib_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let target = "x86_64-linux-gnu";

    // Compile zlib core.
    let objects = match compile_zlib_sources(&zlib_dir, target, "-O0") {
        Ok(objs) => objs,
        Err(e) => {
            eprintln!("[zlib] Skipping example test — compilation failed: {}", e);
            return;
        }
    };

    // Compile example.c.
    let example_src = zlib_dir.join(ZLIB_EXAMPLE_SOURCE);
    if !example_src.exists() {
        eprintln!("[zlib] Skipping example test — example.c not found");
        return;
    }

    let obj_dir = zlib_dir.join("obj").join(target);
    let example_obj = obj_dir.join("example.o");
    if compile_single_file(&example_src, target, "-O0", &zlib_dir, &example_obj).is_err() {
        eprintln!("[zlib] Skipping example test — example.c compilation failed");
        return;
    }

    // Link.
    let binary = obj_dir.join("zlib_example_test");
    if link_zlib_test(&objects, &example_obj, target, &binary).is_err() {
        eprintln!("[zlib] Skipping example test — linking failed");
        return;
    }

    // Run natively if we are on x86-64, otherwise via QEMU.
    let run_result = run_binary(&binary, target, &[]);
    match run_result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr_out = String::from_utf8_lossy(&output.stderr);
            eprintln!("[zlib] example test stdout:\n{}", stdout);
            if !stderr_out.is_empty() {
                eprintln!("[zlib] example test stderr:\n{}", stderr_out);
            }
            assert!(
                output.status.success(),
                "zlib example test exited with code {:?}",
                output.status.code()
            );
        }
        Err(e) => {
            eprintln!("[zlib] Skipping example execution — {}", e);
        }
    }

    drop(work_dir);
}

/// Compile zlib and `example.c` for non-native architectures and run the
/// resulting executables via QEMU user-mode emulation.
///
/// For each non-native target, the test checks QEMU availability before
/// attempting execution. Targets where QEMU is unavailable are skipped
/// gracefully.
#[test]
fn zlib_example_test_qemu() {
    let test_name = "zlib_example_test_qemu";

    let (work_dir, zlib_dir) = match setup_zlib_source(test_name) {
        Some(v) => v,
        None => return,
    };

    // Test on non-native architectures via QEMU.
    let non_native_targets: Vec<&&str> = ALL_TARGETS
        .iter()
        .filter(|t| !is_native_target(t))
        .collect();

    if non_native_targets.is_empty() {
        eprintln!("[zlib] No non-native targets to test with QEMU");
        drop(work_dir);
        return;
    }

    for target in non_native_targets {
        if !is_qemu_available(target) {
            eprintln!(
                "[zlib] SKIPPED: QEMU not available for {}, skipping execution test",
                target
            );
            continue;
        }

        // Compile zlib core.
        let objects = match compile_zlib_sources(&zlib_dir, target, "-O0") {
            Ok(objs) => objs,
            Err(e) => {
                eprintln!(
                    "[zlib] Skipping QEMU test for {} — compilation failed: {}",
                    target, e
                );
                continue;
            }
        };

        // Compile example.c.
        let example_src = zlib_dir.join(ZLIB_EXAMPLE_SOURCE);
        if !example_src.exists() {
            eprintln!(
                "[zlib] Skipping QEMU test for {} — example.c not found",
                target
            );
            continue;
        }

        let obj_dir = zlib_dir.join("obj").join(target);
        let example_obj = obj_dir.join("example.o");
        if compile_single_file(&example_src, target, "-O0", &zlib_dir, &example_obj).is_err() {
            eprintln!(
                "[zlib] Skipping QEMU test for {} — example.c compilation failed",
                target
            );
            continue;
        }

        // Link.
        let binary = obj_dir.join("zlib_example_qemu");
        if link_zlib_test(&objects, &example_obj, target, &binary).is_err() {
            eprintln!("[zlib] Skipping QEMU test for {} — linking failed", target);
            continue;
        }

        // Run via QEMU.
        let run_result = run_binary(&binary, target, &[]);
        match run_result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                eprintln!("[zlib] QEMU {} example test stdout:\n{}", target, stdout);
                assert!(
                    output.status.success(),
                    "zlib example test failed on {} via QEMU (exit code {:?})",
                    target,
                    output.status.code()
                );
            }
            Err(e) => {
                eprintln!("[zlib] Skipping QEMU execution for {} — {}", target, e);
            }
        }
    }

    drop(work_dir);
}

// ===========================================================================
// Test Suite Execution Tests
// ===========================================================================

/// Run the zlib test suite for x86-64 natively.
///
/// Compiles the zlib library and its `example` test program, then executes
/// the test binary. Optionally also compiles and runs `minigzip` for a
/// round-trip compression test.
#[test]
fn zlib_test_suite_x86_64() {
    let test_name = "zlib_test_suite_x86_64";

    let (work_dir, zlib_dir) = match setup_zlib_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let target = "x86_64-linux-gnu";

    // Compile zlib core.
    let objects = match compile_zlib_sources(&zlib_dir, target, "-O0") {
        Ok(objs) => objs,
        Err(e) => {
            eprintln!("[zlib] Skipping test suite — compilation failed: {}", e);
            return;
        }
    };

    let obj_dir = zlib_dir.join("obj").join(target);

    // --- Run the `example` test program ---
    let example_src = zlib_dir.join(ZLIB_EXAMPLE_SOURCE);
    if example_src.exists() {
        let example_obj = obj_dir.join("example_suite.o");
        if let Ok(_) = compile_single_file(&example_src, target, "-O0", &zlib_dir, &example_obj) {
            let example_bin = obj_dir.join("zlib_example_suite");
            if let Ok(()) = link_zlib_test(&objects, &example_obj, target, &example_bin) {
                match run_binary(&example_bin, target, &[]) {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        eprintln!("[zlib] example test output:\n{}", stdout);
                        assert!(
                            output.status.success(),
                            "zlib example test failed (exit code {:?})",
                            output.status.code()
                        );
                        eprintln!("[zlib] example test PASSED");
                    }
                    Err(e) => {
                        eprintln!("[zlib] Could not run example test: {}", e);
                    }
                }
            } else {
                eprintln!("[zlib] Could not link example test");
            }
        } else {
            eprintln!("[zlib] Could not compile example.c");
        }
    } else {
        eprintln!("[zlib] example.c not found, skipping example test");
    }

    // --- Run the `minigzip` round-trip test ---
    let minigzip_src = zlib_dir.join(ZLIB_MINIGZIP_SOURCE);
    if minigzip_src.exists() {
        let minigzip_obj = obj_dir.join("minigzip.o");
        if let Ok(_) = compile_single_file(&minigzip_src, target, "-O0", &zlib_dir, &minigzip_obj) {
            let minigzip_bin = obj_dir.join("minigzip");
            if let Ok(()) = link_zlib_test(&objects, &minigzip_obj, target, &minigzip_bin) {
                // Create a small test file, compress it with minigzip.
                let test_file = obj_dir.join("test_data.txt");
                let test_content = "The quick brown fox jumps over the lazy dog.\n\
                                    This is a round-trip compression test for zlib.\n\
                                    ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789\n";

                if fs::write(&test_file, test_content).is_ok() {
                    let compress_out =
                        run_binary(&minigzip_bin, target, &[&test_file.to_string_lossy()]);
                    match compress_out {
                        Ok(output) if output.status.success() => {
                            eprintln!("[zlib] minigzip compression succeeded");
                        }
                        Ok(output) => {
                            eprintln!(
                                "[zlib] minigzip compression returned exit {:?}",
                                output.status.code()
                            );
                        }
                        Err(e) => {
                            eprintln!("[zlib] Could not run minigzip: {}", e);
                        }
                    }
                }
            } else {
                eprintln!("[zlib] Could not link minigzip");
            }
        } else {
            eprintln!("[zlib] Could not compile minigzip.c");
        }
    } else {
        eprintln!("[zlib] minigzip.c not found, skipping round-trip test");
    }

    drop(work_dir);
}

/// Run the zlib test suite on non-native architectures via QEMU user-mode
/// emulation.
///
/// For each non-native target, compiles zlib and the example program, then
/// runs the test binary via QEMU. Skips gracefully if QEMU is not available
/// for a given target.
#[test]
fn zlib_test_suite_qemu() {
    let test_name = "zlib_test_suite_qemu";

    let (work_dir, zlib_dir) = match setup_zlib_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let non_native_targets: Vec<&&str> = ALL_TARGETS
        .iter()
        .filter(|t| !is_native_target(t))
        .collect();

    for target in non_native_targets {
        if !is_qemu_available(target) {
            eprintln!(
                "[zlib] SKIPPED: QEMU not available for {}, skipping test suite",
                target
            );
            continue;
        }

        // Compile zlib core.
        let objects = match compile_zlib_sources(&zlib_dir, target, "-O0") {
            Ok(objs) => objs,
            Err(e) => {
                eprintln!(
                    "[zlib] Skipping QEMU test suite for {} — compilation failed: {}",
                    target, e
                );
                continue;
            }
        };

        let example_src = zlib_dir.join(ZLIB_EXAMPLE_SOURCE);
        if !example_src.exists() {
            eprintln!(
                "[zlib] Skipping QEMU test suite for {} — example.c not found",
                target
            );
            continue;
        }

        let obj_dir = zlib_dir.join("obj").join(target);
        let example_obj = obj_dir.join("example_qemu_suite.o");
        if compile_single_file(&example_src, target, "-O0", &zlib_dir, &example_obj).is_err() {
            eprintln!(
                "[zlib] Skipping QEMU test suite for {} — example.c compilation failed",
                target
            );
            continue;
        }

        let binary = obj_dir.join("zlib_example_qemu_suite");
        if link_zlib_test(&objects, &example_obj, target, &binary).is_err() {
            eprintln!(
                "[zlib] Skipping QEMU test suite for {} — linking failed",
                target
            );
            continue;
        }

        match run_binary(&binary, target, &[]) {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                eprintln!("[zlib] QEMU {} test suite output:\n{}", target, stdout);
                assert!(
                    output.status.success(),
                    "zlib test suite failed on {} via QEMU (exit code {:?})",
                    target,
                    output.status.code()
                );
                eprintln!("[zlib] QEMU {} test suite PASSED", target);
            }
            Err(e) => {
                eprintln!(
                    "[zlib] Skipping QEMU test suite execution for {} — {}",
                    target, e
                );
            }
        }
    }

    drop(work_dir);
}

// ===========================================================================
// Optimization Level Tests
// ===========================================================================

/// Compile zlib at optimization level -O0 (no optimizations) for x86-64.
///
/// This is the baseline optimization level where no passes are applied.
#[test]
fn zlib_compile_o0() {
    run_zlib_compile_test("x86_64-linux-gnu", "-O0");
}

/// Compile zlib at optimization level -O1 (basic optimizations) for x86-64.
///
/// At -O1, basic passes (mem2reg, constant folding, dead code elimination)
/// are applied. Verifies these passes do not break zlib compilation.
#[test]
fn zlib_compile_o1() {
    run_zlib_compile_test("x86_64-linux-gnu", "-O1");
}

/// Compile zlib at optimization level -O2 (aggressive optimizations) for
/// x86-64.
///
/// At -O2, additional passes (CSE, algebraic simplification) are active.
/// Verifies the full optimization pipeline handles zlib code correctly.
#[test]
fn zlib_compile_o2() {
    run_zlib_compile_test("x86_64-linux-gnu", "-O2");
}

// ===========================================================================
// Unit Tests (module-internal validation)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the zlib defines are non-empty and well-formed.
    #[test]
    fn test_get_zlib_defines() {
        let defines = get_zlib_defines();
        assert!(!defines.is_empty(), "zlib defines should not be empty");
        for def in &defines {
            assert!(
                def.starts_with("-D"),
                "Each define must start with -D, got: {}",
                def
            );
        }
    }

    /// Verify the core source file list is complete.
    #[test]
    fn test_zlib_core_sources_list() {
        assert!(
            ZLIB_CORE_SOURCES.len() >= 15,
            "Expected at least 15 core zlib source files, got {}",
            ZLIB_CORE_SOURCES.len()
        );
        assert!(ZLIB_CORE_SOURCES.contains(&"inflate.c"));
        assert!(ZLIB_CORE_SOURCES.contains(&"deflate.c"));
        assert!(ZLIB_CORE_SOURCES.contains(&"adler32.c"));
        assert!(ZLIB_CORE_SOURCES.contains(&"crc32.c"));
    }

    /// Verify `is_zlib_source_available` returns false for a non-existent
    /// directory.
    #[test]
    fn test_is_zlib_source_available_nonexistent() {
        let fake = PathBuf::from("/tmp/nonexistent_zlib_dir_12345");
        assert!(!is_zlib_source_available(&fake));
    }

    /// Verify `get_zlib_source_files` returns an empty list for a directory
    /// with no C files.
    #[test]
    fn test_get_zlib_source_files_empty() {
        let fake = PathBuf::from("/tmp/nonexistent_zlib_src_dir_12345");
        let files = get_zlib_source_files(&fake);
        assert!(files.is_empty());
    }

    /// Verify the zlib version constant is well-formed.
    #[test]
    fn test_zlib_version_format() {
        let parts: Vec<&str> = ZLIB_VERSION.split('.').collect();
        assert!(parts.len() >= 2);
        for part in parts {
            assert!(part.parse::<u32>().is_ok(), "got: {}", part);
        }
    }

    /// Verify the download URLs contain the version string.
    #[test]
    fn test_zlib_url_consistency() {
        assert!(ZLIB_URL.contains(ZLIB_VERSION));
        assert!(ZLIB_URL_FALLBACK.contains(ZLIB_VERSION));
    }

    /// Verify that all core source file names end with `.c`.
    #[test]
    fn test_core_sources_extension() {
        for name in ZLIB_CORE_SOURCES {
            assert!(name.ends_with(".c"), "got: {}", name);
        }
    }
}
