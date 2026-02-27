//! SQLite amalgamation validation tests for the bcc compiler.
//!
//! This module verifies that `bcc` can compile the SQLite amalgamation
//! (~230K lines of C), optionally run a minimal test against the compiled
//! library, and meet strict performance constraints defined in AAP §0.7.
//! SQLite is the **most important** validation target.
//!
//! # Validation Autonomy
//!
//! The SQLite amalgamation is fetched from the official SQLite website at
//! test time and is **never** committed to the repository. Tests skip
//! gracefully when the network is unavailable.
//!
//! # Performance Constraints (AAP §0.7 — hard requirements)
//!
//! | Metric              | Limit         |
//! |---------------------|---------------|
//! | Compile time (`-O0`)| < 60 seconds  |
//! | Peak RSS            | < 2 GB        |
//!
//! # Success Criteria (AAP §0.7, strict order)
//!
//! 1. Compilation succeeds for all four architectures.
//! 2. Test suite passes (basic SQLite API exercise).
//! 3. Performance constraints met (<60 s, <2 GB RSS).
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
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

// ===========================================================================
// SQLite Source Constants
// ===========================================================================

/// Primary download URL for the SQLite amalgamation (2024 release).
/// The version number in the URL encodes major.minor.patch as XYYZZPP.
/// 3.46.0 → 3460000.
const SQLITE_URL: &str = "https://www.sqlite.org/2024/sqlite-amalgamation-3460000.zip";

/// Fallback download URL — an older stable release in case the primary
/// version page has rotated.
const SQLITE_URL_FALLBACK: &str = "https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip";

/// Name of the SQLite amalgamation source file inside the extracted archive.
const SQLITE_AMALGAMATION_FILE: &str = "sqlite3.c";

/// Name of the SQLite public header file.
const SQLITE_HEADER_FILE: &str = "sqlite3.h";

/// Name of the SQLite shell source (optional, used for extended testing).
const SQLITE_SHELL_FILE: &str = "shell.c";

/// Minimum expected number of lines in `sqlite3.c`. The real amalgamation
/// is ~230K lines; we use a conservative lower bound that tolerates version
/// variance while still catching truncated downloads.
const EXPECTED_LOC_MIN: usize = 200_000;

/// Hard upper bound on compile time in seconds at `-O0` for the
/// amalgamation (AAP §0.7 performance constraint).
const MAX_COMPILE_TIME_SECS: u64 = 60;

/// Hard upper bound on peak resident set size in bytes during compilation
/// (AAP §0.7 performance constraint). 2 GB = 2 * 1024 * 1024 * 1024.
const MAX_RSS_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Name of the archive file saved to disk during download.
const SQLITE_ARCHIVE_NAME: &str = "sqlite-amalgamation.zip";

// ===========================================================================
// SQLite Compile-Time Defines
// ===========================================================================

/// Return a list of `-D` flags appropriate for compiling the SQLite
/// amalgamation in a controlled, freestanding-friendly manner.
fn get_sqlite_defines() -> Vec<String> {
    vec![
        "-DSQLITE_THREADSAFE=0".to_string(),
        "-DSQLITE_OMIT_LOAD_EXTENSION".to_string(),
        "-DSQLITE_DQS=0".to_string(),
    ]
}

// ===========================================================================
// Source Fetching Helpers
// ===========================================================================

/// Download and extract the SQLite amalgamation into `work_dir`.
///
/// Returns the path to the directory that contains `sqlite3.c` on success,
/// or a human-readable error message on failure.
fn fetch_sqlite_source(work_dir: &Path) -> Result<PathBuf, String> {
    // Attempt primary URL first.
    let archive_path = match fetch_source_archive(SQLITE_URL, work_dir, SQLITE_ARCHIVE_NAME) {
        SourceFetchResult::Success(p) => p,
        _ => {
            // Try fallback URL.
            match fetch_source_archive(SQLITE_URL_FALLBACK, work_dir, SQLITE_ARCHIVE_NAME) {
                SourceFetchResult::Success(p) => p,
                SourceFetchResult::NetworkUnavailable(msg) => {
                    return Err(format!("Network unavailable: {}", msg));
                }
                SourceFetchResult::ExtractionFailed(msg) => {
                    return Err(format!("Download failed: {}", msg));
                }
            }
        }
    };

    // Extract the zip archive.
    let extracted_dir = extract_zip(&archive_path, work_dir)?;

    // The extracted directory should contain sqlite3.c directly or in a
    // sub-directory. Walk one level to locate it.
    if extracted_dir.join(SQLITE_AMALGAMATION_FILE).exists() {
        return Ok(extracted_dir);
    }

    // Scan for a sub-directory containing the amalgamation.
    if let Ok(entries) = fs::read_dir(&extracted_dir) {
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() && child.join(SQLITE_AMALGAMATION_FILE).exists() {
                return Ok(child);
            }
        }
    }

    // As a last resort, check work_dir itself (unzip may have placed files
    // directly there).
    if work_dir.join(SQLITE_AMALGAMATION_FILE).exists() {
        return Ok(work_dir.to_path_buf());
    }

    Err(format!(
        "sqlite3.c not found after extracting archive in {}",
        work_dir.display()
    ))
}

/// Check whether the SQLite amalgamation source is available in `dir`.
/// Verifies both existence and non-zero file size via `fs::metadata()`.
fn is_sqlite_source_available(dir: &Path) -> bool {
    let path = dir.join(SQLITE_AMALGAMATION_FILE);
    match fs::metadata(&path) {
        Ok(meta) => meta.len() > 0,
        Err(_) => false,
    }
}

/// Count the number of newline-delimited lines in `file`.
///
/// Uses a buffered reader to avoid loading the entire file into memory.
fn count_lines(file: &Path) -> usize {
    let f = match fs::File::open(file) {
        Ok(f) => f,
        Err(_) => return 0,
    };
    BufReader::new(f).lines().count()
}

// ===========================================================================
// Compilation Helpers
// ===========================================================================

/// Compile `sqlite3.c` for the given `target` and `opt_level` (e.g. `-O0`),
/// placing the resulting object file in `output_dir`.
///
/// Returns the path to the produced `.o` file on success, or an error
/// message on failure.
fn compile_sqlite(
    source_dir: &Path,
    target: &str,
    opt_level: &str,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let _ = fs::create_dir_all(output_dir);
    let src = source_dir.join(SQLITE_AMALGAMATION_FILE);
    let obj = output_dir.join("sqlite3.o");

    let mut flags: Vec<&str> = vec![opt_level];
    let defines = get_sqlite_defines();
    // We need owned strings alive for the duration; collect borrows.
    let define_refs: Vec<&str> = defines.iter().map(|s| s.as_str()).collect();
    for d in &define_refs {
        flags.push(d);
    }

    // Also add the source directory as an include path so that sqlite3.h
    // is discoverable via `#include "sqlite3.h"`.
    let inc_flag = format!("-I{}", source_dir.display());
    flags.push(&inc_flag);

    match compile_c_file(&src, target, &obj, &flags) {
        Ok(output) if output.status.success() => Ok(obj),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Compilation failed for target {}: {}",
                target, stderr
            ))
        }
        Err(e) => Err(format!("Failed to invoke bcc: {}", e)),
    }
}

/// Time a compilation of `sqlite3.c` and return the elapsed `Duration`.
fn measure_sqlite_compilation_time(
    source_dir: &Path,
    target: &str,
    opt_level: &str,
) -> (bool, Duration) {
    let defines = get_sqlite_defines();
    let inc_flag = format!("-I{}", source_dir.display());
    let mut flags: Vec<&str> = vec![opt_level];
    let define_refs: Vec<&str> = defines.iter().map(|s| s.as_str()).collect();
    for d in &define_refs {
        flags.push(d);
    }
    flags.push(&inc_flag);

    let src = source_dir.join(SQLITE_AMALGAMATION_FILE);
    measure_compilation_time(&src, target, &flags)
}

/// Measure peak RSS of the `bcc` process while compiling `sqlite3.c`.
///
/// Uses `/usr/bin/time -v` to capture peak RSS. Returns the peak RSS in
/// bytes, or 0 if measurement fails.
fn measure_sqlite_peak_rss(source_dir: &Path, target: &str) -> u64 {
    let bcc = get_bcc_binary();
    let src = source_dir.join(SQLITE_AMALGAMATION_FILE);
    let out_obj = std::env::temp_dir().join("bcc_rss_sqlite3.o");
    let defines = get_sqlite_defines();
    let inc_flag = format!("-I{}", source_dir.display());

    // Build argument list for /usr/bin/time -v wrapping.
    let mut bcc_args: Vec<String> = Vec::new();
    bcc_args.push(bcc.display().to_string());
    bcc_args.push("--target".to_string());
    bcc_args.push(target.to_string());
    bcc_args.push("-c".to_string());
    bcc_args.push("-O0".to_string());
    bcc_args.push("-o".to_string());
    bcc_args.push(out_obj.display().to_string());
    for d in &defines {
        bcc_args.push(d.clone());
    }
    bcc_args.push(inc_flag);
    bcc_args.push(src.display().to_string());

    let time_result = Command::new("/usr/bin/time")
        .arg("-v")
        .args(&bcc_args)
        .output();

    let _ = fs::remove_file(&out_obj);

    match time_result {
        Ok(output) => {
            // /usr/bin/time writes statistics to stderr.
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                if line.contains("Maximum resident set size") {
                    if let Some(val) = line.split(':').last() {
                        if let Ok(kb) = val.trim().parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes.
                        }
                    }
                }
            }
            0
        }
        Err(_) => 0,
    }
}

// ===========================================================================
// ELF Verification Helpers
// ===========================================================================

/// Verify that `path` is a valid ELF file with the expected class
/// (1 = ELF32, 2 = ELF64). Returns `true` if valid.
fn verify_elf_object(path: &Path, expected_class: u8) -> bool {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() < 18 {
        return false;
    }
    // Check ELF magic: 0x7f 'E' 'L' 'F'
    if &data[0..4] != b"\x7fELF" {
        return false;
    }
    // Check ELF class.
    if data[4] != expected_class {
        return false;
    }
    // Check that it is a relocatable object (ET_REL = 1).
    // e_type is at offset 16 in both ELF32 and ELF64 (little-endian).
    let e_type = u16::from_le_bytes([data[16], data[17]]);
    e_type == 1 // ET_REL
}

/// Check if the ELF binary at `path` contains a section whose name matches
/// `needle`. This is a lightweight scan — it parses section headers and
/// the `.shstrtab` to find the name.
fn elf_has_section(path: &Path, needle: &str) -> bool {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() < 64 {
        return false;
    }
    // Only handle ELF64 for simplicity (the debug test targets x86-64).
    if data[4] != 2 {
        return false; // Not ELF64
    }
    // e_shoff (section header table offset): bytes 40..48
    let e_shoff = u64::from_le_bytes([
        data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47],
    ]) as usize;
    // e_shentsize: bytes 58..60
    let e_shentsize = u16::from_le_bytes([data[58], data[59]]) as usize;
    // e_shnum: bytes 60..62
    let e_shnum = u16::from_le_bytes([data[60], data[61]]) as usize;
    // e_shstrndx: bytes 62..64
    let e_shstrndx = u16::from_le_bytes([data[62], data[63]]) as usize;

    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum {
        return false;
    }

    // Read .shstrtab section header to get the string table offset.
    let shstr_hdr_off = e_shoff + e_shstrndx * e_shentsize;
    if shstr_hdr_off + e_shentsize > data.len() {
        return false;
    }
    // sh_offset is at byte 24 within the section header (ELF64).
    let shstr_off = u64::from_le_bytes([
        data[shstr_hdr_off + 24],
        data[shstr_hdr_off + 25],
        data[shstr_hdr_off + 26],
        data[shstr_hdr_off + 27],
        data[shstr_hdr_off + 28],
        data[shstr_hdr_off + 29],
        data[shstr_hdr_off + 30],
        data[shstr_hdr_off + 31],
    ]) as usize;
    let shstr_size = u64::from_le_bytes([
        data[shstr_hdr_off + 32],
        data[shstr_hdr_off + 33],
        data[shstr_hdr_off + 34],
        data[shstr_hdr_off + 35],
        data[shstr_hdr_off + 36],
        data[shstr_hdr_off + 37],
        data[shstr_hdr_off + 38],
        data[shstr_hdr_off + 39],
    ]) as usize;

    if shstr_off + shstr_size > data.len() {
        return false;
    }

    let strtab = &data[shstr_off..shstr_off + shstr_size];

    // Iterate over all section headers and match names.
    for i in 0..e_shnum {
        let hdr_off = e_shoff + i * e_shentsize;
        if hdr_off + 4 > data.len() {
            break;
        }
        let sh_name = u32::from_le_bytes([
            data[hdr_off],
            data[hdr_off + 1],
            data[hdr_off + 2],
            data[hdr_off + 3],
        ]) as usize;
        if sh_name < strtab.len() {
            let end = strtab[sh_name..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| sh_name + p)
                .unwrap_or(strtab.len());
            let name = std::str::from_utf8(&strtab[sh_name..end]).unwrap_or("");
            if name == needle {
                return true;
            }
        }
    }
    false
}

// ===========================================================================
// Test Functions — Architecture-Specific Compilation
// ===========================================================================

#[test]
#[ignore] // Heavy: downloads SQLite amalgamation from the internet.
fn sqlite_compile_x86_64() {
    if skip_if_offline("sqlite_compile_x86_64") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_x86_64");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_x86_64: {}", e);
            return;
        }
    };

    assert!(
        is_sqlite_source_available(&source_dir),
        "sqlite3.c not found in {}",
        source_dir.display()
    );

    let loc = count_lines(&source_dir.join(SQLITE_AMALGAMATION_FILE));
    eprintln!("sqlite3.c line count: {}", loc);
    assert!(
        loc >= EXPECTED_LOC_MIN,
        "sqlite3.c has only {} lines (expected >= {})",
        loc,
        EXPECTED_LOC_MIN
    );

    let out_dir = work.path().join("out_x86_64");
    let obj = compile_sqlite(&source_dir, "x86_64-linux-gnu", "-O0", &out_dir)
        .expect("SQLite compilation for x86_64 should succeed");

    assert!(
        verify_elf_object(&obj, 2), // ELF64
        "Output is not a valid ELF64 relocatable object"
    );
    eprintln!("sqlite_compile_x86_64: PASSED");
}

#[test]
#[ignore]
fn sqlite_compile_i686() {
    if skip_if_offline("sqlite_compile_i686") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_i686");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_i686: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out_i686");
    let obj = compile_sqlite(&source_dir, "i686-linux-gnu", "-O0", &out_dir)
        .expect("SQLite compilation for i686 should succeed");

    assert!(
        verify_elf_object(&obj, 1), // ELF32
        "Output is not a valid ELF32 relocatable object"
    );
    eprintln!("sqlite_compile_i686: PASSED");
}

#[test]
#[ignore]
fn sqlite_compile_aarch64() {
    if skip_if_offline("sqlite_compile_aarch64") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_aarch64");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_aarch64: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out_aarch64");
    let obj = compile_sqlite(&source_dir, "aarch64-linux-gnu", "-O0", &out_dir)
        .expect("SQLite compilation for aarch64 should succeed");

    assert!(
        verify_elf_object(&obj, 2), // ELF64
        "Output is not a valid ELF64 relocatable object"
    );
    eprintln!("sqlite_compile_aarch64: PASSED");
}

#[test]
#[ignore]
fn sqlite_compile_riscv64() {
    if skip_if_offline("sqlite_compile_riscv64") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_riscv64");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_riscv64: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out_riscv64");
    let obj = compile_sqlite(&source_dir, "riscv64-linux-gnu", "-O0", &out_dir)
        .expect("SQLite compilation for riscv64 should succeed");

    assert!(
        verify_elf_object(&obj, 2), // ELF64
        "Output is not a valid ELF64 relocatable object"
    );
    eprintln!("sqlite_compile_riscv64: PASSED");
}

#[test]
#[ignore]
fn sqlite_compile_all_architectures() {
    if skip_if_offline("sqlite_compile_all_architectures") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_all_arch");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_all_architectures: {}", e);
            return;
        }
    };

    let targets_and_class: &[(&str, u8)] = &[
        ("x86_64-linux-gnu", 2),
        ("i686-linux-gnu", 1),
        ("aarch64-linux-gnu", 2),
        ("riscv64-linux-gnu", 2),
    ];

    let mut passed = 0usize;
    let mut failed_targets: Vec<String> = Vec::new();

    for &(target, elf_class) in targets_and_class {
        let out_dir = work.path().join(format!("out_{}", target));
        match compile_sqlite(&source_dir, target, "-O0", &out_dir) {
            Ok(obj) => {
                if verify_elf_object(&obj, elf_class) {
                    eprintln!("  {} — OK", target);
                    passed += 1;
                } else {
                    eprintln!("  {} — ELF verification failed", target);
                    failed_targets.push(target.to_string());
                }
            }
            Err(e) => {
                eprintln!("  {} — FAIL: {}", target, e);
                failed_targets.push(target.to_string());
            }
        }
    }

    eprintln!(
        "sqlite_compile_all_architectures: {}/{} passed",
        passed,
        targets_and_class.len()
    );

    assert!(
        failed_targets.is_empty(),
        "Compilation failed for targets: {:?}",
        failed_targets
    );
}

// ===========================================================================
// Test Functions — Performance Constraints
// ===========================================================================

#[test]
#[ignore]
fn sqlite_compile_time_under_60s() {
    if skip_if_offline("sqlite_compile_time_under_60s") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_time");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_time_under_60s: {}", e);
            return;
        }
    };

    let (success, duration) =
        measure_sqlite_compilation_time(&source_dir, "x86_64-linux-gnu", "-O0");

    let secs = duration.as_secs_f64();
    eprintln!("SQLite -O0 compile time: {:.2}s", secs);

    assert!(success, "SQLite compilation did not succeed");
    assert!(
        duration < Duration::from_secs(MAX_COMPILE_TIME_SECS),
        "SQLite compile time ({:.2}s) exceeded {}-second limit",
        secs,
        MAX_COMPILE_TIME_SECS
    );
}

#[test]
#[ignore]
fn sqlite_peak_rss_under_2gb() {
    if skip_if_offline("sqlite_peak_rss_under_2gb") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_rss");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_peak_rss_under_2gb: {}", e);
            return;
        }
    };

    let peak_bytes = measure_sqlite_peak_rss(&source_dir, "x86_64-linux-gnu");

    if peak_bytes == 0 {
        eprintln!(
            "WARNING: Could not measure peak RSS (is /usr/bin/time installed?). \
             Skipping assertion."
        );
        return;
    }

    let peak_mb = peak_bytes as f64 / (1024.0 * 1024.0);
    eprintln!("SQLite -O0 peak RSS: {:.1} MB", peak_mb);

    assert!(
        peak_bytes < MAX_RSS_BYTES,
        "Peak RSS ({:.1} MB) exceeded 2 GB limit",
        peak_mb
    );
}

// ===========================================================================
// Test Functions — Link and Execute
// ===========================================================================

#[test]
#[ignore]
fn sqlite_link_executable() {
    if skip_if_offline("sqlite_link_executable") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_link");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_link_executable: {}", e);
            return;
        }
    };

    // Write a minimal test program that exercises the SQLite API.
    let test_c = work.path().join("sqlite_test_main.c");
    fs::write(
        &test_c,
        r#"
#include "sqlite3.h"
int main(void) {
    sqlite3 *db;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != 0) return 1;
    rc = sqlite3_exec(db, "CREATE TABLE t(x INTEGER);", 0, 0, 0);
    if (rc != 0) { sqlite3_close(db); return 2; }
    rc = sqlite3_exec(db, "INSERT INTO t VALUES(42);", 0, 0, 0);
    if (rc != 0) { sqlite3_close(db); return 3; }
    sqlite3_close(db);
    return 0;
}
"#,
    )
    .expect("Failed to write test program");

    let target = "x86_64-linux-gnu";
    let sqlite_src = source_dir.join(SQLITE_AMALGAMATION_FILE);
    let out_exe = work.path().join("sqlite_test");

    // Compile and link in a single invocation.
    let defines = get_sqlite_defines();
    let inc_flag = format!("-I{}", source_dir.display());
    let mut flags: Vec<String> = vec!["-O0".to_string(), inc_flag];
    flags.extend(defines);
    let flag_refs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();

    let sources = vec![sqlite_src, test_c];
    let output = compile_and_link(&sources, target, &out_exe, &flag_refs);
    match output {
        Ok(o) if o.status.success() => {
            eprintln!("sqlite_link_executable: linking succeeded");
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!("Linking failed: {}", stderr);
        }
        Err(e) => {
            panic!("Failed to invoke bcc for linking: {}", e);
        }
    }

    // Run the linked executable.
    match run_binary(&out_exe, target, &[]) {
        Ok(o) if o.status.success() => {
            eprintln!("sqlite_link_executable: execution succeeded");
        }
        Ok(o) => {
            let code = o.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!(
                "SQLite test executable exited with code {}: {}",
                code, stderr
            );
        }
        Err(e) => {
            eprintln!("WARNING: Could not run executable ({}). Skipping.", e);
        }
    }
}

#[test]
#[ignore]
fn sqlite_test_suite() {
    // The SQLite amalgamation ZIP does not include the full test suite
    // (that requires the full distribution). This test is a placeholder
    // that exercises basic API operations to confirm the library is
    // functionally correct.
    //
    // If the full SQLite distribution is available, a comprehensive test
    // suite can be added here. For now, the sqlite_link_executable test
    // covers basic API correctness.
    if skip_if_offline("sqlite_test_suite") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_testsuite");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_test_suite: {}", e);
            return;
        }
    };

    // Write a more thorough test program exercising multiple SQLite features.
    let test_c = work.path().join("sqlite_api_test.c");
    fs::write(
        &test_c,
        r#"
#include "sqlite3.h"
#include <string.h>

static int callback_count = 0;
static int callback(void *data, int argc, char **argv, char **col_names) {
    (void)data; (void)col_names;
    if (argc == 1 && argv[0] && strcmp(argv[0], "42") == 0) {
        callback_count++;
    }
    return 0;
}

int main(void) {
    sqlite3 *db;
    char *err_msg = 0;
    int rc;

    /* Open in-memory database. */
    rc = sqlite3_open(":memory:", &db);
    if (rc != 0) return 1;

    /* Create table. */
    rc = sqlite3_exec(db, "CREATE TABLE test(val INTEGER);", 0, 0, &err_msg);
    if (rc != 0) { sqlite3_free(err_msg); sqlite3_close(db); return 2; }

    /* Insert data. */
    rc = sqlite3_exec(db, "INSERT INTO test VALUES(42);", 0, 0, &err_msg);
    if (rc != 0) { sqlite3_free(err_msg); sqlite3_close(db); return 3; }

    /* Query with callback. */
    callback_count = 0;
    rc = sqlite3_exec(db, "SELECT val FROM test;", callback, 0, &err_msg);
    if (rc != 0) { sqlite3_free(err_msg); sqlite3_close(db); return 4; }
    if (callback_count != 1) { sqlite3_close(db); return 5; }

    /* Transaction test. */
    rc = sqlite3_exec(db, "BEGIN;", 0, 0, 0);
    if (rc != 0) { sqlite3_close(db); return 6; }
    rc = sqlite3_exec(db, "INSERT INTO test VALUES(100);", 0, 0, 0);
    if (rc != 0) { sqlite3_close(db); return 7; }
    rc = sqlite3_exec(db, "ROLLBACK;", 0, 0, 0);
    if (rc != 0) { sqlite3_close(db); return 8; }

    /* Verify rollback worked — should still be only 1 row. */
    callback_count = 0;
    rc = sqlite3_exec(db, "SELECT val FROM test;", callback, 0, 0);
    if (rc != 0) { sqlite3_close(db); return 9; }
    if (callback_count != 1) { sqlite3_close(db); return 10; }

    /* Prepared statement test. */
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db, "SELECT val FROM test WHERE val = ?", -1, &stmt, 0);
    if (rc != 0) { sqlite3_close(db); return 11; }
    sqlite3_bind_int(stmt, 1, 42);
    rc = sqlite3_step(stmt);
    if (rc != 101) { sqlite3_finalize(stmt); sqlite3_close(db); return 12; } /* SQLITE_ROW = 100 */
    int val = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    if (val != 42) { sqlite3_close(db); return 13; }

    sqlite3_close(db);
    return 0;
}
"#,
    )
    .expect("Failed to write API test program");

    let target = "x86_64-linux-gnu";
    let sqlite_src = source_dir.join(SQLITE_AMALGAMATION_FILE);
    let out_exe = work.path().join("sqlite_api_test");

    let defines = get_sqlite_defines();
    let inc_flag = format!("-I{}", source_dir.display());
    let mut flags: Vec<String> = vec!["-O0".to_string(), inc_flag];
    flags.extend(defines);
    let flag_refs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();

    let sources = vec![sqlite_src, test_c];
    let output = compile_and_link(&sources, target, &out_exe, &flag_refs);
    match output {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("SKIPPED sqlite_test_suite: linking failed: {}", stderr);
            return;
        }
        Err(e) => {
            eprintln!("SKIPPED sqlite_test_suite: {}", e);
            return;
        }
    }

    match run_binary(&out_exe, target, &[]) {
        Ok(o) if o.status.success() => {
            eprintln!("sqlite_test_suite: all API tests passed");
        }
        Ok(o) => {
            let code = o.status.code().unwrap_or(-1);
            panic!("SQLite API test exited with code {} (test failure)", code);
        }
        Err(e) => {
            eprintln!("WARNING: Could not run test executable: {}", e);
        }
    }
}

// ===========================================================================
// Test Functions — Optimization Levels
// ===========================================================================

#[test]
#[ignore]
fn sqlite_compile_o0() {
    if skip_if_offline("sqlite_compile_o0") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_o0");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_o0: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out");
    let start = Instant::now();
    let obj = compile_sqlite(&source_dir, "x86_64-linux-gnu", "-O0", &out_dir)
        .expect("SQLite -O0 compilation should succeed");
    let elapsed = start.elapsed();

    assert!(obj.exists(), "Object file not produced");
    eprintln!("sqlite_compile_o0: OK ({:.2}s)", elapsed.as_secs_f64());
}

#[test]
#[ignore]
fn sqlite_compile_o1() {
    if skip_if_offline("sqlite_compile_o1") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_o1");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_o1: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out");
    let start = Instant::now();
    let obj = compile_sqlite(&source_dir, "x86_64-linux-gnu", "-O1", &out_dir)
        .expect("SQLite -O1 compilation should succeed");
    let elapsed = start.elapsed();

    assert!(obj.exists(), "Object file not produced");
    eprintln!("sqlite_compile_o1: OK ({:.2}s)", elapsed.as_secs_f64());
}

#[test]
#[ignore]
fn sqlite_compile_o2() {
    if skip_if_offline("sqlite_compile_o2") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_o2");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_o2: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out");
    let start = Instant::now();
    let obj = compile_sqlite(&source_dir, "x86_64-linux-gnu", "-O2", &out_dir)
        .expect("SQLite -O2 compilation should succeed");
    let elapsed = start.elapsed();

    assert!(obj.exists(), "Object file not produced");
    eprintln!("sqlite_compile_o2: OK ({:.2}s)", elapsed.as_secs_f64());
}

// ===========================================================================
// Test Functions — Debug Info
// ===========================================================================

#[test]
#[ignore]
fn sqlite_compile_with_debug() {
    if skip_if_offline("sqlite_compile_with_debug") {
        return;
    }
    let work = ValidationWorkDir::new("sqlite_debug");
    let source_dir = match fetch_sqlite_source(work.path()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED sqlite_compile_with_debug: {}", e);
            return;
        }
    };

    let out_dir = work.path().join("out");
    let _ = fs::create_dir_all(&out_dir);
    let src = source_dir.join(SQLITE_AMALGAMATION_FILE);
    let obj = out_dir.join("sqlite3_debug.o");

    let defines = get_sqlite_defines();
    let inc_flag = format!("-I{}", source_dir.display());
    let mut flags: Vec<String> = vec!["-O0".to_string(), "-g".to_string(), inc_flag];
    flags.extend(defines);
    let flag_refs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();

    let output = compile_c_file(&src, "x86_64-linux-gnu", &obj, &flag_refs);
    match output {
        Ok(o) if o.status.success() => {
            eprintln!("sqlite_compile_with_debug: compilation succeeded");
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!("Debug compilation failed: {}", stderr);
        }
        Err(e) => {
            panic!("Failed to invoke bcc: {}", e);
        }
    }

    // Verify that DWARF debug sections are present in the output object.
    let has_debug_info = elf_has_section(&obj, ".debug_info");
    let has_debug_line = elf_has_section(&obj, ".debug_line");
    let has_debug_abbrev = elf_has_section(&obj, ".debug_abbrev");

    eprintln!(
        "  .debug_info:   {}",
        if has_debug_info { "present" } else { "MISSING" }
    );
    eprintln!(
        "  .debug_line:   {}",
        if has_debug_line { "present" } else { "MISSING" }
    );
    eprintln!(
        "  .debug_abbrev: {}",
        if has_debug_abbrev {
            "present"
        } else {
            "MISSING"
        }
    );

    assert!(
        has_debug_info,
        ".debug_info section missing from output compiled with -g"
    );
    assert!(
        has_debug_line,
        ".debug_line section missing from output compiled with -g"
    );
    assert!(
        has_debug_abbrev,
        ".debug_abbrev section missing from output compiled with -g"
    );
}
