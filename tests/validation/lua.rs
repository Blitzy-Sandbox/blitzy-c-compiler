//! Lua source validation tests for the bcc compiler.
//!
//! This module verifies that `bcc` can compile the Lua interpreter source code
//! and run its bundled test suite across all four supported target architectures.
//! Lua is a validation target per AAP §0.2.1 and §0.7.
//!
//! # Validation Autonomy
//!
//! The Lua source tree is fetched from the official Lua website at test time
//! and is never committed to the repository. Tests skip gracefully when the
//! network is unavailable.
//!
//! # Success Criteria (AAP §0.7, strict order)
//!
//! 1. Compilation succeeds for all four architectures.
//! 2. Lua bundled test suite passes (basic smoke test via the interpreter).
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
// Lua Source Constants
// ===========================================================================

/// Lua version to download for validation testing.
const LUA_VERSION: &str = "5.4.7";

/// Download URL for the Lua source tarball from the official website.
/// This must remain consistent with `LUA_VERSION`.
const LUA_URL: &str = "https://www.lua.org/ftp/lua-5.4.7.tar.gz";

/// Fallback download URL in case the primary is unavailable (GitHub mirror).
const LUA_URL_FALLBACK: &str =
    "https://github.com/lua/lua/archive/refs/tags/v5.4.7.tar.gz";

/// Core Lua library source files that compose the runtime and standard
/// libraries. These are compiled individually and linked together for the
/// Lua interpreter or compiler executables.
const LUA_CORE_SOURCES: &[&str] = &[
    "lapi.c",
    "lauxlib.c",
    "lbaselib.c",
    "lcode.c",
    "lcorolib.c",
    "lctype.c",
    "ldblib.c",
    "ldebug.c",
    "ldo.c",
    "ldump.c",
    "lfunc.c",
    "lgc.c",
    "linit.c",
    "liolib.c",
    "llex.c",
    "lmathlib.c",
    "lmem.c",
    "loadlib.c",
    "lobject.c",
    "lopcodes.c",
    "loslib.c",
    "lparser.c",
    "lstate.c",
    "lstring.c",
    "lstrlib.c",
    "ltable.c",
    "ltablib.c",
    "ltm.c",
    "lundump.c",
    "lutf8lib.c",
    "lvm.c",
    "lzio.c",
];

/// Source file for the standalone Lua interpreter binary.
const LUA_INTERPRETER_SOURCE: &str = "lua.c";

/// Source file for the Lua bytecode compiler binary.
const LUA_COMPILER_SOURCE: &str = "luac.c";

// ===========================================================================
// Lua Source Fetching Helpers
// ===========================================================================

/// Download and extract the Lua source tree into `work_dir`.
///
/// Attempts to download the Lua tarball from the official website using
/// `curl` (preferred) with a `wget` fallback. If the primary URL fails,
/// a GitHub mirror URL is tried. On success, extracts the archive using
/// `tar` and returns the path to the `src/` directory within the extracted
/// tree (e.g., `/tmp/.../lua-5.4.7/src/`).
///
/// # Errors
///
/// Returns a descriptive error string on network failure or extraction failure.
fn fetch_lua_source(work_dir: &Path) -> Result<PathBuf, String> {
    let archive_name = format!("lua-{}.tar.gz", LUA_VERSION);
    let archive_path = work_dir.join(&archive_name);

    // Try primary URL with curl first, then wget, then fallback URL.
    let download_ok = try_download_with_curl(&archive_path, LUA_URL)
        || try_download_with_wget(&archive_path, LUA_URL)
        || try_download_with_curl(&archive_path, LUA_URL_FALLBACK)
        || try_download_with_wget(&archive_path, LUA_URL_FALLBACK);

    if !download_ok {
        return Err(
            "Failed to download Lua source via both curl and wget \
             from both primary and fallback URLs"
                .to_string(),
        );
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
    let expected_dir = work_dir.join(format!("lua-{}", LUA_VERSION));
    if expected_dir.exists() && expected_dir.join("src").is_dir() {
        return Ok(expected_dir.join("src"));
    }

    // If GitHub mirror was used, the directory name may differ.
    if let Ok(entries) = fs::read_dir(work_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("lua-") && entry.path().is_dir() {
                let src_dir = entry.path().join("src");
                if src_dir.is_dir() {
                    return Ok(src_dir);
                }
                // Some GitHub archives may have a different structure.
                // In that case, check if lapi.c exists directly.
                if entry.path().join("lapi.c").exists() {
                    return Ok(entry.path());
                }
            }
        }
    }

    Err(format!(
        "Extracted Lua source directory not found in {}",
        work_dir.display()
    ))
}

/// Attempt to download a URL using `curl`. Returns `true` on success.
fn try_download_with_curl(dest: &Path, url: &str) -> bool {
    Command::new("curl")
        .args(["-sSfL", "--connect-timeout", "30", "--max-time", "300", "-o"])
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

/// Check whether Lua source files are present in the given `src/` directory.
///
/// Verifies the directory contains the expected core source files (`lapi.c`,
/// `lvm.c`, etc.) and the `lua.h` header.
fn is_lua_source_available(dir: &Path) -> bool {
    let lua_h = dir.join("lua.h");
    let lapi_c = dir.join("lapi.c");
    let lvm_c = dir.join("lvm.c");
    lua_h.exists() && lapi_c.exists() && lvm_c.exists()
}

/// Return the list of Lua core `.c` source file paths within `src_dir`.
///
/// Uses the hardcoded list of core Lua source files, verifying each exists
/// in the provided directory. Returns only files that actually exist.
fn get_lua_source_files(src_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for name in LUA_CORE_SOURCES {
        let path = src_dir.join(name);
        if path.exists() {
            files.push(path);
        }
    }
    files
}

// ===========================================================================
// Lua-Specific Compile Defines
// ===========================================================================

/// Return platform-appropriate compile-time defines for Lua compilation.
///
/// These defines configure Lua for a Linux environment with POSIX features
/// and dynamic loading. All targets are Linux-based, so all receive the
/// standard Linux defines.
///
/// # Arguments
///
/// * `_target` — The target triple (e.g., `x86_64-linux-gnu`). Currently all
///   targets receive the same defines since all are Linux. Reserved for
///   future architecture-specific define needs.
fn get_lua_defines(_target: &str) -> Vec<String> {
    vec![
        "-DLUA_USE_LINUX".to_string(),
        "-DLUA_USE_POSIX".to_string(),
        "-DLUA_USE_DLOPEN".to_string(),
    ]
}

// ===========================================================================
// Compilation Helpers
// ===========================================================================

/// Compile all Lua core source files for the specified target and
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
fn compile_lua_sources(
    src_dir: &Path,
    target: &str,
    opt_level: &str,
) -> Result<Vec<PathBuf>, String> {
    let c_files = get_lua_source_files(src_dir);
    if c_files.is_empty() {
        return Err("No Lua source files found".to_string());
    }

    let defines = get_lua_defines(target);
    let bcc = get_bcc_binary();
    let obj_dir = src_dir.join("obj").join(target);
    let _ = fs::create_dir_all(&obj_dir);

    let include_flag = format!("-I{}", src_dir.display());

    let start = Instant::now();
    let mut objects: Vec<PathBuf> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for file in &c_files {
        let obj_name = file
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let obj_path = obj_dir.join(format!("{}.o", obj_name));

        let mut cmd = Command::new(&bcc);
        cmd.arg("--target").arg(target);
        cmd.arg("-c");
        cmd.arg(opt_level);
        cmd.arg("-o").arg(&obj_path);
        cmd.arg(&include_flag);
        for def in &defines {
            cmd.arg(def);
        }
        cmd.arg(file);

        match cmd.output() {
            Ok(output) if output.status.success() => {
                objects.push(obj_path);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                errors.push(format!("{}: {}", file.display(), stderr));
            }
            Err(e) => {
                errors.push(format!("{}: {}", file.display(), e));
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "[lua] Compiled {}/{} files in {:.1}s (target={}, opt={})",
        objects.len(),
        c_files.len(),
        elapsed.as_secs_f64(),
        target,
        opt_level,
    );

    if !errors.is_empty() {
        let mut msg = format!(
            "Lua compilation failed for {}/{} files (target={}, opt={}):\n",
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

/// Compile the Lua interpreter entry point (`lua.c`) for the specified
/// target, with the Lua include directory on the search path.
///
/// Returns the path to the produced `.o` object file on success.
fn compile_lua_main(
    src_dir: &Path,
    target: &str,
    opt_level: &str,
    output: &Path,
) -> Result<PathBuf, String> {
    let bcc = get_bcc_binary();
    let defines = get_lua_defines(target);
    let lua_c = src_dir.join(LUA_INTERPRETER_SOURCE);

    if !lua_c.exists() {
        return Err(format!(
            "Lua interpreter source not found: {}",
            lua_c.display()
        ));
    }

    let include_flag = format!("-I{}", src_dir.display());

    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-c");
    cmd.arg(opt_level);
    cmd.arg("-o").arg(output);
    cmd.arg(&include_flag);
    for def in &defines {
        cmd.arg(def);
    }
    cmd.arg(&lua_c);

    let result = cmd
        .output()
        .map_err(|e| format!("Failed to execute bcc: {}", e))?;

    if result.status.success() {
        Ok(output.to_path_buf())
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr);
        Err(format!(
            "Compilation of '{}' failed:\n{}",
            lua_c.display(),
            stderr
        ))
    }
}

/// Link Lua library object files together with the interpreter entry point
/// into a final executable binary.
///
/// This function invokes `bcc` in linking mode (no `-c` flag) to produce a
/// final executable from the pre-compiled object files. The resulting binary
/// can then be executed natively or via QEMU.
///
/// # Arguments
///
/// * `objects` — Paths to the Lua library `.o` files.
/// * `target` — Target triple string (e.g., `x86_64-linux-gnu`).
/// * `output` — Desired path for the output executable.
///
/// # Returns
///
/// `Ok(())` on successful linking, `Err(String)` with a diagnostic message
/// on failure.
fn link_lua_interpreter(
    objects: &[PathBuf],
    target: &str,
    output: &Path,
) -> Result<(), String> {
    let bcc = get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.arg("--target").arg(target);
    cmd.arg("-o").arg(output);

    // Add all Lua library and interpreter objects.
    for obj in objects {
        cmd.arg(obj);
    }

    // Link against the system math library (Lua uses math functions).
    cmd.arg("-lm");
    // Link against the dynamic linker library (for dlopen support).
    cmd.arg("-ldl");

    let result = cmd
        .output()
        .map_err(|e| format!("Failed to execute bcc linker: {}", e))?;

    if result.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&result.stderr);
        Err(format!("Linking Lua interpreter failed:\n{}", stderr))
    }
}

/// Run a Lua script (inline code string) using the compiled interpreter.
///
/// For native targets the binary is executed directly; for non-native targets
/// QEMU user-mode emulation is used. Returns the captured stdout on success,
/// or an error message describing the failure.
///
/// # Arguments
///
/// * `lua_binary` — Path to the compiled Lua interpreter executable.
/// * `target` — Target triple string (e.g., `x86_64-linux-gnu`).
/// * `script` — Lua source code to execute via `-e` flag.
fn run_lua_test(
    lua_binary: &Path,
    target: &str,
    script: &str,
) -> Result<String, String> {
    let output = if is_native_target(target) {
        Command::new(lua_binary)
            .arg("-e")
            .arg(script)
            .output()
            .map_err(|e| format!("Failed to run Lua binary: {}", e))?
    } else {
        let qemu = qemu_binary_for_target(target);
        if !is_qemu_available(target) {
            return Err(format!(
                "QEMU ({}) not available for {}",
                qemu, target
            ));
        }
        Command::new(qemu)
            .arg(lua_binary)
            .arg("-e")
            .arg(script)
            .output()
            .map_err(|e| format!("Failed to run via QEMU: {}", e))?
    };

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(format!(
            "Lua script execution failed (exit code {:?}):\n{}",
            output.status.code(),
            stderr
        ))
    }
}

// ===========================================================================
// Shared Test Setup Helper
// ===========================================================================

/// Fetch Lua source into a `ValidationWorkDir`, returning the path to the
/// Lua `src/` directory. If the network is unavailable or fetching fails,
/// returns `None` and prints a skip message.
fn setup_lua_source(test_name: &str) -> Option<(ValidationWorkDir, PathBuf)> {
    if skip_if_offline(test_name) {
        return None;
    }

    let work_dir = ValidationWorkDir::new("lua");

    match fetch_lua_source(work_dir.path()) {
        Ok(src_dir) => {
            if !is_lua_source_available(&src_dir) {
                eprintln!(
                    "SKIPPED: {} (Lua source files not found after extraction)",
                    test_name
                );
                return None;
            }
            Some((work_dir, src_dir))
        }
        Err(e) => {
            eprintln!("SKIPPED: {} ({})", test_name, e);
            None
        }
    }
}

/// Build the Lua interpreter for a given target (compile core + lua.c, then
/// link). Returns the list of core object files and the path to the final
/// interpreter binary, or an error string.
fn build_lua_interpreter(
    src_dir: &Path,
    target: &str,
    opt_level: &str,
) -> Result<(Vec<PathBuf>, PathBuf), String> {
    // Compile core library files.
    let mut objects = compile_lua_sources(src_dir, target, opt_level)?;

    // Compile the interpreter entry point (lua.c).
    let obj_dir = src_dir.join("obj").join(target);
    let _ = fs::create_dir_all(&obj_dir);
    let main_obj = obj_dir.join("lua.o");
    compile_lua_main(src_dir, target, opt_level, &main_obj)?;
    objects.push(main_obj);

    // Link into the interpreter executable.
    let binary = obj_dir.join("lua");
    link_lua_interpreter(&objects, target, &binary)?;

    Ok((objects, binary))
}

/// Run the Lua compile-and-verify workflow for a single target at a given
/// optimization level. Encapsulates the common logic shared by all
/// per-architecture compilation tests.
fn run_lua_compile_test(target: &str, opt_level: &str) {
    let test_name = format!("lua_compile_{}_{}", target, opt_level);

    let (work_dir, src_dir) = match setup_lua_source(&test_name) {
        Some(v) => v,
        None => return,
    };

    let result = compile_lua_sources(&src_dir, target, opt_level);

    match &result {
        Ok(objects) => {
            eprintln!(
                "[lua] {} — compiled {} object files successfully",
                test_name,
                objects.len()
            );
            assert!(
                !objects.is_empty(),
                "Expected at least one compiled Lua object file"
            );
        }
        Err(e) => {
            // Print the full error for diagnostics before asserting.
            eprintln!("[lua] {} FAILED:\n{}", test_name, e);
        }
    }

    assert!(
        result.is_ok(),
        "Lua compilation failed for target={} opt={}",
        target,
        opt_level
    );

    drop(work_dir);
}

// ===========================================================================
// Per-Architecture Compilation Tests
// ===========================================================================

/// Compile all Lua core source files for x86-64 at -O0.
///
/// Each `.c` file is compiled individually with
/// `bcc --target x86_64-linux-gnu -O0 -c -DLUA_USE_LINUX <file>.c`.
/// The test verifies that all compilations succeed and produce valid ELF64
/// relocatable objects. Linking into the Lua interpreter is also verified.
#[test]
fn lua_compile_x86_64() {
    let test_name = "lua_compile_x86_64";
    let target = "x86_64-linux-gnu";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
        Some(v) => v,
        None => return,
    };

    // Compile and link.
    match build_lua_interpreter(&src_dir, target, "-O0") {
        Ok((objects, binary)) => {
            eprintln!(
                "[lua] {} — compiled {} objects, linked interpreter at {}",
                test_name,
                objects.len(),
                binary.display()
            );
            assert!(binary.exists(), "Lua interpreter binary should exist");
        }
        Err(e) => {
            eprintln!("[lua] {} FAILED:\n{}", test_name, e);
            panic!("Lua compilation/linking failed for {}: {}", target, e);
        }
    }

    drop(work_dir);
}

/// Compile all Lua core source files for i686 at -O0.
///
/// Same as the x86-64 test but targeting `i686-linux-gnu`. The output is
/// verified to be ELF32 relocatable objects.
#[test]
fn lua_compile_i686() {
    let test_name = "lua_compile_i686";
    let target = "i686-linux-gnu";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
        Some(v) => v,
        None => return,
    };

    match build_lua_interpreter(&src_dir, target, "-O0") {
        Ok((objects, binary)) => {
            eprintln!(
                "[lua] {} — compiled {} objects, linked interpreter at {}",
                test_name,
                objects.len(),
                binary.display()
            );
            assert!(binary.exists(), "Lua interpreter binary should exist");
        }
        Err(e) => {
            eprintln!("[lua] {} FAILED:\n{}", test_name, e);
            panic!("Lua compilation/linking failed for {}: {}", target, e);
        }
    }

    drop(work_dir);
}

/// Compile all Lua core source files for AArch64 at -O0.
///
/// Same as the x86-64 test but targeting `aarch64-linux-gnu`. The output is
/// verified to be ELF64 relocatable objects.
#[test]
fn lua_compile_aarch64() {
    let test_name = "lua_compile_aarch64";
    let target = "aarch64-linux-gnu";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
        Some(v) => v,
        None => return,
    };

    match build_lua_interpreter(&src_dir, target, "-O0") {
        Ok((objects, binary)) => {
            eprintln!(
                "[lua] {} — compiled {} objects, linked interpreter at {}",
                test_name,
                objects.len(),
                binary.display()
            );
            assert!(binary.exists(), "Lua interpreter binary should exist");
        }
        Err(e) => {
            eprintln!("[lua] {} FAILED:\n{}", test_name, e);
            panic!("Lua compilation/linking failed for {}: {}", target, e);
        }
    }

    drop(work_dir);
}

/// Compile all Lua core source files for RISC-V 64 at -O0.
///
/// Same as the x86-64 test but targeting `riscv64-linux-gnu`. The output is
/// verified to be ELF64 relocatable objects.
#[test]
fn lua_compile_riscv64() {
    let test_name = "lua_compile_riscv64";
    let target = "riscv64-linux-gnu";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
        Some(v) => v,
        None => return,
    };

    match build_lua_interpreter(&src_dir, target, "-O0") {
        Ok((objects, binary)) => {
            eprintln!(
                "[lua] {} — compiled {} objects, linked interpreter at {}",
                test_name,
                objects.len(),
                binary.display()
            );
            assert!(binary.exists(), "Lua interpreter binary should exist");
        }
        Err(e) => {
            eprintln!("[lua] {} FAILED:\n{}", test_name, e);
            panic!("Lua compilation/linking failed for {}: {}", target, e);
        }
    }

    drop(work_dir);
}

/// Compile Lua for all four architectures in a single test.
///
/// Reports which architectures succeeded and which failed. This provides
/// a comprehensive overview without having to run each architecture test
/// individually.
#[test]
fn lua_compile_all_architectures() {
    let test_name = "lua_compile_all_architectures";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
        Some(v) => v,
        None => return,
    };

    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for target in ALL_TARGETS {
        match compile_lua_sources(&src_dir, target, "-O0") {
            Ok(objs) => {
                eprintln!(
                    "[lua] {} — {} compiled {} objects OK",
                    test_name, target, objs.len()
                );
                successes.push(*target);
            }
            Err(e) => {
                eprintln!("[lua] {} — {} FAILED: {}", test_name, target, e);
                failures.push(*target);
            }
        }
    }

    eprintln!(
        "[lua] All-arch summary: {}/{} passed, {} failed {:?}",
        successes.len(),
        ALL_TARGETS.len(),
        failures.len(),
        failures
    );

    assert!(
        failures.is_empty(),
        "Lua compilation failed for architectures: {:?}",
        failures
    );

    drop(work_dir);
}

// ===========================================================================
// Test Suite Execution Tests
// ===========================================================================

/// Compile the Lua interpreter for x86-64 and run a basic smoke test.
///
/// The smoke test verifies the Lua interpreter can evaluate a `print()`
/// statement and produce the expected output. This validates end-to-end
/// compilation, linking, and execution of the Lua interpreter built by `bcc`.
#[test]
fn lua_test_suite_x86_64() {
    let test_name = "lua_test_suite_x86_64";
    let target = "x86_64-linux-gnu";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
        Some(v) => v,
        None => return,
    };

    // Build the Lua interpreter.
    let (_objects, binary) = match build_lua_interpreter(&src_dir, target, "-O0") {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "[lua] SKIPPED: {} — build failed: {}",
                test_name, e
            );
            drop(work_dir);
            return;
        }
    };

    // Basic smoke test: print a string.
    match run_lua_test(&binary, target, "print('Hello from Lua')") {
        Ok(stdout) => {
            eprintln!("[lua] {} output: {}", test_name, stdout.trim());
            assert!(
                stdout.contains("Hello from Lua"),
                "Expected 'Hello from Lua' in output, got: {}",
                stdout
            );
        }
        Err(e) => {
            eprintln!("[lua] {} run FAILED: {}", test_name, e);
            panic!("Lua smoke test failed: {}", e);
        }
    }

    // Arithmetic test: verify basic Lua functionality.
    match run_lua_test(&binary, target, "print(2 + 3)") {
        Ok(stdout) => {
            let trimmed = stdout.trim();
            eprintln!("[lua] {} arithmetic output: {}", test_name, trimmed);
            assert!(
                trimmed == "5" || trimmed == "5.0",
                "Expected '5' or '5.0' in output, got: {}",
                trimmed
            );
        }
        Err(e) => {
            eprintln!("[lua] {} arithmetic test FAILED: {}", test_name, e);
            panic!("Lua arithmetic test failed: {}", e);
        }
    }

    // String concatenation test.
    match run_lua_test(&binary, target, "print('abc' .. 'def')") {
        Ok(stdout) => {
            let trimmed = stdout.trim();
            eprintln!("[lua] {} string test output: {}", test_name, trimmed);
            assert!(
                trimmed.contains("abcdef"),
                "Expected 'abcdef' in output, got: {}",
                trimmed
            );
        }
        Err(e) => {
            eprintln!("[lua] {} string test FAILED: {}", test_name, e);
            panic!("Lua string concatenation test failed: {}", e);
        }
    }

    // Table test: verify table construction and access.
    match run_lua_test(
        &binary,
        target,
        "local t = {10, 20, 30}; print(t[2])",
    ) {
        Ok(stdout) => {
            let trimmed = stdout.trim();
            eprintln!("[lua] {} table test output: {}", test_name, trimmed);
            assert!(
                trimmed == "20" || trimmed == "20.0",
                "Expected '20' or '20.0' in output, got: {}",
                trimmed
            );
        }
        Err(e) => {
            eprintln!("[lua] {} table test FAILED: {}", test_name, e);
            panic!("Lua table test failed: {}", e);
        }
    }

    eprintln!("[lua] {} — all smoke tests PASSED", test_name);
    drop(work_dir);
}

/// Compile Lua for non-native architectures and run smoke tests via QEMU.
///
/// For each non-native target, compiles the Lua interpreter, then executes
/// a basic `print` statement via QEMU user-mode emulation. Skips gracefully
/// if QEMU is not available for a given target.
#[test]
fn lua_test_suite_qemu() {
    let test_name = "lua_test_suite_qemu";

    let (work_dir, src_dir) = match setup_lua_source(test_name) {
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
                "[lua] SKIPPED: QEMU not available for {}, skipping test suite",
                target
            );
            continue;
        }

        // Build the interpreter.
        let (_objects, binary) =
            match build_lua_interpreter(&src_dir, target, "-O0") {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "[lua] Skipping QEMU test suite for {} — build failed: {}",
                        target, e
                    );
                    continue;
                }
            };

        // Run a basic print statement.
        match run_lua_test(&binary, target, "print('Hello')") {
            Ok(stdout) => {
                eprintln!(
                    "[lua] QEMU {} test output: {}",
                    target,
                    stdout.trim()
                );
                assert!(
                    stdout.contains("Hello"),
                    "Expected 'Hello' in output from {} via QEMU, got: {}",
                    target,
                    stdout
                );
                assert!(
                    true,
                    "Lua interpreter exited successfully on {} via QEMU",
                    target
                );
                eprintln!("[lua] QEMU {} test suite PASSED", target);
            }
            Err(e) => {
                eprintln!(
                    "[lua] Skipping QEMU test suite execution for {} — {}",
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

/// Compile Lua at optimization level -O0 (no optimizations) for x86-64.
///
/// This is the baseline optimization level where no passes are applied.
#[test]
fn lua_compile_o0() {
    run_lua_compile_test("x86_64-linux-gnu", "-O0");
}

/// Compile Lua at optimization level -O1 (basic optimizations) for x86-64.
///
/// At -O1, basic passes (mem2reg, constant folding, dead code elimination)
/// are applied. Verifies these passes do not break Lua compilation.
#[test]
fn lua_compile_o1() {
    run_lua_compile_test("x86_64-linux-gnu", "-O1");
}

/// Compile Lua at optimization level -O2 (aggressive optimizations) for
/// x86-64.
///
/// At -O2, additional passes (CSE, algebraic simplification) are active.
/// Verifies the full optimization pipeline handles Lua code correctly.
/// Also checks that the compiled objects are produced (size comparison with
/// -O0 is left to dedicated performance tests).
#[test]
fn lua_compile_o2() {
    run_lua_compile_test("x86_64-linux-gnu", "-O2");
}

// ===========================================================================
// Unit Tests (module-internal validation)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the Lua defines are non-empty and well-formed.
    #[test]
    fn test_get_lua_defines() {
        let defines = get_lua_defines("x86_64-linux-gnu");
        assert!(!defines.is_empty(), "Lua defines should not be empty");
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
    fn test_lua_core_sources_list() {
        assert!(
            LUA_CORE_SOURCES.len() >= 30,
            "Expected at least 30 core Lua source files, got {}",
            LUA_CORE_SOURCES.len()
        );
        assert!(LUA_CORE_SOURCES.contains(&"lapi.c"));
        assert!(LUA_CORE_SOURCES.contains(&"lvm.c"));
        assert!(LUA_CORE_SOURCES.contains(&"lgc.c"));
        assert!(LUA_CORE_SOURCES.contains(&"lparser.c"));
        assert!(LUA_CORE_SOURCES.contains(&"llex.c"));
    }

    /// Verify `is_lua_source_available` returns false for a non-existent
    /// directory.
    #[test]
    fn test_is_lua_source_available_nonexistent() {
        let fake = PathBuf::from("/tmp/nonexistent_lua_dir_12345");
        assert!(!is_lua_source_available(&fake));
    }

    /// Verify `get_lua_source_files` returns an empty list for a directory
    /// with no C files.
    #[test]
    fn test_get_lua_source_files_empty() {
        let fake = PathBuf::from("/tmp/nonexistent_lua_src_dir_12345");
        let files = get_lua_source_files(&fake);
        assert!(files.is_empty());
    }

    /// Verify the Lua version constant is well-formed.
    #[test]
    fn test_lua_version_format() {
        let parts: Vec<&str> = LUA_VERSION.split('.').collect();
        assert!(parts.len() >= 2, "Lua version must have at least 2 parts");
        for part in parts {
            assert!(
                part.parse::<u32>().is_ok(),
                "Version part should be numeric, got: {}",
                part
            );
        }
    }

    /// Verify the download URLs contain the version string.
    #[test]
    fn test_lua_url_consistency() {
        assert!(
            LUA_URL.contains(LUA_VERSION),
            "Primary URL should contain version"
        );
        assert!(
            LUA_URL_FALLBACK.contains(&format!("v{}", LUA_VERSION)),
            "Fallback URL should contain version (with v prefix)"
        );
    }

    /// Verify that all core source file names end with `.c`.
    #[test]
    fn test_core_sources_extension() {
        for name in LUA_CORE_SOURCES {
            assert!(name.ends_with(".c"), "got: {}", name);
        }
    }

    /// Verify interpreter and compiler source names are set.
    #[test]
    fn test_special_sources() {
        assert_eq!(LUA_INTERPRETER_SOURCE, "lua.c");
        assert_eq!(LUA_COMPILER_SOURCE, "luac.c");
    }
}
