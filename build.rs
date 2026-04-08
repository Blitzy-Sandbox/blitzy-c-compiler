// build.rs — Build script for bcc (Blitzy C Compiler)
//
// This Cargo build script embeds the bundled freestanding C header files into
// the bcc binary at compile time. It generates a Rust source file
// (`bundled_headers.rs`) in Cargo's OUT_DIR that contains:
//
//   1. The filesystem path to the `include/` directory (BUNDLED_INCLUDE_DIR)
//   2. A list of all expected bundled header file names (BUNDLED_HEADER_NAMES)
//   3. The embedded content of each header file as static string constants
//   4. A lookup function to retrieve header content by filename
//   5. The default target triple detected from the build environment
//   6. The number of successfully embedded headers for runtime validation
//
// This build script uses ONLY the Rust standard library. No external crates
// are used, in compliance with the project's zero-dependency constraint.

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

/// The nine freestanding C standard headers that bcc ships as bundled headers.
/// These are the headers specified by C11 §4/p6 as required for freestanding
/// implementations, enabling compilation without any external header dependencies.
const EXPECTED_HEADERS: &[&str] = &[
    "stddef.h",
    "stdint.h",
    "stdarg.h",
    "stdbool.h",
    "limits.h",
    "float.h",
    "stdalign.h",
    "stdnoreturn.h",
    "iso646.h",
];

fn main() {
    // =========================================================================
    // Phase 1: Header Path Discovery
    // =========================================================================
    //
    // Locate the project root via CARGO_MANIFEST_DIR and construct the path
    // to the `include/` directory where bundled freestanding headers reside.

    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo");
    let include_dir = Path::new(&manifest_dir).join("include");

    // Determine which headers are available on disk. During development, the
    // include/ directory may not yet exist or may be partially populated as
    // other agents create the header files. We handle this gracefully by
    // embedding whatever is available and noting what is missing.
    let discovered_headers = discover_header_files(&include_dir);

    // =========================================================================
    // Phase 2: Compile-Time Constants Generation
    // =========================================================================
    //
    // Generate `bundled_headers.rs` in OUT_DIR. This file is intended to be
    // included into the compiler's source via `include!(concat!(env!("OUT_DIR"),
    // "/bundled_headers.rs"))` from the preprocessor's include resolution module.

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let generated_path = Path::new(&out_dir).join("bundled_headers.rs");

    let mut file =
        fs::File::create(&generated_path).expect("Failed to create bundled_headers.rs in OUT_DIR");

    write_file_header(&mut file);
    write_include_dir_constant(&mut file, &include_dir);
    write_header_names_constant(&mut file);
    write_embedded_headers(&mut file, &include_dir, &discovered_headers);
    write_lookup_function(&mut file);

    // =========================================================================
    // Phase 3: Target-Specific Constants
    // =========================================================================
    //
    // Detect the build target triple and generate constants the compiler uses
    // to determine the default compilation target when `--target` is not given.

    let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".to_string());
    write_target_constants(&mut file, &target);

    // Emit cargo:rustc-cfg directives for conditional compilation in the
    // main codebase based on the host build target's architecture family.
    emit_target_cfg_directives(&target);

    // Emit the include directory path as a compile-time environment variable
    // so source code can access it via env!("BCC_BUNDLED_INCLUDE_DIR").
    println!(
        "cargo:rustc-env=BCC_BUNDLED_INCLUDE_DIR={}",
        include_dir.display()
    );

    // =========================================================================
    // Phase 4: Cargo Rerun Directives
    // =========================================================================
    //
    // Tell Cargo when to re-run this build script. We trigger on:
    //   - Changes to build.rs itself
    //   - Changes to the include/ directory (new/modified/deleted headers)
    //   - Changes to each individual header file for fine-grained tracking

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/");

    // Register each individual header file for fine-grained change detection.
    // This ensures that modifying a single header re-triggers embedding without
    // relying solely on directory-level change detection.
    for header_name in EXPECTED_HEADERS {
        let header_path = include_dir.join(header_name);
        println!("cargo:rerun-if-changed={}", header_path.display());
    }

    // Also register any additional .h files discovered in include/ that are
    // not in the expected list (for forward-compatibility).
    for header_name in &discovered_headers {
        if !EXPECTED_HEADERS.contains(&header_name.as_str()) {
            let header_path = include_dir.join(header_name);
            println!("cargo:rerun-if-changed={}", header_path.display());
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Discovers all `.h` files present in the include directory.
///
/// Returns a sorted list of header file names (without path prefix).
/// Returns an empty list if the directory does not exist or cannot be read,
/// which is expected during early development when headers are not yet created.
fn discover_header_files(include_dir: &Path) -> Vec<String> {
    if !include_dir.exists() {
        return Vec::new();
    }

    let entries = match fs::read_dir(include_dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };

    let mut headers: Vec<String> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name()?.to_str()?.to_string();
                if name.ends_with(".h") {
                    return Some(name);
                }
            }
            None
        })
        .collect();

    // Sort for deterministic output across builds and platforms.
    headers.sort();
    headers
}

/// Writes the auto-generated file header comment to the output file.
fn write_file_header(file: &mut fs::File) {
    writeln!(file, "// Auto-generated by build.rs — DO NOT EDIT MANUALLY").expect("write failed");
    writeln!(
        file,
        "// This file contains embedded bundled freestanding C headers and"
    )
    .expect("write failed");
    writeln!(
        file,
        "// compile-time constants for the bcc (Blitzy C Compiler) project."
    )
    .expect("write failed");
    writeln!(file).expect("write failed");
}

/// Writes the BUNDLED_INCLUDE_DIR constant pointing to the include/ directory.
fn write_include_dir_constant(file: &mut fs::File, include_dir: &Path) {
    writeln!(
        file,
        "/// Filesystem path to the bundled freestanding C headers directory."
    )
    .expect("write failed");
    writeln!(
        file,
        "/// This is the absolute path to the `include/` directory at the project root,"
    )
    .expect("write failed");
    writeln!(
        file,
        "/// used as a fallback when embedded header content is not available."
    )
    .expect("write failed");
    writeln!(
        file,
        "pub const BUNDLED_INCLUDE_DIR: &str = {:?};",
        include_dir.display().to_string()
    )
    .expect("write failed");
    writeln!(file).expect("write failed");
}

/// Writes the BUNDLED_HEADER_NAMES constant listing all expected header names.
fn write_header_names_constant(file: &mut fs::File) {
    writeln!(
        file,
        "/// Names of all bundled freestanding C standard headers."
    )
    .expect("write failed");
    writeln!(
        file,
        "/// These nine headers enable freestanding C11 compilation without"
    )
    .expect("write failed");
    writeln!(file, "/// external header file dependencies.").expect("write failed");
    writeln!(file, "pub const BUNDLED_HEADER_NAMES: &[&str] = &[").expect("write failed");
    for header in EXPECTED_HEADERS {
        writeln!(file, "    {:?},", header).expect("write failed");
    }
    writeln!(file, "];").expect("write failed");
    writeln!(file).expect("write failed");
}

/// Reads each header file and writes its content as a static string constant.
///
/// For each header, generates a constant like:
///   pub const HEADER_STDDEF_H: &str = "...content...";
///
/// If a header file does not exist on disk, an empty string is emitted and
/// a cargo:warning is printed to alert the developer.
fn write_embedded_headers(file: &mut fs::File, include_dir: &Path, discovered_headers: &[String]) {
    let mut embedded_count: usize = 0;

    // Generate a constant for each expected header.
    for header_name in EXPECTED_HEADERS {
        let header_path = include_dir.join(header_name);
        let constant_name = header_constant_name(header_name);

        writeln!(file).expect("write failed");
        writeln!(
            file,
            "/// Embedded content of `{}`. Empty string if the header was not",
            header_name
        )
        .expect("write failed");
        writeln!(file, "/// available at build time.").expect("write failed");

        if header_path.exists() && header_path.is_file() {
            match fs::read_to_string(&header_path) {
                Ok(content) => {
                    // Use Rust raw string literal to embed header content verbatim.
                    // We use the r#"..."# form to handle any special characters in
                    // the C header content. The triple-hash delimiter (r###) provides
                    // safety against content containing ## (common in C preprocessor
                    // token pasting operators).
                    writeln!(
                        file,
                        "pub const {}: &str = r###\"{}\"###;",
                        constant_name, content
                    )
                    .expect("write failed");
                    embedded_count += 1;
                }
                Err(e) => {
                    println!("cargo:warning=Failed to read header {}: {}", header_name, e);
                    writeln!(file, "pub const {}: &str = \"\";", constant_name)
                        .expect("write failed");
                }
            }
        } else {
            // Header not yet created — this is normal during development.
            // Emit an empty constant so downstream code compiles regardless.
            println!(
                "cargo:warning=Bundled header {} not found at {}; embedding empty placeholder",
                header_name,
                header_path.display()
            );
            writeln!(file, "pub const {}: &str = \"\";", constant_name).expect("write failed");
        }
    }

    writeln!(file).expect("write failed");
    writeln!(
        file,
        "/// Number of headers successfully embedded at build time."
    )
    .expect("write failed");
    writeln!(
        file,
        "pub const EMBEDDED_HEADER_COUNT: usize = {};",
        embedded_count
    )
    .expect("write failed");
    writeln!(file).expect("write failed");

    // Report the discovery/embedding summary for build-time diagnostics.
    let discovered_count = discovered_headers.len();
    let expected_count = EXPECTED_HEADERS.len();
    if discovered_count < expected_count {
        println!(
            "cargo:warning=Only {}/{} bundled headers found in include/. \
             Missing headers will be embedded as empty strings.",
            discovered_count, expected_count
        );
    }
}

/// Writes a lookup function that retrieves embedded header content by filename.
///
/// This function is the primary interface for the preprocessor's include
/// resolution system: given a header name like "stddef.h", it returns the
/// embedded content (or None if the header is not a bundled freestanding header).
fn write_lookup_function(file: &mut fs::File) {
    writeln!(
        file,
        "/// Looks up the embedded content of a bundled freestanding header by name."
    )
    .expect("write failed");
    writeln!(file, "///").expect("write failed");
    writeln!(
        file,
        "/// Returns `Some(content)` if the given filename matches a bundled header,"
    )
    .expect("write failed");
    writeln!(
        file,
        "/// or `None` if the filename is not a recognized bundled header."
    )
    .expect("write failed");
    writeln!(file, "///").expect("write failed");
    writeln!(file, "/// # Arguments").expect("write failed");
    writeln!(
        file,
        "/// * `name` - The header filename (e.g., \"stddef.h\")"
    )
    .expect("write failed");
    writeln!(file, "///").expect("write failed");
    writeln!(file, "/// # Returns").expect("write failed");
    writeln!(
        file,
        "/// * `Some(&str)` containing the header content if found and non-empty"
    )
    .expect("write failed");
    writeln!(
        file,
        "/// * `None` if the header is not bundled or content was empty at build time"
    )
    .expect("write failed");
    writeln!(
        file,
        "pub fn lookup_bundled_header(name: &str) -> Option<&'static str> {{"
    )
    .expect("write failed");
    writeln!(file, "    let content = match name {{").expect("write failed");

    for header_name in EXPECTED_HEADERS {
        let constant_name = header_constant_name(header_name);
        writeln!(file, "        {:?} => {},", header_name, constant_name).expect("write failed");
    }

    writeln!(file, "        _ => return None,").expect("write failed");
    writeln!(file, "    }};").expect("write failed");
    writeln!(file, "    if content.is_empty() {{").expect("write failed");
    writeln!(file, "        None").expect("write failed");
    writeln!(file, "    }} else {{").expect("write failed");
    writeln!(file, "        Some(content)").expect("write failed");
    writeln!(file, "    }}").expect("write failed");
    writeln!(file, "}}").expect("write failed");
    writeln!(file).expect("write failed");

    // Also generate is_bundled_header for quick checks without content retrieval.
    writeln!(
        file,
        "/// Returns `true` if the given filename is a recognized bundled freestanding header."
    )
    .expect("write failed");
    writeln!(file, "pub fn is_bundled_header(name: &str) -> bool {{").expect("write failed");
    writeln!(file, "    BUNDLED_HEADER_NAMES.contains(&name)").expect("write failed");
    writeln!(file, "}}").expect("write failed");
    writeln!(file).expect("write failed");
}

/// Writes the default target triple and architecture-related constants.
fn write_target_constants(file: &mut fs::File, target: &str) {
    writeln!(
        file,
        "/// Default target triple detected from the Cargo build environment."
    )
    .expect("write failed");
    writeln!(
        file,
        "/// When the user does not specify `--target`, bcc compiles for this target."
    )
    .expect("write failed");
    writeln!(file, "pub const DEFAULT_TARGET: &str = {:?};", target).expect("write failed");
    writeln!(file).expect("write failed");

    // Extract the architecture from the target triple for convenience.
    let arch = target.split('-').next().unwrap_or("unknown");
    writeln!(
        file,
        "/// Architecture component of the default target triple."
    )
    .expect("write failed");
    writeln!(file, "pub const DEFAULT_ARCH: &str = {:?};", arch).expect("write failed");
    writeln!(file).expect("write failed");

    // Determine pointer width and ELF class for the default target.
    let (pointer_width, elf_class) = match arch {
        "i686" | "i386" | "i586" => (4u8, "ELF32"),
        _ => (8u8, "ELF64"), // x86_64, aarch64, riscv64 are all 64-bit
    };

    writeln!(
        file,
        "/// Default pointer width in bytes for the build target architecture."
    )
    .expect("write failed");
    writeln!(
        file,
        "pub const DEFAULT_POINTER_WIDTH: u8 = {};",
        pointer_width
    )
    .expect("write failed");
    writeln!(file).expect("write failed");

    writeln!(
        file,
        "/// Default ELF class for the build target architecture."
    )
    .expect("write failed");
    writeln!(file, "pub const DEFAULT_ELF_CLASS: &str = {:?};", elf_class).expect("write failed");
}

/// Emits `cargo:rustc-cfg` directives based on the build target architecture.
///
/// These cfg attributes enable conditional compilation in the main codebase.
/// For example, `#[cfg(bcc_host_64bit)]` can be used to compile host-specific
/// code paths.
fn emit_target_cfg_directives(target: &str) {
    let arch = target.split('-').next().unwrap_or("unknown");

    // Emit architecture family cfg for host-specific code paths.
    match arch {
        "x86_64" => {
            println!("cargo:rustc-cfg=bcc_host_x86_64");
            println!("cargo:rustc-cfg=bcc_host_64bit");
        }
        "i686" | "i386" | "i586" => {
            println!("cargo:rustc-cfg=bcc_host_x86_32");
            println!("cargo:rustc-cfg=bcc_host_32bit");
        }
        "aarch64" => {
            println!("cargo:rustc-cfg=bcc_host_aarch64");
            println!("cargo:rustc-cfg=bcc_host_64bit");
        }
        "riscv64" | "riscv64gc" => {
            println!("cargo:rustc-cfg=bcc_host_riscv64");
            println!("cargo:rustc-cfg=bcc_host_64bit");
        }
        _ => {
            // Unknown architecture — emit a generic cfg.
            println!("cargo:rustc-cfg=bcc_host_unknown");
        }
    }
}

/// Converts a header filename like "stddef.h" into a Rust constant name
/// like "HEADER_STDDEF_H".
///
/// The conversion uppercases all characters and replaces dots with underscores,
/// then prepends "HEADER_".
fn header_constant_name(header_name: &str) -> String {
    let sanitized: String = header_name
        .chars()
        .map(|c| {
            if c == '.' {
                '_'
            } else {
                c.to_ascii_uppercase()
            }
        })
        .collect();
    format!("HEADER_{}", sanitized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_constant_name_stddef() {
        assert_eq!(header_constant_name("stddef.h"), "HEADER_STDDEF_H");
    }

    #[test]
    fn test_header_constant_name_stdint() {
        assert_eq!(header_constant_name("stdint.h"), "HEADER_STDINT_H");
    }

    #[test]
    fn test_header_constant_name_iso646() {
        assert_eq!(header_constant_name("iso646.h"), "HEADER_ISO646_H");
    }

    #[test]
    fn test_header_constant_name_stdbool() {
        assert_eq!(header_constant_name("stdbool.h"), "HEADER_STDBOOL_H");
    }

    #[test]
    fn test_header_constant_name_stdnoreturn() {
        assert_eq!(
            header_constant_name("stdnoreturn.h"),
            "HEADER_STDNORETURN_H"
        );
    }

    #[test]
    fn test_discover_nonexistent_dir() {
        let result = discover_header_files(Path::new("/nonexistent/path"));
        assert!(result.is_empty());
    }

    #[test]
    fn test_expected_headers_count() {
        assert_eq!(EXPECTED_HEADERS.len(), 9);
    }

    #[test]
    fn test_expected_headers_all_dot_h() {
        for header in EXPECTED_HEADERS {
            assert!(
                header.ends_with(".h"),
                "Header '{}' does not end with .h",
                header
            );
        }
    }

    #[test]
    fn test_expected_headers_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for header in EXPECTED_HEADERS {
            assert!(seen.insert(header), "Duplicate header name: {}", header);
        }
    }
}
