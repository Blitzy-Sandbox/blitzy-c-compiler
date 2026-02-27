// build.rs — Build script for bcc (Blitzy C Compiler)
// Embeds bundled freestanding C header paths and generates target-specific constants
// Uses ONLY std library functions (zero external build dependencies)

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn main() {
    // Phase 1: Locate include directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let include_dir = Path::new(&manifest_dir).join("include");

    // Phase 2: Generate bundled_headers.rs in OUT_DIR
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let generated_path = Path::new(&out_dir).join("bundled_headers.rs");
    let mut file = fs::File::create(&generated_path).expect("Failed to create bundled_headers.rs");

    // Write the include directory path as a constant
    writeln!(
        file,
        "/// Path to the bundled freestanding C headers directory"
    )
    .unwrap();
    writeln!(
        file,
        "pub const BUNDLED_INCLUDE_DIR: &str = \"{}\";",
        include_dir.display()
    )
    .unwrap();
    writeln!(file).unwrap();

    // List the expected bundled headers
    let headers = [
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

    writeln!(file, "/// List of bundled freestanding header file names").unwrap();
    writeln!(
        file,
        "pub const BUNDLED_HEADERS: &[&str] = &["
    )
    .unwrap();
    for header in &headers {
        writeln!(file, "    \"{}\",", header).unwrap();
    }
    writeln!(file, "];").unwrap();

    // Phase 3: Target-specific constants
    let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".to_string());
    writeln!(file).unwrap();
    writeln!(
        file,
        "/// Default target triple (detected at build time)"
    )
    .unwrap();
    writeln!(
        file,
        "pub const DEFAULT_TARGET: &str = \"{}\";",
        target
    )
    .unwrap();

    // Phase 4: Cargo rerun directives
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/");
}
