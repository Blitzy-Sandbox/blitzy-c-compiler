//! GCC-compatible command-line argument parsing for the `bcc` compiler.
//!
//! This module parses all GCC-compatible CLI flags into a structured [`CliArgs`] struct
//! consumed by the pipeline orchestrator and target resolver. It supports all 17 flags
//! specified in the AAP: `-c`, `-o`, `-I`, `-D`, `-U`, `-L`, `-l`, `-g`, `-O[012]`,
//! `-shared`, `-fPIC`, `-mretpoline`, `-fcf-protection`, `-static`, `--target`.
//!
//! Zero external dependencies — uses only `std::env::args()` for argument collection
//! and `std::path::Path` for output path derivation.

use std::env;
use std::path::Path;

// ---------------------------------------------------------------------------
// OptLevel — Optimization level enum
// ---------------------------------------------------------------------------

/// Optimization level selected via `-O0`, `-O1`, or `-O2` CLI flags.
///
/// Maps directly to the pass pipeline configuration in `src/passes/pipeline.rs`:
/// - `O0`: No optimization passes (default)
/// - `O1`: Basic optimizations (mem2reg, constant folding, dead code elimination)
/// - `O2`: Aggressive optimizations (all of O1 plus CSE, simplification, iteration)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization (default). Corresponds to `-O0`.
    O0,
    /// Basic optimizations. Corresponds to `-O1`.
    O1,
    /// Aggressive optimizations. Corresponds to `-O2`.
    O2,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::O0
    }
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "-O0"),
            OptLevel::O1 => write!(f, "-O1"),
            OptLevel::O2 => write!(f, "-O2"),
        }
    }
}

// ---------------------------------------------------------------------------
// MacroDefinition — Preprocessor macro from -D flag
// ---------------------------------------------------------------------------

/// A preprocessor macro definition from the `-D` CLI flag.
///
/// Represents the three forms of `-D`:
/// - `-DFOO` → `MacroDefinition { name: "FOO", value: None }` (defines FOO as 1)
/// - `-DFOO=bar` → `MacroDefinition { name: "FOO", value: Some("bar") }`
/// - `-DFOO=` → `MacroDefinition { name: "FOO", value: Some("") }` (defines FOO as empty)
#[derive(Debug, Clone)]
pub struct MacroDefinition {
    /// The macro name (e.g., `"FOO"` from `-DFOO` or `-DFOO=bar`).
    pub name: String,
    /// The macro value, if specified. `None` for `-DFOO`, `Some("bar")` for `-DFOO=bar`,
    /// `Some("")` for `-DFOO=`.
    pub value: Option<String>,
}

// ---------------------------------------------------------------------------
// CliArgs — Parsed CLI arguments struct
// ---------------------------------------------------------------------------

/// Parsed command-line arguments for the `bcc` compiler.
///
/// This struct is the single source of truth for all CLI options. It is produced by
/// [`parse_args`] or [`parse_args_from`] and consumed by:
/// - `driver::target` — resolves `--target` into `TargetConfig`
/// - `driver::pipeline` — orchestrates compilation based on all flags
/// - `frontend::preprocessor` — receives `-I`, `-D`, `-U`
/// - `codegen::x86_64::security` — receives `-mretpoline`, `-fcf-protection`
/// - `linker` — receives `-L`, `-l`, `-static`, `-shared`
#[derive(Debug)]
pub struct CliArgs {
    /// Input source file paths (positional arguments). Must contain at least one entry.
    pub input_files: Vec<String>,

    /// Output file path from `-o` flag. `None` means use the default:
    /// `a.out` for executables, derived `.o` name for `-c` mode.
    pub output: Option<String>,

    /// Compile to relocatable object only, skip linking (`-c` flag).
    pub compile_only: bool,

    /// Include search paths (`-I <dir>`), accumulated in command-line order.
    /// Prepended to the bundled header path during preprocessing.
    pub include_paths: Vec<String>,

    /// Preprocessor macro definitions (`-D <name>[=value]`), accumulated in order.
    pub defines: Vec<MacroDefinition>,

    /// Preprocessor macro undefinitions (`-U <name>`), accumulated in order.
    pub undefines: Vec<String>,

    /// Library search paths (`-L <dir>`), accumulated in order.
    pub library_paths: Vec<String>,

    /// Libraries to link (`-l <lib>`), accumulated in order.
    pub libraries: Vec<String>,

    /// Generate DWARF v4 debug information (`-g` flag).
    pub debug_info: bool,

    /// Optimization level (`-O0`, `-O1`, `-O2`). Defaults to `O0`.
    pub opt_level: OptLevel,

    /// Produce shared library output (`-shared` flag).
    pub shared: bool,

    /// Generate position-independent code (`-fPIC` flag).
    pub pic: bool,

    /// Enable retpoline sequences for indirect branches, x86-64 only (`-mretpoline`).
    pub retpoline: bool,

    /// Enable Intel CET `endbr64` instrumentation, x86-64 only (`-fcf-protection`).
    pub cf_protection: bool,

    /// Force static linking (`-static` flag).
    pub static_link: bool,

    /// Target triple string from `--target <triple>`. `None` means use host default.
    pub target: Option<String>,

    /// Omit the 128-byte red zone from x86-64 stack frames (`-mno-red-zone`).
    pub no_red_zone: bool,

    /// Emit each function into its own `.text.<funcname>` section (`-ffunction-sections`).
    pub function_sections: bool,

    /// Emit each global variable into its own `.data.<varname>` section (`-fdata-sections`).
    pub data_sections: bool,

    /// Whether the compilation is freestanding (`-ffreestanding`).
    pub freestanding: bool,

    /// Cross-compilation sysroot directory (`--sysroot <dir>`).
    /// When specified, system header and library search paths are resolved
    /// relative to this directory rather than the host root filesystem.
    /// For example, `--sysroot /usr/aarch64-linux-gnu` redirects system
    /// include resolution to `/usr/aarch64-linux-gnu/usr/include`.
    pub sysroot: Option<String>,

    /// Warnings accumulated from silently-discarded flags.
    pub discarded_flag_warnings: Vec<String>,

    /// Files to force-include before the main source (`-include <file>`).
    /// Each entry is a file path that will be preprocessed as if `#include "file"`
    /// appeared at the top of the source file.
    pub force_includes: Vec<String>,

    /// Preprocessor-only mode: stop after preprocessing and output the result (`-E`).
    pub preprocess_only: bool,

    /// Suppress warnings (`-w`).
    pub suppress_warnings: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        CliArgs {
            input_files: Vec::new(),
            output: None,
            compile_only: false,
            include_paths: Vec::new(),
            defines: Vec::new(),
            undefines: Vec::new(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            debug_info: false,
            opt_level: OptLevel::O0,
            shared: false,
            pic: false,
            retpoline: false,
            cf_protection: false,
            static_link: false,
            target: None,
            no_red_zone: false,
            function_sections: false,
            data_sections: false,
            freestanding: false,
            sysroot: None,
            discarded_flag_warnings: Vec::new(),
            force_includes: Vec::new(),
            preprocess_only: false,
            suppress_warnings: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parses a `-D` macro definition string into a [`MacroDefinition`].
///
/// Handles three forms:
/// - `"FOO"` → name=`"FOO"`, value=`None`
/// - `"FOO=bar"` → name=`"FOO"`, value=`Some("bar")`
/// - `"FOO="` → name=`"FOO"`, value=`Some("")`
fn parse_macro_def(s: &str) -> MacroDefinition {
    if let Some(eq_pos) = s.find('=') {
        MacroDefinition {
            name: s[..eq_pos].to_string(),
            value: Some(s[eq_pos + 1..].to_string()),
        }
    } else {
        MacroDefinition {
            name: s.to_string(),
            value: None,
        }
    }
}

/// Attempts to extract the value portion of a flag that supports both
/// space-separated and attached forms.
///
/// For a flag like `-I`, this handles:
/// - `-I/usr/include` (attached) → returns `Ok(Some("/usr/include"))`, `consumed` = false
/// - `-I /usr/include` (separated) → returns `Ok(Some("/usr/include"))`, `consumed` = true
/// - `-I` at end of args → returns `Err("missing argument to '-I'")`
///
/// `prefix` is the flag string (e.g., `"-I"`), `arg` is the current argument,
/// `args` is the full argument list, and `idx` is the current index (mutable, advanced
/// if the next argument is consumed).
fn extract_flag_value<'a>(
    prefix: &str,
    arg: &'a str,
    args: &'a [String],
    idx: &mut usize,
) -> Result<String, String> {
    if arg.len() > prefix.len() {
        // Attached form: -I/usr/include
        Ok(arg[prefix.len()..].to_string())
    } else {
        // Space-separated form: -I /usr/include
        let next = *idx + 1;
        if next >= args.len() {
            return Err(format!("missing argument to '{}'", prefix));
        }
        *idx = next;
        Ok(args[next].clone())
    }
}

// ---------------------------------------------------------------------------
// Core parse functions
// ---------------------------------------------------------------------------

/// Parses command-line arguments from `std::env::args()`, skipping `argv[0]`.
///
/// This is the primary entry point called from `main()`. It collects all process
/// arguments, strips the program name, and delegates to [`parse_args_from`].
///
/// # Returns
/// - `Ok(CliArgs)` on successful parse with at least one input file
/// - `Err(String)` on any parse error (missing value, unrecognized flag, no input files)
///
/// # Example
/// ```no_run
/// use bcc::driver::cli::parse_args;
/// let args = parse_args().unwrap_or_else(|e| {
///     eprintln!("bcc: error: {}", e);
///     std::process::exit(1);
/// });
/// ```
pub fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    parse_args_from(&args)
}

/// Parses command-line arguments from a provided slice, enabling testability.
///
/// This function implements the full GCC-compatible argument parsing logic for all
/// 17 supported flags. It is the core implementation backing [`parse_args`].
///
/// # Supported Flags
///
/// **Boolean flags** (no value):
/// `-c`, `-g`, `-shared`, `-fPIC`, `-mretpoline`, `-fcf-protection`, `-static`
///
/// **Value flags** (space-separated or attached):
/// `-o`, `-I`, `-D`, `-U`, `-L`, `-l`, `--target`
///
/// **Enum flags** (always attached):
/// `-O0`, `-O1`, `-O2`
///
/// **Positional arguments**: Any argument not starting with `-` is an input file.
///
/// # Errors
/// Returns `Err(String)` if:
/// - A value flag is missing its required argument
/// - An unrecognized flag is encountered
/// - No input files are provided
///
/// # Example
/// ```
/// use bcc::driver::cli::parse_args_from;
/// let args: Vec<String> = vec!["-c".into(), "-O2".into(), "main.c".into()];
/// let cli = parse_args_from(&args).unwrap();
/// assert!(cli.compile_only);
/// assert_eq!(cli.input_files, vec!["main.c".to_string()]);
/// ```
pub fn parse_args_from(args: &[String]) -> Result<CliArgs, String> {
    let mut cli = CliArgs::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        match arg.as_str() {
            // ---------------------------------------------------------------
            // Boolean flags (no value consumed)
            // ---------------------------------------------------------------
            "-c" => {
                cli.compile_only = true;
            }
            "-E" => {
                cli.preprocess_only = true;
            }
            "-w" => {
                cli.suppress_warnings = true;
            }
            "-g" => {
                cli.debug_info = true;
            }
            "-shared" => {
                cli.shared = true;
            }
            "-fPIC" | "-fpic" => {
                cli.pic = true;
            }
            "-mretpoline" => {
                cli.retpoline = true;
            }
            "-fcf-protection" => {
                cli.cf_protection = true;
            }
            "-static" => {
                cli.static_link = true;
            }

            // ---------------------------------------------------------------
            // Optimization level flags (always attached, no separate value)
            // ---------------------------------------------------------------
            "-O0" => {
                cli.opt_level = OptLevel::O0;
            }
            "-O1" => {
                cli.opt_level = OptLevel::O1;
            }
            "-O2" => {
                cli.opt_level = OptLevel::O2;
            }

            // ---------------------------------------------------------------
            // Output file: -o <file> or -o<file>
            // ---------------------------------------------------------------
            "-o" => {
                let val = extract_flag_value("-o", arg, args, &mut i)?;
                cli.output = Some(val);
            }

            // ---------------------------------------------------------------
            // --target: --target=<triple> or --target <triple>
            // ---------------------------------------------------------------
            "--target" => {
                let next = i + 1;
                if next >= args.len() {
                    return Err("missing argument to '--target'".to_string());
                }
                i = next;
                cli.target = Some(args[next].clone());
            }

            // ---------------------------------------------------------------
            // --sysroot: --sysroot=<dir> or --sysroot <dir>
            // Redirects system header and library search paths through
            // the specified root directory.
            // ---------------------------------------------------------------
            "--sysroot" => {
                let next = i + 1;
                if next >= args.len() {
                    return Err("missing argument to '--sysroot'".to_string());
                }
                i = next;
                cli.sysroot = Some(args[next].clone());
            }

            // ---------------------------------------------------------------
            // --help (optional GCC-compatible help)
            // ---------------------------------------------------------------
            "--help" | "-help" => {
                print_usage();
                std::process::exit(0);
            }

            // ---------------------------------------------------------------
            // --version (optional GCC-compatible version)
            // ---------------------------------------------------------------
            "--version" | "-v" => {
                print_version();
                std::process::exit(0);
            }

            // ---------------------------------------------------------------
            // Flags with attached or space-separated values, and other prefixes
            // ---------------------------------------------------------------
            _ => {
                let s = arg.as_str();

                // --target=<triple> (attached form with equals)
                if let Some(triple) = s.strip_prefix("--target=") {
                    cli.target = Some(triple.to_string());
                }
                // --sysroot=<dir> (attached form with equals)
                else if let Some(dir) = s.strip_prefix("--sysroot=") {
                    cli.sysroot = Some(dir.to_string());
                }
                // -o<file> (attached form)
                else if s.starts_with("-o") && s.len() > 2 {
                    cli.output = Some(s[2..].to_string());
                }
                // -include <file> — force-include before source
                else if s == "-include" {
                    i += 1;
                    if i >= args.len() {
                        return Err("missing argument after '-include'".to_string());
                    }
                    cli.force_includes.push(args[i].clone());
                }
                // -I<dir> or -I <dir>
                else if s == "-I" || s.starts_with("-I") {
                    let val = extract_flag_value("-I", s, args, &mut i)?;
                    cli.include_paths.push(val);
                }
                // -D<macro> or -D <macro>
                else if s == "-D" || s.starts_with("-D") {
                    let val = extract_flag_value("-D", s, args, &mut i)?;
                    cli.defines.push(parse_macro_def(&val));
                }
                // -U<macro> or -U <macro>
                else if s == "-U" || s.starts_with("-U") {
                    let val = extract_flag_value("-U", s, args, &mut i)?;
                    cli.undefines.push(val);
                }
                // -L<dir> or -L <dir>
                else if s == "-L" || s.starts_with("-L") {
                    let val = extract_flag_value("-L", s, args, &mut i)?;
                    cli.library_paths.push(val);
                }
                // -l<lib> or -l <lib>
                else if s == "-l" || s.starts_with("-l") {
                    let val = extract_flag_value("-l", s, args, &mut i)?;
                    cli.libraries.push(val);
                }
                // -mno-red-zone: real semantics — omit red zone in x86-64 stack frames
                else if s == "-mno-red-zone" {
                    cli.no_red_zone = true;
                }
                // -ffunction-sections: real semantics — each function in its own section
                else if s == "-ffunction-sections" {
                    cli.function_sections = true;
                }
                // -fdata-sections: real semantics — each variable in its own section
                else if s == "-fdata-sections" {
                    cli.data_sections = true;
                }
                // -ffreestanding: real semantics — note freestanding mode
                else if s == "-ffreestanding" {
                    cli.freestanding = true;
                }
                // -W... warnings flags — silently accept
                else if s.starts_with("-W") {
                    // Warning control flags accepted and discarded
                }
                // -O (other optimization levels) — accept as -O0
                else if s.starts_with("-O") {
                    // Unrecognized -O level, treat as -O0
                }
                // -f... flags — silently accept (kernel passes many -f flags)
                else if s.starts_with("-f") {
                    // Accepted and discarded: -fno-strict-aliasing, -fno-common,
                    // -fno-delete-null-pointer-checks, -fstack-protector, -fstack-protector-strong,
                    // -fno-asynchronous-unwind-tables, -fno-pie, etc.
                }
                // -m... machine flags with possible =value — silently accept
                else if s.starts_with("-m") {
                    // Accepted and discarded: -mpreferred-stack-boundary=N,
                    // -mindirect-branch=thunk, -msoft-float, -mretpoline-external-thunk, etc.
                    // Value is already attached or there is no separate value
                }
                // -std=... standard flags
                else if s.starts_with("-std=") || s.starts_with("--std=") {
                    // Accepted and discarded
                }
                // -pipe
                else if s == "-pipe" {
                    // Accepted and discarded
                }
                // -E (preprocess only) / -S (assembly only) / -M / -MM / -MF / -MQ / -MT / -MD / -MMD
                else if s == "-E"
                    || s == "-S"
                    || s == "-M"
                    || s == "-MM"
                    || s == "-MMD"
                    || s == "-MD"
                {
                    // Accepted and discarded
                } else if s.starts_with("-MF") || s.starts_with("-MQ") || s.starts_with("-MT") {
                    // These take a value, consume it if space-separated
                    if s.len() == 3 {
                        // Space-separated form: -MF <file>
                        let next = i + 1;
                        if next < args.len() {
                            i = next;
                        }
                    }
                    // Attached form is already consumed
                }
                // -x <language>
                else if s == "-x" {
                    // Consume the next argument (language name)
                    let next = i + 1;
                    if next < args.len() {
                        i = next;
                    }
                }
                // -include <file> (force include)
                else if s == "-include" {
                    // Consume the next argument and treat as forced include
                    let next = i + 1;
                    if next < args.len() {
                        i = next;
                    }
                }
                // -isystem <dir>
                else if s == "-isystem"
                    || s == "-idirafter"
                    || s == "-iquote"
                    || s == "-iprefix"
                    || s == "-iwithprefix"
                    || s == "-iwithprefixbefore"
                {
                    // Consume the next argument
                    let next = i + 1;
                    if next < args.len() {
                        i = next;
                        if s == "-isystem" {
                            cli.include_paths.push(args[next].clone());
                        }
                    }
                }
                // -nostdinc / -nostdlib / -nodefaultlibs / -nostartfiles
                else if s == "-nostdinc"
                    || s == "-nostdlib"
                    || s == "-nodefaultlibs"
                    || s == "-nostartfiles"
                    || s == "-nostdinc++"
                    || s == "-nolibc"
                {
                    // Accepted and discarded
                }
                // -Xlinker / -Xassembler / -Xpreprocessor
                else if s == "-Xlinker" || s == "-Xassembler" || s == "-Xpreprocessor" {
                    // Consume next arg
                    let next = i + 1;
                    if next < args.len() {
                        i = next;
                    }
                }
                // -Wl,... / -Wa,... / -Wp,...
                else if s.starts_with("-Wl,") || s.starts_with("-Wa,") || s.starts_with("-Wp,") {
                    // Accepted and discarded
                }
                // --param name=value or --param=name=value
                else if s == "--param" {
                    let next = i + 1;
                    if next < args.len() {
                        i = next;
                    }
                } else if s.starts_with("--param=") {
                    // Already consumed
                }
                // Input files: anything not starting with '-'
                else if !s.starts_with('-') {
                    cli.input_files.push(s.to_string());
                }
                // Catch-all: silently discard any other unrecognized flag with a warning
                else {
                    cli.discarded_flag_warnings
                        .push(format!("warning: unrecognized option '{}' (ignored)", s));
                }
            }
        }

        i += 1;
    }

    // Validate: at least one input file is required
    if cli.input_files.is_empty() {
        return Err("no input files".to_string());
    }

    Ok(cli)
}

// ---------------------------------------------------------------------------
// Output path derivation
// ---------------------------------------------------------------------------

/// Derives the output file path from parsed CLI arguments.
///
/// Resolution order:
/// 1. If `-o <path>` was specified, returns that path verbatim.
/// 2. If `-c` (compile-only) mode is active, derives from the first input file:
///    `foo.c` → `foo.o` using [`std::path::Path::file_stem`].
/// 3. If `-shared` mode is active, returns `a.so`.
/// 4. Otherwise, returns the default `a.out`.
///
/// # Panics
/// This function assumes `cli_args.input_files` is non-empty (guaranteed by
/// [`parse_args_from`] validation).
///
/// # Examples
/// ```
/// use bcc::driver::cli::{CliArgs, derive_output_path};
/// let mut args = CliArgs::default();
/// args.input_files.push("foo.c".to_string());
/// assert_eq!(derive_output_path(&args), "a.out");
///
/// args.compile_only = true;
/// assert_eq!(derive_output_path(&args), "foo.o");
/// ```
pub fn derive_output_path(cli_args: &CliArgs) -> String {
    // Explicit -o flag takes absolute precedence
    if let Some(ref output) = cli_args.output {
        return output.clone();
    }

    // Compile-only mode: derive .o from input filename
    if cli_args.compile_only {
        if let Some(first_input) = cli_args.input_files.first() {
            let stem = Path::new(first_input)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            return format!("{}.o", stem);
        }
        return "output.o".to_string();
    }

    // Shared library mode: default to a.so
    if cli_args.shared {
        return "a.so".to_string();
    }

    // Default executable output
    "a.out".to_string()
}

// ---------------------------------------------------------------------------
// Usage and version display
// ---------------------------------------------------------------------------

/// Prints a GCC-compatible usage/help message to stdout.
///
/// Lists all supported flags with brief descriptions and supported target triples.
fn print_usage() {
    println!("Usage: bcc [options] <input files>");
    println!();
    println!("Options:");
    println!("  -c                    Compile to object file only, do not link");
    println!("  -o <file>             Place output into <file>");
    println!("  -I <dir>              Add directory to include search path");
    println!("  -D <macro>[=value]    Define preprocessor macro");
    println!("  -U <macro>            Undefine preprocessor macro");
    println!("  -L <dir>              Add directory to library search path");
    println!("  -l <lib>              Link with library <lib>");
    println!("  -g                    Generate DWARF v4 debug information");
    println!("  -O0                   No optimization (default)");
    println!("  -O1                   Basic optimizations");
    println!("  -O2                   Aggressive optimizations");
    println!("  -shared               Produce a shared library");
    println!("  -fPIC                 Generate position-independent code");
    println!("  -mretpoline           Enable retpoline for indirect branches (x86-64)");
    println!("  -fcf-protection       Enable Intel CET endbr64 (x86-64)");
    println!("  -static               Force static linking");
    println!("  --target <triple>     Set target architecture triple");
    println!("  --sysroot <dir>       Set cross-compilation sysroot directory");
    println!("  --help                Display this help message");
    println!("  --version             Display compiler version");
    println!();
    println!("Supported targets:");
    println!("  x86_64-linux-gnu      x86-64 (64-bit, ELF64)");
    println!("  i686-linux-gnu        i686 (32-bit, ELF32)");
    println!("  aarch64-linux-gnu     AArch64 (64-bit, ELF64)");
    println!("  riscv64-linux-gnu     RISC-V 64 (64-bit, ELF64)");
}

/// Prints the compiler version string to stdout.
fn print_version() {
    println!("bcc {} (Blitzy C Compiler)", env!("CARGO_PKG_VERSION"));
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a Vec<String> from string slices for test convenience.
    fn args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    // -------------------------------------------------------------------
    // Boolean flag tests
    // -------------------------------------------------------------------

    #[test]
    fn test_compile_only_flag() {
        let cli = parse_args_from(&args(&["-c", "main.c"])).unwrap();
        assert!(cli.compile_only);
    }

    #[test]
    fn test_debug_info_flag() {
        let cli = parse_args_from(&args(&["-g", "main.c"])).unwrap();
        assert!(cli.debug_info);
    }

    #[test]
    fn test_shared_flag() {
        let cli = parse_args_from(&args(&["-shared", "lib.c"])).unwrap();
        assert!(cli.shared);
    }

    #[test]
    fn test_fpic_flag() {
        let cli = parse_args_from(&args(&["-fPIC", "lib.c"])).unwrap();
        assert!(cli.pic);
    }

    #[test]
    fn test_fpic_lowercase() {
        let cli = parse_args_from(&args(&["-fpic", "lib.c"])).unwrap();
        assert!(cli.pic);
    }

    #[test]
    fn test_retpoline_flag() {
        let cli = parse_args_from(&args(&["-mretpoline", "main.c"])).unwrap();
        assert!(cli.retpoline);
    }

    #[test]
    fn test_cf_protection_flag() {
        let cli = parse_args_from(&args(&["-fcf-protection", "main.c"])).unwrap();
        assert!(cli.cf_protection);
    }

    #[test]
    fn test_static_link_flag() {
        let cli = parse_args_from(&args(&["-static", "main.c"])).unwrap();
        assert!(cli.static_link);
    }

    // -------------------------------------------------------------------
    // Optimization level tests
    // -------------------------------------------------------------------

    #[test]
    fn test_opt_level_o0() {
        let cli = parse_args_from(&args(&["-O0", "main.c"])).unwrap();
        assert_eq!(cli.opt_level, OptLevel::O0);
    }

    #[test]
    fn test_opt_level_o1() {
        let cli = parse_args_from(&args(&["-O1", "main.c"])).unwrap();
        assert_eq!(cli.opt_level, OptLevel::O1);
    }

    #[test]
    fn test_opt_level_o2() {
        let cli = parse_args_from(&args(&["-O2", "main.c"])).unwrap();
        assert_eq!(cli.opt_level, OptLevel::O2);
    }

    #[test]
    fn test_default_opt_level_is_o0() {
        let cli = parse_args_from(&args(&["main.c"])).unwrap();
        assert_eq!(cli.opt_level, OptLevel::O0);
    }

    #[test]
    fn test_last_opt_level_wins() {
        let cli = parse_args_from(&args(&["-O2", "-O1", "main.c"])).unwrap();
        assert_eq!(cli.opt_level, OptLevel::O1);
    }

    // -------------------------------------------------------------------
    // Output flag tests
    // -------------------------------------------------------------------

    #[test]
    fn test_output_flag_separated() {
        let cli = parse_args_from(&args(&["-o", "output.elf", "main.c"])).unwrap();
        assert_eq!(cli.output, Some("output.elf".to_string()));
    }

    #[test]
    fn test_output_flag_attached() {
        let cli = parse_args_from(&args(&["-ooutput.elf", "main.c"])).unwrap();
        assert_eq!(cli.output, Some("output.elf".to_string()));
    }

    #[test]
    fn test_output_flag_missing_value() {
        let result = parse_args_from(&args(&["-o"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing argument"));
    }

    // -------------------------------------------------------------------
    // Include path tests
    // -------------------------------------------------------------------

    #[test]
    fn test_include_path_separated() {
        let cli = parse_args_from(&args(&["-I", "/usr/include", "main.c"])).unwrap();
        assert_eq!(cli.include_paths, vec!["/usr/include".to_string()]);
    }

    #[test]
    fn test_include_path_attached() {
        let cli = parse_args_from(&args(&["-I/usr/include", "main.c"])).unwrap();
        assert_eq!(cli.include_paths, vec!["/usr/include".to_string()]);
    }

    #[test]
    fn test_multiple_include_paths() {
        let cli = parse_args_from(&args(&[
            "-I",
            "/usr/include",
            "-I/usr/local/include",
            "main.c",
        ]))
        .unwrap();
        assert_eq!(
            cli.include_paths,
            vec!["/usr/include".to_string(), "/usr/local/include".to_string()]
        );
    }

    // -------------------------------------------------------------------
    // Macro definition tests
    // -------------------------------------------------------------------

    #[test]
    fn test_define_separated() {
        let cli = parse_args_from(&args(&["-D", "FOO", "main.c"])).unwrap();
        assert_eq!(cli.defines.len(), 1);
        assert_eq!(cli.defines[0].name, "FOO");
        assert_eq!(cli.defines[0].value, None);
    }

    #[test]
    fn test_define_attached_no_value() {
        let cli = parse_args_from(&args(&["-DFOO", "main.c"])).unwrap();
        assert_eq!(cli.defines.len(), 1);
        assert_eq!(cli.defines[0].name, "FOO");
        assert_eq!(cli.defines[0].value, None);
    }

    #[test]
    fn test_define_attached_with_value() {
        let cli = parse_args_from(&args(&["-DFOO=bar", "main.c"])).unwrap();
        assert_eq!(cli.defines.len(), 1);
        assert_eq!(cli.defines[0].name, "FOO");
        assert_eq!(cli.defines[0].value, Some("bar".to_string()));
    }

    #[test]
    fn test_define_attached_empty_value() {
        let cli = parse_args_from(&args(&["-DFOO=", "main.c"])).unwrap();
        assert_eq!(cli.defines.len(), 1);
        assert_eq!(cli.defines[0].name, "FOO");
        assert_eq!(cli.defines[0].value, Some("".to_string()));
    }

    #[test]
    fn test_define_separated_with_value() {
        let cli = parse_args_from(&args(&["-D", "FOO=42", "main.c"])).unwrap();
        assert_eq!(cli.defines.len(), 1);
        assert_eq!(cli.defines[0].name, "FOO");
        assert_eq!(cli.defines[0].value, Some("42".to_string()));
    }

    // -------------------------------------------------------------------
    // Undefine tests
    // -------------------------------------------------------------------

    #[test]
    fn test_undefine_separated() {
        let cli = parse_args_from(&args(&["-U", "FOO", "main.c"])).unwrap();
        assert_eq!(cli.undefines, vec!["FOO".to_string()]);
    }

    #[test]
    fn test_undefine_attached() {
        let cli = parse_args_from(&args(&["-UFOO", "main.c"])).unwrap();
        assert_eq!(cli.undefines, vec!["FOO".to_string()]);
    }

    // -------------------------------------------------------------------
    // Library path tests
    // -------------------------------------------------------------------

    #[test]
    fn test_library_path_separated() {
        let cli = parse_args_from(&args(&["-L", "/usr/lib", "main.c"])).unwrap();
        assert_eq!(cli.library_paths, vec!["/usr/lib".to_string()]);
    }

    #[test]
    fn test_library_path_attached() {
        let cli = parse_args_from(&args(&["-L/usr/lib", "main.c"])).unwrap();
        assert_eq!(cli.library_paths, vec!["/usr/lib".to_string()]);
    }

    // -------------------------------------------------------------------
    // Library link tests
    // -------------------------------------------------------------------

    #[test]
    fn test_library_separated() {
        let cli = parse_args_from(&args(&["-l", "pthread", "main.c"])).unwrap();
        assert_eq!(cli.libraries, vec!["pthread".to_string()]);
    }

    #[test]
    fn test_library_attached() {
        let cli = parse_args_from(&args(&["-lpthread", "main.c"])).unwrap();
        assert_eq!(cli.libraries, vec!["pthread".to_string()]);
    }

    // -------------------------------------------------------------------
    // Target flag tests
    // -------------------------------------------------------------------

    #[test]
    fn test_target_separated() {
        let cli = parse_args_from(&args(&["--target", "x86_64-linux-gnu", "main.c"])).unwrap();
        assert_eq!(cli.target, Some("x86_64-linux-gnu".to_string()));
    }

    #[test]
    fn test_target_equals() {
        let cli = parse_args_from(&args(&["--target=aarch64-linux-gnu", "main.c"])).unwrap();
        assert_eq!(cli.target, Some("aarch64-linux-gnu".to_string()));
    }

    #[test]
    fn test_target_default_none() {
        let cli = parse_args_from(&args(&["main.c"])).unwrap();
        assert_eq!(cli.target, None);
    }

    #[test]
    fn test_target_missing_value() {
        let result = parse_args_from(&args(&["main.c", "--target"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing argument"));
    }

    // -------------------------------------------------------------------
    // Input file tests
    // -------------------------------------------------------------------

    #[test]
    fn test_single_input_file() {
        let cli = parse_args_from(&args(&["main.c"])).unwrap();
        assert_eq!(cli.input_files, vec!["main.c".to_string()]);
    }

    #[test]
    fn test_multiple_input_files() {
        let cli = parse_args_from(&args(&["main.c", "util.c", "lib.c"])).unwrap();
        assert_eq!(
            cli.input_files,
            vec![
                "main.c".to_string(),
                "util.c".to_string(),
                "lib.c".to_string()
            ]
        );
    }

    #[test]
    fn test_no_input_files_error() {
        let result = parse_args_from(&args(&["-c", "-O2"]));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "no input files");
    }

    #[test]
    fn test_empty_args_error() {
        let result = parse_args_from(&args(&[]));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "no input files");
    }

    // -------------------------------------------------------------------
    // Error cases
    // -------------------------------------------------------------------

    #[test]
    fn test_unrecognized_flag_warning() {
        // Per Directive 2: unrecognized flags are silently discarded with a warning,
        // not rejected as errors. This enables GCC-compatible flag passthrough for
        // build systems like the Linux kernel Makefile.
        let result = parse_args_from(&args(&["--unknown-flag", "main.c"]));
        assert!(
            result.is_ok(),
            "Unrecognized flags should be accepted with a warning, not rejected"
        );
        let cli_args = result.unwrap();
        assert_eq!(cli_args.input_files.len(), 1);
        assert_eq!(cli_args.input_files[0], "main.c");
        assert!(
            !cli_args.discarded_flag_warnings.is_empty(),
            "Should have recorded a warning for the unrecognized flag"
        );
    }

    #[test]
    fn test_missing_include_value() {
        let result = parse_args_from(&args(&["-I"]));
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_define_value() {
        let result = parse_args_from(&args(&["-D"]));
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_library_path_value() {
        let result = parse_args_from(&args(&["-L"]));
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_library_value() {
        let result = parse_args_from(&args(&["-l"]));
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // Default values tests
    // -------------------------------------------------------------------

    #[test]
    fn test_defaults() {
        let cli = parse_args_from(&args(&["main.c"])).unwrap();
        assert!(!cli.compile_only);
        assert!(!cli.debug_info);
        assert!(!cli.shared);
        assert!(!cli.pic);
        assert!(!cli.retpoline);
        assert!(!cli.cf_protection);
        assert!(!cli.static_link);
        assert_eq!(cli.opt_level, OptLevel::O0);
        assert_eq!(cli.output, None);
        assert_eq!(cli.target, None);
        assert!(cli.include_paths.is_empty());
        assert!(cli.defines.is_empty());
        assert!(cli.undefines.is_empty());
        assert!(cli.library_paths.is_empty());
        assert!(cli.libraries.is_empty());
    }

    // -------------------------------------------------------------------
    // Combined flags test (realistic usage)
    // -------------------------------------------------------------------

    #[test]
    fn test_realistic_compilation() {
        let cli = parse_args_from(&args(&[
            "-c",
            "-g",
            "-O2",
            "-I/usr/include",
            "-I",
            "/usr/local/include",
            "-DNDEBUG",
            "-DVERSION=2",
            "-UFOO",
            "-fPIC",
            "-mretpoline",
            "-fcf-protection",
            "--target=x86_64-linux-gnu",
            "-o",
            "output.o",
            "main.c",
            "util.c",
        ]))
        .unwrap();

        assert!(cli.compile_only);
        assert!(cli.debug_info);
        assert_eq!(cli.opt_level, OptLevel::O2);
        assert_eq!(
            cli.include_paths,
            vec!["/usr/include".to_string(), "/usr/local/include".to_string()]
        );
        assert_eq!(cli.defines.len(), 2);
        assert_eq!(cli.defines[0].name, "NDEBUG");
        assert_eq!(cli.defines[0].value, None);
        assert_eq!(cli.defines[1].name, "VERSION");
        assert_eq!(cli.defines[1].value, Some("2".to_string()));
        assert_eq!(cli.undefines, vec!["FOO".to_string()]);
        assert!(cli.pic);
        assert!(cli.retpoline);
        assert!(cli.cf_protection);
        assert_eq!(cli.target, Some("x86_64-linux-gnu".to_string()));
        assert_eq!(cli.output, Some("output.o".to_string()));
        assert_eq!(
            cli.input_files,
            vec!["main.c".to_string(), "util.c".to_string()]
        );
    }

    #[test]
    fn test_linking_mode() {
        let cli = parse_args_from(&args(&[
            "-shared",
            "-fPIC",
            "-L/usr/lib",
            "-L",
            "/usr/local/lib",
            "-lpthread",
            "-l",
            "m",
            "-o",
            "libfoo.so",
            "foo.c",
        ]))
        .unwrap();

        assert!(cli.shared);
        assert!(cli.pic);
        assert_eq!(
            cli.library_paths,
            vec!["/usr/lib".to_string(), "/usr/local/lib".to_string()]
        );
        assert_eq!(cli.libraries, vec!["pthread".to_string(), "m".to_string()]);
        assert_eq!(cli.output, Some("libfoo.so".to_string()));
    }

    #[test]
    fn test_static_linking() {
        let cli = parse_args_from(&args(&["-static", "-lc", "main.c"])).unwrap();
        assert!(cli.static_link);
        assert_eq!(cli.libraries, vec!["c".to_string()]);
    }

    // -------------------------------------------------------------------
    // Output path derivation tests
    // -------------------------------------------------------------------

    #[test]
    fn test_derive_output_explicit() {
        let mut cli = CliArgs::default();
        cli.input_files.push("main.c".to_string());
        cli.output = Some("my_program".to_string());
        assert_eq!(derive_output_path(&cli), "my_program");
    }

    #[test]
    fn test_derive_output_compile_only() {
        let mut cli = CliArgs::default();
        cli.input_files.push("foo.c".to_string());
        cli.compile_only = true;
        assert_eq!(derive_output_path(&cli), "foo.o");
    }

    #[test]
    fn test_derive_output_compile_only_nested_path() {
        let mut cli = CliArgs::default();
        cli.input_files.push("src/bar.c".to_string());
        cli.compile_only = true;
        assert_eq!(derive_output_path(&cli), "bar.o");
    }

    #[test]
    fn test_derive_output_shared() {
        let mut cli = CliArgs::default();
        cli.input_files.push("lib.c".to_string());
        cli.shared = true;
        assert_eq!(derive_output_path(&cli), "a.so");
    }

    #[test]
    fn test_derive_output_default_executable() {
        let mut cli = CliArgs::default();
        cli.input_files.push("main.c".to_string());
        assert_eq!(derive_output_path(&cli), "a.out");
    }

    #[test]
    fn test_derive_output_explicit_overrides_compile_only() {
        let mut cli = CliArgs::default();
        cli.input_files.push("main.c".to_string());
        cli.compile_only = true;
        cli.output = Some("custom.o".to_string());
        assert_eq!(derive_output_path(&cli), "custom.o");
    }

    // -------------------------------------------------------------------
    // MacroDefinition parsing tests
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_macro_def_name_only() {
        let def = parse_macro_def("FOO");
        assert_eq!(def.name, "FOO");
        assert_eq!(def.value, None);
    }

    #[test]
    fn test_parse_macro_def_with_value() {
        let def = parse_macro_def("FOO=bar");
        assert_eq!(def.name, "FOO");
        assert_eq!(def.value, Some("bar".to_string()));
    }

    #[test]
    fn test_parse_macro_def_empty_value() {
        let def = parse_macro_def("FOO=");
        assert_eq!(def.name, "FOO");
        assert_eq!(def.value, Some("".to_string()));
    }

    #[test]
    fn test_parse_macro_def_value_with_equals() {
        let def = parse_macro_def("FOO=a=b");
        assert_eq!(def.name, "FOO");
        assert_eq!(def.value, Some("a=b".to_string()));
    }

    // -------------------------------------------------------------------
    // OptLevel Display test
    // -------------------------------------------------------------------

    #[test]
    fn test_opt_level_display() {
        assert_eq!(format!("{}", OptLevel::O0), "-O0");
        assert_eq!(format!("{}", OptLevel::O1), "-O1");
        assert_eq!(format!("{}", OptLevel::O2), "-O2");
    }

    #[test]
    fn test_opt_level_default() {
        let level: OptLevel = Default::default();
        assert_eq!(level, OptLevel::O0);
    }
}
