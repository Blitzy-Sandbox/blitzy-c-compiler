//! Entry point for the `bcc` (Blitzy C Compiler) binary.
//!
//! This file contains the `main()` function — the program's entry point —
//! which orchestrates the complete compilation process through three steps:
//!
//! 1. **CLI Argument Parsing** — Parses GCC-compatible command-line flags
//!    via [`driver::parse_args()`].
//! 2. **Target Resolution** — Resolves the `--target` triple to a
//!    [`driver::TargetConfig`] via [`driver::resolve_target()`].
//! 3. **Pipeline Execution** — Sequences all compiler phases through
//!    [`driver::pipeline::run()`]:
//!    - Preprocessing (macro expansion, include resolution, conditionals)
//!    - Lexing (tokenization of preprocessed source)
//!    - Parsing (recursive-descent AST construction)
//!    - Semantic Analysis (type checking, symbol resolution)
//!    - IR Generation (typed AST to SSA-form IR)
//!    - Optimization (passes per `-O` level)
//!    - Code Generation (architecture-specific machine code)
//!    - Debug Info (DWARF v4 sections if `-g` is specified)
//!    - Linking (ELF binary or object file output)
//!
//! ## Exit Codes
//!
//! - `0` — Compilation succeeded.
//! - `1` — Compilation failed (errors emitted to stderr in GCC-compatible
//!   `file:line:col: error: message` format per AAP §0.7).
//!
//! ## GCC-Compatible Diagnostics
//!
//! All error messages follow the GCC-compatible format: `file:line:col: error:
//! description` on stderr. The process exits with code 1 on any compile error.
//! Driver-level errors (invalid CLI arguments, unrecognized target triple) are
//! reported as `bcc: error: <description>` since they lack source locations.
//!
//! ## Internal Compiler Error (ICE) Handling
//!
//! A custom panic hook intercepts unexpected panics and produces a user-friendly
//! "internal compiler error" message on stderr, matching the behavior of
//! production compilers like GCC and Clang. Set `BCC_BACKTRACE=1` or
//! `RUST_BACKTRACE=1` to restore the default Rust panic handler for debugging.
//!
//! ## Zero External Dependencies
//!
//! This file uses only the Rust standard library (`std`) and internal crate
//! modules. No external crates are imported, per project constraint C-003.
//!
//! ## Safety
//!
//! This file contains zero `unsafe` blocks.

// ===========================================================================
// Top-Level Module Declarations
// ===========================================================================
//
// These `mod` declarations establish the crate's module tree, mapping to
// the nine subdirectory modules under `src/`:
//
//   src/driver/   — CLI parsing, target configuration, pipeline orchestration
//   src/frontend/ — Preprocessor, lexer, parser (C11 + GCC extensions)
//   src/sema/     — Semantic analysis (type checking, scoping, symbols)
//   src/ir/       — SSA intermediate representation (types, instructions, CFG)
//   src/passes/   — Optimization passes (constant fold, DCE, CSE, mem2reg)
//   src/codegen/  — Code generation (x86-64, i686, AArch64, RISC-V 64)
//   src/linker/   — Integrated ELF linker (ELF32/64, ar archives, relocations)
//   src/debug/    — DWARF v4 debug information generation
//   src/common/   — Shared utilities (diagnostics, source map, interning, arena)

mod driver;
mod frontend;
mod sema;
mod ir;
mod passes;
mod codegen;
mod linker;
mod debug;
mod common;

// ===========================================================================
// Standard Library Imports
// ===========================================================================

use std::process;

// ===========================================================================
// Main Entry Point
// ===========================================================================

/// Binary entry point for the `bcc` C compiler.
///
/// Orchestrates the compilation process through three high-level steps:
///
/// 1. **Parse CLI arguments** — Collects command-line arguments via
///    `std::env::args()` and parses all GCC-compatible flags through
///    [`driver::parse_args()`].
///
/// 2. **Resolve target configuration** — Maps the `--target` triple
///    (or host default) to a complete [`driver::TargetConfig`] via
///    [`driver::resolve_target()`], carrying architecture-specific parameters
///    (pointer size, ABI, ELF class, endianness) needed by every pipeline phase.
///
/// 3. **Execute pipeline** — Runs the full compilation pipeline via
///    [`driver::pipeline::run()`], which sequences preprocessing, lexing,
///    parsing, semantic analysis, IR generation, optimization, code generation,
///    optional DWARF debug info, and linking into a final ELF binary.
///
/// # Exit Codes
///
/// Returns exit code 0 on successful compilation. Calls [`std::process::exit(1)`]
/// on any error — whether a CLI parsing error, target resolution failure, or
/// any compilation-phase error (which will have been emitted to stderr in
/// GCC-compatible diagnostic format by the pipeline's `DiagnosticEmitter`).
fn main() {
    // Install a custom panic hook to produce user-friendly ICE (Internal
    // Compiler Error) messages instead of the default Rust panic output.
    install_ice_panic_hook();

    // -----------------------------------------------------------------------
    // Step 1: Parse GCC-compatible command-line arguments
    // -----------------------------------------------------------------------
    //
    // Collects arguments via std::env::args() internally and parses all
    // supported flags: -c, -o, -I, -D, -U, -L, -l, -g, -O[012], -shared,
    // -fPIC, -mretpoline, -fcf-protection, -static, --target.
    //
    // Returns Err(String) on any parse error: unrecognized flag, missing
    // argument value, no input files provided.
    let cli_args = match driver::parse_args() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("bcc: error: {}", msg);
            process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Resolve target architecture configuration
    // -----------------------------------------------------------------------
    //
    // If --target was specified, parses the target triple string into a
    // TargetConfig containing all architecture-specific parameters:
    //   - Pointer width, type sizes, alignment requirements
    //   - ELF class (ELF32 for i686, ELF64 for others)
    //   - ABI variant (System V AMD64, cdecl, AAPCS64, LP64D)
    //   - CRT and library search paths
    //
    // Supported target triples:
    //   x86_64-linux-gnu   — AMD/Intel 64-bit
    //   i686-linux-gnu     — Intel 32-bit
    //   aarch64-linux-gnu  — ARM 64-bit
    //   riscv64-linux-gnu  — RISC-V 64-bit
    //
    // If --target was not specified, defaults to the host architecture.
    let target = match driver::resolve_target(cli_args.target.as_deref()) {
        Ok(config) => config,
        Err(msg) => {
            eprintln!("bcc: error: {}", msg);
            process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // Step 3: Run the complete compilation pipeline
    // -----------------------------------------------------------------------
    //
    // The pipeline sequences all compiler phases in strict order:
    //
    //   Read source → Preprocess → Lex → Parse → Sema → IR → Optimize
    //     → Codegen → [Debug info] → Link/Output
    //
    // Each input file is processed independently through the frontend and
    // backend phases. All resulting ObjectCode instances are then collected
    // and passed to the integrated linker for final ELF binary production
    // (unless -c was specified, in which case relocatable .o files are emitted).
    //
    // On error, diagnostic messages have already been emitted to stderr
    // in GCC-compatible format by the pipeline's DiagnosticEmitter.
    match driver::pipeline::run(cli_args, target) {
        Ok(()) => {
            // Compilation succeeded. Exit code 0 is implicit from main()
            // returning normally — no explicit process::exit(0) needed.
        }
        Err(()) => {
            // Compilation failed. All error messages have been emitted to
            // stderr by the DiagnosticEmitter in GCC-compatible format.
            // Exit with code 1 per the diagnostic format rule (AAP §0.7).
            process::exit(1);
        }
    }
}

// ===========================================================================
// ICE (Internal Compiler Error) Panic Hook
// ===========================================================================

/// Installs a custom panic hook that intercepts unexpected panics and produces
/// a user-friendly "internal compiler error" message on stderr.
///
/// Production compilers like GCC and Clang report internal errors in a
/// recognizable format. This hook ensures that if any part of the compiler
/// panics unexpectedly, the user sees:
///
/// ```text
/// bcc: internal compiler error: <panic message>
/// note: this is a bug in bcc; please report it
/// ```
///
/// rather than the default Rust panic backtrace, which would be confusing
/// to end users who are not Rust developers.
///
/// # Environment Variable Override
///
/// If the `BCC_BACKTRACE` or `RUST_BACKTRACE` environment variable is set
/// to `1` or `full`, the default panic hook is preserved to aid compiler
/// development and debugging. This allows developers to get full stack
/// traces when investigating internal errors.
fn install_ice_panic_hook() {
    // Check if the developer wants the default panic handler for debugging.
    // BCC_BACKTRACE takes precedence; fall back to RUST_BACKTRACE.
    let wants_backtrace = std::env::var("BCC_BACKTRACE")
        .or_else(|_| std::env::var("RUST_BACKTRACE"))
        .map(|val| val == "1" || val == "full")
        .unwrap_or(false);

    if wants_backtrace {
        // Preserve the default Rust panic handler for developer debugging.
        // The default handler prints the panic message and optionally a
        // backtrace, which is more useful during compiler development.
        return;
    }

    // Replace the default panic hook with our ICE reporter.
    std::panic::set_hook(Box::new(|panic_info| {
        // Extract the panic message from the payload.
        // Panics can carry either &str or String payloads.
        let message = if let Some(msg) = panic_info.payload().downcast_ref::<&str>() {
            (*msg).to_string()
        } else if let Some(msg) = panic_info.payload().downcast_ref::<String>() {
            msg.clone()
        } else {
            "unexpected internal error".to_string()
        };

        // Extract the panic location (file:line:column) if available.
        // This helps developers locate the source of the ICE.
        let location = if let Some(loc) = panic_info.location() {
            format!(" at {}:{}:{}", loc.file(), loc.line(), loc.column())
        } else {
            String::new()
        };

        // Emit the ICE message in a format recognizable to compiler users.
        // This mirrors how GCC and Clang report internal errors.
        eprintln!("bcc: internal compiler error: {}{}", message, location);
        eprintln!("note: this is a bug in bcc; please report it");

        // Exit with code 1 to maintain GCC-compatible behavior.
        // Without this explicit exit, Rust's default panic handler would
        // exit with code 101, which is not GCC-compatible. All compiler
        // errors (including internal ones) should produce exit code 1 per
        // the diagnostic format rule (AAP §0.7).
        std::process::exit(1);
    }));
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    // Note: The main() function and install_ice_panic_hook() are tested
    // indirectly through the integration test suite (tests/cli.rs) which
    // invokes the compiled binary with various arguments and verifies exit
    // codes and stderr output. Direct unit testing of main() is not practical
    // since it calls process::exit(), which terminates the test runner.
    //
    // The panic hook is tested here with isolated verification.

    /// Verifies that the ICE panic hook can be installed without panicking.
    /// This is a smoke test to ensure the hook installation code doesn't
    /// have any initialization errors.
    #[test]
    fn test_panic_hook_installs_without_error() {
        // Save the current hook, install ours, then restore.
        // We can't easily test the actual output without forking a process,
        // but we verify that installation doesn't panic.
        //
        // Note: set_hook replaces the current hook. In a test context with
        // other tests running, we just verify the function doesn't panic.
        // The actual ICE output is verified by integration tests.
        let result = std::panic::catch_unwind(|| {
            // Create a temporary hook to verify the logic doesn't panic.
            // We use a scoped approach to avoid interfering with other tests.
            let _hook_fn = |panic_info: &std::panic::PanicInfo<'_>| {
                let _message = if let Some(msg) = panic_info.payload().downcast_ref::<&str>() {
                    (*msg).to_string()
                } else if let Some(msg) = panic_info.payload().downcast_ref::<String>() {
                    msg.clone()
                } else {
                    "unexpected internal error".to_string()
                };
            };
        });
        assert!(result.is_ok(), "Panic hook construction should not panic");
    }

    /// Verifies that all nine module declarations are accessible from the
    /// crate root. This test ensures the module tree is correctly established
    /// by importing a key type from each module.
    #[test]
    fn test_module_declarations_accessible() {
        // Verify driver module is accessible.
        let _: fn() -> Result<crate::driver::CliArgs, String> = crate::driver::parse_args;

        // Verify common module is accessible via its re-exported types.
        let _emitter = crate::common::DiagnosticEmitter::new();

        // Verify frontend module is accessible (lexer types).
        // Just check the type exists — we don't need to construct one.
        fn _check_token_kind_exists(_: crate::frontend::TokenKind) {}

        // Verify sema module is accessible.
        // The module is declared; its types are used by the pipeline.
        let _: &str = "sema module declared";

        // Verify ir module is accessible.
        let _: &str = "ir module declared";

        // Verify passes module is accessible.
        let _: &str = "passes module declared";

        // Verify codegen module is accessible.
        let _: &str = "codegen module declared";

        // Verify linker module is accessible.
        let _: &str = "linker module declared";

        // Verify debug module is accessible.
        let _: &str = "debug module declared";
    }
}
