//! Entry point for the `bcc` C compiler.
//!
//! This binary orchestrates the complete compilation pipeline:
//!
//! 1. **CLI Parsing** — Parse GCC-compatible command-line arguments.
//! 2. **Target Resolution** — Determine the target architecture and ABI.
//! 3. **Preprocessing** — Expand macros, resolve includes, evaluate conditionals.
//! 4. **Lexing** — Tokenize preprocessed source into a token stream.
//! 5. **Parsing** — Build an abstract syntax tree (AST) from the token stream.
//! 6. **Semantic Analysis** — Type-check and resolve symbols in the AST.
//! 7. **IR Generation** — Lower the typed AST to SSA intermediate representation.
//! 8. **Optimization** — Apply optimization passes per the requested `-O` level.
//! 9. **Code Generation** — Select instructions and encode machine code.
//! 10. **Linking** — Resolve symbols, apply relocations, and emit ELF output.
//!
//! ## Exit Codes
//!
//! - `0` — Compilation succeeded.
//! - `1` — Compilation failed (errors reported on stderr in GCC format).
//!
//! ## Zero External Dependencies
//!
//! Uses only the Rust standard library. No external crates.

// Module declarations for all compiler phases.
mod codegen;
mod common;
mod debug;
mod driver;
mod frontend;
mod ir;
mod linker;
mod passes;
mod sema;

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use crate::codegen::generate_code;
use crate::common::{DiagnosticEmitter, Interner, SourceMap};
use crate::debug::{address_size_for_architecture, CompilationUnitDebugInfo, DebugInfoGenerator};
use crate::driver::{derive_output_path, parse_args, resolve_target, CliArgs, TargetConfig};
use crate::frontend::preprocessor::{Preprocessor, PreprocessorOptions};
use crate::frontend::{Lexer, Parser};
use crate::ir::IrBuilder;
use crate::linker::{DebugSections, LinkerConfig, LinkerInput, OutputMode};
use crate::sema::SemanticAnalyzer;

fn main() {
    let exit_code = run();
    process::exit(exit_code);
}

/// Main compilation driver. Returns 0 on success, 1 on error.
fn run() -> i32 {
    // -----------------------------------------------------------------------
    // Phase 1: CLI Argument Parsing
    // -----------------------------------------------------------------------
    let cli = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("bcc: error: {}", msg);
            return 1;
        }
    };

    // -----------------------------------------------------------------------
    // Phase 2: Target Resolution
    // -----------------------------------------------------------------------
    let target = match resolve_target(cli.target.as_deref()) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("bcc: error: {}", msg);
            return 1;
        }
    };

    // -----------------------------------------------------------------------
    // Phase 3: Determine output path
    // -----------------------------------------------------------------------
    let output_path = derive_output_path(&cli);

    // -----------------------------------------------------------------------
    // Phase 4: Compile each input file
    // -----------------------------------------------------------------------
    // For single-file compilation (the common case), we compile and optionally
    // link in a single pass. For multiple input files, each is compiled to an
    // object and then linked together.

    let mut all_objects = Vec::new();
    let mut all_debug_sections: Option<DebugSections> = None;
    let mut had_errors = false;

    for input_path_str in &cli.input_files {
        let input_path = Path::new(input_path_str);

        // Read source file
        let source = match fs::read_to_string(input_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bcc: error: {}: {}", input_path.display(), e);
                had_errors = true;
                continue;
            }
        };

        // Compile this translation unit
        match compile_translation_unit(
            &source,
            input_path,
            &cli,
            &target,
        ) {
            Ok(compiled) => {
                all_objects.push(compiled.object_code);
                if let Some(dbg) = compiled.debug_sections {
                    all_debug_sections = Some(dbg);
                }
            }
            Err(_) => {
                // Errors already reported to stderr by the diagnostic emitter.
                had_errors = true;
            }
        }
    }

    if had_errors {
        return 1;
    }

    // -----------------------------------------------------------------------
    // Phase 5: Linking (unless -c was specified)
    // -----------------------------------------------------------------------
    if cli.compile_only {
        // In compile-only mode (-c), write the relocatable object directly.
        // For simplicity, we produce a minimal ELF relocatable object from
        // the first (and typically only) compiled object.
        if all_objects.is_empty() {
            eprintln!("bcc: error: no input files compiled successfully");
            return 1;
        }

        // Use the linker in relocatable mode to produce a proper .o file.
        let linker_config = LinkerConfig {
            output_mode: OutputMode::Relocatable,
            output_path: PathBuf::from(&output_path),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            force_static: false,
            target: target.clone(),
            entry_point: String::new(),
        };

        let linker_input = LinkerInput {
            objects: all_objects,
            debug_sections: all_debug_sections,
        };

        match linker::link(linker_input, &linker_config) {
            Ok(bytes) => {
                if let Err(e) = fs::write(&output_path, &bytes) {
                    eprintln!("bcc: error: cannot write output '{}': {}", output_path, e);
                    return 1;
                }
            }
            Err(e) => {
                eprintln!("bcc: error: link failed: {}", e);
                return 1;
            }
        }
    } else {
        // Full linking mode: produce an executable or shared library.
        if all_objects.is_empty() {
            eprintln!("bcc: error: no input files compiled successfully");
            return 1;
        }

        let output_mode = if cli.shared {
            OutputMode::SharedLibrary
        } else {
            OutputMode::StaticExecutable
        };

        // Build the library list. Implicitly add libc (-lc) if the user
        // did not already specify it, matching GCC's default behavior.
        let mut libraries = cli.libraries.clone();
        if !libraries.iter().any(|l| l == "c") {
            libraries.push("c".to_string());
        }

        let linker_config = LinkerConfig {
            output_mode,
            output_path: PathBuf::from(&output_path),
            library_paths: cli.library_paths.iter().map(PathBuf::from).collect(),
            libraries,
            force_static: cli.static_link,
            target: target.clone(),
            entry_point: String::from("_start"),
        };

        let linker_input = LinkerInput {
            objects: all_objects,
            debug_sections: all_debug_sections,
        };

        match linker::link(linker_input, &linker_config) {
            Ok(bytes) => {
                if let Err(e) = fs::write(&output_path, &bytes) {
                    eprintln!("bcc: error: cannot write output '{}': {}", output_path, e);
                    return 1;
                }
                // Make the output executable on Unix.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Ok(metadata) = fs::metadata(&output_path) {
                        let mut perms = metadata.permissions();
                        let mode = perms.mode() | 0o111; // Add execute bits
                        perms.set_mode(mode);
                        let _ = fs::set_permissions(&output_path, perms);
                    }
                }
            }
            Err(e) => {
                eprintln!("bcc: error: link failed: {}", e);
                return 1;
            }
        }
    }

    0
}

// ===========================================================================
// Translation Unit Compilation
// ===========================================================================

/// Result of compiling a single translation unit.
struct CompiledUnit {
    /// Machine code, symbols, and relocations ready for linking.
    object_code: codegen::ObjectCode,
    /// Optional DWARF debug sections (present when `-g` is specified).
    debug_sections: Option<DebugSections>,
}

/// Compiles a single C source file through all frontend and backend phases.
///
/// Pipeline: preprocess → lex → parse → sema → IR → optimize → codegen
/// (→ optional debug info generation).
///
/// Returns `Ok(CompiledUnit)` on success, or `Err(())` with errors already
/// reported to stderr via the diagnostic emitter.
fn compile_translation_unit(
    source: &str,
    file_path: &Path,
    cli: &CliArgs,
    target: &TargetConfig,
) -> Result<CompiledUnit, ()> {
    let mut source_map = SourceMap::new();
    let mut diagnostics = DiagnosticEmitter::new();
    let mut interner = Interner::new();

    // Register the main source file with the source map.
    let file_id = source_map.add_file(file_path.to_path_buf(), source.to_string());

    // Sync diagnostics with source map for error location resolution.
    diagnostics.register_file(file_id, &file_path.display().to_string());

    // -------------------------------------------------------------------
    // Preprocessing
    // -------------------------------------------------------------------
    let preprocess_options = build_preprocess_options(cli, file_path);
    let mut preprocessor = Preprocessor::new(preprocess_options, &mut source_map, &mut diagnostics);

    let preprocessed = preprocessor.process(source, file_path, &mut source_map, &mut diagnostics);

    let preprocessed = match preprocessed {
        Ok(text) => text,
        Err(_) => {
            // Sync and flush diagnostics before returning.
            diagnostics.sync_source_map(&source_map);
            return Err(());
        }
    };

    if diagnostics.has_errors() {
        diagnostics.sync_source_map(&source_map);
        return Err(());
    }

    // -------------------------------------------------------------------
    // Lexing
    // -------------------------------------------------------------------
    let mut lexer = Lexer::new(&preprocessed, file_id, &mut interner, &mut diagnostics);
    let tokens = lexer.tokenize();

    if diagnostics.has_errors() {
        diagnostics.sync_source_map(&source_map);
        return Err(());
    }

    // -------------------------------------------------------------------
    // Parsing
    // -------------------------------------------------------------------
    let ast = match Parser::parse(&tokens, &interner, &mut diagnostics) {
        Ok(tu) => tu,
        Err(_) => {
            diagnostics.sync_source_map(&source_map);
            return Err(());
        }
    };

    if diagnostics.has_errors() {
        diagnostics.sync_source_map(&source_map);
        return Err(());
    }

    // -------------------------------------------------------------------
    // Semantic Analysis
    // -------------------------------------------------------------------
    let typed_ast = match SemanticAnalyzer::analyze(&ast, target, &interner, &mut diagnostics) {
        Ok(typed) => typed,
        Err(_) => {
            diagnostics.sync_source_map(&source_map);
            return Err(());
        }
    };

    if diagnostics.has_errors() {
        diagnostics.sync_source_map(&source_map);
        return Err(());
    }

    // -------------------------------------------------------------------
    // IR Generation
    // -------------------------------------------------------------------
    let module_name = file_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "module".to_string());

    let mut ir_builder = IrBuilder::new(target, &mut diagnostics, &module_name);
    let mut ir_module = ir_builder.build(&typed_ast);

    if diagnostics.has_errors() {
        diagnostics.sync_source_map(&source_map);
        return Err(());
    }

    // -------------------------------------------------------------------
    // Optimization
    // -------------------------------------------------------------------
    let pass_opt_level = map_opt_level(&cli.opt_level);
    let pipeline = passes::Pipeline::new(pass_opt_level);
    let _stats = pipeline.run_on_module(&mut ir_module);

    // -------------------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------------------
    let object_code = match generate_code(&ir_module, target) {
        Ok(obj) => obj,
        Err(e) => {
            eprintln!(
                "bcc: error: code generation failed for '{}': {}",
                file_path.display(),
                e
            );
            return Err(());
        }
    };

    // -------------------------------------------------------------------
    // Debug Information (optional, when -g is specified)
    // -------------------------------------------------------------------
    let debug_sections = if cli.debug_info {
        let addr_size = address_size_for_architecture(&target.arch);
        let generator = DebugInfoGenerator::new(addr_size, target.arch.clone());

        let comp_dir = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());

        let cu_info = CompilationUnitDebugInfo {
            producer: "bcc 0.1.0".to_string(),
            language: 0x1d, // DW_LANG_C11
            source_file: file_path.display().to_string(),
            comp_dir,
            low_pc: 0,
            high_pc: 0,
            functions: Vec::new(),
            global_variables: Vec::new(),
            source_files: vec![file_path.display().to_string()],
            include_directories: Vec::new(),
        };

        let dwarf = generator.generate(&cu_info);

        Some(DebugSections {
            debug_info: dwarf.debug_info,
            debug_abbrev: dwarf.debug_abbrev,
            debug_line: dwarf.debug_line,
            debug_str: dwarf.debug_str,
            debug_aranges: dwarf.debug_aranges,
            debug_frame: dwarf.debug_frame,
            debug_loc: dwarf.debug_loc,
            relocations: Vec::new(),
        })
    } else {
        None
    };

    Ok(CompiledUnit {
        object_code,
        debug_sections,
    })
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Builds `PreprocessorOptions` from CLI arguments.
fn build_preprocess_options(cli: &CliArgs, _file_path: &Path) -> PreprocessorOptions {
    let defines: Vec<(String, Option<String>)> = cli
        .defines
        .iter()
        .map(|d| (d.name.clone(), d.value.clone()))
        .collect();

    // Determine the bundled header path. Look for an `include/` directory
    // relative to the compiler binary, or use the compile-time embedded path.
    let bundled_header_path = find_bundled_headers();

    PreprocessorOptions {
        include_dirs: cli.include_paths.iter().map(PathBuf::from).collect(),
        defines,
        undefines: cli.undefines.clone(),
        bundled_header_path,
        system_include_dirs: Vec::new(),
    }
}

/// Locates the bundled freestanding header directory.
///
/// Searches for an `include/` directory relative to the compiler binary,
/// then falls back to a path relative to the repository root.
fn find_bundled_headers() -> Option<PathBuf> {
    // Try relative to the executable.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // Check ../include/ (typical install layout)
            let candidate = parent.join("../include");
            if candidate.is_dir() {
                return Some(candidate.canonicalize().unwrap_or(candidate));
            }
            // Check ../../include/ (development layout: target/debug/bcc → include/)
            let candidate = parent.join("../../include");
            if candidate.is_dir() {
                return Some(candidate.canonicalize().unwrap_or(candidate));
            }
        }
    }

    // Try relative to current directory.
    let candidate = PathBuf::from("include");
    if candidate.is_dir() {
        return Some(candidate.canonicalize().unwrap_or(candidate));
    }

    None
}

/// Maps the driver CLI `OptLevel` to the passes pipeline `OptLevel`.
fn map_opt_level(cli_level: &driver::OptLevel) -> passes::pipeline::OptLevel {
    match cli_level {
        driver::OptLevel::O0 => passes::pipeline::OptLevel::O0,
        driver::OptLevel::O1 => passes::pipeline::OptLevel::O1,
        driver::OptLevel::O2 => passes::pipeline::OptLevel::O2,
    }
}
