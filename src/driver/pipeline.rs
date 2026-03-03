//! Compilation pipeline sequencing for the `bcc` C compiler.
//!
//! This module is the central orchestrator that sequences all compiler phases
//! in strict order:
//!
//! ```text
//! Read source → Preprocess → Lex → Parse → Sema → IR → Optimize → Codegen → [Debug] → Link
//! ```
//!
//! The [`run`] function is the primary entry point called from `main.rs` after
//! CLI argument parsing and target resolution. It coordinates the entire
//! compilation process, propagating errors between phases and halting on the
//! first phase that encounters errors.
//!
//! # Multi-File Compilation
//!
//! Each input source file is processed independently through the frontend
//! pipeline (preprocess → lex → parse → sema → IR → optimize → codegen).
//! All resulting [`ObjectCode`] instances are then collected and passed to the
//! linker together for final ELF binary production.
//!
//! # Error Propagation
//!
//! After each pipeline phase, `DiagnosticEmitter::has_errors()` is checked.
//! If any errors were reported, the pipeline halts immediately and returns
//! `Err(())`. This ensures that downstream phases never operate on invalid
//! intermediate representations.
//!
//! # No External Toolchain Invocations
//!
//! Per AAP §0.7: the compiler must not invoke any external tool during
//! compilation. All preprocessing, lexing, parsing, semantic analysis, IR
//! generation, optimization, instruction encoding, and linking happen within
//! the single `bcc` process.
//!
//! # Performance
//!
//! The pipeline is designed to compile the SQLite amalgamation (~230K LOC)
//! in under 60 seconds on a single core at `-O0`, with peak RSS under 2 GB.
//! String interning and arena allocation (via [`Interner`] and [`Arena`])
//! are key mechanisms for achieving this.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. It is purely orchestration logic.

use std::fs;
use std::path::{Path, PathBuf};

use crate::common::{Arena, DiagnosticEmitter, Interner, SourceMap};
use crate::codegen::{generate_code, ObjectCode};
use crate::debug::{
    CompilationUnitDebugInfo, DebugInfoGenerator, DwarfSections, FunctionDebugInfo,
    FunctionFrameInfo,
};
use crate::driver::cli::{CliArgs, derive_output_path};
use crate::driver::target::TargetConfig;
use crate::frontend::{Lexer, Parser, Preprocessor, PreprocessorOptions};
use crate::ir::{IrBuilder, Module};
use crate::linker::{
    self, DebugSections, LinkerConfig, LinkerInput,
    OutputMode as LinkerOutputMode,
};
use crate::passes::{OptLevel, Pipeline};
use crate::sema::SemanticAnalyzer;

// ===========================================================================
// OutputMode — pipeline-level output mode enum
// ===========================================================================

/// Determines the output mode for the compilation pipeline.
///
/// This enum abstracts over the CLI flags (`-c`, `-shared`) to select the
/// appropriate compilation strategy:
///
/// - [`Object`](OutputMode::Object) — Compile each source file to a
///   relocatable `.o` file, skip linking (`-c` flag).
/// - [`Executable`](OutputMode::Executable) — Full compilation and linking
///   into a static executable (default mode).
/// - [`SharedLibrary`](OutputMode::SharedLibrary) — Full compilation and
///   linking into a shared library (`-shared` flag, usually paired with
///   `-fPIC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Compile to relocatable object only (`-c` flag). Each input file
    /// produces a separate `.o` file. The linker is not invoked.
    Object,
    /// Link into a static executable (default). All input objects are linked
    /// with CRT startup objects and system libraries to produce an ELF
    /// executable.
    Executable,
    /// Link into a shared library (`-shared` flag). All input objects are
    /// linked into a position-independent shared object with dynamic linking
    /// support (`.dynamic`, `.dynsym`, PLT/GOT).
    SharedLibrary,
}

impl OutputMode {
    /// Determines the output mode from parsed CLI arguments.
    ///
    /// Resolution priority:
    /// 1. `-c` flag → `Object` (compile-only, skip linking)
    /// 2. `-shared` flag → `SharedLibrary`
    /// 3. Otherwise → `Executable` (default)
    pub fn from_cli_args(args: &CliArgs) -> Self {
        if args.compile_only {
            OutputMode::Object
        } else if args.shared {
            OutputMode::SharedLibrary
        } else {
            OutputMode::Executable
        }
    }

    /// Converts this pipeline-level output mode to the linker's output mode
    /// enum for passing to `linker::link()`.
    fn to_linker_output_mode(self) -> LinkerOutputMode {
        match self {
            OutputMode::Object => LinkerOutputMode::Relocatable,
            OutputMode::Executable => LinkerOutputMode::StaticExecutable,
            OutputMode::SharedLibrary => LinkerOutputMode::SharedLibrary,
        }
    }
}

// ===========================================================================
// CompilationContext — shared state across pipeline phases
// ===========================================================================

/// Shared compilation state propagated through all pipeline phases.
///
/// `CompilationContext` bundles the parsed CLI arguments, resolved target
/// configuration, and shared infrastructure (source map, diagnostics, interner,
/// arena) into a single struct. This avoids threading many individual parameters
/// through every pipeline function call.
///
/// # Ownership
///
/// The context owns all shared resources. Pipeline phases borrow them via
/// mutable or immutable references as needed. The context is created once at
/// the start of `run()` and lives for the entire compilation.
pub struct CompilationContext {
    /// Parsed CLI arguments providing all compiler flags and input file paths.
    pub cli_args: CliArgs,
    /// Resolved target architecture configuration (pointer size, ABI, ELF
    /// class, endianness, register counts, CRT search paths).
    pub target: TargetConfig,
    /// Source file registry mapping byte offsets to line/column positions.
    /// Populated during source file reading and preprocessing.
    pub source_map: SourceMap,
    /// GCC-compatible diagnostic accumulator and stderr writer. Checked
    /// after each pipeline phase via `has_errors()` for halt-on-error.
    pub diagnostics: DiagnosticEmitter,
    /// String interner deduplicating identifiers, keywords, and string
    /// literals across the entire compilation for O(1) equality comparison.
    pub interner: Interner,
    /// Arena allocator for AST and IR nodes providing O(1) bump allocation
    /// and batch deallocation to meet the <2GB RSS performance constraint.
    pub arena: Arena,
}

impl CompilationContext {
    /// Creates a new `CompilationContext` from parsed CLI arguments and a
    /// resolved target configuration.
    ///
    /// Initializes all shared infrastructure (source map, diagnostics,
    /// interner, arena) to their empty/default states.
    pub fn new(cli_args: CliArgs, mut target: TargetConfig) -> Self {
        // Propagate codegen-relevant CLI flags into TargetConfig so that
        // the backend (which only receives TargetConfig) can access them.
        target.retpoline = cli_args.retpoline;
        target.cf_protection = cli_args.cf_protection;
        target.pic = cli_args.pic;
        CompilationContext {
            cli_args,
            target,
            source_map: SourceMap::new(),
            diagnostics: DiagnosticEmitter::new(),
            interner: Interner::with_keywords(),
            arena: Arena::new(),
        }
    }
}

// ===========================================================================
// run() — main pipeline orchestration entry point
// ===========================================================================

/// Runs the complete compilation pipeline for all input files.
///
/// This is the primary entry point called from `main.rs` after CLI argument
/// parsing and target resolution. It sequences all compiler phases in strict
/// order, halting on the first phase that encounters errors.
///
/// # Pipeline Phases
///
/// 1. **Source file reading** — Read each input `.c` file from disk.
/// 2. **Preprocessing** (F-001) — Macro expansion, include resolution,
///    conditional compilation.
/// 3. **Lexing** (F-002) — Tokenize preprocessed source into `Vec<Token>`.
/// 4. **Parsing** (F-003) — Produce untyped AST via recursive descent.
/// 5. **Semantic analysis** (F-004) — Type checking, symbol resolution.
/// 6. **IR generation** (F-005) — Lower typed AST to SSA-form IR.
/// 7. **Optimization** (F-006) — Apply passes per `-O` level.
/// 8. **Code generation** (F-007) — Generate machine code for target arch.
/// 9. **Debug info** (conditional) — DWARF v4 sections if `-g` is set.
/// 10. **Linking/output** (F-008) — Produce ELF binary or `.o` file.
///
/// # Arguments
///
/// * `cli_args` — Parsed command-line arguments.
/// * `target` — Resolved target architecture configuration.
///
/// # Returns
///
/// * `Ok(())` — Compilation succeeded with no errors.
/// * `Err(())` — Compilation failed; errors have been emitted to stderr.
///
/// # Error Handling
///
/// After each phase, `DiagnosticEmitter::has_errors()` is checked. On any
/// error, the pipeline returns `Err(())` immediately. All error messages
/// follow GCC-compatible `file:line:col: error: message` format on stderr.
pub fn run(cli_args: CliArgs, target: TargetConfig) -> Result<(), ()> {
    let output_mode = OutputMode::from_cli_args(&cli_args);
    let mut ctx = CompilationContext::new(cli_args, target);

    // Collect compiled object code from all input files.
    let mut all_objects: Vec<ObjectCode> = Vec::new();
    // Collect debug info per file (only if -g is specified).
    let mut all_debug_cu_info: Vec<CompilationUnitDebugInfo> = Vec::new();

    // Determine the bundled header path from the build script environment
    // variable or fall back to the repository's `include/` directory.
    let bundled_header_path = resolve_bundled_header_path();

    // ------------------------------------------------------------------
    // Per-file frontend + middle-end + backend pipeline
    // ------------------------------------------------------------------
    let input_files = ctx.cli_args.input_files.clone();
    for input_path_str in &input_files {
        let input_path = Path::new(input_path_str);

        // === Handle pre-compiled object files (.o) directly ===
        if let Some(ext) = input_path.extension() {
            if ext == "o" {
                match read_object_file(input_path, &ctx.target) {
                    Ok(obj) => {
                        all_objects.push(obj);
                        continue;
                    }
                    Err(msg) => {
                        ctx.diagnostics.error_no_loc(msg);
                        return Err(());
                    }
                }
            }
        }

        // === Step 1: Read source file ===
        let source = match fs::read_to_string(input_path) {
            Ok(content) => content,
            Err(err) => {
                ctx.diagnostics.error_no_loc(format!(
                    "{}: {}", input_path.display(), err
                ));
                return Err(());
            }
        };

        // Register the file in the source map.
        let file_id = ctx.source_map.add_file(
            PathBuf::from(input_path),
            source.clone(),
        );
        // Sync file paths so diagnostics can resolve FileIds to file names.
        ctx.diagnostics.sync_source_map(&ctx.source_map);

        // === Step 2: Preprocessing (Phase F-001) ===
        let pp_options = build_preprocessor_options(&ctx.cli_args, &bundled_header_path, &ctx.target);
        let mut preprocessor = Preprocessor::new(
            pp_options,
            &mut ctx.source_map,
            &mut ctx.diagnostics,
        );
        let preprocessed = match preprocessor.process(
            &source,
            input_path,
            &mut ctx.source_map,
            &mut ctx.diagnostics,
        ) {
            Ok(text) => text,
            Err(()) => {
                // Errors already emitted by the preprocessor.
                return Err(());
            }
        };

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // Re-sync source map after preprocessing (includes may have added files).
        ctx.diagnostics.sync_source_map(&ctx.source_map);

        // === Step 3: Lexing (Phase F-002) ===
        let tokens = {
            let mut lexer = Lexer::new(
                &preprocessed,
                file_id,
                &mut ctx.interner,
                &mut ctx.diagnostics,
            );
            lexer.tokenize()
        };

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // === Step 4: Parsing (Phase F-003) ===
        let ast = match Parser::parse(
            &tokens,
            &ctx.interner,
            &mut ctx.diagnostics,
        ) {
            Ok(tu) => tu,
            Err(()) => {
                return Err(());
            }
        };

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // === Step 5: Semantic Analysis (Phase F-004) ===
        let typed_ast = match SemanticAnalyzer::analyze(
            &ast,
            &ctx.target,
            &ctx.interner,
            &mut ctx.diagnostics,
        ) {
            Ok(typed_tu) => typed_tu,
            Err(()) => {
                return Err(());
            }
        };

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // === Step 6: IR Generation (Phase F-005) ===
        let module_name = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");
        let ir_module = {
            let mut builder = IrBuilder::new(
                &ctx.target,
                &mut ctx.diagnostics,
                module_name,
                &ctx.interner,
            );
            builder.build(&typed_ast)
        };

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // === Step 7: Optimization (Phase F-006) ===
        let opt_level = cli_opt_level_to_passes(&ctx.cli_args.opt_level);
        let mut optimized_module = ir_module;
        {
            let pipeline = Pipeline::new(opt_level);
            let _stats = pipeline.run_on_module(&mut optimized_module);
        }

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // === Step 8: Code Generation (Phase F-007) ===
        let object_code = match generate_code(&optimized_module, &ctx.target) {
            Ok(obj) => obj,
            Err(err) => {
                ctx.diagnostics.error_no_loc(format!(
                    "code generation failed: {}", err
                ));
                return Err(());
            }
        };

        if ctx.diagnostics.has_errors() {
            return Err(());
        }

        // === Step 9: Debug Info Generation (conditional) ===
        if ctx.cli_args.debug_info {
            let cu_info = build_debug_cu_info(
                input_path_str,
                &optimized_module,
                &object_code,
                &typed_ast,
                &ctx.interner,
                &ctx.target,
            );
            all_debug_cu_info.push(cu_info);
        }

        all_objects.push(object_code);
    }

    // ------------------------------------------------------------------
    // Output phase: Object files or Linked binary
    // ------------------------------------------------------------------
    match output_mode {
        OutputMode::Object => {
            // In compile-only mode (-c), emit each object as a relocatable
            // ELF .o file. Each input file produces its own .o output.
            emit_object_files(&ctx, all_objects, all_debug_cu_info)?;
        }
        OutputMode::Executable | OutputMode::SharedLibrary => {
            // Full linking mode: pass all compiled objects to the linker.
            link_objects(&ctx, output_mode, all_objects, all_debug_cu_info)?;
        }
    }

    if ctx.diagnostics.has_errors() {
        return Err(());
    }

    Ok(())
}

// ===========================================================================
// Helper: Build preprocessor options from CLI arguments
// ===========================================================================

/// Constructs [`PreprocessorOptions`] from parsed CLI arguments.
///
/// Translates `-I` include paths, `-D` macro definitions, and `-U` macro
/// undefinitions from the [`CliArgs`] format into the preprocessor's expected
/// format. Also sets the bundled freestanding header path.
fn build_preprocessor_options(
    cli_args: &CliArgs,
    bundled_header_path: &Option<PathBuf>,
    target: &crate::driver::target::TargetConfig,
) -> PreprocessorOptions {
    let mut options = PreprocessorOptions::new();

    // Include paths from -I flags.
    for dir in &cli_args.include_paths {
        options.include_dirs.push(PathBuf::from(dir));
    }

    // Macro definitions from -D flags.
    for def in &cli_args.defines {
        options.defines.push((def.name.clone(), def.value.clone()));
    }

    // Macro undefinitions from -U flags.
    for undef in &cli_args.undefines {
        options.undefines.push(undef.clone());
    }

    // Bundled freestanding header path.
    options.bundled_header_path = bundled_header_path.clone();

    // Architecture-specific predefined macros for system header compatibility.
    // These must match what GCC/Clang define for each target so that system
    // headers like <gnu/stubs.h> select the correct architecture paths.
    let arch_str = &target.triple;
    if arch_str.starts_with("x86_64") || arch_str.starts_with("x86-64") {
        options.defines.push(("__x86_64__".to_string(), Some("1".to_string())));
        options.defines.push(("__x86_64".to_string(), Some("1".to_string())));
        options.defines.push(("__amd64__".to_string(), Some("1".to_string())));
        options.defines.push(("__amd64".to_string(), Some("1".to_string())));
        options.defines.push(("__LP64__".to_string(), Some("1".to_string())));
        options.defines.push(("_LP64".to_string(), Some("1".to_string())));
        options.defines.push(("__SIZEOF_POINTER__".to_string(), Some("8".to_string())));
        options.defines.push(("__SIZEOF_LONG__".to_string(), Some("8".to_string())));
        options.defines.push(("__SIZEOF_INT__".to_string(), Some("4".to_string())));
    } else if arch_str.starts_with("i686") || arch_str.starts_with("i386") {
        options.defines.push(("__i686__".to_string(), Some("1".to_string())));
        options.defines.push(("__i386__".to_string(), Some("1".to_string())));
        options.defines.push(("__i386".to_string(), Some("1".to_string())));
        options.defines.push(("i386".to_string(), Some("1".to_string())));
        options.defines.push(("__ILP32__".to_string(), Some("1".to_string())));
        options.defines.push(("__SIZEOF_POINTER__".to_string(), Some("4".to_string())));
        options.defines.push(("__SIZEOF_LONG__".to_string(), Some("4".to_string())));
        options.defines.push(("__SIZEOF_INT__".to_string(), Some("4".to_string())));
    } else if arch_str.starts_with("aarch64") {
        options.defines.push(("__aarch64__".to_string(), Some("1".to_string())));
        options.defines.push(("__LP64__".to_string(), Some("1".to_string())));
        options.defines.push(("_LP64".to_string(), Some("1".to_string())));
        options.defines.push(("__SIZEOF_POINTER__".to_string(), Some("8".to_string())));
        options.defines.push(("__SIZEOF_LONG__".to_string(), Some("8".to_string())));
        options.defines.push(("__SIZEOF_INT__".to_string(), Some("4".to_string())));
        // Suppress ARM NEON and SVE vector type declarations in glibc math.h.
        // Our compiler does not support NEON/SVE intrinsics, so these types
        // (__f32x4_t, __sv_f32_t, etc.) would cause parse errors.
        options.defines.push(("__ARM_NEON".to_string(), Some("0".to_string())));
        options.defines.push(("__ARM_FEATURE_SVE".to_string(), Some("0".to_string())));
        // Define GCC builtin vector types as simple integer types so that
        // glibc's bits/math-vector.h ADVSIMD/SVE sections parse cleanly.
        // These types are only used in SIMD math function declarations that
        // our compiler never invokes, so substituting `int` is sufficient.
        options.defines.push(("__Float32x4_t".to_string(), Some("int".to_string())));
        options.defines.push(("__Float64x2_t".to_string(), Some("long".to_string())));
        options.defines.push(("__SVFloat32_t".to_string(), Some("int".to_string())));
        options.defines.push(("__SVFloat64_t".to_string(), Some("long".to_string())));
        options.defines.push(("__SVBool_t".to_string(), Some("int".to_string())));
        // Suppress the __aarch64_vector_pcs__ calling convention attribute.
        options.defines.push(("__vpcs".to_string(), None));
    } else if arch_str.starts_with("riscv64") {
        options.defines.push(("__riscv".to_string(), Some("1".to_string())));
        options.defines.push(("__riscv_xlen".to_string(), Some("64".to_string())));
        // RISC-V floating-point register length — required by glibc's
        // bits/setjmp.h to determine how many FP registers to save.
        // 64 = double-precision (D extension), matching RV64GC.
        options.defines.push(("__riscv_flen".to_string(), Some("64".to_string())));
        // Double-precision float ABI flag required by glibc's bits/setjmp.h
        // to include FP callee-saved registers in jmp_buf.
        options.defines.push(("__riscv_float_abi_double".to_string(), Some("1".to_string())));
        options.defines.push(("__LP64__".to_string(), Some("1".to_string())));
        options.defines.push(("_LP64".to_string(), Some("1".to_string())));
        options.defines.push(("__SIZEOF_POINTER__".to_string(), Some("8".to_string())));
        options.defines.push(("__SIZEOF_LONG__".to_string(), Some("8".to_string())));
        options.defines.push(("__SIZEOF_INT__".to_string(), Some("4".to_string())));
    }

    // System include directories based on target architecture.
    options.system_include_dirs.push(PathBuf::from("/usr/include"));
    if arch_str.starts_with("x86_64") || arch_str.starts_with("x86-64") {
        options.system_include_dirs.push(PathBuf::from("/usr/include/x86_64-linux-gnu"));
    } else if arch_str.starts_with("i686") || arch_str.starts_with("i386") {
        options.system_include_dirs.push(PathBuf::from("/usr/include/i386-linux-gnu"));
        options.system_include_dirs.push(PathBuf::from("/usr/include/x86_64-linux-gnu"));
    } else if arch_str.starts_with("aarch64") {
        options.system_include_dirs.push(PathBuf::from("/usr/aarch64-linux-gnu/include"));
        options.system_include_dirs.push(PathBuf::from("/usr/include/aarch64-linux-gnu"));
    } else if arch_str.starts_with("riscv64") {
        options.system_include_dirs.push(PathBuf::from("/usr/riscv64-linux-gnu/include"));
        options.system_include_dirs.push(PathBuf::from("/usr/include/riscv64-linux-gnu"));
    }

    options
}

// ===========================================================================
// Helper: Resolve bundled header path
// ===========================================================================

/// Determines the path to the bundled freestanding headers directory.
///
/// Resolution order:
/// 1. The `BCC_BUNDLED_INCLUDE_DIR` environment variable (set by `build.rs`
///    at compile time or at runtime).
/// 2. An `include/` directory adjacent to the current executable.
/// 3. An `include/` directory in the current working directory.
/// 4. `None` if no bundled headers directory is found.
fn resolve_bundled_header_path() -> Option<PathBuf> {
    // 1. Check compile-time or runtime environment variable.
    if let Ok(dir) = std::env::var("BCC_BUNDLED_INCLUDE_DIR") {
        let path = PathBuf::from(&dir);
        if path.is_dir() {
            return Some(path);
        }
    }

    // 2. Check adjacent to the executable.
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let include_dir = exe_dir.join("include");
            if include_dir.is_dir() {
                return Some(include_dir);
            }
        }
    }

    // 3. Check current working directory.
    let cwd_include = PathBuf::from("include");
    if cwd_include.is_dir() {
        return Some(cwd_include);
    }

    None
}

// ===========================================================================
// Helper: Convert CLI OptLevel to passes OptLevel
// ===========================================================================

/// Converts the CLI-level [`crate::driver::cli::OptLevel`] to the passes
/// module's [`crate::passes::OptLevel`].
///
/// Both enums have the same variants (O0, O1, O2) but are defined in
/// different modules for separation of concerns. This function bridges them.
fn cli_opt_level_to_passes(cli_level: &crate::driver::cli::OptLevel) -> OptLevel {
    match cli_level {
        crate::driver::cli::OptLevel::O0 => OptLevel::O0,
        crate::driver::cli::OptLevel::O1 => OptLevel::O1,
        crate::driver::cli::OptLevel::O2 => OptLevel::O2,
    }
}

// ===========================================================================
// Helper: Emit object files in compile-only mode (-c)
// ===========================================================================

/// Emits relocatable ELF object files for compile-only mode (`-c` flag).
///
/// Each input file produces a separate `.o` output. The output path is either
/// specified by `-o` (for single-file compilations) or derived from the input
/// filename (e.g., `foo.c` → `foo.o`).
fn emit_object_files(
    ctx: &CompilationContext,
    objects: Vec<ObjectCode>,
    debug_cu_infos: Vec<CompilationUnitDebugInfo>,
) -> Result<(), ()> {
    let num_objects = objects.len();
    for (idx, obj) in objects.into_iter().enumerate() {
        // Determine the output path for this object file.
        let output_path = if num_objects == 1 {
            // Single file: use -o if specified, otherwise derive from input.
            PathBuf::from(derive_output_path(&ctx.cli_args))
        } else {
            // Multiple files: derive .o path from each input filename.
            let input_path = Path::new(&ctx.cli_args.input_files[idx]);
            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            PathBuf::from(format!("{}.o", stem))
        };

        // Build linker input for relocatable output.
        let debug_sections = if ctx.cli_args.debug_info && idx < debug_cu_infos.len() {
            let dwarf = generate_dwarf_sections(
                &debug_cu_infos[idx],
                &ctx.target,
            );
            Some(convert_dwarf_to_linker_debug(dwarf))
        } else {
            None
        };

        let linker_input = LinkerInput {
            objects: vec![obj],
            debug_sections,
        };

        let linker_config = LinkerConfig {
            output_mode: LinkerOutputMode::Relocatable,
            output_path: output_path.clone(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            force_static: false,
            target: ctx.target.clone(),
            entry_point: String::from("_start"),
        };

        match linker::link(linker_input, &linker_config) {
            Ok(elf_bytes) => {
                if let Err(err) = fs::write(&output_path, &elf_bytes) {
                    eprintln!(
                        "error: failed to write output file '{}': {}",
                        output_path.display(),
                        err
                    );
                    return Err(());
                }
            }
            Err(err) => {
                eprintln!("error: linking failed: {}", err);
                return Err(());
            }
        }
    }

    Ok(())
}

// ===========================================================================
// Helper: Link objects into final binary
// ===========================================================================

/// Links all compiled objects into a final ELF binary (executable or shared
/// library).
///
/// Constructs the [`LinkerConfig`] from CLI arguments and target configuration,
/// packages all [`ObjectCode`] and optional DWARF debug sections into a
/// [`LinkerInput`], and invokes the integrated linker.
fn link_objects(
    ctx: &CompilationContext,
    output_mode: OutputMode,
    objects: Vec<ObjectCode>,
    debug_cu_infos: Vec<CompilationUnitDebugInfo>,
) -> Result<(), ()> {
    let output_path = PathBuf::from(derive_output_path(&ctx.cli_args));

    // Build debug sections if -g was specified and we have debug info.
    let debug_sections = if ctx.cli_args.debug_info && !debug_cu_infos.is_empty() {
        // Merge all CU debug infos into combined DWARF sections.
        let merged_dwarf = merge_debug_sections(&debug_cu_infos, &ctx.target);
        Some(merged_dwarf)
    } else {
        None
    };

    let linker_input = LinkerInput {
        objects,
        debug_sections,
    };

    let library_paths: Vec<PathBuf> = ctx.cli_args.library_paths
        .iter()
        .map(|p| PathBuf::from(p))
        .collect();

    let libraries = ctx.cli_args.libraries.clone();

    let linker_config = LinkerConfig {
        output_mode: output_mode.to_linker_output_mode(),
        output_path: output_path.clone(),
        library_paths,
        libraries,
        force_static: ctx.cli_args.static_link,
        target: ctx.target.clone(),
        entry_point: String::from("_start"),
    };

    match linker::link(linker_input, &linker_config) {
        Ok(elf_bytes) => {
            if let Err(err) = fs::write(&output_path, &elf_bytes) {
                eprintln!(
                    "error: failed to write output file '{}': {}",
                    output_path.display(),
                    err
                );
                return Err(());
            }

            // Set executable permission on Unix for executables.
            #[cfg(unix)]
            if output_mode == OutputMode::Executable {
                set_executable_permission(&output_path);
            }

            Ok(())
        }
        Err(err) => {
            eprintln!("error: linking failed: {}", err);
            Err(())
        }
    }
}

// ===========================================================================
// Helper: Set executable permissions on Unix
// ===========================================================================

/// Sets the executable permission bit on the output file (Unix only).
///
/// This is needed because `fs::write` creates files without the execute bit
/// by default. Without this, the user would need to `chmod +x` the output
/// before running it.
#[cfg(unix)]
fn set_executable_permission(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = fs::metadata(path) {
        let mut perms = metadata.permissions();
        let mode = perms.mode() | 0o111; // Add execute bits for user/group/other.
        perms.set_mode(mode);
        let _ = fs::set_permissions(path, perms);
    }
}

// ===========================================================================
// Helper: Build CompilationUnitDebugInfo
// ===========================================================================

/// Constructs a [`CompilationUnitDebugInfo`] for DWARF generation from the
/// IR module and compiled object code.
///
/// This creates the metadata structure that the debug info generator needs
/// to produce `.debug_info`, `.debug_line`, and other DWARF sections.
fn build_debug_cu_info(
    source_file: &str,
    ir_module: &Module,
    object_code: &ObjectCode,
    typed_ast: &crate::sema::TypedTranslationUnit,
    interner: &Interner,
    target: &TargetConfig,
) -> CompilationUnitDebugInfo {
    // Determine compilation directory (current working directory).
    let comp_dir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    // Build a lookup from function name to (offset, size) in the object code
    // so we can populate low_pc / high_pc with real addresses.
    let sym_map: std::collections::HashMap<&str, (u64, u64)> = object_code
        .symbols
        .iter()
        .filter(|s| {
            s.is_definition
                && matches!(
                    s.symbol_type,
                    crate::codegen::SymbolType::Function
                )
        })
        .map(|s| (s.name.as_str(), (s.offset, s.size)))
        .collect();

    // Track the overall code range.
    let mut cu_low: u64 = u64::MAX;
    let mut cu_high: u64 = 0;

    // Collect function debug info from the IR module, wiring in addresses.
    let mut line_counter: u32 = 1; // Approximate line number per function
    let functions: Vec<FunctionDebugInfo> = ir_module
        .functions
        .iter()
        .filter(|f| f.is_definition)
        .map(|f| {
            let (low_pc, high_pc) = sym_map
                .get(f.name.as_str())
                .copied()
                .map(|(off, sz)| (off, off + sz))
                .unwrap_or((0, 0));

            if low_pc < cu_low {
                cu_low = low_pc;
            }
            if high_pc > cu_high {
                cu_high = high_pc;
            }

            let decl_line = line_counter;
            line_counter += 3; // Approximate spacing

            // Create a minimal line mapping — one entry per function mapping
            // the start address to the function's source line.
            let line_mappings = if low_pc < high_pc {
                vec![crate::debug::SourceMapping {
                    address: low_pc,
                    file_id: 0, // 0-based file index (first source file)
                    line: decl_line,
                    column: 1,
                    is_stmt: true,
                }]
            } else {
                Vec::new()
            };

            // Extract parameter and local variable debug info from typed AST.
            let (params_dbg, locals_dbg, ret_type_dbg) =
                extract_function_debug_types(&f.name, typed_ast, interner);

            FunctionDebugInfo {
                name: f.name.clone(),
                linkage_name: None,
                low_pc,
                high_pc,
                file_id: 0,
                line: decl_line,
                return_type: ret_type_dbg,
                parameters: params_dbg,
                local_variables: locals_dbg,
                line_mappings,
                frame_info: FunctionFrameInfo {
                    cfa_register: 6, // rbp for x86-64
                    cfa_offset: 16,
                    saved_registers: Vec::new(),
                },
            }
        })
        .collect();

    if cu_low == u64::MAX {
        cu_low = 0;
    }

    // Extract struct definitions from the typed AST for DWARF type info.
    let struct_defs = extract_struct_defs(typed_ast, interner, target);

    CompilationUnitDebugInfo {
        producer: "bcc 0.1.0".to_string(),
        language: crate::debug::DW_LANG_C11,
        source_file: source_file.to_string(),
        comp_dir,
        low_pc: cu_low,
        high_pc: cu_high,
        functions,
        global_variables: Vec::new(),
        struct_defs,
        source_files: vec![source_file.to_string()],
        include_directories: Vec::new(),
    }
}

// ===========================================================================
// Helper: Generate DWARF sections
// ===========================================================================

/// Generates DWARF v4 debug sections from compilation unit debug info.
fn generate_dwarf_sections(
    cu_info: &CompilationUnitDebugInfo,
    target: &TargetConfig,
) -> DwarfSections {
    let address_size = if target.is_64bit() { 8u8 } else { 4u8 };
    let generator = DebugInfoGenerator::new(address_size, target.arch);
    generator.generate(cu_info)
}

// ===========================================================================
// Helper: Convert DwarfSections to linker DebugSections
// ===========================================================================

/// Converts the debug module's [`DwarfSections`] to the linker's
/// [`DebugSections`] format.
///
/// Both types carry the same DWARF section byte vectors but are defined in
/// different modules for separation of concerns.
fn convert_dwarf_to_linker_debug(dwarf: DwarfSections) -> DebugSections {
    DebugSections {
        debug_info: dwarf.debug_info,
        debug_abbrev: dwarf.debug_abbrev,
        debug_line: dwarf.debug_line,
        debug_str: dwarf.debug_str,
        debug_aranges: dwarf.debug_aranges,
        debug_frame: dwarf.debug_frame,
        debug_loc: dwarf.debug_loc,
        relocations: Vec::new(),
    }
}

// ===========================================================================
// Helper: Merge debug sections from multiple CUs
// ===========================================================================

/// Merges debug section data from multiple compilation units into a single
/// set of [`DebugSections`] for the linker.
///
/// For multi-file compilations, each source file produces its own DWARF
/// data. This function concatenates the per-CU sections so the linker
/// receives one unified set of debug sections.
fn merge_debug_sections(
    cu_infos: &[CompilationUnitDebugInfo],
    target: &TargetConfig,
) -> DebugSections {
    let mut merged = DebugSections {
        debug_info: Vec::new(),
        debug_abbrev: Vec::new(),
        debug_line: Vec::new(),
        debug_str: Vec::new(),
        debug_aranges: Vec::new(),
        debug_frame: Vec::new(),
        debug_loc: Vec::new(),
        relocations: Vec::new(),
    };

    for cu_info in cu_infos {
        let dwarf = generate_dwarf_sections(cu_info, target);
        merged.debug_info.extend_from_slice(&dwarf.debug_info);
        merged.debug_abbrev.extend_from_slice(&dwarf.debug_abbrev);
        merged.debug_line.extend_from_slice(&dwarf.debug_line);
        merged.debug_str.extend_from_slice(&dwarf.debug_str);
        merged.debug_aranges.extend_from_slice(&dwarf.debug_aranges);
        merged.debug_frame.extend_from_slice(&dwarf.debug_frame);
        merged.debug_loc.extend_from_slice(&dwarf.debug_loc);
    }

    merged
}

// ===========================================================================
// Unit Tests
// ===========================================================================

// ===========================================================================
// read_object_file — parse a pre-compiled .o ELF into ObjectCode
// ===========================================================================

/// Reads a pre-compiled ELF relocatable object (`.o`) file and converts it
/// into the internal [`ObjectCode`] representation so that it can be passed
/// directly to the linker alongside freshly compiled objects.
///
/// This enables the `bcc` CLI to accept `.o` files as inputs for multi-file
/// linking (e.g. `bcc a.o b.o -o out`).
fn read_object_file(path: &Path, target: &TargetConfig) -> Result<ObjectCode, String> {
    use crate::codegen::{
        Architecture, Relocation, RelocationType,
        Section as CgSection, SectionFlags, SectionType as CgSectionType,
        Symbol as CgSymbol, SymbolBinding, SymbolType as CgSymbolType,
        SymbolVisibility,
    };
    use crate::linker::elf;

    let data = fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let elf_obj = elf::ElfObject::parse(&data)
        .map_err(|e| format!("{}: invalid ELF object: {:?}", path.display(), e))?;

    // Determine architecture from ELF machine type.
    let arch = match elf_obj.machine {
        elf::EM_X86_64 => Architecture::X86_64,
        elf::EM_386 => Architecture::I686,
        elf::EM_AARCH64 => Architecture::Aarch64,
        elf::EM_RISCV => Architecture::Riscv64,
        m => return Err(format!("{}: unsupported ELF machine type {}", path.display(), m)),
    };

    let mut obj = ObjectCode::new(arch);

    // Convert parsed sections to codegen sections, skipping metadata sections.
    // Build a mapping from ELF section index → codegen section index.
    let mut elf_to_cg_section: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (elf_idx, sec) in elf_obj.sections.iter().enumerate() {
        // Skip non-content sections.
        if sec.section_type == elf::SHT_NULL
            || sec.section_type == elf::SHT_STRTAB
            || sec.section_type == elf::SHT_SYMTAB
            || sec.section_type == elf::SHT_RELA
            || sec.section_type == elf::SHT_REL
        {
            continue;
        }

        let section_type = match sec.name.as_str() {
            ".text" => CgSectionType::Text,
            ".data" => CgSectionType::Data,
            ".rodata" => CgSectionType::Rodata,
            ".bss" => CgSectionType::Bss,
            _ => CgSectionType::Custom(sec.section_type),
        };

        let flags = SectionFlags {
            writable: (sec.flags & elf::SHF_WRITE) != 0,
            executable: (sec.flags & elf::SHF_EXECINSTR) != 0,
            allocatable: (sec.flags & elf::SHF_ALLOC) != 0,
        };

        let cg_idx = obj.add_section(CgSection {
            name: sec.name.clone(),
            data: sec.data.clone(),
            section_type,
            alignment: sec.alignment as u32,
            flags,
        });
        elf_to_cg_section.insert(elf_idx, cg_idx);
    }

    // Convert parsed symbols. The ELF section_index is 1-based; we map to
    // the 0-based codegen section index via elf_to_cg_section.
    for sym in &elf_obj.symbols {
        if sym.name.is_empty() {
            continue; // Skip the null symbol
        }

        let binding = match sym.binding {
            0 => SymbolBinding::Local,
            1 => SymbolBinding::Global,
            2 => SymbolBinding::Weak,
            _ => SymbolBinding::Global,
        };

        let symbol_type = match sym.symbol_type {
            2 => CgSymbolType::Function,
            1 => CgSymbolType::Object,
            3 => CgSymbolType::Section,
            _ => CgSymbolType::NoType,
        };

        let visibility = match sym.visibility {
            0 => SymbolVisibility::Default,
            2 => SymbolVisibility::Hidden,
            3 => SymbolVisibility::Protected,
            _ => SymbolVisibility::Default,
        };

        let is_def = sym.section_index != 0
            && (sym.section_index as u32) < elf::SHN_ABS as u32;
        let cg_sec_idx = if is_def {
            elf_to_cg_section
                .get(&(sym.section_index as usize))
                .copied()
                .unwrap_or(0)
        } else {
            0
        };

        obj.add_symbol(CgSymbol {
            name: sym.name.clone(),
            section_index: cg_sec_idx,
            offset: sym.value,
            size: sym.size,
            binding,
            symbol_type,
            visibility,
            is_definition: is_def,
        });
    }

    // Convert relocations — map ELF relocation types to our RelocationType.
    for reloc in &elf_obj.relocations {
        // Look up the symbol name from the ELF symbol table.
        let sym_name = if (reloc.symbol_index as usize) < elf_obj.symbols.len() {
            elf_obj.symbols[reloc.symbol_index as usize].name.clone()
        } else {
            String::new()
        };

        let cg_sec_idx = elf_to_cg_section
            .get(&reloc.section_index)
            .copied()
            .unwrap_or(0);

        let reloc_type = match arch {
            Architecture::X86_64 => match reloc.reloc_type {
                1 => RelocationType::X86_64_64,
                2 => RelocationType::X86_64_PC32,
                4 => RelocationType::X86_64_PLT32,
                9 => RelocationType::X86_64_GOTPCREL,
                _ => RelocationType::X86_64_64, // fallback
            },
            Architecture::I686 => match reloc.reloc_type {
                1 => RelocationType::I386_32,
                2 => RelocationType::I386_PC32,
                _ => RelocationType::I386_32,
            },
            Architecture::Aarch64 => match reloc.reloc_type {
                257 => RelocationType::Aarch64_ABS64,
                258 => RelocationType::Aarch64_ABS32,
                283 => RelocationType::Aarch64_CALL26,
                _ => RelocationType::Aarch64_ABS64,
            },
            Architecture::Riscv64 => match reloc.reloc_type {
                1 => RelocationType::Riscv_64,
                2 => RelocationType::Riscv_32,
                _ => RelocationType::Riscv_64,
            },
        };

        obj.add_relocation(Relocation {
            offset: reloc.offset,
            symbol: sym_name,
            reloc_type,
            addend: reloc.addend,
            section_index: cg_sec_idx,
        });
    }

    Ok(obj)
}

// ===========================================================================
// Helper: Extract debug type information from the typed AST
// ===========================================================================

use crate::frontend::parser::ast::{
    TranslationUnit, Declaration, TypeSpecifier, Declarator, DirectDeclarator,
    Statement,
};
use crate::debug::{
    DebugTypeRef, BaseTypeKind, ParameterDebugInfo,
    VariableDebugInfo as DebugVariableDebugInfo,
    VariableLocation, StructDebugDef, StructMemberDebugInfo,
};

/// Convert an AST TypeSpecifier to a DebugTypeRef.
fn type_specifier_to_debug_ref(ts: &TypeSpecifier, interner: &Interner) -> DebugTypeRef {
    match ts {
        TypeSpecifier::Void => DebugTypeRef::Void,
        TypeSpecifier::Char => DebugTypeRef::BaseType(BaseTypeKind::SignedChar),
        TypeSpecifier::Short => DebugTypeRef::BaseType(BaseTypeKind::Short),
        TypeSpecifier::Int => DebugTypeRef::BaseType(BaseTypeKind::Int),
        TypeSpecifier::Long => DebugTypeRef::BaseType(BaseTypeKind::Long),
        TypeSpecifier::LongLong => DebugTypeRef::BaseType(BaseTypeKind::LongLong),
        TypeSpecifier::Float => DebugTypeRef::BaseType(BaseTypeKind::Float),
        TypeSpecifier::Double => DebugTypeRef::BaseType(BaseTypeKind::Double),
        TypeSpecifier::LongDouble => DebugTypeRef::BaseType(BaseTypeKind::LongDouble),
        TypeSpecifier::Bool => DebugTypeRef::BaseType(BaseTypeKind::Bool),
        TypeSpecifier::Signed(inner) => match inner.as_ref() {
            TypeSpecifier::Char => DebugTypeRef::BaseType(BaseTypeKind::SignedChar),
            TypeSpecifier::Short => DebugTypeRef::BaseType(BaseTypeKind::Short),
            TypeSpecifier::Int => DebugTypeRef::BaseType(BaseTypeKind::Int),
            TypeSpecifier::Long => DebugTypeRef::BaseType(BaseTypeKind::Long),
            TypeSpecifier::LongLong => DebugTypeRef::BaseType(BaseTypeKind::LongLong),
            _ => DebugTypeRef::BaseType(BaseTypeKind::Int),
        },
        TypeSpecifier::Unsigned(inner) => match inner.as_ref() {
            TypeSpecifier::Char => DebugTypeRef::BaseType(BaseTypeKind::UnsignedChar),
            TypeSpecifier::Short => DebugTypeRef::BaseType(BaseTypeKind::UnsignedShort),
            TypeSpecifier::Int => DebugTypeRef::BaseType(BaseTypeKind::UnsignedInt),
            TypeSpecifier::Long => DebugTypeRef::BaseType(BaseTypeKind::UnsignedLong),
            TypeSpecifier::LongLong => DebugTypeRef::BaseType(BaseTypeKind::UnsignedLongLong),
            _ => DebugTypeRef::BaseType(BaseTypeKind::UnsignedInt),
        },
        TypeSpecifier::Struct(def) => {
            let name = def.tag.map(|t| interner.resolve(t).to_string()).unwrap_or_default();
            DebugTypeRef::Struct(name)
        }
        TypeSpecifier::StructRef { tag, .. } => {
            let name = interner.resolve(*tag);
            DebugTypeRef::Struct(name.to_string())
        }
        TypeSpecifier::TypedefName { name, .. } => {
            let n = interner.resolve(*name);
            DebugTypeRef::Typedef(n.to_string())
        }
        TypeSpecifier::Qualified { inner, .. } => type_specifier_to_debug_ref(inner, interner),
        _ => DebugTypeRef::BaseType(BaseTypeKind::Int), // fallback
    }
}

/// Check if a declarator is a pointer type and wrap the base type accordingly.
fn wrap_debug_type_for_declarator(
    base: DebugTypeRef,
    decl: &Declarator,
) -> DebugTypeRef {
    let mut result = base;
    // Count pointer levels from the declarator's pointer chain.
    for _ in &decl.pointer {
        result = DebugTypeRef::Pointer(Box::new(result));
    }
    // Check for array declarator in the direct part.
    if let DirectDeclarator::Array { size, .. } = &decl.direct {
        let count = match size {
            crate::frontend::parser::ast::ArraySize::Fixed(expr) => {
                // Try to extract the constant value
                if let crate::frontend::parser::ast::Expression::IntegerLiteral { value, .. } = expr.as_ref() {
                    Some(*value as u64)
                } else {
                    None
                }
            }
            _ => None,
        };
        result = DebugTypeRef::Array(Box::new(result), count);
    }
    result
}

/// Extract function debug info (parameters, locals, return type) from the typed AST.
fn extract_function_debug_types(
    func_name: &str,
    typed_ast: &crate::sema::TypedTranslationUnit,
    interner: &Interner,
) -> (Vec<ParameterDebugInfo>, Vec<DebugVariableDebugInfo>, DebugTypeRef) {
    for typed_decl in &typed_ast.declarations {
        if let Declaration::Function(fdef) = &typed_decl.decl {
            // Match by function name — extract from the direct declarator.
            let decl_name = match &fdef.declarator.direct {
                DirectDeclarator::Identifier(id) => interner.resolve(*id).to_string(),
                DirectDeclarator::Function { base, .. } => {
                    if let DirectDeclarator::Identifier(id) = base.as_ref() {
                        interner.resolve(*id).to_string()
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            if decl_name != func_name {
                continue;
            }

            // Return type from specifiers
            let ret_type = type_specifier_to_debug_ref(&fdef.specifiers.type_specifier, interner);

            // Parameters from declarator
            let params: Vec<ParameterDebugInfo> = match &fdef.declarator.direct {
                DirectDeclarator::Function { params, .. } => {
                    params.params.iter().map(|p| {
                        let param_name = if let Some(ref decl) = p.declarator {
                            match &decl.direct {
                                DirectDeclarator::Identifier(id) => interner.resolve(*id).to_string(),
                                _ => String::new(),
                            }
                        } else {
                            String::new()
                        };
                        let base_type = type_specifier_to_debug_ref(&p.specifiers.type_specifier, interner);
                        let type_ref = if let Some(ref decl) = p.declarator {
                            wrap_debug_type_for_declarator(base_type, decl)
                        } else {
                            base_type
                        };
                        ParameterDebugInfo {
                            name: param_name,
                            type_ref,
                            location: VariableLocation::FrameOffset(0),
                        }
                    }).collect()
                }
                _ => Vec::new(),
            };

            // Local variables from function body
            let locals = extract_locals_from_stmt(&fdef.body, interner);

            return (params, locals, ret_type);
        }
    }

    (Vec::new(), Vec::new(), DebugTypeRef::Void)
}

/// Recursively extract local variable declarations from a statement tree.
fn extract_locals_from_stmt(
    stmt: &Statement,
    interner: &Interner,
) -> Vec<DebugVariableDebugInfo> {
    let mut locals = Vec::new();
    match stmt {
        Statement::Compound { items, .. } => {
            for item in items {
                match item {
                    crate::frontend::parser::ast::BlockItem::Declaration(decl) => {
                        if let Declaration::Variable { specifiers, declarators, .. } = decl.as_ref() {
                            let base_type = type_specifier_to_debug_ref(&specifiers.type_specifier, interner);
                            for init_decl in declarators {
                                let var_name = match &init_decl.declarator.direct {
                                    DirectDeclarator::Identifier(id) => interner.resolve(*id).to_string(),
                                    DirectDeclarator::Array { base, .. } => {
                                        if let DirectDeclarator::Identifier(id) = base.as_ref() {
                                            interner.resolve(*id).to_string()
                                        } else { continue; }
                                    }
                                    _ => continue,
                                };
                                let type_ref = wrap_debug_type_for_declarator(base_type.clone(), &init_decl.declarator);
                                locals.push(DebugVariableDebugInfo {
                                    name: var_name,
                                    type_ref,
                                    location: VariableLocation::FrameOffset(0),
                                    scope_low_pc: 0,
                                    scope_high_pc: 0,
                                });
                            }
                        }
                    }
                    crate::frontend::parser::ast::BlockItem::Statement(inner_stmt) => {
                        locals.extend(extract_locals_from_stmt(inner_stmt, interner));
                    }
                }
            }
        }
        Statement::If { then_branch, else_branch, .. } => {
            locals.extend(extract_locals_from_stmt(then_branch, interner));
            if let Some(e) = else_branch {
                locals.extend(extract_locals_from_stmt(e, interner));
            }
        }
        Statement::While { body, .. } | Statement::DoWhile { body, .. } => {
            locals.extend(extract_locals_from_stmt(body, interner));
        }
        Statement::For { body, .. } => {
            locals.extend(extract_locals_from_stmt(body, interner));
        }
        _ => {}
    }
    locals
}

/// Extract struct definitions from the typed AST.
fn extract_struct_defs(
    typed_ast: &crate::sema::TypedTranslationUnit,
    interner: &Interner,
    _target: &TargetConfig,
) -> Vec<StructDebugDef> {
    let mut defs = Vec::new();
    for typed_decl in &typed_ast.declarations {
        // Look for struct definitions in variable declarations and function definitions
        extract_struct_from_type_specifier(&typed_decl.decl, interner, &mut defs);
    }
    defs
}

/// Extract struct definitions from various declaration contexts.
fn extract_struct_from_type_specifier(
    decl: &Declaration,
    interner: &Interner,
    defs: &mut Vec<StructDebugDef>,
) {
    match decl {
        Declaration::Variable { specifiers, .. } => {
            check_type_for_struct(&specifiers.type_specifier, interner, defs);
        }
        Declaration::Function(fdef) => {
            check_type_for_struct(&fdef.specifiers.type_specifier, interner, defs);
            // Check function body for struct usage
            extract_structs_from_stmt(&fdef.body, interner, defs);
        }
        Declaration::Typedef { specifiers, .. } => {
            check_type_for_struct(&specifiers.type_specifier, interner, defs);
        }
        _ => {}
    }
}

/// Check if a type specifier contains a struct definition and extract it.
fn check_type_for_struct(
    ts: &TypeSpecifier,
    interner: &Interner,
    defs: &mut Vec<StructDebugDef>,
) {
    if let TypeSpecifier::Struct(sdef) = ts {
        if let Some(tag) = sdef.tag {
            let name = interner.resolve(tag).to_string();
            if !defs.iter().any(|d| d.name == name) {
                let mut field_members: Vec<StructMemberDebugInfo> = Vec::new();
                let mut byte_offset: u64 = 0;
                for member in &sdef.members {
                    if let crate::frontend::parser::ast::StructMember::Field { specifiers, declarators, .. } = member {
                        for field_decl in declarators {
                            if let Some(ref decl) = field_decl.declarator {
                                let field_name = match &decl.direct {
                                    DirectDeclarator::Identifier(id) => interner.resolve(*id).to_string(),
                                    _ => continue,
                                };
                                let type_ref = type_specifier_to_debug_ref(&specifiers.type_specifier, interner);
                                field_members.push(StructMemberDebugInfo {
                                    name: field_name,
                                    byte_offset,
                                    type_ref,
                                });
                                byte_offset += 4; // simplified: assume 4 bytes per field
                            }
                        }
                    }
                }
                defs.push(StructDebugDef {
                    name,
                    byte_size: byte_offset,
                    members: field_members,
                });
            }
        }
    }
}

/// Extract struct definitions from statement blocks.
fn extract_structs_from_stmt(
    stmt: &Statement,
    interner: &Interner,
    defs: &mut Vec<StructDebugDef>,
) {
    match stmt {
        Statement::Compound { items, .. } => {
            for item in items {
                match item {
                    crate::frontend::parser::ast::BlockItem::Declaration(decl) => {
                        extract_struct_from_type_specifier(decl, interner, defs);
                    }
                    crate::frontend::parser::ast::BlockItem::Statement(inner) => {
                        extract_structs_from_stmt(inner, interner, defs);
                    }
                }
            }
        }
        Statement::If { then_branch, else_branch, .. } => {
            extract_structs_from_stmt(then_branch, interner, defs);
            if let Some(e) = else_branch {
                extract_structs_from_stmt(e, interner, defs);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::cli::CliArgs;

    // -- OutputMode tests --------------------------------------------------

    /// Verifies that `-c` flag produces Object output mode.
    #[test]
    fn test_output_mode_compile_only() {
        let mut args = CliArgs::default();
        args.compile_only = true;
        assert_eq!(OutputMode::from_cli_args(&args), OutputMode::Object);
    }

    /// Verifies that `-shared` flag produces SharedLibrary output mode.
    #[test]
    fn test_output_mode_shared() {
        let mut args = CliArgs::default();
        args.shared = true;
        assert_eq!(OutputMode::from_cli_args(&args), OutputMode::SharedLibrary);
    }

    /// Verifies that default flags produce Executable output mode.
    #[test]
    fn test_output_mode_default_executable() {
        let args = CliArgs::default();
        assert_eq!(OutputMode::from_cli_args(&args), OutputMode::Executable);
    }

    /// Verifies that `-c` takes priority over `-shared`.
    #[test]
    fn test_output_mode_compile_only_overrides_shared() {
        let mut args = CliArgs::default();
        args.compile_only = true;
        args.shared = true;
        assert_eq!(OutputMode::from_cli_args(&args), OutputMode::Object);
    }

    // -- OutputMode to LinkerOutputMode conversion -------------------------

    #[test]
    fn test_output_mode_to_linker_object() {
        assert_eq!(
            OutputMode::Object.to_linker_output_mode(),
            LinkerOutputMode::Relocatable
        );
    }

    #[test]
    fn test_output_mode_to_linker_executable() {
        assert_eq!(
            OutputMode::Executable.to_linker_output_mode(),
            LinkerOutputMode::StaticExecutable
        );
    }

    #[test]
    fn test_output_mode_to_linker_shared() {
        assert_eq!(
            OutputMode::SharedLibrary.to_linker_output_mode(),
            LinkerOutputMode::SharedLibrary
        );
    }

    // -- Output path derivation tests --------------------------------------

    #[test]
    fn test_output_path_explicit_o_flag() {
        let mut args = CliArgs::default();
        args.input_files.push("foo.c".to_string());
        args.output = Some("my_output".to_string());
        assert_eq!(derive_output_path(&args), "my_output");
    }

    #[test]
    fn test_output_path_compile_only_derives_o() {
        let mut args = CliArgs::default();
        args.input_files.push("foo.c".to_string());
        args.compile_only = true;
        assert_eq!(derive_output_path(&args), "foo.o");
    }

    #[test]
    fn test_output_path_default_a_out() {
        let mut args = CliArgs::default();
        args.input_files.push("foo.c".to_string());
        assert_eq!(derive_output_path(&args), "a.out");
    }

    #[test]
    fn test_output_path_shared_default() {
        let mut args = CliArgs::default();
        args.input_files.push("foo.c".to_string());
        args.shared = true;
        assert_eq!(derive_output_path(&args), "a.so");
    }

    // -- Preprocessor options construction ---------------------------------

    #[test]
    fn test_build_preprocessor_options_include_paths() {
        let mut args = CliArgs::default();
        args.include_paths = vec!["/usr/include".to_string(), "./include".to_string()];
        let bundled = Some(PathBuf::from("/bundled"));
        let target = crate::driver::target::TargetConfig::x86_64();
        let opts = build_preprocessor_options(&args, &bundled, &target);
        assert_eq!(opts.include_dirs.len(), 2);
        assert_eq!(opts.include_dirs[0], PathBuf::from("/usr/include"));
        assert_eq!(opts.include_dirs[1], PathBuf::from("./include"));
        assert_eq!(opts.bundled_header_path, Some(PathBuf::from("/bundled")));
    }

    #[test]
    fn test_build_preprocessor_options_defines() {
        let mut args = CliArgs::default();
        args.defines = vec![
            crate::driver::cli::MacroDefinition {
                name: "DEBUG".to_string(),
                value: Some("1".to_string()),
            },
            crate::driver::cli::MacroDefinition {
                name: "NDEBUG".to_string(),
                value: None,
            },
        ];
        let target = crate::driver::target::TargetConfig::x86_64();
        let opts = build_preprocessor_options(&args, &None, &target);
        // The first two defines are the user-specified ones; additional
        // defines are architecture-specific predefined macros added by
        // build_preprocessor_options for system header compatibility.
        assert!(opts.defines.len() >= 2, "Expected at least 2 defines, got {}", opts.defines.len());
        assert_eq!(opts.defines[0].0, "DEBUG");
        assert_eq!(opts.defines[0].1, Some("1".to_string()));
        assert_eq!(opts.defines[1].0, "NDEBUG");
        assert_eq!(opts.defines[1].1, None);
        // Verify architecture-specific macros are present.
        let has_x86_64 = opts.defines.iter().any(|(k, _)| k == "__x86_64__");
        assert!(has_x86_64, "Expected __x86_64__ in defines for x86_64 target");
    }

    #[test]
    fn test_build_preprocessor_options_undefines() {
        let mut args = CliArgs::default();
        args.undefines = vec!["FOO".to_string(), "BAR".to_string()];
        let target = crate::driver::target::TargetConfig::x86_64();
        let opts = build_preprocessor_options(&args, &None, &target);
        assert_eq!(opts.undefines.len(), 2);
        assert_eq!(opts.undefines[0], "FOO");
        assert_eq!(opts.undefines[1], "BAR");
    }

    // -- CLI OptLevel to passes OptLevel conversion -------------------------

    #[test]
    fn test_cli_opt_level_conversion_o0() {
        assert_eq!(
            cli_opt_level_to_passes(&crate::driver::cli::OptLevel::O0),
            OptLevel::O0
        );
    }

    #[test]
    fn test_cli_opt_level_conversion_o1() {
        assert_eq!(
            cli_opt_level_to_passes(&crate::driver::cli::OptLevel::O1),
            OptLevel::O1
        );
    }

    #[test]
    fn test_cli_opt_level_conversion_o2() {
        assert_eq!(
            cli_opt_level_to_passes(&crate::driver::cli::OptLevel::O2),
            OptLevel::O2
        );
    }

    // -- CompilationContext construction ------------------------------------

    #[test]
    fn test_compilation_context_new() {
        let args = CliArgs::default();
        let target = TargetConfig::x86_64();
        let ctx = CompilationContext::new(args, target);
        assert!(!ctx.diagnostics.has_errors());
        assert_eq!(ctx.diagnostics.error_count(), 0);
        assert_eq!(ctx.target.pointer_size, 8);
        assert!(ctx.target.is_64bit());
    }

    // -- Error propagation tests -------------------------------------------

    #[test]
    fn test_run_no_input_file_returns_error() {
        // When given a nonexistent file, the pipeline should return Err.
        let mut args = CliArgs::default();
        args.input_files.push("/nonexistent/path/to/file.c".to_string());
        let target = TargetConfig::x86_64();
        let result = run(args, target);
        assert!(result.is_err());
    }

    #[test]
    fn test_output_mode_equality() {
        assert_eq!(OutputMode::Object, OutputMode::Object);
        assert_eq!(OutputMode::Executable, OutputMode::Executable);
        assert_eq!(OutputMode::SharedLibrary, OutputMode::SharedLibrary);
        assert_ne!(OutputMode::Object, OutputMode::Executable);
        assert_ne!(OutputMode::Executable, OutputMode::SharedLibrary);
    }
}
