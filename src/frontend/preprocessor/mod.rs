//! C11 Preprocessor Entry Point and Module Coordinator for the `bcc` compiler.
//!
//! This module is the entry point for the preprocessor pipeline phase. It declares
//! the five submodules that implement preprocessing functionality and provides the
//! [`Preprocessor`] struct with its [`process()`](Preprocessor::process) method that
//! coordinates all preprocessing phases:
//!
//! - **Directive processing** — `#include`, `#define`, `#undef`,
//!   `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`,
//!   `#warning`, `#line`
//! - **Macro expansion** — Object-like and function-like macros with `#` and `##`
//! - **Include resolution** — `<header.h>` and `"header.h"` search across `-I`
//!   directories, bundled freestanding headers, and system paths
//! - **Conditional compilation** — Branch evaluation and line inclusion/exclusion
//!
//! # Pipeline Integration
//!
//! - **Upstream**: Receives raw C source text and CLI options from
//!   `src/driver/pipeline.rs`
//! - **Downstream**: Produces fully preprocessed source text consumed by
//!   `src/frontend/lexer/mod.rs`
//! - **Cross-cutting**: Uses `common::diagnostics` for GCC-compatible error output,
//!   `common::source_map` for file tracking
//!
//! # Performance
//!
//! Designed to handle the SQLite amalgamation (~230K LOC) within the <60s
//! compilation and <2GB RSS constraints at `-O0`. Key optimizations:
//! - String pre-allocation with capacity hints
//! - Minimal per-line heap allocation during macro expansion
//! - File content caching to avoid redundant filesystem reads for repeated includes
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Conditional compilation state machine for `#if`/`#else`/`#endif` nesting.
pub mod conditional;

/// Directive dispatch and handling for all 14 C11 preprocessor directive types.
pub mod directives;

/// Preprocessor constant expression evaluator for `#if`/`#elif` conditions.
pub mod expression;

/// Include path resolution for `#include` directives.
pub mod include;

/// Macro storage and expansion engine for `#define` processing.
pub mod macros;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::source_map::{FileId, SourceLocation, SourceMap};

// Submodule type imports for internal use
use self::conditional::ConditionalStack;
use self::directives::{
    parse_directive, process_directive, DirectiveAction, PreprocessorContext,
};
use self::include::{IncludeError, IncludeResolver};
use self::macros::{MacroDefinition, MacroKind, MacroTable, MacroToken};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum `#include` nesting depth to prevent infinite recursion from
/// circular includes that bypass the canonical-path detection. Set to 256,
/// matching GCC's default `#include` nesting limit.
const MAX_INCLUDE_DEPTH: usize = 256;

// ---------------------------------------------------------------------------
// PreprocessorOptions — configuration from CLI flags
// ---------------------------------------------------------------------------

/// Configuration options for the C preprocessor, populated from CLI flags.
///
/// This struct captures all command-line options that affect preprocessing:
/// include search paths (`-I`), macro definitions (`-D`), macro undefinitions
/// (`-U`), bundled header location, and system include directories.
///
/// # Example
///
/// ```ignore
/// let options = PreprocessorOptions {
///     include_dirs: vec![PathBuf::from("./include")],
///     defines: vec![("DEBUG".into(), Some("1".into()))],
///     undefines: vec!["NDEBUG".into()],
///     bundled_header_path: Some(PathBuf::from("./include")),
///     system_include_dirs: vec![PathBuf::from("/usr/include")],
/// };
/// ```
pub struct PreprocessorOptions {
    /// Include directories from `-I` flags, searched in the order specified.
    pub include_dirs: Vec<PathBuf>,

    /// Predefined macros from `-D` flags. Each entry is a `(name, value)` pair.
    /// If the value is `None`, the macro is defined as `1` (matching GCC behavior
    /// for `-D FOO` without an `=value`).
    pub defines: Vec<(String, Option<String>)>,

    /// Macros to undefine from `-U` flags. Applied after `-D` definitions,
    /// so `-D FOO -U FOO` results in `FOO` being undefined.
    pub undefines: Vec<String>,

    /// Path to the bundled freestanding headers directory (`include/` at the
    /// repository root). If `None`, the preprocessor falls back to the compile-time
    /// path set by `build.rs` via the `BCC_BUNDLED_INCLUDE_DIR` environment variable.
    pub bundled_header_path: Option<PathBuf>,

    /// System header search paths (e.g., `/usr/include`, cross-compilation
    /// sysroot paths). Searched last in the include resolution order.
    pub system_include_dirs: Vec<PathBuf>,
}

impl PreprocessorOptions {
    /// Creates a new `PreprocessorOptions` with all fields set to empty/default values.
    pub fn new() -> Self {
        PreprocessorOptions {
            include_dirs: Vec::new(),
            defines: Vec::new(),
            undefines: Vec::new(),
            bundled_header_path: None,
            system_include_dirs: Vec::new(),
        }
    }
}

impl Default for PreprocessorOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Preprocessor — main preprocessor coordinator
// ---------------------------------------------------------------------------

/// C11-compliant preprocessor coordinating directive processing, macro expansion,
/// include resolution, and conditional compilation.
///
/// The `Preprocessor` struct manages all preprocessing state and provides the
/// [`process()`](Preprocessor::process) method as the main entry point. It is
/// typically created once per compilation invocation with CLI-derived options
/// and reused for processing all source files.
///
/// # Usage
///
/// ```ignore
/// let mut source_map = SourceMap::new();
/// let mut diagnostics = DiagnosticEmitter::new();
/// let options = PreprocessorOptions::new();
///
/// let mut pp = Preprocessor::new(options, &mut source_map, &mut diagnostics);
/// let result = pp.process(
///     &source_text,
///     Path::new("main.c"),
///     &mut source_map,
///     &mut diagnostics,
/// );
///
/// match result {
///     Ok(preprocessed) => { /* pass to lexer */ }
///     Err(()) => { /* errors already emitted to stderr */ }
/// }
/// ```
pub struct Preprocessor {
    /// Macro definitions (object-like and function-like).
    macro_table: MacroTable,

    /// Conditional compilation state stack.
    conditional_stack: ConditionalStack,

    /// Include path resolver.
    include_resolver: IncludeResolver,

    /// Preprocessor configuration options (retained for reference).
    options: PreprocessorOptions,

    /// Cache of raw file content read from disk, keyed by canonical path.
    /// Prevents redundant filesystem reads when the same header is included
    /// from multiple translation units or referenced through different paths
    /// that resolve to the same canonical file.
    file_content_cache: HashMap<PathBuf, String>,

    /// Current include nesting depth, tracked to enforce [`MAX_INCLUDE_DEPTH`].
    include_depth: usize,
}

impl Preprocessor {
    // =======================================================================
    // Construction
    // =======================================================================

    /// Creates a new `Preprocessor` with the given options.
    ///
    /// Initializes the macro table with:
    /// 1. Standard C11 predefined macros (`__STDC__`, `__STDC_VERSION__`, etc.)
    /// 2. GCC compatibility macros (`__GNUC__`, `__GNUC_MINOR__`, etc.)
    /// 3. Date/time macros (`__DATE__`, `__TIME__`)
    /// 4. User-specified `-D` definitions
    /// 5. User-specified `-U` undefinitions (applied after `-D`)
    ///
    /// The `source_map` and `diagnostics` parameters are available for any
    /// setup-time operations that require file registration or warning emission.
    pub fn new(
        options: PreprocessorOptions,
        _source_map: &mut SourceMap,
        _diagnostics: &mut DiagnosticEmitter,
    ) -> Self {
        let mut macro_table = MacroTable::new();
        let conditional_stack = ConditionalStack::new();
        let include_resolver = IncludeResolver::new(
            options.include_dirs.clone(),
            options.bundled_header_path.clone(),
            options.system_include_dirs.clone(),
        );

        // Set up standard C11 and GCC compatibility predefined macros.
        Self::setup_predefined_macros(&mut macro_table);

        // Process -D flags: use a HashMap to deduplicate definitions where
        // a later -D for the same name overrides an earlier one.
        let mut define_map: HashMap<String, Option<String>> = HashMap::new();
        for (name, value) in &options.defines {
            define_map.insert(name.clone(), value.clone());
        }

        // Process -U flags: remove from the define map so that -D FOO -U FOO
        // results in FOO being undefined. Also remove any predefined macros.
        for name in &options.undefines {
            if define_map.contains_key(name) {
                define_map.remove(name);
            }
        }

        // Apply remaining definitions to the macro table.
        for (name, value) in &define_map {
            let replacement_text = match define_map.get(name) {
                Some(Some(val)) => val.clone(),
                _ => "1".to_string(),
            };
            let def = MacroDefinition {
                name: name.clone(),
                kind: MacroKind::ObjectLike,
                replacement: vec![MacroToken::Text(replacement_text)],
                is_builtin: false,
                location: None,
            };
            macro_table.define(def);
        }

        // Apply -U flags directly to the macro table as well, so that -U
        // can target predefined macros that were not in the -D set.
        for name in &options.undefines {
            macro_table.undefine(name);
        }

        Preprocessor {
            macro_table,
            conditional_stack,
            include_resolver,
            options,
            file_content_cache: HashMap::new(),
            include_depth: 0,
        }
    }

    // =======================================================================
    // Public API — process()
    // =======================================================================

    /// Preprocesses a C source file, producing fully expanded output text.
    ///
    /// This is the main entry point called by the driver/frontend pipeline.
    /// It registers the source file with the `SourceMap`, processes all
    /// preprocessor directives, expands macros, resolves `#include` directives
    /// recursively, and returns the fully preprocessed source as a `String`.
    ///
    /// # Arguments
    ///
    /// * `source` — The raw C source text to preprocess.
    /// * `file_path` — The path of the source file (used for diagnostics and
    ///   `__FILE__` expansion).
    /// * `source_map` — Mutable reference to the source file registry.
    /// * `diagnostics` — Mutable reference to the diagnostic emitter.
    ///
    /// # Returns
    ///
    /// `Ok(preprocessed_text)` on success, or `Err(())` if any errors were
    /// emitted during preprocessing. Errors are already reported to stderr
    /// via the diagnostic emitter.
    pub fn process(
        &mut self,
        source: &str,
        file_path: &Path,
        source_map: &mut SourceMap,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<String, ()> {
        // Create a PreprocessorContext by temporarily moving our internal
        // fields and the caller's source_map/diagnostics into the owned
        // context structure required by process_directive(). We restore
        // everything after processing completes.
        let mut ctx = PreprocessorContext::new(
            std::mem::replace(&mut self.macro_table, MacroTable::new()),
            std::mem::replace(&mut self.conditional_stack, ConditionalStack::new()),
            std::mem::replace(
                &mut self.include_resolver,
                IncludeResolver::new(vec![], None, vec![]),
            ),
            std::mem::replace(source_map, SourceMap::new()),
            std::mem::replace(diagnostics, DiagnosticEmitter::new()),
        );

        let result = Self::process_source_impl(
            source,
            file_path,
            &mut ctx,
            &mut self.file_content_cache,
            &mut self.include_depth,
        );

        // Restore all fields from the context back to their owners.
        self.macro_table = ctx.macro_table;
        self.conditional_stack = ctx.conditional_stack;
        self.include_resolver = ctx.include_resolver;
        *source_map = ctx.source_map;
        *diagnostics = ctx.diagnostics;

        result
    }

    // =======================================================================
    // Internal — source processing implementation
    // =======================================================================

    /// Core implementation of source preprocessing. Operates on the
    /// `PreprocessorContext` which owns all mutable state.
    fn process_source_impl(
        source: &str,
        file_path: &Path,
        ctx: &mut PreprocessorContext,
        cache: &mut HashMap<PathBuf, String>,
        include_depth: &mut usize,
    ) -> Result<String, ()> {
        // Register the source file in the source map.
        let file_id = ctx
            .source_map
            .add_file(file_path.to_path_buf(), source.to_string());
        ctx.diagnostics
            .register_file(file_id, &file_path.display().to_string());

        // Phase 1: Replace trigraph sequences (C11 §5.2.1.1).
        let after_trigraphs = Self::replace_trigraphs(source);

        // Phase 2: Splice physical lines joined by backslash-newline (C11 §5.1.1.2p2).
        let after_continuations = Self::join_line_continuations(&after_trigraphs);

        // Phase 3: Strip comments, replacing them with single spaces (C11 §5.1.1.2p3).
        let after_comments = Self::strip_comments(&after_continuations);

        // Split into logical lines for line-by-line processing.
        let lines: Vec<&str> = after_comments.split('\n').collect();

        // Pre-allocate output buffer with generous capacity.
        let mut output = String::with_capacity(source.len() + source.len() / 8);

        // Process each line.
        Self::process_lines(
            &lines,
            ctx,
            file_id,
            file_path,
            &mut output,
            cache,
            include_depth,
        )?;

        // Check for unterminated conditionals at end of file.
        // Only check at the top level (include_depth == 0) because
        // conditionals can legally span across include boundaries.
        if *include_depth == 0 {
            ctx.conditional_stack
                .check_unterminated(&mut ctx.diagnostics)?;
        }

        if ctx.diagnostics.has_errors() {
            Err(())
        } else {
            Ok(output)
        }
    }

    // =======================================================================
    // Internal — line-by-line processing loop
    // =======================================================================

    /// Processes a sequence of logical source lines, handling directives,
    /// macro expansion, and conditional compilation.
    ///
    /// For each line:
    /// 1. Update dynamic macros (`__FILE__`, `__LINE__`).
    /// 2. Check if the line is a preprocessor directive.
    /// 3. If directive: dispatch to [`process_directive()`] and handle the
    ///    resulting action.
    /// 4. If not a directive and in an active conditional branch: expand macros
    ///    and append to output.
    /// 5. If not a directive and in an inactive branch: emit an empty line to
    ///    preserve line numbering for downstream phases.
    fn process_lines(
        lines: &[&str],
        ctx: &mut PreprocessorContext,
        file_id: FileId,
        file_path: &Path,
        output: &mut String,
        cache: &mut HashMap<PathBuf, String>,
        include_depth: &mut usize,
    ) -> Result<(), ()> {
        let mut had_error = false;

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = (line_idx + 1) as u32;
            let location = SourceLocation {
                file_id,
                byte_offset: 0,
                line: line_num,
                column: 1,
            };

            // Update __FILE__ and __LINE__ before processing this line so
            // that macro expansion produces correct values.
            Self::update_dynamic_macros(ctx, file_path, line_num);

            // Attempt to classify this line as a preprocessor directive.
            if let Some((kind, rest)) = parse_directive(line) {
                // Known directive. process_directive() already handles the
                // conditional-activity check: it skips non-conditional
                // directives when inside an inactive block.
                match process_directive(kind, rest, ctx, location) {
                    Ok(action) => match action {
                        DirectiveAction::None => {
                            // Directive consumed, no output. Emit newline to
                            // preserve line count for downstream phases.
                            output.push('\n');
                        }
                        DirectiveAction::IncludeFile(path) => {
                            // Recursively process the included file.
                            if let Err(()) = Self::handle_include(
                                &path,
                                ctx,
                                output,
                                cache,
                                include_depth,
                            ) {
                                had_error = true;
                            }
                            output.push('\n');
                        }
                        DirectiveAction::EmitText(text) => {
                            output.push_str(&text);
                            output.push('\n');
                        }
                        DirectiveAction::Error => {
                            had_error = true;
                            output.push('\n');
                        }
                    },
                    Err(()) => {
                        had_error = true;
                        output.push('\n');
                    }
                }
            } else {
                // Not a recognized directive. Check if it starts with '#'
                // (unknown directive) vs. a regular source line.
                let trimmed = line.trim_start();
                if trimmed.starts_with('#') && ctx.conditional_stack.is_active() {
                    // Active block with unknown directive — warn.
                    ctx.diagnostics.warning(
                        location,
                        "unknown preprocessing directive".to_string(),
                    );
                    output.push('\n');
                } else if ctx.conditional_stack.is_active() {
                    // Regular source line in an active block — expand macros.
                    let mut expansion_guard = HashSet::new();
                    let expanded =
                        ctx.macro_table.expand_line(line, &mut expansion_guard);
                    output.push_str(&expanded);
                    output.push('\n');
                } else {
                    // Inactive conditional block — skip line, emit newline.
                    output.push('\n');
                }
            }
        }

        if had_error {
            Err(())
        } else {
            Ok(())
        }
    }

    // =======================================================================
    // Internal — include file handling
    // =======================================================================

    /// Handles a `DirectiveAction::IncludeFile` by reading and recursively
    /// preprocessing the included file.
    ///
    /// Steps:
    /// 1. Enforce the maximum include nesting depth.
    /// 2. Push the file onto the include stack (checks circular includes
    ///    and `#pragma once`).
    /// 3. Read the file content (using the cache when available).
    /// 4. Register the file in the source map.
    /// 5. Recursively preprocess the included content.
    /// 6. Pop the file from the include stack.
    fn handle_include(
        path: &Path,
        ctx: &mut PreprocessorContext,
        output: &mut String,
        cache: &mut HashMap<PathBuf, String>,
        include_depth: &mut usize,
    ) -> Result<(), ()> {
        // Enforce maximum include depth.
        if *include_depth >= MAX_INCLUDE_DEPTH {
            ctx.diagnostics.error(
                SourceLocation::dummy(),
                format!(
                    "#include nested too deeply (limit is {} levels)",
                    MAX_INCLUDE_DEPTH
                ),
            );
            return Err(());
        }

        // Push onto include stack (checks circular includes and pragma once).
        match ctx.include_resolver.push_include(path) {
            Ok(()) => { /* proceed */ }
            Err(IncludeError::PragmaOnce(_)) => {
                // File already included via #pragma once — silently skip.
                return Ok(());
            }
            Err(IncludeError::CircularInclude(p)) => {
                ctx.diagnostics.error(
                    SourceLocation::dummy(),
                    format!("#include cycle detected: '{}'", p.display()),
                );
                return Err(());
            }
            Err(e) => {
                ctx.diagnostics.error(
                    SourceLocation::dummy(),
                    format!("{}", e),
                );
                return Err(());
            }
        }

        // Read file content, using the cache to avoid redundant disk reads.
        let path_buf = path.to_path_buf();
        let content = if cache.contains_key(&path_buf) {
            cache.get(&path_buf).unwrap().clone()
        } else {
            match ctx.include_resolver.read_file(path) {
                Ok(data) => {
                    cache.insert(path_buf.clone(), data.clone());
                    data
                }
                Err(e) => {
                    ctx.include_resolver.pop_include();
                    ctx.diagnostics.error(
                        SourceLocation::dummy(),
                        format!("error reading '{}': {}", path.display(), e),
                    );
                    // Remove any stale cache entry for this path.
                    cache.remove(&path_buf);
                    return Err(());
                }
            }
        };

        // Register the file in the source map.
        let file_id = ctx
            .source_map
            .add_file(path.to_path_buf(), content.clone());
        ctx.diagnostics
            .register_file(file_id, &path.display().to_string());

        // Pre-process the included text (trigraphs, continuations, comments).
        let processed = Self::preprocess_text(&content);
        let lines: Vec<&str> = processed.split('\n').collect();

        // Increment include depth and recursively process.
        *include_depth += 1;
        let result = Self::process_lines(
            &lines, ctx, file_id, path, output, cache, include_depth,
        );
        *include_depth -= 1;

        // Pop from include stack regardless of result.
        ctx.include_resolver.pop_include();

        result
    }

    // =======================================================================
    // Internal — predefined macro setup
    // =======================================================================

    /// Sets up the standard C11 predefined macros and GCC compatibility macros.
    fn setup_predefined_macros(macro_table: &mut MacroTable) {
        let define_simple = |table: &mut MacroTable, name: &str, value: &str| {
            let def = MacroDefinition {
                name: name.to_string(),
                kind: MacroKind::ObjectLike,
                replacement: vec![MacroToken::Text(value.to_string())],
                is_builtin: true,
                location: None,
            };
            table.define(def);
        };

        // C11 standard predefined macros (§6.10.8)
        define_simple(macro_table, "__STDC__", "1");
        define_simple(macro_table, "__STDC_VERSION__", "201112L");
        define_simple(macro_table, "__STDC_HOSTED__", "1");

        // GCC compatibility macros for real-world codebase compatibility
        define_simple(macro_table, "__GNUC__", "4");
        define_simple(macro_table, "__GNUC_MINOR__", "2");
        define_simple(macro_table, "__GNUC_PATCHLEVEL__", "1");

        // Additional GCC/Linux compatibility macros
        define_simple(macro_table, "__ELF__", "1");
        define_simple(macro_table, "__linux__", "1");
        define_simple(macro_table, "__linux", "1");
        define_simple(macro_table, "linux", "1");
        define_simple(macro_table, "__unix__", "1");
        define_simple(macro_table, "__unix", "1");
        define_simple(macro_table, "unix", "1");
        define_simple(macro_table, "__gnu_linux__", "1");

        // Dynamic macros — initial values, updated per-line.
        define_simple(macro_table, "__FILE__", "\"\"");
        define_simple(macro_table, "__LINE__", "0");

        // Date and time macros — computed once at preprocessor construction.
        let (date_str, time_str) = Self::compute_date_time();
        define_simple(macro_table, "__DATE__", &format!("\"{}\"", date_str));
        define_simple(macro_table, "__TIME__", &format!("\"{}\"", time_str));
    }

    /// Computes the `__DATE__` and `__TIME__` strings from the current system time.
    fn compute_date_time() -> (String, String) {
        use std::time::SystemTime;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if now == 0 {
            return ("Jan  1 1970".to_string(), "00:00:00".to_string());
        }

        let secs_per_day: u64 = 86400;
        let secs_per_hour: u64 = 3600;
        let secs_per_min: u64 = 60;

        let time_of_day = now % secs_per_day;
        let hours = time_of_day / secs_per_hour;
        let minutes = (time_of_day % secs_per_hour) / secs_per_min;
        let seconds = time_of_day % secs_per_min;

        let mut days = (now / secs_per_day) as i64;
        let mut year: i64 = 1970;

        loop {
            let days_in_year = if Self::is_leap_year(year) { 366 } else { 365 };
            if days < days_in_year {
                break;
            }
            days -= days_in_year;
            year += 1;
        }

        let month_days = if Self::is_leap_year(year) {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };

        let mut month = 0usize;
        for (i, &md) in month_days.iter().enumerate() {
            if days < md as i64 {
                month = i;
                break;
            }
            days -= md as i64;
        }
        let day = days + 1;

        let month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];

        let date_str = format!("{} {:2} {}", month_names[month], day, year);
        let time_str = format!("{:02}:{:02}:{:02}", hours, minutes, seconds);

        (date_str, time_str)
    }

    /// Returns `true` if the given year is a leap year.
    fn is_leap_year(year: i64) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    // =======================================================================
    // Internal — dynamic macro updates
    // =======================================================================

    /// Updates `__FILE__` and `__LINE__` macros to reflect the current position.
    fn update_dynamic_macros(
        ctx: &mut PreprocessorContext,
        file_path: &Path,
        line_num: u32,
    ) {
        let file_display = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");
        let file_def = MacroDefinition {
            name: "__FILE__".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text(format!("\"{}\"", file_display))],
            is_builtin: true,
            location: None,
        };
        ctx.macro_table.define(file_def);

        let line_def = MacroDefinition {
            name: "__LINE__".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text(line_num.to_string())],
            is_builtin: true,
            location: None,
        };
        ctx.macro_table.define(line_def);
    }

    // =======================================================================
    // Internal — text preprocessing (trigraphs, continuations, comments)
    // =======================================================================

    /// Applies all pre-lexing text transformations.
    fn preprocess_text(source: &str) -> String {
        let after_trigraphs = Self::replace_trigraphs(source);
        let after_continuations = Self::join_line_continuations(&after_trigraphs);
        Self::strip_comments(&after_continuations)
    }

    /// Replaces C11 trigraph sequences (§5.2.1.1).
    fn replace_trigraphs(source: &str) -> String {
        let bytes = source.as_bytes();
        let len = bytes.len();
        if len < 3 {
            return source.to_string();
        }

        let mut result = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            if i + 2 < len && bytes[i] == b'?' && bytes[i + 1] == b'?' {
                let replacement = match bytes[i + 2] {
                    b'=' => Some('#'),
                    b'(' => Some('['),
                    b')' => Some(']'),
                    b'/' => Some('\\'),
                    b'<' => Some('{'),
                    b'>' => Some('}'),
                    b'!' => Some('|'),
                    b'\'' => Some('^'),
                    b'-' => Some('~'),
                    _ => None,
                };
                if let Some(ch) = replacement {
                    result.push(ch);
                    i += 3;
                    continue;
                }
            }
            result.push(bytes[i] as char);
            i += 1;
        }

        result
    }

    /// Joins physical source lines connected by backslash-newline (C11 §5.1.1.2p2).
    fn join_line_continuations(source: &str) -> String {
        let bytes = source.as_bytes();
        let len = bytes.len();
        let mut result = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            if bytes[i] == b'\\' && i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
            } else if bytes[i] == b'\\' && i + 2 < len
                && bytes[i + 1] == b'\r' && bytes[i + 2] == b'\n'
            {
                i += 3;
            } else {
                result.push(bytes[i] as char);
                i += 1;
            }
        }

        result
    }

    /// Strips C-style comments from source text (C11 §5.1.1.2p3).
    fn strip_comments(source: &str) -> String {
        let bytes = source.as_bytes();
        let len = bytes.len();
        let mut result = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            // String literal — preserve everything.
            if bytes[i] == b'"' {
                result.push('"');
                i += 1;
                while i < len && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < len {
                        result.push(bytes[i] as char);
                        result.push(bytes[i + 1] as char);
                        i += 2;
                    } else {
                        result.push(bytes[i] as char);
                        i += 1;
                    }
                }
                if i < len {
                    result.push('"');
                    i += 1;
                }
                continue;
            }

            // Character literal — preserve everything.
            if bytes[i] == b'\'' {
                result.push('\'');
                i += 1;
                while i < len && bytes[i] != b'\'' {
                    if bytes[i] == b'\\' && i + 1 < len {
                        result.push(bytes[i] as char);
                        result.push(bytes[i + 1] as char);
                        i += 2;
                    } else {
                        result.push(bytes[i] as char);
                        i += 1;
                    }
                }
                if i < len {
                    result.push('\'');
                    i += 1;
                }
                continue;
            }

            // Block comment: /* ... */
            if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                i += 2;
                let mut found_end = false;
                while i + 1 < len {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        found_end = true;
                        break;
                    }
                    if bytes[i] == b'\n' {
                        result.push('\n');
                    }
                    i += 1;
                }
                if !found_end {
                    i = len;
                }
                result.push(' ');
                continue;
            }

            // Line comment: // ...
            if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                i += 2;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                result.push(' ');
                continue;
            }

            result.push(bytes[i] as char);
            i += 1;
        }

        result
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn make_preprocessor() -> (Preprocessor, SourceMap, DiagnosticEmitter) {
        let mut source_map = SourceMap::new();
        let mut diagnostics = DiagnosticEmitter::new();
        let options = PreprocessorOptions::new();
        let pp = Preprocessor::new(options, &mut source_map, &mut diagnostics);
        (pp, source_map, diagnostics)
    }

    fn make_preprocessor_with_defines(
        defines: Vec<(String, Option<String>)>,
    ) -> (Preprocessor, SourceMap, DiagnosticEmitter) {
        let mut source_map = SourceMap::new();
        let mut diagnostics = DiagnosticEmitter::new();
        let mut options = PreprocessorOptions::new();
        options.defines = defines;
        let pp = Preprocessor::new(options, &mut source_map, &mut diagnostics);
        (pp, source_map, diagnostics)
    }

    fn make_preprocessor_with_undefines(
        defines: Vec<(String, Option<String>)>,
        undefines: Vec<String>,
    ) -> (Preprocessor, SourceMap, DiagnosticEmitter) {
        let mut source_map = SourceMap::new();
        let mut diagnostics = DiagnosticEmitter::new();
        let mut options = PreprocessorOptions::new();
        options.defines = defines;
        options.undefines = undefines;
        let pp = Preprocessor::new(options, &mut source_map, &mut diagnostics);
        (pp, source_map, diagnostics)
    }

    fn preprocess(
        pp: &mut Preprocessor,
        source: &str,
        source_map: &mut SourceMap,
        diagnostics: &mut DiagnosticEmitter,
    ) -> Result<String, ()> {
        pp.process(source, Path::new("test.c"), source_map, diagnostics)
    }

    fn create_temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bcc_pp_test_{}", prefix));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn create_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dirs");
        }
        let mut f = fs::File::create(&path).expect("create file");
        f.write_all(content.as_bytes()).expect("write file");
        path
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_empty_source() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let result = preprocess(&mut pp, "", &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().trim().is_empty());
    }

    #[test]
    fn test_passthrough_no_directives() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "int main() {\n    return 0;\n}\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("int main()"));
        assert!(output.contains("return 0;"));
    }

    #[test]
    fn test_stdc_macro() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "int x = __STDC__;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("1"));
    }

    #[test]
    fn test_stdc_version_macro() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "long v = __STDC_VERSION__;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("201112L"));
    }

    #[test]
    fn test_define_option_with_value() {
        let (mut pp, mut sm, mut diag) = make_preprocessor_with_defines(vec![
            ("FOO".to_string(), Some("42".to_string())),
        ]);
        let source = "int x = FOO;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("42"));
    }

    #[test]
    fn test_define_option_without_value() {
        let (mut pp, mut sm, mut diag) = make_preprocessor_with_defines(vec![
            ("BAR".to_string(), None),
        ]);
        let source = "int x = BAR;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("1"));
    }

    #[test]
    fn test_undefine_option() {
        let (mut pp, mut sm, mut diag) = make_preprocessor_with_undefines(
            vec![("FOO".to_string(), Some("42".to_string()))],
            vec!["FOO".to_string()],
        );
        let source = "int x = FOO;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("FOO"));
    }

    #[test]
    fn test_undefine_predefined_macro() {
        let (mut pp, mut sm, mut diag) = make_preprocessor_with_undefines(
            vec![],
            vec!["__GNUC__".to_string()],
        );
        let source = "int x = __GNUC__;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("__GNUC__"));
    }

    #[test]
    fn test_if_1_active() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "#if 1\nACTIVE\n#endif\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("ACTIVE"));
    }

    #[test]
    fn test_if_0_inactive() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "#if 0\nINACTIVE\n#endif\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(!result.unwrap().contains("INACTIVE"));
    }

    #[test]
    fn test_nested_conditionals() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "#if 1\nOUTER\n#if 0\nINNER_SKIP\n#endif\nAFTER\n#endif\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("OUTER"));
        assert!(!output.contains("INNER_SKIP"));
        assert!(output.contains("AFTER"));
    }

    #[test]
    fn test_line_continuation() {
        let result = Preprocessor::join_line_continuations("hello \\\nworld\n");
        assert_eq!(result, "hello world\n");
    }

    #[test]
    fn test_line_continuation_in_directive() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "#define LONG_MACRO \\\n    42\nint x = LONG_MACRO;\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("42"));
    }

    #[test]
    fn test_strip_line_comment() {
        let result = Preprocessor::strip_comments("int x = 0; // comment\n");
        assert!(result.contains("int x = 0;"));
        assert!(!result.contains("comment"));
    }

    #[test]
    fn test_strip_block_comment() {
        let result = Preprocessor::strip_comments("int x = /* value */ 0;\n");
        assert!(result.contains("int x ="));
        assert!(result.contains("0;"));
        assert!(!result.contains("value"));
    }

    #[test]
    fn test_multiline_block_comment_preserves_lines() {
        let result = Preprocessor::strip_comments("a\n/* multi\nline\ncomment */b\n");
        let newline_count = result.chars().filter(|&c| c == '\n').count();
        assert!(newline_count >= 3);
    }

    #[test]
    fn test_comment_in_string_literal() {
        let result = Preprocessor::strip_comments("char *s = \"hello // world\";\n");
        assert!(result.contains("hello // world"));
    }

    #[test]
    fn test_trigraph_replacement() {
        assert_eq!(Preprocessor::replace_trigraphs("??="), "#");
        assert_eq!(Preprocessor::replace_trigraphs("??("), "[");
        assert_eq!(Preprocessor::replace_trigraphs("??)"), "]");
        assert_eq!(Preprocessor::replace_trigraphs("??>"), "}");
        assert_eq!(Preprocessor::replace_trigraphs("??<"), "{");
        assert_eq!(Preprocessor::replace_trigraphs("??!"), "|");
        assert_eq!(Preprocessor::replace_trigraphs("??'"), "^");
        assert_eq!(Preprocessor::replace_trigraphs("??-"), "~");
        assert_eq!(Preprocessor::replace_trigraphs("??/"), "\\");
    }

    #[test]
    fn test_trigraph_no_replacement() {
        assert_eq!(Preprocessor::replace_trigraphs("??x"), "??x");
        assert_eq!(Preprocessor::replace_trigraphs("?"), "?");
        assert_eq!(Preprocessor::replace_trigraphs(""), "");
    }

    #[test]
    fn test_include_basic() {
        let dir = create_temp_dir("include_basic");
        create_file(&dir, "header.h", "int from_header;\n");

        let mut source_map = SourceMap::new();
        let mut diagnostics = DiagnosticEmitter::new();
        let mut options = PreprocessorOptions::new();
        options.include_dirs.push(dir.clone());
        let mut pp = Preprocessor::new(options, &mut source_map, &mut diagnostics);

        let source = "#include \"header.h\"\nint from_main;\n";
        let result = pp.process(
            source,
            &dir.join("main.c"),
            &mut source_map,
            &mut diagnostics,
        );
        assert!(result.is_ok(), "Include failed: errors={}", diagnostics.error_count());
        let output = result.unwrap();
        assert!(output.contains("from_header"));
        assert!(output.contains("from_main"));

        cleanup(&dir);
    }

    #[test]
    fn test_unterminated_conditional() {
        let (mut pp, mut sm, mut diag) = make_preprocessor();
        let source = "#if 1\nsome code\n";
        let result = preprocess(&mut pp, source, &mut sm, &mut diag);
        assert!(result.is_err());
        assert!(diag.has_errors());
    }

    #[test]
    fn test_preprocessor_options_default() {
        let opts = PreprocessorOptions::default();
        assert!(opts.include_dirs.is_empty());
        assert!(opts.defines.is_empty());
        assert!(opts.undefines.is_empty());
        assert!(opts.bundled_header_path.is_none());
        assert!(opts.system_include_dirs.is_empty());
    }

    #[test]
    fn test_date_time_format() {
        let (date, time) = Preprocessor::compute_date_time();
        assert_eq!(date.len(), 11, "Date format: '{}'", date);
        assert_eq!(time.len(), 8, "Time format: '{}'", time);
        assert_eq!(time.as_bytes()[2], b':');
        assert_eq!(time.as_bytes()[5], b':');
    }

    #[test]
    fn test_leap_year() {
        assert!(Preprocessor::is_leap_year(2000));
        assert!(Preprocessor::is_leap_year(2024));
        assert!(!Preprocessor::is_leap_year(1900));
        assert!(!Preprocessor::is_leap_year(2023));
    }
}
