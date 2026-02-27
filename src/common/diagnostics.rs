//! GCC-compatible error, warning, and note reporting for the `bcc` compiler.
//!
//! This module provides the cross-cutting diagnostic subsystem used by every
//! pipeline phase — preprocessor, lexer, parser, semantic analyzer, IR generator,
//! optimizer, code generator, and linker — to report errors, warnings, and notes
//! in the standard GCC-compatible format on stderr.
//!
//! # Output Format
//!
//! All diagnostics follow the GCC format:
//!
//! ```text
//! file.c:10:5: error: undeclared identifier 'foo'
//! file.c:10:5: note: did you mean 'bar'?
//! file.c:20:1: warning: unused variable 'z'
//! ```
//!
//! When a diagnostic has no associated source location (e.g., CLI errors), the
//! format omits the file/line/column prefix:
//!
//! ```text
//! error: no input files
//! ```
//!
//! # Exit Code Contract
//!
//! The compiler must exit with code 1 on any compile error. Warnings are
//! non-blocking. The [`DiagnosticEmitter::has_errors`] method reports whether
//! any errors were recorded, enabling the driver to decide the exit code.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use std::fmt;
use std::io::{self, Write};

use super::source_map::{FileId, SourceLocation, SourceMap};

// ---------------------------------------------------------------------------
// Severity — diagnostic severity level
// ---------------------------------------------------------------------------

/// The severity level of a diagnostic message.
///
/// GCC-compatible diagnostics distinguish three severity levels:
/// - [`Severity::Error`] — A hard error that prevents successful compilation.
///   Any error causes the compiler to exit with code 1.
/// - [`Severity::Warning`] — A suspicious but non-blocking condition. The
///   compilation continues and succeeds unless errors are also present.
/// - [`Severity::Note`] — Supplementary information attached to a preceding
///   error or warning to provide additional context (e.g., "did you mean...?").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A hard compilation error. Any error prevents successful compilation.
    Error,
    /// A non-blocking warning about suspicious code.
    Warning,
    /// A supplementary note attached to a preceding diagnostic.
    Note,
}

impl fmt::Display for Severity {
    /// Formats the severity as a lowercase string matching GCC output:
    /// `"error"`, `"warning"`, or `"note"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostic — a single diagnostic message
// ---------------------------------------------------------------------------

/// A single diagnostic message with severity, optional source location,
/// human-readable description, and optional attached sub-diagnostics (notes).
///
/// The [`Display`] implementation produces GCC-compatible formatted output:
///
/// ```text
/// file.c:10:5: error: undeclared identifier 'foo'
/// ```
///
/// or, without a source location:
///
/// ```text
/// error: no input files
/// ```
///
/// # Fields
///
/// - `severity` — The diagnostic severity (error, warning, note).
/// - `location` — An optional [`SourceLocation`] pinpointing the source position.
///   `None` for global/CLI diagnostics (e.g., "no input files").
/// - `message` — Human-readable description of the issue.
/// - `notes` — Zero or more sub-diagnostics providing additional context
///   (typically with [`Severity::Note`]).
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// The severity level of this diagnostic.
    pub severity: Severity,
    /// The source location where the issue was detected, or `None` for
    /// diagnostics without a source position (CLI errors, global issues).
    pub location: Option<SourceLocation>,
    /// Human-readable description of the issue.
    pub message: String,
    /// Attached sub-diagnostics (typically notes) providing extra context.
    /// For example, an error about a redeclaration might have a note pointing
    /// to the previous declaration site.
    pub notes: Vec<Diagnostic>,
    /// Resolved file path string for display formatting. Populated by
    /// [`DiagnosticEmitter`] when the diagnostic is created through its
    /// convenience methods. When empty and a location is present, the
    /// formatter falls back to `"<unknown>"`.
    file_name: String,
}

impl Diagnostic {
    /// Creates a new diagnostic with the given severity, location, and message.
    ///
    /// The `file_name` is provided by the caller (typically [`DiagnosticEmitter`])
    /// after resolving the [`FileId`] from the source location.
    fn new_with_file(
        severity: Severity,
        location: Option<SourceLocation>,
        message: String,
        file_name: String,
    ) -> Self {
        Diagnostic {
            severity,
            location,
            message,
            notes: Vec::new(),
            file_name,
        }
    }

    /// Creates a new diagnostic without a resolved file path.
    ///
    /// Useful for diagnostics created outside of [`DiagnosticEmitter`] or for
    /// diagnostics without a source location.
    pub fn new(severity: Severity, location: Option<SourceLocation>, message: String) -> Self {
        let file_name = String::new();
        Diagnostic {
            severity,
            location,
            message,
            notes: Vec::new(),
            file_name,
        }
    }

    /// Attaches a sub-diagnostic (note) to this diagnostic.
    ///
    /// Returns `self` for builder-style chaining:
    /// ```ignore
    /// let diag = Diagnostic::new(Severity::Error, Some(loc), "msg".into())
    ///     .with_note(note_diag);
    /// ```
    pub fn with_note(mut self, note: Diagnostic) -> Self {
        self.notes.push(note);
        self
    }

    /// Sets the resolved file name for display formatting.
    ///
    /// Called internally by [`DiagnosticEmitter`] when it can resolve the
    /// [`FileId`] from a [`SourceMap`].
    fn set_file_name(&mut self, name: String) {
        self.file_name = name;
    }
}

impl fmt::Display for Diagnostic {
    /// Formats the diagnostic in GCC-compatible format.
    ///
    /// **With source location** (non-dummy, file name resolved):
    /// ```text
    /// hello.c:10:5: error: undeclared identifier 'x'
    /// ```
    ///
    /// **Without source location** (or dummy location):
    /// ```text
    /// error: no input files
    /// ```
    ///
    /// Line and column numbers are 1-based, matching GCC conventions.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.location {
            Some(ref loc) if !loc.is_dummy() => {
                // Resolve the file name: use the stored name if available,
                // fall back to "<unknown>" for unresolved FileIds.
                let file_display = if self.file_name.is_empty() {
                    "<unknown>"
                } else {
                    &self.file_name
                };
                write!(
                    f,
                    "{}:{}:{}: {}: {}",
                    file_display, loc.line, loc.column, self.severity, self.message
                )
            }
            _ => {
                // No location or dummy location — omit file:line:col prefix.
                write!(f, "{}: {}", self.severity, self.message)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticEmitter — central diagnostic accumulator and stderr writer
// ---------------------------------------------------------------------------

/// The central diagnostic accumulator and stderr writer for the `bcc` compiler.
///
/// `DiagnosticEmitter` is typically created once in `main.rs` and passed (by
/// mutable reference) through every pipeline phase. It accumulates all
/// diagnostics emitted during compilation, tracks error and warning counts,
/// and writes each diagnostic to stderr in GCC-compatible format as soon as
/// it is emitted.
///
/// # Usage
///
/// ```ignore
/// let mut emitter = DiagnosticEmitter::new();
///
/// // Register files so FileId can be resolved to paths
/// emitter.register_file(file_id, "hello.c");
///
/// // Emit diagnostics during compilation
/// emitter.error(loc, "undeclared identifier 'x'");
/// emitter.warning(loc2, "unused variable 'z'");
///
/// // Check for errors before proceeding
/// if emitter.has_errors() {
///     std::process::exit(1);
/// }
/// ```
///
/// # File Path Resolution
///
/// [`SourceLocation`] carries a [`FileId`] rather than a file path string.
/// To produce the `file:line:col:` prefix in GCC-compatible output, the
/// emitter maintains an internal file path registry. Paths are registered
/// via [`register_file`](DiagnosticEmitter::register_file) or synced in
/// bulk from a [`SourceMap`] via
/// [`sync_source_map`](DiagnosticEmitter::sync_source_map).
pub struct DiagnosticEmitter {
    /// All diagnostics accumulated during compilation, in emission order.
    diagnostics: Vec<Diagnostic>,
    /// Number of [`Severity::Error`] diagnostics emitted.
    error_count: usize,
    /// Number of [`Severity::Warning`] diagnostics emitted.
    warning_count: usize,
    /// File path lookup table indexed by `FileId.index()`. Each entry holds
    /// the file path string corresponding to that `FileId`. Entries are
    /// registered via [`register_file`] or [`sync_source_map`].
    file_paths: Vec<String>,
}

impl DiagnosticEmitter {
    // -- Construction -------------------------------------------------------

    /// Creates a new, empty `DiagnosticEmitter` with zero diagnostics and
    /// an empty file path registry.
    pub fn new() -> Self {
        DiagnosticEmitter {
            diagnostics: Vec::new(),
            error_count: 0,
            warning_count: 0,
            file_paths: Vec::new(),
        }
    }

    // -- File Path Registration ---------------------------------------------

    /// Registers a file path string for a given [`FileId`], enabling the
    /// emitter to resolve `FileId` values to human-readable file paths when
    /// formatting diagnostics.
    ///
    /// This should be called whenever a new source file is added to the
    /// [`SourceMap`], typically in the preprocessor when processing `#include`
    /// directives.
    ///
    /// If the `FileId` index exceeds the current registry capacity, the
    /// registry is extended with empty strings to accommodate it.
    ///
    /// # Arguments
    /// - `file_id` — The file identifier to register.
    /// - `path` — The file path string (relative or absolute, as provided to
    ///   the compiler).
    pub fn register_file(&mut self, file_id: FileId, path: &str) {
        let idx = file_id.index() as usize;
        // Extend the registry if needed to accommodate the new FileId.
        if idx >= self.file_paths.len() {
            self.file_paths.resize(idx + 1, String::new());
        }
        self.file_paths[idx] = path.to_string();
    }

    /// Synchronizes the file path registry with all files currently registered
    /// in the given [`SourceMap`].
    ///
    /// This bulk method iterates all files in the source map and registers
    /// their paths, ensuring the emitter can resolve any `FileId` produced
    /// by the source map. It is safe to call multiple times; each call
    /// refreshes the registry from scratch.
    ///
    /// # Arguments
    /// - `source_map` — The source map to synchronize from.
    pub fn sync_source_map(&mut self, source_map: &SourceMap) {
        let count = source_map.file_count();
        self.file_paths.clear();
        self.file_paths.reserve(count);
        for i in 0..count {
            let file_id = FileId(i as u32);
            let path = source_map.get_file_path(file_id);
            self.file_paths.push(path.display().to_string());
        }
    }

    // -- File Path Resolution (internal) ------------------------------------

    /// Resolves a [`FileId`] to its registered file path string.
    ///
    /// Returns `"<unknown>"` if the `FileId` is not registered or its path
    /// is empty.
    fn resolve_file_path(&self, file_id: FileId) -> &str {
        let idx = file_id.index() as usize;
        match self.file_paths.get(idx) {
            Some(path) if !path.is_empty() => path.as_str(),
            _ => "<unknown>",
        }
    }

    // -- Diagnostic Construction (internal) ---------------------------------

    /// Creates a [`Diagnostic`] with the file path resolved from the internal
    /// registry. This is used by all convenience emission methods.
    fn make_diagnostic(
        &self,
        severity: Severity,
        location: Option<SourceLocation>,
        message: String,
    ) -> Diagnostic {
        let file_name = match &location {
            Some(loc) if !loc.is_dummy() => self.resolve_file_path(loc.file_id).to_string(),
            _ => String::new(),
        };
        Diagnostic::new_with_file(severity, location, message, file_name)
    }

    // -- Core Emission ------------------------------------------------------

    /// Emits a diagnostic: writes it to stderr in GCC-compatible format,
    /// increments the appropriate counter, and stores it for later inspection.
    ///
    /// If the diagnostic carries attached notes, each note is also written
    /// to stderr immediately after the parent diagnostic. Notes do not
    /// increment the error or warning counts.
    ///
    /// # GCC-compatible Output
    ///
    /// Each diagnostic is written as a single line to stderr:
    /// ```text
    /// file.c:10:5: error: undeclared identifier 'x'
    /// ```
    ///
    /// # Arguments
    /// - `diagnostic` — The diagnostic to emit. Typically constructed via
    ///   the convenience methods ([`error`](Self::error),
    ///   [`warning`](Self::warning), etc.) or via [`Diagnostic::new`].
    pub fn emit(&mut self, diagnostic: Diagnostic) {
        // Write the diagnostic to stderr. We use writeln! on the stderr
        // lock for atomic line output and to avoid interleaving with other
        // threads (defense in depth, even though bcc is single-threaded).
        let stderr = io::stderr();
        let mut handle = stderr.lock();
        let _ = writeln!(handle, "{}", diagnostic);

        // Emit attached notes (recursively handles nested notes).
        for note in &diagnostic.notes {
            let _ = writeln!(handle, "{}", note);
        }

        // Update counters based on severity.
        match diagnostic.severity {
            Severity::Error => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            Severity::Note => { /* Notes do not increment counters. */ }
        }

        // Store the diagnostic for later inspection (e.g., by tests or by
        // the driver when summarizing results).
        self.diagnostics.push(diagnostic);
    }

    // -- Convenience Emission Methods ---------------------------------------

    /// Emits an error diagnostic at the given source location.
    ///
    /// This is the most common entry point for reporting compilation errors.
    /// The file path is resolved from the internal file path registry using
    /// the [`FileId`] embedded in the [`SourceLocation`].
    ///
    /// # Arguments
    /// - `location` — The source position where the error was detected.
    /// - `message` — A human-readable description of the error.
    ///
    /// # Example
    /// ```ignore
    /// emitter.error(loc, "undeclared identifier 'foo'");
    /// // stderr: hello.c:10:5: error: undeclared identifier 'foo'
    /// ```
    pub fn error(&mut self, location: SourceLocation, message: impl Into<String>) {
        let diag = self.make_diagnostic(Severity::Error, Some(location), message.into());
        self.emit(diag);
    }

    /// Emits a warning diagnostic at the given source location.
    ///
    /// Warnings are non-blocking: they do not cause the compiler to exit
    /// with a non-zero code unless accompanied by errors.
    ///
    /// # Arguments
    /// - `location` — The source position where the warning was detected.
    /// - `message` — A human-readable description of the warning.
    pub fn warning(&mut self, location: SourceLocation, message: impl Into<String>) {
        let diag = self.make_diagnostic(Severity::Warning, Some(location), message.into());
        self.emit(diag);
    }

    /// Emits a note diagnostic at the given source location.
    ///
    /// Notes provide supplementary information for a preceding error or
    /// warning. They do not increment any counter.
    ///
    /// # Arguments
    /// - `location` — The source position relevant to the note.
    /// - `message` — A human-readable supplementary message.
    pub fn note(&mut self, location: SourceLocation, message: impl Into<String>) {
        let diag = self.make_diagnostic(Severity::Note, Some(location), message.into());
        self.emit(diag);
    }

    /// Emits an error diagnostic without a source location.
    ///
    /// Used for errors that are not tied to a specific position in source
    /// code, such as CLI errors ("no input files"), linker errors, or
    /// internal compiler errors.
    ///
    /// # Arguments
    /// - `message` — A human-readable description of the error.
    ///
    /// # Example
    /// ```ignore
    /// emitter.error_no_loc("no input files");
    /// // stderr: error: no input files
    /// ```
    pub fn error_no_loc(&mut self, message: impl Into<String>) {
        let diag = self.make_diagnostic(Severity::Error, None, message.into());
        self.emit(diag);
    }

    // -- Query Methods ------------------------------------------------------

    /// Returns `true` if at least one error has been emitted.
    ///
    /// The driver should check this after each pipeline phase (or at the end)
    /// to determine whether to exit with code 1.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Returns the total number of error-severity diagnostics emitted so far.
    #[inline]
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Returns the total number of warning-severity diagnostics emitted so far.
    #[inline]
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    // -- Diagnostic Access --------------------------------------------------

    /// Returns a reference to all accumulated diagnostics in emission order.
    ///
    /// Useful for testing, post-compilation summaries, or IDE integration.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl Default for DiagnosticEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::{FileId, SourceLocation, SourceMap};
    use std::path::PathBuf;

    // -- Helper: create a SourceLocation for testing -----------------------

    fn test_location(file_id: FileId, line: u32, column: u32) -> SourceLocation {
        SourceLocation {
            file_id,
            byte_offset: 0,
            line,
            column,
        }
    }

    // -- Severity Display tests --------------------------------------------

    #[test]
    fn test_severity_display_error() {
        assert_eq!(format!("{}", Severity::Error), "error");
    }

    #[test]
    fn test_severity_display_warning() {
        assert_eq!(format!("{}", Severity::Warning), "warning");
    }

    #[test]
    fn test_severity_display_note() {
        assert_eq!(format!("{}", Severity::Note), "note");
    }

    #[test]
    fn test_severity_clone_copy() {
        let s = Severity::Error;
        let s2 = s; // Copy
        let s3 = s.clone(); // Clone
        assert_eq!(s, s2);
        assert_eq!(s, s3);
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Error, Severity::Warning);
        assert_ne!(Severity::Warning, Severity::Note);
    }

    // -- Diagnostic Display tests ------------------------------------------

    #[test]
    fn test_diagnostic_display_with_location() {
        let loc = test_location(FileId(0), 10, 5);
        let diag = Diagnostic::new_with_file(
            Severity::Error,
            Some(loc),
            "undeclared identifier 'x'".to_string(),
            "hello.c".to_string(),
        );
        assert_eq!(
            format!("{}", diag),
            "hello.c:10:5: error: undeclared identifier 'x'"
        );
    }

    #[test]
    fn test_diagnostic_display_warning_with_location() {
        let loc = test_location(FileId(0), 20, 1);
        let diag = Diagnostic::new_with_file(
            Severity::Warning,
            Some(loc),
            "unused variable 'z'".to_string(),
            "hello.c".to_string(),
        );
        assert_eq!(
            format!("{}", diag),
            "hello.c:20:1: warning: unused variable 'z'"
        );
    }

    #[test]
    fn test_diagnostic_display_note_with_location() {
        let loc = test_location(FileId(0), 10, 5);
        let diag = Diagnostic::new_with_file(
            Severity::Note,
            Some(loc),
            "did you mean 'y'?".to_string(),
            "hello.c".to_string(),
        );
        assert_eq!(format!("{}", diag), "hello.c:10:5: note: did you mean 'y'?");
    }

    #[test]
    fn test_diagnostic_display_without_location() {
        let diag = Diagnostic::new(Severity::Error, None, "no input files".to_string());
        assert_eq!(format!("{}", diag), "error: no input files");
    }

    #[test]
    fn test_diagnostic_display_with_dummy_location() {
        let loc = SourceLocation::dummy();
        let diag = Diagnostic::new_with_file(
            Severity::Error,
            Some(loc),
            "internal error".to_string(),
            String::new(),
        );
        // Dummy location → no file:line:col prefix
        assert_eq!(format!("{}", diag), "error: internal error");
    }

    #[test]
    fn test_diagnostic_display_with_unknown_file() {
        let loc = test_location(FileId(99), 5, 3);
        // Empty file name → falls back to <unknown>
        let diag = Diagnostic::new_with_file(
            Severity::Error,
            Some(loc),
            "something went wrong".to_string(),
            String::new(),
        );
        assert_eq!(
            format!("{}", diag),
            "<unknown>:5:3: error: something went wrong"
        );
    }

    #[test]
    fn test_diagnostic_with_notes() {
        let loc = test_location(FileId(0), 10, 5);
        let note_loc = test_location(FileId(0), 3, 1);

        let note = Diagnostic::new_with_file(
            Severity::Note,
            Some(note_loc),
            "previous declaration here".to_string(),
            "main.c".to_string(),
        );
        let diag = Diagnostic::new_with_file(
            Severity::Error,
            Some(loc),
            "redefinition of 'foo'".to_string(),
            "main.c".to_string(),
        )
        .with_note(note);

        assert_eq!(diag.notes.len(), 1);
        assert_eq!(
            format!("{}", diag),
            "main.c:10:5: error: redefinition of 'foo'"
        );
        assert_eq!(
            format!("{}", diag.notes[0]),
            "main.c:3:1: note: previous declaration here"
        );
    }

    // -- Diagnostic::new tests ---------------------------------------------

    #[test]
    fn test_diagnostic_new_no_location() {
        let diag = Diagnostic::new(Severity::Warning, None, "test warning".to_string());
        assert_eq!(diag.severity, Severity::Warning);
        assert!(diag.location.is_none());
        assert_eq!(diag.message, "test warning");
        assert!(diag.notes.is_empty());
    }

    #[test]
    fn test_diagnostic_new_with_location() {
        let loc = test_location(FileId(0), 1, 1);
        let diag = Diagnostic::new(Severity::Error, Some(loc), "test error".to_string());
        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.location.is_some());
        assert_eq!(diag.message, "test error");
    }

    // -- DiagnosticEmitter construction ------------------------------------

    #[test]
    fn test_emitter_new_is_empty() {
        let emitter = DiagnosticEmitter::new();
        assert!(!emitter.has_errors());
        assert_eq!(emitter.error_count(), 0);
        assert_eq!(emitter.warning_count(), 0);
        assert!(emitter.diagnostics().is_empty());
    }

    #[test]
    fn test_emitter_default() {
        let emitter = DiagnosticEmitter::default();
        assert!(!emitter.has_errors());
        assert_eq!(emitter.error_count(), 0);
    }

    // -- DiagnosticEmitter file registration --------------------------------

    #[test]
    fn test_emitter_register_file() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "main.c");
        emitter.register_file(FileId(1), "header.h");

        assert_eq!(emitter.resolve_file_path(FileId(0)), "main.c");
        assert_eq!(emitter.resolve_file_path(FileId(1)), "header.h");
        assert_eq!(emitter.resolve_file_path(FileId(99)), "<unknown>");
    }

    #[test]
    fn test_emitter_register_file_sparse() {
        let mut emitter = DiagnosticEmitter::new();
        // Register file at index 5 — indices 0-4 should be empty.
        emitter.register_file(FileId(5), "sparse.c");
        assert_eq!(emitter.resolve_file_path(FileId(5)), "sparse.c");
        assert_eq!(emitter.resolve_file_path(FileId(0)), "<unknown>");
        assert_eq!(emitter.resolve_file_path(FileId(3)), "<unknown>");
    }

    #[test]
    fn test_emitter_sync_source_map() {
        let mut source_map = SourceMap::new();
        let id0 = source_map.add_file(PathBuf::from("main.c"), String::from("int main() {}"));
        let id1 = source_map.add_file(PathBuf::from("util.h"), String::from("void f();"));

        let mut emitter = DiagnosticEmitter::new();
        emitter.sync_source_map(&source_map);

        assert_eq!(emitter.resolve_file_path(id0), "main.c");
        assert_eq!(emitter.resolve_file_path(id1), "util.h");
    }

    // -- DiagnosticEmitter error emission -----------------------------------

    #[test]
    fn test_emitter_error_increments_count() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");

        let loc = test_location(FileId(0), 1, 1);
        emitter.error(loc, "test error");

        assert!(emitter.has_errors());
        assert_eq!(emitter.error_count(), 1);
        assert_eq!(emitter.warning_count(), 0);
        assert_eq!(emitter.diagnostics().len(), 1);
    }

    #[test]
    fn test_emitter_multiple_errors() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");

        let loc = test_location(FileId(0), 1, 1);
        emitter.error(loc, "first error");
        emitter.error(loc, "second error");
        emitter.error(loc, "third error");

        assert_eq!(emitter.error_count(), 3);
        assert!(emitter.has_errors());
    }

    // -- DiagnosticEmitter warning emission ---------------------------------

    #[test]
    fn test_emitter_warning_increments_count() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");

        let loc = test_location(FileId(0), 5, 10);
        emitter.warning(loc, "test warning");

        assert!(!emitter.has_errors()); // Warnings do NOT count as errors
        assert_eq!(emitter.error_count(), 0);
        assert_eq!(emitter.warning_count(), 1);
        assert_eq!(emitter.diagnostics().len(), 1);
    }

    #[test]
    fn test_emitter_only_warnings_no_errors() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");

        let loc = test_location(FileId(0), 1, 1);
        emitter.warning(loc, "warning 1");
        emitter.warning(loc, "warning 2");

        assert!(!emitter.has_errors());
        assert_eq!(emitter.error_count(), 0);
        assert_eq!(emitter.warning_count(), 2);
    }

    // -- DiagnosticEmitter note emission ------------------------------------

    #[test]
    fn test_emitter_note_does_not_increment_counters() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");

        let loc = test_location(FileId(0), 1, 1);
        emitter.note(loc, "supplementary information");

        assert!(!emitter.has_errors());
        assert_eq!(emitter.error_count(), 0);
        assert_eq!(emitter.warning_count(), 0);
        assert_eq!(emitter.diagnostics().len(), 1);
    }

    // -- DiagnosticEmitter error_no_loc ------------------------------------

    #[test]
    fn test_emitter_error_no_loc() {
        let mut emitter = DiagnosticEmitter::new();

        emitter.error_no_loc("no input files");

        assert!(emitter.has_errors());
        assert_eq!(emitter.error_count(), 1);

        let diag = &emitter.diagnostics()[0];
        assert!(diag.location.is_none());
        assert_eq!(format!("{}", diag), "error: no input files");
    }

    // -- DiagnosticEmitter mixed diagnostics --------------------------------

    #[test]
    fn test_emitter_mixed_diagnostics() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "main.c");

        let loc1 = test_location(FileId(0), 10, 5);
        let loc2 = test_location(FileId(0), 20, 1);
        let loc3 = test_location(FileId(0), 15, 3);

        emitter.error(loc1, "undeclared identifier 'x'");
        emitter.warning(loc2, "unused variable 'z'");
        emitter.note(loc3, "declared here");
        emitter.error_no_loc("too many errors");

        assert!(emitter.has_errors());
        assert_eq!(emitter.error_count(), 2); // error + error_no_loc
        assert_eq!(emitter.warning_count(), 1);
        assert_eq!(emitter.diagnostics().len(), 4);
    }

    // -- Diagnostic format via emitter (end-to-end) -------------------------

    #[test]
    fn test_emitter_gcc_format_error() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "hello.c");

        let loc = test_location(FileId(0), 10, 5);
        let diag = emitter.make_diagnostic(
            Severity::Error,
            Some(loc),
            "use of undeclared identifier 'x'".to_string(),
        );
        assert_eq!(
            format!("{}", diag),
            "hello.c:10:5: error: use of undeclared identifier 'x'"
        );
    }

    #[test]
    fn test_emitter_gcc_format_warning() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "hello.c");

        let loc = test_location(FileId(0), 20, 1);
        let diag = emitter.make_diagnostic(
            Severity::Warning,
            Some(loc),
            "unused variable 'z'".to_string(),
        );
        assert_eq!(
            format!("{}", diag),
            "hello.c:20:1: warning: unused variable 'z'"
        );
    }

    #[test]
    fn test_emitter_gcc_format_note() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "hello.c");

        let loc = test_location(FileId(0), 10, 5);
        let diag =
            emitter.make_diagnostic(Severity::Note, Some(loc), "did you mean 'y'?".to_string());
        assert_eq!(format!("{}", diag), "hello.c:10:5: note: did you mean 'y'?");
    }

    #[test]
    fn test_emitter_gcc_format_no_location() {
        let emitter = DiagnosticEmitter::new();
        let diag = emitter.make_diagnostic(Severity::Error, None, "no input files".to_string());
        assert_eq!(format!("{}", diag), "error: no input files");
    }

    // -- DiagnosticEmitter with SourceMap integration -----------------------

    #[test]
    fn test_emitter_with_source_map_integration() {
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file(
            PathBuf::from("src/main.c"),
            String::from("int main() {\n  return 0;\n}\n"),
        );

        let mut emitter = DiagnosticEmitter::new();
        emitter.sync_source_map(&source_map);

        let loc = source_map.lookup_location(file_id, 14); // 'r' in 'return'
        emitter.error(loc, "unexpected token");

        assert!(emitter.has_errors());
        let diag = &emitter.diagnostics()[0];
        // Should resolve file path through source map sync
        assert!(format!("{}", diag).starts_with("src/main.c:"));
        assert!(format!("{}", diag).contains("error: unexpected token"));
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_emitter_emit_diagnostic_with_attached_notes() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");

        let loc = test_location(FileId(0), 10, 5);
        let note_loc = test_location(FileId(0), 5, 1);

        let note = Diagnostic::new_with_file(
            Severity::Note,
            Some(note_loc),
            "previous declaration here".to_string(),
            "test.c".to_string(),
        );
        let diag = Diagnostic::new_with_file(
            Severity::Error,
            Some(loc),
            "redefinition of 'foo'".to_string(),
            "test.c".to_string(),
        )
        .with_note(note);

        emitter.emit(diag);

        // Only the Error should be counted, not the Note
        assert_eq!(emitter.error_count(), 1);
        assert_eq!(emitter.warning_count(), 0);
    }

    #[test]
    fn test_emitter_has_errors_initially_false() {
        let emitter = DiagnosticEmitter::new();
        assert!(!emitter.has_errors());
    }

    #[test]
    fn test_emitter_has_errors_after_warning_still_false() {
        let mut emitter = DiagnosticEmitter::new();
        let loc = test_location(FileId(0), 1, 1);
        emitter.warning(loc, "some warning");
        assert!(!emitter.has_errors());
    }

    #[test]
    fn test_emitter_has_errors_after_error_true() {
        let mut emitter = DiagnosticEmitter::new();
        let loc = test_location(FileId(0), 1, 1);
        emitter.error(loc, "some error");
        assert!(emitter.has_errors());
    }

    #[test]
    fn test_emitter_has_errors_after_error_no_loc_true() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.error_no_loc("fatal error");
        assert!(emitter.has_errors());
    }

    #[test]
    fn test_diagnostic_set_file_name() {
        let loc = test_location(FileId(0), 1, 1);
        let mut diag = Diagnostic::new(Severity::Error, Some(loc), "test".to_string());
        // Initially file_name is empty
        assert_eq!(format!("{}", diag), "<unknown>:1:1: error: test");
        // After setting file name
        diag.set_file_name("updated.c".to_string());
        assert_eq!(format!("{}", diag), "updated.c:1:1: error: test");
    }

    #[test]
    fn test_source_location_dummy_usage() {
        let loc = SourceLocation::dummy();
        assert!(loc.is_dummy());
        assert_eq!(loc.line, 0);
        assert_eq!(loc.column, 0);
    }

    #[test]
    fn test_diagnostic_line_column_1_based() {
        let loc = test_location(FileId(0), 1, 1);
        let diag = Diagnostic::new_with_file(
            Severity::Error,
            Some(loc),
            "first character".to_string(),
            "a.c".to_string(),
        );
        // Line 1, column 1 — the first position in the file
        assert_eq!(format!("{}", diag), "a.c:1:1: error: first character");
    }

    #[test]
    fn test_emitter_diagnostics_slice() {
        let mut emitter = DiagnosticEmitter::new();
        emitter.register_file(FileId(0), "test.c");
        let loc = test_location(FileId(0), 1, 1);

        emitter.error(loc, "e1");
        emitter.warning(loc, "w1");
        emitter.error(loc, "e2");

        let diags = emitter.diagnostics();
        assert_eq!(diags.len(), 3);
        assert_eq!(diags[0].severity, Severity::Error);
        assert_eq!(diags[0].message, "e1");
        assert_eq!(diags[1].severity, Severity::Warning);
        assert_eq!(diags[1].message, "w1");
        assert_eq!(diags[2].severity, Severity::Error);
        assert_eq!(diags[2].message, "e2");
    }
}
