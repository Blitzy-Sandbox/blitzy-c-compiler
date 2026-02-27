//! Source file registry and position tracking for the `bcc` compiler.
//!
//! This module provides the central source file registry (`SourceMap`) that tracks
//! all source files loaded during compilation, maps byte offsets to line/column
//! positions, handles macro expansion chains for diagnostic tracing, and supports
//! `#line` directive overrides.
//!
//! # Consumers
//! - `diagnostics.rs` — uses `SourceLocation` and `SourceMap` for GCC-compatible
//!   error location formatting (`file:line:col: error: message`).
//! - `frontend::lexer` — tokens carry `SourceLocation` for position tracking.
//! - `frontend::preprocessor` — registers included files, records macro expansions.
//! - `debug::line_program` — drives DWARF `.debug_line` section generation.
//!
//! # Performance
//! Byte-offset-to-line/column lookups use binary search on precomputed line-start
//! tables, yielding O(log n) performance per lookup where n is the number of lines.
//!
//! # Safety
//! This module contains zero `unsafe` blocks. All operations use safe Rust idioms.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// FileId — lightweight identifier for registered source files
// ---------------------------------------------------------------------------

/// A compact, copyable identifier representing a registered source file within
/// the [`SourceMap`]. The internal `u32` is an index into the `SourceMap`'s file
/// storage vector.
///
/// `FileId` is intentionally kept small (4 bytes) so it can be cheaply embedded
/// in every token and AST node without memory overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

impl FileId {
    /// Returns the raw index value of this file identifier.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// SourceLocation — pinpoint a position in source code
// ---------------------------------------------------------------------------

/// Represents a precise position within a source file.
///
/// `SourceLocation` is `Copy` so it can be cheaply embedded in every token,
/// AST node, and IR instruction without heap allocation overhead. Fields use
/// `u32` to keep the struct at 16 bytes total (4 × u32).
///
/// # Conventions
/// - `line` is **1-based** (line 1 is the first line of the file).
/// - `column` is **1-based** and measured in **bytes** (not characters).
/// - `byte_offset` is **0-based** from the start of the file content.
/// - A `line` value of `0` indicates a synthetic/dummy location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    /// Identifies which source file this location belongs to.
    pub file_id: FileId,
    /// Byte offset from the start of the file (0-based).
    pub byte_offset: u32,
    /// Line number (1-based). A value of 0 indicates a dummy location.
    pub line: u32,
    /// Column number (1-based, measured in bytes).
    pub column: u32,
}

impl SourceLocation {
    /// Creates a dummy/synthetic `SourceLocation` for compiler-generated nodes
    /// that do not correspond to any actual source text.
    ///
    /// The dummy location uses `FileId(0)`, byte offset 0, line 0, and column 0.
    /// Downstream consumers (diagnostics, DWARF) check `line == 0` to detect
    /// synthetic locations and omit them from output.
    #[inline]
    pub fn dummy() -> Self {
        SourceLocation {
            file_id: FileId(0),
            byte_offset: 0,
            line: 0,
            column: 0,
        }
    }

    /// Returns `true` if this location is a dummy/synthetic location.
    #[inline]
    pub fn is_dummy(&self) -> bool {
        self.line == 0
    }
}

impl Default for SourceLocation {
    fn default() -> Self {
        Self::dummy()
    }
}

// ---------------------------------------------------------------------------
// SourceSpan — a range of source text
// ---------------------------------------------------------------------------

/// A contiguous range of source text, defined by its start and end locations.
///
/// `SourceSpan` is `Copy` so it can be cheaply stored alongside AST nodes and
/// tokens to record the extent of each syntactic construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    /// The location of the first byte in the span.
    pub start: SourceLocation,
    /// The location one past the last byte in the span.
    pub end: SourceLocation,
}

impl SourceSpan {
    /// Creates a zero-width span at a single location.
    #[inline]
    pub fn at(loc: SourceLocation) -> Self {
        SourceSpan {
            start: loc,
            end: loc,
        }
    }

    /// Creates a dummy span (both endpoints are dummy locations).
    #[inline]
    pub fn dummy() -> Self {
        SourceSpan {
            start: SourceLocation::dummy(),
            end: SourceLocation::dummy(),
        }
    }
}

impl Default for SourceSpan {
    fn default() -> Self {
        Self::dummy()
    }
}

// ---------------------------------------------------------------------------
// SourceFile — internal per-file storage
// ---------------------------------------------------------------------------

/// Internal representation of a single registered source file.
///
/// Stores the file path, full source text, and a precomputed table of line-start
/// byte offsets for efficient O(log n) line/column lookups via binary search.
struct SourceFile {
    /// File path as provided to the compiler (may be relative or absolute).
    path: PathBuf,
    /// Full source text content of the file.
    content: String,
    /// Byte offsets marking the start of each line within `content`.
    ///
    /// Invariant: `line_starts[0] == 0` (the first line always starts at byte 0).
    /// For each `\n` at byte offset `i` in `content`, there is an entry
    /// `line_starts[k] = i + 1` representing the start of the next line.
    ///
    /// Binary search on this vector converts a byte offset to a 0-based line
    /// index in O(log n) time.
    line_starts: Vec<u32>,
}

impl SourceFile {
    /// Creates a new `SourceFile`, precomputing the line-start table from the
    /// given content.
    fn new(path: PathBuf, content: String) -> Self {
        let line_starts = Self::compute_line_starts(&content);
        SourceFile {
            path,
            content,
            line_starts,
        }
    }

    /// Scans `content` for newline characters and builds a vector of byte
    /// offsets where each line begins.
    ///
    /// The first entry is always `0` (the first line starts at the beginning).
    /// Each `\n` at byte position `i` produces an entry `i + 1`.
    fn compute_line_starts(content: &str) -> Vec<u32> {
        let mut starts = Vec::with_capacity(content.len() / 40 + 1);
        starts.push(0);
        for (i, byte) in content.bytes().enumerate() {
            if byte == b'\n' {
                starts.push((i as u32) + 1);
            }
        }
        starts
    }

    /// Returns the number of lines in this file.
    ///
    /// A file with no newlines has exactly 1 line. Each `\n` adds one line.
    #[inline]
    fn line_count(&self) -> u32 {
        self.line_starts.len() as u32
    }
}

// ---------------------------------------------------------------------------
// ExpansionInfo — macro expansion chain tracking
// ---------------------------------------------------------------------------

/// Records a single link in a macro expansion chain.
///
/// When a macro is expanded, the compiler creates an `ExpansionInfo` entry
/// that maps the expansion site (where the macro was invoked) to the spelling
/// site (the location within the macro definition). This chain allows
/// diagnostics to print notes like:
///
/// ```text
/// header.h:5:10: error: invalid operands
/// main.c:20:3: note: in expansion of macro 'FOO'
/// ```
struct ExpansionInfo {
    /// The location where the macro was invoked.
    expansion_loc: SourceLocation,
    /// The location within the macro definition that is being expanded.
    spelling_loc: SourceLocation,
}

// ---------------------------------------------------------------------------
// LineOverride — #line directive support
// ---------------------------------------------------------------------------

/// Records a `#line` directive override that changes the reported file and/or
/// line number for diagnostics at a particular byte offset.
///
/// When the preprocessor encounters `#line 42 "fake.c"`, it stores a
/// `LineOverride` entry keyed by `(FileId, byte_offset)`. The diagnostic
/// formatter checks for overrides when formatting source locations.
struct LineOverride {
    /// Optional override file path (from `#line N "file"`).
    override_file: Option<PathBuf>,
    /// The overridden line number (from `#line N`).
    override_line: u32,
}

// ---------------------------------------------------------------------------
// SourceMap — central source file registry
// ---------------------------------------------------------------------------

/// The central registry of all source files loaded during compilation.
///
/// `SourceMap` provides:
/// - File registration via [`add_file`](SourceMap::add_file)
/// - Efficient byte-offset-to-line/column lookups via [`lookup_location`](SourceMap::lookup_location)
/// - File path and content retrieval for diagnostics and debug info
/// - Macro expansion chain tracking for diagnostic note generation
/// - `#line` directive override support for the preprocessor
///
/// # Thread Safety
/// `SourceMap` is designed for single-threaded use within the compilation
/// pipeline. Interior mutability (e.g., `RefCell`) is not required because
/// the source map is passed as `&mut self` during preprocessing and as `&self`
/// during later phases.
pub struct SourceMap {
    /// All registered source files, indexed by `FileId`. The `FileId(n)` value
    /// corresponds to `files[n]`.
    files: Vec<SourceFile>,

    /// Macro expansion chain entries. Each entry records an expansion site and
    /// the corresponding spelling site within the macro definition.
    expansion_chains: Vec<ExpansionInfo>,

    /// `#line` directive overrides. Keyed by `(FileId, byte_offset)` where
    /// `byte_offset` is the position in the file where the `#line` directive
    /// appeared. The diagnostic formatter checks this map to override the
    /// reported file and line number.
    line_overrides: HashMap<(FileId, u32), LineOverride>,
}

impl SourceMap {
    // -- Construction -------------------------------------------------------

    /// Creates an empty `SourceMap` with no registered files.
    pub fn new() -> Self {
        SourceMap {
            files: Vec::new(),
            expansion_chains: Vec::new(),
            line_overrides: HashMap::new(),
        }
    }

    // -- File Registration --------------------------------------------------

    /// Registers a new source file in the map and returns its unique `FileId`.
    ///
    /// This method precomputes the line-start table for efficient subsequent
    /// lookups. It is called by the preprocessor each time a new file is opened,
    /// including files reached via `#include`.
    ///
    /// # Arguments
    /// - `path` — The file path as provided to the compiler. May be relative or
    ///   absolute; the map stores it as-is.
    /// - `content` — The full source text of the file.
    ///
    /// # Returns
    /// A `FileId` that can be used to look up the file's path, content, and
    /// byte-offset-to-line/column mappings.
    pub fn add_file(&mut self, path: PathBuf, content: String) -> FileId {
        let id = FileId(self.files.len() as u32);
        self.files.push(SourceFile::new(path, content));
        id
    }

    // -- File Queries -------------------------------------------------------

    /// Returns the path of the file identified by `file_id`.
    ///
    /// Used by the diagnostic emitter to print `file:line:col:` prefixes and
    /// by DWARF debug info for file reference tables.
    ///
    /// # Panics
    /// Panics if `file_id` does not correspond to a registered file.
    pub fn get_file_path(&self, file_id: FileId) -> &Path {
        &self.files[file_id.0 as usize].path
    }

    /// Returns the full source text content of the file identified by `file_id`.
    ///
    /// Used by the lexer to tokenize the file content and by diagnostics to
    /// show source context.
    ///
    /// # Panics
    /// Panics if `file_id` does not correspond to a registered file.
    pub fn get_file_content(&self, file_id: FileId) -> &str {
        &self.files[file_id.0 as usize].content
    }

    /// Returns the number of files currently registered in the source map.
    #[inline]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Returns the number of lines in the file identified by `file_id`.
    ///
    /// # Panics
    /// Panics if `file_id` does not correspond to a registered file.
    #[inline]
    pub fn line_count(&self, file_id: FileId) -> u32 {
        self.files[file_id.0 as usize].line_count()
    }

    // -- Location Lookup ----------------------------------------------------

    /// Converts a byte offset within a file to a full `SourceLocation` with
    /// line and column numbers.
    ///
    /// This is the primary lookup method used during diagnostic emission and
    /// DWARF line program generation. It uses binary search on the precomputed
    /// line-start table for O(log n) performance.
    ///
    /// # Arguments
    /// - `file_id` — The file containing the byte offset.
    /// - `byte_offset` — A 0-based byte offset from the start of the file.
    ///
    /// # Returns
    /// A `SourceLocation` with 1-based line and column numbers.
    ///
    /// # Panics
    /// Panics if `file_id` does not correspond to a registered file or if
    /// `byte_offset` exceeds the file's content length.
    pub fn lookup_location(&self, file_id: FileId, byte_offset: u32) -> SourceLocation {
        let file = &self.files[file_id.0 as usize];

        // Binary search: `partition_point` returns the number of elements for
        // which the predicate is true. Since `line_starts` is sorted and
        // `line_starts[0] == 0`, the result gives us one past the line index.
        //
        // Example: for content "abc\ndef\n", line_starts = [0, 4, 8].
        // byte_offset=5 → partition_point returns 2 (both 0 and 4 are ≤ 5),
        // so line_index = 2 - 1 = 1 (second line, 0-based).
        let line_index = file
            .line_starts
            .partition_point(|&start| start <= byte_offset)
            .saturating_sub(1);

        let column = byte_offset - file.line_starts[line_index] + 1; // 1-based

        SourceLocation {
            file_id,
            byte_offset,
            line: (line_index as u32) + 1, // 1-based
            column,
        }
    }

    /// Returns the text content of a specific line in a file, without the
    /// trailing newline character.
    ///
    /// This is used by the diagnostic emitter to show source context alongside
    /// error messages (e.g., underlining the problematic token).
    ///
    /// # Arguments
    /// - `file_id` — The file to query.
    /// - `line` — The line number (**1-based**).
    ///
    /// # Returns
    /// A string slice of the line content, excluding the trailing `\n` (if any).
    ///
    /// # Panics
    /// Panics if `file_id` is invalid or `line` is out of range (0 or greater
    /// than the file's line count).
    pub fn get_line_content(&self, file_id: FileId, line: u32) -> &str {
        let file = &self.files[file_id.0 as usize];

        // Convert 1-based line to 0-based index
        let line_index = (line - 1) as usize;

        let start = file.line_starts[line_index] as usize;

        // Determine the end of this line: either the next line's start offset
        // (minus 1 for the `\n`) or the end of the file content.
        let end = if line_index + 1 < file.line_starts.len() {
            // There is a next line; the current line ends at the `\n` just
            // before the next line's start.
            let next_line_start = file.line_starts[line_index + 1] as usize;
            // Strip the trailing `\n`
            if next_line_start > start && file.content.as_bytes()[next_line_start - 1] == b'\n' {
                next_line_start - 1
            } else {
                next_line_start
            }
        } else {
            // Last line — extends to end of content
            file.content.len()
        };

        &file.content[start..end]
    }

    // -- Macro Expansion Tracking -------------------------------------------

    /// Records a macro expansion chain entry and returns a synthetic
    /// `SourceLocation` that can be traced back through the chain.
    ///
    /// When the preprocessor expands a macro, it calls this method to record:
    /// - `expansion_loc` — Where in the source the macro was invoked.
    /// - `spelling_loc` — The corresponding position within the macro's definition.
    ///
    /// The returned `SourceLocation` carries a `byte_offset` that encodes the
    /// index into the expansion chain (offset by a sentinel value), allowing
    /// diagnostics to detect expansion locations and print expansion notes.
    ///
    /// # Arguments
    /// - `expansion_loc` — The location where the macro was invoked.
    /// - `spelling_loc` — The location within the macro definition.
    ///
    /// # Returns
    /// The `expansion_loc` enriched with expansion tracking. Currently returns
    /// the `expansion_loc` directly, with the chain stored for later retrieval.
    pub fn add_expansion(
        &mut self,
        expansion_loc: SourceLocation,
        spelling_loc: SourceLocation,
    ) -> SourceLocation {
        self.expansion_chains.push(ExpansionInfo {
            expansion_loc,
            spelling_loc,
        });
        // Return the expansion location so callers can use it directly.
        // The chain index is `expansion_chains.len() - 1` and can be
        // retrieved by walking the chain during diagnostic emission.
        expansion_loc
    }

    /// Returns the number of recorded macro expansion chain entries.
    #[inline]
    pub fn expansion_count(&self) -> usize {
        self.expansion_chains.len()
    }

    /// Retrieves the expansion info at the given chain index, if it exists.
    ///
    /// This is used by diagnostics to walk the macro expansion chain and
    /// emit "in expansion of macro" notes.
    ///
    /// # Returns
    /// A tuple `(expansion_loc, spelling_loc)` if the index is valid, or
    /// `None` if the index is out of range.
    pub fn get_expansion(&self, index: usize) -> Option<(SourceLocation, SourceLocation)> {
        self.expansion_chains
            .get(index)
            .map(|info| (info.expansion_loc, info.spelling_loc))
    }

    // -- #line Directive Overrides ------------------------------------------

    /// Records a `#line` directive override for a specific position in a file.
    ///
    /// When the preprocessor encounters `#line N` or `#line N "filename"`, it
    /// calls this method to register the override. The diagnostic formatter
    /// checks for overrides at a given byte offset to change the reported
    /// file and line number in error messages.
    ///
    /// # Arguments
    /// - `file_id` — The file containing the `#line` directive.
    /// - `byte_offset` — The byte offset where the `#line` directive appeared.
    /// - `new_line` — The new line number to report from this point onward.
    /// - `new_file` — An optional new file path to report from this point onward.
    pub fn set_line_override(
        &mut self,
        file_id: FileId,
        byte_offset: u32,
        new_line: u32,
        new_file: Option<PathBuf>,
    ) {
        self.line_overrides.insert(
            (file_id, byte_offset),
            LineOverride {
                override_file: new_file,
                override_line: new_line,
            },
        );
    }

    /// Retrieves the `#line` override for a specific position, if one exists.
    ///
    /// # Returns
    /// A tuple `(override_line, optional_override_file)` if an override is
    /// registered at exactly `(file_id, byte_offset)`, or `None` otherwise.
    pub fn get_line_override(
        &self,
        file_id: FileId,
        byte_offset: u32,
    ) -> Option<(u32, Option<&Path>)> {
        self.line_overrides.get(&(file_id, byte_offset)).map(|ov| {
            (ov.override_line, ov.override_file.as_deref())
        })
    }
}

impl Default for SourceMap {
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

    // -- FileId tests -------------------------------------------------------

    #[test]
    fn test_file_id_is_copy_and_eq() {
        let a = FileId(1);
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(a.index(), 1);
    }

    #[test]
    fn test_file_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FileId(0));
        set.insert(FileId(1));
        set.insert(FileId(0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_file_id_ne() {
        assert_ne!(FileId(0), FileId(1));
    }

    // -- SourceLocation tests -----------------------------------------------

    #[test]
    fn test_source_location_dummy() {
        let loc = SourceLocation::dummy();
        assert_eq!(loc.file_id, FileId(0));
        assert_eq!(loc.byte_offset, 0);
        assert_eq!(loc.line, 0);
        assert_eq!(loc.column, 0);
        assert!(loc.is_dummy());
    }

    #[test]
    fn test_source_location_is_copy() {
        let loc = SourceLocation {
            file_id: FileId(1),
            byte_offset: 10,
            line: 2,
            column: 5,
        };
        let loc2 = loc; // Copy
        assert_eq!(loc, loc2);
        assert!(!loc.is_dummy());
    }

    #[test]
    fn test_source_location_default() {
        let loc = SourceLocation::default();
        assert!(loc.is_dummy());
    }

    // -- SourceSpan tests ---------------------------------------------------

    #[test]
    fn test_source_span_at() {
        let loc = SourceLocation {
            file_id: FileId(0),
            byte_offset: 5,
            line: 1,
            column: 6,
        };
        let span = SourceSpan::at(loc);
        assert_eq!(span.start, loc);
        assert_eq!(span.end, loc);
    }

    #[test]
    fn test_source_span_dummy() {
        let span = SourceSpan::dummy();
        assert!(span.start.is_dummy());
        assert!(span.end.is_dummy());
    }

    #[test]
    fn test_source_span_default() {
        let span = SourceSpan::default();
        assert!(span.start.is_dummy());
    }

    // -- SourceMap: add_file and getters ------------------------------------

    #[test]
    fn test_add_file_returns_sequential_ids() {
        let mut sm = SourceMap::new();
        let id0 = sm.add_file(PathBuf::from("a.c"), String::from("hello"));
        let id1 = sm.add_file(PathBuf::from("b.c"), String::from("world"));
        assert_eq!(id0, FileId(0));
        assert_eq!(id1, FileId(1));
        assert_eq!(sm.file_count(), 2);
    }

    #[test]
    fn test_get_file_path() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("src/main.c"), String::from("int main() {}"));
        assert_eq!(sm.get_file_path(id), Path::new("src/main.c"));
    }

    #[test]
    fn test_get_file_content() {
        let mut sm = SourceMap::new();
        let content = "int x = 42;\n";
        let id = sm.add_file(PathBuf::from("test.c"), content.to_string());
        assert_eq!(sm.get_file_content(id), content);
    }

    // -- SourceMap: lookup_location -----------------------------------------

    #[test]
    fn test_lookup_location_single_line() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("hello world"));

        let loc = sm.lookup_location(id, 0);
        assert_eq!(loc.file_id, id);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);
        assert_eq!(loc.byte_offset, 0);

        let loc = sm.lookup_location(id, 5);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 6);
    }

    #[test]
    fn test_lookup_location_multi_line() {
        let mut sm = SourceMap::new();
        // Line 1: "abc\n" (bytes 0-3), line 2: "def\n" (bytes 4-7), line 3: "ghi" (bytes 8-10)
        let id = sm.add_file(PathBuf::from("test.c"), String::from("abc\ndef\nghi"));

        // First character of line 1
        let loc = sm.lookup_location(id, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        // Last character of line 1 (the 'c')
        let loc = sm.lookup_location(id, 2);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 3);

        // The newline at end of line 1
        let loc = sm.lookup_location(id, 3);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 4);

        // First character of line 2 ('d')
        let loc = sm.lookup_location(id, 4);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);

        // Middle of line 2 ('e')
        let loc = sm.lookup_location(id, 5);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 2);

        // First character of line 3 ('g')
        let loc = sm.lookup_location(id, 8);
        assert_eq!(loc.line, 3);
        assert_eq!(loc.column, 1);

        // Last character of line 3 ('i')
        let loc = sm.lookup_location(id, 10);
        assert_eq!(loc.line, 3);
        assert_eq!(loc.column, 3);
    }

    #[test]
    fn test_lookup_location_at_line_boundaries() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("a\nb\nc\n"));

        // Byte 0 = 'a' → line 1, col 1
        assert_eq!(sm.lookup_location(id, 0).line, 1);
        assert_eq!(sm.lookup_location(id, 0).column, 1);

        // Byte 1 = '\n' → line 1, col 2
        assert_eq!(sm.lookup_location(id, 1).line, 1);
        assert_eq!(sm.lookup_location(id, 1).column, 2);

        // Byte 2 = 'b' → line 2, col 1
        assert_eq!(sm.lookup_location(id, 2).line, 2);
        assert_eq!(sm.lookup_location(id, 2).column, 1);

        // Byte 3 = '\n' → line 2, col 2
        assert_eq!(sm.lookup_location(id, 3).line, 2);
        assert_eq!(sm.lookup_location(id, 3).column, 2);

        // Byte 4 = 'c' → line 3, col 1
        assert_eq!(sm.lookup_location(id, 4).line, 3);
        assert_eq!(sm.lookup_location(id, 4).column, 1);

        // Byte 5 = '\n' → line 3, col 2
        assert_eq!(sm.lookup_location(id, 5).line, 3);
        assert_eq!(sm.lookup_location(id, 5).column, 2);
    }

    #[test]
    fn test_lookup_location_byte_offset_at_end_of_content() {
        let mut sm = SourceMap::new();
        let content = "abc\ndef";
        let id = sm.add_file(PathBuf::from("test.c"), content.to_string());

        // Byte offset at end of content (offset 6 = 'f')
        let loc = sm.lookup_location(id, 6);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 3);
    }

    // -- SourceMap: get_line_content ----------------------------------------

    #[test]
    fn test_get_line_content_basic() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("hello\nworld\n"));

        assert_eq!(sm.get_line_content(id, 1), "hello");
        assert_eq!(sm.get_line_content(id, 2), "world");
    }

    #[test]
    fn test_get_line_content_no_trailing_newline() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("line1\nline2"));

        assert_eq!(sm.get_line_content(id, 1), "line1");
        assert_eq!(sm.get_line_content(id, 2), "line2");
    }

    #[test]
    fn test_get_line_content_single_line_no_newline() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("only line"));

        assert_eq!(sm.get_line_content(id, 1), "only line");
    }

    #[test]
    fn test_get_line_content_single_line_with_newline() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("only line\n"));

        assert_eq!(sm.get_line_content(id, 1), "only line");
    }

    #[test]
    fn test_get_line_content_empty_lines() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("a\n\nb\n"));

        assert_eq!(sm.get_line_content(id, 1), "a");
        assert_eq!(sm.get_line_content(id, 2), ""); // empty line
        assert_eq!(sm.get_line_content(id, 3), "b");
    }

    // -- SourceMap: empty file edge case ------------------------------------

    #[test]
    fn test_empty_file() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("empty.c"), String::new());

        assert_eq!(sm.get_file_content(id), "");
        assert_eq!(sm.line_count(id), 1); // one empty line

        // Lookup at byte 0 in an empty file
        let loc = sm.lookup_location(id, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        // get_line_content for the single empty line
        assert_eq!(sm.get_line_content(id, 1), "");
    }

    #[test]
    fn test_file_with_only_newline() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("nl.c"), String::from("\n"));

        assert_eq!(sm.line_count(id), 2); // line 1 (empty) and line 2 (empty)

        let loc = sm.lookup_location(id, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        assert_eq!(sm.get_line_content(id, 1), "");
        assert_eq!(sm.get_line_content(id, 2), "");
    }

    // -- SourceMap: line_count ----------------------------------------------

    #[test]
    fn test_line_count() {
        let mut sm = SourceMap::new();

        let id0 = sm.add_file(PathBuf::from("a.c"), String::from("one line"));
        assert_eq!(sm.line_count(id0), 1);

        let id1 = sm.add_file(PathBuf::from("b.c"), String::from("line1\nline2\nline3"));
        assert_eq!(sm.line_count(id1), 3);

        let id2 = sm.add_file(PathBuf::from("c.c"), String::from("line1\nline2\n"));
        assert_eq!(sm.line_count(id2), 3); // trailing \n creates an empty 3rd line
    }

    // -- SourceMap: macro expansion chain -----------------------------------

    #[test]
    fn test_add_expansion() {
        let mut sm = SourceMap::new();
        let _id = sm.add_file(PathBuf::from("main.c"), String::from("FOO(1);\n"));
        let _header_id = sm.add_file(PathBuf::from("header.h"), String::from("#define FOO(x) x+1\n"));

        let expansion_loc = SourceLocation {
            file_id: FileId(0),
            byte_offset: 0,
            line: 1,
            column: 1,
        };
        let spelling_loc = SourceLocation {
            file_id: FileId(1),
            byte_offset: 15,
            line: 1,
            column: 16,
        };

        let returned = sm.add_expansion(expansion_loc, spelling_loc);
        assert_eq!(returned, expansion_loc);
        assert_eq!(sm.expansion_count(), 1);

        let (exp, spell) = sm.get_expansion(0).unwrap();
        assert_eq!(exp, expansion_loc);
        assert_eq!(spell, spelling_loc);
    }

    #[test]
    fn test_get_expansion_out_of_range() {
        let sm = SourceMap::new();
        assert!(sm.get_expansion(0).is_none());
        assert!(sm.get_expansion(999).is_none());
    }

    // -- SourceMap: #line directive overrides --------------------------------

    #[test]
    fn test_set_and_get_line_override() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("line1\nline2\n"));

        // Set a #line override at byte offset 6 (start of line 2)
        sm.set_line_override(id, 6, 100, Some(PathBuf::from("fake.c")));

        let ov = sm.get_line_override(id, 6).unwrap();
        assert_eq!(ov.0, 100);
        assert_eq!(ov.1, Some(Path::new("fake.c")));
    }

    #[test]
    fn test_line_override_without_file() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("content\n"));

        sm.set_line_override(id, 0, 42, None);

        let ov = sm.get_line_override(id, 0).unwrap();
        assert_eq!(ov.0, 42);
        assert_eq!(ov.1, None);
    }

    #[test]
    fn test_line_override_not_found() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("content\n"));

        assert!(sm.get_line_override(id, 0).is_none());
        assert!(sm.get_line_override(id, 999).is_none());
    }

    #[test]
    fn test_line_override_overwrite() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("content\n"));

        sm.set_line_override(id, 0, 10, None);
        sm.set_line_override(id, 0, 20, Some(PathBuf::from("other.c")));

        let ov = sm.get_line_override(id, 0).unwrap();
        assert_eq!(ov.0, 20);
        assert_eq!(ov.1, Some(Path::new("other.c")));
    }

    // -- SourceMap: default trait -------------------------------------------

    #[test]
    fn test_source_map_default() {
        let sm = SourceMap::default();
        assert_eq!(sm.file_count(), 0);
        assert_eq!(sm.expansion_count(), 0);
    }

    // -- SourceMap: binary search correctness --------------------------------

    #[test]
    fn test_binary_search_many_lines() {
        let mut sm = SourceMap::new();

        // Create a file with 100 lines, each containing "line_NNN\n"
        let mut content = String::new();
        for i in 0..100 {
            content.push_str(&format!("line_{:03}\n", i));
        }
        let id = sm.add_file(PathBuf::from("big.c"), content.clone());
        assert_eq!(sm.line_count(id), 101); // 100 lines + 1 trailing empty

        // Verify every line start maps correctly
        let bytes = content.as_bytes();
        let mut line_num = 1u32;
        let mut col = 1u32;
        for (i, &b) in bytes.iter().enumerate() {
            let loc = sm.lookup_location(id, i as u32);
            assert_eq!(
                loc.line, line_num,
                "byte_offset={}, expected line={}, got line={}",
                i, line_num, loc.line
            );
            assert_eq!(
                loc.column, col,
                "byte_offset={}, expected col={}, got col={}",
                i, col, loc.column
            );
            if b == b'\n' {
                line_num += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
    }

    // -- SourceMap: trailing newline handling --------------------------------

    #[test]
    fn test_file_ending_with_newline() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("abc\n"));

        assert_eq!(sm.line_count(id), 2);
        assert_eq!(sm.get_line_content(id, 1), "abc");
        assert_eq!(sm.get_line_content(id, 2), ""); // empty trailing line
    }

    #[test]
    fn test_file_ending_without_newline() {
        let mut sm = SourceMap::new();
        let id = sm.add_file(PathBuf::from("test.c"), String::from("abc"));

        assert_eq!(sm.line_count(id), 1);
        assert_eq!(sm.get_line_content(id, 1), "abc");
    }

    // -- SourceMap: multiple files ------------------------------------------

    #[test]
    fn test_multiple_files_independent() {
        let mut sm = SourceMap::new();
        let id_a = sm.add_file(PathBuf::from("a.c"), String::from("aaa\nbbb\n"));
        let id_b = sm.add_file(PathBuf::from("b.c"), String::from("ccc\n"));

        // File a, line 2
        let loc = sm.lookup_location(id_a, 4);
        assert_eq!(loc.file_id, id_a);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);

        // File b, line 1
        let loc = sm.lookup_location(id_b, 0);
        assert_eq!(loc.file_id, id_b);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        // Paths are independent
        assert_eq!(sm.get_file_path(id_a), Path::new("a.c"));
        assert_eq!(sm.get_file_path(id_b), Path::new("b.c"));
    }

    // -- SourceMap: realistic C source snippet -------------------------------

    #[test]
    fn test_realistic_c_source() {
        let mut sm = SourceMap::new();
        let source = "\
#include <stdio.h>
int main(void) {
    printf(\"Hello, world!\\n\");
    return 0;
}
";
        let id = sm.add_file(PathBuf::from("hello.c"), source.to_string());

        // Line 1: "#include <stdio.h>"
        assert_eq!(sm.get_line_content(id, 1), "#include <stdio.h>");

        // Line 2: "int main(void) {"
        assert_eq!(sm.get_line_content(id, 2), "int main(void) {");

        // Byte offset of 'i' in "int" (line 2)
        // Line 1 is "#include <stdio.h>\n" = 19 bytes + \n = 20 bytes
        let loc = sm.lookup_location(id, 19);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);

        // 'n' in "int" at line 2
        let loc = sm.lookup_location(id, 20);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 2);
    }
}
