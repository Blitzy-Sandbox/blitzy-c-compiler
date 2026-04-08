//! Source position tracking for the `bcc` lexer.
//!
//! This module provides the position-tracking primitives consumed by the lexer
//! and all downstream pipeline phases (parser, semantic analysis, DWARF debug
//! info). It re-exports the canonical [`SourceLocation`], [`SourceSpan`], and
//! [`FileId`] types from [`crate::common::source_map`] and augments
//! [`SourceSpan`] with additional utility methods required by the lexer and
//! parser. It also defines the [`PositionTracker`] state machine that tracks
//! the current file position during character-by-character scanning.
//!
//! # Design
//!
//! Rather than duplicating the common source position types, this module
//! re-exports them so that every compiler phase works with the exact same
//! `SourceLocation` and `SourceSpan` types. The additional `impl` blocks
//! here extend `SourceSpan` with convenience constructors and accessors that
//! are particularly useful for token-level source ranges.
//!
//! # Conventions
//!
//! * **Line and column numbers are 1-based** (line 1 is the first line).
//! * **Byte offsets are 0-based** (the first byte in a file is offset 0).
//! * A line value of `0` indicates a synthetic/dummy location.
//!
//! # Performance
//!
//! Position tracking runs on every character consumed by the lexer, so all
//! operations in this module are designed to be extremely lightweight — field
//! increments only, zero allocation.

use std::fmt;

// ---------------------------------------------------------------------------
// Re-exports from the common source_map module — canonical types used across
// every compiler phase.
// ---------------------------------------------------------------------------

pub use crate::common::source_map::FileId;
pub use crate::common::source_map::SourceLocation;
pub use crate::common::source_map::SourceSpan;

// ---------------------------------------------------------------------------
// Extension methods on SourceSpan
// ---------------------------------------------------------------------------

/// Additional utility methods on [`SourceSpan`] used throughout the lexer and
/// parser. These extend the base `at()` / `dummy()` API defined in
/// `common::source_map` with richer span manipulation.
impl SourceSpan {
    /// Creates a new span from explicit start and end locations.
    ///
    /// `start` is the first byte of the range (inclusive) and `end` is the
    /// byte past the last byte of the range (exclusive).
    #[inline]
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        SourceSpan { start, end }
    }

    /// Creates a zero-width span at a single location.
    ///
    /// This is an alias for [`SourceSpan::at`] using the naming convention
    /// expected by downstream consumers (token builders, AST node
    /// constructors) that conceptually want a "point" span.
    #[inline]
    pub fn point(loc: SourceLocation) -> Self {
        SourceSpan {
            start: loc,
            end: loc,
        }
    }

    /// Merges two spans into the smallest span that covers both.
    ///
    /// The resulting span starts at whichever input begins earlier (by byte
    /// offset) and ends at whichever input ends later. This is used when
    /// combining a sequence of tokens into an AST node span.
    #[inline]
    pub fn merge(self, other: SourceSpan) -> SourceSpan {
        SourceSpan {
            start: if self.start.byte_offset <= other.start.byte_offset {
                self.start
            } else {
                other.start
            },
            end: if self.end.byte_offset >= other.end.byte_offset {
                self.end
            } else {
                other.end
            },
        }
    }

    /// Returns the byte length of this span.
    ///
    /// Computed as `end.byte_offset - start.byte_offset` with saturation to
    /// avoid underflow on malformed spans.
    #[inline]
    pub fn len(&self) -> u32 {
        self.end.byte_offset.saturating_sub(self.start.byte_offset)
    }

    /// Returns `true` if this is a zero-width (empty) span.
    ///
    /// Zero-width spans typically represent synthesized or compiler-generated
    /// tokens that have no corresponding source text.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start.byte_offset == self.end.byte_offset
    }
}

// ---------------------------------------------------------------------------
// Display implementations — GCC-compatible diagnostic formatting
// ---------------------------------------------------------------------------

/// Formats a `SourceLocation` as `line:column` for use in diagnostic messages.
///
/// Example output: `10:5`
impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Formats a `SourceSpan` as `start_line:start_col-end_line:end_col`.
///
/// Example output: `10:5-10:12`
impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}-{}:{}",
            self.start.line, self.start.column, self.end.line, self.end.column
        )
    }
}

// ---------------------------------------------------------------------------
// PositionTracker — lexer-specific position state machine
// ---------------------------------------------------------------------------

/// Tracks the current byte offset, line, and column during lexer scanning.
///
/// A `PositionTracker` is created once per file at the start of lexing and is
/// advanced on every character consumed. Its methods are deliberately minimal
/// (field increments only) to stay on the hot path without incurring allocation
/// or branching overhead.
///
/// # Usage
///
/// ```ignore
/// let tracker = PositionTracker::new(file_id);
/// let start = tracker.current_location();
/// // ... consume characters via advance_byte / advance_newline ...
/// let span = tracker.span_from(start);
/// ```
#[derive(Debug, Clone)]
pub struct PositionTracker {
    /// The file being tokenized.
    file_id: FileId,
    /// Current byte offset from the start of the file (0-based).
    byte_offset: u32,
    /// Current line number (1-based).
    line: u32,
    /// Current column number (1-based, measured in bytes).
    column: u32,
}

impl PositionTracker {
    // -- Construction -------------------------------------------------------

    /// Creates a new position tracker for the given file, starting at line 1,
    /// column 1, byte offset 0.
    #[inline]
    pub fn new(file_id: FileId) -> Self {
        PositionTracker {
            file_id,
            byte_offset: 0,
            line: 1,
            column: 1,
        }
    }

    // -- Position queries ---------------------------------------------------

    /// Returns a [`SourceLocation`] snapshot of the current position.
    ///
    /// This is called at the start of each token to record its beginning, and
    /// again at the end to record the span.
    #[inline]
    pub fn current_location(&self) -> SourceLocation {
        SourceLocation {
            file_id: self.file_id,
            byte_offset: self.byte_offset,
            line: self.line,
            column: self.column,
        }
    }

    /// Returns the current byte offset (0-based).
    #[inline]
    pub fn offset(&self) -> u32 {
        self.byte_offset
    }

    /// Returns the current line number (1-based).
    #[inline]
    pub fn line(&self) -> u32 {
        self.line
    }

    /// Returns the current column number (1-based).
    #[inline]
    pub fn column(&self) -> u32 {
        self.column
    }

    // -- Position advancement -----------------------------------------------

    /// Advances past one non-newline byte.
    ///
    /// Increments `byte_offset` by 1 and `column` by 1. Call this for every
    /// ordinary character consumed (anything except `\n` or `\r\n`).
    #[inline]
    pub fn advance_byte(&mut self) {
        self.byte_offset += 1;
        self.column += 1;
    }

    /// Advances past a Unix newline (`\n`).
    ///
    /// Increments `byte_offset` by 1, increments `line` by 1, and resets
    /// `column` to 1.
    #[inline]
    pub fn advance_newline(&mut self) {
        self.byte_offset += 1;
        self.line += 1;
        self.column = 1;
    }

    /// Advances past a Windows newline (`\r\n`).
    ///
    /// Increments `byte_offset` by 2 (for both `\r` and `\n`), increments
    /// `line` by 1, and resets `column` to 1.
    #[inline]
    pub fn advance_crlf(&mut self) {
        self.byte_offset += 2;
        self.line += 1;
        self.column = 1;
    }

    /// Advances past `count` non-newline bytes at once.
    ///
    /// This is a convenience method for skipping known multi-byte tokens
    /// (e.g., operators like `>>=` or multi-character punctuation like `...`)
    /// without calling [`advance_byte`](Self::advance_byte) in a loop.
    #[inline]
    pub fn advance_bytes(&mut self, count: u32) {
        self.byte_offset += count;
        self.column += count;
    }

    // -- Span construction --------------------------------------------------

    /// Builds a [`SourceSpan`] from a saved start location to the current
    /// position.
    ///
    /// The lexer saves the start location before scanning a token, then calls
    /// this method after consuming all of the token's characters to produce the
    /// span stored in the resulting [`Token`].
    #[inline]
    pub fn span_from(&self, start: SourceLocation) -> SourceSpan {
        SourceSpan {
            start,
            end: self.current_location(),
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper -------------------------------------------------------------

    /// Convenience helper to construct a `SourceLocation` for tests.
    fn loc(file: u32, offset: u32, line: u32, col: u32) -> SourceLocation {
        SourceLocation {
            file_id: FileId(file),
            byte_offset: offset,
            line,
            column: col,
        }
    }

    // -- SourceLocation tests -----------------------------------------------

    #[test]
    fn test_source_location_construction() {
        let l = loc(1, 10, 2, 5);
        assert_eq!(l.file_id, FileId(1));
        assert_eq!(l.byte_offset, 10);
        assert_eq!(l.line, 2);
        assert_eq!(l.column, 5);
    }

    #[test]
    fn test_source_location_is_copy() {
        let a = loc(0, 0, 1, 1);
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_source_location_dummy() {
        let d = SourceLocation::dummy();
        assert_eq!(d.file_id, FileId(0));
        assert_eq!(d.byte_offset, 0);
        assert_eq!(d.line, 0);
        assert_eq!(d.column, 0);
    }

    #[test]
    fn test_source_location_display() {
        let l = loc(0, 0, 10, 5);
        assert_eq!(format!("{}", l), "10:5");
    }

    #[test]
    fn test_source_location_display_first_position() {
        let l = loc(0, 0, 1, 1);
        assert_eq!(format!("{}", l), "1:1");
    }

    // -- SourceSpan tests ---------------------------------------------------

    #[test]
    fn test_source_span_new() {
        let start = loc(0, 5, 1, 6);
        let end = loc(0, 10, 1, 11);
        let span = SourceSpan::new(start, end);
        assert_eq!(span.start, start);
        assert_eq!(span.end, end);
    }

    #[test]
    fn test_source_span_point_zero_width() {
        let l = loc(0, 5, 1, 6);
        let span = SourceSpan::point(l);
        assert_eq!(span.start, l);
        assert_eq!(span.end, l);
        assert!(span.is_empty());
    }

    #[test]
    fn test_source_span_dummy() {
        let span = SourceSpan::dummy();
        assert_eq!(span.start, SourceLocation::dummy());
        assert_eq!(span.end, SourceLocation::dummy());
        assert!(span.is_empty());
    }

    #[test]
    fn test_source_span_is_copy() {
        let span = SourceSpan::new(loc(0, 0, 1, 1), loc(0, 5, 1, 6));
        let span2 = span; // Copy
        assert_eq!(span, span2);
    }

    #[test]
    fn test_source_span_len() {
        let span = SourceSpan::new(loc(0, 5, 1, 6), loc(0, 12, 1, 13));
        assert_eq!(span.len(), 7);
    }

    #[test]
    fn test_source_span_len_zero() {
        let span = SourceSpan::point(loc(0, 5, 1, 6));
        assert_eq!(span.len(), 0);
    }

    #[test]
    fn test_source_span_is_empty_true() {
        let span = SourceSpan::point(loc(0, 3, 1, 4));
        assert!(span.is_empty());
    }

    #[test]
    fn test_source_span_is_empty_false() {
        let span = SourceSpan::new(loc(0, 3, 1, 4), loc(0, 7, 1, 8));
        assert!(!span.is_empty());
    }

    #[test]
    fn test_source_span_merge_non_overlapping() {
        let a = SourceSpan::new(loc(0, 0, 1, 1), loc(0, 5, 1, 6));
        let b = SourceSpan::new(loc(0, 10, 2, 1), loc(0, 15, 2, 6));
        let merged = a.merge(b);
        assert_eq!(merged.start.byte_offset, 0);
        assert_eq!(merged.end.byte_offset, 15);
    }

    #[test]
    fn test_source_span_merge_reversed_order() {
        let a = SourceSpan::new(loc(0, 10, 2, 1), loc(0, 15, 2, 6));
        let b = SourceSpan::new(loc(0, 0, 1, 1), loc(0, 5, 1, 6));
        let merged = a.merge(b);
        assert_eq!(merged.start.byte_offset, 0);
        assert_eq!(merged.end.byte_offset, 15);
    }

    #[test]
    fn test_source_span_merge_overlapping() {
        let a = SourceSpan::new(loc(0, 0, 1, 1), loc(0, 10, 1, 11));
        let b = SourceSpan::new(loc(0, 5, 1, 6), loc(0, 15, 1, 16));
        let merged = a.merge(b);
        assert_eq!(merged.start.byte_offset, 0);
        assert_eq!(merged.end.byte_offset, 15);
    }

    #[test]
    fn test_source_span_merge_identical() {
        let s = SourceSpan::new(loc(0, 3, 1, 4), loc(0, 7, 1, 8));
        let merged = s.merge(s);
        assert_eq!(merged, s);
    }

    #[test]
    fn test_source_span_display() {
        let span = SourceSpan::new(loc(0, 0, 1, 1), loc(0, 5, 1, 6));
        assert_eq!(format!("{}", span), "1:1-1:6");
    }

    #[test]
    fn test_source_span_display_multiline() {
        let span = SourceSpan::new(loc(0, 0, 1, 1), loc(0, 50, 3, 10));
        assert_eq!(format!("{}", span), "1:1-3:10");
    }

    // -- PositionTracker tests ----------------------------------------------

    #[test]
    fn test_tracker_new_initial_state() {
        let file_id = FileId(0);
        let tracker = PositionTracker::new(file_id);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 1);
        assert_eq!(tracker.offset(), 0);
    }

    #[test]
    fn test_tracker_new_one_based_line_and_column() {
        let tracker = PositionTracker::new(FileId(0));
        // Critical invariant: lines and columns start at 1, not 0
        assert_eq!(tracker.line(), 1, "First line must be 1 (1-based)");
        assert_eq!(tracker.column(), 1, "First column must be 1 (1-based)");
        assert_eq!(tracker.offset(), 0, "First byte offset must be 0 (0-based)");
    }

    #[test]
    fn test_tracker_current_location() {
        let file_id = FileId(3);
        let tracker = PositionTracker::new(file_id);
        let loc = tracker.current_location();
        assert_eq!(loc.file_id, file_id);
        assert_eq!(loc.byte_offset, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn test_tracker_advance_byte() {
        let mut tracker = PositionTracker::new(FileId(0));
        tracker.advance_byte();
        assert_eq!(tracker.offset(), 1);
        assert_eq!(tracker.column(), 2);
        assert_eq!(tracker.line(), 1);

        tracker.advance_byte();
        assert_eq!(tracker.offset(), 2);
        assert_eq!(tracker.column(), 3);
        assert_eq!(tracker.line(), 1);
    }

    #[test]
    fn test_tracker_advance_newline() {
        let mut tracker = PositionTracker::new(FileId(0));
        // Advance 3 bytes on line 1
        tracker.advance_byte();
        tracker.advance_byte();
        tracker.advance_byte();
        assert_eq!(tracker.offset(), 3);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 4);

        // Hit newline
        tracker.advance_newline();
        assert_eq!(tracker.offset(), 4);
        assert_eq!(tracker.line(), 2);
        assert_eq!(tracker.column(), 1);
    }

    #[test]
    fn test_tracker_advance_crlf() {
        let mut tracker = PositionTracker::new(FileId(0));
        // Advance some bytes
        tracker.advance_byte();
        tracker.advance_byte();
        assert_eq!(tracker.offset(), 2);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 3);

        // Hit \r\n
        tracker.advance_crlf();
        assert_eq!(tracker.offset(), 4); // skipped 2 bytes (\r + \n)
        assert_eq!(tracker.line(), 2);
        assert_eq!(tracker.column(), 1);
    }

    #[test]
    fn test_tracker_advance_bytes() {
        let mut tracker = PositionTracker::new(FileId(0));
        tracker.advance_bytes(5);
        assert_eq!(tracker.offset(), 5);
        assert_eq!(tracker.column(), 6);
        assert_eq!(tracker.line(), 1);
    }

    #[test]
    fn test_tracker_advance_bytes_zero() {
        let mut tracker = PositionTracker::new(FileId(0));
        tracker.advance_bytes(0);
        assert_eq!(tracker.offset(), 0);
        assert_eq!(tracker.column(), 1);
        assert_eq!(tracker.line(), 1);
    }

    #[test]
    fn test_tracker_span_from() {
        let mut tracker = PositionTracker::new(FileId(0));
        let start = tracker.current_location();

        tracker.advance_byte(); // 'i'
        tracker.advance_byte(); // 'n'
        tracker.advance_byte(); // 't'

        let span = tracker.span_from(start);
        assert_eq!(span.start.byte_offset, 0);
        assert_eq!(span.start.line, 1);
        assert_eq!(span.start.column, 1);
        assert_eq!(span.end.byte_offset, 3);
        assert_eq!(span.end.line, 1);
        assert_eq!(span.end.column, 4);
        assert_eq!(span.len(), 3);
    }

    #[test]
    fn test_tracker_span_from_multiline() {
        let mut tracker = PositionTracker::new(FileId(0));

        // "ab\ncd" — token "ab" on line 1, then newline, then "cd" on line 2
        let start = tracker.current_location();
        tracker.advance_byte(); // 'a'
        tracker.advance_byte(); // 'b'
        tracker.advance_newline(); // '\n'
        tracker.advance_byte(); // 'c'
        tracker.advance_byte(); // 'd'

        let span = tracker.span_from(start);
        assert_eq!(span.start.line, 1);
        assert_eq!(span.start.column, 1);
        assert_eq!(span.end.line, 2);
        assert_eq!(span.end.column, 3);
        assert_eq!(span.len(), 5);
    }

    #[test]
    fn test_tracker_sequential_advancement() {
        // Simulate scanning "int x;\n" character by character
        let mut tracker = PositionTracker::new(FileId(1));

        // 'i' at (1, 1)
        assert_eq!(tracker.offset(), 0);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 1);
        tracker.advance_byte();

        // 'n' at (1, 2)
        assert_eq!(tracker.offset(), 1);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 2);
        tracker.advance_byte();

        // 't' at (1, 3)
        assert_eq!(tracker.offset(), 2);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 3);
        tracker.advance_byte();

        // ' ' at (1, 4)
        assert_eq!(tracker.offset(), 3);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 4);
        tracker.advance_byte();

        // 'x' at (1, 5)
        assert_eq!(tracker.offset(), 4);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 5);
        tracker.advance_byte();

        // ';' at (1, 6)
        assert_eq!(tracker.offset(), 5);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 6);
        tracker.advance_byte();

        // '\n' at (1, 7) — becomes start of line 2 after advance
        assert_eq!(tracker.offset(), 6);
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 7);
        tracker.advance_newline();

        // Now at start of line 2
        assert_eq!(tracker.offset(), 7);
        assert_eq!(tracker.line(), 2);
        assert_eq!(tracker.column(), 1);
    }

    #[test]
    fn test_tracker_multiple_newlines() {
        let mut tracker = PositionTracker::new(FileId(0));
        tracker.advance_newline(); // line 1 → 2
        tracker.advance_newline(); // line 2 → 3
        tracker.advance_newline(); // line 3 → 4
        assert_eq!(tracker.line(), 4);
        assert_eq!(tracker.column(), 1);
        assert_eq!(tracker.offset(), 3);
    }

    #[test]
    fn test_tracker_mixed_newline_styles() {
        // Simulate: "a\r\nb\nc"
        let mut tracker = PositionTracker::new(FileId(0));
        tracker.advance_byte(); // 'a'
        assert_eq!(tracker.line(), 1);
        assert_eq!(tracker.column(), 2);

        tracker.advance_crlf(); // \r\n
        assert_eq!(tracker.line(), 2);
        assert_eq!(tracker.column(), 1);
        assert_eq!(tracker.offset(), 3);

        tracker.advance_byte(); // 'b'
        assert_eq!(tracker.line(), 2);
        assert_eq!(tracker.column(), 2);

        tracker.advance_newline(); // \n
        assert_eq!(tracker.line(), 3);
        assert_eq!(tracker.column(), 1);
        assert_eq!(tracker.offset(), 5);

        tracker.advance_byte(); // 'c'
        assert_eq!(tracker.line(), 3);
        assert_eq!(tracker.column(), 2);
        assert_eq!(tracker.offset(), 6);
    }

    #[test]
    fn test_tracker_file_id_propagated() {
        let file_id = FileId(42);
        let mut tracker = PositionTracker::new(file_id);
        tracker.advance_byte();
        tracker.advance_newline();
        tracker.advance_byte();

        let loc = tracker.current_location();
        assert_eq!(loc.file_id, file_id);
    }

    #[test]
    fn test_source_span_at_vs_point_equivalence() {
        let l = loc(0, 5, 2, 3);
        let span_at = SourceSpan::at(l);
        let span_point = SourceSpan::point(l);
        assert_eq!(span_at, span_point);
    }

    #[test]
    fn test_source_span_len_saturating() {
        // Ensure len() never underflows even with a weird span where end < start
        let weird = SourceSpan {
            start: loc(0, 10, 1, 11),
            end: loc(0, 5, 1, 6),
        };
        assert_eq!(weird.len(), 0); // saturating_sub prevents underflow
    }

    #[test]
    fn test_source_span_merge_with_zero_width() {
        let a = SourceSpan::point(loc(0, 5, 1, 6));
        let b = SourceSpan::new(loc(0, 5, 1, 6), loc(0, 10, 1, 11));
        let merged = a.merge(b);
        assert_eq!(merged.start.byte_offset, 5);
        assert_eq!(merged.end.byte_offset, 10);
    }

    #[test]
    fn test_tracker_span_from_zero_width() {
        let tracker = PositionTracker::new(FileId(0));
        let start = tracker.current_location();
        let span = tracker.span_from(start);
        assert!(span.is_empty());
        assert_eq!(span.len(), 0);
    }
}
