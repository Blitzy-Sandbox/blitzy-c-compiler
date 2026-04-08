//! Lexer entry point for the `bcc` C compiler.
//!
//! This module is the primary interface between the preprocessor and the parser.
//! It converts preprocessed C source text into a `Vec<Token>`, where each token
//! carries its classification ([`TokenKind`]), source span ([`SourceSpan`]), and
//! optional associated value ([`TokenValue`]).
//!
//! # Public API
//!
//! - [`Lexer`] — Struct that holds lexing state and provides `new()` and
//!   `tokenize()` methods.
//! - [`tokenize`] — Convenience free function wrapping `Lexer` for one-shot use.
//! - Re-exports of [`Token`], [`TokenKind`], [`SourceLocation`], and
//!   [`SourceSpan`] for downstream parser consumption.
//!
//! # Design
//!
//! The lexer performs byte-level scanning on the preprocessed source text. It
//! delegates literal scanning (numbers, strings, characters) to the
//! [`literals`] submodule and keyword identification to the [`keywords`]
//! submodule. Source position tracking is handled by [`source::PositionTracker`].
//!
//! # Performance
//!
//! Designed for ~230K LOC (SQLite amalgamation) within the <60s compile-time
//! budget. Key optimizations:
//! - Byte-level scanning (`&[u8]`) avoids UTF-8 decoding overhead for ASCII C.
//! - String interning via [`Interner`] deduplicates all identifiers.
//! - Single-pass tokenization with no backtracking.
//! - Pre-allocated output `Vec<Token>` with estimated capacity.
//!
//! # Integration
//!
//! Per AAP §0.4.1:
//! - **Preprocessor → Lexer**: Receives preprocessed source text (`&str`).
//! - **Lexer → Parser**: Produces `Vec<Token>` terminated by `TokenKind::Eof`.
//! - **Cross-cutting**: All identifier values interned via `common::intern::Interner`.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks.

// ===========================================================================
// Submodule declarations
// ===========================================================================

/// Token type definitions: [`Token`], [`TokenKind`], [`TokenValue`].
pub mod token;

/// C11 + GCC extension keyword lookup table.
pub mod keywords;

/// Numeric, string, and character literal scanning.
pub mod literals;

/// Source position tracking: [`SourceLocation`], [`SourceSpan`],
/// [`PositionTracker`].
pub mod source;

// ===========================================================================
// Re-exports for downstream consumers (parser, sema, etc.)
// ===========================================================================

pub use source::{SourceLocation, SourceSpan};
pub use token::{Token, TokenKind, TokenValue};

// ===========================================================================
// Internal imports
// ===========================================================================

use keywords::lookup_keyword;
use literals::{scan_char, scan_number, scan_string, CharPrefix, StringPrefix};
use source::{FileId, PositionTracker};

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::intern::Interner;

// ===========================================================================
// Lexer — the main tokenizer struct
// ===========================================================================

/// The C lexer, converting preprocessed source text into a `Vec<Token>`.
///
/// `Lexer` holds mutable references to the shared string interner and the
/// diagnostic emitter, allowing it to intern identifiers and report errors
/// in GCC-compatible format as it scans.
///
/// # Lifetime
///
/// The `'a` lifetime ties the lexer to the source text, interner, and
/// diagnostic emitter. The lexer does not outlive any of these references.
///
/// # Usage
///
/// ```ignore
/// let mut interner = Interner::new();
/// let mut diag = DiagnosticEmitter::new();
/// let mut lexer = Lexer::new(source, file_id, &mut interner, &mut diag);
/// let tokens = lexer.tokenize();
/// ```
pub struct Lexer<'a> {
    /// The raw source bytes for efficient byte-level scanning.
    bytes: &'a [u8],
    /// Position tracker maintaining current line, column, and byte offset.
    tracker: PositionTracker,
    /// String interner for identifier deduplication.
    interner: &'a mut Interner,
    /// Diagnostic emitter for error/warning reporting.
    diagnostics: &'a mut DiagnosticEmitter,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given source text.
    ///
    /// # Arguments
    ///
    /// - `source` — Preprocessed C source text to tokenize.
    /// - `file_id` — File identifier for source position tracking.
    /// - `interner` — Mutable reference to the string interner.
    /// - `diagnostics` — Mutable reference to the diagnostic emitter.
    #[inline]
    pub fn new(
        source: &'a str,
        file_id: FileId,
        interner: &'a mut Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Self {
        Lexer {
            bytes: source.as_bytes(),
            tracker: PositionTracker::new(file_id),
            interner,
            diagnostics,
        }
    }

    /// Tokenizes the entire source text, returning all tokens.
    ///
    /// The returned `Vec<Token>` always ends with a `TokenKind::Eof` token.
    /// On encountering lexing errors (unterminated strings, invalid characters,
    /// etc.), the lexer reports the error via the diagnostic emitter and
    /// recovers by skipping the offending character(s), continuing to tokenize
    /// the remainder.
    pub fn tokenize(&mut self) -> Vec<Token> {
        // Estimate ~10 tokens per line, ~15 bytes per line average.
        let estimated_tokens = (self.bytes.len() / 15).max(16);
        let mut tokens = Vec::with_capacity(estimated_tokens);

        loop {
            self.skip_whitespace_and_comments();

            if self.is_at_end() {
                let loc = self.tracker.current_location();
                let span = SourceSpan::point(loc);
                tokens.push(Token::new(TokenKind::Eof, span, TokenValue::None));
                break;
            }

            match self.scan_token() {
                Some(tok) => tokens.push(tok),
                None => {
                    // scan_token returned None — it already reported the
                    // error and advanced past the invalid character.
                }
            }
        }

        tokens
    }

    // =======================================================================
    // Private — main token scanning dispatch
    // =======================================================================

    /// Scans a single token from the current position.
    ///
    /// Returns `Some(Token)` on success, or `None` if an error was
    /// encountered and reported (the invalid character has been skipped).
    fn scan_token(&mut self) -> Option<Token> {
        let start = self.tracker.current_location();
        let b = self.peek()?;

        match b {
            // ---------------------------------------------------------------
            // Numeric literals: [0-9]
            // ---------------------------------------------------------------
            b'0'..=b'9' => self.scan_numeric_literal(start),

            // ---------------------------------------------------------------
            // Dot: could be `.123` float or `.` or `...`
            // ---------------------------------------------------------------
            b'.' => {
                // Check if next char is a digit → float literal like `.5`
                if let Some(next) = self.peek_next() {
                    if next.is_ascii_digit() {
                        return self.scan_numeric_literal(start);
                    }
                }
                // Check for `...` (ellipsis)
                if self.matches_ahead(b"...") {
                    self.tracker.advance_bytes(3);
                    let span = self.tracker.span_from(start);
                    return Some(Token::new(TokenKind::Ellipsis, span, TokenValue::None));
                }
                // Single dot
                self.tracker.advance_byte();
                let span = self.tracker.span_from(start);
                Some(Token::new(TokenKind::Dot, span, TokenValue::None))
            }

            // ---------------------------------------------------------------
            // String literals: "..."
            // ---------------------------------------------------------------
            b'"' => self.scan_string_literal(start, StringPrefix::None, 0),

            // ---------------------------------------------------------------
            // Character literals: '...'
            // ---------------------------------------------------------------
            b'\'' => self.scan_char_literal(start, CharPrefix::None, 0),

            // ---------------------------------------------------------------
            // Identifiers, keywords, and string/char prefix detection.
            //
            // L, u, U prefix handling is checked here BEFORE falling through
            // to identifier scanning, so that `L"..."`, `u"..."`, `U"..."`,
            // `u8"..."`, `L'...'`, `u'...'`, and `U'...'` are correctly
            // recognized as prefixed string/char literals.
            // ---------------------------------------------------------------
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                // Check for string/char literal prefixes: L, u, U, u8.
                match b {
                    b'L' => {
                        if self.peek_at(1) == Some(b'"') {
                            return self.scan_string_literal(start, StringPrefix::Wide, 1);
                        }
                        if self.peek_at(1) == Some(b'\'') {
                            return self.scan_char_literal(start, CharPrefix::Wide, 1);
                        }
                    }
                    b'u' => {
                        if self.peek_at(1) == Some(b'8') && self.peek_at(2) == Some(b'"') {
                            return self.scan_string_literal(start, StringPrefix::Utf8, 2);
                        }
                        if self.peek_at(1) == Some(b'"') {
                            return self.scan_string_literal(start, StringPrefix::Utf16, 1);
                        }
                        if self.peek_at(1) == Some(b'\'') {
                            return self.scan_char_literal(start, CharPrefix::Utf16, 1);
                        }
                    }
                    b'U' => {
                        if self.peek_at(1) == Some(b'"') {
                            return self.scan_string_literal(start, StringPrefix::Utf32, 1);
                        }
                        if self.peek_at(1) == Some(b'\'') {
                            return self.scan_char_literal(start, CharPrefix::Utf32, 1);
                        }
                    }
                    _ => {}
                }
                // Not a string/char prefix — scan as identifier or keyword.
                Some(self.scan_identifier_or_keyword(start))
            }

            // ---------------------------------------------------------------
            // Operators and punctuation
            // ---------------------------------------------------------------
            b'+' | b'-' | b'*' | b'/' | b'%' | b'=' | b'!' | b'<' | b'>' | b'&' | b'|' | b'^'
            | b'~' | b'?' | b':' | b';' | b',' | b'(' | b')' | b'{' | b'}' | b'[' | b']' | b'#' => {
                self.scan_operator_or_punctuation(start)
            }

            // ---------------------------------------------------------------
            // Invalid character
            // ---------------------------------------------------------------
            _ => {
                let loc = self.tracker.current_location();
                self.diagnostics.error(
                    loc,
                    format!("invalid character '{}' (0x{:02X})", b as char, b),
                );
                self.tracker.advance_byte();
                None
            }
        }
    }

    // =======================================================================
    // Identifier / keyword scanning
    // =======================================================================

    /// Scans an identifier or keyword starting at `start`.
    ///
    /// Reads all characters matching `[a-zA-Z0-9_]`, then checks the keyword
    /// table. If it's a keyword, returns the corresponding TokenKind; otherwise
    /// interns the identifier string and returns `TokenKind::Identifier`.
    fn scan_identifier_or_keyword(&mut self, start: SourceLocation) -> Token {
        let byte_start = self.tracker.offset() as usize;

        // Consume all identifier characters.
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.tracker.advance_byte();
            } else {
                break;
            }
        }

        let byte_end = self.tracker.offset() as usize;
        let span = self.tracker.span_from(start);

        // Extract the identifier text from the source bytes.
        let name = std::str::from_utf8(&self.bytes[byte_start..byte_end]).unwrap_or("<invalid>");

        // Check keyword table first.
        if let Some(kind) = lookup_keyword(name) {
            return Token::new(kind, span, TokenValue::None);
        }

        // Plain identifier — intern it.
        let intern_id = self.interner.intern(name);
        Token::new(
            TokenKind::Identifier,
            span,
            TokenValue::Identifier(intern_id),
        )
    }

    // =======================================================================
    // Numeric literal scanning
    // =======================================================================

    /// Scans a numeric literal (integer or floating-point) starting at `start`.
    fn scan_numeric_literal(&mut self, start: SourceLocation) -> Option<Token> {
        let byte_pos = self.tracker.offset() as usize;

        match scan_number(self.bytes, byte_pos) {
            Ok((numeric_value, bytes_consumed)) => {
                let kind = numeric_value.token_kind();
                let value = numeric_value.into_token_value();
                self.tracker.advance_bytes(bytes_consumed as u32);
                let span = self.tracker.span_from(start);
                Some(Token::new(kind, span, value))
            }
            Err(err) => {
                let loc = self.tracker.current_location();
                self.diagnostics.error(loc, &err.message);
                // Skip past any remaining digit-like characters to recover.
                while let Some(b) = self.peek() {
                    if b.is_ascii_alphanumeric() || b == b'.' || b == b'_' {
                        self.tracker.advance_byte();
                    } else {
                        break;
                    }
                }
                None
            }
        }
    }

    // =======================================================================
    // String literal scanning
    // =======================================================================

    /// Scans a string literal with the given prefix.
    ///
    /// `prefix_len` is the number of prefix bytes (0 for none, 1 for L/u/U,
    /// 2 for u8) that need to be advanced past before delegating to
    /// `literals::scan_string`.
    fn scan_string_literal(
        &mut self,
        start: SourceLocation,
        prefix: StringPrefix,
        prefix_len: u32,
    ) -> Option<Token> {
        // Advance past the prefix characters (L, u, U, u8).
        if prefix_len > 0 {
            self.tracker.advance_bytes(prefix_len);
        }

        let byte_pos = self.tracker.offset() as usize;

        match scan_string(self.bytes, byte_pos, prefix) {
            Ok((content, bytes_consumed)) => {
                self.tracker.advance_bytes(bytes_consumed as u32);
                let span = self.tracker.span_from(start);
                Some(Token::new(
                    TokenKind::StringLiteral,
                    span,
                    TokenValue::Str(content),
                ))
            }
            Err(err) => {
                let loc = self.tracker.current_location();
                self.diagnostics.error(loc, &err.message);
                // Attempt recovery: skip to end of line or closing quote.
                self.recover_from_string_error();
                None
            }
        }
    }

    /// Attempts error recovery after a string literal scanning failure.
    ///
    /// Skips forward until we find a closing `"`, a newline, or end of input.
    fn recover_from_string_error(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b'"' => {
                    self.tracker.advance_byte();
                    return;
                }
                b'\n' => {
                    self.tracker.advance_newline();
                    return;
                }
                b'\r' => {
                    if self.peek_next() == Some(b'\n') {
                        self.tracker.advance_crlf();
                    } else {
                        self.tracker.advance_byte();
                    }
                    return;
                }
                _ => {
                    self.tracker.advance_byte();
                }
            }
        }
    }

    // =======================================================================
    // Character literal scanning
    // =======================================================================

    /// Scans a character literal with the given prefix.
    fn scan_char_literal(
        &mut self,
        start: SourceLocation,
        prefix: CharPrefix,
        prefix_len: u32,
    ) -> Option<Token> {
        if prefix_len > 0 {
            self.tracker.advance_bytes(prefix_len);
        }

        let byte_pos = self.tracker.offset() as usize;

        match scan_char(self.bytes, byte_pos, prefix) {
            Ok((char_value, bytes_consumed)) => {
                self.tracker.advance_bytes(bytes_consumed as u32);
                let span = self.tracker.span_from(start);
                Some(Token::new(
                    TokenKind::CharLiteral,
                    span,
                    TokenValue::Char(char_value),
                ))
            }
            Err(err) => {
                let loc = self.tracker.current_location();
                self.diagnostics.error(loc, &err.message);
                // Attempt recovery: skip to closing quote or end of line.
                self.recover_from_char_error();
                None
            }
        }
    }

    /// Attempts error recovery after a character literal scanning failure.
    fn recover_from_char_error(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b'\'' => {
                    self.tracker.advance_byte();
                    return;
                }
                b'\n' => {
                    self.tracker.advance_newline();
                    return;
                }
                b'\r' => {
                    if self.peek_next() == Some(b'\n') {
                        self.tracker.advance_crlf();
                    } else {
                        self.tracker.advance_byte();
                    }
                    return;
                }
                _ => {
                    self.tracker.advance_byte();
                }
            }
        }
    }

    // =======================================================================
    // Operator and punctuation scanning
    // =======================================================================

    /// Scans an operator or punctuation token, using greedy (longest match)
    /// strategy for multi-character operators.
    fn scan_operator_or_punctuation(&mut self, start: SourceLocation) -> Option<Token> {
        let b = self.peek()?;
        let next = self.peek_next();
        let next2 = self.peek_at(2);

        // Three-character operators first.
        let (kind, advance) = match (b, next, next2) {
            (b'<', Some(b'<'), Some(b'=')) => (TokenKind::LessLessEqual, 3),
            (b'>', Some(b'>'), Some(b'=')) => (TokenKind::GreaterGreaterEqual, 3),
            // Note: `...` is handled in scan_token under the `b'.'` arm.
            _ => {
                // Two-character operators.
                match (b, next) {
                    (b'=', Some(b'=')) => (TokenKind::EqualEqual, 2),
                    (b'!', Some(b'=')) => (TokenKind::BangEqual, 2),
                    (b'<', Some(b'=')) => (TokenKind::LessEqual, 2),
                    (b'>', Some(b'=')) => (TokenKind::GreaterEqual, 2),
                    (b'&', Some(b'&')) => (TokenKind::AmpAmp, 2),
                    (b'|', Some(b'|')) => (TokenKind::PipePipe, 2),
                    (b'<', Some(b'<')) => (TokenKind::LessLess, 2),
                    (b'>', Some(b'>')) => (TokenKind::GreaterGreater, 2),
                    (b'+', Some(b'=')) => (TokenKind::PlusEqual, 2),
                    (b'-', Some(b'=')) => (TokenKind::MinusEqual, 2),
                    (b'*', Some(b'=')) => (TokenKind::StarEqual, 2),
                    (b'/', Some(b'=')) => (TokenKind::SlashEqual, 2),
                    (b'%', Some(b'=')) => (TokenKind::PercentEqual, 2),
                    (b'&', Some(b'=')) => (TokenKind::AmpEqual, 2),
                    (b'|', Some(b'=')) => (TokenKind::PipeEqual, 2),
                    (b'^', Some(b'=')) => (TokenKind::CaretEqual, 2),
                    (b'+', Some(b'+')) => (TokenKind::PlusPlus, 2),
                    (b'-', Some(b'-')) => (TokenKind::MinusMinus, 2),
                    (b'-', Some(b'>')) => (TokenKind::Arrow, 2),
                    (b'#', Some(b'#')) => (TokenKind::HashHash, 2),
                    // C digraphs (ISO C 6.4.6)
                    (b'<', Some(b':')) => (TokenKind::LeftBracket, 2),
                    (b':', Some(b'>')) => (TokenKind::RightBracket, 2),
                    (b'<', Some(b'%')) => (TokenKind::LeftBrace, 2),
                    (b'%', Some(b'>')) => (TokenKind::RightBrace, 2),
                    (b'%', Some(b':')) => (TokenKind::Hash, 2),
                    _ => {
                        // Single-character operators and punctuation.
                        let kind = match b {
                            b'+' => TokenKind::Plus,
                            b'-' => TokenKind::Minus,
                            b'*' => TokenKind::Star,
                            b'/' => TokenKind::Slash,
                            b'%' => TokenKind::Percent,
                            b'=' => TokenKind::Equal,
                            b'!' => TokenKind::Bang,
                            b'<' => TokenKind::Less,
                            b'>' => TokenKind::Greater,
                            b'&' => TokenKind::Amp,
                            b'|' => TokenKind::Pipe,
                            b'^' => TokenKind::Caret,
                            b'~' => TokenKind::Tilde,
                            b'?' => TokenKind::Question,
                            b':' => TokenKind::Colon,
                            b';' => TokenKind::Semicolon,
                            b',' => TokenKind::Comma,
                            b'(' => TokenKind::LeftParen,
                            b')' => TokenKind::RightParen,
                            b'{' => TokenKind::LeftBrace,
                            b'}' => TokenKind::RightBrace,
                            b'[' => TokenKind::LeftBracket,
                            b']' => TokenKind::RightBracket,
                            b'#' => TokenKind::Hash,
                            _ => unreachable!(
                                "scan_operator_or_punctuation called with unexpected byte 0x{:02X}",
                                b
                            ),
                        };
                        (kind, 1)
                    }
                }
            }
        };

        self.tracker.advance_bytes(advance);
        let span = self.tracker.span_from(start);
        Some(Token::new(kind, span, TokenValue::None))
    }

    // =======================================================================
    // Whitespace and comment skipping
    // =======================================================================

    /// Skips all whitespace and comments from the current position.
    ///
    /// This is called before each token scan. It handles:
    /// - Spaces, tabs, vertical tabs, form feeds, carriage returns
    /// - Unix newlines (`\n`) and Windows newlines (`\r\n`)
    /// - Line comments (`// ...`)
    /// - Block comments (`/* ... */`)
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                None => return,

                // Whitespace: space, tab, vertical tab, form feed
                Some(b' ') | Some(b'\t') | Some(b'\x0B') | Some(b'\x0C') => {
                    self.tracker.advance_byte();
                }

                // Newline handling
                Some(b'\n') => {
                    self.tracker.advance_newline();
                }
                Some(b'\r') => {
                    if self.peek_next() == Some(b'\n') {
                        self.tracker.advance_crlf();
                    } else {
                        // Bare carriage return treated as newline.
                        self.tracker.advance_newline();
                    }
                }

                // Potential comment start: '/'
                Some(b'/') => {
                    match self.peek_next() {
                        Some(b'/') => {
                            // Line comment: skip to end of line.
                            self.skip_line_comment();
                        }
                        Some(b'*') => {
                            // Block comment: skip to `*/`.
                            self.skip_block_comment();
                        }
                        _ => {
                            // Not a comment — it's the '/' operator.
                            return;
                        }
                    }
                }

                // Not whitespace or comment — done skipping.
                _ => return,
            }
        }
    }

    /// Skips a line comment (`// ...`) through end of line.
    ///
    /// The current position must be at the first `/` of `//`.
    fn skip_line_comment(&mut self) {
        // Skip the two '/' characters.
        self.tracker.advance_bytes(2);

        // Skip to end of line or end of input.
        while let Some(b) = self.peek() {
            match b {
                b'\n' => {
                    self.tracker.advance_newline();
                    return;
                }
                b'\r' => {
                    if self.peek_next() == Some(b'\n') {
                        self.tracker.advance_crlf();
                    } else {
                        self.tracker.advance_newline();
                    }
                    return;
                }
                _ => {
                    self.tracker.advance_byte();
                }
            }
        }
    }

    /// Skips a block comment (`/* ... */`), tracking newlines within.
    ///
    /// The current position must be at the first `/` of `/*`.
    /// Reports an error if the block comment is unterminated.
    fn skip_block_comment(&mut self) {
        let start_loc = self.tracker.current_location();

        // Skip the opening `/*`.
        self.tracker.advance_bytes(2);

        loop {
            match self.peek() {
                None => {
                    // Unterminated block comment.
                    self.diagnostics
                        .error(start_loc, "unterminated block comment");
                    return;
                }
                Some(b'*') => {
                    if self.peek_next() == Some(b'/') {
                        // End of block comment.
                        self.tracker.advance_bytes(2);
                        return;
                    }
                    self.tracker.advance_byte();
                }
                Some(b'\n') => {
                    self.tracker.advance_newline();
                }
                Some(b'\r') => {
                    if self.peek_next() == Some(b'\n') {
                        self.tracker.advance_crlf();
                    } else {
                        self.tracker.advance_newline();
                    }
                }
                _ => {
                    self.tracker.advance_byte();
                }
            }
        }
    }

    // =======================================================================
    // Helper methods — peek, advance, position queries
    // =======================================================================

    /// Returns the current byte without consuming it, or `None` at EOF.
    #[inline]
    fn peek(&self) -> Option<u8> {
        let offset = self.tracker.offset() as usize;
        self.bytes.get(offset).copied()
    }

    /// Returns the byte one position ahead of current, or `None` if unavailable.
    #[inline]
    fn peek_next(&self) -> Option<u8> {
        let offset = self.tracker.offset() as usize + 1;
        self.bytes.get(offset).copied()
    }

    /// Returns the byte `n` positions ahead of current, or `None`.
    #[inline]
    fn peek_at(&self, n: usize) -> Option<u8> {
        let offset = self.tracker.offset() as usize + n;
        self.bytes.get(offset).copied()
    }

    /// Returns `true` if all source has been consumed.
    #[inline]
    fn is_at_end(&self) -> bool {
        self.tracker.offset() as usize >= self.bytes.len()
    }

    /// Checks if the source matches the given byte pattern at the current
    /// position. Does not consume any bytes.
    fn matches_ahead(&self, pattern: &[u8]) -> bool {
        let offset = self.tracker.offset() as usize;
        if offset + pattern.len() > self.bytes.len() {
            return false;
        }
        &self.bytes[offset..offset + pattern.len()] == pattern
    }
}

// ===========================================================================
// Free function — convenience tokenization entry point
// ===========================================================================

/// Tokenizes the given preprocessed C source text into a `Vec<Token>`.
///
/// This is a convenience wrapper around [`Lexer::new`] + [`Lexer::tokenize`]
/// for one-shot tokenization.
///
/// # Arguments
///
/// - `source` — Preprocessed C source text.
/// - `file_id` — File identifier for source position tracking.
/// - `interner` — Mutable reference to the string interner.
/// - `diagnostics` — Mutable reference to the diagnostic emitter.
///
/// # Returns
///
/// A `Vec<Token>` always ending with a `TokenKind::Eof` token.
pub fn tokenize(
    source: &str,
    file_id: FileId,
    interner: &mut Interner,
    diagnostics: &mut DiagnosticEmitter,
) -> Vec<Token> {
    let mut lexer = Lexer::new(source, file_id, interner, diagnostics);
    lexer.tokenize()
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEmitter;
    use crate::common::intern::Interner;
    use crate::common::source_map::FileId;

    // -- Test helpers -------------------------------------------------------

    /// Helper that tokenizes the input string and returns the token vector.
    fn lex(input: &str) -> (Vec<Token>, DiagnosticEmitter) {
        let file_id = FileId(1);
        let mut interner = Interner::new();
        let mut diagnostics = DiagnosticEmitter::new();
        diagnostics.register_file(file_id, "test.c");
        let tokens = tokenize(input, file_id, &mut interner, &mut diagnostics);
        (tokens, diagnostics)
    }

    /// Helper that tokenizes and returns only the token kinds.
    fn lex_kinds(input: &str) -> Vec<TokenKind> {
        let (tokens, _) = lex(input);
        tokens.iter().map(|t| t.kind).collect()
    }

    /// Helper that tokenizes and asserts no errors.
    fn lex_ok(input: &str) -> Vec<Token> {
        let (tokens, diag) = lex(input);
        assert!(
            !diag.has_errors(),
            "unexpected errors while lexing: {:?}",
            diag.diagnostics()
        );
        tokens
    }

    // =======================================================================
    // Empty and whitespace-only input
    // =======================================================================

    #[test]
    fn test_empty_input() {
        let tokens = lex_ok("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_whitespace_only() {
        let tokens = lex_ok("   \t\t  \n\n  \t  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    // =======================================================================
    // Keywords
    // =======================================================================

    #[test]
    fn test_keyword_int() {
        let kinds = lex_kinds("int");
        assert_eq!(kinds, vec![TokenKind::Int, TokenKind::Eof]);
    }

    #[test]
    fn test_keyword_return() {
        let kinds = lex_kinds("return");
        assert_eq!(kinds, vec![TokenKind::Return, TokenKind::Eof]);
    }

    #[test]
    fn test_keyword_void() {
        let kinds = lex_kinds("void");
        assert_eq!(kinds, vec![TokenKind::Void, TokenKind::Eof]);
    }

    #[test]
    fn test_multiple_keywords() {
        let kinds = lex_kinds("int main void");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Int,
                TokenKind::Identifier,
                TokenKind::Void,
                TokenKind::Eof
            ]
        );
    }

    // =======================================================================
    // Identifiers
    // =======================================================================

    #[test]
    fn test_identifier_simple() {
        let tokens = lex_ok("foo");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].kind, TokenKind::Identifier);
        assert_eq!(tokens[1].kind, TokenKind::Eof);
    }

    #[test]
    fn test_identifier_with_underscore() {
        let kinds = lex_kinds("_foo_bar _123");
        assert_eq!(
            kinds,
            vec![TokenKind::Identifier, TokenKind::Identifier, TokenKind::Eof]
        );
    }

    #[test]
    fn test_identifier_interning() {
        let file_id = FileId(1);
        let mut interner = Interner::new();
        let mut diagnostics = DiagnosticEmitter::new();
        let tokens = tokenize("foo bar foo", file_id, &mut interner, &mut diagnostics);

        // First and third tokens should have same InternId.
        if let (TokenValue::Identifier(id1), TokenValue::Identifier(id2)) =
            (&tokens[0].value, &tokens[2].value)
        {
            assert_eq!(id1, id2, "same identifier should yield same InternId");
        } else {
            panic!("expected Identifier token values");
        }
    }

    // =======================================================================
    // Integer literals
    // =======================================================================

    #[test]
    fn test_integer_decimal() {
        let tokens = lex_ok("42");
        assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral);
    }

    #[test]
    fn test_integer_hex() {
        let tokens = lex_ok("0xFF");
        assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral);
    }

    #[test]
    fn test_integer_octal() {
        let tokens = lex_ok("077");
        assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral);
    }

    #[test]
    fn test_integer_binary() {
        let tokens = lex_ok("0b1010");
        assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral);
    }

    #[test]
    fn test_integer_zero() {
        let tokens = lex_ok("0");
        assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral);
    }

    // =======================================================================
    // Floating-point literals
    // =======================================================================

    #[test]
    fn test_float_with_dot() {
        let tokens = lex_ok("3.14");
        assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
    }

    #[test]
    fn test_float_leading_dot() {
        let tokens = lex_ok(".5");
        assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
    }

    #[test]
    fn test_float_with_exponent() {
        let tokens = lex_ok("1e10");
        assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
    }

    #[test]
    fn test_float_full() {
        let tokens = lex_ok("1.5e+2f");
        assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
    }

    // =======================================================================
    // String literals
    // =======================================================================

    #[test]
    fn test_string_literal() {
        let tokens = lex_ok("\"hello\"");
        assert_eq!(tokens[0].kind, TokenKind::StringLiteral);
        if let TokenValue::Str(ref s) = tokens[0].value {
            assert_eq!(s, "hello");
        } else {
            panic!("expected Str token value");
        }
    }

    #[test]
    fn test_string_with_escape() {
        let tokens = lex_ok("\"hello\\nworld\"");
        assert_eq!(tokens[0].kind, TokenKind::StringLiteral);
        if let TokenValue::Str(ref s) = tokens[0].value {
            assert_eq!(s, "hello\nworld");
        } else {
            panic!("expected Str token value");
        }
    }

    #[test]
    fn test_wide_string() {
        let tokens = lex_ok("L\"wide\"");
        assert_eq!(tokens[0].kind, TokenKind::StringLiteral);
    }

    // =======================================================================
    // Character literals
    // =======================================================================

    #[test]
    fn test_char_literal() {
        let tokens = lex_ok("'x'");
        assert_eq!(tokens[0].kind, TokenKind::CharLiteral);
        if let TokenValue::Char(v) = tokens[0].value {
            assert_eq!(v, b'x' as u32);
        } else {
            panic!("expected Char token value");
        }
    }

    #[test]
    fn test_char_escape() {
        let tokens = lex_ok("'\\n'");
        assert_eq!(tokens[0].kind, TokenKind::CharLiteral);
        if let TokenValue::Char(v) = tokens[0].value {
            assert_eq!(v, 0x0A);
        } else {
            panic!("expected Char token value");
        }
    }

    // =======================================================================
    // Operators — single character
    // =======================================================================

    #[test]
    fn test_single_char_operators() {
        let kinds = lex_kinds("+ - * / % = ! < > & | ^ ~ ? : ;");
        let expected = vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::Equal,
            TokenKind::Bang,
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::Amp,
            TokenKind::Pipe,
            TokenKind::Caret,
            TokenKind::Tilde,
            TokenKind::Question,
            TokenKind::Colon,
            TokenKind::Semicolon,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    // =======================================================================
    // Operators — two character
    // =======================================================================

    #[test]
    fn test_two_char_operators() {
        let kinds = lex_kinds("== != <= >= && || << >> += -= *= /= %= &= |= ^= ++ -- -> ##");
        let expected = vec![
            TokenKind::EqualEqual,
            TokenKind::BangEqual,
            TokenKind::LessEqual,
            TokenKind::GreaterEqual,
            TokenKind::AmpAmp,
            TokenKind::PipePipe,
            TokenKind::LessLess,
            TokenKind::GreaterGreater,
            TokenKind::PlusEqual,
            TokenKind::MinusEqual,
            TokenKind::StarEqual,
            TokenKind::SlashEqual,
            TokenKind::PercentEqual,
            TokenKind::AmpEqual,
            TokenKind::PipeEqual,
            TokenKind::CaretEqual,
            TokenKind::PlusPlus,
            TokenKind::MinusMinus,
            TokenKind::Arrow,
            TokenKind::HashHash,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    // =======================================================================
    // Operators — three character
    // =======================================================================

    #[test]
    fn test_three_char_operators() {
        let kinds = lex_kinds("<<= >>=");
        assert_eq!(
            kinds,
            vec![
                TokenKind::LessLessEqual,
                TokenKind::GreaterGreaterEqual,
                TokenKind::Eof,
            ]
        );
    }

    // =======================================================================
    // Punctuation
    // =======================================================================

    #[test]
    fn test_punctuation() {
        let kinds = lex_kinds("( ) { } [ ] ; , ...");
        let expected = vec![
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            TokenKind::Semicolon,
            TokenKind::Comma,
            TokenKind::Ellipsis,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    #[test]
    fn test_ellipsis() {
        let kinds = lex_kinds("...");
        assert_eq!(kinds, vec![TokenKind::Ellipsis, TokenKind::Eof]);
    }

    #[test]
    fn test_dot_not_ellipsis() {
        let kinds = lex_kinds(".x");
        assert_eq!(
            kinds,
            vec![TokenKind::Dot, TokenKind::Identifier, TokenKind::Eof]
        );
    }

    // =======================================================================
    // Comments
    // =======================================================================

    #[test]
    fn test_line_comment_skipped() {
        let kinds = lex_kinds("int // this is a comment\nvoid");
        assert_eq!(kinds, vec![TokenKind::Int, TokenKind::Void, TokenKind::Eof]);
    }

    #[test]
    fn test_block_comment_skipped() {
        let kinds = lex_kinds("int /* block comment */ void");
        assert_eq!(kinds, vec![TokenKind::Int, TokenKind::Void, TokenKind::Eof]);
    }

    #[test]
    fn test_block_comment_multiline() {
        let kinds = lex_kinds("int /* multi\nline\ncomment */ void");
        assert_eq!(kinds, vec![TokenKind::Int, TokenKind::Void, TokenKind::Eof]);
    }

    #[test]
    fn test_block_comment_with_star() {
        let kinds = lex_kinds("int /** docs **/ void");
        assert_eq!(kinds, vec![TokenKind::Int, TokenKind::Void, TokenKind::Eof]);
    }

    // =======================================================================
    // Newline tracking
    // =======================================================================

    #[test]
    fn test_newline_tracking() {
        let tokens = lex_ok("int\nvoid");
        // `int` starts at line 1, `void` starts at line 2.
        assert_eq!(tokens[0].span.start.line, 1);
        assert_eq!(tokens[1].span.start.line, 2);
    }

    #[test]
    fn test_column_tracking() {
        let tokens = lex_ok("  int");
        // `int` starts at column 3 (1-based, 2 spaces before it).
        assert_eq!(tokens[0].span.start.column, 3);
    }

    #[test]
    fn test_crlf_newline() {
        let tokens = lex_ok("int\r\nvoid");
        assert_eq!(tokens[0].span.start.line, 1);
        assert_eq!(tokens[1].span.start.line, 2);
    }

    // =======================================================================
    // Error recovery
    // =======================================================================

    #[test]
    fn test_unterminated_string_recovery() {
        let (tokens, diag) = lex("\"unterminated");
        assert!(diag.has_errors());
        // Should still have EOF token.
        assert!(tokens.last().map_or(false, |t| t.kind == TokenKind::Eof));
    }

    #[test]
    fn test_unterminated_block_comment_recovery() {
        let (tokens, diag) = lex("int /* never closed");
        assert!(diag.has_errors());
        assert!(tokens.last().map_or(false, |t| t.kind == TokenKind::Eof));
    }

    #[test]
    fn test_invalid_character_recovery() {
        let (tokens, diag) = lex("int @ void");
        assert!(diag.has_errors());
        // Should recover and still tokenize `void`.
        let kinds: Vec<_> = tokens.iter().map(|t| t.kind).collect();
        assert!(kinds.contains(&TokenKind::Int));
        assert!(kinds.contains(&TokenKind::Void));
        assert!(kinds.contains(&TokenKind::Eof));
    }

    // =======================================================================
    // Realistic C snippet
    // =======================================================================

    #[test]
    fn test_realistic_c_snippet() {
        let kinds = lex_kinds("int main(void) { return 0; }");
        let expected = vec![
            TokenKind::Int,
            TokenKind::Identifier, // main
            TokenKind::LeftParen,
            TokenKind::Void,
            TokenKind::RightParen,
            TokenKind::LeftBrace,
            TokenKind::Return,
            TokenKind::IntegerLiteral, // 0
            TokenKind::Semicolon,
            TokenKind::RightBrace,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    #[test]
    fn test_function_with_args() {
        let kinds = lex_kinds("int add(int a, int b) { return a + b; }");
        let expected = vec![
            TokenKind::Int,
            TokenKind::Identifier, // add
            TokenKind::LeftParen,
            TokenKind::Int,
            TokenKind::Identifier, // a
            TokenKind::Comma,
            TokenKind::Int,
            TokenKind::Identifier, // b
            TokenKind::RightParen,
            TokenKind::LeftBrace,
            TokenKind::Return,
            TokenKind::Identifier, // a
            TokenKind::Plus,
            TokenKind::Identifier, // b
            TokenKind::Semicolon,
            TokenKind::RightBrace,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    // =======================================================================
    // GCC extension keywords
    // =======================================================================

    #[test]
    fn test_gcc_attribute() {
        let kinds = lex_kinds("__attribute__");
        assert_eq!(kinds, vec![TokenKind::GccAttribute, TokenKind::Eof]);
    }

    #[test]
    fn test_gcc_typeof() {
        let kinds = lex_kinds("typeof __typeof__ __typeof");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Typeof,
                TokenKind::Typeof,
                TokenKind::Typeof,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_gcc_extension() {
        let kinds = lex_kinds("__extension__");
        assert_eq!(kinds, vec![TokenKind::GccExtension, TokenKind::Eof]);
    }

    #[test]
    fn test_gcc_asm() {
        let kinds = lex_kinds("asm __asm__ __asm");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Asm,
                TokenKind::Asm,
                TokenKind::Asm,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_gcc_builtin_va_list() {
        let kinds = lex_kinds("__builtin_va_list");
        assert_eq!(kinds, vec![TokenKind::BuiltinVaList, TokenKind::Eof]);
    }

    // =======================================================================
    // Compound assignment operators
    // =======================================================================

    #[test]
    fn test_compound_assignment_operators() {
        let kinds = lex_kinds("x += 1; y -= 2; z <<= 3; w >>= 4;");
        assert!(kinds.contains(&TokenKind::PlusEqual));
        assert!(kinds.contains(&TokenKind::MinusEqual));
        assert!(kinds.contains(&TokenKind::LessLessEqual));
        assert!(kinds.contains(&TokenKind::GreaterGreaterEqual));
    }

    // =======================================================================
    // Hash / preprocessor tokens
    // =======================================================================

    #[test]
    fn test_hash_token() {
        let kinds = lex_kinds("# ##");
        assert_eq!(
            kinds,
            vec![TokenKind::Hash, TokenKind::HashHash, TokenKind::Eof]
        );
    }

    // =======================================================================
    // Token span correctness
    // =======================================================================

    #[test]
    fn test_token_span_byte_offsets() {
        let tokens = lex_ok("int");
        // `int` at byte 0..3
        assert_eq!(tokens[0].span.start.byte_offset, 0);
        assert_eq!(tokens[0].span.end.byte_offset, 3);
    }

    #[test]
    fn test_eof_always_present() {
        let tokens = lex_ok("int");
        assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
    }

    // =======================================================================
    // Free function tokenize()
    // =======================================================================

    #[test]
    fn test_free_function_tokenize() {
        let file_id = FileId(0);
        let mut interner = Interner::new();
        let mut diagnostics = DiagnosticEmitter::new();
        let tokens = tokenize("42", file_id, &mut interner, &mut diagnostics);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral);
        assert_eq!(tokens[1].kind, TokenKind::Eof);
    }

    // =======================================================================
    // C11 keywords
    // =======================================================================

    #[test]
    fn test_c11_keywords() {
        let kinds = lex_kinds("_Alignas _Alignof _Atomic _Bool _Complex _Generic _Noreturn _Static_assert _Thread_local");
        let expected = vec![
            TokenKind::Alignas,
            TokenKind::Alignof,
            TokenKind::Atomic,
            TokenKind::Bool,
            TokenKind::Complex,
            TokenKind::Generic,
            TokenKind::Noreturn,
            TokenKind::StaticAssert,
            TokenKind::ThreadLocal,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    // =======================================================================
    // GCC alternate spellings
    // =======================================================================

    #[test]
    fn test_gcc_alternate_spellings() {
        let kinds = lex_kinds("__inline__ __volatile__ __const__ __restrict__ __signed__");
        let expected = vec![
            TokenKind::Inline,
            TokenKind::Volatile,
            TokenKind::Const,
            TokenKind::Restrict,
            TokenKind::Signed,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    // =======================================================================
    // Mixed token types
    // =======================================================================

    #[test]
    fn test_mixed_tokens() {
        let kinds = lex_kinds("if (x > 0) { y = x + 1; }");
        let expected = vec![
            TokenKind::If,
            TokenKind::LeftParen,
            TokenKind::Identifier, // x
            TokenKind::Greater,
            TokenKind::IntegerLiteral, // 0
            TokenKind::RightParen,
            TokenKind::LeftBrace,
            TokenKind::Identifier, // y
            TokenKind::Equal,
            TokenKind::Identifier, // x
            TokenKind::Plus,
            TokenKind::IntegerLiteral, // 1
            TokenKind::Semicolon,
            TokenKind::RightBrace,
            TokenKind::Eof,
        ];
        assert_eq!(kinds, expected);
    }

    // =======================================================================
    // Adjacent tokens without whitespace
    // =======================================================================

    #[test]
    fn test_adjacent_tokens() {
        let kinds = lex_kinds("a+b");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Identifier,
                TokenKind::Plus,
                TokenKind::Identifier,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_adjacent_operators() {
        let kinds = lex_kinds("a++");
        assert_eq!(
            kinds,
            vec![TokenKind::Identifier, TokenKind::PlusPlus, TokenKind::Eof]
        );
    }

    // =======================================================================
    // Struct member access
    // =======================================================================

    #[test]
    fn test_arrow_operator() {
        let kinds = lex_kinds("ptr->field");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Identifier,
                TokenKind::Arrow,
                TokenKind::Identifier,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_dot_member() {
        let kinds = lex_kinds("obj.field");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Identifier,
                TokenKind::Dot,
                TokenKind::Identifier,
                TokenKind::Eof,
            ]
        );
    }

    // =======================================================================
    // Variadic function declaration
    // =======================================================================

    #[test]
    fn test_variadic_function() {
        let kinds = lex_kinds("int printf(const char *fmt, ...);");
        assert!(kinds.contains(&TokenKind::Ellipsis));
        assert!(kinds.contains(&TokenKind::Const));
    }

    // =======================================================================
    // Tilde (bitwise NOT)
    // =======================================================================

    #[test]
    fn test_tilde_operator() {
        let kinds = lex_kinds("~x");
        assert_eq!(
            kinds,
            vec![TokenKind::Tilde, TokenKind::Identifier, TokenKind::Eof]
        );
    }
}
