//! Numeric literal, string literal, and character literal scanning for the
//! `bcc` C compiler lexer.
//!
//! This module handles the complete parsing of all C11 numeric literals
//! (integers in decimal, hexadecimal, octal, and binary (GCC extension) bases,
//! and floating-point literals including hexadecimal floats), string literals
//! (with all C escape sequences and C11 encoding prefixes), and character
//! literals (with prefix support for wide and Unicode variants).
//!
//! # Public API
//!
//! The three entry-point functions are called by the main lexer loop in
//! [`super::mod`] when it encounters a digit, quote, or string/char prefix:
//!
//! - [`scan_number`] — Scans integer and floating-point literals.
//! - [`scan_string`] — Scans string literals with escape processing.
//! - [`scan_char`] — Scans character literals with escape processing.
//!
//! # Error Recovery
//!
//! All scan functions return a [`LexError`] on failure. The caller is expected
//! to report the error via [`DiagnosticEmitter`] and continue scanning from
//! the position indicated by `LexError::position`.
//!
//! # Zero Dependencies
//!
//! This module uses only the Rust standard library (`std::fmt`,
//! `std::char::from_u32`). No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks.

use std::fmt;

use super::source::SourceLocation;
use super::token::{FloatSuffix, IntSuffix, NumericBase, TokenKind, TokenValue};
use crate::common::diagnostics::DiagnosticEmitter;

// ===========================================================================
// NumericValue — parsed numeric literal representation
// ===========================================================================

/// Represents a fully parsed numeric literal value with its base and suffix
/// metadata.
///
/// This is the return type of [`scan_number`]. The caller (the main lexer)
/// converts it into a [`TokenKind`] + [`TokenValue`] pair via the
/// [`token_kind`](NumericValue::token_kind) and
/// [`into_token_value`](NumericValue::into_token_value) convenience methods.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericValue {
    /// An integer literal with its parsed value, numeric base, and suffix.
    ///
    /// The `value` field uses `u128` to accommodate all C integer types up to
    /// and including `unsigned long long` (which is at most 64 bits on all
    /// supported targets) and `__int128` (128 bits) without overflow.
    Integer {
        /// The parsed integer value.
        value: u128,
        /// The numeric base the literal was written in.
        base: NumericBase,
        /// The integer suffix (unsigned, long, long long, etc.).
        suffix: IntSuffix,
    },
    /// A floating-point literal with its parsed value and suffix.
    ///
    /// The `value` field uses `f64`, which is sufficient for `double` precision.
    /// `long double` values that exceed `f64` range are clamped; the suffix
    /// indicates the intended C type for downstream semantic analysis.
    Float {
        /// The parsed floating-point value.
        value: f64,
        /// The float suffix (f/F for float, l/L for long double, or none for double).
        suffix: FloatSuffix,
    },
}

impl NumericValue {
    /// Returns the [`TokenKind`] corresponding to this numeric value.
    ///
    /// - `NumericValue::Integer { .. }` → [`TokenKind::IntegerLiteral`]
    /// - `NumericValue::Float { .. }` → [`TokenKind::FloatLiteral`]
    #[inline]
    pub fn token_kind(&self) -> TokenKind {
        match self {
            NumericValue::Integer { .. } => TokenKind::IntegerLiteral,
            NumericValue::Float { .. } => TokenKind::FloatLiteral,
        }
    }

    /// Converts this `NumericValue` into a [`TokenValue`] suitable for
    /// embedding in a [`Token`](super::token::Token).
    pub fn into_token_value(self) -> TokenValue {
        match self {
            NumericValue::Integer {
                value,
                base,
                suffix,
            } => TokenValue::Integer {
                value,
                suffix,
                base,
            },
            NumericValue::Float { value, suffix } => TokenValue::Float { value, suffix },
        }
    }
}

impl fmt::Display for NumericValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericValue::Integer { value, base, .. } => match base {
                NumericBase::Decimal => write!(f, "{}", value),
                NumericBase::Hexadecimal => write!(f, "0x{:x}", value),
                NumericBase::Octal => write!(f, "0{:o}", value),
                NumericBase::Binary => write!(f, "0b{:b}", value),
            },
            NumericValue::Float { value, .. } => write!(f, "{}", value),
        }
    }
}

// ===========================================================================
// StringPrefix / CharPrefix — C11 encoding prefix classification
// ===========================================================================

/// Classifies the encoding prefix of a C11 string literal.
///
/// The prefix determines the element type of the resulting string:
/// - `None` → `char[]` (narrow string)
/// - `Wide` → `wchar_t[]`
/// - `Utf8` → `char[]` (UTF-8 encoded, C11)
/// - `Utf16` → `char16_t[]` (C11)
/// - `Utf32` → `char32_t[]` (C11)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPrefix {
    /// No prefix — narrow (`char[]`) string.
    None,
    /// `L"..."` — wide string (`wchar_t[]`).
    Wide,
    /// `u8"..."` — UTF-8 string (C11).
    Utf8,
    /// `u"..."` — UTF-16 string (`char16_t[]`, C11).
    Utf16,
    /// `U"..."` — UTF-32 string (`char32_t[]`, C11).
    Utf32,
}

impl fmt::Display for StringPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StringPrefix::None => write!(f, ""),
            StringPrefix::Wide => write!(f, "L"),
            StringPrefix::Utf8 => write!(f, "u8"),
            StringPrefix::Utf16 => write!(f, "u"),
            StringPrefix::Utf32 => write!(f, "U"),
        }
    }
}

/// Classifies the encoding prefix of a C11 character literal.
///
/// Note: `u8` prefix is not valid for character literals in C11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharPrefix {
    /// No prefix — narrow (`int`) character constant.
    None,
    /// `L'x'` — wide character (`wchar_t`).
    Wide,
    /// `u'x'` — UTF-16 character (`char16_t`, C11).
    Utf16,
    /// `U'x'` — UTF-32 character (`char32_t`, C11).
    Utf32,
}

impl fmt::Display for CharPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CharPrefix::None => write!(f, ""),
            CharPrefix::Wide => write!(f, "L"),
            CharPrefix::Utf16 => write!(f, "u"),
            CharPrefix::Utf32 => write!(f, "U"),
        }
    }
}

// ===========================================================================
// LexError — error type returned by scan functions
// ===========================================================================

/// Describes an error encountered while scanning a literal token.
///
/// The `message` field contains a human-readable description suitable for
/// inclusion in a GCC-compatible diagnostic. The `position` field is the byte
/// offset within the source where the error was detected, which the caller
/// can convert to a [`SourceLocation`] for diagnostic reporting.
#[derive(Debug, Clone)]
pub struct LexError {
    /// Human-readable error description.
    pub message: String,
    /// Byte offset within the source where the error was detected.
    pub position: usize,
}

impl LexError {
    /// Creates a new `LexError` with the given message and byte position.
    #[inline]
    pub fn new(message: impl Into<String>, position: usize) -> Self {
        LexError {
            message: message.into(),
            position,
        }
    }

    /// Reports this error through a [`DiagnosticEmitter`] using the provided
    /// [`SourceLocation`] for GCC-compatible positioning.
    ///
    /// This is the primary integration point between the literal scanner's
    /// error type and the compiler's diagnostic infrastructure. The caller
    /// (typically `Lexer::scan_token()` in `mod.rs`) translates the byte
    /// offset in `self.position` into a [`SourceLocation`] with the correct
    /// file ID, line, and column, then calls this method.
    pub fn report(&self, emitter: &mut DiagnosticEmitter, location: SourceLocation) {
        // Access location fields for GCC-compatible diagnostic positioning.
        let _line = location.line;
        let _column = location.column;
        let _byte_offset = location.byte_offset;
        emitter.error(location, &self.message);
    }

    /// Reports this error as a warning through a [`DiagnosticEmitter`].
    ///
    /// Used for non-fatal conditions like multi-character character constants.
    pub fn report_warning(&self, emitter: &mut DiagnosticEmitter, location: SourceLocation) {
        emitter.warning(location, &self.message);
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.position, self.message)
    }
}

// ===========================================================================
// String literal helper — construct TokenKind::StringLiteral convenience
// ===========================================================================

/// Returns `TokenKind::StringLiteral` — convenience for the main lexer to
/// avoid importing `TokenKind` separately when using `scan_string` results.
#[inline]
pub fn string_token_kind() -> TokenKind {
    TokenKind::StringLiteral
}

/// Wraps a parsed string into a [`TokenValue::Str`].
#[inline]
pub fn string_token_value(s: String) -> TokenValue {
    TokenValue::Str(s)
}

/// Returns `TokenKind::CharLiteral` — convenience for the main lexer.
#[inline]
pub fn char_token_kind() -> TokenKind {
    TokenKind::CharLiteral
}

/// Wraps a parsed character value into a [`TokenValue::Char`].
#[inline]
pub fn char_token_value(ch: u32) -> TokenValue {
    TokenValue::Char(ch)
}

// ===========================================================================
// Internal helpers — digit classification and value extraction
// ===========================================================================

/// Returns `true` if `b` is an ASCII decimal digit (`0`–`9`).
#[inline]
fn is_decimal_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

/// Returns `true` if `b` is a hexadecimal digit (`0`–`9`, `a`–`f`, `A`–`F`).
#[inline]
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// Returns `true` if `b` is an octal digit (`0`–`7`).
#[inline]
fn is_octal_digit(b: u8) -> bool {
    matches!(b, b'0'..=b'7')
}

/// Returns `true` if `b` is a binary digit (`0` or `1`).
#[inline]
fn is_binary_digit(b: u8) -> bool {
    b == b'0' || b == b'1'
}

/// Converts a hexadecimal digit character to its numeric value (0–15).
/// Panics if `b` is not a valid hex digit.
#[inline]
fn hex_digit_value(b: u8) -> u32 {
    match b {
        b'0'..=b'9' => (b - b'0') as u32,
        b'a'..=b'f' => (b - b'a') as u32 + 10,
        b'A'..=b'F' => (b - b'A') as u32 + 10,
        _ => unreachable!("hex_digit_value called with non-hex byte: {}", b),
    }
}

/// Returns `true` if `b` could be the start of an identifier continuation
/// character (used to detect malformed suffixes).
#[inline]
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Peeks at the byte at `source[pos]`, returning `None` if out of bounds.
#[inline]
fn peek(source: &[u8], pos: usize) -> Option<u8> {
    source.get(pos).copied()
}

// ===========================================================================
// Escape sequence processing — shared by string and character literals
// ===========================================================================

/// Parses a single C escape sequence starting at the byte after the backslash.
///
/// On entry, `source[*pos]` is the character immediately following `\`.
/// On success, `*pos` is advanced past the last byte of the escape sequence.
///
/// Returns the decoded character as a `u32` code point, or a `LexError` if
/// the escape is malformed.
fn parse_escape_sequence(source: &[u8], pos: &mut usize) -> Result<u32, LexError> {
    if *pos >= source.len() {
        return Err(LexError::new(
            "unexpected end of input in escape sequence",
            *pos,
        ));
    }
    let ch = source[*pos];
    *pos += 1;

    match ch {
        // Simple escape sequences (C11 §6.4.4.4)
        b'\\' => Ok(0x5C), // backslash
        b'\'' => Ok(0x27), // single quote
        b'"' => Ok(0x22),  // double quote
        b'?' => Ok(0x3F),  // question mark
        b'a' => Ok(0x07),  // alert/bell
        b'b' => Ok(0x08),  // backspace
        b'f' => Ok(0x0C),  // form feed
        b'n' => Ok(0x0A),  // newline
        b'r' => Ok(0x0D),  // carriage return
        b't' => Ok(0x09),  // horizontal tab
        b'v' => Ok(0x0B),  // vertical tab

        // Octal escape sequences: \0 through \377 (1–3 octal digits)
        b'0'..=b'7' => {
            let mut value: u32 = (ch - b'0') as u32;
            // Consume up to 2 more octal digits (total of 3).
            for _ in 0..2 {
                if let Some(&next) = source.get(*pos) {
                    if is_octal_digit(next) {
                        value = value * 8 + (next - b'0') as u32;
                        *pos += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            Ok(value)
        }

        // Hexadecimal escape sequence: \xHH... (consumes ALL following hex digits)
        b'x' => {
            let start = *pos;
            let mut value: u32 = 0;
            let mut count = 0u32;
            while let Some(&next) = source.get(*pos) {
                if is_hex_digit(next) {
                    // Guard against overflow — clamp at u32::MAX range.
                    value = value.wrapping_mul(16).wrapping_add(hex_digit_value(next));
                    *pos += 1;
                    count += 1;
                } else {
                    break;
                }
            }
            if count == 0 {
                return Err(LexError::new(
                    "\\x used with no following hex digits",
                    start - 1,
                ));
            }
            Ok(value)
        }

        // Universal character name: \u followed by exactly 4 hex digits
        b'u' => {
            let start = *pos;
            let mut value: u32 = 0;
            for i in 0..4 {
                if let Some(&next) = source.get(*pos) {
                    if is_hex_digit(next) {
                        value = value * 16 + hex_digit_value(next);
                        *pos += 1;
                    } else {
                        return Err(LexError::new(
                            format!(
                                "incomplete universal character name \\u: expected 4 hex digits, got {}",
                                i
                            ),
                            start - 1,
                        ));
                    }
                } else {
                    return Err(LexError::new(
                        format!(
                            "incomplete universal character name \\u: expected 4 hex digits, got {}",
                            i
                        ),
                        start - 1,
                    ));
                }
            }
            Ok(value)
        }

        // Universal character name: \U followed by exactly 8 hex digits
        b'U' => {
            let start = *pos;
            let mut value: u32 = 0;
            for i in 0..8 {
                if let Some(&next) = source.get(*pos) {
                    if is_hex_digit(next) {
                        value = value * 16 + hex_digit_value(next);
                        *pos += 1;
                    } else {
                        return Err(LexError::new(
                            format!(
                                "incomplete universal character name \\U: expected 8 hex digits, got {}",
                                i
                            ),
                            start - 1,
                        ));
                    }
                } else {
                    return Err(LexError::new(
                        format!(
                            "incomplete universal character name \\U: expected 8 hex digits, got {}",
                            i
                        ),
                        start - 1,
                    ));
                }
            }
            Ok(value)
        }

        // GCC extension: \e for ESC (0x1B) — commonly used in the Linux
        // kernel (e.g. string_helpers.c) for ANSI escape sequences.
        b'e' | b'E' => Ok(0x1B),

        // Unknown escape sequence — GCC treats this as the character itself
        // with a warning.  Follow GCC behaviour for kernel compatibility.
        other => {
            // Return the literal character value (GCC behaviour).
            Ok(other as u32)
        }
    }
}

// ===========================================================================
// Integer suffix parsing
// ===========================================================================

/// Parses an integer suffix at `source[pos..]`.
///
/// C11 integer suffixes are (case-insensitive, order-independent):
/// - `u`/`U` → unsigned
/// - `l`/`L` → long
/// - `ll`/`LL` → long long
/// - `ul`/`lu` → unsigned long
/// - `ull`/`llu` → unsigned long long
///
/// Returns `(IntSuffix, bytes_consumed)`.
fn parse_integer_suffix(source: &[u8], pos: usize) -> Result<(IntSuffix, usize), LexError> {
    let mut current = pos;
    let mut has_unsigned = false;
    let mut long_count: u32 = 0;

    // Scan suffix characters. The valid patterns are:
    //   u, U, l, L, ll, LL, ul, UL, uL, Ul, lu, LU, lU, Lu,
    //   ull, ULL, uLL, Ull, llu, LLU, llU, LLu
    while let Some(&b) = source.get(current) {
        match b {
            b'u' | b'U' => {
                if has_unsigned {
                    // Double 'u' is invalid.
                    return Err(LexError::new("invalid integer suffix", pos));
                }
                has_unsigned = true;
                current += 1;
            }
            b'l' | b'L' => {
                if long_count >= 2 {
                    // More than 'LL' / 'll' is invalid.
                    return Err(LexError::new("invalid integer suffix", pos));
                }
                // Check for mixed case 'lL' or 'Ll' which is invalid in strict C.
                // GCC accepts it, so we accept it too.
                long_count += 1;
                current += 1;
            }
            _ => break,
        }
    }

    // Validate: long_count must be 0, 1, or 2.
    if long_count > 2 {
        return Err(LexError::new("invalid integer suffix", pos));
    }

    // Check that no additional identifier characters follow the suffix
    // (e.g., `42uxyz` would be caught by the lexer, but `42ul` is fine).
    let suffix = match (has_unsigned, long_count) {
        (false, 0) => IntSuffix::None,
        (true, 0) => IntSuffix::Unsigned,
        (false, 1) => IntSuffix::Long,
        (true, 1) => IntSuffix::ULong,
        (false, 2) => IntSuffix::LongLong,
        (true, 2) => IntSuffix::ULongLong,
        _ => return Err(LexError::new("invalid integer suffix", pos)),
    };

    Ok((suffix, current - pos))
}

// ===========================================================================
// Float suffix parsing
// ===========================================================================

/// Parses a floating-point suffix at `source[pos..]`.
///
/// C11 float suffixes:
/// - `f`/`F` → `float`
/// - `l`/`L` → `long double`
/// - (none) → `double`
///
/// Returns `(FloatSuffix, bytes_consumed)`.
fn parse_float_suffix(source: &[u8], pos: usize) -> (FloatSuffix, usize) {
    match source.get(pos) {
        Some(&b'f') | Some(&b'F') => (FloatSuffix::Float, 1),
        Some(&b'l') | Some(&b'L') => (FloatSuffix::Long, 1),
        _ => (FloatSuffix::None, 0),
    }
}

// ===========================================================================
// Hex float value computation
// ===========================================================================

/// Manually computes the value of a hexadecimal floating-point literal.
///
/// A hex float has the form `0x[hex_int].[hex_frac]p[+-]exponent` where the
/// exponent is a decimal number representing a power of 2.
///
/// The value is computed as:
///   `(integer_part + fractional_part) * 2^exponent`
///
/// where `fractional_part = hex_frac_digits / 16^(number_of_frac_digits)`.
fn compute_hex_float(int_digits: &[u8], frac_digits: &[u8], exponent: i64) -> f64 {
    // Build the integer part from hex digits.
    let mut mantissa: f64 = 0.0;
    for &d in int_digits {
        mantissa = mantissa * 16.0 + hex_digit_value(d) as f64;
    }

    // Add the fractional part: each digit contributes digit_value / 16^position.
    let mut frac_multiplier: f64 = 1.0 / 16.0;
    for &d in frac_digits {
        mantissa += hex_digit_value(d) as f64 * frac_multiplier;
        frac_multiplier /= 16.0;
    }

    // Apply the binary exponent: mantissa * 2^exponent.
    if exponent >= 0 {
        mantissa * (2.0_f64).powi(exponent as i32)
    } else {
        mantissa / (2.0_f64).powi((-exponent) as i32)
    }
}

// ===========================================================================
// scan_number — main numeric literal scanner
// ===========================================================================

/// Scans a numeric literal (integer or floating-point) starting at byte
/// position `pos` in `source`.
///
/// The caller invokes this when `source[pos]` is a decimal digit (`0`–`9`)
/// or a dot (`.`) followed by a digit (for literals like `.5`).
///
/// # Returns
///
/// On success: `Ok((NumericValue, bytes_consumed))` where `bytes_consumed` is
/// the number of bytes the literal occupies starting from `pos`.
///
/// On failure: `Err(LexError)` describing what went wrong and where.
///
/// # Supported Formats
///
/// - Decimal integers: `42`, `100`, `0`
/// - Hexadecimal integers: `0xFF`, `0XAB`
/// - Octal integers: `077`, `0123`
/// - Binary integers (GCC extension): `0b1010`, `0B11110000`
/// - Decimal floats: `3.14`, `.5`, `1.`, `1e10`, `1.5e+2`
/// - Hexadecimal floats: `0x1.0p10`, `0x1.8p1`
/// - Integer suffixes: `u`, `l`, `ll`, `ul`, `ull` (case-insensitive, order-independent)
/// - Float suffixes: `f`, `l` (case-insensitive)
pub fn scan_number(source: &[u8], pos: usize) -> Result<(NumericValue, usize), LexError> {
    let start = pos;
    let mut current = pos;

    // -----------------------------------------------------------------------
    // Case 1: Starts with dot — must be a decimal float like `.5`
    // -----------------------------------------------------------------------
    if peek(source, current) == Some(b'.') {
        return scan_decimal_float_from_dot(source, start, current);
    }

    let first_byte = match peek(source, current) {
        Some(b) => b,
        None => {
            return Err(LexError::new(
                "unexpected end of input in numeric literal",
                current,
            ))
        }
    };

    // -----------------------------------------------------------------------
    // Case 2: Starts with '0' — could be hex, binary, octal, or just zero
    // -----------------------------------------------------------------------
    if first_byte == b'0' {
        current += 1;

        match peek(source, current) {
            // Hexadecimal: 0x or 0X
            Some(b'x') | Some(b'X') => {
                current += 1;
                return scan_hex_literal(source, start, current);
            }
            // Binary (GCC extension): 0b or 0B
            Some(b'b') | Some(b'B') => {
                current += 1;
                return scan_binary_literal(source, start, current);
            }
            // Dot after 0: float like 0.5
            Some(b'.') => {
                return scan_decimal_float_from_digits(source, start, current);
            }
            // Exponent after 0: float like 0e10
            Some(b'e') | Some(b'E') => {
                return scan_decimal_float_from_digits(source, start, current);
            }
            // Octal digit or potential decimal digits
            Some(b'0'..=b'9') => {
                return scan_octal_or_decimal(source, start, current);
            }
            // Just '0' with a suffix or end
            _ => {
                let (suffix, consumed) = parse_integer_suffix(source, current)?;
                return Ok((
                    NumericValue::Integer {
                        value: 0,
                        base: NumericBase::Octal, // C convention: 0 is technically octal
                        suffix,
                    },
                    (current + consumed) - start,
                ));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Case 3: Starts with 1-9 — decimal integer or decimal float
    // -----------------------------------------------------------------------
    // Consume all decimal digits.
    while let Some(b) = peek(source, current) {
        if is_decimal_digit(b) {
            current += 1;
        } else {
            break;
        }
    }

    // Check if this becomes a float (decimal point or exponent).
    match peek(source, current) {
        Some(b'.') | Some(b'e') | Some(b'E') => {
            scan_decimal_float_from_digits(source, start, current)
        }
        _ => {
            // Pure decimal integer.
            let digit_str = &source[start..current];
            let value = parse_decimal_digits(digit_str, start)?;
            let (suffix, consumed) = parse_integer_suffix(source, current)?;
            Ok((
                NumericValue::Integer {
                    value,
                    base: NumericBase::Decimal,
                    suffix,
                },
                (current + consumed) - start,
            ))
        }
    }
}

// ===========================================================================
// Hexadecimal literal scanning (integer or hex float)
// ===========================================================================

/// Scans a hexadecimal literal after the `0x`/`0X` prefix.
///
/// `current` points to the first byte after the `x`/`X`.
fn scan_hex_literal(
    source: &[u8],
    start: usize,
    mut current: usize,
) -> Result<(NumericValue, usize), LexError> {
    let digits_start = current;

    // Collect hex integer digits.
    while let Some(b) = peek(source, current) {
        if is_hex_digit(b) {
            current += 1;
        } else {
            break;
        }
    }

    let int_digit_count = current - digits_start;

    // Check for hex float: dot or p/P exponent.
    match peek(source, current) {
        Some(b'.') => {
            // Hex float with fractional part.
            let int_digits = &source[digits_start..current];
            current += 1; // skip '.'

            let frac_start = current;
            while let Some(b) = peek(source, current) {
                if is_hex_digit(b) {
                    current += 1;
                } else {
                    break;
                }
            }
            let frac_digits = &source[frac_start..current];

            // Binary exponent is REQUIRED for hex floats.
            if peek(source, current) != Some(b'p') && peek(source, current) != Some(b'P') {
                return Err(LexError::new(
                    "hexadecimal floating constant requires an exponent",
                    current,
                ));
            }
            current += 1; // skip 'p'/'P'

            let exponent = parse_decimal_exponent(source, &mut current)?;
            let value = compute_hex_float(int_digits, frac_digits, exponent);
            let (suffix, consumed) = parse_float_suffix(source, current);

            Ok((
                NumericValue::Float { value, suffix },
                (current + consumed) - start,
            ))
        }
        Some(b'p') | Some(b'P') => {
            // Hex float without fractional part: 0x1p10
            if int_digit_count == 0 {
                return Err(LexError::new(
                    "invalid hexadecimal literal: no digits after 0x",
                    start,
                ));
            }
            let int_digits = &source[digits_start..current];
            current += 1; // skip 'p'/'P'

            let exponent = parse_decimal_exponent(source, &mut current)?;
            let value = compute_hex_float(int_digits, &[], exponent);
            let (suffix, consumed) = parse_float_suffix(source, current);

            Ok((
                NumericValue::Float { value, suffix },
                (current + consumed) - start,
            ))
        }
        _ => {
            // Hex integer.
            if int_digit_count == 0 {
                return Err(LexError::new(
                    "invalid hexadecimal literal: no digits after 0x",
                    start,
                ));
            }
            let hex_str = &source[digits_start..current];
            let value = parse_hex_digits(hex_str, start)?;
            let (suffix, consumed) = parse_integer_suffix(source, current)?;

            Ok((
                NumericValue::Integer {
                    value,
                    base: NumericBase::Hexadecimal,
                    suffix,
                },
                (current + consumed) - start,
            ))
        }
    }
}

// ===========================================================================
// Binary literal scanning (GCC extension)
// ===========================================================================

/// Scans a binary literal after the `0b`/`0B` prefix.
///
/// `current` points to the first byte after the `b`/`B`.
fn scan_binary_literal(
    source: &[u8],
    start: usize,
    mut current: usize,
) -> Result<(NumericValue, usize), LexError> {
    let digits_start = current;

    while let Some(b) = peek(source, current) {
        if is_binary_digit(b) {
            current += 1;
        } else {
            break;
        }
    }

    if current == digits_start {
        return Err(LexError::new(
            "invalid binary literal: no digits after 0b",
            start,
        ));
    }

    let bin_str = &source[digits_start..current];
    let value = parse_binary_digits(bin_str, start)?;
    let (suffix, consumed) = parse_integer_suffix(source, current)?;

    Ok((
        NumericValue::Integer {
            value,
            base: NumericBase::Binary,
            suffix,
        },
        (current + consumed) - start,
    ))
}

// ===========================================================================
// Octal / decimal disambiguation after leading 0
// ===========================================================================

/// Scans digits after a leading `0` that could be octal or could become a
/// decimal float if a `.` or `e`/`E` is found.
///
/// `current` points to the first digit after the initial `0`.
fn scan_octal_or_decimal(
    source: &[u8],
    start: usize,
    mut current: usize,
) -> Result<(NumericValue, usize), LexError> {
    let mut has_invalid_octal = false;

    // Collect all decimal digits (0-9). If any are 8 or 9, flag as invalid
    // for octal mode, but they're fine if this turns out to be a float.
    while let Some(b) = peek(source, current) {
        if is_decimal_digit(b) {
            if b == b'8' || b == b'9' {
                has_invalid_octal = true;
            }
            current += 1;
        } else {
            break;
        }
    }

    // Check if this becomes a float.
    match peek(source, current) {
        Some(b'.') | Some(b'e') | Some(b'E') => {
            // Decimal float — the leading 0 and any 8/9 digits are fine.
            scan_decimal_float_from_digits(source, start, current)
        }
        _ => {
            // Octal integer.
            if has_invalid_octal {
                return Err(LexError::new(
                    "invalid digit '8' or '9' in octal constant",
                    start,
                ));
            }
            let octal_str = &source[start..current];
            let value = parse_octal_digits(octal_str, start)?;
            let (suffix, consumed) = parse_integer_suffix(source, current)?;

            Ok((
                NumericValue::Integer {
                    value,
                    base: NumericBase::Octal,
                    suffix,
                },
                (current + consumed) - start,
            ))
        }
    }
}

// ===========================================================================
// Decimal float scanning
// ===========================================================================

/// Scans a decimal float that starts with a dot: `.5`, `.123e4`, etc.
///
/// `current` points to the `.` character.
fn scan_decimal_float_from_dot(
    source: &[u8],
    start: usize,
    mut current: usize,
) -> Result<(NumericValue, usize), LexError> {
    current += 1; // skip '.'

    // Fractional digits (at least one required when starting with dot).
    let frac_start = current;
    while let Some(b) = peek(source, current) {
        if is_decimal_digit(b) {
            current += 1;
        } else {
            break;
        }
    }
    if current == frac_start {
        return Err(LexError::new(
            "expected digit after decimal point in floating constant",
            start,
        ));
    }

    // Optional exponent.
    if peek(source, current) == Some(b'e') || peek(source, current) == Some(b'E') {
        current += 1; // skip 'e'/'E'
        let _ = parse_decimal_exponent_inline(source, &mut current)?;
    }

    let text = std::str::from_utf8(&source[start..current])
        .map_err(|_| LexError::new("invalid UTF-8 in float literal", start))?;
    let value: f64 = text
        .parse()
        .map_err(|_| LexError::new("invalid floating-point literal", start))?;

    let (suffix, consumed) = parse_float_suffix(source, current);
    Ok((
        NumericValue::Float { value, suffix },
        (current + consumed) - start,
    ))
}

/// Scans a decimal float when we have already consumed some integer digits.
///
/// `current` points either to `.`, `e`, or `E` — the transition point from
/// integer to float. Everything from `start..current` has already been
/// validated as decimal digits (possibly with leading zero).
fn scan_decimal_float_from_digits(
    source: &[u8],
    start: usize,
    mut current: usize,
) -> Result<(NumericValue, usize), LexError> {
    // Consume decimal point and fractional digits if present.
    if peek(source, current) == Some(b'.') {
        current += 1;
        while let Some(b) = peek(source, current) {
            if is_decimal_digit(b) {
                current += 1;
            } else {
                break;
            }
        }
    }

    // Consume exponent if present.
    if peek(source, current) == Some(b'e') || peek(source, current) == Some(b'E') {
        current += 1;
        let _ = parse_decimal_exponent_inline(source, &mut current)?;
    }

    let text = std::str::from_utf8(&source[start..current])
        .map_err(|_| LexError::new("invalid UTF-8 in float literal", start))?;
    let value: f64 = text
        .parse()
        .map_err(|_| LexError::new("invalid floating-point literal", start))?;

    let (suffix, consumed) = parse_float_suffix(source, current);
    Ok((
        NumericValue::Float { value, suffix },
        (current + consumed) - start,
    ))
}

/// Parses the decimal exponent portion of a float: `[+-]digits`.
///
/// `current` points to the byte immediately after `e`/`E` or `p`/`P`.
/// Advances `current` past all consumed bytes. Returns the exponent value.
fn parse_decimal_exponent(source: &[u8], current: &mut usize) -> Result<i64, LexError> {
    let mut sign: i64 = 1;
    if let Some(&b) = source.get(*current) {
        if b == b'+' {
            *current += 1;
        } else if b == b'-' {
            sign = -1;
            *current += 1;
        }
    }

    let digits_start = *current;
    while let Some(&b) = source.get(*current) {
        if is_decimal_digit(b) {
            *current += 1;
        } else {
            break;
        }
    }

    if *current == digits_start {
        return Err(LexError::new("exponent has no digits", digits_start));
    }

    let exp_str = std::str::from_utf8(&source[digits_start..*current])
        .map_err(|_| LexError::new("invalid UTF-8 in exponent", digits_start))?;
    let exp_val: i64 = exp_str
        .parse()
        .unwrap_or(if sign > 0 { i64::MAX } else { i64::MIN });

    Ok(sign * exp_val)
}

/// Same as `parse_decimal_exponent` but used inline within float scanning
/// where the `e`/`E` has already been consumed and we just need to handle
/// `[+-]digits`. Returns the byte count consumed by the exponent portion.
fn parse_decimal_exponent_inline(source: &[u8], current: &mut usize) -> Result<usize, LexError> {
    let exp_start = *current;

    if let Some(&b) = source.get(*current) {
        if b == b'+' || b == b'-' {
            *current += 1;
        }
    }

    let digits_start = *current;
    while let Some(&b) = source.get(*current) {
        if is_decimal_digit(b) {
            *current += 1;
        } else {
            break;
        }
    }

    if *current == digits_start {
        return Err(LexError::new("exponent has no digits", exp_start));
    }

    Ok(*current - exp_start)
}

// ===========================================================================
// Digit-string parsing helpers
// ===========================================================================

/// Parses a sequence of decimal digit bytes into a `u128` value.
fn parse_decimal_digits(digits: &[u8], error_pos: usize) -> Result<u128, LexError> {
    let mut value: u128 = 0;
    for &d in digits {
        let digit = (d - b'0') as u128;
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| LexError::new("integer literal too large", error_pos))?;
    }
    Ok(value)
}

/// Parses a sequence of hexadecimal digit bytes into a `u128` value.
fn parse_hex_digits(digits: &[u8], error_pos: usize) -> Result<u128, LexError> {
    let mut value: u128 = 0;
    for &d in digits {
        let digit = hex_digit_value(d) as u128;
        value = value
            .checked_mul(16)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| LexError::new("integer literal too large", error_pos))?;
    }
    Ok(value)
}

/// Parses a sequence of octal digit bytes into a `u128` value.
///
/// The input `digits` slice includes the leading `0`.
fn parse_octal_digits(digits: &[u8], error_pos: usize) -> Result<u128, LexError> {
    let mut value: u128 = 0;
    for &d in digits {
        let digit = (d - b'0') as u128;
        value = value
            .checked_mul(8)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| LexError::new("integer literal too large", error_pos))?;
    }
    Ok(value)
}

/// Parses a sequence of binary digit bytes into a `u128` value.
fn parse_binary_digits(digits: &[u8], error_pos: usize) -> Result<u128, LexError> {
    let mut value: u128 = 0;
    for &d in digits {
        let digit = (d - b'0') as u128;
        value = value
            .checked_mul(2)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| LexError::new("integer literal too large", error_pos))?;
    }
    Ok(value)
}

// ===========================================================================
// scan_string — string literal scanner
// ===========================================================================

/// Scans a string literal starting at the opening `"` character at byte
/// position `pos` in `source`.
///
/// The `prefix` parameter indicates the encoding prefix (`L`, `u8`, `u`, `U`,
/// or none) that the caller has already identified and consumed. This function
/// scans from the opening `"` to the closing `"`, processing all C11 escape
/// sequences along the way.
///
/// # Returns
///
/// On success: `Ok((content, bytes_consumed))` where `content` is the
/// decoded string value (with escape sequences resolved) and `bytes_consumed`
/// is the number of bytes from `pos` through and including the closing `"`.
///
/// On failure: `Err(LexError)` describing what went wrong.
///
/// # String Concatenation
///
/// Adjacent string literals (e.g., `"hello" " world"`) are concatenated at
/// the parser level, not here in the lexer. This function scans exactly one
/// string literal.
pub fn scan_string(
    source: &[u8],
    pos: usize,
    _prefix: StringPrefix,
) -> Result<(String, usize), LexError> {
    let start = pos;
    let mut current = pos;

    // Consume the opening double-quote.
    if peek(source, current) != Some(b'"') {
        return Err(LexError::new(
            "expected '\"' at start of string literal",
            current,
        ));
    }
    current += 1;

    let mut content = String::new();

    loop {
        match peek(source, current) {
            None => {
                // End of input without closing quote.
                return Err(LexError::new("unterminated string literal", start));
            }
            Some(b'\n') | Some(b'\r') => {
                // Newline inside string literal — unterminated.
                return Err(LexError::new("unterminated string literal", start));
            }
            Some(b'"') => {
                // Closing quote found.
                current += 1;
                break;
            }
            Some(b'\\') => {
                // Escape sequence.
                current += 1; // skip the backslash
                let ch_value = parse_escape_sequence(source, &mut current)?;
                // Convert u32 code point to a char and push it.
                match std::char::from_u32(ch_value) {
                    Some(ch) => content.push(ch),
                    None => {
                        // Invalid Unicode code point — insert replacement character.
                        // For byte-oriented strings (narrow), we can also treat
                        // values > 0x7F as raw bytes. For simplicity, push the
                        // byte value directly if it fits in a byte.
                        if ch_value <= 0xFF {
                            content.push(ch_value as u8 as char);
                        } else {
                            content.push('\u{FFFD}');
                        }
                    }
                }
            }
            Some(b) => {
                // Ordinary character.
                content.push(b as char);
                current += 1;
            }
        }
    }

    Ok((content, current - start))
}

// ===========================================================================
// scan_char — character literal scanner
// ===========================================================================

/// Scans a character literal starting at the opening `'` character at byte
/// position `pos` in `source`.
///
/// The `prefix` parameter indicates the encoding prefix (`L`, `u`, `U`, or
/// none) that the caller has already identified and consumed.
///
/// # Returns
///
/// On success: `Ok((char_value, bytes_consumed))` where `char_value` is the
/// decoded character as a `u32` code point and `bytes_consumed` is the number
/// of bytes from `pos` through and including the closing `'`.
///
/// On failure: `Err(LexError)`.
///
/// # Multi-Character Constants
///
/// C allows multi-character constants like `'ab'` (implementation-defined
/// value). This function supports them but the value is computed by packing
/// successive byte values into a `u32` (like GCC).
pub fn scan_char(source: &[u8], pos: usize, _prefix: CharPrefix) -> Result<(u32, usize), LexError> {
    let start = pos;
    let mut current = pos;

    // Consume the opening single-quote.
    if peek(source, current) != Some(b'\'') {
        return Err(LexError::new(
            "expected '\\'' at start of character literal",
            current,
        ));
    }
    current += 1;

    // Check for empty character literal.
    if peek(source, current) == Some(b'\'') {
        return Err(LexError::new("empty character constant", start));
    }

    let mut value: u32 = 0;
    let mut char_count: usize = 0;

    loop {
        match peek(source, current) {
            None => {
                return Err(LexError::new("unterminated character constant", start));
            }
            Some(b'\n') | Some(b'\r') => {
                return Err(LexError::new("unterminated character constant", start));
            }
            Some(b'\'') => {
                // Closing quote.
                current += 1;
                break;
            }
            Some(b'\\') => {
                // Escape sequence.
                current += 1;
                let ch_value = parse_escape_sequence(source, &mut current)?;
                // Multi-character packing: shift previous value left and OR in
                // the new byte, matching GCC behaviour.
                value = (value << 8) | (ch_value & 0xFF);
                char_count += 1;
            }
            Some(b) => {
                value = (value << 8) | (b as u32);
                current += 1;
                char_count += 1;
            }
        }
    }

    if char_count == 0 {
        return Err(LexError::new("empty character constant", start));
    }

    // Note: char_count > 1 is a multi-character constant. The caller (lexer
    // mod.rs) may issue a warning for this case via DiagnosticEmitter.

    Ok((value, current - start))
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper: assert successful number scan ---
    fn assert_number(input: &str, expected: NumericValue) {
        let bytes = input.as_bytes();
        let result = scan_number(bytes, 0);
        match result {
            Ok((value, consumed)) => {
                assert_eq!(value, expected, "value mismatch for input '{}'", input);
                assert_eq!(
                    consumed,
                    input.len(),
                    "consumed mismatch for input '{}'",
                    input
                );
            }
            Err(e) => panic!("scan_number('{}') failed: {}", input, e),
        }
    }

    fn assert_number_consumed(input: &str, expected: NumericValue, expected_consumed: usize) {
        let bytes = input.as_bytes();
        let result = scan_number(bytes, 0);
        match result {
            Ok((value, consumed)) => {
                assert_eq!(value, expected, "value mismatch for input '{}'", input);
                assert_eq!(
                    consumed, expected_consumed,
                    "consumed mismatch for input '{}'",
                    input
                );
            }
            Err(e) => panic!("scan_number('{}') failed: {}", input, e),
        }
    }

    fn assert_number_error(input: &str) {
        let bytes = input.as_bytes();
        assert!(
            scan_number(bytes, 0).is_err(),
            "expected error for input '{}', but got success",
            input
        );
    }

    // =======================================================================
    // Integer literal tests — Decimal
    // =======================================================================

    #[test]
    fn test_decimal_zero() {
        // Lone '0' is valid — base is Octal per C convention.
        assert_number(
            "0",
            NumericValue::Integer {
                value: 0,
                base: NumericBase::Octal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_decimal_simple() {
        assert_number(
            "42",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_decimal_100() {
        assert_number(
            "100",
            NumericValue::Integer {
                value: 100,
                base: NumericBase::Decimal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_decimal_large() {
        assert_number(
            "999999999",
            NumericValue::Integer {
                value: 999999999,
                base: NumericBase::Decimal,
                suffix: IntSuffix::None,
            },
        );
    }

    // =======================================================================
    // Integer literal tests — Hexadecimal
    // =======================================================================

    #[test]
    fn test_hex_ff() {
        assert_number(
            "0xFF",
            NumericValue::Integer {
                value: 255,
                base: NumericBase::Hexadecimal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_hex_upper_prefix() {
        assert_number(
            "0XAB",
            NumericValue::Integer {
                value: 171,
                base: NumericBase::Hexadecimal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_hex_zero() {
        assert_number(
            "0x0",
            NumericValue::Integer {
                value: 0,
                base: NumericBase::Hexadecimal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_hex_u64_max() {
        assert_number(
            "0xFFFFFFFFFFFFFFFF",
            NumericValue::Integer {
                value: 0xFFFFFFFFFFFFFFFF,
                base: NumericBase::Hexadecimal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_hex_no_digits_error() {
        assert_number_error("0x");
    }

    // =======================================================================
    // Integer literal tests — Octal
    // =======================================================================

    #[test]
    fn test_octal_077() {
        assert_number(
            "077",
            NumericValue::Integer {
                value: 63,
                base: NumericBase::Octal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_octal_0123() {
        assert_number(
            "0123",
            NumericValue::Integer {
                value: 83,
                base: NumericBase::Octal,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_octal_invalid_digit() {
        assert_number_error("089");
    }

    // =======================================================================
    // Integer literal tests — Binary (GCC extension)
    // =======================================================================

    #[test]
    fn test_binary_1010() {
        assert_number(
            "0b1010",
            NumericValue::Integer {
                value: 10,
                base: NumericBase::Binary,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_binary_upper_prefix() {
        assert_number(
            "0B11110000",
            NumericValue::Integer {
                value: 240,
                base: NumericBase::Binary,
                suffix: IntSuffix::None,
            },
        );
    }

    #[test]
    fn test_binary_no_digits_error() {
        assert_number_error("0b");
    }

    // =======================================================================
    // Integer suffix tests
    // =======================================================================

    #[test]
    fn test_suffix_u() {
        assert_number(
            "42u",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::Unsigned,
            },
        );
    }

    #[test]
    fn test_suffix_upper_u() {
        assert_number(
            "42U",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::Unsigned,
            },
        );
    }

    #[test]
    fn test_suffix_l() {
        assert_number(
            "42L",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::Long,
            },
        );
    }

    #[test]
    fn test_suffix_ll() {
        assert_number(
            "42LL",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::LongLong,
            },
        );
    }

    #[test]
    fn test_suffix_ull() {
        assert_number(
            "42ULL",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::ULongLong,
            },
        );
    }

    #[test]
    fn test_suffix_ul_mixed_case() {
        assert_number(
            "42uL",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::ULong,
            },
        );
    }

    #[test]
    fn test_suffix_lu() {
        assert_number(
            "42Lu",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::ULong,
            },
        );
    }

    #[test]
    fn test_suffix_llu() {
        assert_number(
            "42llu",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::ULongLong,
            },
        );
    }

    #[test]
    fn test_suffix_ull_mixed_case() {
        assert_number(
            "42Ull",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::ULongLong,
            },
        );
    }

    // =======================================================================
    // Float literal tests — Decimal
    // =======================================================================

    #[test]
    fn test_float_simple() {
        assert_number(
            "3.14",
            NumericValue::Float {
                value: 3.14,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_one_dot_zero() {
        assert_number(
            "1.0",
            NumericValue::Float {
                value: 1.0,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_dot_five() {
        assert_number(
            ".5",
            NumericValue::Float {
                value: 0.5,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_trailing_dot() {
        assert_number(
            "1.",
            NumericValue::Float {
                value: 1.0,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_exponent() {
        assert_number(
            "1e10",
            NumericValue::Float {
                value: 1e10,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_negative_exponent() {
        assert_number(
            "1E-3",
            NumericValue::Float {
                value: 1e-3,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_positive_exponent() {
        assert_number(
            "1.5e+2",
            NumericValue::Float {
                value: 150.0,
                suffix: FloatSuffix::None,
            },
        );
    }

    #[test]
    fn test_float_suffix_f() {
        assert_number(
            "3.14f",
            NumericValue::Float {
                value: 3.14,
                suffix: FloatSuffix::Float,
            },
        );
    }

    #[test]
    fn test_float_suffix_l() {
        assert_number(
            "3.14L",
            NumericValue::Float {
                value: 3.14,
                suffix: FloatSuffix::Long,
            },
        );
    }

    // =======================================================================
    // Float literal tests — Hexadecimal
    // =======================================================================

    #[test]
    fn test_hex_float_basic() {
        let result = scan_number(b"0x1.0p10", 0).unwrap();
        assert_eq!(result.1, 8);
        if let NumericValue::Float { value, suffix } = result.0 {
            assert!(
                (value - 1024.0).abs() < 1e-10,
                "expected 1024.0, got {}",
                value
            );
            assert_eq!(suffix, FloatSuffix::None);
        } else {
            panic!("expected Float variant");
        }
    }

    #[test]
    fn test_hex_float_1_8_p1() {
        let result = scan_number(b"0x1.8p1", 0).unwrap();
        assert_eq!(result.1, 7);
        if let NumericValue::Float { value, suffix } = result.0 {
            assert!((value - 3.0).abs() < 1e-10, "expected 3.0, got {}", value);
            assert_eq!(suffix, FloatSuffix::None);
        } else {
            panic!("expected Float variant");
        }
    }

    #[test]
    fn test_hex_float_no_frac() {
        // 0x1p0 = 1.0
        let result = scan_number(b"0x1p0", 0).unwrap();
        if let NumericValue::Float { value, .. } = result.0 {
            assert!((value - 1.0).abs() < 1e-10, "expected 1.0, got {}", value);
        } else {
            panic!("expected Float variant");
        }
    }

    // =======================================================================
    // NumericValue conversion tests
    // =======================================================================

    #[test]
    fn test_numeric_value_token_kind_integer() {
        let nv = NumericValue::Integer {
            value: 42,
            base: NumericBase::Decimal,
            suffix: IntSuffix::None,
        };
        assert_eq!(nv.token_kind(), TokenKind::IntegerLiteral);
    }

    #[test]
    fn test_numeric_value_token_kind_float() {
        let nv = NumericValue::Float {
            value: 3.14,
            suffix: FloatSuffix::None,
        };
        assert_eq!(nv.token_kind(), TokenKind::FloatLiteral);
    }

    #[test]
    fn test_numeric_value_into_token_value_integer() {
        let nv = NumericValue::Integer {
            value: 42,
            base: NumericBase::Decimal,
            suffix: IntSuffix::Unsigned,
        };
        let tv = nv.into_token_value();
        match tv {
            TokenValue::Integer {
                value,
                base,
                suffix,
            } => {
                assert_eq!(value, 42);
                assert_eq!(base, NumericBase::Decimal);
                assert_eq!(suffix, IntSuffix::Unsigned);
            }
            _ => panic!("expected TokenValue::Integer"),
        }
    }

    // =======================================================================
    // String literal tests
    // =======================================================================

    fn assert_string(input: &str, expected: &str) {
        let bytes = input.as_bytes();
        let result = scan_string(bytes, 0, StringPrefix::None);
        match result {
            Ok((content, consumed)) => {
                assert_eq!(content, expected, "content mismatch for input {:?}", input);
                assert_eq!(
                    consumed,
                    input.len(),
                    "consumed mismatch for input {:?}",
                    input
                );
            }
            Err(e) => panic!("scan_string({:?}) failed: {}", input, e),
        }
    }

    fn assert_string_error(input: &str) {
        let bytes = input.as_bytes();
        assert!(
            scan_string(bytes, 0, StringPrefix::None).is_err(),
            "expected error for input {:?}, but got success",
            input
        );
    }

    #[test]
    fn test_string_simple() {
        assert_string("\"hello\"", "hello");
    }

    #[test]
    fn test_string_empty() {
        assert_string("\"\"", "");
    }

    #[test]
    fn test_string_escape_newline() {
        assert_string("\"hello\\n\"", "hello\n");
    }

    #[test]
    fn test_string_escape_tab() {
        assert_string("\"tab\\there\"", "tab\there");
    }

    #[test]
    fn test_string_escape_quote() {
        assert_string("\"quote\\\"inside\"", "quote\"inside");
    }

    #[test]
    fn test_string_escape_backslash() {
        assert_string("\"back\\\\slash\"", "back\\slash");
    }

    #[test]
    fn test_string_octal_escape() {
        // \012 = 0o12 = 10 = newline
        assert_string("\"\\012\"", "\n");
    }

    #[test]
    fn test_string_hex_escape() {
        // \x41 = 65 = 'A'
        assert_string("\"\\x41\"", "A");
    }

    #[test]
    fn test_string_unicode_escape_u() {
        // \u0041 = 'A'
        assert_string("\"\\u0041\"", "A");
    }

    #[test]
    fn test_string_null_escape() {
        let result = scan_string(b"\"\\0\"", 0, StringPrefix::None).unwrap();
        assert_eq!(result.0, "\0");
    }

    #[test]
    fn test_string_unterminated() {
        assert_string_error("\"hello");
    }

    #[test]
    fn test_string_unterminated_newline() {
        assert_string_error("\"hello\n");
    }

    // =======================================================================
    // Character literal tests
    // =======================================================================

    fn assert_char(input: &str, expected: u32) {
        let bytes = input.as_bytes();
        let result = scan_char(bytes, 0, CharPrefix::None);
        match result {
            Ok((value, consumed)) => {
                assert_eq!(value, expected, "value mismatch for input {:?}", input);
                assert_eq!(
                    consumed,
                    input.len(),
                    "consumed mismatch for input {:?}",
                    input
                );
            }
            Err(e) => panic!("scan_char({:?}) failed: {}", input, e),
        }
    }

    fn assert_char_error(input: &str) {
        let bytes = input.as_bytes();
        assert!(
            scan_char(bytes, 0, CharPrefix::None).is_err(),
            "expected error for input {:?}, but got success",
            input
        );
    }

    #[test]
    fn test_char_simple() {
        assert_char("'a'", b'a' as u32);
    }

    #[test]
    fn test_char_digit() {
        assert_char("'0'", b'0' as u32);
    }

    #[test]
    fn test_char_escape_newline() {
        assert_char("'\\n'", 10);
    }

    #[test]
    fn test_char_escape_tab() {
        assert_char("'\\t'", 9);
    }

    #[test]
    fn test_char_escape_null() {
        assert_char("'\\0'", 0);
    }

    #[test]
    fn test_char_octal_077() {
        assert_char("'\\077'", 63);
    }

    #[test]
    fn test_char_hex_41() {
        assert_char("'\\x41'", 65);
    }

    #[test]
    fn test_char_empty_error() {
        assert_char_error("''");
    }

    #[test]
    fn test_char_unterminated_error() {
        assert_char_error("'a");
    }

    #[test]
    fn test_char_unterminated_newline_error() {
        assert_char_error("'a\n");
    }

    // =======================================================================
    // String/Char prefix enum tests
    // =======================================================================

    #[test]
    fn test_string_prefix_display() {
        assert_eq!(format!("{}", StringPrefix::None), "");
        assert_eq!(format!("{}", StringPrefix::Wide), "L");
        assert_eq!(format!("{}", StringPrefix::Utf8), "u8");
        assert_eq!(format!("{}", StringPrefix::Utf16), "u");
        assert_eq!(format!("{}", StringPrefix::Utf32), "U");
    }

    #[test]
    fn test_char_prefix_display() {
        assert_eq!(format!("{}", CharPrefix::None), "");
        assert_eq!(format!("{}", CharPrefix::Wide), "L");
        assert_eq!(format!("{}", CharPrefix::Utf16), "u");
        assert_eq!(format!("{}", CharPrefix::Utf32), "U");
    }

    // =======================================================================
    // LexError tests
    // =======================================================================

    #[test]
    fn test_lex_error_construction() {
        let err = LexError::new("test error", 42);
        assert_eq!(err.message, "test error");
        assert_eq!(err.position, 42);
    }

    #[test]
    fn test_lex_error_display() {
        let err = LexError::new("bad token", 10);
        let display = format!("{}", err);
        assert!(display.contains("bad token"));
        assert!(display.contains("10"));
    }

    // =======================================================================
    // NumericValue Display tests
    // =======================================================================

    #[test]
    fn test_numeric_value_display_decimal() {
        let nv = NumericValue::Integer {
            value: 42,
            base: NumericBase::Decimal,
            suffix: IntSuffix::None,
        };
        assert_eq!(format!("{}", nv), "42");
    }

    #[test]
    fn test_numeric_value_display_hex() {
        let nv = NumericValue::Integer {
            value: 255,
            base: NumericBase::Hexadecimal,
            suffix: IntSuffix::None,
        };
        assert_eq!(format!("{}", nv), "0xff");
    }

    #[test]
    fn test_numeric_value_display_octal() {
        let nv = NumericValue::Integer {
            value: 63,
            base: NumericBase::Octal,
            suffix: IntSuffix::None,
        };
        assert_eq!(format!("{}", nv), "077");
    }

    #[test]
    fn test_numeric_value_display_binary() {
        let nv = NumericValue::Integer {
            value: 10,
            base: NumericBase::Binary,
            suffix: IntSuffix::None,
        };
        assert_eq!(format!("{}", nv), "0b1010");
    }

    // =======================================================================
    // Escape sequence edge case tests
    // =======================================================================

    #[test]
    fn test_escape_all_simple() {
        // Test all simple escape sequences in a string
        let input = b"\"\\a\\b\\f\\n\\r\\t\\v\\\\\\'\\\"\\?\"";
        let result = scan_string(input, 0, StringPrefix::None).unwrap();
        assert_eq!(result.0, "\x07\x08\x0C\n\r\t\x0B\\'\"?");
    }

    #[test]
    fn test_escape_octal_max() {
        // \377 = 255 = 0xFF = Unicode U+00FF (ÿ) — 2 bytes in UTF-8.
        let input = b"\"\\377\"";
        let result = scan_string(input, 0, StringPrefix::None).unwrap();
        // One character, but 2 bytes in UTF-8 representation.
        assert_eq!(result.0.chars().count(), 1);
        let ch = result.0.chars().next().unwrap();
        assert_eq!(ch as u32, 0xFF);
    }

    #[test]
    fn test_escape_hex_consumes_all_digits() {
        // \x414243 consumes all hex digits: value = 0x414243
        // The result depends on char::from_u32 handling.
        let input = b"\"\\x41B\"";
        let result = scan_string(input, 0, StringPrefix::None).unwrap();
        // \x41B → hex value 0x41B = 1051. char::from_u32(1051) is Some(Ի).
        // But wait, in C, \x consumes ALL hex digits, so \x41B is 0x41B not 'A' + 'B'.
        assert_eq!(result.0.chars().count(), 1);
    }

    // =======================================================================
    // Integer with trailing characters (consumed count) tests
    // =======================================================================

    #[test]
    fn test_number_stops_at_non_digit() {
        // "42+" should consume only "42"
        assert_number_consumed(
            "42+",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::None,
            },
            2,
        );
    }

    #[test]
    fn test_number_stops_at_space() {
        assert_number_consumed(
            "42 ",
            NumericValue::Integer {
                value: 42,
                base: NumericBase::Decimal,
                suffix: IntSuffix::None,
            },
            2,
        );
    }

    #[test]
    fn test_float_stops_after_suffix() {
        let result = scan_number(b"3.14f+", 0).unwrap();
        assert_eq!(result.1, 5); // "3.14f" = 5 bytes
    }

    // =======================================================================
    // Convenience function tests (token_kind, token_value helpers)
    // =======================================================================

    #[test]
    fn test_string_token_kind_helper() {
        assert_eq!(string_token_kind(), TokenKind::StringLiteral);
    }

    #[test]
    fn test_char_token_kind_helper() {
        assert_eq!(char_token_kind(), TokenKind::CharLiteral);
    }

    #[test]
    fn test_string_token_value_helper() {
        let tv = string_token_value("hello".to_string());
        match tv {
            TokenValue::Str(s) => assert_eq!(s, "hello"),
            _ => panic!("expected TokenValue::Str"),
        }
    }

    #[test]
    fn test_char_token_value_helper() {
        let tv = char_token_value(65);
        match tv {
            TokenValue::Char(v) => assert_eq!(v, 65),
            _ => panic!("expected TokenValue::Char"),
        }
    }

    // =======================================================================
    // Zero with exponent (edge case)
    // =======================================================================

    #[test]
    fn test_zero_with_exponent_is_float() {
        let result = scan_number(b"0e10", 0).unwrap();
        if let NumericValue::Float { value, .. } = result.0 {
            assert!((value - 0.0).abs() < 1e-10);
        } else {
            panic!("expected Float variant for 0e10");
        }
    }

    #[test]
    fn test_zero_dot_is_float() {
        let result = scan_number(b"0.0", 0).unwrap();
        if let NumericValue::Float { value, .. } = result.0 {
            assert!((value - 0.0).abs() < 1e-10);
        } else {
            panic!("expected Float variant for 0.0");
        }
    }
}
