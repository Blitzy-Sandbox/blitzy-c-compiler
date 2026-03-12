//! Preprocessor constant expression evaluator for `#if` and `#elif` directives.
//!
//! Implements a full C11 §6.10.1 constant expression parser and evaluator supporting
//! integer arithmetic, logical and comparison operators, bitwise operators, the
//! `defined()` operator, and the ternary conditional operator.
//!
//! All arithmetic uses [`BigInt`] (128-bit capacity) for evaluation, exceeding the
//! C standard's `intmax_t` requirement of at least 64 bits.
//!
//! # C11 Semantics
//!
//! - After macro expansion, any remaining identifiers evaluate to `0` (C11 §6.10.1p4).
//! - `defined(NAME)` and `defined NAME` test whether a macro is defined.
//! - All arithmetic is integer; no floating-point.
//! - Short-circuit evaluation for `&&` and `||`.
//! - Ternary `?:` operator is right-to-left associative.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::numeric::BigInt;
use crate::common::source_map::SourceLocation;

// ===========================================================================
// Expression Token Types
// ===========================================================================

/// Tokens recognized by the preprocessor expression parser.
///
/// This is a simplified token set specific to preprocessor constant expressions.
/// It does not need the full lexer's complexity since preprocessor expressions
/// are a restricted subset of C syntax (integer-only, no function calls, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
enum ExprToken {
    /// Integer literal (decimal, hex, octal, binary).
    Integer(BigInt),
    /// Character literal stored as its integer value.
    Char(BigInt),
    /// The `defined` keyword.
    Defined,
    /// `__has_attribute(name)` — evaluates to 1 if the attribute is supported, 0 otherwise.
    HasAttribute(String),
    /// `__has_builtin(name)` — evaluates to 1 if the builtin is supported, 0 otherwise.
    HasBuiltin(String),
    /// `__has_extension(name)` — evaluates to 1 if the language extension is supported, 0 otherwise.
    HasExtension(String),
    /// `__has_include(<header>)` or `__has_include("header")` — evaluates to 1 if the header exists.
    HasInclude(String),
    /// An identifier (after macro expansion, evaluates to 0 per C11 §6.10.1p4).
    Identifier(String),

    // Arithmetic operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Bitwise operators
    Ampersand,
    Pipe,
    Caret,
    Tilde,

    // Logical operators
    Exclaim,
    AmpAmp,
    PipePipe,

    // Shift operators
    LessLess,
    GreaterGreater,

    // Comparison operators
    EqualEqual,
    ExclaimEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,

    // Ternary
    Question,
    Colon,

    // Grouping
    LeftParen,
    RightParen,

    // Other
    Comma,
}

// ===========================================================================
// Expression Tokenizer
// ===========================================================================

/// Tokenizes a preprocessor expression string into a sequence of [`ExprToken`]s.
///
/// The input is the text after the `#if` or `#elif` keyword, after macro expansion
/// has already been performed by the caller.
fn tokenize_expression(input: &str) -> Result<Vec<ExprToken>, String> {
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut pos = 0;
    let mut tokens = Vec::new();

    while pos < len {
        let ch = chars[pos];

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            pos += 1;
            continue;
        }

        // Character literal
        if ch == '\'' {
            let (value, new_pos) = parse_char_literal(&chars, pos)?;
            tokens.push(ExprToken::Char(value));
            pos = new_pos;
            continue;
        }

        // Integer literal (starts with a digit)
        if ch.is_ascii_digit() {
            let (value, new_pos) = parse_integer_literal(&chars, pos)?;
            tokens.push(ExprToken::Integer(value));
            pos = new_pos;
            continue;
        }

        // Identifier or `defined` / `__has_attribute` / `__has_builtin` keyword
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = pos;
            while pos < len && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            let word: String = chars[start..pos].iter().collect();
            if word == "defined" {
                tokens.push(ExprToken::Defined);
            } else if word == "sizeof"
                || word == "_Alignof"
                || word == "__alignof__"
                || word == "__alignof"
                || word == "alignof"
            {
                // Handle sizeof/alignof/etc. operators in preprocessor expressions.
                // These appear in kernel headers like:
                //   #define ARCH_KMALLOC_MINALIGN __alignof__(unsigned long long)
                //   #elif ARCH_KMALLOC_MINALIGN > 8
                // After macro expansion, the #elif becomes:
                //   __alignof__(unsigned long long) > 8
                // We need to evaluate this to an integer constant.
                let mut p = pos;
                while p < len && chars[p].is_ascii_whitespace() {
                    p += 1;
                }
                if p < len && chars[p] == '(' {
                    p += 1; // skip '('
                    let mut depth: u32 = 1;
                    let type_start = p;
                    while p < len && depth > 0 {
                        match chars[p] {
                            '(' => depth += 1,
                            ')' => depth -= 1,
                            _ => {}
                        }
                        if depth > 0 {
                            p += 1;
                        }
                    }
                    let type_str: String = chars[type_start..p].iter().collect();
                    let type_str = type_str.trim();
                    if p < len {
                        p += 1; // skip closing ')'
                    }
                    pos = p;
                    let is_sizeof = word == "sizeof";
                    let value = evaluate_sizeof_alignof_type(type_str, is_sizeof);
                    tokens.push(ExprToken::Integer(BigInt::from_i64(value)));
                } else {
                    // No parens — treat as identifier evaluating to 0
                    tokens.push(ExprToken::Identifier(word));
                }
            } else if word == "__has_attribute"
                || word == "__has_builtin"
                || word == "__has_extension"
                || word == "__has_include"
            {
                // Parse __has_attribute(name), __has_builtin(name),
                // __has_extension(name), or __has_include(<hdr>/"hdr") as a single token.
                // Skip whitespace between the keyword and '('.
                let mut p = pos;
                while p < len && chars[p].is_ascii_whitespace() {
                    p += 1;
                }
                if p < len && chars[p] == '(' {
                    p += 1; // consume '('
                            // Skip whitespace before the argument
                    while p < len && chars[p].is_ascii_whitespace() {
                        p += 1;
                    }
                    // For __has_include, the argument may be <header> or "header"
                    let arg_name: String;
                    if word == "__has_include" && p < len && (chars[p] == '<' || chars[p] == '"') {
                        // Parse angle-bracket or quoted header name
                        let close_ch = if chars[p] == '<' { '>' } else { '"' };
                        p += 1; // skip opening delimiter
                        let name_start = p;
                        while p < len && chars[p] != close_ch {
                            p += 1;
                        }
                        arg_name = chars[name_start..p].iter().collect();
                        if p < len {
                            p += 1; // skip closing delimiter
                        }
                    } else {
                        // Parse a plain identifier name
                        let name_start = p;
                        while p < len && (chars[p].is_ascii_alphanumeric() || chars[p] == '_') {
                            p += 1;
                        }
                        arg_name = chars[name_start..p].iter().collect();
                    }
                    // Skip whitespace after the argument
                    while p < len && chars[p].is_ascii_whitespace() {
                        p += 1;
                    }
                    if p < len && chars[p] == ')' {
                        p += 1; // consume ')'
                        pos = p;
                        match word.as_str() {
                            "__has_attribute" => tokens.push(ExprToken::HasAttribute(arg_name)),
                            "__has_builtin" => tokens.push(ExprToken::HasBuiltin(arg_name)),
                            "__has_extension" => tokens.push(ExprToken::HasExtension(arg_name)),
                            "__has_include" => tokens.push(ExprToken::HasInclude(arg_name)),
                            _ => unreachable!(),
                        }
                    } else {
                        return Err(format!("expected ')' after {}(name", word));
                    }
                } else {
                    // No parenthesis: treat as regular identifier (evaluates to 0)
                    tokens.push(ExprToken::Identifier(word));
                }
            } else {
                tokens.push(ExprToken::Identifier(word));
            }
            continue;
        }

        // Two-character operators (must check before single-character)
        if pos + 1 < len {
            let next = chars[pos + 1];
            match (ch, next) {
                ('&', '&') => {
                    tokens.push(ExprToken::AmpAmp);
                    pos += 2;
                    continue;
                }
                ('|', '|') => {
                    tokens.push(ExprToken::PipePipe);
                    pos += 2;
                    continue;
                }
                ('<', '<') => {
                    tokens.push(ExprToken::LessLess);
                    pos += 2;
                    continue;
                }
                ('>', '>') => {
                    tokens.push(ExprToken::GreaterGreater);
                    pos += 2;
                    continue;
                }
                ('=', '=') => {
                    tokens.push(ExprToken::EqualEqual);
                    pos += 2;
                    continue;
                }
                ('!', '=') => {
                    tokens.push(ExprToken::ExclaimEqual);
                    pos += 2;
                    continue;
                }
                ('<', '=') => {
                    tokens.push(ExprToken::LessEqual);
                    pos += 2;
                    continue;
                }
                ('>', '=') => {
                    tokens.push(ExprToken::GreaterEqual);
                    pos += 2;
                    continue;
                }
                _ => {}
            }
        }

        // Single-character operators and punctuation
        match ch {
            '+' => tokens.push(ExprToken::Plus),
            '-' => tokens.push(ExprToken::Minus),
            '*' => tokens.push(ExprToken::Star),
            '/' => tokens.push(ExprToken::Slash),
            '%' => tokens.push(ExprToken::Percent),
            '&' => tokens.push(ExprToken::Ampersand),
            '|' => tokens.push(ExprToken::Pipe),
            '^' => tokens.push(ExprToken::Caret),
            '~' => tokens.push(ExprToken::Tilde),
            '!' => tokens.push(ExprToken::Exclaim),
            '<' => tokens.push(ExprToken::Less),
            '>' => tokens.push(ExprToken::Greater),
            '?' => tokens.push(ExprToken::Question),
            ':' => tokens.push(ExprToken::Colon),
            '(' => tokens.push(ExprToken::LeftParen),
            ')' => tokens.push(ExprToken::RightParen),
            ',' => tokens.push(ExprToken::Comma),
            _ => {
                return Err(format!("invalid token '{}' in preprocessor expression", ch));
            }
        }
        pos += 1;
    }

    Ok(tokens)
}

// ===========================================================================
// Integer Literal Parsing
// ===========================================================================

/// Parses an integer literal starting at `pos` in the character array.
///
/// Supports decimal, hexadecimal (`0x`/`0X`), octal (leading `0`), and binary
/// (`0b`/`0B`) formats, with optional C suffixes (`u`/`U`, `l`/`L`, `ll`/`LL`).
fn parse_integer_literal(chars: &[char], start: usize) -> Result<(BigInt, usize), String> {
    let len = chars.len();
    let mut pos = start;

    // Determine the numeric base from the prefix
    let base: u128;
    if pos + 1 < len && chars[pos] == '0' {
        match chars[pos + 1] {
            'x' | 'X' => {
                base = 16;
                pos += 2; // skip "0x" prefix
            }
            'b' | 'B' => {
                base = 2;
                pos += 2; // skip "0b" prefix
            }
            _ => {
                // Octal (leading zero) — don't skip the '0', it's a valid digit
                base = 8;
            }
        }
    } else if chars[pos] == '0' {
        // Single '0' — treated as octal zero (value is the same as decimal zero)
        base = 8;
    } else {
        base = 10;
    }

    // Parse the digit sequence
    let digit_start = pos;
    let mut value: u128 = 0;
    while pos < len {
        let ch = chars[pos];
        let digit = match ch {
            '0'..='9' => (ch as u128) - ('0' as u128),
            'a'..='f' if base == 16 => (ch as u128) - ('a' as u128) + 10,
            'A'..='F' if base == 16 => (ch as u128) - ('A' as u128) + 10,
            _ => break,
        };
        if digit >= base {
            break;
        }
        value = value.wrapping_mul(base).wrapping_add(digit);
        pos += 1;
    }

    // Validate that hex/binary prefixes are followed by at least one valid digit
    if (base == 16 || base == 2) && pos == digit_start {
        if base == 16 {
            return Err("invalid hexadecimal literal: no digits after '0x'".to_string());
        }
        return Err("invalid binary literal: no digits after '0b'".to_string());
    }

    // Parse optional integer suffixes (u/U, l/L, ll/LL, and combinations)
    let mut is_unsigned = false;
    while pos < len {
        match chars[pos] {
            'u' | 'U' => {
                is_unsigned = true;
                pos += 1;
            }
            'l' | 'L' => {
                pos += 1;
                // Check for "ll" / "LL" (long long suffix)
                if pos < len && (chars[pos] == 'l' || chars[pos] == 'L') {
                    pos += 1;
                }
            }
            _ => break,
        }
    }

    let result = if is_unsigned {
        // Use the narrowest constructor that fits the value
        if value <= u64::MAX as u128 {
            BigInt::from_u64(value as u64)
        } else {
            BigInt::from_u128(value)
        }
    } else {
        // For signed interpretation: if value fits in i128, use signed
        if value <= i128::MAX as u128 {
            BigInt::from_i128(value as i128)
        } else {
            BigInt::from_u128(value)
        }
    };

    Ok((result, pos))
}

// ===========================================================================
// Character Literal Parsing
// ===========================================================================

/// Parses a character literal starting at `pos` (which must point to the opening `'`).
///
/// Supports simple characters, C escape sequences, hexadecimal escapes (`\xNN`),
/// octal escapes (`\NNN`), and multi-character constants (GCC behavior).
fn parse_char_literal(chars: &[char], start: usize) -> Result<(BigInt, usize), String> {
    let len = chars.len();
    let mut pos = start + 1; // Skip the opening single quote

    if pos >= len {
        return Err("unterminated character constant".to_string());
    }

    // Collect character values for potential multi-char constants
    let mut values: Vec<u32> = Vec::new();

    while pos < len && chars[pos] != '\'' {
        let (char_value, new_pos) = parse_single_char(chars, pos)?;
        values.push(char_value);
        pos = new_pos;
    }

    if pos >= len || chars[pos] != '\'' {
        return Err("unterminated character constant".to_string());
    }
    pos += 1; // Skip the closing single quote

    if values.is_empty() {
        return Err("empty character constant".to_string());
    }

    // Compute the integer value
    let result = if values.len() == 1 {
        BigInt::from_i64(values[0] as i64)
    } else {
        // Multi-character constant: GCC computes (A << 24) | (B << 16) | (C << 8) | D
        let mut combined: i64 = 0;
        for (i, &v) in values.iter().enumerate() {
            let shift = (values.len() - 1 - i) * 8;
            combined |= (v as i64) << shift;
        }
        BigInt::from_i64(combined)
    };

    Ok((result, pos))
}

/// Parses a single character (possibly an escape sequence) within a character literal.
fn parse_single_char(chars: &[char], pos: usize) -> Result<(u32, usize), String> {
    let len = chars.len();
    if pos >= len {
        return Err("unterminated character constant".to_string());
    }

    if chars[pos] != '\\' {
        // Plain character — return its code point
        return Ok((chars[pos] as u32, pos + 1));
    }

    // Escape sequence: consume the backslash and dispatch on the next character
    if pos + 1 >= len {
        return Err("unterminated escape sequence in character constant".to_string());
    }

    let escape_char = chars[pos + 1];
    let mut new_pos = pos + 2;

    let value = match escape_char {
        'n' => 10,
        't' => 9,
        'r' => 13,
        'a' => 7,
        'b' => 8,
        'f' => 12,
        'v' => 11,
        '\\' => 92,
        '\'' => 39,
        '"' => 34,
        '?' => 63,
        'x' => {
            // Hexadecimal escape: \xNN... (consumes all following hex digits)
            let hex_start = new_pos;
            while new_pos < len && chars[new_pos].is_ascii_hexdigit() {
                new_pos += 1;
            }
            if new_pos == hex_start {
                return Err("invalid hex escape sequence: no digits after '\\x'".to_string());
            }
            let hex_str: String = chars[hex_start..new_pos].iter().collect();
            u32::from_str_radix(&hex_str, 16)
                .map_err(|_| "hex escape sequence out of range".to_string())?
        }
        '0'..='7' => {
            // Octal escape: \N, \NN, or \NNN (up to 3 octal digits)
            let oct_start = pos + 1;
            new_pos = oct_start;
            let mut count = 0;
            while new_pos < len && chars[new_pos] >= '0' && chars[new_pos] <= '7' && count < 3 {
                new_pos += 1;
                count += 1;
            }
            let oct_str: String = chars[oct_start..new_pos].iter().collect();
            u32::from_str_radix(&oct_str, 8)
                .map_err(|_| "octal escape sequence out of range".to_string())?
        }
        c => {
            // Unknown escape sequence — treat as the literal character after backslash
            c as u32
        }
    };

    Ok((value, new_pos))
}

// ===========================================================================
// Expression Evaluator — Recursive Descent with Short-Circuit Support
// ===========================================================================

/// Internal evaluator that parses and evaluates a tokenized preprocessor expression.
///
/// Uses recursive descent with one function per precedence level. This design
/// enables clean short-circuit evaluation for `&&`, `||`, and ternary `?:` by
/// passing a `skip` flag through the recursive call chain.
struct ExprEvaluator<'a> {
    /// Tokenized expression.
    tokens: Vec<ExprToken>,
    /// Current token position index.
    pos: usize,
    /// Callback to check whether a macro name is currently defined.
    is_macro_defined: &'a dyn Fn(&str) -> bool,
    /// Diagnostic emitter for error and warning reporting.
    diagnostics: &'a mut DiagnosticEmitter,
    /// Source location of the directive (for diagnostic messages).
    location: SourceLocation,
    /// Tracks whether an unrecoverable error has occurred during evaluation.
    had_error: bool,
}

impl<'a> ExprEvaluator<'a> {
    /// Creates a new evaluator with the given tokens and context.
    fn new(
        tokens: Vec<ExprToken>,
        is_macro_defined: &'a dyn Fn(&str) -> bool,
        diagnostics: &'a mut DiagnosticEmitter,
        location: SourceLocation,
    ) -> Self {
        ExprEvaluator {
            tokens,
            pos: 0,
            is_macro_defined,
            diagnostics,
            location,
            had_error: false,
        }
    }

    /// Returns a reference to the current token without consuming it.
    fn peek(&self) -> Option<&ExprToken> {
        self.tokens.get(self.pos)
    }

    /// Consumes the current token and returns it.
    fn advance(&mut self) -> Option<ExprToken> {
        if self.pos < self.tokens.len() {
            let token = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Consumes the current token if it matches `expected`, returning `true` on success.
    fn expect(&mut self, expected: &ExprToken) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Emits an error diagnostic and sets the error flag.
    fn emit_error(&mut self, message: &str) {
        self.diagnostics.error(self.location, message.to_string());
        self.had_error = true;
    }

    /// Emits a warning diagnostic (does not set the error flag).
    fn emit_warning(&mut self, message: &str) {
        self.diagnostics.warning(self.location, message.to_string());
    }

    /// Evaluates the complete expression and verifies all tokens are consumed.
    fn evaluate_expression(&mut self) -> Result<BigInt, ()> {
        if self.tokens.is_empty() {
            self.emit_error("expected expression");
            return Err(());
        }

        let result = self.parse_ternary(false)?;

        // Verify the entire token stream was consumed
        if self.pos < self.tokens.len() {
            let remaining: Vec<String> = self.tokens[self.pos..]
                .iter()
                .map(|t| format!("{:?}", t))
                .collect();
            let msg = format!(
                "unexpected token in preprocessor expression (remaining: {})",
                remaining
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            self.emit_error(&msg);
            return Err(());
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 1: Ternary (? :) — right-to-left associative
    // -----------------------------------------------------------------------

    /// Parses a ternary conditional expression: `condition ? true_expr : false_expr`.
    ///
    /// Right-to-left associativity means `a ? b : c ? d : e` is parsed as
    /// `a ? b : (c ? d : e)`. This is achieved naturally by recursing into
    /// `parse_ternary` for both the true and false branches.
    fn parse_ternary(&mut self, skip: bool) -> Result<BigInt, ()> {
        let condition = self.parse_logical_or(skip)?;

        if self.peek() == Some(&ExprToken::Question) {
            self.advance(); // consume '?'

            let cond_is_true = !skip && !condition.is_zero();
            let cond_is_false = !skip && condition.is_zero();

            // True branch: skip if condition is false or parent is already skipping
            let true_val = self.parse_ternary(skip || cond_is_false)?;

            if !self.expect(&ExprToken::Colon) {
                if !skip {
                    self.emit_error("expected ':' in ternary expression");
                }
                return Err(());
            }

            // False branch: skip if condition is true or parent is already skipping
            let false_val = self.parse_ternary(skip || cond_is_true)?;

            if skip {
                Ok(BigInt::zero())
            } else if cond_is_true {
                Ok(true_val)
            } else {
                Ok(false_val)
            }
        } else {
            Ok(condition)
        }
    }

    // -----------------------------------------------------------------------
    // Precedence level 2: Logical OR (||) — short-circuit
    // -----------------------------------------------------------------------

    fn parse_logical_or(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_logical_and(skip)?;

        while self.peek() == Some(&ExprToken::PipePipe) {
            self.advance();

            if skip {
                let _right = self.parse_logical_and(true)?;
            } else if !result.is_zero() {
                // Short-circuit: left is non-zero, skip right side
                let _right = self.parse_logical_and(true)?;
                result = BigInt::one();
            } else {
                let right = self.parse_logical_and(false)?;
                result = if !right.is_zero() {
                    BigInt::one()
                } else {
                    BigInt::zero()
                };
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 3: Logical AND (&&) — short-circuit
    // -----------------------------------------------------------------------

    fn parse_logical_and(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_bitwise_or(skip)?;

        while self.peek() == Some(&ExprToken::AmpAmp) {
            self.advance();

            if skip {
                let _right = self.parse_bitwise_or(true)?;
            } else if result.is_zero() {
                // Short-circuit: left is zero, skip right side
                let _right = self.parse_bitwise_or(true)?;
                result = BigInt::zero();
            } else {
                let right = self.parse_bitwise_or(false)?;
                result = if !right.is_zero() {
                    BigInt::one()
                } else {
                    BigInt::zero()
                };
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 4: Bitwise OR (|)
    // -----------------------------------------------------------------------

    fn parse_bitwise_or(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_bitwise_xor(skip)?;
        while self.peek() == Some(&ExprToken::Pipe) {
            self.advance();
            let right = self.parse_bitwise_xor(skip)?;
            if !skip {
                result = result | right;
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 5: Bitwise XOR (^)
    // -----------------------------------------------------------------------

    fn parse_bitwise_xor(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_bitwise_and(skip)?;
        while self.peek() == Some(&ExprToken::Caret) {
            self.advance();
            let right = self.parse_bitwise_and(skip)?;
            if !skip {
                result = result ^ right;
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 6: Bitwise AND (&)
    // -----------------------------------------------------------------------

    fn parse_bitwise_and(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_equality(skip)?;
        while self.peek() == Some(&ExprToken::Ampersand) {
            self.advance();
            let right = self.parse_equality(skip)?;
            if !skip {
                result = result & right;
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 7: Equality (==, !=)
    // -----------------------------------------------------------------------

    fn parse_equality(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_relational(skip)?;
        loop {
            match self.peek() {
                Some(&ExprToken::EqualEqual) => {
                    self.advance();
                    let right = self.parse_relational(skip)?;
                    if !skip {
                        result = if result.eq_value(&right) {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                Some(&ExprToken::ExclaimEqual) => {
                    self.advance();
                    let right = self.parse_relational(skip)?;
                    if !skip {
                        result = if !result.eq_value(&right) {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                _ => break,
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 8: Relational (<, >, <=, >=)
    // -----------------------------------------------------------------------

    fn parse_relational(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_shift(skip)?;
        loop {
            match self.peek() {
                Some(&ExprToken::Less) => {
                    self.advance();
                    let right = self.parse_shift(skip)?;
                    if !skip {
                        result = if result.less_than(&right) {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                Some(&ExprToken::Greater) => {
                    self.advance();
                    let right = self.parse_shift(skip)?;
                    if !skip {
                        result = if result.greater_than(&right) {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                Some(&ExprToken::LessEqual) => {
                    self.advance();
                    let right = self.parse_shift(skip)?;
                    if !skip {
                        result = if result.less_than(&right) || result.eq_value(&right) {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                Some(&ExprToken::GreaterEqual) => {
                    self.advance();
                    let right = self.parse_shift(skip)?;
                    if !skip {
                        result = if result.greater_than(&right) || result.eq_value(&right) {
                            BigInt::one()
                        } else {
                            BigInt::zero()
                        };
                    }
                }
                _ => break,
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 9: Shift (<<, >>)
    // -----------------------------------------------------------------------

    fn parse_shift(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_additive(skip)?;
        loop {
            match self.peek() {
                Some(&ExprToken::LessLess) => {
                    self.advance();
                    let right = self.parse_additive(skip)?;
                    if !skip {
                        // Negative shift amounts are undefined in C; clamp to 0
                        if right.is_negative() {
                            self.emit_warning(
                                "left shift by negative amount in preprocessor expression",
                            );
                        }
                        let shift_val = right.to_i128();
                        let amount = shift_val.clamp(0, 128) as u32;
                        result = result.shl(amount);
                    }
                }
                Some(&ExprToken::GreaterGreater) => {
                    self.advance();
                    let right = self.parse_additive(skip)?;
                    if !skip {
                        // Negative shift amounts are undefined in C; clamp to 0
                        if right.is_negative() {
                            self.emit_warning(
                                "right shift by negative amount in preprocessor expression",
                            );
                        }
                        let shift_val = right.to_i128();
                        let amount = shift_val.clamp(0, 128) as u32;
                        result = result.shr(amount);
                    }
                }
                _ => break,
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 10: Additive (+, -)
    // -----------------------------------------------------------------------

    fn parse_additive(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_multiplicative(skip)?;
        loop {
            match self.peek() {
                Some(&ExprToken::Plus) => {
                    self.advance();
                    let right = self.parse_multiplicative(skip)?;
                    if !skip {
                        let (sum, overflow) = result.add(&right);
                        if overflow {
                            self.emit_warning("integer overflow in preprocessor expression");
                        }
                        result = sum;
                    }
                }
                Some(&ExprToken::Minus) => {
                    self.advance();
                    let right = self.parse_multiplicative(skip)?;
                    if !skip {
                        let (diff, overflow) = result.sub(&right);
                        if overflow {
                            self.emit_warning("integer overflow in preprocessor expression");
                        }
                        result = diff;
                    }
                }
                _ => break,
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 11: Multiplicative (*, /, %)
    // -----------------------------------------------------------------------

    fn parse_multiplicative(&mut self, skip: bool) -> Result<BigInt, ()> {
        let mut result = self.parse_unary(skip)?;
        loop {
            match self.peek() {
                Some(&ExprToken::Star) => {
                    self.advance();
                    let right = self.parse_unary(skip)?;
                    if !skip {
                        let (prod, overflow) = result.mul(&right);
                        if overflow {
                            self.emit_warning("integer overflow in preprocessor expression");
                        }
                        result = prod;
                    }
                }
                Some(&ExprToken::Slash) => {
                    self.advance();
                    let right = self.parse_unary(skip)?;
                    if !skip {
                        match result.div(&right) {
                            Some(quotient) => result = quotient,
                            None => {
                                self.emit_error("division by zero in preprocessor expression");
                                return Err(());
                            }
                        }
                    }
                }
                Some(&ExprToken::Percent) => {
                    self.advance();
                    let right = self.parse_unary(skip)?;
                    if !skip {
                        match result.rem(&right) {
                            Some(remainder) => result = remainder,
                            None => {
                                self.emit_error("division by zero in preprocessor expression");
                                return Err(());
                            }
                        }
                    }
                }
                _ => break,
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Precedence level 12: Unary (+, -, ~, !)
    // -----------------------------------------------------------------------

    fn parse_unary(&mut self, skip: bool) -> Result<BigInt, ()> {
        match self.peek() {
            Some(&ExprToken::Plus) => {
                self.advance();
                self.parse_unary(skip) // identity
            }
            Some(&ExprToken::Minus) => {
                self.advance();
                let operand = self.parse_unary(skip)?;
                if skip {
                    Ok(BigInt::zero())
                } else {
                    Ok(operand.neg())
                }
            }
            Some(&ExprToken::Tilde) => {
                self.advance();
                let operand = self.parse_unary(skip)?;
                if skip {
                    Ok(BigInt::zero())
                } else {
                    Ok(!operand)
                }
            }
            Some(&ExprToken::Exclaim) => {
                self.advance();
                let operand = self.parse_unary(skip)?;
                if skip {
                    Ok(BigInt::zero())
                } else {
                    Ok(if operand.is_zero() {
                        BigInt::one()
                    } else {
                        BigInt::zero()
                    })
                }
            }
            _ => self.parse_primary(skip),
        }
    }

    // -----------------------------------------------------------------------
    // Precedence level 13: Primary expressions
    // -----------------------------------------------------------------------

    /// Parses primary expressions: integer literals, character literals,
    /// parenthesized sub-expressions, `defined()`, and identifiers.
    fn parse_primary(&mut self, skip: bool) -> Result<BigInt, ()> {
        match self.peek().cloned() {
            Some(ExprToken::Integer(value)) => {
                self.advance();
                Ok(if skip { BigInt::zero() } else { value })
            }
            Some(ExprToken::Char(value)) => {
                self.advance();
                Ok(if skip { BigInt::zero() } else { value })
            }
            Some(ExprToken::LeftParen) => {
                self.advance(); // consume '('
                let result = self.parse_ternary(skip)?;
                if !self.expect(&ExprToken::RightParen) {
                    if !skip {
                        self.emit_error("expected ')'");
                    }
                    return Err(());
                }
                Ok(result)
            }
            Some(ExprToken::Defined) => {
                self.advance(); // consume 'defined'
                self.parse_defined(skip)
            }
            Some(ExprToken::HasAttribute(ref name)) => {
                let attr_name = name.clone();
                self.advance();
                if skip {
                    Ok(BigInt::zero())
                } else {
                    Ok(if is_supported_attribute(&attr_name) {
                        BigInt::one()
                    } else {
                        BigInt::zero()
                    })
                }
            }
            Some(ExprToken::HasBuiltin(ref name)) => {
                let builtin_name = name.clone();
                self.advance();
                if skip {
                    Ok(BigInt::zero())
                } else {
                    Ok(if is_supported_builtin(&builtin_name) {
                        BigInt::one()
                    } else {
                        BigInt::zero()
                    })
                }
            }
            Some(ExprToken::HasExtension(ref name)) => {
                let ext_name = name.clone();
                self.advance();
                if skip {
                    Ok(BigInt::zero())
                } else {
                    Ok(if is_supported_extension(&ext_name) {
                        BigInt::one()
                    } else {
                        BigInt::zero()
                    })
                }
            }
            Some(ExprToken::HasInclude(ref _name)) => {
                // For __has_include, we conservatively return 0 (header not found)
                // since we don't have access to the include path resolver here.
                // Real resolution would require preprocessor context.
                self.advance();
                Ok(BigInt::zero())
            }
            Some(ExprToken::Identifier(_)) => {
                self.advance();
                // Per C11 §6.10.1p4: remaining identifiers evaluate to 0
                Ok(BigInt::zero())
            }
            Some(_) => {
                if !skip {
                    self.emit_error("expected expression");
                }
                Err(())
            }
            None => {
                if !skip {
                    self.emit_error("expected expression");
                }
                Err(())
            }
        }
    }

    // -----------------------------------------------------------------------
    // defined() operator
    // -----------------------------------------------------------------------

    /// Parses the `defined` operator in both forms:
    /// - `defined(NAME)` — parenthesized form
    /// - `defined NAME`  — non-parenthesized form
    fn parse_defined(&mut self, skip: bool) -> Result<BigInt, ()> {
        let has_paren = self.peek() == Some(&ExprToken::LeftParen);
        if has_paren {
            self.advance(); // consume '('
        }

        let name = match self.peek().cloned() {
            Some(ExprToken::Identifier(name)) => {
                self.advance();
                name
            }
            _ => {
                if !skip {
                    self.emit_error("expected identifier after 'defined'");
                }
                return Err(());
            }
        };

        if has_paren && !self.expect(&ExprToken::RightParen) {
            if !skip {
                self.emit_error("expected ')' after 'defined(NAME'");
            }
            return Err(());
        }

        if skip {
            Ok(BigInt::zero())
        } else {
            let is_defined = (self.is_macro_defined)(&name);
            Ok(if is_defined {
                BigInt::one()
            } else {
                BigInt::zero()
            })
        }
    }
}

// ===========================================================================
// __has_attribute / __has_builtin Support Tables
// ===========================================================================

/// Evaluates `sizeof(type)` or `alignof(type)` for common C types in the
/// preprocessor expression evaluator. This handles cases like the Linux kernel's:
///   `#define ARCH_KMALLOC_MINALIGN __alignof__(unsigned long long)`
/// which after expansion appears in `#elif ARCH_KMALLOC_MINALIGN > 8`.
///
/// Returns the size (if `is_sizeof`) or alignment of the type on x86-64.
/// For types not recognized, returns a conservative default of 1 for alignof
/// and 4 for sizeof.
fn evaluate_sizeof_alignof_type(type_str: &str, is_sizeof: bool) -> i64 {
    // Normalize the type string by collapsing whitespace and removing pointer declarators
    let normalized = type_str.replace("*", "").trim().to_string();
    let has_pointer = type_str.contains('*');

    // If it's a pointer type, size=8 and align=8 on x86-64
    if has_pointer {
        return 8;
    }

    // Match common C types — values are for x86-64 (LP64 model)
    let (size, align) = match normalized.as_str() {
        "char" | "signed char" | "unsigned char" | "_Bool" | "bool" => (1, 1),
        "short" | "signed short" | "short int" | "signed short int" | "unsigned short"
        | "unsigned short int" => (2, 2),
        "int" | "signed int" | "signed" | "unsigned int" | "unsigned" => (4, 4),
        "long" | "signed long" | "long int" | "signed long int" | "unsigned long"
        | "unsigned long int" => (8, 8),
        "long long"
        | "signed long long"
        | "long long int"
        | "signed long long int"
        | "unsigned long long"
        | "unsigned long long int" => (8, 8),
        "float" => (4, 4),
        "double" => (8, 8),
        "long double" => (16, 16),
        "void" => (1, 1),
        "__int128" | "unsigned __int128" | "__int128_t" | "__uint128_t" => (16, 16),
        // GCC extension: __int128 type
        _ => {
            // For struct/union/enum or unknown types, use conservative defaults
            if is_sizeof {
                (4, 4)
            } else {
                (4, 4)
            }
        }
    };

    if is_sizeof {
        size
    } else {
        align
    }
}

/// Returns `true` if `name` is an attribute that bcc's codegen implements.
///
/// Covers both the GNU-style double-underscore forms and the plain forms.
/// For example, both `packed` and `__packed__` are recognized.
fn is_supported_attribute(name: &str) -> bool {
    matches!(
        name,
        // Core attributes bcc implements
        "packed"
            | "__packed__"
            | "aligned"
            | "__aligned__"
            | "section"
            | "__section__"
            | "used"
            | "__used__"
            | "weak"
            | "__weak__"
            | "noreturn"
            | "__noreturn__"
            | "visibility"
            | "__visibility__"
            | "alias"
            | "__alias__"
            | "constructor"
            | "__constructor__"
            | "destructor"
            | "__destructor__"
            | "cold"
            | "__cold__"
            | "hot"
            | "__hot__"
            | "noinline"
            | "__noinline__"
            | "always_inline"
            | "__always_inline__"
            | "format"
            | "__format__"
            | "noclone"
            | "__noclone__"
            | "copy"
            | "__copy__"
            | "unused"
            | "__unused__"
            | "deprecated"
            | "__deprecated__"
            | "warn_unused_result"
            | "__warn_unused_result__"
            | "may_alias"
            | "__may_alias__"
            | "pure"
            | "__pure__"
            | "const"
            | "__const__"
            | "malloc"
            | "__malloc__"
            | "nonnull"
            | "__nonnull__"
            | "returns_nonnull"
            | "__returns_nonnull__"
            | "sentinel"
            | "__sentinel__"
            | "format_arg"
            | "__format_arg__"
            | "fallthrough"
            | "__fallthrough__"
            | "no_sanitize"
            | "__no_sanitize__"
            | "no_sanitize_address"
            | "__no_sanitize_address__"
            | "no_instrument_function"
            | "__no_instrument_function__"
            | "error"
            | "__error__"
            | "warning"
            | "__warning__"
            | "externally_visible"
            | "__externally_visible__"
            | "no_reorder"
            | "__no_reorder__"
            | "assume_aligned"
            | "__assume_aligned__"
            | "alloc_size"
            | "__alloc_size__"
            | "designated_init"
            | "__designated_init__"
            | "transparent_union"
            | "__transparent_union__"
            | "mode"
            | "__mode__"
            | "artificial"
            | "__artificial__"
            | "flatten"
            | "__flatten__"
            | "nothrow"
            | "__nothrow__"
            | "leaf"
            | "__leaf__"
            | "gnu_inline"
            | "__gnu_inline__"
            | "noipa"
            | "__noipa__"
            | "target"
            | "__target__"
            | "optimize"
            | "__optimize__"
            | "returns_twice"
            | "__returns_twice__"
            | "naked"
            | "__naked__"
            | "ms_struct"
            | "__ms_struct__"
            | "gcc_struct"
            | "__gcc_struct__"
            | "regparm"
            | "__regparm__"
            | "force_align_arg_pointer"
            | "__force_align_arg_pointer__"
            | "interrupt"
            | "__interrupt__"
            | "no_caller_saved_registers"
            | "__no_caller_saved_registers__"
            | "no_split_stack"
            | "__no_split_stack__"
            | "no_stack_limit"
            | "__no_stack_limit__"
            | "no_stack_protector"
            | "__no_stack_protector__"
            | "noplt"
            | "__noplt__"
            | "nocf_check"
            | "__nocf_check__"
            | "nonstring"
            | "__nonstring__"
            | "access"
            | "__access__"
            | "counted_by"
            | "__counted_by__"
            | "cleanup"
            | "__cleanup__"
    )
}

/// Returns `true` if `name` is a builtin that bcc lowers to real IR instructions.
///
/// Covers the common GCC/Clang builtins used in the Linux kernel and libc.
fn is_supported_builtin(name: &str) -> bool {
    matches!(
        name,
        // Varargs builtins
        "__builtin_va_start"
        | "__builtin_va_end"
        | "__builtin_va_arg"
        | "__builtin_va_copy"
        | "__builtin_va_list"
        // Overflow-checked arithmetic
        | "__builtin_add_overflow"
        | "__builtin_sub_overflow"
        | "__builtin_mul_overflow"
        // Memory/string builtins
        | "__builtin_memcpy"
        | "__builtin_memset"
        | "__builtin_memmove"
        | "__builtin_memcmp"
        | "__builtin_strlen"
        | "__builtin_strcmp"
        | "__builtin_strncmp"
        | "__builtin_strcpy"
        | "__builtin_strncpy"
        // Bit manipulation
        | "__builtin_clz"
        | "__builtin_clzl"
        | "__builtin_clzll"
        | "__builtin_ctz"
        | "__builtin_ctzl"
        | "__builtin_ctzll"
        | "__builtin_ffs"
        | "__builtin_ffsl"
        | "__builtin_ffsll"
        | "__builtin_popcount"
        | "__builtin_popcountl"
        | "__builtin_popcountll"
        | "__builtin_parity"
        | "__builtin_parityl"
        | "__builtin_parityll"
        | "__builtin_bswap16"
        | "__builtin_bswap32"
        | "__builtin_bswap64"
        // Compiler hints
        | "__builtin_expect"
        | "__builtin_expect_with_probability"
        | "__builtin_unreachable"
        | "__builtin_trap"
        | "__builtin_assume"
        | "__builtin_assume_aligned"
        // Object size / type
        | "__builtin_object_size"
        | "__builtin_constant_p"
        | "__builtin_types_compatible_p"
        | "__builtin_choose_expr"
        | "__builtin_classify_type"
        // Offsetof
        | "__builtin_offsetof"
        // Frame / return address
        | "__builtin_frame_address"
        | "__builtin_return_address"
        | "__builtin_extract_return_addr"
        // Atomics
        | "__builtin_atomic_load_n"
        | "__builtin_atomic_store_n"
        | "__builtin_atomic_exchange_n"
        | "__builtin_atomic_compare_exchange_n"
        | "__builtin_atomic_compare_exchange"
        | "__builtin_atomic_add_fetch"
        | "__builtin_atomic_sub_fetch"
        | "__builtin_atomic_and_fetch"
        | "__builtin_atomic_or_fetch"
        | "__builtin_atomic_xor_fetch"
        | "__builtin_atomic_nand_fetch"
        | "__builtin_atomic_fetch_add"
        | "__builtin_atomic_fetch_sub"
        | "__builtin_atomic_fetch_and"
        | "__builtin_atomic_fetch_or"
        | "__builtin_atomic_fetch_xor"
        | "__builtin_atomic_fetch_nand"
        | "__builtin_atomic_test_and_set"
        | "__builtin_atomic_clear"
        | "__builtin_atomic_thread_fence"
        | "__builtin_atomic_signal_fence"
        | "__builtin_atomic_always_lock_free"
        | "__builtin_atomic_is_lock_free"
        // Sync builtins (legacy __sync_*)
        | "__sync_fetch_and_add"
        | "__sync_fetch_and_sub"
        | "__sync_fetch_and_or"
        | "__sync_fetch_and_and"
        | "__sync_fetch_and_xor"
        | "__sync_fetch_and_nand"
        | "__sync_add_and_fetch"
        | "__sync_sub_and_fetch"
        | "__sync_or_and_fetch"
        | "__sync_and_and_fetch"
        | "__sync_xor_and_fetch"
        | "__sync_nand_and_fetch"
        | "__sync_val_compare_and_swap"
        | "__sync_bool_compare_and_swap"
        | "__sync_lock_test_and_set"
        | "__sync_lock_release"
        | "__sync_synchronize"
        // Overflow checking
        | "__builtin_sadd_overflow"
        | "__builtin_saddl_overflow"
        | "__builtin_saddll_overflow"
        | "__builtin_uadd_overflow"
        | "__builtin_uaddl_overflow"
        | "__builtin_uaddll_overflow"
        | "__builtin_ssub_overflow"
        | "__builtin_ssubl_overflow"
        | "__builtin_ssubll_overflow"
        | "__builtin_usub_overflow"
        | "__builtin_usubl_overflow"
        | "__builtin_usubll_overflow"
        | "__builtin_smul_overflow"
        | "__builtin_smull_overflow"
        | "__builtin_smulll_overflow"
        | "__builtin_umul_overflow"
        | "__builtin_umull_overflow"
        | "__builtin_umulll_overflow"
        // Math builtins
        | "__builtin_huge_val"
        | "__builtin_huge_valf"
        | "__builtin_inf"
        | "__builtin_inff"
        | "__builtin_nan"
        | "__builtin_nanf"
        | "__builtin_isnan"
        | "__builtin_isinf"
        | "__builtin_isfinite"
        | "__builtin_isinf_sign"
        | "__builtin_abs"
        | "__builtin_labs"
        | "__builtin_llabs"
        | "__builtin_fabs"
        | "__builtin_fabsf"
        // Prefetch
        | "__builtin_prefetch"
        // Misc
        | "__builtin_alloca"
        | "__builtin_alloca_with_align"
        | "__builtin_LINE"
        | "__builtin_FUNCTION"
        | "__builtin_FILE"
        | "__builtin_likely"
        | "__builtin_unlikely"
    )
}

/// Checks whether a given language extension name is supported by bcc.
///
/// This is used for the `__has_extension(name)` built-in preprocessor operator,
/// which is a Clang extension also checked by glibc headers. We report support
/// for the same extensions that GCC provides compatibility for.
fn is_supported_extension(name: &str) -> bool {
    matches!(
        name,
        // C11 features exposed as extensions
        "c_alignas"
        | "c_alignof"
        | "c_atomic"
        | "c_generic_selections"
        | "c_static_assert"
        | "c_thread_local"
        // GCC-compatible extensions
        | "gnu_asm"
        | "gnu_asm_goto_with_outputs"
        | "attribute_deprecated_with_message"
        | "enumerator_attributes"
        | "address_sanitizer"
        // Statement expressions, typeof, etc.
        | "statement_expressions"
        | "typeof"
    )
}

// ===========================================================================
// Public API
// ===========================================================================

/// Evaluates a preprocessor constant expression for `#if` / `#elif` directives.
///
/// # Arguments
///
/// * `expression_text` — The expression text after the `#if` or `#elif` keyword.
///   This text should already have had macros expanded by the caller.
/// * `is_macro_defined` — A callback that returns `true` if a given macro name
///   is currently defined. Used for the `defined()` operator.
/// * `diagnostics` — The diagnostic emitter for error and warning messages.
/// * `location` — The source location of the directive (for error messages).
///
/// # Returns
///
/// * `Ok(BigInt)` — The evaluation result as an arbitrary-precision integer.
/// * `Err(())` — The expression was invalid; error diagnostics have been emitted.
///
/// # C11 Semantics
///
/// - All arithmetic uses 128-bit integers (exceeding the `intmax_t` requirement).
/// - Undefined identifiers evaluate to `0` per C11 §6.10.1p4.
/// - Short-circuit evaluation applies to `&&` and `||`.
/// - The ternary `?:` operator is right-to-left associative.
pub fn evaluate(
    expression_text: &str,
    is_macro_defined: &dyn Fn(&str) -> bool,
    diagnostics: &mut DiagnosticEmitter,
    location: SourceLocation,
) -> Result<BigInt, ()> {
    // Phase 1: Tokenize the expression text
    let tokens = match tokenize_expression(expression_text) {
        Ok(tokens) => tokens,
        Err(err) => {
            diagnostics.error(location, err);
            return Err(());
        }
    };

    // Phase 2: Handle empty expression (whitespace-only input)
    if tokens.is_empty() {
        diagnostics.error(location, "expected expression".to_string());
        return Err(());
    }

    // Phase 3: Parse and evaluate the token stream
    let mut evaluator = ExprEvaluator::new(tokens, is_macro_defined, diagnostics, location);
    evaluator.evaluate_expression()
}

/// Evaluates a preprocessor constant expression and returns a boolean result.
///
/// This is a convenience wrapper around [`evaluate`] that converts the result
/// to `bool` using C's truthiness rule: zero is `false`, non-zero is `true`.
///
/// This is typically what `conditional.rs` needs for `#if` and `#elif` evaluation.
pub fn evaluate_to_bool(
    expression_text: &str,
    is_macro_defined: &dyn Fn(&str) -> bool,
    diagnostics: &mut DiagnosticEmitter,
    location: SourceLocation,
) -> Result<bool, ()> {
    let result = evaluate(expression_text, is_macro_defined, diagnostics, location)?;
    Ok(!result.is_zero())
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Evaluates an expression with a given set of defined macros.
    fn eval_expr(expr: &str, defined_macros: &[&str]) -> Result<i128, ()> {
        let mut diagnostics = DiagnosticEmitter::new();
        let location = SourceLocation::dummy();
        let defined_set: HashSet<&str> = defined_macros.iter().copied().collect();
        let is_defined = |name: &str| -> bool { defined_set.contains(name) };
        let result = evaluate(expr, &is_defined, &mut diagnostics, location)?;
        Ok(result.to_i128())
    }

    /// Evaluates an expression (no macros defined) and expects success.
    fn eval(expr: &str) -> i128 {
        eval_expr(expr, &[]).unwrap_or_else(|_| panic!("failed to evaluate: {}", expr))
    }

    /// Evaluates an expression with macros and expects success.
    fn eval_with(expr: &str, macros: &[&str]) -> i128 {
        eval_expr(expr, macros).unwrap_or_else(|_| panic!("failed to evaluate: {}", expr))
    }

    /// Evaluates and expects failure (returns true if evaluation fails).
    fn eval_fails(expr: &str) -> bool {
        eval_expr(expr, &[]).is_err()
    }

    /// Evaluates to bool with a given set of defined macros.
    fn eval_bool(expr: &str, defined_macros: &[&str]) -> Result<bool, ()> {
        let mut diagnostics = DiagnosticEmitter::new();
        let location = SourceLocation::dummy();
        let defined_set: HashSet<&str> = defined_macros.iter().copied().collect();
        let is_defined = |name: &str| -> bool { defined_set.contains(name) };
        evaluate_to_bool(expr, &is_defined, &mut diagnostics, location)
    }

    // --- Integer Literal Tests ---

    #[test]
    fn test_decimal_literals() {
        assert_eq!(eval("0"), 0);
        assert_eq!(eval("1"), 1);
        assert_eq!(eval("42"), 42);
        assert_eq!(eval("12345"), 12345);
        assert_eq!(eval("999999"), 999999);
    }

    #[test]
    fn test_hex_literals() {
        assert_eq!(eval("0xFF"), 255);
        assert_eq!(eval("0XFF"), 255);
        assert_eq!(eval("0x2A"), 42);
        assert_eq!(eval("0x0"), 0);
        assert_eq!(eval("0xDeAdBeEf"), 0xDEADBEEF_i128);
    }

    #[test]
    fn test_octal_literals() {
        assert_eq!(eval("077"), 63);
        assert_eq!(eval("010"), 8);
        assert_eq!(eval("052"), 42);
        assert_eq!(eval("00"), 0);
    }

    #[test]
    fn test_binary_literals() {
        assert_eq!(eval("0b1010"), 10);
        assert_eq!(eval("0B11111111"), 255);
        assert_eq!(eval("0b0"), 0);
        assert_eq!(eval("0b101010"), 42);
    }

    #[test]
    fn test_integer_suffixes() {
        assert_eq!(eval("42u"), 42);
        assert_eq!(eval("42U"), 42);
        assert_eq!(eval("42l"), 42);
        assert_eq!(eval("42L"), 42);
        assert_eq!(eval("42ll"), 42);
        assert_eq!(eval("42LL"), 42);
        assert_eq!(eval("42ULL"), 42);
        assert_eq!(eval("42ull"), 42);
        assert_eq!(eval("42LU"), 42);
    }

    // --- Arithmetic Tests ---

    #[test]
    fn test_addition() {
        assert_eq!(eval("2 + 3"), 5);
        assert_eq!(eval("0 + 0"), 0);
        assert_eq!(eval("100 + 200"), 300);
    }

    #[test]
    fn test_subtraction() {
        assert_eq!(eval("10 - 4"), 6);
        assert_eq!(eval("0 - 3"), -3);
        assert_eq!(eval("5 - 5"), 0);
    }

    #[test]
    fn test_multiplication() {
        assert_eq!(eval("3 * 7"), 21);
        assert_eq!(eval("0 * 999"), 0);
        assert_eq!(eval("1 * 42"), 42);
    }

    #[test]
    fn test_division() {
        assert_eq!(eval("10 / 3"), 3);
        assert_eq!(eval("42 / 6"), 7);
        assert_eq!(eval("100 / 10"), 10);
        assert_eq!(eval("7 / 2"), 3);
    }

    #[test]
    fn test_modulo() {
        assert_eq!(eval("10 % 3"), 1);
        assert_eq!(eval("42 % 6"), 0);
        assert_eq!(eval("7 % 2"), 1);
    }

    #[test]
    fn test_negation() {
        assert_eq!(eval("-5"), -5);
        assert_eq!(eval("-0"), 0);
        assert_eq!(eval("- -3"), 3);
        assert_eq!(eval("---1"), -1);
    }

    #[test]
    fn test_unary_plus() {
        assert_eq!(eval("+5"), 5);
        assert_eq!(eval("++5"), 5);
    }

    // --- Logical Operator Tests ---

    #[test]
    fn test_logical_and() {
        assert_eq!(eval("1 && 1"), 1);
        assert_eq!(eval("1 && 0"), 0);
        assert_eq!(eval("0 && 1"), 0);
        assert_eq!(eval("0 && 0"), 0);
        assert_eq!(eval("42 && 99"), 1);
    }

    #[test]
    fn test_logical_or() {
        assert_eq!(eval("1 || 1"), 1);
        assert_eq!(eval("1 || 0"), 1);
        assert_eq!(eval("0 || 1"), 1);
        assert_eq!(eval("0 || 0"), 0);
        assert_eq!(eval("42 || 0"), 1);
    }

    #[test]
    fn test_logical_not() {
        assert_eq!(eval("!0"), 1);
        assert_eq!(eval("!1"), 0);
        assert_eq!(eval("!42"), 0);
        assert_eq!(eval("!!1"), 1);
        assert_eq!(eval("!!0"), 0);
    }

    // --- Comparison Tests ---

    #[test]
    fn test_less_than() {
        assert_eq!(eval("3 < 5"), 1);
        assert_eq!(eval("5 < 3"), 0);
        assert_eq!(eval("3 < 3"), 0);
    }

    #[test]
    fn test_greater_than() {
        assert_eq!(eval("5 > 3"), 1);
        assert_eq!(eval("3 > 5"), 0);
        assert_eq!(eval("3 > 3"), 0);
    }

    #[test]
    fn test_less_equal() {
        assert_eq!(eval("3 <= 5"), 1);
        assert_eq!(eval("3 <= 3"), 1);
        assert_eq!(eval("5 <= 3"), 0);
    }

    #[test]
    fn test_greater_equal() {
        assert_eq!(eval("5 >= 3"), 1);
        assert_eq!(eval("3 >= 3"), 1);
        assert_eq!(eval("3 >= 5"), 0);
    }

    #[test]
    fn test_equality() {
        assert_eq!(eval("3 == 3"), 1);
        assert_eq!(eval("3 == 4"), 0);
        assert_eq!(eval("0 == 0"), 1);
    }

    #[test]
    fn test_inequality() {
        assert_eq!(eval("3 != 4"), 1);
        assert_eq!(eval("3 != 3"), 0);
    }

    // --- Bitwise Operator Tests ---

    #[test]
    fn test_bitwise_and() {
        assert_eq!(eval("0xFF & 0x0F"), 15);
        assert_eq!(eval("0xFF & 0"), 0);
        assert_eq!(eval("0xFF & 0xFF"), 255);
    }

    #[test]
    fn test_bitwise_or() {
        assert_eq!(eval("0xF0 | 0x0F"), 255);
        assert_eq!(eval("0 | 0"), 0);
        assert_eq!(eval("0xFF | 0"), 255);
    }

    #[test]
    fn test_bitwise_xor() {
        assert_eq!(eval("0xFF ^ 0x0F"), 240);
        assert_eq!(eval("0xFF ^ 0xFF"), 0);
        assert_eq!(eval("0xFF ^ 0"), 255);
    }

    #[test]
    fn test_bitwise_not() {
        assert_eq!(eval("~0"), -1);
        assert_eq!(eval("~(-1)"), 0);
    }

    // --- Shift Tests ---

    #[test]
    fn test_left_shift() {
        assert_eq!(eval("1 << 8"), 256);
        assert_eq!(eval("1 << 0"), 1);
        assert_eq!(eval("0xFF << 4"), 0xFF0);
    }

    #[test]
    fn test_right_shift() {
        assert_eq!(eval("256 >> 4"), 16);
        assert_eq!(eval("256 >> 8"), 1);
        assert_eq!(eval("1 >> 0"), 1);
    }

    // --- Ternary Operator Tests ---

    #[test]
    fn test_ternary_true() {
        assert_eq!(eval("1 ? 10 : 20"), 10);
        assert_eq!(eval("42 ? 10 : 20"), 10);
    }

    #[test]
    fn test_ternary_false() {
        assert_eq!(eval("0 ? 10 : 20"), 20);
    }

    #[test]
    fn test_nested_ternary() {
        assert_eq!(eval("1 ? 2 ? 3 : 4 : 5"), 3);
        assert_eq!(eval("0 ? 2 : 3 ? 4 : 5"), 4);
        assert_eq!(eval("0 ? 2 : 0 ? 4 : 5"), 5);
    }

    #[test]
    fn test_ternary_short_circuit() {
        assert_eq!(eval("1 ? 42 : 1/0"), 42);
        assert_eq!(eval("0 ? 1/0 : 42"), 42);
    }

    // --- Parentheses and Precedence Tests ---

    #[test]
    fn test_parentheses() {
        assert_eq!(eval("(2 + 3) * 4"), 20);
        assert_eq!(eval("2 * (3 + 4)"), 14);
        assert_eq!(eval("((1))"), 1);
        assert_eq!(eval("(((2 + 3)))"), 5);
    }

    #[test]
    fn test_operator_precedence_mul_over_add() {
        assert_eq!(eval("2 + 3 * 4"), 14);
        assert_eq!(eval("3 * 4 + 2"), 14);
    }

    #[test]
    fn test_operator_precedence_logical() {
        assert_eq!(eval("0 || 1 && 1"), 1);
        assert_eq!(eval("1 || 0 && 0"), 1);
    }

    #[test]
    fn test_operator_precedence_bitwise() {
        assert_eq!(eval("0xFF & 0x0F ^ 0x0F"), 0);
        assert_eq!(eval("0xFF & 0x0F | 0xF0"), 255);
    }

    #[test]
    fn test_operator_precedence_shift_vs_add() {
        assert_eq!(eval("1 << 2 + 3"), 1 << (2 + 3));
    }

    #[test]
    fn test_operator_precedence_comparison() {
        assert_eq!(eval("1 < 2 == 1"), 1);
    }

    // --- defined() Tests ---

    #[test]
    fn test_defined_with_parens() {
        assert_eq!(eval_with("defined(FOO)", &["FOO"]), 1);
        assert_eq!(eval_with("defined(BAR)", &[]), 0);
    }

    #[test]
    fn test_defined_without_parens() {
        assert_eq!(eval_with("defined FOO", &["FOO"]), 1);
        assert_eq!(eval_with("defined BAR", &[]), 0);
    }

    #[test]
    fn test_defined_in_expression() {
        assert_eq!(eval_with("defined(FOO) && 1", &["FOO"]), 1);
        assert_eq!(eval_with("defined(FOO) && 1", &[]), 0);
        assert_eq!(eval_with("defined(FOO) || defined(BAR)", &["BAR"]), 1);
        assert_eq!(eval_with("!defined(FOO)", &[]), 1);
        assert_eq!(eval_with("!defined(FOO)", &["FOO"]), 0);
    }

    #[test]
    fn test_defined_multiple_macros() {
        assert_eq!(
            eval_with("defined(FOO) && defined(BAR)", &["FOO", "BAR"]),
            1
        );
        assert_eq!(eval_with("defined(FOO) && defined(BAR)", &["FOO"]), 0);
    }

    // --- Identifier Tests ---

    #[test]
    fn test_undefined_identifier_evaluates_to_zero() {
        assert_eq!(eval("UNKNOWN"), 0);
        assert_eq!(eval("UNKNOWN + 5"), 5);
        assert_eq!(eval("!UNKNOWN"), 1);
    }

    // --- Character Literal Tests ---

    #[test]
    fn test_char_literal_simple() {
        assert_eq!(eval("'A'"), 65);
        assert_eq!(eval("'a'"), 97);
        assert_eq!(eval("'0'"), 48);
        assert_eq!(eval("' '"), 32);
    }

    #[test]
    fn test_char_escape_sequences() {
        assert_eq!(eval("'\\n'"), 10);
        assert_eq!(eval("'\\t'"), 9);
        assert_eq!(eval("'\\r'"), 13);
        assert_eq!(eval("'\\0'"), 0);
        assert_eq!(eval("'\\\\'"), 92);
        assert_eq!(eval("'\\''"), 39);
    }

    #[test]
    fn test_hex_char_escape() {
        assert_eq!(eval("'\\x41'"), 65);
        assert_eq!(eval("'\\x00'"), 0);
        assert_eq!(eval("'\\x61'"), 97);
    }

    #[test]
    fn test_octal_char_escape() {
        assert_eq!(eval("'\\101'"), 65);
        assert_eq!(eval("'\\0'"), 0);
        assert_eq!(eval("'\\12'"), 10);
    }

    #[test]
    fn test_char_in_expression() {
        assert_eq!(eval("'A' == 65"), 1);
        assert_eq!(eval("'A' < 'B'"), 1);
        assert_eq!(eval("'Z' - 'A'"), 25);
    }

    // --- Error Handling Tests ---

    #[test]
    fn test_division_by_zero_error() {
        assert!(eval_fails("1 / 0"));
        assert!(eval_fails("10 % 0"));
    }

    #[test]
    fn test_empty_expression_error() {
        assert!(eval_fails(""));
        assert!(eval_fails("   "));
    }

    #[test]
    fn test_missing_right_paren_error() {
        assert!(eval_fails("(2 + 3"));
    }

    #[test]
    fn test_unexpected_token_error() {
        assert!(eval_fails("2 +"));
        assert!(eval_fails("* 3"));
    }

    #[test]
    fn test_invalid_token_in_input() {
        assert!(eval_fails("2 @ 3"));
        assert!(eval_fails("$foo"));
    }

    #[test]
    fn test_trailing_tokens_error() {
        assert!(eval_fails("1 2"));
        assert!(eval_fails("1 + 2 3"));
    }

    // --- Short-Circuit Evaluation Tests ---

    #[test]
    fn test_short_circuit_and_skips_division_by_zero() {
        assert_eq!(eval("0 && (1/0)"), 0);
    }

    #[test]
    fn test_short_circuit_or_skips_division_by_zero() {
        assert_eq!(eval("1 || (1/0)"), 1);
    }

    #[test]
    fn test_short_circuit_chained() {
        assert_eq!(eval("0 && (1/0) && (1/0)"), 0);
        assert_eq!(eval("1 || (1/0) || (1/0)"), 1);
    }

    #[test]
    fn test_no_short_circuit_when_needed() {
        assert!(eval_fails("1 && (1/0)"));
        assert!(eval_fails("0 || (1/0)"));
    }

    // --- Complex Expression Tests ---

    #[test]
    fn test_complex_defined_and_comparison() {
        assert_eq!(eval_with("defined(FOO) && (10 > 5)", &["FOO"]), 1);
        assert_eq!(eval_with("defined(FOO) && (10 > 5)", &[]), 0);
    }

    #[test]
    fn test_complex_arithmetic() {
        assert_eq!(eval("(1 + 2) * 3 - 4 / 2"), 7);
        assert_eq!(eval("10 + 20 * 30"), 610);
        assert_eq!(eval("((10 + 20) * 30)"), 900);
    }

    #[test]
    fn test_complex_bitwise() {
        assert_eq!(eval("(0xFF >> 4) & 0x0F"), 15);
        assert_eq!(eval("(1 << 8) - 1"), 255);
    }

    #[test]
    fn test_complex_logical() {
        assert_eq!(eval("(1 && 1) || (0 && 1)"), 1);
        assert_eq!(eval("(0 || 0) && (1 || 1)"), 0);
    }

    // --- evaluate_to_bool() Tests ---

    #[test]
    fn test_evaluate_to_bool_basic() {
        assert_eq!(eval_bool("1", &[]), Ok(true));
        assert_eq!(eval_bool("0", &[]), Ok(false));
        assert_eq!(eval_bool("42", &[]), Ok(true));
        assert_eq!(eval_bool("-1", &[]), Ok(true));
    }

    #[test]
    fn test_evaluate_to_bool_expressions() {
        assert_eq!(eval_bool("1 + 1", &[]), Ok(true));
        assert_eq!(eval_bool("1 - 1", &[]), Ok(false));
        assert_eq!(eval_bool("3 > 2", &[]), Ok(true));
        assert_eq!(eval_bool("2 > 3", &[]), Ok(false));
    }

    #[test]
    fn test_evaluate_to_bool_defined() {
        assert_eq!(eval_bool("defined(FOO)", &["FOO"]), Ok(true));
        assert_eq!(eval_bool("defined(FOO)", &[]), Ok(false));
    }

    #[test]
    fn test_evaluate_to_bool_error() {
        assert!(eval_bool("", &[]).is_err());
        assert!(eval_bool("1 / 0", &[]).is_err());
    }

    // --- Tokenizer Direct Tests ---

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize_expression("2 + 3").unwrap();
        assert_eq!(tokens.len(), 3);
        assert!(matches!(&tokens[0], ExprToken::Integer(_)));
        assert_eq!(tokens[1], ExprToken::Plus);
        assert!(matches!(&tokens[2], ExprToken::Integer(_)));
    }

    #[test]
    fn test_tokenize_defined() {
        let tokens = tokenize_expression("defined(FOO)").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], ExprToken::Defined);
        assert_eq!(tokens[1], ExprToken::LeftParen);
        assert!(matches!(&tokens[2], ExprToken::Identifier(ref n) if n == "FOO"));
        assert_eq!(tokens[3], ExprToken::RightParen);
    }

    #[test]
    fn test_tokenize_all_operators() {
        let tokens =
            tokenize_expression("+ - * / % & | ^ ~ ! && || << >> == != < > <= >= ? : ( ) ,")
                .unwrap();
        assert_eq!(tokens.len(), 25);
    }

    #[test]
    fn test_tokenize_invalid_char() {
        assert!(tokenize_expression("2 @ 3").is_err());
    }
}
