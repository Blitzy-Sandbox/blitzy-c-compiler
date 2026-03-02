//! Parser module — recursive-descent parser for C11 with GCC extensions.
//!
//! This module is the entry point for the `bcc` parser. It declares all parser
//! submodules, provides the [`Parser`] struct with the public [`Parser::parse()`]
//! method, manages parser state (token position, typedef tracking, error
//! recovery), and re-exports key AST types for consumption by downstream
//! compiler phases.
//!
//! # Architecture
//!
//! The parser is a recursive-descent parser that operates on a flat token stream
//! (`&[Token]`) produced by the lexer. The main parsing loop lives in
//! [`Parser::parse()`], which iterates over top-level declarations, delegating
//! to submodule functions for each syntactic category:
//!
//! | Submodule          | Responsibility                                        |
//! |--------------------|-------------------------------------------------------|
//! | [`ast`]            | AST node type definitions for all C11 + GCC constructs|
//! | [`declarations`]   | Variable, function, typedef, struct/union/enum decls   |
//! | [`expressions`]    | Precedence-climbing expression parser (15 levels)      |
//! | [`statements`]     | Statement parsing: if/for/while/switch/return/goto     |
//! | [`types`]          | Type specifiers, qualifiers, pointer/array declarators |
//! | [`gcc_extensions`] | `__attribute__`, `asm`, `typeof`, computed goto, etc.  |
//!
//! # Error Recovery
//!
//! The parser implements panic-mode error recovery with synchronization tokens
//! (`;`, `}`, `{`, and statement-starting keywords) to achieve ≥80% recovery
//! rate. This allows the parser to continue after syntax errors and report
//! multiple errors in a single compilation pass.
//!
//! # Performance
//!
//! The parser processes an in-memory token stream via index-based traversal
//! (zero copies, zero allocation for token access), meeting the performance
//! target of compiling the SQLite amalgamation (~230K LOC) within the overall
//! <60s constraint at `-O0`.
//!
//! # Integration Points
//!
//! - **Upstream**: Receives `Vec<Token>` from `frontend::lexer::Lexer::tokenize()`
//! - **Downstream**: Produces `TranslationUnit` AST consumed by `sema::SemanticAnalyzer`
//! - Per AAP §0.4.1: "Lexer → Parser: `Vec<Token>` where each `Token` carries
//!   type, value, and `SourceLocation`"
//! - Per AAP §0.4.1: "Parser → Semantic Analyzer: Untyped AST (`TranslationUnit`
//!   root node containing declarations, definitions, and type definitions)"
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

// ===========================================================================
// Submodule Declarations
// ===========================================================================

/// Complete AST node hierarchy for C11 + GCC extension constructs.
pub mod ast;

/// Declaration parsing: variable, function, typedef, struct/union/enum,
/// `_Static_assert`, and designated initializers.
pub mod declarations;

/// Expression parsing with precedence climbing for all 15 C operator levels.
pub mod expressions;

/// Statement parsing: compound blocks, selection, iteration, jump, and labeled
/// statements.
pub mod statements;

/// Type specifier parsing: base types, qualifiers, pointer/array declarators,
/// abstract declarators for casts and sizeof.
pub mod types;

/// GCC extension parsing: `__attribute__`, `asm`, `typeof`, computed goto,
/// statement expressions, `__extension__`, and `__builtin_*` intrinsics.
pub mod gcc_extensions;

// ===========================================================================
// Re-exports — key AST types for downstream module convenience
// ===========================================================================

pub use ast::{
    Declaration, Expression, FunctionDef, Statement, TranslationUnit, TypeSpecifier,
};

// ===========================================================================
// Imports
// ===========================================================================

use std::collections::HashSet;

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::intern::{InternId, Interner};
use crate::common::source_map::{SourceLocation, SourceSpan};
use crate::frontend::lexer::token::{Token, TokenKind, TokenValue};

// ===========================================================================
// Parser Struct — Central Parsing State Container
// ===========================================================================

/// Maximum nesting depth for statement-level recursive constructs (compound
/// statements, i.e., `{...}` blocks). Each nesting level adds approximately
/// 3 stack frames to the recursive descent call chain, so a limit of 1024
/// requires ~3072 frames (~3 MiB), well within the default ~8 MiB stack.
///
/// Real-world C codebases rarely exceed 20-30 levels of block nesting.
const MAX_STATEMENT_NESTING_DEPTH: usize = 1024;

/// Maximum nesting depth for expression-level recursive constructs
/// (parenthesized expressions, i.e., `((...))` chains). Each expression
/// nesting level adds approximately 8 stack frames to the recursive descent
/// call chain (expression → assignment → conditional → binary → unary →
/// postfix → primary → paren → expression), so the limit must be lower than
/// statement nesting to prevent stack overflow.
///
/// A limit of 512 requires ~4096 frames (~4 MiB), safely within the default
/// ~8 MiB stack. Real-world C code rarely exceeds 15-20 levels of expression
/// nesting. GCC and Clang enforce similar internal nesting limits.
const MAX_EXPRESSION_NESTING_DEPTH: usize = 512;

/// The recursive-descent parser for C11 with GCC extensions.
///
/// `Parser` holds all mutable parsing state: the current position in the token
/// stream, the set of known typedef names (essential for C's declaration-vs-
/// expression ambiguity), error recovery tracking, and references to shared
/// infrastructure (diagnostic emitter, string interner).
///
/// All submodule parsing functions (`declarations.rs`, `expressions.rs`, etc.)
/// receive `&mut Parser` as their first parameter, giving them access to the
/// token stream and all parser infrastructure through the methods defined here.
///
/// # Lifetime `'a`
///
/// The lifetime parameter `'a` ties the parser to:
/// - The token stream slice (`&'a [Token]`)
/// - The diagnostic emitter (`&'a mut DiagnosticEmitter`)
/// - The string interner (`&'a Interner`)
///
/// This ensures the parser cannot outlive the data it references.
pub struct Parser<'a> {
    /// Token stream from the lexer (consumed left-to-right). The slice always
    /// terminates with a [`TokenKind::Eof`] token.
    tokens: &'a [Token],

    /// Current position in the token stream (index into `tokens`). Invariant:
    /// `pos < tokens.len()`.
    pos: usize,

    /// Set of currently-visible typedef names. When the parser encounters an
    /// identifier, it checks this set to determine if it should be treated as
    /// a type specifier (typedef name) or a regular identifier. This
    /// disambiguation is critical for correctly parsing C code.
    typedef_names: HashSet<InternId>,

    /// Stack of typedef name snapshots for scope tracking. Each entry is a
    /// snapshot of `typedef_names` at the point a new scope was entered.
    /// [`push_scope()`] saves the current state; [`pop_scope()`] restores it.
    scope_stack: Vec<HashSet<InternId>>,

    /// Reference to the GCC-compatible diagnostic emitter for error reporting.
    /// All parse errors, warnings, and notes are reported through this emitter.
    diagnostics: &'a mut DiagnosticEmitter,

    /// Reference to the string interner. Used for resolving identifier names
    /// during typedef tracking, error message construction, and accessing
    /// token values.
    interner: &'a Interner,

    /// Count of errors encountered during parsing. Used together with
    /// `recovered_count` to compute the error recovery rate.
    error_count: usize,

    /// Count of errors from which the parser successfully recovered. An error
    /// is considered "recovered" when the parser reaches a synchronization
    /// point and resumes normal parsing after entering panic mode.
    recovered_count: usize,

    /// Flag indicating the parser is currently in panic/error-recovery mode.
    /// While in panic mode, the parser skips tokens until it finds a
    /// synchronization point (`;`, `}`, `{`, or a statement-starting keyword).
    in_panic_mode: bool,

    /// Current nesting depth for statement-level constructs (compound
    /// statements / `{...}` blocks). Incremented on entry to a compound
    /// statement and decremented on exit. When this counter exceeds
    /// [`MAX_STATEMENT_NESTING_DEPTH`], the parser emits a clean diagnostic
    /// error instead of continuing to recurse.
    stmt_nesting_depth: usize,

    /// Current nesting depth for expression-level constructs (parenthesized
    /// expressions / `((...))` chains). Incremented on entry to an expression
    /// recursion and decremented on exit. When this counter exceeds
    /// [`MAX_EXPRESSION_NESTING_DEPTH`], the parser emits a clean diagnostic
    /// error instead of continuing to recurse.
    expr_nesting_depth: usize,
}

// ===========================================================================
// Construction
// ===========================================================================

impl<'a> Parser<'a> {
    /// Creates a new `Parser` instance with the given token stream, interner,
    /// and diagnostic emitter.
    ///
    /// # Arguments
    ///
    /// * `tokens` — The token stream from the lexer. Must be non-empty and
    ///   terminated with a [`TokenKind::Eof`] token.
    /// * `interner` — The string interner used during lexing. The parser uses
    ///   it to resolve identifier names for typedef tracking and error messages.
    /// * `diagnostics` — The diagnostic emitter for reporting parse errors in
    ///   GCC-compatible format on stderr.
    ///
    /// # Panics
    ///
    /// Panics if `tokens` is empty. The lexer guarantees at least one token
    /// (the EOF marker) is always present.
    pub fn new(
        tokens: &'a [Token],
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Self {
        assert!(
            !tokens.is_empty(),
            "Parser::new: token stream must not be empty (must contain at least EOF)"
        );

        Parser {
            tokens,
            pos: 0,
            typedef_names: HashSet::new(),
            scope_stack: Vec::new(),
            diagnostics,
            interner,
            error_count: 0,
            recovered_count: 0,
            in_panic_mode: false,
            stmt_nesting_depth: 0,
            expr_nesting_depth: 0,
        }
    }

    // =======================================================================
    // Public Parse API
    // =======================================================================

    /// Parses a complete C translation unit from the given token stream.
    ///
    /// This is the main public entry point for the parser. It creates a
    /// `Parser` instance, loops over top-level declarations until EOF, and
    /// returns the resulting AST.
    ///
    /// # Arguments
    ///
    /// * `tokens` — The token stream produced by the lexer.
    /// * `interner` — The string interner shared with the lexer.
    /// * `diagnostics` — The diagnostic emitter for error reporting.
    ///
    /// # Returns
    ///
    /// * `Ok(TranslationUnit)` — Parsing succeeded (possibly with recovered
    ///   errors). The AST may be partial if errors occurred but recovery was
    ///   successful.
    /// * `Err(())` — Parsing failed with unrecoverable errors. Error details
    ///   are available through `diagnostics.has_errors()` and
    ///   `diagnostics.diagnostics()`.
    ///
    /// # Error Recovery
    ///
    /// The parser uses panic-mode error recovery. When a syntax error is
    /// encountered, it enters panic mode, skips tokens until a synchronization
    /// point is found, then resumes normal parsing. This allows collecting
    /// multiple errors in a single pass.
    pub fn parse(
        tokens: &'a [Token],
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Result<TranslationUnit, ()> {
        let mut parser = Parser::new(tokens, interner, diagnostics);
        parser.parse_translation_unit()
    }

    /// Internal method that drives the main parsing loop over the translation
    /// unit. Collects top-level declarations until EOF.
    fn parse_translation_unit(&mut self) -> Result<TranslationUnit, ()> {
        let start_span = self.current_span();
        let mut decls: Vec<Declaration> = Vec::new();

        while !self.is_at_end() {
            // Skip any stray semicolons at the top level (empty declarations).
            if self.check(TokenKind::Semicolon) {
                let semi_span = self.current_span();
                self.advance();
                decls.push(Declaration::Empty { span: semi_span });
                continue;
            }

            let pos_before = self.pos;

            // Attempt to parse a top-level declaration (variable, function,
            // typedef, struct/union/enum, or _Static_assert).
            let decl = declarations::parse_external_declaration(self);
            decls.push(decl);

            // If we entered panic mode during declaration parsing and haven't
            // exited yet, synchronize to the next declaration boundary.
            if self.in_panic_mode {
                self.synchronize_to_declaration();
                self.in_panic_mode = false;
                self.recovered_count += 1;
            }

            // Guard against infinite loops: if no forward progress was made
            // after a full declaration parse attempt plus synchronization,
            // force-advance past the problematic token.
            if self.pos == pos_before && !self.is_at_end() {
                self.advance();
            }
        }

        let end_span = self.previous_span();
        let span = SourceSpan {
            start: start_span.start,
            end: end_span.end,
        };

        Ok(TranslationUnit {
            declarations: decls,
            span,
        })
    }

    // =======================================================================
    // Token Access Methods
    // =======================================================================

    /// Returns a reference to the current token without consuming it.
    ///
    /// If the parser has reached the end of the token stream, returns the
    /// last token (which is always [`TokenKind::Eof`]).
    #[inline]
    pub(crate) fn current(&self) -> &'a Token {
        let tokens = self.tokens;
        if self.pos < tokens.len() {
            &tokens[self.pos]
        } else {
            &tokens[tokens.len() - 1]
        }
    }

    /// Returns a reference to the next token (one position ahead) without
    /// consuming anything.
    ///
    /// If there is no next token (current is already EOF or the last token),
    /// returns the last token in the stream (the EOF sentinel).
    #[inline]
    pub(crate) fn peek(&self) -> &'a Token {
        let tokens = self.tokens;
        let next_pos = self.pos + 1;
        if next_pos < tokens.len() {
            &tokens[next_pos]
        } else {
            &tokens[tokens.len() - 1]
        }
    }

    /// Returns a reference to the token `n` positions ahead without consuming
    /// anything. `lookahead(0)` is equivalent to `current()`.
    ///
    /// If the requested position is beyond the end of the token stream,
    /// returns the EOF sentinel token.
    #[inline]
    pub(crate) fn lookahead(&self, n: usize) -> &'a Token {
        let tokens = self.tokens;
        let target = self.pos.saturating_add(n);
        if target < tokens.len() {
            &tokens[target]
        } else {
            &tokens[tokens.len() - 1]
        }
    }

    /// Consumes the current token and advances the position by one.
    ///
    /// Returns a reference to the token that was consumed (the token at the
    /// position *before* advancing). If the current token is EOF, the position
    /// does not advance (EOF is never consumed).
    #[inline]
    pub(crate) fn advance(&mut self) -> &'a Token {
        let tokens = self.tokens;
        let current_pos = self.pos;
        // Only advance if we're not already at/past the last token (EOF).
        if current_pos + 1 < tokens.len() {
            self.pos = current_pos + 1;
        }
        &tokens[current_pos]
    }

    /// Returns `true` if the parser has reached the end of the token stream
    /// (current token is [`TokenKind::Eof`]).
    #[inline]
    pub(crate) fn is_at_end(&self) -> bool {
        self.current().kind == TokenKind::Eof
    }

    // =======================================================================
    // Token Matching Methods
    // =======================================================================

    /// Returns `true` if the current token's kind matches the given `kind`.
    ///
    /// Does not consume any tokens.
    #[inline]
    pub(crate) fn check(&self, kind: TokenKind) -> bool {
        self.current().kind == kind
    }

    /// Consumes the current token if its kind matches `kind`, returning `true`.
    /// If it does not match, returns `false` without consuming anything.
    ///
    /// This is the basic "try to match" primitive used throughout the parser
    /// for optional grammar elements.
    #[inline]
    pub(crate) fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expects the current token to match `kind`. If it does, consumes and
    /// returns it. If it does not, reports an error and returns `Err(())`.
    ///
    /// This is the standard "must match" primitive. On failure, it reports
    /// an "expected X, found Y" error at the current token position but does
    /// **not** enter panic mode or synchronize — the caller decides how to
    /// recover from the missing token.
    pub(crate) fn expect(&mut self, kind: TokenKind) -> Result<&'a Token, ()> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            self.unexpected_token(kind.as_str());
            Err(())
        }
    }

    /// Consumes the current token if its kind matches any of the given `kinds`.
    /// Returns `Some(matched_kind)` on success, or `None` if no kind matched.
    pub(crate) fn match_any(&mut self, kinds: &[TokenKind]) -> Option<TokenKind> {
        let current_kind = self.current().kind;
        for &kind in kinds {
            if current_kind == kind {
                self.advance();
                return Some(kind);
            }
        }
        None
    }

    // =======================================================================
    // Error Reporting Methods
    // =======================================================================

    /// Reports a parse error at the current token position.
    ///
    /// The error is emitted through the diagnostic emitter in GCC-compatible
    /// format (`file:line:col: error: message`). The error count is
    /// incremented for recovery rate tracking.
    pub(crate) fn error(&mut self, message: &str) {
        let loc = self.current().span.start;
        self.diagnostics.error(loc, message);
        self.error_count += 1;
        self.in_panic_mode = true;
    }

    /// Reports a parse error at a specific source span.
    ///
    /// Used when the error location differs from the current token position
    /// (e.g., pointing back to the start of a construct that failed to parse).
    pub(crate) fn error_at(&mut self, span: SourceSpan, message: &str) {
        self.diagnostics.error(span.start, message);
        self.error_count += 1;
        self.in_panic_mode = true;
    }

    /// Reports an "expected X, found Y" error for the current token.
    ///
    /// Produces a standardized error message format:
    /// `"expected <expected>, found '<actual>'"`. For identifier tokens, the
    /// actual identifier name is resolved via the interner for more descriptive
    /// error messages.
    pub(crate) fn unexpected_token(&mut self, expected: &str) {
        let found = match self.current().value {
            TokenValue::Identifier(id) => {
                let name = self.interner.resolve(id);
                format!("identifier '{}'", name)
            }
            _ => format!("'{}'", self.current().kind.as_str()),
        };
        let message = format!("expected {}, found {}", expected, found);
        let loc = self.current().span.start;
        self.diagnostics.error(loc, message);
        self.error_count += 1;
        self.in_panic_mode = true;
    }

    /// Reports a warning at the current token position.
    ///
    /// Warnings are non-blocking — they do not prevent successful compilation
    /// and do not affect the error recovery rate.
    pub(crate) fn warning(&mut self, message: &str) {
        let loc = self.current().span.start;
        self.diagnostics.warning(loc, message);
    }

    // =======================================================================
    // Error Recovery Methods
    // =======================================================================

    /// Enters panic mode and skips tokens until a synchronization point is
    /// reached.
    ///
    /// Synchronization tokens for statement-level recovery:
    /// - `;` — end of statement (consumed)
    /// - `}` — end of block (NOT consumed — left for the compound statement
    ///   parser)
    /// - `{` — start of new block (NOT consumed)
    /// - Statement-starting keywords: `if`, `for`, `while`, `do`, `switch`,
    ///   `return`, `break`, `continue`, `goto`, `case`, `default`
    ///
    /// After reaching a synchronization point, `in_panic_mode` is cleared
    /// and `recovered_count` is incremented.
    pub(crate) fn synchronize(&mut self) {
        self.in_panic_mode = false;

        while !self.is_at_end() {
            // If the previous token was `;`, we're at a statement boundary.
            // Note: we check before advancing to avoid skipping the sync token.

            match self.current().kind {
                // End of statement — consume and resume.
                TokenKind::Semicolon => {
                    self.advance();
                    self.recovered_count += 1;
                    return;
                }
                // End of block — don't consume, let the enclosing compound
                // statement parser handle it.
                TokenKind::RightBrace => {
                    self.recovered_count += 1;
                    return;
                }
                // Start of new block — don't consume, let the parser handle it.
                TokenKind::LeftBrace => {
                    self.recovered_count += 1;
                    return;
                }
                // Statement-starting keywords — resume parsing from here.
                TokenKind::If
                | TokenKind::For
                | TokenKind::While
                | TokenKind::Do
                | TokenKind::Switch
                | TokenKind::Return
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Goto
                | TokenKind::Case
                | TokenKind::Default => {
                    self.recovered_count += 1;
                    return;
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Synchronizes to the next top-level declaration boundary.
    ///
    /// Skips tokens until one of the following is found:
    /// - A type specifier keyword (`int`, `void`, `struct`, etc.)
    /// - A storage class keyword (`static`, `extern`, `typedef`, etc.)
    /// - A function specifier keyword (`inline`, `_Noreturn`)
    /// - `}` at brace nesting depth 0 (end of struct/union/enum body,
    ///   followed by consuming it)
    /// - EOF
    ///
    /// Used when a top-level declaration fails to parse and the parser needs
    /// to find the start of the next declaration.
    pub(crate) fn synchronize_to_declaration(&mut self) {
        let mut brace_depth: usize = 0;

        while !self.is_at_end() {
            let kind = self.current().kind;

            match kind {
                TokenKind::LeftBrace => {
                    brace_depth += 1;
                    self.advance();
                }
                TokenKind::RightBrace => {
                    if brace_depth == 0 {
                        // At top level, consume the closing brace and look
                        // for a semicolon that might follow (e.g., after
                        // `struct S { ... };`).
                        self.advance();
                        if self.check(TokenKind::Semicolon) {
                            self.advance();
                        }
                        return;
                    }
                    brace_depth -= 1;
                    self.advance();
                }
                TokenKind::Semicolon if brace_depth == 0 => {
                    self.advance();
                    return;
                }
                _ if brace_depth == 0 => {
                    // Check if this token could start a new declaration.
                    if kind.is_type_specifier()
                        || kind.is_storage_class()
                        || kind == TokenKind::Inline
                        || kind == TokenKind::Noreturn
                        || kind == TokenKind::GccAttribute
                        || kind == TokenKind::GccExtension
                        || kind == TokenKind::StaticAssert
                    {
                        return;
                    }
                    // Also check for typedef names (identifiers that are known
                    // typedef names).
                    if kind == TokenKind::Identifier {
                        if let TokenValue::Identifier(id) = self.current().value {
                            if self.typedef_names.contains(&id) {
                                return;
                            }
                        }
                    }
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Synchronizes to the next statement boundary within a compound block.
    ///
    /// Skips tokens until a `;`, `}`, or statement-starting keyword is found.
    /// Used when a statement fails to parse inside a block.
    pub(crate) fn synchronize_to_statement(&mut self) {
        while !self.is_at_end() {
            let kind = self.current().kind;
            match kind {
                TokenKind::Semicolon => {
                    self.advance();
                    return;
                }
                TokenKind::RightBrace | TokenKind::LeftBrace => {
                    return;
                }
                TokenKind::If
                | TokenKind::For
                | TokenKind::While
                | TokenKind::Do
                | TokenKind::Switch
                | TokenKind::Return
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Goto
                | TokenKind::Case
                | TokenKind::Default => {
                    return;
                }
                _ => {
                    // Check for declaration starters at statement level:
                    // type specifiers, storage class keywords, or any keyword
                    // that could begin a declaration.
                    if kind.is_type_specifier()
                        || kind.is_storage_class()
                        || kind.is_keyword()
                    {
                        return;
                    }
                    self.advance();
                }
            }
        }
    }

    // =======================================================================
    // Typedef Name Management
    // =======================================================================

    /// Registers an identifier as a typedef name.
    ///
    /// Called by `declarations.rs` when a `typedef` declaration is parsed.
    /// After registration, `is_typedef_name(name)` returns `true`, causing
    /// the parser to treat the identifier as a type specifier in subsequent
    /// parsing.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned identifier handle of the typedef name.
    pub(crate) fn register_typedef(&mut self, name: InternId) {
        self.typedef_names.insert(name);
    }

    /// Returns `true` if `name` is a currently-visible typedef name.
    ///
    /// This method is the critical disambiguation point in C parsing: when
    /// the parser encounters an identifier in a position that could be either
    /// a type specifier or a variable/function name, it checks this method
    /// to determine which interpretation to use.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned identifier handle to check.
    #[inline]
    pub(crate) fn is_typedef_name(&self, name: InternId) -> bool {
        self.typedef_names.contains(&name)
    }

    /// Pushes a new scope level for typedef name tracking.
    ///
    /// Called when entering a new lexical scope (compound statement, function
    /// body). A snapshot of the current typedef name set is saved so that
    /// [`pop_scope()`] can restore it when the scope exits.
    ///
    /// This enables correct handling of typedef shadowing: a variable
    /// declaration in an inner scope can shadow a typedef name from an outer
    /// scope without permanently affecting the typedef set.
    pub(crate) fn push_scope(&mut self) {
        self.scope_stack.push(self.typedef_names.clone());
    }

    /// Pops the most recent scope level, restoring the typedef name set to
    /// its state before the corresponding [`push_scope()`] call.
    ///
    /// Called when exiting a lexical scope. If the scope stack is empty (which
    /// should not happen in well-formed code), this method is a no-op.
    pub(crate) fn pop_scope(&mut self) {
        if let Some(saved) = self.scope_stack.pop() {
            self.typedef_names = saved;
        }
    }

    // =======================================================================
    // Nesting Depth Tracking
    // =======================================================================

    /// Attempts to enter a nested statement-level construct (compound
    /// statement / `{...}` block). Increments the statement nesting counter
    /// and checks against [`MAX_STATEMENT_NESTING_DEPTH`].
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the depth is within the allowed limit.
    /// * `Err(())` if the nesting depth limit has been exceeded. An error
    ///   diagnostic is emitted in GCC-compatible format and the caller should
    ///   return an error/recovery AST node rather than recursing further.
    pub(crate) fn enter_stmt_nesting(&mut self) -> Result<(), ()> {
        self.stmt_nesting_depth += 1;
        if self.stmt_nesting_depth > MAX_STATEMENT_NESTING_DEPTH {
            self.error(&format!(
                "nesting depth limit exceeded (maximum {} levels)",
                MAX_STATEMENT_NESTING_DEPTH
            ));
            Err(())
        } else {
            Ok(())
        }
    }

    /// Exits a nested statement-level construct, decrementing the statement
    /// nesting counter.
    pub(crate) fn exit_stmt_nesting(&mut self) {
        self.stmt_nesting_depth = self.stmt_nesting_depth.saturating_sub(1);
    }

    /// Attempts to enter a nested expression-level construct (parenthesized
    /// expression recursion). Increments the expression nesting counter and
    /// checks against [`MAX_EXPRESSION_NESTING_DEPTH`].
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the depth is within the allowed limit.
    /// * `Err(())` if the nesting depth limit has been exceeded. An error
    ///   diagnostic is emitted in GCC-compatible format and the caller should
    ///   return an Error expression rather than recursing further.
    pub(crate) fn enter_expr_nesting(&mut self) -> Result<(), ()> {
        self.expr_nesting_depth += 1;
        if self.expr_nesting_depth > MAX_EXPRESSION_NESTING_DEPTH {
            self.error(&format!(
                "expression nesting depth limit exceeded (maximum {} levels)",
                MAX_EXPRESSION_NESTING_DEPTH
            ));
            Err(())
        } else {
            Ok(())
        }
    }

    /// Exits a nested expression-level construct, decrementing the expression
    /// nesting counter.
    pub(crate) fn exit_expr_nesting(&mut self) {
        self.expr_nesting_depth = self.expr_nesting_depth.saturating_sub(1);
    }

    // =======================================================================
    // Source Span Construction Helpers
    // =======================================================================

    /// Returns the source span of the current token.
    ///
    /// Used by submodule parsers to capture the start position of a construct
    /// before parsing its body.
    #[inline]
    pub(crate) fn current_span(&self) -> SourceSpan {
        self.current().span
    }

    /// Creates a span that extends from `start` to the end of the most
    /// recently consumed token (the "previous" token).
    ///
    /// This is the standard pattern for constructing AST node spans:
    /// 1. Save `start = parser.current_span()` before parsing the construct.
    /// 2. Parse the construct (consuming multiple tokens).
    /// 3. Compute the final span with `parser.span_from(start)`.
    pub(crate) fn span_from(&self, start: SourceSpan) -> SourceSpan {
        let end = self.previous_span();
        SourceSpan {
            start: start.start,
            end: end.end,
        }
    }

    /// Returns the source span of the previously consumed token (the token
    /// at position `pos - 1`).
    ///
    /// If no tokens have been consumed yet (`pos == 0`), returns the span of
    /// the first token.
    #[inline]
    pub(crate) fn previous_span(&self) -> SourceSpan {
        let tokens = self.tokens;
        if self.pos > 0 {
            tokens[self.pos - 1].span
        } else {
            tokens[0].span
        }
    }

    // =======================================================================
    // Query Methods
    // =======================================================================

    /// Returns the total number of parse errors encountered so far.
    #[inline]
    pub(crate) fn get_error_count(&self) -> usize {
        self.error_count
    }

    /// Returns the number of errors from which the parser recovered.
    #[inline]
    pub(crate) fn get_recovered_count(&self) -> usize {
        self.recovered_count
    }

    /// Returns `true` if the parser has encountered any errors.
    #[inline]
    pub(crate) fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Returns the current token position index. Used by the compound statement
    /// and translation unit parsers to detect infinite-loop conditions where
    /// error recovery fails to make forward progress.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    /// Returns the current token's identifier InternId, if it is an
    /// identifier token. Returns `None` for non-identifier tokens.
    ///
    /// This is a convenience method used frequently by declaration and type
    /// parsing for extracting names from identifier tokens.
    pub(crate) fn current_identifier(&self) -> Option<InternId> {
        if let TokenValue::Identifier(id) = self.current().value {
            Some(id)
        } else {
            None
        }
    }

    /// Returns a reference to the interner, allowing submodules to resolve
    /// interned strings for error messages and diagnostics.
    #[inline]
    pub(crate) fn interner(&self) -> &'a Interner {
        self.interner
    }

    /// Looks up a string in the interner without creating a new entry.
    ///
    /// Returns `Some(InternId)` if the string was previously interned (e.g.,
    /// by the lexer), or `None` if it is unknown. Useful for checking built-in
    /// names or compiler-generated identifiers.
    #[inline]
    pub(crate) fn lookup_interned(&self, name: &str) -> Option<InternId> {
        self.interner.get(name)
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEmitter;
    use crate::common::intern::Interner;
    use crate::common::source_map::{SourceLocation, SourceSpan};
    use crate::frontend::lexer::token::{Token, TokenKind, TokenValue};

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Creates a dummy SourceSpan for testing.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Creates a simple token with the given kind and no value.
    fn make_token(kind: TokenKind) -> Token {
        Token::new(kind, dummy_span(), TokenValue::None)
    }

    /// Creates an identifier token with the given InternId.
    fn make_ident_token(id: InternId) -> Token {
        Token::new(TokenKind::Identifier, dummy_span(), TokenValue::Identifier(id))
    }

    /// Creates a token stream with the given tokens plus an EOF sentinel.
    fn make_tokens(kinds: &[TokenKind]) -> Vec<Token> {
        let mut tokens: Vec<Token> = kinds.iter().map(|&k| make_token(k)).collect();
        tokens.push(make_token(TokenKind::Eof));
        tokens
    }

    /// Creates a parser for testing with the given tokens.
    fn make_parser<'a>(
        tokens: &'a [Token],
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Parser<'a> {
        Parser::new(tokens, interner, diagnostics)
    }

    // -----------------------------------------------------------------------
    // Token Access Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_returns_first_token() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.current().kind, TokenKind::Int);
    }

    #[test]
    fn test_peek_returns_next_token() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.peek().kind, TokenKind::Semicolon);
    }

    #[test]
    fn test_lookahead_zero_is_current() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.lookahead(0).kind, TokenKind::Int);
    }

    #[test]
    fn test_lookahead_beyond_end_returns_eof() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.lookahead(100).kind, TokenKind::Eof);
    }

    #[test]
    fn test_advance_returns_consumed_token() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let consumed = parser.advance();
        assert_eq!(consumed.kind, TokenKind::Int);
        assert_eq!(parser.current().kind, TokenKind::Semicolon);
    }

    #[test]
    fn test_advance_does_not_pass_eof() {
        let tokens = make_tokens(&[]); // Just EOF
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.current().kind, TokenKind::Eof);
        parser.advance();
        assert_eq!(parser.current().kind, TokenKind::Eof);
    }

    #[test]
    fn test_is_at_end_with_eof() {
        let tokens = make_tokens(&[]); // Just EOF
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(parser.is_at_end());
    }

    #[test]
    fn test_is_at_end_with_tokens() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(!parser.is_at_end());
    }

    // -----------------------------------------------------------------------
    // Token Matching Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_matches_current() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(parser.check(TokenKind::Int));
        assert!(!parser.check(TokenKind::Float));
    }

    #[test]
    fn test_match_token_consumes_on_match() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        assert!(parser.match_token(TokenKind::Int));
        assert_eq!(parser.current().kind, TokenKind::Semicolon);
    }

    #[test]
    fn test_match_token_does_not_consume_on_mismatch() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        assert!(!parser.match_token(TokenKind::Float));
        assert_eq!(parser.current().kind, TokenKind::Int);
    }

    #[test]
    fn test_expect_success() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parser.expect(TokenKind::Int);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().kind, TokenKind::Int);
        assert_eq!(parser.current().kind, TokenKind::Semicolon);
    }

    #[test]
    fn test_expect_failure_reports_error() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parser.expect(TokenKind::Float);
        assert!(result.is_err());
        assert!(parser.has_errors());
        assert_eq!(parser.get_error_count(), 1);
    }

    #[test]
    fn test_match_any_finds_matching_kind() {
        let tokens = make_tokens(&[TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parser.match_any(&[TokenKind::Comma, TokenKind::Semicolon]);
        assert_eq!(result, Some(TokenKind::Semicolon));
    }

    #[test]
    fn test_match_any_returns_none_on_no_match() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parser.match_any(&[TokenKind::Comma, TokenKind::Semicolon]);
        assert_eq!(result, None);
        assert_eq!(parser.current().kind, TokenKind::Int);
    }

    // -----------------------------------------------------------------------
    // Error Reporting Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_increments_count_and_enters_panic() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.error("test error");
        assert_eq!(parser.get_error_count(), 1);
        assert!(parser.in_panic_mode);
        assert!(diag.has_errors());
    }

    #[test]
    fn test_error_at_uses_specific_span() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let span = dummy_span();
        parser.error_at(span, "error at specific location");
        assert_eq!(parser.get_error_count(), 1);
        assert!(parser.in_panic_mode);
    }

    #[test]
    fn test_unexpected_token_generates_message() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.unexpected_token(";");
        assert_eq!(parser.get_error_count(), 1);
        assert!(diag.has_errors());
    }

    // -----------------------------------------------------------------------
    // Error Recovery Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_synchronize_stops_at_semicolon() {
        let tokens = make_tokens(&[
            TokenKind::Int,    // skip
            TokenKind::Star,   // skip
            TokenKind::Semicolon, // sync point
            TokenKind::Float,  // should be current after sync
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        parser.in_panic_mode = true;

        parser.synchronize();
        // After synchronize, semicolon was consumed, current should be Float
        assert_eq!(parser.current().kind, TokenKind::Float);
        assert!(!parser.in_panic_mode);
    }

    #[test]
    fn test_synchronize_stops_at_right_brace() {
        let tokens = make_tokens(&[
            TokenKind::Int,
            TokenKind::RightBrace, // sync point (not consumed)
            TokenKind::Float,
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        parser.in_panic_mode = true;

        parser.synchronize();
        // RightBrace is NOT consumed — current should be RightBrace
        assert_eq!(parser.current().kind, TokenKind::RightBrace);
    }

    #[test]
    fn test_synchronize_stops_at_statement_keyword() {
        let tokens = make_tokens(&[
            TokenKind::Star,   // skip
            TokenKind::Return, // sync point (not consumed)
            TokenKind::Semicolon,
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        parser.in_panic_mode = true;

        parser.synchronize();
        assert_eq!(parser.current().kind, TokenKind::Return);
    }

    #[test]
    fn test_synchronize_handles_eof() {
        let tokens = make_tokens(&[]); // Just EOF
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);
        parser.in_panic_mode = true;

        parser.synchronize();
        assert!(parser.is_at_end());
    }

    // -----------------------------------------------------------------------
    // Typedef Name Management Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_typedef_and_lookup() {
        let tokens = make_tokens(&[]);
        let mut interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let my_int = InternId::from_raw(42);
        parser.register_typedef(my_int);
        assert!(parser.is_typedef_name(my_int));
    }

    #[test]
    fn test_non_typedef_name_returns_false() {
        let tokens = make_tokens(&[]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        let random_name = InternId::from_raw(99);
        assert!(!parser.is_typedef_name(random_name));
    }

    #[test]
    fn test_push_pop_scope_restores_typedefs() {
        let tokens = make_tokens(&[]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let outer_typedef = InternId::from_raw(10);
        let inner_typedef = InternId::from_raw(20);

        // Register a typedef in the outer scope.
        parser.register_typedef(outer_typedef);
        assert!(parser.is_typedef_name(outer_typedef));

        // Enter inner scope and register another typedef.
        parser.push_scope();
        parser.register_typedef(inner_typedef);
        assert!(parser.is_typedef_name(outer_typedef));
        assert!(parser.is_typedef_name(inner_typedef));

        // Exit inner scope — inner_typedef should no longer be visible.
        parser.pop_scope();
        assert!(parser.is_typedef_name(outer_typedef));
        assert!(!parser.is_typedef_name(inner_typedef));
    }

    #[test]
    fn test_nested_scopes() {
        let tokens = make_tokens(&[]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let name_a = InternId::from_raw(1);
        let name_b = InternId::from_raw(2);
        let name_c = InternId::from_raw(3);

        parser.register_typedef(name_a);
        parser.push_scope();
        parser.register_typedef(name_b);
        parser.push_scope();
        parser.register_typedef(name_c);

        // All visible in innermost scope.
        assert!(parser.is_typedef_name(name_a));
        assert!(parser.is_typedef_name(name_b));
        assert!(parser.is_typedef_name(name_c));

        parser.pop_scope();
        assert!(parser.is_typedef_name(name_a));
        assert!(parser.is_typedef_name(name_b));
        assert!(!parser.is_typedef_name(name_c));

        parser.pop_scope();
        assert!(parser.is_typedef_name(name_a));
        assert!(!parser.is_typedef_name(name_b));
        assert!(!parser.is_typedef_name(name_c));
    }

    #[test]
    fn test_pop_scope_empty_stack_is_noop() {
        let tokens = make_tokens(&[]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        // Popping an empty scope stack should not panic.
        parser.pop_scope();
        assert!(!parser.has_errors());
    }

    // -----------------------------------------------------------------------
    // Span Construction Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_span_matches_current_token() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        let span = parser.current_span();
        assert_eq!(span, parser.current().span);
    }

    #[test]
    fn test_span_from_covers_range() {
        let loc1 = SourceLocation {
            file_id: crate::common::source_map::FileId(0),
            byte_offset: 0,
            line: 1,
            column: 1,
        };
        let loc2 = SourceLocation {
            file_id: crate::common::source_map::FileId(0),
            byte_offset: 10,
            line: 1,
            column: 11,
        };
        let span1 = SourceSpan { start: loc1, end: loc1 };
        let span2 = SourceSpan { start: loc2, end: loc2 };

        let tok1 = Token::new(TokenKind::Int, span1, TokenValue::None);
        let tok2 = Token::new(TokenKind::Semicolon, span2, TokenValue::None);
        let eof_tok = Token::new(TokenKind::Eof, dummy_span(), TokenValue::None);
        let tokens = vec![tok1, tok2, eof_tok];

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let start = parser.current_span();
        parser.advance(); // consume Int
        parser.advance(); // consume Semicolon
        let full_span = parser.span_from(start);

        assert_eq!(full_span.start, loc1);
        assert_eq!(full_span.end, loc2);
    }

    #[test]
    fn test_previous_span_after_advance() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.advance(); // consume Int
        let prev = parser.previous_span();
        assert_eq!(prev, tokens[0].span);
    }

    #[test]
    fn test_previous_span_at_start_returns_first_token() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        let prev = parser.previous_span();
        assert_eq!(prev, tokens[0].span);
    }

    // -----------------------------------------------------------------------
    // EOF Handling Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eof_only_stream() {
        let tokens = make_tokens(&[]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(parser.is_at_end());
        assert_eq!(parser.current().kind, TokenKind::Eof);
        assert_eq!(parser.peek().kind, TokenKind::Eof);
    }

    #[test]
    fn test_advance_sequence_to_eof() {
        let tokens = make_tokens(&[TokenKind::Int, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.advance().kind, TokenKind::Int);
        assert_eq!(parser.advance().kind, TokenKind::Semicolon);
        assert!(parser.is_at_end());
        // Advancing past EOF stays at EOF.
        assert_eq!(parser.advance().kind, TokenKind::Eof);
        assert!(parser.is_at_end());
    }

    // -----------------------------------------------------------------------
    // Identifier Helper Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_identifier_returns_id() {
        let id = InternId::from_raw(42);
        let tok = Token::new(
            TokenKind::Identifier,
            dummy_span(),
            TokenValue::Identifier(id),
        );
        let eof = make_token(TokenKind::Eof);
        let tokens = vec![tok, eof];

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.current_identifier(), Some(id));
    }

    #[test]
    fn test_current_identifier_returns_none_for_keyword() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert_eq!(parser.current_identifier(), None);
    }

    // -----------------------------------------------------------------------
    // Warning Method Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_warning_does_not_affect_error_count() {
        let tokens = make_tokens(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.warning("test warning");
        assert_eq!(parser.get_error_count(), 0);
        assert!(!parser.has_errors());
        assert!(!parser.in_panic_mode);
    }
}
