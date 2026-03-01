//! Statement parsing for the `bcc` C compiler.
//!
//! This module handles all C11 statement types plus GCC extensions:
//!
//! | Category          | Variants                                            |
//! |-------------------|-----------------------------------------------------|
//! | Compound          | `{ ... }` blocks                                    |
//! | Selection         | `if`/`else`, `switch`/`case`/`default`              |
//! | Iteration         | `for`, `while`, `do`-`while`                        |
//! | Jump              | `break`, `continue`, `return`, `goto`               |
//! | Labeled           | `label: stmt`, `case val:`, `default:`              |
//! | Expression        | `expr;` and null `;`                                |
//! | GCC extensions    | `asm`/`__asm__`, computed `goto *expr`, `__extension__` |
//!
//! # Error Recovery
//!
//! On syntax errors, the parser synchronises on statement boundaries (`;`, `}`,
//! or statement-starting keywords) to achieve the ≥80% error recovery rate
//! specified in AAP §0.2.1.
//!
//! # Integration Points
//!
//! - **Called by** `mod.rs` (`Parser::parse`) for function body parsing.
//! - **Called by** `declarations.rs` for function definition bodies.
//! - **Called by** `gcc_extensions.rs` for statement expression bodies.
//! - **Calls** `expressions.rs` for condition parsing, return values, expression
//!   statements.
//! - **Calls** `declarations.rs` for block-level declarations.
//! - **Calls** `gcc_extensions.rs` for asm statements, computed goto, and
//!   `__extension__`-prefixed statements.
//! - Per AAP §0.5.1 Group 2: "Statement parsing: if/else, for, while,
//!   do-while, switch/case, break, continue, return, goto, labeled."
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use super::ast::{BlockItem, Declaration, ForInit, Statement};
use super::declarations;
use super::expressions;
use super::gcc_extensions;
use super::types;
use super::Parser;
use crate::frontend::lexer::token::{TokenKind, TokenValue};

// ===========================================================================
// Public API — Statement & Compound Statement Parsing
// ===========================================================================

/// Parses a single C statement.
///
/// This is the main entry point for statement parsing, dispatching to the
/// appropriate specialised parser based on the current token:
///
/// | Current token                     | Action                                |
/// |-----------------------------------|---------------------------------------|
/// | `{`                               | compound statement                    |
/// | `if`                              | if/else statement                     |
/// | `for`                             | for loop                              |
/// | `while`                           | while loop                            |
/// | `do`                              | do-while loop                         |
/// | `switch`                          | switch statement                      |
/// | `case`                            | case label                            |
/// | `default`                         | default label                         |
/// | `break`                           | break statement                       |
/// | `continue`                        | continue statement                    |
/// | `return`                          | return statement                      |
/// | `goto`                            | goto / computed goto                  |
/// | `asm` / `__asm__`                 | inline assembly                       |
/// | `__extension__`                   | GCC extension prefix                  |
/// | `;`                               | null (empty) statement                |
/// | identifier `:` (lookahead)        | labeled statement                     |
/// | type / storage class / qualifier  | block-level declaration               |
/// | *anything else*                   | expression statement                  |
///
/// # Declaration vs Expression Disambiguation
///
/// When the current token is an identifier, the parser must determine whether
/// it starts a declaration (typedef name used as type specifier) or an
/// expression statement. This is resolved via [`Parser::is_typedef_name()`]:
/// if the identifier is a registered typedef name, the token is treated as a
/// type specifier and routed to declaration parsing.
pub(super) fn parse_statement(parser: &mut Parser<'_>) -> Statement {
    match parser.current().kind {
        // ---------------------------------------------------------------
        // Compound statement (block)
        // ---------------------------------------------------------------
        TokenKind::LeftBrace => parse_compound_statement(parser),

        // ---------------------------------------------------------------
        // Selection statements
        // ---------------------------------------------------------------
        TokenKind::If => parse_if_statement(parser),
        TokenKind::Switch => parse_switch_statement(parser),

        // ---------------------------------------------------------------
        // Iteration statements
        // ---------------------------------------------------------------
        TokenKind::For => parse_for_statement(parser),
        TokenKind::While => parse_while_statement(parser),
        TokenKind::Do => parse_do_while_statement(parser),

        // ---------------------------------------------------------------
        // Switch labels (can appear at statement level within a switch body)
        // ---------------------------------------------------------------
        TokenKind::Case => parse_case_statement(parser),
        TokenKind::Default => parse_default_statement(parser),

        // ---------------------------------------------------------------
        // Jump statements
        // ---------------------------------------------------------------
        TokenKind::Break => parse_break_statement(parser),
        TokenKind::Continue => parse_continue_statement(parser),
        TokenKind::Return => parse_return_statement(parser),
        TokenKind::Goto => parse_goto_statement(parser),

        // ---------------------------------------------------------------
        // GCC extensions delegated to gcc_extensions module
        // ---------------------------------------------------------------
        TokenKind::Asm => gcc_extensions::parse_asm_statement(parser),
        TokenKind::GccExtension => parse_extension_statement(parser),

        // ---------------------------------------------------------------
        // Null (empty) statement: `;`
        // ---------------------------------------------------------------
        TokenKind::Semicolon => {
            let span = parser.current_span();
            parser.advance();
            Statement::Null { span }
        }

        // ---------------------------------------------------------------
        // Identifier: could be a labeled statement, typedef-based
        // declaration, or expression statement.
        // ---------------------------------------------------------------
        TokenKind::Identifier => {
            // Check for labeled statement: identifier followed by `:`
            // but NOT `::` which doesn't exist in C but we guard anyway.
            if parser.peek().kind == TokenKind::Colon {
                // Could be a label — but first check it's not a typedef name
                // starting a declaration. If it IS a typedef name, we still
                // check for label syntax because labels take priority.
                return parse_labeled_statement(parser);
            }

            // Check if identifier is a typedef name → declaration.
            if let TokenValue::Identifier(id) = parser.current().value {
                if parser.is_typedef_name(id) {
                    return parse_declaration_statement(parser);
                }
            }

            // Fall through to expression statement.
            parse_expression_statement(parser)
        }

        // ---------------------------------------------------------------
        // Declaration starters: type specifiers, storage classes,
        // type qualifiers, and C11 special keywords.
        // ---------------------------------------------------------------
        kind if is_declaration_start(parser, kind) => {
            parse_declaration_statement(parser)
        }

        // ---------------------------------------------------------------
        // Fallback: expression statement
        // ---------------------------------------------------------------
        _ => parse_expression_statement(parser),
    }
}

/// Parses a compound statement (block): `{ block-item-list? }`.
///
/// Each block item is either a declaration or a statement. The opening `{`
/// has **not** been consumed yet — this function consumes both the opening
/// `{` and the closing `}`.
///
/// Grammar:
/// ```text
///   compound-statement := '{' block-item-list? '}'
///   block-item-list    := block-item | block-item-list block-item
///   block-item         := declaration | statement
/// ```
///
/// # Scope Management
///
/// A compound statement opens a new lexical scope for typedef name tracking.
/// [`Parser::push_scope()`] is called on entry and [`Parser::pop_scope()`]
/// on exit.
///
/// # Error Recovery
///
/// If an unexpected token is encountered while parsing block items, the parser
/// synchronises to the next `;` or `}` and continues.
pub(super) fn parse_compound_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume the opening `{`.
    if parser.expect(TokenKind::LeftBrace).is_err() {
        // If we cannot even find the opening brace, return a Null statement
        // as a recovery measure.
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Enter a new scope for typedef name tracking.
    parser.push_scope();

    // Parse optional GCC attributes at the start of the block.
    let _attrs = gcc_extensions::try_parse_attributes(parser);

    let mut items: Vec<BlockItem> = Vec::new();

    // Parse block items until `}` or EOF.
    while !parser.check(TokenKind::RightBrace) && !parser.is_at_end() {
        let pos_before = parser.position();

        // Try to parse a block item (declaration or statement).
        let item = parse_block_item(parser);
        items.push(item);

        // Guard against infinite loops: if the parser made no forward
        // progress (the position didn't change after a full block-item
        // parse attempt), force-advance past the problematic token. This
        // handles cases where error recovery returns to an unrecognised
        // token without consuming it (e.g., `_Alignas` before the
        // alignment-specifier is fully implemented).
        if parser.position() == pos_before && !parser.is_at_end() {
            parser.advance();
        }

        // If the parser entered panic mode during the block item parse,
        // synchronise to the next statement boundary.
        if parser.current().kind == TokenKind::Eof {
            break;
        }
    }

    // Exit the scope.
    parser.pop_scope();

    // Consume the closing `}`.
    if parser.expect(TokenKind::RightBrace).is_err() {
        // Missing `}` — report error but continue; span ends at current position.
    }

    let span = parser.span_from(start);
    Statement::Compound { items, span }
}

// ===========================================================================
// Block Item Parsing
// ===========================================================================

/// Parses a single block item: either a declaration or a statement.
///
/// The disambiguation between declaration and statement uses the same logic
/// as `parse_statement`, but wraps declarations in `BlockItem::Declaration`.
fn parse_block_item(parser: &mut Parser<'_>) -> BlockItem {
    let kind = parser.current().kind;

    // Check for declaration starters.
    if is_declaration_start(parser, kind) {
        let decl = parse_block_declaration(parser);
        return BlockItem::Declaration(Box::new(decl));
    }

    // Check for identifier-as-typedef-name starting a declaration.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            // If it's a typedef name AND not followed by `:` (label), it's
            // a declaration.
            if parser.is_typedef_name(id) && parser.peek().kind != TokenKind::Colon {
                let decl = parse_block_declaration(parser);
                return BlockItem::Declaration(Box::new(decl));
            }
        }
    }

    // Otherwise it's a statement.
    let stmt = parse_statement(parser);
    BlockItem::Statement(stmt)
}

// ===========================================================================
// If/Else Statement
// ===========================================================================

/// Parses an if statement with optional else clause.
///
/// Grammar:
/// ```text
///   if-statement := 'if' '(' expression ')' statement
///                 | 'if' '(' expression ')' statement 'else' statement
/// ```
///
/// The dangling-else ambiguity is resolved naturally by the recursive-descent
/// structure: `else` always binds to the nearest unmatched `if`.
fn parse_if_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `if`.
    parser.advance();

    // Expect `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse condition expression.
    let condition = expressions::parse_expression(parser);

    // Expect `)`.
    if parser.expect(TokenKind::RightParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse the then-branch.
    let then_branch = parse_statement(parser);

    // Optionally parse the else-branch.
    let else_branch = if parser.match_token(TokenKind::Else) {
        Some(Box::new(parse_statement(parser)))
    } else {
        None
    };

    let span = parser.span_from(start);
    Statement::If {
        condition: Box::new(condition),
        then_branch: Box::new(then_branch),
        else_branch,
        span,
    }
}

// ===========================================================================
// For Loop
// ===========================================================================

/// Parses a for loop statement.
///
/// Grammar:
/// ```text
///   for-statement := 'for' '(' for-init? ';' expression? ';' expression? ')' statement
///   for-init      := declaration | expression
/// ```
///
/// C99 allows declarations in the for-init clause (e.g., `for (int i = 0; ...)`).
/// When the for-init is a declaration, the parser does NOT expect a separate
/// `;` because the declaration already consumes its trailing semicolon.
fn parse_for_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `for`.
    parser.advance();

    // Expect `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse init clause.
    let init = parse_for_init(parser);

    // Parse condition (optional).
    let condition = if parser.check(TokenKind::Semicolon) {
        None
    } else {
        Some(Box::new(expressions::parse_expression(parser)))
    };

    // Expect `;` after condition.
    if parser.expect(TokenKind::Semicolon).is_err() {
        // Try to recover by looking for `)`.
        if !parser.check(TokenKind::RightParen) {
            synchronize_statement(parser);
            return Statement::Null {
                span: parser.span_from(start),
            };
        }
    }

    // Parse increment (optional).
    let increment = if parser.check(TokenKind::RightParen) {
        None
    } else {
        Some(Box::new(expressions::parse_expression(parser)))
    };

    // Expect `)`.
    if parser.expect(TokenKind::RightParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse loop body.
    let body = parse_statement(parser);

    let span = parser.span_from(start);
    Statement::For {
        init,
        condition,
        increment,
        body: Box::new(body),
        span,
    }
}

/// Parses the for-loop init clause: empty `;`, declaration, or expression `;`.
///
/// If the init is a declaration, its trailing `;` is consumed by the
/// declaration parser and this function does NOT consume an additional `;`.
/// If the init is empty or an expression, this function consumes the `;`.
fn parse_for_init(parser: &mut Parser<'_>) -> Option<Box<ForInit>> {
    // Empty init: just `;`.
    if parser.match_token(TokenKind::Semicolon) {
        return None;
    }

    let kind = parser.current().kind;

    // Check if this looks like a declaration.
    if is_declaration_start(parser, kind) {
        let decl = parse_block_declaration(parser);
        return Some(Box::new(ForInit::Declaration(Box::new(decl))));
    }

    // Check for typedef name starting a declaration.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            if parser.is_typedef_name(id) {
                let decl = parse_block_declaration(parser);
                return Some(Box::new(ForInit::Declaration(Box::new(decl))));
            }
        }
    }

    // Otherwise it's an expression.
    let expr = expressions::parse_expression(parser);

    // Consume the trailing `;`.
    if parser.expect(TokenKind::Semicolon).is_err() {
        // Best-effort recovery — continue anyway.
    }

    Some(Box::new(ForInit::Expression(Box::new(expr))))
}

// ===========================================================================
// While Loop
// ===========================================================================

/// Parses a while loop statement.
///
/// Grammar:
/// ```text
///   while-statement := 'while' '(' expression ')' statement
/// ```
fn parse_while_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `while`.
    parser.advance();

    // Expect `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse condition expression.
    let condition = expressions::parse_expression(parser);

    // Expect `)`.
    if parser.expect(TokenKind::RightParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse loop body.
    let body = parse_statement(parser);

    let span = parser.span_from(start);
    Statement::While {
        condition: Box::new(condition),
        body: Box::new(body),
        span,
    }
}

// ===========================================================================
// Do-While Loop
// ===========================================================================

/// Parses a do-while loop statement.
///
/// Grammar:
/// ```text
///   do-while-statement := 'do' statement 'while' '(' expression ')' ';'
/// ```
fn parse_do_while_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `do`.
    parser.advance();

    // Parse loop body.
    let body = parse_statement(parser);

    // Expect `while`.
    if parser.expect(TokenKind::While).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Expect `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse condition expression.
    let condition = expressions::parse_expression(parser);

    // Expect `)`.
    if parser.expect(TokenKind::RightParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Expect `;` — a common omission error.
    if parser.expect(TokenKind::Semicolon).is_err() {
        // Report but continue: the do-while is structurally complete.
    }

    let span = parser.span_from(start);
    Statement::DoWhile {
        body: Box::new(body),
        condition: Box::new(condition),
        span,
    }
}

// ===========================================================================
// Switch / Case / Default
// ===========================================================================

/// Parses a switch statement.
///
/// Grammar:
/// ```text
///   switch-statement := 'switch' '(' expression ')' statement
/// ```
///
/// The body is typically a compound statement containing `case` and `default`
/// labels, but the C standard allows any statement as the switch body.
fn parse_switch_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `switch`.
    parser.advance();

    // Expect `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse controlling expression.
    let expr = expressions::parse_expression(parser);

    // Expect `)`.
    if parser.expect(TokenKind::RightParen).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse body (usually a compound statement).
    let body = parse_statement(parser);

    let span = parser.span_from(start);
    Statement::Switch {
        expr: Box::new(expr),
        body: Box::new(body),
        span,
    }
}

/// Parses a `case` label.
///
/// Grammar:
/// ```text
///   case-label := 'case' constant-expression ':' statement
///               | 'case' constant-expression '...' constant-expression ':' statement
/// ```
///
/// The second form is the GCC case range extension (`case 1 ... 5:`).
fn parse_case_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `case`.
    parser.advance();

    // Parse the case value (constant expression).
    let value = expressions::parse_assignment_expression(parser);

    // Check for GCC case range extension: `case low ... high:`.
    let range_end = if parser.check(TokenKind::Ellipsis) {
        parser.advance();
        Some(Box::new(expressions::parse_assignment_expression(parser)))
    } else {
        None
    };

    // Expect `:`.
    if parser.expect(TokenKind::Colon).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse the following statement.
    let body = parse_statement(parser);

    let span = parser.span_from(start);
    Statement::Case {
        value: Box::new(value),
        range_end,
        body: Box::new(body),
        span,
    }
}

/// Parses a `default` label.
///
/// Grammar:
/// ```text
///   default-label := 'default' ':' statement
/// ```
fn parse_default_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `default`.
    parser.advance();

    // Expect `:`.
    if parser.expect(TokenKind::Colon).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse the following statement.
    let body = parse_statement(parser);

    let span = parser.span_from(start);
    Statement::Default {
        body: Box::new(body),
        span,
    }
}

// ===========================================================================
// Jump Statements
// ===========================================================================

/// Parses a `break` statement: `break ;`.
fn parse_break_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();
    parser.advance(); // consume `break`

    if parser.expect(TokenKind::Semicolon).is_err() {
        // Missing semicolon — continue anyway.
    }

    let span = parser.span_from(start);
    Statement::Break { span }
}

/// Parses a `continue` statement: `continue ;`.
fn parse_continue_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();
    parser.advance(); // consume `continue`

    if parser.expect(TokenKind::Semicolon).is_err() {
        // Missing semicolon — continue anyway.
    }

    let span = parser.span_from(start);
    Statement::Continue { span }
}

/// Parses a `return` statement: `return ;` or `return expression ;`.
fn parse_return_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();
    parser.advance(); // consume `return`

    // If the next token is `;`, it's a bare `return;`.
    let value = if parser.check(TokenKind::Semicolon) {
        None
    } else {
        Some(Box::new(expressions::parse_expression(parser)))
    };

    if parser.expect(TokenKind::Semicolon).is_err() {
        // Missing semicolon — continue anyway.
    }

    let span = parser.span_from(start);
    Statement::Return { value, span }
}

/// Parses a `goto` statement: `goto label ;` or GCC `goto *expr ;`.
///
/// For computed goto (`goto *expr`), parsing is delegated to
/// [`gcc_extensions::parse_computed_goto`].
fn parse_goto_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();
    parser.advance(); // consume `goto`

    // GCC computed goto: `goto *expr;`
    if parser.check(TokenKind::Star) {
        return gcc_extensions::parse_computed_goto(parser);
    }

    // Standard goto: `goto label;`
    let label = match parser.current().value {
        TokenValue::Identifier(id) => {
            parser.advance();
            id
        }
        _ => {
            parser.error("expected label name after 'goto'");
            synchronize_statement(parser);
            return Statement::Null {
                span: parser.span_from(start),
            };
        }
    };

    if parser.expect(TokenKind::Semicolon).is_err() {
        // Missing semicolon — continue anyway.
    }

    let span = parser.span_from(start);
    Statement::Goto { label, span }
}

// ===========================================================================
// Labeled Statement
// ===========================================================================

/// Parses a labeled statement: `identifier ':' statement`.
///
/// The caller has already determined that the current token is an identifier
/// followed by `:`. GCC `__attribute__` annotations on labels are supported.
///
/// Grammar:
/// ```text
///   labeled-statement := IDENTIFIER ':' attribute-list? statement
/// ```
fn parse_labeled_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Extract the label name.
    let label = match parser.current().value {
        TokenValue::Identifier(id) => id,
        _ => {
            // Should not reach here — caller verified it's an identifier.
            parser.error("expected label name");
            synchronize_statement(parser);
            return Statement::Null {
                span: parser.span_from(start),
            };
        }
    };

    // Consume the identifier.
    parser.advance();

    // Consume `:`.
    if parser.expect(TokenKind::Colon).is_err() {
        synchronize_statement(parser);
        return Statement::Null {
            span: parser.span_from(start),
        };
    }

    // Parse optional GCC attributes on the label.
    let attributes = gcc_extensions::try_parse_attributes(parser);

    // Parse the following statement.
    let body = parse_statement(parser);

    let span = parser.span_from(start);
    Statement::Labeled {
        label,
        body: Box::new(body),
        attributes,
        span,
    }
}

// ===========================================================================
// Expression Statement
// ===========================================================================

/// Parses an expression statement: `expression ';'`.
///
/// This is the fallback when no statement keyword is matched. The full
/// expression (including the comma operator) is parsed.
fn parse_expression_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    let expr = expressions::parse_expression(parser);

    if parser.expect(TokenKind::Semicolon).is_err() {
        // Missing semicolon after expression — report but continue.
    }

    let span = parser.span_from(start);
    Statement::Expression {
        expr: Box::new(expr),
        span,
    }
}

// ===========================================================================
// GCC __extension__ Statement
// ===========================================================================

/// Handles the `__extension__` keyword at statement level.
///
/// `__extension__` is a GCC prefix that suppresses pedantic warnings. At
/// statement level, it can precede a declaration or a statement. We consume
/// the prefix and parse the following construct as a normal statement.
fn parse_extension_statement(parser: &mut Parser<'_>) -> Statement {
    let start = parser.current_span();

    // Consume `__extension__`.
    parser.advance();

    // The construct following __extension__ could be a declaration or statement.
    // Check for declaration starters.
    let kind = parser.current().kind;
    if is_declaration_start(parser, kind) {
        let decl = parse_block_declaration(parser);
        return Statement::Declaration(Box::new(decl));
    }

    // Check for typedef name starting a declaration.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            if parser.is_typedef_name(id) && parser.peek().kind != TokenKind::Colon {
                let decl = parse_block_declaration(parser);
                return Statement::Declaration(Box::new(decl));
            }
        }
    }

    // Otherwise parse as a regular statement.
    parse_statement(parser)
}

// ===========================================================================
// Declaration Helpers
// ===========================================================================

/// Parses a block-level declaration.
///
/// This delegates to the declarations module's `parse_external_declaration`
/// function, which handles all declaration forms (variable, function, typedef,
/// struct/union/enum, `_Static_assert`). In a compound block context, function
/// definitions are semantically invalid but syntactically parsed for better
/// error reporting.
fn parse_block_declaration(parser: &mut Parser<'_>) -> Declaration {
    declarations::parse_external_declaration(parser)
}

/// Parses a declaration and wraps it as a `Statement::Declaration`.
///
/// Used when the statement dispatcher determines the current tokens form a
/// declaration rather than an expression statement.
fn parse_declaration_statement(parser: &mut Parser<'_>) -> Statement {
    let decl = parse_block_declaration(parser);
    Statement::Declaration(Box::new(decl))
}

// ===========================================================================
// Declaration-vs-Expression Disambiguation
// ===========================================================================

/// Returns `true` if the given `TokenKind` can start a declaration.
///
/// This checks for:
/// - Type specifier keywords (`int`, `void`, `struct`, `union`, `enum`, etc.)
/// - Storage class keywords (`static`, `extern`, `auto`, `register`, `typedef`)
/// - Type qualifier keywords (`const`, `volatile`, `restrict`, `_Atomic`)
/// - Function specifiers (`inline`, `_Noreturn`)
/// - C11 `_Static_assert`
/// - GCC `__attribute__` (can precede a declaration)
/// - GCC `__auto_type`
/// - GCC `__int128`
///
/// Note: Typedef names (identifiers) are NOT checked here — that check
/// happens separately via `Parser::is_typedef_name()` in the caller.
fn is_declaration_start(parser: &Parser<'_>, kind: TokenKind) -> bool {
    // Type specifier keywords.
    if kind.is_type_specifier() {
        return true;
    }

    // Storage class keywords.
    if kind.is_storage_class() {
        return true;
    }

    // Type qualifiers can start a declaration (e.g., `const int x;`).
    match kind {
        TokenKind::Const
        | TokenKind::Volatile
        | TokenKind::Restrict
        | TokenKind::Atomic => return true,
        _ => {}
    }

    // Function specifiers.
    match kind {
        TokenKind::Inline | TokenKind::Noreturn => return true,
        _ => {}
    }

    // C11 _Static_assert.
    if kind == TokenKind::StaticAssert {
        return true;
    }

    // GCC __attribute__ can appear before a declaration.
    if kind == TokenKind::GccAttribute {
        return true;
    }

    // GCC typedef keyword.
    if kind == TokenKind::Typedef {
        return true;
    }

    // Also try the types module's comprehensive check.
    // This handles typedef names as well as all type specifier keywords.
    if types::is_type_specifier_start(parser) {
        return true;
    }

    false
}

// ===========================================================================
// Statement-Level Error Recovery
// ===========================================================================

/// Synchronises the parser to the next statement boundary.
///
/// Called when a parse error occurs within a statement. Skips tokens until
/// a synchronisation point is reached:
///
/// - `;` — end of statement (consumed).
/// - `}` — end of block (NOT consumed; left for the compound statement parser).
/// - Statement-starting keywords: `if`, `for`, `while`, `do`, `switch`,
///   `return`, `break`, `continue`, `goto`, `case`, `default`.
///
/// This enables the parser to resume after an error and continue parsing
/// subsequent statements, achieving ≥80% error recovery.
fn synchronize_statement(parser: &mut Parser<'_>) {
    parser.synchronize();
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEmitter;
    use crate::common::intern::{InternId, Interner};
    use crate::common::source_map::SourceSpan;
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
        Token::new(
            TokenKind::Identifier,
            dummy_span(),
            TokenValue::Identifier(id),
        )
    }

    /// Creates an integer literal token.
    fn make_int_token(value: u128) -> Token {
        use crate::frontend::lexer::token::{IntSuffix, NumericBase};
        Token::new(
            TokenKind::IntegerLiteral,
            dummy_span(),
            TokenValue::Integer {
                value,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
            },
        )
    }

    /// Creates a parser with the given tokens. The tokens slice must already
    /// include a trailing `Eof` token.
    fn make_parser_from_tokens<'a>(
        tokens: &'a [Token],
        interner: &'a Interner,
        diagnostics: &'a mut DiagnosticEmitter,
    ) -> Parser<'a> {
        Parser::new(tokens, interner, diagnostics)
    }

    /// Builds a token vector from a list of TokenKinds, appending Eof.
    fn tokens_from_kinds(kinds: &[TokenKind]) -> Vec<Token> {
        let mut tokens: Vec<Token> = kinds.iter().map(|&k| make_token(k)).collect();
        tokens.push(make_token(TokenKind::Eof));
        tokens
    }

    // -----------------------------------------------------------------------
    // Null Statement
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_null_statement() {
        // Input: `;`
        let tokens = tokens_from_kinds(&[TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Null { .. } => {} // ok
            _ => panic!("expected Null statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Break / Continue / Return
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_break_statement() {
        // Input: `break ;`
        let tokens = tokens_from_kinds(&[TokenKind::Break, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Break { .. } => {}
            _ => panic!("expected Break statement, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_continue_statement() {
        // Input: `continue ;`
        let tokens = tokens_from_kinds(&[TokenKind::Continue, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Continue { .. } => {}
            _ => panic!("expected Continue statement, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_return_void() {
        // Input: `return ;`
        let tokens = tokens_from_kinds(&[TokenKind::Return, TokenKind::Semicolon]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Return { value: None, .. } => {}
            _ => panic!("expected Return without value, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_return_with_value() {
        // Input: `return 0 ;`
        let mut tokens = vec![make_token(TokenKind::Return), make_int_token(0)];
        tokens.push(make_token(TokenKind::Semicolon));
        tokens.push(make_token(TokenKind::Eof));

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Return {
                value: Some(_), ..
            } => {}
            _ => panic!("expected Return with value, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Goto Statements
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_goto_statement() {
        // Input: `goto label ;`
        let mut interner = Interner::new();
        let label_id = interner.intern("label");

        let mut tokens = vec![
            make_token(TokenKind::Goto),
            make_ident_token(label_id),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Goto { label, .. } => {
                assert_eq!(label, label_id);
            }
            _ => panic!("expected Goto statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Labeled Statement
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_labeled_statement() {
        // Input: `label : ;`
        let mut interner = Interner::new();
        let label_id = interner.intern("label");

        let mut tokens = vec![
            make_ident_token(label_id),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Labeled { label, .. } => {
                assert_eq!(label, label_id);
            }
            _ => panic!("expected Labeled statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Compound Statement
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_empty_compound_statement() {
        // Input: `{ }`
        let tokens = tokens_from_kinds(&[TokenKind::LeftBrace, TokenKind::RightBrace]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_compound_statement(&mut parser);
        match stmt {
            Statement::Compound { items, .. } => {
                assert!(items.is_empty(), "expected empty compound, got {} items", items.len());
            }
            _ => panic!("expected Compound statement, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_nested_empty_blocks() {
        // Input: `{ { { } } }`
        let tokens = tokens_from_kinds(&[
            TokenKind::LeftBrace,
            TokenKind::LeftBrace,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::RightBrace,
            TokenKind::RightBrace,
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_compound_statement(&mut parser);
        match stmt {
            Statement::Compound { items, .. } => {
                // The outer block has one item: the middle block.
                assert_eq!(items.len(), 1, "outer block should have 1 item");
            }
            _ => panic!("expected Compound statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // If/Else Statement
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_if_statement_simple() {
        // Input: `if ( x ) return 0 ;`
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let mut tokens = vec![
            make_token(TokenKind::If),
            make_token(TokenKind::LeftParen),
            make_ident_token(x_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Return),
            make_int_token(0),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::If {
                else_branch: None, ..
            } => {}
            _ => panic!("expected If without else, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_if_else_statement() {
        // Input: `if ( x ) ; else ;`
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let mut tokens = vec![
            make_token(TokenKind::If),
            make_token(TokenKind::LeftParen),
            make_ident_token(x_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon), // then branch: null stmt
            make_token(TokenKind::Else),
            make_token(TokenKind::Semicolon), // else branch: null stmt
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::If {
                else_branch: Some(_),
                ..
            } => {}
            _ => panic!("expected If with else, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // While Loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_while_statement() {
        // Input: `while ( x ) ;`
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let mut tokens = vec![
            make_token(TokenKind::While),
            make_token(TokenKind::LeftParen),
            make_ident_token(x_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::While { .. } => {}
            _ => panic!("expected While statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Do-While Loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_do_while_statement() {
        // Input: `do ; while ( x ) ;`
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let mut tokens = vec![
            make_token(TokenKind::Do),
            make_token(TokenKind::Semicolon), // body: null stmt
            make_token(TokenKind::While),
            make_token(TokenKind::LeftParen),
            make_ident_token(x_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::DoWhile { .. } => {}
            _ => panic!("expected DoWhile statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Switch / Case / Default
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_switch_statement() {
        // Input: `switch ( x ) { }`
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let mut tokens = vec![
            make_token(TokenKind::Switch),
            make_token(TokenKind::LeftParen),
            make_ident_token(x_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::LeftBrace),
            make_token(TokenKind::RightBrace),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Switch { .. } => {}
            _ => panic!("expected Switch statement, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_case_statement() {
        // Input: `case 1 : ;`
        let mut tokens = vec![
            make_token(TokenKind::Case),
            make_int_token(1),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Case {
                range_end: None, ..
            } => {}
            _ => panic!("expected Case without range, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_case_range_gcc() {
        // Input: `case 1 ... 5 : ;`
        let mut tokens = vec![
            make_token(TokenKind::Case),
            make_int_token(1),
            make_token(TokenKind::Ellipsis),
            make_int_token(5),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Case {
                range_end: Some(_), ..
            } => {}
            _ => panic!("expected Case with GCC range, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_default_statement() {
        // Input: `default : ;`
        let mut tokens = vec![
            make_token(TokenKind::Default),
            make_token(TokenKind::Colon),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Default { .. } => {}
            _ => panic!("expected Default statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // For Loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_for_infinite_loop() {
        // Input: `for ( ; ; ) ;`
        let tokens = tokens_from_kinds(&[
            TokenKind::For,
            TokenKind::LeftParen,
            TokenKind::Semicolon, // empty init
            TokenKind::Semicolon, // empty condition
            // empty increment
            TokenKind::RightParen,
            TokenKind::Semicolon, // body: null stmt
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::For {
                init: None,
                condition: None,
                increment: None,
                ..
            } => {}
            _ => panic!("expected For(;;), got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Expression Statement
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_expression_statement() {
        // Input: `x ;` where x is an identifier (not a typedef)
        let mut interner = Interner::new();
        let x_id = interner.intern("x");

        let mut tokens = vec![
            make_ident_token(x_id),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::Expression { .. } => {}
            _ => panic!("expected Expression statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Error Recovery
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_recovery_missing_semicolon() {
        // Input: `return 0 break ;`
        // The missing `;` after `return 0` should trigger error recovery,
        // and the parser should recover and parse `break ;` next.
        let mut tokens = vec![
            make_token(TokenKind::Return),
            make_int_token(0),
            // Missing semicolon here!
            make_token(TokenKind::Break),
            make_token(TokenKind::Semicolon),
        ];
        tokens.push(make_token(TokenKind::Eof));

        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        // First statement should still produce something (with error).
        let _stmt1 = parse_statement(&mut parser);
        // The parser should have reported an error.
        assert!(diag.has_errors(), "expected error for missing semicolon");
    }

    // -----------------------------------------------------------------------
    // Compound with Statements
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_compound_with_null_statements() {
        // Input: `{ ; ; ; }`
        let tokens = tokens_from_kinds(&[
            TokenKind::LeftBrace,
            TokenKind::Semicolon,
            TokenKind::Semicolon,
            TokenKind::Semicolon,
            TokenKind::RightBrace,
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_compound_statement(&mut parser);
        match stmt {
            Statement::Compound { items, .. } => {
                assert_eq!(items.len(), 3, "expected 3 null statements");
                for item in &items {
                    match item {
                        BlockItem::Statement(Statement::Null { .. }) => {}
                        _ => panic!("expected null statement in block, got {:?}", item),
                    }
                }
            }
            _ => panic!("expected Compound statement, got {:?}", stmt),
        }
    }

    // -----------------------------------------------------------------------
    // Declaration-vs-Expression Disambiguation
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_declaration_start_for_type_keywords() {
        let tokens = tokens_from_kinds(&[TokenKind::Int]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        assert!(is_declaration_start(&parser, TokenKind::Int));
        assert!(is_declaration_start(&parser, TokenKind::Void));
        assert!(is_declaration_start(&parser, TokenKind::Char));
        assert!(is_declaration_start(&parser, TokenKind::Struct));
        assert!(is_declaration_start(&parser, TokenKind::Enum));
        assert!(is_declaration_start(&parser, TokenKind::Union));
    }

    #[test]
    fn test_is_declaration_start_for_storage_classes() {
        let tokens = tokens_from_kinds(&[TokenKind::Static]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        assert!(is_declaration_start(&parser, TokenKind::Static));
        assert!(is_declaration_start(&parser, TokenKind::Extern));
        assert!(is_declaration_start(&parser, TokenKind::Typedef));
    }

    #[test]
    fn test_is_not_declaration_start_for_expression_tokens() {
        let tokens = tokens_from_kinds(&[TokenKind::Plus]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        assert!(!is_declaration_start(&parser, TokenKind::Plus));
        assert!(!is_declaration_start(&parser, TokenKind::Star));
        assert!(!is_declaration_start(&parser, TokenKind::Semicolon));
    }

    // -----------------------------------------------------------------------
    // Dangling Else — else binds to innermost if
    // -----------------------------------------------------------------------

    #[test]
    fn test_dangling_else_binds_to_inner_if() {
        // Input: `if ( a ) if ( b ) ; else ;`
        // The `else` should bind to the inner `if (b)`.
        let mut interner = Interner::new();
        let a_id = interner.intern("a");
        let b_id = interner.intern("b");

        let mut tokens = vec![
            make_token(TokenKind::If),
            make_token(TokenKind::LeftParen),
            make_ident_token(a_id),
            make_token(TokenKind::RightParen),
            // inner if
            make_token(TokenKind::If),
            make_token(TokenKind::LeftParen),
            make_ident_token(b_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon), // then of inner if
            make_token(TokenKind::Else),
            make_token(TokenKind::Semicolon), // else of inner if
        ];
        tokens.push(make_token(TokenKind::Eof));

        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser_from_tokens(&tokens, &interner, &mut diag);

        let stmt = parse_statement(&mut parser);
        match stmt {
            Statement::If {
                else_branch: None,
                then_branch,
                ..
            } => {
                // Outer if has no else, inner if has the else.
                match *then_branch {
                    Statement::If {
                        else_branch: Some(_),
                        ..
                    } => {} // correct: else bound to inner if
                    _ => panic!("expected inner If with else"),
                }
            }
            _ => panic!("expected outer If without else, got {:?}", stmt),
        }
    }
}
