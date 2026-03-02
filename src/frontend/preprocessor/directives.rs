//! Preprocessor directive dispatcher for the `bcc` C11 compiler.
//!
//! This module processes all C11 preprocessor directives. It receives lines
//! beginning with `#`, classifies the directive type, and routes to the
//! appropriate handler — either handling the directive inline or delegating
//! to sibling modules (`macros.rs`, `conditional.rs`, `include.rs`,
//! `expression.rs`).
//!
//! # Supported Directives
//!
//! All 14 C11 preprocessor directive types are handled:
//! `#include`, `#define`, `#undef`, `#if`, `#ifdef`, `#ifndef`, `#elif`,
//! `#else`, `#endif`, `#pragma`, `#error`, `#warning`, `#line`, and the
//! null directive (`#` alone on a line).
//!
//! # Conditional Compilation
//!
//! When the preprocessor is inside an inactive conditional block (e.g.,
//! `#if 0`), only conditional directives (`#if`, `#ifdef`, `#ifndef`,
//! `#elif`, `#else`, `#endif`) are processed for nesting tracking. All
//! other directives are silently skipped.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations are pure
//! text/token manipulation using safe Rust idioms.

use std::collections::HashSet;
use std::path::PathBuf;

use super::conditional::ConditionalStack;
use super::expression::evaluate_to_bool;
use super::include::{IncludeError, IncludeKind, IncludeResolver};
use super::macros::{MacroDefinition, MacroTable, parse_macro_definition};
use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::source_map::{SourceLocation, SourceMap};

// ---------------------------------------------------------------------------
// DirectiveKind — classifies all recognized C11 preprocessor directives
// ---------------------------------------------------------------------------

/// Enumerates all recognized C11 preprocessor directive types.
///
/// The C11 standard defines most of these directives in §6.10. The `Warning`
/// variant is a widely-supported GCC extension. The `Null` variant represents
/// the C11 null directive (a lone `#` on a line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectiveKind {
    /// `#include` — file inclusion (C11 §6.10.2).
    Include,
    /// `#define` — macro definition (C11 §6.10.3).
    Define,
    /// `#undef` — macro undefinition (C11 §6.10.3.5).
    Undef,
    /// `#if` — conditional inclusion with expression (C11 §6.10.1).
    If,
    /// `#ifdef` — conditional inclusion testing macro defined (C11 §6.10.1).
    Ifdef,
    /// `#ifndef` — conditional inclusion testing macro not defined (C11 §6.10.1).
    Ifndef,
    /// `#elif` — alternative conditional branch with expression (C11 §6.10.1).
    Elif,
    /// `#else` — alternative conditional branch (C11 §6.10.1).
    Else,
    /// `#endif` — end of conditional inclusion group (C11 §6.10.1).
    Endif,
    /// `#pragma` — implementation-defined behavior (C11 §6.10.6).
    Pragma,
    /// `#error` — compile-time error message (C11 §6.10.5).
    Error,
    /// `#warning` — compile-time warning message (GCC extension).
    Warning,
    /// `#line` — line number and filename override (C11 §6.10.4).
    Line,
    /// `#` alone (null directive) — valid in C11, produces no output (C11 §6.10.7).
    Null,
}

impl DirectiveKind {
    /// Returns `true` if this directive is a conditional compilation directive.
    ///
    /// Conditional directives must always be processed even inside inactive
    /// blocks to maintain correct nesting depth tracking.
    #[inline]
    pub fn is_conditional(&self) -> bool {
        matches!(
            self,
            DirectiveKind::If
                | DirectiveKind::Ifdef
                | DirectiveKind::Ifndef
                | DirectiveKind::Elif
                | DirectiveKind::Else
                | DirectiveKind::Endif
        )
    }
}

// ---------------------------------------------------------------------------
// DirectiveAction — result of processing a directive
// ---------------------------------------------------------------------------

/// Describes the action the preprocessor main loop should take after a
/// directive has been processed.
///
/// The directive handlers in this module return a `DirectiveAction` to
/// communicate the result back to the preprocessor's main loop without
/// directly performing I/O or recursive processing. This keeps the
/// directive handlers pure and testable.
#[derive(Debug)]
pub enum DirectiveAction {
    /// No output or further action needed (e.g., `#define`, `#undef`,
    /// conditional directives, `#pragma`, null directive).
    None,

    /// Include a file: the preprocessor main loop should read the file at
    /// the given path, push it onto the include stack, and recursively
    /// process its contents.
    IncludeFile(PathBuf),

    /// Emit preprocessed text to the output stream. Currently unused by
    /// directive handlers but reserved for future directive extensions
    /// that produce output (e.g., `#pragma message`).
    EmitText(String),

    /// An error occurred during directive processing. The diagnostic has
    /// already been emitted via [`DiagnosticEmitter`]; the preprocessor
    /// main loop should note the failure and may choose to continue
    /// processing (for error recovery) or stop.
    Error,
}

// ---------------------------------------------------------------------------
// PreprocessorContext — shared state for directive processing
// ---------------------------------------------------------------------------

/// Aggregates all mutable state needed by the preprocessor directive handlers.
///
/// Each field is public so that the preprocessor main loop and directive
/// handlers can access individual components as needed. The struct uses
/// public fields rather than accessor methods to enable Rust's borrow
/// checker to perform field-level borrow splitting (allowing simultaneous
/// mutable access to `diagnostics` and immutable access to `macro_table`).
pub struct PreprocessorContext {
    /// Central macro storage and expansion engine for `#define`/`#undef`.
    pub macro_table: MacroTable,

    /// Conditional compilation state machine for `#if`/`#else`/`#endif`.
    pub conditional_stack: ConditionalStack,

    /// Include path resolution engine for `#include` directives.
    pub include_resolver: IncludeResolver,

    /// Source file registry and position tracking for `#line` overrides.
    pub source_map: SourceMap,

    /// GCC-compatible diagnostic emitter for errors and warnings.
    pub diagnostics: DiagnosticEmitter,
}

impl PreprocessorContext {
    /// Creates a new `PreprocessorContext` with the given components.
    pub fn new(
        macro_table: MacroTable,
        conditional_stack: ConditionalStack,
        include_resolver: IncludeResolver,
        source_map: SourceMap,
        diagnostics: DiagnosticEmitter,
    ) -> Self {
        PreprocessorContext {
            macro_table,
            conditional_stack,
            include_resolver,
            source_map,
            diagnostics,
        }
    }
}

// ---------------------------------------------------------------------------
// parse_directive — classify a source line as a preprocessor directive
// ---------------------------------------------------------------------------

/// Parses a source line to determine if it is a preprocessor directive and,
/// if so, classifies it and extracts the remaining text after the keyword.
///
/// The function handles:
/// - Leading whitespace before `#` (valid in C11)
/// - Whitespace between `#` and the directive keyword
/// - Null directive (`#` followed by nothing or only whitespace)
/// - Unknown directive keywords (returns `None`)
///
/// # Arguments
///
/// * `line` — A single line of source text (without the trailing newline).
///
/// # Returns
///
/// `Some((kind, rest))` where `kind` is the directive classification and
/// `rest` is the trimmed text after the directive keyword. Returns `None`
/// if the line is not a preprocessor directive.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(
///     parse_directive("#include <stdio.h>"),
///     Some((DirectiveKind::Include, "<stdio.h>"))
/// );
/// assert_eq!(
///     parse_directive("  #  define FOO 1"),
///     Some((DirectiveKind::Define, "FOO 1"))
/// );
/// assert_eq!(
///     parse_directive("#"),
///     Some((DirectiveKind::Null, ""))
/// );
/// assert_eq!(parse_directive("int x = 0;"), None);
/// ```
pub fn parse_directive(line: &str) -> Option<(DirectiveKind, &str)> {
    let trimmed = line.trim_start();

    // A directive line must start with '#'
    if !trimmed.starts_with('#') {
        return None;
    }

    // Skip the '#' and any whitespace after it
    let after_hash = trimmed[1..].trim_start();

    // If nothing follows the '#', this is a null directive
    if after_hash.is_empty() {
        return Some((DirectiveKind::Null, ""));
    }

    // Extract the directive keyword (sequence of alphabetic characters)
    let keyword_end = after_hash
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(after_hash.len());

    let keyword = &after_hash[..keyword_end];
    let rest = after_hash[keyword_end..].trim_start();

    // Map the keyword to a DirectiveKind
    let kind = match keyword {
        "include" => DirectiveKind::Include,
        "define" => DirectiveKind::Define,
        "undef" => DirectiveKind::Undef,
        "if" => DirectiveKind::If,
        "ifdef" => DirectiveKind::Ifdef,
        "ifndef" => DirectiveKind::Ifndef,
        "elif" => DirectiveKind::Elif,
        "else" => DirectiveKind::Else,
        "endif" => DirectiveKind::Endif,
        "pragma" => DirectiveKind::Pragma,
        "error" => DirectiveKind::Error,
        "warning" => DirectiveKind::Warning,
        "line" => DirectiveKind::Line,
        "" => {
            // After stripping whitespace, if we have no keyword but
            // non-empty text, it could be a stray `# 42` which is
            // a valid line directive form in some compilers, or a
            // null directive followed by a comment. Treat as null.
            return Some((DirectiveKind::Null, after_hash));
        }
        _ => {
            // Unknown directive keyword — the caller should report an error
            return None;
        }
    };

    Some((kind, rest))
}

// ---------------------------------------------------------------------------
// process_directive — main dispatch function
// ---------------------------------------------------------------------------

/// Processes a classified preprocessor directive by routing it to the
/// appropriate handler.
///
/// This is the central dispatch function called by the preprocessor main
/// loop after [`parse_directive`] has classified the line. It handles
/// conditional compilation semantics: when inside an inactive conditional
/// block (`#if 0`), only conditional directives are processed for nesting
/// tracking; all other directives are silently skipped.
///
/// # Arguments
///
/// * `kind` — The classified directive type from [`parse_directive`].
/// * `rest` — The remaining text after the directive keyword.
/// * `ctx` — Mutable reference to the preprocessor context providing access
///   to the macro table, conditional stack, include resolver, source map,
///   and diagnostic emitter.
/// * `location` — The source position of the directive for diagnostic
///   reporting.
///
/// # Returns
///
/// `Ok(DirectiveAction)` describing what the preprocessor main loop should
/// do next, or `Err(())` if a fatal error occurred (diagnostic already
/// emitted).
pub fn process_directive(
    kind: DirectiveKind,
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let active = ctx.conditional_stack.is_active();

    match kind {
        // -----------------------------------------------------------------
        // Conditional directives — always processed for nesting tracking
        // -----------------------------------------------------------------
        DirectiveKind::If => {
            if active {
                let condition = evaluate_if_expression(rest, ctx, location)?;
                ctx.conditional_stack.push_if(condition, location);
            } else {
                // Parent inactive — track nesting without evaluating
                ctx.conditional_stack.push_if(false, location);
            }
            Ok(DirectiveAction::None)
        }

        DirectiveKind::Ifdef => {
            let name = extract_single_identifier(rest);
            if active {
                if name.is_empty() {
                    ctx.diagnostics
                        .error(location, "no macro name given in #ifdef directive");
                    return Err(());
                }
                let defined = ctx.macro_table.is_defined(name);
                ctx.conditional_stack.push_if(defined, location);
            } else {
                ctx.conditional_stack.push_if(false, location);
            }
            Ok(DirectiveAction::None)
        }

        DirectiveKind::Ifndef => {
            let name = extract_single_identifier(rest);
            if active {
                if name.is_empty() {
                    ctx.diagnostics
                        .error(location, "no macro name given in #ifndef directive");
                    return Err(());
                }
                let not_defined = !ctx.macro_table.is_defined(name);
                ctx.conditional_stack.push_if(not_defined, location);
            } else {
                ctx.conditional_stack.push_if(false, location);
            }
            Ok(DirectiveAction::None)
        }

        DirectiveKind::Elif => {
            // For #elif, we must evaluate even when the current branch is
            // inactive (Inactive state) because this branch might activate.
            // We only skip evaluation when in ParentInactive, but since we
            // cannot directly query that state, we use a pragmatic approach:
            // always try to evaluate and fall back to false on failure.
            let condition =
                evaluate_if_expression(rest, ctx, location).unwrap_or(false);
            ctx.conditional_stack
                .process_elif(condition, location, &mut ctx.diagnostics)?;
            Ok(DirectiveAction::None)
        }

        DirectiveKind::Else => {
            ctx.conditional_stack
                .process_else(location, &mut ctx.diagnostics)?;
            Ok(DirectiveAction::None)
        }

        DirectiveKind::Endif => {
            ctx.conditional_stack
                .pop_endif(location, &mut ctx.diagnostics)?;
            Ok(DirectiveAction::None)
        }

        // -----------------------------------------------------------------
        // Non-conditional directives — skip when in inactive blocks
        // -----------------------------------------------------------------
        _ if !active => Ok(DirectiveAction::None),

        DirectiveKind::Include => handle_include(rest, ctx, location),
        DirectiveKind::Define => handle_define(rest, ctx, location),
        DirectiveKind::Undef => handle_undef(rest, ctx, location),
        DirectiveKind::Pragma => handle_pragma(rest, ctx, location),
        DirectiveKind::Error => handle_error(rest, ctx, location),
        DirectiveKind::Warning => handle_warning(rest, ctx, location),
        DirectiveKind::Line => handle_line(rest, ctx, location),
        DirectiveKind::Null => Ok(DirectiveAction::None),
    }
}

// ---------------------------------------------------------------------------
// evaluate_if_expression — helper for #if and #elif condition evaluation
// ---------------------------------------------------------------------------

/// Evaluates a preprocessor conditional expression for `#if` and `#elif`.
///
/// Constructs an `is_macro_defined` closure from the context's macro table
/// and delegates to the expression evaluator. Uses field-level borrow
/// splitting to allow simultaneous immutable access to `macro_table` and
/// mutable access to `diagnostics`.
fn evaluate_if_expression(
    expr_text: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<bool, ()> {
    let trimmed = expr_text.trim();
    if trimmed.is_empty() {
        ctx.diagnostics.error(location, "#if with no expression");
        return Err(());
    }

    // Phase 1: Pre-evaluate `defined(X)` / `defined X` BEFORE macro expansion.
    // Per C11 §6.10.1p4, defined-expressions are evaluated before macro
    // replacement. We replace them with literal "1" or "0" so that macro
    // expansion doesn't corrupt the identifier arguments.
    let pre_defined = replace_defined_with_literals(trimmed, &ctx.macro_table);

    // Phase 2: Expand remaining macros in the expression text.
    let mut expansion_guard = std::collections::HashSet::new();
    let expanded = ctx.macro_table.expand_line(&pre_defined, &mut expansion_guard);
    let expanded_trimmed = expanded.trim();
    if expanded_trimmed.is_empty() {
        ctx.diagnostics.error(location, "#if with no expression");
        return Err(());
    }

    // Field-level borrow splitting: borrow macro_table immutably and
    // diagnostics mutably simultaneously through the struct fields.
    let macro_table = &ctx.macro_table;
    let diagnostics = &mut ctx.diagnostics;
    let is_defined = |name: &str| -> bool { macro_table.is_defined(name) };
    evaluate_to_bool(expanded_trimmed, &is_defined, diagnostics, location)
}

// ---------------------------------------------------------------------------
// extract_single_identifier — helper for #ifdef, #ifndef, #undef
// ---------------------------------------------------------------------------

/// Replaces all `defined(X)` and `defined X` occurrences in the expression
/// text with literal `1` or `0` based on whether macro X is currently defined.
///
/// This MUST be done before macro expansion, per C11 §6.10.1p4, to prevent
/// the identifier argument of `defined` from being macro-expanded.
fn replace_defined_with_literals(text: &str, macro_table: &MacroTable) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        // Skip string/char literals
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            result.push(quote as char);
            i += 1;
            while i < len && bytes[i] != quote {
                if bytes[i] == b'\\' && i + 1 < len {
                    result.push(bytes[i] as char);
                    i += 1;
                }
                result.push(bytes[i] as char);
                i += 1;
            }
            if i < len {
                result.push(bytes[i] as char);
                i += 1;
            }
            continue;
        }
        // Check for 'defined' keyword
        if i + 7 <= len && &text[i..i + 7] == "defined" {
            // Make sure it's a word boundary (not part of a longer identifier)
            let before_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let after_ok = i + 7 >= len || !(bytes[i + 7].is_ascii_alphanumeric() || bytes[i + 7] == b'_');
            if before_ok && after_ok {
                let mut j = i + 7;
                // Skip whitespace
                while j < len && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < len && bytes[j] == b'(' {
                    // defined(IDENT) form
                    j += 1;
                    while j < len && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    let ident_start = j;
                    while j < len && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                        j += 1;
                    }
                    let ident = &text[ident_start..j];
                    while j < len && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    if j < len && bytes[j] == b')' {
                        j += 1;
                        let val = if !ident.is_empty() && macro_table.is_defined(ident) { "1" } else { "0" };
                        result.push_str(val);
                        i = j;
                        continue;
                    }
                    // Malformed — pass through unchanged
                } else if j < len && (bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
                    // defined IDENT form
                    let ident_start = j;
                    while j < len && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                        j += 1;
                    }
                    let ident = &text[ident_start..j];
                    let val = if macro_table.is_defined(ident) { "1" } else { "0" };
                    result.push_str(val);
                    i = j;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Extracts a single C identifier from the beginning of the text.
///
/// Returns the identifier substring, or an empty string if the text does
/// not start with a valid C identifier character.
fn extract_single_identifier(text: &str) -> &str {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() {
        return "";
    }

    // A C identifier starts with a letter or underscore
    if !bytes[0].is_ascii_alphabetic() && bytes[0] != b'_' {
        return "";
    }

    let end = bytes
        .iter()
        .position(|&b| !b.is_ascii_alphanumeric() && b != b'_')
        .unwrap_or(bytes.len());

    &trimmed[..end]
}

// ---------------------------------------------------------------------------
// handle_include — #include directive
// ---------------------------------------------------------------------------

/// Processes a `#include` directive.
///
/// Supports three forms:
/// - `#include <header.h>` — angle-bracket (system) include
/// - `#include "header.h"` — quoted (local) include
/// - `#include MACRO` — macro-expanded include (expands macros, then re-parses)
fn handle_include(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let trimmed = rest.trim();

    // Determine include kind and extract header name
    let (header_name, kind) = if trimmed.starts_with('<') {
        // Angle-bracket include: #include <header.h>
        match trimmed.find('>') {
            Some(end) => (&trimmed[1..end], IncludeKind::Angle),
            None => {
                ctx.diagnostics
                    .error(location, "missing terminating > character");
                return Ok(DirectiveAction::Error);
            }
        }
    } else if trimmed.starts_with('"') {
        // Quoted include: #include "header.h"
        match trimmed[1..].find('"') {
            Some(end) => (&trimmed[1..1 + end], IncludeKind::Quoted),
            None => {
                ctx.diagnostics
                    .error(location, "missing terminating \" character");
                return Ok(DirectiveAction::Error);
            }
        }
    } else if !trimmed.is_empty() {
        // Macro-expanded include: try expanding macros and re-parsing
        match try_macro_expanded_include(trimmed, ctx, location) {
            Some(result) => return result,
            None => {
                ctx.diagnostics.error(
                    location,
                    "#include expects \"FILENAME\" or <FILENAME>",
                );
                return Ok(DirectiveAction::Error);
            }
        }
    } else {
        ctx.diagnostics
            .error(location, "empty filename in #include directive");
        return Ok(DirectiveAction::Error);
    };

    resolve_and_include(header_name, kind, ctx, location)
}

/// Resolves an include path and returns the appropriate `DirectiveAction`.
fn resolve_and_include(
    header_name: &str,
    kind: IncludeKind,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    // Resolve the current directory for quoted include resolution.
    // Clone to PathBuf to release the borrow on include_resolver before
    // calling resolve(), enabling field-level borrow splitting.
    let current_dir = ctx
        .include_resolver
        .current_dir()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    // Resolve the header path through the include search directories
    match ctx
        .include_resolver
        .resolve(header_name, kind, &current_dir)
    {
        Ok(resolved_path) => Ok(DirectiveAction::IncludeFile(resolved_path)),
        Err(IncludeError::NotFound(name)) => {
            ctx.diagnostics
                .error(location, format!("'{}' file not found", name));
            Ok(DirectiveAction::Error)
        }
        Err(IncludeError::CircularInclude(path)) => {
            ctx.diagnostics.error(
                location,
                format!("circular include detected: '{}'", path.display()),
            );
            Ok(DirectiveAction::Error)
        }
        Err(IncludeError::PragmaOnce(_)) => {
            // File was #pragma once'd — silently skip re-inclusion
            Ok(DirectiveAction::None)
        }
        Err(IncludeError::IoError(err)) => {
            ctx.diagnostics.error(
                location,
                format!("cannot open include file: {}", err),
            );
            Ok(DirectiveAction::Error)
        }
    }
}

/// Attempts to expand macros in the `#include` argument and re-parse it
/// as an angle-bracket or quoted include.
///
/// This handles the C11 §6.10.2p4 case where the `#include` argument is
/// a macro that expands to `<header.h>` or `"header.h"`.
fn try_macro_expanded_include(
    text: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Option<Result<DirectiveAction, ()>> {
    // Expand macros in the include argument
    let mut guard = HashSet::new();
    let expanded = ctx.macro_table.expand_line(text, &mut guard);
    let expanded_trimmed = expanded.trim();

    if expanded_trimmed.starts_with('<') {
        if let Some(end) = expanded_trimmed.find('>') {
            let header = expanded_trimmed[1..end].to_string();
            return Some(resolve_and_include(
                &header,
                IncludeKind::Angle,
                ctx,
                location,
            ));
        }
    } else if expanded_trimmed.starts_with('"') {
        if let Some(end) = expanded_trimmed[1..].find('"') {
            let header = expanded_trimmed[1..1 + end].to_string();
            return Some(resolve_and_include(
                &header,
                IncludeKind::Quoted,
                ctx,
                location,
            ));
        }
    }

    // Expansion did not produce a valid include form
    None
}

// ---------------------------------------------------------------------------
// handle_define — #define directive
// ---------------------------------------------------------------------------

/// Processes a `#define` directive, creating or redefining a macro.
///
/// Handles all macro definition forms:
/// - Object-like: `#define NAME replacement`
/// - Function-like: `#define NAME(params) replacement`
/// - Variadic: `#define NAME(params, ...) replacement`
/// - Empty: `#define NAME` (defines to empty replacement)
///
/// Emits a warning on macro redefinition when the new definition differs
/// from the existing one (C11 §6.10.3p2, GCC behavior).
fn handle_define(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let trimmed = rest.trim_start();
    if trimmed.is_empty() {
        ctx.diagnostics
            .error(location, "no macro name given in #define directive");
        return Err(());
    }

    // Extract the macro name (C identifier)
    let bytes = trimmed.as_bytes();
    if !bytes[0].is_ascii_alphabetic() && bytes[0] != b'_' {
        ctx.diagnostics
            .error(location, "macro names must be identifiers");
        return Err(());
    }

    let name_end = bytes
        .iter()
        .position(|&b| !b.is_ascii_alphanumeric() && b != b'_')
        .unwrap_or(bytes.len());
    let name = &trimmed[..name_end];
    let after_name = &trimmed[name_end..];

    // Parse the macro definition body using the macros module parser.
    // `parse_macro_definition` returns a `MacroDefinition` with `.name`,
    // `.kind`, and `.replacement` populated from the directive text.
    let result: Result<MacroDefinition, String> =
        parse_macro_definition(name, after_name);
    match result {
        Ok(mut def) => {
            def.location = Some(location);

            // Access the parsed definition's name for the redefinition check.
            // `MacroDefinition.name` is the canonical name extracted by the parser.
            let macro_name: &str = &def.name;

            // Check for redefinition with a different body — emit warning
            // per C11 §6.10.3p2. We compare `MacroDefinition.kind` and
            // `MacroDefinition.replacement` via `is_equivalent()`.
            if let Some(old) = ctx.macro_table.get(macro_name) {
                // `is_equivalent` compares `.kind` and `.replacement` fields.
                // When they differ, the macro has been redefined with a
                // different expansion, which warrants a warning.
                let kinds_match = std::mem::discriminant(&old.kind)
                    == std::mem::discriminant(&def.kind);
                let replacements_match = old.replacement == def.replacement;

                if !kinds_match || !replacements_match {
                    // Borrow splitting: old borrows macro_table immutably,
                    // warning borrows diagnostics mutably — disjoint fields.
                    ctx.diagnostics
                        .warning(location, format!("'{}' macro redefined", macro_name));
                    if let Some(old_loc) = old.location {
                        if !old_loc.is_dummy() {
                            ctx.diagnostics.warning(
                                old_loc,
                                format!(
                                    "previous definition of '{}' was here",
                                    old.name
                                ),
                            );
                        }
                    }
                }
            }

            ctx.macro_table.define(def);
            Ok(DirectiveAction::None)
        }
        Err(msg) => {
            ctx.diagnostics.error(location, msg);
            Err(())
        }
    }
}

// ---------------------------------------------------------------------------
// handle_undef — #undef directive
// ---------------------------------------------------------------------------

/// Processes a `#undef` directive, removing a macro definition.
///
/// Per C11 §6.10.3.5, `#undef` of a macro that is not currently defined
/// is silently ignored (not an error).
fn handle_undef(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let name = extract_single_identifier(rest);
    if name.is_empty() {
        ctx.diagnostics
            .error(location, "no macro name given in #undef directive");
        return Err(());
    }
    // Silently succeeds even if the macro was not defined (C11 behavior)
    ctx.macro_table.undefine(name);
    Ok(DirectiveAction::None)
}

// ---------------------------------------------------------------------------
// handle_pragma — #pragma directive
// ---------------------------------------------------------------------------

/// Processes a `#pragma` directive.
///
/// Recognized pragmas:
/// - `#pragma once` — marks the current file to prevent re-inclusion
/// - `#pragma GCC ...` — silently ignored (GCC compatibility)
/// - `#pragma pack(...)` — silently ignored (future enhancement)
/// - Unknown pragmas — silently ignored (C11 §6.10.6: implementation-defined)
fn handle_pragma(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let trimmed = rest.trim();

    if trimmed == "once"
        || trimmed.starts_with("once ")
        || trimmed.starts_with("once\t")
    {
        // #pragma once — prevent re-inclusion of this file.
        // Get the current file path from the source map using the
        // location's file_id, then mark it in the include resolver.
        let current_file = ctx
            .source_map
            .get_file_path(location.file_id)
            .to_path_buf();
        ctx.include_resolver.mark_pragma_once(&current_file);
        return Ok(DirectiveAction::None);
    }

    // All other pragmas (GCC, pack, etc.) are silently ignored.
    // C11 §6.10.6: unrecognized pragmas cause implementation-defined
    // behavior, and ignoring them is a valid implementation choice.
    Ok(DirectiveAction::None)
}

// ---------------------------------------------------------------------------
// handle_error — #error directive
// ---------------------------------------------------------------------------

/// Processes a `#error` directive, emitting a compile-time error.
///
/// The entire text after `#error` is used as the error message. This is
/// a hard error that should prevent successful compilation.
fn handle_error(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let message = rest.trim();
    if message.is_empty() {
        ctx.diagnostics.error(location, "#error");
    } else {
        ctx.diagnostics
            .error(location, format!("#error {}", message));
    }
    Ok(DirectiveAction::Error)
}

// ---------------------------------------------------------------------------
// handle_warning — #warning directive (GCC extension)
// ---------------------------------------------------------------------------

/// Processes a `#warning` directive, emitting a compile-time warning.
///
/// This is a GCC extension that is universally supported by modern C
/// compilers. Unlike `#error`, `#warning` is non-blocking: compilation
/// continues after the warning is emitted.
fn handle_warning(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let message = rest.trim();
    if message.is_empty() {
        ctx.diagnostics.warning(location, "#warning");
    } else {
        ctx.diagnostics
            .warning(location, format!("#warning {}", message));
    }
    Ok(DirectiveAction::None)
}

// ---------------------------------------------------------------------------
// handle_line — #line directive
// ---------------------------------------------------------------------------

/// Processes a `#line` directive, overriding the reported line number and
/// optionally the filename for subsequent diagnostics.
///
/// Supported forms:
/// - `#line N` — set line number to N
/// - `#line N "filename"` — set line number to N and filename
///
/// The line number must be a positive integer (1..=2147483647 per C11).
fn handle_line(
    rest: &str,
    ctx: &mut PreprocessorContext,
    location: SourceLocation,
) -> Result<DirectiveAction, ()> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        ctx.diagnostics.error(
            location,
            "\"line\" after #line is not a positive integer",
        );
        return Err(());
    }

    // Parse the line number
    let (line_str, after_number) = split_at_whitespace(trimmed);
    let new_line: u32 = match line_str.parse() {
        Ok(n) if n > 0 => n,
        _ => {
            ctx.diagnostics.error(
                location,
                format!(
                    "\"{}\" after #line is not a positive integer",
                    line_str
                ),
            );
            return Err(());
        }
    };

    // Parse the optional filename
    let new_file = if !after_number.is_empty() {
        let file_trimmed = after_number.trim();
        if file_trimmed.starts_with('"') {
            match file_trimmed[1..].find('"') {
                Some(end) => Some(PathBuf::from(&file_trimmed[1..1 + end])),
                None => {
                    ctx.diagnostics.error(
                        location,
                        "missing terminating \" in #line directive",
                    );
                    return Err(());
                }
            }
        } else if !file_trimmed.is_empty() {
            ctx.diagnostics.error(
                location,
                "expected filename string in #line directive",
            );
            return Err(());
        } else {
            None
        }
    } else {
        None
    };

    // Apply the line override to the source map
    ctx.source_map.set_line_override(
        location.file_id,
        location.byte_offset,
        new_line,
        new_file,
    );

    Ok(DirectiveAction::None)
}

// ---------------------------------------------------------------------------
// split_at_whitespace — helper for parsing space-separated tokens
// ---------------------------------------------------------------------------

/// Splits a string at the first whitespace boundary, returning the first
/// token and the remainder.
fn split_at_whitespace(s: &str) -> (&str, &str) {
    match s.find(char::is_whitespace) {
        Some(pos) => (&s[..pos], &s[pos..]),
        None => (s, ""),
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::common::source_map::FileId;

    // -----------------------------------------------------------------------
    // Test Helpers
    // -----------------------------------------------------------------------

    /// Creates a test `SourceLocation` at the given line number.
    fn test_loc(line: u32) -> SourceLocation {
        SourceLocation {
            file_id: FileId(0),
            byte_offset: 0,
            line,
            column: 1,
        }
    }

    /// Creates a minimal `PreprocessorContext` for testing.
    fn test_context() -> PreprocessorContext {
        let mut source_map = SourceMap::new();
        // Register a dummy file so FileId(0) is valid
        source_map.add_file(
            PathBuf::from("test.c"),
            String::from("// test file\n"),
        );

        PreprocessorContext::new(
            MacroTable::new(),
            ConditionalStack::new(),
            IncludeResolver::new(vec![], None, vec![]),
            source_map,
            DiagnosticEmitter::new(),
        )
    }

    // =======================================================================
    // parse_directive tests
    // =======================================================================

    #[test]
    fn test_parse_include() {
        let result = parse_directive("#include <stdio.h>");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Include);
        assert_eq!(rest, "<stdio.h>");
    }

    #[test]
    fn test_parse_define() {
        let result = parse_directive("#define FOO 1");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Define);
        assert_eq!(rest, "FOO 1");
    }

    #[test]
    fn test_parse_undef() {
        let result = parse_directive("#undef FOO");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Undef);
        assert_eq!(rest, "FOO");
    }

    #[test]
    fn test_parse_if() {
        let result = parse_directive("#if defined(FOO)");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::If);
        assert_eq!(rest, "defined(FOO)");
    }

    #[test]
    fn test_parse_ifdef() {
        let result = parse_directive("#ifdef FOO");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Ifdef);
        assert_eq!(rest, "FOO");
    }

    #[test]
    fn test_parse_ifndef() {
        let result = parse_directive("#ifndef FOO");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Ifndef);
        assert_eq!(rest, "FOO");
    }

    #[test]
    fn test_parse_elif() {
        let result = parse_directive("#elif 1");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Elif);
        assert_eq!(rest, "1");
    }

    #[test]
    fn test_parse_else() {
        let result = parse_directive("#else");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Else);
        assert!(rest.is_empty());
    }

    #[test]
    fn test_parse_endif() {
        let result = parse_directive("#endif");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Endif);
        assert!(rest.is_empty());
    }

    #[test]
    fn test_parse_pragma() {
        let result = parse_directive("#pragma once");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Pragma);
        assert_eq!(rest, "once");
    }

    #[test]
    fn test_parse_error() {
        let result = parse_directive("#error something went wrong");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Error);
        assert_eq!(rest, "something went wrong");
    }

    #[test]
    fn test_parse_warning() {
        let result = parse_directive("#warning this is deprecated");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Warning);
        assert_eq!(rest, "this is deprecated");
    }

    #[test]
    fn test_parse_line() {
        let result = parse_directive("#line 42");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Line);
        assert_eq!(rest, "42");
    }

    #[test]
    fn test_parse_null_directive() {
        let result = parse_directive("#");
        assert!(result.is_some());
        let (kind, _rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Null);
    }

    #[test]
    fn test_parse_null_directive_with_whitespace() {
        let result = parse_directive("#   ");
        assert!(result.is_some());
        let (kind, _rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Null);
    }

    #[test]
    fn test_parse_not_a_directive() {
        assert!(parse_directive("int x = 0;").is_none());
        assert!(parse_directive("").is_none());
        assert!(parse_directive("  // comment").is_none());
    }

    #[test]
    fn test_parse_directive_with_leading_whitespace() {
        let result = parse_directive("  #define X 1");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Define);
        assert_eq!(rest, "X 1");
    }

    #[test]
    fn test_parse_directive_with_space_after_hash() {
        let result = parse_directive("#  define X 1");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Define);
        assert_eq!(rest, "X 1");
    }

    #[test]
    fn test_parse_unknown_directive() {
        // Unknown directives return None so the caller can report an error
        assert!(parse_directive("#foobar").is_none());
    }

    #[test]
    fn test_directive_kind_is_conditional() {
        assert!(DirectiveKind::If.is_conditional());
        assert!(DirectiveKind::Ifdef.is_conditional());
        assert!(DirectiveKind::Ifndef.is_conditional());
        assert!(DirectiveKind::Elif.is_conditional());
        assert!(DirectiveKind::Else.is_conditional());
        assert!(DirectiveKind::Endif.is_conditional());
        assert!(!DirectiveKind::Include.is_conditional());
        assert!(!DirectiveKind::Define.is_conditional());
        assert!(!DirectiveKind::Undef.is_conditional());
        assert!(!DirectiveKind::Pragma.is_conditional());
        assert!(!DirectiveKind::Error.is_conditional());
        assert!(!DirectiveKind::Warning.is_conditional());
        assert!(!DirectiveKind::Line.is_conditional());
        assert!(!DirectiveKind::Null.is_conditional());
    }

    // =======================================================================
    // process_directive tests — #define
    // =======================================================================

    #[test]
    fn test_define_object_like_macro() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Define, "FOO 42", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.macro_table.is_defined("FOO"));
    }

    #[test]
    fn test_define_function_like_macro() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(
            DirectiveKind::Define,
            "MAX(a, b) ((a) > (b) ? (a) : (b))",
            &mut ctx,
            loc,
        );
        assert!(result.is_ok());
        assert!(ctx.macro_table.is_defined("MAX"));
    }

    #[test]
    fn test_define_empty_macro() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Define, "EMPTY", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.macro_table.is_defined("EMPTY"));
    }

    #[test]
    fn test_define_no_name_error() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Define, "", &mut ctx, loc);
        assert!(result.is_err());
        assert!(ctx.diagnostics.has_errors());
    }

    #[test]
    fn test_define_invalid_name_error() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Define, "123abc", &mut ctx, loc);
        assert!(result.is_err());
        assert!(ctx.diagnostics.has_errors());
    }

    // =======================================================================
    // process_directive tests — #undef
    // =======================================================================

    #[test]
    fn test_undef_removes_macro() {
        let mut ctx = test_context();
        let loc = test_loc(1);

        // First define, then undef
        let _ = process_directive(DirectiveKind::Define, "FOO 1", &mut ctx, loc);
        assert!(ctx.macro_table.is_defined("FOO"));

        let result = process_directive(DirectiveKind::Undef, "FOO", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.macro_table.is_defined("FOO"));
    }

    #[test]
    fn test_undef_nonexistent_silent() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        // #undef of undefined macro is silently ignored
        let result = process_directive(DirectiveKind::Undef, "NONEXISTENT", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.diagnostics.has_errors());
    }

    #[test]
    fn test_undef_no_name_error() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Undef, "", &mut ctx, loc);
        assert!(result.is_err());
    }

    // =======================================================================
    // process_directive tests — #if / #ifdef / #ifndef
    // =======================================================================

    #[test]
    fn test_if_true_activates() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::If, "1", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_if_false_deactivates() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::If, "0", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_ifdef_defined_activates() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::Define, "FOO 1", &mut ctx, loc);
        let result = process_directive(DirectiveKind::Ifdef, "FOO", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_ifdef_undefined_deactivates() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Ifdef, "FOO", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_ifndef_undefined_activates() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Ifndef, "FOO", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_ifndef_defined_deactivates() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::Define, "FOO 1", &mut ctx, loc);
        let result = process_directive(DirectiveKind::Ifndef, "FOO", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.conditional_stack.is_active());
    }

    // =======================================================================
    // process_directive tests — #elif / #else / #endif
    // =======================================================================

    #[test]
    fn test_if_false_elif_true() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::If, "0", &mut ctx, loc);
        assert!(!ctx.conditional_stack.is_active());

        let result = process_directive(DirectiveKind::Elif, "1", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_if_true_else_inactive() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::If, "1", &mut ctx, loc);
        assert!(ctx.conditional_stack.is_active());

        let result = process_directive(DirectiveKind::Else, "", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_if_false_else_active() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::If, "0", &mut ctx, loc);
        assert!(!ctx.conditional_stack.is_active());

        let result = process_directive(DirectiveKind::Else, "", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_endif_pops_stack() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::If, "1", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 1);

        let result = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
        assert!(result.is_ok());
        assert_eq!(ctx.conditional_stack.depth(), 0);
    }

    #[test]
    fn test_endif_without_if_error() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
        assert!(result.is_err());
        assert!(ctx.diagnostics.has_errors());
    }

    // =======================================================================
    // process_directive tests — inactive block skipping
    // =======================================================================

    #[test]
    fn test_define_skipped_in_inactive_block() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let _ = process_directive(DirectiveKind::If, "0", &mut ctx, loc);

        // #define inside #if 0 should be skipped
        let result = process_directive(DirectiveKind::Define, "SKIPPED 1", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.macro_table.is_defined("SKIPPED"));

        // Clean up
        let _ = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
    }

    #[test]
    fn test_nested_if_in_inactive_block() {
        let mut ctx = test_context();
        let loc = test_loc(1);

        // Enter #if 0
        let _ = process_directive(DirectiveKind::If, "0", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 1);

        // Nested #if should still be tracked for nesting
        let _ = process_directive(DirectiveKind::If, "1", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 2);

        // Nested #endif
        let _ = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 1);

        // Outer #endif
        let _ = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 0);
    }

    // =======================================================================
    // process_directive tests — #error and #warning
    // =======================================================================

    #[test]
    fn test_error_emits_diagnostic() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(
            DirectiveKind::Error,
            "something went wrong",
            &mut ctx,
            loc,
        );
        assert!(result.is_ok());
        assert!(ctx.diagnostics.has_errors());
    }

    #[test]
    fn test_warning_emits_diagnostic() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(
            DirectiveKind::Warning,
            "this is deprecated",
            &mut ctx,
            loc,
        );
        assert!(result.is_ok());
        assert!(!ctx.diagnostics.has_errors());
        assert_eq!(ctx.diagnostics.warning_count(), 1);
    }

    #[test]
    fn test_error_empty_message() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Error, "", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(ctx.diagnostics.has_errors());
    }

    // =======================================================================
    // process_directive tests — #line
    // =======================================================================

    #[test]
    fn test_line_number_only() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Line, "42", &mut ctx, loc);
        assert!(result.is_ok());
        // Verify the line override was set
        let override_result = ctx
            .source_map
            .get_line_override(loc.file_id, loc.byte_offset);
        assert!(override_result.is_some());
        let (line_num, file_override) = override_result.unwrap();
        assert_eq!(line_num, 42);
        assert!(file_override.is_none());
    }

    #[test]
    fn test_line_with_filename() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(
            DirectiveKind::Line,
            "100 \"fake.c\"",
            &mut ctx,
            loc,
        );
        assert!(result.is_ok());
        let override_result = ctx
            .source_map
            .get_line_override(loc.file_id, loc.byte_offset);
        assert!(override_result.is_some());
        let (line_num, file_override) = override_result.unwrap();
        assert_eq!(line_num, 100);
        assert_eq!(file_override.unwrap(), Path::new("fake.c"));
    }

    #[test]
    fn test_line_invalid_number() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Line, "abc", &mut ctx, loc);
        assert!(result.is_err());
        assert!(ctx.diagnostics.has_errors());
    }

    #[test]
    fn test_line_zero_number() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        // Line 0 is invalid (must be positive)
        let result = process_directive(DirectiveKind::Line, "0", &mut ctx, loc);
        assert!(result.is_err());
    }

    #[test]
    fn test_line_empty() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Line, "", &mut ctx, loc);
        assert!(result.is_err());
    }

    // =======================================================================
    // process_directive tests — #include parsing
    // =======================================================================

    #[test]
    fn test_include_angle_bracket_parsing() {
        // parse_directive correctly splits the include
        let result = parse_directive("#include <stdio.h>");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Include);
        assert_eq!(rest, "<stdio.h>");
    }

    #[test]
    fn test_include_quoted_parsing() {
        let result = parse_directive("#include \"myheader.h\"");
        assert!(result.is_some());
        let (kind, rest) = result.unwrap();
        assert_eq!(kind, DirectiveKind::Include);
        assert_eq!(rest, "\"myheader.h\"");
    }

    #[test]
    fn test_include_empty_error() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Include, "", &mut ctx, loc);
        assert!(result.is_ok()); // Returns Ok(Error) — diagnostic emitted
        assert!(ctx.diagnostics.has_errors());
    }

    // =======================================================================
    // process_directive tests — #pragma
    // =======================================================================

    #[test]
    fn test_pragma_once() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Pragma, "once", &mut ctx, loc);
        assert!(result.is_ok());
        // Verify pragma once was registered
        let file_path = ctx.source_map.get_file_path(loc.file_id).to_path_buf();
        assert!(ctx.include_resolver.is_pragma_once(&file_path));
    }

    #[test]
    fn test_pragma_unknown_silently_ignored() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(
            DirectiveKind::Pragma,
            "GCC diagnostic push",
            &mut ctx,
            loc,
        );
        assert!(result.is_ok());
        assert!(!ctx.diagnostics.has_errors());
    }

    #[test]
    fn test_pragma_pack_silently_ignored() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result =
            process_directive(DirectiveKind::Pragma, "pack(push, 1)", &mut ctx, loc);
        assert!(result.is_ok());
        assert!(!ctx.diagnostics.has_errors());
    }

    // =======================================================================
    // process_directive tests — null directive
    // =======================================================================

    #[test]
    fn test_null_directive() {
        let mut ctx = test_context();
        let loc = test_loc(1);
        let result = process_directive(DirectiveKind::Null, "", &mut ctx, loc);
        assert!(result.is_ok());
        // Null directive produces no output and no errors
        assert!(!ctx.diagnostics.has_errors());
    }

    // =======================================================================
    // Helper function tests
    // =======================================================================

    #[test]
    fn test_extract_single_identifier() {
        assert_eq!(extract_single_identifier("FOO"), "FOO");
        assert_eq!(extract_single_identifier("  FOO  "), "FOO");
        assert_eq!(extract_single_identifier("FOO(x)"), "FOO");
        assert_eq!(extract_single_identifier("_BAR"), "_BAR");
        assert_eq!(extract_single_identifier("a123"), "a123");
        assert_eq!(extract_single_identifier(""), "");
        assert_eq!(extract_single_identifier("123"), "");
        assert_eq!(extract_single_identifier("  "), "");
    }

    #[test]
    fn test_split_at_whitespace() {
        assert_eq!(split_at_whitespace("42 \"file.c\""), ("42", " \"file.c\""));
        assert_eq!(split_at_whitespace("42"), ("42", ""));
        assert_eq!(split_at_whitespace(""), ("", ""));
        assert_eq!(split_at_whitespace("abc def"), ("abc", " def"));
    }

    // =======================================================================
    // Integration-like tests — multi-directive sequences
    // =======================================================================

    #[test]
    fn test_if_elif_else_endif_sequence() {
        let mut ctx = test_context();
        let loc = test_loc(1);

        // #if 0
        let _ = process_directive(DirectiveKind::If, "0", &mut ctx, loc);
        assert!(!ctx.conditional_stack.is_active());

        // #elif 0
        let _ = process_directive(DirectiveKind::Elif, "0", &mut ctx, loc);
        assert!(!ctx.conditional_stack.is_active());

        // #else — should activate since no branch was taken
        let _ = process_directive(DirectiveKind::Else, "", &mut ctx, loc);
        assert!(ctx.conditional_stack.is_active());

        // #endif
        let _ = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 0);
        assert!(ctx.conditional_stack.is_active());
    }

    #[test]
    fn test_ifndef_guard_pattern() {
        let mut ctx = test_context();
        let loc = test_loc(1);

        // Common header guard pattern:
        // #ifndef HEADER_H
        // #define HEADER_H
        // ... content ...
        // #endif

        let _ = process_directive(DirectiveKind::Ifndef, "HEADER_H", &mut ctx, loc);
        assert!(ctx.conditional_stack.is_active());

        let _ = process_directive(DirectiveKind::Define, "HEADER_H", &mut ctx, loc);
        assert!(ctx.macro_table.is_defined("HEADER_H"));

        let _ = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
        assert_eq!(ctx.conditional_stack.depth(), 0);

        // Second include — ifndef should be false since HEADER_H is defined
        let _ = process_directive(DirectiveKind::Ifndef, "HEADER_H", &mut ctx, loc);
        assert!(!ctx.conditional_stack.is_active());

        let _ = process_directive(DirectiveKind::Endif, "", &mut ctx, loc);
    }

    #[test]
    fn test_macro_redefinition_warning() {
        let mut ctx = test_context();
        let loc = test_loc(1);

        // Define FOO as 1
        let _ = process_directive(DirectiveKind::Define, "FOO 1", &mut ctx, loc);
        assert_eq!(ctx.diagnostics.warning_count(), 0);

        // Redefine FOO as 2 — should trigger a warning
        let _ = process_directive(DirectiveKind::Define, "FOO 2", &mut ctx, test_loc(2));
        assert!(ctx.diagnostics.warning_count() > 0);
    }

    #[test]
    fn test_macro_same_redefinition_no_warning() {
        let mut ctx = test_context();
        let loc = test_loc(1);

        // Define FOO as 1
        let _ = process_directive(DirectiveKind::Define, "FOO 1", &mut ctx, loc);

        // Redefine FOO as 1 (same) — no warning
        let warnings_before = ctx.diagnostics.warning_count();
        let _ = process_directive(DirectiveKind::Define, "FOO 1", &mut ctx, test_loc(2));
        assert_eq!(ctx.diagnostics.warning_count(), warnings_before);
    }
}
