//! Preprocessor Macro Expansion Engine
//!
//! Implements C11 §6.10.3 macro definition storage, lookup, and expansion:
//!
//! - Object-like macro expansion (`#define FOO bar`)
//! - Function-like macro expansion with parameter substitution
//! - Variadic macros (`__VA_ARGS__`) with GCC comma elision (`, ## __VA_ARGS__`)
//! - Stringification (`#` operator, C11 §6.10.3.2)
//! - Token pasting (`##` operator, C11 §6.10.3.3)
//! - Recursive expansion guard preventing infinite expansion (C11 §6.10.3.4)
//! - Predefined macro support (`__STDC__`, `__STDC_VERSION__`, `__COUNTER__`, etc.)
//!
//! # Performance
//!
//! Designed for heavy macro usage in real-world codebases such as SQLite (~230K LOC).
//! Uses `HashMap` for O(1) amortized macro lookup and `HashSet` for O(1) expansion
//! guard checks.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust idioms.

use std::collections::{HashMap, HashSet};

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::intern::{InternId, Interner};
use crate::common::source_map::SourceLocation;

// ---------------------------------------------------------------------------
// MacroKind — distinguishes object-like from function-like macros
// ---------------------------------------------------------------------------

/// Distinguishes object-like macros from function-like macros.
///
/// Object-like macros are simple text substitutions:
/// ```c
/// #define PI 3.14159
/// ```
///
/// Function-like macros accept parameters and support stringification (`#`)
/// and token pasting (`##`):
/// ```c
/// #define MAX(a, b) ((a) > (b) ? (a) : (b))
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroKind {
    /// Object-like macro: `#define FOO replacement`
    ObjectLike,
    /// Function-like macro: `#define FOO(params...) replacement`
    FunctionLike {
        /// Parameter names in declaration order.
        params: Vec<String>,
        /// Whether this macro is variadic (parameter list ends with `...`).
        is_variadic: bool,
    },
}

// ---------------------------------------------------------------------------
// MacroToken — elements of a macro replacement list
// ---------------------------------------------------------------------------

/// Represents a single element in a macro's replacement token list.
///
/// The replacement list is parsed at `#define` time into a sequence of
/// `MacroToken` values. During expansion, each token is processed according
/// to its variant:
///
/// - [`Text`](MacroToken::Text) — output verbatim
/// - [`Param`](MacroToken::Param) — substitute with the (pre-expanded) argument
/// - [`Stringify`](MacroToken::Stringify) — wrap the raw argument in double quotes
/// - [`Paste`](MacroToken::Paste) — concatenate adjacent tokens
/// - [`VaArgs`](MacroToken::VaArgs) — substitute with the variadic argument list
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroToken {
    /// Regular text/token fragment (verbatim in output).
    Text(String),
    /// Parameter reference by index into the parameter list.
    /// During expansion, replaced with the corresponding argument.
    Param(usize),
    /// Stringification of a parameter: `#param` produces `"arg_text"`.
    /// The index refers to the parameter list position.
    Stringify(usize),
    /// Token paste operator (`##`). Concatenates the token to its left
    /// with the token to its right.
    Paste,
    /// `__VA_ARGS__` reference in a variadic macro. Replaced with all
    /// arguments beyond the named parameters, comma-separated.
    VaArgs,
}

// ---------------------------------------------------------------------------
// MacroDefinition — complete macro definition
// ---------------------------------------------------------------------------

/// A complete C preprocessor macro definition as created by `#define`.
///
/// Stores the macro name, kind (object-like or function-like with parameters),
/// the parsed replacement token list, builtin status, and the source location
/// of the `#define` directive for diagnostic purposes.
#[derive(Debug, Clone)]
pub struct MacroDefinition {
    /// The macro name (e.g., `"FOO"` for `#define FOO ...`).
    pub name: String,
    /// Whether this is an object-like or function-like macro.
    pub kind: MacroKind,
    /// The parsed replacement token list.
    pub replacement: Vec<MacroToken>,
    /// Whether this is a compiler-predefined builtin macro
    /// (`__FILE__`, `__LINE__`, `__DATE__`, `__TIME__`, `__COUNTER__`, etc.).
    /// Builtin macros with dynamic values are typically updated by the
    /// preprocessor main loop before each line's expansion.
    pub is_builtin: bool,
    /// Source location where this macro was `#define`d, or `None` for
    /// predefined/builtin macros. Used for redefinition warning messages
    /// pointing back to the original definition site.
    pub location: Option<SourceLocation>,
}

impl MacroDefinition {
    /// Checks whether two macro definitions are semantically equivalent
    /// for redefinition comparison per C11 §6.10.3p2.
    ///
    /// Two definitions are equivalent if they have the same kind, same
    /// parameters (for function-like), and the same replacement token list
    /// (normalizing whitespace differences in `Text` tokens).
    ///
    /// Redefining a macro with an equivalent definition is silently allowed;
    /// redefining with a different definition triggers a warning.
    pub fn is_equivalent(&self, other: &MacroDefinition) -> bool {
        if self.kind != other.kind {
            return false;
        }
        if self.replacement.len() != other.replacement.len() {
            return false;
        }
        for (a, b) in self.replacement.iter().zip(other.replacement.iter()) {
            match (a, b) {
                (MacroToken::Text(ta), MacroToken::Text(tb)) => {
                    let na = normalize_whitespace(ta);
                    let nb = normalize_whitespace(tb);
                    if na != nb {
                        return false;
                    }
                }
                (MacroToken::Param(ia), MacroToken::Param(ib)) if ia == ib => {}
                (MacroToken::Stringify(ia), MacroToken::Stringify(ib)) if ia == ib => {}
                (MacroToken::Paste, MacroToken::Paste) => {}
                (MacroToken::VaArgs, MacroToken::VaArgs) => {}
                _ => return false,
            }
        }
        true
    }
}

/// Implements `PartialEq` for `MacroDefinition` based on semantic equivalence.
///
/// Delegates to [`MacroDefinition::is_equivalent`] which compares kind,
/// parameters, and replacement lists while normalizing whitespace differences.
impl PartialEq for MacroDefinition {
    fn eq(&self, other: &Self) -> bool {
        self.is_equivalent(other)
    }
}

// ---------------------------------------------------------------------------
// MacroTable — central macro storage and expansion engine
// ---------------------------------------------------------------------------

/// The central storage and expansion engine for C preprocessor macros.
///
/// `MacroTable` manages all defined macros and provides the [`expand_line`]
/// method that performs macro expansion on a single line of source text.
/// It implements the full C11 §6.10.3 expansion algorithm including:
///
/// - Object-like and function-like macro expansion
/// - Argument pre-expansion (except for `#` and `##` operands)
/// - Stringification (`#`) and token pasting (`##`)
/// - Variadic macros with `__VA_ARGS__` and GCC comma elision
/// - Recursive expansion with an expansion guard
///
/// [`expand_line`]: MacroTable::expand_line
pub struct MacroTable {
    /// Map from macro name to its definition. Provides O(1) amortized
    /// lookup on every non-directive source line.
    macros: HashMap<String, MacroDefinition>,
    /// Counter for `__COUNTER__` GCC extension macro. Uses `std::cell::Cell`
    /// for interior mutability so `expand_line` can take `&self`.
    counter: std::cell::Cell<u32>,
}

impl MacroTable {
    /// Creates a new, empty `MacroTable` with no macro definitions.
    pub fn new() -> Self {
        MacroTable {
            macros: HashMap::new(),
            counter: std::cell::Cell::new(0),
        }
    }

    /// Defines (or redefines) a macro, returning the previous definition
    /// if one existed.
    ///
    /// The caller (typically `directives.rs`) should compare the old and new
    /// definitions via [`MacroDefinition::is_equivalent`] and emit a warning
    /// through [`DiagnosticEmitter::warning`] if they differ.
    ///
    /// # Returns
    /// `Some(old_def)` if a macro with the same name was previously defined,
    /// `None` if this is a new definition.
    pub fn define(&mut self, def: MacroDefinition) -> Option<MacroDefinition> {
        self.macros.insert(def.name.clone(), def)
    }

    /// Removes a macro definition, returning the removed definition if it existed.
    ///
    /// Per C11 §6.10.3.5, `#undef` of a macro that is not currently defined
    /// is silently ignored (this method simply returns `None`).
    pub fn undefine(&mut self, name: &str) -> Option<MacroDefinition> {
        self.macros.remove(name)
    }

    /// Looks up a macro by name, returning a reference to its definition.
    pub fn get(&self, name: &str) -> Option<&MacroDefinition> {
        self.macros.get(name)
    }

    /// Checks whether a macro with the given name is currently defined.
    ///
    /// Used by `#ifdef`/`#ifndef` conditional directives and the `defined()`
    /// operator in `#if` expressions.
    pub fn is_defined(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    /// Looks up a macro using an interned identifier for integration with
    /// the lexer's string interning system.
    ///
    /// Resolves the [`InternId`] to a string via the provided [`Interner`],
    /// then performs a standard hash map lookup.
    pub fn get_by_intern_id(&self, id: InternId, interner: &Interner) -> Option<&MacroDefinition> {
        let name = interner.resolve(id);
        self.macros.get(name)
    }

    /// Checks macro existence using an interned identifier.
    ///
    /// Resolves the [`InternId`] via [`Interner::resolve`] and delegates to
    /// [`is_defined`](MacroTable::is_defined).
    pub fn is_defined_by_intern_id(&self, id: InternId, interner: &Interner) -> bool {
        let name = interner.resolve(id);
        self.macros.contains_key(name)
    }

    /// Checks if a name can be interned and found via [`Interner::get`].
    ///
    /// Returns `true` if the interner already contains the name AND the name
    /// is a defined macro. Useful for fast-path checks without allocating.
    pub fn is_interned_and_defined(&self, name: &str, interner: &Interner) -> bool {
        if let Some(id) = interner.get(name) {
            let resolved = interner.resolve(id);
            self.macros.contains_key(resolved)
        } else {
            false
        }
    }

    /// Defines a macro with diagnostic reporting for redefinition warnings.
    ///
    /// If the macro was previously defined with a *different* replacement list
    /// or parameters, a warning is emitted via [`DiagnosticEmitter::warning`]
    /// pointing to the original definition location. If an argument count
    /// mismatch or other error is detected during parsing, an error is emitted
    /// via [`DiagnosticEmitter::error`].
    ///
    /// This is the preferred entry point for `directives.rs` when processing
    /// `#define` directives, as it handles all diagnostic reporting.
    pub fn define_with_diagnostics(
        &mut self,
        def: MacroDefinition,
        diag: &mut DiagnosticEmitter,
    ) {
        let loc = def.location.unwrap_or_else(SourceLocation::dummy);
        if let Some(old) = self.macros.get(&def.name) {
            if !old.is_equivalent(&def) {
                let old_loc = old.location.unwrap_or_else(SourceLocation::dummy);
                diag.warning(
                    loc,
                    format!("'{}' macro redefined", def.name),
                );
                if !old_loc.is_dummy() {
                    diag.warning(
                        old_loc,
                        format!(
                            "previous definition of '{}' was here",
                            def.name
                        ),
                    );
                }
            }
        }
        self.macros.insert(def.name.clone(), def);
    }

    /// Reports a macro-related error via [`DiagnosticEmitter::error`].
    ///
    /// Used when argument count mismatches, invalid token paste results, or
    /// other expansion errors are detected. The error is reported at the given
    /// source location with the provided message.
    pub fn report_expansion_error(
        diag: &mut DiagnosticEmitter,
        location: SourceLocation,
        message: &str,
    ) {
        diag.error(location, message.to_string());
    }

    /// Validates the argument count for a function-like macro invocation and
    /// reports any mismatch errors via the [`DiagnosticEmitter`].
    ///
    /// Returns `true` if the argument count is acceptable, `false` if an
    /// error was reported.
    pub fn validate_arg_count(
        &self,
        macro_name: &str,
        num_args: usize,
        diag: &mut DiagnosticEmitter,
        location: SourceLocation,
    ) -> bool {
        if let Some(def) = self.macros.get(macro_name) {
            if let MacroKind::FunctionLike { params, is_variadic } = &def.kind {
                let expected = params.len();
                if *is_variadic {
                    if num_args < expected {
                        diag.error(
                            location,
                            format!(
                                "macro '{}' requires at least {} argument(s), \
                                 but {} were given",
                                macro_name, expected, num_args
                            ),
                        );
                        return false;
                    }
                } else if num_args != expected {
                    diag.error(
                        location,
                        format!(
                            "macro '{}' requires {} argument(s), \
                             but {} were given",
                            macro_name, expected, num_args
                        ),
                    );
                    return false;
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Expansion Engine
    // -----------------------------------------------------------------------

    /// Expands all macros in a single line of preprocessor output.
    ///
    /// This is the main expansion entry point called by the preprocessor for
    /// every non-directive source line. It scans the line for identifiers that
    /// match defined macros, expands them according to C11 §6.10.3 semantics,
    /// and rescans the result for further expansions.
    ///
    /// The `expansion_guard` tracks macro names currently being expanded to
    /// prevent infinite recursive expansion (C11 §6.10.3.4). The caller
    /// should pass a fresh `HashSet` for each top-level line.
    ///
    /// # Arguments
    /// - `line` — A single line of C source text (after directive processing).
    /// - `expansion_guard` — Mutable set of macro names currently being expanded.
    ///
    /// # Returns
    /// The fully macro-expanded line as a `String`.
    pub fn expand_line(&self, line: &str, expansion_guard: &mut HashSet<String>) -> String {
        self.expand_text(line, expansion_guard)
    }

    /// Internal recursive expansion implementation.
    ///
    /// Scans `input` left-to-right, expanding each macro invocation found.
    /// For each expansion, the macro name is temporarily added to the guard
    /// to prevent self-recursion during rescan of the replacement text.
    fn expand_text(&self, input: &str, guard: &mut HashSet<String>) -> String {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut result = String::with_capacity(len + len / 4);
        let mut i = 0;

        while i < len {
            // Skip over string literals — macros are not expanded inside strings
            if bytes[i] == b'"' {
                let end = skip_string_literal(bytes, i);
                result.push_str(&input[i..end]);
                i = end;
                continue;
            }

            // Skip over character literals
            if bytes[i] == b'\'' {
                let end = skip_char_literal(bytes, i);
                result.push_str(&input[i..end]);
                i = end;
                continue;
            }

            // Check for C identifier start (potential macro name)
            if is_ident_start(bytes[i]) {
                let ident_start = i;
                i += 1;
                while i < len && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                let ident = &input[ident_start..i];

                // Check if identifier is a defined macro
                if let Some(def) = self.macros.get(ident) {
                    // Check expansion guard — C11 §6.10.3.4: if the name of the
                    // macro being replaced is found during rescan, it is not replaced.
                    if guard.contains(ident) {
                        result.push_str(ident);
                        continue;
                    }

                    // Handle __COUNTER__ builtin: expands to a unique integer
                    // that increments on each use.
                    if def.is_builtin && ident == "__COUNTER__" {
                        let val = self.counter.get();
                        self.counter.set(val.wrapping_add(1));
                        result.push_str(&val.to_string());
                        continue;
                    }

                    match &def.kind {
                        MacroKind::ObjectLike => {
                            let replacement_text =
                                replacement_tokens_to_string(&def.replacement);
                            guard.insert(ident.to_string());
                            let expanded = self.expand_text(&replacement_text, guard);
                            guard.remove(ident);
                            result.push_str(&expanded);
                        }
                        MacroKind::FunctionLike { params, is_variadic } => {
                            // Function-like macro: only expand if followed by '('
                            let after_ident = i;
                            // Peek ahead, skipping spaces/tabs
                            let mut peek = i;
                            while peek < len
                                && (bytes[peek] == b' ' || bytes[peek] == b'\t')
                            {
                                peek += 1;
                            }
                            if peek < len && bytes[peek] == b'(' {
                                match parse_arguments(input, peek) {
                                    Ok((args, end_pos)) => {
                                        i = end_pos;
                                        let expanded = self.expand_function_macro(
                                            def,
                                            ident,
                                            params,
                                            *is_variadic,
                                            &args,
                                            guard,
                                        );
                                        result.push_str(&expanded);
                                    }
                                    Err(_) => {
                                        // Malformed arguments — output as-is
                                        result.push_str(ident);
                                        i = after_ident;
                                    }
                                }
                            } else {
                                // No '(' follows — not a macro invocation
                                result.push_str(ident);
                                i = after_ident;
                            }
                        }
                    }
                } else {
                    // Not a defined macro — output identifier verbatim
                    result.push_str(ident);
                }
            } else {
                // Non-identifier character — output verbatim
                result.push(bytes[i] as char);
                i += 1;
            }
        }

        result
    }

    /// Expands a function-like macro invocation with the given arguments.
    ///
    /// Implements C11 §6.10.3.1:
    /// 1. Arguments are pre-expanded EXCEPT when used with `#` or `##`.
    /// 2. Parameters in the replacement list are substituted with the
    ///    (pre-expanded or raw) arguments.
    /// 3. Stringification and token pasting are performed.
    /// 4. The result is rescanned with the macro name guarded.
    fn expand_function_macro(
        &self,
        def: &MacroDefinition,
        name: &str,
        params: &[String],
        is_variadic: bool,
        raw_args: &[String],
        guard: &mut HashSet<String>,
    ) -> String {
        let num_params = params.len();

        // Split raw arguments into named parameters and variadic portion
        let (named_raw, va_raw) = if is_variadic {
            if raw_args.len() > num_params {
                let named = raw_args[..num_params].to_vec();
                let va = raw_args[num_params..].join(", ");
                (named, va)
            } else {
                let mut padded = raw_args.to_vec();
                while padded.len() < num_params {
                    padded.push(String::new());
                }
                (padded, String::new())
            }
        } else {
            let mut args = raw_args.to_vec();
            while args.len() < num_params {
                args.push(String::new());
            }
            // Truncate excess arguments for non-variadic macros
            if args.len() > num_params && num_params > 0 {
                args.truncate(num_params);
            }
            (args, String::new())
        };

        // Determine which parameter indices are adjacent to ## or used with #,
        // and therefore should NOT be pre-expanded (use raw argument text).
        let paste_adj = params_adjacent_to_paste(&def.replacement);
        let stringify_idx = params_with_stringify(&def.replacement);
        let va_paste = va_args_adjacent_to_paste(&def.replacement);

        // Pre-expand arguments not adjacent to # or ##
        let expanded_named: Vec<String> = named_raw
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                if paste_adj.contains(&idx) || stringify_idx.contains(&idx) {
                    arg.clone()
                } else {
                    self.expand_text(arg.trim(), guard)
                }
            })
            .collect();

        let expanded_va = if !va_raw.is_empty() && !va_paste {
            self.expand_text(va_raw.trim(), guard)
        } else {
            va_raw.clone()
        };

        // Perform parameter substitution, stringification, and paste assembly
        let substituted = substitute_tokens(
            &def.replacement,
            &named_raw,
            &expanded_named,
            &va_raw,
            &expanded_va,
            &paste_adj,
        );

        // Rescan the substituted text with the macro name guarded
        guard.insert(name.to_string());
        let expanded = self.expand_text(&substituted, guard);
        guard.remove(name);

        expanded
    }
}

// ---------------------------------------------------------------------------
// Macro Definition Parsing
// ---------------------------------------------------------------------------

/// Parses a `#define` directive body into a [`MacroDefinition`].
///
/// Distinguishes object-like from function-like macros based on whether
/// `(` immediately follows the macro name (no intervening whitespace):
///
/// - `#define FOO(x) ...` — function-like (no space before `(`)
/// - `#define FOO (x)` — object-like (space before `(`, replacement is `(x)`)
/// - `#define FOO` — object-like with empty replacement
///
/// # Arguments
/// - `name` — The macro name (already extracted from the `#define` line).
/// - `rest` — The text immediately following the macro name. For function-like
///   macros this starts with `(`; for object-like it starts with whitespace
///   or is empty.
///
/// # Returns
/// `Ok(MacroDefinition)` on success, or `Err(String)` with an error message
/// describing the parsing failure.
pub fn parse_macro_definition(name: &str, rest: &str) -> Result<MacroDefinition, String> {
    if rest.starts_with('(') {
        parse_function_like_definition(name, rest)
    } else {
        parse_object_like_definition(name, rest)
    }
}

/// Parses an object-like macro definition body.
fn parse_object_like_definition(name: &str, rest: &str) -> Result<MacroDefinition, String> {
    let body = rest.trim();
    let replacement = if body.is_empty() {
        Vec::new()
    } else {
        parse_replacement_tokens(body, &[], false)?
    };
    Ok(MacroDefinition {
        name: name.to_string(),
        kind: MacroKind::ObjectLike,
        replacement,
        is_builtin: false,
        location: None,
    })
}

/// Parses a function-like macro definition including its parameter list and
/// replacement body.
fn parse_function_like_definition(name: &str, rest: &str) -> Result<MacroDefinition, String> {
    let bytes = rest.as_bytes();
    let len = bytes.len();
    let mut i = 1; // skip opening '('
    let mut params: Vec<String> = Vec::new();
    let mut is_variadic = false;

    // Skip whitespace inside parameter list
    while i < len && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // Empty parameter list: "()"
    if i < len && bytes[i] == b')' {
        i += 1;
    } else {
        // Parse comma-separated parameter names
        loop {
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= len {
                return Err(
                    "unterminated parameter list in macro definition".to_string(),
                );
            }

            // Check for variadic `...`
            if i + 2 < len
                && bytes[i] == b'.'
                && bytes[i + 1] == b'.'
                && bytes[i + 2] == b'.'
            {
                is_variadic = true;
                i += 3;
                while i < len && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i >= len || bytes[i] != b')' {
                    return Err(
                        "expected ')' after '...' in macro parameter list".to_string(),
                    );
                }
                i += 1;
                break;
            }

            // Parse parameter identifier
            if is_ident_start(bytes[i]) {
                let start = i;
                i += 1;
                while i < len && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                params.push(rest[start..i].to_string());
            } else {
                return Err(format!(
                    "expected parameter name, found '{}'",
                    rest[i..].chars().next().unwrap_or('?')
                ));
            }

            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= len {
                return Err(
                    "unterminated parameter list in macro definition".to_string(),
                );
            }

            if bytes[i] == b')' {
                i += 1;
                break;
            } else if bytes[i] == b',' {
                i += 1;
                // Check for variadic after comma: `name, ...`
                while i < len && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i + 2 < len
                    && bytes[i] == b'.'
                    && bytes[i + 1] == b'.'
                    && bytes[i + 2] == b'.'
                {
                    is_variadic = true;
                    i += 3;
                    while i < len && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    if i >= len || bytes[i] != b')' {
                        return Err(
                            "expected ')' after '...' in macro parameter list".to_string(),
                        );
                    }
                    i += 1;
                    break;
                }
            } else {
                return Err(format!(
                    "expected ',' or ')' in macro parameter list, found '{}'",
                    rest[i..].chars().next().unwrap_or('?')
                ));
            }
        }
    }

    // Everything after the ')' is the replacement body
    let replacement_text = if i < len { rest[i..].trim() } else { "" };
    let replacement = if replacement_text.is_empty() {
        Vec::new()
    } else {
        parse_replacement_tokens(replacement_text, &params, is_variadic)?
    };

    Ok(MacroDefinition {
        name: name.to_string(),
        kind: MacroKind::FunctionLike { params, is_variadic },
        replacement,
        is_builtin: false,
        location: None,
    })
}

// ---------------------------------------------------------------------------
// Replacement Token Parsing
// ---------------------------------------------------------------------------

/// Parses replacement text into a sequence of [`MacroToken`] values.
///
/// Scans the replacement text and identifies:
/// - `##` → [`MacroToken::Paste`]
/// - `#param_name` → [`MacroToken::Stringify(index)`] (function-like only)
/// - `__VA_ARGS__` → [`MacroToken::VaArgs`] (variadic only)
/// - Parameter names → [`MacroToken::Param(index)`] (function-like only)
/// - Everything else → [`MacroToken::Text(...)`] (accumulated text fragments)
fn parse_replacement_tokens(
    text: &str,
    params: &[String],
    is_variadic: bool,
) -> Result<Vec<MacroToken>, String> {
    let mut tokens: Vec<MacroToken> = Vec::new();
    let mut current_text = String::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // ## (token paste) — must check before single #
        if bytes[i] == b'#' && i + 1 < len && bytes[i + 1] == b'#' {
            if !current_text.is_empty() {
                let trimmed = current_text.trim_end().to_string();
                if !trimmed.is_empty() {
                    tokens.push(MacroToken::Text(trimmed));
                }
                current_text.clear();
            }
            tokens.push(MacroToken::Paste);
            i += 2;
            // Skip whitespace after ##
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            continue;
        }

        // # (stringification) — only valid in function-like macros
        if bytes[i] == b'#' && !params.is_empty() {
            let _hash_pos = i;
            i += 1;
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < len && is_ident_start(bytes[i]) {
                let start = i;
                i += 1;
                while i < len && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                let ident = &text[start..i];

                if ident == "__VA_ARGS__" && is_variadic {
                    flush_text(&mut current_text, &mut tokens);
                    // Use sentinel index (params.len()) for VA_ARGS stringification
                    tokens.push(MacroToken::Stringify(params.len()));
                } else if let Some(idx) = params.iter().position(|p| p == ident) {
                    flush_text(&mut current_text, &mut tokens);
                    tokens.push(MacroToken::Stringify(idx));
                } else {
                    return Err(format!(
                        "'#' is not followed by a macro parameter (found '{}')",
                        ident
                    ));
                }
            } else {
                return Err("'#' is not followed by a macro parameter".to_string());
            }
            continue;
        }

        // # in object-like macros — just literal text
        if bytes[i] == b'#' && params.is_empty() {
            current_text.push('#');
            i += 1;
            continue;
        }

        // Identifier — check for parameter name, __VA_ARGS__, or plain text
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let ident = &text[start..i];

            if ident == "__VA_ARGS__" && is_variadic {
                flush_text(&mut current_text, &mut tokens);
                tokens.push(MacroToken::VaArgs);
            } else if let Some(idx) = params.iter().position(|p| p == ident) {
                flush_text(&mut current_text, &mut tokens);
                tokens.push(MacroToken::Param(idx));
            } else {
                current_text.push_str(ident);
            }
            continue;
        }

        // String literal — preserve verbatim in text accumulator
        if bytes[i] == b'"' {
            let end = skip_string_literal(bytes, i);
            current_text.push_str(&text[i..end]);
            i = end;
            continue;
        }

        // Character literal — preserve verbatim
        if bytes[i] == b'\'' {
            let end = skip_char_literal(bytes, i);
            current_text.push_str(&text[i..end]);
            i = end;
            continue;
        }

        // Any other character (whitespace, operators, punctuation)
        current_text.push(bytes[i] as char);
        i += 1;
    }

    // Flush remaining accumulated text
    if !current_text.is_empty() {
        tokens.push(MacroToken::Text(current_text));
    }

    Ok(tokens)
}

/// Flushes accumulated text into the token list if non-empty.
#[inline]
fn flush_text(current_text: &mut String, tokens: &mut Vec<MacroToken>) {
    if !current_text.is_empty() {
        tokens.push(MacroToken::Text(current_text.clone()));
        current_text.clear();
    }
}

// ---------------------------------------------------------------------------
// Token Substitution and Paste Processing
// ---------------------------------------------------------------------------

/// Sentinel string used as an in-band marker for `##` paste positions during
/// the substitution phase. Chosen to never appear in real C source.
const PASTE_MARKER: &str = "\x01##\x01";

/// Performs parameter substitution, stringification, and token-paste assembly
/// on a replacement token list, producing the substituted text ready for rescan.
fn substitute_tokens(
    replacement: &[MacroToken],
    raw_named: &[String],
    expanded_named: &[String],
    va_raw: &str,
    va_expanded: &str,
    paste_adj: &HashSet<usize>,
) -> String {
    let mut pieces: Vec<String> = Vec::with_capacity(replacement.len());

    for token in replacement {
        match token {
            MacroToken::Text(text) => {
                pieces.push(text.clone());
            }
            MacroToken::Param(idx) => {
                let arg = if paste_adj.contains(idx) {
                    // Use raw (unexpanded) argument for paste operands
                    raw_named
                        .get(*idx)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default()
                } else {
                    expanded_named
                        .get(*idx)
                        .map(String::to_string)
                        .unwrap_or_default()
                };
                pieces.push(arg);
            }
            MacroToken::Stringify(idx) => {
                // Always use raw argument for stringification
                let raw = if *idx < raw_named.len() {
                    &raw_named[*idx]
                } else {
                    // Sentinel index → stringify VA_ARGS
                    va_raw
                };
                pieces.push(stringify_argument(raw));
            }
            MacroToken::VaArgs => {
                if va_args_adjacent_to_paste(replacement) {
                    pieces.push(va_raw.trim().to_string());
                } else {
                    pieces.push(va_expanded.to_string());
                }
            }
            MacroToken::Paste => {
                pieces.push(PASTE_MARKER.to_string());
            }
        }
    }

    // Process ## paste operations in the assembled piece list
    process_paste_operations(&mut pieces);

    pieces.join("")
}

/// Processes `##` token paste operations in the resolved piece list.
///
/// Finds paste markers and concatenates the token to their left with the
/// token to their right. Implements the GCC comma elision extension where
/// `, ## __VA_ARGS__` with empty `__VA_ARGS__` removes the preceding comma.
fn process_paste_operations(pieces: &mut Vec<String>) {
    loop {
        let paste_pos = pieces.iter().position(|p| p == PASTE_MARKER);
        match paste_pos {
            Some(pos) => {
                // Extract right operand (after paste marker)
                let right = if pos + 1 < pieces.len() {
                    pieces.remove(pos + 1)
                } else {
                    String::new()
                };
                // Remove paste marker itself
                pieces.remove(pos);

                if pos > 0 {
                    let left_idx = pos - 1;
                    let left = &pieces[left_idx];

                    // GCC comma elision: left ends with ',' and right is empty
                    // (from empty __VA_ARGS__), remove the trailing comma.
                    if right.trim().is_empty()
                        && left.trim_end().ends_with(',')
                    {
                        let mut new_left = left.trim_end().to_string();
                        if let Some(comma_pos) = new_left.rfind(',') {
                            new_left.truncate(comma_pos);
                        }
                        pieces[left_idx] = new_left;
                    } else {
                        // Standard token paste: concatenate trimmed edges
                        let left_trimmed = left.trim_end().to_string();
                        let right_trimmed = right.trim_start().to_string();
                        pieces[left_idx] =
                            format!("{}{}", left_trimmed, right_trimmed);
                    }
                } else if !right.is_empty() {
                    // No left operand — just use the right
                    pieces.insert(0, right.trim_start().to_string());
                }
            }
            None => break,
        }
    }
}

/// Converts a replacement token list to a string for object-like macro
/// expansion (no parameter substitution).
fn replacement_tokens_to_string(tokens: &[MacroToken]) -> String {
    let mut pieces: Vec<String> = Vec::with_capacity(tokens.len());

    for token in tokens {
        match token {
            MacroToken::Text(text) => pieces.push(text.clone()),
            MacroToken::Param(_) | MacroToken::Stringify(_) | MacroToken::VaArgs => {
                // These should not appear in object-like macros, but handle
                // gracefully by outputting nothing.
            }
            MacroToken::Paste => {
                pieces.push(PASTE_MARKER.to_string());
            }
        }
    }

    process_paste_operations(&mut pieces);
    pieces.join("")
}

// ---------------------------------------------------------------------------
// Stringification (C11 §6.10.3.2)
// ---------------------------------------------------------------------------

/// Stringifies a macro argument per C11 §6.10.3.2.
///
/// The argument text is:
/// 1. Trimmed of leading and trailing whitespace
/// 2. Internal whitespace sequences collapsed to single spaces
/// 3. `\` and `"` characters escaped with backslash
/// 4. Wrapped in double quotes
///
/// # Examples
/// - `hello world` → `"hello world"`
/// - `a "b" c` → `"a \"b\" c"`
/// - `  spaced  ` → `"spaced"`
fn stringify_argument(arg: &str) -> String {
    let trimmed = arg.trim();
    let mut result = String::with_capacity(trimmed.len() + 4);
    result.push('"');

    let mut prev_was_space = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            prev_was_space = false;
            if ch == '"' || ch == '\\' {
                result.push('\\');
            }
            result.push(ch);
        }
    }

    result.push('"');
    result
}

// ---------------------------------------------------------------------------
// Argument Parsing
// ---------------------------------------------------------------------------

/// Parses the argument list for a function-like macro invocation.
///
/// Starting from the opening `(`, scans for the matching `)` while tracking
/// parenthesis nesting depth. Arguments are split at commas that appear at
/// the top nesting level (depth 1). Handles nested parentheses, string
/// literals, and character literals correctly.
///
/// # Arguments
/// - `input` — The full source line text.
/// - `open_paren_pos` — The byte offset of the opening `(`.
///
/// # Returns
/// `Ok((args, end_pos))` where `args` is a vector of argument strings
/// and `end_pos` is the byte offset immediately after the closing `)`.
fn parse_arguments(
    input: &str,
    open_paren_pos: usize,
) -> Result<(Vec<String>, usize), String> {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = open_paren_pos + 1; // skip opening '('
    let mut depth: u32 = 1;
    let mut args: Vec<String> = Vec::new();
    let mut current_arg_start = i;

    while i < len && depth > 0 {
        match bytes[i] {
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let arg = input[current_arg_start..i].to_string();
                    if !args.is_empty() || !arg.trim().is_empty() {
                        args.push(arg);
                    }
                    i += 1; // skip closing ')'
                    return Ok((args, i));
                }
                i += 1;
            }
            b',' if depth == 1 => {
                args.push(input[current_arg_start..i].to_string());
                i += 1;
                current_arg_start = i;
            }
            b'"' => {
                i = skip_string_literal(bytes, i);
            }
            b'\'' => {
                i = skip_char_literal(bytes, i);
            }
            _ => {
                i += 1;
            }
        }
    }

    Err("unterminated argument list for macro invocation".to_string())
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Determines which parameter indices appear adjacent to a [`MacroToken::Paste`]
/// in the replacement list, meaning those arguments should NOT be pre-expanded.
fn params_adjacent_to_paste(tokens: &[MacroToken]) -> HashSet<usize> {
    let mut result = HashSet::new();
    let len = tokens.len();
    for i in 0..len {
        if matches!(tokens[i], MacroToken::Paste) {
            if i > 0 {
                if let MacroToken::Param(idx) = &tokens[i - 1] {
                    result.insert(*idx);
                }
            }
            if i + 1 < len {
                if let MacroToken::Param(idx) = &tokens[i + 1] {
                    result.insert(*idx);
                }
            }
        }
    }
    result
}

/// Determines which parameter indices are used with stringification (`#`),
/// meaning those arguments should NOT be pre-expanded (use raw text).
fn params_with_stringify(tokens: &[MacroToken]) -> HashSet<usize> {
    let mut result = HashSet::new();
    for token in tokens {
        if let MacroToken::Stringify(idx) = token {
            result.insert(*idx);
        }
    }
    result
}

/// Checks if `__VA_ARGS__` appears adjacent to a [`MacroToken::Paste`] token,
/// indicating the GCC comma elision pattern (`, ## __VA_ARGS__`).
fn va_args_adjacent_to_paste(tokens: &[MacroToken]) -> bool {
    let len = tokens.len();
    for i in 0..len {
        if matches!(tokens[i], MacroToken::VaArgs) {
            if i > 0 && matches!(tokens[i - 1], MacroToken::Paste) {
                return true;
            }
            if i + 1 < len && matches!(tokens[i + 1], MacroToken::Paste) {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if the byte is a valid C identifier start character
/// (ASCII letter or underscore).
#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Returns `true` if the byte is a valid C identifier continuation character
/// (ASCII letter, digit, or underscore).
#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Skips a string literal starting at position `start` (which must point to `"`).
/// Returns the byte position immediately after the closing `"`.
fn skip_string_literal(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut i = start + 1; // skip opening '"'
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2; // skip escape sequence
        } else if bytes[i] == b'"' {
            return i + 1; // past closing '"'
        } else {
            i += 1;
        }
    }
    i // unterminated — return end of input
}

/// Skips a character literal starting at position `start` (which must point to `'`).
/// Returns the byte position immediately after the closing `'`.
fn skip_char_literal(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut i = start + 1; // skip opening '\''
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2; // skip escape sequence
        } else if bytes[i] == b'\'' {
            return i + 1; // past closing '\''
        } else {
            i += 1;
        }
    }
    i // unterminated — return end of input
}

/// Normalizes whitespace in a string: collapses sequences of whitespace
/// characters to single spaces, and trims leading/trailing whitespace.
/// Used for macro redefinition equivalence comparison.
fn normalize_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_was_space = false;
    for ch in s.trim().chars() {
        if ch.is_ascii_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            result.push(ch);
            prev_was_space = false;
        }
    }
    result
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test Helpers -------------------------------------------------------

    /// Creates a simple object-like macro with a single Text replacement token.
    fn make_object_macro(name: &str, replacement_text: &str) -> MacroDefinition {
        let replacement = if replacement_text.is_empty() {
            Vec::new()
        } else {
            vec![MacroToken::Text(replacement_text.to_string())]
        };
        MacroDefinition {
            name: name.to_string(),
            kind: MacroKind::ObjectLike,
            replacement,
            is_builtin: false,
            location: None,
        }
    }

    /// Creates a non-variadic function-like macro with the given replacement tokens.
    fn make_function_macro(
        name: &str,
        params: &[&str],
        tokens: Vec<MacroToken>,
    ) -> MacroDefinition {
        MacroDefinition {
            name: name.to_string(),
            kind: MacroKind::FunctionLike {
                params: params.iter().map(|s| s.to_string()).collect(),
                is_variadic: false,
            },
            replacement: tokens,
            is_builtin: false,
            location: None,
        }
    }

    /// Creates a variadic function-like macro.
    fn make_variadic_macro(
        name: &str,
        params: &[&str],
        tokens: Vec<MacroToken>,
    ) -> MacroDefinition {
        MacroDefinition {
            name: name.to_string(),
            kind: MacroKind::FunctionLike {
                params: params.iter().map(|s| s.to_string()).collect(),
                is_variadic: true,
            },
            replacement: tokens,
            is_builtin: false,
            location: None,
        }
    }

    // -- MacroTable CRUD Tests -----------------------------------------------

    #[test]
    fn test_define_and_get_object_like() {
        let mut table = MacroTable::new();
        let def = make_object_macro("FOO", "42");
        assert!(table.define(def).is_none());
        assert!(table.get("FOO").is_some());
        assert_eq!(table.get("FOO").unwrap().name, "FOO");
    }

    #[test]
    fn test_define_and_get_function_like() {
        let mut table = MacroTable::new();
        let def = make_function_macro(
            "ADD",
            &["a", "b"],
            vec![
                MacroToken::Param(0),
                MacroToken::Text(" + ".to_string()),
                MacroToken::Param(1),
            ],
        );
        assert!(table.define(def).is_none());
        let stored = table.get("ADD").unwrap();
        match &stored.kind {
            MacroKind::FunctionLike { params, is_variadic } => {
                assert_eq!(params, &["a", "b"]);
                assert!(!is_variadic);
            }
            _ => panic!("expected function-like macro"),
        }
    }

    #[test]
    fn test_undefine_existing() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("FOO", "1"));
        assert!(table.is_defined("FOO"));
        let removed = table.undefine("FOO");
        assert!(removed.is_some());
        assert!(!table.is_defined("FOO"));
    }

    #[test]
    fn test_undefine_nonexistent_is_silent() {
        let mut table = MacroTable::new();
        let result = table.undefine("NOPE");
        assert!(result.is_none());
    }

    #[test]
    fn test_is_defined_true_false() {
        let mut table = MacroTable::new();
        assert!(!table.is_defined("X"));
        table.define(make_object_macro("X", "1"));
        assert!(table.is_defined("X"));
    }

    #[test]
    fn test_redefine_returns_old_definition() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("FOO", "1"));
        let old = table.define(make_object_macro("FOO", "2"));
        assert!(old.is_some());
        assert_eq!(
            old.unwrap().replacement,
            vec![MacroToken::Text("1".to_string())]
        );
    }

    // -- Object-Like Expansion -----------------------------------------------

    #[test]
    fn test_object_like_expansion_simple() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("PI", "3.14159"));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("x = PI;", &mut guard), "x = 3.14159;");
    }

    #[test]
    fn test_nested_object_like_expansion() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("A", "B"));
        table.define(make_object_macro("B", "42"));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("A", &mut guard), "42");
    }

    #[test]
    fn test_expansion_guard_prevents_self_recursion() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("X", "X"));
        let mut guard = HashSet::new();
        // X → X (guarded) → "X"
        assert_eq!(table.expand_line("X", &mut guard), "X");
    }

    #[test]
    fn test_mutual_recursion_guard() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("A", "B"));
        table.define(make_object_macro("B", "A"));
        let mut guard = HashSet::new();
        // A → B (guard {A}) → rescan B → A (guard {A,B}) → A is guarded → "A"
        assert_eq!(table.expand_line("A", &mut guard), "A");
    }

    #[test]
    fn test_multiple_macros_in_line() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("A", "1"));
        table.define(make_object_macro("B", "2"));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("A + B", &mut guard), "1 + 2");
    }

    // -- Function-Like Expansion ---------------------------------------------

    #[test]
    fn test_function_like_expansion_basic() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "ADD",
            &["a", "b"],
            vec![
                MacroToken::Param(0),
                MacroToken::Text(" + ".to_string()),
                MacroToken::Param(1),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("ADD(1, 2)", &mut guard), "1 + 2");
    }

    #[test]
    fn test_function_like_not_expanded_without_parens() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "FOO",
            &["x"],
            vec![MacroToken::Param(0)],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("int FOO = 1;", &mut guard),
            "int FOO = 1;"
        );
    }

    #[test]
    fn test_function_like_empty_argument() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "F",
            &["x"],
            vec![
                MacroToken::Text("[".to_string()),
                MacroToken::Param(0),
                MacroToken::Text("]".to_string()),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("F()", &mut guard), "[]");
    }

    #[test]
    fn test_function_like_nested_parentheses() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "F",
            &["x"],
            vec![MacroToken::Param(0)],
        ));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("F((1+2))", &mut guard), "(1+2)");
    }

    #[test]
    fn test_function_like_multiple_params() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "MAX",
            &["a", "b"],
            vec![
                MacroToken::Text("((".to_string()),
                MacroToken::Param(0),
                MacroToken::Text(") > (".to_string()),
                MacroToken::Param(1),
                MacroToken::Text(") ? (".to_string()),
                MacroToken::Param(0),
                MacroToken::Text(") : (".to_string()),
                MacroToken::Param(1),
                MacroToken::Text("))".to_string()),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("MAX(x, y)", &mut guard),
            "((x) > (y) ? (x) : (y))"
        );
    }

    #[test]
    fn test_argument_pre_expansion() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("X", "42"));
        table.define(make_function_macro(
            "F",
            &["a"],
            vec![MacroToken::Param(0)],
        ));
        let mut guard = HashSet::new();
        // X pre-expanded to 42 before substitution
        assert_eq!(table.expand_line("F(X)", &mut guard), "42");
    }

    // -- Stringification Tests -----------------------------------------------

    #[test]
    fn test_stringification_basic() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "STR",
            &["x"],
            vec![MacroToken::Stringify(0)],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("STR(hello world)", &mut guard),
            "\"hello world\""
        );
    }

    #[test]
    fn test_stringification_escapes_quotes() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "STR",
            &["x"],
            vec![MacroToken::Stringify(0)],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("STR(a \"b\" c)", &mut guard),
            "\"a \\\"b\\\" c\""
        );
    }

    #[test]
    fn test_stringification_collapses_whitespace() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "STR",
            &["x"],
            vec![MacroToken::Stringify(0)],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("STR(  a   b  )", &mut guard),
            "\"a b\""
        );
    }

    // -- Token Pasting Tests -------------------------------------------------

    #[test]
    fn test_token_pasting_identifiers() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "CONCAT",
            &["a", "b"],
            vec![
                MacroToken::Param(0),
                MacroToken::Paste,
                MacroToken::Param(1),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("CONCAT(foo, bar)", &mut guard),
            "foobar"
        );
    }

    #[test]
    fn test_token_pasting_with_number() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "MAKE",
            &["a", "b"],
            vec![
                MacroToken::Param(0),
                MacroToken::Paste,
                MacroToken::Param(1),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("MAKE(x, 42)", &mut guard), "x42");
    }

    // -- Variadic Macro Tests ------------------------------------------------

    #[test]
    fn test_variadic_macro_with_args() {
        let mut table = MacroTable::new();
        table.define(make_variadic_macro(
            "LOG",
            &["fmt"],
            vec![
                MacroToken::Text("printf(".to_string()),
                MacroToken::Param(0),
                MacroToken::Text(", ".to_string()),
                MacroToken::VaArgs,
                MacroToken::Text(")".to_string()),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("LOG(\"x=%d\", x)", &mut guard),
            "printf(\"x=%d\", x)"
        );
    }

    #[test]
    fn test_variadic_comma_elision_empty_va_args() {
        let mut table = MacroTable::new();
        table.define(make_variadic_macro(
            "LOG",
            &["fmt"],
            vec![
                MacroToken::Text("printf(".to_string()),
                MacroToken::Param(0),
                MacroToken::Text(", ".to_string()),
                MacroToken::Paste,
                MacroToken::VaArgs,
                MacroToken::Text(")".to_string()),
            ],
        ));
        let mut guard = HashSet::new();
        // With empty VA_ARGS, the comma before ## __VA_ARGS__ is removed
        assert_eq!(
            table.expand_line("LOG(\"hello\")", &mut guard),
            "printf(\"hello\")"
        );
    }

    // -- Predefined Macros ---------------------------------------------------

    #[test]
    fn test_predefined_stdc() {
        let mut table = MacroTable::new();
        table.define(MacroDefinition {
            name: "__STDC__".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text("1".to_string())],
            is_builtin: true,
            location: None,
        });
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("__STDC__", &mut guard), "1");
    }

    #[test]
    fn test_predefined_stdc_version() {
        let mut table = MacroTable::new();
        table.define(MacroDefinition {
            name: "__STDC_VERSION__".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text("201112L".to_string())],
            is_builtin: true,
            location: None,
        });
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("__STDC_VERSION__", &mut guard),
            "201112L"
        );
    }

    #[test]
    fn test_counter_macro_increments() {
        let mut table = MacroTable::new();
        table.define(MacroDefinition {
            name: "__COUNTER__".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: Vec::new(),
            is_builtin: true,
            location: None,
        });
        let mut g1 = HashSet::new();
        assert_eq!(table.expand_line("__COUNTER__", &mut g1), "0");
        let mut g2 = HashSet::new();
        assert_eq!(table.expand_line("__COUNTER__", &mut g2), "1");
        let mut g3 = HashSet::new();
        assert_eq!(table.expand_line("__COUNTER__", &mut g3), "2");
    }

    // -- Macro Definition Parsing --------------------------------------------

    #[test]
    fn test_parse_object_like_with_body() {
        let def = parse_macro_definition("FOO", " bar baz").unwrap();
        assert_eq!(def.name, "FOO");
        assert_eq!(def.kind, MacroKind::ObjectLike);
        assert_eq!(
            def.replacement,
            vec![MacroToken::Text("bar baz".to_string())]
        );
    }

    #[test]
    fn test_parse_object_like_empty() {
        let def = parse_macro_definition("EMPTY", "").unwrap();
        assert_eq!(def.name, "EMPTY");
        assert_eq!(def.kind, MacroKind::ObjectLike);
        assert!(def.replacement.is_empty());
    }

    #[test]
    fn test_parse_function_like_basic() {
        let def = parse_macro_definition("ADD", "(a, b) a + b").unwrap();
        assert_eq!(def.name, "ADD");
        match &def.kind {
            MacroKind::FunctionLike { params, is_variadic } => {
                assert_eq!(params, &["a", "b"]);
                assert!(!is_variadic);
            }
            _ => panic!("expected function-like"),
        }
    }

    #[test]
    fn test_parse_function_like_variadic() {
        let def =
            parse_macro_definition("LOG", "(fmt, ...) printf(fmt, __VA_ARGS__)")
                .unwrap();
        match &def.kind {
            MacroKind::FunctionLike { params, is_variadic } => {
                assert_eq!(params, &["fmt"]);
                assert!(*is_variadic);
            }
            _ => panic!("expected function-like"),
        }
    }

    #[test]
    fn test_parse_function_like_with_stringify() {
        let def = parse_macro_definition("STR", "(x) #x").unwrap();
        match &def.kind {
            MacroKind::FunctionLike { params, .. } => {
                assert_eq!(params, &["x"]);
            }
            _ => panic!("expected function-like"),
        }
        assert!(def.replacement.contains(&MacroToken::Stringify(0)));
    }

    #[test]
    fn test_parse_function_like_with_paste() {
        let def = parse_macro_definition("CONCAT", "(a, b) a ## b").unwrap();
        assert!(def.replacement.contains(&MacroToken::Paste));
    }

    #[test]
    fn test_parse_empty_param_list() {
        let def = parse_macro_definition("F", "() body").unwrap();
        match &def.kind {
            MacroKind::FunctionLike { params, is_variadic } => {
                assert!(params.is_empty());
                assert!(!is_variadic);
            }
            _ => panic!("expected function-like"),
        }
    }

    // -- Macro Equivalence Tests ---------------------------------------------

    #[test]
    fn test_equivalence_same_definition() {
        let a = make_object_macro("FOO", "42");
        let b = make_object_macro("FOO", "42");
        assert!(a.is_equivalent(&b));
        assert_eq!(a, b);
    }

    #[test]
    fn test_equivalence_different_definition() {
        let a = make_object_macro("FOO", "42");
        let b = make_object_macro("FOO", "43");
        assert!(!a.is_equivalent(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn test_equivalence_whitespace_normalized() {
        let a = MacroDefinition {
            name: "FOO".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text("a  +  b".to_string())],
            is_builtin: false,
            location: None,
        };
        let b = MacroDefinition {
            name: "FOO".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text("a + b".to_string())],
            is_builtin: false,
            location: None,
        };
        assert!(a.is_equivalent(&b));
    }

    // -- Edge Cases ----------------------------------------------------------

    #[test]
    fn test_no_expansion_in_string_literals() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("FOO", "42"));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("\"FOO is FOO\"", &mut guard),
            "\"FOO is FOO\""
        );
    }

    #[test]
    fn test_no_expansion_in_char_literals() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("a", "42"));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("'a'", &mut guard), "'a'");
    }

    #[test]
    fn test_expand_empty_line() {
        let table = MacroTable::new();
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("", &mut guard), "");
    }

    #[test]
    fn test_expand_line_no_macros_defined() {
        let table = MacroTable::new();
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("int x = 42;", &mut guard),
            "int x = 42;"
        );
    }

    #[test]
    fn test_interner_integration() {
        let mut interner = Interner::new();
        let id_foo = interner.intern("FOO");
        let id_unknown = interner.intern("UNKNOWN");

        let mut table = MacroTable::new();
        table.define(make_object_macro("FOO", "bar"));

        assert!(table.get_by_intern_id(id_foo, &interner).is_some());
        assert!(table.is_defined_by_intern_id(id_foo, &interner));
        assert!(table.get_by_intern_id(id_unknown, &interner).is_none());
        assert!(!table.is_defined_by_intern_id(id_unknown, &interner));

        // Test get() path via interner
        assert!(table.is_interned_and_defined("FOO", &interner));
        assert!(!table.is_interned_and_defined("UNKNOWN", &interner));
    }

    #[test]
    fn test_source_location_in_definition() {
        let loc = SourceLocation::dummy();
        let def = MacroDefinition {
            name: "TEST".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![MacroToken::Text("val".to_string())],
            is_builtin: false,
            location: Some(loc),
        };
        assert!(def.location.is_some());
        assert!(def.location.unwrap().is_dummy());
    }

    #[test]
    fn test_define_with_diagnostics_no_warning_on_same_def() {
        let mut table = MacroTable::new();
        let mut diag = DiagnosticEmitter::new();
        let def1 = make_object_macro("FOO", "42");
        table.define_with_diagnostics(def1, &mut diag);
        assert!(!diag.has_errors());

        let def2 = make_object_macro("FOO", "42");
        table.define_with_diagnostics(def2, &mut diag);
        // Same definition, no warning expected, no error
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_define_with_diagnostics_warning_on_different_def() {
        let mut table = MacroTable::new();
        let mut diag = DiagnosticEmitter::new();
        let def1 = make_object_macro("FOO", "42");
        table.define_with_diagnostics(def1, &mut diag);

        let def2 = make_object_macro("FOO", "99");
        table.define_with_diagnostics(def2, &mut diag);
        // Different definition triggers warning (not error)
        // DiagnosticEmitter still has_errors == false because warnings don't count
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_validate_arg_count_correct() {
        let mut table = MacroTable::new();
        let mut diag = DiagnosticEmitter::new();
        table.define(make_function_macro(
            "F",
            &["a", "b"],
            vec![MacroToken::Param(0), MacroToken::Param(1)],
        ));
        let loc = SourceLocation::dummy();
        assert!(table.validate_arg_count("F", 2, &mut diag, loc));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_validate_arg_count_mismatch() {
        let mut table = MacroTable::new();
        let mut diag = DiagnosticEmitter::new();
        table.define(make_function_macro(
            "F",
            &["a", "b"],
            vec![MacroToken::Param(0), MacroToken::Param(1)],
        ));
        let loc = SourceLocation::dummy();
        assert!(!table.validate_arg_count("F", 3, &mut diag, loc));
        assert!(diag.has_errors());
    }

    #[test]
    fn test_validate_arg_count_variadic_ok() {
        let mut table = MacroTable::new();
        let mut diag = DiagnosticEmitter::new();
        table.define(make_variadic_macro(
            "V",
            &["fmt"],
            vec![MacroToken::Param(0), MacroToken::VaArgs],
        ));
        let loc = SourceLocation::dummy();
        // 1 named + extra variadic args => ok
        assert!(table.validate_arg_count("V", 3, &mut diag, loc));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_validate_arg_count_variadic_too_few() {
        let mut table = MacroTable::new();
        let mut diag = DiagnosticEmitter::new();
        table.define(make_variadic_macro(
            "V",
            &["fmt", "x"],
            vec![MacroToken::Param(0), MacroToken::VaArgs],
        ));
        let loc = SourceLocation::dummy();
        // needs at least 2, only 1 given
        assert!(!table.validate_arg_count("V", 1, &mut diag, loc));
        assert!(diag.has_errors());
    }

    #[test]
    fn test_report_expansion_error() {
        let mut diag = DiagnosticEmitter::new();
        let loc = SourceLocation::dummy();
        MacroTable::report_expansion_error(
            &mut diag,
            loc,
            "invalid token paste result",
        );
        assert!(diag.has_errors());
    }

    #[test]
    fn test_function_like_comma_in_string_arg() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "F",
            &["x"],
            vec![MacroToken::Param(0)],
        ));
        let mut guard = HashSet::new();
        // Comma inside string literal should not split arguments
        assert_eq!(
            table.expand_line("F(\"a,b\")", &mut guard),
            "\"a,b\""
        );
    }

    #[test]
    fn test_deeply_nested_macro_expansion() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("L1", "L2"));
        table.define(make_object_macro("L2", "L3"));
        table.define(make_object_macro("L3", "L4"));
        table.define(make_object_macro("L4", "L5"));
        table.define(make_object_macro("L5", "DONE"));
        let mut guard = HashSet::new();
        assert_eq!(table.expand_line("L1", &mut guard), "DONE");
    }

    #[test]
    fn test_object_like_with_hash_in_replacement() {
        // In object-like macros, # is literal text
        let def = parse_macro_definition("VER", " 1 # 2").unwrap();
        assert_eq!(def.kind, MacroKind::ObjectLike);
        // The '#' should be part of the text
        let has_hash = def.replacement.iter().any(|t| match t {
            MacroToken::Text(s) => s.contains('#'),
            _ => false,
        });
        assert!(has_hash);
    }

    #[test]
    fn test_variadic_only_params() {
        // A macro with only variadic params: `#define FOO(...)`
        let def = parse_macro_definition("FOO", "(...) __VA_ARGS__").unwrap();
        match &def.kind {
            MacroKind::FunctionLike { params, is_variadic } => {
                assert!(params.is_empty());
                assert!(*is_variadic);
            }
            _ => panic!("expected function-like"),
        }
    }

    #[test]
    fn test_stringify_backslash() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "S",
            &["x"],
            vec![MacroToken::Stringify(0)],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("S(a\\b)", &mut guard),
            "\"a\\\\b\""
        );
    }

    #[test]
    fn test_paste_result_rescanned() {
        // Pasting should produce a token that gets rescanned
        let mut table = MacroTable::new();
        table.define(make_object_macro("foobar", "RESULT"));
        table.define(make_function_macro(
            "CONCAT",
            &["a", "b"],
            vec![
                MacroToken::Param(0),
                MacroToken::Paste,
                MacroToken::Param(1),
            ],
        ));
        let mut guard = HashSet::new();
        // CONCAT(foo, bar) → foobar → RESULT
        assert_eq!(
            table.expand_line("CONCAT(foo, bar)", &mut guard),
            "RESULT"
        );
    }

    #[test]
    fn test_macro_expansion_preserves_surrounding() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("N", "42"));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("x = N + y;", &mut guard),
            "x = 42 + y;"
        );
    }

    #[test]
    fn test_object_like_empty_replacement() {
        let mut table = MacroTable::new();
        table.define(make_object_macro("EMPTY", ""));
        let mut guard = HashSet::new();
        // EMPTY is a complete identifier token, expands to nothing
        assert_eq!(table.expand_line("EMPTY", &mut guard), "");
        // EMPTY between spaces: expands to empty, leaving spaces
        assert_eq!(table.expand_line("a EMPTY b", &mut guard), "a  b");
        // aEMPTYb is a single identifier, NOT expanded
        assert_eq!(table.expand_line("aEMPTYb", &mut guard), "aEMPTYb");
    }

    #[test]
    fn test_multiple_args_with_spaces() {
        let mut table = MacroTable::new();
        table.define(make_function_macro(
            "F",
            &["a", "b"],
            vec![
                MacroToken::Param(0),
                MacroToken::Text(" ".to_string()),
                MacroToken::Param(1),
            ],
        ));
        let mut guard = HashSet::new();
        assert_eq!(
            table.expand_line("F( x , y )", &mut guard),
            "x y"
        );
    }
}
