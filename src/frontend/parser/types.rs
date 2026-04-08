//! Type specifier parsing for the `bcc` C compiler.
//!
//! This module handles parsing all C11 type specifiers, type qualifiers,
//! pointer declarators, array declarators, function pointer declarators, and
//! abstract declarators used in casts, sizeof, typeof, and declaration contexts.
//!
//! # Design
//!
//! The type parsing follows the C11 grammar where type specifiers are composed
//! of one or more keywords that combine into a single type (e.g.,
//! `unsigned long long int`). A state-machine approach tracks which specifier
//! keywords have been seen and validates combinations, reporting errors for
//! invalid specifier sequences like `int float` or `short double`.
//!
//! # C11 Coverage
//!
//! - **Basic types**: `void`, `char`, `short`, `int`, `long`, `long long`,
//!   `float`, `double`, `long double`
//! - **Signed/unsigned modifiers**: `signed`, `unsigned` combined with any
//!   integer type
//! - **C11 special types**: `_Bool`, `_Complex`, `_Atomic(type)`
//! - **Composite type references**: `struct tag`, `union tag`, `enum tag`
//! - **Typedef names**: Identifiers registered via `typedef` declarations
//! - **GCC extensions**: `typeof(expr)`, `typeof(type)`, `__attribute__`
//!
//! # Type qualifiers
//!
//! `const`, `volatile`, `restrict`, `_Atomic` (without parentheses), including
//! GCC double-underscore variants `__const__`, `__volatile__`, `__restrict__`.
//!
//! # Disambiguation
//!
//! The critical `is_type_specifier_start` function enables the expression
//! parser to disambiguate between type names (for casts, sizeof) and
//! expressions. This requires tracking typedef names in the parser state.
//!
//! # Integration Points
//!
//! - Called by `declarations.rs` for declaration specifiers and parameter types
//! - Called by `expressions.rs` for cast, sizeof, _Alignof type arguments
//! - Called by `gcc_extensions.rs` for typeof type arguments
//! - Per AAP §0.5.1 Group 2: "Type specifier parsing: struct, union, enum,
//!   typedef names, qualifiers, array declarators, function pointers"
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use super::ast::{
    AbstractDeclarator, ArraySize, DeclSpecifiers, DirectAbstractDeclarator, Expression,
    ParamDeclaration, ParamList, Pointer, StorageClass, TypeName, TypeQualifier, TypeSpecifier,
};
use super::expressions::is_type_start_token;
use super::gcc_extensions;
use super::Parser;
use crate::common::intern::InternId;
use crate::frontend::lexer::token::{TokenKind, TokenValue};

// ===========================================================================
// Type Specifier State Machine
// ===========================================================================

/// Internal flags tracking which type specifier keywords have been seen during
/// parsing of a multi-keyword type specifier sequence. Used to validate
/// combinations and resolve the final `TypeSpecifier` variant.
///
/// The state machine approach handles all valid C11 type specifier combinations:
/// - `unsigned long long int` → four flags set
/// - `signed` alone → implies `signed int`
/// - `long double` → long + double
/// - Invalid: `int float`, `short double`, `long long long`
#[derive(Default)]
struct TypeSpecState {
    /// Set when `void` keyword is seen.
    has_void: bool,
    /// Set when `char` keyword is seen.
    has_char: bool,
    /// Set when `short` keyword is seen.
    has_short: bool,
    /// Set when `int` keyword is seen.
    has_int: bool,
    /// Count of `long` keywords seen (0, 1, or 2 for `long long`).
    long_count: u8,
    /// Set when `float` keyword is seen.
    has_float: bool,
    /// Set when `double` keyword is seen.
    has_double: bool,
    /// Set when `signed` keyword is seen.
    has_signed: bool,
    /// Set when `unsigned` keyword is seen.
    has_unsigned: bool,
    /// Set when `_Bool` keyword is seen.
    has_bool: bool,
    /// Set when `_Complex` keyword is seen.
    has_complex: bool,
    /// Set when a struct/union/enum/typedef/typeof specifier is already stored.
    has_other: bool,
    /// The stored "other" specifier (struct/union/enum/typedef/typeof/atomic).
    other_spec: Option<TypeSpecifier>,
    /// Total number of specifier keywords consumed (used for error detection).
    count: u32,
}

impl TypeSpecState {
    /// Returns `true` if no specifier keywords have been consumed yet.
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Resolves the collected flags into a concrete `TypeSpecifier` variant.
    ///
    /// Handles all valid combinations of C11 type specifier keywords and
    /// wraps with `Signed`/`Unsigned` when those modifiers are present.
    /// Returns `TypeSpecifier::Error` only as a last resort (the caller
    /// should have reported errors during flag collection).
    fn resolve(self) -> TypeSpecifier {
        // If we have an "other" specifier (struct/union/enum/typedef/typeof/atomic),
        // that takes precedence. The signed/unsigned/long modifiers are not valid
        // with these — errors were reported during collection.
        if let Some(spec) = self.other_spec {
            return spec;
        }

        // Resolve the base type from the collected flags.
        let base = if self.has_void {
            TypeSpecifier::Void
        } else if self.has_bool {
            TypeSpecifier::Bool
        } else if self.has_char {
            TypeSpecifier::Char
        } else if self.has_short {
            // `short` or `short int`
            TypeSpecifier::Short
        } else if self.long_count == 2 {
            // `long long` or `long long int`
            TypeSpecifier::LongLong
        } else if self.long_count == 1 && self.has_double {
            // `long double`
            TypeSpecifier::LongDouble
        } else if self.long_count == 1 {
            // `long` or `long int`
            TypeSpecifier::Long
        } else if self.has_double {
            TypeSpecifier::Double
        } else if self.has_float {
            TypeSpecifier::Float
        } else if self.has_int || self.has_signed || self.has_unsigned {
            // `int`, `signed`, `unsigned`, `signed int`, `unsigned int`
            TypeSpecifier::Int
        } else {
            // Should not reach here if count > 0, but handle gracefully.
            TypeSpecifier::Error
        };

        // Apply _Complex modifier.
        let base = if self.has_complex {
            TypeSpecifier::Complex(Box::new(base))
        } else {
            base
        };

        // Apply signed/unsigned modifier.
        if self.has_unsigned {
            TypeSpecifier::Unsigned(Box::new(base))
        } else if self.has_signed {
            TypeSpecifier::Signed(Box::new(base))
        } else {
            base
        }
    }
}

// ===========================================================================
// parse_type_name — Main Entry Point for Casts/Sizeof/Typeof
// ===========================================================================

/// Parses a type name used in casts, sizeof, _Alignof, _Generic, and typeof.
///
/// Grammar:
/// ```text
///   type-name := specifier-qualifier-list abstract-declarator?
/// ```
///
/// The specifier-qualifier-list consists of type specifiers and type qualifiers
/// (no storage classes or function specifiers are allowed in type names).
///
/// Returns a `TypeName` with the parsed declaration specifiers and an optional
/// abstract declarator for pointer, array, and function-pointer modifiers.
pub(super) fn parse_type_name(parser: &mut Parser<'_>) -> TypeName {
    let start = parser.current_span();

    // Parse type qualifiers and type specifiers intermixed (no storage classes).
    let mut qualifiers = Vec::new();
    let mut attributes = Vec::new();

    // Collect leading qualifiers and attributes.
    loop {
        // Collect type qualifiers that appear before or intermixed with specifiers.
        let mut new_quals = parse_type_qualifiers(parser);
        qualifiers.append(&mut new_quals);

        // Collect GCC __attribute__ annotations.
        let mut new_attrs = gcc_extensions::try_parse_attributes(parser);
        attributes.append(&mut new_attrs);

        if new_quals.is_empty() && new_attrs.is_empty() {
            break;
        }
    }

    // Parse the base type specifier(s).
    let type_spec = parse_type_specifiers(parser);

    // Collect trailing qualifiers and attributes.
    loop {
        let mut new_quals = parse_type_qualifiers(parser);
        qualifiers.append(&mut new_quals);

        let mut new_attrs = gcc_extensions::try_parse_attributes(parser);
        attributes.append(&mut new_attrs);

        if new_quals.is_empty() && new_attrs.is_empty() {
            break;
        }
    }

    // Parse optional abstract declarator (pointer, array, function modifiers).
    let abstract_declarator = if can_start_abstract_declarator(parser) {
        Some(parse_abstract_declarator(parser))
    } else {
        None
    };

    let span = parser.span_from(start);
    TypeName {
        specifiers: DeclSpecifiers {
            storage_class: None,
            type_qualifiers: qualifiers,
            type_specifier: type_spec,
            function_specifiers: Vec::new(),
            attributes,
            alignment: None,
            span,
        },
        abstract_declarator,
        span,
    }
}

/// Returns `true` if the current token can begin an abstract declarator.
///
/// An abstract declarator starts with `*` (pointer), `(` (grouped/function),
/// or `[` (array).
fn can_start_abstract_declarator(parser: &Parser<'_>) -> bool {
    matches!(
        parser.current().kind,
        TokenKind::Star | TokenKind::LeftParen | TokenKind::LeftBracket
    )
}

// ===========================================================================
// parse_type_specifiers — Base Type Specifier Parsing
// ===========================================================================

/// Parses one or more type specifier tokens that combine into a single type.
///
/// Handles all C11 base type specifiers:
/// - Basic types: `void`, `char`, `short`, `int`, `long`, `float`, `double`
/// - Multi-keyword combinations: `unsigned long long int`, `long double`, etc.
/// - Signed/unsigned modifiers: `signed char`, `unsigned int`, etc.
/// - C11 special types: `_Bool`, `_Complex float`, `_Atomic(type)`
/// - Struct/union/enum references: `struct tag`, `union tag`, `enum tag`
/// - Typedef names: identifiers previously registered via `typedef`
/// - GCC typeof: `typeof(expr)` / `typeof(type)` delegated to gcc_extensions
///
/// Uses a state machine to track consumed keywords and validate combinations.
/// Reports errors for invalid combinations like `int float` or `short double`.
pub(super) fn parse_type_specifiers(parser: &mut Parser<'_>) -> TypeSpecifier {
    let start = parser.current_span();
    let mut state = TypeSpecState::default();

    loop {
        let kind = parser.current().kind;

        match kind {
            // === Basic type keywords ===
            TokenKind::Void => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_int
                    || state.long_count > 0
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'void'");
                    break;
                }
                state.has_void = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Char => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_int
                    || state.long_count > 0
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'char'");
                    break;
                }
                state.has_char = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Short => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.long_count > 0
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'short'");
                    break;
                }
                state.has_short = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Int => {
                if state.has_void
                    || state.has_char
                    || state.has_int
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'int'");
                    break;
                }
                state.has_int = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Long => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_float
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'long'");
                    break;
                }
                if state.long_count >= 2 {
                    parser.error("'long long long' is too long for a type specifier");
                    break;
                }
                state.long_count += 1;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Float => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_int
                    || state.long_count > 0
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_unsigned
                    || state.has_signed
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'float'");
                    break;
                }
                state.has_float = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Double => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_int
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_unsigned
                    || state.has_signed
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'double'");
                    break;
                }
                // Allow `long double` (long_count == 1 is valid).
                if state.long_count > 1 {
                    parser.error("'long long double' is not a valid type specifier");
                    break;
                }
                state.has_double = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Signed => {
                if state.has_signed || state.has_unsigned {
                    parser.error("duplicate 'signed' specifier");
                    break;
                }
                if state.has_void
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'signed'");
                    break;
                }
                state.has_signed = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Unsigned => {
                if state.has_signed || state.has_unsigned {
                    parser.error("duplicate or conflicting 'unsigned' specifier");
                    break;
                }
                if state.has_void
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("invalid combination of type specifiers with 'unsigned'");
                    break;
                }
                state.has_unsigned = true;
                state.count += 1;
                parser.advance();
            }

            // === C11 special types ===
            TokenKind::Bool => {
                if !state.is_empty() {
                    parser.error("'_Bool' cannot be combined with other type specifiers");
                    break;
                }
                state.has_bool = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Complex => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_int
                    || state.has_bool
                    || state.has_complex
                    || state.has_other
                {
                    parser.error(
                        "'_Complex' can only be used with 'float', 'double', or 'long double'",
                    );
                    break;
                }
                state.has_complex = true;
                state.count += 1;
                parser.advance();
            }

            TokenKind::Atomic => {
                // `_Atomic(type)` — type specifier form with parenthesized type.
                // `_Atomic` without parens is a type qualifier, not handled here.
                if parser.peek().kind == TokenKind::LeftParen {
                    if !state.is_empty() {
                        parser.error("'_Atomic' cannot be combined with other type specifiers");
                        break;
                    }
                    parser.advance(); // consume `_Atomic`
                    parser.advance(); // consume `(`
                    let inner = parse_type_name(parser);
                    if parser.expect(TokenKind::RightParen).is_err() {
                        return TypeSpecifier::Error;
                    }
                    let inner_spec = inner.specifiers.type_specifier;
                    state.has_other = true;
                    state.other_spec = Some(TypeSpecifier::Atomic(Box::new(inner_spec)));
                    state.count += 1;
                } else {
                    // `_Atomic` as a qualifier — not a specifier; stop parsing.
                    break;
                }
            }

            // === Struct/union/enum references and definitions ===
            TokenKind::Struct => {
                if !state.is_empty() {
                    parser.error("'struct' cannot be combined with other type specifiers");
                    break;
                }
                let spec = parse_struct_or_union_specifier(parser, true);
                state.has_other = true;
                state.other_spec = Some(spec);
                state.count += 1;
            }

            TokenKind::Union => {
                if !state.is_empty() {
                    parser.error("'union' cannot be combined with other type specifiers");
                    break;
                }
                let spec = parse_struct_or_union_specifier(parser, false);
                state.has_other = true;
                state.other_spec = Some(spec);
                state.count += 1;
            }

            TokenKind::Enum => {
                if !state.is_empty() {
                    parser.error("'enum' cannot be combined with other type specifiers");
                    break;
                }
                let spec = parse_enum_specifier(parser);
                state.has_other = true;
                state.other_spec = Some(spec);
                state.count += 1;
            }

            // === GCC typeof ===
            TokenKind::Typeof => {
                if !state.is_empty() {
                    parser.error("'typeof' cannot be combined with other type specifiers");
                    break;
                }
                let spec = gcc_extensions::parse_typeof(parser);
                state.has_other = true;
                state.other_spec = Some(spec);
                state.count += 1;
            }

            // === Typedef names and identifiers ===
            TokenKind::Identifier => {
                // Check if this identifier is a registered typedef name.
                if let TokenValue::Identifier(id) = parser.current().value {
                    if parser.is_typedef_name(id) && state.is_empty() {
                        let id_span = parser.current_span();
                        parser.advance();
                        let span = parser.span_from(id_span);
                        state.has_other = true;
                        state.other_spec = Some(TypeSpecifier::TypedefName { name: id, span });
                        state.count += 1;
                    } else {
                        // Not a type specifier; stop parsing specifiers.
                        break;
                    }
                } else {
                    break;
                }
            }

            // === GCC __int128 ===
            TokenKind::GccInt128 => {
                if state.has_void
                    || state.has_char
                    || state.has_short
                    || state.has_int
                    || state.long_count > 0
                    || state.has_float
                    || state.has_double
                    || state.has_bool
                    || state.has_other
                {
                    parser.error("'__int128' cannot be combined with other type specifiers");
                    break;
                }
                // Treat __int128 as a LongLong variant for now (the semantic
                // analyzer will differentiate).
                parser.advance();
                state.has_other = true;
                state.other_spec = Some(TypeSpecifier::LongLong);
                state.count += 1;
            }

            // === GCC __auto_type ===
            TokenKind::GccAutoType => {
                if !state.is_empty() {
                    parser.error("'__auto_type' cannot be combined with other type specifiers");
                    break;
                }
                parser.advance();
                state.has_other = true;
                // Represent as Int for now; the semantic analyzer will handle
                // type inference.
                state.other_spec = Some(TypeSpecifier::Int);
                state.count += 1;
            }

            // === __builtin_va_list ===
            TokenKind::BuiltinVaList => {
                if !state.is_empty() {
                    parser
                        .error("'__builtin_va_list' cannot be combined with other type specifiers");
                    break;
                }
                let va_span = parser.current_span();
                parser.advance();
                let span = parser.span_from(va_span);
                // Represent as a typedef-like name. The semantic analyzer
                // knows about __builtin_va_list.
                state.has_other = true;
                state.other_spec = Some(TypeSpecifier::TypedefName {
                    name: if let Some(id) = parser.lookup_interned("__builtin_va_list") {
                        id
                    } else {
                        // Fallback: use a dummy InternId. The lexer should have
                        // already interned all builtin keywords.
                        InternId::from_raw(0)
                    },
                    span,
                });
                state.count += 1;
            }

            // --- Type qualifiers interleaved with type specifiers ---
            // C11 §6.7 allows declaration-specifiers in any order, so
            // qualifiers like `const`, `volatile`, `restrict` can appear
            // between type specifiers (e.g. `unsigned const char`).
            // We skip them here so the type specifier parsing continues.
            TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => {
                parser.advance();
            }
            TokenKind::Atomic => {
                // _Atomic without parens is a qualifier — skip it.
                if parser.peek().kind != TokenKind::LeftParen {
                    parser.advance();
                } else {
                    break;
                }
            }

            // Any other token is not a type specifier — stop.
            _ => break,
        }
    }

    // If no specifiers were consumed, return Error.
    if state.is_empty() {
        return TypeSpecifier::Error;
    }

    let _ = start; // used for span tracking in the state machine above
    state.resolve()
}

// ===========================================================================
// Struct/Union/Enum Specifier Helpers
// ===========================================================================

/// Parses a `struct` or `union` specifier: tag reference or inline definition.
///
/// Grammar:
/// ```text
///   struct-or-union-specifier := ('struct' | 'union') identifier? '{' member-list '}'
///                              | ('struct' | 'union') identifier
/// ```
///
/// If a `{` follows the tag name, this is a definition and we delegate to the
/// declarations module. Otherwise, it is a forward reference.
fn parse_struct_or_union_specifier(parser: &mut Parser<'_>, is_struct: bool) -> TypeSpecifier {
    let start = parser.current_span();

    // Consume `struct` or `union` keyword.
    parser.advance();

    // Parse optional GCC attributes after struct/union keyword.
    let _attrs = gcc_extensions::try_parse_attributes(parser);

    // Parse optional tag name.
    let tag = if parser.current().kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            parser.advance();
            Some(id)
        } else {
            None
        }
    } else {
        None
    };

    // Parse optional GCC attributes after tag name.
    let _attrs2 = gcc_extensions::try_parse_attributes(parser);

    // If `{` follows, this is a definition — but we still emit a reference
    // from types.rs since the definition body parsing lives in declarations.rs.
    // In practice, the caller (declarations.rs) should handle full definitions.
    // Here we handle the reference-only case (no `{`).
    if parser.check(TokenKind::LeftBrace) {
        // This is a struct/union definition. We parse the body here for
        // the type specifier context (e.g., `sizeof(struct { int x; })`).
        let members = parse_struct_or_union_body(parser);
        let span = parser.span_from(start);

        if is_struct {
            TypeSpecifier::Struct(super::ast::StructDef {
                tag,
                members,
                attributes: Vec::new(),
                span,
            })
        } else {
            TypeSpecifier::Union(super::ast::UnionDef {
                tag,
                members,
                attributes: Vec::new(),
                span,
            })
        }
    } else if let Some(tag_name) = tag {
        let span = parser.span_from(start);
        if is_struct {
            TypeSpecifier::StructRef {
                tag: tag_name,
                span,
            }
        } else {
            TypeSpecifier::UnionRef {
                tag: tag_name,
                span,
            }
        }
    } else {
        parser.error("expected identifier or '{' after 'struct'/'union'");
        TypeSpecifier::Error
    }
}

/// Parses the body of a struct or union definition: `{ member-list }`.
///
/// Returns a vector of `StructMember` entries. Each member is a declaration
/// with optional bit-field widths.
fn parse_struct_or_union_body(parser: &mut Parser<'_>) -> Vec<super::ast::StructMember> {
    let mut members = Vec::new();

    // Consume `{`.
    if parser.expect(TokenKind::LeftBrace).is_err() {
        return members;
    }

    while !parser.check(TokenKind::RightBrace) && !parser.is_at_end() {
        // Skip stray semicolons.
        if parser.match_token(TokenKind::Semicolon) {
            continue;
        }

        // Parse a struct member declaration.
        // We parse the specifier-qualifier-list and declarator-list here.
        let member_start = parser.current_span();

        // Handle _Static_assert inside structs.
        if parser.check(TokenKind::StaticAssert) {
            // Skip _Static_assert — consume until `;`.
            while !parser.check(TokenKind::Semicolon)
                && !parser.check(TokenKind::RightBrace)
                && !parser.is_at_end()
            {
                parser.advance();
            }
            let _ = parser.match_token(TokenKind::Semicolon);
            continue;
        }

        // Skip GCC __extension__ keyword (common before anonymous unions
        // in system headers like `__extension__ union { ... };`).
        while parser.check(TokenKind::GccExtension) {
            parser.advance();
        }

        // Parse type specifiers and qualifiers for the member.
        let mut qualifiers = parse_type_qualifiers(parser);
        let type_spec = parse_type_specifiers(parser);
        let mut more_quals = parse_type_qualifiers(parser);
        qualifiers.append(&mut more_quals);

        // Parse GCC __attribute__ that may appear between type specifiers
        // and the declarator name. This handles patterns like:
        //   int __attribute__((aligned(8))) field_name;
        //   __u64 __attribute__((packed)) offset;
        // which are pervasive in the Linux kernel via macros like
        // __aligned_u64 that expand to `__u64 __attribute__((aligned(8)))`.
        let _mid_attrs = super::gcc_extensions::try_parse_attributes(parser);

        if matches!(type_spec, TypeSpecifier::Error) {
            // Recovery: skip to next `;` or `}`.
            while !parser.check(TokenKind::Semicolon)
                && !parser.check(TokenKind::RightBrace)
                && !parser.is_at_end()
            {
                parser.advance();
            }
            let _ = parser.match_token(TokenKind::Semicolon);
            continue;
        }

        let spec = DeclSpecifiers {
            storage_class: None,
            type_qualifiers: qualifiers,
            type_specifier: type_spec,
            function_specifiers: Vec::new(),
            attributes: Vec::new(),
            alignment: None,
            span: parser.span_from(member_start),
        };

        // Parse declarator list (possibly with bit-field widths).
        let mut field_decls = Vec::new();

        // Handle the case of an anonymous struct/union member.
        if parser.check(TokenKind::Semicolon) {
            // Anonymous member or just type specifier with no declarator.
            let member_span = parser.span_from(member_start);
            parser.advance(); // consume `;`
            members.push(super::ast::StructMember::Field {
                specifiers: spec,
                declarators: Vec::new(),
                span: member_span,
            });
            continue;
        }

        loop {
            let decl_start = parser.current_span();
            let mut declarator = None;
            let mut bit_width = None;

            // Parse optional declarator (may be absent for unnamed bit-fields).
            if !parser.check(TokenKind::Colon)
                && !parser.check(TokenKind::Semicolon)
                && !parser.check(TokenKind::Comma)
            {
                // Parse a simplified declarator for struct members.
                declarator = Some(parse_member_declarator(parser));
            }

            // Parse optional bit-field width.
            if parser.match_token(TokenKind::Colon) {
                // Parse the constant expression for bit-field width.
                bit_width = Some(Box::new(parse_constant_expression_simple(parser)));
            }

            let decl_span = parser.span_from(decl_start);
            field_decls.push(super::ast::StructFieldDeclarator {
                declarator,
                bit_width,
                span: decl_span,
            });

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        let member_span = parser.span_from(member_start);

        // Expect `;` after the member declaration.
        let _ = parser.expect(TokenKind::Semicolon);

        members.push(super::ast::StructMember::Field {
            specifiers: spec,
            declarators: field_decls,
            span: member_span,
        });
    }

    // Consume `}`.
    let _ = parser.expect(TokenKind::RightBrace);

    members
}

/// Parses a simple declarator for struct/union members.
///
/// Handles pointers, identifiers, arrays, and parenthesized declarators.
fn parse_member_declarator(parser: &mut Parser<'_>) -> super::ast::Declarator {
    let start = parser.current_span();

    // Parse pointer prefix.
    let ptrs = parse_pointer(parser);

    // Parse identifier or parenthesized declarator.
    let direct = if parser.current().kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            parser.advance();
            super::ast::DirectDeclarator::Identifier(id)
        } else {
            super::ast::DirectDeclarator::Abstract
        }
    } else if parser.match_token(TokenKind::LeftParen) {
        // Parenthesized declarator (e.g., function pointer member).
        let inner = parse_member_declarator(parser);
        let _ = parser.expect(TokenKind::RightParen);
        super::ast::DirectDeclarator::Parenthesized(Box::new(inner))
    } else {
        super::ast::DirectDeclarator::Abstract
    };

    // Parse array suffixes.
    let direct = parse_direct_declarator_suffixes(parser, direct);

    let span = parser.span_from(start);
    super::ast::Declarator {
        pointer: ptrs,
        direct,
        attributes: gcc_extensions::try_parse_attributes(parser),
        span,
    }
}

/// Parses direct declarator suffixes: array `[...]` and function `(...)`.
fn parse_direct_declarator_suffixes(
    parser: &mut Parser<'_>,
    mut base: super::ast::DirectDeclarator,
) -> super::ast::DirectDeclarator {
    loop {
        if parser.check(TokenKind::LeftBracket) {
            let dims = parse_array_dimensions(parser);
            for dim in dims {
                base = super::ast::DirectDeclarator::Array {
                    base: Box::new(base),
                    size: dim,
                    qualifiers: Vec::new(),
                };
            }
        } else if parser.check(TokenKind::LeftParen) {
            // Could be a function parameter list.
            let params = parse_parameter_type_list(parser);
            base = super::ast::DirectDeclarator::Function {
                base: Box::new(base),
                params,
            };
        } else {
            break;
        }
    }
    base
}

/// Parses a simple constant expression (for bit-field widths and array sizes).
///
/// This is a simplified expression parser that handles integer literals,
/// identifiers, sizeof, parenthesized expressions (including cast
/// expressions), and basic arithmetic with proper binary operator handling
/// after ANY primary expression. Supports ternary (`?:`) for preprocessor-
/// expanded macros that use conditional constant expressions.
fn parse_constant_expression_simple(parser: &mut Parser<'_>) -> Expression {
    let start = parser.current_span();

    // Step 1: Parse the primary/unary atom.
    let atom = parse_const_expr_atom(parser, start);

    // Step 2: Handle binary operators after the atom. This loop applies
    // regardless of whether the atom was an integer literal, identifier,
    // parenthesized expression, sizeof, or unary expression.
    let mut result = atom;
    loop {
        let op = match parser.current().kind {
            TokenKind::Pipe => super::ast::BinaryOp::BitwiseOr,
            TokenKind::Amp => super::ast::BinaryOp::BitwiseAnd,
            TokenKind::Caret => super::ast::BinaryOp::BitwiseXor,
            TokenKind::Plus => super::ast::BinaryOp::Add,
            TokenKind::Minus => super::ast::BinaryOp::Sub,
            TokenKind::Star => super::ast::BinaryOp::Mul,
            TokenKind::Slash => super::ast::BinaryOp::Div,
            TokenKind::Percent => super::ast::BinaryOp::Mod,
            TokenKind::LessLess => super::ast::BinaryOp::ShiftLeft,
            TokenKind::GreaterGreater => super::ast::BinaryOp::ShiftRight,
            TokenKind::Less => super::ast::BinaryOp::Less,
            TokenKind::LessEqual => super::ast::BinaryOp::LessEqual,
            TokenKind::Greater => super::ast::BinaryOp::Greater,
            TokenKind::GreaterEqual => super::ast::BinaryOp::GreaterEqual,
            TokenKind::EqualEqual => super::ast::BinaryOp::Equal,
            TokenKind::BangEqual => super::ast::BinaryOp::NotEqual,
            TokenKind::AmpAmp => super::ast::BinaryOp::LogicalAnd,
            TokenKind::PipePipe => super::ast::BinaryOp::LogicalOr,
            _ => break,
        };
        parser.advance();
        let rhs = {
            let rhs_start = parser.current_span();
            parse_const_expr_atom(parser, rhs_start)
        };
        let span = parser.span_from(start);
        result = Expression::Binary {
            op,
            left: Box::new(result),
            right: Box::new(rhs),
            span,
        };
    }

    // Step 3: Handle ternary `? :` for conditional constant expressions.
    if parser.current().kind == TokenKind::Question {
        parser.advance(); // consume `?`
        let then_expr = parse_constant_expression_simple(parser);
        let _ = parser.expect(TokenKind::Colon);
        let else_expr = parse_constant_expression_simple(parser);
        let span = parser.span_from(start);
        result = Expression::Ternary {
            condition: Box::new(result),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
            span,
        };
    }

    result
}

/// Parses a single primary or unary atom for `parse_constant_expression_simple`.
///
/// This handles integer literals, identifiers, parenthesized expressions
/// (including type casts like `(int)x`), unary operators, `sizeof`, and
/// `_Alignof`. It does NOT handle binary operators — those are handled in the
/// caller's loop.
fn parse_const_expr_atom(
    parser: &mut Parser<'_>,
    start: crate::common::source_map::SourceSpan,
) -> Expression {
    match parser.current().kind {
        TokenKind::IntegerLiteral => {
            if let TokenValue::Integer {
                value,
                suffix: _,
                base: _,
            } = parser.current().value
            {
                parser.advance();
                let span = parser.span_from(start);
                Expression::IntegerLiteral {
                    value,
                    suffix: super::ast::IntSuffix::None,
                    base: super::ast::NumericBase::Decimal,
                    span,
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(start),
                }
            }
        }
        TokenKind::CharLiteral => {
            if let TokenValue::Char(ch) = parser.current().value {
                parser.advance();
                let span = parser.span_from(start);
                // Character literals have integer value in constant expressions.
                Expression::IntegerLiteral {
                    value: ch as u128,
                    suffix: super::ast::IntSuffix::None,
                    base: super::ast::NumericBase::Decimal,
                    span,
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(start),
                }
            }
        }
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                let span = parser.span_from(start);
                // Handle function-call-like identifier: `ident(args)` for
                // compiler builtins such as `__builtin_offsetof(type, member)`.
                if parser.current().kind == TokenKind::LeftParen {
                    parser.advance(); // consume `(`
                    let mut depth = 1u32;
                    while depth > 0 && !parser.is_at_end() {
                        match parser.current().kind {
                            TokenKind::LeftParen => depth += 1,
                            TokenKind::RightParen => depth -= 1,
                            _ => {}
                        }
                        if depth > 0 {
                            parser.advance();
                        }
                    }
                    if depth == 0 {
                        parser.advance(); // consume final `)`
                    }
                    // Return a placeholder zero — we cannot evaluate builtins here.
                    Expression::IntegerLiteral {
                        value: 0,
                        suffix: super::ast::IntSuffix::None,
                        base: super::ast::NumericBase::Decimal,
                        span: parser.span_from(start),
                    }
                } else {
                    Expression::Identifier { name: id, span }
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(start),
                }
            }
        }
        TokenKind::LeftParen => {
            parser.advance(); // consume `(`

            // Check if this is a cast expression: `(type_name) expr`
            if is_type_start_token(parser, parser.current()) {
                let type_name = parse_type_name(parser);
                let _ = parser.expect(TokenKind::RightParen);
                // Parse the operand — it's the cast target.
                let cast_start = parser.current_span();
                let operand = parse_const_expr_atom(parser, cast_start);
                let span = parser.span_from(start);
                Expression::Cast {
                    type_name: Box::new(type_name),
                    operand: Box::new(operand),
                    span,
                }
            } else {
                // Regular parenthesized expression.
                let expr = parse_constant_expression_simple(parser);
                let _ = parser.expect(TokenKind::RightParen);
                let span = parser.span_from(start);
                Expression::Paren {
                    inner: Box::new(expr),
                    span,
                }
            }
        }
        // Handle sizeof expressions: `sizeof(type)` or `sizeof expr`
        TokenKind::Sizeof => {
            parser.advance(); // consume `sizeof`
            if parser.current().kind == TokenKind::LeftParen {
                // Peek: is next token a type name?
                if is_type_start_token(parser, parser.peek()) {
                    parser.advance(); // consume `(`
                    let type_name = parse_type_name(parser);
                    let _ = parser.expect(TokenKind::RightParen);
                    let span = parser.span_from(start);
                    Expression::SizeofType {
                        type_name: Box::new(type_name),
                        span,
                    }
                } else {
                    // sizeof(expr) — parse as sizeof of a parenthesized expression.
                    // Consume `(` and parse the inner expression, then `)`.
                    parser.advance(); // consume `(`
                    let inner = parse_constant_expression_simple(parser);
                    let _ = parser.expect(TokenKind::RightParen);
                    let span = parser.span_from(start);
                    Expression::SizeofExpr {
                        expr: Box::new(Expression::Paren {
                            inner: Box::new(inner),
                            span: span.clone(),
                        }),
                        span,
                    }
                }
            } else {
                // sizeof expr (without parens)
                let inner_start = parser.current_span();
                let operand = parse_const_expr_atom(parser, inner_start);
                let span = parser.span_from(start);
                Expression::SizeofExpr {
                    expr: Box::new(operand),
                    span,
                }
            }
        }
        // Handle _Alignof(type_name)
        TokenKind::Alignof => {
            parser.advance(); // consume `_Alignof`
            let _ = parser.expect(TokenKind::LeftParen);
            let type_name = parse_type_name(parser);
            let _ = parser.expect(TokenKind::RightParen);
            let span = parser.span_from(start);
            Expression::Alignof {
                type_name: Box::new(type_name),
                span,
            }
        }
        // Handle unary minus (e.g., `-1` in enum definitions)
        TokenKind::Minus => {
            parser.advance(); // consume `-`
            let inner_start = parser.current_span();
            let operand = parse_const_expr_atom(parser, inner_start);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: super::ast::UnaryOp::Negate,
                operand: Box::new(operand),
                span,
            }
        }
        // Handle unary plus
        TokenKind::Plus => {
            parser.advance(); // consume `+`
            let inner_start = parser.current_span();
            let operand = parse_const_expr_atom(parser, inner_start);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: super::ast::UnaryOp::Plus,
                operand: Box::new(operand),
                span,
            }
        }
        // Handle bitwise NOT (~)
        TokenKind::Tilde => {
            parser.advance(); // consume `~`
            let inner_start = parser.current_span();
            let operand = parse_const_expr_atom(parser, inner_start);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: super::ast::UnaryOp::BitwiseNot,
                operand: Box::new(operand),
                span,
            }
        }
        // Handle logical NOT (!)
        TokenKind::Bang => {
            parser.advance(); // consume `!`
            let inner_start = parser.current_span();
            let operand = parse_const_expr_atom(parser, inner_start);
            let span = parser.span_from(start);
            Expression::UnaryPrefix {
                op: super::ast::UnaryOp::LogicalNot,
                operand: Box::new(operand),
                span,
            }
        }
        // Handle string literals in sizeof context (e.g., `sizeof("hello")`)
        TokenKind::StringLiteral => {
            if let TokenValue::Str(s) = parser.current().value.clone() {
                parser.advance();
                let span = parser.span_from(start);
                Expression::StringLiteral {
                    value: s,
                    prefix: super::ast::StringPrefix::None,
                    span,
                }
            } else {
                parser.advance();
                Expression::Error {
                    span: parser.span_from(start),
                }
            }
        }
        _ => {
            // Unrecognized token — return error expression.
            Expression::Error {
                span: parser.span_from(start),
            }
        }
    }
}

/// Parses an `enum` specifier: tag reference or inline definition.
///
/// Grammar:
/// ```text
///   enum-specifier := 'enum' identifier? '{' enumerator-list ','? '}'
///                   | 'enum' identifier
/// ```
fn parse_enum_specifier(parser: &mut Parser<'_>) -> TypeSpecifier {
    let start = parser.current_span();

    // Consume `enum` keyword.
    parser.advance();

    // Parse optional GCC attributes.
    let _attrs = gcc_extensions::try_parse_attributes(parser);

    // Parse optional tag name.
    let tag = if parser.current().kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            parser.advance();
            Some(id)
        } else {
            None
        }
    } else {
        None
    };

    // Parse optional GCC attributes after tag name.
    let _attrs2 = gcc_extensions::try_parse_attributes(parser);

    // Check for enum definition body.
    if parser.check(TokenKind::LeftBrace) {
        parser.advance(); // consume `{`
        let mut variants = Vec::new();

        while !parser.check(TokenKind::RightBrace) && !parser.is_at_end() {
            let var_start = parser.current_span();

            // Parse enumerator name.
            let name = if let TokenValue::Identifier(id) = parser.current().value {
                if parser.current().kind == TokenKind::Identifier {
                    parser.advance();
                    id
                } else {
                    parser.error("expected enumerator name");
                    break;
                }
            } else {
                parser.error("expected enumerator name");
                break;
            };

            // Parse optional GCC attributes on the enumerator.
            let var_attrs = gcc_extensions::try_parse_attributes(parser);

            // Parse optional `= value`.
            let value = if parser.match_token(TokenKind::Equal) {
                Some(Box::new(parse_constant_expression_simple(parser)))
            } else {
                None
            };

            let var_span = parser.span_from(var_start);
            variants.push(super::ast::EnumVariant {
                name,
                value,
                attributes: var_attrs,
                span: var_span,
            });

            // Expect `,` or `}`.
            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        let _ = parser.expect(TokenKind::RightBrace);
        let span = parser.span_from(start);

        TypeSpecifier::Enum(super::ast::EnumDef {
            tag,
            variants,
            attributes: Vec::new(),
            span,
        })
    } else if let Some(tag_name) = tag {
        let span = parser.span_from(start);
        TypeSpecifier::EnumRef {
            tag: tag_name,
            span,
        }
    } else {
        parser.error("expected identifier or '{' after 'enum'");
        TypeSpecifier::Error
    }
}

// ===========================================================================
// parse_type_qualifiers — Type Qualifier Parsing
// ===========================================================================

/// Parses zero or more type qualifiers from the current token position.
///
/// Recognized qualifiers:
/// - `const` / `__const__` → `TypeQualifier::Const`
/// - `volatile` / `__volatile__` → `TypeQualifier::Volatile`
/// - `restrict` / `__restrict__` → `TypeQualifier::Restrict`
/// - `_Atomic` (without parentheses) → `TypeQualifier::Atomic`
///
/// Returns an empty vector if no qualifiers are present.
pub(super) fn parse_type_qualifiers(parser: &mut Parser<'_>) -> Vec<TypeQualifier> {
    let mut qualifiers = Vec::new();

    loop {
        match parser.current().kind {
            TokenKind::Const => {
                parser.advance();
                qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                // `_Atomic` as a qualifier (without parentheses).
                // If followed by `(`, it is a type specifier and should not
                // be consumed here.
                if parser.peek().kind == TokenKind::LeftParen {
                    break;
                }
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            // Handle GCC double-underscore variants via identifier matching.
            TokenKind::Identifier => {
                if let TokenValue::Identifier(id) = parser.current().value {
                    let name = parser.interner().resolve(id);
                    match name {
                        "__const__" | "__const" => {
                            parser.advance();
                            qualifiers.push(TypeQualifier::Const);
                        }
                        "__volatile__" | "__volatile" => {
                            parser.advance();
                            qualifiers.push(TypeQualifier::Volatile);
                        }
                        "__restrict__" | "__restrict" => {
                            parser.advance();
                            qualifiers.push(TypeQualifier::Restrict);
                        }
                        _ => break,
                    }
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    qualifiers
}

// ===========================================================================
// is_type_specifier_start — Declaration/Expression Disambiguation
// ===========================================================================

/// Returns `true` if the current token could start a type specifier or
/// declaration.
///
/// This function is the CRITICAL disambiguation point between declarations
/// and expressions in C. It determines whether a token sequence like `(foo)`
/// should be interpreted as a cast (if `foo` is a type name) or a
/// parenthesized expression.
///
/// Type-starting tokens include:
/// - Type specifier keywords: `void`, `char`, `int`, `long`, `float`, etc.
/// - Type qualifier keywords: `const`, `volatile`, `restrict`
/// - Storage class keywords: `static`, `extern`, `auto`, `register`,
///   `typedef`, `_Thread_local`
/// - Function specifiers: `inline`, `_Noreturn`
/// - C11 keywords: `_Atomic`, `_Bool`, `_Complex`
/// - GCC extensions: `typeof`, `__typeof__`, `__attribute__`, `__extension__`
/// - Known typedef names (identifiers registered via `typedef`)
pub(super) fn is_type_specifier_start(parser: &Parser<'_>) -> bool {
    let kind = parser.current().kind;

    // Type specifier keywords.
    if kind.is_type_specifier() {
        return true;
    }

    // Type qualifier keywords.
    if kind.is_type_qualifier() {
        return true;
    }

    // Storage class keywords (these start declarations).
    if kind.is_storage_class() {
        return true;
    }

    // Function specifiers.
    if matches!(kind, TokenKind::Inline | TokenKind::Noreturn) {
        return true;
    }

    // GCC __attribute__ and __extension__ can precede type specifiers.
    if matches!(kind, TokenKind::GccAttribute | TokenKind::GccExtension) {
        return true;
    }

    // __builtin_va_list is a type.
    if kind == TokenKind::BuiltinVaList {
        return true;
    }

    // Check for typedef names — identifiers that have been typedef'd.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            return parser.is_typedef_name(id);
        }
    }

    // Check for GCC double-underscore qualifier variants that might appear
    // as identifiers.
    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = parser.current().value {
            let name = parser.interner().resolve(id);
            matches!(
                name,
                "__const__"
                    | "__const"
                    | "__volatile__"
                    | "__volatile"
                    | "__restrict__"
                    | "__restrict"
                    | "__signed__"
                    | "__signed"
            )
        } else {
            false
        }
    } else {
        false
    }
}

// ===========================================================================
// parse_pointer — Pointer Modifier Parsing
// ===========================================================================

/// Parses a chain of pointer modifiers: `*`, `*const`, `*volatile *restrict`.
///
/// Each `*` can be followed by zero or more type qualifiers that apply to
/// that pointer level. Multiple `*` create multiple pointer indirections.
///
/// Returns a `Vec<Pointer>` where each entry represents one level of pointer
/// indirection with its associated qualifiers.
///
/// Example:
/// - `*const *volatile` → `[Pointer{[Const]}, Pointer{[Volatile]}]`
/// - `**` → `[Pointer{[]}, Pointer{[]}]`
pub(super) fn parse_pointer(parser: &mut Parser<'_>) -> Vec<Pointer> {
    let mut pointers = Vec::new();

    while parser.match_token(TokenKind::Star) {
        let qualifiers = parse_type_qualifiers(parser);
        // Also parse GCC attributes on pointers.
        let _attrs = gcc_extensions::try_parse_attributes(parser);
        pointers.push(Pointer { qualifiers });
    }

    pointers
}

// ===========================================================================
// parse_array_dimensions — Array Declarator Parsing
// ===========================================================================

/// Parses one or more array dimension suffixes: `[10]`, `[]`, `[*]`,
/// `[static 10]`.
///
/// Called when `[` is the current token. Parses consecutive `[...]` brackets
/// for multi-dimensional arrays.
///
/// Array dimension types:
/// - `[expr]` → `ArraySize::Fixed(expr)` — fixed-size array
/// - `[]` → `ArraySize::Unspecified` — unspecified size
/// - `[*]` → `ArraySize::VLA` — C99 variable-length array placeholder
/// - `[static expr]` → `ArraySize::Static(expr)` — C99 function parameter
pub(super) fn parse_array_dimensions(parser: &mut Parser<'_>) -> Vec<ArraySize> {
    let mut dimensions = Vec::new();

    while parser.match_token(TokenKind::LeftBracket) {
        // Parse optional type qualifiers in array dimensions (C99).
        let _quals = parse_type_qualifiers(parser);

        // Check for `static` qualifier (C99 function parameter array).
        let is_static = parser.match_token(TokenKind::Static);

        // If we had qualifiers but not static, check for static after.
        if !is_static {
            let _quals2 = parse_type_qualifiers(parser);
        }

        if parser.check(TokenKind::RightBracket) {
            // `[]` — unspecified size.
            parser.advance();
            if is_static {
                parser.error("'static' in array declarator requires a size expression");
                dimensions.push(ArraySize::Unspecified);
            } else {
                dimensions.push(ArraySize::Unspecified);
            }
        } else if parser.check(TokenKind::Star) && peek_is_rbracket(parser) {
            // `[*]` — VLA placeholder.
            parser.advance(); // consume `*`
            let _ = parser.expect(TokenKind::RightBracket);
            dimensions.push(ArraySize::VLA);
        } else {
            // `[expr]` or `[static expr]` — parse size expression.
            let size_expr = parse_constant_expression_simple(parser);

            // Handle more complex expressions: if we stopped at something other
            // than `]`, try to consume additional tokens.
            while !parser.check(TokenKind::RightBracket) && !parser.is_at_end() {
                parser.advance();
            }

            let _ = parser.expect(TokenKind::RightBracket);
            if is_static {
                dimensions.push(ArraySize::Static(Box::new(size_expr)));
            } else {
                dimensions.push(ArraySize::Fixed(Box::new(size_expr)));
            }
        }
    }

    dimensions
}

/// Returns `true` if the next token after the current position is `]`.
fn peek_is_rbracket(parser: &Parser<'_>) -> bool {
    parser.peek().kind == TokenKind::RightBracket
}

// ===========================================================================
// parse_abstract_declarator — Abstract Declarator Parsing
// ===========================================================================

/// Parses an abstract declarator (a declarator without an identifier name).
///
/// Abstract declarators are used in:
/// - Cast expressions: `(int *)expr`
/// - sizeof: `sizeof(int *)`
/// - Function parameters: `void f(int *)` (unnamed parameter)
/// - _Generic associations
/// - typeof type arguments
///
/// Grammar:
/// ```text
///   abstract-declarator := pointer
///                        | pointer? direct-abstract-declarator
/// ```
pub(super) fn parse_abstract_declarator(parser: &mut Parser<'_>) -> AbstractDeclarator {
    let start = parser.current_span();

    // Parse optional pointer prefix.
    let ptrs = parse_pointer(parser);

    // Parse optional direct abstract declarator.
    let direct = parse_direct_abstract_declarator(parser);

    let span = parser.span_from(start);
    AbstractDeclarator {
        pointer: ptrs,
        direct,
        span,
    }
}

/// Parses a direct abstract declarator (array/function suffixes and
/// parenthesized grouping).
///
/// Grammar:
/// ```text
///   direct-abstract-declarator := '(' abstract-declarator ')'
///                               | direct-abstract-declarator? '[' ... ']'
///                               | direct-abstract-declarator? '(' ... ')'
/// ```
fn parse_direct_abstract_declarator(parser: &mut Parser<'_>) -> Option<DirectAbstractDeclarator> {
    let mut result: Option<DirectAbstractDeclarator> = None;

    // Check for parenthesized abstract declarator: `(abstract-declarator)`.
    // We need to distinguish between:
    //   1. Parenthesized abstract declarator: `(*)(int)` → pointer to function
    //   2. Function parameter list: `(int, int)` → function taking (int, int)
    //
    // Heuristic: If `(` is followed by `*`, `)`, `[`, or starts with a type
    // specifier that is followed by `)` or `,`, treat differently.
    if parser.check(TokenKind::LeftParen) {
        // Peek inside the parentheses to decide:
        // - `(*` or `([` → parenthesized abstract declarator
        // - `()` → function with no parameters
        // - `(void)` or `(int,...)` → function parameter list
        let next = parser.peek().kind;

        if next == TokenKind::Star || next == TokenKind::LeftBracket || next == TokenKind::LeftParen
        {
            // Likely a parenthesized abstract declarator (e.g., `(*)(int)`).
            parser.advance(); // consume `(`
            let inner = parse_abstract_declarator(parser);
            let _ = parser.expect(TokenKind::RightParen);
            result = Some(DirectAbstractDeclarator::Parenthesized(Box::new(inner)));
        } else if next == TokenKind::RightParen {
            // `()` — function taking no parameters.
            parser.advance(); // consume `(`
            parser.advance(); // consume `)`
            let params = ParamList {
                params: Vec::new(),
                variadic: false,
                span: parser.previous_span(),
            };
            result = Some(DirectAbstractDeclarator::Function { base: None, params });
        } else {
            // Could be a function parameter list: `(int, float)`.
            // Check if the first token is a type specifier or qualifier.
            if is_type_specifier_start_at(parser, 1) || next == TokenKind::Ellipsis {
                let params = parse_parameter_type_list(parser);
                result = Some(DirectAbstractDeclarator::Function { base: None, params });
            }
            // Otherwise, we don't have a direct abstract declarator starting
            // with `(`, so leave result as None and let the caller handle it.
        }
    }

    // Parse subsequent array and function suffixes.
    loop {
        if parser.check(TokenKind::LeftBracket) {
            let dims = parse_array_dimensions(parser);
            for dim in dims {
                let quals = Vec::new();
                result = Some(DirectAbstractDeclarator::Array {
                    base: result.map(Box::new),
                    size: dim,
                    qualifiers: quals,
                });
            }
        } else if parser.check(TokenKind::LeftParen) {
            // Function suffix: `(param-type-list)`.
            // Need to distinguish from a parenthesized declarator.
            let next = parser.peek().kind;

            // If next token is `)`, `...`, or a type specifier, parse as params.
            if next == TokenKind::RightParen
                || next == TokenKind::Ellipsis
                || is_type_specifier_start_at(parser, 1)
            {
                let params = parse_parameter_type_list(parser);
                result = Some(DirectAbstractDeclarator::Function {
                    base: result.map(Box::new),
                    params,
                });
            } else {
                break;
            }
        } else {
            break;
        }
    }

    result
}

/// Checks if the token at offset `offset` from the current position is a
/// type specifier start (for lookahead-based disambiguation).
fn is_type_specifier_start_at(parser: &Parser<'_>, offset: usize) -> bool {
    let token = parser.lookahead(offset);
    let kind = token.kind;

    if kind.is_type_specifier() || kind.is_type_qualifier() || kind.is_storage_class() {
        return true;
    }

    if matches!(kind, TokenKind::Inline | TokenKind::Noreturn) {
        return true;
    }

    if kind == TokenKind::Identifier {
        if let TokenValue::Identifier(id) = token.value {
            return parser.is_typedef_name(id);
        }
    }

    false
}

/// Parses a function parameter type list: `(param-type-list)` or `()`.
///
/// Grammar:
/// ```text
///   parameter-type-list := parameter-list (',' '...')?
///   parameter-list := parameter-declaration (',' parameter-declaration)*
///   parameter-declaration := declaration-specifiers declarator?
///                          | declaration-specifiers abstract-declarator?
/// ```
fn parse_parameter_type_list(parser: &mut Parser<'_>) -> ParamList {
    let start = parser.current_span();

    // Consume `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        return ParamList {
            params: Vec::new(),
            variadic: false,
            span: parser.span_from(start),
        };
    }

    // Check for empty parameter list `()`.
    if parser.match_token(TokenKind::RightParen) {
        return ParamList {
            params: Vec::new(),
            variadic: false,
            span: parser.span_from(start),
        };
    }

    let mut params = Vec::new();
    let mut variadic = false;

    // Check for `(void)` — single void parameter means no parameters.
    if parser.check(TokenKind::Void) && parser.peek().kind == TokenKind::RightParen {
        parser.advance(); // consume `void`
        parser.advance(); // consume `)`
        return ParamList {
            params: Vec::new(),
            variadic: false,
            span: parser.span_from(start),
        };
    }

    loop {
        // Check for `...` (variadic).
        if parser.match_token(TokenKind::Ellipsis) {
            variadic = true;
            break;
        }

        let param_start = parser.current_span();

        // Parse parameter declaration specifiers.
        let mut qualifiers = parse_type_qualifiers(parser);
        let mut storage = None;
        let func_specs = Vec::new();

        // Parse optional storage class (only `register` is allowed in params).
        if parser.check(TokenKind::Register) {
            storage = Some(StorageClass::Register);
            parser.advance();
        }

        let type_spec = parse_type_specifiers(parser);
        let mut more_quals = parse_type_qualifiers(parser);
        qualifiers.append(&mut more_quals);

        let spec = DeclSpecifiers {
            storage_class: storage,
            type_qualifiers: qualifiers,
            type_specifier: type_spec,
            function_specifiers: func_specs,
            attributes: Vec::new(),
            alignment: None,
            span: parser.span_from(param_start),
        };

        // Parse optional declarator or abstract declarator.
        let declarator = parse_param_declarator(parser);

        let param_span = parser.span_from(param_start);
        params.push(ParamDeclaration {
            specifiers: spec,
            declarator,
            span: param_span,
        });

        // Expect `,` or `)`.
        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    let _ = parser.expect(TokenKind::RightParen);

    let span = parser.span_from(start);
    ParamList {
        params,
        variadic,
        span,
    }
}

/// Parses an optional declarator for a function parameter.
///
/// A parameter can have:
/// - A named declarator: `int x`
/// - An abstract declarator: `int *`
/// - No declarator at all: `int`
fn parse_param_declarator(parser: &mut Parser<'_>) -> Option<super::ast::Declarator> {
    // If the current token cannot start a declarator, return None.
    if !matches!(
        parser.current().kind,
        TokenKind::Star | TokenKind::Identifier | TokenKind::LeftParen | TokenKind::LeftBracket
    ) {
        return None;
    }

    let start = parser.current_span();

    // Parse pointer prefix.
    let ptrs = parse_pointer(parser);

    // Parse identifier or parenthesized declarator.
    let direct = if parser.current().kind == TokenKind::Identifier {
        // Check if this is a typedef name or a parameter name.
        // In parameter context, if we already have a type spec, an identifier
        // that is NOT a typedef name is the parameter name.
        if let TokenValue::Identifier(id) = parser.current().value {
            if parser.is_typedef_name(id) && ptrs.is_empty() {
                // This is a typedef name being used as a parameter name only
                // if we already have specifiers. Since we can't easily tell here,
                // treat non-pointer typedef names as part of the type, not the
                // parameter name. The caller should handle this.
                return None;
            }
            parser.advance();
            super::ast::DirectDeclarator::Identifier(id)
        } else {
            return None;
        }
    } else if parser.match_token(TokenKind::LeftParen) {
        // Parenthesized declarator.
        let inner = if let Some(decl) = parse_param_declarator(parser) {
            decl
        } else {
            // Empty parens — not a valid declarator.
            let _ = parser.expect(TokenKind::RightParen);
            return Some(super::ast::Declarator {
                pointer: ptrs,
                direct: super::ast::DirectDeclarator::Abstract,
                attributes: Vec::new(),
                span: parser.span_from(start),
            });
        };
        let _ = parser.expect(TokenKind::RightParen);
        super::ast::DirectDeclarator::Parenthesized(Box::new(inner))
    } else if parser.check(TokenKind::LeftBracket) {
        // Array declarator without name.
        super::ast::DirectDeclarator::Abstract
    } else if !ptrs.is_empty() {
        // Only pointers, no direct declarator name.
        super::ast::DirectDeclarator::Abstract
    } else {
        return None;
    };

    // Parse array and function suffixes.
    let direct = parse_direct_declarator_suffixes(parser, direct);

    let span = parser.span_from(start);
    Some(super::ast::Declarator {
        pointer: ptrs,
        direct,
        attributes: gcc_extensions::try_parse_attributes(parser),
        span,
    })
}

// ===========================================================================
// is_typedef_name — Typedef Name Checking
// ===========================================================================

/// Returns `true` if the given identifier name is a currently-visible typedef
/// name.
///
/// This is a convenience wrapper around `Parser::is_typedef_name()` that
/// accepts a string slice and looks it up in the interner first. Returns
/// `false` if the name has not been interned (i.e., it was never seen by the
/// lexer).
pub(super) fn is_typedef_name(parser: &Parser<'_>, name: &str) -> bool {
    if let Some(id) = parser.interner().get(name) {
        parser.is_typedef_name(id)
    } else {
        false
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
        Token::new(
            TokenKind::Identifier,
            dummy_span(),
            TokenValue::Identifier(id),
        )
    }

    /// Creates an integer literal token with the given value.
    fn make_int_token(value: u128) -> Token {
        Token::new(
            TokenKind::IntegerLiteral,
            dummy_span(),
            TokenValue::Integer {
                value,
                suffix: crate::frontend::lexer::token::IntSuffix::None,
                base: crate::frontend::lexer::token::NumericBase::Decimal,
            },
        )
    }

    /// Creates a token stream from a list of tokens plus an EOF sentinel.
    fn make_token_stream(tokens: Vec<Token>) -> Vec<Token> {
        let mut stream = tokens;
        stream.push(make_token(TokenKind::Eof));
        stream
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
    // Basic Type Specifier Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_void() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Void)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Void));
    }

    #[test]
    fn test_parse_char() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Char)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Char));
    }

    #[test]
    fn test_parse_int() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Int)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Int));
    }

    #[test]
    fn test_parse_short() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Short)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Short));
    }

    #[test]
    fn test_parse_long() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Long)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Long));
    }

    #[test]
    fn test_parse_long_long() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Long),
            make_token(TokenKind::Long),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::LongLong));
    }

    #[test]
    fn test_parse_long_long_int() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Long),
            make_token(TokenKind::Long),
            make_token(TokenKind::Int),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::LongLong));
    }

    #[test]
    fn test_parse_float() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Float)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Float));
    }

    #[test]
    fn test_parse_double() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Double)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Double));
    }

    #[test]
    fn test_parse_long_double() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Long),
            make_token(TokenKind::Double),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::LongDouble));
    }

    // -----------------------------------------------------------------------
    // Signed/Unsigned Modifier Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_unsigned_int() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Unsigned),
            make_token(TokenKind::Int),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::Unsigned(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Int));
            }
            _ => panic!("expected Unsigned(Int), got {:?}", result),
        }
    }

    #[test]
    fn test_parse_signed_char() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Signed),
            make_token(TokenKind::Char),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::Signed(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Char));
            }
            _ => panic!("expected Signed(Char), got {:?}", result),
        }
    }

    #[test]
    fn test_parse_unsigned_short() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Unsigned),
            make_token(TokenKind::Short),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::Unsigned(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Short));
            }
            _ => panic!("expected Unsigned(Short), got {:?}", result),
        }
    }

    #[test]
    fn test_parse_unsigned_long_long_int() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Unsigned),
            make_token(TokenKind::Long),
            make_token(TokenKind::Long),
            make_token(TokenKind::Int),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::Unsigned(inner) => {
                assert!(matches!(*inner, TypeSpecifier::LongLong));
            }
            _ => panic!("expected Unsigned(LongLong), got {:?}", result),
        }
    }

    #[test]
    fn test_parse_signed_alone() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Signed)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        // `signed` alone implies `signed int`.
        match result {
            TypeSpecifier::Signed(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Int));
            }
            _ => panic!("expected Signed(Int), got {:?}", result),
        }
    }

    #[test]
    fn test_parse_unsigned_alone() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Unsigned)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        // `unsigned` alone implies `unsigned int`.
        match result {
            TypeSpecifier::Unsigned(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Int));
            }
            _ => panic!("expected Unsigned(Int), got {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // C11 Special Type Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_bool() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Bool)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        assert!(matches!(result, TypeSpecifier::Bool));
    }

    #[test]
    fn test_parse_complex_double() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Complex),
            make_token(TokenKind::Double),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::Complex(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Double));
            }
            _ => panic!("expected Complex(Double), got {:?}", result),
        }
    }

    #[test]
    fn test_parse_atomic_int() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Atomic),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Int),
            make_token(TokenKind::RightParen),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::Atomic(inner) => {
                assert!(matches!(*inner, TypeSpecifier::Int));
            }
            _ => panic!("expected Atomic(Int), got {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // Type Qualifier Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_const_qualifier() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Const)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_qualifiers(&mut parser);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], TypeQualifier::Const);
    }

    #[test]
    fn test_parse_volatile_qualifier() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Volatile)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_qualifiers(&mut parser);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], TypeQualifier::Volatile);
    }

    #[test]
    fn test_parse_restrict_qualifier() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Restrict)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_qualifiers(&mut parser);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], TypeQualifier::Restrict);
    }

    #[test]
    fn test_parse_multiple_qualifiers() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Const),
            make_token(TokenKind::Volatile),
            make_token(TokenKind::Restrict),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_qualifiers(&mut parser);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], TypeQualifier::Const);
        assert_eq!(result[1], TypeQualifier::Volatile);
        assert_eq!(result[2], TypeQualifier::Restrict);
    }

    #[test]
    fn test_no_qualifiers() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Int)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_qualifiers(&mut parser);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Pointer Parsing Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_single_pointer() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Star)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_pointer(&mut parser);
        assert_eq!(result.len(), 1);
        assert!(result[0].qualifiers.is_empty());
    }

    #[test]
    fn test_parse_const_pointer() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Star),
            make_token(TokenKind::Const),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_pointer(&mut parser);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].qualifiers.len(), 1);
        assert_eq!(result[0].qualifiers[0], TypeQualifier::Const);
    }

    #[test]
    fn test_parse_double_pointer() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Star),
            make_token(TokenKind::Star),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_pointer(&mut parser);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_pointer_const_pointer_volatile() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Star),
            make_token(TokenKind::Const),
            make_token(TokenKind::Star),
            make_token(TokenKind::Volatile),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_pointer(&mut parser);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].qualifiers, vec![TypeQualifier::Const]);
        assert_eq!(result[1].qualifiers, vec![TypeQualifier::Volatile]);
    }

    // -----------------------------------------------------------------------
    // Array Dimension Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_fixed_array() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::LeftBracket),
            make_int_token(10),
            make_token(TokenKind::RightBracket),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_array_dimensions(&mut parser);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], ArraySize::Fixed(_)));
    }

    #[test]
    fn test_parse_unspecified_array() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::LeftBracket),
            make_token(TokenKind::RightBracket),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_array_dimensions(&mut parser);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], ArraySize::Unspecified));
    }

    #[test]
    fn test_parse_vla_array() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::LeftBracket),
            make_token(TokenKind::Star),
            make_token(TokenKind::RightBracket),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_array_dimensions(&mut parser);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], ArraySize::VLA));
    }

    #[test]
    fn test_parse_multi_dimensional_array() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::LeftBracket),
            make_int_token(10),
            make_token(TokenKind::RightBracket),
            make_token(TokenKind::LeftBracket),
            make_int_token(20),
            make_token(TokenKind::RightBracket),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_array_dimensions(&mut parser);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0], ArraySize::Fixed(_)));
        assert!(matches!(result[1], ArraySize::Fixed(_)));
    }

    // -----------------------------------------------------------------------
    // Type Specifier Start Detection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_type_specifier_start_int() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Int)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(is_type_specifier_start(&parser));
    }

    #[test]
    fn test_is_type_specifier_start_void() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Void)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(is_type_specifier_start(&parser));
    }

    #[test]
    fn test_is_type_specifier_start_struct() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Struct)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(is_type_specifier_start(&parser));
    }

    #[test]
    fn test_is_type_specifier_start_const() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Const)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(is_type_specifier_start(&parser));
    }

    #[test]
    fn test_is_type_specifier_start_storage_class() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Static)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(is_type_specifier_start(&parser));
    }

    #[test]
    fn test_is_not_type_specifier_start_identifier() {
        let mut interner = Interner::new();
        let foo_id = interner.intern("foo");
        let tokens = make_token_stream(vec![make_ident_token(foo_id)]);
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        // `foo` is not a typedef name, so it should not be a type start.
        assert!(!is_type_specifier_start(&parser));
    }

    #[test]
    fn test_is_type_specifier_start_typedef_name() {
        let mut interner = Interner::new();
        let myint_id = interner.intern("MyInt");
        let tokens = make_token_stream(vec![make_ident_token(myint_id)]);
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        // Register "MyInt" as a typedef name.
        parser.register_typedef(myint_id);

        assert!(is_type_specifier_start(&parser));
    }

    // -----------------------------------------------------------------------
    // Typedef Name Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_typedef_name_registered() {
        let mut interner = Interner::new();
        let myint_id = interner.intern("MyInt");
        let tokens = make_token_stream(vec![make_token(TokenKind::Eof)]);
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.register_typedef(myint_id);
        assert!(is_typedef_name(&parser, "MyInt"));
    }

    #[test]
    fn test_is_not_typedef_name() {
        let mut interner = Interner::new();
        let _foo_id = interner.intern("foo");
        let tokens = make_token_stream(vec![make_token(TokenKind::Eof)]);
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        assert!(!is_typedef_name(&parser, "foo"));
    }

    #[test]
    fn test_is_typedef_name_unknown() {
        let interner = Interner::new();
        let tokens = make_token_stream(vec![make_token(TokenKind::Eof)]);
        let mut diag = DiagnosticEmitter::new();
        let parser = make_parser(&tokens, &interner, &mut diag);

        // Name not even interned — should return false.
        assert!(!is_typedef_name(&parser, "UnknownType"));
    }

    // -----------------------------------------------------------------------
    // Struct/Union/Enum Reference Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_struct_ref() {
        let mut interner = Interner::new();
        let tag_id = interner.intern("MyStruct");
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Struct),
            make_ident_token(tag_id),
        ]);
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::StructRef { tag, .. } => {
                assert_eq!(tag, tag_id);
            }
            _ => panic!("expected StructRef, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_union_ref() {
        let mut interner = Interner::new();
        let tag_id = interner.intern("MyUnion");
        let tokens =
            make_token_stream(vec![make_token(TokenKind::Union), make_ident_token(tag_id)]);
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::UnionRef { tag, .. } => {
                assert_eq!(tag, tag_id);
            }
            _ => panic!("expected UnionRef, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_enum_ref() {
        let mut interner = Interner::new();
        let tag_id = interner.intern("Color");
        let tokens = make_token_stream(vec![make_token(TokenKind::Enum), make_ident_token(tag_id)]);
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::EnumRef { tag, .. } => {
                assert_eq!(tag, tag_id);
            }
            _ => panic!("expected EnumRef, got {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // Error Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_int_float() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Int),
            make_token(TokenKind::Float),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let _result = parse_type_specifiers(&mut parser);
        // The second specifier `float` should produce an error.
        // The parser should have emitted a diagnostic.
        assert!(parser.has_errors());
    }

    #[test]
    fn test_error_short_double() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Short),
            make_token(TokenKind::Double),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let _result = parse_type_specifiers(&mut parser);
        assert!(parser.has_errors());
    }

    #[test]
    fn test_error_long_long_long() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Long),
            make_token(TokenKind::Long),
            make_token(TokenKind::Long),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let _result = parse_type_specifiers(&mut parser);
        assert!(parser.has_errors());
    }

    // -----------------------------------------------------------------------
    // Abstract Declarator Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_pointer_abstract_declarator() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Star)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_abstract_declarator(&mut parser);
        assert_eq!(result.pointer.len(), 1);
        assert!(result.direct.is_none());
    }

    #[test]
    fn test_parse_no_pointers_no_abstract() {
        let tokens = make_token_stream(vec![make_token(TokenKind::RightParen)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_abstract_declarator(&mut parser);
        assert!(result.pointer.is_empty());
    }

    // -----------------------------------------------------------------------
    // Type Name Parsing Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_type_name_int() {
        let tokens = make_token_stream(vec![make_token(TokenKind::Int)]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_name(&mut parser);
        assert!(matches!(
            result.specifiers.type_specifier,
            TypeSpecifier::Int
        ));
        assert!(result.abstract_declarator.is_none());
    }

    #[test]
    fn test_parse_type_name_int_pointer() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Int),
            make_token(TokenKind::Star),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_name(&mut parser);
        assert!(matches!(
            result.specifiers.type_specifier,
            TypeSpecifier::Int
        ));
        assert!(result.abstract_declarator.is_some());
        let decl = result.abstract_declarator.unwrap();
        assert_eq!(decl.pointer.len(), 1);
    }

    #[test]
    fn test_parse_type_name_const_int() {
        let tokens = make_token_stream(vec![
            make_token(TokenKind::Const),
            make_token(TokenKind::Int),
        ]);
        let interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        let result = parse_type_name(&mut parser);
        assert!(matches!(
            result.specifiers.type_specifier,
            TypeSpecifier::Int
        ));
        assert_eq!(result.specifiers.type_qualifiers.len(), 1);
        assert_eq!(result.specifiers.type_qualifiers[0], TypeQualifier::Const);
    }

    // -----------------------------------------------------------------------
    // Typedef Name as Type Specifier Test
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_typedef_name_specifier() {
        let mut interner = Interner::new();
        let myint_id = interner.intern("MyInt");
        let tokens = make_token_stream(vec![make_ident_token(myint_id)]);
        let mut diag = DiagnosticEmitter::new();
        let mut parser = make_parser(&tokens, &interner, &mut diag);

        parser.register_typedef(myint_id);

        let result = parse_type_specifiers(&mut parser);
        match result {
            TypeSpecifier::TypedefName { name, .. } => {
                assert_eq!(name, myint_id);
            }
            _ => panic!("expected TypedefName, got {:?}", result),
        }
    }
}
