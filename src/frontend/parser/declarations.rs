//! Declaration parsing for the `bcc` C compiler.
//!
//! This module implements a complete C11 declaration parser with GCC extension
//! support. It handles all declaration forms required for compiling real-world
//! C codebases such as SQLite, Lua, zlib, and Redis.
//!
//! # Declaration Forms Supported
//!
//! - **Variable declarations** — `int x;`, `int x = 5;`, `int x, y, z;`
//! - **Function declarations** — `int foo(int x, int y);`
//! - **Function definitions** — `int main(void) { return 0; }`
//! - **Typedef declarations** — `typedef int MyInt;`, `typedef struct { int x; } Point;`
//! - **Struct definitions** — `struct S { int x; float y; };` with bitfields,
//!   anonymous members, flexible array members
//! - **Union definitions** — `union U { int i; float f; };`
//! - **Enum definitions** — `enum Color { RED, GREEN, BLUE };` with optional values
//!   and trailing commas (C99)
//! - **`_Static_assert`** — `_Static_assert(sizeof(int) == 4, "msg");`
//! - **Forward declarations** — `struct S;`, `union U;`, `enum E;`
//! - **Designated initializers** — `{ .x = 1, [0] = 2, [1 ... 3] = 5 }` (C99+GCC)
//! - **GCC `__attribute__`** — on structs, unions, enums, functions, variables
//!
//! # Declarator Complexity
//!
//! C's "declaration reflects use" design makes declarator parsing the most
//! complex part of the frontend. This module handles:
//! - Pointer chains: `int ***p;`
//! - Qualified pointers: `int *const *volatile p;`
//! - Function pointers: `int (*fp)(int, int);`
//! - Array of function pointers: `int (*fps[10])(void);`
//! - Nested complex: `int (*(*fp)(int))[10];` — pointer to function returning
//!   pointer to array of 10 ints
//!
//! # Error Recovery
//!
//! On syntax errors, the parser synchronizes to `;` or `}` boundaries to
//! maintain the ≥80% error recovery rate. Error nodes
//! (`Declaration::Empty`, `TypeSpecifier::Error`, `Expression::Error`) are
//! inserted to keep the AST well-formed.
//!
//! # Integration Points
//!
//! - Called by `mod.rs` (`Parser::parse_translation_unit()`) for top-level
//!   declarations.
//! - Called by `statements.rs` for block-level declarations and for-init.
//! - Calls `types.rs` for type specifier and qualifier parsing.
//! - Calls `expressions.rs` for initializer and constant expressions.
//! - Calls `statements.rs` for function definition bodies.
//! - Calls `gcc_extensions.rs` for `__attribute__` parsing.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use super::ast::{
    DeclSpecifiers, Declaration, Declarator, Designator,
    DesignatedInitializer, DirectDeclarator, EnumDef, EnumVariant,
    FunctionDef, FunctionSpecifier, GccAttribute, InitDeclarator, Initializer,
    ParamDeclaration, ParamList, StorageClass, StructDef,
    StructFieldDeclarator, StructMember, TypeQualifier, TypeSpecifier,
    UnionDef,
};
use super::expressions;
use super::gcc_extensions;
use super::statements;
use super::types;
use super::Parser;
use crate::common::intern::InternId;
use crate::common::source_map::SourceSpan;
use crate::frontend::lexer::token::{TokenKind, TokenValue};

// ===========================================================================
// parse_external_declaration — Top-Level Entry Point
// ===========================================================================

/// Parses a single top-level (external) declaration.
///
/// This is the main entry point called by `Parser::parse_translation_unit()`
/// in a loop until EOF. It distinguishes between:
///
/// - `_Static_assert` declarations
/// - GCC `__extension__`-prefixed declarations
/// - Struct/union/enum-only declarations (e.g., `struct S { ... };`)
/// - Typedef declarations
/// - Function definitions (declarator followed by `{`)
/// - Variable declarations (one or more declarators followed by `;`)
///
/// # Parsing Strategy
///
/// 1. Parse declaration specifiers (storage, qualifiers, type, function specs)
/// 2. If only specifiers + `;`, return as empty/type-only declaration
/// 3. Parse first declarator
/// 4. If function declarator + `{`, parse as function definition
/// 5. Otherwise, parse remaining declarators (comma-separated) + `;`
pub(super) fn parse_external_declaration(parser: &mut Parser<'_>) -> Declaration {
    let start = parser.current_span();

    // Handle _Static_assert at the top level.
    if parser.check(TokenKind::StaticAssert) {
        return parse_static_assert(parser);
    }

    // Handle GCC __extension__ prefix — just skip it and parse the declaration.
    if parser.check(TokenKind::GccExtension) {
        parser.advance();
        return parse_external_declaration(parser);
    }

    // Handle empty declaration (stray semicolons).
    if parser.check(TokenKind::Semicolon) {
        let span = parser.current_span();
        parser.advance();
        return Declaration::Empty { span };
    }

    // Parse declaration specifiers — returns specifiers and whether `typedef`
    // was present (typedef is not a storage class in our AST, it determines
    // the Declaration variant).
    let (specifiers, is_typedef) = parse_declaration_specifiers(parser);

    // Check if this is a bare struct/union/enum definition followed by `;`.
    // e.g., `struct S { int x; };` or `enum E { A, B };`
    if parser.check(TokenKind::Semicolon) {
        let span = parser.span_from(start);
        parser.advance(); // consume `;`
        // Return as a variable declaration with no declarators — this is how
        // standalone struct/union/enum definitions and forward declarations
        // are represented.
        return Declaration::Variable {
            specifiers,
            declarators: Vec::new(),
            span,
        };
    }

    // Check for EOF after specifiers (error case).
    if parser.is_at_end() {
        parser.error("expected declarator or ';' after declaration specifiers");
        let span = parser.span_from(start);
        return Declaration::Empty { span };
    }

    // Parse the first declarator.
    let first_declarator = parse_declarator(parser);

    // Check for GCC attributes after declarator.
    let post_attrs = gcc_extensions::try_parse_attributes(parser);

    // Skip optional GCC asm label: `__asm__("symbol_name")` used by glibc
    // for assembly-level symbol renaming (e.g., __REDIRECT macro).
    skip_asm_label(parser);

    // Check for GCC attributes that may appear after asm label.
    let _post_asm_attrs = gcc_extensions::try_parse_attributes(parser);

    // Check for function definition: declarator followed by `{`.
    // This only applies to non-typedef declarations.
    if !is_typedef && parser.check(TokenKind::LeftBrace) {
        return parse_function_definition(parser, specifiers, first_declarator, post_attrs, start);
    }

    // Check for old-style K&R parameter declarations before `{`.
    // K&R style: `int foo(a, b) int a; float b; { ... }`
    if !is_typedef
        && is_kr_style_definition(parser, &first_declarator)
    {
        return parse_kr_function_definition(parser, specifiers, first_declarator, post_attrs, start);
    }

    // Not a function definition — parse as variable/typedef declaration.
    if is_typedef {
        parse_typedef_declaration(parser, specifiers, first_declarator, start)
    } else {
        parse_variable_declaration(parser, specifiers, first_declarator, post_attrs, start)
    }
}

// ===========================================================================
// parse_declaration_specifiers — Storage, Qualifiers, Type, Function Specs
// ===========================================================================

/// Parses a sequence of declaration specifiers that can appear in any order.
///
/// Declaration specifiers include:
/// - **Storage class specifiers**: `static`, `extern`, `auto`, `register`,
///   `typedef`, `_Thread_local`
/// - **Type qualifiers**: `const`, `volatile`, `restrict`, `_Atomic`
/// - **Type specifiers**: base type keywords, struct/union/enum, typedef names
/// - **Function specifiers**: `inline`, `_Noreturn`
/// - **GCC attributes**: `__attribute__((...))` annotations
///
/// In C, these can appear in any order: `static const int` and
/// `int static const` are equivalent. This function collects all specifiers
/// in a single pass, validating for conflicts (e.g., multiple storage classes).
pub(super) fn parse_declaration_specifiers(parser: &mut Parser<'_>) -> (DeclSpecifiers, bool) {
    let start = parser.current_span();

    let mut storage_class: Option<StorageClass> = None;
    let mut type_qualifiers: Vec<TypeQualifier> = Vec::new();
    let mut function_specifiers: Vec<FunctionSpecifier> = Vec::new();
    let mut is_typedef = false;
    let mut attributes: Vec<GccAttribute> = Vec::new();
    let mut _has_type_spec = false;

    // Collect leading attributes, storage class, qualifiers, and function
    // specifiers that appear BEFORE the type specifier.
    loop {
        match parser.current().kind {
            // --- Storage class specifiers ---
            TokenKind::Typedef => {
                if is_typedef {
                    parser.error("duplicate 'typedef' specifier in declaration");
                }
                if storage_class.is_some() {
                    parser.error("'typedef' cannot be combined with another storage class specifier");
                }
                is_typedef = true;
                parser.advance();
            }
            TokenKind::Static => {
                if storage_class.is_some() {
                    parser.error("multiple storage class specifiers in declaration");
                }
                storage_class = Some(StorageClass::Static);
                parser.advance();
            }
            TokenKind::Extern => {
                if storage_class.is_some() {
                    parser.error("multiple storage class specifiers in declaration");
                }
                storage_class = Some(StorageClass::Extern);
                parser.advance();
            }
            TokenKind::Auto => {
                if storage_class.is_some() {
                    parser.error("multiple storage class specifiers in declaration");
                }
                storage_class = Some(StorageClass::Auto);
                parser.advance();
            }
            TokenKind::Register => {
                if storage_class.is_some() {
                    parser.error("multiple storage class specifiers in declaration");
                }
                storage_class = Some(StorageClass::Register);
                parser.advance();
            }
            TokenKind::ThreadLocal => {
                // _Thread_local can combine with static or extern,
                // but for simplicity we store it as the primary storage class.
                if storage_class.is_some()
                    && !matches!(
                        storage_class,
                        Some(StorageClass::Static) | Some(StorageClass::Extern)
                    )
                {
                    parser.error(
                        "'_Thread_local' can only be combined with 'static' or 'extern'",
                    );
                }
                storage_class = Some(StorageClass::ThreadLocal);
                parser.advance();
            }

            // --- Type qualifiers ---
            TokenKind::Const => {
                type_qualifiers.push(TypeQualifier::Const);
                parser.advance();
            }
            TokenKind::Volatile => {
                type_qualifiers.push(TypeQualifier::Volatile);
                parser.advance();
            }
            TokenKind::Restrict => {
                type_qualifiers.push(TypeQualifier::Restrict);
                parser.advance();
            }
            TokenKind::Atomic => {
                // _Atomic as qualifier (without parens). If followed by `(`
                // it's a type specifier and will be handled by type parsing.
                if parser.peek().kind == TokenKind::LeftParen {
                    break; // Let type specifier parsing handle it.
                }
                type_qualifiers.push(TypeQualifier::Atomic);
                parser.advance();
            }

            // --- Function specifiers ---
            TokenKind::Inline => {
                function_specifiers.push(FunctionSpecifier::Inline);
                parser.advance();
            }
            TokenKind::Noreturn => {
                function_specifiers.push(FunctionSpecifier::Noreturn);
                parser.advance();
            }

            // --- GCC attributes ---
            TokenKind::GccAttribute => {
                let mut new_attrs = gcc_extensions::try_parse_attributes(parser);
                attributes.append(&mut new_attrs);
            }

            // --- GCC __extension__ ---
            TokenKind::GccExtension => {
                parser.advance();
                // Continue parsing specifiers after __extension__.
            }

            // --- Anything else that starts a type specifier ---
            _ => break,
        }
    }

    // Parse the base type specifier using the type specifier state machine.
    let type_specifier = types::parse_type_specifiers(parser);
    _has_type_spec = !matches!(type_specifier, TypeSpecifier::Error);

    // Collect trailing qualifiers, function specifiers, and attributes that
    // may appear after the type specifier (any order is valid in C).
    loop {
        match parser.current().kind {
            TokenKind::Const => {
                type_qualifiers.push(TypeQualifier::Const);
                parser.advance();
            }
            TokenKind::Volatile => {
                type_qualifiers.push(TypeQualifier::Volatile);
                parser.advance();
            }
            TokenKind::Restrict => {
                type_qualifiers.push(TypeQualifier::Restrict);
                parser.advance();
            }
            TokenKind::Atomic if parser.peek().kind != TokenKind::LeftParen => {
                type_qualifiers.push(TypeQualifier::Atomic);
                parser.advance();
            }
            TokenKind::Inline => {
                function_specifiers.push(FunctionSpecifier::Inline);
                parser.advance();
            }
            TokenKind::Noreturn => {
                function_specifiers.push(FunctionSpecifier::Noreturn);
                parser.advance();
            }
            TokenKind::GccAttribute => {
                let mut new_attrs = gcc_extensions::try_parse_attributes(parser);
                attributes.append(&mut new_attrs);
            }
            _ => break,
        }
    }

    let span = parser.span_from(start);
    (
        DeclSpecifiers {
            storage_class,
            type_qualifiers,
            type_specifier,
            function_specifiers,
            attributes,
            span,
        },
        is_typedef,
    )
}

// ===========================================================================
// parse_declarator — C Declarator (Name + Pointer/Array/Function Modifiers)
// ===========================================================================

/// Parses a C declarator: optional pointer prefix followed by a direct
/// declarator.
///
/// A declarator names the entity being declared and includes pointer, array,
/// and function-pointer modifiers. Examples:
/// - `x` — simple identifier
/// - `*p` — pointer to something
/// - `*const p` — const pointer to something
/// - `(*fp)(int, int)` — pointer to function
/// - `arr[10]` — array of 10 elements
/// - `(*arr)[10]` — pointer to array of 10 elements
///
/// Grammar:
/// ```text
///   declarator := pointer? direct-declarator
///   pointer    := '*' type-qualifier-list? pointer?
/// ```
pub(super) fn parse_declarator(parser: &mut Parser<'_>) -> Declarator {
    let start = parser.current_span();

    // Parse pointer prefix chain: `*`, `*const`, `**`, etc.
    let pointer = types::parse_pointer(parser);

    // Parse the direct declarator (identifier, parenthesized, with suffixes).
    let direct = parse_direct_declarator(parser);

    // Parse optional GCC attributes on the declarator.
    let decl_attrs = gcc_extensions::try_parse_attributes(parser);

    let span = parser.span_from(start);
    Declarator {
        pointer,
        direct,
        attributes: decl_attrs,
        span,
    }
}

/// Parses the direct part of a declarator: identifier, parenthesized
/// declarator, and array/function suffixes.
///
/// Grammar:
/// ```text
///   direct-declarator := IDENTIFIER
///                       | '(' declarator ')'
///                       | direct-declarator '[' ... ']'
///                       | direct-declarator '(' parameter-type-list ')'
///                       | direct-declarator '(' identifier-list? ')'
/// ```
///
/// The suffixes (array `[...]` and function `(...)`) are parsed in a loop
/// to handle chained suffixes like `f(int x)[10]`.
fn parse_direct_declarator(parser: &mut Parser<'_>) -> DirectDeclarator {
    // Parse the base of the direct declarator.
    let mut base = match parser.current().kind {
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                DirectDeclarator::Identifier(id)
            } else {
                parser.error("expected identifier in declarator");
                DirectDeclarator::Abstract
            }
        }
        TokenKind::LeftParen => {
            // Parenthesized declarator — e.g., `(*fp)(int)`.
            // We need to distinguish between:
            //   (1) Parenthesized declarator: `(*fp)` followed by `(params)`
            //   (2) Function parameter list: shouldn't happen here since
            //       direct declarator requires an identifier or `(`
            parser.advance(); // consume `(`
            let inner = parse_declarator(parser);
            let _ = parser.expect(TokenKind::RightParen);
            DirectDeclarator::Parenthesized(Box::new(inner))
        }
        _ => {
            // No identifier — this might be an abstract declarator context
            // or an error. Return Abstract and let the caller decide.
            DirectDeclarator::Abstract
        }
    };

    // Parse array and function suffixes in a loop.
    loop {
        if parser.check(TokenKind::LeftBracket) {
            // Array suffix: `[expr]`, `[]`, `[*]`, `[static expr]`.
            let dims = types::parse_array_dimensions(parser);
            for dim in dims {
                base = DirectDeclarator::Array {
                    base: Box::new(base),
                    size: dim,
                    qualifiers: Vec::new(),
                };
            }
        } else if parser.check(TokenKind::LeftParen) {
            // Function parameter list suffix: `(params)`.
            let params = parse_parameter_list(parser);
            base = DirectDeclarator::Function {
                base: Box::new(base),
                params,
            };
        } else {
            break;
        }
    }

    base
}

// ===========================================================================
// parse_parameter_list — Function Parameter List
// ===========================================================================

/// Parses a function parameter list including the surrounding parentheses.
///
/// Handles:
/// - Empty parameter list: `()` — K&R style, unspecified parameters
/// - Void parameter list: `(void)` — explicitly zero parameters
/// - Parameter declarations: `(int x, float y)`
/// - Variadic: `(int x, ...)`
/// - K&R identifier list: `(a, b, c)` — old-style parameter names only
/// - Abstract parameters: `(int, float)` — unnamed parameters
///
/// Grammar:
/// ```text
///   parameter-type-list := parameter-list
///                        | parameter-list ',' '...'
///   parameter-list      := parameter-declaration
///                        | parameter-list ',' parameter-declaration
///   parameter-declaration := declaration-specifiers declarator
///                          | declaration-specifiers abstract-declarator?
/// ```
fn parse_parameter_list(parser: &mut Parser<'_>) -> ParamList {
    let start = parser.current_span();

    // Consume the opening `(`.
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

    // Check for `(void)` — explicitly zero parameters.
    if parser.check(TokenKind::Void) && parser.peek().kind == TokenKind::RightParen {
        parser.advance(); // consume `void`
        parser.advance(); // consume `)`
        return ParamList {
            params: Vec::new(),
            variadic: false,
            span: parser.span_from(start),
        };
    }

    // Check for K&R-style identifier list: `(a, b, c)`.
    // This is detected by seeing an identifier NOT followed by a type or
    // declarator syntax, but followed by `,` or `)`.
    if is_kr_identifier_list(parser) {
        return parse_kr_identifier_list(parser, start);
    }

    let mut params = Vec::new();
    let mut variadic = false;

    loop {
        // Check for `...` (variadic marker).
        if parser.match_token(TokenKind::Ellipsis) {
            variadic = true;
            break;
        }

        // Parse a single parameter declaration.
        let param = parse_parameter_declaration(parser);
        params.push(param);

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

/// Parses a single parameter declaration within a function parameter list.
///
/// A parameter declaration consists of declaration specifiers followed by
/// an optional declarator (named or abstract).
fn parse_parameter_declaration(parser: &mut Parser<'_>) -> ParamDeclaration {
    let param_start = parser.current_span();

    // Parse parameter declaration specifiers.
    let specifiers = parse_param_specifiers(parser);

    // Parse optional declarator (named or abstract).
    let declarator = try_parse_param_declarator(parser);

    // Parse optional GCC attributes after the parameter.
    let _attrs = gcc_extensions::try_parse_attributes(parser);

    let span = parser.span_from(param_start);
    ParamDeclaration {
        specifiers,
        declarator,
        span,
    }
}

/// Parses declaration specifiers for a function parameter.
///
/// Parameters allow type specifiers, type qualifiers, and `register` storage
/// class (only). Other storage classes are not valid for parameters.
fn parse_param_specifiers(parser: &mut Parser<'_>) -> DeclSpecifiers {
    let start = parser.current_span();

    let mut storage_class: Option<StorageClass> = None;
    let mut type_qualifiers: Vec<TypeQualifier> = Vec::new();
    let mut attributes: Vec<GccAttribute> = Vec::new();

    // Collect leading qualifiers, register, and attributes.
    loop {
        match parser.current().kind {
            TokenKind::Register => {
                storage_class = Some(StorageClass::Register);
                parser.advance();
            }
            TokenKind::Const => {
                type_qualifiers.push(TypeQualifier::Const);
                parser.advance();
            }
            TokenKind::Volatile => {
                type_qualifiers.push(TypeQualifier::Volatile);
                parser.advance();
            }
            TokenKind::Restrict => {
                type_qualifiers.push(TypeQualifier::Restrict);
                parser.advance();
            }
            TokenKind::Atomic if parser.peek().kind != TokenKind::LeftParen => {
                type_qualifiers.push(TypeQualifier::Atomic);
                parser.advance();
            }
            TokenKind::GccAttribute => {
                let mut new_attrs = gcc_extensions::try_parse_attributes(parser);
                attributes.append(&mut new_attrs);
            }
            TokenKind::GccExtension => {
                parser.advance();
            }
            _ => break,
        }
    }

    // Parse the type specifier.
    let type_specifier = types::parse_type_specifiers(parser);

    // Collect trailing qualifiers.
    loop {
        match parser.current().kind {
            TokenKind::Const => {
                type_qualifiers.push(TypeQualifier::Const);
                parser.advance();
            }
            TokenKind::Volatile => {
                type_qualifiers.push(TypeQualifier::Volatile);
                parser.advance();
            }
            TokenKind::Restrict => {
                type_qualifiers.push(TypeQualifier::Restrict);
                parser.advance();
            }
            TokenKind::Atomic if parser.peek().kind != TokenKind::LeftParen => {
                type_qualifiers.push(TypeQualifier::Atomic);
                parser.advance();
            }
            TokenKind::GccAttribute => {
                let mut new_attrs = gcc_extensions::try_parse_attributes(parser);
                attributes.append(&mut new_attrs);
            }
            _ => break,
        }
    }

    let span = parser.span_from(start);
    DeclSpecifiers {
        storage_class,
        type_qualifiers,
        type_specifier,
        function_specifiers: Vec::new(),
        attributes,
        span,
    }
}

/// Attempts to parse a declarator for a function parameter.
///
/// A parameter may have:
/// - A named declarator: `int x`, `int *p`, `int (*fp)(void)`
/// - An abstract declarator: `int *`, `int []`
/// - No declarator: just `int`
fn try_parse_param_declarator(parser: &mut Parser<'_>) -> Option<Declarator> {
    // Check if the current token can start a declarator.
    if !matches!(
        parser.current().kind,
        TokenKind::Star
            | TokenKind::Identifier
            | TokenKind::LeftParen
            | TokenKind::LeftBracket
    ) {
        return None;
    }

    let start = parser.current_span();

    // Parse pointer prefix.
    let ptrs = types::parse_pointer(parser);

    // Try to parse identifier or parenthesized declarator.
    let direct = match parser.current().kind {
        TokenKind::Identifier => {
            if let TokenValue::Identifier(id) = parser.current().value {
                // If this is a typedef name and we have no pointers, it may be
                // part of the type specifier rather than a parameter name.
                // In parameter context, a typedef name after a full type spec
                // IS the parameter name if we have pointers or the type spec
                // was already consumed.
                if parser.is_typedef_name(id) && ptrs.is_empty() {
                    // Could be a second type spec or parameter name.
                    // Lookahead: if followed by `)` or `,`, treat as
                    // parameter name. Otherwise, leave it.
                    let next = parser.peek().kind;
                    if matches!(
                        next,
                        TokenKind::RightParen
                            | TokenKind::Comma
                            | TokenKind::LeftBracket
                    ) {
                        parser.advance();
                        DirectDeclarator::Identifier(id)
                    } else {
                        // Not a parameter name — leave for caller.
                        if ptrs.is_empty() {
                            return None;
                        }
                        DirectDeclarator::Abstract
                    }
                } else {
                    parser.advance();
                    DirectDeclarator::Identifier(id)
                }
            } else {
                return None;
            }
        }
        TokenKind::LeftParen => {
            // Check if this is a parenthesized declarator or function params.
            // Heuristic: If `(` is followed by `*` or another `(`, it's likely
            // a parenthesized declarator (function pointer).
            let next = parser.peek().kind;
            if next == TokenKind::Star || next == TokenKind::LeftParen {
                parser.advance(); // consume `(`
                let inner = parse_declarator(parser);
                let _ = parser.expect(TokenKind::RightParen);
                DirectDeclarator::Parenthesized(Box::new(inner))
            } else if ptrs.is_empty() {
                // No pointers and `(` — could be function params for abstract.
                // Don't consume; return None.
                return None;
            } else {
                DirectDeclarator::Abstract
            }
        }
        TokenKind::LeftBracket => {
            // Array declarator without name — abstract.
            DirectDeclarator::Abstract
        }
        _ => {
            if !ptrs.is_empty() {
                DirectDeclarator::Abstract
            } else {
                return None;
            }
        }
    };

    // Parse array and function suffixes.
    let direct = parse_declarator_suffixes(parser, direct);

    let decl_attrs = gcc_extensions::try_parse_attributes(parser);

    let span = parser.span_from(start);
    Some(Declarator {
        pointer: ptrs,
        direct,
        attributes: decl_attrs,
        span,
    })
}

/// Parses array `[...]` and function `(...)` suffixes on a direct declarator.
fn parse_declarator_suffixes(
    parser: &mut Parser<'_>,
    mut base: DirectDeclarator,
) -> DirectDeclarator {
    loop {
        if parser.check(TokenKind::LeftBracket) {
            let dims = types::parse_array_dimensions(parser);
            for dim in dims {
                base = DirectDeclarator::Array {
                    base: Box::new(base),
                    size: dim,
                    qualifiers: Vec::new(),
                };
            }
        } else if parser.check(TokenKind::LeftParen) {
            let params = parse_parameter_list(parser);
            base = DirectDeclarator::Function {
                base: Box::new(base),
                params,
            };
        } else {
            break;
        }
    }
    base
}

// ===========================================================================
// parse_initializer — Variable Initializers (Simple and Compound)
// ===========================================================================

/// Parses an initializer for a variable declaration.
///
/// The `=` has already been consumed by the caller. This function parses
/// either:
/// - A simple expression initializer: `5`, `x + 1`, `f(a, b)`
/// - A compound (brace-enclosed) initializer: `{ 1, 2, 3 }`,
///   `{ .x = 1, .y = 2 }`, `{ { 1, 2 }, { 3, 4 } }`
///
/// Returns `Initializer::Expression(expr)` or
/// `Initializer::Compound { items, span }`.
pub(super) fn parse_initializer(parser: &mut Parser<'_>) -> Initializer {
    if parser.check(TokenKind::LeftBrace) {
        parse_compound_initializer(parser)
    } else {
        let expr = expressions::parse_assignment_expression(parser);
        Initializer::Expression(Box::new(expr))
    }
}

/// Parses a compound (brace-enclosed) initializer.
///
/// Grammar:
/// ```text
///   initializer := '{' initializer-list ','? '}'
///   initializer-list := designation? initializer
///                     | initializer-list ',' designation? initializer
///   designation := designator-list '='
///   designator := '[' constant-expression ']'
///               | '.' IDENTIFIER
/// ```
///
/// Supports C99 designated initializers with field (`.x`), index (`[0]`),
/// and GCC range (`[0 ... 3]`) designators.
fn parse_compound_initializer(parser: &mut Parser<'_>) -> Initializer {
    let start = parser.current_span();

    // Consume the opening `{`.
    parser.advance();

    let mut items = Vec::new();

    // Handle empty initializer: `{ }` (GCC extension, valid in C23).
    if parser.check(TokenKind::RightBrace) {
        parser.advance();
        let span = parser.span_from(start);
        return Initializer::Compound { items, span };
    }

    loop {
        let item_start = parser.current_span();

        // Parse optional designation (`.field =`, `[index] =`).
        let designators = parse_designation(parser);

        // Parse the initializer value (could be nested compound).
        let init = parse_initializer(parser);

        let item_span = parser.span_from(item_start);
        items.push(DesignatedInitializer {
            designators,
            initializer: init,
            span: item_span,
        });

        // Expect `,` or `}`.
        if !parser.match_token(TokenKind::Comma) {
            break;
        }

        // Allow trailing comma before `}` (C99).
        if parser.check(TokenKind::RightBrace) {
            break;
        }
    }

    let _ = parser.expect(TokenKind::RightBrace);

    let span = parser.span_from(start);
    Initializer::Compound { items, span }
}

/// Parses a designation (designator list followed by `=`).
///
/// Returns an empty Vec if no designation is present (plain initializer).
///
/// Designator forms:
/// - `.field` — struct/union field designator
/// - `[index]` — array index designator
/// - `[low ... high]` — GCC range designator
///
/// Multiple designators can be chained: `[0].field[1] = value`.
fn parse_designation(parser: &mut Parser<'_>) -> Vec<Designator> {
    let mut designators = Vec::new();

    loop {
        if parser.check(TokenKind::Dot) {
            parser.advance(); // consume `.`
            if let TokenValue::Identifier(id) = parser.current().value {
                parser.advance();
                designators.push(Designator::Field(id));
            } else {
                parser.error("expected field name after '.' in designator");
                break;
            }
        } else if parser.check(TokenKind::LeftBracket) {
            parser.advance(); // consume `[`
            let index_expr = expressions::parse_assignment_expression(parser);

            // Check for GCC range designator: `[low ... high]`.
            if parser.check(TokenKind::Ellipsis) {
                parser.advance(); // consume `...`
                let high_expr = expressions::parse_assignment_expression(parser);
                let _ = parser.expect(TokenKind::RightBracket);
                designators.push(Designator::Range(
                    Box::new(index_expr),
                    Box::new(high_expr),
                ));
            } else {
                let _ = parser.expect(TokenKind::RightBracket);
                designators.push(Designator::Index(Box::new(index_expr)));
            }
        } else {
            break;
        }
    }

    // If we collected designators, expect `=` to follow.
    if !designators.is_empty() {
        let _ = parser.expect(TokenKind::Equal);
    }

    designators
}

// ===========================================================================
// parse_struct_or_union — Struct and Union Definitions
// ===========================================================================

/// Parses a struct or union definition or reference.
///
/// Called when the current token is `struct` or `union`. Handles:
/// - Full definition: `struct S { int x; float y; };`
/// - Forward declaration/reference: `struct S;` or `struct S` in a declaration
/// - Anonymous struct/union: `struct { int x; };`
/// - GCC attributes: `struct __attribute__((packed)) S { ... };`
///
/// Returns a `TypeSpecifier::Struct(...)`, `TypeSpecifier::Union(...)`,
/// `TypeSpecifier::StructRef { ... }`, or `TypeSpecifier::UnionRef { ... }`.
pub(super) fn parse_struct_or_union(parser: &mut Parser<'_>) -> TypeSpecifier {
    let start = parser.current_span();
    let is_struct = parser.check(TokenKind::Struct);

    // Consume `struct` or `union` keyword.
    parser.advance();

    // Parse optional GCC attributes after the keyword.
    let pre_attrs = gcc_extensions::try_parse_attributes(parser);

    // Parse optional tag name.
    let tag = if parser.check(TokenKind::Identifier) {
        if let TokenValue::Identifier(id) = parser.current().value {
            parser.advance();
            Some(id)
        } else {
            None
        }
    } else {
        None
    };

    // Parse optional GCC attributes after the tag name.
    let mut all_attrs = pre_attrs;
    let mut post_tag_attrs = gcc_extensions::try_parse_attributes(parser);
    all_attrs.append(&mut post_tag_attrs);

    // Check for struct/union body `{ ... }`.
    if parser.check(TokenKind::LeftBrace) {
        // Parse the member list.
        let members = parse_struct_members(parser);

        // Parse trailing attributes.
        let mut trailing_attrs = gcc_extensions::try_parse_attributes(parser);
        all_attrs.append(&mut trailing_attrs);

        let span = parser.span_from(start);

        if is_struct {
            TypeSpecifier::Struct(StructDef {
                tag,
                members,
                attributes: all_attrs,
                span,
            })
        } else {
            TypeSpecifier::Union(UnionDef {
                tag,
                members,
                attributes: all_attrs,
                span,
            })
        }
    } else {
        // No body — this is a forward declaration or type reference.
        if let Some(tag_name) = tag {
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
            // Anonymous struct/union without a body — error.
            parser.error("expected '{' for anonymous struct/union definition");
            TypeSpecifier::Error
        }
    }
}

/// Parses struct/union member declarations within `{ ... }`.
///
/// Handles:
/// - Normal fields: `int x;`, `int x, y;`
/// - Bitfields: `int x : 3;`, `int : 5;` (unnamed bitfield)
/// - Anonymous struct/union members (C11): `struct { int x; };`
/// - `_Static_assert` in struct body
/// - Flexible array member: `int data[];` as last member
fn parse_struct_members(parser: &mut Parser<'_>) -> Vec<StructMember> {
    let mut members = Vec::new();

    // Consume the opening `{`.
    parser.advance();

    while !parser.check(TokenKind::RightBrace) && !parser.is_at_end() {
        let member_start = parser.current_span();

        // Handle _Static_assert in struct body.
        if parser.check(TokenKind::StaticAssert) {
            if let Declaration::StaticAssert {
                expr,
                message,
                span,
            } = parse_static_assert(parser)
            {
                members.push(StructMember::StaticAssert {
                    expr,
                    message,
                    span,
                });
                continue;
            }
            continue;
        }

        // Handle GCC __extension__ prefix.
        if parser.check(TokenKind::GccExtension) {
            parser.advance();
            continue;
        }

        // Check for anonymous struct/union member (C11).
        // An anonymous struct/union is a struct/union definition without a
        // declarator, directly as a member.
        if (parser.check(TokenKind::Struct) || parser.check(TokenKind::Union))
            && is_anonymous_struct_union_member(parser)
        {
            let type_spec = parse_struct_or_union(parser);
            let _ = parser.expect(TokenKind::Semicolon);
            let span = parser.span_from(member_start);
            members.push(StructMember::Anonymous { type_spec, span });
            continue;
        }

        // Parse member declaration specifiers (typedef is invalid in a
        // struct/union member, so we discard the is_typedef flag).
        let (specifiers, _is_typedef) = parse_declaration_specifiers(parser);

        // Check for anonymous bitfield or empty declaration.
        if parser.check(TokenKind::Semicolon) {
            // Bare type specifier — anonymous struct/union definition or
            // just `int;` (valid but useless).
            parser.advance();
            let span = parser.span_from(member_start);
            members.push(StructMember::Field {
                specifiers,
                declarators: Vec::new(),
                span,
            });
            continue;
        }

        // Check for unnamed bitfield: `: width ;`
        if parser.check(TokenKind::Colon) {
            parser.advance(); // consume `:`
            let width = expressions::parse_assignment_expression(parser);
            let _ = parser.expect(TokenKind::Semicolon);
            let span = parser.span_from(member_start);
            members.push(StructMember::Field {
                specifiers,
                declarators: vec![StructFieldDeclarator {
                    declarator: None,
                    bit_width: Some(Box::new(width)),
                    span,
                }],
                span,
            });
            continue;
        }

        // Parse struct field declarators (comma-separated, with optional
        // bitfield widths).
        let mut declarators = Vec::new();

        loop {
            let decl_start = parser.current_span();
            let field_decl = parse_declarator(parser);

            // Check for bitfield width.
            let bit_width = if parser.match_token(TokenKind::Colon) {
                Some(Box::new(
                    expressions::parse_assignment_expression(parser),
                ))
            } else {
                None
            };

            // Parse optional GCC attributes on the field.
            let _attrs = gcc_extensions::try_parse_attributes(parser);

            let decl_span = parser.span_from(decl_start);
            declarators.push(StructFieldDeclarator {
                declarator: Some(field_decl),
                bit_width,
                span: decl_span,
            });

            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }

        if parser.expect(TokenKind::Semicolon).is_err() {
            // Error recovery: skip to next `;` or `}`.
            synchronize_struct_member(parser);
        }

        let span = parser.span_from(member_start);
        members.push(StructMember::Field {
            specifiers,
            declarators,
            span,
        });
    }

    // Consume the closing `}`.
    let _ = parser.expect(TokenKind::RightBrace);

    members
}

/// Returns `true` if a struct/union keyword starts an anonymous member.
///
/// An anonymous struct/union member is a struct or union definition (with body)
/// that has no declarator name — i.e., `struct { ... };` or
/// `union { ... };` directly inside another struct/union.
fn is_anonymous_struct_union_member(parser: &Parser<'_>) -> bool {
    // Look ahead past `struct`/`union`, optional attributes, optional tag name,
    // and check if there's a `{` body followed by `;` with no declarators.
    let mut offset = 1; // Skip `struct`/`union`.

    // Skip optional __attribute__ blocks.
    while parser.lookahead(offset).kind == TokenKind::GccAttribute {
        offset += 1;
        // Skip past `((`...`))` — approximate by counting parens.
        let mut depth = 0;
        loop {
            let kind = parser.lookahead(offset).kind;
            if kind == TokenKind::LeftParen {
                depth += 1;
                offset += 1;
            } else if kind == TokenKind::RightParen {
                depth -= 1;
                offset += 1;
                if depth == 0 {
                    break;
                }
            } else if kind == TokenKind::Eof {
                return false;
            } else {
                offset += 1;
            }
        }
    }

    // Skip optional tag name.
    if parser.lookahead(offset).kind == TokenKind::Identifier {
        // If there's a tag and then `{`, it could be a definition.
        // If there's a tag and then something else, it's a reference type
        // used in a normal field declaration.
        let after_tag = parser.lookahead(offset + 1).kind;
        if after_tag == TokenKind::LeftBrace {
            // `struct Tag { ... }` — need to check if followed by `;` with no
            // declarator. This is complex to determine by lookahead alone, so
            // we conservatively return false and let normal member parsing
            // handle it.
            return false;
        }
        return false;
    }

    // No tag name — check for `{`.
    parser.lookahead(offset).kind == TokenKind::LeftBrace
}

/// Error recovery within struct member parsing — skip to next `;` or `}`.
fn synchronize_struct_member(parser: &mut Parser<'_>) {
    while !parser.is_at_end() {
        match parser.current().kind {
            TokenKind::Semicolon => {
                parser.advance();
                return;
            }
            TokenKind::RightBrace => {
                return; // Don't consume — let the struct parser handle it.
            }
            _ => {
                parser.advance();
            }
        }
    }
}

// ===========================================================================
// parse_enum — Enum Definition
// ===========================================================================

/// Parses an enum definition or reference.
///
/// Called when the current token is `enum`. Handles:
/// - Full definition: `enum Color { RED, GREEN, BLUE };`
/// - Definition with values: `enum { A = 0, B = 1, C = 2 };`
/// - Forward reference: `enum E` or `enum E;`
/// - Trailing comma: `enum { A, B, C, };` (C99)
/// - GCC attributes on enum and variants
///
/// Returns `TypeSpecifier::Enum(EnumDef { ... })` or
/// `TypeSpecifier::EnumRef { tag, span }`.
pub(super) fn parse_enum(parser: &mut Parser<'_>) -> TypeSpecifier {
    let start = parser.current_span();

    // Consume `enum` keyword.
    parser.advance();

    // Parse optional GCC attributes.
    let pre_attrs = gcc_extensions::try_parse_attributes(parser);

    // Parse optional tag name.
    let tag = if parser.check(TokenKind::Identifier) {
        if let TokenValue::Identifier(id) = parser.current().value {
            parser.advance();
            Some(id)
        } else {
            None
        }
    } else {
        None
    };

    // Parse optional GCC attributes after tag.
    let mut all_attrs = pre_attrs;
    let mut post_tag_attrs = gcc_extensions::try_parse_attributes(parser);
    all_attrs.append(&mut post_tag_attrs);

    // Check for enum body `{ ... }`.
    if parser.check(TokenKind::LeftBrace) {
        parser.advance(); // consume `{`

        let mut variants = Vec::new();

        // Handle empty enum body (GCC extension).
        if !parser.check(TokenKind::RightBrace) {
            loop {
                let variant_start = parser.current_span();

                // Parse variant name.
                let variant_name = if parser.check(TokenKind::Identifier) {
                    if let TokenValue::Identifier(id) = parser.current().value {
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

                // Parse optional GCC attributes on the variant.
                let variant_attrs = gcc_extensions::try_parse_attributes(parser);

                // Parse optional value: `= constant_expression`.
                let value = if parser.match_token(TokenKind::Equal) {
                    Some(Box::new(
                        expressions::parse_assignment_expression(parser),
                    ))
                } else {
                    None
                };

                // Parse trailing attributes on variant.
                let mut all_variant_attrs = variant_attrs;
                let mut trailing = gcc_extensions::try_parse_attributes(parser);
                all_variant_attrs.append(&mut trailing);

                let variant_span = parser.span_from(variant_start);
                variants.push(EnumVariant {
                    name: variant_name,
                    value,
                    attributes: all_variant_attrs,
                    span: variant_span,
                });

                // Expect `,` or `}`.
                if !parser.match_token(TokenKind::Comma) {
                    break;
                }

                // Allow trailing comma before `}` (C99).
                if parser.check(TokenKind::RightBrace) {
                    break;
                }
            }
        }

        let _ = parser.expect(TokenKind::RightBrace);

        // Parse trailing attributes.
        let mut trailing_attrs = gcc_extensions::try_parse_attributes(parser);
        all_attrs.append(&mut trailing_attrs);

        let span = parser.span_from(start);
        TypeSpecifier::Enum(EnumDef {
            tag,
            variants,
            attributes: all_attrs,
            span,
        })
    } else {
        // No body — enum reference.
        if let Some(tag_name) = tag {
            let span = parser.span_from(start);
            TypeSpecifier::EnumRef {
                tag: tag_name,
                span,
            }
        } else {
            parser.error("expected '{' for anonymous enum definition");
            TypeSpecifier::Error
        }
    }
}

// ===========================================================================
// parse_static_assert — _Static_assert Declaration
// ===========================================================================

/// Parses a `_Static_assert` declaration.
///
/// Grammar:
/// ```text
///   _Static_assert '(' constant-expression ',' string-literal ')' ';'
/// ```
///
/// Returns `Declaration::StaticAssert { expr, message, span }`.
pub(super) fn parse_static_assert(parser: &mut Parser<'_>) -> Declaration {
    let start = parser.current_span();

    // Consume `_Static_assert`.
    parser.advance();

    // Expect `(`.
    if parser.expect(TokenKind::LeftParen).is_err() {
        synchronize_declaration(parser);
        return Declaration::Empty {
            span: parser.span_from(start),
        };
    }

    // Parse constant expression.
    let expr = expressions::parse_assignment_expression(parser);

    // Expect `,`.
    if parser.expect(TokenKind::Comma).is_err() {
        // Try to recover — maybe they forgot the comma.
        if !parser.check(TokenKind::StringLiteral) {
            synchronize_declaration(parser);
            return Declaration::Empty {
                span: parser.span_from(start),
            };
        }
    }

    // Parse string literal message.
    let message = if parser.check(TokenKind::StringLiteral) {
        let msg = if let TokenValue::Str(ref s) = parser.current().value {
            s.clone()
        } else {
            String::new()
        };
        parser.advance();
        // Handle concatenated string literals.
        let mut result = msg;
        while parser.check(TokenKind::StringLiteral) {
            if let TokenValue::Str(ref s) = parser.current().value {
                result.push_str(s);
            }
            parser.advance();
        }
        result
    } else {
        parser.error("expected string literal in _Static_assert");
        String::new()
    };

    // Expect `)`.
    let _ = parser.expect(TokenKind::RightParen);

    // Expect `;`.
    let _ = parser.expect(TokenKind::Semicolon);

    let span = parser.span_from(start);
    Declaration::StaticAssert {
        expr: Box::new(expr),
        message,
        span,
    }
}

// ===========================================================================
// Function Definition Parsing
// ===========================================================================

/// Parses a function definition when the declarator is followed by `{`.
///
/// The specifiers and first declarator have already been parsed. This function
/// handles:
/// - GCC attributes before the body: `int f(void) __attribute__((noinline)) { ... }`
/// - The function body as a compound statement.
///
/// Returns `Declaration::Function(Box<FunctionDef>)`.
fn parse_function_definition(
    parser: &mut Parser<'_>,
    specifiers: DeclSpecifiers,
    declarator: Declarator,
    extra_attrs: Vec<GccAttribute>,
    start: SourceSpan,
) -> Declaration {
    // Collect any additional GCC attributes before the body.
    let mut func_attrs = extra_attrs;
    let mut more_attrs = gcc_extensions::try_parse_attributes(parser);
    func_attrs.append(&mut more_attrs);

    // Push a new scope for the function body (typedef name scoping).
    parser.push_scope();

    // Parse the function body as a compound statement.
    let body = statements::parse_compound_statement(parser);

    // Pop the function body scope.
    parser.pop_scope();

    let span = parser.span_from(start);
    Declaration::Function(Box::new(FunctionDef {
        specifiers,
        declarator,
        body: Box::new(body),
        attributes: func_attrs,
        span,
    }))
}

/// Checks if the declaration is a K&R-style function definition.
///
/// K&R style: `int foo(a, b) int a; float b; { ... }`
/// The declarator has a function parameter list consisting only of identifiers,
/// and the next tokens are declaration specifiers (not `{`).
fn is_kr_style_definition(parser: &Parser<'_>, declarator: &Declarator) -> bool {
    // Check if the declarator has a function direct declarator.
    if !has_function_declarator(&declarator.direct) {
        return false;
    }

    // In K&R style, after the declarator (and `)`) we see type specifiers
    // for parameter declarations, not `{` and not `;`.
    let kind = parser.current().kind;
    kind.is_type_specifier()
        || kind.is_storage_class()
        || kind.is_type_qualifier()
}

/// Returns `true` if a direct declarator has a function parameter suffix.
fn has_function_declarator(dd: &DirectDeclarator) -> bool {
    match dd {
        DirectDeclarator::Function { .. } => true,
        DirectDeclarator::Parenthesized(inner) => has_function_declarator(&inner.direct),
        _ => false,
    }
}

/// Parses a K&R-style function definition.
///
/// In K&R style, parameter declarations appear between the `)` and `{`:
/// ```c
/// int foo(a, b)
/// int a;
/// float b;
/// {
///     return a + b;
/// }
/// ```
fn parse_kr_function_definition(
    parser: &mut Parser<'_>,
    specifiers: DeclSpecifiers,
    declarator: Declarator,
    extra_attrs: Vec<GccAttribute>,
    start: SourceSpan,
) -> Declaration {
    // Parse K&R parameter declarations until we hit `{`.
    while !parser.check(TokenKind::LeftBrace) && !parser.is_at_end() {
        // Parse a parameter declaration.
        let (_param_specs, _) = parse_declaration_specifiers(parser);
        // Parse declarator(s).
        loop {
            let _decl = parse_declarator(parser);
            if !parser.match_token(TokenKind::Comma) {
                break;
            }
        }
        let _ = parser.expect(TokenKind::Semicolon);
    }

    // Now parse the function body.
    parse_function_definition(parser, specifiers, declarator, extra_attrs, start)
}

// ===========================================================================
// Typedef Declaration Parsing
// ===========================================================================

/// Parses a typedef declaration.
///
/// When `typedef` is present in the storage class, the declared names become
/// type names (typedef names) that the parser must track for subsequent
/// disambiguation between type names and identifiers.
///
/// Examples:
/// - `typedef int MyInt;`
/// - `typedef struct { int x; int y; } Point;`
/// - `typedef int (*FuncPtr)(int, int);`
fn parse_typedef_declaration(
    parser: &mut Parser<'_>,
    specifiers: DeclSpecifiers,
    first_declarator: Declarator,
    start: SourceSpan,
) -> Declaration {
    let mut declarators = vec![first_declarator.clone()];

    // Register the first typedef name.
    if let Some(name) = extract_declarator_name(&first_declarator) {
        parser.register_typedef(name);
    }

    // Parse additional declarators separated by commas.
    while parser.match_token(TokenKind::Comma) {
        let decl = parse_declarator(parser);
        if let Some(name) = extract_declarator_name(&decl) {
            parser.register_typedef(name);
        }
        declarators.push(decl);
    }

    // Expect `;`.
    let _ = parser.expect(TokenKind::Semicolon);

    let span = parser.span_from(start);
    Declaration::Typedef {
        specifiers,
        declarators,
        span,
    }
}

/// Extracts the identifier name from a declarator, if present.
///
/// Navigates through parenthesized declarators and returns the innermost
/// identifier.
fn extract_declarator_name(decl: &Declarator) -> Option<InternId> {
    extract_direct_declarator_name(&decl.direct)
}

/// Extracts the identifier name from a direct declarator.
fn extract_direct_declarator_name(dd: &DirectDeclarator) -> Option<InternId> {
    match dd {
        DirectDeclarator::Identifier(id) => Some(*id),
        DirectDeclarator::Parenthesized(inner) => extract_declarator_name(inner),
        DirectDeclarator::Array { base, .. } => extract_direct_declarator_name(base),
        DirectDeclarator::Function { base, .. } => extract_direct_declarator_name(base),
        DirectDeclarator::Abstract => None,
    }
}

// ===========================================================================
// Variable Declaration Parsing
// ===========================================================================

/// Parses a variable declaration (non-typedef, non-function-definition).
///
/// The first declarator has already been parsed. This function handles:
/// - Optional initializer on first declarator: `int x = 5;`
/// - Additional comma-separated declarators: `int x, y = 3, z;`
/// - Final semicolon
fn parse_variable_declaration(
    parser: &mut Parser<'_>,
    specifiers: DeclSpecifiers,
    first_declarator: Declarator,
    first_extra_attrs: Vec<GccAttribute>,
    start: SourceSpan,
) -> Declaration {
    let mut declarators = Vec::new();

    // Skip optional GCC asm label on first declarator.
    skip_asm_label(parser);

    // Parse optional initializer for the first declarator.
    let first_init = if parser.match_token(TokenKind::Equal) {
        Some(parse_initializer(parser))
    } else {
        None
    };

    // Parse optional GCC attributes after first initializer.
    let mut first_attrs = first_extra_attrs;
    let mut trailing = gcc_extensions::try_parse_attributes(parser);
    first_attrs.append(&mut trailing);

    let first_span = first_declarator.span;
    declarators.push(InitDeclarator {
        declarator: first_declarator,
        initializer: first_init,
        span: first_span,
    });

    // Parse additional declarators.
    while parser.match_token(TokenKind::Comma) {
        let decl_start = parser.current_span();
        let decl = parse_declarator(parser);

        let init = if parser.match_token(TokenKind::Equal) {
            Some(parse_initializer(parser))
        } else {
            None
        };

        let _attrs = gcc_extensions::try_parse_attributes(parser);

        let decl_span = parser.span_from(decl_start);
        declarators.push(InitDeclarator {
            declarator: decl,
            initializer: init,
            span: decl_span,
        });
    }

    // Expect `;`.
    if parser.expect(TokenKind::Semicolon).is_err() {
        synchronize_declaration(parser);
    }

    let span = parser.span_from(start);
    Declaration::Variable {
        specifiers,
        declarators,
        span,
    }
}

// ===========================================================================
// K&R Identifier List Parsing
// ===========================================================================

/// Returns `true` if the current parameter list appears to be a K&R-style
/// identifier list (just names, no types).
///
/// Heuristic: If the first token is an identifier that is NOT a typedef name,
/// and it's followed by `,` or `)`, it's likely a K&R identifier list.
fn is_kr_identifier_list(parser: &Parser<'_>) -> bool {
    if parser.current().kind != TokenKind::Identifier {
        return false;
    }

    if let TokenValue::Identifier(id) = parser.current().value {
        // If it's a typedef name, it's likely a declaration, not K&R.
        if parser.is_typedef_name(id) {
            return false;
        }

        // Check lookahead: in a K&R list, identifier is followed by `,` or `)`.
        let next = parser.peek().kind;
        matches!(next, TokenKind::Comma | TokenKind::RightParen)
    } else {
        false
    }
}

/// Parses a K&R-style identifier list: `(a, b, c)`.
fn parse_kr_identifier_list(parser: &mut Parser<'_>, start: SourceSpan) -> ParamList {
    let mut params = Vec::new();

    loop {
        if parser.check(TokenKind::Identifier) {
            if let TokenValue::Identifier(id) = parser.current().value {
                let param_start = parser.current_span();
                parser.advance();

                // Create a minimal ParamDeclaration with just an identifier.
                let decl_span = parser.span_from(param_start);
                params.push(ParamDeclaration {
                    specifiers: DeclSpecifiers {
                        storage_class: None,
                        type_qualifiers: Vec::new(),
                        type_specifier: TypeSpecifier::Int, // Default for K&R.
                        function_specifiers: Vec::new(),
                        attributes: Vec::new(),
                        span: decl_span,
                    },
                    declarator: Some(Declarator {
                        pointer: Vec::new(),
                        direct: DirectDeclarator::Identifier(id),
                        attributes: Vec::new(),
                        span: decl_span,
                    }),
                    span: decl_span,
                });
            }
        }

        if !parser.match_token(TokenKind::Comma) {
            break;
        }
    }

    let _ = parser.expect(TokenKind::RightParen);

    let span = parser.span_from(start);
    ParamList {
        params,
        variadic: false,
        span,
    }
}

// ===========================================================================
// Error Recovery
// ===========================================================================

/// Synchronizes the parser to the next declaration boundary after an error.
/// Skips an optional GCC `__asm__("symbol_name")` label attached to a
/// declaration. Used by glibc's `__REDIRECT` macro for assembly-level symbol
/// renaming. The syntax is:
///   `__asm__` `(` string-literal `)` | `asm` `(` string-literal `)`
/// We simply consume the tokens without storing them — the asm name does
/// not affect semantic analysis or code generation in our compiler.
fn skip_asm_label(parser: &mut Parser<'_>) {
    if parser.check(TokenKind::Asm) {
        parser.advance(); // consume __asm__
        if parser.match_token(TokenKind::LeftParen) {
            // Skip everything inside until matching ')'.
            let mut depth = 1u32;
            while !parser.is_at_end() && depth > 0 {
                if parser.check(TokenKind::LeftParen) {
                    depth += 1;
                } else if parser.check(TokenKind::RightParen) {
                    depth -= 1;
                    if depth == 0 {
                        parser.advance(); // consume final ')'
                        break;
                    }
                }
                parser.advance();
            }
        }
    }
}

/// Skips tokens until `;`, `}`, or a declaration-starting keyword is found.
/// This provides error recovery for malformed declarations.
fn synchronize_declaration(parser: &mut Parser<'_>) {
    while !parser.is_at_end() {
        match parser.current().kind {
            TokenKind::Semicolon => {
                parser.advance();
                return;
            }
            TokenKind::RightBrace => {
                return; // Don't consume — let the enclosing scope handle it.
            }
            kind if kind.is_type_specifier()
                || kind.is_storage_class()
                || kind == TokenKind::Inline
                || kind == TokenKind::Noreturn
                || kind == TokenKind::GccAttribute
                || kind == TokenKind::StaticAssert =>
            {
                return; // Found a declaration starter — resume.
            }
            _ => {
                parser.advance();
            }
        }
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
    use crate::common::source_map::SourceSpan;
    use crate::frontend::lexer::token::{Token, TokenKind, TokenValue};
    use crate::frontend::parser::ast::Statement;

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

    /// Creates a string literal token.
    fn make_str_token(s: &str) -> Token {
        Token::new(
            TokenKind::StringLiteral,
            dummy_span(),
            TokenValue::Str(s.to_string()),
        )
    }

    /// Builds a token stream from token constructors and appends EOF.
    fn build_tokens(tokens: Vec<Token>) -> Vec<Token> {
        let mut toks = tokens;
        toks.push(make_token(TokenKind::Eof));
        toks
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
    // Test: Simple variable declaration — `int x;`
    // -----------------------------------------------------------------------
    #[test]
    fn test_simple_variable_declaration() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                assert!(declarators[0].initializer.is_none());
            }
            _ => panic!("expected Variable declaration, got {:?}", decl),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Variable with initializer — `int x = 5;`
    // -----------------------------------------------------------------------
    #[test]
    fn test_variable_with_initializer() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Equal),
            make_int_token(5),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                assert!(declarators[0].initializer.is_some());
            }
            _ => panic!("expected Variable declaration"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Multiple declarators — `int x, y, z;`
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_declarators() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let y_id = interner.intern("y");
        let z_id = interner.intern("z");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Comma),
            make_ident_token(y_id),
            make_token(TokenKind::Comma),
            make_ident_token(z_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 3);
            }
            _ => panic!("expected Variable declaration with 3 declarators"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Pointer declaration — `int *p;`
    // -----------------------------------------------------------------------
    #[test]
    fn test_pointer_declaration() {
        let mut interner = Interner::new();
        let p_id = interner.intern("p");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_token(TokenKind::Star),
            make_ident_token(p_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                assert_eq!(declarators[0].declarator.pointer.len(), 1);
            }
            _ => panic!("expected pointer Variable declaration"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Function declaration — `int foo(int x, int y);`
    // -----------------------------------------------------------------------
    #[test]
    fn test_function_declaration() {
        let mut interner = Interner::new();
        let foo_id = interner.intern("foo");
        let x_id = interner.intern("x");
        let y_id = interner.intern("y");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(foo_id),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Comma),
            make_token(TokenKind::Int),
            make_ident_token(y_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        // Function declaration without a body should be Variable with function
        // declarator.
        match decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                match &declarators[0].declarator.direct {
                    DirectDeclarator::Function { params, .. } => {
                        assert_eq!(params.params.len(), 2);
                    }
                    _ => panic!("expected function declarator"),
                }
            }
            _ => panic!("expected Variable (function declaration)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Function definition — `int main(void) { return 0; }`
    // -----------------------------------------------------------------------
    #[test]
    fn test_function_definition() {
        let mut interner = Interner::new();
        let main_id = interner.intern("main");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(main_id),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Void),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::LeftBrace),
            make_token(TokenKind::Return),
            make_int_token(0),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::RightBrace),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::Function(func_def) => {
                assert!(matches!(*func_def.body, Statement::Compound { .. }));
            }
            _ => panic!("expected Function definition, got {:?}", decl),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Typedef — `typedef int MyInt;`
    // -----------------------------------------------------------------------
    #[test]
    fn test_typedef_declaration() {
        let mut interner = Interner::new();
        let myint_id = interner.intern("MyInt");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Typedef),
            make_token(TokenKind::Int),
            make_ident_token(myint_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::Typedef { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                // Verify typedef name was registered.
                assert!(parser.is_typedef_name(myint_id));
            }
            _ => panic!("expected Typedef declaration"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Struct definition — `struct S { int x; float y; };`
    // -----------------------------------------------------------------------
    #[test]
    fn test_struct_definition() {
        let mut interner = Interner::new();
        let s_id = interner.intern("S");
        let x_id = interner.intern("x");
        let y_id = interner.intern("y");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Struct),
            make_ident_token(s_id),
            make_token(TokenKind::LeftBrace),
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Float),
            make_ident_token(y_id),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::RightBrace),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, .. } => {
                match &specifiers.type_specifier {
                    TypeSpecifier::Struct(sd) => {
                        assert_eq!(sd.tag, Some(s_id));
                        assert_eq!(sd.members.len(), 2);
                    }
                    _ => panic!("expected Struct type specifier"),
                }
            }
            _ => panic!("expected Variable declaration (struct def)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Enum definition — `enum Color { RED, GREEN, BLUE };`
    // -----------------------------------------------------------------------
    #[test]
    fn test_enum_definition() {
        let mut interner = Interner::new();
        let color_id = interner.intern("Color");
        let red_id = interner.intern("RED");
        let green_id = interner.intern("GREEN");
        let blue_id = interner.intern("BLUE");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Enum),
            make_ident_token(color_id),
            make_token(TokenKind::LeftBrace),
            make_ident_token(red_id),
            make_token(TokenKind::Comma),
            make_ident_token(green_id),
            make_token(TokenKind::Comma),
            make_ident_token(blue_id),
            make_token(TokenKind::RightBrace),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, .. } => {
                match &specifiers.type_specifier {
                    TypeSpecifier::Enum(ed) => {
                        assert_eq!(ed.tag, Some(color_id));
                        assert_eq!(ed.variants.len(), 3);
                    }
                    _ => panic!("expected Enum type specifier"),
                }
            }
            _ => panic!("expected Variable declaration (enum def)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Enum with values — `enum { A = 0, B = 1, C = 2 };`
    // -----------------------------------------------------------------------
    #[test]
    fn test_enum_with_values() {
        let mut interner = Interner::new();
        let a_id = interner.intern("A");
        let b_id = interner.intern("B");
        let c_id = interner.intern("C");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Enum),
            make_token(TokenKind::LeftBrace),
            make_ident_token(a_id),
            make_token(TokenKind::Equal),
            make_int_token(0),
            make_token(TokenKind::Comma),
            make_ident_token(b_id),
            make_token(TokenKind::Equal),
            make_int_token(1),
            make_token(TokenKind::Comma),
            make_ident_token(c_id),
            make_token(TokenKind::Equal),
            make_int_token(2),
            make_token(TokenKind::RightBrace),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, .. } => {
                match &specifiers.type_specifier {
                    TypeSpecifier::Enum(ed) => {
                        assert_eq!(ed.variants.len(), 3);
                        assert!(ed.variants[0].value.is_some());
                        assert!(ed.variants[1].value.is_some());
                        assert!(ed.variants[2].value.is_some());
                    }
                    _ => panic!("expected Enum type specifier with values"),
                }
            }
            _ => panic!("expected Variable declaration (enum with values)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Forward struct declaration — `struct S;`
    // -----------------------------------------------------------------------
    #[test]
    fn test_forward_struct_declaration() {
        let mut interner = Interner::new();
        let s_id = interner.intern("S");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Struct),
            make_ident_token(s_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, declarators, .. } => {
                assert!(declarators.is_empty());
                match &specifiers.type_specifier {
                    TypeSpecifier::StructRef { tag, .. } => {
                        assert_eq!(*tag, s_id);
                    }
                    _ => panic!("expected StructRef"),
                }
            }
            _ => panic!("expected Variable declaration (forward struct)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: _Static_assert
    // -----------------------------------------------------------------------
    #[test]
    fn test_static_assert() {
        let mut interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::StaticAssert),
            make_token(TokenKind::LeftParen),
            make_int_token(1),
            make_token(TokenKind::Comma),
            make_str_token("assertion message"),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match decl {
            Declaration::StaticAssert { message, .. } => {
                assert_eq!(message, "assertion message");
            }
            _ => panic!("expected StaticAssert declaration"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Extern and static storage classes
    // -----------------------------------------------------------------------
    #[test]
    fn test_storage_classes() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let mut diag = DiagnosticEmitter::new();

        // `extern int x;`
        let tokens = build_tokens(vec![
            make_token(TokenKind::Extern),
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, .. } => {
                assert_eq!(specifiers.storage_class, Some(StorageClass::Extern));
            }
            _ => panic!("expected Variable with extern storage class"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Union definition — `union U { int i; float f; };`
    // -----------------------------------------------------------------------
    #[test]
    fn test_union_definition() {
        let mut interner = Interner::new();
        let u_id = interner.intern("U");
        let i_id = interner.intern("i");
        let f_id = interner.intern("f");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Union),
            make_ident_token(u_id),
            make_token(TokenKind::LeftBrace),
            make_token(TokenKind::Int),
            make_ident_token(i_id),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Float),
            make_ident_token(f_id),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::RightBrace),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, .. } => {
                match &specifiers.type_specifier {
                    TypeSpecifier::Union(ud) => {
                        assert_eq!(ud.tag, Some(u_id));
                        assert_eq!(ud.members.len(), 2);
                    }
                    _ => panic!("expected Union type specifier"),
                }
            }
            _ => panic!("expected Variable declaration (union def)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Function pointer — `int (*fp)(int, int);`
    // -----------------------------------------------------------------------
    #[test]
    fn test_function_pointer() {
        let mut interner = Interner::new();
        let fp_id = interner.intern("fp");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Star),
            make_ident_token(fp_id),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Int),
            make_token(TokenKind::Comma),
            make_token(TokenKind::Int),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                // The declarator should have a parenthesized inner with
                // pointer and function suffix.
                match &declarators[0].declarator.direct {
                    DirectDeclarator::Function { base, params } => {
                        // base is Parenthesized(pointer declarator)
                        assert_eq!(params.params.len(), 2);
                    }
                    _ => panic!("expected function-type declarator"),
                }
            }
            _ => panic!("expected Variable declaration (function pointer)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Variadic function — `int printf(const char *fmt, ...);`
    // -----------------------------------------------------------------------
    #[test]
    fn test_variadic_function() {
        let mut interner = Interner::new();
        let printf_id = interner.intern("printf");
        let fmt_id = interner.intern("fmt");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(printf_id),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Const),
            make_token(TokenKind::Char),
            make_token(TokenKind::Star),
            make_ident_token(fmt_id),
            make_token(TokenKind::Comma),
            make_token(TokenKind::Ellipsis),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { declarators, .. } => {
                match &declarators[0].declarator.direct {
                    DirectDeclarator::Function { params, .. } => {
                        assert!(params.variadic);
                        assert_eq!(params.params.len(), 1);
                    }
                    _ => panic!("expected function declarator"),
                }
            }
            _ => panic!("expected Variable declaration (variadic function)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Empty parameter list — `void f();`
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_parameter_list() {
        let mut interner = Interner::new();
        let f_id = interner.intern("f");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Void),
            make_ident_token(f_id),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { declarators, .. } => {
                match &declarators[0].declarator.direct {
                    DirectDeclarator::Function { params, .. } => {
                        assert!(params.params.is_empty());
                        assert!(!params.variadic);
                    }
                    _ => panic!("expected function declarator"),
                }
            }
            _ => panic!("expected Variable declaration (empty params)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Void parameter list — `void f(void);`
    // -----------------------------------------------------------------------
    #[test]
    fn test_void_parameter_list() {
        let mut interner = Interner::new();
        let f_id = interner.intern("f");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Void),
            make_ident_token(f_id),
            make_token(TokenKind::LeftParen),
            make_token(TokenKind::Void),
            make_token(TokenKind::RightParen),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { declarators, .. } => {
                match &declarators[0].declarator.direct {
                    DirectDeclarator::Function { params, .. } => {
                        assert!(params.params.is_empty());
                    }
                    _ => panic!("expected function declarator"),
                }
            }
            _ => panic!("expected Variable declaration (void params)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Struct with bitfields — `struct { int x : 3; int y : 5; };`
    // -----------------------------------------------------------------------
    #[test]
    fn test_struct_with_bitfields() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let y_id = interner.intern("y");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Struct),
            make_token(TokenKind::LeftBrace),
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            make_token(TokenKind::Colon),
            make_int_token(3),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::Int),
            make_ident_token(y_id),
            make_token(TokenKind::Colon),
            make_int_token(5),
            make_token(TokenKind::Semicolon),
            make_token(TokenKind::RightBrace),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, .. } => {
                match &specifiers.type_specifier {
                    TypeSpecifier::Struct(sd) => {
                        assert_eq!(sd.members.len(), 2);
                        // Verify first member has bit_width.
                        match &sd.members[0] {
                            StructMember::Field { declarators, .. } => {
                                assert!(declarators[0].bit_width.is_some());
                            }
                            _ => panic!("expected Field member"),
                        }
                    }
                    _ => panic!("expected Struct type specifier"),
                }
            }
            _ => panic!("expected Variable declaration (struct with bitfields)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Compound initializer — `int arr[] = { 1, 2, 3 };`
    // -----------------------------------------------------------------------
    #[test]
    fn test_compound_initializer() {
        let mut interner = Interner::new();
        let arr_id = interner.intern("arr");
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(arr_id),
            make_token(TokenKind::LeftBracket),
            make_token(TokenKind::RightBracket),
            make_token(TokenKind::Equal),
            make_token(TokenKind::LeftBrace),
            make_int_token(1),
            make_token(TokenKind::Comma),
            make_int_token(2),
            make_token(TokenKind::Comma),
            make_int_token(3),
            make_token(TokenKind::RightBrace),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { declarators, .. } => {
                assert_eq!(declarators.len(), 1);
                match &declarators[0].initializer {
                    Some(Initializer::Compound { items, .. }) => {
                        assert_eq!(items.len(), 3);
                    }
                    _ => panic!("expected Compound initializer"),
                }
            }
            _ => panic!("expected Variable declaration (compound init)"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Designated initializer — `{ .x = 1, .y = 2 }`
    // -----------------------------------------------------------------------
    #[test]
    fn test_designated_initializer() {
        let mut interner = Interner::new();
        let s_id = interner.intern("s");
        let x_id = interner.intern("x");
        let y_id = interner.intern("y");
        let mut diag = DiagnosticEmitter::new();

        // For simplicity, test the initializer parser directly.
        let tokens = build_tokens(vec![
            make_token(TokenKind::LeftBrace),
            make_token(TokenKind::Dot),
            make_ident_token(x_id),
            make_token(TokenKind::Equal),
            make_int_token(1),
            make_token(TokenKind::Comma),
            make_token(TokenKind::Dot),
            make_ident_token(y_id),
            make_token(TokenKind::Equal),
            make_int_token(2),
            make_token(TokenKind::RightBrace),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let init = parse_initializer(&mut parser);

        match init {
            Initializer::Compound { items, .. } => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0].designators.len(), 1);
                match &items[0].designators[0] {
                    Designator::Field(id) => assert_eq!(*id, x_id),
                    _ => panic!("expected Field designator"),
                }
            }
            _ => panic!("expected Compound initializer"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: Empty declaration (standalone semicolon)
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_declaration() {
        let mut interner = Interner::new();
        let mut diag = DiagnosticEmitter::new();

        let tokens = build_tokens(vec![make_token(TokenKind::Semicolon)]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        assert!(matches!(decl, Declaration::Empty { .. }));
    }

    // -----------------------------------------------------------------------
    // Test: Error recovery — missing semicolon
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_recovery_missing_semicolon() {
        let mut interner = Interner::new();
        let x_id = interner.intern("x");
        let y_id = interner.intern("y");
        let mut diag = DiagnosticEmitter::new();

        // `int x` (missing `;`) followed by `int y;`
        let tokens = build_tokens(vec![
            make_token(TokenKind::Int),
            make_ident_token(x_id),
            // Missing semicolon
            make_token(TokenKind::Int),
            make_ident_token(y_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        // First declaration should still produce something.
        let _decl1 = parse_external_declaration(&mut parser);
        // Parser should be able to continue.
        assert!(!parser.is_at_end());
    }

    // -----------------------------------------------------------------------
    // Test: const int *p declaration
    // -----------------------------------------------------------------------
    #[test]
    fn test_const_int_pointer() {
        let mut interner = Interner::new();
        let p_id = interner.intern("p");
        let mut diag = DiagnosticEmitter::new();

        // `const int *p;`
        let tokens = build_tokens(vec![
            make_token(TokenKind::Const),
            make_token(TokenKind::Int),
            make_token(TokenKind::Star),
            make_ident_token(p_id),
            make_token(TokenKind::Semicolon),
        ]);

        let mut parser = make_parser(&tokens, &interner, &mut diag);
        let decl = parse_external_declaration(&mut parser);

        match &decl {
            Declaration::Variable { specifiers, declarators, .. } => {
                assert!(specifiers
                    .type_qualifiers
                    .contains(&TypeQualifier::Const));
                assert_eq!(declarators[0].declarator.pointer.len(), 1);
            }
            _ => panic!("expected Variable declaration (const int ptr)"),
        }
    }
}
