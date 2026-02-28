//! Frontend module for the `bcc` C compiler.
//!
//! This module is the root gateway for the entire compiler frontend, aggregating
//! the three major frontend phases — preprocessing, lexing, and parsing — into a
//! single module namespace. It declares all three submodules and re-exports the
//! key public types that downstream compiler phases (semantic analysis, IR
//! generation, driver pipeline, etc.) consume.
//!
//! # Submodules
//!
//! | Submodule        | Responsibility                                                 |
//! |------------------|----------------------------------------------------------------|
//! | [`preprocessor`] | C11 preprocessor: `#include`, `#define`, conditional compilation, macro expansion |
//! | [`lexer`]        | Tokenizer: converts preprocessed source text into `Vec<Token>` |
//! | [`parser`]       | Recursive-descent parser: produces a complete C11 AST with GCC extensions |
//!
//! # Re-Exports
//!
//! For convenience, the most commonly used types from each submodule are
//! re-exported at the `frontend` level. This allows downstream consumers to
//! write `use crate::frontend::Token` instead of the fully qualified
//! `use crate::frontend::lexer::Token`.
//!
//! ## From [`lexer`]
//! - [`Token`] — The token struct carrying kind, span, and value.
//! - [`TokenKind`] — Enum of 137+ token variants (keywords, operators,
//!   punctuation, literals, identifiers).
//! - [`Lexer`] — The lexer entry point struct with [`Lexer::new()`] and
//!   [`Lexer::tokenize()`] methods.
//!
//! ## From [`parser`]
//! - [`Parser`] — The recursive-descent parser with [`Parser::new()`] and
//!   [`Parser::parse()`] methods.
//! - [`TranslationUnit`] — The AST root node containing all top-level
//!   declarations.
//! - [`Declaration`] — Enum representing all C11 declaration forms.
//! - [`FunctionDef`] — Struct representing a function definition with name,
//!   return type, parameters, body, attributes, and source span.
//! - [`Statement`] — Enum representing all C11 statement forms.
//! - [`Expression`] — Enum representing all C11 expression forms.
//! - [`TypeSpecifier`] — Enum representing all C11 type specifier forms.
//!
//! ## From [`preprocessor`]
//! - [`Preprocessor`] — The preprocessor entry point struct with
//!   [`Preprocessor::new()`] and [`Preprocessor::process()`] methods.
//! - [`PreprocessorOptions`] — Configuration struct for CLI-derived
//!   preprocessing options (`-I`, `-D`, `-U` flags).
//!
//! # Pipeline Integration
//!
//! Per AAP §0.4.1, the frontend pipeline is a strict sequential dependency
//! chain:
//!
//! ```text
//! Raw C Source ──▶ Preprocessor ──▶ Preprocessed Text ──▶ Lexer ──▶ Vec<Token>
//!       ──▶ Parser ──▶ TranslationUnit (untyped AST)
//! ```
//!
//! - **Preprocessor → Lexer**: Preprocessed source text (`String`) with all
//!   includes resolved, macros expanded, and conditionals evaluated.
//! - **Lexer → Parser**: `Vec<Token>` where each `Token` carries its
//!   classification, source span, and optional associated value.
//! - **Parser → Semantic Analyzer** (in `src/sema/`): Untyped AST
//!   (`TranslationUnit` root node containing declarations, definitions, and
//!   type definitions) with source locations preserved on all nodes.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. It consists entirely of module
//! declarations and `pub use` re-exports.
//!
//! # Zero External Dependencies
//!
//! This file imports nothing from external crates. All re-exported types
//! originate from the three submodules which themselves only depend on the
//! Rust standard library and `crate::common` utilities.

// ===========================================================================
// Submodule Declarations
// ===========================================================================
//
// Each `pub mod` declaration corresponds to a subdirectory under
// `src/frontend/` containing the implementation files for that phase:
//
//   src/frontend/preprocessor/  — 6 files (mod.rs, directives.rs, macros.rs,
//                                  conditional.rs, expression.rs, include.rs)
//   src/frontend/lexer/         — 5 files (mod.rs, token.rs, keywords.rs,
//                                  literals.rs, source.rs)
//   src/frontend/parser/        — 7 files (mod.rs, ast.rs, declarations.rs,
//                                  expressions.rs, statements.rs, types.rs,
//                                  gcc_extensions.rs)

/// C11 preprocessor module: directive processing, macro expansion, include
/// resolution, and conditional compilation. Entry point: [`Preprocessor`].
pub mod preprocessor;

/// Tokenizer module: converts preprocessed C source text into a stream of
/// classified tokens. Entry point: [`Lexer`].
pub mod lexer;

/// Recursive-descent parser module: produces a complete AST from the token
/// stream, supporting all C11 constructs and GCC extensions. Entry point:
/// [`Parser`].
pub mod parser;

// ===========================================================================
// Public Type Re-Exports — Lexer Types
// ===========================================================================
//
// These re-exports allow downstream modules to import commonly used lexer
// types directly from `crate::frontend` without navigating into the lexer
// submodule hierarchy.

/// Re-export of the core token type carrying kind, span, and value.
/// See [`lexer::Token`] for full documentation.
pub use lexer::Token;

/// Re-export of the token classification enum with 137+ variants covering
/// all C11 keywords, operators, punctuation, literals, and identifiers.
/// See [`lexer::TokenKind`] for full documentation.
pub use lexer::TokenKind;

/// Re-export of the lexer entry point struct providing [`Lexer::new()`] for
/// construction and [`Lexer::tokenize()`] for producing the token stream.
/// See [`lexer::Lexer`] for full documentation.
pub use lexer::Lexer;

// ===========================================================================
// Public Type Re-Exports — Parser and AST Types
// ===========================================================================
//
// These re-exports provide convenient access to the parser entry point and
// the key AST node types that semantic analysis, IR generation, and other
// downstream phases consume.

/// Re-export of the recursive-descent parser providing [`Parser::new()`] for
/// construction and [`Parser::parse()`] for producing the AST.
/// See [`parser::Parser`] for full documentation.
pub use parser::Parser;

/// Re-export of the AST root node type. A `TranslationUnit` holds all
/// top-level declarations parsed from a C source file.
/// See [`parser::TranslationUnit`] for full documentation.
pub use parser::TranslationUnit;

/// Re-export of the declaration enum representing all C11 declaration forms
/// (variables, functions, typedefs, structs, unions, enums, static asserts).
/// See [`parser::Declaration`] for full documentation.
pub use parser::Declaration;

/// Re-export of the function definition struct carrying the function name,
/// return type, parameter list, body statements, GCC attributes, and source
/// span information.
/// See [`parser::FunctionDef`] for full documentation.
pub use parser::FunctionDef;

/// Re-export of the statement enum representing all C11 statement forms
/// (compound, if/else, for, while, do-while, switch, break, continue,
/// return, goto, labeled, expression statements).
/// See [`parser::Statement`] for full documentation.
pub use parser::Statement;

/// Re-export of the expression enum representing all C11 expression forms
/// with support for GCC extensions (statement expressions, `typeof`,
/// `__builtin_*` intrinsics, computed goto targets).
/// See [`parser::Expression`] for full documentation.
pub use parser::Expression;

/// Re-export of the type specifier enum representing all C11 type specifier
/// forms (base types, struct/union/enum specifiers, typedef names, qualifiers,
/// pointer declarators, array declarators, function pointer declarators).
/// See [`parser::TypeSpecifier`] for full documentation.
pub use parser::TypeSpecifier;

// ===========================================================================
// Public Type Re-Exports — Preprocessor Types
// ===========================================================================
//
// These re-exports provide convenient access to the preprocessor entry point
// and its configuration type.

/// Re-export of the preprocessor entry point struct providing
/// [`Preprocessor::new()`] for construction and [`Preprocessor::process()`]
/// for running the full preprocessing pipeline on C source text.
/// See [`preprocessor::Preprocessor`] for full documentation.
pub use preprocessor::Preprocessor;

/// Re-export of the preprocessor configuration struct populated from CLI
/// flags (`-I`, `-D`, `-U`). Used by the driver module when constructing
/// the preprocessor for a compilation invocation.
/// See [`preprocessor::PreprocessorOptions`] for full documentation.
pub use preprocessor::PreprocessorOptions;
