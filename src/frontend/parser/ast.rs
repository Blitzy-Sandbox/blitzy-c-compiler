//! Complete Abstract Syntax Tree (AST) node type definitions for the `bcc` C compiler.
//!
//! This module defines the entire AST node hierarchy for C11 plus GCC extension
//! constructs. It is the **most foundational file in the parser module** — every
//! other parser file, semantic analysis, and IR generation depends on these types.
//!
//! # Design Principles
//!
//! - **Complete C11 Coverage** — Every C11 language construct has a corresponding
//!   AST node: declarations, statements (18 variants), expressions (32 variants),
//!   type specifiers, and all declarator forms.
//! - **GCC Extension Nodes** — Dedicated node types for `__attribute__`,
//!   `__builtin_*` intrinsics, inline assembly with operand constraints, statement
//!   expressions, `typeof`/`__typeof__`, computed goto, and `__extension__`.
//! - **Source Position Tracking** — Every AST node carries a [`SourceSpan`] field
//!   to preserve source location information through parsing, enabling accurate
//!   diagnostics and DWARF debug info generation.
//! - **Memory Efficiency** — Recursive types use `Box<T>` to keep enum sizes
//!   manageable. Identifiers use [`InternId`] (4 bytes) instead of `String`
//!   (24 bytes) for compact storage and O(1) equality comparison.
//! - **Zero External Dependencies** — Only imports from `std` and `crate::common`.
//! - **No `unsafe` Code** — This module defines data types only.
//!
//! # Integration Points
//!
//! - Consumed by all other parser submodules: `declarations.rs`, `expressions.rs`,
//!   `statements.rs`, `types.rs`, `gcc_extensions.rs`.
//! - Consumed by `src/sema/mod.rs` for semantic analysis.
//! - Consumed by `src/ir/builder.rs` for IR generation.
//! - Per AAP §0.4.1: "Parser → Semantic Analyzer: Untyped AST (TranslationUnit
//!   root node containing declarations, definitions, and type definitions)."

use crate::common::source_map::SourceSpan;
use crate::common::intern::InternId;

// ===========================================================================
// TranslationUnit — AST Root
// ===========================================================================

/// Root AST node representing an entire C translation unit (source file).
///
/// A translation unit is a sequence of top-level declarations: variable
/// declarations, function definitions, typedef declarations, struct/union/enum
/// definitions, and `_Static_assert` declarations.
#[derive(Debug, Clone)]
pub struct TranslationUnit {
    /// Top-level declarations (variables, functions, typedefs, structs, etc.).
    pub declarations: Vec<Declaration>,
    /// Source span covering the entire file.
    pub span: SourceSpan,
}

// ===========================================================================
// Declaration Types
// ===========================================================================

/// A top-level or block-level declaration in C.
///
/// Covers all C11 declaration forms: variable declarations with optional
/// initializers, function definitions with bodies, typedef declarations,
/// `_Static_assert`, and empty declarations.
#[derive(Debug, Clone)]
pub enum Declaration {
    /// Variable declaration: `int x = 5;`, `int x, y, z;`
    Variable {
        specifiers: DeclSpecifiers,
        declarators: Vec<InitDeclarator>,
        span: SourceSpan,
    },
    /// Function definition with body: `int main(void) { return 0; }`
    Function(Box<FunctionDef>),
    /// Typedef declaration: `typedef int MyInt;`
    Typedef {
        specifiers: DeclSpecifiers,
        declarators: Vec<Declarator>,
        span: SourceSpan,
    },
    /// `_Static_assert` declaration.
    StaticAssert {
        expr: Box<Expression>,
        message: String,
        span: SourceSpan,
    },
    /// Empty declaration (just `;`).
    Empty {
        span: SourceSpan,
    },
}

/// A function definition: declaration specifiers, declarator (name + params),
/// body (compound statement), optional GCC attributes, and source span.
#[derive(Debug, Clone)]
pub struct FunctionDef {
    /// Declaration specifiers (storage class, type qualifiers, return type, etc.).
    pub specifiers: DeclSpecifiers,
    /// The function declarator (name, parameters, pointer modifiers).
    pub declarator: Declarator,
    /// Function body — always a compound statement `{ ... }`.
    pub body: Box<Statement>,
    /// Optional GCC `__attribute__` annotations on the function.
    pub attributes: Vec<GccAttribute>,
    /// Source span covering the entire function definition.
    pub span: SourceSpan,
}

/// Declaration specifiers: storage class, type qualifiers, type specifier,
/// function specifiers, and optional GCC attributes.
///
/// These can appear in any order in the source: `static const int`,
/// `int static const`, and `const static int` are all equivalent.
#[derive(Debug, Clone)]
pub struct DeclSpecifiers {
    /// Optional storage class: `static`, `extern`, `auto`, `register`,
    /// `_Thread_local`.
    pub storage_class: Option<StorageClass>,
    /// Type qualifiers: `const`, `volatile`, `restrict`, `_Atomic`.
    pub type_qualifiers: Vec<TypeQualifier>,
    /// The base type specifier.
    pub type_specifier: TypeSpecifier,
    /// Function specifiers: `inline`, `_Noreturn`.
    pub function_specifiers: Vec<FunctionSpecifier>,
    /// GCC `__attribute__` annotations.
    pub attributes: Vec<GccAttribute>,
    /// Source span covering all specifiers.
    pub span: SourceSpan,
}

/// A declarator paired with an optional initializer: `x = 5` or just `x`.
#[derive(Debug, Clone)]
pub struct InitDeclarator {
    /// The declarator (name and modifiers).
    pub declarator: Declarator,
    /// Optional initializer expression or compound initializer.
    pub initializer: Option<Initializer>,
    /// Source span.
    pub span: SourceSpan,
}

// ===========================================================================
// Declarator Types — C's Most Complex Syntax
// ===========================================================================

/// A C declarator: optional pointer prefix, direct declarator, and optional
/// GCC attributes.
///
/// Declarators name the entity being declared and may include pointer, array,
/// and function-pointer modifiers. For example, in `int *const *volatile p`,
/// the declarator is `*const *volatile p`.
#[derive(Debug, Clone)]
pub struct Declarator {
    /// Chain of pointer modifiers, each with its own qualifiers.
    pub pointer: Vec<Pointer>,
    /// The direct part of the declarator (identifier, array, function, etc.).
    pub direct: DirectDeclarator,
    /// GCC `__attribute__` annotations on the declarator.
    pub attributes: Vec<GccAttribute>,
    /// Source span.
    pub span: SourceSpan,
}

/// A single pointer indirection level with optional type qualifiers.
///
/// For example, `*const` is `Pointer { qualifiers: [Const] }`.
#[derive(Debug, Clone)]
pub struct Pointer {
    /// Type qualifiers applied to this pointer level.
    pub qualifiers: Vec<TypeQualifier>,
}

/// The direct part of a declarator (without pointer prefixes).
#[derive(Debug, Clone)]
pub enum DirectDeclarator {
    /// Simple identifier: `x`.
    Identifier(InternId),
    /// Parenthesized declarator: `(declarator)` — used for function pointers
    /// like `(*fp)(int, int)`.
    Parenthesized(Box<Declarator>),
    /// Array declarator: `base[size]`.
    Array {
        base: Box<DirectDeclarator>,
        size: ArraySize,
        qualifiers: Vec<TypeQualifier>,
    },
    /// Function declarator: `base(params)`.
    Function {
        base: Box<DirectDeclarator>,
        params: ParamList,
    },
    /// Abstract (no name) — used in type names for casts, sizeof.
    Abstract,
}

/// An abstract declarator (declarator without an identifier name), used in
/// type names for casts, sizeof, _Alignof, function parameters, and _Generic
/// associations.
#[derive(Debug, Clone)]
pub struct AbstractDeclarator {
    /// Chain of pointer modifiers.
    pub pointer: Vec<Pointer>,
    /// Optional direct abstract declarator (array/function suffixes).
    pub direct: Option<DirectAbstractDeclarator>,
    /// Source span.
    pub span: SourceSpan,
}

/// The direct part of an abstract declarator (no identifier).
#[derive(Debug, Clone)]
pub enum DirectAbstractDeclarator {
    /// Parenthesized abstract declarator.
    Parenthesized(Box<AbstractDeclarator>),
    /// Array suffix on an abstract declarator.
    Array {
        base: Option<Box<DirectAbstractDeclarator>>,
        size: ArraySize,
        qualifiers: Vec<TypeQualifier>,
    },
    /// Function suffix on an abstract declarator.
    Function {
        base: Option<Box<DirectAbstractDeclarator>>,
        params: ParamList,
    },
}

/// Array size specification in a declarator.
#[derive(Debug, Clone)]
pub enum ArraySize {
    /// Fixed-size array: `[expr]`.
    Fixed(Box<Expression>),
    /// Unspecified size: `[]`.
    Unspecified,
    /// C99 variable-length array placeholder: `[*]`.
    VLA,
    /// C99 `static` qualifier in function parameter: `[static expr]`.
    Static(Box<Expression>),
}

/// A function parameter list: parameters and whether the function is variadic.
#[derive(Debug, Clone)]
pub struct ParamList {
    /// Individual parameter declarations.
    pub params: Vec<ParamDeclaration>,
    /// Whether the function accepts variadic arguments (`...`).
    pub variadic: bool,
    /// Source span covering the entire parameter list.
    pub span: SourceSpan,
}

/// A single function parameter declaration.
#[derive(Debug, Clone)]
pub struct ParamDeclaration {
    /// Declaration specifiers for this parameter.
    pub specifiers: DeclSpecifiers,
    /// Optional declarator (None for abstract/unnamed parameters like `int`).
    pub declarator: Option<Declarator>,
    /// Source span.
    pub span: SourceSpan,
}

// ===========================================================================
// Type Specifier Types
// ===========================================================================

/// Comprehensive type specifier covering all C11 types plus GCC extensions.
///
/// This enum represents the base type in a declaration, including basic types,
/// signed/unsigned modifiers, C11 special types (`_Bool`, `_Complex`, `_Atomic`),
/// composite types (struct, union, enum), typedef references, GCC `typeof`, and
/// an error recovery placeholder.
#[derive(Debug, Clone)]
pub enum TypeSpecifier {
    // === Basic types ===
    /// `void`
    Void,
    /// `char`
    Char,
    /// `short` / `short int`
    Short,
    /// `int`
    Int,
    /// `long` / `long int`
    Long,
    /// `long long` / `long long int`
    LongLong,
    /// `float`
    Float,
    /// `double`
    Double,
    /// `long double`
    LongDouble,

    // === Signed/Unsigned modifiers ===
    /// `signed char`, `signed int`, etc. Wraps the inner type.
    Signed(Box<TypeSpecifier>),
    /// `unsigned char`, `unsigned int`, etc. Wraps the inner type.
    Unsigned(Box<TypeSpecifier>),

    // === C11 special types ===
    /// `_Bool`
    Bool,
    /// `_Complex float`, `_Complex double`. Wraps the inner float type.
    Complex(Box<TypeSpecifier>),
    /// `_Atomic(type)`. Wraps the inner type.
    Atomic(Box<TypeSpecifier>),

    // === Composite type definitions ===
    /// `struct tag { ... }` — struct definition with body.
    Struct(StructDef),
    /// `union tag { ... }` — union definition with body.
    Union(UnionDef),
    /// `enum tag { ... }` — enum definition with body.
    Enum(EnumDef),

    // === References to previously-defined composite types ===
    /// `struct tag` — reference to a previously defined struct.
    StructRef { tag: InternId, span: SourceSpan },
    /// `union tag` — reference to a previously defined union.
    UnionRef { tag: InternId, span: SourceSpan },
    /// `enum tag` — reference to a previously defined enum.
    EnumRef { tag: InternId, span: SourceSpan },

    // === Typedef name reference ===
    /// An identifier that was previously declared via `typedef`.
    TypedefName { name: InternId, span: SourceSpan },

    // === GCC Extensions ===
    /// `typeof(expr)` — type of an expression.
    Typeof { expr: Box<Expression>, span: SourceSpan },
    /// `typeof(type)` — echoes a type (useful in macros).
    TypeofType { type_name: Box<TypeSpecifier>, span: SourceSpan },

    // === Qualified type (wraps inner type with qualifiers) ===
    /// A type with qualifiers applied, e.g., `const int`.
    Qualified {
        qualifiers: Vec<TypeQualifier>,
        inner: Box<TypeSpecifier>,
    },

    // === Error recovery placeholder ===
    /// Placeholder used when the parser encounters an invalid type specifier
    /// and performs error recovery.
    Error,
}

/// C storage class specifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    /// `static`
    Static,
    /// `extern`
    Extern,
    /// `auto`
    Auto,
    /// `register`
    Register,
    /// `_Thread_local`
    ThreadLocal,
}

/// C type qualifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeQualifier {
    /// `const`
    Const,
    /// `volatile`
    Volatile,
    /// `restrict`
    Restrict,
    /// `_Atomic` (used as a type qualifier, not the type specifier form).
    Atomic,
}

/// C function specifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionSpecifier {
    /// `inline`
    Inline,
    /// `_Noreturn`
    Noreturn,
}

// ===========================================================================
// Struct / Union / Enum Definitions
// ===========================================================================

/// A struct definition: `struct tag { members... }`.
#[derive(Debug, Clone)]
pub struct StructDef {
    /// Optional tag name. `None` for anonymous structs.
    pub tag: Option<InternId>,
    /// Struct member declarations.
    pub members: Vec<StructMember>,
    /// GCC `__attribute__` annotations on the struct.
    pub attributes: Vec<GccAttribute>,
    /// Source span.
    pub span: SourceSpan,
}

/// A member within a struct or union definition.
#[derive(Debug, Clone)]
pub enum StructMember {
    /// Named field declaration with optional bit-width.
    Field {
        specifiers: DeclSpecifiers,
        declarators: Vec<StructFieldDeclarator>,
        span: SourceSpan,
    },
    /// C11 anonymous struct/union member: `struct { int x; };` inside another
    /// struct or union.
    Anonymous {
        type_spec: TypeSpecifier,
        span: SourceSpan,
    },
    /// `_Static_assert` within a struct/union body.
    StaticAssert {
        expr: Box<Expression>,
        message: String,
        span: SourceSpan,
    },
}

/// A single struct field declarator, optionally with a bit-field width.
#[derive(Debug, Clone)]
pub struct StructFieldDeclarator {
    /// The field declarator. `None` for unnamed bit-fields (e.g., `int : 3;`).
    pub declarator: Option<Declarator>,
    /// Optional bit-field width expression.
    pub bit_width: Option<Box<Expression>>,
    /// Source span.
    pub span: SourceSpan,
}

/// A union definition: `union tag { members... }`.
///
/// Shares the same member representation as structs.
#[derive(Debug, Clone)]
pub struct UnionDef {
    /// Optional tag name. `None` for anonymous unions.
    pub tag: Option<InternId>,
    /// Union member declarations.
    pub members: Vec<StructMember>,
    /// GCC `__attribute__` annotations on the union.
    pub attributes: Vec<GccAttribute>,
    /// Source span.
    pub span: SourceSpan,
}

/// An enum definition: `enum tag { A, B = 1, C }`.
#[derive(Debug, Clone)]
pub struct EnumDef {
    /// Optional tag name. `None` for anonymous enums.
    pub tag: Option<InternId>,
    /// Enumeration variants.
    pub variants: Vec<EnumVariant>,
    /// GCC `__attribute__` annotations on the enum.
    pub attributes: Vec<GccAttribute>,
    /// Source span.
    pub span: SourceSpan,
}

/// A single enumeration variant: `NAME` or `NAME = value`.
#[derive(Debug, Clone)]
pub struct EnumVariant {
    /// The variant name.
    pub name: InternId,
    /// Optional explicit value expression.
    pub value: Option<Box<Expression>>,
    /// GCC `__attribute__` annotations on the variant.
    pub attributes: Vec<GccAttribute>,
    /// Source span.
    pub span: SourceSpan,
}

// ===========================================================================
// Initializer Types
// ===========================================================================

/// An initializer for a variable declaration.
#[derive(Debug, Clone)]
pub enum Initializer {
    /// Simple expression initializer: `= expr`.
    Expression(Box<Expression>),
    /// Compound (brace-enclosed) initializer: `= { item1, item2, ... }`.
    Compound {
        items: Vec<DesignatedInitializer>,
        span: SourceSpan,
    },
}

/// A single item in a compound initializer, optionally with designators.
///
/// For example, `.x = 1` has designator `Field("x")` and initializer `1`.
/// Plain `1` has an empty designator list.
#[derive(Debug, Clone)]
pub struct DesignatedInitializer {
    /// Zero or more designators (field, index, or GCC range).
    pub designators: Vec<Designator>,
    /// The initializer value.
    pub initializer: Initializer,
    /// Source span.
    pub span: SourceSpan,
}

/// A designator in a designated initializer.
#[derive(Debug, Clone)]
pub enum Designator {
    /// Array index designator: `[expr]`.
    Index(Box<Expression>),
    /// Struct/union field designator: `.field`.
    Field(InternId),
    /// GCC range designator: `[low ... high]`.
    Range(Box<Expression>, Box<Expression>),
}

// ===========================================================================
// Statement Types (18 Variants — exceeds the 15+ requirement)
// ===========================================================================

/// A C statement. This enum has 18 variants, covering all C11 statement types
/// plus GCC extensions (inline assembly and computed goto).
#[derive(Debug, Clone)]
pub enum Statement {
    /// Compound statement (block): `{ items... }`.
    Compound {
        items: Vec<BlockItem>,
        span: SourceSpan,
    },
    /// If statement: `if (cond) then_branch [else else_branch]`.
    If {
        condition: Box<Expression>,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
        span: SourceSpan,
    },
    /// For loop: `for (init; cond; incr) body`.
    For {
        init: Option<Box<ForInit>>,
        condition: Option<Box<Expression>>,
        increment: Option<Box<Expression>>,
        body: Box<Statement>,
        span: SourceSpan,
    },
    /// While loop: `while (cond) body`.
    While {
        condition: Box<Expression>,
        body: Box<Statement>,
        span: SourceSpan,
    },
    /// Do-while loop: `do body while (cond);`.
    DoWhile {
        body: Box<Statement>,
        condition: Box<Expression>,
        span: SourceSpan,
    },
    /// Switch statement: `switch (expr) body`.
    Switch {
        expr: Box<Expression>,
        body: Box<Statement>,
        span: SourceSpan,
    },
    /// Case label: `case value: body` or GCC range `case low ... high: body`.
    Case {
        value: Box<Expression>,
        /// GCC case range extension: `case 1 ... 5:`.
        range_end: Option<Box<Expression>>,
        body: Box<Statement>,
        span: SourceSpan,
    },
    /// Default label: `default: body`.
    Default {
        body: Box<Statement>,
        span: SourceSpan,
    },
    /// Break statement: `break;`.
    Break { span: SourceSpan },
    /// Continue statement: `continue;`.
    Continue { span: SourceSpan },
    /// Return statement: `return [expr];`.
    Return {
        value: Option<Box<Expression>>,
        span: SourceSpan,
    },
    /// Goto statement: `goto label;`.
    Goto {
        label: InternId,
        span: SourceSpan,
    },
    /// Labeled statement: `label: body`.
    Labeled {
        label: InternId,
        body: Box<Statement>,
        /// GCC `__attribute__` annotations on the label.
        attributes: Vec<GccAttribute>,
        span: SourceSpan,
    },
    /// Expression statement: `expr;`.
    Expression {
        expr: Box<Expression>,
        span: SourceSpan,
    },
    /// Empty/null statement: `;`.
    Null { span: SourceSpan },
    /// Block-level declaration (C99): `int x = 5;` inside a compound block.
    Declaration(Box<Declaration>),
    /// Inline assembly statement (GCC extension).
    Asm(AsmStatement),
    /// Computed goto (GCC extension): `goto *expr;`.
    ComputedGoto {
        target: Box<Expression>,
        span: SourceSpan,
    },
}

/// A block item within a compound statement — either a declaration or a
/// statement.
#[derive(Debug, Clone)]
pub enum BlockItem {
    /// A declaration within a block.
    Declaration(Box<Declaration>),
    /// A statement within a block.
    Statement(Statement),
}

/// The initialization clause of a `for` loop.
#[derive(Debug, Clone)]
pub enum ForInit {
    /// C99 declaration in for-init: `for (int i = 0; ...)`.
    Declaration(Box<Declaration>),
    /// Expression in for-init: `for (i = 0; ...)`.
    Expression(Box<Expression>),
}

// ===========================================================================
// Expression Types (32 Variants — exceeds the 25+ requirement)
// ===========================================================================

/// A C expression. This enum has 32 variants, covering all C11 expression types
/// plus GCC extensions (statement expressions, label addresses, builtins, etc.).
#[derive(Debug, Clone)]
pub enum Expression {
    // === Literals ===
    /// Integer literal: `42`, `0xFF`, `0b1010`, `100ULL`.
    IntegerLiteral {
        value: u128,
        suffix: IntSuffix,
        base: NumericBase,
        span: SourceSpan,
    },
    /// Floating-point literal: `3.14`, `1.0f`, `2.0L`.
    FloatLiteral {
        value: f64,
        suffix: FloatSuffix,
        span: SourceSpan,
    },
    /// String literal: `"hello"`, `L"wide"`, `u8"utf8"`.
    StringLiteral {
        value: String,
        prefix: StringPrefix,
        span: SourceSpan,
    },
    /// Character literal: `'a'`, `L'w'`, `u'x'`.
    CharLiteral {
        value: u32,
        prefix: CharPrefix,
        span: SourceSpan,
    },

    // === Identifier ===
    /// An identifier reference: variable, function name, enum constant.
    Identifier {
        name: InternId,
        span: SourceSpan,
    },

    // === Binary operations ===
    /// Binary operation: `left op right`.
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
        span: SourceSpan,
    },

    // === Unary prefix operations ===
    /// Unary prefix operation: `op operand` (includes `++`, `--`, `+`, `-`,
    /// `~`, `!`, `*`, `&`).
    UnaryPrefix {
        op: UnaryOp,
        operand: Box<Expression>,
        span: SourceSpan,
    },

    // === Postfix operations ===
    /// Post-increment: `operand++`.
    PostIncrement {
        operand: Box<Expression>,
        span: SourceSpan,
    },
    /// Post-decrement: `operand--`.
    PostDecrement {
        operand: Box<Expression>,
        span: SourceSpan,
    },

    // === Function call ===
    /// Function call: `callee(arg1, arg2, ...)`.
    Call {
        callee: Box<Expression>,
        args: Vec<Expression>,
        span: SourceSpan,
    },

    // === Array subscript ===
    /// Array subscript: `array[index]`.
    Subscript {
        array: Box<Expression>,
        index: Box<Expression>,
        span: SourceSpan,
    },

    // === Member access ===
    /// Direct member access: `object.member`.
    MemberAccess {
        object: Box<Expression>,
        member: InternId,
        span: SourceSpan,
    },
    /// Pointer member access: `pointer->member`.
    ArrowAccess {
        pointer: Box<Expression>,
        member: InternId,
        span: SourceSpan,
    },

    // === Assignment ===
    /// Assignment: `target op= value`.
    Assignment {
        op: AssignmentOp,
        target: Box<Expression>,
        value: Box<Expression>,
        span: SourceSpan,
    },

    // === Ternary ===
    /// Ternary conditional: `condition ? then_expr : else_expr`.
    Ternary {
        condition: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>,
        span: SourceSpan,
    },

    // === Comma ===
    /// Comma expression: `expr1, expr2, ...` (evaluates left-to-right, yields
    /// rightmost value).
    Comma {
        exprs: Vec<Expression>,
        span: SourceSpan,
    },

    // === Cast ===
    /// Type cast: `(type_name) operand`.
    Cast {
        type_name: Box<TypeName>,
        operand: Box<Expression>,
        span: SourceSpan,
    },

    // === Sizeof / Alignof ===
    /// `sizeof expr` — size of an expression's type.
    SizeofExpr {
        expr: Box<Expression>,
        span: SourceSpan,
    },
    /// `sizeof(type_name)` — size of a named type.
    SizeofType {
        type_name: Box<TypeName>,
        span: SourceSpan,
    },
    /// `_Alignof(type_name)` — alignment of a type.
    Alignof {
        type_name: Box<TypeName>,
        span: SourceSpan,
    },

    // === C11 _Generic selection ===
    /// `_Generic(controlling, type1: expr1, type2: expr2, default: exprD)`.
    Generic {
        controlling: Box<Expression>,
        associations: Vec<GenericAssociation>,
        span: SourceSpan,
    },

    // === Compound literal (C99) ===
    /// Compound literal: `(type_name){ initializer }`.
    CompoundLiteral {
        type_name: Box<TypeName>,
        initializer: Initializer,
        span: SourceSpan,
    },

    // === GCC Extensions ===
    /// GCC statement expression: `({ int x = 5; x + 1; })`.
    StatementExpr {
        body: Box<Statement>,
        span: SourceSpan,
    },
    /// GCC label address: `&&label`.
    LabelAddr {
        label: InternId,
        span: SourceSpan,
    },
    /// GCC `__extension__` prefix: `__extension__ expr`.
    Extension {
        expr: Box<Expression>,
        span: SourceSpan,
    },
    /// GCC `__builtin_va_arg(ap, type)`.
    BuiltinVaArg {
        ap: Box<Expression>,
        type_name: Box<TypeName>,
        span: SourceSpan,
    },
    /// GCC `__builtin_offsetof(type, member)`.
    BuiltinOffsetof {
        type_name: Box<TypeName>,
        member: InternId,
        span: SourceSpan,
    },
    /// GCC `__builtin_va_start(ap, param)`.
    BuiltinVaStart {
        ap: Box<Expression>,
        param: Box<Expression>,
        span: SourceSpan,
    },
    /// GCC `__builtin_va_end(ap)`.
    BuiltinVaEnd {
        ap: Box<Expression>,
        span: SourceSpan,
    },
    /// GCC `__builtin_va_copy(dest, src)`.
    BuiltinVaCopy {
        dest: Box<Expression>,
        src: Box<Expression>,
        span: SourceSpan,
    },

    // === Parenthesized expression ===
    /// Parenthesized expression: `(inner)`. Useful for preserving source
    /// structure in pretty-printing and diagnostics.
    Paren {
        inner: Box<Expression>,
        span: SourceSpan,
    },

    // === Error recovery placeholder ===
    /// Placeholder for error recovery when an expression cannot be parsed.
    Error {
        span: SourceSpan,
    },
}

/// A type name, used in casts `(type_name)expr`, sizeof `sizeof(type_name)`,
/// `_Alignof`, `_Generic` associations, and `__builtin_va_arg`.
#[derive(Debug, Clone)]
pub struct TypeName {
    /// Declaration specifiers for the type name (type specifier + qualifiers,
    /// no storage class or function specifiers).
    pub specifiers: DeclSpecifiers,
    /// Optional abstract declarator for pointer/array/function modifiers.
    pub abstract_declarator: Option<AbstractDeclarator>,
    /// Source span.
    pub span: SourceSpan,
}

// ===========================================================================
// Operator Enums
// ===========================================================================

/// Binary operators covering all 18 C binary operations (arithmetic, bitwise,
/// logical, comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Mod,
    /// `&`
    BitwiseAnd,
    /// `|`
    BitwiseOr,
    /// `^`
    BitwiseXor,
    /// `<<`
    ShiftLeft,
    /// `>>`
    ShiftRight,
    /// `&&`
    LogicalAnd,
    /// `||`
    LogicalOr,
    /// `==`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,
}

/// Unary prefix operators covering all 8 C unary prefix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Unary `+`
    Plus,
    /// Unary `-` (negation)
    Negate,
    /// `~` (bitwise NOT)
    BitwiseNot,
    /// `!` (logical NOT)
    LogicalNot,
    /// `*` (pointer dereference)
    Dereference,
    /// `&` (address-of)
    AddressOf,
    /// `++` prefix increment
    PreIncrement,
    /// `--` prefix decrement
    PreDecrement,
}

/// Assignment operators covering all 11 C assignment forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentOp {
    /// `=`
    Assign,
    /// `+=`
    AddAssign,
    /// `-=`
    SubAssign,
    /// `*=`
    MulAssign,
    /// `/=`
    DivAssign,
    /// `%=`
    ModAssign,
    /// `&=`
    AndAssign,
    /// `|=`
    OrAssign,
    /// `^=`
    XorAssign,
    /// `<<=`
    ShlAssign,
    /// `>>=`
    ShrAssign,
}

// ===========================================================================
// Literal Type Enums
// ===========================================================================

/// Numeric base for integer literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericBase {
    /// Base-10: `42`
    Decimal,
    /// Base-16: `0xFF`
    Hexadecimal,
    /// Base-8: `0777`
    Octal,
    /// Base-2: `0b1010`
    Binary,
}

/// Integer literal suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntSuffix {
    /// No suffix: `42`
    None,
    /// `U` / `u`: `42U`
    Unsigned,
    /// `L` / `l`: `42L`
    Long,
    /// `UL` / `ul`: `42UL`
    ULong,
    /// `LL` / `ll`: `42LL`
    LongLong,
    /// `ULL` / `ull`: `42ULL`
    ULongLong,
}

/// Floating-point literal suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSuffix {
    /// No suffix: `3.14` (double)
    None,
    /// `f` / `F`: `3.14f` (float)
    Float,
    /// `l` / `L`: `3.14L` (long double)
    Long,
}

/// String literal prefix for character encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPrefix {
    /// No prefix: `"hello"` (char)
    None,
    /// `L`: `L"hello"` (wchar_t)
    Wide,
    /// `u8`: `u8"hello"` (UTF-8)
    Utf8,
    /// `u`: `u"hello"` (char16_t)
    Utf16,
    /// `U`: `U"hello"` (char32_t)
    Utf32,
}

/// Character literal prefix for character encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharPrefix {
    /// No prefix: `'a'` (char)
    None,
    /// `L`: `L'a'` (wchar_t)
    Wide,
    /// `u`: `u'a'` (char16_t)
    Utf16,
    /// `U`: `U'a'` (char32_t)
    Utf32,
}

// ===========================================================================
// GCC Extension AST Types
// ===========================================================================

/// A GCC `__attribute__` annotation: `__attribute__((name(args...)))`.
#[derive(Debug, Clone)]
pub struct GccAttribute {
    /// The attribute name (e.g., `packed`, `aligned`, `section`).
    pub name: InternId,
    /// Arguments to the attribute.
    pub args: Vec<AttributeArg>,
    /// Source span.
    pub span: SourceSpan,
}

/// An argument to a GCC attribute.
#[derive(Debug, Clone)]
pub enum AttributeArg {
    /// An identifier argument: `printf` in `format(printf, 1, 2)`.
    Identifier(InternId),
    /// A string literal argument: `".text"` in `section(".text")`.
    String(String),
    /// An integer argument: `16` in `aligned(16)`.
    Integer(i128),
    /// An arbitrary expression argument.
    Expression(Expression),
    /// A type name argument (used in some GCC builtins).
    Type(TypeName),
}

/// GCC inline assembly statement: `asm [volatile] [inline] [goto] (template
/// : outputs : inputs : clobbers : goto_labels)`.
#[derive(Debug, Clone)]
pub struct AsmStatement {
    /// Whether `volatile` / `__volatile__` qualifier is present.
    pub is_volatile: bool,
    /// Whether `inline` qualifier is present.
    pub is_inline: bool,
    /// Whether `goto` qualifier is present.
    pub is_goto: bool,
    /// The assembly template string.
    pub template: String,
    /// Output operands.
    pub outputs: Vec<AsmOperand>,
    /// Input operands.
    pub inputs: Vec<AsmOperand>,
    /// Clobber list (register/memory clobbers).
    pub clobbers: Vec<String>,
    /// Goto labels (for `asm goto`).
    pub goto_labels: Vec<InternId>,
    /// Source span.
    pub span: SourceSpan,
}

/// An operand in a GCC inline assembly statement.
#[derive(Debug, Clone)]
pub struct AsmOperand {
    /// Optional symbolic name: `[name]` in `[name] "constraint" (expr)`.
    pub symbolic_name: Option<InternId>,
    /// Constraint string: `"=r"`, `"r"`, `"m"`, `"+r"`, etc.
    pub constraint: String,
    /// The C expression for this operand.
    pub expr: Expression,
    /// Source span.
    pub span: SourceSpan,
}

// ===========================================================================
// C11 _Generic Selection
// ===========================================================================

/// An association in a C11 `_Generic` selection expression.
#[derive(Debug, Clone)]
pub enum GenericAssociation {
    /// Type association: `type_name : expr`.
    Type {
        type_name: TypeName,
        expr: Expression,
        span: SourceSpan,
    },
    /// Default association: `default : expr`.
    Default {
        expr: Expression,
        span: SourceSpan,
    },
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::{SourceLocation, SourceSpan};
    use crate::common::intern::InternId;

    /// Helper: create a dummy SourceSpan for testing.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Helper: create a dummy InternId for testing.
    fn dummy_id(raw: u32) -> InternId {
        InternId::from_raw(raw)
    }

    // -----------------------------------------------------------------------
    // TranslationUnit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_translation_unit_empty() {
        let tu = TranslationUnit {
            declarations: vec![],
            span: dummy_span(),
        };
        assert!(tu.declarations.is_empty());
        assert_eq!(tu.span.start, tu.span.end);
    }

    #[test]
    fn test_translation_unit_with_variable_declaration() {
        let decl = Declaration::Variable {
            specifiers: DeclSpecifiers {
                storage_class: None,
                type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Int,
                function_specifiers: vec![],
                attributes: vec![],
                span: dummy_span(),
            },
            declarators: vec![InitDeclarator {
                declarator: Declarator {
                    pointer: vec![],
                    direct: DirectDeclarator::Identifier(dummy_id(0)),
                    attributes: vec![],
                    span: dummy_span(),
                },
                initializer: None,
                span: dummy_span(),
            }],
            span: dummy_span(),
        };

        let tu = TranslationUnit {
            declarations: vec![decl],
            span: dummy_span(),
        };
        assert_eq!(tu.declarations.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Declaration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_declaration_variable() {
        let decl = Declaration::Variable {
            specifiers: DeclSpecifiers {
                storage_class: Some(StorageClass::Static),
                type_qualifiers: vec![TypeQualifier::Const],
                type_specifier: TypeSpecifier::Int,
                function_specifiers: vec![],
                attributes: vec![],
                span: dummy_span(),
            },
            declarators: vec![],
            span: dummy_span(),
        };
        // Verify it can be constructed and cloned
        let _cloned = decl.clone();
        // Verify Debug formatting works
        let _debug = format!("{:?}", decl);
    }

    #[test]
    fn test_declaration_empty() {
        let decl = Declaration::Empty { span: dummy_span() };
        let _debug = format!("{:?}", decl);
    }

    #[test]
    fn test_declaration_static_assert() {
        let decl = Declaration::StaticAssert {
            expr: Box::new(Expression::IntegerLiteral {
                value: 1,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
                span: dummy_span(),
            }),
            message: "must be true".to_string(),
            span: dummy_span(),
        };
        let _debug = format!("{:?}", decl);
    }

    // -----------------------------------------------------------------------
    // FunctionDef tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_function_def_construction() {
        let func = FunctionDef {
            specifiers: DeclSpecifiers {
                storage_class: None,
                type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Int,
                function_specifiers: vec![],
                attributes: vec![],
                span: dummy_span(),
            },
            declarator: Declarator {
                pointer: vec![],
                direct: DirectDeclarator::Function {
                    base: Box::new(DirectDeclarator::Identifier(dummy_id(1))),
                    params: ParamList {
                        params: vec![],
                        variadic: false,
                        span: dummy_span(),
                    },
                },
                attributes: vec![],
                span: dummy_span(),
            },
            body: Box::new(Statement::Compound {
                items: vec![BlockItem::Statement(Statement::Return {
                    value: Some(Box::new(Expression::IntegerLiteral {
                        value: 0,
                        suffix: IntSuffix::None,
                        base: NumericBase::Decimal,
                        span: dummy_span(),
                    })),
                    span: dummy_span(),
                })],
                span: dummy_span(),
            }),
            attributes: vec![],
            span: dummy_span(),
        };
        assert!(func.attributes.is_empty());
        let _cloned = func.clone();
    }

    // -----------------------------------------------------------------------
    // Statement variant count test
    // -----------------------------------------------------------------------

    #[test]
    fn test_statement_variant_count() {
        // Verify all 18 statement variants exist by constructing each one.
        let _compound = Statement::Compound { items: vec![], span: dummy_span() };
        let _if_s = Statement::If {
            condition: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            then_branch: Box::new(Statement::Null { span: dummy_span() }),
            else_branch: None,
            span: dummy_span(),
        };
        let _for_s = Statement::For {
            init: None,
            condition: None,
            increment: None,
            body: Box::new(Statement::Null { span: dummy_span() }),
            span: dummy_span(),
        };
        let _while_s = Statement::While {
            condition: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            body: Box::new(Statement::Null { span: dummy_span() }),
            span: dummy_span(),
        };
        let _do_while = Statement::DoWhile {
            body: Box::new(Statement::Null { span: dummy_span() }),
            condition: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            span: dummy_span(),
        };
        let _switch = Statement::Switch {
            expr: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            body: Box::new(Statement::Null { span: dummy_span() }),
            span: dummy_span(),
        };
        let _case = Statement::Case {
            value: Box::new(Expression::IntegerLiteral { value: 1, suffix: IntSuffix::None, base: NumericBase::Decimal, span: dummy_span() }),
            range_end: None,
            body: Box::new(Statement::Null { span: dummy_span() }),
            span: dummy_span(),
        };
        let _default = Statement::Default {
            body: Box::new(Statement::Null { span: dummy_span() }),
            span: dummy_span(),
        };
        let _break = Statement::Break { span: dummy_span() };
        let _continue = Statement::Continue { span: dummy_span() };
        let _return = Statement::Return { value: None, span: dummy_span() };
        let _goto = Statement::Goto { label: dummy_id(0), span: dummy_span() };
        let _labeled = Statement::Labeled {
            label: dummy_id(0),
            body: Box::new(Statement::Null { span: dummy_span() }),
            attributes: vec![],
            span: dummy_span(),
        };
        let _expr_stmt = Statement::Expression {
            expr: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            span: dummy_span(),
        };
        let _null = Statement::Null { span: dummy_span() };
        let _decl_stmt = Statement::Declaration(Box::new(Declaration::Empty { span: dummy_span() }));
        let _asm = Statement::Asm(AsmStatement {
            is_volatile: false,
            is_inline: false,
            is_goto: false,
            template: "nop".to_string(),
            outputs: vec![],
            inputs: vec![],
            clobbers: vec![],
            goto_labels: vec![],
            span: dummy_span(),
        });
        let _computed_goto = Statement::ComputedGoto {
            target: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            span: dummy_span(),
        };
        // 18 variants total — well above the 15+ requirement.
    }

    // -----------------------------------------------------------------------
    // Expression variant count test
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_variant_count() {
        // Verify all 32 expression variants exist by constructing each one.
        let s = dummy_span();
        let id_expr = || Box::new(Expression::Identifier { name: dummy_id(0), span: s });
        let type_name = || Box::new(TypeName {
            specifiers: DeclSpecifiers {
                storage_class: None,
                type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Int,
                function_specifiers: vec![],
                attributes: vec![],
                span: s,
            },
            abstract_declarator: None,
            span: s,
        });

        let _variants: Vec<Expression> = vec![
            Expression::IntegerLiteral { value: 1, suffix: IntSuffix::None, base: NumericBase::Decimal, span: s },
            Expression::FloatLiteral { value: 1.0, suffix: FloatSuffix::None, span: s },
            Expression::StringLiteral { value: "hi".to_string(), prefix: StringPrefix::None, span: s },
            Expression::CharLiteral { value: b'a' as u32, prefix: CharPrefix::None, span: s },
            Expression::Identifier { name: dummy_id(0), span: s },
            Expression::Binary { op: BinaryOp::Add, left: id_expr(), right: id_expr(), span: s },
            Expression::UnaryPrefix { op: UnaryOp::Negate, operand: id_expr(), span: s },
            Expression::PostIncrement { operand: id_expr(), span: s },
            Expression::PostDecrement { operand: id_expr(), span: s },
            Expression::Call { callee: id_expr(), args: vec![], span: s },
            Expression::Subscript { array: id_expr(), index: id_expr(), span: s },
            Expression::MemberAccess { object: id_expr(), member: dummy_id(1), span: s },
            Expression::ArrowAccess { pointer: id_expr(), member: dummy_id(1), span: s },
            Expression::Assignment { op: AssignmentOp::Assign, target: id_expr(), value: id_expr(), span: s },
            Expression::Ternary { condition: id_expr(), then_expr: id_expr(), else_expr: id_expr(), span: s },
            Expression::Comma { exprs: vec![], span: s },
            Expression::Cast { type_name: type_name(), operand: id_expr(), span: s },
            Expression::SizeofExpr { expr: id_expr(), span: s },
            Expression::SizeofType { type_name: type_name(), span: s },
            Expression::Alignof { type_name: type_name(), span: s },
            Expression::Generic { controlling: id_expr(), associations: vec![], span: s },
            Expression::CompoundLiteral {
                type_name: type_name(),
                initializer: Initializer::Compound { items: vec![], span: s },
                span: s,
            },
            Expression::StatementExpr { body: Box::new(Statement::Null { span: s }), span: s },
            Expression::LabelAddr { label: dummy_id(0), span: s },
            Expression::Extension { expr: id_expr(), span: s },
            Expression::BuiltinVaArg { ap: id_expr(), type_name: type_name(), span: s },
            Expression::BuiltinOffsetof { type_name: type_name(), member: dummy_id(0), span: s },
            Expression::BuiltinVaStart { ap: id_expr(), param: id_expr(), span: s },
            Expression::BuiltinVaEnd { ap: id_expr(), span: s },
            Expression::BuiltinVaCopy { dest: id_expr(), src: id_expr(), span: s },
            Expression::Paren { inner: id_expr(), span: s },
            Expression::Error { span: s },
        ];
        assert_eq!(_variants.len(), 32);
    }

    // -----------------------------------------------------------------------
    // Operator enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_binary_op_covers_all_operators() {
        let ops = [
            BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div,
            BinaryOp::Mod, BinaryOp::BitwiseAnd, BinaryOp::BitwiseOr,
            BinaryOp::BitwiseXor, BinaryOp::ShiftLeft, BinaryOp::ShiftRight,
            BinaryOp::LogicalAnd, BinaryOp::LogicalOr, BinaryOp::Equal,
            BinaryOp::NotEqual, BinaryOp::Less, BinaryOp::Greater,
            BinaryOp::LessEqual, BinaryOp::GreaterEqual,
        ];
        assert_eq!(ops.len(), 18);
        // Verify Copy, PartialEq
        assert_eq!(ops[0], BinaryOp::Add);
    }

    #[test]
    fn test_unary_op_covers_all_operators() {
        let ops = [
            UnaryOp::Plus, UnaryOp::Negate, UnaryOp::BitwiseNot,
            UnaryOp::LogicalNot, UnaryOp::Dereference, UnaryOp::AddressOf,
            UnaryOp::PreIncrement, UnaryOp::PreDecrement,
        ];
        assert_eq!(ops.len(), 8);
    }

    #[test]
    fn test_assignment_op_covers_all_operators() {
        let ops = [
            AssignmentOp::Assign, AssignmentOp::AddAssign, AssignmentOp::SubAssign,
            AssignmentOp::MulAssign, AssignmentOp::DivAssign, AssignmentOp::ModAssign,
            AssignmentOp::AndAssign, AssignmentOp::OrAssign, AssignmentOp::XorAssign,
            AssignmentOp::ShlAssign, AssignmentOp::ShrAssign,
        ];
        assert_eq!(ops.len(), 11);
    }

    // -----------------------------------------------------------------------
    // TypeSpecifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_specifier_basic_types() {
        let types: Vec<TypeSpecifier> = vec![
            TypeSpecifier::Void,
            TypeSpecifier::Char,
            TypeSpecifier::Short,
            TypeSpecifier::Int,
            TypeSpecifier::Long,
            TypeSpecifier::LongLong,
            TypeSpecifier::Float,
            TypeSpecifier::Double,
            TypeSpecifier::LongDouble,
            TypeSpecifier::Bool,
            TypeSpecifier::Error,
        ];
        assert_eq!(types.len(), 11);
    }

    #[test]
    fn test_type_specifier_composite() {
        let struct_def = TypeSpecifier::Struct(StructDef {
            tag: Some(dummy_id(0)),
            members: vec![],
            attributes: vec![],
            span: dummy_span(),
        });
        let _debug = format!("{:?}", struct_def);

        let union_def = TypeSpecifier::Union(UnionDef {
            tag: None,
            members: vec![],
            attributes: vec![],
            span: dummy_span(),
        });
        let _debug = format!("{:?}", union_def);

        let enum_def = TypeSpecifier::Enum(EnumDef {
            tag: Some(dummy_id(1)),
            variants: vec![EnumVariant {
                name: dummy_id(2),
                value: None,
                attributes: vec![],
                span: dummy_span(),
            }],
            attributes: vec![],
            span: dummy_span(),
        });
        let _debug = format!("{:?}", enum_def);
    }

    #[test]
    fn test_type_specifier_refs_and_typedef() {
        let _struct_ref = TypeSpecifier::StructRef { tag: dummy_id(0), span: dummy_span() };
        let _union_ref = TypeSpecifier::UnionRef { tag: dummy_id(1), span: dummy_span() };
        let _enum_ref = TypeSpecifier::EnumRef { tag: dummy_id(2), span: dummy_span() };
        let _typedef = TypeSpecifier::TypedefName { name: dummy_id(3), span: dummy_span() };
    }

    #[test]
    fn test_type_specifier_gcc_typeof() {
        let _typeof_expr = TypeSpecifier::Typeof {
            expr: Box::new(Expression::Identifier { name: dummy_id(0), span: dummy_span() }),
            span: dummy_span(),
        };
        let _typeof_type = TypeSpecifier::TypeofType {
            type_name: Box::new(TypeSpecifier::Int),
            span: dummy_span(),
        };
    }

    #[test]
    fn test_type_specifier_signed_unsigned() {
        let signed_int = TypeSpecifier::Signed(Box::new(TypeSpecifier::Int));
        let unsigned_long = TypeSpecifier::Unsigned(Box::new(TypeSpecifier::Long));
        let _debug1 = format!("{:?}", signed_int);
        let _debug2 = format!("{:?}", unsigned_long);
    }

    #[test]
    fn test_type_specifier_qualified() {
        let qualified = TypeSpecifier::Qualified {
            qualifiers: vec![TypeQualifier::Const, TypeQualifier::Volatile],
            inner: Box::new(TypeSpecifier::Int),
        };
        let _debug = format!("{:?}", qualified);
    }

    // -----------------------------------------------------------------------
    // GCC Attribute tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gcc_attribute_construction() {
        let attr = GccAttribute {
            name: dummy_id(10),
            args: vec![
                AttributeArg::Integer(16),
                AttributeArg::String(".text".to_string()),
                AttributeArg::Identifier(dummy_id(11)),
            ],
            span: dummy_span(),
        };
        assert_eq!(attr.args.len(), 3);
    }

    // -----------------------------------------------------------------------
    // AsmStatement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_asm_statement_construction() {
        let asm = AsmStatement {
            is_volatile: true,
            is_inline: false,
            is_goto: true,
            template: "mov %1, %0".to_string(),
            outputs: vec![AsmOperand {
                symbolic_name: Some(dummy_id(0)),
                constraint: "=r".to_string(),
                expr: Expression::Identifier { name: dummy_id(1), span: dummy_span() },
                span: dummy_span(),
            }],
            inputs: vec![AsmOperand {
                symbolic_name: None,
                constraint: "r".to_string(),
                expr: Expression::Identifier { name: dummy_id(2), span: dummy_span() },
                span: dummy_span(),
            }],
            clobbers: vec!["memory".to_string(), "cc".to_string()],
            goto_labels: vec![dummy_id(3)],
            span: dummy_span(),
        };
        assert!(asm.is_volatile);
        assert!(asm.is_goto);
        assert!(!asm.is_inline);
        assert_eq!(asm.outputs.len(), 1);
        assert_eq!(asm.inputs.len(), 1);
        assert_eq!(asm.clobbers.len(), 2);
        assert_eq!(asm.goto_labels.len(), 1);
    }

    // -----------------------------------------------------------------------
    // TypeName tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_name_construction() {
        let tn = TypeName {
            specifiers: DeclSpecifiers {
                storage_class: None,
                type_qualifiers: vec![TypeQualifier::Const],
                type_specifier: TypeSpecifier::Int,
                function_specifiers: vec![],
                attributes: vec![],
                span: dummy_span(),
            },
            abstract_declarator: Some(AbstractDeclarator {
                pointer: vec![Pointer { qualifiers: vec![] }],
                direct: None,
                span: dummy_span(),
            }),
            span: dummy_span(),
        };
        assert!(tn.abstract_declarator.is_some());
    }

    // -----------------------------------------------------------------------
    // Declarator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_declarator_simple_identifier() {
        let decl = Declarator {
            pointer: vec![],
            direct: DirectDeclarator::Identifier(dummy_id(0)),
            attributes: vec![],
            span: dummy_span(),
        };
        let _debug = format!("{:?}", decl);
    }

    #[test]
    fn test_declarator_pointer_chain() {
        // `int *const *volatile p`
        let decl = Declarator {
            pointer: vec![
                Pointer { qualifiers: vec![TypeQualifier::Volatile] },
                Pointer { qualifiers: vec![TypeQualifier::Const] },
            ],
            direct: DirectDeclarator::Identifier(dummy_id(0)),
            attributes: vec![],
            span: dummy_span(),
        };
        assert_eq!(decl.pointer.len(), 2);
    }

    #[test]
    fn test_declarator_array() {
        let decl = Declarator {
            pointer: vec![],
            direct: DirectDeclarator::Array {
                base: Box::new(DirectDeclarator::Identifier(dummy_id(0))),
                size: ArraySize::Fixed(Box::new(Expression::IntegerLiteral {
                    value: 10,
                    suffix: IntSuffix::None,
                    base: NumericBase::Decimal,
                    span: dummy_span(),
                })),
                qualifiers: vec![],
            },
            attributes: vec![],
            span: dummy_span(),
        };
        let _debug = format!("{:?}", decl);
    }

    #[test]
    fn test_declarator_function_pointer() {
        // `int (*fp)(int, float)`
        let decl = Declarator {
            pointer: vec![],
            direct: DirectDeclarator::Parenthesized(Box::new(Declarator {
                pointer: vec![Pointer { qualifiers: vec![] }],
                direct: DirectDeclarator::Identifier(dummy_id(0)),
                attributes: vec![],
                span: dummy_span(),
            })),
            attributes: vec![],
            span: dummy_span(),
        };
        let _debug = format!("{:?}", decl);
    }

    // -----------------------------------------------------------------------
    // Initializer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_initializer_expression() {
        let init = Initializer::Expression(Box::new(Expression::IntegerLiteral {
            value: 42,
            suffix: IntSuffix::None,
            base: NumericBase::Decimal,
            span: dummy_span(),
        }));
        let _debug = format!("{:?}", init);
    }

    #[test]
    fn test_initializer_compound() {
        let init = Initializer::Compound {
            items: vec![
                DesignatedInitializer {
                    designators: vec![Designator::Field(dummy_id(0))],
                    initializer: Initializer::Expression(Box::new(Expression::IntegerLiteral {
                        value: 1,
                        suffix: IntSuffix::None,
                        base: NumericBase::Decimal,
                        span: dummy_span(),
                    })),
                    span: dummy_span(),
                },
                DesignatedInitializer {
                    designators: vec![Designator::Index(Box::new(Expression::IntegerLiteral {
                        value: 0,
                        suffix: IntSuffix::None,
                        base: NumericBase::Decimal,
                        span: dummy_span(),
                    }))],
                    initializer: Initializer::Expression(Box::new(Expression::IntegerLiteral {
                        value: 2,
                        suffix: IntSuffix::None,
                        base: NumericBase::Decimal,
                        span: dummy_span(),
                    })),
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let _debug = format!("{:?}", init);
    }

    #[test]
    fn test_designator_range() {
        let range = Designator::Range(
            Box::new(Expression::IntegerLiteral {
                value: 0,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
                span: dummy_span(),
            }),
            Box::new(Expression::IntegerLiteral {
                value: 9,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
                span: dummy_span(),
            }),
        );
        let _debug = format!("{:?}", range);
    }

    // -----------------------------------------------------------------------
    // GenericAssociation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generic_association() {
        let type_assoc = GenericAssociation::Type {
            type_name: TypeName {
                specifiers: DeclSpecifiers {
                    storage_class: None,
                    type_qualifiers: vec![],
                    type_specifier: TypeSpecifier::Int,
                    function_specifiers: vec![],
                    attributes: vec![],
                    span: dummy_span(),
                },
                abstract_declarator: None,
                span: dummy_span(),
            },
            expr: Expression::StringLiteral {
                value: "int".to_string(),
                prefix: StringPrefix::None,
                span: dummy_span(),
            },
            span: dummy_span(),
        };
        let _debug = format!("{:?}", type_assoc);

        let default_assoc = GenericAssociation::Default {
            expr: Expression::StringLiteral {
                value: "other".to_string(),
                prefix: StringPrefix::None,
                span: dummy_span(),
            },
            span: dummy_span(),
        };
        let _debug = format!("{:?}", default_assoc);
    }

    // -----------------------------------------------------------------------
    // Literal enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_numeric_base() {
        let bases = [NumericBase::Decimal, NumericBase::Hexadecimal, NumericBase::Octal, NumericBase::Binary];
        assert_eq!(bases.len(), 4);
        assert_eq!(NumericBase::Decimal, NumericBase::Decimal);
    }

    #[test]
    fn test_int_suffix() {
        let suffixes = [
            IntSuffix::None, IntSuffix::Unsigned, IntSuffix::Long,
            IntSuffix::ULong, IntSuffix::LongLong, IntSuffix::ULongLong,
        ];
        assert_eq!(suffixes.len(), 6);
    }

    #[test]
    fn test_float_suffix() {
        let suffixes = [FloatSuffix::None, FloatSuffix::Float, FloatSuffix::Long];
        assert_eq!(suffixes.len(), 3);
    }

    #[test]
    fn test_string_prefix() {
        let prefixes = [
            StringPrefix::None, StringPrefix::Wide, StringPrefix::Utf8,
            StringPrefix::Utf16, StringPrefix::Utf32,
        ];
        assert_eq!(prefixes.len(), 5);
    }

    #[test]
    fn test_char_prefix() {
        let prefixes = [
            CharPrefix::None, CharPrefix::Wide, CharPrefix::Utf16, CharPrefix::Utf32,
        ];
        assert_eq!(prefixes.len(), 4);
    }

    // -----------------------------------------------------------------------
    // StorageClass, TypeQualifier, FunctionSpecifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_storage_class() {
        let classes = [
            StorageClass::Static, StorageClass::Extern, StorageClass::Auto,
            StorageClass::Register, StorageClass::ThreadLocal,
        ];
        assert_eq!(classes.len(), 5);
        assert_eq!(StorageClass::Static, StorageClass::Static);
        assert_ne!(StorageClass::Static, StorageClass::Extern);
    }

    #[test]
    fn test_type_qualifier() {
        let quals = [
            TypeQualifier::Const, TypeQualifier::Volatile,
            TypeQualifier::Restrict, TypeQualifier::Atomic,
        ];
        assert_eq!(quals.len(), 4);
    }

    #[test]
    fn test_function_specifier() {
        let specs = [FunctionSpecifier::Inline, FunctionSpecifier::Noreturn];
        assert_eq!(specs.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Debug and Clone compile-time verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_types_implement_debug_and_clone() {
        // This test verifies at compile time that all major types implement
        // Debug and Clone by formatting and cloning representative instances.
        fn assert_debug_clone<T: std::fmt::Debug + Clone>(val: &T) {
            let _debug = format!("{:?}", val);
            let _cloned = val.clone();
        }

        let s = dummy_span();
        assert_debug_clone(&TranslationUnit { declarations: vec![], span: s });
        assert_debug_clone(&Declaration::Empty { span: s });
        assert_debug_clone(&FunctionDef {
            specifiers: DeclSpecifiers {
                storage_class: None, type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Void, function_specifiers: vec![],
                attributes: vec![], span: s,
            },
            declarator: Declarator { pointer: vec![], direct: DirectDeclarator::Abstract, attributes: vec![], span: s },
            body: Box::new(Statement::Null { span: s }),
            attributes: vec![], span: s,
        });
        assert_debug_clone(&DeclSpecifiers {
            storage_class: None, type_qualifiers: vec![],
            type_specifier: TypeSpecifier::Int, function_specifiers: vec![],
            attributes: vec![], span: s,
        });
        assert_debug_clone(&InitDeclarator {
            declarator: Declarator { pointer: vec![], direct: DirectDeclarator::Abstract, attributes: vec![], span: s },
            initializer: None, span: s,
        });
        assert_debug_clone(&Declarator { pointer: vec![], direct: DirectDeclarator::Abstract, attributes: vec![], span: s });
        assert_debug_clone(&Pointer { qualifiers: vec![] });
        assert_debug_clone(&DirectDeclarator::Abstract);
        assert_debug_clone(&AbstractDeclarator { pointer: vec![], direct: None, span: s });
        assert_debug_clone(&ArraySize::Unspecified);
        assert_debug_clone(&ArraySize::VLA);
        assert_debug_clone(&ParamList { params: vec![], variadic: false, span: s });
        assert_debug_clone(&ParamDeclaration {
            specifiers: DeclSpecifiers {
                storage_class: None, type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Int, function_specifiers: vec![],
                attributes: vec![], span: s,
            },
            declarator: None, span: s,
        });
        assert_debug_clone(&TypeSpecifier::Int);
        assert_debug_clone(&StorageClass::Static);
        assert_debug_clone(&TypeQualifier::Const);
        assert_debug_clone(&FunctionSpecifier::Inline);
        assert_debug_clone(&StructDef { tag: None, members: vec![], attributes: vec![], span: s });
        assert_debug_clone(&UnionDef { tag: None, members: vec![], attributes: vec![], span: s });
        assert_debug_clone(&EnumDef { tag: None, variants: vec![], attributes: vec![], span: s });
        assert_debug_clone(&EnumVariant { name: dummy_id(0), value: None, attributes: vec![], span: s });
        assert_debug_clone(&Initializer::Expression(Box::new(Expression::Error { span: s })));
        assert_debug_clone(&DesignatedInitializer {
            designators: vec![], initializer: Initializer::Expression(Box::new(Expression::Error { span: s })),
            span: s,
        });
        assert_debug_clone(&Designator::Field(dummy_id(0)));
        assert_debug_clone(&Statement::Null { span: s });
        assert_debug_clone(&BlockItem::Statement(Statement::Null { span: s }));
        assert_debug_clone(&ForInit::Expression(Box::new(Expression::Error { span: s })));
        assert_debug_clone(&Expression::Error { span: s });
        assert_debug_clone(&TypeName {
            specifiers: DeclSpecifiers {
                storage_class: None, type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Int, function_specifiers: vec![],
                attributes: vec![], span: s,
            },
            abstract_declarator: None, span: s,
        });
        assert_debug_clone(&BinaryOp::Add);
        assert_debug_clone(&UnaryOp::Plus);
        assert_debug_clone(&AssignmentOp::Assign);
        assert_debug_clone(&NumericBase::Decimal);
        assert_debug_clone(&IntSuffix::None);
        assert_debug_clone(&FloatSuffix::None);
        assert_debug_clone(&StringPrefix::None);
        assert_debug_clone(&CharPrefix::None);
        assert_debug_clone(&GccAttribute { name: dummy_id(0), args: vec![], span: s });
        assert_debug_clone(&AttributeArg::Integer(42));
        assert_debug_clone(&AsmStatement {
            is_volatile: false, is_inline: false, is_goto: false,
            template: String::new(), outputs: vec![], inputs: vec![],
            clobbers: vec![], goto_labels: vec![], span: s,
        });
        assert_debug_clone(&AsmOperand {
            symbolic_name: None, constraint: String::new(),
            expr: Expression::Error { span: s }, span: s,
        });
        assert_debug_clone(&GenericAssociation::Default {
            expr: Expression::Error { span: s }, span: s,
        });
    }

    // -----------------------------------------------------------------------
    // Struct member tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_member_field_with_bitwidth() {
        let member = StructMember::Field {
            specifiers: DeclSpecifiers {
                storage_class: None,
                type_qualifiers: vec![],
                type_specifier: TypeSpecifier::Int,
                function_specifiers: vec![],
                attributes: vec![],
                span: dummy_span(),
            },
            declarators: vec![StructFieldDeclarator {
                declarator: Some(Declarator {
                    pointer: vec![],
                    direct: DirectDeclarator::Identifier(dummy_id(0)),
                    attributes: vec![],
                    span: dummy_span(),
                }),
                bit_width: Some(Box::new(Expression::IntegerLiteral {
                    value: 3,
                    suffix: IntSuffix::None,
                    base: NumericBase::Decimal,
                    span: dummy_span(),
                })),
                span: dummy_span(),
            }],
            span: dummy_span(),
        };
        let _debug = format!("{:?}", member);
    }

    #[test]
    fn test_struct_member_anonymous() {
        let member = StructMember::Anonymous {
            type_spec: TypeSpecifier::Struct(StructDef {
                tag: None,
                members: vec![],
                attributes: vec![],
                span: dummy_span(),
            }),
            span: dummy_span(),
        };
        let _debug = format!("{:?}", member);
    }

    // -----------------------------------------------------------------------
    // Abstract declarator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_abstract_declarator_pointer() {
        let abs_decl = AbstractDeclarator {
            pointer: vec![Pointer { qualifiers: vec![TypeQualifier::Const] }],
            direct: None,
            span: dummy_span(),
        };
        assert_eq!(abs_decl.pointer.len(), 1);
    }

    #[test]
    fn test_direct_abstract_declarator_array() {
        let dad = DirectAbstractDeclarator::Array {
            base: None,
            size: ArraySize::Fixed(Box::new(Expression::IntegerLiteral {
                value: 10,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
                span: dummy_span(),
            })),
            qualifiers: vec![],
        };
        let _debug = format!("{:?}", dad);
    }

    #[test]
    fn test_direct_abstract_declarator_function() {
        let dad = DirectAbstractDeclarator::Function {
            base: None,
            params: ParamList {
                params: vec![],
                variadic: true,
                span: dummy_span(),
            },
        };
        let _debug = format!("{:?}", dad);
    }

    // -----------------------------------------------------------------------
    // Array size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_array_size_variants() {
        let _fixed = ArraySize::Fixed(Box::new(Expression::IntegerLiteral {
            value: 10, suffix: IntSuffix::None, base: NumericBase::Decimal, span: dummy_span(),
        }));
        let _unspec = ArraySize::Unspecified;
        let _vla = ArraySize::VLA;
        let _static = ArraySize::Static(Box::new(Expression::IntegerLiteral {
            value: 10, suffix: IntSuffix::None, base: NumericBase::Decimal, span: dummy_span(),
        }));
    }

    // -----------------------------------------------------------------------
    // BlockItem and ForInit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_item_variants() {
        let _decl = BlockItem::Declaration(Box::new(Declaration::Empty { span: dummy_span() }));
        let _stmt = BlockItem::Statement(Statement::Null { span: dummy_span() });
    }

    #[test]
    fn test_for_init_variants() {
        let _decl = ForInit::Declaration(Box::new(Declaration::Empty { span: dummy_span() }));
        let _expr = ForInit::Expression(Box::new(Expression::Error { span: dummy_span() }));
    }

    // -----------------------------------------------------------------------
    // SourceSpan members_accessed test (start, end)
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_span_members_accessed() {
        let span = dummy_span();
        // Verify `start` and `end` members are accessible as per schema
        let _start = span.start;
        let _end = span.end;
        assert_eq!(_start, _end);
    }
}
