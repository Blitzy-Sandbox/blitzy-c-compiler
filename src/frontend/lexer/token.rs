//! Token type definitions for the `bcc` C compiler lexer.
//!
//! This module defines the foundational data structures consumed by every
//! downstream compiler phase: the [`TokenKind`] enum (138 variants covering all
//! C11 keywords, GCC extension keywords, operators, punctuation, literal types,
//! identifiers, and the EOF marker), the [`Token`] struct carrying kind, source
//! span, and associated value, and supporting enums [`TokenValue`], [`IntSuffix`],
//! [`FloatSuffix`], and [`NumericBase`].
//!
//! # Design
//!
//! * **`TokenKind`** is `Copy + Clone + PartialEq + Eq + Hash` for efficient
//!   pattern-matching and use as hash-map keys.
//! * **`Token`** is `Clone` but not `Copy` because [`TokenValue`] may hold an
//!   owned `String`. Tokens are allocated in bulk (`Vec<Token>`) so the struct
//!   size is kept reasonable.
//! * **Zero external dependencies** — only `std::fmt`, sibling `source`, and
//!   `crate::common::intern` are imported.
//! * **No `unsafe` code** — this module defines data types only.
//!
//! # Integration
//!
//! Per AAP §0.4.1: "`Vec<Token>` where each `Token` carries type, value, and
//! `SourceLocation`" is the Lexer → Parser interface. This module defines the
//! types that flow through every subsequent compilation phase.

use std::fmt;

use super::source::SourceSpan;
use crate::common::intern::InternId;

// ===========================================================================
// TokenKind — exhaustive token classification (138 variants)
// ===========================================================================

/// Enumerates every distinct token type recognised by the `bcc` lexer.
///
/// The enum has 138 variants, partitioned into:
///
/// | Category            | Count |
/// |---------------------|-------|
/// | C11 keywords        |    44 |
/// | GCC extension kw.   |    40 |
/// | Operators           |    39 |
/// | Punctuation         |     9 |
/// | Literal types       |     4 |
/// | Identifier + EOF    |     2 |
///
/// `TokenKind` derives `Copy` so that it can be cheaply passed by value and
/// used in exhaustive `match` expressions throughout the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // =======================================================================
    // C11 Keywords (44)
    // =======================================================================

    /// `auto`
    Auto,
    /// `break`
    Break,
    /// `case`
    Case,
    /// `char`
    Char,
    /// `const`
    Const,
    /// `continue`
    Continue,
    /// `default`
    Default,
    /// `do`
    Do,
    /// `double`
    Double,
    /// `else`
    Else,
    /// `enum`
    Enum,
    /// `extern`
    Extern,
    /// `float`
    Float,
    /// `for`
    For,
    /// `goto`
    Goto,
    /// `if`
    If,
    /// `inline` (also `__inline__`, `__inline`)
    Inline,
    /// `int`
    Int,
    /// `long`
    Long,
    /// `register`
    Register,
    /// `restrict` (also `__restrict__`, `__restrict`)
    Restrict,
    /// `return`
    Return,
    /// `short`
    Short,
    /// `signed` (also `__signed__`, `__signed`)
    Signed,
    /// `sizeof`
    Sizeof,
    /// `static`
    Static,
    /// `struct`
    Struct,
    /// `switch`
    Switch,
    /// `typedef`
    Typedef,
    /// `union`
    Union,
    /// `unsigned`
    Unsigned,
    /// `void`
    Void,
    /// `volatile` (also `__volatile__`, `__volatile`)
    Volatile,
    /// `while`
    While,

    // C11-specific keywords
    /// `_Alignas`
    Alignas,
    /// `_Alignof`
    Alignof,
    /// `_Atomic`
    Atomic,
    /// `_Bool`
    Bool,
    /// `_Complex`
    Complex,
    /// `_Generic`
    Generic,
    /// `_Imaginary`
    Imaginary,
    /// `_Noreturn`
    Noreturn,
    /// `_Static_assert`
    StaticAssert,
    /// `_Thread_local`
    ThreadLocal,

    // =======================================================================
    // GCC Extension Keywords (40)
    // =======================================================================

    /// `__attribute__`
    GccAttribute,
    /// `__extension__`
    GccExtension,
    /// `asm`, `__asm__`, `__asm`
    Asm,
    /// `typeof`, `__typeof__`, `__typeof`
    Typeof,

    // --- varargs builtins ---
    /// `__builtin_va_list`
    BuiltinVaList,
    /// `__builtin_va_start`
    BuiltinVaStart,
    /// `__builtin_va_end`
    BuiltinVaEnd,
    /// `__builtin_va_arg`
    BuiltinVaArg,
    /// `__builtin_va_copy`
    BuiltinVaCopy,

    // --- type/offset builtins ---
    /// `__builtin_offsetof`
    BuiltinOffsetof,
    /// `__builtin_types_compatible_p`
    BuiltinTypesCompatibleP,

    // --- control-flow / optimisation builtins ---
    /// `__builtin_expect`
    BuiltinExpect,
    /// `__builtin_unreachable`
    BuiltinUnreachable,
    /// `__builtin_constant_p`
    BuiltinConstantP,
    /// `__builtin_choose_expr`
    BuiltinChooseExpr,

    // --- byte-swap builtins ---
    /// `__builtin_bswap16`
    BuiltinBswap16,
    /// `__builtin_bswap32`
    BuiltinBswap32,
    /// `__builtin_bswap64`
    BuiltinBswap64,

    // --- bit-manipulation builtins ---
    /// `__builtin_clz`
    BuiltinClz,
    /// `__builtin_ctz`
    BuiltinCtz,
    /// `__builtin_popcount`
    BuiltinPopcount,
    /// `__builtin_ffs`
    BuiltinFfs,

    // --- math builtins ---
    /// `__builtin_abs`
    BuiltinAbs,
    /// `__builtin_fabsf`
    BuiltinFabsf,
    /// `__builtin_fabs`
    BuiltinFabs,
    /// `__builtin_inf`
    BuiltinInf,
    /// `__builtin_inff`
    BuiltinInff,
    /// `__builtin_huge_val`
    BuiltinHugeVal,
    /// `__builtin_huge_valf`
    BuiltinHugeValf,
    /// `__builtin_nan`
    BuiltinNan,
    /// `__builtin_nanf`
    BuiltinNanf,

    // --- misc builtins ---
    /// `__builtin_trap`
    BuiltinTrap,
    /// `__builtin_alloca`
    BuiltinAlloca,
    /// `__builtin_memcpy`
    BuiltinMemcpy,
    /// `__builtin_memset`
    BuiltinMemset,
    /// `__builtin_strlen`
    BuiltinStrlen,
    /// `__builtin_frame_address`
    BuiltinFrameAddress,

    // --- GCC type extensions ---
    /// `__int128`
    GccInt128,
    /// `__label__`
    GccLabel,
    /// `__auto_type`
    GccAutoType,

    // =======================================================================
    // Operators (39)
    // =======================================================================

    // --- Arithmetic ---
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*` (also pointer dereference / pointer type)
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,

    // --- Comparison ---
    /// `==`
    EqualEqual,
    /// `!=`
    BangEqual,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,

    // --- Logical ---
    /// `&&`
    AmpAmp,
    /// `||`
    PipePipe,
    /// `!`
    Bang,

    // --- Bitwise ---
    /// `&`
    Amp,
    /// `|`
    Pipe,
    /// `^`
    Caret,
    /// `~`
    Tilde,
    /// `<<`
    LessLess,
    /// `>>`
    GreaterGreater,

    // --- Assignment ---
    /// `=`
    Equal,
    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    StarEqual,
    /// `/=`
    SlashEqual,
    /// `%=`
    PercentEqual,
    /// `&=`
    AmpEqual,
    /// `|=`
    PipeEqual,
    /// `^=`
    CaretEqual,
    /// `<<=`
    LessLessEqual,
    /// `>>=`
    GreaterGreaterEqual,

    // --- Increment / Decrement ---
    /// `++`
    PlusPlus,
    /// `--`
    MinusMinus,

    // --- Member access ---
    /// `->`
    Arrow,
    /// `.`
    Dot,

    // --- Ternary ---
    /// `?`
    Question,
    /// `:`
    Colon,

    // --- Preprocessor ---
    /// `#`
    Hash,
    /// `##`
    HashHash,

    // =======================================================================
    // Punctuation (9)
    // =======================================================================

    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `{`
    LeftBrace,
    /// `}`
    RightBrace,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `...`
    Ellipsis,

    // =======================================================================
    // Literals (4)
    // =======================================================================

    /// Integer constant — value stored in [`TokenValue::Integer`].
    IntegerLiteral,
    /// Floating-point constant — value stored in [`TokenValue::Float`].
    FloatLiteral,
    /// String literal `"..."` — content stored in [`TokenValue::Str`].
    StringLiteral,
    /// Character literal `'...'` — value stored in [`TokenValue::Char`].
    CharLiteral,

    // =======================================================================
    // Identifier (1)
    // =======================================================================

    /// Any non-keyword identifier — interned handle in [`TokenValue::Identifier`].
    Identifier,

    // =======================================================================
    // Special (1)
    // =======================================================================

    /// End of file / end of input marker.
    Eof,
}

// ===========================================================================
// IntSuffix / FloatSuffix / NumericBase — literal metadata
// ===========================================================================

/// Integer literal suffix classification.
///
/// C11 allows `u`/`U` for unsigned, `l`/`L` for long, `ll`/`LL` for long long,
/// and combinations thereof (order-independent, case-insensitive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntSuffix {
    /// No suffix — plain `int`.
    None,
    /// `u` or `U` — `unsigned int`.
    Unsigned,
    /// `l` or `L` — `long`.
    Long,
    /// `ul`, `UL`, `lu`, `LU` — `unsigned long`.
    ULong,
    /// `ll` or `LL` — `long long`.
    LongLong,
    /// `ull`, `ULL`, `llu`, `LLU` — `unsigned long long`.
    ULongLong,
}

/// Floating-point literal suffix classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSuffix {
    /// No suffix — `double` (default).
    None,
    /// `f` or `F` — `float`.
    Float,
    /// `l` or `L` — `long double`.
    Long,
}

/// Numeric literal base / radix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericBase {
    /// `42`
    Decimal,
    /// `0xFF`
    Hexadecimal,
    /// `077`
    Octal,
    /// `0b1010` (GCC extension)
    Binary,
}

// ===========================================================================
// TokenValue — associated data carried by each Token
// ===========================================================================

/// The associated data payload of a [`Token`].
///
/// Most tokens (keywords, operators, punctuation) carry [`TokenValue::None`].
/// Identifier tokens carry an [`InternId`] handle, and literal tokens carry
/// the parsed value together with suffix / base metadata.
#[derive(Debug, Clone)]
pub enum TokenValue {
    /// No additional value (keywords, operators, punctuation, EOF).
    None,

    /// Interned identifier name — compact `u32` handle for O(1) comparison.
    Identifier(InternId),

    /// Parsed integer literal with value, suffix, and base.
    Integer {
        /// The integer value (128-bit to accommodate all C integer types).
        value: u128,
        /// Integer suffix classification (unsigned, long, long long, etc.).
        suffix: IntSuffix,
        /// Numeric base (decimal, hex, octal, binary).
        base: NumericBase,
    },

    /// Parsed floating-point literal with value and suffix.
    Float {
        /// The parsed `f64` value.
        value: f64,
        /// Float suffix classification (float, long double, or default double).
        suffix: FloatSuffix,
    },

    /// String literal content (owned, after escape processing).
    Str(String),

    /// Character literal code point.
    ///
    /// Uses `u32` to accommodate wide characters and full Unicode range.
    Char(u32),
}

// ===========================================================================
// Token — the primary output type of the lexer
// ===========================================================================

/// A single lexical token produced by the `bcc` lexer.
///
/// Each `Token` carries its classification ([`TokenKind`]), source location
/// ([`SourceSpan`]), and an optional associated value ([`TokenValue`]).
/// Tokens are collected into a `Vec<Token>` that always terminates with
/// [`TokenKind::Eof`].
#[derive(Debug, Clone)]
pub struct Token {
    /// The classification of this token (keyword, operator, literal, etc.).
    pub kind: TokenKind,
    /// The source range this token covers (file, line, column, byte offsets).
    pub span: SourceSpan,
    /// The associated data for this token (literal value, identifier handle, etc.).
    pub value: TokenValue,
}

// ===========================================================================
// Token — constructors and utility methods
// ===========================================================================

impl Token {
    /// Creates a new [`Token`] with the given kind, span, and value.
    #[inline]
    pub fn new(kind: TokenKind, span: SourceSpan, value: TokenValue) -> Self {
        Token { kind, span, value }
    }

    /// Returns `true` if this token's kind matches `kind`.
    #[inline]
    pub fn is(&self, kind: TokenKind) -> bool {
        self.kind == kind
    }

    /// Returns `true` if this token is any keyword (C11 or GCC extension).
    #[inline]
    pub fn is_keyword(&self) -> bool {
        self.kind.is_keyword()
    }

    /// Returns `true` if this token is any literal type
    /// (integer, float, string, or character).
    #[inline]
    pub fn is_literal(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::IntegerLiteral
                | TokenKind::FloatLiteral
                | TokenKind::StringLiteral
                | TokenKind::CharLiteral
        )
    }

    /// Returns `true` if this token is an assignment operator
    /// (`=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`).
    #[inline]
    pub fn is_assignment_op(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Equal
                | TokenKind::PlusEqual
                | TokenKind::MinusEqual
                | TokenKind::StarEqual
                | TokenKind::SlashEqual
                | TokenKind::PercentEqual
                | TokenKind::AmpEqual
                | TokenKind::PipeEqual
                | TokenKind::CaretEqual
                | TokenKind::LessLessEqual
                | TokenKind::GreaterGreaterEqual
        )
    }

    /// Returns `true` if this token is a comparison operator
    /// (`==`, `!=`, `<`, `>`, `<=`, `>=`).
    #[inline]
    pub fn is_comparison_op(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::EqualEqual
                | TokenKind::BangEqual
                | TokenKind::Less
                | TokenKind::Greater
                | TokenKind::LessEqual
                | TokenKind::GreaterEqual
        )
    }

    /// Returns `true` if this is the end-of-file marker.
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.kind == TokenKind::Eof
    }
}

// ===========================================================================
// TokenKind — utility methods
// ===========================================================================

impl TokenKind {
    /// Returns the canonical display string for this token kind.
    ///
    /// Keywords produce their keyword text, operators produce the operator
    /// symbol, punctuation produces the punctuation character, and special
    /// tokens produce a descriptive name suitable for error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            // --- C11 keywords ---
            TokenKind::Auto => "auto",
            TokenKind::Break => "break",
            TokenKind::Case => "case",
            TokenKind::Char => "char",
            TokenKind::Const => "const",
            TokenKind::Continue => "continue",
            TokenKind::Default => "default",
            TokenKind::Do => "do",
            TokenKind::Double => "double",
            TokenKind::Else => "else",
            TokenKind::Enum => "enum",
            TokenKind::Extern => "extern",
            TokenKind::Float => "float",
            TokenKind::For => "for",
            TokenKind::Goto => "goto",
            TokenKind::If => "if",
            TokenKind::Inline => "inline",
            TokenKind::Int => "int",
            TokenKind::Long => "long",
            TokenKind::Register => "register",
            TokenKind::Restrict => "restrict",
            TokenKind::Return => "return",
            TokenKind::Short => "short",
            TokenKind::Signed => "signed",
            TokenKind::Sizeof => "sizeof",
            TokenKind::Static => "static",
            TokenKind::Struct => "struct",
            TokenKind::Switch => "switch",
            TokenKind::Typedef => "typedef",
            TokenKind::Union => "union",
            TokenKind::Unsigned => "unsigned",
            TokenKind::Void => "void",
            TokenKind::Volatile => "volatile",
            TokenKind::While => "while",
            TokenKind::Alignas => "_Alignas",
            TokenKind::Alignof => "_Alignof",
            TokenKind::Atomic => "_Atomic",
            TokenKind::Bool => "_Bool",
            TokenKind::Complex => "_Complex",
            TokenKind::Generic => "_Generic",
            TokenKind::Imaginary => "_Imaginary",
            TokenKind::Noreturn => "_Noreturn",
            TokenKind::StaticAssert => "_Static_assert",
            TokenKind::ThreadLocal => "_Thread_local",

            // --- GCC extension keywords ---
            TokenKind::GccAttribute => "__attribute__",
            TokenKind::GccExtension => "__extension__",
            TokenKind::Asm => "asm",
            TokenKind::Typeof => "typeof",
            TokenKind::BuiltinVaList => "__builtin_va_list",
            TokenKind::BuiltinVaStart => "__builtin_va_start",
            TokenKind::BuiltinVaEnd => "__builtin_va_end",
            TokenKind::BuiltinVaArg => "__builtin_va_arg",
            TokenKind::BuiltinVaCopy => "__builtin_va_copy",
            TokenKind::BuiltinOffsetof => "__builtin_offsetof",
            TokenKind::BuiltinTypesCompatibleP => "__builtin_types_compatible_p",
            TokenKind::BuiltinExpect => "__builtin_expect",
            TokenKind::BuiltinUnreachable => "__builtin_unreachable",
            TokenKind::BuiltinConstantP => "__builtin_constant_p",
            TokenKind::BuiltinChooseExpr => "__builtin_choose_expr",
            TokenKind::BuiltinBswap16 => "__builtin_bswap16",
            TokenKind::BuiltinBswap32 => "__builtin_bswap32",
            TokenKind::BuiltinBswap64 => "__builtin_bswap64",
            TokenKind::BuiltinClz => "__builtin_clz",
            TokenKind::BuiltinCtz => "__builtin_ctz",
            TokenKind::BuiltinPopcount => "__builtin_popcount",
            TokenKind::BuiltinFfs => "__builtin_ffs",
            TokenKind::BuiltinAbs => "__builtin_abs",
            TokenKind::BuiltinFabsf => "__builtin_fabsf",
            TokenKind::BuiltinFabs => "__builtin_fabs",
            TokenKind::BuiltinInf => "__builtin_inf",
            TokenKind::BuiltinInff => "__builtin_inff",
            TokenKind::BuiltinHugeVal => "__builtin_huge_val",
            TokenKind::BuiltinHugeValf => "__builtin_huge_valf",
            TokenKind::BuiltinNan => "__builtin_nan",
            TokenKind::BuiltinNanf => "__builtin_nanf",
            TokenKind::BuiltinTrap => "__builtin_trap",
            TokenKind::BuiltinAlloca => "__builtin_alloca",
            TokenKind::BuiltinMemcpy => "__builtin_memcpy",
            TokenKind::BuiltinMemset => "__builtin_memset",
            TokenKind::BuiltinStrlen => "__builtin_strlen",
            TokenKind::BuiltinFrameAddress => "__builtin_frame_address",
            TokenKind::GccInt128 => "__int128",
            TokenKind::GccLabel => "__label__",
            TokenKind::GccAutoType => "__auto_type",

            // --- Operators ---
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::EqualEqual => "==",
            TokenKind::BangEqual => "!=",
            TokenKind::Less => "<",
            TokenKind::Greater => ">",
            TokenKind::LessEqual => "<=",
            TokenKind::GreaterEqual => ">=",
            TokenKind::AmpAmp => "&&",
            TokenKind::PipePipe => "||",
            TokenKind::Bang => "!",
            TokenKind::Amp => "&",
            TokenKind::Pipe => "|",
            TokenKind::Caret => "^",
            TokenKind::Tilde => "~",
            TokenKind::LessLess => "<<",
            TokenKind::GreaterGreater => ">>",
            TokenKind::Equal => "=",
            TokenKind::PlusEqual => "+=",
            TokenKind::MinusEqual => "-=",
            TokenKind::StarEqual => "*=",
            TokenKind::SlashEqual => "/=",
            TokenKind::PercentEqual => "%=",
            TokenKind::AmpEqual => "&=",
            TokenKind::PipeEqual => "|=",
            TokenKind::CaretEqual => "^=",
            TokenKind::LessLessEqual => "<<=",
            TokenKind::GreaterGreaterEqual => ">>=",
            TokenKind::PlusPlus => "++",
            TokenKind::MinusMinus => "--",
            TokenKind::Arrow => "->",
            TokenKind::Dot => ".",
            TokenKind::Question => "?",
            TokenKind::Colon => ":",
            TokenKind::Hash => "#",
            TokenKind::HashHash => "##",

            // --- Punctuation ---
            TokenKind::LeftParen => "(",
            TokenKind::RightParen => ")",
            TokenKind::LeftBrace => "{",
            TokenKind::RightBrace => "}",
            TokenKind::LeftBracket => "[",
            TokenKind::RightBracket => "]",
            TokenKind::Semicolon => ";",
            TokenKind::Comma => ",",
            TokenKind::Ellipsis => "...",

            // --- Literals ---
            TokenKind::IntegerLiteral => "integer literal",
            TokenKind::FloatLiteral => "float literal",
            TokenKind::StringLiteral => "string literal",
            TokenKind::CharLiteral => "character literal",

            // --- Identifier ---
            TokenKind::Identifier => "identifier",

            // --- Special ---
            TokenKind::Eof => "end of file",
        }
    }

    /// Returns `true` if this token kind is any keyword (C11 or GCC extension).
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            // C11 keywords
            TokenKind::Auto
                | TokenKind::Break
                | TokenKind::Case
                | TokenKind::Char
                | TokenKind::Const
                | TokenKind::Continue
                | TokenKind::Default
                | TokenKind::Do
                | TokenKind::Double
                | TokenKind::Else
                | TokenKind::Enum
                | TokenKind::Extern
                | TokenKind::Float
                | TokenKind::For
                | TokenKind::Goto
                | TokenKind::If
                | TokenKind::Inline
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Register
                | TokenKind::Restrict
                | TokenKind::Return
                | TokenKind::Short
                | TokenKind::Signed
                | TokenKind::Sizeof
                | TokenKind::Static
                | TokenKind::Struct
                | TokenKind::Switch
                | TokenKind::Typedef
                | TokenKind::Union
                | TokenKind::Unsigned
                | TokenKind::Void
                | TokenKind::Volatile
                | TokenKind::While
                | TokenKind::Alignas
                | TokenKind::Alignof
                | TokenKind::Atomic
                | TokenKind::Bool
                | TokenKind::Complex
                | TokenKind::Generic
                | TokenKind::Imaginary
                | TokenKind::Noreturn
                | TokenKind::StaticAssert
                | TokenKind::ThreadLocal
                // GCC extension keywords
                | TokenKind::GccAttribute
                | TokenKind::GccExtension
                | TokenKind::Asm
                | TokenKind::Typeof
                | TokenKind::BuiltinVaList
                | TokenKind::BuiltinVaStart
                | TokenKind::BuiltinVaEnd
                | TokenKind::BuiltinVaArg
                | TokenKind::BuiltinVaCopy
                | TokenKind::BuiltinOffsetof
                | TokenKind::BuiltinTypesCompatibleP
                | TokenKind::BuiltinExpect
                | TokenKind::BuiltinUnreachable
                | TokenKind::BuiltinConstantP
                | TokenKind::BuiltinChooseExpr
                | TokenKind::BuiltinBswap16
                | TokenKind::BuiltinBswap32
                | TokenKind::BuiltinBswap64
                | TokenKind::BuiltinClz
                | TokenKind::BuiltinCtz
                | TokenKind::BuiltinPopcount
                | TokenKind::BuiltinFfs
                | TokenKind::BuiltinAbs
                | TokenKind::BuiltinFabsf
                | TokenKind::BuiltinFabs
                | TokenKind::BuiltinInf
                | TokenKind::BuiltinInff
                | TokenKind::BuiltinHugeVal
                | TokenKind::BuiltinHugeValf
                | TokenKind::BuiltinNan
                | TokenKind::BuiltinNanf
                | TokenKind::BuiltinTrap
                | TokenKind::BuiltinAlloca
                | TokenKind::BuiltinMemcpy
                | TokenKind::BuiltinMemset
                | TokenKind::BuiltinStrlen
                | TokenKind::BuiltinFrameAddress
                | TokenKind::GccInt128
                | TokenKind::GccLabel
                | TokenKind::GccAutoType
        )
    }

    /// Returns `true` if this token kind is a type specifier start.
    ///
    /// Includes `int`, `char`, `void`, `float`, `double`, `long`, `short`,
    /// `signed`, `unsigned`, `struct`, `union`, `enum`, `_Bool`, `_Complex`,
    /// `_Atomic`, `typeof`, `__int128`, and `__auto_type`.
    pub fn is_type_specifier(&self) -> bool {
        matches!(
            self,
            TokenKind::Void
                | TokenKind::Char
                | TokenKind::Short
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Float
                | TokenKind::Double
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Bool
                | TokenKind::Complex
                | TokenKind::Imaginary
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
                | TokenKind::Atomic
                | TokenKind::Typeof
                | TokenKind::GccInt128
                | TokenKind::GccAutoType
                | TokenKind::BuiltinVaList
        )
    }

    /// Returns `true` if this token kind is a type qualifier
    /// (`const`, `volatile`, `restrict`, `_Atomic`).
    pub fn is_type_qualifier(&self) -> bool {
        matches!(
            self,
            TokenKind::Const
                | TokenKind::Volatile
                | TokenKind::Restrict
                | TokenKind::Atomic
        )
    }

    /// Returns `true` if this token kind is a storage class specifier
    /// (`static`, `extern`, `auto`, `register`, `typedef`, `_Thread_local`).
    pub fn is_storage_class(&self) -> bool {
        matches!(
            self,
            TokenKind::Static
                | TokenKind::Extern
                | TokenKind::Auto
                | TokenKind::Register
                | TokenKind::Typedef
                | TokenKind::ThreadLocal
        )
    }

    /// Returns the binary operator precedence for expression parsing.
    ///
    /// Higher values bind tighter. Returns `None` if this token kind is not
    /// a binary operator. The precedence levels follow the C11 standard:
    ///
    /// | Level | Operators                        |
    /// |-------|----------------------------------|
    /// |  1    | `,` (comma)                      |
    /// |  2    | `=`, `+=`, `-=`, … (assignment)  |
    /// |  3    | `?` (ternary conditional)        |
    /// |  4    | `\|\|` (logical or)              |
    /// |  5    | `&&` (logical and)               |
    /// |  6    | `\|` (bitwise or)                |
    /// |  7    | `^` (bitwise xor)                |
    /// |  8    | `&` (bitwise and)                |
    /// |  9    | `==`, `!=` (equality)            |
    /// | 10    | `<`, `>`, `<=`, `>=` (relational)|
    /// | 11    | `<<`, `>>` (shift)               |
    /// | 12    | `+`, `-` (additive)              |
    /// | 13    | `*`, `/`, `%` (multiplicative)   |
    pub fn binary_precedence(&self) -> Option<u8> {
        match self {
            // Precedence 1 — comma
            TokenKind::Comma => Some(1),

            // Precedence 2 — assignment (right-associative)
            TokenKind::Equal
            | TokenKind::PlusEqual
            | TokenKind::MinusEqual
            | TokenKind::StarEqual
            | TokenKind::SlashEqual
            | TokenKind::PercentEqual
            | TokenKind::AmpEqual
            | TokenKind::PipeEqual
            | TokenKind::CaretEqual
            | TokenKind::LessLessEqual
            | TokenKind::GreaterGreaterEqual => Some(2),

            // Precedence 3 — ternary conditional (right-associative)
            TokenKind::Question => Some(3),

            // Precedence 4 — logical OR
            TokenKind::PipePipe => Some(4),

            // Precedence 5 — logical AND
            TokenKind::AmpAmp => Some(5),

            // Precedence 6 — bitwise OR
            TokenKind::Pipe => Some(6),

            // Precedence 7 — bitwise XOR
            TokenKind::Caret => Some(7),

            // Precedence 8 — bitwise AND
            TokenKind::Amp => Some(8),

            // Precedence 9 — equality
            TokenKind::EqualEqual | TokenKind::BangEqual => Some(9),

            // Precedence 10 — relational
            TokenKind::Less
            | TokenKind::Greater
            | TokenKind::LessEqual
            | TokenKind::GreaterEqual => Some(10),

            // Precedence 11 — shift
            TokenKind::LessLess | TokenKind::GreaterGreater => Some(11),

            // Precedence 12 — additive
            TokenKind::Plus | TokenKind::Minus => Some(12),

            // Precedence 13 — multiplicative
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some(13),

            _ => None,
        }
    }

    /// Returns `true` if this operator is right-associative.
    ///
    /// In C, only assignment operators (`=`, `+=`, …) and the ternary
    /// conditional (`?:`) are right-associative. All other binary operators
    /// are left-associative.
    pub fn is_right_associative(&self) -> bool {
        matches!(
            self,
            TokenKind::Equal
                | TokenKind::PlusEqual
                | TokenKind::MinusEqual
                | TokenKind::StarEqual
                | TokenKind::SlashEqual
                | TokenKind::PercentEqual
                | TokenKind::AmpEqual
                | TokenKind::PipeEqual
                | TokenKind::CaretEqual
                | TokenKind::LessLessEqual
                | TokenKind::GreaterGreaterEqual
                | TokenKind::Question
        )
    }
}

// ===========================================================================
// Display implementations — GCC-compatible diagnostic formatting
// ===========================================================================

/// Displays the canonical string form of the token kind.
///
/// For keywords: the keyword text (e.g., `int`, `void`).
/// For operators: the operator symbol (e.g., `+`, `==`).
/// For literals: a descriptive name (e.g., `integer literal`).
impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Displays a token as `'<kind>'` for diagnostics, optionally including its
/// value for identifiers and literals.
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            TokenValue::None => write!(f, "'{}'", self.kind),
            TokenValue::Identifier(_id) => write!(f, "identifier"),
            TokenValue::Integer { value, .. } => write!(f, "integer {}", value),
            TokenValue::Float { value, .. } => write!(f, "float {}", value),
            TokenValue::Str(s) => write!(f, "string \"{}\"", s),
            TokenValue::Char(c) => {
                if let Some(ch) = char::from_u32(*c) {
                    write!(f, "char '{}'", ch)
                } else {
                    write!(f, "char \\x{:04x}", c)
                }
            }
        }
    }
}

/// Displays a `TokenValue` for debugging purposes.
impl fmt::Display for TokenValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenValue::None => f.write_str("(none)"),
            TokenValue::Identifier(id) => write!(f, "id:{}", id),
            TokenValue::Integer { value, base, suffix } => {
                let base_prefix = match base {
                    NumericBase::Decimal => "",
                    NumericBase::Hexadecimal => "0x",
                    NumericBase::Octal => "0",
                    NumericBase::Binary => "0b",
                };
                write!(f, "{}{}{:?}", base_prefix, value, suffix)
            }
            TokenValue::Float { value, suffix } => write!(f, "{}{:?}", value, suffix),
            TokenValue::Str(s) => write!(f, "\"{}\"", s),
            TokenValue::Char(c) => write!(f, "'{}'", c),
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::{FileId, SourceLocation};

    // -- Helpers -------------------------------------------------------------

    /// Build a dummy SourceSpan for test token construction.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    /// Build a simple SourceSpan for tests.
    fn test_span() -> SourceSpan {
        let start = SourceLocation {
            file_id: FileId(0),
            byte_offset: 0,
            line: 1,
            column: 1,
        };
        let end = SourceLocation {
            file_id: FileId(0),
            byte_offset: 3,
            line: 1,
            column: 4,
        };
        SourceSpan::new(start, end)
    }

    // -- TokenKind variant count --------------------------------------------

    /// Verify that TokenKind has at least 137 variants by listing all of them.
    #[test]
    fn test_token_kind_variant_count_at_least_137() {
        // We enumerate a representative set covering all categories. The enum
        // is exhaustive so adding/removing variants causes a compiler error.
        let all_kinds: Vec<TokenKind> = vec![
            // C11 keywords (44)
            TokenKind::Auto,
            TokenKind::Break,
            TokenKind::Case,
            TokenKind::Char,
            TokenKind::Const,
            TokenKind::Continue,
            TokenKind::Default,
            TokenKind::Do,
            TokenKind::Double,
            TokenKind::Else,
            TokenKind::Enum,
            TokenKind::Extern,
            TokenKind::Float,
            TokenKind::For,
            TokenKind::Goto,
            TokenKind::If,
            TokenKind::Inline,
            TokenKind::Int,
            TokenKind::Long,
            TokenKind::Register,
            TokenKind::Restrict,
            TokenKind::Return,
            TokenKind::Short,
            TokenKind::Signed,
            TokenKind::Sizeof,
            TokenKind::Static,
            TokenKind::Struct,
            TokenKind::Switch,
            TokenKind::Typedef,
            TokenKind::Union,
            TokenKind::Unsigned,
            TokenKind::Void,
            TokenKind::Volatile,
            TokenKind::While,
            TokenKind::Alignas,
            TokenKind::Alignof,
            TokenKind::Atomic,
            TokenKind::Bool,
            TokenKind::Complex,
            TokenKind::Generic,
            TokenKind::Imaginary,
            TokenKind::Noreturn,
            TokenKind::StaticAssert,
            TokenKind::ThreadLocal,
            // GCC extension keywords (40)
            TokenKind::GccAttribute,
            TokenKind::GccExtension,
            TokenKind::Asm,
            TokenKind::Typeof,
            TokenKind::BuiltinVaList,
            TokenKind::BuiltinVaStart,
            TokenKind::BuiltinVaEnd,
            TokenKind::BuiltinVaArg,
            TokenKind::BuiltinVaCopy,
            TokenKind::BuiltinOffsetof,
            TokenKind::BuiltinTypesCompatibleP,
            TokenKind::BuiltinExpect,
            TokenKind::BuiltinUnreachable,
            TokenKind::BuiltinConstantP,
            TokenKind::BuiltinChooseExpr,
            TokenKind::BuiltinBswap16,
            TokenKind::BuiltinBswap32,
            TokenKind::BuiltinBswap64,
            TokenKind::BuiltinClz,
            TokenKind::BuiltinCtz,
            TokenKind::BuiltinPopcount,
            TokenKind::BuiltinFfs,
            TokenKind::BuiltinAbs,
            TokenKind::BuiltinFabsf,
            TokenKind::BuiltinFabs,
            TokenKind::BuiltinInf,
            TokenKind::BuiltinInff,
            TokenKind::BuiltinHugeVal,
            TokenKind::BuiltinHugeValf,
            TokenKind::BuiltinNan,
            TokenKind::BuiltinNanf,
            TokenKind::BuiltinTrap,
            TokenKind::BuiltinAlloca,
            TokenKind::BuiltinMemcpy,
            TokenKind::BuiltinMemset,
            TokenKind::BuiltinStrlen,
            TokenKind::BuiltinFrameAddress,
            TokenKind::GccInt128,
            TokenKind::GccLabel,
            TokenKind::GccAutoType,
            // Operators (39)
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::EqualEqual,
            TokenKind::BangEqual,
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::LessEqual,
            TokenKind::GreaterEqual,
            TokenKind::AmpAmp,
            TokenKind::PipePipe,
            TokenKind::Bang,
            TokenKind::Amp,
            TokenKind::Pipe,
            TokenKind::Caret,
            TokenKind::Tilde,
            TokenKind::LessLess,
            TokenKind::GreaterGreater,
            TokenKind::Equal,
            TokenKind::PlusEqual,
            TokenKind::MinusEqual,
            TokenKind::StarEqual,
            TokenKind::SlashEqual,
            TokenKind::PercentEqual,
            TokenKind::AmpEqual,
            TokenKind::PipeEqual,
            TokenKind::CaretEqual,
            TokenKind::LessLessEqual,
            TokenKind::GreaterGreaterEqual,
            TokenKind::PlusPlus,
            TokenKind::MinusMinus,
            TokenKind::Arrow,
            TokenKind::Dot,
            TokenKind::Question,
            TokenKind::Colon,
            TokenKind::Hash,
            TokenKind::HashHash,
            // Punctuation (9)
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            TokenKind::Semicolon,
            TokenKind::Comma,
            TokenKind::Ellipsis,
            // Literals (4)
            TokenKind::IntegerLiteral,
            TokenKind::FloatLiteral,
            TokenKind::StringLiteral,
            TokenKind::CharLiteral,
            // Identifier + EOF (2)
            TokenKind::Identifier,
            TokenKind::Eof,
        ];
        assert!(
            all_kinds.len() >= 137,
            "expected >= 137 TokenKind variants, got {}",
            all_kinds.len()
        );
    }

    // -- TokenKind Copy trait -----------------------------------------------

    #[test]
    fn test_token_kind_is_copy() {
        let k = TokenKind::Int;
        let k2 = k; // Copy
        assert_eq!(k, k2);
    }

    // -- Token construction -------------------------------------------------

    #[test]
    fn test_token_new_and_fields() {
        let span = test_span();
        let tok = Token::new(TokenKind::Int, span, TokenValue::None);
        assert_eq!(tok.kind, TokenKind::Int);
        assert_eq!(tok.span, span);
        assert!(matches!(tok.value, TokenValue::None));
    }

    #[test]
    fn test_token_new_identifier() {
        let span = dummy_span();
        let id = InternId::from_raw(42);
        let tok = Token::new(TokenKind::Identifier, span, TokenValue::Identifier(id));
        assert_eq!(tok.kind, TokenKind::Identifier);
        assert!(matches!(tok.value, TokenValue::Identifier(x) if x == id));
    }

    #[test]
    fn test_token_new_integer_literal() {
        let span = dummy_span();
        let tok = Token::new(
            TokenKind::IntegerLiteral,
            span,
            TokenValue::Integer {
                value: 42,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
            },
        );
        assert_eq!(tok.kind, TokenKind::IntegerLiteral);
        if let TokenValue::Integer { value, suffix, base } = &tok.value {
            assert_eq!(*value, 42);
            assert_eq!(*suffix, IntSuffix::None);
            assert_eq!(*base, NumericBase::Decimal);
        } else {
            panic!("expected Integer token value");
        }
    }

    #[test]
    fn test_token_new_float_literal() {
        let span = dummy_span();
        let tok = Token::new(
            TokenKind::FloatLiteral,
            span,
            TokenValue::Float {
                value: 3.14,
                suffix: FloatSuffix::Float,
            },
        );
        assert_eq!(tok.kind, TokenKind::FloatLiteral);
        if let TokenValue::Float { value, suffix } = &tok.value {
            assert!((*value - 3.14).abs() < f64::EPSILON);
            assert_eq!(*suffix, FloatSuffix::Float);
        } else {
            panic!("expected Float token value");
        }
    }

    #[test]
    fn test_token_new_string_literal() {
        let span = dummy_span();
        let tok = Token::new(
            TokenKind::StringLiteral,
            span,
            TokenValue::Str("hello".to_string()),
        );
        assert_eq!(tok.kind, TokenKind::StringLiteral);
        assert!(matches!(&tok.value, TokenValue::Str(s) if s == "hello"));
    }

    #[test]
    fn test_token_new_char_literal() {
        let span = dummy_span();
        let tok = Token::new(TokenKind::CharLiteral, span, TokenValue::Char(b'A' as u32));
        assert_eq!(tok.kind, TokenKind::CharLiteral);
        assert!(matches!(tok.value, TokenValue::Char(65)));
    }

    // -- Token::is() --------------------------------------------------------

    #[test]
    fn test_token_is_match() {
        let tok = Token::new(TokenKind::Plus, dummy_span(), TokenValue::None);
        assert!(tok.is(TokenKind::Plus));
        assert!(!tok.is(TokenKind::Minus));
    }

    // -- Token::is_keyword() ------------------------------------------------

    #[test]
    fn test_token_is_keyword_true_for_c11() {
        let kw_tok = Token::new(TokenKind::Int, dummy_span(), TokenValue::None);
        assert!(kw_tok.is_keyword());
    }

    #[test]
    fn test_token_is_keyword_true_for_gcc() {
        let gcc_tok = Token::new(TokenKind::GccAttribute, dummy_span(), TokenValue::None);
        assert!(gcc_tok.is_keyword());
    }

    #[test]
    fn test_token_is_keyword_false_for_operator() {
        let op_tok = Token::new(TokenKind::Plus, dummy_span(), TokenValue::None);
        assert!(!op_tok.is_keyword());
    }

    #[test]
    fn test_token_is_keyword_false_for_identifier() {
        let id_tok = Token::new(
            TokenKind::Identifier,
            dummy_span(),
            TokenValue::Identifier(InternId::from_raw(0)),
        );
        assert!(!id_tok.is_keyword());
    }

    // -- Token::is_literal() ------------------------------------------------

    #[test]
    fn test_token_is_literal_integer() {
        let tok = Token::new(
            TokenKind::IntegerLiteral,
            dummy_span(),
            TokenValue::Integer {
                value: 1,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
            },
        );
        assert!(tok.is_literal());
    }

    #[test]
    fn test_token_is_literal_float() {
        let tok = Token::new(
            TokenKind::FloatLiteral,
            dummy_span(),
            TokenValue::Float {
                value: 1.0,
                suffix: FloatSuffix::None,
            },
        );
        assert!(tok.is_literal());
    }

    #[test]
    fn test_token_is_literal_string() {
        let tok = Token::new(
            TokenKind::StringLiteral,
            dummy_span(),
            TokenValue::Str(String::new()),
        );
        assert!(tok.is_literal());
    }

    #[test]
    fn test_token_is_literal_char() {
        let tok = Token::new(TokenKind::CharLiteral, dummy_span(), TokenValue::Char(0));
        assert!(tok.is_literal());
    }

    #[test]
    fn test_token_is_literal_false_for_keyword() {
        let tok = Token::new(TokenKind::Int, dummy_span(), TokenValue::None);
        assert!(!tok.is_literal());
    }

    // -- Token::is_eof() ----------------------------------------------------

    #[test]
    fn test_token_is_eof_true() {
        let tok = Token::new(TokenKind::Eof, dummy_span(), TokenValue::None);
        assert!(tok.is_eof());
    }

    #[test]
    fn test_token_is_eof_false() {
        let tok = Token::new(TokenKind::Int, dummy_span(), TokenValue::None);
        assert!(!tok.is_eof());
    }

    // -- Token::is_assignment_op() ------------------------------------------

    #[test]
    fn test_token_is_assignment_op_equals() {
        let tok = Token::new(TokenKind::Equal, dummy_span(), TokenValue::None);
        assert!(tok.is_assignment_op());
    }

    #[test]
    fn test_token_is_assignment_op_compound() {
        for kind in &[
            TokenKind::PlusEqual,
            TokenKind::MinusEqual,
            TokenKind::StarEqual,
            TokenKind::SlashEqual,
            TokenKind::PercentEqual,
            TokenKind::AmpEqual,
            TokenKind::PipeEqual,
            TokenKind::CaretEqual,
            TokenKind::LessLessEqual,
            TokenKind::GreaterGreaterEqual,
        ] {
            let tok = Token::new(*kind, dummy_span(), TokenValue::None);
            assert!(
                tok.is_assignment_op(),
                "{:?} should be an assignment operator",
                kind
            );
        }
    }

    #[test]
    fn test_token_is_assignment_op_false_for_comparison() {
        let tok = Token::new(TokenKind::EqualEqual, dummy_span(), TokenValue::None);
        assert!(!tok.is_assignment_op());
    }

    // -- Token::is_comparison_op() ------------------------------------------

    #[test]
    fn test_token_is_comparison_op() {
        for kind in &[
            TokenKind::EqualEqual,
            TokenKind::BangEqual,
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::LessEqual,
            TokenKind::GreaterEqual,
        ] {
            let tok = Token::new(*kind, dummy_span(), TokenValue::None);
            assert!(
                tok.is_comparison_op(),
                "{:?} should be a comparison operator",
                kind
            );
        }
    }

    #[test]
    fn test_token_is_comparison_op_false_for_assignment() {
        let tok = Token::new(TokenKind::Equal, dummy_span(), TokenValue::None);
        assert!(!tok.is_comparison_op());
    }

    // -- TokenKind::is_type_specifier() ------------------------------------

    #[test]
    fn test_is_type_specifier_base_types() {
        let specs = [
            TokenKind::Void,
            TokenKind::Char,
            TokenKind::Short,
            TokenKind::Int,
            TokenKind::Long,
            TokenKind::Float,
            TokenKind::Double,
            TokenKind::Signed,
            TokenKind::Unsigned,
        ];
        for kind in &specs {
            assert!(
                kind.is_type_specifier(),
                "{:?} should be a type specifier",
                kind
            );
        }
    }

    #[test]
    fn test_is_type_specifier_c11_and_gcc() {
        assert!(TokenKind::Bool.is_type_specifier());
        assert!(TokenKind::Complex.is_type_specifier());
        assert!(TokenKind::Struct.is_type_specifier());
        assert!(TokenKind::Union.is_type_specifier());
        assert!(TokenKind::Enum.is_type_specifier());
        assert!(TokenKind::Typeof.is_type_specifier());
        assert!(TokenKind::GccInt128.is_type_specifier());
        assert!(TokenKind::GccAutoType.is_type_specifier());
    }

    #[test]
    fn test_is_type_specifier_false_for_non_type() {
        assert!(!TokenKind::If.is_type_specifier());
        assert!(!TokenKind::Plus.is_type_specifier());
        assert!(!TokenKind::Identifier.is_type_specifier());
    }

    // -- TokenKind::is_type_qualifier() ------------------------------------

    #[test]
    fn test_is_type_qualifier() {
        assert!(TokenKind::Const.is_type_qualifier());
        assert!(TokenKind::Volatile.is_type_qualifier());
        assert!(TokenKind::Restrict.is_type_qualifier());
        assert!(TokenKind::Atomic.is_type_qualifier());
    }

    #[test]
    fn test_is_type_qualifier_false() {
        assert!(!TokenKind::Static.is_type_qualifier());
        assert!(!TokenKind::Int.is_type_qualifier());
    }

    // -- TokenKind::is_storage_class() -------------------------------------

    #[test]
    fn test_is_storage_class() {
        assert!(TokenKind::Static.is_storage_class());
        assert!(TokenKind::Extern.is_storage_class());
        assert!(TokenKind::Auto.is_storage_class());
        assert!(TokenKind::Register.is_storage_class());
        assert!(TokenKind::Typedef.is_storage_class());
        assert!(TokenKind::ThreadLocal.is_storage_class());
    }

    #[test]
    fn test_is_storage_class_false() {
        assert!(!TokenKind::Const.is_storage_class());
        assert!(!TokenKind::Int.is_storage_class());
    }

    // -- TokenKind::binary_precedence() ------------------------------------

    #[test]
    fn test_binary_precedence_multiplicative() {
        assert_eq!(TokenKind::Star.binary_precedence(), Some(13));
        assert_eq!(TokenKind::Slash.binary_precedence(), Some(13));
        assert_eq!(TokenKind::Percent.binary_precedence(), Some(13));
    }

    #[test]
    fn test_binary_precedence_additive() {
        assert_eq!(TokenKind::Plus.binary_precedence(), Some(12));
        assert_eq!(TokenKind::Minus.binary_precedence(), Some(12));
    }

    #[test]
    fn test_binary_precedence_shift() {
        assert_eq!(TokenKind::LessLess.binary_precedence(), Some(11));
        assert_eq!(TokenKind::GreaterGreater.binary_precedence(), Some(11));
    }

    #[test]
    fn test_binary_precedence_relational() {
        assert_eq!(TokenKind::Less.binary_precedence(), Some(10));
        assert_eq!(TokenKind::Greater.binary_precedence(), Some(10));
        assert_eq!(TokenKind::LessEqual.binary_precedence(), Some(10));
        assert_eq!(TokenKind::GreaterEqual.binary_precedence(), Some(10));
    }

    #[test]
    fn test_binary_precedence_equality() {
        assert_eq!(TokenKind::EqualEqual.binary_precedence(), Some(9));
        assert_eq!(TokenKind::BangEqual.binary_precedence(), Some(9));
    }

    #[test]
    fn test_binary_precedence_bitwise() {
        assert_eq!(TokenKind::Amp.binary_precedence(), Some(8));
        assert_eq!(TokenKind::Caret.binary_precedence(), Some(7));
        assert_eq!(TokenKind::Pipe.binary_precedence(), Some(6));
    }

    #[test]
    fn test_binary_precedence_logical() {
        assert_eq!(TokenKind::AmpAmp.binary_precedence(), Some(5));
        assert_eq!(TokenKind::PipePipe.binary_precedence(), Some(4));
    }

    #[test]
    fn test_binary_precedence_ternary() {
        assert_eq!(TokenKind::Question.binary_precedence(), Some(3));
    }

    #[test]
    fn test_binary_precedence_assignment() {
        assert_eq!(TokenKind::Equal.binary_precedence(), Some(2));
        assert_eq!(TokenKind::PlusEqual.binary_precedence(), Some(2));
        assert_eq!(TokenKind::GreaterGreaterEqual.binary_precedence(), Some(2));
    }

    #[test]
    fn test_binary_precedence_comma() {
        assert_eq!(TokenKind::Comma.binary_precedence(), Some(1));
    }

    #[test]
    fn test_binary_precedence_none_for_unary() {
        assert_eq!(TokenKind::Bang.binary_precedence(), None);
        assert_eq!(TokenKind::Tilde.binary_precedence(), None);
        assert_eq!(TokenKind::PlusPlus.binary_precedence(), None);
    }

    #[test]
    fn test_binary_precedence_none_for_non_operator() {
        assert_eq!(TokenKind::Int.binary_precedence(), None);
        assert_eq!(TokenKind::Identifier.binary_precedence(), None);
        assert_eq!(TokenKind::Eof.binary_precedence(), None);
    }

    #[test]
    fn test_multiplicative_binds_tighter_than_additive() {
        let mul = TokenKind::Star.binary_precedence().unwrap();
        let add = TokenKind::Plus.binary_precedence().unwrap();
        assert!(mul > add, "* should bind tighter than +");
    }

    // -- TokenKind::is_right_associative() ---------------------------------

    #[test]
    fn test_is_right_associative_assignment() {
        assert!(TokenKind::Equal.is_right_associative());
        assert!(TokenKind::PlusEqual.is_right_associative());
        assert!(TokenKind::MinusEqual.is_right_associative());
        assert!(TokenKind::LessLessEqual.is_right_associative());
        assert!(TokenKind::GreaterGreaterEqual.is_right_associative());
    }

    #[test]
    fn test_is_right_associative_ternary() {
        assert!(TokenKind::Question.is_right_associative());
    }

    #[test]
    fn test_is_right_associative_false_for_left() {
        assert!(!TokenKind::Plus.is_right_associative());
        assert!(!TokenKind::Star.is_right_associative());
        assert!(!TokenKind::AmpAmp.is_right_associative());
        assert!(!TokenKind::EqualEqual.is_right_associative());
    }

    // -- TokenKind::as_str() -----------------------------------------------

    #[test]
    fn test_as_str_keywords() {
        assert_eq!(TokenKind::Int.as_str(), "int");
        assert_eq!(TokenKind::Void.as_str(), "void");
        assert_eq!(TokenKind::Struct.as_str(), "struct");
        assert_eq!(TokenKind::Alignas.as_str(), "_Alignas");
        assert_eq!(TokenKind::Bool.as_str(), "_Bool");
        assert_eq!(TokenKind::StaticAssert.as_str(), "_Static_assert");
    }

    #[test]
    fn test_as_str_gcc_keywords() {
        assert_eq!(TokenKind::GccAttribute.as_str(), "__attribute__");
        assert_eq!(TokenKind::GccExtension.as_str(), "__extension__");
        assert_eq!(TokenKind::Asm.as_str(), "asm");
        assert_eq!(TokenKind::Typeof.as_str(), "typeof");
        assert_eq!(TokenKind::BuiltinVaStart.as_str(), "__builtin_va_start");
    }

    #[test]
    fn test_as_str_operators() {
        assert_eq!(TokenKind::Plus.as_str(), "+");
        assert_eq!(TokenKind::EqualEqual.as_str(), "==");
        assert_eq!(TokenKind::AmpAmp.as_str(), "&&");
        assert_eq!(TokenKind::Arrow.as_str(), "->");
        assert_eq!(TokenKind::LessLessEqual.as_str(), "<<=");
        assert_eq!(TokenKind::GreaterGreaterEqual.as_str(), ">>=");
    }

    #[test]
    fn test_as_str_punctuation() {
        assert_eq!(TokenKind::LeftParen.as_str(), "(");
        assert_eq!(TokenKind::RightBrace.as_str(), "}");
        assert_eq!(TokenKind::Semicolon.as_str(), ";");
        assert_eq!(TokenKind::Ellipsis.as_str(), "...");
    }

    #[test]
    fn test_as_str_literals_and_special() {
        assert_eq!(TokenKind::IntegerLiteral.as_str(), "integer literal");
        assert_eq!(TokenKind::FloatLiteral.as_str(), "float literal");
        assert_eq!(TokenKind::StringLiteral.as_str(), "string literal");
        assert_eq!(TokenKind::CharLiteral.as_str(), "character literal");
        assert_eq!(TokenKind::Identifier.as_str(), "identifier");
        assert_eq!(TokenKind::Eof.as_str(), "end of file");
    }

    // -- Display trait -------------------------------------------------------

    #[test]
    fn test_display_token_kind() {
        assert_eq!(format!("{}", TokenKind::Int), "int");
        assert_eq!(format!("{}", TokenKind::Plus), "+");
        assert_eq!(format!("{}", TokenKind::Eof), "end of file");
        assert_eq!(format!("{}", TokenKind::Ellipsis), "...");
    }

    #[test]
    fn test_display_token_keyword() {
        let tok = Token::new(TokenKind::Int, dummy_span(), TokenValue::None);
        let display = format!("{}", tok);
        assert_eq!(display, "'int'");
    }

    #[test]
    fn test_display_token_integer() {
        let tok = Token::new(
            TokenKind::IntegerLiteral,
            dummy_span(),
            TokenValue::Integer {
                value: 42,
                suffix: IntSuffix::None,
                base: NumericBase::Decimal,
            },
        );
        let display = format!("{}", tok);
        assert_eq!(display, "integer 42");
    }

    #[test]
    fn test_display_token_string() {
        let tok = Token::new(
            TokenKind::StringLiteral,
            dummy_span(),
            TokenValue::Str("hello".to_string()),
        );
        let display = format!("{}", tok);
        assert!(display.contains("hello"));
    }

    // -- IntSuffix / FloatSuffix / NumericBase traits -----------------------

    #[test]
    fn test_int_suffix_variants() {
        let _none = IntSuffix::None;
        let _unsigned = IntSuffix::Unsigned;
        let _long = IntSuffix::Long;
        let _ulong = IntSuffix::ULong;
        let _longlong = IntSuffix::LongLong;
        let _ulonglong = IntSuffix::ULongLong;
        // Verify Copy
        let a = IntSuffix::Long;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_float_suffix_variants() {
        let _none = FloatSuffix::None;
        let _float = FloatSuffix::Float;
        let _long = FloatSuffix::Long;
        // Verify Copy
        let a = FloatSuffix::Float;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_numeric_base_variants() {
        let _dec = NumericBase::Decimal;
        let _hex = NumericBase::Hexadecimal;
        let _oct = NumericBase::Octal;
        let _bin = NumericBase::Binary;
        // Verify Copy
        let a = NumericBase::Hexadecimal;
        let b = a;
        assert_eq!(a, b);
    }

    // -- TokenKind keyword completeness ------------------------------------

    #[test]
    fn test_all_44_c11_keywords_are_keywords() {
        let c11_keywords = [
            TokenKind::Auto,
            TokenKind::Break,
            TokenKind::Case,
            TokenKind::Char,
            TokenKind::Const,
            TokenKind::Continue,
            TokenKind::Default,
            TokenKind::Do,
            TokenKind::Double,
            TokenKind::Else,
            TokenKind::Enum,
            TokenKind::Extern,
            TokenKind::Float,
            TokenKind::For,
            TokenKind::Goto,
            TokenKind::If,
            TokenKind::Inline,
            TokenKind::Int,
            TokenKind::Long,
            TokenKind::Register,
            TokenKind::Restrict,
            TokenKind::Return,
            TokenKind::Short,
            TokenKind::Signed,
            TokenKind::Sizeof,
            TokenKind::Static,
            TokenKind::Struct,
            TokenKind::Switch,
            TokenKind::Typedef,
            TokenKind::Union,
            TokenKind::Unsigned,
            TokenKind::Void,
            TokenKind::Volatile,
            TokenKind::While,
            TokenKind::Alignas,
            TokenKind::Alignof,
            TokenKind::Atomic,
            TokenKind::Bool,
            TokenKind::Complex,
            TokenKind::Generic,
            TokenKind::Imaginary,
            TokenKind::Noreturn,
            TokenKind::StaticAssert,
            TokenKind::ThreadLocal,
        ];
        assert_eq!(c11_keywords.len(), 44, "expected exactly 44 C11 keywords");
        for kw in &c11_keywords {
            assert!(kw.is_keyword(), "{:?} should be a keyword", kw);
        }
    }

    // -- TokenKind::is_keyword() comprehensive check -----------------------

    #[test]
    fn test_is_keyword_false_for_all_operators() {
        let operators = [
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::EqualEqual,
            TokenKind::BangEqual,
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::LessEqual,
            TokenKind::GreaterEqual,
            TokenKind::AmpAmp,
            TokenKind::PipePipe,
            TokenKind::Bang,
            TokenKind::Amp,
            TokenKind::Pipe,
            TokenKind::Caret,
            TokenKind::Tilde,
            TokenKind::LessLess,
            TokenKind::GreaterGreater,
            TokenKind::Equal,
            TokenKind::Arrow,
            TokenKind::Dot,
            TokenKind::Question,
            TokenKind::Colon,
            TokenKind::Hash,
            TokenKind::HashHash,
        ];
        for op in &operators {
            assert!(!op.is_keyword(), "{:?} should not be a keyword", op);
        }
    }

    #[test]
    fn test_is_keyword_false_for_punctuation() {
        let puncts = [
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            TokenKind::Semicolon,
            TokenKind::Ellipsis,
        ];
        for p in &puncts {
            assert!(!p.is_keyword(), "{:?} should not be a keyword", p);
        }
    }

    #[test]
    fn test_is_keyword_false_for_literals_and_special() {
        assert!(!TokenKind::IntegerLiteral.is_keyword());
        assert!(!TokenKind::FloatLiteral.is_keyword());
        assert!(!TokenKind::StringLiteral.is_keyword());
        assert!(!TokenKind::CharLiteral.is_keyword());
        assert!(!TokenKind::Identifier.is_keyword());
        assert!(!TokenKind::Eof.is_keyword());
    }

    // -- Token value accessors via pattern matching -------------------------

    #[test]
    fn test_token_value_none() {
        let v = TokenValue::None;
        assert!(matches!(v, TokenValue::None));
    }

    #[test]
    fn test_token_value_identifier() {
        let id = InternId::from_raw(10);
        let v = TokenValue::Identifier(id);
        assert!(matches!(v, TokenValue::Identifier(x) if x == InternId::from_raw(10)));
    }

    // -- TokenKind Eq / Hash ------------------------------------------------

    #[test]
    fn test_token_kind_eq() {
        assert_eq!(TokenKind::Int, TokenKind::Int);
        assert_ne!(TokenKind::Int, TokenKind::Long);
    }

    #[test]
    fn test_token_kind_hash_consistent() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TokenKind::Int);
        set.insert(TokenKind::Long);
        set.insert(TokenKind::Int); // duplicate
        assert_eq!(set.len(), 2);
    }
}
