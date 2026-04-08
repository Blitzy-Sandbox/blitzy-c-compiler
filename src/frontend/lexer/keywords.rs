//! C11 + GCC Extension Keyword Lookup Table
//!
//! This module provides the keyword lookup table that maps C identifier strings
//! to their corresponding [`TokenKind`] variants. It is the authoritative source
//! for distinguishing keywords from plain identifiers during lexical analysis.
//!
//! # Coverage
//!
//! The keyword table contains:
//!
//! | Category                        | Count |
//! |---------------------------------|-------|
//! | C89/C99 standard keywords       |    34 |
//! | C11-specific keywords           |    10 |
//! | GCC extension keywords          |    ~44 |
//! | GCC qualifier alternate spellings |  10 |
//! | **Total map entries**           | **~98** |
//!
//! All 44 C11 keywords plus all GCC extension keywords required by the AAP
//! (§0.7 "C11 + GCC Extensions Compliance Rule") are present, including
//! `__attribute__`, `__builtin_*` intrinsics, inline assembly keywords
//! (`asm`/`__asm__`/`__asm`), `typeof`/`__typeof__`/`__typeof`,
//! `__extension__`, and double-underscore qualifier alternate spellings
//! (`__inline__`, `__volatile__`, `__const__`, `__restrict__`, `__signed__`).
//!
//! # Performance
//!
//! The keyword map is a [`HashMap<&'static str, TokenKind>`] lazily initialized
//! via [`std::sync::OnceLock`] on first access. Subsequent lookups are O(1)
//! amortized with zero allocation cost, which is critical since keyword lookup
//! executes for every identifier token during lexing.
//!
//! # Integration
//!
//! Called by `Lexer::scan_token()` in [`super::mod`] to classify scanned
//! identifier strings. The lexer passes the complete identifier string to
//! [`lookup_keyword`]; if it returns `Some(kind)`, the token is a keyword;
//! otherwise it is a plain `TokenKind::Identifier`.
//!
//! Per AAP §0.5.1 Group 2: "C11 keyword table plus GCC extension keywords."
//!
//! # Design Decisions
//!
//! * **`OnceLock` over `lazy_static`/`once_cell`**: Uses only `std` (stable
//!   since Rust 1.70), honouring the zero external dependency constraint.
//! * **HashMap over `match`**: A hash map scales cleanly to ~100 entries with
//!   O(1) lookup, while a large `match` would produce a linear scan or require
//!   the compiler to optimise into a jump table — HashMap is explicit and
//!   predictable.
//! * **`with_capacity(100)`**: Pre-allocates for ~98 entries to avoid
//!   re-hashing during initialization.
//! * **`copied()` on lookup**: `TokenKind` is `Copy`, so we return a cheap copy
//!   rather than a reference, simplifying the caller's lifetime handling.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::token::TokenKind;

// ===========================================================================
// Global keyword table — initialized exactly once, shared across all lookups
// ===========================================================================

/// Global keyword table, initialized on first access via [`OnceLock::get_or_init`].
///
/// The `OnceLock` wrapper guarantees thread-safe, once-only initialization and
/// provides a `&'static` reference for zero-cost repeated lookups.
static KEYWORDS: OnceLock<HashMap<&'static str, TokenKind>> = OnceLock::new();

/// Returns a reference to the lazily-initialized keyword map.
///
/// On the first call, this builds the complete keyword table containing all C11
/// keywords, GCC extension keywords, and alternate spellings. Subsequent calls
/// return the cached reference with no allocation.
fn get_keywords() -> &'static HashMap<&'static str, TokenKind> {
    KEYWORDS.get_or_init(|| {
        // Pre-allocate for ~98 entries (44 C11 + ~54 GCC/alternates).
        let mut m = HashMap::with_capacity(100);

        // ===================================================================
        // C89/C99 Standard Keywords (34)
        // ===================================================================
        m.insert("auto", TokenKind::Auto);
        m.insert("break", TokenKind::Break);
        m.insert("case", TokenKind::Case);
        m.insert("char", TokenKind::Char);
        m.insert("const", TokenKind::Const);
        m.insert("continue", TokenKind::Continue);
        m.insert("default", TokenKind::Default);
        m.insert("do", TokenKind::Do);
        m.insert("double", TokenKind::Double);
        m.insert("else", TokenKind::Else);
        m.insert("enum", TokenKind::Enum);
        m.insert("extern", TokenKind::Extern);
        m.insert("float", TokenKind::Float);
        m.insert("for", TokenKind::For);
        m.insert("goto", TokenKind::Goto);
        m.insert("if", TokenKind::If);
        m.insert("inline", TokenKind::Inline);
        m.insert("int", TokenKind::Int);
        m.insert("long", TokenKind::Long);
        m.insert("register", TokenKind::Register);
        m.insert("restrict", TokenKind::Restrict);
        m.insert("return", TokenKind::Return);
        m.insert("short", TokenKind::Short);
        m.insert("signed", TokenKind::Signed);
        m.insert("sizeof", TokenKind::Sizeof);
        m.insert("static", TokenKind::Static);
        m.insert("struct", TokenKind::Struct);
        m.insert("switch", TokenKind::Switch);
        m.insert("typedef", TokenKind::Typedef);
        m.insert("union", TokenKind::Union);
        m.insert("unsigned", TokenKind::Unsigned);
        m.insert("void", TokenKind::Void);
        m.insert("volatile", TokenKind::Volatile);
        m.insert("while", TokenKind::While);

        // ===================================================================
        // C11-Specific Keywords (10)
        // ===================================================================
        m.insert("_Alignas", TokenKind::Alignas);
        m.insert("_Alignof", TokenKind::Alignof);
        m.insert("_Atomic", TokenKind::Atomic);
        m.insert("_Bool", TokenKind::Bool);
        m.insert("_Complex", TokenKind::Complex);
        m.insert("_Generic", TokenKind::Generic);
        m.insert("_Imaginary", TokenKind::Imaginary);
        m.insert("_Noreturn", TokenKind::Noreturn);
        m.insert("_Static_assert", TokenKind::StaticAssert);
        m.insert("_Thread_local", TokenKind::ThreadLocal);

        // ===================================================================
        // GCC Extension Keywords — Core
        // ===================================================================

        // Attribute and extension meta-keywords
        m.insert("__attribute__", TokenKind::GccAttribute);
        m.insert("__extension__", TokenKind::GccExtension);

        // GCC alternate spellings for _Alignof and _Alignas
        m.insert("__alignof__", TokenKind::Alignof);
        m.insert("__alignof", TokenKind::Alignof);
        m.insert("__alignas__", TokenKind::Alignas);

        // Assembly: three accepted spellings all map to the same variant
        m.insert("asm", TokenKind::Asm);
        m.insert("__asm__", TokenKind::Asm);
        m.insert("__asm", TokenKind::Asm);

        // Typeof: three accepted spellings
        m.insert("typeof", TokenKind::Typeof);
        m.insert("__typeof__", TokenKind::Typeof);
        m.insert("__typeof", TokenKind::Typeof);

        // ===================================================================
        // GCC Extension Keywords — Varargs Builtins
        // ===================================================================
        m.insert("__builtin_va_list", TokenKind::BuiltinVaList);
        m.insert("__builtin_va_start", TokenKind::BuiltinVaStart);
        m.insert("__builtin_va_end", TokenKind::BuiltinVaEnd);
        m.insert("__builtin_va_arg", TokenKind::BuiltinVaArg);
        m.insert("__builtin_va_copy", TokenKind::BuiltinVaCopy);

        // ===================================================================
        // GCC Extension Keywords — Type / Offset Builtins
        // ===================================================================
        m.insert("__builtin_offsetof", TokenKind::BuiltinOffsetof);
        m.insert(
            "__builtin_types_compatible_p",
            TokenKind::BuiltinTypesCompatibleP,
        );

        // ===================================================================
        // GCC Extension Keywords — Control-Flow / Optimisation Builtins
        // ===================================================================
        m.insert("__builtin_expect", TokenKind::BuiltinExpect);
        m.insert("__builtin_unreachable", TokenKind::BuiltinUnreachable);
        m.insert("__builtin_constant_p", TokenKind::BuiltinConstantP);
        m.insert("__builtin_choose_expr", TokenKind::BuiltinChooseExpr);

        // ===================================================================
        // GCC Extension Keywords — Byte-Swap Builtins
        // ===================================================================
        m.insert("__builtin_bswap16", TokenKind::BuiltinBswap16);
        m.insert("__builtin_bswap32", TokenKind::BuiltinBswap32);
        m.insert("__builtin_bswap64", TokenKind::BuiltinBswap64);

        // ===================================================================
        // GCC Extension Keywords — Bit-Manipulation Builtins
        // ===================================================================
        m.insert("__builtin_clz", TokenKind::BuiltinClz);
        m.insert("__builtin_ctz", TokenKind::BuiltinCtz);
        m.insert("__builtin_popcount", TokenKind::BuiltinPopcount);
        m.insert("__builtin_ffs", TokenKind::BuiltinFfs);

        // ===================================================================
        // GCC Extension Keywords — Math Builtins
        // ===================================================================
        m.insert("__builtin_abs", TokenKind::BuiltinAbs);
        m.insert("__builtin_fabsf", TokenKind::BuiltinFabsf);
        m.insert("__builtin_fabs", TokenKind::BuiltinFabs);
        m.insert("__builtin_inf", TokenKind::BuiltinInf);
        m.insert("__builtin_inff", TokenKind::BuiltinInff);
        m.insert("__builtin_huge_val", TokenKind::BuiltinHugeVal);
        m.insert("__builtin_huge_valf", TokenKind::BuiltinHugeValf);
        m.insert("__builtin_nan", TokenKind::BuiltinNan);
        m.insert("__builtin_nanf", TokenKind::BuiltinNanf);

        // ===================================================================
        // GCC Extension Keywords — Misc Builtins
        // ===================================================================
        m.insert("__builtin_trap", TokenKind::BuiltinTrap);
        m.insert("__builtin_alloca", TokenKind::BuiltinAlloca);
        m.insert("__builtin_memcpy", TokenKind::BuiltinMemcpy);
        m.insert("__builtin_memset", TokenKind::BuiltinMemset);
        m.insert("__builtin_strlen", TokenKind::BuiltinStrlen);
        m.insert("__builtin_frame_address", TokenKind::BuiltinFrameAddress);

        // ===================================================================
        // GCC Type Extension Keywords
        // ===================================================================
        m.insert("__int128", TokenKind::GccInt128);
        m.insert("__label__", TokenKind::GccLabel);
        m.insert("__auto_type", TokenKind::GccAutoType);

        // ===================================================================
        // GCC Double-Underscore Qualifier Alternate Spellings
        //
        // These map to the same TokenKind as their standard equivalents.
        // GCC accepts these in system headers to avoid conflicts with user
        // macros that might redefine the standard keyword spellings.
        // ===================================================================

        // inline alternates
        m.insert("__inline__", TokenKind::Inline);
        m.insert("__inline", TokenKind::Inline);

        // volatile alternates
        m.insert("__volatile__", TokenKind::Volatile);
        m.insert("__volatile", TokenKind::Volatile);

        // const alternates
        m.insert("__const__", TokenKind::Const);
        m.insert("__const", TokenKind::Const);

        // restrict alternates
        m.insert("__restrict__", TokenKind::Restrict);
        m.insert("__restrict", TokenKind::Restrict);

        // signed alternates
        m.insert("__signed__", TokenKind::Signed);
        m.insert("__signed", TokenKind::Signed);

        m
    })
}

// ===========================================================================
// Public API
// ===========================================================================

/// Look up whether a given identifier string is a C11 or GCC extension keyword.
///
/// Returns the corresponding [`TokenKind`] if `name` matches a keyword in the
/// table, or `None` if it is a plain identifier.
///
/// # Arguments
///
/// * `name` — The complete identifier string scanned by the lexer. This must be
///   the full identifier text (e.g. `"int"`, `"__builtin_va_start"`), not a
///   substring or prefix.
///
/// # Performance
///
/// O(1) amortized — backed by a `HashMap` that is initialized once on first
/// call and then reused for all subsequent lookups.
///
/// # Examples
///
/// ```ignore
/// use crate::frontend::lexer::keywords::lookup_keyword;
/// use crate::frontend::lexer::token::TokenKind;
///
/// assert_eq!(lookup_keyword("int"), Some(TokenKind::Int));
/// assert_eq!(lookup_keyword("__attribute__"), Some(TokenKind::GccAttribute));
/// assert_eq!(lookup_keyword("printf"), None);
/// ```
pub fn lookup_keyword(name: &str) -> Option<TokenKind> {
    get_keywords().get(name).copied()
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::token::TokenKind;
    use super::*;

    // =======================================================================
    // C89/C99 Standard Keywords (34 tests)
    // =======================================================================

    #[test]
    fn test_c89_c99_keywords() {
        let cases: Vec<(&str, TokenKind)> = vec![
            ("auto", TokenKind::Auto),
            ("break", TokenKind::Break),
            ("case", TokenKind::Case),
            ("char", TokenKind::Char),
            ("const", TokenKind::Const),
            ("continue", TokenKind::Continue),
            ("default", TokenKind::Default),
            ("do", TokenKind::Do),
            ("double", TokenKind::Double),
            ("else", TokenKind::Else),
            ("enum", TokenKind::Enum),
            ("extern", TokenKind::Extern),
            ("float", TokenKind::Float),
            ("for", TokenKind::For),
            ("goto", TokenKind::Goto),
            ("if", TokenKind::If),
            ("inline", TokenKind::Inline),
            ("int", TokenKind::Int),
            ("long", TokenKind::Long),
            ("register", TokenKind::Register),
            ("restrict", TokenKind::Restrict),
            ("return", TokenKind::Return),
            ("short", TokenKind::Short),
            ("signed", TokenKind::Signed),
            ("sizeof", TokenKind::Sizeof),
            ("static", TokenKind::Static),
            ("struct", TokenKind::Struct),
            ("switch", TokenKind::Switch),
            ("typedef", TokenKind::Typedef),
            ("union", TokenKind::Union),
            ("unsigned", TokenKind::Unsigned),
            ("void", TokenKind::Void),
            ("volatile", TokenKind::Volatile),
            ("while", TokenKind::While),
        ];
        for (keyword, expected) in &cases {
            assert_eq!(
                lookup_keyword(keyword),
                Some(*expected),
                "Expected keyword '{}' to map to {:?}",
                keyword,
                expected
            );
        }
    }

    // =======================================================================
    // C11-Specific Keywords (10 tests)
    // =======================================================================

    #[test]
    fn test_c11_keywords() {
        let cases: Vec<(&str, TokenKind)> = vec![
            ("_Alignas", TokenKind::Alignas),
            ("_Alignof", TokenKind::Alignof),
            ("_Atomic", TokenKind::Atomic),
            ("_Bool", TokenKind::Bool),
            ("_Complex", TokenKind::Complex),
            ("_Generic", TokenKind::Generic),
            ("_Imaginary", TokenKind::Imaginary),
            ("_Noreturn", TokenKind::Noreturn),
            ("_Static_assert", TokenKind::StaticAssert),
            ("_Thread_local", TokenKind::ThreadLocal),
        ];
        for (keyword, expected) in &cases {
            assert_eq!(
                lookup_keyword(keyword),
                Some(*expected),
                "Expected C11 keyword '{}' to map to {:?}",
                keyword,
                expected
            );
        }
    }

    // =======================================================================
    // GCC Attribute and Extension Keywords
    // =======================================================================

    #[test]
    fn test_gcc_attribute() {
        assert_eq!(
            lookup_keyword("__attribute__"),
            Some(TokenKind::GccAttribute)
        );
    }

    #[test]
    fn test_gcc_extension() {
        assert_eq!(
            lookup_keyword("__extension__"),
            Some(TokenKind::GccExtension)
        );
    }

    // =======================================================================
    // Assembly Keyword — Three Spellings
    // =======================================================================

    #[test]
    fn test_asm_spellings() {
        assert_eq!(lookup_keyword("asm"), Some(TokenKind::Asm));
        assert_eq!(lookup_keyword("__asm__"), Some(TokenKind::Asm));
        assert_eq!(lookup_keyword("__asm"), Some(TokenKind::Asm));
    }

    // =======================================================================
    // Typeof Keyword — Three Spellings
    // =======================================================================

    #[test]
    fn test_typeof_spellings() {
        assert_eq!(lookup_keyword("typeof"), Some(TokenKind::Typeof));
        assert_eq!(lookup_keyword("__typeof__"), Some(TokenKind::Typeof));
        assert_eq!(lookup_keyword("__typeof"), Some(TokenKind::Typeof));
    }

    // =======================================================================
    // GCC Builtin Keywords — Varargs
    // =======================================================================

    #[test]
    fn test_builtin_varargs() {
        assert_eq!(
            lookup_keyword("__builtin_va_list"),
            Some(TokenKind::BuiltinVaList)
        );
        assert_eq!(
            lookup_keyword("__builtin_va_start"),
            Some(TokenKind::BuiltinVaStart)
        );
        assert_eq!(
            lookup_keyword("__builtin_va_end"),
            Some(TokenKind::BuiltinVaEnd)
        );
        assert_eq!(
            lookup_keyword("__builtin_va_arg"),
            Some(TokenKind::BuiltinVaArg)
        );
        assert_eq!(
            lookup_keyword("__builtin_va_copy"),
            Some(TokenKind::BuiltinVaCopy)
        );
    }

    // =======================================================================
    // GCC Builtin Keywords — Type / Offset
    // =======================================================================

    #[test]
    fn test_builtin_offsetof() {
        assert_eq!(
            lookup_keyword("__builtin_offsetof"),
            Some(TokenKind::BuiltinOffsetof)
        );
    }

    #[test]
    fn test_builtin_types_compatible_p() {
        assert_eq!(
            lookup_keyword("__builtin_types_compatible_p"),
            Some(TokenKind::BuiltinTypesCompatibleP)
        );
    }

    // =======================================================================
    // GCC Builtin Keywords — Control-Flow / Optimisation
    // =======================================================================

    #[test]
    fn test_builtin_control_flow() {
        assert_eq!(
            lookup_keyword("__builtin_expect"),
            Some(TokenKind::BuiltinExpect)
        );
        assert_eq!(
            lookup_keyword("__builtin_unreachable"),
            Some(TokenKind::BuiltinUnreachable)
        );
        assert_eq!(
            lookup_keyword("__builtin_constant_p"),
            Some(TokenKind::BuiltinConstantP)
        );
        assert_eq!(
            lookup_keyword("__builtin_choose_expr"),
            Some(TokenKind::BuiltinChooseExpr)
        );
    }

    // =======================================================================
    // GCC Builtin Keywords — Byte-Swap
    // =======================================================================

    #[test]
    fn test_builtin_bswap() {
        assert_eq!(
            lookup_keyword("__builtin_bswap16"),
            Some(TokenKind::BuiltinBswap16)
        );
        assert_eq!(
            lookup_keyword("__builtin_bswap32"),
            Some(TokenKind::BuiltinBswap32)
        );
        assert_eq!(
            lookup_keyword("__builtin_bswap64"),
            Some(TokenKind::BuiltinBswap64)
        );
    }

    // =======================================================================
    // GCC Builtin Keywords — Bit-Manipulation
    // =======================================================================

    #[test]
    fn test_builtin_bit_manipulation() {
        assert_eq!(lookup_keyword("__builtin_clz"), Some(TokenKind::BuiltinClz));
        assert_eq!(lookup_keyword("__builtin_ctz"), Some(TokenKind::BuiltinCtz));
        assert_eq!(
            lookup_keyword("__builtin_popcount"),
            Some(TokenKind::BuiltinPopcount)
        );
        assert_eq!(lookup_keyword("__builtin_ffs"), Some(TokenKind::BuiltinFfs));
    }

    // =======================================================================
    // GCC Builtin Keywords — Math
    // =======================================================================

    #[test]
    fn test_builtin_math() {
        assert_eq!(lookup_keyword("__builtin_abs"), Some(TokenKind::BuiltinAbs));
        assert_eq!(
            lookup_keyword("__builtin_fabsf"),
            Some(TokenKind::BuiltinFabsf)
        );
        assert_eq!(
            lookup_keyword("__builtin_fabs"),
            Some(TokenKind::BuiltinFabs)
        );
        assert_eq!(lookup_keyword("__builtin_inf"), Some(TokenKind::BuiltinInf));
        assert_eq!(
            lookup_keyword("__builtin_inff"),
            Some(TokenKind::BuiltinInff)
        );
        assert_eq!(
            lookup_keyword("__builtin_huge_val"),
            Some(TokenKind::BuiltinHugeVal)
        );
        assert_eq!(
            lookup_keyword("__builtin_huge_valf"),
            Some(TokenKind::BuiltinHugeValf)
        );
        assert_eq!(lookup_keyword("__builtin_nan"), Some(TokenKind::BuiltinNan));
        assert_eq!(
            lookup_keyword("__builtin_nanf"),
            Some(TokenKind::BuiltinNanf)
        );
    }

    // =======================================================================
    // GCC Builtin Keywords — Misc
    // =======================================================================

    #[test]
    fn test_builtin_misc() {
        assert_eq!(
            lookup_keyword("__builtin_trap"),
            Some(TokenKind::BuiltinTrap)
        );
        assert_eq!(
            lookup_keyword("__builtin_alloca"),
            Some(TokenKind::BuiltinAlloca)
        );
        assert_eq!(
            lookup_keyword("__builtin_memcpy"),
            Some(TokenKind::BuiltinMemcpy)
        );
        assert_eq!(
            lookup_keyword("__builtin_memset"),
            Some(TokenKind::BuiltinMemset)
        );
        assert_eq!(
            lookup_keyword("__builtin_strlen"),
            Some(TokenKind::BuiltinStrlen)
        );
        assert_eq!(
            lookup_keyword("__builtin_frame_address"),
            Some(TokenKind::BuiltinFrameAddress)
        );
    }

    // =======================================================================
    // GCC Type Extension Keywords
    // =======================================================================

    #[test]
    fn test_gcc_type_extensions() {
        assert_eq!(lookup_keyword("__int128"), Some(TokenKind::GccInt128));
        assert_eq!(lookup_keyword("__label__"), Some(TokenKind::GccLabel));
        assert_eq!(lookup_keyword("__auto_type"), Some(TokenKind::GccAutoType));
    }

    // =======================================================================
    // GCC Double-Underscore Qualifier Alternate Spellings
    // =======================================================================

    #[test]
    fn test_gcc_inline_alternates() {
        assert_eq!(lookup_keyword("__inline__"), Some(TokenKind::Inline));
        assert_eq!(lookup_keyword("__inline"), Some(TokenKind::Inline));
        // Standard spelling also works
        assert_eq!(lookup_keyword("inline"), Some(TokenKind::Inline));
    }

    #[test]
    fn test_gcc_volatile_alternates() {
        assert_eq!(lookup_keyword("__volatile__"), Some(TokenKind::Volatile));
        assert_eq!(lookup_keyword("__volatile"), Some(TokenKind::Volatile));
        assert_eq!(lookup_keyword("volatile"), Some(TokenKind::Volatile));
    }

    #[test]
    fn test_gcc_const_alternates() {
        assert_eq!(lookup_keyword("__const__"), Some(TokenKind::Const));
        assert_eq!(lookup_keyword("__const"), Some(TokenKind::Const));
        assert_eq!(lookup_keyword("const"), Some(TokenKind::Const));
    }

    #[test]
    fn test_gcc_restrict_alternates() {
        assert_eq!(lookup_keyword("__restrict__"), Some(TokenKind::Restrict));
        assert_eq!(lookup_keyword("__restrict"), Some(TokenKind::Restrict));
        assert_eq!(lookup_keyword("restrict"), Some(TokenKind::Restrict));
    }

    #[test]
    fn test_gcc_signed_alternates() {
        assert_eq!(lookup_keyword("__signed__"), Some(TokenKind::Signed));
        assert_eq!(lookup_keyword("__signed"), Some(TokenKind::Signed));
        assert_eq!(lookup_keyword("signed"), Some(TokenKind::Signed));
    }

    // =======================================================================
    // Non-Keyword Identifiers — Must Return None
    // =======================================================================

    #[test]
    fn test_non_keywords_return_none() {
        let non_keywords = [
            "foo", "bar", "main", "printf", "x", "_custom", "my_var", "argc", "argv", "NULL",
            "stdin", "stdout", "stderr", "size_t", "uint32_t", "true", "false",
        ];
        for name in &non_keywords {
            assert_eq!(
                lookup_keyword(name),
                None,
                "'{}' should not be recognised as a keyword",
                name
            );
        }
    }

    // =======================================================================
    // Case Sensitivity — C Keywords Are Case-Sensitive
    // =======================================================================

    #[test]
    fn test_case_sensitivity() {
        // Uppercase versions of keywords are NOT keywords in C
        assert_eq!(lookup_keyword("INT"), None);
        assert_eq!(lookup_keyword("Int"), None);
        assert_eq!(lookup_keyword("VOID"), None);
        assert_eq!(lookup_keyword("Void"), None);
        assert_eq!(lookup_keyword("RETURN"), None);
        assert_eq!(lookup_keyword("Return"), None);
        assert_eq!(lookup_keyword("IF"), None);
        assert_eq!(lookup_keyword("If"), None);
        assert_eq!(lookup_keyword("WHILE"), None);
        assert_eq!(lookup_keyword("While"), None);
        assert_eq!(lookup_keyword("STRUCT"), None);
        assert_eq!(lookup_keyword("Struct"), None);
    }

    // =======================================================================
    // Edge Cases
    // =======================================================================

    #[test]
    fn test_empty_string_returns_none() {
        assert_eq!(lookup_keyword(""), None);
    }

    #[test]
    fn test_single_character_returns_none() {
        // No C keyword is a single character
        for ch in b'a'..=b'z' {
            let s = String::from(ch as char);
            assert_eq!(
                lookup_keyword(&s),
                None,
                "Single character '{}' should not be a keyword",
                s
            );
        }
    }

    #[test]
    fn test_keyword_prefix_strings_are_not_keywords() {
        // The lexer provides the complete identifier string, so partial
        // matches or superstrings of keywords must return None.
        assert_eq!(lookup_keyword("integer"), None);
        assert_eq!(lookup_keyword("returning"), None);
        assert_eq!(lookup_keyword("doubler"), None);
        assert_eq!(lookup_keyword("shorts"), None);
        assert_eq!(lookup_keyword("voided"), None);
        assert_eq!(lookup_keyword("iffy"), None);
        assert_eq!(lookup_keyword("format"), None);
        assert_eq!(lookup_keyword("while_loop"), None);
        assert_eq!(lookup_keyword("auto_ptr"), None);
        assert_eq!(lookup_keyword("structure"), None);
    }

    #[test]
    fn test_keyword_with_trailing_underscore_is_not_keyword() {
        // "int_" is not the same as "int"
        assert_eq!(lookup_keyword("int_"), None);
        assert_eq!(lookup_keyword("void_"), None);
        assert_eq!(lookup_keyword("return_"), None);
    }

    // =======================================================================
    // Consistency: all 44 C11 keywords present
    // =======================================================================

    #[test]
    fn test_all_44_c11_keywords_present() {
        let c11_keywords = [
            // C89/C99 (34)
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "register",
            "restrict",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            // C11 (10)
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_Bool",
            "_Complex",
            "_Generic",
            "_Imaginary",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        ];
        assert_eq!(
            c11_keywords.len(),
            44,
            "There must be exactly 44 C11 keywords"
        );
        for kw in &c11_keywords {
            assert!(
                lookup_keyword(kw).is_some(),
                "C11 keyword '{}' is missing from the keyword table",
                kw
            );
        }
    }

    // =======================================================================
    // Consistency: keyword table entry count
    // =======================================================================

    #[test]
    fn test_keyword_table_has_expected_entry_count() {
        let map = get_keywords();
        // We expect ~98 entries: 44 C11 + ~54 GCC/alternates
        // Allow for minor adjustments but verify a reasonable minimum.
        assert!(
            map.len() >= 90,
            "Keyword table should have at least 90 entries, got {}",
            map.len()
        );
        assert!(
            map.len() <= 120,
            "Keyword table has unexpectedly many entries: {}",
            map.len()
        );
    }

    // =======================================================================
    // Thread safety: lookup can be called from multiple threads
    // =======================================================================

    #[test]
    fn test_lookup_is_deterministic_across_calls() {
        // Multiple calls must always return the same result.
        for _ in 0..100 {
            assert_eq!(lookup_keyword("int"), Some(TokenKind::Int));
            assert_eq!(
                lookup_keyword("__attribute__"),
                Some(TokenKind::GccAttribute)
            );
            assert_eq!(lookup_keyword("notakeyword"), None);
        }
    }
}
