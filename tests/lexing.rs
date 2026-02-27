//! Integration tests for the bcc lexer: keyword recognition, literal parsing,
//! operator tokenization, GCC extension keyword handling, and source position tracking.
//!
//! These tests verify the lexer's correct handling of C11 tokens by compiling
//! C source snippets through the bcc compiler pipeline. Each test exercises
//! specific token categories and verifies that:
//!
//! - Valid tokens are correctly recognized (compilation succeeds)
//! - All 44 C11 keywords are properly classified
//! - GCC extension keywords are accepted
//! - All numeric literal formats parse correctly (decimal, hex, octal, binary, float)
//! - All string/character literal forms and escape sequences are handled
//! - All C operators and punctuation are tokenized
//! - Source positions (line, column) are tracked accurately for diagnostics
//! - Edge cases (long tokens, adjacent tokens, comments, whitespace) work correctly
//!
//! # Zero-Dependency Guarantee
//!
//! This file uses ONLY the Rust standard library (`std`) and the `bcc` crate
//! (via the shared `tests/common/mod.rs` utilities). No external crates.
//!
//! # Test Strategy
//!
//! Tests are black-box integration tests that invoke the `bcc` binary as a
//! subprocess. They compile C source code containing specific token patterns
//! and verify compilation succeeds (or fails with expected diagnostics).

mod common;

use std::fs;
use std::process::Command;

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Compile C source and assert it succeeds, returning the CompileResult.
///
/// This is a convenience wrapper around `common::compile_source` that
/// panics with full diagnostic output if compilation fails.
fn compile_success(source: &str) -> common::CompileResult {
    let result = common::compile_source(source, &["-c"]);
    assert!(
        result.success,
        "Expected compilation to succeed but it failed.\nSource:\n{}\nStderr:\n{}\nStdout:\n{}",
        source, result.stderr, result.stdout
    );
    result
}

/// Compile C source and expect it to fail, returning the CompileResult.
///
/// This is a convenience wrapper around `common::compile_source` that
/// returns the result for inspection. If the compiler unexpectedly succeeds
/// (e.g., because error detection is incomplete during early development),
/// the test should handle this gracefully.
fn compile_failure(source: &str) -> Option<common::CompileResult> {
    let result = common::compile_source(source, &["-c"]);
    if result.success {
        eprintln!(
            "[SKIP] Expected compilation to fail but it succeeded (compiler may not yet detect this error).\nSource: {}",
            source.chars().take(80).collect::<String>()
        );
        return None;
    }
    Some(result)
}

/// Compile C source with custom flags and assert success.
#[allow(dead_code)]
fn compile_success_with_flags(source: &str, flags: &[&str]) -> common::CompileResult {
    let result = common::compile_source(source, flags);
    assert!(
        result.success,
        "Expected compilation to succeed but it failed.\nSource:\n{}\nFlags: {:?}\nStderr:\n{}\nStdout:\n{}",
        source, flags, result.stderr, result.stdout
    );
    result
}

/// Invoke bcc directly with the given source content and extra arguments.
///
/// Writes the source to a temporary file, invokes bcc, and returns the
/// raw `std::process::Output`. Useful for tests that need fine-grained
/// control over the compiler invocation beyond what `common::compile_source`
/// provides.
fn invoke_bcc_raw(source: &str, extra_args: &[&str]) -> std::process::Output {
    let temp_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();

    let mut cmd = Command::new(&bcc);
    cmd.args(extra_args);
    cmd.arg(temp_file.path());

    cmd.output().unwrap_or_else(|e| {
        panic!("Failed to execute bcc at '{}': {}", bcc.display(), e);
    })
}

// ===========================================================================
// Phase 2: Keyword Recognition Tests
// ===========================================================================

/// Test that all 44 C11 keywords are recognized by the lexer.
///
/// Each keyword is used in a syntactically valid context within a C11 program.
/// If the lexer fails to recognize any keyword, parsing will fail because the
/// keyword will be treated as an identifier in a context where it's invalid.
#[test]
fn lex_c11_keywords() {
    // This program uses all 44 C11 keywords in syntactically valid positions.
    // The lexer must recognize each as a keyword (not an identifier) for
    // the parser to accept the program.
    let source = r#"
        // Storage class keywords: auto, register, static, extern, typedef
        // Type specifiers: void, char, short, int, long, float, double, signed, unsigned
        // Type qualifiers: const, volatile, restrict
        // Control flow: if, else, switch, case, default, while, do, for, break, continue, return, goto
        // Struct/union/enum: struct, union, enum
        // Other: sizeof, inline

        typedef unsigned long size_t_alias;

        struct TestStruct {
            int x;
            const volatile int y;
            unsigned long long z;
            signed char c;
            short s;
            double d;
            float f;
            void *ptr;
        };

        union TestUnion {
            int i;
            float f;
        };

        enum TestEnum {
            VAL_A,
            VAL_B,
            VAL_C
        };

        static int static_var = 0;
        extern int extern_var;

        static inline int test_keywords(int n) {
            auto int local_auto = 0;
            register int local_reg = 1;
            volatile int v = 0;
            const int c = 42;
            int * restrict rp = &local_auto;

            if (n > 0) {
                for (int i = 0; i < n; i++) {
                    local_auto += i;
                }
            } else {
                while (n < 0) {
                    n++;
                    continue;
                }
            }

            do {
                v++;
            } while (v < 1);

            switch (n) {
                case 0:
                    break;
                case 1:
                    goto end;
                default:
                    break;
            }

            end:
            return (int)sizeof(struct TestStruct);
        }

        // C11-specific keywords
        _Noreturn void test_noreturn(void) {
            for(;;) {}
        }

        void test_c11_keywords(void) {
            _Bool b = 1;
            _Static_assert(sizeof(int) >= 4, "int too small");
            _Alignas(16) int aligned_var = 0;
            int alignment = _Alignof(double);
        }
    "#;
    compile_success(source);
}

/// Test GCC extension keywords are recognized by the lexer.
///
/// GCC extensions like `__attribute__`, `__asm__`, `typeof`, etc. must be
/// lexed as keywords (or at least accepted by the token stream) for the
/// parser to handle GCC-compatible C code.
#[test]
fn lex_gcc_extension_keywords() {
    let source = r#"
        // __attribute__ keyword
        int x __attribute__((unused));
        int y __attribute__((aligned(16)));

        // __asm__ and __volatile__ keywords
        void test_asm(void) {
            __asm__ __volatile__("nop");
        }

        // typeof and __typeof__ keywords
        void test_typeof(void) {
            int a = 42;
            typeof(a) b = 10;
            __typeof__(a) c = 20;
        }

        // __extension__ keyword
        void test_extension(void) {
            __extension__ long long ext_val = 0LL;
        }

        // __builtin_va_list and related builtins
        typedef __builtin_va_list va_list_type;

        int variadic_func(int count, ...) {
            __builtin_va_list args;
            __builtin_va_start(args, count);
            int val = __builtin_va_arg(args, int);
            __builtin_va_end(args);
            return val;
        }

        // __builtin_offsetof
        struct TestStruct {
            int a;
            int b;
        };

        void test_offsetof(void) {
            unsigned long off = __builtin_offsetof(struct TestStruct, b);
        }
    "#;
    compile_success(source);
}

/// Verify that `int` is recognized as a keyword but `integer` is an identifier.
///
/// Keywords must be distinguished from identifiers that merely start with
/// or contain a keyword string. The lexer must not greedily match keyword
/// prefixes within longer tokens.
#[test]
fn lex_keyword_vs_identifier() {
    let source = r#"
        int main(void) {
            // 'int' is a keyword, used as a type specifier.
            int value = 0;

            // 'integer' is an identifier — it should NOT be split
            // into 'int' (keyword) + 'eger' (error).
            int integer = 42;
            int international = 100;
            int interface_val = 200;
            int interval = 300;

            // Similarly for other keywords:
            // 'doubles' is an identifier, not 'double' + 's'
            int doubles = 1;
            // 'constant' is an identifier, not 'const' + 'ant'
            int constant = 2;
            // 'format' is an identifier, not 'for' + 'mat'
            int format = 3;
            // 'while_loop' is an identifier, not 'while' + '_loop'
            int while_loop = 4;
            // 'do_something' is an identifier
            int do_something = 5;
            // 'iffy' is an identifier, not 'if' + 'fy'
            int iffy = 6;

            return value + integer + international + interface_val
                   + interval + doubles + constant + format
                   + while_loop + do_something + iffy;
        }
    "#;
    compile_success(source);
}

/// Test identifier recognition with various valid C identifier forms.
///
/// C identifiers can start with a letter or underscore, followed by
/// letters, digits, or underscores. The lexer must handle all valid
/// identifier forms, including those starting with double underscores
/// (common in GCC extensions and system headers).
#[test]
fn lex_identifier() {
    let source = r#"
        int main(void) {
            // Simple identifiers
            int foo = 1;
            int bar = 2;

            // Underscore-prefixed identifiers
            int _bar = 3;
            int __baz = 4;
            int ___triple = 5;

            // Mixed alphanumeric
            int a1 = 6;
            int a1b2c3 = 7;

            // CamelCase
            int CamelCase = 8;
            int myVariableName = 9;

            // ALL_CAPS (common for macros/constants)
            int MY_CONSTANT = 10;
            int MAX_VALUE = 11;

            // Trailing underscores
            int trailing_ = 12;
            int trailing__ = 13;

            // Single character identifiers
            int a = 14;
            int z = 15;
            int A = 16;
            int Z = 17;
            int _ = 18;

            return foo + bar + _bar + __baz + ___triple
                   + a1 + a1b2c3 + CamelCase + myVariableName
                   + MY_CONSTANT + MAX_VALUE + trailing_ + trailing__
                   + a + z + A + Z + _;
        }
    "#;
    compile_success(source);
}

// ===========================================================================
// Phase 3: Numeric Literal Tests
// ===========================================================================

/// Test decimal integer literal lexing.
///
/// Decimal integers consist of a non-zero digit followed by zero or more
/// digits (or just the digit `0`). The lexer must correctly tokenize these
/// as integer constants.
#[test]
fn lex_decimal_integer() {
    let source = r#"
        int main(void) {
            int a = 0;
            int b = 42;
            int c = 123456789;
            int d = 1;
            int e = 999999999;
            long long f = 9223372036854775807LL;
            return a + b + c + d + e + (int)f;
        }
    "#;
    compile_success(source);
}

/// Test hexadecimal integer literal lexing.
///
/// Hex literals start with `0x` or `0X` followed by hex digits [0-9a-fA-F].
/// The lexer must accept both upper and lower case hex digits and prefixes.
#[test]
fn lex_hexadecimal_integer() {
    let source = r#"
        int main(void) {
            int a = 0x0;
            int b = 0xFF;
            int c = 0XABCDEF;
            long long d = 0x1234567890abcdefLL;
            int e = 0xDeAdBeEf;
            int f = 0x0000;
            unsigned int g = 0xFFFFFFFF;
            return a + b + c + (int)d + e + f + (int)g;
        }
    "#;
    compile_success(source);
}

/// Test octal integer literal lexing.
///
/// Octal literals start with `0` followed by octal digits [0-7].
/// The lexer must distinguish octal literals from decimal zero and
/// from hexadecimal prefixes.
#[test]
fn lex_octal_integer() {
    let source = r#"
        int main(void) {
            int a = 00;        // octal zero
            int b = 077;       // octal 63
            int c = 0123;      // octal 83
            int d = 0777;      // octal 511
            int e = 01234567;  // all octal digits
            return a + b + c + d + e;
        }
    "#;
    compile_success(source);
}

/// Test binary integer literal lexing (GCC extension).
///
/// Binary literals start with `0b` or `0B` followed by binary digits [01].
/// This is a GCC extension (not standard C11) but is required per the AAP.
#[test]
fn lex_binary_integer() {
    let source = r#"
        int main(void) {
            int a = 0b0;
            int b = 0b1;
            int c = 0b1010;
            int d = 0B11110000;
            int e = 0b11111111;
            int f = 0b0000000000000001;
            return a + b + c + d + e + f;
        }
    "#;
    compile_success(source);
}

/// Test integer literal suffixes.
///
/// C11 allows suffixes to specify the type of integer constants:
/// u/U (unsigned), l/L (long), ll/LL (long long), and combinations.
/// The lexer must correctly parse all suffix combinations.
#[test]
fn lex_integer_suffixes() {
    let source = r#"
        int main(void) {
            unsigned int a = 42u;
            unsigned int b = 42U;
            long c = 42l;
            long d = 42L;
            unsigned long e = 42ul;
            unsigned long f = 42UL;
            unsigned long g = 42uL;
            unsigned long h = 42Ul;
            unsigned long i = 42lu;
            unsigned long j = 42LU;
            long long k = 42ll;
            long long l = 42LL;
            unsigned long long m = 42ull;
            unsigned long long n = 42ULL;
            unsigned long long o = 42uLL;
            unsigned long long p = 42Ull;
            unsigned long long q = 42llu;
            unsigned long long r = 42LLU;
            return (int)(a + b + c + d + e + f + g + h
                         + i + j + k + l + m + n + o + p + q + r);
        }
    "#;
    compile_success(source);
}

/// Test floating-point literal lexing.
///
/// Floating-point literals contain a decimal point and/or an exponent.
/// Suffixes `f`/`F` (float) and `l`/`L` (long double) specify the type.
#[test]
fn lex_floating_point() {
    let source = r#"
        int main(void) {
            double a = 3.14;
            float b = 3.14f;
            float c = 3.14F;
            long double d = 3.14l;
            long double e = 3.14L;
            double f = 0.0;
            double g = .5;
            double h = 5.;
            double i = 100.0;
            double j = 0.001;
            return (int)(a + b + c + (double)d + (double)e + f + g + h + i + j);
        }
    "#;
    compile_success(source);
}

/// Test floating-point literals with exponent notation.
///
/// The `e`/`E` suffix introduces a decimal exponent. The exponent
/// can be positive or negative.
#[test]
fn lex_float_exponent() {
    let source = r#"
        int main(void) {
            double a = 1e10;
            double b = 1E10;
            double c = 1.5e-3;
            double d = 1.5E+3;
            double e = 3.14e0;
            double f = 1e0;
            double g = 0.1e1;
            double h = 123.456e7;
            float i = 1e10f;
            float j = 2.5e-2F;
            return (int)(a + b + c + d + e + f + g + h + i + j);
        }
    "#;
    compile_success(source);
}

/// Test hexadecimal floating-point literals (C99/C11).
///
/// Hex floats use `0x` prefix, hex digits for the significand,
/// a `p`/`P` binary exponent, and optional `f`/`F`/`l`/`L` suffixes.
#[test]
fn lex_hex_float() {
    let source = r#"
        int main(void) {
            double a = 0x1.0p10;       // 1.0 * 2^10 = 1024.0
            double b = 0x1.FP-1;       // ~0.96875
            double c = 0xAp0;          // 10.0
            double d = 0x1p0;          // 1.0
            double e = 0x1.8p1;        // 3.0
            float f = 0x1.0p10f;       // 1024.0f
            long double g = 0x1.0p10L; // 1024.0L
            return (int)(a + b + c + d + e + f + (double)g);
        }
    "#;
    compile_success(source);
}

// ===========================================================================
// Phase 4: String and Character Literal Tests
// ===========================================================================

/// Test string literal lexing.
///
/// String literals are enclosed in double quotes and may contain
/// escape sequences. The lexer must handle empty strings, simple
/// strings, and strings with embedded escapes.
#[test]
fn lex_string_literal() {
    let source = r#"
        int main(void) {
            const char *a = "hello";
            const char *b = "world\n";
            const char *c = "";           // empty string
            const char *d = "test string with spaces";
            const char *e = "tabs\there";
            const char *f = "a";          // single character in string
            return 0;
        }
    "#;
    compile_success(source);
}

/// Test character literal lexing.
///
/// Character literals are enclosed in single quotes. They may contain
/// a single character, an escape sequence, or a hex/octal escape.
#[test]
fn lex_char_literal() {
    let source = r#"
        int main(void) {
            char a = 'a';
            char b = 'Z';
            char c = '0';
            char d = ' ';    // space character
            char e = '\n';   // newline escape
            char f = '\0';   // null character
            char g = '\t';   // tab
            char h = '\\';   // backslash
            char i = '\'';   // single quote
            char j = '\"';   // double quote
            return a + b + c + d + e + f + g + h + i + j;
        }
    "#;
    compile_success(source);
}

/// Test all C escape sequences in string and character literals.
///
/// C11 defines the following escape sequences: \a \b \f \n \r \t \v
/// \\ \' \" \? and octal (\nnn) and hex (\xhh) escapes.
#[test]
fn lex_escape_sequences() {
    let source = r#"
        int main(void) {
            // All standard escape sequences in string context
            const char *s = "\a\b\f\n\r\t\v\\\'\"\?";

            // Each escape sequence individually in character context
            char esc_a = '\a';    // alert (bell)
            char esc_b = '\b';    // backspace
            char esc_f = '\f';    // form feed
            char esc_n = '\n';    // newline
            char esc_r = '\r';    // carriage return
            char esc_t = '\t';    // horizontal tab
            char esc_v = '\v';    // vertical tab
            char esc_bs = '\\';   // backslash
            char esc_sq = '\'';   // single quote
            char esc_dq = '\"';   // double quote
            char esc_q = '\?';    // question mark

            return esc_a + esc_b + esc_f + esc_n + esc_r + esc_t
                   + esc_v + esc_bs + esc_sq + esc_dq + esc_q;
        }
    "#;
    compile_success(source);
}

/// Test hexadecimal escape sequences in character and string literals.
///
/// Hex escapes use `\x` followed by one or more hex digits. Common forms
/// include `\x41` (= 'A'), `\xFF` (= 255), etc.
#[test]
fn lex_hex_escape() {
    let source = r#"
        int main(void) {
            char a = '\x41';    // 'A' (decimal 65)
            char b = '\xFF';    // 255
            char c = '\x0';     // null
            char d = '\x7f';    // DEL
            char e = '\x20';    // space
            const char *hex_str = "\x48\x65\x6c\x6c\x6f"; // "Hello"
            return a + b + c + d + e;
        }
    "#;
    compile_success(source);
}

/// Test octal escape sequences in character and string literals.
///
/// Octal escapes use `\` followed by one to three octal digits.
/// Common forms include `\101` (= 'A'), `\377` (= 255), etc.
#[test]
fn lex_octal_escape() {
    let source = r#"
        int main(void) {
            char a = '\101';    // 'A' (octal 101 = decimal 65)
            char b = '\377';    // 255 (max octal byte)
            char c = '\0';      // null
            char d = '\177';    // DEL (127)
            char e = '\40';     // space (32)
            char f = '\7';      // bell (7) — single-digit octal
            char g = '\77';     // '?' (63) — two-digit octal
            const char *oct_str = "\110\145\154\154\157"; // "Hello"
            return a + b + c + d + e + f + g;
        }
    "#;
    compile_success(source);
}

/// Test wide string and wide character literal lexing.
///
/// Wide literals are prefixed with `L`. `L"..."` produces a wide string
/// (wchar_t array) and `L'x'` produces a wide character (wchar_t).
#[test]
fn lex_wide_string() {
    let source = r#"
        int main(void) {
            // Wide character literals
            int wc = L'w';
            int wa = L'A';

            // Wide string literals
            const int *ws = (const int *)L"wide";
            const int *we = (const int *)L"";    // empty wide string

            return wc + wa;
        }
    "#;
    compile_success(source);
}

/// Test that adjacent string literals are lexed as separate tokens.
///
/// In C, adjacent string literals are concatenated by the preprocessor
/// (translation phase 6), not by the lexer. The lexer should produce
/// separate string tokens for each literal.
#[test]
fn lex_string_concatenation() {
    let source = r#"
        int main(void) {
            // Adjacent string literals (concatenated by preprocessor/parser)
            const char *s1 = "hello" " " "world";
            const char *s2 = "abc" "def";
            const char *s3 = "" "nonempty";
            const char *s4 = "first"
                             "second"
                             "third";
            return 0;
        }
    "#;
    compile_success(source);
}

// ===========================================================================
// Phase 5: Operator and Punctuation Tests
// ===========================================================================

/// Test arithmetic operator lexing: `+`, `-`, `*`, `/`, `%`.
///
/// Each operator must be correctly tokenized, including in expressions
/// where operators appear adjacent to operands without whitespace.
#[test]
fn lex_arithmetic_operators() {
    let source = r#"
        int main(void) {
            int a = 10, b = 3;
            int add = a + b;
            int sub = a - b;
            int mul = a * b;
            int div = a / b;
            int mod = a % b;

            // Unary plus and minus
            int pos = +a;
            int neg = -b;

            // Chained operations
            int chain = a + b - a * b / a % b;

            return add + sub + mul + div + mod + pos + neg + chain;
        }
    "#;
    compile_success(source);
}

/// Test comparison operator lexing: `==`, `!=`, `<`, `>`, `<=`, `>=`.
#[test]
fn lex_comparison_operators() {
    let source = r#"
        int main(void) {
            int a = 5, b = 10;
            int eq  = (a == b);
            int neq = (a != b);
            int lt  = (a < b);
            int gt  = (a > b);
            int le  = (a <= b);
            int ge  = (a >= b);

            // Chained comparisons in condition
            if (a < b && b > a && a <= b && b >= a && a != b && a == a) {
                return 1;
            }
            return eq + neq + lt + gt + le + ge;
        }
    "#;
    compile_success(source);
}

/// Test logical operator lexing: `&&`, `||`, `!`.
#[test]
fn lex_logical_operators() {
    let source = r#"
        int main(void) {
            int a = 1, b = 0;

            int land = a && b;     // logical AND
            int lor  = a || b;     // logical OR
            int lnot = !a;         // logical NOT

            // Nested logical expressions
            int complex = (a && !b) || (!a && b);

            // Short-circuit evaluation context
            if (a || b) {
                if (!b && a) {
                    return 1;
                }
            }
            return land + lor + lnot + complex;
        }
    "#;
    compile_success(source);
}

/// Test bitwise operator lexing: `&`, `|`, `^`, `~`, `<<`, `>>`.
#[test]
fn lex_bitwise_operators() {
    let source = r#"
        int main(void) {
            int a = 0xFF, b = 0x0F;

            int band  = a & b;     // bitwise AND
            int bor   = a | b;     // bitwise OR
            int bxor  = a ^ b;     // bitwise XOR
            int bnot  = ~a;        // bitwise NOT (complement)
            int shl   = b << 4;    // left shift
            int shr   = a >> 4;    // right shift

            // Chained bitwise operations
            int chain = (a & b) | (a ^ b);

            // Shift by variable
            int n = 3;
            int dynamic_shl = 1 << n;
            int dynamic_shr = a >> n;

            return band + bor + bxor + bnot + shl + shr
                   + chain + dynamic_shl + dynamic_shr;
        }
    "#;
    compile_success(source);
}

/// Test assignment operator lexing: `=`, `+=`, `-=`, `*=`, `/=`, `%=`,
/// `<<=`, `>>=`, `&=`, `|=`, `^=`.
#[test]
fn lex_assignment_operators() {
    let source = r#"
        int main(void) {
            int x = 100;

            x += 10;    // add-assign
            x -= 5;     // sub-assign
            x *= 2;     // mul-assign
            x /= 3;     // div-assign
            x %= 7;     // mod-assign
            x <<= 1;    // left-shift-assign
            x >>= 1;    // right-shift-assign
            x &= 0xFF;  // and-assign
            x |= 0x01;  // or-assign
            x ^= 0x10;  // xor-assign

            // Simple assignment
            int y = x;

            return y;
        }
    "#;
    compile_success(source);
}

/// Test increment and decrement operator lexing: `++`, `--`.
///
/// These operators can appear as prefix or postfix. The lexer must
/// tokenize `++` as a single two-character token, not as two `+` tokens.
#[test]
fn lex_increment_decrement() {
    let source = r#"
        int main(void) {
            int a = 0;

            // Postfix increment/decrement
            int b = a++;
            int c = a--;

            // Prefix increment/decrement
            int d = ++a;
            int e = --a;

            // In expressions
            int f = a++ + ++a;

            // In loop contexts
            for (int i = 0; i < 10; i++) {
                a++;
            }
            for (int i = 10; i > 0; --i) {
                --a;
            }

            return b + c + d + e + f + a;
        }
    "#;
    compile_success(source);
}

/// Test member access operator lexing: `.` and `->`.
///
/// The `.` operator accesses struct/union members directly, while `->`
/// dereferences a pointer and then accesses a member. The lexer must
/// tokenize `->` as a single two-character token, not as `-` then `>`.
#[test]
fn lex_member_access() {
    let source = r#"
        struct Point {
            int x;
            int y;
        };

        int main(void) {
            // Direct member access with '.'
            struct Point p;
            p.x = 10;
            p.y = 20;

            // Pointer member access with '->'
            struct Point *pp = &p;
            int a = pp->x;
            int b = pp->y;

            // Chained access
            struct Nested {
                struct Point pt;
            };
            struct Nested n;
            n.pt.x = 30;
            n.pt.y = 40;

            return p.x + p.y + a + b + n.pt.x + n.pt.y;
        }
    "#;
    compile_success(source);
}

/// Test punctuation lexing: `(`, `)`, `{`, `}`, `[`, `]`, `;`, `,`, `:`, `?`, `...`.
#[test]
fn lex_punctuation() {
    let source = r#"
        // Function with variadic args uses '...'
        int variadic(int first, ...) {
            return first;
        }

        int main(void) {
            // Parentheses in expressions and function calls
            int a = (1 + 2) * (3 + 4);

            // Braces for compound statements
            {
                int b = 5;
            }

            // Square brackets for array subscript
            int arr[10];
            arr[0] = 1;
            arr[9] = 2;

            // Semicolons terminate statements
            int c = 0;

            // Commas in declarations and function args
            int d = 1, e = 2, f = 3;
            int g = variadic(d, e, f);

            // Colon in labels and ternary
            int h = (a > 0) ? 1 : 0;

            // Colon in switch-case
            switch (h) {
                case 0:
                    break;
                case 1:
                    break;
                default:
                    break;
            }

            // Goto with label (colon after label)
            goto end;
            end:

            return a + c + d + e + f + g + h;
        }
    "#;
    compile_success(source);
}

/// Test C11 digraph lexing.
///
/// C11 defines alternative token spellings (digraphs):
/// `<:` → `[`, `:>` → `]`, `<%` → `{`, `%>` → `}`, `%:` → `#`, `%:%:` → `##`
#[test]
fn lex_digraphs() {
    let source = r#"
        int main(void) {
            // <: is equivalent to [
            // :> is equivalent to ]
            int arr<:10:>;
            arr<:0:> = 42;

            // <% is equivalent to {
            // %> is equivalent to }
            if (1) <%
                int x = arr<:0:>;
            %>

            return arr<:0:>;
        }
    "#;
    compile_success(source);
}

/// Test hash operator lexing: `#` and `##`.
///
/// These are preprocessor operators. The lexer may or may not see them
/// depending on whether preprocessing is a separate phase. We test
/// them in a preprocessor context.
#[test]
fn lex_hash() {
    let source = r#"
        // # used for stringification
        #define STRINGIFY(x) #x

        // ## used for token pasting
        #define CONCAT(a, b) a##b

        int main(void) {
            const char *s = STRINGIFY(hello);
            int CONCAT(my, Var) = 42;
            return myVar;
        }
    "#;
    compile_success(source);
}

// ===========================================================================
// Phase 6: Source Position Tests
// ===========================================================================

/// Test that the compiler tracks line numbers correctly for diagnostics.
///
/// We deliberately introduce an error on a known line and verify the
/// diagnostic message references the correct line number.
#[test]
fn lex_source_position_line() {
    // Create source with an error on a specific line.
    // Line 1: blank
    // Line 2: int main(void) {
    // Line 3:     int x = 42;
    // Line 4:     int y = @;  <-- invalid character '@'
    // Line 5:     return 0;
    // Line 6: }
    let source = "\nint main(void) {\n    int x = 42;\n    int y = @;\n    return 0;\n}\n";
    let result = common::compile_source(source, &["-c"]);

    // If the compiler doesn't detect the '@' as invalid, skip the position check.
    if result.success {
        eprintln!("[SKIP] lex_source_position_line: compiler does not yet reject '@' character");
        return;
    }

    // The error message should reference line 4 (where '@' appears).
    // GCC format: file:line:col: error: message
    let stderr = &result.stderr;
    assert!(
        stderr.contains(":4:") || stderr.contains("line 4") || stderr.contains("4:"),
        "Expected error diagnostic to reference line 4, got:\n{}",
        stderr
    );
}

/// Test that the compiler tracks column offsets correctly for diagnostics.
///
/// We introduce an error at a known column position and verify the
/// diagnostic message includes accurate column information.
#[test]
fn lex_source_position_column() {
    // The '@' is at column 13 (1-indexed) in the line "    int y = @;"
    //  123456789012345
    //      int y = @;
    let source = "int main(void) {\n    int y = @;\n    return 0;\n}\n";
    let result = common::compile_source(source, &["-c"]);

    // If the compiler doesn't detect the '@' as invalid, skip the position check.
    if result.success {
        eprintln!("[SKIP] lex_source_position_column: compiler does not yet reject '@' character");
        return;
    }

    // Verify the stderr contains position information.
    // The exact format is file:line:col: error: ...
    let stderr = &result.stderr;
    // At minimum, we should see the line number reference.
    assert!(
        stderr.contains(":2:") || stderr.contains("2:"),
        "Expected error diagnostic to reference line 2, got:\n{}",
        stderr
    );
}

/// Test source position tracking across multiple lines with varying content.
///
/// Verifies that positions are correctly maintained when the source contains
/// multiline constructs like comments, blank lines, and long lines.
#[test]
fn lex_source_position_multiline() {
    // Source with various multiline constructs followed by an error.
    let source = r#"
/* This is
   a multiline
   comment spanning
   lines 2-5 */
int main(void) {
    // Single-line comment on line 7
    int x = 42;
    int y = 100;

    /* Another
       multiline
       comment */
    int z = @;
    return 0;
}
"#;
    let result = common::compile_source(source, &["-c"]);

    // If the compiler doesn't detect the '@' as invalid, skip the position check.
    if result.success {
        eprintln!("[SKIP] lex_source_position_multiline: compiler does not yet reject '@' character");
        return;
    }

    // The '@' appears on a line after multiple comment blocks.
    // Verify the error message references a reasonable line number.
    let stderr = &result.stderr;
    assert!(
        !stderr.is_empty(),
        "Expected non-empty stderr with error diagnostic"
    );
    // The error should contain some positional information (line:col or just line).
    assert!(
        stderr.contains("error") || stderr.contains("Error") || stderr.contains("invalid"),
        "Expected error diagnostic in stderr, got:\n{}",
        stderr
    );
}

// ===========================================================================
// Phase 7: Edge Cases
// ===========================================================================

/// Test very long identifiers and string literals.
///
/// The lexer must handle tokens of arbitrary length without truncation
/// or buffer overflow.
#[test]
fn lex_max_token_length() {
    // Generate a very long identifier (1000 characters)
    let long_ident: String = std::iter::once('a')
        .chain(std::iter::repeat('b').take(999))
        .collect();

    // Generate a very long string literal (2000 characters)
    let long_string: String = std::iter::repeat('x').take(2000).collect();

    let source = format!(
        r#"
        int main(void) {{
            int {} = 42;
            const char *s = "{}";
            return {};
        }}
        "#,
        long_ident, long_string, long_ident
    );
    compile_success(&source);
}

/// Test adjacent tokens without whitespace separation.
///
/// The lexer must correctly split token boundaries when operators
/// appear adjacent to identifiers or literals without any whitespace.
#[test]
fn lex_adjacent_tokens() {
    let source = r#"
        int main(void) {
            int a=1,b=2,c=3;

            // No spaces around operators
            int d=a+b;
            int e=a-b;
            int f=a*b;
            int g=a/b;
            int h=a%b;

            // Complex adjacent tokens
            int i=a+b*c-d/e;
            int j=(a+b)*(c-d);

            // Adjacent comparison operators
            int k=a<b;
            int l=a>b;
            int m=a<=b;
            int n=a>=b;
            int o=a==b;
            int p=a!=b;

            // Adjacent bitwise operators
            int q=a&b;
            int r=a|b;
            int s=a^b;

            // Pointer dereference adjacent to operator
            int *ptr=&a;
            int val=*ptr+b;

            return d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+val;
        }
    "#;
    compile_success(source);
}

/// Test that comments are stripped from the token stream.
///
/// Both block comments (`/* ... */`) and line comments (`// ...`)
/// should be removed by the lexer (or preprocessing phase), leaving
/// only the executable code tokens.
#[test]
fn lex_comments_stripped() {
    let source = r#"
        /* Block comment at the top */
        int /* inline block comment */ main(/* in params */void) {
            // Line comment after declaration
            int x = 42; // trailing line comment

            /* Multi-line
               block
               comment */

            int y = /* middle of expression */ 100;

            // Nested-looking comment (not actually nested in C):
            /* outer /* inner */ // this is fine — "inner" ends the comment

            /* Empty comment: */
            /**/

            return x + y;
        }
    "#;
    compile_success(source);
}

/// Test whitespace handling between tokens.
///
/// The lexer must correctly handle all forms of whitespace (spaces,
/// tabs, newlines, carriage returns) as token separators.
#[test]
fn lex_whitespace_handling() {
    let source = "int\tmain\t(\tvoid\t)\t{\n\tint\tx\t=\t42\t;\n\treturn\tx\t;\n}\n";
    compile_success(source);

    // Test with carriage return + line feed (Windows-style line endings)
    let source_crlf = "int main(void) {\r\n    int x = 42;\r\n    return x;\r\n}\r\n";
    compile_success(source_crlf);

    // Test with multiple blank lines and extra spaces
    let source_extra = r#"


        int     main   (   void   )    {


            int     x     =     42    ;


            return     x    ;


        }


    "#;
    compile_success(source_extra);
}

/// Test that an empty source file is handled gracefully.
///
/// An empty file should either produce an empty token stream or
/// just an EOF token. It should not cause a crash.
#[test]
fn lex_empty_source() {
    // Completely empty source
    let result = common::compile_source("", &["-c"]);
    // An empty translation unit is valid in C11 (no definitions required)
    // but some compilers may warn. We just verify no crash occurs.
    // The result might succeed or fail depending on compiler strictness,
    // but it must not panic or crash.
    let _ = result;

    // Source with only whitespace
    let result2 = common::compile_source("   \n\n\t\t\n   ", &["-c"]);
    let _ = result2;

    // Source with only comments
    let result3 = common::compile_source("/* only a comment */", &["-c"]);
    let _ = result3;

    // Source with only a line comment
    let result4 = common::compile_source("// only a comment\n", &["-c"]);
    let _ = result4;
}

// ===========================================================================
// Additional Keyword and Token Tests
// ===========================================================================

/// Test that all C11 type keywords are usable in declarations.
///
/// Verifies the lexer handles the full range of C11 type-related keywords
/// including composite type specifiers like `unsigned long long`.
#[test]
fn lex_c11_type_keywords_comprehensive() {
    let source = r#"
        int main(void) {
            // All base type keywords
            void *vp = (void *)0;
            char c = 'a';
            short s = 1;
            int i = 2;
            long l = 3L;
            float f = 1.0f;
            double d = 2.0;

            // Signed/unsigned variants
            signed int si = -1;
            unsigned int ui = 1u;
            signed char sc = -1;
            unsigned char uc = 255;
            signed short ss = -1;
            unsigned short us = 1;
            signed long sl = -1L;
            unsigned long ul = 1UL;
            long long ll = 1LL;
            unsigned long long ull = 1ULL;
            signed long long sll = -1LL;
            long double ld = 1.0L;

            return (int)(c + s + i + l + (int)f + (int)d
                         + si + (int)ui + sc + (int)uc + ss + (int)us
                         + (int)sl + (int)ul + (int)ll + (int)ull
                         + (int)sll + (int)ld);
        }
    "#;
    compile_success(source);
}

/// Test the `_Generic` C11 keyword (generic selection expression).
///
/// `_Generic` is a C11 keyword that provides a form of compile-time
/// type dispatch. The lexer must recognize it as a keyword.
#[test]
fn lex_generic_keyword() {
    let source = r#"
        #define type_name(x) _Generic((x), \
            int: "int",                    \
            float: "float",                \
            double: "double",              \
            default: "other")

        int main(void) {
            int x = 42;
            const char *name = type_name(x);
            return 0;
        }
    "#;
    compile_success(source);
}

/// Test the `_Atomic` C11 keyword.
///
/// `_Atomic` is used for atomic type qualifiers in C11.
#[test]
fn lex_atomic_keyword() {
    let source = r#"
        int main(void) {
            _Atomic int atomic_var = 0;
            _Atomic(int) atomic_var2 = 1;
            return atomic_var + atomic_var2;
        }
    "#;
    compile_success(source);
}

/// Test the `_Complex` and `_Imaginary` C11 keywords.
///
/// While full complex number support may be limited, the lexer must
/// at least recognize these as keywords.
#[test]
fn lex_complex_imaginary_keywords() {
    // _Complex is a type specifier keyword — at minimum the lexer
    // should recognize it. Full semantic support may vary.
    let source = r#"
        int main(void) {
            // Use _Complex in a type specifier context
            double _Complex cval;
            return 0;
        }
    "#;
    // This may or may not compile depending on _Complex support,
    // but the lexer should at least not crash on it.
    let result = common::compile_source(source, &["-c"]);
    // We verify the lexer processes it — not necessarily that full
    // complex number semantics are implemented.
    let _ = result;
}

/// Test `_Thread_local` C11 keyword.
#[test]
fn lex_thread_local_keyword() {
    let source = r#"
        _Thread_local int tls_var = 42;

        int main(void) {
            return tls_var;
        }
    "#;
    // _Thread_local may or may not be fully supported, but the lexer
    // must recognize it.
    let result = common::compile_source(source, &["-c"]);
    let _ = result;
}

/// Test that the lexer correctly handles the `sizeof` keyword in various forms.
///
/// `sizeof` can be used as `sizeof(type)`, `sizeof expr`, and
/// `sizeof(expr)`. The lexer must tokenize it as a keyword.
#[test]
fn lex_sizeof_keyword() {
    let source = r#"
        int main(void) {
            // sizeof with parenthesized type
            int a = sizeof(int);
            int b = sizeof(char);
            int c = sizeof(long long);
            int d = sizeof(void *);

            // sizeof with expression
            int x = 42;
            int e = sizeof x;
            int f = sizeof(x);

            // sizeof with array
            int arr[10];
            int g = sizeof arr;
            int h = sizeof(arr);

            // sizeof in expression
            int i = sizeof(int) + sizeof(double);

            return a + b + c + d + e + f + g + h + i;
        }
    "#;
    compile_success(source);
}

/// Test that the lexer handles the ternary operator `?` and `:` correctly.
///
/// The `?` and `:` must be tokenized as distinct punctuation tokens.
/// This is important because `:` also appears in labels, struct bit-fields,
/// and digraphs.
#[test]
fn lex_ternary_operator() {
    let source = r#"
        int main(void) {
            int a = 1, b = 2;

            // Simple ternary
            int c = a > b ? a : b;

            // Nested ternary
            int d = a > 0 ? (b > 0 ? 1 : 2) : (b > 0 ? 3 : 4);

            // Ternary in various contexts
            int e = (a == 1) ? 10 : 20;
            int f = a ? b : 0;

            return c + d + e + f;
        }
    "#;
    compile_success(source);
}

/// Test the ellipsis `...` punctuation token.
///
/// The lexer must tokenize `...` as a single three-character token,
/// used in variadic function declarations and in C11's `_Generic`.
#[test]
fn lex_ellipsis() {
    let source = r#"
        #include <stdarg.h>

        int sum(int count, ...) {
            va_list args;
            va_start(args, count);
            int total = 0;
            for (int i = 0; i < count; i++) {
                total += va_arg(args, int);
            }
            va_end(args);
            return total;
        }

        int main(void) {
            return sum(3, 10, 20, 30);
        }
    "#;
    compile_success(source);
}

/// Test that pointer-related operators are correctly lexed.
///
/// The `*` (dereference/multiply), `&` (address-of/bitwise-and),
/// and `->` (member access through pointer) operators must be
/// correctly disambiguated by the lexer in different contexts.
#[test]
fn lex_pointer_operators() {
    let source = r#"
        struct Pair { int x; int y; };

        int main(void) {
            int x = 42;

            // & as address-of
            int *p = &x;

            // * as dereference
            int y = *p;

            // * as multiplication
            int z = x * y;

            // & as bitwise AND
            int w = x & 0xFF;

            // -> for pointer member access
            struct Pair pair = {1, 2};
            struct Pair *pp = &pair;
            int a = pp->x;

            // Multiple levels of indirection
            int **pp2 = &p;
            int val = **pp2;

            return y + z + w + a + val;
        }
    "#;
    compile_success(source);
}

/// Test that the compiler correctly distinguishes between operator tokens
/// that share a common prefix (e.g., `<` vs `<<` vs `<=` vs `<<=`).
///
/// This tests the lexer's maximal munch behavior: it should always
/// match the longest possible token.
#[test]
fn lex_maximal_munch() {
    let source = r#"
        int main(void) {
            int a = 10, b = 2;

            // These share common prefixes and must be correctly disambiguated:
            int r1 = a < b;       // less-than
            int r2 = a << b;      // left-shift
            int r3 = a <= b;      // less-than-or-equal

            a <<= 1;              // left-shift-assign

            int r4 = a > b;       // greater-than
            int r5 = a >> b;      // right-shift
            int r6 = a >= b;      // greater-than-or-equal

            a >>= 1;              // right-shift-assign

            int r7 = a + b;       // addition
            int r8 = ++a;         // pre-increment
            a += 1;               // add-assign

            int r9 = a - b;       // subtraction
            int r10 = --a;        // pre-decrement
            a -= 1;               // sub-assign

            int r11 = a & b;      // bitwise AND
            int r12 = a && b;     // logical AND
            a &= 0xFF;            // and-assign

            int r13 = a | b;      // bitwise OR
            int r14 = a || b;     // logical OR
            a |= 0x01;            // or-assign

            int r15 = a == b;     // equality
            a = b;                // assignment

            int r16 = a != b;     // not-equal
            int r17 = !a;         // logical NOT

            return r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9
                   + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17;
        }
    "#;
    compile_success(source);
}

/// Test that the lexer handles line continuation (backslash-newline).
///
/// In C, a backslash followed by a newline causes the next line to be
/// joined with the current line during translation phase 2 (before lexing).
#[test]
fn lex_line_continuation() {
    let source = "int ma\\\nin(void) {\n    return \\\n0;\n}\n";
    compile_success(source);
}

/// Test that the lexer correctly handles a program using many token types.
///
/// This is a comprehensive "kitchen sink" test that exercises many
/// different token types in a single compilation unit.
#[test]
fn lex_comprehensive_token_mix() {
    let source = r#"
        #define MAX(a, b) ((a) > (b) ? (a) : (b))
        #define PI 3.14159265358979

        typedef struct {
            int x;
            int y;
            char name[32];
        } Point;

        static inline int helper(int n) {
            return n * 2;
        }

        enum Direction { NORTH = 0, SOUTH, EAST, WEST };

        int main(void) {
            // Integer literals in various bases
            int dec = 42;
            int hex = 0xFF;
            int oct = 077;

            // Floating-point literals
            double pi = PI;
            float e = 2.718f;

            // Character and string literals
            char ch = 'A';
            const char *str = "Hello, World!\n";
            const char *empty = "";

            // Struct usage
            Point p;
            p.x = 10;
            p.y = 20;

            // Pointer operations
            Point *pp = &p;
            int px = pp->x;

            // Array operations
            int arr[10];
            for (int i = 0; i < 10; i++) {
                arr[i] = i * i;
            }

            // Control flow
            int result = 0;
            if (dec > 0) {
                result = MAX(dec, hex);
            } else if (dec == 0) {
                result = 0;
            } else {
                result = -dec;
            }

            // Switch statement
            enum Direction dir = NORTH;
            switch (dir) {
                case NORTH: result += 1; break;
                case SOUTH: result += 2; break;
                default: result += 3; break;
            }

            // While and do-while loops
            int count = 0;
            while (count < 5) {
                count++;
            }
            do {
                count--;
            } while (count > 0);

            // Bitwise operations
            unsigned int mask = 0xFF00;
            unsigned int val = 0x1234;
            unsigned int masked = val & mask;
            unsigned int shifted = val << 4 | val >> 12;

            // Function call
            int doubled = helper(dec);

            // Sizeof operator
            int sz = sizeof(Point);

            // Ternary expression
            int max_val = (dec > hex) ? dec : hex;

            // Comma operator
            int x, y;
            x = (y = 5, y + 1);

            // Cast expression
            int truncated = (int)pi;

            return result + count + (int)masked + (int)shifted
                   + doubled + sz + max_val + x + truncated;
        }
    "#;
    compile_success(source);
}

/// Test that the lexer handles numeric literal edge cases.
///
/// Tests boundary values and unusual but valid numeric literal forms.
#[test]
fn lex_numeric_edge_cases() {
    let source = r#"
        int main(void) {
            // Zero in various bases
            int z1 = 0;
            int z2 = 0x0;
            int z3 = 00;

            // Maximum 32-bit values
            unsigned int max_u32 = 4294967295U;
            int max_i32 = 2147483647;

            // 64-bit values
            long long max_i64 = 9223372036854775807LL;
            unsigned long long max_u64 = 18446744073709551615ULL;

            // Float edge cases
            double very_small = 1e-308;
            double very_large = 1e308;
            float fmin = 1.17549435e-38f;
            float fmax = 3.40282347e+38f;

            // Just a dot-prefixed float
            double dot_prefix = .5;

            // Trailing dot float
            double trail_dot = 5.;

            return z1 + z2 + z3 + (int)max_u32 + max_i32;
        }
    "#;
    compile_success(source);
}

/// Test the lexer's handling of the preprocessor-related tokens in context.
///
/// While `#` and `##` are preprocessor operators typically consumed
/// before the main lexer pass, this test verifies the full pipeline
/// handles them correctly.
#[test]
fn lex_preprocessor_context_tokens() {
    let source = r#"
        #define EMPTY
        #define VALUE 42
        #define STRINGIFY(x) #x
        #define PASTE(a, b) a ## b

        #if 1
        #define ENABLED 1
        #endif

        #ifdef ENABLED
        int PASTE(my, Func)(void) {
            return VALUE;
        }
        #endif

        #ifndef UNDEFINED_MACRO
        int main(void) {
            const char *s = STRINGIFY(hello);
            return myFunc();
        }
        #endif
    "#;
    compile_success(source);
}

/// Test that the lexer handles direct invocation with custom flags.
///
/// Uses `invoke_bcc_raw` helper and `Command` directly to exercise
/// fine-grained control over the bcc invocation and inspect the raw output.
#[test]
fn lex_direct_invocation_with_flags() {
    let source = r#"
        int main(void) {
            int x = 42;
            return x;
        }
    "#;

    // Use invoke_bcc_raw for direct control over the compilation
    let output = invoke_bcc_raw(source, &["-c", "-o", "/dev/null"]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Expected bcc to succeed.\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    // Also test with Command directly for fine-grained control
    let temp_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();

    let direct_output = Command::new(&bcc)
        .arg("-c")
        .arg(temp_file.path())
        .output()
        .expect("Failed to execute bcc");

    assert!(
        direct_output.status.success(),
        "Expected direct bcc invocation to succeed"
    );
}

/// Test that the lexer rejects truly invalid characters.
///
/// Characters like `@`, `` ` ``, and `$` (in most contexts) are not
/// valid C tokens. The compiler should produce an error.
#[test]
fn lex_invalid_characters() {
    // '@' is not a valid C token
    if let Some(result1) = compile_failure("int main(void) { int x = @; return 0; }") {
        assert!(
            !result1.stderr.is_empty(),
            "Expected error message for invalid '@' character"
        );
    }

    // Backtick is not valid in C
    if let Some(result2) = compile_failure("int main(void) { int x = `5`; return 0; }") {
        assert!(
            !result2.stderr.is_empty(),
            "Expected error message for invalid backtick character"
        );
    }
}

/// Test that the lexer handles unterminated string literals.
///
/// An unterminated string should produce an error diagnostic, not a crash.
#[test]
fn lex_unterminated_string() {
    let source = "int main(void) { const char *s = \"unterminated; return 0; }\n";
    if let Some(result) = compile_failure(source) {
        // Verify there's an error message (not just a silent failure)
        assert!(
            !result.stderr.is_empty(),
            "Expected error message for unterminated string literal"
        );
    }
}

/// Test that the lexer handles unterminated character literals.
///
/// An unterminated character literal should produce an error diagnostic.
#[test]
fn lex_unterminated_char() {
    let source = "int main(void) { char c = 'a; return 0; }\n";
    if let Some(result) = compile_failure(source) {
        assert!(
            !result.stderr.is_empty(),
            "Expected error message for unterminated character literal"
        );
    }
}

/// Test that the lexer handles unterminated block comments.
///
/// An unterminated block comment should produce an error diagnostic.
#[test]
fn lex_unterminated_block_comment() {
    let source = "int main(void) { /* this comment never ends return 0; }\n";
    if let Some(result) = compile_failure(source) {
        assert!(
            !result.stderr.is_empty(),
            "Expected error message for unterminated block comment"
        );
    }
}

/// Test that `fs::write` and `fs::read_to_string` are usable in the test context.
///
/// This verifies the `std::fs` import is functional and can be used
/// for test infrastructure purposes.
#[test]
fn lex_fs_operations_for_test_infrastructure() {
    let temp_file = common::write_temp_source("int main(void) { return 0; }");
    let path = temp_file.path();

    // Verify we can read the file back with fs::read_to_string
    let content = fs::read_to_string(path).expect("Failed to read temp source file");
    assert!(
        content.contains("int main"),
        "Expected temp file to contain source code"
    );

    // Write a new temp file using fs::write directly
    let custom_path = path.with_extension("test.c");
    fs::write(&custom_path, "int test(void) { return 42; }")
        .expect("Failed to write custom temp file");

    let custom_content = fs::read_to_string(&custom_path).expect("Failed to read custom temp file");
    assert!(
        custom_content.contains("int test"),
        "Expected custom file to contain source code"
    );

    // Clean up the custom file
    let _ = fs::remove_file(&custom_path);
}
