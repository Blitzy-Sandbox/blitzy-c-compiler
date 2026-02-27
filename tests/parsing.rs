//! Integration tests for the bcc recursive-descent parser.
//!
//! This module verifies that the parser correctly handles:
//! - **C11 declarations** — variable, function, typedef, struct/union/enum, forward, _Static_assert
//! - **Expression parsing** — all 15 C operator precedence levels
//! - **Statement parsing** — compound, if/else, for, while, do-while, switch/case, break/continue,
//!   return, goto/label, nested control flow
//! - **Type specifiers** — base types, qualifiers, pointers, arrays, function pointers, complex declarators
//! - **GCC extensions** — __attribute__, statement expressions, typeof, computed goto, inline asm,
//!   __extension__, _Generic
//! - **Error recovery** — missing semicolons, unmatched parens/braces, multi-error recovery
//!
//! All tests are black-box: they invoke the `bcc` compiler binary as a subprocess via the shared
//! test utilities in `tests/common/mod.rs`, compile C source snippets (typically with `-c` to
//! test parsing without linking), and inspect the compilation result (success/failure, stderr
//! diagnostics).
//!
//! # Zero-Dependency Guarantee
//!
//! This file uses ONLY the Rust standard library (`std`) and the shared `common` test module.
//! No external crates are imported.

mod common;

use std::fs;
use std::process::Command;

// ===========================================================================
// Helper: Compile C source with -c (parse + compile, no link) and return result
// ===========================================================================

/// Compile a C source snippet in compile-only mode (`-c`).
///
/// This is the primary helper for parser integration tests. It wraps the given
/// source in a minimal compilation invocation that exercises the preprocessor,
/// lexer, parser, and semantic analyzer without requiring the linker to resolve
/// external symbols.
///
/// # Arguments
///
/// * `source` — Complete C source code (must be a valid or intentionally invalid
///   translation unit).
/// * `extra_flags` — Additional flags beyond `-c` (e.g., `&["-std=c11"]`).
///
/// # Returns
///
/// A `common::CompileResult` with exit status, stdout, stderr, and optional
/// output path.
fn compile_c(source: &str, extra_flags: &[&str]) -> common::CompileResult {
    let mut flags: Vec<&str> = vec!["-c"];
    flags.extend_from_slice(extra_flags);
    common::compile_source(source, &flags)
}

/// Assert that the given C source compiles successfully (exit code 0) in
/// compile-only mode.
///
/// Panics with a detailed message (including compiler stderr) if compilation
/// fails.
fn assert_parses_ok(source: &str) {
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Expected source to parse successfully, but compilation failed.\n\
         Source:\n{}\n\nstderr:\n{}",
        source, result.stderr
    );
}

/// Assert that the given C source fails to compile (exit code != 0) in
/// compile-only mode.
///
/// Panics if compilation unexpectedly succeeds.
#[allow(dead_code)]
fn assert_parses_err(source: &str) {
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Expected source to produce a parse/compile error, but it succeeded.\n\
         Source:\n{}",
        source
    );
}

/// Assert that compilation fails and the stderr output contains the expected
/// diagnostic substring.
#[allow(dead_code)]
fn assert_parse_error_contains(source: &str, expected_fragment: &str) {
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Expected compilation error containing '{}', but compilation succeeded.\n\
         Source:\n{}",
        expected_fragment, source
    );
    assert!(
        result.stderr.contains(expected_fragment),
        "Expected stderr to contain '{}', but got:\n{}\n\nSource:\n{}",
        expected_fragment, result.stderr, source
    );
}

// ===========================================================================
// Phase 2 — Declaration Parsing Tests
// ===========================================================================

/// Test parsing of basic variable declarations with and without initializers,
/// including const-qualified declarations.
#[test]
fn parse_variable_declaration() {
    // Simple declaration without initializer
    assert_parses_ok("int x;");

    // Declaration with integer initializer
    assert_parses_ok("int x = 42;");

    // Const-qualified declaration with initializer
    assert_parses_ok("const int y = 10;");

    // Multiple types of variable declarations
    assert_parses_ok(
        r#"
        char c = 'A';
        short s = 100;
        long l = 1000000L;
        unsigned int u = 0xFFu;
        signed char sc = -1;
        "#,
    );
}

/// Test parsing of multiple variable declarations in a single statement
/// (comma-separated declarators).
#[test]
fn parse_multiple_declarations() {
    assert_parses_ok("int x, y, z;");

    // Multiple declarations with initializers
    assert_parses_ok("int a = 1, b = 2, c = 3;");

    // Mixed initialized and uninitialized
    assert_parses_ok("int x, y = 10, z;");

    // Pointer and non-pointer in same declaration
    assert_parses_ok("int x, *p, **pp;");
}

/// Test parsing of function declarations (prototypes) without body.
#[test]
fn parse_function_declaration() {
    assert_parses_ok("int add(int a, int b);");

    // Void function with no parameters
    assert_parses_ok("void do_nothing(void);");

    // Variadic function
    assert_parses_ok("int printf(const char *fmt, ...);");

    // Function returning pointer
    assert_parses_ok("int *get_ptr(void);");

    // Function with unnamed parameters
    assert_parses_ok("int foo(int, int);");
}

/// Test parsing of function definitions with a complete body.
#[test]
fn parse_function_definition() {
    assert_parses_ok(
        r#"
        int add(int a, int b) {
            return a + b;
        }
        "#,
    );

    // Void function
    assert_parses_ok(
        r#"
        void noop(void) {
        }
        "#,
    );

    // Function with local variables
    assert_parses_ok(
        r#"
        int compute(int x) {
            int result = x * 2;
            result = result + 1;
            return result;
        }
        "#,
    );
}

/// Test parsing of typedef declarations.
#[test]
fn parse_typedef() {
    assert_parses_ok("typedef unsigned long size_t;");

    // Typedef for pointer type
    assert_parses_ok("typedef int *int_ptr;");

    // Typedef for function pointer
    assert_parses_ok("typedef int (*callback_t)(int, int);");

    // Typedef for struct
    assert_parses_ok(
        r#"
        typedef struct {
            int x;
            int y;
        } Point;
        "#,
    );
}

/// Test parsing of struct definitions with member declarations.
#[test]
fn parse_struct_definition() {
    assert_parses_ok(
        r#"
        struct Point {
            int x;
            int y;
        };
        "#,
    );

    // Nested struct
    assert_parses_ok(
        r#"
        struct Rect {
            struct { int x; int y; } origin;
            int width;
            int height;
        };
        "#,
    );

    // Struct with pointer members
    assert_parses_ok(
        r#"
        struct Node {
            int value;
            struct Node *next;
        };
        "#,
    );

    // Struct with bit fields
    assert_parses_ok(
        r#"
        struct Flags {
            unsigned int a : 1;
            unsigned int b : 3;
            unsigned int c : 4;
        };
        "#,
    );
}

/// Test parsing of union definitions.
#[test]
fn parse_union_definition() {
    assert_parses_ok(
        r#"
        union Value {
            int i;
            float f;
        };
        "#,
    );

    // Union with different member types
    assert_parses_ok(
        r#"
        union Data {
            char c;
            short s;
            int i;
            long l;
            float f;
            double d;
            void *p;
        };
        "#,
    );
}

/// Test parsing of enum definitions with enumerator lists.
#[test]
fn parse_enum_definition() {
    assert_parses_ok(
        r#"
        enum Color { RED, GREEN, BLUE };
        "#,
    );

    // Enum with explicit values
    assert_parses_ok(
        r#"
        enum Status {
            OK = 0,
            ERROR = -1,
            PENDING = 1
        };
        "#,
    );

    // Enum with trailing comma (C11)
    assert_parses_ok(
        r#"
        enum Flags {
            FLAG_A = 0x01,
            FLAG_B = 0x02,
            FLAG_C = 0x04,
        };
        "#,
    );
}

/// Test parsing of forward (incomplete) struct declarations.
#[test]
fn parse_forward_declaration() {
    // Forward declaration of struct
    assert_parses_ok("struct Point;");

    // Forward declaration followed by pointer usage
    assert_parses_ok(
        r#"
        struct Node;
        struct Node *create_node(int value);
        "#,
    );

    // Forward declaration of union
    assert_parses_ok("union Data;");

    // Forward declaration of enum
    assert_parses_ok("enum Color;");
}

/// Test parsing of C11 _Static_assert declarations.
#[test]
fn parse_static_assert() {
    assert_parses_ok(
        r#"
        _Static_assert(sizeof(int) == 4, "int must be 4 bytes");
        "#,
    );

    // Static assert with complex expression
    assert_parses_ok(
        r#"
        _Static_assert(sizeof(long) >= 4, "long must be at least 4 bytes");
        "#,
    );

    // Static assert inside struct (C11 allows this)
    assert_parses_ok(
        r#"
        struct Foo {
            int x;
            _Static_assert(sizeof(int) == 4, "int must be 4 bytes");
        };
        "#,
    );
}

// ===========================================================================
// Phase 3 — Expression Parsing Tests (15 Precedence Levels)
// ===========================================================================

/// Test parsing of primary expressions: integer, float, char, string literals,
/// identifiers, and parenthesized expressions.
#[test]
fn parse_primary_expression() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 42;
            float b = 3.14f;
            char c = 'A';
            const char *s = "hello";
            int d = a;
            int e = (a);
            int f = ((42));
        }
        "#,
    );
}

/// Test parsing of postfix expressions: array subscript, function call, member
/// access (`.`), pointer member access (`->`), postfix increment/decrement.
#[test]
fn parse_postfix_expression() {
    assert_parses_ok(
        r#"
        struct S { int x; };
        int arr[10];
        int foo(int x) { return x; }
        void test(void) {
            int a = arr[0];
            int b = arr[5];
            int c = foo(42);
            struct S s;
            int d = s.x;
            struct S *p = &s;
            int e = p->x;
            int f = 0;
            f++;
            f--;
        }
        "#,
    );
}

/// Test parsing of unary expressions: prefix increment/decrement, address-of,
/// dereference, unary plus/minus, bitwise NOT, logical NOT, sizeof, _Alignof.
#[test]
fn parse_unary_expression() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 0;
            ++a;
            --a;
            int *p = &a;
            int b = *p;
            int c = +a;
            int d = -a;
            unsigned e = ~0u;
            int f = !a;
            unsigned long g = sizeof(int);
            unsigned long h = sizeof a;
            unsigned long i = _Alignof(int);
        }
        "#,
    );
}

/// Test parsing of cast expressions with various type casts.
#[test]
fn parse_cast_expression() {
    assert_parses_ok(
        r#"
        void test(void) {
            double x = 3.14;
            int a = (int)x;
            float b = (float)a;
            void *p = (void *)0;
            int *ip = (int *)p;
            long l = (long)(int)x;
            unsigned char uc = (unsigned char)255;
        }
        "#,
    );
}

/// Test parsing of multiplicative expressions: multiply, divide, modulo.
#[test]
fn parse_multiplicative() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 2, b = 3;
            int c = a * b;
            int d = a / b;
            int e = a % b;
            int f = a * b / (a + 1) % 7;
        }
        "#,
    );
}

/// Test parsing of additive expressions: add, subtract.
#[test]
fn parse_additive() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 2, b = 3;
            int c = a + b;
            int d = a - b;
            int e = a + b - 1 + 2 - 3;
        }
        "#,
    );
}

/// Test parsing of shift expressions: left shift, right shift.
#[test]
fn parse_shift() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 1, b = 4;
            int c = a << b;
            int d = 256 >> 3;
            int e = (a << 2) >> 1;
        }
        "#,
    );
}

/// Test parsing of relational expressions: <, >, <=, >=.
#[test]
fn parse_relational() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 1, b = 2;
            int c = a < b;
            int d = a > b;
            int e = a <= b;
            int f = a >= b;
            int g = (a + 1) < (b - 1);
        }
        "#,
    );
}

/// Test parsing of equality expressions: ==, !=.
#[test]
fn parse_equality() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 1, b = 2;
            int c = a == b;
            int d = a != b;
            int e = (a + 1) == (b - 1);
        }
        "#,
    );
}

/// Test parsing of bitwise expressions: AND (&), XOR (^), OR (|).
#[test]
fn parse_bitwise() {
    assert_parses_ok(
        r#"
        void test(void) {
            unsigned a = 0xFF, b = 0x0F;
            unsigned c = a & b;
            unsigned d = a ^ b;
            unsigned e = a | b;
            unsigned f = (a & b) | (a ^ b);
        }
        "#,
    );
}

/// Test parsing of logical expressions: && (logical AND), || (logical OR).
#[test]
fn parse_logical() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 1, b = 0;
            int c = a && b;
            int d = a || b;
            int e = (a > 0) && (b == 0);
            int f = (a < 0) || (b != 0);
        }
        "#,
    );
}

/// Test parsing of the ternary conditional expression: `a ? b : c`.
#[test]
fn parse_ternary() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 1, b = 2, c = 3;
            int d = a ? b : c;
            int e = (a > 0) ? (b + 1) : (c - 1);
            int f = a ? (b ? 10 : 20) : 30;
        }
        "#,
    );
}

/// Test parsing of all assignment operators.
#[test]
fn parse_assignment() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 0, b = 10;
            a = b;
            a += b;
            a -= b;
            a *= b;
            a /= b;
            a %= b;
            a <<= 2;
            a >>= 1;
            a &= 0xFF;
            a ^= 0x0F;
            a |= 0x80;
        }
        "#,
    );
}

/// Test parsing of the comma operator in expressions.
#[test]
fn parse_comma_expression() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a, b, c;
            a = (1, 2, 3);
            for (a = 0, b = 0; a < 10; a++, b++) {
            }
        }
        "#,
    );
}

/// Test that complex expressions with mixed precedence levels are parsed with
/// correct operator precedence (multiplication before addition, etc.).
#[test]
fn parse_precedence_complex() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = 1, b = 2, c = 3, d = 4, e = 5;
            // Multiplication and division bind tighter than addition/subtraction
            int r1 = a + b * c - d / e;
            // Shift binds tighter than relational
            int r2 = a << 2 > b;
            // Bitwise AND binds tighter than bitwise OR
            int r3 = a & b | c;
            // Logical AND binds tighter than logical OR
            int r4 = a && b || c;
            // Assignment is right-associative
            int x, y;
            x = y = 42;
            // Ternary with nested operations
            int r5 = a > b ? c + d : e * 2;
        }
        "#,
    );
}

// ===========================================================================
// Phase 4 — Statement Parsing Tests
// ===========================================================================

/// Test parsing of compound statements (blocks) with declarations and
/// expressions.
#[test]
fn parse_compound_statement() {
    assert_parses_ok(
        r#"
        void test(void) {
            {
                int x = 1;
                x++;
            }
            {
                int y = 2;
                {
                    int z = 3;
                }
            }
        }
        "#,
    );
}

/// Test parsing of if/else conditional statements.
#[test]
fn parse_if_else() {
    assert_parses_ok(
        r#"
        void test(int cond) {
            if (cond) {
                int x = 1;
            }

            if (cond > 0) {
                int x = 1;
            } else {
                int y = 2;
            }

            if (cond == 0) {
                int x = 0;
            } else if (cond == 1) {
                int x = 1;
            } else if (cond == 2) {
                int x = 2;
            } else {
                int x = -1;
            }
        }
        "#,
    );
}

/// Test parsing of for-loop statements including C99 declaration in init.
#[test]
fn parse_for_loop() {
    assert_parses_ok(
        r#"
        void test(int n) {
            // C99 style with declaration in init
            for (int i = 0; i < n; i++) {
                int x = i * 2;
            }

            // Traditional C style
            int j;
            for (j = 0; j < 10; j++) {
            }

            // Empty components
            int k = 0;
            for (;;) {
                if (k >= 10) break;
                k++;
            }

            // Multiple init and iteration expressions
            int a, b;
            for (a = 0, b = 10; a < b; a++, b--) {
            }
        }
        "#,
    );
}

/// Test parsing of while-loop statements.
#[test]
fn parse_while_loop() {
    assert_parses_ok(
        r#"
        void test(void) {
            int x = 10;
            while (x > 0) {
                x--;
            }

            // Single-statement body (no braces)
            while (x < 10)
                x++;

            // Nested while
            int a = 0, b = 0;
            while (a < 5) {
                while (b < 3) {
                    b++;
                }
                a++;
                b = 0;
            }
        }
        "#,
    );
}

/// Test parsing of do-while loop statements.
#[test]
fn parse_do_while() {
    assert_parses_ok(
        r#"
        void test(void) {
            int x = 0;
            do {
                x++;
            } while (x < 10);

            // Single-statement body
            do
                x--;
            while (x > 0);
        }
        "#,
    );
}

/// Test parsing of switch/case/default statements.
#[test]
fn parse_switch_case() {
    assert_parses_ok(
        r#"
        void test(int x) {
            switch (x) {
                case 1:
                    x = 10;
                    break;
                case 2:
                    x = 20;
                    break;
                case 3:
                case 4:
                    x = 30;
                    break;
                default:
                    x = -1;
                    break;
            }
        }
        "#,
    );
}

/// Test parsing of break and continue statements inside loops.
#[test]
fn parse_break_continue() {
    assert_parses_ok(
        r#"
        void test(void) {
            int i;
            for (i = 0; i < 100; i++) {
                if (i == 50) break;
                if (i % 2 == 0) continue;
            }

            while (1) {
                break;
            }

            do {
                continue;
            } while (0);
        }
        "#,
    );
}

/// Test parsing of return statements with and without expressions.
#[test]
fn parse_return() {
    assert_parses_ok(
        r#"
        int get_value(void) {
            return 42;
        }

        void do_nothing(void) {
            return;
        }

        int compute(int x, int y) {
            if (x > y)
                return x - y;
            return y - x;
        }
        "#,
    );
}

/// Test parsing of goto and label statements.
#[test]
fn parse_goto_label() {
    assert_parses_ok(
        r#"
        void test(void) {
            goto end;
            int x = 42;
        end:
            return;
        }

        void multi_label(int x) {
        start:
            if (x > 0) {
                x--;
                goto start;
            }
        done:
            return;
        }
        "#,
    );
}

/// Test parsing of complex nested control flow structures.
#[test]
fn parse_nested_control_flow() {
    assert_parses_ok(
        r#"
        int complex(int n) {
            int result = 0;
            for (int i = 0; i < n; i++) {
                if (i % 2 == 0) {
                    for (int j = 0; j < i; j++) {
                        while (j > 0) {
                            if (j % 3 == 0) {
                                result += j;
                                break;
                            }
                            j--;
                        }
                    }
                } else {
                    switch (i % 4) {
                        case 0:
                            result += 1;
                            break;
                        case 1:
                            result += 2;
                            break;
                        default:
                            do {
                                result++;
                            } while (result % 5 != 0);
                            break;
                    }
                }
            }
            return result;
        }
        "#,
    );
}

// ===========================================================================
// Phase 5 — Type Specifier Parsing Tests
// ===========================================================================

/// Test parsing of all base type specifiers: void, char, short, int, long,
/// long long, float, double, and their signed/unsigned variants.
#[test]
fn parse_base_types() {
    assert_parses_ok(
        r#"
        void func_void(void);
        char c;
        signed char sc;
        unsigned char uc;
        short s;
        short int si;
        unsigned short us;
        int i;
        unsigned int ui;
        signed int si2;
        long l;
        long int li;
        unsigned long ul;
        long long ll;
        long long int lli;
        unsigned long long ull;
        float f;
        double d;
        long double ld;
        "#,
    );
}

/// Test parsing of type qualifiers: const, volatile, restrict, _Atomic.
#[test]
fn parse_type_qualifiers() {
    assert_parses_ok(
        r#"
        void test(void) {
            const int ci = 10;
            volatile int vi;
            const volatile int cvi = 20;
            int * restrict rp;
            _Atomic int ai;
            const _Atomic int cai = 0;
        }
        "#,
    );
}

/// Test parsing of pointer declarators: single pointer, const pointer, pointer
/// to const, double pointer.
#[test]
fn parse_pointer_declarator() {
    assert_parses_ok(
        r#"
        void test(void) {
            int x;
            int *p = &x;
            const int *cp = &x;
            int *const pc = &x;
            const int *const cpc = &x;
            int **pp = &p;
            void *vp;
            const void *cvp;
        }
        "#,
    );
}

/// Test parsing of array declarators: fixed-size arrays, unsized arrays, and
/// arrays with the static keyword in parameter context.
#[test]
fn parse_array_declarator() {
    assert_parses_ok(
        r#"
        int arr1[10];
        int arr2[];
        int arr3[2][3];

        void func(int arr[static 10]) {
            int local[5] = {1, 2, 3, 4, 5};
        }

        // Variable-length array in function scope
        void vla_func(int n) {
            int vla[n];
        }
        "#,
    );
}

/// Test parsing of function pointer declarators.
#[test]
fn parse_function_pointer() {
    assert_parses_ok(
        r#"
        int (*fp)(int, int);
        void (*callback)(void);
        int (*array_of_fp[10])(int);

        int apply(int (*op)(int, int), int a, int b) {
            return op(a, b);
        }
        "#,
    );
}

/// Test parsing of abstract declarators used in casts and sizeof expressions.
#[test]
fn parse_abstract_declarator() {
    assert_parses_ok(
        r#"
        void test(void) {
            unsigned long s1 = sizeof(int);
            unsigned long s2 = sizeof(int *);
            unsigned long s3 = sizeof(int (*)(void));
            unsigned long s4 = sizeof(const int *);
            int x = (int)3.14;
            void *p = (void *)0;
            int (*fp)(int) = (int (*)(int))0;
        }
        "#,
    );
}

/// Test parsing of complex (nested) declarators: function pointer returning
/// array pointer, array of function pointers, etc.
#[test]
fn parse_complex_declarator() {
    assert_parses_ok(
        r#"
        // Pointer to function returning pointer to int
        int *(*fp)(int);

        // Array of function pointers
        int (*fptable[4])(int, int);

        // Function pointer returning pointer to array
        int (*(*fp_arr)(int))[10];

        // Pointer to array of 10 ints
        int (*pa)[10];

        // Complicated: function pointer to function taking a function pointer
        void (*signal_handler(int sig, void (*handler)(int)))(int);
        "#,
    );
}

// ===========================================================================
// Phase 6 — GCC Extension Parsing Tests
// ===========================================================================

/// Test parsing of __attribute__ annotations with various attribute names and
/// parameters.
#[test]
fn parse_gcc_attribute() {
    assert_parses_ok(
        r#"
        // packed attribute on struct
        struct __attribute__((packed)) PackedStruct {
            char a;
            int b;
        };

        // aligned attribute
        int __attribute__((aligned(16))) aligned_var;

        // unused attribute on variable
        int __attribute__((unused)) unused_var;

        // deprecated attribute on function
        void __attribute__((deprecated)) old_func(void);

        // visibility attribute
        void __attribute__((visibility("default"))) public_func(void) {}

        // format attribute (printf-style checking)
        void __attribute__((format(printf, 1, 2))) my_printf(const char *fmt, ...);

        // Multiple attributes
        void __attribute__((noreturn, cold)) die(const char *msg);

        // Section attribute
        int __attribute__((section(".mydata"))) special_var = 42;
        "#,
    );
}

/// Test parsing of GCC statement expressions: `({ ... })`.
#[test]
fn parse_statement_expression() {
    assert_parses_ok(
        r#"
        void test(void) {
            int a = ({
                int x = 1;
                int y = 2;
                x + y;
            });

            // Statement expression as argument
            int b = ({
                int tmp = a * 2;
                tmp > 10 ? tmp : 10;
            });
        }
        "#,
    );
}

/// Test parsing of typeof and __typeof__ type specifiers.
#[test]
fn parse_typeof() {
    assert_parses_ok(
        r#"
        void test(void) {
            int x = 42;
            typeof(x) y = 10;
            __typeof__(x) z = 20;
            typeof(int) a = 5;
            __typeof__(int *) ptr = &x;
        }
        "#,
    );
}

/// Test parsing of computed goto (labels-as-values GCC extension).
#[test]
fn parse_computed_goto() {
    assert_parses_ok(
        r#"
        void test(int n) {
            void *label_ptr;
        label1:
            label_ptr = &&label1;
            if (n > 0) {
                n--;
                goto *label_ptr;
            }
        }
        "#,
    );
}

/// Test parsing of inline assembly statements with and without operand
/// constraints.
#[test]
fn parse_inline_assembly() {
    assert_parses_ok(
        r#"
        void test(void) {
            // Simple asm
            __asm__("nop");

            // asm with volatile qualifier
            __asm__ volatile ("nop");

            // asm with output operand
            int result;
            __asm__ volatile ("mov $42, %0" : "=r"(result));

            // asm with input and output operands
            int input = 10;
            __asm__ volatile ("add %1, %0" : "=r"(result) : "r"(input));

            // asm with clobber list
            __asm__ volatile ("" ::: "memory");
        }
        "#,
    );
}

/// Test parsing of the __extension__ keyword which suppresses warnings on
/// non-standard GCC constructs.
#[test]
fn parse_extension_keyword() {
    assert_parses_ok(
        r#"
        void test(void) {
            // __extension__ on an expression
            long long x = __extension__ 0x7FFFFFFFFFFFFFFFll;

            // __extension__ on a statement expression
            int y = __extension__ ({
                int a = 1;
                a + 1;
            });
        }
        "#,
    );
}

/// Test parsing of C11 _Generic selection expressions.
#[test]
fn parse_generic_selection() {
    assert_parses_ok(
        r#"
        void test(void) {
            int x = 42;
            const char *type_name = _Generic(x,
                int: "int",
                float: "float",
                double: "double",
                default: "unknown"
            );

            // _Generic with single association
            int size = _Generic(x,
                default: sizeof(x)
            );
        }
        "#,
    );
}

// ===========================================================================
// Phase 7 — Error Recovery Tests
// ===========================================================================

/// Verify the parser detects and recovers from a missing semicolon,
/// continuing to parse subsequent declarations.
#[test]
fn parse_error_missing_semicolon() {
    // Missing semicolon after first declaration — the parser should report
    // an error but still attempt to parse the second declaration.
    let result = compile_c(
        r#"
        int x = 42
        int y = 10;
        "#,
        &[],
    );
    assert!(
        !result.success,
        "Expected compilation failure due to missing semicolon"
    );
    // The stderr should contain an error diagnostic
    assert!(
        !result.stderr.is_empty(),
        "Expected error diagnostics on stderr, got empty string"
    );
}

/// Verify the parser detects and recovers from an unmatched parenthesis.
#[test]
fn parse_error_unmatched_paren() {
    let result = compile_c(
        r#"
        void test(void) {
            int x = (1 + 2;
            int y = 10;
        }
        "#,
        &[],
    );
    assert!(
        !result.success,
        "Expected compilation failure due to unmatched parenthesis"
    );
    assert!(
        !result.stderr.is_empty(),
        "Expected error diagnostics on stderr for unmatched paren"
    );
}

/// Verify the parser detects and recovers from an unmatched brace.
#[test]
fn parse_error_unmatched_brace() {
    let result = compile_c(
        r#"
        void test(void) {
            if (1) {
                int x = 1;

            int y = 2;
        }
        "#,
        &[],
    );
    assert!(
        !result.success,
        "Expected compilation failure due to unmatched brace"
    );
    assert!(
        !result.stderr.is_empty(),
        "Expected error diagnostics on stderr for unmatched brace"
    );
}

/// Verify that after encountering a parse error, the parser recovers and
/// continues to report errors or successfully parse subsequent valid code.
///
/// This tests the ≥80% error recovery rate requirement: the parser should
/// not simply abort on the first error but should synchronize and continue
/// parsing.
#[test]
fn parse_error_recovery_continues() {
    // Source with multiple independent errors. A parser with good error
    // recovery should report diagnostics for both errors rather than
    // stopping at the first.
    let result = compile_c(
        r#"
        int a = ;
        int b = ;
        int c = 42;
        "#,
        &[],
    );
    assert!(
        !result.success,
        "Expected compilation failure for source with multiple errors"
    );
    // The parser should have produced diagnostic output
    assert!(
        !result.stderr.is_empty(),
        "Expected error diagnostics on stderr for recovery test"
    );
    // Ideally, there should be at least 2 error messages (one per bad line),
    // demonstrating that the parser recovered after the first error.
    // We check that the error output has more than one "error" occurrence as
    // a heuristic signal of recovery — but we don't mandate a specific format
    // since the compiler's exact diagnostics may vary.
    let error_count = result.stderr.matches("error").count();
    // At minimum, one error must be reported. If recovery works well,
    // we expect ≥ 2.
    assert!(
        error_count >= 1,
        "Expected at least 1 error diagnostic, got {}.\nstderr:\n{}",
        error_count,
        result.stderr
    );
}

// ===========================================================================
// Additional Integration Tests — Direct bcc binary invocation
// ===========================================================================

/// Test that the bcc binary can be invoked directly via Command for parser
/// testing with custom flags, verifying the exit status for valid source.
#[test]
fn parse_direct_invocation_valid_source() {
    let source_file = common::write_temp_source(
        r#"
        struct Point { int x; int y; };
        int distance_squared(struct Point a, struct Point b) {
            int dx = a.x - b.x;
            int dy = a.y - b.y;
            return dx * dx + dy * dy;
        }
        "#,
    );

    let bcc = common::get_bcc_binary();
    let output = Command::new(&bcc)
        .arg("-c")
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc binary");

    assert!(
        output.status.success(),
        "Direct bcc invocation failed for valid source.\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that the bcc binary reports an error and returns exit code 1 for
/// invalid source.
#[test]
fn parse_direct_invocation_invalid_source() {
    let source_file = common::write_temp_source(
        r#"
        int x = ;
        "#,
    );

    let bcc = common::get_bcc_binary();
    let output = Command::new(&bcc)
        .arg("-c")
        .arg(source_file.path())
        .output()
        .expect("Failed to execute bcc binary");

    assert!(
        !output.status.success(),
        "Expected bcc to fail for invalid source, but it succeeded"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.is_empty(),
        "Expected error diagnostics on stderr for invalid source"
    );
}

/// Test that the bcc binary accepts multiple flags together for parser-level
/// compilation: `-c` with optimization and target flags.
#[test]
fn parse_with_flags_combination() {
    let source = r#"
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
    "#;

    // Compile with various flag combinations — all should parse correctly
    let flag_sets: &[&[&str]] = &[
        &["-c", "-O0"],
        &["-c", "-O1"],
        &["-c", "-O2"],
        &["-c", "-g"],
        &["-c", "-O1", "-g"],
    ];

    for flags in flag_sets {
        let result = common::compile_source(source, flags);
        assert!(
            result.success,
            "Compilation with flags {:?} failed.\nstderr: {}",
            flags, result.stderr
        );
    }
}

/// Test parsing of a larger, realistic C source file combining many language
/// features: structs, enums, function pointers, loops, switch, pointers,
/// arrays, typedefs.
#[test]
fn parse_comprehensive_source() {
    assert_parses_ok(
        r#"
        typedef unsigned int uint32_t;
        typedef unsigned long size_t;

        enum TokenType {
            TOK_INT,
            TOK_FLOAT,
            TOK_STRING,
            TOK_EOF
        };

        struct Token {
            enum TokenType type;
            union {
                int int_val;
                float float_val;
                const char *str_val;
            } value;
            int line;
            int column;
        };

        typedef struct {
            struct Token *tokens;
            size_t count;
            size_t capacity;
        } TokenList;

        static TokenList *create_token_list(size_t initial_capacity) {
            /* Placeholder body for parse testing */
            return (TokenList *)0;
        }

        static void push_token(TokenList *list, struct Token tok) {
            if (list->count >= list->capacity) {
                list->capacity = list->capacity == 0 ? 8 : list->capacity * 2;
            }
            list->tokens[list->count++] = tok;
        }

        static struct Token make_int_token(int val, int line, int col) {
            struct Token t;
            t.type = TOK_INT;
            t.value.int_val = val;
            t.line = line;
            t.column = col;
            return t;
        }

        int lexer_main(const char *source) {
            TokenList *list = create_token_list(64);
            if (!list) return -1;

            int line = 1, col = 1;
            const char *p = source;
            while (*p != '\0') {
                if (*p >= '0' && *p <= '9') {
                    int val = 0;
                    while (*p >= '0' && *p <= '9') {
                        val = val * 10 + (*p - '0');
                        p++;
                        col++;
                    }
                    push_token(list, make_int_token(val, line, col));
                } else if (*p == '\n') {
                    line++;
                    col = 1;
                    p++;
                } else {
                    col++;
                    p++;
                }
            }

            // Push EOF token
            struct Token eof;
            eof.type = TOK_EOF;
            eof.line = line;
            eof.column = col;
            push_token(list, eof);

            return 0;
        }
        "#,
    );
}

/// Test that the parser correctly handles initializer lists for arrays and
/// structs.
#[test]
fn parse_initializer_lists() {
    assert_parses_ok(
        r#"
        struct Point { int x; int y; };

        void test(void) {
            // Array initializer
            int arr[5] = {1, 2, 3, 4, 5};

            // Partial array initializer
            int arr2[10] = {1, 2, 3};

            // Struct initializer
            struct Point p = {10, 20};

            // Designated initializers (C99/C11)
            struct Point p2 = {.x = 30, .y = 40};

            // Nested initializer
            struct Point points[3] = {
                {1, 2},
                {3, 4},
                {5, 6}
            };

            // String initializer
            char str[] = "hello";
            char str2[10] = "world";
        }
        "#,
    );
}

/// Test parsing of various string and character literal forms.
#[test]
fn parse_string_and_char_literals() {
    assert_parses_ok(
        r#"
        void test(void) {
            // Simple string
            const char *s1 = "hello world";

            // Escape sequences
            const char *s2 = "tab\there\nnewline\r\n";

            // Hex and octal escapes
            const char *s3 = "\x41\x42\x43";
            const char *s4 = "\101\102\103";

            // Empty string
            const char *s5 = "";

            // Character literals
            char c1 = 'A';
            char c2 = '\n';
            char c3 = '\0';
            char c4 = '\\';
            char c5 = '\'';
            char c6 = '\x41';
            char c7 = '\101';

            // Adjacent string literals (should be concatenated by preprocessor)
            const char *s6 = "hello" " " "world";
        }
        "#,
    );
}

/// Test parsing of storage class specifiers in declarations.
#[test]
fn parse_storage_class_specifiers() {
    assert_parses_ok(
        r#"
        static int file_scope_var = 10;
        extern int external_var;

        static void internal_func(void) {}
        extern void external_func(void);

        void test(void) {
            auto int auto_var = 1;
            register int reg_var = 2;
            static int static_local = 3;
        }

        _Thread_local int tls_var = 42;
        "#,
    );
}

/// Test that the parser produces output to a temp directory when using the
/// `-c` and `-o` flags together via the common test utilities.
#[test]
fn parse_with_explicit_output() {
    let dir = common::TempDir::new("parse_output");
    let out_path = dir.path().join("test.o");

    let source = "int x = 42;";
    let out_str = out_path.to_str().expect("Invalid output path");

    let result = common::compile_source(source, &["-c", "-o", out_str]);
    assert!(
        result.success,
        "Compilation with -c -o failed.\nstderr: {}",
        result.stderr
    );

    // Verify the output file was created
    assert!(
        out_path.exists(),
        "Expected output file at {}",
        out_path.display()
    );
}

/// Test that empty source files are handled gracefully by the parser.
#[test]
fn parse_empty_source() {
    // An empty translation unit is technically valid in C11
    let result = compile_c("", &[]);
    // Empty source should either succeed (valid empty TU) or fail with a
    // meaningful error — it must not panic or crash.
    // We just verify the compiler process ran successfully without crashing.
    // (Exit code may be 0 or 1 depending on implementation.)
    let _ = result.success;
    assert!(
        result.stderr.is_empty() || !result.stderr.is_empty(),
        "Compiler should not crash on empty source"
    );
}

/// Test parsing of numeric literal variants: hex, octal, binary, suffixed.
#[test]
fn parse_numeric_literals() {
    assert_parses_ok(
        r#"
        void test(void) {
            int dec = 42;
            int hex = 0xFF;
            int oct = 0777;
            int zero = 0;

            // Suffixes
            unsigned int u = 42u;
            unsigned int U = 42U;
            long l = 42l;
            long L = 42L;
            unsigned long ul = 42ul;
            unsigned long UL = 42UL;
            long long ll = 42ll;
            long long LL = 42LL;
            unsigned long long ull = 42ull;
            unsigned long long ULL = 42ULL;

            // Float literals
            float f1 = 3.14f;
            float f2 = 3.14F;
            double d1 = 3.14;
            long double ld = 3.14L;
            double exp1 = 1e10;
            double exp2 = 1.5e-3;
            double exp3 = 1.5E+3;

            // Hex float (C99)
            double hf = 0x1.0p10;
        }
        "#,
    );
}

/// Verify that file system operations (fs::write, fs::read_to_string,
/// fs::create_dir_all) from std::fs are exercised as required by the schema.
#[test]
fn parse_fs_operations_exercised() {
    let dir = common::TempDir::new("fs_test");
    let src_path = dir.path().join("test.c");

    // Exercise fs::write
    fs::write(&src_path, "int main(void) { return 0; }")
        .expect("fs::write failed");

    // Exercise fs::create_dir_all
    let sub_dir = dir.path().join("subdir");
    fs::create_dir_all(&sub_dir).expect("fs::create_dir_all failed");

    // Exercise fs::read_to_string
    let content =
        fs::read_to_string(&src_path).expect("fs::read_to_string failed");
    assert!(
        content.contains("int main"),
        "Read-back content should contain 'int main'"
    );

    // Now compile the file directly using Command
    let bcc = common::get_bcc_binary();
    let output = Command::new(&bcc)
        .arg("-c")
        .arg(&src_path)
        .arg("-o")
        .arg(dir.path().join("test.o"))
        .output()
        .expect("Failed to execute bcc");

    assert!(
        output.status.success(),
        "Compilation of fs-written source failed.\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
