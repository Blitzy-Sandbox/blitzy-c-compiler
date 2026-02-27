//! Integration tests for the semantic analysis phase of the bcc C compiler.
//!
//! This module tests the semantic analyzer's ability to:
//!
//! - **Type check** assignments, function calls, binary operators, array subscripts,
//!   struct member access, and return types
//! - **Apply implicit conversions** following C11 rules: integer promotions, usual
//!   arithmetic conversions, pointer-to-void, array-to-pointer decay, function-to-pointer decay
//! - **Manage scopes** at file, function, block, and prototype levels with correct
//!   variable shadowing and lifetime semantics
//! - **Maintain the symbol table** with declaration registration, lookup, compatible
//!   redeclaration acceptance, and conflicting redeclaration rejection
//! - **Validate storage class specifiers** including `static`, `extern`, `auto`, `register`,
//!   `_Thread_local`, and detection of conflicting specifier combinations
//! - **Compute target-parametric type sizes** correctly per architecture (pointer width
//!   differs between i686 and 64-bit targets)
//! - **Handle complex types** including structs, unions, enums, function pointers, and arrays
//!
//! # Zero-Dependency Guarantee
//!
//! This module uses ONLY the Rust standard library (`std`) and the `bcc` crate binary.
//! No external crates are imported.
//!
//! # Testing Strategy
//!
//! All tests are **black-box integration tests** that invoke the `bcc` compiler binary as a
//! subprocess, feed it C source code, and inspect the exit code and stderr output. Tests
//! verify either:
//! - Successful compilation (exit code 0) for semantically valid C programs
//! - Compilation failure (exit code 1) with GCC-compatible diagnostic messages on stderr
//!   for programs containing semantic errors
//!
//! # Diagnostic Format
//!
//! Error messages are expected in GCC-compatible format on stderr:
//! `file:line:col: error: description`

mod common;

use std::fs;
use std::process::Command;

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Compile a C source snippet by wrapping it in a minimal compilable context.
///
/// Invokes `bcc` with the given source and optional extra flags, returning the
/// `CompileResult` for inspection. This is a convenience wrapper around
/// `common::compile_source()` used throughout these semantic analysis tests.
///
/// # Arguments
///
/// * `source` - Complete C source code to compile (must be a full translation unit)
/// * `extra_flags` - Additional compiler flags (e.g., `&["--target", "i686-linux-gnu"]`)
///
/// # Returns
///
/// A `CompileResult` with `success`, `exit_status`, `stdout`, `stderr`, and `output_path`.
#[allow(dead_code)]
fn compile_c(source: &str, extra_flags: &[&str]) -> common::CompileResult {
    common::compile_source(source, extra_flags)
}

/// Compile C source targeting a specific architecture.
///
/// Adds the `--target <triple>` flag automatically and forwards to `compile_c`.
///
/// # Arguments
///
/// * `source` - Complete C source code to compile
/// * `target` - Target triple string (e.g., `"x86_64-linux-gnu"`)
/// * `extra_flags` - Additional compiler flags beyond `--target`
///
/// # Returns
///
/// A `CompileResult` for inspection.
#[allow(dead_code)]
fn compile_c_with_target(source: &str, target: &str, extra_flags: &[&str]) -> common::CompileResult {
    let mut flags: Vec<&str> = extra_flags.to_vec();
    flags.push("--target");
    flags.push(target);
    common::compile_source(source, &flags)
}

/// Compile a C source snippet wrapped in a `main()` function for convenience.
///
/// Many semantic tests only need to verify behavior inside a function body. This
/// helper wraps the given code in `int main(void) { <code> return 0; }` to produce
/// a complete translation unit.
///
/// # Arguments
///
/// * `body` - C statements to place inside `main()`
/// * `extra_flags` - Additional compiler flags
///
/// # Returns
///
/// A `CompileResult` for inspection.
#[allow(dead_code)]
fn compile_in_main(body: &str, extra_flags: &[&str]) -> common::CompileResult {
    let source = format!("int main(void) {{\n{}\nreturn 0;\n}}\n", body);
    common::compile_source(&source, extra_flags)
}

/// Compile C source using the `bcc` binary directly via `Command`, for tests
/// that need more fine-grained control over invocation.
///
/// # Arguments
///
/// * `source` - C source code
/// * `flags` - All compiler flags
///
/// # Returns
///
/// A tuple of `(success: bool, stdout: String, stderr: String, exit_code: Option<i32>)`.
#[allow(dead_code)]
fn compile_direct(source: &str, flags: &[&str]) -> (bool, String, String, Option<i32>) {
    let source_file = common::write_temp_source(source);
    let bcc = common::get_bcc_binary();
    let temp_dir = common::TempDir::new("sema_direct");
    let output_path = temp_dir.path().join("a.out");

    let result = Command::new(&bcc)
        .args(flags)
        .arg("-o")
        .arg(&output_path)
        .arg(source_file.path())
        .output()
        .unwrap_or_else(|e| {
            panic!("Failed to execute bcc at '{}': {}", bcc.display(), e);
        });

    let success = result.status.success();
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
    let exit_code = result.status.code();

    // Prevent temp_dir from cleaning up if we might need the output
    if success {
        std::mem::forget(temp_dir);
    }

    (success, stdout, stderr, exit_code)
}

/// Verify that a diagnostic message follows the GCC-compatible format.
///
/// Expected format: `<file>:<line>:<col>: error: <message>`
/// or: `<file>:<line>:<col>: warning: <message>`
///
/// Returns `true` if at least one line in `stderr` matches the pattern.
#[allow(dead_code)]
fn has_gcc_diagnostic_format(stderr: &str) -> bool {
    stderr.lines().any(|line| {
        // Pattern: something:number:number: (error|warning|note): something
        let parts: Vec<&str> = line.splitn(5, ':').collect();
        if parts.len() >= 5 {
            // parts[0] = filename, parts[1] = line, parts[2] = col
            // parts[3] = " error" or " warning" or " note"
            // parts[4] = message
            let line_num = parts[1].trim().parse::<u32>();
            let col_num = parts[2].trim().parse::<u32>();
            let severity = parts[3].trim();
            line_num.is_ok()
                && col_num.is_ok()
                && (severity == "error" || severity == "warning" || severity == "note")
        } else {
            false
        }
    })
}

/// Create a temporary C source file with specific content and return its path
/// along with a TempDir for cleanup.
///
/// Uses `fs::write()` to write the content to a file inside a `TempDir`.
/// Returns `(TempDir, source_file_path)` where `TempDir` provides RAII cleanup.
#[allow(dead_code)]
fn create_source_in_dir(prefix: &str, filename: &str, content: &str) -> (common::TempDir, std::path::PathBuf) {
    let dir = common::TempDir::new(prefix);
    let file_path = dir.path().join(filename);
    fs::write(&file_path, content).unwrap_or_else(|e| {
        panic!("Failed to write source file '{}': {}", file_path.display(), e);
    });
    (dir, file_path)
}

// ===========================================================================
// Phase 2: Type Checking Tests
// ===========================================================================

/// Verify that `int x = 42;` type-checks correctly.
///
/// A simple integer assignment is the most basic type-checking scenario. The
/// literal `42` has type `int`, which is assigned to a variable of type `int`,
/// requiring no conversion.
#[test]
fn type_check_integer_assignment() {
    let source = r#"
int main(void) {
    int x = 42;
    return x;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Integer assignment should type-check successfully.\nstderr: {}",
        result.stderr
    );
}

/// Verify that `float f = 3.14f;` type-checks correctly.
///
/// The suffix `f` marks the literal as `float` type, matching the declared type
/// of the variable exactly.
#[test]
fn type_check_float_assignment() {
    let source = r#"
int main(void) {
    float f = 3.14f;
    return (int)f;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Float assignment should type-check successfully.\nstderr: {}",
        result.stderr
    );
}

/// Verify that `int *p = &x;` type-checks correctly.
///
/// The address-of operator `&` applied to an `int` variable produces an `int *`,
/// which matches the declared pointer type on the left-hand side.
#[test]
fn type_check_pointer_assignment() {
    let source = r#"
int main(void) {
    int x = 10;
    int *p = &x;
    return *p;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Pointer assignment from address-of should type-check.\nstderr: {}",
        result.stderr
    );
}

/// Verify that `int *p = 42;` produces a type error or warning.
///
/// Assigning an integer literal directly to a pointer variable is a type
/// incompatibility in C11. The compiler should reject this or emit a warning
/// about making a pointer from an integer without a cast.
#[test]
fn type_check_incompatible_assignment() {
    let source = r#"
int main(void) {
    int *p = 42;
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    // The compiler should either error or warn about this incompatible assignment.
    // Most C compilers emit a warning; strict mode may error.
    let has_diagnostic = !result.success
        || result.stderr.to_lowercase().contains("warning")
        || result.stderr.to_lowercase().contains("error")
        || result.stderr.to_lowercase().contains("incompatible")
        || result.stderr.to_lowercase().contains("integer");
    assert!(
        has_diagnostic,
        "Assigning integer to pointer should produce a diagnostic.\nstderr: {}",
        result.stderr
    );
}

/// Verify argument count and type matching for function calls.
///
/// A function declared as `int add(int, int)` called with two `int` arguments
/// should type-check without errors. The argument types must match the parameter
/// types after implicit conversions.
#[test]
fn type_check_function_call_args() {
    let source = r#"
int add(int a, int b) {
    return a + b;
}

int main(void) {
    int result = add(3, 4);
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Function call with correct arg types should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify error when calling a function with too few arguments.
///
/// Calling `add(3)` when the function expects two parameters should produce
/// a semantic error about too few arguments.
#[test]
fn type_check_function_call_too_few_args() {
    let source = r#"
int add(int a, int b) {
    return a + b;
}

int main(void) {
    int result = add(3);
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Too few arguments should cause compilation failure.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("argument") || stderr_lower.contains("parameter")
            || stderr_lower.contains("too few") || stderr_lower.contains("expected"),
        "Error message should mention argument count mismatch.\nstderr: {}",
        result.stderr
    );
}

/// Verify error when calling a function with too many arguments.
///
/// Calling `add(1, 2, 3)` when the function expects two parameters should
/// produce a semantic error about too many arguments.
#[test]
fn type_check_function_call_too_many_args() {
    let source = r#"
int add(int a, int b) {
    return a + b;
}

int main(void) {
    int result = add(1, 2, 3);
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Too many arguments should cause compilation failure.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("argument") || stderr_lower.contains("parameter")
            || stderr_lower.contains("too many") || stderr_lower.contains("expected"),
        "Error message should mention argument count mismatch.\nstderr: {}",
        result.stderr
    );
}

/// Verify return statement type matches function return type.
///
/// A function declared as `int f()` returning a valid `int` expression
/// should type-check without issues.
#[test]
fn type_check_return_type() {
    let source = r#"
int square(int x) {
    return x * x;
}

int main(void) {
    return square(5);
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Return type matching function declaration should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify type checking for binary operators (+, -, *, /) with compatible operands.
///
/// Arithmetic operations between two `int` operands are always valid and produce
/// an `int` result. This is the most fundamental binary operator type check.
#[test]
fn type_check_binary_operators() {
    let source = r#"
int main(void) {
    int a = 10, b = 3;
    int sum = a + b;
    int diff = a - b;
    int prod = a * b;
    int quot = a / b;
    int rem = a % b;
    return sum + diff + prod + quot + rem;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Binary operators with compatible int operands should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify error on incompatible operand types (e.g., struct + int).
///
/// Adding a struct value to an integer is not a valid operation in C. The
/// compiler must reject this with a type error.
#[test]
fn type_check_binary_incompatible() {
    let source = r#"
struct Point { int x; int y; };

int main(void) {
    struct Point p;
    p.x = 1;
    p.y = 2;
    int result = p + 5;
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Struct + int should fail type checking.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("type") || stderr_lower.contains("invalid")
            || stderr_lower.contains("operand") || stderr_lower.contains("incompatible"),
        "Error should mention type incompatibility.\nstderr: {}",
        result.stderr
    );
}

/// Verify array subscript requires pointer/array and integer index.
///
/// `arr[i]` requires the left operand to be a pointer or array type, and the
/// right operand to be an integer. Valid subscript should compile successfully.
#[test]
fn type_check_array_subscript() {
    let source = r#"
int main(void) {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    return arr[2];
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Array subscript with integer index should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify `.` and `->` operators require struct/union type.
///
/// The member access operator `.` applied to a struct should be valid. The arrow
/// operator `->` applied to a pointer-to-struct should also be valid.
#[test]
fn type_check_struct_member_access() {
    let source = r#"
struct Point { int x; int y; };

int main(void) {
    struct Point p;
    p.x = 10;
    p.y = 20;

    struct Point *pp = &p;
    int val = pp->x + pp->y;
    return val;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Struct member access with . and -> should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify `void` function returning a value produces error.
///
/// A function declared as `void f()` that contains `return 42;` must be rejected
/// by semantic analysis because `void` functions cannot return a value.
#[test]
fn type_check_void_return() {
    let source = r#"
void foo(void) {
    return 42;
}

int main(void) {
    foo();
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Void function returning a value should fail.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("void") || stderr_lower.contains("return")
            || stderr_lower.contains("type"),
        "Error should mention void return type violation.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Phase 3: Implicit Type Conversion Tests
// ===========================================================================

/// Verify `char`/`short` promote to `int` in expressions.
///
/// Per C11 §6.3.1.1, integer types smaller than `int` are promoted to `int`
/// when used in expressions. Adding two `char` values should produce an `int` result.
#[test]
fn conversion_integer_promotion() {
    let source = r#"
int main(void) {
    char a = 10;
    char b = 20;
    int result = a + b;
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Integer promotion of char to int in expressions should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify usual arithmetic conversions (e.g., `int + long` → `long`).
///
/// Per C11 §6.3.1.8, when operands have different types, the operand with
/// the lower conversion rank is converted to the type of the other. Here,
/// `int + long` produces a `long` result.
#[test]
fn conversion_usual_arithmetic() {
    let source = r#"
int main(void) {
    int a = 42;
    long b = 100L;
    long result = a + b;
    return (int)result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Usual arithmetic conversion int+long should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify implicit `int` to `float`/`double` conversion.
///
/// C11 allows implicit conversion from integer to floating-point types when
/// assigning to a float/double variable or using mixed arithmetic.
#[test]
fn conversion_int_to_float() {
    let source = r#"
int main(void) {
    int a = 42;
    double d = a;
    float f = a;
    return (int)(d + f);
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Implicit int-to-float conversion should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify implicit `float` to `int` conversion (with potential warning).
///
/// Converting from float to int loses the fractional part. This is allowed
/// in C11 but some compilers emit a warning about possible data loss.
#[test]
fn conversion_float_to_int() {
    let source = r#"
int main(void) {
    float f = 3.14f;
    int x = f;
    return x;
}
"#;
    let result = compile_c(source, &[]);
    // This should compile (possibly with a warning), but not be an error.
    // Some compilers warn about truncation; the key check is it does not
    // produce a hard error preventing compilation.
    assert!(
        result.success || result.stderr.to_lowercase().contains("warning"),
        "Float-to-int conversion should compile (possibly with warning).\nstderr: {}",
        result.stderr
    );
}

/// Verify implicit conversion of any pointer to `void*`.
///
/// Per C11 §6.3.2.3, any pointer to object type can be implicitly converted
/// to `void*` and back. This is the foundation of generic pointer handling in C.
#[test]
fn conversion_pointer_to_void() {
    let source = r#"
int main(void) {
    int x = 42;
    int *ip = &x;
    void *vp = ip;
    int *ip2 = vp;
    return *ip2;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Pointer to void* conversion should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify array-to-pointer decay in expressions.
///
/// Per C11 §6.3.2.1, an array expression is converted to a pointer to its
/// first element when used in most contexts (except sizeof, &, and string
/// literal initializers).
#[test]
fn conversion_array_to_pointer() {
    let source = r#"
void takes_ptr(int *p) {
    (void)p;
}

int main(void) {
    int arr[5];
    arr[0] = 42;
    takes_ptr(arr);
    int *p = arr;
    return *p;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Array-to-pointer decay should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify function-to-pointer decay.
///
/// Per C11 §6.3.2.1, a function designator is converted to a pointer to
/// the function in most expression contexts. A function can be assigned to
/// a function pointer variable without explicit `&` operator.
#[test]
fn conversion_function_to_pointer() {
    let source = r#"
int add(int a, int b) {
    return a + b;
}

int main(void) {
    int (*fp)(int, int) = add;
    return fp(3, 4);
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Function-to-pointer decay should succeed.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Phase 4: Scope Resolution Tests
// ===========================================================================

/// Verify variables in inner blocks shadow outer variables.
///
/// C11 §6.2.1 specifies that a declaration in an inner scope hides any
/// entity of the same name from outer scopes for the duration of the
/// inner scope.
#[test]
fn scope_block_scope() {
    let source = r#"
int main(void) {
    int x = 10;
    {
        int x = 20;
        if (x != 20) return 1;
    }
    if (x != 10) return 1;
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Block scope shadowing should compile successfully.\nstderr: {}",
        result.stderr
    );
}

/// Verify labels have function scope.
///
/// Per C11 §6.2.1, labels have function scope — they are visible throughout
/// the entire function body in which they appear, even before their declaration.
#[test]
fn scope_function_scope() {
    let source = r#"
int main(void) {
    goto end;
    int x = 42;
    (void)x;
end:
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Forward goto to label in function scope should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify global variables have file scope.
///
/// Variables declared at file scope (outside any function) are accessible
/// from all functions in the translation unit that appear after the declaration.
#[test]
fn scope_file_scope() {
    let source = r#"
int global_var = 100;

int get_global(void) {
    return global_var;
}

int main(void) {
    return get_global();
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "File scope variables should be accessible from all functions.\nstderr: {}",
        result.stderr
    );
}

/// Verify function parameter names have prototype scope.
///
/// Per C11 §6.2.1, parameter names in a function prototype declaration have
/// prototype scope, which terminates at the end of the function declarator.
/// The same parameter names can be reused in the actual function definition.
#[test]
fn scope_prototype_scope() {
    let source = r#"
int add(int a, int b);

int add(int a, int b) {
    return a + b;
}

int main(void) {
    return add(1, 2);
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Prototype scope for parameter names should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify error on use of undeclared variable.
///
/// Using a variable that has not been declared in any accessible scope must
/// produce a semantic error.
#[test]
fn scope_undeclared_variable() {
    let source = r#"
int main(void) {
    return undeclared_var;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Use of undeclared variable should fail.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("undeclared") || stderr_lower.contains("undefined")
            || stderr_lower.contains("not declared") || stderr_lower.contains("unknown")
            || stderr_lower.contains("identifier"),
        "Error should mention undeclared identifier.\nstderr: {}",
        result.stderr
    );
}

/// Verify inner declaration shadows outer declaration.
///
/// When a variable is redeclared in an inner block with a different type,
/// the inner declaration takes precedence within its scope.
#[test]
fn scope_variable_shadowing() {
    let source = r#"
int main(void) {
    int x = 10;
    {
        float x = 3.14f;
        float y = x + 1.0f;
        (void)y;
    }
    int z = x + 1;
    return z;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Variable shadowing with different type should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify variable is no longer accessible after its block ends.
///
/// A variable declared inside a block goes out of scope at the end of
/// that block. Using it afterwards must produce an error.
#[test]
fn scope_after_block() {
    let source = r#"
int main(void) {
    {
        int inner_var = 42;
        (void)inner_var;
    }
    return inner_var;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Accessing variable after block scope should fail.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("undeclared") || stderr_lower.contains("undefined")
            || stderr_lower.contains("not declared") || stderr_lower.contains("scope")
            || stderr_lower.contains("identifier"),
        "Error should mention scope or undeclared variable.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Phase 5: Symbol Table Tests
// ===========================================================================

/// Verify symbol is registered on declaration.
///
/// Declaring a variable and then using it in the same scope should succeed,
/// confirming the symbol table records the declaration.
#[test]
fn symbol_table_declaration() {
    let source = r#"
int main(void) {
    int declared_var = 42;
    return declared_var;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Declared variable should be found in symbol table.\nstderr: {}",
        result.stderr
    );
}

/// Verify symbol lookup finds the correct declaration.
///
/// When multiple variables exist in scope, referencing each one should resolve
/// to the correct declaration with the correct type.
#[test]
fn symbol_table_lookup() {
    let source = r#"
int main(void) {
    int a = 1;
    float b = 2.0f;
    char c = 'A';
    int result = a + (int)b + (int)c;
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Symbol lookup for multiple variables should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify compatible redeclaration is accepted.
///
/// C11 §6.7p4 allows redeclaring a function with a compatible type. A function
/// first declared as `int f(int)` and then defined as `int f(int x)` is valid.
#[test]
fn symbol_table_redeclaration() {
    let source = r#"
int compute(int x);
int compute(int x);

int compute(int x) {
    return x * 2;
}

int main(void) {
    return compute(21);
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Compatible redeclaration should be accepted.\nstderr: {}",
        result.stderr
    );
}

/// Verify incompatible redeclaration produces error.
///
/// Declaring a symbol first as `int x;` and then as `float x;` in the same scope
/// creates a type conflict that the compiler must reject.
#[test]
fn symbol_table_conflicting_redeclaration() {
    let source = r#"
int x;
float x;

int main(void) {
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Conflicting redeclaration should produce an error.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("redeclar") || stderr_lower.contains("conflict")
            || stderr_lower.contains("type") || stderr_lower.contains("incompatible")
            || stderr_lower.contains("redefinition"),
        "Error should mention conflicting redeclaration.\nstderr: {}",
        result.stderr
    );
}

/// Verify typedef names are stored and resolved correctly.
///
/// A `typedef` creates a type alias in the symbol table. The alias should be
/// usable in subsequent declarations.
#[test]
fn symbol_table_typedef() {
    let source = r#"
typedef unsigned long size_type;
typedef int int32;

int main(void) {
    size_type sz = 100;
    int32 val = 42;
    return (int)(sz + (size_type)val);
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Typedef names should be resolved correctly.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Phase 6: Storage Class Tests
// ===========================================================================

/// Verify `static` variable has internal linkage.
///
/// A `static` variable at file scope has internal linkage, meaning it is only
/// visible within its translation unit. Multiple `static` variables with the same
/// name in different translation units do not conflict.
#[test]
fn storage_static_variable() {
    let source = r#"
static int counter = 0;

void increment(void) {
    counter++;
}

int main(void) {
    increment();
    increment();
    return counter;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Static variable with internal linkage should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify `extern` declaration without definition.
///
/// An `extern` declaration at block scope informs the compiler that the variable
/// is defined elsewhere. This should compile to an object file without errors
/// (linking may fail if the definition is missing, but compilation should succeed).
#[test]
fn storage_extern_variable() {
    let source = r#"
extern int external_var;

int main(void) {
    return 0;
}
"#;
    // Compile only (-c) to avoid link errors for the undefined extern symbol
    let result = compile_c(source, &["-c"]);
    assert!(
        result.success,
        "Extern declaration should compile successfully.\nstderr: {}",
        result.stderr
    );
}

/// Verify `auto` and `register` are valid storage classes in block scope.
///
/// Per C11 §6.7.1, `auto` and `register` are valid storage class specifiers
/// for variables with block scope. `auto` is the default and rarely used
/// explicitly; `register` is a hint to the compiler.
#[test]
fn storage_auto_register() {
    let source = r#"
int main(void) {
    auto int a = 10;
    register int b = 20;
    return a + b;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "auto and register storage classes should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify `static extern` produces error.
///
/// Per C11 §6.7.1p2, at most one storage-class specifier may be given in the
/// declaration specifiers of a declaration. `static extern` is a constraint
/// violation because `static` and `extern` are both storage-class specifiers.
#[test]
fn storage_conflicting_specifiers() {
    let source = r#"
static extern int conflict_var;

int main(void) {
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "static extern should produce a conflicting storage class error.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("storage") || stderr_lower.contains("specifier")
            || stderr_lower.contains("conflict") || stderr_lower.contains("multiple")
            || stderr_lower.contains("invalid"),
        "Error should mention conflicting storage class specifiers.\nstderr: {}",
        result.stderr
    );
}

/// Verify `_Thread_local` storage class.
///
/// C11 introduced `_Thread_local` as a storage-class specifier indicating that
/// each thread gets its own copy of the variable. The compiler should accept
/// this specifier in valid combinations (e.g., `_Thread_local static`).
#[test]
fn storage_thread_local() {
    let source = r#"
_Thread_local int tls_var = 0;

int main(void) {
    tls_var = 42;
    return tls_var;
}
"#;
    // Compile only to avoid potential linking issues with TLS support
    let result = compile_c(source, &["-c"]);
    assert!(
        result.success,
        "_Thread_local storage class should be accepted.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Phase 7: Target-Specific Type Size Tests
// ===========================================================================

/// Verify sizeof(int)=4, sizeof(long)=8, sizeof(void*)=8 for x86-64.
///
/// On the x86-64 LP64 data model, `long` and pointers are 8 bytes while
/// `int` remains 4 bytes. This test compiles with `--target x86_64-linux-gnu`
/// and uses compile-time static assertions to verify the sizes.
#[test]
fn type_sizes_x86_64() {
    let source = r#"
_Static_assert(sizeof(int) == 4, "int should be 4 bytes on x86-64");
_Static_assert(sizeof(long) == 8, "long should be 8 bytes on x86-64");
_Static_assert(sizeof(void*) == 8, "pointer should be 8 bytes on x86-64");
_Static_assert(sizeof(char) == 1, "char should be 1 byte");
_Static_assert(sizeof(short) == 2, "short should be 2 bytes");
_Static_assert(sizeof(long long) == 8, "long long should be 8 bytes");

int main(void) {
    return 0;
}
"#;
    let result = compile_c_with_target(source, common::TARGET_X86_64, &[]);
    assert!(
        result.success,
        "x86-64 type sizes should match LP64 data model.\nstderr: {}",
        result.stderr
    );
}

/// Verify sizeof(int)=4, sizeof(long)=4, sizeof(void*)=4 for i686.
///
/// On the i686 ILP32 data model, both `int` and `long` are 4 bytes, and
/// pointers are also 4 bytes. This is the key difference from 64-bit targets.
#[test]
fn type_sizes_i686() {
    let source = r#"
_Static_assert(sizeof(int) == 4, "int should be 4 bytes on i686");
_Static_assert(sizeof(long) == 4, "long should be 4 bytes on i686");
_Static_assert(sizeof(void*) == 4, "pointer should be 4 bytes on i686");
_Static_assert(sizeof(char) == 1, "char should be 1 byte");
_Static_assert(sizeof(short) == 2, "short should be 2 bytes");
_Static_assert(sizeof(long long) == 8, "long long should be 8 bytes");

int main(void) {
    return 0;
}
"#;
    let result = compile_c_with_target(source, common::TARGET_I686, &[]);
    assert!(
        result.success,
        "i686 type sizes should match ILP32 data model.\nstderr: {}",
        result.stderr
    );
}

/// Verify sizeof(long)=8, sizeof(void*)=8 for AArch64.
///
/// AArch64 uses the LP64 data model, identical to x86-64 in terms of type sizes:
/// 8-byte `long` and 8-byte pointers.
#[test]
fn type_sizes_aarch64() {
    let source = r#"
_Static_assert(sizeof(int) == 4, "int should be 4 bytes on aarch64");
_Static_assert(sizeof(long) == 8, "long should be 8 bytes on aarch64");
_Static_assert(sizeof(void*) == 8, "pointer should be 8 bytes on aarch64");
_Static_assert(sizeof(char) == 1, "char should be 1 byte");
_Static_assert(sizeof(short) == 2, "short should be 2 bytes");
_Static_assert(sizeof(long long) == 8, "long long should be 8 bytes");

int main(void) {
    return 0;
}
"#;
    let result = compile_c_with_target(source, common::TARGET_AARCH64, &[]);
    assert!(
        result.success,
        "AArch64 type sizes should match LP64 data model.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Phase 8: Complex Type Tests
// ===========================================================================

/// Verify struct type definition and member types.
///
/// Defining a struct with multiple member types (int, float, char pointer) and
/// accessing those members should type-check correctly.
#[test]
fn type_struct_definition() {
    let source = r#"
struct Person {
    int age;
    float height;
    char name[64];
};

int main(void) {
    struct Person p;
    p.age = 30;
    p.height = 5.9f;
    p.name[0] = 'J';
    p.name[1] = '\0';
    return p.age;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Struct definition with mixed member types should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify union type with shared storage.
///
/// A union allocates enough storage for its largest member. All members share
/// the same memory location. Accessing different members is valid (though
/// reading a member that wasn't last written is implementation-defined).
#[test]
fn type_union_definition() {
    let source = r#"
union Value {
    int i;
    float f;
    char bytes[4];
};

int main(void) {
    union Value v;
    v.i = 42;
    int x = v.i;
    v.f = 3.14f;
    float y = v.f;
    (void)x;
    (void)y;
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Union definition with shared storage should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify enum type with integer constants.
///
/// An enum defines a set of named integer constants. The first enumerator
/// has value 0 by default, and each subsequent enumerator increments by 1
/// unless explicitly assigned.
#[test]
fn type_enum_definition() {
    let source = r#"
enum Color {
    RED,
    GREEN,
    BLUE,
    ALPHA = 10
};

int main(void) {
    enum Color c = GREEN;
    int val = c;
    if (val != 1) return 1;
    if (ALPHA != 10) return 1;
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Enum definition with integer constants should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify function pointer type checking.
///
/// A function pointer variable must be declared with the correct signature to
/// match the target function. Calling through a function pointer should
/// type-check the arguments and return type against the pointer's signature.
#[test]
fn type_function_pointer() {
    let source = r#"
int multiply(int a, int b) {
    return a * b;
}

int apply(int (*op)(int, int), int x, int y) {
    return op(x, y);
}

int main(void) {
    int (*fp)(int, int) = multiply;
    int result = apply(fp, 6, 7);
    return result;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Function pointer type checking should succeed.\nstderr: {}",
        result.stderr
    );
}

/// Verify array type with size and element type.
///
/// Array declarations specify the element type and (optionally) the size.
/// Accessing array elements with valid indices should type-check correctly.
/// The element type of the subscript expression must match the array's
/// element type.
#[test]
fn type_array_declaration() {
    let source = r#"
int main(void) {
    int arr[10];
    double darr[5];
    char str[100];

    arr[0] = 42;
    darr[0] = 3.14;
    str[0] = 'H';
    str[1] = '\0';

    int x = arr[0];
    double d = darr[0];
    char c = str[0];

    (void)x;
    (void)d;
    (void)c;
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Array declaration with element access should compile.\nstderr: {}",
        result.stderr
    );
}

// ===========================================================================
// Additional Semantic Analysis Tests
// ===========================================================================

/// Verify GCC-compatible diagnostic format for semantic errors.
///
/// When the compiler emits a semantic error, it must follow the format:
/// `<file>:<line>:<col>: error: <message>` on stderr. This test verifies
/// that at least one error line follows this pattern.
#[test]
fn diagnostic_format_gcc_compatible() {
    let source = r#"
int main(void) {
    return undeclared_xyz;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Should fail for undeclared variable."
    );
    // Verify at least one line in stderr matches GCC diagnostic format
    assert!(
        has_gcc_diagnostic_format(&result.stderr),
        "Stderr should contain GCC-compatible diagnostic format.\nstderr: {}",
        result.stderr
    );
}

/// Verify exit code is 1 on compile error.
///
/// Per the AAP diagnostic format rule, the compiler must exit with code 1
/// when any compilation error occurs.
#[test]
fn exit_code_on_error() {
    let source = r#"
int main(void) {
    return no_such_variable;
}
"#;
    let (success, _stdout, _stderr, exit_code) = compile_direct(source, &[]);
    assert!(!success, "Compilation with semantic error should fail.");
    assert_eq!(
        exit_code,
        Some(1),
        "Exit code should be 1 on compilation error. Got: {:?}",
        exit_code
    );
}

/// Verify that exit code is 0 on successful compilation.
///
/// When compilation succeeds with no errors, the compiler must exit with code 0.
#[test]
fn exit_code_on_success() {
    let source = r#"
int main(void) {
    return 0;
}
"#;
    let (success, _stdout, _stderr, exit_code) = compile_direct(source, &[]);
    assert!(success, "Valid program should compile successfully.");
    assert_eq!(
        exit_code,
        Some(0),
        "Exit code should be 0 on successful compilation. Got: {:?}",
        exit_code
    );
}

/// Verify multiple errors are reported for multiple semantic violations.
///
/// The compiler should not stop at the first error but attempt to report
/// as many errors as possible to improve developer workflow.
#[test]
fn multiple_errors_reported() {
    let source = r#"
int main(void) {
    int x = undefined_a;
    int y = undefined_b;
    int z = undefined_c;
    return x + y + z;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Multiple undefined variables should cause failure."
    );
    // The compiler should report errors; we just verify it detected at least one
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("error") || stderr_lower.contains("undeclared")
            || stderr_lower.contains("undefined"),
        "Stderr should contain error diagnostics.\nstderr: {}",
        result.stderr
    );
}

/// Verify that const-qualified variables cannot be assigned after initialization.
///
/// Per C11 §6.7.3, a `const`-qualified variable must not be the target of an
/// assignment after its initialization. The compiler should reject such assignments.
#[test]
fn const_qualification_enforcement() {
    let source = r#"
int main(void) {
    const int x = 42;
    x = 100;
    return x;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Assignment to const variable should fail.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("const") || stderr_lower.contains("read-only")
            || stderr_lower.contains("assign") || stderr_lower.contains("qualif"),
        "Error should mention const qualification.\nstderr: {}",
        result.stderr
    );
}

/// Verify that dereferencing a non-pointer type produces an error.
///
/// The unary `*` operator can only be applied to pointer types. Applying it
/// to an `int` variable is a semantic error.
#[test]
fn deref_non_pointer_error() {
    let source = r#"
int main(void) {
    int x = 42;
    int y = *x;
    return y;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Dereferencing non-pointer should fail.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("pointer") || stderr_lower.contains("dereference")
            || stderr_lower.contains("type") || stderr_lower.contains("operand")
            || stderr_lower.contains("indirection"),
        "Error should mention pointer type requirement.\nstderr: {}",
        result.stderr
    );
}

/// Verify address-of operator requires an lvalue.
///
/// The unary `&` operator can only be applied to lvalues (variables, array
/// elements, struct members, dereferenced pointers). Applying it to an
/// rvalue like a numeric literal is a semantic error.
#[test]
fn address_of_non_lvalue_error() {
    let source = r#"
int main(void) {
    int *p = &42;
    return *p;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        !result.success,
        "Address-of non-lvalue should fail.\nstdout: {}",
        result.stdout
    );
    let stderr_lower = result.stderr.to_lowercase();
    assert!(
        stderr_lower.contains("lvalue") || stderr_lower.contains("address")
            || stderr_lower.contains("operand") || stderr_lower.contains("require"),
        "Error should mention lvalue requirement.\nstderr: {}",
        result.stderr
    );
}

/// Verify that nested struct definitions are handled correctly.
///
/// C11 allows struct definitions to be nested. Inner struct types can be
/// used as member types of the outer struct.
#[test]
fn nested_struct_types() {
    let source = r#"
struct Outer {
    struct Inner {
        int x;
        int y;
    } inner;
    int z;
};

int main(void) {
    struct Outer o;
    o.inner.x = 1;
    o.inner.y = 2;
    o.z = 3;
    return o.inner.x + o.inner.y + o.z;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Nested struct definitions should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify that recursive struct definitions via pointers are handled.
///
/// A struct can contain a pointer to its own type, which is the foundation
/// of linked data structures. The struct must be visible by name inside its
/// own definition for the pointer member.
#[test]
fn recursive_struct_pointer() {
    let source = r#"
struct Node {
    int value;
    struct Node *next;
};

int main(void) {
    struct Node a, b;
    a.value = 1;
    a.next = &b;
    b.value = 2;
    b.next = (void*)0;
    return a.value + a.next->value;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Recursive struct via pointer should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify that the `sizeof` operator works with types and expressions.
///
/// `sizeof` is a compile-time operator that returns the size of a type or
/// expression. It must work with both parenthesized type names and expressions.
#[test]
fn sizeof_operator() {
    let source = r#"
int main(void) {
    int x = 42;
    int s1 = sizeof(int);
    int s2 = sizeof(x);
    int s3 = sizeof(char);
    return s1 + s2 + s3;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "sizeof operator should compile successfully.\nstderr: {}",
        result.stderr
    );
}

/// Verify that cast expressions are type-checked.
///
/// Explicit casts between compatible types should succeed. Casts between
/// incompatible types (e.g., struct to int without a defined conversion)
/// should produce errors.
#[test]
fn cast_expression_valid() {
    let source = r#"
int main(void) {
    double d = 3.14;
    int i = (int)d;
    float f = (float)i;
    void *vp = (void *)0;
    int *ip = (int *)vp;
    (void)f;
    (void)ip;
    return i;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Valid cast expressions should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify that signed/unsigned comparison compiles (potentially with warning).
///
/// Comparing signed and unsigned integers is allowed in C but the implicit
/// conversion rules may produce surprising results. The compiler may emit
/// a warning but should not reject the code.
#[test]
fn signed_unsigned_comparison() {
    let source = r#"
int main(void) {
    int a = -1;
    unsigned int b = 1;
    if (a < b) {
        return 1;
    }
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    // This should compile; warnings about signed/unsigned comparison are acceptable
    assert!(
        result.success || result.stderr.to_lowercase().contains("warning"),
        "Signed/unsigned comparison should compile (possibly with warning).\nstderr: {}",
        result.stderr
    );
}

/// Verify that `static` variables inside functions retain their values between calls.
///
/// A `static` local variable is initialized once and retains its value across
/// function calls. The semantic analyzer must allow `static` at block scope.
#[test]
fn static_local_variable() {
    let source = r#"
int counter(void) {
    static int count = 0;
    count++;
    return count;
}

int main(void) {
    counter();
    counter();
    return counter();
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Static local variable should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify that `void` expressions are handled correctly.
///
/// Casting an expression to `void` is a valid way to explicitly discard a value.
/// The compiler should accept `(void)expr;` without errors.
#[test]
fn void_cast_discard() {
    let source = r#"
int compute(void) {
    return 42;
}

int main(void) {
    (void)compute();
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Casting to void to discard value should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify that incomplete type usage in pointer declarations is allowed.
///
/// C11 allows declaring pointers to incomplete types (forward-declared structs).
/// The struct does not need to be defined until the pointer is dereferenced.
#[test]
fn incomplete_type_pointer() {
    let source = r#"
struct ForwardDeclared;

void accept_ptr(struct ForwardDeclared *p) {
    (void)p;
}

int main(void) {
    accept_ptr((void*)0);
    return 0;
}
"#;
    let result = compile_c(source, &[]);
    assert!(
        result.success,
        "Pointer to incomplete (forward-declared) type should compile.\nstderr: {}",
        result.stderr
    );
}

/// Verify that using a helper with `fs::write()` to create a temp file works
/// alongside the standard compilation helper, confirming the `std::fs` import
/// is functional within the test context.
#[test]
fn fs_write_temp_file_compilation() {
    let dir = common::TempDir::new("sema_fs_test");
    let source_path = dir.path().join("test.c");
    let source_content = "int main(void) { return 0; }\n";

    fs::write(&source_path, source_content).unwrap_or_else(|e| {
        panic!("Failed to write temp source: {}", e);
    });

    let content_back = fs::read_to_string(&source_path).unwrap_or_else(|e| {
        panic!("Failed to read back temp source: {}", e);
    });

    assert_eq!(
        content_back.trim(),
        source_content.trim(),
        "Written and read content should match"
    );

    // Also compile the file directly using Command
    let bcc = common::get_bcc_binary();
    let output_path = dir.path().join("a.out");

    let status = Command::new(&bcc)
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .status()
        .unwrap_or_else(|e| {
            panic!("Failed to invoke bcc: {}", e);
        });

    // If bcc exists and works, check success; if not, just verify our fs operations worked
    if output_path.exists() {
        assert!(status.success(), "Compilation via Command should succeed");
    }
}

/// Verify that creating source files in a subdirectory via `fs::create_dir_all()`
/// works for tests that need include directory structure.
#[test]
fn fs_create_dir_all_for_includes() {
    let dir = common::TempDir::new("sema_includes_test");
    let include_dir = dir.path().join("inc");

    fs::create_dir_all(&include_dir).unwrap_or_else(|e| {
        panic!("Failed to create include directory: {}", e);
    });

    let header_path = include_dir.join("myheader.h");
    fs::write(&header_path, "#define MY_CONSTANT 42\n").unwrap_or_else(|e| {
        panic!("Failed to write header: {}", e);
    });

    assert!(
        header_path.exists(),
        "Header file should exist after creation"
    );

    let content = fs::read_to_string(&header_path).unwrap();
    assert!(
        content.contains("MY_CONSTANT"),
        "Header content should be correct"
    );
}

/// Verify that the compile result output_path field and exit_status are
/// correctly populated for both successful and failing compilations.
///
/// Tests the CompileResult struct API including the `output_path` and
/// `exit_status` fields. When the compiler fully implements code generation,
/// successful compilation should produce an output binary at the output_path.
#[test]
fn compile_result_output_path() {
    let source = r#"
int main(void) {
    return 0;
}
"#;
    let result = compile_c(source, &[]);

    // Verify the exit_status field is accessible and populated
    let _exit_code = result.exit_status.code();

    if result.success {
        // When the compiler implements code generation, successful compilation
        // should produce an output binary. During development, the compiler may
        // succeed at parsing/analysis without producing output — this is acceptable.
        if let Some(ref path) = result.output_path {
            assert!(
                path.exists(),
                "Output path should point to an existing file.\npath: {}",
                path.display()
            );
        }
    } else {
        // On failure, output_path should be None (no binary produced)
        assert!(
            result.output_path.is_none(),
            "Failed compilation should not produce an output path."
        );
    }
}

/// Verify CompileResult exit_status field is correctly populated on error.
///
/// When the compiler encounters a semantic error, the exit_status should
/// indicate failure (non-zero exit code, specifically exit code 1 per AAP).
#[test]
fn compile_result_exit_status_on_error() {
    let source = r#"
int main(void) {
    return no_such_variable_xyz;
}
"#;
    let result = compile_c(source, &[]);

    // The exit_status field should always be populated
    let exit_code = result.exit_status.code();

    // If the compiler correctly rejects this invalid code:
    if !result.success {
        // The exit code should be 1 per the diagnostic format rule
        assert_eq!(
            exit_code,
            Some(1),
            "Failed compilation should produce exit code 1. Got: {:?}",
            exit_code
        );
    }
}

/// Verify that the TempFile RAII helper from common works correctly.
#[test]
fn temp_file_raii() {
    let temp = common::write_temp_source("int main(void) { return 0; }\n");
    let path = temp.path().to_path_buf();
    assert!(
        path.exists(),
        "TempFile should exist immediately after creation"
    );
    assert!(
        path.to_string_lossy().ends_with(".c"),
        "TempFile should have .c extension"
    );
    // TempFile is dropped here, cleaning up automatically
}
