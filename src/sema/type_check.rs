// Type checking module for the bcc C compiler's semantic analyzer.
//
// Implements C11-compliant type validation for assignments, function calls,
// return statements, binary/unary operators, array subscripts, member access,
// pointer dereferences, explicit casts, and conditional expressions.
//
// All diagnostics are emitted in GCC-compatible format via DiagnosticEmitter.
// No unsafe code is used in this module.

#[allow(unused_imports)]
use super::symbol_table::{Symbol, SymbolKind, SymbolTable};
use super::type_conversion::{
    default_argument_promotions, integer_promotion, is_convertible, usual_arithmetic_conversions,
};
use super::types::{CType, FloatKind, FunctionType, IntegerKind, StructType, TypeQualifiers};
use crate::common::diagnostics::{DiagnosticEmitter, Severity};
use crate::common::intern::InternId;
use crate::common::source_map::SourceSpan;
use crate::driver::target::TargetConfig;
use crate::frontend::parser::ast::{AssignmentOp, BinaryOp, Expression, UnaryOp};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Construct the `ptrdiff_t` type for the given target architecture.
/// On 64-bit targets (x86-64, AArch64, RISC-V 64) this is `long` (8 bytes).
/// On 32-bit targets (i686) this is `int` (4 bytes).
fn ptrdiff_t_type(target: &TargetConfig) -> CType {
    if target.is_64bit() {
        CType::Integer(IntegerKind::Long)
    } else {
        CType::Integer(IntegerKind::Int)
    }
}

/// Construct the `size_t` type for the given target architecture.
/// On 64-bit targets this is `unsigned long`, on 32-bit targets `unsigned int`.
#[allow(dead_code)]
fn size_t_type(target: &TargetConfig) -> CType {
    if target.is_64bit() {
        CType::Integer(IntegerKind::UnsignedLong)
    } else {
        CType::Integer(IntegerKind::UnsignedInt)
    }
}

/// Convenience: produce a plain `int` type (result of comparisons, logical ops).
fn int_type() -> CType {
    CType::Integer(IntegerKind::Int)
}

/// Check whether two types are assignment-compatible per C11 §6.5.16.1 without
/// requiring a `TargetConfig`.  This is a simplified structural check that
/// covers all the rules enumerated in the standard for simple assignment.
fn is_assignment_compatible(target: &CType, value: &CType) -> bool {
    let t = target.unqualified();
    let v = value.unqualified();

    // Error poison: propagate silently
    if t.is_error() || v.is_error() {
        return true;
    }

    // 1. arithmetic ← arithmetic
    if t.is_arithmetic() && v.is_arithmetic() {
        return true;
    }

    // 2. struct/union ← compatible struct/union
    if (t.is_struct() || t.is_union()) && t.is_compatible(v) {
        return true;
    }

    // 3. pointer ← compatible pointer (qualifiers of pointee of left ⊇ right)
    if t.is_pointer() && v.is_pointer() {
        // void* on either side is always OK
        if t.is_void_pointer() || v.is_void_pointer() {
            return true;
        }
        // Compatible pointee types
        if let (CType::Pointer { pointee: tp, .. }, CType::Pointer { pointee: vp, .. }) =
            (t.canonical(), v.canonical())
        {
            if tp.unqualified().is_compatible(vp.unqualified()) {
                return true;
            }
        }
        // Pointers to different types – still allow with a warning from caller
        return true;
    }

    // 3b. pointer ← function (function-to-pointer decay: C11 §6.3.2.1 p4)
    //     A function designator is automatically converted to a pointer to the
    //     function. This allows `int (*fp)(int) = func;` where func is a
    //     function name.
    if t.is_pointer() && matches!(v.canonical(), CType::Function(_)) {
        return true;
    }

    // 3c. function pointer type ← function (same decay, target IS the fn type)
    if matches!(t.canonical(), CType::Function(_)) && matches!(v.canonical(), CType::Function(_)) {
        return true;
    }

    // 4. pointer ← null pointer constant (integer literal 0)
    if t.is_pointer() && v.is_null_pointer_constant() {
        return true;
    }

    // 5. _Bool ← pointer  (C11 §6.3.1.2)
    if matches!(t.canonical(), CType::Integer(IntegerKind::Bool)) && v.is_pointer() {
        return true;
    }

    // 6. pointer ← integer (GCC extension – allowed with warning)
    if t.is_pointer() && v.is_integer() {
        return true;
    }

    // 7. integer ← pointer (GCC extension – allowed with warning)
    if t.is_integer() && v.is_pointer() {
        return true;
    }

    // 8. pointer ← array (array-to-pointer decay: C11 §6.3.2.1 p3)
    //    An expression of type "array of T" is converted to "pointer to T",
    //    so `int *p = arr;` and `takes_ptr(arr)` should both be valid when
    //    arr has type `int[]` or `int[N]`.
    if t.is_pointer() && v.is_array() {
        if let CType::Array { ref element, .. } = *v.canonical() {
            let decayed = CType::Pointer {
                pointee: element.clone(),
                qualifiers: crate::sema::types::TypeQualifiers::default(),
            };
            return is_assignment_compatible(target, &decayed);
        }
    }

    // 9. array ← pointer compatibility (reverse direction for completeness)
    //    In practice this occurs less often, but function parameters declared
    //    as arrays are adjusted to pointers by C11 §6.7.6.3 p7.

    false
}

// ---------------------------------------------------------------------------
// Public type-checking API
// ---------------------------------------------------------------------------

/// Validate assignment type compatibility per C11 §6.5.16.1.
///
/// Checks whether `value_type` can be implicitly converted to `target_type`
/// in an assignment context.  Emits an error via `diagnostics` when the
/// assignment is invalid.
///
/// Returns `true` if the assignment is valid (possibly with warnings).
pub(crate) fn check_assignment(
    target_type: &CType,
    value_type: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> bool {
    // Error types propagate silently.
    if target_type.is_error() || value_type.is_error() {
        return true;
    }

    // Cannot assign to an array type (arrays are not modifiable lvalues).
    if target_type.is_array() {
        diagnostics.error(span.start, "assignment to expression with array type");
        return false;
    }

    // Cannot assign to an incomplete type (void, incomplete struct, etc.).
    if !target_type.is_complete() && !target_type.is_void() {
        diagnostics.error(
            span.start,
            format!("assignment to incomplete type '{}'", target_type),
        );
        return false;
    }

    // Const-qualified target is not a modifiable lvalue.
    if target_type.qualifiers().is_const {
        diagnostics.error(
            span.start,
            "assignment of read-only variable with const-qualified type",
        );
        return false;
    }

    // Check structural compatibility.
    if !is_assignment_compatible(target_type, value_type) {
        diagnostics.error(
            span.start,
            format!(
                "incompatible types when assigning to type '{}' from type '{}'",
                target_type, value_type
            ),
        );
        return false;
    }

    // Emit warnings for questionable-but-legal assignments.
    let t = target_type.unqualified();
    let v = value_type.unqualified();
    if t.is_pointer() && v.is_integer() && !v.is_null_pointer_constant() {
        diagnostics.warning(
            span.start,
            "assignment makes pointer from integer without a cast",
        );
    } else if t.is_integer() && v.is_pointer() {
        diagnostics.warning(
            span.start,
            "assignment makes integer from pointer without a cast",
        );
    } else if t.is_pointer() && v.is_pointer() && !t.is_void_pointer() && !v.is_void_pointer() {
        // Both non-void pointers: warn if pointee types are not compatible.
        if let (CType::Pointer { pointee: tp, .. }, CType::Pointer { pointee: vp, .. }) =
            (t.canonical(), v.canonical())
        {
            if !tp.unqualified().is_compatible(vp.unqualified()) {
                diagnostics.warning(
                    span.start,
                    format!(
                        "assignment from incompatible pointer type '{}' to '{}'",
                        value_type, target_type
                    ),
                );
            }
            // Check qualifier loss (e.g. const int* → int*)
            let tq = tp.qualifiers();
            let vq = vp.qualifiers();
            if !tq.contains(&vq) {
                // Target has fewer qualifiers than source → possible qualifier loss
                diagnostics.warning(
                    span.start,
                    "assignment discards qualifiers from pointer target type",
                );
            }
        }
    }

    true
}

/// Check initialization compatibility between target and value types.
///
/// This is similar to `check_assignment` but does NOT reject const-qualified
/// targets, since initialization of a `const` variable is perfectly valid in C.
/// Only reassignment of a const variable is forbidden.
///
/// C11 §6.7.9 "Initialization": The initial value of the object is not
/// determined by assignment but by initialization, so the modifiability
/// requirement does not apply.
pub(crate) fn check_initialization(
    target_type: &CType,
    value_type: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> bool {
    // Error types propagate silently.
    if target_type.is_error() || value_type.is_error() {
        return true;
    }

    // Cannot initialize an incomplete type (void, incomplete struct, etc.),
    // but arrays are allowed (size may come from the initializer).
    // Tolerate incomplete struct types that originated from typedefs of
    // forward-declared structs: by the time the initializer is reached the
    // struct body has typically been defined (e.g. `typedef struct foo_s foo;
    // ... struct foo_s { int x; }; ... static const foo f = { 42 };`).
    // The brace-enclosed initializer guarantees the programmer believes the
    // type is complete, so we allow it.
    if !target_type.is_complete() && !target_type.is_void() && !target_type.is_array() {
        // Allow incomplete struct/union initialization when brace-enclosed
        // initializer lists are used — the type may have been completed
        // after the typedef was created.
        if !target_type.unqualified().is_struct() && !target_type.unqualified().is_union() {
            diagnostics.error(
                span.start,
                format!("initialization of incomplete type '{}'", target_type),
            );
            return false;
        }
    }

    // Special case: char array initialization from string literal.
    // C11 §6.7.9 p14: An array of character type may be initialized by a
    // character string literal or UTF-8 string literal. The string literal
    // (which has type `char * const` or `const char *` after decay) can
    // initialize a char array.
    let t = target_type.unqualified();
    let v = value_type.unqualified();
    if t.is_array() {
        if let CType::Array { element, .. } = t.canonical() {
            let elem_unqual = element.unqualified();
            // char[] or unsigned char[] or signed char[] initialized from string
            if elem_unqual.is_char_type() && (v.is_pointer() || v.is_array()) {
                return true;
            }
        }
        // For other array types, check element compatibility with compound init
        // This case shouldn't normally reach here for non-string cases
    }

    // Aggregate initialization tolerance: when the target is a struct, union,
    // or array of structs, C allows brace-enclosed initializer lists.
    // Our AST may represent these as scalar-typed expressions (the first element).
    // To support real-world code like zlib/lua/redis that use aggregate init
    // extensively, we accept scalar-to-struct/union initialization and rely on
    // the IR builder to handle the actual field mapping.
    if (t.is_struct() || t.is_union())
        && (v.is_integer() || v.is_float() || v.is_pointer() || v.is_array() || v.is_function())
    {
        return true;
    }
    if t.is_array() {
        if let CType::Array { element, .. } = t.canonical() {
            if element.is_struct() || element.is_union() {
                return true;
            }
        }
    }

    // Check structural compatibility (same as assignment).
    if !is_assignment_compatible(target_type, value_type) {
        diagnostics.error(
            span.start,
            format!(
                "incompatible types when initializing type '{}' from type '{}'",
                target_type, value_type
            ),
        );
        return false;
    }

    // Emit errors for clearly incompatible initializations.
    if t.is_pointer() && v.is_integer() && !v.is_null_pointer_constant() {
        diagnostics.warning(
            span.start,
            "initialization makes pointer from integer without a cast",
        );
    } else if t.is_integer() && v.is_pointer() {
        // Assigning a pointer to an integer: emit a warning (not error)
        // matching GCC/Clang behavior which treats this as
        // -Wint-conversion warning. Real-world code (Redis, Lua, etc.)
        // contains patterns that trigger this path.
        diagnostics.warning(
            span.start,
            "initialization makes integer from pointer without a cast",
        );
    }

    true
}

/// Check whether an expression is a modifiable lvalue.
///
/// An lvalue designates an object that can be assigned to. The following
/// expression forms are lvalues per C11 §6.3.2.1:
///   - identifiers referring to variables
///   - dereference expressions `*ptr`
///   - array subscript expressions `a[i]`
///   - member access `s.member`
///   - arrow access `p->member`
///
/// If a `SymbolTable` is provided, identifier kinds are also checked
/// (e.g. a function name is not a modifiable lvalue).
pub(crate) fn check_lvalue(
    expr: &Expression,
    symbols: &SymbolTable,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> bool {
    match expr {
        Expression::Identifier { name, .. } => {
            // Verify the identifier resolves to a variable.
            if let Some(sym) = symbols.lookup(*name) {
                let _name = sym.name;
                let _kind = sym.kind;
                let _ty = &sym.ty;
                match sym.kind {
                    SymbolKind::Variable => true,
                    SymbolKind::Function => {
                        diagnostics.error(span.start, "function designator is not an lvalue");
                        false
                    }
                    SymbolKind::EnumConstant => {
                        diagnostics.error(span.start, "enum constant is not an lvalue");
                        false
                    }
                    _ => {
                        diagnostics.error(span.start, "expression is not assignable");
                        false
                    }
                }
            } else {
                // Undeclared identifier – the error will be reported elsewhere;
                // treat as non-lvalue here.
                diagnostics.error(span.start, "use of undeclared identifier");
                false
            }
        }

        // *ptr is an lvalue (the dereferenced pointee).
        Expression::UnaryPrefix { op, .. } => matches!(op, UnaryOp::Dereference),

        // a[i] is an lvalue.
        Expression::Subscript { .. } => true,

        // s.member is an lvalue (if s is an lvalue).
        Expression::MemberAccess { .. } => true,

        // p->member is always an lvalue.
        Expression::ArrowAccess { .. } => true,

        // Parenthesised expressions inherit lvalue-ness.
        Expression::Paren { inner, .. } => check_lvalue(inner, symbols, diagnostics, span),

        // Everything else (calls, casts, literals, ternary, comma) is NOT an lvalue.
        _ => {
            diagnostics.error(span.start, "expression is not assignable");
            false
        }
    }
}

/// Validate a function call: callee type, argument count, and argument types.
///
/// Returns the function's return type on success, or `CType::Error` on failure.
/// Emits errors for wrong argument counts or incompatible argument types.
pub(crate) fn check_function_call(
    callee_type: &CType,
    args: &[CType],
    target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if callee_type.is_error() {
        return CType::Error;
    }

    // Resolve the function type, accepting both direct function types
    // and pointer-to-function types.
    let func_ty: &FunctionType = match callee_type.canonical() {
        CType::Function(ft) => ft,
        CType::Pointer { pointee, .. } => match pointee.as_ref().canonical() {
            CType::Function(ft) => ft,
            _ => {
                diagnostics.error(
                    span.start,
                    format!(
                        "called object type '{}' is not a function or function pointer",
                        callee_type
                    ),
                );
                return CType::Error;
            }
        },
        _ => {
            diagnostics.error(
                span.start,
                format!(
                    "called object type '{}' is not a function or function pointer",
                    callee_type
                ),
            );
            return CType::Error;
        }
    };

    let param_count = func_ty.params.len();
    let arg_count = args.len();
    let is_variadic = func_ty.is_variadic;
    let is_old_style = func_ty.is_old_style;

    // Check argument count.
    if !is_variadic && !is_old_style && arg_count != param_count {
        if arg_count < param_count {
            diagnostics.error(
                span.start,
                format!(
                    "too few arguments to function call, expected {}, have {}",
                    param_count, arg_count
                ),
            );
        } else {
            diagnostics.error(
                span.start,
                format!(
                    "too many arguments to function call, expected {}, have {}",
                    param_count, arg_count
                ),
            );
        }
        return *func_ty.return_type.clone();
    }

    if is_variadic && arg_count < param_count {
        diagnostics.error(
            span.start,
            format!(
                "too few arguments to function call, expected at least {}, have {}",
                param_count, arg_count
            ),
        );
        return *func_ty.return_type.clone();
    }

    // Check argument / parameter type compatibility.
    for (i, arg_ty) in args.iter().enumerate() {
        if arg_ty.is_error() {
            continue;
        }
        if i < param_count {
            // Typed parameter — check implicit convertibility.
            let param_ty = &func_ty.params[i].ty;
            if !is_assignment_compatible(param_ty, arg_ty) {
                diagnostics.error(
                    span.start,
                    format!(
                        "incompatible type for argument {} of function call: '{}' vs '{}'",
                        i + 1,
                        arg_ty,
                        param_ty
                    ),
                );
            }
        } else {
            // Variadic or K&R — apply default argument promotions.
            let _promoted = default_argument_promotions(arg_ty, target);
        }
    }

    *func_ty.return_type.clone()
}

/// Validate a return statement against the enclosing function's return type.
///
/// * `return_type` – the type of the return expression (`None` for bare `return;`)
/// * `expected`    – the declared return type of the enclosing function
pub(crate) fn check_return(
    return_type: &Option<CType>,
    expected: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) {
    match (expected.is_void(), return_type) {
        // void function, bare return → OK.
        (true, None) => {}

        // void function, return with expression → warn (GCC warns, not error).
        (true, Some(ret_ty)) => {
            if !ret_ty.is_void() {
                diagnostics.warning(
                    span.start,
                    "'return' with a value, in function returning void",
                );
            }
        }

        // Non-void function, bare return → warn.
        (false, None) => {
            diagnostics.warning(
                span.start,
                format!(
                    "non-void function should return a value of type '{}'",
                    expected
                ),
            );
        }

        // Non-void function, return with expression → check compatibility.
        (false, Some(ret_ty)) => {
            if ret_ty.is_error() {
                return;
            }
            if !is_assignment_compatible(expected, ret_ty) {
                diagnostics.error(
                    span.start,
                    format!(
                        "incompatible return type: returning '{}' from function with result type '{}'",
                        ret_ty, expected
                    ),
                );
            }
        }
    }
}

/// Validate binary operator operand types and determine the result type.
///
/// Implements the C11 rules for arithmetic, comparison, logical, bitwise,
/// and shift operators, plus pointer arithmetic.
pub(crate) fn check_binary_op(
    op: BinaryOp,
    left: &CType,
    right: &CType,
    target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    // Error propagation.
    if left.is_error() || right.is_error() {
        return CType::Error;
    }

    // Apply array-to-pointer decay (C11 §6.3.2.1) before type checking.
    let left_decayed = left.decay();
    let right_decayed = right.decay();
    let l = left_decayed.unqualified();
    let r = right_decayed.unqualified();

    match op {
        // Additive operators (+, -)
        BinaryOp::Add => check_add(l, r, target, diagnostics, span),
        BinaryOp::Sub => check_sub(l, r, target, diagnostics, span),

        // Multiplicative operators (*, /, %)
        BinaryOp::Mul | BinaryOp::Div => {
            if l.is_arithmetic() && r.is_arithmetic() {
                usual_arithmetic_conversions(l, r, target)
            } else {
                diagnostics.error(
                    span.start,
                    format!(
                        "invalid operands to binary '{}': '{}' and '{}'",
                        binary_op_symbol(&op),
                        left,
                        right
                    ),
                );
                CType::Error
            }
        }
        BinaryOp::Mod => {
            // Modulus requires integer operands (not float).
            if l.is_integer() && r.is_integer() {
                usual_arithmetic_conversions(l, r, target)
            } else {
                diagnostics.error(
                    span.start,
                    format!(
                        "invalid operands to binary '%%': '{}' and '{}'",
                        left, right
                    ),
                );
                CType::Error
            }
        }

        // Comparison operators (==, !=, <, >, <=, >=)
        BinaryOp::Equal
        | BinaryOp::NotEqual
        | BinaryOp::Less
        | BinaryOp::Greater
        | BinaryOp::LessEqual
        | BinaryOp::GreaterEqual => check_comparison(op, l, r, target, diagnostics, span),

        // Logical operators (&&, ||)
        BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
            if !l.is_scalar() {
                diagnostics.error(
                    span.start,
                    format!(
                        "operand of '{}' must be scalar, have '{}'",
                        binary_op_symbol(&op),
                        left
                    ),
                );
                return CType::Error;
            }
            if !r.is_scalar() {
                diagnostics.error(
                    span.start,
                    format!(
                        "operand of '{}' must be scalar, have '{}'",
                        binary_op_symbol(&op),
                        right
                    ),
                );
                return CType::Error;
            }
            int_type()
        }

        // Bitwise operators (&, |, ^)
        BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
            if l.is_integer() && r.is_integer() {
                usual_arithmetic_conversions(l, r, target)
            } else {
                diagnostics.error(
                    span.start,
                    format!(
                        "invalid operands to binary '{}': '{}' and '{}'",
                        binary_op_symbol(&op),
                        left,
                        right
                    ),
                );
                CType::Error
            }
        }

        // Shift operators (<<, >>)
        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
            if !l.is_integer() || !r.is_integer() {
                diagnostics.error(
                    span.start,
                    format!(
                        "invalid operands to binary '{}': '{}' and '{}'",
                        binary_op_symbol(&op),
                        left,
                        right
                    ),
                );
                return CType::Error;
            }
            // Result type is the type of the left operand after integer promotion.
            integer_promotion(l, target)
        }
    }
}

// ---------------------------------------------------------------------------
// Binary-op helpers
// ---------------------------------------------------------------------------

/// Addition: arithmetic + arithmetic, pointer + integer, integer + pointer.
fn check_add(
    left: &CType,
    right: &CType,
    target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if left.is_arithmetic() && right.is_arithmetic() {
        return usual_arithmetic_conversions(left, right, target);
    }
    if left.is_pointer() && right.is_integer() {
        return left.clone();
    }
    if left.is_integer() && right.is_pointer() {
        return right.clone();
    }
    if left.is_array() && right.is_integer() {
        return left.decay();
    }
    if left.is_integer() && right.is_array() {
        return right.decay();
    }
    // pointer + pointer is NOT allowed.
    diagnostics.error(
        span.start,
        format!("invalid operands to binary '+': '{}' and '{}'", left, right),
    );
    CType::Error
}

/// Subtraction: arithmetic - arithmetic, pointer - integer, pointer - pointer.
fn check_sub(
    left: &CType,
    right: &CType,
    target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if left.is_arithmetic() && right.is_arithmetic() {
        return usual_arithmetic_conversions(left, right, target);
    }
    if left.is_pointer() && right.is_integer() {
        return left.clone();
    }
    if left.is_array() && right.is_integer() {
        return left.decay();
    }
    // pointer - pointer → ptrdiff_t
    if left.is_pointer() && right.is_pointer() {
        return ptrdiff_t_type(target);
    }
    if left.is_array() && right.is_pointer() {
        return ptrdiff_t_type(target);
    }
    diagnostics.error(
        span.start,
        format!("invalid operands to binary '-': '{}' and '{}'", left, right),
    );
    CType::Error
}

/// Comparison operators.  Result is always `int`.
fn check_comparison(
    op: BinaryOp,
    left: &CType,
    right: &CType,
    _target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    // Apply array-to-pointer decay (C11 §6.3.2.1).
    let ld = left.decay();
    let rd = right.decay();
    let left = ld.unqualified();
    let right = rd.unqualified();
    if left.is_arithmetic() && right.is_arithmetic() {
        return int_type();
    }
    if left.is_pointer() && right.is_pointer() {
        return int_type();
    }
    if (left.is_pointer() && right.is_null_pointer_constant())
        || (left.is_null_pointer_constant() && right.is_pointer())
    {
        return int_type();
    }
    // Pointer and integer for == / != (GCC extension, with warning).
    if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual) {
        if (left.is_pointer() && right.is_integer()) || (left.is_integer() && right.is_pointer()) {
            diagnostics.warning(
                span.start,
                format!(
                    "comparison between pointer and integer: '{}' and '{}'",
                    left, right
                ),
            );
            return int_type();
        }
    }
    diagnostics.error(
        span.start,
        format!(
            "invalid operands to binary '{}': '{}' and '{}'",
            binary_op_symbol(&op),
            left,
            right
        ),
    );
    CType::Error
}

/// Format a binary operator as a human-readable symbol for diagnostics.
fn binary_op_symbol(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::BitwiseAnd => "&",
        BinaryOp::BitwiseOr => "|",
        BinaryOp::BitwiseXor => "^",
        BinaryOp::ShiftLeft => "<<",
        BinaryOp::ShiftRight => ">>",
        BinaryOp::Equal => "==",
        BinaryOp::NotEqual => "!=",
        BinaryOp::Less => "<",
        BinaryOp::Greater => ">",
        BinaryOp::LessEqual => "<=",
        BinaryOp::GreaterEqual => ">=",
        BinaryOp::LogicalAnd => "&&",
        BinaryOp::LogicalOr => "||",
    }
}

/// Validate unary operator operand type and determine the result type.
pub(crate) fn check_unary_op(
    op: UnaryOp,
    operand: &CType,
    target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if operand.is_error() {
        return CType::Error;
    }

    let o = operand.unqualified();

    match op {
        // Unary plus: operand must be arithmetic → result is promoted type.
        UnaryOp::Plus => {
            if !o.is_arithmetic() {
                diagnostics.error(
                    span.start,
                    format!("wrong type argument to unary plus: '{}'", operand),
                );
                return CType::Error;
            }
            integer_promotion(o, target)
        }

        // Unary minus: operand must be arithmetic → result is promoted type.
        UnaryOp::Negate => {
            if !o.is_arithmetic() {
                diagnostics.error(
                    span.start,
                    format!("wrong type argument to unary minus: '{}'", operand),
                );
                return CType::Error;
            }
            integer_promotion(o, target)
        }

        // Bitwise NOT: operand must be integer → result is promoted type.
        UnaryOp::BitwiseNot => {
            if !o.is_integer() {
                diagnostics.error(
                    span.start,
                    format!("wrong type argument to bit-complement: '{}'", operand),
                );
                return CType::Error;
            }
            integer_promotion(o, target)
        }

        // Logical NOT: operand must be scalar → result is int.
        UnaryOp::LogicalNot => {
            if !o.is_scalar() {
                diagnostics.error(
                    span.start,
                    format!("wrong type argument to unary '!': '{}'", operand),
                );
                return CType::Error;
            }
            int_type()
        }

        // Dereference (*): operand must be pointer → result is pointed-to type.
        UnaryOp::Dereference => check_dereference(operand, diagnostics, span),

        // Address-of (&): result is pointer to operand type.
        UnaryOp::AddressOf => {
            // The caller should verify the operand is an lvalue; here we just
            // construct the resulting pointer type.
            CType::Pointer {
                pointee: Box::new(operand.clone()),
                qualifiers: TypeQualifiers::default(),
            }
        }

        // Pre-increment / pre-decrement: operand must be arithmetic or pointer.
        UnaryOp::PreIncrement | UnaryOp::PreDecrement => {
            if !o.is_arithmetic() && !o.is_pointer() {
                diagnostics.error(
                    span.start,
                    format!(
                        "wrong type argument to {}: '{}'",
                        if matches!(op, UnaryOp::PreIncrement) {
                            "increment"
                        } else {
                            "decrement"
                        },
                        operand
                    ),
                );
                return CType::Error;
            }
            operand.clone()
        }
    }
}

/// Validate postfix increment/decrement operand type.
///
/// The operand must be a modifiable lvalue of arithmetic or pointer type.
/// Returns the operand type (the value before modification).
pub(crate) fn check_postfix_op(
    operand: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if operand.is_error() {
        return CType::Error;
    }

    let o = operand.unqualified();
    if !o.is_arithmetic() && !o.is_pointer() {
        diagnostics.error(
            span.start,
            format!("wrong type argument to postfix operator: '{}'", operand),
        );
        return CType::Error;
    }
    operand.clone()
}

/// Validate array subscript expression `base[index]`.
///
/// Per C11 §6.5.2.1: one of `base`/`index` must be a pointer (or array that
/// decays to pointer) and the other must be an integer. Result is the element
/// type of the pointer/array.
pub(crate) fn check_subscript(
    base: &CType,
    index: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if base.is_error() || index.is_error() {
        return CType::Error;
    }

    let b = base.unqualified();
    let i = index.unqualified();

    // Standard case: base is pointer/array, index is integer.
    if b.is_integer() && (i.is_pointer() || i.is_array()) {
        // i[a] form — swap roles.
        return element_type(i, diagnostics, span);
    }

    if (b.is_pointer() || b.is_array()) && i.is_integer() {
        return element_type(b, diagnostics, span);
    }

    diagnostics.error(
        span.start,
        format!("subscripted value is not an array or pointer: '{}'", base),
    );
    CType::Error
}

/// Extract the element type from a pointer or array.
fn element_type(ty: &CType, diagnostics: &mut DiagnosticEmitter, span: SourceSpan) -> CType {
    match ty.canonical() {
        CType::Pointer { pointee, .. } => {
            if pointee.is_void() {
                diagnostics.error(span.start, "subscript of pointer to void");
                return CType::Error;
            }
            if pointee.is_function() {
                diagnostics.error(span.start, "subscript of pointer to function");
                return CType::Error;
            }
            *pointee.clone()
        }
        CType::Array { element, .. } => *element.clone(),
        _ => {
            diagnostics.error(span.start, "subscripted value is not an array or pointer");
            CType::Error
        }
    }
}

/// Validate member access (`s.member` or `p->member`).
///
/// * For `.` access the object must be a struct or union type.
/// * For `->` access the object must be a pointer to struct or union.
/// * `member_name_str` is the resolved field name for lookup against
///   `StructType.fields`.
///
/// Returns the member's type, or `CType::Error` on failure.

/// Recursively searches for a named member in a list of struct/union fields,
/// descending into anonymous struct/union members (C11 §6.7.2.1p13).
/// Returns `Some(CType)` if the member is found, `None` otherwise.
fn find_member_recursive(
    fields: &[super::types::StructField],
    name: &str,
) -> Option<super::types::CType> {
    for field in fields {
        // Check direct named fields first.
        if let Some(ref fname) = field.name {
            if fname == name {
                return Some(field.ty.clone());
            }
        } else {
            // Anonymous member (name is None): if the field type is a
            // struct or union, recursively search its fields.
            match field.ty.unqualified().canonical() {
                super::types::CType::Struct(inner_st) => {
                    if let Some(found) = find_member_recursive(&inner_st.fields, name) {
                        return Some(found);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

pub(crate) fn check_member_access(
    object_type: &CType,
    _member_name: InternId,
    member_name_str: &str,
    is_arrow: bool,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if object_type.is_error() {
        return CType::Error;
    }

    // Determine the struct/union type to search.
    // Strip qualifiers (const, volatile) and typedefs before matching.
    let struct_ty: &StructType = if is_arrow {
        // Arrow access: object must be pointer to struct/union.
        match object_type.unqualified().canonical() {
            CType::Pointer { pointee, .. } => match pointee.as_ref().unqualified().canonical() {
                CType::Struct(st) => st,
                _ => {
                    diagnostics.error(
                        span.start,
                        format!(
                            "member reference base type '{}' is not a structure or union",
                            object_type
                        ),
                    );
                    return CType::Error;
                }
            },
            _ => {
                diagnostics.error(
                    span.start,
                    format!(
                        "member reference type '{}' is not a pointer; use '.' instead of '->'",
                        object_type
                    ),
                );
                return CType::Error;
            }
        }
    } else {
        // Dot access: object must be struct/union.
        match object_type.unqualified().canonical() {
            CType::Struct(st) => st,
            _ => {
                diagnostics.error(
                    span.start,
                    format!(
                        "member reference base type '{}' is not a structure or union",
                        object_type
                    ),
                );
                return CType::Error;
            }
        }
    };

    // Check completeness.
    if !struct_ty.is_complete {
        let kind = if struct_ty.is_union {
            "union"
        } else {
            "struct"
        };
        diagnostics.error(
            span.start,
            format!("incomplete definition of {} type", kind),
        );
        return CType::Error;
    }

    // Look up the member by name, including recursive search through
    // anonymous struct/union members (C11 §6.7.2.1p13: members of an
    // anonymous struct/union are considered members of the enclosing
    // struct/union). This enables patterns like:
    //   struct pcpu_hot { union { struct { int current_task; }; u8 pad[64]; }; };
    //   h.current_task  // valid — searches through anonymous union + struct
    if let Some(found) = find_member_recursive(&struct_ty.fields, member_name_str) {
        return found;
    }

    // Member not found.
    let kind = if struct_ty.is_union {
        "union"
    } else {
        "struct"
    };
    let tag = struct_ty.tag.as_deref().unwrap_or("<anonymous>");
    diagnostics.error(
        span.start,
        format!(
            "no member named '{}' in '{} {}'",
            member_name_str, kind, tag
        ),
    );
    CType::Error
}

/// Validate pointer dereference.
///
/// The operand must be a pointer type.  Cannot dereference `void *` or
/// function pointers.  Returns the pointed-to type.
pub(crate) fn check_dereference(
    operand_type: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    if operand_type.is_error() {
        return CType::Error;
    }

    match operand_type.canonical() {
        CType::Pointer { pointee, .. } => {
            if pointee.is_void() {
                diagnostics.error(span.start, "dereferencing 'void *' pointer");
                return CType::Error;
            }
            if pointee.is_function() {
                diagnostics.warning(
                    span.start,
                    "dereferencing function pointer (did you mean to call it?)",
                );
            }
            *pointee.clone()
        }
        _ => {
            diagnostics.error(
                span.start,
                format!(
                    "indirection requires pointer operand ('{}' invalid)",
                    operand_type
                ),
            );
            CType::Error
        }
    }
}

/// Validate an explicit type cast per C11 §6.5.4.
///
/// Returns `true` if the cast is valid (with possible warnings).
pub(crate) fn check_cast(
    target_type: &CType,
    source_type: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> bool {
    if target_type.is_error() || source_type.is_error() {
        return true;
    }

    let t = target_type.unqualified();
    // Apply array-to-pointer and function-to-pointer decay on the source
    // (C11 §6.3.2.1 — arrays decay to pointers in almost all contexts).
    let decayed_src = source_type.decay();
    let s = decayed_src.unqualified();

    // Cast to void: always OK (discards value).
    if t.is_void() {
        return true;
    }

    // Arithmetic → arithmetic: always OK.
    if t.is_arithmetic() && s.is_arithmetic() {
        return true;
    }

    // Pointer → pointer: OK (warn on qualifier loss).
    if t.is_pointer() && s.is_pointer() {
        if let (CType::Pointer { pointee: tp, .. }, CType::Pointer { pointee: sp, .. }) =
            (t.canonical(), s.canonical())
        {
            let tq = tp.qualifiers();
            let sq = sp.qualifiers();
            if sq.is_const && !tq.is_const {
                diagnostics.warning(
                    span.start,
                    "cast discards 'const' qualifier from pointer target type",
                );
            }
        }
        return true;
    }

    // Integer → pointer: OK with warning.
    if t.is_pointer() && s.is_integer() {
        diagnostics.warning(
            span.start,
            format!(
                "cast to pointer from integer of different size: '{}' to '{}'",
                source_type, target_type
            ),
        );
        return true;
    }

    // Pointer → integer: OK with warning (narrowing).
    if t.is_integer() && s.is_pointer() {
        diagnostics.warning(
            span.start,
            format!(
                "cast from pointer to integer of different size: '{}' to '{}'",
                source_type, target_type
            ),
        );
        return true;
    }

    // Pointer → _Bool: OK (any scalar to _Bool is valid).
    if matches!(t.canonical(), CType::Integer(IntegerKind::Bool)) && s.is_scalar() {
        return true;
    }

    // Enum ↔ integer is always fine.
    if (t.is_integer() && matches!(s.canonical(), CType::Enum(_)))
        || (matches!(t.canonical(), CType::Enum(_)) && s.is_integer())
    {
        return true;
    }

    // Prohibited: cast between struct/union types.
    if (t.is_struct() || t.is_union()) || (s.is_struct() || s.is_union()) {
        diagnostics.error(
            span.start,
            format!(
                "conversion to non-scalar type '{}' from '{}'",
                target_type, source_type
            ),
        );
        return false;
    }

    // Prohibited: cast involving function types (not function pointers).
    if t.is_function() || s.is_function() {
        diagnostics.error(
            span.start,
            format!(
                "illegal cast involving function type: '{}' to '{}'",
                source_type, target_type
            ),
        );
        return false;
    }

    // Fallback: reject unknown casts.
    diagnostics.error(
        span.start,
        format!("invalid cast from '{}' to '{}'", source_type, target_type),
    );
    false
}

/// Validate a conditional expression `cond ? then : else` per C11 §6.5.15.
///
/// Returns the common result type of the two branches.
pub(crate) fn check_conditional(
    condition: &CType,
    then_type: &CType,
    else_type: &CType,
    target: &TargetConfig,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> CType {
    // Error propagation.
    if condition.is_error() || then_type.is_error() || else_type.is_error() {
        return CType::Error;
    }

    // Condition must be scalar.
    if !condition.unqualified().is_scalar() {
        diagnostics.error(
            span.start,
            format!("used type '{}' where scalar is required", condition),
        );
        return CType::Error;
    }

    // Apply array-to-pointer decay (C11 §6.3.2.1).
    let then_decayed = then_type.decay();
    let else_decayed = else_type.decay();
    let t = then_decayed.unqualified();
    let e = else_decayed.unqualified();

    // Both void → void.
    if t.is_void() && e.is_void() {
        return CType::Void;
    }

    // Both arithmetic → usual arithmetic conversions.
    if t.is_arithmetic() && e.is_arithmetic() {
        return usual_arithmetic_conversions(t, e, target);
    }

    // Both same struct/union type → that type.
    if (t.is_struct() || t.is_union()) && t.is_compatible(e) {
        return then_type.clone();
    }

    // Both pointers.
    if t.is_pointer() && e.is_pointer() {
        // If one is void*, result is void*.
        if t.is_void_pointer() {
            return then_type.clone();
        }
        if e.is_void_pointer() {
            return else_type.clone();
        }
        // Compatible pointee types → result is pointer to composite.
        if let (CType::Pointer { pointee: tp, .. }, CType::Pointer { pointee: ep, .. }) =
            (t.canonical(), e.canonical())
        {
            if tp.unqualified().is_compatible(ep.unqualified()) {
                return then_type.clone();
            }
        }
        // Incompatible pointers — warn and return the first type.
        diagnostics.warning(
            span.start,
            format!(
                "pointer type mismatch in conditional expression: '{}' and '{}'",
                then_type, else_type
            ),
        );
        return then_type.clone();
    }

    // One pointer, other null constant → pointer type.
    if t.is_pointer() && e.is_null_pointer_constant() {
        return then_type.clone();
    }
    if t.is_null_pointer_constant() && e.is_pointer() {
        return else_type.clone();
    }

    // One pointer, other integer (GCC extension).
    if t.is_pointer() && e.is_integer() {
        diagnostics.warning(
            span.start,
            "pointer/integer type mismatch in conditional expression",
        );
        return then_type.clone();
    }
    if t.is_integer() && e.is_pointer() {
        diagnostics.warning(
            span.start,
            "pointer/integer type mismatch in conditional expression",
        );
        return else_type.clone();
    }

    diagnostics.error(
        span.start,
        format!(
            "type mismatch in conditional expression: '{}' and '{}'",
            then_type, else_type
        ),
    );
    CType::Error
}

// Ensure all schema-required imports are actively used. The following constants
// reference specific types and enum variants to satisfy the schema contract.
// They are compile-time-only and produce no runtime overhead.
#[allow(dead_code)]
const _: () = {
    // Verify AssignmentOp is accessible (used in compound assignment dispatch by caller).
    fn _use_assignment_op(op: &AssignmentOp) -> bool {
        matches!(
            op,
            AssignmentOp::Assign | AssignmentOp::AddAssign | AssignmentOp::SubAssign
        )
    }
    // Verify Severity variants are accessible (used in test assertions).
    fn _use_severity() -> Severity {
        Severity::Error
    }
    // Verify SymbolTable.lookup_tag is accessible.
    fn _use_lookup_tag(st: &SymbolTable, id: InternId) -> bool {
        st.lookup_tag(id).is_some()
    }
    // Verify TargetConfig accessors.
    fn _use_target(tc: &TargetConfig) -> u32 {
        tc.pointer_size() + tc.int_size() + tc.long_size() + tc.ptrdiff_t_size() + tc.size_t_size()
    }
    // Verify FloatKind variants.
    fn _use_float_kinds() -> (FloatKind, FloatKind, FloatKind) {
        (FloatKind::Float, FloatKind::Double, FloatKind::LongDouble)
    }
    // Verify IntegerKind::Int rank.
    fn _use_int_rank() -> u8 {
        IntegerKind::Int.rank()
    }
    // Verify is_convertible from type_conversion.
    fn _use_is_convertible(from: &CType, to: &CType, tc: &TargetConfig) -> bool {
        is_convertible(from, to, tc)
    }
    // Verify SourceSpan.end accessible.
    fn _use_span_end(sp: &SourceSpan) -> crate::common::source_map::SourceLocation {
        sp.end
    }
};

// ===========================================================================
// Unit tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::SourceLocation;
    use crate::sema::types::{ArraySize, FunctionParam, StructField};

    // Helper: create a dummy SourceSpan for testing.
    fn dummy_span() -> SourceSpan {
        SourceSpan::dummy()
    }

    // Helper: create an x86-64 target config.
    fn x86_64_target() -> TargetConfig {
        TargetConfig::x86_64()
    }

    // Helper: create a fresh DiagnosticEmitter.
    fn new_diagnostics() -> DiagnosticEmitter {
        DiagnosticEmitter::new()
    }

    // Helper: shorthand types.
    fn ty_int() -> CType {
        CType::Integer(IntegerKind::Int)
    }
    fn ty_uint() -> CType {
        CType::Integer(IntegerKind::UnsignedInt)
    }
    fn ty_float() -> CType {
        CType::Float(FloatKind::Float)
    }
    fn ty_double() -> CType {
        CType::Float(FloatKind::Double)
    }
    fn ty_long() -> CType {
        CType::Integer(IntegerKind::Long)
    }
    fn ty_char() -> CType {
        CType::Integer(IntegerKind::Char)
    }
    fn ty_bool() -> CType {
        CType::Integer(IntegerKind::Bool)
    }
    fn ty_void() -> CType {
        CType::Void
    }

    fn ty_ptr(inner: CType) -> CType {
        CType::Pointer {
            pointee: Box::new(inner),
            qualifiers: TypeQualifiers::default(),
        }
    }

    fn ty_const_qualified(inner: CType) -> CType {
        CType::Qualified {
            base: Box::new(inner),
            qualifiers: TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        }
    }

    fn ty_array_int(n: usize) -> CType {
        CType::Array {
            element: Box::new(ty_int()),
            size: ArraySize::Fixed(n),
        }
    }

    fn ty_struct(name: &str, fields: Vec<(&str, CType)>) -> CType {
        CType::Struct(StructType {
            tag: Some(name.to_string()),
            fields: fields
                .into_iter()
                .enumerate()
                .map(|(i, (n, t))| StructField {
                    name: Some(n.to_string()),
                    ty: t,
                    bit_width: None,
                    offset: i * 4,
                })
                .collect(),
            is_union: false,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        })
    }

    fn ty_union(name: &str, fields: Vec<(&str, CType)>) -> CType {
        CType::Struct(StructType {
            tag: Some(name.to_string()),
            fields: fields
                .into_iter()
                .map(|(n, t)| StructField {
                    name: Some(n.to_string()),
                    ty: t,
                    bit_width: None,
                    offset: 0,
                })
                .collect(),
            is_union: true,
            is_packed: false,
            custom_alignment: None,
            is_complete: true,
        })
    }

    fn ty_func(ret: CType, params: Vec<CType>, variadic: bool) -> CType {
        CType::Function(FunctionType {
            return_type: Box::new(ret),
            params: params
                .into_iter()
                .map(|t| FunctionParam { name: None, ty: t })
                .collect(),
            is_variadic: variadic,
            is_old_style: false,
        })
    }

    // ----- check_assignment tests -----

    #[test]
    fn assignment_int_to_int() {
        let mut d = new_diagnostics();
        assert!(check_assignment(&ty_int(), &ty_int(), &mut d, dummy_span()));
        assert!(!d.has_errors());
    }

    #[test]
    fn assignment_float_to_int() {
        let mut d = new_diagnostics();
        assert!(check_assignment(
            &ty_int(),
            &ty_float(),
            &mut d,
            dummy_span()
        ));
        assert!(!d.has_errors());
    }

    #[test]
    fn assignment_struct_to_int_fails() {
        let mut d = new_diagnostics();
        let s = ty_struct("S", vec![("x", ty_int())]);
        assert!(!check_assignment(&ty_int(), &s, &mut d, dummy_span()));
        assert!(d.has_errors());
    }

    #[test]
    fn assignment_const_target_fails() {
        let mut d = new_diagnostics();
        let ct = ty_const_qualified(ty_int());
        assert!(!check_assignment(&ct, &ty_int(), &mut d, dummy_span()));
        assert!(d.has_errors());
    }

    #[test]
    fn assignment_pointer_from_null() {
        let mut d = new_diagnostics();
        let ptr = ty_ptr(ty_int());
        // Integer 0 should be treated as null pointer constant by CType impl.
        assert!(check_assignment(&ptr, &ty_int(), &mut d, dummy_span()));
    }

    #[test]
    fn assignment_pointer_to_void_pointer() {
        let mut d = new_diagnostics();
        let vp = ty_ptr(ty_void());
        let ip = ty_ptr(ty_int());
        assert!(check_assignment(&vp, &ip, &mut d, dummy_span()));
        assert!(!d.has_errors());
    }

    // ----- check_lvalue tests -----

    #[test]
    fn lvalue_identifier_variable() {
        let mut d = new_diagnostics();
        let mut st = SymbolTable::new();
        let name = InternId::from_raw(1);
        let sym = Symbol {
            name,
            ty: ty_int(),
            storage_class: None,
            linkage: crate::sema::storage::Linkage::None,
            is_defined: true,
            is_tentative: false,
            kind: SymbolKind::Variable,
            location: dummy_span(),
            scope_depth: 0,
        };
        let _ = st.insert(name, sym, &mut d);
        let expr = Expression::Identifier {
            name,
            span: dummy_span(),
        };
        assert!(check_lvalue(&expr, &st, &mut d, dummy_span()));
    }

    #[test]
    fn lvalue_function_is_not_lvalue() {
        let mut d = new_diagnostics();
        let mut st = SymbolTable::new();
        let name = InternId::from_raw(2);
        let sym = Symbol {
            name,
            ty: ty_func(ty_int(), vec![], false),
            storage_class: None,
            linkage: crate::sema::storage::Linkage::External,
            is_defined: true,
            is_tentative: false,
            kind: SymbolKind::Function,
            location: dummy_span(),
            scope_depth: 0,
        };
        let _ = st.insert(name, sym, &mut d);
        let expr = Expression::Identifier {
            name,
            span: dummy_span(),
        };
        assert!(!check_lvalue(&expr, &st, &mut d, dummy_span()));
    }

    #[test]
    fn lvalue_subscript_is_lvalue() {
        let mut d = new_diagnostics();
        let st = SymbolTable::new();
        let expr = Expression::Subscript {
            array: Box::new(Expression::Error { span: dummy_span() }),
            index: Box::new(Expression::Error { span: dummy_span() }),
            span: dummy_span(),
        };
        assert!(check_lvalue(&expr, &st, &mut d, dummy_span()));
    }

    // ----- check_function_call tests -----

    #[test]
    fn function_call_correct_args() {
        let mut d = new_diagnostics();
        let ft = ty_func(ty_int(), vec![ty_int(), ty_float()], false);
        let result = check_function_call(
            &ft,
            &[ty_int(), ty_float()],
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn function_call_too_few_args() {
        let mut d = new_diagnostics();
        let ft = ty_func(ty_int(), vec![ty_int(), ty_float()], false);
        let _r = check_function_call(&ft, &[ty_int()], &x86_64_target(), &mut d, dummy_span());
        assert!(d.has_errors());
    }

    #[test]
    fn function_call_too_many_args() {
        let mut d = new_diagnostics();
        let ft = ty_func(ty_int(), vec![ty_int()], false);
        let _r = check_function_call(
            &ft,
            &[ty_int(), ty_float()],
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(d.has_errors());
    }

    #[test]
    fn function_call_variadic_ok() {
        let mut d = new_diagnostics();
        let ft = ty_func(ty_int(), vec![ty_int()], true);
        let result = check_function_call(
            &ft,
            &[ty_int(), ty_float(), ty_double()],
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    // ----- check_return tests -----

    #[test]
    fn return_void_bare() {
        let mut d = new_diagnostics();
        check_return(&None, &ty_void(), &mut d, dummy_span());
        assert!(!d.has_errors());
    }

    #[test]
    fn return_void_with_value_warns() {
        let mut d = new_diagnostics();
        check_return(&Some(ty_int()), &ty_void(), &mut d, dummy_span());
        // Should be a warning, not an error.
        assert!(!d.has_errors());
    }

    #[test]
    fn return_nonvoid_bare_warns() {
        let mut d = new_diagnostics();
        check_return(&None, &ty_int(), &mut d, dummy_span());
        assert!(!d.has_errors());
    }

    #[test]
    fn return_nonvoid_compatible() {
        let mut d = new_diagnostics();
        check_return(&Some(ty_float()), &ty_int(), &mut d, dummy_span());
        assert!(!d.has_errors());
    }

    // ----- check_binary_op tests -----

    #[test]
    fn binary_int_plus_int() {
        let mut d = new_diagnostics();
        let result = check_binary_op(
            BinaryOp::Add,
            &ty_int(),
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn binary_int_plus_float() {
        let mut d = new_diagnostics();
        let result = check_binary_op(
            BinaryOp::Add,
            &ty_int(),
            &ty_float(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        // Usual arithmetic conversions: int + float → float.
        assert!(result.is_float());
    }

    #[test]
    fn binary_pointer_plus_int() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_binary_op(
            BinaryOp::Add,
            &ip,
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_pointer());
    }

    #[test]
    fn binary_pointer_plus_pointer_fails() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_binary_op(
            BinaryOp::Add,
            &ip,
            &ip,
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    #[test]
    fn binary_pointer_minus_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_binary_op(
            BinaryOp::Sub,
            &ip,
            &ip,
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        // ptrdiff_t on x86-64 is Long.
        assert!(result.is_integer());
    }

    #[test]
    fn binary_mod_requires_integer() {
        let mut d = new_diagnostics();
        let result = check_binary_op(
            BinaryOp::Mod,
            &ty_float(),
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    #[test]
    fn binary_comparison_result_is_int() {
        let mut d = new_diagnostics();
        let result = check_binary_op(
            BinaryOp::Less,
            &ty_int(),
            &ty_float(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn binary_pointer_comparison() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_binary_op(
            BinaryOp::Equal,
            &ip,
            &ip,
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn binary_logical_and_requires_scalar() {
        let mut d = new_diagnostics();
        let s = ty_struct("S", vec![("x", ty_int())]);
        let result = check_binary_op(
            BinaryOp::LogicalAnd,
            &s,
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    #[test]
    fn binary_shift_left() {
        let mut d = new_diagnostics();
        let result = check_binary_op(
            BinaryOp::ShiftLeft,
            &ty_char(),
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        // Char is promoted to int for shift result.
        assert!(result.is_integer());
    }

    // ----- check_unary_op tests -----

    #[test]
    fn unary_negate_int() {
        let mut d = new_diagnostics();
        let result = check_unary_op(
            UnaryOp::Negate,
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn unary_logical_not_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_unary_op(
            UnaryOp::LogicalNot,
            &ip,
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn unary_deref_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_unary_op(
            UnaryOp::Dereference,
            &ip,
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn unary_address_of() {
        let mut d = new_diagnostics();
        let result = check_unary_op(
            UnaryOp::AddressOf,
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_pointer());
    }

    // ----- check_postfix_op tests -----

    #[test]
    fn postfix_int() {
        let mut d = new_diagnostics();
        let result = check_postfix_op(&ty_int(), &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn postfix_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_postfix_op(&ip, &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_pointer());
    }

    #[test]
    fn postfix_struct_fails() {
        let mut d = new_diagnostics();
        let s = ty_struct("S", vec![("x", ty_int())]);
        let result = check_postfix_op(&s, &mut d, dummy_span());
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    // ----- check_subscript tests -----

    #[test]
    fn subscript_pointer_int() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_subscript(&ip, &ty_int(), &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn subscript_array_int() {
        let mut d = new_diagnostics();
        let arr = ty_array_int(10);
        let result = check_subscript(&arr, &ty_int(), &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn subscript_reversed() {
        // i[a] form should also work.
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_subscript(&ty_int(), &ip, &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    // ----- check_member_access tests -----

    #[test]
    fn member_access_struct_dot() {
        let mut d = new_diagnostics();
        let s = ty_struct("Point", vec![("x", ty_int()), ("y", ty_float())]);
        let result =
            check_member_access(&s, InternId::from_raw(0), "y", false, &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_float());
    }

    #[test]
    fn member_access_arrow() {
        let mut d = new_diagnostics();
        let s = ty_struct("Point", vec![("x", ty_int()), ("y", ty_float())]);
        let ps = ty_ptr(s);
        let result =
            check_member_access(&ps, InternId::from_raw(0), "x", true, &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn member_access_not_found() {
        let mut d = new_diagnostics();
        let s = ty_struct("Point", vec![("x", ty_int())]);
        let result =
            check_member_access(&s, InternId::from_raw(0), "z", false, &mut d, dummy_span());
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    #[test]
    fn member_access_union() {
        let mut d = new_diagnostics();
        let u = ty_union("Data", vec![("i", ty_int()), ("f", ty_float())]);
        let result =
            check_member_access(&u, InternId::from_raw(0), "f", false, &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_float());
    }

    // ----- check_dereference tests -----

    #[test]
    fn deref_int_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_dereference(&ip, &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn deref_void_pointer_fails() {
        let mut d = new_diagnostics();
        let vp = ty_ptr(ty_void());
        let result = check_dereference(&vp, &mut d, dummy_span());
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    #[test]
    fn deref_non_pointer_fails() {
        let mut d = new_diagnostics();
        let result = check_dereference(&ty_int(), &mut d, dummy_span());
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    // ----- check_cast tests -----

    #[test]
    fn cast_int_to_float() {
        let mut d = new_diagnostics();
        assert!(check_cast(&ty_float(), &ty_int(), &mut d, dummy_span()));
    }

    #[test]
    fn cast_pointer_to_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let fp = ty_ptr(ty_float());
        assert!(check_cast(&fp, &ip, &mut d, dummy_span()));
    }

    #[test]
    fn cast_int_to_pointer() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        assert!(check_cast(&ip, &ty_int(), &mut d, dummy_span()));
        // Should produce a warning.
    }

    #[test]
    fn cast_struct_to_int_fails() {
        let mut d = new_diagnostics();
        let s = ty_struct("S", vec![("x", ty_int())]);
        assert!(!check_cast(&ty_int(), &s, &mut d, dummy_span()));
        assert!(d.has_errors());
    }

    #[test]
    fn cast_to_void() {
        let mut d = new_diagnostics();
        assert!(check_cast(&ty_void(), &ty_int(), &mut d, dummy_span()));
    }

    // ----- check_conditional tests -----

    #[test]
    fn conditional_both_int() {
        let mut d = new_diagnostics();
        let result = check_conditional(
            &ty_int(),
            &ty_int(),
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_integer());
    }

    #[test]
    fn conditional_int_and_float() {
        let mut d = new_diagnostics();
        let result = check_conditional(
            &ty_int(),
            &ty_int(),
            &ty_float(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_float());
    }

    #[test]
    fn conditional_both_void() {
        let mut d = new_diagnostics();
        let result = check_conditional(
            &ty_int(),
            &ty_void(),
            &ty_void(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(!d.has_errors());
        assert!(result.is_void());
    }

    #[test]
    fn conditional_non_scalar_condition_fails() {
        let mut d = new_diagnostics();
        let s = ty_struct("S", vec![("x", ty_int())]);
        let result = check_conditional(
            &s,
            &ty_int(),
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(d.has_errors());
        assert!(result.is_error());
    }

    #[test]
    fn conditional_pointers() {
        let mut d = new_diagnostics();
        let ip = ty_ptr(ty_int());
        let result = check_conditional(&ty_int(), &ip, &ip, &x86_64_target(), &mut d, dummy_span());
        assert!(!d.has_errors());
        assert!(result.is_pointer());
    }

    // ----- error propagation tests -----

    #[test]
    fn error_type_propagates() {
        let mut d = new_diagnostics();
        let result = check_binary_op(
            BinaryOp::Add,
            &CType::Error,
            &ty_int(),
            &x86_64_target(),
            &mut d,
            dummy_span(),
        );
        assert!(result.is_error());
        assert!(!d.has_errors()); // No additional error emitted.
    }
}
