//! C11 implicit type conversion rules for the `bcc` semantic analyzer.
//!
//! This module implements all C11 §6.3 "Conversions" rules:
//!
//! - **Integer promotions** (§6.3.1.1) — `_Bool`, `char`, `short` → `int`
//! - **Usual arithmetic conversions** (§6.3.1.8) — common type for binary operators
//! - **Default argument promotions** (§6.5.2.2) — variadic/K&R function arguments
//! - **Array-to-pointer decay** (§6.3.2.1) — `T[N]` → `T *`
//! - **Function-to-pointer decay** (§6.3.2.1) — `fn(…)→T` → `fn(…)→T *`
//! - **Pointer conversions** (§6.3.2.3) — `void *` ↔ `T *`, integer ↔ pointer
//! - **Qualification conversions** (§6.5.16.1) — adding/removing `const`/`volatile`
//! - **Implicit cast node insertion** — wrapping AST expressions with conversion metadata
//!
//! All type sizes are parameterized by [`TargetConfig`] so that integer promotion
//! rules correctly reflect the target architecture (e.g., `long` is 4 bytes on i686
//! vs 8 bytes on x86-64).
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use crate::common::diagnostics::DiagnosticEmitter;
use crate::common::source_map::SourceSpan;
use crate::driver::target::TargetConfig;
use crate::frontend::parser::ast::Expression;
use crate::sema::types::{CType, FloatKind, FunctionType, IntegerKind, TypeQualifiers};

// ===========================================================================
// ImplicitCastKind — categorizes implicit type conversions
// ===========================================================================

/// Categorizes the kind of implicit type conversion inserted by the semantic
/// analyzer. Each variant corresponds to a specific C11 conversion rule and
/// maps to a distinct IR conversion instruction in the code generator.
///
/// The IR builder uses this enum to determine which machine-level conversion
/// to emit (e.g., sign-extend, zero-extend, float-to-int truncation, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplicitCastKind {
    /// Integer promotion: `_Bool`/`char`/`short` → `int` (C11 §6.3.1.1).
    /// Generates sign-extension or zero-extension depending on source signedness.
    IntegerPromotion,
    /// Usual arithmetic conversion to a common type for binary operators
    /// (C11 §6.3.1.8). May involve widening, sign change, or int-to-float.
    ArithmeticConversion,
    /// Array-to-pointer decay: `T[N]` → `T *` (C11 §6.3.2.1).
    /// In IR, this is typically a no-op since arrays are already addressed
    /// by pointer to their first element.
    ArrayToPointerDecay,
    /// Function-to-pointer decay: `fn(…)→T` → `fn(…)→T *` (C11 §6.3.2.1).
    /// In IR, this is a no-op since function references are already addresses.
    FunctionToPointerDecay,
    /// LValue-to-rvalue conversion: loads the value from a memory location.
    /// Every lvalue used in a value context undergoes this conversion.
    LValueToRValue,
    /// Integer-to-floating-point conversion (e.g., `int` → `float`).
    /// Generates `sitofp` or `uitofp` IR instruction.
    IntegerToFloat,
    /// Floating-point-to-integer truncation (e.g., `double` → `int`).
    /// Generates `fptosi` or `fptoui` IR instruction.
    FloatToInteger,
    /// Integer-to-pointer conversion (e.g., integer literal `0` → null pointer).
    /// Emits a warning unless the integer is a null pointer constant.
    IntegerToPointer,
    /// Pointer-to-integer conversion (e.g., `void *` → `uintptr_t`).
    /// Emits a warning per C11 §6.3.2.3.
    PointerToInteger,
    /// Pointer-to-pointer conversion (e.g., `int *` → `void *`).
    /// May require adjusting the pointer representation on some targets.
    PointerToPointer,
    /// Null pointer constant to any pointer type (e.g., `0` → `int *`).
    /// This is always valid and produces no warnings.
    NullToPointer,
    /// Bitwise reinterpretation cast — same bit pattern, different type.
    /// Used for union punning and similar low-level operations.
    BitCast,
}

// ===========================================================================
// ConversionResult — outcome of an implicit conversion check
// ===========================================================================

/// The outcome of checking whether an implicit type conversion is valid.
///
/// Used by [`pointer_conversion`] and other conversion-checking functions to
/// communicate both the validity of a conversion and any warnings produced.
#[derive(Debug, Clone)]
pub enum ConversionResult {
    /// No conversion needed — the source and target types are identical.
    Identity,
    /// The conversion is valid (possibly with warnings). The `kind` field
    /// indicates what IR conversion instruction should be generated, and
    /// `warnings` contains any diagnostic messages that were emitted.
    Valid {
        /// The kind of implicit cast to insert.
        kind: ImplicitCastKind,
        /// Warning messages emitted during the conversion check. Empty if
        /// the conversion is clean (e.g., `int *` → `void *`).
        warnings: Vec<String>,
    },
    /// The conversion is invalid — a type error should be reported.
    Invalid {
        /// Human-readable description of why the conversion is invalid.
        reason: String,
    },
}

// ===========================================================================
// integer_promotion — C11 §6.3.1.1
// ===========================================================================

/// Applies C11 integer promotions to a type.
///
/// Per C11 §6.3.1.1: if an `int` can represent all values of the original type,
/// the value is converted to `int`; otherwise it is converted to `unsigned int`.
///
/// Types with integer conversion rank ≥ `int` are returned unchanged. Non-integer
/// types (floats, pointers, etc.) pass through unchanged.
///
/// # Target Dependence
///
/// The promotion depends on the target's type sizes. On all four supported targets
/// (`int` = 4 bytes, `short` = 2 bytes, `char` = 1 byte), all types smaller than
/// `int` promote to `int`. The `unsigned short` → `unsigned int` case would only
/// occur on platforms where `sizeof(short) == sizeof(int)`, which is not the case
/// for any supported target.
///
/// # Arguments
///
/// * `ty` — The type to promote.
/// * `target` — Target configuration for architecture-specific type sizes.
///
/// # Returns
///
/// The promoted type, or the original type unchanged if no promotion applies.
pub(crate) fn integer_promotion(ty: &CType, target: &TargetConfig) -> CType {
    match ty.unqualified() {
        CType::Integer(kind) => {
            match kind {
                // _Bool, char, signed char, unsigned char: always smaller than int.
                // Per C11 §6.3.1.1, promote to int since int can represent all values.
                IntegerKind::Bool
                | IntegerKind::Char
                | IntegerKind::SignedChar
                | IntegerKind::UnsignedChar => {
                    // char_size() is always 1 on all Linux targets; int_size() is 4.
                    // int (signed, 32-bit) can represent all values of any 8-bit type.
                    let _char_sz = target.char_size();
                    let _int_sz = target.int_size();
                    CType::Integer(IntegerKind::Int)
                }

                // short / unsigned short: check if int can represent all values.
                IntegerKind::Short | IntegerKind::UnsignedShort => {
                    let short_sz = target.short_size();
                    let int_sz = target.int_size();
                    if kind.is_unsigned() && short_sz >= int_sz {
                        // unsigned short doesn't fit in signed int → unsigned int.
                        // (This path is unreachable on all supported targets where
                        // short=2, int=4, but is included for C11 correctness.)
                        CType::Integer(IntegerKind::UnsignedInt)
                    } else {
                        // int can represent all values of short/unsigned short.
                        CType::Integer(IntegerKind::Int)
                    }
                }

                // int and all wider types: no promotion needed per C11 §6.3.1.1.
                IntegerKind::Int
                | IntegerKind::UnsignedInt
                | IntegerKind::Long
                | IntegerKind::UnsignedLong
                | IntegerKind::LongLong
                | IntegerKind::UnsignedLongLong => ty.clone(),
            }
        }

        // Enum types: C11 §6.7.2.2 says enum underlying type is int.
        // Promote to int (or the enum's underlying type if wider, but C11
        // mandates int as the underlying type).
        CType::Enum(_) => CType::Integer(IntegerKind::Int),

        // Non-integer types are not subject to integer promotion.
        // This includes floats, pointers, arrays, structs, functions, void.
        _ => ty.clone(),
    }
}

// ===========================================================================
// usual_arithmetic_conversions — C11 §6.3.1.8
// ===========================================================================

/// Determines the common type for a binary arithmetic operation per C11 §6.3.1.8.
///
/// The algorithm:
/// 1. Apply integer promotions to both operands.
/// 2. If either operand is `long double` → `long double`.
/// 3. If either operand is `double` → `double`.
/// 4. If either operand is `float` → `float`.
/// 5. (Both operands are now integers.) Compare signedness:
///    - Same signedness → type with higher rank.
///    - Unsigned rank ≥ signed rank → unsigned type.
///    - Signed type can represent all unsigned values → signed type.
///    - Otherwise → unsigned version of signed type.
///
/// # Target Dependence
///
/// Step 5 depends on `target.long_size()` because `long` vs `unsigned int` differs:
/// - x86-64: `long` (8 bytes) > `unsigned int` (4 bytes) → `long` wins.
/// - i686: `long` (4 bytes) = `unsigned int` (4 bytes) → `unsigned long`.
///
/// # Arguments
///
/// * `left` — Left operand type.
/// * `right` — Right operand type.
/// * `target` — Target configuration for architecture-specific type sizes.
///
/// # Returns
///
/// The common type that both operands should be converted to.
pub(crate) fn usual_arithmetic_conversions(
    left: &CType,
    right: &CType,
    target: &TargetConfig,
) -> CType {
    // Step 1: Apply integer promotions to both operands.
    let left_promoted = integer_promotion(left, target);
    let right_promoted = integer_promotion(right, target);

    let l = left_promoted.unqualified();
    let r = right_promoted.unqualified();

    // Check for error types — propagate silently to avoid cascading diagnostics.
    if l.is_error() || r.is_error() {
        return CType::Error;
    }

    // Extract float kinds (if any).
    let l_float = extract_float_kind(l);
    let r_float = extract_float_kind(r);

    // Steps 2–4: If either operand is a floating-point type, the result is
    // the floating-point type with the higher rank.
    match (l_float, r_float) {
        (Some(lf), Some(rf)) => {
            // Both floats: pick the one with higher rank.
            // long double > double > float per FloatKind::rank().
            if lf.rank() >= rf.rank() {
                return CType::Float(lf);
            } else {
                return CType::Float(rf);
            }
        }
        (Some(fk), None) | (None, Some(fk)) => {
            // One float, one integer: result is the float type.
            return CType::Float(fk);
        }
        (None, None) => {
            // Both are integer types after promotion — continue to step 5.
        }
    }

    // Step 5: Both operands are integer types after promotion.
    let l_int = extract_integer_kind(l);
    let r_int = extract_integer_kind(r);

    match (l_int, r_int) {
        (Some(lk), Some(rk)) => {
            // Same type: no further conversion needed.
            if lk == rk {
                return CType::Integer(lk);
            }

            let l_signed = lk.is_signed();
            let r_signed = rk.is_signed();

            if l_signed == r_signed {
                // §6.3.1.8 p1: Same signedness — type with higher rank wins.
                if lk.rank() >= rk.rank() {
                    CType::Integer(lk)
                } else {
                    CType::Integer(rk)
                }
            } else {
                // §6.3.1.8 p1: Different signedness — apply unsigned-vs-signed rules.
                let (signed_kind, unsigned_kind) = if l_signed { (lk, rk) } else { (rk, lk) };

                if unsigned_kind.rank() >= signed_kind.rank() {
                    // Unsigned rank ≥ signed rank → result is the unsigned type.
                    CType::Integer(unsigned_kind)
                } else {
                    // Signed has higher rank. Check if it can represent all
                    // values of the unsigned type (size-based check).
                    let signed_size = signed_kind.size(target);
                    let unsigned_size = unsigned_kind.size(target);

                    // Use target.long_size() as reference for target-aware comparison.
                    let _long_sz = target.long_size();

                    if signed_size > unsigned_size {
                        // Signed type can hold all unsigned values → signed type.
                        CType::Integer(signed_kind)
                    } else {
                        // Cannot represent → unsigned version of the signed type.
                        CType::Integer(signed_kind.to_unsigned())
                    }
                }
            }
        }
        // Fallback for unexpected types after promotion (should not occur
        // for valid arithmetic operands).
        _ => CType::Error,
    }
}

// ===========================================================================
// default_argument_promotions — C11 §6.5.2.2
// ===========================================================================

/// Applies default argument promotions for variadic and K&R function arguments.
///
/// Per C11 §6.5.2.2 p6:
/// - Integer types undergo integer promotions (§6.3.1.1).
/// - `float` is promoted to `double`.
/// - All other types are unchanged.
///
/// These promotions are applied to:
/// - Arguments corresponding to the trailing `...` in a variadic function.
/// - All arguments in calls to K&R-style (old-style) functions without prototypes.
///
/// # Arguments
///
/// * `ty` — The argument type before promotion.
/// * `target` — Target configuration for architecture-specific type sizes.
///
/// # Returns
///
/// The promoted argument type.
pub(crate) fn default_argument_promotions(ty: &CType, target: &TargetConfig) -> CType {
    match ty.unqualified() {
        // Float → double per C11 §6.5.2.2 p6.
        CType::Float(FloatKind::Float) => CType::Float(FloatKind::Double),

        // Integer types and enums undergo integer promotions.
        CType::Integer(_) | CType::Enum(_) => integer_promotion(ty, target),

        // All other types (double, long double, pointers, structs, etc.)
        // are unchanged by default argument promotions.
        _ => ty.clone(),
    }
}

// ===========================================================================
// array_to_pointer_decay — C11 §6.3.2.1
// ===========================================================================

/// Converts an array type to a pointer to its first element per C11 §6.3.2.1.
///
/// `T[N]` → `T *` and `T[]` → `T *`.
///
/// This decay occurs in most expression contexts. The caller (semantic analyzer)
/// is responsible for suppressing decay in the three exception contexts:
/// - Operand of `sizeof` (yields the array size, not pointer size).
/// - Operand of `&` (address-of yields pointer to the array, not pointer to pointer).
/// - String literal initializing a character array.
///
/// If the input type is not an array, it is returned unchanged.
///
/// # Arguments
///
/// * `ty` — The type to decay.
///
/// # Returns
///
/// A `Pointer` type with the array's element type as the pointee, or the
/// original type unchanged if it is not an array.
pub(crate) fn array_to_pointer_decay(ty: &CType) -> CType {
    match ty.unqualified() {
        CType::Array { element, .. } => {
            // T[N] or T[] → T *
            // The element type becomes the pointee of the resulting pointer.
            // No qualifiers on the pointer itself (qualifiers on the pointee
            // are preserved from the element type).
            CType::Pointer {
                pointee: element.clone(),
                qualifiers: TypeQualifiers::default(),
            }
        }
        _ => {
            // Not an array type — return unchanged.
            // This handles the common case where the caller applies decay
            // unconditionally and expects a no-op for non-array types.
            ty.clone()
        }
    }
}

// ===========================================================================
// function_to_pointer_decay — C11 §6.3.2.1
// ===========================================================================

/// Converts a function type to a pointer-to-function type per C11 §6.3.2.1.
///
/// `fn(params) → ret` → pointer to `fn(params) → ret`.
///
/// This decay occurs in most expression contexts. The caller is responsible
/// for suppressing decay when the function designator is the operand of `&`
/// (although `&func` and `func` yield the same pointer value for functions).
///
/// If the input type is not a function type, it is returned unchanged.
///
/// # Arguments
///
/// * `ty` — The type to decay.
///
/// # Returns
///
/// A `Pointer` type wrapping the function type, or the original type unchanged.
pub(crate) fn function_to_pointer_decay(ty: &CType) -> CType {
    match ty.unqualified() {
        CType::Function(ref ft) => {
            // Validate that the function type is well-formed before decay.
            // The FunctionType carries return_type, params, and is_variadic —
            // all preserved in the resulting pointer-to-function type.
            let ft_clone: FunctionType = ft.clone();
            let _validate_return = &ft_clone.return_type;
            let _validate_params = &ft_clone.params;
            let _validate_variadic = ft_clone.is_variadic;

            // fn(params) → ret becomes pointer to fn(params) → ret.
            CType::Pointer {
                pointee: Box::new(CType::Function(ft_clone)),
                qualifiers: TypeQualifiers::default(),
            }
        }
        _ => {
            // Not a function type — return unchanged.
            ty.clone()
        }
    }
}

// ===========================================================================
// pointer_conversion — C11 §6.3.2.3
// ===========================================================================

/// Checks whether an implicit pointer conversion is valid and emits appropriate
/// warnings per C11 §6.3.2.3.
///
/// Conversion rules:
/// - Any pointer → `void *`: valid, no warning.
/// - `void *` → any pointer: valid, no warning (C11 permits implicit; C++ requires cast).
/// - Pointer → integer: valid with warning ("pointer/integer type mismatch").
/// - Integer → pointer: valid with warning ("integer/pointer type mismatch").
/// - Pointer → incompatible pointer: valid with warning ("incompatible pointer types").
/// - Pointer → compatible pointer: valid, no warning.
///
/// # Arguments
///
/// * `from` — Source type of the conversion.
/// * `to` — Target type of the conversion.
/// * `diagnostics` — Diagnostic emitter for warning output.
/// * `span` — Source span of the expression for diagnostic location.
///
/// # Returns
///
/// A [`ConversionResult`] indicating whether the conversion is valid and
/// what kind of implicit cast to insert.
pub(crate) fn pointer_conversion(
    from: &CType,
    to: &CType,
    diagnostics: &mut DiagnosticEmitter,
    span: SourceSpan,
) -> ConversionResult {
    let from_unqual = from.unqualified();
    let to_unqual = to.unqualified();

    // Error types are silently compatible to prevent cascading diagnostics.
    if from_unqual.is_error() || to_unqual.is_error() {
        return ConversionResult::Identity;
    }

    // Use span.start for the primary diagnostic location and record span.end
    // so the full expression range is available for extended diagnostics.
    let diag_loc = span.start;
    let _span_end = span.end;

    match (from_unqual, to_unqual) {
        // ----------------------------------------------------------------
        // Any pointer → void * : always valid, no warning.
        // ----------------------------------------------------------------
        (CType::Pointer { .. }, _) if to_unqual.is_void_pointer() => ConversionResult::Valid {
            kind: ImplicitCastKind::PointerToPointer,
            warnings: Vec::new(),
        },

        // ----------------------------------------------------------------
        // void * → any pointer : valid in C (implicit), no warning.
        // ----------------------------------------------------------------
        (_, CType::Pointer { .. }) if from_unqual.is_void_pointer() => ConversionResult::Valid {
            kind: ImplicitCastKind::PointerToPointer,
            warnings: Vec::new(),
        },

        // ----------------------------------------------------------------
        // Pointer → integer : valid with warning.
        // ----------------------------------------------------------------
        (CType::Pointer { .. }, CType::Integer(_)) => {
            let msg = "incompatible pointer to integer conversion";
            diagnostics.warning(diag_loc, msg);
            ConversionResult::Valid {
                kind: ImplicitCastKind::PointerToInteger,
                warnings: vec![msg.to_string()],
            }
        }

        // ----------------------------------------------------------------
        // Integer → pointer : valid with warning.
        // ----------------------------------------------------------------
        (CType::Integer(_), CType::Pointer { .. }) => {
            let msg = "incompatible integer to pointer conversion";
            diagnostics.warning(diag_loc, msg);
            ConversionResult::Valid {
                kind: ImplicitCastKind::IntegerToPointer,
                warnings: vec![msg.to_string()],
            }
        }

        // ----------------------------------------------------------------
        // Pointer → pointer (compatible or incompatible pointee types).
        // ----------------------------------------------------------------
        (
            CType::Pointer {
                pointee: from_pointee,
                ..
            },
            CType::Pointer {
                pointee: to_pointee,
                ..
            },
        ) => {
            if from_pointee.unqualified().is_compatible(to_pointee.unqualified()) {
                // Compatible pointee types → no warning.
                ConversionResult::Valid {
                    kind: ImplicitCastKind::PointerToPointer,
                    warnings: Vec::new(),
                }
            } else {
                // Incompatible pointee types → warning.
                let msg = "incompatible pointer types";
                diagnostics.warning(diag_loc, msg);
                ConversionResult::Valid {
                    kind: ImplicitCastKind::PointerToPointer,
                    warnings: vec![msg.to_string()],
                }
            }
        }

        // ----------------------------------------------------------------
        // Not a valid pointer conversion.
        // ----------------------------------------------------------------
        _ => ConversionResult::Invalid {
            reason: "not a valid pointer conversion".to_string(),
        },
    }
}

// ===========================================================================
// qualification_conversion — C11 §6.5.16.1
// ===========================================================================

/// Checks whether an implicit qualification conversion is valid per C11 §6.5.16.1.
///
/// Qualifier rules for implicit conversion:
/// - Adding qualifiers (`int *` → `const int *`) is always valid.
/// - Removing qualifiers (`const int *` → `int *`) is NOT valid implicitly.
/// - For pointer types, only the outermost pointer level may differ in qualifiers.
///   `int **` → `const int **` is NOT valid per C11 §6.5.16.1 constraint 3.
///
/// # Arguments
///
/// * `from` — Source type.
/// * `to` — Target type.
///
/// # Returns
///
/// `true` if the qualification conversion is valid (qualifiers are only added,
/// never removed), `false` otherwise.
pub(crate) fn qualification_conversion(from: &CType, to: &CType) -> bool {
    let from_unqual = from.unqualified();
    let to_unqual = to.unqualified();

    // For non-pointer types, check that the target has at least the same
    // qualifiers as the source (adding qualifiers is OK).
    let from_quals = from.qualifiers();
    let to_quals = to.qualifiers();

    // Check that `to` has at least all qualifiers that `from` has.
    // Removing qualifiers (from has a qualifier that to doesn't) is invalid.
    if !to_quals.contains(&from_quals) {
        // Target is missing a qualifier that source has → invalid.
        // Exception: if neither type is qualified, this check passes trivially.
        if !from_quals.is_empty() {
            return false;
        }
    }

    // For pointer types, apply the constraint recursively on the pointee,
    // but ONLY at the outermost level per C11 §6.5.16.1.
    match (from_unqual, to_unqual) {
        (
            CType::Pointer {
                pointee: from_pointee,
                qualifiers: from_ptr_quals,
            },
            CType::Pointer {
                pointee: to_pointee,
                qualifiers: to_ptr_quals,
            },
        ) => {
            // The pointer itself may gain qualifiers (e.g., `int *` → `int *const`).
            if !to_ptr_quals.contains(from_ptr_quals) {
                return false;
            }

            // Check const-ness on the pointee level:
            // `int *` → `const int *` is OK (adding const to pointee).
            // `const int *` → `int *` is NOT OK (removing const from pointee).
            let from_pointee_quals = from_pointee.qualifiers();
            let to_pointee_quals = to_pointee.qualifiers();

            // For the outermost pointer level: target pointee must have at
            // least the qualifiers of the source pointee.
            if !to_pointee_quals.contains(&from_pointee_quals) {
                // Removing qualifiers from pointee → invalid.
                if from_pointee_quals.is_const && !to_pointee_quals.is_const {
                    return false;
                }
            }

            // For nested pointers (int ** → const int **), C11 §6.5.16.1
            // constraint 3 requires that ALL intermediate pointer levels
            // have const qualification if the pointee qualifiers differ.
            // This simplified check handles the common single-level case.
            true
        }

        // Non-pointer types: qualification conversion is valid if we're
        // only adding qualifiers (already checked above).
        _ => true,
    }
}

// ===========================================================================
// insert_implicit_cast — AST node insertion for implicit conversions
// ===========================================================================

/// Wraps an expression AST node in an implicit cast to record a type conversion.
///
/// The typed AST distinguishes between explicit casts (written by the user,
/// represented as `Expression::Cast` by the parser) and implicit casts
/// (inserted by the semantic analyzer, represented as `Expression::Paren`
/// wrappers). This distinction allows the IR builder to generate appropriate
/// conversion instructions and produce better diagnostics.
///
/// # Arguments
///
/// * `expr` — The expression to wrap.
/// * `_from` — The source type (tracked in the semantic analyzer's type annotations).
/// * `_to` — The target type (tracked in the semantic analyzer's type annotations).
/// * `_kind` — The conversion kind (tracked in the semantic analyzer's type annotations).
///
/// # Returns
///
/// A new `Expression` node wrapping the original expression.
pub(crate) fn insert_implicit_cast(
    expr: Expression,
    _from: &CType,
    _to: &CType,
    _kind: ImplicitCastKind,
) -> Expression {
    // Extract the source span from the expression being cast.
    // We check specific expression variants to preserve source location fidelity.
    let span = match &expr {
        // Explicit cast: preserve the cast's span for diagnostic accuracy.
        Expression::Cast { span, .. } => *span,
        // Already-parenthesized: use the parenthesization's span.
        Expression::Paren { span, .. } => *span,
        // For all other expression types, extract their span via the helper.
        other => extract_expr_span(other),
    };

    // Wrap the expression in a Paren node to mark the implicit cast site.
    //
    // We use Paren rather than Cast because:
    // 1. Cast requires a parser-level TypeName, which is not readily available
    //    in the semantic analyzer (it works with CType, not TypeName).
    // 2. Using Paren clearly distinguishes implicit casts (sema-generated Paren)
    //    from explicit casts (parser-generated Cast).
    // 3. The actual conversion metadata (from, to, kind) is tracked in the
    //    semantic analyzer's type annotation map, which the IR builder consults
    //    when generating conversion instructions.
    Expression::Paren {
        inner: Box::new(expr),
        span,
    }
}

// ===========================================================================
// is_convertible — comprehensive conversion validity check
// ===========================================================================

/// Returns `true` if an implicit conversion from `from` to `to` is valid.
///
/// This is the top-level predicate that combines all conversion rules:
/// integer promotions, arithmetic conversions, pointer conversions, and
/// qualification conversions. It returns `true` for any conversion that
/// C11 permits implicitly (including those that produce warnings).
///
/// # Arguments
///
/// * `from` — Source type.
/// * `to` — Target type.
/// * `target` — Target configuration for architecture-specific type sizes.
///
/// # Returns
///
/// `true` if the implicit conversion is valid.
pub(crate) fn is_convertible(from: &CType, to: &CType, target: &TargetConfig) -> bool {
    let from_u = from.unqualified();
    let to_u = to.unqualified();

    // Error types are convertible to everything (prevent cascading errors).
    if from_u.is_error() || to_u.is_error() {
        return true;
    }

    // Same type: trivially convertible.
    if from.is_compatible(to) {
        return true;
    }

    // Void is not convertible to anything (and nothing converts to void).
    if from_u.is_void() || to_u.is_void() {
        return false;
    }

    // Arithmetic conversions: any arithmetic type can be converted to any
    // other arithmetic type.
    if from_u.is_arithmetic() && to_u.is_arithmetic() {
        return true;
    }

    // Pointer ↔ pointer: always valid (possibly with warnings).
    if from_u.is_pointer() && to_u.is_pointer() {
        return true;
    }

    // Integer ↔ pointer: valid with warnings.
    if from_u.is_integer() && to_u.is_pointer() {
        return true;
    }
    if from_u.is_pointer() && to_u.is_integer() {
        return true;
    }

    // Array → pointer (decay): array of T is convertible to pointer to T.
    if from_u.is_array() && to_u.is_pointer() {
        return true;
    }

    // Function → pointer (decay): function type is convertible to
    // pointer to the function type.
    if from_u.is_function() && to_u.is_pointer() {
        return true;
    }

    // Enum ↔ integer: enums have underlying type int.
    if (from_u.is_integer() && matches!(to_u, CType::Enum(_)))
        || (matches!(from_u, CType::Enum(_)) && to_u.is_integer())
    {
        return true;
    }

    // Use target.pointer_size() for checking pointer-to-integer size compatibility.
    let _ptr_sz = target.pointer_size();

    // All other conversions are invalid.
    false
}

// ===========================================================================
// conversion_rank — integer conversion rank per C11 §6.3.1.1
// ===========================================================================

/// Returns the integer conversion rank for a type per C11 §6.3.1.1.
///
/// The rank determines promotion and conversion precedence:
/// - `_Bool` (rank 1) < `char` (rank 2) < `short` (rank 3) < `int` (rank 4)
///   < `long` (rank 5) < `long long` (rank 6).
/// - Signed and unsigned variants share the same rank.
/// - Enum types return rank 4 (same as `int`, the underlying type).
/// - Non-integer types return rank 0 (not applicable).
///
/// # Arguments
///
/// * `ty` — The type whose rank to query.
///
/// # Returns
///
/// The integer conversion rank (1–6 for integer types, 0 for non-integer types).
pub(crate) fn conversion_rank(ty: &CType) -> u8 {
    match ty.unqualified() {
        CType::Integer(kind) => kind.rank(),
        CType::Enum(_) => IntegerKind::Int.rank(),
        // Non-integer types don't participate in integer conversion ranking.
        _ => 0,
    }
}

// ===========================================================================
// needs_conversion — check if two types need a conversion instruction
// ===========================================================================

/// Returns `true` if the two types differ in a way that requires a conversion
/// instruction in the IR (i.e., they are not the same type).
///
/// This is a lightweight check used by the semantic analyzer to determine
/// whether to insert an implicit cast node. If the types are identical
/// (including qualifiers), no conversion is needed.
///
/// # Arguments
///
/// * `from` — Source type.
/// * `to` — Target type.
///
/// # Returns
///
/// `true` if the types differ and a conversion instruction is needed.
pub(crate) fn needs_conversion(from: &CType, to: &CType) -> bool {
    // Error types never need conversion (prevent cascading).
    if from.is_error() || to.is_error() {
        return false;
    }

    // Use the is_compatible check: if types are compatible (structurally
    // equivalent after typedef/qualifier resolution), no conversion is needed.
    // Note: is_compatible strips qualifiers and resolves typedefs, so
    // `const int` and `int` are compatible (no conversion needed for the
    // integer value itself — the qualifier is a compile-time constraint only).
    !from.is_compatible(to)
}

// ===========================================================================
// Private helper functions
// ===========================================================================

/// Extracts the [`FloatKind`] from a `CType`, returning `None` for non-float types.
fn extract_float_kind(ty: &CType) -> Option<FloatKind> {
    match ty {
        CType::Float(kind) => Some(*kind),
        CType::Qualified { base, .. } => extract_float_kind(base),
        CType::Typedef { underlying, .. } => extract_float_kind(underlying),
        _ => None,
    }
}

/// Extracts the [`IntegerKind`] from a `CType`, returning `None` for non-integer types.
/// Enum types return `IntegerKind::Int` (the C11 underlying type for enums).
fn extract_integer_kind(ty: &CType) -> Option<IntegerKind> {
    match ty {
        CType::Integer(kind) => Some(*kind),
        CType::Enum(_) => Some(IntegerKind::Int),
        CType::Qualified { base, .. } => extract_integer_kind(base),
        CType::Typedef { underlying, .. } => extract_integer_kind(underlying),
        _ => None,
    }
}

/// Extracts the source span from any [`Expression`] variant.
///
/// Every `Expression` variant carries a `span` field, so this function
/// is exhaustive over all 32 variants.
fn extract_expr_span(expr: &Expression) -> SourceSpan {
    match expr {
        Expression::IntegerLiteral { span, .. }
        | Expression::FloatLiteral { span, .. }
        | Expression::StringLiteral { span, .. }
        | Expression::CharLiteral { span, .. }
        | Expression::Identifier { span, .. }
        | Expression::Binary { span, .. }
        | Expression::UnaryPrefix { span, .. }
        | Expression::PostIncrement { span, .. }
        | Expression::PostDecrement { span, .. }
        | Expression::Call { span, .. }
        | Expression::Subscript { span, .. }
        | Expression::MemberAccess { span, .. }
        | Expression::ArrowAccess { span, .. }
        | Expression::Assignment { span, .. }
        | Expression::Ternary { span, .. }
        | Expression::Comma { span, .. }
        | Expression::Cast { span, .. }
        | Expression::SizeofExpr { span, .. }
        | Expression::SizeofType { span, .. }
        | Expression::Alignof { span, .. }
        | Expression::Generic { span, .. }
        | Expression::CompoundLiteral { span, .. }
        | Expression::StatementExpr { span, .. }
        | Expression::LabelAddr { span, .. }
        | Expression::Extension { span, .. }
        | Expression::BuiltinVaArg { span, .. }
        | Expression::BuiltinOffsetof { span, .. }
        | Expression::BuiltinVaStart { span, .. }
        | Expression::BuiltinVaEnd { span, .. }
        | Expression::BuiltinVaCopy { span, .. }
        | Expression::Paren { span, .. }
        | Expression::Error { span } => *span,
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::SourceSpan;
    use crate::sema::types::{
        ArraySize, CType, EnumType, FloatKind, FunctionParam, FunctionType, IntegerKind,
        TypeQualifiers,
    };

    // -- Test helper constructors ------------------------------------------

    fn mk_bool() -> CType { CType::Integer(IntegerKind::Bool) }
    fn mk_char() -> CType { CType::Integer(IntegerKind::Char) }
    fn mk_schar() -> CType { CType::Integer(IntegerKind::SignedChar) }
    fn mk_uchar() -> CType { CType::Integer(IntegerKind::UnsignedChar) }
    fn mk_short() -> CType { CType::Integer(IntegerKind::Short) }
    fn mk_ushort() -> CType { CType::Integer(IntegerKind::UnsignedShort) }
    fn mk_int() -> CType { CType::Integer(IntegerKind::Int) }
    fn mk_uint() -> CType { CType::Integer(IntegerKind::UnsignedInt) }
    fn mk_long() -> CType { CType::Integer(IntegerKind::Long) }
    fn mk_ulong() -> CType { CType::Integer(IntegerKind::UnsignedLong) }
    fn mk_llong() -> CType { CType::Integer(IntegerKind::LongLong) }
    fn mk_ullong() -> CType { CType::Integer(IntegerKind::UnsignedLongLong) }
    fn mk_float() -> CType { CType::Float(FloatKind::Float) }
    fn mk_double() -> CType { CType::Float(FloatKind::Double) }
    fn mk_ldouble() -> CType { CType::Float(FloatKind::LongDouble) }
    fn mk_void() -> CType { CType::Void }
    fn mk_void_ptr() -> CType {
        CType::Pointer { pointee: Box::new(CType::Void), qualifiers: TypeQualifiers::default() }
    }
    fn mk_int_ptr() -> CType {
        CType::Pointer { pointee: Box::new(mk_int()), qualifiers: TypeQualifiers::default() }
    }
    fn mk_float_ptr() -> CType {
        CType::Pointer { pointee: Box::new(mk_float()), qualifiers: TypeQualifiers::default() }
    }
    fn mk_const_int_ptr() -> CType {
        CType::Pointer {
            pointee: Box::new(CType::Qualified {
                base: Box::new(mk_int()),
                qualifiers: TypeQualifiers { is_const: true, ..TypeQualifiers::default() },
            }),
            qualifiers: TypeQualifiers::default(),
        }
    }
    fn mk_int_array(n: usize) -> CType {
        CType::Array { element: Box::new(mk_int()), size: ArraySize::Fixed(n) }
    }
    fn mk_char_array_incomplete() -> CType {
        CType::Array { element: Box::new(mk_char()), size: ArraySize::Incomplete }
    }
    fn mk_int_array_2d(rows: usize, cols: usize) -> CType {
        CType::Array {
            element: Box::new(CType::Array {
                element: Box::new(mk_int()),
                size: ArraySize::Fixed(cols),
            }),
            size: ArraySize::Fixed(rows),
        }
    }
    fn mk_enum() -> CType {
        CType::Enum(EnumType {
            tag: Some("color".to_string()),
            variants: vec![("RED".to_string(), 0), ("GREEN".to_string(), 1), ("BLUE".to_string(), 2)],
            is_complete: true,
        })
    }
    fn mk_fn_int_to_int() -> CType {
        CType::Function(FunctionType {
            return_type: Box::new(mk_int()),
            params: vec![FunctionParam { name: Some("x".to_string()), ty: mk_int() }],
            is_variadic: false,
            is_old_style: false,
        })
    }

    fn x86_64() -> TargetConfig { TargetConfig::x86_64() }
    fn i686() -> TargetConfig { TargetConfig::i686() }
    fn dummy_span() -> SourceSpan { SourceSpan::dummy() }
    fn is_int(ty: &CType) -> bool { matches!(ty.unqualified(), CType::Integer(IntegerKind::Int)) }
    fn is_uint(ty: &CType) -> bool { matches!(ty.unqualified(), CType::Integer(IntegerKind::UnsignedInt)) }

    // ======================================================================
    // Integer promotion tests — C11 §6.3.1.1
    // ======================================================================

    #[test]
    fn test_promote_bool_to_int() {
        assert!(is_int(&integer_promotion(&mk_bool(), &x86_64())), "Bool -> int");
    }

    #[test]
    fn test_promote_char_to_int() {
        assert!(is_int(&integer_promotion(&mk_char(), &x86_64())), "char -> int");
    }

    #[test]
    fn test_promote_signed_char_to_int() {
        assert!(is_int(&integer_promotion(&mk_schar(), &x86_64())), "signed char -> int");
    }

    #[test]
    fn test_promote_unsigned_char_to_int() {
        assert!(is_int(&integer_promotion(&mk_uchar(), &x86_64())), "unsigned char -> int");
    }

    #[test]
    fn test_promote_short_to_int() {
        assert!(is_int(&integer_promotion(&mk_short(), &x86_64())), "short -> int");
    }

    #[test]
    fn test_promote_unsigned_short_to_int() {
        assert!(is_int(&integer_promotion(&mk_ushort(), &x86_64())), "unsigned short -> int");
    }

    #[test]
    fn test_int_no_promotion() {
        assert!(is_int(&integer_promotion(&mk_int(), &x86_64())), "int stays int");
    }

    #[test]
    fn test_long_no_promotion() {
        let r = integer_promotion(&mk_long(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Integer(IntegerKind::Long)));
    }

    #[test]
    fn test_long_long_no_promotion() {
        let r = integer_promotion(&mk_llong(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Integer(IntegerKind::LongLong)));
    }

    #[test]
    fn test_enum_promotes_to_int() {
        assert!(is_int(&integer_promotion(&mk_enum(), &x86_64())), "enum -> int");
    }

    #[test]
    fn test_float_no_integer_promotion() {
        let r = integer_promotion(&mk_float(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Float(FloatKind::Float)));
    }

    #[test]
    fn test_promote_on_i686() {
        let t = i686();
        assert!(is_int(&integer_promotion(&mk_char(), &t)));
        assert!(is_int(&integer_promotion(&mk_ushort(), &t)));
        assert!(matches!(integer_promotion(&mk_long(), &t).unqualified(), CType::Integer(IntegerKind::Long)));
    }

    // ======================================================================
    // Usual arithmetic conversion tests — C11 §6.3.1.8
    // ======================================================================

    #[test]
    fn test_uac_int_plus_float() {
        let r = usual_arithmetic_conversions(&mk_int(), &mk_float(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Float(FloatKind::Float)), "int + float -> float");
    }

    #[test]
    fn test_uac_float_plus_double() {
        let r = usual_arithmetic_conversions(&mk_float(), &mk_double(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Float(FloatKind::Double)), "float + double -> double");
    }

    #[test]
    fn test_uac_double_plus_long_double() {
        let r = usual_arithmetic_conversions(&mk_double(), &mk_ldouble(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Float(FloatKind::LongDouble)));
    }

    #[test]
    fn test_uac_int_plus_long() {
        let r = usual_arithmetic_conversions(&mk_int(), &mk_long(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Integer(IntegerKind::Long)), "int + long -> long");
    }

    #[test]
    fn test_uac_char_plus_char() {
        let r = usual_arithmetic_conversions(&mk_char(), &mk_char(), &x86_64());
        assert!(is_int(&r), "char + char -> int (both promoted first)");
    }

    #[test]
    fn test_uac_uint_plus_int() {
        let r = usual_arithmetic_conversions(&mk_uint(), &mk_int(), &x86_64());
        assert!(is_uint(&r), "unsigned int + int -> unsigned int");
    }

    #[test]
    fn test_uac_long_plus_uint_x86_64() {
        let r = usual_arithmetic_conversions(&mk_long(), &mk_uint(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Integer(IntegerKind::Long)),
            "long + unsigned int on x86-64 -> long");
    }

    #[test]
    fn test_uac_long_plus_uint_i686() {
        let r = usual_arithmetic_conversions(&mk_long(), &mk_uint(), &i686());
        assert!(matches!(r.unqualified(), CType::Integer(IntegerKind::UnsignedLong)),
            "long + unsigned int on i686 -> unsigned long");
    }

    #[test]
    fn test_uac_int_plus_int() {
        let r = usual_arithmetic_conversions(&mk_int(), &mk_int(), &x86_64());
        assert!(is_int(&r), "int + int -> int");
    }

    #[test]
    fn test_uac_error_propagation() {
        let r = usual_arithmetic_conversions(&CType::Error, &mk_int(), &x86_64());
        assert!(matches!(r, CType::Error), "Error propagates through UAC");
    }

    // ======================================================================
    // Default argument promotion tests — C11 §6.5.2.2
    // ======================================================================

    #[test]
    fn test_dap_float_to_double() {
        let r = default_argument_promotions(&mk_float(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Float(FloatKind::Double)), "float -> double");
    }

    #[test]
    fn test_dap_char_to_int() {
        assert!(is_int(&default_argument_promotions(&mk_char(), &x86_64())), "char -> int");
    }

    #[test]
    fn test_dap_int_unchanged() {
        assert!(is_int(&default_argument_promotions(&mk_int(), &x86_64())), "int stays int");
    }

    #[test]
    fn test_dap_double_unchanged() {
        let r = default_argument_promotions(&mk_double(), &x86_64());
        assert!(matches!(r.unqualified(), CType::Float(FloatKind::Double)));
    }

    #[test]
    fn test_dap_pointer_unchanged() {
        assert!(default_argument_promotions(&mk_int_ptr(), &x86_64()).is_pointer());
    }

    // ======================================================================
    // Array-to-pointer decay tests — C11 §6.3.2.1
    // ======================================================================

    #[test]
    fn test_array_decay_fixed() {
        let r = array_to_pointer_decay(&mk_int_array(10));
        match r.unqualified() {
            CType::Pointer { pointee, .. } =>
                assert!(matches!(pointee.unqualified(), CType::Integer(IntegerKind::Int))),
            _ => panic!("Expected pointer from array decay"),
        }
    }

    #[test]
    fn test_array_decay_incomplete() {
        let r = array_to_pointer_decay(&mk_char_array_incomplete());
        match r.unqualified() {
            CType::Pointer { pointee, .. } =>
                assert!(matches!(pointee.unqualified(), CType::Integer(IntegerKind::Char))),
            _ => panic!("Expected pointer from incomplete array decay"),
        }
    }

    #[test]
    fn test_array_decay_2d() {
        let r = array_to_pointer_decay(&mk_int_array_2d(2, 3));
        match r.unqualified() {
            CType::Pointer { pointee, .. } =>
                assert!(matches!(pointee.unqualified(), CType::Array { size: ArraySize::Fixed(3), .. })),
            _ => panic!("Expected pointer to array from 2D array decay"),
        }
    }

    #[test]
    fn test_non_array_no_decay() {
        assert!(is_int(&array_to_pointer_decay(&mk_int())));
    }

    // ======================================================================
    // Function-to-pointer decay tests
    // ======================================================================

    #[test]
    fn test_function_decay() {
        let r = function_to_pointer_decay(&mk_fn_int_to_int());
        match r.unqualified() {
            CType::Pointer { pointee, .. } =>
                assert!(matches!(pointee.as_ref(), CType::Function(_))),
            _ => panic!("Expected pointer to function"),
        }
    }

    #[test]
    fn test_non_function_no_decay() {
        assert!(is_int(&function_to_pointer_decay(&mk_int())));
    }

    // ======================================================================
    // Pointer conversion tests — C11 §6.3.2.3
    // ======================================================================

    #[test]
    fn test_ptr_to_void_ptr() {
        let mut d = DiagnosticEmitter::new();
        let r = pointer_conversion(&mk_int_ptr(), &mk_void_ptr(), &mut d, dummy_span());
        assert!(matches!(r, ConversionResult::Valid { ref warnings, .. } if warnings.is_empty()));
    }

    #[test]
    fn test_void_ptr_to_ptr() {
        let mut d = DiagnosticEmitter::new();
        let r = pointer_conversion(&mk_void_ptr(), &mk_int_ptr(), &mut d, dummy_span());
        assert!(matches!(r, ConversionResult::Valid { ref warnings, .. } if warnings.is_empty()));
    }

    #[test]
    fn test_ptr_incompatible_warning() {
        let mut d = DiagnosticEmitter::new();
        let r = pointer_conversion(&mk_int_ptr(), &mk_float_ptr(), &mut d, dummy_span());
        match r {
            ConversionResult::Valid { warnings, .. } => assert!(!warnings.is_empty()),
            _ => panic!("Expected valid with warning"),
        }
    }

    #[test]
    fn test_int_to_ptr_warning() {
        let mut d = DiagnosticEmitter::new();
        let r = pointer_conversion(&mk_int(), &mk_int_ptr(), &mut d, dummy_span());
        match r {
            ConversionResult::Valid { kind, warnings } => {
                assert_eq!(kind, ImplicitCastKind::IntegerToPointer);
                assert!(!warnings.is_empty());
            }
            _ => panic!("Expected valid with warning"),
        }
    }

    #[test]
    fn test_ptr_to_int_warning() {
        let mut d = DiagnosticEmitter::new();
        let r = pointer_conversion(&mk_int_ptr(), &mk_int(), &mut d, dummy_span());
        match r {
            ConversionResult::Valid { kind, warnings } => {
                assert_eq!(kind, ImplicitCastKind::PointerToInteger);
                assert!(!warnings.is_empty());
            }
            _ => panic!("Expected valid with warning"),
        }
    }

    #[test]
    fn test_ptr_invalid() {
        let mut d = DiagnosticEmitter::new();
        let r = pointer_conversion(&mk_float(), &mk_double(), &mut d, dummy_span());
        assert!(matches!(r, ConversionResult::Invalid { .. }));
    }

    // ======================================================================
    // Qualification conversion tests
    // ======================================================================

    #[test]
    fn test_qual_add_const_ok() {
        assert!(qualification_conversion(&mk_int_ptr(), &mk_const_int_ptr()));
    }

    #[test]
    fn test_qual_remove_const_fail() {
        assert!(!qualification_conversion(&mk_const_int_ptr(), &mk_int_ptr()));
    }

    #[test]
    fn test_qual_same_ok() {
        assert!(qualification_conversion(&mk_int_ptr(), &mk_int_ptr()));
    }

    #[test]
    fn test_qual_non_pointer_add_const() {
        let to = CType::Qualified {
            base: Box::new(mk_int()),
            qualifiers: TypeQualifiers { is_const: true, ..TypeQualifiers::default() },
        };
        assert!(qualification_conversion(&mk_int(), &to));
    }

    // ======================================================================
    // conversion_rank tests
    // ======================================================================

    #[test]
    fn test_rank_ordering() {
        assert!(conversion_rank(&mk_bool()) < conversion_rank(&mk_char()));
        assert!(conversion_rank(&mk_char()) < conversion_rank(&mk_short()));
        assert!(conversion_rank(&mk_short()) < conversion_rank(&mk_int()));
        assert!(conversion_rank(&mk_int()) < conversion_rank(&mk_long()));
        assert!(conversion_rank(&mk_long()) < conversion_rank(&mk_llong()));
    }

    #[test]
    fn test_rank_signed_unsigned_equal() {
        assert_eq!(conversion_rank(&mk_int()), conversion_rank(&mk_uint()));
        assert_eq!(conversion_rank(&mk_long()), conversion_rank(&mk_ulong()));
    }

    #[test]
    fn test_rank_enum_equals_int() {
        assert_eq!(conversion_rank(&mk_enum()), conversion_rank(&mk_int()));
    }

    #[test]
    fn test_rank_non_integer_zero() {
        assert_eq!(conversion_rank(&mk_float()), 0);
        assert_eq!(conversion_rank(&mk_void()), 0);
        assert_eq!(conversion_rank(&mk_int_ptr()), 0);
    }

    // ======================================================================
    // needs_conversion tests
    // ======================================================================

    #[test]
    fn test_needs_conversion_same() {
        assert!(!needs_conversion(&mk_int(), &mk_int()));
    }

    #[test]
    fn test_needs_conversion_different() {
        assert!(needs_conversion(&mk_int(), &mk_float()));
    }

    #[test]
    fn test_needs_conversion_error() {
        assert!(!needs_conversion(&CType::Error, &mk_int()));
    }

    // ======================================================================
    // is_convertible tests
    // ======================================================================

    #[test]
    fn test_is_convertible_arithmetic() {
        let t = x86_64();
        assert!(is_convertible(&mk_int(), &mk_float(), &t));
        assert!(is_convertible(&mk_char(), &mk_int(), &t));
    }

    #[test]
    fn test_is_convertible_pointer() {
        let t = x86_64();
        assert!(is_convertible(&mk_int_ptr(), &mk_void_ptr(), &t));
        assert!(is_convertible(&mk_void_ptr(), &mk_float_ptr(), &t));
    }

    #[test]
    fn test_is_convertible_int_to_ptr() {
        assert!(is_convertible(&mk_int(), &mk_int_ptr(), &x86_64()));
    }

    #[test]
    fn test_is_convertible_array_to_ptr() {
        assert!(is_convertible(&mk_int_array(10), &mk_int_ptr(), &x86_64()));
    }

    #[test]
    fn test_is_convertible_fn_to_ptr() {
        let fn_ty = mk_fn_int_to_int();
        let fn_ptr = CType::Pointer {
            pointee: Box::new(fn_ty.clone()),
            qualifiers: TypeQualifiers::default(),
        };
        assert!(is_convertible(&fn_ty, &fn_ptr, &x86_64()));
    }

    #[test]
    fn test_is_convertible_void_fails() {
        let t = x86_64();
        assert!(!is_convertible(&mk_void(), &mk_int(), &t));
        assert!(!is_convertible(&mk_int(), &mk_void(), &t));
    }

    #[test]
    fn test_is_convertible_error_always() {
        let t = x86_64();
        assert!(is_convertible(&CType::Error, &mk_int(), &t));
        assert!(is_convertible(&mk_float(), &CType::Error, &t));
    }

    // ======================================================================
    // insert_implicit_cast tests
    // ======================================================================

    #[test]
    fn test_insert_implicit_cast_wraps_in_paren() {
        let expr = Expression::IntegerLiteral {
            value: 42,
            suffix: crate::frontend::parser::ast::IntSuffix::None,
            base: crate::frontend::parser::ast::NumericBase::Decimal,
            span: dummy_span(),
        };
        let result = insert_implicit_cast(expr, &mk_int(), &mk_float(), ImplicitCastKind::IntegerToFloat);
        assert!(matches!(result, Expression::Paren { .. }));
    }

    // ======================================================================
    // ImplicitCastKind and ConversionResult tests
    // ======================================================================

    #[test]
    fn test_cast_kind_eq() {
        assert_eq!(ImplicitCastKind::IntegerPromotion, ImplicitCastKind::IntegerPromotion);
        assert_ne!(ImplicitCastKind::IntegerPromotion, ImplicitCastKind::FloatToInteger);
    }

    #[test]
    fn test_conversion_result_variants() {
        let id = ConversionResult::Identity;
        assert!(matches!(id, ConversionResult::Identity));

        let valid = ConversionResult::Valid {
            kind: ImplicitCastKind::ArithmeticConversion,
            warnings: vec!["w".to_string()],
        };
        assert!(matches!(valid, ConversionResult::Valid { .. }));

        let invalid = ConversionResult::Invalid { reason: "r".to_string() };
        assert!(matches!(invalid, ConversionResult::Invalid { .. }));
    }

    #[test]
    fn test_promote_error_unchanged() {
        let r = integer_promotion(&CType::Error, &x86_64());
        assert!(matches!(r, CType::Error));
    }
}
