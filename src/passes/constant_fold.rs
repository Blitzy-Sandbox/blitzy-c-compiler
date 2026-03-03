//! Constant folding optimization pass for the `bcc` compiler.
//!
//! This pass evaluates arithmetic, comparison, logical, bitwise, and cast
//! operations on known compile-time constants, replacing expressions like
//! `Add(Const(2), Const(3))` with `Const(5)`. It handles both integer and
//! floating-point constant operations and propagates constants through the IR
//! within a single forward pass.
//!
//! # Semantics
//!
//! - **Integer overflow**: Wraps using two's complement (C semantics).
//! - **Division by zero**: Left unchanged (undefined behavior in C).
//! - **Floating-point**: Follows IEEE 754 semantics (including NaN propagation
//!   and ±Inf for division by zero).
//! - **Shift by ≥ width or negative**: Left unchanged (undefined behavior in C).
//!
//! # Partial Constant Folding
//!
//! In addition to full folding (both operands constant), this pass handles
//! cases where only one operand is constant but the result is still known:
//! - `Mul(x, 0)` → `Const(0)` (zero absorption)
//! - `And(x, 0)` → `Const(0)` (zero absorption)
//! - `Or(x, all_ones)` → `Const(all_ones)`
//!
//! Identity simplifications (e.g., `Add(x, 0) → x`) are handled by the
//! `simplify` pass instead, as their result is a value — not a new constant.
//!
//! # Branch Folding
//!
//! When the condition of a `CondBranch` or the value of a `Switch` is a known
//! constant, the terminator is replaced with an unconditional `Branch` to the
//! determined target.
//!
//! # Integration
//!
//! Called by `pipeline.rs` at `-O1` (single execution) and `-O2` (iterative
//! fixed-point loop). Results may enable further DCE (dead code after branch
//! folding) and simplification.

use std::collections::HashMap;

use crate::ir::cfg::Terminator;
use crate::ir::instructions::{
    CastOp, CompareOp, Constant, FloatCompareOp, Instruction, Value,
};
use crate::ir::builder::Function;
use crate::ir::types::IrType;
use crate::passes::FunctionPass;

// ---------------------------------------------------------------------------
// ConstantFoldPass — public pass struct
// ---------------------------------------------------------------------------

/// Constant folding optimization pass.
///
/// Evaluates compile-time constant arithmetic, comparisons, casts, and
/// conditional branches. Implements the [`FunctionPass`] trait for integration
/// with the pass pipeline.
///
/// # Usage
///
/// ```ignore
/// let mut pass = ConstantFoldPass::new();
/// let changed = pass.run_on_function(&mut function);
/// ```
pub struct ConstantFoldPass;

impl ConstantFoldPass {
    /// Creates a new constant folding pass instance.
    #[inline]
    pub fn new() -> Self {
        ConstantFoldPass
    }
}

impl FunctionPass for ConstantFoldPass {
    /// Returns the human-readable name of this pass.
    fn name(&self) -> &str {
        "constant_fold"
    }

    /// Runs constant folding on the given function.
    ///
    /// Performs a single forward pass over all basic blocks and instructions,
    /// maintaining a `HashMap<Value, Constant>` of known constant values.
    /// When an instruction's operands are all known constants, the instruction
    /// is replaced with a `Const` instruction producing the folded result.
    ///
    /// # Returns
    ///
    /// `true` if any instructions or terminators were folded, `false` otherwise.
    fn run_on_function(&mut self, function: &mut Function) -> bool {
        let mut constants: HashMap<Value, Constant> = HashMap::new();
        let mut changed = false;

        // Build a set of parameter value IDs. The IR builder emits placeholder
        // Const instructions for function parameters (with value = parameter
        // index). These must NOT be treated as foldable constants — they
        // represent runtime values passed by the caller via ABI registers.
        let param_set: std::collections::HashSet<Value> =
            function.param_values.iter().copied().collect();

        for block_idx in 0..function.blocks.len() {
            let num_insts = function.blocks[block_idx].instructions.len();
            for inst_idx in 0..num_insts {
                // Clone the instruction to avoid borrow conflicts between the
                // instruction reference and the mutable constants map / block list.
                let inst = function.blocks[block_idx].instructions[inst_idx].clone();

                // Register constants from Const instructions into the tracking map,
                // but SKIP parameter placeholder constants — their integer values
                // are parameter indices, not runtime constants.
                if let Instruction::Const { result, ref value } = inst {
                    if !param_set.contains(&result) {
                        constants.insert(result, value.clone());
                    }
                    continue;
                }

                // Attempt to fold the instruction if it produces a result.
                if let Some(result) = inst.result() {
                    if let Some(folded) = try_fold_instruction(&inst, &constants) {
                        constants.insert(result, folded.clone());
                        function.blocks[block_idx].instructions[inst_idx] =
                            Instruction::Const {
                                result,
                                value: folded,
                            };
                        changed = true;
                    }
                }
            }

            // Attempt to fold the block's terminator (conditional branches, switches).
            changed |= try_fold_terminator(&mut function.blocks[block_idx], &constants);
        }

        changed
    }
}

// ---------------------------------------------------------------------------
// BinOp — internal binary operation classification
// ---------------------------------------------------------------------------

/// Internal enum classifying binary operations for the folding helpers.
///
/// This separates signed/unsigned variants of division, modulo, and shift
/// that are encoded as boolean flags in the IR instruction variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    SignedDiv,
    UnsignedDiv,
    SignedMod,
    UnsignedMod,
    And,
    Or,
    Xor,
    Shl,
    ArithShr,
    LogicalShr,
}

// ---------------------------------------------------------------------------
// try_fold_instruction — top-level instruction folding dispatcher
// ---------------------------------------------------------------------------

/// Attempts to fold a single instruction into a compile-time constant.
///
/// Returns `Some(Constant)` if the instruction can be fully or partially
/// evaluated, `None` if it cannot be folded.
fn try_fold_instruction(
    inst: &Instruction,
    constants: &HashMap<Value, Constant>,
) -> Option<Constant> {
    match inst {
        // --- Binary arithmetic (integer or float) ---
        Instruction::Add { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::Add, *lhs, *rhs, ty, constants)
        }
        Instruction::Sub { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::Sub, *lhs, *rhs, ty, constants)
        }
        Instruction::Mul { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::Mul, *lhs, *rhs, ty, constants)
        }
        Instruction::Div {
            lhs,
            rhs,
            ty,
            is_signed,
            ..
        } => {
            let op = if *is_signed {
                BinOp::SignedDiv
            } else {
                BinOp::UnsignedDiv
            };
            fold_binary(op, *lhs, *rhs, ty, constants)
        }
        Instruction::Mod {
            lhs,
            rhs,
            ty,
            is_signed,
            ..
        } => {
            let op = if *is_signed {
                BinOp::SignedMod
            } else {
                BinOp::UnsignedMod
            };
            fold_binary(op, *lhs, *rhs, ty, constants)
        }

        // --- Binary bitwise ---
        Instruction::And { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::And, *lhs, *rhs, ty, constants)
        }
        Instruction::Or { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::Or, *lhs, *rhs, ty, constants)
        }
        Instruction::Xor { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::Xor, *lhs, *rhs, ty, constants)
        }
        Instruction::Shl { lhs, rhs, ty, .. } => {
            fold_binary(BinOp::Shl, *lhs, *rhs, ty, constants)
        }
        Instruction::Shr {
            lhs,
            rhs,
            ty,
            is_arithmetic,
            ..
        } => {
            let op = if *is_arithmetic {
                BinOp::ArithShr
            } else {
                BinOp::LogicalShr
            };
            fold_binary(op, *lhs, *rhs, ty, constants)
        }

        // --- Integer comparison ---
        Instruction::ICmp { op, lhs, rhs, .. } => {
            let lc = constants.get(lhs)?;
            let rc = constants.get(rhs)?;
            if let Some((a, b)) = get_int_pair(lc, rc) {
                Some(fold_icmp(op, a, b))
            } else {
                None
            }
        }

        // --- Floating-point comparison ---
        Instruction::FCmp { op, lhs, rhs, .. } => {
            let lc = constants.get(lhs)?;
            let rc = constants.get(rhs)?;
            if let Some((a, b)) = get_float_pair(lc, rc) {
                Some(fold_fcmp(op, a, b))
            } else {
                None
            }
        }

        // --- Type cast ---
        Instruction::Cast {
            op,
            value,
            from_ty,
            to_ty,
            ..
        } => {
            let vc = constants.get(value)?;
            fold_cast(op, vc, from_ty, to_ty)
        }

        // All other instructions (Load, Store, Call, Alloca, Phi, Select,
        // BitCast, GetElementPtr, Copy, Nop) are not foldable by this pass.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// fold_binary — binary operation folding with full + partial paths
// ---------------------------------------------------------------------------

/// Attempts to fold a binary operation. Tries full constant folding first
/// (both operands known), then falls back to partial folding (one operand
/// known, result still constant).
fn fold_binary(
    op: BinOp,
    lhs: Value,
    rhs: Value,
    ty: &IrType,
    constants: &HashMap<Value, Constant>,
) -> Option<Constant> {
    let lc = constants.get(&lhs);
    let rc = constants.get(&rhs);

    // Full constant fold: both operands are known constants.
    if let (Some(lc), Some(rc)) = (lc, rc) {
        if ty.is_integer() {
            if let Some((a, b)) = get_int_pair(lc, rc) {
                return try_fold_int_binary(op, a, b, ty);
            }
        }
        if ty.is_float() {
            if let Some((a, b)) = get_float_pair(lc, rc) {
                return try_fold_float_binary(op, a, b, ty);
            }
        }
    }

    // Partial constant fold: one operand is constant, but the result is still
    // a compile-time constant (e.g., Mul(x, 0) = 0).
    try_fold_partial(op, lc, rc, ty)
}

// ---------------------------------------------------------------------------
// try_fold_int_binary — integer binary operation folding
// ---------------------------------------------------------------------------

/// Evaluates a binary integer operation on two known constant operands.
///
/// Returns `None` when the operation cannot be folded safely:
/// - Division or modulo by zero (UB in C — leave for runtime).
/// - Signed division overflow (e.g., `i32::MIN / -1`).
/// - Shift by negative amount or amount ≥ bit width (UB in C).
fn try_fold_int_binary(op: BinOp, a: i64, b: i64, ty: &IrType) -> Option<Constant> {
    let bits = ty.integer_bit_width()?;

    let result = match op {
        BinOp::Add => wrap_to_width(a.wrapping_add(b), bits),
        BinOp::Sub => wrap_to_width(a.wrapping_sub(b), bits),
        BinOp::Mul => wrap_to_width(a.wrapping_mul(b), bits),

        BinOp::SignedDiv => {
            if b == 0 {
                return None;
            }
            checked_signed_div(a, b, bits)?
        }
        BinOp::UnsignedDiv => {
            if b == 0 {
                return None;
            }
            unsigned_div(a, b, bits)
        }
        BinOp::SignedMod => {
            if b == 0 {
                return None;
            }
            checked_signed_rem(a, b, bits)?
        }
        BinOp::UnsignedMod => {
            if b == 0 {
                return None;
            }
            unsigned_rem(a, b, bits)
        }

        BinOp::And => wrap_to_width(a & b, bits),
        BinOp::Or => wrap_to_width(a | b, bits),
        BinOp::Xor => wrap_to_width(a ^ b, bits),

        BinOp::Shl => {
            if b < 0 || b as u32 >= bits {
                return None; // UB in C
            }
            wrap_to_width(a.wrapping_shl(b as u32), bits)
        }
        BinOp::ArithShr => {
            if b < 0 || b as u32 >= bits {
                return None; // UB in C
            }
            arithmetic_shr(a, b as u32, bits)
        }
        BinOp::LogicalShr => {
            if b < 0 || b as u32 >= bits {
                return None; // UB in C
            }
            logical_shr(a, b as u32, bits)
        }
    };

    Some(Constant::Integer {
        value: result,
        ty: ty.clone(),
    })
}

// ---------------------------------------------------------------------------
// try_fold_float_binary — floating-point binary operation folding
// ---------------------------------------------------------------------------

/// Evaluates a binary floating-point operation on two known constant operands.
///
/// IEEE 754 semantics are preserved: division by zero produces ±Inf or NaN
/// rather than being left unfolded.
fn try_fold_float_binary(op: BinOp, a: f64, b: f64, ty: &IrType) -> Option<Constant> {
    let result = match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        // IEEE 754 handles div-by-zero (produces ±Inf or NaN).
        BinOp::SignedDiv | BinOp::UnsignedDiv => a / b,
        // Bitwise and shift operations do not apply to floats.
        _ => return None,
    };

    // Ensure F32 precision when the target type is F32.
    let result = if *ty == IrType::F32 {
        (result as f32) as f64
    } else {
        result
    };

    Some(Constant::Float {
        value: result,
        ty: ty.clone(),
    })
}

// ---------------------------------------------------------------------------
// fold_icmp — integer comparison folding
// ---------------------------------------------------------------------------

/// Evaluates an integer comparison on two known constant operands.
///
/// Always produces `Constant::Integer { value: 0 or 1, ty: IrType::I1 }`.
fn fold_icmp(op: &CompareOp, a: i64, b: i64) -> Constant {
    let result = match op {
        CompareOp::Equal => a == b,
        CompareOp::NotEqual => a != b,
        CompareOp::SignedLess => a < b,
        CompareOp::SignedLessEqual => a <= b,
        CompareOp::SignedGreater => a > b,
        CompareOp::SignedGreaterEqual => a >= b,
        CompareOp::UnsignedLess => (a as u64) < (b as u64),
        CompareOp::UnsignedLessEqual => (a as u64) <= (b as u64),
        CompareOp::UnsignedGreater => (a as u64) > (b as u64),
        CompareOp::UnsignedGreaterEqual => (a as u64) >= (b as u64),
    };
    Constant::Integer {
        value: if result { 1 } else { 0 },
        ty: IrType::I1,
    }
}

// ---------------------------------------------------------------------------
// fold_fcmp — floating-point comparison folding
// ---------------------------------------------------------------------------

/// Evaluates a floating-point comparison on two known constant operands.
///
/// IEEE 754 semantics:
/// - **Ordered** comparisons return `false` if either operand is NaN.
/// - **Unordered** comparisons return `true` if either operand is NaN.
///
/// Always produces `Constant::Integer { value: 0 or 1, ty: IrType::I1 }`.
fn fold_fcmp(op: &FloatCompareOp, a: f64, b: f64) -> Constant {
    let result = match op {
        FloatCompareOp::OrderedEqual => !a.is_nan() && !b.is_nan() && a == b,
        FloatCompareOp::OrderedNotEqual => !a.is_nan() && !b.is_nan() && a != b,
        FloatCompareOp::OrderedLess => !a.is_nan() && !b.is_nan() && a < b,
        FloatCompareOp::OrderedLessEqual => !a.is_nan() && !b.is_nan() && a <= b,
        FloatCompareOp::OrderedGreater => !a.is_nan() && !b.is_nan() && a > b,
        FloatCompareOp::OrderedGreaterEqual => !a.is_nan() && !b.is_nan() && a >= b,
        FloatCompareOp::Unordered => a.is_nan() || b.is_nan(),
        FloatCompareOp::UnorderedEqual => a.is_nan() || b.is_nan() || a == b,
    };
    Constant::Integer {
        value: if result { 1 } else { 0 },
        ty: IrType::I1,
    }
}

// ---------------------------------------------------------------------------
// fold_cast — type cast constant folding
// ---------------------------------------------------------------------------

/// Evaluates a type cast on a known constant value.
///
/// Returns `None` for pointer-related casts (`PtrToInt`, `IntToPtr`) which
/// depend on runtime addresses.
fn fold_cast(
    op: &CastOp,
    value: &Constant,
    from_ty: &IrType,
    to_ty: &IrType,
) -> Option<Constant> {
    match op {
        CastOp::Trunc => {
            // Integer truncation: mask to target bit width and sign-extend.
            if let Constant::Integer { value: v, .. } = value {
                let bits = to_ty.integer_bit_width()?;
                let result = wrap_to_width(*v, bits);
                Some(Constant::Integer {
                    value: result,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::ZExt => {
            // Zero-extend: mask to source bit width (clearing high bits).
            if let Constant::Integer { value: v, .. } = value {
                let src_bits = from_ty.integer_bit_width()?;
                let result = mask_to_unsigned_width(*v, src_bits);
                Some(Constant::Integer {
                    value: result,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::SExt => {
            // Sign-extend: wrap to source width (sign-extends naturally in i64).
            if let Constant::Integer { value: v, .. } = value {
                let src_bits = from_ty.integer_bit_width()?;
                let result = wrap_to_width(*v, src_bits);
                Some(Constant::Integer {
                    value: result,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::FPToSI => {
            // Float to signed integer.
            if let Constant::Float { value: v, .. } = value {
                Some(Constant::Integer {
                    value: *v as i64,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::FPToUI => {
            // Float to unsigned integer.
            if let Constant::Float { value: v, .. } = value {
                Some(Constant::Integer {
                    value: *v as u64 as i64,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::SIToFP => {
            // Signed integer to float.
            if let Constant::Integer { value: v, .. } = value {
                let fval = *v as f64;
                let fval = if *to_ty == IrType::F32 {
                    (fval as f32) as f64
                } else {
                    fval
                };
                Some(Constant::Float {
                    value: fval,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::UIToFP => {
            // Unsigned integer to float: mask to source width first.
            if let Constant::Integer { value: v, .. } = value {
                let src_bits = from_ty.integer_bit_width().unwrap_or(64);
                let uval = mask_to_unsigned_width(*v, src_bits) as u64;
                let fval = uval as f64;
                let fval = if *to_ty == IrType::F32 {
                    (fval as f32) as f64
                } else {
                    fval
                };
                Some(Constant::Float {
                    value: fval,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::FPTrunc => {
            // Double to float: truncate precision.
            if let Constant::Float { value: v, .. } = value {
                let result = (*v as f32) as f64;
                Some(Constant::Float {
                    value: result,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        CastOp::FPExt => {
            // Float to double: extend precision (value unchanged in f64 storage).
            if let Constant::Float { value: v, .. } = value {
                Some(Constant::Float {
                    value: *v,
                    ty: to_ty.clone(),
                })
            } else {
                None
            }
        }
        // Pointer-related casts depend on runtime addresses.
        CastOp::PtrToInt | CastOp::IntToPtr => None,
    }
}

// ---------------------------------------------------------------------------
// try_fold_partial — partial constant folding (one operand known)
// ---------------------------------------------------------------------------

/// Attempts partial constant folding where one operand is constant and the
/// result is still a compile-time constant regardless of the other operand.
///
/// Handled cases (where the RESULT is always a constant):
/// - `Mul(x, 0)` → `Const(0)` (zero absorption)
/// - `And(x, 0)` → `Const(0)` (zero absorption)
/// - `Or(x, all_ones)` → `Const(all_ones)` for the type width
fn try_fold_partial(
    op: BinOp,
    lc: Option<&Constant>,
    rc: Option<&Constant>,
    ty: &IrType,
) -> Option<Constant> {
    // Partial folding only applies to integer types.
    if !ty.is_integer() {
        return None;
    }

    match op {
        // Mul(x, 0) = 0  or  Mul(0, x) = 0
        BinOp::Mul => {
            if is_int_zero(lc) || is_int_zero(rc) {
                return Some(Constant::Integer {
                    value: 0,
                    ty: ty.clone(),
                });
            }
        }
        // And(x, 0) = 0  or  And(0, x) = 0
        BinOp::And => {
            if is_int_zero(lc) || is_int_zero(rc) {
                return Some(Constant::Integer {
                    value: 0,
                    ty: ty.clone(),
                });
            }
        }
        // Or(x, all_ones) = all_ones  or  Or(all_ones, x) = all_ones
        BinOp::Or => {
            if let Some(bits) = ty.integer_bit_width() {
                if is_all_ones_for_width(lc, bits) || is_all_ones_for_width(rc, bits) {
                    let all_ones_val = wrap_to_width(-1i64, bits);
                    return Some(Constant::Integer {
                        value: all_ones_val,
                        ty: ty.clone(),
                    });
                }
            }
        }
        _ => {}
    }

    None
}

// ---------------------------------------------------------------------------
// try_fold_terminator — conditional branch and switch folding
// ---------------------------------------------------------------------------

/// Attempts to fold a basic block's terminator when its condition/value is
/// a known constant. Replaces `CondBranch` or `Switch` with an unconditional
/// `Branch` to the determined target.
fn try_fold_terminator(
    block: &mut crate::ir::cfg::BasicBlock,
    constants: &HashMap<Value, Constant>,
) -> bool {
    // Clone the terminator to avoid overlapping borrows.
    let term = match &block.terminator {
        Some(t) => t.clone(),
        None => return false,
    };

    match term {
        Terminator::CondBranch {
            condition,
            true_block,
            false_block,
        } => {
            if let Some(cond_const) = constants.get(&condition) {
                let is_true = match cond_const {
                    Constant::Integer { value, .. } => *value != 0,
                    Constant::Bool(b) => *b,
                    _ => return false,
                };
                let target = if is_true { true_block } else { false_block };
                block.terminator = Some(Terminator::Branch { target });
                true
            } else {
                false
            }
        }
        Terminator::Switch {
            value,
            default,
            ref cases,
        } => {
            if let Some(val_const) = constants.get(&value) {
                if let Constant::Integer {
                    value: switch_val, ..
                } = val_const
                {
                    // Find the matching case, or use the default target.
                    let target = cases
                        .iter()
                        .find(|(case_val, _)| *case_val == *switch_val)
                        .map(|(_, tgt)| *tgt)
                        .unwrap_or(default);
                    block.terminator = Some(Terminator::Branch { target });
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Arithmetic helper functions
// ---------------------------------------------------------------------------

/// Truncates an i64 value to `bits` width and sign-extends back to i64.
///
/// This correctly handles C's wrapping integer semantics: after a 32-bit
/// operation, the result is truncated to 32 bits and then sign-extended.
#[inline]
fn wrap_to_width(val: i64, bits: u32) -> i64 {
    match bits {
        1 => val & 1,
        8 => (val as i8) as i64,
        16 => (val as i16) as i64,
        32 => (val as i32) as i64,
        64 => val,
        _ => val,
    }
}

/// Masks an i64 value to `bits` width as an unsigned value (zero-extends).
///
/// The result is always non-negative when `bits < 64`.
#[inline]
fn mask_to_unsigned_width(val: i64, bits: u32) -> i64 {
    if bits >= 64 {
        val
    } else {
        ((val as u64) & ((1u64 << bits) - 1)) as i64
    }
}

/// Performs signed division with overflow checking.
///
/// Returns `None` when the division would overflow (e.g., `i32::MIN / -1`).
fn checked_signed_div(a: i64, b: i64, bits: u32) -> Option<i64> {
    match bits {
        8 => (a as i8).checked_div(b as i8).map(|r| r as i64),
        16 => (a as i16).checked_div(b as i16).map(|r| r as i64),
        32 => (a as i32).checked_div(b as i32).map(|r| r as i64),
        64 => a.checked_div(b),
        _ => None,
    }
}

/// Performs signed remainder with overflow checking.
///
/// Returns `None` when the remainder would overflow.
fn checked_signed_rem(a: i64, b: i64, bits: u32) -> Option<i64> {
    match bits {
        8 => (a as i8).checked_rem(b as i8).map(|r| r as i64),
        16 => (a as i16).checked_rem(b as i16).map(|r| r as i64),
        32 => (a as i32).checked_rem(b as i32).map(|r| r as i64),
        64 => a.checked_rem(b),
        _ => None,
    }
}

/// Performs unsigned division on values masked to `bits` width.
fn unsigned_div(a: i64, b: i64, bits: u32) -> i64 {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let au = (a as u64) & mask;
    let bu = (b as u64) & mask;
    (au / bu) as i64
}

/// Performs unsigned remainder on values masked to `bits` width.
fn unsigned_rem(a: i64, b: i64, bits: u32) -> i64 {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let au = (a as u64) & mask;
    let bu = (b as u64) & mask;
    (au % bu) as i64
}

/// Arithmetic (signed) right shift: preserves the sign bit.
fn arithmetic_shr(a: i64, shift: u32, bits: u32) -> i64 {
    match bits {
        8 => ((a as i8) >> shift) as i64,
        16 => ((a as i16) >> shift) as i64,
        32 => ((a as i32) >> shift) as i64,
        64 => a >> shift,
        _ => a >> shift,
    }
}

/// Logical (unsigned) right shift: fills with zeros.
fn logical_shr(a: i64, shift: u32, bits: u32) -> i64 {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let au = (a as u64) & mask;
    (au >> shift) as i64
}

// ---------------------------------------------------------------------------
// Constant extraction helpers
// ---------------------------------------------------------------------------

/// Extracts integer values from a pair of constants.
fn get_int_pair(lc: &Constant, rc: &Constant) -> Option<(i64, i64)> {
    match (lc, rc) {
        (
            Constant::Integer { value: a, .. },
            Constant::Integer { value: b, .. },
        ) => Some((*a, *b)),
        _ => None,
    }
}

/// Extracts float values from a pair of constants.
fn get_float_pair(lc: &Constant, rc: &Constant) -> Option<(f64, f64)> {
    match (lc, rc) {
        (
            Constant::Float { value: a, .. },
            Constant::Float { value: b, .. },
        ) => Some((*a, *b)),
        _ => None,
    }
}

/// Returns `true` if the optional constant is an integer zero.
#[inline]
fn is_int_zero(c: Option<&Constant>) -> bool {
    matches!(c, Some(Constant::Integer { value: 0, .. }))
}

/// Returns `true` if the optional constant is an all-ones integer at the
/// given bit width.
fn is_all_ones_for_width(c: Option<&Constant>, bits: u32) -> bool {
    if let Some(Constant::Integer { value, .. }) = c {
        match bits {
            1 => (*value & 1) == 1,
            n if n < 64 => {
                let mask = (1u64 << n) - 1;
                (*value as u64 & mask) == mask
            }
            64 => *value == -1,
            _ => false,
        }
    } else {
        false
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::{BasicBlock, Terminator};
    use crate::ir::instructions::{BlockId, Constant, Instruction, Value};
    use crate::ir::types::IrType;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Creates a minimal single-block function from the given instructions.
    fn make_function(instructions: Vec<Instruction>) -> Function {
        let mut block = BasicBlock::new(BlockId(0), "entry".to_string());
        block.instructions = instructions;
        block.terminator = Some(Terminator::Return { value: None });
        Function {
            name: "test_fn".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    /// Creates a function with multiple blocks for branch folding tests.
    fn make_function_with_branch(
        instructions: Vec<Instruction>,
        terminator: Terminator,
    ) -> Function {
        let mut entry = BasicBlock::new(BlockId(0), "entry".to_string());
        entry.instructions = instructions;
        entry.terminator = Some(terminator);

        let mut true_block = BasicBlock::new(BlockId(1), "true_bb".to_string());
        true_block.terminator = Some(Terminator::Return { value: None });

        let mut false_block = BasicBlock::new(BlockId(2), "false_bb".to_string());
        false_block.terminator = Some(Terminator::Return { value: None });

        Function {
            name: "test_branch".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry, true_block, false_block],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    /// Finds the `Constant` value for a given SSA `Value` in the function.
    fn find_const(func: &Function, val: Value) -> Option<Constant> {
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Const { result, value } = inst {
                    if *result == val {
                        return Some(value.clone());
                    }
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Integer arithmetic folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 2, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I32 },
            },
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 5, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_sub_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 10, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I32 },
            },
            Instruction::Sub {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 7, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_mul_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 4, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 5, ty: IrType::I32 },
            },
            Instruction::Mul {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 20, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_signed_div_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 10, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I32 },
            },
            Instruction::Div {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
                is_signed: true,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 3, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_mod_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 10, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I32 },
            },
            Instruction::Mod {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
                is_signed: true,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 1, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_div_by_zero_not_folded() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 10, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0, ty: IrType::I32 },
            },
            Instruction::Div {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
                is_signed: true,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        // Division by zero must NOT be folded.
        assert!(!pass.run_on_function(&mut func));
        // Value(2) should not be a Const instruction.
        assert!(find_const(&func, Value(2)).is_none());
    }

    // -----------------------------------------------------------------------
    // Integer overflow tests (wrapping semantics)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_wrapping_i32() {
        // i32::MAX + 1 wraps to i32::MIN
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: i32::MAX as i64, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 1, ty: IrType::I32 },
            },
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: i32::MIN as i64, ty: IrType::I32 })
        );
    }

    // -----------------------------------------------------------------------
    // Bitwise folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0xFF, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0x0F, ty: IrType::I32 },
            },
            Instruction::And {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0x0F, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_or_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0xF0, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0x0F, ty: IrType::I32 },
            },
            Instruction::Or {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0xFF, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_xor_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0xFF, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0xFF, ty: IrType::I32 },
            },
            Instruction::Xor {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_shl_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 1, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 4, ty: IrType::I32 },
            },
            Instruction::Shl {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 16, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_shr_logical_constants() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 16, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 2, ty: IrType::I32 },
            },
            Instruction::Shr {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
                is_arithmetic: false,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 4, ty: IrType::I32 })
        );
    }

    // -----------------------------------------------------------------------
    // Comparison folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_icmp_equal_true() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 5, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 5, ty: IrType::I32 },
            },
            Instruction::ICmp {
                result: Value(2),
                op: CompareOp::Equal,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 1, ty: IrType::I1 })
        );
    }

    #[test]
    fn test_icmp_equal_false() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 5, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I32 },
            },
            Instruction::ICmp {
                result: Value(2),
                op: CompareOp::Equal,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0, ty: IrType::I1 })
        );
    }

    #[test]
    fn test_icmp_signed_less() {
        // -1 < 0 signed → true
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: -1, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0, ty: IrType::I32 },
            },
            Instruction::ICmp {
                result: Value(2),
                op: CompareOp::SignedLess,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 1, ty: IrType::I1 })
        );
    }

    #[test]
    fn test_icmp_unsigned_less() {
        // -1 as u64 = u64::MAX, which is NOT less than 0 unsigned.
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: -1, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0, ty: IrType::I64 },
            },
            Instruction::ICmp {
                result: Value(2),
                op: CompareOp::UnsignedLess,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0, ty: IrType::I1 })
        );
    }

    // -----------------------------------------------------------------------
    // Float folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_float_add() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: 1.5, ty: IrType::F64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Float { value: 2.5, ty: IrType::F64 },
            },
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::F64,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Float { value: 4.0, ty: IrType::F64 })
        );
    }

    #[test]
    fn test_float_mul() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: 2.0, ty: IrType::F64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Float { value: 3.0, ty: IrType::F64 },
            },
            Instruction::Mul {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::F64,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Float { value: 6.0, ty: IrType::F64 })
        );
    }

    // -----------------------------------------------------------------------
    // Cast folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cast_trunc() {
        // Trunc i64 256 to i8 → 0 (256 & 0xFF = 0, sign-extended = 0)
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 256, ty: IrType::I64 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::Trunc,
                value: Value(0),
                from_ty: IrType::I64,
                to_ty: IrType::I8,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(1)),
            Some(Constant::Integer { value: 0, ty: IrType::I8 })
        );
    }

    #[test]
    fn test_cast_zext() {
        // ZExt i8 -1 (which is 0xFF=255 unsigned) to i32 → 255
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: -1, ty: IrType::I8 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::ZExt,
                value: Value(0),
                from_ty: IrType::I8,
                to_ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(1)),
            Some(Constant::Integer { value: 255, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_cast_sitofp() {
        // SIToFP i64 42 to f64 → 42.0
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 42, ty: IrType::I64 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::SIToFP,
                value: Value(0),
                from_ty: IrType::I64,
                to_ty: IrType::F64,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(1)),
            Some(Constant::Float { value: 42.0, ty: IrType::F64 })
        );
    }

    // -----------------------------------------------------------------------
    // Branch folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_condbranch_true_folded() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 1, ty: IrType::I1 },
            },
        ];
        let term = Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(1),
            false_block: BlockId(2),
        };
        let mut func = make_function_with_branch(insts, term);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            func.blocks[0].terminator,
            Some(Terminator::Branch { target: BlockId(1) })
        );
    }

    #[test]
    fn test_condbranch_false_folded() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0, ty: IrType::I1 },
            },
        ];
        let term = Terminator::CondBranch {
            condition: Value(0),
            true_block: BlockId(1),
            false_block: BlockId(2),
        };
        let mut func = make_function_with_branch(insts, term);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            func.blocks[0].terminator,
            Some(Terminator::Branch { target: BlockId(2) })
        );
    }

    // -----------------------------------------------------------------------
    // No-change test
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_constants_returns_false() {
        // Function with only non-constant instructions — no folding possible.
        let insts = vec![
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(!pass.run_on_function(&mut func));
    }

    // -----------------------------------------------------------------------
    // Chained folding test
    // -----------------------------------------------------------------------

    #[test]
    fn test_chained_folding() {
        // a = Const(2); b = Const(3); c = Add(a, b); d = Mul(c, Const(4))
        // After folding: c = 5, d = 20
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 2, ty: IrType::I32 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I32 },
            },
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
            Instruction::Const {
                result: Value(3),
                value: Constant::Integer { value: 4, ty: IrType::I32 },
            },
            Instruction::Mul {
                result: Value(4),
                lhs: Value(2),
                rhs: Value(3),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 5, ty: IrType::I32 })
        );
        assert_eq!(
            find_const(&func, Value(4)),
            Some(Constant::Integer { value: 20, ty: IrType::I32 })
        );
    }

    // -----------------------------------------------------------------------
    // Partial constant folding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_by_zero_partial() {
        // Mul(x, 0) should fold to 0 even though x is unknown.
        let insts = vec![
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0, ty: IrType::I32 },
            },
            Instruction::Mul {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_and_with_zero_partial() {
        // And(x, 0) should fold to 0.
        let insts = vec![
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 0, ty: IrType::I32 },
            },
            Instruction::And {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0, ty: IrType::I32 })
        );
    }

    // -----------------------------------------------------------------------
    // Float comparison (NaN) test
    // -----------------------------------------------------------------------

    #[test]
    fn test_fcmp_ordered_with_nan() {
        let nan = f64::NAN;
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: nan, ty: IrType::F64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Float { value: 1.0, ty: IrType::F64 },
            },
            Instruction::FCmp {
                result: Value(2),
                op: FloatCompareOp::OrderedEqual,
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::F64,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        // Ordered comparison with NaN is always false.
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: 0, ty: IrType::I1 })
        );
    }

    // -----------------------------------------------------------------------
    // Switch folding test
    // -----------------------------------------------------------------------

    #[test]
    fn test_switch_folded() {
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 2, ty: IrType::I32 },
            },
        ];
        let term = Terminator::Switch {
            value: Value(0),
            default: BlockId(1),
            cases: vec![(1, BlockId(2)), (2, BlockId(3))],
        };
        // We need 4 blocks: entry(0), default(1), case1(2), case2(3).
        let mut entry = BasicBlock::new(BlockId(0), "entry".to_string());
        entry.instructions = insts;
        entry.terminator = Some(term);
        let mut b1 = BasicBlock::new(BlockId(1), "default".to_string());
        b1.terminator = Some(Terminator::Return { value: None });
        let mut b2 = BasicBlock::new(BlockId(2), "case1".to_string());
        b2.terminator = Some(Terminator::Return { value: None });
        let mut b3 = BasicBlock::new(BlockId(3), "case2".to_string());
        b3.terminator = Some(Terminator::Return { value: None });

        let mut func = Function {
            name: "test_switch".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry, b1, b2, b3],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        // Switch value 2 matches case (2, BlockId(3)).
        assert_eq!(
            func.blocks[0].terminator,
            Some(Terminator::Branch { target: BlockId(3) })
        );
    }

    // -----------------------------------------------------------------------
    // Helper function unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wrap_to_width() {
        assert_eq!(wrap_to_width(256, 8), 0);      // 256 as i8 = 0
        assert_eq!(wrap_to_width(255, 8), -1);      // 255 as i8 = -1
        assert_eq!(wrap_to_width(0x1_0000_0000, 32), 0); // overflow wraps
        assert_eq!(wrap_to_width(-1, 32), -1);       // sign-preserving
    }

    #[test]
    fn test_mask_to_unsigned_width() {
        assert_eq!(mask_to_unsigned_width(-1, 8), 255);
        assert_eq!(mask_to_unsigned_width(-1, 16), 65535);
        assert_eq!(mask_to_unsigned_width(-1, 32), 0xFFFFFFFF);
    }

    #[test]
    fn test_unsigned_div_and_rem() {
        // 0xFFFFFFFE / 2 = 0x7FFFFFFF as unsigned 32-bit
        assert_eq!(unsigned_div(-2, 2, 32), 0x7FFFFFFF);
        // 10 % 3 = 1 as unsigned
        assert_eq!(unsigned_rem(10, 3, 32), 1);
    }

    #[test]
    fn test_arithmetic_vs_logical_shr() {
        // Arithmetic shr of -8 by 2 (i32): -8 >> 2 = -2
        assert_eq!(arithmetic_shr(-8, 2, 32), -2);
        // Logical shr of -8 by 2 (u32): 0xFFFFFFF8 >> 2 = 0x3FFFFFFE
        assert_eq!(logical_shr(-8, 2, 32), 0x3FFFFFFE);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pass_name() {
        let pass = ConstantFoldPass::new();
        assert_eq!(pass.name(), "constant_fold");
    }

    #[test]
    fn test_shr_arithmetic_negative() {
        // Arithmetic shift of -128 >> 3 for i8 = -16
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: -128, ty: IrType::I8 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 3, ty: IrType::I8 },
            },
            Instruction::Shr {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I8,
                is_arithmetic: true,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: -16, ty: IrType::I8 })
        );
    }

    #[test]
    fn test_empty_function_returns_false() {
        let mut func = Function {
            name: "empty".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: Vec::new(),
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };
        let mut pass = ConstantFoldPass::new();
        assert!(!pass.run_on_function(&mut func));
    }

    #[test]
    fn test_cast_fptosi() {
        // FPToSI f64 3.7 to i32 → 3 (truncation toward zero)
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: 3.7, ty: IrType::F64 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::FPToSI,
                value: Value(0),
                from_ty: IrType::F64,
                to_ty: IrType::I32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(1)),
            Some(Constant::Integer { value: 3, ty: IrType::I32 })
        );
    }

    #[test]
    fn test_cast_fptrunc() {
        // FPTrunc f64 1.0 to f32 → 1.0 (exact value preserved)
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Float { value: 1.0, ty: IrType::F64 },
            },
            Instruction::Cast {
                result: Value(1),
                op: CastOp::FPTrunc,
                value: Value(0),
                from_ty: IrType::F64,
                to_ty: IrType::F32,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(1)),
            Some(Constant::Float { value: 1.0, ty: IrType::F32 })
        );
    }

    #[test]
    fn test_switch_default_case() {
        // Switch value doesn't match any case → default
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 99, ty: IrType::I32 },
            },
        ];
        let term = Terminator::Switch {
            value: Value(0),
            default: BlockId(1),
            cases: vec![(1, BlockId(2)), (2, BlockId(3))],
        };
        let mut entry = BasicBlock::new(BlockId(0), "entry".to_string());
        entry.instructions = insts;
        entry.terminator = Some(term);
        let mut b1 = BasicBlock::new(BlockId(1), "default".to_string());
        b1.terminator = Some(Terminator::Return { value: None });
        let mut b2 = BasicBlock::new(BlockId(2), "case1".to_string());
        b2.terminator = Some(Terminator::Return { value: None });
        let mut b3 = BasicBlock::new(BlockId(3), "case2".to_string());
        b3.terminator = Some(Terminator::Return { value: None });

        let mut func = Function {
            name: "test_switch_default".to_string(),
            return_type: IrType::Void,
            params: Vec::new(),
            param_values: Vec::new(),
            blocks: vec![entry, b1, b2, b3],
            entry_block: BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        };

        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            func.blocks[0].terminator,
            Some(Terminator::Branch { target: BlockId(1) })
        );
    }

    #[test]
    fn test_i64_operations() {
        // Large 64-bit operations
        let insts = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: i64::MAX, ty: IrType::I64 },
            },
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 1, ty: IrType::I64 },
            },
            Instruction::Add {
                result: Value(2),
                lhs: Value(0),
                rhs: Value(1),
                ty: IrType::I64,
            },
        ];
        let mut func = make_function(insts);
        let mut pass = ConstantFoldPass::new();
        assert!(pass.run_on_function(&mut func));
        assert_eq!(
            find_const(&func, Value(2)),
            Some(Constant::Integer { value: i64::MIN, ty: IrType::I64 })
        );
    }
}
