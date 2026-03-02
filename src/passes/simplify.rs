//! Algebraic simplification and strength reduction optimization pass.
//!
//! This module implements the [`SimplifyPass`], which performs:
//!
//! - **Identity removal**: Eliminates operations that produce their input unchanged
//!   (e.g., `x + 0 → x`, `x * 1 → x`, `x & -1 → x`)
//! - **Zero absorption**: Replaces operations that always produce zero
//!   (e.g., `x * 0 → 0`, `x & 0 → 0`)
//! - **Self-operation rules**: Simplifies operations where both operands are identical
//!   (e.g., `x - x → 0`, `x ^ x → 0`, `x & x → x`)
//! - **Strength reduction**: Replaces expensive operations with cheaper equivalents
//!   (e.g., `x * 2 → x << 1`, `x / 4 → x >> 2` for unsigned)
//! - **Constant propagation**: Tracks known constant values through the IR
//! - **Boolean simplification**: Simplifies `Select` with constant condition
//! - **Cast elimination**: Removes redundant identity casts
//!
//! # Correctness
//!
//! - Strength reduction (`x * 2^n → x << n`) is only applied to integer types.
//! - Division strength reduction is only applied to **unsigned** division
//!   (signed division has different rounding behavior).
//! - Modulo strength reduction is only applied to **unsigned** modulo.
//! - Float operations are NOT simplified (IEEE 754 edge cases with NaN, ±0, ±Inf).
//!
//! # Performance
//!
//! A single forward pass over all instructions is sufficient. Pattern matching
//! and constant lookups via `HashMap` are O(1) average case.
//!
//! # Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.
//!
//! # Safety
//!
//! This module contains zero `unsafe` blocks. All operations use safe Rust.

use std::collections::HashMap;

use crate::ir::instructions::{Instruction, Value, Constant, CastOp, CompareOp};
use crate::ir::types::IrType;
use crate::ir::cfg::BasicBlock;
use crate::ir::builder::Function;

use super::FunctionPass;

// ---------------------------------------------------------------------------
// SimplifyResult — outcome of attempting to simplify an instruction
// ---------------------------------------------------------------------------

/// Describes how a simplified instruction should replace the original.
///
/// This enum drives the replacement logic in [`SimplifyPass::run_on_function`].
enum SimplifyResult {
    /// Forward all uses of the instruction's result to an existing SSA value.
    ///
    /// The original instruction is replaced with a `Nop`, and a global
    /// use-replacement pass substitutes the old result value with the new one.
    ReplaceWithValue(Value),

    /// Replace the instruction with a constant.
    ///
    /// The original instruction is replaced with `Const { result, value }`,
    /// preserving the original result `Value`.
    ReplaceWithConstant(Constant),

    /// Replace with a single different (simpler) instruction.
    ///
    /// The new instruction must define the same result `Value` as the original.
    ReplaceWithInstruction(Instruction),

    /// Replace with multiple instructions (e.g., new `Const` + `Shl` for strength
    /// reduction). The **last** instruction must define the original result `Value`.
    ReplaceWithInstructions(Vec<Instruction>),
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Returns `Some(log2(n))` if `n` is a positive power of two, `None` otherwise.
///
/// Uses bit manipulation to detect powers of two efficiently: a positive integer
/// `n` is a power of two if and only if `n & (n - 1) == 0`.
///
/// # Examples
///
/// - `is_power_of_two(1)` → `Some(0)`
/// - `is_power_of_two(2)` → `Some(1)`
/// - `is_power_of_two(4)` → `Some(2)`
/// - `is_power_of_two(1024)` → `Some(10)`
/// - `is_power_of_two(3)` → `None`
/// - `is_power_of_two(0)` → `None`
/// - `is_power_of_two(-2)` → `None`
fn is_power_of_two(n: i64) -> Option<u32> {
    if n <= 0 {
        return None;
    }
    let nu = n as u64;
    if nu & (nu - 1) != 0 {
        return None;
    }
    Some(nu.trailing_zeros())
}

/// Extracts the integer value from a constant, treating `Bool` as 0/1.
///
/// Returns `None` for non-integer, non-boolean constants (Float, Null, etc.).
fn get_int_value(c: &Constant) -> Option<i64> {
    match c {
        Constant::Integer { value, .. } => Some(*value),
        Constant::Bool(true) => Some(1),
        Constant::Bool(false) => Some(0),
        _ => None,
    }
}

/// Returns `true` if the constant represents integer zero (or boolean false).
fn is_const_zero(c: &Constant) -> bool {
    match c {
        Constant::Integer { value: 0, .. } => true,
        Constant::Bool(false) => true,
        _ => false,
    }
}

/// Returns `true` if the constant represents integer one (or boolean true).
fn is_const_one(c: &Constant) -> bool {
    match c {
        Constant::Integer { value: 1, .. } => true,
        Constant::Bool(true) => true,
        _ => false,
    }
}

/// Returns `true` if the constant represents an all-ones bit pattern for its type.
///
/// For an N-bit integer, the all-ones value is `2^N - 1` (unsigned) or `-1` (signed).
/// This is the neutral element for bitwise AND.
///
/// Handles both sign-extended representation (value == -1 as i64) and
/// zero-extended representation (value == 0xFFFF_FFFF for i32) to be safe
/// regardless of how the IR builder encodes constants.
fn is_const_all_ones(c: &Constant) -> bool {
    match c {
        Constant::Integer { value, ty } => {
            // Quick check: -1 in two's complement is all ones for any width.
            if *value == -1 {
                return true;
            }
            // Check against the width-specific all-ones mask.
            if let Some(bits) = ty.integer_bit_width() {
                if bits >= 64 {
                    *value == -1
                } else {
                    let mask = (1i64 << bits) - 1;
                    (*value & mask) == mask
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Computes the next available `Value` index by scanning all instructions,
/// phi nodes, and their operands in the function.
///
/// This is needed when strength reduction creates new `Const` instructions
/// that require fresh `Value` identifiers.
fn compute_next_value(function: &Function) -> u32 {
    let mut max_val: u32 = 0;
    for block in &function.blocks {
        max_val = max_val.max(block.id.0);
        for phi in &block.phi_nodes {
            max_val = max_val.max(phi.result.0);
            for (v, _) in &phi.incoming {
                max_val = max_val.max(v.0);
            }
        }
        for inst in &block.instructions {
            if let Some(v) = inst.result() {
                max_val = max_val.max(v.0);
            }
            for op in inst.operands() {
                max_val = max_val.max(op.0);
            }
        }
    }
    max_val.saturating_add(1)
}

// ---------------------------------------------------------------------------
// SimplifyPass — the public pass struct
// ---------------------------------------------------------------------------

/// Algebraic simplification and strength reduction optimization pass.
///
/// Applies peephole optimizations that replace instructions with simpler
/// equivalents based on algebraic identities and known constant values.
/// Runs as part of the `-O2` pipeline, typically after CSE and before DCE.
///
/// # Usage
///
/// ```ignore
/// let mut pass = SimplifyPass::new();
/// let changed = pass.run_on_function(&mut function);
/// ```
pub struct SimplifyPass;

impl SimplifyPass {
    /// Creates a new simplification pass instance.
    pub fn new() -> Self {
        SimplifyPass
    }
}

// ---------------------------------------------------------------------------
// FunctionPass implementation
// ---------------------------------------------------------------------------

impl FunctionPass for SimplifyPass {
    /// Returns the human-readable pass name for diagnostics and debugging.
    fn name(&self) -> &str {
        "simplify"
    }

    /// Runs the algebraic simplification pass on a single function.
    ///
    /// Iterates through all basic blocks and instructions in forward order,
    /// maintaining a map of known constant values. For each instruction,
    /// attempts to apply simplification rules (identity removal, zero
    /// absorption, self-operation, strength reduction, constant propagation,
    /// boolean simplification, and cast elimination).
    ///
    /// # Returns
    ///
    /// `true` if any simplifications were applied, `false` if the function
    /// was unchanged. The return value drives the `-O2` fixed-point iteration
    /// loop — if no pass reports changes, the loop terminates.
    fn run_on_function(&mut self, function: &mut Function) -> bool {
        // Map from SSA Values to their known Constant definitions.
        let mut constants: HashMap<Value, Constant> = HashMap::new();
        // Map from replaced result Values to their forwarding target Values.
        let mut replacements: HashMap<Value, Value> = HashMap::new();
        let mut changed = false;
        // Counter for allocating fresh Value IDs (used by strength reduction).
        let mut next_value = compute_next_value(function);

        // Forward pass through all blocks and instructions.
        for block_idx in 0..function.blocks.len() {
            // Take ownership of the instruction list to allow in-place rebuilding.
            let instructions = std::mem::take(&mut function.blocks[block_idx].instructions);
            let mut new_instructions = Vec::with_capacity(instructions.len() + 4);

            for mut inst in instructions {
                // Apply any pending value replacements to this instruction's operands
                // so that chained simplifications work within a single pass.
                for (&old_val, &new_val) in &replacements {
                    inst.replace_use(old_val, new_val);
                }

                // Register constants from Const instructions for downstream lookups.
                if let Instruction::Const { result, ref value } = inst {
                    constants.insert(result, value.clone());
                }

                // Attempt simplification of this instruction.
                match try_simplify_instruction(&inst, &constants, &mut next_value) {
                    Some(SimplifyResult::ReplaceWithValue(val)) => {
                        // Forward all uses of the original result to `val`.
                        if let Some(res) = inst.result() {
                            replacements.insert(res, resolve_replacement(&replacements, val));
                        }
                        new_instructions.push(Instruction::Nop);
                        changed = true;
                    }
                    Some(SimplifyResult::ReplaceWithConstant(c)) => {
                        if let Some(res) = inst.result() {
                            constants.insert(res, c.clone());
                            new_instructions.push(Instruction::Const {
                                result: res,
                                value: c,
                            });
                        } else {
                            // Should not happen for simplifiable instructions, but be safe.
                            new_instructions.push(inst);
                        }
                        changed = true;
                    }
                    Some(SimplifyResult::ReplaceWithInstruction(new_inst)) => {
                        // Register if the replacement instruction defines a constant.
                        if let Instruction::Const { result, ref value } = new_inst {
                            constants.insert(result, value.clone());
                        }
                        new_instructions.push(new_inst);
                        changed = true;
                    }
                    Some(SimplifyResult::ReplaceWithInstructions(insts)) => {
                        for new_inst in insts {
                            if let Instruction::Const { result, ref value } = new_inst {
                                constants.insert(result, value.clone());
                            }
                            new_instructions.push(new_inst);
                        }
                        changed = true;
                    }
                    None => {
                        // No simplification — keep the original instruction.
                        new_instructions.push(inst);
                    }
                }
            }

            function.blocks[block_idx].instructions = new_instructions;
        }

        // Apply pending value replacements globally to phi nodes and terminators.
        if !replacements.is_empty() {
            apply_replacements(function, &replacements);
        }

        changed
    }
}

/// Resolves a value through the replacement chain to its final target.
///
/// If `val` itself has been replaced, follow the chain to find the ultimate
/// target value. This prevents stale references when simplifications chain
/// (e.g., `%5 → %3` and then `%3 → %1` should resolve `%5 → %1`).
fn resolve_replacement(replacements: &HashMap<Value, Value>, val: Value) -> Value {
    let mut current = val;
    // Follow the chain with a depth limit to prevent infinite loops in case
    // of a bug creating a cycle in the replacement map.
    for _ in 0..64 {
        if let Some(&next) = replacements.get(&current) {
            current = next;
        } else {
            break;
        }
    }
    current
}

/// Applies value replacements to all phi nodes and terminators in the function.
///
/// This handles cross-block references that were not caught during the
/// per-instruction forward pass (phi node incoming values and terminator
/// condition/return/switch values).
fn apply_replacements(function: &mut Function, replacements: &HashMap<Value, Value>) {
    use crate::ir::cfg::Terminator;

    for block in &mut function.blocks {
        // Update phi node incoming values.
        for phi in &mut block.phi_nodes {
            for (val, _) in &mut phi.incoming {
                if let Some(&new_val) = replacements.get(val) {
                    *val = new_val;
                }
            }
        }

        // Update instruction operands (catches any remaining cross-block refs).
        for inst in &mut block.instructions {
            for (&old_val, &new_val) in replacements {
                inst.replace_use(old_val, new_val);
            }
        }

        // Update terminator operands.
        if let Some(ref mut term) = block.terminator {
            match term {
                Terminator::CondBranch { condition, .. } => {
                    if let Some(&new_val) = replacements.get(condition) {
                        *condition = new_val;
                    }
                }
                Terminator::Return { value: Some(v) } => {
                    if let Some(&new_val) = replacements.get(v) {
                        *v = new_val;
                    }
                }
                Terminator::Switch { value, .. } => {
                    if let Some(&new_val) = replacements.get(value) {
                        *value = new_val;
                    }
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core simplification logic
// ---------------------------------------------------------------------------

/// Attempts to simplify a single instruction using algebraic identities,
/// strength reduction, and constant propagation.
///
/// Returns `Some(SimplifyResult)` if the instruction can be simplified,
/// or `None` if no simplification applies.
///
/// # Parameters
///
/// - `inst`: The instruction to attempt to simplify.
/// - `constants`: Map from SSA values to their known constant definitions.
/// - `next_value`: Mutable counter for allocating new `Value` IDs when strength
///   reduction requires inserting new constant instructions.
fn try_simplify_instruction(
    inst: &Instruction,
    constants: &HashMap<Value, Constant>,
    next_value: &mut u32,
) -> Option<SimplifyResult> {
    match inst {
        // =================================================================
        // Add: identity (x+0, 0+x)
        // =================================================================
        Instruction::Add { result: _, lhs, rhs, ty } => {
            // Only simplify integer adds (float +0 has signed-zero issues under IEEE 754).
            if !ty.is_integer() {
                return None;
            }
            // Add(x, 0) → x
            if let Some(c) = constants.get(rhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            // Add(0, x) → x
            if let Some(c) = constants.get(lhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*rhs));
                }
            }
            None
        }

        // =================================================================
        // Sub: identity (x-0), self-operation (x-x → 0)
        // =================================================================
        Instruction::Sub { result: _, lhs, rhs, ty } => {
            if !ty.is_integer() {
                return None;
            }
            // Sub(x, x) → 0
            if lhs == rhs {
                return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                    value: 0,
                    ty: ty.clone(),
                }));
            }
            // Sub(x, 0) → x
            if let Some(c) = constants.get(rhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            None
        }

        // =================================================================
        // Mul: identity (x*1, 1*x), zero absorption (x*0, 0*x),
        //      strength reduction (x*2^n → x<<n)
        // =================================================================
        Instruction::Mul { result, lhs, rhs, ty } => {
            if !ty.is_integer() {
                return None;
            }

            // Check rhs constant first.
            if let Some(c) = constants.get(rhs) {
                // Mul(x, 0) → 0 (zero absorption)
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                        value: 0,
                        ty: ty.clone(),
                    }));
                }
                // Mul(x, 1) → x (identity)
                if is_const_one(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
                // Mul(x, 2^n) → Shl(x, n) (strength reduction)
                if let Some(iv) = get_int_value(c) {
                    if let Some(log2) = is_power_of_two(iv) {
                        return Some(make_shift_left_replacement(
                            *result, *lhs, log2, ty, next_value,
                        ));
                    }
                }
            }

            // Check lhs constant (commutative).
            if let Some(c) = constants.get(lhs) {
                // Mul(0, x) → 0
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                        value: 0,
                        ty: ty.clone(),
                    }));
                }
                // Mul(1, x) → x
                if is_const_one(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*rhs));
                }
                // Mul(2^n, x) → Shl(x, n)
                if let Some(iv) = get_int_value(c) {
                    if let Some(log2) = is_power_of_two(iv) {
                        return Some(make_shift_left_replacement(
                            *result, *rhs, log2, ty, next_value,
                        ));
                    }
                }
            }

            None
        }

        // =================================================================
        // Div: identity (x/1), self-operation (x/x → 1),
        //      unsigned strength reduction (x/2^n → x>>n)
        // =================================================================
        Instruction::Div { result, lhs, rhs, ty, is_signed } => {
            if !ty.is_integer() {
                return None;
            }
            // Div(x, 1) → x (valid for both signed and unsigned)
            if let Some(c) = constants.get(rhs) {
                if is_const_one(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
                // Unsigned Div(x, 2^n) → Shr(x, n, is_arithmetic=false)
                // CRITICAL: Only for unsigned division! Signed division rounds toward
                // zero, while arithmetic right shift rounds toward negative infinity.
                if !is_signed {
                    if let Some(iv) = get_int_value(c) {
                        if let Some(log2) = is_power_of_two(iv) {
                            let shift_val = Value(*next_value);
                            *next_value += 1;
                            let const_inst = Instruction::Const {
                                result: shift_val,
                                value: Constant::Integer {
                                    value: log2 as i64,
                                    ty: ty.clone(),
                                },
                            };
                            let shr_inst = Instruction::Shr {
                                result: *result,
                                lhs: *lhs,
                                rhs: shift_val,
                                ty: ty.clone(),
                                is_arithmetic: false,
                            };
                            return Some(SimplifyResult::ReplaceWithInstructions(vec![
                                const_inst, shr_inst,
                            ]));
                        }
                    }
                }
            }
            // Div(x, x) → 1 (assuming x != 0; division by zero is UB in C)
            if lhs == rhs {
                return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                    value: 1,
                    ty: ty.clone(),
                }));
            }
            None
        }

        // =================================================================
        // Mod: unsigned strength reduction (x%2^n → x & (2^n - 1))
        // =================================================================
        Instruction::Mod { result, lhs, rhs, ty, is_signed } => {
            if !ty.is_integer() {
                return None;
            }
            // Unsigned Mod(x, 2^n) → And(x, 2^n - 1)
            // CRITICAL: Only for unsigned modulo! Signed modulo has different
            // semantics (result sign follows the dividend in C).
            if !is_signed {
                if let Some(c) = constants.get(rhs) {
                    if let Some(iv) = get_int_value(c) {
                        if let Some(_log2) = is_power_of_two(iv) {
                            let mask_value = iv - 1;
                            let mask_val = Value(*next_value);
                            *next_value += 1;
                            let const_inst = Instruction::Const {
                                result: mask_val,
                                value: Constant::Integer {
                                    value: mask_value,
                                    ty: ty.clone(),
                                },
                            };
                            let and_inst = Instruction::And {
                                result: *result,
                                lhs: *lhs,
                                rhs: mask_val,
                                ty: ty.clone(),
                            };
                            return Some(SimplifyResult::ReplaceWithInstructions(vec![
                                const_inst, and_inst,
                            ]));
                        }
                    }
                }
            }
            None
        }

        // =================================================================
        // And: identity (x & -1), zero absorption (x & 0),
        //      self-operation (x & x → x)
        // =================================================================
        Instruction::And { result: _, lhs, rhs, ty } => {
            if !ty.is_integer() {
                return None;
            }
            // Self-operation: And(x, x) → x (idempotent)
            if lhs == rhs {
                return Some(SimplifyResult::ReplaceWithValue(*lhs));
            }
            // Check rhs constant.
            if let Some(c) = constants.get(rhs) {
                // And(x, 0) → 0
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                        value: 0,
                        ty: ty.clone(),
                    }));
                }
                // And(x, all_ones) → x
                if is_const_all_ones(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            // Check lhs constant (commutative).
            if let Some(c) = constants.get(lhs) {
                // And(0, x) → 0
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                        value: 0,
                        ty: ty.clone(),
                    }));
                }
                // And(all_ones, x) → x
                if is_const_all_ones(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*rhs));
                }
            }
            None
        }

        // =================================================================
        // Or: identity (x | 0), self-operation (x | x → x)
        // =================================================================
        Instruction::Or { result: _, lhs, rhs, ty } => {
            if !ty.is_integer() {
                return None;
            }
            // Self-operation: Or(x, x) → x (idempotent)
            if lhs == rhs {
                return Some(SimplifyResult::ReplaceWithValue(*lhs));
            }
            // Or(x, 0) → x
            if let Some(c) = constants.get(rhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            // Or(0, x) → x
            if let Some(c) = constants.get(lhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*rhs));
                }
            }
            None
        }

        // =================================================================
        // Xor: identity (x ^ 0), self-operation (x ^ x → 0)
        // =================================================================
        Instruction::Xor { result: _, lhs, rhs, ty } => {
            if !ty.is_integer() {
                return None;
            }
            // Self-operation: Xor(x, x) → 0
            if lhs == rhs {
                return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                    value: 0,
                    ty: ty.clone(),
                }));
            }
            // Xor(x, 0) → x
            if let Some(c) = constants.get(rhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            // Xor(0, x) → x
            if let Some(c) = constants.get(lhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*rhs));
                }
            }
            None
        }

        // =================================================================
        // Shl: identity (x << 0)
        // =================================================================
        Instruction::Shl { result: _, lhs, rhs, ty } => {
            if !ty.is_integer() {
                return None;
            }
            // Shl(x, 0) → x
            if let Some(c) = constants.get(rhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            None
        }

        // =================================================================
        // Shr: identity (x >> 0)
        // =================================================================
        Instruction::Shr { result: _, lhs, rhs, ty, .. } => {
            if !ty.is_integer() {
                return None;
            }
            // Shr(x, 0) → x
            if let Some(c) = constants.get(rhs) {
                if is_const_zero(c) {
                    return Some(SimplifyResult::ReplaceWithValue(*lhs));
                }
            }
            None
        }

        // =================================================================
        // ICmp: constant operand simplifications
        // =================================================================
        Instruction::ICmp { result: _, op, lhs, rhs, ty } => {
            // ICmp(Equal, x, x) → true (any value equals itself)
            if *op == CompareOp::Equal && lhs == rhs {
                return Some(SimplifyResult::ReplaceWithConstant(Constant::Integer {
                    value: 1,
                    ty: IrType::I1,
                }));
            }
            None
        }

        // =================================================================
        // Select: constant condition
        // =================================================================
        Instruction::Select { result: _, condition, true_val, false_val, ty: _ } => {
            if let Some(c) = constants.get(condition) {
                match c {
                    Constant::Integer { value, .. } => {
                        if *value != 0 {
                            return Some(SimplifyResult::ReplaceWithValue(*true_val));
                        } else {
                            return Some(SimplifyResult::ReplaceWithValue(*false_val));
                        }
                    }
                    Constant::Bool(b) => {
                        if *b {
                            return Some(SimplifyResult::ReplaceWithValue(*true_val));
                        } else {
                            return Some(SimplifyResult::ReplaceWithValue(*false_val));
                        }
                    }
                    _ => {}
                }
            }
            None
        }

        // =================================================================
        // Cast: identity cast (from_ty == to_ty)
        // =================================================================
        Instruction::Cast { result: _, op: _, value, from_ty, to_ty } => {
            // A cast to the same type is a no-op.
            if from_ty == to_ty {
                return Some(SimplifyResult::ReplaceWithValue(*value));
            }
            None
        }

        // All other instruction types: no simplification applies.
        _ => None,
    }
}

/// Creates a `SimplifyResult` that replaces a multiply-by-power-of-two with
/// a left shift instruction.
///
/// Emits two instructions:
/// 1. A `Const` instruction defining the shift amount (`log2_val`).
/// 2. A `Shl` instruction using the operand and the shift amount constant.
///
/// The `Shl` instruction preserves the original `result` value so that all
/// downstream uses remain valid.
fn make_shift_left_replacement(
    result: Value,
    operand: Value,
    log2_val: u32,
    ty: &IrType,
    next_value: &mut u32,
) -> SimplifyResult {
    let shift_val = Value(*next_value);
    *next_value += 1;
    let const_inst = Instruction::Const {
        result: shift_val,
        value: Constant::Integer {
            value: log2_val as i64,
            ty: ty.clone(),
        },
    };
    let shl_inst = Instruction::Shl {
        result,
        lhs: operand,
        rhs: shift_val,
        ty: ty.clone(),
    };
    SimplifyResult::ReplaceWithInstructions(vec![const_inst, shl_inst])
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::{BlockId, Instruction, Value, Constant};
    use crate::ir::types::IrType;
    use crate::ir::cfg::{BasicBlock, Terminator};
    use crate::ir::builder::Function;

    /// Helper: create a minimal function with given instructions in a single block
    /// and a void return terminator.
    fn make_function(instructions: Vec<Instruction>) -> Function {
        let mut block = BasicBlock::new(BlockId(0), "entry".to_string());
        block.instructions = instructions;
        block.terminator = Some(Terminator::Return { value: None });
        Function {
            name: "test_fn".to_string(),
            return_type: IrType::Void,
            params: vec![],
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: BlockId(0),
            is_definition: true,
        }
    }

    // -----------------------------------------------------------------------
    // is_power_of_two tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_power_of_two_one() {
        assert_eq!(is_power_of_two(1), Some(0));
    }

    #[test]
    fn test_power_of_two_two() {
        assert_eq!(is_power_of_two(2), Some(1));
    }

    #[test]
    fn test_power_of_two_four() {
        assert_eq!(is_power_of_two(4), Some(2));
    }

    #[test]
    fn test_power_of_two_1024() {
        assert_eq!(is_power_of_two(1024), Some(10));
    }

    #[test]
    fn test_power_of_two_three_is_none() {
        assert_eq!(is_power_of_two(3), None);
    }

    #[test]
    fn test_power_of_two_zero_is_none() {
        assert_eq!(is_power_of_two(0), None);
    }

    #[test]
    fn test_power_of_two_negative_is_none() {
        assert_eq!(is_power_of_two(-2), None);
    }

    #[test]
    fn test_power_of_two_large() {
        assert_eq!(is_power_of_two(1 << 20), Some(20));
        assert_eq!(is_power_of_two(1 << 30), Some(30));
    }

    // -----------------------------------------------------------------------
    // Identity removal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_x_zero_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Add {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue(Value(0))"),
        }
    }

    #[test]
    fn test_add_zero_x_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(0), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Add {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(1)),
            _ => panic!("Expected ReplaceWithValue(Value(1))"),
        }
    }

    #[test]
    fn test_sub_x_zero_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Sub {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_mul_x_one_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 1, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_div_x_one_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 1, ty: IrType::I32 });
        let inst = Instruction::Div {
            result: Value(2), lhs: Value(0), rhs: Value(1),
            ty: IrType::I32, is_signed: true,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_and_x_all_ones_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: -1, ty: IrType::I32 });
        let inst = Instruction::And {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_or_x_zero_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Or {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_xor_x_zero_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Xor {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_shl_x_zero_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Shl {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_shr_x_zero_identity() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Shr {
            result: Value(2), lhs: Value(0), rhs: Value(1),
            ty: IrType::I32, is_arithmetic: false,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    // -----------------------------------------------------------------------
    // Zero absorption tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_x_zero_absorption() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 0, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(0)"),
        }
    }

    #[test]
    fn test_mul_zero_x_absorption() {
        let mut constants = HashMap::new();
        constants.insert(Value(0), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 0, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(0)"),
        }
    }

    #[test]
    fn test_and_x_zero_absorption() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::And {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 0, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(0)"),
        }
    }

    #[test]
    fn test_and_zero_x_absorption() {
        let mut constants = HashMap::new();
        constants.insert(Value(0), Constant::Integer { value: 0, ty: IrType::I32 });
        let inst = Instruction::And {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 0, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(0)"),
        }
    }

    // -----------------------------------------------------------------------
    // Self-operation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sub_x_x_is_zero() {
        let constants = HashMap::new();
        let inst = Instruction::Sub {
            result: Value(2), lhs: Value(0), rhs: Value(0), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 0, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(0)"),
        }
    }

    #[test]
    fn test_xor_x_x_is_zero() {
        let constants = HashMap::new();
        let inst = Instruction::Xor {
            result: Value(2), lhs: Value(0), rhs: Value(0), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 0, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(0)"),
        }
    }

    #[test]
    fn test_and_x_x_is_x() {
        let constants = HashMap::new();
        let inst = Instruction::And {
            result: Value(2), lhs: Value(0), rhs: Value(0), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_or_x_x_is_x() {
        let constants = HashMap::new();
        let inst = Instruction::Or {
            result: Value(2), lhs: Value(0), rhs: Value(0), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue"),
        }
    }

    #[test]
    fn test_div_x_x_is_one() {
        let constants = HashMap::new();
        let inst = Instruction::Div {
            result: Value(2), lhs: Value(0), rhs: Value(0),
            ty: IrType::I32, is_signed: false,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 1, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(1)"),
        }
    }

    // -----------------------------------------------------------------------
    // Strength reduction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_x_two_to_shl() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 2, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithInstructions(insts)) => {
                assert_eq!(insts.len(), 2);
                // First: Const for shift amount 1
                match &insts[0] {
                    Instruction::Const { value: Constant::Integer { value: 1, .. }, .. } => {}
                    other => panic!("Expected Const(1), got {:?}", other),
                }
                // Second: Shl with original result Value(2)
                match &insts[1] {
                    Instruction::Shl { result, lhs, .. } => {
                        assert_eq!(*result, Value(2));
                        assert_eq!(*lhs, Value(0));
                    }
                    other => panic!("Expected Shl, got {:?}", other),
                }
            }
            other => panic!("Expected ReplaceWithInstructions, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn test_mul_x_four_to_shl() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 4, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithInstructions(insts)) => {
                assert_eq!(insts.len(), 2);
                match &insts[0] {
                    Instruction::Const { value: Constant::Integer { value: 2, .. }, .. } => {}
                    _ => panic!("Expected Const(2)"),
                }
            }
            _ => panic!("Expected ReplaceWithInstructions"),
        }
    }

    #[test]
    fn test_mul_x_sixteen_to_shl() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 16, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithInstructions(insts)) => {
                assert_eq!(insts.len(), 2);
                match &insts[0] {
                    Instruction::Const { value: Constant::Integer { value: 4, .. }, .. } => {}
                    _ => panic!("Expected Const(4)"),
                }
            }
            _ => panic!("Expected ReplaceWithInstructions"),
        }
    }

    #[test]
    fn test_unsigned_div_power_of_two_to_shr() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 4, ty: IrType::I32 });
        let inst = Instruction::Div {
            result: Value(2), lhs: Value(0), rhs: Value(1),
            ty: IrType::I32, is_signed: false,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithInstructions(insts)) => {
                assert_eq!(insts.len(), 2);
                // First: Const for shift amount 2
                match &insts[0] {
                    Instruction::Const { value: Constant::Integer { value: 2, .. }, .. } => {}
                    _ => panic!("Expected Const(2)"),
                }
                // Second: Shr (logical, not arithmetic)
                match &insts[1] {
                    Instruction::Shr { result, lhs, is_arithmetic, .. } => {
                        assert_eq!(*result, Value(2));
                        assert_eq!(*lhs, Value(0));
                        assert!(!is_arithmetic, "Expected logical shift for unsigned div");
                    }
                    other => panic!("Expected Shr, got {:?}", other),
                }
            }
            _ => panic!("Expected ReplaceWithInstructions"),
        }
    }

    #[test]
    fn test_signed_div_power_of_two_not_simplified() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 4, ty: IrType::I32 });
        let inst = Instruction::Div {
            result: Value(2), lhs: Value(0), rhs: Value(1),
            ty: IrType::I32, is_signed: true,
        };
        let mut nv = 10u32;
        // Signed division by power of two should NOT be strength-reduced
        // because signed division rounds toward zero, while arithmetic
        // right shift rounds toward negative infinity.
        let result = try_simplify_instruction(&inst, &constants, &mut nv);
        assert!(result.is_none(), "Signed div by power of 2 should not be simplified");
    }

    #[test]
    fn test_unsigned_mod_power_of_two_to_and() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 8, ty: IrType::I32 });
        let inst = Instruction::Mod {
            result: Value(2), lhs: Value(0), rhs: Value(1),
            ty: IrType::I32, is_signed: false,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithInstructions(insts)) => {
                assert_eq!(insts.len(), 2);
                // First: Const for mask value 7 (8 - 1)
                match &insts[0] {
                    Instruction::Const { value: Constant::Integer { value: 7, .. }, .. } => {}
                    _ => panic!("Expected Const(7)"),
                }
                // Second: And(x, 7)
                match &insts[1] {
                    Instruction::And { result, lhs, .. } => {
                        assert_eq!(*result, Value(2));
                        assert_eq!(*lhs, Value(0));
                    }
                    other => panic!("Expected And, got {:?}", other),
                }
            }
            _ => panic!("Expected ReplaceWithInstructions"),
        }
    }

    #[test]
    fn test_mul_x_three_not_simplified() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Integer { value: 3, ty: IrType::I32 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
        };
        let mut nv = 10u32;
        // 3 is not a power of two, so no strength reduction.
        let result = try_simplify_instruction(&inst, &constants, &mut nv);
        assert!(result.is_none(), "Mul by 3 should not be simplified");
    }

    // -----------------------------------------------------------------------
    // Select simplification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_const_true() {
        let mut constants = HashMap::new();
        constants.insert(Value(0), Constant::Integer { value: 1, ty: IrType::I1 });
        let inst = Instruction::Select {
            result: Value(3), condition: Value(0),
            true_val: Value(1), false_val: Value(2), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(1)),
            _ => panic!("Expected ReplaceWithValue(true_val)"),
        }
    }

    #[test]
    fn test_select_const_false() {
        let mut constants = HashMap::new();
        constants.insert(Value(0), Constant::Integer { value: 0, ty: IrType::I1 });
        let inst = Instruction::Select {
            result: Value(3), condition: Value(0),
            true_val: Value(1), false_val: Value(2), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(2)),
            _ => panic!("Expected ReplaceWithValue(false_val)"),
        }
    }

    #[test]
    fn test_select_bool_true() {
        let mut constants = HashMap::new();
        constants.insert(Value(0), Constant::Bool(true));
        let inst = Instruction::Select {
            result: Value(3), condition: Value(0),
            true_val: Value(1), false_val: Value(2), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(1)),
            _ => panic!("Expected ReplaceWithValue(true_val)"),
        }
    }

    // -----------------------------------------------------------------------
    // Cast simplification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cast_same_type_identity() {
        let constants = HashMap::new();
        let inst = Instruction::Cast {
            result: Value(2), op: CastOp::ZExt, value: Value(0),
            from_ty: IrType::I32, to_ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithValue(v)) => assert_eq!(v, Value(0)),
            _ => panic!("Expected ReplaceWithValue for identity cast"),
        }
    }

    #[test]
    fn test_cast_different_type_not_simplified() {
        let constants = HashMap::new();
        let inst = Instruction::Cast {
            result: Value(2), op: CastOp::ZExt, value: Value(0),
            from_ty: IrType::I16, to_ty: IrType::I32,
        };
        let mut nv = 10u32;
        let result = try_simplify_instruction(&inst, &constants, &mut nv);
        assert!(result.is_none(), "Cast with different types should not be simplified");
    }

    // -----------------------------------------------------------------------
    // ICmp simplification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_icmp_equal_same_value_is_true() {
        let constants = HashMap::new();
        let inst = Instruction::ICmp {
            result: Value(2), op: CompareOp::Equal,
            lhs: Value(0), rhs: Value(0), ty: IrType::I32,
        };
        let mut nv = 10u32;
        match try_simplify_instruction(&inst, &constants, &mut nv) {
            Some(SimplifyResult::ReplaceWithConstant(Constant::Integer { value: 1, .. })) => {}
            _ => panic!("Expected ReplaceWithConstant(1) for x == x"),
        }
    }

    // -----------------------------------------------------------------------
    // No-change test (pass returns false when nothing simplifiable)
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_simplifiable_instructions_returns_false() {
        // Create a function with only a non-simplifiable Add (no constant operands).
        let instructions = vec![
            Instruction::Add {
                result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
            },
        ];
        let mut func = make_function(instructions);
        let mut pass = SimplifyPass::new();
        let changed = pass.run_on_function(&mut func);
        assert!(!changed, "Pass should return false when no simplifications apply");
    }

    // -----------------------------------------------------------------------
    // Changed flag test (pass returns true when simplifications occur)
    // -----------------------------------------------------------------------

    #[test]
    fn test_simplifiable_instructions_returns_true() {
        // %0 = const i32 0
        // %1 = some value (pretend it's a function parameter)
        // %2 = add i32 %1, %0  → should simplify to %1
        let instructions = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0, ty: IrType::I32 },
            },
            Instruction::Add {
                result: Value(2), lhs: Value(1), rhs: Value(0), ty: IrType::I32,
            },
        ];
        let mut func = make_function(instructions);
        let mut pass = SimplifyPass::new();
        let changed = pass.run_on_function(&mut func);
        assert!(changed, "Pass should return true when simplifications apply");
    }

    // -----------------------------------------------------------------------
    // Full pass integration test: verify instruction replacement
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_pass_add_zero_eliminated() {
        // Build: %0 = const 0; %2 = add %1, %0; ret %2
        // Expected after: %2 forwarded to %1
        let instructions = vec![
            Instruction::Const {
                result: Value(0),
                value: Constant::Integer { value: 0, ty: IrType::I32 },
            },
            Instruction::Add {
                result: Value(2), lhs: Value(1), rhs: Value(0), ty: IrType::I32,
            },
        ];
        let mut block = BasicBlock::new(BlockId(0), "entry".to_string());
        block.instructions = instructions;
        block.terminator = Some(Terminator::Return { value: Some(Value(2)) });
        let mut func = Function {
            name: "test".to_string(),
            return_type: IrType::I32,
            params: vec![("x".to_string(), IrType::I32)],
            param_values: Vec::new(),
            blocks: vec![block],
            entry_block: BlockId(0),
            is_definition: true,
        };

        let mut pass = SimplifyPass::new();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        // The return should now use Value(1) instead of Value(2).
        match &func.blocks[0].terminator {
            Some(Terminator::Return { value: Some(v) }) => {
                assert_eq!(*v, Value(1), "Return should use forwarded value");
            }
            _ => panic!("Expected Return terminator"),
        }
    }

    #[test]
    fn test_full_pass_mul_strength_reduction() {
        // Build: %1 = const 8; %2 = mul %0, %1
        // Expected: %2 = shl %0, <const 3>
        let instructions = vec![
            Instruction::Const {
                result: Value(1),
                value: Constant::Integer { value: 8, ty: IrType::I32 },
            },
            Instruction::Mul {
                result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I32,
            },
        ];
        let mut func = make_function(instructions);
        let mut pass = SimplifyPass::new();
        let changed = pass.run_on_function(&mut func);
        assert!(changed);

        // Check that we now have a Const + Shl where the Mul used to be.
        let block_insts = &func.blocks[0].instructions;
        // Should have: Const(Value(1), 8), Const(new_val, 3), Shl(Value(2), Value(0), new_val)
        let has_shl = block_insts.iter().any(|inst| matches!(inst, Instruction::Shl { .. }));
        assert!(has_shl, "Should contain a Shl instruction after strength reduction");
    }

    // -----------------------------------------------------------------------
    // Float operations should NOT be simplified
    // -----------------------------------------------------------------------

    #[test]
    fn test_float_add_zero_not_simplified() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Float { value: 0.0, ty: IrType::F32 });
        let inst = Instruction::Add {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::F32,
        };
        let mut nv = 10u32;
        let result = try_simplify_instruction(&inst, &constants, &mut nv);
        assert!(result.is_none(), "Float add x+0 should not be simplified (IEEE 754 edge cases)");
    }

    #[test]
    fn test_float_mul_not_strength_reduced() {
        let mut constants = HashMap::new();
        constants.insert(Value(1), Constant::Float { value: 2.0, ty: IrType::F64 });
        let inst = Instruction::Mul {
            result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::F64,
        };
        let mut nv = 10u32;
        let result = try_simplify_instruction(&inst, &constants, &mut nv);
        assert!(result.is_none(), "Float mul should not be strength-reduced to shift");
    }

    // -----------------------------------------------------------------------
    // is_const_all_ones helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_ones_i32_minus_one() {
        let c = Constant::Integer { value: -1, ty: IrType::I32 };
        assert!(is_const_all_ones(&c));
    }

    #[test]
    fn test_all_ones_i64_minus_one() {
        let c = Constant::Integer { value: -1, ty: IrType::I64 };
        assert!(is_const_all_ones(&c));
    }

    #[test]
    fn test_all_ones_i8_0xff() {
        let c = Constant::Integer { value: 0xFF, ty: IrType::I8 };
        assert!(is_const_all_ones(&c));
    }

    #[test]
    fn test_not_all_ones() {
        let c = Constant::Integer { value: 42, ty: IrType::I32 };
        assert!(!is_const_all_ones(&c));
    }

    #[test]
    fn test_not_all_ones_float() {
        let c = Constant::Float { value: -1.0, ty: IrType::F64 };
        assert!(!is_const_all_ones(&c));
    }
}
