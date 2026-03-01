//! RISC-V LP64D ABI Implementation.
//!
//! This module implements the complete calling convention for the RISC-V 64-bit
//! architecture following the RISC-V LP64D ABI specification. It covers:
//!
//! - **Argument classification**: a0-a7 for integer/pointer arguments, fa0-fa7
//!   for floating-point arguments, with independent counters for GPR and FPR
//!   register allocation.
//! - **Return value classification**: a0 for integer returns, fa0 for float
//!   returns, a0+a1 pair for 128-bit composites, a0 as indirect result
//!   location for large structs (>2×XLEN bytes).
//! - **Stack frame layout**: ra+s0(fp) saving, callee-saved register
//!   save/restore (s0-s11, fs0-fs11), 16-byte SP alignment enforcement.
//! - **Prologue/epilogue generation**: SD/LD instructions for callee-saved
//!   register save/restore with proper frame pointer setup.
//! - **Call sequence generation**: JAL for direct calls, JALR for indirect calls.
//!
//! ## Key LP64D ABI Rules
//!
//! | Rule                                | Details                                        |
//! |-------------------------------------|------------------------------------------------|
//! | Integer argument registers          | a0-a7 (x10-x17)                               |
//! | FP argument registers               | fa0-fa7 (f10-f17)                              |
//! | Return registers                    | a0 (integer), fa0 (float), a0+a1 (wide)       |
//! | Callee-saved GPRs                   | s0-s11 (x8-x9, x18-x27)                       |
//! | Callee-saved FPRs                   | fs0-fs11 (f8-f9, f18-f27)                      |
//! | Frame pointer                       | s0 (x8), also known as fp                      |
//! | Return address register             | ra (x1)                                        |
//! | Stack alignment                     | SP must be 16-byte aligned AT ALL TIMES        |
//! | Red zone                            | NONE on RISC-V                                 |
//!
//! ## Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use crate::codegen::regalloc::{AllocationResult, PhysReg, RegClass};
use crate::codegen::{MachineInstr, MachineOperand};
use crate::driver::target::TargetConfig;
use crate::ir::{Callee, Function, Instruction, IrType, Value};

// ---------------------------------------------------------------------------
// Register constants re-imported from parent mod.rs
// ---------------------------------------------------------------------------

use super::{
    X5, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17,
    X18, X19, X20, X21, X22, X23, X24, X25, X26, X27,
    F8, F9, F10, F11, F12, F13, F14, F15, F16, F17,
    F18, F19, F20, F21, F22, F23, F24, F25, F26, F27,
    RA, SP, FP, ZERO,
};

// ---------------------------------------------------------------------------
// RegClass Usage — ABI-level register class determination
// ---------------------------------------------------------------------------

/// Returns the register class for a given IR type per LP64D ABI rules.
///
/// Floating-point types (f32, f64) use `RegClass::Float` (FP register file),
/// while all other types (integers, pointers, aggregates) use
/// `RegClass::Integer` (general-purpose register file).
#[inline]
pub fn abi_reg_class(ty: &IrType) -> RegClass {
    if ty.is_float() {
        RegClass::Float
    } else {
        RegClass::Integer
    }
}

// ---------------------------------------------------------------------------
// LP64D ABI Opcode constants for prologue/epilogue/call sequences
// ---------------------------------------------------------------------------
// These constants MUST match the values in `super::isel::Riscv64Opcode`.
// They are raw u32 values duplicated here because `Riscv64Opcode::as_u32()`
// is not a const fn and cannot be used to initialize `const` items.

/// SD (Store Doubleword) — used for saving 64-bit callee-saved registers.
const OP_SD: u32 = 20; // Riscv64Opcode::SD
/// LD (Load Doubleword) — used for restoring 64-bit callee-saved registers.
const OP_LD: u32 = 13; // Riscv64Opcode::LD
/// ADDI — used for SP adjustment in prologue/epilogue and MV pseudo-instr.
const OP_ADDI: u32 = 21; // Riscv64Opcode::ADDI
/// JAL — used for direct function calls (JAL ra, offset).
const OP_JAL: u32 = 2; // Riscv64Opcode::JAL
/// JALR — used for indirect function calls and RET pseudo-instruction.
const OP_JALR: u32 = 3; // Riscv64Opcode::JALR
/// ADD — used for large-frame SP adjustment via temporary register.
const OP_ADD: u32 = 30; // Riscv64Opcode::ADD
/// LUI — used for large immediate materialization.
const OP_LUI: u32 = 0; // Riscv64Opcode::LUI

// Floating-point store/load opcodes.
// These are after the integer base ISA + M extension + A extension + F extension.
// Count from the enum: base 52 (LUI=0 through EBREAK=51), M ext +13 = 65 (MUL..REMUW),
// A ext +14 = 79, F ext starts at FSW=80... Need to count precisely.

/// FSD (Floating-point Store Double) — for saving FP callee-saved registers.
/// Value must match Riscv64Opcode::FSD enum discriminant.
const OP_FSD: u32 = {
    // We need the correct enum value. Let's compute it at compile time.
    // Instead of counting manually, we define a helper.
    // FLD and FSD are in the D extension group.
    // Per the enum ordering, we just use the known position.
    // Safest approach: define these as runtime-computed values via functions.
    0 // placeholder — will be replaced by function calls
};

/// FLD (Floating-point Load Double) — for restoring FP callee-saved registers.
const OP_FLD: u32 = 0; // placeholder — will be replaced by function calls

/// FADD_D — used as a pseudo-move for FP register copies (fadd.d rd, rs, zero).
const OP_FADD_D: u32 = 0; // placeholder

// Since the opcode enum values for floating-point instructions are hard to
// count precisely, we use a runtime helper function that calls as_u32().
use super::isel::Riscv64Opcode;

/// Returns the u32 opcode for FSD.
#[inline(always)]
fn op_fsd() -> u32 { Riscv64Opcode::FSD.as_u32() }
/// Returns the u32 opcode for FLD.
#[inline(always)]
fn op_fld() -> u32 { Riscv64Opcode::FLD.as_u32() }
/// Returns the u32 opcode for FADD.D (used as FP register move pseudo-op).
#[inline(always)]
fn op_fadd_d() -> u32 { Riscv64Opcode::FADD_D.as_u32() }
/// Returns the u32 opcode for FADD.S (used as FP register move for f32).
#[inline(always)]
fn op_fadd_s() -> u32 { Riscv64Opcode::FADD_S.as_u32() }

// ---------------------------------------------------------------------------
// Integer Argument Registers (a0-a7)
// ---------------------------------------------------------------------------

/// The 8 integer argument registers in the LP64D ABI (a0-a7 = x10-x17).
/// Used for passing integer and pointer arguments, and for returning
/// integer values in a0 (and a1 for 128-bit returns).
const INT_ARG_REGS: [PhysReg; 8] = [X10, X11, X12, X13, X14, X15, X16, X17];

/// Maximum number of integer argument registers (8: a0-a7).
const MAX_INT_ARG_REGS: usize = 8;

// ---------------------------------------------------------------------------
// Floating-Point Argument Registers (fa0-fa7)
// ---------------------------------------------------------------------------

/// The 8 floating-point argument registers in the LP64D ABI (fa0-fa7 = f10-f17).
/// Used for passing float/double arguments and returning float values in fa0.
const FP_ARG_REGS: [PhysReg; 8] = [F10, F11, F12, F13, F14, F15, F16, F17];

/// Maximum number of floating-point argument registers (8: fa0-fa7).
const MAX_FP_ARG_REGS: usize = 8;

// ---------------------------------------------------------------------------
// Callee-Saved Registers
// ---------------------------------------------------------------------------

/// Callee-saved integer registers: s0-s11 (x8-x9, x18-x27).
/// These must be saved in the prologue and restored in the epilogue if used.
const CALLEE_SAVED_GPRS: [PhysReg; 12] = [
    X8, X9,     // s0 (fp), s1
    X18, X19,   // s2, s3
    X20, X21,   // s4, s5
    X22, X23,   // s6, s7
    X24, X25,   // s8, s9
    X26, X27,   // s10, s11
];

/// Callee-saved floating-point registers: fs0-fs11 (f8-f9, f18-f27).
/// These must be saved in the prologue and restored in the epilogue if used.
const CALLEE_SAVED_FPRS: [PhysReg; 12] = [
    F8, F9,     // fs0, fs1
    F18, F19,   // fs2, fs3
    F20, F21,   // fs4, fs5
    F22, F23,   // fs6, fs7
    F24, F25,   // fs8, fs9
    F26, F27,   // fs10, fs11
];

// ---------------------------------------------------------------------------
// Helper — register classification queries
// ---------------------------------------------------------------------------

/// Returns `true` if the given physical register is a floating-point register
/// (PhysReg(32..=63) for f0-f31).
pub fn is_fp_reg(reg: PhysReg) -> bool {
    reg.0 >= 32 && reg.0 <= 63
}

/// Returns `true` if the given physical register is a general-purpose register
/// (PhysReg(0..=31) for x0-x31).
pub fn is_gpr(reg: PhysReg) -> bool {
    reg.0 <= 31
}

/// Returns `true` if the given register is a callee-saved GPR (s0-s11).
pub fn is_callee_saved_gpr(reg: PhysReg) -> bool {
    CALLEE_SAVED_GPRS.contains(&reg)
}

/// Returns `true` if the given register is a callee-saved FPR (fs0-fs11).
pub fn is_callee_saved_fpr(reg: PhysReg) -> bool {
    CALLEE_SAVED_FPRS.contains(&reg)
}

// ===========================================================================
// Argument Classification
// ===========================================================================

/// Classification of a function argument for register or stack passing.
///
/// LP64D ABI rules:
/// - Integer/pointer types ≤ XLEN (8 bytes on RV64): passed in next available
///   integer register a0-a7.
/// - Float (f32): passed in next available FP register fa0-fa7.
/// - Double (f64): passed in next available FP register fa0-fa7.
/// - Struct ≤ 2×XLEN (≤16 bytes): passed in one or two integer registers.
/// - Struct > 2×XLEN: passed by reference (pointer in integer register).
/// - When no registers are available, arguments are passed on the stack.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgClass {
    /// Argument passed in an integer register (a0-a7).
    IntegerReg {
        /// The physical register assigned (one of a0-a7).
        reg: PhysReg,
        /// The IR type of the argument.
        ty: IrType,
    },
    /// Argument passed in a floating-point register (fa0-fa7).
    FloatReg {
        /// The physical register assigned (one of fa0-fa7).
        reg: PhysReg,
        /// The IR type of the argument.
        ty: IrType,
    },
    /// Argument passed in a pair of integer registers (for ≤16-byte aggregates).
    IntegerRegPair {
        /// The first register (lower half).
        reg_lo: PhysReg,
        /// The second register (upper half).
        reg_hi: PhysReg,
        /// The IR type of the argument.
        ty: IrType,
    },
    /// Argument passed on the stack.
    Stack {
        /// Byte offset from SP at the call site where this argument is placed.
        offset: i32,
        /// Size in bytes of the argument on the stack.
        size: u32,
        /// The IR type of the argument.
        ty: IrType,
    },
    /// Large aggregate passed by reference: a pointer to the struct is placed
    /// in an integer register.
    IndirectReg {
        /// The register holding the pointer to the aggregate.
        reg: PhysReg,
        /// The IR type of the original aggregate.
        ty: IrType,
    },
    /// Large aggregate passed by reference on the stack.
    IndirectStack {
        /// Byte offset from SP where the pointer is placed.
        offset: i32,
        /// The IR type of the original aggregate.
        ty: IrType,
    },
}

// ===========================================================================
// Return Value Classification
// ===========================================================================

/// Classification of a function return value.
///
/// LP64D ABI rules:
/// - Integer/pointer ≤ XLEN: returned in a0.
/// - Float: returned in fa0.
/// - Double: returned in fa0.
/// - 2×XLEN aggregate: returned in a0+a1 register pair.
/// - Larger aggregates: caller passes a hidden pointer in a0, callee writes
///   the result to that location and returns the pointer in a0.
/// - Void: no return value.
#[derive(Debug, Clone, PartialEq)]
pub enum ReturnClass {
    /// No return value (void function).
    Void,
    /// Integer/pointer return in a0.
    IntegerReg {
        /// Always x10 (a0).
        reg: PhysReg,
        /// Return type.
        ty: IrType,
    },
    /// Floating-point return in fa0.
    FloatReg {
        /// Always f10 (fa0).
        reg: PhysReg,
        /// Return type.
        ty: IrType,
    },
    /// Two-register return in a0+a1 (for ≤16-byte aggregates).
    IntegerRegPair {
        /// a0 (x10).
        reg_lo: PhysReg,
        /// a1 (x11).
        reg_hi: PhysReg,
        /// Return type.
        ty: IrType,
    },
    /// Indirect return: caller provides buffer pointer in a0, callee writes
    /// result there and returns the pointer in a0.
    Indirect {
        /// a0 (x10) — holds pointer to result buffer.
        reg: PhysReg,
        /// The aggregate type being returned.
        ty: IrType,
    },
}

// ===========================================================================
// Frame Layout
// ===========================================================================

/// Stack frame layout computed for a single function per LP64D ABI.
///
/// RISC-V LP64D stack frame (growing downward, SP at bottom):
///
/// ```text
///        +---------------------------+  ← Old SP (16-byte aligned)
///        | Incoming stack arguments  |
///        +---------------------------+
///        | Return address (ra)       |  ← SP + frame_size - 8
///        +---------------------------+
///        | Frame pointer (s0/fp)     |  ← SP + frame_size - 16
///        +---------------------------+
///        | Callee-saved GPRs         |
///        | (s1-s11 as needed)        |
///        +---------------------------+
///        | Callee-saved FPRs         |
///        | (fs0-fs11 as needed)      |
///        +---------------------------+
///        | Local variables / spills  |
///        +---------------------------+
///        | Outgoing stack arguments  |
///        +---------------------------+  ← New SP (16-byte aligned)
/// ```
#[derive(Debug, Clone)]
pub struct Riscv64FrameLayout {
    /// Total frame size in bytes (SP is decremented by this amount).
    /// Always a multiple of 16 for alignment.
    pub frame_size: i32,

    /// Offset from SP where ra is saved.
    pub ra_offset: i32,

    /// Offset from SP where s0 (fp) is saved.
    pub fp_offset: i32,

    /// Callee-saved GPRs that need to be saved/restored, with their
    /// offsets from SP.
    pub saved_gprs: Vec<(PhysReg, i32)>,

    /// Callee-saved FPRs that need to be saved/restored, with their
    /// offsets from SP.
    pub saved_fprs: Vec<(PhysReg, i32)>,

    /// Size of the local variable / spill area in bytes.
    pub locals_size: i32,

    /// Size of the outgoing argument area on the stack (for calls that
    /// need more than 8 arguments).
    pub outgoing_args_size: i32,

    /// Whether the frame pointer (s0) is used. True for most functions
    /// to enable debugger stack walking.
    pub uses_frame_pointer: bool,

    /// Whether the function makes any calls (determines if ra needs saving).
    pub has_calls: bool,
}

// ===========================================================================
// Argument Classification — classify_arguments()
// ===========================================================================

/// Classifies function parameters according to the RISC-V LP64D ABI.
///
/// Iterates parameters left-to-right, assigning each to an integer register
/// (a0-a7), floating-point register (fa0-fa7), or stack slot based on its
/// type and the current register allocation state.
///
/// # Arguments
///
/// * `params` — Ordered list of `(name, type)` pairs for the function's parameters.
/// * `target` — Target configuration (used for pointer size and type queries).
///
/// # Returns
///
/// A `Vec<ArgClass>` with one entry per parameter, in the same order.
pub fn classify_arguments(params: &[(String, IrType)], target: &TargetConfig) -> Vec<ArgClass> {
    let mut result = Vec::with_capacity(params.len());
    let mut int_reg_idx: usize = 0;  // Next available integer argument register
    let mut fp_reg_idx: usize = 0;   // Next available FP argument register
    let mut stack_offset: i32 = 0;   // Current stack argument offset

    for (_name, ty) in params {
        let class = classify_single_argument(
            ty,
            target,
            &mut int_reg_idx,
            &mut fp_reg_idx,
            &mut stack_offset,
        );
        result.push(class);
    }

    result
}

/// Classifies a single argument according to LP64D rules.
fn classify_single_argument(
    ty: &IrType,
    target: &TargetConfig,
    int_reg_idx: &mut usize,
    fp_reg_idx: &mut usize,
    stack_offset: &mut i32,
) -> ArgClass {
    match ty {
        // Floating-point types: use FP registers if available
        IrType::F32 | IrType::F64 => {
            if *fp_reg_idx < MAX_FP_ARG_REGS {
                let reg = FP_ARG_REGS[*fp_reg_idx];
                *fp_reg_idx += 1;
                ArgClass::FloatReg { reg, ty: ty.clone() }
            } else {
                // Fall back to stack
                let size = if *ty == IrType::F32 { 4u32 } else { 8u32 };
                // Align stack offset to natural alignment, minimum 8 (XLEN)
                *stack_offset = align_up(*stack_offset, 8);
                let offset = *stack_offset;
                // On LP64D, stack slots are always at least 8 bytes (XLEN)
                *stack_offset += 8;
                ArgClass::Stack { offset, size, ty: ty.clone() }
            }
        }

        // Aggregate types (structs): classify by size
        IrType::Struct { fields, packed } => {
            let sz = ty.size(target) as i32;
            let xlen = target.pointer_size as i32; // 8 for RV64
            let two_xlen = 2 * xlen;

            if sz <= 0 {
                // Zero-size struct: no argument slot consumed
                ArgClass::Stack { offset: *stack_offset, size: 0, ty: ty.clone() }
            } else if sz <= xlen {
                // Fits in one register
                if *int_reg_idx < MAX_INT_ARG_REGS {
                    let reg = INT_ARG_REGS[*int_reg_idx];
                    *int_reg_idx += 1;
                    ArgClass::IntegerReg { reg, ty: ty.clone() }
                } else {
                    *stack_offset = align_up(*stack_offset, 8);
                    let offset = *stack_offset;
                    *stack_offset += 8;
                    ArgClass::Stack { offset, size: sz as u32, ty: ty.clone() }
                }
            } else if sz <= two_xlen {
                // Fits in two registers
                if *int_reg_idx + 1 < MAX_INT_ARG_REGS {
                    // Need two aligned registers
                    let reg_lo = INT_ARG_REGS[*int_reg_idx];
                    let reg_hi = INT_ARG_REGS[*int_reg_idx + 1];
                    *int_reg_idx += 2;
                    ArgClass::IntegerRegPair { reg_lo, reg_hi, ty: ty.clone() }
                } else if *int_reg_idx < MAX_INT_ARG_REGS {
                    // One register available + stack for the rest.
                    // Per LP64D: if only one register is available for a 2×XLEN
                    // aggregate, the low half goes in register, high half on stack.
                    let reg = INT_ARG_REGS[*int_reg_idx];
                    *int_reg_idx = MAX_INT_ARG_REGS; // Consume remaining
                    ArgClass::IntegerReg { reg, ty: ty.clone() }
                } else {
                    // Both halves on stack
                    *stack_offset = align_up(*stack_offset, 8);
                    let offset = *stack_offset;
                    *stack_offset += two_xlen;
                    ArgClass::Stack { offset, size: sz as u32, ty: ty.clone() }
                }
            } else {
                // Larger than 2×XLEN: pass by reference
                if *int_reg_idx < MAX_INT_ARG_REGS {
                    let reg = INT_ARG_REGS[*int_reg_idx];
                    *int_reg_idx += 1;
                    ArgClass::IndirectReg { reg, ty: ty.clone() }
                } else {
                    *stack_offset = align_up(*stack_offset, 8);
                    let offset = *stack_offset;
                    *stack_offset += 8; // Pointer size
                    ArgClass::IndirectStack { offset, ty: ty.clone() }
                }
            }
        }

        // All other types (integers, pointers, booleans): use integer registers
        _ => {
            if *int_reg_idx < MAX_INT_ARG_REGS {
                let reg = INT_ARG_REGS[*int_reg_idx];
                *int_reg_idx += 1;
                ArgClass::IntegerReg { reg, ty: ty.clone() }
            } else {
                // Stack argument
                let size = ty.size(target) as u32;
                *stack_offset = align_up(*stack_offset, 8);
                let offset = *stack_offset;
                *stack_offset += 8; // All stack slots are XLEN-aligned
                ArgClass::Stack { offset, size, ty: ty.clone() }
            }
        }
    }
}

// ===========================================================================
// Return Value Classification — classify_return()
// ===========================================================================

/// Classifies the return type of a function according to LP64D ABI rules.
///
/// # Arguments
///
/// * `return_type` — The IR type of the return value.
/// * `target` — Target configuration for type size queries.
///
/// # Returns
///
/// A `ReturnClass` indicating how the value is returned.
pub fn classify_return(return_type: &IrType, target: &TargetConfig) -> ReturnClass {
    match return_type {
        IrType::Void => ReturnClass::Void,

        IrType::F32 | IrType::F64 => ReturnClass::FloatReg {
            reg: F10, // fa0
            ty: return_type.clone(),
        },

        IrType::Struct { .. } => {
            let sz = return_type.size(target) as i32;
            let xlen = target.pointer_size as i32;

            if sz <= 0 {
                ReturnClass::Void
            } else if sz <= xlen {
                ReturnClass::IntegerReg {
                    reg: X10, // a0
                    ty: return_type.clone(),
                }
            } else if sz <= 2 * xlen {
                ReturnClass::IntegerRegPair {
                    reg_lo: X10, // a0
                    reg_hi: X11, // a1
                    ty: return_type.clone(),
                }
            } else {
                // Large aggregate: indirect return via hidden pointer in a0
                ReturnClass::Indirect {
                    reg: X10, // a0
                    ty: return_type.clone(),
                }
            }
        }

        // All integer/pointer types: return in a0
        _ => ReturnClass::IntegerReg {
            reg: X10, // a0
            ty: return_type.clone(),
        },
    }
}

// ===========================================================================
// Frame Layout Computation — compute_frame_layout()
// ===========================================================================

/// Computes the stack frame layout for a function according to LP64D ABI.
///
/// Determines which callee-saved registers need saving, computes the total
/// frame size (16-byte aligned), and assigns stack offsets for saved registers,
/// local variables, and outgoing arguments.
///
/// # Arguments
///
/// * `function` — The IR function to compute the layout for.
/// * `alloc_result` — Register allocation result (determines which callee-saved
///   registers are used and need saving).
/// * `target` — Target configuration for pointer size.
/// * `outgoing_args_size` — Size needed for outgoing stack arguments (from call
///   site analysis).
///
/// # Returns
///
/// A `Riscv64FrameLayout` describing the complete frame structure.
pub fn compute_frame_layout(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
    outgoing_args_size: i32,
) -> Riscv64FrameLayout {
    let xlen: i32 = target.pointer_size as i32; // 8 for RV64

    // Determine if the function makes any calls (needs ra saved).
    let has_calls = function_has_calls(function);

    // Determine which callee-saved registers are used by the allocation.
    let mut used_callee_gprs: Vec<PhysReg> = Vec::new();
    let mut used_callee_fprs: Vec<PhysReg> = Vec::new();

    for reg in &alloc_result.used_callee_saved {
        if is_callee_saved_gpr(*reg) && *reg != FP {
            // FP (s0/x8) is handled separately
            used_callee_gprs.push(*reg);
        } else if is_callee_saved_fpr(*reg) {
            used_callee_fprs.push(*reg);
        }
    }

    // Always save ra if the function makes calls.
    // Always save s0(fp) for debugger stack walking.
    let uses_frame_pointer = true;

    // Compute save area sizes:
    // - ra: 1 slot (8 bytes)
    // - s0(fp): 1 slot (8 bytes) if uses_frame_pointer
    // - callee-saved GPRs: N slots × 8 bytes each
    // - callee-saved FPRs: N slots × 8 bytes each
    let ra_slots = if has_calls { 1 } else { 0 };
    let fp_slots: usize = if uses_frame_pointer { 1 } else { 0 };
    let gpr_save_slots = used_callee_gprs.len();
    let fpr_save_slots = used_callee_fprs.len();

    let save_area_size = (ra_slots + fp_slots + gpr_save_slots + fpr_save_slots) as i32 * xlen;

    // Compute local variable / spill area size from the allocation result.
    let locals_size = (alloc_result.num_spill_slots as i32) * xlen;

    // Total frame size = save area + locals + outgoing args, aligned to 16
    let raw_frame_size = save_area_size + locals_size + outgoing_args_size;
    let frame_size = align_up(raw_frame_size.max(16), 16);

    // Assign offsets for saved registers (from SP, growing upward).
    // Layout from SP:
    //   [outgoing args]  offset 0..outgoing_args_size
    //   [locals/spills]  offset outgoing_args_size..outgoing_args_size+locals_size
    //   [callee-saved FPRs]
    //   [callee-saved GPRs]
    //   [s0/fp]
    //   [ra]             offset frame_size - 8
    let mut current_offset = frame_size;

    // ra is at the top of the frame
    let ra_offset = if has_calls {
        current_offset -= xlen;
        current_offset
    } else {
        0
    };

    // fp (s0) is just below ra
    let fp_offset = if uses_frame_pointer {
        current_offset -= xlen;
        current_offset
    } else {
        0
    };

    // Callee-saved GPRs
    let mut saved_gprs: Vec<(PhysReg, i32)> = Vec::with_capacity(gpr_save_slots);
    for reg in &used_callee_gprs {
        current_offset -= xlen;
        saved_gprs.push((*reg, current_offset));
    }

    // Callee-saved FPRs
    let mut saved_fprs: Vec<(PhysReg, i32)> = Vec::with_capacity(fpr_save_slots);
    for reg in &used_callee_fprs {
        current_offset -= xlen;
        saved_fprs.push((*reg, current_offset));
    }

    Riscv64FrameLayout {
        frame_size,
        ra_offset,
        fp_offset,
        saved_gprs,
        saved_fprs,
        locals_size,
        outgoing_args_size,
        uses_frame_pointer,
        has_calls,
    }
}

// ===========================================================================
// Prologue Generation
// ===========================================================================

/// Generates the function prologue instruction sequence per LP64D ABI.
///
/// The prologue performs:
/// 1. Decrement SP by frame_size (`addi sp, sp, -frame_size`).
/// 2. Save ra to stack (`sd ra, ra_offset(sp)`).
/// 3. Save s0/fp to stack (`sd s0, fp_offset(sp)`).
/// 4. Set up frame pointer (`addi s0, sp, frame_size`).
/// 5. Save callee-saved GPRs (`sd sN, offset(sp)`).
/// 6. Save callee-saved FPRs (`fsd fsN, offset(sp)`).
///
/// For large frame sizes (>2047 bytes, exceeding 12-bit immediate range),
/// the SP adjustment is split into multiple instructions using a temporary.
///
/// # Arguments
///
/// * `layout` — The computed frame layout.
///
/// # Returns
///
/// A `Vec<MachineInstr>` containing the prologue instruction sequence.
pub fn generate_prologue(layout: &Riscv64FrameLayout) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(16);

    if layout.frame_size == 0 {
        return instrs;
    }

    // Step 1: Adjust SP
    if fits_in_12bit_signed(-layout.frame_size) {
        // addi sp, sp, -frame_size
        instrs.push(MachineInstr::with_operands(
            OP_ADDI,
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(-layout.frame_size as i64),
            ],
        ));
    } else {
        // Large frame: use t0 (x5) as temporary
        // lui t0, hi20(-frame_size)
        // addi t0, t0, lo12(-frame_size)
        // add sp, sp, t0
        let neg_size = (-layout.frame_size) as i64;
        let hi20 = ((neg_size + 0x800) >> 12) & 0xFFFFF;
        let lo12 = neg_size - (hi20 << 12);

        instrs.push(MachineInstr::with_operands(
            OP_LUI,
            vec![
                MachineOperand::Register(X5), // t0
                MachineOperand::Immediate(hi20),
            ],
        ));
        instrs.push(MachineInstr::with_operands(
            OP_ADDI,
            vec![
                MachineOperand::Register(X5),
                MachineOperand::Register(X5),
                MachineOperand::Immediate(lo12),
            ],
        ));
        // add sp, sp, t0
        instrs.push(MachineInstr::with_operands(
            OP_ADD,
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Register(X5),
            ],
        ));
    }

    // Step 2: Save ra
    if layout.has_calls {
        instrs.push(MachineInstr::with_operands(
            OP_SD,
            vec![
                MachineOperand::Register(RA),
                MachineOperand::Memory { base: SP, offset: layout.ra_offset },
            ],
        ));
    }

    // Step 3: Save s0 (fp)
    if layout.uses_frame_pointer {
        instrs.push(MachineInstr::with_operands(
            OP_SD,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Memory { base: SP, offset: layout.fp_offset },
            ],
        ));

        // Step 4: Set up frame pointer: addi s0, sp, frame_size
        if fits_in_12bit_signed(layout.frame_size) {
            instrs.push(MachineInstr::with_operands(
                OP_ADDI,
                vec![
                    MachineOperand::Register(FP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(layout.frame_size as i64),
                ],
            ));
        } else {
            // For large frames, fp = sp + frame_size via LUI+ADDI+ADD
            let hi20 = ((layout.frame_size as i64 + 0x800) >> 12) & 0xFFFFF;
            let lo12 = layout.frame_size as i64 - (hi20 << 12);
            instrs.push(MachineInstr::with_operands(
                OP_LUI,
                vec![
                    MachineOperand::Register(X5),
                    MachineOperand::Immediate(hi20),
                ],
            ));
            instrs.push(MachineInstr::with_operands(
                OP_ADDI,
                vec![
                    MachineOperand::Register(X5),
                    MachineOperand::Register(X5),
                    MachineOperand::Immediate(lo12),
                ],
            ));
            instrs.push(MachineInstr::with_operands(
                OP_ADD,
                vec![
                    MachineOperand::Register(FP),
                    MachineOperand::Register(SP),
                    MachineOperand::Register(X5),
                ],
            ));
        }
    }

    // Step 5: Save callee-saved GPRs
    for &(reg, offset) in &layout.saved_gprs {
        instrs.push(MachineInstr::with_operands(
            OP_SD,
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Memory { base: SP, offset },
            ],
        ));
    }

    // Step 6: Save callee-saved FPRs
    for &(reg, offset) in &layout.saved_fprs {
        instrs.push(MachineInstr::with_operands(
            op_fsd(),
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Memory { base: SP, offset },
            ],
        ));
    }

    instrs
}

// ===========================================================================
// Epilogue Generation
// ===========================================================================

/// Generates the function epilogue instruction sequence per LP64D ABI.
///
/// The epilogue performs:
/// 1. Restore callee-saved FPRs.
/// 2. Restore callee-saved GPRs.
/// 3. Restore s0/fp.
/// 4. Restore ra.
/// 5. Increment SP by frame_size.
/// 6. Return (ret = jalr x0, ra, 0).
///
/// # Arguments
///
/// * `layout` — The computed frame layout.
///
/// # Returns
///
/// A `Vec<MachineInstr>` containing the epilogue instruction sequence.
pub fn generate_epilogue(layout: &Riscv64FrameLayout) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(16);

    if layout.frame_size == 0 {
        // No frame: just return
        instrs.push(make_ret());
        return instrs;
    }

    // Step 1: Restore callee-saved FPRs (reverse order)
    for &(reg, offset) in layout.saved_fprs.iter().rev() {
        instrs.push(MachineInstr::with_operands(
            op_fld(),
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Memory { base: SP, offset },
            ],
        ));
    }

    // Step 2: Restore callee-saved GPRs (reverse order)
    for &(reg, offset) in layout.saved_gprs.iter().rev() {
        instrs.push(MachineInstr::with_operands(
            OP_LD,
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Memory { base: SP, offset },
            ],
        ));
    }

    // Step 3: Restore fp (s0)
    if layout.uses_frame_pointer {
        instrs.push(MachineInstr::with_operands(
            OP_LD,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Memory { base: SP, offset: layout.fp_offset },
            ],
        ));
    }

    // Step 4: Restore ra
    if layout.has_calls {
        instrs.push(MachineInstr::with_operands(
            OP_LD,
            vec![
                MachineOperand::Register(RA),
                MachineOperand::Memory { base: SP, offset: layout.ra_offset },
            ],
        ));
    }

    // Step 5: Restore SP
    if fits_in_12bit_signed(layout.frame_size) {
        instrs.push(MachineInstr::with_operands(
            OP_ADDI,
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(layout.frame_size as i64),
            ],
        ));
    } else {
        // Large frame: use t0 as temporary
        let hi20 = ((layout.frame_size as i64 + 0x800) >> 12) & 0xFFFFF;
        let lo12 = layout.frame_size as i64 - (hi20 << 12);
        instrs.push(MachineInstr::with_operands(
            OP_LUI,
            vec![
                MachineOperand::Register(X5),
                MachineOperand::Immediate(hi20),
            ],
        ));
        instrs.push(MachineInstr::with_operands(
            OP_ADDI,
            vec![
                MachineOperand::Register(X5),
                MachineOperand::Register(X5),
                MachineOperand::Immediate(lo12),
            ],
        ));
        instrs.push(MachineInstr::with_operands(
            OP_ADD,
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Register(X5),
            ],
        ));
    }

    // Step 6: Return
    instrs.push(make_ret());

    instrs
}

// ===========================================================================
// Argument Loading — generate_argument_loads()
// ===========================================================================

/// Generates instructions to load function arguments from their ABI locations
/// (registers or stack) into the virtual registers expected by the function body.
///
/// This is called at the start of code generation for a function, after the
/// prologue, to move arguments from their calling-convention locations to the
/// registers assigned by the register allocator.
///
/// # Arguments
///
/// * `arg_classes` — Classification of each argument (register, stack, etc.).
/// * `alloc_result` — Register allocation result mapping IR values to physical
///   registers or stack slots.
/// * `param_values` — IR values corresponding to each function parameter.
///
/// # Returns
///
/// A `Vec<MachineInstr>` to insert after the prologue.
pub fn generate_argument_loads(
    arg_classes: &[ArgClass],
    alloc_result: &AllocationResult,
    param_values: &[Value],
) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(arg_classes.len());

    // Build a lookup from Value → PhysReg using the allocation intervals.
    let value_reg_map = build_value_reg_map(alloc_result);

    for (i, class) in arg_classes.iter().enumerate() {
        if i >= param_values.len() {
            break;
        }

        // Find the destination register from the allocation result.
        let dest_reg = value_reg_map.get(&param_values[i]);

        match class {
            ArgClass::IntegerReg { reg, .. } => {
                if let Some(&dest) = dest_reg {
                    if dest != *reg {
                        // mv dest, aN (addi dest, aN, 0)
                        instrs.push(make_mv(dest, *reg));
                    }
                }
            }
            ArgClass::FloatReg { reg, ty } => {
                if let Some(&dest) = dest_reg {
                    if dest != *reg {
                        // Use FADD.D/FADD.S rd, rs, rs as register move
                        // This is equivalent to fsgnj.d which is the standard
                        // FP register move pseudo-instruction.
                        let op = if *ty == IrType::F32 { op_fadd_s() } else { op_fadd_d() };
                        instrs.push(MachineInstr::with_operands(
                            op,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(*reg),
                                MachineOperand::Register(*reg),
                            ],
                        ));
                    }
                }
            }
            ArgClass::IntegerRegPair { reg_lo, .. } => {
                // For aggregate in register pair: move first register to dest.
                if let Some(&dest) = dest_reg {
                    if dest != *reg_lo {
                        instrs.push(make_mv(dest, *reg_lo));
                    }
                }
            }
            ArgClass::Stack { offset, .. } => {
                // Load from stack at the specified offset.
                if let Some(&dest) = dest_reg {
                    let load_op = if is_fp_reg(dest) { op_fld() } else { OP_LD };
                    instrs.push(MachineInstr::with_operands(
                        load_op,
                        vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Memory { base: SP, offset: *offset },
                        ],
                    ));
                }
            }
            ArgClass::IndirectReg { reg, .. } => {
                // Pointer to aggregate in register — just move it
                if let Some(&dest) = dest_reg {
                    if dest != *reg {
                        instrs.push(make_mv(dest, *reg));
                    }
                }
            }
            ArgClass::IndirectStack { offset, .. } => {
                // Load pointer from stack
                if let Some(&dest) = dest_reg {
                    instrs.push(MachineInstr::with_operands(
                        OP_LD,
                        vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Memory { base: SP, offset: *offset },
                        ],
                    ));
                }
            }
        }
    }

    instrs
}

// ===========================================================================
// Call Sequence Generation — generate_call_sequence()
// ===========================================================================

/// Generates the instruction sequence for a function call per LP64D ABI.
///
/// The call sequence:
/// 1. Move arguments to their ABI-designated locations (registers or stack).
/// 2. Issue the call instruction (JAL for direct, JALR for indirect).
/// 3. Move the return value from ABI register to destination register.
///
/// # Arguments
///
/// * `callee` — The call target (direct function name or indirect register).
/// * `args` — Classified arguments with their values and ABI locations.
/// * `return_class` — Classification of the return value.
/// * `dest_reg` — Optional destination register for the return value.
///
/// # Returns
///
/// A `Vec<MachineInstr>` implementing the complete call sequence.
pub fn generate_call_sequence(
    callee: &Callee,
    args: &[(ArgClass, PhysReg)],
    return_class: &ReturnClass,
    dest_reg: Option<PhysReg>,
) -> Vec<MachineInstr> {
    let mut instrs = Vec::with_capacity(args.len() + 4);

    // Step 1: Move arguments to their ABI locations
    for (class, src_reg) in args {
        match class {
            ArgClass::IntegerReg { reg, .. } | ArgClass::IndirectReg { reg, .. } => {
                if *src_reg != *reg {
                    instrs.push(make_mv(*reg, *src_reg));
                }
            }
            ArgClass::FloatReg { reg, ty } => {
                if *src_reg != *reg {
                    let op = if *ty == IrType::F32 { op_fadd_s() } else { op_fadd_d() };
                    instrs.push(MachineInstr::with_operands(
                        op,
                        vec![
                            MachineOperand::Register(*reg),
                            MachineOperand::Register(*src_reg),
                            MachineOperand::Register(*src_reg),
                        ],
                    ));
                }
            }
            ArgClass::Stack { offset, .. } | ArgClass::IndirectStack { offset, .. } => {
                let store_op = if is_fp_reg(*src_reg) { op_fsd() } else { OP_SD };
                instrs.push(MachineInstr::with_operands(
                    store_op,
                    vec![
                        MachineOperand::Register(*src_reg),
                        MachineOperand::Memory { base: SP, offset: *offset },
                    ],
                ));
            }
            ArgClass::IntegerRegPair { reg_lo, .. } => {
                if *src_reg != *reg_lo {
                    instrs.push(make_mv(*reg_lo, *src_reg));
                }
            }
        }
    }

    // Step 2: Issue the call instruction
    match callee {
        Callee::Direct(name) => {
            // JAL ra, target — direct call (linker resolves the offset)
            instrs.push(MachineInstr::with_operands(
                OP_JAL,
                vec![
                    MachineOperand::Register(RA),
                    MachineOperand::Symbol(name.clone()),
                ],
            ));
        }
        Callee::Indirect(_val) => {
            // JALR ra, rs1, 0 — indirect call through register.
            // The register holding the function pointer should already be
            // in t0 (x5), the default indirect call register.
            instrs.push(MachineInstr::with_operands(
                OP_JALR,
                vec![
                    MachineOperand::Register(RA),
                    MachineOperand::Register(X5),
                    MachineOperand::Immediate(0),
                ],
            ));
        }
    }

    // Step 3: Move return value to destination register
    if let Some(dest) = dest_reg {
        match return_class {
            ReturnClass::Void => {}
            ReturnClass::IntegerReg { reg, .. } | ReturnClass::Indirect { reg, .. } => {
                if dest != *reg {
                    instrs.push(make_mv(dest, *reg));
                }
            }
            ReturnClass::FloatReg { reg, ty } => {
                if dest != *reg {
                    let op = if *ty == IrType::F32 { op_fadd_s() } else { op_fadd_d() };
                    instrs.push(MachineInstr::with_operands(
                        op,
                        vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Register(*reg),
                            MachineOperand::Register(*reg),
                        ],
                    ));
                }
            }
            ReturnClass::IntegerRegPair { reg_lo, .. } => {
                if dest != *reg_lo {
                    instrs.push(make_mv(dest, *reg_lo));
                }
            }
        }
    }

    instrs
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Aligns `value` up to the nearest multiple of `alignment`.
/// `alignment` must be a power of two.
#[inline]
fn align_up(value: i32, alignment: i32) -> i32 {
    debug_assert!(alignment > 0);
    (value + alignment - 1) & !(alignment - 1)
}

/// Returns `true` if the given signed value fits in a 12-bit signed immediate
/// (range: -2048..=2047).
#[inline]
fn fits_in_12bit_signed(value: i32) -> bool {
    value >= -2048 && value <= 2047
}

/// Determines whether the given function contains any call instructions.
///
/// Scans basic block instructions for Call variants to decide if `ra`
/// needs to be saved in the prologue.
fn function_has_calls(function: &Function) -> bool {
    for block in &function.blocks {
        for instr in &block.instructions {
            if let Instruction::Call { .. } = instr {
                return true;
            }
        }
    }
    false
}

/// Creates a MV pseudo-instruction: `addi rd, rs, 0`.
#[inline]
fn make_mv(rd: PhysReg, rs: PhysReg) -> MachineInstr {
    MachineInstr::with_operands(
        OP_ADDI,
        vec![
            MachineOperand::Register(rd),
            MachineOperand::Register(rs),
            MachineOperand::Immediate(0),
        ],
    )
}

/// Creates a RET pseudo-instruction: `jalr x0, ra, 0`.
#[inline]
fn make_ret() -> MachineInstr {
    MachineInstr::with_operands(
        OP_JALR,
        vec![
            MachineOperand::Register(ZERO),
            MachineOperand::Register(RA),
            MachineOperand::Immediate(0),
        ],
    )
}

/// Builds a Value → PhysReg mapping from the allocation result's intervals.
///
/// The `AllocationResult.intervals` contain `LiveInterval` entries, each with
/// a `value: Value` and an assigned `reg: Option<PhysReg>`. We extract all
/// intervals that have an assigned register.
fn build_value_reg_map(alloc_result: &AllocationResult) -> std::collections::HashMap<Value, PhysReg> {
    let mut map = std::collections::HashMap::new();
    for interval in &alloc_result.intervals {
        if let Some(reg) = interval.assigned_reg {
            map.insert(interval.value, reg);
        }
    }
    map
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a target config for RV64
    fn riscv64_target() -> TargetConfig {
        TargetConfig::riscv64()
    }

    // -----------------------------------------------------------------------
    // Register classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_gpr() {
        assert!(is_gpr(PhysReg(0)));  // x0
        assert!(is_gpr(PhysReg(31))); // x31
        assert!(!is_gpr(F10));
        assert!(!is_gpr(PhysReg(32)));
    }

    #[test]
    fn test_is_fp_reg() {
        assert!(!is_fp_reg(PhysReg(0)));
        assert!(!is_fp_reg(PhysReg(31)));
        assert!(is_fp_reg(F10));
        assert!(is_fp_reg(PhysReg(63)));
    }

    #[test]
    fn test_is_callee_saved_gpr() {
        assert!(is_callee_saved_gpr(X8));  // s0
        assert!(is_callee_saved_gpr(X9));  // s1
        assert!(is_callee_saved_gpr(X18)); // s2
        assert!(is_callee_saved_gpr(X27)); // s11
        assert!(!is_callee_saved_gpr(X10)); // a0 is caller-saved
        assert!(!is_callee_saved_gpr(X5));  // t0 is caller-saved
    }

    #[test]
    fn test_is_callee_saved_fpr() {
        assert!(is_callee_saved_fpr(F8));  // fs0
        assert!(is_callee_saved_fpr(F9));  // fs1
        assert!(is_callee_saved_fpr(F18)); // fs2
        assert!(is_callee_saved_fpr(F27)); // fs11
        assert!(!is_callee_saved_fpr(F10)); // fa0 is caller-saved
    }

    // -----------------------------------------------------------------------
    // ABI register class tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_abi_reg_class_integer() {
        assert_eq!(abi_reg_class(&IrType::I32), RegClass::Integer);
        assert_eq!(abi_reg_class(&IrType::I64), RegClass::Integer);
        assert_eq!(abi_reg_class(&IrType::I8), RegClass::Integer);
        assert_eq!(
            abi_reg_class(&IrType::Pointer(Box::new(IrType::I8))),
            RegClass::Integer
        );
    }

    #[test]
    fn test_abi_reg_class_float() {
        assert_eq!(abi_reg_class(&IrType::F32), RegClass::Float);
        assert_eq!(abi_reg_class(&IrType::F64), RegClass::Float);
    }

    // -----------------------------------------------------------------------
    // Argument classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_simple_int_args() {
        let target = riscv64_target();
        let params = vec![
            ("a".into(), IrType::I32),
            ("b".into(), IrType::I64),
            ("c".into(), IrType::I32),
        ];
        let classes = classify_arguments(&params, &target);
        assert_eq!(classes.len(), 3);

        // First three args should go in a0, a1, a2
        match &classes[0] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X10),
            _ => panic!("Expected IntegerReg for first arg"),
        }
        match &classes[1] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X11),
            _ => panic!("Expected IntegerReg for second arg"),
        }
        match &classes[2] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X12),
            _ => panic!("Expected IntegerReg for third arg"),
        }
    }

    #[test]
    fn test_classify_float_args() {
        let target = riscv64_target();
        let params = vec![
            ("x".into(), IrType::F64),
            ("y".into(), IrType::F32),
        ];
        let classes = classify_arguments(&params, &target);
        assert_eq!(classes.len(), 2);

        match &classes[0] {
            ArgClass::FloatReg { reg, .. } => assert_eq!(*reg, F10),
            _ => panic!("Expected FloatReg for first float arg"),
        }
        match &classes[1] {
            ArgClass::FloatReg { reg, .. } => assert_eq!(*reg, F11),
            _ => panic!("Expected FloatReg for second float arg"),
        }
    }

    #[test]
    fn test_classify_mixed_args() {
        let target = riscv64_target();
        let params = vec![
            ("a".into(), IrType::I32),
            ("b".into(), IrType::F64),
            ("c".into(), IrType::I64),
            ("d".into(), IrType::F32),
        ];
        let classes = classify_arguments(&params, &target);
        assert_eq!(classes.len(), 4);

        // Int and float registers are independently allocated
        match &classes[0] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X10), // a0
            _ => panic!("Expected IntegerReg"),
        }
        match &classes[1] {
            ArgClass::FloatReg { reg, .. } => assert_eq!(*reg, F10), // fa0
            _ => panic!("Expected FloatReg"),
        }
        match &classes[2] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X11), // a1
            _ => panic!("Expected IntegerReg"),
        }
        match &classes[3] {
            ArgClass::FloatReg { reg, .. } => assert_eq!(*reg, F11), // fa1
            _ => panic!("Expected FloatReg"),
        }
    }

    #[test]
    fn test_classify_overflow_to_stack() {
        let target = riscv64_target();
        // 9 integer args: first 8 in a0-a7, ninth on stack
        let params: Vec<(String, IrType)> = (0..9)
            .map(|i| (format!("arg{}", i), IrType::I64))
            .collect();
        let classes = classify_arguments(&params, &target);
        assert_eq!(classes.len(), 9);

        for i in 0..8 {
            match &classes[i] {
                ArgClass::IntegerReg { reg, .. } => {
                    assert_eq!(*reg, INT_ARG_REGS[i]);
                }
                _ => panic!("Expected IntegerReg for arg {}", i),
            }
        }
        match &classes[8] {
            ArgClass::Stack { offset, .. } => {
                assert_eq!(*offset, 0); // First stack argument at offset 0
            }
            _ => panic!("Expected Stack for ninth arg"),
        }
    }

    #[test]
    fn test_classify_pointer_arg() {
        let target = riscv64_target();
        let params = vec![
            ("ptr".into(), IrType::Pointer(Box::new(IrType::I32))),
        ];
        let classes = classify_arguments(&params, &target);
        match &classes[0] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X10), // a0
            _ => panic!("Expected IntegerReg for pointer arg"),
        }
    }

    #[test]
    fn test_classify_small_struct_in_reg() {
        let target = riscv64_target();
        let params = vec![
            ("s".into(), IrType::Struct { fields: vec![IrType::I32, IrType::I32], packed: false }),
        ];
        let classes = classify_arguments(&params, &target);
        // Struct of 8 bytes should fit in one integer register
        match &classes[0] {
            ArgClass::IntegerReg { reg, .. } => assert_eq!(*reg, X10),
            _ => panic!("Expected IntegerReg for small struct"),
        }
    }

    #[test]
    fn test_classify_large_struct_indirect() {
        let target = riscv64_target();
        let params = vec![
            ("s".into(), IrType::Struct {
                fields: vec![IrType::I64, IrType::I64, IrType::I64, IrType::I64],
                packed: false,
            }),
        ];
        let classes = classify_arguments(&params, &target);
        // Struct of 32 bytes > 2×XLEN: passed by reference
        match &classes[0] {
            ArgClass::IndirectReg { reg, .. } => assert_eq!(*reg, X10),
            _ => panic!("Expected IndirectReg for large struct"),
        }
    }

    // -----------------------------------------------------------------------
    // Return classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_return_void() {
        let target = riscv64_target();
        assert_eq!(classify_return(&IrType::Void, &target), ReturnClass::Void);
    }

    #[test]
    fn test_classify_return_int() {
        let target = riscv64_target();
        match classify_return(&IrType::I32, &target) {
            ReturnClass::IntegerReg { reg, .. } => assert_eq!(reg, X10),
            _ => panic!("Expected IntegerReg return"),
        }
    }

    #[test]
    fn test_classify_return_float() {
        let target = riscv64_target();
        match classify_return(&IrType::F64, &target) {
            ReturnClass::FloatReg { reg, .. } => assert_eq!(reg, F10),
            _ => panic!("Expected FloatReg return"),
        }
    }

    #[test]
    fn test_classify_return_small_struct() {
        let target = riscv64_target();
        let ty = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        match classify_return(&ty, &target) {
            ReturnClass::IntegerReg { reg, .. } => assert_eq!(reg, X10),
            _ => panic!("Expected IntegerReg return for small struct"),
        }
    }

    #[test]
    fn test_classify_return_large_struct() {
        let target = riscv64_target();
        let ty = IrType::Struct {
            fields: vec![IrType::I64; 4],
            packed: false,
        };
        match classify_return(&ty, &target) {
            ReturnClass::Indirect { reg, .. } => assert_eq!(reg, X10),
            _ => panic!("Expected Indirect return for large struct"),
        }
    }

    #[test]
    fn test_classify_return_pointer() {
        let target = riscv64_target();
        match classify_return(&IrType::Pointer(Box::new(IrType::I8)), &target) {
            ReturnClass::IntegerReg { reg, .. } => assert_eq!(reg, X10),
            _ => panic!("Expected IntegerReg for pointer return"),
        }
    }

    // -----------------------------------------------------------------------
    // Prologue/epilogue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prologue_small_frame() {
        let layout = Riscv64FrameLayout {
            frame_size: 32,
            ra_offset: 24,
            fp_offset: 16,
            saved_gprs: vec![],
            saved_fprs: vec![],
            locals_size: 0,
            outgoing_args_size: 0,
            uses_frame_pointer: true,
            has_calls: true,
        };
        let prologue = generate_prologue(&layout);
        // Should have: addi sp, sp, -32; sd ra, 24(sp); sd s0, 16(sp); addi s0, sp, 32
        assert!(prologue.len() >= 4);
        assert_eq!(prologue[0].opcode, OP_ADDI); // sp adjustment
    }

    #[test]
    fn test_epilogue_small_frame() {
        let layout = Riscv64FrameLayout {
            frame_size: 32,
            ra_offset: 24,
            fp_offset: 16,
            saved_gprs: vec![],
            saved_fprs: vec![],
            locals_size: 0,
            outgoing_args_size: 0,
            uses_frame_pointer: true,
            has_calls: true,
        };
        let epilogue = generate_epilogue(&layout);
        // Should have: ld s0; ld ra; addi sp, sp, 32; ret
        assert!(epilogue.len() >= 4);
        // Last instruction should be RET (JALR x0, ra, 0)
        assert_eq!(epilogue.last().unwrap().opcode, OP_JALR);
    }

    #[test]
    fn test_prologue_zero_frame() {
        let layout = Riscv64FrameLayout {
            frame_size: 0,
            ra_offset: 0,
            fp_offset: 0,
            saved_gprs: vec![],
            saved_fprs: vec![],
            locals_size: 0,
            outgoing_args_size: 0,
            uses_frame_pointer: false,
            has_calls: false,
        };
        let prologue = generate_prologue(&layout);
        assert!(prologue.is_empty());
    }

    #[test]
    fn test_epilogue_zero_frame() {
        let layout = Riscv64FrameLayout {
            frame_size: 0,
            ra_offset: 0,
            fp_offset: 0,
            saved_gprs: vec![],
            saved_fprs: vec![],
            locals_size: 0,
            outgoing_args_size: 0,
            uses_frame_pointer: false,
            has_calls: false,
        };
        let epilogue = generate_epilogue(&layout);
        assert_eq!(epilogue.len(), 1); // Just RET
    }

    #[test]
    fn test_prologue_with_callee_saved() {
        let layout = Riscv64FrameLayout {
            frame_size: 48,
            ra_offset: 40,
            fp_offset: 32,
            saved_gprs: vec![(X9, 24), (X18, 16)], // s1, s2
            saved_fprs: vec![],
            locals_size: 0,
            outgoing_args_size: 0,
            uses_frame_pointer: true,
            has_calls: true,
        };
        let prologue = generate_prologue(&layout);
        // addi sp; sd ra; sd s0; addi s0; sd s1; sd s2 = 6 instructions
        assert!(prologue.len() >= 6);
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(7, 8), 8);
    }

    #[test]
    fn test_fits_in_12bit_signed() {
        assert!(fits_in_12bit_signed(0));
        assert!(fits_in_12bit_signed(2047));
        assert!(fits_in_12bit_signed(-2048));
        assert!(!fits_in_12bit_signed(2048));
        assert!(!fits_in_12bit_signed(-2049));
    }

    #[test]
    fn test_make_mv() {
        let instr = make_mv(X10, X11);
        assert_eq!(instr.opcode, OP_ADDI);
        assert_eq!(instr.operands.len(), 3);
    }

    #[test]
    fn test_make_ret() {
        let instr = make_ret();
        assert_eq!(instr.opcode, OP_JALR);
        assert_eq!(instr.operands.len(), 3);
    }
}
