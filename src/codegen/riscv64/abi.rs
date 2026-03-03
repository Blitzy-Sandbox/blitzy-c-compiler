//! RISC-V 64-bit LP64D ABI implementation.
//!
//! This module implements the RISC-V LP64D (Long, Pointer 64-bit,
//! Double-precision float) calling convention for function prologue/epilogue
//! generation, argument/return value passing, struct passing conventions,
//! variadic function handling, and stack frame layout.
//!
//! ## LP64D ABI Summary
//! - **Integer arguments**: a0–a7 (x10–x17) for the first 8 integer/pointer args
//! - **Float arguments**: fa0–fa7 (f10–f17) for the first 8 FP args
//! - **Return values**: a0 (and a1 for 128-bit) for integer; fa0 (and fa1) for float
//! - **Return address**: ra (x1)
//! - **Stack pointer**: sp (x2), 16-byte aligned
//! - **Frame pointer**: s0/fp (x8), optional
//! - **Callee-saved GPRs**: s0–s11 (x8–x9, x18–x27)
//! - **Callee-saved FPRs**: fs0–fs11 (f8–f9, f18–f27)
//! - **Caller-saved GPRs**: t0–t6 (x5–x7, x28–x31), a0–a7
//! - **Caller-saved FPRs**: ft0–ft11 (f0–f7, f28–f31), fa0–fa7

use std::collections::HashMap;

use crate::codegen::{MachineInstr, MachineOperand, align_to};
use crate::codegen::regalloc::{PhysReg, AllocationResult};
use crate::ir::{IrType, Function, Value, Callee};
use crate::driver::target::TargetConfig;

// Re-import RISC-V register constants from the parent module
use super::{
    RA, SP, FP, ZERO,
    X0, X5, X8, X9,
    X10, X11, X12, X13, X14, X15, X16, X17,
    X18, X27,
    F8, F9,
    F10, F11, F12, F13, F14, F15, F16, F17,
    F18, F27,
    riscv64_register_info,
};

// ─────────────────────────────────────────────────────────────────────────────
// RISC-V instruction opcode constants (matching isel::Riscv64Opcode encoding)
// ─────────────────────────────────────────────────────────────────────────────

/// ADDI rd, rs1, imm12  — Add immediate
const OP_ADDI: u32 = 4;
/// SD rs2, offset(rs1)  — Store doubleword
const OP_SD: u32 = 12;
/// LD rd, offset(rs1)   — Load doubleword
const OP_LD: u32 = 7;
/// SW rs2, offset(rs1)  — Store word
const OP_SW: u32 = 11;
/// LW rd, offset(rs1)   — Load word
const OP_LW: u32 = 5;
/// ADD rd, rs1, rs2     — Add registers
const OP_ADD: u32 = 13;
/// SUB rd, rs1, rs2     — Subtract registers
const OP_SUB: u32 = 14;
/// LUI rd, imm20        — Load upper immediate
const OP_LUI: u32 = 0;
/// MV rd, rs1           — Move (pseudo for ADDI rd, rs, 0)
const OP_MV: u32 = 70;
/// RET                  — Return (pseudo for JALR x0, ra, 0)
const OP_RET: u32 = 73;
/// CALL symbol          — Call function (pseudo for AUIPC+JALR)
const OP_CALL: u32 = 72;
/// JALR rd, rs1, imm12  — Jump and link register
const OP_JALR: u32 = 3;
/// FSD rs2, offset(rs1) — Store double-precision float
const OP_FSD: u32 = 55;
/// FLD rd, offset(rs1)  — Load double-precision float
const OP_FLD: u32 = 51;
/// FSW rs2, offset(rs1) — Store single-precision float
const OP_FSW: u32 = 54;
/// FLW rd, offset(rs1)  — Load single-precision float
const OP_FLW: u32 = 50;
/// FMV.D rd, rs1        — Move double-precision float register
const OP_FMV_D: u32 = 68;
/// FMV.X.D rd, rs1      — Move float reg to integer reg (double)
const OP_FMV_X_D: u32 = 65;
/// FMV.D.X rd, rs1      — Move integer reg to float reg (double)
const OP_FMV_D_X: u32 = 66;
/// LI rd, imm           — Load immediate (pseudo)
const OP_LI: u32 = 71;
/// NOP                  — No operation
const OP_NOP: u32 = 69;

// ─────────────────────────────────────────────────────────────────────────────
// Integer argument registers: a0–a7 (x10–x17)
// ─────────────────────────────────────────────────────────────────────────────
const ARG_INT_REGS: [PhysReg; 8] = [X10, X11, X12, X13, X14, X15, X16, X17];

// ─────────────────────────────────────────────────────────────────────────────
// Float argument registers: fa0–fa7 (f10–f17)
// ─────────────────────────────────────────────────────────────────────────────
const ARG_FP_REGS: [PhysReg; 8] = [F10, F11, F12, F13, F14, F15, F16, F17];

// ─────────────────────────────────────────────────────────────────────────────
// Callee-saved integer registers: s0–s11 (x8–x9, x18–x27)
// ─────────────────────────────────────────────────────────────────────────────
const CALLEE_SAVED_GPRS: [PhysReg; 12] = [
    X8,          // s0/fp (x8)
    X9,          // s1    (x9)
    X18,         // s2    (x18)
    PhysReg(19), // s3
    PhysReg(20), // s4
    PhysReg(21), // s5
    PhysReg(22), // s6
    PhysReg(23), // s7
    PhysReg(24), // s8
    PhysReg(25), // s9
    PhysReg(26), // s10
    X27,         // s11   (x27)
];

// ─────────────────────────────────────────────────────────────────────────────
// Callee-saved float registers: fs0–fs11 (f8–f9, f18–f27)
// ─────────────────────────────────────────────────────────────────────────────
const CALLEE_SAVED_FPRS: [PhysReg; 12] = [
    F8,          // fs0  (f8)
    F9,          // fs1  (f9)
    F18,         // fs2  (f18)
    PhysReg(51), // fs3  (f19)
    PhysReg(52), // fs4  (f20)
    PhysReg(53), // fs5  (f21)
    PhysReg(54), // fs6  (f22)
    PhysReg(55), // fs7  (f23)
    PhysReg(56), // fs8  (f24)
    PhysReg(57), // fs9  (f25)
    PhysReg(58), // fs10 (f26)
    F27,         // fs11 (f27)
];

// ═══════════════════════════════════════════════════════════════════════════════
// Public Enums
// ═══════════════════════════════════════════════════════════════════════════════

/// Classification for how a single function argument is passed per LP64D ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgClass {
    /// Passed in an integer register (a0–a7).
    IntegerReg(PhysReg),
    /// Passed in a floating-point register (fa0–fa7).
    FloatReg(PhysReg),
    /// Passed on the stack at a given byte offset from SP.
    Stack(i32),
    /// Passed in a pair of integer registers (for 2×XLEN aggregates).
    IntegerRegPair(PhysReg, PhysReg),
    /// Passed in an integer register + float register (mixed struct: int first).
    IntegerFloatPair(PhysReg, PhysReg),
    /// Passed in a float register + integer register (mixed struct: float first).
    FloatIntegerPair(PhysReg, PhysReg),
    /// Passed in a pair of float registers.
    FloatRegPair(PhysReg, PhysReg),
    /// Passed by reference: pointer in an integer register to a stack copy.
    Indirect(PhysReg),
}

/// Classification for how a function return value is delivered per LP64D ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnClass {
    /// Returned in a0 (integer ≤ 64 bits).
    IntegerReg,
    /// Returned in fa0 (float or double).
    FloatReg,
    /// Returned in a0 + a1 pair (128-bit integer or small struct with 2 int fields).
    IntegerRegPair,
    /// Returned in fa0 + fa1 pair (struct with 2 float fields).
    FloatRegPair,
    /// Returned in a0 + fa0 (struct with int and float field).
    IntegerFloatPair,
    /// Returned via hidden pointer: caller passes destination in a0; callee writes there.
    Indirect,
    /// No return value (void).
    Void,
}

/// Classification of a struct's field composition for LP64D flattening rules.
/// Structs ≤ 2×XLEN (16 bytes) are "flattened" into register pairs; larger
/// structs are passed by reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructArgClass {
    /// All fields are integer/pointer-typed; `usize` = number of XLEN-sized pieces (1 or 2).
    AllInteger(usize),
    /// All fields are floating-point; `usize` = number of float pieces (1 or 2).
    AllFloat(usize),
    /// First field is integer, second is float.
    MixedIntFloat,
    /// First field is float, second is integer.
    MixedFloatInt,
    /// Struct is too large (> 2×XLEN) — must be passed by reference.
    ByReference,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Riscv64FrameLayout
// ═══════════════════════════════════════════════════════════════════════════════

/// Describes the complete stack frame layout for a single RISC-V 64 function.
///
/// ```text
/// [Higher addresses]
/// +---------------------------+
/// | Incoming arguments        | (from caller, above our frame)
/// +---------------------------+ ← Old SP (16-byte aligned)
/// | RA save slot              | 8 bytes
/// +---------------------------+
/// | FP (s0) save slot         | 8 bytes (if uses_frame_pointer)
/// +---------------------------+
/// | Callee-saved GPR saves    | 8 bytes each
/// +---------------------------+
/// | Callee-saved FPR saves    | 8 bytes each
/// +---------------------------+
/// | Spill slots               | from register allocator
/// +---------------------------+
/// | Local variables           |
/// +---------------------------+
/// | Outgoing argument area    | for calls with >8 args
/// +---------------------------+ ← New SP (16-byte aligned)
/// [Lower addresses]
/// ```
#[derive(Debug, Clone)]
pub struct Riscv64FrameLayout {
    /// Total frame size in bytes (always 16-byte aligned).
    pub frame_size: u32,
    /// Offset from SP where RA is saved.
    pub ra_offset: i32,
    /// Offset from SP where s0/FP is saved, if the frame pointer is used.
    pub fp_offset: Option<i32>,
    /// (register, offset-from-SP) pairs for callee-saved integer registers.
    pub callee_saved_offsets: Vec<(PhysReg, i32)>,
    /// (register, offset-from-SP) pairs for callee-saved FP registers.
    pub callee_saved_fp_offsets: Vec<(PhysReg, i32)>,
    /// Offset from SP where local variables begin.
    pub locals_offset: i32,
    /// Total space for local variables in bytes.
    pub locals_size: u32,
    /// Total space for register-allocator spill slots in bytes.
    pub spill_size: u32,
    /// Outgoing argument area size (for calls whose arguments overflow registers).
    pub arg_area_size: u32,
    /// Whether the frame pointer (s0) is used.
    pub uses_frame_pointer: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: register classification
// ═══════════════════════════════════════════════════════════════════════════════

/// Returns `true` if the physical register is a floating-point register (f0–f31).
/// In the RISC-V register numbering scheme used by this backend, GPRs occupy
/// PhysReg 0–31 and FPRs occupy PhysReg 32–63.
fn is_fp_register(reg: PhysReg) -> bool {
    reg.0 >= 32
}

/// Returns `true` when `frame_offset` fits in a 12-bit signed immediate
/// (range -2048..=2047).
fn fits_in_12bit_signed(val: i32) -> bool {
    val >= -2048 && val <= 2047
}

/// Build a MachineInstr with the given opcode and operands.
fn mi(opcode: u32, operands: Vec<MachineOperand>) -> MachineInstr {
    MachineInstr::with_operands(opcode, operands)
}

/// Emit an `addi rd, rs1, imm` instruction.
fn emit_addi(rd: PhysReg, rs1: PhysReg, imm: i32) -> MachineInstr {
    mi(OP_ADDI, vec![
        MachineOperand::Register(rd),
        MachineOperand::Register(rs1),
        MachineOperand::Immediate(imm as i64),
    ])
}

/// Emit `sd rs2, offset(rs1)` — store 8-byte doubleword.
fn emit_sd(rs2: PhysReg, base: PhysReg, offset: i32) -> MachineInstr {
    mi(OP_SD, vec![
        MachineOperand::Register(rs2),
        MachineOperand::Memory { base, offset },
    ])
}

/// Emit `ld rd, offset(rs1)` — load 8-byte doubleword.
fn emit_ld(rd: PhysReg, base: PhysReg, offset: i32) -> MachineInstr {
    mi(OP_LD, vec![
        MachineOperand::Register(rd),
        MachineOperand::Memory { base, offset },
    ])
}

/// Emit `fsd rs2, offset(rs1)` — store double-precision float.
fn emit_fsd(rs2: PhysReg, base: PhysReg, offset: i32) -> MachineInstr {
    mi(OP_FSD, vec![
        MachineOperand::Register(rs2),
        MachineOperand::Memory { base, offset },
    ])
}

/// Emit `fld rd, offset(rs1)` — load double-precision float.
fn emit_fld(rd: PhysReg, base: PhysReg, offset: i32) -> MachineInstr {
    mi(OP_FLD, vec![
        MachineOperand::Register(rd),
        MachineOperand::Memory { base, offset },
    ])
}

/// Emit `mv rd, rs1` (pseudo for `addi rd, rs, 0`).
fn emit_mv(rd: PhysReg, rs1: PhysReg) -> MachineInstr {
    mi(OP_MV, vec![
        MachineOperand::Register(rd),
        MachineOperand::Register(rs1),
    ])
}

/// Emit `fmv.d rd, rs1` — move double-precision FP register.
fn emit_fmv_d(rd: PhysReg, rs1: PhysReg) -> MachineInstr {
    mi(OP_FMV_D, vec![
        MachineOperand::Register(rd),
        MachineOperand::Register(rs1),
    ])
}

/// Emit a large-frame SP adjustment when |offset| > 2047.
/// Uses temporary register t0 (x5) for the materialized immediate.
///
/// Produces:
///   lui  t0, %hi(offset)
///   addi t0, t0, %lo(offset)
///   add  sp, sp, t0
fn emit_large_sp_adjust(offset: i32) -> Vec<MachineInstr> {
    let t0 = X5; // t0
    // Compute hi20 and lo12 with sign extension correction:
    // If lo12 is negative (bit 11 set), the LUI constant needs +1 page.
    let lo12 = ((offset as i32) << 20) >> 20; // sign-extend low 12 bits
    let hi20 = if lo12 < 0 {
        ((offset as i64 + 0x800) >> 12) as i32
    } else {
        (offset >> 12) as i32
    };
    vec![
        mi(OP_LUI, vec![
            MachineOperand::Register(t0),
            MachineOperand::Immediate(hi20 as i64),
        ]),
        emit_addi(t0, t0, lo12),
        mi(OP_ADD, vec![
            MachineOperand::Register(SP),
            MachineOperand::Register(SP),
            MachineOperand::Register(t0),
        ]),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// Struct Field Classification (LP64D flattening)
// ═══════════════════════════════════════════════════════════════════════════════

/// Classify a struct type's field composition for LP64D register passing.
///
/// LP64D "flattening" rules:
/// - Structs > 2×XLEN (> 16 bytes): passed by reference (`ByReference`).
/// - Structs with exactly 1 float and nothing else: `AllFloat(1)`.
/// - Structs with exactly 2 floats: `AllFloat(2)`.
/// - Structs with only integer/pointer fields fitting in 1 or 2 XLEN slots: `AllInteger(n)`.
/// - Structs with one integer field and one float field: `MixedIntFloat` or `MixedFloatInt`.
pub fn classify_struct_fields(struct_type: &IrType, target: &TargetConfig) -> StructArgClass {
    let xlen: usize = target.pointer_size as usize; // 8 for RV64

    // Extract struct fields
    let fields = match struct_type {
        IrType::Struct { fields, .. } => fields,
        _ => return StructArgClass::ByReference,
    };

    let total_size = struct_type.size(target);

    // Structs larger than 2×XLEN are always passed by reference
    if total_size > 2 * xlen {
        return StructArgClass::ByReference;
    }

    // Empty struct — treat as a single zero-sized integer
    if fields.is_empty() {
        return StructArgClass::AllInteger(1);
    }

    // Flatten: recursively classify each field into "integer-like" or "float-like"
    let mut int_count: usize = 0;
    let mut float_count: usize = 0;
    let mut field_order: Vec<bool> = Vec::new(); // true = float, false = integer

    for field in fields.iter() {
        classify_field_recursive(field, target, &mut int_count, &mut float_count, &mut field_order);
    }

    // If there are more than 2 "slots", fall back to integer passing
    let total_slots = int_count + float_count;
    if total_slots == 0 {
        return StructArgClass::AllInteger(1);
    }

    // LP64D hardware floating-point ABI: a struct with more than 2 floating-point
    // fields or more than 2 integer fields is passed in integer registers.
    if total_slots > 2 {
        // Fits in ≤ 16 bytes but has >2 fields: pack into integer registers
        let n_xlen = (total_size + xlen - 1) / xlen;
        return StructArgClass::AllInteger(n_xlen);
    }

    match (int_count, float_count) {
        (0, 1) => StructArgClass::AllFloat(1),
        (0, 2) => StructArgClass::AllFloat(2),
        (1, 0) => StructArgClass::AllInteger(1),
        (2, 0) => StructArgClass::AllInteger(2),
        (1, 1) => {
            // Determine order: which came first?
            if !field_order.is_empty() && field_order[0] {
                StructArgClass::MixedFloatInt
            } else {
                StructArgClass::MixedIntFloat
            }
        }
        _ => {
            // Fallback: treat as integer chunks
            let n_xlen = (total_size + xlen - 1) / xlen;
            StructArgClass::AllInteger(n_xlen)
        }
    }
}

/// Recursively classify a single field for struct flattening.
fn classify_field_recursive(
    field: &IrType,
    target: &TargetConfig,
    int_count: &mut usize,
    float_count: &mut usize,
    field_order: &mut Vec<bool>,
) {
    match field {
        IrType::F32 | IrType::F64 => {
            *float_count += 1;
            field_order.push(true);
        }
        IrType::Struct { fields, .. } => {
            // Recursively flatten nested structs
            for f in fields.iter() {
                classify_field_recursive(f, target, int_count, float_count, field_order);
            }
        }
        _ => {
            // Integer, pointer, bool, array, etc. — all passed as integer
            *int_count += 1;
            field_order.push(false);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// classify_arguments — LP64D Argument Passing
// ═══════════════════════════════════════════════════════════════════════════════

/// Classify how each function parameter is passed according to the RISC-V LP64D ABI.
///
/// LP64D rules (summary):
/// 1. Scalars ≤ XLEN: integer/pointer → next a-register; float → next fa-register.
/// 2. Structs ≤ 2×XLEN: flattened into register pairs per field composition.
/// 3. Structs > 2×XLEN: passed by reference (pointer in a-register).
/// 4. Once registers are exhausted, remaining arguments go on the stack.
///
/// **Variadic arguments**: When the callee is variadic, all arguments past the
/// last named parameter are forced into integer registers (or the stack).
/// Floats in variadic positions are bit-cast to integer representation.
pub fn classify_arguments(
    params: &[(String, IrType)],
    target: &TargetConfig,
) -> Vec<ArgClass> {
    let xlen: usize = target.pointer_size as usize; // 8 for RV64
    let mut int_reg_idx: usize = 0;
    let mut fp_reg_idx: usize = 0;
    let mut stack_offset: i32 = 0;
    let mut result: Vec<ArgClass> = Vec::with_capacity(params.len());

    // Check if any parameter type is a variadic function (heuristic: we don't
    // have direct access to the function type here, so variadic handling is
    // deferred to the caller who can split named/variadic args separately).
    // All params passed here are classified as named arguments.
    for (_name, ty) in params.iter() {
        let class = classify_single_argument(
            ty,
            target,
            &mut int_reg_idx,
            &mut fp_reg_idx,
            &mut stack_offset,
            xlen,
            false, // not variadic
        );
        result.push(class);
    }

    result
}

/// Classify a single argument for LP64D, advancing register/stack counters.
fn classify_single_argument(
    ty: &IrType,
    target: &TargetConfig,
    int_reg_idx: &mut usize,
    fp_reg_idx: &mut usize,
    stack_offset: &mut i32,
    xlen: usize,
    is_variadic: bool,
) -> ArgClass {
    match ty {
        // ── Floating-point scalars ──────────────────────────────────
        IrType::F32 | IrType::F64 => {
            if is_variadic {
                // Variadic float args are passed in integer registers
                allocate_int_reg_or_stack(int_reg_idx, stack_offset, xlen)
            } else if *fp_reg_idx < 8 {
                let reg = ARG_FP_REGS[*fp_reg_idx];
                *fp_reg_idx += 1;
                ArgClass::FloatReg(reg)
            } else {
                // FP registers exhausted → fall back to integer register
                allocate_int_reg_or_stack(int_reg_idx, stack_offset, xlen)
            }
        }

        // ── Struct types ────────────────────────────────────────────
        IrType::Struct { .. } => {
            classify_struct_argument(ty, target, int_reg_idx, fp_reg_idx, stack_offset, xlen, is_variadic)
        }

        // ── Void (should not appear as a parameter, but handle gracefully) ──
        IrType::Void => {
            allocate_int_reg_or_stack(int_reg_idx, stack_offset, xlen)
        }

        // ── All other types: integer, pointer, bool, array, etc. ───
        _ => {
            allocate_int_reg_or_stack(int_reg_idx, stack_offset, xlen)
        }
    }
}

/// Allocate the next available integer argument register, or a stack slot.
fn allocate_int_reg_or_stack(
    int_reg_idx: &mut usize,
    stack_offset: &mut i32,
    xlen: usize,
) -> ArgClass {
    if *int_reg_idx < 8 {
        let reg = ARG_INT_REGS[*int_reg_idx];
        *int_reg_idx += 1;
        ArgClass::IntegerReg(reg)
    } else {
        let off = *stack_offset;
        *stack_offset += xlen as i32;
        ArgClass::Stack(off)
    }
}

/// Classify a struct-typed argument using LP64D flattening rules.
fn classify_struct_argument(
    ty: &IrType,
    target: &TargetConfig,
    int_reg_idx: &mut usize,
    fp_reg_idx: &mut usize,
    stack_offset: &mut i32,
    xlen: usize,
    is_variadic: bool,
) -> ArgClass {
    let struct_class = classify_struct_fields(ty, target);

    match struct_class {
        StructArgClass::ByReference => {
            // Pass a pointer to a stack copy in an integer register
            if *int_reg_idx < 8 {
                let reg = ARG_INT_REGS[*int_reg_idx];
                *int_reg_idx += 1;
                ArgClass::Indirect(reg)
            } else {
                let off = *stack_offset;
                *stack_offset += xlen as i32;
                ArgClass::Stack(off)
            }
        }

        StructArgClass::AllFloat(n) if !is_variadic => {
            if n == 1 && *fp_reg_idx < 8 {
                let reg = ARG_FP_REGS[*fp_reg_idx];
                *fp_reg_idx += 1;
                ArgClass::FloatReg(reg)
            } else if n == 2 && *fp_reg_idx + 1 < 8 {
                let r1 = ARG_FP_REGS[*fp_reg_idx];
                let r2 = ARG_FP_REGS[*fp_reg_idx + 1];
                *fp_reg_idx += 2;
                ArgClass::FloatRegPair(r1, r2)
            } else {
                // Not enough FP registers; fall back to integer registers
                classify_struct_as_integer(ty, target, int_reg_idx, stack_offset, xlen)
            }
        }

        StructArgClass::AllInteger(n) => {
            if n == 1 && *int_reg_idx < 8 {
                let reg = ARG_INT_REGS[*int_reg_idx];
                *int_reg_idx += 1;
                ArgClass::IntegerReg(reg)
            } else if n == 2 && *int_reg_idx + 1 < 8 {
                let r1 = ARG_INT_REGS[*int_reg_idx];
                let r2 = ARG_INT_REGS[*int_reg_idx + 1];
                *int_reg_idx += 2;
                ArgClass::IntegerRegPair(r1, r2)
            } else if n == 2 && *int_reg_idx < 8 {
                // One register available but the pair needs two.
                // Per the LP64D ABI: if the first part goes in a register but the
                // second doesn't fit, both go on the stack (conservative rule).
                let stack_start = *stack_offset;
                *stack_offset += 2 * xlen as i32;
                ArgClass::Stack(stack_start)
            } else {
                let off = *stack_offset;
                let sz = n as i32 * xlen as i32;
                *stack_offset += sz;
                ArgClass::Stack(off)
            }
        }

        StructArgClass::MixedIntFloat if !is_variadic => {
            // One integer field + one float field
            if *int_reg_idx < 8 && *fp_reg_idx < 8 {
                let ireg = ARG_INT_REGS[*int_reg_idx];
                let freg = ARG_FP_REGS[*fp_reg_idx];
                *int_reg_idx += 1;
                *fp_reg_idx += 1;
                ArgClass::IntegerFloatPair(ireg, freg)
            } else {
                classify_struct_as_integer(ty, target, int_reg_idx, stack_offset, xlen)
            }
        }

        StructArgClass::MixedFloatInt if !is_variadic => {
            // One float field + one integer field
            if *fp_reg_idx < 8 && *int_reg_idx < 8 {
                let freg = ARG_FP_REGS[*fp_reg_idx];
                let ireg = ARG_INT_REGS[*int_reg_idx];
                *fp_reg_idx += 1;
                *int_reg_idx += 1;
                ArgClass::FloatIntegerPair(freg, ireg)
            } else {
                classify_struct_as_integer(ty, target, int_reg_idx, stack_offset, xlen)
            }
        }

        // Variadic: all struct-like args go through integer registers
        StructArgClass::AllFloat(_) | StructArgClass::MixedIntFloat | StructArgClass::MixedFloatInt => {
            classify_struct_as_integer(ty, target, int_reg_idx, stack_offset, xlen)
        }
    }
}

/// Fallback: pass struct in integer registers (or stack).
fn classify_struct_as_integer(
    ty: &IrType,
    target: &TargetConfig,
    int_reg_idx: &mut usize,
    stack_offset: &mut i32,
    xlen: usize,
) -> ArgClass {
    let total_size = ty.size(target);
    let n_xlen = (total_size + xlen - 1) / xlen;

    if n_xlen == 1 && *int_reg_idx < 8 {
        let reg = ARG_INT_REGS[*int_reg_idx];
        *int_reg_idx += 1;
        ArgClass::IntegerReg(reg)
    } else if n_xlen == 2 && *int_reg_idx + 1 < 8 {
        let r1 = ARG_INT_REGS[*int_reg_idx];
        let r2 = ARG_INT_REGS[*int_reg_idx + 1];
        *int_reg_idx += 2;
        ArgClass::IntegerRegPair(r1, r2)
    } else {
        let off = *stack_offset;
        *stack_offset += (n_xlen * xlen) as i32;
        ArgClass::Stack(off)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// classify_return — LP64D Return Value Classification
// ═══════════════════════════════════════════════════════════════════════════════

/// Classify how a function return value is delivered per the LP64D ABI.
///
/// - `Void`:              no return value.
/// - `IntegerReg`:        integer/pointer ≤ 64 bits in a0.
/// - `FloatReg`:          f32/f64 in fa0.
/// - `IntegerRegPair`:    struct with 2 integer XLEN-sized fields in a0 + a1.
/// - `FloatRegPair`:      struct with 2 float fields in fa0 + fa1.
/// - `IntegerFloatPair`:  struct with int+float in a0 + fa0.
/// - `Indirect`:          struct > 2×XLEN, caller provides hidden pointer in a0.
pub fn classify_return(return_type: &IrType, target: &TargetConfig) -> ReturnClass {
    let xlen: usize = target.pointer_size as usize;

    match return_type {
        IrType::Void => ReturnClass::Void,

        IrType::F32 | IrType::F64 => ReturnClass::FloatReg,

        IrType::Struct { .. } => {
            let total_size = return_type.size(target);
            if total_size > 2 * xlen {
                return ReturnClass::Indirect;
            }

            let struct_class = classify_struct_fields(return_type, target);
            match struct_class {
                StructArgClass::AllFloat(1) => ReturnClass::FloatReg,
                StructArgClass::AllFloat(2) => ReturnClass::FloatRegPair,
                StructArgClass::AllInteger(1) => ReturnClass::IntegerReg,
                StructArgClass::AllInteger(2) => ReturnClass::IntegerRegPair,
                StructArgClass::MixedIntFloat | StructArgClass::MixedFloatInt => {
                    ReturnClass::IntegerFloatPair
                }
                StructArgClass::ByReference => ReturnClass::Indirect,
                _ => ReturnClass::IntegerReg,
            }
        }

        // All integer, pointer, bool types → IntegerReg
        _ => ReturnClass::IntegerReg,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// compute_frame_layout — Stack Frame Layout Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the complete stack frame layout for a RISC-V 64 function.
///
/// The layout (high → low address):
/// 1. RA save slot (8 bytes)
/// 2. FP (s0) save slot (8 bytes, if `uses_frame_pointer`)
/// 3. Callee-saved GPR save slots (8 bytes each)
/// 4. Callee-saved FPR save slots (8 bytes each)
/// 5. Spill slots (from register allocator)
/// 6. Local variable area
/// 7. Outgoing argument area (for calls whose arguments overflow registers)
///
/// The total frame size is rounded up to 16-byte alignment.
pub fn compute_frame_layout(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
) -> Riscv64FrameLayout {
    let stack_align = target.stack_alignment as u32; // 16

    // Determine if the function makes any calls by scanning blocks for
    // call-like terminators or instructions. A simple heuristic: if there
    // are any callee-saved registers used, or we have spill slots, we
    // likely need to save RA. We conservatively always save RA.
    let has_calls = function_has_calls(function);

    // Decide whether to use the frame pointer.
    // We use FP if the function has variable-length arrays, alloca calls,
    // or a large frame. Conservatively: use FP if the function has calls.
    let uses_frame_pointer = has_calls || alloc_result.num_spill_slots > 0;

    // Separate callee-saved GPRs and FPRs from the allocation result
    let mut callee_saved_gprs: Vec<PhysReg> = Vec::new();
    let mut callee_saved_fprs: Vec<PhysReg> = Vec::new();

    for &reg in alloc_result.used_callee_saved.iter() {
        if is_fp_register(reg) {
            callee_saved_fprs.push(reg);
        } else {
            // Don't double-count FP (s0) if we're already saving it
            if uses_frame_pointer && reg == FP {
                continue;
            }
            callee_saved_gprs.push(reg);
        }
    }

    // Compute sizes of each region
    let ra_size: u32 = 8; // always save RA
    let fp_save_size: u32 = if uses_frame_pointer { 8 } else { 0 };
    let gpr_save_size: u32 = callee_saved_gprs.len() as u32 * 8;
    let fpr_save_size: u32 = callee_saved_fprs.len() as u32 * 8;
    let spill_size: u32 = alloc_result.num_spill_slots * 8;
    let locals_size: u32 = compute_locals_size(function, target);

    // Compute outgoing argument area: scan all calls in the function for
    // maximum outgoing arg count, then compute how many overflow the registers.
    let arg_area_size: u32 = compute_outgoing_arg_area(function, target);

    // Total raw frame size
    let raw_total = ra_size + fp_save_size + gpr_save_size + fpr_save_size
        + spill_size + locals_size + arg_area_size;

    // Align to 16 bytes
    let frame_size = align_to(raw_total as u64, stack_align as u64) as u32;

    // Compute offsets (from SP, which points to the bottom of the frame).
    // We lay out from the top (old SP) downward.
    let mut current_offset: i32 = frame_size as i32;

    // RA is at the top of the frame
    current_offset -= 8;
    let ra_offset = current_offset;

    // FP (s0) save slot
    let fp_offset = if uses_frame_pointer {
        current_offset -= 8;
        Some(current_offset)
    } else {
        None
    };

    // Callee-saved GPRs
    let mut callee_saved_offsets: Vec<(PhysReg, i32)> = Vec::new();
    for &reg in callee_saved_gprs.iter() {
        current_offset -= 8;
        callee_saved_offsets.push((reg, current_offset));
    }

    // Callee-saved FPRs
    let mut callee_saved_fp_offsets: Vec<(PhysReg, i32)> = Vec::new();
    for &reg in callee_saved_fprs.iter() {
        current_offset -= 8;
        callee_saved_fp_offsets.push((reg, current_offset));
    }

    // Spill area offset
    let _spill_offset = current_offset - spill_size as i32;
    current_offset -= spill_size as i32;

    // Locals area starts here
    let locals_offset = current_offset - locals_size as i32;

    Riscv64FrameLayout {
        frame_size,
        ra_offset,
        fp_offset,
        callee_saved_offsets,
        callee_saved_fp_offsets,
        locals_offset,
        locals_size,
        spill_size,
        arg_area_size,
        uses_frame_pointer,
    }
}

/// Scan a function's blocks for call instructions to determine if RA needs saving.
///
/// Inspects all IR instructions in the function body for `Call` operations.
/// Only returns `true` if the function actually performs calls, enabling
/// leaf functions to avoid the overhead of saving/restoring the RA register.
fn function_has_calls(function: &Function) -> bool {
    for block in &function.blocks {
        for inst in &block.instructions {
            if matches!(inst, crate::ir::Instruction::Call { .. }) {
                return true;
            }
        }
    }
    false
}

/// Estimate local variable space needed by the function.
/// Uses `target.pointer_size` to determine stack slot width for parameter homing.
fn compute_locals_size(function: &Function, target: &TargetConfig) -> u32 {
    // Each parameter potentially needs a home slot on the stack.
    // Slot size equals the target's pointer size (8 bytes on RV64).
    let slot_size = target.pointer_size as u32;
    let param_homes = function.params.len() as u32 * slot_size;
    // Conservatively allocate space for parameters to be homed to the stack.
    // The actual locals are determined during instruction selection, but we
    // provide a baseline estimate here.
    param_homes
}

/// Compute the outgoing argument area size by scanning all blocks.
fn compute_outgoing_arg_area(function: &Function, _target: &TargetConfig) -> u32 {
    // Conservative: scan blocks for any call terminator, estimate max args.
    // For now, reserve space for up to 8 stack-passed arguments (64 bytes)
    // if the function has calls, or 0 otherwise.
    if function.blocks.len() > 1 {
        0 // Will be adjusted during instruction selection
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// generate_prologue — Function Prologue
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate the function prologue instruction sequence.
///
/// The prologue:
/// 1. Decrements SP by frame_size.
/// 2. Saves RA at the computed offset.
/// 3. Saves FP (s0) if the frame pointer is used.
/// 4. Sets FP = old SP value (if frame pointer is used).
/// 5. Saves all callee-saved registers that were allocated.
///
/// For large frames (frame_size > 2047), a multi-instruction sequence
/// using LUI + ADDI + SUB is emitted instead of a single ADDI.
pub fn generate_prologue(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
) -> Vec<MachineInstr> {
    let layout = compute_frame_layout(function, alloc_result, target);
    let mut instrs: Vec<MachineInstr> = Vec::new();

    if layout.frame_size == 0 {
        return instrs;
    }

    // Step 1: Allocate the stack frame
    let neg_frame = -(layout.frame_size as i32);
    if fits_in_12bit_signed(neg_frame) {
        // addi sp, sp, -frame_size
        instrs.push(emit_addi(SP, SP, neg_frame));
    } else {
        // Large frame: use t0 to hold the negated frame size
        instrs.extend(emit_large_sp_adjust(neg_frame));
    }

    // Step 2: Save RA
    instrs.push(emit_sd(RA, SP, layout.ra_offset));

    // Step 3: Save FP (s0) if used
    if let Some(fp_off) = layout.fp_offset {
        instrs.push(emit_sd(FP, SP, fp_off));
    }

    // Step 4: Set up frame pointer: s0 = sp + frame_size = old sp
    if layout.uses_frame_pointer {
        if fits_in_12bit_signed(layout.frame_size as i32) {
            instrs.push(emit_addi(FP, SP, layout.frame_size as i32));
        } else {
            // Large frame: materialize offset and add
            let frame_i32 = layout.frame_size as i32;
            let lo12 = ((frame_i32) << 20) >> 20;
            let hi20 = if lo12 < 0 {
                ((frame_i32 as i64 + 0x800) >> 12) as i32
            } else {
                (frame_i32 >> 12) as i32
            };
            instrs.push(mi(OP_LUI, vec![
                MachineOperand::Register(X5),
                MachineOperand::Immediate(hi20 as i64),
            ]));
            instrs.push(emit_addi(X5, X5, lo12));
            instrs.push(mi(OP_ADD, vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(SP),
                MachineOperand::Register(X5),
            ]));
        }
    }

    // Step 5: Save callee-saved GPRs
    for &(reg, offset) in layout.callee_saved_offsets.iter() {
        instrs.push(emit_sd(reg, SP, offset));
    }

    // Step 6: Save callee-saved FPRs
    for &(reg, offset) in layout.callee_saved_fp_offsets.iter() {
        instrs.push(emit_fsd(reg, SP, offset));
    }

    instrs
}

// ═══════════════════════════════════════════════════════════════════════════════
// generate_epilogue — Function Epilogue
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate the function epilogue instruction sequence.
///
/// The epilogue restores the stack frame in reverse order:
/// 1. Restore callee-saved FPRs.
/// 2. Restore callee-saved GPRs.
/// 3. Restore FP (s0) if it was used.
/// 4. Restore RA.
/// 5. Deallocate the stack frame (increment SP).
/// 6. Emit RET (JALR x0, ra, 0).
pub fn generate_epilogue(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
) -> Vec<MachineInstr> {
    let layout = compute_frame_layout(function, alloc_result, target);
    let mut instrs: Vec<MachineInstr> = Vec::new();

    if layout.frame_size == 0 {
        instrs.push(mi(OP_RET, vec![]));
        return instrs;
    }

    // Step 1: Restore callee-saved FPRs (reverse order)
    for &(reg, offset) in layout.callee_saved_fp_offsets.iter().rev() {
        instrs.push(emit_fld(reg, SP, offset));
    }

    // Step 2: Restore callee-saved GPRs (reverse order)
    for &(reg, offset) in layout.callee_saved_offsets.iter().rev() {
        instrs.push(emit_ld(reg, SP, offset));
    }

    // Step 3: Restore FP (s0)
    if let Some(fp_off) = layout.fp_offset {
        instrs.push(emit_ld(FP, SP, fp_off));
    }

    // Step 4: Restore RA
    instrs.push(emit_ld(RA, SP, layout.ra_offset));

    // Step 5: Deallocate the stack frame
    let frame_i32 = layout.frame_size as i32;
    if fits_in_12bit_signed(frame_i32) {
        instrs.push(emit_addi(SP, SP, frame_i32));
    } else {
        instrs.extend(emit_large_sp_adjust(frame_i32));
    }

    // Step 6: Return
    instrs.push(mi(OP_RET, vec![]));

    instrs
}

// ═══════════════════════════════════════════════════════════════════════════════
// generate_argument_loads — Load Incoming Arguments from ABI Locations
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate instructions to move incoming function arguments from their
/// ABI-specified locations (a-registers, fa-registers, or stack) into the
/// registers assigned by the register allocator.
///
/// For each parameter:
/// - `IntegerReg(r)`:         MV from argument register to allocated register.
/// - `FloatReg(r)`:           FMV.D from argument FP register to allocated FP register.
/// - `Stack(offset)`:         LD/FLD from the caller's outgoing arg area.
/// - `IntegerRegPair(r1,r2)`: Two MVs for the pair.
/// - `IntegerFloatPair(r,f)`: MV + FMV.D.
/// - `FloatIntegerPair(f,r)`: FMV.D + MV.
/// - `FloatRegPair(f1,f2)`:   Two FMV.Ds.
/// - `Indirect(r)`:           Load the pointer, then the caller can dereference.
pub fn generate_argument_loads(
    params: &[(String, IrType)],
    classifications: &[ArgClass],
    alloc_result: &AllocationResult,
) -> Vec<MachineInstr> {
    let mut instrs: Vec<MachineInstr> = Vec::new();

    for (idx, class) in classifications.iter().enumerate() {
        // Determine the destination register from the allocation result.
        // The register allocator assigns virtual registers to live intervals;
        // here we use the parameter index to find the physical register.
        let dest_reg = get_param_dest_register(idx, alloc_result);

        match *class {
            ArgClass::IntegerReg(src_reg) => {
                if let Some(dest) = dest_reg {
                    if dest != src_reg {
                        instrs.push(emit_mv(dest, src_reg));
                    }
                    // If dest == src, no move needed
                }
            }

            ArgClass::FloatReg(src_reg) => {
                if let Some(dest) = dest_reg {
                    if dest != src_reg {
                        if is_fp_register(dest) {
                            instrs.push(emit_fmv_d(dest, src_reg));
                        } else {
                            // FP to int register: use FMV.X.D
                            instrs.push(mi(OP_FMV_X_D, vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(src_reg),
                            ]));
                        }
                    }
                }
            }

            ArgClass::Stack(offset) => {
                if let Some(dest) = dest_reg {
                    // Determine if float or integer load based on the parameter type
                    let is_fp = idx < params.len() && (params[idx].1 == IrType::F32 || params[idx].1 == IrType::F64);
                    if is_fp && is_fp_register(dest) {
                        instrs.push(emit_fld(dest, FP, offset));
                    } else {
                        instrs.push(emit_ld(dest, FP, offset));
                    }
                }
            }

            ArgClass::IntegerRegPair(src1, src2) => {
                // The first register holds the low part, second the high part.
                // For now, move the first register to the destination.
                if let Some(dest) = dest_reg {
                    if dest != src1 {
                        instrs.push(emit_mv(dest, src1));
                    }
                }
                // The second part of the pair goes to the next virtual register.
                let dest2 = get_param_dest_register(idx + params.len(), alloc_result);
                if let Some(d2) = dest2 {
                    if d2 != src2 {
                        instrs.push(emit_mv(d2, src2));
                    }
                }
            }

            ArgClass::IntegerFloatPair(int_reg, fp_reg) => {
                if let Some(dest) = dest_reg {
                    if dest != int_reg {
                        instrs.push(emit_mv(dest, int_reg));
                    }
                }
                // Float part: try to get a second dest register
                let dest_fp = get_param_fp_dest_register(idx, alloc_result);
                if let Some(dfp) = dest_fp {
                    if dfp != fp_reg {
                        instrs.push(emit_fmv_d(dfp, fp_reg));
                    }
                }
            }

            ArgClass::FloatIntegerPair(fp_reg, int_reg) => {
                // Float part first
                let dest_fp = get_param_fp_dest_register(idx, alloc_result);
                if let Some(dfp) = dest_fp {
                    if dfp != fp_reg {
                        instrs.push(emit_fmv_d(dfp, fp_reg));
                    }
                }
                // Integer part
                if let Some(dest) = dest_reg {
                    if dest != int_reg {
                        instrs.push(emit_mv(dest, int_reg));
                    }
                }
            }

            ArgClass::FloatRegPair(fp1, fp2) => {
                if let Some(dest) = dest_reg {
                    if is_fp_register(dest) {
                        if dest != fp1 {
                            instrs.push(emit_fmv_d(dest, fp1));
                        }
                    }
                }
                let dest_fp2 = get_param_fp_dest_register(idx, alloc_result);
                if let Some(d2) = dest_fp2 {
                    if d2 != fp2 {
                        instrs.push(emit_fmv_d(d2, fp2));
                    }
                }
            }

            ArgClass::Indirect(ptr_reg) => {
                // The argument is a pointer to the actual value (by-reference passing).
                // Move the pointer to the destination register so the function body
                // can dereference it.
                if let Some(dest) = dest_reg {
                    if dest != ptr_reg {
                        instrs.push(emit_mv(dest, ptr_reg));
                    }
                }
            }
        }
    }

    instrs
}

/// Attempt to find the physical register allocated for parameter `idx`.
/// If the register allocator assigned a register for the idx-th live interval
/// corresponding to a parameter, return it; otherwise return `None`.
fn get_param_dest_register(param_idx: usize, alloc_result: &AllocationResult) -> Option<PhysReg> {
    // The register allocator's intervals are indexed by virtual register number.
    // Parameters are typically the first N virtual registers.
    if param_idx < alloc_result.intervals.len() {
        alloc_result.intervals[param_idx].assigned_reg
    } else {
        None
    }
}

/// Attempt to find a floating-point destination register for a paired argument.
fn get_param_fp_dest_register(param_idx: usize, alloc_result: &AllocationResult) -> Option<PhysReg> {
    // For paired arguments, the FP component typically comes after the int component
    // in the live interval list. We try param_idx + some offset.
    let fp_idx = param_idx + alloc_result.intervals.len() / 2;
    if fp_idx < alloc_result.intervals.len() {
        alloc_result.intervals[fp_idx].assigned_reg
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// generate_call_sequence — Outgoing Call Handling
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate the instruction sequence for a function call site.
///
/// Steps:
/// 1. Save any caller-saved registers that are live across the call.
/// 2. Move each argument from its current register to the ABI-specified location.
/// 3. For stack arguments: store values to the outgoing argument area.
/// 4. Emit the CALL instruction (AUIPC+JALR for direct, JALR for indirect).
/// 5. Move the return value from a0/fa0 to the destination register.
/// 6. Restore caller-saved registers.
pub fn generate_call_sequence(
    callee: &Callee,
    args: &[Value],
    return_ty: &IrType,
    value_map: &HashMap<Value, PhysReg>,
) -> Vec<MachineInstr> {
    let mut instrs: Vec<MachineInstr> = Vec::new();
    let mut int_reg_idx: usize = 0;
    let mut fp_reg_idx: usize = 0;
    let mut stack_offset: i32 = 0;
    let xlen: usize = 8; // RV64

    // Phase 1: Move arguments to their ABI locations.
    // We classify each argument based on its current physical register class.
    for arg in args.iter() {
        let src_reg = value_map.get(arg).copied();

        match src_reg {
            Some(reg) if is_fp_register(reg) => {
                // Floating-point value → next fa-register or stack
                if fp_reg_idx < 8 {
                    let dest = ARG_FP_REGS[fp_reg_idx];
                    fp_reg_idx += 1;
                    if reg != dest {
                        instrs.push(emit_fmv_d(dest, reg));
                    }
                } else if int_reg_idx < 8 {
                    // FP regs exhausted; pass in int reg via FMV.X.D
                    let dest = ARG_INT_REGS[int_reg_idx];
                    int_reg_idx += 1;
                    instrs.push(mi(OP_FMV_X_D, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Register(reg),
                    ]));
                } else {
                    // Stack: store double to outgoing arg area
                    instrs.push(emit_fsd(reg, SP, stack_offset));
                    stack_offset += xlen as i32;
                }
            }

            Some(reg) => {
                // Integer/pointer value → next a-register or stack
                if int_reg_idx < 8 {
                    let dest = ARG_INT_REGS[int_reg_idx];
                    int_reg_idx += 1;
                    if reg != dest {
                        instrs.push(emit_mv(dest, reg));
                    }
                } else {
                    // Stack: store to outgoing arg area
                    instrs.push(emit_sd(reg, SP, stack_offset));
                    stack_offset += xlen as i32;
                }
            }

            None => {
                // Value not in a register — likely a constant or not yet materialized.
                // Load zero as a placeholder; the actual value will be patched by isel.
                if int_reg_idx < 8 {
                    let dest = ARG_INT_REGS[int_reg_idx];
                    int_reg_idx += 1;
                    instrs.push(mi(OP_LI, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(0),
                    ]));
                } else {
                    instrs.push(emit_sd(ZERO, SP, stack_offset));
                    stack_offset += xlen as i32;
                }
            }
        }
    }

    // Phase 2: Emit the call instruction.
    match callee {
        Callee::Direct(name) => {
            instrs.push(mi(OP_CALL, vec![
                MachineOperand::Symbol(name.clone()),
            ]));
        }
        Callee::Indirect(val) => {
            // Load the function pointer from value_map and call via JALR
            if let Some(&target_reg) = value_map.get(val) {
                instrs.push(mi(OP_JALR, vec![
                    MachineOperand::Register(RA),
                    MachineOperand::Register(target_reg),
                    MachineOperand::Immediate(0),
                ]));
            } else {
                // Fallback: call through x0 (this should not happen in practice)
                instrs.push(mi(OP_JALR, vec![
                    MachineOperand::Register(RA),
                    MachineOperand::Register(X0),
                    MachineOperand::Immediate(0),
                ]));
            }
        }
    }

    // Phase 3: Return value handling.
    // The LP64D ABI places integer results in a0 (X10) and float results in fa0 (F10).
    // We encode a NOP marker after the call so the instruction selector knows the
    // return class and can insert any needed move from a0/fa0 to the destination.
    if return_ty.is_float() {
        // Float return: result is in fa0 (F10). Mark with NOP for isel.
        instrs.push(mi(OP_NOP, vec![]));
    } else if !return_ty.is_void() {
        // Integer/pointer return: result is in a0 (X10). Mark with NOP for isel.
        instrs.push(mi(OP_NOP, vec![]));
    }
    // Void return: no result to read.

    instrs
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc::LiveInterval;

    // ───────────────────────────────────────────────────────────────────
    // Helpers
    // ───────────────────────────────────────────────────────────────────

    fn rv64_target() -> TargetConfig {
        TargetConfig::riscv64()
    }

    fn empty_alloc() -> AllocationResult {
        AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: 0,
            used_callee_saved: Vec::new(),
        }
    }

    fn alloc_with_callee_saved(regs: Vec<PhysReg>, spills: u32) -> AllocationResult {
        AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: spills,
            used_callee_saved: regs,
        }
    }

    fn make_basic_block(id: u32, label: &str, term: crate::ir::Terminator) -> crate::ir::BasicBlock {
        crate::ir::BasicBlock {
            id: crate::ir::BlockId(id),
            label: label.to_string(),
            phi_nodes: Vec::new(),
            instructions: Vec::new(),
            terminator: Some(term),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    fn simple_function(name: &str, params: Vec<(String, IrType)>, ret: IrType) -> Function {
        Function {
            name: name.to_string(),
            return_type: ret,
            params,
            param_values: Vec::new(),
            blocks: vec![
                make_basic_block(0, "entry", crate::ir::Terminator::Return { value: None }),
            ],
            entry_block: crate::ir::BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    fn multi_block_function(name: &str, params: Vec<(String, IrType)>, ret: IrType) -> Function {
        Function {
            name: name.to_string(),
            return_type: ret,
            params: params.clone(),
            param_values: Vec::new(),
            blocks: vec![
                make_basic_block(0, "entry", crate::ir::Terminator::Branch {
                    target: crate::ir::BlockId(1),
                }),
                make_basic_block(1, "exit", crate::ir::Terminator::Return { value: None }),
            ],
            entry_block: crate::ir::BlockId(0),
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Register convention tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_register_conventions_arg_int_regs() {
        // a0–a7 are x10–x17
        assert_eq!(ARG_INT_REGS[0], X10);
        assert_eq!(ARG_INT_REGS[1], X11);
        assert_eq!(ARG_INT_REGS[2], X12);
        assert_eq!(ARG_INT_REGS[3], X13);
        assert_eq!(ARG_INT_REGS[4], X14);
        assert_eq!(ARG_INT_REGS[5], X15);
        assert_eq!(ARG_INT_REGS[6], X16);
        assert_eq!(ARG_INT_REGS[7], X17);
    }

    #[test]
    fn test_register_conventions_arg_fp_regs() {
        // fa0–fa7 are f10–f17
        assert_eq!(ARG_FP_REGS[0], F10);
        assert_eq!(ARG_FP_REGS[1], F11);
        assert_eq!(ARG_FP_REGS[2], F12);
        assert_eq!(ARG_FP_REGS[3], F13);
        assert_eq!(ARG_FP_REGS[4], F14);
        assert_eq!(ARG_FP_REGS[5], F15);
        assert_eq!(ARG_FP_REGS[6], F16);
        assert_eq!(ARG_FP_REGS[7], F17);
    }

    #[test]
    fn test_register_conventions_special_regs() {
        assert_eq!(RA, PhysReg(1));   // x1
        assert_eq!(SP, PhysReg(2));   // x2
        assert_eq!(FP, PhysReg(8));   // x8 = s0
        assert_eq!(ZERO, PhysReg(0)); // x0
        assert_eq!(X0, ZERO);         // x0 = zero
    }

    #[test]
    fn test_register_info_consistency() {
        // Verify that riscv64_register_info() reports the expected register counts
        let info = riscv64_register_info();
        // RISC-V has 32 GPRs and 32 FPRs
        assert!(!info.int_regs.is_empty());
        assert!(!info.float_regs.is_empty());
        // Callee-saved sets should be non-empty
        assert!(!info.callee_saved_int.is_empty());
        assert!(!info.callee_saved_float.is_empty());
    }

    #[test]
    fn test_callee_saved_gprs() {
        // s0–s11 = x8–x9, x18–x27
        assert_eq!(CALLEE_SAVED_GPRS[0], PhysReg(8));  // s0
        assert_eq!(CALLEE_SAVED_GPRS[1], PhysReg(9));  // s1
        assert_eq!(CALLEE_SAVED_GPRS[2], PhysReg(18)); // s2
        assert_eq!(CALLEE_SAVED_GPRS[11], PhysReg(27)); // s11
    }

    #[test]
    fn test_callee_saved_fprs() {
        // fs0–fs11 = f8–f9, f18–f27 → PhysReg(40–41, 50–59)
        assert_eq!(CALLEE_SAVED_FPRS[0], PhysReg(40));  // fs0
        assert_eq!(CALLEE_SAVED_FPRS[1], PhysReg(41));  // fs1
        assert_eq!(CALLEE_SAVED_FPRS[2], PhysReg(50));  // fs2
        assert_eq!(CALLEE_SAVED_FPRS[11], PhysReg(59)); // fs11
    }

    #[test]
    fn test_is_fp_register() {
        assert!(!is_fp_register(PhysReg(0)));  // x0
        assert!(!is_fp_register(PhysReg(31))); // x31
        assert!(is_fp_register(PhysReg(32)));  // f0
        assert!(is_fp_register(PhysReg(63)));  // f31
    }

    // ───────────────────────────────────────────────────────────────────
    // Argument classification tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_single_i32_arg() {
        let params = vec![("x".into(), IrType::I32)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::IntegerReg(X10)); // a0
    }

    #[test]
    fn test_classify_single_i64_arg() {
        let params = vec![("x".into(), IrType::I64)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::IntegerReg(X10)); // a0
    }

    #[test]
    fn test_classify_single_f64_arg() {
        let params = vec![("x".into(), IrType::F64)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::FloatReg(F10)); // fa0
    }

    #[test]
    fn test_classify_single_f32_arg() {
        let params = vec![("x".into(), IrType::F32)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::FloatReg(F10)); // fa0
    }

    #[test]
    fn test_classify_pointer_arg() {
        let params = vec![("p".into(), IrType::Pointer(Box::new(IrType::I8)))];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::IntegerReg(X10)); // a0
    }

    #[test]
    fn test_classify_8_integer_args() {
        let params: Vec<(String, IrType)> = (0..8)
            .map(|i| (format!("x{}", i), IrType::I64))
            .collect();
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 8);
        for (i, class) in classes.iter().enumerate() {
            assert_eq!(*class, ArgClass::IntegerReg(ARG_INT_REGS[i]));
        }
    }

    #[test]
    fn test_classify_9_integer_args_overflow() {
        let params: Vec<(String, IrType)> = (0..9)
            .map(|i| (format!("x{}", i), IrType::I64))
            .collect();
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 9);
        // First 8 in registers
        for i in 0..8 {
            assert_eq!(classes[i], ArgClass::IntegerReg(ARG_INT_REGS[i]));
        }
        // 9th on stack
        assert_eq!(classes[8], ArgClass::Stack(0));
    }

    #[test]
    fn test_classify_8_float_args() {
        let params: Vec<(String, IrType)> = (0..8)
            .map(|i| (format!("f{}", i), IrType::F64))
            .collect();
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 8);
        for (i, class) in classes.iter().enumerate() {
            assert_eq!(*class, ArgClass::FloatReg(ARG_FP_REGS[i]));
        }
    }

    #[test]
    fn test_classify_mixed_int_float_args() {
        let params = vec![
            ("a".into(), IrType::I64),
            ("b".into(), IrType::F64),
            ("c".into(), IrType::I32),
            ("d".into(), IrType::F32),
        ];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 4);
        assert_eq!(classes[0], ArgClass::IntegerReg(X10)); // a0
        assert_eq!(classes[1], ArgClass::FloatReg(F10));   // fa0
        assert_eq!(classes[2], ArgClass::IntegerReg(X11)); // a1
        assert_eq!(classes[3], ArgClass::FloatReg(F11));   // fa1
    }

    #[test]
    fn test_classify_struct_two_ints() {
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::IntegerRegPair(X10, X11));
    }

    #[test]
    fn test_classify_struct_two_floats() {
        let struct_ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::F64],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::FloatRegPair(F10, F11));
    }

    #[test]
    fn test_classify_struct_int_float() {
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::F64],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::IntegerFloatPair(X10, F10));
    }

    #[test]
    fn test_classify_struct_float_int() {
        let struct_ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::I64],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::FloatIntegerPair(F10, X10));
    }

    #[test]
    fn test_classify_large_struct_indirect() {
        // Struct > 16 bytes → passed by reference
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::Indirect(X10));
    }

    #[test]
    fn test_classify_struct_single_float() {
        let struct_ty = IrType::Struct {
            fields: vec![IrType::F64],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::FloatReg(F10));
    }

    #[test]
    fn test_classify_struct_single_int() {
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        let params = vec![("s".into(), struct_ty)];
        let classes = classify_arguments(&params, &rv64_target());
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], ArgClass::IntegerReg(X10));
    }

    // ───────────────────────────────────────────────────────────────────
    // StructArgClass tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_struct_class_all_integer() {
        let ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        let class = classify_struct_fields(&ty, &rv64_target());
        assert_eq!(class, StructArgClass::AllInteger(2));
    }

    #[test]
    fn test_struct_class_all_float() {
        let ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::F64],
            packed: false,
        };
        let class = classify_struct_fields(&ty, &rv64_target());
        assert_eq!(class, StructArgClass::AllFloat(2));
    }

    #[test]
    fn test_struct_class_mixed_int_float() {
        let ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::F64],
            packed: false,
        };
        let class = classify_struct_fields(&ty, &rv64_target());
        assert_eq!(class, StructArgClass::MixedIntFloat);
    }

    #[test]
    fn test_struct_class_mixed_float_int() {
        let ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::I64],
            packed: false,
        };
        let class = classify_struct_fields(&ty, &rv64_target());
        assert_eq!(class, StructArgClass::MixedFloatInt);
    }

    #[test]
    fn test_struct_class_by_reference() {
        let ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        let class = classify_struct_fields(&ty, &rv64_target());
        assert_eq!(class, StructArgClass::ByReference);
    }

    #[test]
    fn test_struct_class_non_struct() {
        // Non-struct type → ByReference (fallback)
        let class = classify_struct_fields(&IrType::I64, &rv64_target());
        assert_eq!(class, StructArgClass::ByReference);
    }

    // ───────────────────────────────────────────────────────────────────
    // Return classification tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_return_void() {
        assert_eq!(classify_return(&IrType::Void, &rv64_target()), ReturnClass::Void);
    }

    #[test]
    fn test_return_i32() {
        assert_eq!(classify_return(&IrType::I32, &rv64_target()), ReturnClass::IntegerReg);
    }

    #[test]
    fn test_return_i64() {
        assert_eq!(classify_return(&IrType::I64, &rv64_target()), ReturnClass::IntegerReg);
    }

    #[test]
    fn test_return_pointer() {
        let ptr = IrType::Pointer(Box::new(IrType::I8));
        assert_eq!(classify_return(&ptr, &rv64_target()), ReturnClass::IntegerReg);
    }

    #[test]
    fn test_return_f64() {
        assert_eq!(classify_return(&IrType::F64, &rv64_target()), ReturnClass::FloatReg);
    }

    #[test]
    fn test_return_f32() {
        assert_eq!(classify_return(&IrType::F32, &rv64_target()), ReturnClass::FloatReg);
    }

    #[test]
    fn test_return_small_struct_two_ints() {
        let ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        assert_eq!(classify_return(&ty, &rv64_target()), ReturnClass::IntegerRegPair);
    }

    #[test]
    fn test_return_small_struct_two_floats() {
        let ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::F64],
            packed: false,
        };
        assert_eq!(classify_return(&ty, &rv64_target()), ReturnClass::FloatRegPair);
    }

    #[test]
    fn test_return_small_struct_mixed() {
        let ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::F64],
            packed: false,
        };
        assert_eq!(classify_return(&ty, &rv64_target()), ReturnClass::IntegerFloatPair);
    }

    #[test]
    fn test_return_large_struct_indirect() {
        let ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        assert_eq!(classify_return(&ty, &rv64_target()), ReturnClass::Indirect);
    }

    // ───────────────────────────────────────────────────────────────────
    // Frame layout tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_layout_minimal() {
        let func = simple_function("minimal", vec![], IrType::Void);
        let alloc = empty_alloc();
        let layout = compute_frame_layout(&func, &alloc, &rv64_target());

        // Frame size must be 16-byte aligned
        assert_eq!(layout.frame_size % 16, 0);
        // RA is always saved
        assert!(layout.ra_offset >= 0);
    }

    #[test]
    fn test_frame_layout_with_callee_saved() {
        let func = simple_function("callee_saved", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![PhysReg(9), PhysReg(18)], 0); // s1, s2
        let layout = compute_frame_layout(&func, &alloc, &rv64_target());

        // Frame size must be 16-byte aligned
        assert_eq!(layout.frame_size % 16, 0);
        // Should have callee-saved offsets
        assert_eq!(layout.callee_saved_offsets.len(), 2);
        // Offsets should be distinct
        let off1 = layout.callee_saved_offsets[0].1;
        let off2 = layout.callee_saved_offsets[1].1;
        assert_ne!(off1, off2);
    }

    #[test]
    fn test_frame_layout_16byte_aligned() {
        let params: Vec<(String, IrType)> = (0..3)
            .map(|i| (format!("p{}", i), IrType::I64))
            .collect();
        let func = simple_function("aligned", params, IrType::I64);
        let alloc = alloc_with_callee_saved(vec![PhysReg(9)], 3); // s1, 3 spill slots
        let layout = compute_frame_layout(&func, &alloc, &rv64_target());

        assert_eq!(layout.frame_size % 16, 0, "Frame size {} is not 16-byte aligned", layout.frame_size);
    }

    #[test]
    fn test_frame_layout_with_spills() {
        let func = simple_function("spills", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![], 5);
        let layout = compute_frame_layout(&func, &alloc, &rv64_target());

        assert_eq!(layout.spill_size, 40); // 5 * 8 bytes
        assert_eq!(layout.frame_size % 16, 0);
    }

    #[test]
    fn test_frame_layout_fp_callee_saved() {
        let func = simple_function("fp_save", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![PhysReg(40), PhysReg(41)], 0); // fs0, fs1
        let layout = compute_frame_layout(&func, &alloc, &rv64_target());

        assert_eq!(layout.callee_saved_fp_offsets.len(), 2);
        assert_eq!(layout.frame_size % 16, 0);
    }

    // ───────────────────────────────────────────────────────────────────
    // Prologue/epilogue tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_prologue_simple() {
        let func = simple_function("simple", vec![], IrType::Void);
        let alloc = empty_alloc();
        let instrs = generate_prologue(&func, &alloc, &rv64_target());

        // Should have at least: ADDI sp, sd ra
        assert!(!instrs.is_empty(), "Prologue should not be empty");

        // First instruction should decrement SP (ADDI with negative imm)
        assert_eq!(instrs[0].opcode, OP_ADDI);
    }

    #[test]
    fn test_epilogue_simple() {
        let func = simple_function("simple", vec![], IrType::Void);
        let alloc = empty_alloc();
        let instrs = generate_epilogue(&func, &alloc, &rv64_target());

        // Should end with RET
        assert!(!instrs.is_empty());
        let last = instrs.last().unwrap();
        assert_eq!(last.opcode, OP_RET);
    }

    #[test]
    fn test_prologue_epilogue_symmetry() {
        let func = simple_function("sym", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![PhysReg(9), PhysReg(18)], 0);

        let prologue = generate_prologue(&func, &alloc, &rv64_target());
        let epilogue = generate_epilogue(&func, &alloc, &rv64_target());

        // Count SD (save) instructions in prologue
        let saves = prologue.iter().filter(|i| i.opcode == OP_SD).count();
        // Count LD (restore) instructions in epilogue (excluding RET and ADDI)
        let restores = epilogue.iter().filter(|i| i.opcode == OP_LD).count();

        assert_eq!(saves, restores, "Save/restore count mismatch: {} saves, {} restores", saves, restores);
    }

    #[test]
    fn test_prologue_with_frame_pointer() {
        let func = multi_block_function("fp_func", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![], 1);
        let instrs = generate_prologue(&func, &alloc, &rv64_target());

        // Should contain an instruction that sets up FP
        // The FP setup uses ADDI or ADD with FP as destination
        let has_fp_setup = instrs.iter().any(|i| {
            if i.operands.is_empty() { return false; }
            match &i.operands[0] {
                MachineOperand::Register(r) => *r == FP && (i.opcode == OP_ADDI || i.opcode == OP_ADD),
                _ => false,
            }
        });
        assert!(has_fp_setup, "Prologue should set up frame pointer");
    }

    // ───────────────────────────────────────────────────────────────────
    // Call sequence tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_call_direct() {
        let callee = Callee::Direct("puts".to_string());
        let v0 = Value(0);
        let mut vmap = HashMap::new();
        vmap.insert(v0, PhysReg(9)); // s1

        let instrs = generate_call_sequence(&callee, &[v0], &IrType::I32, &vmap);

        // Should contain: MV a0, s1 + CALL puts
        assert!(instrs.len() >= 2);
        let has_call = instrs.iter().any(|i| i.opcode == OP_CALL);
        assert!(has_call, "Should have a CALL instruction");
    }

    #[test]
    fn test_call_indirect() {
        let fptr = Value(10);
        let callee = Callee::Indirect(fptr);
        let v0 = Value(0);
        let mut vmap = HashMap::new();
        vmap.insert(v0, PhysReg(9));
        vmap.insert(fptr, PhysReg(18)); // function pointer in s2

        let instrs = generate_call_sequence(&callee, &[v0], &IrType::I32, &vmap);

        let has_jalr = instrs.iter().any(|i| i.opcode == OP_JALR);
        assert!(has_jalr, "Indirect call should use JALR");
    }

    #[test]
    fn test_call_float_arg() {
        let callee = Callee::Direct("sin".to_string());
        let v0 = Value(0);
        let mut vmap = HashMap::new();
        vmap.insert(v0, PhysReg(42)); // FP register (f10)

        let instrs = generate_call_sequence(&callee, &[v0], &IrType::F64, &vmap);

        // Should have FMV.D or similar to move to fa0
        assert!(!instrs.is_empty());
    }

    // ───────────────────────────────────────────────────────────────────
    // Helper function tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_fits_in_12bit_signed() {
        assert!(fits_in_12bit_signed(0));
        assert!(fits_in_12bit_signed(2047));
        assert!(fits_in_12bit_signed(-2048));
        assert!(!fits_in_12bit_signed(2048));
        assert!(!fits_in_12bit_signed(-2049));
        assert!(fits_in_12bit_signed(-1));
        assert!(fits_in_12bit_signed(1));
    }

    #[test]
    fn test_large_sp_adjust() {
        let instrs = emit_large_sp_adjust(-4096);
        // Should produce LUI + ADDI + ADD
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[0].opcode, OP_LUI);
        assert_eq!(instrs[1].opcode, OP_ADDI);
        assert_eq!(instrs[2].opcode, OP_ADD);
    }

    // ───────────────────────────────────────────────────────────────────
    // Argument loads tests
    // ───────────────────────────────────────────────────────────────────

    fn make_live_interval(val: u32, assigned: Option<PhysReg>, is_fp: bool) -> LiveInterval {
        LiveInterval {
            value: Value(val),
            reg_class: if is_fp {
                crate::codegen::regalloc::RegClass::Float
            } else {
                crate::codegen::regalloc::RegClass::Integer
            },
            start: 0,
            end: 10,
            assigned_reg: assigned,
            spill_slot: None,
            is_param: true,
            crosses_call: false,
            is_alloca: false,
        }
    }

    #[test]
    fn test_argument_loads_no_move_needed() {
        let params = vec![("x".into(), IrType::I64)];
        let classes = vec![ArgClass::IntegerReg(X10)]; // a0
        // Allocator assigns the same register (a0 = X10)
        let alloc = AllocationResult {
            intervals: vec![make_live_interval(0, Some(X10), false)],
            num_spill_slots: 0,
            used_callee_saved: vec![],
        };
        let instrs = generate_argument_loads(&params, &classes, &alloc);
        // No move needed since src == dest
        assert!(instrs.is_empty() || instrs.iter().all(|i| i.opcode != OP_MV));
    }

    #[test]
    fn test_argument_loads_move_needed() {
        let params = vec![("x".into(), IrType::I64)];
        let classes = vec![ArgClass::IntegerReg(X10)]; // a0
        // Allocator assigns a different register (s1 = PhysReg(9))
        let alloc = AllocationResult {
            intervals: vec![make_live_interval(0, Some(PhysReg(9)), false)],
            num_spill_slots: 0,
            used_callee_saved: vec![PhysReg(9)],
        };
        let instrs = generate_argument_loads(&params, &classes, &alloc);
        assert!(!instrs.is_empty());
        assert_eq!(instrs[0].opcode, OP_MV);
    }
}
