//! AAPCS64 (ARM Architecture Procedure Call Standard for 64-bit) ABI Implementation.
//!
//! This module implements the complete calling convention for the AArch64 (ARMv8-A)
//! architecture following the AAPCS64 specification. It covers:
//!
//! - **Argument classification**: x0-x7 for integer/pointer arguments, v0-v7 for
//!   floating-point/SIMD arguments, with independent counters for GPR and FP
//!   register allocation.
//! - **Return value classification**: x0 for integer returns, v0 for float returns,
//!   x0+x1 pair for 128-bit composites, x8 as indirect result location for
//!   large structs (>16 bytes).
//! - **Stack frame layout**: FP+LR pair saving with STP/LDP, callee-saved register
//!   pairs (x19-x28, d8-d15), 16-byte SP alignment enforcement.
//! - **Prologue/epilogue generation**: STP/LDP pair instructions for efficient
//!   callee-saved register save/restore with proper frame pointer setup.
//! - **Struct passing**: HFA (Homogeneous Floating-point Aggregate) detection for
//!   up to 4 same-typed float members, non-HFA ≤16 bytes in register pairs,
//!   indirect passing for aggregates >16 bytes.
//! - **Call sequence generation**: BL for direct calls, BLR for indirect calls.
//!
//! ## Key AAPCS64 Rules
//!
//! | Rule                                | Details                                        |
//! |-------------------------------------|------------------------------------------------|
//! | Integer argument registers          | x0-x7 (NGRN counter, 0-7)                     |
//! | FP argument registers               | v0-v7 / d0-d7 / s0-s7 (NSRN counter, 0-7)    |
//! | Return registers                    | x0 (integer), v0 (float), x8 (indirect)       |
//! | Callee-saved GPRs                   | x19-x28 (10 registers)                         |
//! | Callee-saved FP (lower 64 bits)     | d8-d15                                         |
//! | Frame pointer                       | x29 (FP)                                       |
//! | Link register                       | x30 (LR)                                       |
//! | Stack alignment                     | SP must be 16-byte aligned AT ALL TIMES        |
//! | Red zone                            | NONE on AArch64                                |
//! | Intra-procedure scratch             | x16 (IP0), x17 (IP1)                           |
//!
//! ## Zero External Dependencies
//!
//! Only imports from `std` and internal crate modules. No external crates.

use std::collections::HashMap;

use crate::codegen::regalloc::{AllocationResult, PhysReg, RegClass};
use crate::codegen::{MachineInstr, MachineOperand};
use crate::driver::target::TargetConfig;
use crate::ir::{Callee, Function, IrType, Value};

// ---------------------------------------------------------------------------
// RegClass Usage — ABI-level register class determination
// ---------------------------------------------------------------------------

/// Returns the register class for a given IR type per AAPCS64 rules.
///
/// Floating-point types use `RegClass::Float` (SIMD/FP register file),
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
// AArch64 Machine Instruction Opcodes (prologue / epilogue / ABI sequences)
// ---------------------------------------------------------------------------
// These opcodes are used by the ABI module for generating prologue, epilogue,
// argument loading, and call sequences. Values are chosen to avoid collision
// with encoding-layer opcodes (which start at lower ranges). The isel and
// encoding modules will reference these same constant values.

/// `STP Xt1, Xt2, [Xn, #imm]!` — Store pair with pre-indexed write-back.
/// Used in prologue for allocating frame and saving FP+LR simultaneously.
const OP_STP_PRE: u32 = 0xA900;

/// `STP Xt1, Xt2, [Xn, #imm]` — Store pair with signed offset.
/// Used for saving callee-saved register pairs at known offsets.
const OP_STP_OFFSET: u32 = 0xA901;

/// `LDP Xt1, Xt2, [Xn], #imm` — Load pair with post-indexed write-back.
/// Used in epilogue for restoring FP+LR and deallocating frame.
const OP_LDP_POST: u32 = 0xA902;

/// `LDP Xt1, Xt2, [Xn, #imm]` — Load pair with signed offset.
/// Used for restoring callee-saved register pairs at known offsets.
const OP_LDP_OFFSET: u32 = 0xA903;

/// `STR Xt, [Xn, #imm]` — Store register (single, unsigned offset).
/// Used for saving an unpaired callee-saved register.
const OP_STR_IMM: u32 = 0xA910;

/// `LDR Xt, [Xn, #imm]` — Load register (single, unsigned offset).
/// Used for restoring an unpaired callee-saved register.
const OP_LDR_IMM: u32 = 0xA911;

/// `MOV Xd, Xn` (alias for ORR Xd, XZR, Xn) — Register-to-register move.
const OP_MOV_RR: u32 = 0xA920;

/// `FMOV Dd, Dn` — FP register-to-register move.
const OP_FMOV_RR: u32 = 0xA921;

/// `SUB Xd, Xn, #imm` — Subtract immediate from register.
const OP_SUB_RI: u32 = 0xA930;

/// `ADD Xd, Xn, #imm` — Add immediate to register.
const OP_ADD_RI: u32 = 0xA931;

/// `SUB Xd, Xn, Xm` — Subtract register from register.
const OP_SUB_RR: u32 = 0xA932;

/// `MOVZ Xd, #imm16, LSL #shift` — Move wide with zero.
const OP_MOVZ: u32 = 0xA940;

/// `MOVK Xd, #imm16, LSL #shift` — Move wide with keep.
const OP_MOVK: u32 = 0xA941;

/// `BL <symbol>` — Branch with link (direct call).
const OP_BL: u32 = 0xA950;

/// `BLR Xn` — Branch with link to register (indirect call).
const OP_BLR: u32 = 0xA951;

/// `RET` — Return from subroutine (Branch to LR / x30).
const OP_RET: u32 = 0xA960;

/// `STP (FP pair) Dt1, Dt2, [Xn, #imm]` — Store FP register pair.
const OP_STP_FP_OFFSET: u32 = 0xA970;

/// `LDP (FP pair) Dt1, Dt2, [Xn, #imm]` — Load FP register pair.
const OP_LDP_FP_OFFSET: u32 = 0xA971;

/// `STR (FP single) Dt, [Xn, #imm]` — Store single FP register.
const OP_STR_FP_IMM: u32 = 0xA972;

/// `LDR (FP single) Dt, [Xn, #imm]` — Load single FP register.
const OP_LDR_FP_IMM: u32 = 0xA973;

// ---------------------------------------------------------------------------
// AArch64 Physical Register Constants
// ---------------------------------------------------------------------------
// Register numbering: GPRs x0-x30 map to PhysReg(0)-PhysReg(30).
// SP is PhysReg(31). FP/SIMD registers v0-v31 map to PhysReg(32)-PhysReg(63).

/// x0 — First integer argument register, integer return register.
pub const X0: PhysReg = PhysReg(0);
/// x1 — Second integer argument register, second return register for pairs.
pub const X1: PhysReg = PhysReg(1);
/// x2 — Third integer argument register.
pub const X2: PhysReg = PhysReg(2);
/// x3 — Fourth integer argument register.
pub const X3: PhysReg = PhysReg(3);
/// x4 — Fifth integer argument register.
pub const X4: PhysReg = PhysReg(4);
/// x5 — Sixth integer argument register.
pub const X5: PhysReg = PhysReg(5);
/// x6 — Seventh integer argument register.
pub const X6: PhysReg = PhysReg(6);
/// x7 — Eighth integer argument register.
pub const X7: PhysReg = PhysReg(7);
/// x8 — Indirect result location register (for returning large structs).
pub const X8: PhysReg = PhysReg(8);
/// x9 — Caller-saved temporary register.
pub const X9: PhysReg = PhysReg(9);
/// x10 — Caller-saved temporary register.
pub const X10: PhysReg = PhysReg(10);
/// x11 — Caller-saved temporary register.
pub const X11: PhysReg = PhysReg(11);
/// x12 — Caller-saved temporary register.
pub const X12: PhysReg = PhysReg(12);
/// x13 — Caller-saved temporary register.
pub const X13: PhysReg = PhysReg(13);
/// x14 — Caller-saved temporary register.
pub const X14: PhysReg = PhysReg(14);
/// x15 — Caller-saved temporary register.
pub const X15: PhysReg = PhysReg(15);
/// x16 — IP0 (intra-procedure-call scratch register 0, linker veneer).
pub const X16: PhysReg = PhysReg(16);
/// x17 — IP1 (intra-procedure-call scratch register 1, linker veneer).
pub const X17: PhysReg = PhysReg(17);
/// x18 — Platform register (caller-saved on Linux).
pub const X18: PhysReg = PhysReg(18);
/// x19 — Callee-saved register.
pub const X19: PhysReg = PhysReg(19);
/// x20 — Callee-saved register.
pub const X20: PhysReg = PhysReg(20);
/// x21 — Callee-saved register.
pub const X21: PhysReg = PhysReg(21);
/// x22 — Callee-saved register.
pub const X22: PhysReg = PhysReg(22);
/// x23 — Callee-saved register.
pub const X23: PhysReg = PhysReg(23);
/// x24 — Callee-saved register.
pub const X24: PhysReg = PhysReg(24);
/// x25 — Callee-saved register.
pub const X25: PhysReg = PhysReg(25);
/// x26 — Callee-saved register.
pub const X26: PhysReg = PhysReg(26);
/// x27 — Callee-saved register.
pub const X27: PhysReg = PhysReg(27);
/// x28 — Callee-saved register.
pub const X28: PhysReg = PhysReg(28);
/// x29 — Frame pointer (FP), callee-saved.
pub const FP: PhysReg = PhysReg(29);
/// x30 — Link register (LR), holds return address after BL.
pub const LR: PhysReg = PhysReg(30);
/// SP — Stack pointer (encoded as x31 in certain contexts).
pub const SP: PhysReg = PhysReg(31);

// FP/SIMD registers — offset by 32 in the PhysReg numbering space.

/// v0 / d0 / s0 — First FP argument register, FP return register.
pub const V0: PhysReg = PhysReg(32);
/// v1 / d1 / s1 — Second FP argument register.
pub const V1: PhysReg = PhysReg(33);
/// v2 / d2 / s2 — Third FP argument register.
pub const V2: PhysReg = PhysReg(34);
/// v3 / d3 / s3 — Fourth FP argument register.
pub const V3: PhysReg = PhysReg(35);
/// v4 / d4 / s4 — Fifth FP argument register.
pub const V4: PhysReg = PhysReg(36);
/// v5 / d5 / s5 — Sixth FP argument register.
pub const V5: PhysReg = PhysReg(37);
/// v6 / d6 / s6 — Seventh FP argument register.
pub const V6: PhysReg = PhysReg(38);
/// v7 / d7 / s7 — Eighth FP argument register.
pub const V7: PhysReg = PhysReg(39);
/// v8 / d8 — Callee-saved FP register (lower 64 bits preserved).
pub const V8: PhysReg = PhysReg(40);
/// v9 / d9 — Callee-saved FP register.
pub const V9: PhysReg = PhysReg(41);
/// v10 / d10 — Callee-saved FP register.
pub const V10: PhysReg = PhysReg(42);
/// v11 / d11 — Callee-saved FP register.
pub const V11: PhysReg = PhysReg(43);
/// v12 / d12 — Callee-saved FP register.
pub const V12: PhysReg = PhysReg(44);
/// v13 / d13 — Callee-saved FP register.
pub const V13: PhysReg = PhysReg(45);
/// v14 / d14 — Callee-saved FP register.
pub const V14: PhysReg = PhysReg(46);
/// v15 / d15 — Callee-saved FP register.
pub const V15: PhysReg = PhysReg(47);
/// v16 — Caller-saved FP register.
pub const V16: PhysReg = PhysReg(48);
/// v17 — Caller-saved FP register.
pub const V17: PhysReg = PhysReg(49);
/// v18 — Caller-saved FP register.
pub const V18: PhysReg = PhysReg(50);
/// v19 — Caller-saved FP register.
pub const V19: PhysReg = PhysReg(51);
/// v20 — Caller-saved FP register.
pub const V20: PhysReg = PhysReg(52);
/// v21 — Caller-saved FP register.
pub const V21: PhysReg = PhysReg(53);
/// v22 — Caller-saved FP register.
pub const V22: PhysReg = PhysReg(54);
/// v23 — Caller-saved FP register.
pub const V23: PhysReg = PhysReg(55);
/// v24 — Caller-saved FP register.
pub const V24: PhysReg = PhysReg(56);
/// v25 — Caller-saved FP register.
pub const V25: PhysReg = PhysReg(57);
/// v26 — Caller-saved FP register.
pub const V26: PhysReg = PhysReg(58);
/// v27 — Caller-saved FP register.
pub const V27: PhysReg = PhysReg(59);
/// v28 — Caller-saved FP register.
pub const V28: PhysReg = PhysReg(60);
/// v29 — Caller-saved FP register.
pub const V29: PhysReg = PhysReg(61);
/// v30 — Caller-saved FP register.
pub const V30: PhysReg = PhysReg(62);
/// v31 — Caller-saved FP register.
pub const V31: PhysReg = PhysReg(63);

// ---------------------------------------------------------------------------
// Register Classification Arrays
// ---------------------------------------------------------------------------

/// Integer argument registers in AAPCS64 calling convention order.
/// NGRN (Next General-purpose Register Number) allocates from these sequentially.
pub const INT_ARG_REGS: [PhysReg; 8] = [X0, X1, X2, X3, X4, X5, X6, X7];

/// FP/SIMD argument registers in AAPCS64 calling convention order.
/// NSRN (Next SIMD and Floating-point Register Number) allocates from these.
pub const FLOAT_ARG_REGS: [PhysReg; 8] = [V0, V1, V2, V3, V4, V5, V6, V7];

/// Callee-saved GPRs: x19-x28 (10 registers).
/// These must be preserved by the called function.
pub const CALLEE_SAVED_GPRS: [PhysReg; 10] = [
    X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
];

/// Callee-saved FP registers: d8-d15 (lower 64 bits of v8-v15).
/// Only the lower 64 bits (Dn view) are callee-saved per AAPCS64.
pub const CALLEE_SAVED_FPRS: [PhysReg; 8] = [V8, V9, V10, V11, V12, V13, V14, V15];

// ---------------------------------------------------------------------------
// Register Helper Functions
// ---------------------------------------------------------------------------

/// Returns `true` if `reg` is an FP/SIMD register (v0-v31).
///
/// FP/SIMD registers occupy PhysReg indices 32-63 in the AArch64 numbering.
#[inline]
pub fn is_fp_reg(reg: PhysReg) -> bool {
    reg.0 >= 32 && reg.0 <= 63
}

/// Returns `true` if `reg` is a GPR (x0-x30, SP).
#[inline]
pub fn is_gpr(reg: PhysReg) -> bool {
    reg.0 <= 31
}

/// Returns `true` if `reg` is a callee-saved GPR (x19-x28).
#[inline]
pub fn is_callee_saved_gpr(reg: PhysReg) -> bool {
    reg.0 >= 19 && reg.0 <= 28
}

/// Returns `true` if `reg` is a callee-saved FP register (v8-v15 / d8-d15).
#[inline]
pub fn is_callee_saved_fpr(reg: PhysReg) -> bool {
    reg.0 >= 40 && reg.0 <= 47
}

// ---------------------------------------------------------------------------
// ArgClass — Argument Classification
// ---------------------------------------------------------------------------

/// Classification of how a function argument is passed per AAPCS64.
///
/// The AAPCS64 uses independent counters for GPR (NGRN) and FP (NSRN)
/// register allocation, meaning integer and floating-point arguments
/// do not compete for the same register slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgClass {
    /// Passed in a general-purpose register (x0-x7).
    IntegerReg(PhysReg),
    /// Passed in a SIMD/FP register (v0-v7 / d0-d7 / s0-s7).
    FloatReg(PhysReg),
    /// Passed on the stack at a given byte offset from the caller's SP.
    Stack(i32),
    /// Passed in two consecutive integer registers (for ≤16-byte composites).
    IntegerRegPair(PhysReg, PhysReg),
    /// Passed by reference: caller copies to memory, pointer in integer register.
    /// Used for composite types >16 bytes.
    Indirect(PhysReg),
}

// ---------------------------------------------------------------------------
// ReturnClass — Return Value Classification
// ---------------------------------------------------------------------------

/// Classification of how a function return value is passed per AAPCS64.
#[derive(Debug, Clone)]
pub enum ReturnClass {
    /// Returned in x0 (integer/pointer ≤ 64 bits).
    IntegerReg,
    /// Returned in v0 (float → s0, double → d0).
    FloatReg,
    /// Returned in x0+x1 pair (composite 9-16 bytes, non-HFA).
    IntegerRegPair,
    /// Returned as HFA in consecutive v-registers (1-4 registers).
    FloatRegMultiple(u8),
    /// Returned via hidden pointer in x8 (composite > 16 bytes).
    /// Caller allocates space and passes address in x8.
    Indirect,
    /// Void return — no return value.
    Void,
}

// ---------------------------------------------------------------------------
// HfaClass — Homogeneous Floating-point Aggregate Classification
// ---------------------------------------------------------------------------

/// Describes a Homogeneous Floating-point Aggregate (HFA).
///
/// An HFA is a struct with 1 to 4 members of the same floating-point type.
/// HFAs receive special treatment in AAPCS64: each member is passed in a
/// consecutive v-register rather than in GPRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfaClass {
    /// 1-4 single-precision (float / f32) members.
    SingleFloat(u8),
    /// 1-4 double-precision (double / f64) members.
    DoubleFloat(u8),
}

// ---------------------------------------------------------------------------
// Aarch64FrameLayout — Stack Frame Layout Descriptor
// ---------------------------------------------------------------------------

/// Describes the complete stack frame layout for an AArch64 function.
///
/// The frame is organized from high to low addresses as:
/// ```text
/// [Higher addresses]
/// +----------------------------+
/// | Incoming stack arguments   | (from caller, above our frame)
/// +----------------------------+ ← Old SP (16-byte aligned)
/// | FP (x29) + LR (x30) save  | 16 bytes (STP pair)
/// +----------------------------+ ← FP points here
/// | Callee-saved GPR pairs     | 16 bytes each (STP pairs)
/// +----------------------------+
/// | Callee-saved FPR pairs     | 16 bytes each (STP pairs)
/// +----------------------------+
/// | Spill slots                | (from register allocator)
/// +----------------------------+
/// | Local variables            |
/// +----------------------------+
/// | Outgoing argument area     | (for calls with >8 args)
/// +----------------------------+ ← New SP (16-byte aligned)
/// [Lower addresses]
/// ```
pub struct Aarch64FrameLayout {
    /// Total frame size in bytes (MUST be 16-byte aligned).
    pub frame_size: u32,
    /// Offset from SP where FP (x29) is saved.
    pub fp_offset: i32,
    /// Offset from SP where LR (x30) is saved.
    pub lr_offset: i32,
    /// Callee-saved GPR pairs with their offsets from SP.
    /// Each entry is ((reg1, reg2), offset).
    pub callee_saved_gpr_pairs: Vec<((PhysReg, PhysReg), i32)>,
    /// Callee-saved FPR pairs with their offsets from SP.
    pub callee_saved_fpr_pairs: Vec<((PhysReg, PhysReg), i32)>,
    /// Unpaired callee-saved registers (if odd count) with their offsets.
    pub callee_saved_singles: Vec<(PhysReg, i32)>,
    /// Offset from FP or SP where local variables begin.
    pub locals_offset: i32,
    /// Total space for local variables in bytes.
    pub locals_size: u32,
    /// Total space for register allocator spill slots in bytes.
    pub spill_size: u32,
    /// Size of outgoing argument area for calls with >8 args.
    pub arg_area_size: u32,
    /// Whether the frame pointer (x29) is used.
    pub uses_frame_pointer: bool,
}

// ---------------------------------------------------------------------------
// HFA Detection
// ---------------------------------------------------------------------------

/// Determines if a struct type qualifies as a Homogeneous Floating-point
/// Aggregate (HFA) per AAPCS64 rules.
///
/// An HFA is a struct with 1 to 4 members, all of the same floating-point
/// type (either all float/f32 or all double/f64). Nested structs that are
/// themselves HFAs with the same base type also qualify.
///
/// # Returns
///
/// `Some(HfaClass)` if the type is an HFA, `None` otherwise.
fn classify_hfa(ty: &IrType, target: &TargetConfig) -> Option<HfaClass> {
    match ty {
        IrType::Struct { fields, .. } => {
            if fields.is_empty() || fields.len() > 4 {
                return None;
            }

            // Flatten the struct fields to find the base float type.
            let mut float_members: Vec<&IrType> = Vec::new();
            for field in fields {
                flatten_hfa_fields(field, &mut float_members);
            }

            // Must have 1-4 float members total after flattening.
            if float_members.is_empty() || float_members.len() > 4 {
                return None;
            }

            // All members must be the same float type.
            let base = float_members[0];
            if !float_members.iter().all(|f| *f == base) {
                return None;
            }

            // Total size check: HFA size must not exceed 16 bytes.
            let total_size = ty.size(target);
            if total_size > 16 {
                return None;
            }

            let count = float_members.len() as u8;
            match base {
                IrType::F32 => Some(HfaClass::SingleFloat(count)),
                IrType::F64 => Some(HfaClass::DoubleFloat(count)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Recursively flattens struct fields to extract the leaf float types
/// for HFA classification. Non-float leaf types cause the accumulator
/// to receive non-float entries, which will fail the HFA check.
fn flatten_hfa_fields<'a>(ty: &'a IrType, out: &mut Vec<&'a IrType>) {
    match ty {
        IrType::F32 | IrType::F64 => {
            out.push(ty);
        }
        IrType::Struct { fields, .. } => {
            for field in fields {
                flatten_hfa_fields(field, out);
            }
        }
        // Any non-float, non-struct field means the containing struct is NOT an HFA.
        _ => {
            out.push(ty);
        }
    }
}

// ---------------------------------------------------------------------------
// classify_arguments — AAPCS64 Argument Classification
// ---------------------------------------------------------------------------

/// Classifies function parameters according to AAPCS64 argument passing rules.
///
/// The AAPCS64 uses two independent counters:
/// - **NGRN** (Next General-purpose Register Number): 0-7, for integer/pointer args
/// - **NSRN** (Next SIMD/Floating-point Register Number): 0-7, for float/double args
///
/// These counters are independent, so integer arguments do not consume FP
/// register slots and vice versa.
///
/// # Arguments
///
/// * `params` — Parameter names and their IR types, in declaration order.
/// * `target` — Target configuration for type sizing.
///
/// # Returns
///
/// A vector of `ArgClass` values, one per parameter, describing where each
/// argument is passed (register, stack, or by reference).
pub fn classify_arguments(params: &[(String, IrType)], target: &TargetConfig) -> Vec<ArgClass> {
    let mut result = Vec::with_capacity(params.len());

    // NGRN: Next General-purpose Register Number (0-7)
    let mut ngrn: usize = 0;
    // NSRN: Next SIMD and Floating-point Register Number (0-7)
    let mut nsrn: usize = 0;
    // NSAA: Next Stacked Argument Address (byte offset from SP for stack args)
    let mut nsaa: i32 = 0;

    for (_name, ty) in params {
        let class = classify_single_argument(ty, target, &mut ngrn, &mut nsrn, &mut nsaa);
        result.push(class);
    }

    result
}

/// Classifies a single argument according to AAPCS64 rules, advancing the
/// appropriate register or stack counters.
fn classify_single_argument(
    ty: &IrType,
    target: &TargetConfig,
    ngrn: &mut usize,
    nsrn: &mut usize,
    nsaa: &mut i32,
) -> ArgClass {
    // Rule 1: Fundamental floating-point types → v-registers (NSRN)
    if ty.is_float() {
        if *nsrn < 8 {
            let reg = FLOAT_ARG_REGS[*nsrn];
            *nsrn += 1;
            return ArgClass::FloatReg(reg);
        }
        // Overflow to stack
        let offset = align_stack_arg(*nsaa, 8);
        *nsaa = offset + 8;
        return ArgClass::Stack(offset);
    }

    // Rule 2: Integer/pointer types ≤ 64 bits → x-registers (NGRN)
    if ty.is_integer() || ty.is_pointer() {
        if *ngrn < 8 {
            let reg = INT_ARG_REGS[*ngrn];
            *ngrn += 1;
            return ArgClass::IntegerReg(reg);
        }
        // Overflow to stack
        let offset = align_stack_arg(*nsaa, 8);
        *nsaa = offset + 8;
        return ArgClass::Stack(offset);
    }

    // Rule 3-5: Composite types (structs, arrays)
    if ty.is_aggregate() {
        let size = ty.size(target);

        // Rule 3: Check for HFA (Homogeneous Floating-point Aggregate)
        if let Some(hfa) = classify_hfa(ty, target) {
            let count = match hfa {
                HfaClass::SingleFloat(n) => n as usize,
                HfaClass::DoubleFloat(n) => n as usize,
            };

            // HFA: each member goes in a consecutive v-register
            if *nsrn + count <= 8 {
                // Enough v-registers available — use first one as the classification.
                // For HFA passing, the caller will place each member in consecutive
                // v-registers starting from NSRN.
                let first_reg = FLOAT_ARG_REGS[*nsrn];
                *nsrn += count;
                // Return the first register; consumers must know this is an HFA
                // and use consecutive registers. For simplicity with the current
                // ArgClass enum, if count == 1 we return FloatReg, otherwise we
                // still return FloatReg for the first member. In practice, multi-member
                // HFAs require the isel to handle consecutive register allocation.
                return ArgClass::FloatReg(first_reg);
            }
            // Not enough v-registers: fall through to stack
            // Per AAPCS64: do NOT split between registers and stack
            *nsrn = 8; // Mark NSRN as exhausted
            let align = std::cmp::max(8, ty.alignment(target) as i32);
            let offset = align_stack_arg(*nsaa, align);
            *nsaa = offset + align_up(size as i32, 8);
            return ArgClass::Stack(offset);
        }

        // Rule 5: Composite types > 16 bytes → indirect (pointer in x-register)
        if size > 16 {
            if *ngrn < 8 {
                let reg = INT_ARG_REGS[*ngrn];
                *ngrn += 1;
                return ArgClass::Indirect(reg);
            }
            // Pointer on stack
            let offset = align_stack_arg(*nsaa, 8);
            *nsaa = offset + 8;
            return ArgClass::Stack(offset);
        }

        // Rule 4: Non-HFA composite ≤ 16 bytes → 1 or 2 x-registers
        let num_regs = if size <= 8 { 1 } else { 2 };
        if *ngrn + num_regs <= 8 {
            if num_regs == 1 {
                let reg = INT_ARG_REGS[*ngrn];
                *ngrn += 1;
                return ArgClass::IntegerReg(reg);
            } else {
                let reg1 = INT_ARG_REGS[*ngrn];
                let reg2 = INT_ARG_REGS[*ngrn + 1];
                *ngrn += 2;
                return ArgClass::IntegerRegPair(reg1, reg2);
            }
        }
        // Per AAPCS64: do NOT split between registers and stack
        *ngrn = 8; // Mark NGRN as exhausted
        let align = std::cmp::max(8, ty.alignment(target) as i32);
        let offset = align_stack_arg(*nsaa, align);
        *nsaa = offset + align_up(size as i32, 8);
        return ArgClass::Stack(offset);
    }

    // Fallback: treat as integer (void pointers, function pointers, etc.)
    if *ngrn < 8 {
        let reg = INT_ARG_REGS[*ngrn];
        *ngrn += 1;
        ArgClass::IntegerReg(reg)
    } else {
        let offset = align_stack_arg(*nsaa, 8);
        *nsaa = offset + 8;
        ArgClass::Stack(offset)
    }
}

// ---------------------------------------------------------------------------
// classify_return — AAPCS64 Return Value Classification
// ---------------------------------------------------------------------------

/// Classifies the return type of a function according to AAPCS64 rules.
///
/// # AAPCS64 Return Rules
///
/// - Void → no return value
/// - Fundamental integer/pointer ≤ 64 bits → x0
/// - Fundamental float → s0, double → d0 (v0 register)
/// - HFA ≤ 4 members → consecutive v-registers (v0, v1, ...)
/// - Non-HFA composite ≤ 8 bytes → x0
/// - Non-HFA composite 9-16 bytes → x0 + x1
/// - Composite > 16 bytes → indirect via x8 (caller provides buffer address)
pub fn classify_return(return_type: &IrType, target: &TargetConfig) -> ReturnClass {
    // Void return
    if return_type.is_void() {
        return ReturnClass::Void;
    }

    // Fundamental floating-point types → v0
    if return_type.is_float() {
        return ReturnClass::FloatReg;
    }

    // Fundamental integer/pointer types → x0
    if return_type.is_integer() || return_type.is_pointer() {
        return ReturnClass::IntegerReg;
    }

    // Composite types
    if return_type.is_aggregate() {
        let size = return_type.size(target);

        // Check for HFA
        if let Some(hfa) = classify_hfa(return_type, target) {
            let count = match hfa {
                HfaClass::SingleFloat(n) => n,
                HfaClass::DoubleFloat(n) => n,
            };
            return ReturnClass::FloatRegMultiple(count);
        }

        // Composite > 16 bytes → indirect via x8
        if size > 16 {
            return ReturnClass::Indirect;
        }

        // Non-HFA composite ≤ 8 bytes → x0
        if size <= 8 {
            return ReturnClass::IntegerReg;
        }

        // Non-HFA composite 9-16 bytes → x0 + x1
        return ReturnClass::IntegerRegPair;
    }

    // Fallback: treat as integer
    ReturnClass::IntegerReg
}

// ---------------------------------------------------------------------------
// compute_frame_layout — Stack Frame Layout Computation
// ---------------------------------------------------------------------------

/// Computes the complete stack frame layout for an AArch64 function.
///
/// The layout is determined by:
/// 1. Which callee-saved registers are used (from AllocationResult)
/// 2. How many spill slots the register allocator consumed
/// 3. Local variable space needed by the function
/// 4. Outgoing argument area for any function calls
///
/// The frame size is always rounded up to a multiple of 16 bytes to satisfy
/// the AArch64 hardware-enforced SP alignment requirement.
pub fn compute_frame_layout(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
) -> Aarch64FrameLayout {
    // Always use frame pointer for debuggability and simplicity.
    let uses_frame_pointer = true;

    // Step 1: Separate used callee-saved registers into GPRs and FPRs.
    let mut used_gpr_callee: Vec<PhysReg> = Vec::new();
    let mut used_fpr_callee: Vec<PhysReg> = Vec::new();

    for &reg in &alloc_result.used_callee_saved {
        if is_callee_saved_gpr(reg) {
            used_gpr_callee.push(reg);
        } else if is_callee_saved_fpr(reg) {
            used_fpr_callee.push(reg);
        }
    }

    // Sort for deterministic layout.
    used_gpr_callee.sort_by_key(|r| r.0);
    used_fpr_callee.sort_by_key(|r| r.0);

    // Step 2: Build STP pairs for callee-saved registers.
    // GPR pairs (x19+x20, x21+x22, etc.)
    let mut gpr_pairs: Vec<(PhysReg, PhysReg)> = Vec::new();
    let mut gpr_singles: Vec<PhysReg> = Vec::new();
    {
        let mut i = 0;
        while i + 1 < used_gpr_callee.len() {
            gpr_pairs.push((used_gpr_callee[i], used_gpr_callee[i + 1]));
            i += 2;
        }
        if i < used_gpr_callee.len() {
            gpr_singles.push(used_gpr_callee[i]);
        }
    }

    // FPR pairs (d8+d9, d10+d11, etc.)
    let mut fpr_pairs: Vec<(PhysReg, PhysReg)> = Vec::new();
    let mut fpr_singles: Vec<PhysReg> = Vec::new();
    {
        let mut i = 0;
        while i + 1 < used_fpr_callee.len() {
            fpr_pairs.push((used_fpr_callee[i], used_fpr_callee[i + 1]));
            i += 2;
        }
        if i < used_fpr_callee.len() {
            fpr_singles.push(used_fpr_callee[i]);
        }
    }

    // Combine all singles (GPR and FPR).
    let mut all_singles: Vec<PhysReg> = Vec::new();
    all_singles.extend_from_slice(&gpr_singles);
    all_singles.extend_from_slice(&fpr_singles);

    // Step 3: Compute sizes of each frame region.
    // FP+LR save area: always 16 bytes (STP x29, x30)
    let fp_lr_size: u32 = 16;

    // Callee-saved GPR pairs: 16 bytes each
    let gpr_pair_size: u32 = (gpr_pairs.len() as u32) * 16;

    // Callee-saved FPR pairs: 16 bytes each (d-register pairs)
    let fpr_pair_size: u32 = (fpr_pairs.len() as u32) * 16;

    // Singles: 8 bytes each
    let singles_size: u32 = (all_singles.len() as u32) * 8;
    // Round singles area up to 16-byte alignment if non-zero.
    let singles_size_aligned: u32 = if singles_size > 0 {
        align_up_u32(singles_size, 16)
    } else {
        0
    };

    // Spill slots: 8 bytes each (pointer-sized on AArch64)
    let spill_size: u32 = alloc_result.num_spill_slots * 8;

    // Local variable space: estimate from function's alloca instructions.
    // For now, use a heuristic based on the number of parameters and blocks.
    let locals_size: u32 = estimate_locals_size(function, target);

    // Outgoing argument area: compute from the maximum call argument count.
    let arg_area_size: u32 = estimate_arg_area_size(function, target);

    // Step 4: Compute total frame size.
    let total_unaligned = fp_lr_size
        + gpr_pair_size
        + fpr_pair_size
        + singles_size_aligned
        + spill_size
        + locals_size
        + arg_area_size;

    // Frame size MUST be 16-byte aligned (hardware requirement).
    let frame_size = align_up_u32(total_unaligned, target.stack_alignment);

    // Step 5: Assign offsets from SP (growing from high to low).
    // FP+LR are at the top of the frame (highest address within frame).
    let fp_offset = (frame_size - fp_lr_size) as i32;
    let lr_offset = fp_offset + 8; // LR is 8 bytes above FP in the pair

    // Callee-saved GPR pairs start below FP+LR area.
    let mut current_offset = fp_offset - 16; // First pair below FP+LR
    let mut callee_saved_gpr_pairs: Vec<((PhysReg, PhysReg), i32)> = Vec::new();
    for &pair in &gpr_pairs {
        callee_saved_gpr_pairs.push((pair, current_offset));
        current_offset -= 16;
    }

    // Callee-saved FPR pairs below GPR pairs.
    let mut callee_saved_fpr_pairs: Vec<((PhysReg, PhysReg), i32)> = Vec::new();
    for &pair in &fpr_pairs {
        callee_saved_fpr_pairs.push((pair, current_offset));
        current_offset -= 16;
    }

    // Singles below FPR pairs.
    let mut callee_saved_singles_with_offsets: Vec<(PhysReg, i32)> = Vec::new();
    for &single in &all_singles {
        callee_saved_singles_with_offsets.push((single, current_offset + 8));
        current_offset -= 8;
    }

    // Align current_offset to 16-byte boundary after singles.
    if !all_singles.is_empty() {
        let aligned = align_down(current_offset + 16, 16);
        current_offset = aligned - 16;
    }

    // Locals offset is below the callee-saved area.
    let locals_offset = current_offset;

    Aarch64FrameLayout {
        frame_size,
        fp_offset,
        lr_offset,
        callee_saved_gpr_pairs,
        callee_saved_fpr_pairs,
        callee_saved_singles: callee_saved_singles_with_offsets,
        locals_offset,
        locals_size,
        spill_size,
        arg_area_size,
        uses_frame_pointer,
    }
}

/// Estimates the local variable space needed by a function.
///
/// Scans the function's basic blocks for Alloca instructions and sums
/// their sizes. Returns a 16-byte-aligned size.
fn estimate_locals_size(function: &Function, target: &TargetConfig) -> u32 {
    use crate::ir::Instruction;

    let mut total: u32 = 0;
    for block in &function.blocks {
        for instr in &block.instructions {
            if let Instruction::Alloca { ty, .. } = instr {
                let elem_size = ty.size(target) as u32;
                // Each alloca allocates at least one element. The count field
                // is an Option<Value> for variable-length allocations; for fixed
                // allocations we assume 1 element.
                total += elem_size;
            }
        }
    }

    // Ensure minimum allocation and 16-byte alignment.
    align_up_u32(total, 16)
}

/// Estimates the outgoing argument area size based on the maximum number
/// of stack-passed arguments in any call within the function.
///
/// The argument area must be large enough for the call with the most
/// stack arguments. Returns a 16-byte-aligned size.
fn estimate_arg_area_size(function: &Function, _target: &TargetConfig) -> u32 {
    use crate::ir::Instruction;

    let mut max_stack_args: u32 = 0;

    for block in &function.blocks {
        for instr in &block.instructions {
            if let Instruction::Call { args, .. } = instr {
                // Count how many arguments would spill to stack.
                let num_args = args.len();
                // Conservative estimate: args beyond 8 go to stack.
                // (In reality, it depends on types, but this is safe.)
                if num_args > 8 {
                    let stack_count = (num_args - 8) as u32;
                    let stack_bytes = stack_count * 8; // 8 bytes per stack slot
                    if stack_bytes > max_stack_args {
                        max_stack_args = stack_bytes;
                    }
                }
            }
        }
    }

    align_up_u32(max_stack_args, 16)
}

// ---------------------------------------------------------------------------
// generate_prologue — Function Prologue Generation
// ---------------------------------------------------------------------------

/// Generates the AAPCS64-compliant function prologue instruction sequence.
///
/// The prologue:
/// 1. Saves FP+LR and allocates the frame (pre-indexed STP for small frames,
///    SUB SP + STP for large frames).
/// 2. Sets up the frame pointer (x29 = SP + fp_offset or MOV x29, sp).
/// 3. Saves all used callee-saved GPR pairs via STP.
/// 4. Saves all used callee-saved FPR pairs via STP.
/// 5. Saves any unpaired callee-saved registers via STR.
pub fn generate_prologue(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
) -> Vec<MachineInstr> {
    let layout = compute_frame_layout(function, alloc_result, target);
    let mut instrs: Vec<MachineInstr> = Vec::new();

    if layout.frame_size == 0 {
        // No frame needed (extremely unlikely but handle gracefully).
        return instrs;
    }

    // Determine the frame allocation strategy based on size.
    if layout.frame_size <= 504 {
        // Small frame: STP x29, x30, [sp, #-frame_size]! (pre-indexed)
        // This atomically allocates the frame and saves FP+LR.
        instrs.push(MachineInstr::with_operands(
            OP_STP_PRE,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(LR),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(-(layout.frame_size as i64)),
            ],
        ));

        // MOV x29, sp (set frame pointer to current SP)
        if layout.uses_frame_pointer {
            instrs.push(MachineInstr::with_operands(
                OP_MOV_RR,
                vec![
                    MachineOperand::Register(FP),
                    MachineOperand::Register(SP),
                ],
            ));
        }
    } else if layout.frame_size <= 4095 {
        // Medium frame: SUB sp, sp, #frame_size then STP at offset.
        instrs.push(MachineInstr::with_operands(
            OP_SUB_RI,
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(layout.frame_size as i64),
            ],
        ));

        // STP x29, x30, [sp, #fp_offset]
        instrs.push(MachineInstr::with_operands(
            OP_STP_OFFSET,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(LR),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(layout.fp_offset as i64),
            ],
        ));

        // ADD x29, sp, #fp_offset
        if layout.uses_frame_pointer {
            instrs.push(MachineInstr::with_operands(
                OP_ADD_RI,
                vec![
                    MachineOperand::Register(FP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(layout.fp_offset as i64),
                ],
            ));
        }
    } else {
        // Very large frame: use x16 (IP0) as scratch to load the size.
        emit_load_large_immediate(&mut instrs, X16, layout.frame_size as u64);

        // SUB sp, sp, x16
        instrs.push(MachineInstr::with_operands(
            OP_SUB_RR,
            vec![
                MachineOperand::Register(SP),
                MachineOperand::Register(SP),
                MachineOperand::Register(X16),
            ],
        ));

        // STP x29, x30, [sp, #fp_offset]
        instrs.push(MachineInstr::with_operands(
            OP_STP_OFFSET,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(LR),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(layout.fp_offset as i64),
            ],
        ));

        // ADD x29, sp, #fp_offset
        if layout.uses_frame_pointer {
            instrs.push(MachineInstr::with_operands(
                OP_ADD_RI,
                vec![
                    MachineOperand::Register(FP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(layout.fp_offset as i64),
                ],
            ));
        }
    }

    // Save callee-saved GPR pairs with STP.
    for &((r1, r2), offset) in &layout.callee_saved_gpr_pairs {
        instrs.push(MachineInstr::with_operands(
            OP_STP_OFFSET,
            vec![
                MachineOperand::Register(r1),
                MachineOperand::Register(r2),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(offset as i64),
            ],
        ));
    }

    // Save callee-saved FPR pairs with STP.
    for &((r1, r2), offset) in &layout.callee_saved_fpr_pairs {
        instrs.push(MachineInstr::with_operands(
            OP_STP_FP_OFFSET,
            vec![
                MachineOperand::Register(r1),
                MachineOperand::Register(r2),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(offset as i64),
            ],
        ));
    }

    // Save unpaired callee-saved registers with STR.
    for &(reg, offset) in &layout.callee_saved_singles {
        let opcode = if is_fp_reg(reg) {
            OP_STR_FP_IMM
        } else {
            OP_STR_IMM
        };
        instrs.push(MachineInstr::with_operands(
            opcode,
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(offset as i64),
            ],
        ));
    }

    instrs
}

// ---------------------------------------------------------------------------
// generate_epilogue — Function Epilogue Generation
// ---------------------------------------------------------------------------

/// Generates the AAPCS64-compliant function epilogue instruction sequence.
///
/// The epilogue reverses the prologue:
/// 1. Restores unpaired callee-saved registers via LDR.
/// 2. Restores callee-saved FPR pairs via LDP.
/// 3. Restores callee-saved GPR pairs via LDP.
/// 4. Restores FP+LR and deallocates frame (post-indexed LDP for small frames,
///    LDP at offset + ADD SP for large frames).
/// 5. Emits RET instruction.
pub fn generate_epilogue(
    function: &Function,
    alloc_result: &AllocationResult,
    target: &TargetConfig,
) -> Vec<MachineInstr> {
    let layout = compute_frame_layout(function, alloc_result, target);
    let mut instrs: Vec<MachineInstr> = Vec::new();

    if layout.frame_size == 0 {
        // No frame to deallocate; just return.
        instrs.push(MachineInstr::new(OP_RET));
        return instrs;
    }

    // Restore unpaired callee-saved registers (reverse order of saves).
    for &(reg, offset) in layout.callee_saved_singles.iter().rev() {
        let opcode = if is_fp_reg(reg) {
            OP_LDR_FP_IMM
        } else {
            OP_LDR_IMM
        };
        instrs.push(MachineInstr::with_operands(
            opcode,
            vec![
                MachineOperand::Register(reg),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(offset as i64),
            ],
        ));
    }

    // Restore callee-saved FPR pairs (reverse order).
    for &((r1, r2), offset) in layout.callee_saved_fpr_pairs.iter().rev() {
        instrs.push(MachineInstr::with_operands(
            OP_LDP_FP_OFFSET,
            vec![
                MachineOperand::Register(r1),
                MachineOperand::Register(r2),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(offset as i64),
            ],
        ));
    }

    // Restore callee-saved GPR pairs (reverse order).
    for &((r1, r2), offset) in layout.callee_saved_gpr_pairs.iter().rev() {
        instrs.push(MachineInstr::with_operands(
            OP_LDP_OFFSET,
            vec![
                MachineOperand::Register(r1),
                MachineOperand::Register(r2),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(offset as i64),
            ],
        ));
    }

    // Restore FP+LR and deallocate frame.
    if layout.frame_size <= 504 {
        // Small frame: LDP x29, x30, [sp], #frame_size (post-indexed)
        instrs.push(MachineInstr::with_operands(
            OP_LDP_POST,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(LR),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(layout.frame_size as i64),
            ],
        ));
    } else {
        // Large frame: LDP at offset, then ADD sp.
        instrs.push(MachineInstr::with_operands(
            OP_LDP_OFFSET,
            vec![
                MachineOperand::Register(FP),
                MachineOperand::Register(LR),
                MachineOperand::Register(SP),
                MachineOperand::Immediate(layout.fp_offset as i64),
            ],
        ));

        if layout.frame_size <= 4095 {
            instrs.push(MachineInstr::with_operands(
                OP_ADD_RI,
                vec![
                    MachineOperand::Register(SP),
                    MachineOperand::Register(SP),
                    MachineOperand::Immediate(layout.frame_size as i64),
                ],
            ));
        } else {
            // Very large frame: use x16 (IP0) as scratch.
            emit_load_large_immediate(&mut instrs, X16, layout.frame_size as u64);
            instrs.push(MachineInstr::with_operands(
                OP_ADD_RI,
                vec![
                    MachineOperand::Register(SP),
                    MachineOperand::Register(SP),
                    MachineOperand::Register(X16),
                ],
            ));
        }
    }

    // RET instruction (branches to x30 / LR).
    instrs.push(MachineInstr::new(OP_RET));

    instrs
}

// ---------------------------------------------------------------------------
// generate_argument_loads — Move arguments from ABI locations to allocated regs
// ---------------------------------------------------------------------------

/// Generates instructions to move function arguments from their ABI-specified
/// locations (argument registers or stack slots) to the registers assigned
/// by the register allocator.
///
/// For each parameter:
/// - If passed in an x-register: MOV from argument register to allocated register.
/// - If passed in a v-register: FMOV from argument FP register.
/// - If passed on the stack: LDR from stack offset.
/// - If passed by reference (indirect): LDR the pointer from the argument register.
pub fn generate_argument_loads(
    params: &[(String, IrType)],
    classifications: &[ArgClass],
    alloc_result: &AllocationResult,
) -> Vec<MachineInstr> {
    let mut instrs: Vec<MachineInstr> = Vec::new();

    for (i, ((_name, ty), class)) in params.iter().zip(classifications.iter()).enumerate() {
        // Find the allocated register for this parameter value.
        // Parameters are assigned Value indices starting from 0.
        let param_value = Value(i as u32);
        let dest_reg = find_allocated_reg(alloc_result, param_value);

        match class {
            ArgClass::IntegerReg(src_reg) => {
                if let Some(dest) = dest_reg {
                    if dest != *src_reg {
                        // MOV dest, src (only if different registers)
                        instrs.push(MachineInstr::with_operands(
                            OP_MOV_RR,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(*src_reg),
                            ],
                        ));
                    }
                }
            }
            ArgClass::FloatReg(src_reg) => {
                if let Some(dest) = dest_reg {
                    if dest != *src_reg {
                        // FMOV dest, src
                        instrs.push(MachineInstr::with_operands(
                            OP_FMOV_RR,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(*src_reg),
                            ],
                        ));
                    }
                }
            }
            ArgClass::Stack(offset) => {
                if let Some(dest) = dest_reg {
                    // LDR dest, [sp, #offset]
                    // Note: the offset is relative to the *caller's* SP, which
                    // after our prologue is at SP + frame_size. The actual
                    // offset computation depends on the frame layout and is
                    // handled during final encoding.
                    let opcode = if ty.is_float() {
                        OP_LDR_FP_IMM
                    } else {
                        OP_LDR_IMM
                    };
                    instrs.push(MachineInstr::with_operands(
                        opcode,
                        vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Register(SP),
                            MachineOperand::Immediate(*offset as i64),
                        ],
                    ));
                }
            }
            ArgClass::IntegerRegPair(r1, r2) => {
                // For register pairs, the first register holds the low 8 bytes
                // and the second holds the high 8 bytes. Move the first to the
                // allocated register (the pair semantics are handled at a higher level).
                if let Some(dest) = dest_reg {
                    if dest != *r1 {
                        instrs.push(MachineInstr::with_operands(
                            OP_MOV_RR,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(*r1),
                            ],
                        ));
                    }
                }
            }
            ArgClass::Indirect(ptr_reg) => {
                // The argument is a pointer to the actual data. Load the pointer
                // into the destination register.
                if let Some(dest) = dest_reg {
                    if dest != *ptr_reg {
                        instrs.push(MachineInstr::with_operands(
                            OP_MOV_RR,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(*ptr_reg),
                            ],
                        ));
                    }
                }
            }
        }
    }

    instrs
}

// ---------------------------------------------------------------------------
// generate_call_sequence — Call Site Instruction Sequence
// ---------------------------------------------------------------------------

/// Generates the instruction sequence for a function call per AAPCS64.
///
/// This includes:
/// 1. Moving arguments to the correct argument registers (x0-x7, v0-v7).
/// 2. Setting up stack arguments for overflow parameters.
/// 3. Emitting BL (direct) or BLR (indirect) call instruction.
/// 4. The return value will be in x0/v0 after the call.
///
/// # Arguments
///
/// * `callee` — Direct (named) or indirect (function pointer) call target.
/// * `args` — IR Value references for each argument.
/// * `return_ty` — The return type to determine return value handling.
/// * `value_map` — Mapping from IR Values to their allocated physical registers.
pub fn generate_call_sequence(
    callee: &Callee,
    args: &[Value],
    return_ty: &IrType,
    value_map: &HashMap<Value, PhysReg>,
) -> Vec<MachineInstr> {
    let mut instrs: Vec<MachineInstr> = Vec::new();

    // Step 1: Classify arguments.
    // We need type info for proper classification, but we only have Values here.
    // Use a simplified classification based on register class of the source value:
    // if the source is in an FP register, classify as float; otherwise integer.
    let mut ngrn: usize = 0; // Next GPR number
    let mut nsrn: usize = 0; // Next SIMD/FP register number

    // Move arguments to their designated argument registers.
    for &arg_val in args {
        let src_reg = value_map.get(&arg_val).copied();
        if let Some(src) = src_reg {
            if is_fp_reg(src) {
                // Float argument → v-register
                if nsrn < 8 {
                    let dest = FLOAT_ARG_REGS[nsrn];
                    nsrn += 1;
                    if dest != src {
                        instrs.push(MachineInstr::with_operands(
                            OP_FMOV_RR,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(src),
                            ],
                        ));
                    }
                } else {
                    // Stack argument: STR to outgoing arg area.
                    // The exact offset is computed relative to SP.
                    let offset = ((nsrn + ngrn - 16) * 8) as i64;
                    instrs.push(MachineInstr::with_operands(
                        OP_STR_FP_IMM,
                        vec![
                            MachineOperand::Register(src),
                            MachineOperand::Register(SP),
                            MachineOperand::Immediate(offset.max(0)),
                        ],
                    ));
                    nsrn += 1;
                }
            } else {
                // Integer/pointer argument → x-register
                if ngrn < 8 {
                    let dest = INT_ARG_REGS[ngrn];
                    ngrn += 1;
                    if dest != src {
                        instrs.push(MachineInstr::with_operands(
                            OP_MOV_RR,
                            vec![
                                MachineOperand::Register(dest),
                                MachineOperand::Register(src),
                            ],
                        ));
                    }
                } else {
                    // Stack argument.
                    let stack_slot = (ngrn + nsrn - 16) as i64;
                    instrs.push(MachineInstr::with_operands(
                        OP_STR_IMM,
                        vec![
                            MachineOperand::Register(src),
                            MachineOperand::Register(SP),
                            MachineOperand::Immediate((stack_slot * 8).max(0)),
                        ],
                    ));
                    ngrn += 1;
                }
            }
        }
    }

    // Step 2: Emit the call instruction.
    match callee {
        Callee::Direct(name) => {
            // BL <symbol>
            instrs.push(MachineInstr::with_operands(
                OP_BL,
                vec![MachineOperand::Symbol(name.clone())],
            ));
        }
        Callee::Indirect(val) => {
            // BLR Xn — branch to register holding function pointer
            if let Some(&reg) = value_map.get(val) {
                instrs.push(MachineInstr::with_operands(
                    OP_BLR,
                    vec![MachineOperand::Register(reg)],
                ));
            } else {
                // Fallback: use x16 (IP0) as the target register.
                instrs.push(MachineInstr::with_operands(
                    OP_BLR,
                    vec![MachineOperand::Register(X16)],
                ));
            }
        }
    }

    instrs
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Aligns a stack argument offset upward to the specified alignment.
#[inline]
fn align_stack_arg(offset: i32, alignment: i32) -> i32 {
    debug_assert!(alignment > 0);
    ((offset + alignment - 1) / alignment) * alignment
}

/// Rounds a signed value up to the next multiple of the given alignment.
#[inline]
fn align_up(value: i32, alignment: i32) -> i32 {
    debug_assert!(alignment > 0);
    ((value + alignment - 1) / alignment) * alignment
}

/// Rounds an unsigned value up to the next multiple of the given alignment.
#[inline]
fn align_up_u32(value: u32, alignment: u32) -> u32 {
    debug_assert!(alignment > 0);
    ((value + alignment - 1) / alignment) * alignment
}

/// Rounds a signed value down to the previous multiple of the given alignment.
#[inline]
fn align_down(value: i32, alignment: i32) -> i32 {
    debug_assert!(alignment > 0);
    (value / alignment) * alignment
}

/// Emits MOVZ/MOVK instruction sequence to load a large immediate into a register.
///
/// For values > 16 bits, this generates:
/// ```asm
/// movz Xd, #imm16, lsl #0    ; Load bits [15:0]
/// movk Xd, #imm16, lsl #16   ; Load bits [31:16]
/// movk Xd, #imm16, lsl #32   ; Load bits [47:32] (if needed)
/// movk Xd, #imm16, lsl #48   ; Load bits [63:48] (if needed)
/// ```
fn emit_load_large_immediate(instrs: &mut Vec<MachineInstr>, dest: PhysReg, value: u64) {
    // MOVZ — load lowest 16 bits and zero the rest.
    let lo16 = (value & 0xFFFF) as i64;
    instrs.push(MachineInstr::with_operands(
        OP_MOVZ,
        vec![
            MachineOperand::Register(dest),
            MachineOperand::Immediate(lo16),
            MachineOperand::Immediate(0), // shift amount
        ],
    ));

    // MOVK for bits [31:16] if non-zero.
    let hi16 = ((value >> 16) & 0xFFFF) as i64;
    if hi16 != 0 || value > 0xFFFF_FFFF {
        instrs.push(MachineInstr::with_operands(
            OP_MOVK,
            vec![
                MachineOperand::Register(dest),
                MachineOperand::Immediate(hi16),
                MachineOperand::Immediate(16), // shift amount
            ],
        ));
    }

    // MOVK for bits [47:32] if value exceeds 32 bits.
    if value > 0xFFFF_FFFF {
        let mid16 = ((value >> 32) & 0xFFFF) as i64;
        instrs.push(MachineInstr::with_operands(
            OP_MOVK,
            vec![
                MachineOperand::Register(dest),
                MachineOperand::Immediate(mid16),
                MachineOperand::Immediate(32),
            ],
        ));
    }

    // MOVK for bits [63:48] if value exceeds 48 bits.
    if value > 0xFFFF_FFFF_FFFF {
        let top16 = ((value >> 48) & 0xFFFF) as i64;
        instrs.push(MachineInstr::with_operands(
            OP_MOVK,
            vec![
                MachineOperand::Register(dest),
                MachineOperand::Immediate(top16),
                MachineOperand::Immediate(48),
            ],
        ));
    }
}

/// Looks up the physical register assigned to an IR Value in the allocation result.
fn find_allocated_reg(alloc_result: &AllocationResult, value: Value) -> Option<PhysReg> {
    for interval in &alloc_result.intervals {
        if interval.value == value {
            return interval.assigned_reg;
        }
    }
    None
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::target::TargetConfig;
    use crate::ir::types::IrType;
    use crate::ir::cfg::BasicBlock;
    use crate::ir::instructions::BlockId;

    /// Creates a default AArch64 target configuration for testing.
    fn test_target() -> TargetConfig {
        TargetConfig::aarch64()
    }

    /// Creates a minimal Function for testing.
    fn make_test_function(
        name: &str,
        params: Vec<(String, IrType)>,
        return_type: IrType,
    ) -> Function {
        Function {
            name: name.to_string(),
            return_type,
            params,
            blocks: vec![BasicBlock::new(BlockId(0), "entry".to_string())],
            entry_block: BlockId(0),
            is_definition: true,
        }
    }

    /// Creates an AllocationResult with no used callee-saved registers.
    fn empty_alloc_result() -> AllocationResult {
        AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: 0,
            used_callee_saved: Vec::new(),
        }
    }

    /// Creates an AllocationResult with specified callee-saved registers.
    fn alloc_with_callee_saved(regs: Vec<PhysReg>) -> AllocationResult {
        AllocationResult {
            intervals: Vec::new(),
            num_spill_slots: 0,
            used_callee_saved: regs,
        }
    }

    // ===================================================================
    // Argument Classification Tests
    // ===================================================================

    #[test]
    fn test_single_i32_arg() {
        let target = test_target();
        let params = vec![("x".to_string(), IrType::I32)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::IntegerReg(X0));
    }

    #[test]
    fn test_single_f64_arg() {
        let target = test_target();
        let params = vec![("x".to_string(), IrType::F64)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::FloatReg(V0));
    }

    #[test]
    fn test_eight_integer_args() {
        let target = test_target();
        let params: Vec<(String, IrType)> = (0..8)
            .map(|i| (format!("arg{}", i), IrType::I64))
            .collect();
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 8);
        for i in 0..8 {
            assert_eq!(result[i], ArgClass::IntegerReg(INT_ARG_REGS[i]));
        }
    }

    #[test]
    fn test_nine_integer_args_overflow() {
        let target = test_target();
        let params: Vec<(String, IrType)> = (0..9)
            .map(|i| (format!("arg{}", i), IrType::I64))
            .collect();
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 9);
        for i in 0..8 {
            assert_eq!(result[i], ArgClass::IntegerReg(INT_ARG_REGS[i]));
        }
        assert_eq!(result[8], ArgClass::Stack(0));
    }

    #[test]
    fn test_eight_float_args() {
        let target = test_target();
        let params: Vec<(String, IrType)> = (0..8)
            .map(|i| (format!("arg{}", i), IrType::F64))
            .collect();
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 8);
        for i in 0..8 {
            assert_eq!(result[i], ArgClass::FloatReg(FLOAT_ARG_REGS[i]));
        }
    }

    #[test]
    fn test_mixed_int_float_independent_counters() {
        // CRITICAL: Integer and float counters are INDEPENDENT on AArch64.
        let target = test_target();
        let params = vec![
            ("a".to_string(), IrType::I32),
            ("b".to_string(), IrType::F64),
            ("c".to_string(), IrType::I64),
            ("d".to_string(), IrType::F32),
        ];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], ArgClass::IntegerReg(X0)); // NGRN=0 → x0
        assert_eq!(result[1], ArgClass::FloatReg(V0));    // NSRN=0 → v0
        assert_eq!(result[2], ArgClass::IntegerReg(X1)); // NGRN=1 → x1
        assert_eq!(result[3], ArgClass::FloatReg(V1));    // NSRN=1 → v1
    }

    #[test]
    fn test_small_struct_two_ints() {
        // struct { int a; int b; } → 8 bytes → single x-register
        let target = test_target();
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        let params = vec![("s".to_string(), struct_ty)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::IntegerReg(X0));
    }

    #[test]
    fn test_medium_struct_pair() {
        // struct { long a; long b; } → 16 bytes → IntegerRegPair
        let target = test_target();
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        let params = vec![("s".to_string(), struct_ty)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::IntegerRegPair(X0, X1));
    }

    #[test]
    fn test_large_struct_indirect() {
        // struct { long a, b, c; } → 24 bytes → Indirect
        let target = test_target();
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        let params = vec![("s".to_string(), struct_ty)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::Indirect(X0));
    }

    #[test]
    fn test_hfa_two_doubles() {
        // struct { double x, y; } → HFA(Double, 2) → FloatReg
        let target = test_target();
        let hfa_ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::F64],
            packed: false,
        };
        let params = vec![("s".to_string(), hfa_ty)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::FloatReg(V0));
    }

    #[test]
    fn test_hfa_four_floats() {
        // struct { float a, b, c, d; } → HFA(Single, 4)
        let target = test_target();
        let hfa_ty = IrType::Struct {
            fields: vec![IrType::F32, IrType::F32, IrType::F32, IrType::F32],
            packed: false,
        };
        let params = vec![("s".to_string(), hfa_ty)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::FloatReg(V0));
    }

    #[test]
    fn test_pointer_arg() {
        let target = test_target();
        let ptr_ty = IrType::Pointer(Box::new(IrType::I32));
        let params = vec![("p".to_string(), ptr_ty)];
        let result = classify_arguments(&params, &target);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ArgClass::IntegerReg(X0));
    }

    // ===================================================================
    // Return Classification Tests
    // ===================================================================

    #[test]
    fn test_return_void() {
        let target = test_target();
        let result = classify_return(&IrType::Void, &target);
        assert!(matches!(result, ReturnClass::Void));
    }

    #[test]
    fn test_return_i32() {
        let target = test_target();
        let result = classify_return(&IrType::I32, &target);
        assert!(matches!(result, ReturnClass::IntegerReg));
    }

    #[test]
    fn test_return_f64() {
        let target = test_target();
        let result = classify_return(&IrType::F64, &target);
        assert!(matches!(result, ReturnClass::FloatReg));
    }

    #[test]
    fn test_return_small_struct() {
        let target = test_target();
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I32, IrType::I32],
            packed: false,
        };
        let result = classify_return(&struct_ty, &target);
        // 8 bytes → fits in x0
        assert!(matches!(result, ReturnClass::IntegerReg));
    }

    #[test]
    fn test_return_medium_struct() {
        let target = test_target();
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64],
            packed: false,
        };
        let result = classify_return(&struct_ty, &target);
        // 16 bytes → x0+x1
        assert!(matches!(result, ReturnClass::IntegerRegPair));
    }

    #[test]
    fn test_return_large_struct_indirect() {
        let target = test_target();
        let struct_ty = IrType::Struct {
            fields: vec![IrType::I64, IrType::I64, IrType::I64],
            packed: false,
        };
        let result = classify_return(&struct_ty, &target);
        // 24 bytes → indirect via x8
        assert!(matches!(result, ReturnClass::Indirect));
    }

    #[test]
    fn test_return_hfa() {
        let target = test_target();
        let hfa_ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::F64],
            packed: false,
        };
        let result = classify_return(&hfa_ty, &target);
        assert!(matches!(result, ReturnClass::FloatRegMultiple(2)));
    }

    // ===================================================================
    // HFA Detection Tests
    // ===================================================================

    #[test]
    fn test_hfa_single_float_pair() {
        let target = test_target();
        let ty = IrType::Struct {
            fields: vec![IrType::F32, IrType::F32],
            packed: false,
        };
        let hfa = classify_hfa(&ty, &target);
        assert_eq!(hfa, Some(HfaClass::SingleFloat(2)));
    }

    #[test]
    fn test_hfa_double_pair() {
        let target = test_target();
        let ty = IrType::Struct {
            fields: vec![IrType::F64, IrType::F64],
            packed: false,
        };
        let hfa = classify_hfa(&ty, &target);
        assert_eq!(hfa, Some(HfaClass::DoubleFloat(2)));
    }

    #[test]
    fn test_not_hfa_mixed_types() {
        let target = test_target();
        let ty = IrType::Struct {
            fields: vec![IrType::I32, IrType::F32],
            packed: false,
        };
        let hfa = classify_hfa(&ty, &target);
        assert!(hfa.is_none());
    }

    #[test]
    fn test_hfa_four_singles() {
        let target = test_target();
        let ty = IrType::Struct {
            fields: vec![IrType::F32, IrType::F32, IrType::F32, IrType::F32],
            packed: false,
        };
        let hfa = classify_hfa(&ty, &target);
        assert_eq!(hfa, Some(HfaClass::SingleFloat(4)));
    }

    #[test]
    fn test_not_hfa_five_members() {
        let target = test_target();
        let ty = IrType::Struct {
            fields: vec![
                IrType::F32, IrType::F32, IrType::F32, IrType::F32, IrType::F32,
            ],
            packed: false,
        };
        let hfa = classify_hfa(&ty, &target);
        assert!(hfa.is_none()); // >4 members
    }

    #[test]
    fn test_not_hfa_non_struct() {
        let target = test_target();
        let hfa = classify_hfa(&IrType::I32, &target);
        assert!(hfa.is_none());
    }

    // ===================================================================
    // Frame Layout Tests
    // ===================================================================

    #[test]
    fn test_minimal_frame_layout() {
        let target = test_target();
        let func = make_test_function("leaf", vec![], IrType::Void);
        let alloc = empty_alloc_result();
        let layout = compute_frame_layout(&func, &alloc, &target);

        // Minimum frame: FP+LR = 16 bytes.
        assert_eq!(layout.frame_size, 16);
        assert!(layout.frame_size % 16 == 0, "Frame must be 16-byte aligned");
        assert!(layout.uses_frame_pointer);
    }

    #[test]
    fn test_frame_with_callee_saved_pairs() {
        let target = test_target();
        let func = make_test_function("f", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![X19, X20]);
        let layout = compute_frame_layout(&func, &alloc, &target);

        // FP+LR (16) + one GPR pair (16) = 32 bytes minimum.
        assert!(layout.frame_size >= 32);
        assert!(layout.frame_size % 16 == 0);
        assert_eq!(layout.callee_saved_gpr_pairs.len(), 1);
    }

    #[test]
    fn test_frame_always_16_aligned() {
        let target = test_target();
        let func = make_test_function("f", vec![], IrType::Void);

        // Test with odd number of callee-saved regs to force alignment padding.
        let alloc = alloc_with_callee_saved(vec![X19, X20, X21]);
        let layout = compute_frame_layout(&func, &alloc, &target);

        assert!(layout.frame_size % 16 == 0, "Frame size {} not 16-byte aligned", layout.frame_size);
    }

    #[test]
    fn test_frame_with_fpr_callee_saved() {
        let target = test_target();
        let func = make_test_function("f", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![V8, V9]);
        let layout = compute_frame_layout(&func, &alloc, &target);

        assert!(layout.frame_size >= 32); // FP+LR (16) + FPR pair (16)
        assert_eq!(layout.callee_saved_fpr_pairs.len(), 1);
    }

    // ===================================================================
    // Prologue / Epilogue Tests
    // ===================================================================

    #[test]
    fn test_simple_prologue() {
        let target = test_target();
        let func = make_test_function("simple", vec![], IrType::Void);
        let alloc = empty_alloc_result();
        let prologue = generate_prologue(&func, &alloc, &target);

        // Should have at least STP (frame alloc + FP/LR save) + MOV (FP setup).
        assert!(!prologue.is_empty(), "Prologue should not be empty");

        // First instruction should be STP (pre-indexed) for small frames.
        assert_eq!(prologue[0].opcode, OP_STP_PRE, "First instr should be STP pre-indexed");

        // Second instruction should be MOV x29, sp (frame pointer setup).
        if prologue.len() > 1 {
            assert_eq!(prologue[1].opcode, OP_MOV_RR, "Second instr should be MOV FP, SP");
        }
    }

    #[test]
    fn test_simple_epilogue() {
        let target = test_target();
        let func = make_test_function("simple", vec![], IrType::Void);
        let alloc = empty_alloc_result();
        let epilogue = generate_epilogue(&func, &alloc, &target);

        assert!(!epilogue.is_empty(), "Epilogue should not be empty");

        // Last instruction should be RET.
        let last = epilogue.last().unwrap();
        assert_eq!(last.opcode, OP_RET, "Last epilogue instruction should be RET");
    }

    #[test]
    fn test_prologue_epilogue_symmetry() {
        let target = test_target();
        let func = make_test_function("sym", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![X19, X20, X21, X22]);
        let prologue = generate_prologue(&func, &alloc, &target);
        let epilogue = generate_epilogue(&func, &alloc, &target);

        // Count STP instructions in prologue.
        let stp_count = prologue
            .iter()
            .filter(|i| i.opcode == OP_STP_PRE || i.opcode == OP_STP_OFFSET)
            .count();

        // Count LDP instructions in epilogue (excluding the final RET).
        let ldp_count = epilogue
            .iter()
            .filter(|i| i.opcode == OP_LDP_POST || i.opcode == OP_LDP_OFFSET)
            .count();

        // Save and restore counts should match (STP count ≤ LDP count due to
        // FP+LR being restored with a separate LDP).
        assert_eq!(stp_count, ldp_count, "STP/LDP counts should be symmetric");
    }

    #[test]
    fn test_epilogue_ends_with_ret() {
        let target = test_target();
        let func = make_test_function("f", vec![], IrType::I32);
        let alloc = alloc_with_callee_saved(vec![X19, X20, V8, V9]);
        let epilogue = generate_epilogue(&func, &alloc, &target);

        let last = epilogue.last().expect("Epilogue should have instructions");
        assert_eq!(last.opcode, OP_RET);
    }

    // ===================================================================
    // Register Convention Tests
    // ===================================================================

    #[test]
    fn test_argument_registers() {
        assert_eq!(INT_ARG_REGS, [X0, X1, X2, X3, X4, X5, X6, X7]);
        assert_eq!(FLOAT_ARG_REGS, [V0, V1, V2, V3, V4, V5, V6, V7]);
    }

    #[test]
    fn test_callee_saved_gprs() {
        assert_eq!(
            CALLEE_SAVED_GPRS,
            [X19, X20, X21, X22, X23, X24, X25, X26, X27, X28]
        );
    }

    #[test]
    fn test_callee_saved_fprs() {
        assert_eq!(
            CALLEE_SAVED_FPRS,
            [V8, V9, V10, V11, V12, V13, V14, V15]
        );
    }

    #[test]
    fn test_fp_is_x29() {
        assert_eq!(FP, PhysReg(29));
    }

    #[test]
    fn test_lr_is_x30() {
        assert_eq!(LR, PhysReg(30));
    }

    #[test]
    fn test_x8_is_indirect_result() {
        assert_eq!(X8, PhysReg(8));
    }

    #[test]
    fn test_is_fp_reg() {
        assert!(is_fp_reg(V0));
        assert!(is_fp_reg(V31));
        assert!(!is_fp_reg(X0));
        assert!(!is_fp_reg(SP));
    }

    #[test]
    fn test_is_gpr() {
        assert!(is_gpr(X0));
        assert!(is_gpr(LR));
        assert!(is_gpr(SP));
        assert!(!is_gpr(V0));
    }

    #[test]
    fn test_is_callee_saved_gpr_fn() {
        assert!(is_callee_saved_gpr(X19));
        assert!(is_callee_saved_gpr(X28));
        assert!(!is_callee_saved_gpr(X0));
        assert!(!is_callee_saved_gpr(X18));
        assert!(!is_callee_saved_gpr(FP)); // x29 is FP, handled separately
    }

    #[test]
    fn test_is_callee_saved_fpr_fn() {
        assert!(is_callee_saved_fpr(V8));
        assert!(is_callee_saved_fpr(V15));
        assert!(!is_callee_saved_fpr(V0));
        assert!(!is_callee_saved_fpr(V16));
    }

    // ===================================================================
    // STP/LDP Pairing Tests
    // ===================================================================

    #[test]
    fn test_even_callee_saved_all_paired() {
        let target = test_target();
        let func = make_test_function("f", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![X19, X20, X21, X22]);
        let layout = compute_frame_layout(&func, &alloc, &target);

        assert_eq!(layout.callee_saved_gpr_pairs.len(), 2);
        assert!(layout.callee_saved_singles.is_empty());
    }

    #[test]
    fn test_odd_callee_saved_one_unpaired() {
        let target = test_target();
        let func = make_test_function("f", vec![], IrType::Void);
        let alloc = alloc_with_callee_saved(vec![X19, X20, X21]);
        let layout = compute_frame_layout(&func, &alloc, &target);

        assert_eq!(layout.callee_saved_gpr_pairs.len(), 1);
        assert_eq!(layout.callee_saved_singles.len(), 1);
    }

    // ===================================================================
    // Stack Alignment Tests
    // ===================================================================

    #[test]
    fn test_sp_always_16_aligned() {
        let target = test_target();
        // Test various combinations of callee-saved regs and spill slots.
        for num_regs in 0..=5 {
            let callee_saved: Vec<PhysReg> = CALLEE_SAVED_GPRS[..num_regs].to_vec();
            let mut alloc = alloc_with_callee_saved(callee_saved);
            alloc.num_spill_slots = 3; // Odd number of spill slots

            let func = make_test_function("test", vec![], IrType::Void);
            let layout = compute_frame_layout(&func, &alloc, &target);

            assert!(
                layout.frame_size % 16 == 0,
                "Frame size {} not 16-byte aligned with {} callee-saved regs and 3 spill slots",
                layout.frame_size,
                num_regs
            );
        }
    }
}
