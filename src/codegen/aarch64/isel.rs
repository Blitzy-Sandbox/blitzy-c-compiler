//! AArch64 instruction selection module.
//!
//! This module implements the core pattern-matching engine that translates
//! SSA-form IR instructions into AArch64 (A64 / ARMv8-A 64-bit) machine
//! instruction sequences.  AArch64 is a fixed-width ISA where every
//! instruction is 32 bits wide.
//!
//! Key AArch64 features exploited by the instruction selector:
//! - **Barrel shifter operands**: many data-processing instructions accept a
//!   shifted second operand, enabling `ADD Xd, Xn, Xm, LSL #n` in a single
//!   instruction.
//! - **Conditional select** (`CSEL`, `CSINC`, `CSINV`, `CSNEG`): branchless
//!   conditional value materialisation.
//! - **Load/store pair** (`LDP`, `STP`): efficient register save/restore.
//! - **PC-relative addressing** via `ADRP`+`ADD` for position-independent
//!   symbol references.
//! - **MOVZ/MOVK constant materialisation** using 16-bit halfword moves.

use std::collections::HashMap;

use crate::codegen::{
    CodeGenError, MachineInstr, MachineOperand, Relocation, RelocationType,
};
use crate::codegen::regalloc::PhysReg;
use crate::codegen::aarch64::abi::{
    generate_call_sequence,
    INT_ARG_REGS, FLOAT_ARG_REGS,
    X0, FP, LR, V0,
    is_fp_reg,
};
use crate::ir::{
    Callee, CastOp, CompareOp, Constant,
    FloatCompareOp, Function, Instruction, IrType,
    Terminator, Value,
};

// ===========================================================================
// Aarch64Opcode — comprehensive AArch64 instruction opcode enum
// ===========================================================================

/// Complete set of AArch64 (A64) opcodes needed for C code generation.
///
/// Each variant corresponds to a single A64 machine instruction.  The
/// `as_u32` / `from_u32` methods provide a stable numeric encoding that
/// is stored in `MachineInstr.opcode` and later consumed by the encoder.
///
/// Opcodes are grouped into ranges by functional category:
/// - `0x1000..` — Integer data-processing (arithmetic, logical, shift, move,
///   extend, bit-manipulation, conditional-select)
/// - `0x2000..` — Load / store (byte, halfword, word, double, pair, FP)
/// - `0x3000..` — PC-relative addressing (ADRP, ADR)
/// - `0x4000..` — Branches and control flow
/// - `0x5000..` — Floating-point data-processing, comparison, conversion
/// - `0x6000..` — System instructions (NOP, SVC, barriers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Aarch64Opcode {
    // -- Integer Arithmetic --
    ADD, ADDS, SUB, SUBS,
    MUL, MADD, MSUB,
    SDIV, UDIV,
    SMULL, UMULL, SMULH, UMULH,

    // -- Logical --
    AND, ANDS, ORR, ORN, EOR, EON, BIC, BICS,

    // -- Shift --
    LSL, LSR, ASR, ROR,

    // -- Move --
    MOV, MVN, MOVZ, MOVK, MOVN,

    // -- Extend --
    SXTB, SXTH, SXTW, UXTB, UXTH,

    // -- Bit Manipulation --
    CLZ, RBIT, REV, REV16, REV32,

    // -- Conditional Select --
    CSEL, CSINC, CSINV, CSNEG,

    // -- Load / Store (integer) --
    LDR, LDRW, LDRH, LDRB,
    LDRSW, LDRSH, LDRSB,
    STR, STRW, STRH, STRB,

    // -- Load / Store Pair --
    LDP, STP,

    // -- Load / Store (SIMD/FP scalar) --
    LDR_S, LDR_D, STR_S, STR_D,
    LDP_D, STP_D,

    // -- PC-Relative Addressing --
    ADRP, ADR,

    // -- Branch --
    B, BL, BR, BLR, RET,
    B_cond, CBZ, CBNZ, TBZ, TBNZ,

    // -- FP Arithmetic --
    FADD_S, FADD_D, FSUB_S, FSUB_D,
    FMUL_S, FMUL_D, FDIV_S, FDIV_D,
    FSQRT_S, FSQRT_D,
    FABS_S, FABS_D, FNEG_S, FNEG_D,
    FMIN_S, FMIN_D, FMAX_S, FMAX_D,
    FMOV,

    // -- FP Comparison --
    FCMP_S, FCMP_D,
    FCSEL_S, FCSEL_D,

    // -- FP Conversion --
    SCVTF_S, SCVTF_D, UCVTF_S, UCVTF_D,
    FCVTZS_S, FCVTZS_D, FCVTZU_S, FCVTZU_D,
    FCVT_S_TO_D, FCVT_D_TO_S,

    // -- System --
    NOP, SVC, DMB, DSB, ISB,
}

impl Aarch64Opcode {
    /// Encode this opcode as a `u32` for storage in `MachineInstr.opcode`.
    ///
    /// Declared `const` so that the value can be used in `const` initializers
    /// (e.g., ABI module opcode constants) as well as at runtime.
    pub const fn as_u32(self) -> u32 {
        match self {
            // Integer arithmetic — 0x1000
            Aarch64Opcode::ADD   => 0x1000,
            Aarch64Opcode::ADDS  => 0x1001,
            Aarch64Opcode::SUB   => 0x1002,
            Aarch64Opcode::SUBS  => 0x1003,
            Aarch64Opcode::MUL   => 0x1004,
            Aarch64Opcode::MADD  => 0x1005,
            Aarch64Opcode::MSUB  => 0x1006,
            Aarch64Opcode::SDIV  => 0x1007,
            Aarch64Opcode::UDIV  => 0x1008,
            Aarch64Opcode::SMULL => 0x1009,
            Aarch64Opcode::UMULL => 0x100A,
            Aarch64Opcode::SMULH => 0x100B,
            Aarch64Opcode::UMULH => 0x100C,

            // Logical — 0x1100
            Aarch64Opcode::AND  => 0x1100,
            Aarch64Opcode::ANDS => 0x1101,
            Aarch64Opcode::ORR  => 0x1102,
            Aarch64Opcode::ORN  => 0x1103,
            Aarch64Opcode::EOR  => 0x1104,
            Aarch64Opcode::EON  => 0x1105,
            Aarch64Opcode::BIC  => 0x1106,
            Aarch64Opcode::BICS => 0x1107,

            // Shift — 0x1200
            Aarch64Opcode::LSL => 0x1200,
            Aarch64Opcode::LSR => 0x1201,
            Aarch64Opcode::ASR => 0x1202,
            Aarch64Opcode::ROR => 0x1203,

            // Move — 0x1300
            Aarch64Opcode::MOV  => 0x1300,
            Aarch64Opcode::MVN  => 0x1301,
            Aarch64Opcode::MOVZ => 0x1302,
            Aarch64Opcode::MOVK => 0x1303,
            Aarch64Opcode::MOVN => 0x1304,

            // Extend — 0x1400
            Aarch64Opcode::SXTB => 0x1400,
            Aarch64Opcode::SXTH => 0x1401,
            Aarch64Opcode::SXTW => 0x1402,
            Aarch64Opcode::UXTB => 0x1403,
            Aarch64Opcode::UXTH => 0x1404,

            // Bit manipulation — 0x1500
            Aarch64Opcode::CLZ   => 0x1500,
            Aarch64Opcode::RBIT  => 0x1501,
            Aarch64Opcode::REV   => 0x1502,
            Aarch64Opcode::REV16 => 0x1503,
            Aarch64Opcode::REV32 => 0x1504,

            // Conditional select — 0x1600
            Aarch64Opcode::CSEL  => 0x1600,
            Aarch64Opcode::CSINC => 0x1601,
            Aarch64Opcode::CSINV => 0x1602,
            Aarch64Opcode::CSNEG => 0x1603,

            // Load / store (integer) — 0x2000
            Aarch64Opcode::LDR   => 0x2000,
            Aarch64Opcode::LDRW  => 0x2001,
            Aarch64Opcode::LDRH  => 0x2002,
            Aarch64Opcode::LDRB  => 0x2003,
            Aarch64Opcode::LDRSW => 0x2004,
            Aarch64Opcode::LDRSH => 0x2005,
            Aarch64Opcode::LDRSB => 0x2006,
            Aarch64Opcode::STR   => 0x2010,
            Aarch64Opcode::STRW  => 0x2011,
            Aarch64Opcode::STRH  => 0x2012,
            Aarch64Opcode::STRB  => 0x2013,

            // Pair — 0x2100
            Aarch64Opcode::LDP => 0x2100,
            Aarch64Opcode::STP => 0x2101,

            // FP load / store — 0x2200
            Aarch64Opcode::LDR_S => 0x2200,
            Aarch64Opcode::LDR_D => 0x2201,
            Aarch64Opcode::STR_S => 0x2210,
            Aarch64Opcode::STR_D => 0x2211,
            Aarch64Opcode::LDP_D => 0x2220,
            Aarch64Opcode::STP_D => 0x2221,

            // PC-relative addressing — 0x3000
            Aarch64Opcode::ADRP => 0x3000,
            Aarch64Opcode::ADR  => 0x3001,

            // Branch — 0x4000
            Aarch64Opcode::B      => 0x4000,
            Aarch64Opcode::BL     => 0x4001,
            Aarch64Opcode::BR     => 0x4002,
            Aarch64Opcode::BLR    => 0x4003,
            Aarch64Opcode::RET    => 0x4004,
            Aarch64Opcode::B_cond => 0x4010,
            Aarch64Opcode::CBZ    => 0x4020,
            Aarch64Opcode::CBNZ   => 0x4021,
            Aarch64Opcode::TBZ    => 0x4030,
            Aarch64Opcode::TBNZ   => 0x4031,

            // FP arithmetic — 0x5000
            Aarch64Opcode::FADD_S  => 0x5000,
            Aarch64Opcode::FADD_D  => 0x5001,
            Aarch64Opcode::FSUB_S  => 0x5002,
            Aarch64Opcode::FSUB_D  => 0x5003,
            Aarch64Opcode::FMUL_S  => 0x5004,
            Aarch64Opcode::FMUL_D  => 0x5005,
            Aarch64Opcode::FDIV_S  => 0x5006,
            Aarch64Opcode::FDIV_D  => 0x5007,
            Aarch64Opcode::FSQRT_S => 0x5008,
            Aarch64Opcode::FSQRT_D => 0x5009,
            Aarch64Opcode::FABS_S  => 0x500A,
            Aarch64Opcode::FABS_D  => 0x500B,
            Aarch64Opcode::FNEG_S  => 0x500C,
            Aarch64Opcode::FNEG_D  => 0x500D,
            Aarch64Opcode::FMIN_S  => 0x500E,
            Aarch64Opcode::FMIN_D  => 0x500F,
            Aarch64Opcode::FMAX_S  => 0x5010,
            Aarch64Opcode::FMAX_D  => 0x5011,
            Aarch64Opcode::FMOV    => 0x5020,

            // FP comparison — 0x5100
            Aarch64Opcode::FCMP_S  => 0x5100,
            Aarch64Opcode::FCMP_D  => 0x5101,
            Aarch64Opcode::FCSEL_S => 0x5110,
            Aarch64Opcode::FCSEL_D => 0x5111,

            // FP conversion — 0x5200
            Aarch64Opcode::SCVTF_S     => 0x5200,
            Aarch64Opcode::SCVTF_D     => 0x5201,
            Aarch64Opcode::UCVTF_S     => 0x5202,
            Aarch64Opcode::UCVTF_D     => 0x5203,
            Aarch64Opcode::FCVTZS_S    => 0x5210,
            Aarch64Opcode::FCVTZS_D    => 0x5211,
            Aarch64Opcode::FCVTZU_S    => 0x5212,
            Aarch64Opcode::FCVTZU_D    => 0x5213,
            Aarch64Opcode::FCVT_S_TO_D => 0x5220,
            Aarch64Opcode::FCVT_D_TO_S => 0x5221,

            // System — 0x6000
            Aarch64Opcode::NOP => 0x6000,
            Aarch64Opcode::SVC => 0x6001,
            Aarch64Opcode::DMB => 0x6010,
            Aarch64Opcode::DSB => 0x6011,
            Aarch64Opcode::ISB => 0x6012,
        }
    }

    /// Decode a `u32` back to an `Aarch64Opcode`, or `None` if unrecognised.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0x1000 => Some(Aarch64Opcode::ADD),
            0x1001 => Some(Aarch64Opcode::ADDS),
            0x1002 => Some(Aarch64Opcode::SUB),
            0x1003 => Some(Aarch64Opcode::SUBS),
            0x1004 => Some(Aarch64Opcode::MUL),
            0x1005 => Some(Aarch64Opcode::MADD),
            0x1006 => Some(Aarch64Opcode::MSUB),
            0x1007 => Some(Aarch64Opcode::SDIV),
            0x1008 => Some(Aarch64Opcode::UDIV),
            0x1009 => Some(Aarch64Opcode::SMULL),
            0x100A => Some(Aarch64Opcode::UMULL),
            0x100B => Some(Aarch64Opcode::SMULH),
            0x100C => Some(Aarch64Opcode::UMULH),
            0x1100 => Some(Aarch64Opcode::AND),
            0x1101 => Some(Aarch64Opcode::ANDS),
            0x1102 => Some(Aarch64Opcode::ORR),
            0x1103 => Some(Aarch64Opcode::ORN),
            0x1104 => Some(Aarch64Opcode::EOR),
            0x1105 => Some(Aarch64Opcode::EON),
            0x1106 => Some(Aarch64Opcode::BIC),
            0x1107 => Some(Aarch64Opcode::BICS),
            0x1200 => Some(Aarch64Opcode::LSL),
            0x1201 => Some(Aarch64Opcode::LSR),
            0x1202 => Some(Aarch64Opcode::ASR),
            0x1203 => Some(Aarch64Opcode::ROR),
            0x1300 => Some(Aarch64Opcode::MOV),
            0x1301 => Some(Aarch64Opcode::MVN),
            0x1302 => Some(Aarch64Opcode::MOVZ),
            0x1303 => Some(Aarch64Opcode::MOVK),
            0x1304 => Some(Aarch64Opcode::MOVN),
            0x1400 => Some(Aarch64Opcode::SXTB),
            0x1401 => Some(Aarch64Opcode::SXTH),
            0x1402 => Some(Aarch64Opcode::SXTW),
            0x1403 => Some(Aarch64Opcode::UXTB),
            0x1404 => Some(Aarch64Opcode::UXTH),
            0x1500 => Some(Aarch64Opcode::CLZ),
            0x1501 => Some(Aarch64Opcode::RBIT),
            0x1502 => Some(Aarch64Opcode::REV),
            0x1503 => Some(Aarch64Opcode::REV16),
            0x1504 => Some(Aarch64Opcode::REV32),
            0x1600 => Some(Aarch64Opcode::CSEL),
            0x1601 => Some(Aarch64Opcode::CSINC),
            0x1602 => Some(Aarch64Opcode::CSINV),
            0x1603 => Some(Aarch64Opcode::CSNEG),
            0x2000 => Some(Aarch64Opcode::LDR),
            0x2001 => Some(Aarch64Opcode::LDRW),
            0x2002 => Some(Aarch64Opcode::LDRH),
            0x2003 => Some(Aarch64Opcode::LDRB),
            0x2004 => Some(Aarch64Opcode::LDRSW),
            0x2005 => Some(Aarch64Opcode::LDRSH),
            0x2006 => Some(Aarch64Opcode::LDRSB),
            0x2010 => Some(Aarch64Opcode::STR),
            0x2011 => Some(Aarch64Opcode::STRW),
            0x2012 => Some(Aarch64Opcode::STRH),
            0x2013 => Some(Aarch64Opcode::STRB),
            0x2100 => Some(Aarch64Opcode::LDP),
            0x2101 => Some(Aarch64Opcode::STP),
            0x2200 => Some(Aarch64Opcode::LDR_S),
            0x2201 => Some(Aarch64Opcode::LDR_D),
            0x2210 => Some(Aarch64Opcode::STR_S),
            0x2211 => Some(Aarch64Opcode::STR_D),
            0x2220 => Some(Aarch64Opcode::LDP_D),
            0x2221 => Some(Aarch64Opcode::STP_D),
            0x3000 => Some(Aarch64Opcode::ADRP),
            0x3001 => Some(Aarch64Opcode::ADR),
            0x4000 => Some(Aarch64Opcode::B),
            0x4001 => Some(Aarch64Opcode::BL),
            0x4002 => Some(Aarch64Opcode::BR),
            0x4003 => Some(Aarch64Opcode::BLR),
            0x4004 => Some(Aarch64Opcode::RET),
            0x4010 => Some(Aarch64Opcode::B_cond),
            0x4020 => Some(Aarch64Opcode::CBZ),
            0x4021 => Some(Aarch64Opcode::CBNZ),
            0x4030 => Some(Aarch64Opcode::TBZ),
            0x4031 => Some(Aarch64Opcode::TBNZ),
            0x5000 => Some(Aarch64Opcode::FADD_S),
            0x5001 => Some(Aarch64Opcode::FADD_D),
            0x5002 => Some(Aarch64Opcode::FSUB_S),
            0x5003 => Some(Aarch64Opcode::FSUB_D),
            0x5004 => Some(Aarch64Opcode::FMUL_S),
            0x5005 => Some(Aarch64Opcode::FMUL_D),
            0x5006 => Some(Aarch64Opcode::FDIV_S),
            0x5007 => Some(Aarch64Opcode::FDIV_D),
            0x5008 => Some(Aarch64Opcode::FSQRT_S),
            0x5009 => Some(Aarch64Opcode::FSQRT_D),
            0x500A => Some(Aarch64Opcode::FABS_S),
            0x500B => Some(Aarch64Opcode::FABS_D),
            0x500C => Some(Aarch64Opcode::FNEG_S),
            0x500D => Some(Aarch64Opcode::FNEG_D),
            0x500E => Some(Aarch64Opcode::FMIN_S),
            0x500F => Some(Aarch64Opcode::FMIN_D),
            0x5010 => Some(Aarch64Opcode::FMAX_S),
            0x5011 => Some(Aarch64Opcode::FMAX_D),
            0x5020 => Some(Aarch64Opcode::FMOV),
            0x5100 => Some(Aarch64Opcode::FCMP_S),
            0x5101 => Some(Aarch64Opcode::FCMP_D),
            0x5110 => Some(Aarch64Opcode::FCSEL_S),
            0x5111 => Some(Aarch64Opcode::FCSEL_D),
            0x5200 => Some(Aarch64Opcode::SCVTF_S),
            0x5201 => Some(Aarch64Opcode::SCVTF_D),
            0x5202 => Some(Aarch64Opcode::UCVTF_S),
            0x5203 => Some(Aarch64Opcode::UCVTF_D),
            0x5210 => Some(Aarch64Opcode::FCVTZS_S),
            0x5211 => Some(Aarch64Opcode::FCVTZS_D),
            0x5212 => Some(Aarch64Opcode::FCVTZU_S),
            0x5213 => Some(Aarch64Opcode::FCVTZU_D),
            0x5220 => Some(Aarch64Opcode::FCVT_S_TO_D),
            0x5221 => Some(Aarch64Opcode::FCVT_D_TO_S),
            0x6000 => Some(Aarch64Opcode::NOP),
            0x6001 => Some(Aarch64Opcode::SVC),
            0x6010 => Some(Aarch64Opcode::DMB),
            0x6011 => Some(Aarch64Opcode::DSB),
            0x6012 => Some(Aarch64Opcode::ISB),
            _ => None,
        }
    }
}

// ===========================================================================
// Aarch64Condition — AArch64 condition codes
// ===========================================================================

/// AArch64 NZCV condition codes used for conditional branching and selection.
///
/// Each variant encodes a 4-bit condition code that the hardware evaluates
/// against the NZCV flags register (set by CMP, ADDS, SUBS, TST, FCMP, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Aarch64Condition {
    EQ, NE,
    HS, LO,
    MI, PL,
    VS, VC,
    HI, LS,
    GE, LT,
    GT, LE,
    AL, NV,
}

impl Aarch64Condition {
    /// Return the 4-bit binary encoding of this condition code.
    pub fn encoding(self) -> u8 {
        match self {
            Aarch64Condition::EQ => 0b0000,
            Aarch64Condition::NE => 0b0001,
            Aarch64Condition::HS => 0b0010,
            Aarch64Condition::LO => 0b0011,
            Aarch64Condition::MI => 0b0100,
            Aarch64Condition::PL => 0b0101,
            Aarch64Condition::VS => 0b0110,
            Aarch64Condition::VC => 0b0111,
            Aarch64Condition::HI => 0b1000,
            Aarch64Condition::LS => 0b1001,
            Aarch64Condition::GE => 0b1010,
            Aarch64Condition::LT => 0b1011,
            Aarch64Condition::GT => 0b1100,
            Aarch64Condition::LE => 0b1101,
            Aarch64Condition::AL => 0b1110,
            Aarch64Condition::NV => 0b1111,
        }
    }

    /// Return the inverted condition (flips the least-significant bit).
    pub fn invert(self) -> Aarch64Condition {
        match self {
            Aarch64Condition::EQ => Aarch64Condition::NE,
            Aarch64Condition::NE => Aarch64Condition::EQ,
            Aarch64Condition::HS => Aarch64Condition::LO,
            Aarch64Condition::LO => Aarch64Condition::HS,
            Aarch64Condition::MI => Aarch64Condition::PL,
            Aarch64Condition::PL => Aarch64Condition::MI,
            Aarch64Condition::VS => Aarch64Condition::VC,
            Aarch64Condition::VC => Aarch64Condition::VS,
            Aarch64Condition::HI => Aarch64Condition::LS,
            Aarch64Condition::LS => Aarch64Condition::HI,
            Aarch64Condition::GE => Aarch64Condition::LT,
            Aarch64Condition::LT => Aarch64Condition::GE,
            Aarch64Condition::GT => Aarch64Condition::LE,
            Aarch64Condition::LE => Aarch64Condition::GT,
            Aarch64Condition::AL => Aarch64Condition::NV,
            Aarch64Condition::NV => Aarch64Condition::AL,
        }
    }
}

// ===========================================================================
// ShiftType, ShiftedOperand — barrel shifter types
// ===========================================================================

/// Barrel shifter operation type for AArch64 data-processing instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShiftType {
    LSL,
    LSR,
    ASR,
    ROR,
}

/// A register operand with an attached barrel-shifter operation and amount.
#[derive(Debug, Clone, Copy)]
pub struct ShiftedOperand {
    pub reg: PhysReg,
    pub shift: ShiftType,
    pub amount: u8,
}

// ===========================================================================
// Free-standing helpers
// ===========================================================================

/// Maps an IR integer comparison to the AArch64 condition code.
fn compare_op_to_condition(op: &CompareOp) -> Aarch64Condition {
    match op {
        CompareOp::Equal              => Aarch64Condition::EQ,
        CompareOp::NotEqual           => Aarch64Condition::NE,
        CompareOp::SignedLess         => Aarch64Condition::LT,
        CompareOp::SignedLessEqual    => Aarch64Condition::LE,
        CompareOp::SignedGreater      => Aarch64Condition::GT,
        CompareOp::SignedGreaterEqual => Aarch64Condition::GE,
        CompareOp::UnsignedLess         => Aarch64Condition::LO,
        CompareOp::UnsignedLessEqual    => Aarch64Condition::LS,
        CompareOp::UnsignedGreater      => Aarch64Condition::HI,
        CompareOp::UnsignedGreaterEqual => Aarch64Condition::HS,
    }
}

/// Maps an IR float comparison to the AArch64 condition code.
fn float_compare_op_to_condition(op: &FloatCompareOp) -> Aarch64Condition {
    match op {
        FloatCompareOp::OrderedEqual        => Aarch64Condition::EQ,
        FloatCompareOp::OrderedNotEqual     => Aarch64Condition::NE,
        FloatCompareOp::OrderedLess         => Aarch64Condition::MI,
        FloatCompareOp::OrderedLessEqual    => Aarch64Condition::LS,
        FloatCompareOp::OrderedGreater      => Aarch64Condition::GT,
        FloatCompareOp::OrderedGreaterEqual => Aarch64Condition::GE,
        FloatCompareOp::Unordered           => Aarch64Condition::VS,
        FloatCompareOp::UnorderedEqual      => Aarch64Condition::EQ,
    }
}

/// Returns `true` when `val` fits in a 12-bit unsigned immediate (0..=4095).
fn fits_imm12(val: i64) -> bool {
    val >= 0 && val <= 4095
}

// ===========================================================================
// Aarch64InstructionSelector
// ===========================================================================

/// AArch64 instruction selector.
///
/// Transforms SSA-form IR instructions into sequences of `MachineInstr`
/// values annotated with `Aarch64Opcode` opcodes.
pub struct Aarch64InstructionSelector {
    value_map: HashMap<Value, MachineOperand>,
    next_vreg: u32,
    instructions: Vec<MachineInstr>,
    relocations: Vec<Relocation>,
    stack_offset: i32,
    label_counter: u32,
    block_labels: HashMap<u32, u32>,
    /// IR Values that represent function parameters — already mapped to
    /// ABI registers by `lower_params`.  Const instructions for these
    /// values must be skipped so the real parameter values are preserved.
    param_value_set: std::collections::HashSet<Value>,
}

impl Aarch64InstructionSelector {
    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    /// Create a new instruction selector with empty internal state.
    pub fn new() -> Self {
        Aarch64InstructionSelector {
            value_map: HashMap::new(),
            next_vreg: 64,
            instructions: Vec::new(),
            relocations: Vec::new(),
            stack_offset: 0,
            label_counter: 0,
            block_labels: HashMap::new(),
            param_value_set: std::collections::HashSet::new(),
        }
    }

    /// Build a mapping from virtual register IDs to IR Values by inverting
    /// the value_map (Value → MachineOperand).  Used for post-isel register
    /// assignment: vreg_id → Value → PhysReg (from regalloc).
    pub fn build_vreg_to_value_map(&self) -> HashMap<u32, Value> {
        let mut m = HashMap::new();
        for (&val, op) in &self.value_map {
            if let MachineOperand::Register(r) = op {
                if r.0 >= 64 {
                    m.insert(r.0 as u32, val);
                }
            }
        }
        m
    }

    // ---------------------------------------------------------------
    // Register / operand helpers
    // ---------------------------------------------------------------

    fn alloc_vreg(&mut self) -> PhysReg {
        let r = PhysReg(self.next_vreg as u16);
        self.next_vreg += 1;
        r
    }

    fn get_operand(&mut self, val: Value) -> MachineOperand {
        if let Some(op) = self.value_map.get(&val) {
            return op.clone();
        }
        let r = self.alloc_vreg();
        let op = MachineOperand::Register(r);
        self.value_map.insert(val, op.clone());
        op
    }

    fn get_reg(&mut self, val: Value) -> PhysReg {
        match self.get_operand(val) {
            MachineOperand::Register(r) => r,
            _ => {
                let r = self.alloc_vreg();
                self.value_map.insert(val, MachineOperand::Register(r));
                r
            }
        }
    }

    fn set_operand(&mut self, val: Value, op: MachineOperand) {
        self.value_map.insert(val, op);
    }

    // ---------------------------------------------------------------
    // Emission helpers
    // ---------------------------------------------------------------

    fn emit(&mut self, instr: MachineInstr) {
        self.instructions.push(instr);
    }

    fn emit_instr(&mut self, opcode: Aarch64Opcode, operands: Vec<MachineOperand>) {
        self.instructions.push(MachineInstr {
            opcode: opcode.as_u32(),
            operands,
            loc: None,
        });
    }

    fn emit_relocation(&mut self, reloc: Relocation) {
        self.relocations.push(reloc);
    }

    fn block_label(&mut self, block_id: u32) -> u32 {
        if let Some(&lbl) = self.block_labels.get(&block_id) {
            lbl
        } else {
            let lbl = self.label_counter;
            self.label_counter += 1;
            self.block_labels.insert(block_id, lbl);
            lbl
        }
    }

    // ---------------------------------------------------------------
    // select_function — top-level entry
    // ---------------------------------------------------------------

    /// Select AArch64 machine instructions for an entire IR function.
    ///
    /// Resets internal state, walks basic blocks in declaration order,
    /// and processes every instruction and terminator.
    pub fn select_function(
        &mut self,
        function: &Function,
    ) -> Result<Vec<MachineInstr>, CodeGenError> {
        // Reset per-function state.
        self.value_map.clear();
        self.instructions.clear();
        self.relocations.clear();
        self.next_vreg = 64;
        self.stack_offset = 0;
        self.label_counter = 0;
        self.block_labels.clear();
        self.param_value_set.clear();

        // Record which IR Values are function parameters so that
        // select_const skips the placeholder Const instructions.
        for &pv in &function.param_values {
            self.param_value_set.insert(pv);
        }

        // Pre-assign labels for every block.
        for blk in &function.blocks {
            let _ = self.block_label(blk.id.0);
        }

        // Map function parameters to ABI register locations.
        self.lower_params(function);

        // Walk blocks.
        for blk in &function.blocks {
            let lbl = self.block_label(blk.id.0);
            // Emit a NOP with a Label operand as a block marker.
            self.emit_instr(Aarch64Opcode::NOP, vec![MachineOperand::Label(lbl)]);

            // Phi nodes — ensure result values have register mappings.
            for phi in &blk.phi_nodes {
                let _ = self.get_operand(phi.result);
            }

            // Instructions.
            for inst in &blk.instructions {
                self.select_instruction(inst)?;
            }

            // Terminator.
            if let Some(ref term) = blk.terminator {
                self.select_terminator(term)?;
            }
        }

        Ok(std::mem::take(&mut self.instructions))
    }

    /// Return (and clear) the relocations accumulated during the last
    /// call to `select_function`.
    pub fn take_relocations(&mut self) -> Vec<Relocation> {
        std::mem::take(&mut self.relocations)
    }

    // ---------------------------------------------------------------
    // Parameter lowering (AAPCS64)
    // ---------------------------------------------------------------

    fn lower_params(&mut self, function: &Function) {
        let mut ngrn: usize = 0;
        let mut nsrn: usize = 0;

        for (i, (_name, ty)) in function.params.iter().enumerate() {
            let val = if i < function.param_values.len() {
                function.param_values[i]
            } else {
                Value(i as u32)
            };
            if ty.is_float() {
                if nsrn < 8 {
                    self.set_operand(val, MachineOperand::Register(FLOAT_ARG_REGS[nsrn]));
                    nsrn += 1;
                } else {
                    let off = 16 + (ngrn.max(8) - 8 + nsrn.max(8) - 8) as i32 * 8;
                    self.set_operand(val, MachineOperand::Memory { base: FP, offset: off });
                }
            } else if ngrn < 8 {
                self.set_operand(val, MachineOperand::Register(INT_ARG_REGS[ngrn]));
                ngrn += 1;
            } else {
                let off = 16 + (ngrn.max(8) - 8 + nsrn.max(8) - 8) as i32 * 8;
                self.set_operand(val, MachineOperand::Memory { base: FP, offset: off });
            }
        }
    }

    // ---------------------------------------------------------------
    // Instruction dispatch
    // ---------------------------------------------------------------

    fn select_instruction(&mut self, inst: &Instruction) -> Result<(), CodeGenError> {
        match inst {
            Instruction::Add { result, lhs, rhs, ty } =>
                self.select_add_sub(*result, *lhs, *rhs, ty, true),
            Instruction::Sub { result, lhs, rhs, ty } =>
                self.select_add_sub(*result, *lhs, *rhs, ty, false),
            Instruction::Mul { result, lhs, rhs, ty } =>
                self.select_mul(*result, *lhs, *rhs, ty),
            Instruction::Div { result, lhs, rhs, ty, is_signed } =>
                self.select_div(*result, *lhs, *rhs, ty, *is_signed),
            Instruction::Mod { result, lhs, rhs, ty, is_signed } =>
                self.select_mod(*result, *lhs, *rhs, ty, *is_signed),
            Instruction::And { result, lhs, rhs, ty } =>
                self.select_bitwise(*result, *lhs, *rhs, ty, Aarch64Opcode::AND),
            Instruction::Or { result, lhs, rhs, ty } =>
                self.select_bitwise(*result, *lhs, *rhs, ty, Aarch64Opcode::ORR),
            Instruction::Xor { result, lhs, rhs, ty } =>
                self.select_bitwise(*result, *lhs, *rhs, ty, Aarch64Opcode::EOR),
            Instruction::Shl { result, lhs, rhs, ty } =>
                self.select_shift(*result, *lhs, *rhs, ty, Aarch64Opcode::LSL),
            Instruction::Shr { result, lhs, rhs, ty, is_arithmetic } => {
                let op = if *is_arithmetic { Aarch64Opcode::ASR } else { Aarch64Opcode::LSR };
                self.select_shift(*result, *lhs, *rhs, ty, op)
            }
            Instruction::ICmp { result, op, lhs, rhs, ty } =>
                self.select_icmp(*result, op, *lhs, *rhs, ty),
            Instruction::FCmp { result, op, lhs, rhs, ty } =>
                self.select_fcmp(*result, op, *lhs, *rhs, ty),
            Instruction::Alloca { result, ty, count } =>
                self.select_alloca(*result, ty, count),
            Instruction::Load { result, ty, ptr } =>
                self.select_load(*result, ty, *ptr),
            Instruction::Store { value, ptr, .. } =>
                self.select_store(*value, *ptr),
            Instruction::GetElementPtr { result, base_ty, ptr, indices, .. } =>
                self.select_gep(*result, base_ty, *ptr, indices),
            Instruction::Call { result, callee, args, return_ty } =>
                self.select_call(result, callee, args, return_ty),
            Instruction::Cast { result, op, value, from_ty, to_ty } =>
                self.select_cast(*result, op, *value, from_ty, to_ty),
            Instruction::BitCast { result, value, from_ty, to_ty } =>
                self.select_bitcast(*result, *value, from_ty, to_ty),
            Instruction::Select { result, condition, true_val, false_val, ty } =>
                self.select_select(*result, *condition, *true_val, *false_val, ty),
            Instruction::Const { result, value } =>
                self.select_const(*result, value),
            Instruction::Phi { result, .. } => {
                let _ = self.get_operand(*result);
                Ok(())
            }
            Instruction::Copy { result, source, .. } => {
                let src = self.get_operand(*source);
                self.set_operand(*result, src);
                Ok(())
            }
            Instruction::Nop => Ok(()),
        }
    }

    // ---------------------------------------------------------------
    // Arithmetic
    // ---------------------------------------------------------------

    fn select_add_sub(
        &mut self, result: Value, lhs: Value, rhs: Value,
        ty: &IrType, is_add: bool,
    ) -> Result<(), CodeGenError> {
        if ty.is_float() {
            let l = self.get_reg(lhs);
            let r = self.get_reg(rhs);
            let d = self.alloc_vreg();
            self.set_operand(result, MachineOperand::Register(d));
            let op = match (is_add, ty) {
                (true,  IrType::F32) => Aarch64Opcode::FADD_S,
                (true,  IrType::F64) => Aarch64Opcode::FADD_D,
                (false, IrType::F32) => Aarch64Opcode::FSUB_S,
                (false, IrType::F64) => Aarch64Opcode::FSUB_D,
                _ => return Err(CodeGenError::UnsupportedInstruction(
                    format!("float add/sub with {:?}", ty),
                )),
            };
            self.emit_instr(op, vec![
                MachineOperand::Register(d),
                MachineOperand::Register(l),
                MachineOperand::Register(r),
            ]);
        } else {
            let l = self.get_reg(lhs);
            let d = self.alloc_vreg();
            self.set_operand(result, MachineOperand::Register(d));
            let base = if is_add { Aarch64Opcode::ADD } else { Aarch64Opcode::SUB };

            let rhs_op = self.get_operand(rhs);
            match rhs_op {
                MachineOperand::Immediate(imm) if fits_imm12(imm) => {
                    self.emit_instr(base, vec![
                        MachineOperand::Register(d),
                        MachineOperand::Register(l),
                        MachineOperand::Immediate(imm),
                    ]);
                }
                _ => {
                    let r = self.get_reg(rhs);
                    self.emit_instr(base, vec![
                        MachineOperand::Register(d),
                        MachineOperand::Register(l),
                        MachineOperand::Register(r),
                    ]);
                }
            }
        }
        Ok(())
    }

    fn select_mul(
        &mut self, result: Value, lhs: Value, rhs: Value, ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let r = self.get_reg(rhs);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        let op = match ty {
            IrType::F32 => Aarch64Opcode::FMUL_S,
            IrType::F64 => Aarch64Opcode::FMUL_D,
            _ => Aarch64Opcode::MUL,
        };
        self.emit_instr(op, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(l),
            MachineOperand::Register(r),
        ]);
        Ok(())
    }

    fn select_div(
        &mut self, result: Value, lhs: Value, rhs: Value,
        ty: &IrType, is_signed: bool,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let r = self.get_reg(rhs);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        let op = match ty {
            IrType::F32 => Aarch64Opcode::FDIV_S,
            IrType::F64 => Aarch64Opcode::FDIV_D,
            _ if is_signed => Aarch64Opcode::SDIV,
            _ => Aarch64Opcode::UDIV,
        };
        self.emit_instr(op, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(l),
            MachineOperand::Register(r),
        ]);
        Ok(())
    }

    /// MOD via SDIV/UDIV + MSUB (AArch64 has no hardware remainder).
    fn select_mod(
        &mut self, result: Value, lhs: Value, rhs: Value,
        _ty: &IrType, is_signed: bool,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let r = self.get_reg(rhs);
        let tmp = self.alloc_vreg();
        let div = if is_signed { Aarch64Opcode::SDIV } else { Aarch64Opcode::UDIV };
        self.emit_instr(div, vec![
            MachineOperand::Register(tmp),
            MachineOperand::Register(l),
            MachineOperand::Register(r),
        ]);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        // MSUB Xd, Xn, Xm, Xa  →  Xa − Xn*Xm
        self.emit_instr(Aarch64Opcode::MSUB, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(tmp),
            MachineOperand::Register(r),
            MachineOperand::Register(l),
        ]);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Bitwise & shift
    // ---------------------------------------------------------------

    fn select_bitwise(
        &mut self, result: Value, lhs: Value, rhs: Value,
        _ty: &IrType, opcode: Aarch64Opcode,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let r = self.get_reg(rhs);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        self.emit_instr(opcode, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(l),
            MachineOperand::Register(r),
        ]);
        Ok(())
    }

    fn select_shift(
        &mut self, result: Value, lhs: Value, rhs: Value,
        _ty: &IrType, opcode: Aarch64Opcode,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        let rhs_op = self.get_operand(rhs);
        match rhs_op {
            MachineOperand::Immediate(imm) if imm >= 0 && imm < 64 => {
                self.emit_instr(opcode, vec![
                    MachineOperand::Register(d),
                    MachineOperand::Register(l),
                    MachineOperand::Immediate(imm),
                ]);
            }
            _ => {
                let r = self.get_reg(rhs);
                self.emit_instr(opcode, vec![
                    MachineOperand::Register(d),
                    MachineOperand::Register(l),
                    MachineOperand::Register(r),
                ]);
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------

    fn select_icmp(
        &mut self, result: Value, op: &CompareOp,
        lhs: Value, rhs: Value, _ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let r = self.get_reg(rhs);
        let xzr = PhysReg(31);
        // CMP → SUBS XZR, Xn, Xm
        self.emit_instr(Aarch64Opcode::SUBS, vec![
            MachineOperand::Register(xzr),
            MachineOperand::Register(l),
            MachineOperand::Register(r),
        ]);
        // Materialise boolean via CSINC Xd, XZR, XZR, inv(cond).
        let cond = compare_op_to_condition(op);
        let inv = cond.invert();
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        self.emit_instr(Aarch64Opcode::CSINC, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(xzr),
            MachineOperand::Register(xzr),
            MachineOperand::Immediate(inv.encoding() as i64),
        ]);
        Ok(())
    }

    fn select_fcmp(
        &mut self, result: Value, op: &FloatCompareOp,
        lhs: Value, rhs: Value, ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let l = self.get_reg(lhs);
        let r = self.get_reg(rhs);
        let cmp_op = match ty {
            IrType::F32 => Aarch64Opcode::FCMP_S,
            IrType::F64 => Aarch64Opcode::FCMP_D,
            _ => return Err(CodeGenError::UnsupportedInstruction(
                format!("fcmp on {:?}", ty),
            )),
        };
        self.emit_instr(cmp_op, vec![
            MachineOperand::Register(l),
            MachineOperand::Register(r),
        ]);
        let cond = float_compare_op_to_condition(op);
        let inv = cond.invert();
        let xzr = PhysReg(31);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        self.emit_instr(Aarch64Opcode::CSINC, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(xzr),
            MachineOperand::Register(xzr),
            MachineOperand::Immediate(inv.encoding() as i64),
        ]);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Memory operations
    // ---------------------------------------------------------------

    fn select_alloca(
        &mut self, result: Value, ty: &IrType, _count: &Option<Value>,
    ) -> Result<(), CodeGenError> {
        let sz = match ty {
            IrType::I1 | IrType::I8 => 1i32,
            IrType::I16 => 2,
            IrType::I32 | IrType::F32 => 4,
            IrType::I64 | IrType::F64 | IrType::Pointer(_) => 8,
            _ => 8,
        };
        let aligned = (sz + 7) & !7;
        self.stack_offset -= aligned;
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        self.emit_instr(Aarch64Opcode::SUB, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(FP),
            MachineOperand::Immediate((-self.stack_offset) as i64),
        ]);
        Ok(())
    }

    fn select_load(
        &mut self, result: Value, ty: &IrType, ptr: Value,
    ) -> Result<(), CodeGenError> {
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        let (base, off) = self.resolve_addr(ptr);
        let op = match ty {
            IrType::I1 | IrType::I8 => Aarch64Opcode::LDRB,
            IrType::I16 => Aarch64Opcode::LDRH,
            IrType::I32 => Aarch64Opcode::LDRW,
            IrType::I64 | IrType::Pointer(_) => Aarch64Opcode::LDR,
            IrType::F32 => Aarch64Opcode::LDR_S,
            IrType::F64 => Aarch64Opcode::LDR_D,
            _ => Aarch64Opcode::LDR,
        };
        self.emit_instr(op, vec![
            MachineOperand::Register(d),
            MachineOperand::Memory { base, offset: off },
        ]);
        Ok(())
    }

    fn select_store(
        &mut self, value: Value, ptr: Value,
    ) -> Result<(), CodeGenError> {
        let v = self.get_reg(value);
        let (base, off) = self.resolve_addr(ptr);
        let op = if is_fp_reg(v) { Aarch64Opcode::STR_D } else { Aarch64Opcode::STR };
        self.emit_instr(op, vec![
            MachineOperand::Register(v),
            MachineOperand::Memory { base, offset: off },
        ]);
        Ok(())
    }

    fn select_gep(
        &mut self, result: Value, base_ty: &IrType,
        ptr: Value, indices: &[Value],
    ) -> Result<(), CodeGenError> {
        let mut cur = self.get_reg(ptr);

        for idx_val in indices {
            let elem_sz: i64 = match base_ty {
                IrType::I1 | IrType::I8 => 1,
                IrType::I16 => 2,
                IrType::I32 | IrType::F32 => 4,
                IrType::I64 | IrType::F64 | IrType::Pointer(_) => 8,
                IrType::Array { element, .. } => match element.as_ref() {
                    IrType::I1 | IrType::I8 => 1,
                    IrType::I16 => 2,
                    IrType::I32 | IrType::F32 => 4,
                    _ => 8,
                },
                _ => 8,
            };

            let new = self.alloc_vreg();
            let idx_op = self.get_operand(*idx_val);

            match idx_op {
                MachineOperand::Immediate(imm) => {
                    let byte_off = imm * elem_sz;
                    if fits_imm12(byte_off) {
                        self.emit_instr(Aarch64Opcode::ADD, vec![
                            MachineOperand::Register(new),
                            MachineOperand::Register(cur),
                            MachineOperand::Immediate(byte_off),
                        ]);
                    } else {
                        let tmp = self.alloc_vreg();
                        self.materialize_constant(byte_off as u64, tmp);
                        self.emit_instr(Aarch64Opcode::ADD, vec![
                            MachineOperand::Register(new),
                            MachineOperand::Register(cur),
                            MachineOperand::Register(tmp),
                        ]);
                    }
                }
                _ => {
                    let ir = self.get_reg(*idx_val);
                    let shift = match elem_sz {
                        1 => 0u8, 2 => 1, 4 => 2, 8 => 3, 16 => 4,
                        _ => {
                            let sr = self.alloc_vreg();
                            self.materialize_constant(elem_sz as u64, sr);
                            let pr = self.alloc_vreg();
                            self.emit_instr(Aarch64Opcode::MUL, vec![
                                MachineOperand::Register(pr),
                                MachineOperand::Register(ir),
                                MachineOperand::Register(sr),
                            ]);
                            self.emit_instr(Aarch64Opcode::ADD, vec![
                                MachineOperand::Register(new),
                                MachineOperand::Register(cur),
                                MachineOperand::Register(pr),
                            ]);
                            cur = new;
                            continue;
                        }
                    };
                    if shift == 0 {
                        self.emit_instr(Aarch64Opcode::ADD, vec![
                            MachineOperand::Register(new),
                            MachineOperand::Register(cur),
                            MachineOperand::Register(ir),
                        ]);
                    } else {
                        // Barrel shifter: ADD Xd, Xn, Xm, LSL #shift
                        self.emit_instr(Aarch64Opcode::ADD, vec![
                            MachineOperand::Register(new),
                            MachineOperand::Register(cur),
                            MachineOperand::Register(ir),
                            MachineOperand::Immediate(shift as i64),
                        ]);
                    }
                }
            }
            cur = new;
        }
        self.set_operand(result, MachineOperand::Register(cur));
        Ok(())
    }

    fn resolve_addr(&mut self, ptr: Value) -> (PhysReg, i32) {
        match self.get_operand(ptr) {
            MachineOperand::Memory { base, offset } => (base, offset),
            MachineOperand::Register(r) => (r, 0),
            _ => {
                let r = self.get_reg(ptr);
                (r, 0)
            }
        }
    }

    // ---------------------------------------------------------------
    // Function calls
    // ---------------------------------------------------------------

    fn select_call(
        &mut self, result: &Option<Value>, callee: &Callee,
        args: &[Value], return_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        // Build a PhysReg-only map from our MachineOperand value_map,
        // since generate_call_sequence expects HashMap<Value, PhysReg>.
        let mut phys_map: HashMap<Value, PhysReg> = HashMap::new();
        for (&v, op) in &self.value_map {
            if let MachineOperand::Register(r) = op {
                phys_map.insert(v, *r);
            }
        }
        // Ensure all arg values are present.
        for a in args {
            if !phys_map.contains_key(a) {
                let r = self.get_reg(*a);
                phys_map.insert(*a, r);
            }
        }
        let seq = generate_call_sequence(callee, args, return_ty, &phys_map);
        for i in seq {
            self.emit(i);
        }
        if let Some(res) = result {
            if return_ty.is_float() {
                self.set_operand(*res, MachineOperand::Register(V0));
            } else if !return_ty.is_void() {
                self.set_operand(*res, MachineOperand::Register(X0));
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Type conversions
    // ---------------------------------------------------------------

    fn select_cast(
        &mut self, result: Value, op: &CastOp, value: Value,
        from_ty: &IrType, to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let s = self.get_reg(value);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));

        match op {
            CastOp::SExt => {
                match from_ty.integer_bit_width() {
                    Some(8)  => self.emit_instr(Aarch64Opcode::SXTB, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                    Some(16) => self.emit_instr(Aarch64Opcode::SXTH, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                    Some(32) => self.emit_instr(Aarch64Opcode::SXTW, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                    _ => self.emit_instr(Aarch64Opcode::MOV, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                }
            }
            CastOp::ZExt => {
                match from_ty.integer_bit_width() {
                    Some(1) => self.emit_instr(Aarch64Opcode::AND, vec![
                        MachineOperand::Register(d),
                        MachineOperand::Register(s),
                        MachineOperand::Immediate(1)]),
                    Some(8)  => self.emit_instr(Aarch64Opcode::UXTB, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                    Some(16) => self.emit_instr(Aarch64Opcode::UXTH, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                    _ => self.emit_instr(Aarch64Opcode::MOV, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                }
            }
            CastOp::Trunc => {
                match to_ty.integer_bit_width() {
                    Some(8) => self.emit_instr(Aarch64Opcode::AND, vec![
                        MachineOperand::Register(d),
                        MachineOperand::Register(s),
                        MachineOperand::Immediate(0xFF)]),
                    Some(16) => self.emit_instr(Aarch64Opcode::AND, vec![
                        MachineOperand::Register(d),
                        MachineOperand::Register(s),
                        MachineOperand::Immediate(0xFFFF)]),
                    _ => self.emit_instr(Aarch64Opcode::MOV, vec![
                        MachineOperand::Register(d), MachineOperand::Register(s)]),
                }
            }
            CastOp::FPToSI => {
                let c = match from_ty {
                    IrType::F32 => Aarch64Opcode::FCVTZS_S,
                    _           => Aarch64Opcode::FCVTZS_D,
                };
                self.emit_instr(c, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
            CastOp::FPToUI => {
                let c = match from_ty {
                    IrType::F32 => Aarch64Opcode::FCVTZU_S,
                    _           => Aarch64Opcode::FCVTZU_D,
                };
                self.emit_instr(c, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
            CastOp::SIToFP => {
                let c = match to_ty {
                    IrType::F32 => Aarch64Opcode::SCVTF_S,
                    _           => Aarch64Opcode::SCVTF_D,
                };
                self.emit_instr(c, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
            CastOp::UIToFP => {
                let c = match to_ty {
                    IrType::F32 => Aarch64Opcode::UCVTF_S,
                    _           => Aarch64Opcode::UCVTF_D,
                };
                self.emit_instr(c, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
            CastOp::FPTrunc => {
                self.emit_instr(Aarch64Opcode::FCVT_D_TO_S, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
            CastOp::FPExt => {
                self.emit_instr(Aarch64Opcode::FCVT_S_TO_D, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
            CastOp::PtrToInt | CastOp::IntToPtr => {
                self.emit_instr(Aarch64Opcode::MOV, vec![
                    MachineOperand::Register(d), MachineOperand::Register(s)]);
            }
        }
        Ok(())
    }

    fn select_bitcast(
        &mut self, result: Value, value: Value,
        from_ty: &IrType, to_ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let s = self.get_reg(value);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        if from_ty.is_float() != to_ty.is_float() {
            self.emit_instr(Aarch64Opcode::FMOV, vec![
                MachineOperand::Register(d), MachineOperand::Register(s)]);
        } else {
            self.emit_instr(Aarch64Opcode::MOV, vec![
                MachineOperand::Register(d), MachineOperand::Register(s)]);
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Select (conditional)
    // ---------------------------------------------------------------

    fn select_select(
        &mut self, result: Value, cond: Value,
        tv: Value, fv: Value, ty: &IrType,
    ) -> Result<(), CodeGenError> {
        let cr = self.get_reg(cond);
        let tr = self.get_reg(tv);
        let fr = self.get_reg(fv);
        let d = self.alloc_vreg();
        self.set_operand(result, MachineOperand::Register(d));
        let xzr = PhysReg(31);
        // CMP cond, #0
        self.emit_instr(Aarch64Opcode::SUBS, vec![
            MachineOperand::Register(xzr),
            MachineOperand::Register(cr),
            MachineOperand::Register(xzr),
        ]);
        let sel = if ty.is_float() {
            match ty { IrType::F32 => Aarch64Opcode::FCSEL_S, _ => Aarch64Opcode::FCSEL_D }
        } else {
            Aarch64Opcode::CSEL
        };
        self.emit_instr(sel, vec![
            MachineOperand::Register(d),
            MachineOperand::Register(tr),
            MachineOperand::Register(fr),
            MachineOperand::Immediate(Aarch64Condition::NE.encoding() as i64),
        ]);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------

    fn select_const(
        &mut self, result: Value, constant: &Constant,
    ) -> Result<(), CodeGenError> {
        // Skip placeholder Const instructions for function parameters —
        // lower_params already mapped them to the correct ABI registers.
        if self.param_value_set.contains(&result) {
            return Ok(());
        }
        match constant {
            Constant::Integer { value, .. } => {
                let d = self.alloc_vreg();
                self.set_operand(result, MachineOperand::Register(d));
                self.materialize_constant(*value as u64, d);
            }
            Constant::Float { value, ty } => {
                let d = self.alloc_vreg();
                self.set_operand(result, MachineOperand::Register(d));
                let bits = match ty {
                    IrType::F32 => (*value as f32).to_bits() as u64,
                    _           => value.to_bits(),
                };
                let tmp = self.alloc_vreg();
                self.materialize_constant(bits, tmp);
                self.emit_instr(Aarch64Opcode::FMOV, vec![
                    MachineOperand::Register(d),
                    MachineOperand::Register(tmp),
                ]);
            }
            Constant::Bool(b) => {
                let d = self.alloc_vreg();
                self.set_operand(result, MachineOperand::Register(d));
                self.materialize_constant(if *b { 1 } else { 0 }, d);
            }
            Constant::Null(_) | Constant::Undef(_) | Constant::ZeroInit(_) => {
                let d = self.alloc_vreg();
                self.set_operand(result, MachineOperand::Register(d));
                self.materialize_constant(0, d);
            }
            Constant::String(_) => {
                let d = self.alloc_vreg();
                self.set_operand(result, MachineOperand::Register(d));
                self.materialize_constant(0, d);
            }
            Constant::GlobalRef(name) => {
                let d = self.alloc_vreg();
                self.set_operand(result, MachineOperand::Register(d));
                // ADRP Xd, sym@PAGE
                self.emit_instr(Aarch64Opcode::ADRP, vec![
                    MachineOperand::Register(d),
                    MachineOperand::Symbol(name.clone()),
                ]);
                self.emit_relocation(Relocation {
                    offset: 0,
                    symbol: name.clone(),
                    reloc_type: RelocationType::Aarch64_ADR_PREL_PG_HI21,
                    addend: 0,
                    section_index: 0,
                });
                // ADD Xd, Xd, sym@PAGEOFF
                self.emit_instr(Aarch64Opcode::ADD, vec![
                    MachineOperand::Register(d),
                    MachineOperand::Register(d),
                    MachineOperand::Symbol(name.clone()),
                ]);
                self.emit_relocation(Relocation {
                    offset: 0,
                    symbol: name.clone(),
                    reloc_type: RelocationType::Aarch64_ADD_ABS_LO12_NC,
                    addend: 0,
                    section_index: 0,
                });
            }
        }
        Ok(())
    }

    /// Materialise a 64-bit constant into `dest` using MOVZ/MOVK sequences.
    ///
    /// Optimisations applied:
    /// - Zero halfwords are skipped (fewer instructions).
    /// - MOVN is preferred when the inverted value has fewer non-zero halfwords.
    /// - A single MOV from XZR is used for zero.
    fn materialize_constant(&mut self, value: u64, dest: PhysReg) {
        if value == 0 {
            let xzr = PhysReg(31);
            self.emit_instr(Aarch64Opcode::MOV, vec![
                MachineOperand::Register(dest),
                MachineOperand::Register(xzr),
            ]);
            return;
        }

        let hw: [u16; 4] = [
            (value & 0xFFFF) as u16,
            ((value >> 16) & 0xFFFF) as u16,
            ((value >> 32) & 0xFFFF) as u16,
            ((value >> 48) & 0xFFFF) as u16,
        ];
        let nz = hw.iter().filter(|&&h| h != 0).count();

        let inv = !value;
        let ihw: [u16; 4] = [
            (inv & 0xFFFF) as u16,
            ((inv >> 16) & 0xFFFF) as u16,
            ((inv >> 32) & 0xFFFF) as u16,
            ((inv >> 48) & 0xFFFF) as u16,
        ];
        let inz = ihw.iter().filter(|&&h| h != 0).count();

        if inz < nz {
            // MOVN path — fewer patches after bit inversion.
            let mut first = true;
            for (i, &h) in ihw.iter().enumerate() {
                if first && h != 0 {
                    self.emit_instr(Aarch64Opcode::MOVN, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(h as i64),
                        MachineOperand::Immediate(i as i64 * 16),
                    ]);
                    first = false;
                } else if !first && hw[i] != 0xFFFF {
                    self.emit_instr(Aarch64Opcode::MOVK, vec![
                        MachineOperand::Register(dest),
                        MachineOperand::Immediate(hw[i] as i64),
                        MachineOperand::Immediate(i as i64 * 16),
                    ]);
                }
            }
            if first {
                // All inverted halfwords zero — value is all-ones.
                self.emit_instr(Aarch64Opcode::MOVN, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Immediate(0),
                    MachineOperand::Immediate(0),
                ]);
            }
        } else {
            // MOVZ + MOVK path.
            let mut first = true;
            for (i, &h) in hw.iter().enumerate() {
                if h != 0 || (first && i == 3) {
                    if first {
                        self.emit_instr(Aarch64Opcode::MOVZ, vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Immediate(h as i64),
                            MachineOperand::Immediate(i as i64 * 16),
                        ]);
                        first = false;
                    } else {
                        self.emit_instr(Aarch64Opcode::MOVK, vec![
                            MachineOperand::Register(dest),
                            MachineOperand::Immediate(h as i64),
                            MachineOperand::Immediate(i as i64 * 16),
                        ]);
                    }
                }
            }
            if first {
                let xzr = PhysReg(31);
                self.emit_instr(Aarch64Opcode::MOV, vec![
                    MachineOperand::Register(dest),
                    MachineOperand::Register(xzr),
                ]);
            }
        }
    }

    // ---------------------------------------------------------------
    // Terminators
    // ---------------------------------------------------------------

    fn select_terminator(&mut self, term: &Terminator) -> Result<(), CodeGenError> {
        match term {
            Terminator::Branch { target } => {
                let lbl = self.block_label(target.0);
                self.emit_instr(Aarch64Opcode::B, vec![MachineOperand::Label(lbl)]);
            }
            Terminator::CondBranch { condition, true_block, false_block } => {
                let cr = self.get_reg(*condition);
                let tl = self.block_label(true_block.0);
                let fl = self.block_label(false_block.0);
                // CBNZ Xn, true_label  (branch if non-zero)
                self.emit_instr(Aarch64Opcode::CBNZ, vec![
                    MachineOperand::Register(cr),
                    MachineOperand::Label(tl),
                ]);
                self.emit_instr(Aarch64Opcode::B, vec![MachineOperand::Label(fl)]);
            }
            Terminator::Return { value } => {
                if let Some(v) = value {
                    let vr = self.get_reg(*v);
                    if is_fp_reg(vr) {
                        if vr != V0 {
                            self.emit_instr(Aarch64Opcode::FMOV, vec![
                                MachineOperand::Register(V0),
                                MachineOperand::Register(vr),
                            ]);
                        }
                    } else if vr != X0 {
                        self.emit_instr(Aarch64Opcode::MOV, vec![
                            MachineOperand::Register(X0),
                            MachineOperand::Register(vr),
                        ]);
                    }
                }
                self.emit_instr(Aarch64Opcode::RET, vec![MachineOperand::Register(LR)]);
            }
            Terminator::Switch { value, default, cases } => {
                let vr = self.get_reg(*value);
                let dl = self.block_label(default.0);
                for (cv, tgt) in cases {
                    let tl = self.block_label(tgt.0);
                    let cr = self.alloc_vreg();
                    self.materialize_constant(*cv as u64, cr);
                    let xzr = PhysReg(31);
                    self.emit_instr(Aarch64Opcode::SUBS, vec![
                        MachineOperand::Register(xzr),
                        MachineOperand::Register(vr),
                        MachineOperand::Register(cr),
                    ]);
                    self.emit_instr(Aarch64Opcode::B_cond, vec![
                        MachineOperand::Label(tl),
                        MachineOperand::Immediate(Aarch64Condition::EQ.encoding() as i64),
                    ]);
                }
                self.emit_instr(Aarch64Opcode::B, vec![MachineOperand::Label(dl)]);
            }
            Terminator::Unreachable => {
                self.emit_instr(Aarch64Opcode::NOP, vec![]);
            }
        }
        Ok(())
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, BlockId, PhiNode};

    fn make_function(name: &str, blocks: Vec<BasicBlock>) -> Function {
        let entry = if blocks.is_empty() { BlockId(0) } else { blocks[0].id };
        Function {
            name: name.to_string(),
            return_type: IrType::I64,
            params: vec![],
            param_values: Vec::new(),
            blocks,
            entry_block: entry,
            is_definition: true,
is_static: false,
is_weak: false,
        }
    }

    fn make_block(id: u32, insts: Vec<Instruction>, term: Option<Terminator>) -> BasicBlock {
        let mut bb = BasicBlock::new(BlockId(id), format!("bb{}", id));
        bb.instructions = insts;
        bb.terminator = term;
        bb
    }

    // -- Opcode round-trip -------------------------------------------------

    #[test]
    fn test_opcode_roundtrip() {
        let ops = [
            Aarch64Opcode::ADD, Aarch64Opcode::SUB, Aarch64Opcode::MUL,
            Aarch64Opcode::SDIV, Aarch64Opcode::AND, Aarch64Opcode::ORR,
            Aarch64Opcode::EOR, Aarch64Opcode::LSL, Aarch64Opcode::MOV,
            Aarch64Opcode::MOVZ, Aarch64Opcode::MOVK, Aarch64Opcode::CSEL,
            Aarch64Opcode::LDR, Aarch64Opcode::STR, Aarch64Opcode::B,
            Aarch64Opcode::BL, Aarch64Opcode::RET, Aarch64Opcode::FADD_D,
            Aarch64Opcode::FCVTZS_D, Aarch64Opcode::NOP,
        ];
        for o in &ops {
            assert_eq!(Aarch64Opcode::from_u32(o.as_u32()), Some(*o));
        }
    }

    // -- Condition encoding ------------------------------------------------

    #[test]
    fn test_condition_encoding() {
        assert_eq!(Aarch64Condition::EQ.encoding(), 0b0000);
        assert_eq!(Aarch64Condition::NE.encoding(), 0b0001);
        assert_eq!(Aarch64Condition::LT.encoding(), 0b1011);
        assert_eq!(Aarch64Condition::GE.encoding(), 0b1010);
        assert_eq!(Aarch64Condition::AL.encoding(), 0b1110);
    }

    #[test]
    fn test_condition_invert() {
        assert_eq!(Aarch64Condition::EQ.invert(), Aarch64Condition::NE);
        assert_eq!(Aarch64Condition::LT.invert(), Aarch64Condition::GE);
        assert_eq!(Aarch64Condition::HI.invert(), Aarch64Condition::LS);
        assert_eq!(Aarch64Condition::GT.invert(), Aarch64Condition::LE);
    }

    // -- Constant materialisation ------------------------------------------

    #[test]
    fn test_materialize_zero() {
        let mut s = Aarch64InstructionSelector::new();
        s.materialize_constant(0, PhysReg(0));
        assert_eq!(s.instructions.len(), 1);
        assert_eq!(s.instructions[0].opcode, Aarch64Opcode::MOV.as_u32());
    }

    #[test]
    fn test_materialize_16bit() {
        let mut s = Aarch64InstructionSelector::new();
        s.materialize_constant(42, PhysReg(0));
        assert_eq!(s.instructions.len(), 1);
        assert_eq!(s.instructions[0].opcode, Aarch64Opcode::MOVZ.as_u32());
    }

    #[test]
    fn test_materialize_32bit() {
        let mut s = Aarch64InstructionSelector::new();
        s.materialize_constant(0x1_0042, PhysReg(0));
        assert_eq!(s.instructions.len(), 2);
        assert_eq!(s.instructions[0].opcode, Aarch64Opcode::MOVZ.as_u32());
        assert_eq!(s.instructions[1].opcode, Aarch64Opcode::MOVK.as_u32());
    }

    #[test]
    fn test_materialize_64bit_all_nonzero() {
        let mut s = Aarch64InstructionSelector::new();
        s.materialize_constant(0x1111_2222_3333_4444, PhysReg(0));
        assert_eq!(s.instructions.len(), 4);
        assert_eq!(s.instructions[0].opcode, Aarch64Opcode::MOVZ.as_u32());
        for i in 1..4 {
            assert_eq!(s.instructions[i].opcode, Aarch64Opcode::MOVK.as_u32());
        }
    }

    #[test]
    fn test_materialize_skip_zero_hw() {
        let mut s = Aarch64InstructionSelector::new();
        // hw0=1, hw1=0, hw2=1, hw3=0
        s.materialize_constant(0x0000_0001_0000_0001, PhysReg(0));
        assert_eq!(s.instructions.len(), 2);
    }

    // -- Arithmetic --------------------------------------------------------

    #[test]
    fn test_select_add_i64() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Add { result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I64 },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "add_test".into(), return_type: IrType::I64,
            params: vec![("a".into(), IrType::I64), ("b".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::ADD.as_u32()));
    }

    #[test]
    fn test_select_mul() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Mul { result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::I64 },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "mul".into(), return_type: IrType::I64,
            params: vec![("a".into(), IrType::I64), ("b".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::MUL.as_u32()));
    }

    #[test]
    fn test_select_sdiv() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Div { result: Value(2), lhs: Value(0), rhs: Value(1),
                ty: IrType::I64, is_signed: true },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "div".into(), return_type: IrType::I64,
            params: vec![("a".into(), IrType::I64), ("b".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::SDIV.as_u32()));
    }

    #[test]
    fn test_select_mod_sdiv_msub() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Mod { result: Value(2), lhs: Value(0), rhs: Value(1),
                ty: IrType::I64, is_signed: true },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "modtest".into(), return_type: IrType::I64,
            params: vec![("a".into(), IrType::I64), ("b".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::SDIV.as_u32()));
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::MSUB.as_u32()));
    }

    // -- Comparison / csel -------------------------------------------------

    #[test]
    fn test_select_icmp_eq() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::ICmp { result: Value(2), op: CompareOp::Equal,
                lhs: Value(0), rhs: Value(1), ty: IrType::I64 },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "cmp".into(), return_type: IrType::I64,
            params: vec![("a".into(), IrType::I64), ("b".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::SUBS.as_u32()));
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::CSINC.as_u32()));
    }

    #[test]
    fn test_select_csel() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::ICmp { result: Value(2), op: CompareOp::SignedLess,
                lhs: Value(0), rhs: Value(1), ty: IrType::I64 },
            Instruction::Select { result: Value(3), condition: Value(2),
                true_val: Value(0), false_val: Value(1), ty: IrType::I64 },
        ], Some(Terminator::Return { value: Some(Value(3)) }));
        let f = Function {
            name: "sel".into(), return_type: IrType::I64,
            params: vec![("a".into(), IrType::I64), ("b".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::CSEL.as_u32()));
    }

    // -- Memory ------------------------------------------------------------

    #[test]
    fn test_select_load_i64() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Load { result: Value(1), ty: IrType::I64, ptr: Value(0) },
        ], Some(Terminator::Return { value: Some(Value(1)) }));
        let f = Function {
            name: "ld".into(), return_type: IrType::I64,
            params: vec![("p".into(), IrType::Pointer(Box::new(IrType::I64)))],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::LDR.as_u32()));
    }

    #[test]
    fn test_select_store_and_load_i32() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Store { value: Value(0), ptr: Value(1), store_ty: None },
            Instruction::Load { result: Value(2), ty: IrType::I32, ptr: Value(1) },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "sl".into(), return_type: IrType::I32,
            params: vec![
                ("v".into(), IrType::I32),
                ("p".into(), IrType::Pointer(Box::new(IrType::I32))),
            ],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::STR.as_u32()
            || i.opcode == Aarch64Opcode::STR_D.as_u32()));
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::LDRW.as_u32()));
    }

    // -- Control flow ------------------------------------------------------

    #[test]
    fn test_select_branch() {
        let mut s = Aarch64InstructionSelector::new();
        let b0 = make_block(0, vec![], Some(Terminator::Branch { target: BlockId(1) }));
        let b1 = make_block(1, vec![], Some(Terminator::Return { value: None }));
        let r = s.select_function(&make_function("br", vec![b0, b1])).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::B.as_u32()));
    }

    #[test]
    fn test_select_cond_branch() {
        let mut s = Aarch64InstructionSelector::new();
        let b0 = make_block(0, vec![], Some(Terminator::CondBranch {
            condition: Value(0), true_block: BlockId(1), false_block: BlockId(2),
        }));
        let b1 = make_block(1, vec![], Some(Terminator::Return { value: None }));
        let b2 = make_block(2, vec![], Some(Terminator::Return { value: None }));
        let f = Function {
            name: "cb".into(), return_type: IrType::Void,
            params: vec![("c".into(), IrType::I1)],
            param_values: Vec::new(),
            blocks: vec![b0, b1, b2], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::CBNZ.as_u32()));
    }

    #[test]
    fn test_select_return() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![], Some(Terminator::Return { value: Some(Value(0)) }));
        let f = Function {
            name: "ret".into(), return_type: IrType::I64,
            params: vec![("v".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::RET.as_u32()));
    }

    // -- Type conversions --------------------------------------------------

    #[test]
    fn test_select_sext_i32_to_i64() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Cast { result: Value(1), op: CastOp::SExt,
                value: Value(0), from_ty: IrType::I32, to_ty: IrType::I64 },
        ], Some(Terminator::Return { value: Some(Value(1)) }));
        let f = Function {
            name: "sx".into(), return_type: IrType::I64,
            params: vec![("v".into(), IrType::I32)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::SXTW.as_u32()));
    }

    #[test]
    fn test_select_fpext() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Cast { result: Value(1), op: CastOp::FPExt,
                value: Value(0), from_ty: IrType::F32, to_ty: IrType::F64 },
        ], Some(Terminator::Return { value: Some(Value(1)) }));
        let f = Function {
            name: "fpe".into(), return_type: IrType::F64,
            params: vec![("v".into(), IrType::F32)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::FCVT_S_TO_D.as_u32()));
    }

    #[test]
    fn test_select_scvtf() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Cast { result: Value(1), op: CastOp::SIToFP,
                value: Value(0), from_ty: IrType::I64, to_ty: IrType::F64 },
        ], Some(Terminator::Return { value: Some(Value(1)) }));
        let f = Function {
            name: "sc".into(), return_type: IrType::F64,
            params: vec![("v".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::SCVTF_D.as_u32()));
    }

    // -- Relocations -------------------------------------------------------

    #[test]
    fn test_global_ref_relocs() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Const { result: Value(0),
                value: Constant::GlobalRef("myg".into()) },
        ], Some(Terminator::Return { value: Some(Value(0)) }));
        let r = s.select_function(&make_function("gr", vec![b])).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::ADRP.as_u32()));
        let rels = s.take_relocations();
        assert!(rels.len() >= 2);
        assert!(rels.iter().any(|r| r.reloc_type == RelocationType::Aarch64_ADR_PREL_PG_HI21));
        assert!(rels.iter().any(|r| r.reloc_type == RelocationType::Aarch64_ADD_ABS_LO12_NC));
    }

    // -- FP arithmetic -----------------------------------------------------

    #[test]
    fn test_select_fadd_d() {
        let mut s = Aarch64InstructionSelector::new();
        let b = make_block(0, vec![
            Instruction::Add { result: Value(2), lhs: Value(0), rhs: Value(1), ty: IrType::F64 },
        ], Some(Terminator::Return { value: Some(Value(2)) }));
        let f = Function {
            name: "fa".into(), return_type: IrType::F64,
            params: vec![("a".into(), IrType::F64), ("b".into(), IrType::F64)],
            param_values: Vec::new(),
            blocks: vec![b], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        assert!(r.iter().any(|i| i.opcode == Aarch64Opcode::FADD_D.as_u32()));
    }

    // -- Switch ------------------------------------------------------------

    #[test]
    fn test_select_switch() {
        let mut s = Aarch64InstructionSelector::new();
        let b0 = make_block(0, vec![], Some(Terminator::Switch {
            value: Value(0), default: BlockId(3),
            cases: vec![(1, BlockId(1)), (2, BlockId(2))],
        }));
        let b1 = make_block(1, vec![], Some(Terminator::Return { value: None }));
        let b2 = make_block(2, vec![], Some(Terminator::Return { value: None }));
        let b3 = make_block(3, vec![], Some(Terminator::Return { value: None }));
        let f = Function {
            name: "sw".into(), return_type: IrType::Void,
            params: vec![("v".into(), IrType::I64)],
            param_values: Vec::new(),
            blocks: vec![b0, b1, b2, b3], entry_block: BlockId(0), is_definition: true,
is_static: false,
is_weak: false,
        };
        let r = s.select_function(&f).unwrap();
        let n = r.iter().filter(|i| i.opcode == Aarch64Opcode::B_cond.as_u32()).count();
        assert_eq!(n, 2);
    }
}
